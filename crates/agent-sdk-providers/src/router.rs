use std::fmt::Write;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use crate::provider::LlmProvider;
use crate::streaming::StreamBox;
use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest, ChatResponse, Message, Role};

/// A capability/cost tier a request can be routed to.
///
/// Tiers are ordered cheapest-and-fastest (`Fast`) to most-capable-and-costly
/// (`Advanced`); [`TaskComplexity::recommended_tier`] maps a classified
/// complexity onto one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Cheapest, fastest tier for trivial single-step work.
    Fast,
    /// Mid tier for multi-step reasoning, summarization, standard tool use.
    Capable,
    /// Most capable tier for creative, multi-step, or domain-heavy work.
    Advanced,
}

/// The complexity a classifier assigns to an incoming request.
///
/// Produced by [`ModelRouter::classify`]. When the classifier itself errors or
/// is rate limited the router falls back to the conservative
/// [`TaskComplexity::Complex`] so a misclassification never silently downgrades
/// a hard request to a weak model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    /// Basic factual questions, lookups, single-step operations.
    Simple,
    /// Multi-step reasoning, summarization, standard tool usage.
    Moderate,
    /// Creative generation, planning, synthesis, deep domain knowledge.
    Complex,
}

impl TaskComplexity {
    #[must_use]
    pub const fn recommended_tier(self) -> ModelTier {
        match self {
            Self::Simple => ModelTier::Fast,
            Self::Moderate => ModelTier::Capable,
            Self::Complex => ModelTier::Advanced,
        }
    }
}

/// Routes each request to a model tier chosen by an LLM classifier.
///
/// A `ModelRouter` wraps a `classifier` provider plus three tier providers and,
/// for every request, makes one extra classifier call to decide whether the work
/// is [`Simple`](TaskComplexity::Simple), [`Moderate`](TaskComplexity::Moderate),
/// or [`Complex`](TaskComplexity::Complex), then dispatches to the `fast`,
/// `capable`, or `advanced` provider respectively. If the classifier call fails
/// (error or rate limit) the router conservatively treats the request as
/// `Complex` rather than risk under-serving it (see [`classify`](Self::classify)).
///
/// `ModelRouter` itself implements [`LlmProvider`], so it can be passed anywhere
/// a `&dyn LlmProvider` is expected (`run_structured`, `RefreshingProvider`,
/// etc.). [`chat`](LlmProvider::chat) classifies then routes; the streaming
/// [`chat_stream`](LlmProvider::chat_stream) classifies first, then streams the
/// chosen tier.
///
/// Note: the `fast` and `capable` tiers currently share one provider type `S`
/// (only `advanced` has its own type `A`), so mixing e.g. a Gemini fast tier with
/// an `OpenAI` capable tier requires both behind the same concrete type. Use
/// `Arc<dyn LlmProvider>` for all three tiers to mix providers freely.
pub struct ModelRouter<C, S, A> {
    classifier: C,
    fast: S,
    capable: S,
    advanced: A,
}

impl<C, S, A> ModelRouter<C, S, A>
where
    C: LlmProvider,
    S: LlmProvider,
    A: LlmProvider,
{
    pub const fn new(classifier: C, fast: S, capable: S, advanced: A) -> Self {
        Self {
            classifier,
            fast,
            capable,
            advanced,
        }
    }

    /// # Errors
    /// Returns an error if the LLM provider fails.
    pub async fn classify(&self, request: &ChatRequest) -> Result<TaskComplexity> {
        let classification_prompt = build_classification_prompt(request);

        let classification_request = ChatRequest {
            system: CLASSIFICATION_SYSTEM.to_owned(),
            messages: vec![Message::user(classification_prompt)],
            tools: None,
            max_tokens: 50,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
        };

        match self.classifier.chat(classification_request).await? {
            ChatOutcome::Success(response) => {
                let complexity = parse_complexity(&response);
                log::debug!(
                    "Model router classified request as {:?} using {}",
                    complexity,
                    self.classifier.model()
                );
                Ok(complexity)
            }
            ChatOutcome::RateLimited => {
                log::warn!("Classifier rate limited, defaulting to Complex");
                Ok(TaskComplexity::Complex)
            }
            ChatOutcome::InvalidRequest(e) => {
                log::error!("Classifier invalid request: {e}, defaulting to Complex");
                Ok(TaskComplexity::Complex)
            }
            ChatOutcome::ServerError(e) => {
                log::error!("Classifier server error: {e}, defaulting to Complex");
                Ok(TaskComplexity::Complex)
            }
            // `ChatOutcome` is `#[non_exhaustive]`; an unrecognized outcome
            // takes the same conservative fallback as the error variants.
            _ => {
                log::error!("Classifier returned unrecognized outcome, defaulting to Complex");
                Ok(TaskComplexity::Complex)
            }
        }
    }

    /// # Errors
    /// Returns an error if the LLM provider fails.
    pub async fn route(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let complexity = self.classify(&request).await?;
        let tier = complexity.recommended_tier();

        log::info!("Routing request to {tier:?} tier (complexity: {complexity:?})");

        match tier {
            ModelTier::Fast => self.fast.chat(request).await,
            ModelTier::Capable => self.capable.chat(request).await,
            ModelTier::Advanced => self.advanced.chat(request).await,
        }
    }

    /// # Errors
    /// Returns an error if the LLM provider fails.
    pub async fn route_with_tier(
        &self,
        request: ChatRequest,
        tier: ModelTier,
    ) -> Result<ChatOutcome> {
        match tier {
            ModelTier::Fast => self.fast.chat(request).await,
            ModelTier::Capable => self.capable.chat(request).await,
            ModelTier::Advanced => self.advanced.chat(request).await,
        }
    }

    #[must_use]
    pub const fn fast_provider(&self) -> &S {
        &self.fast
    }

    #[must_use]
    pub const fn capable_provider(&self) -> &S {
        &self.capable
    }

    #[must_use]
    pub const fn advanced_provider(&self) -> &A {
        &self.advanced
    }
}

#[async_trait]
impl<C, S, A> LlmProvider for ModelRouter<C, S, A>
where
    C: LlmProvider,
    S: LlmProvider,
    A: LlmProvider,
{
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.route(request).await
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let tier = match self.classify(&request).await {
                Ok(complexity) => complexity.recommended_tier(),
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            log::info!("Streaming request to {tier:?} tier");
            let mut stream = match tier {
                ModelTier::Fast => self.fast.chat_stream(request),
                ModelTier::Capable => self.capable.chat_stream(request),
                ModelTier::Advanced => self.advanced.chat_stream(request),
            };
            while let Some(item) = stream.next().await {
                yield item;
            }
        })
    }

    /// Reports the `capable` (mid) tier's model as the router's representative
    /// model identifier.
    fn model(&self) -> &str {
        self.capable.model()
    }

    /// Reports the `capable` (mid) tier's provider as the router's representative
    /// provider identifier.
    fn provider(&self) -> &'static str {
        self.capable.provider()
    }
}

const CLASSIFICATION_SYSTEM: &str = r"You are a task complexity classifier. Analyze the user's request and classify it as one of: SIMPLE, MODERATE, or COMPLEX.

SIMPLE tasks:
- Basic questions with factual answers
- Simple calculations
- Direct lookups or retrievals
- Yes/no questions
- Single-step operations

MODERATE tasks:
- Multi-step reasoning
- Summarization
- Basic analysis
- Comparisons
- Standard tool usage

COMPLEX tasks:
- Creative writing or content generation
- Multi-step planning
- Complex analysis or synthesis
- Nuanced decisions
- Tasks requiring deep domain knowledge
- Financial advice or calculations
- Multi-tool orchestration

Respond with ONLY one word: SIMPLE, MODERATE, or COMPLEX.";

fn build_classification_prompt(request: &ChatRequest) -> String {
    let mut prompt = String::new();

    prompt.push_str("Classify this task:\n\n");

    if !request.system.is_empty() {
        prompt.push_str("System context: ");
        let truncated = truncate_on_char_boundary(&request.system, 200);
        prompt.push_str(truncated);
        if truncated.len() < request.system.len() {
            prompt.push_str("...");
        }
        prompt.push_str("\n\n");
    }

    if let Some(last_user_message) = request.messages.iter().rev().find(|m| m.role == Role::User)
        && let Some(text) = last_user_message.content.first_text()
    {
        prompt.push_str("User request: ");
        let truncated = truncate_on_char_boundary(text, 500);
        prompt.push_str(truncated);
        if truncated.len() < text.len() {
            prompt.push_str("...");
        }
    }

    if let Some(tools) = &request.tools {
        let _ = write!(prompt, "\n\nAvailable tools: {}", tools.len());
    }

    prompt
}

/// Truncate `s` to at most `max_bytes`, backing off to the nearest UTF-8
/// character boundary so the byte slice never panics on a multi-byte character
/// (emoji, CJK, accented text) that straddles the limit.
fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn parse_complexity(response: &ChatResponse) -> TaskComplexity {
    let text = response.first_text().unwrap_or("").to_uppercase();

    if text.contains("SIMPLE") {
        TaskComplexity::Simple
    } else if text.contains("MODERATE") {
        TaskComplexity::Moderate
    } else {
        TaskComplexity::Complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complexity_to_tier() {
        assert_eq!(TaskComplexity::Simple.recommended_tier(), ModelTier::Fast);
        assert_eq!(
            TaskComplexity::Moderate.recommended_tier(),
            ModelTier::Capable
        );
        assert_eq!(
            TaskComplexity::Complex.recommended_tier(),
            ModelTier::Advanced
        );
    }

    #[test]
    fn truncate_on_char_boundary_never_splits_multibyte_char() {
        // "😀" is a 4-byte character. Truncating at byte 1, 2, or 3 would land
        // inside it and panic with naive `&s[..n]`; the helper must back off to a
        // valid boundary instead.
        let s = "😀😀😀";
        for max in 0..=s.len() {
            let truncated = truncate_on_char_boundary(s, max);
            // Must be a valid prefix of the original (never panics, always UTF-8).
            assert!(s.starts_with(truncated));
            assert!(truncated.len() <= max);
        }
        assert_eq!(truncate_on_char_boundary(s, 4), "😀");
        assert_eq!(truncate_on_char_boundary(s, 5), "😀");
        assert_eq!(truncate_on_char_boundary(s, 100), s);
    }

    #[test]
    fn build_classification_prompt_handles_multibyte_at_limit() {
        // A system prompt longer than 200 bytes whose 200th byte falls inside a
        // multi-byte char must not panic when building the classification prompt.
        let system = "é".repeat(150); // 300 bytes; byte 200 is mid-character
        let request = ChatRequest::new(system, vec![Message::user("日本語".repeat(300))]);
        // The bug manifested as a panic; reaching this assertion means no panic.
        let prompt = build_classification_prompt(&request);
        assert!(prompt.contains("System context:"));
        assert!(prompt.ends_with("..."));
    }
}
