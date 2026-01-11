use std::fmt::Write;

use anyhow::Result;

use crate::llm::{ChatOutcome, ChatRequest, ChatResponse, LlmProvider, Message, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Fast,
    Capable,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    Simple,
    Moderate,
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
        prompt.push_str(&request.system[..request.system.len().min(200)]);
        if request.system.len() > 200 {
            prompt.push_str("...");
        }
        prompt.push_str("\n\n");
    }

    if let Some(last_user_message) = request.messages.iter().rev().find(|m| m.role == Role::User)
        && let Some(text) = last_user_message.content.first_text()
    {
        prompt.push_str("User request: ");
        prompt.push_str(&text[..text.len().min(500)]);
        if text.len() > 500 {
            prompt.push_str("...");
        }
    }

    if let Some(tools) = &request.tools {
        let _ = write!(prompt, "\n\nAvailable tools: {}", tools.len());
    }

    prompt
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
}
