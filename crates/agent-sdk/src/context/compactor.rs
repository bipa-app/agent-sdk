//! Context compaction implementation.

use crate::llm::{ChatOutcome, ChatRequest, Content, ContentBlock, LlmProvider, Message, Role};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::fmt::Write;
use std::sync::Arc;

use super::config::CompactionConfig;
use super::estimator::TokenEstimator;

const SUMMARY_PREFIX: &str = "[Previous conversation summary]\n\n";
const COMPACTION_SYSTEM_PROMPT: &str = "You are a precise summarizer. Your task is to create concise but complete summaries of conversations, preserving all technical details needed to continue the work.";
const COMPACTION_SUMMARY_PROMPT_PREFIX: &str = "Summarize this conversation concisely, preserving:\n- Key decisions and conclusions reached\n- Important file paths, code changes, and technical details\n- Current task context and what has been accomplished\n- Any pending items, errors encountered, or next steps\n\nBe specific about technical details (file names, function names, error messages) as these\nare critical for continuing the work.\n\nConversation:\n";
const COMPACTION_SUMMARY_PROMPT_SUFFIX: &str =
    "Provide a concise summary (aim for 500-1000 words):";
const COMPACT_EMPTY_SUMMARY: &str = "No additional context was available to summarize; the previous messages were already compacted.";
const SUMMARY_ACKNOWLEDGMENT: &str =
    "I understand the context from the summary. Let me continue from where we left off.";
const MAX_RETAINED_TAIL_MESSAGE_TOKENS: usize = 20_000;
const MAX_TOOL_RESULT_CHARS: usize = 500;

/// Trait for context compaction strategies.
///
/// Implement this trait to provide custom compaction logic.
#[async_trait]
pub trait ContextCompactor: Send + Sync {
    /// Compact a list of messages into a summary.
    ///
    /// # Errors
    /// Returns an error if summarization fails.
    async fn compact(&self, messages: &[Message]) -> Result<String>;

    /// Estimate tokens for a message list.
    fn estimate_tokens(&self, messages: &[Message]) -> usize;

    /// Check if compaction is needed.
    fn needs_compaction(&self, messages: &[Message]) -> bool;

    /// Perform full compaction, returning new message history.
    ///
    /// # Errors
    /// Returns an error if compaction fails.
    async fn compact_history(&self, messages: Vec<Message>) -> Result<CompactionResult>;
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The new compacted message history.
    pub messages: Vec<Message>,
    /// Number of messages before compaction.
    pub original_count: usize,
    /// Number of messages after compaction.
    pub new_count: usize,
    /// Estimated tokens before compaction.
    pub original_tokens: usize,
    /// Estimated tokens after compaction.
    pub new_tokens: usize,
}

/// LLM-based context compactor.
///
/// Uses the LLM itself to summarize older messages into a compact form.
///
/// `P` is `?Sized` so callers can hold an `Arc<dyn LlmProvider>` —
/// useful when the provider is resolved dynamically per-thread (e.g.
/// inside `agent-server`'s daemon worker, where the same compactor
/// type wraps whichever concrete provider the host's resolver picks).
/// Concrete-type users (`Arc<AnthropicProvider>`, etc.) still work
/// unchanged.
pub struct LlmContextCompactor<P: LlmProvider + ?Sized> {
    provider: Arc<P>,
    config: CompactionConfig,
    system_prompt: String,
    summary_prompt_prefix: String,
    summary_prompt_suffix: String,
}

impl<P: LlmProvider + ?Sized> LlmContextCompactor<P> {
    /// Create a new LLM context compactor.
    #[must_use]
    pub fn new(provider: Arc<P>, config: CompactionConfig) -> Self {
        Self {
            provider,
            config,
            system_prompt: COMPACTION_SYSTEM_PROMPT.to_string(),
            summary_prompt_prefix: COMPACTION_SUMMARY_PROMPT_PREFIX.to_string(),
            summary_prompt_suffix: COMPACTION_SUMMARY_PROMPT_SUFFIX.to_string(),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults(provider: Arc<P>) -> Self {
        Self::new(provider, CompactionConfig::default())
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &CompactionConfig {
        &self.config
    }

    /// Override the prompts used for LLM-based summarization.
    #[must_use]
    pub fn with_prompts(
        mut self,
        system_prompt: impl Into<String>,
        summary_prompt_prefix: impl Into<String>,
        summary_prompt_suffix: impl Into<String>,
    ) -> Self {
        self.system_prompt = system_prompt.into();
        self.summary_prompt_prefix = summary_prompt_prefix.into();
        self.summary_prompt_suffix = summary_prompt_suffix.into();
        self
    }

    /// Return true when a content object is a previously inserted compaction summary marker.
    fn is_summary_message(content: &Content) -> bool {
        match content {
            Content::Text(text) => text.starts_with(SUMMARY_PREFIX),
            Content::Blocks(blocks) => blocks.iter().any(|block| match block {
                ContentBlock::Text { text } => text.starts_with(SUMMARY_PREFIX),
                _ => false,
            }),
        }
    }

    /// Return true when a message contains a tool-use block.
    fn has_tool_use(content: &Content) -> bool {
        matches!(
            content,
            Content::Blocks(blocks)
                if blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        )
    }

    /// Return true when a message contains a tool-result block.
    fn has_tool_result(content: &Content) -> bool {
        matches!(
            content,
            Content::Blocks(blocks)
                if blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
        )
    }

    /// Shift split point backwards until tool-use/result pairs are not split.
    fn split_point_preserves_tool_pairs(messages: &[Message], mut split_point: usize) -> usize {
        while split_point > 0 && split_point < messages.len() {
            let prev = &messages[split_point - 1];
            let next = &messages[split_point];

            let crosses_tool_pair = (prev.role == Role::Assistant
                && Self::has_tool_use(&prev.content)
                && next.role == Role::User
                && Self::has_tool_result(&next.content))
                || (prev.role == Role::User
                    && Self::has_tool_result(&prev.content)
                    && next.role == Role::Assistant
                    && Self::has_tool_use(&next.content));

            if crosses_tool_pair {
                split_point -= 1;
                continue;
            }

            break;
        }

        split_point
    }

    /// Pick a split point that produces a self-consistent `to_keep`.
    ///
    /// `to_keep` is self-consistent (per Anthropic's API contract)
    /// when every `tool_result` block it contains references a
    /// `tool_use` block earlier in `to_keep`. The compactor inserts
    /// a synthetic `[summary, summary_ack]` prefix in front of
    /// `to_keep`, and that prefix has no `tool_use` blocks — so the
    /// only path to a valid wire payload is for `to_keep` itself to
    /// be self-contained.
    ///
    /// Three constraints, applied in order:
    ///
    /// 1. **Token cap (soft)** — push split forward to keep the
    ///    retained tail under `max_tokens` of estimated content. The
    ///    retained-tail cap is a soft hint; a tool chain that doesn't
    ///    fit gets retained anyway because chain safety is hard.
    /// 2. **Pair safety (hard)** — shift split backward to keep
    ///    `assistant_with_tool_use` and the immediately following
    ///    `user_with_tool_result` together. Catches the common case
    ///    where the boundary lands inside a single tool turn.
    /// 3. **Chain safety (hard)** — advance split forward past any
    ///    leading `user_with_tool_result` whose `tool_use_id` isn't
    ///    in the rest of `to_keep`. Catches the case pair-preservation
    ///    can't see: when the message immediately before the original
    ///    boundary is text-only (e.g. a `summary_ack` from a prior
    ///    compaction), pair-preservation has nothing to anchor on
    ///    and silently leaves the orphan in `to_keep[0]`. The wire
    ///    payload would then start `[summary, summary_ack,
    ///    user(orphan_tool_result), …]` — which Anthropic rejects
    ///    with `messages.2.content.0: unexpected tool_use_id`. Step
    ///    3 makes the split-point selection responsible for chain
    ///    integrity instead of post-hoc stripping the output.
    ///
    /// Step 2 and step 3 can pull in opposite directions (step 2
    /// shifts back, step 3 shifts forward), so the function applies
    /// step 3 last: pair-safety puts the candidate as far back as
    /// it needs to go, then chain-safety advances past any leading
    /// orphan that survived because the immediate prev was text-only.
    fn split_point_preserves_tool_pairs_with_cap(
        messages: &[Message],
        split_point: usize,
        max_tokens: usize,
    ) -> usize {
        let cap_limit = Self::retain_tail_with_token_cap(messages, split_point, max_tokens);
        let pair_safe = Self::split_point_preserves_tool_pairs(messages, cap_limit);
        Self::split_point_skips_leading_orphan(messages, pair_safe)
    }

    /// Advance `split_point` forward until `to_keep[0]` doesn't
    /// contain an orphan `tool_result` block — i.e. a `tool_result`
    /// whose `tool_use_id` isn't satisfied by some `tool_use` block
    /// in `to_keep`.
    ///
    /// Implements step 3 of `split_point_preserves_tool_pairs_with_cap`
    /// (chain safety). Pair-preservation alone can't catch the
    /// "synthetic `summary_ack` precedes an orphan" shape because it
    /// only inspects the immediate prev/next pair; this helper
    /// inspects whether `to_keep[0]`'s `tool_result` blocks point
    /// anywhere `to_keep` will host a matching `tool_use`. When they
    /// don't, the `tool_result` belongs in `to_summarize` (where it
    /// gets text-ified into the summary prose), not in `to_keep`.
    ///
    /// Walks at most `messages.len()` steps because each iteration
    /// advances `split_point` by at least 1.
    fn split_point_skips_leading_orphan(messages: &[Message], mut split_point: usize) -> usize {
        while split_point < messages.len() {
            if Self::leading_message_has_orphan_tool_result(&messages[split_point..]) {
                split_point = split_point.saturating_add(1);
                continue;
            }
            break;
        }
        split_point
    }

    /// True when `to_keep[0]` is a `user` message whose `tool_result`
    /// blocks reference at least one `tool_use_id` not present in
    /// `to_keep`. The check is scoped to the first message because
    /// well-formed Anthropic conversations always have `tool_use`
    /// immediately before `tool_result` — an orphan deeper than
    /// `to_keep[0]` would require the input itself to be malformed
    /// upstream of compaction, which is out of scope here.
    fn leading_message_has_orphan_tool_result(to_keep: &[Message]) -> bool {
        let Some(first) = to_keep.first() else {
            return false;
        };
        let Content::Blocks(blocks) = &first.content else {
            return false;
        };

        // Pull the tool_result ids that appear in the first message.
        // If there are none, the first message can't contribute an
        // orphan and we're done early without scanning the tail.
        let mut needed: Vec<&str> = Vec::new();
        for block in blocks {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                needed.push(tool_use_id.as_str());
            }
        }
        if needed.is_empty() {
            return false;
        }

        // Build the set of tool_use ids `to_keep` will host.
        let known_ids: std::collections::HashSet<&str> = to_keep
            .iter()
            .flat_map(|message| match &message.content {
                Content::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                Content::Text(_) => Vec::new(),
            })
            .collect();

        needed.iter().any(|id| !known_ids.contains(id))
    }

    /// Keep most recent messages that fit within the retained-message token budget.
    fn retain_tail_with_token_cap(messages: &[Message], start: usize, max_tokens: usize) -> usize {
        if start >= messages.len() {
            return messages.len();
        }

        if max_tokens == 0 {
            return messages.len();
        }

        let mut used = 0usize;
        let mut retained_start = messages.len();

        for idx in (start..messages.len()).rev() {
            let message_tokens = TokenEstimator::estimate_message(&messages[idx]);
            if used + message_tokens > max_tokens {
                break;
            }

            retained_start = idx;
            used += message_tokens;
        }

        retained_start
    }

    /// Format messages for summarization.
    fn format_messages_for_summary(messages: &[Message]) -> String {
        let mut output = String::new();

        for message in messages {
            let role = match message.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
            };

            let _ = write!(output, "{role}: ");

            match &message.content {
                Content::Text(text) => {
                    let _ = writeln!(output, "{text}");
                }
                Content::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                let _ = writeln!(output, "{text}");
                            }
                            ContentBlock::Thinking { thinking, .. } => {
                                // Include thinking in summaries for context
                                let _ = writeln!(output, "[Thinking: {thinking}]");
                            }
                            ContentBlock::RedactedThinking { .. } => {
                                let _ = writeln!(output, "[Redacted thinking]");
                            }
                            ContentBlock::ToolUse { name, input, .. } => {
                                let _ = writeln!(
                                    output,
                                    "[Called tool: {name} with input: {}]",
                                    serde_json::to_string(input).unwrap_or_default()
                                );
                            }
                            ContentBlock::ToolResult {
                                content, is_error, ..
                            } => {
                                let status = if is_error.unwrap_or(false) {
                                    "error"
                                } else {
                                    "success"
                                };
                                // Truncate long tool results (Unicode-safe; avoid slicing mid-codepoint)
                                let truncated = if content.chars().count() > MAX_TOOL_RESULT_CHARS {
                                    let prefix: String =
                                        content.chars().take(MAX_TOOL_RESULT_CHARS).collect();
                                    format!("{prefix}... (truncated)")
                                } else {
                                    content.clone()
                                };
                                let _ = writeln!(output, "[Tool result ({status}): {truncated}]");
                            }
                            ContentBlock::Image { source } => {
                                let _ = writeln!(output, "[Image: {}]", source.media_type);
                            }
                            ContentBlock::Document { source } => {
                                let _ = writeln!(output, "[Document: {}]", source.media_type);
                            }
                        }
                    }
                }
            }
            output.push('\n');
        }

        output
    }

    /// Build the summarization prompt.
    fn build_summary_prompt(&self, messages_text: &str) -> String {
        format!(
            "{}{}{}",
            self.summary_prompt_prefix, messages_text, self.summary_prompt_suffix
        )
    }
}

#[async_trait]
impl<P: LlmProvider + ?Sized> ContextCompactor for LlmContextCompactor<P> {
    async fn compact(&self, messages: &[Message]) -> Result<String> {
        let messages_to_summarize: Vec<_> = messages
            .iter()
            .filter(|message| !Self::is_summary_message(&message.content))
            .cloned()
            .collect();

        if messages_to_summarize.is_empty() {
            return Ok(COMPACT_EMPTY_SUMMARY.to_string());
        }

        let messages_text = Self::format_messages_for_summary(&messages_to_summarize);
        let prompt = self.build_summary_prompt(&messages_text);

        let request = ChatRequest {
            system: self.system_prompt.clone(),
            messages: vec![Message::user(prompt)],
            tools: None,
            max_tokens: 2000,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
        };

        let outcome = self
            .provider
            .chat(request)
            .await
            .context("Failed to call LLM for summarization")?;

        match outcome {
            ChatOutcome::Success(response) => response
                .first_text()
                .map(String::from)
                .context("No text in summarization response"),
            ChatOutcome::RateLimited => {
                bail!("Rate limited during summarization")
            }
            ChatOutcome::InvalidRequest(msg) => {
                bail!("Invalid request during summarization: {msg}")
            }
            ChatOutcome::ServerError(msg) => {
                bail!("Server error during summarization: {msg}")
            }
        }
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        TokenEstimator::estimate_history(messages)
    }

    fn needs_compaction(&self, messages: &[Message]) -> bool {
        if !self.config.auto_compact {
            return false;
        }

        if messages.len() < self.config.min_messages_for_compaction {
            return false;
        }

        let estimated_tokens = self.estimate_tokens(messages);
        estimated_tokens > self.config.threshold_tokens
    }

    async fn compact_history(&self, messages: Vec<Message>) -> Result<CompactionResult> {
        let original_count = messages.len();
        let original_tokens = self.estimate_tokens(&messages);

        // Ensure we have enough messages to compact
        if messages.len() <= self.config.retain_recent {
            return Ok(CompactionResult {
                messages,
                original_count,
                new_count: original_count,
                original_tokens,
                new_tokens: original_tokens,
            });
        }

        // Split messages: old messages to summarize, recent messages to keep
        let mut split_point = messages.len().saturating_sub(self.config.retain_recent);
        split_point = Self::split_point_preserves_tool_pairs_with_cap(
            &messages,
            split_point,
            MAX_RETAINED_TAIL_MESSAGE_TOKENS,
        );

        let (to_summarize, to_keep) = messages.split_at(split_point);

        // Summarize old messages
        let summary = self.compact(to_summarize).await?;

        // Build new message history
        let mut new_messages = Vec::with_capacity(2 + to_keep.len());

        // Add summary as a user message
        new_messages.push(Message::user(format!("{SUMMARY_PREFIX}{summary}")));

        // Add acknowledgment from assistant only when some recent tail remains.
        // If compaction drops the entire retained tail due to the token cap, ending
        // the request with this synthetic assistant message would act like assistant
        // prefill and Anthropic rejects that shape.
        if !to_keep.is_empty() {
            new_messages.push(Message::assistant(SUMMARY_ACKNOWLEDGMENT));
        }

        // Add recent messages. `to_keep` is guaranteed self-consistent
        // by `split_point_preserves_tool_pairs_with_cap` (steps 2 and
        // 3): any orphan `tool_result` was either folded into the
        // summary (split shifted forward) or paired with its
        // `tool_use` inside `to_keep` (split shifted backward). No
        // post-hoc rewriting of the assembled output is required.
        new_messages.extend(to_keep.iter().cloned());

        let new_count = new_messages.len();
        let new_tokens = self.estimate_tokens(&new_messages);

        Ok(CompactionResult {
            messages: new_messages,
            original_count,
            new_count,
            original_tokens,
            new_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatResponse, StopReason, Usage};
    use std::sync::Mutex;

    struct MockProvider {
        summary_response: String,
        requests: Option<Arc<Mutex<Vec<String>>>>,
    }

    impl MockProvider {
        fn new(summary: &str) -> Self {
            Self {
                summary_response: summary.to_string(),
                requests: None,
            }
        }

        fn new_with_request_log(summary: &str, requests: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                summary_response: summary.to_string(),
                requests: Some(requests),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
            if let Some(requests) = &self.requests {
                let mut entries = requests.lock().unwrap();
                let user_prompt = request
                    .messages
                    .iter()
                    .find_map(|message| match &message.content {
                        Content::Text(text) => Some(text.clone()),
                        Content::Blocks(blocks) => {
                            let text = blocks
                                .iter()
                                .filter_map(|block| {
                                    if let ContentBlock::Text { text } = block {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            if text.is_empty() { None } else { Some(text) }
                        }
                    })
                    .unwrap_or_default();
                entries.push(user_prompt);
            }
            Ok(ChatOutcome::Success(ChatResponse {
                id: "test".to_string(),
                content: vec![ContentBlock::Text {
                    text: self.summary_response.clone(),
                }],
                model: "mock".to_string(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn test_needs_compaction_below_threshold() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(10_000)
            .with_min_messages(5);
        let compactor = LlmContextCompactor::new(provider, config);

        // Only 3 messages, below min_messages
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi"),
            Message::user("How are you?"),
        ];

        assert!(!compactor.needs_compaction(&messages));
    }

    #[test]
    fn test_needs_compaction_above_threshold() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(50) // Very low threshold
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Messages that exceed threshold
        let messages = vec![
            Message::user("Hello, this is a longer message to test compaction"),
            Message::assistant(
                "Hi there! This is also a longer response to help trigger compaction",
            ),
            Message::user("Great, let's continue with even more text here"),
            Message::assistant("Absolutely, adding more content to ensure we exceed the threshold"),
        ];

        assert!(compactor.needs_compaction(&messages));
    }

    #[test]
    fn test_needs_compaction_auto_disabled() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(10) // Very low
            .with_min_messages(1)
            .with_auto_compact(false);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user("Hello, this is a longer message"),
            Message::assistant("Response here"),
        ];

        assert!(!compactor.needs_compaction(&messages));
    }

    #[tokio::test]
    async fn test_compact_history() -> Result<()> {
        let provider = Arc::new(MockProvider::new(
            "User asked about Rust programming. Assistant explained ownership, borrowing, and lifetimes.",
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Use longer messages to ensure compaction actually reduces tokens
        let messages = vec![
            Message::user(
                "What is Rust? I've heard it's a systems programming language but I don't know much about it. Can you explain the key features and why people are excited about it?",
            ),
            Message::assistant(
                "Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without garbage collection through its ownership system. The key features include zero-cost abstractions, guaranteed memory safety, threads without data races, and minimal runtime.",
            ),
            Message::user(
                "Tell me about ownership in detail. How does it work and what are the rules? I want to understand this core concept thoroughly.",
            ),
            Message::assistant(
                "Ownership is Rust's central feature with three rules: each value has one owner, only one owner at a time, and the value is dropped when owner goes out of scope. This system prevents memory leaks, double frees, and dangling pointers at compile time.",
            ),
            Message::user("What about borrowing?"), // Keep
            Message::assistant("Borrowing allows references to data without taking ownership."), // Keep
        ];

        let result = compactor.compact_history(messages).await?;

        // Should have: summary message + ack + 2 recent messages = 4
        assert_eq!(result.new_count, 4);
        assert_eq!(result.original_count, 6);

        // With longer original messages, compaction should reduce tokens
        assert!(
            result.new_tokens < result.original_tokens,
            "Expected fewer tokens after compaction: new={} < original={}",
            result.new_tokens,
            result.original_tokens
        );

        // First message should be the summary
        if let Content::Text(text) = &result.messages[0].content {
            assert!(text.contains("Previous conversation summary"));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_too_few_messages() -> Result<()> {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default().with_retain_recent(5);
        let compactor = LlmContextCompactor::new(provider, config);

        // Only 3 messages, less than retain_recent
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi"),
            Message::user("Bye"),
        ];

        let result = compactor.compact_history(messages.clone()).await?;

        // Should return original messages unchanged
        assert_eq!(result.new_count, 3);
        assert_eq!(result.messages.len(), 3);

        Ok(())
    }

    #[test]
    fn test_format_messages_for_summary() {
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there!")];

        let formatted = LlmContextCompactor::<MockProvider>::format_messages_for_summary(&messages);

        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_format_messages_for_summary_truncates_tool_results_unicode_safely() {
        let long_unicode = "é".repeat(600);

        let messages = vec![Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: long_unicode,
                is_error: Some(false),
            }]),
        }];

        let formatted = LlmContextCompactor::<MockProvider>::format_messages_for_summary(&messages);

        assert!(formatted.contains("... (truncated)"));
    }

    #[tokio::test]
    async fn test_compact_filters_summary_messages() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "Fresh summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}already compacted context")),
            Message::assistant("Continue with the next task using this context."),
        ];

        let summary = compactor.compact(&messages).await?;

        {
            let recorded = requests.lock().unwrap();
            assert_eq!(recorded.len(), 1);
            assert_eq!(summary, "Fresh summary");
            assert!(recorded[0].contains("Continue with the next task using this context."));
            assert!(!recorded[0].contains("already compacted context"));
            drop(recorded);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_ignores_prior_summary_in_candidate_payload() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "Fresh history summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}already compacted context")),
            Message::assistant("Current turn content from the latest exchange."),
            Message::assistant("Recent message that should stay."),
            Message::user("Newest note that should stay."),
        ];

        let result = compactor.compact_history(messages).await?;

        {
            let recorded = requests.lock().unwrap();
            assert_eq!(recorded.len(), 1);
            assert!(recorded[0].contains("Current turn content from the latest exchange."));
            assert!(!recorded[0].contains("already compacted context"));
            drop(recorded);
        }
        assert_eq!(result.new_count, 4);

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_is_no_op_when_candidate_window_has_only_summaries() -> Result<()>
    {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "This summary should not be used",
            requests.clone(),
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}first prior compacted section")),
            Message::assistant(format!("{SUMMARY_PREFIX}second prior compacted section")),
            Message::user(format!("{SUMMARY_PREFIX}third prior compacted section")),
            Message::assistant("final short note"),
        ];

        let result = compactor.compact_history(messages).await?;

        {
            let recorded = requests.lock().unwrap();
            assert!(recorded.is_empty());
            drop(recorded);
        }
        assert_eq!(result.new_count, 4);
        assert_eq!(result.messages.len(), 4);

        if let Content::Text(text) = &result.messages[0].content {
            assert!(text.contains(COMPACT_EMPTY_SUMMARY));
        } else {
            panic!("Expected summary text in first message");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_preserves_tool_use_tool_result_pairs() -> Result<()> {
        let provider = Arc::new(MockProvider::new("Summary of earlier conversation."));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Build a history where the split_point (len - retain_recent = 5 - 2 = 3)
        // would land exactly on the user tool_result message at index 3,
        // which would orphan it from its assistant tool_use at index 2.
        let messages = vec![
            // index 0: user
            Message::user("What files are in the project?"),
            // index 1: assistant text
            Message::assistant("Let me check that for you."),
            // index 2: assistant with tool_use
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "list_files".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                }]),
            },
            // index 3: user with tool_result (naive split would land here)
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "tool_1".to_string(),
                    content: "file1.rs\nfile2.rs".to_string(),
                    is_error: None,
                }]),
            },
            // index 4: assistant final response
            Message::assistant("The project contains file1.rs and file2.rs."),
        ];

        let result = compactor.compact_history(messages).await?;

        // The split_point should have been adjusted back from 3 to 2,
        // so to_keep includes: [assistant tool_use, user tool_result, assistant response]
        // Plus summary + ack = 5 total
        assert_eq!(result.new_count, 5);

        // Verify the kept messages include the tool_use/tool_result pair
        // After summary + ack, the third message should be the assistant with tool_use
        let kept_assistant = &result.messages[2];
        if let Content::Blocks(blocks) = &kept_assistant.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
                "Expected assistant tool_use in kept messages"
            );
        } else {
            panic!("Expected Blocks content for assistant tool_use message");
        }

        // The fourth message should be the user tool_result
        let kept_user = &result.messages[3];
        if let Content::Blocks(blocks) = &kept_user.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. })),
                "Expected user tool_result in kept messages"
            );
        } else {
            panic!("Expected Blocks content for user tool_result message");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_split_skips_leading_orphan_after_summary_ack() -> Result<()> {
        // The user-visible bug at M7.5: a previously
        // compacted history was re-compacted in a later turn. The
        // first compaction left
        // `[summary, summary_ack, user(tool_result toolu_X),
        //  assistant(toolu_X reply), ...]`. On the second pass the
        // default `split_point` (len - retain_recent = 5 - 3 = 2)
        // would have made `to_keep[0] == user(tool_result toolu_X)`,
        // and the synthetic `[summary, summary_ack, …]` prefix the
        // compactor inserts in front of `to_keep` has no `tool_use`
        // blocks — so the next request to Anthropic blew up with
        // `messages.2.content.0: unexpected tool_use_id`.
        //
        // Pair-preservation alone can't fix this: it only inspects
        // the immediate prev/next pair (here `summary_ack` vs
        // `user(tool_result)`) and `summary_ack` is text-only, so the
        // pair check sees no `tool_use` to anchor on and lets the
        // orphan through. The chain-safety pass added in
        // `split_point_preserves_tool_pairs_with_cap` step 3 walks
        // the candidate forward past any leading orphan, so the
        // `tool_result` lands in `to_summarize` and gets folded into
        // the summary's prose where it's harmless.
        //
        // The assertion is structural, not block-counting: every
        // surviving `tool_result` must reference a `tool_use` that
        // appears earlier in the new message list. No
        // post-compaction stripping is involved — the split point
        // alone is responsible for chain integrity.
        let provider = Arc::new(MockProvider::new("Re-summary."));
        let config = CompactionConfig::default()
            .with_retain_recent(3)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}Old summary about toolu_X.")),
            Message::assistant(SUMMARY_ACKNOWLEDGMENT),
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_X".to_string(),
                    content: "result for X".to_string(),
                    is_error: None,
                }]),
            },
            Message::assistant("Result interpreted."),
            Message::user("Now what?"),
        ];

        let result = compactor.compact_history(messages).await?;

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in &result.messages {
            if let Content::Blocks(blocks) = &msg.content {
                for block in blocks {
                    match block {
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            assert!(
                                seen_ids.contains(tool_use_id),
                                "orphan tool_use_id {tool_use_id} survived split selection",
                            );
                        }
                        ContentBlock::ToolUse { id, .. } => {
                            seen_ids.insert(id.clone());
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_keeps_tool_pair_when_immediate_prev_is_text_only() -> Result<()> {
        // Tighter regression for the chain-safety boundary: even
        // when the message *before* the candidate split point is
        // text-only (so pair-preservation has nothing to anchor on),
        // chain-safety must shift the split forward past a leading
        // `user(tool_result)` whose `tool_use` would otherwise be
        // folded into the summary.
        let provider = Arc::new(MockProvider::new("Boundary summary."));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        // Layout (5 messages, retain_recent=2 → initial split=3):
        //   0: user("first turn") — to_summarize
        //   1: assistant("text only") — to_summarize, immediate prev
        //   2: user(tool_result toolu_Y) — orphan in default to_keep
        //   3: assistant("then a reply")
        //   4: user("ok thanks")
        //
        // The corresponding `tool_use` for toolu_Y was lost long
        // ago — there's no `tool_use` anywhere in `messages`. With
        // pair-preservation alone, `to_keep` would start at index 3
        // (or 2 unshifted), leaving the orphan at the head and
        // tripping Anthropic.
        let messages = vec![
            Message::user("first turn"),
            Message::assistant("text only"),
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_Y".to_string(),
                    content: "ancient result".to_string(),
                    is_error: None,
                }]),
            },
            Message::assistant("then a reply"),
            Message::user("ok thanks"),
        ];

        let result = compactor.compact_history(messages).await?;

        // No tool_result block survives anywhere — the only one in
        // input was orphaned and the split-shift folded it into the
        // summary.
        let has_tool_result = result.messages.iter().any(|m| {
            matches!(
                &m.content,
                Content::Blocks(blocks)
                    if blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            )
        });
        assert!(
            !has_tool_result,
            "orphan tool_result should have been pushed into to_summarize, not retained",
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_retained_tail_is_token_capped() -> Result<()> {
        let provider = Arc::new(MockProvider::new(
            "Project summary with a long context and technical context.",
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(8)
            .with_min_messages(1)
            .with_threshold_tokens(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let mut messages = Vec::new();

        // Older messages that will be summarized away.
        messages.extend((0..6).map(|index| Message::user(format!("pre-compaction noise {index}"))));

        // Newer long messages: intentionally large to force retained-tail truncation.
        messages.extend(
            (0..8).map(|index| Message::assistant(format!("kept-{index}: {}", "x".repeat(12_000)))),
        );

        let result = compactor.compact_history(messages).await?;

        // The retained tail should be token capped and therefore shorter than retain_recent.
        let retained_tail = &result.messages[2..];
        assert!(retained_tail.len() < 8);

        let mut latest_index = -1i32;
        let mut all_retained = true;
        for message in retained_tail {
            if let Content::Text(text) = &message.content {
                if let Some(number) = text.split(':').next().and_then(|prefix| {
                    prefix
                        .strip_prefix("kept-")
                        .and_then(|rest| rest.parse::<i32>().ok())
                }) {
                    if number >= 0 {
                        latest_index = latest_index.max(number);
                    }
                } else {
                    all_retained = false;
                }
            } else {
                all_retained = false;
            }
        }

        assert!(all_retained);
        assert_eq!(latest_index, 7);
        assert!(
            TokenEstimator::estimate_history(retained_tail) <= MAX_RETAINED_TAIL_MESSAGE_TOKENS
        );
        assert!(compactor.needs_compaction(&result.messages));

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_skips_summary_ack_when_retained_tail_is_empty() -> Result<()> {
        let provider = Arc::new(MockProvider::new("Summary for oversized user turn."));
        let config = CompactionConfig::default()
            .with_retain_recent(1)
            .with_min_messages(1)
            .with_threshold_tokens(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::assistant("Earlier assistant context."),
            Message::user(format!("oversized-user-turn: {}", "x".repeat(200_000))),
        ];

        let result = compactor.compact_history(messages).await?;

        assert_eq!(result.new_count, 1);
        assert_eq!(result.messages.len(), 1);

        let only_message = &result.messages[0];
        assert_eq!(only_message.role, Role::User);

        if let Content::Text(text) = &only_message.content {
            assert!(text.contains("Previous conversation summary"));
            assert!(!text.contains(SUMMARY_ACKNOWLEDGMENT));
        } else {
            panic!("Expected summary text when retained tail is empty");
        }

        Ok(())
    }
}
