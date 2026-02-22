//! Context compaction implementation.

use crate::llm::{ChatOutcome, ChatRequest, Content, ContentBlock, LlmProvider, Message, Role};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::fmt::Write;
use std::sync::Arc;

use super::config::CompactionConfig;
use super::estimator::TokenEstimator;

const SUMMARY_PREFIX: &str = "[Previous conversation summary]\n\n";
const COMPACTION_SYSTEM_PROMPT: &str =
    "You are a precise summarizer. Your task is to create concise but complete summaries of conversations, preserving all technical details needed to continue the work.";
const COMPACTION_SUMMARY_PROMPT: &str = r#"Summarize this conversation concisely, preserving:
- Key decisions and conclusions reached
- Important file paths, code changes, and technical details
- Current task context and what has been accomplished
- Any pending items, errors encountered, or next steps

Be specific about technical details (file names, function names, error messages) as these
are critical for continuing the work.

Conversation:
{messages_text}

Provide a concise summary (aim for 500-1000 words):"#;
const COMPACT_EMPTY_SUMMARY: &str =
    "No additional context was available to summarize; the previous messages were already compacted.";
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
pub struct LlmContextCompactor<P: LlmProvider> {
    provider: Arc<P>,
    config: CompactionConfig,
}

impl<P: LlmProvider> LlmContextCompactor<P> {
    /// Create a new LLM context compactor.
    #[must_use]
    pub const fn new(provider: Arc<P>, config: CompactionConfig) -> Self {
        Self { provider, config }
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

            let crosses_tool_pair =
                (prev.role == Role::Assistant
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

    /// Shift split point to satisfy both pair safety and retained-tail token cap.
    fn split_point_preserves_tool_pairs_with_cap(
        messages: &[Message],
        mut split_point: usize,
        max_tokens: usize,
    ) -> usize {
        loop {
            let candidate = Self::retain_tail_with_token_cap(messages, split_point, max_tokens);
            let adjusted = Self::split_point_preserves_tool_pairs(messages, candidate);

            if adjusted == split_point {
                return candidate;
            }

            split_point = adjusted;
        }
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
    fn build_summary_prompt(messages_text: &str) -> String {
        COMPACTION_SUMMARY_PROMPT.replace("{messages_text}", messages_text)
    }
}

#[async_trait]
impl<P: LlmProvider> ContextCompactor for LlmContextCompactor<P> {
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
        let prompt = Self::build_summary_prompt(&messages_text);

        let request = ChatRequest {
            system: COMPACTION_SYSTEM_PROMPT.to_string(),
            messages: vec![Message::user(prompt)],
            tools: None,
            max_tokens: 2000,
            thinking: None,
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

        // Add acknowledgment from assistant
        new_messages.push(Message::assistant(
            "I understand the context from the summary. Let me continue from where we left off.",
        ));

        // Add recent messages
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

        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(summary, "Fresh summary");
        assert!(recorded[0].contains("Continue with the next task using this context."));
        assert!(!recorded[0].contains("already compacted context"));

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

        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].contains("Current turn content from the latest exchange."));
        assert!(!recorded[0].contains("already compacted context"));
        assert_eq!(result.new_count, 4);

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_is_no_op_when_candidate_window_has_only_summaries() -> Result<()> {
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

        let recorded = requests.lock().unwrap();
        assert!(recorded.is_empty());
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
    async fn test_compact_history_preserves_tool_result_tool_use_pairs() -> Result<()> {
        let provider = Arc::new(MockProvider::new("Summary around tool pair."));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        // Build a history where split_point would land on tool-use tool-result crossing in the
        // opposite direction:
        // ... user tool_result | assistant tool_use ...
        let messages = vec![
            Message::user("Start a workflow"),
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "tool_odd".to_string(),
                    content: "prior result".to_string(),
                    is_error: None,
                }]),
            },
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: "tool_odd".to_string(),
                    name: "follow_up".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                }]),
            },
            Message::assistant("Follow up done."),
        ];

        let result = compactor.compact_history(messages).await?;

        // Split-point starts at 2 and is adjusted back to 1, keeping the tool result and tool use.
        assert_eq!(result.new_count, 5);

        // tool_result should remain with the kept tail.
        let kept_result = &result.messages[2];
        if let Content::Blocks(blocks) = &kept_result.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. })),
                "Expected kept user tool_result in retained tail"
            );
        } else {
            panic!("Expected tool_result blocks in retained tail");
        }

        // tool_use should remain with the kept tail.
        let kept_tool_use = &result.messages[3];
        if let Content::Blocks(blocks) = &kept_tool_use.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
                "Expected kept assistant tool_use in retained tail"
            );
        } else {
            panic!("Expected tool_use blocks in retained tail");
        }

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
        messages.extend((0..8).map(|index| Message::assistant(format!(
            "kept-{index}: {}",
            "x".repeat(12_000)
        ))));

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
        assert!(TokenEstimator::estimate_history(retained_tail) <= MAX_RETAINED_TAIL_MESSAGE_TOKENS);
        assert!(compactor.needs_compaction(&result.messages));

        Ok(())
    }
}
