//! Context compaction implementation.

use crate::llm::{ChatOutcome, ChatRequest, Content, ContentBlock, LlmProvider, Message, Role};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::fmt::Write;
use std::sync::Arc;

use super::config::CompactionConfig;
use super::estimator::TokenEstimator;

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
                                let truncated = if content.chars().count() > 500 {
                                    let prefix: String = content.chars().take(500).collect();
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
        format!(
            r"Summarize this conversation concisely, preserving:
- Key decisions and conclusions reached
- Important file paths, code changes, and technical details
- Current task context and what has been accomplished
- Any pending items, errors encountered, or next steps

Be specific about technical details (file names, function names, error messages) as these are critical for continuing the work.

Conversation:
{messages_text}

Provide a concise summary (aim for 500-1000 words):"
        )
    }
}

#[async_trait]
impl<P: LlmProvider> ContextCompactor for LlmContextCompactor<P> {
    async fn compact(&self, messages: &[Message]) -> Result<String> {
        let messages_text = Self::format_messages_for_summary(messages);
        let prompt = Self::build_summary_prompt(&messages_text);

        let request = ChatRequest {
            system: "You are a precise summarizer. Your task is to create concise but complete summaries of conversations, preserving all technical details that would be needed to continue the work.".to_string(),
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

        // Adjust split_point backwards if it lands between an assistant message
        // with tool_use and its following user message with tool_result. Breaking
        // these pairs corrupts the conversation history for the Anthropic API.
        while split_point > 0 {
            if let Content::Blocks(blocks) = &messages[split_point].content {
                let has_tool_result = blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
                if has_tool_result && messages[split_point].role == Role::User {
                    split_point -= 1;
                    continue;
                }
            }
            break;
        }

        let (to_summarize, to_keep) = messages.split_at(split_point);

        // Summarize old messages
        let summary = self.compact(to_summarize).await?;

        // Build new message history
        let mut new_messages = Vec::with_capacity(2 + to_keep.len());

        // Add summary as a user message
        new_messages.push(Message::user(format!(
            "[Previous conversation summary]\n\n{summary}"
        )));

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

    struct MockProvider {
        summary_response: String,
    }

    impl MockProvider {
        fn new(summary: &str) -> Self {
            Self {
                summary_response: summary.to_string(),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
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
}
