//! Token estimation for context size calculation.

use crate::llm::{Content, ContentBlock, Message};

/// Estimates token count for messages.
///
/// Uses a simple heuristic of ~4 characters per token, which provides
/// a reasonable approximation for most English text and code.
///
/// For more accurate counting, consider using a tokenizer library
/// specific to your model (e.g., tiktoken for `OpenAI` models).
pub struct TokenEstimator;

impl TokenEstimator {
    /// Characters per token estimate.
    /// This is a conservative estimate; actual ratio varies by content.
    const CHARS_PER_TOKEN: usize = 4;

    /// Overhead tokens per message (role, formatting).
    const MESSAGE_OVERHEAD: usize = 4;

    /// Overhead for tool use blocks (id, name, formatting).
    const TOOL_USE_OVERHEAD: usize = 20;

    /// Overhead for tool result blocks (id, formatting).
    const TOOL_RESULT_OVERHEAD: usize = 10;

    /// Minimum token estimate for redacted thinking blocks.
    ///
    /// Even small redacted thinking blocks carry significant API token cost
    /// because they contain encrypted reasoning that the model must process.
    const REDACTED_THINKING_MIN_TOKENS: usize = 512;

    /// Estimate tokens for a text string.
    #[must_use]
    pub const fn estimate_text(text: &str) -> usize {
        // Simple estimation: ~4 chars per token
        text.len().div_ceil(Self::CHARS_PER_TOKEN)
    }

    /// Estimate tokens for a single message.
    #[must_use]
    pub fn estimate_message(message: &Message) -> usize {
        let content_tokens = match &message.content {
            Content::Text(text) => Self::estimate_text(text),
            Content::Blocks(blocks) => blocks.iter().map(Self::estimate_block).sum(),
        };

        content_tokens + Self::MESSAGE_OVERHEAD
    }

    /// Estimate tokens for a content block.
    #[must_use]
    pub fn estimate_block(block: &ContentBlock) -> usize {
        match block {
            ContentBlock::Text { text } => Self::estimate_text(text),
            ContentBlock::Thinking { thinking, .. } => Self::estimate_text(thinking),
            ContentBlock::RedactedThinking { data } => {
                // The data field is a base64-encoded encrypted blob whose size
                // correlates with the original thinking content.  Base64 encodes
                // 3 bytes into 4 chars, so `data.len() * 3 / 4` approximates
                // the raw byte count.  Using the same chars-per-token heuristic
                // on the raw bytes gives a reasonable lower bound.
                //
                // A floor of REDACTED_THINKING_MIN_TOKENS prevents tiny blocks
                // from being under-counted — the API charges substantial token
                // overhead for every redacted thinking block regardless of size.
                let raw_bytes = data.len() * 3 / 4;
                let estimated = raw_bytes.div_ceil(Self::CHARS_PER_TOKEN);
                estimated.max(Self::REDACTED_THINKING_MIN_TOKENS)
            }
            ContentBlock::ToolUse { name, input, .. } => {
                let input_str = serde_json::to_string(input).unwrap_or_default();
                Self::estimate_text(name)
                    + Self::estimate_text(&input_str)
                    + Self::TOOL_USE_OVERHEAD
            }
            ContentBlock::ToolResult { content, .. } => {
                Self::estimate_text(content) + Self::TOOL_RESULT_OVERHEAD
            }
            ContentBlock::Image { source } | ContentBlock::Document { source } => {
                // Rough estimate: base64 data is ~4/3 of original, 1 token per 4 chars
                source.data.len() / 4 + Self::MESSAGE_OVERHEAD
            }
        }
    }

    /// Estimate total tokens for a message history.
    #[must_use]
    pub fn estimate_history(messages: &[Message]) -> usize {
        messages.iter().map(Self::estimate_message).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use serde_json::json;

    #[test]
    fn test_estimate_text() {
        // Empty text
        assert_eq!(TokenEstimator::estimate_text(""), 0);

        // Short text (less than 4 chars)
        assert_eq!(TokenEstimator::estimate_text("hi"), 1);

        // Exactly 4 chars
        assert_eq!(TokenEstimator::estimate_text("test"), 1);

        // 5 chars should be 2 tokens
        assert_eq!(TokenEstimator::estimate_text("hello"), 2);

        // Longer text
        assert_eq!(TokenEstimator::estimate_text("hello world!"), 3); // 12 chars / 4 = 3
    }

    #[test]
    fn test_estimate_text_message() {
        let message = Message {
            role: Role::User,
            content: Content::Text("Hello, how are you?".to_string()), // 19 chars = 5 tokens
        };

        let estimate = TokenEstimator::estimate_message(&message);
        // 5 content tokens + 4 overhead = 9
        assert_eq!(estimate, 9);
    }

    #[test]
    fn test_estimate_blocks_message() {
        let message = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![
                ContentBlock::Text {
                    text: "Let me help.".to_string(), // 12 chars = 3 tokens
                },
                ContentBlock::ToolUse {
                    id: "tool_123".to_string(),
                    name: "read".to_string(),            // 4 chars = 1 token
                    input: json!({"path": "/test.txt"}), // ~20 chars = 5 tokens
                    thought_signature: None,
                },
            ]),
        };

        let estimate = TokenEstimator::estimate_message(&message);
        // Text: 3 tokens
        // ToolUse: 1 (name) + 5 (input) + 20 (overhead) = 26 tokens
        // Message overhead: 4
        // Total: 3 + 26 + 4 = 33
        assert!(estimate > 25); // Verify it accounts for tool use
    }

    #[test]
    fn test_estimate_tool_result() {
        let message = Message {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tool_123".to_string(),
                content: "File contents here...".to_string(), // 21 chars = 6 tokens
                is_error: None,
            }]),
        };

        let estimate = TokenEstimator::estimate_message(&message);
        // 6 content + 10 overhead + 4 message overhead = 20
        assert_eq!(estimate, 20);
    }

    #[test]
    fn test_estimate_history() {
        let messages = vec![
            Message::user("Hello"),          // 5 chars = 2 tokens + 4 overhead = 6
            Message::assistant("Hi there!"), // 9 chars = 3 tokens + 4 overhead = 7
            Message::user("How are you?"),   // 12 chars = 3 tokens + 4 overhead = 7
        ];

        let estimate = TokenEstimator::estimate_history(&messages);
        assert_eq!(estimate, 20);
    }

    #[test]
    fn test_empty_history() {
        let messages: Vec<Message> = vec![];
        assert_eq!(TokenEstimator::estimate_history(&messages), 0);
    }

    #[test]
    fn test_estimate_redacted_thinking_uses_data_length() {
        // Simulate a realistic redacted thinking blob (~8KB base64 data).
        // 8192 base64 chars → ~6144 raw bytes → 6144/4 = 1536 estimated tokens.
        let data = "A".repeat(8192);
        let block = ContentBlock::RedactedThinking { data };

        let estimate = TokenEstimator::estimate_block(&block);
        assert_eq!(estimate, 1536);
    }

    #[test]
    fn test_estimate_redacted_thinking_respects_minimum() {
        // Tiny data blob: 100 base64 chars → ~75 raw bytes → 75/4 = 19 tokens.
        // Should be clamped to the minimum (512).
        let data = "A".repeat(100);
        let block = ContentBlock::RedactedThinking { data };

        let estimate = TokenEstimator::estimate_block(&block);
        assert_eq!(estimate, TokenEstimator::REDACTED_THINKING_MIN_TOKENS);
    }

    #[test]
    fn test_estimate_redacted_thinking_empty_data() {
        // Empty data should return the minimum floor.
        let block = ContentBlock::RedactedThinking {
            data: String::new(),
        };

        let estimate = TokenEstimator::estimate_block(&block);
        assert_eq!(estimate, TokenEstimator::REDACTED_THINKING_MIN_TOKENS);
    }

    #[test]
    fn test_redacted_thinking_accumulates_in_history() {
        // 5 redacted thinking blocks at ~2000 tokens each should produce a
        // meaningful total that triggers compaction.
        let blocks: Vec<ContentBlock> = (0..5)
            .map(|_| ContentBlock::RedactedThinking {
                data: "B".repeat(10_000), // 10k base64 → 7500 raw → 1875 tokens
            })
            .collect();
        let message = Message {
            role: Role::Assistant,
            content: Content::Blocks(blocks),
        };

        let estimate = TokenEstimator::estimate_message(&message);
        // 5 × 1875 + 4 message overhead = 9379
        assert_eq!(estimate, 9379);
    }
}
