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
            ContentBlock::ToolUse { name, input, .. } => {
                let input_str = serde_json::to_string(input).unwrap_or_default();
                Self::estimate_text(name)
                    + Self::estimate_text(&input_str)
                    + Self::TOOL_USE_OVERHEAD
            }
            ContentBlock::ToolResult { content, .. } => {
                Self::estimate_text(content) + Self::TOOL_RESULT_OVERHEAD
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
}
