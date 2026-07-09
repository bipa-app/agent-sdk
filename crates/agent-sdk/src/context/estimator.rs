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

    /// Minimum estimate for an opaque provider reasoning item.
    ///
    /// The SDK never interprets the item, but it is still replayed into model
    /// context and therefore needs a conservative non-zero estimate.
    const OPAQUE_REASONING_MIN_TOKENS: usize = 512;

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
            ContentBlock::OpaqueReasoning { provider, data } => {
                // Account for the wire-sized JSON without allocating a
                // serialized copy or exposing any payload value to logs.
                let payload_tokens = Self::estimate_json_len(data)
                    .div_ceil(Self::CHARS_PER_TOKEN)
                    .max(Self::OPAQUE_REASONING_MIN_TOKENS);
                Self::estimate_text(provider) + payload_tokens
            }
            ContentBlock::ToolUse { name, input, .. } => {
                // Estimate the serialized JSON length without actually
                // serializing: `needs_compaction` runs before every LLM call,
                // so allocating a String per tool-use block on every round-trip
                // is O(n^2) over a session. The recursive estimator also avoids
                // the silent 0-byte underestimate that `to_string(..)
                // .unwrap_or_default()` produced on a serialization failure.
                let input_len = Self::estimate_json_len(input);
                Self::estimate_text(name)
                    + input_len.div_ceil(Self::CHARS_PER_TOKEN)
                    + Self::TOOL_USE_OVERHEAD
            }
            ContentBlock::ToolResult { content, .. } => {
                Self::estimate_text(content) + Self::TOOL_RESULT_OVERHEAD
            }
            ContentBlock::Image { source } | ContentBlock::Document { source } => {
                // Rough estimate: base64 data is ~4/3 of original, 1 token per 4 chars
                source.data.len() / 4 + Self::MESSAGE_OVERHEAD
            }
            // `ContentBlock` is `#[non_exhaustive]`; charge an unknown future
            // block kind the per-message overhead as a conservative floor.
            _ => Self::MESSAGE_OVERHEAD,
        }
    }

    /// Estimate total tokens for a message history.
    #[must_use]
    pub fn estimate_history(messages: &[Message]) -> usize {
        messages.iter().map(Self::estimate_message).sum()
    }

    /// Approximate the serialized-JSON byte length of a value without
    /// allocating a serialized `String`.
    ///
    /// Mirrors `serde_json::to_string`'s output length closely enough for token
    /// estimation: it sums key/string lengths, structural punctuation, and a
    /// digit count for numbers. It is intentionally slightly conservative
    /// (over-counts a trailing separator per element) since over-estimating
    /// context size is safer than under-estimating it.
    fn estimate_json_len(value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Null => 4, // "null"
            serde_json::Value::Bool(b) => {
                if *b {
                    4 // "true"
                } else {
                    5 // "false"
                }
            }
            serde_json::Value::Number(n) => n.as_u64().map_or_else(
                || {
                    n.as_i64().map_or(
                        // Floating point or arbitrary-precision: a fixed
                        // estimate is fine for a token heuristic.
                        8,
                        |i| Self::decimal_digits(i.unsigned_abs()) + usize::from(i < 0),
                    )
                },
                Self::decimal_digits,
            ),
            // String value plus surrounding quotes.
            serde_json::Value::String(s) => s.len() + 2,
            serde_json::Value::Array(items) => {
                // Brackets plus a separator allowance per element.
                2 + items
                    .iter()
                    .map(|item| Self::estimate_json_len(item) + 1)
                    .sum::<usize>()
            }
            serde_json::Value::Object(entries) => {
                // Braces plus key (quoted) + ':' + value + ',' per entry.
                2 + entries
                    .iter()
                    .map(|(key, val)| key.len() + 2 + 1 + Self::estimate_json_len(val) + 1)
                    .sum::<usize>()
            }
        }
    }

    /// Count the decimal digits in a `u64` without allocating.
    const fn decimal_digits(mut n: u64) -> usize {
        let mut digits = 1;
        while n >= 10 {
            n /= 10;
            digits += 1;
        }
        digits
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
    fn estimate_opaque_reasoning_uses_payload_size_with_a_floor() {
        let small = ContentBlock::OpaqueReasoning {
            provider: "test-provider".to_owned(),
            data: json!({"encrypted_content": "x"}),
        };
        let large = ContentBlock::OpaqueReasoning {
            provider: "test-provider".to_owned(),
            data: json!({"encrypted_content": "x".repeat(8_192)}),
        };

        let small_estimate = TokenEstimator::estimate_block(&small);
        let large_estimate = TokenEstimator::estimate_block(&large);
        assert!(small_estimate >= TokenEstimator::OPAQUE_REASONING_MIN_TOKENS);
        assert!(large_estimate > small_estimate);
    }

    #[test]
    fn test_estimate_json_len_tracks_serialized_size() {
        // The no-allocation estimator should track the real serialized length
        // closely (within the per-element separator slack it intentionally adds).
        for value in [
            json!({"path": "/test.txt"}),
            json!({"a": 1, "b": [1, 2, 3], "c": {"nested": true}}),
            json!([null, false, "string", 12_345]),
            json!("plain string"),
            json!(9_876_543),
        ] {
            let estimated = TokenEstimator::estimate_json_len(&value);
            let actual = serde_json::to_string(&value).map_or(0, |s| s.len());
            assert!(
                estimated >= actual,
                "estimate {estimated} should be >= actual {actual} for {value}"
            );
            assert!(
                estimated <= actual * 2 + 8,
                "estimate {estimated} wildly exceeds actual {actual} for {value}"
            );
        }
    }

    #[test]
    fn test_tool_use_estimate_is_nonzero_for_nonempty_input() {
        // Regression: the old `to_string(..).unwrap_or_default()` path could
        // silently produce a 0-length input estimate. The recursive estimator
        // always accounts for the input.
        let block = ContentBlock::ToolUse {
            id: "tool_1".to_string(),
            name: "bash".to_string(),
            input: json!({"command": "echo hello world"}),
            thought_signature: None,
        };

        let estimate = TokenEstimator::estimate_block(&block);
        // name (1) + overhead (20) is 21; the input must add more on top.
        assert!(
            estimate > 21,
            "input length must contribute to the estimate"
        );
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
