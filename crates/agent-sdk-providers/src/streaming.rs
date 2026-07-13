//! Streaming types for LLM responses.
//!
//! This module provides types for handling streaming responses from LLM providers.
//! The [`StreamDelta`] enum represents individual events in a streaming response,
//! and [`StreamAccumulator`] helps collect these events into a final response.

use agent_sdk_foundation::llm::{ContentBlock, StopReason, Usage};
#[cfg(any(feature = "openai", feature = "openai-codex"))]
use bytes::BytesMut;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

/// Upper bound on the block index [`StreamAccumulator`] will materialize.
///
/// `block_index` is taken verbatim from provider wire data (the SSE `index`
/// field) and `base_url` is user-configurable (any OpenAI-compatible endpoint),
/// so a corrupted or hostile event carrying a huge index could otherwise drive
/// an unbounded `Vec` allocation and exhaust host memory. Text/thinking deltas
/// whose index exceeds this bound are dropped with a warning rather than grown
/// into.
const MAX_BLOCK_INDEX: usize = 4096;

/// Incremental splitter for line-delimited SSE byte streams.
///
/// `reqwest`'s `bytes_stream` yields arbitrary byte boundaries, so a multi-byte
/// UTF-8 character can land split across two network chunks. Decoding each raw
/// chunk independently with `String::from_utf8_lossy` permanently corrupts such
/// characters into `U+FFFD` in user-visible text deltas. This buffer instead
/// accumulates raw bytes and only UTF-8-decodes *complete* lines (terminated by
/// `\n`); because a newline byte (`0x0A`) can never be part of a multi-byte
/// UTF-8 sequence, the end of a complete line is always a valid character
/// boundary and decodes losslessly.
///
/// It also avoids the quadratic `buffer = buffer[pos + 1..].to_string()` copy of
/// the naive splitter: [`BytesMut::split_to`] advances the read cursor without
/// copying the unconsumed tail, so splitting is amortized O(1) per line instead
/// of O(remaining-buffer).
#[cfg(any(feature = "openai", feature = "openai-codex"))]
#[derive(Debug, Default)]
pub(crate) struct SseLineBuffer {
    buf: BytesMut,
}

#[cfg(any(feature = "openai", feature = "openai-codex"))]
impl SseLineBuffer {
    /// Create an empty buffer.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append a freshly received network chunk.
    pub(crate) fn extend(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Pop the next complete line (without its trailing `\n`), or `None` when no
    /// full line is buffered yet. Incomplete trailing bytes — including a
    /// multi-byte character split across a chunk boundary — stay buffered for the
    /// next call.
    pub(crate) fn next_line(&mut self) -> Option<String> {
        let newline = self.buf.iter().position(|&b| b == b'\n')?;
        let mut line = self.buf.split_to(newline + 1);
        line.truncate(newline);
        Some(String::from_utf8_lossy(&line).into_owned())
    }
}

/// Events yielded during streaming LLM responses.
///
/// Each variant represents a different type of event that can occur
/// during a streaming response from an LLM provider.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamDelta {
    /// A text delta for streaming text content.
    TextDelta {
        /// The text fragment to append
        delta: String,
        /// Index of the content block being streamed
        block_index: usize,
    },

    /// A thinking delta for streaming thinking/reasoning content.
    ThinkingDelta {
        /// The thinking fragment to append
        delta: String,
        /// Index of the content block being streamed
        block_index: usize,
    },

    /// Start of a tool use block (name and id are known).
    ToolUseStart {
        /// Unique identifier for this tool call
        id: String,
        /// Name of the tool being called
        name: String,
        /// Index of the content block
        block_index: usize,
        /// Optional thought signature (used by Gemini 3.x models)
        thought_signature: Option<String>,
    },

    /// Incremental JSON for tool input (partial/incomplete JSON).
    ToolInputDelta {
        /// Tool call ID this delta belongs to
        id: String,
        /// JSON fragment to append
        delta: String,
        /// Index of the content block
        block_index: usize,
    },

    /// Usage information (typically at stream end).
    Usage(Usage),

    /// Stream completed with stop reason.
    Done {
        /// Why the stream ended
        stop_reason: Option<StopReason>,
    },

    /// A signature delta for a thinking block.
    SignatureDelta {
        /// The signature fragment to append
        delta: String,
        /// Index of the content block being streamed
        block_index: usize,
    },

    /// A redacted thinking block received at `content_block_start`.
    RedactedThinking {
        /// Opaque data payload
        data: String,
        /// Index of the content block
        block_index: usize,
    },

    /// A complete provider-owned reasoning-state item.
    ///
    /// Unlike text/thinking deltas this item is not user-visible and must not
    /// be interpreted. It is carried through the stream solely so agent
    /// history can replay it to the provider that owns it.
    OpaqueReasoning {
        /// Provider protocol that owns the payload.
        provider: String,
        /// Exact provider response item to preserve.
        data: serde_json::Value,
        /// Index used to retain the provider's output-item ordering.
        block_index: usize,
    },

    /// Error during streaming.
    Error {
        /// Error message
        message: String,
        /// Categorization of the error so downstream consumers can map
        /// it back to the correct [`agent_sdk_foundation::llm::ChatOutcome`]
        /// variant or audit-record `TurnAttemptOutcome` without losing
        /// the rate-limit / server-error / invalid-request distinction.
        kind: StreamErrorKind,
    },
}

/// Classification of a [`StreamDelta::Error`] event.
///
/// Mirrors [`ChatOutcome`](agent_sdk_foundation::llm::ChatOutcome)'s error
/// variants so providers that emit errors via streaming preserve the
/// same precision that non-streaming `chat()` callers see — every
/// supported provider can map its underlying error (HTTP status,
/// validation failure, mid-stream disconnect) directly onto one of
/// these categories at the construction site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamErrorKind {
    /// Provider returned HTTP 429 / explicit rate-limit signal.
    ///
    /// Carries the server-supplied retry delay when the provider gave one —
    /// a `Retry-After` header, or a hint embedded in the error body (Gemini's
    /// `google.rpc.RetryInfo`, `OpenAI`'s "try again in 20s"). `None` when the
    /// provider supplied no usable hint (mid-stream rate-limit events, for
    /// instance, arrive with no headers), in which case the caller falls back
    /// to its own backoff schedule.
    RateLimited(Option<Duration>),
    /// Provider returned HTTP 5xx, the connection dropped mid-stream,
    /// or the provider reported a transient runtime failure.
    ServerError,
    /// Caller-side error: validation failure before dispatch, HTTP
    /// 4xx other than 429, or a non-retriable provider rejection.
    InvalidRequest,
    /// Escape hatch for a streaming error a provider could not classify
    /// into one of the categories above.
    ///
    /// Producers should prefer a specific variant whenever the
    /// underlying signal (HTTP status, validation failure, mid-stream
    /// disconnect) allows it; `Unknown` exists so future error sources
    /// and providers can be added without a breaking change. It is
    /// treated as non-recoverable by [`StreamErrorKind::is_recoverable`]
    /// (callers should not blindly retry an unclassified failure).
    Unknown,
}

impl StreamErrorKind {
    /// `true` when the error is potentially transient and the caller
    /// may retry.  Rate-limit and server errors are recoverable;
    /// invalid-request is not.
    #[must_use]
    pub const fn is_recoverable(self) -> bool {
        matches!(self, Self::RateLimited(_) | Self::ServerError)
    }

    /// The server-supplied retry delay carried by a rate-limit error, if any.
    ///
    /// `None` for every other kind — only a rate limit comes with a hint.
    #[must_use]
    pub const fn retry_after(self) -> Option<Duration> {
        match self {
            Self::RateLimited(retry_after) => retry_after,
            _ => None,
        }
    }
}

/// Type alias for a boxed stream of stream deltas.
pub type StreamBox<'a> = Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send + 'a>>;

/// Helper to accumulate streamed content into a final response.
///
/// This struct collects [`StreamDelta`] events and can convert them
/// into the final content blocks once the stream is complete.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    /// Accumulated text for each block index
    text_blocks: Vec<String>,
    /// Accumulated thinking blocks for each block index
    thinking_blocks: Vec<String>,
    /// Accumulated signatures keyed by block index
    thinking_signatures: HashMap<usize, String>,
    /// Redacted thinking blocks: (`block_index`, data)
    redacted_thinking_blocks: Vec<(usize, String)>,
    /// Provider-owned opaque reasoning: (`block_index`, provider, data)
    opaque_reasoning_blocks: Vec<(usize, String, serde_json::Value)>,
    /// Accumulated tool use calls
    tool_uses: Vec<ToolUseAccumulator>,
    /// Usage information from the stream
    usage: Option<Usage>,
    /// Stop reason from the stream
    stop_reason: Option<StopReason>,
}

/// Accumulator for a single tool use during streaming.
#[derive(Debug, Default)]
pub struct ToolUseAccumulator {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Accumulated JSON input (may be incomplete during streaming)
    pub input_json: String,
    /// Block index for ordering
    pub block_index: usize,
    /// Optional thought signature (used by Gemini 3.x models)
    pub thought_signature: Option<String>,
}

impl StreamAccumulator {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a stream delta to the accumulator.
    pub fn apply(&mut self, delta: &StreamDelta) {
        match delta {
            StreamDelta::TextDelta { delta, block_index } => {
                if *block_index > MAX_BLOCK_INDEX {
                    log::warn!(
                        "dropping TextDelta with out-of-range block_index {block_index} (max {MAX_BLOCK_INDEX})"
                    );
                    return;
                }
                while self.text_blocks.len() <= *block_index {
                    self.text_blocks.push(String::new());
                }
                self.text_blocks[*block_index].push_str(delta);
            }
            StreamDelta::ThinkingDelta { delta, block_index } => {
                if *block_index > MAX_BLOCK_INDEX {
                    log::warn!(
                        "dropping ThinkingDelta with out-of-range block_index {block_index} (max {MAX_BLOCK_INDEX})"
                    );
                    return;
                }
                while self.thinking_blocks.len() <= *block_index {
                    self.thinking_blocks.push(String::new());
                }
                self.thinking_blocks[*block_index].push_str(delta);
            }
            StreamDelta::ToolUseStart {
                id,
                name,
                block_index,
                thought_signature,
            } => {
                self.tool_uses.push(ToolUseAccumulator {
                    id: id.clone(),
                    name: name.clone(),
                    input_json: String::new(),
                    block_index: *block_index,
                    thought_signature: thought_signature.clone(),
                });
            }
            StreamDelta::ToolInputDelta { id, delta, .. } => {
                if let Some(tool) = self.tool_uses.iter_mut().find(|t| t.id == *id) {
                    tool.input_json.push_str(delta);
                }
            }
            StreamDelta::SignatureDelta { delta, block_index } => {
                self.thinking_signatures
                    .entry(*block_index)
                    .or_default()
                    .push_str(delta);
            }
            StreamDelta::RedactedThinking { data, block_index } => {
                self.redacted_thinking_blocks
                    .push((*block_index, data.clone()));
            }
            StreamDelta::OpaqueReasoning {
                provider,
                data,
                block_index,
            } => {
                self.opaque_reasoning_blocks
                    .push((*block_index, provider.clone(), data.clone()));
            }
            StreamDelta::Usage(u) => {
                self.usage = Some(u.clone());
            }
            StreamDelta::Done { stop_reason } => {
                self.stop_reason = *stop_reason;
            }
            StreamDelta::Error { .. } => {}
        }
    }

    /// Get the accumulated usage information.
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Get the stop reason.
    #[must_use]
    pub const fn stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Convert accumulated content to `ContentBlock`s.
    ///
    /// This consumes the accumulator and returns the final content blocks.
    /// Tool use JSON is parsed at this point; invalid JSON results in a null input.
    #[must_use]
    pub fn into_content_blocks(self) -> Vec<ContentBlock> {
        let mut blocks: Vec<(usize, ContentBlock)> = Vec::new();

        // Add thinking blocks with their indices, attaching signatures
        let mut signatures = self.thinking_signatures;
        for (idx, thinking) in self.thinking_blocks.into_iter().enumerate() {
            if !thinking.is_empty() {
                let signature = signatures.remove(&idx).filter(|s| !s.is_empty());
                blocks.push((
                    idx,
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    },
                ));
            }
        }

        // Add redacted thinking blocks
        for (idx, data) in self.redacted_thinking_blocks {
            blocks.push((idx, ContentBlock::RedactedThinking { data }));
        }

        // Add provider-owned reasoning state without interpreting its payload.
        for (idx, provider, data) in self.opaque_reasoning_blocks {
            blocks.push((idx, ContentBlock::OpaqueReasoning { provider, data }));
        }

        // Add text blocks with their indices
        for (idx, text) in self.text_blocks.into_iter().enumerate() {
            if !text.is_empty() {
                blocks.push((idx, ContentBlock::Text { text }));
            }
        }

        // Add tool uses with their indices
        for tool in self.tool_uses {
            let input: serde_json::Value =
                serde_json::from_str(&tool.input_json).unwrap_or_else(|e| {
                    log::warn!(
                        "Failed to parse streamed tool input JSON for tool '{}' (id={}): {} — \
                         input_json ({} bytes): '{}'",
                        tool.name,
                        tool.id,
                        e,
                        tool.input_json.len(),
                        tool.input_json.chars().take(500).collect::<String>(),
                    );
                    serde_json::json!({})
                });
            blocks.push((
                tool.block_index,
                ContentBlock::ToolUse {
                    id: tool.id,
                    name: tool.name,
                    input,
                    thought_signature: tool.thought_signature,
                },
            ));
        }

        // Sort by block index to maintain order
        blocks.sort_by_key(|(idx, _)| *idx);

        blocks.into_iter().map(|(_, block)| block).collect()
    }

    /// Take ownership of accumulated usage.
    pub const fn take_usage(&mut self) -> Option<Usage> {
        self.usage.take()
    }

    /// Take ownership of stop reason.
    pub const fn take_stop_reason(&mut self) -> Option<StopReason> {
        self.stop_reason.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_text_deltas() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "Hello".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: " world".to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello world"));
    }

    #[test]
    fn test_accumulator_multiple_text_blocks() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "First".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: "Second".to_string(),
            block_index: 1,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "First"));
        assert!(matches!(&blocks[1], ContentBlock::Text { text } if text == "Second"));
    }

    #[test]
    fn test_accumulator_thinking_signature() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ThinkingDelta {
            delta: "Reasoning".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::SignatureDelta {
            delta: "sig_123".to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Thinking { thinking, signature }
            if thinking == "Reasoning" && signature.as_deref() == Some("sig_123")
        ));
    }

    #[test]
    fn accumulator_preserves_opaque_reasoning_payload_and_order() {
        let mut acc = StreamAccumulator::new();
        acc.apply(&StreamDelta::TextDelta {
            delta: "visible".to_owned(),
            block_index: 2,
        });
        acc.apply(&StreamDelta::OpaqueReasoning {
            provider: "test-provider".to_owned(),
            data: serde_json::json!({
                "id": "reasoning_1",
                "encrypted_content": "do-not-inspect"
            }),
            block_index: 1,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(
            &blocks[0],
            ContentBlock::OpaqueReasoning { provider, data }
                if provider == "test-provider"
                    && data["id"] == "reasoning_1"
                    && data["encrypted_content"] == "do-not-inspect"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text } if text == "visible"
        ));
    }

    #[test]
    fn test_accumulator_tool_use() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            block_index: 0,
            thought_signature: None,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_123".to_string(),
            delta: r#"{"path":"#.to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_123".to_string(),
            delta: r#""test.txt"}"#.to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "test.txt");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_mixed_content() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "Let me read that file.".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_456".to_string(),
            name: "read_file".to_string(),
            block_index: 1,
            thought_signature: None,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_456".to_string(),
            delta: r#"{"path":"file.txt"}"#.to_string(),
            block_index: 1,
        });
        acc.apply(&StreamDelta::Usage(Usage {
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }));
        acc.apply(&StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        });

        assert!(acc.usage().is_some());
        assert_eq!(acc.usage().map(|u| u.input_tokens), Some(100));
        assert!(matches!(acc.stop_reason(), Some(StopReason::ToolUse)));

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { .. }));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_accumulator_invalid_tool_json() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_789".to_string(),
            name: "test_tool".to_string(),
            block_index: 0,
            thought_signature: None,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_789".to_string(),
            delta: "invalid json {".to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert!(input.is_object());
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_empty_tool_input_falls_back_to_empty_object() {
        // If no ToolInputDelta is received (e.g., stream interrupted or
        // deltas had mismatched IDs), the tool use block should still be
        // produced with an empty object so that the error is attributable
        // to the tool rather than silently lost.
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_empty".to_string(),
            name: "read".to_string(),
            block_index: 0,
            thought_signature: None,
        });
        // No ToolInputDelta applied

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { input, name, .. } => {
                assert_eq!(name, "read");
                assert_eq!(input, &serde_json::json!({}));
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_mismatched_delta_id_drops_input() {
        // If ToolInputDelta has a different ID than any ToolUseStart,
        // the input is silently dropped (the tool gets empty {}).
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_A".to_string(),
            name: "bash".to_string(),
            block_index: 0,
            thought_signature: None,
        });
        // Delta with wrong ID
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_B".to_string(),
            delta: r#"{"command":"ls"}"#.to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { input, .. } => {
                // Input should be empty because the delta had a mismatched ID
                assert_eq!(input, &serde_json::json!({}));
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_empty() {
        let acc = StreamAccumulator::new();
        let blocks = acc.into_content_blocks();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_accumulator_skips_empty_text() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: String::new(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: "Hello".to_string(),
            block_index: 1,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello"));
    }

    #[test]
    fn test_accumulator_ignores_out_of_range_block_index() {
        // A hostile/corrupted event with a huge block_index must not drive an
        // unbounded Vec allocation. The delta is dropped, leaving the accumulator
        // tiny rather than allocating billions of empty Strings.
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "ok".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: "boom".to_string(),
            block_index: usize::MAX,
        });
        acc.apply(&StreamDelta::ThinkingDelta {
            delta: "boom".to_string(),
            block_index: usize::MAX,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "ok"));
    }

    #[cfg(any(feature = "openai", feature = "openai-codex"))]
    #[test]
    fn test_sse_line_buffer_splits_multiple_lines() {
        let mut buf = SseLineBuffer::new();
        buf.extend(b"data: one\ndata: two\n");
        assert_eq!(buf.next_line().as_deref(), Some("data: one"));
        assert_eq!(buf.next_line().as_deref(), Some("data: two"));
        assert_eq!(buf.next_line(), None);
    }

    #[cfg(any(feature = "openai", feature = "openai-codex"))]
    #[test]
    fn test_sse_line_buffer_buffers_partial_line_until_newline() {
        let mut buf = SseLineBuffer::new();
        buf.extend(b"data: par");
        assert_eq!(buf.next_line(), None);
        buf.extend(b"tial\n");
        assert_eq!(buf.next_line().as_deref(), Some("data: partial"));
    }

    #[cfg(any(feature = "openai", feature = "openai-codex"))]
    #[test]
    fn test_sse_line_buffer_handles_utf8_split_across_chunks() {
        // "café" — the 'é' is the two bytes 0xC3 0xA9. Split the chunk boundary
        // *inside* that character: the naive per-chunk from_utf8_lossy would emit
        // a U+FFFD replacement char; the line buffer must decode it losslessly
        // because it only decodes the complete line.
        let mut buf = SseLineBuffer::new();
        let line = "data: café\n";
        let bytes = line.as_bytes();
        let split = bytes.len() - 2; // between 0xC3 and 0xA9
        buf.extend(&bytes[..split]);
        assert_eq!(buf.next_line(), None);
        buf.extend(&bytes[split..]);
        assert_eq!(buf.next_line().as_deref(), Some("data: café"));
    }
}
