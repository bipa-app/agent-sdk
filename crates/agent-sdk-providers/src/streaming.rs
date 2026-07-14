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
    /// The request could not establish a provider connection because DNS,
    /// routing, or the network is unavailable. Callers may wait indefinitely
    /// for connectivity, provided cancellation remains cooperative.
    Connectivity,
    /// An established provider response stream lost its underlying network
    /// connection. It has the same wait policy as [`Self::Connectivity`], but
    /// durable runtimes must close that provider call's audit attempt before
    /// retrying it.
    ConnectionLost,
    /// Provider returned HTTP 429 / explicit rate-limit signal.
    ///
    /// Carries the server-supplied retry delay when the provider gave one —
    /// a `Retry-After` header, or a hint embedded in the error body (Gemini's
    /// `google.rpc.RetryInfo`, `OpenAI`'s "try again in 20s"). `None` when the
    /// provider supplied no usable hint, in which case callers use backoff.
    RateLimited(Option<Duration>),
    /// Provider returned HTTP 5xx or reported a transient runtime failure.
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
    /// may retry. Connectivity, rate-limit, and server errors are
    /// recoverable; invalid-request is not.
    #[must_use]
    pub const fn is_recoverable(self) -> bool {
        matches!(
            self,
            Self::Connectivity | Self::ConnectionLost | Self::RateLimited(_) | Self::ServerError
        )
    }

    /// The server-supplied retry delay carried by a rate-limit error, if any.
    #[must_use]
    pub const fn retry_after(self) -> Option<Duration> {
        match self {
            Self::RateLimited(retry_after) => retry_after,
            _ => None,
        }
    }

    /// `true` for failures governed by the unbounded, cancellable offline wait.
    #[must_use]
    pub const fn is_connectivity(self) -> bool {
        matches!(self, Self::Connectivity | Self::ConnectionLost)
    }
}

/// Classify a typed HTTP client failure without relying on display text.
#[must_use]
pub fn classify_reqwest_error(error: &reqwest::Error) -> StreamErrorKind {
    if is_proxy_tunnel_rejection(error) || is_tls_rejection(error) {
        StreamErrorKind::ServerError
    } else if error.is_connect() {
        StreamErrorKind::Connectivity
    } else if error.is_timeout() || has_connectivity_io_source(error) {
        StreamErrorKind::ConnectionLost
    } else {
        StreamErrorKind::ServerError
    }
}

fn is_proxy_tunnel_rejection(error: &reqwest::Error) -> bool {
    if error.status() == Some(reqwest::StatusCode::PROXY_AUTHENTICATION_REQUIRED) {
        return true;
    }
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let message = cause.to_string();
        if message.contains("tunnel error: unsuccessful")
            || message.contains("proxy authorization required")
        {
            return true;
        }
        source = cause.source();
    }
    false
}

/// `true` when a TLS peer answered the handshake and rejected the session
/// (certificate validation, hostname mismatch, protocol or cipher
/// negotiation, an interception proxy presenting the wrong identity, …).
///
/// A peer that speaks TLS at us is reachable, so these are bounded server
/// errors, never connectivity waits: retrying cannot fix a policy or
/// configuration rejection, and misreading one as "offline" would park the
/// caller in an indefinite wait on a failure that is deterministic. The one
/// exception is a TLS-wrapped *transport* death — a socket that EOFs or
/// resets mid-handshake — which stays on the connectivity path.
fn is_tls_rejection(error: &reqwest::Error) -> bool {
    if has_connectivity_io_source(error) {
        return false;
    }
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        if cause.downcast_ref::<native_tls::Error>().is_some() {
            let message = cause.to_string().to_ascii_lowercase();
            let transport_death = ["eof", "close", "reset", "broken pipe", "timed out"]
                .iter()
                .any(|marker| message.contains(marker));
            return !transport_death;
        }
        source = cause.source();
    }
    false
}

fn has_connectivity_io_source(error: &reqwest::Error) -> bool {
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        if let Some(io_error) = cause.downcast_ref::<std::io::Error>()
            && matches!(
                io_error.kind(),
                std::io::ErrorKind::NotConnected
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::NetworkDown
                    | std::io::ErrorKind::NetworkUnreachable
                    | std::io::ErrorKind::HostUnreachable
            )
        {
            return true;
        }
        source = cause.source();
    }
    false
}

#[must_use]
pub fn reqwest_error_delta(context: &str, error: &reqwest::Error) -> StreamDelta {
    StreamDelta::Error {
        message: format!("{context}: {error}"),
        kind: classify_reqwest_error(error),
    }
}

#[must_use]
pub fn reqwest_body_error_delta(context: &str, error: &reqwest::Error) -> StreamDelta {
    let kind = match classify_reqwest_error(error) {
        StreamErrorKind::Connectivity => StreamErrorKind::ConnectionLost,
        other => other,
    };
    StreamDelta::Error {
        message: format!("{context}: {error}"),
        kind,
    }
}

/// Type alias for a boxed stream of stream deltas.
pub type StreamBox<'a> = Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send + 'a>>;

/// Sum two usage readings, saturating so token counters never wrap.
fn add_usage(carried: Option<&Usage>, usage: &Usage) -> Usage {
    let Some(carried) = carried else {
        return usage.clone();
    };
    Usage {
        input_tokens: carried.input_tokens.saturating_add(usage.input_tokens),
        output_tokens: carried.output_tokens.saturating_add(usage.output_tokens),
        cached_input_tokens: carried
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens),
        cache_creation_input_tokens: carried
            .cache_creation_input_tokens
            .saturating_add(usage.cache_creation_input_tokens),
    }
}

/// Preserves usage across a stream-splicing wrapper's attempt boundary.
///
/// A wrapper that abandons one inner stream and splices in another (failover,
/// credential refresh) has a hazard: [`StreamAccumulator`] keeps only the LAST
/// `Usage` delta it sees, so the retried stream's usage would erase the
/// abandoned one — silently un-billing tokens the provider already charged for.
///
/// The carry closes that: on every outgoing `Usage` delta, the wrapper calls
/// [`running_total`](Self::running_total) to rewrite it to the sum of all
/// abandoned attempts plus this stream's latest reading; when it abandons a
/// stream to retry, it calls [`abandon`](Self::abandon) to fold that stream's
/// usage into the carry. The final delta the consumer sees — the only one the
/// accumulator keeps — is therefore the true total across every attempt.
#[derive(Default)]
pub(crate) struct UsageCarry {
    /// Usage billed by abandoned attempts.
    carried: Option<Usage>,
    /// The current attempt's latest usage reading (last-wins within an attempt).
    current: Option<Usage>,
}

impl UsageCarry {
    pub(crate) const fn new() -> Self {
        Self {
            carried: None,
            current: None,
        }
    }

    /// Record `usage` as the current attempt's reading and return the running
    /// total (abandoned attempts + this reading) to yield in its place.
    pub(crate) fn running_total(&mut self, usage: Usage) -> Usage {
        let total = add_usage(self.carried.as_ref(), &usage);
        self.current = Some(usage);
        total
    }

    /// Fold the current attempt's usage into the carry because its stream is
    /// being abandoned for a retry.
    pub(crate) fn abandon(&mut self) {
        if let Some(current) = self.current.take() {
            self.carried = Some(add_usage(self.carried.as_ref(), &current));
        }
    }
}

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

    #[tokio::test]
    async fn classifies_typed_connect_failure_as_connectivity() -> anyhow::Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        drop(listener);

        let result = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await;
        let Err(error) = result else {
            anyhow::bail!("closed local port unexpectedly accepted a connection")
        };
        assert_eq!(
            classify_reqwest_error(&error),
            StreamErrorKind::Connectivity
        );
        Ok(())
    }

    #[tokio::test]
    async fn proxy_tunnel_rejection_is_not_connectivity() -> anyhow::Result<()> {
        use tokio::io::AsyncWriteExt as _;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            socket
                .write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                .await?;
            anyhow::Ok(())
        });
        let client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(format!("http://{address}"))?)
            .build()?;
        let Err(error) = client.get("https://example.invalid").send().await else {
            anyhow::bail!("rejected proxy tunnel unexpectedly succeeded")
        };
        assert_eq!(classify_reqwest_error(&error), StreamErrorKind::ServerError);
        server.await??;
        Ok(())
    }

    #[tokio::test]
    async fn tls_handshake_transport_drop_is_connectivity() -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await?;
            drop(socket);
            anyhow::Ok(())
        });
        let client = reqwest::Client::builder().no_proxy().build()?;
        let Err(error) = client.get(format!("https://{address}")).send().await else {
            anyhow::bail!("dropped TLS handshake unexpectedly succeeded")
        };
        assert_eq!(
            classify_reqwest_error(&error),
            StreamErrorKind::Connectivity
        );
        server.await??;
        Ok(())
    }

    /// A live TLS peer that fails certificate validation is a policy
    /// rejection, not an outage — waiting for connectivity cannot fix it, so
    /// it must stay on the bounded path. The server presents a self-signed
    /// certificate the client refuses; the assertion also pins the
    /// `native_tls::Error` downcast in `is_tls_rejection` against a version
    /// drift between reqwest's native-tls and the workspace's.
    #[tokio::test]
    async fn tls_certificate_rejection_is_bounded_server_error() -> anyhow::Result<()> {
        const SELF_SIGNED_CERT_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----
MIIDJzCCAg+gAwIBAgIUPiG3JI6c72crNdzYks8mo1pmHMEwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MCAXDTI2MDcxNDE4NDYzMloYDzIxMjYw
NjIwMTg0NjMyWjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCtBOh4EAP48fjE59F+L9qNEp/yUlOJXYJbm6m4nzTg
00RNc+dqsfrObIWJDuAaiKimunkGrSy77ELNAHlJmtOSkq8hu1C5/k6LW0GvPHuC
faPFEevCmxbERVZnt1f9IQ2e77oZz752cNzDlUIKyy5v3LpGaL8vT1bLAFuHT9z/
3mlqEwyK7mQlS3LZvwJQ6NfL2lgr5uVDFdsvfAY4mhbV8uRjKj+IZnOV1WYqQ62o
xbjC/NKXbvqKBigOhbo+idk1sjKbkjm2uvyjmUszRpfh7YX2wkk3UqZgN1+zsRDK
MBMyuZkkr7Vb/8ed07SN8Ma64fwCrrQba4l/R8TJmQpXAgMBAAGjbzBtMB0GA1Ud
DgQWBBT8LxETkCZh4h6qjMlLJMooNHTgkTAfBgNVHSMEGDAWgBT8LxETkCZh4h6q
jMlLJMooNHTgkTAPBgNVHRMBAf8EBTADAQH/MBoGA1UdEQQTMBGCCWxvY2FsaG9z
dIcEfwAAATANBgkqhkiG9w0BAQsFAAOCAQEACjZ8oqjFooFxjS3BnbhrNrF29/Jv
PbX32Tg3+3qUkS5+XnO64mLm+pQzUGs16+TyqdEkck//51KkyvzrnnGRYGc5eHEQ
zorkR1zlE+c8sjKcenvVkkLEKWaWNtEvpb+U0Ps6rP2Y1Jo4/AxTuxXrYxQ+XSTy
V4HyKriK6utlmhGpKUZhTZPTiTC/GaAwimCFgfw4wDuWGow92z3AnR9Q3KFpgrTP
B5z+i0oiNv6GpalGq3oe1ucKt+fduYWsC2Vea/PObZowciqbsA0mv3oHlyT9jPFT
hY9YjeYgUtEnf0BlrUrgbpd9DnVd5TNU0nDbPC7bv/yu8nF1nKUFWa2nsw==
-----END CERTIFICATE-----";
        const SELF_SIGNED_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCtBOh4EAP48fjE
59F+L9qNEp/yUlOJXYJbm6m4nzTg00RNc+dqsfrObIWJDuAaiKimunkGrSy77ELN
AHlJmtOSkq8hu1C5/k6LW0GvPHuCfaPFEevCmxbERVZnt1f9IQ2e77oZz752cNzD
lUIKyy5v3LpGaL8vT1bLAFuHT9z/3mlqEwyK7mQlS3LZvwJQ6NfL2lgr5uVDFdsv
fAY4mhbV8uRjKj+IZnOV1WYqQ62oxbjC/NKXbvqKBigOhbo+idk1sjKbkjm2uvyj
mUszRpfh7YX2wkk3UqZgN1+zsRDKMBMyuZkkr7Vb/8ed07SN8Ma64fwCrrQba4l/
R8TJmQpXAgMBAAECggEAAk8G9RctnmRIMARx4K+tyGUfukGL+NDFHQjSNnL1Zyya
hDgQNfXDBX8gNwh6SBBbw8HIPKUR7D4GVCr181v8B8AqUxZnSNwSWzyv/zEc6sxX
Y5lOHo4oOx07vm2NYITQ5DaJsq95eKYf5AI5W+CDMZ3t5GOgbXavD01la0RPDCD7
d+H9WI7RiKlCaiD174FQfSSwcAHpesrUcopPxMfZzpjxYClGdmMp7/RTmSVg8jex
eGceJvZujmjTnYczIce0Ibtozbq91qbwro32U2wbkvNpbU8GTG+st6nRlNRGmHeF
AJnOw+CiY9x7KaG4ZhsEY4VRk8YRJo/cLrPcx87JwQKBgQDv76cDYUgFzFXKWH+1
hc+oTLuUcn+X6E3ljvMKk9P4nQDgTRDxx5bBm6lHVv3IZoszi60Hvzblqr2HIO5S
Gl9KVBkHCLaYc8ny4rYQKVjKLA2/TnDE8Y8FhFnZTpBeEWhb2axQE5zb8WA6Ku6L
gEl04OSHjMlqAWt2Va5PZnFQQQKBgQC4mlfFFkfu92RKkYfhXkXUd5psSL4/1C1S
wYnqyL7rmAMmKO+y2MdnS1SAwSFGtmexibEcpDu8OASPQoovy4O5De+p/wL7v7aJ
+X2J9zaM2ggQN3tYz/HWCdCSpZJy+ufHtLwW9ESu0wW2G0ESRUxtEKvmBB/b/nrO
pK7VWxW0lwKBgQCWPG1LRIKgfs3JIZj1xI++Ri2+SeNy7ta3wsaT/PRhW43M5PST
L/JJ0HoyXVoTPYI0CGWT0DtDm6GJFymi5zh7hiUVrnMHCpmNKD/v5rPeA6+n9inO
Z6KyRaks1HC5NhUuTiIDEgTKA13JjlBHsVBNivQNnC4R3km3kvbOaMrTAQKBgAoR
6U3H/F6NwjvLGoVxtg90Asl7Yl1q/pnwEszq7Hc/kJRpUUIJTz9UPaTUZDNOSfPG
VhIA531J9P23nIAk8ueKWhOE5K3E9HksUevPv3sJfb0cua7LkR6i5GzLeWSqSTB8
rHH4GzMKMdqQPAl6HEQqz6W5fd9rT1msZBkhYdq7AoGBAJFTgwSK84D707FxGASw
SyuZBVIVd3iF341tsgX48Q1SVq70Uu6AQ0qPJHyxk6pe8aCiermlvVX26nqSQqr7
RrpkOaRQNnAfmLmSHvHWZmErDzlsl7pKdIByHK5nx1ccE8xspPEfHsg00E/SWWD+
CQR0IwmxMNda1bOi/AL4rcN3
-----END PRIVATE KEY-----";

        use anyhow::Context as _;

        let identity = native_tls::Identity::from_pkcs8(SELF_SIGNED_CERT_PEM, SELF_SIGNED_KEY_PEM)?;
        let acceptor = native_tls::TlsAcceptor::new(identity)?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || {
            if let Ok((socket, _)) = listener.accept() {
                // The client aborts after rejecting the certificate, so the
                // server-side handshake result is an expected error.
                drop(acceptor.accept(socket));
            }
        });

        let client = reqwest::Client::builder().no_proxy().build()?;
        let result = client
            .get(format!("https://localhost:{}", address.port()))
            .send()
            .await;
        let Err(error) = result else {
            anyhow::bail!("self-signed certificate unexpectedly accepted")
        };
        assert_eq!(classify_reqwest_error(&error), StreamErrorKind::ServerError);
        server
            .join()
            .ok()
            .context("TLS test server thread panicked")?;
        Ok(())
    }

    #[tokio::test]
    async fn classifies_premature_http_eof_as_connection_lost() -> anyhow::Result<()> {
        use tokio::io::AsyncWriteExt as _;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nx")
                .await?;
            anyhow::Ok(())
        });

        let response = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await?;
        let Err(error) = response.bytes().await else {
            anyhow::bail!("truncated HTTP body unexpectedly completed")
        };
        let StreamDelta::Error { kind, .. } = reqwest_body_error_delta("stream error", &error)
        else {
            anyhow::bail!("body error helper did not return an error delta")
        };
        assert_eq!(kind, StreamErrorKind::ConnectionLost);
        server.await??;
        Ok(())
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
