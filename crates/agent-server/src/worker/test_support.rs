//! Shared streaming-capable scripted LLM provider for the worker test
//! suite.
//!
//! Before this harness existed every worker mock implemented only
//! [`LlmProvider::chat`], so the worker's `chat_stream` +
//! [`StreamAccumulator`](agent_sdk_providers::streaming::StreamAccumulator)
//! commit/ordering path ran the trait's *default* single-shot adapter
//! (`chat()` collapsed into one chunk) and the live streaming-commit
//! path went untested.  17 streaming tests were `#[ignore]`d as a
//! result.
//!
//! [`StreamingScriptedProvider`] drives the real streaming path: it
//! overrides [`LlmProvider::chat_stream`] to emit a caller-specified,
//! per-turn `Vec<Vec<StreamDelta>>` — one inner `Vec` per LLM call —
//! with optional per-delta delays (paired with `tokio::time` virtual
//! time for deterministic ordering tests) and a mid-stream error
//! injection point.  Its `chat()` impl rebuilds an equivalent
//! [`ChatOutcome`] from the same script so a stray non-streaming call
//! still behaves, but every worker path under test goes through
//! `chat_stream`.
//!
//! Ergonomic [`TurnScript`] builders (`text`, `tool_calls`,
//! `thinking_text`, `error`) mirror the shapes of the duplicated
//! chat-only mocks they replace, so porting a `chat()`-only mock is a
//! mechanical translation rather than a hand-written delta stream.
//! The raw [`TurnScript::from_deltas`] constructor backs the
//! streaming edge-case tests (interleaved text+tool deltas,
//! out-of-order `block_index`, truncated tool-use, stream-ends-without-
//! `Done`).
//!
//! No production code lives here; the module is `#[cfg(test)]`.

use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest, ChatResponse, StopReason, Usage};
use agent_sdk_providers::LlmProvider;
use agent_sdk_providers::streaming::{StreamAccumulator, StreamBox, StreamDelta, StreamErrorKind};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::sync::Mutex;
use std::time::Duration;

/// Default per-turn token usage attached to a scripted turn when the
/// caller does not emit an explicit [`StreamDelta::Usage`].  Matches the
/// `Usage{100,50,0,0}` shape the ported chat-only mocks used so token
/// assertions carry over unchanged.
const DEFAULT_USAGE: Usage = Usage {
    input_tokens: 100,
    output_tokens: 50,
    cached_input_tokens: 0,
    cache_creation_input_tokens: 0,
};

/// One scripted LLM call: the ordered deltas to stream, an optional
/// per-delta delay, and an optional mid-stream fault.
///
/// A [`StreamingScriptedProvider`] holds a FIFO queue of these; each
/// `chat_stream` / `chat` call consumes the next one.
#[derive(Clone)]
pub struct TurnScript {
    deltas: Vec<StreamDelta>,
    /// Wall-clock (virtual, under `tokio::test(start_paused)`) delay
    /// applied *before* each delta is yielded.  `None` = no delay.
    per_delta_delay: Option<Duration>,
    /// When set, the stream yields this many deltas, then yields a
    /// terminal [`StreamDelta::Error`] of the given kind and stops —
    /// the remaining scripted deltas are dropped.  Models a provider
    /// that dies mid-SSE (5xx / dropped TCP / 429).
    fail_after: Option<(usize, StreamErrorKind, String)>,
}

impl TurnScript {
    /// Raw constructor for hand-crafted edge-case delta sequences.
    #[must_use]
    pub fn from_deltas(deltas: Vec<StreamDelta>) -> Self {
        Self {
            deltas,
            per_delta_delay: None,
            fail_after: None,
        }
    }

    /// A text-only turn: chunks the text into `TextDelta`s on
    /// `block_index` 0, then `Usage` + `Done{EndTurn}`.
    #[must_use]
    pub fn text(text: &str) -> Self {
        Self::from_deltas(vec![
            StreamDelta::TextDelta {
                delta: text.to_owned(),
                block_index: 0,
            },
            StreamDelta::Usage(DEFAULT_USAGE),
            StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
            },
        ])
    }

    /// A text-only turn that streams `text` as multiple `TextDelta`
    /// chunks on the same block so the per-chunk commit path is
    /// exercised with more than one delta.
    #[must_use]
    pub fn text_chunked(chunks: &[&str]) -> Self {
        let mut deltas: Vec<StreamDelta> = chunks
            .iter()
            .map(|chunk| StreamDelta::TextDelta {
                delta: (*chunk).to_owned(),
                block_index: 0,
            })
            .collect();
        deltas.push(StreamDelta::Usage(DEFAULT_USAGE));
        deltas.push(StreamDelta::Done {
            stop_reason: Some(StopReason::EndTurn),
        });
        Self::from_deltas(deltas)
    }

    /// A text-only turn that ends with an explicit non-`EndTurn` stop
    /// reason (e.g. [`StopReason::Refusal`]).
    #[must_use]
    pub fn text_with_stop(text: &str, stop_reason: StopReason) -> Self {
        Self::from_deltas(vec![
            StreamDelta::TextDelta {
                delta: text.to_owned(),
                block_index: 0,
            },
            StreamDelta::Usage(DEFAULT_USAGE),
            StreamDelta::Done {
                stop_reason: Some(stop_reason),
            },
        ])
    }

    /// A thinking-then-text turn: `ThinkingDelta` + `SignatureDelta` on
    /// block 0, `TextDelta` on block 1, `Usage` + `Done{EndTurn}`.
    #[must_use]
    pub fn thinking_text(thinking: &str, signature: &str, text: &str) -> Self {
        Self::from_deltas(vec![
            StreamDelta::ThinkingDelta {
                delta: thinking.to_owned(),
                block_index: 0,
            },
            StreamDelta::SignatureDelta {
                delta: signature.to_owned(),
                block_index: 0,
            },
            StreamDelta::TextDelta {
                delta: text.to_owned(),
                block_index: 1,
            },
            StreamDelta::Usage(DEFAULT_USAGE),
            StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
            },
        ])
    }

    /// A tool-use turn: a `ToolUseStart` + a single `ToolInputDelta`
    /// carrying the full JSON for every `(id, name, input)`, then
    /// `Usage` + `Done{ToolUse}`.  Each tool occupies its own
    /// `block_index` in call order.
    #[must_use]
    pub fn tool_calls(calls: &[(&str, &str, serde_json::Value)]) -> Self {
        let mut deltas = Vec::new();
        for (block_index, (id, name, input)) in calls.iter().enumerate() {
            deltas.push(StreamDelta::ToolUseStart {
                id: (*id).to_owned(),
                name: (*name).to_owned(),
                block_index,
                thought_signature: None,
            });
            deltas.push(StreamDelta::ToolInputDelta {
                id: (*id).to_owned(),
                delta: input.to_string(),
                block_index,
            });
        }
        deltas.push(StreamDelta::Usage(DEFAULT_USAGE));
        deltas.push(StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        });
        Self::from_deltas(deltas)
    }

    /// A thinking-then-tool-use turn: `ThinkingDelta` + `SignatureDelta`
    /// on block 0, then each tool call on a subsequent block, then
    /// `Usage` + `Done{ToolUse}`.
    #[must_use]
    pub fn thinking_tool_calls(
        thinking: &str,
        signature: &str,
        calls: &[(&str, &str, serde_json::Value)],
    ) -> Self {
        let mut deltas = vec![
            StreamDelta::ThinkingDelta {
                delta: thinking.to_owned(),
                block_index: 0,
            },
            StreamDelta::SignatureDelta {
                delta: signature.to_owned(),
                block_index: 0,
            },
        ];
        for (offset, (id, name, input)) in calls.iter().enumerate() {
            let block_index = offset + 1;
            deltas.push(StreamDelta::ToolUseStart {
                id: (*id).to_owned(),
                name: (*name).to_owned(),
                block_index,
                thought_signature: None,
            });
            deltas.push(StreamDelta::ToolInputDelta {
                id: (*id).to_owned(),
                delta: input.to_string(),
                block_index,
            });
        }
        deltas.push(StreamDelta::Usage(DEFAULT_USAGE));
        deltas.push(StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        });
        Self::from_deltas(deltas)
    }

    /// A turn whose stream immediately yields a single terminal
    /// [`StreamDelta::Error`] of `kind`.
    #[must_use]
    pub fn error(kind: StreamErrorKind, message: &str) -> Self {
        Self::from_deltas(vec![StreamDelta::Error {
            message: message.to_owned(),
            kind,
        }])
    }

    /// Attach a uniform per-delta delay.  Combined with
    /// `#[tokio::test(start_paused = true)]` this advances virtual time
    /// deterministically without real sleeps.
    #[must_use]
    pub const fn with_delay(mut self, delay: Duration) -> Self {
        self.per_delta_delay = Some(delay);
        self
    }

    /// Inject a mid-stream fault: stream `n` deltas, then a terminal
    /// [`StreamDelta::Error`] of `kind`, dropping the rest.
    #[must_use]
    pub fn fail_after(mut self, n: usize, kind: StreamErrorKind, message: &str) -> Self {
        self.fail_after = Some((n, kind, message.to_owned()));
        self
    }

    /// Rebuild the [`ChatOutcome`] this script's deltas accumulate to,
    /// for the rare caller that drives `chat()` directly.
    fn to_chat_outcome(&self, model: &str) -> ChatOutcome {
        // A leading or fail-injected Error maps to the matching
        // non-streaming outcome variant.
        if let Some(StreamDelta::Error { message, kind }) = self.deltas.first() {
            return match kind {
                StreamErrorKind::RateLimited => ChatOutcome::RateLimited,
                StreamErrorKind::InvalidRequest => ChatOutcome::InvalidRequest(message.clone()),
                // `StreamErrorKind::ServerError`, plus the `#[non_exhaustive]`
                // catch-all: an unclassified error maps to a server error.
                _ => ChatOutcome::ServerError(message.clone()),
            };
        }

        let mut accumulator = StreamAccumulator::new();
        for delta in &self.deltas {
            accumulator.apply(delta);
        }
        let usage = accumulator.usage().cloned().unwrap_or(DEFAULT_USAGE);
        let stop_reason = accumulator.stop_reason().copied();
        ChatOutcome::Success(ChatResponse {
            id: "scripted".to_owned(),
            content: accumulator.into_content_blocks(),
            model: model.to_owned(),
            stop_reason,
            usage,
        })
    }
}

/// Streaming-capable scripted [`LlmProvider`] for worker tests.
///
/// Drains a FIFO queue of [`TurnScript`]s — one per `chat` /
/// `chat_stream` call.  When the queue runs dry it returns a fatal
/// error so accidental over-consumption surfaces loudly rather than
/// silently emitting empty turns.
pub struct StreamingScriptedProvider {
    scripts: Mutex<std::collections::VecDeque<TurnScript>>,
    model: &'static str,
    provider: &'static str,
}

impl StreamingScriptedProvider {
    /// Build a provider that serves `scripts` in order.
    #[must_use]
    pub fn new(scripts: Vec<TurnScript>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            model: "mock-model",
            provider: "mock",
        }
    }

    /// Single-turn convenience constructor.
    #[must_use]
    pub fn single(script: TurnScript) -> Self {
        Self::new(vec![script])
    }

    fn next_script(&self) -> Result<TurnScript> {
        let mut queue = self
            .scripts
            .lock()
            .map_err(|_| anyhow!("StreamingScriptedProvider lock poisoned"))?;
        queue
            .pop_front()
            .ok_or_else(|| anyhow!("StreamingScriptedProvider script queue exhausted"))
    }
}

#[async_trait]
impl LlmProvider for StreamingScriptedProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let script = self.next_script()?;
        Ok(script.to_chat_outcome(self.model))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let script = match self.next_script() {
            Ok(script) => script,
            Err(error) => {
                return Box::pin(async_stream::stream! {
                    yield Err(error);
                });
            }
        };

        Box::pin(async_stream::stream! {
            let TurnScript {
                deltas,
                per_delta_delay,
                fail_after,
            } = script;

            for (index, delta) in deltas.into_iter().enumerate() {
                if let Some((n, kind, message)) = &fail_after
                    && index == *n
                {
                    // Inject the fault and stop; remaining scripted
                    // deltas are intentionally dropped.
                    yield Ok(StreamDelta::Error {
                        message: message.clone(),
                        kind: *kind,
                    });
                    return;
                }

                if let Some(delay) = per_delta_delay {
                    tokio::time::sleep(delay).await;
                }
                yield Ok(delta);
            }
        })
    }

    fn model(&self) -> &'static str {
        self.model
    }

    fn provider(&self) -> &'static str {
        self.provider
    }
}
