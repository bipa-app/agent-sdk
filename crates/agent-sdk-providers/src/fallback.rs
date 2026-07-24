//! Provider failover: try an ordered list of providers until one succeeds.
//!
//! [`FallbackProvider`] wraps a primary [`LlmProvider`] plus an ordered list of
//! secondaries. On a *retryable* failure — a [`ChatOutcome::RateLimited`] /
//! [`ChatOutcome::ServerError`], or a transport-level error (timeout, dropped
//! connection) — it advances to the next provider. Non-retryable outcomes
//! ([`ChatOutcome::InvalidRequest`] and a successful response) short-circuit and
//! are returned as-is, since retrying them on a different backend would not
//! help.
//!
//! `FallbackProvider` itself implements [`LlmProvider`], so it composes with the
//! rest of the stack (`run_structured`, `RefreshingProvider`, `ModelRouter`,
//! the agent-sdk facade) anywhere a `&dyn LlmProvider` is expected.

use std::sync::Arc;

use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use crate::provider::LlmProvider;
use crate::streaming::{StreamBox, StreamDelta, UsageCarry};

/// Whether a delta commits the stream to the provider that produced it.
///
/// Failing over after a provider has emitted output would make the secondary
/// re-emit it, so the classification is per-variant:
///
/// * **Commits** — anything the consumer can already see or will replay:
///   `TextDelta`, `ThinkingDelta`, `SignatureDelta`, `RedactedThinking`,
///   `OpaqueReasoning`, `ToolUseStart`, `ToolInputDelta`, and `Done` (a response
///   the caller has, by definition, already received in full). Unknown
///   (`#[non_exhaustive]`) variants commit with them: a delta this SDK version
///   cannot classify might carry content, and duplicating output is a worse
///   failure than missing a failover.
/// * **Does not commit** — `Usage`, which is pure metadata: it never reaches the
///   user, it is folded into token counters, and a provider that reports usage
///   and *then* fails is exactly the case failover exists for. A provider that
///   reports usage before any content and then errors is therefore failed over
///   now where it previously surfaced the error — deliberate: nothing visible is
///   duplicated, because usage is invisible.
/// * **Not classified here** — `Error`, which the caller's own arm handles.
///
/// The per-variant behaviour is pinned by
/// `only_metadata_deltas_leave_the_chain_uncommitted`.
const fn commits_stream(delta: &StreamDelta) -> bool {
    !matches!(delta, StreamDelta::Usage(_))
}

/// An [`LlmProvider`] that fails over across an ordered list of backends.
///
/// Construct with a primary, then layer secondaries with
/// [`with_fallback`](Self::with_fallback) (or build the whole chain at once with
/// [`from_providers`](Self::from_providers)).
pub struct FallbackProvider {
    primary: Arc<dyn LlmProvider>,
    fallbacks: Vec<Arc<dyn LlmProvider>>,
}

impl FallbackProvider {
    /// Create a fallback chain with a single primary provider (no secondaries
    /// yet).
    #[must_use]
    pub fn new(primary: Arc<dyn LlmProvider>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
        }
    }

    /// Append a secondary provider to the end of the failover order.
    #[must_use]
    pub fn with_fallback(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Build a chain from a primary and an ordered iterator of secondaries.
    #[must_use]
    pub fn from_providers(
        primary: Arc<dyn LlmProvider>,
        fallbacks: impl IntoIterator<Item = Arc<dyn LlmProvider>>,
    ) -> Self {
        Self {
            primary,
            fallbacks: fallbacks.into_iter().collect(),
        }
    }

    /// Total number of providers in the chain (primary + secondaries).
    #[must_use]
    pub fn len(&self) -> usize {
        1 + self.fallbacks.len()
    }

    /// Always `false` — a chain always has at least the primary.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The providers in failover order (primary first), cloned for ownership.
    fn ordered(&self) -> Vec<Arc<dyn LlmProvider>> {
        let mut providers = Vec::with_capacity(self.len());
        providers.push(Arc::clone(&self.primary));
        providers.extend(self.fallbacks.iter().map(Arc::clone));
        providers
    }
}

/// A `chat` result is worth retrying on the next provider when the provider was
/// rate-limited or server-errored, or the call failed at the transport layer.
const fn is_retryable(result: &Result<ChatOutcome>) -> bool {
    matches!(
        result,
        Err(_) | Ok(ChatOutcome::RateLimited(_) | ChatOutcome::ServerError(_))
    )
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let providers = self.ordered();
        let last = providers.len() - 1;
        for (idx, provider) in providers.iter().enumerate() {
            let result = provider.chat(request.clone()).await;
            if idx == last || !is_retryable(&result) {
                return result;
            }
            log::warn!(
                "FallbackProvider: provider '{}' failed retryably, failing over to next",
                provider.provider()
            );
        }
        // `ordered()` is never empty (the primary is always present), so the
        // loop above always returns. This keeps the signature total without an
        // `unwrap`.
        Ok(ChatOutcome::ServerError(
            "FallbackProvider: no providers configured".to_owned(),
        ))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        let providers = self.ordered();
        Box::pin(async_stream::stream! {
            let last = providers.len() - 1;
            // Preserves the tokens billed by providers this chain abandoned, so
            // the surviving provider's usage delta reports the running total
            // rather than erasing the abandoned attempt's usage (last-wins
            // accumulator). See `UsageCarry`.
            let mut usage_carry = UsageCarry::new();

            for (idx, provider) in providers.iter().enumerate() {
                let is_last = idx == last;
                let mut stream = provider.chat_stream(request.clone());
                // Whether this provider has emitted a delta that commits the
                // stream to it (see `commits_stream`). Once it has, failing over
                // would double-emit output, so a later error is surfaced as-is.
                let mut committed = false;
                let mut failed_over = false;

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(StreamDelta::Error { message, kind }) => {
                            if !committed && !is_last && kind.is_recoverable() {
                                log::warn!(
                                    "FallbackProvider: provider '{}' recoverable stream error ({kind:?}), failing over",
                                    provider.provider()
                                );
                                failed_over = true;
                                break;
                            }
                            yield Ok(StreamDelta::Error { message, kind });
                        }
                        Ok(delta) => {
                            // `commits_stream` is the single decision point:
                            // metadata leaves the chain free to fail over, and
                            // usage is additionally rewritten to the running
                            // total so the abandoned provider's tokens survive.
                            committed = committed || commits_stream(&delta);
                            let delta = match delta {
                                StreamDelta::Usage(usage) => {
                                    StreamDelta::Usage(usage_carry.running_total(usage))
                                }
                                other => other,
                            };
                            yield Ok(delta);
                        }
                        Err(error) => {
                            if !committed && !is_last {
                                log::warn!(
                                    "FallbackProvider: provider '{}' stream transport error, failing over: {error}",
                                    provider.provider()
                                );
                                failed_over = true;
                                break;
                            }
                            yield Err(error);
                        }
                    }
                }

                if !failed_over {
                    return;
                }
                usage_carry.abandon();
            }
        })
    }

    /// Delegate live model discovery to the primary provider so wrapping in a
    /// fallback never silently loses `list_models`.
    async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
        self.primary.list_models().await
    }

    /// Reachable when any provider in the chain is reachable — the chain can
    /// serve a request as long as one backend can be dialled.
    async fn probe_connectivity(&self) -> bool {
        for provider in self.ordered() {
            if provider.probe_connectivity().await {
                return true;
            }
        }
        false
    }

    fn model(&self) -> &str {
        self.primary.model()
    }

    fn provider(&self) -> &'static str {
        self.primary.provider()
    }

    /// Reports the **configured primary's** route — pre-dispatch identity
    /// only. Post-dispatch attribution does not go through this method: the
    /// provider that actually serves a stream stamps its route on
    /// [`StreamDelta::Done`], which this wrapper forwards untouched, so a
    /// failed-over call still attributes to the backend that answered.
    fn route(&self) -> &str {
        self.primary.route()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agent_sdk_foundation::llm::{ChatResponse, ContentBlock, StopReason, Usage};
    use anyhow::{Context as _, anyhow};

    use crate::streaming::StreamErrorKind;

    /// A provider that replays a queue of `chat` results and counts its calls.
    struct ScriptedProvider {
        name: &'static str,
        results: Mutex<std::collections::VecDeque<Result<ChatOutcome>>>,
        stream_deltas: Mutex<Vec<Result<StreamDelta>>>,
        calls: AtomicUsize,
    }

    impl ScriptedProvider {
        fn chat_only(name: &'static str, results: Vec<Result<ChatOutcome>>) -> Arc<Self> {
            Arc::new(Self {
                name,
                results: Mutex::new(results.into()),
                stream_deltas: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            })
        }

        fn streaming(name: &'static str, deltas: Vec<Result<StreamDelta>>) -> Arc<Self> {
            Arc::new(Self {
                name,
                results: Mutex::new(std::collections::VecDeque::new()),
                stream_deltas: Mutex::new(deltas),
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.results
                .lock()
                .map_err(|_| anyhow!("results lock poisoned"))?
                .pop_front()
                .unwrap_or_else(|| Ok(ChatOutcome::ServerError("exhausted".to_owned())))
        }

        async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
            Ok(vec![crate::provider::ModelInfo {
                id: format!("{}-model", self.name),
                display_name: None,
                context_window: None,
                max_output_tokens: None,
            }])
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let deltas: Vec<Result<StreamDelta>> = self
                .stream_deltas
                .lock()
                .map(|d| {
                    d.iter()
                        .map(|r| match r {
                            Ok(delta) => Ok(delta.clone()),
                            Err(e) => Err(anyhow!("{e}")),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Box::pin(async_stream::stream! {
                for delta in deltas {
                    yield delta;
                }
            })
        }

        fn model(&self) -> &str {
            self.name
        }

        fn provider(&self) -> &'static str {
            self.name
        }
    }

    fn success(text: &str) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "r".to_owned(),
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            model: "m".to_owned(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    fn request() -> ChatRequest {
        ChatRequest::new("sys", vec![agent_sdk_foundation::llm::Message::user("hi")])
    }

    #[tokio::test]
    async fn server_error_fails_over_to_secondary() -> Result<()> {
        let primary = ScriptedProvider::chat_only(
            "primary",
            vec![Ok(ChatOutcome::ServerError("boom".to_owned()))],
        );
        let secondary = ScriptedProvider::chat_only("secondary", vec![Ok(success("ok"))]);
        let fb = FallbackProvider::new(primary.clone()).with_fallback(secondary.clone());

        let outcome = fb.chat(request()).await?;
        assert!(matches!(outcome, ChatOutcome::Success(r) if r.first_text() == Some("ok")));
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn rate_limit_fails_over() -> Result<()> {
        let primary =
            ScriptedProvider::chat_only("primary", vec![Ok(ChatOutcome::RateLimited(None))]);
        let secondary = ScriptedProvider::chat_only("secondary", vec![Ok(success("ok"))]);
        let fb = FallbackProvider::from_providers(primary.clone(), [secondary.clone() as Arc<_>]);

        let outcome = fb.chat(request()).await?;
        assert!(matches!(outcome, ChatOutcome::Success(_)));
        assert_eq!(secondary.calls(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn transport_error_fails_over() -> Result<()> {
        let primary = ScriptedProvider::chat_only("primary", vec![Err(anyhow!("timeout"))]);
        let secondary = ScriptedProvider::chat_only("secondary", vec![Ok(success("ok"))]);
        let fb = FallbackProvider::new(primary.clone()).with_fallback(secondary.clone());

        let outcome = fb.chat(request()).await?;
        assert!(matches!(outcome, ChatOutcome::Success(_)));
        Ok(())
    }

    #[tokio::test]
    async fn invalid_request_does_not_fail_over() -> Result<()> {
        let primary = ScriptedProvider::chat_only(
            "primary",
            vec![Ok(ChatOutcome::InvalidRequest("bad".to_owned()))],
        );
        let secondary = ScriptedProvider::chat_only("secondary", vec![Ok(success("ok"))]);
        let fb = FallbackProvider::new(primary.clone()).with_fallback(secondary.clone());

        let outcome = fb.chat(request()).await?;
        assert!(matches!(outcome, ChatOutcome::InvalidRequest(_)));
        // The non-retryable outcome short-circuits: the secondary is untouched.
        assert_eq!(secondary.calls(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn last_provider_outcome_is_returned_when_all_fail() -> Result<()> {
        let primary = ScriptedProvider::chat_only(
            "primary",
            vec![Ok(ChatOutcome::ServerError("a".to_owned()))],
        );
        let secondary = ScriptedProvider::chat_only(
            "secondary",
            vec![Ok(ChatOutcome::ServerError("b".to_owned()))],
        );
        let fb = FallbackProvider::new(primary).with_fallback(secondary);

        let outcome = fb.chat(request()).await?;
        assert!(matches!(outcome, ChatOutcome::ServerError(msg) if msg == "b"));
        Ok(())
    }

    #[tokio::test]
    async fn list_models_delegates_to_primary() -> Result<()> {
        let primary = ScriptedProvider::chat_only("primary", vec![]);
        let secondary = ScriptedProvider::chat_only("secondary", vec![]);
        let fb = FallbackProvider::new(primary).with_fallback(secondary);

        let models = fb.list_models().await?;
        // Discovery is served by the primary, not the default "unsupported".
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "primary-model");
        Ok(())
    }

    #[test]
    fn only_metadata_deltas_leave_the_chain_uncommitted() {
        // Content — anything the consumer can see or replay — commits: failing
        // over after it would make the secondary emit it a second time.
        for delta in [
            StreamDelta::TextDelta {
                delta: "hi".to_owned(),
                block_index: 0,
            },
            StreamDelta::ThinkingDelta {
                delta: "hmm".to_owned(),
                block_index: 0,
            },
            StreamDelta::ToolUseStart {
                id: "t1".to_owned(),
                name: "echo".to_owned(),
                block_index: 0,
                thought_signature: None,
            },
            StreamDelta::ToolInputDelta {
                id: "t1".to_owned(),
                delta: "{}".to_owned(),
                block_index: 0,
            },
            StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
                served_route: None,
            },
        ] {
            assert!(
                commits_stream(&delta),
                "content/terminal deltas must commit the chain: {delta:?}"
            );
        }

        // Usage is metadata the user never sees, so it must not strand the chain
        // on a provider that reported its billing and then died.
        assert!(!commits_stream(&StreamDelta::Usage(Usage {
            input_tokens: 1,
            output_tokens: 1,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        })));
    }

    #[tokio::test]
    async fn stream_usage_before_an_error_does_not_suppress_failover() -> Result<()> {
        // Providers now report the tokens a failing turn burned *before* the
        // terminal error. That usage delta must not commit the chain to a dead
        // primary — it is metadata the user never sees — and the tokens it
        // reports must survive into the surviving provider's usage, because the
        // accumulator keeps only the last usage delta.
        let primary = ScriptedProvider::streaming(
            "primary",
            vec![
                Ok(StreamDelta::Usage(Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                })),
                Ok(StreamDelta::Error {
                    message: "rate limited".to_owned(),
                    kind: StreamErrorKind::RateLimited(None),
                }),
            ],
        );
        let secondary = ScriptedProvider::streaming(
            "secondary",
            vec![
                Ok(StreamDelta::TextDelta {
                    delta: "hello".to_owned(),
                    block_index: 0,
                }),
                Ok(StreamDelta::Usage(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                })),
                Ok(StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: None,
                }),
            ],
        );
        let fb = FallbackProvider::new(primary.clone()).with_fallback(secondary.clone());

        let mut accumulator = crate::streaming::StreamAccumulator::new();
        let mut stream = fb.chat_stream(request());
        let mut saw_error = false;
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            let delta = item?;
            if let StreamDelta::Error { .. } = &delta {
                saw_error = true;
            }
            if let StreamDelta::TextDelta { delta, .. } = &delta {
                text.push_str(delta);
            }
            accumulator.apply(&delta);
        }
        drop(stream);

        assert!(
            !saw_error,
            "the usage delta must not commit us to the primary"
        );
        assert_eq!(text, "hello", "the secondary must serve the turn");
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 1);

        // The primary genuinely billed 150 tokens before dying; the secondary
        // billed 15. The consumer must end up accounting for both.
        let usage = accumulator
            .usage()
            .context("the failover must still report usage")?;
        assert_eq!(usage.input_tokens, 110);
        assert_eq!(usage.output_tokens, 55);
        Ok(())
    }

    #[tokio::test]
    async fn stream_fails_over_on_recoverable_first_error() -> Result<()> {
        let primary = ScriptedProvider::streaming(
            "primary",
            vec![Ok(StreamDelta::Error {
                message: "rate limited".to_owned(),
                kind: StreamErrorKind::RateLimited(None),
            })],
        );
        let secondary = ScriptedProvider::streaming(
            "secondary",
            vec![
                Ok(StreamDelta::TextDelta {
                    delta: "hello".to_owned(),
                    block_index: 0,
                }),
                Ok(StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: None,
                }),
            ],
        );
        let fb = FallbackProvider::new(primary.clone()).with_fallback(secondary.clone());

        let mut stream = fb.chat_stream(request());
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            if let StreamDelta::TextDelta { delta, .. } = item? {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(primary.calls(), 1);
        assert_eq!(secondary.calls(), 1);
        Ok(())
    }

    #[test]
    fn reports_primary_identity() {
        let primary = ScriptedProvider::chat_only("primary", vec![]);
        let fb = FallbackProvider::new(primary);
        assert_eq!(fb.provider(), "primary");
        assert_eq!(fb.model(), "primary");
        assert_eq!(fb.len(), 1);
        assert!(!fb.is_empty());
    }

    /// A failed-over stream's `Done` still names the backend that served
    /// it: the chain forwards the secondary's `served_route` untouched,
    /// even though the wrapper's own `route()` keeps naming the primary.
    #[tokio::test]
    async fn failed_over_stream_attributes_the_serving_secondary() -> Result<()> {
        let primary = ScriptedProvider::streaming(
            "primary",
            vec![Ok(StreamDelta::Error {
                message: "boom".to_owned(),
                kind: StreamErrorKind::ServerError,
            })],
        );
        let secondary = ScriptedProvider::streaming(
            "secondary",
            vec![
                Ok(StreamDelta::TextDelta {
                    delta: "hello".to_owned(),
                    block_index: 0,
                }),
                Ok(StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: Some("secondary".to_owned()),
                }),
            ],
        );
        let fb = FallbackProvider::new(primary).with_fallback(secondary);
        assert_eq!(fb.route(), "primary");

        let mut stream = fb.chat_stream(request());
        let mut served = None;
        while let Some(item) = stream.next().await {
            if let StreamDelta::Done { served_route, .. } = item? {
                served = served_route;
            }
        }
        assert_eq!(served.as_deref(), Some("secondary"));
        Ok(())
    }
}
