//! Provider wrapper that refreshes credentials on 401 and retries once.
//!
//! [`RefreshingProvider`] wraps any [`LlmProvider`] and adds a host-driven
//! credential refresh step: when the inner provider reports an unauthorized
//! error (HTTP 401, expired OAuth token, invalid API key, etc.), the wrapper
//! calls a host-supplied async callback to rebuild the inner provider with
//! fresh credentials and retries the original request once.
//!
//! This is the generic form of the per-provider refresh wrappers that
//! OAuth-backed hosts would otherwise copy across every provider they use.
//!
//! # Example
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use anyhow::Result;
//! # use agent_sdk_providers::{LlmProvider, RefreshingProvider};
//! # async fn demo<P: LlmProvider + Clone + 'static>(initial: P) -> Result<()> {
//! let refreshed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
//! let counter = Arc::clone(&refreshed_count);
//! let wrapped = RefreshingProvider::new(initial.clone(), move || {
//!     let counter = Arc::clone(&counter);
//!     let provider = initial.clone();
//!     async move {
//!         counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
//!         Ok(provider)
//!     }
//! });
//! # let _ = wrapped;
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming semantics
//!
//! For non-streaming [`chat`](LlmProvider::chat), the retry happens when the
//! first call resolves to [`ChatOutcome::InvalidRequest`] with a 401-looking
//! message. The second call replaces the first — callers only see the final
//! outcome.
//!
//! For [`chat_stream`](LlmProvider::chat_stream), retry happens only when the
//! 401 arrives before any content delta is forwarded. If a
//! [`StreamDelta::TextDelta`], [`StreamDelta::ThinkingDelta`], tool-call
//! delta, or thinking signature has already been yielded to the consumer,
//! the error is forwarded as-is — retrying would duplicate partial output.
//!
//! At most one retry happens per call, whether streaming or not. If the
//! retried call fails in the same way, the wrapper surfaces the second
//! error unchanged.

use std::future::Future;
use std::sync::Arc;

use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest, ThinkingConfig};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::Mutex;

use crate::model_capabilities::ModelCapabilities;
use crate::provider::{LlmProvider, StructuredOutputSupport};
use crate::streaming::{StreamBox, StreamDelta};

/// Wraps a provider with host-driven credential refresh on 401.
///
/// The inner provider is stored behind `Arc<Mutex<P>>` so it can be swapped
/// atomically when the refresh callback produces a new provider. Cloning a
/// wrapper is cheap — clones share the same inner state.
///
/// Metadata (`model`, `provider`, `configured_thinking`) and capability shaping
/// (`capabilities`, `default_max_tokens`, `validate_thinking_config`,
/// `structured_output_support`) are captured from the initial provider at
/// construction time and assumed constant across refreshes (the refresh
/// callback rebuilds the same provider shape with a fresh token, not a
/// different model).
pub struct RefreshingProvider<P, F> {
    inner: Arc<Mutex<P>>,
    refresh: Arc<F>,
    /// A clone of the initial provider kept solely for delegating the
    /// **synchronous** capability methods (which never touch credentials), so
    /// wrapping a provider never changes how requests are shaped. The async
    /// `chat`/`chat_stream` paths still go through the refreshable `inner`.
    template: P,
    model: String,
    provider: &'static str,
    thinking: Option<ThinkingConfig>,
}

impl<P: Clone, F> Clone for RefreshingProvider<P, F> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            refresh: Arc::clone(&self.refresh),
            template: self.template.clone(),
            model: self.model.clone(),
            provider: self.provider,
            thinking: self.thinking.clone(),
        }
    }
}

impl<P, F, Fut> RefreshingProvider<P, F>
where
    P: LlmProvider + Clone + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<P>> + Send + 'static,
{
    /// Build a wrapper from an initial provider and a refresh callback.
    ///
    /// The refresh callback is invoked each time the inner provider emits a
    /// 401 response. It must be idempotent and safe to call concurrently.
    /// The callback should return a fully-built provider ready to use;
    /// typically it reads fresh credentials from its auth store and calls
    /// the inner provider's constructor.
    #[must_use]
    pub fn new(inner: P, refresh: F) -> Self {
        let model = inner.model().to_string();
        let provider = inner.provider();
        let thinking = inner.configured_thinking().cloned();
        let template = inner.clone();
        Self {
            inner: Arc::new(Mutex::new(inner)),
            refresh: Arc::new(refresh),
            template,
            model,
            provider,
            thinking,
        }
    }

    async fn snapshot(&self) -> P {
        self.inner.lock().await.clone()
    }

    async fn run_refresh(&self) -> Result<()> {
        let fresh = (self.refresh)().await?;
        *self.inner.lock().await = fresh;
        Ok(())
    }
}

/// Classify a provider error message as a 401 / unauthorized condition.
///
/// Returns `true` when the message looks like the provider rejected the
/// request because of missing, invalid, or expired credentials. Detection is
/// case-insensitive and matches the error-body shapes emitted by the
/// first-party providers in this crate. Hosts that implement their own retry
/// logic on top of [`LlmProvider`] can gate on the same helper.
#[must_use]
pub fn is_unauthorized_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains(" 401")
        || lower.contains("status=401")
        || lower.contains("unauthorized")
        || lower.contains("authentication")
        || lower.contains("token_expired")
        || lower.contains("invalid api key")
        || lower.contains("invalid_api_key")
}

#[async_trait]
impl<P, F, Fut> LlmProvider for RefreshingProvider<P, F>
where
    P: LlmProvider + Clone + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<P>> + Send + 'static,
{
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let outcome = self.snapshot().await.chat(request.clone()).await?;
        if let ChatOutcome::InvalidRequest(message) = &outcome
            && is_unauthorized_error(message)
        {
            match self.run_refresh().await {
                Ok(()) => return self.snapshot().await.chat(request).await,
                Err(error) => {
                    log::warn!("RefreshingProvider refresh after 401 failed: {error:#}");
                }
            }
        }
        Ok(outcome)
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        let this = self.clone();
        Box::pin(async_stream::stream! {
            let mut refreshed = false;
            'attempts: loop {
                let provider = this.snapshot().await;
                let mut stream = provider.chat_stream(request.clone());
                let mut saw_output = false;

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(StreamDelta::Error { message, kind })
                            if !saw_output
                                && !refreshed
                                && is_unauthorized_error(&message) =>
                        {
                            match this.run_refresh().await {
                                Ok(()) => {
                                    refreshed = true;
                                    continue 'attempts;
                                }
                                Err(error) => {
                                    log::warn!(
                                        "RefreshingProvider refresh after streaming 401 failed: {error:#}"
                                    );
                                    yield Ok(StreamDelta::Error { message, kind });
                                    return;
                                }
                            }
                        }
                        Ok(delta) => {
                            if matches!(
                                delta,
                                StreamDelta::TextDelta { .. }
                                    | StreamDelta::ThinkingDelta { .. }
                                    | StreamDelta::ToolUseStart { .. }
                                    | StreamDelta::ToolInputDelta { .. }
                                    | StreamDelta::SignatureDelta { .. }
                                    | StreamDelta::RedactedThinking { .. }
                            ) {
                                saw_output = true;
                            }
                            let done = matches!(delta, StreamDelta::Done { .. });
                            yield Ok(delta);
                            if done {
                                return;
                            }
                        }
                        Err(error)
                            if !saw_output
                                && !refreshed
                                && is_unauthorized_error(&error.to_string()) =>
                        {
                            match this.run_refresh().await {
                                Ok(()) => {
                                    refreshed = true;
                                    continue 'attempts;
                                }
                                Err(refresh_error) => {
                                    log::warn!(
                                        "RefreshingProvider refresh after stream failure failed: {refresh_error:#}"
                                    );
                                    yield Err(error);
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                }
                return;
            }
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        self.provider
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }

    // Delegate capability shaping to the wrapped provider so that wrapping
    // never silently changes request shaping (e.g. losing Vertex's max-token
    // clamp or a provider's adaptive-thinking validation). These methods are
    // synchronous and credential-independent, so they go through the captured
    // `template` rather than locking the async-refreshable `inner`.

    fn capabilities(&self) -> Option<&'static ModelCapabilities> {
        self.template.capabilities()
    }

    fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
        self.template.validate_thinking_config(thinking)
    }

    fn default_max_tokens(&self) -> u32 {
        self.template.default_max_tokens()
    }

    fn structured_output_support(&self) -> StructuredOutputSupport {
        self.template.structured_output_support()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agent_sdk_foundation::llm::{ChatResponse, ContentBlock, StopReason, Usage};
    use anyhow::Context;

    use crate::streaming::StreamErrorKind;

    #[derive(Clone)]
    enum MockStreamItem {
        Ok(StreamDelta),
        Err(String),
    }

    #[derive(Clone)]
    struct MockProvider {
        model: String,
        provider_name: &'static str,
        outcomes: Arc<StdMutex<VecDeque<ChatOutcome>>>,
        stream_batches: Arc<StdMutex<VecDeque<Vec<MockStreamItem>>>>,
        chat_calls: Arc<AtomicUsize>,
        stream_calls: Arc<AtomicUsize>,
    }

    impl MockProvider {
        fn new() -> Self {
            Self {
                model: "mock-model".to_string(),
                provider_name: "mock",
                outcomes: Arc::new(StdMutex::new(VecDeque::new())),
                stream_batches: Arc::new(StdMutex::new(VecDeque::new())),
                chat_calls: Arc::new(AtomicUsize::new(0)),
                stream_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn queue_chat(&self, outcome: ChatOutcome) -> Result<()> {
            self.outcomes
                .lock()
                .ok()
                .context("outcomes lock poisoned")?
                .push_back(outcome);
            Ok(())
        }

        fn queue_stream(&self, batch: Vec<MockStreamItem>) -> Result<()> {
            self.stream_batches
                .lock()
                .ok()
                .context("stream_batches lock poisoned")?
                .push_back(batch);
            Ok(())
        }

        fn chat_call_count(&self) -> usize {
            self.chat_calls.load(Ordering::SeqCst)
        }

        fn stream_call_count(&self) -> usize {
            self.stream_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            let mut queue = self
                .outcomes
                .lock()
                .ok()
                .context("outcomes lock poisoned")?;
            queue.pop_front().context("MockProvider: no queued outcome")
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            let batch: Vec<MockStreamItem> = self
                .stream_batches
                .lock()
                .ok()
                .and_then(|mut q| q.pop_front())
                .unwrap_or_else(|| vec![MockStreamItem::Err("no queued stream batch".into())]);
            Box::pin(async_stream::stream! {
                for item in batch {
                    match item {
                        MockStreamItem::Ok(delta) => yield Ok(delta),
                        MockStreamItem::Err(msg) => {
                            yield Err(anyhow::anyhow!(msg));
                            return;
                        }
                    }
                }
            })
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn provider(&self) -> &'static str {
            self.provider_name
        }

        // Sentinel capability overrides used to prove the wrapper delegates
        // rather than falling back to trait defaults.
        fn default_max_tokens(&self) -> u32 {
            32_000
        }

        fn structured_output_support(&self) -> StructuredOutputSupport {
            StructuredOutputSupport::Native
        }

        fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
            if thinking.is_some() {
                Err(anyhow::anyhow!("mock rejects thinking"))
            } else {
                Ok(())
            }
        }
    }

    fn success_response() -> ChatResponse {
        ChatResponse {
            id: "msg_test".to_string(),
            content: vec![ContentBlock::Text {
                text: "ok".to_string(),
            }],
            model: "mock-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }
    }

    fn empty_request() -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: Vec::new(),
            tools: None,
            max_tokens: 100,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        }
    }

    type BoxedFut = std::pin::Pin<Box<dyn Future<Output = Result<MockProvider>> + Send>>;
    type RefreshFn = Box<dyn Fn() -> BoxedFut + Send + Sync + 'static>;
    type Wrapped = RefreshingProvider<MockProvider, RefreshFn>;

    fn wrap_success(mock: &MockProvider, counter: &Arc<AtomicUsize>) -> Wrapped {
        let counter = Arc::clone(counter);
        let template = mock.clone();
        let cb: RefreshFn = Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            let provider = template.clone();
            Box::pin(async move { Ok(provider) })
        });
        RefreshingProvider::new(mock.clone(), cb)
    }

    fn wrap_failure(
        mock: &MockProvider,
        counter: &Arc<AtomicUsize>,
        error: &'static str,
    ) -> Wrapped {
        let counter = Arc::clone(counter);
        let cb: RefreshFn = Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Err(anyhow::anyhow!(error)) })
        });
        RefreshingProvider::new(mock.clone(), cb)
    }

    // Test 0: capability delegation (finding #12)
    #[test]
    fn wrapper_delegates_capability_overrides_to_inner() {
        let mock = MockProvider::new();
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        // Without delegation these would return trait defaults (4096 / Native
        // is coincidental / Ok), masking per-provider clamps and validation.
        assert_eq!(wrapped.default_max_tokens(), 32_000);
        assert_eq!(
            wrapped.structured_output_support(),
            StructuredOutputSupport::Native
        );
        assert!(
            wrapped
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_err()
        );
        assert!(wrapped.validate_thinking_config(None).is_ok());
    }

    // Test 1
    #[test]
    fn is_unauthorized_error_matches_expected_strings() {
        assert!(is_unauthorized_error("HTTP 401"));
        assert!(is_unauthorized_error("status=401 Unauthorized"));
        assert!(is_unauthorized_error("Invalid API key"));
        assert!(is_unauthorized_error("invalid_api_key"));
        assert!(is_unauthorized_error("token_expired"));
        assert!(is_unauthorized_error("Authentication failed"));
        assert!(is_unauthorized_error("UNAUTHORIZED"));

        assert!(!is_unauthorized_error("rate limited"));
        assert!(!is_unauthorized_error("network error"));
        assert!(!is_unauthorized_error(""));
        assert!(!is_unauthorized_error("internal server error"));
    }

    // Test 2
    #[tokio::test]
    async fn chat_successful_pass_through_does_not_refresh() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_chat(ChatOutcome::Success(success_response()))?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let outcome = wrapped.chat(empty_request()).await?;
        assert!(matches!(outcome, ChatOutcome::Success(_)));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 0);
        assert_eq!(mock.chat_call_count(), 1);
        Ok(())
    }

    // Test 3
    #[tokio::test]
    async fn chat_401_triggers_refresh_and_retries() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_chat(ChatOutcome::InvalidRequest("401 Unauthorized".into()))?;
        mock.queue_chat(ChatOutcome::Success(success_response()))?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let outcome = wrapped.chat(empty_request()).await?;
        assert!(matches!(outcome, ChatOutcome::Success(_)));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 1);
        assert_eq!(mock.chat_call_count(), 2);
        Ok(())
    }

    // Test 4
    #[tokio::test]
    async fn chat_surfaces_original_401_when_refresh_fails() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_chat(ChatOutcome::InvalidRequest(
            "status=401 Unauthorized".into(),
        ))?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_failure(&mock, &refresh_count, "refresh callback failed");

        let outcome = wrapped.chat(empty_request()).await?;
        match outcome {
            ChatOutcome::InvalidRequest(msg) => assert!(
                msg.contains("401"),
                "expected original 401 message, got {msg}"
            ),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
        assert_eq!(refresh_count.load(Ordering::SeqCst), 1);
        assert_eq!(mock.chat_call_count(), 1);
        Ok(())
    }

    async fn drain(mut stream: StreamBox<'_>) -> Vec<Result<StreamDelta>> {
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item);
        }
        out
    }

    // Test 5
    #[tokio::test]
    async fn chat_stream_successful_pass_through() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_stream(vec![
            MockStreamItem::Ok(StreamDelta::TextDelta {
                delta: "hi".into(),
                block_index: 0,
            }),
            MockStreamItem::Ok(StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
            }),
        ])?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let deltas = drain(wrapped.chat_stream(empty_request())).await;
        assert_eq!(deltas.len(), 2);
        assert!(matches!(
            deltas[0].as_ref().ok(),
            Some(StreamDelta::TextDelta { delta, .. }) if delta == "hi"
        ));
        assert!(matches!(
            deltas[1].as_ref().ok(),
            Some(StreamDelta::Done { .. })
        ));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 0);
        assert_eq!(mock.stream_call_count(), 1);
        Ok(())
    }

    // Test 6
    #[tokio::test]
    async fn chat_stream_401_before_output_retries() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_stream(vec![MockStreamItem::Ok(StreamDelta::Error {
            message: "status=401 Unauthorized".into(),
            kind: StreamErrorKind::InvalidRequest,
        })])?;
        mock.queue_stream(vec![
            MockStreamItem::Ok(StreamDelta::TextDelta {
                delta: "retried".into(),
                block_index: 0,
            }),
            MockStreamItem::Ok(StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
            }),
        ])?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let deltas = drain(wrapped.chat_stream(empty_request())).await;
        // Consumer sees only the post-refresh stream.
        assert_eq!(deltas.len(), 2);
        assert!(matches!(
            deltas[0].as_ref().ok(),
            Some(StreamDelta::TextDelta { delta, .. }) if delta == "retried"
        ));
        assert!(matches!(
            deltas[1].as_ref().ok(),
            Some(StreamDelta::Done { .. })
        ));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 1);
        assert_eq!(mock.stream_call_count(), 2);
        Ok(())
    }

    // Test 7
    #[tokio::test]
    async fn chat_stream_401_after_output_does_not_retry() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_stream(vec![
            MockStreamItem::Ok(StreamDelta::TextDelta {
                delta: "partial".into(),
                block_index: 0,
            }),
            MockStreamItem::Ok(StreamDelta::Error {
                message: "401 Unauthorized".into(),
                kind: StreamErrorKind::InvalidRequest,
            }),
        ])?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let deltas = drain(wrapped.chat_stream(empty_request())).await;
        assert_eq!(deltas.len(), 2);
        assert!(matches!(
            deltas[0].as_ref().ok(),
            Some(StreamDelta::TextDelta { delta, .. }) if delta == "partial"
        ));
        assert!(matches!(
            deltas[1].as_ref().ok(),
            Some(StreamDelta::Error { message, .. }) if message.contains("401")
        ));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 0);
        assert_eq!(mock.stream_call_count(), 1);
        Ok(())
    }

    // Test 8
    #[tokio::test]
    async fn chat_stream_only_one_retry_per_call() -> Result<()> {
        let mock = MockProvider::new();
        mock.queue_stream(vec![MockStreamItem::Ok(StreamDelta::Error {
            message: "status=401 Unauthorized".into(),
            kind: StreamErrorKind::InvalidRequest,
        })])?;
        mock.queue_stream(vec![MockStreamItem::Ok(StreamDelta::Error {
            message: "still 401 Unauthorized".into(),
            kind: StreamErrorKind::InvalidRequest,
        })])?;

        let refresh_count = Arc::new(AtomicUsize::new(0));
        let wrapped = wrap_success(&mock, &refresh_count);

        let deltas = drain(wrapped.chat_stream(empty_request())).await;
        assert_eq!(deltas.len(), 1);
        assert!(matches!(
            deltas[0].as_ref().ok(),
            Some(StreamDelta::Error { message, .. }) if message == "still 401 Unauthorized"
        ));
        assert_eq!(refresh_count.load(Ordering::SeqCst), 1);
        assert_eq!(mock.stream_call_count(), 2);
        Ok(())
    }

    // Custom deterministic mock for the concurrent scenario: the first
    // two chat() calls both wait on a barrier so both concurrent tasks
    // observe a 401 on their initial call, then all subsequent calls
    // return Success. This avoids flakiness from scheduling order in a
    // shared FIFO queue.
    #[derive(Clone)]
    struct ConcurrentMock {
        model: String,
        provider_name: &'static str,
        total_calls: Arc<AtomicUsize>,
        initial_barrier: Arc<tokio::sync::Barrier>,
    }

    type CMFut = std::pin::Pin<Box<dyn Future<Output = Result<ConcurrentMock>> + Send>>;
    type CMRefresh = Box<dyn Fn() -> CMFut + Send + Sync + 'static>;

    #[async_trait]
    impl LlmProvider for ConcurrentMock {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            let call_index = self.total_calls.fetch_add(1, Ordering::SeqCst);
            if call_index < 2 {
                self.initial_barrier.wait().await;
                Ok(ChatOutcome::InvalidRequest("401 Unauthorized".into()))
            } else {
                Ok(ChatOutcome::Success(success_response()))
            }
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            Box::pin(async_stream::stream! {
                yield Err(anyhow::anyhow!("chat_stream not used in this test"));
            })
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn provider(&self) -> &'static str {
            self.provider_name
        }
    }

    // Test 9
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_concurrent_callers_share_refresh() -> Result<()> {
        let mock = ConcurrentMock {
            model: "mock-model".to_string(),
            provider_name: "mock",
            total_calls: Arc::new(AtomicUsize::new(0)),
            initial_barrier: Arc::new(tokio::sync::Barrier::new(2)),
        };
        let call_count = Arc::clone(&mock.total_calls);
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_counter = Arc::clone(&refresh_count);
        let template = mock.clone();

        let cb: CMRefresh = Box::new(move || {
            refresh_counter.fetch_add(1, Ordering::SeqCst);
            let provider = template.clone();
            Box::pin(async move { Ok(provider) })
        });
        let wrapped = RefreshingProvider::new(mock, cb);

        let a = wrapped.clone();
        let b = wrapped.clone();
        let task_a = tokio::spawn(async move { a.chat(empty_request()).await });
        let task_b = tokio::spawn(async move { b.chat(empty_request()).await });

        let outcome_a = task_a.await.context("task_a join")??;
        let outcome_b = task_b.await.context("task_b join")??;

        assert!(matches!(outcome_a, ChatOutcome::Success(_)));
        assert!(matches!(outcome_b, ChatOutcome::Success(_)));
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
        let refreshes = refresh_count.load(Ordering::SeqCst);
        assert!(
            refreshes <= 2,
            "expected at most 2 refresh calls (one per caller), got {refreshes}"
        );
        Ok(())
    }
}
