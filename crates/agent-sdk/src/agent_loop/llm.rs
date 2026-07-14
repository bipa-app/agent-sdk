use super::helpers::{calculate_backoff_delay, send_event};
use super::types::{
    CONNECTIVITY_PROBE_MIN_DELAY, LLM_CALL_TOTAL_TIMEOUT, LLM_STREAM_INACTIVITY_TIMEOUT,
    LlmEventContext, LlmOutcome, LlmStreamIds, MAX_REACHABLE_CONNECTIVITY_FAILURES, StreamError,
};
use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, RequestDecision, ResponseDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamAccumulator, StreamDelta, Usage,
};
use crate::types::{AgentConfig, AgentError};
use futures::StreamExt;
use log::{error, warn};
use std::sync::Arc;
use tokio::time::sleep;

/// Outcome of the [`AgentHooks::pre_llm_request`] input guardrail applied
/// immediately before the provider chat call.
pub(super) enum PreLlmGuardrail {
    /// Proceed with this (possibly hook-substituted) request. Boxed because
    /// [`ChatRequest`] is large relative to the `Blocked` string.
    Proceed(Box<ChatRequest>),
    /// The hook refused the call; the string explains why.
    Blocked(String),
}

/// Run the [`AgentHooks::pre_llm_request`] input guardrail.
///
/// [`DefaultHooks`](crate::hooks::DefaultHooks) returns
/// [`RequestDecision::Proceed`] unchanged, so the default path is a no-op
/// and existing runs are unaffected.
pub(super) async fn apply_pre_llm_request<H>(
    hooks: &Arc<H>,
    request: ChatRequest,
) -> PreLlmGuardrail
where
    H: AgentHooks,
{
    match hooks.pre_llm_request(&request).await {
        RequestDecision::Modify(modified) => PreLlmGuardrail::Proceed(modified),
        RequestDecision::Block(reason) => PreLlmGuardrail::Blocked(reason),
        // `RequestDecision::Proceed` and any future `#[non_exhaustive]`
        // variant proceed unchanged — fail-open matches the historical
        // no-hook path.
        _ => PreLlmGuardrail::Proceed(Box::new(request)),
    }
}

/// Outcome of the [`AgentHooks::on_llm_response`] output guardrail applied
/// after the provider responds but before the response is persisted.
pub(super) enum PostLlmGuardrail {
    /// Accept the response unchanged.
    Accept,
    /// Reject the response; the string explains why.
    Blocked(String),
    /// Reject the response and steer a retry by feeding the string back to
    /// the model on the next turn.
    RetryWithFeedback(String),
}

/// Run the [`AgentHooks::on_llm_response`] output guardrail.
///
/// [`DefaultHooks`](crate::hooks::DefaultHooks) returns
/// [`ResponseDecision::Accept`], so the default path is a no-op.
pub(super) async fn apply_on_llm_response<H>(
    hooks: &Arc<H>,
    response: &ChatResponse,
) -> PostLlmGuardrail
where
    H: AgentHooks,
{
    match hooks.on_llm_response(response).await {
        ResponseDecision::Block(reason) => PostLlmGuardrail::Blocked(reason),
        ResponseDecision::RetryWithFeedback(feedback) => {
            PostLlmGuardrail::RetryWithFeedback(feedback)
        }
        // `ResponseDecision::Accept` and any future `#[non_exhaustive]`
        // variant accept the response unchanged.
        _ => PostLlmGuardrail::Accept,
    }
}

/// Per-call `OTel` observation hooks for the LLM span.
///
/// Bundles the live `chat <model>` span and the supporting metadata
/// the streaming / retry loops need to record events on it. Lives
/// behind `cfg(feature = "otel")` so the `agent-sdk` default build
/// pulls in zero observability code from these paths.
#[cfg(feature = "otel")]
pub(super) struct LlmSpanObserver<'a> {
    pub(super) span: &'a mut opentelemetry::global::BoxedSpan,
    pub(super) provider_name: &'static str,
    /// `gen_ai.request.model` — used as a metric label on the
    /// streaming TTFC / TPOC histograms so dashboards can split the
    /// distribution by the model the SDK *asked* for. Borrowed from
    /// the caller-owned `String` that lives for the duration of the
    /// LLM call so we avoid an extra allocation per chunk.
    pub(super) request_model: &'a str,
}

#[cfg(feature = "otel")]
impl LlmSpanObserver<'_> {
    fn record_first_chunk(&mut self, turn: usize, streaming: bool, ttfc_secs: f64) {
        use crate::observability::{attrs, metrics, spans};
        use opentelemetry::KeyValue;
        use opentelemetry::trace::Span;

        spans::add_event(
            self.span,
            "llm.stream.first_chunk",
            vec![
                attrs::kv_i64(attrs::SDK_TURN_NUMBER, i64::try_from(turn).unwrap_or(0)),
                attrs::kv_bool(attrs::SDK_LLM_STREAMING, streaming),
            ],
        );
        // Stamp TTFC on the span as f64 seconds. Per the GenAI
        // semconv, this attribute MUST be present on streaming runs
        // and absent on non-streaming runs; the latter is enforced
        // because `record_first_chunk` is only ever called from the
        // streaming path.
        self.span.set_attribute(KeyValue::new(
            attrs::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
            ttfc_secs,
        ));

        let metrics_handle = metrics::Metrics::global();
        metrics_handle.time_to_first_chunk.record(
            ttfc_secs,
            &[
                KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "chat"),
                KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, self.provider_name),
                KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, self.request_model.to_string()),
            ],
        );
    }

    /// Record `gen_ai.client.operation.time_per_output_chunk` for one
    /// post-first chunk. Pure metric — no span event, no span
    /// attribute.
    fn record_subsequent_chunk(&self, tpoc_secs: f64) {
        use crate::observability::{attrs, metrics};
        use opentelemetry::KeyValue;

        let metrics_handle = metrics::Metrics::global();
        metrics_handle.time_per_output_chunk.record(
            tpoc_secs,
            &[
                KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "chat"),
                KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, self.provider_name),
                KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, self.request_model.to_string()),
            ],
        );
    }

    fn record_completed(&mut self, delta_count: u64, duration_ms: u64) {
        use crate::observability::{attrs, spans};
        spans::add_event(
            self.span,
            "llm.stream.completed",
            vec![
                attrs::kv_i64(
                    attrs::SDK_LLM_STREAM_DELTA_COUNT,
                    i64::try_from(delta_count).unwrap_or(i64::MAX),
                ),
                attrs::kv_i64(
                    attrs::SDK_LLM_STREAM_DURATION_MS,
                    i64::try_from(duration_ms).unwrap_or(i64::MAX),
                ),
            ],
        );
    }

    fn record_dropped(&mut self, reason: &'static str, delta_count: u64, error_type: &str) {
        use crate::observability::{attrs, spans};
        spans::add_event(
            self.span,
            "llm.stream.dropped",
            vec![
                opentelemetry::KeyValue::new(attrs::SDK_LLM_STREAM_DROP_REASON, reason),
                attrs::kv_i64(
                    attrs::SDK_LLM_STREAM_DELTA_COUNT,
                    i64::try_from(delta_count).unwrap_or(i64::MAX),
                ),
                opentelemetry::KeyValue::new(attrs::ERROR_TYPE, error_type.to_string()),
            ],
        );
    }

    fn record_retry(&mut self, attempt: u32, max_attempts: u32, delay_ms: u64, error_type: &str) {
        use crate::observability::{attrs, metrics, spans};
        spans::add_event(
            self.span,
            "llm.retry",
            vec![
                opentelemetry::KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, self.provider_name),
                attrs::kv_i64(attrs::SDK_LLM_RETRY_ATTEMPT, i64::from(attempt)),
                attrs::kv_i64(attrs::SDK_LLM_RETRY_MAX_ATTEMPTS, i64::from(max_attempts)),
                attrs::kv_i64(
                    attrs::SDK_LLM_RETRY_DELAY_MS,
                    i64::try_from(delay_ms).unwrap_or(i64::MAX),
                ),
                opentelemetry::KeyValue::new(attrs::ERROR_TYPE, error_type.to_string()),
            ],
        );

        let metrics_handle = metrics::Metrics::global();
        metrics_handle.llm_retries.add(
            1,
            &[
                opentelemetry::KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, self.provider_name),
                opentelemetry::KeyValue::new(attrs::ERROR_TYPE, error_type.to_string()),
            ],
        );
    }
}

/// Result of racing a single non-streaming `provider.chat` call against
/// the run's cancel token.
enum ProviderCall {
    Outcome(ChatOutcome),
    Cancelled,
    Error(AgentError),
}

/// Race one non-streaming provider call against cancel.
///
/// Without this race a cancel issued while the model is still composing
/// the full response is ignored until the call returns. `biased` honors
/// a token flipped before the call even starts.
async fn chat_or_cancel<P, H>(
    provider: &Arc<P>,
    request: &ChatRequest,
    event_ctx: &LlmEventContext<'_, H>,
) -> ProviderCall
where
    P: LlmProvider,
    H: AgentHooks,
{
    tokio::select! {
        biased;
        () = event_ctx.cancel_token.cancelled() => {
            log::info!("LLM call cancelled (turn={})", event_ctx.turn);
            ProviderCall::Cancelled
        }
        res = tokio::time::timeout(LLM_CALL_TOTAL_TIMEOUT, provider.chat(request.clone())) => match res {
            Ok(Ok(outcome)) => ProviderCall::Outcome(outcome),
            Ok(Err(e)) => ProviderCall::Error(AgentError::new(format!("LLM error: {e}"), false)),
            Err(_elapsed) => {
                // The non-streaming call stalled past the overall deadline.
                // Surface it as a retryable server error so the retry loop
                // re-attempts instead of hanging the turn indefinitely.
                warn!(
                    "LLM call exceeded inactivity deadline (turn={}), treating as server error",
                    event_ctx.turn
                );
                ProviderCall::Outcome(ChatOutcome::ServerError(
                    "LLM call timed out (no response within deadline)".to_string(),
                ))
            }
        },
    }
}

/// Call the LLM with retry logic for rate limits and server errors.
///
/// Returns the result and the number of retries that occurred.
pub(super) async fn call_llm_with_retry<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    event_ctx: &LlmEventContext<'_, H>,
    #[cfg(feature = "otel")] mut span_observer: Option<LlmSpanObserver<'_>>,
) -> (LlmOutcome, u32)
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let outcome = match chat_or_cancel(provider, &request, event_ctx).await {
            ProviderCall::Outcome(outcome) => outcome,
            ProviderCall::Cancelled => return (LlmOutcome::Cancelled(ZERO_USAGE), attempt),
            ProviderCall::Error(error) => return (LlmOutcome::Error(error), attempt),
        };

        // `retry_reason` is the human-readable label shown on the
        // `AutoRetryStart` event; `failure_message` is the error the
        // run surfaces once the retry budget is exhausted. They differ
        // per variant, so both are resolved here and threaded through.
        // `override_delay` carries a provider-supplied `Retry-After` hint
        // (rate limits only) that supersedes the exponential backoff.
        let (kind, retry_reason, failure_message, override_delay) = match outcome {
            ChatOutcome::Success(response) => {
                if attempt > 0 {
                    send_auto_retry_end_event(event_ctx, attempt, true, None).await;
                }
                return (LlmOutcome::Response(response), attempt);
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!("Invalid request to LLM: {msg}");
                return (
                    LlmOutcome::Error(AgentError::new(format!("Invalid request: {msg}"), false)),
                    attempt,
                );
            }
            ChatOutcome::RateLimited(retry_after) => (
                "rate_limited",
                "Rate limited by LLM provider".to_string(),
                format!("Rate limited after {max_retries} retries"),
                retry_after,
            ),
            ChatOutcome::ServerError(msg) => (
                "server_error",
                msg.clone(),
                format!("Server error after {max_retries} retries: {msg}"),
                None,
            ),
            // `ChatOutcome` is `#[non_exhaustive]`; an unrecognized outcome is
            // handled like a server error (retry, then surface a clear failure).
            _ => (
                "server_error",
                "Unrecognized provider outcome".to_string(),
                format!("Unrecognized provider outcome after {max_retries} retries"),
                None,
            ),
        };

        attempt += 1;
        match handle_retry_backoff(RetryBackoff {
            event_ctx,
            config,
            attempt,
            event_attempt: attempt,
            settled_event_attempt: attempt.saturating_sub(1),
            max_retries,
            event_max_retries: max_retries,
            error_kind: kind,
            retry_reason,
            failure_message,
            override_delay,
            #[cfg(feature = "otel")]
            span_observer: span_observer.as_mut(),
            #[cfg(not(feature = "otel"))]
            _observer: std::marker::PhantomData,
        })
        .await
        {
            RetryStep::Retry => {}
            RetryStep::Cancelled => return (LlmOutcome::Cancelled(ZERO_USAGE), attempt),
            RetryStep::GiveUp(outcome) => return (outcome, attempt),
        }
    }
}

/// Whether the retry loop should retry, was cancelled, or has exhausted
/// its budget.
enum RetryStep {
    Retry,
    Cancelled,
    GiveUp(LlmOutcome),
}

fn fatal_stream_outcome(message: &str) -> LlmOutcome {
    error!("Streaming error (non-recoverable): {message}");
    LlmOutcome::Error(AgentError::new(
        format!("Streaming error: {message}"),
        false,
    ))
}

fn terminal_retry_outcome(
    step: RetryStep,
    retries: u32,
    usage: &Usage,
) -> Option<(LlmOutcome, u32)> {
    match step {
        RetryStep::Retry => None,
        RetryStep::Cancelled => Some((LlmOutcome::Cancelled(usage.clone()), retries)),
        RetryStep::GiveUp(outcome) => Some((outcome, retries)),
    }
}

const fn mark_retry_emitted(last_attempt: &mut u32, attempt: u32, emitted: bool) {
    if emitted {
        *last_attempt = attempt;
    }
}

fn renew_stream_ids(message_id: &mut String, thinking_id: &mut String) {
    *message_id = uuid::Uuid::new_v4().to_string();
    *thinking_id = uuid::Uuid::new_v4().to_string();
}

async fn finish_retry_events<H>(event_ctx: &LlmEventContext<'_, H>, last_attempt: u32)
where
    H: AgentHooks,
{
    if last_attempt > 0 {
        send_auto_retry_end_event(event_ctx, last_attempt, true, None).await;
    }
}

fn transient_retry_context(
    message: &str,
    max_retries: u32,
    waited_for_connectivity: bool,
) -> (String, u32) {
    let failure = format!("Streaming error after {max_retries} retries: {message}");
    let event_max = if waited_for_connectivity {
        u32::MAX
    } else {
        max_retries
    };
    (failure, event_max)
}

/// Terminal failure of a retry loop: settle the auto-retry envelope with
/// `success: false`, surface the `llm.error` event, and build the outcome the
/// run ends with. Shared by the exhausted transient budget and the
/// reachable-connectivity circuit breaker so both fail identically.
async fn give_up_retrying<H>(
    event_ctx: &LlmEventContext<'_, H>,
    settled_event_attempt: u32,
    failure_message: String,
    error_kind: &'static str,
) -> LlmOutcome
where
    H: AgentHooks,
{
    error!("LLM {error_kind} exhausted retries: {failure_message}");
    send_auto_retry_end_event(
        event_ctx,
        settled_event_attempt,
        false,
        Some(failure_message.clone()),
    )
    .await;
    if let Err(error) = send_llm_error_event(event_ctx, &failure_message).await {
        return LlmOutcome::Error(error);
    }
    LlmOutcome::Error(AgentError::new(failure_message, true))
}

struct RetryBackoff<'a, 'o, H> {
    event_ctx: &'a LlmEventContext<'a, H>,
    config: &'a AgentConfig,
    attempt: u32,
    event_attempt: u32,
    settled_event_attempt: u32,
    max_retries: u32,
    event_max_retries: u32,
    /// Stable `error.type` label, e.g. `rate_limited` / `server_error`.
    error_kind: &'static str,
    /// Human-readable reason shown on the `AutoRetryStart` event.
    retry_reason: String,
    /// Error surfaced once the retry budget is exhausted.
    failure_message: String,
    /// Provider-supplied retry delay (from a 429 `Retry-After`) that overrides
    /// the computed exponential backoff. Clamped to `config.retry.max_delay_ms`.
    override_delay: Option<std::time::Duration>,
    #[cfg(feature = "otel")]
    span_observer: Option<&'a mut LlmSpanObserver<'o>>,
    #[cfg(not(feature = "otel"))]
    _observer: std::marker::PhantomData<&'o ()>,
}

/// Shared backoff handling for the retryable `ChatOutcome` variants
/// (rate-limit, server error). Emits the auto-retry telemetry, sleeps
/// the computed backoff raced against cancel, and reports whether the
/// caller should retry, stop for cancel, or give up.
async fn handle_retry_backoff<H>(params: RetryBackoff<'_, '_, H>) -> RetryStep
where
    H: AgentHooks,
{
    let RetryBackoff {
        event_ctx,
        config,
        attempt,
        event_attempt,
        settled_event_attempt,
        max_retries,
        event_max_retries,
        error_kind,
        retry_reason,
        failure_message,
        override_delay,
        #[cfg(feature = "otel")]
        span_observer,
        #[cfg(not(feature = "otel"))]
            _observer: _,
    } = params;

    if attempt > max_retries {
        return RetryStep::GiveUp(
            give_up_retrying(
                event_ctx,
                settled_event_attempt,
                failure_message,
                error_kind,
            )
            .await,
        );
    }

    // A provider `Retry-After` hint wins over the exponential backoff, but is
    // clamped to the configured ceiling so a hostile/oversized header cannot
    // stall the turn. The attempt still counts against `max_retries`.
    let delay = override_delay.map_or_else(
        || calculate_backoff_delay(attempt, &config.retry),
        |hint| clamp_to_max_delay(hint, config.retry.max_delay_ms),
    );
    let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
    warn!(
        "LLM {error_kind}, retrying (retry={event_attempt}, class_attempt={attempt}, delay_ms={delay_ms})"
    );
    #[cfg(feature = "otel")]
    if let Some(observer) = span_observer {
        observer.record_retry(event_attempt, event_max_retries, delay_ms, error_kind);
    }
    send_auto_retry_start_event(
        event_ctx,
        event_attempt,
        event_max_retries,
        delay_ms,
        &retry_reason,
    )
    .await;
    if sleep_or_cancel(delay, event_ctx.cancel_token)
        .await
        .is_break()
    {
        return RetryStep::Cancelled;
    }
    RetryStep::Retry
}

/// Clamp a provider-supplied retry delay to the configured maximum.
fn clamp_to_max_delay(delay: std::time::Duration, max_delay_ms: u64) -> std::time::Duration {
    let ms = u64::try_from(delay.as_millis())
        .unwrap_or(u64::MAX)
        .min(max_delay_ms);
    std::time::Duration::from_millis(ms)
}

/// Sleep for `delay`, but return early if the cancel token fires first.
///
/// Returns [`ControlFlow::Break`] when the cancel won the race so the
/// caller can short-circuit the retry loop into a cancellation; returns
/// [`ControlFlow::Continue`] when the full backoff elapsed.
async fn sleep_or_cancel(
    delay: std::time::Duration,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> std::ops::ControlFlow<()> {
    tokio::select! {
        biased;
        () = cancel_token.cancelled() => std::ops::ControlFlow::Break(()),
        () = sleep(delay) => std::ops::ControlFlow::Continue(()),
    }
}

struct StreamingRetryState {
    total: u32,
    transient: u32,
    /// Consecutive connectivity-class failures whose surrounding probes kept
    /// reporting the provider reachable. Any unreachable probe resets it —
    /// direct evidence of a genuine outage forgives earlier deaths.
    reachable_connectivity_deaths: u32,
    /// Whether the previous recorded failure was connectivity-class. A streak
    /// emits one `AutoRetryStart` envelope, on its first failure.
    in_connectivity_streak: bool,
    /// Whether any connectivity failure occurred during this call; later
    /// transient retry events then advertise the indefinite-wait sentinel so
    /// their attempt number can exceed the transient budget.
    had_connectivity: bool,
    last_emitted: u32,
    /// Usage billed by attempts that died before completing.
    abandoned_usage: Usage,
}

impl StreamingRetryState {
    const fn new() -> Self {
        Self {
            total: 0,
            transient: 0,
            reachable_connectivity_deaths: 0,
            in_connectivity_streak: false,
            had_connectivity: false,
            last_emitted: 0,
            abandoned_usage: ZERO_USAGE,
        }
    }

    /// Record a connectivity-class failure. Returns `true` when it opens a
    /// new streak, i.e. when the caller must emit the streak's
    /// `AutoRetryStart`.
    const fn record_connectivity_failure(&mut self, usage: &Usage) -> bool {
        add_usage(&mut self.abandoned_usage, usage);
        self.total = self.total.saturating_add(1);
        self.reachable_connectivity_deaths = self.reachable_connectivity_deaths.saturating_add(1);
        self.had_connectivity = true;
        let starts_streak = !self.in_connectivity_streak;
        self.in_connectivity_streak = true;
        starts_streak
    }

    const fn record_transient_failure(&mut self, usage: &Usage, max_retries: u32) -> bool {
        add_usage(&mut self.abandoned_usage, usage);
        self.transient = self.transient.saturating_add(1);
        self.total = self.total.saturating_add(1);
        self.in_connectivity_streak = false;
        self.transient <= max_retries
    }
}

/// Delay before connectivity probe `probe_round` (1-based): the configured
/// exponential backoff with a floor, so a zero-delay retry config cannot turn
/// the offline wait into a hot loop.
fn connectivity_probe_delay(probe_round: u32, config: &AgentConfig) -> std::time::Duration {
    calculate_backoff_delay(probe_round, &config.retry).max(CONNECTIVITY_PROBE_MIN_DELAY)
}

/// Park until the provider probe reports reachable again, sleeping the
/// per-round backoff between probes and racing every wait against
/// cancellation. Probes are free — no billable request is dispatched until
/// this returns.
///
/// An unreachable probe is direct evidence of a genuine outage, so it resets
/// the reachable-death circuit breaker: offline time, however long, never
/// fails the turn.
async fn wait_for_connectivity<P, H>(
    provider: &Arc<P>,
    config: &AgentConfig,
    event_ctx: &LlmEventContext<'_, H>,
    retry: &mut StreamingRetryState,
) -> std::ops::ControlFlow<()>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let mut probe_round: u32 = 0;
    loop {
        probe_round = probe_round.saturating_add(1);
        let delay = connectivity_probe_delay(probe_round, config);
        if sleep_or_cancel(delay, event_ctx.cancel_token)
            .await
            .is_break()
        {
            return std::ops::ControlFlow::Break(());
        }
        let reachable = tokio::select! {
            biased;
            () = event_ctx.cancel_token.cancelled() => return std::ops::ControlFlow::Break(()),
            reachable = provider.probe_connectivity() => reachable,
        };
        if reachable {
            return std::ops::ControlFlow::Continue(());
        }
        retry.reachable_connectivity_deaths = 0;
    }
}

/// Handle one connectivity-class stream failure: trip the reachable-death
/// circuit breaker, open the streak's retry envelope when this failure
/// starts one, and park until the provider answers probes again.
///
/// Returns the terminal outcome when the wait ended the call (circuit
/// breaker or cancellation); `None` means connectivity returned and the
/// caller should re-dispatch under fresh stream ids.
async fn handle_connectivity_loss<P, H>(
    provider: &Arc<P>,
    config: &AgentConfig,
    event_ctx: &LlmEventContext<'_, H>,
    retry: &mut StreamingRetryState,
    message: &str,
    usage: &Usage,
    #[cfg(feature = "otel")] span_observer: Option<&mut LlmSpanObserver<'_>>,
) -> Option<(LlmOutcome, u32)>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let starts_streak = retry.record_connectivity_failure(usage);
    if retry.reachable_connectivity_deaths > MAX_REACHABLE_CONNECTIVITY_FAILURES {
        // The endpoint answers probes yet every dispatched stream dies in
        // transit: a broken path, not an outage. Billing a fresh attempt per
        // backoff forever is the one thing the offline wait must not do, so
        // this fails like an exhausted transient budget.
        let failure = format!(
            "Streaming connection died {} consecutive times while the provider \
             stayed reachable: {message}",
            retry.reachable_connectivity_deaths
        );
        let outcome =
            give_up_retrying(event_ctx, retry.last_emitted, failure, "connectivity").await;
        return Some((outcome, retry.total));
    }
    let delay_ms =
        u64::try_from(connectivity_probe_delay(1, config).as_millis()).unwrap_or(u64::MAX);
    warn!(
        "LLM connectivity loss, waiting for reachability (failure={}, reachable_deaths={}): {message}",
        retry.total, retry.reachable_connectivity_deaths
    );
    #[cfg(feature = "otel")]
    if let Some(observer) = span_observer {
        observer.record_retry(retry.total, u32::MAX, delay_ms, "connectivity");
    }
    if starts_streak {
        retry.last_emitted = retry.total;
        send_auto_retry_start_event(event_ctx, retry.total, u32::MAX, delay_ms, message).await;
    }
    if wait_for_connectivity(provider, config, event_ctx, retry)
        .await
        .is_break()
    {
        return Some((
            LlmOutcome::Cancelled(retry.abandoned_usage.clone()),
            retry.total,
        ));
    }
    None
}

/// Call the LLM with streaming, emitting deltas as they arrive.
///
/// This function handles streaming responses from the LLM, emitting `TextDelta`
/// and `Thinking` events in real-time as content arrives. It includes retry logic
/// for recoverable errors (rate limits, server errors).
///
/// Returns the result and the number of retries that occurred.
pub(super) async fn call_llm_streaming<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    event_ctx: &LlmEventContext<'_, H>,
    message_id: &mut String,
    thinking_id: &mut String,
    #[cfg(feature = "otel")] mut span_observer: Option<LlmSpanObserver<'_>>,
) -> (LlmOutcome, u32)
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut retry = StreamingRetryState::new();

    loop {
        let stream_ids = LlmStreamIds {
            message_id: message_id.as_str(),
            thinking_id: thinking_id.as_str(),
        };
        let result = process_stream(
            provider,
            &request,
            event_ctx,
            stream_ids,
            #[cfg(feature = "otel")]
            span_observer.as_mut(),
        )
        .await;

        match result {
            Ok(mut response) => {
                finish_retry_events(event_ctx, retry.last_emitted).await;
                add_usage(&mut response.usage, &retry.abandoned_usage);
                return (LlmOutcome::Response(response), retry.total);
            }
            Err(StreamError::Connectivity { message, usage }) => {
                let terminal = handle_connectivity_loss(
                    provider,
                    config,
                    event_ctx,
                    &mut retry,
                    &message,
                    &usage,
                    #[cfg(feature = "otel")]
                    span_observer.as_mut(),
                )
                .await;
                if let Some(outcome) = terminal {
                    return outcome;
                }
                renew_stream_ids(message_id, thinking_id);
            }
            Err(StreamError::Recoverable {
                message,
                retry_after,
                usage,
            }) => {
                let emits_transient = retry.record_transient_failure(&usage, max_retries);
                mark_retry_emitted(&mut retry.last_emitted, retry.total, emits_transient);
                let (failure_message, event_max_retries) =
                    transient_retry_context(&message, max_retries, retry.had_connectivity);
                let step = handle_retry_backoff(RetryBackoff {
                    event_ctx,
                    config,
                    attempt: retry.transient,
                    event_attempt: retry.total,
                    settled_event_attempt: retry.last_emitted,
                    max_retries,
                    event_max_retries,
                    error_kind: "stream_error",
                    retry_reason: message,
                    failure_message,
                    override_delay: retry_after,
                    #[cfg(feature = "otel")]
                    span_observer: span_observer.as_mut(),
                    #[cfg(not(feature = "otel"))]
                    _observer: std::marker::PhantomData,
                })
                .await;
                if let Some(outcome) =
                    terminal_retry_outcome(step, retry.total, &retry.abandoned_usage)
                {
                    return outcome;
                }
                renew_stream_ids(message_id, thinking_id);
            }
            Err(StreamError::Fatal(message)) => {
                return (fatal_stream_outcome(&message), retry.total);
            }
            Err(StreamError::Cancelled(usage)) => {
                add_usage(&mut retry.abandoned_usage, &usage);
                return (LlmOutcome::Cancelled(retry.abandoned_usage), retry.total);
            }
        }
    }
}

/// Process a single streaming attempt and return the response or error.
async fn process_stream<P, H>(
    provider: &Arc<P>,
    request: &ChatRequest,
    event_ctx: &LlmEventContext<'_, H>,
    stream_ids: LlmStreamIds<'_>,
    #[cfg(feature = "otel")] mut span_observer: Option<&mut LlmSpanObserver<'_>>,
) -> Result<ChatResponse, StreamError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let mut stream = std::pin::pin!(provider.chat_stream(request.clone()));
    let mut accumulator = StreamAccumulator::new();
    let mut delta_count: u64 = 0;
    // `chunk_timing` advances once per *content* delta: the first content
    // delta emits TTFC and seeds the per-chunk clock; every subsequent
    // content delta records `now - last_chunk_at` to TPOC. Sentinel frames
    // (`Usage`, `Done`) are excluded so the distribution stays meaningful —
    // those are metadata, not output chunks.
    #[cfg(feature = "otel")]
    let mut chunk_timing = StreamChunkTiming::start();

    log::debug!("Starting to consume LLM stream");

    loop {
        // Race each frame against the run's cancel token so a cancel
        // mid-stream — or before the first frame — stops further model
        // events promptly instead of waiting for the provider to finish
        // the response. `biased` picks the cancel arm first, so an
        // already-cancelled token (cancel before first token) is
        // honored without consuming any delta, leaving the accumulator
        // empty and emitting no `tool_use`.
        let result = tokio::select! {
            biased;
            () = event_ctx.cancel_token.cancelled() => {
                log::info!(
                    "LLM stream cancelled (turn={}, delta_count={delta_count})",
                    event_ctx.turn
                );
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_dropped("cancelled", delta_count, "cancelled");
                }
                return Err(StreamError::Cancelled(accumulated_usage(&accumulator)));
            }
            next = tokio::time::timeout(LLM_STREAM_INACTIVITY_TIMEOUT, stream.next()) => match next {
                Ok(Some(result)) => result,
                Ok(None) => break,
                Err(_elapsed) => {
                    // No frame arrived within the inactivity window — the
                    // provider connection is stalled (half-open). Surface a
                    // recoverable error so the retry loop re-establishes the
                    // stream instead of hanging the turn forever.
                    warn!(
                        "LLM stream inactivity timeout (turn={}, delta_count={delta_count})",
                        event_ctx.turn
                    );
                    #[cfg(feature = "otel")]
                    if let Some(observer) = span_observer.as_mut() {
                        observer.record_dropped("inactivity_timeout", delta_count, "stream_error");
                    }
                    return Err(StreamError::Connectivity {
                        message: "stream inactivity timeout: no frame within deadline".to_string(),
                        usage: accumulated_usage(&accumulator),
                    });
                }
            },
        };

        if delta_count > 0 && delta_count.is_multiple_of(50) {
            log::debug!("Stream progress: delta_count={delta_count}");
        }

        let delta = match result {
            Ok(d) => d,
            Err(e) => {
                log::error!("Stream iteration error delta_count={delta_count} error={e}");
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_dropped("recoverable_error", delta_count, "stream_error");
                }
                return Err(StreamError::Recoverable {
                    message: format!("Stream error: {e}"),
                    retry_after: None,
                    usage: accumulated_usage(&accumulator),
                });
            }
        };

        delta_count += 1;
        accumulator.apply(&delta);

        // The first content delta records TTFC + seeds the per-chunk
        // clock; every later content delta records TPOC against the
        // prior tick. A single `Instant::now()` per content iteration
        // keeps the per-chunk overhead at one syscall plus the metric
        // label allocations.
        #[cfg(feature = "otel")]
        if is_content_delta(&delta) {
            chunk_timing.record_content_delta(span_observer.as_deref_mut(), event_ctx.turn);
        }

        if let Some(stream_err) = dispatch_stream_delta(
            &delta,
            event_ctx,
            stream_ids,
            delta_count,
            #[cfg(feature = "otel")]
            span_observer.as_deref_mut(),
        )
        .await
        {
            // A provider can report usage and only then fail. The tokens are
            // billed regardless, so the dying attempt hands its usage to the
            // retry loop instead of taking it down with the accumulator.
            return Err(stream_err.with_accumulated_usage(accumulated_usage(&accumulator)));
        }
    }

    log::debug!("Stream while loop exited normally at delta_count={delta_count}");

    #[cfg(feature = "otel")]
    chunk_timing.record_completed(span_observer, delta_count);

    Ok(finalize_stream_response(
        accumulator,
        provider.model(),
        delta_count,
    ))
}

/// Per-stream chunk-latency bookkeeping for the `OTel` `GenAI`
/// `time_to_first_chunk` / `time_per_output_chunk` histograms.
///
/// One `Instant::now()` is taken per content delta: the first records
/// TTFC and seeds the per-chunk clock; each later one records TPOC
/// against the previous tick. Extracted from [`process_stream`] to keep
/// it under the clippy line ceiling.
#[cfg(feature = "otel")]
struct StreamChunkTiming {
    stream_started_at: std::time::Instant,
    first_chunk_recorded: bool,
    last_chunk_at: Option<std::time::Instant>,
}

#[cfg(feature = "otel")]
impl StreamChunkTiming {
    fn start() -> Self {
        Self {
            stream_started_at: std::time::Instant::now(),
            first_chunk_recorded: false,
            last_chunk_at: None,
        }
    }

    /// Record TTFC on the first content delta and TPOC on each later
    /// one, advancing the per-chunk clock even when no observer is set.
    fn record_content_delta(
        &mut self,
        mut observer: Option<&mut LlmSpanObserver<'_>>,
        turn: usize,
    ) {
        let now = std::time::Instant::now();
        if !self.first_chunk_recorded {
            self.first_chunk_recorded = true;
            if let Some(observer) = observer.as_deref_mut() {
                let ttfc_secs = self.stream_started_at.elapsed().as_secs_f64();
                observer.record_first_chunk(turn, true, ttfc_secs);
            }
        } else if let Some(prev) = self.last_chunk_at
            && let Some(observer) = observer
        {
            let tpoc_secs = now.duration_since(prev).as_secs_f64();
            observer.record_subsequent_chunk(tpoc_secs);
        }
        self.last_chunk_at = Some(now);
    }

    /// Emit the `llm.stream.completed` span event with the total stream
    /// duration when an observer is present.
    fn record_completed(&self, observer: Option<&mut LlmSpanObserver<'_>>, delta_count: u64) {
        if let Some(observer) = observer {
            let duration_ms =
                u64::try_from(self.stream_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
            observer.record_completed(delta_count, duration_ms);
        }
    }
}

/// A zero-token [`Usage`], used as the identity when folding attempts together.
const ZERO_USAGE: Usage = Usage {
    input_tokens: 0,
    output_tokens: 0,
    cached_input_tokens: 0,
    cache_creation_input_tokens: 0,
};

/// Usage the stream has reported so far, or zero when it reported none yet.
fn accumulated_usage(accumulator: &StreamAccumulator) -> Usage {
    accumulator.usage().cloned().unwrap_or(ZERO_USAGE)
}

/// Fold `delta` into `total`. Saturating: token counters must never wrap.
const fn add_usage(total: &mut Usage, delta: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(delta.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(delta.output_tokens);
    total.cached_input_tokens = total
        .cached_input_tokens
        .saturating_add(delta.cached_input_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(delta.cache_creation_input_tokens);
}

/// Assemble the final [`ChatResponse`] from a fully-consumed stream
/// accumulator. Extracted from [`process_stream`] to keep it under the
/// clippy line ceiling.
fn finalize_stream_response(
    accumulator: StreamAccumulator,
    model: &str,
    delta_count: u64,
) -> ChatResponse {
    let usage = accumulated_usage(&accumulator);
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    log::debug!(
        "LLM stream completed successfully delta_count={delta_count} stop_reason={stop_reason:?} content_block_count={} input_tokens={} output_tokens={}",
        content_blocks.len(),
        usage.input_tokens,
        usage.output_tokens
    );

    ChatResponse {
        id: uuid::Uuid::new_v4().to_string(),
        content: content_blocks,
        model: model.to_string(),
        stop_reason,
        usage,
    }
}

/// Dispatch a single decoded delta: forward text/thinking content to
/// the event store, classify provider errors, and (when feature
/// enabled) record drop reasons on the LLM span observer.
async fn dispatch_stream_delta<H>(
    delta: &StreamDelta,
    event_ctx: &LlmEventContext<'_, H>,
    stream_ids: LlmStreamIds<'_>,
    delta_count: u64,
    #[cfg(feature = "otel")] mut span_observer: Option<&mut LlmSpanObserver<'_>>,
) -> Option<StreamError>
where
    H: AgentHooks,
{
    let LlmStreamIds {
        message_id,
        thinking_id,
    } = stream_ids;
    match delta {
        StreamDelta::TextDelta { delta, .. } => {
            if let Err(error) = send_event(
                event_ctx.event_store,
                event_ctx.thread_id,
                event_ctx.turn,
                event_ctx.hooks,
                event_ctx.authority,
                AgentEvent::text_delta(message_id, delta.clone()),
            )
            .await
            {
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_dropped(
                        "event_channel_send_failed",
                        delta_count,
                        "event_channel",
                    );
                }
                return Some(StreamError::Fatal(error.message));
            }
        }
        StreamDelta::ThinkingDelta { delta, .. } => {
            if let Err(error) = send_event(
                event_ctx.event_store,
                event_ctx.thread_id,
                event_ctx.turn,
                event_ctx.hooks,
                event_ctx.authority,
                AgentEvent::thinking_delta(thinking_id, delta.clone()),
            )
            .await
            {
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_dropped(
                        "event_channel_send_failed",
                        delta_count,
                        "event_channel",
                    );
                }
                return Some(StreamError::Fatal(error.message));
            }
        }
        StreamDelta::Error { message, kind } => {
            log::warn!(
                "Stream error received delta_count={delta_count} message={message} kind={kind:?}"
            );
            let recoverable = kind.is_recoverable();
            #[cfg(feature = "otel")]
            if let Some(observer) = span_observer.as_mut() {
                let reason = if recoverable {
                    "recoverable_error"
                } else {
                    "fatal_error"
                };
                observer.record_dropped(reason, delta_count, stream_error_kind_attr(*kind));
            }
            return Some(if kind.is_connectivity() {
                StreamError::Connectivity {
                    message: message.clone(),
                    // Filled by the accumulator-owning caller.
                    usage: ZERO_USAGE,
                }
            } else if recoverable {
                StreamError::Recoverable {
                    message: message.clone(),
                    retry_after: kind.retry_after(),
                    usage: ZERO_USAGE,
                }
            } else {
                StreamError::Fatal(message.clone())
            });
        }
        // These are handled by the accumulator or not needed as events. The
        // catch-all also covers future `#[non_exhaustive]` deltas, which the
        // accumulator likewise consumes without emitting an event here.
        _ => {}
    }
    None
}

/// Map a streaming error kind to a stable `error.type` attribute value.
#[cfg(feature = "otel")]
const fn stream_error_kind_attr(kind: crate::llm::StreamErrorKind) -> &'static str {
    match kind {
        crate::llm::StreamErrorKind::Connectivity => "connectivity",
        crate::llm::StreamErrorKind::ConnectionLost => "connection_lost",
        crate::llm::StreamErrorKind::RateLimited(_) => "rate_limited",
        crate::llm::StreamErrorKind::ServerError => "server_error",
        crate::llm::StreamErrorKind::InvalidRequest => "invalid_request",
        // `StreamErrorKind` is `#[non_exhaustive]`; `Unknown` and any future
        // kind map to a stable generic attribute value.
        _ => "unknown",
    }
}

/// Whether a [`StreamDelta`] counts as a content chunk for the
/// `gen_ai.client.operation.time_to_first_chunk` /
/// `time_per_output_chunk` histograms.
///
/// Per the `GenAI` semconv, those histograms measure latency between
/// **output** chunks: text, thinking, tool-use, tool-input, and
/// signature deltas. Sentinel frames (`Usage`, `Done`, `Error`) are
/// metadata and would inflate the per-chunk distribution if counted.
#[cfg(feature = "otel")]
const fn is_content_delta(delta: &StreamDelta) -> bool {
    matches!(
        delta,
        StreamDelta::TextDelta { .. }
            | StreamDelta::ThinkingDelta { .. }
            | StreamDelta::ToolUseStart { .. }
            | StreamDelta::ToolInputDelta { .. }
            | StreamDelta::SignatureDelta { .. }
            | StreamDelta::RedactedThinking { .. }
    )
}

async fn send_llm_error_event<H>(
    event_ctx: &LlmEventContext<'_, H>,
    error_msg: &str,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    send_event(
        event_ctx.event_store,
        event_ctx.thread_id,
        event_ctx.turn,
        event_ctx.hooks,
        event_ctx.authority,
        AgentEvent::error(error_msg, true),
    )
    .await
}

/// Emit `AutoRetryStart` so consumers can render a "Retrying X/N
/// in Yms…" indicator. Errors are swallowed — losing telemetry on
/// the retry events shouldn't break the retry loop itself.
async fn send_auto_retry_start_event<H>(
    event_ctx: &LlmEventContext<'_, H>,
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error_message: &str,
) where
    H: AgentHooks,
{
    let _ = send_event(
        event_ctx.event_store,
        event_ctx.thread_id,
        event_ctx.turn,
        event_ctx.hooks,
        event_ctx.authority,
        AgentEvent::AutoRetryStart {
            attempt,
            max_attempts,
            delay_ms,
            error_message: error_message.to_string(),
        },
    )
    .await;
}

/// Emit `AutoRetryEnd` when a retry sequence settles — `success =
/// true` means a follow-up attempt eventually returned data;
/// `success = false` means the budget was exhausted and
/// `final_error` carries the last error.
async fn send_auto_retry_end_event<H>(
    event_ctx: &LlmEventContext<'_, H>,
    attempt: u32,
    success: bool,
    final_error: Option<String>,
) where
    H: AgentHooks,
{
    let _ = send_event(
        event_ctx.event_store,
        event_ctx.thread_id,
        event_ctx.turn,
        event_ctx.hooks,
        event_ctx.authority,
        AgentEvent::AutoRetryEnd {
            attempt,
            success,
            final_error,
        },
    )
    .await;
}
