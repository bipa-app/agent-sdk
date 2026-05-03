use super::helpers::{calculate_backoff_delay, send_event};
use super::types::{LlmEventContext, LlmStreamIds, StreamError};
use crate::events::AgentEvent;
use crate::hooks::AgentHooks;
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamAccumulator, StreamDelta, Usage,
};
use crate::types::{AgentConfig, AgentError};
use futures::StreamExt;
use log::{error, warn};
use std::sync::Arc;
use tokio::time::sleep;

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
}

#[cfg(feature = "otel")]
impl LlmSpanObserver<'_> {
    fn record_first_chunk(&mut self, turn: usize, streaming: bool) {
        use crate::observability::{attrs, spans};
        spans::add_event(
            self.span,
            "llm.stream.first_chunk",
            vec![
                attrs::kv_i64(attrs::SDK_TURN_NUMBER, i64::try_from(turn).unwrap_or(0)),
                attrs::kv_bool(attrs::SDK_LLM_STREAMING, streaming),
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
        use crate::observability::{attrs, spans};
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
) -> (Result<ChatResponse, AgentError>, u32)
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let outcome = match provider.chat(request.clone()).await {
            Ok(o) => o,
            Err(e) => {
                return (
                    Err(AgentError::new(format!("LLM error: {e}"), false)),
                    attempt,
                );
            }
        };

        match outcome {
            ChatOutcome::Success(response) => {
                if attempt > 0 {
                    send_auto_retry_end_event(event_ctx, attempt, true, None).await;
                }
                return (Ok(response), attempt);
            }
            ChatOutcome::RateLimited => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Rate limited by LLM provider after {max_retries} retries");
                    let error_msg = format!("Rate limited after {max_retries} retries");
                    send_auto_retry_end_event(
                        event_ctx,
                        attempt - 1,
                        false,
                        Some(error_msg.clone()),
                    )
                    .await;
                    if let Err(error) = send_llm_error_event(event_ctx, &error_msg).await {
                        return (Err(error), attempt);
                    }
                    return (Err(AgentError::new(error_msg, true)), attempt);
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Rate limited, retrying after backoff (attempt={}, delay_ms={})",
                    attempt,
                    delay.as_millis()
                );

                let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_retry(attempt, max_retries, delay_ms, "rate_limited");
                }

                send_auto_retry_start_event(
                    event_ctx,
                    attempt,
                    max_retries,
                    delay_ms,
                    "Rate limited by LLM provider",
                )
                .await;
                sleep(delay).await;
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!("Invalid request to LLM: {msg}");
                return (
                    Err(AgentError::new(format!("Invalid request: {msg}"), false)),
                    attempt,
                );
            }
            ChatOutcome::ServerError(msg) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("LLM server error after {max_retries} retries: {msg}");
                    let error_msg = format!("Server error after {max_retries} retries: {msg}");
                    send_auto_retry_end_event(
                        event_ctx,
                        attempt - 1,
                        false,
                        Some(error_msg.clone()),
                    )
                    .await;
                    if let Err(error) = send_llm_error_event(event_ctx, &error_msg).await {
                        return (Err(error), attempt);
                    }
                    return (Err(AgentError::new(error_msg, true)), attempt);
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Server error, retrying after backoff (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

                let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_retry(attempt, max_retries, delay_ms, "server_error");
                }

                send_auto_retry_start_event(event_ctx, attempt, max_retries, delay_ms, &msg).await;
                sleep(delay).await;
            }
        }
    }
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
    stream_ids: LlmStreamIds<'_>,
    #[cfg(feature = "otel")] mut span_observer: Option<LlmSpanObserver<'_>>,
) -> (Result<ChatResponse, AgentError>, u32)
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
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
            Ok(response) => {
                if attempt > 0 {
                    send_auto_retry_end_event(event_ctx, attempt, true, None).await;
                }
                return (Ok(response), attempt);
            }
            Err(StreamError::Recoverable(msg)) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Streaming error after {max_retries} retries: {msg}");
                    let err_msg = format!("Streaming error after {max_retries} retries: {msg}");
                    send_auto_retry_end_event(event_ctx, attempt - 1, false, Some(err_msg.clone()))
                        .await;
                    if let Err(error) = send_llm_error_event(event_ctx, &err_msg).await {
                        return (Err(error), attempt);
                    }
                    return (Err(AgentError::new(err_msg, true)), attempt);
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Streaming error, retrying (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

                let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
                #[cfg(feature = "otel")]
                if let Some(observer) = span_observer.as_mut() {
                    observer.record_retry(attempt, max_retries, delay_ms, "stream_error");
                }

                send_auto_retry_start_event(event_ctx, attempt, max_retries, delay_ms, &msg).await;
                sleep(delay).await;
            }
            Err(StreamError::Fatal(msg)) => {
                error!("Streaming error (non-recoverable): {msg}");
                return (
                    Err(AgentError::new(format!("Streaming error: {msg}"), false)),
                    attempt,
                );
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
    #[cfg(feature = "otel")]
    let stream_started_at = std::time::Instant::now();
    #[cfg(feature = "otel")]
    let mut first_chunk_recorded = false;

    log::debug!("Starting to consume LLM stream");

    while let Some(result) = stream.next().await {
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
                return Err(StreamError::Recoverable(format!("Stream error: {e}")));
            }
        };

        delta_count += 1;
        accumulator.apply(&delta);

        #[cfg(feature = "otel")]
        if !first_chunk_recorded {
            first_chunk_recorded = true;
            if let Some(observer) = span_observer.as_mut() {
                observer.record_first_chunk(event_ctx.turn, true);
            }
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
            return Err(stream_err);
        }
    }

    log::debug!("Stream while loop exited normally at delta_count={delta_count}");

    let usage = accumulator.usage().cloned().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
    });
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    log::debug!(
        "LLM stream completed successfully delta_count={delta_count} stop_reason={stop_reason:?} content_block_count={} input_tokens={} output_tokens={}",
        content_blocks.len(),
        usage.input_tokens,
        usage.output_tokens
    );

    #[cfg(feature = "otel")]
    if let Some(observer) = span_observer.as_mut() {
        let duration_ms =
            u64::try_from(stream_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        observer.record_completed(delta_count, duration_ms);
    }

    Ok(ChatResponse {
        id: uuid::Uuid::new_v4().to_string(),
        content: content_blocks,
        model: provider.model().to_string(),
        stop_reason,
        usage,
    })
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
            return Some(if recoverable {
                StreamError::Recoverable(message.clone())
            } else {
                StreamError::Fatal(message.clone())
            });
        }
        // These are handled by the accumulator or not needed as events
        StreamDelta::Done { .. }
        | StreamDelta::Usage(_)
        | StreamDelta::ToolUseStart { .. }
        | StreamDelta::ToolInputDelta { .. }
        | StreamDelta::SignatureDelta { .. }
        | StreamDelta::RedactedThinking { .. } => {}
    }
    None
}

/// Map a streaming error kind to a stable `error.type` attribute value.
#[cfg(feature = "otel")]
const fn stream_error_kind_attr(kind: crate::llm::StreamErrorKind) -> &'static str {
    match kind {
        crate::llm::StreamErrorKind::RateLimited => "rate_limited",
        crate::llm::StreamErrorKind::ServerError => "server_error",
        crate::llm::StreamErrorKind::InvalidRequest => "invalid_request",
    }
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
