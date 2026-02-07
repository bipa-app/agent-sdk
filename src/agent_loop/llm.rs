use super::helpers::{calculate_backoff_delay, send_event};
use super::types::StreamError;
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamAccumulator, StreamDelta, Usage,
};
use crate::types::{AgentConfig, AgentError};
use futures::StreamExt;
use log::{error, warn};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::sleep;

/// Call the LLM with retry logic for rate limits and server errors.
pub(super) async fn call_llm_with_retry<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
) -> Result<ChatResponse, AgentError>
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
                return Err(AgentError::new(format!("LLM error: {e}"), false));
            }
        };

        match outcome {
            ChatOutcome::Success(response) => return Ok(response),
            ChatOutcome::RateLimited => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Rate limited by LLM provider after {max_retries} retries");
                    let error_msg = format!("Rate limited after {max_retries} retries");
                    send_event(tx, hooks, seq, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Rate limited, retrying after backoff (attempt={}, delay_ms={})",
                    attempt,
                    delay.as_millis()
                );

                sleep(delay).await;
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!("Invalid request to LLM: {msg}");
                return Err(AgentError::new(format!("Invalid request: {msg}"), false));
            }
            ChatOutcome::ServerError(msg) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("LLM server error after {max_retries} retries: {msg}");
                    let error_msg = format!("Server error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, seq, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Server error, retrying after backoff (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

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
pub(super) async fn call_llm_streaming<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    ids: (&str, &str),
) -> Result<ChatResponse, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let (message_id, thinking_id) = ids;
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let result =
            process_stream(provider, &request, tx, hooks, seq, message_id, thinking_id).await;

        match result {
            Ok(response) => return Ok(response),
            Err(StreamError::Recoverable(msg)) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Streaming error after {max_retries} retries: {msg}");
                    let err_msg = format!("Streaming error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, seq, AgentEvent::error(&err_msg, true)).await;
                    return Err(AgentError::new(err_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Streaming error, retrying (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

                sleep(delay).await;
            }
            Err(StreamError::Fatal(msg)) => {
                error!("Streaming error (non-recoverable): {msg}");
                return Err(AgentError::new(format!("Streaming error: {msg}"), false));
            }
        }
    }
}

/// Process a single streaming attempt and return the response or error.
async fn process_stream<P, H>(
    provider: &Arc<P>,
    request: &ChatRequest,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    message_id: &str,
    thinking_id: &str,
) -> Result<ChatResponse, StreamError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let mut stream = std::pin::pin!(provider.chat_stream(request.clone()));
    let mut accumulator = StreamAccumulator::new();
    let mut delta_count: u64 = 0;

    log::debug!("Starting to consume LLM stream");

    // Track channel health
    let mut channel_closed = false;

    while let Some(result) = stream.next().await {
        // Log progress every 50 deltas to show stream is alive
        if delta_count > 0 && delta_count.is_multiple_of(50) {
            log::debug!("Stream progress: delta_count={delta_count}");
        }

        match result {
            Ok(delta) => {
                delta_count += 1;
                accumulator.apply(&delta);
                match &delta {
                    StreamDelta::TextDelta { delta, .. } => {
                        // Check if channel is still open before sending
                        if !channel_closed {
                            if tx.is_closed() {
                                log::warn!(
                                    "Event channel closed by receiver at delta_count={delta_count} - consumer may have disconnected"
                                );
                                channel_closed = true;
                            } else {
                                send_event(
                                    tx,
                                    hooks,
                                    seq,
                                    AgentEvent::text_delta(message_id, delta.clone()),
                                )
                                .await;
                            }
                        }
                    }
                    StreamDelta::ThinkingDelta { delta, .. } => {
                        if !channel_closed {
                            if tx.is_closed() {
                                log::warn!(
                                    "Event channel closed by receiver at delta_count={delta_count}"
                                );
                                channel_closed = true;
                            } else {
                                send_event(
                                    tx,
                                    hooks,
                                    seq,
                                    AgentEvent::thinking_delta(thinking_id, delta.clone()),
                                )
                                .await;
                            }
                        }
                    }
                    StreamDelta::Error {
                        message,
                        recoverable,
                    } => {
                        log::warn!(
                            "Stream error received delta_count={delta_count} message={message} recoverable={recoverable}"
                        );
                        return if *recoverable {
                            Err(StreamError::Recoverable(message.clone()))
                        } else {
                            Err(StreamError::Fatal(message.clone()))
                        };
                    }
                    // These are handled by the accumulator or not needed as events
                    StreamDelta::Done { .. }
                    | StreamDelta::Usage(_)
                    | StreamDelta::ToolUseStart { .. }
                    | StreamDelta::ToolInputDelta { .. }
                    | StreamDelta::SignatureDelta { .. }
                    | StreamDelta::RedactedThinking { .. } => {}
                }
            }
            Err(e) => {
                log::error!("Stream iteration error delta_count={delta_count} error={e}");
                return Err(StreamError::Recoverable(format!("Stream error: {e}")));
            }
        }
    }

    log::debug!("Stream while loop exited normally at delta_count={delta_count}");

    let usage = accumulator.usage().cloned().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
    });
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    log::debug!(
        "LLM stream completed successfully delta_count={delta_count} stop_reason={stop_reason:?} content_block_count={} input_tokens={} output_tokens={}",
        content_blocks.len(),
        usage.input_tokens,
        usage.output_tokens
    );

    Ok(ChatResponse {
        id: String::new(),
        content: content_blocks,
        model: provider.model().to_string(),
        stop_reason,
        usage,
    })
}
