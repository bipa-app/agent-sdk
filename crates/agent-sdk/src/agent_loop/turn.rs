use super::helpers::{build_assistant_message, extract_content, pending_tool_index, send_event};
use super::llm::{call_llm_streaming, call_llm_with_retry};
use super::tool_execution::{append_tool_results, execute_tool_call};
use super::types::{
    ExecuteTurnParameters, InternalTurnResult, LlmCallParams, LlmEventContext, LlmOutcome,
    ProcessedTurnResponse, ToolBatchExecutionParams, ToolCallExecutionContext,
    ToolExecutionOutcome, ToolOutcomeContext, TurnCompletionParams, TurnContext,
    TurnMessageLoadParams, TurnResponseProcessingParams, TurnStopReasonParams, TurnToolPhaseParams,
};

use crate::authority::EventAuthority;
use crate::context::{CompactionConfig, ContextCompactor, LlmContextCompactor};
use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, ToolAuditSink};
use crate::llm::{
    ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, StopReason,
};
use crate::stores::{EventStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, PendingToolCallInfo, ThreadId, TokenUsage,
    ToolResult, ToolRuntime, ToolTier, TurnOptions,
};
use agent_sdk_foundation::audit::AuditProvenance;

use log::{debug, info, warn};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(super) async fn begin_turn<H>(
    ctx: &mut TurnContext,
    max_turns: Option<usize>,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    ctx.turn += 1;
    ctx.state.turn_count = ctx.turn;

    // Warn the agent when approaching turn limit
    if let Some(max_turns) = max_turns {
        let remaining = max_turns.saturating_sub(ctx.turn);
        if remaining == 2 {
            info!(
                "Injecting turn-budget reminder (turn={}, max={max_turns})",
                ctx.turn
            );
            ctx.pending_reminder = Some(
                "TURN BUDGET WARNING: You have only 2 turns remaining before your turn limit. \
                 You MUST provide your final response NOW with everything you've found so far. \
                 Do NOT start any new tool calls — summarize your findings immediately."
                    .to_string(),
            );
        }
    }

    if let Some(max_turns) = max_turns
        && ctx.turn > max_turns
    {
        warn!("Max turns reached (turn={}, max={max_turns})", ctx.turn);
        #[cfg(feature = "otel")]
        crate::observability::instrument::record_root_event(
            "agent.max_turns_reached",
            vec![
                crate::observability::attrs::kv_i64(
                    crate::observability::attrs::SDK_TURN_NUMBER,
                    i64::try_from(ctx.turn).unwrap_or(0),
                ),
                crate::observability::attrs::kv_i64(
                    crate::observability::attrs::SDK_CONFIG_MAX_TURNS,
                    i64::try_from(max_turns).unwrap_or(0),
                ),
            ],
        );
        let message = format!("Maximum turns ({max_turns}) reached");
        send_event(
            event_store,
            &ctx.thread_id,
            ctx.turn,
            hooks,
            authority,
            AgentEvent::error(message.clone(), true),
        )
        .await?;
        return Err(AgentError::new(message, true));
    }

    send_event(
        event_store,
        &ctx.thread_id,
        ctx.turn,
        hooks,
        authority,
        AgentEvent::start(ctx.thread_id.clone(), ctx.turn),
    )
    .await?;
    Ok(())
}

/// Outcome of loading (and optionally compacting) a turn's history.
///
/// Compaction can run a slow LLM summarization that the user may cancel
/// mid-flight; in that case we stop *before* the destructive
/// `replace_history` write and report [`LoadedMessages::Cancelled`] so
/// the turn closes as `Cancelled` with history left untouched.
pub(super) enum LoadedMessages {
    Ready(Vec<Message>),
    Cancelled,
}

pub(super) async fn load_turn_messages<P, H, M>(
    TurnMessageLoadParams {
        thread_id,
        turn,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    }: TurnMessageLoadParams<'_, P, H, M>,
) -> Result<LoadedMessages, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let messages = match message_store.get_history(thread_id).await {
        Ok(m) => m,
        Err(error) => {
            send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                authority,
                AgentEvent::error(format!("Failed to get history: {error}"), false),
            )
            .await?;
            return Err(AgentError::new(
                format!("Failed to get history: {error}"),
                false,
            ));
        }
    };

    maybe_compact_messages(MaybeCompactParams {
        messages,
        turn,
        provider,
        message_store,
        thread_id,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    })
    .await
}

struct MaybeCompactParams<'a, P, H, M> {
    messages: Vec<Message>,
    turn: usize,
    provider: &'a Arc<P>,
    message_store: &'a Arc<M>,
    thread_id: &'a ThreadId,
    compaction_config: Option<&'a CompactionConfig>,
    compactor: Option<&'a Arc<dyn ContextCompactor>>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    cancel_token: &'a CancellationToken,
}

async fn maybe_compact_messages<P, H, M>(
    MaybeCompactParams {
        messages,
        turn,
        provider,
        message_store,
        thread_id,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    }: MaybeCompactParams<'_, P, H, M>,
) -> Result<LoadedMessages, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    if let Some(compactor) = compactor {
        if compactor.needs_compaction(&messages) {
            debug!(
                "Context compaction triggered (turn={}, message_count={})",
                turn,
                messages.len()
            );
            return compact_messages_for_threshold(ThresholdCompactionParams {
                compactor: compactor.as_ref(),
                messages,
                message_store,
                thread_id,
                event_store,
                turn,
                hooks,
                authority,
                cancel_token,
            })
            .await;
        }

        #[cfg(feature = "otel")]
        record_compaction_skipped(&messages);
        return Ok(LoadedMessages::Ready(messages));
    }

    if let Some(compact_config) = compaction_config {
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compact_config.clone());
        if default_compactor.needs_compaction(&messages) {
            debug!(
                "Context compaction triggered (turn={}, message_count={})",
                turn,
                messages.len()
            );
            return compact_messages_for_threshold(ThresholdCompactionParams {
                compactor: &default_compactor,
                messages,
                message_store,
                thread_id,
                event_store,
                turn,
                hooks,
                authority,
                cancel_token,
            })
            .await;
        }

        #[cfg(feature = "otel")]
        record_compaction_skipped(&messages);
    }

    Ok(LoadedMessages::Ready(messages))
}

struct StoredCompactionResult {
    messages: Vec<Message>,
    original_count: usize,
    new_count: usize,
    original_tokens: usize,
    new_tokens: usize,
}

struct ThresholdCompactionParams<'a, C: ?Sized, H, M> {
    compactor: &'a C,
    messages: Vec<Message>,
    message_store: &'a Arc<M>,
    thread_id: &'a ThreadId,
    event_store: &'a Arc<dyn EventStore>,
    turn: usize,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    cancel_token: &'a CancellationToken,
}

/// Outcome of a single compaction attempt that ends in storing the
/// summarized history.
enum CompactionOutcome {
    Stored(StoredCompactionResult),
    /// Cancel fired before the destructive `replace_history` write —
    /// history is untouched.
    Cancelled,
    Failed(AgentError),
}

async fn compact_messages_for_threshold<C, H, M>(
    ThresholdCompactionParams {
        compactor,
        messages,
        message_store,
        thread_id,
        event_store,
        turn,
        hooks,
        authority,
        cancel_token,
    }: ThresholdCompactionParams<'_, C, H, M>,
) -> Result<LoadedMessages, AgentError>
where
    C: ContextCompactor + ?Sized,
    H: AgentHooks,
    M: MessageStore,
{
    let original_messages = messages.clone();
    match compact_history_and_store(
        compactor,
        messages,
        message_store,
        thread_id,
        "threshold",
        "Context compaction failed",
        "Failed to replace history after compaction",
        cancel_token,
    )
    .await
    {
        CompactionOutcome::Stored(result) => {
            send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                authority,
                AgentEvent::context_compacted(
                    result.original_count,
                    result.new_count,
                    result.original_tokens,
                    result.new_tokens,
                ),
            )
            .await?;

            info!(
                "Context compacted successfully (original_count={}, new_count={}, original_tokens={}, new_tokens={})",
                result.original_count, result.new_count, result.original_tokens, result.new_tokens
            );

            Ok(LoadedMessages::Ready(result.messages))
        }
        CompactionOutcome::Cancelled => Ok(LoadedMessages::Cancelled),
        CompactionOutcome::Failed(error) => {
            warn!("Context compaction failed, continuing with full history: {error}");
            Ok(LoadedMessages::Ready(original_messages))
        }
    }
}

#[expect(clippy::too_many_arguments)]
async fn compact_history_and_store<C, M>(
    compactor: &C,
    messages: Vec<Message>,
    message_store: &Arc<M>,
    thread_id: &ThreadId,
    trigger: &'static str,
    compaction_error_prefix: &'static str,
    replace_history_error_prefix: &'static str,
    cancel_token: &CancellationToken,
) -> CompactionOutcome
where
    C: ContextCompactor + ?Sized,
    M: MessageStore,
{
    #[cfg(not(feature = "otel"))]
    let _ = trigger;
    #[cfg(feature = "otel")]
    let mut compaction_span = start_compaction_span(trigger);

    // Race the (potentially slow) LLM summarization against cancel so
    // a cancel during compaction stops us before the destructive write.
    let result = tokio::select! {
        biased;
        () = cancel_token.cancelled() => {
            log::info!("Context compaction cancelled during summarization");
            #[cfg(feature = "otel")]
            finish_compaction_span_error(
                &mut compaction_span,
                "context_compaction_cancelled",
                "cancelled",
            );
            return CompactionOutcome::Cancelled;
        }
        res = compactor.compact_history(messages) => match res {
            Ok(result) => result,
            Err(error) => {
                #[cfg(feature = "otel")]
                finish_compaction_span_error(
                    &mut compaction_span,
                    "context_compaction_failed",
                    &error.to_string(),
                );
                return CompactionOutcome::Failed(AgentError::new(
                    format!("{compaction_error_prefix}: {error}"),
                    false,
                ));
            }
        },
    };

    // Final guard before the destructive `replace_history`: if the
    // cancel landed after summarization but before we wrote, do not
    // mutate history. This is the invariant from gap #2 — an
    // uncancellable destructive `replace_history` must never complete
    // after the user asked to stop.
    if cancel_token.is_cancelled() {
        log::info!("Context compaction cancelled before replace_history write");
        #[cfg(feature = "otel")]
        finish_compaction_span_error(
            &mut compaction_span,
            "context_compaction_cancelled",
            "cancelled",
        );
        return CompactionOutcome::Cancelled;
    }

    let stored_result = StoredCompactionResult {
        messages: result.messages,
        original_count: result.original_count,
        new_count: result.new_count,
        original_tokens: result.original_tokens,
        new_tokens: result.new_tokens,
    };

    #[cfg(feature = "otel")]
    record_compaction_result(&mut compaction_span, &stored_result, trigger);

    if let Err(error) = message_store
        .replace_history(thread_id, stored_result.messages.clone())
        .await
    {
        #[cfg(feature = "otel")]
        finish_compaction_span_error(
            &mut compaction_span,
            "context_compaction_history_replace_failed",
            &error.to_string(),
        );
        return CompactionOutcome::Failed(AgentError::new(
            format!("{replace_history_error_prefix}: {error}"),
            false,
        ));
    }

    #[cfg(feature = "otel")]
    finish_compaction_span_success(&mut compaction_span);

    CompactionOutcome::Stored(stored_result)
}

#[cfg(feature = "otel")]
fn start_compaction_span(trigger: &'static str) -> opentelemetry::global::BoxedSpan {
    use crate::observability::{attrs, baggage, langfuse, spans};

    let mut span = spans::start_internal_span(
        "agent.context_compaction",
        vec![attrs::kv(attrs::SDK_COMPACTION_TRIGGER, trigger)],
    );
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
    span
}

#[cfg(feature = "otel")]
fn record_compaction_result(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &StoredCompactionResult,
    trigger: &'static str,
) {
    use crate::observability::{attrs, metrics, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    span.set_attribute(attrs::kv_i64(
        attrs::SDK_COMPACTION_ORIGINAL_COUNT,
        i64::try_from(result.original_count).unwrap_or(0),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::SDK_COMPACTION_NEW_COUNT,
        i64::try_from(result.new_count).unwrap_or(0),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::SDK_COMPACTION_ORIGINAL_TOKENS,
        i64::try_from(result.original_tokens).unwrap_or(0),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::SDK_COMPACTION_NEW_TOKENS,
        i64::try_from(result.new_tokens).unwrap_or(0),
    ));

    let metrics_handle = metrics::Metrics::global();
    metrics_handle
        .context_compaction
        .add(1, &[KeyValue::new(attrs::SDK_COMPACTION_TRIGGER, trigger)]);

    // `tokens_saved` only makes sense when the compactor actually
    // shrunk the token count. If it grew (or stayed flat with no
    // change), recording a zero-value sample would still skew the
    // histogram and hide a real regression. Per the B3 acceptance
    // criteria we skip the record and surface the anomaly via a
    // dedicated span event so we notice the compactor misbehaving.
    if result.new_tokens > result.original_tokens {
        spans::add_event(
            span,
            "compaction.expanded_unexpectedly",
            vec![
                attrs::kv_i64(
                    attrs::SDK_COMPACTION_ORIGINAL_TOKENS,
                    i64::try_from(result.original_tokens).unwrap_or(0),
                ),
                attrs::kv_i64(
                    attrs::SDK_COMPACTION_NEW_TOKENS,
                    i64::try_from(result.new_tokens).unwrap_or(0),
                ),
            ],
        );
        return;
    }

    let tokens_saved = result.original_tokens - result.new_tokens;
    match u64::try_from(tokens_saved) {
        Ok(tokens_saved_u64) => metrics_handle.context_compaction_tokens_saved.record(
            tokens_saved_u64,
            &[KeyValue::new(attrs::SDK_COMPACTION_TRIGGER, trigger)],
        ),
        Err(err) => {
            log::debug!(
                "skipping agent_sdk.context.compaction.tokens_saved record; \
                 tokens_saved={tokens_saved} did not fit in u64: {err}"
            );
        }
    }
}

#[cfg(feature = "otel")]
fn finish_compaction_span_success(span: &mut opentelemetry::global::BoxedSpan) {
    use crate::observability::attrs;
    use opentelemetry::trace::Span;

    span.set_attribute(attrs::kv(attrs::SDK_OUTCOME, "success"));
    span.end();
}

#[cfg(feature = "otel")]
fn finish_compaction_span_error(
    span: &mut opentelemetry::global::BoxedSpan,
    error_type: &'static str,
    message: &str,
) {
    use crate::observability::{attrs, spans};
    use opentelemetry::trace::Span;

    if error_type == "context_compaction_history_replace_failed" {
        spans::add_event(
            span,
            "compaction.history_replace_failed",
            vec![opentelemetry::KeyValue::new(
                attrs::ERROR_TYPE,
                error_type.to_string(),
            )],
        );
    }
    span.set_attribute(attrs::kv(attrs::SDK_OUTCOME, "error"));
    spans::set_span_error(span, error_type, message);
    span.end();
}

/// Emit a `compaction.skipped_below_threshold` event on the active
/// span.
///
/// Only fires when compaction was *configured* but `needs_compaction`
/// returned false — i.e. the SDK was capable of compacting but chose
/// not to. The event lands on whichever span is current
/// (`invoke_agent` for a fresh turn) so a Tempo viewer sees the
/// compaction-decision history without a dedicated child span.
#[cfg(feature = "otel")]
fn record_compaction_skipped(messages: &[Message]) {
    use crate::observability::attrs;
    use opentelemetry::Context;
    use opentelemetry::trace::TraceContextExt;

    let cx = Context::current();
    let span = cx.span();
    if !span.span_context().is_valid() {
        return;
    }
    span.add_event(
        "compaction.skipped_below_threshold",
        vec![attrs::kv_i64(
            attrs::SDK_COMPACTION_ORIGINAL_COUNT,
            i64::try_from(messages.len()).unwrap_or(0),
        )],
    );
}

pub(super) fn build_turn_request<Ctx, P>(
    config: &AgentConfig,
    provider: &Arc<P>,
    thread_id: &ThreadId,
    messages: Vec<Message>,
    tools: &Arc<ToolRegistry<Ctx>>,
) -> Result<ChatRequest, AgentError>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider,
{
    let llm_tools = if tools.is_empty() {
        None
    } else {
        Some(tools.to_llm_tools())
    };

    let thinking = provider.resolve_thinking_config(None).map_err(|error| {
        AgentError::new(format!("Invalid thinking configuration: {error}"), false)
    })?;

    Ok(ChatRequest {
        system: config.system_prompt.clone(),
        messages,
        tools: llm_tools,
        max_tokens: config
            .max_tokens
            .unwrap_or_else(|| provider.default_max_tokens()),
        max_tokens_explicit: config.max_tokens.is_some(),
        session_id: Some(thread_id.to_string()),
        cached_content: None,
        thinking,
        tool_choice: None,
        response_format: None,
        cache: None,
    })
}

/// Summarize a tool-call input JSON for debug logging without writing the
/// payload itself.
///
/// Returns the serialized byte length and the top-level object field names
/// (key names only, never their values) so an operator can size and shape a
/// call without the log persisting whatever PII / credentials the model
/// embedded in the input.
fn tool_input_shape(input: &serde_json::Value) -> String {
    let len = serde_json::to_string(input).map_or(0, |s| s.len());
    input.as_object().map_or_else(
        || format!("len={len}, non_object"),
        |map| {
            let keys: Vec<&str> = map.keys().map(String::as_str).collect();
            format!("len={len}, top_level_keys={keys:?}")
        },
    )
}

pub(super) fn log_chat_request(request: &ChatRequest) {
    debug!(
        "ChatRequest built: system_prompt_len={} num_messages={} num_tools={} max_tokens={} cached_content={}",
        request.system.len(),
        request.messages.len(),
        request.tools.as_ref().map_or(0, Vec::len),
        request.max_tokens,
        request.cached_content.as_deref().unwrap_or("<none>")
    );

    for (message_idx, message) in request.messages.iter().enumerate() {
        match &message.content {
            Content::Text(text) => {
                debug!(
                    "  message[{message_idx}]: role={:?} content=Text(len={})",
                    message.role,
                    text.len()
                );
            }
            Content::Blocks(blocks) => {
                debug!(
                    "  message[{message_idx}]: role={:?} content=Blocks(count={})",
                    message.role,
                    blocks.len()
                );
                for (block_idx, block) in blocks.iter().enumerate() {
                    match block {
                        ContentBlock::Text { text } => {
                            debug!("    block[{block_idx}]: Text(len={})", text.len());
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            debug!("    block[{block_idx}]: Thinking(len={})", thinking.len());
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            debug!("    block[{block_idx}]: RedactedThinking");
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            // Never log the raw input payload: tool inputs
                            // routinely carry user PII, file contents, or
                            // credentials/tokens the model passes to tools.
                            // Log only a redaction-safe shape summary (byte
                            // length + top-level field names), matching the
                            // length-only treatment of the Text/ToolResult arms.
                            debug!(
                                "    block[{block_idx}]: ToolUse(id={id}, name={name}, input_shape=[{}])",
                                tool_input_shape(input)
                            );
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            debug!(
                                "    block[{block_idx}]: ToolResult(tool_use_id={tool_use_id}, is_error={is_error:?}, content_len={})",
                                content.len()
                            );
                        }
                        ContentBlock::Image { source } => {
                            debug!(
                                "    block[{block_idx}]: Image(media_type={})",
                                source.media_type
                            );
                        }
                        ContentBlock::Document { source } => {
                            debug!(
                                "    block[{block_idx}]: Document(media_type={})",
                                source.media_type
                            );
                        }
                        // `ContentBlock` is `#[non_exhaustive]`; log unknown
                        // future block kinds generically.
                        _ => {
                            debug!("    block[{block_idx}]: <unrecognized block kind>");
                        }
                    }
                }
            }
        }
    }
}

pub(super) async fn request_llm_response<P, H>(
    LlmCallParams {
        provider,
        request,
        config,
        event_store,
        hooks,
        authority,
        thread_id,
        turn,
        message_id,
        thinking_id,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    }: LlmCallParams<'_, P, H>,
) -> LlmOutcome
where
    P: LlmProvider,
    H: AgentHooks,
{
    debug!("Calling LLM (turn={turn}, streaming={})", config.streaming);
    let event_ctx = LlmEventContext {
        event_store,
        hooks,
        authority,
        thread_id,
        turn,
        cancel_token,
    };

    #[cfg(feature = "otel")]
    let mut llm_span = build_llm_span(provider.as_ref(), config);

    #[cfg(feature = "otel")]
    let request_for_capture = observability_store.map(|_| request.clone());
    #[cfg(feature = "otel")]
    let provider_name_for_observer =
        crate::observability::provider_name::normalize(provider.provider());
    #[cfg(feature = "otel")]
    let request_model_for_metrics = provider.model().to_string();
    #[cfg(feature = "otel")]
    let llm_started_at = std::time::Instant::now();

    let (result, retry_count) = if config.streaming {
        call_llm_streaming(
            provider,
            request,
            config,
            &event_ctx,
            message_id,
            thinking_id,
            #[cfg(feature = "otel")]
            Some(super::llm::LlmSpanObserver {
                span: &mut llm_span,
                provider_name: provider_name_for_observer,
                request_model: &request_model_for_metrics,
            }),
        )
        .await
    } else {
        call_llm_with_retry(
            provider,
            request,
            config,
            &event_ctx,
            #[cfg(feature = "otel")]
            Some(super::llm::LlmSpanObserver {
                span: &mut llm_span,
                provider_name: provider_name_for_observer,
                request_model: &request_model_for_metrics,
            }),
        )
        .await
    };

    // Project the three-state outcome onto a `Result` for the OTel
    // span / payload bookkeeping. A cancellation is recorded as an
    // `AgentError` purely so the span carries a stable, filterable
    // terminal status; the caller still receives the original
    // `LlmOutcome::Cancelled` and routes it to the `Cancelled` event.
    #[cfg(feature = "otel")]
    let span_result: Result<ChatResponse, AgentError> = match &result {
        LlmOutcome::Response(response) => Ok(response.clone()),
        LlmOutcome::Cancelled => Err(AgentError::new("LLM call cancelled", false)),
        LlmOutcome::Error(error) => Err(error.clone()),
    };

    #[cfg(feature = "otel")]
    if let (Some(observability_store), Some(request), Ok(response)) = (
        observability_store,
        request_for_capture.as_ref(),
        span_result.as_ref(),
    ) {
        capture_llm_payloads(
            &mut llm_span,
            observability_store.as_ref(),
            provider,
            request,
            response,
            thread_id,
            turn,
        )
        .await;
    }

    #[cfg(feature = "otel")]
    finish_llm_span(FinishLlmSpanParams {
        span: &mut llm_span,
        result: &span_result,
        retry_count,
        llm_started_at,
        provider_name: provider_name_for_observer,
        request_model: &request_model_for_metrics,
    });

    // Silence unused binding when otel is disabled.
    let _ = retry_count;
    let _ = thread_id;

    result
}

/// Build the per-call `chat <model>` span with the
/// `gen_ai` semconv attributes the SDK has decided so far.
///
/// Extracted from [`request_llm_response`] so the request function
/// stays under the clippy `too_many_lines` threshold.
#[cfg(feature = "otel")]
fn build_llm_span<P>(provider: &P, config: &AgentConfig) -> opentelemetry::global::BoxedSpan
where
    P: LlmProvider,
{
    use crate::observability::{attrs, baggage, langfuse, spans};
    use opentelemetry::KeyValue;

    let span_name = format!("chat {}", provider.model());
    let provider_name = crate::observability::provider_name::normalize(provider.provider());
    let mut init_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "chat"),
        KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, provider_name),
        KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, provider.model().to_string()),
        attrs::kv_bool(attrs::SDK_LLM_STREAMING, config.streaming),
        KeyValue::new(attrs::SDK_PROVIDER_ID, provider.provider()),
    ];
    if let Some(max_tokens) = config.max_tokens {
        init_attrs.push(attrs::kv_i64(
            attrs::GEN_AI_REQUEST_MAX_OUTPUT_TOKENS,
            i64::from(max_tokens),
        ));
    }
    let mut span = spans::start_client_span(span_name, init_attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Generation);
    span
}

#[cfg(feature = "otel")]
async fn capture_llm_payloads<P>(
    span: &mut opentelemetry::global::BoxedSpan,
    observability_store: &dyn crate::observability::ObservabilityStore,
    provider: &Arc<P>,
    request: &ChatRequest,
    response: &ChatResponse,
    thread_id: &ThreadId,
    turn: usize,
) where
    P: LlmProvider,
{
    use crate::observability::{CaptureKind, PayloadBundle, spans};
    use opentelemetry::trace::Span;

    let redactor = observability_store.redactor();
    let system_json = redactor.convert_system_instructions(request);
    let input_json = redactor.convert_input_messages(request);
    let output_json = redactor.convert_output_messages(response);
    let bundle = PayloadBundle {
        capture_id: uuid::Uuid::new_v4().to_string(),
        capture_kind: CaptureKind::TurnChat,
        thread_id: thread_id.clone(),
        turn_number: turn,
        provider_name: crate::observability::provider_name::normalize(provider.provider())
            .to_string(),
        provider_id: provider.provider().to_string(),
        span_is_recording: span.is_recording(),
        request_model: provider.model().to_string(),
        response_model: Some(response.model.clone()),
        system_instructions: system_json.clone(),
        input_messages: input_json.clone(),
        output_messages: output_json.clone(),
    };

    match observability_store.capture(&bundle).await {
        Ok(result) => {
            // Enforce the default-deny privacy gate before any
            // payload reaches the span.  `Inline`
            // decisions only land when *both* the operator-level
            // capture flag is on (via `OtelConfig::capture_payloads`)
            // *and* the store has explicitly attested PII safety
            // via `ObservabilityStore::acknowledge_pii_redaction`.
            // Otherwise every `Inline` is downgraded to `Omit`;
            // `Reference` always passes through.
            let gated = crate::observability::payload_capture::gate(observability_store, result);
            spans::record_payload_on_span(
                span,
                &gated,
                system_json.as_ref(),
                &input_json,
                &output_json,
            );
        }
        Err(error) => {
            warn!("Failed to capture observability payloads: {error}");
            span.add_event("payload_capture_failed", vec![]);
        }
    }
}

#[cfg(feature = "otel")]
struct FinishLlmSpanParams<'a> {
    span: &'a mut opentelemetry::global::BoxedSpan,
    result: &'a Result<ChatResponse, AgentError>,
    retry_count: u32,
    /// Wall-clock instant at which the SDK called the provider. Used
    /// to record `gen_ai.client.operation.duration`.
    llm_started_at: std::time::Instant,
    /// Normalized provider name, e.g. `anthropic`, `gcp.gemini`.
    provider_name: &'static str,
    /// `gen_ai.request.model` — the model the SDK *asked* for, before
    /// any redirect the provider may have done. Captured before the
    /// call because the response model can differ.
    request_model: &'a str,
}

#[cfg(feature = "otel")]
fn finish_llm_span(params: FinishLlmSpanParams<'_>) {
    use crate::observability::{attrs, metrics};
    use opentelemetry::trace::Span;

    let FinishLlmSpanParams {
        span,
        result,
        retry_count,
        llm_started_at,
        provider_name,
        request_model,
    } = params;

    // Per-retry `llm.retry` events are emitted on the live span by
    // `call_llm_with_retry` / `call_llm_streaming` so they carry
    // accurate provider / error context. The retry count is still
    // surfaced as an attribute here for filterability.
    if retry_count > 0 {
        span.set_attribute(attrs::kv_i64(
            attrs::SDK_LLM_RETRY_ATTEMPT,
            i64::from(retry_count),
        ));
    }

    let metrics_handle = metrics::Metrics::global();
    let elapsed_secs = llm_started_at.elapsed().as_secs_f64();

    match result {
        Ok(response) => stamp_llm_success(
            span,
            response,
            &metrics_handle,
            elapsed_secs,
            provider_name,
            request_model,
        ),
        Err(err) => stamp_llm_error(
            span,
            err,
            &metrics_handle,
            elapsed_secs,
            provider_name,
            request_model,
        ),
    }
    span.end();
}

/// Stamp a successful `chat` response onto the span and record its
/// duration + per-token-type usage on the `OTel` `gen_ai.client.*`
/// histograms. Called from [`finish_llm_span`] only.
#[cfg(feature = "otel")]
fn stamp_llm_success(
    span: &mut opentelemetry::global::BoxedSpan,
    response: &ChatResponse,
    metrics_handle: &crate::observability::metrics::Metrics,
    elapsed_secs: f64,
    provider_name: &'static str,
    request_model: &str,
) {
    use crate::observability::attrs;
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    if !response.id.is_empty() {
        span.set_attribute(KeyValue::new(
            attrs::GEN_AI_RESPONSE_ID,
            response.id.clone(),
        ));
    }
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_RESPONSE_MODEL,
        response.model.clone(),
    ));
    if let Some(reason) = response.stop_reason {
        span.set_attribute(KeyValue::new(
            attrs::GEN_AI_RESPONSE_FINISH_REASONS,
            attrs::finish_reason_str(reason),
        ));
    }
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        i64::from(response.usage.input_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
        i64::from(response.usage.output_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        i64::from(response.usage.cached_input_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        i64::from(response.usage.cache_creation_input_tokens),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_HAD_TOOL_CALLS,
        response.has_tool_use(),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_TEXT_OUTPUT_PRESENT,
        response.first_text().is_some(),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_THINKING_PRESENT,
        response.first_thinking().is_some(),
    ));

    // Delegate the metric emission to the shared recorders so the
    // daemon-hosted worker (`observability::loop_instrument`) lands in
    // exactly the same series with the same labels.
    metrics_handle.record_chat_token_usage(
        &response.usage,
        provider_name,
        request_model,
        &response.model,
    );
    metrics_handle.record_chat_operation_duration_success(
        elapsed_secs,
        provider_name,
        request_model,
        &response.model,
    );
}

/// Stamp an LLM error onto the span and record its duration with the
/// `error.type` label populated. Called from [`finish_llm_span`] only.
#[cfg(feature = "otel")]
fn stamp_llm_error(
    span: &mut opentelemetry::global::BoxedSpan,
    err: &AgentError,
    metrics_handle: &crate::observability::metrics::Metrics,
    elapsed_secs: f64,
    provider_name: &'static str,
    request_model: &str,
) {
    use crate::observability::spans;

    let error_type = classify_llm_error(&err.message);
    spans::set_span_error(span, error_type, &err.message);

    metrics_handle.record_chat_operation_duration_error(
        elapsed_secs,
        provider_name,
        request_model,
        error_type,
    );
}

/// Map an `AgentError::message` produced by the LLM call path to the
/// stable `error.type` attribute value used on both the span and the
/// `gen_ai.client.operation.duration` histogram.
#[cfg(feature = "otel")]
fn classify_llm_error(msg: &str) -> &'static str {
    if msg.contains("Rate limited") {
        "rate_limited"
    } else if msg.contains("Invalid request") {
        "invalid_request"
    } else if msg.contains("Server error") {
        "server_error"
    } else if msg.contains("Stream") {
        "stream_error"
    } else {
        "_OTHER"
    }
}

pub(super) fn apply_turn_usage(ctx: &mut TurnContext, response: &ChatResponse) -> TokenUsage {
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
    };
    ctx.total_usage.add(&turn_usage);
    ctx.state.total_usage = ctx.total_usage.clone();

    // Capture provider-level provenance into the turn context so that
    // the [`agent_sdk_foundation::TurnSummary`] emitted at outcome time
    // reflects the real response metadata rather than the defaults.
    //
    // Response IDs are only recorded when the provider actually
    // returned one; legacy mock providers that emit an empty string
    // leave the field as `None`.
    if !response.id.is_empty() {
        ctx.response_id = Some(response.id.clone());
    }
    ctx.stop_reason = response.stop_reason;

    turn_usage
}

#[cfg(feature = "otel")]
const fn token_usage_delta(current: &TokenUsage, baseline: &TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: current.input_tokens.saturating_sub(baseline.input_tokens),
        output_tokens: current.output_tokens.saturating_sub(baseline.output_tokens),
        cached_input_tokens: current
            .cached_input_tokens
            .saturating_sub(baseline.cached_input_tokens),
        cache_creation_input_tokens: current
            .cache_creation_input_tokens
            .saturating_sub(baseline.cache_creation_input_tokens),
    }
}

pub(super) async fn process_turn_response<Ctx, H, M>(
    TurnResponseProcessingParams {
        response,
        message_id,
        thinking_id,
        thread_id,
        turn,
        tools,
        message_store,
        event_store,
        hooks,
        authority,
    }: TurnResponseProcessingParams<'_, Ctx, H, M>,
) -> Result<ProcessedTurnResponse, AgentError>
where
    Ctx: Send + Sync + 'static,
    H: AgentHooks,
    M: MessageStore,
{
    let stop_reason = response.stop_reason;
    let (thinking_content, text_content, tool_uses) = extract_content(&response);

    if let Some(thinking) = &thinking_content {
        send_event(
            event_store,
            thread_id,
            turn,
            hooks,
            authority,
            AgentEvent::thinking(thinking_id, thinking.clone()),
        )
        .await?;
    }

    if let Some(text) = &text_content {
        send_event(
            event_store,
            thread_id,
            turn,
            hooks,
            authority,
            AgentEvent::text(message_id, text.clone()),
        )
        .await?;
    }

    let assistant_msg = build_assistant_message(&response);
    if let Err(error) = message_store.append(thread_id, assistant_msg).await {
        send_event(
            event_store,
            thread_id,
            turn,
            hooks,
            authority,
            AgentEvent::error(
                format!("Failed to append assistant message: {error}"),
                false,
            ),
        )
        .await?;
        return Err(AgentError::new(
            format!("Failed to append assistant message: {error}"),
            false,
        ));
    }

    Ok(ProcessedTurnResponse {
        stop_reason,
        text_content,
        pending_tool_calls: build_pending_tool_calls(tools, &tool_uses),
    })
}

pub(super) fn build_pending_tool_calls<Ctx>(
    tools: &Arc<ToolRegistry<Ctx>>,
    tool_uses: &[(String, String, serde_json::Value)],
) -> Vec<PendingToolCallInfo>
where
    Ctx: Send + Sync + 'static,
{
    tool_uses
        .iter()
        .map(|(id, name, input)| {
            // Resolve the tool metadata in one pass so `display_name`
            // and `tier` stay in lockstep. Unknown tools fall back to
            // the strictest tier so downstream audit/policy layers see
            // a conservative default rather than silently observe.
            let (display_name, tier) = tools
                .get(name)
                .map(|tool| (tool.display_name().to_string(), tool.tier()))
                .or_else(|| {
                    tools
                        .get_async(name)
                        .map(|tool| (tool.display_name().to_string(), tool.tier()))
                })
                .or_else(|| {
                    tools
                        .get_listen(name)
                        .map(|tool| (tool.display_name().to_string(), tool.tier()))
                })
                .unwrap_or_else(|| (String::new(), crate::types::ToolTier::Confirm));

            PendingToolCallInfo {
                id: id.clone(),
                name: name.clone(),
                display_name,
                tier,
                input: input.clone(),
                effective_input: input.clone(),
                listen_context: None,
            }
        })
        .collect()
}

pub(super) async fn execute_pending_tool_calls_for_turn<Ctx, H>(
    ToolBatchExecutionParams {
        pending_tool_calls,
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        authority,
        execution_store,
        audit_sink,
        provenance,
        turn,
        total_usage,
        turn_usage,
        state,
        response_id,
        stop_reason,
    }: ToolBatchExecutionParams<'_, Ctx, H>,
) -> Result<Vec<(String, ToolResult)>, InternalTurnResult>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let mut pending_tool_calls = pending_tool_calls;
    let mut tool_results = Vec::new();
    let execution_ctx = ToolCallExecutionContext {
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        turn,
        authority,
        execution_store,
        audit_sink,
        provenance,
    };
    let outcome_ctx = ToolOutcomeContext {
        thread_id,
        turn,
        total_usage,
        turn_usage,
        state,
        response_id: response_id.as_deref(),
        stop_reason,
    };

    // Tool calls are walked in the order the model emitted them. Adjacent
    // `ToolTier::Observe` calls are run concurrently via `join_all` because
    // they are read-only by contract and safe to overlap. Anything else
    // (currently only `ToolTier::Confirm`, plus the listen path that may
    // suspend for confirmation) stays strictly serial: a confirmation gate
    // is a sequencing decision and the next tool's input may depend on the
    // previous tool's effect on the world.
    //
    // Result order is preserved to match the input order, so downstream
    // consumers (tool-result message assembly, audit logs, replay) see no
    // observable change beyond reduced wall time.
    let mut idx = 0;
    while idx < pending_tool_calls.len() {
        let first_tier = pending_tool_calls[idx].tier;

        if first_tier == ToolTier::Observe {
            let end = observe_run_end(&pending_tool_calls, idx);

            // Borrow the slice for the duration of `join_all` only — the
            // borrow ends before we touch `pending_tool_calls` mutably below.
            let outcomes = futures::future::join_all(
                pending_tool_calls[idx..end]
                    .iter()
                    .map(|p| execute_tool_call(p, &execution_ctx)),
            )
            .await;

            let mut outcomes = outcomes.into_iter();
            while let Some(outcome) = outcomes.next() {
                let Some(mut early) = handle_tool_outcome(
                    outcome,
                    &mut pending_tool_calls,
                    &mut tool_results,
                    &outcome_ctx,
                ) else {
                    continue;
                };
                // If this batch element paused for confirmation, the sibling
                // tools *after* it in the same parallel batch already ran to
                // completion (their futures resolved inside `join_all`). Drain
                // their finished results into the continuation so they are not
                // dropped and re-executed on resume — double execution would
                // re-emit tool_call events, duplicate audit Completed records,
                // and repeat side effects. Resume skips any pending tool whose
                // result is already carried in `completed_results`.
                if let InternalTurnResult::AwaitingConfirmation { continuation, .. } = &mut early {
                    for sibling in outcomes.by_ref() {
                        if let ToolExecutionOutcome::Completed { tool_id, result } = sibling {
                            continuation.completed_results.push((tool_id, result));
                        }
                        // A sibling still awaiting confirmation has not run its
                        // `execute()` yet, and an infra error re-surfaces when
                        // the tool is retried on resume — neither yielded a
                        // result to capture here.
                    }
                }
                return Err(early);
            }
            idx = end;
        } else {
            let outcome = execute_tool_call(&pending_tool_calls[idx], &execution_ctx).await;
            if let Some(early) = handle_tool_outcome(
                outcome,
                &mut pending_tool_calls,
                &mut tool_results,
                &outcome_ctx,
            ) {
                return Err(early);
            }
            idx += 1;
        }
    }

    Ok(tool_results)
}

/// Find the end (exclusive) of the run of consecutive `ToolTier::Observe`
/// calls that starts at `start`.
fn observe_run_end(pending: &[PendingToolCallInfo], start: usize) -> usize {
    let mut end = start;
    while end < pending.len() && pending[end].tier == ToolTier::Observe {
        end += 1;
    }
    end
}

/// Process a single tool execution outcome. On `Completed`, append to the
/// results vector and return `None`. On `RequiresConfirmation`, snapshot the
/// turn into a continuation and return the `InternalTurnResult` the caller
/// should propagate. On `Error`, surface it.
///
/// Pulled out of the main loop so the parallel and serial paths share one
/// implementation of the confirmation snapshot.
fn handle_tool_outcome(
    outcome: ToolExecutionOutcome,
    pending_tool_calls: &mut [PendingToolCallInfo],
    tool_results: &mut Vec<(String, ToolResult)>,
    ctx: &ToolOutcomeContext<'_>,
) -> Option<InternalTurnResult> {
    match outcome {
        ToolExecutionOutcome::Completed { tool_id, result } => {
            tool_results.push((tool_id, result));
            None
        }
        ToolExecutionOutcome::RequiresConfirmation {
            tool_id,
            tool_name,
            display_name,
            input,
            description,
            listen_context,
        } => {
            let pending_idx = match pending_tool_index(pending_tool_calls, &tool_id) {
                Ok(index) => index,
                Err(error) => return Some(InternalTurnResult::Error(error)),
            };
            if let Some(context) = listen_context {
                pending_tool_calls[pending_idx].listen_context = Some(context);
            }

            let continuation = AgentContinuation {
                thread_id: ctx.thread_id.clone(),
                turn: ctx.turn,
                total_usage: ctx.total_usage.clone(),
                turn_usage: ctx.turn_usage.clone(),
                pending_tool_calls: pending_tool_calls.to_vec(),
                awaiting_index: pending_idx,
                completed_results: std::mem::take(tool_results),
                state: ctx.state.clone(),
                response_id: ctx.response_id.map(str::to_string),
                stop_reason: ctx.stop_reason,
                response_content: Vec::new(),
            };

            Some(InternalTurnResult::AwaitingConfirmation {
                tool_call_id: tool_id,
                tool_name,
                display_name,
                input,
                description,
                continuation: Box::new(continuation),
            })
        }
        ToolExecutionOutcome::Error(error) => Some(InternalTurnResult::Error(error)),
    }
}

pub(super) async fn append_tool_results_and_emit_turn_complete<H, M>(
    TurnCompletionParams {
        tool_results,
        thread_id,
        turn,
        turn_usage,
        message_store,
        event_store,
        hooks,
        authority,
    }: TurnCompletionParams<'_, H, M>,
) -> Result<(), AgentError>
where
    H: AgentHooks,
    M: MessageStore,
{
    append_tool_results(tool_results, thread_id, message_store).await?;
    send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::TurnComplete {
            turn,
            usage: turn_usage.clone(),
        },
    )
    .await?;
    Ok(())
}

pub(super) async fn execute_turn_tool_phase<Ctx, H, M>(
    TurnToolPhaseParams {
        pending_tool_calls,
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        authority,
        execution_store,
        audit_sink,
        provenance,
        turn,
        total_usage,
        turn_usage,
        state,
        message_store,
        response_id,
        stop_reason,
    }: TurnToolPhaseParams<'_, Ctx, H, M>,
) -> Result<(), InternalTurnResult>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
{
    let tool_results = execute_pending_tool_calls_for_turn(ToolBatchExecutionParams {
        pending_tool_calls,
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        authority,
        execution_store,
        audit_sink,
        provenance,
        turn,
        total_usage,
        turn_usage,
        state,
        response_id,
        stop_reason,
    })
    .await?;

    if let Err(error) = append_tool_results_and_emit_turn_complete(TurnCompletionParams {
        tool_results: &tool_results,
        thread_id,
        turn,
        turn_usage,
        message_store,
        event_store,
        hooks,
        authority,
    })
    .await
    {
        return Err(InternalTurnResult::Error(error));
    }

    Ok(())
}

/// Outcome of an overflow-driven (emergency) compaction.
pub(super) enum OverflowCompaction {
    Done,
    /// Cancel fired during the emergency compaction; history untouched.
    Cancelled,
}

pub(super) async fn compact_after_context_overflow<P, M>(
    provider: &Arc<P>,
    compaction_config: &CompactionConfig,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    message_store: &Arc<M>,
    thread_id: &ThreadId,
    cancel_token: &CancellationToken,
) -> Result<OverflowCompaction, AgentError>
where
    P: LlmProvider,
    M: MessageStore,
{
    let history = message_store
        .get_history(thread_id)
        .await
        .map_err(|error| {
            AgentError::new(
                format!("Failed to get history for compaction after context overflow: {error}"),
                false,
            )
        })?;

    let outcome = if let Some(compactor) = compactor {
        compact_history_and_store(
            compactor.as_ref(),
            history,
            message_store,
            thread_id,
            "overflow",
            "Context compaction failed after overflow",
            "Failed to replace history after overflow compaction",
            cancel_token,
        )
        .await
    } else {
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compaction_config.clone());
        compact_history_and_store(
            &default_compactor,
            history,
            message_store,
            thread_id,
            "overflow",
            "Context compaction failed after overflow",
            "Failed to replace history after overflow compaction",
            cancel_token,
        )
        .await
    };

    match outcome {
        CompactionOutcome::Stored(result) => {
            info!(
                "Context compacted after overflow (original_tokens={}, new_tokens={})",
                result.original_tokens, result.new_tokens
            );
            Ok(OverflowCompaction::Done)
        }
        CompactionOutcome::Cancelled => Ok(OverflowCompaction::Cancelled),
        CompactionOutcome::Failed(error) => Err(error),
    }
}

pub(super) async fn handle_turn_stop_reason<P, H, M>(
    TurnStopReasonParams {
        stop_reason,
        text_content,
        had_tool_calls,
        message_id,
        turn_usage,
        ctx,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    }: TurnStopReasonParams<'_, P, H, M>,
) -> InternalTurnResult
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    match stop_reason {
        Some(StopReason::EndTurn) => {
            info!("Agent completed (end_turn) (turn={})", ctx.turn);
            InternalTurnResult::Done
        }
        Some(StopReason::ToolUse) => {
            debug!("Tool use stop (turn={})", ctx.turn);
            InternalTurnResult::Continue { turn_usage }
        }
        Some(StopReason::Refusal) => {
            warn!(
                "Model refused request (turn={}): {:?}",
                ctx.turn, text_content
            );
            if let Err(error) = send_event(
                event_store,
                &ctx.thread_id,
                ctx.turn,
                hooks,
                authority,
                AgentEvent::refusal(message_id, text_content),
            )
            .await
            {
                return InternalTurnResult::Error(error);
            }
            InternalTurnResult::Refusal
        }
        Some(StopReason::MaxTokens) => {
            if had_tool_calls {
                // Tool calls were executed and their results appended as a user
                // message, so message alternation is preserved. Safe to continue.
                warn!(
                    "Max tokens reached with tool calls (turn={}), continuing",
                    ctx.turn
                );
                InternalTurnResult::Continue { turn_usage }
            } else {
                // No tool calls means no user message was appended after the
                // assistant message. Continuing would create consecutive assistant
                // messages which violates the API's alternation requirement and
                // corrupts thinking block history. Stop gracefully.
                warn!(
                    "Max tokens reached with no tool calls (turn={}, has_text={}). \
                     Stopping to prevent consecutive assistant messages. \
                     Consider increasing max_tokens.",
                    ctx.turn,
                    text_content.is_some(),
                );
                InternalTurnResult::Done
            }
        }
        Some(StopReason::ModelContextWindowExceeded) => {
            handle_context_window_exceeded(
                ctx,
                turn_usage,
                provider,
                message_store,
                compaction_config,
                compactor,
                cancel_token,
            )
            .await
        }
        Some(StopReason::StopSequence) => {
            info!("Stop sequence hit (turn={})", ctx.turn);
            InternalTurnResult::Done
        }
        // `StopReason` is `#[non_exhaustive]`; `StopReason::Unknown` (an
        // unrecognized provider value, via `#[serde(other)]`), any future
        // variant (`Some(_)`), and a missing stop reason (`None`) are all
        // treated the same — fall back to the alternation-preserving logic.
        Some(_) | None => {
            // Unknown/missing stop reason. Only continue if tool results were
            // appended (preserving message alternation).
            if had_tool_calls {
                warn!(
                    "No usable stop reason with tool calls (turn={}), continuing",
                    ctx.turn
                );
                InternalTurnResult::Continue { turn_usage }
            } else {
                warn!(
                    "No usable stop reason and no tool calls (turn={}), stopping",
                    ctx.turn
                );
                InternalTurnResult::Done
            }
        }
    }
}

async fn handle_context_window_exceeded<P, M>(
    ctx: &mut TurnContext,
    turn_usage: TokenUsage,
    provider: &Arc<P>,
    message_store: &Arc<M>,
    compaction_config: Option<&CompactionConfig>,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    cancel_token: &CancellationToken,
) -> InternalTurnResult
where
    P: LlmProvider,
    M: MessageStore,
{
    warn!("Model context window exceeded (turn={})", ctx.turn);
    #[cfg(feature = "otel")]
    crate::observability::instrument::record_root_event(
        "agent.context_window_exceeded",
        vec![crate::observability::attrs::kv_i64(
            crate::observability::attrs::SDK_TURN_NUMBER,
            i64::try_from(ctx.turn).unwrap_or(0),
        )],
    );
    ctx.compaction_retries += 1;
    if ctx.compaction_retries > super::types::MAX_COMPACTION_RETRIES {
        return InternalTurnResult::Error(AgentError::new(
            format!(
                "Context window exceeded after {} compaction retries — giving up",
                super::types::MAX_COMPACTION_RETRIES
            ),
            false,
        ));
    }
    if let Some(compact_config) = compaction_config {
        match compact_after_context_overflow(
            provider,
            compact_config,
            compactor,
            message_store,
            &ctx.thread_id,
            cancel_token,
        )
        .await
        {
            Ok(OverflowCompaction::Done) => {}
            Ok(OverflowCompaction::Cancelled) => {
                return InternalTurnResult::Cancelled { turn_usage };
            }
            Err(error) => return InternalTurnResult::Error(error),
        }
        // Keep the turn counter monotonic. The overflow turn was opened by
        // `begin_turn` and is closed by the looping result handler's
        // `Continue` arm; decrementing here would make the next iteration
        // recompute the same turn key and re-`begin_turn` on an
        // already-finished turn, which the event store rejects. The retry
        // simply runs as the next turn against the freshly compacted history.
        return InternalTurnResult::Continue { turn_usage };
    }

    InternalTurnResult::Error(AgentError::new(
        "Model context window exceeded and no compaction configured".to_string(),
        false,
    ))
}

/// Checks if an error message indicates the prompt exceeds the model's context
/// window. Matches multiple known error patterns from different providers.
fn is_prompt_too_long_error(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("prompt is too long")
        || lower.contains("maximum context length")
        || lower.contains("context_length_exceeded")
        || lower.contains("input is too long")
        || lower.contains("request too large")
}

/// When the prompt exceeds the model's context window at the API level (returned as a 400 error
/// rather than a `stop_reason`), attempt compaction and retry instead of failing immediately.
async fn try_recover_prompt_too_long<P, M>(
    error: &AgentError,
    ctx: &mut TurnContext,
    compaction_config: Option<&CompactionConfig>,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    provider: &Arc<P>,
    message_store: &Arc<M>,
    cancel_token: &CancellationToken,
) -> Option<InternalTurnResult>
where
    P: LlmProvider,
    M: MessageStore,
{
    if is_prompt_too_long_error(&error.message)
        && let Some(compact_config) = compaction_config
    {
        ctx.compaction_retries += 1;
        if ctx.compaction_retries > super::types::MAX_COMPACTION_RETRIES {
            return Some(InternalTurnResult::Error(AgentError::new(
                format!(
                    "Prompt too long after {} compaction retries — giving up",
                    super::types::MAX_COMPACTION_RETRIES
                ),
                false,
            )));
        }
        warn!(
            "Prompt too long, attempting emergency context compaction (turn={}, retry={})",
            ctx.turn, ctx.compaction_retries
        );
        match compact_after_context_overflow(
            provider,
            compact_config,
            compactor,
            message_store,
            &ctx.thread_id,
            cancel_token,
        )
        .await
        {
            Ok(OverflowCompaction::Done) => {}
            Ok(OverflowCompaction::Cancelled) => {
                return Some(InternalTurnResult::Cancelled {
                    turn_usage: TokenUsage::default(),
                });
            }
            Err(compact_err) => {
                return Some(InternalTurnResult::Error(compact_err));
            }
        }
        // Keep the turn counter monotonic (see `handle_context_window_exceeded`):
        // the turn was already opened via `begin_turn`, so the retry runs as the
        // next turn against the compacted history rather than replaying this
        // turn's already-finished event key.
        return Some(InternalTurnResult::Continue {
            turn_usage: TokenUsage::default(),
        });
    }
    None
}

/// Persist a strict-durability state checkpoint, mapping a save failure
/// onto the hard `InternalTurnResult::Error` the turn must surface.
async fn save_strict_checkpoint<S>(
    state_store: &Arc<S>,
    state: &crate::types::AgentState,
    checkpoint: &str,
) -> Result<(), InternalTurnResult>
where
    S: StateStore,
{
    if let Err(error) = state_store.save(state).await {
        return Err(InternalTurnResult::Error(AgentError::new(
            format!("Strict durability: failed to save {checkpoint} state checkpoint: {error}"),
            false,
        )));
    }
    Ok(())
}

pub(super) async fn execute_turn<Ctx, P, H, M, S>(
    params: ExecuteTurnParameters<'_, Ctx, P, H, M, S>,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let ExecuteTurnParameters {
        event_store,
        authority,
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        provenance,
        turn_options,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    } = params;

    #[cfg(feature = "otel")]
    let turn_number = ctx.turn + 1; // begin_turn increments
    #[cfg(feature = "otel")]
    let usage_before_turn = ctx.total_usage.clone();
    #[cfg(feature = "otel")]
    let turn_started_at = std::time::Instant::now();
    #[cfg(feature = "otel")]
    let provider_name_for_turn =
        crate::observability::provider_name::normalize(provider.provider());
    #[cfg(feature = "otel")]
    let turn_input_kind = ctx.input_kind;

    let result = execute_turn_inner(ExecuteTurnParameters {
        event_store,
        authority,
        ctx: &mut *ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        provenance,
        turn_options,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    })
    .await;

    #[cfg(feature = "otel")]
    record_turn_span(&RecordTurnSpanParams {
        turn_number,
        total_usage: &ctx.total_usage,
        usage_before_turn: &usage_before_turn,
        result: &result,
        turn_started_at,
        provider_name: provider_name_for_turn,
        input_kind: turn_input_kind,
    });

    result
}

#[cfg(feature = "otel")]
#[derive(Clone, Copy)]
struct RecordTurnSpanParams<'a> {
    turn_number: usize,
    total_usage: &'a TokenUsage,
    usage_before_turn: &'a TokenUsage,
    result: &'a InternalTurnResult,
    /// Start of `execute_turn` (so the histogram excludes wait time
    /// the agent spent waiting on us upstream).
    turn_started_at: std::time::Instant,
    provider_name: &'static str,
    input_kind: &'static str,
}

#[cfg(feature = "otel")]
fn record_turn_span(params: &RecordTurnSpanParams<'_>) {
    use crate::observability::{attrs, baggage, metrics, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    let RecordTurnSpanParams {
        turn_number,
        total_usage,
        usage_before_turn,
        result,
        turn_started_at,
        provider_name,
        input_kind,
    } = *params;

    let turn_usage = token_usage_delta(total_usage, usage_before_turn);
    let mut turn_span = spans::start_internal_span(
        "agent.turn",
        vec![attrs::kv_i64(
            attrs::SDK_TURN_NUMBER,
            i64::try_from(turn_number).unwrap_or(0),
        )],
    );
    baggage::copy_baggage_to_active_span(&mut turn_span);

    turn_span.set_attribute(attrs::kv_i64(
        attrs::SDK_TURN_INPUT_TOKENS,
        i64::from(turn_usage.input_tokens),
    ));
    turn_span.set_attribute(attrs::kv_i64(
        attrs::SDK_TURN_OUTPUT_TOKENS,
        i64::from(turn_usage.output_tokens),
    ));
    turn_span.set_attribute(attrs::kv_i64(
        attrs::SDK_TURN_CACHE_READ_INPUT_TOKENS,
        i64::from(turn_usage.cached_input_tokens),
    ));
    turn_span.set_attribute(attrs::kv_i64(
        attrs::SDK_TURN_CACHE_CREATION_INPUT_TOKENS,
        i64::from(turn_usage.cache_creation_input_tokens),
    ));

    let outcome = match result {
        InternalTurnResult::Continue { turn_usage } => {
            let had_tools = turn_usage.input_tokens > 0 || turn_usage.output_tokens > 0;
            turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "continue"));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, had_tools));
            "continue"
        }
        InternalTurnResult::Done => {
            turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "done"));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            "done"
        }
        InternalTurnResult::Refusal => {
            turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "refusal"));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            "refusal"
        }
        InternalTurnResult::AwaitingConfirmation { .. } => {
            turn_span.set_attribute(KeyValue::new(
                attrs::SDK_TURN_STOP_REASON,
                "awaiting_confirmation",
            ));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, true));
            "awaiting_confirmation"
        }
        InternalTurnResult::PendingToolCalls { .. } => {
            turn_span.set_attribute(KeyValue::new(
                attrs::SDK_TURN_STOP_REASON,
                "pending_tool_calls",
            ));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, true));
            "pending_tool_calls"
        }
        InternalTurnResult::Error(err) => {
            turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "error"));
            spans::set_span_error(&mut turn_span, "turn_error", &err.message);
            "error"
        }
        InternalTurnResult::Cancelled { .. } => {
            turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "cancelled"));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            "cancelled"
        }
    };
    turn_span.end();

    let elapsed_secs = turn_started_at.elapsed().as_secs_f64();
    let metrics_handle = metrics::Metrics::global();
    metrics_handle.turns_duration.record(
        elapsed_secs,
        &[
            KeyValue::new(attrs::SDK_OUTCOME, outcome),
            KeyValue::new(attrs::SDK_INPUT_KIND, input_kind),
            KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, provider_name),
        ],
    );
}

async fn execute_turn_inner<Ctx, P, H, M, S>(
    ExecuteTurnParameters {
        event_store,
        authority,
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        provenance,
        turn_options,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    }: ExecuteTurnParameters<'_, Ctx, P, H, M, S>,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    if let Err(error) = begin_turn(ctx, config.max_turns, event_store, hooks, authority).await {
        return InternalTurnResult::Error(error);
    }

    let messages = match load_turn_messages(TurnMessageLoadParams {
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    })
    .await
    {
        Ok(LoadedMessages::Ready(messages)) => messages,
        Ok(LoadedMessages::Cancelled) => {
            return InternalTurnResult::Cancelled {
                turn_usage: TokenUsage::default(),
            };
        }
        Err(error) => return InternalTurnResult::Error(error),
    };

    // Inject the turn-budget reminder as a user message if one was set.
    let mut messages = messages;
    if let Some(reminder) = ctx.pending_reminder.take() {
        messages.push(Message::user(reminder));
    }

    let request = match build_turn_request(config, provider, &ctx.thread_id, messages, tools) {
        Ok(request) => request,
        Err(error) => return InternalTurnResult::Error(error),
    };
    log_chat_request(&request);

    // Strict durability: checkpoint before the LLM call. `begin_turn` has
    // already advanced `turn_count`, so persisting here lets a crash during
    // the (potentially long) LLM round-trip resume from the correct turn.
    if turn_options.strict_durability
        && let Err(result) = save_strict_checkpoint(state_store, &ctx.state, "pre-LLM").await
    {
        return result;
    }

    let TurnLlmResponse {
        response,
        message_id,
        thinking_id,
    } = match request_turn_response(TurnLlmRequestParams {
        ctx: &mut *ctx,
        request,
        config,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    })
    .await
    {
        Ok(response) => response,
        Err(result) => return result,
    };

    process_response_and_run_tools(
        response,
        &message_id,
        &thinking_id,
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        provenance,
        turn_options,
        event_store,
        authority,
        cancel_token,
    )
    .await
}

/// The LLM response for a turn, paired with the final-attempt streaming
/// ids the caller reuses for the post-stream Text/Refusal events.
struct TurnLlmResponse {
    response: ChatResponse,
    message_id: String,
    thinking_id: String,
}

struct TurnLlmRequestParams<'a, P, H, M> {
    ctx: &'a mut TurnContext,
    request: ChatRequest,
    config: &'a AgentConfig,
    provider: &'a Arc<P>,
    message_store: &'a Arc<M>,
    compaction_config: Option<&'a CompactionConfig>,
    compactor: Option<&'a Arc<dyn ContextCompactor>>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    cancel_token: &'a CancellationToken,
    #[cfg(feature = "otel")]
    observability_store: Option<&'a Arc<dyn crate::observability::ObservabilityStore>>,
}

/// Generate the per-attempt streaming ids, call the LLM, and decode the
/// outcome.
///
/// `message_id` / `thinking_id` are regenerated per attempt by the
/// streaming retry loop (see `call_llm_streaming`) so each attempt's
/// deltas land under a distinct id; the returned ids are the final
/// attempt's, which the post-stream Text/Refusal events reuse so they
/// stay correlated with the surviving deltas. A cancellation, or an
/// unrecoverable error (after attempting an emergency compaction for a
/// prompt-too-long error), is mapped onto the `InternalTurnResult` the
/// caller should return.
async fn request_turn_response<P, H, M>(
    TurnLlmRequestParams {
        ctx,
        request,
        config,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    }: TurnLlmRequestParams<'_, P, H, M>,
) -> Result<TurnLlmResponse, InternalTurnResult>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let mut message_id = uuid::Uuid::new_v4().to_string();
    let mut thinking_id = uuid::Uuid::new_v4().to_string();
    let response = match request_llm_response(LlmCallParams {
        provider,
        request,
        config,
        event_store,
        hooks,
        authority,
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        message_id: &mut message_id,
        thinking_id: &mut thinking_id,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    })
    .await
    {
        LlmOutcome::Response(response) => response,
        LlmOutcome::Cancelled => {
            // The cancel was honored before any assistant message was
            // persisted, so there is no orphan `tool_use` to balance —
            // history is already consistent. Surface a partial-usage
            // `Cancelled` so the run closes with the terminal event.
            return Err(InternalTurnResult::Cancelled {
                turn_usage: TokenUsage::default(),
            });
        }
        LlmOutcome::Error(error) => {
            if let Some(result) = try_recover_prompt_too_long(
                &error,
                ctx,
                compaction_config,
                compactor,
                provider,
                message_store,
                cancel_token,
            )
            .await
            {
                return Err(result);
            }
            return Err(InternalTurnResult::Error(error));
        }
    };

    Ok(TurnLlmResponse {
        response,
        message_id,
        thinking_id,
    })
}

/// Build the `PendingToolCalls` result that hands an external tool
/// runtime the pending calls (and a resumable continuation) instead of
/// executing them inline.
fn external_pending_tool_calls_result(
    ctx: &TurnContext,
    turn_usage: TokenUsage,
    pending_tool_calls: Vec<PendingToolCallInfo>,
) -> InternalTurnResult {
    let continuation = AgentContinuation {
        thread_id: ctx.thread_id.clone(),
        turn: ctx.turn,
        total_usage: ctx.total_usage.clone(),
        turn_usage: turn_usage.clone(),
        pending_tool_calls: pending_tool_calls.clone(),
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: ctx.state.clone(),
        response_id: ctx.response_id.clone(),
        stop_reason: ctx.stop_reason,
        response_content: Vec::new(),
    };
    InternalTurnResult::PendingToolCalls {
        turn_usage,
        pending_tool_calls,
        continuation: Box::new(continuation),
    }
}

#[expect(clippy::too_many_arguments)]
async fn process_response_and_run_tools<Ctx, P, H, M, S>(
    response: ChatResponse,
    message_id: &str,
    thinking_id: &str,
    ctx: &mut TurnContext,
    tool_context: &ToolContext<Ctx>,
    provider: &Arc<P>,
    tools: &Arc<ToolRegistry<Ctx>>,
    hooks: &Arc<H>,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
    compaction_config: Option<&CompactionConfig>,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    audit_sink: &Arc<dyn ToolAuditSink>,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
    event_store: &Arc<dyn EventStore>,
    authority: &Arc<dyn EventAuthority>,
    cancel_token: &CancellationToken,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let turn_usage = apply_turn_usage(ctx, &response);
    let ProcessedTurnResponse {
        stop_reason,
        text_content,
        pending_tool_calls,
    } = match process_turn_response(TurnResponseProcessingParams {
        response,
        message_id,
        thinking_id,
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        tools,
        message_store,
        event_store,
        hooks,
        authority,
    })
    .await
    {
        Ok(processed) => processed,
        Err(error) => return InternalTurnResult::Error(error),
    };

    // Record how many tool calls the LLM asked for in this turn so
    // the summary can report it without reparsing the message history.
    ctx.tool_call_count = pending_tool_calls.len();

    // A successful, non-overflow LLM call resets the consecutive
    // context-overflow retry budget. `MAX_COMPACTION_RETRIES` is meant to
    // bound a single overflow *episode* (repeated failed compaction retries
    // for the same oversized prompt), not the whole run: without this reset
    // the Nth overflow anywhere in a long run hard-fails even when every
    // prior compaction succeeded and many turns ran in between.
    if stop_reason != Some(StopReason::ModelContextWindowExceeded) {
        ctx.compaction_retries = 0;
    }

    let had_tool_calls = !pending_tool_calls.is_empty();

    // Strict durability: checkpoint after LLM response, before tool execution.
    // A failed checkpoint in strict mode violates the crash-safe contract — a
    // crash here would resume from stale state (wrong turn_count / usage) — so
    // it is surfaced as a hard error rather than warn-and-continue.
    if turn_options.strict_durability {
        if let Err(result) = save_strict_checkpoint(state_store, &ctx.state, "post-LLM").await {
            return result;
        }
    } else if had_tool_calls && let Err(error) = state_store.save(&ctx.state).await {
        warn!("Failed to save pre-tool state checkpoint: {error}");
    }

    // External tool runtime: return pending tool calls to the caller instead
    // of executing them inline.
    if had_tool_calls && turn_options.tool_runtime == ToolRuntime::External {
        return external_pending_tool_calls_result(ctx, turn_usage, pending_tool_calls);
    }

    if let Err(outcome) = execute_turn_tool_phase(TurnToolPhaseParams {
        pending_tool_calls,
        tool_context,
        thread_id: &ctx.thread_id,
        tools,
        hooks,
        event_store,
        authority,
        execution_store,
        audit_sink,
        provenance,
        turn: ctx.turn,
        total_usage: &ctx.total_usage,
        turn_usage: &turn_usage,
        state: &ctx.state,
        message_store,
        response_id: ctx.response_id.clone(),
        stop_reason: ctx.stop_reason,
    })
    .await
    {
        return outcome;
    }

    // Strict durability: checkpoint after tool execution. As with the
    // post-LLM checkpoint, a failed save in strict mode is a hard error so
    // the server never proceeds believing a checkpoint exists when it does not.
    if turn_options.strict_durability
        && let Err(result) = save_strict_checkpoint(state_store, &ctx.state, "post-tool").await
    {
        return result;
    }

    handle_turn_stop_reason(TurnStopReasonParams {
        stop_reason,
        text_content,
        had_tool_calls,
        message_id: message_id.to_string(),
        turn_usage,
        ctx,
        provider,
        message_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        cancel_token,
    })
    .await
}
