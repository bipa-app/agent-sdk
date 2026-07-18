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
    ToolResult, ToolRuntime, ToolTier, TurnOptions, UsageLimits,
};
use agent_sdk_foundation::audit::AuditProvenance;

use futures::StreamExt;
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
    Ready {
        messages: Vec<Message>,
        /// Provider-billed usage of the compaction summarization call(s)
        /// that ran while loading (zero when no compaction happened —
        /// including failed attempts whose billed calls still count).
        /// Folded into the run's cumulative usage by the caller so
        /// compaction spend is visible to budgets and `Done` reporting.
        compaction_usage: TokenUsage,
    },
    Cancelled {
        /// Usage billed by summarization calls that completed before the
        /// cancel was honored (zero when the cancel landed mid-call).
        compaction_usage: TokenUsage,
    },
}

/// A failed history load, carrying any compaction spend billed before the
/// failure.
///
/// The critical case: summarization AND `replace_history` succeeded, then
/// the `ContextCompacted` event append failed — the run terminates on the
/// error, but the billed summarization tokens must still reach the
/// cumulative totals or they vanish from usage/cost accounting.
pub(super) struct LoadFailure {
    pub(super) error: AgentError,
    pub(super) compaction_usage: TokenUsage,
}

impl LoadFailure {
    /// A failure with no billed compaction spend.
    fn unbilled(error: AgentError) -> Self {
        Self {
            error,
            compaction_usage: TokenUsage::default(),
        }
    }
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
) -> Result<LoadedMessages, LoadFailure>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let messages = match message_store.get_history(thread_id).await {
        Ok(m) => m,
        Err(error) => {
            if let Err(send_error) = send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                authority,
                AgentEvent::error(format!("Failed to get history: {error}"), false),
            )
            .await
            {
                return Err(LoadFailure::unbilled(send_error));
            }
            return Err(LoadFailure::unbilled(AgentError::new(
                format!("Failed to get history: {error}"),
                false,
            )));
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
) -> Result<LoadedMessages, LoadFailure>
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
        return Ok(LoadedMessages::Ready {
            messages,
            compaction_usage: TokenUsage::default(),
        });
    }

    if let Some(compact_config) = compaction_config {
        // Attach the run's hooks so the summarization call passes the same
        // pre_llm_request / on_llm_response guardrails as regular turns.
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compact_config.clone())
                .with_guardrail_hooks(Arc::clone(hooks));
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

    Ok(LoadedMessages::Ready {
        messages,
        compaction_usage: TokenUsage::default(),
    })
}

struct StoredCompactionResult {
    messages: Vec<Message>,
    original_count: usize,
    new_count: usize,
    original_tokens: usize,
    new_tokens: usize,
    /// Provider-billed usage of the summarization call(s); see
    /// [`crate::context::CompactionResult::llm_usage`].
    llm_usage: TokenUsage,
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
///
/// Every variant carries the provider-billed usage of summarization calls
/// that actually ran: a failed or cancelled-after-summarize attempt was
/// still billed, and cumulative usage/cost must include every billed call
/// even when the history is left untouched.
enum CompactionOutcome {
    Stored(StoredCompactionResult),
    /// Cancel fired before the destructive `replace_history` write —
    /// history is untouched.
    Cancelled {
        llm_usage: TokenUsage,
    },
    Failed {
        error: AgentError,
        llm_usage: TokenUsage,
    },
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
) -> Result<LoadedMessages, LoadFailure>
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
            if let Err(error) = send_event(
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
            .await
            {
                // Compaction itself succeeded (history replaced, the
                // summarization billed): the spend must survive this
                // terminal error path so it still reaches the totals.
                return Err(LoadFailure {
                    error,
                    compaction_usage: result.llm_usage,
                });
            }

            info!(
                "Context compacted successfully (original_count={}, new_count={}, original_tokens={}, new_tokens={})",
                result.original_count, result.new_count, result.original_tokens, result.new_tokens
            );

            Ok(LoadedMessages::Ready {
                messages: result.messages,
                compaction_usage: result.llm_usage,
            })
        }
        CompactionOutcome::Cancelled { llm_usage } => Ok(LoadedMessages::Cancelled {
            compaction_usage: llm_usage,
        }),
        CompactionOutcome::Failed { error, llm_usage } => {
            warn!("Context compaction failed, continuing with full history: {error}");
            // The failed attempt's summarization calls were still billed —
            // surface their usage so the turn folds it into the totals.
            Ok(LoadedMessages::Ready {
                messages: original_messages,
                compaction_usage: llm_usage,
            })
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
            // The in-flight summarization future was dropped mid-call; no
            // usable usage was observed.
            return CompactionOutcome::Cancelled {
                llm_usage: TokenUsage::default(),
            };
        }
        res = compactor.compact_history_with_usage(messages) => match res {
            Ok(result) => result,
            Err(failure) => {
                #[cfg(feature = "otel")]
                finish_compaction_span_error(
                    &mut compaction_span,
                    "context_compaction_failed",
                    &failure.error.to_string(),
                );
                // Summarization calls made before the failure were still
                // billed; carry their usage so the caller can account it.
                return CompactionOutcome::Failed {
                    error: AgentError::new(
                        format!("{compaction_error_prefix}: {}", failure.error),
                        false,
                    ),
                    llm_usage: failure.llm_usage,
                };
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
        // Summarization completed (and was billed) before the cancel won.
        return CompactionOutcome::Cancelled {
            llm_usage: result.llm_usage,
        };
    }

    let stored_result = StoredCompactionResult {
        messages: result.messages,
        original_count: result.original_count,
        new_count: result.new_count,
        original_tokens: result.original_tokens,
        new_tokens: result.new_tokens,
        llm_usage: result.llm_usage,
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
        return CompactionOutcome::Failed {
            error: AgentError::new(format!("{replace_history_error_prefix}: {error}"), false),
            llm_usage: stored_result.llm_usage,
        };
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
                        ContentBlock::OpaqueReasoning { provider, .. } => {
                            debug!(
                                "    block[{block_idx}]: OpaqueReasoning(provider={provider}, payload=<redacted>)"
                            );
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

/// Run the `on_llm_response` output guardrail on a successful LLM outcome.
///
/// Runs here, BEFORE any observability payload capture, because the hook
/// contract promises the decision fires before the response is persisted or
/// surfaced — and an externally-persisting `ObservabilityStore` is
/// persistence. The hook runs exactly once; the returned decision is handed
/// to the caller, which handles usage accounting, retry caps, and feedback
/// without re-invoking the hook. `None` iff the outcome carries no response.
async fn output_guardrail_decision<H>(
    hooks: &Arc<H>,
    result: &LlmOutcome,
) -> Option<super::llm::PostLlmGuardrail>
where
    H: AgentHooks,
{
    match result {
        LlmOutcome::Response(response) => {
            Some(super::llm::apply_on_llm_response(hooks, response).await)
        }
        LlmOutcome::Cancelled(_) | LlmOutcome::Error(_) => None,
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
) -> (LlmOutcome, Option<super::llm::PostLlmGuardrail>)
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

    let response_decision = output_guardrail_decision(hooks, &result).await;

    // Project the three-state outcome onto a `Result` for the OTel
    // span / payload bookkeeping. A cancellation is recorded as an
    // `AgentError` purely so the span carries a stable, filterable
    // terminal status; the caller still receives the original
    // `LlmOutcome::Cancelled` and routes it to the `Cancelled` event.
    #[cfg(feature = "otel")]
    let span_result: Result<ChatResponse, AgentError> = match &result {
        LlmOutcome::Response(response) => Ok(response.clone()),
        LlmOutcome::Cancelled(_) => Err(AgentError::new("LLM call cancelled", false)),
        LlmOutcome::Error(error) => Err(error.clone()),
    };

    #[cfg(feature = "otel")]
    if let (Some(observability_store), Some(request), Ok(response)) = (
        observability_store,
        request_for_capture.as_ref(),
        span_result.as_ref(),
    ) {
        if matches!(
            response_decision,
            Some(super::llm::PostLlmGuardrail::Accept)
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
        } else {
            // A Blocked / RetryWithFeedback response must not escape to an
            // externally-persisting store. Non-payload metrics (duration,
            // status) are still recorded by `finish_llm_span` below.
            use opentelemetry::trace::Span;
            llm_span.add_event("payload_capture_skipped_by_guardrail", vec![]);
        }
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

    (result, response_decision)
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

/// Fold one LLM call's usage delta into the run totals and the state's
/// accumulated cost, priced at the provenance that served the call.
///
/// This is the single anchor where usage is committed — once per provider
/// response (turn calls, guardrail-rejected calls) and once per compaction
/// summarization — so cost accumulation cannot double-count on retry paths.
pub(super) fn fold_llm_usage(
    ctx: &mut TurnContext,
    provenance: &AuditProvenance,
    delta: &TokenUsage,
) {
    fold_scoped_usage(ctx, provenance, delta, super::budget::UsageScope::Call);
}

/// Fold a compaction's summarization spend into the run totals and cost.
///
/// Distinct from [`fold_llm_usage`] because a compaction may bill two
/// summarization calls, summed into one delta when a truncated summary is
/// retried (see [`crate::context::CompactionResult::llm_usage`]). Pricing that
/// sum as a single call could select a long-context tier neither call reached,
/// so it is priced at base rates.
pub(super) fn fold_compaction_usage(
    ctx: &mut TurnContext,
    provenance: &AuditProvenance,
    delta: &TokenUsage,
) {
    fold_scoped_usage(ctx, provenance, delta, super::budget::UsageScope::Aggregate);
}

/// The single anchor where usage is committed: fold `delta` into the run
/// totals and the state's accumulated cost, priced according to `scope`.
fn fold_scoped_usage(
    ctx: &mut TurnContext,
    provenance: &AuditProvenance,
    delta: &TokenUsage,
    scope: super::budget::UsageScope,
) {
    super::budget::accumulate_cost(
        &mut ctx.state,
        ctx.cost_estimator.as_deref(),
        provenance,
        &ctx.total_usage,
        delta,
        scope,
    );
    ctx.total_usage.add(delta);
    ctx.state.total_usage = ctx.total_usage.clone();
}

pub(super) fn apply_turn_usage(
    ctx: &mut TurnContext,
    provenance: &AuditProvenance,
    response: &ChatResponse,
) -> TokenUsage {
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
    };
    fold_llm_usage(ctx, provenance, &turn_usage);

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
    let accepts_tool_uses = response_allows_tool_execution(stop_reason);

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

    // A provider may produce a malformed or partially assembled response
    // containing tool-use blocks while explicitly reporting a terminal
    // reason such as `end_turn` or `max_tokens`.  Treat the explicit reason
    // as authoritative: neither execute nor persist those tool uses.  Leaving
    // them in history without matching results would poison subsequent calls.
    if !accepts_tool_uses && !tool_uses.is_empty() {
        warn!(
            "Ignoring {} tool-use block(s) paired with terminal stop reason {:?}",
            tool_uses.len(),
            stop_reason
        );
    }

    let assistant_msg = if accepts_tool_uses {
        build_assistant_message(&response)
    } else {
        build_terminal_assistant_message(&response)
    };
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
        pending_tool_calls: if accepts_tool_uses {
            build_pending_tool_calls(tools, &tool_uses)
        } else {
            Vec::new()
        },
    })
}

/// Whether a response may initiate tool execution.
///
/// An explicit terminal stop reason is authoritative even if a provider also
/// emitted `ToolUse` blocks.  `None` remains accepted for compatibility with
/// legacy providers that did not report a stop reason for tool calls.
pub(super) const fn response_allows_tool_execution(stop_reason: Option<StopReason>) -> bool {
    matches!(stop_reason, None | Some(StopReason::ToolUse))
}

/// Build a terminal assistant message without tool-use blocks.
///
/// Terminal responses cannot have their tool uses executed, so retaining them
/// would create an orphaned tool-use entry in persisted conversation history.
fn build_terminal_assistant_message(response: &ChatResponse) -> Message {
    let mut terminal_response = response.clone();
    terminal_response
        .content
        .retain(|block| !matches!(block, ContentBlock::ToolUse { .. }));
    build_assistant_message(&terminal_response)
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
        max_parallel_tools,
        reminder_config,
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

            // Adjacent observe-tier calls run concurrently, but the
            // in-flight count is bounded by `max_parallel_tools`
            // (`None` = unbounded for the batch, `Some(1)` = sequential).
            // `buffered` polls up to `concurrency` futures at once and
            // yields results in input order, so downstream ordering is
            // unchanged regardless of the cap.
            //
            // The slice is borrowed only for the duration of the stream —
            // the borrow ends before we touch `pending_tool_calls` mutably
            // below.
            let batch_len = end - idx;
            let concurrency =
                max_parallel_tools.map_or(batch_len, |cap| cap.clamp(1, batch_len.max(1)));
            // Collect the per-call futures eagerly (as `join_all` did) so the
            // borrow of `execution_ctx` is captured by each future directly,
            // then drive them with bounded concurrency. `buffered` yields in
            // input order so result ordering is unchanged.
            let batch_futures: Vec<_> = pending_tool_calls[idx..end]
                .iter()
                .map(|p| execute_tool_call(p, &execution_ctx))
                .collect();
            let outcomes: Vec<ToolExecutionOutcome> = futures::stream::iter(batch_futures)
                .buffered(concurrency)
                .collect()
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

    if let Some(config) = reminder_config {
        apply_tool_reminders(config, &pending_tool_calls, &mut tool_results);
    }

    Ok(tool_results)
}

/// Append any configured per-tool reminders to the matching tool results.
///
/// For each completed result, looks up the originating tool's
/// [`ToolReminder`](crate::reminders::ToolReminder)s by name and, for every
/// one whose [`ReminderTrigger`](crate::reminders::ReminderTrigger) fires
/// against the requested input and the produced result, appends the reminder
/// (wrapped in `<system-reminder>` tags) to the result the model sees.
fn apply_tool_reminders(
    config: &crate::reminders::ReminderConfig,
    pending_tool_calls: &[PendingToolCallInfo],
    tool_results: &mut [(String, ToolResult)],
) {
    if !config.enabled || config.tool_reminders.is_empty() {
        return;
    }
    for (tool_id, result) in tool_results.iter_mut() {
        let Some(pending) = pending_tool_calls.iter().find(|p| p.id == *tool_id) else {
            continue;
        };
        let Some(reminders) = config.tool_reminders.get(&pending.name) else {
            continue;
        };
        // Evaluate every trigger against an immutable snapshot of the
        // ORIGINAL result: appending a fired reminder mutates the result,
        // and a later `ResultContains` trigger must not fire because an
        // earlier reminder's appended text matched its pattern.
        let original = result.clone();
        for reminder in reminders {
            if reminder.trigger.should_trigger(&pending.input, &original) {
                crate::reminders::append_reminder(result, &reminder.content);
            }
        }
    }
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
        AgentEvent::turn_complete(turn, turn_usage.clone()),
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
        max_parallel_tools,
        reminder_config,
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
        max_parallel_tools,
        reminder_config,
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
    /// Compaction stored a summarized history; carries the provider-billed
    /// usage of the summarization call(s) so callers can fold it into the
    /// run's cumulative usage.
    Done(TokenUsage),
    /// Cancel fired during the emergency compaction; history untouched.
    /// Carries any usage billed before the cancel was honored.
    Cancelled(TokenUsage),
    /// The recovery failed; carries any usage billed before the failure.
    Failed {
        error: AgentError,
        llm_usage: TokenUsage,
    },
}

pub(super) async fn compact_after_context_overflow<P, H, M>(
    provider: &Arc<P>,
    hooks: &Arc<H>,
    compaction_config: &CompactionConfig,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    message_store: &Arc<M>,
    thread_id: &ThreadId,
    cancel_token: &CancellationToken,
) -> OverflowCompaction
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let history = match message_store.get_history(thread_id).await {
        Ok(history) => history,
        Err(error) => {
            return OverflowCompaction::Failed {
                error: AgentError::new(
                    format!("Failed to get history for compaction after context overflow: {error}"),
                    false,
                ),
                llm_usage: TokenUsage::default(),
            };
        }
    };

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
        // Attach the run's hooks so the summarization call passes the same
        // pre_llm_request / on_llm_response guardrails as regular turns.
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compaction_config.clone())
                .with_guardrail_hooks(Arc::clone(hooks));
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
            OverflowCompaction::Done(result.llm_usage)
        }
        CompactionOutcome::Cancelled { llm_usage } => OverflowCompaction::Cancelled(llm_usage),
        CompactionOutcome::Failed { error, llm_usage } => {
            OverflowCompaction::Failed { error, llm_usage }
        }
    }
}

/// Inputs shared by the two overflow-recovery entry points
/// ([`handle_context_window_exceeded`] / [`try_recover_prompt_too_long`]).
struct OverflowRecoveryParams<'a, P, H, M> {
    ctx: &'a mut TurnContext,
    provider: &'a Arc<P>,
    hooks: &'a Arc<H>,
    message_store: &'a Arc<M>,
    compaction_config: Option<&'a CompactionConfig>,
    compactor: Option<&'a Arc<dyn ContextCompactor>>,
    /// Run provenance, used to price the emergency compaction's usage.
    provenance: &'a AuditProvenance,
    /// Event plumbing + limits for the pre-recovery budget gate: when the
    /// overflow turn's own (already folded) usage crossed a limit, the
    /// recovery stops with a terminal budget event instead of paying for
    /// emergency summarization first.
    event_store: &'a Arc<dyn EventStore>,
    authority: &'a Arc<dyn EventAuthority>,
    usage_limits: Option<&'a UsageLimits>,
    cancel_token: &'a CancellationToken,
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
        provenance,
        usage_limits,
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
                turn_usage,
                OverflowRecoveryParams {
                    ctx,
                    provider,
                    hooks,
                    message_store,
                    compaction_config,
                    compactor,
                    provenance,
                    event_store,
                    authority,
                    usage_limits,
                    cancel_token,
                },
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

async fn handle_context_window_exceeded<P, H, M>(
    turn_usage: TokenUsage,
    OverflowRecoveryParams {
        ctx,
        provider,
        hooks,
        message_store,
        compaction_config,
        compactor,
        provenance,
        event_store,
        authority,
        usage_limits,
        cancel_token,
    }: OverflowRecoveryParams<'_, P, H, M>,
) -> InternalTurnResult
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    warn!("Model context window exceeded (turn={})", ctx.turn);
    // The overflow turn's own usage was already folded by
    // `apply_turn_usage`; if it crossed a limit, stop with the terminal
    // budget outcome instead of paying for emergency summarization first.
    if let Some((limit, cost)) = super::budget::status(
        usage_limits,
        ctx.cost_estimator.as_deref(),
        provenance,
        &ctx.total_usage,
        ctx.state.accumulated_cost_usd,
    ) {
        // The overflow turn's own (already-folded) usage is the stopping
        // turn's usage: carry it so the TurnSummary agrees with the totals.
        return budget_exceeded_mid_turn(
            ctx,
            event_store,
            hooks,
            authority,
            limit,
            cost,
            turn_usage,
        )
        .await;
    }
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
            hooks,
            compact_config,
            compactor,
            message_store,
            &ctx.thread_id,
            cancel_token,
        )
        .await
        {
            OverflowCompaction::Done(compaction_usage) => {
                fold_compaction_usage(ctx, provenance, &compaction_usage);
            }
            OverflowCompaction::Cancelled(compaction_usage) => {
                fold_compaction_usage(ctx, provenance, &compaction_usage);
                return InternalTurnResult::Cancelled { turn_usage };
            }
            OverflowCompaction::Failed { error, llm_usage } => {
                // Billed-but-wasted summarization spend still counts.
                fold_compaction_usage(ctx, provenance, &llm_usage);
                return InternalTurnResult::Error(error);
            }
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
        || lower.contains("exceeds the context window")
        || lower.contains("input is too long")
        || lower.contains("request too large")
}

/// When the prompt exceeds the model's context window at the API level (returned as a 400 error
/// rather than a `stop_reason`), attempt compaction and retry instead of failing immediately.
async fn try_recover_prompt_too_long<P, H, M>(
    error: &AgentError,
    // The budget-gate fields are deliberately unused here: this recovery
    // runs after a FAILED LLM call, which bills nothing the SDK accounts,
    // so no new spend can have accrued since the checks that already ran
    // (loop boundary + post-compaction recheck).
    OverflowRecoveryParams {
        ctx,
        provider,
        hooks,
        message_store,
        compaction_config,
        compactor,
        provenance,
        cancel_token,
        ..
    }: OverflowRecoveryParams<'_, P, H, M>,
) -> Option<InternalTurnResult>
where
    P: LlmProvider,
    H: AgentHooks,
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
            hooks,
            compact_config,
            compactor,
            message_store,
            &ctx.thread_id,
            cancel_token,
        )
        .await
        {
            OverflowCompaction::Done(compaction_usage) => {
                fold_compaction_usage(ctx, provenance, &compaction_usage);
            }
            OverflowCompaction::Cancelled(compaction_usage) => {
                fold_compaction_usage(ctx, provenance, &compaction_usage);
                return Some(InternalTurnResult::Cancelled {
                    turn_usage: TokenUsage::default(),
                });
            }
            OverflowCompaction::Failed { error, llm_usage } => {
                // Billed-but-wasted summarization spend still counts.
                fold_compaction_usage(ctx, provenance, &llm_usage);
                return Some(InternalTurnResult::Error(error));
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
        reminder_config,
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
        reminder_config,
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
        InternalTurnResult::BudgetExceeded { .. } => {
            turn_span.set_attribute(KeyValue::new(
                attrs::SDK_TURN_STOP_REASON,
                "budget_exceeded",
            ));
            turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            "budget_exceeded"
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

/// Fold threshold-compaction spend into the run totals and re-evaluate the
/// budget.
///
/// Compaction summarization is a paid LLM call on this run's provider: its
/// usage (and cost, priced at the run's provenance) must land in the
/// cumulative totals so budgets see it and `Done` does not under-report.
/// The loop-boundary budget check ran BEFORE compaction, so if the
/// summarization spend itself crossed a limit this returns the mid-turn
/// terminal result and the main-model call is never paid for.
async fn fold_compaction_spend<H>(
    ctx: &mut TurnContext,
    compaction_usage: &TokenUsage,
    usage_limits: Option<&UsageLimits>,
    provenance: &AuditProvenance,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
) -> Option<InternalTurnResult>
where
    H: AgentHooks,
{
    fold_compaction_usage(ctx, provenance, compaction_usage);
    if !super::budget::usage_is_zero(compaction_usage)
        && let Some((limit, cost)) = super::budget::status(
            usage_limits,
            ctx.cost_estimator.as_deref(),
            provenance,
            &ctx.total_usage,
            ctx.state.accumulated_cost_usd,
        )
    {
        return Some(
            budget_exceeded_mid_turn(
                ctx,
                event_store,
                hooks,
                authority,
                limit,
                cost,
                // Compaction-only stop: the turn made no main LLM call, so
                // the per-turn summary carries zero (the compaction spend
                // rides in the cumulative totals).
                TokenUsage::default(),
            )
            .await,
        );
    }
    None
}

/// Load (and possibly compact) the turn's history, folding any compaction
/// spend into the run totals and re-evaluating the budget.
///
/// `Err` carries the `InternalTurnResult` the turn must return instead:
/// a mid-turn budget stop, a cancellation during compaction (whose billed
/// summarization usage is still folded), or a load error.
async fn load_turn_messages_accounted<P, H, M>(
    load_params: TurnMessageLoadParams<'_, P, H, M>,
    ctx: &mut TurnContext,
    usage_limits: Option<&UsageLimits>,
    provenance: &AuditProvenance,
) -> Result<Vec<Message>, InternalTurnResult>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let event_store = load_params.event_store;
    let hooks = load_params.hooks;
    let authority = load_params.authority;
    match load_turn_messages(load_params).await {
        Ok(LoadedMessages::Ready {
            messages,
            compaction_usage,
        }) => {
            if let Some(stop) = fold_compaction_spend(
                ctx,
                &compaction_usage,
                usage_limits,
                provenance,
                event_store,
                hooks,
                authority,
            )
            .await
            {
                return Err(stop);
            }
            Ok(messages)
        }
        Ok(LoadedMessages::Cancelled { compaction_usage }) => {
            // A summarization that completed before the cancel won was
            // still billed: account it even though the turn is closing.
            fold_compaction_usage(ctx, provenance, &compaction_usage);
            Err(InternalTurnResult::Cancelled {
                turn_usage: TokenUsage::default(),
            })
        }
        Err(failure) => {
            // A failed load can still carry billed compaction spend (e.g.
            // the ContextCompacted event append failed AFTER summarization
            // and replace_history succeeded): fold it before surfacing the
            // error so the totals include every billed call.
            fold_compaction_usage(ctx, provenance, &failure.compaction_usage);
            Err(InternalTurnResult::Error(failure.error))
        }
    }
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
        reminder_config,
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

    // Snapshot the identity fields so the load params can borrow them while
    // `ctx` itself is handed to the helper mutably.
    let turn_thread_id = ctx.thread_id.clone();
    let turn_number = ctx.turn;
    let messages = match load_turn_messages_accounted(
        TurnMessageLoadParams {
            thread_id: &turn_thread_id,
            turn: turn_number,
            provider,
            message_store,
            compaction_config,
            compactor,
            event_store,
            hooks,
            authority,
            cancel_token,
        },
        ctx,
        config.usage_limits.as_ref(),
        provenance,
    )
    .await
    {
        Ok(messages) => messages,
        Err(result) => return result,
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
        state_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        turn_options,
        provenance,
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
        config,
        reminder_config,
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

struct TurnLlmRequestParams<'a, P, H, M, S> {
    ctx: &'a mut TurnContext,
    request: ChatRequest,
    config: &'a AgentConfig,
    provider: &'a Arc<P>,
    message_store: &'a Arc<M>,
    /// State store for the strict-durability checkpoint on the guardrail
    /// `RetryWithFeedback` path, which returns `Continue` without reaching
    /// the normal post-LLM checkpoint in `process_response_and_run_tools`.
    state_store: &'a Arc<S>,
    compaction_config: Option<&'a CompactionConfig>,
    compactor: Option<&'a Arc<dyn ContextCompactor>>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    turn_options: &'a TurnOptions,
    /// Run provenance: prices the usage committed by this call's response
    /// (including guardrail-rejected responses) and any emergency
    /// compaction it triggers.
    provenance: &'a AuditProvenance,
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
async fn request_turn_response<P, H, M, S>(
    TurnLlmRequestParams {
        ctx,
        request,
        config,
        provider,
        message_store,
        state_store,
        compaction_config,
        compactor,
        event_store,
        hooks,
        authority,
        turn_options,
        provenance,
        cancel_token,
        #[cfg(feature = "otel")]
        observability_store,
    }: TurnLlmRequestParams<'_, P, H, M, S>,
) -> Result<TurnLlmResponse, InternalTurnResult>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    // Input guardrail (`pre_llm_request`): runs once per LLM call, before
    // the retry-wrapped chat dispatch. For the FIRST call after fresh
    // `Text` / `Message` input the hook already ran at ingestion time —
    // BEFORE the user message was durably appended, so a Block left nothing
    // in history — and its decision is consumed here instead of
    // re-invoking the hook (exactly once per call). Every later call
    // evaluates the hook against the turn-built request as usual.
    let request = match ctx.pending_first_request.take() {
        Some(super::types::PreEvaluatedRequest::Proceed) => request,
        Some(super::types::PreEvaluatedRequest::Modified(modified)) => *modified,
        None => match super::llm::apply_pre_llm_request(hooks, request).await {
            super::llm::PreLlmGuardrail::Proceed(request) => *request,
            super::llm::PreLlmGuardrail::Blocked(reason) => {
                return Err(guardrail_block_result(
                    &ctx.thread_id,
                    ctx.turn,
                    event_store,
                    hooks,
                    authority,
                    "request",
                    &reason,
                )
                .await);
            }
        },
    };

    let mut message_id = uuid::Uuid::new_v4().to_string();
    let mut thinking_id = uuid::Uuid::new_v4().to_string();
    let (llm_outcome, response_decision) = request_llm_response(LlmCallParams {
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
    .await;
    let response = match llm_outcome {
        LlmOutcome::Response(response) => response,
        LlmOutcome::Cancelled(usage) => {
            // The cancel was honored before any assistant message was
            // persisted, so there is no orphan `tool_use` to balance —
            // history is already consistent. Surface a partial-usage
            // `Cancelled` so the run closes with the terminal event.
            //
            // The provider billed whatever it streamed before the cancel (and
            // any attempt abandoned by a retry before it), so those tokens are
            // committed at the same anchor a completed turn uses — a cancelled
            // run still reports what it spent.
            let turn_usage = TokenUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cached_input_tokens: usage.cached_input_tokens,
                cache_creation_input_tokens: usage.cache_creation_input_tokens,
            };
            fold_llm_usage(ctx, provenance, &turn_usage);
            return Err(InternalTurnResult::Cancelled { turn_usage });
        }
        LlmOutcome::Error(error) => {
            if let Some(result) = try_recover_prompt_too_long(
                &error,
                OverflowRecoveryParams {
                    ctx,
                    provider,
                    hooks,
                    message_store,
                    compaction_config,
                    compactor,
                    provenance,
                    event_store,
                    authority,
                    usage_limits: config.usage_limits.as_ref(),
                    cancel_token,
                },
            )
            .await
            {
                return Err(result);
            }
            return Err(InternalTurnResult::Error(error));
        }
    };

    // Output guardrail decision handling: the hook itself already ran
    // inside `request_llm_response` (BEFORE observability payload capture,
    // honoring the before-persist contract); here the pre-computed decision
    // drives usage accounting, retry caps, and feedback.
    let Some(decision) = response_decision else {
        return Err(InternalTurnResult::Error(AgentError::new(
            "output guardrail decision missing for a successful LLM response",
            false,
        )));
    };
    apply_response_guardrail(ResponseGuardrailParams {
        ctx,
        response: &response,
        decision,
        message_store,
        state_store,
        event_store,
        hooks,
        authority,
        turn_options,
        provenance,
    })
    .await?;

    Ok(TurnLlmResponse {
        response,
        message_id,
        thinking_id,
    })
}

struct ResponseGuardrailParams<'a, H, M, S> {
    ctx: &'a mut TurnContext,
    response: &'a ChatResponse,
    /// The `on_llm_response` decision, already produced by
    /// `request_llm_response` (the hook runs exactly once, before payload
    /// capture).
    decision: super::llm::PostLlmGuardrail,
    message_store: &'a Arc<M>,
    state_store: &'a Arc<S>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    turn_options: &'a TurnOptions,
    /// Run provenance, used to price the rejected response's usage.
    provenance: &'a AuditProvenance,
}

/// Map the pre-computed `on_llm_response` decision onto the turn result.
///
/// `Ok(())` means the response was accepted and the turn proceeds; an `Err`
/// carries the `InternalTurnResult` the turn must return instead (block
/// error, retry-cap error, strict-checkpoint failure, or the
/// retry-with-feedback `Continue`).
async fn apply_response_guardrail<H, M, S>(
    ResponseGuardrailParams {
        ctx,
        response,
        decision,
        message_store,
        state_store,
        event_store,
        hooks,
        authority,
        turn_options,
        provenance,
    }: ResponseGuardrailParams<'_, H, M, S>,
) -> Result<(), InternalTurnResult>
where
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    match decision {
        super::llm::PostLlmGuardrail::Accept => {
            // Any accepted response ends the current rejection streak. The
            // reset rides in `AgentState` so it is persisted by the same
            // checkpoints that save the turn's usage.
            ctx.state.guardrail_retries = 0;
            Ok(())
        }
        super::llm::PostLlmGuardrail::Blocked(reason) => {
            // The provider billed for the rejected response, so its tokens
            // must still land in the cumulative totals before the run ends.
            let turn_usage = apply_turn_usage(ctx, provenance, response);
            // And its completion edge must reach event consumers: the
            // terminal Error state/event carries no usage, so without this
            // the paid call is under-reported (mirrors the retry and
            // retry-cap paths). Best-effort — a failed append must not mask
            // the block error.
            let _ = send_event(
                event_store,
                &ctx.thread_id,
                ctx.turn,
                hooks,
                authority,
                AgentEvent::turn_complete(ctx.turn, turn_usage),
            )
            .await;
            Err(guardrail_block_result(
                &ctx.thread_id,
                ctx.turn,
                event_store,
                hooks,
                authority,
                "response",
                &reason,
            )
            .await)
        }
        super::llm::PostLlmGuardrail::RetryWithFeedback(feedback) => {
            // Account the rejected turn's usage into the cumulative total
            // *before* returning the retry result. Otherwise the rejected
            // turn's tokens are lost (cost under-report) and, with
            // `max_turns: None`, a deterministically-rejecting guardrail
            // would loop forever because `total_usage` never advances and
            // the budget check can never trip.
            let turn_usage = apply_turn_usage(ctx, provenance, response);

            // Every retry pays for another LLM round-trip; a hook that
            // rejects deterministically must not loop (and bill) forever
            // under the default config, so consecutive rejections are
            // capped. The streak lives in `AgentState` (persisted by the
            // Continue-path checkpoints) so it also accumulates across
            // host-driven single-turn `run_turn` invocations.
            ctx.state.guardrail_retries = ctx.state.guardrail_retries.saturating_add(1);
            if ctx.state.guardrail_retries >= super::types::MAX_CONSECUTIVE_GUARDRAIL_RETRIES {
                // The cap-reaching response was a paid LLM round-trip whose
                // usage was folded above; emit its completion edge before
                // erroring out, or event-stream/gRPC consumers under-report
                // the final call (the terminal `Error` carries no usage).
                // Best-effort: a failed append must not mask the cap error.
                let _ = send_event(
                    event_store,
                    &ctx.thread_id,
                    ctx.turn,
                    hooks,
                    authority,
                    AgentEvent::turn_complete(ctx.turn, turn_usage.clone()),
                )
                .await;
                return Err(guardrail_retry_cap_result(
                    &ctx.thread_id,
                    ctx.turn,
                    event_store,
                    hooks,
                    authority,
                )
                .await);
            }

            // Strict durability: the retry returns `Continue` to the caller
            // exactly like a completed turn, so it must pass the same
            // post-LLM checkpoint — a best-effort save could silently fail
            // and leave durable state without the updated turn/usage while
            // the caller already observed `Continue`.
            if turn_options.strict_durability
                && let Err(result) =
                    save_strict_checkpoint(state_store, &ctx.state, "post-LLM").await
            {
                return Err(result);
            }

            Err(retry_with_feedback_result(RetryWithFeedbackParams {
                thread_id: &ctx.thread_id,
                turn: ctx.turn,
                feedback: &feedback,
                turn_usage,
                message_store,
                event_store,
                hooks,
                authority,
            })
            .await)
        }
    }
}

/// Emit a guardrail-block error event and build the terminal
/// [`InternalTurnResult::Error`] the turn returns when an input/output
/// guardrail blocks the LLM call.
async fn guardrail_block_result<H>(
    thread_id: &ThreadId,
    turn: usize,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    phase: &str,
    reason: &str,
) -> InternalTurnResult
where
    H: AgentHooks,
{
    let message = format!("LLM {phase} blocked by guardrail: {reason}");
    warn!("{message}");
    if let Err(error) = send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::error(message.clone(), false),
    )
    .await
    {
        return InternalTurnResult::Error(error);
    }
    InternalTurnResult::Error(AgentError::new(message, false))
}

/// Emit the terminal [`crate::events::AgentEvent::BudgetExceeded`] under the
/// still-open turn and build the mid-turn terminal result.
///
/// Used when spend that accrued *inside* the turn — compaction
/// summarization, or the overflow turn's own usage — crosses a configured
/// limit before the next loop-boundary check would run, so the run stops
/// before paying for another LLM call.
async fn budget_exceeded_mid_turn<H>(
    ctx: &TurnContext,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    limit: crate::types::BudgetLimitKind,
    estimated_cost_usd: Option<f64>,
    turn_usage: TokenUsage,
) -> InternalTurnResult
where
    H: AgentHooks,
{
    warn!(
        "Run-level usage budget exceeded mid-turn (turn={}, limit={limit:?})",
        ctx.turn
    );
    if let Err(error) = send_event(
        event_store,
        &ctx.thread_id,
        ctx.turn,
        hooks,
        authority,
        AgentEvent::budget_exceeded(
            ctx.thread_id.clone(),
            ctx.turn,
            ctx.total_usage.clone(),
            ctx.start_time.elapsed(),
            estimated_cost_usd,
            limit,
        ),
    )
    .await
    {
        return InternalTurnResult::Error(error);
    }
    InternalTurnResult::BudgetExceeded {
        limit,
        estimated_cost_usd,
        turn_usage,
    }
}

/// Emit the error event and build the terminal [`InternalTurnResult::Error`]
/// returned when [`super::types::MAX_CONSECUTIVE_GUARDRAIL_RETRIES`]
/// consecutive `RetryWithFeedback` rejections have been paid for.
async fn guardrail_retry_cap_result<H>(
    thread_id: &ThreadId,
    turn: usize,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
) -> InternalTurnResult
where
    H: AgentHooks,
{
    let message = format!(
        "on_llm_response guardrail rejected {} consecutive responses with RetryWithFeedback; \
         terminating the run instead of paying for further LLM retries",
        super::types::MAX_CONSECUTIVE_GUARDRAIL_RETRIES
    );
    warn!("{message}");
    if let Err(error) = send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::error(message.clone(), false),
    )
    .await
    {
        return InternalTurnResult::Error(error);
    }
    InternalTurnResult::Error(AgentError::new(message, false))
}

struct RetryWithFeedbackParams<'a, H, M> {
    thread_id: &'a ThreadId,
    turn: usize,
    feedback: &'a str,
    /// Usage already applied to `ctx.total_usage` for the rejected turn, so
    /// the returned `Continue { turn_usage }` carries it into the turn
    /// summary (rather than the zeroed default that dropped the usage).
    turn_usage: TokenUsage,
    message_store: &'a Arc<M>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
}

/// Honor [`ResponseDecision::RetryWithFeedback`]: append **only** the
/// guardrail feedback as a user message and continue to the next turn.
///
/// The rejected assistant response is deliberately never written to the
/// message store — the hook contract promises that retry-rejected content
/// stays out of the thread's history and out of the model's context (the
/// secret-leakage use case must not durably store, or re-send, the secret).
/// Because the rejected message (and any `tool_use` blocks it carried) is
/// never persisted, there is no orphan `tool_use` to balance: the history
/// simply gains one feedback user message. The rejected call's tokens were
/// already added to the cumulative totals by the caller.
async fn retry_with_feedback_result<H, M>(
    RetryWithFeedbackParams {
        thread_id,
        turn,
        feedback,
        turn_usage,
        message_store,
        event_store,
        hooks,
        authority,
    }: RetryWithFeedbackParams<'_, H, M>,
) -> InternalTurnResult
where
    H: AgentHooks,
    M: MessageStore,
{
    warn!("LLM response rejected by guardrail; steering a retry with feedback");
    if let Err(error) = message_store
        .append(thread_id, Message::user(feedback))
        .await
    {
        return InternalTurnResult::Error(AgentError::new(
            format!("Failed to append guardrail feedback message: {error}"),
            false,
        ));
    }
    // Best-effort: surface the steering as an error-tier event so streaming
    // consumers can see the retry was triggered. A persistence failure here
    // must not mask the (successful) feedback append, so it is ignored.
    let _ = send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::error(
            format!("LLM response rejected by guardrail; retrying: {feedback}"),
            true,
        ),
    )
    .await;
    // The looping result handler closes this storage turn without ever
    // reaching the tool-phase `TurnComplete` emission, so emit it here —
    // with the rejected attempt's usage — or replay/streaming consumers
    // miss the completion edge for every rejected attempt.
    if let Err(error) = send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::turn_complete(turn, turn_usage.clone()),
    )
    .await
    {
        return InternalTurnResult::Error(error);
    }
    // Carry the rejected turn's usage (already added to `ctx.total_usage`
    // by the caller) into the turn summary so cost accounting and the
    // budget check both advance.
    InternalTurnResult::Continue { turn_usage }
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
    config: &AgentConfig,
    reminder_config: Option<&crate::reminders::ReminderConfig>,
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
    let turn_usage = apply_turn_usage(ctx, provenance, &response);
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
        max_parallel_tools: config.max_parallel_tools,
        reminder_config,
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
        provenance,
        usage_limits: config.usage_limits.as_ref(),
        cancel_token,
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_tool_use_or_missing_stop_reason_allows_tool_execution() {
        assert!(response_allows_tool_execution(None));
        assert!(response_allows_tool_execution(Some(StopReason::ToolUse)));

        for stop_reason in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::StopSequence,
            StopReason::Refusal,
            StopReason::ModelContextWindowExceeded,
            StopReason::Unknown,
        ] {
            assert!(
                !response_allows_tool_execution(Some(stop_reason)),
                "terminal stop reason {stop_reason:?} must not authorize tool execution"
            );
        }
    }
}
