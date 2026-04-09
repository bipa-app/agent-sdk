use super::helpers::{build_assistant_message, extract_content, pending_tool_index, send_event};
use super::llm::{call_llm_streaming, call_llm_with_retry};
use super::tool_execution::{append_tool_results, execute_tool_call};
use super::types::{
    ExecuteTurnParameters, InternalTurnResult, LlmCallParams, LlmEventContext, LlmStreamIds,
    ProcessedTurnResponse, ToolBatchExecutionParams, ToolCallExecutionContext,
    ToolExecutionOutcome, TurnCompletionParams, TurnContext, TurnMessageLoadParams,
    TurnResponseProcessingParams, TurnStopReasonParams, TurnToolPhaseParams,
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
    ToolResult, ToolRuntime, TurnOptions,
};
use agent_sdk_core::audit::AuditProvenance;

use log::{debug, info, warn};
use std::sync::Arc;

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
    }: TurnMessageLoadParams<'_, P, H, M>,
) -> Result<Vec<Message>, AgentError>
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
    }: MaybeCompactParams<'_, P, H, M>,
) -> Result<Vec<Message>, AgentError>
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
            })
            .await;
        }

        return Ok(messages);
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
            })
            .await;
        }
    }

    Ok(messages)
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
    }: ThresholdCompactionParams<'_, C, H, M>,
) -> Result<Vec<Message>, AgentError>
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
    )
    .await
    {
        Ok(result) => {
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

            Ok(result.messages)
        }
        Err(error) => {
            warn!("Context compaction failed, continuing with full history: {error}");
            Ok(original_messages)
        }
    }
}

async fn compact_history_and_store<C, M>(
    compactor: &C,
    messages: Vec<Message>,
    message_store: &Arc<M>,
    thread_id: &ThreadId,
    trigger: &'static str,
    compaction_error_prefix: &'static str,
    replace_history_error_prefix: &'static str,
) -> Result<StoredCompactionResult, AgentError>
where
    C: ContextCompactor + ?Sized,
    M: MessageStore,
{
    #[cfg(not(feature = "otel"))]
    let _ = trigger;
    #[cfg(feature = "otel")]
    let mut compaction_span = start_compaction_span(trigger);

    let result = match compactor.compact_history(messages).await {
        Ok(result) => result,
        Err(error) => {
            #[cfg(feature = "otel")]
            finish_compaction_span_error(
                &mut compaction_span,
                "context_compaction_failed",
                &error.to_string(),
            );
            return Err(AgentError::new(
                format!("{compaction_error_prefix}: {error}"),
                false,
            ));
        }
    };

    let stored_result = StoredCompactionResult {
        messages: result.messages,
        original_count: result.original_count,
        new_count: result.new_count,
        original_tokens: result.original_tokens,
        new_tokens: result.new_tokens,
    };

    #[cfg(feature = "otel")]
    record_compaction_result(&mut compaction_span, &stored_result);

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
        return Err(AgentError::new(
            format!("{replace_history_error_prefix}: {error}"),
            false,
        ));
    }

    #[cfg(feature = "otel")]
    finish_compaction_span_success(&mut compaction_span);

    Ok(stored_result)
}

#[cfg(feature = "otel")]
fn start_compaction_span(trigger: &'static str) -> opentelemetry::global::BoxedSpan {
    use crate::observability::{attrs, spans};

    spans::start_internal_span(
        "agent.context_compaction",
        vec![attrs::kv(attrs::SDK_COMPACTION_TRIGGER, trigger)],
    )
}

#[cfg(feature = "otel")]
fn record_compaction_result(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &StoredCompactionResult,
) {
    use crate::observability::attrs;
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

    span.set_attribute(attrs::kv(attrs::SDK_OUTCOME, "error"));
    spans::set_span_error(span, error_type, message);
    span.end();
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
    })
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
                            debug!(
                                "    block[{block_idx}]: ToolUse(id={id}, name={name}, input={input})"
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
        #[cfg(feature = "otel")]
        observability_store,
    }: LlmCallParams<'_, P, H>,
) -> Result<ChatResponse, AgentError>
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
    };

    #[cfg(feature = "otel")]
    let mut llm_span = {
        use crate::observability::{attrs, spans};
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
        spans::start_client_span(span_name, init_attrs)
    };

    #[cfg(feature = "otel")]
    let request_for_capture = observability_store.map(|_| request.clone());

    let (result, retry_count) = if config.streaming {
        call_llm_streaming(
            provider,
            request,
            config,
            &event_ctx,
            LlmStreamIds {
                message_id,
                thinking_id,
            },
        )
        .await
    } else {
        call_llm_with_retry(provider, request, config, &event_ctx).await
    };

    #[cfg(feature = "otel")]
    if let (Some(observability_store), Some(request), Ok(response)) = (
        observability_store,
        request_for_capture.as_ref(),
        result.as_ref(),
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
    finish_llm_span(&mut llm_span, &result, retry_count);

    // Silence unused binding when otel is disabled.
    let _ = retry_count;
    let _ = thread_id;

    result
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
    use crate::observability::{CaptureKind, PayloadBundle, payload, spans};
    use opentelemetry::trace::Span;

    let system_json = payload::convert_system_instructions(request);
    let input_json = payload::convert_input_messages(request);
    let output_json = payload::convert_output_messages(response);
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
            spans::record_payload_on_span(
                span,
                &result,
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
fn finish_llm_span(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &Result<ChatResponse, AgentError>,
    retry_count: u32,
) {
    use crate::observability::{attrs, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    for attempt in 1..=retry_count {
        span.add_event(
            "llm.retry",
            vec![KeyValue::new("retry.attempt", i64::from(attempt))],
        );
    }

    match result {
        Ok(response) => {
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
        }
        Err(err) => {
            let error_type = if err.message.contains("Rate limited") {
                "rate_limited"
            } else if err.message.contains("Invalid request") {
                "invalid_request"
            } else if err.message.contains("Server error") {
                "server_error"
            } else if err.message.contains("Stream") {
                "stream_error"
            } else {
                "_OTHER"
            };
            spans::set_span_error(span, error_type, &err.message);
        }
    }
    span.end();
}

pub(super) fn apply_turn_usage(ctx: &mut TurnContext, response: &ChatResponse) -> TokenUsage {
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    };
    ctx.total_usage.add(&turn_usage);
    ctx.state.total_usage = ctx.total_usage.clone();

    // Capture provider-level provenance into the turn context so that
    // the [`agent_sdk_core::TurnSummary`] emitted at outcome time
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

    for pending in pending_tool_calls.clone() {
        match execute_tool_call(&pending, &execution_ctx).await {
            ToolExecutionOutcome::Completed { tool_id, result } => {
                tool_results.push((tool_id, result));
            }
            ToolExecutionOutcome::RequiresConfirmation {
                tool_id,
                tool_name,
                display_name,
                input,
                description,
                listen_context,
            } => {
                let pending_idx = match pending_tool_index(&pending_tool_calls, &tool_id) {
                    Ok(index) => index,
                    Err(error) => return Err(InternalTurnResult::Error(error)),
                };
                if let Some(context) = listen_context {
                    pending_tool_calls[pending_idx].listen_context = Some(context);
                }

                let continuation = AgentContinuation {
                    thread_id: thread_id.clone(),
                    turn,
                    total_usage: total_usage.clone(),
                    turn_usage: turn_usage.clone(),
                    pending_tool_calls: pending_tool_calls.clone(),
                    awaiting_index: pending_idx,
                    completed_results: tool_results,
                    state: state.clone(),
                };

                return Err(InternalTurnResult::AwaitingConfirmation {
                    tool_call_id: tool_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                    continuation: Box::new(continuation),
                });
            }
            ToolExecutionOutcome::Error(error) => return Err(InternalTurnResult::Error(error)),
        }
    }

    Ok(tool_results)
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

pub(super) async fn compact_after_context_overflow<P, M>(
    provider: &Arc<P>,
    compaction_config: &CompactionConfig,
    compactor: Option<&Arc<dyn ContextCompactor>>,
    message_store: &Arc<M>,
    thread_id: &ThreadId,
) -> Result<(), AgentError>
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

    let result = if let Some(compactor) = compactor {
        compact_history_and_store(
            compactor.as_ref(),
            history,
            message_store,
            thread_id,
            "overflow",
            "Context compaction failed after overflow",
            "Failed to replace history after overflow compaction",
        )
        .await?
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
        )
        .await?
    };

    info!(
        "Context compacted after overflow (original_tokens={}, new_tokens={})",
        result.original_tokens, result.new_tokens
    );

    Ok(())
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
            warn!("Model context window exceeded (turn={})", ctx.turn);
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
                if let Err(error) = compact_after_context_overflow(
                    provider,
                    compact_config,
                    compactor,
                    message_store,
                    &ctx.thread_id,
                )
                .await
                {
                    return InternalTurnResult::Error(error);
                }
                ctx.turn = ctx.turn.saturating_sub(1);
                return InternalTurnResult::Continue { turn_usage };
            }

            InternalTurnResult::Error(AgentError::new(
                "Model context window exceeded and no compaction configured".to_string(),
                false,
            ))
        }
        Some(StopReason::StopSequence) => {
            info!("Stop sequence hit (turn={})", ctx.turn);
            InternalTurnResult::Done
        }
        None => {
            // Unknown/missing stop reason. Only continue if tool results were
            // appended (preserving message alternation).
            if had_tool_calls {
                warn!(
                    "No stop reason with tool calls (turn={}), continuing",
                    ctx.turn
                );
                InternalTurnResult::Continue { turn_usage }
            } else {
                warn!(
                    "No stop reason and no tool calls (turn={}), stopping",
                    ctx.turn
                );
                InternalTurnResult::Done
            }
        }
    }
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
        if let Err(compact_err) = compact_after_context_overflow(
            provider,
            compact_config,
            compactor,
            message_store,
            &ctx.thread_id,
        )
        .await
        {
            return Some(InternalTurnResult::Error(compact_err));
        }
        // Don't count the failed attempt as a turn
        ctx.turn = ctx.turn.saturating_sub(1);
        return Some(InternalTurnResult::Continue {
            turn_usage: TokenUsage::default(),
        });
    }
    None
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
    #[cfg(feature = "otel")]
    let turn_number = params.ctx.turn + 1; // begin_turn increments

    let result = execute_turn_inner(params).await;

    #[cfg(feature = "otel")]
    {
        // Span is created and ended in one shot after the turn completes.
        // This avoids holding a span across all the awaits inside the turn.
        use crate::observability::{attrs, spans};
        use opentelemetry::KeyValue;
        use opentelemetry::trace::Span;

        let mut turn_span = spans::start_internal_span(
            "agent.turn",
            vec![attrs::kv_i64(
                attrs::SDK_TURN_NUMBER,
                i64::try_from(turn_number).unwrap_or(0),
            )],
        );

        match &result {
            InternalTurnResult::Continue { turn_usage } => {
                let had_tools = turn_usage.input_tokens > 0 || turn_usage.output_tokens > 0;
                turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "continue"));
                turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, had_tools));
                turn_span.set_attribute(attrs::kv_i64(
                    attrs::SDK_TURN_INPUT_TOKENS,
                    i64::from(turn_usage.input_tokens),
                ));
                turn_span.set_attribute(attrs::kv_i64(
                    attrs::SDK_TURN_OUTPUT_TOKENS,
                    i64::from(turn_usage.output_tokens),
                ));
            }
            InternalTurnResult::Done => {
                turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "done"));
                turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            }
            InternalTurnResult::Refusal => {
                turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "refusal"));
                turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, false));
            }
            InternalTurnResult::AwaitingConfirmation { .. } => {
                turn_span.set_attribute(KeyValue::new(
                    attrs::SDK_TURN_STOP_REASON,
                    "awaiting_confirmation",
                ));
                turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, true));
            }
            InternalTurnResult::PendingToolCalls { .. } => {
                turn_span.set_attribute(KeyValue::new(
                    attrs::SDK_TURN_STOP_REASON,
                    "pending_tool_calls",
                ));
                turn_span.set_attribute(attrs::kv_bool(attrs::SDK_TURN_HAD_TOOL_CALLS, true));
            }
            InternalTurnResult::Error(err) => {
                turn_span.set_attribute(KeyValue::new(attrs::SDK_TURN_STOP_REASON, "error"));
                spans::set_span_error(&mut turn_span, "turn_error", &err.message);
            }
        }
        turn_span.end();
    }

    result
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
    })
    .await
    {
        Ok(messages) => messages,
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

    let message_id = uuid::Uuid::new_v4().to_string();
    let thinking_id = uuid::Uuid::new_v4().to_string();
    let response = match request_llm_response(LlmCallParams {
        provider,
        request,
        config,
        event_store,
        hooks,
        authority,
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        message_id: &message_id,
        thinking_id: &thinking_id,
        #[cfg(feature = "otel")]
        observability_store,
    })
    .await
    {
        Ok(response) => response,
        Err(error) => {
            if let Some(result) = try_recover_prompt_too_long(
                &error,
                ctx,
                compaction_config,
                compactor,
                provider,
                message_store,
            )
            .await
            {
                return result;
            }
            return InternalTurnResult::Error(error);
        }
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
    )
    .await
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

    let had_tool_calls = !pending_tool_calls.is_empty();

    // Strict durability: checkpoint after LLM response, before tool execution.
    if turn_options.strict_durability {
        if let Err(error) = state_store.save(&ctx.state).await {
            warn!("Strict durability: failed to save post-LLM state: {error}");
        }
    } else if had_tool_calls && let Err(error) = state_store.save(&ctx.state).await {
        warn!("Failed to save pre-tool state checkpoint: {error}");
    }

    // External tool runtime: return pending tool calls to the caller instead
    // of executing them inline.
    if had_tool_calls && turn_options.tool_runtime == ToolRuntime::External {
        let continuation = AgentContinuation {
            thread_id: ctx.thread_id.clone(),
            turn: ctx.turn,
            total_usage: ctx.total_usage.clone(),
            turn_usage: turn_usage.clone(),
            pending_tool_calls: pending_tool_calls.clone(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: ctx.state.clone(),
        };
        return InternalTurnResult::PendingToolCalls {
            turn_usage,
            pending_tool_calls,
            continuation: Box::new(continuation),
        };
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
    })
    .await
    {
        return outcome;
    }

    // Strict durability: checkpoint after tool execution.
    if turn_options.strict_durability
        && let Err(error) = state_store.save(&ctx.state).await
    {
        warn!("Strict durability: failed to save post-tool state: {error}");
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
    })
    .await
}
