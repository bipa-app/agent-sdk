use super::helpers::{build_assistant_message, extract_content, pending_tool_index, send_event};
use super::llm::{call_llm_streaming, call_llm_with_retry};
use super::tool_execution::{append_tool_results, execute_tool_call};
use super::types::{
    ExecuteTurnParameters, InternalTurnResult, LlmCallParams, ProcessedTurnResponse,
    ToolBatchExecutionParams, ToolCallExecutionContext, ToolExecutionOutcome, TurnCompletionParams,
    TurnContext, TurnMessageLoadParams, TurnResponseProcessingParams, TurnStopReasonParams,
    TurnToolPhaseParams,
};

use crate::context::{CompactionConfig, ContextCompactor, LlmContextCompactor};
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::{
    ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, StopReason,
};
use crate::stores::{MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, PendingToolCallInfo, ThreadId, TokenUsage,
    ToolResult,
};

use log::{debug, info, warn};
use std::sync::Arc;
use tokio::sync::mpsc;

pub(super) async fn begin_turn<H>(
    ctx: &mut TurnContext,
    max_turns: Option<usize>,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    ctx.turn += 1;
    ctx.state.turn_count = ctx.turn;

    if let Some(max_turns) = max_turns
        && ctx.turn > max_turns
    {
        warn!("Max turns reached (turn={}, max={max_turns})", ctx.turn);
        let message = format!("Maximum turns ({max_turns}) reached");
        send_event(tx, hooks, seq, AgentEvent::error(message.clone(), true)).await;
        return Err(AgentError::new(message, true));
    }

    send_event(
        tx,
        hooks,
        seq,
        AgentEvent::start(ctx.thread_id.clone(), ctx.turn),
    )
    .await;
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
        tx,
        hooks,
        seq,
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
                tx,
                hooks,
                seq,
                AgentEvent::error(format!("Failed to get history: {error}"), false),
            )
            .await;
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
        tx,
        hooks,
        seq,
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
    tx: &'a mpsc::Sender<AgentEventEnvelope>,
    hooks: &'a Arc<H>,
    seq: &'a SequenceCounter,
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
        tx,
        hooks,
        seq,
    }: MaybeCompactParams<'_, P, H, M>,
) -> Result<Vec<Message>, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    let maybe_result = if let Some(compactor) = compactor {
        if compactor.needs_compaction(&messages) {
            debug!(
                "Context compaction triggered (turn={}, message_count={})",
                turn,
                messages.len()
            );
            Some(compactor.compact_history(messages.clone()).await)
        } else {
            None
        }
    } else if let Some(compact_config) = compaction_config {
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compact_config.clone());
        if default_compactor.needs_compaction(&messages) {
            debug!(
                "Context compaction triggered (turn={}, message_count={})",
                turn,
                messages.len()
            );
            Some(default_compactor.compact_history(messages.clone()).await)
        } else {
            None
        }
    } else {
        None
    };

    let Some(result) = maybe_result else {
        return Ok(messages);
    };

    match result {
        Ok(result) => {
            if let Err(error) = message_store
                .replace_history(thread_id, result.messages.clone())
                .await
            {
                warn!("Failed to replace history after compaction: {error}");
                Ok(messages)
            } else {
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::context_compacted(
                        result.original_count,
                        result.new_count,
                        result.original_tokens,
                        result.new_tokens,
                    ),
                )
                .await;

                info!(
                    "Context compacted successfully (original_count={}, new_count={}, original_tokens={}, new_tokens={})",
                    result.original_count,
                    result.new_count,
                    result.original_tokens,
                    result.new_tokens
                );
                Ok(result.messages)
            }
        }
        Err(error) => {
            warn!("Context compaction failed, continuing with full history: {error}");
            Ok(messages)
        }
    }
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
        tx,
        hooks,
        seq,
        turn,
        message_id,
        thinking_id,
    }: LlmCallParams<'_, P, H>,
) -> Result<ChatResponse, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    debug!("Calling LLM (turn={turn}, streaming={})", config.streaming);

    if config.streaming {
        call_llm_streaming(
            provider,
            request,
            config,
            tx,
            hooks,
            seq,
            (message_id, thinking_id),
        )
        .await
    } else {
        call_llm_with_retry(provider, request, config, tx, hooks, seq).await
    }
}

pub(super) fn apply_turn_usage(ctx: &mut TurnContext, response: &ChatResponse) -> TokenUsage {
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    };
    ctx.total_usage.add(&turn_usage);
    ctx.state.total_usage = ctx.total_usage.clone();
    turn_usage
}

pub(super) async fn process_turn_response<Ctx, H, M>(
    TurnResponseProcessingParams {
        response,
        message_id,
        thinking_id,
        thread_id,
        tools,
        message_store,
        tx,
        hooks,
        seq,
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
            tx,
            hooks,
            seq,
            AgentEvent::thinking(thinking_id, thinking.clone()),
        )
        .await;
    }

    if let Some(text) = &text_content {
        send_event(tx, hooks, seq, AgentEvent::text(message_id, text.clone())).await;
    }

    let assistant_msg = build_assistant_message(&response);
    if let Err(error) = message_store.append(thread_id, assistant_msg).await {
        send_event(
            tx,
            hooks,
            seq,
            AgentEvent::error(
                format!("Failed to append assistant message: {error}"),
                false,
            ),
        )
        .await;
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
            let display_name = tools
                .get(name)
                .map(|tool| tool.display_name().to_string())
                .or_else(|| {
                    tools
                        .get_async(name)
                        .map(|tool| tool.display_name().to_string())
                })
                .or_else(|| {
                    tools
                        .get_listen(name)
                        .map(|tool| tool.display_name().to_string())
                })
                .unwrap_or_default();

            PendingToolCallInfo {
                id: id.clone(),
                name: name.clone(),
                display_name,
                input: input.clone(),
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
        tx,
        seq,
        execution_store,
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
        tx,
        seq,
        execution_store,
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
        tx,
        hooks,
        seq,
    }: TurnCompletionParams<'_, H, M>,
) -> Result<(), AgentError>
where
    H: AgentHooks,
    M: MessageStore,
{
    append_tool_results(tool_results, thread_id, message_store).await?;
    send_event(
        tx,
        hooks,
        seq,
        AgentEvent::TurnComplete {
            turn,
            usage: turn_usage.clone(),
        },
    )
    .await;
    Ok(())
}

pub(super) async fn execute_turn_tool_phase<Ctx, H, M>(
    TurnToolPhaseParams {
        pending_tool_calls,
        tool_context,
        thread_id,
        tools,
        hooks,
        tx,
        seq,
        execution_store,
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
        tx,
        seq,
        execution_store,
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
        tx,
        hooks,
        seq,
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
        compactor.compact_history(history).await
    } else {
        let default_compactor =
            LlmContextCompactor::new(Arc::clone(provider), compaction_config.clone());
        default_compactor.compact_history(history).await
    }
    .map_err(|error| {
        AgentError::new(
            format!("Context compaction failed after overflow: {error}"),
            false,
        )
    })?;

    message_store
        .replace_history(thread_id, result.messages)
        .await
        .map_err(|error| {
            AgentError::new(
                format!("Failed to replace history after overflow compaction: {error}"),
                false,
            )
        })?;

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
        tx,
        hooks,
        seq,
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
            send_event(
                tx,
                hooks,
                seq,
                AgentEvent::refusal(message_id, text_content),
            )
            .await;
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
    if error.message.contains("prompt is too long")
        && let Some(compact_config) = compaction_config
    {
        warn!(
            "Prompt too long, attempting emergency context compaction (turn={})",
            ctx.turn
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
    ExecuteTurnParameters {
        tx,
        seq,
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
    }: ExecuteTurnParameters<'_, Ctx, P, H, M, S>,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    if let Err(error) = begin_turn(ctx, config.max_turns, tx, hooks, seq).await {
        return InternalTurnResult::Error(error);
    }

    let messages = match load_turn_messages(TurnMessageLoadParams {
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        provider,
        message_store,
        compaction_config,
        compactor,
        tx,
        hooks,
        seq,
    })
    .await
    {
        Ok(messages) => messages,
        Err(error) => return InternalTurnResult::Error(error),
    };

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
        tx,
        hooks,
        seq,
        turn: ctx.turn,
        message_id: &message_id,
        thinking_id: &thinking_id,
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
        tx,
        seq,
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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
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
        tools,
        message_store,
        tx,
        hooks,
        seq,
    })
    .await
    {
        Ok(processed) => processed,
        Err(error) => return InternalTurnResult::Error(error),
    };

    let had_tool_calls = !pending_tool_calls.is_empty();

    if had_tool_calls && let Err(error) = state_store.save(&ctx.state).await {
        warn!("Failed to save pre-tool state checkpoint: {error}");
    }

    if let Err(outcome) = execute_turn_tool_phase(TurnToolPhaseParams {
        pending_tool_calls,
        tool_context,
        thread_id: &ctx.thread_id,
        tools,
        hooks,
        tx,
        seq,
        execution_store,
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
        tx,
        hooks,
        seq,
    })
    .await
}
