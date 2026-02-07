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
use crate::stores::MessageStore;
use crate::tools::ToolRegistry;
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, PendingToolCallInfo, ThreadId, TokenUsage,
    ToolResult,
};

use log::{debug, info, warn};
use std::sync::Arc;
use tokio::sync::mpsc;

pub(super) async fn begin_turn<H>(
    ctx: &mut TurnContext,
    max_turns: usize,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    ctx.turn += 1;
    ctx.state.turn_count = ctx.turn;

    if ctx.turn > max_turns {
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
    let mut messages = match message_store.get_history(thread_id).await {
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

    if let Some(compact_config) = compaction_config {
        let compactor = LlmContextCompactor::new(Arc::clone(provider), compact_config.clone());
        if compactor.needs_compaction(&messages) {
            debug!(
                "Context compaction triggered (turn={}, message_count={})",
                turn,
                messages.len()
            );

            match compactor.compact_history(messages.clone()).await {
                Ok(result) => {
                    if let Err(error) = message_store
                        .replace_history(thread_id, result.messages.clone())
                        .await
                    {
                        warn!("Failed to replace history after compaction: {error}");
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
                        messages = result.messages;
                    }
                }
                Err(error) => {
                    warn!("Context compaction failed, continuing with full history: {error}");
                }
            }
        }
    }

    Ok(messages)
}

pub(super) fn build_turn_request<Ctx>(
    config: &AgentConfig,
    messages: Vec<Message>,
    tools: &Arc<ToolRegistry<Ctx>>,
) -> ChatRequest
where
    Ctx: Send + Sync + 'static,
{
    let llm_tools = if tools.is_empty() {
        None
    } else {
        Some(tools.to_llm_tools())
    };

    ChatRequest {
        system: config.system_prompt.clone(),
        messages,
        tools: llm_tools,
        max_tokens: config.max_tokens,
        thinking: config.thinking.clone(),
    }
}

pub(super) fn log_chat_request(request: &ChatRequest) {
    debug!(
        "ChatRequest built: system_prompt_len={} num_messages={} num_tools={} max_tokens={}",
        request.system.len(),
        request.messages.len(),
        request.tools.as_ref().map_or(0, Vec::len),
        request.max_tokens
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
    message_store: &Arc<M>,
    thread_id: &ThreadId,
) -> Result<(), AgentError>
where
    P: LlmProvider,
    M: MessageStore,
{
    let compactor = LlmContextCompactor::new(Arc::clone(provider), compaction_config.clone());
    let history = message_store
        .get_history(thread_id)
        .await
        .map_err(|error| {
            AgentError::new(
                format!("Failed to get history for compaction after context overflow: {error}"),
                false,
            )
        })?;

    let result = compactor.compact_history(history).await.map_err(|error| {
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
        message_id,
        turn_usage,
        ctx,
        provider,
        message_store,
        compaction_config,
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
        Some(StopReason::ModelContextWindowExceeded) => {
            warn!("Model context window exceeded (turn={})", ctx.turn);
            if let Some(compact_config) = compaction_config {
                if let Err(error) = compact_after_context_overflow(
                    provider,
                    compact_config,
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
        _ => InternalTurnResult::Continue { turn_usage },
    }
}

pub(super) async fn execute_turn<Ctx, P, H, M>(
    ExecuteTurnParameters {
        tx,
        seq,
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        config,
        compaction_config,
        execution_store,
    }: ExecuteTurnParameters<'_, Ctx, P, H, M>,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
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
        tx,
        hooks,
        seq,
    })
    .await
    {
        Ok(messages) => messages,
        Err(error) => return InternalTurnResult::Error(error),
    };

    let request = build_turn_request(config, messages, tools);
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
        Err(error) => return InternalTurnResult::Error(error),
    };

    let turn_usage = apply_turn_usage(ctx, &response);
    let ProcessedTurnResponse {
        stop_reason,
        text_content,
        pending_tool_calls,
    } = match process_turn_response(TurnResponseProcessingParams {
        response,
        message_id: &message_id,
        thinking_id: &thinking_id,
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
        message_id,
        turn_usage,
        ctx,
        provider,
        message_store,
        compaction_config,
        tx,
        hooks,
        seq,
    })
    .await
}
