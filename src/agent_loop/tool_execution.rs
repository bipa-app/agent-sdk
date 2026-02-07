use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use tokio::sync::mpsc;

use super::helpers::{millis_to_u64, send_event, wrap_and_send};
use super::idempotency::{execute_with_idempotency, try_get_cached_result};
use super::listen::{
    build_listen_confirmation_input, cancel_listen_with_warning, wait_for_listen_ready,
};
use super::types::{
    ConfirmedToolExecutionContext, ListenReady, ToolCallExecutionContext, ToolExecutionOutcome,
};
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::{AgentHooks, ToolDecision};
use crate::llm::Message;
use crate::stores::MessageStore;
use crate::tools::{
    ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus, ListenStopReason, ToolContext,
};
use crate::types::{
    AgentError, ListenExecutionContext, PendingToolCallInfo, ThreadId, ToolOutcome, ToolResult,
};

/// Execute a single tool call with hook checks.
///
/// Returns the outcome of the tool execution, which may be:
/// - `Completed`: Tool ran (or was blocked), result captured
/// - `RequiresConfirmation`: Hook requires user confirmation
///
/// Supports both synchronous and asynchronous tools. Async tools are detected
/// automatically and their progress is streamed via events.
pub(super) async fn execute_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(cached_result) = try_get_cached_result(ctx.execution_store, &pending.id).await {
        return ToolExecutionOutcome::Completed {
            tool_id: pending.id.clone(),
            result: cached_result,
        };
    }

    if let Some(listen_tool) = ctx.tools.get_listen(&pending.name) {
        return execute_listen_tool_call(pending, listen_tool, ctx).await;
    }

    if let Some(async_tool) = ctx.tools.get_async(&pending.name) {
        return execute_async_tool_call(pending, async_tool, ctx).await;
    }

    let Some(tool) = ctx.tools.get(&pending.name) else {
        return ToolExecutionOutcome::Completed {
            tool_id: pending.id.clone(),
            result: ToolResult::error(format!("Unknown tool: {}", pending.name)),
        };
    };

    execute_sync_tool_call(pending, tool, ctx).await
}

pub(super) async fn execute_listen_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    listen_tool: &Arc<dyn ErasedListenTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = listen_tool.tier();
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
    )
    .await;

    let tool_start = Instant::now();
    let ready = match wait_for_listen_ready(
        pending,
        listen_tool,
        ctx.tool_context,
        ctx.hooks,
        ctx.tx,
        ctx.seq,
    )
    .await
    {
        Ok(ready) => ready,
        Err(result) => return finish_listen_ready_failure(pending, ctx, tool_start, result).await,
    };

    match ctx
        .hooks
        .pre_tool_use(&pending.name, &pending.input, tier)
        .await
    {
        ToolDecision::Allow => {
            handle_listen_tool_allow(pending, listen_tool, ctx, &ready, tool_start).await
        }
        ToolDecision::Block(reason) => {
            handle_listen_tool_block(pending, listen_tool, ctx, &ready, reason).await
        }
        ToolDecision::RequiresConfirmation(description) => {
            handle_listen_tool_confirmation(pending, ctx, ready, description).await
        }
    }
}

pub(super) async fn finish_listen_ready_failure<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    tool_start: Instant,
    mut result: ToolResult,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
    ctx.hooks.post_tool_use(&pending.name, &result).await;
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await;
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
    }
}

pub(super) async fn handle_listen_tool_allow<Ctx, H>(
    pending: &PendingToolCallInfo,
    listen_tool: &Arc<dyn ErasedListenTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    ready: &ListenReady,
    tool_start: Instant,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let result = execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
        match listen_tool
            .execute(ctx.tool_context, &ready.operation_id, ready.revision)
            .await
        {
            Ok(mut value) => {
                value.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                value
            }
            Err(error) => ToolResult::error(format!("Listen execute error: {error}"))
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
        }
    })
    .await;
    ctx.hooks.post_tool_use(&pending.name, &result).await;
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await;
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
    }
}

pub(super) async fn handle_listen_tool_block<Ctx, H>(
    pending: &PendingToolCallInfo,
    listen_tool: &Arc<dyn ErasedListenTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    ready: &ListenReady,
    reason: String,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    cancel_listen_with_warning(
        listen_tool,
        ctx.tool_context,
        &ready.operation_id,
        ListenStopReason::Blocked,
        &pending.id,
        &pending.name,
    )
    .await;
    let result = ToolResult::error(format!("Blocked: {reason}"));
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await;
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
    }
}

pub(super) async fn handle_listen_tool_confirmation<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    ready: ListenReady,
    description: String,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let input = build_listen_confirmation_input(&pending.input, &ready);
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::ToolRequiresConfirmation {
            id: pending.id.clone(),
            name: pending.name.clone(),
            input: input.clone(),
            description: description.clone(),
        },
    )
    .await;
    ToolExecutionOutcome::RequiresConfirmation {
        tool_id: pending.id.clone(),
        tool_name: pending.name.clone(),
        display_name: pending.display_name.clone(),
        input,
        description,
        listen_context: Some(ListenExecutionContext {
            operation_id: ready.operation_id,
            revision: ready.revision,
            snapshot: ready.snapshot,
            expires_at: ready.expires_at,
        }),
    }
}

pub(super) async fn execute_async_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    async_tool: &Arc<dyn ErasedAsyncTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = async_tool.tier();
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
    )
    .await;

    match ctx
        .hooks
        .pre_tool_use(&pending.name, &pending.input, tier)
        .await
    {
        ToolDecision::Allow => {
            let result =
                execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
                    execute_async_tool(pending, async_tool, ctx.tool_context, ctx.tx, ctx.seq).await
                })
                .await;
            ctx.hooks.post_tool_use(&pending.name, &result).await;
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;
            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::Block(reason) => {
            let result = ToolResult::error(format!("Blocked: {reason}"));
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;
            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::RequiresConfirmation(description) => {
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::ToolRequiresConfirmation {
                    id: pending.id.clone(),
                    name: pending.name.clone(),
                    input: pending.input.clone(),
                    description: description.clone(),
                },
            )
            .await;
            ToolExecutionOutcome::RequiresConfirmation {
                tool_id: pending.id.clone(),
                tool_name: pending.name.clone(),
                display_name: pending.display_name.clone(),
                input: pending.input.clone(),
                description,
                listen_context: None,
            }
        }
    }
}

pub(super) async fn execute_sync_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = tool.tier();
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
    )
    .await;

    match ctx
        .hooks
        .pre_tool_use(&pending.name, &pending.input, tier)
        .await
    {
        ToolDecision::Allow => {
            let tool_start = Instant::now();
            let result =
                execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
                    match tool.execute(ctx.tool_context, pending.input.clone()).await {
                        Ok(mut value) => {
                            value.duration_ms =
                                Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            value
                        }
                        Err(error) => ToolResult::error(format!("Tool error: {error}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    }
                })
                .await;
            ctx.hooks.post_tool_use(&pending.name, &result).await;
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;
            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::Block(reason) => {
            let result = ToolResult::error(format!("Blocked: {reason}"));
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;
            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::RequiresConfirmation(description) => {
            send_event(
                ctx.tx,
                ctx.hooks,
                ctx.seq,
                AgentEvent::ToolRequiresConfirmation {
                    id: pending.id.clone(),
                    name: pending.name.clone(),
                    input: pending.input.clone(),
                    description: description.clone(),
                },
            )
            .await;
            ToolExecutionOutcome::RequiresConfirmation {
                tool_id: pending.id.clone(),
                tool_name: pending.name.clone(),
                display_name: pending.display_name.clone(),
                input: pending.input.clone(),
                description,
                listen_context: None,
            }
        }
    }
}

/// Execute an async tool call and stream progress until completion.
///
/// This function handles the two-phase execution of async tools:
/// 1. Execute the tool (returns immediately with Success/Failed/`InProgress`)
/// 2. If `InProgress`, stream status updates until completion
pub(super) async fn execute_async_tool<Ctx>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedAsyncTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
) -> ToolResult
where
    Ctx: Send + Sync + Clone,
{
    let tool_start = Instant::now();

    // Step 1: Execute (lightweight, returns quickly)
    let outcome = match tool.execute(tool_context, pending.input.clone()).await {
        Ok(o) => o,
        Err(e) => {
            return ToolResult::error(format!("Tool error: {e}"))
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
        }
    };

    match outcome {
        // Synchronous completion - return immediately
        ToolOutcome::Success(mut result) | ToolOutcome::Failed(mut result) => {
            result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
            result
        }

        // Async operation - stream status until completion
        ToolOutcome::InProgress {
            operation_id,
            message,
        } => {
            // Emit initial progress
            wrap_and_send(
                tx,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "started",
                    &message,
                    None,
                ),
                seq,
            )
            .await;

            // Stream status updates
            let mut stream = tool.check_status_stream(tool_context, &operation_id);

            while let Some(status) = stream.next().await {
                match status {
                    ErasedToolStatus::Progress {
                        stage,
                        message,
                        data,
                    } => {
                        wrap_and_send(
                            tx,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                stage,
                                message,
                                data,
                            ),
                            seq,
                        )
                        .await;
                    }
                    ErasedToolStatus::Completed(mut result)
                    | ErasedToolStatus::Failed(mut result) => {
                        result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        return result;
                    }
                }
            }

            // Stream ended without completion (shouldn't happen)
            ToolResult::error("Async tool stream ended without completion")
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis()))
        }
    }
}

/// Execute the confirmed tool call from a resume operation.
///
/// This is called when resuming after a tool required confirmation.
/// Supports both sync and async tools.
pub(super) async fn execute_confirmed_tool<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    rejection_reason: Option<String>,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(reason) = rejection_reason {
        return handle_confirmed_tool_rejection(awaiting_tool, ctx, reason).await;
    }

    if let Some(cached_result) = try_get_cached_result(ctx.execution_store, &awaiting_tool.id).await
    {
        return finish_confirmed_tool(awaiting_tool, ctx, cached_result).await;
    }

    let result = execute_confirmed_tool_inner(awaiting_tool, ctx).await;
    finish_confirmed_tool(awaiting_tool, ctx, result).await
}

pub(super) async fn handle_confirmed_tool_rejection<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
    reason: String,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(listen_tool) = ctx.tools.get_listen(&awaiting_tool.name)
        && let Some(listen) = awaiting_tool.listen_context.as_ref()
    {
        cancel_listen_with_warning(
            listen_tool,
            ctx.tool_context,
            &listen.operation_id,
            ListenStopReason::UserRejected,
            &awaiting_tool.id,
            &awaiting_tool.name,
        )
        .await;
    }

    let result = ToolResult::error(format!("Rejected: {reason}"));
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_end(
            &awaiting_tool.id,
            &awaiting_tool.name,
            &awaiting_tool.display_name,
            result.clone(),
        ),
    )
    .await;
    result
}

pub(super) async fn execute_confirmed_tool_inner<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(listen_tool) = ctx.tools.get_listen(&awaiting_tool.name) {
        let Some(listen) = awaiting_tool.listen_context.as_ref() else {
            return ToolResult::error(format!(
                "Listen context missing for tool: {}",
                awaiting_tool.name
            ));
        };
        let tool_start = Instant::now();
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                match listen_tool
                    .execute(ctx.tool_context, &listen.operation_id, listen.revision)
                    .await
                {
                    Ok(mut value) => {
                        value.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        value
                    }
                    Err(error) => ToolResult::error(format!("Listen execute error: {error}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                }
            },
        )
        .await;
    }

    if let Some(async_tool) = ctx.tools.get_async(&awaiting_tool.name) {
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                execute_async_tool(awaiting_tool, async_tool, ctx.tool_context, ctx.tx, ctx.seq)
                    .await
            },
        )
        .await;
    }

    if let Some(tool) = ctx.tools.get(&awaiting_tool.name) {
        let tool_start = Instant::now();
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                match tool
                    .execute(ctx.tool_context, awaiting_tool.input.clone())
                    .await
                {
                    Ok(mut value) => {
                        value.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        value
                    }
                    Err(error) => ToolResult::error(format!("Tool error: {error}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                }
            },
        )
        .await;
    }

    ToolResult::error(format!("Unknown tool: {}", awaiting_tool.name))
}

pub(super) async fn finish_confirmed_tool<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
    result: ToolResult,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    ctx.hooks.post_tool_use(&awaiting_tool.name, &result).await;
    send_event(
        ctx.tx,
        ctx.hooks,
        ctx.seq,
        AgentEvent::tool_call_end(
            &awaiting_tool.id,
            &awaiting_tool.name,
            &awaiting_tool.display_name,
            result.clone(),
        ),
    )
    .await;
    result
}

/// Append tool results to message history.
pub(super) async fn append_tool_results<M>(
    tool_results: &[(String, ToolResult)],
    thread_id: &ThreadId,
    message_store: &Arc<M>,
) -> Result<(), AgentError>
where
    M: MessageStore,
{
    for (tool_id, result) in tool_results {
        let tool_result_msg = Message::tool_result(tool_id, &result.output, !result.success);
        if let Err(e) = message_store.append(thread_id, tool_result_msg).await {
            return Err(AgentError::new(
                format!("Failed to append tool result: {e}"),
                false,
            ));
        }
    }
    Ok(())
}
