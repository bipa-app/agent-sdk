use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;

use super::helpers::{millis_to_u64, send_event, wrap_and_send};
use super::idempotency::{execute_with_idempotency, try_get_cached_result};
use super::listen::{
    ListenWaitError, build_listen_confirmation_input, cancel_listen_with_warning,
    wait_for_listen_ready,
};
use super::types::{
    ConfirmedToolExecutionContext, ListenReady, ListenUpdateContext, ListenWaitParams,
    ToolCallExecutionContext, ToolExecutionOutcome,
};
use crate::authority::EventAuthority;
use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, ToolAuditSink, ToolDecision};
use crate::llm::{Content, ContentBlock, Message, Role};
use crate::stores::{EventStore, MessageStore};
use crate::tools::{
    ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus, ListenStopReason, ToolContext,
};
use crate::types::{
    AgentError, ListenExecutionContext, PendingToolCallInfo, ThreadId, ToolInvocation, ToolOutcome,
    ToolResult, ToolTier,
};
use agent_sdk_core::audit::{
    AuditProvenance, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams,
};

/// Build a [`ToolInvocation`] from a pending tool call and its tier.
fn build_invocation(pending: &PendingToolCallInfo, tier: ToolTier) -> ToolInvocation {
    ToolInvocation {
        tool_call_id: pending.id.clone(),
        tool_name: pending.name.clone(),
        display_name: pending.display_name.clone(),
        tier,
        requested_input: pending.input.clone(),
        effective_input: pending.effective_input.clone(),
        listen_context: pending.listen_context.clone(),
    }
}

/// Emit a single authoritative audit record for a tool lifecycle transition.
///
/// This is the replacement for the old "`post_tool_use`-only" audit path.
/// Every lifecycle variant flows through this helper so provenance, tier,
/// and identity are built the same way everywhere.
///
/// The `tier` argument is passed explicitly rather than read from
/// `pending.tier` because some callers need to override it — e.g. the
/// "unknown tool" path falls back to [`ToolTier::Confirm`] even if the
/// continuation happens to carry a stale observe-tier value.
async fn emit_audit(
    sink: &Arc<dyn ToolAuditSink>,
    provenance: &AuditProvenance,
    pending: &PendingToolCallInfo,
    tier: ToolTier,
    turn: usize,
    outcome: ToolAuditOutcome,
) {
    let record = ToolAuditRecord::new(ToolAuditRecordParams {
        tool_call_id: pending.id.clone(),
        tool_name: pending.name.clone(),
        display_name: pending.display_name.clone(),
        tier,
        requested_input: pending.input.clone(),
        effective_input: pending.effective_input.clone(),
        turn,
        provenance: provenance.clone(),
        outcome,
    });
    sink.record(record).await;
}

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
    #[cfg(feature = "otel")]
    let mut tool_span = start_tool_span(pending, ctx);

    let outcome = execute_tool_call_inner(pending, ctx).await;

    #[cfg(feature = "otel")]
    finish_tool_span(&mut tool_span, &outcome);

    outcome
}

#[cfg(feature = "otel")]
fn start_tool_span<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> opentelemetry::global::BoxedSpan
where
    Ctx: Send + Sync + 'static,
    H: AgentHooks,
{
    use crate::observability::{attrs, spans};
    use opentelemetry::KeyValue;

    let mut span_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "execute_tool"),
        KeyValue::new(attrs::GEN_AI_TOOL_NAME, pending.name.clone()),
        KeyValue::new(attrs::GEN_AI_TOOL_CALL_ID, pending.id.clone()),
    ];
    if !pending.display_name.is_empty() {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_DISPLAY_NAME,
            pending.display_name.clone(),
        ));
    }

    // Add tool metadata if the tool was found
    if let Some(tool) = ctx.tools.get(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        span_attrs.push(KeyValue::new(attrs::SDK_TOOL_KIND, "sync"));
    } else if let Some(tool) = ctx.tools.get_async(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        span_attrs.push(KeyValue::new(attrs::SDK_TOOL_KIND, "async"));
    } else if let Some(tool) = ctx.tools.get_listen(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        span_attrs.push(KeyValue::new(attrs::SDK_TOOL_KIND, "listen"));
    }

    spans::start_internal_span("execute_tool", span_attrs)
}

#[cfg(feature = "otel")]
fn finish_tool_span(span: &mut opentelemetry::global::BoxedSpan, outcome: &ToolExecutionOutcome) {
    use crate::observability::attrs;
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    match outcome {
        ToolExecutionOutcome::Completed { result, .. } => {
            let outcome_str = if result.output.starts_with("Unknown tool:") {
                span.set_attribute(KeyValue::new(attrs::ERROR_TYPE, "unknown_tool"));
                span.set_status(opentelemetry::trace::Status::error(result.output.clone()));
                "error"
            } else if result.output.starts_with("Blocked:") {
                "blocked"
            } else if result.output.starts_with("Rejected:") {
                "rejected"
            } else if result.success {
                "success"
            } else {
                "error"
            };
            span.set_attribute(KeyValue::new(attrs::SDK_TOOL_OUTCOME, outcome_str));
            if let Some(ms) = result.duration_ms {
                span.set_attribute(attrs::kv_i64(
                    attrs::SDK_TOOL_DURATION_MS,
                    i64::try_from(ms).unwrap_or(i64::MAX),
                ));
            }
        }
        ToolExecutionOutcome::RequiresConfirmation { .. } => {
            span.set_attribute(attrs::kv_bool(attrs::SDK_TOOL_CONFIRMATION_REQUIRED, true));
            span.set_attribute(KeyValue::new(
                attrs::SDK_TOOL_OUTCOME,
                "awaiting_confirmation",
            ));
        }
        ToolExecutionOutcome::Error(error) => {
            span.set_attribute(KeyValue::new(attrs::ERROR_TYPE, "event_store"));
            span.set_status(opentelemetry::trace::Status::error(error.message.clone()));
            span.set_attribute(KeyValue::new(attrs::SDK_TOOL_OUTCOME, "error"));
        }
    }

    span.end();
}

async fn execute_tool_call_inner<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(cached_result) = try_get_cached_result(ctx.execution_store, &pending.id).await {
        // Cached result from a prior in-flight attempt — audit before returning.
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            pending.tier,
            ctx.turn,
            ToolAuditOutcome::Cached {
                result: cached_result.clone(),
            },
        )
        .await;
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
        let result = ToolResult::error(format!("Unknown tool: {}", pending.name));
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            ToolTier::Confirm,
            ctx.turn,
            ToolAuditOutcome::Completed {
                result: result.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Completed {
            tool_id: pending.id.clone(),
            result,
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
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }

    let tool_start = Instant::now();
    let ready = match wait_for_listen_ready(ListenWaitParams {
        pending,
        tool: listen_tool,
        tool_context: ctx.tool_context,
        update_ctx: ListenUpdateContext {
            pending,
            hooks: ctx.hooks,
            event_store: ctx.event_store,
            thread_id: ctx.thread_id,
            turn: ctx.turn,
            authority: ctx.authority,
        },
    })
    .await
    {
        Ok(ready) => ready,
        Err(ListenWaitError::Tool(result)) => {
            return finish_listen_ready_failure(pending, ctx, tool_start, result).await;
        }
        Err(ListenWaitError::Event(error)) => return ToolExecutionOutcome::Error(error),
    };

    match ctx
        .hooks
        .pre_tool_use(&build_invocation(pending, tier))
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
    // A listen tool that failed to become ready is an *invalidated*
    // lifecycle outcome: no user confirmation ever happened because the
    // operation snapshot expired or the stream ended before ready.
    let tier = ctx
        .tools
        .get_listen(&pending.name)
        .map_or(ToolTier::Confirm, |t| t.tier());
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Invalidated {
            reason: result.output.clone(),
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
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
    let tier = listen_tool.tier();
    let result =
        match execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
            Ok(
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
                },
            )
        })
        .await
        {
            Ok(result) => result,
            Err(error) => {
                emit_audit(
                    ctx.audit_sink,
                    ctx.provenance,
                    pending,
                    tier,
                    ctx.turn,
                    ToolAuditOutcome::PersistenceFailed {
                        result: None,
                        error: error.message.clone(),
                    },
                )
                .await;
                return ToolExecutionOutcome::Error(error);
            }
        };
    ctx.hooks.post_tool_use(&pending.name, &result).await;
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Completed {
            result: result.clone(),
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
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
    let tier = listen_tool.tier();
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
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Blocked {
            reason: reason.clone(),
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
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
    let tier = ctx
        .tools
        .get_listen(&pending.name)
        .map_or(ToolTier::Confirm, |t| t.tier());
    let input = build_listen_confirmation_input(&pending.input, &ready);
    let listen_context = ListenExecutionContext {
        operation_id: ready.operation_id.clone(),
        revision: ready.revision,
        snapshot: ready.snapshot.clone(),
        expires_at: ready.expires_at,
    };
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::RequiresConfirmation {
            description: description.clone(),
            listen_context: Some(listen_context.clone()),
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::ToolRequiresConfirmation {
            id: pending.id.clone(),
            name: pending.name.clone(),
            input: input.clone(),
            description: description.clone(),
        },
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
    ToolExecutionOutcome::RequiresConfirmation {
        tool_id: pending.id.clone(),
        tool_name: pending.name.clone(),
        display_name: pending.display_name.clone(),
        input,
        description,
        listen_context: Some(listen_context),
    }
}

async fn send_tool_call_start_event<Ctx, H>(
    pending: &PendingToolCallInfo,
    tier: ToolTier,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> Result<(), AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
    )
    .await
}

async fn send_tool_call_end_event<Ctx, H>(
    pending: &PendingToolCallInfo,
    result: &ToolResult,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> Result<(), AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &pending.id,
            &pending.name,
            &pending.display_name,
            result.clone(),
        ),
    )
    .await
}

async fn block_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    tier: ToolTier,
    reason: String,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let result = ToolResult::error(format!("Blocked: {reason}"));
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Blocked {
            reason: reason.clone(),
        },
    )
    .await;
    if let Err(error) = send_tool_call_end_event(pending, &result, ctx).await {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
    }
}

async fn require_tool_confirmation<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    tier: ToolTier,
    description: String,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::RequiresConfirmation {
            description: description.clone(),
            listen_context: None,
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::ToolRequiresConfirmation {
            id: pending.id.clone(),
            name: pending.name.clone(),
            input: pending.input.clone(),
            description: description.clone(),
        },
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
    ToolExecutionOutcome::RequiresConfirmation {
        tool_id: pending.id.clone(),
        tool_name: pending.name.clone(),
        display_name: pending.display_name.clone(),
        input: pending.input.clone(),
        description,
        listen_context: None,
    }
}

async fn complete_async_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    async_tool: &Arc<dyn ErasedAsyncTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = async_tool.tier();
    let result =
        match execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
            execute_async_tool(
                pending,
                async_tool,
                ctx.tool_context,
                ctx.event_store,
                ctx.thread_id,
                ctx.turn,
                ctx.authority,
            )
            .await
        })
        .await
        {
            Ok(result) => result,
            Err(error) => {
                emit_audit(
                    ctx.audit_sink,
                    ctx.provenance,
                    pending,
                    tier,
                    ctx.turn,
                    ToolAuditOutcome::PersistenceFailed {
                        result: None,
                        error: error.message.clone(),
                    },
                )
                .await;
                return ToolExecutionOutcome::Error(error);
            }
        };
    ctx.hooks.post_tool_use(&pending.name, &result).await;
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Completed {
            result: result.clone(),
        },
    )
    .await;
    if let Err(error) = send_tool_call_end_event(pending, &result, ctx).await {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
    }
}

async fn complete_sync_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = tool.tier();
    let tool_start = Instant::now();
    let result =
        match execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
            Ok(
                match tool.execute(ctx.tool_context, pending.input.clone()).await {
                    Ok(mut value) => {
                        value.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        value
                    }
                    Err(error) => ToolResult::error(format!("Tool error: {error:#}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                },
            )
        })
        .await
        {
            Ok(result) => result,
            Err(error) => {
                emit_audit(
                    ctx.audit_sink,
                    ctx.provenance,
                    pending,
                    tier,
                    ctx.turn,
                    ToolAuditOutcome::PersistenceFailed {
                        result: None,
                        error: error.message.clone(),
                    },
                )
                .await;
                return ToolExecutionOutcome::Error(error);
            }
        };
    ctx.hooks.post_tool_use(&pending.name, &result).await;
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        pending,
        tier,
        ctx.turn,
        ToolAuditOutcome::Completed {
            result: result.clone(),
        },
    )
    .await;
    if let Err(error) = send_tool_call_end_event(pending, &result, ctx).await {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result),
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }
    ToolExecutionOutcome::Completed {
        tool_id: pending.id.clone(),
        result,
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
    if let Err(error) = send_tool_call_start_event(pending, tier, ctx).await {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }

    match ctx
        .hooks
        .pre_tool_use(&build_invocation(pending, tier))
        .await
    {
        ToolDecision::Allow => complete_async_tool_call(pending, async_tool, ctx).await,
        ToolDecision::Block(reason) => block_tool_call(pending, ctx, tier, reason).await,
        ToolDecision::RequiresConfirmation(description) => {
            require_tool_confirmation(pending, ctx, tier, description).await
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
    if let Err(error) = send_tool_call_start_event(pending, tier, ctx).await {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            pending,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: error.message.clone(),
            },
        )
        .await;
        return ToolExecutionOutcome::Error(error);
    }

    match ctx
        .hooks
        .pre_tool_use(&build_invocation(pending, tier))
        .await
    {
        ToolDecision::Allow => complete_sync_tool_call(pending, tool, ctx).await,
        ToolDecision::Block(reason) => block_tool_call(pending, ctx, tier, reason).await,
        ToolDecision::RequiresConfirmation(description) => {
            require_tool_confirmation(pending, ctx, tier, description).await
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
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    authority: &Arc<dyn EventAuthority>,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone,
{
    let tool_start = Instant::now();

    // Step 1: Execute (lightweight, returns quickly)
    let outcome = match tool.execute(tool_context, pending.input.clone()).await {
        Ok(o) => o,
        Err(e) => {
            return Ok(ToolResult::error(format!("Tool error: {e:#}"))
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis())));
        }
    };

    match outcome {
        // Synchronous completion - return immediately
        ToolOutcome::Success(mut result) | ToolOutcome::Failed(mut result) => {
            result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
            Ok(result)
        }

        // Async operation - stream status until completion
        ToolOutcome::InProgress {
            operation_id,
            message,
        } => {
            // Emit initial progress
            wrap_and_send(
                event_store,
                thread_id,
                turn,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "started",
                    &message,
                    None,
                ),
                authority,
            )
            .await?;

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
                            event_store,
                            thread_id,
                            turn,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                stage,
                                message,
                                data,
                            ),
                            authority,
                        )
                        .await?;
                    }
                    ErasedToolStatus::Completed(mut result)
                    | ErasedToolStatus::Failed(mut result) => {
                        result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        return Ok(result);
                    }
                }
            }

            // Stream ended without completion (shouldn't happen)
            Ok(
                ToolResult::error("Async tool stream ended without completion")
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
            )
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
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(reason) = rejection_reason {
        return handle_confirmed_tool_rejection(awaiting_tool, ctx, reason).await;
    }

    // Resume-time policy re-evaluation: re-check pre_tool_use at resume
    // so that policy changes since the original confirmation are
    // respected. The tier is read from the continuation's pending
    // metadata (`awaiting_tool.tier`) rather than re-resolved from the
    // registry because the continuation is the authoritative source at
    // this point.
    let tier = awaiting_tool.tier;

    let hook_decision = ctx
        .hooks
        .pre_tool_use(&build_invocation(awaiting_tool, tier))
        .await;
    // Resume-time policy re-evaluation is authoritative: if the hook
    // now returns Block the tool is rejected instead of executing.
    if let ToolDecision::Block(reason) = &hook_decision {
        log::warn!(
            "pre_tool_use returned Block for confirmed tool '{}': {reason} -- rejecting at resume time",
            awaiting_tool.name
        );
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            awaiting_tool,
            tier,
            ctx.turn,
            ToolAuditOutcome::Blocked {
                reason: reason.clone(),
            },
        )
        .await;
        let result = ToolResult::error(format!("Blocked at resume: {reason}"));
        return finish_confirmed_tool(awaiting_tool, ctx, result).await;
    }

    if let Some(cached_result) = try_get_cached_result(ctx.execution_store, &awaiting_tool.id).await
    {
        // Cached idempotent replay on the confirmed-resume path.
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            awaiting_tool,
            tier,
            ctx.turn,
            ToolAuditOutcome::Cached {
                result: cached_result.clone(),
            },
        )
        .await;
        return finish_confirmed_tool(awaiting_tool, ctx, cached_result).await;
    }

    let result = execute_confirmed_tool_inner(awaiting_tool, ctx).await?;
    finish_confirmed_tool(awaiting_tool, ctx, result).await
}

pub(super) async fn handle_confirmed_tool_rejection<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
    reason: String,
) -> Result<ToolResult, AgentError>
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

    // User rejection during confirmation is recorded as a Blocked
    // lifecycle event — the tool never ran, and the rejection reason is
    // the audit-facing explanation.
    //
    // Read the tier from the continuation's pending-call metadata
    // rather than re-looking it up from the registry — the registry
    // lookup pattern was a pre-Phase-1.6 artifact from when
    // `PendingToolCallInfo` didn't carry `tier`.
    let tier = awaiting_tool.tier;
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        awaiting_tool,
        tier,
        ctx.turn,
        ToolAuditOutcome::Blocked {
            reason: reason.clone(),
        },
    )
    .await;

    let result = ToolResult::error(format!("Rejected: {reason}"));
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &awaiting_tool.id,
            &awaiting_tool.name,
            &awaiting_tool.display_name,
            result.clone(),
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            awaiting_tool,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result.clone()),
                error: error.message.clone(),
            },
        )
        .await;
        return Err(error);
    }
    Ok(result)
}

pub(super) async fn execute_confirmed_tool_inner<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if let Some(listen_tool) = ctx.tools.get_listen(&awaiting_tool.name) {
        let Some(listen) = awaiting_tool.listen_context.as_ref() else {
            return Ok(ToolResult::error(format!(
                "Listen context missing for tool: {}",
                awaiting_tool.name
            )));
        };
        let tool_start = Instant::now();
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                Ok(
                    match listen_tool
                        .execute(ctx.tool_context, &listen.operation_id, listen.revision)
                        .await
                    {
                        Ok(mut value) => {
                            value.duration_ms =
                                Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            value
                        }
                        Err(error) => ToolResult::error(format!("Listen execute error: {error}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    },
                )
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
                execute_async_tool(
                    awaiting_tool,
                    async_tool,
                    ctx.tool_context,
                    ctx.event_store,
                    ctx.thread_id,
                    ctx.turn,
                    ctx.authority,
                )
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
                Ok(
                    match tool
                        .execute(ctx.tool_context, awaiting_tool.input.clone())
                        .await
                    {
                        Ok(mut value) => {
                            value.duration_ms =
                                Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            value
                        }
                        Err(error) => ToolResult::error(format!("Tool error: {error:#}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    },
                )
            },
        )
        .await;
    }

    Ok(ToolResult::error(format!(
        "Unknown tool: {}",
        awaiting_tool.name
    )))
}

pub(super) async fn finish_confirmed_tool<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    ctx: &ConfirmedToolExecutionContext<'_, Ctx, H>,
    result: ToolResult,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    ctx.hooks.post_tool_use(&awaiting_tool.name, &result).await;
    let tier = awaiting_tool.tier;
    // Terminal Completed record for the confirmed-tool path. This
    // covers tools that actually ran after user confirmation and tools
    // that were turned into synthetic error results by upstream logic
    // (e.g. "Blocked at resume: ..."). Callers that emitted a prior
    // Blocked record are still visible because `ToolAuditRecord` is a
    // stream, not a single row.
    emit_audit(
        ctx.audit_sink,
        ctx.provenance,
        awaiting_tool,
        tier,
        ctx.turn,
        ToolAuditOutcome::Completed {
            result: result.clone(),
        },
    )
    .await;
    if let Err(error) = send_event(
        ctx.event_store,
        ctx.thread_id,
        ctx.turn,
        ctx.hooks,
        ctx.authority,
        AgentEvent::tool_call_end(
            &awaiting_tool.id,
            &awaiting_tool.name,
            &awaiting_tool.display_name,
            result.clone(),
        ),
    )
    .await
    {
        emit_audit(
            ctx.audit_sink,
            ctx.provenance,
            awaiting_tool,
            tier,
            ctx.turn,
            ToolAuditOutcome::PersistenceFailed {
                result: Some(result.clone()),
                error: error.message.clone(),
            },
        )
        .await;
        return Err(error);
    }
    Ok(result)
}

/// Append tool results to message history.
///
/// All tool results from a single turn are batched into a single User message
/// containing multiple `ToolResult` content blocks. The Anthropic API requires
/// all `tool_results` from a batch to be in the same user message.
pub(super) async fn append_tool_results<M>(
    tool_results: &[(String, ToolResult)],
    thread_id: &ThreadId,
    message_store: &Arc<M>,
) -> Result<(), AgentError>
where
    M: MessageStore,
{
    if tool_results.is_empty() {
        return Ok(());
    }

    // Build tool result blocks, followed by any native binary attachments the
    // tool wants to pass back to the LLM (e.g. PDFs or images).
    // All blocks for a single agent turn are batched into one user message so
    // the Anthropic API receives them together, as required.
    let mut blocks: Vec<ContentBlock> = Vec::new();
    for (tool_id, result) in tool_results {
        blocks.push(ContentBlock::ToolResult {
            tool_use_id: tool_id.clone(),
            content: result.output.clone(),
            is_error: if result.success { None } else { Some(true) },
        });
        for doc in &result.documents {
            if doc.media_type.starts_with("image/") {
                blocks.push(ContentBlock::Image {
                    source: doc.clone(),
                });
            } else {
                blocks.push(ContentBlock::Document {
                    source: doc.clone(),
                });
            }
        }
    }

    let batch_msg = Message {
        role: Role::User,
        content: Content::Blocks(blocks),
    };

    if let Err(e) = message_store.append(thread_id, batch_msg).await {
        return Err(AgentError::new(
            format!("Failed to append tool results: {e}"),
            false,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::append_tool_results;
    use crate::llm::{Content, ContentBlock};
    use crate::stores::{InMemoryStore, MessageStore};
    use crate::types::{ThreadId, ToolResult};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_append_tool_results_preserves_raw_output_content() -> anyhow::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let thread_id = ThreadId::from_string("thread-structured");
        let result = ToolResult::error("command failed").with_duration(17);

        append_tool_results(&[("tool_1".to_string(), result)], &thread_id, &store).await?;

        let history = store.get_history(&thread_id).await?;
        let Content::Blocks(blocks) = &history[0].content else {
            anyhow::bail!("expected blocks")
        };

        let ContentBlock::ToolResult {
            content, is_error, ..
        } = &blocks[0]
        else {
            anyhow::bail!("expected tool result block")
        };

        assert_eq!(content, "command failed");
        assert_eq!(*is_error, Some(true));
        Ok(())
    }

    #[tokio::test]
    async fn test_append_tool_results_uses_image_block_for_images() -> anyhow::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let thread_id = ThreadId::from_string("thread-1");
        let result = ToolResult::success("attached image").with_documents(vec![
            crate::llm::ContentSource::new("image/png", "ZmFrZQ=="),
        ]);

        append_tool_results(&[("tool_1".to_string(), result)], &thread_id, &store).await?;

        let history = store.get_history(&thread_id).await?;
        assert_eq!(history.len(), 1);

        let Content::Blocks(blocks) = &history[0].content else {
            anyhow::bail!("expected blocks")
        };

        assert!(matches!(blocks[0], ContentBlock::ToolResult { .. }));
        assert!(matches!(blocks[1], ContentBlock::Image { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn test_append_tool_results_uses_document_block_for_pdfs() -> anyhow::Result<()> {
        let store = Arc::new(InMemoryStore::new());
        let thread_id = ThreadId::from_string("thread-2");
        let result = ToolResult::success("attached pdf").with_documents(vec![
            crate::llm::ContentSource::new("application/pdf", "ZmFrZQ=="),
        ]);

        append_tool_results(&[("tool_1".to_string(), result)], &thread_id, &store).await?;

        let history = store.get_history(&thread_id).await?;
        let Content::Blocks(blocks) = &history[0].content else {
            anyhow::bail!("expected blocks")
        };

        assert!(matches!(blocks[1], ContentBlock::Document { .. }));
        Ok(())
    }
}
