use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use super::helpers::{catch_tool_panic, millis_to_u64, send_event, wrap_and_send};
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
use agent_sdk_foundation::audit::{
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
    let (mut tool_span, tool_kind) = start_tool_span(pending, ctx);

    let outcome = execute_tool_call_inner(
        pending,
        ctx,
        #[cfg(feature = "otel")]
        &mut tool_span,
    )
    .await;

    #[cfg(feature = "otel")]
    finish_tool_span(&mut tool_span, &outcome, &pending.name, tool_kind);

    outcome
}

#[cfg(feature = "otel")]
fn start_tool_span<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
) -> (opentelemetry::global::BoxedSpan, &'static str)
where
    Ctx: Send + Sync + 'static,
    H: AgentHooks,
{
    use crate::observability::{attrs, baggage, langfuse, spans};
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

    // Resolve the tool kind exactly once and reuse it on the span and
    // on the per-tool metric labels emitted in `finish_tool_span`. The
    // `unknown` value covers the rare case where the LLM emitted a
    // tool name no registry knows about — `execute_tool_call_inner`
    // surfaces that as a `Completed` outcome with an "Unknown tool:"
    // body, which we still want to be countable.
    let tool_kind: &'static str = if let Some(tool) = ctx.tools.get(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        "sync"
    } else if let Some(tool) = ctx.tools.get_async(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        "async"
    } else if let Some(tool) = ctx.tools.get_listen(&pending.name) {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tool.tier()),
        ));
        "listen"
    } else {
        "unknown"
    };
    span_attrs.push(KeyValue::new(attrs::SDK_TOOL_KIND, tool_kind));

    let mut span = spans::start_internal_span("execute_tool", span_attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Tool);
    (span, tool_kind)
}

#[cfg(feature = "otel")]
fn finish_tool_span(
    span: &mut opentelemetry::global::BoxedSpan,
    outcome: &ToolExecutionOutcome,
    tool_name: &str,
    tool_kind: &'static str,
) {
    use crate::observability::{attrs, metrics, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    let metrics_handle = metrics::Metrics::global();

    // Outcome string + optional duration are reused below for both
    // the span attribute and the metric labels.
    let (outcome_str, duration_ms) = match outcome {
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
            (outcome_str, result.duration_ms)
        }
        ToolExecutionOutcome::RequiresConfirmation { tool_name, .. } => {
            span.set_attribute(attrs::kv_bool(attrs::SDK_TOOL_CONFIRMATION_REQUIRED, true));
            span.set_attribute(KeyValue::new(
                attrs::SDK_TOOL_OUTCOME,
                "awaiting_confirmation",
            ));
            spans::add_event(
                span,
                "tool.confirmation_required",
                vec![KeyValue::new(attrs::GEN_AI_TOOL_NAME, tool_name.clone())],
            );
            ("awaiting_confirmation", None)
        }
        ToolExecutionOutcome::Error(error) => {
            span.set_attribute(KeyValue::new(attrs::ERROR_TYPE, "event_store"));
            span.set_status(opentelemetry::trace::Status::error(error.message.clone()));
            span.set_attribute(KeyValue::new(attrs::SDK_TOOL_OUTCOME, "error"));
            ("error", None)
        }
    };

    // Delegate metric emission to the shared recorder so the
    // daemon-hosted worker lands in the same series with the same
    // labels.
    metrics_handle.record_tool_execution(tool_name, tool_kind, outcome_str, duration_ms);

    span.end();
}

async fn execute_tool_call_inner<Ctx, H>(
    pending: &PendingToolCallInfo,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    #[cfg(feature = "otel")] tool_span: &mut opentelemetry::global::BoxedSpan,
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
        #[cfg(feature = "otel")]
        crate::observability::spans::add_event(
            tool_span,
            "tool.cached_result_returned",
            vec![opentelemetry::KeyValue::new(
                crate::observability::attrs::GEN_AI_TOOL_CALL_ID,
                pending.id.clone(),
            )],
        );
        return ToolExecutionOutcome::Completed {
            tool_id: pending.id.clone(),
            result: cached_result,
        };
    }

    if let Some(listen_tool) = ctx.tools.get_listen(&pending.name) {
        return execute_listen_tool_call(pending, listen_tool, ctx).await;
    }

    if let Some(async_tool) = ctx.tools.get_async(&pending.name) {
        return execute_async_tool_call(
            pending,
            async_tool,
            ctx,
            #[cfg(feature = "otel")]
            tool_span,
        )
        .await;
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
        // `ToolDecision` is `#[non_exhaustive]`; an unrecognized decision is
        // fail-safe: block the tool rather than executing it.
        _ => {
            handle_listen_tool_block(
                pending,
                listen_tool,
                ctx,
                &ready,
                "unrecognized tool decision".to_string(),
            )
            .await
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
    let controls = ToolBoundaryControls::from_tool_context(ctx.tool_context);
    let result =
        match execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
            race_boundary_result(
                &controls,
                tool_start,
                catch_tool_panic(listen_execute_ok(
                    listen_tool,
                    ctx.tool_context,
                    &ready.operation_id,
                    ready.revision,
                    tool_start,
                )),
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
            display_name: pending.display_name.clone(),
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
            display_name: pending.display_name.clone(),
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
    #[cfg(feature = "otel")] tool_span: &mut opentelemetry::global::BoxedSpan,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let tier = async_tool.tier();
    let result = match execute_with_idempotency(
        ctx.execution_store,
        pending,
        ctx.thread_id,
        // `execute_async_tool` already composes panic isolation inside
        // the cancel / timeout race, so no outer `catch_tool_panic` here.
        execute_async_tool(AsyncToolExecutionParams {
            pending,
            tool: async_tool,
            tool_context: ctx.tool_context,
            event_store: ctx.event_store,
            thread_id: ctx.thread_id,
            turn: ctx.turn,
            authority: ctx.authority,
            #[cfg(feature = "otel")]
            tool_span: Some(tool_span),
        }),
    )
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

/// `ToolResult` content the SDK synthesises when the run's cancel
/// token fires while a tool is still executing. Picked to read
/// naturally for the LLM's next turn and to be searchable in
/// dashboards as a single, well-known string.
const CANCELLED_BY_USER: &str = "Cancelled by user";

/// `ToolResult` content the SDK synthesises when a tool exceeds its
/// configured [`crate::types::AgentConfig::tool_timeout_ms`] budget.
/// Like [`CANCELLED_BY_USER`], this is a single well-known string so
/// it reads naturally for the LLM and is searchable in dashboards.
const TIMED_OUT_AT_BOUNDARY: &str = "Tool timed out";

/// SDK-boundary controls for racing a tool's `execute()` future.
///
/// Bundles the run's cancel token and the optional per-tool timeout so
/// every tool kind (sync, async, listen) can apply the same balanced
/// cancel / timeout synthesis through [`race_boundary_result`].
#[derive(Clone)]
pub(super) struct ToolBoundaryControls {
    cancel: Option<CancellationToken>,
    timeout: Option<std::time::Duration>,
}

impl ToolBoundaryControls {
    /// Read the cancel token and per-tool timeout off the tool context.
    pub(super) fn from_tool_context<Ctx>(tool_context: &ToolContext<Ctx>) -> Self {
        Self {
            cancel: tool_context.cancel_token(),
            timeout: tool_context.tool_timeout(),
        }
    }

    /// True when neither a cancel token nor a timeout is configured, so
    /// callers can take the plain `execute.await` fast path.
    const fn is_noop(&self) -> bool {
        self.cancel.is_none() && self.timeout.is_none()
    }
}

/// Why the SDK boundary stopped a tool before it produced its own result.
#[derive(Clone, Copy)]
enum BoundaryStop {
    Cancelled,
    TimedOut,
}

impl BoundaryStop {
    fn synthesize(self, tool_start: Instant) -> ToolResult {
        let duration = millis_to_u64(tool_start.elapsed().as_millis());
        match self {
            // Cancellation is synthesised as a *successful* result so the
            // pair stays balanced and the LLM reads it as a clean stop
            // rather than a tool failure. Timeout is an error: the tool
            // did not do its job in the allotted time.
            Self::Cancelled => ToolResult::success(CANCELLED_BY_USER).with_duration(duration),
            Self::TimedOut => ToolResult::error(TIMED_OUT_AT_BOUNDARY).with_duration(duration),
        }
    }
}

/// Future that resolves to the [`BoundaryStop`] reason whenever the
/// run's cancel token fires or the per-tool timeout elapses.
///
/// An absent cancel token / timeout is modelled as a future that never
/// resolves, so a configured-but-partial `controls` still arms only the
/// arm(s) that are set. Cancellation is preferred over timeout when both
/// are ready in the same poll.
async fn wait_for_boundary_stop(controls: &ToolBoundaryControls) -> BoundaryStop {
    let cancel = controls.cancel.clone();
    let cancelled = async move {
        match cancel {
            Some(token) => token.cancelled().await,
            None => std::future::pending::<()>().await,
        }
    };
    let timeout = controls.timeout;
    let timed_out = async move {
        match timeout {
            Some(duration) => tokio::time::sleep(duration).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        biased;
        () = cancelled => BoundaryStop::Cancelled,
        () = timed_out => BoundaryStop::TimedOut,
    }
}

/// Race an arbitrary tool future that yields a `Result<ToolResult,
/// AgentError>` against the run's cancel token and the per-tool timeout.
///
/// Returns the tool's own result if it finishes first, or a synthetic
/// balanced result ([`CANCELLED_BY_USER`] / [`TIMED_OUT_AT_BOUNDARY`],
/// as `Ok`) if the boundary fires. This is the shared primitive behind
/// the sync, async, and listen execution paths — every kind of tool
/// gets the same cancel + timeout semantics.
///
/// A boundary stop (cancel / timeout) is synthesised as
/// `Ok(balanced ToolResult)`; the inner `Err` only propagates when the
/// tool's own future produces it first. This is what lets the async
/// poll loop be cancelled mid-stream while still committing one
/// balanced `tool_result`. The `produce` future is the panic-isolated
/// tool future ([`catch_tool_panic`]), so a panic becomes an error
/// `ToolResult` that the race returns normally while a cancel / timeout
/// still wins.
///
/// When `controls` is a no-op (no cancel token, no timeout — e.g. unit
/// tests that never wire either) it short-circuits to `produce.await`
/// so the helper is a true superset of the previous behaviour.
async fn race_boundary_result<F>(
    controls: &ToolBoundaryControls,
    tool_start: Instant,
    produce: F,
) -> Result<ToolResult, AgentError>
where
    F: std::future::Future<Output = Result<ToolResult, AgentError>>,
{
    if controls.is_noop() {
        return produce.await;
    }
    tokio::select! {
        biased;
        result = produce => result,
        stop = wait_for_boundary_stop(controls) => Ok(stop.synthesize(tool_start)),
    }
}

/// Run a sync tool's `execute()` future and normalise its
/// `anyhow::Result` into a duration-stamped [`ToolResult`], wrapped as
/// `Ok` so it can be fed through [`catch_tool_panic`] (panic isolation)
/// and then [`race_boundary_result`] (cancel / timeout).
///
/// This is the *innermost* layer of the sync-tool composition. A panic
/// inside `execute` unwinds into the surrounding `catch_tool_panic`
/// (becoming an error `ToolResult`); a cancel / timeout is decided by
/// the surrounding `race_boundary_result`. Tools that observe
/// `ToolContext::cancel_token()` themselves and return early are
/// unaffected — their outcome wins the race before the boundary arms
/// fire.
async fn sync_execute_ok<F>(execute: F, tool_start: Instant) -> Result<ToolResult, AgentError>
where
    F: std::future::Future<Output = anyhow::Result<ToolResult>>,
{
    Ok(match execute.await {
        Ok(mut value) => {
            value.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
            value
        }
        Err(error) => ToolResult::error(format!("Tool error: {error:#}"))
            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
    })
}

/// Run a listen tool's `execute()` and normalise its `Result` into a
/// duration-stamped [`ToolResult`], wrapped as `Ok` so it can be fed
/// through [`catch_tool_panic`] then [`race_boundary_result`].
///
/// Pulled out so both the first-pass listen path and the confirmed
/// resume path share the same Ok/Err mapping and the same
/// panic-isolation + cancel/timeout composition. `operation_id` /
/// `revision` are passed directly because the two call sites carry the
/// listen metadata in different wrapper types (`ListenReady` vs
/// `ListenExecutionContext`).
async fn listen_execute_ok<Ctx>(
    listen_tool: &Arc<dyn ErasedListenTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    operation_id: &str,
    revision: u64,
    tool_start: Instant,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone,
{
    Ok(
        match listen_tool
            .execute(tool_context, operation_id, revision)
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
    let controls = ToolBoundaryControls::from_tool_context(ctx.tool_context);
    let result =
        match execute_with_idempotency(ctx.execution_store, pending, ctx.thread_id, async {
            race_boundary_result(
                &controls,
                tool_start,
                catch_tool_panic(sync_execute_ok(
                    tool.execute(ctx.tool_context, pending.input.clone()),
                    tool_start,
                )),
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

pub(super) async fn execute_async_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    async_tool: &Arc<dyn ErasedAsyncTool<Ctx>>,
    ctx: &ToolCallExecutionContext<'_, Ctx, H>,
    #[cfg(feature = "otel")] tool_span: &mut opentelemetry::global::BoxedSpan,
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
        ToolDecision::Allow => {
            complete_async_tool_call(
                pending,
                async_tool,
                ctx,
                #[cfg(feature = "otel")]
                tool_span,
            )
            .await
        }
        ToolDecision::Block(reason) => block_tool_call(pending, ctx, tier, reason).await,
        ToolDecision::RequiresConfirmation(description) => {
            require_tool_confirmation(pending, ctx, tier, description).await
        }
        // `ToolDecision` is `#[non_exhaustive]`; an unrecognized decision is
        // fail-safe: block the tool rather than executing it.
        _ => block_tool_call(pending, ctx, tier, "unrecognized tool decision".to_string()).await,
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
        // `ToolDecision` is `#[non_exhaustive]`; an unrecognized decision is
        // fail-safe: block the tool rather than executing it.
        _ => block_tool_call(pending, ctx, tier, "unrecognized tool decision".to_string()).await,
    }
}

/// Parameters for [`execute_async_tool`].
///
/// Bundled into a struct so the function stays under the
/// `too_many_arguments` clippy threshold.
pub(super) struct AsyncToolExecutionParams<'a, Ctx> {
    pub(super) pending: &'a PendingToolCallInfo,
    pub(super) tool: &'a Arc<dyn ErasedAsyncTool<Ctx>>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    /// Optional tool span; the resume-after-confirmation path passes
    /// `None` because that path does not yet open a tool span.
    #[cfg(feature = "otel")]
    pub(super) tool_span: Option<&'a mut opentelemetry::global::BoxedSpan>,
}

/// Execute an async tool call and stream progress until completion.
///
/// This function handles the two-phase execution of async tools:
/// 1. Execute the tool (returns immediately with Success/Failed/`InProgress`)
/// 2. If `InProgress`, stream status updates until completion
///
/// Both phases run under the SDK-boundary cancel / timeout race
/// ([`race_boundary_result`]). A cancel or timeout that fires during
/// the lightweight execute or the long-running poll loop drops the
/// in-flight future and synthesises a single balanced `ToolResult`, so
/// the async path never leaves an orphan `tool_use`.
///
/// The inner execution is also panic-isolated ([`catch_tool_panic`])
/// *inside* the race, so a panic in the async tool / poll loop becomes
/// an error `ToolResult` while a cancel / timeout still wins the race.
/// Because panic isolation is applied here, the async call sites must
/// **not** wrap [`execute_async_tool`] in `catch_tool_panic` again.
pub(super) async fn execute_async_tool<Ctx>(
    params: AsyncToolExecutionParams<'_, Ctx>,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone,
{
    let controls = ToolBoundaryControls::from_tool_context(params.tool_context);
    let tool_start = Instant::now();
    race_boundary_result(
        &controls,
        tool_start,
        catch_tool_panic(execute_async_tool_inner(params, tool_start)),
    )
    .await
}

async fn execute_async_tool_inner<Ctx>(
    params: AsyncToolExecutionParams<'_, Ctx>,
    tool_start: Instant,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone,
{
    let AsyncToolExecutionParams {
        pending,
        tool,
        tool_context,
        event_store,
        thread_id,
        turn,
        authority,
        #[cfg(feature = "otel")]
        tool_span,
    } = params;

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
            stream_async_tool_progress(StreamAsyncToolProgressParams {
                pending,
                tool,
                tool_context,
                event_store,
                thread_id,
                turn,
                authority,
                operation_id,
                initial_message: message,
                tool_start,
                #[cfg(feature = "otel")]
                tool_span,
            })
            .await
        }
    }
}

/// Parameters for [`stream_async_tool_progress`].
///
/// Inlined struct because the function is private and takes a mix of
/// borrows / owned operation metadata.
struct StreamAsyncToolProgressParams<'a, Ctx> {
    pending: &'a PendingToolCallInfo,
    tool: &'a Arc<dyn ErasedAsyncTool<Ctx>>,
    tool_context: &'a ToolContext<Ctx>,
    event_store: &'a Arc<dyn EventStore>,
    thread_id: &'a ThreadId,
    turn: usize,
    authority: &'a Arc<dyn EventAuthority>,
    operation_id: String,
    initial_message: String,
    tool_start: Instant,
    #[cfg(feature = "otel")]
    tool_span: Option<&'a mut opentelemetry::global::BoxedSpan>,
}

async fn stream_async_tool_progress<Ctx>(
    params: StreamAsyncToolProgressParams<'_, Ctx>,
) -> Result<ToolResult, AgentError>
where
    Ctx: Send + Sync + Clone,
{
    let StreamAsyncToolProgressParams {
        pending,
        tool,
        tool_context,
        event_store,
        thread_id,
        turn,
        authority,
        operation_id,
        initial_message,
        tool_start,
        #[cfg(feature = "otel")]
        mut tool_span,
    } = params;

    #[cfg(feature = "otel")]
    if let Some(span) = tool_span.as_deref_mut() {
        crate::observability::spans::add_event(
            span,
            "tool.async.started",
            vec![opentelemetry::KeyValue::new(
                crate::observability::attrs::GEN_AI_TOOL_NAME,
                pending.name.clone(),
            )],
        );
    }

    wrap_and_send(
        event_store,
        thread_id,
        turn,
        AgentEvent::tool_progress(
            &pending.id,
            &pending.name,
            &pending.display_name,
            "started",
            &initial_message,
            None,
        ),
        authority,
    )
    .await?;

    let mut stream = tool.check_status_stream(tool_context, &operation_id);
    #[cfg(feature = "otel")]
    let mut poll_index: u64 = 0;

    while let Some(status) = stream.next().await {
        match status {
            ErasedToolStatus::Progress {
                stage,
                message,
                data,
            } => {
                #[cfg(feature = "otel")]
                if let Some(span) = tool_span.as_deref_mut() {
                    poll_index += 1;
                    crate::observability::spans::add_event(
                        span,
                        "tool.async.poll",
                        vec![
                            opentelemetry::KeyValue::new(
                                crate::observability::attrs::SDK_TOOL_PROGRESS_STAGE,
                                stage.clone(),
                            ),
                            crate::observability::attrs::kv_i64(
                                crate::observability::attrs::SDK_TOOL_POLL_INDEX,
                                i64::try_from(poll_index).unwrap_or(i64::MAX),
                            ),
                        ],
                    );
                }
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
            ErasedToolStatus::Completed(mut result) | ErasedToolStatus::Failed(mut result) => {
                result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                return Ok(result);
            }
        }
    }

    Ok(
        ToolResult::error("Async tool stream ended without completion")
            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
    )
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
        let controls = ToolBoundaryControls::from_tool_context(ctx.tool_context);
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                race_boundary_result(
                    &controls,
                    tool_start,
                    catch_tool_panic(listen_execute_ok(
                        listen_tool,
                        ctx.tool_context,
                        &listen.operation_id,
                        listen.revision,
                        tool_start,
                    )),
                )
                .await
            },
        )
        .await;
    }

    if let Some(async_tool) = ctx.tools.get_async(&awaiting_tool.name) {
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            // `execute_async_tool` already composes panic isolation inside
            // the cancel / timeout race, so no outer `catch_tool_panic` here.
            execute_async_tool(AsyncToolExecutionParams {
                pending: awaiting_tool,
                tool: async_tool,
                tool_context: ctx.tool_context,
                event_store: ctx.event_store,
                thread_id: ctx.thread_id,
                turn: ctx.turn,
                authority: ctx.authority,
                #[cfg(feature = "otel")]
                tool_span: None,
            }),
        )
        .await;
    }

    if let Some(tool) = ctx.tools.get(&awaiting_tool.name) {
        let tool_start = Instant::now();
        let controls = ToolBoundaryControls::from_tool_context(ctx.tool_context);
        return execute_with_idempotency(
            ctx.execution_store,
            awaiting_tool,
            ctx.thread_id,
            async {
                race_boundary_result(
                    &controls,
                    tool_start,
                    catch_tool_panic(sync_execute_ok(
                        tool.execute(ctx.tool_context, awaiting_tool.input.clone()),
                        tool_start,
                    )),
                )
                .await
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
