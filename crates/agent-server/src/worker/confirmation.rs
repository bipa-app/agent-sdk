//! Phase 5.3 confirmation pause/resume, prepared operations, and
//! authoritative policy rechecks for tool-runtime child tasks.
//!
//! When a tool-runtime child task requires user confirmation before
//! execution (typically `Confirm`-tier tools), the worker must:
//!
//! 1. **Pause** the child task durably via the journal, persisting
//!    the continuation and any prepared listen/execute operation.
//! 2. **Wait** for an external confirmation decision (out of scope
//!    for this module — the transport layer delivers decisions).
//! 3. **Resume** by re-checking authoritative policy before execution
//!    proceeds, and handling prepared operations deterministically.
//!
//! # Confirmation lifecycle
//!
//! ```text
//!   Child Running ──▶ pause_tool_for_confirmation
//!                         │
//!                         ▼
//!                  AwaitingConfirmation
//!                         │
//!                    ┌────┼────────┐
//!                    │    │        │
//!                    ▼    ▼        ▼
//!              Approved  Rejected  Timeout
//!                  │       │        │
//!                  ▼       ▼        ▼
//!           resume_from  reject_ reject_
//!           confirmation confirm confirm
//!                  │       │        │
//!                  ▼       ▼        ▼
//!              Pending   Failed   Failed
//!                  │
//!             re-acquire
//!                  │
//!                  ▼
//!              Running
//!                  │
//!           policy recheck
//!                  │
//!             ┌────┴────┐
//!             ▼         ▼
//!          Allowed   Denied
//!             │         │
//!             ▼         ▼
//!          execute    fail
//!             │         │
//!             ▼         ▼
//!         Completed   Failed
//! ```
//!
//! # Prepared operations
//!
//! Listen-tier tools (`Confirm` tier with `listen_context`) stage a
//! prepared [`ListenExecutionContext`] before the confirmation pause.
//! The operation is persisted as part of the
//! [`TaskState::AwaitingConfirmation`] payload so it survives
//! restarts.
//!
//! On resume:
//! - **Approved**: the prepared operation is available for the
//!   executor via the re-bootstrapped
//!   [`PendingToolCallInfo::listen_context`].
//! - **Rejected/Timeout**: the prepared operation is abandoned when
//!   the child is failed. No executor callback runs.
//! - **Restart during wait**: the recovery matrix fails the child
//!   closed if a prepared operation exists (Phase 2.5 rule),
//!   preventing double-execution of the staged external operation.
//!
//! # Authoritative policy recheck
//!
//! User approval is necessary but not sufficient for execution. The
//! [`ConfirmationPolicy`] trait is re-checked at resume time so
//! policy changes between the pause and the resume are respected.
//! This is the "authoritative recheck" the Phase 5.3 contract
//! requires — the resume path trusts the journal and the policy,
//! not the replay of a previously approved decision.
//!
//! [`ListenExecutionContext`]: agent_sdk_foundation::ListenExecutionContext
//! [`TaskState::AwaitingConfirmation`]: crate::journal::task_state::TaskState::AwaitingConfirmation
//! [`PendingToolCallInfo::listen_context`]: agent_sdk_foundation::PendingToolCallInfo

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ListenExecutionContext, PendingToolCallInfo, ToolResult};
use anyhow::{Context, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::future::Future;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::committed_event::CommittedEvent;
use crate::journal::event_repository::EventRepository;
use crate::journal::execution_intent::{
    GuardedExecutionDeps, classify_tool_effect, guarded_tool_execution,
};
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, AgentTaskId, TaskKind};
use crate::worker::tool_task::{ToolTaskBootstrap, ToolTaskOutcome};

// ─────────────────────────────────────────────────────────────────────
// Confirmation decision
// ─────────────────────────────────────────────────────────────────────

/// Decision from the external confirmation transport.
///
/// This is the durable user decision that drives the resume path
/// for confirmation-paused tool tasks. The transport layer (out of
/// scope for Phase 5.3) is responsible for delivering one of these
/// decisions to [`apply_confirmation_decision`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfirmationDecision {
    /// User approved the tool execution. The resume path will
    /// re-check authoritative policy before proceeding.
    Approved,
    /// User explicitly rejected the tool execution.
    Rejected { reason: String },
    /// Confirmation timed out without a user decision within
    /// the allowed window.
    Timeout,
}

// ─────────────────────────────────────────────────────────────────────
// Policy verdict and trait
// ─────────────────────────────────────────────────────────────────────

/// Authoritative policy verdict at execution time.
///
/// Returned by [`ConfirmationPolicy::check_policy`] during the
/// resume path's authoritative recheck.
#[derive(Clone, Debug)]
pub enum PolicyVerdict {
    /// Tool execution is allowed by current server-side policy.
    Allowed,
    /// Tool execution is denied by current server-side policy.
    Denied { reason: String },
}

/// Authoritative policy recheck at resume time.
///
/// Implementations check whether the tool is still allowed given
/// current server-side policy, regardless of whether the user
/// approved the confirmation prompt. This is the mechanism that
/// prevents execution when a tool was disabled, permissions were
/// revoked, or rate limits were exceeded between the pause and
/// the resume.
///
/// # Contract
///
/// - **Stateless**: each call independently evaluates the current
///   policy. No caching of earlier decisions.
/// - **Fallible**: a policy-check infrastructure failure is surfaced
///   as an `Err`, which the resume path propagates — the tool is
///   not executed without a definitive policy verdict.
#[async_trait]
pub trait ConfirmationPolicy: Send + Sync {
    /// Check whether `tool_call` is allowed under the current
    /// server-side policy.
    async fn check_policy(&self, tool_call: &PendingToolCallInfo) -> anyhow::Result<PolicyVerdict>;
}

// ─────────────────────────────────────────────────────────────────────
// Decision outcome
// ─────────────────────────────────────────────────────────────────────

/// Outcome of [`apply_confirmation_decision`].
#[derive(Debug)]
pub enum ConfirmationDecisionOutcome {
    /// User approved. Child is now [`TaskStatus::Pending`](crate::journal::task::TaskStatus::Pending) and will
    /// be re-acquired by a worker for policy recheck + execution.
    ///
    /// `prepared_operation` is the listen/execute context that was
    /// persisted on the `TaskState::AwaitingConfirmation` payload,
    /// extracted before the resume cleared the typed state. `None`
    /// for non-listen tools.
    Approved {
        child: AgentTask,
        prepared_operation: Option<ListenExecutionContext>,
    },
    /// User rejected. Child is now [`TaskStatus::Failed`](crate::journal::task::TaskStatus::Failed).
    Rejected {
        child: AgentTask,
        parent: Option<AgentTask>,
        reason: String,
    },
    /// Confirmation timed out. Child is now [`TaskStatus::Failed`](crate::journal::task::TaskStatus::Failed).
    TimedOut {
        child: AgentTask,
        parent: Option<AgentTask>,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Resume outcome
// ─────────────────────────────────────────────────────────────────────

/// Outcome of [`resume_confirmed_tool`].
#[derive(Debug)]
pub enum ConfirmationResumeOutcome {
    /// Tool executed after approval and policy recheck.
    Executed(ToolTaskOutcome),
    /// Policy denied execution despite user approval. Child is
    /// now [`TaskStatus::Failed`](crate::journal::task::TaskStatus::Failed).
    PolicyDenied {
        child: AgentTask,
        parent: Option<AgentTask>,
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Pause path
// ─────────────────────────────────────────────────────────────────────

/// Pause a running tool-runtime child task for user confirmation.
///
/// Calls [`AgentTaskStore::pause_on_confirmation`] with the parent's
/// continuation envelope and the tool call's prepared operation (from
/// [`PendingToolCallInfo::listen_context`], if any).
///
/// # Preconditions
///
/// - `bootstrap.child_task` must be in [`TaskStatus::Running`](crate::journal::task::TaskStatus::Running).
/// - `bootstrap.child_task.kind` must be [`TaskKind::ToolRuntime`].
/// - The parent task must carry a continuation (it should, since it
///   is in [`TaskStatus::WaitingOnChildren`](crate::journal::task::TaskStatus::WaitingOnChildren)).
///
/// # Errors
///
/// Returns an error if the parent's continuation cannot be read or if
/// the store's CAS guards reject the pause (wrong worker/lease, task
/// not running, etc.).
pub async fn pause_tool_for_confirmation(
    bootstrap: &ToolTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    event_repo: &dyn EventRepository,
    now: OffsetDateTime,
) -> anyhow::Result<(AgentTask, Vec<CommittedEvent>)> {
    ensure!(
        bootstrap.child_task.kind == TaskKind::ToolRuntime,
        "confirmation pause requires a ToolRuntime task, got {:?}",
        bootstrap.child_task.kind,
    );

    let continuation = bootstrap
        .parent_task
        .state
        .continuation()
        .context("parent task has no continuation for confirmation pause")?
        .clone();

    let prepared_operation = bootstrap.tool_call.listen_context.clone();

    let paused_task = task_store
        .pause_on_confirmation(
            &bootstrap.task_id,
            &bootstrap.worker_id,
            &bootstrap.lease_id,
            continuation,
            prepared_operation,
            now,
        )
        .await
        .with_context(|| {
            format!(
                "failed to pause child {} for confirmation",
                bootstrap.task_id
            )
        })?;

    // Commit ToolRequiresConfirmation event after state transition.
    let confirm_event = AgentEvent::tool_requires_confirmation(
        &bootstrap.tool_call.id,
        &bootstrap.tool_call.name,
        &bootstrap.tool_call.display_name,
        bootstrap.tool_call.input.clone(),
        format!(
            "Tool {} requires confirmation",
            bootstrap.tool_call.display_name
        ),
    );
    let committed = event_repo
        .commit_event(&bootstrap.thread_id, confirm_event, now)
        .await
        .context("commit ToolRequiresConfirmation event")?;

    Ok((paused_task, vec![committed]))
}

// ─────────────────────────────────────────────────────────────────────
// Decision handling
// ─────────────────────────────────────────────────────────────────────

/// Apply a confirmation decision to an
/// [`AwaitingConfirmation`](crate::journal::task::TaskStatus::AwaitingConfirmation) child
/// task.
///
/// This is the entry point the external transport calls after
/// receiving the user's decision. It reads the child's paused state,
/// then:
///
/// - **Approved**: extracts the prepared operation, resumes the child
///   to [`TaskStatus::Pending`](crate::journal::task::TaskStatus::Pending) via
///   [`AgentTaskStore::resume_from_confirmation`], and returns the
///   extracted prepared operation so the caller can pass it to the
///   worker on re-acquisition.
/// - **Rejected**: fails the child via
///   [`AgentTaskStore::reject_confirmation`] with a canonical
///   `confirmation_rejected:` error prefix, then commits a failure
///   `ToolCallEnd` event so the tool lifecycle closes in the event
///   stream (mirroring the policy-denied path).
/// - **Timeout**: fails the child via
///   [`AgentTaskStore::reject_confirmation`] with a canonical
///   `confirmation_timeout:` error prefix, then commits a failure
///   `ToolCallEnd` event for the same reason.
///
/// Without the terminal `ToolCallEnd`, the journal would hold a
/// `ToolCallStart` + `ToolRequiresConfirmation` with no close, so
/// replay and live clients would render the tool as running forever.
///
/// # Errors
///
/// Returns an error if the child does not exist, is not in
/// [`TaskStatus::AwaitingConfirmation`](crate::journal::task::TaskStatus::AwaitingConfirmation), or if the store transition
/// fails. The closing `ToolCallEnd` commit is best-effort (logged on
/// failure) — the child is already durably failed by then, so an event
/// commit blip must not override that outcome.
pub async fn apply_confirmation_decision(
    child_id: &AgentTaskId,
    decision: ConfirmationDecision,
    task_store: &dyn AgentTaskStore,
    event_repo: &dyn EventRepository,
    now: OffsetDateTime,
) -> anyhow::Result<ConfirmationDecisionOutcome> {
    match decision {
        ConfirmationDecision::Approved => {
            // Atomically resume and extract the prepared operation
            // under a single write lock, closing the TOCTOU window
            // that a concurrent rejection could exploit.
            let (resumed, prepared_operation) = task_store
                .resume_from_confirmation(child_id, now)
                .await
                .with_context(|| format!("failed to resume child {child_id} from confirmation"))?;

            Ok(ConfirmationDecisionOutcome::Approved {
                child: resumed,
                prepared_operation,
            })
        }
        ConfirmationDecision::Rejected { reason } => {
            // Read the tool-call info before the reject clears the
            // child's typed state, so we can close the lifecycle below.
            let tool_call = paused_tool_call(child_id, task_store).await;
            let error = format!("{CONFIRMATION_REJECTED_PREFIX} {reason}");
            let (child, parent) = task_store
                .reject_confirmation(child_id, error.clone(), now)
                .await
                .with_context(|| format!("failed to reject child {child_id}"))?;

            emit_confirmation_tool_call_end(event_repo, tool_call, error, now).await;

            Ok(ConfirmationDecisionOutcome::Rejected {
                child,
                parent,
                reason,
            })
        }
        ConfirmationDecision::Timeout => {
            let tool_call = paused_tool_call(child_id, task_store).await;
            let error = format!(
                "{CONFIRMATION_TIMEOUT_PREFIX} no decision received within the allowed window"
            );
            let (child, parent) = task_store
                .reject_confirmation(child_id, error.clone(), now)
                .await
                .with_context(|| format!("failed to time out child {child_id}"))?;

            emit_confirmation_tool_call_end(event_repo, tool_call, error, now).await;

            Ok(ConfirmationDecisionOutcome::TimedOut { child, parent })
        }
    }
}

/// Read the [`PendingToolCallInfo`] (and the thread it belongs to) for
/// an `AwaitingConfirmation` child from its persisted continuation, so
/// the Rejected / Timeout arms can close the tool lifecycle with a
/// `ToolCallEnd` event.
///
/// Best-effort: returns `None` (the caller then skips the terminal
/// event) when the child is missing, no longer paused, carries no
/// continuation, or its `spawn_index` is out of range. The decision
/// itself still applies in those rare cases — observers just miss the
/// closing event.
async fn paused_tool_call(
    child_id: &AgentTaskId,
    task_store: &dyn AgentTaskStore,
) -> Option<(agent_sdk_foundation::ThreadId, PendingToolCallInfo)> {
    let child = task_store.get(child_id).await.ok()??;
    let continuation = child.state.continuation()?;
    let spawn_index = usize::try_from(child.spawn_index?).ok()?;
    let tool_call = continuation
        .payload
        .pending_tool_calls
        .get(spawn_index)?
        .clone();
    Some((child.thread_id.clone(), tool_call))
}

/// Commit a failure `ToolCallEnd` closing a rejected / timed-out
/// confirmation's tool lifecycle. No-op when `tool_call` is `None`.
///
/// Best-effort: the child is already durably failed, so an event-commit
/// failure is logged rather than propagated (it must not override the
/// reject outcome).
async fn emit_confirmation_tool_call_end(
    event_repo: &dyn EventRepository,
    tool_call: Option<(agent_sdk_foundation::ThreadId, PendingToolCallInfo)>,
    error: String,
    now: OffsetDateTime,
) {
    let Some((thread_id, tool_call)) = tool_call else {
        return;
    };
    let error_result = ToolResult {
        success: false,
        output: error,
        data: None,
        documents: Vec::new(),
        duration_ms: None,
    };
    let end_event = AgentEvent::tool_call_end(
        &tool_call.id,
        &tool_call.name,
        &tool_call.display_name,
        error_result,
    );
    if let Err(error) = event_repo.commit_event(&thread_id, end_event, now).await {
        log::warn!(
            "failed to commit ToolCallEnd after confirmation rejection/timeout \
             for tool {} on thread {thread_id}: {error:#}",
            tool_call.id,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Resume execution
// ─────────────────────────────────────────────────────────────────────

/// Execute a tool-runtime child task after confirmation approval and
/// authoritative policy recheck.
///
/// This function is the Phase 5.3 resume path that a worker calls
/// after re-acquiring a child task that was previously approved via
/// [`apply_confirmation_decision`]. It:
///
/// 1. **Re-checks authoritative policy** via
///    [`ConfirmationPolicy::check_policy`]. If denied, the child is
///    failed with a `confirmation_policy_denied:` error prefix
///    and no executor callback runs.
/// 2. **Executes** the tool through
///    [`guarded_tool_execution`],
///    which applies the Phase 5.2 durable intent guard on top of the
///    Phase 5.1 executor lifecycle.
///
/// # Preconditions
///
/// - `bootstrap` must be a valid [`ToolTaskBootstrap`] for a
///   [`TaskStatus::Running`](crate::journal::task::TaskStatus::Running) child (i.e. the child was re-acquired
///   after the approval resumed it to [`TaskStatus::Pending`](crate::journal::task::TaskStatus::Pending)).
/// - The caller must have verified that the child was previously in
///   [`TaskStatus::AwaitingConfirmation`](crate::journal::task::TaskStatus::AwaitingConfirmation) and was approved. This
///   function does not re-read the confirmation decision.
///
/// # Errors
///
/// Returns an error if the policy check fails (infrastructure
/// error), or if the underlying
/// [`guarded_tool_execution`]
/// or store CAS operations fail.
pub async fn resume_confirmed_tool<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    deps: &GuardedExecutionDeps<'_>,
    policy: &dyn ConfirmationPolicy,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ConfirmationResumeOutcome>
where
    F: FnOnce(PendingToolCallInfo, crate::worker::tool_task::ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    // ── Authoritative policy recheck ────────────────────────────
    let verdict = policy
        .check_policy(&bootstrap.tool_call)
        .await
        .context("authoritative policy recheck failed")?;

    if let PolicyVerdict::Denied { reason } = verdict {
        let error = format!("{CONFIRMATION_POLICY_DENIED_PREFIX} {reason}");
        let (child, parent) = deps
            .task_store
            .fail_task(
                &bootstrap.task_id,
                &bootstrap.worker_id,
                &bootstrap.lease_id,
                error.clone(),
                now,
            )
            .await
            .with_context(|| format!("fail_task after policy denial for {}", bootstrap.task_id))?;

        // Close the tool lifecycle with a ToolCallEnd event so observers
        // see a complete ToolCallStart → ToolCallEnd pair.
        let error_result = ToolResult {
            success: false,
            output: error,
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        };
        let end_event = AgentEvent::tool_call_end(
            &bootstrap.tool_call.id,
            &bootstrap.tool_call.name,
            &bootstrap.tool_call.display_name,
            error_result,
        );
        deps.event_repo
            .commit_event(&bootstrap.thread_id, end_event, now)
            .await
            .context("commit ToolCallEnd event after policy denial")?;

        return Ok(ConfirmationResumeOutcome::PolicyDenied {
            child,
            parent,
            reason,
        });
    }

    // ── Execute through the guarded path ────────────────────────
    let effect_class = classify_tool_effect(&bootstrap.tool_call);
    let outcome =
        guarded_tool_execution(bootstrap, deps, cancel, effect_class, executor, now).await?;

    Ok(ConfirmationResumeOutcome::Executed(outcome))
}

// ─────────────────────────────────────────────────────────────────────
// Error prefix constants
// ─────────────────────────────────────────────────────────────────────

/// Stable error prefix for rejected confirmations.
pub const CONFIRMATION_REJECTED_PREFIX: &str = "confirmation_rejected:";

/// Stable error prefix for timed-out confirmations.
pub const CONFIRMATION_TIMEOUT_PREFIX: &str = "confirmation_timeout:";

/// Stable error prefix for policy-denied confirmations.
pub const CONFIRMATION_POLICY_DENIED_PREFIX: &str = "confirmation_policy_denied:";
