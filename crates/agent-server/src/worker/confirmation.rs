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
//! [`ListenExecutionContext`]: agent_sdk_core::ListenExecutionContext
//! [`TaskState::AwaitingConfirmation`]: crate::journal::task_state::TaskState::AwaitingConfirmation
//! [`PendingToolCallInfo::listen_context`]: agent_sdk_core::PendingToolCallInfo

use agent_sdk_core::{ListenExecutionContext, PendingToolCallInfo, ToolResult};
use anyhow::{Context, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::future::Future;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::execution_intent::{
    ExecutionIntentStore, classify_tool_effect, guarded_tool_execution,
};
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, AgentTaskId, TaskKind, TaskStatus};
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
    /// User approved. Child is now [`TaskStatus::Pending`] and will
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
    /// User rejected. Child is now [`TaskStatus::Failed`].
    Rejected {
        child: AgentTask,
        parent: Option<AgentTask>,
        reason: String,
    },
    /// Confirmation timed out. Child is now [`TaskStatus::Failed`].
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
    /// now [`TaskStatus::Failed`].
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
/// - `bootstrap.child_task` must be in [`TaskStatus::Running`].
/// - `bootstrap.child_task.kind` must be [`TaskKind::ToolRuntime`].
/// - The parent task must carry a continuation (it should, since it
///   is in [`TaskStatus::WaitingOnChildren`]).
///
/// # Errors
///
/// Returns an error if the parent's continuation cannot be read or if
/// the store's CAS guards reject the pause (wrong worker/lease, task
/// not running, etc.).
pub async fn pause_tool_for_confirmation(
    bootstrap: &ToolTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    now: OffsetDateTime,
) -> anyhow::Result<AgentTask> {
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

    task_store
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
        })
}

// ─────────────────────────────────────────────────────────────────────
// Decision handling
// ─────────────────────────────────────────────────────────────────────

/// Apply a confirmation decision to an
/// [`AwaitingConfirmation`](TaskStatus::AwaitingConfirmation) child
/// task.
///
/// This is the entry point the external transport calls after
/// receiving the user's decision. It reads the child's paused state,
/// then:
///
/// - **Approved**: extracts the prepared operation, resumes the child
///   to [`TaskStatus::Pending`] via
///   [`AgentTaskStore::resume_from_confirmation`], and returns the
///   extracted prepared operation so the caller can pass it to the
///   worker on re-acquisition.
/// - **Rejected**: fails the child via
///   [`AgentTaskStore::reject_confirmation`] with a canonical
///   `confirmation_rejected:` error prefix.
/// - **Timeout**: fails the child via
///   [`AgentTaskStore::reject_confirmation`] with a canonical
///   `confirmation_timeout:` error prefix.
///
/// # Errors
///
/// Returns an error if the child does not exist, is not in
/// [`TaskStatus::AwaitingConfirmation`], or if the store transition
/// fails.
pub async fn apply_confirmation_decision(
    child_id: &AgentTaskId,
    decision: ConfirmationDecision,
    task_store: &dyn AgentTaskStore,
    now: OffsetDateTime,
) -> anyhow::Result<ConfirmationDecisionOutcome> {
    match decision {
        ConfirmationDecision::Approved => {
            // Read the current state before resume clears it.
            let child = task_store
                .get(child_id)
                .await
                .context("failed to read child task")?
                .with_context(|| format!("child task {child_id} not found"))?;

            ensure!(
                child.status == TaskStatus::AwaitingConfirmation,
                "expected AwaitingConfirmation for child {child_id}, got {:?}",
                child.status,
            );

            let prepared_operation = child.state.prepared_operation().cloned();

            // Resume: AwaitingConfirmation → Pending.
            let resumed = task_store
                .resume_from_confirmation(child_id, now)
                .await
                .with_context(|| format!("failed to resume child {child_id} from confirmation"))?;

            Ok(ConfirmationDecisionOutcome::Approved {
                child: resumed,
                prepared_operation,
            })
        }
        ConfirmationDecision::Rejected { reason } => {
            let error = format!("confirmation_rejected: {reason}");
            let (child, parent) = task_store
                .reject_confirmation(child_id, error, now)
                .await
                .with_context(|| format!("failed to reject child {child_id}"))?;

            Ok(ConfirmationDecisionOutcome::Rejected {
                child,
                parent,
                reason,
            })
        }
        ConfirmationDecision::Timeout => {
            let error =
                "confirmation_timeout: no decision received within the allowed window".to_owned();
            let (child, parent) = task_store
                .reject_confirmation(child_id, error, now)
                .await
                .with_context(|| format!("failed to time out child {child_id}"))?;

            Ok(ConfirmationDecisionOutcome::TimedOut { child, parent })
        }
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
///   [`TaskStatus::Running`] child (i.e. the child was re-acquired
///   after the approval resumed it to [`TaskStatus::Pending`]).
/// - The caller must have verified that the child was previously in
///   [`TaskStatus::AwaitingConfirmation`] and was approved. This
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
    task_store: &dyn AgentTaskStore,
    intent_store: &dyn ExecutionIntentStore,
    policy: &dyn ConfirmationPolicy,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ConfirmationResumeOutcome>
where
    F: FnOnce(PendingToolCallInfo) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    // ── Authoritative policy recheck ────────────────────────────
    let verdict = policy
        .check_policy(&bootstrap.tool_call)
        .await
        .context("authoritative policy recheck failed")?;

    if let PolicyVerdict::Denied { reason } = verdict {
        let error = format!("confirmation_policy_denied: {reason}");
        let (child, parent) = task_store
            .fail_task(
                &bootstrap.task_id,
                &bootstrap.worker_id,
                &bootstrap.lease_id,
                error,
                now,
            )
            .await
            .with_context(|| format!("fail_task after policy denial for {}", bootstrap.task_id))?;

        return Ok(ConfirmationResumeOutcome::PolicyDenied {
            child,
            parent,
            reason,
        });
    }

    // ── Execute through the guarded path ────────────────────────
    let effect_class = classify_tool_effect(&bootstrap.tool_call);
    let outcome = guarded_tool_execution(
        bootstrap,
        task_store,
        intent_store,
        cancel,
        effect_class,
        executor,
        now,
    )
    .await?;

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
