//! Phase 5.1 tool-runtime worker: bootstrap, execution, and lifecycle
//! for [`TaskKind::ToolRuntime`] child tasks.
//!
//! When a root turn suspends at the tool boundary (Phase 4.4), it
//! spawns one `ToolRuntime` child task per tool call. This module
//! provides the worker path that:
//!
//! 1. **Bootstraps** the child task by reading the parent's durable
//!    [`TaskState`] to reconstruct the [`PendingToolCallInfo`] for
//!    exactly one tool call.
//! 2. **Executes** the tool through a caller-supplied async callback
//!    (the tool registry and dispatch live outside `agent-server`).
//! 3. **Completes or fails** the child task through the journal,
//!    letting the store recompute the parent's `pending_child_count`
//!    so the parent can resume when every child reaches a terminal
//!    state.
//!
//! # Positional mapping
//!
//! Child tasks are spawned in the same order as the parent's
//! [`AgentContinuation::pending_tool_calls`] array. The bootstrap
//! step sorts siblings by `created_at` and matches each child to its
//! tool call by position. This coupling is safe because both arrays
//! are created atomically under the same store write lock in
//! [`AgentTaskStore::spawn_tool_children`].
//!
//! # Cancellation and lease awareness
//!
//! The worker checks the supplied [`CancellationToken`] before and
//! after tool execution. If cancelled, it returns early without
//! driving the child to a terminal state — the
//! [`AgentTaskStore::cancel_tree`] sweep handles cleanup. Lease
//! expiry is caught by the store's CAS guards on `complete_task` /
//! `fail_task`.
//!
//! [`TaskState`]: crate::journal::task_state::TaskState
//! [`PendingToolCallInfo`]: agent_sdk_core::PendingToolCallInfo
//! [`AgentContinuation`]: agent_sdk_core::AgentContinuation
//! [`AgentTaskStore::spawn_tool_children`]: crate::journal::store::AgentTaskStore::spawn_tool_children
//! [`AgentTaskStore::cancel_tree`]: crate::journal::store::AgentTaskStore::cancel_tree
//! [`CancellationToken`]: tokio_util::sync::CancellationToken

use std::future::Future;

use agent_sdk_core::{PendingToolCallInfo, ThreadId, ToolResult};
use anyhow::{Context, bail, ensure};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;

// ─────────────────────────────────────────────────────────────────────
// Bootstrap context
// ─────────────────────────────────────────────────────────────────────

/// Trusted bootstrapping inputs for a tool-runtime child task.
///
/// Built by [`resolve_tool_bootstrap`], which validates the child
/// row, reads the parent's durable state, and resolves exactly one
/// [`PendingToolCallInfo`] from the parent's continuation.
#[derive(Clone, Debug)]
pub struct ToolTaskBootstrap {
    /// The acquired child task row.
    pub child_task: AgentTask,
    /// The parent task row (read-only snapshot for context).
    pub parent_task: AgentTask,
    /// Thread the task family is bound to.
    pub thread_id: ThreadId,
    /// The child task's durable identity.
    pub task_id: AgentTaskId,
    /// The worker that owns the lease.
    pub worker_id: WorkerId,
    /// The current lease identifier.
    pub lease_id: LeaseId,
    /// The tool call this child is responsible for executing.
    pub tool_call: PendingToolCallInfo,
}

// ─────────────────────────────────────────────────────────────────────
// Bootstrap resolution
// ─────────────────────────────────────────────────────────────────────

/// Validate a running tool-runtime child task and resolve its
/// execution inputs from the parent's durable state.
///
/// # Preconditions
///
/// - `child.kind` must be [`TaskKind::ToolRuntime`].
/// - `child.status` must be [`TaskStatus::Running`].
/// - `child.worker_id` and `child.lease_id` must be present.
/// - `child.parent_id` must point to an existing parent task.
/// - The parent's [`TaskState`] must carry a continuation with a
///   `pending_tool_calls` entry at this child's positional index.
///
/// # Positional index
///
/// The child's position among its siblings determines which
/// `PendingToolCallInfo` it owns. Siblings are sorted by
/// `created_at` to match the spawn order from
/// [`AgentTaskStore::spawn_tool_children`].
/// # Errors
///
/// Returns an error if any precondition fails, the parent task
/// cannot be read, or the positional tool-call index is out of
/// bounds.
pub async fn resolve_tool_bootstrap(
    child: AgentTask,
    task_store: &dyn AgentTaskStore,
) -> anyhow::Result<ToolTaskBootstrap> {
    // ── Precondition: tool-runtime kind ──────────────────────────
    ensure!(
        child.kind == TaskKind::ToolRuntime,
        "tool bootstrap requires a ToolRuntime task, got {:?}",
        child.kind,
    );

    // ── Precondition: Running status ─────────────────────────────
    ensure!(
        child.status == TaskStatus::Running,
        "tool bootstrap requires a Running task, got {:?}",
        child.status,
    );

    // ── Extract lease fields ─────────────────────────────────────
    let worker_id = child
        .worker_id
        .clone()
        .context("Running task missing worker_id")?;
    let lease_id = child
        .lease_id
        .clone()
        .context("Running task missing lease_id")?;

    // ── Read the parent ─────────────────────────────────���────────
    let parent_id = child
        .parent_id
        .as_ref()
        .context("ToolRuntime task missing parent_id")?;

    let parent = task_store
        .get(parent_id)
        .await
        .context("failed to read parent task")?
        .with_context(|| format!("parent task {parent_id} does not exist"))?;

    // ── Extract continuation from parent state ───────────────────
    let continuation = match &parent.state {
        TaskState::WaitingOnChildren { continuation, .. }
        | TaskState::ReadyToResume { continuation, .. } => &continuation.payload,
        other => {
            bail!("parent task {parent_id} has unexpected state for tool bootstrap: {other:?}",)
        }
    };

    // ── Resolve positional index ─────────────────────────────────
    let mut siblings = task_store
        .list_children(parent_id)
        .await
        .context("failed to list sibling tasks")?;
    siblings.sort_by_key(|t| t.created_at);

    let child_index = siblings
        .iter()
        .position(|t| t.id == child.id)
        .with_context(|| {
            format!(
                "child task {} not found among siblings of parent {parent_id}",
                child.id
            )
        })?;

    ensure!(
        child_index < continuation.pending_tool_calls.len(),
        "child index {child_index} out of bounds for {} pending tool calls on parent {parent_id}",
        continuation.pending_tool_calls.len(),
    );

    let tool_call = continuation.pending_tool_calls[child_index].clone();
    let thread_id = child.thread_id.clone();
    let task_id = child.id.clone();

    Ok(ToolTaskBootstrap {
        child_task: child,
        parent_task: parent,
        thread_id,
        task_id,
        worker_id,
        lease_id,
        tool_call,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// Result of [`execute_tool_task`].
#[derive(Debug)]
pub enum ToolTaskOutcome {
    /// Tool executed successfully. The child task is now `Completed`
    /// and the parent's `pending_child_count` has been recomputed.
    Completed {
        child: AgentTask,
        parent: Option<AgentTask>,
        result: ToolResult,
    },
    /// Tool execution failed. The child task is now `Failed` and the
    /// parent's `pending_child_count` has been recomputed.
    Failed {
        child: AgentTask,
        parent: Option<AgentTask>,
        error: String,
    },
    /// The worker was cancelled before completing. The child task
    /// was **not** driven to a terminal state.
    Cancelled,
}

// ─────────────────────────────────────────────────────────────────────
// Execution
// ─────────────────────────────────────────────────────────────────────

/// Execute a single tool-runtime child task through the journal.
///
/// The `executor` callback receives the resolved [`PendingToolCallInfo`]
/// and produces the [`ToolResult`]. The worker owns the lifecycle:
///
/// 1. Check cancellation.
/// 2. Run `executor(tool_call)`.
/// 3. On success → [`AgentTaskStore::complete_task`].
/// 4. On failure → [`AgentTaskStore::fail_task`].
/// 5. Check cancellation again (post-execution).
///
/// The callback signature is intentionally minimal — the caller
/// (which has access to the tool registry, hooks, audit sink, etc.)
/// provides whatever execution logic it needs. This keeps
/// `agent-server` independent of `agent-sdk`'s runtime.
///
/// [`AgentTaskStore::complete_task`]: crate::journal::store::AgentTaskStore::complete_task
/// [`AgentTaskStore::fail_task`]: crate::journal::store::AgentTaskStore::fail_task
/// # Errors
///
/// Returns an error if the store's `complete_task` or `fail_task`
/// CAS check fails (e.g. lease expired or wrong worker).
pub async fn execute_tool_task<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    // ── Pre-execution cancellation check ─────────────────────────
    if cancel.is_cancelled() {
        return Ok(ToolTaskOutcome::Cancelled);
    }

    let task_id = &bootstrap.task_id;
    let worker_id = &bootstrap.worker_id;
    let lease_id = &bootstrap.lease_id;

    // ── Execute the tool ─────────────────────────────────────────
    let tool_result = executor(bootstrap.tool_call.clone()).await;

    // ── Post-execution cancellation check ────────────────────────
    if cancel.is_cancelled() {
        return Ok(ToolTaskOutcome::Cancelled);
    }

    // ── Drive to terminal state ──────────────────────────────────
    match tool_result {
        Ok(result) => {
            let (child, parent) = task_store
                .complete_task(task_id, worker_id, lease_id, now)
                .await
                .with_context(|| format!("complete_task failed for child {task_id}"))?;

            Ok(ToolTaskOutcome::Completed {
                child,
                parent,
                result,
            })
        }
        Err(err) => {
            let error_msg = format!("{err:#}");
            let (child, parent) = task_store
                .fail_task(task_id, worker_id, lease_id, error_msg.clone(), now)
                .await
                .with_context(|| format!("fail_task failed for child {task_id}"))?;

            Ok(ToolTaskOutcome::Failed {
                child,
                parent,
                error: error_msg,
            })
        }
    }
}
