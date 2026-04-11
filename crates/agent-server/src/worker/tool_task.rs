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
//! [`AgentContinuation::pending_tool_calls`](agent_sdk_core::AgentContinuation::pending_tool_calls) array. Each child
//! carries a `spawn_index` field set during
//! [`AgentTaskStore::spawn_tool_children`] that records its position
//! within the batch. The bootstrap step reads this index directly
//! instead of relying on sibling sort order, making the mapping
//! correct regardless of the store's `list_children` return order.
//!
//! # Cancellation and lease awareness
//!
//! The worker checks the supplied [`CancellationToken`] **before**
//! tool execution. If cancelled at that point, it returns early
//! without driving the child to a terminal state — the
//! [`AgentTaskStore::cancel_tree`] sweep handles cleanup.
//!
//! After the executor returns, the result is **always** committed
//! to the journal regardless of cancellation state, because the
//! tool's side effects have already been applied and dropping the
//! result would cause the recovery matrix to re-execute the tool.
//! Lease expiry is caught by the store's CAS guards on
//! `complete_task` / `fail_task`.
//!
//! [`TaskState`]: crate::journal::task_state::TaskState
//! [`PendingToolCallInfo`]: agent_sdk_core::PendingToolCallInfo
//! [`AgentContinuation`]: agent_sdk_core::AgentContinuation
//! [`AgentTaskStore::spawn_tool_children`]: crate::journal::store::AgentTaskStore::spawn_tool_children
//! [`AgentTaskStore::cancel_tree`]: crate::journal::store::AgentTaskStore::cancel_tree
//! [`CancellationToken`]: tokio_util::sync::CancellationToken

use std::future::Future;

use agent_sdk_core::events::AgentEvent;
use agent_sdk_core::{PendingToolCallInfo, ThreadId, ToolResult};
use anyhow::{Context, bail, ensure};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::committed_event::CommittedEvent;
use crate::journal::event_repository::EventRepository;
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
/// The child's `spawn_index` field determines which
/// `PendingToolCallInfo` it owns. This index is set during
/// [`AgentTaskStore::spawn_tool_children`] and read directly,
/// making the mapping independent of `list_children` ordering.
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

    // ── Resolve positional index via spawn_index ─────────────────
    let spawn_index = child
        .spawn_index
        .context("ToolRuntime child missing spawn_index")?;

    let child_index = usize::try_from(spawn_index).context("spawn_index exceeds usize")?;

    ensure!(
        child_index < continuation.pending_tool_calls.len(),
        "child spawn_index {child_index} out of bounds for {} pending tool calls on parent {parent_id}",
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
        /// `ToolCallEnd` event committed after the state transition.
        committed_events: Vec<CommittedEvent>,
    },
    /// Tool execution failed. The child task is now `Failed` and the
    /// parent's `pending_child_count` has been recomputed.
    Failed {
        child: AgentTask,
        parent: Option<AgentTask>,
        error: String,
        /// `ToolCallEnd` event committed after the state transition.
        committed_events: Vec<CommittedEvent>,
    },
    /// The worker was cancelled before the executor ran. The child
    /// task was **not** driven to a terminal state.
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
/// 1. Check cancellation (bail early if already cancelled).
/// 2. Run `executor(tool_call)`.
/// 3. On success → [`AgentTaskStore::complete_task_with_result`].
/// 4. On failure → [`AgentTaskStore::fail_task`].
///
/// The callback signature is intentionally minimal — the caller
/// (which has access to the tool registry, hooks, audit sink, etc.)
/// provides whatever execution logic it needs. This keeps
/// `agent-server` independent of `agent-sdk`'s runtime.
///
/// [`AgentTaskStore::complete_task_with_result`]: crate::journal::store::AgentTaskStore::complete_task_with_result
/// [`AgentTaskStore::fail_task`]: crate::journal::store::AgentTaskStore::fail_task
/// # Errors
///
/// Returns an error if the store's `complete_task_with_result` or
/// `fail_task` CAS check fails (e.g. lease expired or wrong worker).
pub async fn execute_tool_task<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    event_repo: &dyn EventRepository,
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

    // NOTE: No post-execution cancellation check here. Once the
    // executor has returned, side effects have already been applied
    // and the result must be committed to the journal. Dropping it
    // would leave the child in Running state, causing the recovery
    // matrix to re-execute the tool on restart (double execution of
    // non-idempotent operations).

    // ── Drive to terminal state ──────────────────────────────────
    match tool_result {
        Ok(result) => {
            // Serialize the tool result so it is durably persisted on the
            // child row. The parent's resume path reads it back via
            // `aggregate_child_outcomes` without relying on in-memory state.
            let result_payload = serde_json::to_value(&result)
                .context("serialize tool result for durable storage")?;

            let (child, parent) = task_store
                .complete_task_with_result(task_id, worker_id, lease_id, result_payload, now)
                .await
                .with_context(|| format!("complete_task_with_result failed for child {task_id}"))?;

            // Commit ToolCallEnd event after state transition.
            let end_event = AgentEvent::tool_call_end(
                &bootstrap.tool_call.id,
                &bootstrap.tool_call.name,
                &bootstrap.tool_call.display_name,
                result.clone(),
            );
            let committed = event_repo
                .commit_event(&bootstrap.thread_id, end_event, now)
                .await
                .context("commit ToolCallEnd event")?;

            Ok(ToolTaskOutcome::Completed {
                child,
                parent,
                result,
                committed_events: vec![committed],
            })
        }
        Err(err) => {
            let error_msg = format!("{err:#}");
            let (child, parent) = task_store
                .fail_task(task_id, worker_id, lease_id, error_msg.clone(), now)
                .await
                .with_context(|| format!("fail_task failed for child {task_id}"))?;

            // Commit ToolCallEnd event with error result.
            let error_result = ToolResult {
                success: false,
                output: error_msg.clone(),
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
            let committed = event_repo
                .commit_event(&bootstrap.thread_id, end_event, now)
                .await
                .context("commit ToolCallEnd event on failure")?;

            Ok(ToolTaskOutcome::Failed {
                child,
                parent,
                error: error_msg,
                committed_events: vec![committed],
            })
        }
    }
}
