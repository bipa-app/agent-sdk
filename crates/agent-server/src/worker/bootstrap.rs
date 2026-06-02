//! Worker bootstrapping: resolve a running root task into the trusted
//! inputs the worker needs before execution begins.
//!
//! [`WorkerBootstrapContext`] is the single struct a worker receives
//! after task acquisition. It contains the acquired task row, the
//! resolved [`AgentDefinition`], and the validated lease identifiers.
//!
//! [`resolve_bootstrap_context`] is the only entry point for building
//! this context. It enforces all preconditions (root-turn kind,
//! Running status, valid lease fields) and calls the registry for
//! deterministic definition resolution.

use agent_sdk_foundation::ThreadId;
use anyhow::{Context, ensure};

use super::definition::AgentDefinition;
use super::registry::AgentDefinitionRegistry;
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};

// ─────────────────────────────────────────────────────────────────────
// Bootstrap context
// ─────────────────────────────────────────────────────────────────────

/// The trusted bootstrapping inputs a worker receives before execution.
///
/// Every field is validated and resolved by [`resolve_bootstrap_context`].
/// Later Phase 4 slices (staged message reconstruction, `run_turn`
/// invocation, tool-task dispatch) consume this context without
/// repeating the validation.
#[derive(Clone, Debug)]
pub struct WorkerBootstrapContext {
    /// The acquired root-turn task row.
    pub task: AgentTask,
    /// The deterministically resolved agent definition.
    pub definition: AgentDefinition,
    /// Thread the task is bound to (denormalized for convenience).
    pub thread_id: ThreadId,
    /// The task's durable identity.
    pub task_id: AgentTaskId,
    /// The worker that owns the lease.
    pub worker_id: WorkerId,
    /// The current lease identifier.
    pub lease_id: LeaseId,
}

// ─────────────────────────────────────────────────────────────────────
// Resolution
// ─────────────────────────────────────────────────────────────────────

/// Validate a running root task and resolve its [`AgentDefinition`]
/// into a complete [`WorkerBootstrapContext`].
///
/// # Preconditions
///
/// - `task.kind` must be [`TaskKind::RootTurn`].
/// - `task.status` must be [`TaskStatus::Running`].
/// - `task.worker_id` and `task.lease_id` must be present (enforced
///   by the `Running` status invariant).
///
/// # Errors
///
/// Returns an error if any precondition fails or if the registry
/// cannot resolve a definition for the task.
pub async fn resolve_bootstrap_context(
    task: AgentTask,
    registry: &dyn AgentDefinitionRegistry,
) -> anyhow::Result<WorkerBootstrapContext> {
    // ── Precondition: root turn ──────────────────────────────────
    ensure!(
        task.kind == TaskKind::RootTurn,
        "bootstrap requires a RootTurn task, got {:?}",
        task.kind,
    );

    // ── Precondition: Running status ─────────────────────────────
    ensure!(
        task.status == TaskStatus::Running,
        "bootstrap requires a Running task, got {:?}",
        task.status,
    );

    // ── Extract lease fields ─────────────────────────────────────
    let worker_id = task
        .worker_id
        .clone()
        .context("Running task missing worker_id")?;
    let lease_id = task
        .lease_id
        .clone()
        .context("Running task missing lease_id")?;

    // ── Resolve definition ───────────────────────────────────────
    let definition = registry
        .resolve(&task)
        .await
        .context("failed to resolve AgentDefinition for task")?;

    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();

    Ok(WorkerBootstrapContext {
        task,
        definition,
        thread_id,
        task_id,
        worker_id,
        lease_id,
    })
}
