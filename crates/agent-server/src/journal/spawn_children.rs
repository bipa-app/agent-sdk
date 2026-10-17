//! The spawn methods, each a wrapper over [`DurableStore::commit`].
//!
//! Each method commits one [`Write::SpawnChildren`], so a store applies
//! every spawn in its `commit` alone, and a decorating store that adds
//! behaviour to a spawn decorates the write its `commit` receives. No store
//! can implement a spawn any other way: these methods are not on
//! [`AgentTaskStore`].

use anyhow::{Context, Result};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
#[cfg(doc)]
use super::store::validate_mixed_children_spawn;
use super::store::{
    AgentTaskStore, MixedChildrenSpawn, SpawnedMixedChildren, SubagentBatchSpawn,
    SubagentInvocationSpawn,
};
use super::subagent_spawn_transaction::{
    SubagentSpawnCommit, SubagentSpawnEvent, SubagentSpawnOutcome,
};
use super::task::{AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, WorkerId};
#[cfg(doc)]
use super::task::{TaskKind, TaskStatus};
#[cfg(doc)]
use super::transition::Write;
use super::transition::{
    ChildSpawn, DurableStore, SpawnedChildren, SubagentSpawn, ToolChildrenSpawn, Transition,
    wrapper_error,
};

/// Commit `spawn` under `parent` and return the rows it wrote.
async fn spawn<S: AgentTaskStore + ?Sized>(
    store: &S,
    parent: &AgentTaskId,
    (worker, lease): (&WorkerId, &LeaseId),
    spawn: ChildSpawn,
    now: OffsetDateTime,
) -> Result<SpawnedChildren> {
    let transition =
        Transition::spawn_children(parent.clone(), worker.clone(), lease.clone(), spawn);
    let committed = DurableStore::commit(store, transition, now)
        .await
        .map_err(|rejected| wrapper_error("spawn", rejected))?;
    committed
        .spawned
        .into_iter()
        .next()
        .context("a spawn transition returns its spawned children")
}

/// The spawn methods of every [`AgentTaskStore`].
#[async_trait]
pub trait SpawnChildrenExt: AgentTaskStore {
    /// Spawn one subagent as one unit: the child thread row, the parent's
    /// `Running` → `WaitingOnChildren` transition, the invocation and child
    /// root rows, the delivered injections' completion, the parent's
    /// `SubagentProgress` start event, the child's `ThreadCreated`, and
    /// each thread's `thread_events_available` advisory.
    ///
    /// The worker's single-subagent spawn goes through this method.
    ///
    /// # Errors
    /// Same CAS and non-leaf envelope as
    /// [`Self::spawn_subagent_invocation`]; store-level write
    /// errors. The durable backends then write nothing.
    async fn commit_subagent_spawn(
        &self,
        params: SubagentSpawnCommit,
    ) -> Result<SubagentSpawnOutcome> {
        let SubagentSpawnCommit {
            delivered_injection_ids,
            parent_id,
            worker,
            lease,
            spawn: subagent,
            event,
            outbox_max_attempts,
            now,
        } = params;
        let spawned = spawn(
            self,
            &parent_id,
            (&worker, &lease),
            ChildSpawn::Subagent(SubagentSpawn {
                spawn: subagent,
                event,
                delivered_injection_ids,
                outbox_max_attempts,
            }),
            now,
        )
        .await?;
        let (invocation_task, child_root_task) = spawned
            .subagents
            .into_iter()
            .next()
            .context("subagent spawn wrote no invocation")?;
        Ok(SubagentSpawnOutcome {
            parent_task: spawned.parent,
            invocation_task,
            child_thread: spawned
                .child_threads
                .into_iter()
                .next()
                .context("subagent spawn created no child thread")?,
            child_root_task,
            committed_event: spawned
                .events
                .into_iter()
                .next()
                .context("subagent spawn committed no start event")?,
        })
    }

    /// Atomically persist a batch of [`TaskKind::ToolRuntime`] children
    /// under a running parent and transition the parent to
    /// [`TaskStatus::WaitingOnChildren`] in a single store write.
    ///
    /// This is the Phase 2.6 child-creation entry point. A successful
    /// call:
    ///
    /// 1. CAS-checks that the parent exists, is in
    ///    [`TaskStatus::Running`], is owned by `(worker, lease)`, and
    ///    is not a leaf kind.
    /// 2. Builds one fresh [`TaskKind::ToolRuntime`] row per entry in
    ///    `specs` via [`AgentTask::new_child`]. Each child inherits
    ///    the parent's `thread_id` and `root_id`, sets `depth =
    ///    parent.depth + 1`, starts in [`TaskStatus::Pending`] with
    ///    `attempt = 0`, and takes its retry budget from
    ///    [`ChildSpawnSpec::max_attempts`].
    /// 3. Inserts every child into the store under the same write
    ///    lock the parent transition will run on, so a crash between
    ///    children cannot leave a partially-spawned batch behind.
    /// 4. Transitions the parent to
    ///    [`TaskStatus::WaitingOnChildren`] with
    ///    `pending_child_count = specs.len()` and a typed
    ///    [`crate::journal::TaskState::WaitingOnChildren`] payload carrying
    ///    `continuation`. The parent's lease is dropped atomically
    ///    with the transition, so the row is invisible to every
    ///    acquisition / heartbeat / sweep call site the moment the
    ///    children become runnable.
    /// 5. Registers every child on the Phase 2.3 runnable index so
    ///    `acquire_next_runnable` picks them up in
    ///    `(created_at, id)` order alongside every other runnable
    ///    row.
    ///
    /// Returns `(parent, children)` with the rows as persisted.
    /// Callers can rely on the returned `children` vector matching
    /// the order of the input `specs`.
    ///
    /// `specs` must be non-empty. Zero-child spawns are rejected the
    /// same way [`AgentTask::wait_on_children`] rejects a zero
    /// `child_count`, because there is nothing to wait on and the
    /// parent would be stuck in [`TaskStatus::WaitingOnChildren`]
    /// forever.
    ///
    /// # Errors
    /// - `spawn rejected: task ... does not exist` — no parent row
    ///   with the supplied id.
    /// - `spawn rejected: task ... is not running` — parent in any
    ///   non-`Running` status.
    /// - `spawn rejected: worker mismatch` — `worker` does not match
    ///   the parent's current `worker_id`.
    /// - `spawn rejected: lease mismatch` — `lease` does not match
    ///   the parent's current `lease_id`.
    /// - `spawn rejected: parent ... is a leaf kind ...` — parent
    ///   is a [`TaskKind::ToolRuntime`] (leaf) and cannot have
    ///   children.
    /// - `spawn rejected: specs must be non-empty` — zero-child
    ///   spawn.
    /// - `spawn rejected: precompleted_result at index ... requires
    ///   spawn_mixed_children` — precompleted children need the mixed
    ///   primitive's atomic row + boundary-event transaction.
    /// - Schema errors from [`AgentTask::new_child`],
    ///   [`AgentTask::wait_on_children`], or [`AgentTask::validate`].
    #[allow(clippy::too_many_arguments)]
    async fn spawn_tool_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        specs: Vec<ChildSpawnSpec>,
        payload: SuspensionPayload,
        child_otel_traceparent: Option<String>,
        delivered_injection_ids: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<AgentTask>)> {
        let spawned = spawn(
            self,
            parent_id,
            (worker, lease),
            ChildSpawn::Tools(ToolChildrenSpawn {
                specs,
                payload,
                child_otel_traceparent,
                delivered_injection_ids,
            }),
            now,
        )
        .await?;
        Ok((spawned.parent, spawned.tool_children))
    }

    /// Atomically persist the durable task records for one subagent
    /// spawn.
    ///
    /// A successful call:
    ///
    /// 1. CAS-checks the running parent against `(worker, lease)`.
    /// 2. Allocates one parent-visible `subagent` invocation task
    ///    under the parent.
    /// 3. Allocates one fresh child-thread `root_turn` task on the
    ///    provided child thread.
    /// 4. Persists durable linkage by storing the authoritative spawn
    ///    spec plus the child thread / child root ids on the
    ///    invocation task's [`crate::journal::task_state::SubagentInvocationState`]
    ///    payload.
    /// 5. Transitions the parent to [`TaskStatus::WaitingOnChildren`]
    ///    with the invocation task id as its child-wait target.
    ///
    /// Callers must materialize the child thread row before invoking
    /// this method so durable backends can satisfy task → thread
    /// foreign keys while inserting the child root task.
    ///
    /// Returns `(parent, invocation, child_root)` with the rows as
    /// persisted.
    ///
    /// # Errors
    /// Same CAS and non-leaf envelope as
    /// [`Self::spawn_tool_children`], plus schema errors from
    /// [`AgentTask::new_subagent_invocation`].
    async fn spawn_subagent_invocation(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn_entry: SubagentInvocationSpawn,
        delivered_injection_ids: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let spawned = spawn(
            self,
            parent_id,
            (worker, lease),
            ChildSpawn::SubagentRows {
                spawn: spawn_entry,
                delivered_injection_ids,
            },
            now,
        )
        .await?;
        let (invocation, child_root) = spawned
            .subagents
            .into_iter()
            .next()
            .context("subagent spawn wrote no invocation")?;
        Ok((spawned.parent, invocation, child_root))
    }

    /// Atomically persist N durable subagent invocations under one
    /// parent transition.
    ///
    /// This is the fan-out flavour of
    /// [`Self::spawn_subagent_invocation`]. A successful
    /// call:
    ///
    /// 1. CAS-checks the running parent against `(worker, lease)`.
    /// 2. Allocates one parent-visible `subagent` invocation task per
    ///    entry in `spawns`.
    /// 3. Allocates one fresh child-thread `root_turn` task per
    ///    entry, on the per-entry `child_thread_id`.
    /// 4. Persists durable linkage on each invocation task's
    ///    [`crate::journal::task_state::SubagentInvocationState`]
    ///    payload.
    /// 5. Transitions the parent to [`TaskStatus::WaitingOnChildren`]
    ///    with `pending_child_count = spawns.len()` and the full
    ///    invocation-task-id list as its child-wait targets.
    ///
    /// One shared [`SuspensionPayload`] is used for all entries — the
    /// continuation envelope is the same parent suspension regardless
    /// of how many children fan out from it.
    ///
    /// Callers must materialize each child thread row before invoking
    /// this method so durable backends can satisfy task → thread
    /// foreign keys while inserting the per-entry child root tasks.
    /// They must also ensure no two entries share the same
    /// `child_thread_id` (the implementation rejects duplicates).
    ///
    /// Returns `(parent, [(invocation_i, child_root_i); spawns.len()])`
    /// with the rows as persisted, in the same order as `spawns`.
    ///
    /// # Errors
    /// Same CAS and non-leaf envelope as
    /// [`Self::spawn_subagent_invocation`], plus:
    /// - `spawn rejected: spawns must be non-empty` — empty input.
    /// - `spawn rejected: duplicate child_thread_id ...` — two
    ///   entries collide on the same child thread.
    /// - schema errors from [`AgentTask::new_subagent_invocation`].
    /// `events` carries one [`SubagentSpawnEvent`] per entry in
    /// `spawns` (same order). Durable backends commit the rows, every
    /// per-entry `SubagentProgress` start event, each child thread's
    /// `ThreadCreated`, and the outbox advisories in ONE transaction —
    /// a crash can never persist children whose start events were
    /// lost. The in-memory store commits events through its wired
    /// marker sink (bare stores commit rows only; in-memory rows have
    /// no crash story to protect). The returned committed events are
    /// the parent-journal start events, in entry order.
    async fn spawn_subagent_batch(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        batch: SubagentBatchSpawn,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>, Vec<CommittedEvent>)> {
        let spawned = spawn(
            self,
            parent_id,
            (worker, lease),
            ChildSpawn::SubagentBatch(batch),
            now,
        )
        .await?;
        Ok((spawned.parent, spawned.subagents, spawned.events))
    }

    /// Atomically persist a **mixed** batch — N durable subagent
    /// invocations plus M tool-runtime children — under one parent
    /// transition.
    ///
    /// This is the primitive behind a coordinator turn that emits
    /// subagent calls alongside ordinary tool calls (an LLM routinely
    /// mixes both in one response). It is the union of
    /// [`Self::spawn_subagent_batch`] and
    /// [`Self::spawn_tool_children`], not a new contract:
    ///
    /// 1. CAS-checks the running parent against `(worker, lease)` —
    ///    same envelope as both pure paths.
    /// 2. Allocates one invocation task + one child-thread `root_turn`
    ///    task per [`MixedChildrenSpawn::subagents`] entry, with the
    ///    same durable linkage `spawn_subagent_batch` persists.
    /// 3. Allocates one [`TaskKind::ToolRuntime`] child per
    ///    [`MixedChildrenSpawn::tool_children`] entry, each stamped with
    ///    its caller-supplied `spawn_index`.
    /// 4. Transitions the parent to [`TaskStatus::WaitingOnChildren`]
    ///    with `pending_child_count = N + M` and every invocation /
    ///    tool-child id as its child-wait targets.
    ///
    /// Every row is committed in one store mutation: on any failure
    /// path nothing is persisted, so a rejected mixed batch never
    /// leaves half a fan-out behind.
    ///
    /// The spawn indices of the two groups must together cover the
    /// shared suspension's `pending_tool_calls` exactly once each — see
    /// [`validate_mixed_children_spawn`], which every backend runs first.
    ///
    /// Callers must materialize each subagent child thread before
    /// invoking this method (same contract as `spawn_subagent_batch`).
    ///
    /// # Errors
    /// Same CAS, leaf-parent, duplicate-child-thread, and duplicate-id
    /// envelope as [`Self::spawn_subagent_batch`], plus every
    /// [`validate_mixed_children_spawn`] rejection.
    /// `events` carries one [`SubagentSpawnEvent`] per entry in
    /// `spawn.subagents` (same order), with the same atomicity contract
    /// as [`Self::spawn_subagent_batch`].
    async fn spawn_mixed_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        mixed: MixedChildrenSpawn,
        events: Vec<SubagentSpawnEvent>,
        now: OffsetDateTime,
    ) -> Result<SpawnedMixedChildren> {
        let spawned = spawn(
            self,
            parent_id,
            (worker, lease),
            ChildSpawn::Mixed {
                spawn: mixed,
                events,
            },
            now,
        )
        .await?;
        Ok(SpawnedMixedChildren {
            committed_events: spawned.events,
            parent: spawned.parent,
            subagents: spawned.subagents,
            tool_children: spawned.tool_children,
        })
    }
}

impl<S: AgentTaskStore + ?Sized> SpawnChildrenExt for S {}
