//! One state change and the events that report it, applied in one
//! transaction.
//!
//! A [`Transition`] is a value: guards that must hold, then writes. A
//! [`DurableStore`] applies it all or not at all. The guards are checked
//! before the first write, and the events are sequenced and given their
//! `thread_events_available` outbox row inside the same transaction, so a
//! crash can never leave a state change without the event that reports it
//! (design invariants I1–I4, bip `docs/durable-transitions-design-2026-10-12.md`).

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm;
use agent_sdk_foundation::{
    ContinuationEnvelope, ListenExecutionContext, TerminalReason, ThreadId,
};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::document::{DocScope, DocumentKey, DocumentOutcome, DocumentWrite, ExpectationFailed};
use super::effect::{EffectKey, EffectRecord, EffectWrite};
use super::epoch::EpochRecord;
use super::message::MessageProjection;
use super::store::{MixedChildrenSpawn, SubagentBatchSpawn, SubagentInvocationSpawn};
use super::subagent_spawn_transaction::SubagentSpawnEvent;
use super::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
};
use super::thread::{Thread, ThreadOperation};

/// Guards and writes applied in one transaction.
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub guards: Vec<Guard>,
    pub writes: Vec<Write>,
}

/// A condition every guard checks before the first write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Guard {
    /// The task exists, is in `status`, and when `lease` is given is held
    /// by exactly that worker and lease.
    TaskIs {
        task: AgentTaskId,
        status: TaskStatus,
        lease: Option<(WorkerId, LeaseId)>,
    },
    /// No effect row has `key` yet: the claim an
    /// [`EffectWrite::Start`] needs.
    EffectFree { key: EffectKey },
}

/// One write of a transition, applied in order.
#[derive(Clone, Debug)]
pub enum Write {
    /// Drive a running task to a terminal status. The backend also
    /// recomputes the parent and releases held siblings, in the same
    /// transaction. Needs a [`Guard::TaskIs`] with a lease on the same task.
    SettleTask { task: AgentTaskId, outcome: Settle },
    /// Fail a task parked on a confirmation (rejected or timed out) and
    /// recompute its parent. Needs a [`Guard::TaskIs`] for
    /// [`TaskStatus::AwaitingConfirmation`] on the same task.
    RejectConfirmation { task: AgentTaskId, error: String },
    /// Park a running task on a confirmation: it drops its lease and keeps
    /// the parent's continuation and the call's prepared operation. Needs a
    /// [`Guard::TaskIs`] with a lease on the same task.
    PauseOnConfirmation {
        task: AgentTaskId,
        continuation: Box<ContinuationEnvelope>,
        prepared_operation: Option<ListenExecutionContext>,
    },
    /// Re-park a steering exchange's parent on `reattach`, its still-live
    /// children, under `(worker, lease)`. Unlike the other task writes it
    /// takes no guard, and a guard on its task is refused: the backend
    /// checks the parent's ownership itself, after locking the children, so
    /// the store's child-before-parent lock order holds.
    ReparkAfterSteering {
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        payload: Box<SuspensionPayload>,
        reattach: Vec<AgentTaskId>,
    },
    /// Start or finish an effect (see [`super::effect`]). Its row is
    /// owned by its owner task's lock.
    Effect(EffectWrite),
    /// Append a compaction to `thread`'s message projection: `messages`, the
    /// compactor's output, replace the `source_message_count` messages of
    /// its effective view, and their last `retained_message_count` are that
    /// view's unchanged tail (see [`MessageProjection::append_compaction`]).
    AppendCompaction {
        thread: ThreadId,
        messages: Vec<llm::Message>,
        source_message_count: usize,
        retained_message_count: usize,
    },
    /// Events on `thread`, without sequences: the backend allocates them
    /// and writes one coalesced outbox row per thread.
    AppendEvents {
        thread: ThreadId,
        events: Vec<AgentEvent>,
    },
    /// Park a running parent on new children: the child rows, the parent's
    /// `Running` → `WaitingOnChildren` transition, the delivered injections'
    /// completion and the events the spawn reports. Needs a
    /// [`Guard::TaskIs`] that the parent is running under its lease.
    SpawnChildren {
        parent: AgentTaskId,
        spawn: Box<ChildSpawn>,
    },
    /// Put or retire one document (see [`super::document`]). Its
    /// [`DocExpect`](super::document::DocExpect) is checked against the
    /// row as locked; a failed one rejects the transition like a guard. A
    /// task document needs its task to exist. Backends apply document
    /// writes after the transition's other writes, in their given order.
    Document(DocumentWrite),
    /// Wake a parent parked on its children to run a steering exchange
    /// with `steering`, which must not be empty. A parent that is no
    /// longer parked (its last child fanned it in, an exchange is already
    /// running, or it is terminal) is left alone: the rest of the
    /// transition still commits, and [`Committed::steering_resumed`]
    /// reports `None`, so the steering waits for the next boundary.
    EnqueueSteeringResume {
        parent: AgentTaskId,
        steering: Vec<llm::ContentBlock>,
    },

    /// Record the start of a host process's epoch. Every lease the store
    /// grants after the commit carries this epoch (see
    /// [`super::epoch`]). An epoch id that already exists is refused.
    BeginEpoch(EpochRecord),
}

/// What a [`Write::SpawnChildren`] spawns, one shape per spawn method of
/// [`super::SpawnChildrenExt`].
#[derive(Clone, Debug)]
pub enum ChildSpawn {
    /// Tool-runtime children, one per pending call in input order, with no
    /// events.
    Tools(ToolChildrenSpawn),
    /// One subagent on a child thread the spawn creates, with its
    /// `SubagentProgress` start event on the parent's thread and the
    /// child's `ThreadCreated`.
    Subagent(SubagentSpawn),
    /// One subagent's rows on a child thread that already exists, with no
    /// events.
    SubagentRows {
        spawn: SubagentInvocationSpawn,
        delivered_injection_ids: Vec<AgentTaskId>,
    },
    /// Subagents on child threads that already exist, with one start event
    /// each.
    SubagentBatch(SubagentBatchSpawn),
    /// Subagents, tool children and re-attached children under one
    /// suspension, with the suspension's events.
    Mixed {
        spawn: MixedChildrenSpawn,
        events: Vec<SubagentSpawnEvent>,
    },
}

/// The tool children of a [`ChildSpawn::Tools`].
#[derive(Clone, Debug)]
pub struct ToolChildrenSpawn {
    pub specs: Vec<ChildSpawnSpec>,
    pub payload: SuspensionPayload,
    pub child_otel_traceparent: Option<String>,
    /// Boundary-injection rows this turn already delivered.
    pub delivered_injection_ids: Vec<AgentTaskId>,
}

/// The subagent of a [`ChildSpawn::Subagent`].
#[derive(Clone, Debug)]
pub struct SubagentSpawn {
    pub spawn: SubagentInvocationSpawn,
    pub event: SubagentSpawnEvent,
    /// Boundary-injection rows this turn already delivered.
    pub delivered_injection_ids: Vec<AgentTaskId>,
    /// The attempts of the start event's outbox row.
    pub outbox_max_attempts: u32,
}

impl ChildSpawn {
    /// The child threads of the subagents this spawn creates, in input
    /// order.
    #[must_use]
    pub fn child_threads(&self) -> Vec<&ThreadId> {
        match self {
            Self::Tools(_) => Vec::new(),
            Self::Subagent(SubagentSpawn { spawn, .. }) | Self::SubagentRows { spawn, .. } => {
                vec![&spawn.child_thread_id]
            }
            Self::SubagentBatch(batch) => batch
                .spawns
                .iter()
                .map(|spawn| &spawn.child_thread_id)
                .collect(),
            Self::Mixed { spawn, .. } => spawn
                .subagents
                .iter()
                .map(|spawn| &spawn.child_thread_id)
                .collect(),
        }
    }

    /// Whether the spawn creates its child thread rows rather than
    /// requiring them to exist.
    #[must_use]
    pub const fn creates_child_threads(&self) -> bool {
        match self {
            Self::Subagent(_) => true,
            Self::Tools(_)
            | Self::SubagentRows { .. }
            | Self::SubagentBatch(_)
            | Self::Mixed { .. } => false,
        }
    }

    /// Refuse a malformed spawn before any backend opens a transaction, so
    /// bad input loses to no purge fence, lease check or row lock: an empty
    /// batch, an event count that does not match the subagents, tool
    /// specs the pure tool spawn cannot persist, and a mixed batch that
    /// does not cover its pending calls or whose precompleted children lack
    /// their boundary events. A re-attached child that is missing or not
    /// the parent's is refused later, under its row lock.
    ///
    /// # Errors
    /// The same `spawn rejected: …` error the spawn method returned for
    /// that input before it became a wrapper.
    pub fn validate_shape(&self) -> anyhow::Result<()> {
        use super::store::{
            validate_mixed_children_spawn, validate_precompleted_boundary_events,
            validate_subagent_event_count, validate_tool_child_specs_for_pure_spawn,
        };

        match self {
            Self::Tools(tools) => {
                validate_tool_child_specs_for_pure_spawn(&tools.specs)?;
                super::sibling_order::pure_spawn_waits(&tools.specs)?;
            }
            Self::Subagent(_) | Self::SubagentRows { .. } => {}
            Self::SubagentBatch(batch) => {
                anyhow::ensure!(
                    !batch.spawns.is_empty(),
                    "spawn rejected: spawns must be non-empty"
                );
                anyhow::ensure!(
                    batch.events.len() == batch.spawns.len(),
                    "spawn rejected: {} events for {} spawns",
                    batch.events.len(),
                    batch.spawns.len()
                );
            }
            Self::Mixed { spawn, events } => {
                validate_mixed_children_spawn(spawn)?;
                validate_precompleted_boundary_events(spawn)?;
                validate_subagent_event_count(events.len(), spawn.subagents.len())?;
            }
        }
        Ok(())
    }

    /// The boundary-injection rows the spawn completes.
    #[must_use]
    pub fn delivered_injection_ids(&self) -> &[AgentTaskId] {
        match self {
            Self::Tools(tools) => &tools.delivered_injection_ids,
            Self::Subagent(subagent) => &subagent.delivered_injection_ids,
            Self::SubagentRows {
                delivered_injection_ids,
                ..
            } => delivered_injection_ids,
            Self::SubagentBatch(batch) => &batch.delivered_injection_ids,
            Self::Mixed { spawn, .. } => &spawn.delivered_injection_ids,
        }
    }

    /// The existing children the spawn re-attaches to its parent.
    #[must_use]
    pub fn reattached(&self) -> Vec<&AgentTaskId> {
        match self {
            Self::Mixed { spawn, .. } => spawn
                .reattach
                .iter()
                .map(|attached| &attached.child_id)
                .collect(),
            Self::Tools(_)
            | Self::Subagent(_)
            | Self::SubagentRows { .. }
            | Self::SubagentBatch(_) => Vec::new(),
        }
    }
}

/// The rows a [`Write::SpawnChildren`] wrote, as persisted.
#[derive(Clone, Debug)]
pub struct SpawnedChildren {
    /// The parent, waiting on its children.
    pub parent: AgentTask,
    /// `(invocation, child_root)` per subagent, in input order.
    pub subagents: Vec<(AgentTask, AgentTask)>,
    /// The tool-runtime children, in input order.
    pub tool_children: Vec<AgentTask>,
    /// The child threads the spawn created ([`ChildSpawn::Subagent`]).
    pub child_threads: Vec<Thread>,
    /// The events the spawn committed on the parent's thread, in sequence
    /// order. A child's `ThreadCreated` is written but not returned.
    pub events: Vec<CommittedEvent>,
}

/// A row a transition locks before its first write (design §6.2, step 2).
///
/// A backend that takes row locks takes every thread lock, ordered by
/// [`ThreadTier`] and then by thread key, and then every task lock, a child
/// before its parent. A cancel locks the threads of a tree parent first,
/// discovering each subagent's thread only by walking down to it, so a
/// transition that also locks a parent's thread and its subagents' threads
/// locks them in that same tree order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lock {
    /// A thread row. With `fence`, the thread must allow its operation;
    /// with `create`, the row is created when it does not exist.
    Thread {
        thread: ThreadId,
        tier: ThreadTier,
        fence: Option<Fence>,
        create: bool,
    },
    /// The thread row of `task`, at [`ThreadTier::Own`]. With `fence`, the
    /// thread must allow its operation.
    ThreadOf {
        task: AgentTaskId,
        fence: Option<Fence>,
    },
    /// A task row.
    Task(AgentTaskId),
    /// A [`DocScope::Host`] document row, locked in key order after every
    /// other row the transition writes: a backend applies a transition's
    /// document writes after all its other writes, so a settle's parent row
    /// is never taken after a host document. Thread and task documents need
    /// no lock of their own: their thread row (and a task document's task
    /// row) is locked first, with the append fence.
    Document(DocumentKey),
}

/// The purge fence a thread lock checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fence {
    /// The operation the thread's status must allow.
    pub operation: ThreadOperation,
    /// What a thread with no row means for the operation.
    pub missing: MissingThread,
}

/// What a [`Fence`] does with a thread that has no row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingThread {
    /// Refuse the transition with
    /// [`ThreadNotFound`](super::thread::ThreadNotFound): a spawn needs its
    /// threads to exist.
    Refuse,
    /// Let the transition through: thread rows are created lazily, and a
    /// purge always leaves a `Deleted` row behind, so a missing row is never
    /// a purged thread. Appends allow it.
    Allow,
}

impl Fence {
    /// An append's fence: a purged thread refuses it, a missing one does
    /// not.
    pub const APPEND: Self = Self {
        operation: ThreadOperation::Append,
        missing: MissingThread::Allow,
    };
    /// A fork source's fence: a purged source refuses the fork, a missing
    /// one does not.
    pub const FORK_SOURCE: Self = Self {
        operation: ThreadOperation::Fork,
        missing: MissingThread::Allow,
    };
    /// A spawn's fence: a purged or missing thread refuses it.
    pub const SPAWN: Self = Self {
        operation: ThreadOperation::Spawn,
        missing: MissingThread::Refuse,
    };

    /// Whether a backend checks this fence after the transition's guards.
    /// Every backend checks an append's fence after the guards, so a stale
    /// lease is reported before a purge. A spawn's fence keeps the order
    /// each backend's spawn methods checked it in before they became
    /// wrappers (Postgres: before the guards; SQLite: the parent's before
    /// and the children's after; in memory: before the write lock).
    #[must_use]
    pub const fn after_guards(&self) -> bool {
        match self.operation {
            ThreadOperation::Append => true,
            ThreadOperation::Create
            | ThreadOperation::Fork
            | ThreadOperation::Submit
            | ThreadOperation::Spawn
            | ThreadOperation::Promote
            | ThreadOperation::Acquire
            | ThreadOperation::Commit => false,
        }
    }

    /// Check `thread`, the row as locked (`None` when it has none).
    ///
    /// # Errors
    /// [`ThreadNotFound`](super::thread::ThreadNotFound) for a missing row
    /// under [`MissingThread::Refuse`], and the thread's
    /// [`ThreadStateConflict`](super::thread::ThreadStateConflict) when its
    /// status refuses the operation.
    pub fn check(&self, thread_id: &ThreadId, thread: Option<&Thread>) -> anyhow::Result<()> {
        match (thread, self.missing) {
            (Some(thread), _) => thread
                .require_active(self.operation)
                .map_err(anyhow::Error::new),
            (None, MissingThread::Allow) => Ok(()),
            (None, MissingThread::Refuse) => {
                Err(anyhow::Error::new(super::thread::ThreadNotFound {
                    thread_id: thread_id.clone(),
                }))
            }
        }
    }
}

/// Where a thread lock falls in a transition's thread order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreadTier {
    /// The threads the transition's own rows live on.
    Own,
    /// The threads of the subagents it spawns, locked after every thread
    /// of its own.
    Spawned,
}

/// How [`Write::SettleTask`] ends a task.
#[derive(Clone, Debug)]
pub enum Settle {
    Complete {
        result: serde_json::Value,
    },
    Fail {
        error: String,
    },
    FailWithReason {
        error: String,
        reason: TerminalReason,
        /// What the parent reads in place of the error, as a
        /// [`ToolResult`](agent_sdk_foundation::ToolResult) value; `None`
        /// leaves it the error alone.
        result: Option<serde_json::Value>,
    },
    /// Complete with no result payload, as a root turn completes.
    CompleteWithoutResult,
}

/// What a committed transition produced, in write order.
#[derive(Clone, Debug, Default)]
pub struct Committed {
    /// For each [`Write::SettleTask`] and [`Write::RejectConfirmation`]:
    /// the terminal task and its parent
    /// after the parent was recomputed (`None` for a root).
    pub settled: Vec<(AgentTask, Option<AgentTask>)>,
    /// For each [`Write::PauseOnConfirmation`]: the parked task.
    pub paused: Vec<AgentTask>,
    /// For each [`Write::ReparkAfterSteering`]: the re-parked parent.
    pub reparked: Vec<AgentTask>,
    /// For each [`Write::AppendCompaction`]: the compacted projection.
    pub compacted: Vec<MessageProjection>,
    /// For each [`Write::SpawnChildren`]: the rows it wrote.
    pub spawned: Vec<SpawnedChildren>,
    /// Every event the transition appended, with its sequence, except a
    /// spawned child's `ThreadCreated`.
    pub events: Vec<CommittedEvent>,
    /// For each [`Write::Document`]: the document's live revision after it.
    pub documents: Vec<DocumentOutcome>,
    /// For each [`Write::EnqueueSteeringResume`]: the woken parent, or
    /// `None` when it was no longer parked.
    pub steering_resumed: Vec<Option<AgentTask>>,
}

/// Why a transition wrote nothing.
#[derive(Debug, thiserror::Error)]
pub enum Rejected {
    /// The guard at `index` did not hold. An `index` at or past the
    /// number of guards names the write at `index - guards.len()`, whose
    /// own expectation (a document's [`DocExpect`](super::document::DocExpect))
    /// did not hold when the backend read its row.
    #[error("transition rejected by guard {index}: {reason}")]
    Guard { index: usize, reason: String },
    /// The transition is malformed, or the store failed; nothing was
    /// written.
    #[error("transition failed: {0:#}")]
    Store(#[from] anyhow::Error),
}

impl Rejected {
    /// The rejection for an error the write at `write` returned: a
    /// [`ExpectationFailed`] is a guard rejection at `guard_count + write`,
    /// anything else a store failure.
    #[must_use]
    pub fn from_write_error(error: anyhow::Error, guard_count: usize, write: usize) -> Self {
        match error.downcast::<ExpectationFailed>() {
            Ok(failed) => Self::Guard {
                index: guard_count + write,
                reason: failed.reason,
            },
            Err(error) => Self::Store(error),
        }
    }
}

/// A store that applies a [`Transition`] in one transaction.
#[async_trait]
pub trait DurableStore: Send + Sync {
    /// Check every guard, then apply every write, in one transaction.
    ///
    /// # Errors
    ///
    /// [`Rejected::Guard`] when a guard does not hold, and
    /// [`Rejected::Store`] when the transition is malformed or the store
    /// fails. Either way nothing was written.
    async fn commit(
        &self,
        transition: Transition,
        now: OffsetDateTime,
    ) -> Result<Committed, Rejected>;

    /// The record of the effect `key`, if it has one.
    ///
    /// # Errors
    /// The store could not be read.
    async fn effect(&self, key: &EffectKey) -> anyhow::Result<Option<EffectRecord>>;

    /// Every effect record of `owner`, oldest first.
    ///
    /// # Errors
    /// The store could not be read.
    async fn effects_for_task(&self, owner: &AgentTaskId) -> anyhow::Result<Vec<EffectRecord>>;
}

impl Transition {
    /// Wake `parent`, parked on its children, for a steering exchange.
    #[must_use]
    pub fn enqueue_steering_resume(parent: AgentTaskId, steering: Vec<llm::ContentBlock>) -> Self {
        Self {
            guards: Vec::new(),
            writes: vec![Write::EnqueueSteeringResume { parent, steering }],
        }
    }

    /// Settle a running, leased task and append `events`, one transaction.
    #[must_use]
    pub fn settle(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        outcome: Settle,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![
                Write::SettleTask { task, outcome },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Settle a running, leased task with no events.
    #[must_use]
    pub fn settle_only(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        outcome: Settle,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![Write::SettleTask { task, outcome }],
        }
    }

    /// Fail a task parked on a confirmation and append `events`, one
    /// transaction.
    #[must_use]
    pub fn reject_confirmation(
        task: AgentTaskId,
        error: String,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::AwaitingConfirmation,
                lease: None,
            }],
            writes: vec![
                Write::RejectConfirmation { task, error },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Park a running, leased task on a confirmation and append `events`,
    /// one transaction.
    #[must_use]
    pub fn pause_on_confirmation(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        let mut transition =
            Self::pause_only(task, worker, lease, continuation, prepared_operation);
        transition
            .writes
            .push(Write::AppendEvents { thread, events });
        transition
    }

    /// Park a running, leased task on a confirmation with no events.
    #[must_use]
    pub fn pause_only(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![Write::PauseOnConfirmation {
                task,
                continuation: Box::new(continuation),
                prepared_operation,
            }],
        }
    }

    /// Re-park a steering exchange's parent on its still-live children and
    /// append `events`, one transaction.
    #[must_use]
    pub fn repark_after_steering(
        task: AgentTaskId,
        (worker, lease): (WorkerId, LeaseId),
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: Vec::new(),
            writes: vec![
                Write::ReparkAfterSteering {
                    task,
                    worker,
                    lease,
                    payload: Box::new(payload),
                    reattach,
                },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Append a compaction to `thread`'s message projection and append
    /// `events` to the same thread, one transaction.
    #[must_use]
    pub fn append_compaction(
        thread: ThreadId,
        messages: Vec<llm::Message>,
        source_message_count: usize,
        retained_message_count: usize,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: Vec::new(),
            writes: vec![
                Write::AppendCompaction {
                    thread: thread.clone(),
                    messages,
                    source_message_count,
                    retained_message_count,
                },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Spawn `spawn` under a running, leased parent, one transaction.
    #[must_use]
    pub fn spawn_children(
        parent: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        spawn: ChildSpawn,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: parent.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![Write::SpawnChildren {
                parent,
                spawn: Box::new(spawn),
            }],
        }
    }

    /// The rows this transition locks before its first write: each
    /// guard's task, then each write's rows (see [`Write::locks`]).
    #[must_use]
    pub fn locks(&self) -> Vec<Lock> {
        self.guards
            .iter()
            .filter_map(Guard::task)
            .cloned()
            .map(Lock::Task)
            .chain(self.writes.iter().flat_map(Write::locks))
            .collect()
    }

    /// Whether a guard holds `task` in `status`.
    #[must_use]
    pub fn guards_status(&self, task: &AgentTaskId, status: TaskStatus) -> bool {
        self.guards.iter().any(|guard| match guard {
            Guard::TaskIs {
                task: guarded,
                status: guarded_status,
                ..
            } => guarded == task && *guarded_status == status,
            Guard::EffectFree { .. } => false,
        })
    }

    /// The (worker, lease) a guard holds on `task`, which a
    /// [`Write::SettleTask`] on that task settles under.
    #[must_use]
    pub fn lease_for(&self, task: &AgentTaskId) -> Option<(&WorkerId, &LeaseId)> {
        self.guards.iter().find_map(|guard| match guard {
            Guard::TaskIs {
                task: guarded,
                lease: Some((worker, lease)),
                ..
            } if guarded == task => Some((worker, lease)),
            Guard::TaskIs { .. } | Guard::EffectFree { .. } => None,
        })
    }

    /// Whether any [`Write::AppendEvents`] appends at least one event. A
    /// [`Write::SpawnChildren`] carries its events in its own shape.
    #[must_use]
    pub fn appends_events(&self) -> bool {
        self.writes.iter().any(|write| match write {
            Write::AppendEvents { events, .. } => !events.is_empty(),
            Write::SpawnChildren { .. }
            | Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::Effect(_)
            | Write::AppendCompaction { .. }
            | Write::Document(_)
            | Write::EnqueueSteeringResume { .. }
            | Write::BeginEpoch(_) => false,
        })
    }

    /// The epochs the transition begins.
    pub fn begun_epochs(&self) -> impl Iterator<Item = &EpochRecord> {
        self.writes.iter().filter_map(|write| match write {
            Write::BeginEpoch(record) => Some(record),
            Write::Document(_)
            | Write::AppendEvents { .. }
            | Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::Effect(_)
            | Write::AppendCompaction { .. }
            | Write::SpawnChildren { .. }
            | Write::EnqueueSteeringResume { .. } => None,
        })
    }
    /// Whether any write appends a compaction.
    #[must_use]
    pub fn appends_compactions(&self) -> bool {
        self.writes
            .iter()
            .any(|write| matches!(write, Write::AppendCompaction { .. }))
    }

    /// The threads whose journal a write appends to: the event threads of
    /// non-empty event batches, the compacted threads and the threads of
    /// thread documents. A purged thread refuses all of them.
    pub fn appended_threads(&self) -> impl Iterator<Item = &ThreadId> {
        self.writes.iter().filter_map(|write| match write {
            Write::AppendEvents { thread, events } if !events.is_empty() => Some(thread),
            Write::AppendCompaction { thread, .. }
            | Write::Document(DocumentWrite {
                key:
                    DocumentKey {
                        scope: DocScope::Thread(thread),
                        ..
                    },
                ..
            }) => Some(thread),
            Write::AppendEvents { .. }
            | Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::Effect(_)
            | Write::SpawnChildren { .. }
            | Write::Document(_)
            | Write::EnqueueSteeringResume { .. }
            | Write::BeginEpoch(_) => None,
        })
    }

    /// Refuse an effect write that comes after the write ending its owner:
    /// ending an owner abandons its still-`Started` effects, so the effect
    /// would meet an abandoned record (or, in memory, overwrite one).
    fn validate_effect_order(&self) -> Result<(), Rejected> {
        let mut ended = std::collections::BTreeSet::new();
        for write in &self.writes {
            match write {
                Write::SettleTask { task, .. } | Write::RejectConfirmation { task, .. } => {
                    ended.insert(task);
                }
                Write::Effect(effect) if ended.contains(effect.owner()) => {
                    return Err(Rejected::Store(anyhow::anyhow!(
                        "effect {} must be written before the write that ends its owner {}, \
                     which abandons its started effects",
                        effect.key(),
                        effect.owner()
                    )));
                }
                Write::Effect(_)
                | Write::PauseOnConfirmation { .. }
                | Write::ReparkAfterSteering { .. }
                | Write::AppendCompaction { .. }
                | Write::AppendEvents { .. }
                | Write::SpawnChildren { .. }
                | Write::Document(_)
                | Write::EnqueueSteeringResume { .. }
                | Write::BeginEpoch(_) => {}
            }
        }
        Ok(())
    }

    /// Check the shape every backend relies on before it opens a
    /// transaction: each settle has a leased guard on its task, each pause
    /// a leased guard that its task is running, each confirmation reject a
    /// guard that it is awaiting confirmation, and no guard names a
    /// re-parked task (its lock would come before the re-attached
    /// children's, against the store's child-before-parent order).
    ///
    /// # Errors
    ///
    /// [`Rejected::Store`] naming the first malformed write.
    pub fn validate(&self) -> Result<(), Rejected> {
        for write in &self.writes {
            match write {
                Write::SettleTask { task, .. } => {
                    if self.lease_for(task).is_none() {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "settling task {task} needs a TaskIs guard with its lease"
                        )));
                    }
                }
                Write::PauseOnConfirmation { task, .. } => {
                    if self.lease_for(task).is_none()
                        || !self.guards_status(task, TaskStatus::Running)
                    {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "pausing task {task} needs a TaskIs guard that it is running \
                             under its lease"
                        )));
                    }
                }
                Write::RejectConfirmation { task, .. } => {
                    if !self.guards_status(task, TaskStatus::AwaitingConfirmation) {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "rejecting the confirmation of task {task} needs a TaskIs guard \
                             that it is awaiting confirmation"
                        )));
                    }
                }
                Write::ReparkAfterSteering { task, .. } => {
                    if self.guards.iter().any(|guard| guard.task() == Some(task)) {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "re-parking task {task} takes no guard on it: the backend checks \
                             its ownership after locking its children"
                        )));
                    }
                }
                Write::SpawnChildren { parent, spawn } => {
                    if self.lease_for(parent).is_none()
                        || !self.guards_status(parent, TaskStatus::Running)
                    {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "spawning children under task {parent} needs a TaskIs guard that \
                             it is running under its lease"
                        )));
                    }
                    spawn.validate_shape()?;
                }
                Write::Document(document) => {
                    document.validate().map_err(Rejected::Store)?;
                }
                Write::Effect(EffectWrite::Start(intent)) => {
                    let claimed = self.guards.iter().any(
                        |guard| matches!(guard, Guard::EffectFree { key } if key == &intent.key),
                    );
                    if !claimed || self.lease_for(&intent.owner).is_none() {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "starting effect {} needs an EffectFree guard on its key and a \
                             TaskIs guard with the lease of its owner {}",
                            intent.key,
                            intent.owner
                        )));
                    }
                }
                Write::Effect(EffectWrite::Finish { key, state, .. }) => {
                    if state.is_started() {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "finishing effect {key} needs an end state"
                        )));
                    }
                }
                Write::EnqueueSteeringResume { steering, .. } => {
                    if steering.is_empty() {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "steering wake rejected: steering content must be non-empty"
                        )));
                    }
                }
                Write::AppendEvents { .. }
                | Write::AppendCompaction { .. }
                | Write::BeginEpoch(_) => {}
            }
        }
        self.validate_effect_order()?;
        let mut keys = std::collections::BTreeSet::new();
        for effect in self.effect_writes() {
            if !keys.insert(effect.key()) {
                return Err(Rejected::Store(anyhow::anyhow!(
                    "effect {} is written twice in one transition",
                    effect.key()
                )));
            }
        }
        Ok(())
    }
}

impl Write {
    /// The rows this write locks before the transition's first write, so a
    /// backend takes them in its one sorted order rather than in write
    /// order. A settle, pause or confirmation reject locks its task through
    /// its guard, and the rows a settle reaches after it (the parent) come
    /// later in the same order.
    #[must_use]
    pub fn locks(&self) -> Vec<Lock> {
        match self {
            Self::AppendEvents { thread, events } if !events.is_empty() => vec![Lock::Thread {
                thread: thread.clone(),
                tier: ThreadTier::Own,
                fence: Some(Fence::APPEND),
                create: false,
            }],
            Self::AppendCompaction { thread, .. } => vec![Lock::Thread {
                thread: thread.clone(),
                tier: ThreadTier::Own,
                fence: Some(Fence::APPEND),
                create: false,
            }],
            Self::Effect(effect) => vec![Lock::Task(effect.owner().clone())],
            Self::ReparkAfterSteering { task, reattach, .. } => reattach
                .iter()
                .chain(std::iter::once(task))
                .cloned()
                .map(Lock::Task)
                .collect(),
            Self::SpawnChildren { parent, spawn } => {
                let mut locks = vec![Lock::ThreadOf {
                    task: parent.clone(),
                    fence: Some(Fence::SPAWN),
                }];
                let create = spawn.creates_child_threads();
                locks.extend(
                    spawn
                        .child_threads()
                        .into_iter()
                        .map(|thread| Lock::Thread {
                            thread: thread.clone(),
                            tier: ThreadTier::Spawned,
                            fence: Some(Fence::SPAWN),
                            create,
                        }),
                );
                locks.extend(spawn.reattached().into_iter().cloned().map(Lock::Task));
                locks.extend(
                    spawn
                        .delivered_injection_ids()
                        .iter()
                        .cloned()
                        .map(Lock::Task),
                );
                locks.push(Lock::Task(parent.clone()));
                locks
            }
            Self::Document(document) => match &document.key.scope {
                DocScope::Host => vec![Lock::Document(document.key.clone())],
                DocScope::Thread(thread) => vec![Lock::Thread {
                    thread: thread.clone(),
                    tier: ThreadTier::Own,
                    fence: Some(Fence::APPEND),
                    create: false,
                }],
                DocScope::Task(task) => vec![
                    Lock::ThreadOf {
                        task: task.clone(),
                        fence: Some(Fence::APPEND),
                    },
                    Lock::Task(task.clone()),
                ],
            },
            Self::EnqueueSteeringResume { parent, .. } => vec![Lock::Task(parent.clone())],
            Self::AppendEvents { .. }
            | Self::SettleTask { .. }
            | Self::RejectConfirmation { .. }
            | Self::PauseOnConfirmation { .. }
            | Self::BeginEpoch(_) => Vec::new(),
        }
    }
}

/// A transition's writes in the order every backend applies them.
///
/// Each write keeps its index in the transition. The non-document writes
/// come first and the document writes last, each group in transition order,
/// so a document row is the last row a transition locks (see
/// [`Lock::Document`]).
#[must_use]
pub fn writes_in_apply_order(transition: &Transition) -> Vec<(usize, Write)> {
    let mut writes: Vec<(usize, Write)> = transition.writes.iter().cloned().enumerate().collect();
    writes.sort_by_key(|(_, write)| matches!(write, Write::Document(_)));
    writes
}

/// How a guard names the status it expects, in the words the store's own
/// rejections have always used.
const fn status_words(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Queued => "queued",
        TaskStatus::Pending => "pending",
        TaskStatus::Running => "running",
        TaskStatus::WaitingOnChildren => "waiting on children",
        TaskStatus::AwaitingConfirmation => "awaiting confirmation",
        TaskStatus::AwaitingQuestion => "awaiting a question",
        TaskStatus::Completed => "completed",
        TaskStatus::Failed => "failed",
        TaskStatus::Cancelled => "cancelled",
    }
}

/// The error a store method that wraps [`DurableStore::commit`] returns
/// for `rejected`: a guard rejection reads `{method} rejected: {reason}`,
/// the text that method returned before it became a wrapper.
#[must_use]
pub fn wrapper_error(method: &str, rejected: Rejected) -> anyhow::Error {
    match rejected {
        Rejected::Guard { reason, .. } => anyhow::anyhow!("{method} rejected: {reason}"),
        Rejected::Store(error) => error,
    }
}

impl Guard {
    /// Check a [`Self::TaskIs`] guard against the stored row of its task.
    /// An effect guard passes here; [`Transition::check_effects`] checks
    /// it against its effect row.
    ///
    /// # Errors
    ///
    /// [`Rejected::Guard`] with `index` when the guard does not hold.
    pub fn check(&self, index: usize, stored: Option<&AgentTask>) -> Result<(), Rejected> {
        match self {
            Self::TaskIs {
                task,
                status,
                lease,
            } => {
                let reject = |reason: String| Rejected::Guard { index, reason };
                let Some(stored) = stored else {
                    return Err(reject(format!("task {task} does not exist")));
                };
                if stored.status != *status {
                    return Err(reject(format!(
                        "task {task} is not {} (status {:?})",
                        status_words(*status),
                        stored.status
                    )));
                }
                if let Some((worker, lease)) = lease {
                    if stored.worker_id.as_ref() != Some(worker) {
                        return Err(reject(format!("worker mismatch on task {task}")));
                    }
                    if stored.lease_id.as_ref() != Some(lease) {
                        return Err(reject(format!("lease mismatch on task {task}")));
                    }
                }
                Ok(())
            }
            Self::EffectFree { .. } => Ok(()),
        }
    }

    /// The task a [`Self::TaskIs`] guard reads; an effect guard reads an
    /// effect row instead (see [`Transition::check_effects`]).
    #[must_use]
    pub const fn task(&self) -> Option<&AgentTaskId> {
        match self {
            Self::TaskIs { task, .. } => Some(task),
            Self::EffectFree { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(worker: &str, lease: &str) -> AgentTask {
        let mut task =
            AgentTask::new_root_turn(ThreadId::from_string("t"), OffsetDateTime::UNIX_EPOCH, 3);
        task.status = TaskStatus::Running;
        task.worker_id = Some(WorkerId::from_string(worker));
        task.lease_id = Some(LeaseId::from_string(lease));
        task
    }

    fn guard_on(task: &AgentTask, worker: &str, lease: &str) -> Guard {
        Guard::TaskIs {
            task: task.id.clone(),
            status: TaskStatus::Running,
            lease: Some((WorkerId::from_string(worker), LeaseId::from_string(lease))),
        }
    }

    #[test]
    fn a_guard_holds_only_for_its_status_and_lease() {
        let task = running("w1", "l1");
        assert!(guard_on(&task, "w1", "l1").check(0, Some(&task)).is_ok());
        for guard in [guard_on(&task, "w2", "l1"), guard_on(&task, "w1", "l2")] {
            assert!(matches!(
                guard.check(3, Some(&task)),
                Err(Rejected::Guard { index: 3, .. })
            ));
        }
        let mut done = task.clone();
        done.status = TaskStatus::Completed;
        assert!(guard_on(&task, "w1", "l1").check(0, Some(&done)).is_err());
        assert!(guard_on(&task, "w1", "l1").check(0, None).is_err());
    }

    #[test]
    fn a_settle_needs_a_leased_guard_on_its_task() {
        let task = running("w1", "l1");
        let mut transition = Transition::settle(
            task.id.clone(),
            WorkerId::from_string("w1"),
            LeaseId::from_string("l1"),
            Settle::Fail {
                error: "boom".into(),
            },
            task.thread_id,
            Vec::new(),
        );
        assert!(transition.validate().is_ok());
        transition.guards.clear();
        assert!(matches!(transition.validate(), Err(Rejected::Store(_))));
    }

    #[test]
    fn a_repark_refuses_a_guard_on_its_own_task() {
        let task = running("w1", "l1");
        let mut transition = Transition::repark_after_steering(
            task.id.clone(),
            (WorkerId::from_string("w1"), LeaseId::from_string("l1")),
            SuspensionPayload {
                continuation: ContinuationEnvelope::wrap(agent_sdk_foundation::AgentContinuation {
                    thread_id: task.thread_id.clone(),
                    turn: 1,
                    total_usage: agent_sdk_foundation::TokenUsage::default(),
                    turn_usage: agent_sdk_foundation::TokenUsage::default(),
                    pending_tool_calls: Vec::new(),
                    awaiting_index: 0,
                    completed_results: Vec::new(),
                    state: agent_sdk_foundation::AgentState::new(task.thread_id.clone()),
                    response_id: None,
                    stop_reason: None,
                    response_content: Vec::new(),
                }),
                suspended_messages: Vec::new(),
                child_join_policy: crate::ChildJoinPolicy::default(),
            },
            Vec::new(),
            task.thread_id.clone(),
            Vec::new(),
        );
        assert!(transition.validate().is_ok());
        transition
            .guards
            .push(guard_on(&running("w1", "l1"), "w1", "l1"));
        assert!(
            transition.validate().is_ok(),
            "a guard on another task is allowed"
        );
        transition.guards.push(guard_on(&task, "w1", "l1"));
        assert!(matches!(transition.validate(), Err(Rejected::Store(_))));
    }
}
