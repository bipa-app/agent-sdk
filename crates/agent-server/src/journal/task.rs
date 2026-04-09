//! Agent-task schema, identity, status model, and invariants.
//!
//! This module implements the Phase 2.1 scope of ENG-7915: it defines the
//! durable shape of an [`AgentTask`] row, the [`TaskKind`] and [`TaskStatus`]
//! enums, the identity types ([`AgentTaskId`], [`WorkerId`], [`LeaseId`]),
//! and the structural / state-machine invariants every persisted row must
//! satisfy.
//!
//! Phase 2.2 (ENG-7916) builds directly on top of this module without
//! changing any of its rows. The only new surface it adds here is the
//! [`TaskStatus::blocks_root_admission`] predicate, which the store uses
//! to key its partial-unique "one active root per thread" index — the
//! queue itself lives in [`super::store`].
//!
//! Later phases layer on top:
//!
//! | Phase | What it adds |
//! |-------|--------------|
//! | 2.2 | Same-thread FIFO queue admission (`Queued` promotion) |
//! | 2.3 | Lease acquisition API, CAS guards, expiry sweeps |
//! | 2.4 | Confirmation / resume wiring |
//! | 2.5 | Retry budget + recovery workers |
//! | 2.6 | Tool-runtime child-task orchestration |
//!
//! Phase 2.1 is **schema-only**: no acquisition API, no workers, no tool
//! execution logic. Everything here is pure data, pure invariants, and pure
//! state-transition helpers that the later phases can build on without
//! re-modelling the state machine.

use agent_sdk_core::ThreadId;
use serde::{Deserialize, Serialize};
use std::fmt;
use time::OffsetDateTime;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────
// Identity newtypes
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for an agent task row.
///
/// Task IDs are UUID v4-backed and formatted as `task_<uuid>` to make them
/// obvious in logs. They are newtypes over `String` so they serialize as
/// plain JSON strings (matching [`ThreadId`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AgentTaskId(pub String);

impl AgentTaskId {
    /// Allocate a fresh task ID.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("task_{}", Uuid::new_v4()))
    }

    /// Wrap an existing string as a task ID (used by stores when
    /// rehydrating rows from durable storage).
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentTaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentTaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Durable identity of a worker process.
///
/// A `WorkerId` is stable across lease acquisitions for the same physical
/// worker and will be used as the CAS guard in Phase 2.3's acquisition API.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl WorkerId {
    /// Allocate a fresh worker ID.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("worker_{}", Uuid::new_v4()))
    }

    /// Wrap an existing string as a worker ID.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Per-acquisition lease token.
///
/// A fresh `LeaseId` is minted every time a task transitions into
/// [`TaskStatus::Running`]. Pairing `(worker_id, lease_id)` prevents a stale
/// worker from writing over a row that has since been leased by another
/// worker — the CAS check will land in Phase 2.3.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LeaseId(pub String);

impl LeaseId {
    /// Allocate a fresh lease token.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("lease_{}", Uuid::new_v4()))
    }

    /// Wrap an existing string as a lease token.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for LeaseId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LeaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// TaskKind
// ─────────────────────────────────────────────────────────────────────

/// The kind of work a task represents.
///
/// The three variants are accepted by the schema today even though only
/// [`TaskKind::RootTurn`] and [`TaskKind::ToolRuntime`] are exercised by
/// Phase 2. Including [`TaskKind::Subagent`] now means Phase 3 doesn't
/// need a disruptive schema migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// One turn of the root agent loop on a thread.
    RootTurn,
    /// A single tool-runtime execution spawned under a root turn.
    ToolRuntime,
    /// Reserved for Phase 3 subagent work. Accepted by the schema now so
    /// later phases don't need a disruptive migration.
    Subagent,
}

impl TaskKind {
    /// `true` if this kind represents a top-of-tree root task.
    #[must_use]
    pub const fn is_root(self) -> bool {
        matches!(self, Self::RootTurn)
    }

    /// `true` if tasks of this kind are leaf nodes and must never be the
    /// parent of another task.
    #[must_use]
    pub const fn is_leaf(self) -> bool {
        matches!(self, Self::ToolRuntime)
    }
}

impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootTurn => f.write_str("root_turn"),
            Self::ToolRuntime => f.write_str("tool_runtime"),
            Self::Subagent => f.write_str("subagent"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// TaskStatus
// ─────────────────────────────────────────────────────────────────────

/// Lifecycle state of an [`AgentTask`].
///
/// The state space is designed so that **at most one worker** ever sees a
/// task as leased (`Running`), and every terminal state carries enough
/// information to close the row out without further updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Admitted to the same-thread FIFO queue but not yet runnable. Only
    /// valid for [`TaskKind::RootTurn`] (enforced by [`AgentTask::validate`]).
    Queued,
    /// Runnable. Workers may scan for `Pending` rows and acquire them.
    Pending,
    /// Leased by a worker — the lease fields on [`AgentTask`] are populated
    /// while a row is in this state.
    Running,
    /// Paused until every child task has reached a terminal state. The
    /// `pending_child_count` counter tracks how many children are still
    /// outstanding.
    WaitingOnChildren,
    /// Paused on an out-of-band confirmation (e.g. a user approval for a
    /// tool call that is in the `confirm` tier).
    AwaitingConfirmation,
    /// Terminal: task finished successfully.
    Completed,
    /// Terminal: task finished with an error.
    Failed,
    /// Terminal: task was cancelled by the caller or a parent task.
    Cancelled,
}

impl TaskStatus {
    /// `true` if a worker is allowed to lease a row in this status.
    ///
    /// Only [`TaskStatus::Pending`] is runnable — [`TaskStatus::Queued`] is
    /// admitted but not yet runnable; it becomes runnable when Phase 2.2's
    /// queue promotion fires.
    #[must_use]
    pub const fn is_runnable(self) -> bool {
        matches!(self, Self::Pending)
    }

    /// `true` if this row is paused waiting for out-of-band progress
    /// (child tasks finishing or a confirmation arriving).
    #[must_use]
    pub const fn is_waiting(self) -> bool {
        matches!(self, Self::WaitingOnChildren | Self::AwaitingConfirmation)
    }

    /// `true` if this row has reached a terminal state and will never
    /// transition again.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// `true` if this row is still in the active lifecycle (any non-terminal
    /// status, including `Queued`, `Pending`, `Running`, and both waiting
    /// states).
    ///
    /// Use this for "is this row still alive?" questions (e.g. whether to
    /// include it in a thread's current work). To answer the narrower
    /// question of "does this row occupy the thread's single active-root
    /// slot?" use [`TaskStatus::blocks_root_admission`] instead.
    #[must_use]
    pub const fn is_active(self) -> bool {
        !self.is_terminal()
    }

    /// `true` if a [`TaskKind::RootTurn`] in this status is holding the
    /// thread's **active-root slot** and therefore blocks new root-turn
    /// submissions from admitting directly as [`TaskStatus::Pending`].
    ///
    /// The blocking states are exactly:
    ///
    /// - [`TaskStatus::Pending`] — a root that has been admitted but is
    ///   still waiting for a worker to pick it up.
    /// - [`TaskStatus::Running`] — a root that a worker has leased.
    /// - [`TaskStatus::WaitingOnChildren`] — a root that is paused on its
    ///   spawned tool-runtime children.
    /// - [`TaskStatus::AwaitingConfirmation`] — a root that is paused on
    ///   an out-of-band confirmation.
    ///
    /// [`TaskStatus::Queued`] is deliberately **not** blocking: a queued
    /// root has been durably admitted to the same-thread FIFO queue and is
    /// waiting behind whichever root currently holds the slot. Multiple
    /// queued roots may coexist; they promote in FIFO order as the slot
    /// frees up. Terminal states never block admission because the row is
    /// closed out.
    ///
    /// This is the predicate the Phase 2.2 "one active root per thread"
    /// partial-unique index is keyed on, and the predicate
    /// [`super::store::AgentTaskStore::submit_root_turn`] uses to decide
    /// whether to admit a new root as `Pending` or `Queued`.
    ///
    /// [`TaskKind::RootTurn`]: crate::journal::task::TaskKind::RootTurn
    #[must_use]
    pub const fn blocks_root_admission(self) -> bool {
        matches!(
            self,
            Self::Pending | Self::Running | Self::WaitingOnChildren | Self::AwaitingConfirmation,
        )
    }

    /// Alias for [`TaskStatus::is_runnable`] used by Phase 2.3's CAS guard.
    #[must_use]
    pub const fn can_be_leased(self) -> bool {
        self.is_runnable()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors that can be raised when constructing, validating, or
/// transitioning an [`AgentTask`].
///
/// [`AgentTask::validate`] enforces **single-row** invariants only: identity
/// consistency with respect to `depth` / `parent_id` / `root_id`, lease
/// atomicity, status/child/terminal/error rules, retry budget, and
/// kind × status. Cross-row invariants such as a child's `depth` or
/// `thread_id` matching its parent are enforced at the construction boundary
/// ([`AgentTask::new_child`]) and at the store boundary
/// ([`super::store::AgentTaskStore::insert`]) rather than by `validate()`,
/// because `validate()` takes `&self` and has no access to the parent row.
///
/// Every code path that mutates a task ends in a call to
/// [`AgentTask::validate`], which returns the first violated invariant as
/// one of these variants. Tests pattern-match on these variants for clear
/// failure messages.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TaskSchemaError {
    #[error("root task must not have a parent")]
    RootHasParent,
    #[error("non-root task must have a parent")]
    NonRootMissingParent,
    #[error("root_id must equal id for a root task")]
    RootIdMismatchForRoot,
    #[error("child root_id must match parent root_id")]
    ChildRootIdMismatch,
    #[error("lease fields must all be set together or all be clear")]
    LeaseFieldsInconsistent,
    #[error("status {status:?} must not carry a worker lease")]
    LeasedWhenNotRunning { status: TaskStatus },
    #[error("status Running requires worker_id, lease_id, and lease_expires_at")]
    RunningRequiresLease,
    #[error("last_heartbeat_at requires the row to carry a lease")]
    HeartbeatWithoutLease,
    #[error("terminal status {status:?} must not carry a lease")]
    TerminalWithLease { status: TaskStatus },
    #[error("terminal status {status:?} must set completed_at")]
    TerminalMissingCompletedAt { status: TaskStatus },
    #[error("non-terminal status {status:?} must not set completed_at")]
    NonTerminalWithCompletedAt { status: TaskStatus },
    #[error("Failed status requires last_error to be set")]
    FailedMissingError,
    #[error("non-Failed status {status:?} must not carry last_error")]
    NonFailedWithError { status: TaskStatus },
    #[error("WaitingOnChildren requires pending_child_count > 0")]
    WaitingWithoutChildren,
    #[error("pending_child_count > 0 is only valid in WaitingOnChildren")]
    PendingChildrenInNonWaitingStatus,
    #[error("tool_runtime tasks cannot spawn children")]
    ToolRuntimeCannotSpawnChildren,
    #[error("attempt ({attempt}) exceeds max_attempts ({max})")]
    AttemptExceedsMax { attempt: u32, max: u32 },
    #[error("Queued is only valid for root_turn tasks")]
    QueuedOnlyForRootTurns,
    #[error("invalid transition: {from:?} -> {to:?}")]
    InvalidTransition { from: TaskStatus, to: TaskStatus },
    #[error("heartbeat worker_id does not match the task's current lease")]
    HeartbeatWorkerMismatch,
}

// ─────────────────────────────────────────────────────────────────────
// AgentTask
// ─────────────────────────────────────────────────────────────────────

/// One row in the `agent_tasks` journal.
///
/// An `AgentTask` carries everything a worker needs to reason about a unit
/// of durable work: identity, parent/root linkage, lease bookkeeping, retry
/// budget, and a typed state blob owned by the task kind. Phase 2.1 only
/// defines the shape and invariants; later phases fill in the state blob
/// per kind and wire in the acquisition / retry / children orchestration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTask {
    // ── identity ────────────────────────────────────────────
    pub id: AgentTaskId,
    pub kind: TaskKind,
    pub status: TaskStatus,

    // ── parent / root linkage ───────────────────────────────
    /// `None` for root tasks (depth 0). Required for all non-root tasks.
    pub parent_id: Option<AgentTaskId>,
    /// Points at the root task. For a root task, `root_id == id`.
    pub root_id: AgentTaskId,
    /// 0 for root, parent.depth + 1 for everything else.
    pub depth: u32,

    // ── thread binding ──────────────────────────────────────
    /// Present on root tasks and forwarded to children for locality.
    /// Required on every task kind so rows can be queried per thread.
    pub thread_id: ThreadId,

    // ── lease ───────────────────────────────────────────────
    pub worker_id: Option<WorkerId>,
    pub lease_id: Option<LeaseId>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub lease_expires_at: Option<OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub last_heartbeat_at: Option<OffsetDateTime>,

    // ── typed durable state blob ────────────────────────────
    /// Opaque JSON payload owned by the task kind. Phase 2.1 only reserves
    /// the field; later phases type it per kind.
    #[serde(default)]
    pub state: serde_json::Value,

    // ── retry / failure (stubs, real logic in 2.5) ──────────
    pub attempt: u32,
    pub max_attempts: u32,
    /// Populated on terminal `Failed`.
    pub last_error: Option<String>,

    // ── child blocking counter (used by 2.6) ────────────────
    /// Number of child tasks that still need to reach a terminal state
    /// before this task can leave [`TaskStatus::WaitingOnChildren`].
    pub pending_child_count: u32,

    // ── timestamps ──────────────────────────────────────────
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339::option")]
    pub completed_at: Option<OffsetDateTime>,
}

impl AgentTask {
    /// Default retry budget for root turns before Phase 2.5 tunes it.
    pub const DEFAULT_MAX_ATTEMPTS: u32 = 1;

    /// Allocate a fresh [`TaskKind::RootTurn`] for the given thread.
    ///
    /// The new task starts in [`TaskStatus::Pending`] (runnable), with an
    /// auto-generated `id`/`root_id`, `depth == 0`, no lease, and the
    /// `created_at`/`updated_at` timestamps set to `now`.
    #[must_use]
    pub fn new_root_turn(thread_id: ThreadId, now: OffsetDateTime, max_attempts: u32) -> Self {
        let id = AgentTaskId::new();
        Self {
            root_id: id.clone(),
            id,
            kind: TaskKind::RootTurn,
            status: TaskStatus::Pending,
            parent_id: None,
            depth: 0,
            thread_id,
            worker_id: None,
            lease_id: None,
            lease_expires_at: None,
            last_heartbeat_at: None,
            state: serde_json::Value::Null,
            attempt: 0,
            max_attempts,
            last_error: None,
            pending_child_count: 0,
            created_at: now,
            updated_at: now,
            completed_at: None,
        }
    }

    /// Allocate a fresh child task under `parent`.
    ///
    /// The child inherits `parent.root_id` and `parent.thread_id`, sets
    /// `depth = parent.depth + 1`, and starts in [`TaskStatus::Pending`].
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::ToolRuntimeCannotSpawnChildren`] if the
    /// parent is a [`TaskKind::ToolRuntime`] (tool-runtime tasks are leaves).
    pub fn new_child(
        parent: &Self,
        kind: TaskKind,
        now: OffsetDateTime,
        max_attempts: u32,
    ) -> Result<Self, TaskSchemaError> {
        if parent.kind.is_leaf() {
            return Err(TaskSchemaError::ToolRuntimeCannotSpawnChildren);
        }
        let task = Self {
            id: AgentTaskId::new(),
            kind,
            status: TaskStatus::Pending,
            parent_id: Some(parent.id.clone()),
            root_id: parent.root_id.clone(),
            depth: parent.depth.saturating_add(1),
            thread_id: parent.thread_id.clone(),
            worker_id: None,
            lease_id: None,
            lease_expires_at: None,
            last_heartbeat_at: None,
            state: serde_json::Value::Null,
            attempt: 0,
            max_attempts,
            last_error: None,
            pending_child_count: 0,
            created_at: now,
            updated_at: now,
            completed_at: None,
        };
        task.validate()?;
        Ok(task)
    }

    /// `true` if this task is the root of its tree.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Check every **single-row** invariant on this task.
    ///
    /// The first violation is returned as a [`TaskSchemaError`] so test
    /// failures point at a single clear problem. Call sites that mutate a
    /// task (state transitions, store writes) invoke `validate()` before
    /// committing the change, so no invariant-violating row is ever
    /// reachable through the public API.
    ///
    /// # Cross-row invariants are out of scope
    ///
    /// `validate()` only has access to `self`. Cross-row invariants such as
    /// `child.depth == parent.depth + 1` or `child.thread_id ==
    /// parent.thread_id` are enforced at the construction boundary
    /// ([`AgentTask::new_child`]) and again at the store boundary
    /// ([`super::store::AgentTaskStore::insert`]). Callers that bypass both
    /// boundaries (e.g. by manually mutating public fields or
    /// deserializing an untrusted row) must re-run the store insert to pick
    /// up the cross-row checks.
    ///
    /// # Errors
    /// Returns the first violated invariant.
    pub fn validate(&self) -> Result<(), TaskSchemaError> {
        // Identity: root ⇔ depth 0 ⇔ no parent ⇔ root_id == id
        match (self.depth, self.parent_id.as_ref()) {
            (0, None) => {
                if self.root_id != self.id {
                    return Err(TaskSchemaError::RootIdMismatchForRoot);
                }
            }
            (0, Some(_)) => return Err(TaskSchemaError::RootHasParent),
            (_, None) => return Err(TaskSchemaError::NonRootMissingParent),
            (_, Some(_)) => {
                if self.root_id == self.id {
                    // Non-root cannot share its id with root_id.
                    return Err(TaskSchemaError::ChildRootIdMismatch);
                }
            }
        }

        // Retry budget
        if self.attempt > self.max_attempts {
            return Err(TaskSchemaError::AttemptExceedsMax {
                attempt: self.attempt,
                max: self.max_attempts,
            });
        }

        // Kind ↔ status
        if self.status == TaskStatus::Queued && self.kind != TaskKind::RootTurn {
            return Err(TaskSchemaError::QueuedOnlyForRootTurns);
        }

        // Lease field atomicity
        let lease_set = [
            self.worker_id.is_some(),
            self.lease_id.is_some(),
            self.lease_expires_at.is_some(),
        ];
        let all_set = lease_set.iter().all(|b| *b);
        let none_set = lease_set.iter().all(|b| !*b);
        if !(all_set || none_set) {
            return Err(TaskSchemaError::LeaseFieldsInconsistent);
        }
        if self.last_heartbeat_at.is_some() && !all_set {
            return Err(TaskSchemaError::HeartbeatWithoutLease);
        }

        match self.status {
            TaskStatus::Running => {
                if !all_set {
                    return Err(TaskSchemaError::RunningRequiresLease);
                }
            }
            other => {
                if all_set {
                    if other.is_terminal() {
                        return Err(TaskSchemaError::TerminalWithLease { status: other });
                    }
                    return Err(TaskSchemaError::LeasedWhenNotRunning { status: other });
                }
            }
        }

        // Waiting invariants
        match (self.status, self.pending_child_count) {
            (TaskStatus::WaitingOnChildren, 0) => {
                return Err(TaskSchemaError::WaitingWithoutChildren);
            }
            (s, n) if n > 0 && s != TaskStatus::WaitingOnChildren => {
                return Err(TaskSchemaError::PendingChildrenInNonWaitingStatus);
            }
            _ => {}
        }

        // Terminal invariants
        if self.status.is_terminal() {
            if self.completed_at.is_none() {
                return Err(TaskSchemaError::TerminalMissingCompletedAt {
                    status: self.status,
                });
            }
        } else if self.completed_at.is_some() {
            return Err(TaskSchemaError::NonTerminalWithCompletedAt {
                status: self.status,
            });
        }

        match self.status {
            TaskStatus::Failed => {
                if self.last_error.is_none() {
                    return Err(TaskSchemaError::FailedMissingError);
                }
            }
            other => {
                if self.last_error.is_some() {
                    return Err(TaskSchemaError::NonFailedWithError { status: other });
                }
            }
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────
    // State-transition helpers (pure, consume & return)
    // ─────────────────────────────────────────────────────────

    /// Transition a root turn into the same-thread FIFO [`TaskStatus::Queued`]
    /// state.
    ///
    /// Phase 2.2's [`super::store::AgentTaskStore::submit_root_turn`]
    /// calls this when a new root lands on a thread whose active-root
    /// slot is already held, so the row becomes durably queued behind
    /// the blocking root and the store's queue index can surface it via
    /// [`super::store::AgentTaskStore::list_queued_roots`]. The store's
    /// FIFO index is keyed on the immutable `(created_at, id)` pair, so
    /// queue ordering is determined entirely by `created_at` and is
    /// unaffected by the `now` argument here.
    ///
    /// `now` is written into [`Self::updated_at`]; callers that want the
    /// audit trail to show the original submission time (rather than the
    /// queuing-decision time) should pass `self.created_at`.
    ///
    /// Accepts `Pending` (freshly constructed) or `Queued` (idempotent)
    /// as the source status. Only valid for [`TaskKind::RootTurn`].
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if called from any
    /// other source status and [`TaskSchemaError::QueuedOnlyForRootTurns`]
    /// if the task is not a root turn.
    pub fn admit_as_queued(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if !matches!(self.status, TaskStatus::Pending | TaskStatus::Queued) {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Queued,
            });
        }
        if self.kind != TaskKind::RootTurn {
            return Err(TaskSchemaError::QueuedOnlyForRootTurns);
        }
        self.status = TaskStatus::Queued;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Promote a queued root turn into [`TaskStatus::Pending`].
    ///
    /// Phase 2.2's [`super::store::AgentTaskStore::promote_next_queued_root`]
    /// invokes this on the FIFO head of a thread's queue when the
    /// active-root slot is free. The `created_at` timestamp is
    /// deliberately unchanged so the queue head on a retry of the same
    /// row still sorts back to its original submission moment.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::Queued`].
    pub fn promote_to_pending(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::Queued {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        self.status = TaskStatus::Pending;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Transition from [`TaskStatus::Pending`] to [`TaskStatus::Running`]
    /// and stamp the lease fields.
    ///
    /// The caller supplies the worker ID, a freshly-minted lease token, and
    /// the absolute expiry timestamp. Increments `attempt` (Phase 2.5 uses
    /// this as the retry counter).
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] from non-`Pending`
    /// statuses, or [`TaskSchemaError::AttemptExceedsMax`] if the new
    /// attempt count would exceed `max_attempts`.
    pub fn mark_running(
        mut self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::Pending {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Running,
            });
        }
        let next_attempt = self.attempt.saturating_add(1);
        if next_attempt > self.max_attempts {
            return Err(TaskSchemaError::AttemptExceedsMax {
                attempt: next_attempt,
                max: self.max_attempts,
            });
        }
        self.status = TaskStatus::Running;
        self.worker_id = Some(worker);
        self.lease_id = Some(lease);
        self.lease_expires_at = Some(expires_at);
        self.last_heartbeat_at = Some(now);
        self.attempt = next_attempt;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Release the lease on a running task and send it back to
    /// [`TaskStatus::Pending`].
    ///
    /// Used by Phase 2.3's expiry sweeper when a worker misses its
    /// heartbeat. Note that this does NOT decrement `attempt` — the failed
    /// attempt still counts against the retry budget.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::Running`].
    pub fn release_lease(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::Running {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        self.status = TaskStatus::Pending;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Pause a running task until `child_count` children reach terminal
    /// states. Drops the lease.
    ///
    /// # Errors
    /// - [`TaskSchemaError::InvalidTransition`] if the task is not in
    ///   [`TaskStatus::Running`].
    /// - [`TaskSchemaError::WaitingWithoutChildren`] if `child_count == 0`.
    pub fn wait_on_children(
        mut self,
        child_count: u32,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::Running {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::WaitingOnChildren,
            });
        }
        if child_count == 0 {
            return Err(TaskSchemaError::WaitingWithoutChildren);
        }
        self.status = TaskStatus::WaitingOnChildren;
        self.pending_child_count = child_count;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Record that one child task has reached a terminal state.
    ///
    /// Decrements `pending_child_count`. When the counter hits zero, the
    /// task transitions back to [`TaskStatus::Pending`] so a worker can
    /// pick it up and re-enter the loop.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::WaitingOnChildren`].
    pub fn child_resolved(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::WaitingOnChildren {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        self.pending_child_count = self.pending_child_count.saturating_sub(1);
        if self.pending_child_count == 0 {
            self.status = TaskStatus::Pending;
        }
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Pause a running task on an out-of-band confirmation. Drops the lease.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::Running`].
    pub fn await_confirmation(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::Running {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::AwaitingConfirmation,
            });
        }
        self.status = TaskStatus::AwaitingConfirmation;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Resume from [`TaskStatus::AwaitingConfirmation`] back to
    /// [`TaskStatus::Pending`].
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::AwaitingConfirmation`].
    pub fn resume_from_confirmation(
        mut self,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::AwaitingConfirmation {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        self.status = TaskStatus::Pending;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Mark the task [`TaskStatus::Completed`].
    ///
    /// Accepts either [`TaskStatus::Running`] or
    /// [`TaskStatus::WaitingOnChildren`] (the latter because a waiting task
    /// whose last child resolves *and* requires no further work can finish
    /// directly; call sites generally go through `child_resolved` first,
    /// but the schema accepts both for symmetry with `fail`/`cancel`).
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] from any other source.
    pub fn complete(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if !matches!(
            self.status,
            TaskStatus::Running | TaskStatus::WaitingOnChildren
        ) {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Completed,
            });
        }
        self.status = TaskStatus::Completed;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.pending_child_count = 0;
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Mark the task [`TaskStatus::Failed`] with the given error.
    ///
    /// Accepted from any non-terminal source. Drops the lease and sets
    /// `last_error` / `completed_at`.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is
    /// already in a terminal state.
    pub fn fail(mut self, error: String, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status.is_terminal() {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Failed,
            });
        }
        self.status = TaskStatus::Failed;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.pending_child_count = 0;
        self.last_error = Some(error);
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Mark the task [`TaskStatus::Cancelled`].
    ///
    /// Accepted from any non-terminal source. Drops the lease and sets
    /// `completed_at`.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is
    /// already in a terminal state.
    pub fn cancel(mut self, now: OffsetDateTime) -> Result<Self, TaskSchemaError> {
        if self.status.is_terminal() {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Cancelled,
            });
        }
        self.status = TaskStatus::Cancelled;
        self.worker_id = None;
        self.lease_id = None;
        self.lease_expires_at = None;
        self.last_heartbeat_at = None;
        self.pending_child_count = 0;
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Refresh the heartbeat timestamp on a running lease.
    ///
    /// # Errors
    /// - [`TaskSchemaError::InvalidTransition`] if the task is not in
    ///   [`TaskStatus::Running`].
    /// - [`TaskSchemaError::HeartbeatWorkerMismatch`] if the `worker`
    ///   argument does not match the current lease holder.
    pub fn touch_heartbeat(
        &mut self,
        worker: &WorkerId,
        now: OffsetDateTime,
    ) -> Result<(), TaskSchemaError> {
        if self.status != TaskStatus::Running {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Running,
            });
        }
        match &self.worker_id {
            Some(current) if current == worker => {}
            _ => return Err(TaskSchemaError::HeartbeatWorkerMismatch),
        }
        self.last_heartbeat_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread() -> ThreadId {
        ThreadId::from_string("t-test")
    }

    fn fresh_root() -> AgentTask {
        AgentTask::new_root_turn(thread(), t0(), 3)
    }

    // ── construction / round-trip ─────────────────────────────────

    #[test]
    fn root_task_round_trip_through_json() -> Result<()> {
        let task = fresh_root();
        task.validate().context("fresh root must validate")?;
        let json = serde_json::to_string(&task).context("serializes")?;
        let recovered: AgentTask = serde_json::from_str(&json).context("deserializes")?;
        recovered.validate().context("round-trip must validate")?;
        assert_eq!(task, recovered);
        Ok(())
    }

    #[test]
    fn child_task_inherits_root_id_thread_id_and_depth() -> Result<()> {
        let root = fresh_root();
        let child = AgentTask::new_child(&root, TaskKind::ToolRuntime, t0(), 1).context("child")?;
        assert_eq!(child.root_id, root.id);
        assert_eq!(child.thread_id, root.thread_id);
        assert_eq!(child.depth, root.depth + 1);
        assert!(!child.is_root());
        child.validate().context("child validates")?;
        Ok(())
    }

    #[test]
    fn tool_runtime_cannot_be_parent() -> Result<()> {
        let root = fresh_root();
        let tool = AgentTask::new_child(&root, TaskKind::ToolRuntime, t0(), 1).context("tool")?;
        let err = AgentTask::new_child(&tool, TaskKind::ToolRuntime, t0(), 1).unwrap_err();
        assert_eq!(err, TaskSchemaError::ToolRuntimeCannotSpawnChildren);
        Ok(())
    }

    // ── validate() — identity ─────────────────────────────────────

    #[test]
    fn validate_rejects_root_with_parent() {
        let mut task = fresh_root();
        task.parent_id = Some(AgentTaskId::new());
        assert_eq!(task.validate(), Err(TaskSchemaError::RootHasParent));
    }

    #[test]
    fn validate_rejects_non_root_without_parent() -> Result<()> {
        let root = fresh_root();
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t0(), 1).context("child")?;
        child.parent_id = None;
        assert_eq!(child.validate(), Err(TaskSchemaError::NonRootMissingParent));
        Ok(())
    }

    #[test]
    fn validate_rejects_mismatched_root_id() -> Result<()> {
        let root = fresh_root();
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t0(), 1).context("child")?;
        // Child id == root_id is forbidden on non-roots
        child.root_id = child.id.clone();
        assert_eq!(child.validate(), Err(TaskSchemaError::ChildRootIdMismatch));
        Ok(())
    }

    #[test]
    fn validate_rejects_bad_depth() {
        // depth 0 with a parent_id is the RootHasParent guard, not depth —
        // depth > 0 with parent None is NonRootMissingParent. Both are
        // structural invariants validate() can catch without a parent row.
        let mut task = fresh_root();
        task.depth = 1; // depth > 0 but parent is None
        assert_eq!(task.validate(), Err(TaskSchemaError::NonRootMissingParent));
    }

    // ── validate() — lease invariants ─────────────────────────────

    #[test]
    fn validate_rejects_partial_lease_fields() {
        let mut task = fresh_root();
        task.worker_id = Some(WorkerId::from_string("w1"));
        // lease_id and lease_expires_at still None
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::LeaseFieldsInconsistent)
        );
    }

    #[test]
    fn validate_requires_lease_when_running() {
        let mut task = fresh_root();
        task.status = TaskStatus::Running;
        assert_eq!(task.validate(), Err(TaskSchemaError::RunningRequiresLease));
    }

    #[test]
    fn validate_rejects_lease_when_not_running() {
        let mut task = fresh_root();
        // Pending with a full lease set
        task.worker_id = Some(WorkerId::from_string("w1"));
        task.lease_id = Some(LeaseId::from_string("l1"));
        task.lease_expires_at = Some(t_plus(60));
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::LeasedWhenNotRunning {
                status: TaskStatus::Pending
            })
        );
    }

    #[test]
    fn validate_rejects_terminal_with_lease() -> Result<()> {
        let task = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("mark running")?;
        // Force terminal while keeping lease
        let mut bad = task;
        bad.status = TaskStatus::Completed;
        bad.completed_at = Some(t_plus(2));
        assert_eq!(
            bad.validate(),
            Err(TaskSchemaError::TerminalWithLease {
                status: TaskStatus::Completed
            })
        );
        Ok(())
    }

    // ── validate() — terminal/error invariants ────────────────────

    #[test]
    fn validate_requires_completed_at_on_terminal() {
        let mut task = fresh_root();
        task.status = TaskStatus::Completed;
        // completed_at missing
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::TerminalMissingCompletedAt {
                status: TaskStatus::Completed
            })
        );
    }

    #[test]
    fn validate_requires_error_on_failed() {
        let mut task = fresh_root();
        task.status = TaskStatus::Failed;
        task.completed_at = Some(t_plus(5));
        // last_error missing
        assert_eq!(task.validate(), Err(TaskSchemaError::FailedMissingError));
    }

    #[test]
    fn validate_rejects_error_on_non_failed() {
        let mut task = fresh_root();
        task.last_error = Some("whoops".into());
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::NonFailedWithError {
                status: TaskStatus::Pending
            })
        );
    }

    // ── validate() — children / queue invariants ──────────────────

    #[test]
    fn validate_requires_pending_children_in_waiting() {
        let mut task = fresh_root();
        task.status = TaskStatus::WaitingOnChildren;
        // pending_child_count == 0
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::WaitingWithoutChildren)
        );
    }

    #[test]
    fn validate_rejects_pending_children_outside_waiting() {
        let mut task = fresh_root();
        task.pending_child_count = 2;
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::PendingChildrenInNonWaitingStatus)
        );
    }

    #[test]
    fn validate_rejects_attempt_over_max() {
        let mut task = fresh_root();
        task.attempt = 10;
        task.max_attempts = 3;
        assert_eq!(
            task.validate(),
            Err(TaskSchemaError::AttemptExceedsMax {
                attempt: 10,
                max: 3
            })
        );
    }

    #[test]
    fn validate_rejects_queued_on_tool_runtime() -> Result<()> {
        let root = fresh_root();
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t0(), 1).context("child")?;
        child.status = TaskStatus::Queued;
        assert_eq!(
            child.validate(),
            Err(TaskSchemaError::QueuedOnlyForRootTurns)
        );
        Ok(())
    }

    // ── blocks_root_admission classification ──────────────────────
    //
    // Phase 2.2's "one active root per thread" partial-unique invariant
    // is keyed on `blocks_root_admission`, so drift here would silently
    // corrupt the root queue. Lock every variant in a single table test
    // so any future TaskStatus addition forces an explicit classification.

    #[test]
    fn blocks_root_admission_classification_is_stable() {
        let table = [
            (TaskStatus::Queued, false),
            (TaskStatus::Pending, true),
            (TaskStatus::Running, true),
            (TaskStatus::WaitingOnChildren, true),
            (TaskStatus::AwaitingConfirmation, true),
            (TaskStatus::Completed, false),
            (TaskStatus::Failed, false),
            (TaskStatus::Cancelled, false),
        ];
        for (status, expected) in table {
            assert_eq!(
                status.blocks_root_admission(),
                expected,
                "blocks_root_admission classification drifted for {status:?}"
            );
        }

        // is_active must still cover both Queued (blocks_root_admission =
        // false) and every blocking non-terminal state, otherwise the
        // Phase 2.1 "row is alive" checks fall out of sync with the
        // Phase 2.2 slot semantics.
        for status in [
            TaskStatus::Queued,
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::WaitingOnChildren,
            TaskStatus::AwaitingConfirmation,
        ] {
            assert!(status.is_active(), "{status:?} must still be is_active");
        }
        for status in [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            assert!(!status.is_active(), "{status:?} must not be is_active");
            assert!(
                !status.blocks_root_admission(),
                "terminal {status:?} must not block root admission"
            );
        }
    }

    // ── wire-format lock ──────────────────────────────────────────

    #[test]
    fn all_task_kind_and_status_variants_round_trip_through_snake_case_json() -> Result<()> {
        let kinds = [
            TaskKind::RootTurn,
            TaskKind::ToolRuntime,
            TaskKind::Subagent,
        ];
        let kind_wire = ["\"root_turn\"", "\"tool_runtime\"", "\"subagent\""];
        for (kind, wire) in kinds.iter().zip(kind_wire.iter()) {
            let encoded = serde_json::to_string(kind).context("kind serializes")?;
            assert_eq!(&encoded, wire);
            let decoded: TaskKind = serde_json::from_str(&encoded).context("kind round-trips")?;
            assert_eq!(&decoded, kind);
        }

        let statuses = [
            TaskStatus::Queued,
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::WaitingOnChildren,
            TaskStatus::AwaitingConfirmation,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ];
        let status_wire = [
            "\"queued\"",
            "\"pending\"",
            "\"running\"",
            "\"waiting_on_children\"",
            "\"awaiting_confirmation\"",
            "\"completed\"",
            "\"failed\"",
            "\"cancelled\"",
        ];
        for (status, wire) in statuses.iter().zip(status_wire.iter()) {
            let encoded = serde_json::to_string(status).context("status serializes")?;
            assert_eq!(&encoded, wire);
            let decoded: TaskStatus =
                serde_json::from_str(&encoded).context("status round-trips")?;
            assert_eq!(&decoded, status);
        }
        Ok(())
    }

    // ── state-transition tests ────────────────────────────────────

    #[test]
    fn queued_then_pending_then_running_round_trip() -> Result<()> {
        let root = fresh_root();
        let queued = root.admit_as_queued(t_plus(1)).context("queue")?;
        assert_eq!(queued.status, TaskStatus::Queued);
        let pending = queued.promote_to_pending(t_plus(2)).context("pending")?;
        assert_eq!(pending.status, TaskStatus::Pending);
        let running = pending
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(3),
            )
            .context("running")?;
        assert_eq!(running.status, TaskStatus::Running);
        assert_eq!(running.attempt, 1);
        assert_eq!(running.worker_id, Some(WorkerId::from_string("w1")));
        assert_eq!(running.lease_id, Some(LeaseId::from_string("l1")));
        assert_eq!(running.lease_expires_at, Some(t_plus(60)));
        Ok(())
    }

    #[test]
    fn running_to_waiting_on_children_drops_lease() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running.wait_on_children(2, t_plus(2)).context("wait")?;
        assert_eq!(waiting.status, TaskStatus::WaitingOnChildren);
        assert_eq!(waiting.pending_child_count, 2);
        assert!(waiting.worker_id.is_none());
        assert!(waiting.lease_id.is_none());
        assert!(waiting.lease_expires_at.is_none());
        assert!(waiting.last_heartbeat_at.is_none());
        Ok(())
    }

    #[test]
    fn child_resolved_zero_children_returns_to_pending() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running.wait_on_children(1, t_plus(2)).context("wait")?;
        let resolved = waiting.child_resolved(t_plus(3)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::Pending);
        assert_eq!(resolved.pending_child_count, 0);
        Ok(())
    }

    #[test]
    fn child_resolved_non_zero_stays_waiting() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running.wait_on_children(3, t_plus(2)).context("wait")?;
        let resolved = waiting.child_resolved(t_plus(3)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::WaitingOnChildren);
        assert_eq!(resolved.pending_child_count, 2);
        Ok(())
    }

    #[test]
    fn running_to_awaiting_confirmation_drops_lease() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running.await_confirmation(t_plus(2)).context("await")?;
        assert_eq!(waiting.status, TaskStatus::AwaitingConfirmation);
        assert!(waiting.worker_id.is_none());
        Ok(())
    }

    #[test]
    fn awaiting_confirmation_to_pending() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let awaiting = running.await_confirmation(t_plus(2)).context("await")?;
        let resumed = awaiting
            .resume_from_confirmation(t_plus(3))
            .context("resume")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        Ok(())
    }

    #[test]
    fn complete_from_running_and_from_waiting() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let done = running.clone().complete(t_plus(2)).context("complete")?;
        assert_eq!(done.status, TaskStatus::Completed);
        assert_eq!(done.completed_at, Some(t_plus(2)));
        assert!(done.worker_id.is_none());

        let waiting = running.wait_on_children(1, t_plus(2)).context("wait")?;
        let done2 = waiting
            .complete(t_plus(3))
            .context("complete from waiting")?;
        assert_eq!(done2.status, TaskStatus::Completed);
        assert_eq!(done2.pending_child_count, 0);
        Ok(())
    }

    #[test]
    fn fail_requires_error_and_sets_completed_at() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let failed = running
            .fail("provider timeout".into(), t_plus(2))
            .context("fail")?;
        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(failed.last_error.as_deref(), Some("provider timeout"));
        assert_eq!(failed.completed_at, Some(t_plus(2)));
        assert!(failed.worker_id.is_none());
        Ok(())
    }

    #[test]
    fn cancel_from_every_non_terminal_state() -> Result<()> {
        // Pending
        let task = fresh_root().cancel(t_plus(1)).context("cancel pending")?;
        assert_eq!(task.status, TaskStatus::Cancelled);

        // Queued
        let task = fresh_root()
            .admit_as_queued(t_plus(1))
            .context("queue")?
            .cancel(t_plus(2))
            .context("cancel queued")?;
        assert_eq!(task.status, TaskStatus::Cancelled);

        // Running
        let task = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?
            .cancel(t_plus(2))
            .context("cancel running")?;
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert!(task.worker_id.is_none());

        // WaitingOnChildren
        let task = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?
            .wait_on_children(2, t_plus(2))
            .context("wait")?
            .cancel(t_plus(3))
            .context("cancel waiting")?;
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert_eq!(task.pending_child_count, 0);

        // AwaitingConfirmation
        let task = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?
            .await_confirmation(t_plus(2))
            .context("await")?
            .cancel(t_plus(3))
            .context("cancel awaiting")?;
        assert_eq!(task.status, TaskStatus::Cancelled);

        // Already-terminal is rejected
        let terminal = fresh_root().cancel(t_plus(1)).context("cancel")?;
        let err = terminal.cancel(t_plus(2)).unwrap_err();
        assert_eq!(
            err,
            TaskSchemaError::InvalidTransition {
                from: TaskStatus::Cancelled,
                to: TaskStatus::Cancelled,
            }
        );
        Ok(())
    }

    #[test]
    fn release_lease_returns_to_pending() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        assert_eq!(running.attempt, 1);
        let released = running.release_lease(t_plus(2)).context("released")?;
        assert_eq!(released.status, TaskStatus::Pending);
        assert!(released.worker_id.is_none());
        assert!(released.lease_id.is_none());
        assert!(released.lease_expires_at.is_none());
        assert!(released.last_heartbeat_at.is_none());
        // Attempt counter is NOT rolled back.
        assert_eq!(released.attempt, 1);
        Ok(())
    }

    #[test]
    fn heartbeat_requires_matching_worker_and_running_state() -> Result<()> {
        // Wrong state
        let mut pending = fresh_root();
        let err = pending
            .touch_heartbeat(&WorkerId::from_string("w1"), t_plus(1))
            .unwrap_err();
        assert!(matches!(err, TaskSchemaError::InvalidTransition { .. }));

        // Right state, wrong worker
        let mut running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let err = running
            .touch_heartbeat(&WorkerId::from_string("other"), t_plus(2))
            .unwrap_err();
        assert_eq!(err, TaskSchemaError::HeartbeatWorkerMismatch);

        // Right state, right worker
        running
            .touch_heartbeat(&WorkerId::from_string("w1"), t_plus(3))
            .context("heartbeat ok")?;
        assert_eq!(running.last_heartbeat_at, Some(t_plus(3)));
        Ok(())
    }
}
