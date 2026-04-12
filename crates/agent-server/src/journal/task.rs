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
//! Phase 2.3 (ENG-7917) extends [`AgentTask::touch_heartbeat`] to CAS
//! on **both** the worker id and the lease id (a fresh
//! [`TaskSchemaError::HeartbeatLeaseMismatch`] variant covers the
//! lease half) and to bump `lease_expires_at` alongside the heartbeat
//! timestamp. The acquisition / scan / sweep wiring lives in
//! [`super::store`].
//!
//! Phase 2.4 (ENG-7918) replaces the Phase 2.1 untyped `state` field
//! with the strongly typed [`TaskState`] enum and reshapes the pause
//! transitions to take their typed payload at the call site:
//!
//! - [`AgentTask::wait_on_children`] now takes a
//!   [`agent_sdk_core::ContinuationEnvelope`] alongside the child
//!   count, so a parent that pauses on children can never lose the
//!   continuation it needs to resume.
//! - [`AgentTask::await_confirmation`] now takes the same envelope and
//!   an optional [`agent_sdk_core::ListenExecutionContext`] for
//!   prepared listen/execute operations.
//! - [`AgentTask::child_resolved`] and
//!   [`AgentTask::resume_from_confirmation`] clear the typed payload
//!   when the row leaves a paused state.
//! - [`AgentTask::validate`] enforces the **state ↔ status invariant**
//!   so a row whose status disagrees with its [`TaskState`] cannot
//!   round-trip through the store.
//!
//! Phase 2.6 (ENG-7920) adds two new surfaces on top of the schema
//! without changing any of the existing invariants:
//!
//! - [`ChildSpawnSpec`] — the per-child input struct the store's new
//!   `spawn_tool_children` entry point consumes. Phase 2.6 keeps the
//!   struct deliberately narrow (just `max_attempts`) so the schema
//!   layer does not accrete tool-runtime payloads that belong on a
//!   later phase's typed task-state.
//! - [`AgentTask::recompute_pending_children`] — a pure helper that
//!   authoritatively replaces the parent's `pending_child_count`
//!   from a live-children count and, when the count hits zero, flips
//!   the row back to [`TaskStatus::Pending`] with [`TaskState::None`].
//!   The store calls this from `complete_task` / `fail_task` so the
//!   parent's counter is derived from the journal (`by_parent` +
//!   status) every time instead of drifting through saturating
//!   arithmetic.
//!
//! Later phases layer on top:
//!
//! | Phase | What it adds |
//! |-------|--------------|
//! | 2.2 | Same-thread FIFO queue admission (`Queued` promotion) |
//! | 2.3 | Lease acquisition API, CAS guards, expiry sweeps |
//! | 2.4 | Typed pause-state, journal-guarded pause / resume transitions |
//! | 2.5 | Retry budget + recovery workers |
//! | 2.6 | Tool-runtime child-task orchestration and cancellation tree |
//!
//! Phase 2.1 was schema-only; Phase 2.4 keeps that flavour by limiting
//! its surface to typed data, pure transitions, and `validate()` rules
//! that the store can rely on without re-implementing them. Phase 2.6
//! follows the same rule — it adds one pure transition helper and one
//! narrow input struct, and leaves cross-row orchestration to the
//! store.

use agent_sdk_core::{ContinuationEnvelope, ListenExecutionContext, ThreadId};
use serde::{Deserialize, Serialize};
use std::fmt;
use time::OffsetDateTime;
use uuid::Uuid;

use super::recovery::FailureReason;
use super::task_state::{SubagentInvocationState, TaskState};

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
// ChildSpawnSpec (Phase 2.6)
// ─────────────────────────────────────────────────────────────────────

/// Suspension state captured at a tool-boundary pause point.
///
/// The continuation carries the agent state at suspension time, and the
/// messages carry the conversation that was not yet committed. These
/// two are always paired.
#[derive(Clone, Debug)]
pub struct SuspensionPayload {
    pub continuation: ContinuationEnvelope,
    pub suspended_messages: Vec<agent_sdk_core::llm::Message>,
}

/// Input struct for [`super::store::AgentTaskStore::spawn_tool_children`].
///
/// Phase 2.6 (ENG-7920) adds the store-level entry point that atomically
/// persists a batch of [`TaskKind::ToolRuntime`] children under a running
/// parent and transitions the parent to [`TaskStatus::WaitingOnChildren`].
/// Each child row is built from a `ChildSpawnSpec` via
/// [`AgentTask::new_child`], inheriting the parent's `thread_id`,
/// `root_id`, and `depth + 1` the same way the Phase 2.1 constructor
/// already does.
///
/// The struct is deliberately narrow: the only per-child knob is
/// [`ChildSpawnSpec::max_attempts`]. Tool-specific payload (name,
/// input, tier, listen/execute staging, etc.) lives on the typed
/// task-state a later phase's tool-runtime worker owns — it does not
/// belong on the schema row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildSpawnSpec {
    /// Per-child retry budget. Independent of the parent's budget
    /// because child retries do not count against the parent's
    /// attempt counter.
    pub max_attempts: u32,
}

impl ChildSpawnSpec {
    /// Construct a spec with the given retry budget.
    #[must_use]
    pub const fn new(max_attempts: u32) -> Self {
        Self { max_attempts }
    }
}

impl Default for ChildSpawnSpec {
    /// Default spec uses [`AgentTask::DEFAULT_MAX_ATTEMPTS`] — matching
    /// the retry budget a freshly constructed root turn would start
    /// with in Phase 2.1.
    fn default() -> Self {
        Self::new(AgentTask::DEFAULT_MAX_ATTEMPTS)
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
    #[error("heartbeat lease_id does not match the task's current lease")]
    HeartbeatLeaseMismatch,
    #[error("task state {state_kind} requires status {expected_statuses}, found {actual_status:?}")]
    StateStatusMismatch {
        state_kind: &'static str,
        expected_statuses: &'static str,
        actual_status: TaskStatus,
    },
    #[error(
        "paused status {status:?} requires a matching TaskState payload, found TaskState::None"
    )]
    PausedStatusMissingPayload { status: TaskStatus },
}

// ─────────────────────────────────────────────────────────────────────
// AgentTask
// ─────────────────────────────────────────────────────────────────────

/// Durably captured client input admitted to a root task.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubmittedInputItem {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        data_base64: String,
    },
    Document {
        media_type: String,
        data_base64: String,
    },
}

/// One row in the `agent_tasks` journal.
///
/// An `AgentTask` carries everything a worker needs to reason about a unit
/// of durable work: identity, parent/root linkage, lease bookkeeping, retry
/// budget, and a typed state blob owned by the task kind. Phase 2.1 only
/// defines the shape and invariants; later phases fill in the state blob
/// per kind and wire in the acquisition / retry / children orchestration.
///
/// `AgentTask` deliberately does **not** derive `PartialEq` / `Eq`:
/// the [`TaskState`] payload it carries embeds upstream
/// `agent-sdk-core` types that do not impl `PartialEq`, and the
/// canonical equality contract for a durable row is its serialized
/// JSON form. Tests use [`serde_json::to_value`] for round-trip
/// comparisons.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    // ── submitted input ─────────────────────────────────────
    /// Durably captured root-turn input admitted through an external
    /// transport.
    ///
    /// This is empty for schema-only fixtures created with
    /// [`Self::new_root_turn`]. Transport-owned submissions use
    /// [`Self::new_root_turn_with_input`] so a background worker can
    /// execute the task without relying on an in-memory side channel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub submitted_input: Vec<SubmittedInputItem>,

    // ── lease ───────────────────────────────────────────────
    pub worker_id: Option<WorkerId>,
    pub lease_id: Option<LeaseId>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub lease_expires_at: Option<OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub last_heartbeat_at: Option<OffsetDateTime>,

    // ── typed durable state blob ────────────────────────────
    /// Strongly typed pause-state owned by the task kind.
    ///
    /// Phase 2.1 reserved this field as a `serde_json::Value`; Phase 2.4
    /// promotes it to [`TaskState`] so the row's pause status and the
    /// payload it carries cannot drift apart. Active and terminal rows
    /// hold [`TaskState::None`]; paused rows carry the matching variant
    /// (`WaitingOnChildren { continuation }` or
    /// `AwaitingConfirmation { continuation, prepared_operation }`).
    /// The state ↔ status invariant is enforced by [`Self::validate`].
    #[serde(default)]
    pub state: TaskState,

    // ── retry / failure (stubs, real logic in 2.5) ──────────
    pub attempt: u32,
    pub max_attempts: u32,
    /// Populated on terminal `Failed`.
    pub last_error: Option<String>,

    // ── child blocking counter (used by 2.6) ────────────────
    /// Number of child tasks that still need to reach a terminal state
    /// before this task can leave [`TaskStatus::WaitingOnChildren`].
    pub pending_child_count: u32,

    // ── spawn ordering (used by 5.1) ───────────────────────
    /// Positional index within a batch spawned by
    /// [`super::store::AgentTaskStore::spawn_tool_children`].
    ///
    /// `None` for root tasks and any child not created through a
    /// batch spawn. The bootstrap step in the tool-runtime worker
    /// sorts siblings by this field to map each child to the correct
    /// `PendingToolCallInfo` in the parent's continuation.
    #[serde(default)]
    pub spawn_index: Option<u32>,

    // ── durable tool result (used by 5.4) ──────────────────
    /// Serialized tool result persisted when a tool-runtime child
    /// task reaches [`TaskStatus::Completed`].
    ///
    /// The parent's resume path deserializes this field via
    /// [`aggregate_child_outcomes`](crate::worker::root_turn::aggregate_child_outcomes)
    /// to reconstruct the `(tool_call_id, ToolResult)` pairs the
    /// LLM needs. `None` for root tasks, failed children (whose
    /// error result is derived from [`Self::last_error`]), and any
    /// child that has not yet completed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_payload: Option<serde_json::Value>,

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
            submitted_input: Vec::new(),
            worker_id: None,
            lease_id: None,
            lease_expires_at: None,
            last_heartbeat_at: None,
            state: TaskState::None,
            attempt: 0,
            max_attempts,
            last_error: None,
            pending_child_count: 0,
            spawn_index: None,
            result_payload: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        }
    }

    /// Allocate a fresh [`TaskKind::RootTurn`] with durably captured
    /// external input.
    #[must_use]
    pub fn new_root_turn_with_input(
        thread_id: ThreadId,
        input: Vec<SubmittedInputItem>,
        now: OffsetDateTime,
        max_attempts: u32,
    ) -> Self {
        let mut task = Self::new_root_turn(thread_id, now, max_attempts);
        task.submitted_input = input;
        task
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
            submitted_input: Vec::new(),
            worker_id: None,
            lease_id: None,
            lease_expires_at: None,
            last_heartbeat_at: None,
            state: TaskState::None,
            attempt: 0,
            max_attempts,
            last_error: None,
            pending_child_count: 0,
            spawn_index: None,
            result_payload: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        };
        task.validate()?;
        Ok(task)
    }

    /// Allocate a fresh parent-visible `subagent` invocation task.
    ///
    /// The invocation task is the durable supervisor for a spawned
    /// child thread. It lives under the parent's task tree on the
    /// parent thread, starts immediately in
    /// [`TaskStatus::WaitingOnChildren`], and carries the durable
    /// child-thread linkage on [`TaskState::SubagentInvocation`].
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::ToolRuntimeCannotSpawnChildren`] if the
    /// parent is a [`TaskKind::ToolRuntime`] (tool-runtime tasks are
    /// leaves).
    pub fn new_subagent_invocation(
        parent: &Self,
        invocation: SubagentInvocationState,
        spawn_index: u32,
        now: OffsetDateTime,
        max_attempts: u32,
    ) -> Result<Self, TaskSchemaError> {
        if parent.kind.is_leaf() {
            return Err(TaskSchemaError::ToolRuntimeCannotSpawnChildren);
        }
        let task = Self {
            id: AgentTaskId::new(),
            kind: TaskKind::Subagent,
            status: TaskStatus::WaitingOnChildren,
            parent_id: Some(parent.id.clone()),
            root_id: parent.root_id.clone(),
            depth: parent.depth.saturating_add(1),
            thread_id: parent.thread_id.clone(),
            submitted_input: Vec::new(),
            worker_id: None,
            lease_id: None,
            lease_expires_at: None,
            last_heartbeat_at: None,
            state: TaskState::SubagentInvocation {
                invocation: Box::new(invocation),
            },
            attempt: 0,
            max_attempts,
            last_error: None,
            pending_child_count: 1,
            spawn_index: Some(spawn_index),
            result_payload: None,
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

    /// `true` if this row has already used every attempt in its
    /// retry budget.
    ///
    /// Phase 2.5 (ENG-7919) uses this predicate in
    /// [`super::recovery::classify_recovery`] to decide whether an
    /// acquisition-time or expiry-sweep classification should
    /// requeue the row or fail it closed. A row with
    /// `attempt == max_attempts` has either just finished its final
    /// attempt (if it was leased) or has nothing left to spend on
    /// another acquire — either way, another `mark_running` would
    /// overflow the budget and [`TaskSchemaError::AttemptExceedsMax`]
    /// would bubble back up, which Phase 2.5 wants to replace with
    /// a single deterministic fail-closed transition.
    #[must_use]
    pub const fn is_budget_exhausted(&self) -> bool {
        self.attempt >= self.max_attempts
    }

    /// `true` if this row carries a prepared listen/execute
    /// operation on its [`TaskState`].
    ///
    /// Only [`TaskState::AwaitingConfirmation`] can embed a prepared
    /// operation, and only if the tool that staged the confirmation
    /// used the listen/execute tier. Phase 2.5 uses this predicate
    /// to identify rows that must fail closed on recovery — a
    /// tool-runtime child with a staged external side-effect has no
    /// safe resume contract in the rewrite yet, and blindly
    /// re-running it would risk double-executing the external
    /// operation.
    #[must_use]
    pub const fn has_prepared_operation(&self) -> bool {
        self.state.prepared_operation().is_some()
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

        self.validate_state_status_invariant()?;

        Ok(())
    }

    /// Check the Phase 2.4 state ↔ status invariant.
    ///
    /// Split out of [`Self::validate`] both for readability and to keep
    /// `validate()` under clippy's `too_many_lines` budget. The
    /// invariant is symmetric:
    ///
    /// - A paused status that carries [`TaskState::None`] is rejected
    ///   with [`TaskSchemaError::PausedStatusMissingPayload`] so a
    ///   worker can never look at the row, see the paused status,
    ///   and find no continuation to resume.
    /// - A status that is incompatible with the current payload is
    ///   rejected with [`TaskSchemaError::StateStatusMismatch`] so
    ///   the row can never claim an impossible lifecycle shape. The
    ///   one intentional exception is
    ///   [`TaskState::ReadyToResume`], which is valid both while the
    ///   parent waits in `Pending` and after acquisition in
    ///   `Running`.
    ///
    /// The mismatch returns the variant name as a `&'static str` for
    /// clean error messages without coupling the schema to the wire
    /// format.
    const fn validate_state_status_invariant(&self) -> Result<(), TaskSchemaError> {
        let state_kind: &'static str = match &self.state {
            TaskState::None => "none",
            TaskState::WaitingOnChildren { .. } => "waiting_on_children",
            TaskState::AwaitingConfirmation { .. } => "awaiting_confirmation",
            TaskState::SubagentInvocation { .. } => "subagent_invocation",
            TaskState::ReadyToResume { .. } => "ready_to_resume",
        };
        match &self.state {
            TaskState::None
                if matches!(
                    self.status,
                    TaskStatus::WaitingOnChildren | TaskStatus::AwaitingConfirmation
                ) =>
            {
                Err(TaskSchemaError::PausedStatusMissingPayload {
                    status: self.status,
                })
            }
            _ if !self.state.is_compatible_with_status(self.status) => {
                Err(TaskSchemaError::StateStatusMismatch {
                    state_kind,
                    expected_statuses: self.state.compatible_statuses_label(),
                    actual_status: self.status,
                })
            }
            _ => Ok(()),
        }
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
        // ReadyToResume is a continuation after child completion, not a
        // new attempt — don't consume the retry budget on resume.
        if !matches!(self.state, TaskState::ReadyToResume { .. }) {
            let next_attempt = self.attempt.saturating_add(1);
            if next_attempt > self.max_attempts {
                return Err(TaskSchemaError::AttemptExceedsMax {
                    attempt: next_attempt,
                    max: self.max_attempts,
                });
            }
            self.attempt = next_attempt;
        }
        self.status = TaskStatus::Running;
        self.worker_id = Some(worker);
        self.lease_id = Some(lease);
        self.lease_expires_at = Some(expires_at);
        self.last_heartbeat_at = Some(now);
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
    /// states. Drops the lease and stores the typed
    /// [`TaskState::WaitingOnChildren`] payload that the resume path
    /// will read back on the way out of the pause.
    ///
    /// The `continuation` argument is the same versioned envelope the
    /// SDK already persists for `TurnOutcome::PendingToolCalls`. It is
    /// boxed in [`TaskState::WaitingOnChildren`] so callers do not pay
    /// the size of `AgentContinuation` on every active row.
    ///
    /// # Errors
    /// - [`TaskSchemaError::InvalidTransition`] if the task is not in
    ///   [`TaskStatus::Running`].
    /// - [`TaskSchemaError::WaitingWithoutChildren`] if `child_count == 0`.
    pub fn wait_on_children(
        mut self,
        child_count: u32,
        payload: SuspensionPayload,
        child_ids: Vec<AgentTaskId>,
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
        self.state = TaskState::WaitingOnChildren {
            continuation: Box::new(payload.continuation),
            suspended_messages: payload.suspended_messages,
            child_ids,
        };
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Record that one child task has reached a terminal state.
    ///
    /// Decrements `pending_child_count`. When the counter hits zero, the
    /// task transitions back to [`TaskStatus::Pending`] with
    /// [`TaskState::ReadyToResume`] — the continuation and suspended
    /// messages are preserved so a worker that acquires the parent can
    /// resume the turn from durable state alone. While there are still
    /// outstanding children, the [`TaskState::WaitingOnChildren`]
    /// payload is kept intact.
    ///
    /// Prefer [`Self::recompute_pending_children`] for Phase 2.6 and
    /// later call sites: it derives the counter from the journal's
    /// live-children index instead of trusting a caller-maintained
    /// running total, so a double-complete or dropped-complete cannot
    /// silently corrupt the parent. This helper is kept because the
    /// pure transition is a useful building block for the recompute
    /// path and for tests that want to exercise the counter
    /// semantics in isolation.
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
            // Phase 4.5: move the continuation and suspended messages
            // from WaitingOnChildren into ReadyToResume so the worker
            // that acquires this parent can resume the turn from
            // durable state. The old approach wiped the payload here;
            // now we preserve it through the Pending transition.
            self.state = match self.state {
                TaskState::WaitingOnChildren {
                    continuation,
                    suspended_messages,
                    child_ids,
                } => TaskState::ReadyToResume {
                    continuation,
                    suspended_messages,
                    child_ids,
                },
                // Defensive: if state is somehow not WaitingOnChildren,
                // clear it. This shouldn't happen because the status
                // guard above ensures we are in WaitingOnChildren.
                other => other,
            };
        }
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Authoritatively replace the parent's `pending_child_count` with
    /// `live_children` and, when the count hits zero, flip the row back
    /// to [`TaskStatus::Pending`] with [`TaskState::ReadyToResume`] so
    /// the continuation and suspended messages survive the transition
    /// and a worker can resume the turn from durable state.
    ///
    /// This is the Phase 2.6 replacement for [`Self::child_resolved`]'s
    /// saturating subtraction: the store passes the live child count
    /// derived from the `by_parent` index so a double-complete or a
    /// dropped-complete cannot silently drift the parent's counter.
    ///
    /// `live_children` is the number of children still in a
    /// non-terminal state. When the caller passes zero the parent
    /// resumes; when the caller passes a positive number the parent
    /// stays in [`TaskStatus::WaitingOnChildren`] with its typed
    /// [`TaskState::WaitingOnChildren`] payload intact so the eventual
    /// resume transition still has the continuation it was paused
    /// with.
    ///
    /// # Errors
    /// - [`TaskSchemaError::InvalidTransition`] if the task is not in
    ///   [`TaskStatus::WaitingOnChildren`].
    pub fn recompute_pending_children(
        mut self,
        live_children: u32,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        if self.status != TaskStatus::WaitingOnChildren {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        self.pending_child_count = live_children;
        if live_children == 0 {
            self.status = TaskStatus::Pending;
            // Phase 4.5: preserve the continuation and suspended
            // messages so the resume path can rebuild the turn from
            // durable state. Subagent invocations preserve their
            // durable child-thread linkage for the final
            // materialization worker.
            self.state = match self.state {
                TaskState::WaitingOnChildren {
                    continuation,
                    suspended_messages,
                    child_ids,
                } => TaskState::ReadyToResume {
                    continuation,
                    suspended_messages,
                    child_ids,
                },
                TaskState::SubagentInvocation { invocation } => {
                    TaskState::SubagentInvocation { invocation }
                }
                other => other,
            };
        }
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Pause a running task on an out-of-band confirmation. Drops the
    /// lease and stores the typed [`TaskState::AwaitingConfirmation`]
    /// payload that the resume path will read back on the way out of
    /// the pause.
    ///
    /// `continuation` is the versioned envelope captured at pause time,
    /// matching the value the SDK already persists for
    /// `TurnOutcome::AwaitingConfirmation`. `prepared_operation` is the
    /// optional listen/execute context for tools that staged a
    /// long-running operation before the pause; pass `None` for the
    /// common case where the awaited tool is not in the listen tier.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::Running`].
    pub fn await_confirmation(
        mut self,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
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
        self.state = TaskState::AwaitingConfirmation {
            continuation: Box::new(continuation),
            prepared_operation,
        };
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Resume from [`TaskStatus::AwaitingConfirmation`] back to
    /// [`TaskStatus::Pending`].
    ///
    /// Clears the typed [`TaskState::AwaitingConfirmation`] payload so
    /// the resumed row passes the state ↔ status invariant. The
    /// resume path is responsible for reading the continuation and
    /// the prepared operation **before** calling this helper.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is not in
    /// [`TaskStatus::AwaitingConfirmation`].
    pub fn resume_from_confirmation(
        mut self,
        now: OffsetDateTime,
    ) -> Result<(Self, Option<ListenExecutionContext>), TaskSchemaError> {
        if self.status != TaskStatus::AwaitingConfirmation {
            return Err(TaskSchemaError::InvalidTransition {
                from: self.status,
                to: TaskStatus::Pending,
            });
        }
        let prepared_operation = self.state.prepared_operation().cloned();
        self.status = TaskStatus::Pending;
        self.state = TaskState::None;
        self.updated_at = now;
        self.validate()?;
        Ok((self, prepared_operation))
    }

    /// Mark the task [`TaskStatus::Completed`].
    ///
    /// Accepts either [`TaskStatus::Running`] or
    /// [`TaskStatus::WaitingOnChildren`] (the latter because a waiting task
    /// whose last child resolves *and* requires no further work can finish
    /// directly; call sites generally go through `child_resolved` first,
    /// but the schema accepts both for symmetry with `fail`/`cancel`).
    /// Either way the typed [`TaskState`] payload is cleared, since
    /// terminal rows must hold [`TaskState::None`].
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
        self.state = TaskState::None;
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Mark the task [`TaskStatus::Completed`] with a durable result
    /// payload.
    ///
    /// Identical to [`Self::complete`] but additionally persists
    /// `result_payload` on the row so the parent's resume path can
    /// read tool results from the journal without relying on
    /// in-memory state. Used by Phase 5.4's
    /// [`complete_task_with_result`](super::store::AgentTaskStore::complete_task_with_result).
    ///
    /// # Errors
    /// Same as [`Self::complete`].
    pub fn complete_with_result(
        mut self,
        result_payload: serde_json::Value,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        self.result_payload = Some(result_payload);
        self.complete(now)
    }

    /// Mark the task [`TaskStatus::Failed`] with the given error.
    ///
    /// Accepted from any non-terminal source. Drops the lease, clears
    /// the typed [`TaskState`] payload, and sets `last_error` /
    /// `completed_at`.
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
        self.state = TaskState::None;
        self.last_error = Some(error);
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Fail-closed transition driven by a Phase 2.5 [`FailureReason`].
    ///
    /// This is a thin wrapper around [`Self::fail`] that writes a
    /// canonical `last_error` string built from the reason and the
    /// row's retry budget so every Phase 2.5 call site (acquisition
    /// fail-closed, expiry-sweep fail-closed, recovery matrix)
    /// produces the same wire format. Downstream tools can
    /// pattern-match on [`FailureReason::error_prefix`] without
    /// reparsing free-form text.
    ///
    /// The underlying state transition, lease drop, and
    /// `last_error` / `completed_at` bookkeeping are all inherited
    /// from [`Self::fail`] — this helper exists purely to keep the
    /// error-message shape in one place.
    ///
    /// # Errors
    /// Returns [`TaskSchemaError::InvalidTransition`] if the task is
    /// already in a terminal state.
    pub fn fail_with_reason(
        self,
        reason: FailureReason,
        now: OffsetDateTime,
    ) -> Result<Self, TaskSchemaError> {
        let error = reason.error_message(&self);
        self.fail(error, now)
    }

    /// Mark the task [`TaskStatus::Cancelled`].
    ///
    /// Accepted from any non-terminal source. Drops the lease, clears
    /// the typed [`TaskState`] payload, and sets `completed_at`.
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
        self.state = TaskState::None;
        self.completed_at = Some(now);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Refresh the heartbeat timestamp on a running lease.
    ///
    /// The caller must identify the current lease holder on **both**
    /// dimensions — the `worker` and `lease` arguments are checked against
    /// [`Self::worker_id`] and [`Self::lease_id`] so that a stale worker
    /// that still has the right `WorkerId` but an older `LeaseId` (for
    /// example after its lease expired and was re-acquired) cannot refresh
    /// the row it no longer owns.
    ///
    /// In addition to stamping `last_heartbeat_at = now`, this helper also
    /// bumps `lease_expires_at = expires_at`, because heartbeats are the
    /// mechanism workers use to extend their exclusive window on a row.
    /// Pass `now` for both if the caller wants to keep the existing
    /// expiry; Phase 2.3's store wrapper supplies the bumped value.
    ///
    /// # Errors
    /// - [`TaskSchemaError::InvalidTransition`] if the task is not in
    ///   [`TaskStatus::Running`].
    /// - [`TaskSchemaError::HeartbeatWorkerMismatch`] if the `worker`
    ///   argument does not match the current lease holder.
    /// - [`TaskSchemaError::HeartbeatLeaseMismatch`] if the `lease`
    ///   argument does not match the current `lease_id`.
    pub fn touch_heartbeat(
        &mut self,
        worker: &WorkerId,
        lease: &LeaseId,
        expires_at: OffsetDateTime,
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
        match &self.lease_id {
            Some(current) if current == lease => {}
            _ => return Err(TaskSchemaError::HeartbeatLeaseMismatch),
        }
        self.last_heartbeat_at = Some(now);
        self.lease_expires_at = Some(expires_at);
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
    use agent_sdk_core::{AgentContinuation, AgentState, ContinuationEnvelope, TokenUsage};
    use anyhow::{Context, Result};
    use std::collections::BTreeSet;
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

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    /// Build a sample [`ContinuationEnvelope`] for the Phase 2.4 typed
    /// pause-state tests. The exact contents do not matter — what
    /// matters is that the envelope round-trips through JSON and
    /// surfaces back through [`TaskState::continuation`] on resume.
    fn sample_continuation() -> ContinuationEnvelope {
        let thread = thread();
        ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        })
    }

    /// Build a sample [`ListenExecutionContext`] for the Phase 2.4
    /// typed pause-state tests that exercise the listen/execute path.
    fn sample_prepared_op() -> ListenExecutionContext {
        ListenExecutionContext {
            operation_id: "op-test".into(),
            revision: 3,
            snapshot: serde_json::json!({"preview": "yes"}),
            expires_at: None,
        }
    }

    // ── construction / round-trip ─────────────────────────────────

    #[test]
    fn root_task_round_trip_through_json() -> Result<()> {
        let task = fresh_root();
        task.validate().context("fresh root must validate")?;
        let json = serde_json::to_string(&task).context("serializes")?;
        let recovered: AgentTask = serde_json::from_str(&json).context("deserializes")?;
        recovered.validate().context("round-trip must validate")?;
        // AgentTask intentionally does not impl PartialEq because the
        // typed [`TaskState`] payload embeds upstream SDK types
        // without `PartialEq`. Compare via the canonical JSON wire
        // form instead — that is also the durable contract a future
        // SQL store will use.
        assert_eq!(
            serde_json::to_value(&recovered).context("recovered to value")?,
            serde_json::to_value(&task).context("task to value")?
        );
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
    fn subagent_invocation_inherits_parent_lineage_and_parks_waiting() -> Result<()> {
        let root = fresh_root()
            .mark_running(
                WorkerId::from_string("w-subagent"),
                LeaseId::from_string("l-subagent"),
                t_plus(60),
                t_plus(1),
            )
            .context("mark root running")?;
        let child_thread_id = ThreadId::from_string("t-subagent-child");
        let invocation = AgentTask::new_subagent_invocation(
            &root,
            SubagentInvocationState {
                spec: crate::worker::subagent::EffectiveSubagentSpec {
                    task: "Summarize durable recovery".into(),
                    prompt: String::new(),
                    model: "claude-sonnet-4-5-20250929".into(),
                    max_turns: 4,
                    timeout_ms: 30_000,
                    depth: 1,
                    max_parallel_subagents: 1,
                    nickname: Some("Scout".into()),
                    sandbox: crate::worker::subagent::SubagentSandboxPolicy::read_only(),
                    mcp: crate::worker::subagent::EffectiveSubagentMcpPolicy {
                        allowed_servers: set(&["docs"]),
                    },
                    audit_provenance: Some(agent_sdk_core::audit::AuditProvenance::new(
                        "anthropic",
                        "claude-sonnet-4-5-20250929",
                    )),
                    inherited_policy: crate::worker::subagent::InheritedSubagentPolicy {
                        default_model: "claude-sonnet-4-5-20250929".into(),
                        allowed_models: set(&["claude-sonnet-4-5-20250929"]),
                        default_max_turns: 4,
                        max_turns: 4,
                        default_timeout_ms: 30_000,
                        max_timeout_ms: 30_000,
                        capability_profiles: std::collections::BTreeMap::from([(
                            "research".into(),
                            crate::worker::subagent::SubagentCapabilityProfile {
                                capabilities: set(&["read_file", "rg"]),
                                sandbox: crate::worker::subagent::SubagentSandboxPolicy::read_only(
                                ),
                                allowed_mcp_servers: set(&["docs"]),
                            },
                        )]),
                        allowed_capabilities: set(&["read_file", "rg"]),
                        max_depth: 3,
                        max_parallel_subagents: 1,
                        sandbox: crate::worker::subagent::SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: set(&["docs"]),
                        audit_provider: "anthropic".into(),
                    },
                    capabilities: crate::worker::subagent::EffectiveSubagentCapabilities {
                        profile: "research".into(),
                        allowed: set(&["read_file", "rg"]),
                    },
                },
                child_thread_id: child_thread_id.clone(),
                child_root_task_id: AgentTaskId::from_string("task_child_root"),
            },
            0,
            t_plus(2),
            2,
        )
        .context("new_subagent_invocation")?;
        assert_eq!(invocation.kind, TaskKind::Subagent);
        assert_eq!(invocation.status, TaskStatus::WaitingOnChildren);
        assert_eq!(invocation.parent_id.as_ref(), Some(&root.id));
        assert_eq!(invocation.root_id, root.root_id);
        assert_eq!(invocation.thread_id, root.thread_id);
        assert_eq!(invocation.spawn_index, Some(0));
        assert_eq!(invocation.depth, root.depth + 1);
        assert_eq!(invocation.pending_child_count, 1);
        let linked = invocation
            .state
            .subagent_invocation()
            .context("linkage missing")?;
        assert_eq!(linked.child_thread_id, child_thread_id);
        assert_eq!(linked.child_root_task_id.as_str(), "task_child_root");
        invocation.validate().context("invocation validates")?;
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

    #[test]
    fn validate_ready_to_resume_mismatch_reports_both_valid_statuses() {
        let mut task = fresh_root();
        task.status = TaskStatus::Queued;
        task.state = TaskState::ReadyToResume {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };

        let err = task
            .validate()
            .expect_err("queued ready-to-resume must fail");
        assert_eq!(
            err,
            TaskSchemaError::StateStatusMismatch {
                state_kind: "ready_to_resume",
                expected_statuses: "Pending or Running",
                actual_status: TaskStatus::Queued,
            }
        );
        assert_eq!(
            err.to_string(),
            "task state ready_to_resume requires status Pending or Running, found Queued"
        );
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
        let waiting = running
            .wait_on_children(
                2,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        assert_eq!(waiting.status, TaskStatus::WaitingOnChildren);
        assert_eq!(waiting.pending_child_count, 2);
        assert!(waiting.worker_id.is_none());
        assert!(waiting.lease_id.is_none());
        assert!(waiting.lease_expires_at.is_none());
        assert!(waiting.last_heartbeat_at.is_none());
        // Phase 2.4: typed payload is now stored on the row.
        assert_eq!(
            waiting.state.required_status(),
            Some(TaskStatus::WaitingOnChildren)
        );
        assert!(
            waiting.state.continuation().is_some(),
            "WaitingOnChildren must persist the continuation envelope"
        );
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
        let waiting = running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let resolved = waiting.child_resolved(t_plus(3)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::Pending);
        assert_eq!(resolved.pending_child_count, 0);
        // Phase 4.5: resuming back to Pending preserves the continuation
        // in a ReadyToResume payload so the worker can resume the turn.
        assert!(
            matches!(resolved.state, TaskState::ReadyToResume { .. }),
            "Pending row must hold TaskState::ReadyToResume after children drain"
        );
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
        let waiting = running
            .wait_on_children(
                3,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let resolved = waiting.child_resolved(t_plus(3)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::WaitingOnChildren);
        assert_eq!(resolved.pending_child_count, 2);
        // While children are still outstanding the typed payload stays
        // in place so a future resolve can fire the resume transition.
        assert!(
            resolved.state.continuation().is_some(),
            "continuation must survive partial child resolution"
        );
        Ok(())
    }

    // ── Phase 2.6 pure transition: recompute_pending_children ─────

    #[test]
    fn recompute_pending_children_with_live_count_preserves_state() -> Result<()> {
        // Derive-from-index semantics: passing a non-zero live count
        // overwrites the counter but preserves the typed payload so
        // the eventual zero-children transition can still read it.
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running
            .wait_on_children(
                3,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let recomputed = waiting
            .recompute_pending_children(1, t_plus(3))
            .context("recompute")?;
        assert_eq!(recomputed.status, TaskStatus::WaitingOnChildren);
        assert_eq!(recomputed.pending_child_count, 1);
        assert!(
            recomputed.state.continuation().is_some(),
            "continuation must survive non-terminal recompute"
        );
        Ok(())
    }

    #[test]
    fn recompute_pending_children_with_zero_live_count_resumes_parent() -> Result<()> {
        // Zero live children is the resume trigger — status flips back
        // to Pending and the typed payload is cleared to satisfy the
        // state ↔ status invariant.
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running
            .wait_on_children(
                2,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let recomputed = waiting
            .recompute_pending_children(0, t_plus(3))
            .context("recompute")?;
        assert_eq!(recomputed.status, TaskStatus::Pending);
        assert_eq!(recomputed.pending_child_count, 0);
        assert!(
            matches!(recomputed.state, TaskState::ReadyToResume { .. }),
            "Pending row must hold TaskState::ReadyToResume after recompute resume"
        );
        Ok(())
    }

    #[test]
    fn mark_running_preserves_ready_to_resume_payload() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let resumable = waiting
            .recompute_pending_children(0, t_plus(3))
            .context("recompute")?;

        let reacquired = resumable
            .mark_running(
                WorkerId::from_string("w2"),
                LeaseId::from_string("l2"),
                t_plus(90),
                t_plus(4),
            )
            .context("reacquire")?;

        assert_eq!(reacquired.status, TaskStatus::Running);
        assert!(matches!(reacquired.state, TaskState::ReadyToResume { .. }));
        Ok(())
    }

    #[test]
    fn recompute_pending_children_is_idempotent_after_resume() -> Result<()> {
        // A second recompute on an already-resumed parent is not
        // legal through the pure transition — the row is no longer
        // WaitingOnChildren so the helper refuses. This pins the
        // invariant that "once the parent is Pending, only the normal
        // acquisition path can move it".
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let waiting = running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let recomputed = waiting
            .recompute_pending_children(0, t_plus(3))
            .context("recompute")?;
        assert_eq!(recomputed.status, TaskStatus::Pending);

        let err = recomputed
            .recompute_pending_children(0, t_plus(4))
            .unwrap_err();
        assert_eq!(
            err,
            TaskSchemaError::InvalidTransition {
                from: TaskStatus::Pending,
                to: TaskStatus::Pending,
            }
        );
        Ok(())
    }

    // ── Phase 2.6 ChildSpawnSpec ───────────────────────────────────

    #[test]
    fn child_spawn_spec_default_matches_default_max_attempts() {
        let spec = ChildSpawnSpec::default();
        assert_eq!(spec.max_attempts, AgentTask::DEFAULT_MAX_ATTEMPTS);
    }

    #[test]
    fn child_spawn_spec_new_carries_the_given_budget() {
        let spec = ChildSpawnSpec::new(5);
        assert_eq!(spec.max_attempts, 5);
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
        let waiting = running
            .await_confirmation(sample_continuation(), Some(sample_prepared_op()), t_plus(2))
            .context("await")?;
        assert_eq!(waiting.status, TaskStatus::AwaitingConfirmation);
        assert!(waiting.worker_id.is_none());
        // Typed payload survives — both the continuation and the
        // prepared listen/execute operation must be reachable.
        assert!(waiting.state.continuation().is_some());
        let op = waiting
            .state
            .prepared_operation()
            .expect("prepared op present");
        assert_eq!(op.operation_id, "op-test");
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
        let awaiting = running
            .await_confirmation(sample_continuation(), None, t_plus(2))
            .context("await")?;
        let (resumed, _prepared) = awaiting
            .resume_from_confirmation(t_plus(3))
            .context("resume")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        // Phase 2.4 invariant: resume must clear the typed payload.
        assert!(resumed.state.is_none());
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
        assert!(done.state.is_none());

        let waiting = running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        let done2 = waiting
            .complete(t_plus(3))
            .context("complete from waiting")?;
        assert_eq!(done2.status, TaskStatus::Completed);
        assert_eq!(done2.pending_child_count, 0);
        // Terminal rows must hold TaskState::None even when reached
        // directly from a waiting state.
        assert!(done2.state.is_none());
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
            .wait_on_children(
                2,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?
            .cancel(t_plus(3))
            .context("cancel waiting")?;
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert_eq!(task.pending_child_count, 0);
        assert!(task.state.is_none());

        // AwaitingConfirmation
        let task = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?
            .await_confirmation(sample_continuation(), Some(sample_prepared_op()), t_plus(2))
            .context("await")?
            .cancel(t_plus(3))
            .context("cancel awaiting")?;
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert!(task.state.is_none());

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
    fn heartbeat_requires_matching_worker_lease_and_running_state() -> Result<()> {
        // Wrong state
        let mut pending = fresh_root();
        let err = pending
            .touch_heartbeat(
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
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
            .touch_heartbeat(
                &WorkerId::from_string("other"),
                &LeaseId::from_string("l1"),
                t_plus(120),
                t_plus(2),
            )
            .unwrap_err();
        assert_eq!(err, TaskSchemaError::HeartbeatWorkerMismatch);

        // Right state, right worker, wrong lease
        let err = running
            .touch_heartbeat(
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l-stale"),
                t_plus(120),
                t_plus(2),
            )
            .unwrap_err();
        assert_eq!(err, TaskSchemaError::HeartbeatLeaseMismatch);

        // Right state, right worker, right lease: bumps both heartbeat
        // and expiry timestamps.
        running
            .touch_heartbeat(
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(180),
                t_plus(3),
            )
            .context("heartbeat ok")?;
        assert_eq!(running.last_heartbeat_at, Some(t_plus(3)));
        assert_eq!(running.lease_expires_at, Some(t_plus(180)));
        Ok(())
    }

    // ── Phase 2.5 helpers ─────────────────────────────────────────

    #[test]
    fn is_budget_exhausted_tracks_attempt_vs_max_attempts() -> Result<()> {
        // Fresh row: attempt == 0 < max == 3, not exhausted.
        let fresh = fresh_root();
        assert!(!fresh.is_budget_exhausted());

        // After one attempt: still room.
        let running = fresh
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("mark running")?;
        assert_eq!(running.attempt, 1);
        assert!(!running.is_budget_exhausted());

        // After three attempts: exhausted.
        let mut row = fresh_root();
        for i in 0..3u32 {
            let offset = i64::from(i);
            let running = row
                .mark_running(
                    WorkerId::from_string(format!("w-{i}")),
                    LeaseId::from_string(format!("l-{i}")),
                    t_plus(60),
                    t_plus(1 + offset),
                )
                .context("mark running")?;
            row = running
                .release_lease(t_plus(2 + offset))
                .context("release")?;
        }
        assert_eq!(row.attempt, 3);
        assert_eq!(row.max_attempts, 3);
        assert!(row.is_budget_exhausted());
        Ok(())
    }

    #[test]
    fn has_prepared_operation_is_true_only_for_awaiting_confirmation_with_op() -> Result<()> {
        // Fresh row: none.
        let fresh = fresh_root();
        assert!(!fresh.has_prepared_operation());

        // Waiting on children: none, even with a continuation.
        let running = fresh
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("mark running")?;
        let waiting = running
            .clone()
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation(),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait")?;
        assert!(!waiting.has_prepared_operation());

        // Awaiting confirmation with None prepared_operation: still
        // false — the field is None.
        let awaiting_none = running
            .clone()
            .await_confirmation(sample_continuation(), None, t_plus(2))
            .context("await none")?;
        assert!(!awaiting_none.has_prepared_operation());

        // Awaiting confirmation with Some(prepared): true.
        let awaiting_op = running
            .await_confirmation(sample_continuation(), Some(sample_prepared_op()), t_plus(2))
            .context("await some")?;
        assert!(awaiting_op.has_prepared_operation());
        Ok(())
    }

    #[test]
    fn fail_with_reason_writes_canonical_error_message() -> Result<()> {
        let running = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        let failed = running
            .fail_with_reason(FailureReason::RetryBudgetExhausted, t_plus(2))
            .context("fail with reason")?;
        assert_eq!(failed.status, TaskStatus::Failed);
        let message = failed.last_error.as_deref().expect("last_error set");
        assert!(
            message.starts_with("retry_budget_exhausted:"),
            "unexpected: {message}"
        );
        assert!(message.contains("attempt=1"), "missing attempt: {message}");
        assert!(message.contains("max=3"), "missing max: {message}");
        assert!(failed.completed_at.is_some());
        assert!(failed.worker_id.is_none());
        assert!(failed.lease_id.is_none());
        assert!(failed.state.is_none());
        Ok(())
    }

    #[test]
    fn fail_with_reason_rejects_already_terminal_rows() -> Result<()> {
        let completed = fresh_root()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?
            .complete(t_plus(2))
            .context("complete")?;
        let err = completed
            .fail_with_reason(FailureReason::RetryBudgetExhausted, t_plus(3))
            .unwrap_err();
        assert_eq!(
            err,
            TaskSchemaError::InvalidTransition {
                from: TaskStatus::Completed,
                to: TaskStatus::Failed,
            }
        );
        Ok(())
    }

    // ── Regression: ReadyToResume must not burn retry budget ─────────

    #[test]
    fn mark_running_does_not_increment_attempt_for_ready_to_resume() -> Result<()> {
        // With DEFAULT_MAX_ATTEMPTS=1, the full child-spawn → resume
        // lifecycle must work without exhausting the retry budget.
        // Before the fix, mark_running would increment attempt to 2,
        // exceeding max_attempts=1 and producing AttemptExceedsMax.
        let root = AgentTask::new_root_turn(thread(), t0(), 1);
        assert_eq!(root.attempt, 0);
        assert_eq!(root.max_attempts, 1);

        let running = root.mark_running(
            WorkerId::from_string("w1"),
            LeaseId::from_string("l1"),
            t_plus(60),
            t_plus(1),
        )?;
        assert_eq!(running.attempt, 1, "initial acquire must increment");

        let waiting = running.wait_on_children(
            1,
            SuspensionPayload {
                continuation: sample_continuation(),
                suspended_messages: Vec::new(),
            },
            Vec::new(),
            t_plus(2),
        )?;
        assert_eq!(waiting.attempt, 1);

        let resumable = waiting.recompute_pending_children(0, t_plus(3))?;
        assert_eq!(resumable.status, TaskStatus::Pending);
        assert!(matches!(resumable.state, TaskState::ReadyToResume { .. }));
        assert_eq!(resumable.attempt, 1);

        // This is the critical assertion: re-acquire must succeed
        // with max_attempts=1 because ReadyToResume is a continuation.
        let reacquired = resumable.mark_running(
            WorkerId::from_string("w2"),
            LeaseId::from_string("l2"),
            t_plus(90),
            t_plus(4),
        )?;
        assert_eq!(reacquired.status, TaskStatus::Running);
        assert!(matches!(reacquired.state, TaskState::ReadyToResume { .. }));
        assert_eq!(reacquired.attempt, 1, "resume must not increment attempt");

        Ok(())
    }

    #[test]
    fn mark_running_still_increments_attempt_for_non_resume_tasks() -> Result<()> {
        // Normal retry path: mark_running → release_lease → mark_running
        // must still increment the attempt counter.
        let root = AgentTask::new_root_turn(thread(), t0(), 3);
        let running = root.mark_running(
            WorkerId::from_string("w1"),
            LeaseId::from_string("l1"),
            t_plus(60),
            t_plus(1),
        )?;
        assert_eq!(running.attempt, 1);

        let released = running.release_lease(t_plus(2))?;
        let retried = released.mark_running(
            WorkerId::from_string("w2"),
            LeaseId::from_string("l2"),
            t_plus(120),
            t_plus(3),
        )?;
        assert_eq!(retried.attempt, 2, "normal retry must increment attempt");

        Ok(())
    }
}
