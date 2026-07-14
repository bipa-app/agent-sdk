//! Durable-friendly storage traits for [`AgentTask`] rows.
//!
//! This module defines the [`AgentTaskStore`] trait and a reference
//! [`InMemoryAgentTaskStore`] implementation. The trait surface deliberately
//! mirrors the indexes a production store must expose (see the module-level
//! documentation on [`super`]), so Phase 2.2+ can depend on the semantics
//! without re-designing the API.
//!
//! Every `insert` / `update` calls [`AgentTask::validate`] before committing
//! the row, so no invariant-violating task can ever be reached through the
//! public API — not even in-memory during tests. The in-memory store also
//! enforces the **partial-unique "one blocking root per thread"** invariant,
//! which is the single most important durability guarantee the acquisition
//! API (Phase 2.3) relies on.
//!
//! # Root admission and FIFO queueing (Phase 2.2)
//!
//! Phase 2.2 splits the "one active root per thread" rule along a finer
//! boundary than Phase 2.1's `is_active`: the slot is held only by
//! [`TaskStatus`]es that satisfy [`TaskStatus::blocks_root_admission`], which
//! excludes the `Queued` state. Concretely:
//!
//! - At most one [`TaskKind::RootTurn`] per thread may sit in
//!   `Pending | Running | WaitingOnChildren | AwaitingConfirmation`.
//! - Any number of additional root turns may coexist in [`TaskStatus::Queued`]
//!   while that slot is held. They promote in deterministic FIFO order
//!   (`created_at`, tiebroken by `id`) as the slot frees up.
//! - Tool-runtime children are never gated by this slot — they are leaf
//!   tasks under whatever root currently holds the slot.
//!
//! The Phase 2.2 admission entry points are [`AgentTaskStore::submit_root_turn`]
//! and [`AgentTaskStore::promote_next_queued_root`]. Callers should use
//! `submit_root_turn` instead of `insert` for every externally-submitted
//! root turn so queueing is applied automatically.
//!
//! # Acquisition, lease ownership, and heartbeats (Phase 2.3)
//!
//! Phase 2.3 layers a single guarded acquisition / lease / heartbeat
//! model on top of the schema, shared by every task kind:
//!
//! - [`AgentTaskStore::try_acquire_task`] is a targeted CAS claim by id:
//!   it succeeds when the row is [`TaskStatus::Pending`] and silently
//!   returns `None` for every other status (waiting, queued, running,
//!   terminal). Two workers racing on the same id are serialised by the
//!   store's write lock so exactly one observes `Some(task)`.
//! - [`AgentTaskStore::acquire_next_runnable`] is a scan-and-claim that
//!   walks the global runnable index in `(created_at, id)` ascending
//!   order and atomically leases the head. Root turns and tool-runtime
//!   children share the same scan, which is the "one acquisition and
//!   lease model that prevents double ownership across task kinds" the
//!   issue requires.
//! - [`AgentTaskStore::heartbeat_task`] CAS-checks **both** the
//!   `WorkerId` and the `LeaseId` before bumping `last_heartbeat_at`
//!   and `lease_expires_at`. A stale worker that lost its lease cannot
//!   refresh a row it no longer owns.
//! - [`AgentTaskStore::release_expired_leases`] sweeps the global
//!   lease-expiry index for every row whose `lease_expires_at <= now`,
//!   returns each one to `Pending`, and leaves still-live leases
//!   untouched. The sweep cost is proportional to the number of expired
//!   leases, not the size of the live worker pool.
//!
//! # Pause / resume entry points (Phase 2.4)
//!
//! Phase 2.4 layers two paused statuses on top of the lease model:
//! [`TaskStatus::WaitingOnChildren`] (root parent waiting on
//! tool-runtime children) and [`TaskStatus::AwaitingConfirmation`]
//! (task waiting for a user confirmation). Both are reachable
//! through journal-guarded pause / resume helpers on the trait:
//!
//! - [`AgentTaskStore::pause_on_children`] CAS-checks `(worker,
//!   lease)` ownership, transitions the row to
//!   [`TaskStatus::WaitingOnChildren`] with a typed
//!   [`crate::journal::TaskState::WaitingOnChildren`] payload, and **drops the
//!   lease** atomically. The paused parent is invisible to
//!   `try_acquire_task`, `acquire_next_runnable`, and the
//!   lease-expiry sweep so it cannot pretend to still own a
//!   worker slot.
//! - [`AgentTaskStore::pause_on_confirmation`] does the same for
//!   [`TaskStatus::AwaitingConfirmation`], persisting both the
//!   continuation and the optional listen/execute prepared
//!   operation.
//! - [`AgentTaskStore::resume_from_confirmation`] flips a
//!   confirmation-paused row back to [`TaskStatus::Pending`] and
//!   clears the typed payload after the caller has read its
//!   continuation and prepared operation.
//!
//! All three entry points run their CAS check, the typed-state
//! mutation, and the index rebalance under a single write lock,
//! so the journal is the single source of truth for the
//! paused-state transitions. The schema layer
//! ([`super::task::AgentTask::validate`]) refuses to round-trip
//! any row whose `status` and `state` disagree, so a buggy
//! caller cannot leave a paused row without a continuation or
//! leak a stale continuation onto a runnable row.
//!
//! The Phase 2.4 [`AgentTaskStore`] previously exposed a
//! `resolve_child(parent_id)` helper that decremented the
//! outstanding-child counter by one. Phase 2.6 replaced it with
//! the journal-driven [`AgentTaskStore::complete_task`] /
//! [`AgentTaskStore::fail_task`] pair — see the Phase 2.6
//! section below for why the decrement had to move.
//!
//! # Retry budget, failure handling, and the recovery matrix (Phase 2.5)
//!
//! Phase 2.5 replaces the Phase 2.3 "loud failure on
//! retry exhaustion" placeholder with a deterministic recovery
//! matrix shared by every acquisition and sweep call site:
//!
//! - [`super::recovery::classify_recovery`] is the single pure
//!   entry point that decides what to do with any row the store
//!   looks at. It inspects kind, status, retry budget, and the
//!   prepared-operation bit on [`TaskStatus::AwaitingConfirmation`]
//!   rows and returns a [`super::recovery::RecoveryAction`].
//! - [`AgentTaskStore::try_acquire_task`] and
//!   [`AgentTaskStore::acquire_next_runnable`] consult the matrix
//!   before leasing a row. Retry-exhausted rows now transition
//!   atomically to [`TaskStatus::Failed`] with a canonical
//!   `last_error` built from [`super::recovery::FailureReason`]
//!   (instead of bubbling up an `AttemptExceedsMax` error that
//!   would poison the worker pool).
//! - [`AgentTaskStore::acquire_next_runnable`] keeps scanning the
//!   runnable index on a fail-closed decision — a single exhausted
//!   head can no longer block every younger runnable row behind it.
//! - [`AgentTaskStore::release_expired_leases`] returns a
//!   <code>Vec<[super::recovery::RecoveryRecord]></code> so the caller
//!   can distinguish rows that were requeued for another attempt
//!   from rows that were failed closed because the sweep happened
//!   to find them on their last remaining attempt.
//! - A tool-runtime row whose [`TaskStatus::AwaitingConfirmation`]
//!   payload carries a `Some(prepared_operation)` is **always**
//!   failed closed via
//!   [`super::recovery::FailureReason::UnsafePreparedOperationRecovery`].
//!   Tool-runtime tasks have no safe resume contract for staged
//!   listen/execute operations in Phase 2.5 — a blind recovery
//!   would risk double-executing the external operation, so the
//!   journal refuses to run it.
//! - Duplicate ownership across a requeue + retry is still
//!   guarded by the Phase 2.3 `(worker_id, lease_id)` CAS. Phase
//!   2.5 adds explicit regression coverage in the test module
//!   below to prove an old worker cannot heartbeat or re-lease a
//!   row that has been released or failed-closed by the sweep.
//!
//! # Tool-runtime child tasks and cancellation tree (Phase 2.6)
//!
//! Phase 2.6 takes the child-task story from schema
//! placeholder to a real orchestration contract on top of the
//! Phase 2.1–2.5 foundation. Four new entry points land on the
//! trait and one placeholder is retired:
//!
//! - [`AgentTaskStore::spawn_tool_children`] is the single atomic
//!   entry point for creating tool-runtime child tasks. A successful
//!   call CAS-checks `(worker, lease)` against the parent, builds
//!   one fresh [`TaskKind::ToolRuntime`] row per
//!   [`super::ChildSpawnSpec`] via [`AgentTask::new_child`],
//!   transitions the parent to [`TaskStatus::WaitingOnChildren`]
//!   with a typed continuation and drops the parent's lease, and
//!   commits every child to the primary key and secondary indexes
//!   under the same write lock. No partial batch is ever
//!   observable because the transition, the inserts, and the
//!   index updates all run inside a single
//!   `inner.write().await` scope.
//! - [`AgentTaskStore::spawn_mixed_children`] is the same atomic
//!   entry point for a batch that mixes durable subagent invocations
//!   with tool-runtime children (an LLM coordinator turn routinely
//!   emits both). It CAS-checks the parent once, materializes every
//!   invocation / child-thread root / tool-runtime row, and parks the
//!   parent on `N + M` children under one write lock — the union of
//!   the two pure paths, with no partially-spawned batch on any
//!   failure path.
//! - [`AgentTaskStore::complete_task`] and
//!   [`AgentTaskStore::fail_task`] drive a running child to its
//!   terminal state (`Completed` / `Failed`) and, under the same
//!   write lock, recompute the parent's `pending_child_count`
//!   from the `by_parent` live-children index via
//!   [`AgentTask::recompute_pending_children`]. The parent's
//!   counter is derived from the journal every time, so a
//!   double-complete or a dropped-complete cannot corrupt it —
//!   the recompute always produces the correct live count.
//! - [`AgentTaskStore::cancel_tree`] walks the `by_parent` index
//!   transitively from `root_id` **and** follows
//!   [`crate::journal::TaskState::SubagentInvocation`] linkage so
//!   cancellation cascades across thread boundaries into nested
//!   subagent trees (Phase 7.6). Each step atomically cancels
//!   every non-terminal descendant (and the root itself) via
//!   [`AgentTask::cancel`]. Leases on `Running` rows are dropped
//!   as part of the transition, so a stale worker's next
//!   heartbeat / [`AgentTaskStore::complete_task`] /
//!   [`AgentTaskStore::fail_task`] call fails the
//!   Phase 2.3 `(worker_id, lease_id)` CAS on the way out.
//!   Already-terminal rows are skipped so repeat sweeps are
//!   idempotent.
//! - The Phase 2.4 `resolve_child(parent_id)` entry point is
//!   **removed**. It was a placeholder that subtracted one from
//!   a parent's counter on every call, incompatible with Phase
//!   2.6's journal-driven recompute. Its role is now served by
//!   `complete_task` / `fail_task`, which both take a child id
//!   and a worker / lease CAS.
//!
//! The result is that a completed child batch makes the parent
//! runnable **through journal state alone**: there is no channel,
//! no in-memory queue, and no caller-maintained counter. A
//! crashed worker can restart mid-batch by reading the parent's
//! typed [`super::TaskState::WaitingOnChildren`] payload (which
//! carries the continuation envelope) plus the
//! `list_children(parent_id)` snapshot (which carries the
//! aggregated success / failure outcomes). The counter is
//! re-derived the next time any child reaches a terminal state.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ContinuationEnvelope, ListenExecutionContext, ThreadId, TokenUsage};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::RwLock;

use super::commit::DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS;
use super::committed_event::CommittedEvent;
use super::event_repository::EventRepository;
use super::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};
use super::outbox::{NewOutboxRow, OutboxStatus, OutboxStore};
use super::outbox_message::{OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload};
use super::recovery::{
    FailureReason, RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
use super::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskKind, TaskStatus, WorkerId,
};
use super::task_state::SubagentInvocationState;
use super::thread_store::ThreadStore;
use crate::worker::definition::RuntimePolicy;
use crate::worker::subagent::EffectiveSubagentSpec;

/// Input struct for [`AgentTaskStore::spawn_subagent_invocation`].
///
/// The worker materializes `child_thread_id` through [`super::ThreadStore`]
/// before calling into the task store so durable backends can satisfy the
/// task → thread foreign key while inserting the child root task.
#[derive(Clone, Debug)]
pub struct SubagentInvocationSpawn {
    pub child_thread_id: ThreadId,
    pub spec: EffectiveSubagentSpec,
    pub child_root_input: Vec<SubmittedInputItem>,
    pub spawn_index: u32,
    pub payload: SuspensionPayload,
    /// Opaque caller metadata for the child root turn (see
    /// [`crate::worker::subagent_spawn_selector::SubagentSpawnPlan::child_caller_metadata`]).
    /// `None` = child root has no caller metadata (prior behavior).
    pub child_caller_metadata: Option<serde_json::Value>,
}

/// One tool-runtime child inside a mixed batch
/// ([`AgentTaskStore::spawn_mixed_children`]).
///
/// Carries an explicit `spawn_index` because a mixed batch's tool
/// children do **not** occupy contiguous slots: the subagent entries
/// hold some positions of the parent's `pending_tool_calls`, so the
/// tool children fill the gaps. (In
/// [`AgentTaskStore::spawn_tool_children`] the index is implicit —
/// every slot is a tool child, in input order.)
#[derive(Clone, Debug)]
pub struct ToolChildSpawn {
    /// Index into the shared suspension's `pending_tool_calls` that
    /// this child resolves.
    pub spawn_index: u32,
    /// Per-child retry budget, same contract as
    /// [`AgentTaskStore::spawn_tool_children`].
    pub spec: ChildSpawnSpec,
}

/// Input struct for [`AgentTaskStore::spawn_mixed_children`].
///
/// The subagent entries reuse [`SubagentInvocationSpawn`] so the
/// invocation rows are byte-identical to the ones
/// [`AgentTaskStore::spawn_subagent_batch`] persists; their per-entry
/// `payload` field is ignored in favour of the shared [`Self::payload`]
/// (one parent suspension covers the whole batch).
#[derive(Clone, Debug)]
pub struct MixedChildrenSpawn {
    /// Durable subagent invocations to persist, in input order.
    pub subagents: Vec<SubagentInvocationSpawn>,
    /// Tool-runtime children to persist, in input order.
    pub tool_children: Vec<ToolChildSpawn>,
    /// The one parent suspension shared by every child in the batch.
    pub payload: SuspensionPayload,
    /// Trace parent stamped on the tool children (subagent invocations
    /// inherit the parent's traceparent through
    /// [`AgentTask::new_subagent_invocation`]).
    pub child_otel_traceparent: Option<String>,
}

/// Rows persisted by a successful [`AgentTaskStore::spawn_mixed_children`].
#[derive(Clone, Debug)]
pub struct SpawnedMixedChildren {
    /// Parent post-transition: `WaitingOnChildren` with
    /// `pending_child_count == subagents.len() + tool_children.len()`.
    pub parent: AgentTask,
    /// `(invocation, child_root)` per subagent entry, in input order.
    pub subagents: Vec<(AgentTask, AgentTask)>,
    /// One [`TaskKind::ToolRuntime`] child per tool entry, in input order.
    pub tool_children: Vec<AgentTask>,
}

/// Validate a mixed batch against the shared suspension envelope.
///
/// Every backend runs this before materializing a single row, so the
/// in-memory, `SQLite`, and `PostgreSQL` implementations reject exactly
/// the same inputs with the same message.
///
/// The slot-coverage rule is the load-bearing one: the resume path maps
/// children back to tool calls positionally
/// ([`crate::worker::aggregate_child_outcomes`]) and fails if any
/// `pending_tool_calls` slot has no child. A mixed batch that left a
/// slot uncovered — or double-covered it — would therefore park the
/// parent on a fan-in it can never complete.
///
/// # Errors
/// - `spawn rejected: mixed batch requires at least one subagent`
/// - `spawn rejected: mixed batch requires at least one tool child`
/// - `spawn rejected: spawn_index ... out of bounds ...`
/// - `spawn rejected: duplicate spawn_index ... in mixed batch`
/// - `spawn rejected: mixed batch covers ... of ... pending tool calls`
pub fn validate_mixed_children_spawn(spawn: &MixedChildrenSpawn) -> Result<()> {
    let subagent_slots: Vec<u32> = spawn
        .subagents
        .iter()
        .map(|entry| entry.spawn_index)
        .collect();
    let tool_slots: Vec<u32> = spawn
        .tool_children
        .iter()
        .map(|child| child.spawn_index)
        .collect();
    validate_mixed_slot_coverage(
        &subagent_slots,
        &tool_slots,
        spawn.payload.continuation.payload.pending_tool_calls.len(),
    )
}

/// Validate the slot partition of a mixed batch on its own, before any
/// row — task **or** child-thread projection — is materialized.
///
/// Split out from [`validate_mixed_children_spawn`] so a caller that
/// creates child threads ahead of the store call can reject a bad batch
/// before it writes anything: the store transaction rolls back its own
/// rows, but a thread projection created outside it would survive as an
/// orphan.
///
/// # Errors
/// - `spawn rejected: mixed batch requires at least one subagent`
/// - `spawn rejected: mixed batch requires at least one tool child`
/// - `spawn rejected: spawn_index ... out of bounds ...`
/// - `spawn rejected: duplicate spawn_index ... in mixed batch`
/// - `spawn rejected: mixed batch covers ... of ... pending tool calls`
pub fn validate_mixed_slot_coverage(
    subagent_slots: &[u32],
    tool_slots: &[u32],
    pending_tool_calls: usize,
) -> Result<()> {
    if subagent_slots.is_empty() {
        return Err(anyhow!(
            "spawn rejected: mixed batch requires at least one subagent"
        ));
    }
    if tool_slots.is_empty() {
        return Err(anyhow!(
            "spawn rejected: mixed batch requires at least one tool child"
        ));
    }

    let pending = pending_tool_calls;
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for index in subagent_slots.iter().chain(tool_slots).copied() {
        let slot = usize::try_from(index).context("spawn rejected: spawn_index exceeds usize")?;
        if slot >= pending {
            return Err(anyhow!(
                "spawn rejected: spawn_index {slot} out of bounds for {pending} pending tool calls"
            ));
        }
        if !seen.insert(index) {
            return Err(anyhow!(
                "spawn rejected: duplicate spawn_index {index} in mixed batch"
            ));
        }
    }

    if seen.len() != pending {
        let covered = seen.len();
        return Err(anyhow!(
            "spawn rejected: mixed batch covers {covered} of {pending} pending tool calls"
        ));
    }
    Ok(())
}

/// Order a mixed batch's children by the `pending_tool_calls` slot each
/// one resolves.
///
/// The parent's persisted child-id vector is **positional**:
/// [`AgentTaskStore::repark_after_steering`] re-derives every child's
/// `spawn_index` from that vector's index. Persisting a mixed batch's
/// two halves grouped by kind would therefore rebind an interleaved
/// batch's children to the wrong tool calls the first time a steering
/// wake reverts — a child could execute or report against a tool call
/// that was never its own. Sorting by slot keeps the vector position
/// and the `spawn_index` identical, which is the identity mapping the
/// re-park assumes.
///
/// # Errors
/// Returns an error if any row is missing its `spawn_index`.
pub fn mixed_child_ids_in_slot_order(
    subagents: &[(AgentTask, AgentTask)],
    tool_children: &[AgentTask],
) -> Result<Vec<AgentTaskId>> {
    let mut by_slot: Vec<(u32, AgentTaskId)> =
        Vec::with_capacity(subagents.len() + tool_children.len());
    for (invocation, _child_root) in subagents {
        let slot = invocation
            .spawn_index
            .context("spawn rejected: subagent invocation missing spawn_index")?;
        by_slot.push((slot, invocation.id.clone()));
    }
    for child in tool_children {
        let slot = child
            .spawn_index
            .context("spawn rejected: tool child missing spawn_index")?;
        by_slot.push((slot, child.id.clone()));
    }
    by_slot.sort_unstable_by_key(|(slot, _)| *slot);
    Ok(by_slot.into_iter().map(|(_, id)| id).collect())
}

/// Build one [`TaskKind::ToolRuntime`] child row for a mixed batch.
///
/// Same shape [`AgentTaskStore::spawn_tool_children`] produces, except
/// the `spawn_index` comes from the caller rather than the input
/// position — see [`ToolChildSpawn`]. Shared by every backend so the
/// three implementations cannot drift on child construction.
///
/// # Errors
/// Schema errors from [`AgentTask::new_child`].
pub fn new_mixed_tool_child(
    parent: &AgentTask,
    tool: &ToolChildSpawn,
    child_otel_traceparent: Option<&str>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    let mut child =
        AgentTask::new_child(parent, TaskKind::ToolRuntime, now, tool.spec.max_attempts)
            .context("spawn rejected: new_child failed")?;
    child.spawn_index = Some(tool.spawn_index);
    child.otel_traceparent = child_otel_traceparent.map(ToOwned::to_owned);
    Ok(child)
}

/// The idempotency key a [`AgentTaskStore::submit_root_turn_idempotent`]
/// call claims atomically inside the admission transaction.
#[derive(Clone, Debug)]
pub struct SubmitRootIdempotency {
    /// Caller-supplied idempotency key.
    pub request_id: String,
    /// Fingerprint of the submission payload (everything but the key).
    pub fingerprint: Vec<u8>,
    /// Operation-specific durable references the gRPC layer needs to
    /// reconstruct the original response on a replay. The journal layer
    /// treats this as opaque.
    pub result_json: serde_json::Value,
}

/// Parameters for [`AgentTaskStore::submit_root_turn_idempotent`].
///
/// Phase 10 · E bundles the durable idempotency claim and the
/// queued-root depth cap into the same admission transaction as the task
/// insert so dedup, back-pressure, and admission are atomic (no TOCTOU).
#[derive(Clone, Debug)]
pub struct SubmitRootTurnParams {
    /// The freshly-constructed root turn to admit (same shape contract
    /// as [`AgentTaskStore::submit_root_turn`]).
    pub task: AgentTask,
    /// Optional idempotency claim. `None` skips dedup (used by internal
    /// callers that already guarantee uniqueness).
    pub idempotency: Option<SubmitRootIdempotency>,
    /// Maximum number of *queued* (not the active/blocking) roots a
    /// thread may hold before a new submission is rejected with
    /// [`SubmitRootTurnError::QueueDepthExceeded`]. `None` disables the
    /// cap.
    pub max_queued_depth: Option<u32>,
}

/// Successful outcome of [`AgentTaskStore::submit_root_turn_idempotent`].
#[derive(Clone, Debug)]
pub struct SubmitRootTurnOutcome {
    /// The admitted (or, on replay, originally-admitted) task.
    pub task: AgentTask,
    /// `true` if this call replayed a previously-recorded submission
    /// rather than admitting a fresh one.
    pub replayed: bool,
    /// Stored idempotency references, present only on a replay so the
    /// caller can reconstruct the original response.
    pub replayed_result: Option<serde_json::Value>,
    /// Number of queued roots on the thread *after* this admission
    /// (excludes the active/blocking root). Surfaced to the caller so
    /// at-least-once clients can observe back-pressure.
    pub queued_depth: u32,
}

/// Outcome of [`AgentTaskStore::requeue_owned_task`].
#[derive(Clone, Debug)]
pub enum RequeueOutcome {
    /// The row was returned to [`TaskStatus::Pending`] with the lease
    /// cleared; the dispatcher will re-acquire it and re-run the turn
    /// from the fresh committed head. Boxed so the enum's unit
    /// variants do not pay the full row's size.
    Requeued(Box<AgentTask>),
    /// The row has no retry budget left (`attempt == max_attempts`);
    /// NO transition was applied and the row is still `Running` under
    /// the caller's lease. The caller owns routing it through its full
    /// terminal-failure envelope (attempt close, `Error` event,
    /// successor promotion) — the store must not fail it silently.
    BudgetExhausted,
    /// The row is no longer a `Running` row owned by the caller
    /// (cancelled, swept, or re-acquired); NO transition was applied.
    /// Whichever path took the row over owns its terminal state — the
    /// caller must skip cleanly, exactly like the fail path's
    /// still-owned guard.
    NotOwned,
}

/// Typed failure modes for
/// [`AgentTaskStore::submit_root_turn_idempotent`] that the transport
/// layer maps to specific gRPC status codes.
#[derive(Debug)]
pub enum SubmitRootTurnError {
    /// The `request_id` was already used for a different operation kind
    /// or a different payload fingerprint. Maps to `ALREADY_EXISTS`.
    IdempotencyConflict,
    /// The thread already holds the maximum number of queued roots.
    /// Maps to `RESOURCE_EXHAUSTED`. Carries the configured cap and the
    /// current depth so the caller can report both.
    QueueDepthExceeded {
        /// The configured maximum queued-root depth.
        cap: u32,
        /// The current queued-root depth at rejection time.
        current_depth: u32,
    },
    /// Any other store-level failure (validation, IO, …).
    Other(anyhow::Error),
}

impl std::fmt::Display for SubmitRootTurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdempotencyConflict => {
                write!(f, "request_id already used for a different request")
            }
            Self::QueueDepthExceeded { cap, current_depth } => write!(
                f,
                "thread queued-root depth {current_depth} reached the cap {cap}"
            ),
            Self::Other(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for SubmitRootTurnError {}

impl From<anyhow::Error> for SubmitRootTurnError {
    fn from(error: anyhow::Error) -> Self {
        Self::Other(error)
    }
}

/// Maximum rows one [`AgentTaskStore::release_expired_leases`] call
/// reclaims.
///
/// Uniform across every backend so callers can drain a backlog with a
/// simple "loop until a call returns fewer than this" contract: the
/// SQL backends bound their sweep transaction with `LIMIT` (a mass
/// worker outage must not lock every expired task at once) and the
/// in-memory reference store applies the same cap so the drain
/// contract is testable without a database.
pub const LEASE_RELEASE_BATCH: usize = 256;

/// Shared shape gate for root-turn admission.
///
/// Both [`AgentTaskStore::submit_root_turn`] and
/// [`AgentTaskStore::submit_root_turn_idempotent`] require a
/// freshly-constructed [`TaskKind::RootTurn`] in [`TaskStatus::Pending`]
/// with `attempt == 0`.
///
/// # Errors
/// Returns an error if the task is not a freshly-constructed root turn
/// or fails schema validation.
pub fn validate_submit_root_shape(task: &AgentTask) -> Result<()> {
    if task.kind != TaskKind::RootTurn {
        let kind = task.kind;
        return Err(anyhow!(
            "submit_root_turn rejected: expected root_turn, got {kind:?}"
        ));
    }
    if task.status != TaskStatus::Pending {
        let status = task.status;
        return Err(anyhow!(
            "submit_root_turn rejected: new root must start in Pending (got {status:?})"
        ));
    }
    if task.attempt != 0 {
        let attempt = task.attempt;
        return Err(anyhow!(
            "submit_root_turn rejected: new root must have attempt == 0 (got {attempt})"
        ));
    }
    if !task.is_root() {
        return Err(anyhow!("submit_root_turn rejected: task must be a root"));
    }
    task.validate()
        .context("submit_root_turn rejected: task failed schema validation")?;
    Ok(())
}

/// Extract the admitted task id stored in a `SubmitWork` idempotency
/// record's `result_json` (encoded by the gRPC layer as
/// `{"task_id": "..."}`).
///
/// # Errors
/// Returns an error if the record's `result_json` does not carry a
/// `task_id` string.
pub fn submit_record_task_id(record: &IdempotencyRecord) -> Result<AgentTaskId> {
    let task_id = record
        .result_json
        .get("task_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("submit idempotency record missing task_id reference"))?;
    Ok(AgentTaskId::from_string(task_id))
}

/// Outcome of [`AgentTaskStore::cancel_tree`].
#[derive(Clone, Debug, Default)]
pub struct CancelTreeOutcome {
    /// Task ids actually transitioned to [`TaskStatus::Cancelled`], in
    /// BFS (root-first) order. Rows that were already terminal are not
    /// included.
    pub transitioned: Vec<AgentTaskId>,
    /// Terminal [`AgentEvent::Cancelled`] markers committed by this
    /// call — exactly one per *blocking* root turn transitioned (the
    /// pre-cancel occupant of its thread's active-root slot, per
    /// [`TaskStatus::blocks_root_admission`]), each on that root's own
    /// thread. Queued-behind-active cancels and non-root rows emit
    /// nothing. Callers should forward these to the in-process
    /// [`EventNotifier`](super::event_notifier::EventNotifier) for
    /// same-process live-tail latency; cross-process followers are
    /// covered by the `thread_events_available` outbox advisory
    /// committed alongside each marker.
    pub markers: Vec<CommittedEvent>,
}

/// Store handles the in-memory [`AgentTaskStore`] uses to commit the
/// terminal `Cancelled` marker (and its outbox advisory) as part of
/// [`AgentTaskStore::cancel_tree`].
///
/// The durable SQL backends commit the marker inside `cancel_tree`'s
/// own transaction — task transitions, marker event, and outbox row
/// are all-or-nothing. The in-memory backend keeps tasks, events, and
/// outbox in separate stores, so it takes this sink at construction
/// ([`InMemoryAgentTaskStore::with_cancellation_markers`]) and commits
/// the marker in the same `cancel_tree` call, while still holding the
/// task-store write lock — a promoted successor cannot be acquired
/// until that lock drops, so the marker's sequence always precedes
/// the successor's `Start`. Exactly-once holds because the marker is
/// derived from the rows this call transitions — an idempotent retry
/// sees a terminal tree and emits nothing.
///
/// **A `Cancelled` row always has its marker in the journal.** No
/// cross-store transaction exists here, so the ordering supplies the
/// invariant instead: the whole marker set is committed BEFORE any row
/// flips, and a sink failure aborts the call with nothing transitioned.
/// A cancelled task therefore can never be left without the marker that
/// closes its followers, and the tree stays fully cancellable — a retry
/// re-reads it and completes.
///
/// # Orphan markers (this backend only)
///
/// The converse holds only once the cancel actually settles. The markers
/// of a cascade are appended one per thread, so a sink failure PART-WAY
/// through that set leaves the markers it already journaled behind while
/// their roots stay live (nothing was flipped). Marker closes are
/// ungated, so a follower that reaches such an orphan marker closes
/// `TurnCancelled` on a task that then keeps running and completes — its
/// answer is still journaled, and the follower reaches it by
/// reconnecting.
///
/// A retry does NOT compound that: the marker append is idempotent per
/// candidate (keyed on the marker's emitter task id), so the retry reuses
/// the markers already on the thread, writes only the missing ones, and
/// flips the tree. Exactly one `Cancelled` marker per effective cancel
/// survives, and the orphan is retroactively made honest — its root is
/// now genuinely cancelled. The residual is therefore narrow and
/// terminal: a marker on a live root, on a cancel the caller ABANDONS
/// after a sink failure.
///
/// Pairing each marker with its own row flip would avoid the orphan
/// window entirely, but the flip clears the typed state that carries
/// `SubagentInvocation` linkage: a half-flipped cascade is no longer
/// discoverable by `collect_subtree`, so a retry could never reach the
/// child-thread root it stranded, and that root would stay live forever.
/// Keeping the flips all-or-nothing preserves the reachability a retry
/// needs to heal anything at all. The durable backends have neither
/// problem — marker set and cancellation share one transaction, so a
/// partial set cannot exist.
///
/// A bare `InMemoryAgentTaskStore::new()` (no sink) emits no markers;
/// composed deployments (the service host's in-memory registry, test
/// fixtures that assert on markers) must wire the sink.
#[derive(Clone)]
pub struct CancellationMarkerSink {
    /// Committed-event journal the marker is appended to.
    pub event_repo: Arc<dyn EventRepository>,
    /// Outbox the `thread_events_available` advisory row is written to.
    pub outbox_store: Arc<dyn OutboxStore>,
    /// Thread projection consulted for the marker's `turn` /
    /// fallback `usage` payload.
    pub thread_store: Arc<dyn ThreadStore>,
}

/// Marker candidate captured while cancelling a blocking root, before
/// [`AgentTask::cancel`] clears its typed state.
struct CancelMarkerCandidate {
    thread_id: ThreadId,
    /// The cancelled root itself — the marker is attributed to the task
    /// whose work the cancel ended, not to the successor promoted in the
    /// same transaction.
    task_id: AgentTaskId,
    /// Cumulative usage from the parked turn's durable continuation,
    /// when the root carried one. Preferred over the thread aggregate
    /// because it includes the suspended turn's already-billed calls.
    continuation_usage: Option<TokenUsage>,
}

/// Persistent store for [`AgentTask`] rows.
///
/// Implementations are required to expose the index surface documented on
/// the [`super`] module. The reference in-memory implementation below is
/// used in tests and also acts as the semantic spec for any future SQL- or
/// Redis-backed store.
///
/// # CAS discipline (Phase 2.7)
///
/// Every **state transition** a worker drives goes through a CAS-guarded
/// method on this trait. The `(worker_id, lease_id)` tuple on every
/// `Running` row is the ownership token — no mutation reaches the row
/// unless the caller proves it holds the current lease.
///
/// **Structural primitives** (`insert`, `update`, `clear`) exist for
/// store rehydration, test scaffolding, and internal batch plumbing.
/// Worker code must never use them to drive lifecycle transitions —
/// use the CAS helpers below instead.
///
/// ## Worker-visible CAS entry points
///
/// | Method | Transition | Guards |
/// |--------|------------|--------|
/// | [`submit_root_turn`](Self::submit_root_turn) | → `Pending` / `Queued` | kind, status, root invariants |
/// | [`promote_next_queued_root`](Self::promote_next_queued_root) | `Queued` → `Pending` | active-root slot free |
/// | [`try_acquire_task`](Self::try_acquire_task) / [`acquire_next_runnable`](Self::acquire_next_runnable) | `Pending` → `Running` | status CAS, retry budget |
/// | [`heartbeat_task`](Self::heartbeat_task) | `Running` → `Running` | `(worker, lease)` CAS |
/// | [`pause_on_children`](Self::pause_on_children) | `Running` → `WaitingOnChildren` | `(worker, lease)` CAS |
/// | [`pause_on_confirmation`](Self::pause_on_confirmation) | `Running` → `AwaitingConfirmation` | `(worker, lease)` CAS |
/// | [`spawn_tool_children`](Self::spawn_tool_children) | parent `Running` → `WaitingOnChildren` + children `Pending` | `(worker, lease)` CAS, non-leaf |
/// | [`spawn_subagent_invocation`](Self::spawn_subagent_invocation) | parent `Running` → `WaitingOnChildren` + invocation `WaitingOnChildren` + child root `Pending` | `(worker, lease)` CAS, non-leaf |
/// | [`complete_task`](Self::complete_task) | `Running` → `Completed` + parent recompute | `(worker, lease)` CAS |
/// | [`fail_task`](Self::fail_task) | `Running` → `Failed` + parent recompute | `(worker, lease)` CAS |
/// | [`resume_from_confirmation`](Self::resume_from_confirmation) | `AwaitingConfirmation` → `Pending` | status CAS |
/// | [`approve_confirmation_and_acquire`](Self::approve_confirmation_and_acquire) | `AwaitingConfirmation` → `Running` | status CAS, retry budget |
/// | [`reject_confirmation`](Self::reject_confirmation) | `AwaitingConfirmation` → `Failed` + parent recompute | status CAS |
/// | [`cancel_tree`](Self::cancel_tree) | subtree → `Cancelled` | existence check |
/// | [`release_expired_leases`](Self::release_expired_leases) | `Running` → `Pending` / `Failed` | lease-expiry CAS, recovery matrix |
/// The bounded result of [`AgentTaskStore::probe_children`].
#[derive(Debug, Default, Clone)]
pub struct ChildProbe {
    /// At most `live_limit` NON-TERMINAL children — the frontier the stall
    /// walk will expand.
    pub live: Vec<AgentTask>,
    /// Whether ANY child of the parent — terminal or live — showed a sign of
    /// life at or after the cutoff.
    pub fresh_activity: bool,
}

impl ChildProbe {
    /// A task row's effective activity: `last_activity_at ?? created_at`
    /// (ADR-0003 I1).
    ///
    /// A row carries no `last_activity_at` until it is first acquired, and
    /// rows predating migration 0012 carry none at all. Its creation is still
    /// its initial evidence of life — treating `None` as "stale forever" reaps
    /// parents that parked on children they had just spawned.
    ///
    /// This is the single definition every backend's `probe_children` must
    /// agree with, in Rust or in SQL.
    #[must_use]
    pub fn effective_activity(task: &AgentTask) -> OffsetDateTime {
        task.last_activity_at.unwrap_or(task.created_at)
    }

    /// Fold a full child list into the bounded shape. Used by the default
    /// (unbounded) implementation and by the in-memory store, and it is the
    /// reference semantics the SQL backends' bounded queries must match.
    #[must_use]
    pub fn from_rows(children: Vec<AgentTask>, cutoff: OffsetDateTime, live_limit: usize) -> Self {
        let mut out = Self::default();
        for child in children {
            if Self::effective_activity(&child) >= cutoff {
                out.fresh_activity = true;
            }
            if !child.status.is_terminal() && out.live.len() < live_limit {
                out.live.push(child);
            }
        }
        out
    }
}

#[async_trait]
pub trait AgentTaskStore: Send + Sync {
    /// Insert a new task row. **Not a mutation API** — see the
    /// [CAS discipline](Self#cas-discipline-phase-27) table for the
    /// correct entry point for each state transition.
    ///
    /// This is a structural insertion primitive for store rehydration
    /// and internal batch plumbing. Worker code should use
    /// [`submit_root_turn`](Self::submit_root_turn) for root turns
    /// and the CAS helpers for every other transition.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, if a row with the
    /// same `id` already exists, or if the blocking-root invariant would
    /// be violated.
    async fn insert(&self, task: AgentTask) -> Result<()>;

    /// Durably admit a new root turn on a thread, respecting the
    /// same-thread FIFO queue.
    ///
    /// The supplied task must be a freshly-constructed
    /// [`TaskKind::RootTurn`] in [`TaskStatus::Pending`] — i.e. the output
    /// of [`AgentTask::new_root_turn`] (possibly with a custom retry budget
    /// but not yet mutated by any other state-transition helper). The store
    /// decides, atomically, whether to admit it as `Pending` (if the thread
    /// has no blocking root yet) or convert it to `Queued` (if a blocking
    /// root already holds the slot).
    ///
    /// Returns the admitted row **as persisted**, so callers can observe the
    /// final status without a second `get`. This is how Phase 2.2's
    /// admission contract is expressed in the trait surface:
    ///
    /// - If `task.thread_id` has no [`TaskStatus::blocks_root_admission`]
    ///   root, the row lands in [`TaskStatus::Pending`].
    /// - Otherwise, the row lands in [`TaskStatus::Queued`] behind every
    ///   previously queued root on the same thread. `created_at` (and, as a
    ///   tie-breaker, `id`) define the FIFO order.
    ///
    /// # Errors
    /// Returns an error if the task is not a freshly-constructed root turn,
    /// if the row fails validation, or if the underlying store write fails.
    async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask>;

    /// Durably admit a root turn with at-least-once idempotency and
    /// queued-root back-pressure, all inside a single transaction.
    ///
    /// This is the transport-facing admission entry point (Phase 10 ·
    /// E). It composes three guarantees that must be atomic with each
    /// other so a retried, restart-surviving client request produces
    /// exactly one effect:
    ///
    /// 1. **Idempotency** — when `params.idempotency` is set, the store
    ///    looks up the `request_id` in the durable idempotency table
    ///    *inside the admission transaction*. A matching record replays
    ///    the original task; a mismatching kind/fingerprint returns
    ///    [`SubmitRootTurnError::IdempotencyConflict`]; an absent record
    ///    is claimed and committed together with the task insert (no
    ///    time-of-check / time-of-use window).
    /// 2. **Back-pressure** — when `params.max_queued_depth` is set, a
    ///    submission that would exceed the per-thread queued-root cap is
    ///    rejected with [`SubmitRootTurnError::QueueDepthExceeded`]
    ///    before anything is written.
    /// 3. **Admission** — the same FIFO admission rule as
    ///    [`Self::submit_root_turn`].
    ///
    /// The returned [`SubmitRootTurnOutcome`] reports whether the call
    /// replayed a prior submission and the resulting queued-root depth.
    ///
    /// # Errors
    /// Returns [`SubmitRootTurnError`] for idempotency conflicts, queue
    /// back-pressure, and any underlying store failure.
    async fn submit_root_turn_idempotent(
        &self,
        params: SubmitRootTurnParams,
    ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError>;

    /// Atomically claim a `request_id` for a non-submission control-plane
    /// operation (`CreateThread`, `ForkThread`, `DecideConfirmation`)
    /// against the durable idempotency table.
    ///
    /// This is an **atomic reservation**: the first caller inserts a
    /// placeholder row (`result_json = JSON null`) inside the same
    /// lock/transaction that checks for an existing row, so two
    /// concurrent retries of the same key can never both run the effect.
    ///
    /// Returns [`IdempotencyClaim::Fresh`] only for the caller that won
    /// the reservation (it now owns the key, must run the effect, then
    /// call [`Self::record_idempotency`] to fill in the result),
    /// [`IdempotencyClaim::Replay`] when a matching record with a
    /// recorded result already exists, and [`IdempotencyClaim::Conflict`]
    /// when the key was used for a different kind/fingerprint **or** is
    /// still reserved by a concurrent in-flight claim (placeholder not
    /// yet filled). A `Conflict` caller must not run the effect.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn claim_idempotency(
        &self,
        request_id: &str,
        kind: IdempotencyKind,
        fingerprint: &[u8],
    ) -> Result<IdempotencyClaim>;

    /// Persist the durable references for a freshly-completed
    /// non-submission idempotent operation.
    ///
    /// Fills the reservation placeholder written by
    /// [`Self::claim_idempotency`]: it sets `result_json` on the existing
    /// row only when that row is still the placeholder (`JSON null`).
    /// Idempotent on the key — a second call once a real result has been
    /// recorded is a no-op, so a retry that races past
    /// [`Self::claim_idempotency`] cannot corrupt the record. If no prior
    /// claim row exists (defensive), the full record is inserted.
    ///
    /// # Errors
    /// Returns an error if the store write fails.
    async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()>;

    /// Look up a task by id.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>>;

    /// Replace a task row with an updated version. **Not a mutation
    /// API** — see the [CAS discipline](Self#cas-discipline-phase-27)
    /// table for the correct entry point for each state transition.
    ///
    /// This is a structural replacement primitive for store
    /// rehydration and test scaffolding. If you are reaching for
    /// `update()` in worker code, you almost certainly want one of
    /// the CAS-guarded helpers on this trait instead.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, does not exist, or
    /// if the blocking-root invariant would be violated after the update.
    async fn update(&self, task: AgentTask) -> Result<()>;

    /// List every task (root and descendants) bound to a thread.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>>;

    /// List the direct children of a task.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>>;

    /// The direct children of `parent_id`, read for the subagent stall probe
    /// with the READ ITSELF BOUNDED (ADR-0003 I5).
    ///
    /// Returns at most `live_limit` **non-terminal** children — the live
    /// frontier the walk will expand — plus a single flag: whether **any**
    /// child, terminal or not, has effective activity
    /// (`last_activity_at ?? created_at`, per I1) at or after `cutoff`.
    ///
    /// ## Why not just `list_children`
    ///
    /// The probe used to read every child and filter terminal rows in memory.
    /// Terminal task rows are RETAINED, so a parent that has run thousands of
    /// tool calls materializes thousands of rows on **every** probe of **every**
    /// parked ancestor. Past a few thousand, the probe cannot finish inside its
    /// timeout (I5) — and a probe that times out is treated as *alive*. The
    /// result is the exact inverse failure: unbounded database work, and a
    /// genuinely wedged child that can never be reaped because its probe never
    /// completes. Bounding the read is what keeps enforcement able to converge.
    ///
    /// Terminal children are folded into `fresh_activity` rather than returned:
    /// their activity stamp IS their terminal instant (every terminal
    /// transition stamps it, per I3), which is at or after any event they
    /// committed — and they share their parent's `thread_id` (`new_child`
    /// inherits it), so they contribute no thread the walk has not already
    /// collected. A nested subagent root, which DOES live on its own thread, is
    /// a root task reached by the invocation-linkage hop, never by this call.
    ///
    /// The default implementation derives the answer from [`Self::list_children`]
    /// and is therefore **unbounded** — correct, but only appropriate for
    /// in-memory and decorator stores. Durable backends must override it with
    /// bounded queries.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn probe_children(
        &self,
        parent_id: &AgentTaskId,
        cutoff: OffsetDateTime,
        live_limit: usize,
    ) -> Result<ChildProbe> {
        let children = self.list_children(parent_id).await?;
        Ok(ChildProbe::from_rows(children, cutoff, live_limit))
    }

    /// List every task currently in the given status.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>>;

    /// Return the currently blocking [`TaskKind::RootTurn`] for a thread,
    /// if one exists.
    ///
    /// "Blocking" means the row is a root turn whose status satisfies
    /// [`TaskStatus::blocks_root_admission`] (i.e. `Pending`, `Running`,
    /// `WaitingOnChildren`, or `AwaitingConfirmation`). This is the primary
    /// consumer of the partial-unique "one blocking root per thread"
    /// invariant. Queued roots are explicitly **not** returned here — use
    /// [`AgentTaskStore::list_queued_roots`] to see them.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>>;

    /// Return the queued root turns on a thread in strict FIFO order.
    ///
    /// Rows are ordered by `created_at` ascending, tiebroken by `id` so the
    /// order is deterministic even at sub-microsecond time resolution. The
    /// returned vector is empty if the thread has no queued roots.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>>;

    /// Promote the FIFO head of the thread's queued-root list into
    /// [`TaskStatus::Pending`], if the thread has no blocking root.
    ///
    /// Returns `Some(task)` with the newly-promoted row when a promotion
    /// fired. Returns `None` if the thread has no queued roots to promote,
    /// or if the thread's active-root slot is still held by a blocking root
    /// (so no promotion is legal yet).
    ///
    /// This method is a no-op idempotent operation: callers may invoke it
    /// every time a root completes, fails, or cancels, without needing to
    /// know whether another queued root is waiting. Retries of the active
    /// root that release the lease back to `Pending` do **not** fire a
    /// promotion, because the active-root slot is still held — queued
    /// roots cannot overtake a retry.
    ///
    /// # Errors
    /// Returns an error if the promotion transition or the store write
    /// fails.
    async fn promote_next_queued_root(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>>;

    /// Atomically acquire a lease on a specific [`TaskStatus::Pending`]
    /// task by id.
    ///
    /// This is the single guarded admission point for worker ownership
    /// in Phase 2.3: a successful call transitions the row from
    /// `Pending` to [`TaskStatus::Running`], stamps `(worker, lease,
    /// expires_at)`, and increments `attempt`. Two workers racing to
    /// claim the same `id` are serialised by the store's write lock, so
    /// exactly one observes `Ok(Some(task))` and the other observes
    /// `Ok(None)` because the row's status is no longer runnable.
    ///
    /// The method is intentionally **narrow**:
    ///
    /// - Returns `Ok(Some(task))` on a successful claim, where `task`
    ///   is the row as persisted (status `Running`, lease fields set).
    /// - Returns `Ok(None)` if the task exists but is not runnable
    ///   (any non-`Pending` status, including waiting / terminal /
    ///   already-running), or if the task does not exist at all.
    /// - Returns `Ok(None)` if the row is `Pending` but its retry
    ///   budget is already exhausted. Phase 2.5 fails the row closed
    ///   under the store's write lock via
    ///   [`super::recovery::classify_recovery`] before returning, so
    ///   the caller sees the same `None` it would see for any other
    ///   non-runnable row and the worker pool is never poisoned by a
    ///   row that will never make progress.
    /// - Returns `Err` only on true store-level failures: validation
    ///   errors or downstream write errors.
    ///
    /// Callers who want scan-and-claim semantics should use
    /// [`AgentTaskStore::acquire_next_runnable`] instead.
    ///
    /// # Errors
    /// - Row-level state transitions that [`AgentTask::mark_running`]
    ///   rejects for reasons other than retry exhaustion (Phase 2.5
    ///   handles retry exhaustion as a fail-closed transition, not an
    ///   error).
    /// - Store-level write errors.
    async fn try_acquire_task(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>>;

    /// Scan for the oldest runnable task across every kind and thread,
    /// then atomically claim it for `worker`.
    ///
    /// Runnable means status [`TaskStatus::Pending`]. Waiting states
    /// (`WaitingOnChildren`, `AwaitingConfirmation`), the `Queued`
    /// submission queue, and terminal states are never visible to this
    /// scan — callers that drive the worker loop can call this method
    /// in a loop and never accidentally lease a paused task.
    ///
    /// FIFO order is `(created_at, id)` ascending, so the oldest
    /// submission wins and ties are broken deterministically by `id`.
    /// Root turns and tool-runtime children share the same scan, which
    /// is exactly the "one acquisition and lease model that prevents
    /// double ownership across task kinds" Phase 2.3 requires.
    ///
    /// Phase 2.5 adds fail-closed skip-on-exhaustion semantics: every
    /// scanned row is first classified by
    /// [`super::recovery::classify_recovery`], and any head whose
    /// retry budget is already exhausted is atomically transitioned
    /// to [`TaskStatus::Failed`] with a canonical `last_error` prefix
    /// and removed from the runnable index. The scan then continues
    /// with the next head, so a single exhausted row can never
    /// poison the entire worker pool. `Ok(None)` is returned only
    /// when every row in the runnable index has been successfully
    /// claimed or failed closed.
    ///
    /// Returns `Ok(Some(task))` on a successful claim, `Ok(None)` if
    /// the runnable pool drains without a valid claim.
    ///
    /// # Errors
    /// - Row-level state transitions that [`AgentTask::mark_running`]
    ///   or the Phase 2.5 fail-closed path reject for structural
    ///   reasons (never retry exhaustion, which is handled in-band).
    /// - Store-level write errors.
    async fn acquire_next_runnable(
        &self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>>;

    /// Refresh the lease on a running task, guarded by the caller's
    /// `(worker, lease)` ownership tuple.
    ///
    /// The CAS check covers **both** the `WorkerId` and the `LeaseId`:
    /// a stale worker that re-used the same `WorkerId` but whose lease
    /// has been re-acquired by another worker cannot extend the row it
    /// no longer owns. This is the guard the acceptance criterion
    /// "heartbeats from the wrong worker fail" relies on.
    ///
    /// On success the row's `last_heartbeat_at` becomes `now` and its
    /// `lease_expires_at` becomes `expires_at`. Pass the existing expiry
    /// in if the caller does not want to extend it; passing a later
    /// expiry is how workers hold their lease open across a long-running
    /// tool call.
    ///
    /// # Activity
    ///
    /// `activity` is the newest **evidence of work** the running task has
    /// produced (a provider frame, a tool progress emission, an event
    /// commit), or `None` if it produced none since the last beat. It
    /// rides the heartbeat because the heartbeat already holds the
    /// `(worker, lease)` CAS and already writes this row — so activity is
    /// persisted once per tick instead of once per provider frame.
    ///
    /// It only ever moves `last_activity_at` **forward**; a beat carrying
    /// `None` leaves the column untouched. Note this is emphatically NOT
    /// `last_heartbeat_at`: the heartbeat renews unconditionally while the
    /// task future is alive, so a child wedged on a half-open connection
    /// beats forever. A beat proves the process is alive; `activity`
    /// proves work happened. The subagent stall budget reads the latter.
    ///
    /// Returns the refreshed row on success.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `task is not running` — if the row is in any non-`Running`
    ///   status. A heartbeat against a terminal / waiting / queued /
    ///   pending row fails rather than silently succeeding.
    /// - `heartbeat rejected: worker mismatch` — if `worker` does not
    ///   match the current `worker_id`.
    /// - `heartbeat rejected: lease mismatch` — if `lease` does not
    ///   match the current `lease_id`.
    async fn heartbeat_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        expires_at: OffsetDateTime,
        activity: Option<OffsetDateTime>,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

    /// Sweep every expired lease whose `lease_expires_at <= now` and
    /// deterministically decide, per row, whether to return it to
    /// [`TaskStatus::Pending`] or fail it closed via Phase 2.5's
    /// recovery matrix.
    ///
    /// This is the recovery sweep that guarantees a crashed worker's
    /// rows never stay leased forever. It walks the lease-expiry index
    /// in ascending order and stops as soon as it hits a row whose
    /// expiry is still in the future, so the cost is proportional to
    /// the number of actually-expired leases rather than the size of
    /// the live worker pool.
    ///
    /// For each expired row Phase 2.5 runs
    /// [`super::recovery::classify_recovery`] under the store write
    /// lock and then atomically applies the matching transition:
    ///
    /// - [`RecoveryAction::Requeue`] — the row returns to
    ///   [`TaskStatus::Pending`] with the failed attempt counted
    ///   against the retry budget (Phase 2.3 behavior).
    /// - [`RecoveryAction::FailClosed`] — the row is transitioned to
    ///   [`TaskStatus::Failed`] with a canonical `last_error` prefix
    ///   from [`super::recovery::FailureReason`]. Retry-budget
    ///   exhaustion (`LeaseExpiredBudgetExhausted`) and staged
    ///   listen/execute operations on tool-runtime children
    ///   (`UnsafePreparedOperationRecovery`) both take this path,
    ///   so the journal never enters an endless requeue loop and
    ///   never risks double-executing an in-flight external
    ///   operation.
    ///
    /// Returns one [`RecoveryRecord`] per swept row in expiry order
    /// so callers can log and react per-outcome without a second
    /// round trip to `get()`.
    ///
    /// Each call reclaims at most [`LEASE_RELEASE_BATCH`] rows (in
    /// expiry order) so a mass worker outage cannot turn one sweep
    /// into a giant write set that locks every expired task at once.
    /// A caller draining a backlog must loop until a call returns
    /// fewer than [`LEASE_RELEASE_BATCH`] records.
    ///
    /// # Errors
    /// Returns an error if a release or fail-closed transition, or
    /// the underlying store write, fails.
    async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>>;

    /// Return a `Running` row this worker still owns to the runnable
    /// queue — the worker-initiated twin of the expiry sweep's requeue
    /// arm (issue #354).
    ///
    /// The one production caller is the host's completed-turn
    /// collision handler: a promoted successor whose commit lost its
    /// turn slot to a foreign FULL-TURN checkpoint must not shift past
    /// it (its snapshot lacks the occupant's billed usage) and must
    /// not fail terminally either — the cancellation contract promises
    /// follow-up work completes. Requeueing re-runs the turn from the
    /// fresh committed head with crash-equivalent semantics: the turn
    /// never committed, so re-execution is exactly what the durable
    /// journal already promises after a worker crash, and the same
    /// retry budget applies.
    ///
    /// A successful call CAS-checks that the row exists, is
    /// [`TaskStatus::Running`], and is owned by exactly
    /// `(worker, lease)`, then applies [`AgentTask::release_lease`] —
    /// the identical transition (and identical budget accounting) the
    /// expiry sweep uses — and persists the row. The three non-error
    /// outcomes are reported as [`RequeueOutcome`] variants rather
    /// than errors because each has a distinct correct caller
    /// reaction; see the variant docs.
    ///
    /// # Boundary event
    ///
    /// `boundary`, when supplied, is committed to the task's thread
    /// journal (with its `thread_events_available` outbox advisory)
    /// ATOMICALLY with the ownership CAS and the release — inside the
    /// same transaction on the durable backends, under the same task
    /// write lock (via the wired [`CancellationMarkerSink`]) on the
    /// in-memory store, exactly like `cancel_tree`'s terminal markers.
    /// This is what makes the "abandoned streamed attempt" terminator
    /// race-free: the event lands iff THIS caller
    /// still owned the row, and it is durable before the row becomes
    /// acquirable — so a replacement's `Start` can never precede it
    /// and a stale caller can never inject it into a replacement's
    /// stream. `NotOwned` / `BudgetExhausted` commit nothing. An
    /// in-memory store with no sink wired requeues without the event
    /// (bare test-support stores only; production wiring and the
    /// journal bundle always carry the sink).
    ///
    /// # Errors
    /// Returns an error only for store I/O failures, a boundary-event
    /// commit failure, or an invalid release transition; ownership and
    /// budget conditions are reported through the outcome enum. On the
    /// non-atomic in-memory path a boundary error leaves the row still
    /// `Running` under the caller's lease (nothing was released).
    async fn requeue_owned_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        boundary: Option<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<RequeueOutcome>;

    /// Pause a running task on outstanding child work, dropping the
    /// lease atomically with the typed-state mutation.
    ///
    /// This is one of Phase 2.4's two journal-guarded pause entry
    /// points. A successful call:
    ///
    /// 1. CAS-checks that the row exists, is in
    ///    [`TaskStatus::Running`], and is owned by `(worker, lease)`.
    /// 2. Transitions the row to [`TaskStatus::WaitingOnChildren`]
    ///    via [`AgentTask::wait_on_children`], stamping the typed
    ///    [`crate::journal::TaskState::WaitingOnChildren`] payload with the supplied
    ///    `continuation`.
    /// 3. Drops the lease (`worker_id` / `lease_id` /
    ///    `lease_expires_at` / `last_heartbeat_at` all cleared) so the
    ///    row is no longer reachable from the lease-expiry index and
    ///    cannot be re-acquired by [`AgentTaskStore::try_acquire_task`]
    ///    or [`AgentTaskStore::acquire_next_runnable`].
    /// 4. Persists the row under the same write lock.
    ///
    /// Returns the persisted paused row on success.
    ///
    /// `child_count` must be `> 0`. The caller is responsible for
    /// having spawned (or about to spawn) that many child tasks; this
    /// method only updates the parent and does not insert the
    /// children — Phase 2.6 will own the child orchestration.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `pause rejected: not running` — if the row is in any
    ///   non-`Running` status.
    /// - `pause rejected: worker mismatch` — if `worker` does not
    ///   match the current `worker_id`.
    /// - `pause rejected: lease mismatch` — if `lease` does not
    ///   match the current `lease_id`.
    /// - Row-level errors from [`AgentTask::wait_on_children`]
    ///   (e.g. zero `child_count`).
    async fn pause_on_children(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        child_count: u32,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

    /// Wake a parent parked in [`TaskStatus::WaitingOnChildren`] for an
    /// early **steering** resume (R2), atomically flipping it to
    /// [`TaskStatus::Pending`] with a
    /// [`crate::journal::TaskState::ReadyToResume`] payload carrying a
    /// non-empty `steering` block (the overload that keeps the durable
    /// state `kind` unchanged).
    ///
    /// This is the host-driven entry point a steering-wake sweep calls
    /// when its mailbox is non-empty and the parent is parked. The
    /// transition is a single CAS on the parent's status:
    ///
    /// - If the parent is in [`TaskStatus::WaitingOnChildren`], it is
    ///   moved to `Pending` + `ReadyToResume{ steering }` (dropping
    ///   `pending_child_count` to zero) and `Ok(Some(parent))` is
    ///   returned. A worker then acquires the `Pending` row and runs
    ///   [`crate::worker::resume_for_steering`].
    /// - If the parent is in **any other status** (already woken,
    ///   running its exchange, already fanned in to `ReadyToResume`,
    ///   or terminal), the wake is a no-op and `Ok(None)` is returned.
    ///   This makes the sweep idempotent and closes the double-resume
    ///   race: if the last child fans the parent in to `ReadyToResume`
    ///   first, the wake simply declines and the ordinary resume path
    ///   delivers the steering note at that boundary instead.
    ///
    /// # Errors
    /// - `steering content must be non-empty` — an empty payload is
    ///   ambiguous with an ordinary fan-in and is rejected.
    /// - `task does not exist` — if no row with `parent_id` is stored.
    /// - Row-level errors from
    ///   [`AgentTask::begin_steering_resume`].
    async fn enqueue_steering_resume(
        &self,
        parent_id: &AgentTaskId,
        steering: Vec<agent_sdk_foundation::llm::ContentBlock>,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>>;

    /// Re-park a running steering exchange (a
    /// [`crate::journal::TaskState::ReadyToResume`] row whose `steering`
    /// payload is set) on the still-running children, updating each
    /// re-attach child's
    /// `spawn_index` to its position in the new continuation's
    /// `pending_tool_calls` and transitioning the parent back to
    /// [`TaskStatus::WaitingOnChildren`] (or straight to
    /// [`TaskStatus::Pending`] + [`crate::journal::TaskState::ReadyToResume`]
    /// when every re-attach child has already finished).
    ///
    /// A successful call, under a single write:
    ///
    /// 1. CAS-checks the parent exists, is in [`TaskStatus::Running`],
    ///    is owned by `(worker, lease)`, and carries a steering
    ///    `ReadyToResume` state (non-empty `steering`).
    /// 2. Rewrites `reattach[i].spawn_index = i` for every re-attach
    ///    child so the ordinary fan-in aggregation maps them onto the
    ///    new continuation's tool-use ids.
    /// 3. Derives the live child count from the journal (non-terminal
    ///    children among `reattach`) and re-parks the parent via
    ///    [`AgentTask::repark_after_steering`], dropping the lease.
    ///
    /// `reattach` is the ordered list of still-running child ids; the
    /// new `child_ids` recorded on the parent equals `reattach`. The
    /// completed children from the parked batch are intentionally
    /// dropped from the parent's tracked `child_ids` (their real
    /// results were already folded into the steering LLM history).
    ///
    /// # Errors
    /// - `repark rejected: task does not exist` — no row with
    ///   `parent_id`.
    /// - `repark rejected: not running` / `worker mismatch` / `lease
    ///   mismatch` — CAS failure.
    /// - `repark rejected: not a steering resume` — the row is running
    ///   but not in `SteeringResume`.
    /// - `repark rejected: re-attach child … missing` — a `reattach`
    ///   id is not a child of the parent.
    /// - Row-level errors from [`AgentTask::repark_after_steering`].
    async fn repark_after_steering(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

    /// Pause a running task on a user confirmation, dropping the
    /// lease atomically with the typed-state mutation.
    ///
    /// This is the second of Phase 2.4's two journal-guarded pause
    /// entry points. A successful call:
    ///
    /// 1. CAS-checks that the row exists, is in
    ///    [`TaskStatus::Running`], and is owned by `(worker, lease)`.
    /// 2. Transitions the row to [`TaskStatus::AwaitingConfirmation`]
    ///    via [`AgentTask::await_confirmation`], stamping the typed
    ///    [`crate::journal::TaskState::AwaitingConfirmation`] payload with the
    ///    supplied `continuation` and (optional) `prepared_operation`.
    /// 3. Drops the lease so the row is no longer reachable from the
    ///    lease-expiry index and cannot be re-acquired by either
    ///    acquisition path.
    /// 4. Persists the row under the same write lock.
    ///
    /// Returns the persisted paused row on success. The pause is
    /// **idempotent through the durable layer** — workers may safely
    /// retry on transient store errors because the CAS guard rejects
    /// any retry whose lease no longer owns the row.
    ///
    /// `prepared_operation` is `None` for non-listen tools (the common
    /// case) and `Some(...)` when the awaited tool staged a
    /// long-running listen/execute operation that the resume path
    /// will need to either execute or cancel.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `pause rejected: not running` — if the row is in any
    ///   non-`Running` status.
    /// - `pause rejected: worker mismatch` — if `worker` does not
    ///   match the current `worker_id`.
    /// - `pause rejected: lease mismatch` — if `lease` does not
    ///   match the current `lease_id`.
    async fn pause_on_confirmation(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

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
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<AgentTask>)>;

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
    /// [`AgentTaskStore::spawn_tool_children`], plus schema errors from
    /// [`AgentTask::new_subagent_invocation`].
    async fn spawn_subagent_invocation(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: SubagentInvocationSpawn,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, AgentTask, AgentTask)>;

    /// Atomically persist N durable subagent invocations under one
    /// parent transition.
    ///
    /// This is the fan-out flavour of
    /// [`AgentTaskStore::spawn_subagent_invocation`]. A successful
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
    /// [`AgentTaskStore::spawn_subagent_invocation`], plus:
    /// - `spawn rejected: spawns must be non-empty` — empty input.
    /// - `spawn rejected: duplicate child_thread_id ...` — two
    ///   entries collide on the same child thread.
    /// - schema errors from [`AgentTask::new_subagent_invocation`].
    async fn spawn_subagent_batch(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawns: Vec<SubagentInvocationSpawn>,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>)>;

    /// Atomically persist a **mixed** batch — N durable subagent
    /// invocations plus M tool-runtime children — under one parent
    /// transition.
    ///
    /// This is the primitive behind a coordinator turn that emits
    /// subagent calls alongside ordinary tool calls (an LLM routinely
    /// mixes both in one response). It is the union of
    /// [`AgentTaskStore::spawn_subagent_batch`] and
    /// [`AgentTaskStore::spawn_tool_children`], not a new contract:
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
    /// envelope as [`AgentTaskStore::spawn_subagent_batch`], plus every
    /// [`validate_mixed_children_spawn`] rejection.
    async fn spawn_mixed_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: MixedChildrenSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedMixedChildren>;

    /// Find the parked [`TaskKind::Subagent`] invocation whose durable
    /// [`crate::journal::TaskState::SubagentInvocation`] linkage points
    /// at `child_root_id`.
    ///
    /// This is the read-only counterpart of the terminal-transition
    /// wake path (`resume_linked_subagent_invocation` in each backend):
    /// it returns the invocation row only while it is still
    /// [`TaskStatus::WaitingOnChildren`] on that child root — exactly
    /// the window in which the child-thread root can be live. Once the
    /// child root reaches a terminal state and the invocation has been
    /// woken, the linkage is consumed and this returns `None`.
    ///
    /// The service host uses this at child-root acquisition time to
    /// read the resolved `spec.timeout_ms` off the invocation and
    /// derive the child's wall-clock execution deadline without any
    /// schema addition — a re-acquiring worker (crash, lease expiry)
    /// re-resolves the same deadline from the same durable rows.
    ///
    /// # Errors
    /// Returns an error only for store-level read failures. "No linked
    /// invocation" (plain root turns, already-woken invocations) is
    /// `Ok(None)`, not an error.
    async fn find_subagent_invocation_for_child_root(
        &self,
        child_root_id: &AgentTaskId,
    ) -> Result<Option<AgentTask>>;

    /// List every [`TaskKind::Subagent`] invocation currently parked in
    /// [`TaskStatus::WaitingOnChildren`], ordered by `(created_at, id)`.
    ///
    /// This is the candidate feed for the host's subagent-deadline
    /// sweep (issue #299): an invocation stays `WaitingOnChildren` for
    /// its linked child root's whole lifetime, so this listing covers
    /// every live child regardless of the child's own state. Compared
    /// to `list_by_status(WaitingOnChildren)` + a client-side kind
    /// filter, the kind predicate is pushed into the store so the
    /// sweep never materializes unrelated parked parents (whose
    /// `state_json` continuations dominate row size) — the SQL
    /// backends serve this from the status index plus the kind
    /// predicate.
    ///
    /// # Errors
    /// Returns an error only for store-level read failures.
    async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>>;

    /// Transition a running child to [`TaskStatus::Completed`] and,
    /// under the same write lock, recompute the parent's
    /// `pending_child_count` from the live-children index so the
    /// parent can resume as soon as its last live child reaches a
    /// terminal state.
    ///
    /// This is the Phase 2.6 successful-child resume trigger. A
    /// successful call:
    ///
    /// 1. CAS-checks that the child exists, is in
    ///    [`TaskStatus::Running`], and is owned by `(worker, lease)`.
    /// 2. Transitions the child via [`AgentTask::complete`] and
    ///    drops its lease.
    /// 3. If the child has a parent row (it always does for Phase
    ///    2.6 — tool-runtime rows are leaves spawned under a parent),
    ///    walks the parent's `by_parent` bucket, counts the live
    ///    (non-terminal) children, and calls
    ///    [`AgentTask::recompute_pending_children`] on the parent.
    ///    When the live count hits zero the parent flips back to
    ///    [`TaskStatus::Pending`] and its typed
    ///    [`crate::journal::TaskState::WaitingOnChildren`] payload is
    ///    cleared so a worker can re-acquire it.
    /// 4. Leaves the parent alone if it is terminal (e.g. a
    ///    [`AgentTaskStore::cancel_tree`] sweep ran between the
    ///    worker's start and finish). The child's own terminal
    ///    transition still runs so stale workers can still report
    ///    their last result on the way out.
    ///
    /// Returns `(child, parent)` where `parent` is `None` if the
    /// child is a root (Phase 2.6 tool-runtime children always have
    /// a parent, but the signature is kept symmetric with
    /// [`AgentTaskStore::fail_task`] so a future phase can reuse
    /// the entry point for any kind of leaf).
    ///
    /// # Errors
    /// - `complete_task rejected: task ... does not exist` — no
    ///   row with the supplied id.
    /// - `complete_task rejected: task ... is not running` — row
    ///   is in any non-`Running` status.
    /// - `complete_task rejected: worker mismatch` — `worker`
    ///   does not match the row's current `worker_id`.
    /// - `complete_task rejected: lease mismatch` — `lease` does
    ///   not match the row's current `lease_id`.
    /// - Schema errors from [`AgentTask::complete`] or
    ///   [`AgentTask::recompute_pending_children`].
    async fn complete_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)>;

    /// Transition a running child to [`TaskStatus::Completed`] with a
    /// durable result payload, then recompute the parent's
    /// `pending_child_count` exactly as [`AgentTaskStore::complete_task`]
    /// does.
    ///
    /// This is the Phase 5.4 entry point for tool-runtime children
    /// that carry a serialized [`agent_sdk_foundation::ToolResult`] back to
    /// the parent. The `result_payload` is persisted on the child row's
    /// [`AgentTask::result_payload`] field so the parent's resume path
    /// can read it from the journal via
    /// [`AgentTaskStore::list_children`] without relying on in-memory
    /// state.
    ///
    /// # Errors
    /// Same error envelope as [`AgentTaskStore::complete_task`].
    async fn complete_task_with_result(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        result_payload: serde_json::Value,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)>;

    /// Transition a running child to [`TaskStatus::Failed`] with
    /// `error` and recompute the parent's `pending_child_count` the
    /// same way [`AgentTaskStore::complete_task`] does.
    ///
    /// Phase 2.6 treats child failures exactly like child completions
    /// at the **journal** layer: a failed child still counts as a
    /// terminal child, so the parent's counter decrements and the
    /// parent becomes runnable as soon as the last live child reaches
    /// any terminal state. The parent's agent loop is responsible for
    /// inspecting [`AgentTaskStore::list_children`] on resume and
    /// deciding how to aggregate mixed success / failure outcomes —
    /// that decision belongs in the agent layer, not the journal.
    ///
    /// # Errors
    /// Same error envelope as
    /// [`AgentTaskStore::complete_task`] but routed through
    /// [`AgentTask::fail`] instead of [`AgentTask::complete`].
    async fn fail_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)>;

    /// Cancel `root_id` and every descendant in its subtree under a
    /// single store write, producing a fully-terminal subtree in
    /// deterministic breadth-first (level) order.
    ///
    /// This is the Phase 2.6 cancellation-tree entry point, extended
    /// by Phase 7.6 to cascade across thread boundaries through
    /// [`crate::journal::TaskState::SubagentInvocation`] linkage. A
    /// successful call:
    ///
    /// 1. Walks the `by_parent` index transitively from `root_id`
    ///    to collect every live descendant id. When a task carries
    ///    [`crate::journal::TaskState::SubagentInvocation`] state, the
    ///    linked `child_root_task_id` is followed so cancellation
    ///    reaches child-thread roots and their descendants.
    /// 2. Transitions each non-terminal row via [`AgentTask::cancel`]
    ///    in the order `[root, child_1, child_2, ..., grandchildren,
    ///    ...]`. Each cancel drops the row's lease (if any), removes
    ///    the row from the runnable / lease-expiry / queued / active-root
    ///    indexes, and clears the typed
    ///    [`crate::journal::TaskState`] payload so the state ↔ status
    ///    invariant holds.
    /// 3. Skips rows that are already terminal, so repeated calls on
    ///    the same subtree are idempotent.
    /// 4. For every transitioned root turn that occupied its thread's
    ///    active-root slot pre-cancel
    ///    ([`TaskStatus::blocks_root_admission`]), commits one terminal
    ///    [`AgentEvent::Cancelled`] marker on that root's **own**
    ///    thread — including child-thread roots reached across
    ///    `SubagentInvocation` linkage — together with a
    ///    `thread_events_available` outbox advisory row. On the
    ///    durable SQL backends both writes share `cancel_tree`'s
    ///    transaction, so the marker exists iff the cancellation
    ///    committed: a crash cannot separate them, an idempotent retry
    ///    (terminal tree, nothing transitioned) emits nothing, and
    ///    cross-host followers are woken by the outbox relay. The
    ///    in-memory backend has no cross-store transaction and gets the
    ///    same guarantee from ordering plus an idempotent append — it
    ///    journals the markers before applying any transition, so a
    ///    cancelled row always has exactly one marker, and a retry after a
    ///    partial sink failure reuses the markers already written instead
    ///    of duplicating them. Its one residual is an orphan marker on a
    ///    still-live root when the caller ABANDONS a failed cancel (see
    ///    [`CancellationMarkerSink`]). The
    ///    marker carries `turn = thread.committed_turns` and the
    ///    cumulative usage from the root's parked continuation when
    ///    present (falling back to the thread aggregate). Cancelling
    ///    a QUEUED root behind a live active root never emits a
    ///    marker — the active root's followers stay open.
    ///
    /// Because every transition is a pure in-memory state change
    /// running under the store's write lock, cancellation is
    /// observable atomically: a worker that reads any row in the
    /// subtree after this call sees the final terminal status, and
    /// any worker whose lease was dropped fails the Phase 2.3
    /// `(worker_id, lease_id)` CAS on its next heartbeat / complete
    /// / fail attempt.
    ///
    /// Returns a [`CancelTreeOutcome`]: the ids actually transitioned
    /// (in transition order, already-terminal rows omitted) plus the
    /// committed `Cancelled` markers so callers can wake same-process
    /// followers through the event notifier.
    ///
    /// # Errors
    /// - `cancel_tree rejected: task ... does not exist` — no row
    ///   with the supplied id.
    /// - Schema errors from [`AgentTask::cancel`] or
    ///   [`AgentTask::validate`].
    /// - Marker commit failures (event journal / outbox / thread
    ///   projection). No cancellation survives one: the durable
    ///   backends roll the transaction back, and the in-memory backend
    ///   commits markers before it applies any transition, so it fails
    ///   with the tree untouched and still cancellable.
    async fn cancel_tree(
        &self,
        root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome>;

    /// Resume a [`TaskStatus::AwaitingConfirmation`] task back to
    /// [`TaskStatus::Pending`], atomically extracting the prepared
    /// listen/execute operation and clearing the typed
    /// [`crate::journal::TaskState::AwaitingConfirmation`] payload with
    /// the status flip.
    ///
    /// Returns `(resumed_task, prepared_operation)` where
    /// `prepared_operation` is the [`ListenExecutionContext`] that was
    /// embedded in the `AwaitingConfirmation` payload (or `None` for
    /// non-listen tools). Both are extracted under a single write
    /// lock so there is no window for a concurrent rejection to
    /// invalidate the snapshot.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `resume rejected: not awaiting confirmation` — if the row is
    ///   in any status other than [`TaskStatus::AwaitingConfirmation`].
    async fn resume_from_confirmation(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)>;

    /// Atomically approve an [`AwaitingConfirmation`](TaskStatus::AwaitingConfirmation)
    /// task and claim it for immediate execution under `(worker, lease)`.
    ///
    /// This closes the race between a transport's "approved" decision
    /// and the background worker pool's next scan. A successful call:
    ///
    /// 1. Verifies that the row exists and is currently
    ///    [`TaskStatus::AwaitingConfirmation`].
    /// 2. Extracts the prepared listen/execute operation from the
    ///    typed confirmation payload.
    /// 3. Flips the row back to [`TaskStatus::Pending`] and immediately
    ///    claims it as [`TaskStatus::Running`] under the supplied
    ///    `(worker, lease)` ownership token.
    /// 4. Persists the claimed row under the same write lock so no
    ///    concurrent scan can observe an intermediate `Pending` row.
    ///
    /// Returns `(claimed_task, prepared_operation)` where
    /// `prepared_operation` is the listen-tier operation staged before
    /// the confirmation pause, if any.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `approve rejected: not awaiting confirmation` — if the row is
    ///   in any status other than [`TaskStatus::AwaitingConfirmation`].
    /// - Schema errors from [`AgentTask::resume_from_confirmation`] or
    ///   [`AgentTask::mark_running`].
    async fn approve_confirmation_and_acquire(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)>;

    /// Reject or time out a [`TaskStatus::AwaitingConfirmation`] task,
    /// transitioning it directly to [`TaskStatus::Failed`] and
    /// recomputing the parent's `pending_child_count` under the same
    /// write lock.
    ///
    /// This is the Phase 5.3 entry point for confirmation decisions
    /// that do **not** lead to execution (rejection, timeout, or
    /// revocation). Unlike [`AgentTaskStore::fail_task`], this method
    /// does not require a `(worker, lease)` CAS because the task has
    /// no lease while in [`TaskStatus::AwaitingConfirmation`].
    ///
    /// The child's [`crate::journal::TaskState::AwaitingConfirmation`] payload
    /// (continuation + prepared operation) is cleared atomically with
    /// the terminal transition, same as every other terminal helper.
    ///
    /// Returns `(child, parent)` where `parent` is the recomputed
    /// parent row (or `None` if the child is a root task). If the
    /// parent was in [`TaskStatus::WaitingOnChildren`] and this was
    /// its last live child, the parent flips to
    /// [`TaskStatus::Pending`] with [`crate::journal::TaskState::ReadyToResume`].
    ///
    /// # Errors
    /// - `reject_confirmation rejected: task ... does not exist`
    /// - `reject_confirmation rejected: task ... is not awaiting
    ///   confirmation`
    /// - Schema errors from [`AgentTask::fail`] or parent recompute.
    async fn reject_confirmation(
        &self,
        id: &AgentTaskId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)>;

    /// Remove every stored task. **Test-only housekeeping** — worker
    /// code must use the CAS-guarded helpers to drive individual
    /// tasks through their lifecycle.
    ///
    /// # Errors
    /// Returns an error if the store cannot be cleared.
    async fn clear(&self) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

/// Output shape of [`Inner::build_batch_invocation_pairs`]: the
/// per-entry `(invocation, child_root)` pairs plus the ordered
/// `child_ids` vector the caller hands to
/// [`AgentTask::wait_on_children`] when transitioning the parent.
type BatchInvocationRows = (Vec<(AgentTask, AgentTask)>, Vec<AgentTaskId>);

#[derive(Default)]
struct Inner {
    /// Primary key index.
    by_id: HashMap<AgentTaskId, AgentTask>,
    /// `(thread_id)` index: every task row scoped to a thread.
    by_thread: HashMap<ThreadId, Vec<AgentTaskId>>,
    /// `(parent_id)` index: direct children of a parent task.
    by_parent: HashMap<AgentTaskId, Vec<AgentTaskId>>,
    /// `(status)` index: rows currently in each lifecycle state.
    by_status: HashMap<TaskStatus, BTreeSet<AgentTaskId>>,
    /// Partial unique index on `(thread_id)` where
    /// `kind = root_turn AND status.blocks_root_admission()`. Holds the
    /// id of whichever root currently occupies the thread's single
    /// active-root slot, if any. Queued roots are **not** stored here —
    /// they live in `queued_roots_by_thread` instead.
    active_root_by_thread: HashMap<ThreadId, AgentTaskId>,
    /// Per-thread FIFO queue of root turns waiting behind the active-root
    /// slot. Entries are stored as `(created_at, id)` so iteration is
    /// deterministic even if two roots share a timestamp. A `BTreeSet`
    /// gives ordered insert / iterate / remove in `O(log n)` without
    /// needing a custom priority queue.
    queued_roots_by_thread: HashMap<ThreadId, BTreeSet<(OffsetDateTime, AgentTaskId)>>,
    /// Phase 2.3 global runnable index: `(created_at, id)` of every row
    /// whose status is [`TaskStatus::Pending`], across all kinds and
    /// threads. This is the scan target for
    /// [`AgentTaskStore::acquire_next_runnable`] and gives
    /// deterministic oldest-first FIFO ordering (tiebroken by `id`) for
    /// the work pool.
    runnable_by_created_at: BTreeSet<(OffsetDateTime, AgentTaskId)>,
    /// Phase 2.3 global lease-expiry index: `(lease_expires_at, id)` of
    /// every row whose status is [`TaskStatus::Running`]. Ordered
    /// ascending so [`AgentTaskStore::release_expired_leases`] can walk
    /// the head of the set until it hits a still-live lease and stop.
    leased_by_expiry: BTreeSet<(OffsetDateTime, AgentTaskId)>,
    /// Phase 10 · E durable idempotency table, keyed by `request_id`.
    /// Mirrors the `agent_sdk_idempotency` row the SQL backends own so
    /// in-memory tests exercise the same dedup contract.
    idempotency: HashMap<String, IdempotencyRecord>,
    /// Phase 7.6 reverse index: `child_root_task_id → invocation task id`
    /// for every live [`TaskKind::Subagent`] invocation. Lets
    /// [`Inner::resume_linked_subagent_invocation`] wake the parked
    /// parent invocation in `O(1)` instead of scanning every row in
    /// `by_id` under the global write lock on every terminal child-root
    /// transition (e.g. each `cancel_tree` fan-out, each fail-closed
    /// sweep). Populated when an invocation row is indexed and pruned
    /// when it is resumed or reaches a terminal state.
    invocation_by_child_root: HashMap<AgentTaskId, AgentTaskId>,
}

/// In-memory reference implementation of [`AgentTaskStore`].
///
/// Designed for tests and single-process usage. The implementation takes
/// the index surface seriously: every required index from the module-level
/// doc is kept in sync on every write, and the partial-unique "one active
/// root per thread" constraint is enforced on both `insert` and `update`.
#[derive(Clone, Default)]
pub struct InMemoryAgentTaskStore {
    inner: Arc<RwLock<Inner>>,
    /// Optional sink for the terminal `Cancelled` markers
    /// [`AgentTaskStore::cancel_tree`] commits per cancelled blocking
    /// root. `None` (bare stores in unit tests) emits no markers; the
    /// composed in-memory backend wires it via
    /// [`Self::with_cancellation_markers`].
    marker_sink: Option<Arc<CancellationMarkerSink>>,
}

impl InMemoryAgentTaskStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Wire the event / outbox / thread stores `cancel_tree` commits
    /// its terminal `Cancelled` markers through (see
    /// [`CancellationMarkerSink`]).
    #[must_use]
    pub fn with_cancellation_markers(mut self, sink: CancellationMarkerSink) -> Self {
        self.marker_sink = Some(Arc::new(sink));
        self
    }

    /// The marker this candidate already carries on its thread, if any.
    ///
    /// Keyed on the marker's emitter — the cancelled root's own task id —
    /// which is exact: a task is cancellable once (`Cancelled` is
    /// absorbing), and a thread that is cancelled, resubmitted, and
    /// cancelled again gets a NEW root with a new id, so its marker is
    /// never mistaken for the old one.
    async fn journaled_marker_for(
        sink: &CancellationMarkerSink,
        candidate: &CancelMarkerCandidate,
    ) -> Result<Option<CommittedEvent>> {
        Ok(sink
            .event_repo
            .get_events(&candidate.thread_id)
            .await
            .with_context(|| {
                format!(
                    "cancel_tree: read thread {} for marker idempotency",
                    candidate.thread_id
                )
            })?
            .into_iter()
            .find(|committed| {
                matches!(committed.event, AgentEvent::Cancelled { .. })
                    && committed.event.emitter_task_id() == Some(candidate.task_id.as_str())
            }))
    }

    /// Commit one terminal `Cancelled` marker (+ its
    /// `thread_events_available` outbox advisory) per candidate thread
    /// through the wired sink. Called by `cancel_tree` before the task
    /// transitions are applied; a missing sink emits nothing.
    ///
    /// The append is IDEMPOTENT per candidate. A previous attempt can
    /// have journaled some of a cascade's markers before failing on a
    /// later one — the flips are all-or-nothing, so those roots stayed
    /// live and are candidates again on the retry. Re-appending would put
    /// a second `Cancelled` event for one effective cancel on the thread,
    /// which replay would show and a usage-summing consumer would
    /// double-count, so an already-journaled marker is REUSED instead:
    /// it is returned to the caller (whose notifier still wakes
    /// same-process followers with it) and not written again.
    async fn commit_cancellation_markers(
        &self,
        candidates: Vec<CancelMarkerCandidate>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        let Some(sink) = &self.marker_sink else {
            return Ok(Vec::new());
        };
        let mut markers = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if let Some(existing) = Self::journaled_marker_for(sink, &candidate).await? {
                // The advisory is written after the marker, so the attempt
                // that journaled this one may have died in between —
                // leaving a marker no relay would ever announce. Ensure it
                // now, or a parked root that commits no further event
                // would leave cross-host followers unwoken even though the
                // cancel succeeded.
                //
                // The reused marker keeps the `turn` / `usage` payload the
                // FIRST attempt computed. If the root committed a turn
                // between the attempts (a cancel racing the unfenced
                // finish line), the surviving marker understates both
                // against the moment the cancel was honored; the task's
                // own row is the authority on how it ended.
                Self::ensure_marker_advisory(sink, &existing, now).await?;
                markers.push(existing);
                continue;
            }
            let (turn, thread_usage) = match sink
                .thread_store
                .get(&candidate.thread_id)
                .await
                .with_context(|| {
                    format!(
                        "cancel_tree: read thread {} for marker",
                        candidate.thread_id
                    )
                })? {
                Some(thread) => (
                    usize::try_from(thread.committed_turns).unwrap_or(0),
                    thread.total_usage,
                ),
                None => (0, TokenUsage::default()),
            };
            let usage = candidate.continuation_usage.unwrap_or(thread_usage);
            let committed = sink
                .event_repo
                .commit_event(
                    &candidate.thread_id,
                    AgentEvent::cancelled(turn, usage)
                        .with_emitter_task_id(candidate.task_id.as_str()),
                    now,
                )
                .await
                .with_context(|| {
                    format!(
                        "cancel_tree: commit cancelled marker on thread {}",
                        candidate.thread_id,
                    )
                })?;
            Self::insert_marker_advisory(sink, &committed, now).await?;
            markers.push(committed);
        }
        Ok(markers)
    }

    /// Write the `thread_events_available` advisory that wakes cross-host
    /// followers of a journaled marker.
    async fn insert_marker_advisory(
        sink: &CancellationMarkerSink,
        committed: &CommittedEvent,
        now: OffsetDateTime,
    ) -> Result<()> {
        let payload_json = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: committed.thread_id.clone(),
            last_sequence: committed.sequence,
        })
        .to_payload_json()
        .context("cancel_tree: serialise marker advisory payload")?;
        sink.outbox_store
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: committed.thread_id.clone(),
                event_id: Some(committed.event_id),
                sequence: Some(committed.sequence),
                payload_json,
                max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                now,
            }])
            .await
            .with_context(|| {
                format!(
                    "cancel_tree: insert marker outbox advisory for thread {}",
                    committed.thread_id,
                )
            })?;
        Ok(())
    }

    /// Give a reused marker an advisory the relay can still publish.
    ///
    /// [`OutboxStore::insert_batch`] is a plain insert with no dedup, so a
    /// blind re-insert would queue a second wake for the same marker. The
    /// advisory is therefore looked up by the marker's `event_id`, which
    /// every `ThreadEventsAvailable` row carries.
    ///
    /// An [`OutboxStatus::Expired`] row does NOT count: it exhausted its
    /// relay attempts, and `claim_pending` only ever selects `Pending`
    /// rows, so it will never be published — leaving it in place would
    /// report a cancellation that no cross-host follower is ever woken
    /// for. Expired is terminal with no path back (`reclaim_expired_claims`
    /// deliberately skips terminal rows), so the repair is a FRESH row
    /// with a fresh attempt budget; nothing keys the outbox by `event_id`,
    /// so the expired row and its replacement coexist. `Delivered` and
    /// `Claimed` rows DO count — the wake was published, or is in flight
    /// and will be retried by the relay's own machinery.
    async fn ensure_marker_advisory(
        sink: &CancellationMarkerSink,
        committed: &CommittedEvent,
        now: OffsetDateTime,
    ) -> Result<()> {
        let existing = sink
            .outbox_store
            .list_by_thread(&committed.thread_id)
            .await
            .with_context(|| {
                format!(
                    "cancel_tree: read outbox for thread {} to reuse a marker advisory",
                    committed.thread_id,
                )
            })?;
        if existing.iter().any(|row| {
            row.kind == OutboxMessageKind::ThreadEventsAvailable
                && row.event_id == Some(committed.event_id)
                && row.status != OutboxStatus::Expired
        }) {
            return Ok(());
        }
        Self::insert_marker_advisory(sink, committed, now).await
    }
}

impl Inner {
    fn add_to_indexes(&mut self, task: &AgentTask) {
        self.by_thread
            .entry(task.thread_id.clone())
            .or_default()
            .push(task.id.clone());

        if let Some(parent_id) = &task.parent_id {
            self.by_parent
                .entry(parent_id.clone())
                .or_default()
                .push(task.id.clone());
        }

        self.by_status
            .entry(task.status)
            .or_default()
            .insert(task.id.clone());

        if task.kind == TaskKind::RootTurn {
            if task.status.blocks_root_admission() {
                self.active_root_by_thread
                    .insert(task.thread_id.clone(), task.id.clone());
            } else if task.status == TaskStatus::Queued {
                self.queued_roots_by_thread
                    .entry(task.thread_id.clone())
                    .or_default()
                    .insert((task.created_at, task.id.clone()));
            }
        }

        // Phase 7.6 reverse index: map the subagent invocation's linked
        // child-thread root id back to the invocation so a terminal
        // child root can wake the parent invocation in O(1).
        if task.kind == TaskKind::Subagent
            && let Some(invocation) = task.state.subagent_invocation()
        {
            self.invocation_by_child_root
                .insert(invocation.child_root_task_id.clone(), task.id.clone());
        }

        // Phase 2.3 global runnable / lease-expiry indexes are
        // status-driven and apply to every kind. A `Pending` row
        // (runnable) and a `Running` row (leased) always carry
        // exactly one entry in the relevant index.
        self.register_runnable_lease_indexes(task);
    }

    /// Add `task` to the `runnable_by_created_at` index if it is
    /// [`TaskStatus::Pending`], or to `leased_by_expiry` if it is
    /// [`TaskStatus::Running`]. Other statuses are ignored.
    ///
    /// The two indexes are disjoint by construction because a row's
    /// status is a single enum variant, and the index maintenance path
    /// always removes the row from both indexes before re-inserting.
    fn register_runnable_lease_indexes(&mut self, task: &AgentTask) {
        match task.status {
            TaskStatus::Pending => {
                self.runnable_by_created_at
                    .insert((task.created_at, task.id.clone()));
            }
            TaskStatus::Running => {
                if let Some(expires_at) = task.lease_expires_at {
                    self.leased_by_expiry.insert((expires_at, task.id.clone()));
                }
            }
            _ => {}
        }
    }

    /// Drop `task` from the runnable and lease-expiry indexes if it was
    /// present. Called before re-registering under a new status so the
    /// two indexes never hold stale entries.
    fn unregister_runnable_lease_indexes(&mut self, task: &AgentTask) {
        if task.status == TaskStatus::Pending {
            self.runnable_by_created_at
                .remove(&(task.created_at, task.id.clone()));
        }
        if task.status == TaskStatus::Running
            && let Some(expires_at) = task.lease_expires_at
        {
            self.leased_by_expiry.remove(&(expires_at, task.id.clone()));
        }
    }

    fn remove_from_status_index(&mut self, task: &AgentTask) {
        if let Some(set) = self.by_status.get_mut(&task.status) {
            set.remove(&task.id);
            if set.is_empty() {
                self.by_status.remove(&task.status);
            }
        }
    }

    fn remove_active_root_if_match(&mut self, task: &AgentTask) {
        if task.kind != TaskKind::RootTurn {
            return;
        }
        if let Some(current) = self.active_root_by_thread.get(&task.thread_id)
            && *current == task.id
        {
            self.active_root_by_thread.remove(&task.thread_id);
        }
    }

    /// Remove a task from the per-thread queued-root FIFO index, if it is
    /// a queued root.
    fn remove_queued_root_if_match(&mut self, task: &AgentTask) {
        if task.kind != TaskKind::RootTurn {
            return;
        }
        if let Some(queue) = self.queued_roots_by_thread.get_mut(&task.thread_id) {
            queue.remove(&(task.created_at, task.id.clone()));
            if queue.is_empty() {
                self.queued_roots_by_thread.remove(&task.thread_id);
            }
        }
    }

    /// Is the thread's active-root slot currently held by a blocking root?
    fn thread_has_blocking_root(&self, thread_id: &ThreadId) -> bool {
        self.active_root_by_thread.contains_key(thread_id)
    }

    /// Number of queued (not active/blocking) root turns on a thread.
    fn queued_root_count(&self, thread_id: &ThreadId) -> u32 {
        self.queued_roots_by_thread
            .get(thread_id)
            .map_or(0, |queue| u32::try_from(queue.len()).unwrap_or(u32::MAX))
    }

    /// Promote the FIFO head of `thread_id`'s queued-root list to
    /// `Pending`, if the thread has no blocking root, under the caller's
    /// existing write lock.
    ///
    /// This is the lock-held core shared by
    /// [`AgentTaskStore::promote_next_queued_root`] and the in-store
    /// terminal-transition paths ([`Self::fail_row_closed`],
    /// [`Self::cancel_row_in_place`], and the root-turn branch of
    /// [`Self::propagate_terminal_child_transition`]). Promoting inside
    /// the same locked scope as the root's terminal transition is what
    /// makes promotion crash-safe: a queued root can never be stranded
    /// behind a root that was failed closed by a recovery sweep or
    /// cancelled by `cancel_tree`.
    fn promote_next_queued_root_locked(
        &mut self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        // If the slot is still held, the active root is retrying or
        // waiting on something and no promotion may fire — retries of
        // the active root must never be overtaken by queued roots.
        if self.thread_has_blocking_root(thread_id) {
            return Ok(None);
        }

        // Pop the FIFO head. `BTreeSet` iterates in ascending key order,
        // and the key is `(created_at, id)` so the earliest submission
        // wins, with `id` breaking ties deterministically.
        let head_key = self
            .queued_roots_by_thread
            .get(thread_id)
            .and_then(|q| q.iter().next().cloned());
        let Some((_, id)) = head_key else {
            return Ok(None);
        };

        let queued_row = self
            .by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("promote rejected: queue head {id} missing from by_id"))?;
        let promoted = queued_row
            .clone()
            .promote_to_pending(now)
            .context("promote rejected: promotion transition failed")?;

        self.rebalance_after_row_change(&queued_row, &promoted);
        self.by_id.insert(id, promoted.clone());
        Ok(Some(promoted))
    }

    /// Rebalance every secondary index after mutating a row from `old`
    /// into `new`. Assumes both rows share the same row invariants
    /// (`id`, `kind`, `parent_id`, `root_id`, `depth`, `thread_id`,
    /// `created_at`, `max_attempts`) — the [`AgentTaskStore::update`]
    /// path checks this up-front, and internal Phase 2.3 call sites
    /// (`try_acquire_task`, `heartbeat_task`, expiry sweeps) only ever
    /// mutate lease fields and status.
    ///
    /// This helper does **not** touch `by_id`; callers must overwrite
    /// the primary key themselves.
    fn rebalance_after_row_change(&mut self, old: &AgentTask, new: &AgentTask) {
        // Status bucket.
        self.remove_from_status_index(old);
        self.by_status
            .entry(new.status)
            .or_default()
            .insert(new.id.clone());

        // Runnable / leased indexes: always drop the old classification
        // first so we never leave a stale entry behind, then re-register
        // under the new classification. A no-op on statuses that don't
        // participate in either index.
        self.unregister_runnable_lease_indexes(old);
        self.register_runnable_lease_indexes(new);

        // Root-turn-specific indexes (active-root slot + queued FIFO).
        // `kind` and `thread_id` are row invariants, so the old row's
        // classification can only mutate between root-turn statuses.
        if new.kind == TaskKind::RootTurn {
            if old.status.blocks_root_admission() {
                self.remove_active_root_if_match(old);
            } else if old.status == TaskStatus::Queued {
                self.remove_queued_root_if_match(old);
            }

            if new.status.blocks_root_admission() {
                self.active_root_by_thread
                    .insert(new.thread_id.clone(), new.id.clone());
            } else if new.status == TaskStatus::Queued {
                self.queued_roots_by_thread
                    .entry(new.thread_id.clone())
                    .or_default()
                    .insert((new.created_at, new.id.clone()));
            }
        }

        // Phase 7.6 reverse index: once an invocation reaches a terminal
        // state its child-root linkage can no longer wake it, so drop the
        // entry to keep the index bounded and free of stale rows.
        if old.kind == TaskKind::Subagent
            && new.status.is_terminal()
            && let Some(invocation) = old.state.subagent_invocation()
        {
            self.invocation_by_child_root
                .remove(&invocation.child_root_task_id);
        }
    }

    /// Phase 2.5 fail-closed transition under the store write lock.
    ///
    /// Runs [`AgentTask::fail_with_reason`] on a clone of `old`,
    /// rebalances every secondary index onto the new `Failed` row,
    /// and overwrites the primary key entry. All three steps happen
    /// atomically under whatever write lock the caller already
    /// holds on `Inner` — `try_acquire_task`,
    /// `acquire_next_runnable`, and `release_expired_leases` all
    /// invoke this helper from inside their existing write-locked
    /// scope so the row is never observed in an intermediate state.
    fn fail_row_closed(
        &mut self,
        old: &AgentTask,
        reason: FailureReason,
        now: OffsetDateTime,
    ) -> Result<()> {
        let failed = old
            .clone()
            .fail_with_reason(reason, now)
            .context("fail_row_closed: fail_with_reason transition failed")?;
        self.rebalance_after_row_change(old, &failed);
        self.by_id.insert(failed.id.clone(), failed.clone());
        let _ = self.propagate_terminal_child_transition(&failed, now, "fail_row_closed")?;
        Ok(())
    }

    /// Count the direct children of `parent_id` whose status is
    /// non-terminal, used by Phase 2.6's `complete_task` /
    /// `fail_task` to recompute the parent's outstanding-child
    /// counter from the journal's `by_parent` index instead of a
    /// caller-maintained running total.
    ///
    /// A row that no longer exists in `by_id` (e.g. because a later
    /// phase introduces a hard-delete) is treated as terminal — the
    /// parent's counter should never count a ghost.
    fn count_live_children(&self, parent_id: &AgentTaskId) -> u32 {
        let Some(children) = self.by_parent.get(parent_id) else {
            return 0;
        };
        let live = children
            .iter()
            .filter_map(|id| self.by_id.get(id))
            .filter(|child| !child.status.is_terminal())
            .count();
        u32::try_from(live).unwrap_or(u32::MAX)
    }

    /// Collect the ids of `root_id` and every descendant in its
    /// subtree in breadth-first order, used by Phase 2.6's
    /// `cancel_tree`.
    ///
    /// The returned vector always starts with `root_id` so the
    /// caller can iterate it linearly and cancel the root last if
    /// the walk order matters — Phase 2.6's sweep cancels in BFS
    /// order (root first) because the cancellation is idempotent
    /// and the order only drives the returned transitioned-id
    /// slice.
    ///
    /// Phase 7.6 extends the walk to follow
    /// [`TaskState::SubagentInvocation`] linkage: when a task
    /// carries durable subagent invocation state the linked
    /// `child_root_task_id` is added to the BFS frontier so
    /// cancellation cascades across thread boundaries into nested
    /// subagent trees.
    ///
    /// Depth is bounded by the actual journal tree, and `by_parent`
    /// is append-only within a root's lifetime, so the walk
    /// terminates naturally without needing a visited set for
    /// single-thread trees — but with Phase 7.6 cross-thread
    /// linkage the `visited` `BTreeSet` is the primary termination
    /// guarantee: it prevents infinite loops if cyclic subagent
    /// references ever appear in a corrupted journal.
    fn collect_subtree(&self, root_id: &AgentTaskId) -> Vec<AgentTaskId> {
        let mut visited: BTreeSet<AgentTaskId> = BTreeSet::new();
        let mut out: Vec<AgentTaskId> = Vec::new();
        let mut frontier: std::collections::VecDeque<AgentTaskId> =
            std::collections::VecDeque::new();
        frontier.push_back(root_id.clone());
        while let Some(id) = frontier.pop_front() {
            if !visited.insert(id.clone()) {
                continue;
            }
            let Some(task) = self.by_id.get(&id) else {
                continue;
            };
            out.push(id.clone());

            // Phase 7.6: follow subagent invocation linkage so
            // cancellation cascades across thread boundaries into
            // child-thread root tasks and their descendants.
            if let Some(invocation) = task.state.subagent_invocation() {
                let child_root = &invocation.child_root_task_id;
                if !visited.contains(child_root) {
                    frontier.push_back(child_root.clone());
                }
            }

            if let Some(children) = self.by_parent.get(&id) {
                for child_id in children {
                    if !visited.contains(child_id) {
                        frontier.push_back(child_id.clone());
                    }
                }
            }
        }
        out
    }

    /// Cancel a single row in place under the store write lock,
    /// rebalancing every secondary index. A no-op on rows that are
    /// already terminal (the caller decides whether that should
    /// show up in the returned transitioned-id slice).
    ///
    /// Returns `true` if the row was actually transitioned.
    fn cancel_row_in_place(&mut self, id: &AgentTaskId, now: OffsetDateTime) -> Result<bool> {
        let Some(old) = self.by_id.get(id).cloned() else {
            return Ok(false);
        };
        if old.status.is_terminal() {
            return Ok(false);
        }
        let cancelled = old
            .clone()
            .cancel(now)
            .context("cancel_tree: cancel transition failed")?;
        self.rebalance_after_row_change(&old, &cancelled);
        self.by_id.insert(cancelled.id.clone(), cancelled.clone());
        let _ = self.propagate_terminal_child_transition(&cancelled, now, "cancel_tree")?;
        Ok(true)
    }

    /// Drive a terminal child transition (complete / fail) and, under
    /// the same write lock, recompute the parent's
    /// `pending_child_count` from the live-children index so the
    /// parent can resume as soon as its last live child reaches a
    /// terminal state.
    ///
    /// `transition` takes the old child row and returns the new
    /// terminal row, so the caller can choose between
    /// [`AgentTask::complete`] and [`AgentTask::fail`] without
    /// duplicating the CAS / rebalance / recompute plumbing.
    fn apply_task_terminal_transition(
        &mut self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
        error_prefix: &'static str,
        transition: impl FnOnce(AgentTask) -> Result<AgentTask, super::task::TaskSchemaError>,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let old_child =
            self.by_id.get(child_id).cloned().ok_or_else(|| {
                anyhow!("{error_prefix} rejected: task {child_id} does not exist")
            })?;

        if old_child.status != TaskStatus::Running {
            let status = old_child.status;
            return Err(anyhow!(
                "{error_prefix} rejected: task {child_id} is not running (status {status:?})"
            ));
        }
        match &old_child.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "{error_prefix} rejected: worker mismatch on task {child_id}"
                ));
            }
        }
        match &old_child.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "{error_prefix} rejected: lease mismatch on task {child_id}"
                ));
            }
        }

        let new_child = transition(old_child.clone())
            .with_context(|| format!("{error_prefix}: terminal transition failed"))?;
        self.rebalance_after_row_change(&old_child, &new_child);
        self.by_id.insert(new_child.id.clone(), new_child.clone());

        let parent = self.propagate_terminal_child_transition(&new_child, now, error_prefix)?;

        Ok((new_child, parent))
    }

    fn propagate_terminal_child_transition(
        &mut self,
        new_child: &AgentTask,
        now: OffsetDateTime,
        error_prefix: &'static str,
    ) -> Result<Option<AgentTask>> {
        // Phase 2.6 recompute: the parent's counter is derived from
        // the `by_parent` index so a double-complete or a dropped
        // complete cannot silently corrupt it. If the parent has
        // already been cancelled out-of-band (e.g. by a
        // `cancel_tree` sweep between the worker's start and
        // finish), we still let the child's terminal transition
        // land but leave the parent alone — the parent has already
        // been moved to a terminal status and must not be reopened.
        if let Some(parent_id) = &new_child.parent_id {
            let Some(old_parent) = self.by_id.get(parent_id).cloned() else {
                return Err(anyhow!(
                    "{error_prefix}: child {} references missing parent {parent_id}",
                    new_child.id
                ));
            };
            if old_parent.status == TaskStatus::WaitingOnChildren {
                let live = self.count_live_children(parent_id);
                let new_parent = old_parent
                    .clone()
                    .recompute_pending_children(live, now)
                    .with_context(|| {
                        format!("{error_prefix}: recompute_pending_children transition failed")
                    })?;
                self.rebalance_after_row_change(&old_parent, &new_parent);
                self.by_id.insert(new_parent.id.clone(), new_parent.clone());
                Ok(Some(new_parent))
            } else {
                Ok(Some(old_parent))
            }
        } else if new_child.kind == TaskKind::RootTurn && new_child.is_root() {
            // A terminal root turn frees the thread's active-root slot.
            // Promote the next queued FIFO head in the SAME locked scope
            // as the terminal transition so a queued root can never be
            // stranded by a store-side terminal transition — recovery
            // fail-close (`fail_row_closed`) or `cancel_tree`
            // (`cancel_row_in_place`) — not only the host's post-commit
            // promotion path. Promotion is idempotent, so the host's
            // later `promote_next_queued_root` becomes a no-op.
            let _promoted = self.promote_next_queued_root_locked(&new_child.thread_id, now)?;
            self.resume_linked_subagent_invocation(new_child, now, error_prefix)
        } else {
            Ok(None)
        }
    }

    fn resume_linked_subagent_invocation(
        &mut self,
        child_root: &AgentTask,
        now: OffsetDateTime,
        error_prefix: &'static str,
    ) -> Result<Option<AgentTask>> {
        // O(1) reverse-index lookup instead of scanning every row in
        // `by_id` under the global write lock.
        let Some(invocation_id) = self.invocation_by_child_root.get(&child_root.id).cloned() else {
            return Ok(None);
        };
        let Some(old_invocation) = self.by_id.get(&invocation_id).cloned() else {
            // Stale index entry (the invocation row was removed); prune
            // it and treat as no linked invocation.
            self.invocation_by_child_root.remove(&child_root.id);
            return Ok(None);
        };
        // Verify the linkage still holds and the invocation is still
        // parked — the index is a hint, the row state is authoritative.
        if old_invocation.kind != TaskKind::Subagent
            || old_invocation.status != TaskStatus::WaitingOnChildren
            || old_invocation
                .state
                .subagent_invocation()
                .is_none_or(|invocation| invocation.child_root_task_id != child_root.id)
        {
            return Ok(None);
        }

        let new_invocation = old_invocation
            .clone()
            .recompute_pending_children(0, now)
            .with_context(|| {
                format!("{error_prefix}: subagent invocation resume transition failed")
            })?;
        self.rebalance_after_row_change(&old_invocation, &new_invocation);
        self.by_id
            .insert(new_invocation.id.clone(), new_invocation.clone());
        // The invocation has been woken; the child-root linkage is
        // consumed, so drop the reverse-index entry.
        self.invocation_by_child_root.remove(&child_root.id);
        Ok(Some(new_invocation))
    }

    // ─── Spawn-batch helpers (Phase 7.2 fan-out) ─────────────────────

    /// CAS-validate that `parent_id` is currently a `Running`, non-leaf
    /// task held by `(worker, lease)`. Returns the parent row on
    /// success — callers that intend to mutate the parent must hold
    /// the inner write lock and reuse this snapshot.
    ///
    /// Centralizes the "spawn rejected: …" prelude shared by
    /// [`AgentTaskStore::spawn_tool_children`],
    /// [`AgentTaskStore::spawn_subagent_invocation`], and
    /// [`AgentTaskStore::spawn_subagent_batch`] so the validation
    /// surface stays in one place. Error messages are stable for
    /// the `spawn_*_rejects_wrong_worker_or_lease` tests.
    fn validate_running_owned_non_leaf_parent(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
    ) -> Result<AgentTask> {
        let old_parent = self
            .by_id
            .get(parent_id)
            .cloned()
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        if old_parent.status != TaskStatus::Running {
            let status = old_parent.status;
            return Err(anyhow!(
                "spawn rejected: task {parent_id} is not running (status {status:?})"
            ));
        }
        match &old_parent.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: worker mismatch on task {parent_id}"
                ));
            }
        }
        match &old_parent.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: lease mismatch on task {parent_id}"
                ));
            }
        }
        if old_parent.kind.is_leaf() {
            let parent_kind = old_parent.kind;
            return Err(anyhow!(
                "spawn rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
            ));
        }
        Ok(old_parent)
    }

    /// Reject a fan-out batch when:
    /// - any two entries point at the same `child_thread_id`, or
    /// - any entry's `child_thread_id` already has tasks indexed under it.
    ///
    /// Both checks must pass before the impl materializes any
    /// invocation/child rows so a partial materialization can never
    /// orphan child thread state.
    fn validate_batch_thread_uniqueness(&self, spawns: &[SubagentInvocationSpawn]) -> Result<()> {
        let mut seen_thread_ids: std::collections::HashSet<&ThreadId> =
            std::collections::HashSet::with_capacity(spawns.len());
        for spawn in spawns {
            if !seen_thread_ids.insert(&spawn.child_thread_id) {
                return Err(anyhow!(
                    "spawn rejected: duplicate child_thread_id {} in batch",
                    spawn.child_thread_id
                ));
            }
        }

        for spawn in spawns {
            if self.by_thread.contains_key(&spawn.child_thread_id)
                || self
                    .active_root_by_thread
                    .contains_key(&spawn.child_thread_id)
            {
                return Err(anyhow!(
                    "spawn rejected: child thread id {} already has tasks",
                    spawn.child_thread_id
                ));
            }
        }
        Ok(())
    }

    /// Materialize the per-entry `(invocation, child_root)` pair for
    /// every spawn in the batch.
    ///
    /// Mutates **nothing** in `self`: returns the prepared rows plus
    /// the ordered child id list so the caller can transition the
    /// parent atomically. A schema error on entry K therefore leaves
    /// the store untouched.
    fn build_batch_invocation_pairs(
        &self,
        old_parent: &AgentTask,
        spawns: Vec<SubagentInvocationSpawn>,
        now: OffsetDateTime,
    ) -> Result<BatchInvocationRows> {
        let mut prepared: Vec<(AgentTask, AgentTask)> = Vec::with_capacity(spawns.len());
        let mut child_ids: Vec<AgentTaskId> = Vec::with_capacity(spawns.len());
        for spawn in spawns {
            let SubagentInvocationSpawn {
                child_thread_id,
                spec,
                child_root_input,
                spawn_index,
                payload: _per_entry_payload,
                child_caller_metadata,
            } = spawn;
            let child_root = AgentTask::new_root_turn_with_optional_caller(
                child_thread_id.clone(),
                child_root_input,
                child_caller_metadata,
                now,
                RuntimePolicy::server_default().max_attempts,
            );
            if self.by_id.contains_key(&child_root.id) {
                let id = &child_root.id;
                return Err(anyhow!(
                    "spawn rejected: child root task id {id} already exists"
                ));
            }

            let invocation = AgentTask::new_subagent_invocation(
                old_parent,
                SubagentInvocationState {
                    spec,
                    child_thread_id,
                    child_root_task_id: child_root.id.clone(),
                },
                spawn_index,
                now,
                AgentTask::DEFAULT_MAX_ATTEMPTS,
            )
            .context("spawn rejected: new_subagent_invocation failed")?;
            if self.by_id.contains_key(&invocation.id) {
                let id = &invocation.id;
                return Err(anyhow!(
                    "spawn rejected: invocation task id {id} already exists"
                ));
            }
            child_ids.push(invocation.id.clone());
            prepared.push((invocation, child_root));
        }
        Ok((prepared, child_ids))
    }

    /// Materialize the tool-runtime child row for every entry in a
    /// mixed batch.
    ///
    /// Mutates **nothing** in `self` — same contract as
    /// [`Self::build_batch_invocation_pairs`], so a collision on entry K
    /// leaves the store untouched.
    fn build_mixed_tool_children(
        &self,
        old_parent: &AgentTask,
        tool_children: &[ToolChildSpawn],
        child_otel_traceparent: Option<&str>,
        now: OffsetDateTime,
    ) -> Result<Vec<AgentTask>> {
        let mut children: Vec<AgentTask> = Vec::with_capacity(tool_children.len());
        for tool in tool_children {
            let child = new_mixed_tool_child(old_parent, tool, child_otel_traceparent, now)?;
            if self.by_id.contains_key(&child.id)
                || children.iter().any(|existing| existing.id == child.id)
            {
                let id = &child.id;
                return Err(anyhow!("spawn rejected: child id {id} already exists"));
            }
            children.push(child);
        }
        Ok(children)
    }
}

#[async_trait]
impl AgentTaskStore for InMemoryAgentTaskStore {
    async fn insert(&self, task: AgentTask) -> Result<()> {
        task.validate()
            .context("insert rejected: task failed schema validation")?;

        let mut inner = self.inner.write().await;

        if inner.by_id.contains_key(&task.id) {
            return Err(anyhow!(
                "insert rejected: task id {} already exists",
                task.id
            ));
        }

        // Cross-row invariants for non-root tasks: the parent must already
        // exist in the store, and the child's `thread_id`, `root_id`, and
        // `depth` must match the parent row. These are enforced here (and
        // not by `AgentTask::validate`) because `validate` has no access to
        // the parent. Bypassing these checks would let a mutated or
        // deserialized child slip into the wrong `by_thread` / `by_parent`
        // bucket and silently corrupt list queries and the active-root
        // invariant.
        if let Some(parent_id) = &task.parent_id {
            let parent = inner.by_id.get(parent_id).ok_or_else(|| {
                anyhow!("insert rejected: child task references unknown parent {parent_id}")
            })?;
            if parent.kind.is_leaf() {
                let parent_kind = parent.kind;
                return Err(anyhow!(
                    "insert rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
                ));
            }
            if parent.thread_id != task.thread_id {
                let child_thread = &task.thread_id;
                let parent_thread = &parent.thread_id;
                return Err(anyhow!(
                    "insert rejected: child thread_id {child_thread} does not match parent thread_id {parent_thread}"
                ));
            }
            if parent.root_id != task.root_id {
                let child_root = &task.root_id;
                let parent_root = &parent.root_id;
                return Err(anyhow!(
                    "insert rejected: child root_id {child_root} does not match parent root_id {parent_root}"
                ));
            }
            let expected_depth = parent.depth.saturating_add(1);
            if task.depth != expected_depth {
                let child_depth = task.depth;
                let parent_depth = parent.depth;
                return Err(anyhow!(
                    "insert rejected: child depth {child_depth} must be parent.depth + 1 ({parent_depth} + 1 = {expected_depth})"
                ));
            }
        }

        // Partial-unique: one *blocking* root per thread.
        //
        // `blocks_root_admission()` excludes [`TaskStatus::Queued`], so a
        // fresh root turn may be inserted in [`TaskStatus::Queued`] even
        // when another root already holds the slot — that is how
        // [`AgentTaskStore::submit_root_turn`] persists queued
        // submissions. A caller that bypasses `submit_root_turn` and
        // tries to insert a *blocking* root while the slot is already
        // held is rejected here.
        if task.kind == TaskKind::RootTurn
            && task.status.blocks_root_admission()
            && let Some(existing) = inner.active_root_by_thread.get(&task.thread_id)
        {
            let thread_id = &task.thread_id;
            return Err(anyhow!(
                "insert rejected: thread {thread_id} already has active root task {existing}"
            ));
        }

        inner.add_to_indexes(&task);
        inner.by_id.insert(task.id.clone(), task);
        drop(inner);
        Ok(())
    }

    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
        let inner = self.inner.read().await;
        let result = inner.by_id.get(id).cloned();
        drop(inner);
        Ok(result)
    }

    async fn update(&self, task: AgentTask) -> Result<()> {
        task.validate()
            .context("update rejected: task failed schema validation")?;

        let mut inner = self.inner.write().await;

        let old = inner.by_id.get(&task.id).cloned().ok_or_else(|| {
            let id = &task.id;
            anyhow!("update rejected: task {id} does not exist")
        })?;

        // Row invariants: `kind`, `parent_id`, `root_id`, `depth`, `thread_id`,
        // `created_at`, and `max_attempts` are fixed at construction time and
        // must never change across updates. Silently accepting a mutation here
        // would corrupt the secondary indexes (`by_thread`, `by_parent`,
        // `active_root_by_thread`) because those are keyed by the old values.
        if old.kind != task.kind {
            let old_kind = old.kind;
            let new_kind = task.kind;
            return Err(anyhow!(
                "update rejected: task kind is immutable (was {old_kind:?}, got {new_kind:?})"
            ));
        }
        if old.parent_id != task.parent_id {
            let old_parent = &old.parent_id;
            let new_parent = &task.parent_id;
            return Err(anyhow!(
                "update rejected: parent_id is immutable (was {old_parent:?}, got {new_parent:?})"
            ));
        }
        if old.root_id != task.root_id {
            let old_root = &old.root_id;
            let new_root = &task.root_id;
            return Err(anyhow!(
                "update rejected: root_id is immutable (was {old_root}, got {new_root})"
            ));
        }
        if old.depth != task.depth {
            let old_depth = old.depth;
            let new_depth = task.depth;
            return Err(anyhow!(
                "update rejected: depth is immutable (was {old_depth}, got {new_depth})"
            ));
        }
        if old.thread_id != task.thread_id {
            let old_thread = &old.thread_id;
            let new_thread = &task.thread_id;
            return Err(anyhow!(
                "update rejected: thread_id is immutable (was {old_thread}, got {new_thread})"
            ));
        }
        if old.created_at != task.created_at {
            return Err(anyhow!("update rejected: created_at is immutable"));
        }
        if old.max_attempts != task.max_attempts {
            let old_max = old.max_attempts;
            let new_max = task.max_attempts;
            return Err(anyhow!(
                "update rejected: max_attempts is immutable (was {old_max}, got {new_max})"
            ));
        }

        // Partial-unique check: if the new row is a blocking root, make
        // sure no *other* row already holds the active-root slot on the
        // same thread. Queued roots skip this check because they don't
        // occupy the slot.
        if task.kind == TaskKind::RootTurn
            && task.status.blocks_root_admission()
            && let Some(current) = inner.active_root_by_thread.get(&task.thread_id)
            && *current != task.id
        {
            let thread_id = &task.thread_id;
            return Err(anyhow!(
                "update rejected: thread {thread_id} already has a different active root task {current}"
            ));
        }

        // Thread / parent indexes are append-only and keyed by `id`, so the
        // old entries stay in place and still point at the (now-updated)
        // row in `by_id`. This is safe because `thread_id` / `parent_id` are
        // row invariants enforced above and never mutate across updates.
        //
        // Everything else — status bucket, runnable/lease indexes, and the
        // per-thread root indexes — is handled by the shared rebalance
        // helper so that Phase 2.3's acquire / heartbeat / sweep call
        // sites can share the same logic with `update()`.
        inner.rebalance_after_row_change(&old, &task);
        inner.by_id.insert(task.id.clone(), task);
        drop(inner);
        Ok(())
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_thread
            .get(thread_id)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_parent
            .get(parent_id)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_status
            .get(&status)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
        let inner = self.inner.read().await;
        let result = inner
            .active_root_by_thread
            .get(thread_id)
            .and_then(|id| inner.by_id.get(id).cloned());
        drop(inner);
        Ok(result)
    }

    async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
        // Shape gate: only Phase 2.2 root admission flows through here,
        // and only with freshly-constructed rows. Everything else must
        // go through `insert` / `update` and pays for the tighter
        // per-call invariants there.
        validate_submit_root_shape(&task)?;

        let mut inner = self.inner.write().await;

        if inner.by_id.contains_key(&task.id) {
            let id = &task.id;
            return Err(anyhow!(
                "submit_root_turn rejected: task id {id} already exists"
            ));
        }

        // Decide the admission target atomically under the write lock so
        // two concurrent submissions on the same thread always serialize
        // and end up in a deterministic FIFO order.  We must check both
        // the active-root slot *and* the queued-roots index: a new
        // submission arriving between a root completing and
        // `promote_next_queued_root` being called would otherwise skip
        // the queue and claim the active slot, violating FIFO.
        let admitted = if inner.thread_has_blocking_root(&task.thread_id)
            || inner.queued_roots_by_thread.contains_key(&task.thread_id)
        {
            // Convert the incoming row from `Pending` to `Queued` so it
            // lives on the same-thread FIFO queue behind the blocking
            // root. Pass `created_at` as `now` so the queued row's
            // `updated_at` equals its `created_at`, preserving the
            // original submission timestamp in the audit field rather
            // than recording the queuing-decision time. FIFO ordering is
            // unaffected by this choice — the queue index is keyed on
            // `(created_at, id)` and `created_at` is immutable.
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")?
        } else {
            task
        };

        inner.add_to_indexes(&admitted);
        inner.by_id.insert(admitted.id.clone(), admitted.clone());
        drop(inner);
        Ok(admitted)
    }

    async fn submit_root_turn_idempotent(
        &self,
        params: SubmitRootTurnParams,
    ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
        let SubmitRootTurnParams {
            task,
            idempotency,
            max_queued_depth,
        } = params;
        validate_submit_root_shape(&task)?;

        // Everything below runs under a single write lock so the
        // idempotency claim, the queue-depth check, and the admission
        // insert are atomic — a concurrent retry on the same key cannot
        // slip between the lookup and the insert.
        let mut inner = self.inner.write().await;

        // 1. Idempotency replay / conflict, checked before any write.
        if let Some(claim) = &idempotency
            && let Some(record) = inner.idempotency.get(&claim.request_id)
        {
            if record.kind != IdempotencyKind::SubmitWork || record.fingerprint != claim.fingerprint
            {
                return Err(SubmitRootTurnError::IdempotencyConflict);
            }
            let recorded = record.clone();
            let task_id = submit_record_task_id(&recorded)?;
            let admitted = inner
                .by_id
                .get(&task_id)
                .cloned()
                .ok_or_else(|| anyhow!("idempotent submit record points at missing task"))?;
            let queued_depth = inner.queued_root_count(&admitted.thread_id);
            drop(inner);
            return Ok(SubmitRootTurnOutcome {
                task: admitted,
                replayed: true,
                replayed_result: Some(recorded.result_json),
                queued_depth,
            });
        }

        if inner.by_id.contains_key(&task.id) {
            let id = &task.id;
            return Err(SubmitRootTurnError::Other(anyhow!(
                "submit_root_turn rejected: task id {id} already exists"
            )));
        }

        let has_blocking = inner.thread_has_blocking_root(&task.thread_id);
        let current_queued = inner.queued_root_count(&task.thread_id);

        // 2. Back-pressure: a submission that would be *queued* (the
        // thread already holds the active slot or has queued roots) and
        // would push the queued depth past the cap is rejected before
        // anything is written. A submission that lands in the active
        // slot (no blocking root, no queued roots) never counts against
        // the cap.
        let would_queue = has_blocking || current_queued > 0;
        if let Some(cap) = max_queued_depth
            && would_queue
            && current_queued >= cap
        {
            return Err(SubmitRootTurnError::QueueDepthExceeded {
                cap,
                current_depth: current_queued,
            });
        }

        let admitted = if would_queue {
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")?
        } else {
            task
        };

        inner.add_to_indexes(&admitted);
        inner.by_id.insert(admitted.id.clone(), admitted.clone());

        // 3. Claim the idempotency key in the same critical section.
        if let Some(claim) = idempotency {
            inner.idempotency.insert(
                claim.request_id.clone(),
                IdempotencyRecord {
                    request_id: claim.request_id,
                    kind: IdempotencyKind::SubmitWork,
                    fingerprint: claim.fingerprint,
                    result_json: claim.result_json,
                },
            );
        }

        let queued_depth = inner.queued_root_count(&admitted.thread_id);
        drop(inner);
        Ok(SubmitRootTurnOutcome {
            task: admitted,
            replayed: false,
            replayed_result: None,
            queued_depth,
        })
    }

    async fn claim_idempotency(
        &self,
        request_id: &str,
        kind: IdempotencyKind,
        fingerprint: &[u8],
    ) -> Result<IdempotencyClaim> {
        // Write lock: the existence check and the placeholder insert must
        // be one atomic step so two concurrent retries on the same key
        // cannot both observe `Fresh` and both run the effect.
        let mut inner = self.inner.write().await;
        let existing = inner.idempotency.get(request_id).cloned();
        let claim = match existing {
            None => {
                // Reserve the key with a placeholder. `JSON null` marks
                // "claimed, effect in flight, result not yet recorded".
                inner.idempotency.insert(
                    request_id.to_owned(),
                    IdempotencyRecord {
                        request_id: request_id.to_owned(),
                        kind,
                        fingerprint: fingerprint.to_vec(),
                        result_json: serde_json::Value::Null,
                    },
                );
                IdempotencyClaim::Fresh
            }
            Some(record) if record.kind == kind && record.fingerprint == fingerprint => {
                if record.result_json.is_null() {
                    // Reserved by a concurrent in-flight claim; fail closed.
                    IdempotencyClaim::Conflict
                } else {
                    IdempotencyClaim::Replay(Box::new(record))
                }
            }
            Some(_) => IdempotencyClaim::Conflict,
        };
        drop(inner);
        Ok(claim)
    }

    async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
        let mut inner = self.inner.write().await;
        // Check placeholder state with a short-lived immutable borrow so
        // the `insert` below doesn't overlap a `get_mut` borrow.
        let existing_is_placeholder = inner
            .idempotency
            .get(&record.request_id)
            .map(|existing| existing.result_json.is_null());
        match existing_is_placeholder {
            // Fill the reservation placeholder written by `claim`.
            Some(true) => {
                if let Some(existing) = inner.idempotency.get_mut(&record.request_id) {
                    existing.result_json = record.result_json;
                }
            }
            // A real result is already recorded — idempotent no-op so a
            // racing retry cannot clobber it.
            Some(false) => {}
            // No prior claim (defensive): insert the full record.
            None => {
                inner.idempotency.insert(record.request_id.clone(), record);
            }
        }
        drop(inner);
        Ok(())
    }

    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .queued_roots_by_thread
            .get(thread_id)
            .into_iter()
            .flatten()
            .filter_map(|(_, id)| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn promote_next_queued_root(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut inner = self.inner.write().await;
        let promoted = inner.promote_next_queued_root_locked(thread_id, now)?;
        drop(inner);
        Ok(promoted)
    }

    async fn try_acquire_task(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut inner = self.inner.write().await;
        let Some(old) = inner.by_id.get(id).cloned() else {
            return Ok(None);
        };
        // CAS guard: only `Pending` rows are runnable. Every other
        // status (`Queued`, waiting states, `Running`, terminal states)
        // silently returns `None` so callers can loop on scan-and-claim
        // without caring about lost races.
        if !old.status.can_be_leased() {
            return Ok(None);
        }
        // Phase 2.5: consult the recovery matrix before transitioning
        // the row to `Running`. An exhausted-budget row is failed
        // closed under the same write lock so the caller observes the
        // same `Ok(None)` it would see for any other non-runnable row
        // and the worker never burns a guaranteed-to-fail attempt.
        match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
            RecoveryAction::NoAction => {}
            RecoveryAction::FailClosed(reason) => {
                inner.fail_row_closed(&old, reason, now)?;
                return Ok(None);
            }
            RecoveryAction::Requeue => {
                // Acquisition-time classification never produces
                // `Requeue` — only the expiry sweep does — but the
                // match must still be exhaustive, so we surface an
                // explicit bookkeeping error rather than silently
                // mis-routing the transition.
                return Err(anyhow!(
                    "try_acquire_task: recovery matrix produced Requeue for acquisition-time row {id}",
                ));
            }
        }
        let claimed = old
            .clone()
            .mark_running(worker, lease, expires_at, now)
            .context("try_acquire_task rejected: mark_running transition failed")?;
        inner.rebalance_after_row_change(&old, &claimed);
        inner.by_id.insert(claimed.id.clone(), claimed.clone());
        drop(inner);
        Ok(Some(claimed))
    }

    async fn acquire_next_runnable(
        &self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        // Phase 2.5: scan the oldest runnable rows in FIFO order and
        // classify each head before claiming it. A single exhausted
        // head must never poison the scan — if the classifier returns
        // `FailClosed`, we fail the row closed in place and loop to
        // the next head. The loop terminates when either the runnable
        // pool drains (we return `None`) or a head classifies as
        // `NoAction` (we claim it).
        //
        // `BTreeSet` iterates ascending, so `(created_at, id)` pulls
        // the FIFO head. The index is populated across every kind
        // and thread, so this single scan covers root turns and
        // tool-runtime children — workers never need to double-check
        // the kind and never accidentally lease a waiting / terminal
        // / queued row (those statuses are absent from the index by
        // construction).
        let mut inner = self.inner.write().await;
        let result = loop {
            let Some((_, id)) = inner.runnable_by_created_at.iter().next().cloned() else {
                break None;
            };
            let old = inner.by_id.get(&id).cloned().ok_or_else(|| {
                anyhow!("acquire_next_runnable: runnable head {id} missing from by_id")
            })?;
            // Defense in depth: the index should only contain
            // `Pending` rows. A non-runnable row here is an internal
            // bookkeeping bug.
            if !old.status.can_be_leased() {
                let status = old.status;
                return Err(anyhow!(
                    "acquire_next_runnable: runnable index held non-pending row {id} in status {status:?}"
                ));
            }

            match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
                RecoveryAction::NoAction => {
                    let claimed = old
                        .clone()
                        .mark_running(worker, lease, expires_at, now)
                        .context(
                            "acquire_next_runnable rejected: mark_running transition failed",
                        )?;
                    inner.rebalance_after_row_change(&old, &claimed);
                    inner.by_id.insert(claimed.id.clone(), claimed.clone());
                    break Some(claimed);
                }
                RecoveryAction::FailClosed(reason) => {
                    // Fail the exhausted head closed and keep
                    // scanning so one bad row can never poison the
                    // whole worker pool. The fail-closed transition
                    // removes the row from the runnable index, so
                    // the next loop iteration picks up a fresh head.
                    inner.fail_row_closed(&old, reason, now)?;
                }
                RecoveryAction::Requeue => {
                    return Err(anyhow!(
                        "acquire_next_runnable: recovery matrix produced Requeue for acquisition-time row {id}",
                    ));
                }
            }
        };
        drop(inner);
        Ok(result)
    }

    async fn heartbeat_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        expires_at: OffsetDateTime,
        activity: Option<OffsetDateTime>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("heartbeat rejected: task {id} does not exist"))?;

        // Drive the row through the pure transition helper: it enforces
        // `status == Running`, the worker CAS, and the lease CAS, and
        // bumps `last_heartbeat_at` / `lease_expires_at` together, moving
        // `last_activity_at` forward only when this beat carried work.
        let mut refreshed = old.clone();
        refreshed
            .touch_heartbeat(worker, lease, expires_at, activity, now)
            .context("heartbeat rejected")?;

        inner.rebalance_after_row_change(&old, &refreshed);
        inner.by_id.insert(refreshed.id.clone(), refreshed.clone());
        drop(inner);
        Ok(refreshed)
    }

    async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>> {
        let mut inner = self.inner.write().await;

        // Walk the lease-expiry index in ascending order, collecting
        // rows whose expiry is `<= now`. Iteration stops on the first
        // still-live lease, so the sweep cost is O(expired) — and is
        // capped at [`LEASE_RELEASE_BATCH`] rows per call, matching the
        // SQL backends' `LIMIT`, so the drain contract ("loop until a
        // call returns fewer than the batch") holds on every backend.
        // Collecting up-front lets us mutate `inner` inside the loop
        // without fighting the borrow checker.
        let expired_keys: Vec<(OffsetDateTime, AgentTaskId)> = inner
            .leased_by_expiry
            .iter()
            .take_while(|(expires_at, _)| *expires_at <= now)
            .take(LEASE_RELEASE_BATCH)
            .cloned()
            .collect();

        let mut released = Vec::with_capacity(expired_keys.len());
        for (_, id) in expired_keys {
            let old = inner.by_id.get(&id).cloned().ok_or_else(|| {
                anyhow!("release_expired_leases: leased index held missing row {id}")
            })?;
            // Defense in depth: only `Running` rows belong in the
            // expiry index. A non-Running row here is an internal
            // bookkeeping bug.
            if old.status != TaskStatus::Running {
                let status = old.status;
                return Err(anyhow!(
                    "release_expired_leases: expiry index held non-running row {id} in status {status:?}"
                ));
            }

            // Phase 2.5: the recovery matrix chooses requeue vs
            // fail-closed per row. Budget-exhausted rows and rows
            // carrying an unsafe prepared operation take the fail
            // path; everything else requeues with the old Phase 2.3
            // behavior.
            let record = match classify_recovery(&old, RecoveryContext::ExpiredLease) {
                RecoveryAction::Requeue => {
                    let released_row = old
                        .clone()
                        .release_lease(now)
                        .context("release_expired_leases: release transition failed")?;
                    inner.rebalance_after_row_change(&old, &released_row);
                    inner.by_id.insert(id.clone(), released_row);
                    RecoveryRecord::requeued(id)
                }
                RecoveryAction::FailClosed(reason) => {
                    inner.fail_row_closed(&old, reason, now)?;
                    RecoveryRecord::failed_closed(id, reason)
                }
                RecoveryAction::NoAction => {
                    // The expiry sweep should only ever classify
                    // running rows, and the matrix deliberately maps
                    // every running row to either Requeue or
                    // FailClosed. A NoAction here would mean the
                    // classifier drifted from the call site; we
                    // surface the bug instead of silently leaving a
                    // lease stuck.
                    return Err(anyhow!(
                        "release_expired_leases: recovery matrix produced NoAction for expired row {id}",
                    ));
                }
            };
            released.push(record);
        }

        drop(inner);

        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_lease_expiry_outcomes(&released);

        Ok(released)
    }

    async fn requeue_owned_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        boundary: Option<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<RequeueOutcome> {
        let mut inner = self.inner.write().await;
        let Some(old) = inner.by_id.get(id).cloned() else {
            return Ok(RequeueOutcome::NotOwned);
        };
        let still_owned = old.status == TaskStatus::Running
            && old.worker_id.as_ref() == Some(worker)
            && old.lease_id.as_ref() == Some(lease);
        if !still_owned {
            return Ok(RequeueOutcome::NotOwned);
        }
        if old.is_budget_exhausted() {
            return Ok(RequeueOutcome::BudgetExhausted);
        }
        // Boundary first, while STILL holding the write lock (the
        // cancel_tree marker pattern): no acquire can interleave until
        // the lock drops, so the event provably precedes anything a
        // successor emits, and a failure here leaves the row Running
        // under the caller's lease (nothing released yet).
        if let Some(event) = boundary {
            if let Some(sink) = &self.marker_sink {
                let committed = sink
                    .event_repo
                    .commit_event(&old.thread_id, event, now)
                    .await
                    .context("requeue_owned_task: commit boundary event")?;
                let payload_json =
                    OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                        thread_id: committed.thread_id.clone(),
                        last_sequence: committed.sequence,
                    })
                    .to_payload_json()
                    .context("requeue_owned_task: serialise boundary advisory payload")?;
                sink.outbox_store
                    .insert_batch(vec![NewOutboxRow {
                        kind: OutboxMessageKind::ThreadEventsAvailable,
                        thread_id: committed.thread_id.clone(),
                        event_id: Some(committed.event_id),
                        sequence: Some(committed.sequence),
                        payload_json,
                        max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                        now,
                    }])
                    .await
                    .context("requeue_owned_task: insert boundary outbox advisory")?;
            } else {
                log::warn!(
                    "requeue_owned_task: no marker sink wired; requeueing task {id} \
                     without its boundary event (test-support store)",
                );
            }
        }
        let released_row = old
            .clone()
            .release_lease(now)
            .context("requeue_owned_task: release transition failed")?;
        inner.rebalance_after_row_change(&old, &released_row);
        inner.by_id.insert(id.clone(), released_row.clone());
        drop(inner);
        Ok(RequeueOutcome::Requeued(Box::new(released_row)))
    }

    async fn pause_on_children(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        child_count: u32,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("pause rejected: task {id} does not exist"))?;

        // CAS guard: only the worker that holds the lease may pause.
        // Status check first so the error message is friendliest when
        // the row is in the wrong state for any reason.
        if old.status != TaskStatus::Running {
            let status = old.status;
            return Err(anyhow!(
                "pause rejected: task {id} is not running (status {status:?})"
            ));
        }
        match &old.worker_id {
            Some(current) if current == worker => {}
            _ => return Err(anyhow!("pause rejected: worker mismatch on task {id}")),
        }
        match &old.lease_id {
            Some(current) if current == lease => {}
            _ => return Err(anyhow!("pause rejected: lease mismatch on task {id}")),
        }

        let paused = old
            .clone()
            .wait_on_children(child_count, payload, Vec::new(), now)
            .context("pause rejected: wait_on_children transition failed")?;
        inner.rebalance_after_row_change(&old, &paused);
        inner.by_id.insert(paused.id.clone(), paused.clone());
        drop(inner);
        Ok(paused)
    }

    async fn enqueue_steering_resume(
        &self,
        parent_id: &AgentTaskId,
        steering: Vec<agent_sdk_foundation::llm::ContentBlock>,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        // A steering wake is distinguished from an ordinary fan-in only
        // by a non-empty steering payload (both share the ReadyToResume
        // kind). An empty payload would create a ReadyToResume whose
        // children are still running — a broken fan-in — so reject it.
        if steering.is_empty() {
            return Err(anyhow!(
                "steering wake rejected: steering content must be non-empty"
            ));
        }
        let mut inner = self.inner.write().await;
        let old =
            inner.by_id.get(parent_id).cloned().ok_or_else(|| {
                anyhow!("steering wake rejected: task {parent_id} does not exist")
            })?;

        // Idempotent CAS: only a parked parent can be woken. Any other
        // status means the parent already left the WaitingOnChildren
        // state (a concurrent last-child fan-in reached ReadyToResume,
        // a prior wake is mid-exchange, or the row is terminal) — the
        // sweep declines rather than racing.
        if old.status != TaskStatus::WaitingOnChildren {
            drop(inner);
            return Ok(None);
        }

        let woken = old
            .clone()
            .begin_steering_resume(steering, now)
            .context("steering wake rejected: begin_steering_resume transition failed")?;
        inner.rebalance_after_row_change(&old, &woken);
        inner.by_id.insert(woken.id.clone(), woken.clone());
        drop(inner);
        Ok(Some(woken))
    }

    async fn repark_after_steering(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(parent_id)
            .cloned()
            .ok_or_else(|| anyhow!("repark rejected: task {parent_id} does not exist"))?;

        if old.status != TaskStatus::Running {
            let status = old.status;
            return Err(anyhow!(
                "repark rejected: task {parent_id} is not running (status {status:?})"
            ));
        }
        match &old.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "repark rejected: worker mismatch on task {parent_id}"
                ));
            }
        }
        match &old.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "repark rejected: lease mismatch on task {parent_id}"
                ));
            }
        }
        if !old.state.is_steering_resume() {
            return Err(anyhow!(
                "repark rejected: task {parent_id} is not a steering resume"
            ));
        }

        // Re-index the re-attach children to dense spawn_index
        // positions matching the new continuation's pending_tool_calls
        // order. Build every updated child row first so a validation
        // error on any child rolls the whole batch back cleanly.
        let mut updated_children: Vec<AgentTask> = Vec::with_capacity(reattach.len());
        for (idx, child_id) in reattach.iter().enumerate() {
            let mut child =
                inner.by_id.get(child_id).cloned().ok_or_else(|| {
                    anyhow!("repark rejected: re-attach child {child_id} missing")
                })?;
            if child.parent_id.as_ref() != Some(parent_id) {
                return Err(anyhow!(
                    "repark rejected: re-attach child {child_id} is not a child of {parent_id}"
                ));
            }
            child.spawn_index =
                Some(u32::try_from(idx).context("repark rejected: reattach index exceeds u32")?);
            child.updated_at = now;
            child
                .validate()
                .context("repark rejected: re-attach child validate failed")?;
            updated_children.push(child);
        }
        // spawn_index is not an indexed field, so a plain primary-key
        // overwrite keeps every secondary index consistent.
        for child in updated_children {
            inner.by_id.insert(child.id.clone(), child);
        }

        // Live count is journal-authoritative (matches the completion
        // path's recompute): a re-attach child that finished during the
        // steering LLM call is already terminal here and excluded.
        let live = inner.count_live_children(parent_id);
        let reparked = old
            .clone()
            .repark_after_steering(live, payload, reattach, now)
            .context("repark rejected: repark_after_steering transition failed")?;
        inner.rebalance_after_row_change(&old, &reparked);
        inner.by_id.insert(reparked.id.clone(), reparked.clone());
        drop(inner);
        Ok(reparked)
    }

    async fn pause_on_confirmation(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("pause rejected: task {id} does not exist"))?;

        if old.status != TaskStatus::Running {
            let status = old.status;
            return Err(anyhow!(
                "pause rejected: task {id} is not running (status {status:?})"
            ));
        }
        match &old.worker_id {
            Some(current) if current == worker => {}
            _ => return Err(anyhow!("pause rejected: worker mismatch on task {id}")),
        }
        match &old.lease_id {
            Some(current) if current == lease => {}
            _ => return Err(anyhow!("pause rejected: lease mismatch on task {id}")),
        }

        let paused = old
            .clone()
            .await_confirmation(continuation, prepared_operation, now)
            .context("pause rejected: await_confirmation transition failed")?;
        inner.rebalance_after_row_change(&old, &paused);
        inner.by_id.insert(paused.id.clone(), paused.clone());
        drop(inner);
        Ok(paused)
    }

    #[allow(clippy::too_many_arguments)]
    async fn spawn_tool_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        specs: Vec<ChildSpawnSpec>,
        payload: SuspensionPayload,
        child_otel_traceparent: Option<String>,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<AgentTask>)> {
        if specs.is_empty() {
            return Err(anyhow!("spawn rejected: specs must be non-empty"));
        }

        let mut inner = self.inner.write().await;

        // CAS + structural guards on the parent.
        let old_parent = inner
            .by_id
            .get(parent_id)
            .cloned()
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        if old_parent.status != TaskStatus::Running {
            let status = old_parent.status;
            return Err(anyhow!(
                "spawn rejected: task {parent_id} is not running (status {status:?})"
            ));
        }
        match &old_parent.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: worker mismatch on task {parent_id}"
                ));
            }
        }
        match &old_parent.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: lease mismatch on task {parent_id}"
                ));
            }
        }
        if old_parent.kind.is_leaf() {
            let parent_kind = old_parent.kind;
            return Err(anyhow!(
                "spawn rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
            ));
        }

        // Build every child row **before** mutating any index so a
        // schema error on child N rolls back the whole batch cleanly.
        // Each child passes through `AgentTask::new_child` which
        // calls `validate()` on the way out, so the children are
        // invariant-safe by construction. We still assert no id
        // collision because `AgentTaskId::new` is UUIDv4 and a
        // hand-crafted test that uses fixed ids could otherwise slip
        // a duplicate into the batch.
        let mut children: Vec<AgentTask> = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.into_iter().enumerate() {
            let mut child =
                AgentTask::new_child(&old_parent, TaskKind::ToolRuntime, now, spec.max_attempts)
                    .context("spawn rejected: new_child failed")?;
            child.spawn_index =
                Some(u32::try_from(idx).context("spawn rejected: batch index exceeds u32::MAX")?);
            // Re-parent the child's spans under the turn's root
            // `invoke_agent` span (not the inherited inbound client span)
            // so its `execute_tool` span nests under the turn.
            child.otel_traceparent.clone_from(&child_otel_traceparent);
            if inner.by_id.contains_key(&child.id)
                || children.iter().any(|existing| existing.id == child.id)
            {
                let id = &child.id;
                return Err(anyhow!("spawn rejected: child id {id} already exists"));
            }
            children.push(child);
        }

        // Transition the parent to WaitingOnChildren first so the
        // typed state carries the continuation before the children
        // become visible to the acquisition path. The pure
        // transition takes `child_count` so we pass the freshly
        // computed batch size, not a caller-supplied number.
        let child_count = u32::try_from(children.len())
            .context("spawn rejected: child count exceeds u32::MAX")?;
        let child_ids: Vec<AgentTaskId> = children.iter().map(|c| c.id.clone()).collect();
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        inner.rebalance_after_row_change(&old_parent, &new_parent);
        inner
            .by_id
            .insert(new_parent.id.clone(), new_parent.clone());

        // Commit every child to the primary key and secondary
        // indexes under the same write lock as the parent transition.
        // `add_to_indexes` keeps `by_thread`, `by_parent`,
        // `by_status`, and the Phase 2.3 runnable index in sync.
        for child in &children {
            inner.add_to_indexes(child);
            inner.by_id.insert(child.id.clone(), child.clone());
        }

        drop(inner);
        Ok((new_parent, children))
    }

    async fn spawn_subagent_invocation(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: SubagentInvocationSpawn,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let mut inner = self.inner.write().await;
        let SubagentInvocationSpawn {
            child_thread_id,
            spec,
            child_root_input,
            spawn_index,
            payload,
            child_caller_metadata,
        } = spawn;

        let old_parent = inner
            .by_id
            .get(parent_id)
            .cloned()
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        if old_parent.status != TaskStatus::Running {
            let status = old_parent.status;
            return Err(anyhow!(
                "spawn rejected: task {parent_id} is not running (status {status:?})"
            ));
        }
        match &old_parent.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: worker mismatch on task {parent_id}"
                ));
            }
        }
        match &old_parent.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "spawn rejected: lease mismatch on task {parent_id}"
                ));
            }
        }
        if old_parent.kind.is_leaf() {
            let parent_kind = old_parent.kind;
            return Err(anyhow!(
                "spawn rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
            ));
        }

        if inner.by_thread.contains_key(&child_thread_id)
            || inner.active_root_by_thread.contains_key(&child_thread_id)
        {
            return Err(anyhow!(
                "spawn rejected: child thread id {child_thread_id} already has tasks"
            ));
        }

        let child_root = AgentTask::new_root_turn_with_optional_caller(
            child_thread_id.clone(),
            child_root_input,
            child_caller_metadata,
            now,
            RuntimePolicy::server_default().max_attempts,
        );
        if inner.by_id.contains_key(&child_root.id) {
            let id = &child_root.id;
            return Err(anyhow!(
                "spawn rejected: child root task id {id} already exists"
            ));
        }

        let invocation = AgentTask::new_subagent_invocation(
            &old_parent,
            SubagentInvocationState {
                spec,
                child_thread_id,
                child_root_task_id: child_root.id.clone(),
            },
            spawn_index,
            now,
            AgentTask::DEFAULT_MAX_ATTEMPTS,
        )
        .context("spawn rejected: new_subagent_invocation failed")?;
        if inner.by_id.contains_key(&invocation.id) {
            let id = &invocation.id;
            return Err(anyhow!(
                "spawn rejected: invocation task id {id} already exists"
            ));
        }

        let new_parent = old_parent
            .clone()
            .wait_on_children(1, payload, vec![invocation.id.clone()], now)
            .context("spawn rejected: wait_on_children transition failed")?;
        inner.rebalance_after_row_change(&old_parent, &new_parent);
        inner
            .by_id
            .insert(new_parent.id.clone(), new_parent.clone());

        inner.add_to_indexes(&invocation);
        inner
            .by_id
            .insert(invocation.id.clone(), invocation.clone());

        inner.add_to_indexes(&child_root);
        inner
            .by_id
            .insert(child_root.id.clone(), child_root.clone());

        drop(inner);
        Ok((new_parent, invocation, child_root))
    }

    async fn spawn_subagent_batch(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawns: Vec<SubagentInvocationSpawn>,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>)> {
        if spawns.is_empty() {
            return Err(anyhow!("spawn rejected: spawns must be non-empty"));
        }
        let mut inner = self.inner.write().await;

        let old_parent = inner.validate_running_owned_non_leaf_parent(parent_id, worker, lease)?;
        inner.validate_batch_thread_uniqueness(&spawns)?;

        // Materialize the (invocation, child_root) pair per entry —
        // mutate nothing in `inner` until every row is built so a
        // schema error on entry K leaves the store untouched.
        let (prepared, child_ids) = inner.build_batch_invocation_pairs(&old_parent, spawns, now)?;

        let child_count =
            u32::try_from(prepared.len()).context("spawn rejected: child count exceeds u32")?;
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;

        // All validation passed — apply mutations.
        inner.rebalance_after_row_change(&old_parent, &new_parent);
        inner
            .by_id
            .insert(new_parent.id.clone(), new_parent.clone());
        for (invocation, child_root) in &prepared {
            inner.add_to_indexes(invocation);
            inner
                .by_id
                .insert(invocation.id.clone(), invocation.clone());
            inner.add_to_indexes(child_root);
            inner
                .by_id
                .insert(child_root.id.clone(), child_root.clone());
        }

        drop(inner);
        Ok((new_parent, prepared))
    }

    async fn spawn_mixed_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: MixedChildrenSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedMixedChildren> {
        validate_mixed_children_spawn(&spawn)?;
        let MixedChildrenSpawn {
            subagents,
            tool_children,
            payload,
            child_otel_traceparent,
        } = spawn;

        let mut inner = self.inner.write().await;

        let old_parent = inner.validate_running_owned_non_leaf_parent(parent_id, worker, lease)?;
        inner.validate_batch_thread_uniqueness(&subagents)?;

        // Materialize every row — invocation pairs and tool children —
        // before touching a single index, so a schema error or an id
        // collision anywhere in the batch leaves the store untouched.
        let (prepared, _batch_child_ids) =
            inner.build_batch_invocation_pairs(&old_parent, subagents, now)?;
        let tool_rows = inner.build_mixed_tool_children(
            &old_parent,
            &tool_children,
            child_otel_traceparent.as_deref(),
            now,
        )?;
        let child_ids = mixed_child_ids_in_slot_order(&prepared, &tool_rows)?;

        let child_count =
            u32::try_from(child_ids.len()).context("spawn rejected: child count exceeds u32")?;
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;

        // All validation passed — apply mutations.
        inner.rebalance_after_row_change(&old_parent, &new_parent);
        inner
            .by_id
            .insert(new_parent.id.clone(), new_parent.clone());
        for (invocation, child_root) in &prepared {
            inner.add_to_indexes(invocation);
            inner
                .by_id
                .insert(invocation.id.clone(), invocation.clone());
            inner.add_to_indexes(child_root);
            inner
                .by_id
                .insert(child_root.id.clone(), child_root.clone());
        }
        for child in &tool_rows {
            inner.add_to_indexes(child);
            inner.by_id.insert(child.id.clone(), child.clone());
        }

        drop(inner);
        Ok(SpawnedMixedChildren {
            parent: new_parent,
            subagents: prepared,
            tool_children: tool_rows,
        })
    }

    async fn find_subagent_invocation_for_child_root(
        &self,
        child_root_id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        let inner = self.inner.read().await;
        // The Phase 7.6 reverse index is a hint; the row state is
        // authoritative (same contract as
        // `resume_linked_subagent_invocation`). Verify kind, parked
        // status, and the linkage before returning the row so a stale
        // index entry can never surface a woken or repurposed
        // invocation.
        let invocation = inner
            .invocation_by_child_root
            .get(child_root_id)
            .and_then(|invocation_id| inner.by_id.get(invocation_id));
        let linked = invocation.is_some_and(|invocation| {
            invocation.kind == TaskKind::Subagent
                && invocation.status == TaskStatus::WaitingOnChildren
                && invocation
                    .state
                    .subagent_invocation()
                    .is_some_and(|state| state.child_root_task_id == *child_root_id)
        });
        let result = if linked { invocation.cloned() } else { None };
        drop(inner);
        Ok(result)
    }

    async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let mut result: Vec<AgentTask> = inner
            .by_status
            .get(&TaskStatus::WaitingOnChildren)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id))
            .filter(|task| task.kind == TaskKind::Subagent)
            .cloned()
            .collect();
        drop(inner);
        // `by_status` is hash-ordered; sort to the SQL backends'
        // deterministic `(created_at, id)` order.
        result.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id)));
        Ok(result)
    }

    async fn complete_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut inner = self.inner.write().await;
        let result = inner.apply_task_terminal_transition(
            child_id,
            worker,
            lease,
            now,
            "complete_task",
            |child| child.complete(now),
        )?;
        drop(inner);
        Ok(result)
    }

    async fn complete_task_with_result(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        result_payload: serde_json::Value,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut inner = self.inner.write().await;
        let result = inner.apply_task_terminal_transition(
            child_id,
            worker,
            lease,
            now,
            "complete_task_with_result",
            |child| child.complete_with_result(result_payload, now),
        )?;
        drop(inner);
        Ok(result)
    }

    async fn fail_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut inner = self.inner.write().await;
        let result = inner.apply_task_terminal_transition(
            child_id,
            worker,
            lease,
            now,
            "fail_task",
            move |child| child.fail(error, now),
        )?;
        drop(inner);
        Ok(result)
    }

    async fn cancel_tree(
        &self,
        root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome> {
        let mut inner = self.inner.write().await;

        if !inner.by_id.contains_key(root_id) {
            return Err(anyhow!(
                "cancel_tree rejected: task {root_id} does not exist"
            ));
        }

        // Snapshot the subtree ids under the write lock and then
        // walk them in BFS order. The cancel transition is pure and
        // does not mutate `by_parent`, so the snapshot stays
        // consistent for the duration of the sweep even though we
        // are holding a mutable borrow of `inner`.
        //
        // The snapshot, every transition, and the
        // marker gating all run under this one write lock, so no
        // concurrent writer can settle a row mid-sweep (and
        // `cancel_row_in_place` re-reads `by_id` for each id anyway) —
        // this backend is immune to the stale-snapshot race the
        // Postgres path had to close with locked re-reads.
        let subtree = inner.collect_subtree(root_id);

        // Decide, under this lock, exactly which rows will cancel and
        // which of them are due a terminal marker. The predicate mirrors
        // `cancel_row_in_place`'s gate (row present, not already
        // terminal), and the marker payload is read here because the
        // cancel clears the typed state it prefers.
        //
        // The decision is exact, not a guess: the write lock excludes
        // every concurrent writer, and a row can only reach a terminal
        // state through its OWN `cancel_row_in_place` below —
        // `propagate_terminal_child_transition` recomputes a PARENT's
        // pending-child count, it never settles a descendant. So each
        // predicted row still transitions when the loop reaches it,
        // which is what keeps markers exactly-once: an idempotent retry
        // over a terminal tree predicts nothing and emits nothing.
        let mut to_cancel = Vec::with_capacity(subtree.len());
        let mut marker_candidates: Vec<CancelMarkerCandidate> = Vec::new();
        for id in subtree {
            let Some(row) = inner.by_id.get(&id) else {
                continue;
            };
            if row.status.is_terminal() {
                continue;
            }
            if row.kind == TaskKind::RootTurn && row.is_root() && row.status.blocks_root_admission()
            {
                marker_candidates.push(CancelMarkerCandidate {
                    thread_id: row.thread_id.clone(),
                    task_id: row.id.clone(),
                    continuation_usage: row
                        .state
                        .continuation()
                        .map(|continuation| continuation.payload.total_usage.clone()),
                });
            }
            to_cancel.push(id);
        }

        // The whole marker set is journaled BEFORE any row flips, while
        // STILL holding the task-store write lock.
        //
        // Ordering is the invariant, not an optimisation: a `Cancelled`
        // row always has its marker in the journal. A sink failure
        // returns here with not one transition applied, so the tree stays
        // intact and a retry can complete it — there is no state in which
        // a follower sees a cancelled emitter with no marker to close it.
        // No rollback exists across separate stores, so the only way to
        // get that half is to run the fallible writes first.
        //
        // What this ordering does NOT buy — and the flips cannot be
        // paired per row to buy it, because a flip clears the typed state
        // carrying `SubagentInvocation` linkage, and a half-flipped
        // cascade is unreachable to the `collect_subtree` a retry depends
        // on: a failure PART-WAY through a multi-thread cascade's markers
        // leaves the already-journaled ones behind on roots that stay
        // live, and marker closes are ungated, so a follower can close
        // `TurnCancelled` on a task that then completes. The retry does
        // not compound it — the append is idempotent per candidate, so
        // one marker per effective cancel survives and the retry's flips
        // make the orphan honest. See `CancellationMarkerSink`; the SQL
        // backends commit the set in one transaction and have neither
        // window.
        //
        // Holding the lock across the sink's awaits is what orders the
        // markers ahead of anything a promoted successor emits: the
        // successor cannot be acquired (so cannot commit its `Start`)
        // until this lock drops — matching the SQL backends. The sink
        // stores are in-process leaf stores that never call back into
        // the task store, so holding the lock across them cannot
        // deadlock.
        let markers = self
            .commit_cancellation_markers(marker_candidates, now)
            .await?;

        // The transitions: pure, synchronous, no await. Once the markers
        // are durable nothing can interrupt the flips they describe.
        let mut transitioned = Vec::with_capacity(to_cancel.len());
        for id in to_cancel {
            if inner.cancel_row_in_place(&id, now)? {
                transitioned.push(id);
            }
        }

        drop(inner);
        Ok(CancelTreeOutcome {
            transitioned,
            markers,
        })
    }

    async fn resume_from_confirmation(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("resume rejected: task {id} does not exist"))?;

        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "resume rejected: task {id} is not awaiting confirmation (status {status:?})"
            ));
        }

        let (resumed, prepared_operation) = old
            .clone()
            .resume_from_confirmation(now)
            .context("resume rejected: resume_from_confirmation transition failed")?;
        inner.rebalance_after_row_change(&old, &resumed);
        inner.by_id.insert(resumed.id.clone(), resumed.clone());
        drop(inner);
        Ok((resumed, prepared_operation))
    }

    async fn approve_confirmation_and_acquire(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("approve rejected: task {id} does not exist"))?;

        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "approve rejected: task {id} is not awaiting confirmation (status {status:?})"
            ));
        }

        let (resumed, prepared_operation) = old
            .clone()
            .resume_from_confirmation(now)
            .context("approve rejected: resume_from_confirmation transition failed")?;
        let claimed = resumed
            .mark_running(worker, lease, expires_at, now)
            .context("approve rejected: mark_running transition failed")?;
        inner.rebalance_after_row_change(&old, &claimed);
        inner.by_id.insert(claimed.id.clone(), claimed.clone());
        drop(inner);
        Ok((claimed, prepared_operation))
    }

    async fn reject_confirmation(
        &self,
        id: &AgentTaskId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("reject_confirmation rejected: task {id} does not exist"))?;

        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "reject_confirmation rejected: task {id} is not awaiting confirmation \
                 (status {status:?})"
            ));
        }

        let failed = old
            .clone()
            .fail(error, now)
            .context("reject_confirmation: fail transition failed")?;
        inner.rebalance_after_row_change(&old, &failed);
        inner.by_id.insert(failed.id.clone(), failed.clone());

        // Recompute the parent's pending_child_count the same way
        // `apply_task_terminal_transition` does in complete_task /
        // fail_task. If this was the last live child the parent
        // becomes Pending with ReadyToResume.
        let parent = if let Some(parent_id) = &failed.parent_id {
            let Some(old_parent) = inner.by_id.get(parent_id).cloned() else {
                return Err(anyhow!(
                    "reject_confirmation: child {id} references missing parent {parent_id}"
                ));
            };
            if old_parent.status == TaskStatus::WaitingOnChildren {
                let live = inner.count_live_children(parent_id);
                let new_parent = old_parent
                    .clone()
                    .recompute_pending_children(live, now)
                    .context("reject_confirmation: recompute_pending_children failed")?;
                inner.rebalance_after_row_change(&old_parent, &new_parent);
                inner
                    .by_id
                    .insert(new_parent.id.clone(), new_parent.clone());
                Some(new_parent)
            } else {
                Some(old_parent)
            }
        } else {
            None
        };

        drop(inner);
        Ok((failed, parent))
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        *inner = Inner::default();
        drop(inner);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::{LeaseId, TaskSchemaError, WorkerId};
    use crate::journal::task_state::TaskState;
    use agent_sdk_foundation::{AgentContinuation, AgentState, ContinuationEnvelope, TokenUsage};
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t0(), 3)
    }

    /// Sample [`ContinuationEnvelope`] used by every Phase 2.4 pause /
    /// resume test in this module. The exact contents do not matter —
    /// we only care that the envelope round-trips through the typed
    /// [`TaskState`] payload and survives writes to the store.
    fn sample_continuation(name: &str) -> ContinuationEnvelope {
        let thread = thread(name);
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

    /// Sample listen/execute prepared operation paired with the
    /// confirmation continuation in tests that exercise the typed
    /// [`TaskState::AwaitingConfirmation`] payload.
    fn sample_prepared_op() -> agent_sdk_foundation::ListenExecutionContext {
        agent_sdk_foundation::ListenExecutionContext {
            operation_id: "op-store".into(),
            revision: 1,
            snapshot: serde_json::json!({"preview": true}),
            expires_at: None,
        }
    }

    /// Build a fresh root turn on `thread` with `created_at` at `t0 +
    /// secs` seconds. Used by Phase 2.2 FIFO tests that need two roots
    /// with deterministic, distinct wall-clock timestamps.
    fn fresh_root_at(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t_plus(secs), 3)
    }

    // ── idempotency claim atomicity (Phase 10 · E) ───────────────

    #[tokio::test]
    async fn claim_idempotency_reserves_then_replays_after_record() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let fp: &[u8] = b"fp-1";

        // First claim wins and reserves the key.
        assert!(matches!(
            store
                .claim_idempotency("req-1", IdempotencyKind::CreateThread, fp)
                .await?,
            IdempotencyClaim::Fresh
        ));

        // A second claim before the effect records its result must NOT
        // also be Fresh — the placeholder reservation makes it Conflict.
        assert!(matches!(
            store
                .claim_idempotency("req-1", IdempotencyKind::CreateThread, fp)
                .await?,
            IdempotencyClaim::Conflict
        ));

        // Once the winner records the durable references, retries replay.
        store
            .record_idempotency(IdempotencyRecord {
                request_id: "req-1".into(),
                kind: IdempotencyKind::CreateThread,
                fingerprint: fp.to_vec(),
                result_json: serde_json::json!({ "thread_id": "th-1" }),
            })
            .await?;

        match store
            .claim_idempotency("req-1", IdempotencyKind::CreateThread, fp)
            .await?
        {
            IdempotencyClaim::Replay(record) => {
                assert_eq!(
                    record.result_json,
                    serde_json::json!({ "thread_id": "th-1" })
                );
            }
            other => panic!("expected Replay, got {other:?}"),
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn two_concurrent_claims_yield_exactly_one_fresh() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let fp: &[u8] = b"fp-x";

        let s1 = store.clone();
        let s2 = store.clone();
        let (a, b) = tokio::join!(
            async move {
                s1.claim_idempotency("req-c", IdempotencyKind::CreateThread, b"fp-x")
                    .await
            },
            async move {
                s2.claim_idempotency("req-c", IdempotencyKind::CreateThread, b"fp-x")
                    .await
            },
        );
        let fresh = [a?, b?]
            .into_iter()
            .filter(|c| matches!(c, IdempotencyClaim::Fresh))
            .count();
        assert_eq!(fresh, 1, "exactly one concurrent claim must be Fresh");
        let _ = fp;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_and_get_round_trip() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.context("insert")?;
        let got = store
            .get(&id)
            .await
            .context("get")?
            .context("task exists")?;
        // AgentTask does not impl PartialEq; the canonical equality
        // contract is the JSON wire form (Phase 2.4 typed-state move).
        assert_eq!(
            serde_json::to_value(&got).context("got to value")?,
            serde_json::to_value(&task).context("task to value")?
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_invalid_task() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let mut task = fresh_root("t1");
        task.status = TaskStatus::Running; // invalid: missing lease fields
        let err = store.insert(task).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_invalid_task() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.context("insert")?;

        // Load and corrupt the row.
        let mut bad = store
            .get(&id)
            .await
            .context("get")?
            .context("task exists")?;
        bad.status = TaskStatus::Running; // still no lease fields -> invalid
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_children_returns_only_direct_children() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert root")?;

        let tool_a =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("a")?;
        let tool_b =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("b")?;
        store.insert(tool_a.clone()).await.context("insert a")?;
        store.insert(tool_b.clone()).await.context("insert b")?;

        let children = store.list_children(&root.id).await.context("children")?;
        assert_eq!(children.len(), 2);
        let ids: std::collections::HashSet<_> = children.iter().map(|c| c.id.clone()).collect();
        assert!(ids.contains(&tool_a.id));
        assert!(ids.contains(&tool_b.id));

        // Tool runtimes have no children.
        let empty = store.list_children(&tool_a.id).await.context("empty")?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_thread_returns_root_and_descendants() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        let tool =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("tool")?;
        store.insert(tool.clone()).await.context("tool")?;

        let other = fresh_root("t2");
        store.insert(other.clone()).await.context("other")?;

        let t1 = store
            .list_by_thread(&thread("t1"))
            .await
            .context("list t1")?;
        let ids: std::collections::HashSet<_> = t1.iter().map(|t| t.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&root.id));
        assert!(ids.contains(&tool.id));

        let t2 = store
            .list_by_thread(&thread("t2"))
            .await
            .context("list t2")?;
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, other.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_status_reflects_current_state() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .context("pending")?;
        assert_eq!(pending.len(), 1);

        // Promote to running.
        let running = root
            .clone()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store.update(running.clone()).await.context("update")?;

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .context("pending after")?;
        assert!(pending.is_empty());
        let running_list = store
            .list_by_status(TaskStatus::Running)
            .await
            .context("running list")?;
        assert_eq!(running_list.len(), 1);
        assert_eq!(running_list[0].id, running.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn active_root_for_thread_returns_only_non_terminal_root() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert_eq!(active.map(|t| t.id), Some(root.id.clone()));

        // Complete the root via the legal transition path.
        let running = root
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let done = running.complete(t_plus(2)).context("complete")?;
        store.update(done).await.context("update done")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active after")?;
        assert!(active.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_rejects_second_active_root_on_same_thread() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first).await.context("first")?;
        let second = fresh_root("t1");
        let err = store.insert(second).await.unwrap_err();
        assert!(
            err.to_string().contains("already has active root"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_new_root_after_previous_root_completes() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first.clone()).await.context("first")?;

        // Walk the first root to Completed through the legal path.
        let running = first
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let done = running.complete(t_plus(2)).context("complete")?;
        store.update(done).await.context("update done")?;

        // A brand-new root on the same thread must now be admissible.
        let second = fresh_root("t1");
        store.insert(second.clone()).await.context("second")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert_eq!(active.map(|t| t.id), Some(second.id));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_tool_runtime_children_while_root_is_running() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        // Transition the root to Running.
        let running = root
            .clone()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;

        // Insert a tool-runtime child under the still-active root.
        let tool =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 1).context("tool")?;
        store.insert(tool.clone()).await.context("tool insert")?;

        let children = store.list_children(&running.id).await.context("children")?;
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, tool.id);

        // The active root must still be reachable.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("still active")?;
        assert_eq!(active.id, running.id);
        assert_eq!(active.status, TaskStatus::Running);
        Ok(())
    }

    // ── regression tests for row-invariant guards ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_kind_mutation_even_when_status_is_active() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        // Mutate kind: RootTurn -> ToolRuntime while keeping status active.
        // Without the guard this would corrupt active_root_by_thread: the
        // cleanup branch only fires when !task.status.is_active(), and the
        // re-registration branch skips non-RootTurn kinds, leaving a stale
        // root-id pointer on the thread.
        let mut mutated = root.clone();
        mutated.kind = TaskKind::ToolRuntime;
        // Give the "child" the shape a ToolRuntime would have so only the
        // kind mutation is under test (parent_id/root_id still satisfy
        // validate() because this is still the root row).
        let err = store.update(mutated).await.unwrap_err();
        assert!(
            err.to_string().contains("kind is immutable"),
            "unexpected: {err}"
        );

        // Regression check: the active-root pointer and its kind are intact.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("still active")?;
        assert_eq!(active.id, root.id);
        assert_eq!(active.kind, TaskKind::RootTurn);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_mutation_of_any_row_invariant() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        // thread_id
        let mut bad = root.clone();
        bad.thread_id = thread("t-other");
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("thread_id is immutable"),
            "unexpected: {err}"
        );

        // parent_id (root has None; setting Some must be rejected)
        let mut bad = root.clone();
        bad.parent_id = Some(AgentTaskId::new());
        let err = store.update(bad).await.unwrap_err();
        // parent_id guard fires before validate() on the store path,
        // but validate() would also reject this as RootHasParent. Accept
        // either wording — the important thing is the row is not written.
        assert!(
            err.to_string().contains("parent_id is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // root_id
        let mut bad = root.clone();
        bad.root_id = AgentTaskId::new();
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("root_id is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // depth
        let mut bad = root.clone();
        bad.depth = 1;
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("depth is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // created_at
        let mut bad = root.clone();
        bad.created_at = t_plus(9999);
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("created_at is immutable"),
            "unexpected: {err}"
        );

        // max_attempts
        let mut bad = root;
        bad.max_attempts = 99;
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("max_attempts is immutable"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_unknown_parent() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let ghost_root = fresh_root("t1");
        // Build a child referencing ghost_root without ever inserting it.
        let child = AgentTask::new_child(&ghost_root, TaskKind::ToolRuntime, t_plus(1), 1)
            .context("child")?;
        let err = store.insert(child).await.unwrap_err();
        assert!(
            err.to_string().contains("unknown parent"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_mutated_thread_id() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.thread_id = thread("t-other");
        let err = store.insert(child).await.unwrap_err();
        assert!(err.to_string().contains("thread_id"), "unexpected: {err}");

        // Regression check: the wrong thread must not now bucket a child.
        let other = store
            .list_by_thread(&thread("t-other"))
            .await
            .context("list other")?;
        assert!(other.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_mutated_root_id_or_depth() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        // Mutated root_id
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.root_id = AgentTaskId::new();
        // validate() already catches this as ChildRootIdMismatch is not
        // the only path; the store's cross-row check fires first.
        let err = store.insert(child).await.unwrap_err();
        assert!(
            err.to_string().contains("root_id") || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // Mutated depth
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.depth = 42;
        let err = store.insert(child).await.unwrap_err();
        assert!(err.to_string().contains("depth"), "unexpected: {err}");
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.2 — Root submission queue and FIFO promotion
    // ──────────────────────────────────────────────────────────────
    //
    // These tests exercise the queue admission contract across every
    // blocking active-root state (`Pending`, `Running`,
    // `WaitingOnChildren`, `AwaitingConfirmation`), FIFO promotion order,
    // retry safety, and tool-child isolation. They form the acceptance
    // suite for Phase 2.2 and should cover every rule in the Linear
    // issue.

    /// Helper: mark a task Running via the legal transition path and
    /// persist the resulting row. Used to put a thread's active root
    /// into a blocking non-Pending state in tests.
    async fn running_root(store: &InMemoryAgentTaskStore, root: AgentTask) -> Result<AgentTask> {
        let running = root
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store.update(running.clone()).await.context("update")?;
        Ok(running)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_admits_first_root_as_pending() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        let admitted = store
            .submit_root_turn(first.clone())
            .await
            .context("submit")?;
        assert_eq!(admitted.status, TaskStatus::Pending);
        assert_eq!(admitted.id, first.id);

        // The active-root slot is now held and queued list is empty.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);
        assert_eq!(active.status, TaskStatus::Pending);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert!(queued.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_queues_second_root_behind_pending() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        let admitted = store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        assert_eq!(admitted.status, TaskStatus::Queued);
        assert_eq!(admitted.id, second.id);

        // The active-root slot is still held by the first root.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);
        assert_eq!(queued[0].status, TaskStatus::Queued);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_queues_behind_every_blocking_state() -> Result<()> {
        for &(label, setup) in &[
            ("Pending", 0i64),
            ("Running", 1),
            ("WaitingOnChildren", 2),
            ("AwaitingConfirmation", 3),
        ] {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            store
                .submit_root_turn(first.clone())
                .await
                .context("first")?;

            // Drive `first` into the blocking state under test. Each
            // branch ends with the active root in a non-Pending blocking
            // status, so a second submission must queue.
            match setup {
                0 => {
                    // Already Pending — no-op.
                }
                1 => {
                    running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                }
                2 => {
                    let running = running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                    let waiting = running
                        .wait_on_children(
                            2,
                            SuspensionPayload {
                                continuation: sample_continuation("t1"),
                                suspended_messages: Vec::new(),
                            },
                            Vec::new(),
                            t_plus(2),
                        )
                        .context("wait_on_children")?;
                    store.update(waiting).await.context("update waiting")?;
                }
                3 => {
                    let running = running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                    let awaiting = running
                        .await_confirmation(sample_continuation("t1"), None, t_plus(2))
                        .context("await_confirmation")?;
                    store.update(awaiting).await.context("update awaiting")?;
                }
                _ => unreachable!(),
            }

            let second = fresh_root_at("t1", 10);
            let admitted = store
                .submit_root_turn(second.clone())
                .await
                .context("second submit")?;
            assert_eq!(
                admitted.status,
                TaskStatus::Queued,
                "second root did not queue behind blocking state {label}",
            );

            let queued = store
                .list_queued_roots(&thread("t1"))
                .await
                .context("queued")?;
            assert_eq!(queued.len(), 1, "queue length wrong for state {label}");
            assert_eq!(
                queued[0].id, second.id,
                "queue head wrong for state {label}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_promote_in_fifo_order_after_active_completes() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);
        let third = fresh_root_at("t1", 2);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        store
            .submit_root_turn(third.clone())
            .await
            .context("third")?;

        // Walk the first root to Completed.
        let running = running_root(&store, first.clone())
            .await
            .context("drive running")?;
        let done = running.complete(t_plus(10)).context("complete")?;
        store.update(done).await.context("update done")?;

        // Queued roots still in FIFO order in `list_queued_roots`.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 2);
        assert_eq!(queued[0].id, second.id);
        assert_eq!(queued[1].id, third.id);

        // Promote the head: it must be the *oldest* queued root.
        let promoted = store
            .promote_next_queued_root(&thread("t1"), t_plus(11))
            .await
            .context("promote")?
            .context("promotion fired")?;
        assert_eq!(promoted.id, second.id);
        assert_eq!(promoted.status, TaskStatus::Pending);

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, second.id);

        // The second promotion must not fire while the slot is held.
        let no_promotion = store
            .promote_next_queued_root(&thread("t1"), t_plus(12))
            .await
            .context("promote blocked")?;
        assert!(no_promotion.is_none());

        // Walk `second` to Completed, then promote `third`.
        let running2 = running_root(&store, promoted.clone())
            .await
            .context("drive running 2")?;
        let done2 = running2.complete(t_plus(20)).context("complete 2")?;
        store.update(done2).await.context("update done 2")?;

        let promoted2 = store
            .promote_next_queued_root(&thread("t1"), t_plus(21))
            .await
            .context("promote 2")?
            .context("promotion 2 fired")?;
        assert_eq!(promoted2.id, third.id);

        // Queue is now drained.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued after")?;
        assert!(queued.is_empty());
        Ok(())
    }

    /// Regression for finding #4/#8: a store-side terminal transition
    /// that frees the active-root slot must promote the queued FIFO head
    /// in the SAME locked scope — not rely on the host's post-commit
    /// promotion path. `cancel_tree` (the `cancel_row_in_place` path)
    /// must auto-promote the successor.
    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_of_active_root_auto_promotes_queued_successor() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        // Cancel the active root. No explicit promote call follows.
        let cancelled = store
            .cancel_tree(&first.id, t_plus(10))
            .await
            .context("cancel_tree")?
            .transitioned;
        assert_eq!(cancelled, vec![first.id.clone()]);

        // The queued successor must already be Pending and hold the slot.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot should be held by the promoted successor")?;
        assert_eq!(active.id, second.id);
        assert_eq!(active.status, TaskStatus::Pending);
        assert!(
            store
                .list_queued_roots(&thread("t1"))
                .await
                .context("queued")?
                .is_empty()
        );
        Ok(())
    }

    /// Regression for finding #4/#8: failing the active root closed (the
    /// recovery / `fail_task` terminal path) must auto-promote the queued
    /// FIFO head without an explicit promotion call.
    #[tokio::test(flavor = "multi_thread")]
    async fn fail_of_active_root_auto_promotes_queued_successor() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        // Drive `first` to Running, then fail it closed via the CAS path.
        running_root(&store, first.clone())
            .await
            .context("running")?;
        store
            .fail_task(
                &first.id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                "budget exhausted".into(),
                t_plus(10),
            )
            .await
            .context("fail_task")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot should be held by the promoted successor")?;
        assert_eq!(active.id, second.id);
        assert_eq!(active.status, TaskStatus::Pending);
        Ok(())
    }

    /// Regression test for the FIFO queue-jump bug: a new root submitted
    /// after the active root completes but *before* `promote_next_queued_root`
    /// is called must be queued behind any already-waiting roots — not
    /// admitted directly as Pending (which would jump the queue).
    #[tokio::test(flavor = "multi_thread")]
    async fn submit_after_active_completes_does_not_jump_queue() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        // Submit two roots: first is active/Pending, second is Queued.
        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        // Complete the first root — this frees the active-root slot but
        // does NOT auto-promote queued roots.
        let running = running_root(&store, first.clone())
            .await
            .context("drive running")?;
        let done = running.complete(t_plus(10)).context("complete")?;
        store.update(done).await.context("update done")?;

        // Submit a third root *before* promote is called.
        // Without the fix, thread_has_blocking_root returns false (slot
        // is free) and the newcomer would be admitted as Pending, jumping
        // ahead of `second`.
        let third = fresh_root_at("t1", 11);
        let admitted_third = store
            .submit_root_turn(third.clone())
            .await
            .context("third")?;
        assert_eq!(
            admitted_third.status,
            TaskStatus::Queued,
            "third must be queued behind second, not admitted as Pending"
        );

        // Verify FIFO order: second is still first in the queue.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 2);
        assert_eq!(queued[0].id, second.id, "second must be ahead of third");
        assert_eq!(queued[1].id, third.id);

        // Promote should yield `second`, not `third`.
        let promoted = store
            .promote_next_queued_root(&thread("t1"), t_plus(12))
            .await
            .context("promote")?
            .context("promotion fired")?;
        assert_eq!(promoted.id, second.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_is_noop_when_queue_is_empty() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;

        // Complete without any queued roots behind.
        let running = running_root(&store, first.clone())
            .await
            .context("drive running")?;
        let done = running.complete(t_plus(10)).context("complete")?;
        store.update(done).await.context("update done")?;

        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(11))
            .await
            .context("promote empty")?;
        assert!(nothing.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_is_noop_while_active_root_is_still_blocking() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store.submit_root_turn(first).await.context("first")?;
        store.submit_root_turn(second).await.context("second")?;

        // Slot still held by `first` in Pending: promotion must refuse.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(2))
            .await
            .context("promote while blocking")?;
        assert!(nothing.is_none());

        // Queued list is still intact.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retry_of_active_root_does_not_let_queued_roots_overtake() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        let admitted_first = store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        // Lease the first, then release it back to Pending (a retry).
        // `release_lease` keeps `attempt` at 1 but clears the lease.
        let running = running_root(&store, admitted_first.clone())
            .await
            .context("drive running")?;
        let released = running.release_lease(t_plus(5)).context("release")?;
        assert_eq!(released.status, TaskStatus::Pending);
        assert_eq!(released.attempt, 1);
        store.update(released).await.context("update released")?;

        // The active-root slot is still held by the first root — the
        // promotion attempt must be a no-op so queued roots cannot
        // overtake a retry.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(6))
            .await
            .context("promote during retry")?;
        assert!(nothing.is_none());

        // And the second root is still queued, not promoted.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);
        assert_eq!(active.status, TaskStatus::Pending);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);

        // Running → complete the first; now the second must promote.
        let running2 = running_root(&store, active)
            .await
            .context("drive running again")?;
        let done = running2.complete(t_plus(7)).context("complete")?;
        store.update(done).await.context("update done")?;

        let promoted = store
            .promote_next_queued_root(&thread("t1"), t_plus(8))
            .await
            .context("promote after complete")?
            .context("promotion fired")?;
        assert_eq!(promoted.id, second.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_promote_after_failure_and_cancellation() -> Result<()> {
        // Failure.
        {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            let second = fresh_root_at("t1", 1);
            store.submit_root_turn(first.clone()).await.context("f1")?;
            store.submit_root_turn(second.clone()).await.context("q1")?;

            let running = running_root(&store, first.clone())
                .await
                .context("running")?;
            let failed = running.fail("boom".into(), t_plus(10)).context("fail")?;
            store.update(failed).await.context("update failed")?;

            let promoted = store
                .promote_next_queued_root(&thread("t1"), t_plus(11))
                .await
                .context("promote after fail")?
                .context("promotion fired")?;
            assert_eq!(promoted.id, second.id);
        }

        // Cancellation (straight from Pending).
        {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            let second = fresh_root_at("t1", 1);
            store.submit_root_turn(first.clone()).await.context("f2")?;
            store.submit_root_turn(second.clone()).await.context("q2")?;

            let cancelled = first.cancel(t_plus(10)).context("cancel")?;
            store.update(cancelled).await.context("update cancelled")?;

            let promoted = store
                .promote_next_queued_root(&thread("t1"), t_plus(11))
                .await
                .context("promote after cancel")?
                .context("promotion fired")?;
            assert_eq!(promoted.id, second.id);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_runtime_children_never_block_root_admission() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit root")?;

        // Drive root to Running and spawn a tool-runtime child under it.
        let running = running_root(&store, root.clone())
            .await
            .context("drive running")?;
        let tool =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 1).context("tool")?;
        store.insert(tool.clone()).await.context("insert tool")?;

        // A second root submission must queue behind `running`, not the
        // tool child — tool children do not participate in root
        // admission.
        let second = fresh_root_at("t1", 3);
        let admitted = store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        assert_eq!(admitted.status, TaskStatus::Queued);

        // The active-root slot is exactly the root, not the tool.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, running.id);
        assert_eq!(active.kind, TaskKind::RootTurn);

        // Now complete the tool child. Tool children completing must
        // not by themselves promote queued roots — the root is still
        // running. We drive the tool to Running + Completed to prove
        // the full tool lifecycle doesn't leak into root admission.
        let tool_running = tool
            .mark_running(
                WorkerId::from_string("w-tool"),
                LeaseId::from_string("l-tool"),
                t_plus(60),
                t_plus(3),
            )
            .context("tool running")?;
        store
            .update(tool_running.clone())
            .await
            .context("update tool running")?;
        let tool_done = tool_running.complete(t_plus(4)).context("tool done")?;
        store.update(tool_done).await.context("update tool done")?;

        // Queued list still holds the second root; the active slot is
        // still held by the running root.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);

        // Promotion attempt still a no-op while the root itself is
        // blocking.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(5))
            .await
            .context("promote blocked")?;
        assert!(nothing.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_do_not_cross_thread_boundaries() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        // Thread A: blocking root + one queued root.
        let a_first = fresh_root_at("a", 0);
        let a_second = fresh_root_at("a", 1);
        store
            .submit_root_turn(a_first.clone())
            .await
            .context("a1")?;
        store
            .submit_root_turn(a_second.clone())
            .await
            .context("a2")?;

        // Thread B: its first root must admit as Pending, not queue
        // behind thread A's root.
        let b_first = fresh_root_at("b", 0);
        let b_admitted = store
            .submit_root_turn(b_first.clone())
            .await
            .context("b1")?;
        assert_eq!(b_admitted.status, TaskStatus::Pending);

        // Each thread sees only its own queue.
        let a_queued = store
            .list_queued_roots(&thread("a"))
            .await
            .context("a queue")?;
        assert_eq!(a_queued.len(), 1);
        assert_eq!(a_queued[0].id, a_second.id);

        let b_queued = store
            .list_queued_roots(&thread("b"))
            .await
            .context("b queue")?;
        assert!(b_queued.is_empty());

        // Completing thread B's root must not fire a promotion on A.
        let b_running = running_root(&store, b_first.clone())
            .await
            .context("b running")?;
        let b_done = b_running.complete(t_plus(10)).context("b done")?;
        store.update(b_done).await.context("update b done")?;

        let a_active = store
            .active_root_for_thread(&thread("a"))
            .await
            .context("a active")?
            .context("still held")?;
        assert_eq!(a_active.id, a_first.id);
        let a_queued = store
            .list_queued_roots(&thread("a"))
            .await
            .context("a queue after")?;
        assert_eq!(a_queued.len(), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_rejects_non_root_or_non_pending_input() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        // Wrong kind: tool runtime.
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;
        let tool =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("tool")?;
        let err = store.submit_root_turn(tool).await.unwrap_err();
        assert!(
            err.to_string().contains("expected root_turn"),
            "unexpected: {err}"
        );

        // Wrong status: Queued (caller trying to pre-queue).
        let mut queued = fresh_root_at("t2", 0);
        queued.status = TaskStatus::Queued;
        queued.updated_at = queued.created_at;
        let err = store.submit_root_turn(queued).await.unwrap_err();
        assert!(
            err.to_string().contains("must start in Pending"),
            "unexpected: {err}"
        );

        // Wrong attempt counter: pretending to re-submit.
        let mut retried = fresh_root("t3");
        retried.attempt = 1;
        let err = store.submit_root_turn(retried).await.unwrap_err();
        assert!(
            err.to_string().contains("attempt == 0"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_queued_roots_is_deterministic_even_with_same_timestamp() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let blocker = fresh_root_at("t1", 0);
        store.submit_root_turn(blocker).await.context("blocker")?;

        // Submit two roots with the exact same `created_at` — FIFO
        // order must still be deterministic, tie-broken by `id`.
        let a = fresh_root_at("t1", 5);
        let b = fresh_root_at("t1", 5);
        store.submit_root_turn(a.clone()).await.context("a")?;
        store.submit_root_turn(b.clone()).await.context("b")?;

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 2);
        // Order must match the ascending `id` order for reproducible
        // scheduling at sub-microsecond tiebreaks.
        let (lo, hi) = if a.id <= b.id { (&a, &b) } else { (&b, &a) };
        assert_eq!(queued[0].id, lo.id);
        assert_eq!(queued[1].id, hi.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_root_cancellation_removes_it_from_queue() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let blocker = fresh_root_at("t1", 0);
        let queued = fresh_root_at("t1", 1);
        store.submit_root_turn(blocker).await.context("blocker")?;
        let admitted = store
            .submit_root_turn(queued.clone())
            .await
            .context("queued")?;
        assert_eq!(admitted.status, TaskStatus::Queued);

        // Cancel the queued row directly: the queue index must drop it.
        let cancelled = admitted.cancel(t_plus(2)).context("cancel")?;
        store
            .update(cancelled.clone())
            .await
            .context("update cancelled")?;

        let remaining = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queue after cancel")?;
        assert!(remaining.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_adding_another_blocking_root_bypassing_submit() -> Result<()> {
        // `submit_root_turn` is the FIFO admission entry point, but the
        // raw `insert` path must still enforce the partial-unique
        // invariant so callers that bypass `submit_root_turn` can't
        // accidentally push two blocking roots onto the same thread.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store
            .submit_root_turn(first)
            .await
            .context("submit first")?;

        let second = fresh_root("t1");
        let err = store.insert(second).await.unwrap_err();
        assert!(
            err.to_string().contains("already has active root"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_allows_raw_queued_root_next_to_blocking_root() -> Result<()> {
        // Even outside `submit_root_turn`, a caller with a pre-queued
        // row must be able to insert it alongside a blocking root,
        // because `Queued` does not occupy the slot. This is how
        // recovery paths will rehydrate queued roots from durable
        // storage after a crash.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        store
            .submit_root_turn(first)
            .await
            .context("submit first")?;

        // Build a queued row by hand (mirrors the shape after
        // `admit_as_queued`): Pending → Queued.
        let raw_second = fresh_root_at("t1", 1);
        let queued_second = raw_second
            .admit_as_queued(t_plus(1))
            .context("admit queued")?;
        store
            .insert(queued_second.clone())
            .await
            .context("insert queued")?;

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("list queue")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, queued_second.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rebalances_indexes_across_queued_and_pending() -> Result<()> {
        // Cover every transition that mutates a root-turn row's
        // classification across the two Phase 2.2 indexes:
        //
        //   1. Pending (active slot) → Completed (drops both)
        //   2. Queued → Pending (via update, after slot freed)
        //   3. Pending → WaitingOnChildren → Pending (slot stays held)
        //
        // This is a smoke test for the update() rebalancing logic; the
        // higher-level tests above already prove the queue behaviour
        // end-to-end.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);
        store
            .submit_root_turn(first.clone())
            .await
            .context("submit first")?;
        let admitted_second = store
            .submit_root_turn(second.clone())
            .await
            .context("submit second")?;
        assert_eq!(admitted_second.status, TaskStatus::Queued);

        // Drive the first to Completed.
        let running = running_root(&store, first.clone())
            .await
            .context("running")?;
        let done = running.complete(t_plus(5)).context("complete")?;
        store.update(done).await.context("update done")?;

        // The active-root slot is now free; the queued index still holds
        // the second root. Queued rows persist until promotion is
        // explicitly triggered (via `promote_next_queued_root` or via
        // `update` rebalancing in this test).
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert!(active.is_none());
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);

        // Now manually promote the queued row via `update` (rather than
        // `promote_next_queued_root`). This exercises the same index
        // rebalancing that the dedicated promotion path uses.
        let promoted_row = admitted_second
            .promote_to_pending(t_plus(6))
            .context("promote to pending")?;
        store
            .update(promoted_row.clone())
            .await
            .context("update promoted")?;

        // After the manual promotion, the slot must be held and the
        // queue must be empty.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active after")?
            .context("slot held")?;
        assert_eq!(active.id, second.id);
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued after")?;
        assert!(queued.is_empty());

        // Exercise Pending → WaitingOnChildren → Pending.
        let running2 = running_root(&store, active).await.context("running 2")?;
        let waiting = running2
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t1"),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(7),
            )
            .context("wait")?;
        store
            .update(waiting.clone())
            .await
            .context("update waiting")?;

        // The slot is still held (WaitingOnChildren is blocking).
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active waiting")?
            .context("still held")?;
        assert_eq!(active.status, TaskStatus::WaitingOnChildren);

        let resolved = waiting.child_resolved(t_plus(8)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::Pending);
        store.update(resolved).await.context("update resolved")?;

        // And the slot is *still* held — Pending blocks admission.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active final")?
            .context("still held")?;
        assert_eq!(active.status, TaskStatus::Pending);
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.3 — Runnable acquisition, lease ownership, heartbeats
    //
    // ──────────────────────────────────────────────────────────────
    //
    // These tests pin down Phase 2.3's acceptance criteria:
    //
    // * Two workers cannot acquire the same task, either by id or via
    //   the scan-and-claim `acquire_next_runnable` path.
    // * Heartbeats from the wrong worker / wrong lease fail and do
    //   not mutate the row.
    // * Waiting, queued, and terminal tasks are never treated as
    //   runnable work — neither by targeted `try_acquire_task` nor by
    //   the scanning path.
    // * The expiry sweep releases leases whose `lease_expires_at <=
    //   now`, leaves still-live leases alone, and restores the row to
    //   `Pending` so a new worker can acquire it.

    /// Build a fresh root turn with a generous retry budget so the
    /// acquisition tests can exercise the CAS path without fighting
    /// the retry-budget guard.
    fn acquirable_root(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t_plus(secs), 5)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_claims_pending_row_and_stamps_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        let claimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claim should succeed")?;
        assert_eq!(claimed.status, TaskStatus::Running);
        assert_eq!(claimed.worker_id.as_ref().map(WorkerId::as_str), Some("w1"));
        assert_eq!(claimed.lease_id.as_ref().map(LeaseId::as_str), Some("l1"));
        assert_eq!(claimed.lease_expires_at, Some(t_plus(60)));
        assert_eq!(claimed.last_heartbeat_at, Some(t_plus(1)));
        assert_eq!(claimed.attempt, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_is_exclusive_across_two_workers() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        // Worker A wins the race.
        let a = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-a"),
                LeaseId::from_string("l-a"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("claim a")?
            .context("a claims")?;
        assert_eq!(a.worker_id.as_ref().map(WorkerId::as_str), Some("w-a"));

        // Worker B tries to claim the same row and must observe None,
        // because the row is no longer Pending.
        let b = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-b"),
                LeaseId::from_string("l-b"),
                t_plus(60),
                t_plus(2),
            )
            .await
            .context("claim b")?;
        assert!(b.is_none(), "second worker should not claim: {b:?}");

        // The persisted row is still owned by worker A.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(
            persisted.worker_id.as_ref().map(WorkerId::as_str),
            Some("w-a")
        );
        assert_eq!(
            persisted.lease_id.as_ref().map(LeaseId::as_str),
            Some("l-a")
        );
        assert_eq!(persisted.attempt, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_returns_none_for_missing_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let ghost = AgentTaskId::new();
        let result = store
            .try_acquire_task(
                &ghost,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?;
        assert!(result.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_refuses_waiting_on_children_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t-wait", 0);
        let id = root.id.clone();
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit")?;
        let running = root
            .mark_running(
                WorkerId::from_string("w-setup"),
                LeaseId::from_string("l-setup"),
                t_plus(60),
                t_plus(1),
            )
            .context("setup running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let waiting = running
            .wait_on_children(
                2,
                SuspensionPayload {
                    continuation: sample_continuation("t-wait"),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(2),
            )
            .context("wait_on_children")?;
        store.update(waiting).await.context("update waiting")?;

        let claim = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("acquire waiting")?;
        assert!(
            claim.is_none(),
            "WaitingOnChildren must never be treated as runnable"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_refuses_awaiting_confirmation_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t-await", 0);
        let id = root.id.clone();
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit")?;
        let running = root
            .mark_running(
                WorkerId::from_string("w-setup"),
                LeaseId::from_string("l-setup"),
                t_plus(60),
                t_plus(1),
            )
            .context("setup running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let awaiting = running
            .await_confirmation(sample_continuation("t-await"), None, t_plus(2))
            .context("await_confirmation")?;
        store.update(awaiting).await.context("update awaiting")?;

        let claim = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("acquire awaiting")?;
        assert!(
            claim.is_none(),
            "AwaitingConfirmation must never be treated as runnable"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_refuses_terminal_rows() -> Result<()> {
        for (name, secs) in [("t-done", 0i64), ("t-failed", 1), ("t-cancelled", 2)] {
            let store = InMemoryAgentTaskStore::new();
            let root = acquirable_root(name, secs);
            let id = root.id.clone();
            store
                .submit_root_turn(root.clone())
                .await
                .context("submit")?;
            let running = root
                .mark_running(
                    WorkerId::from_string("w-setup"),
                    LeaseId::from_string("l-setup"),
                    t_plus(60),
                    t_plus(secs + 1),
                )
                .context("setup running")?;
            store
                .update(running.clone())
                .await
                .context("update running")?;
            let terminal = match name {
                "t-done" => running.complete(t_plus(secs + 2)).context("complete")?,
                "t-failed" => running
                    .fail("boom".into(), t_plus(secs + 2))
                    .context("fail")?,
                "t-cancelled" => running.cancel(t_plus(secs + 2)).context("cancel")?,
                _ => unreachable!(),
            };
            store.update(terminal).await.context("update terminal")?;
            let claim = store
                .try_acquire_task(
                    &id,
                    WorkerId::from_string("w-new"),
                    LeaseId::from_string("l-new"),
                    t_plus(300),
                    t_plus(secs + 3),
                )
                .await
                .context("acquire terminal")?;
            assert!(
                claim.is_none(),
                "{name}: terminal status must never be treated as runnable"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_refuses_queued_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = acquirable_root("t-queued", 0);
        let second = acquirable_root("t-queued", 1);
        let second_id = second.id.clone();
        store
            .submit_root_turn(first)
            .await
            .context("submit first")?;
        store
            .submit_root_turn(second)
            .await
            .context("submit second")?;

        let claim = store
            .try_acquire_task(
                &second_id,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(300),
                t_plus(2),
            )
            .await
            .context("acquire queued")?;
        assert!(claim.is_none(), "Queued must never be treated as runnable");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_acquire_task_fails_closed_exhausted_rows_and_returns_none() -> Result<()> {
        // Phase 2.5 replaces the Phase 2.3 "loud error on
        // retry exhaustion" placeholder: the store now fails the row
        // closed atomically and returns `Ok(None)`, the same signal it
        // returns for any other non-runnable row. This guarantees no
        // caller has to special-case `AttemptExceedsMax` anymore.
        let store = InMemoryAgentTaskStore::new();
        // max_attempts = 1: a single acquire is legal, the second
        // attempt (after release) must fail closed because the budget
        // is used up.
        let root = AgentTask::new_root_turn(thread("t1"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        let claimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("first claim")?
            .context("first claim exists")?;

        let released = claimed.release_lease(t_plus(2)).context("release")?;
        store.update(released).await.context("update released")?;

        // attempt == 1, max == 1 → Phase 2.5 fails the row closed
        // under the store's write lock and returns `Ok(None)` —
        // identical to the signal for any other non-runnable row.
        let result = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w2"),
                LeaseId::from_string("l2"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("second claim")?;
        assert!(result.is_none(), "exhausted row must return None");

        // The row is now terminal with the canonical Phase 2.5
        // failure prefix, no lease, and the failed `completed_at`
        // timestamp set.
        let row = store.get(&id).await.context("get")?.context("row exists")?;
        assert_eq!(row.status, TaskStatus::Failed);
        assert!(row.worker_id.is_none());
        assert!(row.lease_id.is_none());
        assert_eq!(row.completed_at, Some(t_plus(3)));
        let message = row.last_error.as_deref().expect("last_error set");
        assert!(
            message.starts_with("retry_budget_exhausted:"),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_picks_oldest_pending_first() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        // Three roots on different threads so they can all be Pending
        // simultaneously (the same-thread FIFO queue only allows one
        // active root per thread).
        let a = acquirable_root("a", 0);
        let b = acquirable_root("b", 1);
        let c = acquirable_root("c", 2);
        store.submit_root_turn(a.clone()).await.context("a")?;
        store.submit_root_turn(b.clone()).await.context("b")?;
        store.submit_root_turn(c.clone()).await.context("c")?;

        let first = store
            .acquire_next_runnable(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(5),
            )
            .await
            .context("acquire 1")?
            .context("first claim")?;
        assert_eq!(first.id, a.id, "oldest created_at should win");

        let second = store
            .acquire_next_runnable(
                WorkerId::from_string("w2"),
                LeaseId::from_string("l2"),
                t_plus(60),
                t_plus(6),
            )
            .await
            .context("acquire 2")?
            .context("second claim")?;
        assert_eq!(second.id, b.id);

        let third = store
            .acquire_next_runnable(
                WorkerId::from_string("w3"),
                LeaseId::from_string("l3"),
                t_plus(60),
                t_plus(7),
            )
            .await
            .context("acquire 3")?
            .context("third claim")?;
        assert_eq!(third.id, c.id);

        // Pool drained: a fourth scan returns None.
        let none = store
            .acquire_next_runnable(
                WorkerId::from_string("w4"),
                LeaseId::from_string("l4"),
                t_plus(60),
                t_plus(8),
            )
            .await
            .context("acquire 4")?;
        assert!(none.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_skips_queued_waiting_and_terminal_rows() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        // Blocking root on thread A (Pending, will be acquired).
        let a_root = acquirable_root("a", 0);
        let a_id = a_root.id.clone();
        store.submit_root_turn(a_root).await.context("submit a")?;

        // Queue a second root behind A — it is Queued, must not be
        // scanned.
        let a_queued = acquirable_root("a", 1);
        store
            .submit_root_turn(a_queued.clone())
            .await
            .context("submit a-queued")?;

        // Thread B: put its root into WaitingOnChildren so the whole
        // tree is a waiting root + a runnable child. The child must
        // be acquired by the scan, the root must be skipped.
        let b_root = acquirable_root("b", 0);
        store
            .submit_root_turn(b_root.clone())
            .await
            .context("submit b")?;
        let b_running = b_root
            .mark_running(
                WorkerId::from_string("w-setup"),
                LeaseId::from_string("l-setup"),
                t_plus(60),
                t_plus(1),
            )
            .context("setup running")?;
        store
            .update(b_running.clone())
            .await
            .context("update b running")?;
        let b_child = AgentTask::new_child(&b_running, TaskKind::ToolRuntime, t_plus(2), 1)
            .context("child")?;
        let b_child_id = b_child.id.clone();
        store
            .insert(b_child.clone())
            .await
            .context("insert child")?;
        let b_waiting = b_running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation("b"),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(3),
            )
            .context("wait_on_children")?;
        store.update(b_waiting).await.context("update waiting")?;

        // Thread C: a terminal root, must not be scanned.
        let c_root = acquirable_root("c", 0);
        store
            .submit_root_turn(c_root.clone())
            .await
            .context("submit c")?;
        let c_running = c_root
            .mark_running(
                WorkerId::from_string("w-setup"),
                LeaseId::from_string("l-setup"),
                t_plus(60),
                t_plus(1),
            )
            .context("c running")?;
        store.update(c_running.clone()).await.context("update c")?;
        let c_done = c_running.complete(t_plus(2)).context("complete")?;
        store.update(c_done).await.context("update c done")?;

        // First scan-and-claim: the oldest runnable row is thread A's
        // root (created_at = 0). Queued / waiting / terminal must all
        // be invisible.
        let first = store
            .acquire_next_runnable(
                WorkerId::from_string("w-1"),
                LeaseId::from_string("l-1"),
                t_plus(60),
                t_plus(10),
            )
            .await
            .context("acquire 1")?
            .context("first claim")?;
        assert_eq!(first.id, a_id);
        assert_eq!(first.kind, TaskKind::RootTurn);

        // Second scan-and-claim: next runnable row is the tool child
        // (created_at = 2). Root turn and tool child share the same
        // acquire path, as the issue requires.
        let second = store
            .acquire_next_runnable(
                WorkerId::from_string("w-2"),
                LeaseId::from_string("l-2"),
                t_plus(60),
                t_plus(11),
            )
            .await
            .context("acquire 2")?
            .context("second claim")?;
        assert_eq!(second.id, b_child_id);
        assert_eq!(second.kind, TaskKind::ToolRuntime);

        // Third scan-and-claim: nothing left. The queued A row, the
        // waiting B root, and the terminal C row are all invisible.
        let none = store
            .acquire_next_runnable(
                WorkerId::from_string("w-3"),
                LeaseId::from_string("l-3"),
                t_plus(60),
                t_plus(12),
            )
            .await
            .context("acquire 3")?;
        assert!(none.is_none(), "non-runnable rows must not be scanned");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_two_workers_one_task_yields_exactly_one_winner() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        // The trait's write lock serialises the two calls. The first
        // wins, the second observes `None`.
        let a = store
            .acquire_next_runnable(
                WorkerId::from_string("w-a"),
                LeaseId::from_string("l-a"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire a")?
            .context("a wins")?;
        assert_eq!(a.id, id);
        assert_eq!(a.worker_id.as_ref().map(WorkerId::as_str), Some("w-a"));

        let b = store
            .acquire_next_runnable(
                WorkerId::from_string("w-b"),
                LeaseId::from_string("l-b"),
                t_plus(60),
                t_plus(2),
            )
            .await
            .context("acquire b")?;
        assert!(b.is_none(), "second worker must not claim: {b:?}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn heartbeat_task_bumps_timestamps_under_owner_cas() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let claimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claim")?;
        assert_eq!(claimed.last_heartbeat_at, Some(t_plus(1)));
        assert_eq!(claimed.lease_expires_at, Some(t_plus(60)));

        let refreshed = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(180),
                None,
                t_plus(30),
            )
            .await
            .context("heartbeat")?;
        assert_eq!(refreshed.last_heartbeat_at, Some(t_plus(30)));
        assert_eq!(refreshed.lease_expires_at, Some(t_plus(180)));

        // Heartbeat mutates the persisted row, not just the return
        // value.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(persisted.lease_expires_at, Some(t_plus(180)));
        assert_eq!(persisted.last_heartbeat_at, Some(t_plus(30)));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn heartbeat_task_rejects_wrong_worker_or_wrong_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claim")?;

        // Wrong worker: imposter tries to heartbeat a row it does not
        // own. The call fails loud, and the persisted row is
        // unchanged.
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w-imposter"),
                &LeaseId::from_string("l1"),
                t_plus(120),
                None,
                t_plus(10),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("worker_id") || message.contains("WorkerMismatch"),
            "unexpected: {message}"
        );

        let after_wrong_worker = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(after_wrong_worker.lease_expires_at, Some(t_plus(60)));
        assert_eq!(after_wrong_worker.last_heartbeat_at, Some(t_plus(1)));

        // Wrong lease: original worker id, but the lease has been
        // rotated (e.g. after an expiry sweep re-acquired the row).
        // CAS must fail on the lease dimension too.
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l-stale"),
                t_plus(120),
                None,
                t_plus(10),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("lease_id") || message.contains("LeaseMismatch"),
            "unexpected: {message}"
        );

        let after_wrong_lease = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(after_wrong_lease.lease_expires_at, Some(t_plus(60)));
        assert_eq!(after_wrong_lease.last_heartbeat_at, Some(t_plus(1)));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn heartbeat_task_rejects_non_running_and_missing_rows() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        // Missing row.
        let ghost = AgentTaskId::new();
        let err = store
            .heartbeat_task(
                &ghost,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(120),
                None,
                t_plus(10),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("does not exist"), "unexpected: {message}");

        // Pending row (never acquired).
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(120),
                None,
                t_plus(10),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("invalid transition") || message.contains("InvalidTransition"),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn release_expired_leases_sweeps_expired_rows_and_leaves_live_ones() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        // Three roots on three threads, each with a different lease
        // expiry: 10, 20, 30.
        let a = acquirable_root("a", 0);
        let b = acquirable_root("b", 0);
        let c = acquirable_root("c", 0);
        let (a_id, b_id, c_id) = (a.id.clone(), b.id.clone(), c.id.clone());
        store.submit_root_turn(a).await.context("a")?;
        store.submit_root_turn(b).await.context("b")?;
        store.submit_root_turn(c).await.context("c")?;

        store
            .try_acquire_task(
                &a_id,
                WorkerId::from_string("w-a"),
                LeaseId::from_string("l-a"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire a")?
            .context("a claimed")?;
        store
            .try_acquire_task(
                &b_id,
                WorkerId::from_string("w-b"),
                LeaseId::from_string("l-b"),
                t_plus(20),
                t_plus(1),
            )
            .await
            .context("acquire b")?
            .context("b claimed")?;
        store
            .try_acquire_task(
                &c_id,
                WorkerId::from_string("w-c"),
                LeaseId::from_string("l-c"),
                t_plus(30),
                t_plus(1),
            )
            .await
            .context("acquire c")?
            .context("c claimed")?;

        // Sweep at `now = t+15`: `a` expired, `b` and `c` still live.
        let released = store
            .release_expired_leases(t_plus(15))
            .await
            .context("sweep 15")?;
        assert_eq!(released, vec![RecoveryRecord::requeued(a_id.clone())]);

        // `a` is now Pending and has no lease fields, but retains its
        // attempt counter.
        let a_row = store
            .get(&a_id)
            .await
            .context("get a")?
            .context("a exists")?;
        assert_eq!(a_row.status, TaskStatus::Pending);
        assert!(a_row.worker_id.is_none());
        assert!(a_row.lease_id.is_none());
        assert!(a_row.lease_expires_at.is_none());
        assert!(a_row.last_heartbeat_at.is_none());
        assert_eq!(a_row.attempt, 1, "failed attempt counts against budget");

        // `b` and `c` are still Running and still owned by their
        // original workers.
        let b_row = store
            .get(&b_id)
            .await
            .context("get b")?
            .context("b exists")?;
        assert_eq!(b_row.status, TaskStatus::Running);
        assert_eq!(b_row.worker_id.as_ref().map(WorkerId::as_str), Some("w-b"));
        let c_row = store
            .get(&c_id)
            .await
            .context("get c")?
            .context("c exists")?;
        assert_eq!(c_row.status, TaskStatus::Running);
        assert_eq!(c_row.worker_id.as_ref().map(WorkerId::as_str), Some("w-c"));

        // A second sweep at `now = t+25` catches only `b`.
        let released = store
            .release_expired_leases(t_plus(25))
            .await
            .context("sweep 25")?;
        assert_eq!(released, vec![RecoveryRecord::requeued(b_id.clone())]);

        // Empty sweep: still-live `c` survives.
        let released = store
            .release_expired_leases(t_plus(25))
            .await
            .context("sweep idempotent")?;
        assert!(released.is_empty());
        Ok(())
    }

    /// Issue #299 round 4: every backend reclaims at most
    /// [`LEASE_RELEASE_BATCH`] rows per call (the SQL backends via
    /// `LIMIT`, this reference store via the same cap) so the host's
    /// drain loop has one uniform "keep calling until a pass returns
    /// fewer than the batch" contract.
    #[tokio::test(flavor = "multi_thread")]
    async fn release_expired_leases_caps_each_call_at_the_batch_size() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let backlog = LEASE_RELEASE_BATCH + 44;
        for index in 0..backlog {
            let root = AgentTask::new_root_turn(thread(&format!("t-batch-{index}")), t0(), 3);
            let id = root.id.clone();
            store.submit_root_turn(root).await.context("submit")?;
            store
                .try_acquire_task(
                    &id,
                    WorkerId::from_string(format!("w-{index}")),
                    LeaseId::new(),
                    t_plus(5),
                    t_plus(1),
                )
                .await
                .context("acquire")?
                .context("row claims")?;
        }

        // Every lease is expired at t+10; one call reclaims exactly one
        // batch, the next reclaims the remainder, the third is empty.
        let first = store
            .release_expired_leases(t_plus(10))
            .await
            .context("first sweep")?;
        assert_eq!(first.len(), LEASE_RELEASE_BATCH);
        let second = store
            .release_expired_leases(t_plus(10))
            .await
            .context("second sweep")?;
        assert_eq!(second.len(), 44);
        let third = store
            .release_expired_leases(t_plus(10))
            .await
            .context("third sweep")?;
        assert!(third.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn release_expired_leases_lets_a_new_worker_reacquire_the_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        // Give the row two attempts so we can prove the sweep path
        // leaves the row runnable for a fresh worker.
        let root = AgentTask::new_root_turn(thread("t1"), t0(), 2);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("first acquire")?
            .context("dead worker claims")?;

        // The dead worker never heartbeats. Sweep at `t+15` boots it.
        let swept = store
            .release_expired_leases(t_plus(15))
            .await
            .context("sweep")?;
        assert_eq!(swept.len(), 1);

        // A fresh worker acquires the row and bumps the attempt
        // counter.
        let reclaimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(120),
                t_plus(20),
            )
            .await
            .context("second acquire")?
            .context("new worker claims")?;
        assert_eq!(
            reclaimed.worker_id.as_ref().map(WorkerId::as_str),
            Some("w-new")
        );
        assert_eq!(
            reclaimed.lease_id.as_ref().map(LeaseId::as_str),
            Some("l-new")
        );
        assert_eq!(reclaimed.attempt, 2);

        // The original worker's stale heartbeat now fails with a lease
        // mismatch — the acceptance-criterion scenario from the issue.
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w-dead"),
                &LeaseId::from_string("l-dead"),
                t_plus(300),
                None,
                t_plus(21),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("worker_id") || message.contains("WorkerMismatch"),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn heartbeat_extends_lease_past_previous_expiry() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claim")?;

        // Heartbeat bumps expiry to t+100 before the sweep fires.
        store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                t_plus(100),
                None,
                t_plus(5),
            )
            .await
            .context("heartbeat")?;

        // The sweep at t+15 now treats the lease as live.
        let released = store
            .release_expired_leases(t_plus(15))
            .await
            .context("sweep")?;
        assert!(released.is_empty(), "heartbeat should extend lease");

        // The row is still Running and still owned by w1.
        let row = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(row.lease_expires_at, Some(t_plus(100)));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_runtime_children_share_the_same_lease_path_as_roots() -> Result<()> {
        // Phase 2.3 requires one acquisition and lease model that
        // prevents double ownership across task kinds. This test
        // drives a tool-runtime child through try_acquire /
        // heartbeat / release_expired so any divergence between the
        // root and child paths would show up here.
        let store = InMemoryAgentTaskStore::new();
        let root = acquirable_root("t1", 0);
        store.submit_root_turn(root.clone()).await.context("root")?;
        let running = root
            .mark_running(
                WorkerId::from_string("w-root"),
                LeaseId::from_string("l-root"),
                t_plus(60),
                t_plus(1),
            )
            .context("root running")?;
        store.update(running.clone()).await.context("update root")?;
        let child =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 2).context("child")?;
        let child_id = child.id.clone();
        store.insert(child.clone()).await.context("insert child")?;

        // Park the root in WaitingOnChildren so its lease is dropped —
        // we want the sweep below to act on the tool child only, not
        // accidentally on an expired root lease.
        let waiting_root = running
            .wait_on_children(
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t1"),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(3),
            )
            .context("root waiting")?;
        store
            .update(waiting_root)
            .await
            .context("update root waiting")?;

        // Targeted claim of the child.
        let claimed = store
            .try_acquire_task(
                &child_id,
                WorkerId::from_string("w-tool"),
                LeaseId::from_string("l-tool"),
                t_plus(30),
                t_plus(4),
            )
            .await
            .context("acquire child")?
            .context("child claimed")?;
        assert_eq!(claimed.kind, TaskKind::ToolRuntime);
        assert_eq!(claimed.status, TaskStatus::Running);

        // A second worker cannot claim the same tool child.
        let b = store
            .try_acquire_task(
                &child_id,
                WorkerId::from_string("w-imposter"),
                LeaseId::from_string("l-imposter"),
                t_plus(30),
                t_plus(5),
            )
            .await
            .context("second child acquire")?;
        assert!(b.is_none());

        // Heartbeat CAS applies the same way.
        let refreshed = store
            .heartbeat_task(
                &child_id,
                &WorkerId::from_string("w-tool"),
                &LeaseId::from_string("l-tool"),
                t_plus(200),
                None,
                t_plus(6),
            )
            .await
            .context("heartbeat child")?;
        assert_eq!(refreshed.lease_expires_at, Some(t_plus(200)));

        // Expire the heartbeat by advancing the sweep past it. The
        // child returns to Pending so a new worker can pick it up.
        let expired = store
            .release_expired_leases(t_plus(300))
            .await
            .context("sweep child")?;
        assert_eq!(expired, vec![RecoveryRecord::requeued(child_id.clone())]);
        let after = store
            .get(&child_id)
            .await
            .context("get child")?
            .context("child exists")?;
        assert_eq!(after.status, TaskStatus::Pending);
        assert!(after.worker_id.is_none());
        assert_eq!(after.attempt, 1);

        // Scan-and-claim picks the child up again (the root is still
        // WaitingOnChildren, so only the child is runnable).
        let reclaimed = store
            .acquire_next_runnable(
                WorkerId::from_string("w-fresh"),
                LeaseId::from_string("l-fresh"),
                t_plus(500),
                t_plus(301),
            )
            .await
            .context("scan child")?
            .context("fresh claim")?;
        assert_eq!(reclaimed.id, child_id);
        assert_eq!(reclaimed.attempt, 2);
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.4 — Parent waiting states, confirmation pause/resume,
    // and typed durable task state
    // ──────────────────────────────────────────────────────────────
    //
    // Acceptance criteria these tests pin down:
    //
    // * A paused parent **releases its lease** and is invisible to
    //   `try_acquire_task` / `acquire_next_runnable` and to the
    //   lease-expiry sweep.
    // * Confirmation-paused tasks persist the typed
    //   [`TaskState::AwaitingConfirmation`] payload (continuation +
    //   optional prepared listen/execute operation) so the resume
    //   path has everything it needs to either execute or cancel the
    //   staged operation.
    // * Pause / resume transitions are journal-guarded: only the
    //   worker that holds the lease may pause, and resume from
    //   confirmation rejects rows in any other status.
    // * Typed-state round-trips through the store (insert / update /
    //   get / list_by_status) without losing the embedded continuation
    //   on either pause variant.

    /// Helper: drive a fresh root through `submit_root_turn`,
    /// acquire its lease via `try_acquire_task`, and return the
    /// claimed row alongside its `(WorkerId, LeaseId)` tuple. Used
    /// by the pause-path tests so the journal-guarded pause helper
    /// is exercised against a row that is genuinely owned by a
    /// worker, not just hand-crafted.
    async fn submitted_and_claimed_root(
        store: &InMemoryAgentTaskStore,
        thread_name: &str,
        worker: &str,
        lease: &str,
    ) -> Result<(AgentTask, WorkerId, LeaseId)> {
        let root = AgentTask::new_root_turn(thread(thread_name), t_plus(0), 5);
        let id = root.id.clone();
        store
            .submit_root_turn(root)
            .await
            .context("submit root for pause test")?;
        let claimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string(worker),
                LeaseId::from_string(lease),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire root for pause test")?
            .context("acquire returned None")?;
        Ok((
            claimed,
            WorkerId::from_string(worker),
            LeaseId::from_string(lease),
        ))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_children_drops_lease_and_persists_typed_state() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-pause-children", "w1", "l1").await?;
        let id = claimed.id.clone();

        let paused = store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                2,
                SuspensionPayload {
                    continuation: sample_continuation("t-pause-children"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("pause_on_children")?;

        // Status flipped, lease dropped, typed payload populated.
        assert_eq!(paused.status, TaskStatus::WaitingOnChildren);
        assert_eq!(paused.pending_child_count, 2);
        assert!(paused.worker_id.is_none(), "lease must be dropped");
        assert!(paused.lease_id.is_none());
        assert!(paused.lease_expires_at.is_none());
        assert!(paused.last_heartbeat_at.is_none());
        assert!(matches!(paused.state, TaskState::WaitingOnChildren { .. }));
        assert!(paused.state.continuation().is_some());

        // The persisted row matches what `pause_on_children` returned.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(persisted.status, TaskStatus::WaitingOnChildren);
        assert!(persisted.worker_id.is_none());
        assert!(persisted.state.continuation().is_some());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_confirmation_drops_lease_and_persists_typed_state() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-pause-confirm", "w1", "l1").await?;
        let id = claimed.id.clone();

        let paused = store
            .pause_on_confirmation(
                &id,
                &worker,
                &lease,
                sample_continuation("t-pause-confirm"),
                Some(sample_prepared_op()),
                t_plus(2),
            )
            .await
            .context("pause_on_confirmation")?;

        assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);
        assert!(paused.worker_id.is_none(), "lease must be dropped");
        assert!(paused.lease_id.is_none());
        assert!(paused.lease_expires_at.is_none());
        assert!(matches!(
            paused.state,
            TaskState::AwaitingConfirmation { .. }
        ));
        assert!(paused.state.continuation().is_some());
        let op = paused
            .state
            .prepared_operation()
            .expect("prepared op present");
        assert_eq!(op.operation_id, "op-store");

        // Persisted row carries both the continuation and the
        // prepared operation.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert!(persisted.state.continuation().is_some());
        assert!(persisted.state.prepared_operation().is_some());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn paused_parent_is_invisible_to_targeted_acquire() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-invisible-target", "w1", "l1").await?;
        let id = claimed.id.clone();
        store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-invisible-target"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("pause")?;

        // A fresh worker tries to claim the same id directly. The
        // CAS guard inside `try_acquire_task` must refuse because
        // the row is no longer `Pending`.
        let claim = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("targeted acquire")?;
        assert!(
            claim.is_none(),
            "paused parent must not be acquirable by id"
        );

        // Same check for the AwaitingConfirmation pause path.
        let (claimed2, worker2, lease2) =
            submitted_and_claimed_root(&store, "t-invisible-conf", "w2", "l2").await?;
        let id2 = claimed2.id.clone();
        store
            .pause_on_confirmation(
                &id2,
                &worker2,
                &lease2,
                sample_continuation("t-invisible-conf"),
                None,
                t_plus(3),
            )
            .await
            .context("pause confirm")?;
        let claim = store
            .try_acquire_task(
                &id2,
                WorkerId::from_string("w-new"),
                LeaseId::from_string("l-new"),
                t_plus(180),
                t_plus(4),
            )
            .await
            .context("targeted acquire confirm")?;
        assert!(
            claim.is_none(),
            "confirmation-paused row must not be acquirable by id"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn paused_parent_is_invisible_to_scan_acquire_and_expiry_sweep() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-scan-invisible", "w1", "l1").await?;
        let id = claimed.id.clone();
        store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-scan-invisible"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("pause")?;

        // The runnable scan must observe an empty pool — the paused
        // parent is the only row in the store and it must not be
        // visible to `acquire_next_runnable`.
        let scanned = store
            .acquire_next_runnable(
                WorkerId::from_string("w-scan"),
                LeaseId::from_string("l-scan"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("scan")?;
        assert!(
            scanned.is_none(),
            "paused parent must not appear in runnable scan"
        );

        // The expiry sweep walks the lease-expiry index, which only
        // contains `Running` rows. A paused row dropped its lease, so
        // even at `now = t+1_000_000` the sweep must not touch it.
        let swept = store
            .release_expired_leases(t_plus(1_000_000))
            .await
            .context("sweep")?;
        assert!(
            swept.is_empty(),
            "paused parent must not be in lease-expiry index"
        );

        // The row is still WaitingOnChildren after the sweep.
        let after = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(after.status, TaskStatus::WaitingOnChildren);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_children_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-cas", "w1", "l1").await?;
        let id = claimed.id.clone();

        // Wrong worker.
        let err = store
            .pause_on_children(
                &id,
                &WorkerId::from_string("w-imposter"),
                &lease,
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-cas"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("worker mismatch"), "unexpected: {message}");

        // Wrong lease.
        let err = store
            .pause_on_children(
                &id,
                &worker,
                &LeaseId::from_string("l-stale"),
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-cas"),
                    suspended_messages: Vec::new(),
                },
                t_plus(3),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("lease mismatch"), "unexpected: {message}");

        // Persisted row is still Running and unchanged.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert_eq!(
            persisted.worker_id.as_ref().map(WorkerId::as_str),
            Some("w1")
        );
        assert!(
            persisted.state.is_none(),
            "failed pause must not stamp typed state"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_confirmation_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-cas-conf", "w1", "l1").await?;
        let id = claimed.id.clone();

        let err = store
            .pause_on_confirmation(
                &id,
                &WorkerId::from_string("w-imposter"),
                &lease,
                sample_continuation("t-cas-conf"),
                None,
                t_plus(2),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("worker mismatch"), "unexpected: {message}");

        let err = store
            .pause_on_confirmation(
                &id,
                &worker,
                &LeaseId::from_string("l-stale"),
                sample_continuation("t-cas-conf"),
                None,
                t_plus(3),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("lease mismatch"), "unexpected: {message}");

        // Row is still Running.
        let persisted = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(persisted.state.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_children_rejects_non_running_rows() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-pending"), t_plus(0), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        // Row is Pending — pause must refuse.
        let err = store
            .pause_on_children(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-pending"),
                    suspended_messages: Vec::new(),
                },
                t_plus(1),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("not running"), "unexpected: {message}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pause_on_confirmation_rejects_non_running_rows() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-pending-conf"), t_plus(0), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        let err = store
            .pause_on_confirmation(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                sample_continuation("t-pending-conf"),
                None,
                t_plus(1),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("not running"), "unexpected: {message}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resume_from_confirmation_clears_typed_state_and_returns_to_pending() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-resume-conf", "w1", "l1").await?;
        let id = claimed.id.clone();
        store
            .pause_on_confirmation(
                &id,
                &worker,
                &lease,
                sample_continuation("t-resume-conf"),
                Some(sample_prepared_op()),
                t_plus(2),
            )
            .await
            .context("pause")?;

        let (resumed, _prepared) = store
            .resume_from_confirmation(&id, t_plus(3))
            .await
            .context("resume")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        assert!(resumed.state.is_none(), "resume must clear typed state");

        // The persisted row is now Pending and runnable.
        let scanned = store
            .acquire_next_runnable(
                WorkerId::from_string("w-resume"),
                LeaseId::from_string("l-resume"),
                t_plus(120),
                t_plus(4),
            )
            .await
            .context("scan resumed")?
            .context("scan returned row")?;
        assert_eq!(scanned.id, id);
        assert_eq!(scanned.status, TaskStatus::Running);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn approve_confirmation_and_acquire_claims_without_pending_race() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-approve-conf", "w1", "l1").await?;
        let id = claimed.id.clone();
        store
            .pause_on_confirmation(
                &id,
                &worker,
                &lease,
                sample_continuation("t-approve-conf"),
                Some(sample_prepared_op()),
                t_plus(2),
            )
            .await
            .context("pause")?;

        let (approved, prepared) = store
            .approve_confirmation_and_acquire(
                &id,
                WorkerId::from_string("w-approved"),
                LeaseId::from_string("l-approved"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("approve and acquire")?;
        assert_eq!(approved.status, TaskStatus::Running);
        assert_eq!(
            approved.worker_id,
            Some(WorkerId::from_string("w-approved"))
        );
        assert_eq!(approved.lease_id, Some(LeaseId::from_string("l-approved")));

        let prepared = prepared.context("prepared operation missing")?;
        assert_eq!(prepared.operation_id, "op-store");

        let persisted = store
            .get(&id)
            .await
            .context("get approved task")?
            .context("approved task missing")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert_eq!(
            persisted.worker_id,
            Some(WorkerId::from_string("w-approved"))
        );
        assert_eq!(persisted.lease_id, Some(LeaseId::from_string("l-approved")));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resume_from_confirmation_rejects_non_awaiting_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-not-awaiting"), t_plus(0), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        let err = store
            .resume_from_confirmation(&id, t_plus(1))
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("not awaiting confirmation"),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn paused_parent_round_trips_through_get_and_list_by_status() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-round-trip", "w1", "l1").await?;
        let id = claimed.id.clone();
        let paused = store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                1,
                SuspensionPayload {
                    continuation: sample_continuation("t-round-trip"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("pause")?;

        // `get` returns the same typed payload.
        let fetched = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(fetched.id, paused.id);
        assert_eq!(fetched.status, TaskStatus::WaitingOnChildren);
        assert!(fetched.state.continuation().is_some());

        // `list_by_status(WaitingOnChildren)` returns the row.
        let rows = store
            .list_by_status(TaskStatus::WaitingOnChildren)
            .await
            .context("list waiting")?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, id);

        // The status indexes do **not** double-count the row.
        let pending_rows = store
            .list_by_status(TaskStatus::Pending)
            .await
            .context("list pending")?;
        assert!(
            pending_rows.iter().all(|r| r.id != id),
            "paused parent must not show up under Pending"
        );
        let running_rows = store
            .list_by_status(TaskStatus::Running)
            .await
            .context("list running")?;
        assert!(
            running_rows.iter().all(|r| r.id != id),
            "paused parent must not show up under Running"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn awaiting_confirmation_payload_round_trips_through_json() -> Result<()> {
        // The Phase 2.4 acceptance criterion "Confirmation-paused
        // tasks persist the state needed for later resume" must hold
        // through the durable wire form too: a paused row that gets
        // serialised and deserialised must still carry the
        // continuation envelope and prepared operation it was paused
        // with.
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-json-round", "w1", "l1").await?;
        let id = claimed.id.clone();
        let paused = store
            .pause_on_confirmation(
                &id,
                &worker,
                &lease,
                sample_continuation("t-json-round"),
                Some(sample_prepared_op()),
                t_plus(2),
            )
            .await
            .context("pause")?;

        let json = serde_json::to_string(&paused).context("serialize paused row")?;
        let recovered: AgentTask = serde_json::from_str(&json).context("deserialize paused row")?;
        recovered
            .validate()
            .context("recovered paused row must validate")?;
        assert_eq!(recovered.status, TaskStatus::AwaitingConfirmation);
        assert!(matches!(
            recovered.state,
            TaskState::AwaitingConfirmation { .. }
        ));
        assert!(recovered.state.continuation().is_some());
        assert_eq!(
            recovered
                .state
                .prepared_operation()
                .map(|op| op.operation_id.clone()),
            Some("op-store".into())
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn waiting_on_children_payload_round_trips_through_json() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-json-children", "w1", "l1").await?;
        let id = claimed.id.clone();
        let paused = store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                4,
                SuspensionPayload {
                    continuation: sample_continuation("t-json-children"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("pause")?;

        let json = serde_json::to_string(&paused).context("serialize")?;
        let recovered: AgentTask = serde_json::from_str(&json).context("deserialize")?;
        recovered.validate().context("recovered must validate")?;
        assert_eq!(recovered.status, TaskStatus::WaitingOnChildren);
        assert_eq!(recovered.pending_child_count, 4);
        assert!(matches!(
            recovered.state,
            TaskState::WaitingOnChildren { .. }
        ));
        assert!(recovered.state.continuation().is_some());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn validate_rejects_paused_status_with_none_payload() -> Result<()> {
        // Defense in depth: the schema layer must reject any row
        // whose status is paused but whose state field is `None`.
        // This is the durable form of the "no continuation, no
        // resume" guarantee — even a hand-crafted row that bypasses
        // the typed transition helpers cannot land in the journal.
        let mut bad = AgentTask::new_root_turn(thread("t-bad"), t_plus(0), 3);
        bad.status = TaskStatus::WaitingOnChildren;
        bad.pending_child_count = 1;
        bad.state = TaskState::None;
        let err = bad.validate().unwrap_err();
        assert!(
            matches!(err, TaskSchemaError::PausedStatusMissingPayload { .. }),
            "unexpected: {err:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn validate_rejects_non_paused_status_with_payload() -> Result<()> {
        // The opposite half of the invariant: a Pending or Running
        // row that carries a paused payload is also rejected, so a
        // resume bug that forgets to clear the state cannot leak a
        // stale continuation onto a runnable row.
        let mut bad = AgentTask::new_root_turn(thread("t-bad-2"), t_plus(0), 3);
        bad.state = TaskState::WaitingOnChildren {
            continuation: Box::new(sample_continuation("t-bad-2")),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };
        let err = bad.validate().unwrap_err();
        assert!(
            matches!(err, TaskSchemaError::StateStatusMismatch { .. }),
            "unexpected: {err:?}"
        );
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.5 — Retry budget, failure handling, and recovery matrix
    //
    // ──────────────────────────────────────────────────────────────
    //
    // Acceptance criteria these tests pin down:
    //
    // * Retry-budget exhaustion produces a terminal `Failed` row
    //   with a canonical `last_error` prefix, not an endless
    //   requeue loop or a loud `AttemptExceedsMax` bubble-up.
    // * `acquire_next_runnable` skips exhausted heads in-band and
    //   keeps scanning, so one bad row can never poison the whole
    //   worker pool.
    // * `release_expired_leases` returns a `Vec<RecoveryRecord>`
    //   that distinguishes requeued rows from fail-closed rows.
    // * Tool-runtime rows carrying a staged listen/execute
    //   prepared operation are always failed closed on recovery,
    //   because the rewrite has no safe resume contract for them
    //   and a blind requeue would risk double-executing the
    //   external operation.
    // * Duplicate ownership across a requeue + retry cycle is
    //   prevented: an old worker cannot heartbeat or re-lease a
    //   row that has been released or failed-closed by the sweep.

    use crate::journal::recovery::{FailureReason, RecoveryAction, RecoveryRecord};

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_skips_exhausted_head_and_keeps_scanning() -> Result<()> {
        // Two roots on distinct threads, the older one with
        // `max_attempts == 1` already consumed so it must be failed
        // closed, the younger one healthy so it must be claimed.
        let store = InMemoryAgentTaskStore::new();

        let old = AgentTask::new_root_turn(thread("old"), t_plus(0), 1);
        let old_id = old.id.clone();
        store.submit_root_turn(old).await.context("submit old")?;
        // Burn the old row's budget via a claim + release so it
        // ends up back in `Pending` with `attempt == max_attempts`.
        let claimed = store
            .try_acquire_task(
                &old_id,
                WorkerId::from_string("w-old"),
                LeaseId::from_string("l-old"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("old acquire")?
            .context("old claimed")?;
        let released = claimed.release_lease(t_plus(2)).context("release old")?;
        store.update(released).await.context("update old")?;

        let young = AgentTask::new_root_turn(thread("young"), t_plus(100), 3);
        let young_id = young.id.clone();
        store
            .submit_root_turn(young)
            .await
            .context("submit young")?;

        // Scan: the old row is the oldest `created_at`, so it is the
        // FIFO head. Phase 2.5 must fail it closed and keep scanning
        // to the young row.
        let claimed_young = store
            .acquire_next_runnable(
                WorkerId::from_string("w-fresh"),
                LeaseId::from_string("l-fresh"),
                t_plus(500),
                t_plus(200),
            )
            .await
            .context("scan")?
            .context("young claimed")?;
        assert_eq!(claimed_young.id, young_id);
        assert_eq!(claimed_young.status, TaskStatus::Running);

        // The old row is now terminal with the retry-budget
        // fail-closed prefix.
        let old_after = store
            .get(&old_id)
            .await
            .context("get old")?
            .context("old exists")?;
        assert_eq!(old_after.status, TaskStatus::Failed);
        let message = old_after.last_error.as_deref().expect("last_error");
        assert!(
            message.starts_with(FailureReason::RetryBudgetExhausted.error_prefix()),
            "unexpected: {message}"
        );

        // And the runnable index is empty — both rows have left it.
        let inner = store.inner.read().await;
        assert!(inner.runnable_by_created_at.is_empty());
        drop(inner);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_returns_none_when_every_head_is_exhausted() -> Result<()> {
        // Three roots on three threads, every one exhausted. The
        // scan must drain them all to `Failed` and return `None`.
        let store = InMemoryAgentTaskStore::new();
        let mut ids = Vec::new();
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            let secs = i64::try_from(i).context("enumerate fits i64")? * 10;
            let root = AgentTask::new_root_turn(thread(name), t_plus(secs), 1);
            let id = root.id.clone();
            store.submit_root_turn(root).await.context("submit")?;
            let claimed = store
                .try_acquire_task(
                    &id,
                    WorkerId::from_string(format!("w-{name}")),
                    LeaseId::from_string(format!("l-{name}")),
                    t_plus(secs + 50),
                    t_plus(secs + 1),
                )
                .await
                .context("acquire")?
                .context("claimed")?;
            let released = claimed.release_lease(t_plus(secs + 2)).context("release")?;
            store.update(released).await.context("update")?;
            ids.push(id);
        }

        let result = store
            .acquire_next_runnable(
                WorkerId::from_string("w-scan"),
                LeaseId::from_string("l-scan"),
                t_plus(1_000),
                t_plus(900),
            )
            .await
            .context("scan")?;
        assert!(result.is_none(), "every head must be failed closed");

        // Every row is now Failed.
        for id in &ids {
            let row = store.get(id).await.context("get")?.context("exists")?;
            assert_eq!(
                row.status,
                TaskStatus::Failed,
                "expected Failed for {id}, got {:?}",
                row.status
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn release_expired_leases_fails_closed_when_budget_exhausted() -> Result<()> {
        // A row on its last attempt whose lease expires must NOT be
        // requeued — Phase 2.5 fails it closed so the journal never
        // enters an endless requeue loop.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        // attempt == 1, max_attempts == 1 → sweep must fail closed.
        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep")?;
        assert_eq!(
            swept,
            vec![RecoveryRecord::failed_closed(
                id.clone(),
                FailureReason::LeaseExpiredBudgetExhausted,
            )],
        );

        let row = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(row.status, TaskStatus::Failed);
        let message = row.last_error.as_deref().expect("last_error");
        assert!(
            message.starts_with(FailureReason::LeaseExpiredBudgetExhausted.error_prefix()),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn release_expired_leases_requeues_when_budget_remains() -> Result<()> {
        // Positive counterpart: a row with budget still available is
        // requeued (the Phase 2.3 behavior), and the sweep returns
        // `RecoveryRecord::Requeue` for it.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep")?;
        assert_eq!(swept, vec![RecoveryRecord::requeued(id.clone())]);

        let row = store.get(&id).await.context("get")?.context("exists")?;
        assert_eq!(row.status, TaskStatus::Pending);
        assert!(row.last_error.is_none());
        assert_eq!(row.attempt, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_runtime_awaiting_confirmation_with_prepared_op_fails_closed_on_acquire()
    -> Result<()> {
        // Hand-build a tool-runtime child that is parked in
        // `AwaitingConfirmation` with a `Some(prepared_operation)`,
        // then flip it back to `Pending` out-of-band to simulate a
        // buggy resume path that forgot to fail-close it. Phase 2.5
        // must refuse to lease the row — the matrix fires on the
        // unsafe prepared operation regardless of the current
        // status, so the row transitions straight to `Failed` with
        // the unsafe-prepared-operation prefix.
        let store = InMemoryAgentTaskStore::new();

        // Build a fresh parent root (needed to satisfy parent-id
        // cross-row invariants on `insert`).
        let root = acquirable_root("root", 0);
        let root_id = root.id.clone();
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit root")?;
        let running_root = root
            .mark_running(
                WorkerId::from_string("w-root"),
                LeaseId::from_string("l-root"),
                t_plus(60),
                t_plus(1),
            )
            .context("root running")?;
        store
            .update(running_root.clone())
            .await
            .context("update root")?;

        // Tool child: Pending with a hand-installed unsafe
        // prepared operation on its TaskState. The child must be
        // inserted in a valid status first and then mutated, since
        // the `AwaitingConfirmation` flip requires a real transition.
        let child = AgentTask::new_child(&running_root, TaskKind::ToolRuntime, t_plus(2), 3)
            .context("child")?;
        let child_id = child.id.clone();
        store.insert(child.clone()).await.context("insert child")?;

        // Lease the child, park it in AwaitingConfirmation with a
        // prepared operation, then release the lease back to Pending
        // while forging the unsafe state — we bypass the store's
        // safer `pause_on_confirmation` path on purpose so the test
        // exercises the recovery matrix on an actual store row that
        // has drifted into the unsafe shape.
        let _claimed_child = store
            .try_acquire_task(
                &child_id,
                WorkerId::from_string("w-tool"),
                LeaseId::from_string("l-tool"),
                t_plus(30),
                t_plus(3),
            )
            .await
            .context("child acquire")?
            .context("child claimed")?;
        let paused = store
            .pause_on_confirmation(
                &child_id,
                &WorkerId::from_string("w-tool"),
                &LeaseId::from_string("l-tool"),
                sample_continuation("t-forge"),
                Some(sample_prepared_op()),
                t_plus(4),
            )
            .await
            .context("pause")?;
        assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);
        assert!(paused.has_prepared_operation());
        // At this point the child is legitimately awaiting
        // confirmation with an unsafe prepared operation. Any Phase
        // 2.5 call site that looks at it must fail it closed.
        let fail_closed_action = classify_recovery(&paused, RecoveryContext::AcquisitionAttempt);
        assert_eq!(
            fail_closed_action,
            RecoveryAction::FailClosed(FailureReason::UnsafePreparedOperationRecovery),
        );

        // Now simulate a resume bug: manually flip the row back to
        // `Pending` while forcibly keeping the state empty. This is
        // the "drifted row" the matrix is designed to catch. We go
        // through `resume_from_confirmation` so the store rebalances
        // its indexes, then the child is legitimately `Pending` and
        // the unsafe-prepared-operation bit is no longer present —
        // that is the *legal* resume path. The Phase 2.5
        // acquisition classification returns NoAction because the
        // unsafe state was cleared.
        let (resumed, _prepared) = store
            .resume_from_confirmation(&child_id, t_plus(5))
            .await
            .context("resume")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        assert!(!resumed.has_prepared_operation());
        assert_eq!(
            classify_recovery(&resumed, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
            "legal resume path must be safe"
        );

        // Sanity: the parent root is still visible so we are not
        // checking a corrupted store.
        let root_row = store
            .get(&root_id)
            .await
            .context("get root")?
            .context("root exists")?;
        assert_eq!(root_row.status, TaskStatus::Running);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requeued_row_rejects_heartbeat_from_stale_worker_lease() -> Result<()> {
        // Phase 2.5 regression: an old worker whose row has been
        // released by the expiry sweep must not be able to heartbeat
        // it. The `(worker_id, lease_id)` CAS from Phase 2.3 already
        // guards this; the test pins the behavior so a future
        // refactor cannot silently regress it.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep")?;
        assert_eq!(swept.len(), 1);
        assert!(swept[0].action.is_requeue());

        // Stale heartbeat with the old worker + old lease must
        // fail: the row is no longer `Running`, so the row-level
        // CAS rejects it with an invalid-transition error.
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w-dead"),
                &LeaseId::from_string("l-dead"),
                t_plus(300),
                None,
                t_plus(30),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("invalid transition") || message.contains("InvalidTransition"),
            "unexpected: {message}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requeue_owned_task_returns_running_row_to_pending() -> Result<()> {
        // The worker-initiated requeue twin of the sweep: a promoted
        // successor whose commit lost its turn slot re-runs from the
        // fresh head instead of failing terminally.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let worker = WorkerId::from_string("w-collided");
        let lease = LeaseId::from_string("l-collided");
        store
            .try_acquire_task(&id, worker.clone(), lease.clone(), t_plus(10), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;

        let outcome = store
            .requeue_owned_task(&id, &worker, &lease, None, t_plus(2))
            .await
            .context("requeue")?;
        let RequeueOutcome::Requeued(row) = outcome else {
            anyhow::bail!("expected Requeued, got {outcome:?}");
        };
        assert_eq!(row.status, TaskStatus::Pending);
        assert!(row.worker_id.is_none() && row.lease_id.is_none());

        // The row is genuinely runnable again: a fresh worker can
        // re-acquire it, and the old lease can no longer heartbeat.
        let reacquired = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-fresh"),
                LeaseId::from_string("l-fresh"),
                t_plus(30),
                t_plus(3),
            )
            .await
            .context("re-acquire")?
            .context("re-claimed")?;
        assert_eq!(reacquired.status, TaskStatus::Running);
        assert!(
            store
                .heartbeat_task(&id, &worker, &lease, t_plus(60), None, t_plus(4))
                .await
                .is_err(),
            "the requeued-then-reacquired row must reject the old lease",
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requeue_owned_task_reports_not_owned_without_touching_the_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let worker = WorkerId::from_string("w-owner");
        let lease = LeaseId::from_string("l-owner");
        store
            .try_acquire_task(&id, worker.clone(), lease.clone(), t_plus(10), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;

        // Wrong lease: NotOwned, row untouched.
        let outcome = store
            .requeue_owned_task(
                &id,
                &worker,
                &LeaseId::from_string("l-stale"),
                None,
                t_plus(2),
            )
            .await
            .context("requeue with stale lease")?;
        assert!(matches!(outcome, RequeueOutcome::NotOwned));
        let row = store.get(&id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(row.lease_id.as_ref(), Some(&lease));

        // Cancelled row: NotOwned even with the right lease — the
        // cancel owns the terminal state and a requeue must never
        // resurrect it.
        store.cancel_tree(&id, t_plus(3)).await.context("cancel")?;
        let outcome = store
            .requeue_owned_task(&id, &worker, &lease, None, t_plus(4))
            .await
            .context("requeue after cancel")?;
        assert!(matches!(outcome, RequeueOutcome::NotOwned));
        let row = store.get(&id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Cancelled);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requeue_owned_task_reports_budget_exhausted_leaving_row_owned() -> Result<()> {
        // A budget-1 root that collides has nothing left to spend on a
        // re-run; the store reports it WITHOUT transitioning so the
        // caller can route it through its full terminal-failure
        // envelope (which CAS-checks ownership itself).
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let worker = WorkerId::from_string("w-last");
        let lease = LeaseId::from_string("l-last");
        store
            .try_acquire_task(&id, worker.clone(), lease.clone(), t_plus(10), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;

        let outcome = store
            .requeue_owned_task(&id, &worker, &lease, None, t_plus(2))
            .await
            .context("requeue")?;
        assert!(matches!(outcome, RequeueOutcome::BudgetExhausted));
        let row = store.get(&id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(row.worker_id.as_ref(), Some(&worker));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_closed_row_rejects_heartbeat_and_reacquire() -> Result<()> {
        // Phase 2.5 regression: an old worker whose row was
        // failed-closed by the sweep must not be able to heartbeat
        // or re-acquire it. `try_acquire_task` returns `Ok(None)`
        // for terminal rows and `heartbeat_task` rejects the
        // wrong-status row.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        // Sweep: budget exhausted → fail closed.
        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep")?;
        assert!(swept[0].action.is_fail_closed());

        // Heartbeat rejected: the row is terminal, so the
        // row-level CAS rejects it with an invalid-transition error.
        let err = store
            .heartbeat_task(
                &id,
                &WorkerId::from_string("w-dead"),
                &LeaseId::from_string("l-dead"),
                t_plus(300),
                None,
                t_plus(30),
            )
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("invalid transition") || message.contains("InvalidTransition"),
            "unexpected: {message}"
        );

        // Re-acquire: `try_acquire_task` must silently return None
        // (the row is Failed, which is non-runnable).
        let result = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-dead"),
                LeaseId::from_string("l-dead-2"),
                t_plus(400),
                t_plus(40),
            )
            .await
            .context("reacquire")?;
        assert!(result.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_closed_row_does_not_reappear_in_runnable_index() -> Result<()> {
        // Phase 2.5 bookkeeping check: a row failed closed through
        // the acquisition path must leave the runnable index empty,
        // so a subsequent `acquire_next_runnable` returns `None`
        // immediately and does not re-visit the terminal row.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let claimed = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;
        let released = claimed.release_lease(t_plus(2)).context("release")?;
        store.update(released).await.context("update")?;

        // Trigger the fail-closed via `try_acquire_task`.
        let result = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w2"),
                LeaseId::from_string("l2"),
                t_plus(120),
                t_plus(3),
            )
            .await
            .context("second acquire")?;
        assert!(result.is_none());

        // A follow-up scan finds nothing.
        let scan = store
            .acquire_next_runnable(
                WorkerId::from_string("w3"),
                LeaseId::from_string("l3"),
                t_plus(200),
                t_plus(100),
            )
            .await
            .context("scan")?;
        assert!(scan.is_none(), "failed-closed row must not re-appear");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn phase_2_5_recovery_is_idempotent() -> Result<()> {
        // Running the sweep twice in a row, then a scan-acquire
        // twice in a row, must produce the same outcome the second
        // time — nothing new to do.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t"), t0(), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(10),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        // First sweep: one fail-closed record.
        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep 1")?;
        assert_eq!(swept.len(), 1);
        assert!(swept[0].action.is_fail_closed());

        // Second sweep: nothing to do.
        let swept = store
            .release_expired_leases(t_plus(20))
            .await
            .context("sweep 2")?;
        assert!(swept.is_empty());

        // Scan: the row is terminal, so there is nothing to claim.
        let scan = store
            .acquire_next_runnable(
                WorkerId::from_string("w2"),
                LeaseId::from_string("l2"),
                t_plus(200),
                t_plus(100),
            )
            .await
            .context("scan 1")?;
        assert!(scan.is_none());
        let scan = store
            .acquire_next_runnable(
                WorkerId::from_string("w3"),
                LeaseId::from_string("l3"),
                t_plus(300),
                t_plus(101),
            )
            .await
            .context("scan 2")?;
        assert!(scan.is_none());
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.6 — Tool-runtime child tasks, cancellation tree, and
    // journal-driven parent resume triggers
    // ──────────────────────────────────────────────────────────────
    //
    // Acceptance criteria these tests pin down:
    //
    // * `spawn_tool_children` atomically persists a batch of
    //   tool-runtime children under a running parent and pauses the
    //   parent on them, dropping the parent's lease in the same
    //   write.
    // * Child rows are real `ToolRuntime` rows with the correct
    //   `parent_id` / `root_id` / `thread_id` / `depth` — they share
    //   the Phase 2.3 acquisition path and never accidentally
    //   duplicate or re-acquire.
    // * `complete_task` / `fail_task` recompute the parent's
    //   `pending_child_count` from the live `by_parent` index so a
    //   double-complete is a no-op and a mixed success/failure batch
    //   still resumes the parent deterministically.
    // * `cancel_tree` cascades cancellation through every descendant
    //   in `by_parent` and drops leases along the way, and the
    //   resulting terminal subtree plays nicely with the Phase 2.2
    //   FIFO queue.
    // * A completed child batch makes the parent resumable through
    //   **journal state alone** — no channels, no caller-tracked
    //   counters, and the parent's resume envelope survives a
    //   round-trip through the serialized wire form.

    use crate::journal::task::ChildSpawnSpec;
    use crate::worker::subagent::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
        InheritedSubagentPolicy, SubagentSandboxPolicy,
    };
    use agent_sdk_foundation::audit::AuditProvenance;

    /// Spawn a parent root, lease it, and return
    /// `(parent, worker, lease)` so Phase 2.6 tests can exercise
    /// `spawn_tool_children` against a genuinely running parent.
    async fn running_root_for_spawn(
        store: &InMemoryAgentTaskStore,
        thread_name: &str,
    ) -> Result<(AgentTask, WorkerId, LeaseId)> {
        submitted_and_claimed_root(store, thread_name, "w-parent", "l-parent").await
    }

    fn sample_subagent_spec() -> EffectiveSubagentSpec {
        EffectiveSubagentSpec {
            task: "Inspect durable linkage".into(),
            prompt: "Stay in read-only mode.".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            max_turns: 5,
            timeout_ms: 15_000,
            depth: 1,
            max_parallel_subagents: 1,
            nickname: Some("Scout".into()),
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy {
                allowed_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
            },
            audit_provenance: Some(AuditProvenance::new(
                "anthropic",
                "claude-sonnet-4-5-20250929",
            )),
            inherited_policy: InheritedSubagentPolicy {
                default_model: "claude-sonnet-4-5-20250929".into(),
                allowed_models: std::collections::BTreeSet::from([String::from(
                    "claude-sonnet-4-5-20250929",
                )]),
                default_max_turns: 5,
                max_turns: 5,
                default_timeout_ms: 15_000,
                max_timeout_ms: 15_000,
                capability_profiles: std::collections::BTreeMap::from([(
                    "research".into(),
                    crate::worker::subagent::SubagentCapabilityProfile {
                        capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
                        sandbox: SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
                    },
                )]),
                allowed_capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
                max_depth: 3,
                max_parallel_subagents: 1,
                sandbox: SubagentSandboxPolicy::read_only(),
                allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
                audit_provider: "anthropic".into(),
            },
            capabilities: EffectiveSubagentCapabilities {
                profile: "research".into(),
                allowed: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
            },
        }
    }

    fn sample_subagent_input() -> Vec<SubmittedInputItem> {
        vec![SubmittedInputItem::Text {
            text: "Stay in read-only mode.\n\nInspect durable linkage".into(),
        }]
    }

    async fn spawn_subagent_fixture(
        store: &InMemoryAgentTaskStore,
        thread_name: &str,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let (parent, worker, lease) = running_root_for_spawn(store, thread_name).await?;
        let parent_id = parent.id.clone();
        let child_thread_id = thread(&format!("{thread_name}-child"));
        store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id,
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation(thread_name),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(2),
            )
            .await
    }

    async fn exhaust_retry_budget(
        store: &InMemoryAgentTaskStore,
        task_id: &AgentTaskId,
        max_attempts: u32,
        worker_prefix: &str,
        start_secs: i64,
    ) -> Result<Vec<RecoveryRecord>> {
        let mut last_sweep = Vec::new();
        for attempt in 0..max_attempts {
            let offset = i64::from(attempt) * 10;
            let running = store
                .try_acquire_task(
                    task_id,
                    WorkerId::from_string(format!("{worker_prefix}-{attempt}")),
                    LeaseId::from_string(format!("l-{worker_prefix}-{attempt}")),
                    t_plus(start_secs + offset + 5),
                    t_plus(start_secs + offset + 1),
                )
                .await?
                .with_context(|| format!("claim {task_id} attempt {}", attempt + 1))?;
            assert_eq!(running.status, TaskStatus::Running);
            last_sweep = store
                .release_expired_leases(t_plus(start_secs + offset + 6))
                .await?;
        }
        Ok(last_sweep)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_creates_batch_and_parks_parent() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-spawn").await?;
        let parent_id = parent.id.clone();

        let specs = vec![
            ChildSpawnSpec::new(2),
            ChildSpawnSpec::new(2),
            ChildSpawnSpec::new(2),
        ];
        let (parked_parent, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                specs,
                SuspensionPayload {
                    continuation: sample_continuation("t-spawn"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Parent is paused on exactly the spawned batch.
        assert_eq!(parked_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(parked_parent.pending_child_count, 3);
        assert!(parked_parent.worker_id.is_none(), "lease dropped");
        assert!(parked_parent.lease_id.is_none());
        assert!(parked_parent.lease_expires_at.is_none());
        assert!(parked_parent.last_heartbeat_at.is_none());
        assert!(matches!(
            parked_parent.state,
            TaskState::WaitingOnChildren { .. }
        ));

        // Children inherit the parent's identity fields and start
        // Pending with attempt == 0.
        assert_eq!(children.len(), 3);
        for child in &children {
            assert_eq!(child.kind, TaskKind::ToolRuntime);
            assert_eq!(child.status, TaskStatus::Pending);
            assert_eq!(child.parent_id.as_ref(), Some(&parent_id));
            assert_eq!(child.root_id, parent.root_id);
            assert_eq!(child.thread_id, parent.thread_id);
            assert_eq!(child.depth, parent.depth + 1);
            assert_eq!(child.attempt, 0);
            assert_eq!(child.max_attempts, 2);
            assert!(child.state.is_none());
        }

        // The persisted rows match what `spawn_tool_children` returned.
        let fetched_parent = store
            .get(&parent_id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(fetched_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(fetched_parent.pending_child_count, 3);
        let listed_children = store
            .list_children(&parent_id)
            .await
            .context("list children")?;
        assert_eq!(listed_children.len(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-spawn-cas").await?;
        let parent_id = parent.id.clone();

        let err = store
            .spawn_tool_children(
                &parent_id,
                &WorkerId::from_string("w-imposter"),
                &lease,
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-spawn-cas"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("worker mismatch"),
            "unexpected: {err:#}"
        );

        let err = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &LeaseId::from_string("l-stale"),
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-spawn-cas"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(3),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("lease mismatch"),
            "unexpected: {err:#}"
        );

        // Parent must still be Running with its original lease.
        let persisted = store
            .get(&parent_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert_eq!(
            persisted.worker_id.as_ref().map(WorkerId::as_str),
            Some("w-parent")
        );
        // No children were created.
        let children = store.list_children(&parent_id).await.context("list")?;
        assert!(children.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_invocation_creates_durable_linkage() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-subagent-spawn").await?;
        let parent_id = parent.id.clone();
        let child_thread_id = thread("t-subagent-child");

        let (parked_parent, invocation, child_root) = store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id: child_thread_id.clone(),
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation("t-subagent-spawn"),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(2),
            )
            .await
            .context("spawn_subagent_invocation")?;

        assert_eq!(parked_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(parked_parent.pending_child_count, 1);
        assert!(parked_parent.worker_id.is_none());
        assert_eq!(invocation.spawn_index, Some(0));

        assert_eq!(invocation.kind, TaskKind::Subagent);
        assert_eq!(invocation.status, TaskStatus::WaitingOnChildren);
        assert_eq!(invocation.parent_id.as_ref(), Some(&parent_id));
        assert_eq!(invocation.root_id, parent.root_id);
        assert_eq!(invocation.thread_id, parent.thread_id);
        assert_eq!(invocation.depth, parent.depth + 1);
        assert_eq!(invocation.pending_child_count, 1);
        let linkage = invocation
            .state
            .subagent_invocation()
            .context("subagent linkage missing")?;
        assert_eq!(linkage.child_root_task_id, child_root.id);
        assert_eq!(linkage.child_thread_id, child_root.thread_id);
        assert_eq!(linkage.spec, sample_subagent_spec());
        assert_eq!(child_root.thread_id, child_thread_id);

        assert_eq!(child_root.kind, TaskKind::RootTurn);
        assert_eq!(child_root.status, TaskStatus::Pending);
        assert!(child_root.parent_id.is_none());
        assert_eq!(child_root.depth, 0);
        assert_eq!(child_root.root_id, child_root.id);

        let parent_children = store
            .list_children(&parent_id)
            .await
            .context("list parent children")?;
        assert_eq!(parent_children.len(), 1);
        assert_eq!(parent_children[0].id, invocation.id);

        let child_thread_tasks = store
            .list_by_thread(&child_root.thread_id)
            .await
            .context("list child thread")?;
        assert_eq!(child_thread_tasks.len(), 1);
        assert_eq!(child_thread_tasks[0].id, child_root.id);

        let active_child_root = store
            .active_root_for_thread(&child_root.thread_id)
            .await
            .context("active child root")?
            .context("child root should block admission on child thread")?;
        assert_eq!(active_child_root.id, child_root.id);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_invocation_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) =
            running_root_for_spawn(&store, "t-subagent-spawn-cas").await?;
        let parent_id = parent.id.clone();
        let child_thread_id = thread("t-subagent-cas-child");

        let err = store
            .spawn_subagent_invocation(
                &parent_id,
                &WorkerId::from_string("w-imposter"),
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id: child_thread_id.clone(),
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation("t-subagent-spawn-cas"),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(2),
            )
            .await
            .err()
            .context("wrong worker should fail")?;
        assert!(
            format!("{err:#}").contains("worker mismatch"),
            "unexpected: {err:#}"
        );

        let err = store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &LeaseId::from_string("l-stale"),
                SubagentInvocationSpawn {
                    child_thread_id,
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation("t-subagent-spawn-cas"),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(3),
            )
            .await
            .err()
            .context("wrong lease should fail")?;
        assert!(
            format!("{err:#}").contains("lease mismatch"),
            "unexpected: {err:#}"
        );

        let persisted = store
            .get(&parent_id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_batch_creates_n_invocations_under_one_parent() -> Result<()> {
        // Fan-out: a single CAS on the parent transitions it into
        // WaitingOnChildren with `pending_child_count = N`, allocates
        // N invocation tasks, and N child-thread root tasks. The
        // parent's `child_ids` list contains every invocation id in
        // the same order as the input `spawns`.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-subagent-batch").await?;
        let parent_id = parent.id.clone();
        let child_thread_ids = [
            thread("t-subagent-batch-child-a"),
            thread("t-subagent-batch-child-b"),
            thread("t-subagent-batch-child-c"),
        ];

        let spawns: Vec<SubagentInvocationSpawn> = child_thread_ids
            .iter()
            .enumerate()
            .map(|(idx, child_thread_id)| SubagentInvocationSpawn {
                child_thread_id: child_thread_id.clone(),
                spec: sample_subagent_spec(),
                child_root_input: sample_subagent_input(),
                spawn_index: u32::try_from(idx).expect("test idx fits in u32"),
                payload: SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch"),
                    suspended_messages: Vec::new(),
                },
                child_caller_metadata: None,
            })
            .collect();

        let (parked_parent, prepared) = store
            .spawn_subagent_batch(
                &parent_id,
                &worker,
                &lease,
                spawns,
                SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .context("spawn_subagent_batch")?;

        // Parent transitioned once with the cumulative count.
        assert_eq!(parked_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(parked_parent.pending_child_count, 3);
        assert!(parked_parent.worker_id.is_none());
        match &parked_parent.state {
            crate::journal::task_state::TaskState::WaitingOnChildren { child_ids, .. } => {
                assert_eq!(child_ids.len(), 3);
                for ((idx, (invocation, _child_root)), child_id) in
                    prepared.iter().enumerate().zip(child_ids.iter())
                {
                    assert_eq!(
                        &invocation.id, child_id,
                        "child_ids[{idx}] should match invocation order"
                    );
                }
            }
            other => panic!("expected WaitingOnChildren state, got {other:?}"),
        }

        // Per-entry assertions.
        assert_eq!(prepared.len(), 3);
        for (idx, (invocation, child_root)) in prepared.iter().enumerate() {
            assert_eq!(invocation.kind, TaskKind::Subagent);
            assert_eq!(invocation.status, TaskStatus::WaitingOnChildren);
            assert_eq!(invocation.parent_id.as_ref(), Some(&parent_id));
            assert_eq!(
                invocation.spawn_index,
                Some(u32::try_from(idx).expect("test idx fits in u32"))
            );
            assert_eq!(invocation.thread_id, parent.thread_id);
            assert_eq!(invocation.depth, parent.depth + 1);

            let linkage = invocation
                .state
                .subagent_invocation()
                .context("subagent linkage missing")?;
            assert_eq!(linkage.child_root_task_id, child_root.id);
            assert_eq!(linkage.child_thread_id, child_thread_ids[idx]);

            assert_eq!(child_root.kind, TaskKind::RootTurn);
            assert_eq!(child_root.status, TaskStatus::Pending);
            assert_eq!(child_root.thread_id, child_thread_ids[idx]);
            assert!(child_root.parent_id.is_none());
        }

        // Index assertions: parent's child set sees all N invocations.
        let parent_children = store
            .list_children(&parent_id)
            .await
            .context("list parent children")?;
        assert_eq!(parent_children.len(), 3);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_batch_rejects_empty_input() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) =
            running_root_for_spawn(&store, "t-subagent-batch-empty").await?;
        let parent_id = parent.id.clone();

        let err = store
            .spawn_subagent_batch(
                &parent_id,
                &worker,
                &lease,
                Vec::new(),
                SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-empty"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .err()
            .context("empty batch should fail")?;
        assert!(
            format!("{err:#}").contains("spawns must be non-empty"),
            "unexpected: {err:#}"
        );

        // Parent state untouched after the rejected call.
        let persisted = store
            .get(&parent_id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    /// Continuation carrying `count` pending tool call slots — the
    /// mixed primitive validates its spawn indices against this list.
    fn continuation_with_slots(name: &str, count: usize) -> ContinuationEnvelope {
        let mut continuation = sample_continuation(name);
        continuation.payload.pending_tool_calls = (0..count)
            .map(|idx| agent_sdk_foundation::PendingToolCallInfo {
                id: format!("call_{idx}"),
                name: format!("tool_{idx}"),
                display_name: format!("Tool {idx}"),
                tier: agent_sdk_foundation::ToolTier::Confirm,
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            })
            .collect();
        continuation
    }

    fn mixed_spawn_for(
        name: &str,
        subagent_threads: Vec<ThreadId>,
        tool_slots: &[u32],
    ) -> MixedChildrenSpawn {
        let slots = subagent_threads.len() + tool_slots.len();
        let payload = SuspensionPayload {
            continuation: continuation_with_slots(name, slots),
            suspended_messages: Vec::new(),
        };
        let subagents = subagent_threads
            .into_iter()
            .enumerate()
            .map(|(idx, child_thread_id)| SubagentInvocationSpawn {
                child_thread_id,
                spec: sample_subagent_spec(),
                child_root_input: sample_subagent_input(),
                spawn_index: u32::try_from(idx).unwrap_or_default(),
                payload: payload.clone(),
                child_caller_metadata: None,
            })
            .collect();
        let tool_children = tool_slots
            .iter()
            .map(|slot| ToolChildSpawn {
                spawn_index: *slot,
                spec: ChildSpawnSpec::new(3),
            })
            .collect();
        MixedChildrenSpawn {
            subagents,
            tool_children,
            payload,
            child_otel_traceparent: None,
        }
    }

    /// The mixed primitive is not a substitute for the two pure paths: a
    /// batch with no tool children belongs on `spawn_subagent_batch`, one
    /// with no subagents on `spawn_tool_children`. Accepting either here
    /// would silently duplicate a contract that already exists — and the
    /// pure paths carry guarantees (e.g. `spawn_tool_children`'s implicit
    /// contiguous indices) the mixed validation does not restate.
    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_mixed_children_rejects_single_kind_batches() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-mixed-single-kind").await?;
        let parent_id = parent.id.clone();

        let mut subagents_only = mixed_spawn_for(
            "t-mixed-single-kind",
            vec![thread("t-mixed-single-kind-child")],
            &[1],
        );
        subagents_only.tool_children.clear();
        let err = store
            .spawn_mixed_children(&parent_id, &worker, &lease, subagents_only, t_plus(2))
            .await
            .err()
            .context("a subagent-only batch must be rejected")?;
        assert!(
            format!("{err:#}").contains("at least one tool child"),
            "unexpected: {err:#}"
        );

        let mut tools_only =
            mixed_spawn_for("t-mixed-single-kind", vec![thread("t-mixed-child-2")], &[1]);
        tools_only.subagents.clear();
        let err = store
            .spawn_mixed_children(&parent_id, &worker, &lease, tools_only, t_plus(2))
            .await
            .err()
            .context("a tool-only batch must be rejected")?;
        assert!(
            format!("{err:#}").contains("at least one subagent"),
            "unexpected: {err:#}"
        );

        let persisted = store
            .get(&parent_id)
            .await?
            .context("parent survives a rejected spawn")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    /// Two children claiming the same `pending_tool_calls` slot would
    /// overwrite each other at fan-in time (the positional aggregation
    /// keeps one result per slot), so the batch is rejected before any
    /// row is written.
    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_mixed_children_rejects_duplicate_spawn_index() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-mixed-dup-slot").await?;
        let parent_id = parent.id.clone();

        // Subagent takes slot 0; the tool child claims slot 0 as well.
        let spawn = mixed_spawn_for("t-mixed-dup-slot", vec![thread("t-mixed-dup-child")], &[0]);
        let err = store
            .spawn_mixed_children(&parent_id, &worker, &lease, spawn, t_plus(2))
            .await
            .err()
            .context("a duplicated slot must be rejected")?;
        assert!(
            format!("{err:#}").contains("duplicate spawn_index"),
            "unexpected: {err:#}"
        );

        let persisted = store
            .get(&parent_id)
            .await?
            .context("parent survives a rejected spawn")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_batch_rejects_duplicate_child_thread_ids() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) =
            running_root_for_spawn(&store, "t-subagent-batch-dup").await?;
        let parent_id = parent.id.clone();
        let dup_thread = thread("t-subagent-batch-dup-child");

        let spawns = vec![
            SubagentInvocationSpawn {
                child_thread_id: dup_thread.clone(),
                spec: sample_subagent_spec(),
                child_root_input: sample_subagent_input(),
                spawn_index: 0,
                payload: SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-dup"),
                    suspended_messages: Vec::new(),
                },
                child_caller_metadata: None,
            },
            SubagentInvocationSpawn {
                child_thread_id: dup_thread.clone(),
                spec: sample_subagent_spec(),
                child_root_input: sample_subagent_input(),
                spawn_index: 1,
                payload: SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-dup"),
                    suspended_messages: Vec::new(),
                },
                child_caller_metadata: None,
            },
        ];

        let err = store
            .spawn_subagent_batch(
                &parent_id,
                &worker,
                &lease,
                spawns,
                SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-dup"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .err()
            .context("duplicate thread id should fail")?;
        assert!(
            format!("{err:#}").contains("duplicate child_thread_id"),
            "unexpected: {err:#}"
        );

        let persisted = store
            .get(&parent_id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_subagent_batch_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, _worker, lease) =
            running_root_for_spawn(&store, "t-subagent-batch-cas").await?;
        let parent_id = parent.id.clone();
        let child_thread_id = thread("t-subagent-batch-cas-child");
        let make_spawns = || {
            vec![SubagentInvocationSpawn {
                child_thread_id: child_thread_id.clone(),
                spec: sample_subagent_spec(),
                child_root_input: sample_subagent_input(),
                spawn_index: 0,
                payload: SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-cas"),
                    suspended_messages: Vec::new(),
                },
                child_caller_metadata: None,
            }]
        };

        let err = store
            .spawn_subagent_batch(
                &parent_id,
                &WorkerId::from_string("w-imposter"),
                &lease,
                make_spawns(),
                SuspensionPayload {
                    continuation: sample_continuation("t-subagent-batch-cas"),
                    suspended_messages: Vec::new(),
                },
                t_plus(2),
            )
            .await
            .err()
            .context("wrong worker should fail")?;
        assert!(
            format!("{err:#}").contains("worker mismatch"),
            "unexpected: {err:#}"
        );

        let persisted = store
            .get(&parent_id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        assert!(store.list_children(&parent_id).await?.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn child_root_terminal_transition_wakes_linked_subagent_invocation() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        let (parent, worker, lease) = running_root_for_spawn(&store, "t-subagent-terminal").await?;
        let parent_id = parent.id.clone();
        let child_thread_id = thread("t-subagent-terminal-child");

        let (_parked_parent, invocation, child_root) = store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id,
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation("t-subagent-terminal"),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(2),
            )
            .await
            .context("spawn_subagent_invocation")?;

        let child_running = store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-child"),
                LeaseId::from_string("l-child"),
                t_plus(60),
                t_plus(3),
            )
            .await?
            .context("claim child root")?;
        assert_eq!(child_running.status, TaskStatus::Running);

        let (_completed_child, resumed_invocation) = store
            .complete_task(
                &child_root.id,
                &WorkerId::from_string("w-child"),
                &LeaseId::from_string("l-child"),
                t_plus(4),
            )
            .await
            .context("complete child root")?;

        let resumed_invocation = resumed_invocation.context("linked invocation should resume")?;
        assert_eq!(resumed_invocation.id, invocation.id);
        assert_eq!(resumed_invocation.status, TaskStatus::Pending);
        assert_eq!(resumed_invocation.pending_child_count, 0);
        assert!(resumed_invocation.state.subagent_invocation().is_some());

        let claimed_invocation = store
            .try_acquire_task(
                &invocation.id,
                WorkerId::from_string("w-subagent"),
                LeaseId::from_string("l-subagent"),
                t_plus(60),
                t_plus(5),
            )
            .await?
            .context("claim resumed invocation")?;
        assert_eq!(claimed_invocation.status, TaskStatus::Running);
        assert!(claimed_invocation.state.subagent_invocation().is_some());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_on_child_root_wakes_linked_subagent_invocation() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(&store, "t-subagent-cancel").await?;

        let transitioned = store
            .cancel_tree(&child_root.id, t_plus(3))
            .await?
            .transitioned;
        assert_eq!(transitioned, vec![child_root.id.clone()]);

        let resumed_invocation = store
            .get(&invocation.id)
            .await?
            .context("invocation exists after cancel")?;
        assert_eq!(resumed_invocation.status, TaskStatus::Pending);
        assert_eq!(resumed_invocation.pending_child_count, 0);
        assert!(resumed_invocation.state.subagent_invocation().is_some());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fail_closed_subagent_paths_keep_invocation_and_parent_runnable() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(&store, "t-subagent-fail-closed").await?;

        let child_sweep = exhaust_retry_budget(
            &store,
            &child_root.id,
            child_root.max_attempts,
            "w-child",
            3,
        )
        .await?;
        assert_eq!(
            child_sweep,
            vec![RecoveryRecord::failed_closed(
                child_root.id.clone(),
                FailureReason::LeaseExpiredBudgetExhausted,
            )],
        );

        let resumed_invocation = store
            .get(&invocation.id)
            .await?
            .context("invocation exists after child fail-closed")?;
        assert_eq!(resumed_invocation.status, TaskStatus::Pending);
        assert_eq!(resumed_invocation.pending_child_count, 0);

        let invocation_sweep = exhaust_retry_budget(
            &store,
            &invocation.id,
            invocation.max_attempts,
            "w-subagent",
            40,
        )
        .await?;
        assert_eq!(
            invocation_sweep,
            vec![RecoveryRecord::failed_closed(
                invocation.id.clone(),
                FailureReason::LeaseExpiredBudgetExhausted,
            )],
        );

        let resumed_parent = store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(resumed_parent.status, TaskStatus::Pending);
        assert_eq!(resumed_parent.pending_child_count, 0);
        assert!(matches!(
            resumed_parent.state,
            TaskState::ReadyToResume { .. }
        ));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_rejects_non_running_parent() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-spawn-pending"), t_plus(0), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        // Row is Pending — spawn must refuse with "not running".
        let err = store
            .spawn_tool_children(
                &id,
                &WorkerId::from_string("w1"),
                &LeaseId::from_string("l1"),
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-spawn-pending"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(1),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("not running"),
            "unexpected: {err:#}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_rejects_leaf_parent() -> Result<()> {
        // A tool-runtime row is a leaf. Trying to spawn children
        // under one must fail with the same "parent is a leaf" shape
        // that `insert` already enforces.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-leaf").await?;
        let parent_id = parent.id.clone();

        // Spawn one tool child under the root so we have a real
        // leaf to try to reparent onto.
        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-leaf"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn first")?;
        let leaf = &children[0];
        let leaf_id = leaf.id.clone();

        // Lease the leaf so it satisfies the running + CAS guards,
        // then try to spawn grandchildren under it.
        let claimed = store
            .try_acquire_task(
                &leaf_id,
                WorkerId::from_string("w-leaf"),
                LeaseId::from_string("l-leaf"),
                t_plus(30),
                t_plus(3),
            )
            .await
            .context("acquire leaf")?
            .context("leaf claimed")?;
        assert_eq!(claimed.kind, TaskKind::ToolRuntime);

        let err = store
            .spawn_tool_children(
                &leaf_id,
                &WorkerId::from_string("w-leaf"),
                &LeaseId::from_string("l-leaf"),
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-leaf"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(4),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("leaf kind"),
            "unexpected: {err:#}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_rejects_empty_specs() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-empty").await?;
        let parent_id = parent.id.clone();

        let err = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                Vec::new(),
                SuspensionPayload {
                    continuation: sample_continuation("t-empty"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("specs must be non-empty"),
            "unexpected: {err:#}"
        );

        // The parent must still be Running — a zero-child spawn is
        // a caller error, not a legal pause.
        let persisted = store
            .get(&parent_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted.status, TaskStatus::Running);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_tool_children_indexes_children_as_runnable() -> Result<()> {
        // After the spawn, the children must be on the runnable
        // index and the parent must **not** be. The scan picks the
        // children in FIFO order and skips the paused parent.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-scan-children").await?;
        let parent_id = parent.id.clone();

        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-scan-children"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(10),
            )
            .await
            .context("spawn")?;
        let child_ids: std::collections::HashSet<_> =
            children.iter().map(|c| c.id.clone()).collect();

        for i in 0..2 {
            let offset = i64::from(i);
            let claimed = store
                .acquire_next_runnable(
                    WorkerId::from_string(format!("w-scan-{i}")),
                    LeaseId::from_string(format!("l-scan-{i}")),
                    t_plus(60),
                    t_plus(20 + offset),
                )
                .await
                .context("scan")?
                .context("scan returned row")?;
            assert_eq!(claimed.kind, TaskKind::ToolRuntime);
            assert!(
                child_ids.contains(&claimed.id),
                "scan returned unexpected row {id:?}",
                id = claimed.id
            );
        }

        // Third scan drains.
        let none = store
            .acquire_next_runnable(
                WorkerId::from_string("w-scan-drain"),
                LeaseId::from_string("l-scan-drain"),
                t_plus(60),
                t_plus(30),
            )
            .await
            .context("scan drain")?;
        assert!(
            none.is_none(),
            "paused parent must not appear in runnable scan"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complete_task_resumes_parent_on_final_decrement() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-complete").await?;
        let parent_id = parent.id.clone();

        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-complete"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Lease and complete each child via the Phase 2.6 path.
        for (i, child) in children.iter().enumerate() {
            let offset = i64::try_from(i).context("enumerate fits i64")?;
            let claimed = store
                .try_acquire_task(
                    &child.id,
                    WorkerId::from_string(format!("w-c-{i}")),
                    LeaseId::from_string(format!("l-c-{i}")),
                    t_plus(60),
                    t_plus(10 + offset),
                )
                .await
                .context("acquire child")?
                .context("child claimed")?;
            let (done, observed_parent) = store
                .complete_task(
                    &claimed.id,
                    &WorkerId::from_string(format!("w-c-{i}")),
                    &LeaseId::from_string(format!("l-c-{i}")),
                    t_plus(20 + offset),
                )
                .await
                .context("complete child")?;
            assert_eq!(done.status, TaskStatus::Completed);
            let observed = observed_parent.context("parent returned")?;
            if i == children.len() - 1 {
                assert_eq!(observed.status, TaskStatus::Pending);
                assert_eq!(observed.pending_child_count, 0);
                assert!(matches!(observed.state, TaskState::ReadyToResume { .. }));
            } else {
                assert_eq!(observed.status, TaskStatus::WaitingOnChildren);
                assert!(observed.state.continuation().is_some());
            }
        }

        // The parent now scans as runnable again.
        let resumed = store
            .acquire_next_runnable(
                WorkerId::from_string("w-resume"),
                LeaseId::from_string("l-resume"),
                t_plus(200),
                t_plus(100),
            )
            .await
            .context("scan resumed")?
            .context("resumed parent found")?;
        assert_eq!(resumed.id, parent_id);
        assert_eq!(resumed.status, TaskStatus::Running);
        // The resumed parent is still on attempt 1 — the ReadyToResume
        // re-acquisition is a continuation, not a retry, so the attempt
        // counter is not incremented.
        assert_eq!(resumed.attempt, 1);
        Ok(())
    }

    /// Drive a fresh root through child-spawn → resume so the row
    /// reaches `Pending + ReadyToResume + attempt=1`. Returns the
    /// parent id for the regression test below to drive the
    /// crashed-resume cycle against.
    async fn submit_resumable_root(
        store: &InMemoryAgentTaskStore,
        thread_name: &str,
        budget: u32,
    ) -> Result<AgentTaskId> {
        let root = AgentTask::new_root_turn(thread(thread_name), t_plus(0), budget);
        let parent_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let worker = WorkerId::from_string("w-parent");
        let lease = LeaseId::from_string("l-parent");
        store
            .try_acquire_task(
                &parent_id,
                worker.clone(),
                lease.clone(),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire parent")?
            .context("parent claimed")?;
        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation(thread_name),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn child")?;
        let child = &children[0];
        store
            .try_acquire_task(
                &child.id,
                WorkerId::from_string("w-child"),
                LeaseId::from_string("l-child"),
                t_plus(60),
                t_plus(3),
            )
            .await
            .context("acquire child")?
            .context("child claimed")?;
        store
            .complete_task(
                &child.id,
                &WorkerId::from_string("w-child"),
                &LeaseId::from_string("l-child"),
                t_plus(4),
            )
            .await
            .context("complete child")?;
        Ok(parent_id)
    }

    /// Acquire a `ReadyToResume` row, then drive the lease past `now`
    /// and let the sweep classify it. Returns the recovery action so
    /// the caller can assert on `Requeue` vs `FailClosed`.
    ///
    /// The lease is set to `acquired_at + 1` — a deliberately tiny 1s
    /// window so the test runs on a compressed time scale. The
    /// production `lease_duration_secs` (default 30s) is irrelevant
    /// here; what we exercise is the *shape* of expired-lease →
    /// recovery, not the duration. `sweep_at` must be past
    /// `acquired_at + 1` so the lease is observably expired.
    async fn crash_resume_cycle(
        store: &InMemoryAgentTaskStore,
        parent_id: &AgentTaskId,
        cycle_label: &str,
        acquired_at: i64,
        sweep_at: i64,
    ) -> Result<crate::journal::recovery::RecoveryAction> {
        store
            .try_acquire_task(
                parent_id,
                WorkerId::from_string(format!("w-{cycle_label}")),
                LeaseId::from_string(format!("l-{cycle_label}")),
                t_plus(acquired_at + 1),
                t_plus(acquired_at),
            )
            .await
            .context("acquire resume")?
            .context("resume claimed")?;
        let swept = store
            .release_expired_leases(t_plus(sweep_at))
            .await
            .context("sweep")?;
        assert_eq!(swept.len(), 1, "exactly one row should sweep per cycle");
        Ok(swept[0].action)
    }

    /// Regression: a `ReadyToResume` row whose worker keeps losing its
    /// lease must reach a terminal state in a bounded number of cycles.
    ///
    /// Bug shape (pre-fix):
    /// 1. A root turn ran, suspended on tool children, and resumed back
    ///    to `Pending + ReadyToResume` after the children finished.
    /// 2. The resume worker's LLM call ran longer than the lease
    ///    (`lease_duration_secs = 30` by default in the host); no
    ///    heartbeat existed, so the lease expired mid-flight.
    /// 3. The expiry sweep saw `ExpiredLease + Running + ReadyToResume`
    ///    and the recovery matrix carved this case out of the budget
    ///    check — it always returned `Requeue`. `release_lease` did not
    ///    bump `attempt` for `ReadyToResume`.
    /// 4. The next worker re-acquired the row, ran the same resume,
    ///    hit the same lease expiry, requeued again. Forever. The user
    ///    saw the "resume root task from durable child results" warning
    ///    fire every ~30 seconds with no termination.
    ///
    /// Fix shape:
    /// - `release_lease` bumps `attempt` (capped at `max_attempts`)
    ///   when the row is in `TaskState::ReadyToResume`, so the budget
    ///   counter actually moves on each crashed resume cycle.
    /// - `classify_recovery` no longer carves `ReadyToResume` out of
    ///   the `ExpiredLease` budget check, so the row fails closed once
    ///   `is_budget_exhausted()` returns true.
    ///
    /// This test simulates the cycle with `max_attempts = 3`:
    /// - The original `Running` consumes one slot.
    /// - Crashed resume #1 → `release_lease` bumps to 2 → requeue.
    /// - Crashed resume #2 → bump to 3 → requeue.
    /// - Crashed resume #3 → matrix sees budget exhausted → fail
    ///   closed with `LeaseExpiredBudgetExhausted`.
    ///
    /// Without the fix the row would still be `Running` (or oscillating
    /// between `Running` and `Pending`) at the end of the test instead
    /// of `Failed`.
    #[tokio::test(flavor = "multi_thread")]
    async fn ready_to_resume_with_repeated_lease_expiries_fails_closed() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let parent_id = submit_resumable_root(&store, "t-resume-loop", 3).await?;

        // Sanity check the starting state: Pending+ReadyToResume on
        // attempt 1, with two slots left in the budget.
        let resumable = store.get(&parent_id).await?.context("parent exists")?;
        assert_eq!(resumable.status, TaskStatus::Pending);
        assert!(matches!(resumable.state, TaskState::ReadyToResume { .. }));
        assert_eq!(resumable.attempt, 1);
        assert_eq!(resumable.max_attempts, 3);

        // Cycle 1: resume worker crashes, sweep requeues, attempt → 2.
        let action = crash_resume_cycle(&store, &parent_id, "r1", 5, 15).await?;
        assert!(
            action.is_requeue(),
            "cycle 1 should requeue, got {action:?}"
        );
        let after = store.get(&parent_id).await?.context("exists")?;
        assert_eq!(after.attempt, 2, "release_lease must bump to 2");

        // Cycle 2: same shape, attempt → 3 (budget now exhausted).
        let action = crash_resume_cycle(&store, &parent_id, "r2", 16, 25).await?;
        assert!(
            action.is_requeue(),
            "cycle 2 should requeue, got {action:?}"
        );
        let after = store.get(&parent_id).await?.context("exists")?;
        assert_eq!(after.attempt, 3);
        assert!(after.is_budget_exhausted());

        // Cycle 3: budget already exhausted → matrix fails closed
        // instead of requeuing forever (the bug we're fixing).
        let action = crash_resume_cycle(&store, &parent_id, "r3", 26, 35).await?;
        assert!(
            action.is_fail_closed(),
            "cycle 3 must fail closed (budget exhausted), got {action:?}",
        );

        let final_row = store.get(&parent_id).await?.context("exists")?;
        assert_eq!(final_row.status, TaskStatus::Failed);
        let last_error = final_row.last_error.as_deref().unwrap_or_default();
        assert!(
            last_error.starts_with("lease_expired_budget_exhausted"),
            "expected budget-exhaustion error, got: {last_error}",
        );
        Ok(())
    }

    /// Companion to `ready_to_resume_with_repeated_lease_expiries_fails_closed`.
    /// Pins down the pre-fix behavior so a future regression that
    /// re-introduces either side of the bug shows up in CI.
    ///
    /// Two assertions describe the fix:
    ///
    /// 1. `release_lease` must bump `attempt` on a `ReadyToResume` row.
    ///    Without that bump the recovery matrix's budget check would
    ///    never fire, no matter how many times the row crashed.
    /// 2. `classify_recovery(ExpiredLease, Running, ReadyToResume)`
    ///    with an exhausted budget must return `FailClosed` rather
    ///    than the old "always Requeue" carve-out.
    #[tokio::test(flavor = "multi_thread")]
    async fn ready_to_resume_recovery_matrix_no_longer_carves_out_budget() -> Result<()> {
        use crate::journal::recovery::{
            FailureReason, RecoveryAction, RecoveryContext, classify_recovery,
        };

        // Set up a single-attempt root that has consumed its slot on
        // the original Running and is now in `Pending+ReadyToResume`.
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-resume-budget"), t_plus(0), 1);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let worker = WorkerId::from_string("w-budget");
        let lease = LeaseId::from_string("l-budget");
        store
            .try_acquire_task(&id, worker.clone(), lease.clone(), t_plus(60), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;
        let (_, children) = store
            .spawn_tool_children(
                &id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-resume-budget"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;
        let child = &children[0];
        store
            .try_acquire_task(
                &child.id,
                WorkerId::from_string("w-c"),
                LeaseId::from_string("l-c"),
                t_plus(60),
                t_plus(3),
            )
            .await
            .context("acquire child")?
            .context("claimed")?;
        store
            .complete_task(
                &child.id,
                &WorkerId::from_string("w-c"),
                &LeaseId::from_string("l-c"),
                t_plus(4),
            )
            .await
            .context("complete")?;

        let resumable = store.get(&id).await?.context("exists")?;
        assert_eq!(resumable.attempt, 1);
        assert_eq!(resumable.max_attempts, 1);
        assert!(resumable.is_budget_exhausted());
        assert!(matches!(resumable.state, TaskState::ReadyToResume { .. }));

        // Re-acquire so we have a `Running+ReadyToResume` row to
        // classify under the `ExpiredLease` context, just like the
        // sweep would.
        let running = store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-resume"),
                LeaseId::from_string("l-resume"),
                t_plus(120),
                t_plus(5),
            )
            .await
            .context("re-acquire")?
            .context("claimed")?;
        assert_eq!(running.status, TaskStatus::Running);
        assert!(matches!(running.state, TaskState::ReadyToResume { .. }));
        assert!(running.is_budget_exhausted());

        // Pre-fix: this returned `RecoveryAction::Requeue` regardless
        // of the budget. That carve-out is exactly what produced the
        // "resume root task from durable child results" infinite loop
        // in production.
        assert_eq!(
            classify_recovery(&running, RecoveryContext::ExpiredLease),
            RecoveryAction::FailClosed(FailureReason::LeaseExpiredBudgetExhausted),
            "ReadyToResume + budget exhausted must fail closed under ExpiredLease",
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complete_task_stays_waiting_until_last() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-partial").await?;
        let parent_id = parent.id.clone();

        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                ],
                SuspensionPayload {
                    continuation: sample_continuation("t-partial"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Complete only the first child.
        let first = &children[0];
        store
            .try_acquire_task(
                &first.id,
                WorkerId::from_string("w-c0"),
                LeaseId::from_string("l-c0"),
                t_plus(60),
                t_plus(10),
            )
            .await
            .context("acquire c0")?
            .context("c0 claimed")?;
        let (_, observed) = store
            .complete_task(
                &first.id,
                &WorkerId::from_string("w-c0"),
                &LeaseId::from_string("l-c0"),
                t_plus(11),
            )
            .await
            .context("complete c0")?;
        let observed = observed.context("parent returned")?;
        assert_eq!(observed.status, TaskStatus::WaitingOnChildren);
        assert_eq!(observed.pending_child_count, 2);
        assert!(observed.state.continuation().is_some());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complete_task_recomputes_from_index_not_caller_counter() -> Result<()> {
        // Phase 2.6 derives the parent's counter from the live
        // `by_parent` index, not a caller-maintained running total.
        // Drive a child to Completed out-of-band through `update()`
        // (simulating an older buggy path), then call
        // `complete_task` on a sibling. The recompute must see the
        // pre-completed child as terminal and report the parent's
        // new live count as 1, not 2.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-recompute").await?;
        let parent_id = parent.id.clone();

        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                ],
                SuspensionPayload {
                    continuation: sample_continuation("t-recompute"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Drive child[0] to Completed out-of-band.
        let c0 = children[0].clone();
        let c0_running = c0
            .mark_running(
                WorkerId::from_string("w-oob"),
                LeaseId::from_string("l-oob"),
                t_plus(60),
                t_plus(3),
            )
            .context("oob running")?;
        store
            .update(c0_running.clone())
            .await
            .context("update oob")?;
        let c0_done = c0_running.complete(t_plus(4)).context("oob complete")?;
        store.update(c0_done).await.context("update oob complete")?;

        // Now complete child[1] through the real path. The
        // recompute must see 1 live child (child[2]), not 2.
        let c1 = &children[1];
        store
            .try_acquire_task(
                &c1.id,
                WorkerId::from_string("w-c1"),
                LeaseId::from_string("l-c1"),
                t_plus(60),
                t_plus(5),
            )
            .await
            .context("acquire c1")?
            .context("c1 claimed")?;
        let (_, observed) = store
            .complete_task(
                &c1.id,
                &WorkerId::from_string("w-c1"),
                &LeaseId::from_string("l-c1"),
                t_plus(6),
            )
            .await
            .context("complete c1")?;
        let observed = observed.context("parent returned")?;
        assert_eq!(
            observed.pending_child_count, 1,
            "counter must be derived from live-children index, not subtraction"
        );
        assert_eq!(observed.status, TaskStatus::WaitingOnChildren);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complete_task_rejects_wrong_worker_or_lease() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-cc-cas").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-cc-cas"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;
        let child = &children[0];
        store
            .try_acquire_task(
                &child.id,
                WorkerId::from_string("w-c"),
                LeaseId::from_string("l-c"),
                t_plus(60),
                t_plus(3),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        let err = store
            .complete_task(
                &child.id,
                &WorkerId::from_string("w-imposter"),
                &LeaseId::from_string("l-c"),
                t_plus(4),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("worker mismatch"),
            "unexpected: {err:#}"
        );

        let err = store
            .complete_task(
                &child.id,
                &WorkerId::from_string("w-c"),
                &LeaseId::from_string("l-stale"),
                t_plus(5),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("lease mismatch"),
            "unexpected: {err:#}"
        );

        // Child is still Running with its original lease; parent
        // still waiting.
        let persisted_child = store
            .get(&child.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted_child.status, TaskStatus::Running);
        let persisted_parent = store
            .get(&parent.id)
            .await
            .context("get parent")?
            .context("parent exists")?;
        assert_eq!(persisted_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(persisted_parent.pending_child_count, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fail_task_mirrors_complete_task_but_stamps_last_error() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-fail").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-fail"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Complete one, fail the other.
        let first = &children[0];
        store
            .try_acquire_task(
                &first.id,
                WorkerId::from_string("w-a"),
                LeaseId::from_string("l-a"),
                t_plus(60),
                t_plus(3),
            )
            .await
            .context("acquire a")?
            .context("a claimed")?;
        store
            .complete_task(
                &first.id,
                &WorkerId::from_string("w-a"),
                &LeaseId::from_string("l-a"),
                t_plus(4),
            )
            .await
            .context("complete a")?;

        let second = &children[1];
        store
            .try_acquire_task(
                &second.id,
                WorkerId::from_string("w-b"),
                LeaseId::from_string("l-b"),
                t_plus(60),
                t_plus(5),
            )
            .await
            .context("acquire b")?
            .context("b claimed")?;
        let (failed, observed_parent) = store
            .fail_task(
                &second.id,
                &WorkerId::from_string("w-b"),
                &LeaseId::from_string("l-b"),
                "tool timed out".into(),
                t_plus(6),
            )
            .await
            .context("fail b")?;
        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(failed.last_error.as_deref(), Some("tool timed out"));
        let observed_parent = observed_parent.context("parent returned")?;
        // Second child is the last live one → parent resumes.
        assert_eq!(observed_parent.status, TaskStatus::Pending);
        assert_eq!(observed_parent.pending_child_count, 0);

        // `list_children` still returns the full batch with their
        // terminal statuses so the agent loop can aggregate.
        let listed = store.list_children(&parent.id).await.context("list")?;
        assert_eq!(listed.len(), 2);
        let statuses: std::collections::HashSet<_> = listed.iter().map(|c| c.status).collect();
        assert!(statuses.contains(&TaskStatus::Completed));
        assert!(statuses.contains(&TaskStatus::Failed));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mixed_success_failure_batch_resumes_parent() -> Result<()> {
        // Three children: success, failure, success. The parent
        // resumes after the last child reaches a terminal state,
        // regardless of which child failed.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-mixed").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                ],
                SuspensionPayload {
                    continuation: sample_continuation("t-mixed"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Claim + terminate each child in a different way.
        for (i, child) in children.iter().enumerate() {
            let offset = i64::try_from(i).context("enumerate fits i64")?;
            let worker_id = WorkerId::from_string(format!("w-m-{i}"));
            let lease_id = LeaseId::from_string(format!("l-m-{i}"));
            store
                .try_acquire_task(
                    &child.id,
                    worker_id.clone(),
                    lease_id.clone(),
                    t_plus(60 + offset),
                    t_plus(10 + offset),
                )
                .await
                .context("acquire")?
                .context("claimed")?;
            let (_, observed) = if i == 1 {
                store
                    .fail_task(
                        &child.id,
                        &worker_id,
                        &lease_id,
                        format!("child {i} failed"),
                        t_plus(20 + offset),
                    )
                    .await
                    .context("fail")?
            } else {
                store
                    .complete_task(&child.id, &worker_id, &lease_id, t_plus(20 + offset))
                    .await
                    .context("complete")?
            };
            let observed = observed.context("parent returned")?;
            if i == children.len() - 1 {
                assert_eq!(observed.status, TaskStatus::Pending);
            } else {
                assert_eq!(observed.status, TaskStatus::WaitingOnChildren);
            }
        }

        // The agent loop can now inspect the aggregated outcome via
        // `list_children`.
        let listed = store.list_children(&parent.id).await.context("list")?;
        assert_eq!(listed.len(), 3);
        let success_count = listed
            .iter()
            .filter(|c| c.status == TaskStatus::Completed)
            .count();
        let failure_count = listed
            .iter()
            .filter(|c| c.status == TaskStatus::Failed)
            .count();
        assert_eq!(success_count, 2);
        assert_eq!(failure_count, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complete_task_on_already_cancelled_parent_is_silent() -> Result<()> {
        // Edge case: a tree cancel lands between the worker's
        // mark_running and its complete_task call. The child's
        // terminal transition still runs (so the stale worker can
        // report its outcome), but the parent is already Cancelled
        // and must be left alone.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-late-complete").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-late-complete"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        let child = &children[0];
        store
            .try_acquire_task(
                &child.id,
                WorkerId::from_string("w-late"),
                LeaseId::from_string("l-late"),
                t_plus(60),
                t_plus(3),
            )
            .await
            .context("acquire late")?
            .context("late claimed")?;

        // Tree cancel lands while the child is still Running.
        let cancelled_ids = store
            .cancel_tree(&parent.id, t_plus(4))
            .await
            .context("cancel tree")?
            .transitioned;
        assert!(cancelled_ids.contains(&parent.id));
        assert!(cancelled_ids.contains(&child.id));

        // The late complete_task call now fails because the child
        // is no longer Running (tree cancel already transitioned it).
        let err = store
            .complete_task(
                &child.id,
                &WorkerId::from_string("w-late"),
                &LeaseId::from_string("l-late"),
                t_plus(5),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("not running"),
            "unexpected: {err:#}"
        );

        // Parent and every child are terminal.
        let persisted_parent = store
            .get(&parent.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted_parent.status, TaskStatus::Cancelled);
        for child in &children {
            let row = store
                .get(&child.id)
                .await
                .context("get child")?
                .context("child exists")?;
            assert_eq!(row.status, TaskStatus::Cancelled);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_cancels_root_and_all_tool_children() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-tree").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                ],
                SuspensionPayload {
                    continuation: sample_continuation("t-tree"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        let transitioned = store
            .cancel_tree(&parent.id, t_plus(3))
            .await
            .context("cancel tree")?
            .transitioned;
        // Root first, then every child.
        assert_eq!(transitioned.len(), 4);
        assert_eq!(transitioned[0], parent.id);
        for child in &children {
            assert!(
                transitioned.contains(&child.id),
                "child {child_id} missing from transitioned list",
                child_id = child.id
            );
        }

        // Root and children are Cancelled.
        let persisted_parent = store
            .get(&parent.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted_parent.status, TaskStatus::Cancelled);
        assert_eq!(persisted_parent.pending_child_count, 0);
        assert!(persisted_parent.state.is_none());
        for child in &children {
            let row = store
                .get(&child.id)
                .await
                .context("get child")?
                .context("child exists")?;
            assert_eq!(row.status, TaskStatus::Cancelled);
            assert!(row.worker_id.is_none());
            assert!(row.state.is_none());
        }

        // Runnable index is empty; a fresh scan returns None.
        let scanned = store
            .acquire_next_runnable(
                WorkerId::from_string("w-scan"),
                LeaseId::from_string("l-scan"),
                t_plus(60),
                t_plus(4),
            )
            .await
            .context("scan empty")?;
        assert!(scanned.is_none());

        // `list_children` still returns the rows for audit.
        let listed = store
            .list_children(&parent.id)
            .await
            .context("list after cancel")?;
        assert_eq!(listed.len(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_cancels_running_children_and_drops_leases() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-tree-live").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-tree-live"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Lease one child — it is now Running with a real lease.
        let leased_child = &children[0];
        store
            .try_acquire_task(
                &leased_child.id,
                WorkerId::from_string("w-live"),
                LeaseId::from_string("l-live"),
                t_plus(30),
                t_plus(3),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        let transitioned = store
            .cancel_tree(&parent.id, t_plus(4))
            .await
            .context("cancel tree")?
            .transitioned;
        assert_eq!(transitioned.len(), 3);

        // The leased child is now Cancelled with no lease.
        let row = store
            .get(&leased_child.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(row.status, TaskStatus::Cancelled);
        assert!(row.worker_id.is_none());
        assert!(row.lease_id.is_none());
        assert!(row.lease_expires_at.is_none());

        // Heartbeat from the stale worker now fails.
        let err = store
            .heartbeat_task(
                &leased_child.id,
                &WorkerId::from_string("w-live"),
                &LeaseId::from_string("l-live"),
                t_plus(60),
                None,
                t_plus(5),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("invalid transition"),
            "unexpected: {err:#}"
        );

        // complete_task from the stale worker also fails — the row
        // is no longer Running.
        let err = store
            .complete_task(
                &leased_child.id,
                &WorkerId::from_string("w-live"),
                &LeaseId::from_string("l-live"),
                t_plus(6),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("not running"),
            "unexpected: {err:#}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_is_idempotent_on_already_terminal_rows() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-tree-idem").await?;
        store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::default(), ChildSpawnSpec::default()],
                SuspensionPayload {
                    continuation: sample_continuation("t-tree-idem"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        let first = store
            .cancel_tree(&parent.id, t_plus(3))
            .await
            .context("cancel first")?
            .transitioned;
        assert_eq!(first.len(), 3);

        let second = store
            .cancel_tree(&parent.id, t_plus(4))
            .await
            .context("cancel second")?
            .transitioned;
        assert!(
            second.is_empty(),
            "second cancel_tree must be a no-op on terminal rows"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_on_waiting_root_clears_typed_state_and_counter() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-tree-waiting").await?;
        let (parked_parent, _) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                    ChildSpawnSpec::new(2),
                ],
                SuspensionPayload {
                    continuation: sample_continuation("t-tree-waiting"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;
        assert_eq!(parked_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(parked_parent.pending_child_count, 3);

        store
            .cancel_tree(&parent.id, t_plus(3))
            .await
            .context("cancel")?;

        let persisted = store
            .get(&parent.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(persisted.status, TaskStatus::Cancelled);
        assert_eq!(persisted.pending_child_count, 0);
        assert!(persisted.state.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_on_queued_root_removes_it_from_queue_index() -> Result<()> {
        // A queued root is a legitimate cancel target — calling
        // cancel_tree on it must drop it out of the per-thread
        // queue index so `list_queued_roots` stops returning it.
        let store = InMemoryAgentTaskStore::new();
        let blocker = fresh_root_at("t-tree-queue", 0);
        let queued = fresh_root_at("t-tree-queue", 1);
        store
            .submit_root_turn(blocker.clone())
            .await
            .context("blocker")?;
        let queued_admitted = store
            .submit_root_turn(queued.clone())
            .await
            .context("queued")?;
        assert_eq!(queued_admitted.status, TaskStatus::Queued);

        store
            .cancel_tree(&queued.id, t_plus(2))
            .await
            .context("cancel queued")?;

        let queue = store
            .list_queued_roots(&thread("t-tree-queue"))
            .await
            .context("list queue")?;
        assert!(queue.is_empty(), "queue should drop cancelled entry");

        // The blocker is untouched.
        let active = store
            .active_root_for_thread(&thread("t-tree-queue"))
            .await
            .context("active")?
            .context("blocker still active")?;
        assert_eq!(active.id, blocker.id);
        Ok(())
    }

    /// Journal that refuses to append the terminal marker, standing in
    /// for any sink failure (event store, outbox, thread projection).
    #[derive(Clone)]
    struct RefusingMarkerEventRepo {
        inner: crate::journal::event_repository::InMemoryEventRepository,
        /// Refuse only this thread's marker, so a cascade can fail at its
        /// SECOND candidate; `None` refuses every marker.
        refuse_on: Option<ThreadId>,
        /// Cleared to heal the sink for a retry.
        refusing: Arc<std::sync::atomic::AtomicBool>,
    }

    impl RefusingMarkerEventRepo {
        fn refuses(&self, thread_id: &ThreadId) -> bool {
            self.refusing.load(std::sync::atomic::Ordering::SeqCst)
                && self
                    .refuse_on
                    .as_ref()
                    .is_none_or(|refused| refused == thread_id)
        }
    }

    #[async_trait]
    impl EventRepository for RefusingMarkerEventRepo {
        async fn commit_event(
            &self,
            thread_id: &ThreadId,
            event: AgentEvent,
            now: OffsetDateTime,
        ) -> Result<CommittedEvent> {
            if matches!(event, AgentEvent::Cancelled { .. }) && self.refuses(thread_id) {
                anyhow::bail!("injected marker-commit failure");
            }
            self.inner.commit_event(thread_id, event, now).await
        }
        async fn commit_event_batch(
            &self,
            thread_id: &ThreadId,
            events: Vec<AgentEvent>,
            now: OffsetDateTime,
        ) -> Result<Vec<CommittedEvent>> {
            self.inner.commit_event_batch(thread_id, events, now).await
        }
        async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
            self.inner.next_sequence(thread_id).await
        }
        async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
            self.inner.get_events(thread_id).await
        }
        async fn get_events_in_range(
            &self,
            thread_id: &ThreadId,
            after: u64,
            up_to: u64,
        ) -> Result<Vec<CommittedEvent>> {
            self.inner
                .get_events_in_range(thread_id, after, up_to)
                .await
        }
        async fn threads_with_events_before(
            &self,
            cutoff: OffsetDateTime,
            limit: u32,
        ) -> Result<Vec<ThreadId>> {
            self.inner.threads_with_events_before(cutoff, limit).await
        }
        async fn max_sequence_before(
            &self,
            thread_id: &ThreadId,
            cutoff: OffsetDateTime,
        ) -> Result<Option<u64>> {
            self.inner.max_sequence_before(thread_id, cutoff).await
        }
        async fn min_sequence_at_or_after(
            &self,
            thread_id: &ThreadId,
            cutoff: OffsetDateTime,
        ) -> Result<Option<u64>> {
            self.inner.min_sequence_at_or_after(thread_id, cutoff).await
        }
    }

    /// A `Cancelled` row always has its marker in the journal. The
    /// in-memory backend has no cross-store transaction, so it gets that
    /// invariant from ordering: the marker is journaled first, and a sink
    /// failure aborts with the tree untouched.
    ///
    /// Without it a cancelled root could exist with no marker and no
    /// successor — nothing would ever close its followers, because a
    /// terminal frame from a cancelled emitter is deliberately suppressed
    /// and the idempotent retry emits nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn failed_marker_commit_leaves_the_tree_cancellable() -> Result<()> {
        let events = crate::journal::event_repository::InMemoryEventRepository::new();
        let threads = crate::journal::thread_store::InMemoryThreadStore::new();
        let store =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(RefusingMarkerEventRepo {
                    inner: events.clone(),
                    refuse_on: None,
                    refusing: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                }),
                outbox_store: Arc::new(crate::journal::outbox::InMemoryOutboxStore::new()),
                thread_store: Arc::new(threads.clone()),
            });

        let root = fresh_root("t-marker-refused");
        threads.get_or_create(&root.thread_id, t0()).await?;
        store.submit_root_turn(root.clone()).await?;

        let error = store
            .cancel_tree(&root.id, t_plus(2))
            .await
            .expect_err("a marker-commit failure must fail the cancel");
        assert!(
            format!("{error:#}").contains("injected marker-commit failure"),
            "{error:#}",
        );

        // Nothing transitioned: the row is still live, so no follower can
        // read it as a cancelled emitter, and a retry can still cancel it.
        let row = store.get(&root.id).await?.context("row")?;
        assert_ne!(
            row.status,
            TaskStatus::Cancelled,
            "a cancelled row without its marker would strand every follower",
        );
        assert!(
            events.get_events(&root.thread_id).await?.is_empty(),
            "the refused marker must not be journaled either",
        );

        // The active-root slot was never freed, so the thread is intact.
        let active = store
            .active_root_for_thread(&root.thread_id)
            .await?
            .context("the root must still hold its slot")?;
        assert_eq!(active.id, root.id);
        Ok(())
    }

    /// The healthy path still cancels and still journals the marker —
    /// the ordering change must not turn markers off.
    #[tokio::test(flavor = "multi_thread")]
    async fn successful_cancel_journals_its_marker_before_transitioning() -> Result<()> {
        let events = crate::journal::event_repository::InMemoryEventRepository::new();
        let threads = crate::journal::thread_store::InMemoryThreadStore::new();
        let store =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(events.clone()),
                outbox_store: Arc::new(crate::journal::outbox::InMemoryOutboxStore::new()),
                thread_store: Arc::new(threads.clone()),
            });

        let root = fresh_root("t-marker-committed");
        threads.get_or_create(&root.thread_id, t0()).await?;
        store.submit_root_turn(root.clone()).await?;

        let outcome = store.cancel_tree(&root.id, t_plus(2)).await?;
        assert_eq!(outcome.transitioned, vec![root.id.clone()]);
        assert_eq!(outcome.markers.len(), 1);

        let row = store.get(&root.id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Cancelled);

        let journal = events.get_events(&root.thread_id).await?;
        let marker = journal
            .iter()
            .find(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
            .context("a cancelled root must journal its marker")?;
        assert_eq!(
            marker.event.emitter_task_id(),
            Some(root.id.as_str()),
            "the marker names the cancelled root",
        );
        Ok(())
    }

    /// A cascade whose SECOND marker fails leaves the FIRST marker
    /// orphaned on a root that is still live — the documented residual,
    /// pinned so it cannot drift silently. The retry then HEALS the tree
    /// without duplicating that marker: the append is idempotent per
    /// candidate, so exactly one `Cancelled` marker per effective cancel
    /// survives a partial failure.
    ///
    /// The all-or-nothing flips are what make the retry possible at all:
    /// a flip clears the typed state carrying `SubagentInvocation`
    /// linkage, so pairing each marker with its own flip would leave a
    /// half-flipped cascade whose child-thread root `collect_subtree` can
    /// no longer reach — stranding it live forever. The invariant the
    /// close gate depends on holds throughout: no row is ever `Cancelled`
    /// without its marker.
    #[tokio::test(flavor = "multi_thread")]
    async fn partial_marker_failure_orphans_markers_but_keeps_the_tree_healable() -> Result<()> {
        let events = crate::journal::event_repository::InMemoryEventRepository::new();
        let outbox = crate::journal::outbox::InMemoryOutboxStore::new();
        let refusing = Arc::new(std::sync::atomic::AtomicBool::new(true));
        // The cascade's second marker candidate is the child-thread root;
        // refuse only its thread, so the parent's marker commits first.
        let child_thread = thread("t-partial-child");
        let store =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(RefusingMarkerEventRepo {
                    inner: events.clone(),
                    refuse_on: Some(child_thread.clone()),
                    refusing: Arc::clone(&refusing),
                }),
                outbox_store: Arc::new(outbox.clone()),
                thread_store: Arc::new(crate::journal::thread_store::InMemoryThreadStore::new()),
            });

        // parent (root) → invocation → child_root (root on its own thread):
        // two marker candidates.
        let (parent, invocation, child_root) = spawn_subagent_fixture(&store, "t-partial").await?;
        assert_eq!(child_root.thread_id, child_thread, "fixture topology");

        let error = store
            .cancel_tree(&parent.id, t_plus(10))
            .await
            .expect_err("the child root's marker must fail the cancel");
        assert!(
            format!("{error:#}").contains("injected marker-commit failure"),
            "{error:#}",
        );

        // The gate-critical half holds: NOTHING is cancelled, so no
        // follower can read a cancelled emitter with no marker.
        for id in [&parent.id, &invocation.id, &child_root.id] {
            assert_ne!(
                store.get(id).await?.context("row")?.status,
                TaskStatus::Cancelled,
                "a partial marker set must not cancel anything",
            );
        }

        // The residual: the parent's marker is already journaled, on a
        // root that is still running.
        assert_eq!(
            cancel_markers(&events, &parent.thread_id).await?,
            1,
            "the first marker is orphaned on a live root",
        );
        assert_eq!(cancel_markers(&events, &child_thread).await?, 0);

        // The tree is still whole, so a retry completes the cancellation —
        // this is what the all-or-nothing flip buys, and what per-row
        // pairing would destroy (the invocation's cleared linkage would
        // hide the child root from `collect_subtree`).
        refusing.store(false, std::sync::atomic::Ordering::SeqCst);
        let outcome = store.cancel_tree(&parent.id, t_plus(11)).await?;
        assert_eq!(outcome.transitioned.len(), 3, "the retry heals the tree");
        for id in [&parent.id, &invocation.id, &child_root.id] {
            assert_eq!(
                store.get(id).await?.context("row")?.status,
                TaskStatus::Cancelled,
            );
        }

        // ...reusing the marker it already wrote rather than writing a
        // second one: exactly one marker per effective cancel, even
        // across a partial failure and its retry.
        assert_eq!(
            cancel_markers(&events, &parent.thread_id).await?,
            1,
            "the retry must reuse the orphaned marker, not duplicate it",
        );
        assert_eq!(
            cancel_markers(&events, &child_thread).await?,
            1,
            "the child root's marker is written exactly once by the retry",
        );

        // The reused marker is still handed back, so the caller's notifier
        // wakes same-process followers with it.
        assert_eq!(outcome.markers.len(), 2, "both markers are reported");

        // The parent's advisory was queued by the FIRST attempt, so the
        // reuse must not queue a second wake for the same marker.
        assert_eq!(
            marker_advisories(&outbox, &parent.thread_id).await?,
            1,
            "a reused marker must not be announced twice",
        );
        assert_eq!(marker_advisories(&outbox, &child_thread).await?, 1);
        Ok(())
    }

    /// Outbox that refuses to queue advisories, standing in for an
    /// attempt that journaled its marker and then died before announcing
    /// it.
    #[derive(Clone)]
    struct RefusingAdvisoryOutbox {
        inner: crate::journal::outbox::InMemoryOutboxStore,
        refusing: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl crate::journal::outbox::OutboxStore for RefusingAdvisoryOutbox {
        async fn insert_batch(
            &self,
            rows: Vec<crate::journal::outbox::NewOutboxRow>,
        ) -> Result<Vec<crate::journal::outbox::OutboxRow>> {
            if self.refusing.load(std::sync::atomic::Ordering::SeqCst) {
                anyhow::bail!("injected advisory-insert failure");
            }
            self.inner.insert_batch(rows).await
        }
        async fn claim_pending(
            &self,
            worker_id: &str,
            limit: u32,
            now: OffsetDateTime,
        ) -> Result<Vec<crate::journal::outbox::OutboxRow>> {
            self.inner.claim_pending(worker_id, limit, now).await
        }
        async fn mark_delivered(
            &self,
            id: &crate::journal::outbox::OutboxRowId,
            worker_id: &str,
            now: OffsetDateTime,
        ) -> Result<()> {
            self.inner.mark_delivered(id, worker_id, now).await
        }
        async fn mark_failed(
            &self,
            id: &crate::journal::outbox::OutboxRowId,
            worker_id: &str,
            error: &str,
            next_attempt_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<()> {
            self.inner
                .mark_failed(id, worker_id, error, next_attempt_at, now)
                .await
        }
        async fn reclaim_expired_claims(
            &self,
            now: OffsetDateTime,
            claim_lease: Duration,
        ) -> Result<u64> {
            self.inner.reclaim_expired_claims(now, claim_lease).await
        }
        async fn get(
            &self,
            id: &crate::journal::outbox::OutboxRowId,
        ) -> Result<Option<crate::journal::outbox::OutboxRow>> {
            self.inner.get(id).await
        }
        async fn list_by_thread(
            &self,
            thread_id: &ThreadId,
        ) -> Result<Vec<crate::journal::outbox::OutboxRow>> {
            self.inner.list_by_thread(thread_id).await
        }
        async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
            self.inner.count_pending(thread_id).await
        }
        async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>> {
            self.inner.min_unpublished_sequence(thread_id).await
        }
    }

    /// A marker is journaled BEFORE its advisory, so an attempt can leave
    /// a marker no relay would ever announce. The retry's reuse branch
    /// must repair that: a parked root commits no further event, so
    /// without the advisory its cross-host followers would never be woken
    /// even though `cancel_tree` reported success.
    #[tokio::test(flavor = "multi_thread")]
    async fn retry_supplies_the_advisory_a_reused_marker_never_got() -> Result<()> {
        let events = crate::journal::event_repository::InMemoryEventRepository::new();
        let outbox = crate::journal::outbox::InMemoryOutboxStore::new();
        let refusing = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let store =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(events.clone()),
                outbox_store: Arc::new(RefusingAdvisoryOutbox {
                    inner: outbox.clone(),
                    refusing: Arc::clone(&refusing),
                }),
                thread_store: Arc::new(crate::journal::thread_store::InMemoryThreadStore::new()),
            });

        let root = fresh_root("t-advisory-refused");
        store.submit_root_turn(root.clone()).await?;

        // Attempt 1: the marker lands, the advisory does not.
        let error = store
            .cancel_tree(&root.id, t_plus(2))
            .await
            .expect_err("a refused advisory must fail the cancel");
        assert!(
            format!("{error:#}").contains("injected advisory-insert failure"),
            "{error:#}",
        );
        assert_eq!(cancel_markers(&events, &root.thread_id).await?, 1);
        assert_eq!(
            marker_advisories(&outbox, &root.thread_id).await?,
            0,
            "the advisory was refused",
        );
        assert_ne!(
            store.get(&root.id).await?.context("row")?.status,
            TaskStatus::Cancelled,
            "nothing flips while the marker set is incomplete",
        );

        // Attempt 2 reuses the marker — and must supply the advisory it
        // never got, exactly once.
        refusing.store(false, std::sync::atomic::Ordering::SeqCst);
        store.cancel_tree(&root.id, t_plus(3)).await?;

        assert_eq!(
            store.get(&root.id).await?.context("row")?.status,
            TaskStatus::Cancelled,
        );
        assert_eq!(
            cancel_markers(&events, &root.thread_id).await?,
            1,
            "still exactly one marker",
        );
        assert_eq!(
            marker_advisories(&outbox, &root.thread_id).await?,
            1,
            "the reused marker must end up announced exactly once",
        );
        Ok(())
    }

    /// An advisory whose relay attempts are exhausted is terminal and will
    /// never be claimed, so it announces nothing. A retry that reuses the
    /// marker must queue a FRESH advisory rather than count the dead one —
    /// otherwise the cancel reports success while cross-host followers of
    /// a parked root are never woken at all.
    #[tokio::test(flavor = "multi_thread")]
    async fn retry_replaces_an_expired_advisory_on_a_reused_marker() -> Result<()> {
        let events = crate::journal::event_repository::InMemoryEventRepository::new();
        let outbox = crate::journal::outbox::InMemoryOutboxStore::new();
        let refusing = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let child_thread = thread("t-expired-child");
        let store =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(RefusingMarkerEventRepo {
                    inner: events.clone(),
                    refuse_on: Some(child_thread.clone()),
                    refusing: Arc::clone(&refusing),
                }),
                outbox_store: Arc::new(outbox.clone()),
                thread_store: Arc::new(crate::journal::thread_store::InMemoryThreadStore::new()),
            });

        // Attempt 1: the parent's marker AND its advisory land; the child's
        // marker fails, so nothing flips and the parent stays live.
        let (parent, _invocation, _child_root) =
            spawn_subagent_fixture(&store, "t-expired").await?;
        store
            .cancel_tree(&parent.id, t_plus(10))
            .await
            .expect_err("the child root's marker must fail the cancel");
        assert_eq!(marker_advisories(&outbox, &parent.thread_id).await?, 1);

        // The relay exhausts that advisory: terminal, never claimable again.
        expire_thread_advisories(&outbox, &parent.thread_id).await?;
        assert!(
            claimable_advisories(&outbox, &parent.thread_id)
                .await?
                .is_empty(),
            "precondition: the advisory is dead",
        );

        // The retry reuses the marker — and must re-announce it.
        refusing.store(false, std::sync::atomic::Ordering::SeqCst);
        store.cancel_tree(&parent.id, t_plus(11)).await?;

        assert_eq!(
            store.get(&parent.id).await?.context("parent")?.status,
            TaskStatus::Cancelled,
        );
        assert_eq!(
            cancel_markers(&events, &parent.thread_id).await?,
            1,
            "still exactly one marker",
        );
        let claimable = claimable_advisories(&outbox, &parent.thread_id).await?;
        assert_eq!(
            claimable.len(),
            1,
            "the cancelled root must end up with a publishable wake",
        );
        Ok(())
    }

    /// Burn every pending advisory on a thread through its retry budget
    /// until the relay marks it `Expired`.
    async fn expire_thread_advisories(
        outbox: &crate::journal::outbox::InMemoryOutboxStore,
        thread_id: &ThreadId,
    ) -> Result<()> {
        use crate::journal::outbox::OutboxStore as _;
        let worker = "relay-test";
        // Later than every insert in these fixtures, so the rows are
        // eligible for pickup (`claim_pending` gates on `next_attempt_at`).
        let relay_now = t_plus(100);
        for _ in 0..=DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS {
            let claimed = outbox.claim_pending(worker, 16, relay_now).await?;
            if claimed.is_empty() {
                break;
            }
            for row in claimed {
                outbox
                    .mark_failed(&row.id, worker, "relay down", relay_now, relay_now)
                    .await?;
            }
        }
        let dead = outbox
            .list_by_thread(thread_id)
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::ThreadEventsAvailable)
            .all(|row| row.status == OutboxStatus::Expired);
        anyhow::ensure!(dead, "the advisories must be exhausted for this fixture");
        Ok(())
    }

    /// Advisories the relay can still pick up.
    async fn claimable_advisories(
        outbox: &crate::journal::outbox::InMemoryOutboxStore,
        thread_id: &ThreadId,
    ) -> Result<Vec<crate::journal::outbox::OutboxRow>> {
        use crate::journal::outbox::OutboxStore as _;
        Ok(outbox
            .list_by_thread(thread_id)
            .await?
            .into_iter()
            .filter(|row| {
                row.kind == OutboxMessageKind::ThreadEventsAvailable
                    && row.status != OutboxStatus::Expired
            })
            .collect())
    }

    async fn marker_advisories(
        outbox: &crate::journal::outbox::InMemoryOutboxStore,
        thread_id: &ThreadId,
    ) -> Result<usize> {
        use crate::journal::outbox::OutboxStore as _;
        Ok(outbox
            .list_by_thread(thread_id)
            .await?
            .iter()
            .filter(|row| row.kind == OutboxMessageKind::ThreadEventsAvailable)
            .count())
    }

    async fn cancel_markers(
        events: &crate::journal::event_repository::InMemoryEventRepository,
        thread_id: &ThreadId,
    ) -> Result<usize> {
        Ok(events
            .get_events(thread_id)
            .await?
            .iter()
            .filter(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
            .count())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_frees_the_active_root_slot_on_a_thread() -> Result<()> {
        // Phase 2.2 interaction: cancelling the blocking root via
        // `cancel_tree` must free the active-root slot so the next
        // queued root can promote.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t-tree-promote", 0);
        let second = fresh_root_at("t-tree-promote", 1);
        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        store
            .cancel_tree(&first.id, t_plus(2))
            .await
            .context("cancel tree")?;

        // Finding #4/#8: cancelling the blocking root auto-promotes the
        // queued successor in the SAME locked scope, so `second` is
        // already Pending and holds the slot — no explicit promote needed.
        let active = store
            .active_root_for_thread(&thread("t-tree-promote"))
            .await
            .context("active")?
            .context("slot should be held by the auto-promoted successor")?;
        assert_eq!(active.id, second.id);
        assert_eq!(active.status, TaskStatus::Pending);

        // A redundant explicit promote is an idempotent no-op now that the
        // slot is held by the freshly-promoted root.
        let redundant = store
            .promote_next_queued_root(&thread("t-tree-promote"), t_plus(3))
            .await
            .context("redundant promote")?;
        assert!(redundant.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_rejects_missing_root() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let ghost = AgentTaskId::new();
        let err = store.cancel_tree(&ghost, t_plus(1)).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("does not exist"),
            "unexpected: {err:#}"
        );
        Ok(())
    }

    // ── Phase 7.6: Cascade cancellation across subagent trees ─────

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_cascades_through_subagent_invocation_to_child_thread() -> Result<()> {
        // Acceptance criterion: "Parent cancellation stops the entire
        // descendant subagent tree deterministically."
        //
        // Topology: parent (root_turn) → invocation (subagent) → child_root (root_turn, different thread)
        // cancel_tree(parent) must cancel all three.
        let store = InMemoryAgentTaskStore::new();
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(&store, "t-cascade-parent").await?;

        let transitioned = store
            .cancel_tree(&parent.id, t_plus(10))
            .await?
            .transitioned;

        // All three tasks must have been transitioned.
        assert!(
            transitioned.contains(&parent.id),
            "parent must be in transitioned set"
        );
        assert!(
            transitioned.contains(&invocation.id),
            "invocation must be in transitioned set"
        );
        assert!(
            transitioned.contains(&child_root.id),
            "child_root must be in transitioned set"
        );
        assert_eq!(transitioned.len(), 3);

        // Every task must be in terminal Cancelled state.
        let parent_after = store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(parent_after.status, TaskStatus::Cancelled);

        let invocation_after = store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(invocation_after.status, TaskStatus::Cancelled);

        let child_after = store.get(&child_root.id).await?.context("child exists")?;
        assert_eq!(child_after.status, TaskStatus::Cancelled);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_cascades_through_nested_subagent_trees() -> Result<()> {
        // Acceptance criterion: "Nested subagents follow the same
        // cancellation and failure rules recursively."
        //
        // Topology:
        //   grandparent (root_turn, thread A)
        //     → invocation_1 (subagent, thread A)
        //       → child_root_1 (root_turn, thread B)
        //         → invocation_2 (subagent, thread B)
        //           → grandchild_root (root_turn, thread C)
        //
        // cancel_tree(grandparent) must cancel all five.
        let store = InMemoryAgentTaskStore::new();

        // Level 1: grandparent → invocation_1 → child_root_1
        let (grandparent, inv1, child1) = spawn_subagent_fixture(&store, "t-nested-gp").await?;

        // Level 2: child_root_1 is Pending on its child thread.
        // Acquire it, then spawn a nested subagent.
        let child1_running = store
            .try_acquire_task(
                &child1.id,
                WorkerId::from_string("w-child1"),
                LeaseId::from_string("l-child1"),
                t_plus(60),
                t_plus(3),
            )
            .await?
            .context("claim child_root_1")?;
        assert_eq!(child1_running.status, TaskStatus::Running);

        let child_thread_2 = thread("t-nested-gc");
        let (child1_waiting, inv2, grandchild) = store
            .spawn_subagent_invocation(
                &child1.id,
                &WorkerId::from_string("w-child1"),
                &LeaseId::from_string("l-child1"),
                SubagentInvocationSpawn {
                    child_thread_id: child_thread_2,
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    payload: SuspensionPayload {
                        continuation: sample_continuation("t-nested-gp-child"),
                        suspended_messages: Vec::new(),
                    },
                    child_caller_metadata: None,
                },
                t_plus(4),
            )
            .await
            .context("spawn nested subagent")?;
        assert_eq!(child1_waiting.status, TaskStatus::WaitingOnChildren);

        // Now cancel the entire tree from the grandparent.
        let transitioned = store
            .cancel_tree(&grandparent.id, t_plus(20))
            .await?
            .transitioned;

        // All five tasks must be cancelled.
        assert_eq!(
            transitioned.len(),
            5,
            "five tasks must be transitioned: {transitioned:?}"
        );
        for id in [
            &grandparent.id,
            &inv1.id,
            &child1.id,
            &inv2.id,
            &grandchild.id,
        ] {
            assert!(
                transitioned.contains(id),
                "task {id} must be in transitioned set"
            );
            let task = store
                .get(id)
                .await?
                .with_context(|| format!("{id} exists"))?;
            assert_eq!(
                task.status,
                TaskStatus::Cancelled,
                "task {id} must be Cancelled"
            );
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_idempotent_on_already_cancelled_subagent_tree() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(&store, "t-cascade-idempotent").await?;

        // First cancel.
        let first = store
            .cancel_tree(&parent.id, t_plus(10))
            .await?
            .transitioned;
        assert_eq!(first.len(), 3);

        // Second cancel — all tasks are already terminal.
        let second = store
            .cancel_tree(&parent.id, t_plus(11))
            .await?
            .transitioned;
        assert!(
            second.is_empty(),
            "second cancel must be a no-op on terminal rows"
        );

        // All tasks still Cancelled.
        for id in [&parent.id, &invocation.id, &child_root.id] {
            let task = store
                .get(id)
                .await?
                .with_context(|| format!("{id} exists"))?;
            assert_eq!(task.status, TaskStatus::Cancelled);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_tree_with_running_child_thread_root_drops_lease() -> Result<()> {
        // Verify that cascading cancellation through a subagent invocation
        // drops the lease on a running child-thread root, so stale workers
        // fail CAS on their next heartbeat.
        let store = InMemoryAgentTaskStore::new();
        let (parent, _invocation, child_root) =
            spawn_subagent_fixture(&store, "t-cascade-lease").await?;

        // Acquire and run the child root.
        let child_running = store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-child"),
                LeaseId::from_string("l-child"),
                t_plus(60),
                t_plus(3),
            )
            .await?
            .context("claim child root")?;
        assert_eq!(child_running.status, TaskStatus::Running);

        // Cancel from the parent.
        store.cancel_tree(&parent.id, t_plus(10)).await?;

        // Child root lease is dropped.
        let child_after = store.get(&child_root.id).await?.context("child exists")?;
        assert_eq!(child_after.status, TaskStatus::Cancelled);
        assert!(child_after.worker_id.is_none());
        assert!(child_after.lease_id.is_none());

        // Stale worker cannot heartbeat.
        let err = store
            .heartbeat_task(
                &child_root.id,
                &WorkerId::from_string("w-child"),
                &LeaseId::from_string("l-child"),
                t_plus(120),
                None,
                t_plus(11),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("invalid transition"),
            "stale heartbeat should fail: {err:#}"
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_child_root_wakes_invocation_as_pending() -> Result<()> {
        // When a child root fails, the invocation task wakes up and
        // becomes runnable (Pending) with pending_child_count == 0,
        // so the worker can materialize the failure into a
        // deterministic error tool result on its next turn.
        //
        // NOTE: ToolResult materialization is tested at the worker
        // layer, not here — this test covers the store-level state
        // transition only.
        let store = InMemoryAgentTaskStore::new();
        let (parent, invocation, child_root) = spawn_subagent_fixture(&store, "t-fail-map").await?;

        // Acquire child root.
        let _child_running = store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-child"),
                LeaseId::from_string("l-child"),
                t_plus(60),
                t_plus(3),
            )
            .await?
            .context("claim child root")?;

        // Fail the child root with a specific error.
        let (failed_child, _) = store
            .fail_task(
                &child_root.id,
                &WorkerId::from_string("w-child"),
                &LeaseId::from_string("l-child"),
                "max turns exhausted: 5/5".to_owned(),
                t_plus(5),
            )
            .await
            .context("fail child root")?;
        assert_eq!(failed_child.status, TaskStatus::Failed);
        assert_eq!(
            failed_child.last_error.as_deref(),
            Some("max turns exhausted: 5/5")
        );

        // The invocation task should now be runnable (Pending) because
        // its only child reached terminal state.
        let inv_after = store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(inv_after.status, TaskStatus::Pending);
        assert_eq!(inv_after.pending_child_count, 0);

        // The parent should still be WaitingOnChildren (waiting on the invocation).
        let parent_after = store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(parent_after.status, TaskStatus::WaitingOnChildren);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelled_subagent_child_root_maps_to_deterministic_result() -> Result<()> {
        // Acceptance criterion: "Timeout and max-turn exhaustion map
        // into deterministic failed subagent results."
        //
        // When a child root is cancelled, the invocation wakes and
        // becomes runnable with its SubagentInvocation state preserved.
        let store = InMemoryAgentTaskStore::new();
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(&store, "t-cancel-map").await?;

        // Cancel only the child root (simulating a timeout cancellation).
        let transitioned = store
            .cancel_tree(&child_root.id, t_plus(5))
            .await?
            .transitioned;
        assert_eq!(transitioned, vec![child_root.id.clone()]);

        // Invocation should wake up (Pending, SubagentInvocation state preserved).
        let inv_after = store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(inv_after.status, TaskStatus::Pending);
        assert!(inv_after.state.subagent_invocation().is_some());

        // Parent should still be waiting on the invocation.
        let parent_after = store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(parent_after.status, TaskStatus::WaitingOnChildren);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn completed_child_batch_is_resumable_from_journal_alone() -> Result<()> {
        // Acceptance criterion: "Completed child-task batches can
        // make the parent resumable through journal state alone."
        // Spawn two children, complete both, then rehydrate the row
        // set into a **fresh** store instance via the serialized
        // wire form. The parent must still be runnable and its
        // typed state still cleared, matching the live store
        // exactly.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-journal").await?;
        let (_, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-journal"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        for (i, child) in children.iter().enumerate() {
            let offset = i64::try_from(i).context("enumerate fits i64")?;
            let worker_id = WorkerId::from_string(format!("w-j-{i}"));
            let lease_id = LeaseId::from_string(format!("l-j-{i}"));
            store
                .try_acquire_task(
                    &child.id,
                    worker_id.clone(),
                    lease_id.clone(),
                    t_plus(60),
                    t_plus(10 + offset),
                )
                .await
                .context("acquire")?
                .context("claimed")?;
            store
                .complete_task(&child.id, &worker_id, &lease_id, t_plus(20 + offset))
                .await
                .context("complete")?;
        }

        let live_parent = store
            .get(&parent.id)
            .await
            .context("live parent")?
            .context("parent exists")?;
        assert_eq!(live_parent.status, TaskStatus::Pending);
        assert!(matches!(live_parent.state, TaskState::ReadyToResume { .. }));

        // Round-trip every row through JSON and rebuild a fresh
        // store from the persisted bytes alone.
        let mut rows: Vec<AgentTask> = Vec::new();
        rows.push(live_parent.clone());
        for child in &children {
            let row = store
                .get(&child.id)
                .await
                .context("get live child")?
                .context("child exists")?;
            rows.push(row);
        }
        // Snapshot the rows as their wire form so the "journal
        // alone" claim is really about the durable shape.
        let mut serialized: Vec<String> = Vec::new();
        for row in &rows {
            serialized.push(serde_json::to_string(row).context("serialize")?);
        }
        let rehydrated_store = InMemoryAgentTaskStore::new();
        // Children must be inserted after the parent so the
        // cross-row parent-exists guard is satisfied.
        let mut rehydrated_rows: Vec<AgentTask> = Vec::with_capacity(serialized.len());
        for json in &serialized {
            rehydrated_rows.push(serde_json::from_str(json).context("deserialize")?);
        }
        for row in &rehydrated_rows {
            rehydrated_store
                .insert(row.clone())
                .await
                .context("rehydrate insert")?;
        }

        let rehydrated_parent = rehydrated_store
            .get(&parent.id)
            .await
            .context("rehydrate get parent")?
            .context("parent exists in rehydrated store")?;
        assert_eq!(rehydrated_parent.status, TaskStatus::Pending);
        assert!(matches!(
            rehydrated_parent.state,
            TaskState::ReadyToResume { .. }
        ));
        // The parent is runnable in the rehydrated store too.
        let scanned = rehydrated_store
            .acquire_next_runnable(
                WorkerId::from_string("w-rehydrate"),
                LeaseId::from_string("l-rehydrate"),
                t_plus(200),
                t_plus(100),
            )
            .await
            .context("scan rehydrated")?
            .context("scan returned row")?;
        assert_eq!(scanned.id, parent.id);
        assert_eq!(scanned.status, TaskStatus::Running);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn completed_child_with_no_live_siblings_has_empty_parent_counter() -> Result<()> {
        // Acceptance-criterion pin for the recompute edge case where
        // the parent's `by_parent` index still holds entries but
        // every one of them is already terminal. The
        // journal-derived counter must read as zero and the parent
        // must flip back to `Pending` with `TaskState::None` — exactly
        // the behavior `completed_child_batch_is_resumable_from_journal_alone`
        // relies on, but driven through the recompute path itself
        // so a regression in `count_live_children` would surface
        // here before it broke the resumability story.
        let store = InMemoryAgentTaskStore::new();
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-empty-counter").await?;
        let parent_id = parent.id.clone();
        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation("t-empty-counter"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        // Drive **every** child to Completed out-of-band via
        // `update()` so the Phase 2.6 recompute path has never run
        // but `by_parent` still holds both entries. The parent is
        // left in `WaitingOnChildren{2}` with a stale counter.
        for (i, child) in children.iter().enumerate() {
            let offset = i64::try_from(i).context("fits i64")?;
            let running = child
                .clone()
                .mark_running(
                    WorkerId::from_string(format!("w-oob-{i}")),
                    LeaseId::from_string(format!("l-oob-{i}")),
                    t_plus(60),
                    t_plus(10 + offset),
                )
                .context("oob running")?;
            store
                .update(running.clone())
                .await
                .context("update oob running")?;
            let done = running.complete(t_plus(20 + offset)).context("oob done")?;
            store.update(done).await.context("update oob done")?;
        }

        // Confirm the setup: parent still in WaitingOnChildren{2}
        // and every child present in `by_parent` but terminal.
        let stale_parent = store
            .get(&parent_id)
            .await
            .context("get stale parent")?
            .context("parent exists")?;
        assert_eq!(stale_parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(stale_parent.pending_child_count, 2);
        let still_children = store
            .list_children(&parent_id)
            .await
            .context("list children")?;
        assert_eq!(still_children.len(), 2);
        assert!(
            still_children.iter().all(|c| c.status.is_terminal()),
            "every child must be terminal before the recompute"
        );

        // Recompute the parent's counter directly via the pure
        // transition helper so the edge case is exercised without a
        // new child-terminal call site. Live count is zero because
        // every `by_parent` entry is already terminal.
        let recomputed = stale_parent
            .clone()
            .recompute_pending_children(0, t_plus(30))
            .context("recompute")?;
        assert_eq!(recomputed.status, TaskStatus::Pending);
        assert_eq!(recomputed.pending_child_count, 0);
        assert!(
            matches!(recomputed.state, TaskState::ReadyToResume { .. }),
            "terminal-count-zero must transition to ReadyToResume"
        );
        store
            .update(recomputed)
            .await
            .context("persist recompute")?;

        // The store now reflects the resumed parent via the
        // journal-derived counter alone — no complete_task call,
        // no caller-maintained counter.
        let final_parent = store
            .get(&parent_id)
            .await
            .context("get resumed parent")?
            .context("parent exists")?;
        assert_eq!(final_parent.status, TaskStatus::Pending);
        assert_eq!(final_parent.pending_child_count, 0);
        assert!(matches!(
            final_parent.state,
            TaskState::ReadyToResume { .. }
        ));
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.7 — Concurrency regression suite
    // ──────────────────────────────────────────────────────────────
    //
    // Every test below spawns real async tasks via `tokio::spawn` and
    // joins them, so the store's internal `RwLock` serialises the
    // mutations under genuine parallel scheduling.  The assertions
    // are scheduler-agnostic: they check final state invariants, not
    // interleaving order.

    // ── Root queue under concurrency ──────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_submits_on_same_thread_serialize_with_exactly_one_pending() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let count = 10usize;
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let st = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let root = AgentTask::new_root_turn(
                    thread("t-race"),
                    t_plus(i64::try_from(idx).unwrap()),
                    3,
                );
                st.submit_root_turn(root).await
            }));
        }
        let mut results = Vec::with_capacity(count);
        for handle in handles {
            results.push(handle.await.context("join")?.context("submit")?);
        }
        let pending = results
            .iter()
            .filter(|r| r.status == TaskStatus::Pending)
            .count();
        let queued = results
            .iter()
            .filter(|r| r.status == TaskStatus::Queued)
            .count();
        assert_eq!(pending, 1, "exactly one root must be Pending");
        assert_eq!(queued, count - 1, "the rest must be Queued");
        let active = store
            .active_root_for_thread(&thread("t-race"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.status, TaskStatus::Pending);
        let queue = store
            .list_queued_roots(&thread("t-race"))
            .await
            .context("queued")?;
        assert_eq!(queue.len(), count - 1);
        for pair in queue.windows(2) {
            assert!(
                pair[0].created_at <= pair[1].created_at,
                "FIFO ordering broken"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_promotes_fire_exactly_once_after_slot_frees() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let first = AgentTask::new_root_turn(thread("t-promo"), t_plus(0), 3);
        let second = AgentTask::new_root_turn(thread("t-promo"), t_plus(1), 3);
        let second_id = second.id.clone();
        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store.submit_root_turn(second).await.context("second")?;
        let running = store
            .try_acquire_task(
                &first.id,
                WorkerId::from_string("w-promo"),
                LeaseId::from_string("l-promo"),
                t_plus(60),
                t_plus(2),
            )
            .await
            .context("acquire")?
            .context("claimed")?;
        store
            .update(running.complete(t_plus(3)).context("complete")?)
            .await
            .context("done")?;

        let count = 8usize;
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let st = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                st.promote_next_queued_root(&thread("t-promo"), t_plus(4))
                    .await
            }));
        }
        let mut promoted = 0usize;
        for handle in handles {
            if handle.await.context("join")?.context("promote")?.is_some() {
                promoted += 1;
            }
        }
        assert_eq!(promoted, 1, "exactly one promotion must fire");
        let active = store
            .active_root_for_thread(&thread("t-promo"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, second_id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_submits_across_threads_never_cross_boundaries() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let threads_count = 5usize;
        let roots_per = 4usize;
        let mut handles = Vec::new();
        for ti in 0..threads_count {
            for ri in 0..roots_per {
                let st = Arc::clone(&store);
                let name = format!("t-cross-{ti}");
                let secs = i64::try_from(ti * roots_per + ri).unwrap();
                handles.push(tokio::spawn(async move {
                    let root =
                        AgentTask::new_root_turn(ThreadId::from_string(&name), t_plus(secs), 3);
                    st.submit_root_turn(root).await
                }));
            }
        }
        for handle in handles {
            handle.await.context("join")?.context("submit")?;
        }
        for ti in 0..threads_count {
            let tid = ThreadId::from_string(format!("t-cross-{ti}"));
            let active = store
                .active_root_for_thread(&tid)
                .await
                .context("active")?
                .context("slot")?;
            assert_eq!(active.status, TaskStatus::Pending);
            let queue = store.list_queued_roots(&tid).await.context("queued")?;
            assert_eq!(queue.len(), roots_per - 1);
        }
        Ok(())
    }

    // ── Runnable acquisition under concurrency ────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_try_acquire_same_id_yields_exactly_one_winner() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-acq1"), t_plus(0), 5);
        let task_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let count = 10usize;
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let st = Arc::clone(&store);
            let tid = task_id.clone();
            handles.push(tokio::spawn(async move {
                st.try_acquire_task(
                    &tid,
                    WorkerId::from_string(format!("w-{idx}")),
                    LeaseId::from_string(format!("l-{idx}")),
                    t_plus(60),
                    t_plus(1),
                )
                .await
            }));
        }
        let mut winners = 0usize;
        for handle in handles {
            if handle.await.context("join")?.context("acquire")?.is_some() {
                winners += 1;
            }
        }
        assert_eq!(winners, 1, "exactly one worker must win the claim");
        let row = store
            .get(&task_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(row.attempt, 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_scan_acquires_never_double_lease() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let task_count = 5usize;
        for idx in 0..task_count {
            let root = AgentTask::new_root_turn(
                thread(&format!("t-scan-{idx}")),
                t_plus(i64::try_from(idx).unwrap()),
                5,
            );
            store.submit_root_turn(root).await.context("submit")?;
        }
        let worker_count = 12usize;
        let mut handles = Vec::with_capacity(worker_count);
        for idx in 0..worker_count {
            let st = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                st.acquire_next_runnable(
                    WorkerId::from_string(format!("w-scan-{idx}")),
                    LeaseId::from_string(format!("l-scan-{idx}")),
                    t_plus(60),
                    t_plus(1),
                )
                .await
            }));
        }
        let mut claimed_ids = std::collections::HashSet::new();
        for handle in handles {
            if let Some(task) = handle.await.context("join")?.context("scan")? {
                assert_eq!(task.status, TaskStatus::Running);
                assert!(
                    claimed_ids.insert(task.id.clone()),
                    "double-lease on task {}",
                    task.id
                );
            }
        }
        assert_eq!(
            claimed_ids.len(),
            task_count,
            "exactly K rows should be claimed"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_acquire_skips_exhausted_head_fails_closed_once() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let exhausted = AgentTask::new_root_turn(thread("t-exhaust-conc"), t_plus(0), 1);
        let exhausted_id = exhausted.id.clone();
        store
            .submit_root_turn(exhausted)
            .await
            .context("submit exhausted")?;
        let running = store
            .try_acquire_task(
                &exhausted_id,
                WorkerId::from_string("w-ex"),
                LeaseId::from_string("l-ex"),
                t_plus(60),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;
        store
            .update(running.release_lease(t_plus(2)).context("release")?)
            .await
            .context("update")?;

        let healthy = AgentTask::new_root_turn(thread("t-healthy-conc"), t_plus(3), 5);
        let healthy_id = healthy.id.clone();
        store
            .submit_root_turn(healthy)
            .await
            .context("submit healthy")?;

        let count = 8usize;
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let st = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                st.acquire_next_runnable(
                    WorkerId::from_string(format!("w-fc-{idx}")),
                    LeaseId::from_string(format!("l-fc-{idx}")),
                    t_plus(60),
                    t_plus(4),
                )
                .await
            }));
        }
        let mut claimed_healthy = 0usize;
        for handle in handles {
            if let Some(task) = handle.await.context("join")?.context("scan")? {
                assert_eq!(
                    task.id, healthy_id,
                    "only the healthy root should be claimed"
                );
                claimed_healthy += 1;
            }
        }
        assert_eq!(claimed_healthy, 1, "healthy root claimed exactly once");
        let ex_row = store
            .get(&exhausted_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(ex_row.status, TaskStatus::Failed);
        Ok(())
    }

    // ── Child-task waiting states under concurrency ───────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_complete_task_on_batch_drives_parent_to_pending() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-conc-batch").await?;
        let parent_id = parent.id.clone();
        let child_count = 6usize;
        let specs: Vec<ChildSpawnSpec> = (0..child_count).map(|_| ChildSpawnSpec::new(2)).collect();
        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                specs,
                SuspensionPayload {
                    continuation: sample_continuation("t-conc-batch"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        let mut child_owners = Vec::with_capacity(child_count);
        for (idx, child) in children.iter().enumerate() {
            let wid = WorkerId::from_string(format!("w-cb-{idx}"));
            let lid = LeaseId::from_string(format!("l-cb-{idx}"));
            let offset = i64::try_from(idx).context("fits")?;
            store
                .try_acquire_task(
                    &child.id,
                    wid.clone(),
                    lid.clone(),
                    t_plus(60),
                    t_plus(10 + offset),
                )
                .await
                .context("acquire")?
                .context("claimed")?;
            child_owners.push((child.id.clone(), wid, lid));
        }

        let mut handles = Vec::with_capacity(child_count);
        for (idx, (cid, wid, lid)) in child_owners.into_iter().enumerate() {
            let st = Arc::clone(&store);
            let offset = i64::try_from(idx).context("fits")?;
            handles.push(tokio::spawn(async move {
                st.complete_task(&cid, &wid, &lid, t_plus(20 + offset))
                    .await
            }));
        }
        for handle in handles {
            handle.await.context("join")?.context("complete")?;
        }

        let final_parent = store
            .get(&parent_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(final_parent.status, TaskStatus::Pending);
        assert_eq!(final_parent.pending_child_count, 0);
        assert!(matches!(
            final_parent.state,
            TaskState::ReadyToResume { .. }
        ));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_mixed_complete_fail_task_batch_resumes_parent() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let (parent, worker, lease) = running_root_for_spawn(&store, "t-conc-mix").await?;
        let parent_id = parent.id.clone();
        let child_count = 5usize;
        let specs: Vec<ChildSpawnSpec> = (0..child_count).map(|_| ChildSpawnSpec::new(2)).collect();
        let (_, children) = store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                specs,
                SuspensionPayload {
                    continuation: sample_continuation("t-conc-mix"),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn")?;

        let mut child_owners = Vec::with_capacity(child_count);
        for (idx, child) in children.iter().enumerate() {
            let wid = WorkerId::from_string(format!("w-mx-{idx}"));
            let lid = LeaseId::from_string(format!("l-mx-{idx}"));
            let offset = i64::try_from(idx).context("fits")?;
            store
                .try_acquire_task(
                    &child.id,
                    wid.clone(),
                    lid.clone(),
                    t_plus(60),
                    t_plus(10 + offset),
                )
                .await
                .context("acquire")?
                .context("claimed")?;
            child_owners.push((child.id.clone(), wid, lid));
        }

        let mut handles = Vec::with_capacity(child_count);
        for (idx, (cid, wid, lid)) in child_owners.into_iter().enumerate() {
            let st = Arc::clone(&store);
            let offset = i64::try_from(idx).context("fits")?;
            handles.push(tokio::spawn(async move {
                if idx % 2 == 0 {
                    st.complete_task(&cid, &wid, &lid, t_plus(20 + offset))
                        .await
                } else {
                    st.fail_task(
                        &cid,
                        &wid,
                        &lid,
                        format!("child {idx} oops"),
                        t_plus(20 + offset),
                    )
                    .await
                }
            }));
        }
        for handle in handles {
            handle.await.context("join")?.context("terminal")?;
        }

        let final_parent = store
            .get(&parent_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(final_parent.status, TaskStatus::Pending);
        assert_eq!(final_parent.pending_child_count, 0);
        let listed = store.list_children(&parent_id).await.context("list")?;
        assert_eq!(listed.len(), child_count);
        let completed = listed
            .iter()
            .filter(|c| c.status == TaskStatus::Completed)
            .count();
        let failed = listed
            .iter()
            .filter(|c| c.status == TaskStatus::Failed)
            .count();
        assert_eq!(completed, 3); // indices 0, 2, 4
        assert_eq!(failed, 2); // indices 1, 3
        Ok(())
    }

    // ── Recovery edges under concurrency ──────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stale_heartbeat_after_sweep_fails() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-stale-hb"), t_plus(0), 3);
        let task_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let old_w = WorkerId::from_string("w-old");
        let old_l = LeaseId::from_string("l-old");
        store
            .try_acquire_task(&task_id, old_w.clone(), old_l.clone(), t_plus(5), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;
        store
            .release_expired_leases(t_plus(10))
            .await
            .context("sweep")?;
        let err = store
            .heartbeat_task(&task_id, &old_w, &old_l, t_plus(60), None, t_plus(11))
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not running") || msg.contains("invalid transition"),
            "stale heartbeat should be rejected: {msg}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stale_complete_task_after_reacquire_fails_cas() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-stale-ct"), t_plus(0), 5);
        let task_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let old_w = WorkerId::from_string("w-stale-old");
        let old_l = LeaseId::from_string("l-stale-old");
        store
            .try_acquire_task(&task_id, old_w.clone(), old_l.clone(), t_plus(5), t_plus(1))
            .await
            .context("acquire old")?
            .context("claimed old")?;
        store
            .release_expired_leases(t_plus(10))
            .await
            .context("sweep")?;
        store
            .try_acquire_task(
                &task_id,
                WorkerId::from_string("w-stale-new"),
                LeaseId::from_string("l-stale-new"),
                t_plus(60),
                t_plus(11),
            )
            .await
            .context("acquire new")?
            .context("claimed new")?;
        let err = store
            .complete_task(&task_id, &old_w, &old_l, t_plus(12))
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("worker mismatch") || msg.contains("lease mismatch"),
            "stale complete should fail CAS: {msg}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_heartbeat_and_sweep_before_expiry_preserves_lease() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-hb-vs-sweep"), t_plus(0), 3);
        let task_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let wid = WorkerId::from_string("w-hbsweep");
        let lid = LeaseId::from_string("l-hbsweep");
        store
            .try_acquire_task(&task_id, wid.clone(), lid.clone(), t_plus(15), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;

        let st1 = Arc::clone(&store);
        let st2 = Arc::clone(&store);
        let id1 = task_id.clone();
        let hb_w = wid.clone();
        let hb_l = lid.clone();
        let hb_handle = tokio::spawn(async move {
            st1.heartbeat_task(&id1, &hb_w, &hb_l, t_plus(120), None, t_plus(10))
                .await
        });
        let sweep_handle =
            tokio::spawn(async move { st2.release_expired_leases(t_plus(10)).await });
        let _ = hb_handle.await.context("hb join")?;
        let _ = sweep_handle.await.context("sweep join")?;

        let row = store
            .get(&task_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(
            row.worker_id.as_ref().map(WorkerId::as_str),
            Some(wid.as_str())
        );
        assert_eq!(
            row.lease_id.as_ref().map(LeaseId::as_str),
            Some(lid.as_str())
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_sweep_on_budget_exhausted_fails_closed_once() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-conc-exhaust"), t_plus(0), 1);
        let task_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        store
            .try_acquire_task(
                &task_id,
                WorkerId::from_string("w-ce"),
                LeaseId::from_string("l-ce"),
                t_plus(5),
                t_plus(1),
            )
            .await
            .context("acquire")?
            .context("claimed")?;

        let count = 6usize;
        let mut handles: Vec<tokio::task::JoinHandle<Result<Vec<RecoveryRecord>>>> =
            Vec::with_capacity(count);
        for _ in 0..count {
            let st = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                st.release_expired_leases(t_plus(10)).await
            }));
        }
        let mut total_records = 0usize;
        for handle in handles {
            let records = handle.await.context("join")?.context("sweep")?;
            total_records += records.len();
        }
        assert_eq!(total_records, 1, "exactly one sweep should touch the row");
        let row = store
            .get(&task_id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(row.status, TaskStatus::Failed);
        let err = store
            .heartbeat_task(
                &task_id,
                &WorkerId::from_string("w-ce"),
                &LeaseId::from_string("l-ce"),
                t_plus(60),
                None,
                t_plus(11),
            )
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not running") || msg.contains("invalid transition"),
            "stale heartbeat after fail-closed should fail: {msg}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn complete_task_vs_cancel_tree_converges_to_single_terminal() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root = AgentTask::new_root_turn(thread("t-ct-vs-cancel"), t_plus(0), 3);
        let root_id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;
        let wid = WorkerId::from_string("w-ctc");
        let lid = LeaseId::from_string("l-ctc");
        store
            .try_acquire_task(&root_id, wid.clone(), lid.clone(), t_plus(60), t_plus(1))
            .await
            .context("acquire")?
            .context("claimed")?;

        let st1 = Arc::clone(&store);
        let st2 = Arc::clone(&store);
        let rid1 = root_id.clone();
        let rid2 = root_id.clone();
        let cw = wid.clone();
        let cl = lid.clone();
        let complete_handle =
            tokio::spawn(async move { st1.complete_task(&rid1, &cw, &cl, t_plus(2)).await });
        let cancel_handle = tokio::spawn(async move { st2.cancel_tree(&rid2, t_plus(2)).await });
        let complete_res = complete_handle.await.context("complete join")?;
        let cancel_res = cancel_handle.await.context("cancel join")?;

        let row = store
            .get(&root_id)
            .await
            .context("get")?
            .context("exists")?;
        assert!(
            row.status.is_terminal(),
            "row must be terminal: {:?}",
            row.status
        );
        match (complete_res.is_ok(), cancel_res.is_ok()) {
            (true, true) => {
                assert!(row.status == TaskStatus::Completed || row.status == TaskStatus::Cancelled);
            }
            (true, false) => assert_eq!(row.status, TaskStatus::Completed),
            (false, true) => assert_eq!(row.status, TaskStatus::Cancelled),
            (false, false) => panic!("both complete and cancel failed"),
        }
        Ok(())
    }

    // ── Scale / soak ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn many_workers_many_roots_drain_runnable_pool_without_duplication() -> Result<()> {
        let store = Arc::new(InMemoryAgentTaskStore::new());
        let root_count = 20usize;
        let worker_count = 10usize;

        for idx in 0..root_count {
            let root = AgentTask::new_root_turn(
                thread(&format!("t-soak-{idx}")),
                t_plus(i64::try_from(idx).unwrap()),
                5,
            );
            store.submit_root_turn(root).await.context("submit")?;
        }

        let claimed: Arc<tokio::sync::Mutex<Vec<(AgentTaskId, WorkerId, LeaseId)>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::with_capacity(worker_count);
        for idx in 0..worker_count {
            let st = Arc::clone(&store);
            let cl = Arc::clone(&claimed);
            handles.push(tokio::spawn(async move {
                loop {
                    let wid = WorkerId::from_string(format!("w-soak-{idx}"));
                    let lid = LeaseId::new();
                    match st
                        .acquire_next_runnable(wid.clone(), lid.clone(), t_plus(120), t_plus(50))
                        .await
                    {
                        Ok(Some(task)) => {
                            cl.lock().await.push((task.id.clone(), wid, lid));
                        }
                        Ok(None) => break,
                        Err(err) => return Err(err),
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.await.context("join")?.context("acquire loop")?;
        }

        let claimed_vec = claimed.lock().await.clone();
        assert_eq!(claimed_vec.len(), root_count, "all roots claimed");
        let unique: std::collections::HashSet<_> =
            claimed_vec.iter().map(|(id, _, _)| id.clone()).collect();
        assert_eq!(unique.len(), root_count, "no duplicates");

        let mut handles = Vec::with_capacity(root_count);
        for (tid, wid, lid) in &claimed_vec {
            let st = Arc::clone(&store);
            let tid = tid.clone();
            let wid = wid.clone();
            let lid = lid.clone();
            handles.push(tokio::spawn(async move {
                st.complete_task(&tid, &wid, &lid, t_plus(100)).await
            }));
        }
        for handle in handles {
            handle.await.context("join")?.context("complete")?;
        }

        for idx in 0..root_count {
            let all = store
                .list_by_thread(&thread(&format!("t-soak-{idx}")))
                .await
                .context("list")?;
            assert_eq!(all.len(), 1);
            assert_eq!(all[0].status, TaskStatus::Completed);
            assert_eq!(all[0].attempt, 1, "each root touched exactly once");
        }
        Ok(())
    }

    // ── R2 steering wake: enqueue_steering_resume / repark_after_steering ──

    /// A non-empty steering payload (a wake requires non-empty content).
    fn steering_note() -> Vec<agent_sdk_foundation::llm::ContentBlock> {
        vec![agent_sdk_foundation::llm::ContentBlock::Text {
            text: "status?".into(),
        }]
    }

    /// Park a fresh parent on two tool children (both Pending).
    async fn parked_with_two_children(
        store: &InMemoryAgentTaskStore,
        name: &str,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let (parent, worker, lease) = running_root_for_spawn(store, name).await?;
        let (parked, children) = store
            .spawn_tool_children(
                &parent.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2), ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation: sample_continuation(name),
                    suspended_messages: Vec::new(),
                },
                None,
                t_plus(2),
            )
            .await
            .context("spawn tool children")?;
        Ok((parked, children[0].clone(), children[1].clone()))
    }

    async fn complete_child(
        store: &InMemoryAgentTaskStore,
        child_id: &AgentTaskId,
        output: &str,
        secs: i64,
    ) -> Result<()> {
        let worker = WorkerId::from_string("w-child");
        let lease = LeaseId::from_string("l-child");
        store
            .try_acquire_task(
                child_id,
                worker.clone(),
                lease.clone(),
                t_plus(secs + 5),
                t_plus(secs),
            )
            .await?
            .context("acquire child")?;
        let result = agent_sdk_foundation::ToolResult {
            success: true,
            output: output.to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        };
        store
            .complete_task_with_result(
                child_id,
                &worker,
                &lease,
                serde_json::to_value(result).context("serialize child result")?,
                t_plus(secs + 1),
            )
            .await
            .context("complete child")?;
        Ok(())
    }

    #[tokio::test]
    async fn enqueue_steering_resume_declines_when_parent_not_parked() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        // A running root is not WaitingOnChildren — the wake must decline.
        let (parent, _worker, _lease) = running_root_for_spawn(&store, "t-steer-decline").await?;
        let out = store
            .enqueue_steering_resume(
                &parent.id,
                vec![agent_sdk_foundation::llm::ContentBlock::Text { text: "hi".into() }],
                t_plus(5),
            )
            .await?;
        assert!(
            out.is_none(),
            "wake must decline a parent that is not parked"
        );
        // The row is untouched.
        let after = store.get(&parent.id).await?.context("parent")?;
        assert_eq!(after.status, TaskStatus::Running);
        Ok(())
    }

    #[tokio::test]
    async fn repark_after_steering_rejects_bad_cas_and_non_steering_state() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parked, c0, c1) = parked_with_two_children(&store, "t-steer-cas").await?;
        store
            .enqueue_steering_resume(&parked.id, steering_note(), t_plus(10))
            .await?
            .context("woken")?;
        let worker = WorkerId::from_string("w-steer");
        let lease = LeaseId::from_string("l-steer");
        store
            .try_acquire_task(
                &parked.id,
                worker.clone(),
                lease.clone(),
                t_plus(900),
                t_plus(11),
            )
            .await?
            .context("acquire woken parent")?;

        // Wrong lease → CAS reject; the parent is left running.
        let bad = store
            .repark_after_steering(
                &parked.id,
                &worker,
                &LeaseId::from_string("l-wrong"),
                SuspensionPayload {
                    continuation: sample_continuation("t-steer-cas"),
                    suspended_messages: Vec::new(),
                },
                vec![c0.id.clone(), c1.id.clone()],
                t_plus(12),
            )
            .await;
        assert!(
            format!("{:#}", bad.unwrap_err()).contains("lease mismatch"),
            "wrong lease must be rejected",
        );

        // A genuinely running root (state None) is not a steering resume.
        let (running_root, w2, l2) = running_root_for_spawn(&store, "t-not-steer").await?;
        let err = store
            .repark_after_steering(
                &running_root.id,
                &w2,
                &l2,
                SuspensionPayload {
                    continuation: sample_continuation("t-not-steer"),
                    suspended_messages: Vec::new(),
                },
                Vec::new(),
                t_plus(13),
            )
            .await;
        assert!(
            format!("{:#}", err.unwrap_err()).contains("not a steering resume"),
            "non-steering running row must be rejected",
        );
        Ok(())
    }

    #[tokio::test]
    async fn repark_after_steering_all_children_terminal_goes_ready_to_resume() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (parked, c0, c1) = parked_with_two_children(&store, "t-steer-ready").await?;
        assert_eq!(store.list_children(&parked.id).await?.len(), 2);

        // Wake; then BOTH children finish while the parent sits in
        // SteeringResume (Pending). The completions must not flip the
        // parent — the race guard keeps the wake authoritative.
        store
            .enqueue_steering_resume(&parked.id, steering_note(), t_plus(10))
            .await?
            .context("woken")?;
        complete_child(&store, &c0.id, "r0", 11).await?;
        complete_child(&store, &c1.id, "r1", 13).await?;
        let mid = store.get(&parked.id).await?.context("mid")?;
        assert_eq!(mid.status, TaskStatus::Pending);
        assert!(
            mid.state.is_steering_resume(),
            "child completions under SteeringResume must not re-flip the parent",
        );

        // Acquire + re-park: every re-attach child is already terminal,
        // so live == 0 and the parent lands in ReadyToResume for the
        // ordinary fan-in. Re-attach in reversed order to prove the
        // dense spawn_index re-mapping.
        let worker = WorkerId::from_string("w-steer");
        let lease = LeaseId::from_string("l-steer");
        store
            .try_acquire_task(
                &parked.id,
                worker.clone(),
                lease.clone(),
                t_plus(900),
                t_plus(20),
            )
            .await?
            .context("acquire woken parent")?;
        let reparked = store
            .repark_after_steering(
                &parked.id,
                &worker,
                &lease,
                SuspensionPayload {
                    continuation: sample_continuation("t-steer-ready"),
                    suspended_messages: Vec::new(),
                },
                vec![c1.id.clone(), c0.id.clone()],
                t_plus(21),
            )
            .await
            .context("repark")?;
        assert_eq!(reparked.status, TaskStatus::Pending);
        match &reparked.state {
            TaskState::ReadyToResume { child_ids, .. } => {
                assert_eq!(child_ids, &vec![c1.id.clone(), c0.id.clone()]);
            }
            other => panic!("expected ReadyToResume, got {other:?}"),
        }
        // Re-attach order drives the dense spawn_index (c1 → 0, c0 → 1).
        assert_eq!(store.get(&c1.id).await?.context("c1")?.spawn_index, Some(0));
        assert_eq!(store.get(&c0.id).await?.context("c0")?.spawn_index, Some(1));
        // Budget-neutral: the wake / re-park spawned nothing.
        assert_eq!(store.list_children(&parked.id).await?.len(), 2);
        Ok(())
    }
}
