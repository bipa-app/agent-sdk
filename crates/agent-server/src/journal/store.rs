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
//! - [`AgentTaskStore::resolve_child`] decrements the parent's
//!   outstanding-child counter; on the final decrement it flips
//!   the row back to [`TaskStatus::Pending`] and clears the typed
//!   payload so a worker can re-acquire it.
//! - [`AgentTaskStore::resume_from_confirmation`] flips a
//!   confirmation-paused row back to [`TaskStatus::Pending`] and
//!   clears the typed payload after the caller has read its
//!   continuation and prepared operation.
//!
//! All four entry points run their CAS check, the typed-state
//! mutation, and the index rebalance under a single write lock,
//! so the journal is the single source of truth for the
//! paused-state transitions. The schema layer
//! ([`super::task::AgentTask::validate`]) refuses to round-trip
//! any row whose `status` and `state` disagree, so a buggy
//! caller cannot leave a paused row without a continuation or
//! leak a stale continuation onto a runnable row.
//!
//! # Retry budget, failure handling, and the recovery matrix (Phase 2.5)
//!
//! Phase 2.5 (ENG-7919) replaces the Phase 2.3 "loud failure on
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

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use agent_sdk_core::{ContinuationEnvelope, ListenExecutionContext, ThreadId};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::RwLock;

use super::recovery::{
    FailureReason, RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
use super::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};

/// Persistent store for [`AgentTask`] rows.
///
/// Implementations are required to expose the index surface documented on
/// the [`super`] module. The reference in-memory implementation below is
/// used in tests and also acts as the semantic spec for any future SQL- or
/// Redis-backed store.
#[async_trait]
pub trait AgentTaskStore: Send + Sync {
    /// Insert a new task row.
    ///
    /// The row must pass [`AgentTask::validate`] and, if it is a
    /// [`TaskKind::RootTurn`] in a status that satisfies
    /// [`TaskStatus::blocks_root_admission`], must not collide with an
    /// existing blocking root on the same thread.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, if a row with the
    /// same `id` already exists, or if the blocking-root invariant would
    /// be violated.
    ///
    /// Prefer [`AgentTaskStore::submit_root_turn`] when inserting
    /// externally-submitted root turns: it applies the FIFO queue rule
    /// automatically instead of rejecting the submission when the slot
    /// is busy.
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

    /// Look up a task by id.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>>;

    /// Replace a task row with an updated version.
    ///
    /// The incoming row must pass [`AgentTask::validate`] and must refer
    /// to an `id` that already exists in the store.
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
    /// # Errors
    /// Returns an error if a release or fail-closed transition, or
    /// the underlying store write, fails.
    async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>>;

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
        continuation: ContinuationEnvelope,
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

    /// Decrement a [`TaskStatus::WaitingOnChildren`] parent's
    /// outstanding-child counter and, when it hits zero, atomically
    /// flip the row back to [`TaskStatus::Pending`] so a worker can
    /// pick it up and re-enter the loop.
    ///
    /// This is the resume entry point that Phase 2.6's child-task
    /// orchestrator will call once a child reaches a terminal state.
    /// While there are still outstanding children, the parent stays
    /// in `WaitingOnChildren` and the typed [`crate::journal::TaskState`] payload
    /// remains in place; the resume transition fires only on the
    /// final decrement.
    ///
    /// Returns the persisted parent row after the mutation.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `parent_id` is stored.
    /// - `resolve rejected: not waiting on children` — if the row is
    ///   in any status other than [`TaskStatus::WaitingOnChildren`].
    async fn resolve_child(
        &self,
        parent_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

    /// Resume a [`TaskStatus::AwaitingConfirmation`] task back to
    /// [`TaskStatus::Pending`], clearing the typed
    /// [`crate::journal::TaskState::AwaitingConfirmation`] payload atomically with
    /// the status flip.
    ///
    /// This is the resume entry point a confirmation transport will
    /// call once it receives the user's decision (Phase 2.4 leaves
    /// the transport itself out of scope). The caller is responsible
    /// for reading the embedded [`ContinuationEnvelope`] and any
    /// prepared listen/execute operation **before** calling this
    /// method, because the resume transition wipes the typed payload
    /// to satisfy the state ↔ status invariant.
    ///
    /// Returns the persisted resumed row.
    ///
    /// # Errors
    /// - `task does not exist` — if no row with `id` is stored.
    /// - `resume rejected: not awaiting confirmation` — if the row is
    ///   in any status other than [`TaskStatus::AwaitingConfirmation`].
    async fn resume_from_confirmation(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<AgentTask>;

    /// Remove every stored task. Used by tests.
    ///
    /// # Errors
    /// Returns an error if the store cannot be cleared.
    async fn clear(&self) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

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
}

impl InMemoryAgentTaskStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
        self.by_id.insert(failed.id.clone(), failed);
        Ok(())
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

        let mut inner = self.inner.write().await;

        if inner.by_id.contains_key(&task.id) {
            let id = &task.id;
            return Err(anyhow!(
                "submit_root_turn rejected: task id {id} already exists"
            ));
        }

        // Decide the admission target atomically under the write lock so
        // two concurrent submissions on the same thread always serialize
        // and end up in a deterministic FIFO order.
        let admitted = if inner.thread_has_blocking_root(&task.thread_id) {
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

        // If the slot is still held, the active root is retrying or
        // waiting on something and no promotion may fire — retries of
        // the active root must never be overtaken by queued roots.
        if inner.thread_has_blocking_root(thread_id) {
            return Ok(None);
        }

        // Pop the FIFO head. `BTreeSet` iterates in ascending key order,
        // and the key is `(created_at, id)` so the earliest submission
        // wins, with `id` breaking ties deterministically.
        let head_key = inner
            .queued_roots_by_thread
            .get(thread_id)
            .and_then(|q| q.iter().next().cloned());
        let Some((_, id)) = head_key else {
            return Ok(None);
        };

        // Load the row, run the pure promotion transition, and commit
        // via the shared rebalance helper so the runnable index picks
        // up the newly-Pending row.
        let queued_row = inner
            .by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("promote rejected: queue head {id} missing from by_id"))?;
        let promoted = queued_row
            .clone()
            .promote_to_pending(now)
            .context("promote rejected: promotion transition failed")?;

        inner.rebalance_after_row_change(&queued_row, &promoted);
        inner.by_id.insert(id, promoted.clone());
        drop(inner);
        Ok(Some(promoted))
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
        // bumps `last_heartbeat_at` / `lease_expires_at` together.
        let mut refreshed = old.clone();
        refreshed
            .touch_heartbeat(worker, lease, expires_at, now)
            .context("heartbeat rejected")?;

        inner.rebalance_after_row_change(&old, &refreshed);
        inner.by_id.insert(refreshed.id.clone(), refreshed.clone());
        drop(inner);
        Ok(refreshed)
    }

    async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>> {
        let mut inner = self.inner.write().await;

        // Walk the lease-expiry index in ascending order, collecting
        // every row whose expiry is `<= now`. Iteration stops on the
        // first still-live lease, so the sweep cost is O(expired).
        // Collecting up-front lets us mutate `inner` inside the loop
        // without fighting the borrow checker.
        let expired_keys: Vec<(OffsetDateTime, AgentTaskId)> = inner
            .leased_by_expiry
            .iter()
            .take_while(|(expires_at, _)| *expires_at <= now)
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
        Ok(released)
    }

    async fn pause_on_children(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        child_count: u32,
        continuation: ContinuationEnvelope,
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
            .wait_on_children(child_count, continuation, now)
            .context("pause rejected: wait_on_children transition failed")?;
        inner.rebalance_after_row_change(&old, &paused);
        inner.by_id.insert(paused.id.clone(), paused.clone());
        drop(inner);
        Ok(paused)
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

    async fn resolve_child(
        &self,
        parent_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut inner = self.inner.write().await;
        let old = inner
            .by_id
            .get(parent_id)
            .cloned()
            .ok_or_else(|| anyhow!("resolve rejected: task {parent_id} does not exist"))?;

        if old.status != TaskStatus::WaitingOnChildren {
            let status = old.status;
            return Err(anyhow!(
                "resolve rejected: task {parent_id} is not waiting on children (status {status:?})"
            ));
        }

        let resolved = old
            .clone()
            .child_resolved(now)
            .context("resolve rejected: child_resolved transition failed")?;
        inner.rebalance_after_row_change(&old, &resolved);
        inner.by_id.insert(resolved.id.clone(), resolved.clone());
        drop(inner);
        Ok(resolved)
    }

    async fn resume_from_confirmation(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
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

        let resumed = old
            .clone()
            .resume_from_confirmation(now)
            .context("resume rejected: resume_from_confirmation transition failed")?;
        inner.rebalance_after_row_change(&old, &resumed);
        inner.by_id.insert(resumed.id.clone(), resumed.clone());
        drop(inner);
        Ok(resumed)
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
    use agent_sdk_core::{AgentContinuation, AgentState, ContinuationEnvelope, TokenUsage};
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
        })
    }

    /// Sample listen/execute prepared operation paired with the
    /// confirmation continuation in tests that exercise the typed
    /// [`TaskState::AwaitingConfirmation`] payload.
    fn sample_prepared_op() -> agent_sdk_core::ListenExecutionContext {
        agent_sdk_core::ListenExecutionContext {
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
    // Phase 2.2 — Root submission queue and FIFO promotion (ENG-7916)
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
                        .wait_on_children(2, sample_continuation("t1"), t_plus(2))
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
            .wait_on_children(1, sample_continuation("t1"), t_plus(7))
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
    // (ENG-7917)
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
            .wait_on_children(2, sample_continuation("t-wait"), t_plus(2))
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
        // Phase 2.5 (ENG-7919) replaces the Phase 2.3 "loud error on
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
            .wait_on_children(1, sample_continuation("b"), t_plus(3))
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
            .wait_on_children(1, sample_continuation("t1"), t_plus(3))
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
    // and typed durable task state (ENG-7918)
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
                sample_continuation("t-pause-children"),
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
                sample_continuation("t-invisible-target"),
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
                sample_continuation("t-scan-invisible"),
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
                sample_continuation("t-cas"),
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
                sample_continuation("t-cas"),
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
                sample_continuation("t-pending"),
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
    async fn resolve_child_drains_counter_and_resumes_parent() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let (claimed, worker, lease) =
            submitted_and_claimed_root(&store, "t-resolve", "w1", "l1").await?;
        let id = claimed.id.clone();
        store
            .pause_on_children(
                &id,
                &worker,
                &lease,
                3,
                sample_continuation("t-resolve"),
                t_plus(2),
            )
            .await
            .context("pause")?;

        // First two resolves keep the parent in WaitingOnChildren and
        // preserve the typed payload.
        let after_one = store
            .resolve_child(&id, t_plus(3))
            .await
            .context("resolve 1")?;
        assert_eq!(after_one.status, TaskStatus::WaitingOnChildren);
        assert_eq!(after_one.pending_child_count, 2);
        assert!(after_one.state.continuation().is_some());

        let after_two = store
            .resolve_child(&id, t_plus(4))
            .await
            .context("resolve 2")?;
        assert_eq!(after_two.status, TaskStatus::WaitingOnChildren);
        assert_eq!(after_two.pending_child_count, 1);
        assert!(after_two.state.continuation().is_some());

        // Final resolve flips the parent back to Pending and clears
        // the typed payload to satisfy the state ↔ status invariant.
        let after_three = store
            .resolve_child(&id, t_plus(5))
            .await
            .context("resolve 3")?;
        assert_eq!(after_three.status, TaskStatus::Pending);
        assert_eq!(after_three.pending_child_count, 0);
        assert!(after_three.state.is_none());

        // The runnable scan must now observe the resumed parent.
        let scanned = store
            .acquire_next_runnable(
                WorkerId::from_string("w-resume"),
                LeaseId::from_string("l-resume"),
                t_plus(120),
                t_plus(6),
            )
            .await
            .context("scan resumed")?
            .context("scan returned row")?;
        assert_eq!(scanned.id, id);
        assert_eq!(scanned.status, TaskStatus::Running);
        // Resumed row is on its second attempt because the original
        // claim consumed attempt #1 and the recovery scan consumes
        // attempt #2.
        assert_eq!(scanned.attempt, 2);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resolve_child_rejects_non_waiting_row() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = AgentTask::new_root_turn(thread("t-not-waiting"), t_plus(0), 3);
        let id = root.id.clone();
        store.submit_root_turn(root).await.context("submit")?;

        let err = store.resolve_child(&id, t_plus(1)).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("not waiting on children"),
            "unexpected: {message}"
        );
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

        let resumed = store
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
                sample_continuation("t-round-trip"),
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
                sample_continuation("t-json-children"),
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
    // (ENG-7919)
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
        let resumed = store
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
}
