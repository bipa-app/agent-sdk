//! Durable task journal for agent-server.
//!
//! The journal owns the durable-friendly **`agent_tasks`** rows that
//! represent every unit of work the server orchestrates:
//!
//! - **root turns** — one per `run_turn` invocation on a thread
//! - **tool-runtime child tasks** — one per tool execution spawned by a root
//! - **subagents** — reserved for Phase 3
//!
//! Phase 2.1 (ENG-7915) shipped the schema layer:
//!
//! - [`AgentTask`], [`TaskKind`], [`TaskStatus`], identity types, and
//!   structural / state-machine invariants
//! - Pure state-transition helpers every later phase can build on
//!
//! Phase 2.2 (ENG-7916) layered same-thread FIFO root queueing on top:
//!
//! - [`AgentTaskStore::submit_root_turn`] durably admits a new root as
//!   `Pending` when the thread's active-root slot is free, or converts
//!   it to `Queued` when another root is still blocking.
//! - [`AgentTaskStore::list_queued_roots`] exposes the per-thread FIFO
//!   queue in deterministic order (`created_at` ascending, tiebroken by
//!   `id`).
//! - [`AgentTaskStore::promote_next_queued_root`] advances the FIFO head
//!   into `Pending` when the slot is free — a no-op otherwise, so
//!   retries of the active root never let queued roots overtake it.
//! - The partial-unique index now keys on
//!   [`TaskStatus::blocks_root_admission`] instead of `is_active`, so
//!   `Queued` rows do not occupy the slot and may coexist with the
//!   blocking root.
//!
//! Phase 2.3 (ENG-7917) layers guarded acquisition, lease ownership,
//! and heartbeat plumbing on top of the schema:
//!
//! - [`AgentTaskStore::try_acquire_task`] performs a targeted CAS
//!   claim on a single task id and returns `None` if the row is not
//!   runnable. Two workers racing on the same id are serialised by
//!   the store's write lock so exactly one wins.
//! - [`AgentTaskStore::acquire_next_runnable`] scans the global
//!   runnable index (oldest `created_at` first) and atomically claims
//!   the head. Root turns and tool-runtime children share the same
//!   lease path, which is the "one acquisition and lease model that
//!   prevents double ownership across task kinds" the issue requires.
//! - [`AgentTaskStore::heartbeat_task`] refreshes the lease under a
//!   `(worker_id, lease_id)` CAS guard so a stale worker that lost
//!   its lease cannot extend a row it no longer owns.
//! - [`AgentTaskStore::release_expired_leases`] walks the global
//!   lease-expiry index and returns every leased row whose
//!   `lease_expires_at <= now` to `Pending`, leaving still-live
//!   leases untouched.
//! - Waiting (`WaitingOnChildren`, `AwaitingConfirmation`), terminal,
//!   and `Queued` rows are deliberately invisible to both the
//!   targeted and scanning acquire paths.
//!
//! Later phases layer retries, recovery matrices, and tool execution
//! on top without re-modelling the state machine.
//!
//! # Required store indexes
//!
//! The [`AgentTaskStore`] trait is deliberately thin, but every
//! production implementation is expected to expose the indexes listed
//! below. The in-memory reference implementation enforces them directly
//! so Phase 2.3+ can rely on their semantics.
//!
//! 1. Primary key on `id`
//! 2. `(thread_id)` — for listing a thread's history and for 2.2 FIFO
//!    queueing
//! 3. `(parent_id)` — for 2.6 child-task resolution
//! 4. `(status)` — for 2.3 runnable scanning
//! 5. Partial unique on `(thread_id)` where
//!    `kind = root_turn AND status.blocks_root_admission()` — at most
//!    one blocking root per thread (the "active-root slot"). `Queued`
//!    roots are not part of this index (see 6 below).
//! 6. Per-thread FIFO index on `(thread_id, created_at, id)` where
//!    `kind = root_turn AND status = queued` — powers
//!    [`AgentTaskStore::list_queued_roots`] and
//!    [`AgentTaskStore::promote_next_queued_root`].
//! 7. Global runnable FIFO index on `(created_at, id)` where
//!    `status = pending` — powers
//!    [`AgentTaskStore::acquire_next_runnable`] across both root
//!    turns and tool-runtime children.
//! 8. Global lease-expiry index on `(lease_expires_at, id)` where
//!    `status = running` — powers
//!    [`AgentTaskStore::release_expired_leases`].
//!
//! Phase 2.4 (ENG-7918) layers typed durable pause-state on top:
//!
//! - The previously-untyped `state` field on [`AgentTask`] is now a
//!   strongly typed [`TaskState`] enum, with one variant per pause
//!   status. Active and terminal rows carry [`TaskState::None`] and
//!   pay zero per-row metadata.
//! - [`AgentTask::wait_on_children`] and [`AgentTask::await_confirmation`]
//!   take their typed payload at the call site so the row's status and
//!   state can never drift apart. The pure transitions drop the lease,
//!   so a paused parent never looks runnable to
//!   [`AgentTaskStore::acquire_next_runnable`] or
//!   [`AgentTaskStore::try_acquire_task`].
//! - [`AgentTaskStore::pause_on_children`],
//!   [`AgentTaskStore::pause_on_confirmation`],
//!   [`AgentTaskStore::resolve_child`], and
//!   [`AgentTaskStore::resume_from_confirmation`] are the journal-guarded
//!   pause / resume entry points. They run the row's status CAS, the
//!   typed-state mutation, and the index rebalance under one write
//!   lock so the journal is the single source of truth for the
//!   paused-state transitions.
//! - The state ↔ status invariant is enforced by
//!   [`AgentTask::validate`] (rejecting any row whose typed payload
//!   disagrees with its status) and is round-tripped through JSON for
//!   every variant by the schema regression suite.
//!
//! Phase 2.5 (ENG-7919) layers retry budget, failure handling, and the
//! stale-task recovery matrix on top:
//!
//! - [`recovery::classify_recovery`] is a pure entry point that every
//!   Phase 2.5 call site shares. It inspects the row's kind, status,
//!   retry budget, and prepared-operation bit and returns a
//!   [`recovery::RecoveryAction`] — `NoAction`, `Requeue`, or
//!   `FailClosed(reason)` — that callers must honor atomically under
//!   the store's write lock.
//! - [`AgentTaskStore::try_acquire_task`] and
//!   [`AgentTaskStore::acquire_next_runnable`] consult the matrix
//!   before leasing a row. Retry-budget exhaustion is now a **terminal
//!   failure**: the row transitions to [`TaskStatus::Failed`] with a
//!   canonical `last_error`, the call returns `Ok(None)`, and the
//!   worker pool is never poisoned by an exhausted head sitting on
//!   the runnable index forever.
//! - [`AgentTaskStore::release_expired_leases`] returns a
//!   <code>Vec<[recovery::RecoveryRecord]></code> so the caller can
//!   distinguish rows that were requeued for another attempt from
//!   rows that were failed closed because their budget was used up.
//! - A tool-runtime task that carries a staged listen/execute
//!   prepared operation on its [`TaskState::AwaitingConfirmation`]
//!   payload is **always** failed closed on recovery via
//!   [`recovery::FailureReason::UnsafePreparedOperationRecovery`] so
//!   the journal never blindly re-runs a tool with an in-flight
//!   external side-effect.
//! - Duplicate ownership across a requeue + retry is still guarded
//!   by the Phase 2.3 `(worker_id, lease_id)` CAS — Phase 2.5 adds
//!   explicit regression coverage in the store test suite so an old
//!   worker can never heartbeat a row that the sweep has released
//!   back to `Pending` or failed closed.
//!
//! # What is **not** here yet
//!
//! | Scope | Phase |
//! |-------|-------|
//! | Tool-runtime child orchestration | 2.6 |
//! | Confirmation transport APIs | post-2.4 |

pub mod recovery;
pub mod store;
pub mod task;
pub mod task_state;

pub use recovery::{
    FailureReason, RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
pub use store::{AgentTaskStore, InMemoryAgentTaskStore};
pub use task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskSchemaError, TaskStatus, WorkerId};
pub use task_state::TaskState;
