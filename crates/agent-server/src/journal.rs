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
//! # What is **not** here yet
//!
//! | Scope | Phase |
//! |-------|-------|
//! | Confirmation / resume wiring | 2.4 |
//! | Retry budget + recovery workers | 2.5 |
//! | Tool-runtime child orchestration | 2.6 |

pub mod store;
pub mod task;

pub use store::{AgentTaskStore, InMemoryAgentTaskStore};
pub use task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskSchemaError, TaskStatus, WorkerId};
