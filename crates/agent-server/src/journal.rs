//! Durable task journal for agent-server.
//!
//! The journal owns the durable-friendly **`agent_tasks`** rows that
//! represent every unit of work the server orchestrates:
//!
//! - **root turns** — one per `run_turn` invocation on a thread
//! - **tool-runtime child tasks** — one per tool execution spawned by a root
//! - **subagents** — reserved for Phase 3
//!
//! Phase 2.1 (ENG-7915) ships the schema layer only:
//!
//! - [`AgentTask`], [`TaskKind`], [`TaskStatus`], identity types, and
//!   structural / state-machine invariants
//! - Pure state-transition helpers every later phase can build on
//!
//! Later phases layer acquisition, queueing, retries, and tool execution
//! on top without re-modelling the state machine.
//!
//! # Required store indexes
//!
//! The [`AgentTaskStore`] trait (Phase 2.1) is deliberately thin, but every
//! production implementation is expected to expose the indexes listed
//! below. The in-memory reference implementation enforces them directly so
//! Phase 2.2+ can rely on their semantics.
//!
//! 1. Primary key on `id`
//! 2. `(thread_id)` — for listing a thread's history and for 2.2 FIFO
//!    queueing
//! 3. `(parent_id)` — for 2.6 child-task resolution
//! 4. `(status)` — for 2.3 runnable scanning
//! 5. Partial unique on `(thread_id)` where
//!    `kind = root_turn AND status NOT IN (completed, failed, cancelled)`
//!    — at most one active root per thread
//! 6. `(worker_id, lease_expires_at)` — for 2.3 lease-expiry sweeping (the
//!    fields live on the row in 2.1; the index itself lands in 2.3)
//!
//! # What is **not** here yet
//!
//! | Scope | Phase |
//! |-------|-------|
//! | Same-thread FIFO queue promotion | 2.2 |
//! | Lease acquisition / expiry sweep API | 2.3 |
//! | Confirmation / resume wiring | 2.4 |
//! | Retry budget + recovery workers | 2.5 |
//! | Tool-runtime child orchestration | 2.6 |

pub mod store;
pub mod task;

pub use store::{AgentTaskStore, InMemoryAgentTaskStore};
pub use task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskSchemaError, TaskStatus, WorkerId};
