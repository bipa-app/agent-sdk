//! Durable task journal for agent-server.
//!
//! The journal owns the durable-friendly **`agent_tasks`** rows that
//! represent every unit of work the server orchestrates:
//!
//! - **root turns** — one per `run_turn` invocation on a thread
//! - **tool-runtime child tasks** — one per tool execution spawned by a root
//! - **subagents** — reserved for Phase 3
//!
//! Phase 2.1 shipped the schema layer:
//!
//! - [`AgentTask`], [`TaskKind`], [`TaskStatus`], identity types, and
//!   structural / state-machine invariants
//! - Pure state-transition helpers every later phase can build on
//!
//! Phase 2.2 layered same-thread FIFO root queueing on top:
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
//! Phase 2.3 layers guarded acquisition, lease ownership,
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
//! Phase 2.4 layers typed durable pause-state on top:
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
//!   [`AgentTaskStore::pause_on_confirmation`], and
//!   [`AgentTaskStore::resume_from_confirmation`] are the journal-guarded
//!   pause / resume entry points. They run the row's status CAS, the
//!   typed-state mutation, and the index rebalance under one write
//!   lock so the journal is the single source of truth for the
//!   paused-state transitions. (Phase 2.4 also shipped a one-shot
//!   `resolve_child(parent_id)` decrement helper — it was retired in
//!   Phase 2.6 when the parent counter became journal-derived, see
//!   [`AgentTaskStore::complete_task`] / [`AgentTaskStore::fail_task`]
//!   for the replacement.)
//! - The state ↔ status invariant is enforced by
//!   [`AgentTask::validate`] (rejecting any row whose typed payload
//!   disagrees with its status) and is round-tripped through JSON for
//!   every variant by the schema regression suite.
//!
//! Phase 2.5 layers retry budget, failure handling, and the
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
//! Phase 2.6 layers tool-runtime child orchestration,
//! cancellation cascade, and fully journal-driven parent resume
//! triggers on top:
//!
//! - [`AgentTaskStore::spawn_tool_children`] is the single atomic
//!   entry point for creating tool-runtime child tasks. A successful
//!   call CAS-checks the parent's lease, persists one fresh
//!   [`TaskKind::ToolRuntime`] row per [`task::ChildSpawnSpec`],
//!   transitions the parent to [`TaskStatus::WaitingOnChildren`] with
//!   a typed continuation, and drops the parent's lease — all under
//!   the same write lock so no partial batch can ever be observed.
//! - [`AgentTaskStore::complete_task`] and
//!   [`AgentTaskStore::fail_task`] drive a running child to its
//!   terminal state (`Completed` / `Failed`) and, under the same
//!   write lock, recompute the parent's `pending_child_count`
//!   authoritatively from the `by_parent` live-children index via
//!   [`AgentTask::recompute_pending_children`]. The parent becomes
//!   runnable the moment its last live child reaches any terminal
//!   state — there is no channel, no in-memory queue, and no
//!   caller-maintained counter involved.
//! - [`AgentTaskStore::cancel_tree`] cascades cancellation through
//!   the `by_parent` index. It walks `root_id` and every descendant
//!   in BFS order under the store write lock, runs
//!   [`AgentTask::cancel`] on each non-terminal row, and drops any
//!   live leases along the way. Stale workers that held those
//!   leases fail their next Phase 2.3 `(worker_id, lease_id)` CAS
//!   and cannot mutate the row on the way out.
//! - The journal-driven resume contract means a crashed worker can
//!   restart mid-batch entirely from the durable row set: the
//!   parent's typed [`TaskState::WaitingOnChildren`] carries the
//!   continuation, `list_children(parent_id)` carries the aggregated
//!   outcomes, and `recompute_pending_children` re-derives the
//!   counter from scratch on every terminal child transition.
//!
//! # Phase 2.7 — CAS contract, lifecycle model, and worker call sequences
//!
//! Phase 2.7 finalizes the journal API surface for worker consumption.
//! Every mutation is now CAS-guarded — `insert` / `update` / `clear`
//! are reserved for store rehydration and test scaffolding only.
//!
//! ## Task lifecycle state machine
//!
//! ```text
//! ┌──────────┐  submit_root_turn   ┌──────────┐
//! │ (new)    ├────────────────────►│ Pending  │◄──────────────────────┐
//! └──────────┘  (slot free)        └────┬─────┘                      │
//!       │                               │ try_acquire / scan         │
//!       │  submit_root_turn             ▼                            │
//!       │  (slot busy)           ┌──────────┐  heartbeat_task        │
//!       ▼                        │ Running  ├───────► (extends lease) │
//! ┌──────────┐  promote          └─┬──┬──┬──┘                        │
//! │ Queued   ├──────────────────►  │  │  │                           │
//! └──────────┘  (slot free)        │  │  │   pause_on_children /     │
//!                                  │  │  │   spawn_tool_children     │
//!                                  │  │  ▼                           │
//!                                  │  │ ┌────────────────────┐       │
//!                                  │  │ │ WaitingOnChildren  ├───────┘
//!                                  │  │ └────────────────────┘
//!                                  │  │   (last child terminal →
//!                                  │  │    recompute → Pending)
//!                                  │  │
//!                                  │  │  pause_on_confirmation
//!                                  │  ▼
//!                                  │ ┌─────────────────────────┐
//!                                  │ │ AwaitingConfirmation    ├─────┘
//!                                  │ └─────────────────────────┘
//!                                  │   (resume_from_confirmation
//!                                  │    → Pending)
//!                                  │
//!           complete_task ─────────┤
//!           fail_task ─────────────┤
//!           cancel_tree ───────────┤
//!                                  ▼
//!                          ┌──────────────────┐
//!                          │ Completed / Failed│
//!                          │ / Cancelled       │
//!                          └──────────────────┘
//! ```
//!
//! ## CAS guards summary
//!
//! | Entry point | Status guard | Worker/Lease CAS | Recovery matrix |
//! |-------------|-------------|------------------|-----------------|
//! | `submit_root_turn` | kind=root, status=Pending | — | — |
//! | `promote_next_queued_root` | slot free, queue non-empty | — | — |
//! | `try_acquire_task` | Pending | — | budget check |
//! | `acquire_next_runnable` | Pending (scan) | — | budget check, skip-on-exhaust |
//! | `heartbeat_task` | Running | ✓ worker + lease | — |
//! | `pause_on_children` | Running | ✓ worker + lease | — |
//! | `pause_on_confirmation` | Running | ✓ worker + lease | — |
//! | `spawn_tool_children` | Running, non-leaf | ✓ worker + lease | — |
//! | `complete_task` | Running | ✓ worker + lease | parent recompute |
//! | `fail_task` | Running | ✓ worker + lease | parent recompute |
//! | `resume_from_confirmation` | `AwaitingConfirmation` | — | — |
//! | `cancel_tree` | exists | — | subtree walk |
//! | `release_expired_leases` | Running, expired | — | budget + prepared-op check |
//!
//! ## Happy-path call sequences
//!
//! **Root turn:**
//! ```text
//! submit_root_turn → try_acquire_task → heartbeat_task* →
//!   (complete_task | fail_task | pause_on_children → ... → complete_task)
//! promote_next_queued_root  (fires after terminal root frees slot)
//! ```
//!
//! **Tool-runtime child:**
//! ```text
//! spawn_tool_children  (under parent's lease — parent → WaitingOnChildren)
//!   → acquire_next_runnable → heartbeat_task* → complete_task | fail_task
//!   (last child terminal → parent recompute → parent Pending)
//! ```
//!
//! **Recovery path:**
//! ```text
//! release_expired_leases → classify_recovery →
//!   Requeue (budget ok)  →  row returns to Pending
//!   FailClosed (exhausted / unsafe prepared op)  →  row → Failed
//! Stale worker's next heartbeat / complete_task / fail_task →
//!   CAS mismatch → clean rejection
//! ```
//!
//! # Phase 3.1 — Threads projection and aggregate ownership
//!
//! Phase 3.1 adds the **threads projection** — a durable materialized
//! aggregate view of committed conversation-level counters and status.
//! The key design property is **single ownership**: thread aggregates
//! (`committed_turns`, `total_usage`) are updated exclusively through
//! the [`thread_store::ThreadStore::commit_turn`] entry point, which
//! delegates to [`thread::Thread::apply_committed_turn`] under the
//! store's write lock. There is no raw `update()` that touches
//! counters, so no worker can mutate thread-level aggregates outside
//! the completed-turn commit path.
//!
//! - [`thread::Thread`] is the durable row: identity, status
//!   (`Active` / `Completed`), committed turn count, cumulative
//!   token usage, and timestamps.
//! - [`thread_store::ThreadStore`] is the narrow trait surface:
//!   `get_or_create`, `get`, `commit_turn`, `mark_completed`,
//!   `list`.
//! - [`thread_store::InMemoryThreadStore`] is the reference in-memory
//!   implementation, following the same `Arc<RwLock<Inner>>` pattern
//!   as [`store::InMemoryAgentTaskStore`].
//!
//! # Phase 3.2 — Message projection and transactional `replace_history`
//!
//! Phase 3.2 adds the **message projection** — a durable ordered
//! record of committed conversation messages per thread. The key
//! design properties:
//!
//! 1. **No mid-turn writes** — message projection writes happen
//!    exclusively at turn completion, so a crash can never leave the
//!    projection ahead of the latest completed checkpoint.
//! 2. **Atomic `replace_history`** — context compaction swaps the
//!    entire history under the store's write lock, so readers never
//!    see a partially replaced history.
//! 3. **Versioned rows** — every mutation bumps a monotonic `version`
//!    counter, enabling future optimistic concurrency control.
//!
//! - [`message::MessageProjection`] is the durable row: thread id,
//!   ordered message history, version, and timestamps.
//! - [`message_store::MessageProjectionStore`] is the narrow trait
//!   surface: `get_or_create`, `get`, `get_history`,
//!   `commit_messages`, `replace_history`.
//! - [`message_store::InMemoryMessageProjectionStore`] is the
//!   reference in-memory implementation, following the same
//!   `Arc<RwLock<Inner>>` pattern as [`thread_store::InMemoryThreadStore`].
//!
//! # Phase 3.3 — Turn-attempt schema and append-only audit repository
//!
//! Phase 3.3 adds the **turn-attempt audit table** — an append-only
//! execution log for LLM request/response cycles. Every attempt
//! records full model/provider provenance so the audit trail survives
//! retries, failovers, and provider rotations. The key design
//! properties:
//!
//! 1. **Append-only apart from close** — rows are inserted via
//!    [`turn_attempt_store::TurnAttemptStore::open_attempt`] and the
//!    only mutation is
//!    [`turn_attempt_store::TurnAttemptStore::close_attempt`]. There
//!    is no `update()` or `delete()`.
//! 2. **No continuation state** — the table records what happened
//!    during an LLM call, not what should happen next. Scheduler and
//!    continuation state stay on [`task::AgentTask`] /
//!    [`task_state::TaskState`].
//! 3. **Full provenance** — every row carries provider, requested
//!    model, response model, response id, and request/response blobs.
//!
//! - [`turn_attempt::TurnAttempt`] is the durable row with open/close
//!   lifecycle and structural validation.
//! - [`turn_attempt_store::TurnAttemptStore`] is the narrow trait
//!   surface: `open_attempt`, `close_attempt`, `get`, `list_by_task`.
//! - [`turn_attempt_store::InMemoryTurnAttemptStore`] is the
//!   reference in-memory implementation.
//!
//! # Phase 3.4 — Completed-turn checkpoints and atomic commit path
//!
//! Phase 3.4 adds the **completed-turn checkpoint** — an immutable
//! snapshot of conversation state at the instant a turn commits
//! successfully — and the **atomic commit path** that ties all
//! projections together.
//!
//! 1. **Thread-scoped uniqueness** — exactly one checkpoint exists
//!    per `(thread_id, turn_number)`. The store enforces this with a
//!    partial-unique index and rejects duplicates.
//! 2. **Full snapshot for v1 recovery** — each checkpoint carries the
//!    complete message history and an opaque agent-state blob, so
//!    recovery can restore a thread to any committed turn without
//!    replaying from epoch.
//! 3. **Atomic commit path** — [`commit::commit_completed_turn`] is
//!    the single entry point that advances the turn-attempt audit,
//!    thread aggregate, message projection, and checkpoint table
//!    together. Failed or cancelled turns that never call this
//!    function do not create checkpoints.
//!
//! - [`checkpoint::Checkpoint`] is the immutable snapshot row.
//! - [`checkpoint_store::CheckpointStore`] is the narrow trait
//!   surface: `commit_checkpoint`, `get`, `get_by_turn`,
//!   `list_by_thread`.
//! - [`checkpoint_store::InMemoryCheckpointStore`] is the reference
//!   in-memory implementation.
//! - [`commit::commit_completed_turn`] is the atomic commit
//!   orchestrator.
//! - [`commit::CompletedTurnCommit`] / [`commit::CommitOutcome`] are
//!   the input/output types for the commit path.
//!
//! # Phase 3.5 — Thread-scoped checkpoint recovery and rebuild API
//!
//! Phase 3.5 adds the **recovery loader** — given a thread, it loads
//! the latest completed checkpoint and rebuilds the next-turn view.
//! The key design properties:
//!
//! 1. **Checkpoint-only recovery** — the recovered view comes
//!    exclusively from the latest completed checkpoint. Failed or
//!    in-progress attempts that never called
//!    [`commit::commit_completed_turn`] do not pollute the state.
//! 2. **Sequential root-task continuity** — checkpoints are
//!    task-agnostic at recovery time. A new root task on the same
//!    thread picks up from the last committed turn regardless of
//!    which task produced it.
//! 3. **Consistency invariant** — if the thread has committed turns,
//!    the latest checkpoint's `turn_number` must equal
//!    `thread.committed_turns`. A mismatch is a journal-level data
//!    corruption error.
//!
//! - [`thread_recover::recover_thread`] is the single entry point
//!   for thread recovery.
//! - [`thread_recover::ThreadRecoveryView`] is the output type
//!   containing messages, agent-state snapshot, and next turn number.
//! - [`checkpoint_store::CheckpointStore::get_latest_by_thread`] is
//!   the new store method that returns the highest-turn checkpoint
//!   for a thread.
//!
//! # Phase 4.2 — `ExecutionContextFactory`, checkpoint-seeding, and staged stores
//!
//! Phase 4.2 adds the **staged execution model** — the root worker
//! reconstructs trusted execution context from durable task, thread,
//! and checkpoint state, and keeps all message/state mutations buffered
//! in memory until commit time.
//!
//! - [`staged::StagedMessageStore`] and [`staged::StagedStateStore`]
//!   implement the SDK's [`agent_sdk_tools::stores::MessageStore`] and
//!   [`agent_sdk_tools::stores::StateStore`] traits while keeping all
//!   writes in-memory. Seeded from the latest completed checkpoint (or
//!   empty for fresh threads).
//! - [`execution_context::RootWorkerInputs`] bundles the Phase 4.1
//!   bootstrap context, recovery view, and staged stores into a
//!   single "factory input" struct.
//! - [`execution_context::build_root_worker_inputs`] is the primary
//!   entry point: it recovers thread state, seeds staged stores, and
//!   returns everything the worker needs to begin a turn.
//!
//! # What is **not** here yet
//!
//! | Scope | Phase |
//! |-------|-------|
//! | Text-only turn execution | 4.3 |
//! | Tool-batch suspension | 4.4+ |
//! | Event replay | future |
//! | Subagent runtime | future |
//! | Confirmation transport APIs | post-2.4 |
//!
//! # Phase 5.2 — Durable execution intent and fail-closed mutation guard
//!
//! Phase 5.2 adds the **execution intent** layer — a durable
//! pre-execution record that side-effecting and resumable tools must
//! persist before their executor callback runs. This closes the gap
//! between "child task exists" and "tool was actually invoked",
//! giving the system a reliable signal for retry, replay, and
//! manual intervention decisions.
//!
//! - [`execution_intent::OperationId`] is a stable identity that
//!   combines the server-generated child task id with the LLM-assigned
//!   tool call id, surviving retries and restarts.
//! - [`execution_intent::ToolEffectClass`] classifies every tool call
//!   as `ReplaySafe`, `SideEffecting`, or `Resumable` to drive
//!   durability and retry rules.
//! - [`execution_intent::ExecutionIntent`] is the durable record
//!   persisted before execution, updated to `Completed`/`Failed`
//!   after.
//! - [`execution_intent::ExecutionIntentStore`] is the trait for
//!   durable intent persistence, with
//!   [`execution_intent::InMemoryExecutionIntentStore`] as the
//!   reference in-memory implementation.
//! - [`execution_intent::guarded_tool_execution`] wraps
//!   [`super::worker::tool_task::execute_tool_task`] with the
//!   fail-closed guard: if intent persistence fails for a
//!   side-effecting tool, the executor callback is never invoked.
//! - [`execution_intent::check_retry_safety`] inspects a prior
//!   intent record and returns a [`execution_intent::RetryDecision`]
//!   so retry logic can distinguish replay-safe operations from
//!   unsafe duplicate mutation.
//!
//! # Phase 6.1 — Durable event committer and thread-scoped sequencing
//!
//! Phase 6.1 adds the **event repository** — the authoritative commit
//! surface for durable events.  Every event committed through this
//! layer receives server-owned metadata (UUID v7 `event_id`, monotonic
//! per-thread `sequence`, commit-time `timestamp`), replacing
//! SDK-local [`SequenceCounter`](agent_sdk_foundation::events::SequenceCounter)
//! semantics for the authoritative server path.
//!
//! Key design properties:
//!
//! 1. **Single allocation authority** — the event repository is the
//!    only place that assigns `event_id`, `sequence`, and `timestamp`
//!    for committed events.  Workers submit raw [`AgentEvent`](agent_sdk_foundation::events::AgentEvent)s;
//!    the repository wraps them atomically.
//! 2. **Thread-scoped sequencing** — each thread gets an independent
//!    monotonic counter starting at 0.  Sequences are contiguous
//!    within batches and across single/batch operations on the same
//!    thread.
//! 3. **Append-only with `(thread_id, sequence)` uniqueness** —
//!    enforced by allocating and persisting under the same write lock
//!    (or transaction in a database backend).
//! 4. **Batched append** — workers can commit an entire turn's event
//!    stream atomically, reserving a contiguous sequence range.
//!
//! - [`committed_event::CommittedEvent`] is the server-owned event
//!   record with envelope conversion methods.
//! - [`event_repository::EventRepository`] is the commit surface
//!   trait: `commit_event`, `commit_event_batch`, `next_sequence`,
//!   `get_events`.
//! - [`event_repository::InMemoryEventRepository`] is the reference
//!   in-memory implementation.
//!
//! # Phase 6.3 — Replay API and race-free replay-to-live handoff
//!
//! Phase 6.3 adds the **replay surface** — the public API for
//! reconnecting clients to resume from their last-seen sequence and
//! seamlessly transition to live committed event delivery.
//!
//! Key design properties:
//!
//! 1. **Race-free handoff** — the subscribe-before-read protocol
//!    ensures no event committed after the subscribe call is missed,
//!    even under concurrent production.  Overlap between the durable
//!    replay and live tail is deduplicated by tracking the last
//!    yielded sequence.
//! 2. **No gaps, no duplicates** — the `(subscribe → watermark →
//!    replay → live)` ordering closes every race window at the
//!    watermark boundary.
//! 3. **Unpublished events never exposed** — clients only see events
//!    that have been durably committed to the [`event_repository::EventRepository`].
//!    The live tail is fed by [`event_notifier::EventNotifier::notify`],
//!    which callers invoke only after durable commit.
//! 4. **Thread-scoped, stateless** — each stream is bound to a single
//!    thread. There is no hidden per-connection cursor state on the
//!    server; the client's `after_sequence` is the only resume token.
//!
//! - [`event_notifier::EventNotifier`] is the same-process live tail
//!   notification hub backed by per-thread broadcast channels.
//! - [`event_stream::stream_events`] is the public entry point that
//!   returns an [`event_stream::EventStream`] with the race-free
//!   handoff baked in.
//! - [`event_stream::StreamEvent`] is the stream item type:
//!   `Event(CommittedEvent)` for normal delivery, `Lagged { skipped }`
//!   when the subscriber falls behind the broadcast buffer.
//!
//! # Phase 6.4 — Live tail hub, lag detection, and bounded-wait disconnect
//!
//! Phase 6.4 adds the **live tail hub** — a same-process fanout surface
//! with per-subscriber bounded buffers, lag detection, grace-period
//! handling, and replay-required disconnect semantics.
//!
//! Key design properties:
//!
//! 1. **Per-subscriber isolation** — each subscriber gets an independent
//!    bounded `mpsc` channel.  A slow subscriber never affects other
//!    subscribers on the same thread.
//! 2. **Workers never block** — [`live_tail::LiveTailHub::publish`]
//!    uses `try_send` and never awaits.  Slow or dead subscribers are
//!    skipped, not waited on.
//! 3. **Lag detection with bounded-wait grace period** — when a
//!    subscriber's buffer fills up, the hub enters a grace period.
//!    During the grace period, the hub stops delivering events to the
//!    subscriber (they are durable and recoverable via replay).  If
//!    the subscriber remains behind after the grace period, the hub
//!    disconnects with replay-required semantics.
//! 4. **Replay-required disconnect contract** — the disconnect signal
//!    includes the last successfully delivered durable sequence.  The
//!    subscriber reconnects via [`event_stream::stream_events`] with
//!    `after_sequence` set to this value.
//! 5. **Only committed envelopes** — the hub delivers events only
//!    after they have been durably committed to the
//!    [`event_repository::EventRepository`].
//!
//! - [`live_tail::LiveTailHub`] is the fanout hub with per-subscriber
//!   bounded buffers and lag management.
//! - [`live_tail::LiveTailReceiver`] is the subscriber's receive handle.
//! - [`live_tail::LiveTailEvent`] is the stream item type:
//!   `Event(CommittedEvent)` for normal delivery,
//!   `ReplayRequired { last_delivered_sequence }` on lag disconnect.
//! - [`live_tail::LiveTailConfig`] controls buffer capacity and grace
//!   period duration.

pub mod broker;
pub mod checkpoint;
pub mod checkpoint_store;
pub mod commit;
pub mod committed_event;
pub mod completed_turn_transaction;
/// Reusable journal/store conformance battery.
///
/// Compiled both for this crate's own tests and for downstream
/// consumers that enable the `test-support` feature, so the SQLite and
/// Postgres backends in `agent-service-host` can run the same battery.
#[cfg(any(test, feature = "test-support"))]
pub mod conformance;
#[cfg(test)]
mod durability_dst_test;
pub mod event_notifier;
pub mod event_outbox_transaction;
pub mod event_repository;
pub mod event_stream;
pub mod execution_context;
pub mod execution_intent;
pub mod fork_transaction;
pub mod idempotency;
pub mod live_tail;
#[cfg(test)]
mod live_tail_test;
pub mod message;
pub mod message_store;
pub mod outbox;
pub mod outbox_message;
#[cfg(test)]
mod persistence_regression;
pub mod recovery;
pub mod redaction;
pub mod relay;
#[cfg(test)]
mod replay_determinism_test;
#[cfg(test)]
mod replay_test;
pub mod retention;
pub mod retention_janitor;
pub mod staged;
pub mod store;
pub mod task;
pub mod task_state;
pub mod task_wakeup;
pub mod thread;
pub mod thread_events_watch;
#[cfg(test)]
mod thread_events_watch_regression;
pub mod thread_recover;
pub mod thread_store;
pub mod tool_audit;
pub mod turn_attempt;
pub mod turn_attempt_store;

pub use broker::{BrokerAdapter, InMemoryBrokerAdapter};
pub use checkpoint::{Checkpoint, CheckpointId, CheckpointSchemaError, NewCheckpointParams};
pub use checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
pub use commit::{CommitOutcome, CompletedTurnCommit, commit_completed_turn};
pub use committed_event::CommittedEvent;
pub use completed_turn_transaction::AtomicCompletedTurnCommitter;
pub use event_notifier::{EventNotifier, EventReceiver};
pub use event_outbox_transaction::{
    AtomicEventOutboxCommitter, EventOutboxCommit, EventOutboxCommitOutcome,
};
pub use event_repository::{EventRepository, InMemoryEventRepository};
pub use event_stream::{EventStream, StreamEvent, stream_events};
pub use execution_context::{RootWorkerInputs, build_root_worker_inputs};
pub use execution_intent::{
    ExecutionIntent, ExecutionIntentStore, InMemoryExecutionIntentStore, IntentStatus, OperationId,
    RetryDecision, ToolEffectClass, check_retry_safety, classify_tool_effect,
    guarded_tool_execution,
};
pub use live_tail::{LiveTailConfig, LiveTailEvent, LiveTailHub, LiveTailReceiver, SubscriberId};
pub use message::{MessageProjection, MessageProjectionError};
pub use message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
pub use outbox::{
    InMemoryOutboxStore, NewOutboxRow, OutboxRow, OutboxRowId, OutboxStatus, OutboxStore,
};
pub use outbox_message::{
    OutboxMessage, OutboxMessageKind, TaskWakeupPayload, ThreadEventsAvailablePayload,
};
pub use recovery::{
    FailureReason, RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
pub use redaction::{
    REDACTED_MARKER, RedactionLevel, RedactionPolicy, redact_error, redact_string, redact_value,
};
pub use relay::{
    BrokerPublisher, InMemoryTaskWakeupEmitter, OutboxRelayWorker, Publisher, RelayTick,
    RelayWorker, RetryBackoff, TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS, TaskWakeupEmitter,
    TaskWakeupTrigger,
};
pub use retention::{InMemoryRetentionStore, RetentionCursor, RetentionStore};
pub use retention_janitor::{
    JanitorCycleReport, RetentionJanitorDeps, RetentionPolicy, run_janitor_cycle,
};
pub use staged::{StagedMessageStore, StagedStateStore, StagedStores};
pub use store::{
    AgentTaskStore, CancelTreeOutcome, CancellationMarkerSink, InMemoryAgentTaskStore,
    SubagentInvocationSpawn,
};
pub use task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskKind, TaskSchemaError,
    TaskStatus, WorkerId,
};
pub use task_state::{SubagentInvocationState, TaskState};
pub use task_wakeup::{
    CapturingTaskWakeupHandler, FallbackWakeupSweep, JournalTaskWakeupHandler, TaskWakeupHandler,
    TaskWakeupOutcome, WakeupSignal, dispatch_payload,
};
pub use thread::{Thread, ThreadSchemaError, ThreadStatus};
#[cfg(any(test, feature = "test-support"))]
pub use thread_events_watch::CapturingThreadEventsWatchHandler;
pub use thread_events_watch::{
    NotifierThreadEventsWatchHandler, ThreadEventsWatchHandler, ThreadEventsWatchOutcome,
    dispatch_thread_events_payload,
};
pub use thread_recover::{ThreadRecoveryView, recover_thread};
pub use thread_store::{InMemoryThreadStore, ThreadStore};
pub use tool_audit::{
    InMemoryToolAuditEventStore, ToolAuditEvent, ToolAuditEventId, ToolAuditEventKind,
    ToolAuditEventParams, ToolAuditEventStore,
};
pub use turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId, TurnAttemptOutcome,
    TurnAttemptSchemaError,
};
pub use turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
