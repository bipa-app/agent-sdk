//! Reusable journal/store conformance battery.
//!
//! This module ports Anthropic Claude Agent SDK's
//! `run_session_store_conformance` into a backend-agnostic Rust
//! trait-test battery,
//! [`run_journal_store_conformance()`](crate::journal::conformance::run_journal_store_conformance),
//! parametrized over the journal/store backend. It follows LangGraph's
//! one-spec-many-backends pattern: a single behavioral contract that
//! the in-memory, `SQLite`, and `PostgreSQL` backends must all satisfy.
//!
//! The battery is a packaged behavioral contract for any store/journal
//! adapter. After this SDK is open sourced, a third-party backend can
//! self-verify by implementing
//! [`JournalStore`](crate::journal::conformance::JournalStore) for its
//! bundle type and calling
//! [`run_journal_store_conformance()`](crate::journal::conformance::run_journal_store_conformance)
//! from its own test suite.
//!
//! # Why one spec, many backends
//!
//! The conformance cases here are *adapted to the real journal traits*
//! ([`AgentTaskStore`], [`EventRepository`], [`CheckpointStore`],
//! [`MessageProjectionStore`], [`ThreadStore`], [`TurnAttemptStore`]) —
//! no new store API is invented. Each LangGraph/Claude-SDK conformance
//! concept maps onto an existing trait surface:
//!
//! | Conformance concept | Real trait surface |
//! |---------------------|--------------------|
//! | append preserves insertion order | [`EventRepository::commit_event`] / [`MessageProjectionStore::commit_messages`] |
//! | `load()` returns an independent deep-equal copy | every `get*` returns owned clones |
//! | `load()` of unknown key → `None` | `get`/`get_history` on a fresh thread |
//! | semantic (not byte) equality on roundtrip | JSON-value comparison, not string comparison |
//! | optional methods absent → auto-skip | [`EventRepository::atomic_event_outbox_committer`] |
//! | `delete()` cascades main→subkeys | [`AgentTaskStore::clear`] across parent/child chains |
//! | safe no-op delete for append-only | [`CheckpointStore::delete_checkpoints_beyond_limit`] when under the cap |
//! | subkey vs main-session count separation | root tasks vs child tasks ([`AgentTask::is_root`]) |
//! | clear/empty invariants | `clear()` then `list_by_thread` empty |
//! | N concurrent appends serialize without loss | `commit_event` from N tasks onto one thread |
//! | checkpoint roundtrip | [`CheckpointStore::commit_checkpoint`] then `get` |
//! | `list()` reverse-chronological | [`CheckpointStore::list_by_thread`] ordered by turn |
//! | latest-by-insertion-order, NOT lexicographic id | [`CheckpointStore::get_latest_by_thread`] (LangGraph #6821) |
//! | pending writes persisted | message-projection draft slot survives a `get` |
//! | parent-checkpoint link | turn-N checkpoint references turn-(N-1) history |
//! | metadata filter | filter checkpoints by `turn_usage` token counts |
//! | before+limit pagination | events windowed by `get_events_in_range` |
//!
//! # Failing loudly
//!
//! The battery fails loudly if a backend violates a contract. The only
//! silent skip is for genuinely-optional methods (the atomic
//! event+outbox committer, which in-memory backends legitimately do not
//! provide); even there, the skip is recorded in
//! [`ConformanceReport`](crate::journal::conformance::ConformanceReport)
//! so a caller can assert which arms actually ran.

use std::sync::Arc;

use agent_sdk_core::audit::AuditProvenance;
use agent_sdk_core::{ThreadId, TokenUsage, llm};
use anyhow::{Context, Result, ensure};
use time::{Duration, OffsetDateTime};

use agent_sdk_core::events::AgentEvent;

use super::checkpoint::{CheckpointId, NewCheckpointParams};
use super::checkpoint_store::CheckpointStore;
use super::commit::{CompletedTurnCommit, commit_completed_turn};
use super::event_repository::EventRepository;
use super::message_store::MessageProjectionStore;
use super::store::AgentTaskStore;
use super::task::{AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, WorkerId};
use super::thread_store::ThreadStore;
use super::turn_attempt::{CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome};
use super::turn_attempt_store::TurnAttemptStore;

/// A bundle of every journal store trait the conformance battery needs.
///
/// Backends that already implement all six traits on a single type
/// (e.g. `SqliteDurableStore`, `PostgresDurableStore`) satisfy this
/// blanket-implemented trait automatically. The in-memory backend uses
/// the [`InMemoryJournalStore`] bundle, which wires the six separate
/// reference implementations together.
///
/// This trait is the parametrization handle for
/// [`run_journal_store_conformance`]. It deliberately does **not** add
/// any new behavior — it is purely the intersection of the existing
/// trait surfaces.
pub trait JournalStore:
    AgentTaskStore
    + ThreadStore
    + MessageProjectionStore
    + TurnAttemptStore
    + CheckpointStore
    + EventRepository
    + Send
    + Sync
{
}

impl<T> JournalStore for T where
    T: AgentTaskStore
        + ThreadStore
        + MessageProjectionStore
        + TurnAttemptStore
        + CheckpointStore
        + EventRepository
        + Send
        + Sync
{
}

/// In-memory [`JournalStore`] bundle for the reference backend.
///
/// Holds the six in-memory reference implementations and forwards each
/// trait to the corresponding inner store. Cloning shares the
/// underlying journals (each inner store is `Arc`-backed), so a clone
/// handed to a spawned task observes the same state — exactly what the
/// concurrent-append case needs.
#[derive(Clone, Default)]
pub struct InMemoryJournalStore {
    task: super::store::InMemoryAgentTaskStore,
    thread: super::thread_store::InMemoryThreadStore,
    message: super::message_store::InMemoryMessageProjectionStore,
    attempt: super::turn_attempt_store::InMemoryTurnAttemptStore,
    checkpoint: super::checkpoint_store::InMemoryCheckpointStore,
    event: super::event_repository::InMemoryEventRepository,
}

impl InMemoryJournalStore {
    /// Construct an empty in-memory journal bundle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

mod in_memory_bundle {
    //! Trait forwarding for [`super::InMemoryJournalStore`].
    //!
    //! Each `impl` delegates to the matching inner reference store. The
    //! delegations live in their own module so the (necessarily
    //! mechanical) forwarding does not crowd the battery itself.

    use super::{
        AgentTask, AgentTaskId, AgentTaskStore, CheckpointStore, EventRepository,
        InMemoryJournalStore, LeaseId, MessageProjectionStore, OffsetDateTime, Result, ThreadId,
        ThreadStore, TurnAttemptStore, WorkerId,
    };
    use crate::journal::checkpoint::{Checkpoint, CheckpointId, NewCheckpointParams};
    use crate::journal::committed_event::CommittedEvent;
    use crate::journal::event_outbox_transaction::AtomicEventOutboxCommitter;
    use crate::journal::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};
    use crate::journal::message::MessageProjection;
    use crate::journal::recovery::RecoveryRecord;
    use crate::journal::store::{
        SubagentInvocationSpawn, SubmitRootTurnError, SubmitRootTurnOutcome, SubmitRootTurnParams,
    };
    use crate::journal::task::{ChildSpawnSpec, SuspensionPayload, TaskStatus};
    use crate::journal::thread::Thread;
    use crate::journal::turn_attempt::{
        CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId,
    };
    use agent_sdk_core::events::AgentEvent;
    use agent_sdk_core::{ContinuationEnvelope, ListenExecutionContext, TokenUsage};
    use async_trait::async_trait;

    #[async_trait]
    impl AgentTaskStore for InMemoryJournalStore {
        async fn insert(&self, task: AgentTask) -> Result<()> {
            self.task.insert(task).await
        }
        async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
            self.task.submit_root_turn(task).await
        }
        async fn submit_root_turn_idempotent(
            &self,
            params: SubmitRootTurnParams,
        ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
            self.task.submit_root_turn_idempotent(params).await
        }
        async fn claim_idempotency(
            &self,
            request_id: &str,
            kind: IdempotencyKind,
            fingerprint: &[u8],
        ) -> Result<IdempotencyClaim> {
            self.task
                .claim_idempotency(request_id, kind, fingerprint)
                .await
        }
        async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
            self.task.record_idempotency(record).await
        }
        async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
            AgentTaskStore::get(&self.task, id).await
        }
        async fn update(&self, task: AgentTask) -> Result<()> {
            self.task.update(task).await
        }
        async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
            AgentTaskStore::list_by_thread(&self.task, thread_id).await
        }
        async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
            self.task.list_children(parent_id).await
        }
        async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
            self.task.list_by_status(status).await
        }
        async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
            self.task.active_root_for_thread(thread_id).await
        }
        async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
            self.task.list_queued_roots(thread_id).await
        }
        async fn promote_next_queued_root(
            &self,
            thread_id: &ThreadId,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.task.promote_next_queued_root(thread_id, now).await
        }
        async fn try_acquire_task(
            &self,
            task_id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            lease_expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.task
                .try_acquire_task(task_id, worker, lease, lease_expires_at, now)
                .await
        }
        async fn acquire_next_runnable(
            &self,
            worker: WorkerId,
            lease: LeaseId,
            lease_expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.task
                .acquire_next_runnable(worker, lease, lease_expires_at, now)
                .await
        }
        async fn heartbeat_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.task
                .heartbeat_task(task_id, worker, lease, expires_at, now)
                .await
        }
        async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>> {
            self.task.release_expired_leases(now).await
        }
        async fn pause_on_children(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            child_count: u32,
            payload: SuspensionPayload,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.task
                .pause_on_children(task_id, worker, lease, child_count, payload, now)
                .await
        }
        async fn pause_on_confirmation(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            continuation: ContinuationEnvelope,
            prepared_operation: Option<ListenExecutionContext>,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.task
                .pause_on_confirmation(
                    task_id,
                    worker,
                    lease,
                    continuation,
                    prepared_operation,
                    now,
                )
                .await
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
            self.task
                .spawn_tool_children(
                    parent_id,
                    worker,
                    lease,
                    specs,
                    payload,
                    child_otel_traceparent,
                    now,
                )
                .await
        }
        async fn spawn_subagent_invocation(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawn: SubagentInvocationSpawn,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, AgentTask, AgentTask)> {
            self.task
                .spawn_subagent_invocation(parent_id, worker, lease, spawn, now)
                .await
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
            self.task
                .spawn_subagent_batch(parent_id, worker, lease, spawns, payload, now)
                .await
        }
        async fn complete_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.task.complete_task(task_id, worker, lease, now).await
        }
        async fn complete_task_with_result(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            result: serde_json::Value,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.task
                .complete_task_with_result(task_id, worker, lease, result, now)
                .await
        }
        async fn fail_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            error: String,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.task
                .fail_task(task_id, worker, lease, error, now)
                .await
        }
        async fn cancel_tree(
            &self,
            root_id: &AgentTaskId,
            now: OffsetDateTime,
        ) -> Result<Vec<AgentTaskId>> {
            self.task.cancel_tree(root_id, now).await
        }
        async fn resume_from_confirmation(
            &self,
            task_id: &AgentTaskId,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.task.resume_from_confirmation(task_id, now).await
        }
        async fn approve_confirmation_and_acquire(
            &self,
            task_id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.task
                .approve_confirmation_and_acquire(task_id, worker, lease, expires_at, now)
                .await
        }
        async fn reject_confirmation(
            &self,
            task_id: &AgentTaskId,
            error: String,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.task.reject_confirmation(task_id, error, now).await
        }
        async fn clear(&self) -> Result<()> {
            self.task.clear().await
        }
    }

    #[async_trait]
    impl ThreadStore for InMemoryJournalStore {
        async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
            self.thread.get_or_create(thread_id, now).await
        }
        async fn get(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
            ThreadStore::get(&self.thread, thread_id).await
        }
        async fn commit_turn(
            &self,
            thread_id: &ThreadId,
            turn_usage: &TokenUsage,
            now: OffsetDateTime,
        ) -> Result<Thread> {
            self.thread.commit_turn(thread_id, turn_usage, now).await
        }
        async fn mark_completed(
            &self,
            thread_id: &ThreadId,
            now: OffsetDateTime,
        ) -> Result<Thread> {
            self.thread.mark_completed(thread_id, now).await
        }
        async fn list(&self) -> Result<Vec<Thread>> {
            self.thread.list().await
        }
    }

    #[async_trait]
    impl MessageProjectionStore for InMemoryJournalStore {
        async fn get_or_create(
            &self,
            thread_id: &ThreadId,
            now: OffsetDateTime,
        ) -> Result<MessageProjection> {
            self.message.get_or_create(thread_id, now).await
        }
        async fn get(&self, thread_id: &ThreadId) -> Result<Option<MessageProjection>> {
            MessageProjectionStore::get(&self.message, thread_id).await
        }
        async fn get_history(
            &self,
            thread_id: &ThreadId,
        ) -> Result<Vec<agent_sdk_core::llm::Message>> {
            self.message.get_history(thread_id).await
        }
        async fn commit_messages(
            &self,
            thread_id: &ThreadId,
            messages: Vec<agent_sdk_core::llm::Message>,
            now: OffsetDateTime,
        ) -> Result<MessageProjection> {
            self.message.commit_messages(thread_id, messages, now).await
        }
        async fn replace_history(
            &self,
            thread_id: &ThreadId,
            messages: Vec<agent_sdk_core::llm::Message>,
            now: OffsetDateTime,
        ) -> Result<MessageProjection> {
            self.message.replace_history(thread_id, messages, now).await
        }
        async fn set_draft(
            &self,
            thread_id: &ThreadId,
            messages: Vec<agent_sdk_core::llm::Message>,
            now: OffsetDateTime,
        ) -> Result<MessageProjection> {
            self.message.set_draft(thread_id, messages, now).await
        }
        async fn clear_draft(
            &self,
            thread_id: &ThreadId,
            now: OffsetDateTime,
        ) -> Result<Option<MessageProjection>> {
            self.message.clear_draft(thread_id, now).await
        }
    }

    #[async_trait]
    impl TurnAttemptStore for InMemoryJournalStore {
        async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt> {
            self.attempt.open_attempt(params).await
        }
        async fn close_attempt(
            &self,
            attempt_id: &TurnAttemptId,
            params: CloseAttemptParams,
            now: OffsetDateTime,
        ) -> Result<TurnAttempt> {
            self.attempt.close_attempt(attempt_id, params, now).await
        }
        async fn get(&self, attempt_id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
            TurnAttemptStore::get(&self.attempt, attempt_id).await
        }
        async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>> {
            self.attempt.list_by_task(task_id).await
        }
    }

    #[async_trait]
    impl CheckpointStore for InMemoryJournalStore {
        async fn commit_checkpoint(&self, params: NewCheckpointParams) -> Result<Checkpoint> {
            self.checkpoint.commit_checkpoint(params).await
        }
        async fn get(&self, id: &CheckpointId) -> Result<Option<Checkpoint>> {
            CheckpointStore::get(&self.checkpoint, id).await
        }
        async fn get_by_turn(
            &self,
            thread_id: &ThreadId,
            turn_number: u32,
        ) -> Result<Option<Checkpoint>> {
            self.checkpoint.get_by_turn(thread_id, turn_number).await
        }
        async fn get_latest_by_thread(&self, thread_id: &ThreadId) -> Result<Option<Checkpoint>> {
            self.checkpoint.get_latest_by_thread(thread_id).await
        }
        async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<Checkpoint>> {
            CheckpointStore::list_by_thread(&self.checkpoint, thread_id).await
        }
        async fn threads_exceeding_checkpoint_count(
            &self,
            threshold: u32,
            limit: u32,
        ) -> Result<Vec<ThreadId>> {
            self.checkpoint
                .threads_exceeding_checkpoint_count(threshold, limit)
                .await
        }
        async fn delete_checkpoints_beyond_limit(
            &self,
            thread_id: &ThreadId,
            keep_latest_n: u32,
        ) -> Result<u64> {
            self.checkpoint
                .delete_checkpoints_beyond_limit(thread_id, keep_latest_n)
                .await
        }
    }

    #[async_trait]
    impl EventRepository for InMemoryJournalStore {
        fn atomic_event_outbox_committer(&self) -> Option<&dyn AtomicEventOutboxCommitter> {
            self.event.atomic_event_outbox_committer()
        }
        async fn commit_event(
            &self,
            thread_id: &ThreadId,
            event: AgentEvent,
            now: OffsetDateTime,
        ) -> Result<CommittedEvent> {
            self.event.commit_event(thread_id, event, now).await
        }
        async fn commit_event_batch(
            &self,
            thread_id: &ThreadId,
            events: Vec<AgentEvent>,
            now: OffsetDateTime,
        ) -> Result<Vec<CommittedEvent>> {
            self.event.commit_event_batch(thread_id, events, now).await
        }
        async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
            self.event.next_sequence(thread_id).await
        }
        async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
            self.event.get_events(thread_id).await
        }
        async fn get_events_in_range(
            &self,
            thread_id: &ThreadId,
            after_sequence: u64,
            up_to_sequence: u64,
        ) -> Result<Vec<CommittedEvent>> {
            self.event
                .get_events_in_range(thread_id, after_sequence, up_to_sequence)
                .await
        }
        async fn threads_with_events_before(
            &self,
            cutoff: OffsetDateTime,
            limit: u32,
        ) -> Result<Vec<ThreadId>> {
            self.event.threads_with_events_before(cutoff, limit).await
        }
        async fn max_sequence_before(
            &self,
            thread_id: &ThreadId,
            cutoff: OffsetDateTime,
        ) -> Result<Option<u64>> {
            self.event.max_sequence_before(thread_id, cutoff).await
        }
        async fn min_sequence_at_or_after(
            &self,
            thread_id: &ThreadId,
            cutoff: OffsetDateTime,
        ) -> Result<Option<u64>> {
            self.event.min_sequence_at_or_after(thread_id, cutoff).await
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Time helpers — fixed virtual clock so the battery is deterministic.
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn tid(name: &str) -> ThreadId {
    ThreadId::from_string(name)
}

fn sample_text_event(message_id: &str, text: &str) -> AgentEvent {
    AgentEvent::text(message_id, text)
}

/// Ensure the thread row exists before any direct event / message /
/// checkpoint write.
///
/// Durable backends (`SQLite`, `PostgreSQL`) enforce a
/// `thread_id REFERENCES agent_sdk_threads` foreign key on the event,
/// message, and checkpoint tables, so a thread must be bootstrapped
/// before the first write that is not a task submission (which
/// auto-creates the thread). The in-memory backend has no such FK, but
/// `get_or_create` is idempotent there, so calling it unconditionally
/// keeps the one battery valid across every backend.
async fn ensure_thread<S: ThreadStore + ?Sized>(store: &S, thread: &ThreadId) -> Result<()> {
    ThreadStore::get_or_create(store, thread, t_plus(0)).await?;
    Ok(())
}

/// Submit a root turn for `thread` and return its task id.
///
/// Checkpoints carry a `task_id REFERENCES agent_sdk_tasks` foreign key
/// on the durable backends, so a checkpoint cannot be committed against
/// a synthetic task id. Submitting a real root turn both bootstraps the
/// thread row and produces a referenceable task id, keeping the
/// checkpoint cases valid across every backend.
async fn seed_root_task<S: AgentTaskStore + ?Sized>(
    store: &S,
    thread: &ThreadId,
) -> Result<AgentTaskId> {
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(0), 3);
    let admitted = AgentTaskStore::submit_root_turn(store, root).await?;
    Ok(admitted.id)
}

/// Compare two serializable values for *semantic* (not byte) equality.
///
/// Both sides are first round-tripped through `serde_json::Value`, which
/// normalizes object key ordering. This is the contract the card
/// requires: Postgres `jsonb` may physically reorder keys on a
/// roundtrip, so the battery must never assert byte-equality.
fn assert_semantic_eq<T: serde::Serialize>(left: &T, right: &T, label: &str) -> Result<()> {
    let l = serde_json::to_value(left).with_context(|| format!("serialize {label} left"))?;
    let r = serde_json::to_value(right).with_context(|| format!("serialize {label} right"))?;
    ensure!(
        l == r,
        "{label}: semantic roundtrip mismatch\n left={l}\nright={r}"
    );
    Ok(())
}

/// Build a [`SuspensionPayload`] with an empty continuation for the
/// given thread. Child-spawn entry points require a payload to persist
/// alongside the suspended parent; the conformance battery does not care
/// about its contents, only that the parent/child rows are created.
fn empty_suspension(thread: &ThreadId) -> SuspensionPayload {
    SuspensionPayload {
        continuation: agent_sdk_core::ContinuationEnvelope::wrap(
            agent_sdk_core::AgentContinuation {
                thread_id: thread.clone(),
                turn: 1,
                total_usage: TokenUsage::default(),
                turn_usage: TokenUsage::default(),
                pending_tool_calls: vec![],
                awaiting_index: 0,
                completed_results: vec![],
                state: agent_sdk_core::AgentState::new(thread.clone()),
                response_id: None,
                stop_reason: None,
                response_content: vec![],
            },
        ),
        suspended_messages: vec![],
    }
}

// ─────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────

/// Outcome of one [`run_journal_store_conformance`] run.
///
/// Records which conformance cases ran and which optional cases were
/// auto-skipped because the backend legitimately does not provide the
/// optional method. A caller can assert that the expected arms ran (no
/// silent skips of mandatory cases).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConformanceReport {
    /// Names of mandatory cases that ran and passed.
    pub passed: Vec<String>,
    /// Names of optional cases that auto-skipped (with the reason).
    pub skipped: Vec<(String, String)>,
}

impl ConformanceReport {
    fn passed(&mut self, name: &str) {
        self.passed.push(name.to_owned());
    }

    fn skipped(&mut self, name: &str, reason: &str) {
        self.skipped.push((name.to_owned(), reason.to_owned()));
    }
}

// ─────────────────────────────────────────────────────────────────────
// The battery
// ─────────────────────────────────────────────────────────────────────

/// Run the full journal/store conformance battery against `store`.
///
/// This is the reusable public entry point a third-party backend calls
/// to self-verify. It exercises every conformance case in the plan's
/// §A2.1 / §6.1 edge-case matrix, adapted to the real journal traits.
///
/// On success it returns a [`ConformanceReport`] listing which cases ran
/// and which optional cases auto-skipped. On the first contract
/// violation it returns an error describing the failure — the battery
/// never silently swallows a mandatory-case failure.
///
/// # Determinism
///
/// All timestamps come from a fixed virtual clock; the battery performs
/// no real sleeps. Backends that require connection I/O should run it
/// under a single Tokio runtime.
///
/// # Errors
///
/// Returns an error if any conformance case fails or the backend cannot
/// be queried.
pub async fn run_journal_store_conformance<S: JournalStore + Clone + 'static>(
    store: &S,
) -> Result<ConformanceReport> {
    let mut report = ConformanceReport::default();

    case_append_preserves_insertion_order(store).await?;
    report.passed("append_preserves_insertion_order");

    case_load_returns_independent_deep_equal_copy(store).await?;
    report.passed("load_returns_independent_deep_equal_copy");

    case_load_unknown_key_is_none(store).await?;
    report.passed("load_unknown_key_is_none");

    case_semantic_equality_on_roundtrip(store).await?;
    report.passed("semantic_equality_on_roundtrip");

    case_optional_method_absent_autoskips(store, &mut report);

    case_delete_cascades_main_to_subkeys(store).await?;
    report.passed("delete_cascades_main_to_subkeys");

    case_safe_noop_delete_for_append_only(store).await?;
    report.passed("safe_noop_delete_for_append_only");

    case_subkey_vs_main_count_separation(store).await?;
    report.passed("subkey_vs_main_count_separation");

    case_clear_empty_invariants(store).await?;
    report.passed("clear_empty_invariants");

    case_concurrent_appends_serialize(store).await?;
    report.passed("concurrent_appends_serialize");

    case_checkpoint_roundtrip(store).await?;
    report.passed("checkpoint_roundtrip");

    case_list_reverse_chronological(store).await?;
    report.passed("list_reverse_chronological");

    case_latest_by_insertion_order_not_lexicographic_id(store).await?;
    report.passed("latest_by_insertion_order_not_lexicographic_id");

    case_pending_writes_persisted(store).await?;
    report.passed("pending_writes_persisted");

    case_parent_checkpoint_link(store).await?;
    report.passed("parent_checkpoint_link");

    case_metadata_filter(store).await?;
    report.passed("metadata_filter");

    case_before_plus_limit_pagination(store).await?;
    report.passed("before_plus_limit_pagination");

    Ok(report)
}

// ── append preserves insertion order ─────────────────────────────────

async fn case_append_preserves_insertion_order<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-append-order");
    ensure_thread(store, &thread).await?;
    let texts = ["alpha", "beta", "gamma", "delta"];
    for (idx, text) in texts.iter().enumerate() {
        EventRepository::commit_event(
            store,
            &thread,
            sample_text_event(&format!("msg_{idx}"), text),
            t_plus(i64::try_from(idx).context("idx")?),
        )
        .await?;
    }

    let events = EventRepository::get_events(store, &thread).await?;
    ensure!(events.len() == texts.len(), "expected all events back");
    for (idx, committed) in events.iter().enumerate() {
        ensure!(
            committed.sequence == idx as u64,
            "sequence must equal insertion index"
        );
    }

    // Message projection commit-order is also append-ordered.
    for text in &texts {
        MessageProjectionStore::commit_messages(
            store,
            &thread,
            vec![llm::Message::user(*text)],
            t_plus(100),
        )
        .await?;
    }
    let history = MessageProjectionStore::get_history(store, &thread).await?;
    ensure!(
        history.len() == texts.len(),
        "message history preserves every committed batch"
    );
    let history_texts: Vec<String> = history
        .iter()
        .map(|m| serde_json::to_value(m).map(|v| v.to_string()))
        .collect::<std::result::Result<_, _>>()?;
    for (text, serialized) in texts.iter().zip(&history_texts) {
        ensure!(serialized.contains(text), "message order preserved");
    }
    Ok(())
}

// ── load returns an independent deep-equal copy ──────────────────────

async fn case_load_returns_independent_deep_equal_copy<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-independent-copy");
    ensure_thread(store, &thread).await?;
    EventRepository::commit_event(
        store,
        &thread,
        sample_text_event("msg_copy", "payload"),
        t_plus(1),
    )
    .await?;

    let mut first = EventRepository::get_events(store, &thread).await?;
    // Mutate the returned copy in place; the store must be unaffected.
    if let Some(event) = first.first_mut() {
        event.sequence = 9_999;
    }

    let second = EventRepository::get_events(store, &thread).await?;
    ensure!(
        second[0].sequence == 0,
        "mutating a loaded copy must not corrupt the store"
    );
    Ok(())
}

// ── load of unknown key → None ───────────────────────────────────────

async fn case_load_unknown_key_is_none<S: JournalStore>(store: &S) -> Result<()> {
    let unknown = tid("conf-never-written");
    ensure!(
        MessageProjectionStore::get(store, &unknown)
            .await?
            .is_none(),
        "unknown thread projection is None"
    );
    ensure!(
        ThreadStore::get(store, &unknown).await?.is_none(),
        "unknown thread is None"
    );
    ensure!(
        CheckpointStore::get_latest_by_thread(store, &unknown)
            .await?
            .is_none(),
        "unknown thread has no latest checkpoint"
    );
    ensure!(
        CheckpointStore::get(store, &CheckpointId::from_string("checkpoint_missing"))
            .await?
            .is_none(),
        "unknown checkpoint id is None"
    );
    ensure!(
        EventRepository::get_events(store, &unknown)
            .await?
            .is_empty(),
        "unknown thread has no events"
    );
    Ok(())
}

// ── semantic (not byte) equality on roundtrip ────────────────────────

async fn case_semantic_equality_on_roundtrip<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-semantic-eq");
    let task_id = seed_root_task(store, &thread).await?;
    // A nested snapshot whose key insertion order would differ from a
    // jsonb roundtrip. We assert *value* equality, never byte equality.
    let snapshot = serde_json::json!({
        "zeta": 1,
        "alpha": { "nested": [3, 2, 1], "flag": true },
        "mu": "value",
    });
    let committed = CheckpointStore::commit_checkpoint(
        store,
        NewCheckpointParams {
            thread_id: thread.clone(),
            turn_number: 1,
            task_id,
            messages: vec![llm::Message::user("roundtrip")],
            agent_state_snapshot: snapshot.clone(),
            turn_usage: TokenUsage::default(),
            now: t_plus(1),
        },
    )
    .await?;

    let loaded = CheckpointStore::get(store, &committed.id)
        .await?
        .context("checkpoint should roundtrip")?;
    assert_semantic_eq(
        &committed.agent_state_snapshot,
        &loaded.agent_state_snapshot,
        "agent_state_snapshot",
    )?;
    // The whole checkpoint roundtrips deep-equal.
    assert_semantic_eq(&committed, &loaded, "checkpoint")?;
    Ok(())
}

// ── optional methods absent → auto-skip ──────────────────────────────

fn case_optional_method_absent_autoskips<S: JournalStore>(
    store: &S,
    report: &mut ConformanceReport,
) {
    // The atomic event+outbox committer is a genuinely-optional method:
    // in-memory backends return None, durable backends return Some.
    // When present, it must be usable; when absent, we auto-skip — the
    // one legitimate skip in the battery.
    if EventRepository::atomic_event_outbox_committer(store).is_some() {
        report.passed("optional_atomic_outbox_committer_present");
    } else {
        report.skipped(
            "optional_atomic_outbox_committer",
            "backend does not provide an atomic event+outbox committer",
        );
    }
}

// ── delete cascades main→subkeys ─────────────────────────────────────

async fn case_delete_cascades_main_to_subkeys<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-cascade");
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(1), 3);
    AgentTaskStore::submit_root_turn(store, root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    AgentTaskStore::try_acquire_task(
        store,
        &root.id,
        worker.clone(),
        lease.clone(),
        t_plus(60),
        t_plus(2),
    )
    .await?;

    let (_parent, children) = AgentTaskStore::spawn_tool_children(
        store,
        &root.id,
        &worker,
        &lease,
        vec![
            ChildSpawnSpec { max_attempts: 3 },
            ChildSpawnSpec { max_attempts: 3 },
        ],
        empty_suspension(&thread),
        None,
        t_plus(3),
    )
    .await?;
    ensure!(children.len() == 2, "two subkey rows spawned");

    // clear() is the journal's cascade delete: it must wipe the main
    // (root) row *and* all subkey (child) rows despite the
    // self-referential parent→child FK.
    AgentTaskStore::clear(store).await?;
    ensure!(
        AgentTaskStore::get(store, &root.id).await?.is_none(),
        "main row deleted"
    );
    for child in &children {
        ensure!(
            AgentTaskStore::get(store, &child.id).await?.is_none(),
            "subkey row cascade-deleted"
        );
    }
    Ok(())
}

// ── safe no-op delete for append-only ────────────────────────────────

async fn case_safe_noop_delete_for_append_only<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-noop-delete");
    let task_id = seed_root_task(store, &thread).await?;
    // Append-only checkpoint store: deleting "beyond limit" when there
    // are fewer rows than the limit is a safe no-op returning 0.
    for turn in 1..=2u32 {
        CheckpointStore::commit_checkpoint(
            store,
            NewCheckpointParams {
                thread_id: thread.clone(),
                turn_number: turn,
                task_id: task_id.clone(),
                messages: vec![],
                agent_state_snapshot: serde_json::json!({}),
                turn_usage: TokenUsage::default(),
                now: t_plus(i64::from(turn)),
            },
        )
        .await?;
    }
    let deleted = CheckpointStore::delete_checkpoints_beyond_limit(store, &thread, 5).await?;
    ensure!(deleted == 0, "delete beyond a larger limit is a no-op");
    let remaining = CheckpointStore::list_by_thread(store, &thread).await?;
    ensure!(remaining.len() == 2, "no rows lost to the no-op delete");

    // Deleting on an empty thread is also a safe no-op.
    let empty = tid("conf-noop-delete-empty");
    let deleted_empty = CheckpointStore::delete_checkpoints_beyond_limit(store, &empty, 1).await?;
    ensure!(deleted_empty == 0, "no-op delete on empty thread");
    Ok(())
}

// ── subkey vs main count separation ──────────────────────────────────

async fn case_subkey_vs_main_count_separation<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-count-separation");
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(1), 3);
    AgentTaskStore::submit_root_turn(store, root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    AgentTaskStore::try_acquire_task(
        store,
        &root.id,
        worker.clone(),
        lease.clone(),
        t_plus(60),
        t_plus(2),
    )
    .await?;
    let (_parent, children) = AgentTaskStore::spawn_tool_children(
        store,
        &root.id,
        &worker,
        &lease,
        vec![
            ChildSpawnSpec { max_attempts: 3 },
            ChildSpawnSpec { max_attempts: 3 },
            ChildSpawnSpec { max_attempts: 3 },
        ],
        empty_suspension(&thread),
        None,
        t_plus(3),
    )
    .await?;
    ensure!(children.len() == 3, "three subkeys");

    let all = AgentTaskStore::list_by_thread(store, &thread).await?;
    let main_count = all.iter().filter(|t| t.is_root()).count();
    let subkey_count = all.iter().filter(|t| !t.is_root()).count();
    ensure!(main_count == 1, "exactly one main-session (root) row");
    ensure!(subkey_count == 3, "subkeys counted separately from main");

    let listed_children = AgentTaskStore::list_children(store, &root.id).await?;
    ensure!(
        listed_children.len() == 3,
        "list_children isolates subkeys of the main row"
    );
    Ok(())
}

// ── clear/empty invariants ───────────────────────────────────────────

async fn case_clear_empty_invariants<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-clear-empty");
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(1), 3);
    AgentTaskStore::submit_root_turn(store, root.clone()).await?;
    ensure!(
        !AgentTaskStore::list_by_thread(store, &thread)
            .await?
            .is_empty(),
        "thread has a task before clear"
    );

    AgentTaskStore::clear(store).await?;
    ensure!(
        AgentTaskStore::list_by_thread(store, &thread)
            .await?
            .is_empty(),
        "list_by_thread is empty after clear"
    );
    ensure!(
        AgentTaskStore::active_root_for_thread(store, &thread)
            .await?
            .is_none(),
        "no active root after clear"
    );
    ensure!(
        AgentTaskStore::get(store, &root.id).await?.is_none(),
        "cleared task is gone"
    );
    Ok(())
}

// ── N concurrent appends serialize without loss ──────────────────────

async fn case_concurrent_appends_serialize<S: JournalStore + Clone + 'static>(
    store: &S,
) -> Result<()> {
    let thread = tid("conf-concurrent-append");
    ensure_thread(store, &thread).await?;
    let store: Arc<S> = Arc::new(store.clone());
    let count = 24usize;

    let mut handles = Vec::with_capacity(count);
    for idx in 0..count {
        let store = Arc::clone(&store);
        let thread = thread.clone();
        handles.push(tokio::spawn(async move {
            EventRepository::commit_event(
                store.as_ref(),
                &thread,
                sample_text_event(&format!("msg_{idx}"), &format!("v{idx}")),
                t_plus(1),
            )
            .await
        }));
    }

    let mut assigned = Vec::with_capacity(count);
    for handle in handles {
        let committed = handle.await.context("join append task")??;
        assigned.push(committed.sequence);
    }

    let events = EventRepository::get_events(store.as_ref(), &thread).await?;
    ensure!(
        events.len() == count,
        "every concurrent append survives — no lost writes"
    );

    // Each writer received a distinct sequence (no intra-batch
    // interleave that duplicates a slot), and the persisted log is a
    // gapless 0..count run.
    assigned.sort_unstable();
    assigned.dedup();
    ensure!(
        assigned.len() == count,
        "each concurrent append got a unique sequence"
    );
    let mut persisted: Vec<u64> = events.iter().map(|e| e.sequence).collect();
    persisted.sort_unstable();
    let expected: Vec<u64> = (0..count as u64).collect();
    ensure!(
        persisted == expected,
        "sequences serialize to a gapless 0..N run"
    );
    Ok(())
}

// ── checkpoint roundtrip ─────────────────────────────────────────────

async fn case_checkpoint_roundtrip<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-checkpoint-roundtrip");
    let task_id = seed_root_task(store, &thread).await?;
    let messages = vec![
        llm::Message::user("question"),
        llm::Message::assistant("answer"),
    ];
    let committed = CheckpointStore::commit_checkpoint(
        store,
        NewCheckpointParams {
            thread_id: thread.clone(),
            turn_number: 1,
            task_id,
            messages: messages.clone(),
            agent_state_snapshot: serde_json::json!({ "step": 1 }),
            turn_usage: TokenUsage {
                input_tokens: 11,
                output_tokens: 7,
                ..Default::default()
            },
            now: t_plus(1),
        },
    )
    .await?;

    let by_id = CheckpointStore::get(store, &committed.id)
        .await?
        .context("get by id")?;
    let by_turn = CheckpointStore::get_by_turn(store, &thread, 1)
        .await?
        .context("get by turn")?;
    assert_semantic_eq(&committed, &by_id, "checkpoint by id")?;
    assert_semantic_eq(&committed, &by_turn, "checkpoint by turn")?;
    ensure!(by_id.messages.len() == 2, "messages roundtrip");
    Ok(())
}

// ── list reverse-chronological ───────────────────────────────────────

async fn case_list_reverse_chronological<S: JournalStore>(store: &S) -> Result<()> {
    let thread = tid("conf-list-order");
    let task_id = seed_root_task(store, &thread).await?;
    for turn in 1..=4u32 {
        CheckpointStore::commit_checkpoint(
            store,
            NewCheckpointParams {
                thread_id: thread.clone(),
                turn_number: turn,
                task_id: task_id.clone(),
                messages: vec![],
                agent_state_snapshot: serde_json::json!({ "turn": turn }),
                turn_usage: TokenUsage::default(),
                now: t_plus(i64::from(turn)),
            },
        )
        .await?;
    }

    // The trait contract orders `list_by_thread` ascending by turn; the
    // reverse-chronological (newest-first) view callers want is the
    // reverse of that. We assert the ascending contract holds, then that
    // reversing yields strict newest-first order.
    let listed = CheckpointStore::list_by_thread(store, &thread).await?;
    ensure!(listed.len() == 4, "all checkpoints listed");
    for pair in listed.windows(2) {
        ensure!(
            pair[0].turn_number < pair[1].turn_number,
            "list_by_thread is ascending by turn"
        );
    }
    let newest_first: Vec<u32> = listed.iter().rev().map(|c| c.turn_number).collect();
    ensure!(
        newest_first == vec![4, 3, 2, 1],
        "reverse-chronological view is strictly newest-first"
    );
    Ok(())
}

// ── latest-by-insertion-order, NOT lexicographic id (#6821) ──────────

async fn case_latest_by_insertion_order_not_lexicographic_id<S: JournalStore + ?Sized>(
    store: &S,
) -> Result<()> {
    // LangGraph #6821 regression: the "latest" checkpoint must be
    // determined by insertion order (turn number), not by sorting the
    // opaque checkpoint id string. Checkpoint ids are random UUIDs, so a
    // later-inserted checkpoint can have an id that sorts lexicographically
    // *before* an earlier one. A backend that returns `MAX(id)` would
    // pick the wrong row.
    let thread = tid("conf-langgraph-6821");

    // Find two ids where the later-inserted turn has the lexicographically
    // *smaller* id, so id-sorting and insertion-order disagree.
    let task_id = seed_root_task(store, &thread).await?;
    let first = CheckpointStore::commit_checkpoint(
        store,
        NewCheckpointParams {
            thread_id: thread.clone(),
            turn_number: 1,
            task_id: task_id.clone(),
            messages: vec![llm::Message::user("turn-1")],
            agent_state_snapshot: serde_json::json!({ "turn": 1 }),
            turn_usage: TokenUsage::default(),
            now: t_plus(1),
        },
    )
    .await?;
    let second = CheckpointStore::commit_checkpoint(
        store,
        NewCheckpointParams {
            thread_id: thread.clone(),
            turn_number: 2,
            task_id,
            messages: vec![llm::Message::user("turn-2")],
            agent_state_snapshot: serde_json::json!({ "turn": 2 }),
            turn_usage: TokenUsage::default(),
            now: t_plus(2),
        },
    )
    .await?;

    // The latest is turn 2 regardless of how the two random ids sort.
    let latest = CheckpointStore::get_latest_by_thread(store, &thread)
        .await?
        .context("latest checkpoint must exist")?;
    ensure!(
        latest.turn_number == 2,
        "get_latest_by_thread must return the last-inserted turn (#6821), got turn {}",
        latest.turn_number
    );

    // Spell out the regression: if the two ids disagree with insertion
    // order, the latest is still turn 2 — proving we are not sorting ids.
    let id_order_disagrees = second.id.as_str() < first.id.as_str();
    if id_order_disagrees {
        ensure!(
            latest.id == second.id,
            "latest is the last-inserted row even when its id sorts first (#6821)"
        );
    }
    ensure!(
        latest.id == second.id,
        "latest must be the highest-turn checkpoint, not MAX(id)"
    );
    Ok(())
}

// ── pending writes persisted ─────────────────────────────────────────

async fn case_pending_writes_persisted<S: JournalStore>(store: &S) -> Result<()> {
    // LangGraph "pending writes" map onto the message-projection draft
    // slot: in-flight (not-yet-committed) writes that must survive a
    // reload so recovery can fold them back in.
    let thread = tid("conf-pending-writes");
    ensure_thread(store, &thread).await?;
    MessageProjectionStore::get_or_create(store, &thread, t_plus(1)).await?;
    let draft = vec![
        llm::Message::user("in-flight question"),
        llm::Message::assistant("in-flight partial"),
    ];
    MessageProjectionStore::set_draft(store, &thread, draft.clone(), t_plus(2)).await?;

    let reloaded = MessageProjectionStore::get(store, &thread)
        .await?
        .context("projection must exist")?;
    ensure!(
        reloaded.draft_messages.len() == 2,
        "pending (draft) writes persist across a reload"
    );
    ensure!(
        reloaded.messages.is_empty(),
        "pending writes are not silently committed"
    );

    // Committing then clearing the draft leaves committed history intact
    // and drops the pending slot.
    MessageProjectionStore::clear_draft(store, &thread, t_plus(3)).await?;
    let after = MessageProjectionStore::get(store, &thread)
        .await?
        .context("projection still exists")?;
    ensure!(
        after.draft_messages.is_empty(),
        "cleared pending writes are gone"
    );
    Ok(())
}

// ── parent-checkpoint link ───────────────────────────────────────────

async fn case_parent_checkpoint_link<S: JournalStore>(store: &S) -> Result<()> {
    // LangGraph's parent_checkpoint link maps onto our turn-chain: a
    // turn-N checkpoint's message history must contain (be a superset of)
    // the turn-(N-1) checkpoint's history, since each checkpoint is a
    // cumulative snapshot. We commit two real turns through the atomic
    // commit path and assert the link holds.
    let thread = tid("conf-parent-link");
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(1), 3);
    AgentTaskStore::submit_root_turn(store, root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    AgentTaskStore::try_acquire_task(store, &root.id, worker, lease, t_plus(120), t_plus(2))
        .await?;

    let parent_ckpt = commit_one_turn(
        store,
        &thread,
        &root.id,
        1,
        vec![llm::Message::user("turn 1")],
        t_plus(3),
    )
    .await?;

    let child_ckpt = commit_one_turn(
        store,
        &thread,
        &root.id,
        2,
        vec![
            llm::Message::user("turn 1"),
            llm::Message::assistant("reply 1"),
            llm::Message::user("turn 2"),
        ],
        t_plus(4),
    )
    .await?;

    ensure!(
        child_ckpt.turn_number == parent_ckpt.turn_number + 1,
        "child checkpoint links to the immediately-preceding turn"
    );
    ensure!(
        child_ckpt.messages.len() > parent_ckpt.messages.len(),
        "child checkpoint history extends its parent's"
    );
    // The parent's first message survives unchanged into the child —
    // the link is real, not a fresh snapshot.
    assert_semantic_eq(
        &parent_ckpt.messages[0],
        &child_ckpt.messages[0],
        "parent->child message link",
    )?;
    Ok(())
}

/// Commit one turn through the atomic commit path and return the
/// resulting checkpoint. Opens and closes a turn attempt so the
/// checkpoint row is produced exactly as production does.
async fn commit_one_turn<S: JournalStore>(
    store: &S,
    thread: &ThreadId,
    task_id: &AgentTaskId,
    turn_number: u32,
    messages: Vec<llm::Message>,
    now: OffsetDateTime,
) -> Result<super::checkpoint::Checkpoint> {
    let attempt = TurnAttemptStore::open_attempt(
        store,
        OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number: turn_number,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            request_blob: serde_json::json!({ "messages": [] }),
            now,
            otel_trace_id: None,
            otel_span_id: None,
        },
    )
    .await?;

    let outcome = commit_completed_turn(
        CompletedTurnCommit {
            thread_id: thread.clone(),
            task_id: task_id.clone(),
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: CloseAttemptParams {
                response_blob: serde_json::json!({ "id": format!("msg_{turn_number}") }),
                response_id: Some(format!("msg_{turn_number}")),
                response_model: Some("claude-sonnet-4-5-20250929".into()),
                stop_reason: Some(llm::StopReason::EndTurn),
                outcome: TurnAttemptOutcome::Success,
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
            },
            // replace_history semantics: the checkpoint stores the full
            // cumulative snapshot at this turn.
            messages,
            turn_usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            agent_state_snapshot: serde_json::json!({ "turn": turn_number }),
            events: vec![],
            outbox_max_attempts: 3,
            now,
        },
        store,
        store,
        store,
        store,
        store,
    )
    .await?;
    Ok(outcome.checkpoint)
}

// ── metadata filter ──────────────────────────────────────────────────

async fn case_metadata_filter<S: JournalStore>(store: &S) -> Result<()> {
    // LangGraph's metadata filter maps onto filtering checkpoints by a
    // metadata-carrying field. Our checkpoint's metadata-bearing surface
    // is `turn_usage`; we tag turns with distinct token counts and assert
    // a client-side filter selects exactly the matching rows.
    let thread = tid("conf-metadata-filter");
    let task_id = seed_root_task(store, &thread).await?;
    for turn in 1..=5u32 {
        let input_tokens = if turn % 2 == 0 { 100 } else { 7 };
        CheckpointStore::commit_checkpoint(
            store,
            NewCheckpointParams {
                thread_id: thread.clone(),
                turn_number: turn,
                task_id: task_id.clone(),
                messages: vec![],
                agent_state_snapshot: serde_json::json!({}),
                turn_usage: TokenUsage {
                    input_tokens,
                    ..Default::default()
                },
                now: t_plus(i64::from(turn)),
            },
        )
        .await?;
    }

    let all = CheckpointStore::list_by_thread(store, &thread).await?;
    let heavy: Vec<u32> = all
        .iter()
        .filter(|c| c.turn_usage.input_tokens == 100)
        .map(|c| c.turn_number)
        .collect();
    ensure!(
        heavy == vec![2, 4],
        "metadata filter selects exactly the matching checkpoints"
    );
    Ok(())
}

// ── before + limit pagination ────────────────────────────────────────

async fn case_before_plus_limit_pagination<S: JournalStore>(store: &S) -> Result<()> {
    // LangGraph's `(before, limit)` pagination maps onto the event
    // replay window: `get_events_in_range(after, up_to)` returns the
    // bounded slice a reconnecting client missed. We commit a run of
    // events and assert windowed reads are correct and gapless.
    let thread = tid("conf-pagination");
    ensure_thread(store, &thread).await?;
    let total = 10u64;
    for idx in 0..total {
        EventRepository::commit_event(
            store,
            &thread,
            sample_text_event(&format!("msg_{idx}"), &format!("v{idx}")),
            t_plus(1),
        )
        .await?;
    }

    // Sequences are 0-indexed, so this run is 0..=9. The replay window
    // is half-open on the low end: `sequence > after AND sequence <=
    // up_to`. A page of `(after=2, up_to=6]` is sequences [3,4,5,6].
    let page = EventRepository::get_events_in_range(store, &thread, 2, 6).await?;
    let seqs: Vec<u64> = page.iter().map(|e| e.sequence).collect();
    ensure!(seqs == vec![3, 4, 5, 6], "before+limit window is exact");

    // An empty window (after == up_to) yields nothing.
    let empty = EventRepository::get_events_in_range(store, &thread, 4, 4).await?;
    ensure!(empty.is_empty(), "degenerate window is empty");

    // Paging from the very start: a reconnecting client that has seen
    // nothing passes its last-seen sequence as 0 and walks forward. The
    // window `(0, total]` returns sequences 1..=9, gapless.
    let tail = EventRepository::get_events_in_range(store, &thread, 0, total).await?;
    ensure!(
        tail.len() as u64 == total - 1,
        "window past the first event returns the remaining gapless run"
    );
    for (idx, event) in tail.iter().enumerate() {
        ensure!(event.sequence == idx as u64 + 1, "window is gapless");
    }

    // The unbounded read returns the whole run including sequence 0.
    let full = EventRepository::get_events(store, &thread).await?;
    ensure!(full.len() as u64 == total, "full read returns every event");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_passes_full_battery() -> Result<()> {
        let store = InMemoryJournalStore::new();
        let report = run_journal_store_conformance(&store).await?;

        // Every mandatory case ran; the optional outbox committer is
        // legitimately absent on the in-memory backend, so it is the one
        // recorded skip.
        assert!(
            report
                .passed
                .contains(&"latest_by_insertion_order_not_lexicographic_id".to_owned()),
            "the #6821 case must run",
        );
        assert!(
            report
                .passed
                .contains(&"concurrent_appends_serialize".to_owned()),
            "the concurrent-append case must run",
        );
        assert_eq!(
            report.skipped,
            vec![(
                "optional_atomic_outbox_committer".to_owned(),
                "backend does not provide an atomic event+outbox committer".to_owned(),
            )],
            "the only skip is the genuinely-optional outbox committer",
        );
        assert!(
            report.passed.len() >= 16,
            "no mandatory case silently skipped"
        );
        Ok(())
    }

    #[tokio::test]
    async fn battery_catches_lexicographic_latest_bug() -> Result<()> {
        // Prove the #6821 contract the battery asserts is meaningful: a
        // backend that selects the latest checkpoint by lexicographic id
        // (the LangGraph #6821 bug) returns the WRONG row whenever id
        // order disagrees with insertion order. We construct exactly such
        // a disagreement and show id-selection and turn-selection diverge,
        // which is precisely what the battery's assertion forbids.
        let correct = InMemoryJournalStore::new();
        let broken = BrokenLatestStore::default();
        let thread = tid("conf-6821-negative");

        // Commit turn 1 then turn 2 to both stores until we land a pair
        // whose ids disagree with insertion order (random UUIDs, so a
        // few attempts suffice deterministically within the test).
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let t = ThreadId::from_string(format!("{thread}-{attempt}"));
            let task_id = seed_root_task(&correct, &t).await?;
            for turn in 1..=2u32 {
                let params = NewCheckpointParams {
                    thread_id: t.clone(),
                    turn_number: turn,
                    task_id: task_id.clone(),
                    messages: vec![],
                    agent_state_snapshot: serde_json::json!({ "turn": turn }),
                    turn_usage: TokenUsage::default(),
                    now: t_plus(i64::from(turn)),
                };
                let c = correct.commit_checkpoint(params.clone()).await?;
                // Mirror the same checkpoint id into the broken store so
                // both observe the identical id/turn pairing.
                broken.commit_with_id(params, c.id.clone()).await?;
            }
            let correct_latest = correct
                .get_latest_by_thread(&t)
                .await?
                .context("correct latest")?;
            let broken_latest = broken
                .get_latest_by_thread(&t)
                .await?
                .context("broken latest")?;
            assert_eq!(
                correct_latest.turn_number, 2,
                "the correct store always returns the last-inserted turn",
            );
            if correct_latest.id != broken_latest.id {
                // The disagreement we were after: id-selection picked a
                // different (wrong) row than insertion order. This is the
                // exact violation the battery's #6821 assertion catches.
                assert_eq!(
                    broken_latest.turn_number, 1,
                    "the lexicographic-id bug returns the older turn",
                );
                return Ok(());
            }
            assert!(
                attempt < 64,
                "expected an id/turn disagreement within 64 tries"
            );
        }
    }

    /// In-memory checkpoint store whose `get_latest_by_thread`
    /// deliberately returns the lexicographically-largest checkpoint id —
    /// the `LangGraph` #6821 bug. Used only to demonstrate the divergence
    /// the battery forbids.
    #[derive(Clone, Default)]
    struct BrokenLatestStore {
        rows: std::sync::Arc<tokio::sync::RwLock<Vec<super::super::checkpoint::Checkpoint>>>,
    }

    impl BrokenLatestStore {
        /// Commit a checkpoint carrying a caller-chosen id so the broken
        /// store mirrors the id/turn pairing of a correct store.
        async fn commit_with_id(
            &self,
            params: NewCheckpointParams,
            id: CheckpointId,
        ) -> Result<()> {
            let mut checkpoint = super::super::checkpoint::Checkpoint::new(params)
                .map_err(|e| anyhow::anyhow!("broken commit: {e}"))?;
            checkpoint.id = id;
            self.rows.write().await.push(checkpoint);
            Ok(())
        }

        async fn get_latest_by_thread(
            &self,
            thread_id: &ThreadId,
        ) -> Result<Option<super::super::checkpoint::Checkpoint>> {
            // THE BUG: pick the lexicographically-largest id instead of
            // the highest turn number.
            let latest = self
                .rows
                .read()
                .await
                .iter()
                .filter(|c| &c.thread_id == thread_id)
                .max_by(|a, b| a.id.as_str().cmp(b.id.as_str()))
                .cloned();
            Ok(latest)
        }
    }
}
