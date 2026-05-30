//! Atomic completed-turn commit path.
//!
//! [`commit_completed_turn`] is the single entry point that advances
//! all projections when a turn has committed successfully. A single
//! call atomically:
//!
//! 1. Closes the turn attempt (audit record).
//! 2. Commits the turn to the thread aggregate (bumps
//!    `committed_turns` and `total_usage`).
//! 3. Appends messages to the message projection.
//! 4. Creates a checkpoint for `(thread_id, turn_number)`.
//! 5. Clears any in-flight draft on the message projection — the
//!    suspension paths populate this slot at every tool boundary so
//!    a mid-turn failure can still surface the conversation through
//!    [`super::thread_recover`]. A successful commit subsumes the
//!    draft, so it must be wiped before the next turn starts.
//!
//! If any step fails the function returns `Err` and no subsequent
//! steps execute. For the in-memory stores each individual call is
//! internally atomic (single write-lock scope). Durable backends can
//! expose an atomic transaction hook through
//! [`super::thread_store::ThreadStore::atomic_completed_turn_committer`]
//! so this helper routes steps 1-5 — plus the turn's lifecycle events
//! and their coalesced advisory outbox row — through one database
//! transaction (see "Event + state atomicity" below).
//! Atomic backends are responsible for clearing the draft as part of
//! their transaction — the helper does not call `clear_draft` after
//! the atomic hook returns, because doing so outside the transaction
//! would open a tiny crash window where committed history and draft
//! could both contain the same messages.
//!
//! # Guarantees
//!
//! - A completed turn creates **exactly one** checkpoint for
//!   `(thread_id, turn_number)`.
//! - Thread aggregate, message projection, closed turn attempt,
//!   checkpoint, and draft cleanup commit together.
//! - Failed or cancelled turns that never call this function do not
//!   create checkpoints, but the draft persisted at the most recent
//!   suspension boundary survives so the next turn can recover the
//!   in-flight conversation.
//!
//! # Event + state atomicity (Phase 10 · D)
//!
//! On a durable backend the lifecycle events **and** their single
//! coalesced `thread_events_available` advisory outbox row are written
//! inside the same transaction as the state projections, via
//! [`super::completed_turn_transaction::AtomicCompletedTurnCommitter`].
//! The previous design committed events in a *separate* transaction
//! after the state commit, so a crash in between could leave a
//! committed turn with no persisted events and a lost in-process
//! notify. Folding both into one transaction makes the invariant
//! "events exist iff the turn committed" hold on Postgres and `SQLite`.
//! The in-memory path keeps committing events after the state steps
//! because each individual store mutation is already atomic.
//!
//! # What this module does **not** own
//!
//! - Recovery loaders — see [`super::thread_recover`].
//! - Task state transitions — the caller is responsible for calling
//!   [`super::store::AgentTaskStore::complete_task`] separately.

use agent_sdk_core::events::AgentEvent;
use agent_sdk_core::{ThreadId, TokenUsage, llm};
use anyhow::{Context, Result};
use time::OffsetDateTime;

use super::checkpoint::Checkpoint;
use super::checkpoint::NewCheckpointParams;
use super::checkpoint_store::CheckpointStore;
use super::committed_event::CommittedEvent;
use super::event_repository::EventRepository;
use super::message_store::MessageProjectionStore;
use super::task::AgentTaskId;
use super::thread::Thread;
use super::thread_store::ThreadStore;
use super::turn_attempt::{CloseAttemptParams, TurnAttempt, TurnAttemptId};
use super::turn_attempt_store::TurnAttemptStore;

// ─────────────────────────────────────────────────────────────────────
// Param struct
// ─────────────────────────────────────────────────────────────────────

/// Arguments for [`commit_completed_turn`].
///
/// Named fields prevent positional confusion — the commit path is
/// the most critical write in the journal.
#[derive(Clone, Debug)]
pub struct CompletedTurnCommit {
    /// Thread this turn belongs to.
    pub thread_id: ThreadId,
    /// Task that produced this turn.
    pub task_id: AgentTaskId,
    /// The open attempt to close.
    pub turn_attempt_id: TurnAttemptId,
    /// Parameters for closing the turn attempt.
    pub close_attempt_params: CloseAttemptParams,
    /// Messages to append to the message projection.
    pub messages: Vec<llm::Message>,
    /// Token usage for this turn.
    pub turn_usage: TokenUsage,
    /// Opaque agent-state snapshot for v1 recovery.
    pub agent_state_snapshot: serde_json::Value,
    /// Lifecycle events to commit atomically with the turn.
    ///
    /// On a durable backend that exposes an
    /// [`AtomicCompletedTurnCommitter`](super::completed_turn_transaction::AtomicCompletedTurnCommitter),
    /// these events — together with the single coalesced advisory outbox
    /// row — are written inside the **same** transaction as the state
    /// projections (attempt close, thread aggregate, message head, raw
    /// message batch, checkpoint). A crash between the state commit and
    /// the event commit can therefore no longer leave a committed turn
    /// with zero persisted events.
    ///
    /// On the non-atomic in-memory path each store mutation is already
    /// atomic under its own write lock, so the events are committed via
    /// [`EventRepository::commit_event_batch`]
    /// after the state steps. An empty vec skips the event-commit step.
    pub events: Vec<AgentEvent>,
    /// Maximum relay attempts for the coalesced advisory outbox row
    /// written alongside `events` on the atomic durable path.
    ///
    /// Ignored when `events` is empty or when the backend has no atomic
    /// committer (the in-memory path writes no outbox row from this
    /// helper).
    pub outbox_max_attempts: u32,
    /// Current wall-clock time.
    pub now: OffsetDateTime,
}

/// Default relay attempt budget for the coalesced `thread_events_available`
/// advisory row written alongside a completed turn's lifecycle events.
///
/// Matches the long-standing 3-attempt default used elsewhere for the
/// event outbox; production callers may override per [`CompletedTurnCommit`].
pub const DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS: u32 = 3;

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// The result of a successful [`commit_completed_turn`].
///
/// Contains the committed state of all projections so the caller can
/// inspect the new thread aggregate and checkpoint without extra
/// round-trips.
#[derive(Clone, Debug)]
pub struct CommitOutcome {
    /// The thread aggregate after the turn committed.
    pub thread: Thread,
    /// The checkpoint created for this turn.
    pub checkpoint: Checkpoint,
    /// The closed turn attempt.
    pub closed_attempt: TurnAttempt,
    /// Lifecycle events committed as part of this turn.
    pub committed_events: Vec<CommittedEvent>,
}

// ─────────────────────────────────────────────────────────────────────
// Atomic commit
// ─────────────────────────────────────────────────────────────────────

/// Atomically commit a completed turn across all projections.
///
/// This is the heart of the crash-safety guarantee. A single call
/// advances the turn-attempt audit, thread aggregate, message
/// projection, and checkpoint table together. If any step fails the
/// function returns `Err` before touching subsequent stores.
///
/// # Steps
///
/// 1. **Close attempt** — fills response fields on the open turn
///    attempt.
/// 2. **Commit thread** — increments `committed_turns` and
///    accumulates `total_usage` on the thread aggregate.
/// 3. **Append messages** — extends the committed message history.
/// 4. **Create checkpoint** — writes a snapshot at
///    `(thread_id, thread.committed_turns)`.
///
/// # Errors
///
/// Returns an error if any step fails. Callers should treat a
/// partial failure as a bug — the in-memory stores guarantee
/// per-call atomicity, and durable backends should surface an atomic
/// transaction hook.
pub async fn commit_completed_turn(
    mut params: CompletedTurnCommit,
    thread_store: &dyn ThreadStore,
    message_store: &dyn MessageProjectionStore,
    turn_attempt_store: &dyn TurnAttemptStore,
    checkpoint_store: &dyn CheckpointStore,
    event_repo: &dyn EventRepository,
) -> Result<CommitOutcome> {
    #[cfg(feature = "otel")]
    let started_at = std::time::Instant::now();

    if let Some(committer) = thread_store.atomic_completed_turn_committer() {
        // Atomic durable path: events + the coalesced advisory outbox
        // row commit inside the SAME transaction as the state
        // projections, so the params (including `events`) are handed
        // straight to the committer. This closes the crash window where
        // a host death between the state commit and a separate event
        // commit left a committed turn with no persisted events.
        //
        // Phase 10 · D × 11 · A: because the events now commit inside the
        // atomic transaction, there is no separate post-commit event write
        // on this path. The `commit.before_event_commit` failpoint that
        // 11 · A placed here therefore moves to its true crash boundary —
        // immediately before the atomic `tx.commit()` in the backend
        // committer (`agent-service-host`), alongside the
        // `InjectedCommitFailure` test hook. Both still target the
        // "about to commit the events/outbox" boundary, now inside the
        // single transaction.
        let outcome = committer
            .commit_completed_turn_atomic(params)
            .await
            .context("commit: atomic completed-turn transaction")?;

        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_journal_commit(
            crate::observability::attrs::COMMIT_KIND_ATOMIC,
            started_at.elapsed().as_secs_f64(),
        );

        return Ok(outcome);
    }

    // Non-atomic in-memory path: each store mutation is already atomic
    // under its own write lock. Pull the events out so the state steps
    // run first, then commit the events as the final step.
    let events = std::mem::take(&mut params.events);
    let thread_id_for_events = params.thread_id.clone();

    // 1. Close the turn attempt.
    let closed_attempt = turn_attempt_store
        .close_attempt(
            &params.turn_attempt_id,
            params.close_attempt_params,
            params.now,
        )
        .await
        .context("commit: close turn attempt")?;

    // 2. Commit the turn to the thread aggregate.
    let thread = thread_store
        .commit_turn(&params.thread_id, &params.turn_usage, params.now)
        .await
        .context("commit: advance thread aggregate")?;

    // 3. Append messages to the message projection.
    let updated_projection = message_store
        .commit_messages(&params.thread_id, params.messages, params.now)
        .await
        .context("commit: append messages")?;

    // 4. Create the checkpoint with the *full* accumulated history,
    //    not the delta — recovery must be able to restore the thread
    //    to this turn without replaying prior turns.
    //
    let checkpoint = checkpoint_store
        .commit_checkpoint(NewCheckpointParams {
            thread_id: params.thread_id.clone(),
            turn_number: thread.committed_turns,
            task_id: params.task_id,
            messages: updated_projection.messages,
            agent_state_snapshot: params.agent_state_snapshot,
            turn_usage: params.turn_usage,
            now: params.now,
        })
        .await
        .context("commit: create checkpoint")?;

    // 5. Clear the in-flight draft so the next turn starts with a
    //    clean slot. The committed history and checkpoint we just
    //    wrote already contain everything the suspension path
    //    snapshotted into the draft. Calling `clear_draft` is safe
    //    when the slot is already empty — the store returns `None`
    //    on first-turn threads with no projection row yet, which
    //    means there was nothing to clear in the first place.
    //
    //    The non-atomic path (in-memory stores) accepts a tiny race
    //    window: a concurrent reader between step 4 and step 5 could
    //    observe the committed history together with a stale draft.
    //    Durable backends close this gap by clearing the draft inside
    //    their atomic completed-turn transaction; see the module
    //    docs.
    message_store
        .clear_draft(&params.thread_id, params.now)
        .await
        .context("commit: clear in-flight draft")?;

    // Failpoint: simulate a crash after the thread/checkpoint/draft
    // steps committed but before lifecycle events are persisted. No-op
    // (and not compiled) unless the `failpoints` feature is enabled.
    crate::fail_point!("commit.before_event_commit");

    // 6. Commit lifecycle events. Skipped when no events are provided
    //    (e.g. callers that don't produce lifecycle events yet).
    let committed_events = if events.is_empty() {
        Vec::new()
    } else {
        event_repo
            .commit_event_batch(&thread_id_for_events, events, params.now)
            .await
            .context("commit: persist lifecycle events")?
    };

    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global().record_journal_commit(
        crate::observability::attrs::COMMIT_KIND_NON_ATOMIC,
        started_at.elapsed().as_secs_f64(),
    );

    Ok(CommitOutcome {
        thread,
        checkpoint,
        closed_attempt,
        committed_events,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::checkpoint_store::InMemoryCheckpointStore;
    use super::super::event_repository::InMemoryEventRepository;
    use super::super::message_store::InMemoryMessageProjectionStore;
    use super::super::thread_store::InMemoryThreadStore;
    use super::super::turn_attempt::{OpenAttemptParams, TurnAttemptOutcome};
    use super::super::turn_attempt_store::InMemoryTurnAttemptStore;
    use super::*;
    use agent_sdk_core::audit::AuditProvenance;
    use anyhow::Context;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-commit-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-commit-b")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        }
    }

    fn sample_messages() -> Vec<llm::Message> {
        vec![
            llm::Message::user("hello"),
            llm::Message::assistant("hi there"),
        ]
    }

    fn sample_close_params() -> CloseAttemptParams {
        CloseAttemptParams {
            response_blob: serde_json::json!({
                "id": "msg_01",
                "content": [{"type": "text", "text": "hi there"}]
            }),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_core::llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
        }
    }

    struct Stores {
        threads: InMemoryThreadStore,
        messages: InMemoryMessageProjectionStore,
        attempts: InMemoryTurnAttemptStore,
        checkpoints: InMemoryCheckpointStore,
        events: InMemoryEventRepository,
    }

    impl Stores {
        fn new() -> Self {
            Self {
                threads: InMemoryThreadStore::new(),
                messages: InMemoryMessageProjectionStore::new(),
                attempts: InMemoryTurnAttemptStore::new(),
                checkpoints: InMemoryCheckpointStore::new(),
                events: InMemoryEventRepository::new(),
            }
        }

        /// Open an attempt and return its ID (helper for tests).
        async fn open_attempt(&self, task_id: &AgentTaskId, attempt_number: u32) -> TurnAttemptId {
            let attempt = self
                .attempts
                .open_attempt(OpenAttemptParams {
                    task_id: task_id.clone(),
                    attempt_number,
                    provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                    request_blob: serde_json::json!({"messages": []}),
                    now: t0(),
                    otel_trace_id: None,
                    otel_span_id: None,
                })
                .await
                .expect("open attempt");
            attempt.id
        }
    }

    // ── Happy path ───────────────────────────────────────────────

    #[tokio::test]
    async fn commit_creates_checkpoint_and_advances_projections() -> Result<()> {
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_commit-1");
        let attempt_id = s.open_attempt(&task_id, 1).await;

        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task_id.clone(),
                turn_attempt_id: attempt_id.clone(),
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(5),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("commit")?;

        // Thread aggregate advanced.
        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.thread.total_usage, usage(100, 50));

        // Checkpoint created at turn 1.
        assert_eq!(outcome.checkpoint.turn_number, 1);
        assert_eq!(outcome.checkpoint.thread_id, thread_a());
        assert_eq!(outcome.checkpoint.task_id, task_id);
        assert_eq!(outcome.checkpoint.messages.len(), 2);

        // Turn attempt closed.
        assert!(outcome.closed_attempt.is_closed());
        assert_eq!(
            outcome.closed_attempt.outcome,
            Some(TurnAttemptOutcome::Success),
        );

        // Message projection updated.
        let history = s
            .messages
            .get_history(&thread_a())
            .await
            .context("history")?;
        assert_eq!(history.len(), 2);

        Ok(())
    }

    // ── Multiple turns ───────────────────────────────────────────

    #[tokio::test]
    async fn multiple_commits_create_sequential_checkpoints() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_turn-1");
        let task2 = AgentTaskId::from_string("task_turn-2");
        let a1 = s.open_attempt(&task1, 1).await;
        let a2 = s.open_attempt(&task2, 1).await;

        // Turn 1
        let o1 = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task1,
                turn_attempt_id: a1,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 1")],
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("turn 1")?;

        // Turn 2
        let o2 = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task2,
                turn_attempt_id: a2,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 2")],
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({"turn": 2}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("turn 2")?;

        assert_eq!(o1.checkpoint.turn_number, 1);
        assert_eq!(o2.checkpoint.turn_number, 2);
        assert_eq!(o2.thread.committed_turns, 2);
        assert_eq!(o2.thread.total_usage, usage(300, 130));

        // Both checkpoints exist.
        let list = s.checkpoints.list_by_thread(&thread_a()).await?;
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].turn_number, 1);
        assert_eq!(list[1].turn_number, 2);

        // Messages accumulated.
        let history = s.messages.get_history(&thread_a()).await?;
        assert_eq!(history.len(), 2);

        // Checkpoints capture full history, not just the delta.
        assert_eq!(
            o1.checkpoint.messages.len(),
            1,
            "turn 1 checkpoint: 1 message"
        );
        assert_eq!(
            o2.checkpoint.messages.len(),
            2,
            "turn 2 checkpoint: full history"
        );

        Ok(())
    }

    // ── Thread isolation ─────────────────────────────────────────

    #[tokio::test]
    async fn commits_on_different_threads_are_isolated() -> Result<()> {
        let s = Stores::new();
        let task_a = AgentTaskId::from_string("task_a");
        let task_b = AgentTaskId::from_string("task_b");
        let a_a = s.open_attempt(&task_a, 1).await;
        let a_b = s.open_attempt(&task_b, 1).await;

        commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task_a,
                turn_attempt_id: a_a,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("thread a")?;

        commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_b(),
                task_id: task_b,
                turn_attempt_id: a_b,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("thread b")?;

        // Each thread has exactly 1 checkpoint.
        let a_list = s.checkpoints.list_by_thread(&thread_a()).await?;
        let b_list = s.checkpoints.list_by_thread(&thread_b()).await?;
        assert_eq!(a_list.len(), 1);
        assert_eq!(b_list.len(), 1);

        // Thread aggregates are independent.
        let a_thread = s.threads.get(&thread_a()).await?.unwrap();
        let b_thread = s.threads.get(&thread_b()).await?.unwrap();
        assert_eq!(a_thread.total_usage, usage(100, 50));
        assert_eq!(b_thread.total_usage, usage(200, 80));

        Ok(())
    }

    // ── Failure: attempt not found ───────────────────────────────

    #[tokio::test]
    async fn commit_fails_if_attempt_not_found() {
        let s = Stores::new();
        let err = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: AgentTaskId::from_string("task_x"),
                turn_attempt_id: TurnAttemptId::from_string("attempt_nonexistent"),
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .unwrap_err();

        assert!(
            err.to_string().contains("close turn attempt"),
            "expected attempt error, got: {err}",
        );

        // No projections advanced.
        assert!(s.threads.get(&thread_a()).await.unwrap().is_none());
        assert!(
            s.checkpoints
                .list_by_thread(&thread_a())
                .await
                .unwrap()
                .is_empty()
        );
    }

    // ── Failure: already-closed attempt ──────────────────────────

    #[tokio::test]
    async fn commit_fails_if_attempt_already_closed() -> Result<()> {
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_closed");
        let attempt_id = s.open_attempt(&task_id, 1).await;

        // Close the attempt manually first.
        s.attempts
            .close_attempt(&attempt_id, sample_close_params(), t_plus(1))
            .await
            .context("manual close")?;

        // Attempt to commit with the already-closed attempt.
        let err = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .unwrap_err();

        assert!(
            err.to_string().contains("close turn attempt"),
            "expected close error, got: {err}",
        );

        // No projections advanced.
        assert!(s.threads.get(&thread_a()).await.unwrap().is_none());
        Ok(())
    }

    // ── Failure: completed thread rejects further commits ────────

    #[tokio::test]
    async fn commit_fails_on_completed_thread() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_1");
        let task2 = AgentTaskId::from_string("task_2");
        let a1 = s.open_attempt(&task1, 1).await;
        let a2 = s.open_attempt(&task2, 1).await;

        // Commit first turn then mark thread completed.
        commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task1,
                turn_attempt_id: a1,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("first commit")?;

        s.threads
            .mark_completed(&thread_a(), t_plus(2))
            .await
            .context("complete")?;

        // Second commit should fail at thread aggregate step.
        let err = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id: task2,
                turn_attempt_id: a2,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(3),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .unwrap_err();

        assert!(
            err.to_string().contains("thread aggregate"),
            "expected thread error, got: {err}",
        );

        // Only 1 checkpoint exists (from the first commit).
        let list = s.checkpoints.list_by_thread(&thread_a()).await?;
        assert_eq!(list.len(), 1);
        Ok(())
    }

    // ── Draft cleanup ────────────────────────────────────────────

    #[tokio::test]
    async fn commit_clears_in_flight_draft() -> Result<()> {
        // Simulate a turn that suspended at one or more tool boundaries
        // (populating the message-projection draft) and then completed
        // text-only. The commit must wipe the draft so the next turn
        // doesn't see the in-flight messages duplicated against the
        // committed history.
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_draft_clear");
        let attempt_id = s.open_attempt(&task_id, 1).await;

        // Pre-populate a draft as the suspension paths would.
        s.messages
            .set_draft(
                &thread_a(),
                vec![
                    llm::Message::user("user prompt"),
                    llm::Message::assistant("calling tool"),
                ],
                t_plus(2),
            )
            .await
            .context("seed draft")?;
        let projection = s
            .messages
            .get(&thread_a())
            .await?
            .context("projection bootstrapped")?;
        assert!(projection.has_draft(), "draft must be present pre-commit");

        commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(5),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("commit")?;

        let projection_after = s
            .messages
            .get(&thread_a())
            .await?
            .context("projection still present after commit")?;
        assert!(
            !projection_after.has_draft(),
            "commit must clear the in-flight draft so the next turn starts clean"
        );
        // Committed history still has the turn's messages.
        assert_eq!(projection_after.message_count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn commit_succeeds_when_no_draft_was_set() -> Result<()> {
        // First-turn happy path: nothing ever called set_draft, the
        // projection row may not even exist yet. The commit's
        // clear_draft call must not error out — the store returns
        // `None` and the commit proceeds.
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_no_draft");
        let attempt_id = s.open_attempt(&task_id, 1).await;

        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("commit without prior draft")?;

        assert_eq!(outcome.checkpoint.turn_number, 1);
        Ok(())
    }

    // ── No checkpoint for failed/cancelled turns ─────────────────

    #[tokio::test]
    async fn no_checkpoint_created_when_commit_not_called() -> Result<()> {
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_failed");

        // Open an attempt but never call commit_completed_turn.
        let _attempt_id = s.open_attempt(&task_id, 1).await;

        // No checkpoint exists.
        let list = s.checkpoints.list_by_thread(&thread_a()).await?;
        assert!(list.is_empty());

        // No thread aggregate exists.
        assert!(s.threads.get(&thread_a()).await?.is_none());
        Ok(())
    }

    // ── Snapshot data fidelity ───────────────────────────────────

    #[tokio::test]
    async fn checkpoint_preserves_agent_state_snapshot() -> Result<()> {
        let s = Stores::new();
        let task_id = AgentTaskId::from_string("task_snap");
        let attempt_id = s.open_attempt(&task_id, 1).await;

        let snapshot = serde_json::json!({
            "thread_id": "t-commit-a",
            "turn_count": 1,
            "metadata": { "model": "claude" },
        });

        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread_a(),
                task_id,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: snapshot.clone(),
                events: Vec::new(),
                outbox_max_attempts: 3,
                now: t_plus(5),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
            &s.events,
        )
        .await
        .context("commit")?;

        assert_eq!(outcome.checkpoint.agent_state_snapshot, snapshot);

        // Verify via store lookup too.
        let loaded = s
            .checkpoints
            .get_by_turn(&thread_a(), 1)
            .await?
            .context("not found")?;
        assert_eq!(loaded.agent_state_snapshot, snapshot);
        Ok(())
    }
}
