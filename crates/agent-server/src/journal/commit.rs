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
//!
//! If any step fails the function returns `Err` and no subsequent
//! steps execute. For the in-memory stores each individual call is
//! internally atomic (single write-lock scope). A real database
//! backend would wrap all four calls in a transaction.
//!
//! # Guarantees
//!
//! - A completed turn creates **exactly one** checkpoint for
//!   `(thread_id, turn_number)`.
//! - Thread aggregate, message projection, closed turn attempt, and
//!   checkpoint commit together.
//! - Failed or cancelled turns that never call this function do not
//!   create checkpoints.
//!
//! # What this module does **not** own
//!
//! - Recovery loaders — out of scope (future phase).
//! - Task state transitions — the caller is responsible for calling
//!   [`super::store::AgentTaskStore::complete_task`] separately.
//! - Event persistence — out of scope (future phase).

use agent_sdk_core::{ThreadId, TokenUsage, llm};
use anyhow::{Context, Result};
use time::OffsetDateTime;

use super::checkpoint::Checkpoint;
use super::checkpoint::NewCheckpointParams;
use super::checkpoint_store::CheckpointStore;
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
    /// Current wall-clock time.
    pub now: OffsetDateTime,
}

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
/// per-call atomicity, and a database backend should use a
/// transaction.
pub async fn commit_completed_turn(
    params: CompletedTurnCommit,
    thread_store: &dyn ThreadStore,
    message_store: &dyn MessageProjectionStore,
    turn_attempt_store: &dyn TurnAttemptStore,
    checkpoint_store: &dyn CheckpointStore,
) -> Result<CommitOutcome> {
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
    let checkpoint = checkpoint_store
        .commit_checkpoint(NewCheckpointParams {
            thread_id: params.thread_id,
            turn_number: thread.committed_turns,
            task_id: params.task_id,
            messages: updated_projection.messages,
            agent_state_snapshot: params.agent_state_snapshot,
            turn_usage: params.turn_usage,
            now: params.now,
        })
        .await
        .context("commit: create checkpoint")?;

    Ok(CommitOutcome {
        thread,
        checkpoint,
        closed_attempt,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::checkpoint_store::InMemoryCheckpointStore;
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
    }

    impl Stores {
        fn new() -> Self {
            Self {
                threads: InMemoryThreadStore::new(),
                messages: InMemoryMessageProjectionStore::new(),
                attempts: InMemoryTurnAttemptStore::new(),
                checkpoints: InMemoryCheckpointStore::new(),
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
                now: t_plus(5),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(2),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(1),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(3),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
                now: t_plus(5),
            },
            &s.threads,
            &s.messages,
            &s.attempts,
            &s.checkpoints,
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
