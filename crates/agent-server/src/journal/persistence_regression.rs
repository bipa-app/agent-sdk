//! Persistence regression suite — crash, compaction, and restart.
//!
//! Proves the persistence layer under the exact failure modes that
//! motivated the rewrite:
//!
//! 1. **Crash timing** — a crash at any point during the commit path
//!    must not leave projections ahead of the latest completed
//!    checkpoint. The atomic commit path is sequential: close attempt →
//!    advance thread → append messages → create checkpoint. If a crash
//!    interrupts any step, recovery must see a consistent view based on
//!    the last fully-committed turn.
//!
//! 2. **Compaction / `replace_history`** — context compaction rewrites
//!    the message projection without touching checkpoints. Recovery
//!    must still produce a coherent view. The checkpoint retains the
//!    raw snapshot; the message projection carries the compacted form.
//!
//! 3. **Restart invariants** — across simulated restart boundaries
//!    (fresh store instances loaded from the same durable state),
//!    thread/message/checkpoint invariants remain true.

#[cfg(test)]
mod tests {
    use crate::journal::checkpoint::CheckpointKind;
    use crate::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
    use crate::journal::commit::{CompletedTurnCommit, commit_completed_turn};
    use crate::journal::event_repository::InMemoryEventRepository;
    use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
    use crate::journal::task::AgentTaskId;
    use crate::journal::thread_recover::{ThreadRecoveryView, recover_thread};
    use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
    use crate::journal::turn_attempt::{
        CloseAttemptParams, OpenAttemptParams, TurnAttemptId, TurnAttemptOutcome,
    };
    use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    // ── Time helpers ────────────────────────────────────────────────

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        }
    }

    fn sample_close_params() -> CloseAttemptParams {
        CloseAttemptParams {
            response_blob: serde_json::json!({
                "id": "msg_01",
                "content": [{"type": "text", "text": "response"}]
            }),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
        }
    }

    // ── Shared test harness ─────────────────────────────────────────

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

        async fn open_attempt(
            &self,
            task_id: &AgentTaskId,
            attempt_number: u32,
        ) -> Result<TurnAttemptId> {
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
                .context("open attempt")?;
            Ok(attempt.id)
        }

        async fn commit_turn(
            &self,
            thread_id: &ThreadId,
            task_id: &AgentTaskId,
            messages: Vec<llm::Message>,
            state_snapshot: serde_json::Value,
            at: OffsetDateTime,
        ) -> Result<crate::journal::commit::CommitOutcome> {
            let attempt_id = self.open_attempt(task_id, 1).await?;
            let expected_turn = self
                .threads
                .get(thread_id)
                .await
                .context("read committed turns for expected-turn guard")?
                .map_or(0, |thread| thread.committed_turns)
                .saturating_add(1);
            commit_completed_turn(
                CompletedTurnCommit {
                    checkpoint_kind: CheckpointKind::FullTurn,
                    thread_id: thread_id.clone(),
                    task_id: task_id.clone(),
                    expected_turn,
                    turn_attempt_id: attempt_id,
                    close_attempt_params: sample_close_params(),
                    messages,
                    turn_usage: usage(100, 50),
                    agent_state_snapshot: state_snapshot,
                    events: Vec::new(),
                    outbox_max_attempts: 3,
                    owner_guard: None,
                    now: at,
                },
                &self.threads,
                &self.messages,
                &self.attempts,
                &self.checkpoints,
                &self.events,
            )
            .await
        }

        async fn recover(&self, thread_id: &ThreadId) -> Result<ThreadRecoveryView> {
            recover_thread(
                thread_id,
                &self.threads,
                &self.checkpoints,
                &self.messages,
                t0(),
            )
            .await
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 1. CRASH-TIMING REGRESSIONS
    // ════════════════════════════════════════════════════════════════
    //
    // These tests simulate partial failures during the commit path and
    // verify that recovery never sees projections ahead of checkpoints.

    /// Crash before commit: open attempt + thread advance but no
    /// checkpoint. Recovery must see only the prior committed state.
    #[tokio::test]
    async fn crash_after_thread_advance_without_checkpoint_shows_prior_state() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-crash-timing-1");
        let task1 = AgentTaskId::from_string("task_good");
        let _task2 = AgentTaskId::from_string("task_crash");

        // Commit turn 1 through the normal path.
        s.commit_turn(
            &thread_id,
            &task1,
            vec![llm::Message::user("turn 1")],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        // Simulate a partial crash: advance the thread aggregate for
        // turn 2, but never create the matching checkpoint.
        // This simulates crashing between step 2 (thread advance)
        // and step 4 (checkpoint creation) of commit_completed_turn.
        s.threads
            .commit_turn(&thread_id, &usage(200, 80), t_plus(2))
            .await
            .context("manual advance")?;

        // Recovery must fail with a corruption error because the
        // thread says 2 committed turns but the latest checkpoint
        // is at turn 1.
        let err = s.recover(&thread_id).await.unwrap_err();
        assert!(
            err.to_string().contains("turn_number (1)")
                && err.to_string().contains("committed_turns (2)"),
            "expected checkpoint/thread mismatch, got: {err}",
        );
        Ok(())
    }

    /// Crash before any step: open attempt but never commit.
    /// Recovery sees a completely clean slate for a fresh thread.
    #[tokio::test]
    async fn crash_before_commit_on_fresh_thread_leaves_clean_state() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-crash-fresh");
        let task = AgentTaskId::from_string("task_crashed");

        // Open attempt but never call commit_completed_turn.
        let _attempt_id = s.open_attempt(&task, 1).await?;

        // Recovery: fresh thread with no committed state.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 0);
        assert!(view.messages.is_empty());
        assert_eq!(view.agent_state_snapshot, serde_json::Value::Null);
        assert!(view.latest_checkpoint.is_none());
        assert_eq!(view.next_turn_number, 1);

        // No checkpoint or message projection exists.
        let checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert!(checkpoints.is_empty());
        let history = s.messages.get_history(&thread_id).await?;
        assert!(history.is_empty());
        Ok(())
    }

    /// Crash after one successful commit, before the second commit
    /// completes. Recovery must show exactly the first committed turn.
    #[tokio::test]
    async fn crash_mid_second_turn_recovers_to_first_committed_turn() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-crash-mid-2");
        let task1 = AgentTaskId::from_string("task_turn-1");
        let task_fail = AgentTaskId::from_string("task_turn-2-fail");

        // Turn 1 commits successfully.
        s.commit_turn(
            &thread_id,
            &task1,
            vec![
                llm::Message::user("hello"),
                llm::Message::assistant("hi there"),
            ],
            serde_json::json!({"turn": 1, "model": "claude"}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        // Turn 2: open attempt but crash before commit.
        let _attempt_id = s.open_attempt(&task_fail, 1).await?;

        // Recovery sees exactly turn 1.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.next_turn_number, 2);
        assert_eq!(view.messages.len(), 2);
        assert_eq!(
            view.agent_state_snapshot,
            serde_json::json!({"turn": 1, "model": "claude"}),
        );

        let ckpt = view
            .latest_checkpoint
            .as_ref()
            .context("checkpoint expected")?;
        assert_eq!(ckpt.turn_number, 1);
        assert_eq!(ckpt.task_id, task1);
        Ok(())
    }

    /// After a commit succeeds, a subsequent failed attempt does not
    /// advance any projection. Message history and checkpoint stay
    /// consistent.
    #[tokio::test]
    async fn failed_attempt_after_commit_does_not_advance_projections() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-fail-no-advance");
        let task = AgentTaskId::from_string("task_good");
        let task_fail = AgentTaskId::from_string("task_fail");

        // Commit turn 1.
        s.commit_turn(
            &thread_id,
            &task,
            vec![llm::Message::user("committed")],
            serde_json::json!({"committed": true}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        // Failed turn 2: open attempt, close with failure outcome,
        // but never call commit_completed_turn.
        let attempt_id = s.open_attempt(&task_fail, 1).await?;
        s.attempts
            .close_attempt(
                &attempt_id,
                CloseAttemptParams {
                    response_blob: serde_json::json!({"error": "rate limited"}),
                    response_id: None,
                    response_model: None,
                    stop_reason: None,
                    outcome: TurnAttemptOutcome::RateLimited,
                    input_tokens: 50,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                },
                t_plus(2),
            )
            .await
            .context("close failed attempt")?;

        // Thread aggregate unchanged.
        let thread = s.threads.get(&thread_id).await?.context("thread missing")?;
        assert_eq!(thread.committed_turns, 1);

        // Message projection unchanged.
        let history = s.messages.get_history(&thread_id).await?;
        assert_eq!(history.len(), 1);

        // Only 1 checkpoint exists.
        let checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert_eq!(checkpoints.len(), 1);

        // Recovery consistent.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.next_turn_number, 2);
        Ok(())
    }

    /// A commit with an already-closed attempt fails at step 1 and
    /// no subsequent projections are touched.
    #[tokio::test]
    async fn double_close_attempt_aborts_entire_commit() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-double-close");
        let task = AgentTaskId::from_string("task_dc");
        let attempt_id = s.open_attempt(&task, 1).await?;

        // Close attempt outside the commit path.
        s.attempts
            .close_attempt(&attempt_id, sample_close_params(), t_plus(1))
            .await
            .context("pre-close")?;

        // Attempt the full commit — should fail at step 1.
        let err = commit_completed_turn(
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_id.clone(),
                task_id: task,
                // Fresh thread; the commit aborts at the attempt-close
                // step before the guard, but 1 is the correct turn.
                expected_turn: 1,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("should not appear")],
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
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
            "expected attempt error, got: {err}",
        );

        // No projections advanced.
        assert!(s.threads.get(&thread_id).await?.is_none());
        assert!(s.messages.get(&thread_id).await?.is_none());
        assert!(s.checkpoints.list_by_thread(&thread_id).await?.is_empty());
        Ok(())
    }

    /// Multiple failed attempts between two successful commits do not
    /// pollute any projection. The commit path guarantees that only
    /// successful calls advance state.
    #[tokio::test]
    async fn multiple_failed_attempts_between_commits_are_invisible() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-multi-fail");
        let task1 = AgentTaskId::from_string("task_1");
        let task2 = AgentTaskId::from_string("task_2");
        let fail_a = AgentTaskId::from_string("task_fail_a");
        let fail_b = AgentTaskId::from_string("task_fail_b");
        let fail_c = AgentTaskId::from_string("task_fail_c");

        // Commit turn 1.
        s.commit_turn(
            &thread_id,
            &task1,
            vec![llm::Message::user("turn 1")],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        // Three failed attempts (opened but never committed).
        let _a = s.open_attempt(&fail_a, 1).await?;
        let _b = s.open_attempt(&fail_b, 1).await?;
        let _c = s.open_attempt(&fail_c, 1).await?;

        // Commit turn 2.
        s.commit_turn(
            &thread_id,
            &task2,
            vec![llm::Message::assistant("turn 2")],
            serde_json::json!({"turn": 2}),
            t_plus(5),
        )
        .await
        .context("turn 2")?;

        // Recovery sees exactly 2 committed turns.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 2);
        assert_eq!(view.next_turn_number, 3);
        assert_eq!(view.messages.len(), 2);
        assert_eq!(view.agent_state_snapshot, serde_json::json!({"turn": 2}));

        // Only 2 checkpoints exist.
        let checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].turn_number, 1);
        assert_eq!(checkpoints[1].turn_number, 2);
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════
    // 2. COMPACTION AND replace_history REGRESSIONS
    // ════════════════════════════════════════════════════════════════
    //
    // Verify that context compaction (replace_history) preserves
    // coherent checkpoint state and that recovery still works.

    /// `replace_history` changes the message projection but does not
    /// affect the checkpoint. The checkpoint retains the raw history.
    #[tokio::test]
    async fn compaction_does_not_modify_checkpoint_history() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-compact-ckpt");
        // Commit 3 turns to build up history.
        for i in 1..=3i64 {
            let tid = AgentTaskId::from_string(format!("task_t{i}"));
            s.commit_turn(
                &thread_id,
                &tid,
                vec![llm::Message::user(format!("msg {i}"))],
                serde_json::json!({"turn": i}),
                t_plus(i),
            )
            .await
            .context(format!("turn {i}"))?;
        }

        // Verify pre-compaction state.
        let pre_history = s.messages.get_history(&thread_id).await?;
        assert_eq!(pre_history.len(), 3);

        let pre_checkpoint = s
            .checkpoints
            .get_latest_by_thread(&thread_id)
            .await?
            .context("checkpoint")?;
        assert_eq!(pre_checkpoint.messages.len(), 3);

        // Run compaction: replace the 3-message history with a
        // 1-message summary.
        s.messages
            .replace_history(
                &thread_id,
                vec![llm::Message::user("[Summary of turns 1-3]")],
                t_plus(10),
            )
            .await
            .context("replace_history")?;

        // Message projection shows the compacted form.
        let post_history = s.messages.get_history(&thread_id).await?;
        assert_eq!(post_history.len(), 1);

        // Checkpoint is unchanged — it retains the full raw history.
        let post_checkpoint = s
            .checkpoints
            .get_latest_by_thread(&thread_id)
            .await?
            .context("checkpoint after compact")?;
        assert_eq!(post_checkpoint.id, pre_checkpoint.id);
        assert_eq!(post_checkpoint.messages.len(), 3);
        assert_eq!(post_checkpoint.turn_number, 3);
        Ok(())
    }

    /// Recovery after compaction returns the **compacted projection**,
    /// not the checkpoint's frozen pre-compaction snapshot.
    ///
    /// The previous contract — recovery reads checkpoint.messages —
    /// was the load-bearing bug behind the compaction-recovery
    /// regression: when the daemon
    /// worker rewrote the projection mid-turn (a non-commit mutation
    /// driven by auto-compaction) and the turn then failed before
    /// committing, the next attempt re-read the stale checkpoint
    /// snapshot and tripped the same context-window error that
    /// triggered the compaction. Switching recovery to read the
    /// projection makes mid-turn `replace_history` durable across
    /// attempts; the checkpoint remains the source of truth for the
    /// agent-state snapshot and the turn-number consistency guard.
    #[tokio::test]
    async fn recovery_after_compaction_returns_projection_history() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-recover-compact");
        let task = AgentTaskId::from_string("task_rc1");

        // Commit 2 turns.
        s.commit_turn(
            &thread_id,
            &task,
            vec![
                llm::Message::user("msg 1"),
                llm::Message::assistant("rsp 1"),
            ],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        let task2 = AgentTaskId::from_string("task_rc2");
        s.commit_turn(
            &thread_id,
            &task2,
            vec![
                llm::Message::user("msg 2"),
                llm::Message::assistant("rsp 2"),
            ],
            serde_json::json!({"turn": 2}),
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        // Compact the projection.
        s.messages
            .replace_history(
                &thread_id,
                vec![llm::Message::user("[Compacted summary]")],
                t_plus(10),
            )
            .await
            .context("compact")?;

        // Recovery now follows the projection — the compacted single
        // summary message — not the checkpoint's 4-message frozen
        // snapshot.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 2);
        assert_eq!(view.next_turn_number, 3);
        assert_eq!(view.messages.len(), 1, "expected compacted projection");
        assert!(matches!(
            &view.messages[0].content,
            llm::Content::Text(t) if t == "[Compacted summary]"
        ));
        // Agent-state snapshot still comes from the latest checkpoint.
        assert_eq!(view.agent_state_snapshot, serde_json::json!({"turn": 2}),);
        Ok(())
    }

    /// Committing a new turn after compaction uses the message
    /// projection (compacted form) and creates a checkpoint that
    /// reflects the compacted + new messages.
    #[tokio::test]
    async fn commit_after_compaction_builds_on_compacted_projection() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-compact-then-commit");

        // Commit 2 turns.
        let task1 = AgentTaskId::from_string("task_ct1");
        s.commit_turn(
            &thread_id,
            &task1,
            vec![
                llm::Message::user("old 1"),
                llm::Message::assistant("old 2"),
            ],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        let task2 = AgentTaskId::from_string("task_ct2");
        s.commit_turn(
            &thread_id,
            &task2,
            vec![llm::Message::user("old 3")],
            serde_json::json!({"turn": 2}),
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        // Compact: replace 3 messages with 1 summary.
        s.messages
            .replace_history(&thread_id, vec![llm::Message::user("[Summary]")], t_plus(5))
            .await
            .context("compact")?;

        // Commit turn 3 on top of compacted history.
        let task3 = AgentTaskId::from_string("task_ct3");
        s.commit_turn(
            &thread_id,
            &task3,
            vec![llm::Message::assistant("new response")],
            serde_json::json!({"turn": 3}),
            t_plus(6),
        )
        .await
        .context("turn 3")?;

        // The message projection now has summary + new message.
        let history = s.messages.get_history(&thread_id).await?;
        assert_eq!(history.len(), 2);

        // Turn 3 checkpoint captures the current projection state
        // (compacted + appended).
        let ckpt3 = s
            .checkpoints
            .get_by_turn(&thread_id, 3)
            .await?
            .context("checkpoint 3")?;
        assert_eq!(ckpt3.messages.len(), 2);

        // Recovery from turn 3 gets the compacted + new form.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.committed_turns, 3);
        assert_eq!(view.messages.len(), 2);
        assert_eq!(view.next_turn_number, 4);

        // Earlier checkpoints are unmodified.
        let ckpt1 = s
            .checkpoints
            .get_by_turn(&thread_id, 1)
            .await?
            .context("checkpoint 1")?;
        assert_eq!(ckpt1.messages.len(), 2, "turn 1 checkpoint untouched");

        let ckpt2 = s
            .checkpoints
            .get_by_turn(&thread_id, 2)
            .await?
            .context("checkpoint 2")?;
        assert_eq!(ckpt2.messages.len(), 3, "turn 2 checkpoint untouched");
        Ok(())
    }

    /// Clearing the message projection via `replace_history` with an
    /// empty vec does not affect thread aggregate or checkpoints.
    #[tokio::test]
    async fn empty_compaction_preserves_thread_and_checkpoint_state() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-empty-compact");
        let task = AgentTaskId::from_string("task_ec");

        s.commit_turn(
            &thread_id,
            &task,
            vec![llm::Message::user("data")],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        // Clear messages entirely.
        s.messages
            .replace_history(&thread_id, vec![], t_plus(5))
            .await
            .context("clear")?;

        // Thread unaffected.
        let thread = s.threads.get(&thread_id).await?.context("thread")?;
        assert_eq!(thread.committed_turns, 1);

        // Checkpoint unaffected.
        let ckpt = s
            .checkpoints
            .get_latest_by_thread(&thread_id)
            .await?
            .context("ckpt")?;
        assert_eq!(ckpt.messages.len(), 1);

        // Recovery still works from checkpoint.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.messages.len(), 1, "recovery reads from checkpoint");
        Ok(())
    }

    /// Compacting the projection for one thread leaves other threads
    /// untouched.
    #[tokio::test]
    async fn compaction_is_thread_isolated() -> Result<()> {
        let s = Stores::new();
        let thread_a = ThreadId::from_string("t-compact-a");
        let thread_b = ThreadId::from_string("t-compact-b");

        let task_a = AgentTaskId::from_string("task_a");
        let task_b = AgentTaskId::from_string("task_b");

        s.commit_turn(
            &thread_a,
            &task_a,
            vec![llm::Message::user("A msg")],
            serde_json::json!({"thread": "A"}),
            t_plus(1),
        )
        .await
        .context("thread A commit")?;

        s.commit_turn(
            &thread_b,
            &task_b,
            vec![llm::Message::user("B msg 1"), llm::Message::user("B msg 2")],
            serde_json::json!({"thread": "B"}),
            t_plus(2),
        )
        .await
        .context("thread B commit")?;

        // Compact only thread A.
        s.messages
            .replace_history(&thread_a, vec![], t_plus(10))
            .await
            .context("compact A")?;

        // Thread B projection unchanged.
        let b_history = s.messages.get_history(&thread_b).await?;
        assert_eq!(b_history.len(), 2);

        // Both threads recover correctly.
        let view_a = s.recover(&thread_a).await.context("recover A")?;
        let view_b = s.recover(&thread_b).await.context("recover B")?;

        assert_eq!(view_a.messages.len(), 1, "A: from checkpoint");
        assert_eq!(view_b.messages.len(), 2, "B: from checkpoint");
        Ok(())
    }

    /// The version counter on the message projection increments
    /// across compaction boundaries — it never resets.
    #[tokio::test]
    async fn compaction_increments_projection_version() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-compact-version");
        let task = AgentTaskId::from_string("task_v");

        s.commit_turn(
            &thread_id,
            &task,
            vec![llm::Message::user("m1")],
            serde_json::json!({}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        let pre = s.messages.get(&thread_id).await?.context("pre")?;
        let pre_version = pre.version;

        s.messages
            .replace_history(
                &thread_id,
                vec![llm::Message::user("[compacted]")],
                t_plus(5),
            )
            .await
            .context("compact")?;

        let post = s.messages.get(&thread_id).await?.context("post")?;
        assert!(
            post.version > pre_version,
            "version must increase: {} > {}",
            post.version,
            pre_version,
        );
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════
    // 3. THREAD CONTINUITY AND RESTART INVARIANT TESTS
    // ════════════════════════════════════════════════════════════════
    //
    // Verify that thread/message/checkpoint invariants hold across
    // restart boundaries (simulated by continuing to use the same
    // stores, which represents the durable state).

    /// A new root task on the same thread picks up from the last
    /// committed turn's checkpoint and continues sequentially.
    #[tokio::test]
    async fn sequential_root_tasks_continue_from_last_checkpoint() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-continuity");

        // Root task A commits turns 1-2.
        let task_a = AgentTaskId::from_string("task_root-a");
        s.commit_turn(
            &thread_id,
            &task_a,
            vec![llm::Message::user("A-1")],
            serde_json::json!({"task": "A", "turn": 1}),
            t_plus(1),
        )
        .await
        .context("A turn 1")?;
        s.commit_turn(
            &thread_id,
            &task_a,
            vec![llm::Message::assistant("A-2")],
            serde_json::json!({"task": "A", "turn": 2}),
            t_plus(2),
        )
        .await
        .context("A turn 2")?;

        // Simulate restart: root task B recovers and continues.
        let view_b = s.recover(&thread_id).await.context("recover for B")?;
        assert_eq!(view_b.thread.committed_turns, 2);
        assert_eq!(view_b.next_turn_number, 3);
        assert_eq!(view_b.messages.len(), 2);

        // Root task B commits turn 3.
        let task_b = AgentTaskId::from_string("task_root-b");
        s.commit_turn(
            &thread_id,
            &task_b,
            vec![llm::Message::user("B-3")],
            serde_json::json!({"task": "B", "turn": 3}),
            t_plus(3),
        )
        .await
        .context("B turn 3")?;

        // Simulate another restart: root task C recovers.
        let view_c = s.recover(&thread_id).await.context("recover for C")?;
        assert_eq!(view_c.thread.committed_turns, 3);
        assert_eq!(view_c.next_turn_number, 4);
        assert_eq!(view_c.messages.len(), 3);

        // The latest checkpoint is from task B.
        let ckpt = view_c
            .latest_checkpoint
            .as_ref()
            .context("checkpoint expected")?;
        assert_eq!(ckpt.task_id, task_b);
        assert_eq!(ckpt.turn_number, 3);
        Ok(())
    }

    /// Each checkpoint is a complete snapshot. Recovering from any
    /// checkpoint should produce a self-consistent view.
    #[tokio::test]
    async fn each_checkpoint_is_a_complete_snapshot() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-snapshots");

        for i in 1..=5u32 {
            let task = AgentTaskId::from_string(format!("task_turn-{i}"));
            s.commit_turn(
                &thread_id,
                &task,
                vec![llm::Message::user(format!("turn {i}"))],
                serde_json::json!({"turn": i, "complete_history_len": i}),
                t_plus(i64::from(i)),
            )
            .await
            .context(format!("turn {i}"))?;
        }

        // Verify each checkpoint carries the complete accumulated
        // history up to that turn.
        let checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert_eq!(checkpoints.len(), 5);

        for (expected_turn, ckpt) in (1u32..).zip(&checkpoints) {
            assert_eq!(ckpt.turn_number, expected_turn);
            assert_eq!(
                ckpt.messages.len(),
                expected_turn as usize,
                "checkpoint at turn {expected_turn} should have {expected_turn} messages",
            );
            assert_eq!(
                ckpt.agent_state_snapshot["complete_history_len"],
                serde_json::json!(expected_turn),
            );
        }
        Ok(())
    }

    /// Token usage accumulates correctly across restarts. Each commit
    /// adds to the thread's total, and recovery sees the sum.
    #[tokio::test]
    async fn token_usage_accumulates_across_restart_boundaries() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-usage-accum");

        // Task A: commit turn 1.
        let task_a = AgentTaskId::from_string("task_a");
        let outcome_1 = s
            .commit_turn(
                &thread_id,
                &task_a,
                vec![llm::Message::user("turn 1")],
                serde_json::json!({}),
                t_plus(1),
            )
            .await
            .context("turn 1")?;
        assert_eq!(outcome_1.thread.total_usage, usage(100, 50));

        // Simulate restart, recover state.
        let view = s.recover(&thread_id).await.context("recover")?;
        assert_eq!(view.thread.total_usage, usage(100, 50));

        // Task B: commit turn 2 (adds another 100/50).
        let task_b = AgentTaskId::from_string("task_b");
        let outcome_2 = s
            .commit_turn(
                &thread_id,
                &task_b,
                vec![llm::Message::assistant("turn 2")],
                serde_json::json!({}),
                t_plus(2),
            )
            .await
            .context("turn 2")?;
        assert_eq!(outcome_2.thread.total_usage, usage(200, 100));

        // Final recovery shows accumulated usage.
        let final_view = s.recover(&thread_id).await.context("final recover")?;
        assert_eq!(final_view.thread.total_usage, usage(200, 100));
        assert_eq!(final_view.thread.committed_turns, 2);
        Ok(())
    }

    /// Completing a thread after commits prevents further recovery.
    /// This is the terminal state for a conversation.
    #[tokio::test]
    async fn completed_thread_rejects_recovery_and_further_commits() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-completed-guard");
        let task = AgentTaskId::from_string("task_final");

        // Commit and mark completed.
        s.commit_turn(
            &thread_id,
            &task,
            vec![llm::Message::user("final")],
            serde_json::json!({"final": true}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        s.threads
            .mark_completed(&thread_id, t_plus(2))
            .await
            .context("complete")?;

        // Recovery rejects completed thread.
        let err = s.recover(&thread_id).await.unwrap_err();
        assert!(err.to_string().contains("already completed"));

        // Further commits also rejected.
        let task2 = AgentTaskId::from_string("task_too_late");
        let attempt_id = s.open_attempt(&task2, 1).await?;
        let commit_err = commit_completed_turn(
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_id.clone(),
                task_id: task2,
                // The thread already has 1 committed turn, so the next
                // turn is 2; the guard passes and the commit is rejected
                // at the (completed) thread-aggregate step instead.
                expected_turn: 2,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("rejected")],
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
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
            commit_err.to_string().contains("thread aggregate"),
            "expected thread rejection, got: {commit_err}",
        );

        // Checkpoint count unchanged.
        let checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert_eq!(checkpoints.len(), 1);
        Ok(())
    }

    /// Recovery is idempotent: calling it multiple times yields the
    /// same view without side effects.
    #[tokio::test]
    async fn recovery_is_idempotent_across_multiple_calls() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-idempotent");
        let task = AgentTaskId::from_string("task_idem");

        s.commit_turn(
            &thread_id,
            &task,
            vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
            serde_json::json!({"state": "stable"}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        let v1 = s.recover(&thread_id).await.context("recover 1")?;
        let v2 = s.recover(&thread_id).await.context("recover 2")?;
        let v3 = s.recover(&thread_id).await.context("recover 3")?;

        for (label, view) in [("v1", &v1), ("v2", &v2), ("v3", &v3)] {
            assert_eq!(view.thread.committed_turns, 1, "{label}");
            assert_eq!(view.next_turn_number, 2, "{label}");
            assert_eq!(view.messages.len(), 2, "{label}");
            assert_eq!(
                view.agent_state_snapshot,
                serde_json::json!({"state": "stable"}),
                "{label}",
            );
        }

        // All views reference the same checkpoint.
        assert_eq!(
            v1.latest_checkpoint.as_ref().map(|c| &c.id),
            v2.latest_checkpoint.as_ref().map(|c| &c.id),
        );
        assert_eq!(
            v2.latest_checkpoint.as_ref().map(|c| &c.id),
            v3.latest_checkpoint.as_ref().map(|c| &c.id),
        );
        Ok(())
    }

    /// Verify the full lifecycle: commit → recover → compact →
    /// commit → recover → complete. This exercises every major
    /// persistence operation in sequence.
    #[tokio::test]
    async fn full_lifecycle_commit_recover_compact_commit_recover_complete() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-lifecycle");

        // Phase 1: Initial turns.
        let task1 = AgentTaskId::from_string("task_lc1");
        s.commit_turn(
            &thread_id,
            &task1,
            vec![
                llm::Message::user("hello"),
                llm::Message::assistant("hi there"),
            ],
            serde_json::json!({"phase": 1, "turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        let task2 = AgentTaskId::from_string("task_lc2");
        s.commit_turn(
            &thread_id,
            &task2,
            vec![
                llm::Message::user("help me"),
                llm::Message::assistant("sure thing"),
            ],
            serde_json::json!({"phase": 1, "turn": 2}),
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        // Recover after phase 1.
        let view1 = s.recover(&thread_id).await.context("recover phase 1")?;
        assert_eq!(view1.thread.committed_turns, 2);
        assert_eq!(view1.messages.len(), 4);

        // Phase 2: Compaction.
        s.messages
            .replace_history(
                &thread_id,
                vec![llm::Message::user("[Summary of turns 1-2]")],
                t_plus(10),
            )
            .await
            .context("compact")?;

        // Recovery now follows the projection — see
        // `recovery_after_compaction_returns_projection_history`
        // for the full rationale.
        let view_post_compact = s
            .recover(&thread_id)
            .await
            .context("recover post compact")?;
        assert_eq!(view_post_compact.messages.len(), 1);

        // Phase 3: New commit on top of compacted projection.
        let task3 = AgentTaskId::from_string("task_lc3");
        s.commit_turn(
            &thread_id,
            &task3,
            vec![llm::Message::user("new question")],
            serde_json::json!({"phase": 3, "turn": 3}),
            t_plus(11),
        )
        .await
        .context("turn 3")?;

        // Recovery shows turn 3 checkpoint (built on compacted
        // projection).
        let view3 = s.recover(&thread_id).await.context("recover phase 3")?;
        assert_eq!(view3.thread.committed_turns, 3);
        assert_eq!(view3.next_turn_number, 4);
        // Turn 3 checkpoint has compacted summary + new message.
        assert_eq!(view3.messages.len(), 2);

        // Phase 4: Complete the thread.
        s.threads
            .mark_completed(&thread_id, t_plus(20))
            .await
            .context("complete")?;

        // No further recovery or commits allowed.
        let err = s.recover(&thread_id).await.unwrap_err();
        assert!(err.to_string().contains("already completed"));

        // All 3 checkpoints preserved.
        let all_checkpoints = s.checkpoints.list_by_thread(&thread_id).await?;
        assert_eq!(all_checkpoints.len(), 3);
        Ok(())
    }

    /// Two threads interleaving commits do not interfere with each
    /// other's projections, checkpoints, or recovery views.
    #[tokio::test]
    async fn interleaved_commits_across_threads_are_independent() -> Result<()> {
        let s = Stores::new();
        let thread_a = ThreadId::from_string("t-interleave-a");
        let thread_b = ThreadId::from_string("t-interleave-b");

        // Interleave: A-1, B-1, A-2, B-2.
        let ta1 = AgentTaskId::from_string("task_a1");
        s.commit_turn(
            &thread_a,
            &ta1,
            vec![llm::Message::user("A-1")],
            serde_json::json!({"thread": "A", "turn": 1}),
            t_plus(1),
        )
        .await
        .context("A-1")?;

        let tb1 = AgentTaskId::from_string("task_b1");
        s.commit_turn(
            &thread_b,
            &tb1,
            vec![llm::Message::user("B-1")],
            serde_json::json!({"thread": "B", "turn": 1}),
            t_plus(2),
        )
        .await
        .context("B-1")?;

        let ta2 = AgentTaskId::from_string("task_a2");
        s.commit_turn(
            &thread_a,
            &ta2,
            vec![llm::Message::assistant("A-2")],
            serde_json::json!({"thread": "A", "turn": 2}),
            t_plus(3),
        )
        .await
        .context("A-2")?;

        let tb2 = AgentTaskId::from_string("task_b2");
        s.commit_turn(
            &thread_b,
            &tb2,
            vec![llm::Message::assistant("B-2")],
            serde_json::json!({"thread": "B", "turn": 2}),
            t_plus(4),
        )
        .await
        .context("B-2")?;

        // Verify isolation.
        let view_a = s.recover(&thread_a).await.context("recover A")?;
        let view_b = s.recover(&thread_b).await.context("recover B")?;

        assert_eq!(view_a.thread.committed_turns, 2);
        assert_eq!(view_b.thread.committed_turns, 2);
        assert_eq!(view_a.messages.len(), 2);
        assert_eq!(view_b.messages.len(), 2);
        assert_eq!(
            view_a.agent_state_snapshot,
            serde_json::json!({"thread": "A", "turn": 2}),
        );
        assert_eq!(
            view_b.agent_state_snapshot,
            serde_json::json!({"thread": "B", "turn": 2}),
        );

        // Checkpoint isolation.
        let a_ckpts = s.checkpoints.list_by_thread(&thread_a).await?;
        let b_ckpts = s.checkpoints.list_by_thread(&thread_b).await?;
        assert_eq!(a_ckpts.len(), 2);
        assert_eq!(b_ckpts.len(), 2);

        // Usage isolation.
        assert_eq!(view_a.thread.total_usage, usage(200, 100));
        assert_eq!(view_b.thread.total_usage, usage(200, 100));
        Ok(())
    }

    /// The checkpoint `turn_number` must always equal the thread's
    /// `committed_turns` after each commit. This is the core consistency
    /// invariant that recovery depends on.
    #[tokio::test]
    async fn checkpoint_turn_equals_committed_turns_after_each_commit() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-invariant");

        for i in 1..=10u32 {
            let task = AgentTaskId::from_string(format!("task_{i}"));
            let outcome = s
                .commit_turn(
                    &thread_id,
                    &task,
                    vec![llm::Message::user(format!("msg {i}"))],
                    serde_json::json!({"turn": i}),
                    t_plus(i64::from(i)),
                )
                .await
                .context(format!("turn {i}"))?;

            // After each commit: checkpoint.turn_number ==
            // thread.committed_turns.
            assert_eq!(
                outcome.checkpoint.turn_number, outcome.thread.committed_turns,
                "invariant broken at turn {i}",
            );

            // Also verify via recovery.
            let view = s
                .recover(&thread_id)
                .await
                .context(format!("recover {i}"))?;
            assert_eq!(view.thread.committed_turns, i);
            assert_eq!(view.next_turn_number, i + 1);
            assert_eq!(
                view.latest_checkpoint
                    .as_ref()
                    .context("checkpoint")?
                    .turn_number,
                i,
            );
        }
        Ok(())
    }

    /// A thread that has had some turns committed but was never
    /// completed can be recovered indefinitely with new root tasks.
    #[tokio::test]
    async fn long_running_thread_supports_many_root_task_handoffs() -> Result<()> {
        let s = Stores::new();
        let thread_id = ThreadId::from_string("t-long-running");

        // Simulate 5 different root tasks each committing 1 turn.
        for i in 1..=5u32 {
            // Recover first (simulates restart).
            let view = s
                .recover(&thread_id)
                .await
                .context(format!("recover {i}"))?;
            assert_eq!(view.next_turn_number, i);

            // New root task commits next turn.
            let task = AgentTaskId::from_string(format!("task_root-{i}"));
            s.commit_turn(
                &thread_id,
                &task,
                vec![llm::Message::user(format!("root {i} speaking"))],
                serde_json::json!({"root_task": i}),
                t_plus(i64::from(i)),
            )
            .await
            .context(format!("commit {i}"))?;
        }

        // Final state after 5 handoffs.
        let final_view = s.recover(&thread_id).await.context("final")?;
        assert_eq!(final_view.thread.committed_turns, 5);
        assert_eq!(final_view.next_turn_number, 6);
        assert_eq!(final_view.messages.len(), 5);
        assert_eq!(
            final_view.agent_state_snapshot,
            serde_json::json!({"root_task": 5}),
        );

        // Latest checkpoint from the final root task.
        let ckpt = final_view
            .latest_checkpoint
            .as_ref()
            .context("checkpoint")?;
        assert_eq!(ckpt.task_id, AgentTaskId::from_string("task_root-5"));
        assert_eq!(ckpt.turn_number, 5);
        Ok(())
    }
}
