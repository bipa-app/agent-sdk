//! Thread-scoped checkpoint recovery and rebuild API.
//!
//! Once a thread has committed turns, recovery must not rebuild from
//! the thread start or from partially written projections. It must
//! load the latest completed checkpoint and continue from there.
//!
//! [`recover_thread`] is the single entry point. Given a thread id
//! it:
//!
//! 1. Loads the thread aggregate (or bootstraps a fresh one).
//! 2. Rejects completed threads — no new turns can be committed.
//! 3. If the thread has committed turns, loads the latest completed
//!    checkpoint and validates that its `turn_number` matches
//!    `thread.committed_turns`.
//! 4. Loads any in-flight draft from the message projection so a
//!    turn that suspended at a tool boundary and then failed
//!    mid-stream surfaces its accumulated conversation through the
//!    next-turn view instead of vanishing.
//! 5. Returns a [`ThreadRecoveryView`] containing the committed
//!    message history followed by any in-flight draft messages,
//!    agent-state snapshot, and next turn number — everything a
//!    caller needs to resume the conversation.
//!
//! # Design properties
//!
//! - **Checkpoint-anchored recovery** — the committed-history portion
//!   of the view is built exclusively from the latest completed
//!   checkpoint. Failed or in-progress attempts that never called
//!   [`super::commit::commit_completed_turn`] do not advance the
//!   committed state.
//! - **Draft awareness** — when a turn fails after one or more
//!   suspension boundaries, the draft slot on the message projection
//!   carries the in-flight `suspended_messages`. The recovery view
//!   appends them to the committed history so the next turn picks up
//!   the work the failed turn already did. The draft is cleared on
//!   the next successful commit.
//! - **Sequential root-task continuity** — checkpoints record which
//!   task produced them, but the recovery view is task-agnostic. A
//!   new root task on the same thread picks up from the last
//!   committed turn regardless of which task committed it.
//! - **Consistency invariant** — if the thread has `committed_turns >
//!   0`, the latest checkpoint's `turn_number` must equal
//!   `committed_turns`. A mismatch is a journal-level data corruption
//!   and returns an error rather than silently serving stale state.
//!
//! # What this module does **not** own
//!
//! - Event replay — out of scope.
//! - Task-level recovery (retry budget, lease expiry) — see
//!   [`super::recovery`].
//! - Context compaction — a future concern; recovery serves the raw
//!   checkpoint history.
//! - Dangling tool-use repair — when the draft ends in an assistant
//!   `tool_use` block whose results never landed (the failure
//!   happened during the resume LLM call), the upstream caller is
//!   responsible for synthesising failed tool-result blocks before
//!   submitting the history to a model. The journal preserves the
//!   raw draft as captured at the last suspension.

use agent_sdk_foundation::{ThreadId, llm};
use anyhow::{Context, Result, bail};
use time::OffsetDateTime;

use super::checkpoint::Checkpoint;
use super::checkpoint_store::CheckpointStore;
use super::message_store::MessageProjectionStore;
use super::thread::Thread;
use super::thread_store::ThreadStore;

// ─────────────────────────────────────────────────────────────────────
// Recovery view
// ─────────────────────────────────────────────────────────────────────

/// The next-turn view rebuilt from the latest completed checkpoint
/// plus any in-flight draft.
///
/// This is the single output type that a caller (future agent loop,
/// transport layer, etc.) needs to resume a conversation on a thread.
///
/// # Fields
///
/// | Group | Fields |
/// |-------|--------|
/// | Aggregate | `thread` |
/// | Snapshot | `messages`, `agent_state_snapshot` |
/// | Cursor | `next_turn_number` |
/// | Source | `latest_checkpoint`, `draft_messages` |
#[derive(Clone, Debug)]
pub struct ThreadRecoveryView {
    /// The thread aggregate at recovery time.
    pub thread: Thread,

    /// Full message history available to the next turn.
    ///
    /// Layout: `[committed history from latest checkpoint] + [draft
    /// messages captured at the most recent suspension]`. Either
    /// half can be empty:
    ///
    /// - A fresh thread (no committed turns, no in-flight turn) has
    ///   an empty list.
    /// - A thread with only committed turns has just the checkpoint
    ///   history.
    /// - A thread that failed mid-turn has the committed history
    ///   followed by the draft snapshot from the last suspension —
    ///   exactly what the failed turn was working with up to its
    ///   last tool boundary.
    pub messages: Vec<llm::Message>,

    /// Opaque agent-state blob from the latest checkpoint.
    ///
    /// `serde_json::Value::Null` for a fresh thread.
    pub agent_state_snapshot: serde_json::Value,

    /// The checkpoint the view was built from, or `None` if the
    /// thread has no committed turns.
    pub latest_checkpoint: Option<Checkpoint>,

    /// In-flight draft messages folded into [`Self::messages`], or
    /// an empty vec when no draft is present.
    ///
    /// Exposed separately so callers that need to distinguish "the
    /// committed conversation" from "what an interrupted turn left
    /// behind" can do so without diffing against the checkpoint —
    /// for example, to render a "previous attempt was interrupted
    /// — continuing" notice.
    pub draft_messages: Vec<llm::Message>,

    /// The committed conversation head WITHOUT the in-flight draft —
    /// i.e. [`Self::messages`] minus [`Self::draft_messages`].
    ///
    /// Sourced from the message projection (post-compaction when a
    /// mid-turn `replace_history` ran), falling back to the
    /// checkpoint's frozen snapshot only when the projection row is
    /// absent. This is the compaction-durable seed for resuming an
    /// already-suspended root task: the resume path re-supplies the
    /// suspended draft + child results itself, so the seed must
    /// exclude the draft — but it must NOT revert to the checkpoint's
    /// pre-compaction `messages`, or every resume re-compacts the same
    /// history. See [`super::staged::StagedStores::from_recovery_view_committed_only`].
    pub committed_messages: Vec<llm::Message>,

    /// The turn number the next attempt should target.
    ///
    /// Equal to `thread.committed_turns + 1` (1 for a fresh thread).
    pub next_turn_number: u32,
}

// ─────────────────────────────────────────────────────────────────────
// Recovery loader
// ─────────────────────────────────────────────────────────────────────

/// Recover the next-turn view for `thread_id` from the latest
/// completed checkpoint and any in-flight draft.
///
/// This is the primary entry point for Phase 3.5 thread recovery.
///
/// # Fresh threads
///
/// If the thread has zero committed turns AND no draft, the function
/// returns a view with empty messages, a null agent-state snapshot,
/// and `next_turn_number = 1`. A draft on a fresh thread (rare but
/// possible: the very first turn suspended at a tool boundary and
/// then failed before any commit) still surfaces in `messages` so
/// the next turn picks up the in-flight conversation.
///
/// # Consistency guard
///
/// If the thread has `committed_turns > 0` the function loads the
/// latest checkpoint and verifies that its `turn_number` equals
/// `committed_turns`. A mismatch indicates journal-level data
/// corruption and returns an error.
///
/// # Errors
///
/// - Store-level read errors.
/// - Thread is already completed.
/// - Missing checkpoint for a thread with committed turns.
/// - Checkpoint turn number ≠ `thread.committed_turns`.
pub async fn recover_thread(
    thread_id: &ThreadId,
    thread_store: &dyn ThreadStore,
    checkpoint_store: &dyn CheckpointStore,
    message_store: &dyn MessageProjectionStore,
    now: OffsetDateTime,
) -> Result<ThreadRecoveryView> {
    // 1. Load or bootstrap the thread aggregate.
    let thread = thread_store
        .get_or_create(thread_id, now)
        .await
        .context("recover: load thread aggregate")?;

    // 2. Completed thread — no new turns can be committed.
    if thread.status.is_completed() {
        bail!("recover: thread {thread_id} is already completed, no new turns can be committed");
    }

    // 3. Load the in-flight draft AND the committed projection head.
    //    The projection row may not exist yet (very first turn never
    //    wrote anything), in which case both are empty.
    //
    //    Reading the committed history straight from the projection
    //    (rather than the checkpoint's frozen `messages` snapshot)
    //    is what makes auto-compaction durable across attempts: when
    //    the worker rewrites the projection via
    //    `MessageProjectionStore::replace_history` mid-turn (a
    //    non-commit mutation), the checkpoint is still at the
    //    pre-compaction snapshot, but the projection holds the
    //    canonical compacted head. Recovering from the projection
    //    means a fresh attempt after lease expiry — or a brand-new
    //    turn after a failed-but-already-compacted prior attempt —
    //    picks up the compacted state instead of replaying the
    //    pre-compaction history that just blew the context window.
    //    The checkpoint is still load-bearing for the agent-state
    //    snapshot and the turn_number consistency guard, but the
    //    `messages` field is now treated as a snapshot artifact, not
    //    the recovery source of truth.
    let (committed_messages, draft_messages) = match message_store
        .get(thread_id)
        .await
        .context("recover: load message projection")?
    {
        Some(projection) => (projection.messages, projection.draft_messages),
        None => (Vec::new(), Vec::new()),
    };

    // 4. Fresh thread — no checkpoints to load. The draft may still
    //    be populated if the very first turn suspended and then
    //    failed before any commit, so it's included in `messages`
    //    even on the no-checkpoint path.
    if thread.committed_turns == 0 {
        return Ok(ThreadRecoveryView {
            thread,
            messages: draft_messages.clone(),
            agent_state_snapshot: serde_json::Value::Null,
            latest_checkpoint: None,
            // No committed turns yet — the committed head is empty; the
            // draft (if any) lives only in `messages`/`draft_messages`.
            committed_messages: Vec::new(),
            draft_messages,
            next_turn_number: 1,
        });
    }

    // 5. Load the latest completed checkpoint.
    let checkpoint = checkpoint_store
        .get_latest_by_thread(thread_id)
        .await
        .context("recover: load latest checkpoint")?;

    let Some(checkpoint) = checkpoint else {
        bail!(
            "recover: thread {} has {} committed turns but no checkpoint",
            thread_id,
            thread.committed_turns,
        );
    };

    // 6. Consistency guard: checkpoint turn_number must match the
    //    thread's committed_turns.
    if checkpoint.turn_number != thread.committed_turns {
        bail!(
            "recover: checkpoint turn_number ({}) != thread.committed_turns ({}) for thread {}",
            checkpoint.turn_number,
            thread.committed_turns,
            thread_id,
        );
    }

    // 7. Build the merged message list. Source of truth: the
    //    projection's committed `messages` (pre-compaction OR post-
    //    compaction, whichever is current), followed by the
    //    in-flight draft. Falls back to the checkpoint's frozen
    //    snapshot only when the projection row is missing — which
    //    shouldn't happen for a thread with `committed_turns > 0`
    //    (the commit path always writes the projection before the
    //    checkpoint), but keeps recovery robust against rare
    //    storage divergence.
    let next_turn = thread.committed_turns + 1;
    // Committed head: the projection's current `messages` (post-
    // compaction when a mid-turn `replace_history` ran), or the
    // checkpoint's frozen snapshot only when the projection row is
    // missing. Captured before folding in the draft so it can seed
    // resume attempts without the draft (see `committed_messages`).
    let committed_head = if committed_messages.is_empty() {
        checkpoint.messages.clone()
    } else {
        committed_messages
    };
    let mut messages = committed_head.clone();
    messages.extend(draft_messages.iter().cloned());
    Ok(ThreadRecoveryView {
        thread,
        messages,
        agent_state_snapshot: checkpoint.agent_state_snapshot.clone(),
        latest_checkpoint: Some(checkpoint),
        committed_messages: committed_head,
        draft_messages,
        next_turn_number: next_turn,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::checkpoint_store::InMemoryCheckpointStore;
    use super::super::commit::{CompletedTurnCommit, commit_completed_turn};
    use super::super::event_repository::InMemoryEventRepository;
    use super::super::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
    use super::super::task::AgentTaskId;
    use super::super::thread_store::InMemoryThreadStore;
    use super::super::turn_attempt::{OpenAttemptParams, TurnAttemptOutcome};
    use super::super::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
    use super::*;
    use crate::journal::checkpoint::CheckpointKind;
    use agent_sdk_foundation::TokenUsage;
    use agent_sdk_foundation::audit::AuditProvenance;
    use anyhow::Context;
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-recover-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-recover-b")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        }
    }

    fn sample_close_params() -> super::super::turn_attempt::CloseAttemptParams {
        super::super::turn_attempt::CloseAttemptParams {
            response_blob: serde_json::json!({
                "id": "msg_01",
                "content": [{"type": "text", "text": "hi"}]
            }),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_mode: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
        }
    }

    struct Stores {
        tasks: crate::journal::store::InMemoryAgentTaskStore,
        threads: InMemoryThreadStore,
        messages: InMemoryMessageProjectionStore,
        attempts: InMemoryTurnAttemptStore,
        checkpoints: InMemoryCheckpointStore,
        events: InMemoryEventRepository,
    }

    impl Stores {
        fn new() -> Self {
            Self {
                tasks: crate::journal::store::InMemoryAgentTaskStore::new(),
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
        ) -> Result<super::super::turn_attempt::TurnAttemptId> {
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

        /// Commit a turn end-to-end through the atomic commit path.
        async fn commit_turn(
            &self,
            thread_id: &ThreadId,
            task_id: &AgentTaskId,
            messages: Vec<llm::Message>,
            state_snapshot: serde_json::Value,
            at: OffsetDateTime,
        ) -> Result<super::super::commit::CommitOutcome> {
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
                    delivered_injection_ids: Vec::new(),
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
                &self.tasks,
                &self.threads,
                &self.messages,
                &self.attempts,
                &self.checkpoints,
                &self.events,
            )
            .await
        }
    }

    // ── Fresh thread ─────────────────────────────────────────────

    #[tokio::test]
    async fn recover_fresh_thread_returns_empty_view() -> Result<()> {
        let s = Stores::new();

        let view = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover")?;

        assert_eq!(view.thread.committed_turns, 0);
        assert!(view.messages.is_empty());
        assert_eq!(view.agent_state_snapshot, serde_json::Value::Null);
        assert!(view.latest_checkpoint.is_none());
        assert_eq!(view.next_turn_number, 1);
        Ok(())
    }

    // ── Single-turn recovery ─────────────────────────────────────

    #[tokio::test]
    async fn recover_after_one_committed_turn() -> Result<()> {
        let s = Stores::new();
        let task = AgentTaskId::from_string("task_t1");
        let msgs = vec![llm::Message::user("hello"), llm::Message::assistant("hi")];
        let snapshot = serde_json::json!({"model": "claude", "turn": 1});

        s.commit_turn(
            &thread_a(),
            &task,
            msgs.clone(),
            snapshot.clone(),
            t_plus(1),
        )
        .await
        .context("commit")?;

        let view = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover")?;

        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.messages.len(), 2);
        assert_eq!(view.agent_state_snapshot, snapshot);
        assert_eq!(view.next_turn_number, 2);

        let ckpt = view
            .latest_checkpoint
            .as_ref()
            .context("checkpoint should be present")?;
        assert_eq!(ckpt.turn_number, 1);
        assert_eq!(ckpt.task_id, task);
        Ok(())
    }

    // ── Multi-turn recovery uses latest checkpoint ───────────────

    #[tokio::test]
    async fn recover_after_multiple_turns_uses_latest() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_t1");
        let task2 = AgentTaskId::from_string("task_t2");
        let task3 = AgentTaskId::from_string("task_t3");

        s.commit_turn(
            &thread_a(),
            &task1,
            vec![llm::Message::user("turn 1")],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        s.commit_turn(
            &thread_a(),
            &task2,
            vec![llm::Message::assistant("turn 2")],
            serde_json::json!({"turn": 2}),
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        s.commit_turn(
            &thread_a(),
            &task3,
            vec![llm::Message::user("turn 3")],
            serde_json::json!({"turn": 3}),
            t_plus(3),
        )
        .await
        .context("turn 3")?;

        let view = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover")?;

        assert_eq!(view.thread.committed_turns, 3);
        assert_eq!(view.next_turn_number, 4);

        // The latest checkpoint captures the full accumulated history.
        assert_eq!(view.messages.len(), 3);
        assert_eq!(view.agent_state_snapshot, serde_json::json!({"turn": 3}));

        let ckpt = view
            .latest_checkpoint
            .as_ref()
            .context("checkpoint should be present")?;
        assert_eq!(ckpt.turn_number, 3);
        assert_eq!(ckpt.task_id, task3);
        Ok(())
    }

    // ── Sequential root tasks share committed history ────────────

    #[tokio::test]
    async fn sequential_root_tasks_share_committed_history() -> Result<()> {
        let s = Stores::new();

        // Task A commits turns 1 and 2.
        let task_a = AgentTaskId::from_string("task_root-a");
        s.commit_turn(
            &thread_a(),
            &task_a,
            vec![llm::Message::user("A turn 1")],
            serde_json::json!({"task": "A", "turn": 1}),
            t_plus(1),
        )
        .await
        .context("A turn 1")?;

        s.commit_turn(
            &thread_a(),
            &task_a,
            vec![llm::Message::assistant("A turn 2")],
            serde_json::json!({"task": "A", "turn": 2}),
            t_plus(2),
        )
        .await
        .context("A turn 2")?;

        // Task B (a new root task on the same thread) recovers.
        let view = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover for B")?;

        // View reflects committed state from task A.
        assert_eq!(view.thread.committed_turns, 2);
        assert_eq!(view.next_turn_number, 3);
        assert_eq!(view.messages.len(), 2);
        assert_eq!(
            view.agent_state_snapshot,
            serde_json::json!({"task": "A", "turn": 2}),
        );

        // Task B can now commit turn 3.
        let task_b = AgentTaskId::from_string("task_root-b");
        s.commit_turn(
            &thread_a(),
            &task_b,
            vec![llm::Message::user("B turn 3")],
            serde_json::json!({"task": "B", "turn": 3}),
            t_plus(3),
        )
        .await
        .context("B turn 3")?;

        let view2 = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover after B")?;

        assert_eq!(view2.thread.committed_turns, 3);
        assert_eq!(view2.next_turn_number, 4);
        assert_eq!(view2.messages.len(), 3);

        let ckpt = view2
            .latest_checkpoint
            .as_ref()
            .context("checkpoint should be present")?;
        assert_eq!(ckpt.task_id, task_b);
        Ok(())
    }

    // ── Compaction durability ────────────────────────────────────

    /// After a mid-turn `replace_history` rewrites the projection head
    /// (auto-compaction), `recover_thread` must expose the COMPACTED
    /// projection as `committed_messages` — the resume seed — while
    /// `latest_checkpoint.messages` stays at the frozen pre-compaction
    /// snapshot. This is the invariant that stops a resumed turn from
    /// re-compacting the same over-threshold history every tool round.
    #[tokio::test]
    async fn recover_sources_committed_messages_from_projection_after_compaction() -> Result<()> {
        let s = Stores::new();
        let task = AgentTaskId::from_string("task_compact");

        s.commit_turn(
            &thread_a(),
            &task,
            vec![
                llm::Message::user("turn 1 question"),
                llm::Message::assistant("turn 1 answer"),
            ],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("commit turn 1")?;

        // Pre-compaction: projection and checkpoint agree (2 messages).
        let before = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover before compaction")?;
        assert_eq!(before.committed_messages.len(), 2);

        // Mid-turn auto-compaction rewrites the projection head only.
        let compacted = vec![llm::Message::user("[summary of turn 1]")];
        s.messages
            .replace_history(&thread_a(), compacted.clone(), t_plus(2))
            .await
            .context("replace_history (compaction)")?;

        let after = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover after compaction")?;

        // committed_messages follows the compacted projection...
        assert_eq!(after.committed_messages.len(), 1);
        assert_eq!(
            serde_json::to_value(&after.committed_messages)?,
            serde_json::to_value(&compacted)?,
        );
        // ...while the checkpoint still holds the frozen 2-message snapshot.
        let ckpt = after
            .latest_checkpoint
            .as_ref()
            .context("checkpoint present")?;
        assert_eq!(
            ckpt.messages.len(),
            2,
            "checkpoint keeps the pre-compaction snapshot",
        );

        Ok(())
    }

    // ── Failed attempts do not pollute ───────────────────────────

    #[tokio::test]
    async fn failed_attempt_does_not_pollute_recovery() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_good");
        let task_fail = AgentTaskId::from_string("task_fail");

        // Commit one good turn.
        s.commit_turn(
            &thread_a(),
            &task1,
            vec![llm::Message::user("good turn")],
            serde_json::json!({"good": true}),
            t_plus(1),
        )
        .await
        .context("good turn")?;

        // Open an attempt for task_fail but never commit it —
        // simulates a failed or cancelled turn.
        let _failed_attempt_id = s.open_attempt(&task_fail, 1).await?;

        // Recovery should still see only the good turn.
        let view = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .context("recover")?;

        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.next_turn_number, 2);
        assert_eq!(view.messages.len(), 1);
        assert_eq!(view.agent_state_snapshot, serde_json::json!({"good": true}));

        let ckpt = view
            .latest_checkpoint
            .as_ref()
            .context("checkpoint should be present")?;
        assert_eq!(ckpt.turn_number, 1);
        assert_eq!(ckpt.task_id, task1);
        Ok(())
    }

    // ── Thread isolation ─────────────────────────────────────────

    #[tokio::test]
    async fn recovery_is_isolated_across_threads() -> Result<()> {
        let s = Stores::new();
        let task_a = AgentTaskId::from_string("task_a");
        let task_b = AgentTaskId::from_string("task_b");

        // Thread A: 2 turns.
        s.commit_turn(
            &thread_a(),
            &task_a,
            vec![llm::Message::user("A-1")],
            serde_json::json!({"thread": "A", "turn": 1}),
            t_plus(1),
        )
        .await?;
        s.commit_turn(
            &thread_a(),
            &task_a,
            vec![llm::Message::assistant("A-2")],
            serde_json::json!({"thread": "A", "turn": 2}),
            t_plus(2),
        )
        .await?;

        // Thread B: 1 turn.
        s.commit_turn(
            &thread_b(),
            &task_b,
            vec![llm::Message::user("B-1")],
            serde_json::json!({"thread": "B", "turn": 1}),
            t_plus(3),
        )
        .await?;

        let view_a =
            recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0()).await?;
        let view_b =
            recover_thread(&thread_b(), &s.threads, &s.checkpoints, &s.messages, t0()).await?;

        assert_eq!(view_a.thread.committed_turns, 2);
        assert_eq!(view_a.next_turn_number, 3);
        assert_eq!(view_a.messages.len(), 2);

        assert_eq!(view_b.thread.committed_turns, 1);
        assert_eq!(view_b.next_turn_number, 2);
        assert_eq!(view_b.messages.len(), 1);
        Ok(())
    }

    // ── Completed thread guard ────────────────────────────────────

    #[tokio::test]
    async fn recover_rejects_completed_thread() -> Result<()> {
        let s = Stores::new();
        let task = AgentTaskId::from_string("task_done");

        s.commit_turn(
            &thread_a(),
            &task,
            vec![llm::Message::user("final turn")],
            serde_json::json!({"done": true}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        s.threads
            .mark_completed(&thread_a(), t_plus(2))
            .await
            .context("mark completed")?;

        let err = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("already completed"),
            "expected completed error, got: {err}",
        );
        Ok(())
    }

    // ── Consistency guard: missing checkpoint ────────────────────

    #[tokio::test]
    async fn recover_fails_if_checkpoint_missing_for_committed_thread() -> Result<()> {
        let s = Stores::new();

        // Manually advance the thread aggregate without creating a
        // checkpoint — simulates data corruption.
        s.threads
            .commit_turn(&thread_a(), 1, &usage(100, 50), t_plus(1))
            .await
            .context("advance")?;

        let err = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no checkpoint"),
            "expected missing checkpoint error, got: {err}",
        );
        Ok(())
    }

    // ── Consistency guard: turn number mismatch ──────────────────

    #[tokio::test]
    async fn recover_fails_on_turn_number_mismatch() -> Result<()> {
        let s = Stores::new();

        // Commit two turns through the normal path.
        let task = AgentTaskId::from_string("task_mismatch");
        s.commit_turn(
            &thread_a(),
            &task,
            vec![llm::Message::user("turn 1")],
            serde_json::json!({}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;
        s.commit_turn(
            &thread_a(),
            &task,
            vec![llm::Message::assistant("turn 2")],
            serde_json::json!({}),
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        // Now manually advance the thread aggregate a third time
        // without creating the matching checkpoint.
        s.threads
            .commit_turn(&thread_a(), 3, &usage(100, 50), t_plus(3))
            .await
            .context("manual advance")?;

        // Thread says 3 committed turns, but latest checkpoint is
        // at turn 2.
        let err = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("turn_number (2)")
                && err.to_string().contains("committed_turns (3)"),
            "expected mismatch error, got: {err}",
        );
        Ok(())
    }

    // ── Idempotent: recovering twice yields same view ────────────

    #[tokio::test]
    async fn recovery_is_idempotent() -> Result<()> {
        let s = Stores::new();
        let task = AgentTaskId::from_string("task_idem");

        s.commit_turn(
            &thread_a(),
            &task,
            vec![llm::Message::user("hello")],
            serde_json::json!({"k": "v"}),
            t_plus(1),
        )
        .await
        .context("commit")?;

        let v1 = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0()).await?;
        let v2 = recover_thread(&thread_a(), &s.threads, &s.checkpoints, &s.messages, t0()).await?;

        assert_eq!(v1.thread.committed_turns, v2.thread.committed_turns);
        assert_eq!(v1.next_turn_number, v2.next_turn_number);
        assert_eq!(v1.messages.len(), v2.messages.len());
        assert_eq!(v1.agent_state_snapshot, v2.agent_state_snapshot);

        let c1 = v1
            .latest_checkpoint
            .as_ref()
            .context("v1 checkpoint should be present")?;
        let c2 = v2
            .latest_checkpoint
            .as_ref()
            .context("v2 checkpoint should be present")?;
        assert_eq!(c1.id, c2.id);
        Ok(())
    }
}
