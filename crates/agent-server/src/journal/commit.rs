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
//! Every backend commits all of it, plus the delivered injections'
//! completion, the turn's lifecycle events and their coalesced
//! `thread_events_available` advisory, through the required
//! [`AgentTaskStore::commit_completed_turn`].
//! The durable backends run one database transaction, which also clears
//! the draft, so committed history and draft never both hold the same
//! messages. The in-memory task store holds its write lock across every
//! step.
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
//! # Event + state atomicity
//!
//! The lifecycle events **and** their single coalesced
//! `thread_events_available` advisory outbox row are written in the same
//! unit as the state projections, so "events exist iff the turn
//! committed" holds on every backend.
//!
//! # What this module does **not** own
//!
//! - Recovery loaders — see [`super::thread_recover`].
//! - Task state transitions — the caller is responsible for calling
//!   [`super::store::AgentTaskStore::complete_task`] separately.

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
use anyhow::Result;
use time::OffsetDateTime;

use super::checkpoint::Checkpoint;
use super::checkpoint::CheckpointKind;
use super::committed_event::CommittedEvent;
use super::document::{DocScope, DocumentWrite};
use super::effect::EffectWrite;
use super::store::AgentTaskStore;
use super::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use super::thread::Thread;
use super::transition::{Guard, Transition, Write};
use super::turn_attempt::{CloseAttemptParams, TurnAttempt, TurnAttemptId};

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
    /// The turn number this commit is expected to produce — i.e.
    /// `thread.committed_turns + 1` at the moment the worker decided to
    /// commit.
    ///
    /// Every committer (in-memory, Postgres, `SQLite`) asserts this
    /// equals `thread.committed_turns + 1` against the freshly
    /// locked/loaded thread row *before* applying the increment. A
    /// stale-lease worker that lost a race to another worker therefore
    /// fails the assertion instead of durably double-committing the
    /// turn (double-incrementing `committed_turns`, appending the same
    /// messages twice, and double-billing usage).
    pub expected_turn: u32,
    /// Boundary-injection rows this turn already delivered; completed
    /// Queued→Completed inside the same transaction as the turn commit,
    /// so "turn durably committed" and "delivered injections completed"
    /// are one atomic fact.
    pub delivered_injection_ids: Vec<AgentTaskId>,
    /// The open attempt to close.
    pub turn_attempt_id: TurnAttemptId,
    /// Parameters for closing the turn attempt.
    pub close_attempt_params: CloseAttemptParams,
    /// Messages to append to the message projection.
    pub messages: Vec<llm::Message>,
    /// Token usage for this turn.
    pub turn_usage: TokenUsage,
    /// Provenance recorded on the checkpoint row this commit creates.
    ///
    /// [`CheckpointKind::CancelSalvage`] is set **only** by the
    /// cancellation seam-B salvage commit; every full-turn commit path
    /// passes [`CheckpointKind::FullTurn`]. The worker's turn-slot-shift
    /// retry keys its shift-eligibility decision off this durable
    /// discriminator — never off token usage, which can legitimately be
    /// all-zero on a real completion.
    pub checkpoint_kind: CheckpointKind,
    /// Opaque agent-state snapshot for v1 recovery.
    pub agent_state_snapshot: serde_json::Value,
    /// Lifecycle events to commit with the turn, together with their single
    /// coalesced advisory outbox row, in the same unit as the state
    /// projections (attempt close, thread aggregate, message head, raw
    /// message batch, checkpoint). An empty vec writes no event and no
    /// advisory.
    pub events: Vec<AgentEvent>,
    /// Maximum relay attempts for the coalesced advisory outbox row
    /// written alongside `events`. Ignored when `events` is empty.
    pub outbox_max_attempts: u32,
    /// Task-row ownership validation performed **inside** the commit
    /// transaction.
    ///
    /// When set, the store re-reads the task row (`FOR UPDATE` on
    /// Postgres; under `SQLite`'s exclusive write transaction; under the
    /// in-memory task store's write lock) and rejects the commit with
    /// [`LostCommitOwnership`]
    /// unless the row is still `Running` and owned by exactly this
    /// `(worker, lease)` pair — closing the window where a
    /// cancellation or lease loss lands between a caller-side
    /// ownership check and the commit. The worker passes it on every root
    /// completion, the first commit and each turn-slot-shift retry alike,
    /// because the root's settle among [`Self::extra_writes`] needs it.
    pub owner_guard: Option<CommitOwnerGuard>,
    /// Further writes applied inside the completed turn's transaction,
    /// after its own rows and events: the task's own [`Write::SettleTask`],
    /// which needs [`Self::owner_guard`] so the turn and the task's terminal
    /// row commit together (design I2), and documents on the turn's own
    /// thread or task, which a host's
    /// [`super::extension::TransitionExtension`] adds (see
    /// [`Self::validate_extra_writes`]).
    pub extra_writes: Vec<Write>,
    /// Current wall-clock time.
    pub now: OffsetDateTime,
}

impl CompletedTurnCommit {
    /// Check the extra writes every backend applies inside the turn's
    /// transaction, only with an [`Self::owner_guard`] for its lease: the
    /// finishes of the task's own effects (its LLM request's), then at most
    /// one settle of the task itself; and documents, each scoped to the
    /// turn's own thread or its own task.
    ///
    /// The scope rule keeps the purge fence and the lock order without a
    /// lock pass of its own: every completed turn locks its thread row first
    /// and refuses a thread that is being purged, and a document on that
    /// thread or on its task is reached only under that lock, as a
    /// transition reaches it under `Lock::Thread` / `Lock::ThreadOf`.
    ///
    /// # Errors
    /// Names the first write a completed turn may not carry, the first
    /// document outside the turn's own scope, or the first malformed
    /// document.
    pub fn validate_extra_writes(&self) -> Result<()> {
        let mut settles = 0;
        for write in &self.extra_writes {
            match write {
                Write::SettleTask { task, .. } if *task == self.task_id => {
                    anyhow::ensure!(
                        self.owner_guard.is_some(),
                        "a completed turn settles its task {task} only under an owner guard"
                    );
                    settles += 1;
                }
                Write::Effect(EffectWrite::Finish { owner, .. }) if *owner == self.task_id => {
                    anyhow::ensure!(
                        self.owner_guard.is_some(),
                        "a completed turn finishes its task's effects only under an owner guard"
                    );
                    anyhow::ensure!(
                        settles == 0,
                        "a completed turn finishes its effects before it settles its task"
                    );
                }
                Write::Document(document) => self.check_own_document(document)?,
                Write::SettleTask { .. }
                | Write::RejectConfirmation { .. }
                | Write::PauseOnConfirmation { .. }
                | Write::ReparkAfterSteering { .. }
                | Write::Effect(_)
                | Write::AppendCompaction { .. }
                | Write::AppendEvents { .. }
                | Write::SpawnChildren { .. }
                | Write::EnqueueSteeringResume { .. }
                | Write::BeginEpoch(_)
                | Write::TurnAttempt(_) => {
                    anyhow::bail!(
                        "a completed turn may carry only its own task's effect finishes, \
                         settle and documents among its extra writes, not {write:?}"
                    );
                }
            }
        }
        anyhow::ensure!(
            settles <= 1,
            "a completed turn settles its task at most once"
        );
        self.extra_transition()
            .validate()
            .map_err(|rejected| anyhow::anyhow!("{rejected}"))?;
        Ok(())
    }

    /// Check that `document` is on the turn's own thread or its own task,
    /// and well-formed.
    ///
    /// # Errors
    /// Names the document when it is outside the turn's scope or malformed.
    pub(crate) fn check_own_document(&self, document: &DocumentWrite) -> Result<()> {
        let own = match &document.key.scope {
            DocScope::Thread(thread) => thread == &self.thread_id,
            DocScope::Task(task) => task == &self.task_id,
            DocScope::Host => false,
        };
        anyhow::ensure!(
            own,
            "a completed turn may write documents only on its own thread {} or task {}, not {}",
            self.thread_id,
            self.task_id,
            document.key
        );
        document.validate()
    }

    /// The extra writes as a transition for the backends' shared write
    /// arms, guarded by the owner's lease when the turn has an owner
    /// guard (which the commit itself checks first).
    #[must_use]
    pub fn extra_transition(&self) -> Transition {
        Transition {
            guards: self
                .owner_guard
                .iter()
                .map(|owner| Guard::TaskIs {
                    task: self.task_id.clone(),
                    status: TaskStatus::Running,
                    lease: Some((owner.worker_id.clone(), owner.lease_id.clone())),
                })
                .collect(),
            writes: self.extra_writes.clone(),
        }
    }
}

/// The `(worker, lease)` pair a guarded commit must still own — see
/// [`CompletedTurnCommit::owner_guard`].
#[derive(Clone, Debug)]
pub struct CommitOwnerGuard {
    /// Worker that believes it owns the task row.
    pub worker_id: WorkerId,
    /// Lease under which the worker acquired the row.
    pub lease_id: LeaseId,
}

/// Default relay attempt budget for the coalesced `thread_events_available`
/// advisory row written alongside a completed turn's lifecycle events.
///
/// Matches the long-standing 3-attempt default used elsewhere for the
/// event outbox; production callers may override per [`CompletedTurnCommit`].
pub const DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS: u32 = 3;

/// Typed root cause for the completed-turn slot CAS rejection.
///
/// Every committer — the in-memory guard in [`commit_completed_turn`],
/// the Postgres / `SQLite` atomic committers, and the worker's
/// pre-commit idempotency check — returns this as the error's root
/// cause when `expected_turn` no longer matches the thread's committed
/// turn count, so callers (e.g. the worker's turn-slot-shift path) can
/// `downcast_ref::<StaleTurnCommit>()` instead of matching message
/// strings. The `Display` text is kept byte-identical to the
/// historical message
/// (`"stale turn commit: expected {expected}, thread at {committed}"`)
/// for logs and existing assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaleTurnCommit {
    /// The turn number the rejected commit expected to produce.
    pub expected_turn: u32,
    /// The thread's committed turn count observed at rejection time.
    pub committed_turns: u32,
}

impl std::fmt::Display for StaleTurnCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stale turn commit: expected {}, thread at {}",
            self.expected_turn, self.committed_turns,
        )
    }
}

impl std::error::Error for StaleTurnCommit {}

/// Typed root cause for a rejected owner-guarded commit.
///
/// Returned when the task row is no longer a `Running` row owned by
/// the presenting `(worker, lease)` — see
/// [`CompletedTurnCommit::owner_guard`]. Deliberately distinct from
/// [`StaleTurnCommit`]: the slot-shift path retries on the latter but
/// must treat this as terminal (the cancel / timeout / requeue owner
/// has the row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LostCommitOwnership {
    /// The task whose ownership validation failed.
    pub task_id: AgentTaskId,
}

impl std::fmt::Display for LostCommitOwnership {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "commit ownership lost: task {} is no longer a Running row owned by the presenting lease",
            self.task_id,
        )
    }
}

impl std::error::Error for LostCommitOwnership {}

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
    /// The task and its parent (or the subagent invocation it resumed)
    /// after a [`Write::SettleTask`] among the extra writes, as
    /// [`super::Committed::settled`] reports it.
    pub settled: Option<(AgentTask, Option<AgentTask>)>,
}

// ─────────────────────────────────────────────────────────────────────
// Atomic commit
// ─────────────────────────────────────────────────────────────────────

/// Commit a completed turn across all projections through
/// [`AgentTaskStore::commit_completed_turn`], recording the commit's
/// duration.
///
/// # Errors
///
/// Whatever the store returns; see
/// [`AgentTaskStore::commit_completed_turn`].
pub async fn commit_completed_turn(
    params: CompletedTurnCommit,
    task_store: &dyn AgentTaskStore,
) -> Result<CommitOutcome> {
    #[cfg(feature = "otel")]
    let started_at = std::time::Instant::now();
    let task_id = params.task_id.clone();

    let result = task_store.commit_completed_turn(params).await;

    let outcome = commit_outcome_label(&result);
    #[cfg(feature = "otel")]
    {
        let metrics = crate::observability::ServerMetrics::global();
        metrics.record_completed_turn_commit(outcome);
        if result.is_ok() {
            metrics.record_journal_commit(
                crate::observability::attrs::COMMIT_KIND_ATOMIC,
                started_at.elapsed().as_secs_f64(),
            );
        }
    }
    if let Err(error) = &result
        && outcome != "error"
    {
        log::warn!(
            task_id = task_id.as_str(),
            outcome = outcome;
            "completed turn refused; the finished turn is discarded: {error:#}"
        );
    }
    result
}

/// The bounded outcome of a completed-turn commit: `ok`, `lost_ownership`
/// (the worker no longer owns the task), `stale_turn` (another commit took
/// the turn slot) or `error`.
#[must_use]
pub fn commit_outcome_label(result: &Result<CommitOutcome>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(error) if error.downcast_ref::<LostCommitOwnership>().is_some() => "lost_ownership",
        Err(error) if error.downcast_ref::<StaleTurnCommit>().is_some() => "stale_turn",
        Err(_) => "error",
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::checkpoint_store::CheckpointStore;
    use super::super::checkpoint_store::InMemoryCheckpointStore;
    use super::super::event_repository::InMemoryEventRepository;
    use super::super::message_store::InMemoryMessageProjectionStore;
    use super::super::message_store::MessageProjectionStore;
    use super::super::thread_store::InMemoryThreadStore;
    use super::super::thread_store::ThreadStore;
    use super::super::turn_attempt::{OpenAttemptParams, TurnAttemptOutcome};
    use super::super::turn_attempt_store::InMemoryTurnAttemptStore;
    use super::super::turn_attempt_store::TurnAttemptStore;
    use super::*;
    use agent_sdk_foundation::audit::AuditProvenance;
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
            stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
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
    }

    impl Stores {
        fn new() -> Self {
            let threads = InMemoryThreadStore::new();
            let messages = InMemoryMessageProjectionStore::new();
            let attempts = InMemoryTurnAttemptStore::new();
            let checkpoints = InMemoryCheckpointStore::new();
            let tasks = crate::journal::store::InMemoryAgentTaskStore::committing_turns_to(
                &threads,
                &messages,
                &attempts,
                &checkpoints,
                &InMemoryEventRepository::new(),
            );
            Self {
                tasks,
                threads,
                messages,
                attempts,
                checkpoints,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task_id.clone(),
                expected_turn: 1,
                turn_attempt_id: attempt_id.clone(),
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(5),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task1,
                expected_turn: 1,
                turn_attempt_id: a1,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 1")],
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
        )
        .await
        .context("turn 1")?;

        // Turn 2
        let o2 = commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task2,
                expected_turn: 2,
                turn_attempt_id: a2,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 2")],
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({"turn": 2}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(2),
            },
            &s.tasks,
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

    #[tokio::test]
    async fn checkpoint_and_fork_seed_follow_effective_view_after_compaction() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_compacted_turn-1");
        let task2 = AgentTaskId::from_string("task_compacted_turn-2");
        let attempt1 = s.open_attempt(&task1, 1).await;
        let attempt2 = s.open_attempt(&task2, 1).await;
        let original = sample_messages();

        commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task1,
                expected_turn: 1,
                turn_attempt_id: attempt1,
                close_attempt_params: sample_close_params(),
                messages: original.clone(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
        )
        .await
        .context("commit pre-compaction turn")?;

        let summary = llm::Message::assistant("[summary of turn 1]");
        s.messages
            .append_compaction(
                &thread_a(),
                vec![summary.clone()],
                original.len(),
                0,
                t_plus(2),
            )
            .await
            .context("append compaction lineage")?;

        let new_message = llm::Message::user("turn 2");
        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task2,
                expected_turn: 2,
                turn_attempt_id: attempt2,
                close_attempt_params: sample_close_params(),
                messages: vec![new_message.clone()],
                turn_usage: usage(10, 5),
                agent_state_snapshot: serde_json::json!({"turn": 2}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(3),
            },
            &s.tasks,
        )
        .await
        .context("commit post-compaction turn")?;

        let projection = s
            .messages
            .get(&thread_a())
            .await?
            .context("projection after compacted commit")?;
        assert_eq!(
            projection.messages.len(),
            original.len() + 1,
            "raw transcript must retain the compacted source and append turn 2",
        );
        let effective = projection.context_history();
        assert_eq!(
            serde_json::to_value(&outcome.checkpoint.messages)?,
            serde_json::to_value(&effective)?,
            "checkpoint must snapshot the effective compacted view",
        );
        assert_eq!(
            serde_json::to_value(&outcome.checkpoint.messages)?,
            serde_json::to_value(vec![summary, new_message])?,
        );

        // `fork_thread` seeds both the destination projection and checkpoint
        // from the selected checkpoint's messages. Keeping this fork input
        // effective prevents the fork from resurrecting the compacted prefix.
        let fork_seed = outcome.checkpoint.messages.clone();
        assert_eq!(
            serde_json::to_value(fork_seed)?,
            serde_json::to_value(effective)?,
        );
        Ok(())
    }

    // ── Stale turn double-commit guard ───────────────────────────

    #[tokio::test]
    async fn stale_expected_turn_is_rejected_while_correct_one_succeeds() -> Result<()> {
        let s = Stores::new();
        let task1 = AgentTaskId::from_string("task_stale-1");
        let task2_stale = AgentTaskId::from_string("task_stale-2a");
        let task2_ok = AgentTaskId::from_string("task_stale-2b");
        let a1 = s.open_attempt(&task1, 1).await;
        let a2_stale = s.open_attempt(&task2_stale, 1).await;
        let a2_ok = s.open_attempt(&task2_ok, 1).await;

        // Turn 1 commits cleanly; the thread now sits at committed_turns = 1.
        commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task1,
                expected_turn: 1,
                turn_attempt_id: a1,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 1")],
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
        )
        .await
        .context("turn 1")?;

        // A stale-lease worker still believes it is producing turn 1.
        // The guard rejects it before any projection advances.
        let err = commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task2_stale,
                expected_turn: 1,
                turn_attempt_id: a2_stale,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("stale turn 2")],
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({"turn": 2}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(2),
            },
            &s.tasks,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("stale turn commit"),
            "expected stale-turn rejection, got: {err}",
        );

        // The thread did not advance and no second checkpoint landed.
        let thread = s.threads.get(&thread_a()).await?.context("thread")?;
        assert_eq!(thread.committed_turns, 1, "stale commit must not advance");
        assert_eq!(s.checkpoints.list_by_thread(&thread_a()).await?.len(), 1);

        // The correct expected_turn for the next turn succeeds.
        let ok = commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task2_ok,
                expected_turn: 2,
                turn_attempt_id: a2_ok,
                close_attempt_params: sample_close_params(),
                messages: vec![llm::Message::user("turn 2")],
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({"turn": 2}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(3),
            },
            &s.tasks,
        )
        .await
        .context("turn 2")?;
        assert_eq!(ok.thread.committed_turns, 2);
        assert_eq!(ok.checkpoint.turn_number, 2);
        assert_eq!(s.checkpoints.list_by_thread(&thread_a()).await?.len(), 2);

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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task_a,
                expected_turn: 1,
                turn_attempt_id: a_a,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
        )
        .await
        .context("thread a")?;

        commit_completed_turn(
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_b(),
                task_id: task_b,
                expected_turn: 1,
                turn_attempt_id: a_b,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(2),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: AgentTaskId::from_string("task_x"),
                expected_turn: 1,
                turn_attempt_id: TurnAttemptId::from_string("attempt_nonexistent"),
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(2),
            },
            &s.tasks,
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

    /// Two IN-PROCESS committers racing for the same turn slot
    /// serialize on the store's sequential-commit lock. Exactly one
    /// lands; the loser is rejected by the decisive pre-check with a
    /// typed `StaleTurnCommit` and — critically — its attempt is
    /// STILL OPEN, so the slot-shift retry can close it on the
    /// retried commit. The loser then lands cleanly on turn 2.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn racing_commits_serialize_and_the_loser_keeps_its_attempt_open() -> Result<()> {
        let s = std::sync::Arc::new(Stores::new());
        let task_a = AgentTaskId::from_string("task_racer_a");
        let task_b = AgentTaskId::from_string("task_racer_b");
        let attempt_a = s.open_attempt(&task_a, 1).await;
        let attempt_b = s.open_attempt(&task_b, 1).await;

        let commit_for = |task_id: AgentTaskId,
                          attempt_id: crate::journal::turn_attempt::TurnAttemptId,
                          expected_turn: u32| CompletedTurnCommit {
            extra_writes: Vec::new(),
            delivered_injection_ids: Vec::new(),
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_a(),
            task_id,
            expected_turn,
            turn_attempt_id: attempt_id,
            close_attempt_params: sample_close_params(),
            messages: sample_messages(),
            turn_usage: usage(100, 50),
            agent_state_snapshot: serde_json::json!({}),
            events: Vec::new(),
            outbox_max_attempts: 3,
            owner_guard: None,
            now: t_plus(2),
        };

        let run = |params: CompletedTurnCommit, stores: std::sync::Arc<Stores>| async move {
            commit_completed_turn(params, &stores.tasks).await
        };

        let first = tokio::spawn(run(
            commit_for(task_a.clone(), attempt_a.clone(), 1),
            std::sync::Arc::clone(&s),
        ));
        let second = tokio::spawn(run(
            commit_for(task_b.clone(), attempt_b.clone(), 1),
            std::sync::Arc::clone(&s),
        ));
        let (first, second) = (first.await?, second.await?);

        let (winner_task, loser_task, loser_attempt, loser_err) = match (first, second) {
            (Ok(_), Err(err)) => (task_a, task_b, attempt_b, err),
            (Err(err), Ok(_)) => (task_b, task_a, attempt_a, err),
            (Ok(_), Ok(_)) => anyhow::bail!("both racers committed turn 1 — CAS is broken"),
            (Err(e1), Err(e2)) => anyhow::bail!("both racers failed: {e1:#} / {e2:#}"),
        };

        // The loser's rejection is the typed slot collision…
        assert!(
            loser_err
                .chain()
                .any(|cause| cause.downcast_ref::<StaleTurnCommit>().is_some()),
            "expected StaleTurnCommit, got: {loser_err:#}",
        );
        // …and its attempt is untouched, which is what the slot-shift
        // retry depends on.
        let open_attempt = s
            .attempts
            .get(&loser_attempt)
            .await?
            .context("loser attempt")?;
        assert!(
            !open_attempt.is_closed(),
            "the losing commit must leave its attempt open",
        );
        let thread = s.threads.get(&thread_a()).await?.context("thread")?;
        assert_eq!(thread.committed_turns, 1);
        let occupant = s
            .checkpoints
            .get_by_turn(&thread_a(), 1)
            .await?
            .context("winner checkpoint")?;
        assert_eq!(occupant.task_id, winner_task);

        // The loser retries at the next slot and lands cleanly.
        let retried =
            commit_completed_turn(commit_for(loser_task.clone(), loser_attempt, 2), &s.tasks)
                .await
                .context("the loser's retried commit must land on turn 2")?;
        assert_eq!(retried.thread.committed_turns, 2);
        assert_eq!(retried.checkpoint.task_id, loser_task);
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task1,
                expected_turn: 1,
                turn_attempt_id: a1,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id: task2,
                expected_turn: 2,
                turn_attempt_id: a2,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(200, 80),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(3),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(5),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: serde_json::json!({}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(1),
            },
            &s.tasks,
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
                extra_writes: Vec::new(),
                delivered_injection_ids: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt_id,
                close_attempt_params: sample_close_params(),
                messages: sample_messages(),
                turn_usage: usage(100, 50),
                agent_state_snapshot: snapshot.clone(),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(5),
            },
            &s.tasks,
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
