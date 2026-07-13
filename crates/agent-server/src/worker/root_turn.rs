//! Root turn execution: text-only commit, tool-boundary suspension, and
//! journal-driven resume from completed child tool results.
//!
//! This module drives a single root-turn attempt from LLM call through
//! to either a completed-turn commit (text-only), a tool-boundary
//! suspension that spawns child tasks (tool calls present), or a
//! resumed turn after child tool tasks reach terminal states.
//!
//! # Execution flow
//!
//! 1. **Open turn attempt** — audit record via [`TurnAttemptStore`].
//! 2. **Build chat request** — system prompt + staged messages + user
//!    prompt from [`AgentDefinition`] and [`RootWorkerInputs`].
//! 3. **Call LLM** — `LlmProvider::chat()`.
//! 4. **Branch on response content:**
//!    - **Text-only** (Phase 4.3): buffer messages, drain staged stores,
//!      [`commit_completed_turn`], advance task to `Completed`.
//!    - **Tool calls** (Phase 4.4): build [`ContinuationEnvelope`],
//!      close the turn attempt, [`spawn_tool_children`] to atomically
//!      create `tool_runtime` children and park the parent in
//!      `WaitingOnChildren`. No checkpoint is created.
//!
//! # Guarantees
//!
//! - No durable writes before the commit path succeeds (text-only).
//! - Turn attempt is closed atomically as part of the commit (text-only)
//!   or explicitly before child spawn (tool suspension).
//! - Task advances to `Completed` only after commit succeeds (text-only).
//! - On the suspension path, the parent task holds a
//!   [`ContinuationEnvelope`] in its [`TaskState::WaitingOnChildren`]
//!   payload — enough for Phase 5 to resume.
//!
//! [`ContinuationEnvelope`]: agent_sdk_foundation::ContinuationEnvelope
//! [`spawn_tool_children`]: crate::journal::store::AgentTaskStore::spawn_tool_children
//! [`TaskState::WaitingOnChildren`]: crate::journal::task_state::TaskState::WaitingOnChildren

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{self, ChatRequest};
use agent_sdk_foundation::{
    AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, TokenUsage,
    ToolResult, ToolTier,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_providers::streaming::{StreamAccumulator, StreamDelta, StreamErrorKind};
use agent_sdk_tools::stores::{MessageStore, StateStore};
use anyhow::{Context, Result, bail, ensure};
use futures::StreamExt;
use std::collections::BTreeMap;
use std::time::Duration;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::definition::{AgentDefinition, ThinkingPolicy};
use crate::journal::checkpoint::CheckpointKind;
use crate::journal::checkpoint_store::CheckpointStore;
use crate::journal::commit::{
    CommitOutcome, CommitOwnerGuard, CompletedTurnCommit, DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
    StaleTurnCommit, commit_completed_turn,
};
use crate::journal::committed_event::CommittedEvent;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::EventRepository;
use crate::journal::execution_context::RootWorkerInputs;
use crate::journal::message_store::MessageProjectionStore;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
};
use crate::journal::task_state::TaskState;
use crate::journal::task_wakeup::WakeupSignal;
use crate::journal::thread_store::ThreadStore;
use crate::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptOutcome, TurnAttemptSchemaError,
};
use crate::journal::turn_attempt_store::TurnAttemptStore;

// ─────────────────────────────────────────────────────────────────────
// Store dependencies
// ─────────────────────────────────────────────────────────────────────

/// Durable store handles needed by [`execute_root_turn`].
///
/// Separating these from [`RootWorkerInputs`] keeps the inputs struct
/// (which is reconstructed from durable state) independent from the
/// runtime store wiring.
pub struct RootTurnDeps<'a> {
    pub task_store: &'a dyn AgentTaskStore,
    pub thread_store: &'a dyn ThreadStore,
    pub message_store: &'a dyn MessageProjectionStore,
    pub attempt_store: &'a dyn TurnAttemptStore,
    pub checkpoint_store: &'a dyn CheckpointStore,
    pub event_repo: &'a dyn EventRepository,
    /// Same-process live-tail broadcaster.  Required so per-delta
    /// `TextDelta` / `ThinkingDelta` events committed during streaming
    /// reach `StreamThreadEvents` subscribers — the durable
    /// [`EventRepository::commit_event`] path doesn't go through the
    /// outbox + relay, so the worker has to notify directly.
    pub event_notifier: &'a EventNotifier,
    /// Optional per-call routing selector consulted at the tool
    /// boundary.  When `Some`, batches that resolve to a single
    /// subagent decision route through
    /// [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation)
    /// instead of the regular `spawn_tool_children` path; everything
    /// else flows through `spawn_tool_children` exactly as before.
    ///
    /// Defaults to `None` so every existing call site preserves
    /// pre-PR behaviour without wiring changes.  Hosts that want
    /// durable subagent routing pass a real selector — see
    /// [`crate::worker::subagent_spawn_selector`].
    pub subagent_spawn_selector:
        Option<&'a (dyn super::subagent_spawn_selector::SubagentSpawnSelector + 'a)>,
    /// Optional auto-compaction policy supplied by the host. When
    /// set, the worker's `build_chat_request` (and the resume-path
    /// equivalent) run a pre-call threshold check against the staged
    /// history and rewrite the durable [`MessageProjectionStore`]
    /// when the threshold is crossed, and `call_llm_with_retry`
    /// reactively compacts on `prompt is too long` errors instead
    /// of going fatal. (Both helpers are private — see
    /// `crates/agent-server/src/worker/compaction.rs` for the public
    /// surface that drives them.)
    ///
    /// `None` (the default) preserves today's "no automatic
    /// compaction" behaviour for every existing host.
    pub compaction_config: Option<&'a agent_sdk::context::CompactionConfig>,
    /// `Arc` handle to the same provider passed positionally to
    /// [`execute_root_turn`] / [`resume_root_turn`]. Required when
    /// `compaction_config` is `Some` so the worker can build an
    /// [`agent_sdk::context::LlmContextCompactor`] (which takes
    /// ownership of an `Arc<dyn LlmProvider>` for the duration of
    /// the summarisation call). Leaving the borrowed-vs-owned
    /// shapes side-by-side keeps every existing call site —
    /// including the in-crate test fixtures that pass
    /// `&MockTextProvider` — compiling unchanged; only hosts
    /// opting into compaction need to supply the Arc.
    pub compaction_provider: Option<&'a std::sync::Arc<dyn agent_sdk_providers::LlmProvider>>,
    /// Optional cooperative-cancellation token for the root turn.
    ///
    /// When `Some` and the token is tripped (via `cancel_root_turn` /
    /// `cancel_tree` or a host shutdown), the worker stops consuming the
    /// LLM stream, aborts retry backoff sleeps, and skips
    /// auto-compaction instead of running them to completion — so a
    /// cancelled turn stops burning provider tokens and surfaces to the
    /// user promptly rather than only failing at the final commit CAS.
    ///
    /// `None` (the default) preserves the prior "run to completion
    /// regardless of cancellation" behaviour for every existing host.
    pub cancel: Option<&'a CancellationToken>,
    /// Optional worker-pool wakeup signal.
    ///
    /// When `Some`, the worker fires a nudge the instant it journals a
    /// batch of runnable tool children (via `spawn_tool_children` or a
    /// subagent-invocation spawn) so parked workers pick the children up
    /// immediately instead of waiting out the host's
    /// `acquisition_interval` ticker — and so the whole batch starts
    /// together rather than staggered across successive ticks.
    ///
    /// `None` (the default) preserves the poll-only behaviour for every
    /// existing host; the ticker remains the lost-wakeup backstop even
    /// when a signal is wired.
    pub wakeup: Option<&'a WakeupSignal>,
}

impl RootTurnDeps<'_> {
    /// True when a cancellation token is wired and has been tripped.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(CancellationToken::is_cancelled)
    }

    /// Nudge every parked worker that a batch of children just became
    /// runnable. No-op when no [`WakeupSignal`] is wired.
    ///
    /// Uses `wake_all_now` (a broadcast) rather than a single
    /// `notify_workers` permit because a batch typically makes several
    /// children runnable at once and every idle worker should be free to
    /// claim one — the `Pending → Running` CAS in `acquire_next_runnable`
    /// still serialises them, so a broadcast only ever costs a few
    /// no-op wakeups, never a double execution.
    fn wake_workers_for_batch(&self) {
        if let Some(signal) = self.wakeup {
            signal.wake_all_now();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Streaming content IDs
// ─────────────────────────────────────────────────────────────────────

/// Per-block message IDs assigned during streaming.
///
/// Both maps are keyed by the LLM's `block_index` so multi-block
/// responses (e.g. two text blocks separated by a tool-use block) get
/// distinct IDs per block.  Each ID is generated lazily on the first
/// non-empty `TextDelta` / `ThinkingDelta` for its block, so blocks
/// that never produce any content (or only empty deltas) never
/// allocate an ID.
///
/// The same IDs are reused for the consolidated [`AgentEvent::Text`] /
/// [`AgentEvent::Thinking`] events emitted at turn close, so streaming
/// clients can correlate every delta with the final event by id.
///
/// IDs are [`Uuid`] values (`UUIDv7`) rather than `String` so the type
/// system enforces the invariant that they are well-formed UUIDs and
/// debug output stays compact.  `UUIDv7` is time-ordered, so journal
/// readers that sort events by id observe the same chronological
/// ordering as the event stream itself.
#[derive(Debug, Default)]
struct ContentIds {
    text_ids: BTreeMap<usize, Uuid>,
    thinking_ids: BTreeMap<usize, Uuid>,
}

impl ContentIds {
    /// Get-or-insert the message id for a text block, generating a
    /// fresh `UUIDv7` on first use.
    fn text_id_for(&mut self, block_index: usize) -> Uuid {
        *self
            .text_ids
            .entry(block_index)
            .or_insert_with(Uuid::now_v7)
    }

    /// Get-or-insert the message id for a thinking block.
    fn thinking_id_for(&mut self, block_index: usize) -> Uuid {
        *self
            .thinking_ids
            .entry(block_index)
            .or_insert_with(Uuid::now_v7)
    }

    /// First text id in block-index order, if any.
    ///
    /// Used to assign a stable id to the [`AgentEvent::Refusal`]
    /// emitted when the model refuses with text.
    fn first_text_id(&self) -> Option<Uuid> {
        self.text_ids.values().next().copied()
    }
}

/// Outcome of a single [`call_llm_once`] call: the synthesized
/// [`llm::ChatResponse`] plus the per-block IDs assigned during
/// streaming.  Owned by the retry loop — callers see the
/// [`StreamedTurn`] the wrapper assembles.
struct OnceOutcome {
    response: llm::ChatResponse,
    content_ids: ContentIds,
}

/// Outcome of [`call_llm_with_retry`]: the synthesized
/// [`llm::ChatResponse`], the per-block IDs assigned during streaming,
/// and the [`TurnAttempt`] that produced them.
///
/// The attempt is owned by this struct because [`call_llm_with_retry`]
/// may have minted fresh attempts during the retry loop — callers need
/// the *successful* attempt for the audit close in the
/// commit/suspension paths, not the one they passed in.
struct StreamedTurn {
    response: llm::ChatResponse,
    content_ids: ContentIds,
    attempt: TurnAttempt,
}

/// Bundle of context needed to commit / suspend a root turn after
/// streaming completes.
///
/// Groups the per-block IDs (so the consolidated content events match
/// the streamed deltas by id) with the per-turn admission events
/// produced at turn open (so the outcome's `committed_events` includes
/// every event written for the turn).
struct TurnCloseContext {
    content_ids: ContentIds,
    /// Durable user-input event committed on the first attempt of
    /// the task. `None` on retry attempts (the prompt has already
    /// been committed once for the turn) and for resume / empty
    /// inputs that do not surface a prompt to clients.
    user_input_committed: Option<CommittedEvent>,
    start_committed: CommittedEvent,
}

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// Result of a successful [`execute_root_turn`].
///
/// The two variants correspond to the text-only (Phase 4.3) and
/// tool-suspension (Phase 4.4) paths.
#[derive(Debug)]
pub enum RootTurnOutcome {
    /// Text-only turn completed and committed.
    ///
    /// A checkpoint was created, the message projection was updated,
    /// and the task is now `Completed`.
    Completed {
        /// The commit outcome (thread, checkpoint, closed attempt).
        ///
        /// Boxed to keep the enum size small (the `Suspended` variant
        /// is much smaller).
        commit: Box<CommitOutcome>,
        /// The completed task after advancing.
        completed_task: AgentTask,
        /// The assistant's text response.
        response_text: String,
        /// Lifecycle events committed with this turn.
        committed_events: Vec<CommittedEvent>,
    },

    /// Turn suspended at the tool boundary.
    ///
    /// The parent task is now `WaitingOnChildren` with a
    /// [`ContinuationEnvelope`] persisted on its [`TaskState`].
    /// One `tool_runtime` child task was created per tool call.
    /// No checkpoint or message-projection write occurred.
    ///
    /// [`TaskState`]: crate::journal::task_state::TaskState
    Suspended {
        /// The parent task after transitioning to `WaitingOnChildren`.
        parent_task: AgentTask,
        /// The child tasks created, one per tool call.
        child_tasks: Vec<AgentTask>,
        /// `ToolCallStart` events committed for each spawned child.
        committed_events: Vec<CommittedEvent>,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Failure & cancellation
// ─────────────────────────────────────────────────────────────────────

/// Fail a root turn after `execute_root_turn` or `resume_root_turn`
/// returns `Err`.
///
/// Transitions the task to [`TaskStatus::Failed`] via
/// [`AgentTaskStore::fail_task`] and best-effort closes any open turn
/// attempts for the task so the audit trail is clean.
///
/// Because staged projections are in-memory only, they are naturally
/// discarded when the `RootWorkerInputs` is dropped on the error path.
/// This function handles the durable side: marking the task terminal
/// and closing open attempts.
///
/// # Errors
///
/// Returns an error if the store rejects the transition (e.g. task is
/// already terminal from a concurrent cancel).
pub async fn fail_root_turn(
    task_id: &AgentTaskId,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    thread_id: &agent_sdk_foundation::ThreadId,
    error: &anyhow::Error,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    // Best-effort close any open turn attempts for this task.
    best_effort_close_open_attempts(task_id, deps.attempt_store, now).await;
    fail_root_turn_inner(task_id, worker_id, lease_id, thread_id, error, deps, now).await
}

/// [`fail_root_turn`] without the open-attempt pre-close, for callers
/// that fail a root turn whose worker is still LIVE in this process —
/// the host's subagent-timeout heartbeat path.
///
/// The pre-close would race the live worker: in the window between a
/// successful stream and `commit_completed_turn`, closing its open
/// attempt as `Cancelled` with zero tokens both clobbers the
/// real-usage close (the billing source of truth on attempt rows) and
/// makes the in-transaction close hit `AlreadyClosed`, aborting the
/// whole commit. The live worker's own abort path (mid-stream cancel
/// close, or its terminal commit) owns its attempts; this variant only
/// performs the durable fail + error event.
///
/// # Errors
///
/// Same envelope as [`fail_root_turn`].
pub async fn fail_root_turn_leaving_attempts_open(
    task_id: &AgentTaskId,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    thread_id: &agent_sdk_foundation::ThreadId,
    error: &anyhow::Error,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    fail_root_turn_inner(task_id, worker_id, lease_id, thread_id, error, deps, now).await
}

async fn fail_root_turn_inner(
    task_id: &AgentTaskId,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    thread_id: &agent_sdk_foundation::ThreadId,
    error: &anyhow::Error,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    let error_msg = format!("{error:#}");
    let (failed_task, _parent) = deps
        .task_store
        .fail_task(task_id, worker_id, lease_id, error_msg.clone(), now)
        .await
        .context("fail root task")?;

    // Best-effort: the task is durably Failed; event commit failure
    // must not override that outcome. Consistent with
    // best_effort_close_open_attempts.
    let error_event = AgentEvent::error(error_msg, false);
    let _ = deps
        .event_repo
        .commit_event(thread_id, error_event, now)
        .await;

    // Finalize the root `invoke_agent` span with the error outcome and
    // its full duration (no-op if a prior process owns the live span).
    #[cfg(feature = "otel")]
    agent_sdk::observability::loop_instrument::finalize_root_turn_span(
        task_id.as_str(),
        0,
        None,
        "error",
    );

    Ok(failed_task)
}

/// Cancel a root turn and its entire subtree.
///
/// Before tearing the subtree down, this commits the completed prefix of
/// a **parked** turn (`WaitingOnChildren`, or `Pending` +
/// [`TaskState::ReadyToResume`]) so the next turn's LLM context survives
/// the cancel — see `commit_partial_turn_on_cancel`. A `Running` task
/// is skipped here: its live worker loop owns the commit (seam B), and
/// committing from both seams would race on the same turn number. The
/// partial commit is best-effort — any error is logged and swallowed so
/// the cancel always wins.
///
/// Then it best-effort closes any open turn attempts — with the same
/// ownership split as the prefix commit: a **`Running`** row's attempts
/// are left to its live worker, whose cancel path closes the streaming
/// attempt when the tripped token aborts the stream and whose
/// suspension / commit closes record the round's **real token usage**.
/// [`TurnAttemptStore::close_attempt`] is single-shot (an
/// already-closed row rejects), so closing here first would pin a
/// zero-usage `Cancelled` row over that usage — or abort an in-flight
/// completed-turn transaction whose in-transaction close then hits
/// `AlreadyClosed`. Parked / queued rows have no live worker, so this
/// seam closes them. Finally it calls
/// [`AgentTaskStore::cancel_tree`] to atomically cancel the
/// root task and any live descendant tasks (e.g. `tool_runtime` children
/// spawned during suspension). `cancel_tree` clears each row's
/// `TaskState` to `None`; the durable message-projection draft slot
/// survives, so a re-seeded suffix (the dropped trailing
/// `assistant + tool_use`) is closed by the next turn's
/// `backfill_orphaned_tool_results`.
///
/// The terminal [`AgentEvent::Cancelled`] marker is committed by
/// [`AgentTaskStore::cancel_tree`] itself, atomically with the
/// cancellation (same transaction on the durable backends): one marker
/// per transitioned *blocking* root
/// ([`TaskStatus::blocks_root_admission`]) on that root's **own**
/// thread — including child-thread roots cancelled across
/// `SubagentInvocation` links — each with a `thread_events_available`
/// outbox advisory for cross-host followers. Event-stream followers
/// get a closing frame instead of waiting forever for a `Done` that
/// will never come; cancelling a QUEUED root behind a live active root
/// stays silent so the active root's followers are not closed
/// mid-stream. Idempotent cancel retries transition nothing and emit
/// nothing, and the running worker's abort path (seam B) commits no
/// lifecycle events, so the marker lands exactly once per effective
/// cancel — a crash after `cancel_tree` commits can no longer lose it.
/// This function forwards the returned markers to the in-process event
/// notifier for same-process live-tail latency.
///
/// # Post-marker salvage ordering (cancelled `Running` root)
///
/// A cancelled `Running` root's worker only notices on its next
/// rejected heartbeat, so for up to one heartbeat interval it may keep
/// committing streaming delta events — which now land at sequences
/// strictly **after** the marker — and then its seam-B salvage commits
/// state projections (messages / checkpoint / thread aggregate) with
/// no lifecycle events at all. Followers that closed on the marker
/// miss only that post-terminal salvage; a follower that reconnects
/// with a cursor past the marker may replay trailing salvage deltas
/// and then wait (no further lifecycle close arrives until a successor
/// turn commits). Treat the marker as the authoritative close for the
/// cancelled turn.
///
/// # Cancelled `Running` root vs the successor turn
///
/// `cancel_tree` frees the thread's active-root slot and promotes a
/// queued successor in the same store transaction, while the cancelled
/// worker's seam-B salvage may still commit into the turn slot it
/// bootstrapped with. The `expected_turn` CAS makes whichever commit
/// lands second fail: a losing salvage is dropped silently (benign),
/// and a successor that loses to the salvage no longer fails — its
/// commit detects the cross-task slot collision and shifts to the next
/// turn number (see `commit_completed_turn_shifting_slot`).
///
/// # Returns
///
/// The task IDs that were actually transitioned (excludes rows that
/// were already terminal).
/// # Errors
///
/// Returns an error if the store rejects the transition.
pub async fn cancel_root_turn(
    task_id: &AgentTaskId,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<Vec<AgentTaskId>> {
    // Snapshot the attempt list BEFORE reading the row. The snapshot ↔
    // row-read order closes the acquisition TOCTOU: a worker that
    // acquires the task after the snapshot either flips the row to
    // `Running` before we read it (→ the close below is skipped), or
    // opens its attempt after the snapshot (→ the attempt is not in the
    // snapshot and is never pre-closed). Nothing else fences acquisition
    // until `cancel_tree` removes the row from the runnable indexes.
    // Residual: a zombie worker whose lease already expired (row
    // requeued to `Pending`) can still see its old attempt pre-closed —
    // the same pre-existing class as `fail_root_turn`'s unconditional
    // close.
    let attempts_snapshot = deps
        .attempt_store
        .list_by_task(task_id)
        .await
        .unwrap_or_default();

    // Read the row once: it sources the parked-turn prefix commit and
    // decides who owns the attempt close. A read error degrades to the
    // prior behaviour (no salvage, snapshot attempts closed here).
    let task = deps.task_store.get(task_id).await.ok().flatten();

    // Commit the completed prefix of a parked turn before the task goes
    // terminal — sourced from the durable draft / suspended messages
    // *before* `cancel_tree` clears the typed state.
    if let Some(task) = &task {
        best_effort_commit_parked_cancel(task, deps, now).await;
    }

    // A `Running` row's live worker owns the attempt close (see the doc
    // above): pre-closing here would win the store's single-close CAS
    // with a zeroed row and discard the real usage its suspension /
    // commit close carries, or roll back an in-flight completed-turn
    // transaction. If that worker dies before closing, the attempt row
    // stays open — the same already-tolerated state every non-cancel
    // process death leaves behind (nothing sweeps orphaned attempts).
    let has_live_worker = task
        .as_ref()
        .is_some_and(|task| task.status == TaskStatus::Running);
    if !has_live_worker {
        best_effort_close_attempts(&attempts_snapshot, deps.attempt_store, now).await;
    }

    let outcome = deps
        .task_store
        .cancel_tree(task_id, now)
        .await
        .context("cancel root turn tree")?;
    let cancelled = outcome.transitioned;

    // The store committed the terminal `Cancelled` marker(s) durably
    // (atomically with the cancellation, one per affected thread, with
    // an outbox advisory for cross-host followers). Forward them to
    // the in-process notifier so same-process live followers — on the
    // root's thread AND on cascade-cancelled child threads — close
    // immediately instead of waiting on the outbox relay.
    if !outcome.markers.is_empty() {
        deps.event_notifier.notify(&outcome.markers);
    }

    // Finalize the root `invoke_agent` span with the cancelled outcome
    // (no-op if a prior process owns the live span).
    #[cfg(feature = "otel")]
    agent_sdk::observability::loop_instrument::finalize_root_turn_span(
        task_id.as_str(),
        0,
        None,
        "cancelled",
    );

    Ok(cancelled)
}

/// Revert a failed steering wake (R2) back to its pre-wake parked
/// state instead of failing the root task.
///
/// A steering wake is a [`TaskState::ReadyToResume`] row carrying a
/// non-empty `steering` payload that woke a parent while its mission
/// children were still **running**. If the bounded steering exchange
/// fails — e.g. a provider outage exactly when the user asked "how is
/// it going?", or a deterministic interim-construction error — the
/// naive path ([`fail_root_turn`]) marks the parent `Failed` while its
/// children keep running. [`AgentTaskStore::fail_task`] does not
/// cascade (only [`AgentTaskStore::cancel_tree`] does), so those
/// workers are stranded: they burn budget on a dead mission and their
/// eventual results are orphaned on the "parent not waiting" branch.
///
/// The steering exchange writes nothing to the task row before it
/// fails, so the durable `ReadyToResume` state still carries the
/// original continuation, suspended messages, and child ids untouched.
/// This re-parks the parent on exactly those children via
/// [`AgentTaskStore::repark_after_steering`], which derives the live
/// child count from the journal and either parks again in
/// [`TaskStatus::WaitingOnChildren`] (children still running) or flips
/// straight to `Pending` + [`TaskState::ReadyToResume`] with an empty
/// steering payload (every child already finished) so the ordinary
/// fan-in ([`resume_from_children`]) resumes the mission.
///
/// The (now-unanswerable) steering note is intentionally dropped.
/// Dropping it also clears the wake trigger, so the row is never
/// re-woken into the same failing exchange — that removes the
/// wake→error→re-park loop the durable steering-attempt cap (which
/// only bites on lease expiry) would not otherwise bound. The failure
/// is surfaced as a non-fatal error event so the note's author still
/// learns the reply did not land.
///
/// `parent` is the owned steering-resume row (its durable state carries
/// the intact continuation, suspended messages, and child ids);
/// `worker_id` / `lease_id` are the caller's validated ownership token.
///
/// # Errors
/// - The `parent`'s durable state carries no continuation (defends the
///   state ↔ status invariant).
/// - The store rejects the re-park (CAS: not running / not owned / not
///   a steering resume — e.g. a sweep already requeued the row).
pub async fn revert_steering_wake(
    parent: &AgentTask,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    error: &anyhow::Error,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    // Best-effort close the steering exchange's open turn attempt — the
    // LLM round failed, so the attempt never closed on its own.
    best_effort_close_open_attempts(&parent.id, deps.attempt_store, now).await;

    let continuation = parent
        .state
        .continuation()
        .context("steering revert: durable state carries no continuation")?
        .clone();
    let payload = SuspensionPayload {
        continuation,
        suspended_messages: parent.state.suspended_messages().to_vec(),
    };
    // The original child ids, in spawn order — the re-park re-indexes
    // them to the continuation's original `pending_tool_calls`, i.e. the
    // identity mapping they already had, so the fan-in resolves exactly
    // as if the wake never happened.
    let child_ids = parent.state.child_ids().to_vec();

    let reparked = deps
        .task_store
        .repark_after_steering(&parent.id, worker_id, lease_id, payload, child_ids, now)
        .await
        .context("revert failed steering wake to parked state")?;

    // Surface the failure as a non-fatal error event (best-effort — the
    // durable re-park already succeeded; an event-commit failure must
    // not override that outcome).
    let error_event = AgentEvent::error(format!("{error:#}"), false);
    let _ = deps
        .event_repo
        .commit_event(&parent.thread_id, error_event, now)
        .await;

    Ok(reparked)
}

/// Best-effort close any open (non-closed) turn attempts for a task.
///
/// Iterates all attempts and closes any that are still open with the
/// `Cancelled` outcome. Errors are swallowed — the caller's primary
/// operation (fail or cancel) takes precedence.
///
/// Public for the service host's force-drop path (issue #299): when an
/// execution future ignores cancellation past the abort grace and is
/// dropped, no live worker remains to close the attempts the drop
/// orphaned, so the host settles them with the same best-effort close.
pub async fn best_effort_close_open_attempts(
    task_id: &AgentTaskId,
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
) {
    let Ok(attempts) = attempt_store.list_by_task(task_id).await else {
        return;
    };
    best_effort_close_attempts(&attempts, attempt_store, now).await;
}

/// Close every still-open attempt in `attempts` with the `Cancelled`
/// outcome. Split from [`best_effort_close_open_attempts`] so
/// [`cancel_root_turn`] can close a pre-read snapshot instead of a
/// fresh listing (see the acquisition-TOCTOU comment there).
async fn best_effort_close_attempts(
    attempts: &[TurnAttempt],
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
) {
    for attempt in attempts {
        if !attempt.is_closed() {
            close_attempt_with(attempt, TurnAttemptOutcome::Cancelled, attempt_store, now).await;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Commit the completed prefix of a cancelled turn
// ─────────────────────────────────────────────────────────────────────

/// Zero-sized marker attached to the error a root turn bails with when
/// it is cancelled (`stop` / ESC → `cancel_tree`).
///
/// The live worker loop propagates the cancel as an ordinary
/// `anyhow::Error`, but the caller that still owns the in-memory staged
/// buffer needs to distinguish "the turn was cancelled" from any other
/// failure so it can commit the completed prefix before the task goes
/// terminal. Callers detect it via
/// `error.chain().any(|c| c.is::<RootTurnCancelledMarker>())` — see
/// [`is_root_turn_cancelled`].
#[derive(Debug)]
pub(crate) struct RootTurnCancelledMarker;

impl std::fmt::Display for RootTurnCancelledMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("root turn cancelled")
    }
}

impl std::error::Error for RootTurnCancelledMarker {}

/// True when `error` carries a [`RootTurnCancelledMarker`] anywhere in
/// its cause chain — i.e. the turn bailed because it was cancelled
/// mid-stream rather than for any other reason.
pub(crate) fn is_root_turn_cancelled(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<RootTurnCancelledMarker>().is_some())
}

/// Split `messages` into `(committable prefix, retained suffix)` per the
/// provider-validity rule: the prefix is the largest leading slice with
/// no assistant `tool_use` block left unanswered by a `tool_result`
/// within that same slice.
///
/// Walks back from the tail while
/// [`llm::has_unbalanced_tool_use`](agent_sdk_foundation::llm::has_unbalanced_tool_use)
/// holds, so the returned prefix is always a provider-legal request
/// history — it ends on a user message (a bare prompt or a tool-results
/// message), which every provider accepts. Messages are moved, never
/// mutated or re-ordered: text, `tool_use`, `tool_result`, `thinking`
/// (+ signature), and `redacted_thinking` blocks commit byte-verbatim.
pub(crate) fn provider_valid_split(
    mut messages: Vec<llm::Message>,
) -> (Vec<llm::Message>, Vec<llm::Message>) {
    let mut n = messages.len();
    while n > 0 && llm::has_unbalanced_tool_use(&messages[..n]) {
        n -= 1;
    }
    let suffix = messages.split_off(n);
    (messages, suffix)
}

/// Parameters for `commit_partial_turn_on_cancel`.
pub(crate) struct PartialCancelCommit<'a> {
    /// Thread the cancelled turn belongs to.
    pub(crate) thread_id: &'a agent_sdk_foundation::ThreadId,
    /// Task that was running (or parked on) the cancelled turn.
    pub(crate) task_id: &'a AgentTaskId,
    /// Candidate messages accumulated for the turn, in order. The source
    /// depends on the seam: the live loop supplies `[user prompt] +`
    /// staged delta (fresh) or the staged `[suspended…, tool_results]`
    /// delta (resume); the external-cancel seam supplies the durable
    /// draft or the parked `suspended_messages`.
    pub(crate) candidate: Vec<llm::Message>,
    /// `thread.committed_turns + 1` — the turn number this commit
    /// consumes. The in-transaction CAS in `commit_completed_turn` is the
    /// sole idempotency authority.
    pub(crate) expected_turn: u32,
    /// Agent-state snapshot for the checkpoint written at `expected_turn`.
    pub(crate) agent_state_snapshot: serde_json::Value,
}

/// Durably commit the largest provider-valid prefix of a cancelled
/// turn's accumulated messages, returning the retained suffix.
///
/// A no-op when the prefix is empty (`commit_completed_turn` rejects
/// empty batches anyway) or when the turn was already committed. The
/// partial commit rides `commit_completed_turn`'s in-transaction
/// `expected_turn` CAS, so a racing full commit, a duplicate cancel, or
/// a stale-lease worker can never double-append the prefix. Token usage
/// is zeroed — the real usage was already audited on the per-attempt
/// rows of the cancelled attempt(s), so billing it again on the thread
/// aggregate would double-count. No lifecycle events are emitted: the
/// partial transcript already streamed as delta events, and the absence
/// of a `TurnComplete` / `Done` lets replay render this as a cancelled
/// turn.
pub(crate) async fn commit_partial_turn_on_cancel(
    params: PartialCancelCommit<'_>,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<Vec<llm::Message>> {
    let PartialCancelCommit {
        thread_id,
        task_id,
        candidate,
        expected_turn,
        agent_state_snapshot,
    } = params;

    let (prefix, suffix) = provider_valid_split(candidate);
    if prefix.is_empty() {
        // Nothing provider-valid to commit. Strict no-op: no attempt row,
        // no thread mutation.
        return Ok(suffix);
    }

    // Cheap pre-check; the in-transaction CAS below is the authority.
    ensure_turn_not_already_committed(deps.thread_store, thread_id, expected_turn).await?;

    // Open a synthetic attempt: `commit_completed_turn` unconditionally
    // closes an attempt, and by the time either seam reaches this point
    // every prior attempt for the task is already closed (mid-stream
    // cancel closed it `Cancelled`; a parked parent's attempt closed
    // `Success` at suspension). The row doubles as the audit record tying
    // the partial commit to the cancel.
    let existing = deps
        .attempt_store
        .list_by_task(task_id)
        .await
        .context("list attempts for cancel-commit")?;
    let attempt_number = u32::try_from(existing.len()).context("attempt count overflow")? + 1;
    let synthetic = deps
        .attempt_store
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number,
            provenance: AuditProvenance::new("cancel-commit", "cancel-commit"),
            request_blob: serde_json::json!({ "user_prompt": "<cancel-commit>" }),
            now,
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await
        .context("open synthetic cancel-commit attempt")?;

    let commit_result = commit_completed_turn(
        CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::CancelSalvage,
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            expected_turn,
            turn_attempt_id: synthetic.id.clone(),
            close_attempt_params: CloseAttemptParams {
                response_blob: serde_json::json!({ "partial_commit_on_cancel": true }),
                response_id: None,
                response_model: None,
                stop_reason: None,
                outcome: TurnAttemptOutcome::Cancelled,
                // Zero by design: this synthetic attempt exists only
                // because the commit transaction requires an attempt id
                // to close — the turn's REAL token usage already lives
                // on its real attempt rows, closed by the cancel path.
                // Repeating those numbers here would double-count in
                // every consumer that sums attempt rows (cost ledgers,
                // usage sweeps).
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
            },
            messages: prefix,
            // Zero for the same reason: `turn_usage` advances the
            // thread's aggregate token totals, which never included
            // uncommitted turns — the measured usage stays on the
            // attempt rows (the billing source of truth). Folding it in
            // here would change aggregate semantics and double-count
            // against attempt-summing readers.
            turn_usage: TokenUsage::default(),
            agent_state_snapshot,
            // Empty by design: the turn's real events (tool calls,
            // deltas) were committed INCREMENTALLY during streaming —
            // that persistence is the very thing this salvage relies
            // on. This field carries lifecycle events (`turn_complete`
            // etc.), and fabricating one would falsely mark a cancelled
            // turn as completed to every downstream consumer.
            events: Vec::new(),
            outbox_max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            owner_guard: None,
            now,
        },
        deps.thread_store,
        deps.message_store,
        deps.attempt_store,
        deps.checkpoint_store,
        deps.event_repo,
    )
    .await;
    if let Err(error) = commit_result {
        // This seam owns the synthetic attempt it just opened: a failed
        // commit (stale-turn CAS lost to a racing successor, store
        // error) must not leak it open forever on a soon-terminal task —
        // nothing sweeps orphaned attempts, and the cancel seam's
        // snapshot-scoped close was taken before this attempt existed.
        // Best-effort: the atomic backends roll the in-transaction close
        // back on failure (row still open), while the in-memory
        // non-atomic path may have already closed it (step 1) — that
        // close then rejects `AlreadyClosed` and is swallowed.
        close_attempt_with(
            &synthetic,
            TurnAttemptOutcome::Cancelled,
            deps.attempt_store,
            now,
        )
        .await;
        return Err(error.context("commit partial turn on cancel"));
    }

    Ok(suffix)
}

/// Best snapshot of the agent state for a cancelled turn's checkpoint:
/// the staged state if present, otherwise the recovery view's committed
/// snapshot. Falls back to the recovery snapshot if serialization fails.
fn staged_or_recovery_snapshot(inputs: &RootWorkerInputs) -> serde_json::Value {
    match inputs.staged_stores.state.snapshot_state() {
        Ok(Some(state)) => serde_json::to_value(&state)
            .unwrap_or_else(|_| inputs.recovery_view.agent_state_snapshot.clone()),
        _ => inputs.recovery_view.agent_state_snapshot.clone(),
    }
}

/// Seam B: the live worker loop's cancellation branch. Commit the
/// completed prefix of the cancelled turn from the in-memory staged
/// buffer, then let the caller propagate the original cancel error.
///
/// Guarded on a fresh re-read of the task row: only a task whose
/// durable status is already terminal — [`TaskStatus::Cancelled`]
/// (external `cancel_tree`) or [`TaskStatus::Failed`] (the host's
/// subagent-timeout path fails the row durably, then trips the same
/// per-task token) — may consume this turn number, because no future
/// lease holder can exist for a terminal row. A lease-lost requeue
/// (the row bounced back to `Pending` / was re-acquired) is left
/// untouched so the new lease holder owns turn `N`. Every error is
/// logged and swallowed — a partial-commit failure must never mask
/// the cancellation the caller is about to surface.
async fn commit_cancelled_partial_turn(
    inputs: &RootWorkerInputs,
    candidate: Vec<llm::Message>,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) {
    let task_id = &inputs.bootstrap.task_id;
    let thread_id = &inputs.bootstrap.thread_id;

    match deps.task_store.get(task_id).await {
        Ok(Some(task)) if matches!(task.status, TaskStatus::Cancelled | TaskStatus::Failed) => {}
        // Not terminal (lease-lost requeue), missing, or unreadable:
        // do not partial-commit — a re-run will own turn N.
        Ok(_) => return,
        Err(error) => {
            log::warn!("cancel-commit: re-read task {task_id} failed: {error:#}");
            return;
        }
    }

    let agent_state_snapshot = staged_or_recovery_snapshot(inputs);
    let suffix = match commit_partial_turn_on_cancel(
        PartialCancelCommit {
            thread_id,
            task_id,
            candidate,
            expected_turn: inputs.recovery_view.next_turn_number,
            agent_state_snapshot,
        },
        deps,
        now,
    )
    .await
    {
        Ok(suffix) => suffix,
        Err(error) => {
            log::warn!("cancel-commit: partial commit failed on thread {thread_id}: {error:#}");
            return;
        }
    };

    // A non-empty suffix is only reachable if a resume delta ends
    // unbalanced (defensive) — re-seed it so the next turn's backfill
    // closes it. Best-effort: the draft is a recovery aid.
    if !suffix.is_empty()
        && let Err(error) = deps.message_store.set_draft(thread_id, suffix, now).await
    {
        log::warn!("cancel-commit: re-seed draft failed on thread {thread_id}: {error:#}");
    }
}

/// Seam A: commit the completed prefix of a **parked** cancelled turn.
///
/// Handles only the states the external-cancel caller owns:
/// `WaitingOnChildren`, or `Pending` + [`TaskState::ReadyToResume`]. A
/// `Running` task is skipped — its live loop (seam B) owns the commit.
/// A parked task with no continuation (e.g. a `SubagentInvocation`
/// parent) or an empty candidate is a no-op. All errors are logged and
/// swallowed so the cancel always proceeds.
async fn best_effort_commit_parked_cancel(
    task: &AgentTask,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) {
    let parked = match task.status {
        TaskStatus::WaitingOnChildren => true,
        TaskStatus::Pending => matches!(task.state, TaskState::ReadyToResume { .. }),
        _ => false,
    };
    if !parked {
        return;
    }

    // A parked turn always carries a continuation; its embedded agent
    // state seeds the checkpoint. Absent (e.g. `SubagentInvocation`) →
    // there is no root-turn transcript to commit here.
    let Some(continuation) = task.state.continuation() else {
        return;
    };
    let agent_state_snapshot = match serde_json::to_value(&continuation.payload.state) {
        Ok(value) => value,
        Err(error) => {
            log::warn!(
                "cancel-commit: serialize agent state failed on thread {}: {error:#}",
                task.thread_id,
            );
            return;
        }
    };

    // Prefer the durable draft (richer); fall back to the parked
    // suspended_messages. Read BEFORE `cancel_tree` clears the state.
    let candidate = match deps.message_store.get(&task.thread_id).await {
        Ok(Some(projection)) if !projection.draft_messages.is_empty() => projection.draft_messages,
        Ok(_) => task.state.suspended_messages().to_vec(),
        Err(error) => {
            log::warn!(
                "cancel-commit: read draft failed on thread {}: {error:#}",
                task.thread_id,
            );
            task.state.suspended_messages().to_vec()
        }
    };
    if candidate.is_empty() {
        return;
    }

    let expected_turn = match deps.thread_store.get(&task.thread_id).await {
        Ok(Some(thread)) => thread.committed_turns.saturating_add(1),
        Ok(None) => return,
        Err(error) => {
            log::warn!(
                "cancel-commit: read thread failed on {}: {error:#}",
                task.thread_id,
            );
            return;
        }
    };

    let suffix = match commit_partial_turn_on_cancel(
        PartialCancelCommit {
            thread_id: &task.thread_id,
            task_id: &task.id,
            candidate,
            expected_turn,
            agent_state_snapshot,
        },
        deps,
        now,
    )
    .await
    {
        Ok(suffix) => suffix,
        Err(error) => {
            log::warn!(
                "cancel-commit: partial commit failed on thread {}: {error:#}",
                task.thread_id,
            );
            return;
        }
    };

    // Re-seed the draft with the dropped trailing `assistant + tool_use`
    // (the commit cleared the slot in-transaction). The next turn's
    // `backfill_orphaned_tool_results` closes it with
    // `USER_CANCELLED_TOOL_RESULT` without duplicating the committed
    // prefix. Best-effort — a crash here loses only that trailing
    // message, strictly better than today.
    if !suffix.is_empty()
        && let Err(error) = deps
            .message_store
            .set_draft(&task.thread_id, suffix, now)
            .await
    {
        log::warn!(
            "cancel-commit: re-seed draft failed on thread {}: {error:#}",
            task.thread_id,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────

/// Execute a root turn end to end.
///
/// Depending on the LLM response this either:
///
/// - **Text-only** (no tool calls): buffers messages, commits the turn,
///   and advances the task to `Completed`.
/// - **Tool handoff**: when tool-use blocks accompany `ToolUse` (or a
///   legacy missing) stop reason, builds a [`ContinuationEnvelope`],
///   spawns one `tool_runtime` child per tool call, and parks the parent
///   task in `WaitingOnChildren`.
///
/// # Errors
///
/// - LLM returns a non-success outcome (rate limit, server error, etc.)
/// - Commit path fails (text-only)
/// - Task completion fails (text-only)
/// - Child spawn fails (tool suspension)
///
/// [`ContinuationEnvelope`]: agent_sdk_foundation::ContinuationEnvelope
pub async fn execute_root_turn(
    inputs: RootWorkerInputs,
    user_input: impl Into<super::user_input::UserInput>,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    #[cfg(feature = "otel")]
    let started_at = std::time::Instant::now();

    let user_input = user_input.into();

    // Box-pin the inner future to keep the outer function's state
    // machine small — without this, clippy::large_futures fires on
    // every test that calls execute_root_turn because the outer
    // future stores both itself and the inner future inline.
    let result = Box::pin(execute_root_turn_inner(
        inputs, user_input, provider, deps, now,
    ))
    .await;

    #[cfg(feature = "otel")]
    {
        let metrics = crate::observability::ServerMetrics::global();
        let outcome = match &result {
            Ok(RootTurnOutcome::Completed { .. }) => crate::observability::attrs::OUTCOME_DONE,
            Ok(RootTurnOutcome::Suspended { .. }) => crate::observability::attrs::OUTCOME_SUSPENDED,
            Err(_) => crate::observability::attrs::OUTCOME_ERROR,
        };
        metrics.record_task_execution(
            crate::observability::attrs::KIND_ROOT,
            outcome,
            started_at.elapsed().as_secs_f64(),
        );
    }

    result
}

/// Inner body of [`execute_root_turn`].  Kept separate so the outer
/// function can wrap the entire call in a metric-recording shim
/// without having to thread a stopwatch through every early return.
#[allow(clippy::too_many_lines)]
async fn execute_root_turn_inner(
    inputs: RootWorkerInputs,
    user_input: super::user_input::UserInput,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let definition = inputs.definition();
    let thread_id = &inputs.bootstrap.thread_id;

    // 1. Open turn attempt. The audit blob stores a string projection
    //    of the user input — image / document blocks render as
    //    `[<media_type> attachment]` placeholders so journal replay
    //    keeps something descriptive without bloating the audit row
    //    with binary payloads.
    let audit_prompt = user_input.audit_summary();

    // Open the root `invoke_agent` span for the turn, continuing the
    // inbound client trace when the gRPC caller propagated one (see
    // `AgentTask::otel_traceparent`), so every `chat` call and
    // `execute_tool` execution across the suspend/resume hop forms one
    // coherent trace. The span's ids are persisted on the first attempt
    // (below) so the resume path and child tool tasks re-parent under it;
    // the live span is stashed in a process-global registry so the
    // terminal path finalizes it with the correct full turn duration.
    #[cfg(feature = "otel")]
    let root_otel_ids: Option<(String, String)> = {
        use agent_sdk::observability::loop_instrument;
        let parent = inputs
            .bootstrap
            .task
            .otel_traceparent
            .as_deref()
            .and_then(loop_instrument::context_from_traceparent);
        let conversation_id = thread_id.to_string();
        let model = provider.model().to_string();
        let started = {
            let _parent_guard = parent.map(opentelemetry::Context::attach);
            loop_instrument::start_root_turn_span(loop_instrument::RootTurnSpanParams {
                provider_id: provider.provider(),
                model: &model,
                conversation_id: &conversation_id,
            })
        };
        let ids = (started.trace_id_hex.clone(), started.span_id_hex.clone());
        loop_instrument::stash_root_turn_span(inputs.bootstrap.task_id.as_str(), started.span);
        Some(ids)
    };
    #[cfg(not(feature = "otel"))]
    let root_otel_ids: Option<(String, String)> = None;

    let attempt = open_attempt(
        &inputs,
        definition,
        &audit_prompt,
        deps.attempt_store,
        now,
        root_otel_ids.clone(),
    )
    .await
    .context("open turn attempt")?;

    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global()
        .record_task_acquired(crate::observability::attrs::KIND_ROOT);

    // 2. Commit the durable user-input event for this turn,
    //    once-per-task (gated on `attempt_number == 1`). The
    //    projection's user-role message for this prompt is
    //    written downstream, but it carries no sequence and gets
    //    commingled with tool-result / compaction-summary
    //    user-role rows on replay. Persisting the prompt as a
    //    first-class event gives replay clients a clean,
    //    chronological signal that's anchored to the event log
    //    just like every other observable. Skipped for resume
    //    inputs (they have no admitted prompt to surface).
    let user_input_committed =
        if attempt.attempt_number == 1 && !user_input.is_resume() && !user_input.is_empty() {
            let user_input_event =
                AgentEvent::user_input(thread_id.clone(), user_input.blocks().to_vec());
            let committed = deps
                .event_repo
                .commit_event(thread_id, user_input_event, now)
                .await
                .context("commit user_input event")?;
            deps.event_notifier.notify(std::slice::from_ref(&committed));
            Some(committed)
        } else {
            None
        };

    // 3. Commit the `Start` event NOW, before streaming begins, so
    //    later TextDelta / ThinkingDelta events have a parent in the
    //    journal.  This trades the previous "Start committed atomically
    //    with the rest of the turn" guarantee for live streaming: if
    //    the LLM call fails (rate limit, transport error, cancel) the
    //    Start is left orphaned and replay clients see
    //    `Start … Error` (Error from `fail_root_turn`) or
    //    `Start … <task Cancelled>`.  Both shapes are interpretable as
    //    an abandoned turn; the next attempt for this task will commit
    //    a fresh Start with the next sequence number.
    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);
    let start_committed = deps
        .event_repo
        .commit_event(
            thread_id,
            AgentEvent::start(thread_id.clone(), turn_number),
            now,
        )
        .await
        .context("commit start event")?;
    deps.event_notifier
        .notify(std::slice::from_ref(&start_committed));

    // 2.4 Backfill any tool_use loop the prior turn left open.
    //
    //     A fresh root turn executes only after the previous root turn
    //     on this thread is terminal (`promote_next_root` serialises one
    //     root turn per thread), so nothing is legitimately awaiting a
    //     tool result here. `recover_thread` preserves the raw suspension
    //     draft verbatim, which may end in an assistant `tool_use` whose
    //     `tool_result`s never landed — the user answered one question
    //     and cancelled the rest, or the prior turn was cancelled. Those
    //     unanswered `tool_use` blocks are abandoned and are closed
    //     durably now (not patched into the outgoing request), so an
    //     orphaned `tool_use` can never reach a provider and the thread
    //     is balanced the moment it is next loaded.
    backfill_orphaned_tool_results(deps, &inputs.staged_stores.messages, thread_id, now)
        .await
        .context("pre-call orphaned tool_use backfill")?;

    // 2.5 Pre-call auto-compaction.
    //
    //     Compaction operates on the staged history (which excludes
    //     the fresh user prompt — that lives only inside
    //     `chat_request.messages` after `build_chat_request` runs)
    //     so the durable projection can be safely rewritten without
    //     duplicating the user prompt at commit time. No-op when
    //     `deps.compaction_config` is `None`.
    super::compaction::maybe_compact_staged_history(
        deps,
        &inputs.staged_stores.messages,
        thread_id,
        now,
    )
    .await
    .context("pre-call auto-compaction")?;

    // 3. Build, send LLM request, and resolve the outcome — closing
    //    the attempt on any non-success path.  `call_llm` allocates
    //    per-block message/thinking IDs lazily as deltas arrive and
    //    returns them so the consolidated content events emitted at
    //    turn close reuse the same IDs.
    let chat_request = build_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
        &user_input,
        inputs.bootstrap.task.caller_metadata.as_ref(),
    )
    .await
    .context("build chat request")?;

    let llm_call = call_llm_with_retry(LlmRetryParams {
        inputs: &inputs,
        definition,
        user_input: &user_input,
        attempt_audit_prompt: &audit_prompt,
        provider,
        chat_request,
        initial_attempt: attempt,
        deps,
        thread_id,
        now,
    });
    // Re-parent the turn's `chat` span(s) under the root `invoke_agent`
    // span. `with_context` propagates the parent per-poll — a held
    // `ContextGuard` across this `.await` would make the worker future
    // `!Send` and unspawnable.
    #[cfg(feature = "otel")]
    let streamed = {
        use opentelemetry::trace::FutureExt;
        match root_otel_ids.as_ref().and_then(|(trace, span)| {
            agent_sdk::observability::loop_instrument::remote_parent_context(trace, span)
        }) {
            Some(cx) => llm_call.with_context(cx).await,
            None => llm_call.await,
        }
    };
    #[cfg(not(feature = "otel"))]
    let streamed = llm_call.await;

    let StreamedTurn {
        response,
        content_ids,
        attempt,
    } = match streamed {
        Ok(streamed) => streamed,
        Err(error) => {
            // Seam B (fresh turn): on a mid-stream (or between-retry)
            // cancel, commit the completed prefix from the staged buffer
            // before the task goes terminal, then propagate the original
            // error unchanged. The candidate is the user prompt (`Some`
            // on non-resume turns) plus any staged post-seed delta, so
            // the agent at least remembers what was asked.
            if is_root_turn_cancelled(&error) {
                let mut candidate: Vec<llm::Message> =
                    user_input.clone().into_message().into_iter().collect();
                match inputs.staged_stores.messages.snapshot_appended_messages() {
                    Ok(delta) => candidate.extend(delta),
                    Err(snapshot_err) => log::warn!(
                        "cancel-commit: snapshot staged delta failed on thread {}: \
                         {snapshot_err:#}",
                        inputs.bootstrap.thread_id,
                    ),
                }
                commit_cancelled_partial_turn(&inputs, candidate, deps, now).await;
            }
            return Err(error);
        }
    };

    // Capture a post-LLM timestamp so the turn attempt's duration_ms
    // reflects actual wall-clock latency instead of always being 0.
    let commit_now = OffsetDateTime::now_utc();
    let close_ctx = TurnCloseContext {
        content_ids,
        user_input_committed,
        start_committed,
    };

    // 4. Branch: authorized tool handoff → suspend; otherwise commit.
    //
    // Start was committed in step 2 (before streaming).  The branches
    // commit only the consolidated content events, `TurnComplete`, and
    // `Done` — atomically, with the message-projection write.
    if response_requests_tool_dispatch(&response) {
        return suspend_at_tool_boundary(
            inputs,
            &user_input,
            response,
            attempt,
            deps,
            commit_now,
            close_ctx,
        )
        .await;
    }

    // ── Text-only path (Phase 4.3) ──────────────────────────────
    commit_text_only_turn(
        inputs,
        &user_input,
        response,
        attempt,
        deps,
        commit_now,
        close_ctx,
    )
    .await
}

/// Complete and commit a text-only turn (no tool calls).
///
/// The IDs in `close_ctx.content_ids` are reused for the consolidated
/// `Text` / `Thinking` events committed alongside `TurnComplete` /
/// `Done`, matching the ids used for the per-delta events streamed
/// from `call_llm`.
async fn commit_text_only_turn(
    inputs: RootWorkerInputs,
    user_input: &super::user_input::UserInput,
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
    close_ctx: TurnCloseContext,
) -> Result<RootTurnOutcome> {
    let thread_id = &inputs.bootstrap.thread_id;
    let task_id = &inputs.bootstrap.task_id;
    let response_text = response.first_text().unwrap_or("").to_owned();

    buffer_turn_messages(
        &inputs.staged_stores.messages,
        &inputs.staged_stores.state,
        thread_id,
        user_input,
        &response,
    )
    .await
    .context("buffer staged messages")?;

    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);
    let close_params = build_close_params(&response, &attempt);
    let turn_usage = response_token_usage(&response);

    let drained_messages = inputs
        .staged_stores
        .messages
        .drain_messages()
        .context("drain staged messages")?;
    let drained_state = inputs
        .staged_stores
        .state
        .drain_state()
        .context("drain staged state")?
        .context("staged state was None after turn")?;
    let agent_state_snapshot =
        serde_json::to_value(&drained_state).context("serialize agent state")?;

    // Start was committed by `execute_root_turn` before streaming so
    // the per-delta TextDelta / ThinkingDelta events have a parent in
    // the journal.  This vec only carries the consolidated content
    // events plus the `TurnComplete` / `Done` lifecycle edges.
    let duration = (now - attempt.opened_at).unsigned_abs();
    let lifecycle_events = build_turn_complete_events(
        &response,
        thread_id,
        turn_number,
        &turn_usage,
        &drained_state.total_usage,
        duration,
        &close_ctx.content_ids,
    );

    let commit = commit_completed_turn_shifting_slot(
        CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            expected_turn: inputs.recovery_view.next_turn_number,
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params,
            messages: drained_messages,
            turn_usage,
            agent_state_snapshot,
            events: lifecycle_events,
            outbox_max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            owner_guard: None,
            now,
        },
        &inputs.bootstrap.worker_id,
        &inputs.bootstrap.lease_id,
        deps,
    )
    .await
    .context("commit completed turn")?;

    // Prepend the durable per-turn admission events (`UserInput`
    // when present + `Start`) committed before streaming so the
    // outcome's `committed_events` represents every event committed
    // for this turn (matching the pre-streaming contract).
    let mut committed_events: Vec<CommittedEvent> = Vec::new();
    committed_events.extend(close_ctx.user_input_committed);
    committed_events.push(close_ctx.start_committed);
    committed_events.extend(commit.committed_events.iter().cloned());

    let (completed_task, resumed_invocation) = deps
        .task_store
        .complete_task(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            now,
        )
        .await
        .context("complete root task")?;

    // When this root turn is a subagent's child root, its terminal
    // transition resumes the linked subagent invocation task
    // (`WaitingOnChildren` → `Pending`) in the same locked scope. Nudge
    // a parked worker so it claims the now-runnable invocation
    // immediately instead of waiting out the acquisition ticker. A
    // `None` (no linked invocation) or non-runnable row leaves the poll
    // backstop to handle it.
    if resumed_invocation.is_some_and(|t| t.status.is_runnable())
        && let Some(signal) = deps.wakeup
    {
        signal.notify_workers();
    }

    // Text-only completion: the turn is done, so finalize the root
    // `invoke_agent` span with its full duration + outcome. Usage is
    // `None` (the per-call `chat` spans carry authoritative usage).
    #[cfg(feature = "otel")]
    agent_sdk::observability::loop_instrument::finalize_root_turn_span(
        task_id.as_str(),
        turn_number + 1,
        None,
        "done",
    );

    Ok(RootTurnOutcome::Completed {
        commit: Box::new(commit),
        completed_task,
        response_text,
        committed_events,
    })
}

/// Idempotency guard: bail if a prior worker already committed the
/// expected turn on this thread.
///
/// Called from both the fresh-turn and resumed-turn commit paths so a
/// stale-lease worker that re-acquired the task cannot double-commit
/// after another worker already advanced the thread.
async fn ensure_turn_not_already_committed(
    thread_store: &dyn ThreadStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    expected_turn: u32,
) -> Result<()> {
    let current_thread = thread_store
        .get(thread_id)
        .await
        .context("re-read thread for idempotency check")?
        .context("thread disappeared during turn execution")?;

    if current_thread.committed_turns >= expected_turn {
        return Err(anyhow::Error::new(StaleTurnCommit {
            expected_turn,
            committed_turns: current_thread.committed_turns,
        })
        .context(format!(
            "turn {expected_turn} was already committed on thread {thread_id} \
             (committed_turns={}); skipping duplicate commit",
            current_thread.committed_turns,
        )));
    }

    Ok(())
}

const MAX_TURN_SLOT_SHIFTS: u32 = 3;

/// Does this error carry the completed-turn slot CAS rejection?
///
/// All three backends (and the [`ensure_turn_not_already_committed`]
/// pre-check) attach a typed [`StaleTurnCommit`] root cause, so this is
/// a downcast instead of message-string matching. A
/// [`LostCommitOwnership`](crate::journal::commit::LostCommitOwnership)
/// rejection from an owner-guarded retry is deliberately NOT a
/// collision: the cancel / timeout / requeue owner has the row, and
/// the shift loop must propagate it as terminal.
fn is_turn_slot_collision(error: &anyhow::Error) -> bool {
    error.downcast_ref::<StaleTurnCommit>().is_some()
}

/// Rewrite the turn index carried by the batch's turn-boundary
/// lifecycle events after a slot shift, so `TurnComplete` / `Done`
/// agree with the turn number the commit actually lands on.
fn remap_turn_indexed_events(events: &mut [AgentEvent], turn: usize) {
    for event in events {
        match event {
            AgentEvent::TurnComplete {
                turn: event_turn, ..
            } => *event_turn = turn,
            AgentEvent::Done { total_turns, .. } => *total_turns = turn,
            _ => {}
        }
    }
}

/// Commit a completed turn, shifting past a **cross-task** turn-slot
/// collision instead of failing the turn (issue #354, residual 5).
///
/// The race: cancelling a `Running` root frees the thread's
/// active-root slot and promotes a queued successor in the same store
/// transaction, but the cancelled worker's seam-B salvage still
/// commits into the turn slot it bootstrapped with. A successor that
/// bootstrapped before that salvage landed pins the **same**
/// `expected_turn`, and the commit CAS fails whichever lands second.
/// The losing salvage is already dropped silently (benign); before
/// this wrapper, a losing successor failed **terminally** and had to
/// be resubmitted.
///
/// On a slot-CAS rejection this wrapper distinguishes the two cases
/// the CAS conflates:
///
/// - **Cross-task collision** — the checkpoint occupying
///   `expected_turn` belongs to a *different* task AND this worker
///   still owns its live `Running` row (status + `(worker, lease)`
///   re-read). The turn's work is valid, only its slot number is
///   stale: shift `expected_turn` forward ONE slot, remap the
///   turn-indexed lifecycle events, and retry (bounded by
///   [`MAX_TURN_SLOT_SHIFTS`]). Single-stepping matters — if the
///   thread advanced several turns, each intervening occupant is
///   validated by its own shift round instead of being skipped.
/// - **Same-task duplicate / lost ownership** — the occupying
///   checkpoint is this task's own commit (stale-lease double
///   commit), the task is no longer a live `Running` row this worker
///   owns (e.g. it was cancelled — its salvage must never re-land as
///   a full turn), or the collided slot has no checkpoint to inspect.
///   The original CAS error propagates unchanged.
///
/// The retry is safe because a rejected commit leaves no side effects
/// on any backend: the in-memory guard runs before the first
/// projection write, and the durable committers roll the whole
/// transaction back (the attempt row stays open for the retry to
/// close).
///
/// # Ownership is re-validated INSIDE the retried commit
///
/// The eligibility re-read in [`shifted_turn_slot`] and the retried
/// commit are not atomic — a cancellation or lease loss can land
/// between them. The retry therefore carries
/// [`CompletedTurnCommit::owner_guard`]: the durable committers
/// re-read the task row inside the commit transaction (`FOR UPDATE`
/// on Postgres) and reject with
/// [`LostCommitOwnership`](crate::journal::commit::LostCommitOwnership)
/// unless the row is still `Running` under this `(worker, lease)` —
/// so a shifted commit can never splice a dead root's turn into the
/// slot on a durable backend. The non-atomic in-memory backend cannot
/// enforce the guard transactionally (no cross-store transaction);
/// there the pre-retry re-read is the strongest available check and a
/// residual in-process window remains — the same class as every other
/// non-atomic in-memory commit step. Closing that generally (CAS task
/// ownership inside every completed-turn commit) is a store-level
/// follow-up; it was deliberately scoped to the shifted retry here
/// because unconditional fencing would discard fully-billed answers
/// on benign cancel-after-completion races.
///
/// # Event turn indices after a shift
///
/// The turn's `Start` (and any streamed deltas) were durably committed
/// BEFORE the collision was detected, carrying the original slot
/// number, while the remapped `TurnComplete` / `Done` carry the landed
/// slot — a shifted turn's journal reads `Start{N} … TurnComplete{N+1}
/// / Done{N+1}`. Consumers must derive turn boundaries from event
/// ORDER, not from the turn indices (see the `events.proto` contract).
///
/// # The shifted answer did not see the salvaged prefix — accepted
///
/// In this race the successor built its LLM request before the salvage
/// landed, so the answer it commits never saw the cancelled turn's
/// salvaged prefix. That gap is inherent to the RACE, not to the
/// shift: in the mirror ordering — the successor's commit lands first —
/// the losing salvage is dropped silently and the prefix never enters
/// the transcript at all, with the very same one-answer visibility
/// gap. Shifting is strictly more preserving than that (benign)
/// ordering: the prefix survives at its own slot, attributed to the
/// cancelled attempt, and the successor's checkpoint — rebuilt from
/// the projection read fresh inside the commit — includes it for every
/// future bootstrap. Only the single racing answer predates it.
///
/// The two rejected cures are worse than the disease: re-executing the
/// successor against the salvaged head would duplicate every
/// non-idempotent tool effect the turn already executed, and deferring
/// successor execution until the salvage lands is unbounded (the
/// cancelled worker may be gone; only the deadline sweep eventually
/// reaps it).
///
/// # Cancelled state is audit, not lineage
///
/// The salvage checkpoint's agent-state snapshot (the cancelled turn's
/// staged state: accumulated cost, metadata mutations, any parked
/// continuation) is an audit artifact preserved at its own slot, not
/// forward state lineage. Cancellation MEANS the turn's uncommitted
/// state effects are discarded — its parked children were cancelled
/// with it and must never be resumed — so the successor's snapshot
/// superseding it as latest is the cancel semantics, not corruption.
/// Billing truth is unaffected: the cancelled turn's real usage lives
/// on its attempt rows and its `Cancelled` event, and the thread
/// aggregate advanced by zero for the salvage by construction.
pub(crate) async fn commit_completed_turn_shifting_slot(
    mut params: CompletedTurnCommit,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    deps: &RootTurnDeps<'_>,
) -> Result<CommitOutcome> {
    let mut shifts = 0u32;
    loop {
        let precheck = ensure_turn_not_already_committed(
            deps.thread_store,
            &params.thread_id,
            params.expected_turn,
        )
        .await;
        let error = match precheck {
            Ok(()) => {
                match commit_completed_turn(
                    params.clone(),
                    deps.thread_store,
                    deps.message_store,
                    deps.attempt_store,
                    deps.checkpoint_store,
                    deps.event_repo,
                )
                .await
                {
                    Ok(outcome) => return Ok(outcome),
                    Err(error) => error,
                }
            }
            Err(error) => error,
        };

        if !is_turn_slot_collision(&error) {
            // Non-collision commit failure. An ownership rejection
            // (`LostCommitOwnership`) settles here — no later path owns
            // the attempt; every other failure leaves the row owned,
            // and the host's terminal envelope closes the attempt.
            settle_attempt_after_lost_ownership(&error, &params, deps).await;
            return Err(error);
        }
        if shifts >= MAX_TURN_SLOT_SHIFTS {
            // Walk exhausted on a genuine collision (codex round-8
            // P2): ownership may have been lost concurrently WITHOUT
            // the error carrying `LostCommitOwnership` — the cancel
            // seam leaves a live worker's attempt open and the host
            // skips rows it no longer owns. Settle unconditionally,
            // exactly like the no-shift exit below.
            settle_own_attempt(&params, deps).await;
            return Err(error);
        }
        let Some(next_turn) = shifted_turn_slot(&params, worker_id, lease_id, deps).await else {
            // No-shift exit (codex round-7 P2): when the eligibility
            // re-read found the row cancelled or requeued, no later
            // path owns this attempt — the cancel seam deliberately
            // left the live worker's attempt alone and the host skips
            // rows it no longer owns. Settle unconditionally: in the
            // still-owned refusal cases the host's terminal path would
            // close it as a zero-usage best-effort Cancelled anyway
            // (this close carries the REAL usage, and the later
            // best-effort close is swallowed as `AlreadyClosed`), and
            // in the requeue case the next execution's leftover settle
            // becomes a no-op.
            settle_own_attempt(&params, deps).await;
            return Err(error);
        };
        log::info!(
            "turn-slot collision on thread {}: turn {} was consumed by another task's \
             commit; shifting task {} to turn {next_turn}",
            params.thread_id,
            params.expected_turn,
            params.task_id,
        );
        params.expected_turn = next_turn;
        remap_turn_indexed_events(&mut params.events, usize::try_from(next_turn).unwrap_or(0));
        // The snapshot's turn_count was built for the original slot;
        // recovery seeds staged state from the landed checkpoint, so a
        // stale count would leave every later turn one behind
        // Thread::committed_turns and Done.total_turns (codex round-19).
        if let Some(turn_count) = params
            .agent_state_snapshot
            .get_mut("turn_count")
            .filter(|value| value.is_u64())
        {
            *turn_count = serde_json::json!(next_turn);
        }
        // Every shifted retry is owner-guarded: the durable committers
        // re-validate (Running, worker, lease) on the task row inside
        // the commit transaction, closing the guard→retry TOCTOU.
        params.owner_guard = Some(CommitOwnerGuard {
            worker_id: worker_id.clone(),
            lease_id: lease_id.clone(),
        });
        shifts += 1;
    }
}

/// Best-effort settle of the in-flight attempt when the owner-guarded
/// shifted retry is rejected with
/// [`LostCommitOwnership`](crate::journal::commit::LostCommitOwnership)
/// (codex round-6 P2).
///
/// The rejected durable commit rolled its in-transaction attempt close
/// back, and no later path will close this attempt: the cancel seam's
/// snapshot-scoped close ran while the task was still `Running` and
/// deliberately left the live worker's attempt alone, and the host's
/// failure path skips a row it no longer owns. Errors without an
/// ownership rejection in their chain are left alone — the task is
/// still owned there, and the host's terminal or requeue path owns
/// the attempt's fate.
pub(crate) async fn settle_attempt_after_lost_ownership(
    error: &anyhow::Error,
    params: &CompletedTurnCommit,
    deps: &RootTurnDeps<'_>,
) {
    let lost_ownership = error.chain().any(|cause| {
        cause
            .downcast_ref::<crate::journal::commit::LostCommitOwnership>()
            .is_some()
    });
    if !lost_ownership {
        return;
    }
    settle_own_attempt(params, deps).await;
}

/// Best-effort close of THIS execution's attempt — the id was opened
/// by this worker, so it cannot belong to a replacement worker. The
/// close carries the REAL token usage the commit would have recorded
/// (attempt rows are the billing source of truth) under the
/// `Cancelled` outcome, since the turn did not commit. A concurrent
/// close (the cancel path, or the host's terminal envelope, racing
/// this settle) makes the store reject `AlreadyClosed`, which is
/// swallowed as usual for best-effort closes.
pub(crate) async fn settle_own_attempt(params: &CompletedTurnCommit, deps: &RootTurnDeps<'_>) {
    let close = CloseAttemptParams {
        outcome: TurnAttemptOutcome::Cancelled,
        ..params.close_attempt_params.clone()
    };
    if let Err(close_error) = deps
        .attempt_store
        .close_attempt(&params.turn_attempt_id, close, params.now)
        .await
    {
        log::warn!(
            "settle own attempt after slot-collision exit: closing attempt {} failed: {close_error:#}",
            params.turn_attempt_id,
        );
    }
}

/// Decide whether a slot-CAS rejection is a shiftable cross-task
/// collision, and if so return the turn number to retry on.
///
/// Returns `None` — "do not shift, propagate the original error" — on
/// any read failure or when the eligibility conditions fail (see
/// [`commit_completed_turn_shifting_slot`]). Read errors are logged:
/// the caller still surfaces the original CAS error, which is the
/// meaningful one.
async fn shifted_turn_slot(
    params: &CompletedTurnCommit,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    deps: &RootTurnDeps<'_>,
) -> Option<u32> {
    // 1. This worker must still own a live `Running` row. A cancelled
    //    or requeued task must never shift: its in-flight work lost
    //    the thread the moment it lost the row, and re-landing it
    //    after the collision would splice a dead root's turn into a
    //    successor's history. This re-read is only the ELIGIBILITY
    //    check — it is not atomic with the retried commit. The
    //    authoritative enforcement is the `owner_guard` the retry
    //    carries: the durable committers re-validate (Running, worker,
    //    lease) on the task row INSIDE the commit transaction. On the
    //    non-atomic in-memory backend this re-read is the strongest
    //    available check (documented residual window).
    let own_row = match deps.task_store.get(&params.task_id).await {
        Ok(row) => row,
        Err(error) => {
            log::warn!(
                "turn-slot shift: re-read of task {} failed: {error:#}",
                params.task_id,
            );
            return None;
        }
    };
    let still_owned = own_row.as_ref().is_some_and(|row| {
        row.status == TaskStatus::Running
            && row.worker_id.as_ref() == Some(worker_id)
            && row.lease_id.as_ref() == Some(lease_id)
    });
    if !still_owned {
        return None;
    }

    // 2. The commit occupying our slot must belong to a DIFFERENT
    //    task. Our own task id there means this call is a duplicate of
    //    a commit that already landed — rejecting is the idempotency
    //    guarantee the CAS exists for.
    //
    // The read synchronizes on the sequential-commit lock when the
    // backend exposes one (codex round-14): the in-memory committer
    // holds that lock from the stale-turn pre-check through the
    // checkpoint write, so acquiring it here guarantees any commit
    // whose aggregate advance we lost to has finished writing its
    // checkpoint — a `None` under the lock is authoritative, not a
    // visibility gap. Durable backends return no lock; their atomic
    // transaction gives the same read consistency.
    let occupant = {
        let _commit_guard = match deps.thread_store.sequential_commit_lock() {
            Some(lock) => Some(lock.lock().await),
            None => None,
        };
        match deps
            .checkpoint_store
            .get_by_turn(&params.thread_id, params.expected_turn)
            .await
        {
            Ok(occupant) => occupant,
            Err(error) => {
                log::warn!(
                    "turn-slot shift: checkpoint lookup for turn {} on thread {} failed: {error:#}",
                    params.expected_turn,
                    params.thread_id,
                );
                return None;
            }
        }
    };
    match occupant {
        // Only a foreign CANCEL-SALVAGE checkpoint is shift-eligible.
        // The discriminator is the checkpoint's durable `kind`, never
        // its token usage: providers may omit usage deltas, so a real
        // completion can legitimately land an all-zero `turn_usage`,
        // and mistaking it for salvage would overwrite its state with
        // our stale snapshot. Salvage checkpoints carry zero aggregate
        // usage by construction (see `commit_partial_turn_on_cancel`),
        // which is what keeps the shift state-compatible: our snapshot
        // and `Done` totals, built before the collision, remain
        // correct. A FULL foreign turn in our slot means a live
        // predecessor's completion landed after we bootstrapped — our
        // cumulative state lacks its usage/cost, so shifting would
        // durably drop it; refuse and let the collision surface. The
        // host's collision handler requeues the refused task (via
        // `AgentTaskStore::requeue_owned_task`, sweep-identical budget
        // accounting) so it re-runs from the fresh committed head
        // instead of failing terminally.
        Some(checkpoint)
            if checkpoint.task_id != params.task_id
                && checkpoint.kind == CheckpointKind::CancelSalvage => {}
        // Full-turn foreign occupant, no checkpoint (thread advanced
        // through a path we don't understand), or our own duplicate —
        // do not shift.
        _ => return None,
    }

    // 3. Advance exactly ONE slot. The thread may have advanced
    //    further (two late commits can land back-to-back on the
    //    non-atomic in-memory backend), but jumping straight to the
    //    head would skip validating the intervening occupants — one of
    //    them could be a billed full turn the shift must refuse.
    //    Landing on the very next slot re-collides when it is also
    //    occupied, and the retry loop re-enters this function to
    //    validate THAT occupant, so every skipped slot is inspected
    //    (bounded by `MAX_TURN_SLOT_SHIFTS`).
    params.expected_turn.checked_add(1)
}

/// Project [`llm::Usage`] from a chat response into the SDK's
/// [`TokenUsage`] type used by the journal and audit records.
const fn response_token_usage(response: &llm::ChatResponse) -> TokenUsage {
    TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

async fn open_attempt(
    inputs: &RootWorkerInputs,
    definition: &AgentDefinition,
    user_prompt: &str,
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
    otel_ids: Option<(String, String)>,
) -> Result<TurnAttempt> {
    let task_id = &inputs.bootstrap.task_id;
    let provenance = AuditProvenance::new(&definition.provider, &definition.model);
    let request_blob = serde_json::json!({
        "user_prompt": user_prompt,
        "system_prompt_len": definition.system_prompt.len(),
    });

    let existing = attempt_store
        .list_by_task(task_id)
        .await
        .context("list existing attempts")?;
    // Leftover OPEN attempts from a predecessor execution are left
    // alone (codex round-9 P2): after a lease expiry, the old worker
    // may still be live — executing, unaware of its heartbeat
    // rejection — and closing its attempt here would permanently
    // record zero usage where the provider's real billed usage
    // belongs (its own close then loses to `AlreadyClosed`). The old
    // worker's own exit paths settle the attempt with real usage; a
    // hard-crashed worker's attempt stays open, the same pre-existing
    // audit residual every crash-requeue has. Duplicate-commit
    // protection does not depend on closing here: the expected-turn
    // CAS inside `ThreadStore::commit_turn` rejects a stale late
    // commit atomically on every backend.
    let attempt_number = u32::try_from(existing.len()).context("attempt count overflow")? + 1;

    let (otel_trace_id, otel_span_id) = match otel_ids {
        Some((trace_id, span_id)) => (Some(trace_id), Some(span_id)),
        None => (None, None),
    };

    attempt_store
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number,
            provenance,
            request_blob,
            now,
            // The root `invoke_agent` span's ids, persisted on the first
            // attempt so the resume path and child tool tasks can
            // re-parent under it across the daemon's task hops (see
            // `execute_root_turn_inner`). `None` on retry / resume
            // attempts and when the worker runs without an OTel exporter.
            otel_trace_id,
            otel_span_id,
        })
        .await
        .context("open_attempt via store")
}

/// Load the root `invoke_agent` span's `(trace_id, span_id)` for a turn
/// from its first attempt (where `execute_root_turn_inner` persisted
/// them), so the resume path can re-parent the resumed `chat` span under
/// the turn root across the task hop. Returns `None` when no attempt
/// carries them (a turn that ran without an `OTel` exporter, or a
/// pre-migration row).
#[cfg(feature = "otel")]
async fn load_root_span_ids(
    attempt_store: &dyn TurnAttemptStore,
    task_id: &AgentTaskId,
) -> Option<(String, String)> {
    let attempts = attempt_store.list_by_task(task_id).await.ok()?;
    attempts.into_iter().find_map(
        |attempt| match (attempt.otel_trace_id, attempt.otel_span_id) {
            (Some(trace_id), Some(span_id)) => Some((trace_id, span_id)),
            _ => None,
        },
    )
}

async fn build_chat_request(
    definition: &AgentDefinition,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    user_input: &super::user_input::UserInput,
    caller_metadata: Option<&serde_json::Value>,
) -> Result<ChatRequest> {
    // Get existing message history from staged store.
    let mut messages = staged_messages
        .get_history(thread_id)
        .await
        .context("get staged history")?;

    // Append the new user message. `user_input` carries the typed
    // block list (text + image + document) so the wire format
    // preserves binary attachments end-to-end. Resume inputs (which
    // have no fresh user prompt) skip the append because the staged
    // store already contains every relevant message.
    if let Some(message) = user_input.clone().into_message() {
        messages.push(message);
    }

    let thinking = match &definition.thinking {
        ThinkingPolicy::Disabled => None,
        ThinkingPolicy::Enabled { budget_tokens } => Some(llm::ThinkingConfig::new(*budget_tokens)),
        ThinkingPolicy::Adaptive { effort } => {
            let mut cfg = llm::ThinkingConfig::adaptive();
            if let Some(e) = effort {
                cfg = cfg.with_effort(*e);
            }
            Some(cfg)
        }
    };

    // Resolve tool list: `tools_fn` takes precedence over `tools` when
    // set AND caller_metadata is present, enabling per-turn filtering
    // based on caller identity.
    let resolved_tools = definition.resolve_tools(caller_metadata);

    Ok(ChatRequest {
        system: definition.system_prompt.clone(),
        messages,
        tools: if resolved_tools.is_empty() {
            None
        } else {
            Some(resolved_tools)
        },
        max_tokens: definition.max_tokens,
        max_tokens_explicit: true,
        // Not yet wired — session/cache affinity requires provider-side
        // state that the server doesn't manage in Phase 4.3.
        session_id: None,
        cached_content: None,
        thinking,
        tool_choice: None,
        response_format: None,
        cache: None,
    })
}

/// Durably close any `tool_use` loop left open in the recovered history
/// before a fresh turn builds its request.
///
/// [`recover_thread`] deliberately preserves the raw suspension draft (its
/// module doc delegates "dangling tool-use repair" to the caller). That
/// draft may end in an assistant `tool_use` whose `tool_result`s never
/// landed — the user answered one question and cancelled the rest, or the
/// prior turn was cancelled / abandoned. Because a fresh root turn only
/// runs once the prior turn on the thread is terminal, no result is
/// legitimately pending, so every unanswered `tool_use` is closed with a
/// [`USER_CANCELLED_TOOL_RESULT`](agent_sdk_foundation::llm::USER_CANCELLED_TOOL_RESULT)
/// error result.
///
/// The repair is written through to the durable projection (mirroring the
/// pre-call compaction rewrite: `replace_history` + `clear_draft`) rather
/// than patched into the outgoing request, so the thread is balanced the
/// moment it is next loaded and an orphaned `tool_use` can never reach a
/// provider. No-op when the recovered history is already balanced.
async fn backfill_orphaned_tool_results(
    deps: &RootTurnDeps<'_>,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> Result<()> {
    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for orphan backfill")?;

    if !llm::has_unbalanced_tool_use(&history) {
        return Ok(());
    }

    let balanced = llm::balance_tool_results(&history, llm::USER_CANCELLED_TOOL_RESULT);
    log::warn!(
        "thread {thread_id}: closing {} unanswered tool_use block(s) with cancelled results",
        balanced.len().saturating_sub(history.len()),
    );

    // Durable projection: fold the balanced history into the committed
    // head. `replace_history` atomically drops the in-flight draft in the
    // same transaction (the draft is exactly what we just folded in), so
    // there is no window where a crash leaves a stale draft to double-fold.
    // `recover_thread` treats the committed projection as the source of
    // truth, so the next load sees a balanced thread.
    deps.message_store
        .replace_history(thread_id, balanced.clone(), now)
        .await
        .context("persist balanced projection history")?;

    // In-memory staged buffer the request is built from.
    staged_messages
        .replace_history(thread_id, balanced)
        .await
        .context("replace staged buffer with balanced history")?;

    Ok(())
}

/// Maximum retries for a transient LLM stream error.  After this many
/// recoverable failures (rate-limit / server / dropped connection) the
/// turn is failed permanently and the user sees the last error.
///
/// This budget is per-turn — a fresh user submission starts at zero —
/// and applies on top of the journal's task-level `max_attempts`
/// (which governs lease re-acquisition).  Empirically `Stream ended
/// unexpectedly without completion` from the Anthropic SSE provider
/// almost always succeeds on the first retry; budgeting three
/// attempts (~5 s total at the default backoff: ~0.5s/1s/2s + jitter)
/// covers a longer transient blip without dragging out a genuinely
/// poisoned turn.
const STREAM_MAX_RETRIES: u32 = 3;

/// Base backoff for the exponential retry schedule (`base * 2^(n-1)`
/// with jitter, capped at [`STREAM_MAX_DELAY_MS`]).
const STREAM_BASE_DELAY_MS: u64 = 500;

/// Upper bound on the retry backoff.  The cap matches the
/// daemon-reconnect ceiling in the bipi/desktop loops so a transient
/// blip and a daemon respawn share the same wall-clock vocabulary.
const STREAM_MAX_DELAY_MS: u64 = 8_000;

/// Maximum time a freshly opened LLM stream may go without yielding its
/// FIRST event before the attempt is treated as a stalled connection
/// and retried through the normal [`StreamAttemptError::Recoverable`]
/// path.
///
/// Without this bound the poll loop in [`call_llm_once_inner`] awaits
/// `stream.next()` forever: a request written to a half-open pooled
/// connection produces no events, no error, and no journal activity —
/// on 2026-07-11 three workers hung exactly this way until an external
/// watchdog tree-killed them, losing all of their work, when a simple
/// retry would have succeeded within a second.
///
/// This is deliberately the OUTERMOST, most generous guard: providers
/// and consumer-side decorators with protocol knowledge (ping/liveness
/// visibility, thinking-mode awareness) should always fire first. The
/// widest such guard in the fleet allows healthy reasoning streams
/// 300s of pre-first-delta silence, so this backstop sits just above
/// it.
const STREAM_FIRST_EVENT_TIMEOUT: Duration = Duration::from_secs(330);

/// Maximum silence BETWEEN stream events before the attempt is treated
/// as stalled (same recovery path as [`STREAM_FIRST_EVENT_TIMEOUT`]).
///
/// Once a stream has yielded anything, healthy gaps are bounded by
/// token cadence plus provider keep-alives; two minutes of mid-stream
/// silence means the connection died. Kept at or above every
/// provider-level inter-event guard (60–120s) so those get first
/// crack; a tie is benign — both surface the same recoverable error.
const STREAM_INTER_EVENT_TIMEOUT: Duration = Duration::from_mins(2);

/// Maximum number of consecutive emergency compactions the worker
/// will run inside a single [`call_llm_with_retry`] invocation
/// before giving up and propagating the provider's
/// `prompt is too long` error.  Mirrors
/// `agent_sdk::agent_loop::types::MAX_COMPACTION_RETRIES` so the
/// daemon and in-process loops fail at the same point — usually a
/// signal that the system prompt + tools alone exceed the model's
/// context window and no amount of summarisation will help.
const MAX_COMPACTION_RETRIES: u32 = 3;

/// Classified outcome of [`call_llm_once`].  The retry wrapper
/// distinguishes recoverable transient failures from fatal ones so
/// only the former enter the backoff loop.
enum StreamAttemptError {
    /// Provider returned `RateLimited` or `ServerError`, or the
    /// underlying byte stream surfaced a transport error.  The turn
    /// attempt was already closed with the matching outcome.
    Recoverable {
        kind: StreamErrorKind,
        message: String,
    },
    /// Provider returned `InvalidRequest` (caller-side error) — no
    /// retry will help.  The turn attempt was already closed with
    /// `InvalidRequest`.
    Fatal { message: String },
    /// The root turn was cancelled mid-stream via the
    /// [`RootTurnDeps::cancel`] token.  The turn attempt was already
    /// closed with `Cancelled`; the retry wrapper bails immediately,
    /// skipping both retry/backoff and the commit path.
    Cancelled { message: String },
}

/// Bundle of context [`call_llm_with_retry`] needs to drive the
/// streaming-LLM retry loop.
///
/// Holding these fields together keeps the retry wrapper's signature
/// readable and lets every call site (the fresh-turn path in
/// [`execute_root_turn`] and the resume path in
/// [`resume_root_turn`]) build the params with the same vocabulary —
/// only `user_input` and `attempt_audit_prompt` differ.  The
/// lifetime parameter ties every borrow to a single scope so the
/// wrapper can't outlive the store handles or the input bundle.
pub(crate) struct LlmRetryParams<'a> {
    /// Worker inputs needed to re-open a [`TurnAttempt`] after a
    /// recoverable failure (the audit row references this task and
    /// the next-attempt-number derives from `attempt_store.list_by_task`).
    pub(crate) inputs: &'a RootWorkerInputs,
    /// Definition the freshly opened attempt records as provenance.
    /// Re-resolving via the registry on every retry would be safer
    /// against live-edit changes but is overkill for transient
    /// LLM-stream blips.
    pub(crate) definition: &'a AgentDefinition,
    /// Typed user input for the fresh-turn path. The compaction-
    /// recovery branch needs this to rebuild the chat request after
    /// rewriting the staged history — without it, image / document
    /// blocks would be silently lost on retry. Resume turns pass a
    /// resume-marked input via [`super::user_input::UserInput::resume`]
    /// (its block list is empty, its `is_resume()` is true).
    pub(crate) user_input: &'a super::user_input::UserInput,
    /// String written into the per-attempt `request_blob.user_prompt`
    /// audit field. Derived from `user_input.audit_summary()` —
    /// `<resume>` for resume retries, otherwise text content with
    /// binary attachments rendered as `[<media_type> attachment]`.
    pub(crate) attempt_audit_prompt: &'a str,
    /// LLM provider — borrowed because it can't be cloned cheaply.
    pub(crate) provider: &'a dyn LlmProvider,
    /// Chat request built once at the top of the turn and reused on
    /// every retry. NOTE: [`ChatRequest`] is fully owned, so the retry
    /// loop deep-clones the system prompt, every message (including any
    /// base64 image / document data), and every tool schema per attempt.
    /// The common case is a single first-and-only attempt, so the clone
    /// is paid even on the success path — tracked as a follow-up
    /// optimization (avoid cloning on the first attempt / wrap in `Arc`).
    pub(crate) chat_request: ChatRequest,
    /// Initial [`TurnAttempt`] opened by the caller before streaming
    /// begins.  Subsequent attempts are minted by the wrapper.
    pub(crate) initial_attempt: TurnAttempt,
    /// Store handles for committing delta events, closing failed
    /// attempts, and emitting `AutoRetryStart` / `AutoRetryEnd`.
    pub(crate) deps: &'a RootTurnDeps<'a>,
    /// Thread under which every event commit and audit close is
    /// attributed.
    pub(crate) thread_id: &'a agent_sdk_foundation::ThreadId,
    /// Wall-clock timestamp the first attempt records on its closed
    /// attempt row when the streaming attempt fails.  Subsequent
    /// attempts capture a fresh `now` per retry.
    pub(crate) now: OffsetDateTime,
}

/// Drive the LLM stream with retry-on-transient.
///
/// One call into `call_llm_once` runs a single streaming attempt.  On
/// success the result is returned; on a recoverable failure
/// ([`StreamAttemptError::Recoverable`]) the wrapper:
///
///   1. emits an [`AgentEvent::AutoRetryStart`] so live observers can
///      render a "Retrying X/N in Yms…" indicator,
///   2. sleeps for the exponential-backoff delay,
///   3. opens a fresh [`TurnAttempt`] (the previous one was already
///      closed inside `call_llm_once`), and
///   4. tries again.
///
/// When the budget is exhausted, an [`AgentEvent::AutoRetryEnd`] with
/// `success: false` is emitted before bailing.  When a retry
/// eventually succeeds, the matching `AutoRetryEnd { success: true }`
/// fires.  These mirror the agent-sdk's in-process retry loop in
/// `agent_loop::llm::call_llm_streaming` so both paths share a
/// vocabulary.
///
/// ## Why retry inside the worker
///
/// Without this, a single transient `Stream ended unexpectedly
/// without completion` from the Anthropic SSE provider — extremely
/// common on long resumes after a subagent — fails the entire
/// `RootTurn` task.  The journal's task-level `max_attempts` then
/// burns through its budget on the same error and the user sees a
/// `resume root task from durable child results: LLM stream error
/// (kind=ServerError)` cascade with no recovery.
///
/// Retries committed delta events from the failed attempt remain in
/// the journal — they share the per-attempt `ContentIds`, so a
/// successful retry generates fresh ids and the renderer can
/// distinguish "previous attempt's partial deltas" from "current
/// attempt's full content" via the surrounding
/// `AutoRetryStart`/`AutoRetryEnd` envelope.
async fn call_llm_with_retry(params: LlmRetryParams<'_>) -> Result<StreamedTurn> {
    let LlmRetryParams {
        inputs,
        definition,
        user_input,
        attempt_audit_prompt,
        provider,
        mut chat_request,
        initial_attempt,
        deps,
        thread_id,
        now,
    } = params;

    let mut attempt = initial_attempt;
    let mut retries: u32 = 0;
    // Separate budget for compaction-driven retries so transient
    // network blips can't burn through it before a real overflow
    // arrives, and a runaway compaction loop can't hide a genuine
    // bug behind unlimited retries. Mirrors
    // `agent_sdk::agent_loop::types::MAX_COMPACTION_RETRIES`.
    let mut compaction_retries: u32 = 0;

    loop {
        // Cooperative cancellation: bail before opening a (billed) LLM
        // attempt when the root turn has been cancelled between retries.
        // Tag the error with the cancel marker so the caller that owns
        // the staged buffer commits the completed prefix (seam B).
        if deps.is_cancelled() {
            // The current attempt row is OPEN but never started
            // streaming (this fires on entry — e.g. a subagent
            // deadline landing between `open_attempt` and the first
            // poll — and between retries). Nothing streamed, so a
            // zero-usage Cancelled close is honest, and no later path
            // owns this row: the host's timeout fail deliberately
            // leaves live-worker attempts open for THIS code to
            // close, and the task is already terminal so the normal
            // commit-path close never runs. Without this close the
            // row would stay OPEN forever in the billing/audit trail.
            close_attempt_with(
                &attempt,
                TurnAttemptOutcome::Cancelled,
                deps.attempt_store,
                OffsetDateTime::now_utc(),
            )
            .await;
            return Err(anyhow::Error::new(RootTurnCancelledMarker)
                .context("root turn cancelled before LLM attempt"));
        }
        let attempt_now = if retries == 0 && compaction_retries == 0 {
            now
        } else {
            OffsetDateTime::now_utc()
        };
        match call_llm_once(
            provider,
            chat_request.clone(),
            &attempt,
            deps,
            thread_id,
            attempt_now,
        )
        .await
        {
            Ok(OnceOutcome {
                response,
                content_ids,
            }) => {
                if retries > 0 {
                    emit_auto_retry_end(deps, thread_id, retries, true, None).await;
                }
                return Ok(StreamedTurn {
                    response,
                    content_ids,
                    attempt,
                });
            }
            Err(StreamAttemptError::Recoverable { kind, message }) => {
                attempt = handle_recoverable_stream_error(RecoverableRetryParams {
                    kind,
                    message: &message,
                    retries: &mut retries,
                    inputs,
                    definition,
                    attempt_audit_prompt,
                    deps,
                    thread_id,
                })
                .await?;
            }
            Err(StreamAttemptError::Fatal { message }) => {
                if let Some(next_attempt) =
                    try_recover_with_compaction(PromptTooLongRecoveryParams {
                        message: &message,
                        chat_request: &mut chat_request,
                        user_input,
                        attempt_audit_prompt,
                        compaction_retries: &mut compaction_retries,
                        inputs,
                        definition,
                        deps,
                        thread_id,
                    })
                    .await?
                {
                    attempt = next_attempt;
                    continue;
                }
                bail!("{message}");
            }
            Err(StreamAttemptError::Cancelled { message }) => {
                // The turn was cancelled mid-stream — the attempt is
                // already closed `Cancelled`. Bail immediately with the
                // cancel marker so the caller that still owns the staged
                // buffer commits the completed prefix (seam B); skip both
                // retry/backoff and the normal commit path.
                return Err(anyhow::Error::new(RootTurnCancelledMarker).context(message));
            }
        }
    }
}

/// Parameters for [`handle_recoverable_stream_error`]. Bundled to
/// dodge `clippy::too_many_arguments` and keep the call site at
/// [`call_llm_with_retry`] readable.
struct RecoverableRetryParams<'a> {
    kind: StreamErrorKind,
    message: &'a str,
    retries: &'a mut u32,
    inputs: &'a RootWorkerInputs,
    definition: &'a AgentDefinition,
    attempt_audit_prompt: &'a str,
    deps: &'a RootTurnDeps<'a>,
    thread_id: &'a agent_sdk_foundation::ThreadId,
}

/// Handle a recoverable streaming failure: bump the retry counter,
/// surface `AutoRetryStart`/`AutoRetryEnd`, sleep for the
/// exponential-backoff delay, and open a fresh turn attempt for the
/// next iteration. Returns the new attempt the caller should retry
/// against.
async fn handle_recoverable_stream_error(
    params: RecoverableRetryParams<'_>,
) -> Result<TurnAttempt> {
    let RecoverableRetryParams {
        kind,
        message,
        retries,
        inputs,
        definition,
        attempt_audit_prompt,
        deps,
        thread_id,
    } = params;

    *retries = retries.saturating_add(1);
    if *retries > STREAM_MAX_RETRIES {
        let final_msg = format!(
            "LLM stream error after {STREAM_MAX_RETRIES} retries (kind={kind:?}): {message}"
        );
        emit_auto_retry_end(
            deps,
            thread_id,
            retries.saturating_sub(1),
            false,
            Some(final_msg.clone()),
        )
        .await;
        bail!("{final_msg}");
    }
    let delay = stream_backoff_delay(*retries);
    let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
    log::warn!(
        "LLM stream {kind:?} on attempt {retries}/{STREAM_MAX_RETRIES} \
         for thread {thread_id}; retrying in {delay_ms} ms — {message}",
    );
    emit_auto_retry_start(
        deps,
        thread_id,
        *retries,
        STREAM_MAX_RETRIES,
        delay_ms,
        message,
    )
    .await;
    // Cooperative cancellation: abort the backoff promptly instead of
    // sleeping out the full (up to 8s) delay on an already-cancelled
    // turn.
    match deps.cancel {
        Some(cancel) => {
            tokio::select! {
                biased;
                () = cancel.cancelled() => bail!("root turn cancelled during retry backoff"),
                () = tokio::time::sleep(delay) => {}
            }
        }
        None => tokio::time::sleep(delay).await,
    }
    open_attempt(
        inputs,
        definition,
        attempt_audit_prompt,
        deps.attempt_store,
        OffsetDateTime::now_utc(),
        None,
    )
    .await
    .context("open retry turn attempt")
}

/// Parameters for [`try_recover_with_compaction`]. Bundled because
/// the helper would otherwise hit `clippy::too_many_arguments`, and
/// because every field is borrowed from the same retry-loop scope.
struct PromptTooLongRecoveryParams<'a> {
    message: &'a str,
    chat_request: &'a mut ChatRequest,
    /// Typed user input — if non-resume, the matching `Message` is
    /// pushed back onto the rebuilt chat request so image / document
    /// blocks survive emergency compaction.
    user_input: &'a super::user_input::UserInput,
    /// Audit string for the retry attempt's `request_blob`.
    attempt_audit_prompt: &'a str,
    compaction_retries: &'a mut u32,
    inputs: &'a RootWorkerInputs,
    definition: &'a AgentDefinition,
    deps: &'a RootTurnDeps<'a>,
    thread_id: &'a agent_sdk_foundation::ThreadId,
}

/// Detect a `prompt is too long`-class fatal error, run an emergency
/// compaction against the durable projection + staged buffer, rebuild
/// the chat request from the compacted history, and open a fresh
/// turn attempt for the retry.
///
/// Returns `Ok(Some(attempt))` when compaction ran and the caller
/// should retry with the rebuilt `chat_request` under the new
/// `attempt`, `Ok(None)` when the error doesn't match the
/// prompt-too-long shape (or compaction isn't configured / has been
/// exhausted), and `Err` if compaction itself failed in a way the
/// caller must surface.
async fn try_recover_with_compaction(
    params: PromptTooLongRecoveryParams<'_>,
) -> Result<Option<TurnAttempt>> {
    let PromptTooLongRecoveryParams {
        message,
        chat_request,
        user_input,
        attempt_audit_prompt,
        compaction_retries,
        inputs,
        definition,
        deps,
        thread_id,
    } = params;

    if !super::compaction::is_prompt_too_long_error(message)
        || deps.compaction_config.is_none()
        || deps.compaction_provider.is_none()
        || *compaction_retries >= MAX_COMPACTION_RETRIES
    {
        return Ok(None);
    }

    *compaction_retries = compaction_retries.saturating_add(1);
    let now_compact = OffsetDateTime::now_utc();
    log::warn!(
        "Provider rejected turn with prompt-too-long on thread {thread_id} \
         (compaction retry {}/{MAX_COMPACTION_RETRIES}); attempting emergency \
         compaction before retry",
        *compaction_retries,
    );
    let did_compact = super::compaction::compact_after_overflow(
        deps,
        &inputs.staged_stores.messages,
        thread_id,
        now_compact,
    )
    .await
    .context("emergency compaction after prompt-too-long")?;

    if !did_compact {
        return Ok(None);
    }

    // Rebuild the chat request from the compacted staged history.
    // For the fresh path, re-append the original user message —
    // including any image / document blocks — that
    // `build_chat_request` pushed onto the original request. The
    // staged buffer doesn't carry it yet (the commit path appends it
    // via `buffer_turn_messages` after a successful turn). For the
    // resume path `user_input.into_message()` returns `None` and the
    // staged buffer already contains everything the LLM needs.
    let mut messages = inputs
        .staged_stores
        .messages
        .get_history(thread_id)
        .await
        .context("read staged history after emergency compaction")?;
    if let Some(message) = user_input.clone().into_message() {
        messages.push(message);
    }
    chat_request.messages = messages;

    let next_attempt = open_attempt(
        inputs,
        definition,
        attempt_audit_prompt,
        deps.attempt_store,
        OffsetDateTime::now_utc(),
        None,
    )
    .await
    .context("open retry turn attempt after emergency compaction")?;

    Ok(Some(next_attempt))
}

/// Exponential backoff with bounded jitter.  Mirrors
/// `agent_sdk::agent_loop::helpers::calculate_backoff_delay` but
/// is duplicated here to avoid leaking a `pub(crate)` boundary across
/// crates for a tiny pure helper.
fn stream_backoff_delay(attempt: u32) -> Duration {
    // Exponential: base, base*2, base*4, ...
    let base = STREAM_BASE_DELAY_MS.saturating_mul(1u64 << attempt.saturating_sub(1).min(20));
    // Add jitter (0..min(base, 1000)ms) to spread out colliding
    // retries from independent turns hitting the same upstream blip.
    let max_jitter = STREAM_BASE_DELAY_MS.min(1000);
    let jitter = if max_jitter > 0 {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        RandomState::new().build_hasher().finish() % max_jitter
    } else {
        0
    };
    let delay_ms = base.saturating_add(jitter).min(STREAM_MAX_DELAY_MS);
    Duration::from_millis(delay_ms)
}

/// Best-effort emit of an `AutoRetryStart` event so renderers can
/// render a "Retrying X/N in Yms…" pill.  Errors are swallowed —
/// losing telemetry on the retry events shouldn't block the retry
/// loop itself.
async fn emit_auto_retry_start(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error_message: &str,
) {
    let event = AgentEvent::AutoRetryStart {
        attempt,
        max_attempts,
        delay_ms,
        error_message: error_message.to_string(),
    };
    match deps
        .event_repo
        .commit_event(thread_id, event, OffsetDateTime::now_utc())
        .await
    {
        Ok(committed) => {
            deps.event_notifier.notify(std::slice::from_ref(&committed));
        }
        Err(error) => {
            log::warn!("failed to commit auto_retry_start event for thread {thread_id}: {error:#}");
        }
    }
}

/// Best-effort emit of an `AutoRetryEnd` event.  `success = true`
/// means a follow-up attempt eventually returned data; `success =
/// false` means the budget was exhausted and `final_error` carries
/// the last error.
async fn emit_auto_retry_end(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    attempt: u32,
    success: bool,
    final_error: Option<String>,
) {
    let event = AgentEvent::AutoRetryEnd {
        attempt,
        success,
        final_error,
    };
    match deps
        .event_repo
        .commit_event(thread_id, event, OffsetDateTime::now_utc())
        .await
    {
        Ok(committed) => {
            deps.event_notifier.notify(std::slice::from_ref(&committed));
        }
        Err(error) => {
            log::warn!("failed to commit auto_retry_end event for thread {thread_id}: {error:#}");
        }
    }
}

/// Call the LLM via `chat_stream`, committing `TextDelta` /
/// `ThinkingDelta` events to the journal as they arrive so live
/// observers (TUI, desktop) see streaming output character-by-character
/// instead of one consolidated [`AgentEvent::Text`] event at the end of
/// the turn.
///
/// Closes the turn attempt on any non-success path before returning an
/// error.  On success returns a [`StreamedTurn`] containing the
/// synthesized [`llm::ChatResponse`] (rebuilt from the
/// [`StreamAccumulator`]) and the [`ContentIds`] map of per-block
/// message/thinking IDs.  The IDs are generated lazily on the first
/// non-empty delta for each `block_index` so the same id is reused for
/// both the streamed delta events and the consolidated `Text` /
/// `Thinking` events emitted at turn close.
///
/// Each delta event is committed with a freshly captured
/// `OffsetDateTime::now_utc()` rather than the turn-open `now`, so the
/// journal records true wall-clock arrival times for each delta.
///
/// Returns [`StreamAttemptError::Recoverable`] for transient failures
/// (the retry wrapper [`call_llm_with_retry`] handles backoff +
/// re-attempt) and [`StreamAttemptError::Fatal`] for caller-side
/// errors that no retry can fix.
///
/// ## Observability
///
/// This is the daemon-hosted equivalent of the in-process loop's
/// `agent_loop::turn::request_llm_response`, so it opens the same
/// `chat {model}` CLIENT span and records the same
/// `gen_ai.client.token.usage` / `gen_ai.client.operation.duration`
/// metrics — via the shared
/// [`agent_sdk::observability::loop_instrument`] helpers — when the
/// `otel` feature is on. The streaming/journal/retry logic lives in
/// [`call_llm_once_inner`]; this wrapper only brackets it with the
/// span so dashboards built against the in-process loop light up
/// unchanged on the daemon path.
async fn call_llm_once(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    attempt: &TurnAttempt,
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> Result<OnceOutcome, StreamAttemptError> {
    #[cfg(feature = "otel")]
    {
        call_llm_once_instrumented(provider, request, attempt, deps, thread_id, now).await
    }
    #[cfg(not(feature = "otel"))]
    {
        call_llm_once_inner(provider, request, attempt, deps, thread_id, now).await
    }
}

/// Bracket [`call_llm_once_inner`] with the `chat {model}` CLIENT span
/// + `gen_ai.client.*` metrics, reusing the shared SDK helpers.
///
/// This is the daemon-hosted analogue of the in-process loop's
/// `agent_loop::turn::request_llm_response` instrumentation: same span
/// name, same `gen_ai.*` attribute set, same token-usage /
/// operation-duration metrics under the `agent-sdk` meter scope.
#[cfg(feature = "otel")]
async fn call_llm_once_instrumented(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    attempt: &TurnAttempt,
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> Result<OnceOutcome, StreamAttemptError> {
    use agent_sdk::observability::loop_instrument;

    // Mirror the in-process loop, which only stamps
    // `gen_ai.request.max_output_tokens` when the caller explicitly
    // configured a cap (`config.max_tokens.is_some()`). The worker's
    // `ChatRequest::max_tokens` is always populated with a resolved
    // default, so gate on `max_tokens_explicit` to avoid emitting a
    // synthetic value the in-process span would omit.
    let max_tokens = request.max_tokens_explicit.then_some(request.max_tokens);
    let request_model = provider.model().to_string();
    let provider_id = provider.provider();
    let mut span = loop_instrument::build_chat_span(loop_instrument::ChatSpanParams {
        provider_id,
        model: &request_model,
        // The worker only ever drives the streaming API
        // (`provider.chat_stream`), so this span always reflects a
        // streaming call.
        streaming: true,
        max_tokens,
    });

    let started_at = std::time::Instant::now();
    let result = call_llm_once_inner(provider, request, attempt, deps, thread_id, now).await;
    let elapsed = started_at.elapsed().as_secs_f64();

    match &result {
        Ok(outcome) => {
            loop_instrument::finish_chat_span_success(
                &mut span,
                &outcome.response,
                elapsed,
                provider_id,
                &request_model,
            );
        }
        Err(error) => {
            let (error_type, message) = match error {
                StreamAttemptError::Recoverable { kind, message } => {
                    (stream_error_kind_type(*kind), message.as_str())
                }
                StreamAttemptError::Fatal { message } => ("invalid_request", message.as_str()),
                StreamAttemptError::Cancelled { message } => ("cancelled", message.as_str()),
            };
            loop_instrument::finish_chat_span_error(
                &mut span,
                error_type,
                message,
                elapsed,
                provider_id,
                &request_model,
            );
        }
    }

    result
}

/// Map a [`StreamErrorKind`] to the stable `error.type` attribute /
/// metric label, matching the vocabulary
/// `agent_sdk::observability::loop_instrument::classify_llm_error`
/// produces for the in-process loop.
#[cfg(feature = "otel")]
const fn stream_error_kind_type(kind: StreamErrorKind) -> &'static str {
    match kind {
        StreamErrorKind::RateLimited => "rate_limited",
        StreamErrorKind::ServerError => "server_error",
        StreamErrorKind::InvalidRequest => "invalid_request",
        // `StreamErrorKind` is #[non_exhaustive]; `Unknown` and any future
        // variant map to the same stable label the in-process loop uses.
        _ => "unknown",
    }
}

/// Number of buffered streaming deltas that forces a coalesced
/// [`EventRepository::commit_event_batch`] flush.
///
/// Committing each `TextDelta` / `ThinkingDelta` inline cost one full
/// DB transaction per token chunk (BEGIN / SELECT MAX(seq) / INSERT /
/// COMMIT) and serialized stream consumption behind each round-trip.
/// Buffering and flushing in bounded batches turns hundreds of
/// per-token transactions into a handful of batch commits. Replay
/// completeness does not depend on these deltas — the consolidated
/// `Text` / `Thinking` events at turn close carry the full content — so
/// coalescing is purely an optimization.
const STREAMING_DELTA_BATCH_SIZE: usize = 16;

/// Maximum wall-clock a buffered delta waits before being flushed, so
/// short responses (fewer than [`STREAMING_DELTA_BATCH_SIZE`] chunks)
/// still reach live-tail subscribers promptly instead of only at turn
/// close. The flush fires on whichever bound trips first.
const STREAMING_DELTA_FLUSH_INTERVAL: Duration = Duration::from_millis(50);

/// Parameters for [`close_stalled_attempt`]. Bundled to dodge
/// `clippy::too_many_arguments` and keep the poll loop in
/// [`call_llm_once_inner`] readable.
struct StalledAttemptParams<'a, 'deps> {
    deps: &'a RootTurnDeps<'deps>,
    thread_id: &'a agent_sdk_foundation::ThreadId,
    pending_deltas: &'a mut Vec<AgentEvent>,
    attempt: &'a TurnAttempt,
    received_first_item: bool,
    stall_budget: Duration,
    now: OffsetDateTime,
}

/// Close an attempt whose stream went silent past its stall budget
/// ([`STREAM_FIRST_EVENT_TIMEOUT`] / [`STREAM_INTER_EVENT_TIMEOUT`]).
///
/// A stalled stream is indistinguishable from a dead connection at this
/// layer — the attempt is closed like any other transport failure so
/// the retry wrapper re-sends on a fresh attempt.
async fn close_stalled_attempt(params: StalledAttemptParams<'_, '_>) -> StreamAttemptError {
    let StalledAttemptParams {
        deps,
        thread_id,
        pending_deltas,
        attempt,
        received_first_item,
        stall_budget,
        now,
    } = params;
    let stage = if received_first_item {
        "mid-stream"
    } else {
        "before its first event"
    };
    flush_and_close(
        deps,
        thread_id,
        pending_deltas,
        attempt,
        TurnAttemptOutcome::ServerError,
        now,
    )
    .await;
    StreamAttemptError::Recoverable {
        kind: StreamErrorKind::ServerError,
        message: format!(
            "LLM stream stalled {stage}: no events for {}s — treating the connection as dead",
            stall_budget.as_secs(),
        ),
    }
}

/// Close an attempt whose root turn was cancelled mid-stream via the
/// [`RootTurnDeps::cancel`] token and build the matching non-retryable
/// error (see [`StreamAttemptError::Cancelled`]).
async fn close_cancelled_attempt(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    pending_deltas: &mut Vec<AgentEvent>,
    attempt: &TurnAttempt,
    now: OffsetDateTime,
) -> StreamAttemptError {
    flush_and_close(
        deps,
        thread_id,
        pending_deltas,
        attempt,
        TurnAttemptOutcome::Cancelled,
        now,
    )
    .await;
    StreamAttemptError::Cancelled {
        message: "root turn cancelled mid-stream".to_owned(),
    }
}

/// Streaming/journal/retry body of [`call_llm_once`]. Consumes the LLM
/// stream, coalescing per-token delta events into batched commits,
/// honouring cooperative cancellation, and requiring a completion
/// marker before treating the turn as successful. The `chat {model}`
/// span lives in the [`call_llm_once`] wrapper.
async fn call_llm_once_inner(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    attempt: &TurnAttempt,
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> Result<OnceOutcome, StreamAttemptError> {
    let mut stream = std::pin::pin!(provider.chat_stream(request));
    let mut accumulator = StreamAccumulator::new();
    let mut content_ids = ContentIds::default();
    // Buffered `TextDelta` / `ThinkingDelta` events awaiting a batched
    // commit. Flushed when the buffer reaches `STREAMING_DELTA_BATCH_SIZE`
    // or `STREAMING_DELTA_FLUSH_INTERVAL` elapses, at stream end, and
    // before any early return so a failed/cancelled attempt still lands
    // its partial transcript in the journal.
    let mut pending_deltas: Vec<AgentEvent> = Vec::new();
    let mut last_flush = std::time::Instant::now();
    // Latched on the first yielded item; selects which stall budget the
    // next poll runs under (first-event vs inter-event).
    let mut received_first_item = false;

    loop {
        let stall_budget = if received_first_item {
            STREAM_INTER_EVENT_TIMEOUT
        } else {
            STREAM_FIRST_EVENT_TIMEOUT
        };
        // Poll the next stream item, racing it against the cancellation
        // token (so a cancelled root turn stops consuming the (billed)
        // stream promptly instead of draining it to completion) and the
        // stall budget (so a silent stream — e.g. a request written to
        // a half-open connection — is retried instead of hanging until
        // an external watchdog kills the whole task tree).
        let polled = match deps.cancel {
            Some(cancel) => {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        return Err(close_cancelled_attempt(
                            deps,
                            thread_id,
                            &mut pending_deltas,
                            attempt,
                            now,
                        )
                        .await);
                    }
                    polled = tokio::time::timeout(stall_budget, stream.next()) => polled,
                }
            }
            None => tokio::time::timeout(stall_budget, stream.next()).await,
        };
        let next = match polled {
            Ok(next) => next,
            Err(_elapsed) => {
                return Err(close_stalled_attempt(StalledAttemptParams {
                    deps,
                    thread_id,
                    pending_deltas: &mut pending_deltas,
                    attempt,
                    received_first_item,
                    stall_budget,
                    now,
                })
                .await);
            }
        };
        let Some(result) = next else { break };
        received_first_item = true;

        let delta = match result {
            Ok(delta) => delta,
            Err(error) => {
                // Transport-level stream errors (HTTP read failures,
                // dropped TCP, etc.) are treated as recoverable —
                // they're the most common shape of "Anthropic SSE
                // stream died mid-flight" and almost always succeed
                // on retry.
                flush_and_close(
                    deps,
                    thread_id,
                    &mut pending_deltas,
                    attempt,
                    TurnAttemptOutcome::ServerError,
                    now,
                )
                .await;
                return Err(StreamAttemptError::Recoverable {
                    kind: StreamErrorKind::ServerError,
                    message: format!("LLM stream iteration error: {error:#}"),
                });
            }
        };

        accumulator.apply(&delta);

        match buffer_stream_delta(&delta, &mut content_ids, &mut pending_deltas) {
            DeltaStep::Skip => continue,
            DeltaStep::Buffered => {}
            DeltaStep::Fail(outcome, error) => {
                flush_and_close(deps, thread_id, &mut pending_deltas, attempt, outcome, now).await;
                return Err(error);
            }
        }

        maybe_flush_batch(deps, thread_id, &mut pending_deltas, &mut last_flush).await;
    }

    // Flush any deltas buffered since the last batch before closing.
    flush_streaming_deltas(deps, thread_id, &mut pending_deltas).await;

    // Require a completion marker before treating the stream as a
    // successful turn. Some providers (e.g. the OpenAI-compatible impl)
    // emit partial content when the connection ends without a terminal
    // `[DONE]`, and the accumulator coerces unparseable streamed
    // tool-input JSON to `{}` — so committing a `stop_reason`-less
    // stream would durably record a cut-off assistant message or spawn
    // tool children with empty inputs. Treat it as a recoverable error
    // so `call_llm_with_retry` re-attempts instead.
    if accumulator.stop_reason().is_none() {
        close_attempt_with(
            attempt,
            TurnAttemptOutcome::ServerError,
            deps.attempt_store,
            now,
        )
        .await;
        return Err(StreamAttemptError::Recoverable {
            kind: StreamErrorKind::ServerError,
            message: "LLM stream ended without a completion marker (stop_reason); \
                      treating as a truncated response"
                .to_owned(),
        });
    }

    let response = synthesize_response(accumulator, provider, thread_id);
    Ok(OnceOutcome {
        response,
        content_ids,
    })
}

/// What [`call_llm_once_inner`] should do after classifying a single
/// streamed delta.
enum DeltaStep {
    /// The delta was buffered (or carried no content); evaluate the
    /// batch-flush bound and keep consuming.
    Buffered,
    /// Empty delta — skip the batch-flush check and poll the next item.
    Skip,
    /// Provider surfaced a stream error: flush, close the attempt with
    /// `outcome`, and propagate `error`.
    Fail(TurnAttemptOutcome, StreamAttemptError),
}

/// Classify one streamed [`StreamDelta`], buffering renderable text /
/// thinking deltas into `pending_deltas` and allocating their per-block
/// ids on first non-empty content.
fn buffer_stream_delta(
    delta: &StreamDelta,
    content_ids: &mut ContentIds,
    pending_deltas: &mut Vec<AgentEvent>,
) -> DeltaStep {
    match delta {
        StreamDelta::TextDelta {
            delta: text_chunk,
            block_index,
        } => {
            if text_chunk.is_empty() {
                // Empty deltas carry no content; skipping them also
                // avoids allocating an id for blocks that never produce
                // non-empty text.
                return DeltaStep::Skip;
            }
            let message_id = content_ids.text_id_for(*block_index);
            pending_deltas.push(AgentEvent::text_delta(
                message_id.to_string(),
                text_chunk.clone(),
            ));
            DeltaStep::Buffered
        }
        StreamDelta::ThinkingDelta {
            delta: thinking_chunk,
            block_index,
        } => {
            if thinking_chunk.is_empty() {
                return DeltaStep::Skip;
            }
            let thinking_id = content_ids.thinking_id_for(*block_index);
            pending_deltas.push(AgentEvent::thinking_delta(
                thinking_id.to_string(),
                thinking_chunk.clone(),
            ));
            DeltaStep::Buffered
        }
        StreamDelta::Error { message, kind } => {
            // Map the kind directly onto the audit outcome so a genuine
            // 5xx is recorded as `ServerError` (not `RateLimited`) and a
            // validation rejection is recorded as `InvalidRequest` (not
            // `ServerError`).
            let outcome = match kind {
                StreamErrorKind::RateLimited => TurnAttemptOutcome::RateLimited,
                StreamErrorKind::InvalidRequest => TurnAttemptOutcome::InvalidRequest,
                // `StreamErrorKind::ServerError`, plus the
                // `#[non_exhaustive]` catch-all (`Unknown` / future kinds):
                // audited as a server error, the most conservative
                // non-rate-limit category.
                _ => TurnAttemptOutcome::ServerError,
            };
            let kind = *kind;
            let message = message.clone();
            let error = if kind.is_recoverable() {
                StreamAttemptError::Recoverable { kind, message }
            } else {
                StreamAttemptError::Fatal {
                    message: format!("LLM stream error (kind={kind:?}): {message}"),
                }
            };
            DeltaStep::Fail(outcome, error)
        }
        // Done / Usage / ToolUseStart / ToolInputDelta / SignatureDelta /
        // RedactedThinking / OpaqueReasoning are handled by the accumulator and don't need
        // to be re-emitted as events. The catch-all also covers future
        // `#[non_exhaustive]` deltas, which the accumulator likewise
        // consumes.
        _ => DeltaStep::Buffered,
    }
}

/// Build the synthetic [`llm::ChatResponse`] from a closed
/// [`StreamAccumulator`].
///
/// Pre-PR `provider.chat()` returned a non-`Option` `Usage` whose
/// `input_tokens`/`output_tokens` are required serde fields, so a
/// missing-usage response surfaced as a hard parse error.  After the
/// streaming refactor, providers that never emit `StreamDelta::Usage`
/// (e.g. the `openai` impl when pointed at `moonshot.ai` / `api.z.ai`
/// / `minimax.io`, where `use_stream_usage_options` returns `false`)
/// would silently default to `Usage{0,0,0,0}` and feed that into
/// billing, quota, and audit columns indistinguishably from a
/// genuinely free turn.
///
/// The fallback is preserved (the same default exists in
/// `provider::collect_stream` and `agent_loop::llm`, so this isn't
/// unique to this site), but a `log::warn!` makes the gap loud and
/// searchable in operational logs so cost-tracking dashboards can
/// detect the under-counting.
///
/// The synthesized response is assembled from streaming state and
/// never reaches a provider's response-id space.  The downstream
/// commit/suspension paths route every event through `content_ids`
/// and never read `response.id`, so leaving it empty avoids
/// allocating a UUID that nothing consumes.
fn synthesize_response(
    accumulator: StreamAccumulator,
    provider: &dyn LlmProvider,
    thread_id: &agent_sdk_foundation::ThreadId,
) -> llm::ChatResponse {
    let usage = accumulator.usage().cloned().unwrap_or_else(|| {
        log::warn!(
            "provider {} streamed turn for thread {thread_id} without a Usage delta; \
             recording Usage{{0,0,0,0}} — token counts under-count on this turn",
            provider.provider(),
        );
        llm::Usage {
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }
    });
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    llm::ChatResponse {
        id: String::new(),
        content: content_blocks,
        model: provider.model().to_string(),
        stop_reason,
        usage,
    }
}

/// Flush buffered streaming delta events as a single coalesced
/// [`EventRepository::commit_event_batch`], then notify live-tail
/// subscribers. No-op when the buffer is empty.
///
/// Delta commit failures are intentionally non-fatal: the consolidated
/// `Text` / `Thinking` event committed at turn close still captures the
/// full content for replay clients, so a transient journal error during
/// streaming should not abort the turn. The whole batch shares one
/// freshly captured timestamp rather than one per delta — coalescing
/// trades exact per-delta arrival times for a fraction of the write
/// amplification.
async fn flush_streaming_deltas(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    buffer: &mut Vec<AgentEvent>,
) {
    if buffer.is_empty() {
        return;
    }
    let batch = std::mem::take(buffer);
    let batch_len = batch.len();
    let flush_now = OffsetDateTime::now_utc();
    match deps
        .event_repo
        .commit_event_batch(thread_id, batch, flush_now)
        .await
    {
        Ok(committed) => deps.event_notifier.notify(&committed),
        Err(error) => {
            log::warn!(
                "failed to commit streaming delta batch for thread {thread_id} \
                 (batch_len={batch_len}): {error:#}",
            );
        }
    }
}

/// Best-effort close of a turn attempt with a failure outcome.
async fn close_attempt_with(
    attempt: &TurnAttempt,
    outcome: TurnAttemptOutcome,
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
) {
    let params = CloseAttemptParams {
        response_blob: serde_json::json!(null),
        response_id: None,
        response_model: None,
        stop_reason: None,
        outcome,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
    };
    // Best-effort: if closing fails, the primary error is more important.
    let _ = attempt_store.close_attempt(&attempt.id, params, now).await;
}

/// Flush any buffered deltas, then close the turn attempt with `outcome`.
///
/// Used on every early-return path so a failed / cancelled attempt still
/// lands its partial transcript before the audit row is closed.
async fn flush_and_close(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    pending_deltas: &mut Vec<AgentEvent>,
    attempt: &TurnAttempt,
    outcome: TurnAttemptOutcome,
    now: OffsetDateTime,
) {
    flush_streaming_deltas(deps, thread_id, pending_deltas).await;
    close_attempt_with(attempt, outcome, deps.attempt_store, now).await;
}

/// Coalesce: flush buffered deltas once the count or time bound trips, so
/// a long response issues a handful of batched commits rather than one DB
/// transaction per token chunk while the time bound keeps short responses
/// live.
async fn maybe_flush_batch(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    pending_deltas: &mut Vec<AgentEvent>,
    last_flush: &mut std::time::Instant,
) {
    if pending_deltas.len() >= STREAMING_DELTA_BATCH_SIZE
        || (!pending_deltas.is_empty() && last_flush.elapsed() >= STREAMING_DELTA_FLUSH_INTERVAL)
    {
        flush_streaming_deltas(deps, thread_id, pending_deltas).await;
        *last_flush = std::time::Instant::now();
    }
}

async fn buffer_turn_messages(
    staged_messages: &crate::journal::staged::StagedMessageStore,
    staged_state: &crate::journal::staged::StagedStateStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    user_input: &super::user_input::UserInput,
    response: &llm::ChatResponse,
) -> Result<()> {
    // Append user message — preserves any image / document blocks
    // alongside the text content. Resume inputs (`user_input.is_resume()`)
    // produce no message; the staged store already contains the
    // suspended-turn history that the resume path buffered earlier.
    if let Some(message) = user_input.clone().into_message() {
        staged_messages
            .append(thread_id, message)
            .await
            .context("append user message")?;
    }

    // Build and append assistant message from response content.
    let assistant_msg = build_assistant_message(response);
    staged_messages
        .append(thread_id, assistant_msg)
        .await
        .context("append assistant message")?;

    // Update agent state.
    let current_state = staged_state
        .load(thread_id)
        .await?
        .unwrap_or_else(|| AgentState::new(thread_id.clone()));

    let mut total_usage = current_state.total_usage.clone();
    total_usage.add(&response_token_usage(response));
    let updated_state = AgentState {
        turn_count: current_state.turn_count + 1,
        total_usage,
        ..current_state
    };
    staged_state
        .save(&updated_state)
        .await
        .context("save updated agent state")?;

    Ok(())
}

/// Whether a response is allowed to start a new tool-dispatch wave.
///
/// A reported stop reason is the provider's authoritative declaration of
/// whether the response is a tool handoff.  If a malformed response combines
/// a terminal reason with `ToolUse` blocks, the terminal commit path drops
/// those blocks from persisted history rather than spawning children that the
/// provider did not authorize.  `None` remains a legacy-compatible fallback
/// for older provider adapters that emitted tool calls without a stop reason.
fn response_requests_tool_dispatch(response: &llm::ChatResponse) -> bool {
    response.has_tool_use() && stop_reason_allows_tool_dispatch(response.stop_reason)
}

const fn stop_reason_allows_tool_dispatch(stop_reason: Option<llm::StopReason>) -> bool {
    matches!(stop_reason, None | Some(llm::StopReason::ToolUse))
}

/// Build an assistant message from a chat response, preserving text
/// and thinking blocks.
fn build_assistant_message(response: &llm::ChatResponse) -> llm::Message {
    let blocks: Vec<llm::ContentBlock> = response
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text { text } => {
                Some(llm::ContentBlock::Text { text: text.clone() })
            }
            llm::ContentBlock::Thinking {
                thinking,
                signature,
            } => Some(llm::ContentBlock::Thinking {
                thinking: thinking.clone(),
                signature: signature.clone(),
            }),
            llm::ContentBlock::RedactedThinking { data } => {
                Some(llm::ContentBlock::RedactedThinking { data: data.clone() })
            }
            llm::ContentBlock::OpaqueReasoning { provider, data } => {
                Some(llm::ContentBlock::OpaqueReasoning {
                    provider: provider.clone(),
                    data: data.clone(),
                })
            }
            // Tool-use blocks filtered out (text-only path).
            _ => None,
        })
        .collect();

    llm::Message {
        role: llm::Role::Assistant,
        content: llm::Content::Blocks(blocks),
    }
}

/// Build an assistant message from a chat response, preserving ALL
/// content blocks including tool-use. Used by the suspension path to
/// capture the full response for later resume.
fn build_full_assistant_message(response: &llm::ChatResponse) -> llm::Message {
    let blocks: Vec<llm::ContentBlock> = response.content.clone();
    llm::Message {
        role: llm::Role::Assistant,
        content: llm::Content::Blocks(blocks),
    }
}

/// Sanitize a [`llm::ChatResponse`]'s `id` field for durable storage.
///
/// The streaming path leaves `response.id` empty because the
/// provider's response id is not yet plumbed through the
/// [`StreamDelta`] protocol.  Persisting `Some("")` would violate the
/// documented contract on
/// [`AgentContinuation::response_id`](agent_sdk_foundation::AgentContinuation::response_id)
/// — `None` is the sentinel for "provider did not return an id" — so
/// every caller that wants to durably record the response id must
/// route through this helper.
fn sanitized_response_id(response: &llm::ChatResponse) -> Option<String> {
    Some(response.id.clone()).filter(|s| !s.is_empty())
}

fn build_close_params(response: &llm::ChatResponse, _attempt: &TurnAttempt) -> CloseAttemptParams {
    let response_id = sanitized_response_id(response);

    // The audit blob mirrors the typed `response_id` column so the two
    // never disagree.  When the provider didn't return an id, the
    // blob simply omits the `id` field rather than recording an empty
    // string that contradicts `response_id IS NULL`.
    let response_blob = response_id.as_ref().map_or_else(
        || serde_json::json!({ "model": response.model }),
        |id| serde_json::json!({ "id": id, "model": response.model }),
    );

    CloseAttemptParams {
        response_blob,
        response_id,
        response_model: Some(response.model.clone()),
        stop_reason: response.stop_reason,
        outcome: TurnAttemptOutcome::Success,
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
    }
}

/// Extract `Thinking` and `Text` content events from the LLM response.
///
/// These events give replay observers the fine-grained content that
/// the root turn produced — thinking blocks (model reasoning) and
/// text blocks (final response). Tool-use blocks are excluded because
/// they are covered by `ToolCallStart` events on the suspension path.
///
/// Each consolidated event reuses the per-block id allocated during
/// streaming so a client can match each delta with the final event by
/// id (see [`call_llm`](super::call_llm)).
///
/// # Invariant
///
/// `response.content` is sorted by `block_index` (see
/// [`StreamAccumulator::into_content_blocks`]).  `content_ids.text_ids`
/// and `content_ids.thinking_ids` are [`BTreeMap`]s keyed by
/// `block_index`, so iterating their values yields the same
/// block-index ordering.  This lets us match each non-empty
/// `Text` / `Thinking` block to its id positionally without needing
/// the `block_index` attached to each [`llm::ContentBlock`].
fn build_content_events(response: &llm::ChatResponse, content_ids: &ContentIds) -> Vec<AgentEvent> {
    let mut text_iter = content_ids.text_ids.values();
    let mut thinking_iter = content_ids.thinking_ids.values();

    response
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => thinking_iter
                .next()
                .map(|id| AgentEvent::thinking(id.to_string(), thinking)),
            llm::ContentBlock::Text { text } if !text.is_empty() => text_iter
                .next()
                .map(|id| AgentEvent::text(id.to_string(), text)),
            _ => None,
        })
        .collect()
}

/// Build lifecycle events for a completed turn: content events,
/// optional `Refusal`, `TurnComplete`, and `Done`.
///
/// Content events (`Thinking`, `Text`) are emitted first so replay
/// observers see the model's output before the lifecycle edges.
fn build_turn_complete_events(
    response: &llm::ChatResponse,
    thread_id: &agent_sdk_foundation::ThreadId,
    turn_number: usize,
    turn_usage: &TokenUsage,
    total_usage: &TokenUsage,
    duration: std::time::Duration,
    content_ids: &ContentIds,
) -> Vec<AgentEvent> {
    let mut events = build_content_events(response, content_ids);
    if response.stop_reason == Some(llm::StopReason::Refusal) {
        // The refusal text is the first text block, so reuse the id
        // that streaming assigned to that block.  If the model refused
        // without producing any text, fall back to a fresh `UUIDv7` so
        // the event still has a stable id for downstream consumers.
        let refusal_id = content_ids.first_text_id().unwrap_or_else(Uuid::now_v7);
        events.push(AgentEvent::refusal(
            refusal_id.to_string(),
            response.first_text().map(str::to_owned),
        ));
    }
    events.push(AgentEvent::TurnComplete {
        turn: turn_number,
        usage: turn_usage.clone(),
    });
    events.push(AgentEvent::done(
        thread_id.clone(),
        turn_number,
        total_usage.clone(),
        duration,
    ));
    events
}

// ─────────────────────────────────────────────────────────────────────
// Tool-boundary suspension (Phase 4.4)
// ─────────────────────────────────────────────────────────────────────

/// Suspend execution at the tool boundary.
///
/// Called when the LLM response contains tool-use blocks. This path:
///
/// 1. Closes the turn attempt with `Success` (the LLM call succeeded).
/// 2. Builds a [`ContinuationEnvelope`] capturing the agent state,
///    pending tool calls, and LLM response metadata.
/// 3. Creates one [`ChildSpawnSpec`] per tool call.
/// 4. Atomically spawns children and parks the parent via
///    [`AgentTaskStore::spawn_tool_children`].
///
/// No checkpoint or message-projection write occurs on this path.
///
/// [`ContinuationEnvelope`]: agent_sdk_foundation::ContinuationEnvelope
async fn suspend_at_tool_boundary(
    inputs: RootWorkerInputs,
    user_input: &super::user_input::UserInput,
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
    close_ctx: TurnCloseContext,
) -> Result<RootTurnOutcome> {
    let task_id = &inputs.bootstrap.task_id;

    // Idempotency guard: re-read the task from the durable store to
    // detect if a prior worker already completed this suspension (e.g.
    // our lease expired between spawn_tool_children and returning the
    // result, and another worker re-acquired and completed the flow).
    let current_task = deps
        .task_store
        .get(task_id)
        .await
        .context("re-read task for suspension idempotency check")?
        .context("task disappeared during suspension")?;

    if current_task.status == TaskStatus::WaitingOnChildren {
        // Close the attempt opened at the start of execute_root_turn so
        // it doesn't remain permanently unclosed in the audit trail.
        close_attempt_with(
            &attempt,
            TurnAttemptOutcome::Cancelled,
            deps.attempt_store,
            now,
        )
        .await;
        bail!(
            "task {task_id} already transitioned to WaitingOnChildren; \
             skipping duplicate suspension",
        );
    }

    // 1. Close the turn attempt — the LLM call itself succeeded.
    close_attempt_or_propagate_already_closed(
        &attempt,
        &response,
        deps,
        "close attempt on tool suspension",
        now,
    )
    .await?;

    // 2. Build the continuation envelope from current state + response.
    let continuation = build_continuation(&inputs, &response)
        .await
        .context("build continuation for tool suspension")?;

    // 3. Capture the suspended messages (user prompt + full assistant
    //    response including tool-use blocks) so the resume path can
    //    reconstruct the conversation from durable state alone.
    //    `user_input.into_message()` returns `None` for resume turns —
    //    those don't reach this branch (resume goes through
    //    `resume_root_turn` / `suspend_resumed_turn`), but the guard
    //    keeps the call site type-safe.
    let mut suspended_messages = Vec::with_capacity(2);
    if let Some(user_message) = user_input.clone().into_message() {
        suspended_messages.push(user_message);
    }
    suspended_messages.push(build_full_assistant_message(&response));

    // 4. One child task per tool call, inheriting the configured retry budget.
    let specs = child_spawn_specs_for_response(&response, &inputs);

    // Build content events (Thinking) from the tool-call response so
    // replay observers see the model's reasoning before tool dispatch.
    let content_events = build_content_events(&response, &close_ctx.content_ids);

    // Build ToolCallStart events before continuation is moved.
    let tool_call_events = build_tool_call_start_events(&continuation);

    // 4.b Consult the per-call subagent-spawn selector (if wired).
    //     A `SingleSubagent` / `MultiSubagent` verdict re-routes
    //     section 5 onto `spawn_subagent_invocation` /
    //     `spawn_subagent_batch_invocations`; everything else falls
    //     through to the regular `spawn_tool_children` path.
    let routing = classify_batch_for_inputs(&inputs, deps, &continuation).await?;

    // 5. Atomically spawn children and park the parent.
    //
    //    Clone `suspended_messages` before move into `SuspensionPayload`
    //    so the post-spawn `set_draft` call below can persist the same
    //    snapshot to the message projection's draft slot.
    let draft_snapshot = suspended_messages.clone();
    let payload = SuspensionPayload {
        continuation,
        suspended_messages,
    };
    let (parent_task, child_tasks) = apply_batch_routing(
        &inputs,
        deps,
        routing,
        payload,
        specs,
        "spawn tool children",
        now,
    )
    .await?;

    // 6. Snapshot the in-flight conversation to the projection's
    //    draft slot now that the suspension is durable on the parent
    //    task.
    snapshot_suspension_draft(
        deps,
        &inputs.bootstrap.thread_id,
        task_id,
        draft_snapshot,
        now,
        "snapshot suspended_messages to message projection draft",
    )
    .await;

    // `Start` was committed by `execute_root_turn` before streaming so
    // the per-delta TextDelta / ThinkingDelta events have a parent in
    // the journal.  This batch carries only the consolidated content
    // events (Thinking) plus the ToolCallStart events emitted AFTER
    // spawn_tool_children.  Since spawn_tool_children is CAS-guarded
    // (only the lease-holder can succeed), only the winning worker
    // writes these events.
    let mut suspension_events = content_events;
    suspension_events.extend(tool_call_events);
    let suspension_committed = deps
        .event_repo
        .commit_event_batch(&inputs.bootstrap.thread_id, suspension_events, now)
        .await
        .context("commit suspension events")?;

    // Prepend the durable per-turn admission events (`UserInput`
    // when present + `Start`) committed before streaming so the
    // outcome's `committed_events` represents every event committed
    // for this turn (matching the pre-streaming contract).
    let mut committed_events: Vec<CommittedEvent> = Vec::new();
    committed_events.extend(close_ctx.user_input_committed);
    committed_events.push(close_ctx.start_committed);
    committed_events.extend(suspension_committed);

    Ok(RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        committed_events,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Subagent spawn routing helpers
// ─────────────────────────────────────────────────────────────────────

/// Consult the configured [`SubagentSpawnSelector`] (if any) and
/// classify the resulting per-call decisions into a single
/// [`BatchRouting`] verdict.
///
/// Returns [`BatchRouting::AllTools`] when no selector is wired or when
/// the selector returns the wrong number of decisions (a defensive
/// fallback for a count contract the trait does not enforce).
///
/// A selector **error**, however, is propagated as the turn error: the
/// [`SubagentSpawnSelector`](super::subagent_spawn_selector::SubagentSpawnSelector)
/// contract reserves `Err` for "genuinely unrecoverable conditions"
/// and requires routing-level failures to resolve to
/// [`SubagentSpawnDecision::SpawnAsTool`](super::subagent_spawn_selector::SubagentSpawnDecision::SpawnAsTool)
/// instead. Swallowing the error into an `AllTools` reroute would both
/// contradict that contract and risk divergent decisions across a
/// lease-expiry retry (the selector is required to be deterministic),
/// corrupting the durable parent/child linkage.
async fn classify_batch_for_inputs(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    continuation: &ContinuationEnvelope,
) -> Result<super::subagent_spawn_selector::BatchRouting> {
    let Some(selector) = deps.subagent_spawn_selector else {
        return Ok(super::subagent_spawn_selector::BatchRouting::AllTools);
    };
    let pending = &continuation.payload.pending_tool_calls;
    let decisions = selector
        .decide(&inputs.bootstrap.thread_id, pending)
        .await
        .with_context(|| {
            format!(
                "subagent spawn selector failed on thread {} (unrecoverable per trait contract)",
                inputs.bootstrap.thread_id,
            )
        })?;
    if decisions.len() != pending.len() {
        log::warn!(
            "subagent spawn selector returned {} decisions for {} tool calls on thread {}; falling back to spawn_tool_children",
            decisions.len(),
            pending.len(),
            inputs.bootstrap.thread_id,
        );
        return Ok(super::subagent_spawn_selector::BatchRouting::AllTools);
    }
    Ok(super::subagent_spawn_selector::classify_batch(decisions))
}

/// Materialize a `SingleSubagent` routing verdict into a durable
/// [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation)
/// call, returning the durable records that the caller maps back
/// into [`RootTurnOutcome::Suspended`].
///
/// The invocation task surfaces in `child_tasks` of the outcome —
/// the host's task acquisition loop will pick it up under
/// [`TaskKind::Subagent`](crate::journal::task::TaskKind::Subagent)
/// and dispatch it through `execute_subagent_task_entry`. The child
/// thread's first root-turn task is also persisted by
/// `spawn_subagent_invocation` and rides the regular root-task
/// runnable queue from there.
async fn spawn_single_subagent_invocation(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    plan: &super::subagent_spawn_selector::SubagentSpawnPlan,
    payload: SuspensionPayload,
    spawn_index: usize,
    now: OffsetDateTime,
) -> Result<super::subagent::SpawnedSubagentInvocation> {
    let spawn_index_u32 = u32::try_from(spawn_index).context("subagent spawn_index exceeds u32")?;
    let spawn = crate::journal::store::SubagentInvocationSpawn {
        child_thread_id: plan.child_thread_id.clone(),
        spec: plan.spec.clone(),
        child_root_input: plan.child_root_input.clone(),
        spawn_index: spawn_index_u32,
        payload,
        child_caller_metadata: plan.child_caller_metadata.clone(),
    };
    let invocation_deps = super::subagent::SubagentInvocationDeps {
        task_store: deps.task_store,
        thread_store: deps.thread_store,
        event_repo: deps.event_repo,
    };
    super::subagent::spawn_subagent_invocation(
        &inputs.bootstrap.task_id,
        &inputs.bootstrap.worker_id,
        &inputs.bootstrap.lease_id,
        spawn,
        &invocation_deps,
        now,
    )
    .await
    .context("spawn subagent invocation")
}

/// Materialize a `MultiSubagent` routing verdict into N durable
/// subagent invocations under one parent transition.
///
/// Mirrors [`spawn_single_subagent_invocation`] for the fan-out
/// case. The shared [`SuspensionPayload`] is passed once (not cloned
/// per entry); the worker helper
/// [`super::subagent::spawn_subagent_batch_invocations`] then issues
/// a single store call against
/// [`AgentTaskStore::spawn_subagent_batch`](crate::journal::store::AgentTaskStore::spawn_subagent_batch).
async fn spawn_multi_subagent_invocations(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    plans: Vec<(
        usize,
        Box<super::subagent_spawn_selector::SubagentSpawnPlan>,
    )>,
    payload: SuspensionPayload,
    now: OffsetDateTime,
) -> Result<super::subagent::SpawnedSubagentBatch> {
    let mut entries = Vec::with_capacity(plans.len());
    for (spawn_index, plan) in plans {
        let spawn_index_u32 =
            u32::try_from(spawn_index).context("subagent spawn_index exceeds u32")?;
        entries.push(super::subagent::SubagentBatchEntry {
            child_thread_id: plan.child_thread_id.clone(),
            spec: plan.spec.clone(),
            child_root_input: plan.child_root_input.clone(),
            spawn_index: spawn_index_u32,
            child_caller_metadata: plan.child_caller_metadata.clone(),
        });
    }
    let invocation_deps = super::subagent::SubagentInvocationDeps {
        task_store: deps.task_store,
        thread_store: deps.thread_store,
        event_repo: deps.event_repo,
    };
    super::subagent::spawn_subagent_batch_invocations(
        &inputs.bootstrap.task_id,
        &inputs.bootstrap.worker_id,
        &inputs.bootstrap.lease_id,
        entries,
        payload,
        &invocation_deps,
        now,
    )
    .await
    .context("spawn subagent batch invocations")
}

/// Drive the routing match arm common to both `suspend_at_tool_boundary`
/// (initial suspend) and `suspend_resumed_turn` (post-resume re-suspend).
///
/// Maps a [`super::subagent_spawn_selector::BatchRouting`] verdict to
/// the correct store path:
///
/// * [`BatchRouting::SingleSubagent`](super::subagent_spawn_selector::BatchRouting::SingleSubagent)
///   → [`spawn_single_subagent_invocation`]
/// * [`BatchRouting::MultiSubagent`](super::subagent_spawn_selector::BatchRouting::MultiSubagent)
///   → [`spawn_multi_subagent_invocations`]
/// * [`BatchRouting::AllTools`](super::subagent_spawn_selector::BatchRouting::AllTools)
///   and [`BatchRouting::UnsupportedMixedBatch`](super::subagent_spawn_selector::BatchRouting::UnsupportedMixedBatch)
///   → [`AgentTaskStore::spawn_tool_children`](crate::journal::store::AgentTaskStore::spawn_tool_children)
///
/// Returns the parked parent task plus the materialized child tasks
/// (one entry for the single/multi subagent paths, N entries for
/// `spawn_tool_children`).
///
/// `tool_children_context` distinguishes the original "spawn tool
/// children" call site from the "re-spawn tool children on resume"
/// site — only used in error context strings so backtraces still
/// point at the right call.
async fn apply_batch_routing(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    routing: super::subagent_spawn_selector::BatchRouting,
    payload: SuspensionPayload,
    specs: Vec<ChildSpawnSpec>,
    tool_children_context: &'static str,
    now: OffsetDateTime,
) -> Result<(AgentTask, Vec<AgentTask>)> {
    let task_id = &inputs.bootstrap.task_id;
    let spawned = match routing {
        super::subagent_spawn_selector::BatchRouting::SingleSubagent { spawn_index, plan } => {
            let spawned =
                spawn_single_subagent_invocation(inputs, deps, &plan, payload, spawn_index, now)
                    .await?;
            (spawned.parent_task, vec![spawned.invocation_task])
        }
        super::subagent_spawn_selector::BatchRouting::MultiSubagent { plans } => {
            let batch = spawn_multi_subagent_invocations(inputs, deps, plans, payload, now).await?;
            let invocation_tasks = batch
                .invocations
                .into_iter()
                .map(|inv| inv.invocation_task)
                .collect();
            (batch.parent_task, invocation_tasks)
        }
        super::subagent_spawn_selector::BatchRouting::AllTools
        | super::subagent_spawn_selector::BatchRouting::UnsupportedMixedBatch => {
            // Stamp each child tool task with the turn's root
            // `invoke_agent` span as its trace parent (loaded from the
            // first attempt) so the child's `execute_tool` span nests
            // under the turn root rather than the inbound client span.
            #[cfg(feature = "otel")]
            let child_otel_traceparent = load_root_span_ids(deps.attempt_store, task_id)
                .await
                .and_then(|(trace, span)| {
                    agent_sdk::observability::loop_instrument::traceparent_from_ids(&trace, &span)
                });
            #[cfg(not(feature = "otel"))]
            let child_otel_traceparent: Option<String> = None;
            deps.task_store
                .spawn_tool_children(
                    task_id,
                    &inputs.bootstrap.worker_id,
                    &inputs.bootstrap.lease_id,
                    specs,
                    payload,
                    child_otel_traceparent,
                    now,
                )
                .await
                .context(tool_children_context)?
        }
    };

    // The batch is durably runnable now (children `Pending`, parent
    // parked). Nudge the pool so a parked worker claims a child on this
    // tick instead of the next `acquisition_interval` poll, and so every
    // child of the batch starts together rather than staggered across
    // successive ticks. The ticker remains the lost-wakeup backstop.
    deps.wake_workers_for_batch();
    Ok(spawned)
}

/// Close the parent's turn attempt, treating
/// [`TurnAttemptSchemaError::AlreadyClosed`] as a non-error (a prior
/// recovery sweep may have closed it after the lease expired and
/// before this worker reached the suspension point).
///
/// `error_context` is appended to any non-`AlreadyClosed` error so
/// callers preserve their original "close attempt on …" context
/// without duplicating the match arm.
async fn close_attempt_or_propagate_already_closed(
    attempt: &TurnAttempt,
    response: &llm::ChatResponse,
    deps: &RootTurnDeps<'_>,
    error_context: &'static str,
    now: OffsetDateTime,
) -> Result<()> {
    let params = build_close_params(response, attempt);
    match deps
        .attempt_store
        .close_attempt(&attempt.id, params, now)
        .await
    {
        Ok(_) => Ok(()),
        Err(e)
            if e.downcast_ref::<TurnAttemptSchemaError>()
                == Some(&TurnAttemptSchemaError::AlreadyClosed) =>
        {
            // Recovery sweep already closed this attempt — safe to proceed.
            Ok(())
        }
        Err(e) => Err(e.context(error_context)),
    }
}

/// Build the per-tool-call `ChildSpawnSpec` vector for one suspension.
///
/// One spec per pending tool call, all inheriting the configured retry
/// budget. `ChildSpawnSpec::default()` would silently use
/// `DEFAULT_MAX_ATTEMPTS=1` instead of the policy's budget, so do not
/// substitute it.
fn child_spawn_specs_for_response(
    response: &llm::ChatResponse,
    inputs: &RootWorkerInputs,
) -> Vec<ChildSpawnSpec> {
    let tool_call_count = response.tool_uses().count();
    let child_max_attempts = inputs.bootstrap.definition.policy.max_attempts;
    (0..tool_call_count)
        .map(|_| ChildSpawnSpec::new(child_max_attempts))
        .collect()
}

/// Materialize the [`AgentEvent::ToolCallStart`] vector from a
/// continuation's `pending_tool_calls`.
///
/// Same projection in both `suspend_at_tool_boundary` and
/// `suspend_resumed_turn` — extracted so the two suspend paths stay
/// in lockstep when the start-event shape changes.
fn build_tool_call_start_events(
    continuation: &agent_sdk_foundation::ContinuationEnvelope,
) -> Vec<AgentEvent> {
    continuation
        .payload
        .pending_tool_calls
        .iter()
        .map(|tc| {
            AgentEvent::tool_call_start(
                &tc.id,
                &tc.name,
                &tc.display_name,
                tc.input.clone(),
                tc.tier,
            )
        })
        .collect()
}

/// Snapshot the post-spawn `suspended_messages` to the projection's
/// draft slot.
///
/// Best-effort: a draft write failure must not roll back a successful
/// spawn. The recovery path simply degrades to the pre-fix behaviour
/// of seeing only committed history. Errors are logged and dropped.
async fn snapshot_suspension_draft(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    task_id: &AgentTaskId,
    draft_snapshot: Vec<llm::Message>,
    now: OffsetDateTime,
    label: &'static str,
) {
    if let Err(error) = deps
        .message_store
        .set_draft(thread_id, draft_snapshot, now)
        .await
    {
        log::warn!("failed to {label} for thread {thread_id} task {task_id}: {error:#}");
    }
}

/// Build a [`ContinuationEnvelope`] capturing the state at the tool
/// boundary.
///
/// The continuation includes:
/// - Thread/turn identity
/// - Cumulative and per-turn token usage
/// - The pending tool calls extracted from the LLM response
/// - The agent state snapshot (turn count and usage updated)
/// - LLM response metadata (response ID, stop reason)
///
/// [`ContinuationEnvelope`]: agent_sdk_foundation::ContinuationEnvelope
async fn build_continuation(
    inputs: &RootWorkerInputs,
    response: &llm::ChatResponse,
) -> Result<ContinuationEnvelope> {
    let thread_id = &inputs.bootstrap.thread_id;

    // Load current agent state from staged stores (seeded from the
    // latest checkpoint during recovery).
    let current_state = inputs
        .staged_stores
        .state
        .load(thread_id)
        .await?
        .unwrap_or_else(|| AgentState::new(thread_id.clone()));

    let turn_usage = response_token_usage(response);

    let mut total_usage = current_state.total_usage.clone();
    total_usage.add(&turn_usage);

    let updated_state = AgentState {
        turn_count: current_state.turn_count + 1,
        total_usage: total_usage.clone(),
        ..current_state
    };

    // Resolve the tier / display_name from the SAME per-turn tool list
    // the LLM request was built from (`build_chat_request` →
    // `definition.resolve_tools(caller_metadata)`), not the static
    // `definition.tools`. A `tools_fn` that hardens a tool to `Confirm`
    // for this caller (or exposes a tool absent from the static list)
    // must drive the confirmation gate — resolving against the static
    // list would silently weaken it. `resolve_tools` already falls back
    // to `definition.tools` when no `tools_fn` / caller metadata exists.
    let resolved_tools = inputs
        .bootstrap
        .definition
        .resolve_tools(inputs.bootstrap.task.caller_metadata.as_ref());
    let pending_tool_calls = extract_pending_tool_calls(response, &resolved_tools);
    let turn_number =
        usize::try_from(inputs.recovery_view.next_turn_number).context("turn number overflow")?;

    // `awaiting_index` and `completed_results` are artifacts of the
    // SDK's inline confirmation flow (sequential tool processing with
    // mid-batch pauses). The server dispatches all tool calls as child
    // tasks simultaneously, so neither field is meaningful here. They
    // are set to their zero values so the struct is well-formed.
    let continuation = AgentContinuation {
        thread_id: thread_id.clone(),
        turn: turn_number,
        total_usage,
        turn_usage,
        pending_tool_calls,
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: updated_state,
        response_id: sanitized_response_id(response),
        stop_reason: response.stop_reason,
        response_content: response.content.clone(),
    };

    Ok(ContinuationEnvelope::wrap(continuation))
}

/// Extract [`PendingToolCallInfo`] from each tool-use block in the
/// LLM response, resolving `tier` and `display_name` from `tool_defs`.
///
/// `tool_defs` MUST be the same per-turn list the chat request was
/// built from (`AgentDefinition::resolve_tools(caller_metadata)`), not
/// the static `definition.tools` — otherwise a `tools_fn` that hardens
/// a tool's tier (or exposes a tool absent from the static list) would
/// be silently overridden and the host's confirmation gate (which keys
/// on this tier) could be skipped. Tools missing from `tool_defs`
/// conservatively default to [`ToolTier::Confirm`].
///
/// [`PendingToolCallInfo`]: agent_sdk_foundation::PendingToolCallInfo
fn extract_pending_tool_calls(
    response: &llm::ChatResponse,
    tool_defs: &[llm::Tool],
) -> Vec<PendingToolCallInfo> {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let def = tool_defs.iter().find(|t| t.name == *name);
                Some(PendingToolCallInfo {
                    id: id.clone(),
                    name: name.clone(),
                    display_name: def.map_or_else(|| name.clone(), |d| d.display_name.clone()),
                    tier: def.map_or(ToolTier::Confirm, |d| d.tier),
                    input: input.clone(),
                    effective_input: input.clone(),
                    listen_context: None,
                })
            }
            _ => None,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// Resume from completed child tool results (Phase 4.5)
// ─────────────────────────────────────────────────────────────────────

/// The prior turn's continuation plus the replay-history prefix
/// (original user prompt + original assistant + the tool-results /
/// steering-interim user message) threaded into a re-suspension's
/// `suspended_messages`. Bundled so [`suspend_resumed_turn`] stays
/// within the argument-count budget.
struct ResumeSuspension<'a> {
    continuation: &'a AgentContinuation,
    history_prefix: Vec<llm::Message>,
}

/// Resume a suspended root turn after all child tool tasks have
/// reached terminal states.
///
/// This is the journal-driven resume path: the caller provides:
///
/// - `inputs`: execution context recovered from the latest checkpoint
///   (same as a fresh turn — the checkpoint is still at the pre-
///   suspension point because the suspension path does not commit).
/// - `continuation`: the [`AgentContinuation`] extracted from the
///   parent's [`TaskState::ReadyToResume`] payload before the worker
///   acquired the task.
/// - `suspended_messages`: the user prompt and full assistant response
///   (including tool-use blocks) captured at the original suspension
///   point, also from the [`TaskState::ReadyToResume`] payload.
/// - `child_results`: completed tool results, one per pending tool
///   call, keyed by tool-call ID.
///
/// # Execution flow
///
/// 1. Open a new turn attempt (audit record).
/// 2. Buffer suspended messages (user prompt + assistant with tool-use)
///    and tool-result messages into staged stores.
/// 3. Build a chat request from the full staged history.
/// 4. Call the LLM.
/// 5. Branch on response:
///    - **Text-only**: buffer the final response, drain staged stores,
///      commit via the same path as Phase 4.3, advance to `Completed`.
///    - **Tool calls**: re-suspend at the tool boundary (Phase 4.4
///      path), spawning new child tasks.
///
/// # Errors
///
/// - Staged store operations fail.
/// - LLM returns a non-success outcome.
/// - Commit path or task completion fails.
///
/// [`AgentContinuation`]: agent_sdk_foundation::AgentContinuation
/// [`TaskState::ReadyToResume`]: crate::journal::task_state::TaskState::ReadyToResume
#[allow(clippy::too_many_lines)]
pub async fn resume_root_turn(
    inputs: RootWorkerInputs,
    continuation: AgentContinuation,
    suspended_messages: Vec<llm::Message>,
    child_results: Vec<(String, ToolResult)>,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let definition = inputs.definition();
    let thread_id = &inputs.bootstrap.thread_id;

    // 1. Open a new turn attempt for the resume LLM call. The
    //    audit prompt is the canonical resume sentinel exported as
    //    `super::user_input::RESUME_AUDIT_PROMPT` so the worker and
    //    the resume path share a single source of truth for what
    //    "this attempt is a resume" looks like in the audit row.
    let resume_input = super::user_input::UserInput::resume();
    let resume_audit = resume_input.audit_summary();
    let attempt = open_attempt(
        &inputs,
        definition,
        &resume_audit,
        deps.attempt_store,
        now,
        None,
    )
    .await
    .context("open resume turn attempt")?;

    // Load the turn's root `invoke_agent` span context (created on the
    // fresh turn; ids persisted on the first attempt) so the resumed LLM
    // call's `chat` span joins the same trace. Applied via `with_context`
    // on the call future below — never a held `ContextGuard`, which would
    // make this future `!Send`.
    #[cfg(feature = "otel")]
    let resume_root_cx = load_root_span_ids(deps.attempt_store, &inputs.bootstrap.task_id)
        .await
        .and_then(|(trace, span)| {
            agent_sdk::observability::loop_instrument::remote_parent_context(&trace, &span)
        });

    // 2. Pre-call auto-compaction — run BEFORE buffering the in-flight
    //    suspended messages + tool results. The staged seed here is the
    //    COMMITTED-ONLY checkpoint history (the resume path seeds via
    //    `from_recovery_view_committed_only`), so compacting now keeps
    //    the durable `replace_history` rewrite restricted to committed
    //    messages. The uncommitted in-flight transcript never enters the
    //    committed projection, so a later permanent failure + recovery
    //    can't fold it in twice — once via the compacted projection and
    //    once via the raw draft. No-op when `deps.compaction_config` is
    //    `None`. (Reactive overflow compaction still folds the in-flight
    //    transcript in, but `apply_compaction` clears the draft so that
    //    path is duplication-safe too.)
    super::compaction::maybe_compact_staged_history(
        deps,
        &inputs.staged_stores.messages,
        thread_id,
        now,
    )
    .await
    .context("pre-call auto-compaction (resume path)")?;

    // 3. Buffer the suspended messages (user prompt + assistant with
    //    tool calls) and tool-result messages into the staged stores,
    //    on top of the (possibly compacted) committed seed.
    buffer_resume_messages(
        &inputs.staged_stores.messages,
        &inputs.staged_stores.state,
        thread_id,
        &continuation,
        &suspended_messages,
        &child_results,
    )
    .await
    .context("buffer resume messages")?;

    // 4. Refresh the projection's draft slot with the completed child
    //    results so recovery surfaces the full in-flight transcript.
    //    Best-effort: the draft is purely a recovery aid that degrades
    //    to committed-history-only, and the in-flight transcript is
    //    independently reconstructable from the parent's ReadyToResume
    //    state, so a transient store error must not abort an otherwise
    //    healthy resume (matches the suspend paths' draft contract).
    let resumed_draft = build_resumed_draft_messages(&suspended_messages, &child_results);
    snapshot_suspension_draft(
        deps,
        thread_id,
        &inputs.bootstrap.task_id,
        resumed_draft,
        now,
        "refresh draft with completed child results before resume LLM call",
    )
    .await;

    // 5. Build the chat request from staged history. `resume_input` is
    //    a resume sentinel whose `into_message()` returns `None`, so
    //    `build_chat_request` appends no extra user prompt — everything
    //    is already buffered. (This is why the resume path shares
    //    `build_chat_request` rather than a near-duplicate helper.)
    let chat_request = build_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
        &resume_input,
        inputs.bootstrap.task.caller_metadata.as_ref(),
    )
    .await
    .context("build resume chat request")?;

    // 4. Stream the LLM call.  The resumed turn doesn't get a fresh
    //    `Start` event — the original turn's Start is already in the
    //    journal and replay treats the resume as a continuation of
    //    that turn.  `call_llm` allocates per-block message/thinking
    //    IDs lazily during streaming so the per-delta TextDelta /
    //    ThinkingDelta events match the consolidated `Text` /
    //    `Thinking` events committed at turn close.
    let llm_call = call_llm_with_retry(LlmRetryParams {
        inputs: &inputs,
        definition,
        user_input: &resume_input,
        attempt_audit_prompt: &resume_audit,
        provider,
        chat_request,
        initial_attempt: attempt,
        deps,
        thread_id,
        now,
    });
    #[cfg(feature = "otel")]
    let streamed = {
        use opentelemetry::trace::FutureExt;
        match resume_root_cx {
            Some(cx) => llm_call.with_context(cx).await,
            None => llm_call.await,
        }
    };
    #[cfg(not(feature = "otel"))]
    let streamed = llm_call.await;

    let StreamedTurn {
        response,
        content_ids,
        attempt,
    } = match streamed {
        Ok(streamed) => streamed,
        Err(error) => {
            // Seam B (resume): the staged post-seed delta is
            // `[suspended…, tool_results]` — balanced by construction, so
            // it commits in full and the interrupted resume response is
            // dropped. This also permanently fixes the balanced-draft
            // drop: the delta becomes committed history instead of a
            // draft that dies at the next commit.
            if is_root_turn_cancelled(&error) {
                let candidate = inputs
                    .staged_stores
                    .messages
                    .snapshot_appended_messages()
                    .unwrap_or_default();
                commit_cancelled_partial_turn(&inputs, candidate, deps, now).await;
            }
            return Err(error);
        }
    };
    let commit_now = OffsetDateTime::now_utc();

    // 5. Branch: authorized tool handoff → re-suspend; otherwise commit.
    if response_requests_tool_dispatch(&response) {
        return suspend_resumed_turn(
            inputs,
            ResumeSuspension {
                continuation: &continuation,
                history_prefix: build_resumed_draft_messages(&suspended_messages, &child_results),
            },
            response,
            attempt,
            deps,
            commit_now,
            &content_ids,
        )
        .await;
    }

    commit_resumed_turn(
        inputs,
        &continuation,
        &response,
        &attempt,
        deps,
        commit_now,
        &content_ids,
    )
    .await
}

/// Buffer the final assistant message and update the staged agent state
/// with accumulated LLM usage from the resume response.
async fn buffer_resumed_assistant(
    inputs: &RootWorkerInputs,
    continuation: &AgentContinuation,
    response: &llm::ChatResponse,
) -> Result<()> {
    let thread_id = &inputs.bootstrap.thread_id;

    let assistant_msg = build_assistant_message(response);
    inputs
        .staged_stores
        .messages
        .append(thread_id, assistant_msg)
        .await
        .context("append resumed assistant message")?;

    // Add resume LLM usage to the continuation's running total.
    // Do NOT increment turn_count — it was already counted at the
    // original suspension point.
    let mut total_usage = continuation.state.total_usage.clone();
    total_usage.add(&response_token_usage(response));
    let updated_state = AgentState {
        total_usage,
        ..continuation.state.clone()
    };
    inputs
        .staged_stores
        .state
        .save(&updated_state)
        .await
        .context("save resumed agent state")?;

    Ok(())
}

/// Combine the prior continuation's per-turn usage with the new
/// response's usage so the resume commit records cumulative token
/// usage for the full turn (suspension + resume LLM calls).
fn merged_turn_usage(prior: &TokenUsage, response: &llm::ChatResponse) -> TokenUsage {
    let mut usage = prior.clone();
    usage.add(&response_token_usage(response));
    usage
}

/// Drain the staged message and state stores after a resumed turn so
/// the commit path has the messages to persist and the snapshot to
/// checkpoint.  Returns the drained messages, the snapshot value, and
/// the cumulative `total_usage` from the staged state.
fn drain_resumed_turn_state(
    inputs: &RootWorkerInputs,
) -> Result<(Vec<llm::Message>, serde_json::Value, TokenUsage)> {
    let drained_messages = inputs
        .staged_stores
        .messages
        .drain_messages()
        .context("drain resumed staged messages")?;
    let drained_state = inputs
        .staged_stores
        .state
        .drain_state()
        .context("drain resumed staged state")?
        .context("staged state was None after resume")?;
    let agent_state_snapshot =
        serde_json::to_value(&drained_state).context("serialize resumed agent state")?;
    Ok((
        drained_messages,
        agent_state_snapshot,
        drained_state.total_usage,
    ))
}

/// Commit a resumed turn whose LLM response is text-only.
///
/// Buffers the final assistant message, updates agent state, runs
/// the idempotency guard, drains staged stores, commits, and
/// advances the task to `Completed`.
async fn commit_resumed_turn(
    inputs: RootWorkerInputs,
    continuation: &AgentContinuation,
    response: &llm::ChatResponse,
    attempt: &TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
    content_ids: &ContentIds,
) -> Result<RootTurnOutcome> {
    let thread_id = &inputs.bootstrap.thread_id;
    let task_id = &inputs.bootstrap.task_id;

    let response_text = response.first_text().unwrap_or("").to_owned();

    buffer_resumed_assistant(&inputs, continuation, response).await?;

    let close_params = build_close_params(response, attempt);
    let turn_usage = merged_turn_usage(&continuation.turn_usage, response);

    let (drained_messages, agent_state_snapshot, total_usage) = drain_resumed_turn_state(&inputs)?;

    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);
    let duration = (now - attempt.opened_at).unsigned_abs();
    let lifecycle_events = build_turn_complete_events(
        response,
        thread_id,
        turn_number,
        &turn_usage,
        &total_usage,
        duration,
        content_ids,
    );

    let commit = commit_completed_turn_shifting_slot(
        CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            expected_turn: inputs.recovery_view.next_turn_number,
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params,
            messages: drained_messages,
            turn_usage,
            agent_state_snapshot,
            events: lifecycle_events,
            outbox_max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            owner_guard: None,
            now,
        },
        &inputs.bootstrap.worker_id,
        &inputs.bootstrap.lease_id,
        deps,
    )
    .await
    .context("commit resumed turn")?;

    let committed_events = commit.committed_events.clone();

    // Advance the root task to Completed.
    let (completed_task, resumed_invocation) = deps
        .task_store
        .complete_task(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            now,
        )
        .await
        .context("complete resumed root task")?;

    // When this root turn is a subagent's child root, its terminal
    // transition resumes the linked subagent invocation task
    // (`WaitingOnChildren` → `Pending`) in the same locked scope. Nudge
    // a parked worker so it claims the now-runnable invocation
    // immediately instead of waiting out the acquisition ticker. A
    // `None` (no linked invocation) or non-runnable row leaves the poll
    // backstop to handle it.
    if resumed_invocation.is_some_and(|t| t.status.is_runnable())
        && let Some(signal) = deps.wakeup
    {
        signal.notify_workers();
    }

    // The resumed turn completed (no further tool calls) — finalize the
    // root `invoke_agent` span with its full duration spanning the tool
    // executions and the resume. Usage lives on the per-call `chat` spans.
    #[cfg(feature = "otel")]
    agent_sdk::observability::loop_instrument::finalize_root_turn_span(
        task_id.as_str(),
        turn_number + 1,
        None,
        "done",
    );

    Ok(RootTurnOutcome::Completed {
        commit: Box::new(commit),
        completed_task,
        response_text,
        committed_events,
    })
}

/// Buffer the suspended messages, tool results, and agent state into
/// staged stores for the resume LLM call.
///
/// After this function, the staged message store contains:
/// `[checkpoint messages] + [user prompt] + [assistant with tool calls] + [tool results]`
async fn buffer_resume_messages(
    staged_messages: &crate::journal::staged::StagedMessageStore,
    staged_state: &crate::journal::staged::StagedStateStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    continuation: &AgentContinuation,
    suspended_messages: &[llm::Message],
    child_results: &[(String, ToolResult)],
) -> Result<()> {
    // Append the suspended messages (user prompt + assistant response
    // with tool-use blocks) captured at the original suspension point.
    for msg in suspended_messages {
        staged_messages
            .append(thread_id, msg.clone())
            .await
            .context("append suspended message")?;
    }

    // Build and append the tool-result user message from child results.
    let tool_result_msg = build_tool_results_message(child_results);
    staged_messages
        .append(thread_id, tool_result_msg)
        .await
        .context("append tool results message")?;

    // Seed the agent state from the continuation's snapshot so the
    // staged state store reflects the state at the suspension point.
    staged_state
        .save(&continuation.state)
        .await
        .context("save continuation agent state")?;

    Ok(())
}

/// Build a user message containing tool-result blocks for each
/// completed child task.
fn build_tool_results_message(child_results: &[(String, ToolResult)]) -> llm::Message {
    let blocks: Vec<llm::ContentBlock> = child_results
        .iter()
        .map(|(tool_use_id, result)| llm::ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: result.output.clone(),
            is_error: if result.success { None } else { Some(true) },
        })
        .collect();

    llm::Message::user_with_content(blocks)
}

fn build_resumed_draft_messages(
    suspended_messages: &[llm::Message],
    child_results: &[(String, ToolResult)],
) -> Vec<llm::Message> {
    let mut messages = Vec::with_capacity(suspended_messages.len() + 1);
    messages.extend_from_slice(suspended_messages);
    messages.push(build_tool_results_message(child_results));
    messages
}

/// Re-suspend a resumed turn when the LLM responds with more tool
/// calls. Buffers the full conversation (original suspended messages +
/// tool results + new assistant response) into the new suspension's
/// `suspended_messages` so a subsequent resume can reconstruct the
/// complete history.
async fn suspend_resumed_turn(
    inputs: RootWorkerInputs,
    prior: ResumeSuspension<'_>,
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
    content_ids: &ContentIds,
) -> Result<RootTurnOutcome> {
    let task_id = &inputs.bootstrap.task_id;

    // Close the turn attempt — the LLM call succeeded.
    close_attempt_or_propagate_already_closed(
        &attempt,
        &response,
        deps,
        "close attempt on resumed tool suspension",
        now,
    )
    .await?;

    // Build a new continuation that accumulates usage from the prior
    // continuation plus the new response.
    let new_continuation = build_resume_continuation(&inputs, prior.continuation, &response)
        .context("build resume continuation")?;

    // Build suspended messages that capture the FULL conversation
    // through this point: the caller-supplied replay-history prefix
    // (original user prompt + original assistant + the tool-results /
    // steering-interim user message) followed by the new assistant
    // response. Threading the prefix through — rather than rebuilding it
    // from `child_results` here — lets the steering all-terminal path
    // preserve its interim-results-plus-note user message in the replay
    // history instead of dropping the directive.
    let mut new_suspended = prior.history_prefix;
    // New assistant response (with new tool-use blocks).
    new_suspended.push(build_full_assistant_message(&response));

    // One child task per new tool call, inheriting the configured retry budget.
    let specs = child_spawn_specs_for_response(&response, &inputs);

    // Build content events (Thinking) from the resume response.
    let content_events = build_content_events(&response, content_ids);

    // Build ToolCallStart events before continuation is moved.
    let tool_call_events = build_tool_call_start_events(&new_continuation);

    // Consult the per-call subagent-spawn selector — same contract
    // as `suspend_at_tool_boundary`.
    let routing = classify_batch_for_inputs(&inputs, deps, &new_continuation).await?;

    // Clone the new accumulated `suspended_messages` so we can mirror
    // them into the projection's draft slot once the spawn succeeds.
    // See `suspend_at_tool_boundary` for the recovery contract.
    let draft_snapshot = new_suspended.clone();
    let payload = SuspensionPayload {
        continuation: new_continuation,
        suspended_messages: new_suspended,
    };
    let (parent_task, child_tasks) = apply_batch_routing(
        &inputs,
        deps,
        routing,
        payload,
        specs,
        "re-spawn tool children on resume",
        now,
    )
    .await?;

    // Refresh the projection draft so a subsequent failure on the
    // *next* resume LLM call still surfaces the full in-flight
    // history through `recover_thread`. Best-effort, same rationale
    // as `suspend_at_tool_boundary`.
    snapshot_suspension_draft(
        deps,
        &inputs.bootstrap.thread_id,
        task_id,
        draft_snapshot,
        now,
        "refresh resumed suspended_messages draft",
    )
    .await;

    // Commit content events (Thinking) followed by ToolCallStart events.
    let mut suspension_events = content_events;
    suspension_events.extend(tool_call_events);
    let committed_events = if suspension_events.is_empty() {
        Vec::new()
    } else {
        deps.event_repo
            .commit_event_batch(&inputs.bootstrap.thread_id, suspension_events, now)
            .await
            .context("commit resume suspension events")?
    };

    Ok(RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        committed_events,
    })
}

/// Build a [`ContinuationEnvelope`] for a re-suspension during resume,
/// accumulating usage from the prior continuation plus the new response.
fn build_resume_continuation(
    inputs: &RootWorkerInputs,
    prior: &AgentContinuation,
    response: &llm::ChatResponse,
) -> Result<ContinuationEnvelope> {
    let thread_id = &inputs.bootstrap.thread_id;

    let response_usage = response_token_usage(response);

    let mut new_turn_usage = prior.turn_usage.clone();
    new_turn_usage.add(&response_usage);

    let mut new_total_usage = prior.state.total_usage.clone();
    new_total_usage.add(&response_usage);

    let updated_state = AgentState {
        total_usage: new_total_usage.clone(),
        ..prior.state.clone()
    };

    // Resolve tier / display_name from the per-turn tool list (see
    // `build_continuation`) so a `tools_fn`-hardened tier drives the
    // confirmation gate on the resumed turn as well.
    let resolved_tools = inputs
        .bootstrap
        .definition
        .resolve_tools(inputs.bootstrap.task.caller_metadata.as_ref());
    let pending_tool_calls = extract_pending_tool_calls(response, &resolved_tools);
    let turn_number =
        usize::try_from(inputs.recovery_view.next_turn_number).context("turn number overflow")?;

    let continuation = AgentContinuation {
        thread_id: thread_id.clone(),
        turn: turn_number,
        total_usage: new_total_usage,
        turn_usage: new_turn_usage,
        pending_tool_calls,
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: updated_state,
        response_id: sanitized_response_id(response),
        stop_reason: response.stop_reason,
        response_content: response.content.clone(),
    };

    Ok(ContinuationEnvelope::wrap(continuation))
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.4: child outcome aggregation and durable resume
// ─────────────────────────────────────────────────────────────────────

/// Aggregate child task outcomes into the `(tool_call_id, ToolResult)`
/// pairs that [`resume_root_turn`] consumes.
///
/// This is the Phase 5.4 durable bridge: the parent's resume path
/// reads child outcomes exclusively from the journal, never from
/// in-memory channels or hidden state. Each child task row carries:
///
/// - **Completed**: a serialized [`ToolResult`] on
///   [`AgentTask::result_payload`], persisted by
///   [`execute_tool_task`](super::tool_task::execute_tool_task) via
///   [`AgentTaskStore::complete_task_with_result`].
/// - **Failed / Cancelled**: a deterministic error [`ToolResult`]
///   derived from [`AgentTask::last_error`] and [`AgentTask::status`].
///
/// Children are mapped to their parent's
/// [`AgentContinuation::pending_tool_calls`] via the positional
/// [`AgentTask::spawn_index`] field set during
/// [`AgentTaskStore::spawn_tool_children`]. The returned vector is
/// ordered by `spawn_index` so tool-result messages appear in the
/// same order the LLM emitted the tool-use blocks.
///
/// # Errors
///
/// - A child is not in a terminal state.
/// - A completed child has no `result_payload`.
/// - A child's `spawn_index` is out of bounds for the parent's
///   `pending_tool_calls`.
/// - The parent has no children.
///
/// [`AgentTaskStore::complete_task_with_result`]: crate::journal::store::AgentTaskStore::complete_task_with_result
/// [`AgentTaskStore::spawn_tool_children`]: crate::journal::store::AgentTaskStore::spawn_tool_children
pub async fn aggregate_child_outcomes(
    parent: &AgentTask,
    continuation: &AgentContinuation,
    task_store: &dyn AgentTaskStore,
) -> Result<Vec<(String, ToolResult)>> {
    let child_ids = parent.state.child_ids();

    ensure!(
        !child_ids.is_empty(),
        "parent {} has no child_ids to aggregate",
        parent.id
    );

    let pending = &continuation.pending_tool_calls;
    let mut results: Vec<Option<(String, ToolResult)>> = vec![None; pending.len()];

    for child_id in child_ids {
        let child = task_store
            .get(child_id)
            .await
            .with_context(|| format!("read child {child_id}"))?
            .with_context(|| format!("child {child_id} does not exist"))?;

        ensure!(
            child.status.is_terminal(),
            "child {} is not terminal (status {:?}); \
             cannot aggregate until all children finish",
            child.id,
            child.status,
        );

        let spawn_index = child.spawn_index.context("child missing spawn_index")?;
        let idx = usize::try_from(spawn_index).context("spawn_index exceeds usize")?;

        ensure!(
            idx < pending.len(),
            "child {} spawn_index {idx} out of bounds for {} pending tool calls",
            child.id,
            pending.len(),
        );

        let tool_call_id = pending[idx].id.clone();
        let tool_result = extract_child_tool_result(&child)?;

        results[idx] = Some((tool_call_id, tool_result));
    }

    results
        .into_iter()
        .enumerate()
        .map(|(i, slot)| {
            slot.with_context(|| format!("no child resolved pending tool call at index {i}"))
        })
        .collect()
}

/// Extract a [`ToolResult`] from a terminal child task.
///
/// - **Completed**: deserializes [`AgentTask::result_payload`].
/// - **Failed**: builds a deterministic error result from
///   [`AgentTask::last_error`].
/// - **Cancelled**: builds a deterministic cancellation result.
fn extract_child_tool_result(child: &AgentTask) -> Result<ToolResult> {
    match child.status {
        TaskStatus::Completed => {
            let payload = child
                .result_payload
                .as_ref()
                .with_context(|| format!("completed child {} has no result_payload", child.id))?;
            serde_json::from_value(payload.clone()).with_context(|| {
                format!("failed to deserialize result_payload on child {}", child.id)
            })
        }
        TaskStatus::Failed => {
            let error = child.last_error.as_deref().unwrap_or("unknown error");
            Ok(ToolResult {
                success: false,
                output: error.to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            })
        }
        TaskStatus::Cancelled => Ok(ToolResult {
            success: false,
            output: "tool execution was cancelled".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        }),
        other => bail!(
            "child {} has non-terminal status {:?}; cannot extract tool result",
            child.id,
            other,
        ),
    }
}

/// Resume a suspended root turn by aggregating child outcomes from
/// the journal and calling [`resume_root_turn`].
///
/// This is the Phase 5.4 entry point that bridges child completion
/// to root resume entirely through durable state. The caller
/// provides:
///
/// - `inputs`: execution context recovered from the latest checkpoint
///   (same as a fresh turn).
/// - `parent`: the parent task row, which must be in
///   [`TaskState::ReadyToResume`].
///
/// The function:
///
/// 1. Extracts the continuation and suspended messages from the
///    parent's [`TaskState::ReadyToResume`] payload.
/// 2. Aggregates child outcomes from the journal via
///    [`aggregate_child_outcomes`].
/// 3. Delegates to [`resume_root_turn`] with the aggregated results.
///
/// # Errors
///
/// - Parent is not in [`TaskState::ReadyToResume`].
/// - Child aggregation fails (non-terminal children, missing results).
/// - The underlying [`resume_root_turn`] fails.
pub async fn resume_from_children(
    inputs: RootWorkerInputs,
    parent: &AgentTask,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    // 1. Extract continuation and suspended messages.
    let (continuation, suspended_messages) = match &parent.state {
        TaskState::ReadyToResume {
            continuation,
            suspended_messages,
            ..
        } => (continuation.payload.clone(), suspended_messages.clone()),
        other => bail!(
            "resume_from_children requires ReadyToResume state, got {:?}",
            std::mem::discriminant(other),
        ),
    };

    // 2. Aggregate child outcomes from the journal.
    let child_results = aggregate_child_outcomes(parent, &continuation, deps.task_store)
        .await
        .context("aggregate child outcomes for resume")?;

    // 3. Delegate to the existing resume path. Box-pin the call so
    //    `resume_from_children`'s own future stays under the
    //    `clippy::large_futures` threshold — same pattern
    //    `execute_root_turn` uses for `execute_root_turn_inner`.
    Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        provider,
        deps,
        now,
    ))
    .await
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.5: steering wake (R2) — early resume of a parked parent
// ─────────────────────────────────────────────────────────────────────

/// `UUIDv5` namespace for deriving re-attach tool-use ids from the
/// original tool-use id. Deterministic derivation makes the re-park
/// idempotent across a lease-expiry requeue: re-running the steering
/// exchange from the durable steering `ReadyToResume` state re-issues the
/// identical ids, so a crash between the wake and the re-park never
/// orphans or duplicates a binding.
const STEERING_REATTACH_NAMESPACE: Uuid =
    Uuid::from_u128(0x9c47_5a17_e2b1_4f3d_8a6c_0b1e_5d90_2f7a);

/// Derive the re-issued tool-use id for a still-running child from its
/// current tool-use id. Stable across retries and distinct across
/// wake generations (each generation derives from the previous id).
pub(crate) fn derive_reattach_tool_use_id(original_id: &str) -> String {
    Uuid::new_v5(&STEERING_REATTACH_NAMESPACE, original_id.as_bytes()).to_string()
}

/// Typed interim tool-result payload for a child that is still running
/// when a steering wake fans in. Honest — never a fabricated child
/// result — the coordinator sees the child has not finished and that
/// its real result will arrive at the next fan-in.
fn steering_interim_tool_result() -> ToolResult {
    let payload = serde_json::json!({
        "status": "running",
        "note": "child is still executing; its final result will follow at the next fan-in",
    });
    ToolResult {
        success: true,
        output: payload.to_string(),
        data: None,
        documents: Vec::new(),
        duration_ms: None,
    }
}

/// A child still running at steering-wake time that must re-attach
/// under a freshly re-issued tool-use id.
struct SurvivingChild {
    /// Child task id — its journal binding is preserved (never
    /// re-spawned).
    child_id: AgentTaskId,
    /// Index into the *original* continuation's `pending_tool_calls`
    /// this child resolved. Used to look up its tool name/input for the
    /// re-issued binding.
    pending_index: usize,
}

/// Bundle of everything the steering LLM round + re-park consumes,
/// grouped so [`run_steering_exchange`] stays within the argument-count
/// budget.
struct SteeringExchange {
    /// The original (unchanged) continuation from the parked batch.
    continuation: AgentContinuation,
    /// Messages buffered at the original suspension point.
    suspended_messages: Vec<llm::Message>,
    /// Opaque steering content appended to the interim user message.
    steering: Vec<llm::ContentBlock>,
    /// Interim fan-in plan (results for every pending id + survivors).
    interim: SteeringInterim,
}

/// Interim fan-in plan built at steering-wake time.
struct SteeringInterim {
    /// One `(tool_use_id, ToolResult)` per original pending tool call,
    /// in pending order: real payloads for finished children, typed
    /// `status: running` payloads for still-running children. Every
    /// pending id is present — the wire contract forbids a partial
    /// fan-in.
    interim_results: Vec<(String, ToolResult)>,
    /// Still-running children, in pending order.
    surviving: Vec<SurvivingChild>,
}

/// Build the interim fan-in: read every parked child, emit a real or
/// typed-interim tool result for each pending tool-use id, and collect
/// the still-running children that will re-attach.
async fn build_steering_interim(
    continuation: &AgentContinuation,
    child_ids: &[AgentTaskId],
    task_store: &dyn AgentTaskStore,
) -> Result<SteeringInterim> {
    let pending = &continuation.pending_tool_calls;

    // Map spawn_index → child from the parked batch.
    let mut by_index: BTreeMap<usize, AgentTask> = BTreeMap::new();
    for child_id in child_ids {
        let child = task_store
            .get(child_id)
            .await
            .with_context(|| format!("read steering child {child_id}"))?
            .with_context(|| format!("steering child {child_id} does not exist"))?;
        let spawn_index = child
            .spawn_index
            .context("steering child missing spawn_index")?;
        let idx = usize::try_from(spawn_index).context("spawn_index exceeds usize")?;
        by_index.insert(idx, child);
    }

    let mut interim_results = Vec::with_capacity(pending.len());
    let mut surviving = Vec::new();
    for (idx, call) in pending.iter().enumerate() {
        let child = by_index.get(&idx).with_context(|| {
            format!("steering wake: no child resolves pending tool call at index {idx}")
        })?;
        if child.status.is_terminal() {
            // Finished during the wave — fold its real, already-augmented
            // result into the coordinator's history now.
            let result = extract_child_tool_result(child)?;
            interim_results.push((call.id.clone(), result));
        } else {
            interim_results.push((call.id.clone(), steering_interim_tool_result()));
            surviving.push(SurvivingChild {
                child_id: child.id.clone(),
                pending_index: idx,
            });
        }
    }

    Ok(SteeringInterim {
        interim_results,
        surviving,
    })
}

/// Build the one user message the steering LLM call sees: an interim
/// tool-result block for every pending tool-use id, followed by the
/// drained steering content. Exactly one user message keeps the
/// alternating-role wire contract intact.
fn build_steering_resume_message(
    interim_results: &[(String, ToolResult)],
    steering: &[llm::ContentBlock],
) -> llm::Message {
    let mut blocks: Vec<llm::ContentBlock> = interim_results
        .iter()
        .map(|(tool_use_id, result)| llm::ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: result.output.clone(),
            is_error: if result.success { None } else { Some(true) },
        })
        .collect();
    blocks.extend(steering.iter().cloned());
    llm::Message::user_with_content(blocks)
}

/// Build the re-attach bindings for the still-running children: one
/// [`PendingToolCallInfo`] (dense-ordered) plus its matching assistant
/// `ToolUse` content block, both keyed by a freshly re-issued
/// deterministic tool-use id.
fn build_steering_reattach(
    prior: &AgentContinuation,
    surviving: &[SurvivingChild],
) -> Result<(Vec<PendingToolCallInfo>, Vec<llm::ContentBlock>)> {
    let mut pending = Vec::with_capacity(surviving.len());
    let mut blocks = Vec::with_capacity(surviving.len());
    for child in surviving {
        let orig = prior
            .pending_tool_calls
            .get(child.pending_index)
            .with_context(|| {
                format!(
                    "steering re-attach: pending index {} out of bounds",
                    child.pending_index
                )
            })?;
        let new_id = derive_reattach_tool_use_id(&orig.id);
        pending.push(PendingToolCallInfo {
            id: new_id.clone(),
            name: orig.name.clone(),
            display_name: orig.display_name.clone(),
            tier: orig.tier,
            input: orig.input.clone(),
            effective_input: orig.effective_input.clone(),
            listen_context: orig.listen_context.clone(),
        });
        blocks.push(llm::ContentBlock::ToolUse {
            id: new_id,
            name: orig.name.clone(),
            input: orig.input.clone(),
            thought_signature: None,
        });
    }
    Ok((pending, blocks))
}

/// Build the combined assistant message re-parked into history: the
/// model's answer (text / thinking only — any tool-use it emitted is
/// dropped, since a steering reply is text and the re-park owns the
/// tool-use batch) followed by the synthetic re-attach `ToolUse`
/// blocks.
fn build_steering_assistant_message(
    response: &llm::ChatResponse,
    reattach_blocks: Vec<llm::ContentBlock>,
) -> llm::Message {
    let mut blocks: Vec<llm::ContentBlock> = response
        .content
        .iter()
        .filter(|block| {
            matches!(
                block,
                llm::ContentBlock::Text { .. }
                    | llm::ContentBlock::Thinking { .. }
                    | llm::ContentBlock::RedactedThinking { .. }
                    | llm::ContentBlock::OpaqueReasoning { .. }
            )
        })
        .cloned()
        .collect();
    blocks.extend(reattach_blocks);
    llm::Message {
        role: llm::Role::Assistant,
        content: llm::Content::Blocks(blocks),
    }
}

/// Build the re-parked continuation: accumulate usage, swap in the
/// re-issued pending tool calls, keep the same turn number (the mission
/// turn has not completed).
fn build_steering_continuation(
    prior: &AgentContinuation,
    response: &llm::ChatResponse,
    reattach_pending: Vec<PendingToolCallInfo>,
) -> ContinuationEnvelope {
    let response_usage = response_token_usage(response);
    let mut turn_usage = prior.turn_usage.clone();
    turn_usage.add(&response_usage);
    let mut total_usage = prior.state.total_usage.clone();
    total_usage.add(&response_usage);
    let updated_state = AgentState {
        total_usage: total_usage.clone(),
        ..prior.state.clone()
    };
    ContinuationEnvelope::wrap(AgentContinuation {
        thread_id: prior.thread_id.clone(),
        turn: prior.turn,
        total_usage,
        turn_usage,
        pending_tool_calls: reattach_pending,
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: updated_state,
        response_id: sanitized_response_id(response),
        stop_reason: response.stop_reason,
        response_content: response.content.clone(),
    })
}

/// Resume a parent woken for a steering exchange.
///
/// The parent is a [`TaskState::ReadyToResume`] row carrying a
/// non-empty `steering` payload; this runs one bounded LLM round and
/// then deterministically re-parks on the still-running children.
///
/// This is the R2 "steering wake" entry point (sibling of
/// [`resume_from_children`]). Unlike the all-at-once fan-in, it fires
/// while children are still running: the host acquired the `Pending`
/// row a steering-wake sweep produced via
/// [`AgentTaskStore::enqueue_steering_resume`], so the parent is now
/// `Running` and this worker owns its lease.
///
/// The exchange:
///
/// 1. Reads every parked child and builds ONE user message carrying an
///    interim tool-result for **every** pending tool-use id (real
///    payloads for finished children, typed `status: running` payloads
///    for still-running ones) followed by the drained steering content.
///    The wire contract is preserved — no pending id is omitted.
/// 2. Calls the LLM once (bounded — the founder-accepted "one extra
///    round per steering exchange").
/// 3. If children are still running, re-issues deterministic synthetic
///    tool-use ids for them, appends a combined assistant message
///    (the answer + the re-attach batch) to the durable history, and
///    re-parks via [`AgentTaskStore::repark_after_steering`] — the
///    still-running children re-bind to the new ids by `spawn_index`,
///    so their eventual real results reach the coordinator with nothing
///    lost or duplicated. No child is cancelled, re-spawned, or
///    re-counted.
/// 4. If every child already finished, commits the answer as the
///    completed turn (the ordinary text-only resume commit).
///
/// # Errors
/// - Parent is not a `ReadyToResume` row with a steering payload.
/// - Interim construction fails (a pending id with no resolving child).
/// - The LLM call, event commit, re-park, or commit fails.
pub async fn resume_for_steering(
    inputs: RootWorkerInputs,
    parent: &AgentTask,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    // An early steering resume is a `ReadyToResume` row carrying a
    // non-empty `steering` payload (the overload that keeps the durable
    // `kind` unchanged). An empty steering payload is an ordinary
    // fan-in and must go through `resume_from_children` instead.
    let (continuation, suspended_messages, child_ids, steering) = match &parent.state {
        TaskState::ReadyToResume {
            continuation,
            suspended_messages,
            child_ids,
            steering,
        } if !steering.is_empty() => (
            continuation.payload.clone(),
            suspended_messages.clone(),
            child_ids.clone(),
            steering.clone(),
        ),
        other => bail!(
            "resume_for_steering requires a ReadyToResume state with a steering payload, got {:?}",
            std::mem::discriminant(other),
        ),
    };

    let interim = build_steering_interim(&continuation, &child_ids, deps.task_store)
        .await
        .context("build steering interim fan-in")?;

    // Box-pin the heavy body so `resume_for_steering`'s own future
    // stays under the `clippy::large_futures` threshold — same pattern
    // `resume_from_children` uses.
    Box::pin(run_steering_exchange(
        inputs,
        SteeringExchange {
            continuation,
            suspended_messages,
            steering,
            interim,
        },
        provider,
        deps,
        now,
    ))
    .await
}

async fn run_steering_exchange(
    inputs: RootWorkerInputs,
    exchange: SteeringExchange,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let SteeringExchange {
        continuation,
        suspended_messages,
        steering,
        interim,
    } = exchange;

    // Stage the interim fan-in + steering note and run one bounded LLM
    // round.
    let steering_message = build_steering_resume_message(&interim.interim_results, &steering);
    let StreamedTurn {
        response,
        content_ids,
        attempt,
    } = stage_and_call_steering_llm(
        &inputs,
        deps,
        provider,
        &suspended_messages,
        &steering_message,
        &continuation.state,
        now,
    )
    .await?;
    let commit_now = OffsetDateTime::now_utc();

    // Every child already finished: the interim message carried all
    // real results, so the steering answer resolves the turn.
    if interim.surviving.is_empty() {
        // A steering reply that itself acts on the redirect ("change of
        // plan: also do X") emits tool_use. With no survivors to
        // re-attach, there is no re-park batch to conflict with, so we
        // mirror the ordinary resume path and spawn the follow-up wave
        // instead of dropping the tool_use and force-completing the
        // mission turn (which `commit_resumed_turn` would do via
        // `build_assistant_message`'s tool_use filter). The steering
        // user message (interim results + note) is threaded into the
        // replay history so the directive survives for the next resume.
        if response_requests_tool_dispatch(&response) {
            let mut history_prefix = suspended_messages;
            history_prefix.push(steering_message);
            return suspend_resumed_turn(
                inputs,
                ResumeSuspension {
                    continuation: &continuation,
                    history_prefix,
                },
                response,
                attempt,
                deps,
                commit_now,
                &content_ids,
            )
            .await;
        }

        // Text-only steering answer: the ordinary text-only resume
        // commit completes the mission turn.
        return commit_resumed_turn(
            inputs,
            &continuation,
            &response,
            &attempt,
            deps,
            commit_now,
            &content_ids,
        )
        .await;
    }

    // Otherwise re-park on the still-running children under fresh bindings.
    repark_after_steering_exchange(
        &inputs,
        deps,
        SteeringRepark {
            continuation,
            suspended_messages,
            steering_message,
            surviving: interim.surviving,
            response,
            content_ids,
            attempt,
        },
        commit_now,
    )
    .await
}

/// Open a resume-style turn attempt, buffer the steering transcript
/// into the staged stores, and stream one bounded LLM round. Returns
/// the streamed turn (response + content ids + successful attempt).
///
/// Mirrors the resume path's setup: no fresh `Start` event (the
/// original turn's `Start` anchors replay), pre-call auto-compaction on
/// the committed-only seed, and the resume sentinel user input so
/// `build_chat_request` appends no extra prompt.
async fn stage_and_call_steering_llm(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    provider: &dyn LlmProvider,
    suspended_messages: &[llm::Message],
    steering_message: &llm::Message,
    state: &AgentState,
    now: OffsetDateTime,
) -> Result<StreamedTurn> {
    let definition = inputs.definition();
    let thread_id = &inputs.bootstrap.thread_id;

    let resume_input = super::user_input::UserInput::resume();
    let resume_audit = resume_input.audit_summary();
    let attempt = open_attempt(
        inputs,
        definition,
        &resume_audit,
        deps.attempt_store,
        now,
        None,
    )
    .await
    .context("open steering resume turn attempt")?;

    #[cfg(feature = "otel")]
    let resume_root_cx = load_root_span_ids(deps.attempt_store, &inputs.bootstrap.task_id)
        .await
        .and_then(|(trace, span)| {
            agent_sdk::observability::loop_instrument::remote_parent_context(&trace, &span)
        });

    // Pre-call auto-compaction on the committed-only seed (same
    // contract as the resume path). No-op when unconfigured.
    super::compaction::maybe_compact_staged_history(
        deps,
        &inputs.staged_stores.messages,
        thread_id,
        now,
    )
    .await
    .context("pre-call auto-compaction (steering resume path)")?;

    // Buffer suspended messages + the interim-results-and-steering user
    // message on top of the committed seed, and refresh the recovery
    // draft with the full in-flight transcript.
    for msg in suspended_messages {
        inputs
            .staged_stores
            .messages
            .append(thread_id, msg.clone())
            .await
            .context("append suspended message (steering)")?;
    }
    inputs
        .staged_stores
        .messages
        .append(thread_id, steering_message.clone())
        .await
        .context("append steering interim + note message")?;
    inputs
        .staged_stores
        .state
        .save(state)
        .await
        .context("save continuation agent state (steering)")?;
    let mut draft = suspended_messages.to_vec();
    draft.push(steering_message.clone());
    snapshot_suspension_draft(
        deps,
        thread_id,
        &inputs.bootstrap.task_id,
        draft,
        now,
        "snapshot steering resume draft",
    )
    .await;

    let chat_request = build_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
        &resume_input,
        inputs.bootstrap.task.caller_metadata.as_ref(),
    )
    .await
    .context("build steering resume chat request")?;

    let llm_call = call_llm_with_retry(LlmRetryParams {
        inputs,
        definition,
        user_input: &resume_input,
        attempt_audit_prompt: &resume_audit,
        provider,
        chat_request,
        initial_attempt: attempt,
        deps,
        thread_id,
        now,
    });
    #[cfg(feature = "otel")]
    {
        use opentelemetry::trace::FutureExt;
        match resume_root_cx {
            Some(cx) => llm_call.with_context(cx).await,
            None => llm_call.await,
        }
    }
    #[cfg(not(feature = "otel"))]
    {
        llm_call.await
    }
}

/// Everything the steering re-park tail consumes, grouped so
/// [`repark_after_steering_exchange`] stays within the argument-count
/// budget.
struct SteeringRepark {
    continuation: AgentContinuation,
    suspended_messages: Vec<llm::Message>,
    steering_message: llm::Message,
    surviving: Vec<SurvivingChild>,
    response: llm::ChatResponse,
    content_ids: ContentIds,
    attempt: TurnAttempt,
}

/// Re-park a running steering exchange on the still-running children:
/// close the attempt, re-issue deterministic tool-use ids, append the
/// combined assistant answer + re-attach batch to the durable history,
/// re-park via [`AgentTaskStore::repark_after_steering`], and commit the
/// answer's content events so the user sees the mid-wave reply.
async fn repark_after_steering_exchange(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    repark: SteeringRepark,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let SteeringRepark {
        continuation,
        suspended_messages,
        steering_message,
        surviving,
        response,
        content_ids,
        attempt,
    } = repark;
    let thread_id = &inputs.bootstrap.thread_id;

    if response.has_tool_use() {
        log::warn!(
            "steering resume on thread {thread_id}: model emitted tool_use in a steering \
             reply; dropping it — a steering exchange is bounded to a reply and the re-park \
             owns the tool-use batch. Directives take effect at the next fan-in.",
        );
    }

    // Close the attempt — the LLM call itself succeeded.
    close_attempt_or_propagate_already_closed(
        &attempt,
        &response,
        deps,
        "close attempt on steering re-park",
        now,
    )
    .await?;

    let (reattach_pending, reattach_blocks) = build_steering_reattach(&continuation, &surviving)
        .context("build steering re-attach bindings")?;
    let reattach_child_ids: Vec<AgentTaskId> =
        surviving.iter().map(|c| c.child_id.clone()).collect();

    let assistant_message = build_steering_assistant_message(&response, reattach_blocks);
    let new_continuation = build_steering_continuation(&continuation, &response, reattach_pending);

    // Full replay history for the re-parked continuation: original
    // suspended messages + interim/steering user message + combined
    // assistant answer with the re-attach batch.
    let mut new_suspended = Vec::with_capacity(suspended_messages.len() + 2);
    new_suspended.extend(suspended_messages);
    new_suspended.push(steering_message);
    new_suspended.push(assistant_message);

    let payload = SuspensionPayload {
        continuation: new_continuation,
        suspended_messages: new_suspended.clone(),
    };
    let reparked = deps
        .task_store
        .repark_after_steering(
            &inputs.bootstrap.task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            payload,
            reattach_child_ids,
            now,
        )
        .await
        .context("re-park parent after steering exchange")?;

    // Refresh the projection draft with the re-parked transcript so a
    // recovery before the next fan-in surfaces the full history.
    snapshot_suspension_draft(
        deps,
        thread_id,
        &inputs.bootstrap.task_id,
        new_suspended,
        now,
        "refresh steering re-park draft",
    )
    .await;

    // Commit the answer's content events so the user sees the mid-wave
    // reply. No `TurnComplete` / `Done` — the mission turn is still
    // parked. No `ToolCallStart` for the re-attach batch — those are
    // synthetic re-bindings, not new tool starts.
    let content_events = build_content_events(&response, &content_ids);
    let committed_events = if content_events.is_empty() {
        Vec::new()
    } else {
        deps.event_repo
            .commit_event_batch(thread_id, content_events, now)
            .await
            .context("commit steering answer content events")?
    };

    Ok(RootTurnOutcome::Suspended {
        parent_task: reparked,
        child_tasks: Vec::new(),
        committed_events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_tool_use_or_missing_stop_reason_allows_tool_dispatch() {
        assert!(stop_reason_allows_tool_dispatch(None));
        assert!(stop_reason_allows_tool_dispatch(Some(
            llm::StopReason::ToolUse
        )));

        for stop_reason in [
            llm::StopReason::EndTurn,
            llm::StopReason::MaxTokens,
            llm::StopReason::StopSequence,
            llm::StopReason::Refusal,
            llm::StopReason::ModelContextWindowExceeded,
            llm::StopReason::Unknown,
        ] {
            assert!(
                !stop_reason_allows_tool_dispatch(Some(stop_reason)),
                "terminal stop reason {stop_reason:?} must not authorize tool dispatch"
            );
        }
    }
}
