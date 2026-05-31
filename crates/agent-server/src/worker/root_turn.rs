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
//! [`ContinuationEnvelope`]: agent_sdk_core::ContinuationEnvelope
//! [`spawn_tool_children`]: crate::journal::store::AgentTaskStore::spawn_tool_children
//! [`TaskState::WaitingOnChildren`]: crate::journal::task_state::TaskState::WaitingOnChildren

use agent_sdk_core::audit::AuditProvenance;
use agent_sdk_core::events::AgentEvent;
use agent_sdk_core::llm::{self, ChatRequest};
use agent_sdk_core::{
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
use uuid::Uuid;

use super::definition::{AgentDefinition, ThinkingPolicy};
use crate::journal::checkpoint_store::CheckpointStore;
use crate::journal::commit::{
    CommitOutcome, CompletedTurnCommit, DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS, commit_completed_turn,
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
/// # Returns
///
/// # Errors
///
/// Returns an error if the store rejects the transition (e.g. task is
/// already terminal from a concurrent cancel).
pub async fn fail_root_turn(
    task_id: &AgentTaskId,
    worker_id: &WorkerId,
    lease_id: &LeaseId,
    thread_id: &agent_sdk_core::ThreadId,
    error: &anyhow::Error,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<AgentTask> {
    // Best-effort close any open turn attempts for this task.
    best_effort_close_open_attempts(task_id, deps.attempt_store, now).await;

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

    Ok(failed_task)
}

/// Cancel a root turn and its entire subtree.
///
/// Calls [`AgentTaskStore::cancel_tree`] to atomically cancel the root
/// task and any live descendant tasks (e.g. `tool_runtime` children
/// spawned during suspension). Best-effort closes any open turn
/// attempts for the root task.
///
/// Because staged projections are in-memory only and no durable writes
/// occur on the suspension path, cancellation at any lifecycle point
/// leaves the journal in a coherent state: the task is terminal, its
/// `TaskState` is cleared to `None`, and no stale projections exist.
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
    best_effort_close_open_attempts(task_id, deps.attempt_store, now).await;

    deps.task_store
        .cancel_tree(task_id, now)
        .await
        .context("cancel root turn tree")
}

/// Best-effort close any open (non-closed) turn attempts for a task.
///
/// Iterates all attempts and closes any that are still open with the
/// `Cancelled` outcome. Errors are swallowed — the caller's primary
/// operation (fail or cancel) takes precedence.
async fn best_effort_close_open_attempts(
    task_id: &AgentTaskId,
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
) {
    let Ok(attempts) = attempt_store.list_by_task(task_id).await else {
        return;
    };
    for attempt in &attempts {
        if !attempt.is_closed() {
            close_attempt_with(attempt, TurnAttemptOutcome::Cancelled, attempt_store, now).await;
        }
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
/// - **Tool calls present**: builds a [`ContinuationEnvelope`], spawns
///   one `tool_runtime` child per tool call, and parks the parent task
///   in `WaitingOnChildren`.
///
/// # Errors
///
/// - LLM returns a non-success outcome (rate limit, server error, etc.)
/// - Commit path fails (text-only)
/// - Task completion fails (text-only)
/// - Child spawn fails (tool suspension)
///
/// [`ContinuationEnvelope`]: agent_sdk_core::ContinuationEnvelope
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
    let attempt = open_attempt(&inputs, definition, &audit_prompt, deps.attempt_store, now)
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

    let StreamedTurn {
        response,
        content_ids,
        attempt,
    } = call_llm_with_retry(LlmRetryParams {
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
    })
    .await?;

    // Capture a post-LLM timestamp so the turn attempt's duration_ms
    // reflects actual wall-clock latency instead of always being 0.
    let commit_now = OffsetDateTime::now_utc();
    let close_ctx = TurnCloseContext {
        content_ids,
        user_input_committed,
        start_committed,
    };

    // 4. Branch: tool calls → suspend; text-only → commit.
    //
    // Start was committed in step 2 (before streaming).  The branches
    // commit only the consolidated content events, `TurnComplete`, and
    // `Done` — atomically, with the message-projection write.
    if response.has_tool_use() {
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

    ensure_turn_not_already_committed(
        deps.thread_store,
        thread_id,
        inputs.recovery_view.next_turn_number,
    )
    .await?;

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

    let commit = commit_completed_turn(
        CompletedTurnCommit {
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params,
            messages: drained_messages,
            turn_usage,
            agent_state_snapshot,
            events: lifecycle_events,
            outbox_max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        },
        deps.thread_store,
        deps.message_store,
        deps.attempt_store,
        deps.checkpoint_store,
        deps.event_repo,
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

    let (completed_task, _parent) = deps
        .task_store
        .complete_task(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            now,
        )
        .await
        .context("complete root task")?;

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
    thread_id: &agent_sdk_core::ThreadId,
    expected_turn: u32,
) -> Result<()> {
    let current_thread = thread_store
        .get(thread_id)
        .await
        .context("re-read thread for idempotency check")?
        .context("thread disappeared during turn execution")?;

    ensure!(
        current_thread.committed_turns < expected_turn,
        "turn {expected_turn} was already committed on thread {} \
         (committed_turns={}); skipping duplicate commit",
        thread_id,
        current_thread.committed_turns,
    );

    Ok(())
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
    let attempt_number = u32::try_from(existing.len()).context("attempt count overflow")? + 1;

    attempt_store
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number,
            provenance,
            request_blob,
            now,
            // TODO: once `agent-server` opts into OpenTelemetry, capture
            // the live span context here so `call_llm_with_retry` can attach
            // a `replay-of` link on the next attempt.
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await
        .context("open_attempt via store")
}

async fn build_chat_request(
    definition: &AgentDefinition,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
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
    })
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
/// attempts (~14 s worst-case at the default backoff) covers a longer
/// transient blip without dragging out a genuinely poisoned turn.
const STREAM_MAX_RETRIES: u32 = 3;

/// Base backoff for the exponential retry schedule (`base * 2^(n-1)`
/// with jitter, capped at [`STREAM_MAX_DELAY_MS`]).
const STREAM_BASE_DELAY_MS: u64 = 500;

/// Upper bound on the retry backoff.  The cap matches the
/// daemon-reconnect ceiling in the bipi/desktop loops so a transient
/// blip and a daemon respawn share the same wall-clock vocabulary.
const STREAM_MAX_DELAY_MS: u64 = 8_000;

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
    /// every retry.  Cloning per attempt is cheap because the
    /// expensive parts (system prompt + staged history) are shared
    /// `String` / `Vec<Message>` payloads.
    pub(crate) chat_request: ChatRequest,
    /// Initial [`TurnAttempt`] opened by the caller before streaming
    /// begins.  Subsequent attempts are minted by the wrapper.
    pub(crate) initial_attempt: TurnAttempt,
    /// Store handles for committing delta events, closing failed
    /// attempts, and emitting `AutoRetryStart` / `AutoRetryEnd`.
    pub(crate) deps: &'a RootTurnDeps<'a>,
    /// Thread under which every event commit and audit close is
    /// attributed.
    pub(crate) thread_id: &'a agent_sdk_core::ThreadId,
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
    thread_id: &'a agent_sdk_core::ThreadId,
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
    tokio::time::sleep(delay).await;
    open_attempt(
        inputs,
        definition,
        attempt_audit_prompt,
        deps.attempt_store,
        OffsetDateTime::now_utc(),
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
    thread_id: &'a agent_sdk_core::ThreadId,
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
    thread_id: &agent_sdk_core::ThreadId,
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
    thread_id: &agent_sdk_core::ThreadId,
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
async fn call_llm_once(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    attempt: &TurnAttempt,
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_core::ThreadId,
    now: OffsetDateTime,
) -> Result<OnceOutcome, StreamAttemptError> {
    let mut stream = std::pin::pin!(provider.chat_stream(request));
    let mut accumulator = StreamAccumulator::new();
    let mut content_ids = ContentIds::default();
    let mut delta_count: u64 = 0;

    while let Some(result) = stream.next().await {
        let delta = match result {
            Ok(delta) => delta,
            Err(error) => {
                // Transport-level stream errors (HTTP read failures,
                // dropped TCP, etc.) are treated as recoverable —
                // they're the most common shape of "Anthropic SSE
                // stream died mid-flight" and almost always succeed
                // on retry.
                close_attempt_with(
                    attempt,
                    TurnAttemptOutcome::ServerError,
                    deps.attempt_store,
                    now,
                )
                .await;
                return Err(StreamAttemptError::Recoverable {
                    kind: StreamErrorKind::ServerError,
                    message: format!("LLM stream iteration error: {error:#}"),
                });
            }
        };

        delta_count = delta_count.saturating_add(1);
        accumulator.apply(&delta);

        match &delta {
            StreamDelta::TextDelta {
                delta: text_chunk,
                block_index,
            } => {
                if text_chunk.is_empty() {
                    // Empty deltas carry no content; skipping them
                    // also avoids allocating an id for blocks that
                    // never produce non-empty text.
                    continue;
                }
                let message_id = content_ids.text_id_for(*block_index);
                commit_streaming_delta(
                    deps,
                    thread_id,
                    AgentEvent::text_delta(message_id.to_string(), text_chunk.clone()),
                    "text_delta",
                    delta_count,
                )
                .await;
            }
            StreamDelta::ThinkingDelta {
                delta: thinking_chunk,
                block_index,
            } => {
                if thinking_chunk.is_empty() {
                    continue;
                }
                let thinking_id = content_ids.thinking_id_for(*block_index);
                commit_streaming_delta(
                    deps,
                    thread_id,
                    AgentEvent::thinking_delta(thinking_id.to_string(), thinking_chunk.clone()),
                    "thinking_delta",
                    delta_count,
                )
                .await;
            }
            StreamDelta::Error { message, kind } => {
                // Map the kind directly onto the audit outcome so a
                // genuine 5xx is recorded as `ServerError` (not
                // `RateLimited`) and a validation rejection is
                // recorded as `InvalidRequest` (not `ServerError`).
                let outcome = match kind {
                    StreamErrorKind::RateLimited => TurnAttemptOutcome::RateLimited,
                    StreamErrorKind::ServerError => TurnAttemptOutcome::ServerError,
                    StreamErrorKind::InvalidRequest => TurnAttemptOutcome::InvalidRequest,
                };
                close_attempt_with(attempt, outcome, deps.attempt_store, now).await;
                let message = message.clone();
                let kind = *kind;
                if kind.is_recoverable() {
                    return Err(StreamAttemptError::Recoverable { kind, message });
                }
                return Err(StreamAttemptError::Fatal {
                    message: format!("LLM stream error (kind={kind:?}): {message}"),
                });
            }
            // Done / Usage / ToolUseStart / ToolInputDelta /
            // SignatureDelta / RedactedThinking are handled by the
            // accumulator and don't need to be re-emitted as events.
            StreamDelta::Done { .. }
            | StreamDelta::Usage(_)
            | StreamDelta::ToolUseStart { .. }
            | StreamDelta::ToolInputDelta { .. }
            | StreamDelta::SignatureDelta { .. }
            | StreamDelta::RedactedThinking { .. } => {}
        }
    }

    let response = synthesize_response(accumulator, provider, thread_id);
    Ok(OnceOutcome {
        response,
        content_ids,
    })
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
    thread_id: &agent_sdk_core::ThreadId,
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

/// Commit a single streaming delta event with a freshly captured
/// timestamp and best-effort live-tail notification.
///
/// Delta commit failures are intentionally non-fatal: the consolidated
/// `Text` / `Thinking` event committed at turn close still captures
/// the full content for replay clients, so a transient journal error
/// during streaming should not abort the turn.
async fn commit_streaming_delta(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_core::ThreadId,
    event: AgentEvent,
    event_label: &str,
    delta_count: u64,
) {
    // Use a fresh timestamp per delta so the journal records true
    // wall-clock arrival times, not the stale turn-open `now` shared
    // by every delta in a long streaming response.
    let delta_now = OffsetDateTime::now_utc();
    match deps
        .event_repo
        .commit_event(thread_id, event, delta_now)
        .await
    {
        Ok(committed) => {
            deps.event_notifier.notify(std::slice::from_ref(&committed));
        }
        Err(error) => {
            log::warn!(
                "failed to commit {event_label} event for thread {thread_id} \
                 (delta_count={delta_count}): {error:#}",
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

async fn buffer_turn_messages(
    staged_messages: &crate::journal::staged::StagedMessageStore,
    staged_state: &crate::journal::staged::StagedStateStore,
    thread_id: &agent_sdk_core::ThreadId,
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

    let updated_state = AgentState {
        turn_count: current_state.turn_count + 1,
        total_usage: TokenUsage {
            input_tokens: current_state
                .total_usage
                .input_tokens
                .saturating_add(response.usage.input_tokens),
            output_tokens: current_state
                .total_usage
                .output_tokens
                .saturating_add(response.usage.output_tokens),
            cached_input_tokens: current_state
                .total_usage
                .cached_input_tokens
                .saturating_add(response.usage.cached_input_tokens),
            cache_creation_input_tokens: current_state
                .total_usage
                .cache_creation_input_tokens
                .saturating_add(response.usage.cache_creation_input_tokens),
        },
        ..current_state
    };
    staged_state
        .save(&updated_state)
        .await
        .context("save updated agent state")?;

    Ok(())
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
/// [`AgentContinuation::response_id`](agent_sdk_core::AgentContinuation::response_id)
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
    thread_id: &agent_sdk_core::ThreadId,
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
/// [`ContinuationEnvelope`]: agent_sdk_core::ContinuationEnvelope
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
/// Returns [`BatchRouting::AllTools`] when no selector is wired, when
/// the selector returns the wrong number of decisions, or when the
/// selector itself errors — in every "something looked off" case we
/// fall through to the legacy `spawn_tool_children` path so a broken
/// selector cannot break a turn entirely.
async fn classify_batch_for_inputs(
    inputs: &RootWorkerInputs,
    deps: &RootTurnDeps<'_>,
    continuation: &ContinuationEnvelope,
) -> Result<super::subagent_spawn_selector::BatchRouting> {
    let Some(selector) = deps.subagent_spawn_selector else {
        return Ok(super::subagent_spawn_selector::BatchRouting::AllTools);
    };
    let pending = &continuation.payload.pending_tool_calls;
    let decisions = match selector.decide(&inputs.bootstrap.thread_id, pending).await {
        Ok(decisions) => decisions,
        Err(error) => {
            log::warn!(
                "subagent spawn selector failed on thread {}: {error:#} — falling back to spawn_tool_children",
                inputs.bootstrap.thread_id,
            );
            return Ok(super::subagent_spawn_selector::BatchRouting::AllTools);
        }
    };
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
/// case. The shared [`SuspensionPayload`] flows once into every
/// per-entry `SubagentInvocationSpawn`; the worker helper
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
    let mut spawns = Vec::with_capacity(plans.len());
    for (spawn_index, plan) in plans {
        let spawn_index_u32 =
            u32::try_from(spawn_index).context("subagent spawn_index exceeds u32")?;
        spawns.push(crate::journal::store::SubagentInvocationSpawn {
            child_thread_id: plan.child_thread_id.clone(),
            spec: plan.spec.clone(),
            child_root_input: plan.child_root_input.clone(),
            spawn_index: spawn_index_u32,
            // `spawn_subagent_batch_invocations` reads only the first
            // entry's `payload.continuation` for index validation;
            // the store-side primitive uses the explicit shared
            // `payload` argument for the parent's suspension. Cloning
            // keeps each `SubagentInvocationSpawn` self-contained so
            // selector callers can construct them in any order.
            payload: payload.clone(),
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
        spawns,
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
    match routing {
        super::subagent_spawn_selector::BatchRouting::SingleSubagent { spawn_index, plan } => {
            let spawned =
                spawn_single_subagent_invocation(inputs, deps, &plan, payload, spawn_index, now)
                    .await?;
            Ok((spawned.parent_task, vec![spawned.invocation_task]))
        }
        super::subagent_spawn_selector::BatchRouting::MultiSubagent { plans } => {
            let batch = spawn_multi_subagent_invocations(inputs, deps, plans, payload, now).await?;
            let invocation_tasks = batch
                .invocations
                .into_iter()
                .map(|inv| inv.invocation_task)
                .collect();
            Ok((batch.parent_task, invocation_tasks))
        }
        super::subagent_spawn_selector::BatchRouting::AllTools
        | super::subagent_spawn_selector::BatchRouting::UnsupportedMixedBatch => deps
            .task_store
            .spawn_tool_children(
                task_id,
                &inputs.bootstrap.worker_id,
                &inputs.bootstrap.lease_id,
                specs,
                payload,
                now,
            )
            .await
            .context(tool_children_context),
    }
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
    continuation: &agent_sdk_core::ContinuationEnvelope,
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
    thread_id: &agent_sdk_core::ThreadId,
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
/// [`ContinuationEnvelope`]: agent_sdk_core::ContinuationEnvelope
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

    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
    };

    let total_usage = TokenUsage {
        input_tokens: current_state
            .total_usage
            .input_tokens
            .saturating_add(turn_usage.input_tokens),
        output_tokens: current_state
            .total_usage
            .output_tokens
            .saturating_add(turn_usage.output_tokens),
        cached_input_tokens: current_state
            .total_usage
            .cached_input_tokens
            .saturating_add(turn_usage.cached_input_tokens),
        cache_creation_input_tokens: current_state
            .total_usage
            .cache_creation_input_tokens
            .saturating_add(turn_usage.cache_creation_input_tokens),
    };

    let updated_state = AgentState {
        turn_count: current_state.turn_count + 1,
        total_usage: total_usage.clone(),
        ..current_state
    };

    let pending_tool_calls =
        extract_pending_tool_calls(response, &inputs.bootstrap.definition.tools);
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
/// LLM response, resolving `tier` and `display_name` from the
/// authoritative tool definitions on the [`AgentDefinition`].
///
/// [`PendingToolCallInfo`]: agent_sdk_core::PendingToolCallInfo
/// [`AgentDefinition`]: super::definition::AgentDefinition
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

/// Prior suspension state passed to the resume path so the LLM sees the
/// full conversation that led to the tool calls whose results are now
/// available.
struct ResumeContext<'a> {
    continuation: &'a AgentContinuation,
    suspended_messages: &'a [llm::Message],
    child_results: &'a [(String, ToolResult)],
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
/// [`AgentContinuation`]: agent_sdk_core::AgentContinuation
/// [`TaskState::ReadyToResume`]: crate::journal::task_state::TaskState::ReadyToResume
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
    let attempt = open_attempt(&inputs, definition, &resume_audit, deps.attempt_store, now)
        .await
        .context("open resume turn attempt")?;

    // 2. Buffer the suspended messages (user prompt + assistant with
    //    tool calls) and tool-result messages into the staged stores.
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

    let resumed_draft = build_resumed_draft_messages(&suspended_messages, &child_results);
    deps.message_store
        .set_draft(thread_id, resumed_draft, now)
        .await
        .context("refresh draft with completed child results before resume LLM call")?;

    // 2.5 Pre-call auto-compaction.
    //
    //     The resume path's staged history already includes the
    //     suspended messages + tool results buffered above, so
    //     compacting here folds the prior turn's tool transcript
    //     into the summary before the LLM sees it. No-op when
    //     `deps.compaction_config` is `None`.
    super::compaction::maybe_compact_staged_history(
        deps,
        &inputs.staged_stores.messages,
        thread_id,
        now,
    )
    .await
    .context("pre-call auto-compaction (resume path)")?;

    // 3. Build the chat request from staged history — no extra user
    //    prompt to append because everything is already buffered.
    let chat_request = build_resume_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
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
    let StreamedTurn {
        response,
        content_ids,
        attempt,
    } = call_llm_with_retry(LlmRetryParams {
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
    })
    .await?;
    let commit_now = OffsetDateTime::now_utc();

    let prior = ResumeContext {
        continuation: &continuation,
        suspended_messages: &suspended_messages,
        child_results: &child_results,
    };

    // 5. Branch: tool calls → re-suspend; text-only → commit.
    if response.has_tool_use() {
        return suspend_resumed_turn(
            inputs,
            &prior,
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
    let updated_state = AgentState {
        total_usage: TokenUsage {
            input_tokens: continuation
                .state
                .total_usage
                .input_tokens
                .saturating_add(response.usage.input_tokens),
            output_tokens: continuation
                .state
                .total_usage
                .output_tokens
                .saturating_add(response.usage.output_tokens),
            cached_input_tokens: continuation
                .state
                .total_usage
                .cached_input_tokens
                .saturating_add(response.usage.cached_input_tokens),
            cache_creation_input_tokens: continuation
                .state
                .total_usage
                .cache_creation_input_tokens
                .saturating_add(response.usage.cache_creation_input_tokens),
        },
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
const fn merged_turn_usage(prior: &TokenUsage, response: &llm::ChatResponse) -> TokenUsage {
    TokenUsage {
        input_tokens: prior
            .input_tokens
            .saturating_add(response.usage.input_tokens),
        output_tokens: prior
            .output_tokens
            .saturating_add(response.usage.output_tokens),
        cached_input_tokens: prior
            .cached_input_tokens
            .saturating_add(response.usage.cached_input_tokens),
        cache_creation_input_tokens: prior
            .cache_creation_input_tokens
            .saturating_add(response.usage.cache_creation_input_tokens),
    }
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

    ensure_turn_not_already_committed(
        deps.thread_store,
        thread_id,
        inputs.recovery_view.next_turn_number,
    )
    .await
    .context("resume idempotency check")?;

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

    let commit = commit_completed_turn(
        CompletedTurnCommit {
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params,
            messages: drained_messages,
            turn_usage,
            agent_state_snapshot,
            events: lifecycle_events,
            outbox_max_attempts: DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        },
        deps.thread_store,
        deps.message_store,
        deps.attempt_store,
        deps.checkpoint_store,
        deps.event_repo,
    )
    .await
    .context("commit resumed turn")?;

    let committed_events = commit.committed_events.clone();

    // Advance the root task to Completed.
    let (completed_task, _parent) = deps
        .task_store
        .complete_task(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            now,
        )
        .await
        .context("complete resumed root task")?;

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
    thread_id: &agent_sdk_core::ThreadId,
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

/// Build a chat request for the resume path. Unlike the fresh-turn
/// `build_chat_request`, this uses the staged history as-is (no
/// additional user prompt to append).
async fn build_resume_chat_request(
    definition: &AgentDefinition,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
    caller_metadata: Option<&serde_json::Value>,
) -> Result<ChatRequest> {
    let messages = staged_messages
        .get_history(thread_id)
        .await
        .context("get staged history for resume")?;

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
        session_id: None,
        cached_content: None,
        thinking,
        tool_choice: None,
        response_format: None,
    })
}

/// Re-suspend a resumed turn when the LLM responds with more tool
/// calls. Buffers the full conversation (original suspended messages +
/// tool results + new assistant response) into the new suspension's
/// `suspended_messages` so a subsequent resume can reconstruct the
/// complete history.
async fn suspend_resumed_turn(
    inputs: RootWorkerInputs,
    prior: &ResumeContext<'_>,
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
    // through this point: original user prompt + original assistant +
    // tool results + new assistant response.
    let mut new_suspended =
        build_resumed_draft_messages(prior.suspended_messages, prior.child_results);
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

    let new_turn_usage = TokenUsage {
        input_tokens: prior
            .turn_usage
            .input_tokens
            .saturating_add(response.usage.input_tokens),
        output_tokens: prior
            .turn_usage
            .output_tokens
            .saturating_add(response.usage.output_tokens),
        cached_input_tokens: prior
            .turn_usage
            .cached_input_tokens
            .saturating_add(response.usage.cached_input_tokens),
        cache_creation_input_tokens: prior
            .turn_usage
            .cache_creation_input_tokens
            .saturating_add(response.usage.cache_creation_input_tokens),
    };

    let new_total_usage = TokenUsage {
        input_tokens: prior
            .state
            .total_usage
            .input_tokens
            .saturating_add(response.usage.input_tokens),
        output_tokens: prior
            .state
            .total_usage
            .output_tokens
            .saturating_add(response.usage.output_tokens),
        cached_input_tokens: prior
            .state
            .total_usage
            .cached_input_tokens
            .saturating_add(response.usage.cached_input_tokens),
        cache_creation_input_tokens: prior
            .state
            .total_usage
            .cache_creation_input_tokens
            .saturating_add(response.usage.cache_creation_input_tokens),
    };

    let updated_state = AgentState {
        total_usage: new_total_usage.clone(),
        ..prior.state.clone()
    };

    let pending_tool_calls =
        extract_pending_tool_calls(response, &inputs.bootstrap.definition.tools);
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
