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
use agent_sdk_core::llm::{self, ChatOutcome, ChatRequest};
use agent_sdk_core::{
    AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, TokenUsage,
    ToolResult, ToolTier,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::{MessageStore, StateStore};
use anyhow::{Context, Result, bail, ensure};
use time::OffsetDateTime;

use super::definition::{AgentDefinition, ThinkingPolicy};
use crate::journal::checkpoint_store::CheckpointStore;
use crate::journal::commit::{CommitOutcome, CompletedTurnCommit, commit_completed_turn};
use crate::journal::committed_event::CommittedEvent;
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
    user_prompt: &str,
    provider: &dyn LlmProvider,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let definition = inputs.definition();
    let thread_id = &inputs.bootstrap.thread_id;

    // 1. Open turn attempt.
    let attempt = open_attempt(&inputs, definition, user_prompt, deps.attempt_store, now)
        .await
        .context("open turn attempt")?;

    // 2. Build, send LLM request, and resolve the outcome — closing
    //    the attempt on any non-success path.
    let chat_request = build_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
        user_prompt,
        &inputs.bootstrap.task.caller_metadata,
    )
    .await
    .context("build chat request")?;

    let response = call_llm(provider, chat_request, &attempt, deps.attempt_store, now).await?;

    // Capture a post-LLM timestamp so the turn attempt's duration_ms
    // reflects actual wall-clock latency instead of always being 0.
    let commit_now = OffsetDateTime::now_utc();

    // 3. Branch: tool calls → suspend; text-only → commit.
    //
    // The Start event is committed inside each branch, AFTER its
    // idempotency guard fires, so a stale-lease worker that
    // re-acquired the task cannot produce a duplicate Start.
    if response.has_tool_use() {
        return suspend_at_tool_boundary(inputs, user_prompt, response, attempt, deps, commit_now)
            .await;
    }

    // ── Text-only path (Phase 4.3) ──────────────────────────────
    commit_text_only_turn(inputs, user_prompt, response, attempt, deps, commit_now).await
}

/// Complete and commit a text-only turn (no tool calls).
async fn commit_text_only_turn(
    inputs: RootWorkerInputs,
    user_prompt: &str,
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let thread_id = &inputs.bootstrap.thread_id;
    let task_id = &inputs.bootstrap.task_id;
    let response_text = response.first_text().unwrap_or("").to_owned();

    buffer_turn_messages(
        &inputs.staged_stores.messages,
        &inputs.staged_stores.state,
        thread_id,
        user_prompt,
        &response,
    )
    .await
    .context("buffer staged messages")?;

    // Idempotency guard.
    let expected_turn = inputs.recovery_view.next_turn_number;
    let current_thread = deps
        .thread_store
        .get(thread_id)
        .await
        .context("re-read thread for idempotency check")?
        .context("thread disappeared during turn execution")?;

    if current_thread.committed_turns >= expected_turn {
        bail!(
            "turn {expected_turn} was already committed on thread {} \
             (committed_turns={}); skipping duplicate commit",
            thread_id,
            current_thread.committed_turns,
        );
    }

    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);
    let close_params = build_close_params(&response, &attempt);
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cached_input_tokens: response.usage.cached_input_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
    };

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

    // Build lifecycle events with Start prepended. Start is included
    // in the events vec passed to commit_completed_turn so it is
    // committed atomically as step 5 — after all CAS-guarded state
    // projections succeed. This prevents a stale-lease worker from
    // producing orphaned Start events.
    let duration = (now - attempt.opened_at).unsigned_abs();
    let mut lifecycle_events = vec![AgentEvent::start(thread_id.clone(), turn_number)];
    lifecycle_events.extend(build_turn_complete_events(
        &response,
        thread_id,
        turn_number,
        &turn_usage,
        &drained_state.total_usage,
        duration,
    ));

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

    let committed_events = commit.committed_events.clone();

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
        })
        .await
        .context("open_attempt via store")
}

async fn build_chat_request(
    definition: &AgentDefinition,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
    user_prompt: &str,
    caller_metadata: &serde_json::Value,
) -> Result<ChatRequest> {
    // Get existing message history from staged store.
    let mut messages = staged_messages
        .get_history(thread_id)
        .await
        .context("get staged history")?;

    // Append the new user prompt.
    messages.push(llm::Message::user(user_prompt));

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
    // set, enabling per-turn filtering based on caller identity.
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
    })
}

/// Call the LLM and resolve the outcome, closing the turn attempt on
/// any non-success path before returning an error.
///
/// On success the response is returned as-is — it may contain tool-use
/// blocks. The caller is responsible for branching on text-only vs
/// tool-call paths.
async fn call_llm(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    attempt: &TurnAttempt,
    attempt_store: &dyn TurnAttemptStore,
    now: OffsetDateTime,
) -> Result<llm::ChatResponse> {
    let outcome = provider.chat(request).await.context("LLM provider call")?;

    let response = match outcome {
        ChatOutcome::Success(r) => r,
        ChatOutcome::RateLimited => {
            close_attempt_with(attempt, TurnAttemptOutcome::RateLimited, attempt_store, now).await;
            bail!("LLM rate limited");
        }
        ChatOutcome::InvalidRequest(msg) => {
            close_attempt_with(
                attempt,
                TurnAttemptOutcome::InvalidRequest,
                attempt_store,
                now,
            )
            .await;
            bail!("LLM invalid request: {msg}");
        }
        ChatOutcome::ServerError(msg) => {
            close_attempt_with(attempt, TurnAttemptOutcome::ServerError, attempt_store, now).await;
            bail!("LLM server error: {msg}");
        }
    };

    Ok(response)
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
    user_prompt: &str,
    response: &llm::ChatResponse,
) -> Result<()> {
    // Append user message.
    staged_messages
        .append(thread_id, llm::Message::user(user_prompt))
        .await
        .context("append user message")?;

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

fn build_close_params(response: &llm::ChatResponse, _attempt: &TurnAttempt) -> CloseAttemptParams {
    CloseAttemptParams {
        response_blob: serde_json::json!({
            "id": response.id,
            "model": response.model,
        }),
        response_id: Some(response.id.clone()),
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
fn build_content_events(response: &llm::ChatResponse) -> Vec<AgentEvent> {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
                Some(AgentEvent::thinking(&response.id, thinking))
            }
            llm::ContentBlock::Text { text } if !text.is_empty() => {
                Some(AgentEvent::text(&response.id, text))
            }
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
) -> Vec<AgentEvent> {
    let mut events = build_content_events(response);
    if response.stop_reason == Some(llm::StopReason::Refusal) {
        events.push(AgentEvent::refusal(
            response.id.clone(),
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
    user_prompt: &str,
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let task_id = &inputs.bootstrap.task_id;
    let thread_id = &inputs.bootstrap.thread_id;

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

    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);

    // 1. Close the turn attempt — the LLM call itself succeeded.
    //    AlreadyClosed is non-fatal: a prior recovery sweep
    //    (best_effort_close_open_attempts) may have closed the attempt
    //    after our lease expired but before we reached this point. The
    //    work is done — just continue with the suspension.
    let close_params = build_close_params(&response, &attempt);
    match deps
        .attempt_store
        .close_attempt(&attempt.id, close_params, now)
        .await
    {
        Ok(_) => {}
        Err(e)
            if e.downcast_ref::<TurnAttemptSchemaError>()
                == Some(&TurnAttemptSchemaError::AlreadyClosed) =>
        {
            // Recovery sweep already closed this attempt — safe to proceed.
        }
        Err(e) => return Err(e.context("close attempt on tool suspension")),
    }

    // 2. Build the continuation envelope from current state + response.
    let continuation = build_continuation(&inputs, &response)
        .await
        .context("build continuation for tool suspension")?;

    // 3. Capture the suspended messages (user prompt + full assistant
    //    response including tool-use blocks) so the resume path can
    //    reconstruct the conversation from durable state alone.
    let suspended_messages = vec![
        llm::Message::user(user_prompt),
        build_full_assistant_message(&response),
    ];

    // 4. One child task per tool call.
    // 3. One child task per tool call, inheriting the configured retry budget.
    let tool_call_count = response.tool_uses().count();
    let child_max_attempts = inputs.bootstrap.definition.policy.max_attempts;
    let specs: Vec<ChildSpawnSpec> = (0..tool_call_count)
        .map(|_| ChildSpawnSpec::new(child_max_attempts))
        .collect();

    // Build content events (Thinking) from the tool-call response so
    // replay observers see the model's reasoning before tool dispatch.
    let content_events = build_content_events(&response);

    // Build ToolCallStart events before continuation is moved.
    let tool_call_events: Vec<AgentEvent> = continuation
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
        .collect();

    // 5. Atomically spawn children and park the parent.
    let (parent_task, child_tasks) = deps
        .task_store
        .spawn_tool_children(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            specs,
            SuspensionPayload {
                continuation,
                suspended_messages,
            },
            now,
        )
        .await
        .context("spawn tool children")?;

    // Commit Start + content events (Thinking) + ToolCallStart events
    // in a single batch AFTER spawn_tool_children. Since
    // spawn_tool_children is CAS-guarded (only the lease-holder can
    // succeed), only the winning worker writes events — preventing
    // orphaned Start events from stale-lease workers.
    let mut suspension_events = vec![AgentEvent::start(thread_id.clone(), turn_number)];
    suspension_events.extend(content_events);
    suspension_events.extend(tool_call_events);
    let committed_events = deps
        .event_repo
        .commit_event_batch(&inputs.bootstrap.thread_id, suspension_events, now)
        .await
        .context("commit suspension events")?;

    Ok(RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        committed_events,
    })
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
        response_id: Some(response.id.clone()),
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

    // 1. Open a new turn attempt for the resume LLM call.
    let attempt = open_attempt(&inputs, definition, "<resume>", deps.attempt_store, now)
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

    // 3. Build the chat request from staged history — no extra user
    //    prompt to append because everything is already buffered.
    let chat_request = build_resume_chat_request(
        definition,
        &inputs.staged_stores.messages,
        thread_id,
        &inputs.bootstrap.task.caller_metadata,
    )
    .await
    .context("build resume chat request")?;

    // 4. Call the LLM.
    let response = call_llm(provider, chat_request, &attempt, deps.attempt_store, now).await?;
    let commit_now = OffsetDateTime::now_utc();

    let prior = ResumeContext {
        continuation: &continuation,
        suspended_messages: &suspended_messages,
        child_results: &child_results,
    };

    // 5. Branch: tool calls → re-suspend; text-only → commit.
    if response.has_tool_use() {
        return suspend_resumed_turn(inputs, &prior, response, attempt, deps, commit_now).await;
    }

    commit_resumed_turn(inputs, &continuation, &response, &attempt, deps, commit_now).await
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
) -> Result<RootTurnOutcome> {
    let thread_id = &inputs.bootstrap.thread_id;
    let task_id = &inputs.bootstrap.task_id;

    let response_text = response.first_text().unwrap_or("").to_owned();

    buffer_resumed_assistant(&inputs, continuation, response).await?;

    // Idempotency guard (same as text-only path).
    let expected_turn = inputs.recovery_view.next_turn_number;
    let current_thread = deps
        .thread_store
        .get(thread_id)
        .await
        .context("re-read thread for resume idempotency check")?
        .context("thread disappeared during resume")?;

    if current_thread.committed_turns >= expected_turn {
        bail!(
            "turn {expected_turn} was already committed on thread {} \
             (committed_turns={}); skipping duplicate resume commit",
            thread_id,
            current_thread.committed_turns,
        );
    }

    // Drain staged stores and commit.
    let close_params = build_close_params(response, attempt);
    let turn_usage = TokenUsage {
        input_tokens: continuation
            .turn_usage
            .input_tokens
            .saturating_add(response.usage.input_tokens),
        output_tokens: continuation
            .turn_usage
            .output_tokens
            .saturating_add(response.usage.output_tokens),
        cached_input_tokens: continuation
            .turn_usage
            .cached_input_tokens
            .saturating_add(response.usage.cached_input_tokens),
        cache_creation_input_tokens: continuation
            .turn_usage
            .cache_creation_input_tokens
            .saturating_add(response.usage.cache_creation_input_tokens),
    };

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

    let turn_number = usize::try_from(inputs.recovery_view.next_turn_number).unwrap_or(0);
    let duration = (now - attempt.opened_at).unsigned_abs();
    let lifecycle_events = build_turn_complete_events(
        response,
        thread_id,
        turn_number,
        &turn_usage,
        &drained_state.total_usage,
        duration,
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

/// Build a chat request for the resume path. Unlike the fresh-turn
/// `build_chat_request`, this uses the staged history as-is (no
/// additional user prompt to append).
async fn build_resume_chat_request(
    definition: &AgentDefinition,
    staged_messages: &crate::journal::staged::StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
    caller_metadata: &serde_json::Value,
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
) -> Result<RootTurnOutcome> {
    let task_id = &inputs.bootstrap.task_id;

    // Close the turn attempt — the LLM call succeeded.
    //
    // AlreadyClosed is non-fatal: a prior recovery sweep
    // (best_effort_close_open_attempts) may have closed the attempt
    // after our lease expired but before we reached this point. The
    // work is done — just continue with the suspension.
    let close_params = build_close_params(&response, &attempt);
    match deps
        .attempt_store
        .close_attempt(&attempt.id, close_params, now)
        .await
    {
        Ok(_) => {}
        Err(e)
            if e.downcast_ref::<TurnAttemptSchemaError>()
                == Some(&TurnAttemptSchemaError::AlreadyClosed) =>
        {
            // Recovery sweep already closed this attempt — safe to proceed.
        }
        Err(e) => return Err(e.context("close attempt on resumed tool suspension")),
    }

    // Build a new continuation that accumulates usage from the prior
    // continuation plus the new response.
    let new_continuation = build_resume_continuation(&inputs, prior.continuation, &response)
        .context("build resume continuation")?;

    // Build suspended messages that capture the FULL conversation
    // through this point: original user prompt + original assistant +
    // tool results + new assistant response.
    let mut new_suspended = Vec::with_capacity(prior.suspended_messages.len() + 2);
    // Original user prompt + original assistant response (tool calls).
    new_suspended.extend_from_slice(prior.suspended_messages);
    // Tool results from child tasks.
    new_suspended.push(build_tool_results_message(prior.child_results));
    // New assistant response (with new tool-use blocks).
    new_suspended.push(build_full_assistant_message(&response));

    // One child task per new tool call, inheriting the configured retry budget
    // (same as suspend_at_tool_boundary — ChildSpawnSpec::default() would use
    // DEFAULT_MAX_ATTEMPTS=1, not the policy's budget).
    let tool_call_count = response.tool_uses().count();
    let child_max_attempts = inputs.bootstrap.definition.policy.max_attempts;
    let specs: Vec<ChildSpawnSpec> = (0..tool_call_count)
        .map(|_| ChildSpawnSpec::new(child_max_attempts))
        .collect();

    // Build content events (Thinking) from the resume response.
    let content_events = build_content_events(&response);

    // Build ToolCallStart events before continuation is moved.
    let tool_call_events: Vec<AgentEvent> = new_continuation
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
        .collect();

    let (parent_task, child_tasks) = deps
        .task_store
        .spawn_tool_children(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            specs,
            SuspensionPayload {
                continuation: new_continuation,
                suspended_messages: new_suspended,
            },
            now,
        )
        .await
        .context("re-spawn tool children on resume")?;

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
        response_id: Some(response.id.clone()),
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

    // 3. Delegate to the existing resume path.
    resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        provider,
        deps,
        now,
    )
    .await
}
