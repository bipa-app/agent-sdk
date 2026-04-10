//! Root turn execution: text-only commit and tool-boundary suspension.
//!
//! This module drives a single root-turn attempt from LLM call through
//! to either a completed-turn commit (text-only) or a tool-boundary
//! suspension that spawns child tasks (tool calls present).
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
use agent_sdk_core::llm::{self, ChatOutcome, ChatRequest};
use agent_sdk_core::{
    AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, TokenUsage, ToolTier,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::{MessageStore, StateStore};
use anyhow::{Context, Result, bail};
use time::OffsetDateTime;

use super::definition::{AgentDefinition, ThinkingPolicy};
use crate::journal::checkpoint_store::CheckpointStore;
use crate::journal::commit::{CommitOutcome, CompletedTurnCommit, commit_completed_turn};
use crate::journal::execution_context::RootWorkerInputs;
use crate::journal::message_store::MessageProjectionStore;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, ChildSpawnSpec};
use crate::journal::thread_store::ThreadStore;
use crate::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptOutcome,
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
    },
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
    let task_id = &inputs.bootstrap.task_id;

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
    )
    .await
    .context("build chat request")?;

    let response = call_llm(provider, chat_request, &attempt, deps.attempt_store, now).await?;

    // Capture a post-LLM timestamp so the turn attempt's duration_ms
    // reflects actual wall-clock latency instead of always being 0.
    let commit_now = OffsetDateTime::now_utc();

    // 3. Branch: tool calls → suspend; text-only → commit.
    if response.has_tool_use() {
        return suspend_at_tool_boundary(inputs, response, attempt, deps, now).await;
    }

    // ── Text-only path (Phase 4.3) ──────────────────────────────

    let response_text = response.first_text().unwrap_or("").to_owned();

    // 4. Buffer in staged stores.
    buffer_turn_messages(
        &inputs.staged_stores.messages,
        &inputs.staged_stores.state,
        thread_id,
        user_prompt,
        &response,
    )
    .await
    .context("buffer staged messages")?;

    // 4. Idempotency guard: re-read the thread from the durable store
    //    to detect if a prior worker already committed the turn we're
    //    targeting (e.g. our lease expired between commit and
    //    complete_task, and another worker re-acquired the task).
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

    // 5. Drain staged stores and commit.
    let close_params = build_close_params(&response, &attempt);
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
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

    let commit = commit_completed_turn(
        CompletedTurnCommit {
            thread_id: thread_id.clone(),
            task_id: task_id.clone(),
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params,
            messages: drained_messages,
            turn_usage,
            agent_state_snapshot,
            now: commit_now,
        },
        deps.thread_store,
        deps.message_store,
        deps.attempt_store,
        deps.checkpoint_store,
    )
    .await
    .context("commit completed turn")?;

    // 6. Advance the root task to Completed.
    let (completed_task, _parent) = deps
        .task_store
        .complete_task(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            commit_now,
        )
        .await
        .context("complete root task")?;

    Ok(RootTurnOutcome::Completed {
        commit: Box::new(commit),
        completed_task,
        response_text,
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

    Ok(ChatRequest {
        system: definition.system_prompt.clone(),
        messages,
        tools: if definition.tools.is_empty() {
            None
        } else {
            Some(definition.tools.clone())
        },
        max_tokens: definition.max_tokens,
        max_tokens_explicit: true,
        // Not yet wired — session/cache affinity requires provider-side
        // state that the server doesn't manage in Phase 4.3.
        session_id: None,
        cached_content: None,
        thinking,
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
    response: llm::ChatResponse,
    attempt: TurnAttempt,
    deps: &RootTurnDeps<'_>,
    now: OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let task_id = &inputs.bootstrap.task_id;

    // 1. Close the turn attempt — the LLM call itself succeeded.
    let close_params = build_close_params(&response, &attempt);
    deps.attempt_store
        .close_attempt(&attempt.id, close_params, now)
        .await
        .context("close attempt on tool suspension")?;

    // 2. Build the continuation envelope from current state + response.
    let continuation = build_continuation(&inputs, &response)
        .await
        .context("build continuation for tool suspension")?;

    // 3. One child task per tool call.
    let tool_call_count = response.tool_uses().count();
    let specs: Vec<ChildSpawnSpec> = (0..tool_call_count)
        .map(|_| ChildSpawnSpec::default())
        .collect();

    // 4. Atomically spawn children and park the parent.
    let (parent_task, child_tasks) = deps
        .task_store
        .spawn_tool_children(
            task_id,
            &inputs.bootstrap.worker_id,
            &inputs.bootstrap.lease_id,
            specs,
            continuation,
            now,
        )
        .await
        .context("spawn tool children")?;

    Ok(RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
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
