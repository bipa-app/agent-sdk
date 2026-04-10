//! Text-only root turn execution and completed-turn commit.
//!
//! This is the Phase 4.3 tracer bullet: the first end-to-end root worker
//! path that acquires a task, runs one model-side turn (no tool calls),
//! and atomically commits the result.
//!
//! # Execution flow
//!
//! 1. **Open turn attempt** — audit record via [`TurnAttemptStore`].
//! 2. **Build chat request** — system prompt + staged messages + user
//!    prompt from [`AgentDefinition`] and [`RootWorkerInputs`].
//! 3. **Call LLM** — `LlmProvider::chat()` directly (no full SDK agent
//!    loop needed for a single text-only turn).
//! 4. **Buffer response** — append user + assistant messages to staged
//!    stores and update agent state.
//! 5. **Commit** — drain staged stores → [`commit_completed_turn`].
//! 6. **Complete task** — [`AgentTaskStore::complete_task`].
//!
//! # Scope
//!
//! This module handles **text-only** turns only. Tool-call suspension,
//! child-task creation, and resume flows are out of scope (Phase 4.4+).
//!
//! # Guarantees
//!
//! - No durable writes before the commit path succeeds.
//! - Turn attempt is closed atomically as part of the commit.
//! - Task advances to `Completed` only after commit succeeds.

use agent_sdk_core::audit::AuditProvenance;
use agent_sdk_core::llm::{self, ChatOutcome, ChatRequest};
use agent_sdk_core::{AgentState, TokenUsage};
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
#[derive(Debug)]
pub struct RootTurnOutcome {
    /// The commit outcome (thread, checkpoint, closed attempt).
    pub commit: CommitOutcome,
    /// The completed task after advancing.
    pub completed_task: crate::journal::task::AgentTask,
    /// The assistant's text response.
    pub response_text: String,
}

// ─────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────

/// Execute a text-only root turn end to end.
///
/// This is the Phase 4.3 happy-path tracer bullet. It:
///
/// 1. Opens a turn attempt for audit.
/// 2. Calls the LLM with the staged message history + user prompt.
/// 3. Buffers the response in staged stores.
/// 4. Atomically commits via [`commit_completed_turn`].
/// 5. Advances the root task to `Completed`.
///
/// # Errors
///
/// - LLM returns a non-success outcome (rate limit, server error, etc.)
/// - LLM response contains tool calls (not supported in this phase)
/// - Commit path fails
/// - Task completion fails
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

    let response_text = response.first_text().unwrap_or("").to_owned();

    // 3. Buffer in staged stores.
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

    Ok(RootTurnOutcome {
        commit,
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
        // Text-only path — never advertise tools to the LLM (Phase 4.4+).
        tools: None,
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

    // Validate text-only (no tool calls in Phase 4.3).
    let tool_count = response.tool_uses().count();
    if tool_count > 0 {
        close_attempt_with(attempt, TurnAttemptOutcome::Cancelled, attempt_store, now).await;
        bail!(
            "Phase 4.3 text-only path received {tool_count} tool call(s); \
             tool-call suspension is not yet implemented (Phase 4.4+)"
        );
    }

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
