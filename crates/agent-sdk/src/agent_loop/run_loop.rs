use super::helpers::{pending_tool_index, send_event, turns_to_u32};
use super::tool_execution::{append_tool_results, execute_confirmed_tool, execute_tool_call};
use super::turn::execute_turn;
use super::types::{
    ConfirmedToolExecutionContext, ConvertTurnResultParams, ExecuteTurnParameters,
    InitializedState, InternalTurnResult, PersistentDoneParams, ResumeData,
    ResumeProcessingParameters, ResumeProcessingResult, RunLoopParameters, RunLoopResumeParams,
    RunLoopTurnResultParams, RunLoopTurnsParams, SingleTurnResumeParams, ToolCallExecutionContext,
    ToolExecutionOutcome, TurnContext, TurnParameters,
};

use crate::types::TurnOptions;

use crate::authority::EventAuthority;
use crate::events::AgentEvent;
use crate::hooks::AgentHooks;
use crate::llm::{Content, ContentBlock, LlmProvider, Message, Role};
use crate::stores::{EventStore, MessageStore, StateStore, ToolExecutionStore};
use crate::types::{
    AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState, ContinuationEnvelope,
    ThreadId, TokenUsage, TurnOutcome,
};
use log::warn;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

enum RunLoopTurnAction {
    Continue,
    FinishRun,
    Return(AgentRunState),
}

/// Initialize agent state from the given input.
///
/// Handles the three input variants:
/// - `Text`/`Message`: Creates/loads state, appends user message
/// - `Resume`: Restores from continuation state
/// - `Continue`: Loads existing state to continue execution
pub(super) async fn initialize_from_input<M, S>(
    input: AgentInput,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
) -> Result<InitializedState, AgentError>
where
    M: MessageStore,
    S: StateStore,
{
    match input {
        AgentInput::Text(user_message) => {
            recover_orphaned_tool_use(thread_id, message_store).await?;
            let msg = Message::user(&user_message);
            initialize_from_message(msg, thread_id, message_store, state_store).await
        }
        AgentInput::Message(blocks) => {
            let msg = Message::user_with_content(blocks);
            initialize_from_message(msg, thread_id, message_store, state_store).await
        }
        AgentInput::Resume {
            continuation: envelope,
            tool_call_id,
            confirmed,
            rejection_reason,
        } => {
            // Validate continuation version
            let continuation = Box::new(
                envelope
                    .unwrap_validated()
                    .map_err(|msg| AgentError::new(msg, false))?,
            );

            // Validate thread_id matches
            if continuation.thread_id != *thread_id {
                return Err(AgentError::new(
                    format!(
                        "Thread ID mismatch: continuation is for {}, but resuming on {}",
                        continuation.thread_id, thread_id
                    ),
                    false,
                ));
            }

            Ok(InitializedState {
                turn: continuation.turn,
                total_usage: continuation.total_usage.clone(),
                state: continuation.state.clone(),
                resume_data: Some(ResumeData {
                    continuation,
                    tool_call_id,
                    confirmed,
                    rejection_reason,
                }),
            })
        }
        AgentInput::SubmitToolResults {
            continuation: envelope,
            results,
        } => {
            let continuation = Box::new(
                envelope
                    .unwrap_validated()
                    .map_err(|msg| AgentError::new(msg, false))?,
            );
            initialize_from_tool_results(
                continuation,
                results,
                thread_id,
                message_store,
                execution_store,
                audit_sink,
                provenance,
            )
            .await
        }
        AgentInput::Continue => {
            let state = match state_store.load(thread_id).await {
                Ok(Some(s)) => s,
                Ok(None) => {
                    return Err(AgentError::new(
                        "Cannot continue: no state found for thread",
                        false,
                    ));
                }
                Err(e) => {
                    return Err(AgentError::new(format!("Failed to load state: {e}"), false));
                }
            };

            recover_orphaned_tool_use(thread_id, message_store).await?;

            Ok(InitializedState {
                turn: state.turn_count,
                total_usage: state.total_usage.clone(),
                state,
                resume_data: None,
            })
        }
    }
}

/// Shared initialization for `Text` and `Message` inputs: load or create state,
/// append the user message, and return a fresh turn-zero `InitializedState`.
async fn initialize_from_message<M, S>(
    user_msg: Message,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
) -> Result<InitializedState, AgentError>
where
    M: MessageStore,
    S: StateStore,
{
    let state = match state_store.load(thread_id).await {
        Ok(Some(s)) => s,
        Ok(None) => AgentState::new(thread_id.clone()),
        Err(e) => {
            return Err(AgentError::new(format!("Failed to load state: {e}"), false));
        }
    };

    if let Err(e) = message_store.append(thread_id, user_msg).await {
        return Err(AgentError::new(
            format!("Failed to append message: {e}"),
            false,
        ));
    }

    Ok(InitializedState {
        turn: 0,
        total_usage: TokenUsage::default(),
        state,
        resume_data: None,
    })
}

/// Handle `AgentInput::SubmitToolResults`: validate, detect replay, append
/// results, return state.
///
/// This is the **external tool runtime** audit point. For each submitted
/// [`ExternalToolResult`] the SDK emits one of:
///
/// - [`ToolAuditOutcome::Replayed`] — the execution store already has a
///   completed record for this `tool_call_id`, so the SDK served the
///   previously recorded result instead of re-appending.
/// - [`ToolAuditOutcome::Completed`] — the first-time submission; the
///   SDK records the execution and appends the result.
/// - [`ToolAuditOutcome::PersistenceFailed`] — the durable append failed
///   after the `Completed` record was already emitted.
///
/// Replay detection uses the same execution store the inline path uses
/// for idempotency. When no execution store is wired the SDK always
/// treats submissions as first-time `Completed`.
async fn initialize_from_tool_results<M: MessageStore>(
    continuation: Box<AgentContinuation>,
    results: Vec<crate::types::ExternalToolResult>,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
) -> Result<InitializedState, AgentError> {
    use agent_sdk_core::audit::ToolAuditOutcome;

    if continuation.thread_id != *thread_id {
        return Err(AgentError::new(
            format!(
                "Thread ID mismatch: continuation is for {}, but resuming on {}",
                continuation.thread_id, thread_id
            ),
            false,
        ));
    }

    validate_external_tool_results(&continuation, &results)?;

    let tool_results: Vec<(String, crate::types::ToolResult)> = results
        .into_iter()
        .map(|r| (r.tool_call_id, r.result))
        .collect();

    // Partition submissions into first-time and replays. A replay is a
    // submission whose tool_call_id already has a completed entry in the
    // execution store. Replays are audited as `Replayed` and do NOT get
    // re-appended to the message store; first-time submissions are
    // audited as `Completed` and persisted normally.
    let mut fresh_results: Vec<(String, crate::types::ToolResult)> =
        Vec::with_capacity(tool_results.len());
    for (tool_call_id, result) in tool_results {
        let already_completed = match execution_store {
            Some(store) => store
                .get_execution(&tool_call_id)
                .await
                .ok()
                .flatten()
                .is_some_and(|execution| execution.is_completed()),
            None => false,
        };
        if already_completed {
            emit_external_tool_audit(
                audit_sink,
                provenance,
                &continuation,
                &tool_call_id,
                ToolAuditOutcome::Replayed {
                    result: result.clone(),
                },
            )
            .await;
        } else {
            emit_external_tool_audit(
                audit_sink,
                provenance,
                &continuation,
                &tool_call_id,
                ToolAuditOutcome::Completed {
                    result: result.clone(),
                },
            )
            .await;
            record_external_tool_execution(
                execution_store,
                &continuation,
                thread_id,
                &tool_call_id,
                &result,
            )
            .await;
            fresh_results.push((tool_call_id, result));
        }
    }

    if fresh_results.is_empty() {
        // All submissions were replays — nothing new to persist. The
        // audit records already captured the outcome.
        return Ok(InitializedState {
            turn: continuation.turn,
            total_usage: continuation.total_usage.clone(),
            state: continuation.state.clone(),
            resume_data: None,
        });
    }

    if let Err(error) = append_tool_results(&fresh_results, thread_id, message_store).await {
        // Persistence of the fresh external tool batch failed — emit one
        // PersistenceFailed record per fresh result so consumers see
        // both the intended outcome and the durability gap.
        for (tool_call_id, result) in &fresh_results {
            emit_external_tool_audit(
                audit_sink,
                provenance,
                &continuation,
                tool_call_id,
                ToolAuditOutcome::PersistenceFailed {
                    result: Some(result.clone()),
                    error: error.message.clone(),
                },
            )
            .await;
        }
        return Err(error);
    }

    Ok(InitializedState {
        turn: continuation.turn,
        total_usage: continuation.total_usage.clone(),
        state: continuation.state.clone(),
        resume_data: None,
    })
}

/// Record a completed external-tool execution in the execution store.
///
/// This is the external-runtime analogue of the inline
/// [`execute_with_idempotency`](super::idempotency::execute_with_idempotency)
/// path: it writes the same `ToolExecution` row, only with `started_at`
/// and `completed_at` both set to "now" because the SDK never observed
/// the tool running. A subsequent submission of the same `tool_call_id`
/// will be detected as a replay.
async fn record_external_tool_execution(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    continuation: &AgentContinuation,
    thread_id: &ThreadId,
    tool_call_id: &str,
    result: &crate::types::ToolResult,
) {
    let Some(store) = execution_store else {
        return;
    };
    let pending = continuation
        .pending_tool_calls
        .iter()
        .find(|p| p.id == tool_call_id);
    let (tool_name, display_name, input) = pending.map_or_else(
        || (String::new(), String::new(), serde_json::Value::Null),
        |p| (p.name.clone(), p.display_name.clone(), p.input.clone()),
    );
    let started_at = time::OffsetDateTime::now_utc();
    let mut execution = crate::types::ToolExecution::new_in_flight(
        tool_call_id,
        thread_id.clone(),
        tool_name,
        display_name,
        input,
        started_at,
    );
    execution.complete(result.clone());
    if let Err(e) = store.record_execution(execution).await {
        warn!("Failed to record external tool execution (tool_call_id={tool_call_id}, error={e})",);
    }
}

/// Emit one audit record for a submitted external tool result.
///
/// Looks the tool metadata up on the continuation's pending list; if the
/// caller submitted a result whose `tool_call_id` is not in the pending
/// list (which normally can't happen because `validate_external_tool_results`
/// rejects that case first) the record falls back to empty identity
/// fields so the audit log still gets a row.
async fn emit_external_tool_audit(
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
    continuation: &AgentContinuation,
    tool_call_id: &str,
    outcome: agent_sdk_core::audit::ToolAuditOutcome,
) {
    use crate::types::ToolTier;
    use agent_sdk_core::audit::ToolAuditRecord;

    let pending = continuation
        .pending_tool_calls
        .iter()
        .find(|p| p.id == tool_call_id);
    let (tool_name, display_name, requested_input, effective_input) = pending.map_or_else(
        || {
            (
                String::new(),
                String::new(),
                serde_json::Value::Null,
                serde_json::Value::Null,
            )
        },
        |p| {
            (
                p.name.clone(),
                p.display_name.clone(),
                p.input.clone(),
                p.effective_input.clone(),
            )
        },
    );
    let record = ToolAuditRecord::new(
        tool_call_id.to_string(),
        tool_name,
        display_name,
        ToolTier::Observe,
        requested_input,
        effective_input,
        continuation.turn,
        provenance.clone(),
        outcome,
    );
    audit_sink.record(record).await;
}

fn validate_resume_continuation(
    cont: &AgentContinuation,
    tool_call_id: &str,
) -> Result<(), AgentError> {
    if cont.awaiting_index >= cont.pending_tool_calls.len() {
        return Err(AgentError::new(
            format!(
                "Invalid continuation: awaiting_index {} out of bounds ({})",
                cont.awaiting_index,
                cont.pending_tool_calls.len()
            ),
            false,
        ));
    }
    let awaiting_tool = &cont.pending_tool_calls[cont.awaiting_index];
    if awaiting_tool.id != tool_call_id {
        return Err(AgentError::new(
            format!(
                "Tool call ID mismatch: expected {}, got {}",
                awaiting_tool.id, tool_call_id
            ),
            false,
        ));
    }
    Ok(())
}

/// Validate that the caller provided exactly one result per pending tool call.
fn validate_external_tool_results(
    cont: &AgentContinuation,
    results: &[crate::types::ExternalToolResult],
) -> Result<(), AgentError> {
    if cont.pending_tool_calls.is_empty() {
        return Err(AgentError::new(
            "Invalid continuation: no pending tool calls to resolve",
            false,
        ));
    }

    // Check for missing results.
    for pending in &cont.pending_tool_calls {
        if !results.iter().any(|r| r.tool_call_id == pending.id) {
            return Err(AgentError::new(
                format!(
                    "Missing result for tool call '{}' (tool '{}')",
                    pending.id, pending.name,
                ),
                false,
            ));
        }
    }

    // Check for unknown or duplicate tool call IDs.
    let mut seen = HashSet::with_capacity(results.len());
    for result in results {
        if !cont
            .pending_tool_calls
            .iter()
            .any(|p| p.id == result.tool_call_id)
        {
            return Err(AgentError::new(
                format!(
                    "Unknown tool call ID '{}' — not in the pending tool calls",
                    result.tool_call_id,
                ),
                false,
            ));
        }
        if !seen.insert(&result.tool_call_id) {
            return Err(AgentError::new(
                format!(
                    "Duplicate result for tool call ID '{}'",
                    result.tool_call_id,
                ),
                false,
            ));
        }
    }

    Ok(())
}

pub(super) async fn process_resume<Ctx, H, M>(
    ResumeProcessingParameters {
        resume_data,
        turn,
        total_usage,
        state,
        thread_id,
        tool_context,
        tools,
        hooks,
        event_store,
        authority,
        message_store,
        execution_store,
        audit_sink,
        provenance,
    }: ResumeProcessingParameters<'_, Ctx, H, M>,
) -> Result<ResumeProcessingResult, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
{
    let ResumeData {
        continuation: cont,
        tool_call_id,
        confirmed,
        rejection_reason,
    } = resume_data;
    validate_resume_continuation(&cont, &tool_call_id)?;
    let awaiting_tool = &cont.pending_tool_calls[cont.awaiting_index];

    let mut tool_results = cont.completed_results.clone();
    let rejection =
        (!confirmed).then(|| rejection_reason.unwrap_or_else(|| "User rejected".to_string()));
    let confirmed_ctx = ConfirmedToolExecutionContext {
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        turn,
        authority,
        execution_store,
        audit_sink,
        provenance,
    };
    let result = execute_confirmed_tool(awaiting_tool, rejection, &confirmed_ctx).await?;
    tool_results.push((awaiting_tool.id.clone(), result));

    let execution_ctx = ToolCallExecutionContext {
        tool_context,
        thread_id,
        tools,
        hooks,
        event_store,
        turn,
        authority,
        execution_store,
        audit_sink,
        provenance,
    };
    for pending in cont.pending_tool_calls.iter().skip(cont.awaiting_index + 1) {
        match execute_tool_call(pending, &execution_ctx).await {
            ToolExecutionOutcome::Completed { tool_id, result } => {
                tool_results.push((tool_id, result));
            }
            ToolExecutionOutcome::RequiresConfirmation {
                tool_id,
                tool_name,
                display_name,
                input,
                description,
                listen_context,
            } => {
                let pending_idx = pending_tool_index(&cont.pending_tool_calls, &tool_id)?;
                let mut pending_tool_calls = cont.pending_tool_calls.clone();
                if let Some(context) = listen_context {
                    pending_tool_calls[pending_idx].listen_context = Some(context);
                }

                return Ok(ResumeProcessingResult::AwaitingConfirmation {
                    tool_call_id: tool_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                    continuation: Box::new(AgentContinuation {
                        thread_id: thread_id.clone(),
                        turn,
                        total_usage: total_usage.clone(),
                        turn_usage: cont.turn_usage.clone(),
                        pending_tool_calls,
                        awaiting_index: pending_idx,
                        completed_results: tool_results,
                        state: state.clone(),
                    }),
                });
            }
            ToolExecutionOutcome::Error(error) => return Err(error),
        }
    }

    append_tool_results(&tool_results, thread_id, message_store).await?;
    send_event(
        event_store,
        thread_id,
        turn,
        hooks,
        authority,
        AgentEvent::TurnComplete {
            turn,
            usage: cont.turn_usage.clone(),
        },
    )
    .await?;

    Ok(ResumeProcessingResult::Completed {
        turn_usage: cont.turn_usage.clone(),
    })
}

pub(super) async fn handle_run_loop_resume<Ctx, H, M>(
    RunLoopResumeParams {
        resume_data,
        turn,
        total_usage,
        state,
        thread_id,
        tool_context,
        tools,
        hooks,
        event_store,
        authority,
        message_store,
        execution_store,
        audit_sink,
        provenance,
    }: RunLoopResumeParams<'_, Ctx, H, M>,
) -> Result<Option<AgentRunState>, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
{
    match process_resume(ResumeProcessingParameters {
        resume_data,
        turn,
        total_usage,
        state,
        thread_id,
        tool_context,
        tools,
        hooks,
        event_store,
        authority,
        message_store,
        execution_store,
        audit_sink,
        provenance,
    })
    .await?
    {
        ResumeProcessingResult::Completed { .. } => Ok(None),
        ResumeProcessingResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        } => Ok(Some(AgentRunState::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
        })),
    }
}

async fn finish_turn_or_error(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
) -> Result<(), AgentError> {
    event_store
        .finish_turn(thread_id, turn)
        .await
        .map_err(|error| {
            AgentError::new(format!("Failed to finish turn event store: {error}"), false)
        })
}

async fn initialize_run_loop_state<M, S>(
    input: AgentInput,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
) -> Result<InitializedState, AgentRunState>
where
    M: MessageStore,
    S: StateStore,
{
    initialize_from_input(
        input,
        thread_id,
        message_store,
        state_store,
        execution_store,
        audit_sink,
        provenance,
    )
    .await
    .map_err(AgentRunState::Error)
}

async fn handle_run_loop_resume_state<Ctx, H, M>(
    params: RunLoopResumeParams<'_, Ctx, H, M>,
) -> Option<AgentRunState>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
{
    let turn = params.turn;
    let thread_id = params.thread_id;
    let event_store = params.event_store;
    let hooks = params.hooks;
    let authority = params.authority;

    match handle_run_loop_resume(params).await {
        Ok(Some(outcome)) => Some(outcome),
        Ok(None) => {
            if let Err(store_error) = finish_turn_or_error(event_store, thread_id, turn).await {
                return Some(AgentRunState::Error(store_error));
            }
            None
        }
        Err(error) => {
            if let Err(store_error) = send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                authority,
                AgentEvent::error(&error.message, error.recoverable),
            )
            .await
            {
                return Some(AgentRunState::Error(store_error));
            }
            if let Err(store_error) = finish_turn_or_error(event_store, thread_id, turn).await {
                return Some(AgentRunState::Error(store_error));
            }
            Some(AgentRunState::Error(error))
        }
    }
}

async fn initialize_single_turn_state<M, S>(
    input: AgentInput,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
) -> Result<InitializedState, TurnOutcome>
where
    M: MessageStore,
    S: StateStore,
{
    match initialize_from_input(
        input,
        thread_id,
        message_store,
        state_store,
        execution_store,
        audit_sink,
        provenance,
    )
    .await
    {
        Ok(state) => Ok(state),
        Err(error) => Err(TurnOutcome::Error(error)),
    }
}

async fn handle_single_turn_resume_state<Ctx, H, M, S>(
    params: SingleTurnResumeParams<Ctx, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let turn = params.turn;
    let thread_id = params.thread_id.clone();
    let event_store = Arc::clone(&params.event_store);
    let outcome = handle_single_turn_resume(params).await;
    if !turn_outcome_keeps_turn_open(&outcome)
        && let Err(store_error) = finish_turn_or_error(&event_store, &thread_id, turn).await
    {
        return TurnOutcome::Error(store_error);
    }
    outcome
}

fn build_turn_context(
    thread_id: &ThreadId,
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    start_time: Instant,
) -> TurnContext {
    TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
        compaction_retries: 0,
        pending_reminder: None,
    }
}

const fn cancelled_turn_outcome() -> TurnOutcome {
    TurnOutcome::Cancelled {
        total_turns: 0,
        input_tokens: 0,
        output_tokens: 0,
    }
}

const fn turn_outcome_keeps_turn_open(outcome: &TurnOutcome) -> bool {
    matches!(outcome, TurnOutcome::AwaitingConfirmation { .. })
}

fn done_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Done {
        total_turns: turns_to_u32(ctx.turn),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
}

fn cancelled_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Cancelled {
        total_turns: turns_to_u32(ctx.turn),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
}

fn refusal_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Refusal {
        total_turns: turns_to_u32(ctx.turn),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
}

async fn emit_persistent_turn_complete<H>(
    ctx: &TurnContext,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    current_turn: usize,
) -> Result<(), AgentRunState>
where
    H: AgentHooks,
{
    if let Err(error) = send_event(
        event_store,
        &ctx.thread_id,
        current_turn,
        hooks,
        authority,
        AgentEvent::TurnComplete {
            turn: ctx.turn,
            usage: ctx.total_usage.clone(),
        },
    )
    .await
    {
        return Err(AgentRunState::Error(error));
    }
    if let Err(error) = finish_turn_or_error(event_store, &ctx.thread_id, current_turn).await {
        return Err(AgentRunState::Error(error));
    }
    Ok(())
}

async fn handle_persistent_done<M, H>(
    PersistentDoneParams {
        ctx,
        rx,
        message_store,
        event_store,
        hooks,
        authority,
        current_turn,
        cancel_token,
    }: PersistentDoneParams<'_, H, M>,
) -> Option<AgentRunState>
where
    M: MessageStore,
    H: AgentHooks,
{
    if let Err(state) =
        emit_persistent_turn_complete(ctx, event_store, hooks, authority, current_turn).await
    {
        return Some(state);
    }

    tokio::select! {
        msg = rx.recv() => {
            match msg {
                Some(AgentInput::Text(text)) => {
                    let user_msg = Message::user(&text);
                    if let Err(error) = message_store.append(&ctx.thread_id, user_msg).await {
                        warn!("Failed to append injected message: {error}");
                        return Some(done_run_state(ctx));
                    }
                    None
                }
                Some(AgentInput::Message(blocks)) => {
                    let user_msg = Message::user_with_content(blocks);
                    if let Err(error) = message_store.append(&ctx.thread_id, user_msg).await {
                        warn!("Failed to append injected message: {error}");
                        return Some(done_run_state(ctx));
                    }
                    None
                }
                _ => Some(done_run_state(ctx)),
            }
        }
        () = cancel_token.cancelled() => Some(cancelled_run_state(ctx)),
    }
}

async fn finish_turn_or_run_state(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
) -> Result<(), RunLoopTurnAction> {
    finish_turn_or_error(event_store, thread_id, turn)
        .await
        .map_err(|error| RunLoopTurnAction::Return(AgentRunState::Error(error)))
}

async fn handle_run_loop_turn_result<H, M, S>(
    RunLoopTurnResultParams {
        result,
        ctx,
        input_rx,
        message_store,
        state_store,
        event_store,
        hooks,
        authority,
        cancel_token,
        current_turn,
    }: RunLoopTurnResultParams<'_, H, M, S>,
) -> RunLoopTurnAction
where
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    match result {
        InternalTurnResult::Continue { .. } => {
            if let Err(error) = state_store.save(&ctx.state).await {
                warn!("Failed to save state checkpoint: {error}");
            }
            finish_turn_or_run_state(event_store, &ctx.thread_id, current_turn)
                .await
                .map_or_else(std::convert::identity, |()| RunLoopTurnAction::Continue)
        }
        InternalTurnResult::Done => {
            if let Some(rx) = input_rx {
                handle_persistent_done(super::types::PersistentDoneParams {
                    ctx,
                    rx,
                    message_store,
                    event_store,
                    hooks,
                    authority,
                    current_turn,
                    cancel_token,
                })
                .await
                .map_or(RunLoopTurnAction::Continue, RunLoopTurnAction::Return)
            } else {
                RunLoopTurnAction::FinishRun
            }
        }
        InternalTurnResult::Refusal => {
            finish_turn_or_run_state(event_store, &ctx.thread_id, current_turn)
                .await
                .map_or_else(std::convert::identity, |()| {
                    RunLoopTurnAction::Return(refusal_run_state(ctx))
                })
        }
        InternalTurnResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        } => RunLoopTurnAction::Return(AgentRunState::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
        }),
        InternalTurnResult::PendingToolCalls { .. } => finish_turn_or_run_state(
            event_store,
            &ctx.thread_id,
            current_turn,
        )
        .await
        .map_or_else(std::convert::identity, |()| {
            RunLoopTurnAction::Return(AgentRunState::Error(crate::types::AgentError::new(
                "PendingToolCalls returned in looping mode (expected inline tool execution)",
                false,
            )))
        }),
        InternalTurnResult::Error(error) => {
            finish_turn_or_run_state(event_store, &ctx.thread_id, current_turn)
                .await
                .map_or_else(std::convert::identity, |()| {
                    RunLoopTurnAction::Return(AgentRunState::Error(error))
                })
        }
    }
}

async fn finish_run_loop_success<H, S>(
    ctx: TurnContext,
    state_store: &Arc<S>,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
) -> AgentRunState
where
    H: AgentHooks,
    S: StateStore,
{
    if let Err(error) = state_store.save(&ctx.state).await {
        warn!("Failed to save final state: {error}");
    }

    let duration = ctx.start_time.elapsed();
    if let Err(error) = send_event(
        event_store,
        &ctx.thread_id,
        ctx.turn,
        hooks,
        authority,
        AgentEvent::done(
            ctx.thread_id.clone(),
            ctx.turn,
            ctx.total_usage.clone(),
            duration,
        ),
    )
    .await
    {
        return AgentRunState::Error(error);
    }
    if let Err(error) = finish_turn_or_error(event_store, &ctx.thread_id, ctx.turn).await {
        return AgentRunState::Error(error);
    }

    AgentRunState::Done {
        total_turns: turns_to_u32(ctx.turn),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
}

pub(super) async fn run_loop_turns<Ctx, P, H, M, S>(
    RunLoopTurnsParams {
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        event_store,
        authority,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        provenance,
        cancel_token,
        mut input_rx,
        turn_options,
        #[cfg(feature = "otel")]
        observability_store,
    }: RunLoopTurnsParams<'_, Ctx, P, H, M, S>,
) -> Option<AgentRunState>
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    loop {
        if cancel_token.is_cancelled() {
            log::info!("Agent run cancelled before turn {}", ctx.turn);
            return Some(cancelled_run_state(ctx));
        }

        let current_turn = ctx.turn.saturating_add(1);
        let turn_tool_context = tool_context.clone().with_event_store(
            Arc::clone(event_store),
            ctx.thread_id.clone(),
            current_turn,
            Arc::clone(authority),
        );
        let result = execute_turn(ExecuteTurnParameters {
            event_store,
            authority,
            ctx,
            tool_context: &turn_tool_context,
            provider,
            tools,
            hooks,
            message_store,
            state_store,
            config,
            compaction_config,
            compactor,
            execution_store,
            audit_sink,
            provenance,
            turn_options,
            #[cfg(feature = "otel")]
            observability_store,
        })
        .await;

        match handle_run_loop_turn_result(super::types::RunLoopTurnResultParams {
            result,
            ctx,
            input_rx: input_rx.as_deref_mut(),
            message_store,
            state_store,
            event_store,
            hooks,
            authority,
            cancel_token,
            current_turn,
        })
        .await
        {
            RunLoopTurnAction::Continue => {}
            RunLoopTurnAction::FinishRun => return None,
            RunLoopTurnAction::Return(state) => return Some(state),
        }
    }
}

pub(super) async fn handle_single_turn_resume<Ctx, H, M, S>(
    SingleTurnResumeParams {
        resume_data,
        turn,
        total_usage,
        state,
        thread_id,
        tool_context,
        tools,
        hooks,
        event_store,
        authority,
        message_store,
        state_store,
        execution_store,
        audit_sink,
        provenance,
    }: SingleTurnResumeParams<Ctx, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let resume_result = process_resume(ResumeProcessingParameters {
        resume_data,
        turn,
        total_usage: &total_usage,
        state: &state,
        thread_id: &thread_id,
        tool_context: &tool_context,
        tools: &tools,
        hooks: &hooks,
        event_store: &event_store,
        authority: &authority,
        message_store: &message_store,
        execution_store: execution_store.as_ref(),
        audit_sink: &audit_sink,
        provenance: &provenance,
    })
    .await;

    match resume_result {
        Ok(ResumeProcessingResult::Completed { turn_usage }) => {
            let mut updated_state = state;
            updated_state.turn_count = turn;
            if let Err(error) = state_store.save(&updated_state).await {
                warn!("Failed to save state checkpoint: {error}");
            }
            TurnOutcome::NeedsMoreTurns {
                turn,
                turn_usage,
                total_usage,
            }
        }
        Ok(ResumeProcessingResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        }) => TurnOutcome::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
        },
        Err(error) => {
            if let Err(store_error) = send_event(
                &event_store,
                &thread_id,
                turn,
                &hooks,
                &authority,
                AgentEvent::error(&error.message, error.recoverable),
            )
            .await
            {
                return TurnOutcome::Error(store_error);
            }
            TurnOutcome::Error(error)
        }
    }
}

/// Checks if the last message in the history is an assistant message with
/// `ToolUse` content blocks but no subsequent user message containing
/// `ToolResult` blocks. This indicates a crash between persisting the
/// assistant response and executing tools.
fn has_orphaned_tool_use(messages: &[Message]) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };
    if last.role != Role::Assistant {
        return false;
    }
    let Content::Blocks(blocks) = &last.content else {
        return false;
    };
    blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
}

/// Synthesizes error `ToolResult` blocks for every `ToolUse` in the last
/// assistant message, allowing the conversation to continue after a crash.
fn synthesize_error_tool_results(messages: &[Message]) -> Option<Message> {
    let last = messages.last()?;
    let Content::Blocks(blocks) = &last.content else {
        return None;
    };

    let result_blocks: Vec<ContentBlock> = blocks
        .iter()
        .filter_map(|b| {
            if let ContentBlock::ToolUse { id, .. } = b {
                Some(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: "Tool execution was interrupted by a crash. Please retry.".to_string(),
                    is_error: Some(true),
                })
            } else {
                None
            }
        })
        .collect();

    if result_blocks.is_empty() {
        return None;
    }

    Some(Message {
        role: Role::User,
        content: Content::Blocks(result_blocks),
    })
}

/// Recovers from orphaned `tool_use` messages by appending synthetic error
/// `tool_result` blocks so the conversation can continue.
async fn recover_orphaned_tool_use<M>(
    thread_id: &ThreadId,
    message_store: &Arc<M>,
) -> Result<(), AgentError>
where
    M: MessageStore,
{
    let history = message_store
        .get_history(thread_id)
        .await
        .map_err(|e| AgentError::new(format!("Failed to get history for recovery: {e}"), false))?;

    if has_orphaned_tool_use(&history) {
        warn!("Detected orphaned tool_use blocks — synthesizing error results for crash recovery");
        if let Some(recovery_msg) = synthesize_error_tool_results(&history) {
            message_store
                .append(thread_id, recovery_msg)
                .await
                .map_err(|e| {
                    AgentError::new(
                        format!("Failed to append recovery tool results: {e}"),
                        false,
                    )
                })?;
        }
    }
    Ok(())
}

pub(super) async fn run_loop<Ctx, P, H, M, S>(
    params: RunLoopParameters<Ctx, P, H, M, S>,
) -> AgentRunState
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    #[cfg(feature = "otel")]
    let mut root_span = crate::observability::instrument::start_root_span(
        params.provider.as_ref(),
        &params.tools,
        &params.config,
        &params.thread_id,
        &params.input,
        "loop",
    );
    #[cfg(feature = "otel")]
    let root_context = {
        use opentelemetry::trace::Span;

        crate::observability::context::current_with_span_context(root_span.span_context().clone())
    };

    #[cfg(feature = "otel")]
    let result = {
        use opentelemetry::trace::FutureExt;

        run_loop_inner(params).with_context(root_context).await
    };
    #[cfg(not(feature = "otel"))]
    let result = run_loop_inner(params).await;

    #[cfg(feature = "otel")]
    {
        use crate::observability::instrument::{end_root_span, run_state_outcome};
        let (turns, inp, out) = match &result {
            AgentRunState::Done {
                total_turns,
                input_tokens,
                output_tokens,
            }
            | AgentRunState::Refusal {
                total_turns,
                input_tokens,
                output_tokens,
            } => (
                usize::try_from(*total_turns).unwrap_or(0),
                *input_tokens,
                *output_tokens,
            ),
            _ => (0, 0, 0),
        };
        end_root_span(&mut root_span, turns, inp, out, run_state_outcome(&result));
    }

    result
}

async fn run_loop_inner<Ctx, P, H, M, S>(
    RunLoopParameters {
        event_store,
        authority,
        thread_id,
        input,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        cancel_token,
        mut input_rx,
        #[cfg(feature = "otel")]
        observability_store,
    }: RunLoopParameters<Ctx, P, H, M, S>,
) -> AgentRunState
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let tool_context = tool_context.with_cancel_token(cancel_token.clone());
    let provenance =
        agent_sdk_core::audit::AuditProvenance::new(provider.provider(), provider.model());
    let start_time = Instant::now();
    let init_state = match initialize_run_loop_state(
        input,
        &thread_id,
        &message_store,
        &state_store,
        execution_store.as_ref(),
        &audit_sink,
        &provenance,
    )
    .await
    {
        Ok(state) => state,
        Err(error) => return error,
    };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume_data) = resume_data {
        let resume_tool_context = tool_context.clone().with_event_store(
            Arc::clone(&event_store),
            thread_id.clone(),
            turn,
            Arc::clone(&authority),
        );
        if let Some(outcome) = handle_run_loop_resume_state(RunLoopResumeParams {
            resume_data,
            turn,
            total_usage: &total_usage,
            state: &state,
            thread_id: &thread_id,
            tool_context: &resume_tool_context,
            tools: &tools,
            hooks: &hooks,
            event_store: &event_store,
            authority: &authority,
            message_store: &message_store,
            execution_store: execution_store.as_ref(),
            audit_sink: &audit_sink,
            provenance: &provenance,
        })
        .await
        {
            return outcome;
        }
    }

    let mut ctx = build_turn_context(&thread_id, turn, total_usage, state, start_time);

    let default_turn_options = TurnOptions::default();

    if let Some(outcome) = run_loop_turns(RunLoopTurnsParams {
        ctx: &mut ctx,
        tool_context: &tool_context,
        provider: &provider,
        tools: &tools,
        hooks: &hooks,
        message_store: &message_store,
        state_store: &state_store,
        event_store: &event_store,
        authority: &authority,
        config: &config,
        compaction_config: compaction_config.as_ref(),
        compactor: compactor.as_ref(),
        execution_store: execution_store.as_ref(),
        audit_sink: &audit_sink,
        provenance: &provenance,
        cancel_token: &cancel_token,
        input_rx: input_rx.as_mut(),
        turn_options: &default_turn_options,
        #[cfg(feature = "otel")]
        observability_store: observability_store.as_ref(),
    })
    .await
    {
        return outcome;
    }

    finish_run_loop_success(ctx, &state_store, &event_store, &hooks, &authority).await
}

/// Run a single turn of the agent loop.
///
/// This is similar to `run_loop` but only executes one turn and returns.
/// The caller is responsible for continuing execution by calling again with
/// `AgentInput::Continue`.
pub(super) async fn run_single_turn<Ctx, P, H, M, S>(
    params: TurnParameters<Ctx, P, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    #[cfg(feature = "otel")]
    let mut root_span = crate::observability::instrument::start_root_span(
        params.provider.as_ref(),
        &params.tools,
        &params.config,
        &params.thread_id,
        &params.input,
        "single_turn",
    );
    #[cfg(feature = "otel")]
    let root_context = {
        use opentelemetry::trace::Span;

        crate::observability::context::current_with_span_context(root_span.span_context().clone())
    };

    #[cfg(feature = "otel")]
    let outcome = {
        use opentelemetry::trace::FutureExt;

        run_single_turn_inner(params)
            .with_context(root_context)
            .await
    };
    #[cfg(not(feature = "otel"))]
    let outcome = run_single_turn_inner(params).await;

    #[cfg(feature = "otel")]
    {
        use crate::observability::instrument::{end_root_span, turn_outcome_str};
        let (turns, inp, out) = match &outcome {
            TurnOutcome::Done {
                total_turns,
                input_tokens,
                output_tokens,
            }
            | TurnOutcome::Refusal {
                total_turns,
                input_tokens,
                output_tokens,
            } => (
                usize::try_from(*total_turns).unwrap_or(0),
                *input_tokens,
                *output_tokens,
            ),
            TurnOutcome::NeedsMoreTurns {
                turn, total_usage, ..
            } => (
                *turn,
                u64::from(total_usage.input_tokens),
                u64::from(total_usage.output_tokens),
            ),
            _ => (0, 0, 0),
        };
        end_root_span(&mut root_span, turns, inp, out, turn_outcome_str(&outcome));
    }

    outcome
}

#[allow(clippy::too_many_lines)]
async fn run_single_turn_inner<Ctx, P, H, M, S>(
    TurnParameters {
        event_store,
        authority,
        thread_id,
        input,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        config,
        compaction_config,
        compactor,
        execution_store,
        audit_sink,
        cancel_token,
        turn_options,
        #[cfg(feature = "otel")]
        observability_store,
    }: TurnParameters<Ctx, P, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    // Check for cancellation before starting any work
    if cancel_token.is_cancelled() {
        log::info!("Agent turn cancelled before execution started");
        return cancelled_turn_outcome();
    }

    let tool_context = tool_context.with_cancel_token(cancel_token.clone());
    let provenance =
        agent_sdk_core::audit::AuditProvenance::new(provider.provider(), provider.model());
    let start_time = Instant::now();
    let init_state = match initialize_single_turn_state(
        input,
        &thread_id,
        &message_store,
        &state_store,
        execution_store.as_ref(),
        &audit_sink,
        &provenance,
    )
    .await
    {
        Ok(state) => state,
        Err(outcome) => return outcome,
    };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume_data) = resume_data {
        let resume_tool_context = tool_context.clone().with_event_store(
            Arc::clone(&event_store),
            thread_id.clone(),
            turn,
            Arc::clone(&authority),
        );
        return handle_single_turn_resume_state(SingleTurnResumeParams {
            resume_data,
            turn,
            total_usage,
            state,
            thread_id: thread_id.clone(),
            tool_context: resume_tool_context,
            tools,
            hooks,
            event_store: Arc::clone(&event_store),
            authority,
            message_store,
            state_store,
            execution_store,
            audit_sink,
            provenance,
        })
        .await;
    }

    let mut ctx = build_turn_context(&thread_id, turn, total_usage, state, start_time);

    let current_turn = ctx.turn.saturating_add(1);
    let turn_tool_context = tool_context.clone().with_event_store(
        Arc::clone(&event_store),
        thread_id.clone(),
        current_turn,
        Arc::clone(&authority),
    );
    let result = execute_turn(ExecuteTurnParameters {
        event_store: &event_store,
        authority: &authority,
        ctx: &mut ctx,
        tool_context: &turn_tool_context,
        provider: &provider,
        tools: &tools,
        hooks: &hooks,
        message_store: &message_store,
        state_store: &state_store,
        config: &config,
        compaction_config: compaction_config.as_ref(),
        compactor: compactor.as_ref(),
        execution_store: execution_store.as_ref(),
        audit_sink: &audit_sink,
        provenance: &provenance,
        turn_options: &turn_options,
        #[cfg(feature = "otel")]
        observability_store: observability_store.as_ref(),
    })
    .await;

    let outcome = convert_turn_result(ConvertTurnResultParams {
        result,
        ctx,
        event_store: &event_store,
        hooks: &hooks,
        authority: &authority,
        thread_id: thread_id.clone(),
        current_turn,
        state_store: &state_store,
    })
    .await;

    if !turn_outcome_keeps_turn_open(&outcome)
        && let Err(store_error) = finish_turn_or_error(&event_store, &thread_id, current_turn).await
    {
        return TurnOutcome::Error(store_error);
    }

    outcome
}

pub(super) async fn convert_turn_result<H: AgentHooks, S: StateStore>(
    ConvertTurnResultParams {
        result,
        ctx,
        event_store,
        hooks,
        authority,
        thread_id,
        current_turn,
        state_store,
    }: ConvertTurnResultParams<'_, H, S>,
) -> TurnOutcome {
    match result {
        InternalTurnResult::Continue { turn_usage } => {
            if let Err(e) = state_store.save(&ctx.state).await {
                warn!("Failed to save state checkpoint: {e}");
            }
            TurnOutcome::NeedsMoreTurns {
                turn: ctx.turn,
                turn_usage,
                total_usage: ctx.total_usage,
            }
        }
        InternalTurnResult::Done => {
            if let Err(e) = state_store.save(&ctx.state).await {
                warn!("Failed to save final state: {e}");
            }
            let duration = ctx.start_time.elapsed();
            if let Err(error) = send_event(
                event_store,
                &thread_id,
                current_turn,
                hooks,
                authority,
                AgentEvent::done(
                    thread_id.clone(),
                    ctx.turn,
                    ctx.total_usage.clone(),
                    duration,
                ),
            )
            .await
            {
                return TurnOutcome::Error(error);
            }
            TurnOutcome::Done {
                total_turns: turns_to_u32(ctx.turn),
                input_tokens: u64::from(ctx.total_usage.input_tokens),
                output_tokens: u64::from(ctx.total_usage.output_tokens),
            }
        }
        InternalTurnResult::Refusal => TurnOutcome::Refusal {
            total_turns: turns_to_u32(ctx.turn),
            input_tokens: u64::from(ctx.total_usage.input_tokens),
            output_tokens: u64::from(ctx.total_usage.output_tokens),
        },
        InternalTurnResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        } => TurnOutcome::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
        },
        InternalTurnResult::PendingToolCalls {
            turn_usage,
            pending_tool_calls,
            continuation,
        } => TurnOutcome::PendingToolCalls {
            turn: ctx.turn,
            turn_usage,
            total_usage: ctx.total_usage,
            tool_calls: pending_tool_calls,
            continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
        },
        InternalTurnResult::Error(e) => TurnOutcome::Error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_orphaned_tool_use_empty_history() {
        assert!(!has_orphaned_tool_use(&[]));
    }

    #[test]
    fn test_has_orphaned_tool_use_user_last() {
        let messages = vec![Message::user("hello")];
        assert!(!has_orphaned_tool_use(&messages));
    }

    #[test]
    fn test_has_orphaned_tool_use_assistant_text_only() {
        let messages = vec![Message::assistant("Sure, I can help.")];
        assert!(!has_orphaned_tool_use(&messages));
    }

    #[test]
    fn test_has_orphaned_tool_use_assistant_with_tool_use() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"path": "/test"}),
                thought_signature: None,
            }]),
        }];
        assert!(has_orphaned_tool_use(&messages));
    }

    #[test]
    fn test_synthesize_error_tool_results() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![
                ContentBlock::Text {
                    text: "Let me read that.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({"path": "/test"}),
                    thought_signature: None,
                },
                ContentBlock::ToolUse {
                    id: "tool_2".to_string(),
                    name: "grep".to_string(),
                    input: serde_json::json!({"pattern": "foo"}),
                    thought_signature: None,
                },
            ]),
        }];

        let recovery = synthesize_error_tool_results(&messages);
        assert!(recovery.is_some());

        let msg = recovery.unwrap();
        assert_eq!(msg.role, Role::User);

        let Content::Blocks(blocks) = &msg.content else {
            panic!("Expected Blocks");
        };
        assert_eq!(blocks.len(), 2);
        for block in blocks {
            let ContentBlock::ToolResult { is_error, .. } = block else {
                panic!("Expected ToolResult");
            };
            assert_eq!(*is_error, Some(true));
        }
    }

    #[test]
    fn test_validate_external_tool_results_ok() {
        use crate::types::{
            AgentContinuation, AgentState, ExternalToolResult, PendingToolCallInfo, TokenUsage,
            ToolResult,
        };

        let thread = ThreadId::new();
        let cont = AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_1".into(),
                name: "echo".into(),
                display_name: "Echo".into(),
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
        };

        let results = vec![ExternalToolResult {
            tool_call_id: "call_1".into(),
            result: ToolResult::success("ok"),
        }];

        assert!(validate_external_tool_results(&cont, &results).is_ok());
    }

    #[test]
    fn test_validate_external_tool_results_missing() {
        use crate::types::{AgentContinuation, AgentState, PendingToolCallInfo, TokenUsage};

        let thread = ThreadId::new();
        let cont = AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![
                PendingToolCallInfo {
                    id: "call_1".into(),
                    name: "echo".into(),
                    display_name: "Echo".into(),
                    input: serde_json::json!({}),
                    effective_input: serde_json::json!({}),
                    listen_context: None,
                },
                PendingToolCallInfo {
                    id: "call_2".into(),
                    name: "write".into(),
                    display_name: "Write".into(),
                    input: serde_json::json!({}),
                    effective_input: serde_json::json!({}),
                    listen_context: None,
                },
            ],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
        };

        // Only provide one result for two pending calls
        let results = vec![crate::types::ExternalToolResult {
            tool_call_id: "call_1".into(),
            result: crate::types::ToolResult::success("ok"),
        }];

        let err = validate_external_tool_results(&cont, &results);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("call_2"),
            "Error should mention missing call_2: {msg}"
        );
    }

    #[test]
    fn test_validate_external_tool_results_unknown_id() {
        use crate::types::{
            AgentContinuation, AgentState, ExternalToolResult, PendingToolCallInfo, TokenUsage,
            ToolResult,
        };

        let thread = ThreadId::new();
        let cont = AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_1".into(),
                name: "echo".into(),
                display_name: "Echo".into(),
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
        };

        // Provide the correct result AND an extra unknown one
        let results = vec![
            ExternalToolResult {
                tool_call_id: "call_1".into(),
                result: ToolResult::success("ok"),
            },
            ExternalToolResult {
                tool_call_id: "bogus_id".into(),
                result: ToolResult::success("extra"),
            },
        ];

        let err = validate_external_tool_results(&cont, &results);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("bogus_id"),
            "Error should mention bogus_id: {msg}"
        );
    }

    #[test]
    fn test_validate_external_tool_results_empty_continuation() {
        use crate::types::{AgentContinuation, AgentState, TokenUsage};

        let thread = ThreadId::new();
        let cont = AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
        };

        let err = validate_external_tool_results(&cont, &[]);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("no pending tool calls"), "Error: {msg}");
    }
}
