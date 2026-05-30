use super::helpers::{pending_tool_index, send_event, turns_to_u32};
use super::tool_execution::{append_tool_results, execute_confirmed_tool, execute_tool_call};
use super::turn::execute_turn;
use super::types::{
    ConfirmedToolExecutionContext, ConvertTurnResultParams, ExecuteTurnParameters,
    InitializedState, InternalTurnResult, PersistentDoneParams, ResumeData,
    ResumeProcessingParameters, ResumeProcessingResult, ResumeSummaryMetrics, RunLoopParameters,
    RunLoopResumeParams, RunLoopTurnResultParams, RunLoopTurnsParams, SingleTurnResumeParams,
    ToolCallExecutionContext, ToolExecutionOutcome, TurnContext, TurnParameters,
};

use crate::types::TurnOptions;

use crate::authority::EventAuthority;
use crate::context::{CompactionConfig, ContextCompactor};
use crate::events::AgentEvent;
use crate::hooks::AgentHooks;
use crate::llm::{Content, ContentBlock, LlmProvider, Message, Role, StopReason};
use crate::stores::{EventStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    ContinuationEnvelope, ThreadId, TokenUsage, ToolResult, TurnOutcome, TurnSummary,
};
use agent_sdk_core::audit::AuditProvenance;
use log::warn;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

enum RunLoopTurnAction {
    Continue,
    FinishRun,
    Return(AgentRunState),
}

/// Stamp the run's SDK-boundary controls onto the tool context.
///
/// Injects the run's cancel token and the configured per-tool timeout
/// (`AgentConfig::tool_timeout_ms`) so the tool-execution boundary can
/// race every tool's `execute()` against both. Mirrors the historical
/// `with_cancel_token` injection and is the single place the timeout
/// crosses from config into the tool context.
fn apply_tool_boundary_controls<Ctx>(
    tool_context: ToolContext<Ctx>,
    cancel_token: &tokio_util::sync::CancellationToken,
    tool_timeout_ms: Option<u64>,
) -> ToolContext<Ctx> {
    let tool_context = tool_context.with_cancel_token(cancel_token.clone());
    match tool_timeout_ms {
        Some(ms) => tool_context.with_tool_timeout(std::time::Duration::from_millis(ms)),
        None => tool_context,
    }
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
///
/// ## Durability ordering
///
/// The SDK writes to the message store **first** and only then marks the
/// execution store, so that if `append_tool_results` fails the caller can
/// safely retry with the same continuation and have every submission
/// re-evaluated as fresh (rather than short-circuited as a replay).
/// Otherwise a transient append failure would permanently break the
/// conversation: every subsequent retry would see a "completed"
/// execution entry, skip the append, and leave the assistant message
/// with dangling `ToolUse` blocks.
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
    //
    // NOTE: we do NOT write to the execution store here. The execution
    // store is only updated AFTER `append_tool_results` succeeds, so
    // that a transient message-store failure can be safely retried
    // without leaving orphaned "completed" execution rows that would
    // short-circuit the retry as a replay. See the durability ordering
    // note on the function doc above.
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
            // Audit `Completed` *before* the append. Sink consumers see
            // the provider-reported outcome first; if the append then
            // fails a follow-up `PersistenceFailed` record is emitted
            // below so consumers can reconcile the durability gap.
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
        // both the intended outcome and the durability gap. The
        // execution store has *not* been written yet, so the caller can
        // safely retry the same continuation and every submission will
        // be re-evaluated as fresh.
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

    // Message-store append succeeded — now it's safe to mark the
    // execution store so the next submission of the same tool_call_id
    // is detected as a replay.
    for (tool_call_id, result) in &fresh_results {
        record_external_tool_execution(
            execution_store,
            &continuation,
            thread_id,
            tool_call_id,
            result,
        )
        .await;
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
        warn!("Failed to record external tool execution (tool_call_id={tool_call_id}, error={e})");
    }
}

/// Emit one audit record for a submitted external tool result.
///
/// Looks the tool metadata up on the continuation's pending list so the
/// record carries the correct tier, name, and input — the values the
/// continuation already persisted at the moment the LLM requested the
/// tool call, not a wildcard guess.
///
/// `validate_external_tool_results` rejects submissions whose
/// `tool_call_id` is not in the pending list before this function runs,
/// so the fallback identity arm exists only as a defensive
/// belt-and-braces guard and should be unreachable in practice.
async fn emit_external_tool_audit(
    audit_sink: &Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &agent_sdk_core::audit::AuditProvenance,
    continuation: &AgentContinuation,
    tool_call_id: &str,
    outcome: agent_sdk_core::audit::ToolAuditOutcome,
) {
    use crate::types::ToolTier;
    use agent_sdk_core::audit::{ToolAuditRecord, ToolAuditRecordParams};

    let pending = continuation
        .pending_tool_calls
        .iter()
        .find(|p| p.id == tool_call_id);
    let (tool_name, display_name, tier, requested_input, effective_input) = pending.map_or_else(
        || {
            (
                String::new(),
                String::new(),
                ToolTier::Confirm,
                serde_json::Value::Null,
                serde_json::Value::Null,
            )
        },
        |p| {
            (
                p.name.clone(),
                p.display_name.clone(),
                p.tier,
                p.input.clone(),
                p.effective_input.clone(),
            )
        },
    );
    let record = ToolAuditRecord::new(ToolAuditRecordParams {
        tool_call_id: tool_call_id.to_string(),
        tool_name,
        display_name,
        tier,
        requested_input,
        effective_input,
        turn: continuation.turn,
        provenance: provenance.clone(),
        outcome,
    });
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
    // Snapshot the turn-closing LLM metadata before we mutate `cont`.
    // These describe the pre-pause LLM call (the one that produced
    // `pending_tool_calls`) and must survive any nested
    // `AwaitingConfirmation` hand-off so every resume branch reports
    // the same values as the pre-pause summary for this turn.
    let carried_metadata = CarriedTurnMetadata {
        response_id: cont.response_id.clone(),
        stop_reason: cont.stop_reason,
        tool_call_count: cont.pending_tool_calls.len(),
    };

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

    if let Some(result) = execute_remaining_pending_tools(ExecuteRemainingParams {
        cont: &cont,
        tool_results: &mut tool_results,
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
        total_usage,
        state,
        carried: &carried_metadata,
    })
    .await?
    {
        return Ok(result);
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
        metrics: ResumeSummaryMetrics {
            response_id: carried_metadata.response_id,
            stop_reason: carried_metadata.stop_reason,
            tool_call_count: carried_metadata.tool_call_count,
        },
    })
}

/// Turn-closing LLM metadata threaded through [`process_resume`].
///
/// The resume path never calls the LLM, so every `TurnSummary` it
/// emits has to re-use the values captured into the continuation at
/// pause time. Keeping them in a small struct makes the plumbing
/// self-documenting and stops any future resume branch from silently
/// dropping them on the floor.
struct CarriedTurnMetadata {
    response_id: Option<String>,
    stop_reason: Option<StopReason>,
    tool_call_count: usize,
}

struct ExecuteRemainingParams<'a, Ctx, H> {
    cont: &'a AgentContinuation,
    tool_results: &'a mut Vec<(String, ToolResult)>,
    tool_context: &'a ToolContext<Ctx>,
    thread_id: &'a ThreadId,
    tools: &'a Arc<ToolRegistry<Ctx>>,
    hooks: &'a Arc<H>,
    event_store: &'a Arc<dyn EventStore>,
    turn: usize,
    authority: &'a Arc<dyn EventAuthority>,
    execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    audit_sink: &'a Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: &'a AuditProvenance,
    total_usage: &'a TokenUsage,
    state: &'a AgentState,
    carried: &'a CarriedTurnMetadata,
}

/// Execute every pending tool after the one that was just confirmed.
///
/// Returns `Ok(Some(AwaitingConfirmation))` if another tool needs
/// user confirmation, `Ok(None)` if the whole batch completed, or an
/// `Err` if any tool's event plumbing failed.
///
/// The nested continuation carries the same `response_id` /
/// `stop_reason` that was threaded into this call, so the next resume
/// still reports the original turn-closing LLM metadata.
async fn execute_remaining_pending_tools<Ctx, H>(
    ExecuteRemainingParams {
        cont,
        tool_results,
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
        total_usage,
        state,
        carried,
    }: ExecuteRemainingParams<'_, Ctx, H>,
) -> Result<Option<ResumeProcessingResult>, AgentError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
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

                return Ok(Some(ResumeProcessingResult::AwaitingConfirmation {
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
                        completed_results: std::mem::take(tool_results),
                        state: state.clone(),
                        response_id: carried.response_id.clone(),
                        stop_reason: carried.stop_reason,
                        response_content: Vec::new(),
                    }),
                }));
            }
            ToolExecutionOutcome::Error(error) => return Err(error),
        }
    }
    Ok(None)
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
    #[cfg(feature = "otel")] input_kind: &'static str,
) -> TurnContext {
    TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
        compaction_retries: 0,
        pending_reminder: None,
        response_id: None,
        stop_reason: None,
        tool_call_count: 0,
        #[cfg(feature = "otel")]
        input_kind,
    }
}

/// Build a structured [`TurnSummary`] from a [`TurnContext`] and the
/// active turn options / provenance.
///
/// Used by every `TurnOutcome` variant that carries a summary on the
/// regular turn path. Duration is measured from `ctx.start_time` so
/// every variant reports the same wall-clock interval from the start
/// of `run_turn` to the outcome construction site.
///
/// The resume path uses [`build_turn_summary_from_parts`] instead —
/// it cannot synthesise a faithful [`TurnContext`] because it never
/// called the LLM for this turn.
fn build_turn_summary(
    ctx: &TurnContext,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
    turn_usage: TokenUsage,
) -> TurnSummary {
    build_turn_summary_from_parts(TurnSummaryParts {
        thread_id: &ctx.thread_id,
        turn: ctx.turn,
        turn_usage,
        total_usage: &ctx.total_usage,
        provenance,
        response_id: ctx.response_id.as_deref(),
        stop_reason: ctx.stop_reason,
        tool_call_count: ctx.tool_call_count,
        start_time: ctx.start_time,
        turn_options,
    })
}

/// Flat inputs to [`build_turn_summary_from_parts`].
///
/// Used instead of [`TurnContext`] by call sites that do not own a
/// full turn context — currently the resume handler, which relies on
/// [`ResumeSummaryMetrics`] rehydrated from the continuation rather
/// than from a live LLM call.
struct TurnSummaryParts<'a> {
    thread_id: &'a ThreadId,
    turn: usize,
    turn_usage: TokenUsage,
    total_usage: &'a TokenUsage,
    provenance: &'a AuditProvenance,
    response_id: Option<&'a str>,
    stop_reason: Option<agent_sdk_core::llm::StopReason>,
    tool_call_count: usize,
    start_time: Instant,
    turn_options: &'a TurnOptions,
}

fn build_turn_summary_from_parts(parts: TurnSummaryParts<'_>) -> TurnSummary {
    TurnSummary {
        thread_id: parts.thread_id.clone(),
        turn: parts.turn,
        total_turns: turns_to_u32(parts.turn),
        turn_usage: parts.turn_usage,
        total_usage: parts.total_usage.clone(),
        provenance: parts.provenance.clone(),
        response_id: parts.response_id.map(str::to_string),
        stop_reason: parts.stop_reason,
        tool_call_count: parts.tool_call_count,
        duration_ms: duration_ms_saturating(parts.start_time.elapsed()),
        tool_runtime: parts.turn_options.tool_runtime.clone(),
        strict_durability: parts.turn_options.strict_durability,
    }
}

/// Saturating conversion from [`Duration`] to milliseconds, clamped to
/// [`u64::MAX`] on the unlikely overflow so the summary never panics on
/// a pathological clock.
fn duration_ms_saturating(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Build a synthetic summary for outcome paths that do not go through
/// `TurnContext` — currently only the "cancelled before the first LLM
/// call" path from [`cancelled_turn_outcome`]. All LLM-level fields
/// (`response_id`, `stop_reason`, `tool_call_count`) are `None` / zero
/// because the turn never reached the LLM.
fn empty_turn_summary(
    thread_id: &ThreadId,
    turn: usize,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
) -> TurnSummary {
    TurnSummary {
        thread_id: thread_id.clone(),
        turn,
        total_turns: 0,
        turn_usage: TokenUsage::default(),
        total_usage: TokenUsage::default(),
        provenance: provenance.clone(),
        response_id: None,
        stop_reason: None,
        tool_call_count: 0,
        duration_ms: 0,
        tool_runtime: turn_options.tool_runtime.clone(),
        strict_durability: turn_options.strict_durability,
    }
}

fn cancelled_turn_outcome(
    thread_id: &ThreadId,
    turn: usize,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
) -> TurnOutcome {
    TurnOutcome::Cancelled {
        total_turns: 0,
        total_usage: TokenUsage::default(),
        summary: empty_turn_summary(thread_id, turn, provenance, turn_options),
    }
}

/// Handle a single-turn run cancelled before any work started: record
/// the root event, emit the terminal `Cancelled` event so streaming
/// consumers see a closing marker, and return the cancelled outcome.
async fn precheck_single_turn_cancelled<H>(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
) -> TurnOutcome
where
    H: AgentHooks,
{
    log::info!("Agent turn cancelled before execution started");
    #[cfg(feature = "otel")]
    crate::observability::instrument::record_root_event(
        "agent.cancelled",
        vec![crate::observability::attrs::kv(
            crate::observability::attrs::SDK_CANCEL_REASON,
            "cancel_token",
        )],
    );
    let _ = send_event(
        event_store,
        thread_id,
        0,
        hooks,
        authority,
        AgentEvent::cancelled(0, TokenUsage::default()),
    )
    .await;
    cancelled_turn_outcome(thread_id, 0, provenance, turn_options)
}

const fn turn_outcome_keeps_turn_open(outcome: &TurnOutcome) -> bool {
    matches!(outcome, TurnOutcome::AwaitingConfirmation { .. })
}

fn done_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Done {
        total_turns: turns_to_u32(ctx.turn),
        total_usage: ctx.total_usage.clone(),
    }
}

fn cancelled_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Cancelled {
        total_turns: turns_to_u32(ctx.turn),
        total_usage: ctx.total_usage.clone(),
    }
}

/// Emit the terminal [`AgentEvent::Cancelled`] so a streaming consumer
/// receives a closing marker and does not hang waiting for `Done`.
/// Mirrors the `Done` / `Refusal` emission shape.
///
/// `event_turn` is the **storage** turn the event is keyed under — it
/// must not be a turn that was already closed via `finish_turn`, or the
/// append is rejected. The `Cancelled` payload itself always reports
/// `ctx.turn` (the last turn the run actually reached).
async fn emit_cancelled_event<H>(
    ctx: &TurnContext,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    event_turn: usize,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    send_event(
        event_store,
        &ctx.thread_id,
        event_turn,
        hooks,
        authority,
        AgentEvent::cancelled(ctx.turn, ctx.total_usage.clone()),
    )
    .await
}

fn refusal_run_state(ctx: &TurnContext) -> AgentRunState {
    AgentRunState::Refusal {
        total_turns: turns_to_u32(ctx.turn),
        total_usage: ctx.total_usage.clone(),
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
        () = cancel_token.cancelled() => {
            #[cfg(feature = "otel")]
            crate::observability::instrument::record_root_event(
                "agent.cancelled",
                vec![
                    crate::observability::attrs::kv(
                        crate::observability::attrs::SDK_CANCEL_REASON,
                        "cancel_token",
                    ),
                    crate::observability::attrs::kv_i64(
                        crate::observability::attrs::SDK_TURN_NUMBER,
                        i64::try_from(ctx.turn).unwrap_or(0),
                    ),
                ],
            );
            // `current_turn` was just closed by
            // `emit_persistent_turn_complete`; key the terminal event
            // under the next turn so the append is accepted.
            let event_turn = current_turn.saturating_add(1);
            if let Err(error) =
                emit_cancelled_event(ctx, event_store, hooks, authority, event_turn).await
            {
                return Some(AgentRunState::Error(error));
            }
            let _ = finish_turn_or_error(event_store, &ctx.thread_id, event_turn).await;
            Some(cancelled_run_state(ctx))
        }
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
        InternalTurnResult::Cancelled { .. } => {
            // The LLM call or compaction was cancelled mid-turn. The
            // turn is still open (begin_turn started it, no finish has
            // run), so emit the terminal `Cancelled` event under
            // `current_turn`. History is already balanced (no orphan
            // tool_use), then close the turn and end the run.
            if let Err(error) =
                emit_cancelled_event(ctx, event_store, hooks, authority, current_turn).await
            {
                return RunLoopTurnAction::Return(AgentRunState::Error(error));
            }
            finish_turn_or_run_state(event_store, &ctx.thread_id, current_turn)
                .await
                .map_or_else(std::convert::identity, |()| {
                    RunLoopTurnAction::Return(cancelled_run_state(ctx))
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
        total_usage: ctx.total_usage.clone(),
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
            #[cfg(feature = "otel")]
            crate::observability::instrument::record_root_event(
                "agent.cancelled",
                vec![
                    crate::observability::attrs::kv(
                        crate::observability::attrs::SDK_CANCEL_REASON,
                        "cancel_token",
                    ),
                    crate::observability::attrs::kv_i64(
                        crate::observability::attrs::SDK_TURN_NUMBER,
                        i64::try_from(ctx.turn).unwrap_or(0),
                    ),
                ],
            );
            // The previous turn (`ctx.turn`) was already finished by
            // the in-loop result handler, so key the terminal event
            // under the next (never-started) turn to avoid appending to
            // a closed turn, then close it.
            let event_turn = ctx.turn.saturating_add(1);
            if let Err(error) =
                emit_cancelled_event(ctx, event_store, hooks, authority, event_turn).await
            {
                return Some(AgentRunState::Error(error));
            }
            let _ = finish_turn_or_error(event_store, &ctx.thread_id, event_turn).await;
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
            cancel_token,
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
        turn_options,
        start_time,
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
        Ok(ResumeProcessingResult::Completed {
            turn_usage,
            metrics,
        }) => {
            let mut updated_state = state;
            updated_state.turn_count = turn;
            if let Err(error) = state_store.save(&updated_state).await {
                warn!("Failed to save state checkpoint: {error}");
            }
            // Build the summary from real data threaded through
            // `process_resume` — the metrics describe the pre-pause
            // LLM call that produced this turn's tool calls, so the
            // resume-side summary matches the pre-pause summary for
            // the same turn.
            let summary = build_turn_summary_from_parts(TurnSummaryParts {
                thread_id: &thread_id,
                turn,
                turn_usage: turn_usage.clone(),
                total_usage: &total_usage,
                provenance: &provenance,
                response_id: metrics.response_id.as_deref(),
                stop_reason: metrics.stop_reason,
                tool_call_count: metrics.tool_call_count,
                start_time,
                turn_options: &turn_options,
            });
            TurnOutcome::NeedsMoreTurns {
                turn,
                turn_usage,
                total_usage,
                summary,
            }
        }
        Ok(ResumeProcessingResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        }) => {
            let turn_usage = continuation.turn_usage.clone();
            // The nested continuation carries the same pre-pause LLM
            // metadata that the resume read from the incoming
            // continuation, so the summary stays consistent across
            // every `AwaitingConfirmation` hop within this turn.
            let summary = build_turn_summary_from_parts(TurnSummaryParts {
                thread_id: &thread_id,
                turn,
                turn_usage,
                total_usage: &total_usage,
                provenance: &provenance,
                response_id: continuation.response_id.as_deref(),
                stop_reason: continuation.stop_reason,
                tool_call_count: continuation.pending_tool_calls.len(),
                start_time,
                turn_options: &turn_options,
            });
            TurnOutcome::AwaitingConfirmation {
                tool_call_id,
                tool_name,
                display_name,
                input,
                description,
                continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
                summary,
            }
        }
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
    let started = crate::observability::instrument::start_root_span(
        &crate::observability::instrument::StartRootSpanParams {
            provider: params.provider.as_ref(),
            tools: &params.tools,
            config: &params.config,
            thread_id: &params.thread_id,
            input: &params.input,
            run_mode: "loop",
            run_options: &params.run_options,
        },
    );
    #[cfg(feature = "otel")]
    let trace_state = crate::observability::instrument::build_root_trace_state(
        started.is_recording,
        &params.run_options,
    );
    #[cfg(feature = "otel")]
    let root_context = {
        let cx = crate::observability::instrument::build_root_context(
            started.span_context.clone(),
            &params.run_options,
        );
        let cx =
            crate::observability::instrument::attach_root_event_sink(&cx, started.sink.clone());
        match trace_state.clone() {
            Some(state) => state.attach_to(&cx),
            None => cx,
        }
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
        use crate::observability::instrument::{
            end_root_span, flush_root_trace_state, run_state_outcome,
        };

        let (turns, total_usage) = match &result {
            AgentRunState::Done {
                total_turns,
                total_usage,
            }
            | AgentRunState::Refusal {
                total_turns,
                total_usage,
            } => (usize::try_from(*total_turns).unwrap_or(0), total_usage),
            _ => {
                static EMPTY: TokenUsage = TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                };
                (0, &EMPTY)
            }
        };
        if let Some(state) = trace_state {
            flush_root_trace_state(&started.sink, state.as_ref());
        }
        end_root_span(started.sink, turns, total_usage, run_state_outcome(&result));
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
            run_options: _,
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
    let tool_context =
        apply_tool_boundary_controls(tool_context, &cancel_token, config.tool_timeout_ms);
    let provenance =
        agent_sdk_core::audit::AuditProvenance::new(provider.provider(), provider.model());
    let start_time = Instant::now();
    #[cfg(feature = "otel")]
    let input_kind = crate::observability::attrs::input_kind_str(&input);
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

    let mut ctx = build_turn_context(
        &thread_id,
        turn,
        total_usage,
        state,
        start_time,
        #[cfg(feature = "otel")]
        input_kind,
    );

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
    let started = crate::observability::instrument::start_root_span(
        &crate::observability::instrument::StartRootSpanParams {
            provider: params.provider.as_ref(),
            tools: &params.tools,
            config: &params.config,
            thread_id: &params.thread_id,
            input: &params.input,
            run_mode: "single_turn",
            run_options: &params.run_options,
        },
    );
    #[cfg(feature = "otel")]
    let trace_state = crate::observability::instrument::build_root_trace_state(
        started.is_recording,
        &params.run_options,
    );
    #[cfg(feature = "otel")]
    let root_context = {
        let cx = crate::observability::instrument::build_root_context(
            started.span_context.clone(),
            &params.run_options,
        );
        let cx =
            crate::observability::instrument::attach_root_event_sink(&cx, started.sink.clone());
        match trace_state.clone() {
            Some(state) => state.attach_to(&cx),
            None => cx,
        }
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
        use crate::observability::instrument::{
            end_root_span, flush_root_trace_state, turn_outcome_str,
        };

        let (turns, total_usage) = match &outcome {
            TurnOutcome::Done {
                total_turns,
                total_usage,
                ..
            }
            | TurnOutcome::Refusal {
                total_turns,
                total_usage,
                ..
            }
            | TurnOutcome::Cancelled {
                total_turns,
                total_usage,
                ..
            } => (usize::try_from(*total_turns).unwrap_or(0), total_usage),
            TurnOutcome::NeedsMoreTurns {
                turn, total_usage, ..
            } => (*turn, total_usage),
            _ => {
                static EMPTY: TokenUsage = TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                };
                (0, &EMPTY)
            }
        };
        if let Some(state) = trace_state {
            flush_root_trace_state(&started.sink, state.as_ref());
        }
        end_root_span(started.sink, turns, total_usage, turn_outcome_str(&outcome));
    }

    outcome
}

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
            run_options: _,
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
    // Build provenance early so we can include it in the summary even
    // when the turn is cancelled before the first LLM call.
    let provenance =
        agent_sdk_core::audit::AuditProvenance::new(provider.provider(), provider.model());

    // Check for cancellation before starting any work.
    if cancel_token.is_cancelled() {
        return precheck_single_turn_cancelled(
            &event_store,
            &thread_id,
            &hooks,
            &authority,
            &provenance,
            &turn_options,
        )
        .await;
    }

    let tool_context =
        apply_tool_boundary_controls(tool_context, &cancel_token, config.tool_timeout_ms);
    let start_time = Instant::now();
    #[cfg(feature = "otel")]
    let input_kind = crate::observability::attrs::input_kind_str(&input);
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
            turn_options: turn_options.clone(),
            start_time,
        })
        .await;
    }

    run_single_turn_execute(SingleTurnExecuteParams {
        event_store,
        authority,
        thread_id,
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
        provenance,
        turn_options,
        cancel_token,
        turn,
        total_usage,
        state,
        start_time,
        #[cfg(feature = "otel")]
        input_kind,
        #[cfg(feature = "otel")]
        observability_store,
    })
    .await
}

/// Parameters for the non-resume single-turn execution path.
///
/// Split out of `run_single_turn_inner` so the top-level function stays
/// under the clippy too-many-lines threshold. The resume path never
/// hits this function — it branches earlier via
/// `handle_single_turn_resume_state`.
struct SingleTurnExecuteParams<Ctx, P, H, M, S> {
    event_store: Arc<dyn EventStore>,
    authority: Arc<dyn EventAuthority>,
    thread_id: ThreadId,
    tool_context: crate::tools::ToolContext<Ctx>,
    provider: Arc<P>,
    tools: Arc<crate::tools::ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    message_store: Arc<M>,
    state_store: Arc<S>,
    config: AgentConfig,
    compaction_config: Option<CompactionConfig>,
    compactor: Option<Arc<dyn ContextCompactor>>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
    audit_sink: Arc<dyn crate::hooks::ToolAuditSink>,
    provenance: agent_sdk_core::audit::AuditProvenance,
    turn_options: TurnOptions,
    cancel_token: CancellationToken,
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    start_time: Instant,
    #[cfg(feature = "otel")]
    input_kind: &'static str,
    #[cfg(feature = "otel")]
    observability_store: Option<Arc<dyn crate::observability::ObservabilityStore>>,
}

async fn run_single_turn_execute<Ctx, P, H, M, S>(
    SingleTurnExecuteParams {
        event_store,
        authority,
        thread_id,
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
        provenance,
        turn_options,
        cancel_token,
        turn,
        total_usage,
        state,
        start_time,
        #[cfg(feature = "otel")]
        input_kind,
        #[cfg(feature = "otel")]
        observability_store,
    }: SingleTurnExecuteParams<Ctx, P, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let mut ctx = build_turn_context(
        &thread_id,
        turn,
        total_usage,
        state,
        start_time,
        #[cfg(feature = "otel")]
        input_kind,
    );

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
        cancel_token: &cancel_token,
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
        provenance: &provenance,
        turn_options: &turn_options,
    })
    .await;

    if !turn_outcome_keeps_turn_open(&outcome)
        && let Err(store_error) = finish_turn_or_error(&event_store, &thread_id, current_turn).await
    {
        return TurnOutcome::Error(store_error);
    }

    outcome
}

/// Build the single-turn `Cancelled` outcome: emit the terminal
/// `Cancelled` event under the still-open turn (`ctx.turn`, started by
/// `begin_turn` and never finished in single-turn mode), then return the
/// cancelled `TurnOutcome`.
async fn convert_cancelled_turn<H>(
    ctx: &TurnContext,
    event_store: &Arc<dyn EventStore>,
    hooks: &Arc<H>,
    authority: &Arc<dyn EventAuthority>,
    provenance: &AuditProvenance,
    turn_options: &TurnOptions,
    turn_usage: TokenUsage,
) -> TurnOutcome
where
    H: AgentHooks,
{
    if let Err(error) = emit_cancelled_event(ctx, event_store, hooks, authority, ctx.turn).await {
        return TurnOutcome::Error(error);
    }
    let summary = build_turn_summary(ctx, provenance, turn_options, turn_usage);
    TurnOutcome::Cancelled {
        total_turns: turns_to_u32(ctx.turn),
        total_usage: ctx.total_usage.clone(),
        summary,
    }
}

struct ConvertDoneParams<'a, H, S> {
    ctx: &'a TurnContext,
    state_store: &'a Arc<S>,
    event_store: &'a Arc<dyn EventStore>,
    hooks: &'a Arc<H>,
    authority: &'a Arc<dyn EventAuthority>,
    thread_id: &'a ThreadId,
    current_turn: usize,
    provenance: &'a AuditProvenance,
    turn_options: &'a TurnOptions,
}

/// Build the `Done` outcome: persist final state, emit the terminal
/// `Done` event, and report cumulative usage as the summary's turn
/// usage. Extracted from `convert_turn_result` to keep it under the
/// clippy line ceiling.
async fn convert_done_turn<H, S>(params: ConvertDoneParams<'_, H, S>) -> TurnOutcome
where
    H: AgentHooks,
    S: StateStore,
{
    let ConvertDoneParams {
        ctx,
        state_store,
        event_store,
        hooks,
        authority,
        thread_id,
        current_turn,
        provenance,
        turn_options,
    } = params;
    if let Err(e) = state_store.save(&ctx.state).await {
        warn!("Failed to save final state: {e}");
    }
    let duration = ctx.start_time.elapsed();
    if let Err(error) = send_event(
        event_store,
        thread_id,
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
    let summary = build_turn_summary(ctx, provenance, turn_options, ctx.total_usage.clone());
    TurnOutcome::Done {
        total_turns: turns_to_u32(ctx.turn),
        total_usage: ctx.total_usage.clone(),
        summary,
    }
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
        provenance,
        turn_options,
    }: ConvertTurnResultParams<'_, H, S>,
) -> TurnOutcome {
    match result {
        InternalTurnResult::Continue { turn_usage } => {
            if let Err(e) = state_store.save(&ctx.state).await {
                warn!("Failed to save state checkpoint: {e}");
            }
            let summary = build_turn_summary(&ctx, provenance, turn_options, turn_usage.clone());
            TurnOutcome::NeedsMoreTurns {
                turn: ctx.turn,
                turn_usage,
                total_usage: ctx.total_usage,
                summary,
            }
        }
        InternalTurnResult::Done => {
            convert_done_turn(ConvertDoneParams {
                ctx: &ctx,
                state_store,
                event_store,
                hooks,
                authority,
                thread_id: &thread_id,
                current_turn,
                provenance,
                turn_options,
            })
            .await
        }
        InternalTurnResult::Refusal => {
            let summary =
                build_turn_summary(&ctx, provenance, turn_options, ctx.total_usage.clone());
            TurnOutcome::Refusal {
                total_turns: turns_to_u32(ctx.turn),
                total_usage: ctx.total_usage.clone(),
                summary,
            }
        }
        InternalTurnResult::Cancelled { turn_usage } => {
            convert_cancelled_turn(
                &ctx,
                event_store,
                hooks,
                authority,
                provenance,
                turn_options,
                turn_usage,
            )
            .await
        }
        InternalTurnResult::AwaitingConfirmation {
            tool_call_id,
            tool_name,
            display_name,
            input,
            description,
            continuation,
        } => {
            let turn_usage = continuation.turn_usage.clone();
            let summary = build_turn_summary(&ctx, provenance, turn_options, turn_usage);
            TurnOutcome::AwaitingConfirmation {
                tool_call_id,
                tool_name,
                display_name,
                input,
                description,
                continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
                summary,
            }
        }
        InternalTurnResult::PendingToolCalls {
            turn_usage,
            pending_tool_calls,
            continuation,
        } => {
            let summary = build_turn_summary(&ctx, provenance, turn_options, turn_usage.clone());
            TurnOutcome::PendingToolCalls {
                turn: ctx.turn,
                turn_usage,
                total_usage: ctx.total_usage,
                tool_calls: pending_tool_calls,
                continuation: Box::new(ContinuationEnvelope::wrap(*continuation)),
                summary,
            }
        }
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
                tier: crate::types::ToolTier::Observe,
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
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
                    tier: crate::types::ToolTier::Observe,
                    input: serde_json::json!({}),
                    effective_input: serde_json::json!({}),
                    listen_context: None,
                },
                PendingToolCallInfo {
                    id: "call_2".into(),
                    name: "write".into(),
                    display_name: "Write".into(),
                    tier: crate::types::ToolTier::Confirm,
                    input: serde_json::json!({}),
                    effective_input: serde_json::json!({}),
                    listen_context: None,
                },
            ],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
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
                tier: crate::types::ToolTier::Observe,
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
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
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        };

        let err = validate_external_tool_results(&cont, &[]);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("no pending tool calls"), "Error: {msg}");
    }
}
