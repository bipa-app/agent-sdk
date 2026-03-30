use super::helpers::{pending_tool_index, send_event, turns_to_u32};
use super::tool_execution::{append_tool_results, execute_confirmed_tool, execute_tool_call};
use super::turn::execute_turn;
use super::types::{
    ConfirmedToolExecutionContext, ExecuteTurnParameters, InitializedState, InternalTurnResult,
    ResumeData, ResumeProcessingParameters, ResumeProcessingResult, RunLoopParameters,
    RunLoopResumeParams, RunLoopTurnsParams, SingleTurnResumeParams, ToolCallExecutionContext,
    ToolExecutionOutcome, TurnContext, TurnParameters,
};

use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::{Content, ContentBlock, LlmProvider, Message, Role};
use crate::stores::{MessageStore, StateStore};
use crate::types::{
    AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState, ThreadId, TokenUsage,
    TurnOutcome,
};
use log::warn;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Initialize agent state from the given input.
///
/// Handles the three input variants:
/// - `Text`: Creates/loads state, appends user message
/// - `Resume`: Restores from continuation state
/// - `Continue`: Loads existing state to continue execution
pub(super) async fn initialize_from_input<M, S>(
    input: AgentInput,
    thread_id: &ThreadId,
    message_store: &Arc<M>,
    state_store: &Arc<S>,
) -> Result<InitializedState, AgentError>
where
    M: MessageStore,
    S: StateStore,
{
    match input {
        AgentInput::Text(user_message) => {
            // Load or create state
            let state = match state_store.load(thread_id).await {
                Ok(Some(s)) => s,
                Ok(None) => AgentState::new(thread_id.clone()),
                Err(e) => {
                    return Err(AgentError::new(format!("Failed to load state: {e}"), false));
                }
            };

            // Recover from orphaned tool_use blocks (crash between persist and execution)
            recover_orphaned_tool_use(thread_id, message_store).await?;

            // Add user message to history
            let user_msg = Message::user(&user_message);
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
        AgentInput::Message(blocks) => {
            let state = match state_store.load(thread_id).await {
                Ok(Some(s)) => s,
                Ok(None) => AgentState::new(thread_id.clone()),
                Err(e) => {
                    return Err(AgentError::new(format!("Failed to load state: {e}"), false));
                }
            };

            let user_msg = Message::user_with_content(blocks);
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
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed,
            rejection_reason,
        } => {
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
        AgentInput::Continue => {
            // Load existing state to continue execution
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

            // Recover from orphaned tool_use blocks (crash between persist and execution)
            recover_orphaned_tool_use(thread_id, message_store).await?;

            // Continue from where we left off
            Ok(InitializedState {
                turn: state.turn_count,
                total_usage: state.total_usage.clone(),
                state,
                resume_data: None,
            })
        }
    }
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
        tx,
        seq,
        message_store,
        execution_store,
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
        tx,
        seq,
        execution_store,
    };
    let result = execute_confirmed_tool(awaiting_tool, rejection, &confirmed_ctx).await;
    tool_results.push((awaiting_tool.id.clone(), result));

    let execution_ctx = ToolCallExecutionContext {
        tool_context,
        thread_id,
        tools,
        hooks,
        tx,
        seq,
        execution_store,
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
        }
    }

    append_tool_results(&tool_results, thread_id, message_store).await?;
    send_event(
        tx,
        hooks,
        seq,
        AgentEvent::TurnComplete {
            turn,
            usage: cont.turn_usage.clone(),
        },
    )
    .await;

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
        tx,
        seq,
        message_store,
        execution_store,
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
        tx,
        seq,
        message_store,
        execution_store,
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
            continuation,
        })),
    }
}

#[allow(clippy::too_many_lines)]
pub(super) async fn run_loop_turns<Ctx, P, H, M, S>(
    RunLoopTurnsParams {
        ctx,
        tool_context,
        provider,
        tools,
        hooks,
        message_store,
        state_store,
        tx,
        seq,
        config,
        compaction_config,
        compactor,
        execution_store,
        cancel_token,
        mut input_rx,
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
        // Check for cancellation before starting a new turn
        if cancel_token.is_cancelled() {
            log::info!("Agent run cancelled before turn {}", ctx.turn);
            return Some(AgentRunState::Cancelled {
                total_turns: turns_to_u32(ctx.turn),
                input_tokens: u64::from(ctx.total_usage.input_tokens),
                output_tokens: u64::from(ctx.total_usage.output_tokens),
            });
        }

        let result = execute_turn(ExecuteTurnParameters {
            tx,
            seq,
            ctx,
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
        })
        .await;

        match result {
            InternalTurnResult::Continue { .. } => {
                if let Err(error) = state_store.save(&ctx.state).await {
                    warn!("Failed to save state checkpoint: {error}");
                }
            }
            InternalTurnResult::Done => {
                // Persistent mode: wait for new input instead of exiting
                if let Some(ref mut rx) = input_rx {
                    // Emit a TurnComplete event so the caller knows we're idle
                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::TurnComplete {
                            turn: ctx.turn,
                            usage: ctx.total_usage.clone(),
                        },
                    )
                    .await;

                    // Wait for new input or cancellation
                    tokio::select! {
                        msg = rx.recv() => {
                            match msg {
                                Some(AgentInput::Text(text)) => {
                                    let user_msg = Message::user(&text);
                                    if let Err(e) = message_store.append(&ctx.thread_id, user_msg).await {
                                        warn!("Failed to append injected message: {e}");
                                        return None;
                                    }
                                    ctx.turn += 1;
                                    // Continue the loop for the next turn
                                }
                                Some(AgentInput::Message(blocks)) => {
                                    let user_msg = Message::user_with_content(blocks);
                                    if let Err(e) = message_store.append(&ctx.thread_id, user_msg).await {
                                        warn!("Failed to append injected message: {e}");
                                        return None;
                                    }
                                    ctx.turn += 1;
                                }
                                _ => return None, // Channel closed or unsupported input
                            }
                        }
                        () = cancel_token.cancelled() => {
                            return Some(AgentRunState::Cancelled {
                                total_turns: turns_to_u32(ctx.turn),
                                input_tokens: u64::from(ctx.total_usage.input_tokens),
                                output_tokens: u64::from(ctx.total_usage.output_tokens),
                            });
                        }
                    }
                } else {
                    return None; // Non-persistent mode: exit as before
                }
            }
            InternalTurnResult::Refusal => {
                return Some(AgentRunState::Refusal {
                    total_turns: turns_to_u32(ctx.turn),
                    input_tokens: u64::from(ctx.total_usage.input_tokens),
                    output_tokens: u64::from(ctx.total_usage.output_tokens),
                });
            }
            InternalTurnResult::AwaitingConfirmation {
                tool_call_id,
                tool_name,
                display_name,
                input,
                description,
                continuation,
            } => {
                return Some(AgentRunState::AwaitingConfirmation {
                    tool_call_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                    continuation,
                });
            }
            InternalTurnResult::Error(error) => return Some(AgentRunState::Error(error)),
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
        tx,
        seq,
        message_store,
        state_store,
        execution_store,
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
        tx: &tx,
        seq: &seq,
        message_store: &message_store,
        execution_store: execution_store.as_ref(),
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
            continuation,
        },
        Err(error) => {
            send_event(
                &tx,
                &hooks,
                &seq,
                AgentEvent::error(&error.message, error.recoverable),
            )
            .await;
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
        tx,
        seq,
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
        cancel_token,
        mut input_rx,
        #[cfg(feature = "otel")]
            observability_store: _observability_store,
    }: RunLoopParameters<Ctx, P, H, M, S>,
) -> AgentRunState
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let tool_context = tool_context
        .with_event_tx(tx.clone(), seq.clone())
        .with_cancel_token(cancel_token.clone());
    let start_time = Instant::now();
    let init_state =
        match initialize_from_input(input, &thread_id, &message_store, &state_store).await {
            Ok(state) => state,
            Err(error) => return AgentRunState::Error(error),
        };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume_data) = resume_data {
        let resume_result = handle_run_loop_resume(RunLoopResumeParams {
            resume_data,
            turn,
            total_usage: &total_usage,
            state: &state,
            thread_id: &thread_id,
            tool_context: &tool_context,
            tools: &tools,
            hooks: &hooks,
            tx: &tx,
            seq: &seq,
            message_store: &message_store,
            execution_store: execution_store.as_ref(),
        })
        .await;

        match resume_result {
            Ok(Some(outcome)) => return outcome,
            Ok(None) => {}
            Err(error) => {
                send_event(
                    &tx,
                    &hooks,
                    &seq,
                    AgentEvent::error(&error.message, error.recoverable),
                )
                .await;
                return AgentRunState::Error(error);
            }
        }
    }

    let mut ctx = TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
        compaction_retries: 0,
    };

    if let Some(outcome) = run_loop_turns(RunLoopTurnsParams {
        ctx: &mut ctx,
        tool_context: &tool_context,
        provider: &provider,
        tools: &tools,
        hooks: &hooks,
        message_store: &message_store,
        state_store: &state_store,
        tx: &tx,
        seq: &seq,
        config: &config,
        compaction_config: compaction_config.as_ref(),
        compactor: compactor.as_ref(),
        execution_store: execution_store.as_ref(),
        cancel_token: &cancel_token,
        input_rx: input_rx.as_mut(),
    })
    .await
    {
        return outcome;
    }

    if let Err(e) = state_store.save(&ctx.state).await {
        warn!("Failed to save final state: {e}");
    }

    let duration = ctx.start_time.elapsed();
    send_event(
        &tx,
        &hooks,
        &seq,
        AgentEvent::done(thread_id, ctx.turn, ctx.total_usage.clone(), duration),
    )
    .await;

    AgentRunState::Done {
        total_turns: turns_to_u32(ctx.turn),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
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

async fn run_single_turn_inner<Ctx, P, H, M, S>(
    TurnParameters {
        tx,
        seq,
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
        cancel_token,
        #[cfg(feature = "otel")]
            observability_store: _observability_store,
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
        return TurnOutcome::Cancelled {
            total_turns: 0,
            input_tokens: 0,
            output_tokens: 0,
        };
    }

    let tool_context = tool_context
        .with_event_tx(tx.clone(), seq.clone())
        .with_cancel_token(cancel_token.clone());
    let start_time = Instant::now();
    let init_state =
        match initialize_from_input(input, &thread_id, &message_store, &state_store).await {
            Ok(state) => state,
            Err(error) => {
                send_event(
                    &tx,
                    &hooks,
                    &seq,
                    AgentEvent::error(&error.message, error.recoverable),
                )
                .await;
                return TurnOutcome::Error(error);
            }
        };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume_data) = resume_data {
        return handle_single_turn_resume(SingleTurnResumeParams {
            resume_data,
            turn,
            total_usage,
            state,
            thread_id,
            tool_context,
            tools,
            hooks,
            tx,
            seq,
            message_store,
            state_store,
            execution_store,
        })
        .await;
    }

    let mut ctx = TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
        compaction_retries: 0,
    };

    let result = execute_turn(ExecuteTurnParameters {
        tx: &tx,
        seq: &seq,
        ctx: &mut ctx,
        tool_context: &tool_context,
        provider: &provider,
        tools: &tools,
        hooks: &hooks,
        message_store: &message_store,
        state_store: &state_store,
        config: &config,
        compaction_config: compaction_config.as_ref(),
        compactor: compactor.as_ref(),
        execution_store: execution_store.as_ref(),
    })
    .await;

    convert_turn_result(result, ctx, &tx, &hooks, &seq, thread_id, &state_store).await
}

pub(super) async fn convert_turn_result<H: AgentHooks, S: StateStore>(
    result: InternalTurnResult,
    ctx: TurnContext,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    thread_id: ThreadId,
    state_store: &Arc<S>,
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
            send_event(
                tx,
                hooks,
                seq,
                AgentEvent::done(thread_id, ctx.turn, ctx.total_usage.clone(), duration),
            )
            .await;
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
            continuation,
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
}
