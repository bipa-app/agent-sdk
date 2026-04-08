//! Subagent support for spawning child agents.
//!
//! This module provides the ability to spawn subagents from within an agent.
//! Subagents are isolated agent instances that run to completion and return
//! only their final response to the parent agent.
//!
//! # Overview
//!
//! Subagents are useful for:
//! - Delegating complex subtasks to specialized agents
//! - Running parallel investigations
//! - Isolating context for specific operations
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::subagent::{SubagentTool, SubagentConfig};
//!
//! let config = SubagentConfig::new("researcher")
//!     .with_system_prompt("You are a research specialist...")
//!     .with_max_turns(10);
//!
//! let tool = SubagentTool::new(config, provider, tools, || {
//!     std::sync::Arc::new(agent_sdk::InMemoryEventStore::new())
//! });
//! registry.register(tool);
//! ```
//!
//! # Behavior
//!
//! When a subagent runs:
//! 1. A new isolated thread is created
//! 2. The subagent runs until completion or max turns
//! 3. Only the final text response is returned to the parent
//! 4. The parent does not see the subagent's intermediate tool calls

mod factory;

pub use factory::SubagentFactory;

use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, DefaultHooks};
use crate::llm::LlmProvider;
use crate::stores::{EventStore, InMemoryStore, MessageStore, StateStore};
use crate::tools::{DynamicToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, ThreadId, TokenUsage, ToolResult, ToolTier};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Metadata key for tracking the current subagent nesting depth.
///
/// When a subagent spawns another subagent, the depth is incremented.
/// Tools check this value against the configured maximum depth.
pub const METADATA_SUBAGENT_DEPTH: &str = "subagent_depth";

/// Metadata key for the maximum allowed subagent nesting depth.
///
/// Set by the host application (e.g. bip) to prevent unbounded recursion.
pub const METADATA_MAX_SUBAGENT_DEPTH: &str = "max_subagent_depth";

/// Configuration for a subagent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Name of the subagent (for identification).
    pub name: String,
    /// Human-friendly nickname assigned by the parent (e.g., "Zara").
    pub nickname: Option<String>,
    /// System prompt for the subagent.
    pub system_prompt: String,
    /// Maximum number of turns before stopping.
    pub max_turns: Option<usize>,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Optional model override for this subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SubagentConfig {
    /// Create a new subagent configuration.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nickname: None,
            system_prompt: String::new(),
            max_turns: None,
            timeout_ms: None,
            model: None,
        }
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the maximum number of turns.
    #[must_use]
    pub const fn with_max_turns(mut self, max: usize) -> Self {
        self.max_turns = Some(max);
        self
    }

    /// Set the timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = Some(timeout);
        self
    }

    /// Set the model override for this subagent.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set a human-friendly nickname for this subagent.
    #[must_use]
    pub fn with_nickname(mut self, nickname: impl Into<String>) -> Self {
        self.nickname = Some(nickname.into());
        self
    }
}

/// Log entry for a single tool call within a subagent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallLog {
    /// Tool name.
    pub name: String,
    /// Tool display name.
    pub display_name: String,
    /// Brief context/args (e.g., file path, command).
    pub context: String,
    /// Brief result summary.
    pub result: String,
    /// Whether the tool call succeeded.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: Option<u64>,
}

/// Result from a subagent execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentResult {
    /// Name of the subagent.
    pub name: String,
    /// The final text response (only visible part to parent).
    pub final_response: String,
    /// Total number of turns taken.
    pub total_turns: usize,
    /// Number of tool calls made by the subagent.
    pub tool_count: u32,
    /// Log of tool calls made by the subagent.
    pub tool_logs: Vec<ToolCallLog>,
    /// Token usage statistics.
    pub usage: TokenUsage,
    /// Whether the subagent completed successfully.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Detailed error information when `success` is false.
    ///
    /// Contains the raw error message from the agent event, which may include
    /// stack trace information or structured error context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_details: Option<String>,
    /// Retained for serialization compatibility with older clients.
    ///
    /// The previous implementation inferred this from the "last pending tool"
    /// when any generic error occurred, which was incorrect for LLM or
    /// transport failures. The field is currently never populated until the
    /// SDK has deterministic error provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_tool: Option<String>,
}

/// Tool for spawning subagents.
///
/// This tool allows an agent to spawn a child agent that runs independently
/// and returns only its final response.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::subagent::{SubagentTool, SubagentConfig};
///
/// let config = SubagentConfig::new("analyzer")
///     .with_system_prompt("You analyze code...");
///
/// let tool = SubagentTool::new(config, provider.clone(), tools.clone(), || {
///     std::sync::Arc::new(agent_sdk::InMemoryEventStore::new())
/// });
/// ```
pub struct SubagentTool<P, H = DefaultHooks, M = InMemoryStore, S = InMemoryStore>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    config: SubagentConfig,
    provider: Arc<P>,
    tools: Arc<ToolRegistry<()>>,
    hooks: Arc<H>,
    message_store_factory: Arc<dyn Fn() -> M + Send + Sync>,
    state_store_factory: Arc<dyn Fn() -> S + Send + Sync>,
    event_store_factory: Arc<dyn Fn() -> Arc<dyn EventStore> + Send + Sync>,
    /// Cached display name to avoid `Box::leak` on every call.
    cached_display_name: &'static str,
    /// Cached description to avoid `Box::leak` on every call.
    cached_description: &'static str,
}

impl<P> SubagentTool<P, DefaultHooks, InMemoryStore, InMemoryStore>
where
    P: LlmProvider + 'static,
{
    /// Create a new subagent tool with default hooks and in-memory message/state stores.
    #[must_use]
    pub fn new<EF>(
        config: SubagentConfig,
        provider: Arc<P>,
        tools: Arc<ToolRegistry<()>>,
        event_store_factory: EF,
    ) -> Self
    where
        EF: Fn() -> Arc<dyn EventStore> + Send + Sync + 'static,
    {
        // Cache leaked strings at construction time (bounded by number of tools)
        let cached_display_name = Box::leak(format!("Subagent: {}", config.name).into_boxed_str());
        let cached_description = Box::leak(
            format!(
                "Spawn a subagent named '{}' to handle a task. The subagent will work independently and return only its final response.",
                config.name
            )
            .into_boxed_str(),
        );
        Self {
            config,
            provider,
            tools,
            hooks: Arc::new(DefaultHooks),
            message_store_factory: Arc::new(InMemoryStore::new),
            state_store_factory: Arc::new(InMemoryStore::new),
            event_store_factory: Arc::new(event_store_factory),
            cached_display_name,
            cached_description,
        }
    }
}

impl<P, H, M, S> SubagentTool<P, H, M, S>
where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    /// Create with custom hooks.
    #[must_use]
    pub fn with_hooks<H2: AgentHooks + Clone + 'static>(
        self,
        hooks: Arc<H2>,
    ) -> SubagentTool<P, H2, M, S> {
        SubagentTool {
            config: self.config,
            provider: self.provider,
            tools: self.tools,
            hooks,
            message_store_factory: self.message_store_factory,
            state_store_factory: self.state_store_factory,
            event_store_factory: self.event_store_factory,
            cached_display_name: self.cached_display_name,
            cached_description: self.cached_description,
        }
    }

    /// Create with custom store factories.
    #[must_use]
    pub fn with_stores<M2, S2, MF, SF>(
        self,
        message_factory: MF,
        state_factory: SF,
    ) -> SubagentTool<P, H, M2, S2>
    where
        M2: MessageStore + 'static,
        S2: StateStore + 'static,
        MF: Fn() -> M2 + Send + Sync + 'static,
        SF: Fn() -> S2 + Send + Sync + 'static,
    {
        SubagentTool {
            config: self.config,
            provider: self.provider,
            tools: self.tools,
            hooks: self.hooks,
            message_store_factory: Arc::new(message_factory),
            state_store_factory: Arc::new(state_factory),
            event_store_factory: self.event_store_factory,
            cached_display_name: self.cached_display_name,
            cached_description: self.cached_description,
        }
    }

    /// Get the subagent configuration.
    #[must_use]
    pub const fn config(&self) -> &SubagentConfig {
        &self.config
    }

    /// Run the subagent with a task.
    ///
    /// The `parent_cancel` token links the subagent's lifecycle to its parent.
    /// Cancelling the parent token will also cancel the subagent.
    async fn run_subagent(
        &self,
        task: &str,
        subagent_id: String,
        parent_ctx: &ToolContext<()>,
        parent_cancel: CancellationToken,
    ) -> Result<SubagentResult> {
        use crate::agent_loop::AgentLoop;

        let start = Instant::now();
        // Each subagent run gets its own thread id, so a shared event store
        // only returns this subagent's events when queried with `thread_id`.
        let thread_id = ThreadId::new();

        // Create stores for this subagent run
        let message_store = (self.message_store_factory)();
        let state_store = (self.state_store_factory)();
        let event_store = (self.event_store_factory)();

        // Create agent config with a default max_turns to prevent unbounded execution
        let agent_config = AgentConfig {
            max_turns: Some(self.config.max_turns.unwrap_or(100)),
            system_prompt: self.config.system_prompt.clone(),
            ..Default::default()
        };

        // Build the subagent
        let agent = AgentLoop::new(
            (*self.provider).clone(),
            (*self.tools).clone(),
            (*self.hooks).clone(),
            message_store,
            state_store,
            Arc::clone(&event_store),
            agent_config,
        );

        // Create tool context
        let tool_ctx = ToolContext::new(());

        // Run with a child cancellation token so parent cancellation propagates.
        // Use `run_abortable` so we can abort the spawned task on timeout
        // instead of leaving a detached task that continues making API calls.
        let cancel_token = parent_cancel.child_token();
        let timeout_cancel = cancel_token.clone();
        let (state_rx, task_handle) = agent.run_abortable(
            thread_id.clone(),
            AgentInput::Text(task.to_string()),
            tool_ctx,
            cancel_token,
        );

        let wait_result = wait_for_subagent_state(self.config.timeout_ms, start, state_rx).await;
        let mut state = SubagentExecutionState::new();
        let replay_events = apply_subagent_wait_outcome(
            classify_subagent_wait_result(wait_result.as_ref()),
            &self.config,
            &timeout_cancel,
            &task_handle,
            &mut state,
        );

        if replay_events {
            replay_subagent_events(
                &event_store,
                &thread_id,
                parent_ctx,
                &self.config,
                &subagent_id,
                &mut state,
            )
            .await?;
        }

        let result = state.into_result(self.config.name.clone(), start);
        emit_subagent_observability(self, &result);
        Ok(result)
    }
}

fn mark_subagent_timeout(
    config: &SubagentConfig,
    final_response: &mut String,
    error_details: &mut Option<String>,
    success: &mut bool,
) {
    *final_response = "Subagent timed out".to_string();
    *error_details = Some(format!(
        "Subagent '{}' timed out after {}ms",
        config.name,
        config.timeout_ms.unwrap_or(0)
    ));
    *success = false;
}

fn mark_subagent_disconnected(
    config: &SubagentConfig,
    final_response: &mut String,
    error_details: &mut Option<String>,
    success: &mut bool,
) {
    *final_response = "Subagent ended unexpectedly".to_string();
    *error_details = Some(format!(
        "Subagent '{}' ended before returning a final state",
        config.name
    ));
    *success = false;
}

fn mark_subagent_cancelled(
    config: &SubagentConfig,
    final_response: &mut String,
    error_details: &mut Option<String>,
    success: &mut bool,
) {
    *final_response = "Subagent cancelled".to_string();
    *error_details = Some(format!("Subagent '{}' was cancelled", config.name));
    *success = false;
}

fn mark_subagent_awaiting_confirmation(
    config: &SubagentConfig,
    final_response: &mut String,
    error_details: &mut Option<String>,
    success: &mut bool,
) {
    *final_response = "Subagent requires confirmation".to_string();
    *error_details = Some(format!(
        "Subagent '{}' requested confirmation, which is not supported in nested runs",
        config.name
    ));
    *success = false;
}

fn mark_subagent_agent_error(
    final_response: &mut String,
    error_details: &mut Option<String>,
    success: &mut bool,
    message: &str,
) {
    *final_response = message.to_string();
    *error_details = Some(message.to_string());
    *success = false;
}

type SubagentWaitResult = Result<
    Result<crate::types::AgentRunState, tokio::sync::oneshot::error::RecvError>,
    tokio::time::error::Elapsed,
>;

struct SubagentExecutionState {
    final_response: String,
    total_turns: usize,
    tool_count: u32,
    tool_logs: Vec<ToolCallLog>,
    pending_tools: HashMap<String, (String, String)>,
    total_usage: TokenUsage,
    success: bool,
    error_details: Option<String>,
    failed_tool: Option<String>,
}

impl SubagentExecutionState {
    fn new() -> Self {
        Self {
            final_response: String::new(),
            total_turns: 0,
            tool_count: 0,
            tool_logs: Vec::new(),
            pending_tools: HashMap::new(),
            total_usage: TokenUsage::default(),
            success: true,
            error_details: None,
            failed_tool: None,
        }
    }

    fn into_result(self, name: String, start: Instant) -> SubagentResult {
        SubagentResult {
            name,
            final_response: self.final_response,
            total_turns: self.total_turns,
            tool_count: self.tool_count,
            tool_logs: self.tool_logs,
            usage: self.total_usage,
            success: self.success,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            error_details: self.error_details,
            failed_tool: self.failed_tool,
        }
    }
}

fn subagent_total_tokens(total_usage: &TokenUsage) -> u64 {
    u64::from(total_usage.input_tokens) + u64::from(total_usage.output_tokens)
}

struct SubagentProgressUpdate<'a> {
    subagent_id: &'a str,
    total_turns: usize,
    total_usage: &'a TokenUsage,
    tool_name: String,
    tool_context: String,
    completed: bool,
    success: bool,
    tool_count: u32,
}

enum SubagentWaitOutcome {
    ReplayEvents,
    TimedOut,
    Disconnected,
    Cancelled,
    AwaitingConfirmation,
    Error(crate::types::AgentError),
}

async fn wait_for_subagent_state(
    timeout_ms: Option<u64>,
    start: Instant,
    state_rx: tokio::sync::oneshot::Receiver<crate::types::AgentRunState>,
) -> Option<SubagentWaitResult> {
    let timeout_duration = timeout_ms.map(Duration::from_millis);
    if timeout_duration.is_some_and(|timeout| timeout.saturating_sub(start.elapsed()).is_zero()) {
        return None;
    }
    if let Some(timeout) = timeout_duration {
        let remaining = timeout.saturating_sub(start.elapsed());
        Some(tokio::time::timeout(remaining, state_rx).await)
    } else {
        Some(Ok(state_rx.await))
    }
}

fn classify_subagent_wait_result(wait_result: Option<&SubagentWaitResult>) -> SubagentWaitOutcome {
    match wait_result {
        Some(Ok(Ok(
            crate::types::AgentRunState::Done { .. } | crate::types::AgentRunState::Refusal { .. },
        ))) => SubagentWaitOutcome::ReplayEvents,
        Some(Ok(Ok(crate::types::AgentRunState::Cancelled { .. }))) => {
            SubagentWaitOutcome::Cancelled
        }
        Some(Ok(Ok(crate::types::AgentRunState::AwaitingConfirmation { .. }))) => {
            SubagentWaitOutcome::AwaitingConfirmation
        }
        Some(Ok(Ok(crate::types::AgentRunState::Error(error)))) => {
            SubagentWaitOutcome::Error(error.clone())
        }
        Some(Ok(Err(_))) => SubagentWaitOutcome::Disconnected,
        None | Some(Err(_)) => SubagentWaitOutcome::TimedOut,
    }
}

fn apply_subagent_wait_outcome(
    outcome: SubagentWaitOutcome,
    config: &SubagentConfig,
    timeout_cancel: &CancellationToken,
    task_handle: &tokio::task::JoinHandle<()>,
    state: &mut SubagentExecutionState,
) -> bool {
    match outcome {
        SubagentWaitOutcome::ReplayEvents => true,
        SubagentWaitOutcome::TimedOut => {
            timeout_cancel.cancel();
            task_handle.abort();
            mark_subagent_timeout(
                config,
                &mut state.final_response,
                &mut state.error_details,
                &mut state.success,
            );
            false
        }
        SubagentWaitOutcome::Disconnected => {
            timeout_cancel.cancel();
            task_handle.abort();
            mark_subagent_disconnected(
                config,
                &mut state.final_response,
                &mut state.error_details,
                &mut state.success,
            );
            false
        }
        SubagentWaitOutcome::Cancelled => {
            timeout_cancel.cancel();
            task_handle.abort();
            mark_subagent_cancelled(
                config,
                &mut state.final_response,
                &mut state.error_details,
                &mut state.success,
            );
            false
        }
        SubagentWaitOutcome::AwaitingConfirmation => {
            timeout_cancel.cancel();
            task_handle.abort();
            mark_subagent_awaiting_confirmation(
                config,
                &mut state.final_response,
                &mut state.error_details,
                &mut state.success,
            );
            false
        }
        SubagentWaitOutcome::Error(error) => {
            timeout_cancel.cancel();
            task_handle.abort();
            mark_subagent_agent_error(
                &mut state.final_response,
                &mut state.error_details,
                &mut state.success,
                &error.message,
            );
            false
        }
    }
}

async fn replay_subagent_events(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    parent_ctx: &ToolContext<()>,
    config: &SubagentConfig,
    subagent_id: &str,
    state: &mut SubagentExecutionState,
) -> Result<()> {
    for envelope in event_store.get_events(thread_id).await? {
        match envelope.event {
            AgentEvent::Text {
                message_id: _,
                text,
            } => {
                state.final_response.push_str(&text);
            }
            AgentEvent::ToolCallStart {
                id, name, input, ..
            } => {
                state.tool_count += 1;
                let context = extract_tool_context(&name, &input);
                state
                    .pending_tools
                    .insert(id, (name.clone(), context.clone()));

                emit_subagent_progress_if_possible(
                    parent_ctx,
                    config,
                    SubagentProgressUpdate {
                        subagent_id,
                        total_turns: state.total_turns,
                        total_usage: &state.total_usage,
                        tool_name: name,
                        tool_context: context,
                        completed: false,
                        success: false,
                        tool_count: state.tool_count,
                    },
                )
                .await;
            }
            AgentEvent::ToolCallEnd {
                id,
                name,
                display_name,
                result,
            } => {
                let context = state
                    .pending_tools
                    .remove(&id)
                    .map(|(_, ctx)| ctx)
                    .unwrap_or_default();
                let tool_success = result.success;
                state.tool_logs.push(ToolCallLog {
                    name: name.clone(),
                    display_name: display_name.clone(),
                    context: context.clone(),
                    result: summarize_tool_result(&name, &result),
                    success: tool_success,
                    duration_ms: result.duration_ms,
                });

                emit_subagent_progress_if_possible(
                    parent_ctx,
                    config,
                    SubagentProgressUpdate {
                        subagent_id,
                        total_turns: state.total_turns,
                        total_usage: &state.total_usage,
                        tool_name: name,
                        tool_context: context,
                        completed: true,
                        success: tool_success,
                        tool_count: state.tool_count,
                    },
                )
                .await;
            }
            AgentEvent::TurnComplete { turn, usage, .. } => {
                state.total_turns = turn;
                state.total_usage.add(&usage);
            }
            AgentEvent::Done {
                total_turns: turns, ..
            } => {
                state.total_turns = turns;
                break;
            }
            AgentEvent::Refusal { text, .. } => {
                let refusal_message =
                    text.unwrap_or_else(|| "Subagent refused the request".to_string());
                state.error_details = Some(refusal_message.clone());
                state.final_response = refusal_message;
                state.success = false;
                break;
            }
            AgentEvent::Error { message, .. } => {
                state.error_details = Some(message.clone());
                state.final_response = message;
                state.success = false;
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

async fn emit_subagent_progress_if_possible(
    parent_ctx: &ToolContext<()>,
    config: &SubagentConfig,
    update: SubagentProgressUpdate<'_>,
) {
    if let Err(error) = emit_subagent_progress(parent_ctx, config, update).await {
        log::warn!("Failed to emit subagent progress event: {error}");
    }
}

async fn emit_subagent_progress(
    parent_ctx: &ToolContext<()>,
    config: &SubagentConfig,
    SubagentProgressUpdate {
        subagent_id,
        total_turns,
        total_usage,
        tool_name,
        tool_context,
        completed,
        success,
        tool_count,
    }: SubagentProgressUpdate<'_>,
) -> Result<()> {
    let max_turns = config.max_turns.map(usize_to_u32_saturating);
    let current_turn = Some(usize_to_u32_saturating(total_turns));

    parent_ctx
        .emit_event(AgentEvent::SubagentProgress {
            subagent_id: subagent_id.to_string(),
            subagent_name: config.name.clone(),
            nickname: config.nickname.clone(),
            max_turns,
            current_turn,
            model: config.model.clone(),
            tool_name,
            tool_context,
            completed,
            success,
            tool_count,
            total_tokens: subagent_total_tokens(total_usage),
        })
        .await
}

fn usize_to_u32_saturating(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(feature = "otel")]
fn emit_subagent_observability<P, H, M, S>(tool: &SubagentTool<P, H, M, S>, result: &SubagentResult)
where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    use crate::observability::{attrs, provider_name, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    let mut span = spans::start_internal_span(
        "invoke_agent",
        vec![
            KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "invoke_agent"),
            KeyValue::new(attrs::GEN_AI_AGENT_NAME, tool.config.name.clone()),
            KeyValue::new(
                attrs::GEN_AI_PROVIDER_NAME,
                provider_name::normalize(tool.provider.provider()),
            ),
            KeyValue::new(
                attrs::GEN_AI_REQUEST_MODEL,
                tool.provider.model().to_string(),
            ),
            KeyValue::new(attrs::SDK_RUN_MODE, "loop"),
        ],
    );
    let outcome = if result.success { "done" } else { "error" };
    span.set_attribute(KeyValue::new(attrs::SDK_OUTCOME, outcome));
    span.set_attribute(attrs::kv_i64(
        attrs::SDK_TOTAL_TURNS,
        i64::try_from(result.total_turns).unwrap_or(0),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        i64::from(result.usage.input_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
        i64::from(result.usage.output_tokens),
    ));
    if outcome == "error" {
        spans::set_span_error(&mut span, "agent_error", "subagent invocation failed");
    }
    span.end();
}

#[cfg(not(feature = "otel"))]
fn emit_subagent_observability<P, H, M, S>(
    _tool: &SubagentTool<P, H, M, S>,
    _result: &SubagentResult,
) where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
}

/// Extracts context information from tool input for display.
fn extract_tool_context(name: &str, input: &Value) -> String {
    match name {
        "read" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "write" | "edit" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "bash" => {
            let cmd = input.get("command").and_then(Value::as_str).unwrap_or("");
            // Truncate long commands (UTF-8 safe)
            if cmd.len() > 60 {
                format!("{}...", crate::primitive_tools::truncate_str(cmd, 57))
            } else {
                cmd.to_string()
            }
        }
        "glob" | "grep" => input
            .get("pattern")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "web_search" => input
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Summarizes tool result for logging.
fn summarize_tool_result(name: &str, result: &ToolResult) -> String {
    if !result.success {
        let first_line = result.output.lines().next().unwrap_or("Error");
        return if first_line.len() > 50 {
            format!(
                "{}...",
                crate::primitive_tools::truncate_str(first_line, 47)
            )
        } else {
            first_line.to_string()
        };
    }

    match name {
        "read" => {
            let line_count = result.output.lines().count();
            format!("{line_count} lines")
        }
        "write" => "wrote file".to_string(),
        "edit" => "edited".to_string(),
        "bash" => {
            let lines: Vec<&str> = result.output.lines().collect();
            if lines.is_empty() {
                "done".to_string()
            } else if lines.len() == 1 {
                let line = lines[0];
                if line.len() > 50 {
                    format!("{}...", crate::primitive_tools::truncate_str(line, 47))
                } else {
                    line.to_string()
                }
            } else {
                format!("{} lines", lines.len())
            }
        }
        "glob" => {
            let count = result.output.lines().count();
            format!("{count} files")
        }
        "grep" => {
            let count = result.output.lines().count();
            format!("{count} matches")
        }
        _ => {
            let line_count = result.output.lines().count();
            if line_count == 0 {
                "done".to_string()
            } else {
                format!("{line_count} lines")
            }
        }
    }
}

impl<P, H, M, S> Tool<()> for SubagentTool<P, H, M, S>
where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(format!("subagent_{}", self.config.name))
    }

    fn display_name(&self) -> &'static str {
        self.cached_display_name
    }

    fn description(&self) -> &'static str {
        self.cached_description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or question for the subagent to handle"
                }
            },
            "required": ["task"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Subagent spawning requires confirmation
        ToolTier::Confirm
    }

    async fn execute(&self, ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let task = input
            .get("task")
            .and_then(Value::as_str)
            .context("Missing 'task' parameter")?;

        // ── Depth limit enforcement ───────────────────────────────────
        let current_depth = ctx
            .metadata
            .get(METADATA_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_depth = ctx
            .metadata
            .get(METADATA_MAX_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(3); // default: 3 levels deep

        if current_depth >= max_depth {
            bail!(
                "Subagent depth limit exceeded ({current_depth}/{max_depth}). \
                 Cannot spawn nested subagent '{}' — maximum nesting depth reached.",
                self.config.name
            );
        }

        // ── Thread limit enforcement (semaphore) ──────────────────────
        let _permit = if let Some(ref sem) = ctx.subagent_semaphore() {
            match sem.clone().try_acquire_owned() {
                Ok(permit) => Some(permit),
                Err(_) => {
                    return Ok(ToolResult {
                        success: false,
                        output: format!(
                            "Cannot spawn subagent '{}': maximum concurrent subagent limit reached. \
                             Try again when another subagent completes.",
                            self.config.name
                        ),
                        data: None,
                        documents: Vec::new(),
                        duration_ms: Some(0),
                    });
                }
            }
        } else {
            None
        };

        // Generate a unique ID for this subagent execution
        let subagent_id = format!(
            "{}_{:x}",
            self.config.name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        // Use the context's cancellation token if available, otherwise create a standalone one.
        // This ensures that when a parent agent is cancelled, subagents are also cancelled.
        let cancel_token = ctx.cancel_token().unwrap_or_default();

        let result = self
            .run_subagent(task, subagent_id, ctx, cancel_token)
            .await?;

        Ok(ToolResult {
            success: result.success,
            output: result.final_response.clone(),
            data: Some(serde_json::to_value(&result).unwrap_or_default()),
            documents: Vec::new(),
            duration_ms: Some(result.duration_ms),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
    use crate::llm::{ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage};
    use crate::stores::{EventStore, InMemoryEventStore, StoredTurnEvents};
    use anyhow::{Context, Result, bail};
    use async_trait::async_trait;
    use tokio::sync::Mutex;

    #[derive(Clone)]
    struct TestProvider {
        responses: Arc<Mutex<Vec<ChatOutcome>>>,
        delay: Option<Duration>,
    }

    impl TestProvider {
        fn new(responses: Vec<ChatOutcome>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
                delay: None,
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        fn text_response(text: &str) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "resp_text".to_string(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                model: "test-model".to_string(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cached_input_tokens: 0,
                },
            })
        }

        fn tool_use_response(tool_id: &str, tool_name: &str, input: Value) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "resp_tool".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: tool_id.to_string(),
                    name: tool_name.to_string(),
                    input,
                    thought_signature: None,
                }],
                model: "test-model".to_string(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 15,
                    output_tokens: 25,
                    cached_input_tokens: 0,
                },
            })
        }

        fn refusal_response(text: Option<&str>) -> ChatOutcome {
            let content = text.map_or_else(Vec::new, |text| {
                vec![ContentBlock::Text {
                    text: text.to_string(),
                }]
            });
            ChatOutcome::Success(ChatResponse {
                id: "resp_refusal".to_string(),
                content,
                model: "test-model".to_string(),
                stop_reason: Some(StopReason::Refusal),
                usage: Usage {
                    input_tokens: 12,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                },
            })
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }

            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                Ok(Self::text_response("default"))
            } else {
                Ok(responses.remove(0))
            }
        }

        fn model(&self) -> &'static str {
            "test-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    struct TestEchoTool;

    impl Tool<()> for TestEchoTool {
        type Name = DynamicToolName;

        fn name(&self) -> DynamicToolName {
            DynamicToolName::new("echo")
        }

        fn display_name(&self) -> &'static str {
            "Echo"
        }

        fn description(&self) -> &'static str {
            "Echo the input"
        }

        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }

        fn tier(&self) -> ToolTier {
            ToolTier::Observe
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
            let message = input
                .get("message")
                .and_then(Value::as_str)
                .context("missing echo message")?;
            Ok(ToolResult::success(format!("Echo: {message}")))
        }
    }

    #[derive(Clone, Default)]
    struct RecordingEventStore {
        inner: Arc<InMemoryEventStore>,
        appended: Arc<Mutex<Vec<(ThreadId, usize, AgentEventEnvelope)>>>,
    }

    impl RecordingEventStore {
        async fn appended_events(&self) -> Vec<(ThreadId, usize, AgentEventEnvelope)> {
            self.appended.lock().await.clone()
        }
    }

    #[async_trait]
    impl EventStore for RecordingEventStore {
        async fn append(
            &self,
            thread_id: &ThreadId,
            turn: usize,
            envelope: AgentEventEnvelope,
        ) -> Result<()> {
            self.appended
                .lock()
                .await
                .push((thread_id.clone(), turn, envelope.clone()));
            self.inner.append(thread_id, turn, envelope).await
        }

        async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
            self.inner.finish_turn(thread_id, turn).await
        }

        async fn get_turn(
            &self,
            thread_id: &ThreadId,
            turn: usize,
        ) -> Result<Option<StoredTurnEvents>> {
            self.inner.get_turn(thread_id, turn).await
        }

        async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
            self.inner.get_turns(thread_id).await
        }

        async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
            self.inner.clear(thread_id).await
        }
    }

    #[derive(Clone, Default)]
    struct AlwaysFailAppendEventStore;

    #[async_trait]
    impl EventStore for AlwaysFailAppendEventStore {
        async fn append(
            &self,
            _thread_id: &ThreadId,
            _turn: usize,
            _envelope: AgentEventEnvelope,
        ) -> Result<()> {
            bail!("append failed")
        }

        async fn finish_turn(&self, _thread_id: &ThreadId, _turn: usize) -> Result<()> {
            Ok(())
        }

        async fn get_turn(
            &self,
            _thread_id: &ThreadId,
            _turn: usize,
        ) -> Result<Option<StoredTurnEvents>> {
            Ok(None)
        }

        async fn get_turns(&self, _thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
            Ok(Vec::new())
        }

        async fn clear(&self, _thread_id: &ThreadId) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct NoReadAfterFailureEventStore {
        inner: Arc<InMemoryEventStore>,
    }

    #[async_trait]
    impl EventStore for NoReadAfterFailureEventStore {
        async fn append(
            &self,
            thread_id: &ThreadId,
            turn: usize,
            envelope: AgentEventEnvelope,
        ) -> Result<()> {
            self.inner.append(thread_id, turn, envelope).await
        }

        async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
            self.inner.finish_turn(thread_id, turn).await
        }

        async fn get_turn(
            &self,
            thread_id: &ThreadId,
            turn: usize,
        ) -> Result<Option<StoredTurnEvents>> {
            self.inner.get_turn(thread_id, turn).await
        }

        async fn get_turns(&self, _thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
            bail!("get_events should not be called after subagent failure")
        }

        async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
            self.inner.clear(thread_id).await
        }
    }

    #[derive(Clone, Default)]
    struct PanicProvider;

    #[async_trait]
    impl LlmProvider for PanicProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            panic!("panic provider should disconnect subagent");
        }

        fn model(&self) -> &'static str {
            "panic-model"
        }

        fn provider(&self) -> &'static str {
            "panic"
        }
    }

    #[test]
    fn test_subagent_config_builder() {
        let config = SubagentConfig::new("test")
            .with_system_prompt("Test prompt")
            .with_max_turns(5)
            .with_timeout_ms(30000);

        assert_eq!(config.name, "test");
        assert_eq!(config.system_prompt, "Test prompt");
        assert_eq!(config.max_turns, Some(5));
        assert_eq!(config.timeout_ms, Some(30000));
    }

    #[test]
    fn test_subagent_config_defaults() {
        let config = SubagentConfig::new("default");

        assert_eq!(config.name, "default");
        assert!(config.system_prompt.is_empty());
        assert_eq!(config.max_turns, None);
        assert_eq!(config.timeout_ms, None);
    }

    #[test]
    fn test_subagent_result_serialization() -> Result<()> {
        let result = SubagentResult {
            name: "test".to_string(),
            final_response: "Done".to_string(),
            total_turns: 3,
            tool_count: 5,
            tool_logs: vec![
                ToolCallLog {
                    name: "read".to_string(),
                    display_name: "Read file".to_string(),
                    context: "/tmp/test.rs".to_string(),
                    result: "50 lines".to_string(),
                    success: true,
                    duration_ms: Some(10),
                },
                ToolCallLog {
                    name: "grep".to_string(),
                    display_name: "Grep TODO".to_string(),
                    context: "TODO".to_string(),
                    result: "3 matches".to_string(),
                    success: true,
                    duration_ms: Some(5),
                },
            ],
            usage: TokenUsage::default(),
            success: true,
            duration_ms: 1000,
            error_details: None,
            failed_tool: None,
        };

        let json = serde_json::to_string(&result).context("failed to serialize subagent result")?;
        assert!(json.contains("test"));
        assert!(json.contains("Done"));
        assert!(json.contains("tool_count"));
        assert!(json.contains("tool_logs"));
        assert!(json.contains("/tmp/test.rs"));

        Ok(())
    }

    #[test]
    fn test_subagent_result_field_extraction() -> Result<()> {
        let result = SubagentResult {
            name: "explore".to_string(),
            final_response: "Found 3 config files".to_string(),
            total_turns: 2,
            tool_count: 5,
            tool_logs: vec![ToolCallLog {
                name: "glob".to_string(),
                display_name: "Glob config files".to_string(),
                context: "**/*.toml".to_string(),
                result: "3 files".to_string(),
                success: true,
                duration_ms: Some(15),
            }],
            usage: TokenUsage {
                input_tokens: 1500,
                output_tokens: 500,
            },
            success: true,
            duration_ms: 2500,
            error_details: None,
            failed_tool: None,
        };

        let value =
            serde_json::to_value(&result).context("failed to convert subagent result to json")?;

        let tool_count = value.get("tool_count").and_then(Value::as_u64);
        assert_eq!(tool_count, Some(5));

        let usage = value.get("usage").context("missing usage field")?;
        let input_tokens = usage.get("input_tokens").and_then(Value::as_u64);
        let output_tokens = usage.get("output_tokens").and_then(Value::as_u64);
        assert_eq!(input_tokens, Some(1500));
        assert_eq!(output_tokens, Some(500));

        let logs = value
            .get("tool_logs")
            .and_then(Value::as_array)
            .context("missing tool_logs array")?;
        assert_eq!(logs.len(), 1);

        let first_log = &logs[0];
        assert_eq!(first_log.get("name").and_then(Value::as_str), Some("glob"));
        assert_eq!(
            first_log.get("context").and_then(Value::as_str),
            Some("**/*.toml")
        );
        assert_eq!(
            first_log.get("result").and_then(Value::as_str),
            Some("3 files")
        );
        assert_eq!(
            first_log.get("success").and_then(Value::as_bool),
            Some(true)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_uses_isolated_child_thread() -> Result<()> {
        let event_store = Arc::new(RecordingEventStore::default());
        let provider = Arc::new(TestProvider::new(vec![
            TestProvider::tool_use_response("tool_1", "echo", json!({ "message": "child" })),
            TestProvider::text_response("Subagent complete"),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(TestEchoTool);

        let tool = SubagentTool::new(SubagentConfig::new("worker"), provider, Arc::new(tools), {
            let store = Arc::clone(&event_store);
            move || -> Arc<dyn EventStore> { store.clone() }
        });
        let parent_thread = ThreadId::new();
        let parent_ctx = ToolContext::new(()).with_event_store(
            event_store.clone(),
            parent_thread.clone(),
            1,
            SequenceCounter::new(),
        );

        let result = tool
            .run_subagent(
                "Inspect the repo",
                "subagent_1".to_string(),
                &parent_ctx,
                CancellationToken::new(),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.tool_count, 1);
        assert_eq!(result.tool_logs.len(), 1);

        let parent_turn = event_store
            .get_turn(&parent_thread, 1)
            .await?
            .context("missing parent turn")?;
        assert!(!parent_turn.events.is_empty());
        assert!(
            parent_turn
                .events
                .iter()
                .all(|envelope| { matches!(envelope.event, AgentEvent::SubagentProgress { .. }) })
        );

        let appended = event_store.appended_events().await;
        let child_thread = appended
            .iter()
            .map(|(thread_id, _, _)| thread_id.clone())
            .find(|thread_id| thread_id != &parent_thread)
            .context("missing child thread events")?;
        let child_turn = event_store
            .get_turn(&child_thread, 1)
            .await?
            .context("missing child turn")?;
        let child_events = event_store.get_events(&child_thread).await?;

        assert!(
            child_turn
                .events
                .iter()
                .any(|envelope| { matches!(envelope.event, AgentEvent::ToolCallStart { .. }) })
        );
        assert!(
            child_events
                .iter()
                .any(|envelope| { matches!(envelope.event, AgentEvent::Done { .. }) })
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_timeout_marks_result_as_failed() -> Result<()> {
        let event_store = Arc::new(NoReadAfterFailureEventStore::default());
        let provider = Arc::new(
            TestProvider::new(vec![TestProvider::text_response("Too late")])
                .with_delay(Duration::from_millis(50)),
        );
        let tool = SubagentTool::new(
            SubagentConfig::new("worker").with_timeout_ms(10),
            provider,
            Arc::new(ToolRegistry::<()>::new()),
            {
                let store = Arc::clone(&event_store);
                move || -> Arc<dyn EventStore> { store.clone() }
            },
        );

        let result = tool
            .run_subagent(
                "Take too long",
                "subagent_timeout".to_string(),
                &ToolContext::new(()),
                CancellationToken::new(),
            )
            .await?;

        assert!(!result.success);
        assert_eq!(result.final_response, "Subagent timed out");
        assert!(
            result
                .error_details
                .context("missing timeout details")?
                .contains("timed out")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_progress_failures_do_not_abort_successful_runs() -> Result<()> {
        let provider = Arc::new(TestProvider::new(vec![
            TestProvider::tool_use_response("tool_1", "echo", json!({ "message": "child" })),
            TestProvider::text_response("Subagent complete"),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(TestEchoTool);

        let tool = SubagentTool::new(SubagentConfig::new("worker"), provider, Arc::new(tools), {
            move || -> Arc<dyn EventStore> { Arc::new(InMemoryEventStore::new()) }
        });
        let parent_ctx = ToolContext::new(()).with_event_store(
            Arc::new(AlwaysFailAppendEventStore),
            ThreadId::new(),
            1,
            SequenceCounter::new(),
        );

        let result = tool
            .run_subagent(
                "Inspect the repo",
                "subagent_progress".to_string(),
                &parent_ctx,
                CancellationToken::new(),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.final_response, "Subagent complete");
        assert_eq!(result.tool_count, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_disconnected_marks_result_as_failed() -> Result<()> {
        let tool = SubagentTool::new(
            SubagentConfig::new("worker"),
            Arc::new(PanicProvider),
            Arc::new(ToolRegistry::<()>::new()),
            move || -> Arc<dyn EventStore> { Arc::new(InMemoryEventStore::new()) },
        );

        let result = tool
            .run_subagent(
                "Crash",
                "subagent_panic".to_string(),
                &ToolContext::new(()),
                CancellationToken::new(),
            )
            .await?;

        assert!(!result.success);
        assert_eq!(result.final_response, "Subagent ended unexpectedly");
        assert!(
            result
                .error_details
                .context("missing disconnect details")?
                .contains("ended before returning a final state")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_refusal_marks_result_as_failed() -> Result<()> {
        let tool = SubagentTool::new(
            SubagentConfig::new("worker"),
            Arc::new(TestProvider::new(vec![TestProvider::refusal_response(
                Some("Refused for policy reasons"),
            )])),
            Arc::new(ToolRegistry::<()>::new()),
            || Arc::new(InMemoryEventStore::new()),
        );

        let result = tool
            .run_subagent(
                "Refuse",
                "subagent_refusal".to_string(),
                &ToolContext::new(()),
                CancellationToken::new(),
            )
            .await?;

        assert!(!result.success);
        assert_eq!(result.final_response, "Refused for policy reasons");
        assert_eq!(
            result.error_details.as_deref(),
            Some("Refused for policy reasons")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_cancelled_marks_result_as_failed() -> Result<()> {
        let tool = SubagentTool::new(
            SubagentConfig::new("worker"),
            Arc::new(
                TestProvider::new(vec![TestProvider::text_response("Too late")])
                    .with_delay(Duration::from_millis(50)),
            ),
            Arc::new(ToolRegistry::<()>::new()),
            || Arc::new(InMemoryEventStore::new()),
        );
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();

        let result = tool
            .run_subagent(
                "Cancel",
                "subagent_cancelled".to_string(),
                &ToolContext::new(()),
                cancel_token,
            )
            .await?;

        assert!(!result.success);
        assert_eq!(result.final_response, "Subagent cancelled");
        assert!(
            result
                .error_details
                .context("missing cancellation details")?
                .contains("cancelled")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_run_subagent_llm_error_does_not_infer_failed_tool() -> Result<()> {
        let provider = Arc::new(TestProvider::new(vec![
            ChatOutcome::ServerError("llm transport failed".to_string()),
            ChatOutcome::ServerError("llm transport failed".to_string()),
            ChatOutcome::ServerError("llm transport failed".to_string()),
            ChatOutcome::ServerError("llm transport failed".to_string()),
            ChatOutcome::ServerError("llm transport failed".to_string()),
            ChatOutcome::ServerError("llm transport failed".to_string()),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(TestEchoTool);

        let tool = SubagentTool::new(
            SubagentConfig::new("worker"),
            provider,
            Arc::new(tools),
            || Arc::new(InMemoryEventStore::new()),
        );

        let result = tool
            .run_subagent(
                "Trigger an llm failure",
                "subagent_llm_error".to_string(),
                &ToolContext::new(()),
                CancellationToken::new(),
            )
            .await?;

        assert!(!result.success);
        assert!(result.failed_tool.is_none());
        assert!(
            result
                .error_details
                .as_deref()
                .unwrap_or_default()
                .contains("Server error")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_replay_subagent_events_stops_after_error() -> Result<()> {
        let event_store: Arc<dyn EventStore> = Arc::new(InMemoryEventStore::new());
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();
        event_store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(
                    AgentEvent::Error {
                        message: "subagent boom".to_string(),
                        recoverable: false,
                    },
                    &seq,
                ),
            )
            .await?;
        event_store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(
                    AgentEvent::Text {
                        message_id: "msg_after_error".to_string(),
                        text: "should not be appended".to_string(),
                    },
                    &seq,
                ),
            )
            .await?;

        let mut state = SubagentExecutionState::new();
        replay_subagent_events(
            &event_store,
            &thread_id,
            &ToolContext::new(()),
            &SubagentConfig::new("worker"),
            "subagent_error",
            &mut state,
        )
        .await?;

        assert!(!state.success);
        assert_eq!(state.final_response, "subagent boom");
        assert_eq!(state.error_details.as_deref(), Some("subagent boom"));

        Ok(())
    }
}
