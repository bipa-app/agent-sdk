//! Agent loop orchestration module.
//!
//! This module contains the core agent loop that orchestrates LLM calls,
//! tool execution, and event handling. The agent loop is the main entry point
//! for running an AI agent.
//!
//! # Architecture
//!
//! The agent loop works as follows:
//! 1. Receives a user message
//! 2. Sends the message to the LLM provider
//! 3. Processes the LLM response (text or tool calls)
//! 4. If tool calls are present, executes them and feeds results back to LLM
//! 5. Repeats until the LLM responds with only text (no tool calls)
//! 6. Emits events throughout for real-time UI updates
//!
//! # Building an Agent
//!
//! Use the builder pattern via [`builder()`] or [`AgentLoopBuilder`]:
//!
//! ```ignore
//! use agent_sdk::{builder, providers::AnthropicProvider};
//!
//! let agent = builder()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .tools(my_tools)
//!     .build();
//! ```

use crate::context::{CompactionConfig, ContextCompactor, LlmContextCompactor};
use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, DefaultHooks, ToolDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamAccumulator, StreamDelta, Usage,
};
use crate::skills::Skill;
use crate::stores::{InMemoryStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ErasedAsyncTool, ErasedToolStatus, ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    ExecutionStatus, PendingToolCallInfo, RetryConfig, ThreadId, TokenUsage, ToolExecution,
    ToolOutcome, ToolResult, TurnOutcome,
};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Internal result of executing a single turn.
///
/// This is used internally by both `run_loop` and `run_single_turn`.
enum InternalTurnResult {
    /// Turn completed, more turns needed (tools were executed)
    Continue { turn_usage: TokenUsage },
    /// Done - no more tool calls
    Done,
    /// Awaiting confirmation (yields)
    AwaitingConfirmation {
        tool_call_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
        continuation: Box<AgentContinuation>,
    },
    /// Error
    Error(AgentError),
}

/// Mutable context for turn execution.
///
/// This holds all the state that's modified during execution.
struct TurnContext {
    thread_id: ThreadId,
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    start_time: Instant,
}

/// Data extracted from `AgentInput::Resume` after validation.
struct ResumeData {
    continuation: Box<AgentContinuation>,
    tool_call_id: String,
    confirmed: bool,
    rejection_reason: Option<String>,
}

/// Result of initializing state from agent input.
struct InitializedState {
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    resume_data: Option<ResumeData>,
}

/// Outcome of executing a single tool call.
enum ToolExecutionOutcome {
    /// Tool executed successfully (or failed), result captured
    Completed { tool_id: String, result: ToolResult },
    /// Tool requires user confirmation before execution
    RequiresConfirmation {
        tool_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
    },
}

/// Builder for constructing an `AgentLoop`.
///
/// # Example
///
/// ```ignore
/// let agent = AgentLoop::builder()
///     .provider(my_provider)
///     .tools(my_tools)
///     .config(AgentConfig::default())
///     .build();
/// ```
pub struct AgentLoopBuilder<Ctx, P, H, M, S> {
    provider: Option<P>,
    tools: Option<ToolRegistry<Ctx>>,
    hooks: Option<H>,
    message_store: Option<M>,
    state_store: Option<S>,
    config: Option<AgentConfig>,
    compaction_config: Option<CompactionConfig>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
}

impl<Ctx> AgentLoopBuilder<Ctx, (), (), (), ()> {
    /// Create a new builder with no components set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: None,
            hooks: None,
            message_store: None,
            state_store: None,
            config: None,
            compaction_config: None,
            execution_store: None,
        }
    }
}

impl<Ctx> Default for AgentLoopBuilder<Ctx, (), (), (), ()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx, P, H, M, S> AgentLoopBuilder<Ctx, P, H, M, S> {
    /// Set the LLM provider.
    #[must_use]
    pub fn provider<P2: LlmProvider>(self, provider: P2) -> AgentLoopBuilder<Ctx, P2, H, M, S> {
        AgentLoopBuilder {
            provider: Some(provider),
            tools: self.tools,
            hooks: self.hooks,
            message_store: self.message_store,
            state_store: self.state_store,
            config: self.config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }

    /// Set the tool registry.
    #[must_use]
    pub fn tools(mut self, tools: ToolRegistry<Ctx>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the agent hooks.
    #[must_use]
    pub fn hooks<H2: AgentHooks>(self, hooks: H2) -> AgentLoopBuilder<Ctx, P, H2, M, S> {
        AgentLoopBuilder {
            provider: self.provider,
            tools: self.tools,
            hooks: Some(hooks),
            message_store: self.message_store,
            state_store: self.state_store,
            config: self.config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }

    /// Set the message store.
    #[must_use]
    pub fn message_store<M2: MessageStore>(
        self,
        message_store: M2,
    ) -> AgentLoopBuilder<Ctx, P, H, M2, S> {
        AgentLoopBuilder {
            provider: self.provider,
            tools: self.tools,
            hooks: self.hooks,
            message_store: Some(message_store),
            state_store: self.state_store,
            config: self.config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }

    /// Set the state store.
    #[must_use]
    pub fn state_store<S2: StateStore>(
        self,
        state_store: S2,
    ) -> AgentLoopBuilder<Ctx, P, H, M, S2> {
        AgentLoopBuilder {
            provider: self.provider,
            tools: self.tools,
            hooks: self.hooks,
            message_store: self.message_store,
            state_store: Some(state_store),
            config: self.config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }

    /// Set the execution store for tool idempotency.
    ///
    /// When set, tool executions will be tracked using a write-ahead pattern:
    /// 1. Record execution intent BEFORE calling the tool
    /// 2. Update with result AFTER completion
    /// 3. On retry, return cached result if execution already completed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use agent_sdk::{builder, stores::InMemoryExecutionStore};
    ///
    /// let agent = builder()
    ///     .provider(my_provider)
    ///     .execution_store(InMemoryExecutionStore::new())
    ///     .build();
    /// ```
    #[must_use]
    pub fn execution_store(mut self, store: impl ToolExecutionStore + 'static) -> Self {
        self.execution_store = Some(Arc::new(store));
        self
    }

    /// Set the agent configuration.
    #[must_use]
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable context compaction with the given configuration.
    ///
    /// When enabled, the agent will automatically compact conversation history
    /// when it exceeds the configured token threshold.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use agent_sdk::{builder, context::CompactionConfig};
    ///
    /// let agent = builder()
    ///     .provider(my_provider)
    ///     .with_compaction(CompactionConfig::default())
    ///     .build();
    /// ```
    #[must_use]
    pub const fn with_compaction(mut self, config: CompactionConfig) -> Self {
        self.compaction_config = Some(config);
        self
    }

    /// Enable context compaction with default settings.
    ///
    /// This is a convenience method equivalent to:
    /// ```ignore
    /// builder.with_compaction(CompactionConfig::default())
    /// ```
    #[must_use]
    pub fn with_auto_compaction(self) -> Self {
        self.with_compaction(CompactionConfig::default())
    }

    /// Apply a skill configuration.
    ///
    /// This merges the skill's system prompt with the existing configuration
    /// and filters tools based on the skill's allowed/denied lists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let skill = Skill::new("code-review", "You are a code reviewer...")
    ///     .with_denied_tools(vec!["bash".into()]);
    ///
    /// let agent = builder()
    ///     .provider(provider)
    ///     .tools(tools)
    ///     .with_skill(skill)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_skill(mut self, skill: Skill) -> Self
    where
        Ctx: Send + Sync + 'static,
    {
        // Filter tools based on skill configuration first (before moving skill)
        if let Some(ref mut tools) = self.tools {
            tools.filter(|name| skill.is_tool_allowed(name));
        }

        // Merge system prompt
        let mut config = self.config.take().unwrap_or_default();
        if config.system_prompt.is_empty() {
            config.system_prompt = skill.system_prompt;
        } else {
            config.system_prompt = format!("{}\n\n{}", config.system_prompt, skill.system_prompt);
        }
        self.config = Some(config);

        self
    }
}

impl<Ctx, P> AgentLoopBuilder<Ctx, P, (), (), ()>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider + 'static,
{
    /// Build the agent loop with default hooks and in-memory stores.
    ///
    /// This is a convenience method that uses:
    /// - `DefaultHooks` for hooks
    /// - `InMemoryStore` for message store
    /// - `InMemoryStore` for state store
    /// - `AgentConfig::default()` if no config is set
    ///
    /// # Panics
    ///
    /// Panics if a provider has not been set.
    #[must_use]
    pub fn build(self) -> AgentLoop<Ctx, P, DefaultHooks, InMemoryStore, InMemoryStore> {
        let provider = self.provider.expect("provider is required");
        let tools = self.tools.unwrap_or_default();
        let config = self.config.unwrap_or_default();

        AgentLoop {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(DefaultHooks),
            message_store: Arc::new(InMemoryStore::new()),
            state_store: Arc::new(InMemoryStore::new()),
            config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }
}

impl<Ctx, P, H, M, S> AgentLoopBuilder<Ctx, P, H, M, S>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider + 'static,
    H: AgentHooks + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    /// Build the agent loop with all custom components.
    ///
    /// # Panics
    ///
    /// Panics if any of the following have not been set:
    /// - `provider`
    /// - `hooks`
    /// - `message_store`
    /// - `state_store`
    #[must_use]
    pub fn build_with_stores(self) -> AgentLoop<Ctx, P, H, M, S> {
        let provider = self.provider.expect("provider is required");
        let tools = self.tools.unwrap_or_default();
        let hooks = self
            .hooks
            .expect("hooks is required when using build_with_stores");
        let message_store = self
            .message_store
            .expect("message_store is required when using build_with_stores");
        let state_store = self
            .state_store
            .expect("state_store is required when using build_with_stores");
        let config = self.config.unwrap_or_default();

        AgentLoop {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(hooks),
            message_store: Arc::new(message_store),
            state_store: Arc::new(state_store),
            config,
            compaction_config: self.compaction_config,
            execution_store: self.execution_store,
        }
    }
}

/// The main agent loop that orchestrates LLM calls and tool execution.
///
/// `AgentLoop` is the core component that:
/// - Manages conversation state via message and state stores
/// - Calls the LLM provider and processes responses
/// - Executes tools through the tool registry
/// - Emits events for real-time updates via an async channel
/// - Enforces hooks for tool permissions and lifecycle events
///
/// # Type Parameters
///
/// - `Ctx`: Application-specific context passed to tools (e.g., user ID, database)
/// - `P`: The LLM provider implementation
/// - `H`: The hooks implementation for lifecycle customization
/// - `M`: The message store implementation
/// - `S`: The state store implementation
///
/// # Running the Agent
///
/// ```ignore
/// let (mut events, final_state) = agent.run(
///     thread_id,
///     AgentInput::Text("Hello!".to_string()),
///     tool_ctx,
/// );
/// while let Some(event) = events.recv().await {
///     match event {
///         AgentEvent::Text { text } => println!("{}", text),
///         AgentEvent::Done { .. } => break,
///         _ => {}
///     }
/// }
/// ```
pub struct AgentLoop<Ctx, P, H, M, S>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    provider: Arc<P>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    message_store: Arc<M>,
    state_store: Arc<S>,
    config: AgentConfig,
    compaction_config: Option<CompactionConfig>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
}

/// Create a new builder for constructing an `AgentLoop`.
#[must_use]
pub fn builder<Ctx>() -> AgentLoopBuilder<Ctx, (), (), (), ()> {
    AgentLoopBuilder::new()
}

impl<Ctx, P, H, M, S> AgentLoop<Ctx, P, H, M, S>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider + 'static,
    H: AgentHooks + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    /// Create a new agent loop with all components specified directly.
    #[must_use]
    pub fn new(
        provider: P,
        tools: ToolRegistry<Ctx>,
        hooks: H,
        message_store: M,
        state_store: S,
        config: AgentConfig,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(hooks),
            message_store: Arc::new(message_store),
            state_store: Arc::new(state_store),
            config,
            compaction_config: None,
            execution_store: None,
        }
    }

    /// Create a new agent loop with compaction enabled.
    #[must_use]
    pub fn with_compaction(
        provider: P,
        tools: ToolRegistry<Ctx>,
        hooks: H,
        message_store: M,
        state_store: S,
        config: AgentConfig,
        compaction_config: CompactionConfig,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(hooks),
            message_store: Arc::new(message_store),
            state_store: Arc::new(state_store),
            config,
            compaction_config: Some(compaction_config),
            execution_store: None,
        }
    }

    /// Run the agent loop.
    ///
    /// This method allows the agent to pause when a tool requires confirmation,
    /// returning an `AgentRunState::AwaitingConfirmation` that contains the
    /// state needed to resume.
    ///
    /// # Arguments
    ///
    /// * `thread_id` - The thread identifier for this conversation
    /// * `input` - Either a new text message or a resume with confirmation decision
    /// * `tool_context` - Context passed to tools
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `mpsc::Receiver<AgentEvent>` - Channel for streaming events
    /// - `oneshot::Receiver<AgentRunState>` - Channel for the final state
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (events, final_state) = agent.run(
    ///     thread_id,
    ///     AgentInput::Text("Hello".to_string()),
    ///     tool_ctx,
    /// );
    ///
    /// while let Some(event) = events.recv().await {
    ///     // Handle events...
    /// }
    ///
    /// match final_state.await.unwrap() {
    ///     AgentRunState::Done { .. } => { /* completed */ }
    ///     AgentRunState::AwaitingConfirmation { continuation, .. } => {
    ///         // Get user decision, then resume:
    ///         let (events2, state2) = agent.run(
    ///             thread_id,
    ///             AgentInput::Resume {
    ///                 continuation,
    ///                 tool_call_id: id,
    ///                 confirmed: true,
    ///                 rejection_reason: None,
    ///             },
    ///             tool_ctx,
    ///         );
    ///     }
    ///     AgentRunState::Error(e) => { /* handle error */ }
    /// }
    /// ```
    pub fn run(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
    ) -> (mpsc::Receiver<AgentEvent>, oneshot::Receiver<AgentRunState>)
    where
        Ctx: Clone,
    {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (state_tx, state_rx) = oneshot::channel();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let execution_store = self.execution_store.clone();

        tokio::spawn(async move {
            let result = run_loop(
                event_tx,
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
                execution_store,
            )
            .await;

            let _ = state_tx.send(result);
        });

        (event_rx, state_rx)
    }

    /// Run a single turn of the agent loop.
    ///
    /// Unlike `run()`, this method executes exactly one turn and returns control
    /// to the caller. This enables external orchestration where each turn can be
    /// dispatched as a separate message (e.g., via Artemis or another message queue).
    ///
    /// # Arguments
    ///
    /// * `thread_id` - The thread identifier for this conversation
    /// * `input` - Text to start, Resume after confirmation, or Continue after a turn
    /// * `tool_context` - Context passed to tools
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `mpsc::Receiver<AgentEvent>` - Channel for streaming events from this turn
    /// - `TurnOutcome` - The turn's outcome
    ///
    /// # Turn Outcomes
    ///
    /// - `NeedsMoreTurns` - Turn completed, call again with `AgentInput::Continue`
    /// - `Done` - Agent completed successfully
    /// - `AwaitingConfirmation` - Tool needs confirmation, call again with `AgentInput::Resume`
    /// - `Error` - An error occurred
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start conversation
    /// let (events, outcome) = agent.run_turn(
    ///     thread_id.clone(),
    ///     AgentInput::Text("What is 2+2?".to_string()),
    ///     tool_ctx.clone(),
    /// ).await;
    ///
    /// // Process events...
    /// while let Some(event) = events.recv().await { /* ... */ }
    ///
    /// // Check outcome
    /// match outcome {
    ///     TurnOutcome::NeedsMoreTurns { turn, .. } => {
    ///         // Dispatch another message to continue
    ///         // (e.g., schedule an Artemis message)
    ///     }
    ///     TurnOutcome::Done { .. } => {
    ///         // Conversation complete
    ///     }
    ///     TurnOutcome::AwaitingConfirmation { continuation, .. } => {
    ///         // Get user confirmation, then resume
    ///     }
    ///     TurnOutcome::Error(e) => {
    ///         // Handle error
    ///     }
    /// }
    /// ```
    pub async fn run_turn(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
    ) -> (mpsc::Receiver<AgentEvent>, oneshot::Receiver<TurnOutcome>)
    where
        Ctx: Clone,
    {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (outcome_tx, outcome_rx) = oneshot::channel();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let execution_store = self.execution_store.clone();

        tokio::spawn(async move {
        let result = run_single_turn(TurnParameters {
            tx: event_tx,
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
            execution_store,
        })
        .await;

            let _ = outcome_tx.send(result);
        });

        (event_rx, outcome_rx)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Initialize agent state from the given input.
///
/// Handles the three input variants:
/// - `Text`: Creates/loads state, appends user message
/// - `Resume`: Restores from continuation state
/// - `Continue`: Loads existing state to continue execution
async fn initialize_from_input<M, S>(
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

/// Execute a single tool call with hook checks.
///
/// Returns the outcome of the tool execution, which may be:
/// - `Completed`: Tool ran (or was blocked), result captured
/// - `RequiresConfirmation`: Hook requires user confirmation
///
/// Supports both synchronous and asynchronous tools. Async tools are detected
/// automatically and their progress is streamed via events.
#[allow(clippy::too_many_lines)]
async fn execute_tool_call<Ctx, H>(
    pending: &PendingToolCallInfo,
    tool_context: &ToolContext<Ctx>,
    tools: &ToolRegistry<Ctx>,
    hooks: &Arc<H>,
    tx: &mpsc::Sender<AgentEvent>,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    // Check for async tool first
    if let Some(async_tool) = tools.get_async(&pending.name) {
        let tier = async_tool.tier();

        // Emit tool call start
        let _ = tx
            .send(AgentEvent::tool_call_start(
                &pending.id,
                &pending.name,
                &pending.display_name,
                pending.input.clone(),
                tier,
            ))
            .await;

        // Check hooks for permission
        let decision = hooks
            .pre_tool_use(&pending.name, &pending.input, tier)
            .await;

        return match decision {
            ToolDecision::Allow => {
                let result = execute_async_tool(pending, async_tool, tool_context, tx).await;

                hooks.post_tool_use(&pending.name, &result).await;

                send_event(
                    tx,
                    hooks,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;

                ToolExecutionOutcome::Completed {
                    tool_id: pending.id.clone(),
                    result,
                }
            }
            ToolDecision::Block(reason) => {
                let result = ToolResult::error(format!("Blocked: {reason}"));
                send_event(
                    tx,
                    hooks,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;
                ToolExecutionOutcome::Completed {
                    tool_id: pending.id.clone(),
                    result,
                }
            }
            ToolDecision::RequiresConfirmation(description) => {
                send_event(
                    tx,
                    hooks,
                    AgentEvent::ToolRequiresConfirmation {
                        id: pending.id.clone(),
                        name: pending.name.clone(),
                        input: pending.input.clone(),
                        description: description.clone(),
                    },
                )
                .await;

                ToolExecutionOutcome::RequiresConfirmation {
                    tool_id: pending.id.clone(),
                    tool_name: pending.name.clone(),
                    display_name: pending.display_name.clone(),
                    input: pending.input.clone(),
                    description,
                }
            }
        };
    }

    // Fall back to sync tool
    let Some(tool) = tools.get(&pending.name) else {
        let result = ToolResult::error(format!("Unknown tool: {}", pending.name));
        return ToolExecutionOutcome::Completed {
            tool_id: pending.id.clone(),
            result,
        };
    };

    let tier = tool.tier();

    // Emit tool call start
    let _ = tx
        .send(AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ))
        .await;

    // Check hooks for permission
    let decision = hooks
        .pre_tool_use(&pending.name, &pending.input, tier)
        .await;

    match decision {
        ToolDecision::Allow => {
            let tool_start = Instant::now();
            let result = match tool.execute(tool_context, pending.input.clone()).await {
                Ok(mut r) => {
                    r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                    r
                }
                Err(e) => ToolResult::error(format!("Tool error: {e}"))
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
            };

            hooks.post_tool_use(&pending.name, &result).await;

            send_event(
                tx,
                hooks,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;

            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::Block(reason) => {
            let result = ToolResult::error(format!("Blocked: {reason}"));
            send_event(
                tx,
                hooks,
                AgentEvent::tool_call_end(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    result.clone(),
                ),
            )
            .await;
            ToolExecutionOutcome::Completed {
                tool_id: pending.id.clone(),
                result,
            }
        }
        ToolDecision::RequiresConfirmation(description) => {
            send_event(
                tx,
                hooks,
                AgentEvent::ToolRequiresConfirmation {
                    id: pending.id.clone(),
                    name: pending.name.clone(),
                    input: pending.input.clone(),
                    description: description.clone(),
                },
            )
            .await;

            ToolExecutionOutcome::RequiresConfirmation {
                tool_id: pending.id.clone(),
                tool_name: pending.name.clone(),
                display_name: pending.display_name.clone(),
                input: pending.input.clone(),
                description,
            }
        }
    }
}

/// Execute an async tool call and stream progress until completion.
///
/// This function handles the two-phase execution of async tools:
/// 1. Execute the tool (returns immediately with Success/Failed/`InProgress`)
/// 2. If `InProgress`, stream status updates until completion
async fn execute_async_tool<Ctx>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedAsyncTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    tx: &mpsc::Sender<AgentEvent>,
) -> ToolResult
where
    Ctx: Send + Sync + Clone,
{
    let tool_start = Instant::now();

    // Step 1: Execute (lightweight, returns quickly)
    let outcome = match tool.execute(tool_context, pending.input.clone()).await {
        Ok(o) => o,
        Err(e) => {
            return ToolResult::error(format!("Tool error: {e}"))
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
        }
    };

    match outcome {
        // Synchronous completion - return immediately
        ToolOutcome::Success(mut result) | ToolOutcome::Failed(mut result) => {
            result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
            result
        }

        // Async operation - stream status until completion
        ToolOutcome::InProgress {
            operation_id,
            message,
        } => {
            // Emit initial progress
            let _ = tx
                .send(AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "started",
                    &message,
                    None,
                ))
                .await;

            // Stream status updates
            let mut stream = tool.check_status_stream(tool_context, &operation_id);

            while let Some(status) = stream.next().await {
                match status {
                    ErasedToolStatus::Progress {
                        stage,
                        message,
                        data,
                    } => {
                        let _ = tx
                            .send(AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                stage,
                                message,
                                data,
                            ))
                            .await;
                    }
                    ErasedToolStatus::Completed(mut result)
                    | ErasedToolStatus::Failed(mut result) => {
                        result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        return result;
                    }
                }
            }

            // Stream ended without completion (shouldn't happen)
            ToolResult::error("Async tool stream ended without completion")
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis()))
        }
    }
}

/// Execute the confirmed tool call from a resume operation.
///
/// This is called when resuming after a tool required confirmation.
/// Supports both sync and async tools.
async fn execute_confirmed_tool<Ctx, H>(
    awaiting_tool: &PendingToolCallInfo,
    confirmed: bool,
    rejection_reason: Option<String>,
    tool_context: &ToolContext<Ctx>,
    tools: &ToolRegistry<Ctx>,
    hooks: &Arc<H>,
    tx: &mpsc::Sender<AgentEvent>,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if confirmed {
        // Check for async tool first
        if let Some(async_tool) = tools.get_async(&awaiting_tool.name) {
            let result = execute_async_tool(awaiting_tool, async_tool, tool_context, tx).await;

            hooks.post_tool_use(&awaiting_tool.name, &result).await;

            let _ = tx
                .send(AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ))
                .await;

            return result;
        }

        // Fall back to sync tool
        if let Some(tool) = tools.get(&awaiting_tool.name) {
            let tool_start = Instant::now();
            let result = match tool
                .execute(tool_context, awaiting_tool.input.clone())
                .await
            {
                Ok(mut r) => {
                    r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                    r
                }
                Err(e) => ToolResult::error(format!("Tool error: {e}"))
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
            };

            hooks.post_tool_use(&awaiting_tool.name, &result).await;

            let _ = tx
                .send(AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ))
                .await;

            result
        } else {
            ToolResult::error(format!("Unknown tool: {}", awaiting_tool.name))
        }
    } else {
        let reason = rejection_reason.unwrap_or_else(|| "User rejected".to_string());
        let result = ToolResult::error(format!("Rejected: {reason}"));
        send_event(
            tx,
            hooks,
            AgentEvent::tool_call_end(
                &awaiting_tool.id,
                &awaiting_tool.name,
                &awaiting_tool.display_name,
                result.clone(),
            ),
        )
        .await;
        result
    }
}

/// Append tool results to message history.
async fn append_tool_results<M>(
    tool_results: &[(String, ToolResult)],
    thread_id: &ThreadId,
    message_store: &Arc<M>,
) -> Result<(), AgentError>
where
    M: MessageStore,
{
    for (tool_id, result) in tool_results {
        let tool_result_msg = Message::tool_result(tool_id, &result.output, !result.success);
        if let Err(e) = message_store.append(thread_id, tool_result_msg).await {
            return Err(AgentError::new(
                format!("Failed to append tool result: {e}"),
                false,
            ));
        }
    }
    Ok(())
}

/// Call the LLM with retry logic for rate limits and server errors.
async fn call_llm_with_retry<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    tx: &mpsc::Sender<AgentEvent>,
    hooks: &Arc<H>,
) -> Result<ChatResponse, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let outcome = match provider.chat(request.clone()).await {
            Ok(o) => o,
            Err(e) => {
                return Err(AgentError::new(format!("LLM error: {e}"), false));
            }
        };

        match outcome {
            ChatOutcome::Success(response) => return Ok(response),
            ChatOutcome::RateLimited => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Rate limited by LLM provider after {max_retries} retries");
                    let error_msg = format!("Rate limited after {max_retries} retries");
                    send_event(tx, hooks, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Rate limited, retrying after backoff"
                );
                let _ = tx
                    .send(AgentEvent::text(format!(
                        "\n[Rate limited, retrying in {:.1}s... (attempt {attempt}/{max_retries})]\n",
                        delay.as_secs_f64()
                    )))
                    .await;
                sleep(delay).await;
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!(msg, "Invalid request to LLM");
                return Err(AgentError::new(format!("Invalid request: {msg}"), false));
            }
            ChatOutcome::ServerError(msg) => {
                attempt += 1;
                if attempt > max_retries {
                    error!(msg, "LLM server error after {max_retries} retries");
                    let error_msg = format!("Server error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    error = msg,
                    "Server error, retrying after backoff"
                );
                send_event(
                    tx,
                    hooks,
                    AgentEvent::text(format!(
                        "\n[Server error: {msg}, retrying in {:.1}s... (attempt {attempt}/{max_retries})]\n",
                        delay.as_secs_f64()
                    )),
                )
                .await;
                sleep(delay).await;
            }
        }
    }
}

/// Call the LLM with streaming, emitting deltas as they arrive.
///
/// This function handles streaming responses from the LLM, emitting `TextDelta`
/// and `Thinking` events in real-time as content arrives. It includes retry logic
/// for recoverable errors (rate limits, server errors).
async fn call_llm_streaming<P, H>(
    provider: &Arc<P>,
    request: ChatRequest,
    config: &AgentConfig,
    tx: &mpsc::Sender<AgentEvent>,
    hooks: &Arc<H>,
) -> Result<ChatResponse, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let result = process_stream(provider, &request, tx, hooks).await;

        match result {
            Ok(response) => return Ok(response),
            Err(StreamError::Recoverable(msg)) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Streaming error after {max_retries} retries: {msg}");
                    let err_msg = format!("Streaming error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, AgentEvent::error(&err_msg, true)).await;
                    return Err(AgentError::new(err_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    error = msg,
                    "Streaming error, retrying"
                );
                send_event(
                    tx,
                    hooks,
                    AgentEvent::text(format!(
                        "\n[Streaming error: {msg}, retrying in {:.1}s... (attempt {attempt}/{max_retries})]\n",
                        delay.as_secs_f64()
                    )),
                )
                .await;
                sleep(delay).await;
            }
            Err(StreamError::Fatal(msg)) => {
                error!("Streaming error (non-recoverable): {msg}");
                return Err(AgentError::new(format!("Streaming error: {msg}"), false));
            }
        }
    }
}

/// Error type for stream processing.
enum StreamError {
    Recoverable(String),
    Fatal(String),
}

/// Process a single streaming attempt and return the response or error.
async fn process_stream<P, H>(
    provider: &Arc<P>,
    request: &ChatRequest,
    tx: &mpsc::Sender<AgentEvent>,
    hooks: &Arc<H>,
) -> Result<ChatResponse, StreamError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let mut stream = std::pin::pin!(provider.chat_stream(request.clone()));
    let mut accumulator = StreamAccumulator::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(delta) => {
                accumulator.apply(&delta);
                match &delta {
                    StreamDelta::TextDelta { delta, .. } => {
                        send_event(tx, hooks, AgentEvent::text_delta(delta.clone())).await;
                    }
                    StreamDelta::ThinkingDelta { delta, .. } => {
                        send_event(tx, hooks, AgentEvent::thinking(delta.clone())).await;
                    }
                    StreamDelta::Error {
                        message,
                        recoverable,
                    } => {
                        return if *recoverable {
                            Err(StreamError::Recoverable(message.clone()))
                        } else {
                            Err(StreamError::Fatal(message.clone()))
                        };
                    }
                    // These are handled by the accumulator or not needed as events
                    StreamDelta::Done { .. }
                    | StreamDelta::Usage(_)
                    | StreamDelta::ToolUseStart { .. }
                    | StreamDelta::ToolInputDelta { .. } => {}
                }
            }
            Err(e) => return Err(StreamError::Recoverable(format!("Stream error: {e}"))),
        }
    }

    let usage = accumulator.usage().cloned().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
    });
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    Ok(ChatResponse {
        id: String::new(),
        content: content_blocks,
        model: provider.model().to_string(),
        stop_reason,
        usage,
    })
}

// =============================================================================
// Main Loop Functions
// =============================================================================

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_loop<Ctx, P, H, M, S>(
    tx: mpsc::Sender<AgentEvent>,
    thread_id: ThreadId,
    input: AgentInput,
    tool_context: ToolContext<Ctx>,
    provider: Arc<P>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    message_store: Arc<M>,
    state_store: Arc<S>,
    config: AgentConfig,
    compaction_config: Option<CompactionConfig>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
) -> AgentRunState
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    // Add event channel to tool context so tools can emit events
    let tool_context = tool_context.with_event_tx(tx.clone());
    let start_time = Instant::now();

    // Initialize state from input
    let init_state =
        match initialize_from_input(input, &thread_id, &message_store, &state_store).await {
            Ok(s) => s,
            Err(e) => return AgentRunState::Error(e),
        };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume) = resume_data {
        let ResumeData {
            continuation: cont,
            tool_call_id,
            confirmed,
            rejection_reason,
        } = resume;
        let mut tool_results = cont.completed_results.clone();
        let awaiting_tool = &cont.pending_tool_calls[cont.awaiting_index];

        if awaiting_tool.id != tool_call_id {
            let message = format!(
                "Tool call ID mismatch: expected {}, got {}",
                awaiting_tool.id, tool_call_id
            );
            let recoverable = false;
            send_event(&tx, &hooks, AgentEvent::error(&message, recoverable)).await;
            return AgentRunState::Error(AgentError::new(&message, recoverable));
        }

        let result = execute_confirmed_tool(
            awaiting_tool,
            confirmed,
            rejection_reason,
            &tool_context,
            &tools,
            &hooks,
            &tx,
        )
        .await;
        tool_results.push((awaiting_tool.id.clone(), result));

        for pending in cont.pending_tool_calls.iter().skip(cont.awaiting_index + 1) {
            match execute_tool_call(pending, &tool_context, &tools, &hooks, &tx).await {
                ToolExecutionOutcome::Completed { tool_id, result } => {
                    tool_results.push((tool_id, result));
                }
                ToolExecutionOutcome::RequiresConfirmation {
                    tool_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                } => {
                    let pending_idx = cont
                        .pending_tool_calls
                        .iter()
                        .position(|p| p.id == tool_id)
                        .unwrap_or(0);

                    let new_continuation = AgentContinuation {
                        thread_id: thread_id.clone(),
                        turn,
                        total_usage: total_usage.clone(),
                        turn_usage: cont.turn_usage.clone(),
                        pending_tool_calls: cont.pending_tool_calls.clone(),
                        awaiting_index: pending_idx,
                        completed_results: tool_results,
                        state: state.clone(),
                    };

                    return AgentRunState::AwaitingConfirmation {
                        tool_call_id: tool_id,
                        tool_name,
                        display_name,
                        input,
                        description,
                        continuation: Box::new(new_continuation),
                    };
                }
            }
        }

        if let Err(e) = append_tool_results(&tool_results, &thread_id, &message_store).await {
            send_event(
                &tx,
                &hooks,
                AgentEvent::Error {
                    message: e.message.clone(),
                    recoverable: e.recoverable,
                },
            )
            .await;
            return AgentRunState::Error(e);
        }

        send_event(
            &tx,
            &hooks,
            AgentEvent::TurnComplete {
                turn,
                usage: cont.turn_usage.clone(),
            },
        )
        .await;
    }

    let mut ctx = TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
    };

    loop {
        let result = execute_turn(
            &tx,
            &mut ctx,
            &tool_context,
            &provider,
            &tools,
            &hooks,
            &message_store,
            &config,
            compaction_config.as_ref(),
            execution_store.as_ref(),
        )
        .await;

        match result {
            InternalTurnResult::Continue { .. } => {
                if let Err(e) = state_store.save(&ctx.state).await {
                    warn!(error = %e, "Failed to save state checkpoint");
                }
            }
            InternalTurnResult::Done => {
                break;
            }
            InternalTurnResult::AwaitingConfirmation {
                tool_call_id,
                tool_name,
                display_name,
                input,
                description,
                continuation,
            } => {
                return AgentRunState::AwaitingConfirmation {
                    tool_call_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                    continuation,
                };
            }
            InternalTurnResult::Error(e) => {
                return AgentRunState::Error(e);
            }
        }
    }

    if let Err(e) = state_store.save(&ctx.state).await {
        warn!(error = %e, "Failed to save final state");
    }

    let duration = ctx.start_time.elapsed();
    send_event(
        &tx,
        &hooks,
        AgentEvent::done(thread_id, ctx.turn, ctx.total_usage.clone(), duration),
    )
    .await;

    AgentRunState::Done {
        total_turns: u32::try_from(ctx.turn).unwrap_or(u32::MAX),
        input_tokens: u64::from(ctx.total_usage.input_tokens),
        output_tokens: u64::from(ctx.total_usage.output_tokens),
    }
}

struct TurnParameters<Ctx, P, H, M, S> {
    tx: mpsc::Sender<AgentEvent>,
    thread_id: ThreadId,
    input: AgentInput,
    tool_context: ToolContext<Ctx>,
    provider: Arc<P>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    message_store: Arc<M>,
    state_store: Arc<S>,
    config: AgentConfig,
    compaction_config: Option<CompactionConfig>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
}

/// Run a single turn of the agent loop.
///
/// This is similar to `run_loop` but only executes one turn and returns.
/// The caller is responsible for continuing execution by calling again with
/// `AgentInput::Continue`.
async fn run_single_turn<Ctx, P, H, M, S>(
    TurnParameters {
        tx,
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
        execution_store,
    }: TurnParameters<Ctx, P, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let tool_context = tool_context.with_event_tx(tx.clone());
    let start_time = Instant::now();

    let init_state =
        match initialize_from_input(input, &thread_id, &message_store, &state_store).await {
            Ok(s) => s,
            Err(e) => {
                send_event(&tx, &hooks, AgentEvent::error(&e.message, e.recoverable)).await;
                return TurnOutcome::Error(e);
            }
        };

    let InitializedState {
        turn,
        total_usage,
        state,
        resume_data,
    } = init_state;

    if let Some(resume_data_val) = resume_data {
        return handle_resume_case(ResumeCaseParameters {
            resume_data: resume_data_val,
            turn,
            total_usage,
            state,
            thread_id,
            tool_context,
            tools,
            hooks,
            tx,
            message_store,
            state_store,
        })
        .await;
    }

    let mut ctx = TurnContext {
        thread_id: thread_id.clone(),
        turn,
        total_usage,
        state,
        start_time,
    };

    let result = execute_turn(
        &tx,
        &mut ctx,
        &tool_context,
        &provider,
        &tools,
        &hooks,
        &message_store,
        &config,
        compaction_config.as_ref(),
        execution_store.as_ref(),
    )
    .await;

    // Convert InternalTurnResult to TurnOutcome
    match result {
        InternalTurnResult::Continue { turn_usage } => {
            // Save state checkpoint
            if let Err(e) = state_store.save(&ctx.state).await {
                warn!(error = %e, "Failed to save state checkpoint");
            }

            TurnOutcome::NeedsMoreTurns {
                turn: ctx.turn,
                turn_usage,
                total_usage: ctx.total_usage,
            }
        }
        InternalTurnResult::Done => {
            // Final state save
            if let Err(e) = state_store.save(&ctx.state).await {
                warn!(error = %e, "Failed to save final state");
            }

            // Emit done
            let duration = ctx.start_time.elapsed();
            send_event(
                &tx,
                &hooks,
                AgentEvent::done(thread_id, ctx.turn, ctx.total_usage.clone(), duration),
            )
            .await;

            TurnOutcome::Done {
                total_turns: u32::try_from(ctx.turn).unwrap_or(u32::MAX),
                input_tokens: u64::from(ctx.total_usage.input_tokens),
                output_tokens: u64::from(ctx.total_usage.output_tokens),
            }
        }
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

struct ResumeCaseParameters<Ctx, H, M, S> {
    resume_data: ResumeData,
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    thread_id: ThreadId,
    tool_context: ToolContext<Ctx>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    tx: mpsc::Sender<AgentEvent>,
    message_store: Arc<M>,
    state_store: Arc<S>,
}

async fn handle_resume_case<Ctx, H, M, S>(
    ResumeCaseParameters {
        resume_data,
        turn,
        total_usage,
        state,
        thread_id,
        tool_context,
        tools,
        hooks,
        tx,
        message_store,
        state_store,
    }: ResumeCaseParameters<Ctx, H, M, S>,
) -> TurnOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let ResumeData {
        continuation: cont,
        tool_call_id,
        confirmed,
        rejection_reason,
    } = resume_data;
    // Handle resume case - complete the pending tool calls
    let mut tool_results = cont.completed_results.clone();
    let awaiting_tool = &cont.pending_tool_calls[cont.awaiting_index];

    // Validate tool_call_id matches
    if awaiting_tool.id != tool_call_id {
        let message = format!(
            "Tool call ID mismatch: expected {}, got {}",
            awaiting_tool.id, tool_call_id
        );
        let recoverable = false;
        send_event(&tx, &hooks, AgentEvent::error(&message, recoverable)).await;
        return TurnOutcome::Error(AgentError::new(&message, recoverable));
    }

    let result = execute_confirmed_tool(
        awaiting_tool,
        confirmed,
        rejection_reason,
        &tool_context,
        &tools,
        &hooks,
        &tx,
    )
    .await;
    tool_results.push((awaiting_tool.id.clone(), result));

    for pending in cont.pending_tool_calls.iter().skip(cont.awaiting_index + 1) {
        match execute_tool_call(pending, &tool_context, &tools, &hooks, &tx).await {
            ToolExecutionOutcome::Completed { tool_id, result } => {
                tool_results.push((tool_id, result));
            }
            ToolExecutionOutcome::RequiresConfirmation {
                tool_id,
                tool_name,
                display_name,
                input,
                description,
            } => {
                let pending_idx = cont
                    .pending_tool_calls
                    .iter()
                    .position(|p| p.id == tool_id)
                    .unwrap_or(0);

                let new_continuation = AgentContinuation {
                    thread_id: thread_id.clone(),
                    turn,
                    total_usage: total_usage.clone(),
                    turn_usage: cont.turn_usage.clone(),
                    pending_tool_calls: cont.pending_tool_calls.clone(),
                    awaiting_index: pending_idx,
                    completed_results: tool_results,
                    state: state.clone(),
                };

                return TurnOutcome::AwaitingConfirmation {
                    tool_call_id: tool_id,
                    tool_name,
                    display_name,
                    input,
                    description,
                    continuation: Box::new(new_continuation),
                };
            }
        }
    }

    if let Err(e) = append_tool_results(&tool_results, &thread_id, &message_store).await {
        send_event(&tx, &hooks, AgentEvent::error(&e.message, e.recoverable)).await;
        return TurnOutcome::Error(e);
    }

    send_event(
        &tx,
        &hooks,
        AgentEvent::TurnComplete {
            turn,
            usage: cont.turn_usage.clone(),
        },
    )
    .await;

    let mut updated_state = state;
    updated_state.turn_count = turn;
    if let Err(e) = state_store.save(&updated_state).await {
        warn!(error = %e, "Failed to save state checkpoint");
    }

    TurnOutcome::NeedsMoreTurns {
        turn,
        turn_usage: cont.turn_usage.clone(),
        total_usage,
    }
}

// =============================================================================
// Tool Execution Idempotency Helpers
// =============================================================================

/// Check for an existing completed execution and return cached result.
///
/// Returns `Some(result)` if the execution was completed, `None` if not found
/// or still in-flight.
async fn try_get_cached_result(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    tool_call_id: &str,
) -> Option<ToolResult> {
    let store = execution_store?;
    let execution = store.get_execution(tool_call_id).await.ok()??;

    match execution.status {
        ExecutionStatus::Completed => execution.result,
        ExecutionStatus::InFlight => {
            // Log warning that we found an in-flight execution
            // This means a previous attempt crashed mid-execution
            warn!(
                tool_call_id = tool_call_id,
                tool_name = execution.tool_name,
                "Found in-flight execution from previous attempt, re-executing"
            );
            None
        }
    }
}

/// Record that we're about to start executing a tool (write-ahead).
async fn record_execution_start(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    started_at: time::OffsetDateTime,
) {
    if let Some(store) = execution_store {
        let execution = ToolExecution::new_in_flight(
            &pending.id,
            thread_id.clone(),
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            started_at,
        );
        if let Err(e) = store.record_execution(execution).await {
            warn!(
                tool_call_id = pending.id,
                error = %e,
                "Failed to record execution start"
            );
        }
    }
}

/// Record that tool execution completed.
async fn record_execution_complete(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    result: &ToolResult,
    started_at: time::OffsetDateTime,
) {
    if let Some(store) = execution_store {
        let mut execution = ToolExecution::new_in_flight(
            &pending.id,
            thread_id.clone(),
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            started_at,
        );
        execution.complete(result.clone());
        if let Err(e) = store.update_execution(execution).await {
            warn!(
                tool_call_id = pending.id,
                error = %e,
                "Failed to record execution completion"
            );
        }
    }
}

/// Execute a single turn of the agent loop.
///
/// This is the core turn execution logic shared by both `run_loop` (looping mode)
/// and `run_single_turn` (single-turn mode).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn execute_turn<Ctx, P, H, M>(
    tx: &mpsc::Sender<AgentEvent>,
    ctx: &mut TurnContext,
    tool_context: &ToolContext<Ctx>,
    provider: &Arc<P>,
    tools: &Arc<ToolRegistry<Ctx>>,
    hooks: &Arc<H>,
    message_store: &Arc<M>,
    config: &AgentConfig,
    compaction_config: Option<&CompactionConfig>,
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
) -> InternalTurnResult
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
{
    ctx.turn += 1;
    ctx.state.turn_count = ctx.turn;

    if ctx.turn > config.max_turns {
        warn!(turn = ctx.turn, max = config.max_turns, "Max turns reached");
        send_event(
            tx,
            hooks,
            AgentEvent::error(
                format!("Maximum turns ({}) reached", config.max_turns),
                true,
            ),
        )
        .await;
        return InternalTurnResult::Error(AgentError::new(
            format!("Maximum turns ({}) reached", config.max_turns),
            true,
        ));
    }

    // Emit start event
    send_event(
        tx,
        hooks,
        AgentEvent::start(ctx.thread_id.clone(), ctx.turn),
    )
    .await;

    // Get message history
    let mut messages = match message_store.get_history(&ctx.thread_id).await {
        Ok(m) => m,
        Err(e) => {
            send_event(
                tx,
                hooks,
                AgentEvent::error(format!("Failed to get history: {e}"), false),
            )
            .await;
            return InternalTurnResult::Error(AgentError::new(
                format!("Failed to get history: {e}"),
                false,
            ));
        }
    };

    // Check if compaction is needed
    if let Some(compact_config) = compaction_config {
        let compactor = LlmContextCompactor::new(Arc::clone(provider), compact_config.clone());
        if compactor.needs_compaction(&messages) {
            debug!(
                turn = ctx.turn,
                message_count = messages.len(),
                "Context compaction triggered"
            );

            match compactor.compact_history(messages.clone()).await {
                Ok(result) => {
                    if let Err(e) = message_store
                        .replace_history(&ctx.thread_id, result.messages.clone())
                        .await
                    {
                        warn!(error = %e, "Failed to replace history after compaction");
                    } else {
                        send_event(
                            tx,
                            hooks,
                            AgentEvent::context_compacted(
                                result.original_count,
                                result.new_count,
                                result.original_tokens,
                                result.new_tokens,
                            ),
                        )
                        .await;

                        info!(
                            original_count = result.original_count,
                            new_count = result.new_count,
                            original_tokens = result.original_tokens,
                            new_tokens = result.new_tokens,
                            "Context compacted successfully"
                        );

                        messages = result.messages;
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Context compaction failed, continuing with full history");
                }
            }
        }
    }

    // Build chat request
    let llm_tools = if tools.is_empty() {
        None
    } else {
        Some(tools.to_llm_tools())
    };

    let request = ChatRequest {
        system: config.system_prompt.clone(),
        messages,
        tools: llm_tools,
        max_tokens: config.max_tokens,
        thinking: config.thinking.clone(),
    };

    // Call LLM with retry logic (streaming or non-streaming based on config)
    debug!(turn = ctx.turn, streaming = config.streaming, "Calling LLM");
    let response = if config.streaming {
        // Streaming mode: events are emitted as content arrives
        match call_llm_streaming(provider, request, config, tx, hooks).await {
            Ok(r) => r,
            Err(e) => {
                return InternalTurnResult::Error(e);
            }
        }
    } else {
        // Non-streaming mode: wait for full response
        match call_llm_with_retry(provider, request, config, tx, hooks).await {
            Ok(r) => r,
            Err(e) => {
                return InternalTurnResult::Error(e);
            }
        }
    };

    // Track usage
    let turn_usage = TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    };
    ctx.total_usage.add(&turn_usage);
    ctx.state.total_usage = ctx.total_usage.clone();

    // Process response content
    let (thinking_content, text_content, tool_uses) = extract_content(&response);

    // Emit events only in non-streaming mode (streaming already emitted deltas)
    if !config.streaming {
        // Emit thinking if present (before text)
        if let Some(thinking) = &thinking_content {
            send_event(tx, hooks, AgentEvent::thinking(thinking.clone())).await;
        }

        // Emit text if present
        if let Some(text) = &text_content {
            send_event(tx, hooks, AgentEvent::text(text.clone())).await;
        }
    }

    // If no tool uses, we're done
    if tool_uses.is_empty() {
        info!(turn = ctx.turn, "Agent completed (no tool use)");
        return InternalTurnResult::Done;
    }

    // Store assistant message with tool uses
    let assistant_msg = build_assistant_message(&response);
    if let Err(e) = message_store.append(&ctx.thread_id, assistant_msg).await {
        send_event(
            tx,
            hooks,
            AgentEvent::error(format!("Failed to append assistant message: {e}"), false),
        )
        .await;
        return InternalTurnResult::Error(AgentError::new(
            format!("Failed to append assistant message: {e}"),
            false,
        ));
    }

    // Build pending tool calls (check both sync and async tools for display_name)
    let pending_tool_calls: Vec<PendingToolCallInfo> = tool_uses
        .iter()
        .map(|(id, name, input)| {
            let display_name = tools
                .get(name)
                .map(|t| t.display_name().to_string())
                .or_else(|| tools.get_async(name).map(|t| t.display_name().to_string()))
                .unwrap_or_default();
            PendingToolCallInfo {
                id: id.clone(),
                name: name.clone(),
                display_name,
                input: input.clone(),
            }
        })
        .collect();

    // Execute tools (supports both sync and async tools)
    let mut tool_results = Vec::new();
    for (idx, pending) in pending_tool_calls.iter().enumerate() {
        // IDEMPOTENCY: Check for cached result from a previous execution attempt
        if let Some(cached_result) = try_get_cached_result(execution_store, &pending.id).await {
            debug!(
                tool_call_id = pending.id,
                tool_name = pending.name,
                "Using cached result from previous execution"
            );
            tool_results.push((pending.id.clone(), cached_result));
            continue;
        }

        // Check for async tool first
        if let Some(async_tool) = tools.get_async(&pending.name) {
            let tier = async_tool.tier();

            // Emit tool call start
            send_event(
                tx,
                hooks,
                AgentEvent::tool_call_start(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    pending.input.clone(),
                    tier,
                ),
            )
            .await;

            // Check hooks for permission
            let decision = hooks
                .pre_tool_use(&pending.name, &pending.input, tier)
                .await;

            match decision {
                ToolDecision::Allow => {
                    // IDEMPOTENCY: Record execution start (write-ahead)
                    let started_at = time::OffsetDateTime::now_utc();
                    record_execution_start(execution_store, pending, &ctx.thread_id, started_at)
                        .await;

                    let result = execute_async_tool(pending, async_tool, tool_context, tx).await;

                    // IDEMPOTENCY: Record execution completion
                    record_execution_complete(
                        execution_store,
                        pending,
                        &ctx.thread_id,
                        &result,
                        started_at,
                    )
                    .await;

                    hooks.post_tool_use(&pending.name, &result).await;

                    send_event(
                        tx,
                        hooks,
                        AgentEvent::tool_call_end(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            result.clone(),
                        ),
                    )
                    .await;

                    tool_results.push((pending.id.clone(), result));
                }
                ToolDecision::Block(reason) => {
                    let result = ToolResult::error(format!("Blocked: {reason}"));
                    send_event(
                        tx,
                        hooks,
                        AgentEvent::tool_call_end(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            result.clone(),
                        ),
                    )
                    .await;
                    tool_results.push((pending.id.clone(), result));
                }
                ToolDecision::RequiresConfirmation(description) => {
                    // Emit event and yield
                    send_event(
                        tx,
                        hooks,
                        AgentEvent::ToolRequiresConfirmation {
                            id: pending.id.clone(),
                            name: pending.name.clone(),
                            input: pending.input.clone(),
                            description: description.clone(),
                        },
                    )
                    .await;

                    let continuation = AgentContinuation {
                        thread_id: ctx.thread_id.clone(),
                        turn: ctx.turn,
                        total_usage: ctx.total_usage.clone(),
                        turn_usage: turn_usage.clone(),
                        pending_tool_calls: pending_tool_calls.clone(),
                        awaiting_index: idx,
                        completed_results: tool_results,
                        state: ctx.state.clone(),
                    };

                    return InternalTurnResult::AwaitingConfirmation {
                        tool_call_id: pending.id.clone(),
                        tool_name: pending.name.clone(),
                        display_name: pending.display_name.clone(),
                        input: pending.input.clone(),
                        description,
                        continuation: Box::new(continuation),
                    };
                }
            }
            continue;
        }

        // Fall back to sync tool
        let Some(tool) = tools.get(&pending.name) else {
            let result = ToolResult::error(format!("Unknown tool: {}", pending.name));
            tool_results.push((pending.id.clone(), result));
            continue;
        };

        let tier = tool.tier();

        // Emit tool call start
        send_event(
            tx,
            hooks,
            AgentEvent::tool_call_start(
                &pending.id,
                &pending.name,
                &pending.display_name,
                pending.input.clone(),
                tier,
            ),
        )
        .await;

        // Check hooks for permission
        let decision = hooks
            .pre_tool_use(&pending.name, &pending.input, tier)
            .await;

        match decision {
            ToolDecision::Allow => {
                // IDEMPOTENCY: Record execution start (write-ahead)
                let started_at = time::OffsetDateTime::now_utc();
                record_execution_start(execution_store, pending, &ctx.thread_id, started_at).await;

                let tool_start = Instant::now();
                let result = match tool.execute(tool_context, pending.input.clone()).await {
                    Ok(mut r) => {
                        r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        r
                    }
                    Err(e) => ToolResult::error(format!("Tool error: {e}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                };

                // IDEMPOTENCY: Record execution completion
                record_execution_complete(
                    execution_store,
                    pending,
                    &ctx.thread_id,
                    &result,
                    started_at,
                )
                .await;

                hooks.post_tool_use(&pending.name, &result).await;

                send_event(
                    tx,
                    hooks,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;

                tool_results.push((pending.id.clone(), result));
            }
            ToolDecision::Block(reason) => {
                let result = ToolResult::error(format!("Blocked: {reason}"));
                send_event(
                    tx,
                    hooks,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;
                tool_results.push((pending.id.clone(), result));
            }
            ToolDecision::RequiresConfirmation(description) => {
                // Emit event and yield
                send_event(
                    tx,
                    hooks,
                    AgentEvent::ToolRequiresConfirmation {
                        id: pending.id.clone(),
                        name: pending.name.clone(),
                        input: pending.input.clone(),
                        description: description.clone(),
                    },
                )
                .await;

                let continuation = AgentContinuation {
                    thread_id: ctx.thread_id.clone(),
                    turn: ctx.turn,
                    total_usage: ctx.total_usage.clone(),
                    turn_usage: turn_usage.clone(),
                    pending_tool_calls: pending_tool_calls.clone(),
                    awaiting_index: idx,
                    completed_results: tool_results,
                    state: ctx.state.clone(),
                };

                return InternalTurnResult::AwaitingConfirmation {
                    tool_call_id: pending.id.clone(),
                    tool_name: pending.name.clone(),
                    display_name: pending.display_name.clone(),
                    input: pending.input.clone(),
                    description,
                    continuation: Box::new(continuation),
                };
            }
        }
    }

    // Add tool results to message history
    if let Err(e) = append_tool_results(&tool_results, &ctx.thread_id, message_store).await {
        send_event(
            tx,
            hooks,
            AgentEvent::error(format!("Failed to append tool results: {e}"), false),
        )
        .await;
        return InternalTurnResult::Error(e);
    }

    // Emit turn complete
    send_event(
        tx,
        hooks,
        AgentEvent::TurnComplete {
            turn: ctx.turn,
            usage: turn_usage.clone(),
        },
    )
    .await;

    // Check stop reason
    if response.stop_reason == Some(StopReason::EndTurn) {
        info!(turn = ctx.turn, "Agent completed (end_turn)");
        return InternalTurnResult::Done;
    }

    InternalTurnResult::Continue { turn_usage }
}

/// Convert u128 milliseconds to u64, capping at `u64::MAX`
#[allow(clippy::cast_possible_truncation)]
const fn millis_to_u64(millis: u128) -> u64 {
    if millis > u64::MAX as u128 {
        u64::MAX
    } else {
        millis as u64
    }
}

/// Calculate exponential backoff delay with jitter.
///
/// Uses exponential backoff with the formula: `base * 2^(attempt-1) + jitter`,
/// capped at the maximum delay. Jitter (0-1000ms) helps avoid thundering herd.
fn calculate_backoff_delay(attempt: u32, config: &RetryConfig) -> Duration {
    // Exponential backoff: base, base*2, base*4, base*8, ...
    let base_delay = config
        .base_delay_ms
        .saturating_mul(1u64 << (attempt.saturating_sub(1)));

    // Add jitter (0-1000ms or 10% of base, whichever is smaller) to avoid thundering herd
    let max_jitter = config.base_delay_ms.min(1000);
    let jitter = if max_jitter > 0 {
        u64::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos(),
        ) % max_jitter
    } else {
        0
    };

    let delay_ms = base_delay.saturating_add(jitter).min(config.max_delay_ms);
    Duration::from_millis(delay_ms)
}

/// Extracted content from an LLM response: (thinking, text, `tool_uses`).
type ExtractedContent = (
    Option<String>,
    Option<String>,
    Vec<(String, String, serde_json::Value)>,
);

/// Extract content from an LLM response.
fn extract_content(response: &ChatResponse) -> ExtractedContent {
    let mut thinking_parts = Vec::new();
    let mut text_parts = Vec::new();
    let mut tool_uses = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::Thinking { thinking } => {
                thinking_parts.push(thinking.clone());
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_uses.push((id.clone(), name.clone(), input.clone()));
            }
            ContentBlock::ToolResult { .. } => {
                // Shouldn't appear in response, but ignore if it does
            }
        }
    }

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n"))
    };

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };

    (thinking, text, tool_uses)
}

async fn send_event<H>(tx: &mpsc::Sender<AgentEvent>, hooks: &Arc<H>, event: AgentEvent)
where
    H: AgentHooks,
{
    hooks.on_event(&event).await;
    let _ = tx.send(event).await;
}

fn build_assistant_message(response: &ChatResponse) -> Message {
    let mut blocks = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                blocks.push(ContentBlock::Text { text: text.clone() });
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolResult { .. } => {
                // Thinking blocks are ephemeral - not stored in conversation history
                // ToolResult shouldn't appear in response, but ignore if it does
            }
            ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature,
            } => {
                blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    thought_signature: thought_signature.clone(),
                });
            }
        }
    }

    Message {
        role: Role::Assistant,
        content: Content::Blocks(blocks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::AllowAllHooks;
    use crate::llm::{ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage};
    use crate::stores::InMemoryStore;
    use crate::tools::{Tool, ToolContext, ToolRegistry};
    use crate::types::{AgentConfig, AgentInput, ToolResult, ToolTier};
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::RwLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ===================
    // Mock LLM Provider
    // ===================

    struct MockProvider {
        responses: RwLock<Vec<ChatOutcome>>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatOutcome>) -> Self {
            Self {
                responses: RwLock::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }

        fn text_response(text: &str) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "msg_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                model: "mock-model".to_string(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
            })
        }

        fn tool_use_response(
            tool_id: &str,
            tool_name: &str,
            input: serde_json::Value,
        ) -> ChatOutcome {
            ChatOutcome::Success(ChatResponse {
                id: "msg_1".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: tool_id.to_string(),
                    name: tool_name.to_string(),
                    input,
                    thought_signature: None,
                }],
                model: "mock-model".to_string(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
            })
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let responses = self.responses.read().unwrap();
            if idx < responses.len() {
                Ok(responses[idx].clone())
            } else {
                // Default: end conversation
                Ok(Self::text_response("Done"))
            }
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    // Make ChatOutcome clonable for tests
    impl Clone for ChatOutcome {
        fn clone(&self) -> Self {
            match self {
                Self::Success(r) => Self::Success(r.clone()),
                Self::RateLimited => Self::RateLimited,
                Self::InvalidRequest(s) => Self::InvalidRequest(s.clone()),
                Self::ServerError(s) => Self::ServerError(s.clone()),
            }
        }
    }

    // ===================
    // Mock Tool
    // ===================

    struct EchoTool;

    // Test tool name enum for tests
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum TestToolName {
        Echo,
    }

    impl crate::tools::ToolName for TestToolName {}

    impl Tool<()> for EchoTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::Echo
        }

        fn display_name(&self) -> &'static str {
            "Echo"
        }

        fn description(&self) -> &'static str {
            "Echo the input message"
        }

        fn input_schema(&self) -> serde_json::Value {
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

        async fn execute(
            &self,
            _ctx: &ToolContext<()>,
            input: serde_json::Value,
        ) -> Result<ToolResult> {
            let message = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("no message");
            Ok(ToolResult::success(format!("Echo: {message}")))
        }
    }

    // ===================
    // Builder Tests
    // ===================

    #[test]
    fn test_builder_creates_agent_loop() {
        let provider = MockProvider::new(vec![]);
        let agent = builder::<()>().provider(provider).build();

        assert_eq!(agent.config.max_turns, 10);
        assert_eq!(agent.config.max_tokens, 4096);
    }

    #[test]
    fn test_builder_with_custom_config() {
        let provider = MockProvider::new(vec![]);
        let config = AgentConfig {
            max_turns: 5,
            max_tokens: 2048,
            system_prompt: "Custom prompt".to_string(),
            model: "custom-model".to_string(),
            ..Default::default()
        };

        let agent = builder::<()>().provider(provider).config(config).build();

        assert_eq!(agent.config.max_turns, 5);
        assert_eq!(agent.config.max_tokens, 2048);
        assert_eq!(agent.config.system_prompt, "Custom prompt");
    }

    #[test]
    fn test_builder_with_tools() {
        let provider = MockProvider::new(vec![]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let agent = builder::<()>().provider(provider).tools(tools).build();

        assert_eq!(agent.tools.len(), 1);
    }

    #[test]
    fn test_builder_with_custom_stores() {
        let provider = MockProvider::new(vec![]);
        let message_store = InMemoryStore::new();
        let state_store = InMemoryStore::new();

        let agent = builder::<()>()
            .provider(provider)
            .hooks(AllowAllHooks)
            .message_store(message_store)
            .state_store(state_store)
            .build_with_stores();

        // Just verify it builds without panicking
        assert_eq!(agent.config.max_turns, 10);
    }

    // ===================
    // Run Loop Tests
    // ===================

    #[tokio::test]
    async fn test_simple_text_response() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello, user!")]);

        let agent = builder::<()>().provider(provider).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have: Start, Text, Done
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Text { .. })));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Done { .. })));

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_execution() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            // First call: request tool use
            MockProvider::tool_use_response("tool_1", "echo", json!({"message": "test"})),
            // Second call: respond with text
            MockProvider::text_response("Tool executed successfully"),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let agent = builder::<()>().provider(provider).tools(tools).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) = agent.run(
            thread_id,
            AgentInput::Text("Run echo".to_string()),
            tool_ctx,
        );

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have tool call events
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCallStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCallEnd { .. }))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_max_turns_limit() -> anyhow::Result<()> {
        // Provider that always requests a tool
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("tool_1", "echo", json!({"message": "1"})),
            MockProvider::tool_use_response("tool_2", "echo", json!({"message": "2"})),
            MockProvider::tool_use_response("tool_3", "echo", json!({"message": "3"})),
            MockProvider::tool_use_response("tool_4", "echo", json!({"message": "4"})),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let config = AgentConfig {
            max_turns: 2,
            ..Default::default()
        };

        let agent = builder::<()>()
            .provider(provider)
            .tools(tools)
            .config(config)
            .build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Loop".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have an error about max turns
        assert!(events.iter().any(|e| {
            matches!(e, AgentEvent::Error { message, .. } if message.contains("Maximum turns"))
        }));

        Ok(())
    }

    #[tokio::test]
    async fn test_unknown_tool_handling() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            // Request unknown tool
            MockProvider::tool_use_response("tool_1", "nonexistent_tool", json!({})),
            // LLM gets tool error and ends conversation
            MockProvider::text_response("I couldn't find that tool."),
        ]);

        // Empty tool registry
        let tools = ToolRegistry::new();

        let agent = builder::<()>().provider(provider).tools(tools).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) = agent.run(
            thread_id,
            AgentInput::Text("Call unknown".to_string()),
            tool_ctx,
        );

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Unknown tool errors are returned to the LLM (not emitted as ToolCallEnd)
        // The conversation should complete successfully with a Done event
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Done { .. })));

        // The LLM's response about the missing tool should be in the events
        assert!(
            events.iter().any(|e| {
                matches!(e, AgentEvent::Text { text } if text.contains("couldn't find"))
            })
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rate_limit_handling() -> anyhow::Result<()> {
        // Provide enough RateLimited responses to exhaust all retries (max_retries + 1)
        let provider = MockProvider::new(vec![
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited, // 6th attempt exceeds max_retries (5)
        ]);

        // Use fast retry config for faster tests
        let config = AgentConfig {
            retry: crate::types::RetryConfig::fast(),
            ..Default::default()
        };

        let agent = builder::<()>().provider(provider).config(config).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have rate limit error after exhausting retries
        assert!(events.iter().any(|e| {
            matches!(e, AgentEvent::Error { message, recoverable: true } if message.contains("Rate limited"))
        }));

        // Should have retry text events
        assert!(
            events
                .iter()
                .any(|e| { matches!(e, AgentEvent::Text { text } if text.contains("retrying")) })
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rate_limit_recovery() -> anyhow::Result<()> {
        // Rate limited once, then succeeds
        let provider = MockProvider::new(vec![
            ChatOutcome::RateLimited,
            MockProvider::text_response("Recovered after rate limit"),
        ]);

        // Use fast retry config for faster tests
        let config = AgentConfig {
            retry: crate::types::RetryConfig::fast(),
            ..Default::default()
        };

        let agent = builder::<()>().provider(provider).config(config).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have successful completion after retry
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Done { .. })));

        // Should have retry text event
        assert!(
            events
                .iter()
                .any(|e| { matches!(e, AgentEvent::Text { text } if text.contains("retrying")) })
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_server_error_handling() -> anyhow::Result<()> {
        // Provide enough ServerError responses to exhaust all retries (max_retries + 1)
        let provider = MockProvider::new(vec![
            ChatOutcome::ServerError("Internal error".to_string()),
            ChatOutcome::ServerError("Internal error".to_string()),
            ChatOutcome::ServerError("Internal error".to_string()),
            ChatOutcome::ServerError("Internal error".to_string()),
            ChatOutcome::ServerError("Internal error".to_string()),
            ChatOutcome::ServerError("Internal error".to_string()), // 6th attempt exceeds max_retries
        ]);

        // Use fast retry config for faster tests
        let config = AgentConfig {
            retry: crate::types::RetryConfig::fast(),
            ..Default::default()
        };

        let agent = builder::<()>().provider(provider).config(config).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have server error after exhausting retries
        assert!(events.iter().any(|e| {
            matches!(e, AgentEvent::Error { message, recoverable: true } if message.contains("Server error"))
        }));

        // Should have retry text events
        assert!(
            events
                .iter()
                .any(|e| { matches!(e, AgentEvent::Text { text } if text.contains("retrying")) })
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_server_error_recovery() -> anyhow::Result<()> {
        // Server error once, then succeeds
        let provider = MockProvider::new(vec![
            ChatOutcome::ServerError("Temporary error".to_string()),
            MockProvider::text_response("Recovered after server error"),
        ]);

        // Use fast retry config for faster tests
        let config = AgentConfig {
            retry: crate::types::RetryConfig::fast(),
            ..Default::default()
        };

        let agent = builder::<()>().provider(provider).config(config).build();

        let thread_id = ThreadId::new();
        let tool_ctx = ToolContext::new(());
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have successful completion after retry
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Done { .. })));

        // Should have retry text event
        assert!(
            events
                .iter()
                .any(|e| { matches!(e, AgentEvent::Text { text } if text.contains("retrying")) })
        );

        Ok(())
    }

    // ===================
    // Helper Function Tests
    // ===================

    #[test]
    fn test_extract_content_text_only() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert_eq!(text, Some("Hello".to_string()));
        assert!(tool_uses.is_empty());
    }

    #[test]
    fn test_extract_content_tool_use() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "test_tool".to_string(),
                input: json!({"key": "value"}),
                thought_signature: None,
            }],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert!(text.is_none());
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].1, "test_tool");
    }

    #[test]
    fn test_extract_content_mixed() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me help".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "helper".to_string(),
                    input: json!({}),
                    thought_signature: None,
                },
            ],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert_eq!(text, Some("Let me help".to_string()));
        assert_eq!(tool_uses.len(), 1);
    }

    #[test]
    fn test_millis_to_u64() {
        assert_eq!(millis_to_u64(0), 0);
        assert_eq!(millis_to_u64(1000), 1000);
        assert_eq!(millis_to_u64(u128::from(u64::MAX)), u64::MAX);
        assert_eq!(millis_to_u64(u128::from(u64::MAX) + 1), u64::MAX);
    }

    #[test]
    fn test_build_assistant_message() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Response text".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "echo".to_string(),
                    input: json!({"message": "test"}),
                    thought_signature: None,
                },
            ],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };

        let msg = build_assistant_message(&response);
        assert_eq!(msg.role, Role::Assistant);

        if let Content::Blocks(blocks) = msg.content {
            assert_eq!(blocks.len(), 2);
        } else {
            panic!("Expected Content::Blocks");
        }
    }
}
