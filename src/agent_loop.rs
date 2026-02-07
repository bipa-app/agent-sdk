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
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::{AgentHooks, DefaultHooks, ToolDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamAccumulator, StreamDelta, Usage,
};
use crate::skills::Skill;
use crate::stores::{InMemoryStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{
    DeferredToolState, ErasedAsyncTool, ErasedDeferredTool, ErasedListenTool, ErasedToolStatus,
    ListenStopReason, ListenToolUpdate, ToolContext, ToolRegistry,
};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    ExecutionStatus, ListenExecutionContext, PendingToolCallInfo, RetryConfig, ThreadId,
    TokenUsage, ToolExecution, ToolOutcome, ToolResult, TurnOutcome,
};
use futures::StreamExt;
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;

/// Internal result of executing a single turn.
///
/// This is used internally by both `run_loop` and `run_single_turn`.
enum InternalTurnResult {
    /// Turn completed, more turns needed (tools were executed)
    Continue { turn_usage: TokenUsage },
    /// Done - no more tool calls
    Done,
    /// Model refused the request (safety/policy)
    Refusal,
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
        listen_context: Option<ListenExecutionContext>,
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
/// # Event Channel Behavior
///
/// The agent uses a bounded channel (capacity 100) for events. Events are sent
/// using non-blocking sends:
///
/// - If the channel has space, events are sent immediately
/// - If the channel is full, the agent waits up to 30 seconds before timing out
/// - If the receiver is dropped, the agent continues processing without blocking
///
/// This design ensures that slow consumers don't stall the LLM stream, but events
/// may be dropped if the consumer is too slow or disconnects.
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
    ) -> (
        mpsc::Receiver<AgentEventEnvelope>,
        oneshot::Receiver<AgentRunState>,
    )
    where
        Ctx: Clone,
    {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (state_tx, state_rx) = oneshot::channel();
        let seq = SequenceCounter::new();

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
    pub fn run_turn(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
    ) -> (
        mpsc::Receiver<AgentEventEnvelope>,
        oneshot::Receiver<TurnOutcome>,
    )
    where
        Ctx: Clone,
    {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (outcome_tx, outcome_rx) = oneshot::channel();
        let seq = SequenceCounter::new();

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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
) -> ToolExecutionOutcome
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    const MAX_LISTEN_UPDATES: usize = 240;
    const MAX_DEFERRED_POLLS: usize = 240;
    const MIN_DEFERRED_POLL_MS: u64 = 50;
    const MAX_DEFERRED_POLL_MS: u64 = 30_000;

    struct ListenReady {
        operation_id: String,
        revision: u64,
        snapshot: serde_json::Value,
        expires_at: Option<String>,
    }

    fn build_listen_confirmation_input(
        original_input: &serde_json::Value,
        ready: &ListenReady,
    ) -> serde_json::Value {
        serde_json::json!({
            "requested_input": original_input,
            "prepared_snapshot": ready.snapshot,
            "operation_id": ready.operation_id,
            "revision": ready.revision,
            "expires_at": ready.expires_at,
        })
    }

    async fn wait_for_listen_ready<Ctx>(
        pending: &PendingToolCallInfo,
        tool: &Arc<dyn ErasedListenTool<Ctx>>,
        tool_context: &ToolContext<Ctx>,
        tx: &mpsc::Sender<AgentEventEnvelope>,
        seq: &SequenceCounter,
    ) -> Result<ListenReady, ToolResult>
    where
        Ctx: Send + Sync + Clone + 'static,
    {
        let mut updates = tool.listen_stream(tool_context, pending.input.clone());
        let mut update_count = 0usize;
        let mut last_operation_id: Option<String> = None;

        while let Some(update) = updates.next().await {
            update_count += 1;

            match update {
                ListenToolUpdate::Listening {
                    operation_id,
                    revision,
                    message,
                    snapshot,
                    expires_at,
                } => {
                    last_operation_id = Some(operation_id.clone());

                    let data = Some(serde_json::json!({
                        "operation_id": operation_id,
                        "revision": revision,
                        "snapshot": snapshot,
                        "expires_at": expires_at,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "listen_update",
                            message,
                            data,
                        ),
                        seq,
                    )
                    .await;
                }
                ListenToolUpdate::Ready {
                    operation_id,
                    revision,
                    message,
                    snapshot,
                    expires_at,
                } => {
                    let data = Some(serde_json::json!({
                        "operation_id": operation_id,
                        "revision": revision,
                        "snapshot": snapshot,
                        "expires_at": expires_at,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "listen_ready",
                            message,
                            data,
                        ),
                        seq,
                    )
                    .await;

                    return Ok(ListenReady {
                        operation_id,
                        revision,
                        snapshot,
                        expires_at,
                    });
                }
                ListenToolUpdate::Invalidated {
                    operation_id,
                    message,
                    recoverable,
                } => {
                    let data = Some(serde_json::json!({
                        "operation_id": operation_id,
                        "recoverable": recoverable,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "listen_invalidated",
                            message.clone(),
                            data,
                        ),
                        seq,
                    )
                    .await;

                    let prefix = if recoverable {
                        "Listen operation invalidated (recoverable)"
                    } else {
                        "Listen operation invalidated"
                    };
                    return Err(ToolResult::error(format!("{prefix}: {message}")));
                }
            }

            if tx.is_closed() {
                if let Some(operation_id) = last_operation_id.as_deref() {
                    let _ = tool
                        .cancel(
                            tool_context,
                            operation_id,
                            ListenStopReason::StreamDisconnected,
                        )
                        .await;
                }
                return Err(ToolResult::error(
                    "Listen stream disconnected before operation became ready",
                ));
            }

            if update_count >= MAX_LISTEN_UPDATES {
                if let Some(operation_id) = last_operation_id.as_deref() {
                    let _ = tool
                        .cancel(tool_context, operation_id, ListenStopReason::StreamEnded)
                        .await;
                }
                return Err(ToolResult::error(format!(
                    "Listen tool exceeded max updates ({MAX_LISTEN_UPDATES})"
                )));
            }
        }

        if let Some(operation_id) = last_operation_id.as_deref() {
            let _ = tool
                .cancel(tool_context, operation_id, ListenStopReason::StreamEnded)
                .await;
        }

        Err(ToolResult::error(
            "Listen stream ended before operation became ready",
        ))
    }

    struct DeferredReady {
        session_id: String,
        revision: u64,
        snapshot: serde_json::Value,
        expires_at: Option<String>,
    }

    fn build_deferred_confirmation_input(
        original_input: &serde_json::Value,
        ready: &DeferredReady,
    ) -> serde_json::Value {
        serde_json::json!({
            "requested_input": original_input,
            "prepared_snapshot": ready.snapshot,
            "session_id": ready.session_id,
            "revision": ready.revision,
            "expires_at": ready.expires_at,
        })
    }

    async fn wait_for_deferred_ready<Ctx>(
        pending: &PendingToolCallInfo,
        tool: &Arc<dyn ErasedDeferredTool<Ctx>>,
        tool_context: &ToolContext<Ctx>,
        tx: &mpsc::Sender<AgentEventEnvelope>,
        seq: &SequenceCounter,
    ) -> Result<DeferredReady, ToolResult>
    where
        Ctx: Send + Sync + Clone + 'static,
    {
        let mut state = match tool.open(tool_context, pending.input.clone()).await {
            Ok(state) => state,
            Err(e) => return Err(ToolResult::error(format!("Deferred open error: {e}"))),
        };

        for _ in 0..MAX_DEFERRED_POLLS {
            match state {
                DeferredToolState::Pending {
                    session_id,
                    revision,
                    message,
                    snapshot,
                    poll_after_ms,
                } => {
                    let data = Some(serde_json::json!({
                        "session_id": session_id,
                        "revision": revision,
                        "snapshot": snapshot,
                        "poll_after_ms": poll_after_ms,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "deferred_pending",
                            message,
                            data,
                        ),
                        seq,
                    )
                    .await;

                    let delay_ms = poll_after_ms
                        .max(MIN_DEFERRED_POLL_MS)
                        .min(MAX_DEFERRED_POLL_MS);
                    sleep(Duration::from_millis(delay_ms)).await;

                    state = match tool.poll(tool_context, &session_id).await {
                        Ok(next) => next,
                        Err(e) => {
                            return Err(ToolResult::error(format!("Deferred poll error: {e}")));
                        }
                    };
                }
                DeferredToolState::Ready {
                    session_id,
                    revision,
                    message,
                    snapshot,
                    expires_at,
                } => {
                    let data = Some(serde_json::json!({
                        "session_id": session_id,
                        "revision": revision,
                        "snapshot": snapshot,
                        "expires_at": expires_at,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "deferred_ready",
                            message,
                            data,
                        ),
                        seq,
                    )
                    .await;

                    return Ok(DeferredReady {
                        session_id,
                        revision,
                        snapshot,
                        expires_at,
                    });
                }
                DeferredToolState::Invalidated {
                    session_id,
                    message,
                    recoverable,
                } => {
                    let data = Some(serde_json::json!({
                        "session_id": session_id,
                        "recoverable": recoverable,
                    }));
                    wrap_and_send(
                        tx,
                        AgentEvent::tool_progress(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            "deferred_invalidated",
                            message.clone(),
                            data,
                        ),
                        seq,
                    )
                    .await;

                    let prefix = if recoverable {
                        "Deferred session invalidated (recoverable)"
                    } else {
                        "Deferred session invalidated"
                    };
                    return Err(ToolResult::error(format!("{prefix}: {message}")));
                }
            }
        }

        Err(ToolResult::error(format!(
            "Deferred tool exceeded max polling attempts ({MAX_DEFERRED_POLLS})"
        )))
    }

    // Check for listen/execute tool first
    if let Some(listen_tool) = tools.get_listen(&pending.name) {
        let tier = listen_tool.tier();

        // Emit tool call start
        wrap_and_send(
            tx,
            AgentEvent::tool_call_start(
                &pending.id,
                &pending.name,
                &pending.display_name,
                pending.input.clone(),
                tier,
            ),
            seq,
        )
        .await;

        let tool_start = Instant::now();
        let ready = match wait_for_listen_ready(pending, listen_tool, tool_context, tx, seq).await {
            Ok(ready) => ready,
            Err(mut result) => {
                result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                hooks.post_tool_use(&pending.name, &result).await;
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;
                return ToolExecutionOutcome::Completed {
                    tool_id: pending.id.clone(),
                    result,
                };
            }
        };

        let decision = hooks
            .pre_tool_use(&pending.name, &pending.input, tier)
            .await;

        return match decision {
            ToolDecision::Allow => {
                let result = match listen_tool
                    .execute(tool_context, &ready.operation_id, ready.revision)
                    .await
                {
                    Ok(mut r) => {
                        r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        r
                    }
                    Err(e) => ToolResult::error(format!("Listen execute error: {e}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                };

                hooks.post_tool_use(&pending.name, &result).await;
                send_event(
                    tx,
                    hooks,
                    seq,
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
                let _ = listen_tool
                    .cancel(tool_context, &ready.operation_id, ListenStopReason::Blocked)
                    .await;
                let result = ToolResult::error(format!("Blocked: {reason}"));
                send_event(
                    tx,
                    hooks,
                    seq,
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
                let input = build_listen_confirmation_input(&pending.input, &ready);
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::ToolRequiresConfirmation {
                        id: pending.id.clone(),
                        name: pending.name.clone(),
                        input: input.clone(),
                        description: description.clone(),
                    },
                )
                .await;

                ToolExecutionOutcome::RequiresConfirmation {
                    tool_id: pending.id.clone(),
                    tool_name: pending.name.clone(),
                    display_name: pending.display_name.clone(),
                    input,
                    description,
                    listen_context: Some(ListenExecutionContext {
                        operation_id: ready.operation_id,
                        revision: ready.revision,
                        snapshot: ready.snapshot,
                        expires_at: ready.expires_at,
                    }),
                }
            }
        };
    }

    // Check for deferred tool first
    if let Some(deferred_tool) = tools.get_deferred(&pending.name) {
        let tier = deferred_tool.tier();

        // Emit tool call start
        wrap_and_send(
            tx,
            AgentEvent::tool_call_start(
                &pending.id,
                &pending.name,
                &pending.display_name,
                pending.input.clone(),
                tier,
            ),
            seq,
        )
        .await;

        let tool_start = Instant::now();
        let ready =
            match wait_for_deferred_ready(pending, deferred_tool, tool_context, tx, seq).await {
                Ok(ready) => ready,
                Err(mut result) => {
                    result.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                    hooks.post_tool_use(&pending.name, &result).await;
                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::tool_call_end(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            result.clone(),
                        ),
                    )
                    .await;
                    return ToolExecutionOutcome::Completed {
                        tool_id: pending.id.clone(),
                        result,
                    };
                }
            };

        let decision = hooks
            .pre_tool_use(&pending.name, &pending.input, tier)
            .await;

        return match decision {
            ToolDecision::Allow => {
                let result = match deferred_tool
                    .commit(tool_context, &ready.session_id, ready.revision)
                    .await
                {
                    Ok(mut r) => {
                        r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                        r
                    }
                    Err(e) => ToolResult::error(format!("Deferred commit error: {e}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                };

                hooks.post_tool_use(&pending.name, &result).await;
                send_event(
                    tx,
                    hooks,
                    seq,
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
                    seq,
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
                let input = build_deferred_confirmation_input(&pending.input, &ready);
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::ToolRequiresConfirmation {
                        id: pending.id.clone(),
                        name: pending.name.clone(),
                        input: input.clone(),
                        description: description.clone(),
                    },
                )
                .await;

                ToolExecutionOutcome::RequiresConfirmation {
                    tool_id: pending.id.clone(),
                    tool_name: pending.name.clone(),
                    display_name: pending.display_name.clone(),
                    input,
                    description,
                    listen_context: Some(ListenExecutionContext {
                        operation_id: ready.session_id,
                        revision: ready.revision,
                        snapshot: ready.snapshot,
                        expires_at: ready.expires_at,
                    }),
                }
            }
        };
    }

    // Check for async tool first
    if let Some(async_tool) = tools.get_async(&pending.name) {
        let tier = async_tool.tier();

        // Emit tool call start
        wrap_and_send(
            tx,
            AgentEvent::tool_call_start(
                &pending.id,
                &pending.name,
                &pending.display_name,
                pending.input.clone(),
                tier,
            ),
            seq,
        )
        .await;

        // Check hooks for permission
        let decision = hooks
            .pre_tool_use(&pending.name, &pending.input, tier)
            .await;

        return match decision {
            ToolDecision::Allow => {
                let result = execute_async_tool(pending, async_tool, tool_context, tx, seq).await;

                hooks.post_tool_use(&pending.name, &result).await;

                send_event(
                    tx,
                    hooks,
                    seq,
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
                    seq,
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
                    seq,
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
                    listen_context: None,
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
    wrap_and_send(
        tx,
        AgentEvent::tool_call_start(
            &pending.id,
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            tier,
        ),
        seq,
    )
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
                seq,
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
                seq,
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
                seq,
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
                listen_context: None,
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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
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
            wrap_and_send(
                tx,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "started",
                    &message,
                    None,
                ),
                seq,
            )
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
                        wrap_and_send(
                            tx,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                stage,
                                message,
                                data,
                            ),
                            seq,
                        )
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
    rejection_reason: Option<String>,
    tool_context: &ToolContext<Ctx>,
    tools: &ToolRegistry<Ctx>,
    hooks: &Arc<H>,
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
) -> ToolResult
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    if rejection_reason.is_none() {
        // Check for listen/execute tool first
        if let Some(listen_tool) = tools.get_listen(&awaiting_tool.name) {
            let Some(listen) = awaiting_tool.listen_context.as_ref() else {
                return ToolResult::error(format!(
                    "Listen context missing for tool: {}",
                    awaiting_tool.name
                ));
            };

            let tool_start = Instant::now();
            let result = match listen_tool
                .execute(tool_context, &listen.operation_id, listen.revision)
                .await
            {
                Ok(mut r) => {
                    r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                    r
                }
                Err(e) => ToolResult::error(format!("Listen execute error: {e}"))
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
            };

            hooks.post_tool_use(&awaiting_tool.name, &result).await;

            wrap_and_send(
                tx,
                AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ),
                seq,
            )
            .await;

            return result;
        }

        // Check for deferred tool first
        if let Some(deferred_tool) = tools.get_deferred(&awaiting_tool.name) {
            let Some(deferred) = awaiting_tool.listen_context.as_ref() else {
                return ToolResult::error(format!(
                    "Deferred context missing for tool: {}",
                    awaiting_tool.name
                ));
            };

            let tool_start = Instant::now();
            let result = match deferred_tool
                .commit(tool_context, &deferred.operation_id, deferred.revision)
                .await
            {
                Ok(mut r) => {
                    r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                    r
                }
                Err(e) => ToolResult::error(format!("Deferred commit error: {e}"))
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
            };

            hooks.post_tool_use(&awaiting_tool.name, &result).await;

            wrap_and_send(
                tx,
                AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ),
                seq,
            )
            .await;

            return result;
        }
        // Check for async tool first
        if let Some(async_tool) = tools.get_async(&awaiting_tool.name) {
            let result = execute_async_tool(awaiting_tool, async_tool, tool_context, tx, seq).await;

            hooks.post_tool_use(&awaiting_tool.name, &result).await;

            wrap_and_send(
                tx,
                AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ),
                seq,
            )
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

            wrap_and_send(
                tx,
                AgentEvent::tool_call_end(
                    &awaiting_tool.id,
                    &awaiting_tool.name,
                    &awaiting_tool.display_name,
                    result.clone(),
                ),
                seq,
            )
            .await;

            result
        } else {
            ToolResult::error(format!("Unknown tool: {}", awaiting_tool.name))
        }
    } else {
        if let Some(listen_tool) = tools.get_listen(&awaiting_tool.name) {
            if let Some(listen) = awaiting_tool.listen_context.as_ref() {
                let _ = listen_tool
                    .cancel(
                        tool_context,
                        &listen.operation_id,
                        ListenStopReason::UserRejected,
                    )
                    .await;
            }
        } else if let Some(deferred_tool) = tools.get_deferred(&awaiting_tool.name) {
            if let Some(deferred) = awaiting_tool.listen_context.as_ref() {
                let _ = deferred_tool
                    .cancel(tool_context, &deferred.operation_id)
                    .await;
            }
        }

        let reason = rejection_reason.unwrap_or_else(|| "User rejected".to_string());
        let result = ToolResult::error(format!("Rejected: {reason}"));
        send_event(
            tx,
            hooks,
            seq,
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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
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
                    send_event(tx, hooks, seq, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Rate limited, retrying after backoff (attempt={}, delay_ms={})",
                    attempt,
                    delay.as_millis()
                );

                sleep(delay).await;
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!("Invalid request to LLM: {msg}");
                return Err(AgentError::new(format!("Invalid request: {msg}"), false));
            }
            ChatOutcome::ServerError(msg) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("LLM server error after {max_retries} retries: {msg}");
                    let error_msg = format!("Server error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, seq, AgentEvent::error(&error_msg, true)).await;
                    return Err(AgentError::new(error_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Server error, retrying after backoff (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    ids: (&str, &str),
) -> Result<ChatResponse, AgentError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let (message_id, thinking_id) = ids;
    let max_retries = config.retry.max_retries;
    let mut attempt = 0u32;

    loop {
        let result =
            process_stream(provider, &request, tx, hooks, seq, message_id, thinking_id).await;

        match result {
            Ok(response) => return Ok(response),
            Err(StreamError::Recoverable(msg)) => {
                attempt += 1;
                if attempt > max_retries {
                    error!("Streaming error after {max_retries} retries: {msg}");
                    let err_msg = format!("Streaming error after {max_retries} retries: {msg}");
                    send_event(tx, hooks, seq, AgentEvent::error(&err_msg, true)).await;
                    return Err(AgentError::new(err_msg, true));
                }
                let delay = calculate_backoff_delay(attempt, &config.retry);
                warn!(
                    "Streaming error, retrying (attempt={attempt}, delay_ms={}, error={msg})",
                    delay.as_millis()
                );

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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    message_id: &str,
    thinking_id: &str,
) -> Result<ChatResponse, StreamError>
where
    P: LlmProvider,
    H: AgentHooks,
{
    let mut stream = std::pin::pin!(provider.chat_stream(request.clone()));
    let mut accumulator = StreamAccumulator::new();
    let mut delta_count: u64 = 0;

    log::debug!("Starting to consume LLM stream");

    // Track channel health
    let mut channel_closed = false;

    while let Some(result) = stream.next().await {
        // Log progress every 50 deltas to show stream is alive
        if delta_count > 0 && delta_count.is_multiple_of(50) {
            log::debug!("Stream progress: delta_count={delta_count}");
        }

        match result {
            Ok(delta) => {
                delta_count += 1;
                accumulator.apply(&delta);
                match &delta {
                    StreamDelta::TextDelta { delta, .. } => {
                        // Check if channel is still open before sending
                        if !channel_closed {
                            if tx.is_closed() {
                                log::warn!(
                                    "Event channel closed by receiver at delta_count={delta_count} - consumer may have disconnected"
                                );
                                channel_closed = true;
                            } else {
                                send_event(
                                    tx,
                                    hooks,
                                    seq,
                                    AgentEvent::text_delta(message_id, delta.clone()),
                                )
                                .await;
                            }
                        }
                    }
                    StreamDelta::ThinkingDelta { delta, .. } => {
                        if !channel_closed {
                            if tx.is_closed() {
                                log::warn!(
                                    "Event channel closed by receiver at delta_count={delta_count}"
                                );
                                channel_closed = true;
                            } else {
                                send_event(
                                    tx,
                                    hooks,
                                    seq,
                                    AgentEvent::thinking_delta(thinking_id, delta.clone()),
                                )
                                .await;
                            }
                        }
                    }
                    StreamDelta::Error {
                        message,
                        recoverable,
                    } => {
                        log::warn!(
                            "Stream error received delta_count={delta_count} message={message} recoverable={recoverable}"
                        );
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
                    | StreamDelta::ToolInputDelta { .. }
                    | StreamDelta::SignatureDelta { .. }
                    | StreamDelta::RedactedThinking { .. } => {}
                }
            }
            Err(e) => {
                log::error!("Stream iteration error delta_count={delta_count} error={e}");
                return Err(StreamError::Recoverable(format!("Stream error: {e}")));
            }
        }
    }

    log::debug!("Stream while loop exited normally at delta_count={delta_count}");

    let usage = accumulator.usage().cloned().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
    });
    let stop_reason = accumulator.stop_reason().copied();
    let content_blocks = accumulator.into_content_blocks();

    log::debug!(
        "LLM stream completed successfully delta_count={delta_count} stop_reason={stop_reason:?} content_block_count={} input_tokens={} output_tokens={}",
        content_blocks.len(),
        usage.input_tokens,
        usage.output_tokens
    );

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
    tx: mpsc::Sender<AgentEventEnvelope>,
    seq: SequenceCounter,
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
    let tool_context = tool_context.with_event_tx(tx.clone(), seq.clone());
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
            send_event(&tx, &hooks, &seq, AgentEvent::error(&message, recoverable)).await;
            return AgentRunState::Error(AgentError::new(&message, recoverable));
        }

        let rejection =
            (!confirmed).then(|| rejection_reason.unwrap_or_else(|| "User rejected".to_string()));
        let result = execute_confirmed_tool(
            awaiting_tool,
            rejection,
            &tool_context,
            &tools,
            &hooks,
            &tx,
            &seq,
        )
        .await;
        tool_results.push((awaiting_tool.id.clone(), result));

        for pending in cont.pending_tool_calls.iter().skip(cont.awaiting_index + 1) {
            match execute_tool_call(pending, &tool_context, &tools, &hooks, &tx, &seq).await {
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
                    let pending_idx = cont
                        .pending_tool_calls
                        .iter()
                        .position(|p| p.id == tool_id)
                        .unwrap_or(0);

                    let mut pending_tool_calls = cont.pending_tool_calls.clone();
                    if let Some(context) = listen_context {
                        if let Some(item) = pending_tool_calls.get_mut(pending_idx) {
                            item.listen_context = Some(context);
                        }
                    }

                    let new_continuation = AgentContinuation {
                        thread_id: thread_id.clone(),
                        turn,
                        total_usage: total_usage.clone(),
                        turn_usage: cont.turn_usage.clone(),
                        pending_tool_calls,
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
                &seq,
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
            &seq,
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
            &seq,
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
                    warn!("Failed to save state checkpoint: {e}");
                }
            }
            InternalTurnResult::Done => {
                break;
            }
            InternalTurnResult::Refusal => {
                return AgentRunState::Refusal {
                    total_turns: turns_to_u32(ctx.turn),
                    input_tokens: u64::from(ctx.total_usage.input_tokens),
                    output_tokens: u64::from(ctx.total_usage.output_tokens),
                };
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

struct TurnParameters<Ctx, P, H, M, S> {
    tx: mpsc::Sender<AgentEventEnvelope>,
    seq: SequenceCounter,
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
    let tool_context = tool_context.with_event_tx(tx.clone(), seq.clone());
    let start_time = Instant::now();

    let init_state =
        match initialize_from_input(input, &thread_id, &message_store, &state_store).await {
            Ok(s) => s,
            Err(e) => {
                send_event(
                    &tx,
                    &hooks,
                    &seq,
                    AgentEvent::error(&e.message, e.recoverable),
                )
                .await;
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
            seq,
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
        &seq,
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

    convert_turn_result(result, ctx, &tx, &hooks, &seq, thread_id, &state_store).await
}

async fn convert_turn_result<H: AgentHooks, S: StateStore>(
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

struct ResumeCaseParameters<Ctx, H, M, S> {
    resume_data: ResumeData,
    turn: usize,
    total_usage: TokenUsage,
    state: AgentState,
    thread_id: ThreadId,
    tool_context: ToolContext<Ctx>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    tx: mpsc::Sender<AgentEventEnvelope>,
    seq: SequenceCounter,
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
        seq,
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
    let mut tool_results = cont.completed_results.clone();
    let awaiting_tool = &cont.pending_tool_calls[cont.awaiting_index];

    if awaiting_tool.id != tool_call_id {
        let msg = format!(
            "Tool call ID mismatch: expected {}, got {}",
            awaiting_tool.id, tool_call_id
        );
        send_event(&tx, &hooks, &seq, AgentEvent::error(&msg, false)).await;
        return TurnOutcome::Error(AgentError::new(&msg, false));
    }

    let rejection =
        (!confirmed).then(|| rejection_reason.unwrap_or_else(|| "User rejected".to_string()));
    let result = execute_confirmed_tool(
        awaiting_tool,
        rejection,
        &tool_context,
        &tools,
        &hooks,
        &tx,
        &seq,
    )
    .await;
    tool_results.push((awaiting_tool.id.clone(), result));

    for pending in cont.pending_tool_calls.iter().skip(cont.awaiting_index + 1) {
        match execute_tool_call(pending, &tool_context, &tools, &hooks, &tx, &seq).await {
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
                let pending_idx = cont
                    .pending_tool_calls
                    .iter()
                    .position(|p| p.id == tool_id)
                    .unwrap_or(0);

                let mut pending_tool_calls = cont.pending_tool_calls.clone();
                if let Some(context) = listen_context {
                    if let Some(item) = pending_tool_calls.get_mut(pending_idx) {
                        item.listen_context = Some(context);
                    }
                }

                let new_continuation = AgentContinuation {
                    thread_id: thread_id.clone(),
                    turn,
                    total_usage: total_usage.clone(),
                    turn_usage: cont.turn_usage.clone(),
                    pending_tool_calls,
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
        send_event(
            &tx,
            &hooks,
            &seq,
            AgentEvent::error(&e.message, e.recoverable),
        )
        .await;
        return TurnOutcome::Error(e);
    }

    send_event(
        &tx,
        &hooks,
        &seq,
        AgentEvent::TurnComplete {
            turn,
            usage: cont.turn_usage.clone(),
        },
    )
    .await;

    let mut updated_state = state;
    updated_state.turn_count = turn;
    if let Err(e) = state_store.save(&updated_state).await {
        warn!("Failed to save state checkpoint: {e}");
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
                "Found in-flight execution from previous attempt, re-executing (tool_call_id={}, tool_name={})",
                tool_call_id, execution.tool_name
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
                "Failed to record execution start (tool_call_id={}, error={})",
                pending.id, e
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
                "Failed to record execution completion (tool_call_id={}, error={})",
                pending.id, e
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
    tx: &mpsc::Sender<AgentEventEnvelope>,
    seq: &SequenceCounter,
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
        warn!(
            "Max turns reached (turn={}, max={})",
            ctx.turn, config.max_turns
        );
        send_event(
            tx,
            hooks,
            seq,
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
        seq,
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
                seq,
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
                "Context compaction triggered (turn={}, message_count={})",
                ctx.turn,
                messages.len()
            );

            match compactor.compact_history(messages.clone()).await {
                Ok(result) => {
                    if let Err(e) = message_store
                        .replace_history(&ctx.thread_id, result.messages.clone())
                        .await
                    {
                        warn!("Failed to replace history after compaction: {e}");
                    } else {
                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::context_compacted(
                                result.original_count,
                                result.new_count,
                                result.original_tokens,
                                result.new_tokens,
                            ),
                        )
                        .await;

                        info!(
                            "Context compacted successfully (original_count={}, new_count={}, original_tokens={}, new_tokens={})",
                            result.original_count,
                            result.new_count,
                            result.original_tokens,
                            result.new_tokens
                        );

                        messages = result.messages;
                    }
                }
                Err(e) => {
                    warn!("Context compaction failed, continuing with full history: {e}");
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

    // Log the request messages for debugging context issues
    debug!(
        "ChatRequest built: system_prompt_len={} num_messages={} num_tools={} max_tokens={}",
        request.system.len(),
        request.messages.len(),
        request.tools.as_ref().map_or(0, Vec::len),
        request.max_tokens
    );
    for (i, msg) in request.messages.iter().enumerate() {
        match &msg.content {
            Content::Text(text) => {
                debug!(
                    "  message[{}]: role={:?} content=Text(len={})",
                    i,
                    msg.role,
                    text.len()
                );
            }
            Content::Blocks(blocks) => {
                debug!(
                    "  message[{}]: role={:?} content=Blocks(count={})",
                    i,
                    msg.role,
                    blocks.len()
                );
                for (j, block) in blocks.iter().enumerate() {
                    match block {
                        ContentBlock::Text { text } => {
                            debug!("    block[{}]: Text(len={})", j, text.len());
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            debug!("    block[{}]: Thinking(len={})", j, thinking.len());
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            debug!("    block[{j}]: RedactedThinking");
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            debug!("    block[{j}]: ToolUse(id={id}, name={name}, input={input})");
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            debug!(
                                "    block[{}]: ToolResult(tool_use_id={}, is_error={:?}, content_len={})",
                                j,
                                tool_use_id,
                                is_error,
                                content.len()
                            );
                        }
                    }
                }
            }
        }
    }

    // Call LLM with retry logic (streaming or non-streaming based on config)
    debug!(
        "Calling LLM (turn={}, streaming={})",
        ctx.turn, config.streaming
    );
    let message_id = uuid::Uuid::new_v4().to_string();
    let thinking_id = uuid::Uuid::new_v4().to_string();
    let response = if config.streaming {
        // Streaming mode: events are emitted as content arrives
        match call_llm_streaming(
            provider,
            request,
            config,
            tx,
            hooks,
            seq,
            (&message_id, &thinking_id),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                return InternalTurnResult::Error(e);
            }
        }
    } else {
        // Non-streaming mode: wait for full response
        match call_llm_with_retry(provider, request, config, tx, hooks, seq).await {
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

    // Emit the complete Thinking event.
    // In non-streaming mode this is the only thinking event.
    // In streaming mode this comes after all ThinkingDelta events.
    if let Some(thinking) = &thinking_content {
        send_event(
            tx,
            hooks,
            seq,
            AgentEvent::thinking(thinking_id, thinking.clone()),
        )
        .await;
    }

    // Always emit the final complete Text event so consumers know the full response is ready
    // (in streaming mode this comes after all TextDelta events)
    if let Some(text) = &text_content {
        send_event(tx, hooks, seq, AgentEvent::text(&message_id, text.clone())).await;
    }

    // Store assistant message in conversation history (includes text and tool uses)
    let assistant_msg = build_assistant_message(&response);
    if let Err(e) = message_store.append(&ctx.thread_id, assistant_msg).await {
        send_event(
            tx,
            hooks,
            seq,
            AgentEvent::error(format!("Failed to append assistant message: {e}"), false),
        )
        .await;
        return InternalTurnResult::Error(AgentError::new(
            format!("Failed to append assistant message: {e}"),
            false,
        ));
    }

    // Build pending tool calls (check both sync and async tools for display_name)
    let mut pending_tool_calls: Vec<PendingToolCallInfo> = tool_uses
        .iter()
        .map(|(id, name, input)| {
            let display_name = tools
                .get(name)
                .map(|t| t.display_name().to_string())
                .or_else(|| tools.get_async(name).map(|t| t.display_name().to_string()))
                .or_else(|| tools.get_listen(name).map(|t| t.display_name().to_string()))
                .or_else(|| {
                    tools
                        .get_deferred(name)
                        .map(|t| t.display_name().to_string())
                })
                .unwrap_or_default();
            PendingToolCallInfo {
                id: id.clone(),
                name: name.clone(),
                display_name,
                input: input.clone(),
                listen_context: None,
            }
        })
        .collect();

    // Execute tools (supports both sync and async tools)
    let mut tool_results = Vec::new();
    for idx in 0..pending_tool_calls.len() {
        let pending = pending_tool_calls[idx].clone();
        // IDEMPOTENCY: Check for cached result from a previous execution attempt
        if let Some(cached_result) = try_get_cached_result(execution_store, &pending.id).await {
            debug!(
                "Using cached result from previous execution (tool_call_id={}, tool_name={})",
                pending.id, pending.name
            );
            tool_results.push((pending.id.clone(), cached_result));
            continue;
        }

        // Check for listen/execute tool first
        if let Some(listen_tool) = tools.get_listen(&pending.name) {
            let tier = listen_tool.tier();

            send_event(
                tx,
                hooks,
                seq,
                AgentEvent::tool_call_start(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    pending.input.clone(),
                    tier,
                ),
            )
            .await;

            const MAX_LISTEN_UPDATES: usize = 240;

            let tool_start = Instant::now();
            let mut updates = listen_tool.listen_stream(tool_context, pending.input.clone());
            let mut update_count = 0usize;
            let mut last_operation_id: Option<String> = None;
            let mut ready_context: Option<ListenExecutionContext> = None;
            let mut confirmation_input: Option<serde_json::Value> = None;
            let mut terminal_result: Option<ToolResult> = None;

            while let Some(update) = updates.next().await {
                update_count += 1;

                match update {
                    ListenToolUpdate::Listening {
                        operation_id,
                        revision,
                        message,
                        snapshot,
                        expires_at,
                    } => {
                        last_operation_id = Some(operation_id.clone());
                        let data = Some(serde_json::json!({
                            "operation_id": operation_id,
                            "revision": revision,
                            "snapshot": snapshot,
                            "expires_at": expires_at,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "listen_update",
                                message,
                                data,
                            ),
                        )
                        .await;
                    }
                    ListenToolUpdate::Ready {
                        operation_id,
                        revision,
                        message,
                        snapshot,
                        expires_at,
                    } => {
                        let data = Some(serde_json::json!({
                            "operation_id": operation_id,
                            "revision": revision,
                            "snapshot": snapshot,
                            "expires_at": expires_at,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "listen_ready",
                                message,
                                data,
                            ),
                        )
                        .await;

                        ready_context = Some(ListenExecutionContext {
                            operation_id: operation_id.clone(),
                            revision,
                            snapshot: snapshot.clone(),
                            expires_at: expires_at.clone(),
                        });
                        confirmation_input = Some(serde_json::json!({
                            "requested_input": pending.input.clone(),
                            "prepared_snapshot": snapshot,
                            "operation_id": operation_id,
                            "revision": revision,
                            "expires_at": expires_at,
                        }));
                        break;
                    }
                    ListenToolUpdate::Invalidated {
                        operation_id,
                        message,
                        recoverable,
                    } => {
                        let data = Some(serde_json::json!({
                            "operation_id": operation_id,
                            "recoverable": recoverable,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "listen_invalidated",
                                message.clone(),
                                data,
                            ),
                        )
                        .await;

                        let result = if recoverable {
                            ToolResult::error(format!(
                                "Listen operation invalidated (recoverable): {message}"
                            ))
                        } else {
                            ToolResult::error(format!("Listen operation invalidated: {message}"))
                        }
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_call_end(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                result.clone(),
                            ),
                        )
                        .await;
                        terminal_result = Some(result);
                        break;
                    }
                }

                if tx.is_closed() {
                    if let Some(operation_id) = last_operation_id.as_deref() {
                        let _ = listen_tool
                            .cancel(
                                tool_context,
                                operation_id,
                                ListenStopReason::StreamDisconnected,
                            )
                            .await;
                    }
                    let result = ToolResult::error(
                        "Listen stream disconnected before operation became ready",
                    )
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                    terminal_result = Some(result);
                    break;
                }

                if update_count >= MAX_LISTEN_UPDATES {
                    if let Some(operation_id) = last_operation_id.as_deref() {
                        let _ = listen_tool
                            .cancel(tool_context, operation_id, ListenStopReason::StreamEnded)
                            .await;
                    }
                    let result = ToolResult::error(format!(
                        "Listen tool exceeded max updates ({MAX_LISTEN_UPDATES})"
                    ))
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::tool_call_end(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            result.clone(),
                        ),
                    )
                    .await;
                    terminal_result = Some(result);
                    break;
                }
            }

            if let Some(result) = terminal_result {
                tool_results.push((pending.id.clone(), result));
                continue;
            }

            let Some(listen_context) = ready_context else {
                if let Some(operation_id) = last_operation_id.as_deref() {
                    let _ = listen_tool
                        .cancel(tool_context, operation_id, ListenStopReason::StreamEnded)
                        .await;
                }
                let result = ToolResult::error("Listen stream ended before operation became ready")
                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;
                tool_results.push((pending.id.clone(), result));
                continue;
            };

            let decision = hooks
                .pre_tool_use(&pending.name, &pending.input, tier)
                .await;

            match decision {
                ToolDecision::Allow => {
                    // IDEMPOTENCY: Record execution start (write-ahead)
                    let started_at = time::OffsetDateTime::now_utc();
                    record_execution_start(execution_store, &pending, &ctx.thread_id, started_at)
                        .await;

                    let result = match listen_tool
                        .execute(
                            tool_context,
                            &listen_context.operation_id,
                            listen_context.revision,
                        )
                        .await
                    {
                        Ok(mut r) => {
                            r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            r
                        }
                        Err(e) => ToolResult::error(format!("Listen execute error: {e}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    };

                    // IDEMPOTENCY: Record execution completion
                    record_execution_complete(
                        execution_store,
                        &pending,
                        &ctx.thread_id,
                        &result,
                        started_at,
                    )
                    .await;

                    hooks.post_tool_use(&pending.name, &result).await;

                    send_event(
                        tx,
                        hooks,
                        seq,
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
                    let _ = listen_tool
                        .cancel(
                            tool_context,
                            &listen_context.operation_id,
                            ListenStopReason::Blocked,
                        )
                        .await;
                    let result = ToolResult::error(format!("Blocked: {reason}"));
                    send_event(
                        tx,
                        hooks,
                        seq,
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
                    if let Some(item) = pending_tool_calls.get_mut(idx) {
                        item.listen_context = Some(listen_context.clone());
                    }
                    let input = confirmation_input.unwrap_or_else(|| pending.input.clone());

                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::ToolRequiresConfirmation {
                            id: pending.id.clone(),
                            name: pending.name.clone(),
                            input: input.clone(),
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
                        input,
                        description,
                        continuation: Box::new(continuation),
                    };
                }
            }
            continue;
        }

        // Check for deferred tool first
        if let Some(deferred_tool) = tools.get_deferred(&pending.name) {
            let tier = deferred_tool.tier();

            send_event(
                tx,
                hooks,
                seq,
                AgentEvent::tool_call_start(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    pending.input.clone(),
                    tier,
                ),
            )
            .await;

            let tool_start = Instant::now();
            const MAX_DEFERRED_POLLS: usize = 240;
            const MIN_DEFERRED_POLL_MS: u64 = 50;
            const MAX_DEFERRED_POLL_MS: u64 = 30_000;

            let mut deferred_state = match deferred_tool
                .open(tool_context, pending.input.clone())
                .await
            {
                Ok(state) => state,
                Err(e) => {
                    let result = ToolResult::error(format!("Deferred open error: {e}"))
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::tool_call_end(
                            &pending.id,
                            &pending.name,
                            &pending.display_name,
                            result.clone(),
                        ),
                    )
                    .await;
                    tool_results.push((pending.id.clone(), result));
                    continue;
                }
            };

            let mut ready_context: Option<ListenExecutionContext> = None;
            let mut confirmation_input: Option<serde_json::Value> = None;
            let mut terminal_result: Option<ToolResult> = None;

            for _ in 0..MAX_DEFERRED_POLLS {
                match deferred_state {
                    DeferredToolState::Pending {
                        session_id,
                        revision,
                        message,
                        snapshot,
                        poll_after_ms,
                    } => {
                        let data = Some(serde_json::json!({
                            "session_id": session_id,
                            "revision": revision,
                            "snapshot": snapshot,
                            "poll_after_ms": poll_after_ms,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "deferred_pending",
                                message,
                                data,
                            ),
                        )
                        .await;

                        let delay_ms = poll_after_ms
                            .max(MIN_DEFERRED_POLL_MS)
                            .min(MAX_DEFERRED_POLL_MS);
                        sleep(Duration::from_millis(delay_ms)).await;

                        deferred_state = match deferred_tool.poll(tool_context, &session_id).await {
                            Ok(next) => next,
                            Err(e) => {
                                let result = ToolResult::error(format!("Deferred poll error: {e}"))
                                    .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                                send_event(
                                    tx,
                                    hooks,
                                    seq,
                                    AgentEvent::tool_call_end(
                                        &pending.id,
                                        &pending.name,
                                        &pending.display_name,
                                        result.clone(),
                                    ),
                                )
                                .await;
                                terminal_result = Some(result);
                                break;
                            }
                        };
                    }
                    DeferredToolState::Ready {
                        session_id,
                        revision,
                        message,
                        snapshot,
                        expires_at,
                    } => {
                        let data = Some(serde_json::json!({
                            "session_id": session_id,
                            "revision": revision,
                            "snapshot": snapshot,
                            "expires_at": expires_at,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "deferred_ready",
                                message,
                                data,
                            ),
                        )
                        .await;

                        ready_context = Some(ListenExecutionContext {
                            operation_id: session_id.clone(),
                            revision,
                            snapshot: snapshot.clone(),
                            expires_at: expires_at.clone(),
                        });
                        confirmation_input = Some(serde_json::json!({
                            "requested_input": pending.input.clone(),
                            "prepared_snapshot": snapshot,
                            "session_id": session_id,
                            "revision": revision,
                            "expires_at": expires_at,
                        }));
                        break;
                    }
                    DeferredToolState::Invalidated {
                        session_id,
                        message,
                        recoverable,
                    } => {
                        let result = if recoverable {
                            ToolResult::error(format!(
                                "Deferred session invalidated (recoverable): {message}"
                            ))
                        } else {
                            ToolResult::error(format!("Deferred session invalidated: {message}"))
                        }
                        .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));

                        let data = Some(serde_json::json!({
                            "session_id": session_id,
                            "recoverable": recoverable,
                        }));

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_progress(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                "deferred_invalidated",
                                message,
                                data,
                            ),
                        )
                        .await;

                        send_event(
                            tx,
                            hooks,
                            seq,
                            AgentEvent::tool_call_end(
                                &pending.id,
                                &pending.name,
                                &pending.display_name,
                                result.clone(),
                            ),
                        )
                        .await;
                        terminal_result = Some(result);
                        break;
                    }
                }
            }

            if let Some(result) = terminal_result {
                tool_results.push((pending.id.clone(), result));
                continue;
            }

            let Some(listen_context) = ready_context else {
                let result = ToolResult::error(format!(
                    "Deferred tool exceeded max polling attempts ({MAX_DEFERRED_POLLS})"
                ))
                .with_duration(millis_to_u64(tool_start.elapsed().as_millis()));
                send_event(
                    tx,
                    hooks,
                    seq,
                    AgentEvent::tool_call_end(
                        &pending.id,
                        &pending.name,
                        &pending.display_name,
                        result.clone(),
                    ),
                )
                .await;
                tool_results.push((pending.id.clone(), result));
                continue;
            };

            let decision = hooks
                .pre_tool_use(&pending.name, &pending.input, tier)
                .await;

            match decision {
                ToolDecision::Allow => {
                    // IDEMPOTENCY: Record execution start (write-ahead)
                    let started_at = time::OffsetDateTime::now_utc();
                    record_execution_start(execution_store, &pending, &ctx.thread_id, started_at)
                        .await;

                    let result = match deferred_tool
                        .commit(
                            tool_context,
                            &listen_context.operation_id,
                            listen_context.revision,
                        )
                        .await
                    {
                        Ok(mut r) => {
                            r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            r
                        }
                        Err(e) => ToolResult::error(format!("Deferred commit error: {e}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    };

                    // IDEMPOTENCY: Record execution completion
                    record_execution_complete(
                        execution_store,
                        &pending,
                        &ctx.thread_id,
                        &result,
                        started_at,
                    )
                    .await;

                    hooks.post_tool_use(&pending.name, &result).await;

                    send_event(
                        tx,
                        hooks,
                        seq,
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
                        seq,
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
                    if let Some(item) = pending_tool_calls.get_mut(idx) {
                        item.listen_context = Some(listen_context.clone());
                    }
                    let input = confirmation_input.unwrap_or_else(|| pending.input.clone());

                    send_event(
                        tx,
                        hooks,
                        seq,
                        AgentEvent::ToolRequiresConfirmation {
                            id: pending.id.clone(),
                            name: pending.name.clone(),
                            input: input.clone(),
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
                        input,
                        description,
                        continuation: Box::new(continuation),
                    };
                }
            }
            continue;
        }

        // Check for async tool first
        if let Some(async_tool) = tools.get_async(&pending.name) {
            let tier = async_tool.tier();

            // Emit tool call start
            send_event(
                tx,
                hooks,
                seq,
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
                    record_execution_start(execution_store, &pending, &ctx.thread_id, started_at)
                        .await;

                    let result =
                        execute_async_tool(&pending, async_tool, tool_context, tx, seq).await;

                    // IDEMPOTENCY: Record execution completion
                    record_execution_complete(
                        execution_store,
                        &pending,
                        &ctx.thread_id,
                        &result,
                        started_at,
                    )
                    .await;

                    hooks.post_tool_use(&pending.name, &result).await;

                    send_event(
                        tx,
                        hooks,
                        seq,
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
                        seq,
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
                        seq,
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
            seq,
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
                record_execution_start(execution_store, &pending, &ctx.thread_id, started_at).await;

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
                    &pending,
                    &ctx.thread_id,
                    &result,
                    started_at,
                )
                .await;

                hooks.post_tool_use(&pending.name, &result).await;

                send_event(
                    tx,
                    hooks,
                    seq,
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
                    seq,
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
                    seq,
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
            seq,
            AgentEvent::error(format!("Failed to append tool results: {e}"), false),
        )
        .await;
        return InternalTurnResult::Error(e);
    }

    // Emit turn complete
    send_event(
        tx,
        hooks,
        seq,
        AgentEvent::TurnComplete {
            turn: ctx.turn,
            usage: turn_usage.clone(),
        },
    )
    .await;

    // Check stop reason
    match response.stop_reason {
        Some(StopReason::EndTurn) => {
            info!("Agent completed (end_turn) (turn={})", ctx.turn);
            return InternalTurnResult::Done;
        }
        Some(StopReason::Refusal) => {
            warn!(
                "Model refused request (turn={}): {:?}",
                ctx.turn, text_content
            );
            send_event(
                tx,
                hooks,
                seq,
                AgentEvent::refusal(message_id, text_content),
            )
            .await;
            return InternalTurnResult::Refusal;
        }
        Some(StopReason::ModelContextWindowExceeded) => {
            warn!("Model context window exceeded (turn={})", ctx.turn);
            if let Some(compact_config) = compaction_config {
                let compactor =
                    LlmContextCompactor::new(Arc::clone(provider), compact_config.clone());
                let history = match message_store.get_history(&ctx.thread_id).await {
                    Ok(h) => h,
                    Err(e) => {
                        return InternalTurnResult::Error(AgentError::new(
                            format!(
                                "Failed to get history for compaction after context overflow: {e}"
                            ),
                            false,
                        ));
                    }
                };
                match compactor.compact_history(history).await {
                    Ok(result) => {
                        if let Err(e) = message_store
                            .replace_history(&ctx.thread_id, result.messages)
                            .await
                        {
                            return InternalTurnResult::Error(AgentError::new(
                                format!("Failed to replace history after overflow compaction: {e}"),
                                false,
                            ));
                        }
                        info!(
                            "Context compacted after overflow (original_tokens={}, new_tokens={})",
                            result.original_tokens, result.new_tokens
                        );
                        // Decrement turn so the retry doesn't count as a new turn
                        ctx.turn -= 1;
                        return InternalTurnResult::Continue { turn_usage };
                    }
                    Err(e) => {
                        return InternalTurnResult::Error(AgentError::new(
                            format!("Context compaction failed after overflow: {e}"),
                            false,
                        ));
                    }
                }
            }
            return InternalTurnResult::Error(AgentError::new(
                "Model context window exceeded and no compaction configured".to_string(),
                false,
            ));
        }
        _ => {}
    }

    InternalTurnResult::Continue { turn_usage }
}

/// Saturating conversion from usize to u32.
#[allow(clippy::cast_possible_truncation)]
const fn turns_to_u32(turns: usize) -> u32 {
    if turns > u32::MAX as usize {
        u32::MAX
    } else {
        turns as u32
    }
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
            ContentBlock::Thinking { thinking, .. } => {
                thinking_parts.push(thinking.clone());
            }
            ContentBlock::RedactedThinking { .. } | ContentBlock::ToolResult { .. } => {
                // Redacted thinking is opaque; ToolResult shouldn't appear in response
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let input = if input.is_null() {
                    serde_json::json!({})
                } else {
                    input.clone()
                };
                tool_uses.push((id.clone(), name.clone(), input.clone()));
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

/// Send an event to the consumer channel with non-blocking behavior.
///
/// This function first calls the hook's `on_event` method, then attempts to send
/// the event to the consumer channel. The sending behavior is designed to be
/// resilient to slow or disconnected consumers:
///
/// 1. First attempts a non-blocking send via `try_send`
/// 2. If the channel is full, waits up to 30 seconds for space
/// 3. If the channel is closed, logs and continues without blocking
/// 4. On timeout, logs an error and continues
///
/// This ensures that the agent loop doesn't block indefinitely if the consumer
/// is slow or has disconnected.
async fn send_event<H>(
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    event: AgentEvent,
) where
    H: AgentHooks,
{
    hooks.on_event(&event).await;

    let envelope = AgentEventEnvelope::wrap(event, seq);

    // Try non-blocking send first to detect backpressure
    match tx.try_send(envelope) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(envelope)) => {
            // Channel is full - consumer is slow or blocked
            log::debug!("Event channel full, waiting for consumer...");
            // Fall back to blocking send with timeout
            match tokio::time::timeout(std::time::Duration::from_secs(30), tx.send(envelope)).await
            {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    log::warn!("Event channel closed while sending - consumer disconnected");
                }
                Err(_) => {
                    log::error!("Timeout waiting to send event - consumer may be deadlocked");
                }
            }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            log::debug!("Event channel closed - consumer disconnected");
        }
    }
}

/// Send an event directly to the channel without going through hooks.
///
/// Used by async tool execution for progress events that bypass the hook system.
async fn wrap_and_send(
    tx: &mpsc::Sender<AgentEventEnvelope>,
    event: AgentEvent,
    seq: &SequenceCounter,
) {
    let envelope = AgentEventEnvelope::wrap(event, seq);
    let _ = tx.send(envelope).await;
}

fn build_assistant_message(response: &ChatResponse) -> Message {
    let mut blocks = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                blocks.push(ContentBlock::Text { text: text.clone() });
            }
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                blocks.push(ContentBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                });
            }
            ContentBlock::RedactedThinking { data } => {
                blocks.push(ContentBlock::RedactedThinking { data: data.clone() });
            }
            ContentBlock::ToolResult { .. } => {
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
    use crate::tools::{
        DeferredTool, DeferredToolState, ListenExecuteTool, ListenStopReason, ListenToolUpdate,
        Tool, ToolContext, ToolRegistry,
    };
    use crate::types::{AgentConfig, AgentInput, ToolResult, ToolTier, TurnOutcome};
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, RwLock};

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
        DeferredEcho,
        ListenEcho,
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

    struct DeferredEchoTool;

    impl DeferredTool<()> for DeferredEchoTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::DeferredEcho
        }

        fn display_name(&self) -> &'static str {
            "Deferred Echo"
        }

        fn description(&self) -> &'static str {
            "Deferred tool used for confirmation flow tests"
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

        async fn open(
            &self,
            _ctx: &ToolContext<()>,
            _input: serde_json::Value,
        ) -> Result<DeferredToolState> {
            Ok(DeferredToolState::Pending {
                session_id: "deferred-session-1".to_string(),
                revision: 1,
                message: "Waiting for deferred readiness".to_string(),
                snapshot: None,
                poll_after_ms: 1,
            })
        }

        async fn poll(
            &self,
            _ctx: &ToolContext<()>,
            session_id: &str,
        ) -> Result<DeferredToolState> {
            Ok(DeferredToolState::Ready {
                session_id: session_id.to_string(),
                revision: 2,
                message: "Ready to commit".to_string(),
                snapshot: json!({ "preview": "v2" }),
                expires_at: None,
            })
        }

        async fn commit(
            &self,
            _ctx: &ToolContext<()>,
            _session_id: &str,
            _expected_revision: u64,
        ) -> Result<ToolResult> {
            Ok(ToolResult::success("Deferred commit complete"))
        }
    }

    struct ListenEchoTool {
        cancel_calls: Arc<AtomicUsize>,
    }

    impl ListenExecuteTool<()> for ListenEchoTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::ListenEcho
        }

        fn display_name(&self) -> &'static str {
            "Listen Echo"
        }

        fn description(&self) -> &'static str {
            "Listen/execute tool used for confirmation flow tests"
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

        fn listen(
            &self,
            _ctx: &ToolContext<()>,
            _input: serde_json::Value,
        ) -> impl futures::Stream<Item = ListenToolUpdate> + Send {
            futures::stream::iter(vec![
                ListenToolUpdate::Listening {
                    operation_id: "listen-op-1".to_string(),
                    revision: 1,
                    message: "Preparing operation".to_string(),
                    snapshot: Some(json!({ "preview": "v1" })),
                    expires_at: None,
                },
                ListenToolUpdate::Ready {
                    operation_id: "listen-op-1".to_string(),
                    revision: 2,
                    message: "Ready to execute".to_string(),
                    snapshot: json!({ "preview": "v2" }),
                    expires_at: None,
                },
            ])
        }

        async fn execute(
            &self,
            _ctx: &ToolContext<()>,
            _operation_id: &str,
            _expected_revision: u64,
        ) -> Result<ToolResult> {
            Ok(ToolResult::success("Listen execute complete"))
        }

        async fn cancel(
            &self,
            _ctx: &ToolContext<()>,
            _operation_id: &str,
            _reason: ListenStopReason,
        ) -> Result<()> {
            self.cancel_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
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
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Text { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Done { .. }))
        );

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
                .any(|e| matches!(e.event, AgentEvent::ToolCallStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::ToolCallEnd { .. }))
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
            matches!(&e.event, AgentEvent::Error { message, .. } if message.contains("Maximum turns"))
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
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Done { .. }))
        );

        // The LLM's response about the missing tool should be in the events
        assert!(events.iter().any(|e| {
            matches!(&e.event, AgentEvent::Text { text, .. } if text.contains("couldn't find"))
        }));

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
            matches!(&e.event, AgentEvent::Error { message, recoverable: true } if message.contains("Rate limited"))
        }));

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
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Done { .. }))
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
            matches!(&e.event, AgentEvent::Error { message, recoverable: true } if message.contains("Server error"))
        }));

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
        assert!(
            events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Done { .. }))
        );

        Ok(())
    }

    // ================================
    // Event Envelope Idempotency Tests
    // ================================

    #[tokio::test]
    async fn test_envelope_event_ids_are_unique() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let agent = builder::<()>().provider(provider).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        let mut ids = std::collections::HashSet::new();
        while let Some(envelope) = rx.recv().await {
            assert!(
                ids.insert(envelope.event_id),
                "duplicate event_id: {}",
                envelope.event_id
            );
        }
        assert!(ids.len() >= 3, "expected at least Start+Text+Done events");

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_sequences_are_strictly_increasing() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let agent = builder::<()>().provider(provider).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        let mut envelopes = Vec::new();
        while let Some(envelope) = rx.recv().await {
            envelopes.push(envelope);
        }

        for pair in envelopes.windows(2) {
            assert!(
                pair[1].sequence > pair[0].sequence,
                "sequence not strictly increasing: {} -> {}",
                pair[0].sequence,
                pair[1].sequence,
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_sequences_start_at_zero() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let agent = builder::<()>().provider(provider).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        let first = rx.recv().await.expect("should have at least one event");
        assert_eq!(first.sequence, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_sequences_have_no_gaps() -> anyhow::Result<()> {
        // Use a tool call to generate more events (Start, ToolCallStart, ToolCallEnd, Text, TurnComplete, Done, etc.)
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("t1", "echo", json!({"message": "test"})),
            MockProvider::text_response("Done"),
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);
        let agent = builder::<()>().provider(provider).tools(tools).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Go".into()),
            ToolContext::new(()),
        );

        let mut sequences = Vec::new();
        while let Some(envelope) = rx.recv().await {
            sequences.push(envelope.sequence);
        }

        let expected: Vec<u64> = (0..sequences.len() as u64).collect();
        assert_eq!(
            sequences, expected,
            "sequences should be 0, 1, 2, ... with no gaps"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_timestamps_are_non_decreasing() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let agent = builder::<()>().provider(provider).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        let mut envelopes = Vec::new();
        while let Some(envelope) = rx.recv().await {
            envelopes.push(envelope);
        }

        for pair in envelopes.windows(2) {
            assert!(
                pair[1].timestamp >= pair[0].timestamp,
                "timestamp went backwards: {:?} -> {:?}",
                pair[0].timestamp,
                pair[1].timestamp,
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_separate_runs_have_independent_sequences() -> anyhow::Result<()> {
        let provider_a = MockProvider::new(vec![MockProvider::text_response("A")]);
        let provider_b = MockProvider::new(vec![MockProvider::text_response("B")]);

        let agent_a = builder::<()>().provider(provider_a).build();
        let agent_b = builder::<()>().provider(provider_b).build();

        let (mut rx_a, _) = agent_a.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );
        let (mut rx_b, _) = agent_b.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        let first_a = rx_a.recv().await.expect("run A should emit events");
        let first_b = rx_b.recv().await.expect("run B should emit events");

        // Both runs start at sequence 0
        assert_eq!(first_a.sequence, 0);
        assert_eq!(first_b.sequence, 0);

        // But event_ids are different
        assert_ne!(first_a.event_id, first_b.event_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_event_ids_are_valid_uuid_v4() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hi")]);
        let agent = builder::<()>().provider(provider).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Hi".into()),
            ToolContext::new(()),
        );

        while let Some(envelope) = rx.recv().await {
            assert_eq!(
                envelope.event_id.get_version(),
                Some(uuid::Version::Random),
                "event_id should be UUID v4"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_envelope_with_tool_calls_maintains_invariants() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("t1", "echo", json!({"message": "a"})),
            MockProvider::tool_use_response("t2", "echo", json!({"message": "b"})),
            MockProvider::text_response("All done"),
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);
        let agent = builder::<()>().provider(provider).tools(tools).build();

        let (mut rx, _) = agent.run(
            ThreadId::new(),
            AgentInput::Text("Go".into()),
            ToolContext::new(()),
        );

        let mut envelopes = Vec::new();
        while let Some(envelope) = rx.recv().await {
            envelopes.push(envelope);
        }

        // All event_ids unique
        let ids: std::collections::HashSet<uuid::Uuid> =
            envelopes.iter().map(|e| e.event_id).collect();
        assert_eq!(ids.len(), envelopes.len(), "all event_ids must be unique");

        // Sequences: 0, 1, 2, ... no gaps
        let expected: Vec<u64> = (0..envelopes.len() as u64).collect();
        let actual: Vec<u64> = envelopes.iter().map(|e| e.sequence).collect();
        assert_eq!(actual, expected, "sequences must be contiguous from 0");

        // Timestamps non-decreasing
        for pair in envelopes.windows(2) {
            assert!(pair[1].timestamp >= pair[0].timestamp);
        }

        // Should contain tool call events wrapped in envelopes
        assert!(
            envelopes
                .iter()
                .any(|e| matches!(e.event, AgentEvent::ToolCallStart { .. }))
        );
        assert!(
            envelopes
                .iter()
                .any(|e| matches!(e.event, AgentEvent::ToolCallEnd { .. }))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_deferred_tool_confirmation_flow() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("tool_1", "deferred_echo", json!({"message": "test"})),
            MockProvider::text_response("Deferred flow complete"),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register_deferred(DeferredEchoTool);

        let agent = builder::<()>().provider(provider).tools(tools).build();
        let thread_id = ThreadId::new();

        // Turn 1: reaches awaiting confirmation after deferred pre-runtime
        let (_events_1, outcome_rx_1) = agent.run_turn(
            thread_id.clone(),
            AgentInput::Text("Run deferred".to_string()),
            ToolContext::new(()),
        );
        let outcome_1 = outcome_rx_1.await?;

        let (continuation, tool_call_id) = match outcome_1 {
            TurnOutcome::AwaitingConfirmation {
                continuation,
                tool_call_id,
                ..
            } => (continuation, tool_call_id),
            other => panic!("Expected AwaitingConfirmation, got {other:?}"),
        };

        // Turn 2: confirm and execute commit
        let (_events_2, outcome_rx_2) = agent.run_turn(
            thread_id.clone(),
            AgentInput::Resume {
                continuation,
                tool_call_id,
                confirmed: true,
                rejection_reason: None,
            },
            ToolContext::new(()),
        );
        let outcome_2 = outcome_rx_2.await?;
        assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

        // Turn 3: continue and finish
        let (_events_3, outcome_rx_3) =
            agent.run_turn(thread_id, AgentInput::Continue, ToolContext::new(()));
        let outcome_3 = outcome_rx_3.await?;
        assert!(matches!(outcome_3, TurnOutcome::Done { .. }));

        Ok(())
    }

    #[tokio::test]
    async fn test_listen_tool_confirmation_flow() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
            MockProvider::text_response("Listen flow complete"),
        ]);

        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let mut tools = ToolRegistry::new();
        tools.register_listen(ListenEchoTool {
            cancel_calls: cancel_calls.clone(),
        });

        let agent = builder::<()>().provider(provider).tools(tools).build();
        let thread_id = ThreadId::new();

        // Turn 1: reaches awaiting confirmation after listen pre-runtime
        let (_events_1, outcome_rx_1) = agent.run_turn(
            thread_id.clone(),
            AgentInput::Text("Run listen tool".to_string()),
            ToolContext::new(()),
        );
        let outcome_1 = outcome_rx_1.await?;

        let (continuation, tool_call_id) = match outcome_1 {
            TurnOutcome::AwaitingConfirmation {
                continuation,
                tool_call_id,
                ..
            } => (continuation, tool_call_id),
            other => panic!("Expected AwaitingConfirmation, got {other:?}"),
        };

        // Turn 2: confirm and execute
        let (_events_2, outcome_rx_2) = agent.run_turn(
            thread_id.clone(),
            AgentInput::Resume {
                continuation,
                tool_call_id,
                confirmed: true,
                rejection_reason: None,
            },
            ToolContext::new(()),
        );
        let outcome_2 = outcome_rx_2.await?;
        assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

        // Turn 3: continue and finish
        let (_events_3, outcome_rx_3) =
            agent.run_turn(thread_id, AgentInput::Continue, ToolContext::new(()));
        let outcome_3 = outcome_rx_3.await?;
        assert!(matches!(outcome_3, TurnOutcome::Done { .. }));
        assert_eq!(cancel_calls.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_listen_tool_rejection_cancels_operation() -> anyhow::Result<()> {
        let provider = MockProvider::new(vec![
            MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
            MockProvider::text_response("Rejected flow complete"),
        ]);

        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let mut tools = ToolRegistry::new();
        tools.register_listen(ListenEchoTool {
            cancel_calls: cancel_calls.clone(),
        });

        let agent = builder::<()>().provider(provider).tools(tools).build();
        let thread_id = ThreadId::new();

        let (_events_1, outcome_rx_1) = agent.run_turn(
            thread_id.clone(),
            AgentInput::Text("Run listen tool".to_string()),
            ToolContext::new(()),
        );
        let outcome_1 = outcome_rx_1.await?;

        let (continuation, tool_call_id) = match outcome_1 {
            TurnOutcome::AwaitingConfirmation {
                continuation,
                tool_call_id,
                ..
            } => (continuation, tool_call_id),
            other => panic!("Expected AwaitingConfirmation, got {other:?}"),
        };

        let (_events_2, outcome_rx_2) = agent.run_turn(
            thread_id,
            AgentInput::Resume {
                continuation,
                tool_call_id,
                confirmed: false,
                rejection_reason: Some("nope".to_string()),
            },
            ToolContext::new(()),
        );
        let _ = outcome_rx_2.await?;

        assert_eq!(cancel_calls.load(Ordering::SeqCst), 1);
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
