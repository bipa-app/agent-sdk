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

mod builder;
mod helpers;
mod idempotency;
mod listen;
mod llm;
mod run_loop;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;
mod tool_execution;
mod turn;
mod types;

use self::run_loop::{run_loop, run_single_turn};
use self::types::{RunLoopParameters, TurnParameters};

pub use self::builder::AgentLoopBuilder;

use crate::context::{CompactionConfig, ContextCompactor};
use crate::events::{AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::LlmProvider;
use crate::stores::{MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, AgentRunState, ThreadId, TurnOutcome};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Handle to a persistent agent thread.
///
/// Returned by [`AgentLoop::run_persistent`]. Allows the caller to send
/// new messages to the running agent, observe events, and cancel execution.
pub struct AgentHandle {
    /// Send new messages to the running agent. The agent will process
    /// them as new user turns after completing the current turn.
    pub input_tx: mpsc::Sender<AgentInput>,
    /// Receive agent events (text, tool calls, progress).
    pub events_rx: mpsc::Receiver<AgentEventEnvelope>,
    /// Final run state (sent once when the agent completes).
    pub state_rx: oneshot::Receiver<AgentRunState>,
    /// Cancel the running agent.
    pub cancel_token: CancellationToken,
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
/// while let Some(envelope) = events.recv().await {
///     match envelope.event {
///         AgentEvent::Text {
///             message_id: _,
///             text,
///         } => println!("{}", text),
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
    pub(super) provider: Arc<P>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) config: AgentConfig,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) compactor: Option<Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<Arc<dyn crate::observability::ObservabilityStore>>,
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
            compactor: None,
            execution_store: None,
            #[cfg(feature = "otel")]
            observability_store: None,
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
            compactor: None,
            execution_store: None,
            #[cfg(feature = "otel")]
            observability_store: None,
        }
    }

    /// Set the observability store for `GenAI` payload capture.
    ///
    /// When set, the store is called at each LLM request boundary to decide
    /// whether payloads are inlined on spans, externalized, or omitted.
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn with_observability_store(
        mut self,
        store: impl crate::observability::ObservabilityStore + 'static,
    ) -> Self {
        self.observability_store = Some(Arc::new(store));
        self
    }

    /// Run the agent loop.
    ///
    /// This method allows the agent to pause when a tool requires confirmation,
    /// returning an `AgentRunState::AwaitingConfirmation` that contains the
    /// state needed to resume.
    ///
    /// When the `cancel_token` is cancelled, the agent will stop after the
    /// current turn completes (no new turns will start). The final state will
    /// be `AgentRunState::Cancelled`.
    ///
    /// # Arguments
    ///
    /// * `thread_id` - The thread identifier for this conversation
    /// * `input` - Either a new text message or a resume with confirmation decision
    /// * `tool_context` - Context passed to tools
    /// * `cancel_token` - Token to signal cancellation from outside
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
    /// let cancel = CancellationToken::new();
    /// let (events, final_state) = agent.run(
    ///     thread_id,
    ///     AgentInput::Text("Hello".to_string()),
    ///     tool_ctx,
    ///     cancel.clone(),
    /// );
    ///
    /// while let Some(envelope) = events.recv().await {
    ///     // Handle events...
    /// }
    ///
    /// match final_state.await.unwrap() {
    ///     AgentRunState::Done { .. } => { /* completed */ }
    ///     AgentRunState::Cancelled { .. } => { /* user cancelled */ }
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
    ///             cancel.clone(),
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
        cancel_token: CancellationToken,
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
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
            let result = run_loop(RunLoopParameters {
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
                compactor,
                execution_store,
                cancel_token,
                input_rx: None,
                #[cfg(feature = "otel")]
                observability_store,
            })
            .await;

            let _ = state_tx.send(result);
        };

        #[cfg(feature = "otel")]
        let task = {
            use opentelemetry::trace::FutureExt;
            task.with_context(parent_cx)
        };

        tokio::spawn(task);

        (event_rx, state_rx)
    }

    /// Run the agent with a persistent input channel.
    ///
    /// Unlike [`run`], this returns an [`AgentHandle`] that allows the caller
    /// to inject new user messages into the running agent via `input_tx`.
    /// The agent will process the initial input, then wait for new messages
    /// on the channel between turns instead of exiting on `Done`.
    ///
    /// The agent exits when:
    /// - The `input_tx` sender is dropped (no more messages)
    /// - The `cancel_token` is cancelled
    /// - Max turns exceeded
    pub fn run_persistent(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
    ) -> AgentHandle
    where
        Ctx: Clone,
    {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (state_tx, state_rx) = oneshot::channel();
        let (input_tx, input_rx) = mpsc::channel(32);
        let seq = SequenceCounter::new();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        let cancel_handle = cancel_token.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
            let result = run_loop(RunLoopParameters {
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
                compactor,
                execution_store,
                cancel_token,
                input_rx: Some(input_rx),
                #[cfg(feature = "otel")]
                observability_store,
            })
            .await;

            let _ = state_tx.send(result);
        };

        #[cfg(feature = "otel")]
        let task = {
            use opentelemetry::trace::FutureExt;
            task.with_context(parent_cx)
        };

        tokio::spawn(task);

        AgentHandle {
            input_tx,
            events_rx: event_rx,
            state_rx,
            cancel_token: cancel_handle,
        }
    }

    /// Run a single turn of the agent loop.
    ///
    /// Unlike `run()`, this method executes exactly one turn and returns control
    /// to the caller. This enables external orchestration where each turn can be
    /// dispatched as a separate message (e.g., via Artemis or another message queue).
    ///
    /// When the `cancel_token` is cancelled, the turn will be aborted before
    /// starting execution and return `TurnOutcome::Cancelled`.
    ///
    /// # Arguments
    ///
    /// * `thread_id` - The thread identifier for this conversation
    /// * `input` - Text to start, Resume after confirmation, or Continue after a turn
    /// * `tool_context` - Context passed to tools
    /// * `cancel_token` - Token to signal cancellation from outside
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
    /// - `Cancelled` - Turn was cancelled via the token
    /// - `Error` - An error occurred
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cancel = CancellationToken::new();
    /// // Start conversation
    /// let (events, outcome) = agent.run_turn(
    ///     thread_id.clone(),
    ///     AgentInput::Text("What is 2+2?".to_string()),
    ///     tool_ctx.clone(),
    ///     cancel,
    /// ).await;
    ///
    /// // Process events...
    /// while let Some(_envelope) = events.recv().await { /* ... */ }
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
        cancel_token: CancellationToken,
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
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
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
                compactor,
                execution_store,
                cancel_token,
                #[cfg(feature = "otel")]
                observability_store,
            })
            .await;

            let _ = outcome_tx.send(result);
        };

        #[cfg(feature = "otel")]
        let task = {
            use opentelemetry::trace::FutureExt;
            task.with_context(parent_cx)
        };

        tokio::spawn(task);

        (event_rx, outcome_rx)
    }
}
