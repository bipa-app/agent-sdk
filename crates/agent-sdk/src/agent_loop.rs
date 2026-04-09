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
//! 6. Persists events throughout to the configured event store
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
//!     .event_store(event_store)
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
use crate::types::TurnOptions;

pub use self::builder::AgentLoopBuilder;

use crate::authority::{EventAuthority, LocalEventAuthority};
use crate::context::{CompactionConfig, ContextCompactor};
use crate::hooks::AgentHooks;
use crate::llm::LlmProvider;
use crate::stores::{EventStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, AgentRunState, ThreadId};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Handle to a persistent agent thread.
///
/// Returned by [`AgentLoop::run_persistent`]. Allows the caller to send
/// new messages to the running agent and cancel execution.
pub struct AgentHandle {
    /// Send new messages to the running agent. The agent will process
    /// them as new user turns after completing the current turn.
    pub input_tx: mpsc::Sender<AgentInput>,
    /// Final run state (sent once when the agent completes).
    pub state_rx: oneshot::Receiver<AgentRunState>,
    /// Cancel the running agent.
    pub cancel_token: CancellationToken,
}

/// Configuration bundle for constructing an [`AgentLoop`] with compaction.
pub struct AgentLoopCompactionConfig {
    pub agent_config: AgentConfig,
    pub compaction_config: CompactionConfig,
}

impl AgentLoopCompactionConfig {
    #[must_use]
    pub const fn new(agent_config: AgentConfig, compaction_config: CompactionConfig) -> Self {
        Self {
            agent_config,
            compaction_config,
        }
    }
}

/// The main agent loop that orchestrates LLM calls and tool execution.
///
/// `AgentLoop` is the core component that:
/// - Manages conversation state via message and state stores
/// - Calls the LLM provider and processes responses
/// - Executes tools through the tool registry
/// - Persists events to the configured event store
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
/// # Event Storage
///
/// Every loop instance requires an [`EventStore`] configured at construction
/// time. Events are written to that store for the entire lifecycle of the loop,
/// and callers read them back from the store instead of receiving an in-process
/// channel from the runtime.
///
/// # Running the Agent
///
/// ```ignore
/// let final_state = agent.run(
///     thread_id,
///     AgentInput::Text("Hello!".to_string()),
///     tool_ctx,
/// );
/// let state = final_state.await?;
/// let events = event_store.get_events(&thread_id).await?;
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
    pub(super) event_store: Arc<dyn EventStore>,
    pub(super) event_authority: Option<Arc<dyn EventAuthority>>,
    pub(super) config: AgentConfig,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) compactor: Option<Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: Arc<dyn crate::hooks::ToolAuditSink>,
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
        event_store: Arc<dyn EventStore>,
        config: AgentConfig,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(hooks),
            message_store: Arc::new(message_store),
            state_store: Arc::new(state_store),
            event_store,
            event_authority: None,
            config,
            compaction_config: None,
            compactor: None,
            execution_store: None,
            audit_sink: Arc::new(crate::hooks::NoopAuditSink),
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
        event_store: Arc<dyn EventStore>,
        config: AgentLoopCompactionConfig,
    ) -> Self {
        let AgentLoopCompactionConfig {
            agent_config,
            compaction_config,
        } = config;
        Self {
            provider: Arc::new(provider),
            tools: Arc::new(tools),
            hooks: Arc::new(hooks),
            message_store: Arc::new(message_store),
            state_store: Arc::new(state_store),
            event_store,
            event_authority: None,
            config: agent_config,
            compaction_config: Some(compaction_config),
            compactor: None,
            execution_store: None,
            audit_sink: Arc::new(crate::hooks::NoopAuditSink),
            #[cfg(feature = "otel")]
            observability_store: None,
        }
    }

    /// Set the authoritative tool audit sink.
    ///
    /// When set, the loop emits a [`ToolAuditRecord`](crate::ToolAuditRecord)
    /// at every tool-lifecycle transition (blocked, requires-confirmation,
    /// cached, replayed, invalidated, completed, persistence-failed).
    ///
    /// The default is [`NoopAuditSink`](crate::hooks::NoopAuditSink) which
    /// discards every record — suitable for local/CLI usage. Servers should
    /// swap in a durable sink.
    #[must_use]
    pub fn with_audit_sink(mut self, sink: impl crate::hooks::ToolAuditSink + 'static) -> Self {
        self.audit_sink = Arc::new(sink);
        self
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

    /// Resolve the event authority for this run.
    ///
    /// If an external authority was configured via the builder, use it.
    /// Otherwise create a fresh [`LocalEventAuthority`] that starts at 0
    /// (the pre-existing local/CLI behaviour).
    fn resolve_authority(&self) -> Arc<dyn EventAuthority> {
        self.event_authority
            .clone()
            .unwrap_or_else(|| Arc::new(LocalEventAuthority::new()))
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
    /// A [`oneshot::Receiver`] that resolves to the final run state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cancel = CancellationToken::new();
    /// let final_state = agent.run(
    ///     thread_id,
    ///     AgentInput::Text("Hello".to_string()),
    ///     tool_ctx,
    ///     cancel.clone(),
    /// );
    ///
    /// match final_state.await.unwrap() {
    ///     AgentRunState::Done { .. } => { /* completed */ }
    ///     AgentRunState::Cancelled { .. } => { /* user cancelled */ }
    ///     AgentRunState::AwaitingConfirmation { continuation, .. } => {
    ///         // Get user decision, then resume:
    ///         let state2 = agent.run(
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
    ) -> oneshot::Receiver<AgentRunState>
    where
        Ctx: Clone,
    {
        let (state_rx, _handle) = self.run_abortable(thread_id, input, tool_context, cancel_token);
        state_rx
    }

    /// Like [`run`](Self::run), but also returns the [`tokio::task::JoinHandle`] for the
    /// spawned task.
    ///
    /// Callers that need to forcibly abort the agent loop (e.g. subagent
    /// timeout) can call [`tokio::task::JoinHandle::abort`] on the returned handle.
    /// Aborting the handle drops the in-flight LLM stream immediately
    /// instead of waiting for the current turn to finish.
    pub fn run_abortable(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
    ) -> (
        oneshot::Receiver<AgentRunState>,
        tokio::task::JoinHandle<()>,
    )
    where
        Ctx: Clone,
    {
        let (state_tx, state_rx) = oneshot::channel();
        let authority = self.resolve_authority();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let event_store = Arc::clone(&self.event_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        let audit_sink = Arc::clone(&self.audit_sink);
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
            let result = run_loop(RunLoopParameters {
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

        let handle = tokio::spawn(task);

        (state_rx, handle)
    }

    /// Run the agent with a persistent input channel.
    ///
    /// Unlike [`Self::run`], this returns an [`AgentHandle`] that allows the caller
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
        let (state_tx, state_rx) = oneshot::channel();
        let (input_tx, input_rx) = mpsc::channel(32);
        let authority = self.resolve_authority();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let event_store = Arc::clone(&self.event_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        let audit_sink = Arc::clone(&self.audit_sink);
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        let cancel_handle = cancel_token.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
            let result = run_loop(RunLoopParameters {
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
            state_rx,
            cancel_token: cancel_handle,
        }
    }

    /// Run a single turn of the agent loop — the authoritative server boundary.
    ///
    /// Unlike `run()`, this method executes exactly one turn **directly in the
    /// caller's task** (no `tokio::spawn`) and returns the result inline. This
    /// enables external orchestration where each turn can be dispatched as a
    /// separate message (e.g., via Artemis or another message queue).
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
    /// * `options` - Execution options (tool runtime strategy, durability)
    ///
    /// # Returns
    ///
    /// A [`crate::types::TurnOutcome`] returned only after the configured event store's
    /// `finish_turn(thread_id, turn)` barrier has completed.
    ///
    /// # Turn Outcomes
    ///
    /// - `NeedsMoreTurns` - Turn completed, call again with `AgentInput::Continue`
    /// - `Done` - Agent completed successfully
    /// - `AwaitingConfirmation` - Tool needs confirmation, call again with `AgentInput::Resume`
    /// - `PendingToolCalls` - Tools need external execution (only with `ToolRuntime::External`)
    /// - `Cancelled` - Turn was cancelled via the token
    /// - `Error` - An error occurred
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use agent_sdk::{InMemoryEventStore, TurnOptions};
    ///
    /// let cancel = CancellationToken::new();
    /// let event_store = Arc::new(InMemoryEventStore::new());
    /// let outcome = agent.run_turn(
    ///     thread_id.clone(),
    ///     AgentInput::Text("What is 2+2?".to_string()),
    ///     tool_ctx.clone(),
    ///     cancel,
    ///     TurnOptions::default(),
    /// ).await;
    ///
    /// let events = event_store.get_events(&thread_id).await?;
    ///
    /// // Check outcome
    /// match outcome {
    ///     TurnOutcome::NeedsMoreTurns { turn, .. } => {
    ///         // Dispatch another message to continue
    ///     }
    ///     TurnOutcome::Done { .. } => {
    ///         // Conversation complete
    ///     }
    ///     TurnOutcome::PendingToolCalls { tool_calls, .. } => {
    ///         // Execute tools externally, then call run_turn with Continue
    ///     }
    ///     _ => {}
    /// }
    /// ```
    pub async fn run_turn(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        options: TurnOptions,
    ) -> crate::types::TurnOutcome
    where
        Ctx: Clone,
    {
        let authority = self.resolve_authority();

        run_single_turn(TurnParameters {
            event_store: Arc::clone(&self.event_store),
            authority,
            thread_id,
            input,
            tool_context,
            provider: Arc::clone(&self.provider),
            tools: Arc::clone(&self.tools),
            hooks: Arc::clone(&self.hooks),
            message_store: Arc::clone(&self.message_store),
            state_store: Arc::clone(&self.state_store),
            config: self.config.clone(),
            compaction_config: self.compaction_config.clone(),
            compactor: self.compactor.clone(),
            execution_store: self.execution_store.clone(),
            audit_sink: Arc::clone(&self.audit_sink),
            cancel_token,
            turn_options: options,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store.clone(),
        })
        .await
    }
}
