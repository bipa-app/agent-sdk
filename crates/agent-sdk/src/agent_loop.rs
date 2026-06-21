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

mod budget;
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
use crate::events::{AgentEvent, AgentEventEnvelope};
use crate::hooks::AgentHooks;
use crate::llm::LlmProvider;
use crate::stores::{EventStore, MessageStore, StateStore, StoredTurnEvents, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentError, AgentInput, AgentRunState, RunOptions, ThreadId};
use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Bound on the [`AgentLoop::run_stream`] tee channel.
///
/// The event store is the durable source of truth, so the live stream is a
/// best-effort mirror. Bounding the channel keeps a slow (or stalled) stream
/// consumer from growing memory without limit: once this many events are
/// buffered, [`TeeEventStore::append`] drops the newest event rather than
/// blocking the run loop (whose forward progress and persistence must not
/// depend on a consumer's read rate). Callers that need every event should
/// read the configured [`EventStore`] back instead of relying on the stream.
const RUN_STREAM_CHANNEL_CAPACITY: usize = 1024;

/// An [`EventStore`] decorator that forwards a clone of every appended
/// [`AgentEvent`] to a bounded channel before delegating to the wrapped
/// store.
///
/// This is the tee behind [`AgentLoop::run_stream`]: the run loop writes
/// every event through the configured store as usual, and the matching
/// [`AgentEvent`] is mirrored onto the stream so callers consume events live
/// without implementing an [`EventStore`]. The forward is best-effort and
/// lossy under backpressure — if the consumer has dropped the stream, or is
/// too slow and the bounded buffer ([`RUN_STREAM_CHANNEL_CAPACITY`]) is full,
/// the send is skipped (the newest event is dropped) and the run continues
/// unaffected. The dropped events remain durably recorded in `inner`.
struct TeeEventStore {
    inner: Arc<dyn EventStore>,
    tx: mpsc::Sender<AgentEvent>,
}

#[async_trait]
impl EventStore for TeeEventStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> anyhow::Result<()> {
        // `try_send` (never `send().await`): the run loop must not stall on a
        // slow stream consumer. A `Full` error means the consumer is behind,
        // so the event is dropped from the live stream only — it is still
        // persisted to `inner` below and readable via the store.
        if let Err(mpsc::error::TrySendError::Full(_)) = self.tx.try_send(envelope.event.clone()) {
            log::debug!(
                "run_stream tee channel full (capacity {RUN_STREAM_CHANNEL_CAPACITY}); \
                 dropping event from live stream (still persisted to the store)"
            );
        }
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> anyhow::Result<()> {
        self.inner.finish_turn(thread_id, turn).await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> anyhow::Result<Option<StoredTurnEvents>> {
        self.inner.get_turn(thread_id, turn).await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<StoredTurnEvents>> {
        self.inner.get_turns(thread_id).await
    }

    async fn get_events(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<AgentEventEnvelope>> {
        self.inner.get_events(thread_id).await
    }

    async fn event_count(&self, thread_id: &ThreadId) -> anyhow::Result<usize> {
        self.inner.event_count(thread_id).await
    }

    async fn get_events_since(
        &self,
        thread_id: &ThreadId,
        offset: usize,
    ) -> anyhow::Result<Vec<AgentEventEnvelope>> {
        self.inner.get_events_since(thread_id, offset).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> anyhow::Result<()> {
        self.inner.clear(thread_id).await
    }
}

/// Run the agent loop with panic isolation at the spawned-task boundary.
///
/// A panic anywhere inside the run loop — most importantly inside the
/// LLM provider call or the compaction provider call, neither of which
/// is otherwise guarded — would unwind the spawned task, drop the
/// `state_tx` oneshot, and surface to the caller as an opaque
/// [`oneshot::error::RecvError`] (and, for subagents, be misclassified
/// as `Disconnected`). Catching the unwind here turns it into a
/// structured [`AgentRunState::Error`] that the task can still send on
/// `state_tx`, so the run ends observably rather than silently tearing
/// down the channel.
///
/// `AssertUnwindSafe` is sound at this boundary: the run loop owns its
/// `RunLoopParameters` by value, so nothing it touches outlives the
/// caught panic and is observed afterwards. The only value produced is
/// the returned `AgentRunState`. Tool-level panics are already caught
/// closer to the tool boundary (see
/// `agent_loop::helpers::catch_tool_panic`) so the assistant
/// `tool_use` / `tool_result` history stays balanced; this guard is the
/// outer safety net for everything else.
async fn run_loop_isolated<Ctx, P, H, M, S>(
    params: RunLoopParameters<Ctx, P, H, M, S>,
) -> AgentRunState
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    match AssertUnwindSafe(run_loop(params)).catch_unwind().await {
        Ok(state) => state,
        Err(payload) => {
            let message = self::helpers::panic_payload_message(payload.as_ref());
            log::error!("agent run loop panicked: {message}");
            AgentRunState::Error(AgentError::new(
                format!("Agent run panicked: {message}"),
                false,
            ))
        }
    }
}

/// Drop a spawned run task's [`tokio::task::JoinHandle`], logging a
/// `debug!` to make the detach visible.
///
/// `run` / `run_with_options` intentionally drop the handle: the run is
/// stopped through the cancel token or the per-tool timeout, not by
/// aborting the task. Surfacing the detach at `debug` level gives a
/// breadcrumb when a subprocess-backed tool that ignores the
/// cooperative-cancel contract leaks a process after cancellation.
fn warn_on_detached_run_handle(handle: tokio::task::JoinHandle<()>) {
    log::debug!(
        "agent run JoinHandle dropped (task detached); the run can only be \
         stopped via its cancel token or per-tool timeout. Subprocess-backed \
         tools must honour kill_on_drop or a token-aware kill to avoid leaks"
    );
    drop(handle);
}

/// Await a run's state receiver, mapping a dropped channel to an `anyhow`
/// error so `run`/`run_with_options` can present an `impl Future` instead of
/// a bare [`oneshot::Receiver`].
///
/// A panic inside the run is already converted to
/// [`AgentRunState::Error`] before the channel send (see
/// [`run_loop_isolated`]), so the only way `recv` fails is the sender being
/// dropped without sending — a runtime shutdown rather than an agent error.
async fn recv_run_state(
    state_rx: oneshot::Receiver<AgentRunState>,
) -> anyhow::Result<AgentRunState> {
    state_rx
        .await
        .map_err(|_| anyhow::anyhow!("agent run task was dropped before reporting a final state"))
}

/// Handle to a persistent agent thread.
///
/// Returned by [`AgentLoop::run_persistent`]. Allows the caller to send
/// new messages to the running agent and cancel execution.
pub struct AgentHandle {
    /// Send new messages to the running agent. The agent processes them as new
    /// user turns once it parks between turns.
    ///
    /// Only [`AgentInput::Text`] and [`AgentInput::Message`] are supported on
    /// this channel — they are the only variants that represent a fresh user
    /// turn. Injecting [`AgentInput::Resume`], [`AgentInput::SubmitToolResults`],
    /// or [`AgentInput::Continue`] ends the run with [`AgentRunState::Error`]
    /// (those belong to the single-turn `run_turn` flow, not the persistent
    /// loop). Dropping the sender ends the run cleanly with `Done`.
    pub input_tx: mpsc::Sender<AgentInput>,
    /// Final run state (sent once when the agent completes).
    pub state_rx: oneshot::Receiver<AgentRunState>,
    /// Cancel the running agent.
    pub cancel_token: CancellationToken,
}

/// Inputs shared by the three `spawn_run_loop` callers
/// (`run_abortable_with_options`, `run_persistent_with_options`,
/// `run_stream_with_options`). Bundled so the private spawn helper takes a
/// single argument instead of a long positional list.
struct SpawnRunLoopParams<Ctx> {
    event_store: Arc<dyn EventStore>,
    thread_id: ThreadId,
    input: AgentInput,
    tool_context: ToolContext<Ctx>,
    cancel_token: CancellationToken,
    run_options: RunOptions,
    input_rx: Option<mpsc::Receiver<AgentInput>>,
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
    pub(super) reminder_config: Option<crate::reminders::ReminderConfig>,
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
            reminder_config: None,
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
            reminder_config: None,
            #[cfg(feature = "otel")]
            observability_store: None,
        }
    }

    /// Set the authoritative tool audit sink.
    ///
    /// When set, the loop emits a [`ToolAuditRecord`](crate::advanced::ToolAuditRecord)
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
    /// When the `cancel_token` is cancelled, the agent interrupts in-flight
    /// work at the SDK boundary: the LLM stream and any non-streaming LLM call
    /// are raced against the token (see `agent_loop/llm.rs`), and in-flight
    /// tool executions are dropped with balanced `ToolResult`s synthesized so
    /// the conversation history stays consistent. The run then ends with
    /// `AgentRunState::Cancelled` — cancellation is *not* deferred to a turn
    /// boundary. Tools that hold OS resources must honour the
    /// [cooperative-cancel contract](crate::tools::Tool#cooperative-cancellation);
    /// see the section below.
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
    /// A future that resolves to the final [`AgentRunState`]. Awaiting it
    /// drives the run to completion:
    ///
    /// ```ignore
    /// let final_state = agent.run(thread_id, input, tool_ctx, cancel).await?;
    /// ```
    ///
    /// The future is `'static` — the run is already spawned on a Tokio task
    /// before this returns, so dropping the future does **not** stop the run
    /// (use `cancel_token`). Awaiting it only waits for the result.
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
    /// ).await?;
    ///
    /// match final_state {
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
    ///         ).await?;
    ///     }
    ///     AgentRunState::Error(e) => { /* handle error */ }
    /// }
    /// ```
    /// # Cancellation, timeout, and the dropped `JoinHandle`
    ///
    /// `run` spawns the agent loop on a Tokio task and returns a future over
    /// the state channel — it **drops** the task's
    /// [`tokio::task::JoinHandle`]. Dropping a `JoinHandle` *detaches* the
    /// task rather than aborting it, so the only ways to stop an in-flight
    /// run are the `cancel_token` (cooperative) or the per-tool
    /// [`AgentConfig::tool_timeout_ms`](crate::types::AgentConfig::tool_timeout_ms)
    /// boundary. Callers that need to forcibly abort must use
    /// [`run_abortable`](Self::run_abortable) and keep the handle.
    ///
    /// Because the handle is dropped, a tool that holds a subprocess open
    /// must obey the
    /// [cooperative-cancel contract](crate::tools::Tool#cooperative-cancellation)
    /// (`kill_on_drop` or a token-aware `kill`) or the subprocess will
    /// outlive the cancelled / timed-out run. A `debug!` is logged here so
    /// the detach is visible when chasing a leaked subprocess.
    ///
    /// # Errors
    ///
    /// Returns an error only if the spawned run task is dropped before it can
    /// report a final state (e.g. a runtime shutdown). A panic inside the run
    /// is caught and surfaced as [`AgentRunState::Error`], not as an `Err`.
    pub fn run(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
    ) -> impl Future<Output = anyhow::Result<AgentRunState>> + Send + 'static
    where
        Ctx: Clone,
    {
        let (state_rx, handle) = self.run_abortable(thread_id, input, tool_context, cancel_token);
        warn_on_detached_run_handle(handle);
        recv_run_state(state_rx)
    }

    /// Like [`run`](Self::run), but with caller-supplied trace metadata.
    ///
    /// Equivalent to `run` except that the supplied [`RunOptions`]
    /// configure session/user IDs (propagated as `session.id` /
    /// `user.id` baggage), `langfuse.trace.{name,tags,metadata.*,
    /// input,output}`, `langfuse.{release,environment}`, and the
    /// trace-text truncation ceiling.
    ///
    /// Use this instead of `run` whenever the consumer needs the
    /// SDK to populate Langfuse trace metadata; `run` itself
    /// continues to delegate here with `RunOptions::default()`.
    ///
    /// # Errors
    ///
    /// See [`run`](Self::run).
    pub fn run_with_options(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        run_options: RunOptions,
    ) -> impl Future<Output = anyhow::Result<AgentRunState>> + Send + 'static
    where
        Ctx: Clone,
    {
        let (state_rx, handle) = self.run_abortable_with_options(
            thread_id,
            input,
            tool_context,
            cancel_token,
            run_options,
        );
        warn_on_detached_run_handle(handle);
        recv_run_state(state_rx)
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
        self.run_abortable_with_options(
            thread_id,
            input,
            tool_context,
            cancel_token,
            RunOptions::default(),
        )
    }

    /// Like [`run_abortable`](Self::run_abortable), but with
    /// caller-supplied trace metadata. See [`run_with_options`](Self::run_with_options).
    pub fn run_abortable_with_options(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        run_options: RunOptions,
    ) -> (
        oneshot::Receiver<AgentRunState>,
        tokio::task::JoinHandle<()>,
    )
    where
        Ctx: Clone,
    {
        self.spawn_run_loop(SpawnRunLoopParams {
            event_store: Arc::clone(&self.event_store),
            thread_id,
            input,
            tool_context,
            cancel_token,
            run_options,
            input_rx: None,
        })
    }

    /// Spawn the run loop on a Tokio task and return its state channel +
    /// join handle.
    ///
    /// Shared by [`run_abortable_with_options`](Self::run_abortable_with_options),
    /// [`run_persistent_with_options`](Self::run_persistent_with_options), and
    /// [`run_stream_with_options`](Self::run_stream_with_options) so all three
    /// build [`RunLoopParameters`] identically. The `event_store` is a
    /// parameter (not always `self.event_store`) so the streaming path can
    /// inject a teeing decorator.
    fn spawn_run_loop(
        &self,
        SpawnRunLoopParams {
            event_store,
            thread_id,
            input,
            tool_context,
            cancel_token,
            run_options,
            input_rx,
        }: SpawnRunLoopParams<Ctx>,
    ) -> (
        oneshot::Receiver<AgentRunState>,
        tokio::task::JoinHandle<()>,
    )
    where
        Ctx: Clone,
    {
        // `run_options` only feeds OTel root-span metadata. On
        // non-otel builds the value is genuinely not needed —
        // explicitly drop it so the unused-variable / needless-pass
        // lints stay quiet without us reaching for an
        // `#[allow(...)]`.
        #[cfg(not(feature = "otel"))]
        drop(run_options);

        let (state_tx, state_rx) = oneshot::channel();
        let authority = self.resolve_authority();

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let config = self.config.clone();
        let compaction_config = self.compaction_config.clone();
        let compactor = self.compactor.clone();
        let execution_store = self.execution_store.clone();
        let audit_sink = Arc::clone(&self.audit_sink);
        let reminder_config = self.reminder_config.clone();
        #[cfg(feature = "otel")]
        let observability_store = self.observability_store.clone();
        #[cfg(feature = "otel")]
        let parent_cx = crate::observability::context::capture_context();

        let task = async move {
            let result = run_loop_isolated(RunLoopParameters {
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
                input_rx,
                reminder_config,
                #[cfg(feature = "otel")]
                run_options,
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
    /// - The `input_tx` sender is dropped (no more messages) — reports `Done`
    /// - The `cancel_token` is cancelled — reports `Cancelled`
    /// - Max turns exceeded — reports `Error`
    /// - An unsupported input variant is injected, or appending an injected
    ///   message fails — reports `Error` (see [`AgentHandle::input_tx`])
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
        self.run_persistent_with_options(
            thread_id,
            input,
            tool_context,
            cancel_token,
            RunOptions::default(),
        )
    }

    /// Like [`run_persistent`](Self::run_persistent), but with
    /// caller-supplied trace metadata. See
    /// [`run_with_options`](Self::run_with_options).
    pub fn run_persistent_with_options(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        run_options: RunOptions,
    ) -> AgentHandle
    where
        Ctx: Clone,
    {
        let (input_tx, input_rx) = mpsc::channel(32);
        let cancel_handle = cancel_token.clone();

        let (state_rx, handle) = self.spawn_run_loop(SpawnRunLoopParams {
            event_store: Arc::clone(&self.event_store),
            thread_id,
            input,
            tool_context,
            cancel_token,
            run_options,
            input_rx: Some(input_rx),
        });
        // The persistent run is stopped via the cancel token or by dropping
        // `input_tx`, not by aborting the task, so detach the handle.
        drop(handle);

        AgentHandle {
            input_tx,
            state_rx,
            cancel_token: cancel_handle,
        }
    }

    /// Stream the agent's [`AgentEvent`]s live as they are emitted.
    ///
    /// Returns a [`Stream`] that yields each [`AgentEvent`] the moment the
    /// run loop writes it to the event store, so callers consume events
    /// in real time without implementing an [`EventStore`]. The same events
    /// are still persisted to the loop's configured store — the stream is an
    /// additional tee, not a replacement.
    ///
    /// The run is spawned on a Tokio task before this returns; the stream
    /// ends when the run finishes (or is cancelled via `cancel_token`).
    /// Dropping the stream does **not** stop the run — use `cancel_token`.
    ///
    /// This is additive alongside [`run`](Self::run) /
    /// [`run_persistent`](Self::run_persistent): use it when you want a
    /// live event feed without reading the store back.
    pub fn run_stream(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
    ) -> impl Stream<Item = AgentEvent> + Send + 'static
    where
        Ctx: Clone,
    {
        self.run_stream_with_options(
            thread_id,
            input,
            tool_context,
            cancel_token,
            RunOptions::default(),
        )
    }

    /// Like [`run_stream`](Self::run_stream), but with caller-supplied trace
    /// metadata. See [`run_with_options`](Self::run_with_options).
    pub fn run_stream_with_options(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        run_options: RunOptions,
    ) -> impl Stream<Item = AgentEvent> + Send + 'static
    where
        Ctx: Clone,
    {
        let (tx, rx) = mpsc::channel(RUN_STREAM_CHANNEL_CAPACITY);
        let event_store: Arc<dyn EventStore> = Arc::new(TeeEventStore {
            inner: Arc::clone(&self.event_store),
            tx,
        });

        let (state_rx, handle) = self.spawn_run_loop(SpawnRunLoopParams {
            event_store,
            thread_id,
            input,
            tool_context,
            cancel_token,
            run_options,
            input_rx: None,
        });
        // The run drives itself to completion; the stream only needs the
        // teed events. Detach the join handle and drop the state receiver —
        // when the run ends, the tee store (and its sender) drop, closing
        // the stream.
        warn_on_detached_run_handle(handle);
        drop(state_rx);

        tokio_stream::wrappers::ReceiverStream::new(rx)
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
    /// Every variant except [`crate::types::TurnOutcome::Error`] carries a
    /// structured [`crate::types::TurnSummary`] in the `summary` field. This
    /// summary is the **authoritative** server-facing outcome contract —
    /// it contains the provider/model provenance, response ID, stop reason,
    /// tool-call count, duration, and execution options for the turn. Server
    /// code should read from `summary` rather than the legacy per-variant
    /// fields (`total_turns`, `input_tokens`, `output_tokens`, …), which are
    /// retained only for backwards compatibility with local callers.
    ///
    /// # Turn Outcomes
    ///
    /// - `NeedsMoreTurns` - Turn completed, call again with `AgentInput::Continue`
    /// - `Done` - Agent completed successfully
    /// - `AwaitingConfirmation` - Tool needs confirmation, call again with `AgentInput::Resume`
    /// - `PendingToolCalls` - Tools need external execution (only with `ToolRuntime::External`)
    /// - `Cancelled` - Turn was cancelled via the token
    /// - `Error` - An error occurred (no summary attached)
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
    /// // Read server-facing metadata from the TurnSummary.
    /// if let Some(summary) = outcome.summary() {
    ///     println!(
    ///         "turn={} provider={} model={} stop={:?} response_id={:?}",
    ///         summary.turn,
    ///         summary.provenance.provider,
    ///         summary.provenance.model,
    ///         summary.stop_reason,
    ///         summary.response_id,
    ///     );
    /// }
    ///
    /// // Branch on the variant for flow control.
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
        self.run_turn_with_options(
            thread_id,
            input,
            tool_context,
            cancel_token,
            options,
            RunOptions::default(),
        )
        .await
    }

    /// Like [`run_turn`](Self::run_turn), but with caller-supplied
    /// trace metadata.
    ///
    /// See [`run_with_options`](Self::run_with_options) for the full
    /// [`RunOptions`] contract. The `turn_options` parameter retains
    /// its existing semantics (tool runtime / strict durability);
    /// `run_options` is layered on top to populate Langfuse trace
    /// metadata on the root `invoke_agent` span.
    pub async fn run_turn_with_options(
        &self,
        thread_id: ThreadId,
        input: AgentInput,
        tool_context: ToolContext<Ctx>,
        cancel_token: CancellationToken,
        turn_options: TurnOptions,
        run_options: RunOptions,
    ) -> crate::types::TurnOutcome
    where
        Ctx: Clone,
    {
        // See `run_abortable_with_options` for why we explicitly
        // drop `run_options` on non-otel builds.
        #[cfg(not(feature = "otel"))]
        drop(run_options);

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
            turn_options,
            reminder_config: self.reminder_config.clone(),
            #[cfg(feature = "otel")]
            run_options,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store.clone(),
        })
        .await
    }
}

/// High-level convenience API for agents whose tools take no application
/// context (`Ctx = ()`).
///
/// [`ask`](Self::ask) and [`send`](Self::send) collapse the four pieces of
/// ceremony in the low-level [`run`](Self::run) path — constructing a
/// [`ToolContext::new(())`](crate::ToolContext::new), creating a
/// [`CancellationToken`], awaiting the run, and reassembling the assistant
/// text out of the event store — into a single call that returns a `String`.
///
/// Reach for [`run`](Self::run) or [`run_turn`](Self::run_turn) when you need
/// application context, cancellation, confirmation flow, or access to the raw
/// [`AgentRunState`].
impl<P, H, M, S> AgentLoop<(), P, H, M, S>
where
    P: LlmProvider + 'static,
    H: AgentHooks + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    /// Ask the agent a question and return its assembled reply.
    ///
    /// This is the 30-second on-ramp: it builds a fresh
    /// [`ToolContext::new(())`](crate::ToolContext::new) and a
    /// [`CancellationToken`] internally, runs the agent to completion, and
    /// returns the assistant text emitted during this call concatenated into
    /// one `String`.
    ///
    /// For confirmation flows, application context, or explicit cancellation,
    /// use [`run`](Self::run) directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the run task is dropped before reporting a state,
    /// if the run ends in [`AgentRunState::Error`], or if the event store
    /// cannot be read back.
    pub async fn ask(
        &self,
        thread_id: ThreadId,
        text: impl Into<String>,
    ) -> anyhow::Result<String> {
        self.send(thread_id, AgentInput::Text(text.into())).await
    }

    /// Send an [`AgentInput`] to the agent and return its assembled reply.
    ///
    /// Like [`ask`](Self::ask) but accepts a full [`AgentInput`] (e.g. to
    /// resume after confirmation). Builds the
    /// [`ToolContext`](crate::ToolContext) and [`CancellationToken`]
    /// internally and returns the assistant text emitted during this call.
    ///
    /// # Errors
    ///
    /// Returns an error if the run task is dropped before reporting a state,
    /// if the run ends in [`AgentRunState::Error`], or if the event store
    /// cannot be read back.
    pub async fn send(&self, thread_id: ThreadId, input: AgentInput) -> anyhow::Result<String> {
        use crate::events::AgentEvent;

        // Snapshot the existing event count so we only assemble text emitted
        // by this call, not earlier turns persisted on the same thread.
        // `event_count` + `get_events_since` avoid materializing (and cloning)
        // the entire thread history twice per call — repeated sends on a
        // long-lived thread would otherwise be O(n^2) cumulative.
        let baseline = self.event_store.event_count(&thread_id).await?;

        let state = self
            .run(
                thread_id.clone(),
                input,
                ToolContext::new(()),
                CancellationToken::new(),
            )
            .await?;

        if let AgentRunState::Error(error) = state {
            return Err(anyhow::Error::new(error));
        }

        let events = self
            .event_store
            .get_events_since(&thread_id, baseline)
            .await?;
        let reply = events
            .into_iter()
            .filter_map(|envelope| match envelope.event {
                AgentEvent::Text { text, .. } => Some(text),
                _ => None,
            })
            .collect::<String>();

        Ok(reply)
    }
}
