use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, DefaultHooks, ToolDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason,
};
use crate::stores::{InMemoryStore, MessageStore, StateStore};
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentState, ThreadId, TokenUsage, ToolResult};
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

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
}

impl<Ctx> AgentLoopBuilder<Ctx, (), (), (), ()> {
    /// Create a new builder with no components set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            provider: None,
            tools: None,
            hooks: None,
            message_store: None,
            state_store: None,
            config: None,
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
        }
    }

    /// Set the agent configuration.
    #[must_use]
    pub fn config(mut self, config: AgentConfig) -> Self {
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
        }
    }
}

/// The main agent loop that orchestrates LLM calls and tool execution.
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
}

/// Create a new builder for constructing an `AgentLoop`.
#[must_use]
pub const fn builder<Ctx>() -> AgentLoopBuilder<Ctx, (), (), (), ()> {
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
        }
    }

    /// Run the agent loop for a single user message.
    /// Returns a channel receiver that yields `AgentEvents`.
    pub fn run(
        &self,
        thread_id: ThreadId,
        user_message: String,
        tool_context: ToolContext<Ctx>,
    ) -> mpsc::Receiver<AgentEvent>
    where
        Ctx: Clone,
    {
        let (tx, rx) = mpsc::channel(100);

        let provider = Arc::clone(&self.provider);
        let tools = Arc::clone(&self.tools);
        let hooks = Arc::clone(&self.hooks);
        let message_store = Arc::clone(&self.message_store);
        let state_store = Arc::clone(&self.state_store);
        let config = self.config.clone();

        tokio::spawn(async move {
            let result = run_loop(
                tx.clone(),
                thread_id,
                user_message,
                tool_context,
                provider,
                tools,
                hooks,
                message_store,
                state_store,
                config,
            )
            .await;

            if let Err(e) = result {
                let _ = tx.send(AgentEvent::error(e.to_string(), false)).await;
            }
        });

        rx
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_loop<Ctx, P, H, M, S>(
    tx: mpsc::Sender<AgentEvent>,
    thread_id: ThreadId,
    user_message: String,
    tool_context: ToolContext<Ctx>,
    provider: Arc<P>,
    tools: Arc<ToolRegistry<Ctx>>,
    hooks: Arc<H>,
    message_store: Arc<M>,
    state_store: Arc<S>,
    config: AgentConfig,
) -> Result<()>
where
    Ctx: Send + Sync + Clone + 'static,
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    let start_time = Instant::now();
    let mut turn = 0;
    let mut total_usage = TokenUsage::default();

    // Load or create state
    let mut state = state_store
        .load(&thread_id)
        .await?
        .unwrap_or_else(|| AgentState::new(thread_id.clone()));

    // Add user message to history
    let user_msg = Message::user(&user_message);
    message_store.append(&thread_id, user_msg).await?;

    // Main agent loop
    loop {
        turn += 1;
        state.turn_count = turn;

        if turn > config.max_turns {
            warn!(turn, max = config.max_turns, "Max turns reached");
            tx.send(AgentEvent::error(
                format!("Maximum turns ({}) reached", config.max_turns),
                true,
            ))
            .await?;
            break;
        }

        // Emit start event
        tx.send(AgentEvent::start(thread_id.clone(), turn)).await?;
        hooks
            .on_event(&AgentEvent::start(thread_id.clone(), turn))
            .await;

        // Get message history
        let messages = message_store.get_history(&thread_id).await?;

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
        };

        // Call LLM
        debug!(turn, "Calling LLM");
        let response = match provider.chat(request).await? {
            ChatOutcome::Success(response) => response,
            ChatOutcome::RateLimited => {
                error!("Rate limited by LLM provider");
                tx.send(AgentEvent::error("Rate limited", true)).await?;
                break;
            }
            ChatOutcome::InvalidRequest(msg) => {
                error!(msg, "Invalid request to LLM");
                tx.send(AgentEvent::error(format!("Invalid request: {msg}"), false))
                    .await?;
                break;
            }
            ChatOutcome::ServerError(msg) => {
                error!(msg, "LLM server error");
                tx.send(AgentEvent::error(format!("Server error: {msg}"), true))
                    .await?;
                break;
            }
        };

        // Track usage
        let turn_usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        };
        total_usage.add(&turn_usage);
        state.total_usage = total_usage.clone();

        // Process response content
        let (text_content, tool_uses) = extract_content(&response);

        // Emit text if present
        if let Some(text) = &text_content {
            tx.send(AgentEvent::text(text.clone())).await?;
            hooks.on_event(&AgentEvent::text(text.clone())).await;
        }

        // If no tool uses, we're done
        if tool_uses.is_empty() {
            info!(turn, "Agent completed (no tool use)");
            break;
        }

        // Store assistant message with tool uses
        let assistant_msg = build_assistant_message(&response);
        message_store.append(&thread_id, assistant_msg).await?;

        // Execute tools
        let mut tool_results = Vec::new();
        for (tool_id, tool_name, tool_input) in &tool_uses {
            let Some(tool) = tools.get(tool_name) else {
                let result = ToolResult::error(format!("Unknown tool: {tool_name}"));
                tool_results.push((tool_id.clone(), result));
                continue;
            };

            let tier = tool.tier();

            // Emit tool call start
            tx.send(AgentEvent::tool_call_start(
                tool_id,
                tool_name,
                tool_input.clone(),
                tier,
            ))
            .await?;

            // Check hooks for permission
            let decision = hooks.pre_tool_use(tool_name, tool_input, tier).await;

            match decision {
                ToolDecision::Allow => {
                    // Execute tool
                    let tool_start = Instant::now();
                    let result = match tool.execute(&tool_context, tool_input.clone()).await {
                        Ok(mut r) => {
                            r.duration_ms = Some(millis_to_u64(tool_start.elapsed().as_millis()));
                            r
                        }
                        Err(e) => ToolResult::error(format!("Tool error: {e}"))
                            .with_duration(millis_to_u64(tool_start.elapsed().as_millis())),
                    };

                    // Post-tool hook
                    hooks.post_tool_use(tool_name, &result).await;

                    // Emit tool call end
                    tx.send(AgentEvent::tool_call_end(
                        tool_id,
                        tool_name,
                        result.clone(),
                    ))
                    .await?;

                    tool_results.push((tool_id.clone(), result));
                }
                ToolDecision::Block(reason) => {
                    let result = ToolResult::error(format!("Blocked: {reason}"));
                    tx.send(AgentEvent::tool_call_end(
                        tool_id,
                        tool_name,
                        result.clone(),
                    ))
                    .await?;
                    tool_results.push((tool_id.clone(), result));
                }
                ToolDecision::RequiresConfirmation(description) => {
                    tx.send(AgentEvent::ToolRequiresConfirmation {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        input: tool_input.clone(),
                        description,
                    })
                    .await?;
                    // For now, treat as blocked - caller should handle confirmation flow
                    let result = ToolResult::error("Awaiting user confirmation");
                    tool_results.push((tool_id.clone(), result));
                }
                ToolDecision::RequiresPin(description) => {
                    tx.send(AgentEvent::ToolRequiresPin {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        input: tool_input.clone(),
                        description,
                    })
                    .await?;
                    // For now, treat as blocked - caller should handle PIN flow
                    let result = ToolResult::error("Awaiting PIN verification");
                    tool_results.push((tool_id.clone(), result));
                }
            }
        }

        // Add tool results to message history
        for (tool_id, result) in &tool_results {
            let tool_result_msg = Message::tool_result(tool_id, &result.output, !result.success);
            message_store.append(&thread_id, tool_result_msg).await?;
        }

        // Emit turn complete
        tx.send(AgentEvent::TurnComplete {
            turn,
            usage: turn_usage,
        })
        .await?;

        // Check stop reason
        if response.stop_reason == Some(StopReason::EndTurn) {
            info!(turn, "Agent completed (end_turn)");
            break;
        }

        // Save state checkpoint
        state_store.save(&state).await?;
    }

    // Final state save
    state_store.save(&state).await?;

    // Emit done
    let duration = start_time.elapsed();
    tx.send(AgentEvent::done(thread_id, turn, total_usage, duration))
        .await?;

    Ok(())
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

fn extract_content(
    response: &ChatResponse,
) -> (Option<String>, Vec<(String, String, serde_json::Value)>) {
    let mut text_parts = Vec::new();
    let mut tool_uses = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_uses.push((id.clone(), name.clone(), input.clone()));
            }
            ContentBlock::ToolResult { .. } => {
                // Shouldn't appear in response, but ignore if it does
            }
        }
    }

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };

    (text, tool_uses)
}

fn build_assistant_message(response: &ChatResponse) -> Message {
    let mut blocks = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                blocks.push(ContentBlock::Text { text: text.clone() });
            }
            ContentBlock::ToolUse { id, name, input } => {
                blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
            ContentBlock::ToolResult { .. } => {}
        }
    }

    Message {
        role: Role::Assistant,
        content: Content::Blocks(blocks),
    }
}
