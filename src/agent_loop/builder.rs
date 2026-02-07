use crate::context::CompactionConfig;
use crate::hooks::{AgentHooks, DefaultHooks};
use crate::llm::LlmProvider;
use crate::skills::Skill;
use crate::stores::{InMemoryStore, MessageStore, StateStore, ToolExecutionStore};
use crate::tools::ToolRegistry;
use crate::types::AgentConfig;
use std::sync::Arc;

use super::AgentLoop;

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
