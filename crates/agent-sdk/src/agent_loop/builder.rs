use crate::authority::EventAuthority;
use crate::context::{CompactionConfig, ContextCompactor};
use crate::hooks::{AgentHooks, DefaultHooks};
use crate::llm::LlmProvider;
#[cfg(feature = "skills")]
use crate::skills::Skill;
use crate::stores::{EventStore, InMemoryStore, MessageStore, StateStore, ToolExecutionStore};
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
    // `provider` / `hooks` / `message_store` / `state_store` are stored as the
    // bare generic type, not `Option<_>`: the type-transitioning setters
    // ([`provider`](Self::provider), [`hooks`](Self::hooks),
    // [`message_store`](Self::message_store), [`state_store`](Self::state_store))
    // move the value in and flip the corresponding type parameter from the
    // unset `()` to the concrete type. The build methods are only reachable
    // once those parameters satisfy their trait bounds, so the values are
    // always present — there is no runtime "not set" state to guard against.
    provider: P,
    tools: Option<ToolRegistry<Ctx>>,
    hooks: H,
    message_store: M,
    state_store: S,
    event_store: Option<Arc<dyn EventStore>>,
    event_authority: Option<Arc<dyn EventAuthority>>,
    config: Option<AgentConfig>,
    compaction_config: Option<CompactionConfig>,
    compactor: Option<Arc<dyn ContextCompactor>>,
    execution_store: Option<Arc<dyn ToolExecutionStore>>,
    audit_sink: Option<Arc<dyn crate::hooks::ToolAuditSink>>,
    reminder_config: Option<crate::reminders::ReminderConfig>,
    #[cfg(feature = "otel")]
    observability_store: Option<Arc<dyn crate::observability::ObservabilityStore>>,
}

impl<Ctx> AgentLoopBuilder<Ctx, (), (), (), ()> {
    /// Create a new builder with no components set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: (),
            tools: None,
            hooks: (),
            message_store: (),
            state_store: (),
            event_store: None,
            event_authority: None,
            config: None,
            compaction_config: None,
            compactor: None,
            execution_store: None,
            audit_sink: None,
            reminder_config: None,
            #[cfg(feature = "otel")]
            observability_store: None,
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
            provider,
            tools: self.tools,
            hooks: self.hooks,
            message_store: self.message_store,
            state_store: self.state_store,
            event_store: self.event_store,
            event_authority: self.event_authority,
            config: self.config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self.audit_sink,
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
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
            hooks,
            message_store: self.message_store,
            state_store: self.state_store,
            event_store: self.event_store,
            event_authority: self.event_authority,
            config: self.config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self.audit_sink,
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
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
            message_store,
            state_store: self.state_store,
            event_store: self.event_store,
            event_authority: self.event_authority,
            config: self.config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self.audit_sink,
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
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
            state_store,
            event_store: self.event_store,
            event_authority: self.event_authority,
            config: self.config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self.audit_sink,
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
        }
    }

    /// Set the authoritative event store for the loop lifecycle.
    #[must_use]
    pub fn event_store(mut self, store: Arc<dyn EventStore>) -> Self {
        self.event_store = Some(store);
        self
    }

    /// Set the event authority for envelope creation.
    ///
    /// When set, the authority governs how events are wrapped in envelopes
    /// (sequence numbers, event IDs, timestamps).  In server mode the
    /// authority seeds sequences from durable storage so ordering is
    /// continuous across turns within a thread.
    ///
    /// When not set, a fresh [`LocalEventAuthority`](crate::authority::LocalEventAuthority)
    /// starting at sequence 0 is created for each run.
    #[must_use]
    pub fn event_authority(mut self, authority: Arc<dyn EventAuthority>) -> Self {
        self.event_authority = Some(authority);
        self
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

    /// Set the execution store from a shared `Arc`.
    ///
    /// Use this when the caller needs to retain a handle to the store
    /// (for inspection, pre-population, or sharing across loops). See
    /// [`Self::execution_store`] for the standard owned form.
    #[must_use]
    pub fn execution_store_shared(mut self, store: Arc<dyn ToolExecutionStore>) -> Self {
        self.execution_store = Some(store);
        self
    }

    /// Set the authoritative tool audit sink.
    ///
    /// When set, the agent loop emits a
    /// [`ToolAuditRecord`](crate::advanced::ToolAuditRecord) at every tool
    /// lifecycle transition — blocked, requires-confirmation, cached,
    /// replayed, invalidated, completed, and persistence-failed. This
    /// gives servers a complete audit trail without relying on the weaker
    /// `post_tool_use` hook.
    ///
    /// Defaults to [`NoopAuditSink`](crate::hooks::NoopAuditSink) when
    /// not set.
    #[must_use]
    pub fn audit_sink(mut self, sink: impl crate::hooks::ToolAuditSink + 'static) -> Self {
        self.audit_sink = Some(Arc::new(sink));
        self
    }

    /// Set the audit sink from a shared `Arc`.
    ///
    /// Use this when the caller needs to retain a handle to the sink
    /// (e.g. to inspect captured records from tests, or to share a
    /// single durable sink across multiple agent loops). Passing an
    /// `Arc<dyn ToolAuditSink>` here avoids the `Arc<Arc<S>>` double
    /// wrap that happens when callers `Arc::clone(&sink)` a sink they
    /// already wrapped and hand it to [`Self::audit_sink`].
    ///
    /// See [`Self::audit_sink`] for the standard owned form.
    #[must_use]
    pub fn audit_sink_shared(mut self, sink: Arc<dyn crate::hooks::ToolAuditSink>) -> Self {
        self.audit_sink = Some(sink);
        self
    }

    /// Set the observability store for `GenAI` payload capture.
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn observability_store(
        mut self,
        store: impl crate::observability::ObservabilityStore + 'static,
    ) -> Self {
        self.observability_store = Some(Arc::new(store));
        self
    }

    /// Set the agent configuration.
    #[must_use]
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable per-tool system reminders.
    ///
    /// When set, the agent loop evaluates the configuration's
    /// [`tool_reminders`](crate::reminders::ReminderConfig::tool_reminders)
    /// after each tool executes and appends any reminder whose
    /// [`ReminderTrigger`](crate::reminders::ReminderTrigger) matches
    /// (wrapped in `<system-reminder>` tags) to the tool result the model
    /// reads on the next turn.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use agent_sdk::{builder, reminders::{ReminderConfig, ToolReminder}};
    ///
    /// let reminders = ReminderConfig::new()
    ///     .with_tool_reminder("write", ToolReminder::always(
    ///         "Consider reading the file back to verify the content.",
    ///     ));
    /// let agent = builder()
    ///     .provider(my_provider)
    ///     .with_reminders(reminders)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_reminders(mut self, reminders: crate::reminders::ReminderConfig) -> Self {
        self.reminder_config = Some(reminders);
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

    /// Override the default compactor with a custom implementation.
    ///
    /// **Guardrail and pricing boundary:** the loop only wires its
    /// `pre_llm_request` / `on_llm_response` guardrail hooks into the
    /// compactor it constructs itself — it cannot reach inside an arbitrary
    /// [`ContextCompactor`]'s own LLM calls. A custom compactor that talks
    /// to a model is responsible for its own guardrail coverage (for the
    /// built-in summarizer, construct it with
    /// `LlmContextCompactor::with_guardrail_hooks`). Its reported usage is
    /// priced at the *run's* provider/model for cost budgets, so
    /// `max_cost_usd` is approximate when the compactor uses a different
    /// backend.
    #[must_use]
    pub fn with_custom_compactor(mut self, compactor: impl ContextCompactor + 'static) -> Self {
        self.compactor = Some(Arc::new(compactor));
        self
    }

    /// Apply a skill configuration.
    ///
    /// This merges the skill's system prompt with the existing configuration
    /// and filters tools based on the skill's allowed/denied lists.
    ///
    /// Available when the `skills` feature is enabled.
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
    #[cfg(feature = "skills")]
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
    /// Build the agent loop with default hooks and in-memory message/state stores.
    ///
    /// This is a convenience method that uses:
    /// - `DefaultHooks` for hooks
    /// - `InMemoryStore` for message store
    /// - `InMemoryStore` for state store
    /// - `InMemoryEventStore` for the event store, when none was set
    /// - `AgentConfig::default()` if no config is set
    ///
    /// Supplying an [`event_store`](Self::event_store) is optional for this
    /// convenience build — a fresh [`InMemoryEventStore`](crate::InMemoryEventStore)
    /// is used by default so the 30-second path needs no `Arc` ceremony. Wire
    /// a durable store explicitly when you need persistence across process
    /// restarts.
    #[must_use]
    pub fn build(self) -> AgentLoop<Ctx, P, DefaultHooks, InMemoryStore, InMemoryStore> {
        // `self.provider` is the bare `P` moved in by `provider()`. This
        // method is only reachable once `P: LlmProvider`, so the provider is
        // always present — no runtime "unset" state to guard.
        let event_store = self
            .event_store
            .unwrap_or_else(|| Arc::new(crate::stores::InMemoryEventStore::new()));
        let tools = self.tools.unwrap_or_default();
        let config = self.config.unwrap_or_default();

        AgentLoop {
            provider: Arc::new(self.provider),
            tools: Arc::new(tools),
            hooks: Arc::new(DefaultHooks),
            message_store: Arc::new(InMemoryStore::new()),
            state_store: Arc::new(InMemoryStore::new()),
            event_store,
            event_authority: self.event_authority,
            config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self
                .audit_sink
                .unwrap_or_else(|| Arc::new(crate::hooks::NoopAuditSink)),
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
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
    /// `provider`, `hooks`, `message_store`, and `state_store` are guaranteed
    /// present by the type-state builder (this method is only callable once
    /// each is set to a concrete type), so they cannot be missing at runtime.
    ///
    /// # Panics
    ///
    /// Panics if an [`event_store`](Self::event_store) has not been set — it is
    /// the one component supplied via a plain `Arc` setter rather than a
    /// type-transitioning one, so it has no compile-time "set" guarantee.
    #[must_use]
    pub fn build_with_stores(self) -> AgentLoop<Ctx, P, H, M, S> {
        let tools = self.tools.unwrap_or_default();
        let Some(event_store) = self.event_store else {
            panic!("event_store is required when using build_with_stores");
        };
        let config = self.config.unwrap_or_default();

        AgentLoop {
            provider: Arc::new(self.provider),
            tools: Arc::new(tools),
            hooks: Arc::new(self.hooks),
            message_store: Arc::new(self.message_store),
            state_store: Arc::new(self.state_store),
            event_store,
            event_authority: self.event_authority,
            config,
            compaction_config: self.compaction_config,
            compactor: self.compactor,
            execution_store: self.execution_store,
            audit_sink: self
                .audit_sink
                .unwrap_or_else(|| Arc::new(crate::hooks::NoopAuditSink)),
            reminder_config: self.reminder_config,
            #[cfg(feature = "otel")]
            observability_store: self.observability_store,
        }
    }
}
