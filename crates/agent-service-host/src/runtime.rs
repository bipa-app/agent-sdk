//! Host-side runtime dependencies for executing durable tasks.
//!
//! The service host owns worker orchestration and durable state, but
//! it does not own concrete LLM-provider or tool implementations.
//! This module defines the narrow runtime traits the host needs so
//! tests, local daemons, and future production binaries can supply
//! execution behavior without coupling the host to a specific stack.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use agent_sdk::context::CompactionConfig;
use agent_sdk_foundation::{PendingToolCallInfo, ToolResult};
use agent_sdk_providers::LlmProvider;
use agent_server::worker::{
    AgentDefinition, ConfirmationPolicy, NoopSubagentSpawnSelector, PolicyVerdict,
    SubagentSpawnSelector, ToolEventCollector, ToolTaskBootstrap,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

/// Resolves a runtime LLM provider for an agent definition.
#[async_trait]
pub trait ProviderResolver: Send + Sync {
    /// Return the provider instance that should execute `definition`.
    async fn resolve_provider(&self, definition: &AgentDefinition) -> Result<Arc<dyn LlmProvider>>;
}

/// Executes one durable tool-runtime child task.
#[async_trait]
pub trait ToolCallExecutor: Send + Sync {
    /// Execute the tool owned by `bootstrap`, emitting any progress
    /// events into `collector`.
    async fn execute_tool_call(
        &self,
        bootstrap: &ToolTaskBootstrap,
        collector: ToolEventCollector,
        cancel: CancellationToken,
    ) -> Result<ToolResult>;
}

/// Concrete host runtime wiring shared by the worker loop and the
/// transport layer.
#[derive(Clone)]
pub struct ExecutionRuntime {
    provider_resolver: Arc<dyn ProviderResolver>,
    tool_executor: Arc<dyn ToolCallExecutor>,
    confirmation_policy: Arc<dyn ConfirmationPolicy>,
    /// Optional per-call routing selector consulted at the tool
    /// boundary. Defaults to [`NoopSubagentSpawnSelector`] when
    /// constructed via [`Self::new`], which keeps the legacy
    /// behaviour (every tool call routes through
    /// `spawn_tool_children`).
    subagent_spawn_selector: Arc<dyn SubagentSpawnSelector>,
    /// Optional auto-compaction policy. When set, the worker runs
    /// the SDK compactor before each LLM call once the staged
    /// history crosses [`CompactionConfig::threshold_tokens`] and,
    /// reactively, when the provider rejects a turn with a
    /// `prompt is too long` error. Defaults to `None` so existing
    /// hosts keep today's "no automatic compaction" behaviour
    /// unchanged.
    compaction_config: Option<CompactionConfig>,
}

impl ExecutionRuntime {
    #[must_use]
    pub fn new(
        provider_resolver: Arc<dyn ProviderResolver>,
        tool_executor: Arc<dyn ToolCallExecutor>,
        confirmation_policy: Arc<dyn ConfirmationPolicy>,
    ) -> Self {
        Self {
            provider_resolver,
            tool_executor,
            confirmation_policy,
            subagent_spawn_selector: Arc::new(NoopSubagentSpawnSelector),
            compaction_config: None,
        }
    }

    /// Replace the active subagent-spawn selector. Hosts that wire a
    /// real selector (e.g. bip's `BipSubagentSpawnSelector`) call
    /// this once at construction and keep the noop default until
    /// then.
    #[must_use]
    pub fn with_subagent_spawn_selector(
        mut self,
        selector: Arc<dyn SubagentSpawnSelector>,
    ) -> Self {
        self.subagent_spawn_selector = selector;
        self
    }

    /// Enable host-driven auto-compaction with the given policy.
    ///
    /// The worker reads this config back via [`Self::compaction_config`]
    /// once per turn:
    ///
    /// * **Pre-call**: if the staged message history exceeds
    ///   [`CompactionConfig::threshold_tokens`], the worker compacts
    ///   the durable projection (via `MessageProjectionStore::replace_history`),
    ///   re-seeds the staged buffer, and proceeds with the compacted
    ///   history.
    /// * **Post-failure**: if the provider returns an
    ///   `InvalidRequest` whose message matches the well-known
    ///   "prompt is too long" patterns (Anthropic's 1M context cap,
    ///   `OpenAI`'s `context_length_exceeded`, Gemini's "input is
    ///   too long", Bedrock's "request too large"), the worker
    ///   compacts and retries up to
    ///   [`CompactionConfig::auto_compact`]'s budget instead of
    ///   failing the turn fatally.
    ///
    /// Without a configured `CompactionConfig`, both paths preserve
    /// today's behaviour (no automatic compaction; `InvalidRequest`
    /// goes fatal).
    #[must_use]
    pub const fn with_compaction(mut self, config: CompactionConfig) -> Self {
        self.compaction_config = Some(config);
        self
    }

    #[must_use]
    pub fn provider_resolver(&self) -> &Arc<dyn ProviderResolver> {
        &self.provider_resolver
    }

    #[must_use]
    pub fn tool_executor(&self) -> &Arc<dyn ToolCallExecutor> {
        &self.tool_executor
    }

    #[must_use]
    pub fn confirmation_policy(&self) -> &Arc<dyn ConfirmationPolicy> {
        &self.confirmation_policy
    }

    #[must_use]
    pub fn subagent_spawn_selector(&self) -> &Arc<dyn SubagentSpawnSelector> {
        &self.subagent_spawn_selector
    }

    #[must_use]
    pub const fn compaction_config(&self) -> Option<&CompactionConfig> {
        self.compaction_config.as_ref()
    }
}

/// Confirmation policy that allows every approved tool to proceed.
pub struct AllowAllConfirmationPolicy;

#[async_trait]
impl ConfirmationPolicy for AllowAllConfirmationPolicy {
    async fn check_policy(&self, _tool_call: &PendingToolCallInfo) -> Result<PolicyVerdict> {
        Ok(PolicyVerdict::Allowed)
    }
}

/// Tool executor placeholder for runtimes that do not expose tools.
pub struct NoopToolExecutor;

#[async_trait]
impl ToolCallExecutor for NoopToolExecutor {
    async fn execute_tool_call(
        &self,
        bootstrap: &ToolTaskBootstrap,
        _collector: ToolEventCollector,
        _cancel: CancellationToken,
    ) -> Result<ToolResult> {
        Err(anyhow!(
            "tool '{}' is not configured on this host runtime",
            bootstrap.tool_call.name
        ))
    }
}

/// Provider resolver that routes by `(provider, model)` and supports
/// an optional fallback provider.
#[derive(Default)]
pub struct StaticProviderResolver {
    providers: RwLock<HashMap<(String, String), Arc<dyn LlmProvider>>>,
    fallback: RwLock<Option<Arc<dyn LlmProvider>>>,
}

impl StaticProviderResolver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// # Errors
    ///
    /// Returns an error if the resolver registry lock is poisoned.
    pub fn register(
        &self,
        provider: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<dyn LlmProvider>,
    ) -> Result<()> {
        self.providers
            .write()
            .ok()
            .context("lock poisoned")?
            .insert((provider.into(), model.into()), runtime);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the fallback-provider lock is poisoned.
    pub fn set_fallback(&self, runtime: Arc<dyn LlmProvider>) -> Result<()> {
        *self.fallback.write().ok().context("lock poisoned")? = Some(runtime);
        Ok(())
    }
}

#[async_trait]
impl ProviderResolver for StaticProviderResolver {
    async fn resolve_provider(&self, definition: &AgentDefinition) -> Result<Arc<dyn LlmProvider>> {
        let provider_key = (definition.provider.clone(), definition.model.clone());
        if let Some(provider) = self
            .providers
            .read()
            .ok()
            .context("lock poisoned")?
            .get(&provider_key)
            .cloned()
        {
            return Ok(provider);
        }

        self.fallback
            .read()
            .ok()
            .context("lock poisoned")?
            .clone()
            .ok_or_else(|| {
                anyhow!(
                    "no runtime provider registered for provider='{}' model='{}'",
                    definition.provider,
                    definition.model,
                )
            })
    }
}
