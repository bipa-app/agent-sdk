//! Host-side runtime dependencies for executing durable tasks.
//!
//! The service host owns worker orchestration and durable state, but
//! it does not own concrete LLM-provider or tool implementations.
//! This module defines the narrow runtime traits the host needs so
//! tests, local daemons, and future production binaries can supply
//! execution behavior without coupling the host to a specific stack.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use agent_sdk_core::{PendingToolCallInfo, ToolResult};
use agent_sdk_providers::LlmProvider;
use agent_server::worker::{
    AgentDefinition, ConfirmationPolicy, PolicyVerdict, ToolEventCollector, ToolTaskBootstrap,
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
        }
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
