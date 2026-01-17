//! Factory for creating subagents with common configurations.
//!
//! This module provides a convenient API for creating subagents with
//! pre-configured tool registries and common patterns.
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::subagent::{SubagentFactory, SubagentConfig};
//! use agent_sdk::ToolRegistry;
//!
//! // Create tool registries for different access levels
//! let mut read_tools = ToolRegistry::new();
//! read_tools.register(glob_tool);
//! read_tools.register(grep_tool);
//! read_tools.register(read_tool);
//!
//! let factory = SubagentFactory::new(provider.clone())
//!     .with_read_only_registry(read_tools);
//!
//! // Create a read-only subagent for exploration
//! let explorer = factory.create_read_only(
//!     SubagentConfig::new("explorer")
//!         .with_system_prompt("You explore codebases...")
//!         .with_max_turns(20)
//! );
//! ```

use std::sync::Arc;

use crate::llm::LlmProvider;
use crate::tools::ToolRegistry;

use super::{SubagentConfig, SubagentTool};

/// Factory for creating subagents with shared resources.
///
/// The factory holds references to shared resources (provider, tool registries)
/// and provides convenient methods to create subagents with common configurations.
pub struct SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    provider: Arc<P>,
    read_only_registry: Option<Arc<ToolRegistry<()>>>,
    full_registry: Option<Arc<ToolRegistry<()>>>,
}

impl<P> SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    /// Creates a new factory with the given LLM provider.
    #[must_use]
    pub const fn new(provider: Arc<P>) -> Self {
        Self {
            provider,
            read_only_registry: None,
            full_registry: None,
        }
    }

    /// Sets the read-only tool registry (typically glob, grep, read tools).
    ///
    /// This registry is used when creating read-only subagents.
    #[must_use]
    pub fn with_read_only_registry(mut self, registry: ToolRegistry<()>) -> Self {
        self.read_only_registry = Some(Arc::new(registry));
        self
    }

    /// Sets the full tool registry for full-access subagents.
    #[must_use]
    pub fn with_full_registry(mut self, registry: ToolRegistry<()>) -> Self {
        self.full_registry = Some(Arc::new(registry));
        self
    }

    /// Creates a read-only subagent with only read/search tools.
    ///
    /// This is useful for exploration, research, and investigation tasks
    /// where the subagent should not modify any files.
    ///
    /// # Panics
    ///
    /// Panics if no read-only registry has been set.
    #[must_use]
    pub fn create_read_only(&self, config: SubagentConfig) -> SubagentTool<P> {
        let registry = self
            .read_only_registry
            .as_ref()
            .expect("read-only registry not set; call with_read_only_registry first");
        SubagentTool::new(config, Arc::clone(&self.provider), Arc::clone(registry))
    }

    /// Creates a subagent with all available tools.
    ///
    /// # Panics
    ///
    /// Panics if no full registry has been set.
    #[must_use]
    pub fn create_full_access(&self, config: SubagentConfig) -> SubagentTool<P> {
        let registry = self
            .full_registry
            .as_ref()
            .expect("full registry not set; call with_full_registry first");
        SubagentTool::new(config, Arc::clone(&self.provider), Arc::clone(registry))
    }

    /// Creates a subagent with a custom tool registry.
    #[must_use]
    pub fn create_with_registry(
        &self,
        config: SubagentConfig,
        registry: Arc<ToolRegistry<()>>,
    ) -> SubagentTool<P> {
        SubagentTool::new(config, Arc::clone(&self.provider), registry)
    }

    /// Returns the provider for manual subagent construction.
    #[must_use]
    pub fn provider(&self) -> Arc<P> {
        Arc::clone(&self.provider)
    }
}

impl<P> Clone for SubagentFactory<P>
where
    P: LlmProvider + 'static,
{
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            read_only_registry: self.read_only_registry.clone(),
            full_registry: self.full_registry.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_config_builder() {
        let config = SubagentConfig::new("test")
            .with_system_prompt("You are a test agent")
            .with_max_turns(5)
            .with_timeout_ms(30000);

        assert_eq!(config.name, "test");
        assert_eq!(config.system_prompt, "You are a test agent");
        assert_eq!(config.max_turns, 5);
        assert_eq!(config.timeout_ms, Some(30000));
    }
}
