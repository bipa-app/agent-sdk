//! Agent lifecycle hooks for customization.
//!
//! Hooks allow you to intercept and customize agent behavior at key points:
//!
//! - [`AgentHooks::pre_tool_use`] - Control tool execution permissions
//! - [`AgentHooks::post_tool_use`] - React to tool completion
//! - [`AgentHooks::on_event`] - Log or process events
//! - [`AgentHooks::on_error`] - Handle errors and decide recovery
//!
//! # Built-in Implementations
//!
//! - [`DefaultHooks`] - Tier-based permissions (default)
//! - [`AllowAllHooks`] - Allow all tools without confirmation
//! - [`LoggingHooks`] - Debug logging for all events

use crate::events::AgentEvent;
use crate::llm;
use crate::types::{ToolResult, ToolTier};
use async_trait::async_trait;
use serde_json::Value;

/// Decision returned by pre-tool hooks
#[derive(Debug, Clone)]
pub enum ToolDecision {
    /// Allow the tool to execute
    Allow,
    /// Block the tool execution with a message
    Block(String),
    /// Tool requires user confirmation
    RequiresConfirmation(String),
    /// Tool requires PIN verification
    RequiresPin(String),
}

/// Lifecycle hooks for the agent loop.
/// Implement this trait to customize agent behavior.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before a tool is executed.
    /// Return `ToolDecision::Allow` to proceed, or block/require confirmation.
    async fn pre_tool_use(&self, tool_name: &str, input: &Value, tier: ToolTier) -> ToolDecision {
        // Default: allow Observe tier, require confirmation for others
        // input is available for implementors but not used in default
        let _ = input;
        match tier {
            ToolTier::Observe => ToolDecision::Allow,
            ToolTier::Confirm => {
                ToolDecision::RequiresConfirmation(format!("Confirm {tool_name}?"))
            }
            ToolTier::RequiresPin => {
                ToolDecision::RequiresPin(format!("{tool_name} requires PIN verification"))
            }
        }
    }

    /// Called after a tool completes execution.
    async fn post_tool_use(&self, _tool_name: &str, _result: &ToolResult) {
        // Default: no-op
    }

    /// Called when the agent emits an event.
    /// Can be used for logging, metrics, or custom handling.
    async fn on_event(&self, _event: &AgentEvent) {
        // Default: no-op
    }

    /// Called when an error occurs.
    /// Return true to attempt recovery, false to abort.
    async fn on_error(&self, _error: &anyhow::Error) -> bool {
        // Default: don't recover
        false
    }

    /// Called when context is about to be compacted due to length.
    /// Return a summary to use, or None to use default summarization.
    async fn on_context_compact(&self, _messages: &[llm::Message]) -> Option<String> {
        // Default: use built-in summarization
        None
    }
}

/// Default hooks implementation that uses tier-based decisions
pub struct DefaultHooks;

#[async_trait]
impl AgentHooks for DefaultHooks {}

/// Hooks that allow all tools without confirmation
pub struct AllowAllHooks;

#[async_trait]
impl AgentHooks for AllowAllHooks {
    async fn pre_tool_use(
        &self,
        _tool_name: &str,
        _input: &Value,
        _tier: ToolTier,
    ) -> ToolDecision {
        ToolDecision::Allow
    }
}

/// Hooks that log all events (useful for debugging)
pub struct LoggingHooks;

#[async_trait]
impl AgentHooks for LoggingHooks {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value, tier: ToolTier) -> ToolDecision {
        tracing::debug!(tool = tool_name, ?input, ?tier, "Pre-tool use");
        DefaultHooks.pre_tool_use(tool_name, input, tier).await
    }

    async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
        tracing::debug!(
            tool = tool_name,
            success = result.success,
            duration_ms = result.duration_ms,
            "Post-tool use"
        );
    }

    async fn on_event(&self, event: &AgentEvent) {
        tracing::debug!(?event, "Agent event");
    }

    async fn on_error(&self, error: &anyhow::Error) -> bool {
        tracing::error!(?error, "Agent error");
        false
    }
}
