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

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm;
use agent_sdk_foundation::types::{ToolInvocation, ToolResult, ToolTier};
use async_trait::async_trait;

/// Decision returned by pre-tool hooks
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolDecision {
    /// Allow the tool to execute
    Allow,
    /// Block the tool execution with a message
    Block(String),
    /// Tool requires user confirmation.
    RequiresConfirmation(String),
}

/// Decision returned by [`AgentHooks::pre_llm_request`] — an input guardrail
/// that runs before the outbound [`llm::ChatRequest`] is sent to the provider.
///
/// This is the place for prompt-injection scrubbing, PII gating, or
/// system-prompt policy enforcement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RequestDecision {
    /// Send the request unchanged.
    Proceed,
    /// Send a modified request instead of the original.
    Modify(Box<llm::ChatRequest>),
    /// Refuse to call the model; the string explains why.
    Block(String),
}

/// Decision returned by [`AgentHooks::on_llm_response`] — an output guardrail
/// that runs after the provider responds but before the response is persisted
/// and surfaced.
///
/// This is the place for output moderation or secret-leakage detection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ResponseDecision {
    /// Accept the response as-is.
    Accept,
    /// Reject the response; the string explains why.
    Block(String),
    /// Reject the response and feed the string back to the model so it can
    /// retry on the next turn.
    RetryWithFeedback(String),
}

/// Lifecycle hooks for the agent loop.
/// Implement this trait to customize agent behavior.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before a tool is executed.
    ///
    /// Receives a structured [`ToolInvocation`] that bundles tool identity,
    /// tier, requested input, effective input, and listen-context — everything
    /// a server-side policy engine needs for an allow / block / confirm decision.
    ///
    /// Return [`ToolDecision::Allow`] to proceed, [`ToolDecision::Block`] to
    /// reject, or [`ToolDecision::RequiresConfirmation`] to yield for user
    /// approval.
    async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
        match invocation.tier {
            ToolTier::Observe => ToolDecision::Allow,
            ToolTier::Confirm => {
                ToolDecision::RequiresConfirmation(format!("Confirm {}?", invocation.tool_name))
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

    /// Input guardrail: called with the outbound [`llm::ChatRequest`] before it
    /// is sent to the provider.
    ///
    /// Return [`RequestDecision::Proceed`] to send it unchanged,
    /// [`RequestDecision::Modify`] to substitute a sanitized request, or
    /// [`RequestDecision::Block`] to refuse the call (e.g. prompt-injection or
    /// PII policy). The default proceeds unchanged.
    async fn pre_llm_request(&self, _request: &llm::ChatRequest) -> RequestDecision {
        RequestDecision::Proceed
    }

    /// Output guardrail: called with the provider's [`llm::ChatResponse`] before
    /// it is persisted or surfaced.
    ///
    /// Return [`ResponseDecision::Accept`] to keep it,
    /// [`ResponseDecision::Block`] to reject it, or
    /// [`ResponseDecision::RetryWithFeedback`] to reject it and steer a retry
    /// (e.g. output moderation or secret-leakage detection). The default
    /// accepts.
    async fn on_llm_response(&self, _response: &llm::ChatResponse) -> ResponseDecision {
        ResponseDecision::Accept
    }
}

/// Default hooks implementation that uses tier-based decisions
#[derive(Clone, Copy, Default)]
pub struct DefaultHooks;

#[async_trait]
impl AgentHooks for DefaultHooks {}

/// Hooks that allow all tools without confirmation
#[derive(Clone, Copy, Default)]
pub struct AllowAllHooks;

#[async_trait]
impl AgentHooks for AllowAllHooks {
    async fn pre_tool_use(&self, _invocation: &ToolInvocation) -> ToolDecision {
        ToolDecision::Allow
    }
}

/// Hooks that log all events (useful for debugging)
#[derive(Clone, Copy, Default)]
pub struct LoggingHooks;

#[async_trait]
impl AgentHooks for LoggingHooks {
    async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
        log::debug!(
            "Pre-tool use tool={} input={:?} tier={:?}",
            invocation.tool_name,
            invocation.requested_input,
            invocation.tier,
        );
        DefaultHooks.pre_tool_use(invocation).await
    }

    async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
        log::debug!(
            "Post-tool use tool={tool_name} success={} duration_ms={:?}",
            result.success,
            result.duration_ms
        );
    }

    async fn on_event(&self, event: &AgentEvent) {
        log::debug!("Agent event {event:?}");
    }

    async fn on_error(&self, error: &anyhow::Error) -> bool {
        log::error!("Agent error {error:?}");
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn invocation(tier: ToolTier) -> ToolInvocation {
        ToolInvocation {
            tool_call_id: "call_1".to_string(),
            tool_name: "danger".to_string(),
            display_name: "Danger".to_string(),
            tier,
            requested_input: json!({}),
            effective_input: json!({}),
            listen_context: None,
        }
    }

    #[tokio::test]
    async fn default_hooks_gate_confirm_tier() {
        // A Confirm-tier tool must yield for confirmation under the default
        // policy — side-effecting tools never auto-run.
        let decision = DefaultHooks
            .pre_tool_use(&invocation(ToolTier::Confirm))
            .await;
        assert!(
            matches!(decision, ToolDecision::RequiresConfirmation(_)),
            "Confirm tier must require confirmation, got {decision:?}"
        );
    }

    #[tokio::test]
    async fn default_hooks_auto_allow_observe_tier() {
        let decision = DefaultHooks
            .pre_tool_use(&invocation(ToolTier::Observe))
            .await;
        assert!(
            matches!(decision, ToolDecision::Allow),
            "Observe tier may auto-run, got {decision:?}"
        );
    }

    #[tokio::test]
    async fn default_hooks_llm_guardrails_are_permissive_noops() {
        let request = llm::ChatRequest::new("sys", vec![llm::Message::user("hi")]);
        assert!(matches!(
            DefaultHooks.pre_llm_request(&request).await,
            RequestDecision::Proceed
        ));

        let response = llm::ChatResponse {
            id: "resp_1".to_string(),
            content: Vec::new(),
            model: "test-model".to_string(),
            stop_reason: None,
            usage: llm::Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        };
        assert!(matches!(
            DefaultHooks.on_llm_response(&response).await,
            ResponseDecision::Accept
        ));
    }
}
