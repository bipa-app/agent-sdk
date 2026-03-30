//! Subagent support for spawning child agents.
//!
//! This module provides the ability to spawn subagents from within an agent.
//! Subagents are isolated agent instances that run to completion and return
//! only their final response to the parent agent.
//!
//! # Overview
//!
//! Subagents are useful for:
//! - Delegating complex subtasks to specialized agents
//! - Running parallel investigations
//! - Isolating context for specific operations
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::subagent::{SubagentTool, SubagentConfig};
//!
//! let config = SubagentConfig::new("researcher")
//!     .with_system_prompt("You are a research specialist...")
//!     .with_max_turns(10);
//!
//! let tool = SubagentTool::new(config, provider, tools);
//! registry.register(tool);
//! ```
//!
//! # Behavior
//!
//! When a subagent runs:
//! 1. A new isolated thread is created
//! 2. The subagent runs until completion or max turns
//! 3. Only the final text response is returned to the parent
//! 4. The parent does not see the subagent's intermediate tool calls

mod factory;

pub use factory::SubagentFactory;

use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::{AgentHooks, DefaultHooks};
use crate::llm::LlmProvider;
use crate::stores::{InMemoryStore, MessageStore, StateStore};
use crate::tools::{DynamicToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, ThreadId, TokenUsage, ToolResult, ToolTier};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Metadata key for tracking the current subagent nesting depth.
///
/// When a subagent spawns another subagent, the depth is incremented.
/// Tools check this value against the configured maximum depth.
pub const METADATA_SUBAGENT_DEPTH: &str = "subagent_depth";

/// Metadata key for the maximum allowed subagent nesting depth.
///
/// Set by the host application (e.g. bip) to prevent unbounded recursion.
pub const METADATA_MAX_SUBAGENT_DEPTH: &str = "max_subagent_depth";

/// Configuration for a subagent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Name of the subagent (for identification).
    pub name: String,
    /// System prompt for the subagent.
    pub system_prompt: String,
    /// Maximum number of turns before stopping.
    pub max_turns: Option<usize>,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Optional model override for this subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SubagentConfig {
    /// Create a new subagent configuration.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: String::new(),
            max_turns: None,
            timeout_ms: None,
            model: None,
        }
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the maximum number of turns.
    #[must_use]
    pub const fn with_max_turns(mut self, max: usize) -> Self {
        self.max_turns = Some(max);
        self
    }

    /// Set the timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = Some(timeout);
        self
    }

    /// Set the model override for this subagent.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Log entry for a single tool call within a subagent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallLog {
    /// Tool name.
    pub name: String,
    /// Tool display name.
    pub display_name: String,
    /// Brief context/args (e.g., file path, command).
    pub context: String,
    /// Brief result summary.
    pub result: String,
    /// Whether the tool call succeeded.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: Option<u64>,
}

/// Result from a subagent execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentResult {
    /// Name of the subagent.
    pub name: String,
    /// The final text response (only visible part to parent).
    pub final_response: String,
    /// Total number of turns taken.
    pub total_turns: usize,
    /// Number of tool calls made by the subagent.
    pub tool_count: u32,
    /// Log of tool calls made by the subagent.
    pub tool_logs: Vec<ToolCallLog>,
    /// Token usage statistics.
    pub usage: TokenUsage,
    /// Whether the subagent completed successfully.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Detailed error information when `success` is false.
    ///
    /// Contains the raw error message from the agent event, which may include
    /// stack trace information or structured error context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_details: Option<String>,
    /// Name of the tool that caused the failure (if applicable).
    ///
    /// Populated when the subagent encountered an error during a specific
    /// tool execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_tool: Option<String>,
}

/// Tool for spawning subagents.
///
/// This tool allows an agent to spawn a child agent that runs independently
/// and returns only its final response.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::subagent::{SubagentTool, SubagentConfig};
///
/// let config = SubagentConfig::new("analyzer")
///     .with_system_prompt("You analyze code...");
///
/// let tool = SubagentTool::new(config, provider.clone(), tools.clone());
/// ```
pub struct SubagentTool<P, H = DefaultHooks, M = InMemoryStore, S = InMemoryStore>
where
    P: LlmProvider,
    H: AgentHooks,
    M: MessageStore,
    S: StateStore,
{
    config: SubagentConfig,
    provider: Arc<P>,
    tools: Arc<ToolRegistry<()>>,
    hooks: Arc<H>,
    message_store_factory: Arc<dyn Fn() -> M + Send + Sync>,
    state_store_factory: Arc<dyn Fn() -> S + Send + Sync>,
    /// Cached display name to avoid `Box::leak` on every call.
    cached_display_name: &'static str,
    /// Cached description to avoid `Box::leak` on every call.
    cached_description: &'static str,
}

impl<P> SubagentTool<P, DefaultHooks, InMemoryStore, InMemoryStore>
where
    P: LlmProvider + 'static,
{
    /// Create a new subagent tool with default hooks and in-memory stores.
    #[must_use]
    pub fn new(config: SubagentConfig, provider: Arc<P>, tools: Arc<ToolRegistry<()>>) -> Self {
        // Cache leaked strings at construction time (bounded by number of tools)
        let cached_display_name = Box::leak(format!("Subagent: {}", config.name).into_boxed_str());
        let cached_description = Box::leak(
            format!(
                "Spawn a subagent named '{}' to handle a task. The subagent will work independently and return only its final response.",
                config.name
            )
            .into_boxed_str(),
        );
        Self {
            config,
            provider,
            tools,
            hooks: Arc::new(DefaultHooks),
            message_store_factory: Arc::new(InMemoryStore::new),
            state_store_factory: Arc::new(InMemoryStore::new),
            cached_display_name,
            cached_description,
        }
    }
}

impl<P, H, M, S> SubagentTool<P, H, M, S>
where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    /// Create with custom hooks.
    #[must_use]
    pub fn with_hooks<H2: AgentHooks + Clone + 'static>(
        self,
        hooks: Arc<H2>,
    ) -> SubagentTool<P, H2, M, S> {
        SubagentTool {
            config: self.config,
            provider: self.provider,
            tools: self.tools,
            hooks,
            message_store_factory: self.message_store_factory,
            state_store_factory: self.state_store_factory,
            cached_display_name: self.cached_display_name,
            cached_description: self.cached_description,
        }
    }

    /// Create with custom store factories.
    #[must_use]
    pub fn with_stores<M2, S2, MF, SF>(
        self,
        message_factory: MF,
        state_factory: SF,
    ) -> SubagentTool<P, H, M2, S2>
    where
        M2: MessageStore + 'static,
        S2: StateStore + 'static,
        MF: Fn() -> M2 + Send + Sync + 'static,
        SF: Fn() -> S2 + Send + Sync + 'static,
    {
        SubagentTool {
            config: self.config,
            provider: self.provider,
            tools: self.tools,
            hooks: self.hooks,
            message_store_factory: Arc::new(message_factory),
            state_store_factory: Arc::new(state_factory),
            cached_display_name: self.cached_display_name,
            cached_description: self.cached_description,
        }
    }

    /// Get the subagent configuration.
    #[must_use]
    pub const fn config(&self) -> &SubagentConfig {
        &self.config
    }

    /// Run the subagent with a task.
    ///
    /// If `parent_tx` is provided, the subagent will emit `SubagentProgress` events
    /// to the parent's event channel, allowing the UI to show live progress.
    ///
    /// The `parent_cancel` token links the subagent's lifecycle to its parent.
    /// Cancelling the parent token will also cancel the subagent.
    #[allow(clippy::too_many_lines)]
    async fn run_subagent(
        &self,
        task: &str,
        subagent_id: String,
        parent_tx: Option<mpsc::Sender<AgentEventEnvelope>>,
        parent_seq: Option<SequenceCounter>,
        parent_cancel: CancellationToken,
    ) -> Result<SubagentResult> {
        use crate::agent_loop::AgentLoop;

        let start = Instant::now();
        let thread_id = ThreadId::new();

        // Create stores for this subagent run
        let message_store = (self.message_store_factory)();
        let state_store = (self.state_store_factory)();

        // Create agent config with a default max_turns to prevent unbounded execution
        let agent_config = AgentConfig {
            max_turns: Some(self.config.max_turns.unwrap_or(100)),
            system_prompt: self.config.system_prompt.clone(),
            ..Default::default()
        };

        // Build the subagent
        let agent = AgentLoop::new(
            (*self.provider).clone(),
            (*self.tools).clone(),
            (*self.hooks).clone(),
            message_store,
            state_store,
            agent_config,
        );

        // Create tool context
        let tool_ctx = ToolContext::new(());

        // Run with a child cancellation token so parent cancellation propagates
        let cancel_token = parent_cancel.child_token();
        let timeout_cancel = cancel_token.clone();
        let (mut rx, _final_state) = agent.run(
            thread_id,
            AgentInput::Text(task.to_string()),
            tool_ctx,
            cancel_token,
        );

        let mut final_response = String::new();
        let mut total_turns = 0;
        let mut tool_count = 0u32;
        let mut tool_logs: Vec<ToolCallLog> = Vec::new();
        let mut pending_tools: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        let mut total_usage = TokenUsage::default();
        let mut success = true;
        let mut error_details: Option<String> = None;
        let mut failed_tool: Option<String> = None;

        let timeout_duration = self.config.timeout_ms.map(Duration::from_millis);

        loop {
            let recv_result = if let Some(timeout) = timeout_duration {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    timeout_cancel.cancel(); // Cancel the child agent on timeout
                    final_response = "Subagent timed out".to_string();
                    error_details = Some(format!(
                        "Subagent '{}' timed out after {}ms",
                        self.config.name,
                        self.config.timeout_ms.unwrap_or(0)
                    ));
                    success = false;
                    break;
                }
                tokio::time::timeout(remaining, rx.recv()).await
            } else {
                Ok(rx.recv().await)
            };

            match recv_result {
                Ok(Some(envelope)) => match envelope.event {
                    AgentEvent::Text {
                        message_id: _,
                        text,
                    } => {
                        final_response.push_str(&text);
                    }
                    AgentEvent::ToolCallStart {
                        id, name, input, ..
                    } => {
                        // Track tool calls made by the subagent
                        tool_count += 1;
                        let context = extract_tool_context(&name, &input);
                        pending_tools.insert(id, (name.clone(), context.clone()));

                        // Emit progress event to parent
                        if let (Some(tx), Some(seq)) = (&parent_tx, &parent_seq) {
                            let event = AgentEvent::SubagentProgress {
                                subagent_id: subagent_id.clone(),
                                subagent_name: self.config.name.clone(),
                                tool_name: name,
                                tool_context: context,
                                completed: false,
                                success: false,
                                tool_count,
                                total_tokens: u64::from(total_usage.input_tokens)
                                    + u64::from(total_usage.output_tokens),
                            };
                            let _ = tx.send(AgentEventEnvelope::wrap(event, seq)).await;
                        }
                    }
                    AgentEvent::ToolCallEnd {
                        id,
                        name,
                        display_name,
                        result,
                    } => {
                        // Create log entry when tool completes
                        let context = pending_tools
                            .remove(&id)
                            .map(|(_, ctx)| ctx)
                            .unwrap_or_default();
                        let result_summary = summarize_tool_result(&name, &result);
                        let tool_success = result.success;
                        tool_logs.push(ToolCallLog {
                            name: name.clone(),
                            display_name: display_name.clone(),
                            context: context.clone(),
                            result: result_summary,
                            success: tool_success,
                            duration_ms: result.duration_ms,
                        });

                        // Emit progress event to parent
                        if let (Some(tx), Some(seq)) = (&parent_tx, &parent_seq) {
                            let event = AgentEvent::SubagentProgress {
                                subagent_id: subagent_id.clone(),
                                subagent_name: self.config.name.clone(),
                                tool_name: name,
                                tool_context: context,
                                completed: true,
                                success: tool_success,
                                tool_count,
                                total_tokens: u64::from(total_usage.input_tokens)
                                    + u64::from(total_usage.output_tokens),
                            };
                            let _ = tx.send(AgentEventEnvelope::wrap(event, seq)).await;
                        }
                    }
                    AgentEvent::TurnComplete { turn, usage, .. } => {
                        total_turns = turn;
                        total_usage.add(&usage);
                    }
                    AgentEvent::Done {
                        total_turns: turns, ..
                    } => {
                        total_turns = turns;
                        break;
                    }
                    AgentEvent::Error { message, .. } => {
                        error_details = Some(message.clone());
                        // If there are pending tool calls, the last one is likely the culprit.
                        if let Some(last_tool) = pending_tools.values().last() {
                            failed_tool = Some(last_tool.0.clone());
                        }
                        final_response = message;
                        success = false;
                        break;
                    }
                    _ => {}
                },
                Ok(None) => break,
                Err(_) => {
                    timeout_cancel.cancel(); // Cancel the child agent on timeout
                    final_response = "Subagent timed out".to_string();
                    error_details = Some(format!(
                        "Subagent '{}' timed out waiting for event",
                        self.config.name,
                    ));
                    success = false;
                    break;
                }
            }
        }

        let result = SubagentResult {
            name: self.config.name.clone(),
            final_response,
            total_turns,
            tool_count,
            tool_logs,
            usage: total_usage,
            success,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            error_details,
            failed_tool,
        };

        #[cfg(feature = "otel")]
        {
            use crate::observability::{attrs, provider_name, spans};
            use opentelemetry::KeyValue;
            use opentelemetry::trace::Span;

            let mut span = spans::start_internal_span(
                "invoke_agent",
                vec![
                    KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "invoke_agent"),
                    KeyValue::new(attrs::GEN_AI_AGENT_NAME, self.config.name.clone()),
                    KeyValue::new(
                        attrs::GEN_AI_PROVIDER_NAME,
                        provider_name::normalize(self.provider.provider()),
                    ),
                    KeyValue::new(
                        attrs::GEN_AI_REQUEST_MODEL,
                        self.provider.model().to_string(),
                    ),
                    KeyValue::new(attrs::SDK_RUN_MODE, "loop"),
                ],
            );
            let outcome = if result.success { "done" } else { "error" };
            span.set_attribute(KeyValue::new(attrs::SDK_OUTCOME, outcome));
            span.set_attribute(attrs::kv_i64(
                attrs::SDK_TOTAL_TURNS,
                i64::try_from(result.total_turns).unwrap_or(0),
            ));
            span.set_attribute(attrs::kv_i64(
                attrs::GEN_AI_USAGE_INPUT_TOKENS,
                i64::from(result.usage.input_tokens),
            ));
            span.set_attribute(attrs::kv_i64(
                attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
                i64::from(result.usage.output_tokens),
            ));
            if outcome == "error" {
                spans::set_span_error(&mut span, "agent_error", "subagent invocation failed");
            }
            span.end();
        }

        Ok(result)
    }
}

/// Extracts context information from tool input for display.
fn extract_tool_context(name: &str, input: &Value) -> String {
    match name {
        "read" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "write" | "edit" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "bash" => {
            let cmd = input.get("command").and_then(Value::as_str).unwrap_or("");
            // Truncate long commands (UTF-8 safe)
            if cmd.len() > 60 {
                format!("{}...", crate::primitive_tools::truncate_str(cmd, 57))
            } else {
                cmd.to_string()
            }
        }
        "glob" | "grep" => input
            .get("pattern")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "web_search" => input
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Summarizes tool result for logging.
fn summarize_tool_result(name: &str, result: &ToolResult) -> String {
    if !result.success {
        let first_line = result.output.lines().next().unwrap_or("Error");
        return if first_line.len() > 50 {
            format!(
                "{}...",
                crate::primitive_tools::truncate_str(first_line, 47)
            )
        } else {
            first_line.to_string()
        };
    }

    match name {
        "read" => {
            let line_count = result.output.lines().count();
            format!("{line_count} lines")
        }
        "write" => "wrote file".to_string(),
        "edit" => "edited".to_string(),
        "bash" => {
            let lines: Vec<&str> = result.output.lines().collect();
            if lines.is_empty() {
                "done".to_string()
            } else if lines.len() == 1 {
                let line = lines[0];
                if line.len() > 50 {
                    format!("{}...", crate::primitive_tools::truncate_str(line, 47))
                } else {
                    line.to_string()
                }
            } else {
                format!("{} lines", lines.len())
            }
        }
        "glob" => {
            let count = result.output.lines().count();
            format!("{count} files")
        }
        "grep" => {
            let count = result.output.lines().count();
            format!("{count} matches")
        }
        _ => {
            let line_count = result.output.lines().count();
            if line_count == 0 {
                "done".to_string()
            } else {
                format!("{line_count} lines")
            }
        }
    }
}

impl<P, H, M, S> Tool<()> for SubagentTool<P, H, M, S>
where
    P: LlmProvider + Clone + 'static,
    H: AgentHooks + Clone + 'static,
    M: MessageStore + 'static,
    S: StateStore + 'static,
{
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(format!("subagent_{}", self.config.name))
    }

    fn display_name(&self) -> &'static str {
        self.cached_display_name
    }

    fn description(&self) -> &'static str {
        self.cached_description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or question for the subagent to handle"
                }
            },
            "required": ["task"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Subagent spawning requires confirmation
        ToolTier::Confirm
    }

    async fn execute(&self, ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let task = input
            .get("task")
            .and_then(Value::as_str)
            .context("Missing 'task' parameter")?;

        // ── Depth limit enforcement ───────────────────────────────────
        let current_depth = ctx
            .metadata
            .get(METADATA_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_depth = ctx
            .metadata
            .get(METADATA_MAX_SUBAGENT_DEPTH)
            .and_then(Value::as_u64)
            .unwrap_or(3); // default: 3 levels deep

        if current_depth >= max_depth {
            bail!(
                "Subagent depth limit exceeded ({current_depth}/{max_depth}). \
                 Cannot spawn nested subagent '{}' — maximum nesting depth reached.",
                self.config.name
            );
        }

        // ── Thread limit enforcement (semaphore) ──────────────────────
        let _permit = if let Some(ref sem) = ctx.subagent_semaphore() {
            match sem.clone().try_acquire_owned() {
                Ok(permit) => Some(permit),
                Err(_) => {
                    return Ok(ToolResult {
                        success: false,
                        output: format!(
                            "Cannot spawn subagent '{}': maximum concurrent subagent limit reached. \
                             Try again when another subagent completes.",
                            self.config.name
                        ),
                        data: None,
                        documents: Vec::new(),
                        duration_ms: Some(0),
                    });
                }
            }
        } else {
            None
        };

        // Get event channel and sequence counter from context for progress updates
        let parent_tx = ctx.event_tx();
        let parent_seq = ctx.event_seq();

        // Generate a unique ID for this subagent execution
        let subagent_id = format!(
            "{}_{:x}",
            self.config.name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        // Use the context's cancellation token if available, otherwise create a standalone one.
        // This ensures that when a parent agent is cancelled, subagents are also cancelled.
        let cancel_token = ctx.cancel_token().unwrap_or_default();

        let result = self
            .run_subagent(task, subagent_id, parent_tx, parent_seq, cancel_token)
            .await?;

        Ok(ToolResult {
            success: result.success,
            output: result.final_response.clone(),
            data: Some(serde_json::to_value(&result).unwrap_or_default()),
            documents: Vec::new(),
            duration_ms: Some(result.duration_ms),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_config_builder() {
        let config = SubagentConfig::new("test")
            .with_system_prompt("Test prompt")
            .with_max_turns(5)
            .with_timeout_ms(30000);

        assert_eq!(config.name, "test");
        assert_eq!(config.system_prompt, "Test prompt");
        assert_eq!(config.max_turns, Some(5));
        assert_eq!(config.timeout_ms, Some(30000));
    }

    #[test]
    fn test_subagent_config_defaults() {
        let config = SubagentConfig::new("default");

        assert_eq!(config.name, "default");
        assert!(config.system_prompt.is_empty());
        assert_eq!(config.max_turns, None);
        assert_eq!(config.timeout_ms, None);
    }

    #[test]
    fn test_subagent_result_serialization() {
        let result = SubagentResult {
            name: "test".to_string(),
            final_response: "Done".to_string(),
            total_turns: 3,
            tool_count: 5,
            tool_logs: vec![
                ToolCallLog {
                    name: "read".to_string(),
                    display_name: "Read file".to_string(),
                    context: "/tmp/test.rs".to_string(),
                    result: "50 lines".to_string(),
                    success: true,
                    duration_ms: Some(10),
                },
                ToolCallLog {
                    name: "grep".to_string(),
                    display_name: "Grep TODO".to_string(),
                    context: "TODO".to_string(),
                    result: "3 matches".to_string(),
                    success: true,
                    duration_ms: Some(5),
                },
            ],
            usage: TokenUsage::default(),
            success: true,
            duration_ms: 1000,
            error_details: None,
            failed_tool: None,
        };

        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("Done"));
        assert!(json.contains("tool_count"));
        assert!(json.contains("tool_logs"));
        assert!(json.contains("/tmp/test.rs"));
    }

    #[test]
    fn test_subagent_result_field_extraction() {
        // Test that verifies the exact JSON structure expected by bip's tui_session.rs
        let result = SubagentResult {
            name: "explore".to_string(),
            final_response: "Found 3 config files".to_string(),
            total_turns: 2,
            tool_count: 5,
            tool_logs: vec![ToolCallLog {
                name: "glob".to_string(),
                display_name: "Glob config files".to_string(),
                context: "**/*.toml".to_string(),
                result: "3 files".to_string(),
                success: true,
                duration_ms: Some(15),
            }],
            usage: TokenUsage {
                input_tokens: 1500,
                output_tokens: 500,
            },
            success: true,
            duration_ms: 2500,
            error_details: None,
            failed_tool: None,
        };

        let value = serde_json::to_value(&result).expect("serialize to value");

        // Test tool_count extraction (as_u64 should work for u32)
        let tool_count = value.get("tool_count").and_then(Value::as_u64);
        assert_eq!(tool_count, Some(5));

        // Test usage extraction
        let usage = value.get("usage").expect("usage field");
        let input_tokens = usage.get("input_tokens").and_then(Value::as_u64);
        let output_tokens = usage.get("output_tokens").and_then(Value::as_u64);
        assert_eq!(input_tokens, Some(1500));
        assert_eq!(output_tokens, Some(500));

        // Test tool_logs extraction
        let tool_logs = value.get("tool_logs").and_then(Value::as_array);
        assert!(tool_logs.is_some());
        let logs = tool_logs.unwrap();
        assert_eq!(logs.len(), 1);

        let first_log = &logs[0];
        assert_eq!(first_log.get("name").and_then(Value::as_str), Some("glob"));
        assert_eq!(
            first_log.get("context").and_then(Value::as_str),
            Some("**/*.toml")
        );
        assert_eq!(
            first_log.get("result").and_then(Value::as_str),
            Some("3 files")
        );
        assert_eq!(
            first_log.get("success").and_then(Value::as_bool),
            Some(true)
        );
    }
}
