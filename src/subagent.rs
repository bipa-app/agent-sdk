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

use crate::events::AgentEvent;
use crate::hooks::{AgentHooks, DefaultHooks};
use crate::llm::LlmProvider;
use crate::stores::{InMemoryStore, MessageStore, StateStore};
use crate::tools::{DynamicToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, ThreadId, TokenUsage, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Configuration for a subagent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Name of the subagent (for identification).
    pub name: String,
    /// System prompt for the subagent.
    pub system_prompt: String,
    /// Maximum number of turns before stopping.
    pub max_turns: usize,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl SubagentConfig {
    /// Create a new subagent configuration.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: String::new(),
            max_turns: 10,
            timeout_ms: None,
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
        self.max_turns = max;
        self
    }

    /// Set the timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = Some(timeout);
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
}

impl<P> SubagentTool<P, DefaultHooks, InMemoryStore, InMemoryStore>
where
    P: LlmProvider + 'static,
{
    /// Create a new subagent tool with default hooks and in-memory stores.
    #[must_use]
    pub fn new(config: SubagentConfig, provider: Arc<P>, tools: Arc<ToolRegistry<()>>) -> Self {
        Self {
            config,
            provider,
            tools,
            hooks: Arc::new(DefaultHooks),
            message_store_factory: Arc::new(InMemoryStore::new),
            state_store_factory: Arc::new(InMemoryStore::new),
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
    #[allow(clippy::too_many_lines)]
    async fn run_subagent(
        &self,
        task: &str,
        subagent_id: String,
        parent_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<SubagentResult> {
        use crate::agent_loop::AgentLoop;

        let start = Instant::now();
        let thread_id = ThreadId::new();

        // Create stores for this subagent run
        let message_store = (self.message_store_factory)();
        let state_store = (self.state_store_factory)();

        // Create agent config
        let agent_config = AgentConfig {
            max_turns: self.config.max_turns,
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

        // Run with optional timeout
        let (mut rx, _final_state) =
            agent.run(thread_id, AgentInput::Text(task.to_string()), tool_ctx);

        let mut final_response = String::new();
        let mut total_turns = 0;
        let mut tool_count = 0u32;
        let mut tool_logs: Vec<ToolCallLog> = Vec::new();
        let mut pending_tools: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        let mut total_usage = TokenUsage::default();
        let mut success = true;

        let timeout_duration = self.config.timeout_ms.map(Duration::from_millis);

        loop {
            let recv_result = if let Some(timeout) = timeout_duration {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    final_response = "Subagent timed out".to_string();
                    success = false;
                    break;
                }
                tokio::time::timeout(remaining, rx.recv()).await
            } else {
                Ok(rx.recv().await)
            };

            match recv_result {
                Ok(Some(event)) => match event {
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
                        if let Some(ref tx) = parent_tx {
                            let _ = tx
                                .send(AgentEvent::SubagentProgress {
                                    subagent_id: subagent_id.clone(),
                                    subagent_name: self.config.name.clone(),
                                    tool_name: name,
                                    tool_context: context,
                                    completed: false,
                                    success: false,
                                    tool_count,
                                    total_tokens: u64::from(total_usage.input_tokens)
                                        + u64::from(total_usage.output_tokens),
                                })
                                .await;
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
                        if let Some(ref tx) = parent_tx {
                            let _ = tx
                                .send(AgentEvent::SubagentProgress {
                                    subagent_id: subagent_id.clone(),
                                    subagent_name: self.config.name.clone(),
                                    tool_name: name,
                                    tool_context: context,
                                    completed: true,
                                    success: tool_success,
                                    tool_count,
                                    total_tokens: u64::from(total_usage.input_tokens)
                                        + u64::from(total_usage.output_tokens),
                                })
                                .await;
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
                        final_response = message;
                        success = false;
                        break;
                    }
                    _ => {}
                },
                Ok(None) => break,
                Err(_) => {
                    final_response = "Subagent timed out".to_string();
                    success = false;
                    break;
                }
            }
        }

        Ok(SubagentResult {
            name: self.config.name.clone(),
            final_response,
            total_turns,
            tool_count,
            tool_logs,
            usage: total_usage,
            success,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
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
            // Truncate long commands
            if cmd.len() > 60 {
                format!("{}...", &cmd[..57])
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
            format!("{}...", &first_line[..47])
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
                    format!("{}...", &line[..47])
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
        // Leak the name to get 'static lifetime (acceptable for long-lived tools)
        Box::leak(format!("Subagent: {}", self.config.name).into_boxed_str())
    }

    fn description(&self) -> &'static str {
        Box::leak(
            format!(
                "Spawn a subagent named '{}' to handle a task. The subagent will work independently and return only its final response.",
                self.config.name
            )
            .into_boxed_str(),
        )
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

        // Get event channel from context for progress updates
        let parent_tx = ctx.event_tx();

        // Generate a unique ID for this subagent execution
        let subagent_id = format!(
            "{}_{:x}",
            self.config.name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let result = self.run_subagent(task, subagent_id, parent_tx).await?;

        Ok(ToolResult {
            success: result.success,
            output: result.final_response.clone(),
            data: Some(serde_json::to_value(&result).unwrap_or_default()),
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
        assert_eq!(config.max_turns, 5);
        assert_eq!(config.timeout_ms, Some(30000));
    }

    #[test]
    fn test_subagent_config_defaults() {
        let config = SubagentConfig::new("default");

        assert_eq!(config.name, "default");
        assert!(config.system_prompt.is_empty());
        assert_eq!(config.max_turns, 10);
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
