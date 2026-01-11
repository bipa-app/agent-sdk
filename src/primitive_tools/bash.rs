use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt::Write;
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for executing shell commands
pub struct BashTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> BashTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct BashInput {
    /// Command to execute
    command: String,
    /// Timeout in milliseconds (default: 120000 = 2 minutes)
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}

const fn default_timeout() -> u64 {
    120_000 // 2 minutes
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for BashTool<E> {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command. Use for git, npm, cargo, and other CLI tools. Returns stdout, stderr, and exit code."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds. Default: 120000 (2 minutes)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: BashInput =
            serde_json::from_value(input).context("Invalid input for bash tool")?;

        // Check exec capability
        if !self.ctx.capabilities.exec {
            return Ok(ToolResult::error(
                "Permission denied: command execution is disabled",
            ));
        }

        // Check if command is allowed
        if !self.ctx.capabilities.can_exec(&input.command) {
            return Ok(ToolResult::error(format!(
                "Permission denied: command '{}' is not allowed",
                truncate_command(&input.command, 100)
            )));
        }

        // Validate timeout
        let timeout_ms = input.timeout_ms.min(600_000); // Max 10 minutes

        // Execute command
        let result = self
            .ctx
            .environment
            .exec(&input.command, Some(timeout_ms))
            .await
            .context("Failed to execute command")?;

        // Format output
        let mut output = String::new();

        if !result.stdout.is_empty() {
            output.push_str(&result.stdout);
        }

        if !result.stderr.is_empty() {
            if !output.is_empty() {
                output.push_str("\n\n--- stderr ---\n");
            }
            output.push_str(&result.stderr);
        }

        if output.is_empty() {
            output = "(no output)".to_string();
        }

        // Truncate if too long
        let max_output_len = 30_000;
        if output.len() > max_output_len {
            output = format!(
                "{}...\n\n(output truncated, {} total characters)",
                &output[..max_output_len],
                output.len()
            );
        }

        // Include exit code in output
        let _ = write!(output, "\n\nExit code: {}", result.exit_code);

        if result.success() {
            Ok(ToolResult::success(output))
        } else {
            Ok(ToolResult::error(output))
        }
    }
}

fn truncate_command(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
