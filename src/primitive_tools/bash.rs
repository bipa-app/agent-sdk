use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
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
    /// Timeout in milliseconds (default: 120000 = 2 minutes).
    /// Accepts either an integer or a numeric string such as "5000".
    /// Uses `Option` so that explicit `null` from the model is handled
    /// gracefully (falls back to the default rather than failing
    /// deserialization).
    #[serde(
        default,
        deserialize_with = "super::deserialize_optional_u64_from_string_or_int"
    )]
    timeout_ms: Option<u64>,
}

const DEFAULT_TIMEOUT_MS: u64 = 120_000; // 2 minutes

impl<E: Environment + 'static> Tool<()> for BashTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Bash
    }

    fn display_name(&self) -> &'static str {
        "Run Command"
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
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string", "pattern": "^[0-9]+$"}
                    ],
                    "description": "Timeout in milliseconds. Accepts either an integer or a numeric string. Default: 120000 (2 minutes)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: BashInput = serde_json::from_value(input.clone())
            .with_context(|| format!("Invalid input for bash tool: {input}"))?;

        // Check exec capability and command allow/deny rules
        if let Err(reason) = self.ctx.capabilities.check_exec(&input.command) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot execute '{}': {reason}",
                truncate_command(&input.command, 100)
            )));
        }

        // Validate timeout
        let timeout_ms = input.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS).min(600_000); // Max 10 minutes

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

        let tool_result = if result.success() {
            ToolResult::success(output)
        } else {
            ToolResult::error(output)
        };

        Ok(tool_result)
    }
}

fn truncate_command(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentCapabilities;
    use crate::environment::ExecResult;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::RwLock;

    // Mock environment for testing bash execution
    struct MockBashEnvironment {
        root: String,
        // Map of command to (stdout, stderr, exit_code)
        commands: RwLock<HashMap<String, (String, String, i32)>>,
    }

    impl MockBashEnvironment {
        fn new() -> Self {
            Self {
                root: "/workspace".to_string(),
                commands: RwLock::new(HashMap::new()),
            }
        }

        fn add_command(&self, cmd: &str, stdout: &str, stderr: &str, exit_code: i32) {
            self.commands.write().unwrap().insert(
                cmd.to_string(),
                (stdout.to_string(), stderr.to_string(), exit_code),
            );
        }
    }

    #[async_trait]
    impl crate::Environment for MockBashEnvironment {
        async fn read_file(&self, _path: &str) -> Result<String> {
            Ok(String::new())
        }

        async fn read_file_bytes(&self, _path: &str) -> Result<Vec<u8>> {
            Ok(vec![])
        }

        async fn write_file(&self, _path: &str, _content: &str) -> Result<()> {
            Ok(())
        }

        async fn write_file_bytes(&self, _path: &str, _content: &[u8]) -> Result<()> {
            Ok(())
        }

        async fn list_dir(&self, _path: &str) -> Result<Vec<crate::environment::FileEntry>> {
            Ok(vec![])
        }

        async fn exists(&self, _path: &str) -> Result<bool> {
            Ok(false)
        }

        async fn is_dir(&self, _path: &str) -> Result<bool> {
            Ok(false)
        }

        async fn is_file(&self, _path: &str) -> Result<bool> {
            Ok(false)
        }

        async fn create_dir(&self, _path: &str) -> Result<()> {
            Ok(())
        }

        async fn delete_file(&self, _path: &str) -> Result<()> {
            Ok(())
        }

        async fn delete_dir(&self, _path: &str, _recursive: bool) -> Result<()> {
            Ok(())
        }

        async fn grep(
            &self,
            _pattern: &str,
            _path: &str,
            _recursive: bool,
        ) -> Result<Vec<crate::environment::GrepMatch>> {
            Ok(vec![])
        }

        async fn glob(&self, _pattern: &str) -> Result<Vec<String>> {
            Ok(vec![])
        }

        async fn exec(&self, command: &str, _timeout_ms: Option<u64>) -> Result<ExecResult> {
            let commands = self.commands.read().unwrap();
            if let Some((stdout, stderr, exit_code)) = commands.get(command) {
                Ok(ExecResult {
                    stdout: stdout.clone(),
                    stderr: stderr.clone(),
                    exit_code: *exit_code,
                })
            } else {
                // Default: command not found
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: format!("command not found: {command}"),
                    exit_code: 127,
                })
            }
        }

        fn root(&self) -> &str {
            &self.root
        }
    }

    fn create_test_tool(
        env: Arc<MockBashEnvironment>,
        capabilities: AgentCapabilities,
    ) -> BashTool<MockBashEnvironment> {
        BashTool::new(env, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_bash_simple_command() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hello", "hello\n", "", 0);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "echo hello"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("Exit code: 0"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_command_with_stderr() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("cmd", "stdout output", "stderr output", 0);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool.execute(&tool_ctx(), json!({"command": "cmd"})).await?;

        assert!(result.success);
        assert!(result.output.contains("stdout output"));
        assert!(result.output.contains("stderr output"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_command_nonzero_exit() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("failing_cmd", "", "error occurred", 1);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "failing_cmd"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Exit code: 1"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_command_not_found() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "nonexistent_cmd"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Exit code: 127"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_bash_exec_disabled() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());

        // Read-only capabilities (exec disabled)
        let caps = AgentCapabilities::read_only();

        let tool = create_test_tool(env, caps);
        let result = tool.execute(&tool_ctx(), json!({"command": "ls"})).await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        assert!(result.output.contains("execution is disabled"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_denied_commands() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());

        // Client configures denied commands
        let caps = AgentCapabilities::full_access()
            .with_denied_commands(vec![r"rm\s+-rf\s+/".into(), r"^sudo\s".into()]);

        let tool = create_test_tool(Arc::clone(&env), caps.clone());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "rm -rf /"}))
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        assert!(result.output.contains("denied pattern"));

        let tool = create_test_tool(env, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"command": "sudo apt-get install foo"}))
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_allowed_commands_restriction() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("cargo build", "Compiling...", "", 0);

        // Only allow cargo and git commands
        let caps = AgentCapabilities::full_access()
            .with_allowed_commands(vec![r"^cargo ".into(), r"^git ".into()]);

        let tool = create_test_tool(Arc::clone(&env), caps.clone());

        // cargo should be allowed
        let result = tool
            .execute(&tool_ctx(), json!({"command": "cargo build"}))
            .await?;
        assert!(result.success);

        // ls should be denied
        let tool = create_test_tool(env, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"command": "ls -la"}))
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("not in allowed list"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_bash_empty_output() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("true", "", "", 0);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "true"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("(no output)"));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_custom_timeout() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("slow_cmd", "done", "", 0);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"command": "slow_cmd", "timeout_ms": 5000}),
            )
            .await?;

        assert!(result.success);
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_tool_metadata() {
        let env = Arc::new(MockBashEnvironment::new());
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        assert_eq!(tool.name(), PrimitiveToolName::Bash);
        assert_eq!(tool.tier(), ToolTier::Confirm);
        assert!(tool.description().contains("Execute"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("command").is_some());
        assert!(schema["properties"].get("timeout_ms").is_some());
    }

    #[tokio::test]
    async fn test_bash_invalid_input() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        // Missing required command field
        let result = tool.execute(&tool_ctx(), json!({})).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_null_timeout_ms() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hello", "hello", "", 0);
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        // Model may send explicit null for optional fields — must not fail
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"command": "echo hello", "timeout_ms": null}),
            )
            .await?;

        assert!(result.success);
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_missing_timeout_uses_default() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hi", "hi", "", 0);
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        // Omitted timeout_ms should use the default
        let result = tool
            .execute(&tool_ctx(), json!({"command": "echo hi"}))
            .await?;

        assert!(result.success);
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_string_timeout_ms() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo timeout", "ok", "", 0);
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        let result = tool
            .execute(
                &tool_ctx(),
                json!({"command": "echo timeout", "timeout_ms": "5000"}),
            )
            .await?;

        assert!(result.success);
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_long_output_truncated() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        let long_output = "x".repeat(40_000);
        env.add_command("long_output_cmd", &long_output, "", 0);

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "long_output_cmd"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("output truncated"));
        assert!(result.output.len() < 35_000); // Should be truncated
        Ok(())
    }

    #[tokio::test]
    async fn test_truncate_command_function() {
        assert_eq!(truncate_command("short", 10), "short");
        assert_eq!(
            truncate_command("this is a longer command", 10),
            "this is a ..."
        );
    }
}
