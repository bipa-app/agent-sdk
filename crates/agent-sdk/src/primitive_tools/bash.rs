use crate::environment::{
    ExecSinkSpec, ExecStreamCapture, ExecStreamResult, HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES,
    HARD_MAX_EXEC_SPOOL_BYTES_PER_STREAM,
};
use crate::filesystem::create_private_exec_spool;
use crate::{
    DEFAULT_INLINE_OUTPUT_BUDGET_BYTES, Environment, PrimitiveToolName, Tool, ToolContext,
    ToolResult, ToolTier, cap_inline_from_windows,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt::Write;
use std::io::{Cursor, Read, Seek, SeekFrom};
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

/// Hard upper bound on the command timeout. Larger requests are clamped to this
/// value (and the clamp is surfaced in the tool result).
const MAX_TIMEOUT_MS: u64 = 600_000; // 10 minutes

const STDERR_SEPARATOR: &[u8] = b"\n\n--- stderr ---\n";
const NO_OUTPUT: &[u8] = b"(no output)";

impl<E: Environment + 'static, Ctx: Send + Sync + 'static> Tool<Ctx> for BashTool<E> {
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
                    "description": "Timeout in milliseconds. Accepts either an integer or a numeric string. Default: 120000 (2 minutes). Maximum: 600000 (10 minutes); larger values are clamped."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: BashInput = BashInput::deserialize(&input)
            .with_context(|| format!("Invalid input for bash tool: {input}"))?;

        if let Err(reason) = self.ctx.capabilities.check_exec(&input.command) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot execute '{}': {reason}",
                truncate_command(&input.command, 100)
            )));
        }

        let requested_timeout_ms = input.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
        let timeout_ms = requested_timeout_ms.min(MAX_TIMEOUT_MS);
        let artifact_store = ctx.artifact_store().cloned();
        let inline_budget = artifact_store.as_deref().map_or(
            DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
            agent_sdk_tools::artifacts::ArtifactStore::inline_budget,
        );
        let capture_window = inline_budget.min(HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES);
        let max_stream_bytes = artifact_store
            .as_deref()
            .map_or(
                inline_budget as u64,
                agent_sdk_tools::ArtifactStore::max_bytes_per_thread,
            )
            .min(HARD_MAX_EXEC_SPOOL_BYTES_PER_STREAM);

        let result = self
            .ctx
            .environment
            .exec_streamed(
                &input.command,
                Some(timeout_ms),
                ExecSinkSpec {
                    stdout: create_private_exec_spool()?,
                    stderr: create_private_exec_spool()?,
                    head_bytes: capture_window,
                    tail_bytes: capture_window,
                    max_bytes_per_stream: max_stream_bytes,
                },
            )
            .await
            .context("Failed to execute command")?;

        let suffix = exit_suffix(result.exit_code, requested_timeout_ms);
        let total_bytes = composed_output_len(&result, &suffix)?;
        let success = result.success();

        if total_bytes <= inline_budget as u64
            && let Some(bytes) = complete_output(&result, &suffix)
            && let Ok(output) = String::from_utf8(bytes)
        {
            return Ok(if success {
                ToolResult::success(output)
            } else {
                ToolResult::error(output)
            });
        }

        let Some(store) = artifact_store else {
            return Ok(ToolResult::error(format!(
                "Command output was {total_bytes} bytes, but no artifact store is configured. \
                 The output was not placed in the transcript; configure artifact storage or \
                 re-run the command with narrower output."
            )));
        };

        let (head, tail) = composed_windows(&result, &suffix, inline_budget)?;
        let persist_suffix = suffix.clone();
        match run_blocking_io("joining bash artifact persistence", move || {
            persist_output(&store, result, &persist_suffix)
        })
        .await
        {
            Ok(saved) => {
                let output =
                    cap_inline_from_windows(&head, &tail, total_bytes, inline_budget, saved.id);
                Ok(if success {
                    ToolResult::success(output)
                } else {
                    ToolResult::error(output)
                })
            }
            Err(error) => {
                log::warn!("bash artifact spill failed: {error:#}");
                Ok(ToolResult::error(format!(
                    "Command output was {total_bytes} bytes, but lossless artifact persistence \
                     failed. The output was not placed in the transcript; re-run the command \
                     with narrower output."
                )))
            }
        }
    }
}

async fn run_blocking_io<T, F>(context: &'static str, operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .with_context(|| context)?
}

#[derive(Clone, Copy)]
enum OutputSegment<'a> {
    Bytes(&'a [u8]),
    Capture(&'a ExecStreamCapture),
}

impl OutputSegment<'_> {
    const fn len(self) -> u64 {
        match self {
            Self::Bytes(bytes) => bytes.len() as u64,
            Self::Capture(capture) => capture.total_bytes,
        }
    }

    fn append_prefix(self, output: &mut Vec<u8>, bytes: usize) -> Result<()> {
        match self {
            Self::Bytes(value) => output.extend_from_slice(&value[..bytes]),
            Self::Capture(capture) => {
                ensure!(
                    bytes <= capture.head.len(),
                    "process capture head window was shorter than requested"
                );
                output.extend_from_slice(&capture.head[..bytes]);
            }
        }
        Ok(())
    }

    fn append_suffix(self, output: &mut Vec<u8>, bytes: usize) -> Result<()> {
        match self {
            Self::Bytes(value) => output.extend_from_slice(&value[value.len() - bytes..]),
            Self::Capture(capture) => {
                if bytes <= capture.tail.len() {
                    output.extend_from_slice(&capture.tail[capture.tail.len() - bytes..]);
                    return Ok(());
                }
                let captured = capture
                    .head
                    .len()
                    .checked_add(capture.tail.len())
                    .context("process capture window length overflowed usize")?;
                ensure!(
                    captured as u64 == capture.total_bytes && bytes <= captured,
                    "process capture tail window was shorter than requested"
                );
                let from_head = bytes - capture.tail.len();
                output.extend_from_slice(&capture.head[capture.head.len() - from_head..]);
                output.extend_from_slice(&capture.tail);
            }
        }
        Ok(())
    }
}

fn exit_suffix(exit_code: i32, requested_timeout_ms: u64) -> Vec<u8> {
    let mut suffix = format!("\n\nExit code: {exit_code}");
    if requested_timeout_ms > MAX_TIMEOUT_MS {
        let _ = write!(
            suffix,
            "\n\n(requested timeout {requested_timeout_ms}ms exceeds the maximum of {MAX_TIMEOUT_MS}ms; clamped to {MAX_TIMEOUT_MS}ms)"
        );
    }
    suffix.into_bytes()
}

fn output_segments<'a>(result: &'a ExecStreamResult, suffix: &'a [u8]) -> Vec<OutputSegment<'a>> {
    let mut segments = Vec::with_capacity(5);
    if result.stdout.total_bytes == 0 && result.stderr.total_bytes == 0 {
        segments.push(OutputSegment::Bytes(NO_OUTPUT));
    } else {
        segments.push(OutputSegment::Capture(&result.stdout));
        if result.stdout.total_bytes > 0 && result.stderr.total_bytes > 0 {
            segments.push(OutputSegment::Bytes(STDERR_SEPARATOR));
        }
        segments.push(OutputSegment::Capture(&result.stderr));
    }
    segments.push(OutputSegment::Bytes(suffix));
    segments
}

fn composed_output_len(result: &ExecStreamResult, suffix: &[u8]) -> Result<u64> {
    output_segments(result, suffix)
        .into_iter()
        .try_fold(0_u64, |total, segment| {
            total
                .checked_add(segment.len())
                .context("formatted command output length overflowed u64")
        })
}

fn complete_output(result: &ExecStreamResult, suffix: &[u8]) -> Option<Vec<u8>> {
    let capacity = usize::try_from(composed_output_len(result, suffix).ok()?).ok()?;
    let mut output = Vec::with_capacity(capacity);
    for segment in output_segments(result, suffix) {
        match segment {
            OutputSegment::Bytes(bytes) => output.extend_from_slice(bytes),
            OutputSegment::Capture(capture) => {
                output.extend_from_slice(&capture.complete_bytes()?);
            }
        }
    }
    Some(output)
}

fn composed_windows(
    result: &ExecStreamResult,
    suffix: &[u8],
    window_bytes: usize,
) -> Result<(String, String)> {
    let segments = output_segments(result, suffix);
    let mut head = Vec::with_capacity(window_bytes);
    for segment in &segments {
        let remaining = window_bytes - head.len();
        if remaining == 0 {
            break;
        }
        let take = usize::try_from(segment.len().min(remaining as u64))
            .context("head window length overflowed usize")?;
        segment.append_prefix(&mut head, take)?;
    }

    let mut reversed_tail = Vec::new();
    let mut retained = 0_usize;
    for segment in segments.iter().rev() {
        let remaining = window_bytes - retained;
        if remaining == 0 {
            break;
        }
        let take = usize::try_from(segment.len().min(remaining as u64))
            .context("tail window length overflowed usize")?;
        let mut piece = Vec::with_capacity(take);
        segment.append_suffix(&mut piece, take)?;
        retained += piece.len();
        reversed_tail.push(piece);
    }
    let mut tail = Vec::with_capacity(retained);
    for piece in reversed_tail.into_iter().rev() {
        tail.extend_from_slice(&piece);
    }

    Ok((
        String::from_utf8_lossy(&head).into_owned(),
        String::from_utf8_lossy(&tail).into_owned(),
    ))
}

fn persist_output(
    store: &agent_sdk_tools::artifacts::ArtifactStore,
    result: ExecStreamResult,
    suffix: &[u8],
) -> Result<agent_sdk_tools::artifacts::SavedArtifact> {
    ensure!(
        result.stdout.spool.metadata()?.len() == result.stdout.total_bytes,
        "private stdout spool was incomplete"
    );
    ensure!(
        result.stderr.spool.metadata()?.len() == result.stderr.total_bytes,
        "private stderr spool was incomplete"
    );

    let stdout_bytes = result.stdout.total_bytes;
    let stderr_bytes = result.stderr.total_bytes;
    let no_output = stdout_bytes == 0 && stderr_bytes == 0;
    let separator = if stdout_bytes > 0 && stderr_bytes > 0 {
        STDERR_SEPARATOR
    } else {
        &[]
    };
    let mut stdout = result.stdout.spool;
    let mut stderr = result.stderr.spool;
    stdout
        .seek(SeekFrom::Start(0))
        .context("failed to rewind private stdout spool")?;
    stderr
        .seek(SeekFrom::Start(0))
        .context("failed to rewind private stderr spool")?;

    let prefix = if no_output { NO_OUTPUT } else { &[] };
    let mut source = Cursor::new(prefix)
        .chain(stdout.take(stdout_bytes))
        .chain(Cursor::new(separator))
        .chain(stderr.take(stderr_bytes))
        .chain(Cursor::new(suffix));
    store.save_streamed("bash", &mut source)
}

fn truncate_command(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", super::truncate_str(s, max_len))
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
        // Records the timeout forwarded to the most recent `exec` call so tests
        // can assert default/parsed/clamped values are passed through.
        last_timeout_ms: RwLock<Option<u64>>,
    }

    impl MockBashEnvironment {
        fn new() -> Self {
            Self {
                root: "/workspace".to_string(),
                commands: RwLock::new(HashMap::new()),
                last_timeout_ms: RwLock::new(None),
            }
        }

        fn add_command(&self, cmd: &str, stdout: &str, stderr: &str, exit_code: i32) -> Result<()> {
            self.commands.write().ok().context("lock poisoned")?.insert(
                cmd.to_string(),
                (stdout.to_string(), stderr.to_string(), exit_code),
            );
            Ok(())
        }

        fn recorded_timeout(&self) -> Result<Option<u64>> {
            Ok(*self.last_timeout_ms.read().ok().context("lock poisoned")?)
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

        async fn exec(&self, command: &str, timeout_ms: Option<u64>) -> Result<ExecResult> {
            *self.last_timeout_ms.write().ok().context("lock poisoned")? = timeout_ms;
            let commands = self.commands.read().ok().context("lock poisoned")?;
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

    fn artifact_id_from_footer(output: &str) -> Result<u64> {
        let (_, suffix) = output
            .rsplit_once("artifact://")
            .context("missing artifact footer")?;
        suffix
            .strip_suffix(']')
            .context("malformed artifact footer")?
            .parse()
            .context("invalid artifact id")
    }

    fn assert_zero_bytes(reader: &mut impl Read, mut remaining: u64) -> Result<()> {
        let mut chunk = [1_u8; 8192];
        while remaining > 0 {
            let take = usize::try_from(remaining.min(chunk.len() as u64))
                .context("zero-byte assertion length overflowed usize")?;
            reader.read_exact(&mut chunk[..take])?;
            assert!(chunk[..take].iter().all(|byte| *byte == 0));
            remaining -= take as u64;
        }
        Ok(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_bash_simple_command() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hello", "hello\n", "", 0)?;

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
        env.add_command("cmd", "stdout output", "stderr output", 0)?;

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
        env.add_command("failing_cmd", "", "error occurred", 1)?;

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
        env.add_command("cargo build", "Compiling...", "", 0)?;

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
        env.add_command("true", "", "", 0)?;

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
        env.add_command("slow_cmd", "done", "", 0)?;

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

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::Bash);
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Confirm);
        assert!(Tool::<()>::description(&tool).contains("Execute"));

        let schema = Tool::<()>::input_schema(&tool);
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
        env.add_command("echo hello", "hello", "", 0)?;
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
        env.add_command("echo hi", "hi", "", 0)?;
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
        env.add_command("echo timeout", "ok", "", 0)?;
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
    async fn test_bash_long_output_reaches_shared_budget_checkpoint() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        let long_output = "x".repeat(40_000);
        env.add_command("long_output_cmd", &long_output, "", 0)?;

        let tool = create_test_tool(env, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"command": "long_output_cmd"}))
            .await?;

        assert!(result.success);
        assert_eq!(
            result.output,
            format!("{long_output}\n\nExit code: 0"),
            "the producer must preserve every byte for the shared spill authority",
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_storeless_overflow_fails_explicitly_and_bounded() -> Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        let long_output = "x".repeat(DEFAULT_INLINE_OUTPUT_BUDGET_BYTES + 1);
        env.add_command("overflow", &long_output, "", 0)?;
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        let error = tool
            .execute(&tool_ctx(), json!({"command": "overflow"}))
            .await
            .err()
            .context("storeless overflow must fail")?;
        let rendered = format!("{error:#}");
        assert!(rendered.contains("spool limit"), "got: {rendered}");
        assert!(rendered.len() < 512, "error must remain bounded");
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_streams_raw_stdout_stderr_to_one_durable_artifact() -> Result<()> {
        const STDOUT_BYTES: u64 = 2 * 1024 * 1024;
        const STDERR_BYTES: u64 = 3 * 1024 * 1024;

        let temp = tempfile::tempdir()?;
        let environment = Arc::new(crate::LocalFileSystem::new(temp.path()));
        let store = Arc::new(
            crate::ArtifactStore::new(temp.path().join("artifacts")).with_inline_budget(4096),
        );
        let tool = BashTool::new(environment, AgentCapabilities::full_access());
        let ctx = ToolContext::new(()).with_artifact_store(Arc::clone(&store));
        let result = tool
            .execute(
                &ctx,
                json!({
                    "command": "dd if=/dev/zero bs=1048576 count=2 2>/dev/null; \
                                dd if=/dev/zero bs=1048576 count=3 1>&2 2>/dev/null; \
                                exit 7"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.len() <= store.inline_budget());
        let id = artifact_id_from_footer(&result.output)?;
        assert!(result.output.ends_with(&crate::artifact_footer(id)));

        let mut artifact = store.resolve(id)?;
        let expected_len = STDOUT_BYTES
            + STDERR_SEPARATOR.len() as u64
            + STDERR_BYTES
            + b"\n\nExit code: 7".len() as u64;
        assert_eq!(artifact.metadata()?.len(), expected_len);
        assert_zero_bytes(&mut artifact, STDOUT_BYTES)?;
        let mut separator = vec![0_u8; STDERR_SEPARATOR.len()];
        artifact.read_exact(&mut separator)?;
        assert_eq!(separator, STDERR_SEPARATOR);
        assert_zero_bytes(&mut artifact, STDERR_BYTES)?;
        let mut suffix = [0_u8; 14];
        artifact.read_exact(&mut suffix)?;
        assert_eq!(&suffix, b"\n\nExit code: 7");
        let mut extra = [0_u8; 1];
        assert_eq!(artifact.read(&mut extra)?, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_preserves_small_invalid_utf8_in_artifact() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let environment = Arc::new(crate::LocalFileSystem::new(temp.path()));
        let store = Arc::new(
            crate::ArtifactStore::new(temp.path().join("artifacts")).with_inline_budget(4096),
        );
        let tool = BashTool::new(environment, AgentCapabilities::full_access());
        let ctx = ToolContext::new(()).with_artifact_store(Arc::clone(&store));
        let result = tool
            .execute(&ctx, json!({"command": "printf '\\377\\376'"}))
            .await?;

        assert!(result.success);
        let id = artifact_id_from_footer(&result.output)?;
        let mut artifact = store.resolve(id)?;
        let mut raw = [0_u8; 16];
        artifact.read_exact(&mut raw)?;
        assert_eq!(&raw, b"\xff\xfe\n\nExit code: 0");
        let mut extra = [0_u8; 1];
        assert_eq!(artifact.read(&mut extra)?, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_custom_environment_uses_compatible_default_streaming_exec() -> Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("custom", "stdout", "stderr", 0)?;
        let tool = create_test_tool(env, AgentCapabilities::full_access());

        let result = tool
            .execute(&tool_ctx(), json!({"command": "custom"}))
            .await?;

        assert_eq!(
            result.output,
            "stdout\n\n--- stderr ---\nstderr\n\nExit code: 0"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_default_timeout_forwarded() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hi", "hi", "", 0)?;

        let tool = create_test_tool(Arc::clone(&env), AgentCapabilities::full_access());
        tool.execute(&tool_ctx(), json!({"command": "echo hi"}))
            .await?;

        // Omitted timeout forwards the default.
        assert_eq!(env.recorded_timeout()?, Some(DEFAULT_TIMEOUT_MS));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_explicit_timeout_forwarded() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hi", "hi", "", 0)?;

        let tool = create_test_tool(Arc::clone(&env), AgentCapabilities::full_access());
        tool.execute(
            &tool_ctx(),
            json!({"command": "echo hi", "timeout_ms": 5000}),
        )
        .await?;

        assert_eq!(env.recorded_timeout()?, Some(5000));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_string_timeout_forwarded() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hi", "hi", "", 0)?;

        let tool = create_test_tool(Arc::clone(&env), AgentCapabilities::full_access());
        tool.execute(
            &tool_ctx(),
            json!({"command": "echo hi", "timeout_ms": "5000"}),
        )
        .await?;

        // Numeric strings are parsed before being forwarded.
        assert_eq!(env.recorded_timeout()?, Some(5000));
        Ok(())
    }

    #[tokio::test]
    async fn test_bash_timeout_clamped_to_max() -> anyhow::Result<()> {
        let env = Arc::new(MockBashEnvironment::new());
        env.add_command("echo hi", "hi", "", 0)?;

        let tool = create_test_tool(Arc::clone(&env), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"command": "echo hi", "timeout_ms": 999_999_999_u64}),
            )
            .await?;

        // Oversized requests are clamped to the maximum and the clamp is surfaced.
        assert_eq!(env.recorded_timeout()?, Some(MAX_TIMEOUT_MS));
        assert!(result.output.contains("clamped"));
        Ok(())
    }
    #[tokio::test(flavor = "current_thread")]
    async fn artifact_persistence_keeps_async_runtime_responsive() -> Result<()> {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let started = std::time::Instant::now();
        let persistence = tokio::spawn(run_blocking_io("join blocking I/O probe", move || {
            let _ = entered_tx.send(());
            release_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .context("blocking I/O probe timed out")?;
            Ok(())
        }));

        entered_rx
            .await
            .context("blocking I/O probe did not start")?;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            started.elapsed() < std::time::Duration::from_millis(500),
            "synchronous persistence blocked the async executor"
        );
        release_tx.send(()).context("release blocking I/O probe")?;
        persistence.await.context("join persistence probe")??;
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
