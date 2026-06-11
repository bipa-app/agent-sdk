use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for finding files by glob pattern
pub struct GlobTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> GlobTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct GlobInput {
    /// Glob pattern to match files (e.g., "**/*.rs", "src/*.ts").
    /// Relative patterns are resolved against `path` (or the environment root);
    /// an absolute pattern (starting with `/`) is used as-is.
    pattern: String,
    /// Optional directory to search in (defaults to environment root)
    #[serde(default)]
    path: Option<String>,
}

impl<E: Environment + 'static, Ctx: Send + Sync + 'static> Tool<Ctx> for GlobTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Glob
    }

    fn display_name(&self) -> &'static str {
        "Find Files"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern. Supports ** for recursive matching."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.rs', 'src/**/*.ts'). Relative to 'path' (or the environment root); an absolute pattern starting with '/' is used as-is. Must not contain '..' segments."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to environment root."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: GlobInput = GlobInput::deserialize(&input)
            .with_context(|| format!("Invalid input for glob tool: {input}"))?;

        // The raw pattern is concatenated onto the (normalized) base and passed
        // straight to `Environment::glob`, so `..` segments could escape the
        // search root in a custom Environment that does not re-normalize. Reject
        // them up front rather than relying solely on the per-result filter.
        if pattern_has_parent_segment(&input.pattern) {
            return Ok(ToolResult::error(
                "pattern must not contain '..' path segments; use the 'path' parameter to choose a search directory",
            ));
        }

        // Build the full pattern.
        //
        // - An absolute pattern (leading `/`) is used as-is; each result is
        //   still gated by the per-path `check_read` filter below.
        // - A relative pattern is joined onto the resolved base. The base is a
        //   literal filesystem path, but `glob` uses `/` as its separator on
        //   every platform and treats `\` as an escape character — so a Windows
        //   base such as `C:\Users\…` would be parsed as escape sequences.
        //   Normalise the base's separators to `/` before joining (a no-op on
        //   Unix); the user's `pattern` is left as-is so its glob
        //   metacharacters keep their meaning.
        //
        // NOTE: `Environment::glob`/`grep` implementations MUST enforce read
        // permissions per traversed path; the root-only capability check plus
        // the per-result filter here are defense-in-depth, not a substitute.
        let pattern = if input.pattern.starts_with('/') {
            input.pattern.clone()
        } else if let Some(ref base_path) = input.path {
            let base = self
                .ctx
                .environment
                .resolve_path(base_path)
                .replace('\\', "/");
            format!("{}/{}", base.trim_end_matches('/'), input.pattern)
        } else {
            let root = self.ctx.environment.root().replace('\\', "/");
            format!("{}/{}", root.trim_end_matches('/'), input.pattern)
        };

        // Check read capability for the search path
        let search_path = input.path.as_ref().map_or_else(
            || self.ctx.environment.root().to_string(),
            |p| self.ctx.environment.resolve_path(p),
        );

        if let Err(reason) = self.ctx.capabilities.check_read(&search_path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot search in '{search_path}': {reason}"
            )));
        }

        // Execute glob. A malformed pattern is the model's own input, so report
        // it as a correctable tool error rather than an infrastructure failure.
        let matches = match self.ctx.environment.glob(&pattern).await {
            Ok(matches) => matches,
            Err(err) => {
                return Ok(ToolResult::error(format!(
                    "Invalid glob pattern '{}': {err:#}",
                    input.pattern
                )));
            }
        };

        // Filter out files that the agent can't read
        let accessible_matches: Vec<_> = matches
            .into_iter()
            .filter(|path| self.ctx.capabilities.check_read(path).is_ok())
            .collect();

        if accessible_matches.is_empty() {
            return Ok(ToolResult::success(format!(
                "No files found matching pattern '{}'",
                input.pattern
            )));
        }

        let count = accessible_matches.len();
        let output = if count > 100 {
            format!(
                "Found {count} files (showing first 100):\n{}",
                accessible_matches[..100].join("\n")
            )
        } else {
            format!("Found {count} files:\n{}", accessible_matches.join("\n"))
        };

        Ok(ToolResult::success(output))
    }
}

/// Returns true if any `/`-separated component of the pattern is exactly `..`.
fn pattern_has_parent_segment(pattern: &str) -> bool {
    pattern.split('/').any(|segment| segment == "..")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> GlobTool<InMemoryFileSystem> {
        GlobTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_glob_simple_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib.rs", "pub mod foo;").await?;
        fs.write_file("README.md", "# README").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "src/*.rs"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 files"));
        assert!(result.output.contains("main.rs"));
        assert!(result.output.contains("lib.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_recursive_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib/utils.rs", "pub fn util() {}")
            .await?;
        fs.write_file("tests/test.rs", "// test").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "**/*.rs"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 3 files"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_no_matches() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "*.py"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No files found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_with_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("tests/test.rs", "// test").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"pattern": "*.rs", "path": "/workspace/src"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 files"));
        assert!(result.output.contains("main.rs"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_glob_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;

        // No read permission
        let caps = AgentCapabilities::none();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "**/*.rs"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_filters_inaccessible_files() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("secrets/key.rs", "// secret").await?;

        // Allow src but deny secrets
        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "**/*.rs"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 files"));
        assert!(result.output.contains("main.rs"));
        assert!(!result.output.contains("key.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_allowed_paths_restriction() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("config/settings.toml", "key = value").await?;

        // Full access with denied paths for config
        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/config/**".into()]);

        let tool = create_test_tool(fs, caps);

        // Searching should return src files but not config
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "**/*"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("main.rs"));
        assert!(!result.output.contains("settings.toml"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_glob_empty_directory() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.create_dir("/workspace/empty").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"pattern": "*", "path": "/workspace/empty"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No files found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_many_files_truncated() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create 150 files
        for i in 0..150 {
            fs.write_file(&format!("files/file{i}.txt"), "content")
                .await?;
        }

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "files/*.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 150 files"));
        assert!(result.output.contains("showing first 100"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::Glob);
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
        assert!(Tool::<()>::description(&tool).contains("glob"));

        let schema = Tool::<()>::input_schema(&tool);
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("pattern").is_some());
        assert!(schema["properties"].get("path").is_some());
    }

    #[tokio::test]
    async fn test_glob_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required pattern field
        let result = tool.execute(&tool_ctx(), json!({})).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_absolute_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib.rs", "pub mod foo;").await?;
        fs.write_file("README.md", "# README").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        // An absolute pattern must NOT be re-joined onto the root (which would
        // produce '/workspace//workspace/src/*.rs' and match nothing).
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "/workspace/src/*.rs"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 files"));
        assert!(result.output.contains("main.rs"));
        assert!(result.output.contains("lib.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_rejects_parent_segment() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "../*.rs"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains(".."));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_invalid_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        // An unbalanced char class is invalid; the model should get a
        // correctable tool error, not an infrastructure failure.
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "[unclosed"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Invalid glob pattern"));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_specific_file_extension() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("main.rs", "fn main() {}").await?;
        fs.write_file("main.go", "package main").await?;
        fs.write_file("main.py", "def main(): pass").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "*.rs"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 files"));
        assert!(result.output.contains("main.rs"));
        assert!(!result.output.contains("main.go"));
        assert!(!result.output.contains("main.py"));
        Ok(())
    }
}
