use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
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
    /// Glob pattern to match files (e.g., "**/*.rs", "src/*.ts")
    pattern: String,
    /// Optional directory to search in (defaults to environment root)
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for GlobTool<E> {
    fn name(&self) -> &'static str {
        "glob"
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
                    "description": "Glob pattern to match files (e.g., '**/*.rs', 'src/**/*.ts')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to environment root."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: GlobInput =
            serde_json::from_value(input).context("Invalid input for glob tool")?;

        // Build the full pattern
        let pattern = if let Some(ref base_path) = input.path {
            let base = self.ctx.environment.resolve_path(base_path);
            format!("{}/{}", base.trim_end_matches('/'), input.pattern)
        } else {
            let root = self.ctx.environment.root();
            format!("{}/{}", root.trim_end_matches('/'), input.pattern)
        };

        // Check read capability for the search path
        let search_path = input.path.as_ref().map_or_else(
            || self.ctx.environment.root().to_string(),
            |p| self.ctx.environment.resolve_path(p),
        );

        if !self.ctx.capabilities.can_read(&search_path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot search in '{search_path}'"
            )));
        }

        // Execute glob
        let matches = self
            .ctx
            .environment
            .glob(&pattern)
            .await
            .context("Failed to execute glob")?;

        // Filter out files that the agent can't read
        let accessible_matches: Vec<_> = matches
            .into_iter()
            .filter(|path| self.ctx.capabilities.can_read(path))
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

        assert_eq!(tool.name(), "glob");
        assert_eq!(tool.tier(), ToolTier::Observe);
        assert!(tool.description().contains("glob"));

        let schema = tool.input_schema();
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
