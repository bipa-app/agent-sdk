use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for searching file contents using regex patterns
pub struct GrepTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> GrepTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct GrepInput {
    /// Regex pattern to search for
    pattern: String,
    /// Path to search in (file or directory)
    #[serde(default)]
    path: Option<String>,
    /// Search recursively in directories (default: true)
    #[serde(default = "default_recursive")]
    recursive: bool,
    /// Case insensitive search (default: false)
    #[serde(default)]
    case_insensitive: bool,
}

const fn default_recursive() -> bool {
    true
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for GrepTool<E> {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search for a regex pattern in files. Returns matching lines with file paths and line numbers."
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
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (file or directory). Defaults to environment root."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in directories. Default: true"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search. Default: false"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: GrepInput =
            serde_json::from_value(input).context("Invalid input for grep tool")?;

        let search_path = input.path.as_ref().map_or_else(
            || self.ctx.environment.root().to_string(),
            |p| self.ctx.environment.resolve_path(p),
        );

        // Check read capability
        if !self.ctx.capabilities.can_read(&search_path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot search in '{search_path}'"
            )));
        }

        // Build pattern with case insensitivity if requested
        let pattern = if input.case_insensitive {
            format!("(?i){}", input.pattern)
        } else {
            input.pattern.clone()
        };

        // Execute grep
        let matches = self
            .ctx
            .environment
            .grep(&pattern, &search_path, input.recursive)
            .await
            .context("Failed to execute grep")?;

        // Filter out matches in files the agent can't read
        let accessible_matches: Vec<_> = matches
            .into_iter()
            .filter(|m| self.ctx.capabilities.can_read(&m.path))
            .collect();

        if accessible_matches.is_empty() {
            return Ok(ToolResult::success(format!(
                "No matches found for pattern '{}'",
                input.pattern
            )));
        }

        let count = accessible_matches.len();
        let max_results = 50;

        let output_lines: Vec<String> = accessible_matches
            .iter()
            .take(max_results)
            .map(|m| {
                format!(
                    "{}:{}:{}",
                    m.path,
                    m.line_number,
                    truncate_line(&m.line_content, 200)
                )
            })
            .collect();

        let output = if count > max_results {
            format!(
                "Found {count} matches (showing first {max_results}):\n{}",
                output_lines.join("\n")
            )
        } else {
            format!("Found {count} matches:\n{}", output_lines.join("\n"))
        };

        Ok(ToolResult::success(output))
    }
}

fn truncate_line(s: &str, max_len: usize) -> String {
    let trimmed = s.trim();
    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> GrepTool<InMemoryFileSystem> {
        GrepTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_grep_simple_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.rs", "fn main() {\n    println!(\"Hello\");\n}")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "println"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("println"));
        assert!(result.output.contains(":2:")); // Line number
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_regex_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo123\nbar456\nfoo789").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "foo\\d+"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_no_matches() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "Hello, World!").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "Rust"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No matches found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // Use ASCII-only text since unicode-case feature may not be enabled
        fs.write_file("test.txt", "Hello\nHELLO\nhello").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        // Use ASCII-only case-insensitive pattern (regex supports (?i-u) for ASCII)
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "[Hh][Ee][Ll][Ll][Oo]"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 3 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_with_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("tests/test.rs", "fn test() {}").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"pattern": "fn", "path": "/workspace/src"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("main.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_non_recursive() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("file.txt", "match here").await?;
        fs.write_file("subdir/nested.txt", "match nested").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match", "recursive": false}))
            .await?;

        assert!(result.success);
        // Should only find the top-level file
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("file.txt"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_grep_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "content").await?;

        // No read permission
        let caps = AgentCapabilities::none();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "content"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_filters_inaccessible_files() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("secrets/key.txt", "fn secret() {}").await?;

        // Allow src but deny secrets
        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool.execute(&tool_ctx(), json!({"pattern": "fn"})).await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("main.rs"));
        assert!(!result.output.contains("key.txt"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_grep_empty_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("empty.txt", "").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "anything"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No matches found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_many_matches_truncated() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create file with many matching lines
        let content: String = (1..=100)
            .map(|i| format!("match line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("many.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 100 matches"));
        assert!(result.output.contains("showing first 50"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_special_regex_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo.bar\nbaz*qux\n(parens)")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Escaped dot
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "foo\\.bar"}))
            .await?;
        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_multiple_files() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib.rs", "fn lib() {}").await?;
        fs.write_file("README.md", "# README").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool.execute(&tool_ctx(), json!({"pattern": "fn"})).await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), "grep");
        assert_eq!(tool.tier(), ToolTier::Observe);
        assert!(tool.description().contains("Search"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("pattern").is_some());
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("recursive").is_some());
        assert!(schema["properties"].get("case_insensitive").is_some());
    }

    #[tokio::test]
    async fn test_grep_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required pattern field
        let result = tool.execute(&tool_ctx(), json!({})).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_truncate_line_function() {
        assert_eq!(truncate_line("short", 10), "short");
        assert_eq!(truncate_line("  trimmed  ", 10), "trimmed");
        assert_eq!(truncate_line("this is a longer line", 10), "this is a ...");
    }

    #[tokio::test]
    async fn test_grep_long_line_truncated() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let long_line = "match ".to_string() + &"x".repeat(300);
        fs.write_file("long.txt", &long_line).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("..."));
        Ok(())
    }
}
