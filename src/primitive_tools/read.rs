use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Maximum tokens allowed per file read (approximately 4 chars per token)
const MAX_TOKENS: usize = 25_000;
const CHARS_PER_TOKEN: usize = 4;

/// Tool for reading file contents
pub struct ReadTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> ReadTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadInput {
    /// Path to the file to read (also accepts `file_path` for compatibility)
    #[serde(alias = "file_path")]
    path: String,
    /// Optional line offset to start from (1-based)
    #[serde(default)]
    offset: Option<usize>,
    /// Optional number of lines to read
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for ReadTool<E> {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read file contents. Can optionally specify offset and limit for large files."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (1-based). Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Optional."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: ReadInput =
            serde_json::from_value(input).context("Invalid input for read tool")?;

        let path = self.ctx.environment.resolve_path(&input.path);

        // Check capabilities
        if !self.ctx.capabilities.can_read(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot read '{path}'"
            )));
        }

        // Check if file exists
        let exists = self
            .ctx
            .environment
            .exists(&path)
            .await
            .context("Failed to check file existence")?;

        if !exists {
            return Ok(ToolResult::error(format!("File not found: '{path}'")));
        }

        // Check if it's a directory
        let is_dir = self
            .ctx
            .environment
            .is_dir(&path)
            .await
            .context("Failed to check if path is directory")?;

        if is_dir {
            return Ok(ToolResult::error(format!(
                "'{path}' is a directory, not a file"
            )));
        }

        // Read file
        let content = self
            .ctx
            .environment
            .read_file(&path)
            .await
            .context("Failed to read file")?;

        // Apply offset and limit if specified
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let offset = input.offset.unwrap_or(1).saturating_sub(1); // Convert to 0-based

        // Calculate the content that would be returned
        let selected_lines: Vec<&str> = lines.iter().copied().skip(offset).collect();

        // Check if user specified a limit, otherwise we need to check token limit
        let limit = if let Some(user_limit) = input.limit {
            user_limit
        } else {
            // Estimate tokens for the selected content
            let selected_content_len: usize =
                selected_lines.iter().map(|line| line.len() + 1).sum(); // +1 for newline
            let estimated_tokens = selected_content_len / CHARS_PER_TOKEN;

            if estimated_tokens > MAX_TOKENS {
                // File exceeds token limit, return helpful message
                let suggested_limit = estimate_lines_for_tokens(&selected_lines, MAX_TOKENS);
                return Ok(ToolResult::success(format!(
                    "File too large to read at once (~{estimated_tokens} tokens, max {MAX_TOKENS}).\n\
                     Total lines: {total_lines}\n\n\
                     Use 'offset' and 'limit' parameters to read specific portions.\n\
                     Suggested: Start with offset=1, limit={suggested_limit} to read the first ~{MAX_TOKENS} tokens.\n\n\
                     Example: {{\"path\": \"{path}\", \"offset\": 1, \"limit\": {suggested_limit}}}"
                )));
            }
            selected_lines.len()
        };

        let selected_lines: Vec<String> = lines
            .into_iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(i, line)| format!("{:>6}\t{}", offset + i + 1, line))
            .collect();

        let output = if selected_lines.is_empty() {
            "(empty file)".to_string()
        } else {
            let header = if input.offset.is_some() || input.limit.is_some() {
                format!(
                    "Showing lines {}-{} of {} total\n",
                    offset + 1,
                    (offset + selected_lines.len()).min(total_lines),
                    total_lines
                )
            } else {
                String::new()
            };
            format!("{header}{}", selected_lines.join("\n"))
        };

        Ok(ToolResult::success(output))
    }
}

/// Estimate how many lines can fit within a token budget
fn estimate_lines_for_tokens(lines: &[&str], max_tokens: usize) -> usize {
    let max_chars = max_tokens * CHARS_PER_TOKEN;
    let mut total_chars = 0;
    let mut line_count = 0;

    for line in lines {
        let line_chars = line.len() + 1; // +1 for newline
        if total_chars + line_chars > max_chars {
            break;
        }
        total_chars += line_chars;
        line_count += 1;
    }

    // Return at least 1 to avoid suggesting limit=0
    line_count.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> ReadTool<InMemoryFileSystem> {
        ReadTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_read_entire_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("line 1"));
        assert!(result.output.contains("line 2"));
        assert!(result.output.contains("line 3"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_offset() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 3}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 3-5 of 5 total"));
        assert!(result.output.contains("line 3"));
        assert!(result.output.contains("line 4"));
        assert!(result.output.contains("line 5"));
        assert!(!result.output.contains("\tline 1")); // Should not include line 1
        assert!(!result.output.contains("\tline 2")); // Should not include line 2
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 1-2 of 5 total"));
        assert!(result.output.contains("line 1"));
        assert!(result.output.contains("line 2"));
        assert!(!result.output.contains("\tline 3"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 2, "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 2-3 of 5 total"));
        assert!(result.output.contains("line 2"));
        assert!(result.output.contains("line 3"));
        assert!(!result.output.contains("\tline 1"));
        assert!(!result.output.contains("\tline 4"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/nonexistent.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("File not found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_directory_returns_error() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.create_dir("/workspace/subdir").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/subdir"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("is a directory"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_read_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secret.txt", "secret content").await?;

        // Create read-only capabilities that deny all paths
        let caps = AgentCapabilities::none();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/secret.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_denied_path_via_capabilities() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secrets/api_key.txt", "API_KEY=secret")
            .await?;

        // Custom capabilities that deny secrets directory with absolute path pattern
        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/secrets/api_key.txt"}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_allowed_path_restriction() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("config/settings.toml", "key = value").await?;

        // Only allow reading from src/
        let caps = AgentCapabilities::read_only()
            .with_denied_paths(vec![])
            .with_allowed_paths(vec!["/workspace/src/**".into()]);

        let tool = create_test_tool(Arc::clone(&fs), caps.clone());

        // Should be able to read from src/
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/src/main.rs"}))
            .await?;
        assert!(result.success);

        // Should NOT be able to read from config/
        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/config/settings.toml"}),
            )
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_read_empty_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("empty.txt", "").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/empty.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("(empty file)"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_with_pagination() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create a file with 100 lines
        let content: String = (1..=100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("large.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Read lines 50-60
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/large.txt", "offset": 50, "limit": 10}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 50-59 of 100 total"));
        assert!(result.output.contains("line 50"));
        assert!(result.output.contains("line 59"));
        assert!(!result.output.contains("\tline 49"));
        assert!(!result.output.contains("\tline 60"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_offset_beyond_file_length() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("short.txt", "line 1\nline 2").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/short.txt", "offset": 100}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("(empty file)"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_file_with_special_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content = "特殊字符\néàü\n🎉emoji\ntab\there";
        fs.write_file("special.txt", content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/special.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("特殊字符"));
        assert!(result.output.contains("éàü"));
        assert!(result.output.contains("🎉emoji"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), "read");
        assert_eq!(tool.tier(), ToolTier::Observe);
        assert!(tool.description().contains("Read"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("path").is_some());
    }

    #[tokio::test]
    async fn test_read_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required path field
        let result = tool.execute(&tool_ctx(), json!({})).await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_exceeds_token_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create a file that exceeds 25k tokens (~100k chars)
        // Each line is ~100 chars, need ~1000 lines to exceed limit
        let line = "x".repeat(100);
        let content: String = (1..=1500)
            .map(|i| format!("{i}: {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("huge.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/huge.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("File too large to read at once"));
        assert!(result.output.contains("Total lines: 1500"));
        assert!(result.output.contains("offset"));
        assert!(result.output.contains("limit"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_with_explicit_limit_bypasses_check() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create a large file
        let line = "x".repeat(100);
        let content: String = (1..=1500)
            .map(|i| format!("{i}: {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("huge.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // With explicit limit, should return the requested lines
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/huge.txt", "offset": 1, "limit": 10}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 1-10 of 1500 total"));
        assert!(!result.output.contains("File too large"));
        Ok(())
    }

    #[test]
    fn test_estimate_lines_for_tokens() {
        let lines: Vec<&str> = vec![
            "short line",           // 11 chars
            "another short line",   // 19 chars
            "x".repeat(100).leak(), // 100 chars
        ];

        // With 10 tokens (40 chars), should fit first 2 lines (11 + 19 = 30 chars)
        let count = estimate_lines_for_tokens(&lines, 10);
        assert_eq!(count, 2);

        // With 1 token (4 chars), should return at least 1
        let count = estimate_lines_for_tokens(&lines, 1);
        assert_eq!(count, 1);
    }
}
