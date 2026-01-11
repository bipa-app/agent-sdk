use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for writing file contents
pub struct WriteTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> WriteTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct WriteInput {
    /// Path to the file to write
    path: String,
    /// Content to write to the file
    content: String,
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for WriteTool<E> {
    fn name(&self) -> &'static str {
        "write"
    }

    fn description(&self) -> &'static str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: WriteInput =
            serde_json::from_value(input).context("Invalid input for write tool")?;

        let path = self.ctx.environment.resolve_path(&input.path);

        // Check capabilities
        if !self.ctx.capabilities.can_write(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot write to '{path}'"
            )));
        }

        // Check if target is a directory
        let exists = self
            .ctx
            .environment
            .exists(&path)
            .await
            .context("Failed to check path existence")?;

        if exists {
            let is_dir = self
                .ctx
                .environment
                .is_dir(&path)
                .await
                .context("Failed to check if path is directory")?;

            if is_dir {
                return Ok(ToolResult::error(format!(
                    "'{path}' is a directory, cannot write"
                )));
            }
        }

        // Write file
        self.ctx
            .environment
            .write_file(&path, &input.content)
            .await
            .context("Failed to write file")?;

        let lines = input.content.lines().count();
        let bytes = input.content.len();

        Ok(ToolResult::success(format!(
            "Successfully wrote {lines} lines ({bytes} bytes) to '{path}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> WriteTool<InMemoryFileSystem> {
        WriteTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_write_new_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/new_file.txt", "content": "Hello, World!"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Successfully wrote"));
        assert!(result.output.contains("1 lines"));
        assert!(result.output.contains("13 bytes"));

        // Verify file was created
        let content = fs.read_file("/workspace/new_file.txt").await?;
        assert_eq!(content, "Hello, World!");
        Ok(())
    }

    #[tokio::test]
    async fn test_write_overwrite_existing_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("existing.txt", "old content").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/existing.txt", "content": "new content"}),
            )
            .await?;

        assert!(result.success);

        // Verify file was overwritten
        let content = fs.read_file("/workspace/existing.txt").await?;
        assert_eq!(content, "new content");
        Ok(())
    }

    #[tokio::test]
    async fn test_write_multiline_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content = "line 1\nline 2\nline 3\nline 4";

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/multi.txt", "content": content}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("4 lines"));

        // Verify content
        let read_content = fs.read_file("/workspace/multi.txt").await?;
        assert_eq!(read_content, content);
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_write_permission_denied_no_write_capability() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Read-only capabilities
        let caps = AgentCapabilities::read_only();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "content": "content"}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_write_permission_denied_via_denied_paths() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Full access but deny secrets directory
        let caps = AgentCapabilities::full_access()
            .with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/secrets/key.txt", "content": "secret"}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_write_allowed_path_restriction() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Only allow writing to src/
        let caps = AgentCapabilities::full_access()
            .with_denied_paths(vec![])
            .with_allowed_paths(vec!["/workspace/src/**".into()]);

        let tool = create_test_tool(Arc::clone(&fs), caps.clone());

        // Should be able to write to src/
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/src/main.rs", "content": "fn main() {}"}),
            )
            .await?;
        assert!(result.success);

        // Should NOT be able to write to config/
        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/config/settings.toml", "content": "key = value"}),
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
    async fn test_write_to_nested_directory() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/deep/nested/dir/file.txt", "content": "nested content"}),
            )
            .await?;

        assert!(result.success);

        // Verify file was created
        let content = fs.read_file("/workspace/deep/nested/dir/file.txt").await?;
        assert_eq!(content, "nested content");
        Ok(())
    }

    #[tokio::test]
    async fn test_write_empty_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/empty.txt", "content": ""}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("0 lines"));
        assert!(result.output.contains("0 bytes"));

        // Verify file was created
        let content = fs.read_file("/workspace/empty.txt").await?;
        assert_eq!(content, "");
        Ok(())
    }

    #[tokio::test]
    async fn test_write_to_directory_path_returns_error() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.create_dir("/workspace/subdir").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/subdir", "content": "content"}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("is a directory"));
        Ok(())
    }

    #[tokio::test]
    async fn test_write_content_with_special_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content = "特殊字符\néàü\n🎉emoji\ntab\there";

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/special.txt", "content": content}),
            )
            .await?;

        assert!(result.success);

        // Verify content preserved
        let read_content = fs.read_file("/workspace/special.txt").await?;
        assert_eq!(read_content, content);
        Ok(())
    }

    #[tokio::test]
    async fn test_write_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), "write");
        assert_eq!(tool.tier(), ToolTier::Confirm);
        assert!(tool.description().contains("Write"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("content").is_some());
    }

    #[tokio::test]
    async fn test_write_invalid_input_missing_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required path field
        let result = tool
            .execute(&tool_ctx(), json!({"content": "some content"}))
            .await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_write_invalid_input_missing_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required content field
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_write_large_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create content with 1000 lines
        let content: String = (1..=1000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/large.txt", "content": content}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("1000 lines"));

        // Verify content
        let read_content = fs.read_file("/workspace/large.txt").await?;
        assert_eq!(read_content, content);
        Ok(())
    }
}
