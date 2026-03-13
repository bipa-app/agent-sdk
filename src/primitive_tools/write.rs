use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

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
    #[serde(alias = "file_path")]
    path: String,
    content: String,
}

impl<E: Environment + 'static> Tool<()> for WriteTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Write
    }

    fn display_name(&self) -> &'static str {
        "Write File"
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

        if !self.ctx.capabilities.can_write(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot write to '{path}'"
            )));
        }

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

    #[tokio::test]
    async fn writes_new_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/new.txt", "content": "Hello, World!"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("1 lines"));
        assert!(result.output.contains("13 bytes"));

        let content = fs.read_file("/workspace/new.txt").await?;
        assert_eq!(content, "Hello, World!");
        Ok(())
    }

    #[tokio::test]
    async fn overwrites_existing_file() -> anyhow::Result<()> {
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
        let content = fs.read_file("/workspace/existing.txt").await?;
        assert_eq!(content, "new content");
        Ok(())
    }

    #[tokio::test]
    async fn writes_multiline_content() -> anyhow::Result<()> {
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
        let read_content = fs.read_file("/workspace/multi.txt").await?;
        assert_eq!(read_content, content);
        Ok(())
    }

    #[tokio::test]
    async fn errors_on_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::read_only());

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
    async fn errors_on_denied_paths() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
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
    async fn errors_on_directory_target() -> anyhow::Result<()> {
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
    async fn writes_to_nested_directory() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/deep/nested/file.txt", "content": "nested"}),
            )
            .await?;

        assert!(result.success);
        let content = fs.read_file("/workspace/deep/nested/file.txt").await?;
        assert_eq!(content, "nested");
        Ok(())
    }

    #[tokio::test]
    async fn writes_empty_content() -> anyhow::Result<()> {
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
        Ok(())
    }

    #[tokio::test]
    async fn tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), PrimitiveToolName::Write);
        assert_eq!(tool.tier(), ToolTier::Confirm);

        let schema = tool.input_schema();
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("content").is_some());
    }
}
