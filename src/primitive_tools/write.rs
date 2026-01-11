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
