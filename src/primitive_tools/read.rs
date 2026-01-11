use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

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
    /// Path to the file to read
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
        let limit = input.limit.unwrap_or(lines.len());

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
