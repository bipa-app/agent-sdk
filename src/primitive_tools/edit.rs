use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for editing files via string replacement
pub struct EditTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> EditTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct EditInput {
    /// Path to the file to edit
    path: String,
    /// String to find and replace
    old_string: String,
    /// Replacement string
    new_string: String,
    /// Replace all occurrences (default: false)
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for EditTool<E> {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Edit a file by replacing a string. The old_string must match exactly and uniquely (unless replace_all is true)."
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
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of requiring unique match. Default: false"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: EditInput =
            serde_json::from_value(input).context("Invalid input for edit tool")?;

        let path = self.ctx.environment.resolve_path(&input.path);

        // Check capabilities
        if !self.ctx.capabilities.can_write(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot edit '{path}'"
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
                "'{path}' is a directory, cannot edit"
            )));
        }

        // Read current content
        let content = self
            .ctx
            .environment
            .read_file(&path)
            .await
            .context("Failed to read file")?;

        // Count occurrences
        let count = content.matches(&input.old_string).count();

        if count == 0 {
            return Ok(ToolResult::error(format!(
                "String not found in '{}': '{}'",
                path,
                truncate_string(&input.old_string, 100)
            )));
        }

        if count > 1 && !input.replace_all {
            return Ok(ToolResult::error(format!(
                "Found {count} occurrences of the string in '{path}'. Use replace_all: true to replace all, or provide a more specific string."
            )));
        }

        // Perform replacement
        let new_content = if input.replace_all {
            content.replace(&input.old_string, &input.new_string)
        } else {
            content.replacen(&input.old_string, &input.new_string, 1)
        };

        // Write back
        self.ctx
            .environment
            .write_file(&path, &new_content)
            .await
            .context("Failed to write file")?;

        let replacements = if input.replace_all { count } else { 1 };
        Ok(ToolResult::success(format!(
            "Successfully replaced {replacements} occurrence(s) in '{path}'"
        )))
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
