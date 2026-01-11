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
