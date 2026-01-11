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
