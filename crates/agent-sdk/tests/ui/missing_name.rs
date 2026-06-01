//! `#[derive(Tool)]` must reject a missing required `name` attribute.

use agent_sdk::{ToolContext, ToolLogic, ToolResult};
use serde_json::Value;

#[derive(agent_sdk::Tool)]
#[tool(description = "no name provided")]
struct BadTool;

impl ToolLogic<()> for BadTool {
    type Input = Value;

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

fn main() {}
