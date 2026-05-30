//! Custom tool implementation example.
//!
//! This example shows how to implement a custom tool that the agent can use.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example custom_tool
//! ```
//!
//! To see debug logs from the SDK:
//! ```bash
//! RUST_LOG=agent_sdk=debug ANTHROPIC_API_KEY=your_key cargo run --example custom_tool
//! ```

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore, SimpleTool,
    ThreadId, ToolContext, ToolRegistry, ToolResult, builder, providers::AnthropicProvider,
};
use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

/// A simple calculator tool that can add two numbers.
///
/// Implemented via [`SimpleTool`], so it needs no `ToolName` type — just a
/// `&str` name.
struct CalculatorTool;

impl SimpleTool<()> for CalculatorTool {
    fn name(&self) -> &'static str {
        "calculator"
    }

    fn display_name(&self) -> &'static str {
        "Calculator"
    }

    fn description(&self) -> &'static str {
        "Add two numbers together. Use this when you need to perform addition."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number to add"
                },
                "b": {
                    "type": "number",
                    "description": "Second number to add"
                }
            },
            "required": ["a", "b"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let a = input
            .get("a")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'a' parameter"))?;

        let b = input
            .get("b")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'b' parameter"))?;

        let result = a + b;

        Ok(ToolResult::success(format!("{a} + {b} = {result}")))
    }
}

/// A tool that gets the current time.
struct CurrentTimeTool;

impl SimpleTool<()> for CurrentTimeTool {
    fn name(&self) -> &'static str {
        "current_time"
    }

    fn display_name(&self) -> &'static str {
        "Current Time"
    }

    fn description(&self) -> &'static str {
        "Get the current date and time in UTC."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
        let now = time::OffsetDateTime::now_utc();
        let formatted = now
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "unknown".to_string());

        Ok(ToolResult::success(format!(
            "Current UTC time: {formatted}"
        )))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Create a tool registry and register our custom tools. `register_simple`
    // wraps a `SimpleTool` so no `ToolName` type is required.
    let mut tools = ToolRegistry::new();
    tools.register_simple(CalculatorTool);
    tools.register_simple(CurrentTimeTool);

    println!("Registered {} tools:", tools.len());
    for tool in tools.all() {
        println!("  - {}: {}", tool.name_str(), tool.description());
    }
    println!();

    // Build the agent with our tools. We keep an explicit event store here so
    // we can inspect the tool-call events after the run.
    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::try_from_env()?)
        .tools(tools)
        .event_store(event_store.clone())
        .build();

    let thread_id = ThreadId::new();

    // Ask the agent something that requires using our tools
    let _ = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("What is 42 + 17? Also, what time is it right now?".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    for envelope in event_store.get_events(&thread_id).await? {
        match envelope.event {
            AgentEvent::ToolCallStart { name, input, .. } => {
                println!("[Tool] Calling {name} with {input}");
            }
            AgentEvent::ToolCallEnd { name, result, .. } => {
                println!("[Tool] {name} returned: {}", result.output);
            }
            AgentEvent::Text {
                message_id: _,
                text,
            } => {
                println!("\nAgent: {text}");
            }
            AgentEvent::Done { total_turns, .. } => {
                println!("\n(Completed in {total_turns} turns)");
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("Error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
