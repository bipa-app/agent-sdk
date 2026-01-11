//! Custom tool implementation example.
//!
//! This example shows how to implement a custom tool that the agent can use.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example custom_tool
//! ```

use agent_sdk::{
    AgentEvent, ThreadId, Tool, ToolContext, ToolRegistry, ToolResult, ToolTier, builder,
    providers::AnthropicProvider,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

/// A simple calculator tool that can add two numbers.
struct CalculatorTool;

#[async_trait]
impl Tool<()> for CalculatorTool {
    fn name(&self) -> &'static str {
        "calculator"
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

    fn tier(&self) -> ToolTier {
        // Observe tier means no confirmation needed
        ToolTier::Observe
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

#[async_trait]
impl Tool<()> for CurrentTimeTool {
    fn name(&self) -> &'static str {
        "current_time"
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

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
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
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Create a tool registry and register our custom tools
    let mut tools = ToolRegistry::new();
    tools.register(CalculatorTool);
    tools.register(CurrentTimeTool);

    println!("Registered {} tools:", tools.len());
    for tool in tools.all() {
        println!("  - {}: {}", tool.name(), tool.description());
    }
    println!();

    // Build the agent with our tools
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .tools(tools)
        .build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    // Ask the agent something that requires using our tools
    let mut events = agent.run(
        thread_id,
        "What is 42 + 17? Also, what time is it right now?".to_string(),
        tool_ctx,
    );

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::ToolCallStart { name, input, .. } => {
                println!("[Tool] Calling {name} with {input}");
            }
            AgentEvent::ToolCallEnd { name, result, .. } => {
                println!("[Tool] {name} returned: {}", result.output);
            }
            AgentEvent::Text { text } => {
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
