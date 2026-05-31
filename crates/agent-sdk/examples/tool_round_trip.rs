//! Tool round-trip example.
//!
//! Demonstrates a full tool round-trip: the LLM decides to call a tool, the SDK
//! executes it, feeds the result back, and the LLM produces a final answer that
//! incorporates the tool output. We register a single `get_weather` tool and ask
//! a question that forces the model to use it.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example tool_round_trip
//! ```

use std::sync::Arc;

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, DynamicToolName, EventStore, InMemoryEventStore,
    ThreadId, Tool, ToolContext, ToolRegistry, ToolResult, ToolTier, builder,
    providers::AnthropicProvider,
};
use serde_json::{Value, json};

/// A pretend weather service. In a real app this would hit an HTTP API.
struct WeatherTool;

impl Tool<()> for WeatherTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("get_weather")
    }

    fn display_name(&self) -> &'static str {
        "Weather"
    }

    fn description(&self) -> &'static str {
        "Get the current weather for a city. Always call this before answering weather questions."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Read-only: no confirmation required.
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
        let city = input
            .get("city")
            .and_then(Value::as_str)
            .unwrap_or("Unknown");
        Ok(ToolResult::success(format!(
            "{city}: 18°C, light rain, wind 12 km/h"
        )))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable must be set"))?;

    let mut tools = ToolRegistry::new();
    tools.register(WeatherTool);

    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .tools(tools)
        .event_store(event_store.clone())
        .build();

    let thread_id = ThreadId::new();

    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text(
            "What's the weather in Lisbon right now? Should I take an umbrella?".to_string(),
        ),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let _ = final_state.await?;

    // Walk the persisted events to show each leg of the round-trip.
    for envelope in event_store.get_events(&thread_id).await? {
        match envelope.event {
            AgentEvent::ToolCallStart { name, input, .. } => {
                println!("→ tool call: {name}({input})");
            }
            AgentEvent::ToolCallEnd { name, result, .. } => {
                println!("← tool result: {name} => {}", result.output);
            }
            AgentEvent::Text { text, .. } => {
                println!("\nAgent: {text}");
            }
            AgentEvent::Done { total_turns, .. } => {
                println!("\n(completed in {total_turns} turns)");
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
