//! Typed tool-argument validation example.
//!
//! Demonstrates the [`TypedTool`] surface: instead of receiving a raw
//! `serde_json::Value`, the tool declares a `Serialize`/`Deserialize`
//! [`Input`](agent_sdk::TypedTool::Input) struct. The SDK deserializes the
//! model-emitted arguments into that type **before** `execute` runs, and on a
//! mismatch returns a structured validation-error `tool_result` so the model
//! can self-correct — `execute` is never reached with invalid arguments.
//!
//! Contrast with `tool_round_trip.rs`, which uses the untyped [`Tool`] trait
//! and pulls fields out of a `Value` by hand.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example typed_tool
//! ```

use std::sync::Arc;

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore, ThreadId,
    ToolContext, ToolRegistry, ToolResult, ToolTier, TypedTool, builder,
    providers::AnthropicProvider,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Strongly-typed arguments for the weather tool. The model's JSON arguments
/// are deserialized into this before `execute` is called.
#[derive(Debug, Serialize, Deserialize)]
struct WeatherArgs {
    /// City name, e.g. "Lisbon".
    city: String,
    /// Temperature unit; defaults to Celsius when the model omits it.
    #[serde(default)]
    unit: Unit,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Unit {
    #[default]
    Celsius,
    Fahrenheit,
}

/// A pretend weather service. In a real app this would hit an HTTP API.
struct WeatherTool;

impl TypedTool<()> for WeatherTool {
    type Input = WeatherArgs;

    fn name(&self) -> &'static str {
        "get_weather"
    }

    fn display_name(&self) -> &'static str {
        "Weather"
    }

    fn description(&self) -> &'static str {
        "Get the current weather for a city. Always call this before answering weather questions."
    }

    // The schema stays hand-written JSON: it is NOT auto-derived from
    // `WeatherArgs`. That keeps the provider-facing contract explicit.
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)"
                }
            },
            "required": ["city"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Read-only: no confirmation required.
        ToolTier::Observe
    }

    // `execute` receives the already-validated, typed input — no `Value`
    // poking, no `unwrap_or`, no manual type checks.
    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        input: WeatherArgs,
    ) -> anyhow::Result<ToolResult> {
        let (temp, unit) = match input.unit {
            Unit::Celsius => (18, "°C"),
            Unit::Fahrenheit => (64, "°F"),
        };
        Ok(ToolResult::success(format!(
            "{}: {temp}{unit}, light rain, wind 12 km/h",
            input.city
        )))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable must be set"))?;

    let mut tools = ToolRegistry::new();
    // `register_typed` wraps the tool so model args are validated before
    // `execute`. A malformed call becomes a self-correction `tool_result`.
    tools.register_typed(WeatherTool);

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
