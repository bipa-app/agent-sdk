//! Macro-based tool ergonomics (Phase 13·E).
//!
//! Defines the same kinds of tools as `custom_tool.rs` / `typed_tool.rs`, but
//! with the derive + declarative macros instead of hand-written trait impls:
//!
//! * [`#[derive(Tool)]`](agent_sdk::Tool) — untyped `Value`-in tool; you write
//!   only the `execute` body, the metadata comes from `#[tool(...)]`.
//! * [`#[derive(TypedTool)]`](agent_sdk::TypedTool) — typed `Input` with
//!   runtime validation; the JSON schema is auto-derived from the `Input` type
//!   via `schemars` when built with the `macros-schema` feature.
//! * [`tool!`](agent_sdk::tool) — a quick inline tool, no named struct.
//! * [`#[derive(ToolName)]`](agent_sdk::ToolName) — a strongly-typed tool-name
//!   enum with no serde / marker-impl boilerplate.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example derive_tool
//! # With auto-derived JSON schemas:
//! ANTHROPIC_API_KEY=your_key cargo run --example derive_tool --features macros-schema
//! ```

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore, ThreadId,
    ToolContext, ToolLogic, ToolRegistry, ToolResult, builder, providers::AnthropicProvider, tool,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

// ── #[derive(Tool)] — untyped, Value-in ──────────────────────────────
// The whole tool is a unit struct + the `#[tool(...)]` attribute + an
// inherent `execute`. Compare with `custom_tool.rs`'s ~30-line trait impl.

#[derive(agent_sdk::Tool)]
#[tool(
    name = "calculator",
    display_name = "Calculator",
    description = "Add two numbers together.",
    schema = json!({
        "type": "object",
        "properties": {
            "a": { "type": "number", "description": "First number" },
            "b": { "type": "number", "description": "Second number" }
        },
        "required": ["a", "b"]
    }),
)]
struct CalculatorTool;

impl ToolLogic<()> for CalculatorTool {
    type Input = Value;

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let a = input["a"].as_f64().unwrap_or_default();
        let b = input["b"].as_f64().unwrap_or_default();
        Ok(ToolResult::success(format!("{a} + {b} = {}", a + b)))
    }
}

// ── #[derive(TypedTool)] — typed Input + validation ──────────────────

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "macros-schema", derive(schemars::JsonSchema))]
struct WeatherArgs {
    /// City name, e.g. "Lisbon".
    city: String,
}

#[derive(agent_sdk::TypedTool)]
// With the `macros-schema` feature the schema is generated from `WeatherArgs`
// (`schema = "derive"`); otherwise we provide it by hand. Both spellings are
// first-class — the derived schema is purely opt-in sugar.
#[cfg_attr(
    feature = "macros-schema",
    tool(name = "get_weather", description = "Get the weather for a city", input = WeatherArgs, schema = "derive")
)]
#[cfg_attr(
    not(feature = "macros-schema"),
    tool(
        name = "get_weather",
        description = "Get the weather for a city",
        input = WeatherArgs,
        schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    )
)]
struct WeatherTool;

impl ToolLogic<()> for WeatherTool {
    type Input = WeatherArgs;

    async fn execute(&self, _ctx: &ToolContext<()>, input: WeatherArgs) -> Result<ToolResult> {
        Ok(ToolResult::success(format!(
            "{}: 18°C, light rain",
            input.city
        )))
    }
}

// ── #[derive(ToolName)] — strongly-typed name enum, no boilerplate ────
// One derive replaces `#[derive(Serialize, Deserialize)] #[serde(rename_all =
// "snake_case")] + impl ToolName for ...`.

#[derive(Clone, Copy, Debug, PartialEq, Eq, agent_sdk::ToolName)]
#[allow(dead_code)] // illustrative; not all variants are wired to tools here.
enum MyToolName {
    Calculator,
    GetWeather,
    CurrentTime,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // `#[derive(ToolName)]` gives the enum serde + `ToolName` for free.
    assert_eq!(
        serde_json::to_string(&MyToolName::GetWeather)?,
        "\"get_weather\""
    );

    let mut tools = ToolRegistry::new();
    tools.register_simple(CalculatorTool);
    tools.register_typed(WeatherTool);

    // `tool!` — a one-off inline tool with no named struct.
    tools.register_simple(tool! {
        name: "current_time",
        description: "Get the current UTC time.",
        schema: json!({ "type": "object", "properties": {} }),
        |_ctx, _input| async move {
            let now = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "unknown".to_string());
            Ok(ToolResult::success(format!("Current UTC time: {now}")))
        }
    });

    println!("Registered {} tools:", tools.len());
    for t in tools.all() {
        println!("  - {}: {}", t.name_str(), t.description());
    }
    println!();

    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::try_from_env()?)
        .tools(tools)
        .event_store(event_store.clone())
        .build();

    let thread_id = ThreadId::new();
    let _ = agent
        .run(
            thread_id.clone(),
            AgentInput::Text(
                "What is 42 + 17, what's the weather in Lisbon, and what time is it?".to_string(),
            ),
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
            AgentEvent::Text { text, .. } => println!("\nAgent: {text}"),
            AgentEvent::Done { total_turns, .. } => {
                println!("\n(Completed in {total_turns} turns)");
            }
            AgentEvent::Error { message, .. } => eprintln!("Error: {message}"),
            _ => {}
        }
    }

    Ok(())
}
