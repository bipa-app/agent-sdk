//! Phase 13 · E — macro ergonomics equivalence tests.
//!
//! The contract this card promises: the macros are *additive sugar*. A tool
//! defined with `#[derive(Tool)]` / `#[derive(TypedTool)]` / `tool!` must
//! behave **identically** to the equivalent hand-written tool — same metadata
//! (name/description/schema/tier), same `execute` output, same typed-arg
//! validation + self-correction, and the same `Value` back-compat passthrough.
//!
//! These tests drive the tools directly through the [`ToolRegistry`] /
//! `ErasedTool` boundary (no provider needed) so the equivalence is asserted on
//! the exact runtime surface the agent loop uses.

use agent_sdk::{
    SimpleTool, ToolContext, ToolLogic, ToolName, ToolRegistry, ToolResult, ToolTier, TypedTool,
    tool,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ════════════════════════════════════════════════════════════════════
// #[derive(Tool)] vs hand-written SimpleTool
// ════════════════════════════════════════════════════════════════════

/// Hand-written baseline (untyped `Value`-in `SimpleTool`).
struct HandWrittenWeather;

impl SimpleTool<()> for HandWrittenWeather {
    fn name(&self) -> &'static str {
        "get_weather"
    }

    fn display_name(&self) -> &'static str {
        "Weather"
    }

    fn description(&self) -> &'static str {
        "Get the current weather for a city"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let city = input["city"].as_str().unwrap_or("Unknown");
        Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
    }
}

/// Derived equivalent — only the `execute` body is hand-written; metadata comes
/// from the `#[tool(...)]` attribute.
#[derive(agent_sdk::Tool)]
#[tool(
    name = "get_weather",
    display_name = "Weather",
    description = "Get the current weather for a city",
    schema = json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
)]
struct DerivedWeather;

impl ToolLogic<()> for DerivedWeather {
    type Input = Value;

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let city = input["city"].as_str().unwrap_or("Unknown");
        Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
    }
}

#[tokio::test]
async fn derived_tool_metadata_matches_hand_written() -> Result<()> {
    let mut hand = ToolRegistry::<()>::new();
    hand.register_simple(HandWrittenWeather);
    let mut derived = ToolRegistry::<()>::new();
    derived.register_simple(DerivedWeather);

    let h = hand.get("get_weather").context("hand tool registered")?;
    let d = derived
        .get("get_weather")
        .context("derived tool registered")?;

    assert_eq!(d.name_str(), h.name_str());
    assert_eq!(d.display_name(), h.display_name());
    assert_eq!(d.description(), h.description());
    assert_eq!(d.input_schema(), h.input_schema());
    assert_eq!(d.tier(), h.tier());
    Ok(())
}

#[tokio::test]
async fn derived_tool_execute_matches_hand_written() -> Result<()> {
    let ctx = ToolContext::new(());
    let hand = HandWrittenWeather;
    let derived = DerivedWeather;

    let args = json!({ "city": "Lisbon" });
    let h = SimpleTool::execute(&hand, &ctx, args.clone()).await?;
    let d = SimpleTool::execute(&derived, &ctx, args).await?;

    assert_eq!(d.success, h.success);
    assert_eq!(d.output, h.output);
    assert_eq!(d.output, "Weather in Lisbon: Sunny");
    Ok(())
}

#[tokio::test]
async fn derived_tool_defaults_schema_and_tier() -> Result<()> {
    // No `schema`, no `tier`, no `display_name` → trait defaults.
    #[derive(agent_sdk::Tool)]
    #[tool(name = "ping", description = "Ping")]
    struct Ping;

    impl ToolLogic<()> for Ping {
        type Input = Value;

        async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
            Ok(ToolResult::success("pong"))
        }
    }

    let ping = Ping;
    assert_eq!(SimpleTool::input_schema(&ping), json!({ "type": "object" }));
    assert_eq!(SimpleTool::tier(&ping), ToolTier::Observe);
    assert_eq!(SimpleTool::display_name(&ping), "");
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// #[derive(TypedTool)] vs hand-written TypedTool (incl. validation)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize)]
struct GreetArgs {
    name: String,
    greeting: String,
}

struct HandWrittenGreet;

impl TypedTool<()> for HandWrittenGreet {
    type Input = GreetArgs;

    fn name(&self) -> &'static str {
        "greet"
    }

    fn description(&self) -> &'static str {
        "Greet someone by name"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "greeting": { "type": "string" }
            },
            "required": ["name", "greeting"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: GreetArgs) -> Result<ToolResult> {
        Ok(ToolResult::success(format!(
            "{}, {}!",
            input.greeting, input.name
        )))
    }
}

#[derive(agent_sdk::TypedTool)]
#[tool(
    name = "greet",
    description = "Greet someone by name",
    input = GreetArgs,
    schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "greeting": { "type": "string" }
        },
        "required": ["name", "greeting"]
    }),
)]
struct DerivedGreet;

impl ToolLogic<()> for DerivedGreet {
    type Input = GreetArgs;

    async fn execute(&self, _ctx: &ToolContext<()>, input: GreetArgs) -> Result<ToolResult> {
        Ok(ToolResult::success(format!(
            "{}, {}!",
            input.greeting, input.name
        )))
    }
}

#[tokio::test]
async fn derived_typed_tool_metadata_matches_hand_written() -> Result<()> {
    assert_eq!(
        TypedTool::name(&DerivedGreet),
        TypedTool::name(&HandWrittenGreet)
    );
    assert_eq!(
        TypedTool::description(&DerivedGreet),
        TypedTool::description(&HandWrittenGreet)
    );
    assert_eq!(
        TypedTool::input_schema(&DerivedGreet),
        TypedTool::input_schema(&HandWrittenGreet)
    );
    Ok(())
}

#[tokio::test]
async fn derived_typed_tool_validates_args_through_registry() -> Result<()> {
    // Registered via `register_typed`, the derived tool gets the exact same
    // deserialize-then-dispatch (or synthesise-error) adapter as a
    // hand-written `TypedTool`.
    let mut registry = ToolRegistry::<()>::new();
    registry.register_typed(DerivedGreet);
    let tool = registry.get("greet").context("typed tool registered")?;
    let ctx = ToolContext::new(());

    // Happy path: well-formed args reach `execute`.
    let ok = tool
        .execute(&ctx, json!({ "name": "Ada", "greeting": "Hello" }))
        .await?;
    assert!(ok.success);
    assert_eq!(ok.output, "Hello, Ada!");

    // Invalid path: missing `greeting` → structured self-correction error,
    // `execute` is never reached.
    let bad = tool.execute(&ctx, json!({ "name": "Ada" })).await?;
    assert!(!bad.success, "validation failure is an error result");
    assert!(
        bad.output.contains("Invalid arguments for tool `greet`"),
        "must identify the tool: {}",
        bad.output
    );
    assert!(
        bad.output.contains("greeting"),
        "must surface the missing field: {}",
        bad.output
    );
    Ok(())
}

/// Back-compat: a derived `TypedTool` whose `Input = Value` is the identity
/// passthrough — any JSON deserializes, exactly like the untyped baseline.
#[derive(agent_sdk::TypedTool)]
#[tool(name = "echo", description = "Echo any JSON", input = Value)]
struct DerivedEcho;

impl ToolLogic<()> for DerivedEcho {
    type Input = Value;

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        Ok(ToolResult::success(input.to_string()))
    }
}

#[tokio::test]
async fn derived_typed_tool_value_input_is_identity_passthrough() -> Result<()> {
    let mut registry = ToolRegistry::<()>::new();
    registry.register_typed(DerivedEcho);
    let tool = registry.get("echo").context("echo registered")?;
    let ctx = ToolContext::new(());

    // Arbitrary shape — `Value` always "deserializes".
    let res = tool
        .execute(
            &ctx,
            json!({ "anything": [1, 2, 3], "nested": { "ok": true } }),
        )
        .await?;
    assert!(res.success);
    // No `schema` attribute → default object schema.
    assert_eq!(tool.input_schema(), json!({ "type": "object" }));
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// tool! declarative macro
// ════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn declarative_tool_macro_round_trips() -> Result<()> {
    let weather = tool! {
        name: "get_weather",
        description: "Get the current weather for a city",
        schema: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
        |_ctx, input| async move {
            let city = input["city"].as_str().unwrap_or("Unknown");
            Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
        }
    };

    let mut registry = ToolRegistry::<()>::new();
    registry.register_simple(weather);
    let tool = registry
        .get("get_weather")
        .context("inline tool registered")?;
    let ctx = ToolContext::new(());

    let res = tool.execute(&ctx, json!({ "city": "Porto" })).await?;
    assert!(res.success);
    assert_eq!(res.output, "Weather in Porto: Sunny");
    // The inline `schema:` flows through to `input_schema()` unchanged.
    assert_eq!(tool.input_schema()["required"][0], json!("city"));
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// #[derive(ToolName)] vs hand-written ToolName enum
// ════════════════════════════════════════════════════════════════════

/// Hand-written baseline using the documented enum + marker-impl pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HandName {
    ReadFile,
    WriteFile,
}

impl ToolName for HandName {}

/// Derived equivalent — one derive, no serde boilerplate, no marker impl.
#[derive(Clone, Copy, Debug, PartialEq, Eq, agent_sdk::ToolName)]
enum DerivedName {
    ReadFile,
    WriteFile,
}

#[test]
fn derived_tool_name_serializes_identically() -> Result<()> {
    // Same wire form as the hand-written `#[serde(rename_all = "snake_case")]`.
    let hand = serde_json::to_string(&HandName::ReadFile)?;
    let derived = serde_json::to_string(&DerivedName::ReadFile)?;
    assert_eq!(hand, derived);
    assert_eq!(derived, "\"read_file\"");

    // Round-trips back through the same string tags.
    let parsed: DerivedName = serde_json::from_str("\"write_file\"")?;
    assert_eq!(parsed, DerivedName::WriteFile);
    Ok(())
}

#[test]
fn derived_tool_name_is_usable_as_tool_name() {
    // It satisfies the `ToolName` bound (compile-time proof via a generic fn).
    fn assert_tool_name<N: ToolName>() {}
    assert_tool_name::<DerivedName>();
}

// ════════════════════════════════════════════════════════════════════
// schema = "derive" (opt-in schemars sugar over the 13·B JSON baseline)
// ════════════════════════════════════════════════════════════════════

#[cfg(feature = "macros-schema")]
mod schema_derive {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct SearchArgs {
        /// What to search for.
        query: String,
        /// Max results to return.
        #[serde(default)]
        limit: u32,
    }

    /// Same tool, schema auto-derived from `SearchArgs` instead of hand-written.
    #[derive(agent_sdk::TypedTool)]
    #[tool(
        name = "search",
        description = "Search the corpus",
        input = SearchArgs,
        schema = "derive"
    )]
    struct SearchTool;

    impl ToolLogic<()> for SearchTool {
        type Input = SearchArgs;

        async fn execute(&self, _ctx: &ToolContext<()>, input: SearchArgs) -> Result<ToolResult> {
            Ok(ToolResult::success(format!(
                "searched {} (limit {})",
                input.query, input.limit
            )))
        }
    }

    #[tokio::test]
    async fn schema_is_derived_from_input_type() -> Result<()> {
        let mut registry = ToolRegistry::<()>::new();
        registry.register_typed(SearchTool);
        let tool = registry.get("search").context("search registered")?;

        let schema = tool.input_schema();
        // The derived schema describes the typed `Input` — object with the two
        // fields, `query` required. This is real schemars output, not the
        // `{"type":"object"}` fallback.
        assert_eq!(schema["type"], json!("object"));
        assert!(
            schema["properties"]["query"].is_object(),
            "derived schema must describe `query`: {schema}"
        );
        assert!(
            schema["properties"]["limit"].is_object(),
            "derived schema must describe `limit`: {schema}"
        );
        let required = schema["required"]
            .as_array()
            .context("derived schema must have a `required` array")?;
        assert!(
            required.iter().any(|v| v == "query"),
            "`query` must be required (no serde default): {schema}"
        );

        // And the tool still validates + executes through the typed adapter.
        let ctx = ToolContext::new(());
        let ok = tool.execute(&ctx, json!({ "query": "rust" })).await?;
        assert!(ok.success);
        assert_eq!(ok.output, "searched rust (limit 0)");
        Ok(())
    }
}
