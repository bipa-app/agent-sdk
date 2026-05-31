//! Ergonomic procedural and declarative macros for the Agent SDK.
//!
//! This crate is the **macro / ergonomics layer** (Phase 13·E) sitting on top
//! of the hand-written tool traits in `agent-sdk-tools` (the untyped
//! `Tool`/`SimpleTool` baseline and the typed `TypedTool` API from 13·B). It
//! turns today's boilerplate — hand-written `name`/`description`/`input_schema`
//! methods, `ToolName` enums, and inline tool scaffolding — into derives and a
//! declarative macro, **without** changing or replacing the traits themselves.
//! Everything here is additive sugar: hand-written tools keep compiling.
//!
//! You almost never depend on this crate directly. The macros are re-exported
//! from the `agent-sdk` façade (and its prelude), so the generated code refers
//! to `::agent_sdk::…` paths. The examples below are written as a user would
//! write them, against `agent_sdk`.
//!
//! # `#[derive(Tool)]`
//!
//! Derives a `SimpleTool` impl (the untyped, `Value`-in baseline) from
//! struct-level `#[tool(...)]` attributes. The derive wires up `name` /
//! `display_name` / `description` / `input_schema` / `tier`; you supply the
//! behaviour by implementing `agent_sdk::ToolLogic` (with `type Input = Value`).
//!
//! ```ignore
//! use agent_sdk::{Tool, ToolContext, ToolLogic, ToolResult};
//! use serde_json::{json, Value};
//!
//! #[derive(Tool)]
//! #[tool(
//!     name = "get_weather",
//!     description = "Get the current weather for a city",
//!     schema = json!({
//!         "type": "object",
//!         "properties": { "city": { "type": "string" } },
//!         "required": ["city"]
//!     }),
//! )]
//! struct WeatherTool;
//!
//! impl ToolLogic<()> for WeatherTool {
//!     type Input = Value;
//!     async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
//!         let city = input["city"].as_str().unwrap_or("Unknown");
//!         Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
//!     }
//! }
//! ```
//!
//! # `#[derive(TypedTool)]`
//!
//! Derives a `TypedTool` impl with a typed `Input`. The `input_schema` can
//! either be hand-provided (`schema = <expr>`, matching 13·B's baseline) or
//! auto-derived from the `Input` type via `schemars`
//! (`schema = "derive"`, requires the `schema-derive` feature). Your
//! `ToolLogic::execute` receives the already-validated, typed `Input`.
//!
//! ```ignore
//! use agent_sdk::{TypedTool, ToolContext, ToolLogic, ToolResult};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, schemars::JsonSchema)]
//! struct WeatherArgs { city: String }
//!
//! #[derive(TypedTool)]
//! #[tool(name = "get_weather", description = "Weather", input = WeatherArgs, schema = "derive")]
//! struct WeatherTool;
//!
//! impl ToolLogic<()> for WeatherTool {
//!     type Input = WeatherArgs;
//!     async fn execute(&self, _ctx: &ToolContext<()>, input: WeatherArgs) -> anyhow::Result<ToolResult> {
//!         Ok(ToolResult::success(format!("Weather in {}: Sunny", input.city)))
//!     }
//! }
//! ```
//!
//! # `#[derive(ToolName)]`
//!
//! Removes the `ToolName` marker-impl boilerplate: it derives the `Serialize`
//! / `Deserialize` / `ToolName` trio (with `snake_case` renaming by default)
//! for a plain enum, so a strongly-typed tool-name enum is a single derive.
//!
//! # `tool!`
//!
//! A declarative macro for quick *inline* tools — handy in examples, tests, and
//! one-off scripts where defining a named struct + impl is overkill.

mod attr;
mod tool_derive;
mod tool_name_derive;
mod typed_tool_derive;

use proc_macro::TokenStream;

/// Derive an `agent_sdk::SimpleTool` impl from struct-level `#[tool(...)]`
/// attributes.
///
/// See the [crate-level docs](crate) for the full attribute list and an
/// example. Required: `name` and `description`. Optional: `display_name`,
/// `tier`, `schema` (a `serde_json::Value` expression; defaults to
/// `{"type":"object"}`), and `context` (the `Ctx` type; defaults to `()`).
///
/// The struct must implement `agent_sdk::ToolLogic<Ctx>` with
/// `type Input = serde_json::Value` to supply the `execute` body.
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    tool_derive::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive an `agent_sdk::TypedTool` impl (typed `Input` + runtime validation)
/// from struct-level `#[tool(...)]` attributes.
///
/// See the [crate-level docs](crate). Required: `name`, `description`, and
/// `input` (the typed `Input` type). The `schema` attribute is either a
/// `serde_json::Value` expression (hand-provided, the 13·B baseline) or the
/// literal string `"derive"` to auto-generate the schema from `Input` via
/// `schemars` (requires this crate's `schema-derive` feature).
///
/// The struct must implement `agent_sdk::ToolLogic<Ctx>` with
/// `type Input = <the typed input>` to supply the `execute` body.
#[proc_macro_derive(TypedTool, attributes(tool))]
pub fn derive_typed_tool(input: TokenStream) -> TokenStream {
    typed_tool_derive::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive `agent_sdk::ToolName` (plus the `Serialize` / `Deserialize` it
/// requires) for an enum, removing the marker-impl boilerplate.
///
/// By default variants serialize as `snake_case` (matching the SDK's built-in
/// `PrimitiveToolName`). Override with `#[tool_name(rename_all = "...")]`.
#[proc_macro_derive(ToolName, attributes(tool_name))]
pub fn derive_tool_name(input: TokenStream) -> TokenStream {
    tool_name_derive::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
