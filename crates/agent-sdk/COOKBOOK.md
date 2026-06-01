# Agent SDK Cookbook

Task-oriented recipes for the common building blocks. Every recipe maps to a
runnable example under [`examples/`](./examples) — run any of them with
`cargo run --example <name>` (add `--features <feat>` where noted). The
[Quickstart](#quickstart) is the documented happy path; reach for the rest as
you need them.

| Recipe | Example | Feature |
|--------|---------|---------|
| [Quickstart — `ask()`](#quickstart) | `basic_agent` | — |
| [Tools (untyped)](#tools-untyped) | `tool_round_trip`, `custom_tool` | — |
| [Typed tools (`TypedTool`)](#typed-tools) | `typed_tool` | — |
| [Structured output](#structured-output) | `structured_output` | — |
| [Streaming](#streaming) | `streaming` | — |
| [MCP — local (stdio)](#mcp-local-stdio) | `mcp_filesystem` | `mcp` |
| [MCP — remote (HTTP)](#mcp-remote-http) | `mcp_http_remote` | `mcp` |
| [Human-in-the-loop (HITL)](#human-in-the-loop) | `custom_hooks` | — |
| [Durable serving](#durable-serving) | `server_turn_summary` | — |
| [OpenTelemetry](#opentelemetry) | `otel` | `otel` |

## Quickstart

The 30-second path: build an agent, ask a question, read the answer. `ask()`
builds the tool context + cancellation token for you.

```rust,no_run
use agent_sdk::prelude::*;
use agent_sdk::ThreadId;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = builder::<()>()
        .provider(AnthropicProvider::from_env()) // reads ANTHROPIC_API_KEY
        .build();

    let answer = agent.ask(ThreadId::new(), "What is the capital of France?").await?;
    println!("{answer}");
    Ok(())
}
```

When you need application context, a confirmation flow, explicit cancellation,
or the raw run state, drop down to `run()` (see [HITL](#human-in-the-loop)).

## Tools (untyped)

The base [`Tool`] trait receives the model's arguments as a `serde_json::Value`
and returns a [`ToolResult`]. Register it on a [`ToolRegistry`] and hand it to
the builder.

```rust,no_run
use agent_sdk::prelude::*;
use serde_json::{json, Value};

struct Adder;
impl Tool<()> for Adder {
    type Name = DynamicToolName;
    fn name(&self) -> DynamicToolName { DynamicToolName::new("add") }
    fn description(&self) -> &'static str { "Add two integers a + b." }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]})
    }
    fn tier(&self) -> ToolTier { ToolTier::Observe }
    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
        let a = input["a"].as_i64().unwrap_or_default();
        let b = input["b"].as_i64().unwrap_or_default();
        Ok(ToolResult::success((a + b).to_string()))
    }
}

# fn wire() {
let mut tools = ToolRegistry::new();
tools.register(Adder);
# }
```

Run it: `cargo run --example tool_round_trip`.

## Typed tools

[`TypedTool`] removes the `Value`-poking: declare a `Deserialize` input type and
the SDK deserializes + validates the model's arguments **before** `execute`
runs. A malformed call becomes a structured self-correction `tool_result` — your
`execute` never sees invalid input.

```rust,no_run
use agent_sdk::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct WeatherArgs { city: String }

struct Weather;
impl TypedTool<()> for Weather {
    type Input = WeatherArgs;
    fn name(&self) -> &'static str { "get_weather" }
    fn description(&self) -> &'static str { "Get the current weather for a city." }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})
    }
    fn tier(&self) -> ToolTier { ToolTier::Observe }
    async fn execute(&self, _ctx: &ToolContext<()>, input: WeatherArgs) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(format!("{}: 18°C, light rain", input.city)))
    }
}

# fn wire() {
let mut tools = ToolRegistry::new();
tools.register_typed(Weather); // validation wrapper installed here
# }
```

Run it: `cargo run --example typed_tool`.

## Structured output

Constrain the model's final answer to a JSON Schema with `run_structured`. The
runner validates the output and bounded-re-prompts on mismatch before failing
with a typed error; the returned value is guaranteed schema-valid. Works on any
provider — native JSON mode (OpenAI/Gemini) or a forced-tool fallback
(Anthropic) is selected automatically.

```rust,no_run
use agent_sdk::llm::{ChatRequest, Message, ResponseFormat};
use agent_sdk::{run_structured, StructuredConfig};
use agent_sdk::providers::AnthropicProvider;
use serde_json::json;

# async fn run() -> anyhow::Result<()> {
let schema = json!({
    "type": "object",
    "properties": { "name": {"type":"string"}, "age": {"type":"integer"} },
    "required": ["name", "age"]
});

let request = ChatRequest::new(
    "Extract the person described by the user.",
    vec![Message::user("Ada Lovelace is 36.")],
)
.with_response_format(ResponseFormat::new("person", schema));

let out = run_structured(&AnthropicProvider::from_env(), request, StructuredConfig::default()).await?;
println!("schema-valid value: {} (after {} re-prompts)", out.value, out.retries);
# Ok(())
# }
```

Run it (offline, stub provider): `cargo run --example structured_output`.

## Streaming

Wrap the configured [`EventStore`] to observe every [`AgentEvent`] as the loop
emits it — the agent writes an `AgentEvent::TextDelta` per streamed chunk, so
you can print tokens as they arrive.

```bash
cargo run --example streaming
```

The example's `PrintingEventStore` delegates persistence to an inner store and
flushes each `TextDelta` to stdout. The same pattern drives a websocket/SSE UI.

## MCP — local (stdio)

With the `mcp` feature, connect to a Model Context Protocol server spawned as a
local subprocess and expose its tools to the agent.

```bash
cargo run --example mcp_filesystem --features mcp
```

## MCP — remote (HTTP)

The broadened MCP support (Phase 13·C) adds the **streamable-HTTP / SSE**
transport for hosted MCP servers — a single HTTPS endpoint with OAuth/bearer
auth, protocol-revision negotiation, and resources/prompts discovery.

```rust,no_run
use agent_sdk::mcp::{McpAuth, McpClient, StreamableHttpTransport};
use std::sync::Arc;

# async fn run() -> anyhow::Result<()> {
let transport = StreamableHttpTransport::new("https://example.com/mcp", McpAuth::Bearer("token".into()))?;
let client = Arc::new(McpClient::new(transport, "remote".to_string()).await?);

if let Some(v) = client.protocol_version() { println!("MCP revision: {v}"); }
for tool in client.list_tools().await? { println!("tool: {}", tool.name); }
if client.supports_resources() {
    for r in client.list_resources().await? { println!("resource: {}", r.uri); }
}
# Ok(())
# }
```

Run it: `MCP_URL=… MCP_TOKEN=… cargo run --example mcp_http_remote --features mcp`.

## Human-in-the-loop

Gate sensitive tool calls behind a confirmation. Tools declare a
[`ToolTier`]; `ToolTier::Confirm` tools pause the loop with a yield, and you
resume with the user's decision via [`AgentHooks`] / the
[`ToolDecision`](crate::ToolDecision) flow on the `run()` path.

```bash
cargo run --example custom_hooks
```

`custom_hooks` shows intercepting a `send_email` tool, requiring approval, and
either allowing or denying the call — the building block for an approval UI.

## Durable serving

For server deployments, the SDK runs **one turn at a time** and hands back an
authoritative [`TurnOutcome`](crate::advanced::TurnOutcome) /
`TurnSummary` plus a durable continuation envelope. The orchestrator (e.g. the
`agent-service-host` gRPC host) owns tool-task dispatch and persistence, so a
crash mid-turn resumes exactly where it left off.

```bash
cargo run --example server_turn_summary
```

This is the "Temporal-grade backend" surface: `ToolRuntime::External` +
`strict_durability: true` yields a `PendingToolCalls` outcome the host commits
durably before executing tools. See `crate::advanced` for the contract types.

## OpenTelemetry

With the `otel` feature the agent loop emits spans for invocations, turns, LLM
requests, tool execution, subagents, MCP ops, and compaction. Provide an
`ObservabilityStore` to control whether GenAI payloads are inlined, stored
externally, or omitted.

```bash
cargo run --example otel --features otel
```

See the [README](../../README.md#opentelemetry) for the full wiring and the
local Langfuse stack.

[`Tool`]: https://docs.rs/agent-sdk/latest/agent_sdk/trait.Tool.html
[`TypedTool`]: https://docs.rs/agent-sdk/latest/agent_sdk/trait.TypedTool.html
[`ToolResult`]: https://docs.rs/agent-sdk/latest/agent_sdk/struct.ToolResult.html
[`ToolRegistry`]: https://docs.rs/agent-sdk/latest/agent_sdk/struct.ToolRegistry.html
[`ToolTier`]: https://docs.rs/agent-sdk/latest/agent_sdk/enum.ToolTier.html
[`EventStore`]: https://docs.rs/agent-sdk/latest/agent_sdk/trait.EventStore.html
[`AgentEvent`]: https://docs.rs/agent-sdk/latest/agent_sdk/enum.AgentEvent.html
[`AgentHooks`]: https://docs.rs/agent-sdk/latest/agent_sdk/trait.AgentHooks.html
