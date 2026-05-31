# Agent SDK

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A Rust SDK for building AI agents powered by large language models (LLMs). Create agents that can reason, use tools, and take actions through a streaming, event-driven architecture.

Agent SDK is open source under the [MIT License](LICENSE).

## What is an Agent?

An agent is an LLM that can do more than just chat—it can use tools to interact with the world. This SDK provides the infrastructure to:

- Send messages to an LLM and receive streaming responses
- Define tools the LLM can call (APIs, file operations, databases, etc.)
- Execute tool calls and feed results back to the LLM
- Control the agent loop with hooks for logging, security, and approval workflows

## Features

- **Agent Loop** - Core orchestration that handles the LLM conversation and tool execution cycle
- **Provider Agnostic** - Built-in support for Anthropic (Claude), OpenAI, and Google Gemini, plus a trait for custom providers
- **Tool System** - Define tools with JSON schema validation and typed tool names; the LLM decides when to use them
- **Async Tools** - Long-running operations with progress streaming via `AsyncTool` trait
- **Lifecycle Hooks** - Intercept tool calls for logging, user confirmation, rate limiting, or security checks
- **Streaming Events** - Real-time event stream for building responsive UIs
- **Thinking Configuration** - Provider-owned reasoning and thinking controls via `ThinkingConfig`
- **Primitive Tools** - Ready-to-use tools for file operations (Read, Write, Edit, Glob, Grep, Bash, Notebooks)
- **Web Tools** - Web search and URL fetching with SSRF protection
- **Subagents** - Spawn isolated child agents for complex subtasks
- **MCP Support** - Model Context Protocol integration for external tool servers
- **Task Tracking** - Built-in todo system for tracking multi-step tasks
- **User Interaction** - Tools for asking questions and requesting confirmations
- **Security Model** - Capability-based permissions and tool tiers (Observe, Confirm)
- **Yield/Resume Pattern** - Pause agent execution for tool confirmation and resume with user decision
- **Single-Turn Execution** - Run one turn at a time for external orchestration (e.g., message queues)
- **Persistence** - Trait-based storage for conversation history, agent state, and tool execution tracking
- **Context Compaction** - Automatic token management to handle long conversations

## Requirements

- Rust 1.85+ (2024 edition)
- An API key for your chosen LLM provider

## Installation

Add the SDK to your project with Cargo:

```bash
cargo add agent-sdk
cargo add tokio --features rt-multi-thread,macros
cargo add anyhow
```

Or add it directly to your `Cargo.toml`:

```toml
[dependencies]
agent-sdk = "0.8"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"
```

The default build is lightweight — a plain `cargo add agent-sdk` enables
only the `anthropic` provider and the core runtime, with no `sqlx`, `tonic`,
`lapin`, `prost`, or `opentelemetry` dependencies. Everything else is opt-in
through Cargo features, so a minimal consumer never pays for WebSocket, HTML,
or YAML deps it does not use:

| Feature | Enables |
|---------|---------|
| `anthropic` *(default)* | Anthropic (Claude) provider |
| `openai` | OpenAI Chat Completions provider |
| `openai-codex` | OpenAI Codex/Responses provider |
| `gemini` | Google Gemini provider |
| `vertex` | Google Vertex AI provider |
| `cloudflare` | Cloudflare Workers AI provider |
| `web` | Web search + URL fetch tools (pulls in `html2text`) |
| `mcp` | Model Context Protocol client |
| `skills` | Skill/command loading from markdown (pulls in `serde_yaml_ng`) |
| `otel` | OpenTelemetry spans (see below) |

Enable the ones you need, for example:

```toml
[dependencies]
agent-sdk = { version = "0.8", features = ["openai", "gemini", "web", "mcp"] }
```

## OpenTelemetry

Enable the optional `otel` feature to emit OpenTelemetry spans from the agent loop:

```toml
[dependencies]
agent-sdk = { version = "0.8", features = ["otel"] }
opentelemetry = "0.31"
opentelemetry_sdk = { version = "0.31", features = ["rt-tokio"] }
# Add the exporter crate that matches your backend, for example opentelemetry-otlp.
```

Then configure a global tracer provider in your application and optionally provide an `ObservabilityStore` to control whether GenAI payloads are inlined, stored externally, or omitted:

```rust
use agent_sdk::{
    builder,
    observability::{CaptureDecision, CaptureResult, ObservabilityStore, PayloadBundle},
    providers::AnthropicProvider,
};
use async_trait::async_trait;
use opentelemetry::global;
use opentelemetry_sdk::trace::SdkTracerProvider;

struct InlinePayloadStore;

#[async_trait]
impl ObservabilityStore for InlinePayloadStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> anyhow::Result<CaptureResult> {
        Ok(CaptureResult {
            system_instructions: CaptureDecision::Inline,
            input_messages: CaptureDecision::Inline,
            output_messages: CaptureDecision::Inline,
        })
    }
}

let tracer_provider = SdkTracerProvider::builder().build();
global::set_tracer_provider(tracer_provider);

let agent = builder::<()>()
    .provider(AnthropicProvider::sonnet(api_key))
    .observability_store(InlinePayloadStore)
    .build();
```

If you only want spans, you can omit `.observability_store(...)` and just configure the global tracer provider. For a complete runnable example, see `examples/otel.rs`:

```bash
cargo run --example otel --features otel
```

### Local Langfuse stack

A self-contained Langfuse + OTel collector compose stack lives at
`dev/observability/langfuse/`. From this repo:

```bash
docker compose -f dev/observability/langfuse/docker-compose.yml up -d
```

For a downstream consumer (e.g. `bip`) the workspace ships a small CLI
(`agent-sdk-cli`) that materializes the same compose files into the
consumer's working tree:

```bash
cargo install --path crates/agent-sdk-cli   # or --git for remote consumers
agent-sdk local-langfuse init                 # writes ./dev/observability/langfuse/...
agent-sdk doctor                              # checks docker, ports, dest writability
```

See `crates/agent-sdk/docs/observability/LANGFUSE.md` for the full
setup walkthrough.

## Quick Start

Build an agent, run a prompt, and read the reply from the agent's event store.
The agent loop writes every [`AgentEvent`] to the configured `EventStore`; you
read them back once the run completes (or wrap the store to observe them live —
see the [`streaming`](crates/agent-sdk/examples/streaming.rs) example).

```rust
use std::sync::Arc;
use agent_sdk::{
    builder, AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore,
    ThreadId, ToolContext, providers::AnthropicProvider,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Read your API key from the environment.
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    // Build an agent with the Anthropic provider.
    // `::<()>` means "no custom tool context" (see "Custom Context" below).
    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .event_store(event_store.clone())
        .build();

    // Each conversation gets a unique thread id.
    let thread_id = ThreadId::new();

    // Run the agent; await the final run state.
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("What is the capital of France?".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let _ = final_state.await?;

    // Read the persisted events.
    for envelope in event_store.get_events(&thread_id).await? {
        match envelope.event {
            AgentEvent::Text { text, .. } => print!("{text}"),
            AgentEvent::Done { .. } => break,
            AgentEvent::Error { message, .. } => eprintln!("Error: {message}"),
            _ => {} // ToolCallStart, ToolCallEnd, TextDelta, etc.
        }
    }

    Ok(())
}
```

This Quickstart talks to Anthropic and needs `ANTHROPIC_API_KEY`. The same code
is exercised by a **keyless doctest** in the crate root (it swaps in a tiny stub
provider so `cargo test --doc` compiles and runs the flow with no network and no
key); see the "Quick Start" `rustdoc` example in
[`crates/agent-sdk/src/lib.rs`](crates/agent-sdk/src/lib.rs).

### Try it from the CLI

The `agent-sdk` binary can run an agent directly — no code required:

```bash
# Single prompt, streamed to stdout
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk-cli -- run "Explain Rust ownership in two sentences."

# Interactive chat (keeps conversation history for the session)
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk-cli -- chat
```

`agent-sdk run`/`chat` build an Anthropic-backed agent, read the key from
`ANTHROPIC_API_KEY`, and stream the reply as it arrives. The CLI also keeps the
`local-langfuse` and `doctor` subcommands.

## Examples

Clone the repository and run the examples from the repository root. The
first three are the quickstart trio referenced above:

```bash
# Stream the reply to stdout token-by-token
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example streaming

# Full tool round-trip: model calls a tool, gets the result, answers
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example tool_round_trip

# Connect to an external MCP server (stdio) and use its tools
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example mcp_filesystem

# Basic conversation (no tools)
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example basic_agent

# Agent with custom tools
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example custom_tool

# Using lifecycle hooks for logging and rate limiting
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example custom_hooks

# Agent with file operation tools
ANTHROPIC_API_KEY=your_key cargo run -p agent-sdk --example with_primitive_tools

# OpenTelemetry instrumentation (self-contained, no API key required)
cargo run -p agent-sdk --example otel --features otel
```

## Creating Custom Tools

Tools let your agent interact with external systems. Implement the `Tool` trait:

```rust
use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier, ToolRegistry, DynamicToolName};
use serde_json::{Value, json};

/// A tool that fetches the current weather for a city
struct WeatherTool;

impl Tool<()> for WeatherTool {
    // Tool names are now typed - DynamicToolName for runtime names
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("get_weather")
    }

    // Optional: human-readable display name for UIs
    fn display_name(&self) -> &'static str {
        "Get Weather"
    }

    fn description(&self) -> &'static str {
        "Get the current weather for a city. Returns temperature and conditions."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'San Francisco'"
                }
            },
            "required": ["city"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Observe = no confirmation needed
        // Confirm = requires user approval (triggers yield/resume)
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
        let city = input["city"].as_str().unwrap_or("Unknown");

        // In a real implementation, call a weather API here
        let weather = format!("Weather in {city}: 72°F, Sunny");

        Ok(ToolResult::success(weather))
    }
}

// Register tools with the agent
let mut tools = ToolRegistry::new();
tools.register(WeatherTool);

let agent = builder::<()>()
    .provider(provider)
    .tools(tools)
    .build();
```

## Async Tools (Long-Running Operations)

For operations that take time (API calls, file processing, etc.), implement `AsyncTool`:

```rust
use agent_sdk::{AsyncTool, ToolContext, ToolTier, ToolOutcome, DynamicToolName, ProgressStage, ToolStatus, ToolResult};
use serde::{Serialize, Deserialize};
use serde_json::{Value, json};
use futures::Stream;

// Define progress stages for your operation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferStage {
    Initiated,
    Processing,
    Completed,
}

impl ProgressStage for TransferStage {}

struct TransferTool;

impl AsyncTool<()> for TransferTool {
    type Name = DynamicToolName;
    type Stage = TransferStage;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("transfer_funds")
    }

    fn display_name(&self) -> &'static str { "Transfer Funds" }
    fn description(&self) -> &'static str { "Transfer funds between accounts" }
    fn input_schema(&self) -> Value { json!({"type": "object"}) }
    fn tier(&self) -> ToolTier { ToolTier::Confirm }

    // Start the operation - returns quickly
    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolOutcome> {
        let operation_id = "op_123"; // Start your operation, get an ID
        Ok(ToolOutcome::in_progress(operation_id, "Transfer initiated"))
    }

    // Stream progress updates until completion
    fn check_status(&self, _ctx: &ToolContext<()>, operation_id: &str)
        -> impl Stream<Item = ToolStatus<TransferStage>> + Send
    {
        async_stream::stream! {
            yield ToolStatus::Progress {
                stage: TransferStage::Processing,
                message: "Processing transfer...".into(),
                data: None,
            };
            // Poll your service, yield progress...
            yield ToolStatus::Completed(ToolResult::success("Transfer complete"));
        }
    }
}

// Register async tools separately
let mut tools = ToolRegistry::new();
tools.register_async(TransferTool);
```

## Extended Thinking

Configure thinking on the provider when you choose the model:

```rust
use agent_sdk::{builder, Effort, ThinkingConfig, providers::{AnthropicProvider, OpenAIProvider}};

// Anthropic Claude Sonnet 4.6 / Opus 4.6 support adaptive thinking.
let anthropic_agent = builder::<()>()
    .provider(
        AnthropicProvider::sonnet(api_key.clone())
            .with_thinking(ThinkingConfig::adaptive()),
    )
    .build();

// OpenAI models use explicit reasoning effort, not adaptive mode.
let openai_agent = builder::<()>()
    .provider(
        OpenAIProvider::gpt54(openai_api_key)
            .with_thinking(ThinkingConfig::new(10_000).with_effort(Effort::High)),
    )
    .build();
```

Adaptive thinking is only supported for Anthropic `claude-sonnet-4-6` and `claude-opus-4-6`.
When thinking is enabled, the agent emits `AgentEvent::Thinking` and `AgentEvent::ThinkingDelta`
events with the model's reasoning output.

## Lifecycle Hooks

Hooks let you intercept and control agent behavior:

```rust
use agent_sdk::{AgentEvent, AgentHooks, ToolDecision, ToolInvocation, ToolResult, ToolTier};
use async_trait::async_trait;

struct MyHooks;

#[async_trait]
impl AgentHooks for MyHooks {
    /// Called before each tool execution.
    ///
    /// `invocation` carries the tool call ID, name, tier, requested input,
    /// effective input (after SDK preparation), and any listen-tool context.
    async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
        println!("[LOG] Tool call: {}", invocation.tool_name);

        // You could implement:
        // - User confirmation dialogs
        // - Rate limiting
        // - Input validation
        // - Audit logging

        match invocation.tier {
            ToolTier::Observe => ToolDecision::Allow,
            ToolTier::Confirm => {
                // In a real app, prompt the user here.
                // The agent can yield via `RequiresConfirmation` and resume
                // after the user decides.
                ToolDecision::Allow
            }
        }
    }

    /// Called after a tool returns a successful or failing result.
    async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
        println!("[LOG] {tool_name} completed: success={}", result.success);
    }

    /// Called for every agent event (optional)
    async fn on_event(&self, event: &AgentEvent) {
        // Track events, update UI, etc.
    }
}

let agent = builder::<()>()
    .provider(provider)
    .hooks(MyHooks)
    .build();
```

### Audit Sink

`post_tool_use` only fires for successful tool completion. Servers that need
to explain **every** lifecycle outcome — blocked, requires-confirmation,
cached, replayed, invalidated, completed, persistence-failed — should attach
a [`ToolAuditSink`] instead. The agent loop emits one [`ToolAuditRecord`] per
lifecycle transition with provider/model provenance, tier, and the variant
payload.

```rust
use agent_sdk::{NoopAuditSink, ToolAuditRecord, ToolAuditSink};
use async_trait::async_trait;

struct DurableAuditSink {
    // your durable backend handle
}

#[async_trait]
impl ToolAuditSink for DurableAuditSink {
    async fn record(&self, record: ToolAuditRecord) {
        // Persist to your audit log. The discriminant is `record.outcome_kind()`.
    }
}

let agent = builder::<()>()
    .provider(provider)
    .audit_sink(DurableAuditSink { /* ... */ })
    .build();
```

The default sink is [`NoopAuditSink`], which is appropriate for local/CLI use.

## Custom Context

The generic parameter `T` in `Tool<T>` and `builder::<T>()` lets you pass custom data to your tools:

```rust
// Define your context type
struct MyContext {
    user_id: String,
    database: Database,
}

// Implement tools with access to context
impl Tool<MyContext> for MyTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("my_tool")
    }

    // ... other trait methods ...

    async fn execute(&self, ctx: &ToolContext<MyContext>, input: Value) -> anyhow::Result<ToolResult> {
        // Access your context
        let user = &ctx.app.user_id;
        let db = &ctx.app.database;
        // ...
    }
}

// Build agent with your context type
let agent = builder::<MyContext>()
    .provider(provider)
    .tools(tools)
    .build();

// Pass context when running
let tool_ctx = ToolContext::new(MyContext {
    user_id: "user_123".to_string(),
    database: db,
});
agent.run(thread_id, prompt, tool_ctx, CancellationToken::new());
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Agent Loop                                   │
│      Orchestrates: prompt → LLM → tool calls → results → LLM            │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ LlmProvider  │  │    Tools     │  │    Hooks     │  │   Events     │ │
│  │  (trait)     │  │   Registry   │  │ (pre/post)   │  │  (stream)    │ │
│  │              │  │              │  │              │  │              │ │
│  │ - Anthropic  │  │ - Tool       │  │ - Default    │  │ - Text       │ │
│  │ - OpenAI     │  │ - AsyncTool  │  │ - AllowAll   │  │ - ToolCall   │ │
│  │ - Gemini     │  │ - MCP Bridge │  │ - Logging    │  │ - Progress   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ MessageStore │  │  StateStore  │  │  ToolExec    │  │ Environment  │ │
│  │  (trait)     │  │   (trait)    │  │   Store      │  │  (trait)     │ │
│  │              │  │              │  │  (trait)     │  │              │ │
│  │ Conversation │  │ Agent state  │  │ Idempotency  │  │ File + exec  │ │
│  │   history    │  │ checkpoints  │  │  tracking    │  │  operations  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Subagents   │  │     MCP      │  │  Web Tools   │  │    Todo      │ │
│  │              │  │              │  │              │  │   System     │ │
│  │ Nested agent │  │ External     │  │ Search +     │  │              │ │
│  │  execution   │  │ tool servers │  │ fetch URLs   │  │ Task track   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

## Streaming Events

The agent emits events during execution for real-time UI updates.

### Event Channel Behavior

The agent uses a bounded channel (capacity 100) for events. The SDK is designed to be resilient to slow consumers:

- **Non-blocking sends**: Events are sent using `try_send` first. If the channel is full, the SDK waits up to 30 seconds before timing out.
- **Consumer disconnection**: If the event receiver is dropped, the agent continues processing the LLM response without blocking.
- **Backpressure handling**: If your consumer is slow, you'll see warnings like `Event channel full, waiting for consumer...` in the logs.

**Best practices for consuming events:**

```rust
// GOOD: Process events quickly, offload heavy work
while let Some(envelope) = events.recv().await {
    match envelope.event {
        AgentEvent::TextDelta {
            message_id: _,
            delta,
        } => {
            // Quick: just buffer or forward
            buffer.push_str(&delta);
        }
        AgentEvent::Done { .. } => break,
        _ => {}
    }
}

// GOOD: Spawn heavy processing to avoid blocking
while let Some(envelope) = events.recv().await {
    let envelope = envelope.clone();
    tokio::spawn(async move {
        // Heavy processing in background
        save_to_database(&envelope).await;
    });
}

// BAD: Blocking I/O in the event loop
while let Some(_envelope) = events.recv().await {
    // This blocks the consumer, causing backpressure
    std::thread::sleep(Duration::from_secs(1));
}
```

### Event Types

| Event | Description |
|-------|-------------|
| `Start` | Agent begins processing a turn |
| `Thinking` | Extended thinking output (when enabled) |
| `TextDelta` | Streaming text chunk from LLM |
| `Text` | Complete text block from LLM |
| `ToolCallStart` | Tool execution starting |
| `ToolCallEnd` | Tool execution completed |
| `ToolProgress` | Progress update from async tool |
| `ToolRequiresConfirmation` | Tool needs user approval |
| `TurnComplete` | One LLM round-trip finished |
| `ContextCompacted` | Conversation was summarized to save tokens |
| `SubagentProgress` | Progress from nested subagent |
| `Done` | Agent completed successfully |
| `Error` | An error occurred |

```rust
while let Some(envelope) = events.recv().await {
    match envelope.event {
        AgentEvent::Start { thread_id, turn } => {
            println!("Starting turn {turn}");
        }
        AgentEvent::TextDelta {
            message_id: _,
            delta,
        } => {
            print!("{delta}"); // Stream to UI
        }
        AgentEvent::ToolCallStart { name, .. } => {
            println!("Calling tool: {name}");
        }
        AgentEvent::ToolProgress { stage, message, .. } => {
            println!("Progress: {stage} - {message}");
        }
        AgentEvent::Done { total_turns, total_usage, .. } => {
            let total_tokens = total_usage.input_tokens + total_usage.output_tokens;
            println!("Completed in {total_turns} turns, {total_tokens} tokens");
            break;
        }
        _ => {}
    }
}
```

## Built-in Providers

| Provider | Models | Usage |
|----------|--------|-------|
| Anthropic | Claude Sonnet, Opus, Haiku | `AnthropicProvider::sonnet(api_key)` |
| OpenAI | GPT-5.4, GPT-5.3-Codex, GPT-4.1, o-series | `OpenAIProvider::gpt54(api_key)` |
| Google | Gemini 3.x and 2.x families | `GeminiProvider::new(api_key, model)` |

Implement `LlmProvider` trait to add your own.

### OpenAI-Compatible Endpoints

`OpenAIProvider` can target compatible Chat Completions APIs with provider-specific helpers:

```rust
use agent_sdk::providers::OpenAIProvider;

let kimi_api_key = std::env::var("MOONSHOT_API_KEY")?;
let zai_api_key = std::env::var("ZAI_API_KEY")?;
let minimax_api_key = std::env::var("MINIMAX_API_KEY")?;

// Default reasoning models
let kimi = OpenAIProvider::kimi_k2_5(kimi_api_key);
let zai = OpenAIProvider::zai_glm5(zai_api_key);
let minimax = OpenAIProvider::minimax_m2_5(minimax_api_key);

// Custom model overrides
let kimi_custom = OpenAIProvider::kimi(
    std::env::var("MOONSHOT_API_KEY")?,
    "kimi-k2-thinking".to_string(),
);
let zai_custom = OpenAIProvider::zai(
    std::env::var("ZAI_API_KEY")?,
    "glm-4.5".to_string(),
);
let minimax_custom = OpenAIProvider::minimax(
    std::env::var("MINIMAX_API_KEY")?,
    "MiniMax-M2.5".to_string(),
);
```

## Built-in Primitive Tools

For agents that need file system access:

| Tool | Description |
|------|-------------|
| `ReadTool` | Read file contents |
| `WriteTool` | Create or overwrite files |
| `EditTool` | Make targeted edits to files |
| `GlobTool` | Find files matching patterns |
| `GrepTool` | Search file contents with regex |
| `BashTool` | Execute shell commands |
| `NotebookReadTool` | Read Jupyter notebook contents |
| `NotebookEditTool` | Edit Jupyter notebook cells |

These require an `Environment` (use `InMemoryFileSystem` for sandboxed testing or `LocalFileSystem` for real file access).

## Web Tools

For agents that need internet access:

| Tool | Description |
|------|-------------|
| `WebSearchTool` | Search the web via pluggable providers |
| `LinkFetchTool` | Fetch URL content with SSRF protection |

```rust
use agent_sdk::web::{WebSearchTool, LinkFetchTool, BraveSearchProvider};

// Web search with Brave
let search_provider = BraveSearchProvider::new(brave_api_key);
let search_tool = WebSearchTool::new(search_provider);

// URL fetching
let fetch_tool = LinkFetchTool::new();
```

## Task Tracking

Built-in todo system for tracking multi-step tasks:

```rust
use agent_sdk::todo::{TodoState, TodoWriteTool, TodoReadTool};
use std::sync::Arc;
use tokio::sync::RwLock;

let state = Arc::new(RwLock::new(TodoState::new()));
let write_tool = TodoWriteTool::new(Arc::clone(&state));
let read_tool = TodoReadTool::new(state);
```

Task statuses: `Pending` (○), `InProgress` (⚡), `Completed` (✓)

## Subagents

Spawn isolated child agents for complex subtasks:

```rust
use agent_sdk::{SubagentFactory, SubagentConfig, SubagentTool};

let factory = SubagentFactory::new(provider, subagent_tools);
let config = SubagentConfig {
    system_prompt: "You are a research assistant.".into(),
    max_turns: Some(10),
    ..Default::default()
};
let subagent_tool = SubagentTool::new(factory, config);
```

Subagents run in isolated threads with their own context and stream progress events back to the parent.

## MCP Support

Integrate external tools via the Model Context Protocol:

```rust
use agent_sdk::mcp::{McpClient, StdioTransport, register_mcp_tools};

// Connect to an MCP server
let transport = StdioTransport::spawn("mcp-server", &["--stdio"])?;
let client = McpClient::new(transport);

// Register all tools from the MCP server
register_mcp_tools(&mut registry, &client).await?;
```

## User Interaction

Tools for agent-initiated questions and confirmations:

```rust
use agent_sdk::user_interaction::AskUserQuestionTool;

let question_tool = AskUserQuestionTool::new(question_tx, response_rx);
```

## Persistence

The SDK provides trait-based storage for production deployments:

| Store | Purpose |
|-------|---------|
| `MessageStore` | Conversation history per thread |
| `StateStore` | Agent state checkpoints for recovery |
| `ToolExecutionStore` | Write-ahead tool execution tracking (idempotency) |

`InMemoryStore` and `InMemoryExecutionStore` are provided for testing. For production, implement the traits with your database (Postgres, Redis, etc.):

```rust
use agent_sdk::{MessageStore, StateStore, ToolExecutionStore, InMemoryStore, InMemoryExecutionStore};

// Use in-memory stores for development
let message_store = InMemoryStore::new();
let state_store = InMemoryStore::new();
let exec_store = InMemoryExecutionStore::new();

let agent = builder::<()>()
    .provider(provider)
    .message_store(message_store)
    .state_store(state_store)
    .execution_store(exec_store)
    .build();
```

The `ToolExecutionStore` enables crash recovery by recording tool calls before execution, ensuring idempotency on retry.

## Security Considerations

- **`#[forbid(unsafe_code)]`** - No unsafe Rust anywhere in the codebase
- **Capability-based permissions** - Control read/write/exec access via `AgentCapabilities`
- **Tool tiers** - Classify tools by risk level; use hooks to require confirmation
- **Sandboxing** - Use `InMemoryFileSystem` for testing without real file access

See [SECURITY.md](SECURITY.md) for the full security policy.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow:

- Development setup
- Code quality requirements
- Pull request process

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2025-2026 Bipa.
