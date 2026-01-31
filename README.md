# Agent SDK

[![Crates.io](https://img.shields.io/crates/v/agent-sdk.svg)](https://crates.io/crates/agent-sdk)
[![Documentation](https://docs.rs/agent-sdk/badge.svg)](https://docs.rs/agent-sdk)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A Rust SDK for building AI agents powered by large language models (LLMs). Create agents that can reason, use tools, and take actions through a streaming, event-driven architecture.

> **⚠️ Early Development**: This library is in active development (v0.4.x). APIs may change between versions and there may be bugs. Use in production at your own risk. Feedback and contributions are welcome!

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
- **Extended Thinking** - Support for Anthropic's extended thinking feature via `ThinkingConfig`
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

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-sdk = "0.4"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"
```

Or to install the latest development version from git:

```toml
[dependencies]
agent-sdk = { git = "https://github.com/bipa-app/agent-sdk", branch = "main" }
```

## Quick Start

```rust
use agent_sdk::{builder, ThreadId, ToolContext, AgentEvent, AgentInput, providers::AnthropicProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Get your API key from the environment
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    // Build an agent with the Anthropic provider
    // The ::<()> specifies no custom context type (see "Custom Context" below)
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .build();

    // Each conversation gets a unique thread ID
    let thread_id = ThreadId::new();

    // ToolContext can carry custom data to your tools (empty here)
    let tool_ctx = ToolContext::new(());

    // Run the agent and get a stream of events plus final state
    let (mut events, _final_state) = agent.run(
        thread_id,
        AgentInput::Text("What is the capital of France?".to_string()),
        tool_ctx,
    );

    // Process events as they arrive
    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::Text { text } => print!("{text}"),
            AgentEvent::Done { .. } => break,
            AgentEvent::Error { message, .. } => eprintln!("Error: {message}"),
            _ => {} // Other events: ToolCallStart, ToolCallEnd, etc.
        }
    }

    Ok(())
}
```

## Examples

Clone the repo and run the examples:

```bash
git clone https://github.com/bipa-app/agent-sdk
cd agent-sdk

# Basic conversation (no tools)
ANTHROPIC_API_KEY=your_key cargo run --example basic_agent

# Agent with custom tools
ANTHROPIC_API_KEY=your_key cargo run --example custom_tool

# Using lifecycle hooks for logging and rate limiting
ANTHROPIC_API_KEY=your_key cargo run --example custom_hooks

# Agent with file operation tools
ANTHROPIC_API_KEY=your_key cargo run --example with_primitive_tools
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

Enable Anthropic's extended thinking for complex reasoning:

```rust
use agent_sdk::{builder, AgentConfig, ThinkingConfig, providers::AnthropicProvider};

let agent = builder::<()>()
    .provider(AnthropicProvider::sonnet(api_key))
    .config(AgentConfig {
        thinking: Some(ThinkingConfig {
            budget_tokens: 10000, // Token budget for thinking
        }),
        ..Default::default()
    })
    .build();
```

When enabled, the agent emits `AgentEvent::Thinking` events with the model's reasoning process.

## Lifecycle Hooks

Hooks let you intercept and control agent behavior:

```rust
use agent_sdk::{AgentHooks, AgentEvent, ToolDecision, ToolResult, ToolTier};
use async_trait::async_trait;
use serde_json::Value;

struct MyHooks;

#[async_trait]
impl AgentHooks for MyHooks {
    /// Called before each tool execution
    async fn pre_tool_use(&self, tool_name: &str, input: &Value, tier: ToolTier) -> ToolDecision {
        println!("[LOG] Tool call: {tool_name}");

        // You could implement:
        // - User confirmation dialogs
        // - Rate limiting
        // - Input validation
        // - Audit logging

        match tier {
            ToolTier::Observe => ToolDecision::Allow,
            ToolTier::Confirm => {
                // In a real app, prompt the user here
                // The agent can yield and resume after user confirmation
                ToolDecision::Allow
            }
        }
    }

    /// Called after each tool execution
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
agent.run(thread_id, prompt, tool_ctx);
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
while let Some(event) = events.recv().await {
    match event {
        AgentEvent::TextDelta { delta } => {
            // Quick: just buffer or forward
            buffer.push_str(&delta);
        }
        AgentEvent::Done { .. } => break,
        _ => {}
    }
}

// GOOD: Spawn heavy processing to avoid blocking
while let Some(event) = events.recv().await {
    let event = event.clone();
    tokio::spawn(async move {
        // Heavy processing in background
        save_to_database(&event).await;
    });
}

// BAD: Blocking I/O in the event loop
while let Some(event) = events.recv().await {
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
while let Some(event) = events.recv().await {
    match event {
        AgentEvent::Start { thread_id, turn } => {
            println!("Starting turn {turn}");
        }
        AgentEvent::TextDelta { delta } => {
            print!("{delta}"); // Stream to UI
        }
        AgentEvent::ToolCallStart { name, .. } => {
            println!("Calling tool: {name}");
        }
        AgentEvent::ToolProgress { stage, message, .. } => {
            println!("Progress: {stage} - {message}");
        }
        AgentEvent::Done { total_turns, total_usage, .. } => {
            println!("Completed in {total_turns} turns, {} tokens", total_usage.total());
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
| OpenAI | GPT-4, GPT-3.5, etc. | `OpenAiProvider::new(api_key, model)` |
| Google | Gemini Pro, etc. | `GeminiProvider::new(api_key, model)` |

Implement `LlmProvider` trait to add your own.

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
    max_turns: 10,
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

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code quality requirements
- Pull request process

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
