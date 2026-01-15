# Agent SDK

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A Rust SDK for building AI agents powered by large language models (LLMs). Create agents that can reason, use tools, and take actions through a streaming, event-driven architecture.

> **⚠️ Early Development**: This library is in active development (v0.1.x). APIs may change between versions and there may be bugs. Use in production at your own risk. Feedback and contributions are welcome!

## What is an Agent?

An agent is an LLM that can do more than just chat—it can use tools to interact with the world. This SDK provides the infrastructure to:

- Send messages to an LLM and receive streaming responses
- Define tools the LLM can call (APIs, file operations, databases, etc.)
- Execute tool calls and feed results back to the LLM
- Control the agent loop with hooks for logging, security, and approval workflows

## Features

- **Agent Loop** - Core orchestration that handles the LLM conversation and tool execution cycle
- **Provider Agnostic** - Built-in support for Anthropic (Claude), OpenAI, and Google Gemini, plus a trait for custom providers
- **Tool System** - Define tools with JSON schema validation; the LLM decides when to use them
- **Lifecycle Hooks** - Intercept tool calls for logging, user confirmation, rate limiting, or security checks
- **Streaming Events** - Real-time event stream for building responsive UIs
- **Primitive Tools** - Ready-to-use tools for file operations (Read, Write, Edit, Glob, Grep, Bash)
- **Security Model** - Capability-based permissions and tool tiers (Observe, Confirm, RequiresPin)
- **Persistence** - Trait-based storage for conversation history and agent state

## Requirements

- Rust 1.85+ (2024 edition)
- An API key for your chosen LLM provider

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-sdk = "0.1"
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
use agent_sdk::{builder, ThreadId, ToolContext, AgentEvent, providers::AnthropicProvider};

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

    // Run the agent and get a stream of events
    let mut events = agent.run(
        thread_id,
        "What is the capital of France?".to_string(),
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
use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier, ToolRegistry};
use async_trait::async_trait;
use serde_json::{Value, json};

/// A tool that fetches the current weather for a city
struct WeatherTool;

#[async_trait]
impl Tool<()> for WeatherTool {
    fn name(&self) -> &'static str {
        "get_weather"
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
        // Confirm = requires user approval
        // RequiresPin = requires PIN verification
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
                ToolDecision::Allow
            }
            ToolTier::RequiresPin => {
                ToolDecision::Block("PIN verification not implemented".to_string())
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
#[async_trait]
impl Tool<MyContext> for MyTool {
    async fn execute(&self, ctx: &ToolContext<MyContext>, input: Value) -> anyhow::Result<ToolResult> {
        // Access your context
        let user = &ctx.data.user_id;
        let db = &ctx.data.database;
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
┌─────────────────────────────────────────────────────────────┐
│                        Agent Loop                           │
│  Orchestrates: prompt → LLM → tool calls → results → LLM   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ LlmProvider │  │    Tools    │  │       Hooks         │ │
│  │  (trait)    │  │  Registry   │  │  (pre/post tool)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │MessageStore │  │ StateStore  │  │    Environment      │ │
│  │  (trait)    │  │  (trait)    │  │  (file/exec ops)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
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

These require an `Environment` (use `InMemoryFileSystem` for sandboxed testing or `LocalFileSystem` for real file access).

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
