# Agent SDK

[![Crates.io](https://img.shields.io/crates/v/agent-sdk.svg)](https://crates.io/crates/agent-sdk)
[![Documentation](https://docs.rs/agent-sdk/badge.svg)](https://docs.rs/agent-sdk)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A Rust SDK for building LLM-powered agents with tool execution, streaming events, and lifecycle hooks.

> **⚠️ Early Development**: This library is in active development (v0.1.x). APIs may change between versions and there may be bugs. Use in production at your own risk. Feedback and contributions are welcome!

## Features

- **Agent Loop** - Core orchestration for LLM conversations with tool calling
- **Provider Agnostic** - Works with Anthropic, OpenAI, Google Gemini, or custom providers
- **Tool System** - Define custom tools with JSON schema validation
- **Lifecycle Hooks** - Pre/post tool execution hooks for logging, security, and customization
- **Streaming Events** - Real-time event stream for agent actions
- **Primitive Tools** - Built-in tools for file operations (Read, Write, Edit, Glob, Grep, Bash)
- **Security Model** - Capability-based permissions and tool tiers
- **Persistence** - Trait-based message and state storage

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-sdk = "0.1"
```

Or install from git:

```toml
[dependencies]
agent-sdk = { git = "https://github.com/bipa-app/agent-sdk", branch = "main" }
```

## Quick Start

```rust
use agent_sdk::{builder, ThreadId, ToolContext, AgentEvent, providers::AnthropicProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    // Build the agent
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .build();

    // Run a conversation
    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    let mut events = agent.run(
        thread_id,
        "What is the capital of France?".to_string(),
        tool_ctx,
    );

    // Process streaming events
    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::Text { text } => print!("{text}"),
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
```

## Examples

The `examples/` directory contains runnable examples:

```bash
# Basic agent
ANTHROPIC_API_KEY=your_key cargo run --example basic_agent

# Custom tools
ANTHROPIC_API_KEY=your_key cargo run --example custom_tool

# Lifecycle hooks
ANTHROPIC_API_KEY=your_key cargo run --example custom_hooks

# Primitive file tools
ANTHROPIC_API_KEY=your_key cargo run --example with_primitive_tools
```

## Custom Tools

Implement the `Tool` trait to create custom tools:

```rust
use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier};
use async_trait::async_trait;
use serde_json::{Value, json};

struct CalculatorTool;

#[async_trait]
impl Tool<()> for CalculatorTool {
    fn name(&self) -> &'static str {
        "calculator"
    }

    fn description(&self) -> &'static str {
        "Add two numbers together"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);
        Ok(ToolResult::success(format!("{} + {} = {}", a, b, a + b)))
    }
}
```

## Lifecycle Hooks

Control agent behavior with hooks:

```rust
use agent_sdk::{AgentHooks, AgentEvent, ToolDecision, ToolResult, ToolTier};
use async_trait::async_trait;
use serde_json::Value;

struct MyHooks;

#[async_trait]
impl AgentHooks for MyHooks {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value, tier: ToolTier) -> ToolDecision {
        println!("About to call: {tool_name}");

        match tier {
            ToolTier::Observe => ToolDecision::Allow,
            ToolTier::Confirm => {
                // Implement confirmation logic
                ToolDecision::Allow
            }
            ToolTier::RequiresPin => {
                ToolDecision::RequiresPin("PIN required".to_string())
            }
        }
    }

    async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
        println!("{tool_name} completed in {:?}ms", result.duration_ms);
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Loop                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ LlmProvider │  │    Tools    │  │       Hooks         │ │
│  │  (trait)    │  │  Registry   │  │  (pre/post tool)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │MessageStore │  │ StateStore  │  │    Environment      │ │
│  │  (trait)    │  │  (trait)    │  │     (trait)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `llm` | LLM provider trait and message types |
| `tools` | Tool trait, registry, and execution context |
| `hooks` | Lifecycle hooks for tool execution |
| `stores` | Message and state persistence traits |
| `environment` | File and command execution abstraction |
| `capabilities` | Security model for agent operations |
| `primitive_tools` | Built-in file operation tools |
| `providers` | LLM provider implementations |
| `subagent` | Nested agent execution |
| `mcp` | Model Context Protocol support |

## Providers

Built-in LLM providers:

- **Anthropic** - Claude models (Sonnet, Opus, Haiku)
- **OpenAI** - GPT models
- **Google Gemini** - Gemini models

Implement `LlmProvider` to add custom providers.

## Security

- Uses `#[forbid(unsafe_code)]`
- Capability-based permissions (`AgentCapabilities`)
- Tool tiers for operation classification
- Hooks for approval workflows

See [SECURITY.md](SECURITY.md) for security policy and best practices.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
