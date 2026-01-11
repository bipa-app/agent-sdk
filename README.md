# Agent SDK

A Rust SDK for building LLM-powered agents with tool execution, streaming, and lifecycle hooks.

## Features

- **Agent Loop**: Core orchestration for LLM conversations with tool calling
- **LLM Abstraction**: Provider-agnostic interface for chat completions
- **Tool System**: Define and register custom tools with JSON schema validation
- **Lifecycle Hooks**: Pre/post tool execution hooks for logging, security, and customization
- **Environment Abstraction**: File and command execution with security capabilities
- **Primitive Tools**: Built-in tools for file operations (Read, Write, Edit, Glob, Grep, Bash)
- **Persistence**: Trait-based message and state storage with in-memory default

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-sdk = { git = "https://github.com/bipa-app/agent-sdk", branch = "main" }
```

## Quick Start

```rust
use agent_sdk::{AgentLoop, AgentConfig, InMemoryStore};

// Create agent with your LLM provider
let agent = AgentLoop::builder()
    .provider(your_llm_provider)
    .config(AgentConfig::default())
    .store(InMemoryStore::new())
    .build();

// Run agent loop
let events = agent.run("Hello, what can you help me with?").await;

// Handle events (streaming)
while let Some(event) = events.next().await {
    match event {
        AgentEvent::Text { text } => println!("{}", text),
        AgentEvent::ToolCall { name, .. } => println!("Calling tool: {}", name),
        AgentEvent::Done { usage, .. } => break,
        _ => {}
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
| `tools` | Tool trait, registry, and context |
| `hooks` | Lifecycle hooks for tool execution |
| `stores` | Message and state persistence traits |
| `environment` | File/command execution abstraction |
| `capabilities` | Security model for agent operations |
| `primitive_tools` | Built-in file operation tools |
| `providers` | LLM provider implementations (Anthropic) |

## License

MIT OR Apache-2.0
