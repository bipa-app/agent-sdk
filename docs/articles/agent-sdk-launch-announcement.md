# agent-sdk: durable, crash-safe agent loops in Rust

Today we're open-sourcing **agent-sdk** — a Rust SDK for building durable, crash-safe LLM agent loops with structured tool execution. It's MIT-licensed and developed in the open at [github.com/bipa-app/agent-sdk](https://github.com/bipa-app/agent-sdk).

To set expectations up front: this is a **technical preview** (v0.9, pre-1.0). The core local-agent loop is solid and exercised by a real test suite; the durable serving backend is feature-complete and hardened under fault-injection testing. We're developing openly and hardening toward production. No benchmarks, no battle-tested claims — just an honest look at what's here.

## The problem

Agent loops are deceptively hard to get *right* once you care about reliability. What happens when a user cancels mid-tool-call? Does your message history end up with an orphaned `tool_use` block that the next API call rejects? What happens when the process crashes between "model asked to run a tool" and "tool finished"? Does the tool run twice? Do events get dropped or replayed out of order?

Most agent frameworks treat these as edge cases. agent-sdk treats them as the contract.

## What it does

**Durable execution.** A pluggable `EventStore`, `MessageStore`, and `StateStore` persist conversation state on every turn boundary. A `ToolExecutionStore` records tool intent *before* execution using a write-ahead pattern, so a crash mid-tool resumes idempotently. The append-only event journal both persists *and* replays without gaps.

**Harness robustness.** Cancellation is cooperative via `tokio-util`'s `CancellationToken`, and the SDK cancels in-flight tool futures at its own boundary — synthesizing a balanced `tool_result` so your history stays valid for the next provider call. Tool panics are caught close to the boundary; the run loop itself is panic-isolated. Optional per-tool timeouts produce a synthetic error result rather than wedging the loop.

**Real tool ergonomics.** Choose your level of ceremony: `SimpleTool` (untyped `Value` in), `TypedTool` (deserializes and validates into your struct, with self-correction on malformed input), `Tool` (full control), or `AsyncTool` (long-running ops with progress streaming). Derive macros (`#[derive(Tool)]`, `#[derive(TypedTool)]`, `#[derive(ToolName)]`) and a `tool!` declarative macro cut the boilerplate. Tools carry a `ToolTier` — `Observe` (read-only) or `Confirm` (pauses for human approval).

**Multiple providers, one trait.** Anthropic ships by default; OpenAI, Gemini, Vertex AI, and Cloudflare are opt-in feature flags. Structured output, extended thinking, and prompt caching are supported where the provider allows.

**Optional extras behind flags:** Model Context Protocol clients (`mcp`, stdio + streamable-HTTP), web search and SSRF-protected fetch (`web`), markdown skill loading (`skills`), and OpenTelemetry instrumentation (`otel`). There's also a durable gRPC + Postgres/SQLite serving host for when you need a Temporal-grade backend for agent loops.

`#![forbid(unsafe_code)]` throughout. Edition 2024, MSRV 1.91.

## Quickstart

```bash
cargo add agent-sdk
```

```rust
use agent_sdk::{ThreadId, builder, providers::AnthropicProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // `try_from_env` reads ANTHROPIC_API_KEY. The builder uses an in-memory
    // event store by default — no Arc/store ceremony to get started.
    let agent = builder::<()>()
        .provider(AnthropicProvider::try_from_env()?)
        .build();

    // One call: ask a question, get the assembled answer back.
    let answer = agent
        .ask(ThreadId::new(), "What is the capital of France?")
        .await?;

    println!("Agent: {answer}");
    Ok(())
}
```

When you outgrow `.ask()`, drop down to `.run()` for full control over context, cancellation, and confirmation flows — or `.run_turn()` for server orchestration with external tool execution and strict-durability checkpointing.

## Where it stands

Working today: the agent loop, streaming, tool execution with mid-turn cancellation, typed tools, structured output, primitive file tools, MCP, hooks for human-in-the-loop, context compaction, and OpenTelemetry. Still hardening: the gRPC/HTTP transport surface, multi-instance fanout under chaos, and provider error modes under adverse conditions. We're tracking these in the open.

## Try it

- **Install:** `cargo add agent-sdk`
- **Source & issues:** [github.com/bipa-app/agent-sdk](https://github.com/bipa-app/agent-sdk)
- **API docs:** [docs.rs/agent-sdk](https://docs.rs/agent-sdk)

Browse the `examples/` directory for runnable demos — basic agents, custom and typed tools, derive macros, streaming, structured output, MCP, and custom hooks. We'd love your issues, PRs, and honest feedback as we harden toward 1.0.