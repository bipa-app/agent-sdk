# agent-sdk

[![crates.io](https://img.shields.io/crates/v/agent-sdk.svg)](https://crates.io/crates/agent-sdk)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk)](https://docs.rs/agent-sdk)
[![Rust 1.91+](https://img.shields.io/badge/rust-1.91%2B-orange.svg)](https://www.rust-lang.org)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

A Rust SDK for building AI agents powered by large language models. Agents
reason, call tools, and take actions through a streaming, event-driven loop.

```toml
[dependencies]
agent-sdk = "0.9"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"
```

## Quickstart — `ask()`

The 30-second path: build an agent, ask a question, get the answer. `ask()`
builds the tool context and cancellation token for you and returns the
assembled assistant text.

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

`use agent_sdk::prelude::*;` brings the dozen names a newcomer needs: the
[`builder`], config + I/O types, the `Tool` surface, the in-memory event
store, the cancellation token, and the default `AnthropicProvider`. When you
need application context, a confirmation flow, explicit cancellation, or the
raw run state, drop down to [`run()`].

## Features

The default build is lightweight: a plain `cargo add agent-sdk` enables only
the `anthropic` provider and the core runtime — no `sqlx`, `tonic`, `lapin`,
`prost`, or `opentelemetry`. Everything else is opt-in:

| Feature | Enables |
|---------|---------|
| `anthropic` *(default)* | Anthropic (Claude) provider |
| `openai` | OpenAI Chat Completions provider |
| `openai-codex` | OpenAI Codex / Responses provider |
| `gemini` | Google Gemini provider |
| `vertex` | Google Vertex AI provider |
| `cloudflare` | Cloudflare Workers AI provider |
| `web` | Web search + URL fetch tools (pulls in `html2text`) |
| `mcp` | Model Context Protocol client (stdio + streamable-HTTP/SSE) |
| `skills` | Skill / command loading from markdown |
| `otel` | OpenTelemetry spans |

```toml
agent-sdk = { version = "0.9", features = ["openai", "gemini", "web", "mcp"] }
```

## Cookbook

Runnable recipes for the common building blocks live in
[`COOKBOOK.md`](https://github.com/bipa-app/agent-sdk/blob/main/crates/agent-sdk/COOKBOOK.md)
and as compiling examples under
[`examples/`](https://github.com/bipa-app/agent-sdk/tree/main/crates/agent-sdk/examples):
typed tools (`TypedTool`), schema-validated structured output, streaming, MCP
(including the HTTP transport), durable serving, and human-in-the-loop (HITL)
confirmation.

```bash
cargo run --example basic_agent
cargo run --example typed_tool
cargo run --example streaming
cargo run --example mcp_http_remote --features mcp
```

## Documentation

- API reference: <https://docs.rs/agent-sdk>
- Repository, full README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).

[`builder`]: https://docs.rs/agent-sdk/latest/agent_sdk/fn.builder.html
[`run()`]: https://docs.rs/agent-sdk/latest/agent_sdk/struct.AgentLoop.html#method.run
