# agent-sdk-tools

[![crates.io](https://img.shields.io/crates/v/agent-sdk-tools.svg)](https://crates.io/crates/agent-sdk-tools)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-tools)](https://docs.rs/agent-sdk-tools)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

Tool and store contracts for the [Agent SDK](https://github.com/bipa-app/agent-sdk):
the `Tool` / `TypedTool` / `AsyncTool` traits, the `ToolRegistry`, lifecycle
`AgentHooks`, the audit-sink protocol, and the `MessageStore` / `StateStore` /
`EventStore` / `ToolExecutionStore` persistence traits (plus in-memory
implementations).

## Most users want `agent-sdk`

This crate is consumed through the [`agent-sdk`](https://crates.io/crates/agent-sdk)
façade, which re-exports these traits alongside the agent loop, providers, and
built-in tools. Depend on the façade unless you are building a narrow,
tools-only integration.

```toml
[dependencies]
agent-sdk = "0.9"
```

## Documentation

- API reference: <https://docs.rs/agent-sdk-tools>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
