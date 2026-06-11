# agent-sdk-foundation

[![crates.io](https://img.shields.io/crates/v/agent-sdk-foundation.svg)](https://crates.io/crates/agent-sdk-foundation)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-foundation)](https://docs.rs/agent-sdk-foundation)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

Shared, dependency-light contract types for the [Agent SDK](https://github.com/bipa-app/agent-sdk):
thread/message IDs, the `AgentEvent` stream, LLM message and request/response
types, token usage, turn outcomes, and the redaction policy. These are the wire
and domain primitives the rest of the workspace builds on.

## Most users want `agent-sdk`

This is a foundational building block. Unless you are implementing a custom
provider, tool, or store against the raw contracts, depend on the
[`agent-sdk`](https://crates.io/crates/agent-sdk) façade instead — it re-exports
the types you need.

```toml
[dependencies]
agent-sdk = "0.9"
```

## Documentation

- API reference: <https://docs.rs/agent-sdk-foundation>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
