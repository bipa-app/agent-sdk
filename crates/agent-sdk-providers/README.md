# agent-sdk-providers

[![crates.io](https://img.shields.io/crates/v/agent-sdk-providers.svg)](https://crates.io/crates/agent-sdk-providers)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-providers)](https://docs.rs/agent-sdk-providers)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

The `LlmProvider` trait, streaming primitives, schema-validated structured
output (`run_structured`), and the first-party provider implementations for the
[Agent SDK](https://github.com/bipa-app/agent-sdk).

## Providers

Each provider is an opt-in Cargo feature (no default features when depended on
directly):

| Feature | Provider |
|---------|----------|
| `anthropic` | Anthropic (Claude) |
| `openai` | OpenAI Chat Completions |
| `openai-codex` | OpenAI Codex / Responses (WebSocket) |
| `gemini` | Google Gemini |
| `vertex` | Google Vertex AI |
| `cloudflare` | Cloudflare AI Gateway |

## Most users want `agent-sdk`

Provider selection is normally done through the
[`agent-sdk`](https://crates.io/crates/agent-sdk) façade's feature flags, which
re-exports these providers under `agent_sdk::providers`. Depend on this crate
directly only to implement a custom provider against the `LlmProvider` trait.

```toml
[dependencies]
agent-sdk = { version = "0.9", features = ["openai", "gemini"] }
```

## Documentation

- API reference: <https://docs.rs/agent-sdk-providers>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
