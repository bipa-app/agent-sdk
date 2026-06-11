# agent-sdk-macros

[![crates.io](https://img.shields.io/crates/v/agent-sdk-macros.svg)](https://crates.io/crates/agent-sdk-macros)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-macros)](https://docs.rs/agent-sdk-macros)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

Ergonomic derive and declarative macros for the
[Agent SDK](https://github.com/bipa-app/agent-sdk):

- `#[derive(Tool)]` — derive a tool's `name` / `description` / `input_schema` /
  `tier` from a `#[tool(...)]` attribute; you implement only `ToolLogic`.
- `#[derive(TypedTool)]` — typed-`Input` validation with optional `schemars`
  schema derivation (`#[tool(schema = "derive")]`).
- `tool! { ... }` — define an inline, one-off tool without a named struct.
- `#[derive(ToolName)]` — collapse the strongly-typed tool-name enum boilerplate.

## Do not depend on this crate directly

These macros are re-exported from the [`agent-sdk`](https://crates.io/crates/agent-sdk)
façade (importing the `Tool` / `TypedTool` / `ToolName` traits also brings in
the matching derive). The generated code refers to `::agent_sdk::…` paths, so it
only compiles when used through the façade.

```toml
[dependencies]
agent-sdk = "0.9"
```

See `examples/derive_tool.rs` in the repository for all four macros in one agent.

## Documentation

- API reference: <https://docs.rs/agent-sdk-macros>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
