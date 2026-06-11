# agent-sdk-otel

[![crates.io](https://img.shields.io/crates/v/agent-sdk-otel.svg)](https://crates.io/crates/agent-sdk-otel)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-otel)](https://docs.rs/agent-sdk-otel)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

OpenTelemetry bootstrap helpers for the
[Agent SDK](https://github.com/bipa-app/agent-sdk): OTLP exporter wiring, trace
context propagators, and a process-wide payload-capture gate that pairs with the
SDK's `otel` feature.

The [`agent-sdk`](https://crates.io/crates/agent-sdk) crate emits the spans and
metrics (under its own `otel` feature); this crate installs a global tracer
provider so they are exported. Use it when you want batteries-included OTLP setup
rather than configuring `opentelemetry_sdk` by hand.

```toml
[dependencies]
agent-sdk = { version = "0.9", features = ["otel"] }
agent-sdk-otel = "0.9"
```

See the `otel` example in the repository and
`crates/agent-sdk/docs/observability/LANGFUSE.md` for a full walkthrough.

## Documentation

- API reference: <https://docs.rs/agent-sdk-otel>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
