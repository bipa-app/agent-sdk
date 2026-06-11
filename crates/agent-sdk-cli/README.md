# agent-sdk-cli

[![crates.io](https://img.shields.io/crates/v/agent-sdk-cli.svg)](https://crates.io/crates/agent-sdk-cli)
[![docs.rs](https://img.shields.io/docsrs/agent-sdk-cli)](https://docs.rs/agent-sdk-cli)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE)

Developer-experience CLI for the [Agent SDK](https://github.com/bipa-app/agent-sdk).
It installs the binary `agent-sdk`, which can run an agent directly and scaffold
the local observability stack.

## Install

```bash
cargo install agent-sdk-cli
```

## Usage

```bash
# Run a single prompt (reads ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=your_key agent-sdk run "Explain Rust ownership in two sentences."

# Interactive chat (keeps history for the session)
ANTHROPIC_API_KEY=your_key agent-sdk chat

# Materialize the local Langfuse + OTel collector compose stack
agent-sdk local-langfuse init

# Check docker, ports, and destination writability
agent-sdk doctor
```

## Documentation

- API reference: <https://docs.rs/agent-sdk-cli>
- Repository, README, and cookbook: <https://github.com/bipa-app/agent-sdk>

## License

MIT. See [LICENSE](https://github.com/bipa-app/agent-sdk/blob/main/LICENSE).
