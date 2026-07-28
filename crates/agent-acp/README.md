# agent-acp

ACP (Agent Client Protocol) stdio transport for the Agent SDK: expose an
agent as a newline-delimited JSON-RPC server, as spoken by the
[buzz-acp](https://github.com/block/buzz) harness.

The harness spawns the agent process and drives it over stdio:
`initialize` → `session/new` → `session/prompt` (blocking, resolves with a
`stopReason`), consuming streamed `session/update` notifications, cancelling
via the `session/cancel` notification. This crate owns the wire and dispatch
loop; behavior is supplied through the `PromptHandler` seam (superseded by
the durable-host `AcpBackend` in later milestones).

Compatibility contract: the recorded fixture set in `tests/fixtures/`,
captured from block/buzz rev `7e34bee` — see the fixtures README for
provenance and the crate docs for the protocol-version stance (including why
the published `agent-client-protocol` crate is not used for wire types).

Part of the Satoshi Buzz Agent initiative; design and contracts live in the
satoshi repo under `plans/buzz-acp-integration.md` and
`plans/buzz-acp-contracts.md`.
