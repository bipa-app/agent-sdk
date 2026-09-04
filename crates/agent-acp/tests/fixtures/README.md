# Recorded buzz-acp wire fixtures

Golden client→agent request **params**, captured from the buzz-acp harness
source at block/buzz `v0.5.2` (rev `3e48f1b2365d326ee1c9582448d86a99b44ecd5d`)
— the release the Bipa team relay is deployed from. These files are the
compatibility contract this server is tested against. Originally captured
at rev `7e34bee`; re-verified at `3e48f1b` against a live harness run
(satoshi `scripts/buzz-e2e.sh run`, 2026-09-04): every method's shape was
identical, `initialize` byte-for-byte.

Provenance (all paths in block/buzz at `3e48f1b`):

- `initialize.json` — verbatim shape of `build_initialize_params()` +
  `build_client_capabilities()` (`crates/buzz-acp/src/acp.rs:124`, `:390`).
  `clientInfo.version` is the workspace version at that rev (`0.1.0`).
- `session_new.json` — the shape sent by `session_new_full()`
  (`acp.rs:629`): `cwd`, `mcpServers`, optional `systemPrompt`, and an
  optional `_meta` (session title) the harness omits in our deployment.
  Values are representative; keys/nesting are the contract.
- `session_prompt.json` — the shape of `build_prompt_params()`
  (`acp.rs:1953`); multi-block form exercises the slash-command pass-through
  path (`session_prompt_blocks_with_idle_timeout`, `acp.rs:747`).
  `<SESSION_ID>` is substituted by the test with the id returned from
  `session/new`.
- `session_cancel.json` — the shape of `session_cancel()` (`acp.rs:818`).

When bumping the relay's pinned rev, re-verify these builders and update the
rev hash above. The satoshi repo's `scripts/buzz-e2e/captured/` holds the
LIVE-recorded counterparts (real values, paths redacted), refreshed by the
same harness run.
