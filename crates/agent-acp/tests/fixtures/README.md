# Recorded buzz-acp wire fixtures

Golden client→agent request **params**, captured from the buzz-acp harness
source at block/buzz rev `7e34bee62cacaa9d8a96c14d5892a471b59a1983` — the
revision the Bipa team relay is deployed from. These files are the
compatibility contract this server is tested against.

Provenance (all paths in block/buzz at that rev):

- `initialize.json` — verbatim shape of `build_initialize_params()` +
  `build_client_capabilities()` (`crates/buzz-acp/src/acp.rs:124`, `:96`).
  `clientInfo.version` is the workspace version at that rev (`0.1.0`).
- `session_new.json` — the shape sent by `session_new_full()`
  (`acp.rs:560`): `cwd`, `mcpServers`, optional `systemPrompt`. Values are
  representative; keys/nesting are the contract.
- `session_prompt.json` — the shape of `build_prompt_params()`
  (`acp.rs:1764`); multi-block form exercises the slash-command pass-through
  path (`session_prompt_blocks_with_idle_timeout`). `<SESSION_ID>` is
  substituted by the test with the id returned from `session/new`.
- `session_cancel.json` — the shape of `session_cancel()` (`acp.rs:742`).

When bumping the relay's pinned rev, re-verify these builders and update the
rev hash above. M0.4 (ENG-9398) additionally captures fixtures from a LIVE
harness run, superseding representative values with recorded ones.
