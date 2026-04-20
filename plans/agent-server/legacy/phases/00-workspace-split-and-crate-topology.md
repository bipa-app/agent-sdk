## Summary

Zeroth replacement slice: convert the current single-crate repository into a workspace that can host the SDK rewrite and the future server implementation without forcing both efforts through one monolithic crate.

## Locked Decisions

- All rewrite PRs opened from this repository target `sdk/v2` as the base branch.
- The root manifest becomes a Cargo workspace manifest.
- The rewrite prefers a small number of purpose-built crates over one monolith or an over-fragmented graph.
- The server must depend on narrow SDK crates, not on an everything crate that drags the full loop, tools, and providers into every dependency edge.
- A compatibility-facing `agent-sdk` crate remains available as the user entrypoint, even if it re-exports from smaller internal crates.

## Target Workspace Shape

- `crates/agent-sdk-core`
  - ids, events, llm message types, turn inputs and outcomes, continuation payloads, shared serde contracts
- `crates/agent-sdk-tools`
  - tool traits, tool registry, tool runtime contracts, primitive tool abstractions
- `crates/agent-sdk-runtime`
  - authoritative `run_turn`, loop orchestration, context compaction hooks, runtime plumbing
- `crates/agent-sdk-providers`
  - provider traits and first-party provider implementations
- `crates/agent-server`
  - journal, persistence, workers, replay, and transport-adjacent server code
- `crates/agent-sdk`
  - public facade crate that re-exports the supported SDK surface

The exact crate names may adjust during implementation, but the dependency direction is locked: `agent-server` depends on narrower SDK crates, and the facade depends on them from the outside in.

## Tasks

### 1. Workspace Manifest and Shared Cargo Policy
- Convert the repo root to a virtual workspace manifest.
- Move package metadata, shared dependencies, and lint policy to workspace scope where practical.
- Define a crate layout that supports incremental extraction without breaking every downstream path at once.

### 2. Foundation Crate Extraction
- Pull shared contracts out of the monolith first:
  - ids
  - llm messages
  - event envelopes
  - turn input and outcome types
  - continuation payloads
- Keep these crates free of provider implementations and loop-specific runtime state.

### 3. Runtime, Tools, and Provider Boundaries
- Separate the loop/runtime code from tool definitions and provider implementations.
- Remove circular assumptions that today exist because everything lives in one crate.
- Make the future server depend on tool/runtime contracts rather than concrete monolithic wiring.

### 4. Server Crate Insertion
- Add the `agent-server` crate to the same workspace early, even if it starts as planning-aligned scaffolding.
- Ensure the crate graph allows close SDK/server development without temporary path hacks.

### 5. Facade, Examples, and CI Migration
- Rebuild the public `agent-sdk` facade on top of the new crate graph.
- Move examples, tests, and workspace checks to the new layout.
- Keep a clear migration path for users who still import `agent_sdk` from one crate.

## Acceptance

- The workspace builds from the root with one Cargo entrypoint.
- The planned `agent-server` crate can depend on narrow SDK crates instead of the current monolith.
- Shared contracts are isolated enough that Phase 1 does not need to reach across unrelated runtime/provider code.
- Examples, tests, and CI operate against the workspace layout.
- Rewrite PRs and stacked branches are documented to target `sdk/v2`.

## Dependencies

- None. This phase must land before the remaining rewrite phases are implemented in earnest.
