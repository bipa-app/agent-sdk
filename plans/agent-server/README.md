# Agent Server Replacement Plan Set

This directory is the local replacement planning set for the `agent-server` rewrite. The active planning view now lives under [`gpt-pass`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/README.md). [`legacy`](/Users/luizparreira/work/agent-sdk/plans/agent-server/legacy) is archive material from rejected or superseded plan passes.

## Global Decisions

- Rewrite work merges to `main`. Worktrees should branch from `origin/main`; `sdk/v2` is no longer the merge target for this rewrite.
- The repo is already a workspace. Further crate splits are allowed where they reduce coupling between SDK, storage, transport, and service-host concerns.
- `agent_tasks` is the sole execution authority. AMQP and Artemis-style systems can wake or relay work, but they do not own correctness.
- Tool execution is a task-owned lifecycle, not inline loop behavior. Root turns suspend at the tool boundary and resume from durable child outcomes.
- The server core runs one externally orchestrated turn at a time. `run_turn` is the authoritative boundary; `run()` remains convenience API only.
- `turn_snapshots` are replaced by immutable `turn_attempts` owned by tasks.
- Public replay and streaming are based on durable `AgentEventEnvelope` rows with server-owned per-thread ordering.
- The durable core remains storage-agnostic, but the plan now explicitly requires:
  - a production PostgreSQL backend
  - a local durable backend for desktop and CLI usage
  - a reference service or daemon host
  - a gRPC transport for desktop integration

## File Map

- [`legacy/`](/Users/luizparreira/work/agent-sdk/plans/agent-server/legacy): archived phase docs and earlier plan material
- [`claude-pass/`](/Users/luizparreira/work/agent-sdk/plans/agent-server/claude-pass): earlier exploratory review artifacts
- [`gpt-pass/README.md`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/README.md): current GPT pass index
- [`gpt-pass/project-overview.md`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/project-overview.md): current architectural overview
- [`gpt-pass/dependency-map.md`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/dependency-map.md): dependency and delivery graph
- [`gpt-pass/storage-and-transport-addendum.md`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/storage-and-transport-addendum.md): update covering concrete storage, service packaging, gRPC, and deployment targets
- [`gpt-pass/linear-project-audit-2026-04-08.md`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/linear-project-audit-2026-04-08.md): earlier Linear audit pass
- [`gpt-pass/visuals/architecture-atlas.html`](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/visuals/architecture-atlas.html): interactive architecture dashboard

## Scope

This replacement set is correctness-first rather than feature-first.

- In scope for v1:
  - durable execution journal
  - replay-safe turn processing
  - task-owned tool execution
  - confirmation durability
  - durable event replay
  - stable multi-agent hierarchy
  - concrete PostgreSQL support
  - concrete local durable storage support
  - reference service or daemon packaging
  - gRPC transport needed by the desktop client
- Explicitly deferred until the core is proven:
  - transport proliferation beyond required deployment targets
  - admin UI
  - generalized multi-tenant policy
  - speculative optimizations that weaken the durable source-of-truth model

## Adoption Rule

These files are the local draft planning source for the rewrite. When there is a conflict between older archived docs and the active GPT pass, the GPT pass wins.
