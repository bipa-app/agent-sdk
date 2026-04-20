# Agent Server Rewrite Overview

## What Is Being Built

The rewrite is building a stable agent server on top of the SDK workspace, not a thin transport wrapper around the current loop. The target system is a correctness-first execution platform where durable task state, completed-turn checkpoints, replayable events, and restart-safe tool runtime all work together under one model.

The central architectural decision is that `agent_tasks` owns execution. Threads and messages are conversation projections. Workers do not pull truth from AMQP, channels, or ad hoc loop state. They acquire durable tasks, rebuild execution context from persisted state, run one turn boundary at a time, and hand control back to the journal through guarded state transitions.

This rewrite also changes the SDK contract so the server can externalize tool work instead of blocking inside the loop. `run_turn` becomes a direct async server boundary with an explicit event flush barrier, typed tool-runtime handoff, versioned continuation state, and richer audit surfaces. The server does not rely on `run()` as its core orchestration path.

The rewrite is also no longer only about an embeddable library. It must support three concrete deployment shapes:

- a remote internal service on Kubernetes backed by PostgreSQL
- an app-facing server integration backed by PostgreSQL and potentially Artemis-driven workers
- a local coding-agent daemon on the user machine with local durable storage and a gRPC API for the desktop app

## Core Architectural Bets

### 1. Task-owned execution

- `agent_tasks` is the sole authority for scheduling, retries, leases, recovery, confirmation pause or resume, parent waiting states, tool runtime work, and later subagent trees.
- One active root task per thread is enforced in the database.
- Parent tasks waiting on child tool tasks move to `waiting_on_children` and release their worker lease.

### 2. Turn-level durability

- One externally orchestrated turn is the unit of durable execution.
- `turn_attempts` is the immutable execution log.
- `turn_checkpoints` stores one full-message checkpoint after each completed turn.
- Recovery always resumes from the latest completed checkpoint, never from half-written message or state stores.

### 3. Task-owned tool runtime

- Tool execution becomes child-task lifecycle, not inline loop behavior.
- Side-effecting tools persist durable execution intent before execution.
- Confirmation state lives on tool runtime tasks, not on the root loop.
- Parent resume happens only from durable child outcomes.

### 4. Durable event contract

- `events` stores replayable `AgentEventEnvelope` rows with server-owned per-thread sequences.
- Live streaming is a tail of the committed log, not an independent truth path.
- The server can guarantee that `TurnOutcome` becomes visible only after the current turn boundary's events have reached an explicit completion barrier.

### 5. Durable multi-agent hierarchy

- Subagents become child threads plus task trees.
- Parent threads receive summary progress and final results with durable child refs.
- Cancellation, policy inheritance, and recovery operate across the full descendant tree.

### 6. AMQP as latency layer only

- `msgs` is advisory outbox, not execution truth.
- Broker loss or duplication can change latency, not correctness.
- Cross-instance wakeup and watch fanout still re-check durable task state and replay from `events`.

### 7. Storage-agnostic core, concrete mandatory backends

- The durable core remains storage-agnostic at the trait boundary.
- PostgreSQL is a required production backend, not an optional future adapter.
- A local durable backend is also required for desktop and CLI usage.
- Local persistence should default to an embedded transactional store, not ad hoc file metadata, unless an ADR proves otherwise.

### 8. Reference service-host plus gRPC boundary

- `agent-server` must remain usable as a library.
- The rewrite also needs a real host process shape for service and daemon use.
- gRPC is the first required transport because the desktop app needs to talk to a local server process.
- HTTP and SSE remain possible later, but they are no longer the primary transport assumption.

## System Shape

### Current workspace today

- `agent-sdk-core`: IDs, events, messages, turn inputs and outcomes, continuation payloads.
- `agent-sdk-tools`: tool contracts, registries, primitive tool abstractions, runtime metadata.
- `agent-sdk-providers`: provider traits and implementations.
- `agent-server`: journal, persistence traits, workers, replay model, and server-side runtime integration.
- `agent-sdk`: compatibility facade.

### Likely next crate or module additions

- a PostgreSQL storage implementation
- a local durable storage implementation
- a service-host crate or binary target
- a gRPC transport surface once the runtime and replay contracts are ready

### Durable server components

- Root turn worker
- Tool runtime worker
- Event committer and live tail hub
- Outbox relay worker
- Retention janitor
- Recovery sweeper
- Service-host bootstrap and lifecycle manager
- gRPC server surface for local and remote clients

### Durable data model

- `threads`
- `messages`
- `agent_tasks`
- `turn_attempts`
- `turn_checkpoints`
- `tool_executions`
- `tool_call_log`
- `events`
- `msgs`

## What Each Phase Unlocks

### Phase 0

Makes the repo structurally capable of hosting the SDK rewrite and server work in separate crates on `main`-based worktrees without forcing the server, storage, and transport surfaces to evolve in one crate.

### Phase 1

Makes the SDK safe for server orchestration by fixing `run_turn`, continuation, tool handoff, event ordering, and audit surfaces.

### Phase 2

Creates the durable execution journal and task lifecycle so workers have one source of truth.

### Phase 3

Adds conversation projections, immutable turn attempts, and completed-turn checkpoints so crash-safe turn continuity works.

### Phase 4

Adds the first real root turn worker that runs one turn, stages state, commits only at completed-turn boundaries, and suspends at the tool boundary.

### Phase 5

Moves tools into durable child-task lifecycle with restart-safe execution, confirmation, and audit.

### Phase 6

Makes event replay durable and makes live streaming a committed tail over that log.

### Phase 7

Turns subagents into durable child threads and descendant task trees with inherited controls.

### Phase 8

Adds AMQP as an outbox-backed latency layer and finishes GA operational hardening.

## Explicit Non-Goals For V1

- generalized multi-tenant policy
- MCP provenance registry storage
- admin UI
- transport proliferation beyond required deployment targets
- any optimization that weakens the durable source-of-truth model

## Reading Order For Teammates

1. [README.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/README.md)
2. [project-overview.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/project-overview.md)
3. [dependency-map.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/dependency-map.md)
4. [storage-and-transport-addendum.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/storage-and-transport-addendum.md)
5. [linear-project-audit-2026-04-08.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/linear-project-audit-2026-04-08.md)
6. [architecture-atlas.html](/Users/luizparreira/work/agent-sdk/plans/agent-server/gpt-pass/visuals/architecture-atlas.html)

## Current Documentation Gaps

- The Linear initiative and project descriptions are still describing the rejected earlier architecture.
- The current project milestones still reflect the stale pre-rewrite phase model.
- Phase 5 is still missing its detailed child implementation breakdown in Linear.
- The repo has the durable core abstractions but still lacks concrete PostgreSQL, local durable storage, service-host, and gRPC plan detail unless the addendum is read alongside this file.
- There is no single invariant catalog or scenario library yet.
