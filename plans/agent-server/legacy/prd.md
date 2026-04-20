# Agent Server — Replacement PRD

## Problem Statement

The current server planning set is not stable enough to guide implementation. The main blockers are:

1. Execution ownership is split across AMQP, turn rows, and a later task journal.
2. The server plan assumes SDK contracts that do not exist yet.
3. The current repo layout is a single crate, which makes close SDK/server development harder than it needs to be.
4. Tool execution is still modeled as inline loop behavior, which blocks the turn worker and makes lifecycle, replay, and restart semantics too weak.
5. Event streaming and replay are not defined against a durable source of truth.
6. Isolation and policy enforcement are not strong enough for a multi-agent server.

## Product Goal

Build a Rust workspace that contains a stable SDK surface plus a Rust library crate, `agent_server`, that lets a host application run many durable agent threads safely. The correctness bar is:

- a crate graph that lets the SDK and the server evolve in close coordination
- one authoritative task journal
- one model-turn worker at a time per thread
- lightweight task-owned tool execution instead of inline blocking tool calls
- exact replay semantics for persisted events
- durable confirmation pause/resume
- deterministic recovery from crash between completed turns
- stable parent/child task hierarchy for subagents

## Core Product Decisions

### 1. Workspace and Delivery Branch

- This rewrite is developed on top of the `sdk/v2` feature branch.
- Rewrite PRs from this repository target `sdk/v2`, not `main`.
- The repo is split into a workspace with smaller SDK crates plus `agent-server`.

### 2. Execution Authority

- `agent_tasks` is the source of truth for scheduling, leases, retries, cancellation, confirmation pause/resume, recovery, and subagent hierarchy.
- AMQP is optional transport for wakeup and fanout only.
- A worker must always re-check task state in Postgres before acting on a queue message.

### 3. Turn Boundary

- The server executes exactly one externally orchestrated turn at a time.
- The core loop is `run_turn`-style orchestration, not `run()`.
- `turn_attempts` records are immutable execution log rows for each turn attempt.
- In server mode, `run_turn` must be able to surface tool work as typed child-task handoff instead of executing tools inline.

### 4. Thread Concurrency

- A thread may have at most one runnable root task at a time.
- Concurrency exists across threads and child tasks, not across root turns on the same thread.
- Tool runtime work uses child tasks beneath the root task, not extra active roots.

### 5. Event Contract

- Public event streaming and replay use `AgentEventEnvelope`, not bare `AgentEvent`.
- Sequence numbers are monotonic per thread across restarts and retries.
- Deltas may be streamed live, but only persisted envelopes are replayable.

### 6. Tool Runtime Safety

- Tool execution is modeled as a lightweight task lifecycle owned by `agent_tasks`.
- Root turn workers do not block on inline tool execution in server mode.
- Each tool runtime task must support lease ownership, retry rules, cancellation, and restart-safe recovery.
- Tool execution intent must be durably recorded before side-effecting execution begins.
- Resume-time policy checks are authoritative; a previously approved tool may still be blocked.
- `listen()` and long-running async tools must persist enough prepared-operation state to cancel or resume safely.

### 7. Isolation

- Filesystem root is a hard sandbox boundary with symlink containment.
- MCP tools are namespaced and treated as untrusted by default.
- Subagents inherit parent depth, concurrency, and policy context.

## Required SDK Contract Changes

- workspace-friendly crate boundaries for core types, tools, runtime, providers, and server code
- externally seeded event sequencing
- structured turn outcome metadata with model/provider provenance
- externalized tool-batch handoff for server mode
- durable continuation serialization as a first-class server contract
- richer tool audit callbacks
- explicit `ExecutionContextFactory` boundary for async workers
- fail-closed persistence hooks for side-effecting execution

## Consumers

| Consumer | Integration Style | Special Concern |
| --- | --- | --- |
| Bipa main agent | in-process library | gRPC loopback tools, encrypted persistence |
| Coding agent | standalone binary | filesystem sandbox, subagents, MCP |
| Satoshi/internal agent | standalone binary | background durability, event replay |

## Out of Scope for V1

- experimentation and A/B assignment
- MCP schema registry and provenance persistence
- HTTP/SSE/WebSocket transport surface
- shared multi-tenant database design
- admin console

## Acceptance Criteria

- crash after any completed turn resumes from the latest checkpoint
- a disconnected client can replay all persisted envelopes after `after_sequence`
- same-thread concurrent submissions do not create two runnable root tasks
- tool runtime tasks can be retried or restarted without pretending inline loop state still exists
- subagent trees survive cancellation and recovery as task hierarchy
- queue loss or duplication affects latency only, never correctness
