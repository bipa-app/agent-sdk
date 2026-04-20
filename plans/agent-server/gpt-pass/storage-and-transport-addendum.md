# Storage, Service, and Transport Addendum

This addendum updates the replacement plan to match two realities:

1. the actual code already landed in this repo
2. the real deployment targets now include remote services, app-server embedding, and local desktop or CLI usage

The earlier replacement phases were correct about journal ownership, turn boundaries, checkpoints, tool runtime, and replay. What they did not pin down tightly enough was how the system becomes durable in a real database, how local persistence works, and where the networked server boundary actually lives.

## 1. Repo Reality Check

The current workspace already contains:

- `agent-sdk-core`
- `agent-sdk-tools`
- `agent-sdk-providers`
- `agent-server`
- `agent-sdk`

The current `agent-server` crate is a library crate, not a network service. The code that exists today already proves several important parts of the model:

- a task journal and state machine under `crates/agent-server/src/journal.rs`
- durable-friendly store traits for tasks, threads, message projections, turn attempts, and checkpoints
- in-memory reference implementations of those stores
- a Phase 4 root-turn worker that can:
  - execute a text-only turn
  - commit completed-turn checkpoints
  - suspend at the tool boundary
  - spawn `tool_runtime` child tasks

The code does not yet contain:

- a PostgreSQL store implementation
- a local on-disk durable store implementation
- SQL migrations or retention jobs against a concrete database
- a gRPC server
- an HTTP server or SSE surface
- a reference daemon or service host

This matters because the current plan can no longer pretend these are late polish items. They are part of making the model real.

## 2. Linear Snapshot On 2026-04-10

Using the current Linear project state:

- total issues: 66
- done: 30
- backlog: 27
- todo: 2
- in review: 1
- canceled legacy issues: 6

By phase parent:

- Phase 0: done, all child issues done
- Phase 1: done, all child issues done
- Phase 2: child issues effectively complete, parent status still not advanced
- Phase 3: child issues effectively complete, parent status still not advanced
- Phase 4: partially in flight
- Phase 5: planned at the parent level but still missing child implementation breakdown
- Phase 6 to Phase 8: broken down, but still backlog

This means the planning gap is no longer about the early journal model. It is now about making the later runtime usable in actual deployments.

## 3. Deployment Targets We Must Support

The rewrite is not serving only one environment.

### Remote internal agent service

- runs on Kubernetes
- uses PostgreSQL
- needs a real multi-process durable backend

### App-facing server integration

- runs inside the main application server
- uses PostgreSQL
- may use Artemis or another durable worker system for wakeup and worker execution

### Local coding agent

- runs on the end-user machine
- needs local durable state
- desktop app should talk to it over gRPC

These targets change the plan in a concrete way: `agent-server` is no longer just an embeddable runtime library. It is both:

- a reusable durable runtime library
- a reference service or daemon host that exposes the runtime over transport

## 4. New Required Planning Tracks

### Track A: concrete storage backends

The storage-agnostic traits stay. That part of the architecture is still correct. But v1 now has mandatory backend requirements:

- PostgreSQL backend for production and server deployments
- local durable backend for desktop and CLI deployments

The PostgreSQL backend must cover:

- `agent_tasks`
- `threads`
- `messages`
- `turn_attempts`
- `turn_checkpoints`
- later `events`
- later `tool_executions`
- later `tool_call_log`
- later `msgs`

It must also define:

- migrations
- indexes
- transactional commit boundaries
- compare-and-swap lease transitions
- stale-lease sweeps
- retention primitives

The local durable backend should not be treated as an afterthought. The core question is whether we want:

- SQLite on the local filesystem
- raw files plus directories
- another embedded store

The current recommendation is:

- default to an embedded database model, most likely SQLite-on-filesystem
- use raw files only if an ADR proves they are materially simpler without weakening recovery, replay, or concurrency control

Reason: the system model already depends on ordered commits, atomic state transitions, indexed replay, and lease-style recovery. Those align naturally with an embedded database and poorly with ad hoc file trees.

### Track B: service or daemon packaging

The runtime needs a host process shape. Today `agent-server` is a library crate. That is useful, but it is not sufficient for the real consumers we discussed.

The plan must now include a reference host that owns:

- process boot
- configuration loading
- store wiring
- worker pool startup
- sweep loops
- outbox or relay workers
- health and readiness checks
- metrics and tracing integration
- shutdown and restart semantics

Recommended direction:

- keep `agent-server` as the durable runtime library
- add a dedicated service-host crate once the interfaces settle
- keep transport concerns out of the pure runtime wherever possible

This keeps local embedding possible while still giving the team a real server binary to run in Kubernetes or behind desktop clients.

### Track C: gRPC-first transport

We now have a concrete gRPC use case: the desktop app should talk to a local agent server over gRPC.

That means transport is no longer “nice to have.” The plan should explicitly target:

- gRPC as the first supported network transport
- event streaming over committed server envelopes
- request and stream semantics that preserve turn-event ordering guarantees

HTTP and SSE can remain later decisions. They are not banned, but they are no longer the first transport to plan around.

### Track D: Artemis and external durable worker integration

The app-facing deployment may use Artemis for durable workers. That can fit the model, but only if the ownership rule stays intact:

- `agent_tasks` remains the source of truth
- external queues only wake workers or route work
- no external worker system is allowed to become the authoritative lifecycle owner

The plan therefore needs an explicit integration contract for:

- wakeup semantics
- idempotent re-delivery
- lease re-check on dequeue
- duplicate-safe worker execution
- startup backfill from the durable journal

## 5. How These Tracks Fit The Existing Phases

The Phase 0 to Phase 8 runtime ladder still largely holds. What changes is that we need concrete delivery tracks around it.

### Runtime phases remain the core correctness path

- Phase 0: workspace and crate topology
- Phase 1: SDK contract and turn boundary
- Phase 2: journal and lifecycle authority
- Phase 3: persistence model, turn attempts, checkpoints
- Phase 4: root worker
- Phase 5: tool runtime
- Phase 6: events and replay
- Phase 7: subagents
- Phase 8: outbox and GA hardening

### Storage track starts before “production-ready” claims

We cannot claim durable correctness solely from in-memory reference stores. The concrete storage track should begin as soon as the core store contracts are stable enough to map onto SQL transactions.

Recommended staging:

1. freeze the store contract and commit invariants by the end of Phase 3
2. implement PostgreSQL support while Phase 4 and Phase 5 are still being completed
3. add the local durable backend before desktop integration work is treated as real

### Service and transport track starts after the root runtime is believable

We should not expose a public transport before the worker and replay model are coherent. Recommended staging:

1. root worker semantics first
2. event envelope and replay contract next
3. gRPC contract on top of committed runtime state
4. HTTP or SSE only if needed by a real consumer

## 6. Missing Pieces That Must Be Added To The Plan

These are the concrete plan gaps that still need explicit phase or task coverage:

1. PostgreSQL schema and migration plan
2. local durable storage ADR and implementation plan
3. service-host crate topology and lifecycle plan
4. gRPC API contract and streaming model
5. Artemis integration rules
6. environment-specific configuration model
7. health, readiness, tracing, and metrics plan

## 7. Recommended Architectural Decisions

### Keep the core runtime storage-agnostic

This preserves embeddability and keeps the journal model independent from a single backing store.

### Make PostgreSQL the required production backend

This is the backend that proves the real multi-process semantics.

### Use an embedded local durable backend for desktop and CLI

Prefer SQLite-on-filesystem unless an ADR demonstrates a better option without giving up ordered replay and atomic commit.

### Keep service-host concerns outside the pure runtime core

A reference daemon or service host should wire runtime, storage, and transport together, but it should not force local embedders to import network-server code just to run a worker.

### Make gRPC the first-class network transport

This matches the desktop integration requirement and forces us to define ordered stream semantics clearly instead of hand-waving transport behavior.

### Treat Artemis, AMQP, and similar systems as advisory

They can schedule and wake work. They cannot own correctness.

## 8. What This Changes Immediately

The replacement plan should now be read with these additions:

- “stable multi-agent server” includes real database-backed durability, not only in-memory proof models
- the rewrite needs a reference host process, not only library APIs
- gRPC is part of the required delivery surface
- local durable storage is a first-class problem, not a later portability detail

If any future phase or issue description ignores these constraints, it is incomplete.
