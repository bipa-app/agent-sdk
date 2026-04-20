# Linear Audit — 2026-04-08

This review compares the current Linear initiative and project state against the replacement architecture now planned in the repo and issue set.

## 1. Milestones Are Not Aligned

The current project milestones are still the rejected phase model:

- Phase 1: Scaffolding + Crypto + Threads
- Phase 2: Encrypted MessageStore
- Phase 3: Msgs Integration + Turn Dispatch
- Phase 4: Turn Processing Engine
- Phase 5: Event Pipeline + Streaming
- Phase 6: Tool Execution + Confirmation
- Phase 7: Task Journal + Crash Recovery
- Phase 8: AgentServer Builder + Integration

That sequence no longer matches the rewrite. The new plan is:

- Phase 0: Workspace Split and Server-Ready Crate Topology
- Phase 1: SDK Contract Lockdown for Durable Server Orchestration
- Phase 2: Task Journal and Lifecycle Core
- Phase 3: Conversation Persistence, Turn Attempts, and Completed-Turn Checkpoints
- Phase 4: Root Turn Worker and Tool-Task Dispatch
- Phase 5: Tool Runtime Workers, Confirmation, and Audit
- Phase 6: Events, Replay, and Streaming
- Phase 7: Subagent Hierarchy and Cancellation
- Phase 8: AMQP Outbox and GA Hardening

### Current state problem

- None of the current rewrite parent issues are attached to a project milestone.
- None of the child implementation issues are attached to a project milestone.
- The milestone descriptions themselves still encode rejected assumptions such as encrypted stores, gRPC-first slicing, turn dispatch through AMQP, and old `turn_snapshots` thinking.

### Recommendation

Replace the project milestones completely and use them as implementation-complete exits, not design-checkpoint exits.

## 2. Initiative And Project Overview Are Stale

### Initiative description today

The initiative description still says:

`Reusable Rust library for production-grade agent infrastructure: encrypted persistence, AMQP dispatch, durable execution, crash recovery, and gRPC transport.`

### Project description today

The project description still says:

`agent-server Rust library crate — encrypted stores, AMQP dispatch, durable task journal, gRPC transport. Library-first: consumers import and provide their own DB, cipher, and AMQP connection.`

### Why this is wrong now

- encryption and blind-index work is no longer the lead planning theme
- AMQP is no longer the execution backbone
- gRPC is not the architectural spine of the rewrite
- the core concepts that should appear are missing: `agent_tasks`, checkpoints, task-owned tool runtime, durable events, and durable subagent trees

### Recommendation

Update both overview texts to describe the actual rewrite:

- task-owned execution journal
- one-turn-at-a-time worker model
- completed-turn checkpoints
- child-task tool runtime
- durable event replay
- durable subagent hierarchy
- AMQP as advisory outbox only

## 3. Labeling Needs To Be More Informative

### Current counts

- `agent-server`: 61
- `Chore`: 55
- `component:backend`: 55
- `phase:build`: 51
- `phase:design`: 4

### Problems

- `agent-server` is useful because it groups the rewrite, but it does not help triage inside the rewrite.
- `component:backend` is redundant on every issue in an ENG project called `Server SDK`.
- `Chore` is too generic. It hides whether an issue is SDK contract, persistence, runtime, docs, testing, or operations.
- `phase:design` and `phase:build` are directionally useful but are incomplete on some parent issues and do not tell the reader what architectural surface the issue belongs to.
- stale canceled issues use only `agent-server`, which weakens archive readability.

### Recommendation

Keep one grouping label and replace the rest with more informative categories:

- keep: `agent-server`
- keep or normalize: `kind:design`, `kind:build`, `kind:docs`, `kind:test`
- add area labels:
  - `area:sdk-contract`
  - `area:journal`
  - `area:persistence`
  - `area:root-worker`
  - `area:tool-runtime`
  - `area:events`
  - `area:subagents`
  - `area:outbox`
  - `area:ops`
- optionally add `risk:correctness-critical` to issues that can compromise replay, recovery, or safe execution

If label count must stay low, drop `component:backend` and `Chore` first.

## 4. Dependency Modeling Is Correct At The Top, But Still Too Opaque

### What is good already

The parent phase graph is now materially better:

- Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
- Phase 4 -> Phase 5
- Phase 4 -> Phase 6
- Phase 4 -> Phase 7
- Phase 5 -> Phase 6
- Phase 5 -> Phase 8
- Phase 6 -> Phase 7
- Phase 6 -> Phase 8
- Phase 7 -> Phase 8

### What is still missing

- There is no single document that explains why those edges exist.
- There is no visual separation between critical path dependencies and optional complexity-increasing dependencies.
- There is no phase-exit matrix that says what must be proven before unlocking the next branch.
- There is no explicit statement of which later-phase work can be prototyped in isolation and which cannot.

### Recommendation

Add a dependency and delivery map that distinguishes:

- critical path needed for first durable end-to-end root turn:
  - 0 -> 1 -> 2 -> 3 -> 4
- mutation-safe tool runtime track:
  - 4 -> 5
- replay and streaming track:
  - 4 -> 6 with late enrichment from 5 and 7
- subagent track:
  - 4 -> 7 with replay dependency from 6
- latency and GA hardening track:
  - 5 + 6 + 7 -> 8

## 5. Important Missing Pieces

### Missing planning artifacts

- project overview document
- milestone realignment document
- dependency and phase-exit document
- invariant catalog
- scenario library for success and failure flows
- architecture visualization

### Missing consistency pass across current plan set

- local Phase 6, Phase 7, and Phase 8 markdown docs are behind the newer Linear replacement specs
- `data-model.md` still says persisted replay excludes ephemeral deltas unless promoted later, but the newer Phase 6 replacement spec says all current public event variants are replayable
- initiative and project descriptions are stale

### Missing implementation-level control documents

- exact foreign key and unique-index catalog by table
- default lease, heartbeat, retry, retention, and buffer policy values
- explicit redaction policy matrix by storage surface
- test matrix that maps failure scenarios to the phase where they are proven

## 6. Recommended Immediate Follow-Ups

1. Replace project milestones with the rewrite phase model and attach parent issues to them.
2. Update the initiative and project descriptions to the replacement architecture.
3. Normalize labels before the issue graph grows further.
4. Add the overview, dependency map, and architecture atlas to the repo and link them from the project overview.
5. Bring the local Phase 6, Phase 7, and Phase 8 markdown docs back in sync with the current Linear specs.
6. Add one invariant catalog and one scenario matrix before Phase 4 implementation expands the system surface further.
