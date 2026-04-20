# Agent Server - Unresolved Questions and Clarifications Needed

> These questions were surfaced during the 2026-04-08 review pass.
> Each question blocks or de-risks specific implementation decisions.
> Priority: P0 = blocks current phase, P1 = blocks next phase, P2 = blocks later phase

---

## A. SDK Contract Boundary (Phase 1)

### A1. [P0] What does "strict server durability" mean concretely for `run_turn`?

Phase 1 Task 1 says: "Make strict server durability explicit in the authoritative path."

But the plan does not define what "strict" means:
- Does `run_turn` guarantee that ALL events are persisted before returning `TurnOutcome`?
- Or does it guarantee they are committed to an in-process buffer that the server can then persist?
- If the server crashes between `run_turn` returning and the server persisting events, what is the recovery contract?

**Why this matters**: If `run_turn` is responsible for durability, the SDK needs a persistence hook. If the server is responsible, the SDK just needs to guarantee ordering.

### A2. [P0] How does the execution-mode input work for `run_turn`?

Phase 1 Task 1 says the SDK should distinguish "local inline tool execution" from "server externalized tool-task handoff."

- Is this a runtime flag? A generic parameter? A trait implementation?
- Does the caller pass a strategy object, or does `run_turn` check a mode enum?
- How does this interact with `AgentConfig` vs `AgentDefinition`?

### A3. [P1] What is the serialization format for `AgentContinuation`?

Phase 1 Task 4 says: "Document serialization, versioning, and compatibility expectations."

- Is this `serde` JSON? `bincode`? Protobuf?
- What versioning strategy? Schema version field? Envelope wrapper?
- What happens when a continuation was serialized with SDK v1 and the server is now running SDK v2?

---

## B. Task Journal (Phase 2)

### B1. [P1] What are the retry semantics for root tasks vs tool tasks?

Phase 2 Task 3 mentions recovery rules, but doesn't specify:
- How many retries before a root task is permanently failed?
- Is `max_attempts` configurable per task kind?
- What is the backoff strategy between retries?
- Does the retry count reset when the task kind changes (e.g., root -> child tool)?

### B2. [P1] What is the lease duration and heartbeat interval?

The data model has `lease_expires_at` and `last_heartbeat_at`, but:
- What is the default lease duration? 30 seconds? 5 minutes?
- How frequently should workers heartbeat?
- What happens if a heartbeat fails due to transient DB error? Does the worker abort or retry the heartbeat?

### B3. [P1] How does `acquire_runnable_task` prevent thundering herd?

With multiple worker instances competing for tasks:
- Is acquisition `SELECT ... FOR UPDATE SKIP LOCKED`?
- Is there a worker affinity or sticky routing strategy?
- What's the expected contention pattern with N workers and M threads?

### B4. [P2] What does `context_blob` actually contain?

The data model says it's "the durable source for rebuilding worker execution context."
- Is this the `AgentConfig`? The resolved `AgentDefinition`? The full tool registry?
- How large can this blob get?
- Is it encrypted at rest (like message content)?

---

## C. Persistence Layer (Phase 3)

### C1. [P1] What goes into `messages.content` and how is it serialized?

The data model says "stores the serialized `llm::Message` body, including tool, image, document, thinking, and redacted-thinking blocks."

- Is this `serde_json` serialization of the SDK `Message` type?
- How are binary payloads (images, documents) handled? Inline base64 or external reference?
- What happens to `RedactedThinking` blocks? Are they stored as-is (opaque bytes)?
- Is `content` encrypted? The column is `BYTEA` which suggests yes.

### C2. [P1] What is the compaction strategy for `replace_history`?

Phase 3 says "Preserve transactional `replace_history` semantics for compaction."

- When does compaction happen? After every turn? Periodically? On demand?
- What triggers it? Context window pressure? Message count threshold?
- Does compaction run inside the turn commit transaction or separately?
- If compaction fails, does the turn fail?

### C3. [P2] How large are `turn_checkpoints` and what is the retention policy?

Each checkpoint stores `messages_snapshot` and `agent_state_snapshot`.
- For a long conversation (100+ turns), how large does a snapshot get?
- The data model says "V1 recovery always resumes from the latest completed-turn checkpoint" - does this mean we only need the LATEST checkpoint?
- Can old checkpoints be garbage-collected?
- Phase 8 mentions "retention policy for checkpoints" but no specifics.

---

## D. Turn Worker (Phase 4)

### D1. [P1] What is `AgentDefinition` and how does it differ from `AgentConfig`?

Phase 4 Task 1 says: "Define `AgentDefinition` as server-owned policy, not a thin mirror of `AgentConfig`."

- What fields does `AgentDefinition` have that `AgentConfig` doesn't?
- Where is it stored? In the database? A config file? Hardcoded per consumer?
- How is it resolved for a given task? By thread metadata? By task kind?

### D2. [P1] What happens when `run_turn` returns `ModelContextWindowExceeded`?

The current SDK agent loop handles this with compaction and retry. But in server mode:
- Does the root turn worker handle compaction, or is it delegated?
- Does the `turn_attempt` get marked as failed and a new attempt created?
- Is there a limit to compaction retries?

### D3. [P2] How does the "resume from completed tool results" path work concretely?

Phase 4 Task 4 says: "Define the resume contract for continuing a turn after completed child tool tasks provide results."

- Does the root worker call `run_turn` again with tool results injected?
- Is there a separate `resume_turn` method?
- How are tool results marshaled from child task `result_blob` back into SDK-compatible tool result messages?

---

## E. Tool Runtime (Phase 5)

### E1. [P2] How are tool concurrency limits enforced?

Phase 5 says: "Sibling tool tasks run with bounded parallelism."

- What is the bound? Per-thread? Per-worker? Global?
- Is this enforced by the journal (limit on concurrent running child tasks) or by the worker pool?
- Can different tools have different concurrency limits?

### E2. [P2] What makes an `AsyncTool` "qualified" for server mode?

Phase 5 says: "Allow `AsyncTool` only when durable `operation_id` plus fresh-worker status polling is enough to resume safely."

- Who decides if an async tool is qualified? The tool itself? A server-side registry?
- What metadata does a tool need to expose to be "qualified"?
- Is there a trait method or annotation for this?

### E3. [P2] What is the `ListenExecuteTool.listen()` -> `execute()` flow in server mode?

The plan describes a prepare/confirm/execute flow, but:
- Does `listen()` happen in the tool worker? Or before the tool task is created?
- How is the `prepared_snapshot` structured?
- What is the expiry policy for prepared operations?
- If the user takes 10 minutes to confirm, does the lease expire? How is this handled?

### E4. [P2] What happens to tool results when a parent task is cancelled while tools are running?

Phase 5 says "remaining siblings are cancelled, cancellation is allowed to settle."

- What does "settle" mean? A timeout? All children reaching terminal state?
- What if a child tool has already made an external side effect (e.g., sent an email)?
- Is there a compensating action mechanism?

---

## F. Event Pipeline (Phase 6)

### F1. [P2] Which event types are persisted vs stream-only?

Phase 6 Task 4 says: "explicitly document which event types are streamed live only, persisted and replayable, internal-only."

- The plan says to document this but doesn't actually specify the classification.
- At minimum: `ThinkingDelta` (stream-only?), `ToolProgress` (stream-only?), `TurnCompleted` (persisted?), `Error` (persisted?)
- This classification is needed before the event store schema is finalized.

### F2. [P2] How does "replay-to-live handoff" work without gaps or duplicates?

Phase 6 sub-task ENG-7948 mentions "Race-Free Replay-to-Live Handoff."

- Does the client specify `after_sequence` and the server replays from the DB, then switches to live tail at the current sequence?
- What if an event is committed between the DB query and the live subscription?
- Is there an overlap window where the client might see duplicates?

---

## G. Subagent System (Phase 7)

### G1. [P2] Does a subagent get its own thread or share the parent thread?

Phase 7 says "child tasks in `agent_tasks`" but doesn't clarify:
- Does each subagent get a separate `thread_id`?
- If so, are parent and child threads linked? How?
- If not, how are parent and child message histories separated?

### G2. [P2] What are the default depth and concurrency limits?

Phase 7 Task 2 says "inherit subagent depth and concurrency budget."

- What are the default limits?
- Are they hardcoded? Configurable per-consumer? Per-agent-definition?
- What happens when a subagent tries to exceed the depth limit? Immediate fail? Queued?

---

## H. Cross-Cutting Concerns

### H1. [P0] Encryption strategy for stored data

The PRD mentions "encrypted persistence" and the data model uses `BYTEA` for content fields.

- Is encryption handled at the column level (application-side) or at the database level (TDE)?
- What cipher? AES-256-GCM? (The old canceled Phase 1 had this)
- Is there a key rotation strategy?
- Which columns are encrypted? All `BYTEA` columns? Or only `content`, `context_blob`, `result_blob`?

### H2. [P1] How does the server handle multiple consumers with different capabilities?

The PRD lists three consumers (Bipa, Coding agent, Satoshi) with different concerns.

- Are these different `AgentDefinition` configurations?
- Do they share the same database?
- Is there tenant isolation? Or is this single-tenant per deployment?

### H3. [P1] What is the migration strategy from the current SDK to the server library?

Phase 0 converts to a workspace, but:
- How do existing SDK users migrate?
- Is the `agent-sdk` facade crate backwards compatible?
- What's the transition path for consumers currently using `run()` and `run_persistent()`?

### H4. [P2] What observability is built in?

The current SDK has OpenTelemetry instrumentation. For the server:
- Are there standard metrics (task queue depth, lease expiry rate, event lag)?
- Are there standard traces (per-turn, per-tool-task)?
- Is there structured logging for operational events?
- Phase 8 mentions "metrics for lease expiry, retries, replay lag, and event backlog" but nothing earlier.

---

## Priority Summary

| Priority | Count | Blocks |
|---|---|---|
| P0 | 3 | Current Phase 1 implementation |
| P1 | 10 | Phases 2-4 implementation |
| P2 | 11 | Phases 5-8 implementation |

**Top 3 to resolve immediately**:
1. A1: Server durability contract for `run_turn`
2. A2: Execution-mode input mechanism
3. H1: Encryption strategy
