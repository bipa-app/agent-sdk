## Summary

First replacement slice: lock the SDK/server boundary before any server implementation proceeds. This phase exists to remove the contract gaps that made the previous plan internally inconsistent and to introduce a server-safe tool-runtime handoff instead of inline tool execution.

## Locked Decisions

- Phase 0 workspace split is a prerequisite. These contract changes are made in the workspace layout, and rewrite PRs target `sdk/v2`.
- The server core is single-turn external orchestration.
- In server mode, `run_turn` does not execute tool calls inline. It surfaces typed tool work that the journal can dispatch as lightweight child tasks.
- `run_turn` remains a direct async boundary, but `TurnOutcome` must not resolve before the current turn boundary's event stream has been durably published and explicitly closed for that boundary.
- Public event replay uses `AgentEventEnvelope`.
- Sequence numbers are server-seeded per thread.
- Tool audit data comes from explicit SDK audit callbacks, not inferred from `post_tool_use`.
- Async workers rebuild tool context from a durable `ExecutionContextFactory` contract.

## Tasks

### 1. Direct `run_turn` Boundary and Execution Modes
- Make `run_turn` the direct async authoritative boundary for server execution.
- Keep `run()` and `run_persistent()` as convenience wrappers only.
- Replace implicit channel-drain ordering with an explicit authoritative event sink and per-turn flush or closure barrier.
- Require `run_turn` to await that barrier before returning `TurnOutcome`.
- Define the contract precisely:
  - all events for the current completed or suspended turn boundary must be committed and published to the authoritative local observer before `TurnOutcome` resolves
  - the SDK does not wait for arbitrary downstream subscribers to drain those events
- Add an execution-mode or tool-runtime strategy input so the SDK can distinguish:
  - local inline tool execution
  - server externalized tool-task handoff
- Preserve or add a local observed convenience shape, such as `TurnHandle { events, outcome }`, where the outcome resolves only after the associated turn stream has closed.
- Make strict server durability explicit in the authoritative path.

### 2. Event Sequencing Contract
- Add SDK support for externally seeded event sequencing.
- Allow server code to supply the starting sequence for a run/turn.
- Document that replay-safe sequencing is server-owned, not per-run ephemeral state.

### 3. Turn Outcome and External Tool Handoff Contract
- Extend single-turn outcomes to expose provider/model provenance and response identifiers cleanly.
- Add a typed server outcome for "tool work is required" so the root worker can create child tool tasks without executing them inline.
- Define the resume contract for continuing a turn after completed tool child tasks provide results.
- Do not build server logic around `run()`.

### 4. Durable Continuation Contract
- Make `AgentContinuation` an explicit server persistence contract.
- Document serialization, versioning, and compatibility expectations.
- Require continuation state to be sufficient for:
  - durable tool-batch suspension
  - confirmation pause/resume
  - turn resume after child tool-task completion

### 5. Tool Audit Contract
- Add a structured tool audit callback that includes:
  - tool_call_id
  - tool_name and display_name
  - tier and tool kind
  - input
  - outcome (`started`, `blocked`, `requires_confirmation`, `completed`, `rejected`, `cached`, `replayed`)
  - provider/model provenance when relevant
- Clarify that `post_tool_use` alone is insufficient for server audit.

### 6. Worker Context Contract
- Define `ExecutionContextFactory` for reconstructing `ToolContext` from durable task/thread records.
- Make request-local/auth/session data an explicit input to server workers, never hidden ambient state.
- Remove raw event sender and per-run sequence state from the authoritative `ToolContext` construction path.

### 7. Fail-Closed Persistence Hooks and Regression Surface
- Define where SDK execution must stop if durable execution intent or checkpoint persistence fails.
- Cover side-effecting tools, resumable tools, and externalized tool-batch handoff.
- Rewrite the regression suite so server mode proves it can stop at the tool boundary instead of executing tool calls inline.

## Acceptance

- The server can execute one turn without relying on hidden per-process state.
- The server can run one model turn, surface a typed tool batch, and return control without inline tool execution.
- The server can seed event sequences from persisted thread state.
- The server can observe turn events in real time while still guaranteeing that `TurnOutcome` becomes visible only after the turn's event stream has reached its explicit completion barrier.
- The server has a lossless audit callback surface for tool lifecycle.
- Confirmation pause/resume and tool-batch suspend/resume are defined as durable contracts, not in-memory patterns.

## Dependencies

- Phase 0 workspace split.
