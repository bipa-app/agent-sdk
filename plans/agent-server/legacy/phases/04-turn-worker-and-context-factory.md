## Summary

Fourth replacement slice: implement the root turn worker that runs one model-side turn at a time using the Phase 1 SDK contracts and the Phase 2/3 persistence layer.

## Locked Decisions

- Workers acquire from `agent_tasks`, not directly from queue messages.
- The authoritative worker path uses `run_turn`, never `run()` or `run_persistent()`.
- In server mode, the root turn worker does not execute tool calls inline.
- Tool calls are surfaced as typed work and dispatched into child `tool_runtime` tasks.
- A root task waiting on child tool work moves to `waiting_on_children` and releases its worker lease.
- `ExecutionContextFactory` owns reconstruction of `ToolContext` and policy inputs.
- Durable message and state writes stay staged until a completed-turn commit path succeeds.

## Tasks

### 1. Agent Definition Surface
- Define `AgentDefinition` as server-owned policy, not a thin mirror of `AgentConfig`.
- Keep provider selection, resolved model choice, and tool set under the server’s control.

### 2. ExecutionContextFactory
- Define a factory that builds:
  - `ToolContext`
  - provider-specific runtime dependencies
  - thread/task-scoped policy context
- Ensure worker context is derived from persisted task/thread state plus trusted host dependencies.
- Pair this with staged message/state adapters seeded from the latest committed checkpoint.

### 3. Worker Flow
- acquire runnable task
- create `turn_attempt`
- rebuild thread state from latest checkpoint
- compute next `AgentInput`
- run one externally orchestrated model turn
- if the turn completes without tool work:
  - close the `turn_attempt`
  - commit staged projections atomically
  - advance task state
- if the turn surfaces tool work:
  - persist suspended continuation and turn-attempt linkage
  - create child `tool_runtime` tasks
  - move the parent task to `waiting_on_children`
  - return control to the journal without inline tool execution

### 4. Input Rules
- first root turn: initial user input
- continuation turn: `Continue`
- tool-result resume turn: resume from durable continuation plus completed child tool results
- confirmation resume turn: deferred to tool-runtime task flow in Phase 5
- child task: subagent-owned task payload is deferred to Phase 7

## Acceptance

- Workers never call `run()` as the server core.
- A worker can rebuild context deterministically from task/thread records.
- The same thread never has two workers executing root turns concurrently.
- A root turn worker can suspend at the tool boundary and create child tool tasks without committing a completed-turn checkpoint prematurely.
- Text-only turns commit through the completed-turn commit path without leaking mid-turn durable writes.

## Dependencies

- Phases 0-3.
