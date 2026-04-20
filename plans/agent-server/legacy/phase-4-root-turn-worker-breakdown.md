# Plan: Phase 4 Root Turn Worker and Tool-Task Dispatch

> Source PRD: [ENG-7895](https://linear.app/bipa/issue/ENG-7895) and [04-turn-worker-and-context-factory.md](/Users/luizparreira/work/agent-sdk/plans/agent-server/phases/04-turn-worker-and-context-factory.md)

## Architectural decisions

Durable decisions that apply across all Phase 4 slices:

- **Branching**: all rewrite PRs target `sdk/v2`.
- **Task ownership**: the Phase 4 worker acquires only `root_turn` tasks from `agent_tasks`.
- **Worker boundary**: the authoritative server path uses `run_turn`, never `run()` or `run_persistent()`.
- **State recovery**: the worker rebuilds its next-turn view from the latest completed checkpoint, not from mid-turn durable writes.
- **Projection writes**: durable thread/message/state projection changes only happen through the completed-turn commit path.
- **Tool boundary**: in server mode, the root worker does not execute tools inline. It suspends at the tool boundary, persists continuation state, and creates `tool_runtime` child tasks.
- **Parent waiting state**: a root task blocked on tool child tasks moves to `waiting_on_children` and releases its lease.
- **Future boundary**: Phase 4 may create tool child tasks and suspend/resume around them, but actual tool-runtime execution belongs to Phase 5.

## User stories

- **User story 1**: As the server, I can resolve an `AgentDefinition` and deterministic execution context for a runnable root task.
- **User story 2**: As the server, I can rebuild staged message/state context from the latest completed checkpoint and keep durable writes buffered until commit time.
- **User story 3**: As the server, I can run a text-only or no-tool model turn end-to-end and commit the result atomically.
- **User story 4**: As the server, I can stop at the tool boundary, persist suspended turn state, and dispatch `tool_runtime` child tasks instead of executing tools inline.
- **User story 5**: As the server, I can resume a suspended root turn from completed child tool results through the journal.
- **User story 6**: As the server, I can handle cancellation and failure without leaking staged writes or corrupting the task lifecycle.

---

## Phase 4.1: Agent Definition Resolution and Worker Bootstrapping

**User stories**: 1

### What to build

Define the worker-facing `AgentDefinition` and registry surface that maps a runnable `root_turn` task to the correct server-owned runtime policy. This slice should prove the worker can load one runnable root task, resolve its definition, and construct the trusted bootstrapping inputs needed for later execution.

### Acceptance criteria

- [ ] A runnable root task can be mapped deterministically to one `AgentDefinition`.
- [ ] The worker bootstrapping path does not depend on the old monolithic `AgentConfig` shape as the server contract.
- [ ] The resolved definition exposes the runtime policy inputs the later Phase 4 slices need.

---

## Phase 4.2: Checkpoint-Seeded Execution Context and Staged Stores

**User stories**: 1, 2

### What to build

Implement the `ExecutionContextFactory` inputs and the staged message/state adapters used by the root worker. This slice should prove the worker can rebuild its execution context from task, thread, and checkpoint state without performing durable mid-turn projection writes.

### Acceptance criteria

- [ ] The worker can reconstruct its trusted execution context from durable task/thread inputs plus host dependencies.
- [ ] Message and state mutations remain buffered in staged adapters during a turn.
- [ ] The staged context is seeded from the latest completed checkpoint for an existing thread, or from an empty thread state when no checkpoint exists.

---

## Phase 4.3: Text-Only Root Turn Execution and Completed-Turn Commit

**User stories**: 2, 3

### What to build

Implement the first end-to-end root worker path for a turn that completes without externalized tool work. This slice should acquire a runnable root task, open a `turn_attempt`, run one model-side turn through `run_turn`, and atomically commit the resulting projections and task transition.

### Acceptance criteria

- [ ] A root worker can execute a no-tool turn end-to-end from acquisition through completed-turn commit.
- [ ] The worker closes the `turn_attempt`, commits staged projections atomically, and advances the root task correctly.
- [ ] No durable projection writes occur before the completed-turn commit path succeeds.

---

## Phase 4.4: Tool-Boundary Suspension and Child Task Dispatch

**User stories**: 4

### What to build

Implement the server-mode path where `run_turn` surfaces tool work instead of executing it inline. This slice should persist the suspended continuation, create `tool_runtime` child tasks, and move the parent root task to `waiting_on_children` without creating a completed-turn checkpoint.

### Acceptance criteria

- [ ] A root worker can stop cleanly at the tool boundary without inline tool execution.
- [ ] The parent task persists the continuation and turns into `waiting_on_children`.
- [ ] One or more `tool_runtime` child tasks are created with enough durable state for Phase 5 to execute them.

---

## Phase 4.5: Resume Root Turn from Completed Tool Results

**User stories**: 5

### What to build

Implement the journal-driven resume path for a suspended root turn after its tool child tasks have already reached a resumable terminal state. This slice should rebuild the suspended turn from durable continuation plus completed child results and continue the model-side turn through the same staged commit model used by text-only turns.

### Acceptance criteria

- [ ] A suspended root task can resume from durable child tool results without reconstructing inline loop state.
- [ ] The resumed root turn uses the same staged commit rules as other completed turns.
- [ ] Parent resume is driven by durable journal state, not by ad hoc channels.

---

## Phase 4.6: Root Worker Cancellation, Failure, and Regression Coverage

**User stories**: 3, 4, 5, 6

### What to build

Finalize the root worker lifecycle by covering cancellation, non-tool failure, suspended-turn edges, and regression tests for the full Phase 4 model. This slice should prove the worker can fail or cancel safely without leaking staged projection writes or corrupting root/child task state.

### Acceptance criteria

- [ ] Cancelled and failed root turns do not durably advance thread/message projections.
- [ ] Failed or cancelled suspended turns leave the task lifecycle in a coherent journal state.
- [ ] Regression coverage exercises text-only completion, tool-boundary suspension, resume-from-child-results, and failure/cancel paths.
