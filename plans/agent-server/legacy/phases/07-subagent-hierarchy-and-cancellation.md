## Summary

Seventh replacement slice: make subagents first-class durable child tasks before calling the system a stable multi-agent server.

## Locked Decisions

- Durable subagents are child tasks in `agent_tasks`.
- Parent cancellation cascades through the task tree.
- Child tasks inherit parent policy, depth limits, and concurrency controls.

## Tasks

### 1. Child Task Model
- create child task from parent task
- persist child task input/context separately
- link child completion back to parent turn progression

### 2. Inherited Controls
- inherit subagent depth
- inherit subagent concurrency budget
- inherit policy context and audit context

### 3. Cascade Cancellation
- parent cancel propagates to all descendants
- child failure policy is explicit per task kind

### 4. Recovery
- stale child tasks recover through the same journal rules as root tasks
- parent tasks observe child terminal state through the journal, not ad hoc channels

## Acceptance

- A subagent tree survives restart as durable task state.
- Parent cancellation stops descendant work deterministically.
- Child tasks cannot bypass parent fanout or policy limits.

## Dependencies

- Phase 0 workspace split.
- Phases 1-6.
