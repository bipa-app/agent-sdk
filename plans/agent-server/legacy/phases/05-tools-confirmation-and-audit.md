## Summary

Fifth replacement slice: turn tool execution into journal-owned `tool_runtime` child tasks and make tool mutation safe enough for production by landing durable execution intent, confirmation pause/resume, restart-safe execution, and replay-safe audit.

## Locked Decisions

- Tool execution is owned by child `tool_runtime` tasks, not by inline root-turn loop execution.
- One tool call maps to one `tool_runtime` child task with durable linkage to `parent_task_id`, `root_task_id`, `turn_attempt_id`, `tool_call_id`, `tool_name`, `display_name`, `tier`, `tool_kind`, requested input, effective or prepared input, stable operation key, and prepared-operation state.
- The server keeps a tool runtime policy or catalog per tool name that declares execution class, server qualification, cancel or restart support, redaction policy, and optional concurrency override.
- Unqualified async or listen tools are blocked fail-closed in server mode.
- Side-effecting tools fail closed if durable execution intent cannot be persisted first.
- Resume-time policy checks are authoritative.
- Root tasks resume only from durable child-task outcomes, not from inline in-memory loop state.
- Sibling tool tasks run with bounded parallelism, but only one `Confirm` child may be active or awaiting confirmation per parent batch at a time.
- If a `Confirm` child reaches a terminal rejection, invalidation, expiry, resume-time block, or unsuccessful terminal result in a mixed batch, remaining siblings are cancelled, cancellation is allowed to settle, and the parent turn fails deterministically.

## Tasks

### 1. Tool Runtime Worker
- Implement the worker that acquires only `tool_runtime` child tasks from `agent_tasks`.
- Remove server reliance on inline sync, async, or listen execution.
- Drive tool tasks through `pending`, `running`, `awaiting_confirmation`, `completed`, `failed`, and `cancelled`.

### 2. Durable Execution Intent and Operation Identity
- Persist stable execution intent before side-effecting execution begins.
- Support a server-owned operation identity beyond raw `tool_call_id`.
- Keep `tool_executions` as replay and external-operation state owned by the tool task, not as the scheduler.
- Treat persistence failure before execution as terminal and do not call the external tool.

### 3. Execution Semantics by Tool Class
- Run `Tool` execution fully inside the child worker.
- Allow `AsyncTool` only when durable `operation_id` plus fresh-worker status polling is enough to resume safely.
- Use `ListenExecuteTool.listen()` as durable prepare, then persist `operation_id`, `revision`, `prepared_snapshot`, and `expires_at` before moving to `awaiting_confirmation`.
- Fail closed whenever a prior external operation cannot be resumed or cancelled safely from durable state.

### 4. Confirmation Flow
- Keep confirmation state on the tool runtime task, not the root turn worker.
- Persist continuation or prepared-operation state on transition to `awaiting_confirmation`.
- On confirmed resume, re-check policy authoritatively before `execute()`.
- On rejection, invalidation, expiry, or resume-time block, prevent execution and move the tool task to a deterministic terminal state.

### 5. Parent Resume and Batch Aggregation
- Aggregate completed child tool results back into the parent root task through guarded journal transitions only.
- For `Observe`-only batches, wait for all children to settle and resume the parent with a full result batch, including deterministic error results.
- For batches containing `Confirm` tools, stop launching new siblings while a confirm child is active or awaiting confirmation.
- If a confirm child fails in a way that should abort the parent, cancel remaining siblings and fail the parent deterministically after sibling cancellation settles.

### 6. Tool Audit Log
- Populate `tool_call_log` from the new SDK audit callback and the tool-task lifecycle.
- Include `started`, `prepared`, `awaiting_confirmation`, `confirmed`, `blocked`, `rejected`, `invalidated`, `cancelled`, `completed`, `cached`, `replayed`, `resume_blocked`, and `persistence_failed`.
- Redact secrets in logs and durable storage according to server policy.

## Acceptance

- Sync `Observe` tool tasks succeed or fail deterministically and resume the parent with a full result batch.
- Qualified `AsyncTool` tasks survive worker restart through durable operation identity and fresh-worker status polling.
- Unqualified async or listen tools are blocked fail-closed before external execution.
- `ListenExecuteTool` flows through durable prepare, `awaiting_confirmation`, and confirmed `execute()` with persisted prepared state.
- Resume-time `Block` prevents confirmed execution.
- Mixed batches with confirm-task failure cancel siblings and fail the parent deterministically.
- Only one confirm child is active or awaiting confirmation per parent batch.
- Existing `InFlight` execution state never triggers unsafe blind re-execution.
- Audit logs explain success, blocked, rejected, invalidated, cancelled, cached, replayed, and persistence-failed paths.

## Dependencies

- Phases 0-4.
