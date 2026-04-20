## Summary

Second replacement slice: build the journal before building processing. No worker should run real turns until task ownership, leases, retries, child-task blocking, and recovery invariants exist.

## Locked Decisions

- `agent_tasks` is the sole execution authority.
- One runnable root task per thread is enforced in Postgres.
- Tool execution is represented as lightweight child tasks, not inline loop state.
- AMQP does not own retries or state transitions.
- Root tasks can block on child task completion via `waiting_on_children`.

## Tasks

### 1. Create `agent_tasks`
- Add root and child task support for:
  - `root_turn`
  - `tool_runtime`
  - future `subagent`
- Add lease ownership and retry counters.
- Add root-task FIFO queue state and parent waiting state.
- Add typed durable fields for:
  - `continuation_state`
  - `prepared_operation_state`
  - `context_blob`
  - `result_blob`
  - `error_blob`

### 2. Task Store API
- `create_root_task`
- `create_child_task`
- `acquire_runnable_task`
- `heartbeat_task`
- `complete_task_and_promote_next_root`
- `fail_task_or_requeue`
- `move_to_awaiting_confirmation`
- `resume_confirmed_task`
- `move_to_waiting_on_children`
- `resume_parent_after_children`
- `cancel_task_tree`
- `find_stale_tasks`

### 3. Recovery Rules
- Recovery only resumes from the latest completed-turn checkpoint.
- A stale root task without a completed-turn checkpoint may restart only when no child tool-task or prepared-operation state makes that unsafe.
- A stale tool-runtime task uses its own durable execution/prepared state to decide retry, requeue, or fail-closed behavior.
- Recovery must re-check task state before requeueing to prevent double ownership.

### 4. Root-Task Concurrency Rule
- Add a partial unique index preventing more than one root task in `pending|running|waiting_on_children|awaiting_confirmation` per thread.
- Same-thread submissions use a durable FIFO queue.
- Child tool tasks do not participate in root admission.

### 5. Parent/Child Lifecycle Rule
- A root task waiting on tool child tasks has no worker lease and cannot be reacquired as runnable work.
- Tool child tasks can move independently through `pending`, `running`, `awaiting_confirmation`, `completed`, `failed`, or `cancelled`.
- Parent resume happens through guarded journal transitions only after the child-task batch reaches a resumable terminal state.

## Acceptance

- Two workers cannot own the same runnable task.
- A thread cannot acquire two runnable root tasks at once.
- A root task blocked on tool child tasks is durable, queue-blocking, and not accidentally reacquired as runnable work.
- Lease expiry moves work back to the journal without AMQP being authoritative.
- Confirmation-paused tasks and tool-blocked parent tasks survive restart as journal state.

## Dependencies

- Phase 0 workspace split.
- Phase 1 contract work.
