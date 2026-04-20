# Agent Server — Replacement Data Model

This document replaces the previous lifecycle around `turn_snapshots`. The new execution model is:

- `threads` and `messages` are conversation projections
- `agent_tasks` is the execution authority for root turns, tool runtimes, and later subagents
- `turn_attempts` is the immutable execution log
- `turn_checkpoints` stores full-message checkpoints after completed turns
- `events` stores replayable envelopes
- `msgs` is an optional outbox/wakeup table

## Global Rules

1. All server-owned primary keys are UUID v7.
2. `ThreadId` remains an opaque SDK string, but server-issued thread ids must contain UUID v7 text.
3. No queue row or in-memory channel is authoritative for task state.
4. One active root task per thread is enforced in the database.
5. Tool execution is modeled as child task lifecycle, not inline blocking turn state.
6. Continuation state and prepared-operation state live in typed task fields, not ad hoc metadata.

## Table 1: `threads`

Purpose: durable conversation identity and aggregate status.

Core columns:

- `id UUID PRIMARY KEY`
- `status TEXT CHECK (status IN ('active', 'completed', 'archived'))`
- `title TEXT NULL`
- `metadata BYTEA NULL`
- `turn_count INTEGER NOT NULL`
- `input_tokens BIGINT NOT NULL`
- `output_tokens BIGINT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `completed_at TIMESTAMPTZ NULL`

Rules:

- `threads` owns user-facing lifecycle only.
- `messages` and `agent_tasks` reference `threads`.
- thread aggregates are written by the task owner, not by multiple stores independently.

## Table 2: `messages`

Purpose: SDK-compatible message history.

Core columns:

- `id UUID PRIMARY KEY`
- `thread_id UUID NOT NULL`
- `sequence_number INTEGER NOT NULL`
- `role TEXT NOT NULL`
- `content BYTEA NULL`
- `created_at TIMESTAMPTZ NOT NULL`

Rules:

- do not require `model_id` here; provider/model provenance belongs to `turn_attempts`
- `content` stores the serialized `llm::Message` body, including tool, image, document, thinking, and redacted-thinking blocks
- `(thread_id, sequence_number)` is unique
- compaction uses transactional replace-history semantics

## Table 3: `agent_tasks`

Purpose: authoritative execution journal.

Core columns:

- `id UUID PRIMARY KEY`
- `thread_id UUID NOT NULL`
- `parent_task_id UUID NULL`
- `root_task_id UUID NOT NULL`
- `depth INTEGER NOT NULL`
- `status TEXT NOT NULL`
- `task_kind TEXT NOT NULL`
- `attempt INTEGER NOT NULL`
- `max_attempts INTEGER NOT NULL`
- `worker_id TEXT NULL`
- `lease_expires_at TIMESTAMPTZ NULL`
- `last_heartbeat_at TIMESTAMPTZ NULL`
- `cancel_requested_at TIMESTAMPTZ NULL`
- `next_turn_number INTEGER NOT NULL`
- `current_turn_attempt_id UUID NULL`
- `continuation_state BYTEA NULL`
- `prepared_operation_state BYTEA NULL`
- `context_blob BYTEA NOT NULL`
- `result_blob BYTEA NULL`
- `error_blob BYTEA NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `completed_at TIMESTAMPTZ NULL`

Statuses:

- `queued`
- `pending`
- `running`
- `waiting_on_children`
- `awaiting_confirmation`
- `completed`
- `failed`
- `cancelled`

Rules:

- `task_kind` distinguishes at least:
  - `root_turn`
  - `tool_runtime`
  - `subagent`
- partial unique index enforces one root task in `pending|running|waiting_on_children|awaiting_confirmation` per thread
- root-task FIFO admission applies only to root tasks
- tool runtime child tasks and future subagent tasks use the same authoritative lifecycle table
- `waiting_on_children` means the parent task is blocked on child-task completion and holds no worker lease
- `awaiting_confirmation` is valid for any task kind that is blocked on durable user approval
- `context_blob` is the durable source for rebuilding worker execution context

## Table 4: `turn_attempts`

Purpose: immutable log of one executed turn attempt.

Core columns:

- `id UUID PRIMARY KEY`
- `task_id UUID NOT NULL`
- `thread_id UUID NOT NULL`
- `turn_number INTEGER NOT NULL`
- `attempt INTEGER NOT NULL`
- `status TEXT NOT NULL`
- `requested_model TEXT NOT NULL`
- `response_model TEXT NULL`
- `response_id TEXT NULL`
- `provider_name TEXT NOT NULL`
- `outcome_kind TEXT NOT NULL`
- `request_blob BYTEA NOT NULL`
- `response_blob BYTEA NULL`
- `stop_reason TEXT NULL`
- `input_tokens BIGINT NULL`
- `output_tokens BIGINT NULL`
- `cached_input_tokens BIGINT NULL`
- `llm_duration_ms BIGINT NULL`
- `started_at TIMESTAMPTZ NOT NULL`
- `completed_at TIMESTAMPTZ NULL`

Rules:

- append-only; never updated except to close the same attempt row from `running` to terminal status
- owns provider/model provenance and request/response audit data
- never stores continuation state
- a turn attempt may stay open while the root task waits on tool child tasks, but it does not checkpoint until the turn is fully committed

## Table 5: `turn_checkpoints`

Purpose: recovery after completed turns.

Core columns:

- `id UUID PRIMARY KEY`
- `task_id UUID NOT NULL`
- `thread_id UUID NOT NULL`
- `turn_number INTEGER NOT NULL`
- `turn_attempt_id UUID NOT NULL`
- `messages_snapshot BYTEA NOT NULL`
- `agent_state_snapshot BYTEA NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`

Rules:

- full-message checkpoint after each completed turn
- one checkpoint per `(thread_id, turn_number)`
- v1 recovery always resumes from the latest completed-turn checkpoint

## Table 6: `tool_executions`

Purpose: durable execution-intent and replay control.

Core columns:

- `id UUID PRIMARY KEY`
- `task_id UUID NOT NULL`
- `thread_id UUID NOT NULL`
- `tool_call_id TEXT NOT NULL`
- `stable_operation_key TEXT NULL`
- `prepared_operation_id TEXT NULL`
- `tool_name TEXT NOT NULL`
- `display_name TEXT NOT NULL`
- `input_blob BYTEA NOT NULL`
- `status TEXT NOT NULL`
- `result_blob BYTEA NULL`
- `started_at TIMESTAMPTZ NOT NULL`
- `completed_at TIMESTAMPTZ NULL`

Rules:

- `tool_executions` is not the scheduler; `agent_tasks` owns lifecycle and leasing for tool runtime work
- `tool_call_id` alone is not the durable replay identity for side-effecting tools
- v1 must support a stable server-owned operation key or equivalent dedupe identity
- persistence failure before execution is fatal to the tool run
- rows in this table are owned by `tool_runtime` child tasks

## Table 7: `tool_call_log`

Purpose: append-only audit trail.

Core columns:

- `id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY`
- `task_id UUID NOT NULL`
- `thread_id UUID NOT NULL`
- `turn_attempt_id UUID NULL`
- `tool_call_id TEXT NOT NULL`
- `tool_name TEXT NOT NULL`
- `display_name TEXT NOT NULL`
- `tier TEXT NOT NULL`
- `tool_kind TEXT NOT NULL`
- `outcome TEXT NOT NULL`
- `input_blob BYTEA NOT NULL`
- `result_blob BYTEA NULL`
- `metadata JSONB NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`

Rules:

- includes blocked, rejected, cached, replayed, and requires-confirmation outcomes
- populated from richer SDK audit hooks, not only `post_tool_use`
- logs must explain tool-task lifecycle transitions even when a tool never reaches inline completion

## Table 8: `events`

Purpose: replayable public event log.

Core columns:

- `id UUID PRIMARY KEY`
- `thread_id UUID NOT NULL`
- `turn_attempt_id UUID NULL`
- `sequence BIGINT NOT NULL`
- `event_type TEXT NOT NULL`
- `envelope_blob BYTEA NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`

Rules:

- stores serialized `AgentEventEnvelope`
- `(thread_id, sequence)` is unique
- sequence allocation is owned by the server commit path
- persisted replay excludes ephemeral deltas unless explicitly promoted later

## Table 9: `msgs`

Purpose: optional outbox/wakeup and watch fanout.

Core columns:

- `id UUID PRIMARY KEY`
- `kind TEXT NOT NULL`
- `address TEXT NOT NULL`
- `key TEXT NOT NULL`
- `payload JSONB NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `published_at TIMESTAMPTZ NULL`

Rules:

- outbox only; correctness must not depend on immediate relay
- startup relay must backfill unpublished rows

## Explicit Deferrals

Not in the replacement v1 schema:

- experiment tables
- MCP registry/provenance tables
- generalized blind-index search layer

If those return later, they must integrate with the task-owned execution model rather than revive the old split ownership.
