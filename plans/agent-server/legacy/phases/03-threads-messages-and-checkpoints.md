## Summary

Third replacement slice: add conversation persistence and full-message checkpoints on top of the journal. This phase proves that crash-safe turn-to-turn durability works before tool-runtime execution or event replay is added.

## Locked Decisions

- `messages` remains SDK-compatible history storage, but server-mode writes are staged in memory during a turn and committed atomically only at completed-turn commit.
- Provider/model provenance lives in `turn_attempts`, not `messages`.
- V1 checkpointing is full snapshot after each completed turn.
- Recovery in this phase is checkpoint-only. Persisted events are not part of Phase 3 recovery.

## Tasks

### 1. Threads and Messages
- Implement `PgThreadStore` for thread lifecycle and aggregates.
- Implement `PgMessageStore` strictly around SDK message history semantics.
- Keep thread aggregate writes owned by the task processor, not split across multiple writers.
- Preserve transactional `replace_history` semantics for compaction.

### 2. Turn Attempts
- Replace lifecycle-style `turn_snapshots` with immutable `turn_attempts`.
- Record one row per executed attempt with request/response audit fields.
- Keep continuation state out of this table.

### 3. Turn Checkpoints
- After every completed turn, persist:
  - full message history snapshot
  - agent state snapshot
  - link to the completed `turn_attempt`

### 4. Recovery Proof
- Rebuild a thread from:
  - thread row
  - latest checkpoint
- Show that the next turn resumes from the checkpointed state, not from thread start or mid-turn durable writes.

## Acceptance

- Crash after a completed turn resumes from the latest checkpoint.
- Message history is not corrupted by compaction or replace-history.
- A completed turn always yields exactly one checkpoint for recovery.
- Durable thread/message projections never advance ahead of the latest completed checkpoint.

## Dependencies

- Phase 0 workspace split.
- Phase 2 journal core.
