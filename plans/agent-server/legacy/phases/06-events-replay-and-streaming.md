## Summary

Sixth replacement slice: make persisted events the source of truth for replay and make live streaming a tail of that durable log.

## Locked Decisions

- Replay is defined against persisted `AgentEventEnvelope` rows.
- Sequence allocation is server-owned per thread.
- In-process streaming is an optimization, not the durable contract.

## Tasks

### 1. Event Store
- Persist replayable envelopes in `events`.
- Use server-assigned per-thread sequence numbers.
- Persist only the event classes the product commits to replay.

### 2. Replay API
- `StreamEvents(thread_id, after_sequence)`
- replay persisted envelopes first
- then tail the live stream for the same thread

### 3. Live Tail Rules
- define bounded backpressure behavior
- do not let slow subscribers stall task execution
- reconnect semantics are always “replay from durable log, then tail”

### 4. Event Filtering
- explicitly document which event types are:
  - streamed live only
  - persisted and replayable
  - internal-only

## Acceptance

- Reconnect after restart yields the exact persisted envelope stream after `after_sequence`.
- Same-thread event sequences stay monotonic across retries and resumed turns.
- Slow or disconnected subscribers do not block workers indefinitely.

## Dependencies

- Phase 0 workspace split.
- Phases 1-5.
