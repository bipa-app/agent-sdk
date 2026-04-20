## Summary

Final replacement slice: add AMQP only after correctness works without it, then finish the operational hardening required for GA.

## Locked Decisions

- AMQP is a wakeup/fanout layer over durable state.
- Queue loss or duplication may change latency, never correctness.
- Startup relay backfills from durable outbox rows.

## Tasks

### 1. Outbox/Wakeup Integration
- persist outbox rows transactionally with journal changes
- relay unpublished rows on startup
- make consumers duplicate-safe by re-checking task state before acting

### 2. Watch/Fanout
- use AMQP watch/fanout for cross-instance notification
- keep replay dependent on `events`, not watch payloads

### 3. GA Hardening
- retention policy for checkpoints, events, and audit tables
- health checks around journal, relay, and worker pool
- metrics for lease expiry, retries, replay lag, and event backlog

### 4. Explicit Deferrals
- experimentation
- MCP registry/provenance persistence
- additional transport surfaces

## Acceptance

- Lost or duplicated AMQP delivery does not violate journal correctness.
- A restarted relay publishes previously unpublished outbox rows.
- Multi-instance deployment uses durable replay as the reconnect path.

## Dependencies

- Phase 0 workspace split.
- Phases 1-7.
