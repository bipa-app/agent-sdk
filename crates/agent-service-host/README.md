# Agent Service Host

## PostgreSQL backend

The host can now boot with `storage.backend=postgres` for the current
durable-core tables:

- task journal rows
- thread projections
- message projections
- turn attempts
- checkpoints
- committed event repository
- transactional outbox rows
- retention cursors

Example config:

```yaml
storage:
  backend: postgres
  postgres:
    database_url: postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk
    schema: agent_service_host
    max_connections: 8
transport:
  grpc_enabled: true
  grpc_addr: 127.0.0.1:50051
```

If `storage.postgres.database_url` is omitted, the host falls back to
`DATABASE_URL`.

## Local prerequisites

For local development, `compose.yml` starts a compatible Postgres
instance:

```bash
docker compose up postgres18
```

Default connection string:

```text
postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk
```

Use `storage.postgres.schema` when multiple local daemons or tests need
isolated durable-core tables inside one physical database.

## Operational expectations

Startup applies the embedded durable-core migrations before the host
starts serving work. The Postgres pool is shared across the task,
thread, message, turn-attempt, checkpoint, event-journal, outbox,
retention, execution-intent, and tool-audit stores.

All SQL-backed host surfaces now survive process restart. When
`storage.backend=postgres` or `storage.backend=sqlite`, the store
registry reports every surface as durable in its startup health
snapshot. The in-memory backend is still available for tests and local
development, and continues to report every surface as process-local.

### Tool audit redaction

Tool audit events are written through a
[`RedactingToolAuditEventStore`](../agent-server/src/journal/tool_audit.rs)
wrapper that applies the default durable-write policy: baseline
redaction for the event's `input`, `output`, and `error` fields
(including the `error` carried by the `Failed` lifecycle variant)
before they reach the durable table. Read paths return the
already-redacted rows, so sensitive tool data never lands in durable
storage even if an operator replays or exports audit history.
