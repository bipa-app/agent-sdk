# Agent Service Host

## PostgreSQL backend

The host can now boot with `storage.backend=postgres` for the current
durable-core tables:

- task journal rows
- thread projections
- message projections
- turn attempts
- checkpoints

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
thread, message, turn-attempt, and checkpoint stores.

The current Postgres backend is intentionally partial. These host
surfaces still remain process-local in-memory state and do not survive
restart:

- committed event repository
- execution intent store
- tool audit store

The host logs each of those gaps when `storage.backend=postgres` so the
deployment shape is explicit.
