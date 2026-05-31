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

## Run with observability

The host wires the
[`agent-sdk-otel`](../agent-sdk-otel) bootstrap helper into the host
binary so a single config section turns on traces + metrics.

Build the host with the `otel` feature to pull in the OTLP exporter
and the `agent_server.*` / `agent_service_host.*` / `gen_ai.*`
metric instruments:

```bash
cargo run -p agent-service-host --features otel
```

Default builds remain dependency-free — the `otel` feature is
strictly opt-in.

### Quickest local-stack flow

```bash
# Bring up the local Tempo + Prometheus + Grafana stack (and
# optionally fan out to Langfuse). See dev/observability/up.sh.
./dev/observability/up.sh both

# Tell the host to push to the local OTel collector.
cat > /tmp/config.yaml <<'YAML'
observability:
  enabled: true
  otlp_endpoint: "http://localhost:4317"
  service_name: agent-service-host
  deployment_environment: local
  sampler: parentbased_traceidratio
  sample_ratio: 1.0
YAML

AGENT_SERVICE_CONFIG=/tmp/config.yaml \
  cargo run -p agent-service-host --features otel
```

Open <http://localhost:3001> (Grafana, `admin`/`admin`) and the
starter `agent-sdk` dashboard renders as soon as the host emits its
first batch of metrics.

### Configuration reference

| YAML key                                  | Env var equivalent                  | Default                                       |
| ----------------------------------------- | ----------------------------------- | --------------------------------------------- |
| `observability.enabled`                   | —                                   | `false` (hard skip)                           |
| `observability.service_name`              | `OTEL_SERVICE_NAME`                 | `agent-service-host`                          |
| `observability.service_instance_id`       | `OTEL_SERVICE_INSTANCE_ID`          | random UUID v7                                |
| `observability.deployment_environment`    | `OTEL_DEPLOYMENT_ENVIRONMENT`       | `unknown_deployment`                          |
| `observability.otlp_endpoint`             | `OTEL_EXPORTER_OTLP_ENDPOINT`       | unset (exporter disabled)                     |
| `observability.otlp_headers`              | `OTEL_EXPORTER_OTLP_HEADERS`        | empty                                         |
| `observability.sampler`                   | `OTEL_TRACES_SAMPLER`               | `parentbased_traceidratio`                    |
| `observability.sample_ratio`              | `OTEL_TRACES_SAMPLER_ARG`           | `1.0`                                         |
| `observability.propagated_baggage_keys`   | —                                   | baseline allow-list                           |
| `observability.capture_payloads`          | —                                   | `false` (default-deny)                        |

Resolution order: env vars are read first, then YAML overrides any
field that is set explicitly. This keeps containerised deploys
12-factor while still letting static config files pin values.

### Prometheus exposition

This release does **not** spin up an in-process `/metrics` server.
The `opentelemetry-prometheus` crate is officially discontinued and
the recommended path is to push OTLP and let an OTel collector or
Prometheus' native OTLP receiver handle scraping. The local
Grafana stack already does exactly that — see
`dev/observability/grafana/otel-collector.yaml` for the collector
config and `crates/agent-sdk/docs/observability/GRAFANA.md` for the
end-to-end flow.
