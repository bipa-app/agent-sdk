# Local Grafana + Tempo + Prometheus stack

A self-contained Grafana stack for exploring the metrics and traces
emitted by `agent-sdk` without going through Langfuse. Tempo holds the
distributed traces, Prometheus stores the histograms and counters, and
Grafana provides the UI for both.

This stack runs **alongside** the [Langfuse stack](./LANGFUSE.md) — same
OTLP data, different sinks. Use `dev/observability/up.sh both` to bring
them up together.

> **Local dev only.** Default Grafana credentials are `admin` / `admin`.
> The collector ships a baked-in Langfuse Basic-auth header that only
> matches the local-dev keys checked into the Langfuse compose. Don't
> point a production agent at this stack.

---

## 1. Bring it up

```bash
./dev/observability/up.sh grafana
```

That starts four containers under the Compose project
`agent-sdk-grafana`:

- `tempo` — single-binary Tempo, OTLP gRPC ingest on `:4317` (in-network).
- `prometheus` — scrapes the collector's `/metrics` on `:8889`.
- `otel-collector` — OTLP receivers on host ports `4317` (gRPC) /
  `4318` (HTTP), fans out to Tempo (traces) + Prometheus (metrics) +
  optionally Langfuse (see §3).
- `grafana` — UI on host port `3001`, datasources for Tempo and
  Prometheus auto-provisioned at boot.

Allow ~15 s for Grafana to finish provisioning before logging in.

| URL | What |
| --- | --- |
| <http://localhost:3001> | Grafana UI (`admin` / `admin`) |
| <http://localhost:3200> | Tempo HTTP API |
| <http://localhost:9090> | Prometheus UI |
| `http://localhost:4317` | OTLP gRPC (point the SDK here) |
| `http://localhost:4318` | OTLP HTTP |

## 2. Send a trace + metric from the example

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run --example otel --features otel
```

In Grafana → **Explore** → **Tempo**, search for service
`agent-sdk` (or whatever your `OtelConfig::service_name` is) and you
should see the `invoke_agent` root span with its `chat <model>`,
`execute_tool`, and `agent.turn` children.

In Grafana → **Explore** → **Prometheus**, query
`agent_sdk_gen_ai_client_token_usage_count` (or any other metric the
SDK emits — see [`OBSERVABILITY_INVENTORY.md`](./OBSERVABILITY_INVENTORY.md)
§4) to confirm the histogram landed.

The starter dashboard for token usage / operation duration / tool
outcomes lands in work item D3 (see
[`OBSERVABILITY_INVENTORY.md`](./OBSERVABILITY_INVENTORY.md)). Until then
the **Agent SDK** folder in Grafana will be empty.

## 3. Run both stacks together

```bash
./dev/observability/up.sh both
```

This brings up Langfuse without its bundled collector
(`--scale otel-collector=0`) and starts the Grafana stack with
`LANGFUSE_OTLP_ENDPOINT` pointing at `host.docker.internal:4000`. A
single collector now fans out to Tempo, Prometheus, **and** Langfuse —
one trace, three sinks.

If Langfuse is down, the Langfuse exporter logs delivery failures
once per batch and the Tempo + Prometheus pipelines keep working
(`sending_queue` and `retry_on_failure` are off in
`otel-collector.yaml`).

## 4. Tear down

```bash
./dev/observability/up.sh down
```

Removes both compose projects. The script never errors on a stack
that isn't running, so it's safe to call repeatedly.

To drop the Grafana volumes (Tempo blocks, Prometheus TSDB, Grafana
local config) and start clean on the next boot:

```bash
docker compose -f dev/observability/grafana/docker-compose.yml down -v
```

The named volumes are `agent_sdk_grafana_tempo`,
`agent_sdk_grafana_prometheus`, and `agent_sdk_grafana_grafana`.

## 5. Troubleshooting

- **Port `4317` / `4318` already bound.** The Langfuse stack and the
  Grafana stack both want those ports — only one collector can hold
  them at a time. Run `docker compose ls` to see which project is
  active. If you need both, use `./dev/observability/up.sh both`,
  which owns the port arbitration for you.
- **Grafana port `3001` already bound.** Some other tool (Storybook,
  another Grafana, a microservice) probably has it. Edit the
  `3001:3000` mapping in `dev/observability/grafana/docker-compose.yml`
  to a free port.
- **`host.docker.internal` doesn't resolve on Linux.** The compose
  file declares `extra_hosts: ["host.docker.internal:host-gateway"]`
  on the collector, which Docker Engine 20.10+ honours. If you're on
  an older engine, upgrade or replace `host.docker.internal` in
  `LANGFUSE_OTLP_ENDPOINT` with your host's LAN IP.
- **Tempo refuses spans for ~10 s after boot.** The ingester needs to
  open its WAL before it accepts traffic. The collector retries
  internally; spans emitted during that window aren't dropped.
- **Prometheus shows no data points.** The collector exports metrics
  on a 10 s window, and Prometheus scrapes every 15 s, so the first
  sample lands ~25 s after the SDK records it. If a query still shows
  nothing after a minute, check
  `docker compose -f dev/observability/grafana/docker-compose.yml logs otel-collector`
  for `prometheus exporter` errors.
- **Trace appears in Tempo but not Langfuse during `both`.** Confirm
  the Langfuse stack has finished migrating
  (`docker compose -f dev/observability/langfuse/docker-compose.yml logs langfuse-web`)
  and that the collector resolved `host.docker.internal:4000`
  (`docker compose -f dev/observability/grafana/docker-compose.yml logs otel-collector`).
