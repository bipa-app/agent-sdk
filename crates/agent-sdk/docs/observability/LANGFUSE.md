# Local Langfuse stack

A self-contained Langfuse + OpenTelemetry collector stack for testing the
`agent-sdk` OTLP exporter on a developer's laptop. Spans emitted by
`agent-sdk-otel::install_global_provider` land in the local Langfuse UI within
seconds, so you can verify span shape, attribute names, and trace metadata
without a real production endpoint.

> **Local dev only.** The credentials in this stack are well-known
> placeholders (`pk-lf-local-public-key`, `changeme123`, …) and live entirely
> on your laptop. Do not point a production agent at this collector.

---

## 1. Bring it up

```bash
docker compose -f dev/observability/langfuse/docker-compose.yml up -d
```

The first boot pulls Postgres, ClickHouse, Redis, MinIO, the Langfuse
web/worker, and the OTel collector. Allow ~30 s for `langfuse-web` to finish
its migrations.

Sign in at <http://localhost:4000>:

| Field | Value |
| --- | --- |
| Email | `otel@example.com` |
| Password | `changeme123` |

The first project is auto-provisioned with public key
`pk-lf-local-public-key` and secret key `sk-lf-local-secret-key`. The OTel
collector already carries these as a Basic-auth header, so no manual API-key
wiring is required.

## 2. Send a trace from the example

In another shell, point the SDK at the collector and run the bundled example:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run --example otel --features otel
```

A trace named `agent-sdk.otel-example` should appear in Langfuse within
~60 s. Drill in to see the root `invoke_agent` span, the nested
`chat <model>` LLM span, and the `agent.turn` summary spans.

Set `OTEL_AGENT_SDK_CAPTURE_PAYLOADS=true` only if your `ObservabilityStore`
implements `acknowledge_pii_redaction` — the SDK forces every `Inline`
decision down to `Omit` otherwise. See [Phase 9 · C2](./PHASE_9_INVENTORY.md)
for the gate.

## 3. Where to go next

- [`PHASE_9_INVENTORY.md`](./PHASE_9_INVENTORY.md) — every span / attribute /
  metric the SDK emits today, plus the open gaps each Phase 9 card closes.
- **A4 — Langfuse helpers.** Stamps `langfuse.observation.type` (`agent`,
  `generation`, `tool`, `chain`) on the right spans so the Langfuse UI groups
  them correctly.
- **A5 — `RunOptions` for trace metadata.** Lets you attach
  `langfuse.session.id`, `langfuse.user.id`, `langfuse.trace.tags`, and
  `langfuse.trace.metadata.*` from a single struct (`agent.run_with_options`).

## 4. Tear down

```bash
docker compose -f dev/observability/langfuse/docker-compose.yml down
```

Add `-v` to drop the named volumes
(`agent_sdk_langfuse_postgres`, `agent_sdk_langfuse_clickhouse`,
`agent_sdk_langfuse_clickhouse_logs`, `agent_sdk_langfuse_redis`,
`agent_sdk_langfuse_minio`) and reset the Langfuse database on the next
boot.

## 5. Using this from a downstream consumer (bip, etc.)

If your project depends on `agent-sdk` as a Cargo git dependency, the
compose files don't materialize automatically — Cargo only fetches
`crates/**`. Use the `agent-sdk-cli` to drop the same files into your
working tree.

```bash
# Install the CLI (binary name: `agent-sdk`).
cargo install --git https://github.com/bipa-app/agent-sdk.git \
  --rev <pin-from-your-Cargo.toml> agent-sdk-cli

# Or, when developing against a local checkout:
cargo install --path /path/to/agent-sdk/crates/agent-sdk-cli
```

Then, from your project root:

```bash
agent-sdk doctor                 # checks docker, ports 4000/4317/4318
agent-sdk local-langfuse init    # writes ./dev/observability/langfuse/{docker-compose.yml,otel-collector.yaml,LANGFUSE.md}
agent-sdk local-langfuse up      # docker compose up -d (read this file before running)
```

`agent-sdk local-langfuse init --force` overwrites an existing copy when
you bump the SDK pin and want the latest stack. The CLI is just a
distribution channel — the bytes it writes are exactly the ones checked
into the SDK at the rev you installed.

Once your binary calls
`agent_sdk_otel::install_global_provider(OtelConfig::from_env()?)?` at
startup (Phase 9 · E1), pointing it at the local collector is a single
env var:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 cargo run -p <your-binary>
```

## 6. Troubleshooting

- **Port `4000`, `4317`, or `4318` already bound.** A parallel Bipa stack
  (`bipa-langfuse`) or another collector probably has the port. Run
  `docker compose ls` to confirm — the Compose project name is
  `agent-sdk-langfuse`. Stop the conflicting stack or remap the host ports
  in `docker-compose.yml`.
- **Trace never lands.** Check `docker compose logs otel-collector` — the
  `debug` exporter prints every batch it forwards. If the batch is leaving
  the collector but never reaches Langfuse, check
  `docker compose logs langfuse-web` for `4xx`s on
  `/api/public/otel/v1/traces`. The Basic-auth header in
  `otel-collector.yaml` must match the project's `pk-lf-…:sk-lf-…`
  credentials; the bundled value is base64 of
  `pk-lf-local-public-key:sk-lf-local-secret-key`.
- **ClickHouse takes a while to boot.** Especially on Apple silicon. The
  worker keeps retrying its migrations; give it 60 s before assuming
  something is wrong.
- **Stack collides with Bipa's identical stack.** It shouldn't — the
  Compose project name and named volumes are both prefixed `agent-sdk-`. If
  `docker volume ls` shows both `langfuse_postgres` and
  `agent_sdk_langfuse_postgres`, you're fine.
