# Health & Readiness Runbook

> **Phase 8 GA reference.**  Documents the
> `agent-service-host` two-axis health model and the HTTP probe
> contract Kubernetes (or any other probe runner) consumes.

## 1. Health model

The host exposes **two independent axes**:

| Axis | Source | Failure meaning |
|------|--------|-----------------|
| **Core** | Journal store reachable; lease-sweep loop alive; worker pool alive | Service is broken — stop routing traffic |
| **Latency layer** | Optional relay/broker (AMQP) reachable; backlog under threshold | Extra latency but correctness intact |

The aggregate `status` is derived:

| Core | Latency layer | Aggregate `status` | Ready? | Live? |
|------|---------------|--------------------|--------|-------|
| Healthy | Healthy / NotConfigured | `healthy` | yes | yes |
| Healthy | Degraded | `degraded` | **yes** | yes |
| Unhealthy | * | `unhealthy` | no | no |

> **Why readiness is unaffected by the latency layer.**  The journal
> guarantees correctness even when the broker is down: writes are
> durable, workers keep polling via the fallback sweep, and clients
> always get the authoritative event tail by replaying from the
> durable repository.  Pulling a degraded pod from the load balancer
> would only spread the problem.

## 2. HTTP probe endpoints

The host's `http_health` server exposes three routes on
`transport.http_addr` (default `127.0.0.1:8080`):

| Route | Purpose | 200 condition | Non-200 |
|-------|---------|---------------|---------|
| `GET /healthz` | Liveness | `is_live()` true (`status != "unhealthy"`) | `503 Service Unavailable` |
| `GET /readyz`  | Readiness | `is_ready()` true (`core == "healthy"`) | `503 Service Unavailable` |
| `GET /health`  | Diagnostics snapshot | always 200 | n/a |

All three return the full snapshot JSON in the body so operators can
inspect the reason for a non-200 from `kubectl describe`:

```json
{
  "status": "degraded",
  "core": "healthy",
  "latency_layer": "degraded",
  "sweep_loop_alive": true,
  "worker_pool_alive": true
}
```

## 3. Kubernetes probe sample

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /readyz
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

* `livenessProbe` — restart only on `Unhealthy`.  `Degraded` should
  **not** restart the pod; the latency layer recovers on its own.
* `readinessProbe` — pull from load balancer on Unhealthy core only.
  Latency-layer outages do not pull traffic.

## 4. Health → action matrix

| `status` | `core` | `latency_layer` | Operator action |
|----------|--------|-----------------|-----------------|
| `healthy` | `healthy` | `healthy` / `not_configured` | none |
| `degraded` | `healthy` | `degraded` | check broker; no traffic action |
| `unhealthy` | `unhealthy` | * | restart pod; investigate journal store |

## 5. Component liveness signals

The `core` axis is derived from three flags inside the snapshot:

* `sweep_loop_alive` — the lease-expiry sweep is running.  Failure
  means task leases never get released and stuck workers cannot be
  recovered.
* `worker_pool_alive` — at least one worker is running.  Failure
  means no execution will happen.
* `core` — explicit failure flag set when the journal store
  reports persistent errors.

Any one being false flips `core` to `unhealthy` and `status` to
`unhealthy`.

## 6. Latency-layer transitions

The latency layer flips between `Healthy` and `Degraded` based on
relay observations:

* **Flip to `Degraded`** when:
  * A relay tick reports `failed > 0` or `expired > 0`.
  * A claim-reclaim sweep returns an error (broker or store is
    unreachable).
  * Backlog protection observes `pending > soft_threshold` (when
    `relay.backlog_threshold` is configured).

* **Flip to `Healthy`** when:
  * A tick claims zero rows AND the most recent reclaim succeeded
    AND the backlog is below threshold.

* **Stay `NotConfigured`** when `relay.enabled = false`.  The host
  reports this as a distinct state so `Degraded` cannot be mistaken
  for "no relay running".

## 7. Alerting reference

| Alert | Source signal | Severity | Action |
|-------|---------------|----------|--------|
| `core-unhealthy` | `/readyz != 200` for 60 s | page | Restart, investigate |
| `liveness-fail` | `/healthz != 200` for 60 s | page | Kubernetes restarts; investigate root cause |
| `latency-layer-degraded` | `latency_layer == "degraded"` for 10 m | warning | Check broker, see relay runbook |
| `relay-backlog-soft` | `relay_backlog_max > soft` for 5 m | warning | Tune `batch_size` or scale relay workers |
| `relay-backlog-hard` | `relay_backlog_max > hard` for 5 m | page | Broker likely down; emergency response |
| `lease-sweep-stalled` | `lease_sweep_cycles` rate == 0 for 5 m | page | Sweep loop crashed; restart |
| `worker-pool-empty` | `worker_pool_alive == false` | page | Workers all crashed; restart |

The metric names map to the host's [`MetricsRecorder`](../../crates/agent-service-host/src/metrics.rs)
contract and the field names emitted by the
`LoggingMetricsRecorder`.

## 8. Local diagnostic recipes

```bash
# Quick snapshot
curl -s localhost:8080/health | jq

# Watch for transitions
watch -n 1 "curl -s localhost:8080/health | jq -r '.status + \" core=\" + .core + \" latency=\" + .latency_layer'"

# Confirm liveness vs readiness divergence
curl -sw "%{http_code}\n" -o /dev/null localhost:8080/healthz
curl -sw "%{http_code}\n" -o /dev/null localhost:8080/readyz
```

A degraded latency layer will show `200 200` from the two `curl`
commands above — the contract holds.

See [`amqp-relay-operations.md`](./amqp-relay-operations.md) for
the broker-side runbook.
