# AMQP Relay Operations Runbook

> **Phase 8 GA reference.**  Covers the brokered latency layer that
> ships in `agent-service-host`: outbox relay, task wakeup, cross
> instance thread-event watch, and the retention janitor.

## 1. Architecture in one diagram

```text
   ┌───────────────────────────────────────────────────────────────┐
   │                       ServiceHost                             │
   │                                                               │
   │   worker pool                                                 │
   │       │ acquire_next_runnable                                 │
   │       ▼                                                       │
   │   journal (durable)  ─►  outbox table  ─►  relay scheduler    │
   │       ▲                                          │            │
   │       │                                          ▼            │
   │   wakeup                                      AMQP            │
   │   signal ◄── fallback sweep ◄── consumer  ◄────┘             │
   │       ▲                                                       │
   │       └────── thread-events watch consumer ◄── AMQP ◄── peer  │
   │                                                                │
   │   retention janitor ─► event repo + outbox safety bound        │
   └───────────────────────────────────────────────────────────────┘
```

The journal is the **only** source of truth.  Every broker delivery is
advisory: the consumer always re-reads the journal and never executes
work directly from a broker payload.

## 2. Normal operation

| Component | Cadence | What it does |
|-----------|---------|--------------|
| Lease sweep | `worker.sweep_interval_secs` (default 5 s) | Returns expired-lease tasks to `Pending` |
| Worker pool | `worker.acquisition_interval_secs` (default 1 s) **+** wakeup signal | CAS the next runnable task to `Running` |
| Relay startup reclaim | once at boot | Returns abandoned `Claimed` outbox rows past lease to `Pending` |
| Relay backfill | drain on boot | Publishes everything that was sitting in the outbox before the host came up |
| Relay steady state | `relay.poll_interval_secs` (default 2 s) | Claim → publish → mark loop |
| Relay reclaim | `relay.reclaim_interval_secs` (default 30 s) | Reclaim stale relay claims left by crashed peers |
| Task wakeup consumer | per AMQP delivery | Re-checks journal and nudges local workers |
| Fallback wakeup sweep | `wakeup.fallback_interval_secs` (default 5 s) | Pulses the local wakeup signal whether or not the broker delivers |
| Thread-event watch consumer | per AMQP delivery | Replays missing events from the durable repo and forwards to the local notifier |
| Retention janitor | `retention.janitor_interval_secs` (default 60 s) | Advances retention floor for expired events; prunes excess checkpoints |

### Expected steady-state log signature

```text
level=info metric=relay_tick claimed=N delivered=N failed=0 expired=0 duration_ms=…
level=info metric=lease_sweep released=0
level=info metric=janitor_cycle threads_scanned=… events_purged=0 checkpoints_pruned=0 floors_advanced=0
```

`failed=0`, `expired=0`, and `released=0` for an idle deployment is
the desired baseline.

## 3. Degraded mode

A degraded relay does **not** make the server unready.  Readiness is
tied to the durable core (journal + worker pool); the latency layer
is monitored on a separate axis.

### Symptoms

* `/health` reports `latency_layer: degraded`.
* Relay tick logs show `failed > 0` or `expired > 0`.
* Outbox row counts grow.

### Causes and resolutions

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| AMQP connection refused | Broker down / network blip | Wait — supervisor reconnects with bounded backoff (250 ms → 30 s) |
| Repeated `claim reclaim failed` | Connection pool exhausted on store backend | Tune `storage.postgres.max_connections` |
| `expired` counter rising | Persistent broker outage; rows exhausted retry budget | Either raise `max_attempts` per kind or accept the row as dead |
| `Degraded` despite empty backlog | Stale flip from a previous failure | Rectifies on the next clean tick |

### Behaviour while degraded

1. **Workers keep running.**  The fallback wakeup sweep keeps the
   journal moving; per-worker acquisition tickers also poll
   `acquire_next_runnable` independently.
2. **Outbox accumulates.**  Pending rows are durable.  When the
   broker returns, the relay publishes the backlog.
3. **Cross-instance fanout pauses.**  Remote pods will not see a
   thread-events advisory until the broker is back.  Clients that
   reconnect always replay from the durable event log, so
   correctness is preserved.

## 4. Backlog protection

Set `relay.backlog_threshold` to opt into in-process protection:

```yaml
relay:
  enabled: true
  backlog_threshold:
    soft: 1000     # latency_layer flips to Degraded above this
    hard: 10000    # alerting target — page on-call
  broker:
    !amqp …
```

**The relay never stops publishing.**  The thresholds drive the
operator response, not the workload.

### What the soft band does

* The relay observes the unpublished count after each backfill drain
  and on every empty steady-state tick.
* Soft band exceeded → `LatencyLayerHealth::Degraded`, JSON status
  field `degraded`, log line tagged `metric=relay_backlog`.

### What the hard band does

* Logs a separate `WARN` line with `pending`, `soft`, `hard` so the
  alert pipeline can route to a different rule.
* No additional behaviour change in process — the goal is operator
  visibility.

### Recommended alerting

| Alert | Expression | Severity |
|-------|------------|----------|
| `relay-backlog-soft` | `relay_backlog_max > soft` for 5 m | warning |
| `relay-backlog-hard` | `relay_backlog_max > hard` for 5 m | page |
| `relay-failures` | `relay_failed_total` rate > 0 for 10 m | warning |
| `relay-expired` | `relay_expired_total` rate > 0 for 5 m | warning (data loss imminent) |
| `relay-degraded` | `latency_layer == "degraded"` for 10 m | warning |

Counter names refer to the `MetricsRecorder` field schema; the
`LoggingMetricsRecorder` emits each as a `metric=<name>` log line.

## 5. Multi-instance topology

The thread-events watch consumer **must** receive every advisory on
**every** instance — competing-consumer queues are wrong here, because
a peer that wins the delivery cannot nudge another peer's local
subscribers.

### Correct topology

```text
    ┌──────────┐     ┌──── pod-A queue ──── pod-A consumer
    │ producer │  ►  │
    │ (any pod)│  ►  │
    └──────────┘     └──── pod-B queue ──── pod-B consumer
                          (one queue per pod, fanout exchange)
```

* Each pod declares its own queue, e.g.
  `agent_sdk.thread_events.<pod-id>`.
* The exchange is `topic` with the routing key
  `<routing_key_prefix>.thread_events_available`.
* Queue names should include the pod identity so a pod re-binding to
  the same queue picks up the buffered backlog after a restart.

### Wrong topology (will silently lose live-tail wakeups)

* All pods consume from the same queue → only one pod gets the
  advisory → other pods' subscribers stall until the durable replay
  fires on reconnect.

The contract is documented in
`crates/agent-service-host/src/broker/amqp_thread_events_consumer.rs`
module docs.

## 6. Retention janitor + replay floor semantics

The janitor advances the retention floor for committed events past
the configured `event_ttl_secs`.  It always respects the **outbox
safety bound** — it never advances past a sequence with an
unpublished outbox row, because doing so would create a replay gap
visible to a remote pod once the relay catches up.

### Behaviour

* `events_purged > 0`, `floors_advanced > 0` → janitor is working.
* `events_purged == 0` while events are clearly past TTL →
  unpublished outbox rows are pinning the floor.  Inspect the relay.
* `checkpoints_pruned` always respects `checkpoint_max_per_thread`
  and never deletes the most-recent checkpoint.

### Replay-gap behaviour

A "replay gap" is a window where an event has been retention-pruned
locally but a remote pod has never received the corresponding
advisory.  The Phase 8.6 contract guarantees this cannot happen as
long as the relay is making progress.  When the relay is degraded,
gaps still cannot form because `min_unpublished_sequence` keeps the
floor pinned to the oldest unpublished row.

## 7. Recovery procedures

### Broker restored after extended outage

1. Confirm `latency_layer: healthy` returns within
   `relay.poll_interval_secs` × 2.
2. Watch `relay_delivered` rate — it should burst until the backlog
   drains.
3. Confirm `outbox_pending_total` returns to baseline.
4. Confirm cross-instance live-tail nudges work: trigger a write on
   pod A and confirm pod B's subscribers receive within
   `poll_interval` × 2.

### Suspected duplicate publishes

This is **expected** after a relay crash window: the row was
published, the broker acked, and the worker crashed before
`mark_delivered`.  The reclaim sweep returns the row to `Pending`
and the next worker republishes.  Consumer-side idempotency handles
the duplicate.

The Phase 8.7 regression test
`phase_8_publish_then_crash_preserves_correctness` proves this end
to end.

### Stuck `Claimed` row not being reclaimed

1. Check `metric=relay_reclaim` logs — value of `reclaimed` is
   non-zero on every reclaim cycle.
2. Confirm the row's `claimed_at` is older than `claim_lease_secs`.
3. If a real worker is still alive but slow, increase
   `claim_lease_secs` so its work is not stolen mid-publish.

## 8. Configuration reference

```yaml
relay:
  enabled: true
  worker_id: "pod-1"
  batch_size: 128
  poll_interval_secs: 2
  claim_lease_secs: 60
  reclaim_interval_secs: 30
  retry_backoff_secs: 30
  backlog_threshold:
    soft: 1000
    hard: 10000
  broker:
    !amqp
      url: "amqp://user:pass@broker:5672/vhost"
      exchange: "agent_sdk.outbox"
      exchange_kind: topic
      declare_exchange: false

wakeup:
  enabled: true
  fallback_interval_secs: 5
  amqp_consumer:
    enabled: true
    config:
      queue: "agent_sdk.wakeup.pod-1"
      consumer_tag_prefix: "pod-1"
      declare_queue: true
      bind_queue: true

watch:
  enabled: true
  amqp_consumer:
    enabled: true
    config:
      queue: "agent_sdk.thread_events.pod-1"
      consumer_tag_prefix: "pod-1-watch"
      declare_queue: true
      bind_queue: true

retention:
  janitor_enabled: true
  janitor_interval_secs: 60
  janitor_batch_size: 100
  event_ttl_secs: 2592000   # 30 days
  checkpoint_max_per_thread: 100
```

See [`health-and-readiness.md`](./health-and-readiness.md) for the
HTTP probe contract and Kubernetes integration.
