//! `agent_server.*` metric instruments.
//!
//! Compiled only with `feature = "otel"`. The module owns a small,
//! lazily-initialised [`ServerMetrics`] singleton that the rest of
//! `agent-server` reaches via [`ServerMetrics::global`]. Every
//! histogram declared here pins its own bucket boundaries — we never
//! accept the SDK default boundaries because they are tuned for HTTP
//! timings, not the journal/relay/wakeup ranges this crate cares
//! about.
//!
//! ## Lifecycle
//!
//! Instruments are bound to whichever meter provider is current at
//! the first call to [`ServerMetrics::global`] / [`ServerMetrics::init`].
//! Tests that swap in a fresh `opentelemetry_sdk::metrics::SdkMeterProvider`
//! between cases must call [`ServerMetrics::reset_for_testing`] so
//! the next lookup rebuilds against the new provider.
//!
//! ## Naming
//!
//! Every instrument is namespaced under `agent_server.*` so
//! dashboards and alerts can correlate a metric with the durable
//! `agent-server` codepaths without translation. The `gen_ai.*`
//! namespace is reserved for the LLM client and is never
//! touched here.
//!
//! ## Workers and the host boundary
//!
//! `agent-server` does not own a worker pool — the actual worker
//! lifecycle (spawn, register, shutdown) lives in the
//! `agent-service-host` crate and the host application's shim. The
//! [`ServerMetrics::worker_started`] and [`ServerMetrics::worker_stopped`]
//! helpers are exposed here so those host crates can adjust the
//! shared `agent_server.workers.active` `UpDownCounter` without
//! redeclaring it. The metric itself lives next to its peers because
//! the dashboard story for "active workers vs queued tasks vs
//! lease-expired tasks" requires them all to come from the same
//! meter scope.
//!
//! ## Outbox-depth gauge
//!
//! [`ServerMetrics::register_outbox_depth_callback`] wires a
//! caller-supplied `Fn() -> u64` onto an `ObservableGauge` so each
//! metric export pulls the latest depth. Owners of the closure are
//! responsible for any TTL caching needed to keep durable backends
//! (typically Postgres) from being stampeded by the meter export
//! interval. The in-memory test path uses a trivial closure over a
//! `HashMap` walk; the host wiring passes a Postgres-backed closure
//! with a short TTL.

use std::sync::{Arc, RwLock};

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram, ObservableGauge, UpDownCounter};

use crate::journal::execution_intent::ToolEffectClass;
use crate::journal::recovery::{FailureReason, RecoveryAction, RecoveryRecord};
use crate::journal::relay::RelayTick;
use crate::journal::tool_audit::ToolAuditEvent;

// ─────────────────────────────────────────────────────────────────────
// Bucket boundaries
// ─────────────────────────────────────────────────────────────────────

/// Bucket boundaries for `agent_server.tasks.execution.duration`
/// (seconds). Spans an LLM-call's typical 10 ms .. multi-minute range
/// in the same coarse log-2 cadence the SDK already uses for LLM
/// metrics.
const TASK_DURATION_BUCKETS_S: &[f64] = &[0.01, 0.04, 0.16, 0.64, 2.56, 10.24, 40.96];

/// Bucket boundaries for `agent_server.journal.commit.duration`
/// (seconds). Journal commits are dominated by Postgres round-trips
/// so the boundaries are tighter at the short end.
const COMMIT_DURATION_BUCKETS_S: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

/// Bucket boundaries for
/// `agent_server.thread_events_watch.lag_ms`. Tuned for AMQP
/// advisory consumer lag; we expect the bulk of healthy lag to land
/// under 100 ms, with stragglers flagged at 2 s and pathological
/// cases at 10 s.
const WATCH_LAG_BUCKETS_MS: &[f64] = &[1.0, 5.0, 25.0, 100.0, 500.0, 2_000.0, 10_000.0];

// ─────────────────────────────────────────────────────────────────────
// Attribute keys and stable values
// ─────────────────────────────────────────────────────────────────────

/// Stable string keys and values used as metric attributes.
///
/// Centralised here so the dashboard / alert configuration in the
/// `agent-service-host` crate and the host application can reference the same
/// constants instead of re-typing magic strings.
pub mod attrs {
    /// Task kind: `root` / `subagent` / `tool` / `listen`.
    pub const KIND: &str = "kind";
    /// Task outcome: `done` / `error` / `cancelled` / `awaiting_confirmation` / `suspended`.
    pub const OUTCOME: &str = "outcome";
    /// Journal commit step: `atomic` / `non_atomic`.
    pub const EVENT_KIND: &str = "event_kind";
    /// Relay tick outcome: `claimed` / `delivered` / `failed` / `expired`.
    pub const RELAY_OUTCOME: &str = "outcome";
    /// Lease-expiry reason: `requeued` / `<failure_reason>`.
    pub const REASON: &str = "reason";
    /// Tool audit event outcome: discriminant of [`ToolAuditEventKind`](crate::journal::tool_audit::ToolAuditEventKind).
    pub const AUDIT_OUTCOME: &str = "outcome";
    /// Tool effect class on audit events.
    pub const EFFECT_CLASS: &str = "effect_class";

    pub const KIND_ROOT: &str = "root";
    pub const KIND_SUBAGENT: &str = "subagent";
    pub const KIND_TOOL: &str = "tool";
    /// Reserved variant. The current journal does not model a
    /// dedicated "listen" task kind separately from
    /// [`crate::journal::TaskKind::ToolRuntime`]. Kept here so a
    /// future split does not need a migration.
    pub const KIND_LISTEN: &str = "listen";

    pub const OUTCOME_DONE: &str = "done";
    pub const OUTCOME_ERROR: &str = "error";
    pub const OUTCOME_CANCELLED: &str = "cancelled";
    pub const OUTCOME_AWAITING_CONFIRMATION: &str = "awaiting_confirmation";
    pub const OUTCOME_SUSPENDED: &str = "suspended";

    pub const RELAY_CLAIMED: &str = "claimed";
    pub const RELAY_DELIVERED: &str = "delivered";
    pub const RELAY_FAILED: &str = "failed";
    pub const RELAY_EXPIRED: &str = "expired";

    pub const REASON_REQUEUED: &str = "requeued";

    pub const COMMIT_KIND_ATOMIC: &str = "atomic";
    pub const COMMIT_KIND_NON_ATOMIC: &str = "non_atomic";
}

// ─────────────────────────────────────────────────────────────────────
// ServerMetrics
// ─────────────────────────────────────────────────────────────────────

/// Container for every metric instrument the `agent-server` crate
/// records.
///
/// Held behind an `Arc` so call sites that need frequent recording
/// (per task, per tick) can clone the handle into an async future
/// without re-walking a `Mutex`.
pub struct ServerMetrics {
    pub(crate) workers_active: UpDownCounter<i64>,
    pub(crate) tasks_acquired: Counter<u64>,
    pub(crate) tasks_execution_duration: Histogram<f64>,
    pub(crate) tasks_lease_expired: Counter<u64>,
    pub(crate) journal_commit_duration: Histogram<f64>,
    pub(crate) relay_tick: Counter<u64>,
    pub(crate) wakeup_fallback_sweep: Counter<u64>,
    pub(crate) thread_events_watch_lag_ms: Histogram<f64>,
    pub(crate) tool_audit_outcome: Counter<u64>,
    /// Shared meter handle reused by `register_outbox_depth_callback`
    /// callers so they bind their gauge to the same provider as
    /// every other instrument on this struct.
    meter: opentelemetry::metrics::Meter,
}

impl std::fmt::Debug for ServerMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerMetrics").finish_non_exhaustive()
    }
}

static METRICS: RwLock<Option<Arc<ServerMetrics>>> = RwLock::new(None);

impl ServerMetrics {
    /// Build instruments under the supplied meter scope and cache
    /// the resulting handle for subsequent [`ServerMetrics::global`]
    /// callers.
    ///
    /// If the cache is already populated the supplied `name` is
    /// ignored and the cached handle is returned — the first caller
    /// in the process wins. Tests that rotate the global meter
    /// provider between cases must call
    /// [`ServerMetrics::reset_for_testing`] beforehand so the next
    /// `init` rebuilds against the fresh provider.
    #[must_use]
    pub fn init(name: &'static str) -> Arc<Self> {
        if let Some(existing) = read_cached() {
            return existing;
        }

        let built = Arc::new(Self::build(name));
        write_cached(Arc::clone(&built));
        built
    }

    /// Convenience wrapper that initialises the singleton under the
    /// `agent-server` meter scope.
    #[must_use]
    pub fn global() -> Arc<Self> {
        Self::init(env!("CARGO_PKG_NAME"))
    }

    /// Drop the cached singleton.
    ///
    /// Test-only escape hatch. Production code never calls this —
    /// the instruments are bound to whichever meter provider was
    /// current when [`init`](Self::init) ran, and rebuilding them
    /// against a different provider mid-run would silently lose
    /// data points.
    pub fn reset_for_testing() {
        let mut guard = match METRICS.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
    }

    /// Increment `agent_server.workers.active`.
    ///
    /// Call this from the host worker loop when a worker registers.
    /// The matching [`worker_stopped`](Self::worker_stopped) keeps
    /// the gauge balanced.
    pub fn worker_started(&self) {
        self.workers_active.add(1, &[]);
    }

    /// Decrement `agent_server.workers.active`.
    ///
    /// Call this when a worker exits its loop (graceful shutdown,
    /// panic-recovery, lease loss, …).
    pub fn worker_stopped(&self) {
        self.workers_active.add(-1, &[]);
    }

    /// Record a single `agent_server.tasks.execution.duration`
    /// sample.
    ///
    /// `kind` and `outcome` are intentionally `&'static str` so the
    /// caller cannot accidentally pass an empty string — every
    /// sample is guaranteed to carry both attributes.
    pub fn record_task_execution(
        &self,
        kind: &'static str,
        outcome: &'static str,
        duration_secs: f64,
    ) {
        self.tasks_execution_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::KIND, kind),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
    }

    /// Record a single `agent_server.tasks.acquired` increment.
    pub fn record_task_acquired(&self, kind: &'static str) {
        self.tasks_acquired
            .add(1, &[KeyValue::new(attrs::KIND, kind)]);
    }

    /// Record one `agent_server.tasks.lease_expired` increment per
    /// row in `records`, attributing each by its sweep outcome.
    ///
    /// `Requeue` rows carry `reason="requeued"`; `FailClosed` rows
    /// carry the failure-reason discriminant
    /// (`error_prefix` from [`FailureReason`]). `NoAction` rows are
    /// not counted because the sweep does not actually touch them.
    pub fn record_lease_expiry_outcomes(&self, records: &[RecoveryRecord]) {
        for record in records {
            let reason = match record.action {
                RecoveryAction::Requeue => attrs::REASON_REQUEUED,
                RecoveryAction::FailClosed(reason) => failure_reason_label(reason),
                RecoveryAction::NoAction => continue,
            };
            self.tasks_lease_expired
                .add(1, &[KeyValue::new(attrs::REASON, reason)]);
        }
    }

    /// Record a single `agent_server.journal.commit.duration` sample
    /// for one of the two commit paths.
    pub fn record_journal_commit(&self, event_kind: &'static str, duration_secs: f64) {
        self.journal_commit_duration.record(
            duration_secs,
            &[KeyValue::new(attrs::EVENT_KIND, event_kind)],
        );
    }

    /// Fan out a [`RelayTick`]'s counters onto
    /// `agent_server.relay.tick`.
    ///
    /// Each non-zero counter contributes one `add()` call with the
    /// matching outcome attribute and the counter's value.
    pub fn record_relay_tick(&self, tick: &RelayTick) {
        let entries: [(usize, &'static str); 4] = [
            (tick.claimed, attrs::RELAY_CLAIMED),
            (tick.delivered, attrs::RELAY_DELIVERED),
            (tick.failed, attrs::RELAY_FAILED),
            (tick.expired, attrs::RELAY_EXPIRED),
        ];
        for (count, outcome) in entries {
            if count == 0 {
                continue;
            }
            let attr = [KeyValue::new(attrs::RELAY_OUTCOME, outcome)];
            self.relay_tick.add(count as u64, &attr);
        }
    }

    /// Record a single `agent_server.wakeup.fallback_sweep`
    /// increment. Called once per sweep tick.
    pub fn record_fallback_sweep(&self) {
        self.wakeup_fallback_sweep.add(1, &[]);
    }

    /// Record one `agent_server.thread_events_watch.lag_ms` sample.
    pub fn record_thread_events_watch_lag(&self, lag_ms: f64) {
        self.thread_events_watch_lag_ms.record(lag_ms, &[]);
    }

    /// Record one `agent_server.tool_audit.outcome` increment for
    /// the given tool-audit event.
    pub fn record_tool_audit(&self, event: &ToolAuditEvent) {
        let attrs = [
            KeyValue::new(attrs::AUDIT_OUTCOME, event.kind.as_str()),
            KeyValue::new(attrs::EFFECT_CLASS, effect_class_label(event.effect_class)),
        ];
        self.tool_audit_outcome.add(1, &attrs);
    }

    /// Register a callback that publishes
    /// `agent_server.journal.outbox_depth` on every metric export.
    ///
    /// The returned [`ObservableGauge`] is held by the caller for
    /// the lifetime of the registration; dropping it deregisters
    /// the callback. Pass an `Arc<AtomicU64>` (or any cheap shared
    /// state) into the closure so the gauge keeps reading the
    /// caller's latest value.
    ///
    /// Owners are responsible for caching the underlying value
    /// (typically a 5 s TTL) to avoid stampeding durable backends
    /// on every export.
    #[must_use]
    pub fn register_outbox_depth_callback(
        &self,
        provider: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> ObservableGauge<u64> {
        self.meter
            .u64_observable_gauge("agent_server.journal.outbox_depth")
            .with_description("Pending outbox rows awaiting relay.")
            .with_callback(move |observer| observer.observe(provider(), &[]))
            .build()
    }

    fn build(scope: &'static str) -> Self {
        let meter = global::meter(scope);

        let workers_active = meter
            .i64_up_down_counter("agent_server.workers.active")
            .with_description("Active worker processes ready to acquire tasks.")
            .build();
        let tasks_acquired = meter
            .u64_counter("agent_server.tasks.acquired")
            .with_description("Tasks acquired by a worker for execution.")
            .build();
        let tasks_execution_duration = meter
            .f64_histogram("agent_server.tasks.execution.duration")
            .with_unit("s")
            .with_description("Wall-clock duration of a single task execution.")
            .with_boundaries(TASK_DURATION_BUCKETS_S.to_vec())
            .build();
        let tasks_lease_expired = meter
            .u64_counter("agent_server.tasks.lease_expired")
            .with_description(
                "Leases released by the expiry sweep, attributed by recovery outcome.",
            )
            .build();
        let journal_commit_duration = meter
            .f64_histogram("agent_server.journal.commit.duration")
            .with_unit("s")
            .with_description("Duration of an atomic completed-turn commit.")
            .with_boundaries(COMMIT_DURATION_BUCKETS_S.to_vec())
            .build();
        let relay_tick = meter
            .u64_counter("agent_server.relay.tick")
            .with_description("Outbox relay rows observed per tick, attributed by outcome.")
            .build();
        let wakeup_fallback_sweep = meter
            .u64_counter("agent_server.wakeup.fallback_sweep")
            .with_description("Times the fallback wakeup sweep fired a pulse.")
            .build();
        let thread_events_watch_lag_ms = meter
            .f64_histogram("agent_server.thread_events_watch.lag_ms")
            .with_unit("ms")
            .with_description("Time spent dispatching a thread_events_available advisory.")
            .with_boundaries(WATCH_LAG_BUCKETS_MS.to_vec())
            .build();
        let tool_audit_outcome = meter
            .u64_counter("agent_server.tool_audit.outcome")
            .with_description("Tool audit events recorded, attributed by lifecycle outcome.")
            .build();

        Self {
            workers_active,
            tasks_acquired,
            tasks_execution_duration,
            tasks_lease_expired,
            journal_commit_duration,
            relay_tick,
            wakeup_fallback_sweep,
            thread_events_watch_lag_ms,
            tool_audit_outcome,
            meter,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────

fn read_cached() -> Option<Arc<ServerMetrics>> {
    let guard = match METRICS.read() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.as_ref().map(Arc::clone)
}

fn write_cached(value: Arc<ServerMetrics>) {
    let mut guard = match METRICS.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(value);
    }
    // If a concurrent caller raced us in, drop the freshly-built
    // handle on the floor. Both copies are functionally equivalent
    // (same scope, same global meter), so the duplicate is
    // harmless.
}

const fn failure_reason_label(reason: FailureReason) -> &'static str {
    reason.error_prefix()
}

const fn effect_class_label(class: ToolEffectClass) -> &'static str {
    match class {
        ToolEffectClass::ReplaySafe => "replay_safe",
        ToolEffectClass::SideEffecting => "side_effecting",
        ToolEffectClass::Resumable => "resumable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_reason_label_matches_error_prefix() {
        // The label IS the error prefix — keep them coupled so a
        // dashboard query like `reason="retry_budget_exhausted"`
        // exactly matches the wire format written by the journal.
        for r in [
            FailureReason::RetryBudgetExhausted,
            FailureReason::LeaseExpiredBudgetExhausted,
            FailureReason::UnsafePreparedOperationRecovery,
        ] {
            assert_eq!(failure_reason_label(r), r.error_prefix());
        }
    }

    #[test]
    fn effect_class_label_is_snake_case_serde() {
        // The label is the same snake_case discriminant the
        // `ToolEffectClass` serde representation uses, keeping the
        // metric attribute aligned with the audit-row JSON value.
        for (class, expected) in [
            (ToolEffectClass::ReplaySafe, "replay_safe"),
            (ToolEffectClass::SideEffecting, "side_effecting"),
            (ToolEffectClass::Resumable, "resumable"),
        ] {
            assert_eq!(effect_class_label(class), expected);
        }
    }
}
