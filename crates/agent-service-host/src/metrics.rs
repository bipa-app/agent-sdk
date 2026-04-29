//! GA observability surface for the service host.
//!
//! Phase 8.7 closes Phase 8 by exposing **programmatic counters** for
//! every background loop the relay/wakeup/watch/janitor stack already
//! drives.  The trait-based [`MetricsRecorder`] sits underneath the
//! existing `tracing` calls so a future Prometheus or OpenTelemetry
//! exporter can attach without rewriting the schedulers.
//!
//! # Why a trait, not a concrete exporter
//!
//! The host already emits structured `tracing` events for every
//! interesting transition (relay tick, reclaim, janitor cycle).  What
//! GA needs is a stable seam an alerting pipeline can consume.  Adding
//! a real exporter (`prometheus`, `opentelemetry`) right now would
//! drag a wide dependency surface into the host crate before we know
//! what platform a deploy target uses.  The trait deliverable is
//! intentionally narrow — concrete exporters live in deploy crates.
//!
//! # Recorders shipped here
//!
//! | Recorder | Use |
//! |----------|-----|
//! | [`InMemoryMetricsRecorder`] | Tests and `/metrics` JSON dumps |
//! | [`LoggingMetricsRecorder`] | Production: emits `tracing::info!` events with stable field names |
//! | [`NoopMetricsRecorder`] | Default when no observability is wired |
//!
//! Every recorder is `Send + Sync` and cheap to clone via `Arc`, so
//! schedulers hold a single `Arc<dyn MetricsRecorder>` and call it on
//! every transition.
//!
//! # Backlog protection
//!
//! [`BacklogThreshold`] carries the soft and hard alerting bands.
//! The relay scheduler reads the thresholds (when configured) and
//! flips [`crate::health::LatencyLayerHealth`] to `Degraded` once
//! observed per-thread pending counts cross the soft band.  The hard
//! band is alerting-only — it does **not** stop publishing, because
//! the journal already guarantees correctness even when the relay
//! lags.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_server::journal::JanitorCycleReport;
use agent_server::journal::TaskWakeupOutcome;
use agent_server::journal::ThreadEventsWatchOutcome;
use agent_server::journal::relay::RelayTick;
use serde::{Deserialize, Serialize};
use tracing::info;

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Where a wakeup nudge originated.
///
/// Recorders use this to break wakeup throughput out by source so
/// operators can spot a silent broker (everything coming in via the
/// fallback sweep) or a saturated consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeupSource {
    /// A broker delivery (AMQP consumer).
    Broker,
    /// The periodic [`agent_server::journal::FallbackWakeupSweep`].
    FallbackSweep,
    /// The per-worker acquisition ticker.
    AcquisitionTicker,
}

/// The narrow seam every background loop in the service host reports
/// observability events through.
///
/// Methods take `&self` so a single `Arc<dyn MetricsRecorder>` can be
/// shared across schedulers without locks.  Implementations are
/// expected to be cheap on every call (atomic increments, log-line
/// emissions) — none of the schedulers await on the recorder.
pub trait MetricsRecorder: Send + Sync + std::fmt::Debug {
    /// One full relay tick (claim → publish → mark) finished.
    fn record_relay_tick(&self, tick: &RelayTick, duration_ms: u64);

    /// A claim-reclaim sweep released `reclaimed` stale claims.
    fn record_relay_reclaim(&self, reclaimed: u64);

    /// Snapshot of the current unpublished outbox backlog and the
    /// configured soft threshold.  `Some(threshold)` means the host
    /// has backlog protection wired; `None` means no threshold is
    /// configured and the value is informational only.
    fn record_relay_backlog(&self, pending: u64, soft_threshold: Option<u64>);

    /// A wakeup nudge dispatched by the scheduler.  `outcome` reports
    /// what the journal-side handler did with the nudge.
    fn record_wakeup_nudge(&self, source: WakeupSource, outcome: &TaskWakeupOutcome);

    /// A `thread_events_available` advisory was processed.
    fn record_watch_advisory(&self, outcome: &ThreadEventsWatchOutcome);

    /// The retention janitor finished one cycle.
    fn record_janitor_cycle(&self, report: &JanitorCycleReport);

    /// One lease-sweep pass released `released` expired leases.
    fn record_lease_sweep(&self, released: usize);
}

// ─────────────────────────────────────────────────────────────────────
// No-op recorder
// ─────────────────────────────────────────────────────────────────────

/// Recorder that drops every event.  Used when the host runs without
/// any observability wiring (e.g. in tests that don't care about
/// metrics).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopMetricsRecorder;

impl MetricsRecorder for NoopMetricsRecorder {
    fn record_relay_tick(&self, _tick: &RelayTick, _duration_ms: u64) {}
    fn record_relay_reclaim(&self, _reclaimed: u64) {}
    fn record_relay_backlog(&self, _pending: u64, _soft_threshold: Option<u64>) {}
    fn record_wakeup_nudge(&self, _source: WakeupSource, _outcome: &TaskWakeupOutcome) {}
    fn record_watch_advisory(&self, _outcome: &ThreadEventsWatchOutcome) {}
    fn record_janitor_cycle(&self, _report: &JanitorCycleReport) {}
    fn record_lease_sweep(&self, _released: usize) {}
}

// ─────────────────────────────────────────────────────────────────────
// Logging recorder
// ─────────────────────────────────────────────────────────────────────

/// Production recorder: emits structured `tracing::info!` events with
/// stable field names so an external aggregator (Vector, Fluent Bit,
/// Loki) can scrape them without parsing free-text log lines.
///
/// The field names are the documented metric contract — changing
/// them is a breaking observability change and must be called out in
/// the changelog.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoggingMetricsRecorder;

impl MetricsRecorder for LoggingMetricsRecorder {
    fn record_relay_tick(&self, tick: &RelayTick, duration_ms: u64) {
        info!(
            metric = "relay_tick",
            claimed = tick.claimed,
            delivered = tick.delivered,
            failed = tick.failed,
            expired = tick.expired,
            duration_ms,
            "relay tick observed",
        );
    }

    fn record_relay_reclaim(&self, reclaimed: u64) {
        info!(
            metric = "relay_reclaim",
            reclaimed, "relay reclaim observed",
        );
    }

    fn record_relay_backlog(&self, pending: u64, soft_threshold: Option<u64>) {
        info!(
            metric = "relay_backlog",
            pending,
            soft_threshold = soft_threshold.unwrap_or(0),
            threshold_configured = soft_threshold.is_some(),
            "relay backlog observed",
        );
    }

    fn record_wakeup_nudge(&self, source: WakeupSource, outcome: &TaskWakeupOutcome) {
        info!(
            metric = "wakeup_nudge",
            source = wakeup_source_label(source),
            outcome = wakeup_outcome_label(outcome),
            "wakeup nudge observed",
        );
    }

    fn record_watch_advisory(&self, outcome: &ThreadEventsWatchOutcome) {
        info!(
            metric = "watch_advisory",
            outcome = watch_outcome_label(outcome),
            emitted = outcome.emitted_count(),
            "watch advisory observed",
        );
    }

    fn record_janitor_cycle(&self, report: &JanitorCycleReport) {
        info!(
            metric = "janitor_cycle",
            threads_scanned = report.threads_scanned,
            events_purged = report.events_purged,
            checkpoints_pruned = report.checkpoints_pruned,
            floors_advanced = report.floors_advanced,
            "janitor cycle observed",
        );
    }

    fn record_lease_sweep(&self, released: usize) {
        info!(metric = "lease_sweep", released, "lease sweep observed",);
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory recorder (tests + diagnostics)
// ─────────────────────────────────────────────────────────────────────

/// Aggregate counters maintained by [`InMemoryMetricsRecorder`].
///
/// Snapshots are cheap to construct and trivially comparable in tests;
/// every field is a saturating counter that never decrements.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MetricsSnapshot {
    pub relay_ticks: u64,
    pub relay_delivered: u64,
    pub relay_failed: u64,
    pub relay_expired: u64,
    pub relay_claimed: u64,
    pub relay_total_duration_ms: u64,
    pub relay_reclaimed: u64,
    pub relay_backlog_observations: u64,
    pub relay_backlog_max: u64,
    pub relay_backlog_threshold_breaches: u64,
    pub wakeup_broker: u64,
    pub wakeup_fallback_sweep: u64,
    pub wakeup_acquisition_ticker: u64,
    pub wakeup_nudged: u64,
    pub wakeup_not_runnable: u64,
    pub wakeup_missing: u64,
    pub watch_forwarded: u64,
    pub watch_already_current: u64,
    pub watch_unknown_thread: u64,
    pub watch_events_emitted: u64,
    pub janitor_cycles: u64,
    pub janitor_threads_scanned: u64,
    pub janitor_events_purged: u64,
    pub janitor_checkpoints_pruned: u64,
    pub janitor_floors_advanced: u64,
    pub lease_sweep_cycles: u64,
    pub lease_sweep_released: u64,
}

#[derive(Debug, Default)]
struct InMemoryCounters {
    relay_ticks: AtomicU64,
    relay_delivered: AtomicU64,
    relay_failed: AtomicU64,
    relay_expired: AtomicU64,
    relay_claimed: AtomicU64,
    relay_total_duration_ms: AtomicU64,
    relay_reclaimed: AtomicU64,
    relay_backlog_observations: AtomicU64,
    relay_backlog_max: AtomicU64,
    relay_backlog_threshold_breaches: AtomicU64,
    wakeup_broker: AtomicU64,
    wakeup_fallback_sweep: AtomicU64,
    wakeup_acquisition_ticker: AtomicU64,
    wakeup_nudged: AtomicU64,
    wakeup_not_runnable: AtomicU64,
    wakeup_missing: AtomicU64,
    watch_forwarded: AtomicU64,
    watch_already_current: AtomicU64,
    watch_unknown_thread: AtomicU64,
    watch_events_emitted: AtomicU64,
    janitor_cycles: AtomicU64,
    janitor_threads_scanned: AtomicU64,
    janitor_events_purged: AtomicU64,
    janitor_checkpoints_pruned: AtomicU64,
    janitor_floors_advanced: AtomicU64,
    lease_sweep_cycles: AtomicU64,
    lease_sweep_released: AtomicU64,
}

/// In-memory recorder that maintains atomic counters readable through
/// [`InMemoryMetricsRecorder::snapshot`].
///
/// Cheap to clone; every clone shares the same underlying counters,
/// so an instance handed to the relay scheduler and another handed to
/// a test assertion observe the same totals.
#[derive(Clone, Debug, Default)]
pub struct InMemoryMetricsRecorder {
    inner: Arc<InMemoryCounters>,
}

impl InMemoryMetricsRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a snapshot of the current counter values.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let c = &self.inner;
        MetricsSnapshot {
            relay_ticks: c.relay_ticks.load(Ordering::Relaxed),
            relay_delivered: c.relay_delivered.load(Ordering::Relaxed),
            relay_failed: c.relay_failed.load(Ordering::Relaxed),
            relay_expired: c.relay_expired.load(Ordering::Relaxed),
            relay_claimed: c.relay_claimed.load(Ordering::Relaxed),
            relay_total_duration_ms: c.relay_total_duration_ms.load(Ordering::Relaxed),
            relay_reclaimed: c.relay_reclaimed.load(Ordering::Relaxed),
            relay_backlog_observations: c.relay_backlog_observations.load(Ordering::Relaxed),
            relay_backlog_max: c.relay_backlog_max.load(Ordering::Relaxed),
            relay_backlog_threshold_breaches: c
                .relay_backlog_threshold_breaches
                .load(Ordering::Relaxed),
            wakeup_broker: c.wakeup_broker.load(Ordering::Relaxed),
            wakeup_fallback_sweep: c.wakeup_fallback_sweep.load(Ordering::Relaxed),
            wakeup_acquisition_ticker: c.wakeup_acquisition_ticker.load(Ordering::Relaxed),
            wakeup_nudged: c.wakeup_nudged.load(Ordering::Relaxed),
            wakeup_not_runnable: c.wakeup_not_runnable.load(Ordering::Relaxed),
            wakeup_missing: c.wakeup_missing.load(Ordering::Relaxed),
            watch_forwarded: c.watch_forwarded.load(Ordering::Relaxed),
            watch_already_current: c.watch_already_current.load(Ordering::Relaxed),
            watch_unknown_thread: c.watch_unknown_thread.load(Ordering::Relaxed),
            watch_events_emitted: c.watch_events_emitted.load(Ordering::Relaxed),
            janitor_cycles: c.janitor_cycles.load(Ordering::Relaxed),
            janitor_threads_scanned: c.janitor_threads_scanned.load(Ordering::Relaxed),
            janitor_events_purged: c.janitor_events_purged.load(Ordering::Relaxed),
            janitor_checkpoints_pruned: c.janitor_checkpoints_pruned.load(Ordering::Relaxed),
            janitor_floors_advanced: c.janitor_floors_advanced.load(Ordering::Relaxed),
            lease_sweep_cycles: c.lease_sweep_cycles.load(Ordering::Relaxed),
            lease_sweep_released: c.lease_sweep_released.load(Ordering::Relaxed),
        }
    }
}

impl MetricsRecorder for InMemoryMetricsRecorder {
    fn record_relay_tick(&self, tick: &RelayTick, duration_ms: u64) {
        let c = &self.inner;
        c.relay_ticks.fetch_add(1, Ordering::Relaxed);
        c.relay_delivered
            .fetch_add(tick.delivered as u64, Ordering::Relaxed);
        c.relay_failed
            .fetch_add(tick.failed as u64, Ordering::Relaxed);
        c.relay_expired
            .fetch_add(tick.expired as u64, Ordering::Relaxed);
        c.relay_claimed
            .fetch_add(tick.claimed as u64, Ordering::Relaxed);
        c.relay_total_duration_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn record_relay_reclaim(&self, reclaimed: u64) {
        self.inner
            .relay_reclaimed
            .fetch_add(reclaimed, Ordering::Relaxed);
    }

    fn record_relay_backlog(&self, pending: u64, soft_threshold: Option<u64>) {
        let c = &self.inner;
        c.relay_backlog_observations.fetch_add(1, Ordering::Relaxed);
        c.relay_backlog_max.fetch_max(pending, Ordering::Relaxed);
        if let Some(threshold) = soft_threshold
            && pending > threshold
        {
            c.relay_backlog_threshold_breaches
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_wakeup_nudge(&self, source: WakeupSource, outcome: &TaskWakeupOutcome) {
        let c = &self.inner;
        match source {
            WakeupSource::Broker => c.wakeup_broker.fetch_add(1, Ordering::Relaxed),
            WakeupSource::FallbackSweep => c.wakeup_fallback_sweep.fetch_add(1, Ordering::Relaxed),
            WakeupSource::AcquisitionTicker => {
                c.wakeup_acquisition_ticker.fetch_add(1, Ordering::Relaxed)
            }
        };
        match outcome {
            TaskWakeupOutcome::Nudged { .. } => c.wakeup_nudged.fetch_add(1, Ordering::Relaxed),
            TaskWakeupOutcome::NotRunnable { .. } => {
                c.wakeup_not_runnable.fetch_add(1, Ordering::Relaxed)
            }
            TaskWakeupOutcome::Missing => c.wakeup_missing.fetch_add(1, Ordering::Relaxed),
        };
    }

    fn record_watch_advisory(&self, outcome: &ThreadEventsWatchOutcome) {
        let c = &self.inner;
        match outcome {
            ThreadEventsWatchOutcome::Forwarded { emitted_count, .. } => {
                c.watch_forwarded.fetch_add(1, Ordering::Relaxed);
                c.watch_events_emitted
                    .fetch_add(u64::from(*emitted_count), Ordering::Relaxed);
            }
            ThreadEventsWatchOutcome::AlreadyCurrent { .. } => {
                c.watch_already_current.fetch_add(1, Ordering::Relaxed);
            }
            ThreadEventsWatchOutcome::UnknownThread => {
                c.watch_unknown_thread.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_janitor_cycle(&self, report: &JanitorCycleReport) {
        let c = &self.inner;
        c.janitor_cycles.fetch_add(1, Ordering::Relaxed);
        c.janitor_threads_scanned
            .fetch_add(u64::from(report.threads_scanned), Ordering::Relaxed);
        c.janitor_events_purged
            .fetch_add(report.events_purged, Ordering::Relaxed);
        c.janitor_checkpoints_pruned
            .fetch_add(report.checkpoints_pruned, Ordering::Relaxed);
        c.janitor_floors_advanced
            .fetch_add(u64::from(report.floors_advanced), Ordering::Relaxed);
    }

    fn record_lease_sweep(&self, released: usize) {
        let c = &self.inner;
        c.lease_sweep_cycles.fetch_add(1, Ordering::Relaxed);
        c.lease_sweep_released
            .fetch_add(released as u64, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Backlog protection
// ─────────────────────────────────────────────────────────────────────

/// Soft / hard alerting bands for unpublished outbox backlog.
///
/// The relay scheduler treats `soft` as the point past which the
/// latency layer is reported `Degraded`.  `hard` is alerting-only —
/// the relay does **not** stop publishing when the hard band is
/// exceeded because the journal still guarantees correctness;
/// operators read the snapshot and decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BacklogThreshold {
    /// Backlog size above which the latency layer is `Degraded`.
    pub soft: u64,
    /// Backlog size at which alerts page on-call.
    pub hard: u64,
}

impl Default for BacklogThreshold {
    fn default() -> Self {
        Self {
            soft: 1_000,
            hard: 10_000,
        }
    }
}

impl BacklogThreshold {
    /// Returns `true` when `pending` exceeds the soft band.
    #[must_use]
    pub const fn breaches_soft(&self, pending: u64) -> bool {
        pending > self.soft
    }

    /// Returns `true` when `pending` exceeds the hard band.
    #[must_use]
    pub const fn breaches_hard(&self, pending: u64) -> bool {
        pending > self.hard
    }
}

// ─────────────────────────────────────────────────────────────────────
// Stable label helpers
// ─────────────────────────────────────────────────────────────────────

const fn wakeup_source_label(source: WakeupSource) -> &'static str {
    match source {
        WakeupSource::Broker => "broker",
        WakeupSource::FallbackSweep => "fallback_sweep",
        WakeupSource::AcquisitionTicker => "acquisition_ticker",
    }
}

const fn wakeup_outcome_label(outcome: &TaskWakeupOutcome) -> &'static str {
    match outcome {
        TaskWakeupOutcome::Nudged { .. } => "nudged",
        TaskWakeupOutcome::NotRunnable { .. } => "not_runnable",
        TaskWakeupOutcome::Missing => "missing",
    }
}

const fn watch_outcome_label(outcome: &ThreadEventsWatchOutcome) -> &'static str {
    match outcome {
        ThreadEventsWatchOutcome::Forwarded { .. } => "forwarded",
        ThreadEventsWatchOutcome::AlreadyCurrent { .. } => "already_current",
        ThreadEventsWatchOutcome::UnknownThread => "unknown_thread",
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_server::journal::task::TaskStatus;

    #[test]
    fn in_memory_recorder_aggregates_relay_ticks() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_relay_tick(
            &RelayTick {
                claimed: 4,
                delivered: 3,
                failed: 1,
                expired: 0,
            },
            42,
        );
        recorder.record_relay_tick(
            &RelayTick {
                claimed: 2,
                delivered: 2,
                failed: 0,
                expired: 0,
            },
            18,
        );

        let snap = recorder.snapshot();
        assert_eq!(snap.relay_ticks, 2);
        assert_eq!(snap.relay_delivered, 5);
        assert_eq!(snap.relay_failed, 1);
        assert_eq!(snap.relay_claimed, 6);
        assert_eq!(snap.relay_total_duration_ms, 60);
    }

    #[test]
    fn in_memory_recorder_tracks_backlog_breaches() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_relay_backlog(50, Some(100));
        recorder.record_relay_backlog(150, Some(100));
        recorder.record_relay_backlog(200, Some(100));

        let snap = recorder.snapshot();
        assert_eq!(snap.relay_backlog_observations, 3);
        assert_eq!(snap.relay_backlog_max, 200);
        assert_eq!(snap.relay_backlog_threshold_breaches, 2);
    }

    #[test]
    fn backlog_observation_with_no_threshold_does_not_count_breach() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_relay_backlog(10_000, None);

        let snap = recorder.snapshot();
        assert_eq!(snap.relay_backlog_observations, 1);
        assert_eq!(snap.relay_backlog_max, 10_000);
        assert_eq!(snap.relay_backlog_threshold_breaches, 0);
    }

    #[test]
    fn in_memory_recorder_breaks_wakeup_by_source_and_outcome() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_wakeup_nudge(
            WakeupSource::Broker,
            &TaskWakeupOutcome::Nudged {
                status: TaskStatus::Pending,
            },
        );
        recorder.record_wakeup_nudge(
            WakeupSource::FallbackSweep,
            &TaskWakeupOutcome::NotRunnable {
                status: TaskStatus::Running,
            },
        );
        recorder.record_wakeup_nudge(WakeupSource::Broker, &TaskWakeupOutcome::Missing);

        let snap = recorder.snapshot();
        assert_eq!(snap.wakeup_broker, 2);
        assert_eq!(snap.wakeup_fallback_sweep, 1);
        assert_eq!(snap.wakeup_nudged, 1);
        assert_eq!(snap.wakeup_not_runnable, 1);
        assert_eq!(snap.wakeup_missing, 1);
    }

    #[test]
    fn in_memory_recorder_aggregates_watch_advisories() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_watch_advisory(&ThreadEventsWatchOutcome::Forwarded {
            emitted_count: 3,
            emitted_up_to: 7,
        });
        recorder.record_watch_advisory(&ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 7 });
        recorder.record_watch_advisory(&ThreadEventsWatchOutcome::UnknownThread);

        let snap = recorder.snapshot();
        assert_eq!(snap.watch_forwarded, 1);
        assert_eq!(snap.watch_already_current, 1);
        assert_eq!(snap.watch_unknown_thread, 1);
        assert_eq!(snap.watch_events_emitted, 3);
    }

    #[test]
    fn in_memory_recorder_aggregates_janitor_cycles() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_janitor_cycle(&JanitorCycleReport {
            threads_scanned: 5,
            events_purged: 10,
            checkpoints_pruned: 2,
            floors_advanced: 3,
        });
        recorder.record_janitor_cycle(&JanitorCycleReport::default());

        let snap = recorder.snapshot();
        assert_eq!(snap.janitor_cycles, 2);
        assert_eq!(snap.janitor_threads_scanned, 5);
        assert_eq!(snap.janitor_events_purged, 10);
        assert_eq!(snap.janitor_checkpoints_pruned, 2);
        assert_eq!(snap.janitor_floors_advanced, 3);
    }

    #[test]
    fn in_memory_recorder_aggregates_lease_sweeps() {
        let recorder = InMemoryMetricsRecorder::new();
        recorder.record_lease_sweep(0);
        recorder.record_lease_sweep(4);

        let snap = recorder.snapshot();
        assert_eq!(snap.lease_sweep_cycles, 2);
        assert_eq!(snap.lease_sweep_released, 4);
    }

    #[test]
    fn backlog_threshold_band_helpers() {
        let threshold = BacklogThreshold {
            soft: 100,
            hard: 1_000,
        };
        assert!(!threshold.breaches_soft(100));
        assert!(threshold.breaches_soft(101));
        assert!(!threshold.breaches_hard(1_000));
        assert!(threshold.breaches_hard(1_001));
    }

    #[test]
    fn backlog_threshold_default_is_documented_band() {
        let threshold = BacklogThreshold::default();
        assert_eq!(threshold.soft, 1_000);
        assert_eq!(threshold.hard, 10_000);
    }

    #[test]
    fn cloned_in_memory_recorder_shares_counters() {
        let original = InMemoryMetricsRecorder::new();
        let cloned = original.clone();
        original.record_relay_reclaim(7);
        cloned.record_relay_reclaim(3);
        let snap = original.snapshot();
        assert_eq!(snap.relay_reclaimed, 10);
    }

    #[test]
    fn noop_recorder_does_nothing() {
        let recorder = NoopMetricsRecorder;
        recorder.record_relay_tick(
            &RelayTick {
                claimed: 1,
                delivered: 1,
                failed: 0,
                expired: 0,
            },
            10,
        );
        recorder.record_relay_reclaim(5);
        recorder.record_relay_backlog(100, Some(50));
        // The contract: recorder is reachable + cheap, no panic.
    }

    #[test]
    fn logging_recorder_invocations_do_not_panic() {
        let recorder = LoggingMetricsRecorder;
        recorder.record_relay_tick(
            &RelayTick {
                claimed: 1,
                delivered: 1,
                failed: 0,
                expired: 0,
            },
            10,
        );
        recorder.record_relay_reclaim(2);
        recorder.record_relay_backlog(50, Some(100));
        recorder.record_wakeup_nudge(
            WakeupSource::FallbackSweep,
            &TaskWakeupOutcome::Nudged {
                status: TaskStatus::Pending,
            },
        );
        recorder.record_watch_advisory(&ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 1 });
        recorder.record_janitor_cycle(&JanitorCycleReport::default());
        recorder.record_lease_sweep(0);
    }
}
