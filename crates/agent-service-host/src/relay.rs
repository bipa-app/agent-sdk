//! Outbox relay scheduler: drives [`RelayWorker::tick`] on a schedule
//! with a crash-safe startup backfill and periodic claim reclaim.
//!
//! The relay worker contract ships from [`agent_server::journal::relay`].
//! This module owns the **long-running driver** the host runtime wraps
//! that contract in: a single tokio task per relay that drains the
//! outbox backlog on startup, then polls the store at a configured
//! interval.  It is intentionally small — the tick is already a full
//! claim → publish → mark pass, so the scheduler only decides *when*
//! to call it and *how often* to reclaim crashed workers' claims.
//!
//! # Lifecycle
//!
//! ```text
//!   ┌──────────────────── startup reclaim ─────────────────────┐
//!   │ rows with `claimed_at <= now - claim_lease` → `Pending` │
//!   └──────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//!   ┌─────────────── backfill drain ──────────────┐
//!   │ tick() until claimed == 0 (no sleeping)     │
//!   │ — drains any rows that predated this worker │
//!   └─────────────────────────────────────────────┘
//!                            │
//!                            ▼
//!   ┌───────────── steady state ─────────────┐
//!   │ tick() on poll_interval                │
//!   │ reclaim stale claims on reclaim_tick   │
//!   └────────────────────────────────────────┘
//! ```
//!
//! Each phase respects the host's cancellation token so graceful
//! shutdown drains cleanly even mid-backfill.
//!
//! # Crash safety
//!
//! The outbox contract guarantees at-least-once via the
//! `claim_pending → publish → mark_delivered` sequence.  If the worker
//! crashes between the broker ack and the `mark_delivered` write, the
//! row stays in `Claimed` forever unless someone reclaims it.  That is
//! what [`OutboxStore::reclaim_expired_claims`] — added in Phase 8.2 —
//! does, and it runs here under two triggers:
//!
//! 1. **On startup**, with the configured `claim_lease`, so rows
//!    abandoned by a previously-crashed worker are reclaimed before
//!    the new worker enters the backfill loop without stealing fresh
//!    claims from still-live peers during a rolling restart.
//! 2. **Periodically during steady state**, with the configured
//!    `claim_lease`, so rows claimed by siblings that subsequently
//!    crashed get rescued automatically.
//!
//! The broker is at-least-once; a duplicate republish of a row the
//! previous worker had already successfully published is expected and
//! handled by consumer-side idempotency.
//!
//! [`OutboxStore::reclaim_expired_claims`]: agent_server::journal::outbox::OutboxStore::reclaim_expired_claims

use std::sync::Arc;
use std::time::Duration as StdDuration;

use agent_server::journal::broker::BrokerAdapter;
use agent_server::journal::outbox::OutboxStore;
use agent_server::journal::relay::{BrokerPublisher, OutboxRelayWorker, RelayWorker, RetryBackoff};
use anyhow::{Context, Result};
use time::{Duration as TimeDuration, OffsetDateTime};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::health::{HealthSurface, LatencyLayerHealth};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Tunables for a single relay scheduler.
#[derive(Clone, Debug)]
pub struct RelaySchedulerConfig {
    /// Stable worker identifier recorded on every claim.
    pub worker_id: String,
    /// Maximum rows claimed per tick.
    pub batch_size: u32,
    /// Interval between steady-state ticks when there is no backlog.
    pub poll_interval: StdDuration,
    /// Lease duration used by [`OutboxStore::reclaim_expired_claims`] in
    /// steady state.  Rows whose `claimed_at` is older than this are
    /// assumed to have been abandoned by a crashed worker.
    pub claim_lease: TimeDuration,
    /// Interval between periodic claim-reclaim sweeps.
    pub reclaim_interval: StdDuration,
    /// Backoff applied to failed publishes.  Uses fixed delay by
    /// default to match Phase 8.1's semantics; Phase 8.3 may upgrade
    /// to exponential.
    pub retry_backoff: RetryBackoff,
}

impl Default for RelaySchedulerConfig {
    fn default() -> Self {
        Self {
            worker_id: default_worker_id(),
            batch_size: 128,
            poll_interval: StdDuration::from_secs(2),
            claim_lease: TimeDuration::seconds(60),
            reclaim_interval: StdDuration::from_secs(30),
            retry_backoff: RetryBackoff::fixed_seconds(30),
        }
    }
}

fn default_worker_id() -> String {
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    format!("relay-{}", &suffix[..12])
}

// ─────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────

/// Long-running driver that polls an [`OutboxStore`] and publishes via
/// a [`BrokerAdapter`].
pub struct RelayScheduler {
    store: Arc<dyn OutboxStore>,
    worker: Arc<dyn RelayWorker>,
    config: RelaySchedulerConfig,
    health: Option<Arc<HealthSurface>>,
}

impl RelayScheduler {
    /// Construct a scheduler from an outbox store and a broker adapter.
    ///
    /// Internally wraps the adapter in the default [`BrokerPublisher`]
    /// and the default [`OutboxRelayWorker`] so callers never need to
    /// compose the four Phase 8.1 layers by hand.
    #[must_use]
    pub fn new(
        store: Arc<dyn OutboxStore>,
        broker: Arc<dyn BrokerAdapter>,
        config: RelaySchedulerConfig,
    ) -> Self {
        let publisher = Arc::new(BrokerPublisher::new(broker));
        let worker: Arc<dyn RelayWorker> = Arc::new(OutboxRelayWorker::new(
            Arc::clone(&store),
            publisher,
            config.batch_size,
            config.retry_backoff,
        ));
        Self {
            store,
            worker,
            config,
            health: None,
        }
    }

    /// Construct a scheduler around a pre-built [`RelayWorker`] — useful
    /// in tests that swap in a counting or failing worker.
    #[must_use]
    pub fn with_worker(
        store: Arc<dyn OutboxStore>,
        worker: Arc<dyn RelayWorker>,
        config: RelaySchedulerConfig,
    ) -> Self {
        Self {
            store,
            worker,
            config,
            health: None,
        }
    }

    /// Attach a shared health surface.  The scheduler updates the
    /// latency-layer axis on every tick.
    #[must_use]
    pub fn with_health(mut self, health: Arc<HealthSurface>) -> Self {
        self.health = Some(health);
        self
    }

    /// Reclaim every currently-`Claimed` row before the relay starts
    /// processing new work when the claim lease has already expired.
    /// Used on startup after a crash to return stuck rows to `Pending`
    /// without stealing fresh claims from still-live peers.
    ///
    /// # Errors
    /// Returns an error if the store reclaim query fails.
    pub async fn reclaim_on_startup(&self, now: OffsetDateTime) -> Result<u64> {
        let reclaimed = self
            .store
            .reclaim_expired_claims(now, self.config.claim_lease)
            .await
            .context("startup reclaim of outbox claims")?;
        if reclaimed > 0 {
            info!(
                reclaimed,
                claim_lease_secs = self.config.claim_lease.whole_seconds(),
                "reclaimed stale outbox claims before backfill",
            );
        }
        Ok(reclaimed)
    }

    /// Drain the outbox backlog without sleeping.
    ///
    /// Loops [`RelayWorker::tick`] until either (a) a tick claims zero
    /// rows or (b) the shutdown token is cancelled.  Returns the
    /// aggregate of the three delivery outcomes.
    ///
    /// # Errors
    /// Returns the first [`RelayWorker::tick`] error encountered — the
    /// caller decides whether to retry or escalate to degraded mode.
    pub async fn run_backfill(
        &self,
        cancel: &CancellationToken,
        now_fn: impl Fn() -> OffsetDateTime,
    ) -> Result<BackfillOutcome> {
        info!(worker_id = %self.config.worker_id, "relay backfill starting");
        let mut outcome = BackfillOutcome::default();
        loop {
            if cancel.is_cancelled() {
                info!("relay backfill cancelled mid-drain");
                outcome.cancelled = true;
                return Ok(outcome);
            }

            let tick = self
                .worker
                .tick(&self.config.worker_id, now_fn())
                .await
                .context("relay backfill tick")?;
            outcome.ticks += 1;
            outcome.delivered += tick.delivered;
            outcome.failed += tick.failed;
            outcome.expired += tick.expired;

            debug!(
                delivered = tick.delivered,
                failed = tick.failed,
                expired = tick.expired,
                claimed = tick.claimed,
                "relay backfill tick",
            );
            if tick.failed > 0 || tick.expired > 0 {
                self.mark_degraded();
            }

            if tick.claimed == 0 {
                info!(
                    ticks = outcome.ticks,
                    delivered = outcome.delivered,
                    failed = outcome.failed,
                    expired = outcome.expired,
                    "relay backfill drained",
                );
                if outcome.failed > 0 || outcome.expired > 0 {
                    self.mark_degraded();
                } else {
                    self.mark_healthy();
                }
                return Ok(outcome);
            }
        }
    }

    /// Run the steady-state polling loop until `cancel` fires.
    ///
    /// Alternates between a relay tick and a periodic claim-reclaim
    /// sweep.  A tick that claims zero rows triggers a sleep up to
    /// `poll_interval`; a non-empty tick immediately retries so a
    /// bursty thread does not wait a full interval per batch.
    pub async fn run_steady_state(
        &self,
        cancel: &CancellationToken,
        now_fn: impl Fn() -> OffsetDateTime,
    ) {
        info!(worker_id = %self.config.worker_id, "relay steady state starting");
        let mut reclaim_timer = tokio::time::interval(self.config.reclaim_interval);
        let mut latency_layer_degraded = self
            .health
            .as_ref()
            .is_some_and(|health| health.snapshot().latency_layer == LatencyLayerHealth::Degraded);
        // The first tick fires immediately — skip it so we don't
        // reclaim right after the startup reclaim already ran.
        reclaim_timer.tick().await;

        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    info!("relay steady state shutting down");
                    return;
                }
                _ = reclaim_timer.tick() => {
                    match self
                        .store
                        .reclaim_expired_claims(now_fn(), self.config.claim_lease)
                        .await
                    {
                        Ok(count) if count > 0 => {
                            warn!(
                                reclaimed = count,
                                "reclaimed stale outbox claims — other worker likely crashed",
                            );
                            latency_layer_degraded = false;
                        }
                        Ok(_) => {
                            latency_layer_degraded = false;
                        }
                        Err(err) => {
                            warn!(error = %err, "claim reclaim failed");
                            latency_layer_degraded = true;
                            self.mark_degraded();
                        }
                    }
                }
                result = self.worker.tick(&self.config.worker_id, now_fn()) => {
                    match result {
                        Ok(tick) if tick.claimed == 0 => {
                            // No work — sleep out the poll interval,
                            // unless we're shutting down.
                            if !latency_layer_degraded {
                                self.mark_healthy();
                            }
                            tokio::select! {
                                () = cancel.cancelled() => {
                                    info!("relay steady state shutting down");
                                    return;
                                }
                                () = tokio::time::sleep(self.config.poll_interval) => {}
                            }
                        }
                        Ok(tick) => {
                            debug!(
                                delivered = tick.delivered,
                                failed = tick.failed,
                                expired = tick.expired,
                                claimed = tick.claimed,
                                "relay steady-state tick",
                            );
                            if tick.failed > 0 || tick.expired > 0 {
                                latency_layer_degraded = true;
                                self.mark_degraded();
                            } else {
                                latency_layer_degraded = false;
                                self.mark_healthy();
                            }
                        }
                        Err(err) => {
                            warn!(error = %err, "relay tick failed");
                            latency_layer_degraded = true;
                            self.mark_degraded();
                            // Back off before retrying so we don't pin
                            // a CPU spinning through a broken broker.
                            tokio::select! {
                                () = cancel.cancelled() => {
                                    info!("relay steady state shutting down");
                                    return;
                                }
                                () = tokio::time::sleep(self.config.poll_interval) => {}
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run the scheduler end-to-end: startup reclaim → backfill →
    /// steady state.
    ///
    /// Returns when `cancel` fires or when the backfill encounters an
    /// unrecoverable error.
    ///
    /// # Errors
    /// Returns an error if startup reclaim or backfill fails.
    /// Steady-state errors are logged but not propagated because the
    /// relay is a best-effort latency layer.
    pub async fn run(&self, cancel: CancellationToken) -> Result<()> {
        let now = OffsetDateTime::now_utc();
        self.reclaim_on_startup(now).await?;
        let backfill = self.run_backfill(&cancel, OffsetDateTime::now_utc).await?;
        if backfill.cancelled {
            return Ok(());
        }
        self.run_steady_state(&cancel, OffsetDateTime::now_utc)
            .await;
        Ok(())
    }

    fn mark_healthy(&self) {
        if let Some(health) = &self.health {
            health.set_latency_layer(LatencyLayerHealth::Healthy);
        }
    }

    fn mark_degraded(&self) {
        if let Some(health) = &self.health {
            health.set_latency_layer(LatencyLayerHealth::Degraded);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Backfill outcome
// ─────────────────────────────────────────────────────────────────────

/// Aggregate counters returned by [`RelayScheduler::run_backfill`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackfillOutcome {
    /// Number of tick iterations executed.
    pub ticks: usize,
    /// Total rows successfully delivered during the backfill.
    pub delivered: usize,
    /// Total rows that failed but still have retry budget.
    pub failed: usize,
    /// Total rows that exhausted their retry budget during the backfill.
    pub expired: usize,
    /// Whether the backfill exited because the cancellation token fired.
    pub cancelled: bool,
}

// ─────────────────────────────────────────────────────────────────────
// Handle (spawned scheduler)
// ─────────────────────────────────────────────────────────────────────

/// Handle returned by [`RelayScheduler::spawn`].
///
/// Drops cancel the scheduler's loop; prefer an explicit
/// [`RelaySchedulerHandle::shutdown`] so the caller can await drain.
pub struct RelaySchedulerHandle {
    cancel: CancellationToken,
    join: JoinHandle<Result<()>>,
}

impl RelaySchedulerHandle {
    /// Cancellation token the scheduler observes.
    #[must_use]
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Cancel the scheduler and await its drain.
    ///
    /// # Errors
    /// Returns an error if the task panicked or the inner run loop
    /// returned an error.
    pub async fn shutdown(self) -> Result<()> {
        self.cancel.cancel();
        self.join
            .await
            .context("relay scheduler task panicked")?
            .context("relay scheduler run loop returned an error")
    }
}

impl RelayScheduler {
    /// Spawn the scheduler on the current tokio runtime and return a
    /// handle for graceful shutdown.
    ///
    /// The scheduler owns itself — the caller drops the handle when
    /// shutting down.
    #[must_use]
    pub fn spawn(self, cancel: CancellationToken) -> RelaySchedulerHandle {
        let token = cancel.clone();
        let join = tokio::spawn(async move { self.run(cancel).await });
        RelaySchedulerHandle {
            cancel: token,
            join,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::ThreadId;
    use agent_server::journal::broker::InMemoryBrokerAdapter;
    use agent_server::journal::outbox::{InMemoryOutboxStore, NewOutboxRow, OutboxStatus};
    use agent_server::journal::outbox_message::{
        OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
    };
    use agent_server::journal::relay::{Publisher, RelayTick};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + TimeDuration::seconds(1_700_000_000)
    }

    fn thread_id() -> ThreadId {
        ThreadId::from_string("t-relay-scheduler")
    }

    async fn seed_rows(store: &Arc<dyn OutboxStore>, count: usize) -> Result<()> {
        let mut rows = Vec::with_capacity(count);
        for seq in 0..count {
            let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                thread_id: thread_id(),
                last_sequence: seq as u64,
            })
            .to_payload_json()?;
            rows.push(NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_id(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(seq as u64),
                payload_json: payload,
                max_attempts: 3,
                now: t0(),
            });
        }
        store.insert_batch(rows).await?;
        Ok(())
    }

    fn test_config() -> RelaySchedulerConfig {
        RelaySchedulerConfig {
            worker_id: "test-relay".into(),
            // Small batch so multi-batch backfills take multiple ticks.
            batch_size: 2,
            poll_interval: StdDuration::from_millis(50),
            claim_lease: TimeDuration::seconds(30),
            reclaim_interval: StdDuration::from_millis(50),
            retry_backoff: RetryBackoff::fixed_seconds(0),
        }
    }

    // ── Startup backfill drains the queue in batches ────────────────

    #[tokio::test]
    async fn backfill_drains_multi_batch_backlog_in_order() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let broker = InMemoryBrokerAdapter::new();
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());

        seed_rows(&store, 7).await?;
        let scheduler = RelayScheduler::new(Arc::clone(&store), adapter, test_config());
        let cancel = CancellationToken::new();
        let outcome = scheduler.run_backfill(&cancel, t0).await?;

        // 7 rows with batch_size=2 → 4 ticks (2,2,2,1), one trailing empty tick.
        assert_eq!(outcome.delivered, 7);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.expired, 0);
        assert_eq!(
            broker.published_count().await,
            7,
            "all seeded rows must reach the broker before steady state",
        );
        assert!(
            outcome.ticks >= 4,
            "backfill should have taken at least 4 batched ticks, got {}",
            outcome.ticks,
        );
        Ok(())
    }

    #[tokio::test]
    async fn backfill_returns_cleanly_when_nothing_pending() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let broker = InMemoryBrokerAdapter::new();
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());
        let scheduler = RelayScheduler::new(store, adapter, test_config());
        let cancel = CancellationToken::new();
        let outcome = scheduler.run_backfill(&cancel, t0).await?;
        assert_eq!(outcome.delivered, 0);
        assert_eq!(outcome.ticks, 1, "one empty tick is enough to prove drain");
        Ok(())
    }

    // ── Post-ack marking: publish failure keeps rows unpublished ────

    /// Publisher that fails the first N publishes, then succeeds.
    /// Used to simulate "broker never acks" — the row must stay
    /// unmarked so a later run republishes it.
    struct AlwaysFailPublisher {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Publisher for AlwaysFailPublisher {
        async fn publish_row(&self, _row: &agent_server::journal::outbox::OutboxRow) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("simulated broker unreachable")
        }
    }

    #[tokio::test]
    async fn publish_failure_leaves_row_unpublished_for_retry() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        seed_rows(&store, 1).await?;

        let publisher: Arc<dyn Publisher> = Arc::new(AlwaysFailPublisher {
            calls: AtomicUsize::new(0),
        });
        // Build a worker with max_attempts high enough that the row
        // stays Pending across a single tick.
        let worker: Arc<dyn RelayWorker> = Arc::new(OutboxRelayWorker::new(
            Arc::clone(&store),
            publisher,
            16,
            RetryBackoff::fixed_seconds(0),
        ));

        let scheduler = RelayScheduler::with_worker(Arc::clone(&store), worker, test_config());
        let cancel = CancellationToken::new();
        let _ = scheduler.run_backfill(&cancel, t0).await?;

        // The row must have NOT transitioned to delivered — published_at
        // is only written after broker ack.
        let rows = store.list_by_thread(&thread_id()).await?;
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0].delivered_at.is_none(),
            "publish failure must not set delivered_at",
        );
        // It should be back at Pending (retry budget remaining) or
        // Expired (budget exhausted).  Neither is Delivered.
        assert_ne!(rows[0].status, OutboxStatus::Delivered);
        Ok(())
    }

    // ── Crash recovery: claimed-but-unmarked rows get republished ──

    /// Worker that simulates a successful broker publish but then
    /// "crashes" — it tracks the rows it would have marked delivered
    /// and returns them without touching the store.  Combined with
    /// `reclaim_expired_claims`, this models the
    /// publish-ack-before-mark crash window.
    struct CrashBetweenPublishAndMark {
        broker: InMemoryBrokerAdapter,
        store: Arc<dyn OutboxStore>,
    }

    #[async_trait]
    impl RelayWorker for CrashBetweenPublishAndMark {
        async fn tick(&self, worker_id: &str, now: OffsetDateTime) -> Result<RelayTick> {
            let claimed = self.store.claim_pending(worker_id, 16, now).await?;
            let mut tick = RelayTick {
                claimed: claimed.len(),
                ..RelayTick::default()
            };
            for row in claimed {
                let message = OutboxMessage::from_payload_json(row.kind, row.payload_json.clone())?;
                self.broker.publish(&message).await?;
                tick.delivered += 1;
                // Intentionally DO NOT call store.mark_delivered —
                // this is the crash window.
            }
            Ok(tick)
        }
    }

    #[tokio::test]
    async fn crashed_worker_claims_are_reclaimed_and_republished() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        seed_rows(&store, 3).await?;

        // First worker publishes but crashes before marking.
        let broker = InMemoryBrokerAdapter::new();
        let crasher: Arc<dyn RelayWorker> = Arc::new(CrashBetweenPublishAndMark {
            broker: broker.clone(),
            store: Arc::clone(&store),
        });
        let first_config = RelaySchedulerConfig {
            worker_id: "worker-crashed".into(),
            ..test_config()
        };
        let first = RelayScheduler::with_worker(Arc::clone(&store), crasher, first_config);
        let cancel = CancellationToken::new();
        let _ = first.run_backfill(&cancel, t0).await?;

        // All rows reached the broker once but stayed Claimed.
        assert_eq!(broker.published_count().await, 3);
        let rows = store.list_by_thread(&thread_id()).await?;
        for row in &rows {
            assert_eq!(row.status, OutboxStatus::Claimed);
            assert!(row.delivered_at.is_none());
        }

        // Second worker starts up — should reclaim and republish.
        let broker2 = InMemoryBrokerAdapter::new();
        // Share the *same* broker across attempts to observe duplicate
        // republish at the broker layer.
        let combined_broker = broker.clone();
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(combined_broker.clone());
        let second = RelayScheduler::new(
            Arc::clone(&store),
            adapter,
            RelaySchedulerConfig {
                worker_id: "worker-restarted".into(),
                ..test_config()
            },
        );
        second
            .reclaim_on_startup(t0() + TimeDuration::seconds(40))
            .await?;
        let outcome = second
            .run_backfill(&cancel, || t0() + TimeDuration::seconds(40))
            .await?;

        // Second worker must republish every reclaimed row.
        assert_eq!(outcome.delivered, 3);
        // Broker has now received each message twice — the duplicate
        // republish is the expected at-least-once behaviour.
        assert_eq!(
            combined_broker.published_count().await,
            6,
            "duplicate republish is expected after reclaim",
        );
        // All rows now terminal-delivered.
        for row in store.list_by_thread(&thread_id()).await? {
            assert_eq!(row.status, OutboxStatus::Delivered);
            assert!(row.delivered_at.is_some());
        }
        // Keep broker2 alive to silence unused warnings.
        drop(broker2);
        Ok(())
    }

    #[tokio::test]
    async fn startup_reclaim_skips_live_claims_from_other_workers() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        seed_rows(&store, 1).await?;
        store.claim_pending("worker-live", 8, t0()).await?;

        let scheduler = RelayScheduler::new(
            Arc::clone(&store),
            Arc::new(InMemoryBrokerAdapter::new()),
            RelaySchedulerConfig {
                claim_lease: TimeDuration::seconds(30),
                ..test_config()
            },
        );
        let reclaimed = scheduler
            .reclaim_on_startup(t0() + TimeDuration::seconds(10))
            .await?;
        assert_eq!(reclaimed, 0);

        let rows = store.list_by_thread(&thread_id()).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, OutboxStatus::Claimed);
        assert_eq!(rows[0].claimed_by.as_deref(), Some("worker-live"));
        Ok(())
    }

    #[derive(Default)]
    struct IdleWorker;

    #[async_trait]
    impl RelayWorker for IdleWorker {
        async fn tick(&self, _worker_id: &str, _now: OffsetDateTime) -> Result<RelayTick> {
            Ok(RelayTick::default())
        }
    }

    struct FlakyReclaimStore {
        inner: InMemoryOutboxStore,
        failures_remaining: AtomicUsize,
        reclaim_calls: AtomicUsize,
    }

    impl FlakyReclaimStore {
        fn new(failures: usize) -> Self {
            Self {
                inner: InMemoryOutboxStore::new(),
                failures_remaining: AtomicUsize::new(failures),
                reclaim_calls: AtomicUsize::new(0),
            }
        }

        fn reclaim_calls(&self) -> usize {
            self.reclaim_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl OutboxStore for FlakyReclaimStore {
        async fn insert_batch(
            &self,
            rows: Vec<agent_server::journal::outbox::NewOutboxRow>,
        ) -> Result<Vec<agent_server::journal::outbox::OutboxRow>> {
            self.inner.insert_batch(rows).await
        }

        async fn claim_pending(
            &self,
            worker_id: &str,
            limit: u32,
            now: OffsetDateTime,
        ) -> Result<Vec<agent_server::journal::outbox::OutboxRow>> {
            self.inner.claim_pending(worker_id, limit, now).await
        }

        async fn mark_delivered(
            &self,
            id: &agent_server::journal::outbox::OutboxRowId,
            now: OffsetDateTime,
        ) -> Result<()> {
            self.inner.mark_delivered(id, now).await
        }

        async fn mark_failed(
            &self,
            id: &agent_server::journal::outbox::OutboxRowId,
            error: &str,
            next_attempt_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<()> {
            self.inner
                .mark_failed(id, error, next_attempt_at, now)
                .await
        }

        async fn reclaim_expired_claims(
            &self,
            now: OffsetDateTime,
            claim_lease: TimeDuration,
        ) -> Result<u64> {
            self.reclaim_calls.fetch_add(1, Ordering::SeqCst);
            if self.failures_remaining.load(Ordering::SeqCst) > 0 {
                self.failures_remaining.fetch_sub(1, Ordering::SeqCst);
                anyhow::bail!("transient reclaim failure");
            }
            self.inner.reclaim_expired_claims(now, claim_lease).await
        }

        async fn get(
            &self,
            id: &agent_server::journal::outbox::OutboxRowId,
        ) -> Result<Option<agent_server::journal::outbox::OutboxRow>> {
            self.inner.get(id).await
        }

        async fn list_by_thread(
            &self,
            thread_id: &ThreadId,
        ) -> Result<Vec<agent_server::journal::outbox::OutboxRow>> {
            self.inner.list_by_thread(thread_id).await
        }

        async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
            self.inner.count_pending(thread_id).await
        }
    }

    // ── Delivered rows are marked only AFTER broker ack ─────────────

    #[tokio::test]
    async fn mark_delivered_runs_only_after_publish_succeeds() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        seed_rows(&store, 1).await?;

        let broker = InMemoryBrokerAdapter::new();
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());
        let scheduler = RelayScheduler::new(Arc::clone(&store), adapter, test_config());
        let cancel = CancellationToken::new();
        let _ = scheduler.run_backfill(&cancel, t0).await?;

        let rows = store.list_by_thread(&thread_id()).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, OutboxStatus::Delivered);
        // delivered_at is stamped only on success.
        assert!(rows[0].delivered_at.is_some());
        assert_eq!(broker.published_count().await, 1);
        Ok(())
    }

    // ── Spawned scheduler drains, then exits cleanly on cancel ──────

    #[tokio::test]
    async fn spawned_scheduler_exits_cleanly_on_cancel() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let broker = InMemoryBrokerAdapter::new();
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker);
        seed_rows(&store, 2).await?;
        let scheduler = RelayScheduler::new(
            Arc::clone(&store),
            adapter,
            RelaySchedulerConfig {
                poll_interval: StdDuration::from_millis(20),
                ..test_config()
            },
        );
        let cancel = CancellationToken::new();
        let handle = scheduler.spawn(cancel.clone());

        // Give the scheduler a beat to process the backlog.
        tokio::time::sleep(StdDuration::from_millis(100)).await;
        handle.shutdown().await?;

        let rows = store.list_by_thread(&thread_id()).await?;
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(row.status, OutboxStatus::Delivered);
        }
        Ok(())
    }

    #[tokio::test]
    async fn failed_publish_marks_latency_layer_degraded() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        seed_rows(&store, 1).await?;
        let health = HealthSurface::shared();

        let publisher: Arc<dyn Publisher> = Arc::new(AlwaysFailPublisher {
            calls: AtomicUsize::new(0),
        });
        let worker: Arc<dyn RelayWorker> = Arc::new(OutboxRelayWorker::new(
            Arc::clone(&store),
            publisher,
            16,
            RetryBackoff::fixed_seconds(30),
        ));
        let scheduler = RelayScheduler::with_worker(Arc::clone(&store), worker, test_config())
            .with_health(Arc::clone(&health));

        let outcome = scheduler
            .run_backfill(&CancellationToken::new(), t0)
            .await?;
        assert_eq!(outcome.failed, 1);
        assert_eq!(
            health.snapshot().latency_layer,
            LatencyLayerHealth::Degraded
        );
        Ok(())
    }

    #[tokio::test]
    async fn steady_state_recovers_after_transient_reclaim_failure_on_idle_queue() -> Result<()> {
        let concrete_store = Arc::new(FlakyReclaimStore::new(1));
        let store: Arc<dyn OutboxStore> = concrete_store.clone();
        let health = HealthSurface::shared();
        health.set_latency_layer(LatencyLayerHealth::Healthy);

        let scheduler = RelayScheduler::with_worker(
            store,
            Arc::new(IdleWorker),
            RelaySchedulerConfig {
                poll_interval: StdDuration::from_millis(10),
                reclaim_interval: StdDuration::from_millis(10),
                ..test_config()
            },
        )
        .with_health(Arc::clone(&health));
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(async move {
            scheduler
                .run_steady_state(&cancel_clone, OffsetDateTime::now_utc)
                .await;
        });

        tokio::time::sleep(StdDuration::from_millis(80)).await;
        cancel.cancel();
        handle.await?;

        assert!(
            concrete_store.reclaim_calls() >= 2,
            "expected one failed reclaim and at least one successful retry",
        );
        assert_eq!(health.snapshot().latency_layer, LatencyLayerHealth::Healthy);
        Ok(())
    }
}
