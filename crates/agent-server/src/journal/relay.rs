//! Outbox relay: publisher and worker abstractions.
//!
//! The relay path is split into the four trait layers shown in
//! [`super::broker`].  This module owns the middle two:
//!
//! - [`Publisher`] decodes a row's advisory payload and forwards it to
//!   a [`BrokerAdapter`].
//! - [`RelayWorker`] is the unit of work that scans pending rows,
//!   dispatches them through a publisher, and updates row status.
//!
//! It also defines [`TaskWakeupEmitter`], the analogue of
//! [`super::event_outbox_transaction::AtomicEventOutboxCommitter`] for
//! task-journal mutations:
//! same-transaction insertion of a `task_wakeup` row when a task
//! becomes runnable.  Durable backends implement it as the task
//! journal acquires the necessary commit hooks.
//!
//! # Why these abstractions are traits
//!
//! - The relay loop is *driven* by the host runtime, but the inner
//!   [`RelayWorker::tick`] is small enough to test on its own with the
//!   in-memory adapter, so the long-running scheduler is intentionally
//!   out of scope.
//! - Splitting [`Publisher`] from [`BrokerAdapter`] means a unit test
//!   can swap in a publisher that fails on demand without writing a
//!   full broker double.
//!
//! # Retry policy
//!
//! `OutboxRelayWorker::tick` retries failures with a caller-provided
//! [`RetryBackoff`] strategy.  The default
//! [`RetryBackoff::fixed_seconds`] keeps things simple; a later
//! iteration may switch to exponential backoff once production
//! data shapes are known.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::broker::BrokerAdapter;
use super::outbox::{NewOutboxRow, OutboxRow, OutboxRowId, OutboxStore};
use super::outbox_message::{OutboxMessage, OutboxMessageKind};

// ─────────────────────────────────────────────────────────────────────
// Retry backoff
// ─────────────────────────────────────────────────────────────────────

/// Backoff strategy used to compute the next attempt timestamp for a
/// failed outbox row.
///
/// Kept intentionally simple for now; a later iteration may replace
/// this with an exponential or jittered variant.
#[derive(Clone, Copy, Debug)]
pub enum RetryBackoff {
    /// Wait the same amount of wall-clock time after every failure.
    Fixed(Duration),
}

impl RetryBackoff {
    /// Convenience constructor for a fixed-second backoff.
    #[must_use]
    pub const fn fixed_seconds(seconds: u64) -> Self {
        Self::Fixed(Duration::from_secs(seconds))
    }

    /// Compute the next attempt timestamp from the current time and
    /// the row's attempt count.
    fn next_attempt_at(self, now: OffsetDateTime, _attempt_count: u32) -> OffsetDateTime {
        match self {
            Self::Fixed(delay) => now + delay,
        }
    }
}

impl Default for RetryBackoff {
    fn default() -> Self {
        // 30 seconds matches the existing in-memory tests' assumptions
        // about retry windows; production deployments override.
        Self::fixed_seconds(30)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Publisher
// ─────────────────────────────────────────────────────────────────────

/// Adapts an [`OutboxRow`] into the [`OutboxMessage`] envelope and
/// forwards to a broker.
///
/// Splitting this from [`BrokerAdapter`] lets the relay worker stay
/// agnostic to broker semantics — a publisher can apply per-kind
/// filtering, header injection, or per-tenant routing without changes
/// to the worker loop.
#[async_trait]
pub trait Publisher: Send + Sync {
    async fn publish_row(&self, row: &OutboxRow) -> Result<()>;
}

/// Default publisher that decodes the row's advisory payload and
/// forwards it to a [`BrokerAdapter`].
///
/// This is the only publisher 8.1 ships.  Subsequent phases may add
/// kind-aware routers (e.g. one publisher per AMQP exchange).
pub struct BrokerPublisher {
    broker: Arc<dyn BrokerAdapter>,
}

impl BrokerPublisher {
    #[must_use]
    pub fn new(broker: Arc<dyn BrokerAdapter>) -> Self {
        Self { broker }
    }
}

#[async_trait]
impl Publisher for BrokerPublisher {
    async fn publish_row(&self, row: &OutboxRow) -> Result<()> {
        let message = OutboxMessage::from_payload_json(row.kind, row.payload_json.clone())
            .with_context(|| format!("decode advisory payload for outbox row {}", row.id))?;
        self.broker
            .publish(&message)
            .await
            .with_context(|| format!("broker publish for outbox row {}", row.id))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// RelayWorker contract
// ─────────────────────────────────────────────────────────────────────

/// Per-tick observability counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelayTick {
    /// Rows claimed for relay this tick.
    pub claimed: usize,
    /// Rows successfully published this tick.
    pub delivered: usize,
    /// Rows that failed but still have retry budget.
    pub failed: usize,
    /// Rows that exhausted their retry budget this tick.
    pub expired: usize,
}

/// Unit of work the host runtime drives on a schedule.
///
/// One [`RelayWorker::tick`] call performs a single pass: claim a
/// batch of pending rows, publish each, and update row status.  The
/// long-running scheduler that calls `tick` repeatedly lives in the
/// host runtime — not here — so the worker contract stays small and
/// drivable from tests.
#[async_trait]
pub trait RelayWorker: Send + Sync {
    /// Run one iteration of the relay loop.
    ///
    /// `worker_id` is a stable identifier (hostname, pod ID, etc.) the
    /// store records on each claim.  `now` is the wall-clock time used
    /// for both the claim deadline and any retry-backoff math.  The
    /// returned [`RelayTick`] carries observability counters.
    async fn tick(&self, worker_id: &str, now: OffsetDateTime) -> Result<RelayTick>;
}

// ─────────────────────────────────────────────────────────────────────
// OutboxRelayWorker — default implementation
// ─────────────────────────────────────────────────────────────────────

/// Default relay worker that wires an [`OutboxStore`] to a
/// [`Publisher`].
pub struct OutboxRelayWorker {
    store: Arc<dyn OutboxStore>,
    publisher: Arc<dyn Publisher>,
    batch_limit: u32,
    backoff: RetryBackoff,
}

impl OutboxRelayWorker {
    /// Construct a new relay worker.
    ///
    /// `batch_limit` caps how many rows a single tick claims; tune it
    /// to balance throughput and broker connection pressure.
    #[must_use]
    pub fn new(
        store: Arc<dyn OutboxStore>,
        publisher: Arc<dyn Publisher>,
        batch_limit: u32,
        backoff: RetryBackoff,
    ) -> Self {
        Self {
            store,
            publisher,
            batch_limit,
            backoff,
        }
    }
}

#[async_trait]
impl RelayWorker for OutboxRelayWorker {
    async fn tick(&self, worker_id: &str, now: OffsetDateTime) -> Result<RelayTick> {
        let claimed_rows = self
            .store
            .claim_pending(worker_id, self.batch_limit, now)
            .await?;

        let mut tick = RelayTick {
            claimed: claimed_rows.len(),
            ..RelayTick::default()
        };

        // A store error on one row must NOT abort the whole tick: the
        // remaining claimed rows would otherwise sit `Claimed` until the
        // full claim-lease expires (`reclaim_expired_claims` is the only
        // rescue), and metrics for the tick would be lost.  Contain
        // per-row store failures, keep processing, surface metrics, and
        // return the first error only after every claimed row has been
        // attempted.
        let mut first_error: Option<anyhow::Error> = None;

        for row in claimed_rows {
            let row_result = match self.publisher.publish_row(&row).await {
                Ok(()) => match self.store.mark_delivered(&row.id, worker_id, now).await {
                    Ok(()) => {
                        tick.delivered += 1;
                        Ok(())
                    }
                    Err(err) => {
                        Err(err.context(format!("mark_delivered for outbox row {}", row.id)))
                    }
                },
                Err(publish_err) => {
                    let next_attempt = self.backoff.next_attempt_at(now, row.attempt_count + 1);
                    let error_message = format!("{publish_err:#}");
                    match self
                        .store
                        .mark_failed(&row.id, worker_id, &error_message, next_attempt, now)
                        .await
                    {
                        Ok(()) => {
                            if row.attempt_count + 1 >= row.max_attempts {
                                tick.expired += 1;
                            } else {
                                tick.failed += 1;
                            }
                            Ok(())
                        }
                        Err(err) => {
                            Err(err.context(format!("mark_failed for outbox row {}", row.id)))
                        }
                    }
                }
            };

            if let Err(err) = row_result {
                let detail = format!("{err:#}");
                log::warn!(
                    outbox_row = row.id.as_str(),
                    error = detail.as_str();
                    "relay tick: store update failed for a claimed row; \
                     continuing with the rest of the batch",
                );
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }

        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_relay_tick(&tick);

        first_error.map_or_else(|| Ok(tick), Err)
    }
}

// ─────────────────────────────────────────────────────────────────────
// TaskWakeupEmitter — analogue of AtomicEventOutboxCommitter for tasks
// ─────────────────────────────────────────────────────────────────────

/// Reference to the task-journal mutation that produced a wakeup.
///
/// Carried into [`TaskWakeupEmitter::emit_in_transaction`] so the
/// implementation can record what made the task runnable in the same
/// SQL transaction as the wakeup row.
#[derive(Clone, Debug)]
pub struct TaskWakeupTrigger {
    /// Task that became runnable.
    pub task_id: super::task::AgentTaskId,
    /// Thread the task belongs to.
    pub thread_id: agent_sdk_foundation::ThreadId,
    /// Wall-clock time of the triggering mutation.
    pub now: OffsetDateTime,
    /// Maximum relay attempts for this row.
    pub max_attempts: u32,
}

/// Default relay attempt budget for a durable `task_wakeup` advisory
/// row.
///
/// The wakeup is only a latency nudge — the worker pool's acquisition
/// ticker and [`super::task_wakeup::FallbackWakeupSweep`] still make
/// progress if every delivery is lost — so a small fixed budget is
/// plenty.  Matches the 3-attempt default used for the event outbox.
pub const TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS: u32 = 3;

/// Same-transaction emitter for `task_wakeup` outbox rows.
///
/// Implementations MUST insert the outbox row inside the same SQL
/// transaction as the task-journal mutation that made the task
/// runnable (admit, queue promotion, lease release after suspension).
/// If the surrounding transaction rolls back, the wakeup row MUST NOT
/// become visible.
///
/// Currently only the trait and the in-memory implementation ship.
/// Durable backends (`Postgres`, `SQLite`) will gain implementations
/// as the task-journal commit hooks are extended — at which point the
/// worker code that admits / promotes tasks calls this emitter inside
/// the same transaction.
#[async_trait]
pub trait TaskWakeupEmitter: Send + Sync {
    async fn emit_in_transaction(&self, trigger: TaskWakeupTrigger) -> Result<OutboxRowId>;
}

/// In-memory implementation suitable for tests and worker-side code
/// that already runs against [`super::outbox::InMemoryOutboxStore`].
pub struct InMemoryTaskWakeupEmitter {
    store: Arc<dyn OutboxStore>,
}

impl InMemoryTaskWakeupEmitter {
    #[must_use]
    pub fn new(store: Arc<dyn OutboxStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl TaskWakeupEmitter for InMemoryTaskWakeupEmitter {
    async fn emit_in_transaction(&self, trigger: TaskWakeupTrigger) -> Result<OutboxRowId> {
        ensure!(
            trigger.max_attempts >= 1,
            "task_wakeup max_attempts must be at least 1"
        );

        let payload = OutboxMessage::TaskWakeup(super::outbox_message::TaskWakeupPayload {
            task_id: trigger.task_id,
            thread_id: trigger.thread_id.clone(),
        })
        .to_payload_json()?;

        let mut rows = self
            .store
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::TaskWakeup,
                thread_id: trigger.thread_id,
                event_id: None,
                sequence: None,
                payload_json: payload,
                max_attempts: trigger.max_attempts,
                now: trigger.now,
            }])
            .await?;

        ensure!(
            rows.len() == 1,
            "task_wakeup emit must insert exactly one row"
        );
        Ok(rows.remove(0).id)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::broker::InMemoryBrokerAdapter;
    use crate::journal::outbox::InMemoryOutboxStore;
    use crate::journal::outbox_message::{TaskWakeupPayload, ThreadEventsAvailablePayload};
    use crate::journal::task::AgentTaskId;
    use agent_sdk_foundation::ThreadId;
    use std::sync::Mutex;
    use time::Duration as TimeDuration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + TimeDuration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + TimeDuration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-relay-a")
    }

    async fn seed_thread_events_row(
        store: &Arc<dyn OutboxStore>,
        thread_id: &ThreadId,
        last_sequence: u64,
        max_attempts: u32,
        now: OffsetDateTime,
    ) -> Result<OutboxRowId> {
        let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_id.clone(),
            last_sequence,
        })
        .to_payload_json()?;

        let mut rows = store
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_id.clone(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(last_sequence),
                payload_json: payload,
                max_attempts,
                now,
            }])
            .await?;
        Ok(rows.remove(0).id)
    }

    // ── Publisher ─────────────────────────────────────────────────

    #[tokio::test]
    async fn broker_publisher_decodes_and_forwards() -> Result<()> {
        let broker = InMemoryBrokerAdapter::new();
        let publisher = BrokerPublisher::new(Arc::new(broker.clone()));

        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let id = seed_thread_events_row(&store, &thread_a(), 9, 3, t0()).await?;
        let row = store.get(&id).await?.context("row missing")?;

        publisher.publish_row(&row).await?;

        let captured = broker.published().await;
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured[0],
            OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                thread_id: thread_a(),
                last_sequence: 9,
            }),
        );
        Ok(())
    }

    // ── OutboxRelayWorker happy path ──────────────────────────────

    #[tokio::test]
    async fn tick_marks_published_rows_delivered() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let broker = InMemoryBrokerAdapter::new();
        let publisher: Arc<dyn Publisher> =
            Arc::new(BrokerPublisher::new(Arc::new(broker.clone())));
        let worker = OutboxRelayWorker::new(
            store.clone(),
            publisher,
            16,
            RetryBackoff::fixed_seconds(30),
        );

        let id = seed_thread_events_row(&store, &thread_a(), 0, 3, t0()).await?;

        let tick = worker.tick("worker-a", t_plus(1)).await?;
        assert_eq!(tick.claimed, 1);
        assert_eq!(tick.delivered, 1);
        assert_eq!(tick.failed, 0);
        assert_eq!(tick.expired, 0);

        let row = store.get(&id).await?.context("row missing")?;
        assert!(row.delivered_at.is_some());
        assert_eq!(broker.published_count().await, 1);
        Ok(())
    }

    // ── Failure path: row stays Pending with retry budget ─────────

    /// Publisher that fails the first N publishes, then succeeds.
    struct FlakyPublisher {
        failures_remaining: Arc<Mutex<u32>>,
    }

    #[async_trait]
    impl Publisher for FlakyPublisher {
        async fn publish_row(&self, _row: &OutboxRow) -> Result<()> {
            let still_failing = {
                let mut remaining = self
                    .failures_remaining
                    .lock()
                    .ok()
                    .context("lock poisoned")?;
                if *remaining > 0 {
                    *remaining -= 1;
                    true
                } else {
                    false
                }
            };
            if still_failing {
                anyhow::bail!("simulated broker failure");
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn tick_records_failed_row_with_backoff() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let publisher: Arc<dyn Publisher> = Arc::new(FlakyPublisher {
            failures_remaining: Arc::new(Mutex::new(1)),
        });
        let worker = OutboxRelayWorker::new(
            store.clone(),
            publisher,
            16,
            RetryBackoff::fixed_seconds(30),
        );

        let id = seed_thread_events_row(&store, &thread_a(), 0, 3, t0()).await?;

        let tick = worker.tick("worker-a", t_plus(1)).await?;
        assert_eq!(tick.claimed, 1);
        assert_eq!(tick.delivered, 0);
        assert_eq!(tick.failed, 1);
        assert_eq!(tick.expired, 0);

        let row = store.get(&id).await?.context("row missing")?;
        assert_eq!(row.attempt_count, 1);
        assert_eq!(row.next_attempt_at, t_plus(1) + Duration::from_secs(30));
        Ok(())
    }

    // ── Exhaustion path: row hits max attempts ────────────────────

    #[tokio::test]
    async fn tick_expires_rows_when_budget_exhausted() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let publisher: Arc<dyn Publisher> = Arc::new(FlakyPublisher {
            failures_remaining: Arc::new(Mutex::new(99)),
        });
        let worker =
            OutboxRelayWorker::new(store.clone(), publisher, 16, RetryBackoff::fixed_seconds(0));

        let id = seed_thread_events_row(&store, &thread_a(), 0, 1, t0()).await?;

        let tick = worker.tick("worker-a", t_plus(1)).await?;
        assert_eq!(tick.claimed, 1);
        assert_eq!(tick.expired, 1);
        assert_eq!(tick.delivered, 0);
        assert_eq!(tick.failed, 0);

        let row = store.get(&id).await?.context("row missing")?;
        assert_eq!(row.status, super::super::outbox::OutboxStatus::Expired);
        assert!(row.last_error.is_some());
        Ok(())
    }

    // ── No work: empty tick is a clean no-op ──────────────────────

    #[tokio::test]
    async fn tick_with_no_pending_rows_returns_zero_counters() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let broker = InMemoryBrokerAdapter::new();
        let publisher: Arc<dyn Publisher> =
            Arc::new(BrokerPublisher::new(Arc::new(broker.clone())));
        let worker = OutboxRelayWorker::new(store, publisher, 16, RetryBackoff::fixed_seconds(30));

        let tick = worker.tick("worker-a", t0()).await?;
        assert_eq!(tick, RelayTick::default());
        assert_eq!(broker.published_count().await, 0);
        Ok(())
    }

    // ── TaskWakeupEmitter ─────────────────────────────────────────

    #[tokio::test]
    async fn task_wakeup_emitter_inserts_advisory_row() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let emitter = InMemoryTaskWakeupEmitter::new(store.clone());

        let task_id = AgentTaskId::new();
        let id = emitter
            .emit_in_transaction(TaskWakeupTrigger {
                task_id: task_id.clone(),
                thread_id: thread_a(),
                now: t0(),
                max_attempts: 3,
            })
            .await?;

        let row = store.get(&id).await?.context("row missing")?;
        assert_eq!(row.kind, OutboxMessageKind::TaskWakeup);
        assert_eq!(row.thread_id, thread_a());
        assert!(row.event_id.is_none());
        assert!(row.sequence.is_none());

        let payload = OutboxMessage::from_payload_json(row.kind, row.payload_json)?;
        assert_eq!(
            payload,
            OutboxMessage::TaskWakeup(TaskWakeupPayload {
                task_id,
                thread_id: thread_a(),
            }),
        );
        Ok(())
    }

    #[tokio::test]
    async fn task_wakeup_emitter_rejects_zero_max_attempts() -> Result<()> {
        let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
        let emitter = InMemoryTaskWakeupEmitter::new(store);

        let result = emitter
            .emit_in_transaction(TaskWakeupTrigger {
                task_id: AgentTaskId::new(),
                thread_id: thread_a(),
                now: t0(),
                max_attempts: 0,
            })
            .await;
        assert!(result.is_err());
        Ok(())
    }

    // ── Per-row store error containment ───────────────────────────

    /// Wraps an [`InMemoryOutboxStore`] and fails the first
    /// `mark_delivered` call, succeeding afterwards.  Everything else
    /// delegates to the inner store.
    struct FlakyMarkStore {
        inner: InMemoryOutboxStore,
        fail_remaining: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl OutboxStore for FlakyMarkStore {
        async fn insert_batch(&self, rows: Vec<NewOutboxRow>) -> Result<Vec<OutboxRow>> {
            self.inner.insert_batch(rows).await
        }

        async fn claim_pending(
            &self,
            worker_id: &str,
            limit: u32,
            now: OffsetDateTime,
        ) -> Result<Vec<OutboxRow>> {
            self.inner.claim_pending(worker_id, limit, now).await
        }

        async fn mark_delivered(
            &self,
            id: &OutboxRowId,
            worker_id: &str,
            now: OffsetDateTime,
        ) -> Result<()> {
            use std::sync::atomic::Ordering;
            if self.fail_remaining.load(Ordering::SeqCst) > 0 {
                self.fail_remaining.fetch_sub(1, Ordering::SeqCst);
                anyhow::bail!("simulated mark_delivered store failure");
            }
            self.inner.mark_delivered(id, worker_id, now).await
        }

        async fn mark_failed(
            &self,
            id: &OutboxRowId,
            worker_id: &str,
            error: &str,
            next_attempt_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<()> {
            self.inner
                .mark_failed(id, worker_id, error, next_attempt_at, now)
                .await
        }

        async fn reclaim_expired_claims(
            &self,
            now: OffsetDateTime,
            claim_lease: TimeDuration,
        ) -> Result<u64> {
            self.inner.reclaim_expired_claims(now, claim_lease).await
        }

        async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>> {
            self.inner.get(id).await
        }

        async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>> {
            self.inner.list_by_thread(thread_id).await
        }

        async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
            self.inner.count_pending(thread_id).await
        }

        async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>> {
            self.inner.min_unpublished_sequence(thread_id).await
        }
    }

    #[tokio::test]
    async fn tick_continues_after_per_row_store_error() -> Result<()> {
        use crate::journal::outbox::OutboxStatus;

        let store: Arc<dyn OutboxStore> = Arc::new(FlakyMarkStore {
            inner: InMemoryOutboxStore::new(),
            fail_remaining: std::sync::atomic::AtomicU32::new(1),
        });
        let broker = InMemoryBrokerAdapter::new();
        let publisher: Arc<dyn Publisher> =
            Arc::new(BrokerPublisher::new(Arc::new(broker.clone())));
        let worker = OutboxRelayWorker::new(
            store.clone(),
            publisher,
            16,
            RetryBackoff::fixed_seconds(30),
        );

        // Two claimable rows.  The publisher succeeds for both; the
        // store fails `mark_delivered` for whichever row is processed
        // first, then succeeds for the second.
        let id_a = seed_thread_events_row(&store, &thread_a(), 0, 3, t0()).await?;
        let id_b = seed_thread_events_row(&store, &thread_a(), 1, 3, t0()).await?;

        // The tick surfaces the store error but only AFTER attempting
        // every claimed row.
        let result = worker.tick("worker-a", t_plus(1)).await;
        assert!(result.is_err(), "tick must surface the store error");

        // Both rows were published despite the store error.
        assert_eq!(broker.published_count().await, 2);

        // Exactly one row reached Delivered; the other stayed Claimed
        // (its mark_delivered failed) — not stranded across the rest of
        // the batch.
        let row_a = store.get(&id_a).await?.context("row a missing")?;
        let row_b = store.get(&id_b).await?.context("row b missing")?;
        let statuses = [row_a.status, row_b.status];
        let delivered = statuses
            .iter()
            .filter(|s| **s == OutboxStatus::Delivered)
            .count();
        let claimed = statuses
            .iter()
            .filter(|s| **s == OutboxStatus::Claimed)
            .count();
        assert_eq!(delivered, 1, "one row should be delivered");
        assert_eq!(
            claimed, 1,
            "the failed row should remain claimed, not be skipped"
        );
        Ok(())
    }
}
