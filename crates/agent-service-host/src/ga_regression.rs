//! Phase 8 GA regression suite — single home for the composed
//! invariants every prior Phase 8 slice ships.
//!
//! Every test here exercises **multiple Phase 8 contracts together**,
//! proving the system as a whole behaves as documented.  Per-contract
//! unit tests still live in their owning modules
//! (`relay::tests`, `host::tests`, `journal::thread_events_watch_regression`,
//! `journal::retention_janitor`); this suite codifies the GA story so
//! one file demonstrates all four GA acceptance criteria.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration as StdDuration;

use agent_sdk_foundation::ThreadId;
use agent_server::journal::broker::{BrokerAdapter, InMemoryBrokerAdapter};
use agent_server::journal::outbox::{InMemoryOutboxStore, NewOutboxRow, OutboxStatus, OutboxStore};
use agent_server::journal::outbox_message::{
    OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
};
use agent_server::journal::relay::{RelayTick, RelayWorker, RetryBackoff};
use anyhow::{Context, Result};
use async_trait::async_trait;
use time::{Duration as TimeDuration, OffsetDateTime};
use tokio_util::sync::CancellationToken;

use crate::health::{HealthSurface, LatencyLayerHealth};
use crate::metrics::{BacklogThreshold, InMemoryMetricsRecorder, MetricsRecorder};
use crate::relay::{RelayScheduler, RelaySchedulerConfig};

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + TimeDuration::seconds(1_700_000_000)
}

fn thread_id(name: &str) -> ThreadId {
    ThreadId::from_string(format!("t-ga-{name}"))
}

async fn seed_rows(store: &Arc<dyn OutboxStore>, thread: &ThreadId, count: usize) -> Result<()> {
    let mut rows = Vec::with_capacity(count);
    for seq in 0..count {
        let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread.clone(),
            last_sequence: seq as u64,
        })
        .to_payload_json()?;
        rows.push(NewOutboxRow {
            kind: OutboxMessageKind::ThreadEventsAvailable,
            thread_id: thread.clone(),
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

/// Seed `count` `TaskWakeup` outbox rows (finding #18, SQL arms).
///
/// Unlike [`seed_rows`], `TaskWakeup` rows carry `event_id = None` and
/// `sequence = None`, so they only need the thread FK (no committed-event
/// row) — which lets the SQL-backed arms seed the outbox without first
/// materialising a whole event journal. The thread row itself must be
/// created by the caller before seeding.
#[cfg(any(feature = "sqlite", feature = "postgres"))]
async fn seed_wakeup_rows(
    store: &Arc<dyn OutboxStore>,
    thread: &ThreadId,
    count: usize,
) -> Result<()> {
    use agent_server::journal::outbox_message::TaskWakeupPayload;
    use agent_server::journal::task::AgentTaskId;
    let mut rows = Vec::with_capacity(count);
    for seq in 0..count {
        let task_id = AgentTaskId::from_string(format!("task-ga-{seq}-{}", uuid::Uuid::now_v7()));
        let payload = OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id,
            thread_id: thread.clone(),
        })
        .to_payload_json()?;
        rows.push(NewOutboxRow {
            kind: OutboxMessageKind::TaskWakeup,
            thread_id: thread.clone(),
            event_id: None,
            sequence: None,
            payload_json: payload,
            max_attempts: 3,
            now: t0(),
        });
    }
    store.insert_batch(rows).await?;
    Ok(())
}

fn fast_relay_config(worker_id: &str) -> RelaySchedulerConfig {
    RelaySchedulerConfig {
        worker_id: worker_id.into(),
        batch_size: 16,
        poll_interval: StdDuration::from_millis(50),
        claim_lease: TimeDuration::seconds(30),
        reclaim_interval: StdDuration::from_millis(50),
        retry_backoff: RetryBackoff::fixed_seconds(0),
    }
}

/// A per-test Postgres schema for the GA outbox arms (finding #18).
/// Dropped on `Drop` so a real-database run leaves nothing behind.
/// Mirrors the gating used in `durability_suite.rs` / `postgres::store`.
#[cfg(feature = "postgres")]
struct PgGaSchema {
    database_url: String,
    schema: String,
}

#[cfg(feature = "postgres")]
impl PgGaSchema {
    /// Allocate a fresh isolated schema, or `None` when no test database
    /// is configured (so the arm skips cleanly / keyless CI stays green).
    async fn create() -> Result<Option<Self>> {
        use sqlx::Connection;
        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };
        let schema = format!("ga_pg_{}", uuid::Uuid::new_v4().simple());
        let mut admin = sqlx::postgres::PgConnection::connect(&database_url)
            .await
            .context("connect postgres admin for GA outbox suite")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create GA outbox test schema {schema}"))?;
        Ok(Some(Self {
            database_url,
            schema,
        }))
    }

    /// Open a fresh migrated store scoped to this schema.
    async fn open_store(&self) -> Result<crate::postgres::store::PostgresDurableStore> {
        let search_path = self.schema.clone();
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(4)
            .after_connect(move |conn, _meta| {
                let sql = format!("SET search_path TO {search_path}");
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&self.database_url)
            .await
            .context("connect schema-scoped postgres GA outbox pool")?;
        let store = crate::postgres::store::PostgresDurableStore::from_pool(pool);
        store
            .migrate()
            .await
            .context("migrate postgres GA outbox store")?;
        Ok(store)
    }
}

#[cfg(feature = "postgres")]
impl Drop for PgGaSchema {
    fn drop(&mut self) {
        use sqlx::Connection;
        let url = self.database_url.clone();
        let schema = self.schema.clone();
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            rt.block_on(async move {
                if let Ok(mut conn) = sqlx::postgres::PgConnection::connect(&url).await {
                    let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                        "DROP SCHEMA IF EXISTS {schema} CASCADE"
                    )))
                    .execute(&mut conn)
                    .await;
                }
            });
        })
        .join();
    }
}

// ─────────────────────────────────────────────────────────────────────
// AC1: Duplicate delivery survives a relay-crash window
// ─────────────────────────────────────────────────────────────────────

/// Worker double that successfully publishes to the broker but
/// "crashes" before stamping `mark_delivered`, leaving the row
/// `Claimed`.  Combined with `reclaim_expired_claims`, this exercises
/// the publish-then-crash window the relay must survive.
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
            // Intentionally do NOT mark delivered.
        }
        Ok(tick)
    }
}

/// Publish-then-crash battery, parameterised over the [`OutboxStore`]
/// implementation (finding #18). The store must already be seeded with
/// `count` pending rows. The first relay publishes every row but
/// "crashes" before stamping `delivered_at` (rows stay `Claimed`); the
/// second relay reclaims the expired claims and republishes (duplicate
/// delivery), driving every row to `Delivered`. Running this on the SQL
/// backends exercises their `mark_delivered` / `reclaim_expired_claims`
/// CAS paths that the in-memory store cannot cover.
async fn run_publish_then_crash_battery(
    store: Arc<dyn OutboxStore>,
    thread: &ThreadId,
    count: usize,
) -> Result<()> {
    // First relay publishes everything but "crashes" before marking.
    let broker = InMemoryBrokerAdapter::new();
    let crasher: Arc<dyn RelayWorker> = Arc::new(CrashBetweenPublishAndMark {
        broker: broker.clone(),
        store: Arc::clone(&store),
    });
    let cancel = CancellationToken::new();
    let first = RelayScheduler::with_worker(
        Arc::clone(&store),
        crasher,
        fast_relay_config("relay-crashed"),
    );
    let _ = first.run_backfill(&cancel, t0).await?;

    // Broker received every message exactly once but the rows stayed
    // Claimed because the crasher never stamped delivered_at.
    assert_eq!(broker.published_count().await, count);
    for row in store.list_by_thread(thread).await? {
        assert_eq!(row.status, OutboxStatus::Claimed);
        assert!(row.delivered_at.is_none());
    }

    // Second relay starts after the claim lease has expired and the
    // shared broker observes duplicate republishes, exactly as the
    // at-least-once contract documents.
    let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());
    let second = RelayScheduler::new(
        Arc::clone(&store),
        adapter,
        fast_relay_config("relay-recovered"),
    );
    second
        .reclaim_on_startup(t0() + TimeDuration::seconds(40))
        .await?;
    let outcome = second
        .run_backfill(&cancel, || t0() + TimeDuration::seconds(40))
        .await?;

    assert_eq!(outcome.delivered, count);
    assert_eq!(broker.published_count().await, count * 2);
    for row in store.list_by_thread(thread).await? {
        assert_eq!(row.status, OutboxStatus::Delivered);
        assert!(row.delivered_at.is_some());
    }
    Ok(())
}

#[tokio::test]
async fn phase_8_publish_then_crash_preserves_correctness() -> Result<()> {
    let thread = thread_id("crash-window");
    let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
    seed_rows(&store, &thread, 3).await?;
    run_publish_then_crash_battery(store, &thread, 3).await
}

/// `SQLite` arm of the publish-then-crash battery (finding #18).
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn phase_8_publish_then_crash_preserves_correctness_sqlite() -> Result<()> {
    use agent_server::journal::thread_store::ThreadStore;

    let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
    let thread = thread_id("sqlite-crash-window");
    ThreadStore::get_or_create(&store, &thread, t0()).await?;
    let store: Arc<dyn OutboxStore> = Arc::new(store);
    seed_wakeup_rows(&store, &thread, 3).await?;
    run_publish_then_crash_battery(store, &thread, 3).await
}

/// Postgres arm of the publish-then-crash battery (finding #18). A
/// keyless no-op when `TEST_DATABASE_URL` is unset.
#[cfg(feature = "postgres")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase_8_publish_then_crash_preserves_correctness_postgres() -> Result<()> {
    use agent_server::journal::thread_store::ThreadStore;

    let Some(schema) = PgGaSchema::create().await? else {
        return Ok(());
    };
    let store = schema.open_store().await?;
    let thread = thread_id("pg-crash-window");
    ThreadStore::get_or_create(&store, &thread, t0()).await?;
    let store: Arc<dyn OutboxStore> = Arc::new(store);
    seed_wakeup_rows(&store, &thread, 3).await?;
    run_publish_then_crash_battery(store, &thread, 3).await
}

// ─────────────────────────────────────────────────────────────────────
// AC1: Duplicate delivery is observable through the metrics recorder
// ─────────────────────────────────────────────────────────────────────

/// Duplicate-delivery-through-reclaim battery, parameterised over the
/// [`OutboxStore`] implementation (finding #18). Same crash/reclaim shape
/// as [`run_publish_then_crash_battery`], but the recovery relay carries
/// a metrics recorder so the reclaim + duplicate-republish counters are
/// asserted end to end. The store must already be seeded with `count`
/// pending rows.
async fn run_duplicate_delivery_battery(
    store: Arc<dyn OutboxStore>,
    thread: &ThreadId,
    count: usize,
) -> Result<()> {
    let count_u64 = u64::try_from(count).context("seeded row count fits in u64")?;

    // First relay claims and "crashes", so the rows stay Claimed.
    let broker = InMemoryBrokerAdapter::new();
    let crasher: Arc<dyn RelayWorker> = Arc::new(CrashBetweenPublishAndMark {
        broker: broker.clone(),
        store: Arc::clone(&store),
    });
    let first = RelayScheduler::with_worker(
        Arc::clone(&store),
        crasher,
        fast_relay_config("relay-crashed-duplicate"),
    );
    let cancel = CancellationToken::new();
    let _ = first.run_backfill(&cancel, t0).await?;

    // Second relay reclaims and republishes; its metrics recorder
    // observes the duplicate-safe republish path.
    let recorder = Arc::new(InMemoryMetricsRecorder::new());
    let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());
    let second = RelayScheduler::new(
        Arc::clone(&store),
        adapter,
        fast_relay_config("relay-duplicate"),
    )
    .with_metrics(Arc::clone(&recorder) as Arc<dyn MetricsRecorder>);
    let reclaimed = second
        .reclaim_on_startup(t0() + TimeDuration::seconds(40))
        .await?;
    let outcome = second
        .run_backfill(&cancel, || t0() + TimeDuration::seconds(40))
        .await?;

    let snap = recorder.snapshot();
    assert_eq!(reclaimed, count_u64);
    assert_eq!(
        snap.relay_reclaimed, count_u64,
        "reclaim recorded exactly once"
    );
    assert_eq!(outcome.delivered, count);
    assert_eq!(
        snap.relay_delivered, count_u64,
        "duplicate republish recorded"
    );
    assert!(snap.relay_ticks >= 1, "at least one tick recorded");
    assert_eq!(
        broker.published_count().await,
        count * 2,
        "duplicate republish",
    );
    for row in store.list_by_thread(thread).await? {
        assert_eq!(row.status, OutboxStatus::Delivered);
    }
    Ok(())
}

#[tokio::test]
async fn phase_8_duplicate_delivery_safe_through_reclaim() -> Result<()> {
    let thread = thread_id("duplicate-delivery");
    let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
    seed_rows(&store, &thread, 2).await?;
    run_duplicate_delivery_battery(store, &thread, 2).await
}

/// `SQLite` arm of the duplicate-delivery battery (finding #18).
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn phase_8_duplicate_delivery_safe_through_reclaim_sqlite() -> Result<()> {
    use agent_server::journal::thread_store::ThreadStore;

    let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
    let thread = thread_id("sqlite-duplicate-delivery");
    ThreadStore::get_or_create(&store, &thread, t0()).await?;
    let store: Arc<dyn OutboxStore> = Arc::new(store);
    seed_wakeup_rows(&store, &thread, 2).await?;
    run_duplicate_delivery_battery(store, &thread, 2).await
}

/// Postgres arm of the duplicate-delivery battery (finding #18). A
/// keyless no-op when `TEST_DATABASE_URL` is unset.
#[cfg(feature = "postgres")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase_8_duplicate_delivery_safe_through_reclaim_postgres() -> Result<()> {
    use agent_server::journal::thread_store::ThreadStore;

    let Some(schema) = PgGaSchema::create().await? else {
        return Ok(());
    };
    let store = schema.open_store().await?;
    let thread = thread_id("pg-duplicate-delivery");
    ThreadStore::get_or_create(&store, &thread, t0()).await?;
    let store: Arc<dyn OutboxStore> = Arc::new(store);
    seed_wakeup_rows(&store, &thread, 2).await?;
    run_duplicate_delivery_battery(store, &thread, 2).await
}

// ─────────────────────────────────────────────────────────────────────
// AC2: Backlog protection threshold flips latency layer to Degraded
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn phase_8_backlog_threshold_marks_latency_layer_degraded() -> Result<()> {
    let thread = thread_id("backlog-threshold");
    let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
    let health = HealthSurface::shared();
    health.set_latency_layer(LatencyLayerHealth::Healthy);

    // Seed three rows with a soft threshold of 1.  An always-failing
    // broker forces every row to bounce back to Pending so the
    // scheduler observes a backlog larger than the soft band — and
    // its threshold check (independent of the per-tick failed count)
    // must flip the latency layer to Degraded.
    seed_rows(&store, &thread, 3).await?;
    let broker: Arc<dyn BrokerAdapter> = Arc::new(AlwaysFailBroker);
    let recorder = Arc::new(InMemoryMetricsRecorder::new());
    // Non-zero backoff so the failed rows stay Pending past the
    // backfill drain instead of being immediately re-claimed and
    // driven to Expired before the backlog check runs.
    let cfg = RelaySchedulerConfig {
        retry_backoff: RetryBackoff::fixed_seconds(60),
        ..fast_relay_config("relay-backlog")
    };
    let scheduler = RelayScheduler::new(Arc::clone(&store), broker, cfg)
        .with_health(Arc::clone(&health))
        .with_metrics(Arc::clone(&recorder) as Arc<dyn MetricsRecorder>)
        .with_backlog_threshold(BacklogThreshold { soft: 1, hard: 5 });

    let outcome = scheduler
        .run_backfill(&CancellationToken::new(), t0)
        .await?;
    assert!(outcome.failed > 0, "publish failures must register");

    let snap = recorder.snapshot();
    assert!(
        snap.relay_backlog_observations >= 1,
        "backlog observed at least once",
    );
    assert!(
        snap.relay_backlog_threshold_breaches >= 1,
        "soft threshold breach recorded",
    );
    assert_eq!(
        health.snapshot().latency_layer,
        LatencyLayerHealth::Degraded,
        "soft-threshold breach must flip latency layer to Degraded",
    );
    Ok(())
}

/// Broker adapter that fails every publish — used to exercise the
/// backlog protection signal without needing a real broker outage.
#[derive(Debug)]
struct AlwaysFailBroker;

#[async_trait]
impl BrokerAdapter for AlwaysFailBroker {
    async fn publish(
        &self,
        _message: &agent_server::journal::outbox_message::OutboxMessage,
    ) -> Result<()> {
        anyhow::bail!("simulated broker unreachable")
    }
}

// ─────────────────────────────────────────────────────────────────────
// AC3: Metrics recorder observes the relay tick lifecycle end to end
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn phase_8_metrics_recorder_observes_full_lifecycle() -> Result<()> {
    let thread = thread_id("lifecycle");
    let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
    seed_rows(&store, &thread, 4).await?;

    let broker = InMemoryBrokerAdapter::new();
    let recorder = Arc::new(InMemoryMetricsRecorder::new());
    let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker.clone());
    let scheduler = RelayScheduler::new(
        Arc::clone(&store),
        adapter,
        fast_relay_config("relay-lifecycle"),
    )
    .with_metrics(Arc::clone(&recorder) as Arc<dyn MetricsRecorder>);

    // Drive the same lifecycle the host runs at startup: reclaim
    // (no-op here) then a backfill drain.
    scheduler.reclaim_on_startup(t0()).await?;
    let outcome = scheduler
        .run_backfill(&CancellationToken::new(), t0)
        .await?;
    assert_eq!(outcome.delivered, 4);

    let snap = recorder.snapshot();
    assert_eq!(
        snap.relay_delivered, 4,
        "every successful publish hits the recorder",
    );
    assert_eq!(snap.relay_failed, 0);
    assert!(snap.relay_ticks >= 1);
    assert!(
        snap.relay_total_duration_ms < 60_000,
        "duration is recorded but bounded for a fast in-memory drain",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// AC4: Latency layer recovers when the backlog drains below threshold
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn phase_8_backlog_recovers_after_drain() -> Result<()> {
    let thread = thread_id("backlog-recovery");
    let store: Arc<dyn OutboxStore> = Arc::new(InMemoryOutboxStore::new());
    let health = HealthSurface::shared();
    health.set_latency_layer(LatencyLayerHealth::Healthy);

    // Seed three rows; soft threshold = 1; the in-memory broker
    // always succeeds so the backfill drains everything.  After the
    // drain the backlog count returns to zero and the latency layer
    // must report Healthy.
    seed_rows(&store, &thread, 3).await?;
    let broker = InMemoryBrokerAdapter::new();
    let adapter: Arc<dyn BrokerAdapter> = Arc::new(broker);
    let recorder = Arc::new(InMemoryMetricsRecorder::new());
    let scheduler = RelayScheduler::new(
        Arc::clone(&store),
        adapter,
        fast_relay_config("relay-drain"),
    )
    .with_health(Arc::clone(&health))
    .with_metrics(Arc::clone(&recorder) as Arc<dyn MetricsRecorder>)
    .with_backlog_threshold(BacklogThreshold { soft: 1, hard: 5 });

    let outcome = scheduler
        .run_backfill(&CancellationToken::new(), t0)
        .await?;
    assert_eq!(outcome.delivered, 3);

    assert_eq!(
        health.snapshot().latency_layer,
        LatencyLayerHealth::Healthy,
        "fully drained backlog must leave the latency layer healthy",
    );

    let snap = recorder.snapshot();
    assert!(
        snap.relay_backlog_observations >= 1,
        "drained backlog still produces an observation",
    );
    Ok(())
}
