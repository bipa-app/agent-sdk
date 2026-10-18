//! The store's leader lock under
//! [`EpochRecovery::PostgresLeader`](crate::config::EpochRecovery::PostgresLeader).
//!
//! One host per Postgres store runs workers. Before it begins its epoch, a
//! host takes a session advisory lock on a dedicated connection and holds
//! it for its whole life, so a second host (a rolling deploy) waits without
//! workers until the first one is gone. The lock dies with its connection,
//! so a killed host frees it at once.
//!
//! Losing the lock does not stop a host's in-flight work at once: it runs
//! until the monitor notices ([`DETECTION_WINDOW`]) and its executions are
//! aborted. Two things keep that window safe. The store fences a superseded
//! epoch (its acquires, heartbeats and lease-guarded commits are refused),
//! and the successor waits out the window before it reclaims the old
//! leases (see `ServiceHost::begin_epoch`).
//!
//! The key is per schema: hosts on different schemas of one database do
//! not exclude each other. It is a session lock, so it does not work
//! behind a transaction-mode pooler (`PgBouncer`), and it costs one server
//! connection beyond the pool.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use sqlx::Connection;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::health::{HealthSurface, Leadership};
use crate::stores::StoreRegistry;

/// The advisory-lock class. With `hashtext(current_schema())` as the second
/// key it forms the two-key lock, which keeps this lock out of the
/// single-key space other advisory-lock users share.
const LOCK_CLASS: i32 = 0x4147_4C44;

/// How often a waiting host tries the lock again.
const RETRY: Duration = Duration::from_millis(500);

/// How often a leader checks its lock connection, independent of the
/// worker heartbeat.
const PING_INTERVAL: Duration = Duration::from_secs(1);

/// How long one ping may take before the connection counts as dead. A
/// half-open TCP connection would otherwise hang the ping for minutes while
/// the server has already freed the lock.
const PING_TIMEOUT: Duration = Duration::from_secs(2);

/// The longest a leader runs on after its lock is gone before it notices:
/// one ping interval plus one ping timeout.
pub(super) const DETECTION_WINDOW: Duration = PING_INTERVAL.saturating_add(PING_TIMEOUT);

/// The held lock. Dropping it stops the monitor and drops the connection,
/// which ends the session and frees the lock, so a host that fails or is
/// dropped after it took the lock never keeps it.
pub(super) struct LeaderLock {
    connection: Option<Arc<Mutex<sqlx::PgConnection>>>,
    monitor: Option<tokio::task::JoinHandle<()>>,
    stop_monitor: CancellationToken,
}

impl LeaderLock {
    /// Release the lock: stop the monitor and close the connection. Call
    /// it only after the workers have drained, so a successor never
    /// reclaims the lease of a worker that is still finishing.
    pub(super) async fn release(mut self) {
        self.stop_monitor.cancel();
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.await;
        }
        // The monitor has exited, so this is the last handle on the
        // connection: closing it ends the session and frees the lock now.
        if let Some(connection) = self.connection.take().and_then(Arc::into_inner) {
            let _ = connection.into_inner().close().await;
        }
        record_state("released");
        info!("released the store's leader lock");
    }
}

impl Drop for LeaderLock {
    fn drop(&mut self) {
        if self.connection.is_none() {
            return;
        }
        // The monitor holds the other handle on the connection. Once it
        // stops, the connection drops with the last handle and the server
        // frees the lock.
        self.stop_monitor.cancel();
        record_state("released");
        warn!("dropped the store's leader lock without a drain; it is freed with its connection");
    }
}

/// What the lock's monitor tells the rest of the host.
pub(super) struct LeaderSignals {
    pub(super) health: Arc<HealthSurface>,
    /// Cleared the moment the lock is lost; workers check it before each
    /// acquire.
    pub(super) acquiring: Arc<AtomicBool>,
    pub(super) shutdown: CancellationToken,
}

/// Wait until this host holds the store's leader lock. `Ok(None)` when the
/// store is not Postgres: there is no lock to take.
///
/// While it waits, a query error (a failover, a restart) reopens the lock
/// connection and tries again. After it returns, a monitor pings the lock
/// connection every [`PING_INTERVAL`], each ping bounded by [`PING_TIMEOUT`].
/// When a ping fails, the lock is gone and another host may already lead,
/// so the monitor stops acquisition, marks the host `lost` and cancels
/// `shutdown`.
///
/// # Errors
/// The pool could not open a connection for the lock.
pub(super) async fn acquire(
    stores: &StoreRegistry,
    signals: LeaderSignals,
) -> Result<Option<LeaderLock>> {
    let Some(pool) = stores.postgres_pool() else {
        return Ok(None);
    };
    let span = AcquireSpan::start();
    let started = Instant::now();
    let mut waiting = false;
    let connection = 'reconnect: loop {
        let mut connection = pool
            .acquire()
            .await
            .context("open the leader-lock connection")?
            .detach();
        loop {
            match try_lock(&mut connection).await {
                Ok(true) => break 'reconnect connection,
                Ok(false) => {}
                Err(err) => {
                    warn!(error = %err, "the leader-lock query failed; reconnecting and trying again");
                    tokio::time::sleep(RETRY).await;
                    continue 'reconnect;
                }
            }
            if !waiting {
                waiting = true;
                signals.health.set_leadership(Leadership::Standby);
                record_state("waiting");
                info!(
                    "another host leads this store; waiting for its leader lock before starting \
                     workers"
                );
            }
            tokio::time::sleep(RETRY).await;
        }
    };
    let waited = started.elapsed();
    signals.health.set_leadership(Leadership::Leader);
    signals.acquiring.store(true, Ordering::SeqCst);
    record_state("acquired");
    span.end(waited);
    info!(
        waited_ms = u64::try_from(waited.as_millis()).unwrap_or(u64::MAX),
        "acquired the store's leader lock"
    );

    let connection = Arc::new(Mutex::new(connection));
    let stop_monitor = CancellationToken::new();
    let monitor = tokio::spawn(monitor(
        Arc::clone(&connection),
        signals,
        stop_monitor.clone(),
    ));
    Ok(Some(LeaderLock {
        connection: Some(connection),
        monitor: Some(monitor),
        stop_monitor,
    }))
}

async fn try_lock(connection: &mut sqlx::PgConnection) -> Result<bool> {
    sqlx::query_scalar!(
        r#"SELECT pg_try_advisory_lock($1, hashtext(current_schema())) AS "acquired!""#,
        LOCK_CLASS,
    )
    .fetch_one(connection)
    .await
    .context("try the store's leader lock")
}

async fn monitor(
    connection: Arc<Mutex<sqlx::PgConnection>>,
    signals: LeaderSignals,
    stop: CancellationToken,
) {
    let mut ticker = tokio::time::interval(PING_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await;
    loop {
        tokio::select! {
            () = stop.cancelled() => return,
            _ = ticker.tick() => {
                let alive = {
                    let mut connection = connection.lock().await;
                    tokio::time::timeout(PING_TIMEOUT, connection.ping()).await
                };
                let failure = match alive {
                    Ok(Ok(())) => continue,
                    Ok(Err(err)) => err.to_string(),
                    Err(_) => format!("no answer within {PING_TIMEOUT:?}"),
                };
                signals.acquiring.store(false, Ordering::SeqCst);
                signals.health.set_leadership(Leadership::Lost);
                record_state("lost");
                error!(
                    error = %failure,
                    "lost the store's leader lock; stopped acquiring and shutting down so a \
                     single host runs workers"
                );
                signals.shutdown.cancel();
                return;
            }
        }
    }
}

#[cfg(feature = "otel")]
fn record_state(state: &'static str) {
    agent_server::observability::ServerMetrics::global().record_leader(state);
}

#[cfg(not(feature = "otel"))]
const fn record_state(_state: &'static str) {}

/// The `leader.acquire` span: how long the host waited for the lock.
struct AcquireSpan {
    #[cfg(feature = "otel")]
    span: opentelemetry::global::BoxedSpan,
}

impl AcquireSpan {
    #[cfg(feature = "otel")]
    fn start() -> Self {
        use opentelemetry::trace::Tracer;
        Self {
            span: opentelemetry::global::tracer("agent-service-host").start("leader.acquire"),
        }
    }

    #[cfg(not(feature = "otel"))]
    const fn start() -> Self {
        Self {}
    }

    #[cfg(feature = "otel")]
    fn end(mut self, waited: Duration) {
        use opentelemetry::KeyValue;
        use opentelemetry::trace::Span;
        self.span.set_attribute(KeyValue::new(
            "agent_server.leader.waited_ms",
            i64::try_from(waited.as_millis()).unwrap_or(i64::MAX),
        ));
        self.span.end();
    }

    #[cfg(not(feature = "otel"))]
    const fn end(self, _waited: Duration) {}
}
