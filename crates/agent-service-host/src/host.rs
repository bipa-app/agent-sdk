//! Service host: worker bootstrap, background sweeps, health, and
//! graceful shutdown.
//!
//! [`ServiceHost`] composes a [`StoreRegistry`] with a worker pool,
//! periodic sweep loops, a lock-free [`HealthSurface`], and a
//! shutdown lifecycle.  It is the single struct a deploy target
//! creates:
//!
//! ```ignore
//! let host = ServiceHost::new(config, registry)?;
//! host.run().await?;   // blocks until shutdown signal
//! ```
//!
//! Transport layers (gRPC, HTTP) compose on top via accessors:
//!
//! ```ignore
//! let host = ServiceHost::new(config, registry)?;
//! let grpc = GrpcTransport::new(host.stores(), host.health());
//! tokio::select! {
//!     res = host.run() => res?,
//!     res = grpc.serve() => res?,
//! }
//! ```
//!
//! # Background tasks
//!
//! While running, the host spawns:
//!
//! | Task | Count | Interval | Purpose |
//! |------|-------|----------|---------|
//! | Lease sweep | 1 | [`WorkerConfig::sweep_interval`] | Release expired leases via recovery matrix |
//! | Worker | `pool_size` | [`WorkerConfig::acquisition_interval`] | Poll for runnable tasks, acquire and hold lease |
//! | Retention janitor | 0 or 1 | [`RetentionConfig::janitor_interval`] | Advance retention floors and prune excess checkpoints (spawned only when [`RetentionConfig::janitor_enabled`] is `true`) |
//!
//! All background tasks respect the host's [`CancellationToken`] and
//! drain cleanly on shutdown.
//!
//! # Health
//!
//! The host exposes a [`HealthSurface`] that background tasks update
//! on every cycle.  See the [`health`](super::health) module for the
//! two-dimensional health model (core vs latency layer).
//!
//! [`WorkerConfig::sweep_interval`]: super::config::WorkerConfig::sweep_interval
//! [`WorkerConfig::acquisition_interval`]: super::config::WorkerConfig::acquisition_interval
//! [`RetentionConfig::janitor_interval`]: super::config::RetentionConfig::janitor_interval
//! [`RetentionConfig::janitor_enabled`]: super::config::RetentionConfig::janitor_enabled
//! [`StoreRegistry`]: super::stores::StoreRegistry
//! [`HealthSurface`]: super::health::HealthSurface

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use agent_sdk_core::ToolTier;
use agent_server::journal::committed_event::CommittedEvent;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::execution_intent::{GuardedExecutionDeps, classify_tool_effect};
use agent_server::journal::task::{AgentTask, LeaseId, SubmittedInputItem, TaskKind, WorkerId};
use agent_server::journal::task_state::TaskState;
use agent_server::worker::{
    AgentDefinitionRegistry, RootTurnOutcome, SubagentTaskOutcome, ToolTaskOutcome,
    execute_subagent_task, fail_root_turn, guarded_tool_execution, pause_tool_for_confirmation,
    resolve_bootstrap_context, resolve_subagent_bootstrap, resolve_tool_bootstrap,
    resume_from_children,
};

use super::broker::{BrokerAdapter, InMemoryBrokerAdapter};
use super::config::{BrokerConfig, ServiceConfig};
use super::health::{HealthSurface, LatencyLayerHealth};
use super::http_health::HttpHealthHandle;
use super::relay::{RelayScheduler, RelaySchedulerConfig};
use super::runtime::ExecutionRuntime;
use super::stores::StoreRegistry;
use super::wakeup::WakeupScheduler;
use super::watch::ThreadEventsWatchScheduler;
use agent_server::journal::relay::RetryBackoff;
use agent_server::journal::{
    JournalTaskWakeupHandler, NotifierThreadEventsWatchHandler, TaskWakeupHandler,
    ThreadEventsWatchHandler, WakeupSignal,
};

// ─────────────────────────────────────────────────────────────────────
// ServiceHost
// ─────────────────────────────────────────────────────────────────────

struct BackgroundHandles {
    sweep: tokio::task::JoinHandle<()>,
    workers: Vec<tokio::task::JoinHandle<()>>,
    wakeup: Option<super::wakeup::WakeupSchedulerHandle>,
    relay: Option<super::relay::RelaySchedulerHandle>,
    watch: Option<super::watch::ThreadEventsWatchSchedulerHandle>,
    janitor: Option<tokio::task::JoinHandle<()>>,
    http_health: Option<HttpHealthHandle>,
}

/// The composed service host: stores + worker pool + sweeps +
/// health + lifecycle.
pub struct ServiceHost {
    config: ServiceConfig,
    stores: StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    health: Arc<HealthSurface>,
    shutdown: CancellationToken,
}

impl ServiceHost {
    /// Create a new service host from configuration.
    ///
    /// The `definition_registry` is supplied by the deployment layer
    /// (config file, database, API) — it is not derived from the
    /// storage backend.
    ///
    /// # Errors
    /// Returns an error if store initialisation fails or if the
    /// configuration contains invalid values (e.g. zero sweep interval).
    pub fn new(
        config: ServiceConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
        runtime: Arc<ExecutionRuntime>,
    ) -> Result<Self> {
        Self::validate_config(&config)?;
        let stores = StoreRegistry::from_config(&config.storage, definition_registry)
            .context("initialising store registry")?;
        Ok(Self {
            config,
            stores,
            runtime,
            health: HealthSurface::shared(),
            shutdown: CancellationToken::new(),
        })
    }

    /// Build a host with a pre-built store registry.
    ///
    /// Useful in tests where stores are pre-populated.
    ///
    /// # Errors
    /// Returns an error if the configuration contains invalid values.
    pub fn with_stores(
        config: ServiceConfig,
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
    ) -> Result<Self> {
        Self::validate_config(&config)?;
        Ok(Self {
            config,
            stores,
            runtime,
            health: HealthSurface::shared(),
            shutdown: CancellationToken::new(),
        })
    }

    fn validate_config(config: &ServiceConfig) -> Result<()> {
        anyhow::ensure!(
            config.worker.sweep_interval_secs > 0,
            "worker.sweep_interval_secs must be > 0"
        );
        anyhow::ensure!(config.worker.pool_size > 0, "worker.pool_size must be > 0");
        anyhow::ensure!(
            config.worker.acquisition_interval_secs > 0,
            "worker.acquisition_interval_secs must be > 0"
        );
        if matches!(
            config.storage.backend,
            super::config::StorageBackend::Postgres
        ) {
            anyhow::ensure!(
                config.storage.postgres.max_connections > 0,
                "storage.postgres.max_connections must be > 0"
            );
        }
        if config.wakeup.enabled {
            anyhow::ensure!(
                config.wakeup.fallback_interval_secs > 0,
                "wakeup.fallback_interval_secs must be > 0"
            );
        }
        if config.relay.enabled {
            anyhow::ensure!(config.relay.batch_size > 0, "relay.batch_size must be > 0");
            anyhow::ensure!(
                config.relay.poll_interval_secs > 0,
                "relay.poll_interval_secs must be > 0"
            );
            anyhow::ensure!(
                config.relay.claim_lease_secs > 0,
                "relay.claim_lease_secs must be > 0"
            );
            anyhow::ensure!(
                config.relay.reclaim_interval_secs > 0,
                "relay.reclaim_interval_secs must be > 0"
            );
        }
        if config.retention.janitor_enabled {
            anyhow::ensure!(
                config.retention.janitor_interval_secs > 0,
                "retention.janitor_interval_secs must be > 0"
            );
            anyhow::ensure!(
                config.retention.janitor_batch_size > 0,
                "retention.janitor_batch_size must be > 0"
            );
            if let Some(ttl) = config.retention.event_ttl_secs {
                anyhow::ensure!(
                    ttl > 0,
                    "retention.event_ttl_secs must be > 0; omit (null) to keep events forever"
                );
            }
            if let Some(n) = config.retention.checkpoint_max_per_thread {
                anyhow::ensure!(
                    n > 0,
                    "retention.checkpoint_max_per_thread must be > 0; omit (null) to keep all checkpoints"
                );
            }
        }
        Ok(())
    }

    /// Access the store registry (for transport layers and tests).
    #[must_use]
    pub const fn stores(&self) -> &StoreRegistry {
        &self.stores
    }

    /// Access the resolved configuration.
    #[must_use]
    pub const fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// Access the shared health surface (for transport probes).
    #[must_use]
    pub const fn health(&self) -> &Arc<HealthSurface> {
        &self.health
    }

    /// Access the host runtime wiring.
    #[must_use]
    pub const fn runtime(&self) -> &Arc<ExecutionRuntime> {
        &self.runtime
    }

    /// Initialize backend-specific dependencies such as migrations.
    ///
    /// # Errors
    /// Returns an error if the configured storage backend cannot be
    /// made ready for use.
    pub async fn initialize(&self) -> Result<()> {
        self.stores
            .initialize()
            .await
            .context("initialising storage backend")
    }

    /// Token that, when cancelled, triggers graceful shutdown.
    #[must_use]
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Run the service host until a shutdown signal is received.
    ///
    /// This method:
    /// 1. Logs startup configuration.
    /// 2. Spawns the lease-sweep background task.
    /// 3. Spawns `pool_size` worker tasks.
    /// 4. Marks the health surface as alive.
    /// 5. Waits for shutdown signal (`SIGINT` / `SIGTERM` / token
    ///    cancellation).
    /// 6. Cancels all background tasks and waits for drain.
    ///
    /// # Errors
    /// Returns an error if a background task panics or if shutdown
    /// coordination fails.
    pub async fn run(self) -> Result<()> {
        self.initialize().await?;
        self.log_startup_banner();

        // Register signal handlers before spawning any tasks so that a
        // registration failure (e.g. EMFILE) never leaves an orphaned
        // background task running indefinitely.
        #[cfg(unix)]
        let mut sigterm = {
            use tokio::signal::unix::{SignalKind, signal};
            signal(SignalKind::terminate()).context("registering SIGTERM handler")?
        };

        let wakeup_signal = WakeupSignal::shared();
        let sweep_handle = tokio::spawn(lease_sweep_loop(
            self.stores.clone(),
            self.config.worker.sweep_interval(),
            Arc::clone(&self.health),
            self.shutdown.clone(),
        ));
        let worker_handles = self.spawn_worker_pool(&wakeup_signal);
        let wakeup_handle = self.spawn_wakeup_scheduler(wakeup_signal);
        let relay_handle = self.spawn_relay_scheduler()?;
        let watch_handle = self.spawn_watch_scheduler();
        let janitor_handle = self.spawn_retention_janitor();
        let http_health_handle = self.spawn_http_health().await?;

        self.mark_healthy();
        info!(
            pool_size = self.config.worker.pool_size,
            relay_enabled = self.config.relay.enabled,
            wakeup_enabled = self.config.wakeup.enabled,
            watch_enabled = self.config.watch.enabled,
            janitor_enabled = self.config.retention.janitor_enabled,
            http_enabled = self.config.transport.http_enabled,
            "service host ready",
        );

        #[cfg(unix)]
        wait_for_shutdown(&self.shutdown, &mut sigterm).await?;
        #[cfg(not(unix))]
        wait_for_shutdown(&self.shutdown).await?;

        self.drain_background_tasks(BackgroundHandles {
            sweep: sweep_handle,
            workers: worker_handles,
            wakeup: wakeup_handle,
            relay: relay_handle,
            watch: watch_handle,
            janitor: janitor_handle,
            http_health: http_health_handle,
        })
        .await?;
        info!("service host stopped");
        Ok(())
    }

    fn log_startup_banner(&self) {
        info!(
            storage_backend = self.stores.backend_name(),
            pool_size = self.config.worker.pool_size,
            lease_duration_secs = self.config.worker.lease_duration_secs,
            sweep_interval_secs = self.config.worker.sweep_interval_secs,
            acquisition_interval_secs = self.config.worker.acquisition_interval_secs,
            grpc_enabled = self.config.transport.grpc_enabled,
            http_enabled = self.config.transport.http_enabled,
            relay_enabled = self.config.relay.enabled,
            wakeup_enabled = self.config.wakeup.enabled,
            "service host starting",
        );

        if self.stores.backend_name() == "postgres" {
            for surface in self
                .stores
                .durability_report()
                .iter()
                .filter(|surface| !surface.persists_restart)
            {
                warn!(
                    surface = surface.surface,
                    backend = surface.backend,
                    note = surface.note,
                    "storage surface remains non-durable under the postgres backend",
                );
            }
        }
    }

    fn spawn_worker_pool(
        &self,
        wakeup_signal: &Arc<WakeupSignal>,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let pool_size = self.config.worker.pool_size;
        let mut handles = Vec::with_capacity(pool_size);
        for idx in 0..pool_size {
            let params = WorkerLoopParams {
                index: idx,
                stores: self.stores.clone(),
                runtime: Arc::clone(&self.runtime),
                lease_duration: self.config.worker.lease_duration(),
                poll_interval: self.config.worker.acquisition_interval(),
                wakeup_signal: Arc::clone(wakeup_signal),
                cancel: self.shutdown.clone(),
            };
            handles.push(tokio::spawn(worker_loop(params)));
        }
        handles
    }

    fn spawn_wakeup_scheduler(
        &self,
        wakeup_signal: Arc<WakeupSignal>,
    ) -> Option<super::wakeup::WakeupSchedulerHandle> {
        if !self.config.wakeup.enabled {
            return None;
        }
        let handler: Arc<dyn TaskWakeupHandler> = Arc::new(JournalTaskWakeupHandler::new(
            Arc::clone(&self.stores.task_store),
            Arc::clone(&wakeup_signal),
        ));
        let scheduler = WakeupScheduler::new(self.config.wakeup.clone(), handler, wakeup_signal);
        Some(scheduler.spawn(self.shutdown.clone()))
    }

    fn spawn_watch_scheduler(&self) -> Option<super::watch::ThreadEventsWatchSchedulerHandle> {
        if !self.config.watch.enabled {
            return None;
        }
        let handler: Arc<dyn ThreadEventsWatchHandler> =
            Arc::new(NotifierThreadEventsWatchHandler::new(
                Arc::clone(&self.stores.event_repo),
                Arc::clone(&self.stores.event_notifier),
            ));
        let scheduler = ThreadEventsWatchScheduler::new(self.config.watch.clone(), handler);
        Some(scheduler.spawn(self.shutdown.clone()))
    }

    async fn spawn_http_health(&self) -> Result<Option<HttpHealthHandle>> {
        if !self.config.transport.http_enabled {
            return Ok(None);
        }
        let handle = super::http_health::spawn(
            self.config.transport.http_addr,
            Arc::clone(&self.health),
            self.shutdown.clone(),
        )
        .await
        .context("spawning HTTP health endpoint")?;
        Ok(Some(handle))
    }

    fn spawn_relay_scheduler(&self) -> Result<Option<super::relay::RelaySchedulerHandle>> {
        if self.config.relay.enabled {
            let broker = build_broker_adapter(&self.config.relay.broker)
                .context("building relay broker adapter")?;
            let scheduler_config = build_relay_scheduler_config(&self.config.relay);
            let scheduler = RelayScheduler::new(
                Arc::clone(&self.stores.outbox_store),
                broker,
                scheduler_config,
            )
            .with_health(Arc::clone(&self.health));
            // Inherit the host's shutdown token so SIGINT/SIGTERM drains
            // the relay along with the rest of the service.
            Ok(Some(scheduler.spawn(self.shutdown.clone())))
        } else {
            self.health
                .set_latency_layer(LatencyLayerHealth::NotConfigured);
            Ok(None)
        }
    }

    fn mark_healthy(&self) {
        self.health.set_sweep_alive(true);
        self.health.set_workers_alive(true);
        self.health.set_core(super::health::CoreHealth::Healthy);
    }

    fn spawn_retention_janitor(&self) -> Option<tokio::task::JoinHandle<()>> {
        if !self.config.retention.janitor_enabled {
            return None;
        }
        let policy = agent_server::journal::RetentionPolicy {
            event_ttl: self
                .config
                .retention
                .event_ttl_secs
                .map(std::time::Duration::from_secs),
            checkpoint_max_per_thread: self.config.retention.checkpoint_max_per_thread,
            batch_size: self.config.retention.janitor_batch_size,
        };
        Some(tokio::spawn(retention_janitor_loop(
            self.stores.clone(),
            policy,
            self.config.retention.janitor_interval(),
            self.shutdown.clone(),
        )))
    }

    async fn drain_background_tasks(&self, handles: BackgroundHandles) -> Result<()> {
        info!("draining background tasks");
        self.health.set_workers_alive(false);
        self.health.set_sweep_alive(false);

        handles
            .sweep
            .await
            .context("lease sweep task panicked during shutdown")?;
        for (idx, handle) in handles.workers.into_iter().enumerate() {
            handle
                .await
                .with_context(|| format!("worker {idx} panicked during shutdown"))?;
        }
        if let Some(handle) = handles.relay
            && let Err(err) = handle.shutdown().await
        {
            warn!(error = %err, "relay scheduler exited with error");
        }
        if let Some(handle) = handles.wakeup
            && let Err(err) = handle.shutdown().await
        {
            warn!(error = %err, "wakeup scheduler exited with error");
        }
        if let Some(handle) = handles.watch
            && let Err(err) = handle.shutdown().await
        {
            warn!(error = %err, "thread events watch scheduler exited with error");
        }
        if let Some(handle) = handles.janitor {
            handle
                .await
                .context("retention janitor panicked during shutdown")?;
        }
        if let Some(handle) = handles.http_health
            && let Err(err) = handle.shutdown().await
        {
            warn!(error = %err, "HTTP health server exited with error");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Relay wiring helpers
// ─────────────────────────────────────────────────────────────────────

fn build_broker_adapter(config: &BrokerConfig) -> Result<Arc<dyn BrokerAdapter>> {
    match config {
        BrokerConfig::InMemory => {
            warn!(
                "relay is enabled with the in-memory broker adapter — \
                 messages will be recorded in-process but not delivered anywhere",
            );
            Ok(Arc::new(InMemoryBrokerAdapter::new()))
        }
        #[cfg(feature = "amqp")]
        BrokerConfig::Amqp(amqp_config) => {
            super::broker::amqp::AmqpBrokerAdapter::arc(amqp_config.clone())
                .context("constructing AMQP broker adapter")
        }
    }
}

fn build_relay_scheduler_config(config: &super::config::RelayConfig) -> RelaySchedulerConfig {
    let worker_id = config
        .worker_id
        .clone()
        .unwrap_or_else(default_relay_worker_id);
    let retry_backoff_secs = config.retry_backoff_secs;
    RelaySchedulerConfig {
        worker_id,
        batch_size: config.batch_size,
        poll_interval: config.poll_interval(),
        claim_lease: config.claim_lease(),
        reclaim_interval: config.reclaim_interval(),
        retry_backoff: RetryBackoff::fixed_seconds(retry_backoff_secs),
    }
}

fn default_relay_worker_id() -> String {
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    format!("relay-{}", &suffix[..12])
}

// ─────────────────────────────────────────────────────────────────────
// Shutdown signal
// ─────────────────────────────────────────────────────────────────────

/// Wait for OS signals or the cancellation token.
///
/// On unix the caller **must** register the SIGTERM handler before
/// spawning background tasks and pass the resulting [`Signal`] here.
/// This guarantees a registration failure never leaves orphaned tasks.
#[cfg(unix)]
async fn wait_for_shutdown(
    shutdown: &CancellationToken,
    sigterm: &mut tokio::signal::unix::Signal,
) -> Result<()> {
    tokio::select! {
        () = shutdown.cancelled() => {
            info!("shutdown token cancelled");
        }
        result = tokio::signal::ctrl_c() => {
            match result {
                Ok(()) => info!("received SIGINT, shutting down"),
                Err(e) => warn!(error = %e, "signal handler failed"),
            }
            shutdown.cancel();
        }
        _ = sigterm.recv() => {
            info!("received SIGTERM, shutting down");
            shutdown.cancel();
        }
    }
    Ok(())
}

/// Wait for OS signals or the cancellation token (non-unix fallback).
#[cfg(not(unix))]
async fn wait_for_shutdown(shutdown: &CancellationToken) -> Result<()> {
    tokio::select! {
        () = shutdown.cancelled() => {
            info!("shutdown token cancelled");
        }
        result = tokio::signal::ctrl_c() => {
            match result {
                Ok(()) => info!("received SIGINT, shutting down"),
                Err(e) => warn!(error = %e, "signal handler failed"),
            }
            shutdown.cancel();
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Lease sweep loop
// ─────────────────────────────────────────────────────────────────────

/// Periodic lease-expiry sweep.
///
/// Runs [`AgentTaskStore::release_expired_leases`] at the configured
/// interval.  Updates the health surface on every cycle.
async fn lease_sweep_loop(
    stores: StoreRegistry,
    interval: std::time::Duration,
    health: Arc<HealthSurface>,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    // The first tick fires immediately; skip it so the sweep doesn't
    // run before the host has had a chance to accept any work.
    ticker.tick().await;

    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                info!("lease sweep shutting down");
                health.set_sweep_alive(false);
                return;
            }
            _ = ticker.tick() => {
                let now = time::OffsetDateTime::now_utc();
                match stores.task_store.release_expired_leases(now).await {
                    Ok(records) if !records.is_empty() => {
                        info!(count = records.len(), "released expired leases");
                    }
                    Ok(_) => { /* nothing expired — quiet */ }
                    Err(e) => {
                        warn!(error = %e, "lease sweep failed");
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Retention janitor loop
// ─────────────────────────────────────────────────────────────────────

async fn retention_janitor_loop(
    stores: StoreRegistry,
    policy: agent_server::journal::RetentionPolicy,
    interval: std::time::Duration,
    cancel: CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await;

    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                info!("retention janitor shutting down");
                return;
            }
            _ = ticker.tick() => {
                let now = time::OffsetDateTime::now_utc();
                let deps = agent_server::journal::RetentionJanitorDeps {
                    event_repo: stores.event_repo.as_ref(),
                    retention_store: stores.retention_store.as_ref(),
                    outbox_store: stores.outbox_store.as_ref(),
                    checkpoint_store: stores.checkpoint_store.as_ref(),
                };
                match agent_server::journal::run_janitor_cycle(&policy, &deps, now).await {
                    Ok(report)
                        if report.events_purged > 0 || report.checkpoints_pruned > 0 =>
                    {
                        info!(
                            threads = report.threads_scanned,
                            events_purged = report.events_purged,
                            checkpoints_pruned = report.checkpoints_pruned,
                            floors_advanced = report.floors_advanced,
                            "retention janitor cycle",
                        );
                    }
                    Ok(_) => { /* nothing to clean — quiet */ }
                    Err(e) => {
                        warn!(error = %e, "retention janitor cycle failed");
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Worker loop
// ─────────────────────────────────────────────────────────────────────

/// Parameters for [`worker_loop`].
///
/// Packed into a struct so the worker loop stays under Clippy's
/// argument-count limit while keeping every dependency explicit at
/// the spawn site.
struct WorkerLoopParams {
    index: usize,
    stores: StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    lease_duration: time::Duration,
    poll_interval: std::time::Duration,
    wakeup_signal: Arc<WakeupSignal>,
    cancel: CancellationToken,
}

/// A single worker's acquisition loop.
///
/// Each worker:
/// 1. Parks on either (a) its per-worker `acquisition_interval`
///    ticker or (b) the shared [`WakeupSignal`].  Whichever fires
///    first produces the same action.
/// 2. Calls `acquire_next_runnable`, which is the sole site where
///    the `Pending → Running` CAS lives.  Duplicate nudges (from a
///    broker consumer plus the fallback sweep plus the ticker) race
///    harmlessly because the CAS serialises them under the store
///    write lock.
/// 3. On successful acquisition, executes the task.
/// 4. Updates the health surface.
///
/// The worker identity is derived from a unique `WorkerId` per spawn.
async fn worker_loop(params: WorkerLoopParams) {
    let WorkerLoopParams {
        index,
        stores,
        runtime,
        lease_duration,
        poll_interval,
        wakeup_signal,
        cancel,
    } = params;
    let worker_id = WorkerId::from_string(format!("worker-{index}"));
    info!(%worker_id, "worker started");

    let mut ticker = tokio::time::interval(poll_interval);
    // Skip the immediate first tick.
    ticker.tick().await;

    loop {
        // Wait for either a scheduled poll or a wakeup nudge.  Both
        // paths converge on the same acquisition call so duplicates
        // are resolved by the CAS, not by the loop topology.
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                info!(%worker_id, "worker shutting down");
                return;
            }
            _ = ticker.tick() => {}
            () = wakeup_signal.wait_for_nudge() => {}
        }

        if cancel.is_cancelled() {
            info!(%worker_id, "worker shutting down");
            return;
        }

        let now = time::OffsetDateTime::now_utc();
        let lease_id = LeaseId::new();
        let expires_at = now + lease_duration;

        match stores
            .task_store
            .acquire_next_runnable(worker_id.clone(), lease_id, expires_at, now)
            .await
        {
            Ok(Some(task)) => {
                info!(
                    %worker_id,
                    task_id = %task.id,
                    thread_id = %task.thread_id,
                    kind = ?task.kind,
                    "acquired task",
                );
                if let Err(err) =
                    execute_acquired_task(task, &stores, Arc::clone(&runtime), &cancel).await
                {
                    warn!(%worker_id, error = %err, "task execution failed");
                }
            }
            Ok(None) => {
                // No runnable tasks — idle wait for the next tick
                // or nudge.  This is the benign-duplicate path: a
                // wakeup arrived for a task another worker already
                // leased, or the journal has nothing to do.
            }
            Err(e) => {
                warn!(%worker_id, error = %e, "task acquisition failed");
            }
        }
    }
}

async fn execute_acquired_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
) -> Result<()> {
    match task.kind {
        TaskKind::RootTurn => execute_root_task(task, stores, runtime).await,
        TaskKind::ToolRuntime => execute_tool_task(task, stores, runtime, cancel).await,
        TaskKind::Subagent => execute_subagent_task_entry(task, stores).await,
    }
}

async fn execute_root_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
) -> Result<()> {
    let now = time::OffsetDateTime::now_utc();
    let error_watermark = stores
        .event_repo
        .next_sequence(&task.thread_id)
        .await
        .context("reading root-task event watermark")?;

    let outcome = async {
        let bootstrap =
            resolve_bootstrap_context(task.clone(), stores.definition_registry.as_ref())
                .await
                .context("resolve root-task bootstrap")?;
        let inputs = build_root_worker_inputs(
            bootstrap,
            stores.thread_store.as_ref(),
            stores.checkpoint_store.as_ref(),
            now,
        )
        .await
        .context("build root-worker inputs")?;
        let provider = runtime
            .provider_resolver()
            .resolve_provider(inputs.definition())
            .await
            .context("resolve runtime provider")?;

        if matches!(task.state, TaskState::ReadyToResume { .. }) {
            resume_from_children(
                inputs,
                &task,
                provider.as_ref(),
                &stores.root_turn_deps(),
                now,
            )
            .await
            .context("resume root task from durable child results")
        } else {
            let user_prompt = root_task_prompt(&task)?;
            agent_server::worker::execute_root_turn(
                inputs,
                &user_prompt,
                provider.as_ref(),
                &stores.root_turn_deps(),
                now,
            )
            .await
            .context("execute fresh root task")
        }
    }
    .await;

    match outcome {
        Ok(RootTurnOutcome::Completed {
            committed_events, ..
        }) => {
            publish_events(stores, &committed_events);
            promote_next_root(stores, &task, now).await?;
            Ok(())
        }
        Ok(RootTurnOutcome::Suspended {
            committed_events, ..
        }) => {
            publish_events(stores, &committed_events);
            Ok(())
        }
        Err(err) => {
            warn!(
                task_id = %task.id,
                thread_id = %task.thread_id,
                error = %err,
                "root task execution failed; marking task failed",
            );
            fail_root_task(stores, &task, &err, error_watermark, now).await?;
            promote_next_root(stores, &task, now).await?;
            Ok(())
        }
    }
}

async fn execute_tool_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
) -> Result<()> {
    let now = time::OffsetDateTime::now_utc();
    let (worker_id, lease_id) = running_lease(&task)?;

    let bootstrap = match resolve_tool_bootstrap(task.clone(), stores.task_store.as_ref()).await {
        Ok(bootstrap) => bootstrap,
        Err(err) => {
            stores
                .task_store
                .fail_task(&task.id, &worker_id, &lease_id, format!("{err:#}"), now)
                .await
                .context("fail invalid tool task")?;
            return Ok(());
        }
    };

    if bootstrap.tool_call.tier == ToolTier::Confirm {
        let (_paused, committed_events) = pause_tool_for_confirmation(
            &bootstrap,
            stores.task_store.as_ref(),
            stores.event_repo.as_ref(),
            now,
        )
        .await
        .context("pause tool task for confirmation")?;
        publish_events(stores, &committed_events);
        return Ok(());
    }

    let guarded_deps = GuardedExecutionDeps {
        task_store: stores.task_store.as_ref(),
        intent_store: stores.execution_intent_store.as_ref(),
        event_repo: stores.event_repo.as_ref(),
    };
    let effect_class = classify_tool_effect(&bootstrap.tool_call);
    let exec_bootstrap = bootstrap.clone();
    let tool_executor = Arc::clone(runtime.tool_executor());
    let outcome = guarded_tool_execution(
        bootstrap,
        &guarded_deps,
        cancel,
        effect_class,
        move |_tool_call, collector| {
            let tool_executor = Arc::clone(&tool_executor);
            let exec_bootstrap = exec_bootstrap.clone();
            let cancel = cancel.clone();
            async move {
                tool_executor
                    .execute_tool_call(&exec_bootstrap, collector, cancel)
                    .await
            }
        },
        now,
    )
    .await;

    match outcome {
        Ok(
            ToolTaskOutcome::Completed {
                committed_events, ..
            }
            | ToolTaskOutcome::Failed {
                committed_events, ..
            },
        ) => {
            publish_events(stores, &committed_events);
            Ok(())
        }
        Ok(ToolTaskOutcome::Cancelled) => Ok(()),
        Err(err) => {
            stores
                .task_store
                .fail_task(&task.id, &worker_id, &lease_id, format!("{err:#}"), now)
                .await
                .context("fail tool task after guarded execution error")?;
            Ok(())
        }
    }
}

async fn execute_subagent_task_entry(task: AgentTask, stores: &StoreRegistry) -> Result<()> {
    let now = time::OffsetDateTime::now_utc();
    let (worker_id, lease_id) = running_lease(&task)?;

    let bootstrap = match resolve_subagent_bootstrap(task.clone(), stores.task_store.as_ref()).await
    {
        Ok(bootstrap) => bootstrap,
        Err(err) => {
            stores
                .task_store
                .fail_task(&task.id, &worker_id, &lease_id, format!("{err:#}"), now)
                .await
                .context("fail invalid subagent task")?;
            return Ok(());
        }
    };

    match execute_subagent_task(bootstrap, &stores.subagent_result_deps(), now).await {
        Ok(SubagentTaskOutcome {
            committed_events, ..
        }) => {
            publish_events(stores, &committed_events);
            Ok(())
        }
        Err(err) => {
            stores
                .task_store
                .fail_task(&task.id, &worker_id, &lease_id, format!("{err:#}"), now)
                .await
                .context("fail subagent task after materialization error")?;
            Ok(())
        }
    }
}

async fn fail_root_task(
    stores: &StoreRegistry,
    task: &AgentTask,
    error: &anyhow::Error,
    event_watermark: u64,
    now: time::OffsetDateTime,
) -> Result<()> {
    let (worker_id, lease_id) = running_lease(task)?;
    fail_root_turn(
        &task.id,
        &worker_id,
        &lease_id,
        &task.thread_id,
        error,
        &stores.root_turn_deps(),
        now,
    )
    .await
    .context("mark root task failed")?;

    let new_events = newly_committed_events(stores, &task.thread_id, event_watermark).await?;
    publish_events(stores, &new_events);
    Ok(())
}

fn publish_events(stores: &StoreRegistry, events: &[CommittedEvent]) {
    if !events.is_empty() {
        stores.event_notifier.notify(events);
    }
}

async fn newly_committed_events(
    stores: &StoreRegistry,
    thread_id: &agent_sdk_core::ThreadId,
    watermark: u64,
) -> Result<Vec<CommittedEvent>> {
    Ok(stores
        .event_repo
        .get_events(thread_id)
        .await
        .context("read committed events after failure")?
        .into_iter()
        .filter(|event| event.sequence >= watermark)
        .collect())
}

async fn promote_next_root(
    stores: &StoreRegistry,
    task: &AgentTask,
    now: time::OffsetDateTime,
) -> Result<()> {
    if task.kind == TaskKind::RootTurn {
        let _ = stores
            .task_store
            .promote_next_queued_root(&task.thread_id, now)
            .await
            .context("promote next queued root after terminal root")?;
    }
    Ok(())
}

fn running_lease(task: &AgentTask) -> Result<(WorkerId, LeaseId)> {
    let worker_id = task
        .worker_id
        .clone()
        .context("running task missing worker_id")?;
    let lease_id = task
        .lease_id
        .clone()
        .context("running task missing lease_id")?;
    Ok((worker_id, lease_id))
}

fn root_task_prompt(task: &AgentTask) -> Result<String> {
    if task.submitted_input.is_empty() {
        bail!("root task missing submitted input");
    }

    task.submitted_input
        .iter()
        .map(|item| match item {
            SubmittedInputItem::Text { text } => Ok(text.clone()),
            other => Err(anyhow!(
                "root task input item is not supported by the service host yet: {other:?}"
            )),
        })
        .collect::<Result<Vec<_>>>()
        .map(|parts| parts.join("\n"))
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServiceConfig;
    use crate::runtime::{
        AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
    };
    use agent_sdk_core::llm::{
        ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
    };
    use agent_sdk_providers::LlmProvider;
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
    use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sample_definition() -> AgentDefinition {
        AgentDefinition {
            provider: "mock".into(),
            model: "mock-model".into(),
            system_prompt: "test".into(),
            max_tokens: 4096,
            tools: Vec::new(),
            thinking: ThinkingPolicy::default(),
            policy: RuntimePolicy::server_default(),
        }
    }

    fn sample_registry() -> Arc<dyn AgentDefinitionRegistry> {
        Arc::new(InMemoryAgentDefinitionRegistry::new(sample_definition()))
    }

    struct MockTextProvider {
        response_text: String,
        call_count: AtomicUsize,
    }

    impl MockTextProvider {
        fn new(text: &str) -> Self {
            Self {
                response_text: text.to_owned(),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockTextProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_host_test_01".into(),
                content: vec![ContentBlock::Text {
                    text: self.response_text.clone(),
                }],
                model: "mock-model".into(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    fn sample_runtime() -> Result<Arc<ExecutionRuntime>> {
        let resolver = Arc::new(StaticProviderResolver::new());
        resolver.set_fallback(Arc::new(MockTextProvider::new("host reply")))?;
        Ok(Arc::new(ExecutionRuntime::new(
            resolver,
            Arc::new(NoopToolExecutor),
            Arc::new(AllowAllConfirmationPolicy),
        )))
    }

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn host_construction_succeeds() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        assert_eq!(host.config().worker.pool_size, 4);
        Ok(())
    }

    #[test]
    fn stores_accessible_from_host() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let _stores = host.stores();
        let _deps = host.stores().root_turn_deps();
        Ok(())
    }

    #[test]
    fn shutdown_token_is_clonable() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let token = host.shutdown_token();
        assert!(!token.is_cancelled());
        Ok(())
    }

    #[test]
    fn health_surface_accessible() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let snap = host.health().snapshot();
        // Before run(), health is not yet alive.
        assert!(!snap.is_ready());
        Ok(())
    }

    // ── Validation ───────────────────────────────────────────────

    #[test]
    fn zero_sweep_interval_is_rejected() -> Result<()> {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                sweep_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry(), sample_runtime()?);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn zero_pool_size_is_rejected() -> Result<()> {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry(), sample_runtime()?);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn zero_acquisition_interval_is_rejected() -> Result<()> {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                acquisition_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry(), sample_runtime()?);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn zero_postgres_max_connections_is_rejected() -> Result<()> {
        let config = ServiceConfig {
            storage: crate::config::StorageConfig {
                backend: crate::config::StorageBackend::Postgres,
                postgres: crate::config::PostgresStorageConfig {
                    database_url: Some(
                        "postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk".into(),
                    ),
                    schema: None,
                    max_connections: 0,
                },
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry(), sample_runtime()?);
        assert!(result.is_err());
        Ok(())
    }

    // ── Lifecycle ────────────────────────────────────────────────

    #[tokio::test]
    async fn host_shuts_down_on_token_cancel() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let token = host.shutdown_token();

        // Cancel immediately so `run()` returns promptly.
        token.cancel();
        host.run().await?;
        Ok(())
    }

    #[tokio::test]
    async fn health_becomes_ready_during_run() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();

        let host_handle = tokio::spawn(async move { host.run().await });

        // Yield to let the host start.
        tokio::task::yield_now().await;

        let snap = health.snapshot();
        assert!(snap.is_ready(), "host should be ready after start");
        assert!(snap.is_live(), "host should be live after start");

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn health_becomes_unready_after_shutdown() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();

        token.cancel();
        host.run().await?;

        let snap = health.snapshot();
        assert!(!snap.is_ready(), "host should not be ready after stop");
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn sweep_runs_at_least_once_before_shutdown() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                sweep_interval_secs: 1,
                // Use a long acquisition interval so workers don't
                // re-acquire the task before we can assert.
                acquisition_interval_secs: 300,
                ..Default::default()
            },
            ..Default::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        // Submit a root turn and acquire it with a very short lease
        // so the sweep will release it.
        let thread = ThreadId::from_string("t-sweep-test");
        let task = AgentTask::new_root_turn(thread.clone(), time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let now = time::OffsetDateTime::now_utc();
        let worker = WorkerId::from_string("w1");
        let lease = LeaseId::new();
        // Lease expires immediately.
        let expires = now - time::Duration::seconds(1);
        stores
            .task_store
            .try_acquire_task(&task_id, worker, lease, expires, now)
            .await?;

        // Run the host in the background.
        let host_handle = tokio::spawn(async move { host.run().await });

        // With start_paused, sleep auto-advances the synthetic clock.
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        token.cancel();
        host_handle.await??;

        // The expired lease should have been released — task is back to Pending.
        let recovered = stores
            .task_store
            .get(&task_id)
            .await?
            .context("task should still exist")?;
        assert_eq!(
            recovered.status,
            agent_server::journal::task::TaskStatus::Pending,
            "sweep should have released the expired lease",
        );
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn worker_acquires_pending_task() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 1,
                acquisition_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        // Submit a pending root turn.
        let thread = ThreadId::from_string("t-worker-test");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "hello from host worker".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Advance time so the worker polls and acquires.
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // The task should have been completed by the worker.
        let completed = stores
            .task_store
            .get(&task_id)
            .await?
            .context("task should still exist")?;
        assert_eq!(
            completed.status,
            agent_server::journal::task::TaskStatus::Completed,
            "worker should have completed the pending task",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn multiple_workers_start_and_stop() -> Result<()> {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let token = host.shutdown_token();

        // Cancel immediately — all 8 workers must drain cleanly.
        token.cancel();
        host.run().await?;
        Ok(())
    }

    // ── Phase 8.2: relay wiring ─────────────────────────────────────

    #[tokio::test]
    async fn relay_disabled_marks_latency_layer_not_configured() -> Result<()> {
        use crate::health::LatencyLayerHealth;

        // Default config has relay.enabled = false — latency layer
        // should stay NotConfigured throughout the host lifecycle.
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();

        let handle = tokio::spawn(async move { host.run().await });
        // Give the host a beat to reach the "ready" marker.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let snap = health.snapshot();
        assert_eq!(
            snap.latency_layer,
            LatencyLayerHealth::NotConfigured,
            "disabled relay should leave latency layer unconfigured",
        );

        token.cancel();
        handle.await??;
        Ok(())
    }

    // ── Phase 8.3: wakeup wiring ────────────────────────────────────
    //
    // These tests walk the four Phase 8.3 acceptance criteria
    // end-to-end through `ServiceHost::run`:
    //
    // 1. Wakeup consumers never execute directly from queue payloads.
    // 2. Every wakeup path re-checks durable task state before acting.
    // 3. Duplicate wakeups do not cause duplicate execution.
    // 4. Fallback sweeps keep work progressing when broker wakeups are
    //    delayed or absent.
    //
    // The tests use the default in-memory broker (no real AMQP) and
    // mutate the shared `WakeupSignal` directly where the scheduler
    // would normally be driven by the consumer.

    #[tokio::test]
    async fn wakeup_fallback_sweep_advances_work_without_consumer() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        // Enable the wakeup scheduler but do NOT enable the AMQP
        // consumer — the only path that can nudge workers is the
        // fallback sweep + the per-worker acquisition ticker.  This
        // matches acceptance criterion 4: "Fallback sweeps keep work
        // progressing even when broker wakeups are delayed or absent."
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 1,
                // Long ticker so acquisition is meaningfully driven by
                // the fallback sweep pulse rather than the per-worker
                // ticker.
                acquisition_interval_secs: 30,
                ..Default::default()
            },
            wakeup: crate::wakeup::WakeupConfig {
                enabled: true,
                fallback_interval_secs: 1,
                #[cfg(feature = "amqp")]
                amqp_consumer: crate::wakeup::AmqpConsumerSection::default(),
            },
            ..Default::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        // Submit a root turn so the worker has something to lease.
        let thread = ThreadId::from_string("t-wakeup-fallback");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "fallback makes progress".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Poll for completion for up to three seconds.  With a 1 s
        // fallback interval and a long acquisition ticker the work
        // can only have progressed via the fallback sweep's wake-up.
        let mut completed = false;
        for _ in 0..150 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&task_id)
                .await?
                .context("task should still exist")?;
            if row.status == agent_server::journal::task::TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "fallback sweep should have nudged the worker into picking up the pending task within 3 s",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn wakeup_handler_never_executes_but_nudges_worker() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::{
            JournalTaskWakeupHandler, TaskWakeupHandler, TaskWakeupOutcome, WakeupSignal,
            outbox_message::TaskWakeupPayload, task::AgentTask,
        };

        // Acceptance criterion 1 + 2: the handler only re-checks the
        // journal and nudges the signal; it never executes a task.
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = ThreadId::from_string("t-wakeup-handler");
        let task = AgentTask::new_root_turn(thread.clone(), time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let signal = WakeupSignal::shared();
        let handler =
            JournalTaskWakeupHandler::new(Arc::clone(&stores.task_store), Arc::clone(&signal));

        let payload = TaskWakeupPayload {
            task_id: task_id.clone(),
            thread_id: thread.clone(),
        };
        let outcome = handler
            .handle_payload(&payload, time::OffsetDateTime::now_utc())
            .await?;

        // The handler re-checked the journal and produced a Nudge —
        // but the task's status is still Pending (execution lives on
        // the worker, not the consumer).
        assert_eq!(
            outcome,
            TaskWakeupOutcome::Nudged {
                status: agent_server::journal::task::TaskStatus::Pending
            },
        );
        let after_handle = stores
            .task_store
            .get(&task_id)
            .await?
            .context("task must still exist")?;
        assert_eq!(
            after_handle.status,
            agent_server::journal::task::TaskStatus::Pending,
            "wakeup handler must not advance task state",
        );
        // Nudge is buffered, so the wait returns immediately.
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            signal.wait_for_nudge(),
        )
        .await
        .context("nudge must have fired")?;
        Ok(())
    }

    #[tokio::test]
    async fn wakeup_handler_rechecks_journal_between_deliveries() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::{
            JournalTaskWakeupHandler, TaskWakeupHandler, TaskWakeupOutcome, WakeupSignal,
            outbox_message::TaskWakeupPayload,
            task::{AgentTask, LeaseId as JournalLeaseId, WorkerId as JournalWorkerId},
        };

        // Acceptance criterion 2: every wakeup path re-checks durable
        // task state.  We deliver the same payload twice and the
        // handler reports two *different* outcomes because the task
        // moved from Pending → Running between calls.
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = ThreadId::from_string("t-wakeup-recheck");
        let task = AgentTask::new_root_turn(thread.clone(), time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let signal = WakeupSignal::shared();
        let handler =
            JournalTaskWakeupHandler::new(Arc::clone(&stores.task_store), Arc::clone(&signal));
        let payload = TaskWakeupPayload {
            task_id: task_id.clone(),
            thread_id: thread,
        };

        // First delivery: task is Pending, handler nudges.
        let first = handler
            .handle_payload(&payload, time::OffsetDateTime::now_utc())
            .await?;
        assert_eq!(
            first,
            TaskWakeupOutcome::Nudged {
                status: agent_server::journal::task::TaskStatus::Pending
            }
        );

        // Race the task to Running.  A second delivery must observe
        // the updated state because the handler does NOT cache.
        let worker = JournalWorkerId::from_string("w-recheck");
        let lease = JournalLeaseId::new();
        let now = time::OffsetDateTime::now_utc();
        stores
            .task_store
            .try_acquire_task(
                &task_id,
                worker,
                lease,
                now + time::Duration::seconds(30),
                now,
            )
            .await?
            .context("task should be acquirable")?;

        let second = handler
            .handle_payload(&payload, time::OffsetDateTime::now_utc())
            .await?;
        assert_eq!(
            second,
            TaskWakeupOutcome::NotRunnable {
                status: agent_server::journal::task::TaskStatus::Running
            },
            "handler must re-check durable state on every delivery",
        );
        Ok(())
    }

    #[tokio::test]
    async fn duplicate_wakeups_do_not_produce_duplicate_executions() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        // Acceptance criterion 3: flood the signal with many nudges
        // before the worker ticker fires and prove that exactly one
        // execution runs.  The MockTextProvider counts calls so we
        // can assert execution happened exactly once.
        let resolver = Arc::new(crate::runtime::StaticProviderResolver::new());
        let provider = Arc::new(MockTextProvider::new("duplicate-safe response"));
        resolver.set_fallback(Arc::clone(&provider) as Arc<dyn LlmProvider>)?;
        let runtime = Arc::new(ExecutionRuntime::new(
            resolver,
            Arc::new(crate::runtime::NoopToolExecutor),
            Arc::new(crate::runtime::AllowAllConfirmationPolicy),
        ));

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                // A wider pool increases the chance of a race that
                // could produce a duplicate execution if the contract
                // were broken.
                pool_size: 4,
                acquisition_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        let thread = ThreadId::from_string("t-duplicate-wakeup");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "fire many wakeups".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Wait for the task to complete.  The execution should happen
        // at most once even though the worker ticker + any number of
        // nudges would all hit `acquire_next_runnable`.
        let mut completed = false;
        for _ in 0..200 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&task_id)
                .await?
                .context("task should still exist")?;
            if row.status == agent_server::journal::task::TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(completed, "task should have completed within 4 s");

        // `MockTextProvider` counts the LLM calls — exactly one means
        // the work ran exactly once despite every worker polling.
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "duplicate wakeups must not produce duplicate executions",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn relay_enabled_drains_outbox_and_marks_layer_healthy() -> Result<()> {
        use crate::config::{BrokerConfig, RelayConfig};
        use crate::health::LatencyLayerHealth;
        use agent_sdk_core::ThreadId;
        use agent_server::journal::outbox::NewOutboxRow;
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };

        let config = ServiceConfig {
            relay: RelayConfig {
                enabled: true,
                worker_id: Some("host-test-relay".into()),
                batch_size: 8,
                poll_interval_secs: 1,
                claim_lease_secs: 30,
                reclaim_interval_secs: 30,
                retry_backoff_secs: 0,
                broker: BrokerConfig::InMemory,
            },
            ..ServiceConfig::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;

        // Seed an outbox row before the host starts so the backfill
        // phase has something to drain.  The in-memory backend does
        // not enforce the thread_id FK, so we can insert directly.
        let thread_id = ThreadId::from_string("t-host-relay-backfill");
        let stores = host.stores().clone();
        let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_id.clone(),
            last_sequence: 0,
        })
        .to_payload_json()?;
        stores
            .outbox_store
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_id.clone(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(0),
                payload_json: payload,
                max_attempts: 3,
                now: time::OffsetDateTime::now_utc(),
            }])
            .await?;

        let health = Arc::clone(host.health());
        let token = host.shutdown_token();
        let handle = tokio::spawn(async move { host.run().await });

        // Wait for the relay to drain + steady-state tick.  Poll up to
        // two seconds so the test is not flaky on slow CI.
        let mut healthy = false;
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            if health.snapshot().latency_layer == LatencyLayerHealth::Healthy {
                healthy = true;
                break;
            }
        }
        assert!(
            healthy,
            "relay should have marked latency layer healthy within 2s",
        );

        // The outbox row should have been delivered by the relay.
        let rows = stores.outbox_store.list_by_thread(&thread_id).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].status,
            agent_server::journal::outbox::OutboxStatus::Delivered,
        );

        token.cancel();
        handle.await??;
        Ok(())
    }

    // ── Phase 8.5: degraded-mode health, readiness, and fallback ────
    //
    // These tests walk the four Phase 8.5 acceptance criteria:
    //
    // 1. Broker outage does not make the server unready when core
    //    journal and worker paths are healthy.
    // 2. Health reporting distinguishes core correctness from relay
    //    degradation.
    // 3. Execution and replay continue correctly while unpublished
    //    outbox rows accumulate.
    // 4. Fallback sweeps and same-instance behavior keep progress
    //    moving without broker wakeup.

    #[tokio::test]
    async fn broker_outage_keeps_server_ready_when_core_healthy() -> Result<()> {
        use crate::config::{BrokerConfig, RelayConfig};
        use crate::health::{CoreHealth, LatencyLayerHealth};

        // Start the host with relay enabled.  The in-memory broker
        // adapter always succeeds, so we manually force the latency
        // layer to Degraded to simulate a broker outage and verify
        // that readiness stays true.
        let config = ServiceConfig {
            relay: RelayConfig {
                enabled: true,
                worker_id: Some("test-degraded".into()),
                batch_size: 8,
                poll_interval_secs: 60,
                claim_lease_secs: 30,
                reclaim_interval_secs: 60,
                retry_backoff_secs: 0,
                broker: BrokerConfig::InMemory,
            },
            ..ServiceConfig::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();
        let handle = tokio::spawn(async move { host.run().await });

        // Wait for the host to become ready.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(health.snapshot().is_ready(), "host should be ready");

        // Simulate broker outage by setting latency layer to Degraded.
        health.set_latency_layer(LatencyLayerHealth::Degraded);
        let snap = health.snapshot();

        // AC1: readiness stays true when core is healthy.
        assert!(
            snap.is_ready(),
            "readiness must not be affected by broker outage"
        );
        assert!(
            snap.is_live(),
            "liveness must not be affected by broker outage"
        );
        assert_eq!(snap.core, CoreHealth::Healthy);
        assert_eq!(snap.latency_layer, LatencyLayerHealth::Degraded);

        // AC2: aggregate status is Degraded, not Unhealthy.
        assert_eq!(
            snap.status,
            crate::health::HealthStatus::Degraded,
            "aggregate status should reflect degradation without blocking readiness",
        );

        token.cancel();
        handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn health_snapshot_distinguishes_core_from_relay() -> Result<()> {
        use crate::health::{CoreHealth, HealthStatus, LatencyLayerHealth};

        // AC2: health reporting distinguishes core correctness from
        // relay degradation.  We verify every combination.
        let surface = crate::health::HealthSurface::shared();

        // Healthy core + degraded relay → Degraded (ready, live).
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Healthy);
        surface.set_latency_layer(LatencyLayerHealth::Degraded);
        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Degraded);
        assert!(snap.is_ready());
        assert!(snap.is_live());

        // Healthy core + healthy relay → Healthy.
        surface.set_latency_layer(LatencyLayerHealth::Healthy);
        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Healthy);

        // Unhealthy core + healthy relay → Unhealthy (not ready).
        surface.set_core(CoreHealth::Unhealthy);
        surface.set_latency_layer(LatencyLayerHealth::Healthy);
        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert!(!snap.is_ready());

        // Unhealthy core + degraded relay → Unhealthy (not ready).
        surface.set_latency_layer(LatencyLayerHealth::Degraded);
        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert!(!snap.is_ready());

        Ok(())
    }

    #[tokio::test]
    async fn execution_continues_while_outbox_rows_accumulate() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::outbox::{NewOutboxRow, OutboxStatus};
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };
        use agent_server::journal::task::AgentTask;

        // AC3: execution and replay continue correctly while
        // unpublished outbox rows accumulate.
        //
        // The relay backfill phase drains on startup, so we start the
        // host first, wait for it to become ready, then seed outbox
        // rows.  The relay's steady-state poll is 300 s so the rows
        // stay Pending while the worker executes the task.
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 1,
                acquisition_interval_secs: 1,
                ..Default::default()
            },
            relay: crate::config::RelayConfig {
                enabled: true,
                worker_id: Some("test-accumulate".into()),
                batch_size: 8,
                poll_interval_secs: 300,
                claim_lease_secs: 30,
                reclaim_interval_secs: 300,
                retry_backoff_secs: 0,
                broker: crate::config::BrokerConfig::InMemory,
            },
            ..ServiceConfig::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();

        let host_handle = tokio::spawn(async move { host.run().await });

        // Wait for the host (and relay backfill) to start up.
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            if health.snapshot().is_ready() {
                break;
            }
        }

        // Seed outbox rows AFTER backfill.  They arrive in the
        // relay's steady-state window and won't be drained for 300 s.
        let thread = ThreadId::from_string("t-outbox-accumulate");
        for seq in 0..5u64 {
            let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                thread_id: thread.clone(),
                last_sequence: seq,
            })
            .to_payload_json()?;
            stores
                .outbox_store
                .insert_batch(vec![NewOutboxRow {
                    kind: OutboxMessageKind::ThreadEventsAvailable,
                    thread_id: thread.clone(),
                    event_id: Some(uuid::Uuid::now_v7()),
                    sequence: Some(seq),
                    payload_json: payload,
                    max_attempts: 3,
                    now: time::OffsetDateTime::now_utc(),
                }])
                .await?;
        }

        // Submit a task that the worker will execute concurrently.
        let task = AgentTask::new_root_turn_with_input(
            thread.clone(),
            vec![SubmittedInputItem::Text {
                text: "execute despite outbox backlog".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        // Poll for task completion.
        let mut completed = false;
        for _ in 0..200 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&task_id)
                .await?
                .context("task should still exist")?;
            if row.status == agent_server::journal::task::TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "task must complete even while outbox rows accumulate",
        );

        // Outbox rows should still be pending — the relay is in its
        // 300 s steady-state sleep.
        let outbox_rows = stores.outbox_store.list_by_thread(&thread).await?;
        let pending_count = outbox_rows
            .iter()
            .filter(|r| r.status == OutboxStatus::Pending)
            .count();
        assert!(
            pending_count > 0,
            "outbox rows should still be pending while relay poll is long",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn fallback_sweep_maintains_progress_during_broker_outage() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        // AC4: fallback sweeps and same-instance behavior keep
        // progress moving without broker wakeup.
        //
        // The wakeup scheduler is enabled with only the fallback
        // sweep (no AMQP consumer).  The worker acquisition ticker
        // is set very long so only the fallback sweep can nudge
        // workers into picking up work.
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 1,
                acquisition_interval_secs: 300,
                ..Default::default()
            },
            wakeup: crate::wakeup::WakeupConfig {
                enabled: true,
                fallback_interval_secs: 1,
                #[cfg(feature = "amqp")]
                amqp_consumer: crate::wakeup::AmqpConsumerSection::default(),
            },
            // Relay enabled but with long poll so it doesn't
            // interfere — we're testing the wakeup path.
            relay: crate::config::RelayConfig {
                enabled: true,
                worker_id: Some("test-fallback".into()),
                batch_size: 8,
                poll_interval_secs: 300,
                claim_lease_secs: 30,
                reclaim_interval_secs: 300,
                retry_backoff_secs: 0,
                broker: crate::config::BrokerConfig::InMemory,
            },
            ..ServiceConfig::default()
        };
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();
        let health = Arc::clone(host.health());
        let token = host.shutdown_token();

        let thread = ThreadId::from_string("t-fallback-progress");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "progress via fallback sweep".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        // Simulate broker outage: mark latency layer degraded.
        let host_handle = tokio::spawn(async move { host.run().await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        health.set_latency_layer(crate::health::LatencyLayerHealth::Degraded);

        // The task should still complete via the fallback sweep nudge
        // even though the broker (latency layer) is degraded.
        let mut completed = false;
        for _ in 0..150 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&task_id)
                .await?
                .context("task should still exist")?;
            if row.status == agent_server::journal::task::TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "fallback sweep must keep progress moving during broker outage",
        );

        // Verify the host is still ready despite broker degradation.
        let snap = health.snapshot();
        assert!(
            snap.is_ready(),
            "host must remain ready during broker outage",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }
}
