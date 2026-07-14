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

use agent_sdk_foundation::ToolTier;
use agent_server::journal::commit::StaleTurnCommit;
use agent_server::journal::committed_event::CommittedEvent;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::execution_intent::{GuardedExecutionDeps, classify_tool_effect};
use agent_server::journal::store::{AgentTaskStore, RequeueOutcome};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId,
};
use agent_server::journal::task_state::TaskState;
use agent_server::worker::{
    ActivityBeacon, AgentDefinitionRegistry, RootTurnOutcome, SubagentTaskOutcome, ToolTaskOutcome,
    best_effort_close_open_attempts, execute_subagent_task, fail_root_turn,
    fail_root_turn_leaving_attempts_open, guarded_tool_execution, pause_tool_for_confirmation,
    resolve_bootstrap_context, resolve_subagent_bootstrap, resolve_tool_bootstrap,
    resume_for_steering, resume_from_children, revert_steering_wake,
};

use super::broker::{BrokerAdapter, InMemoryBrokerAdapter};
use super::config::{BrokerConfig, ServiceConfig};
use super::health::{HealthSurface, LatencyLayerHealth};
use super::http_health::HttpHealthHandle;
use super::metrics::{LoggingMetricsRecorder, MetricsRecorder};
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
    metrics: Arc<dyn MetricsRecorder>,
    shutdown: CancellationToken,
    /// Hold the `db.pool.connections.{active,idle}`
    /// `ObservableGauge` handles for the host's lifetime so the
    /// callback registrations are not dropped early.  Empty unless
    /// the host was built with `--features otel` *and* the registry
    /// is Postgres-backed.
    #[cfg(feature = "otel")]
    _pool_gauges: Vec<opentelemetry::metrics::ObservableGauge<u64>>,
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
        #[cfg(feature = "otel")]
        let pool_gauges = super::observability::install_postgres_pool_gauges(&stores);
        Ok(Self {
            config,
            stores,
            runtime,
            health: HealthSurface::shared(),
            metrics: Arc::new(LoggingMetricsRecorder),
            shutdown: CancellationToken::new(),
            #[cfg(feature = "otel")]
            _pool_gauges: pool_gauges,
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
        #[cfg(feature = "otel")]
        let pool_gauges = super::observability::install_postgres_pool_gauges(&stores);
        Ok(Self {
            config,
            stores,
            runtime,
            health: HealthSurface::shared(),
            metrics: Arc::new(LoggingMetricsRecorder),
            shutdown: CancellationToken::new(),
            #[cfg(feature = "otel")]
            _pool_gauges: pool_gauges,
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
        // A zero heartbeat interval makes `tokio::time::interval` panic
        // inside the spawned heartbeat task, leaving every task with no
        // lease extension (leases expire → tasks double-execute). A zero
        // lease duration yields instantly-expired leases with the same
        // effect. Fail fast at startup instead.
        anyhow::ensure!(
            config.worker.heartbeat_interval_secs > 0,
            "worker.heartbeat_interval_secs must be > 0"
        );
        anyhow::ensure!(
            config.worker.lease_duration_secs > 0,
            "worker.lease_duration_secs must be > 0"
        );
        anyhow::ensure!(
            config.worker.lease_duration_secs > config.worker.heartbeat_interval_secs,
            "worker.lease_duration_secs must exceed worker.heartbeat_interval_secs so a lease survives between heartbeats"
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

    /// Access the shared metrics recorder (for tests and embedders).
    #[must_use]
    pub fn metrics(&self) -> &Arc<dyn MetricsRecorder> {
        &self.metrics
    }

    /// Replace the default [`LoggingMetricsRecorder`] with a caller
    /// provided implementation — typically an
    /// [`InMemoryMetricsRecorder`](crate::metrics::InMemoryMetricsRecorder)
    /// for tests or a Prometheus/OTel-backed recorder for production.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<dyn MetricsRecorder>) -> Self {
        self.metrics = metrics;
        self
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
        // Hand the same signal to the execution runtime *before* the
        // worker pool spawns so the worker task-execution paths can nudge
        // a parked worker the instant they journal new runnable work
        // (a tool-child batch, a parent whose last child just finished)
        // instead of waiting out the `acquisition_interval` ticker. The
        // ticker stays wired as the lost-wakeup backstop.
        self.runtime.set_wakeup_signal(Arc::clone(&wakeup_signal));
        let sweep_handle = tokio::spawn(lease_sweep_loop(
            self.stores.clone(),
            self.config.worker.sweep_interval(),
            self.config.worker.heartbeat_interval(),
            Arc::clone(&self.health),
            Arc::clone(&self.metrics),
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

        // Phase 10 · D: on a durable (SQL) backend the completed-turn
        // commit and task admission both write advisory outbox rows
        // (`thread_events_available`, `task_wakeup`) in the same
        // transaction as the journal mutation. Those rows are only
        // delivered to a broker — and thus drive cross-process broker
        // event delivery and durable cross-process wakeup — when the
        // relay is running. Warn loudly if it is disabled so a
        // multi-process deploy is not silently left with no durable
        // fan-out (the per-worker acquisition ticker still makes
        // progress, but at polling latency, and broker subscribers
        // receive nothing).
        let backend = self.stores.backend_name();
        if (backend == "postgres" || backend == "sqlite") && !self.config.relay.enabled {
            warn!(
                storage_backend = backend,
                "relay is DISABLED on a durable backend: advisory outbox rows \
                 (thread_events_available, task_wakeup) will accumulate without \
                 broker delivery, so multi-process deploys have no durable \
                 cross-process wakeup or broker event fan-out — progress falls \
                 back to the per-worker acquisition ticker. Set relay.enabled=true \
                 to deliver them.",
            );
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
                heartbeat_interval: self.config.worker.heartbeat_interval(),
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
            let mut scheduler = RelayScheduler::new(
                Arc::clone(&self.stores.outbox_store),
                broker,
                scheduler_config,
            )
            .with_health(Arc::clone(&self.health))
            .with_metrics(Arc::clone(&self.metrics));
            if let Some(threshold) = self.config.relay.backlog_threshold {
                scheduler = scheduler.with_backlog_threshold(threshold);
            }
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
            Arc::clone(&self.metrics),
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

// The `Result` return is needed by the fallible `amqp` arm; without the
// `amqp` feature only the infallible in-memory arm is compiled, so clippy
// sees an unconditionally-`Ok` wrapper. The signature stays uniform across
// feature combinations, so scope the `unnecessary_wraps` allow to exactly
// the build where the wrap is trivially infallible.
#[cfg_attr(not(feature = "amqp"), allow(clippy::unnecessary_wraps))]
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
    heartbeat_interval: std::time::Duration,
    health: Arc<HealthSurface>,
    metrics: Arc<dyn MetricsRecorder>,
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
                match drain_expired_leases(&stores, now).await {
                    Ok(count) if count > 0 => {
                        info!(count, "released expired leases");
                        metrics.record_lease_sweep(count);
                    }
                    Ok(_) => {
                        metrics.record_lease_sweep(0);
                    }
                    Err(e) => {
                        warn!(error = %e, "lease sweep failed");
                    }
                }

                // Subagent deadline enforcement for PARKED child roots
                // (issue #299): a child root suspended on its own tool
                // children has no live heartbeat, so a hung tool child
                // would otherwise wedge the parent past `timeout_ms`
                // indefinitely. Piggybacks on the same sweep cadence,
                // and MUST run after the expired-lease drain above:
                // the deadline pass skips Running rows on the strength
                // of that ordering (ghost leases were just requeued to
                // Pending, so Running there implies a live worker) —
                // which is also why the drain loops past the store's
                // per-call batch instead of reclaiming one batch.
                match enforce_subagent_deadlines(
                    &stores,
                    now,
                    min_stall_budget_ms(heartbeat_interval),
                    &cancel,
                )
                .await
                {
                    Ok(0) => {}
                    Ok(count) => {
                        info!(count, "failed timed-out parked subagent child roots");
                    }
                    Err(e) => {
                        warn!(error = %e, "subagent deadline sweep failed");
                    }
                }
            }
        }
    }
}

/// Rounds of [`AgentTaskStore::release_expired_leases`] one sweep tick
/// may run while draining a backlog — a guard against a pathological
/// store that keeps reporting full batches (the remainder defers to
/// the next tick). 40 rounds × the store batch ≈ 10k rows per tick.
const MAX_LEASE_SWEEP_ROUNDS: usize = 40;

/// Drain expired leases until the backlog is exhausted, returning the
/// total rows reclaimed.
///
/// `release_expired_leases` reclaims at most
/// [`agent_server::journal::store::LEASE_RELEASE_BATCH`] rows per call
/// on every backend. A single call under a larger backlog (mass worker
/// outage) would leave later rows ghost-Running through this tick —
/// and the subagent deadline pass that follows in the same tick skips
/// Running rows on the assumption that this drain already requeued
/// every ghost lease. Loop until a call returns fewer than the batch,
/// bounded by [`MAX_LEASE_SWEEP_ROUNDS`].
///
/// # Errors
/// Propagates the first store failure; rows reclaimed by earlier
/// rounds stay reclaimed (each round commits independently).
async fn drain_expired_leases(stores: &StoreRegistry, now: time::OffsetDateTime) -> Result<usize> {
    let mut total = 0usize;
    for _ in 0..MAX_LEASE_SWEEP_ROUNDS {
        let records = stores
            .task_store
            .release_expired_leases(now)
            .await
            .context("release expired leases")?;
        total = total.saturating_add(records.len());
        if records.len() < agent_server::journal::store::LEASE_RELEASE_BATCH {
            return Ok(total);
        }
    }
    warn!(
        rounds = MAX_LEASE_SWEEP_ROUNDS,
        reclaimed = total,
        "expired-lease drain hit its round guard; remaining backlog defers to the next tick",
    );
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────
// Retention janitor loop
// ─────────────────────────────────────────────────────────────────────

async fn retention_janitor_loop(
    stores: StoreRegistry,
    policy: agent_server::journal::RetentionPolicy,
    interval: std::time::Duration,
    metrics: Arc<dyn MetricsRecorder>,
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
                    Ok(report) => {
                        if report.events_purged > 0 || report.checkpoints_pruned > 0 {
                            info!(
                                threads = report.threads_scanned,
                                events_purged = report.events_purged,
                                checkpoints_pruned = report.checkpoints_pruned,
                                floors_advanced = report.floors_advanced,
                                "retention janitor cycle",
                            );
                        }
                        metrics.record_janitor_cycle(&report);
                    }
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
    heartbeat_interval: std::time::Duration,
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
        heartbeat_interval,
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
                run_task_with_heartbeat(
                    task,
                    &worker_id,
                    &stores,
                    Arc::clone(&runtime),
                    &cancel,
                    lease_duration,
                    heartbeat_interval,
                )
                .await;
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

/// Run a task to completion while extending its lease in the
/// background.
///
/// The lease the worker acquired is bounded by
/// [`super::config::WorkerConfig::lease_duration`] (default 30s). LLM
/// calls and tool executions routinely run longer than that, and
/// without a heartbeat the lease-expiry sweep would requeue a
/// still-live task to `Pending`. A second worker would then re-acquire
/// it while the first is mid-flight, producing a stampede that is the
/// usual precondition for the "resume root task from durable child
/// results" fail-loop in production.
///
/// We spawn a heartbeat ticker that calls
/// [`AgentTaskStore::heartbeat_task`] at `heartbeat_interval` and
/// extends `lease_expires_at` by `lease_duration`. The ticker uses a
/// child of the worker's cancellation token so it inherits shutdown
/// propagation from the parent while remaining independently
/// cancellable when the task execution future returns — without that
/// independent cancel we'd have no way to stop the ticker on natural
/// task completion (cancelling the parent would also abort the worker
/// loop).
///
/// Error policy: any failure from `heartbeat_task` exits the ticker —
/// no retry. Two distinct causes are conflated here:
/// * **Lease rejection** (worker / lease / status CAS mismatch). The
///   row no longer belongs to this worker; whoever owns it now will
///   carry it forward, and re-trying from this ticker would only
///   muddy the logs.
/// * **Transient store error** (e.g. a momentary DB hiccup). The task
///   keeps running but its lease is no longer being extended, so the
///   sweep may requeue it. We bail rather than retry because the
///   task's own next store call will surface the same fault, and the
///   default config (`lease_duration_secs = 30`,
///   `heartbeat_interval_secs = 10`) gives the lease ~3 heartbeats of
///   headroom — a single missed beat is benign.
async fn run_task_with_heartbeat(
    task: AgentTask,
    worker_id: &WorkerId,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
    lease_duration: time::Duration,
    heartbeat_interval: std::time::Duration,
) {
    let task_id = task.id.clone();
    let thread_id = task.thread_id.clone();
    let Some(lease_id) = task.lease_id.clone() else {
        warn!(
            %worker_id,
            task_id = %task_id,
            "acquired task missing lease id; skipping heartbeat",
        );
        // No lease means no heartbeat, and therefore nobody to persist a
        // beacon: this task's activity cannot be recorded, so it reports
        // none. Harmless — a row with no lease has no stall budget being
        // enforced against it either.
        if let Err(err) = Box::pin(execute_acquired_task(
            task,
            stores,
            runtime,
            cancel,
            &ActivityBeacon::new(),
        ))
        .await
        {
            warn!(%worker_id, error = %err, "task execution failed");
        }
        return;
    };

    // Store-free eligibility precheck (issue #299): a child-thread root may
    // carry a subagent deadline; everything else is exempt. Only this
    // store-free classification runs here; the linkage LOOKUP that resolves
    // an `Unresolved` deadline is deferred to the heartbeat loop's first
    // tick, deliberately after the heartbeat is up — a stalled lookup
    // before the first heartbeat could outlive the just-acquired lease,
    // letting the sweep requeue the row and a second worker acquire it
    // while this one later dispatched blind: duplicate LLM/tool execution.
    let initial_deadline = initial_deadline_state(&task);

    // Per-task cancellation token wired into the worker's `RootTurnDeps`
    // (seam B). The heartbeat loop trips it on a terminal lease rejection
    // — a `cancel_tree` drops the lease, so a running worker aborts the
    // stream and commits the completed prefix of the cancelled turn
    // within one heartbeat interval. A child of `cancel`, so host
    // shutdown still cascades to it.
    let task_cancel = cancel.child_token();
    let heartbeat_cancel = cancel.child_token();
    // This task's live-progress beacon. Execution bumps it at every sign of
    // work; the heartbeat below reads it once per tick and persists it to
    // THIS row (it is the actor holding this row's lease, so it is the only
    // one that can). Both halves must therefore be minted here, together —
    // a beacon bumped in one task's execution but persisted by another
    // task's heartbeat would record activity against the wrong row.
    let activity = ActivityBeacon::new();
    let heartbeat_handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
        stores: stores.clone(),
        task_id: task_id.clone(),
        thread_id: thread_id.clone(),
        worker_id: worker_id.clone(),
        lease_id: lease_id.clone(),
        lease_duration,
        heartbeat_interval,
        cancel: heartbeat_cancel.clone(),
        task_cancel: task_cancel.clone(),
        deadline: initial_deadline,
        activity: activity.clone(),
    }));

    // No up-front stall check: under stall semantics acquisition is itself
    // evidence of work (`mark_running` stamps `last_activity_at`), so a
    // freshly-(re)acquired child is never stalled at dispatch — however
    // long it queued or however stale its pre-crash activity was. The old
    // since-spawn leg that failed a child here without dispatching punished
    // exactly the queue-wait/age the founder's ruling forbids; a genuinely
    // wedged child is now reaped by the heartbeat's own per-tick check,
    // one budget after this acquisition, and never-started children by the
    // parked sweep off their `created_at` floor.
    let exec_result =
        execute_with_abort_grace(task, worker_id, stores, runtime, &task_cancel, &activity).await;

    heartbeat_cancel.cancel();
    if let Err(join_err) = heartbeat_handle.await {
        warn!(
            %worker_id,
            task_id = %task_id,
            error = %join_err,
            "heartbeat loop join failed",
        );
    }

    if let Some(Err(err)) = exec_result {
        warn!(%worker_id, error = %err, "task execution failed");
    }
}

/// Grace window between the per-task token tripping and the worker
/// force-dropping an execution future that has not returned.
///
/// Cooperative cancellation paths (mid-stream abort, seam-B partial
/// commit, attempt closes) complete in well under this; only a
/// token-blind await (a hung provider resolve, a stuck DNS lookup)
/// burns the full window before the slot is reclaimed. `pub(crate)`
/// because the detached approved-confirmation drive (grpc) applies
/// the same abort-grace pattern to Confirm-tier tools.
pub(crate) const EXECUTION_ABORT_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// Race the execution future against the per-task token.
///
/// A turn blocked in a token-blind await (e.g.
/// `ProviderResolver::resolve_provider`, which receives no token)
/// would otherwise pin its worker-pool slot forever even though the
/// row no longer needs this worker — after a timeout fail the row is
/// terminal; after a requeue-triggered lease rejection it is Pending
/// or already re-owned by a successor. On a small pool the woken
/// parent could never materialize. Cooperative paths get a bounded
/// grace window after the token trips (the stream abort and seam-B
/// salvage must finish); only a future still pending past the grace is
/// dropped (`None`). Dropping is crash-equivalent, not
/// terminal-guaranteed: every durable write on the execute path is a
/// single transactional store call, so dropping between awaits is
/// indistinguishable from a worker crash, which the journal already
/// tolerates (a requeued row is simply re-run by its next owner).
async fn execute_with_abort_grace(
    task: AgentTask,
    worker_id: &WorkerId,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    task_cancel: &CancellationToken,
    activity: &ActivityBeacon,
) -> Option<Result<()>> {
    let task_id = task.id.clone();
    let thread_id = task.thread_id.clone();
    let lease_id = task.lease_id.clone();
    let mut exec_fut = Box::pin(execute_acquired_task(
        task,
        stores,
        runtime,
        task_cancel,
        activity,
    ));
    let outcome = tokio::select! {
        biased;
        result = &mut exec_fut => Some(result),
        () = task_cancel.cancelled() => {
            match tokio::time::timeout(EXECUTION_ABORT_GRACE, &mut exec_fut).await {
                Ok(result) => Some(result),
                Err(_elapsed) => {
                    warn!(
                        %worker_id,
                        task_id = %task_id,
                        thread_id = %thread_id,
                        grace_secs = EXECUTION_ABORT_GRACE.as_secs(),
                        "execution future ignored cancellation past the grace window; \
                         dropping it (crash-equivalent; the row is terminal or owned elsewhere)",
                    );
                    None
                }
            }
        }
    };
    if outcome.is_none() {
        drop(exec_fut);
        settle_attempts_after_force_drop(
            stores,
            &task_id,
            &thread_id,
            worker_id,
            lease_id.as_ref(),
        )
        .await;
    }
    outcome
}

/// Settle turn attempts orphaned by a force-drop — but only when it is
/// provably safe to do so.
///
/// The drop killed this worker's execution mid-flight, so the
/// `LeaveOpenForLiveWorker` assumption behind the timeout fail no
/// longer holds: an attempt opened outside `call_llm_with_retry`
/// (Start persistence, completed-turn persistence) would stay OPEN
/// forever. But two of the three ways the per-task token trips mean
/// ownership already moved on (a terminal heartbeat lease rejection
/// after a sweep requeue; a `deadline_tick` clean-skip Stop) — and in
/// the requeue case a SUCCESSOR worker may already be streaming with
/// its own open attempt. Stamping that attempt Cancelled/zero would
/// make the successor's in-transaction real-usage close hit
/// `AlreadyClosed` and terminally fail a genuinely recovered turn.
///
/// So, mirroring [`fail_root_task_if_owned`]'s still-owned guard:
/// re-read the row and close open attempts only when the row is
/// terminal (no future worker can exist) or still `Running` under OUR
/// `(worker, lease)` (the dropped execution was the row's only live
/// worker). Anything else — requeued, re-owned, missing — belongs to
/// its next owner, who settles its own attempts.
async fn settle_attempts_after_force_drop(
    stores: &StoreRegistry,
    task_id: &agent_server::journal::task::AgentTaskId,
    thread_id: &agent_sdk_foundation::ThreadId,
    worker_id: &WorkerId,
    lease_id: Option<&LeaseId>,
) {
    let current = match stores.task_store.get(task_id).await {
        Ok(current) => current,
        Err(err) => {
            warn!(
                %worker_id,
                task_id = %task_id,
                thread_id = %thread_id,
                error = %err,
                "could not re-read task after force-drop; skipping attempt settlement",
            );
            return;
        }
    };
    let safe_to_close = match &current {
        None => false,
        Some(row) if row.status.is_terminal() => true,
        Some(row) => {
            row.status == TaskStatus::Running
                && row.worker_id.as_ref() == Some(worker_id)
                && row.lease_id.as_ref() == lease_id
        }
    };
    if !safe_to_close {
        let observed_status = current.as_ref().map(|row| row.status);
        warn!(
            %worker_id,
            task_id = %task_id,
            thread_id = %thread_id,
            ?observed_status,
            "skip force-drop attempt settlement: row requeued or re-owned \
             (its next owner settles its own attempts)",
        );
        return;
    }
    best_effort_close_open_attempts(
        task_id,
        stores.attempt_store.as_ref(),
        time::OffsetDateTime::now_utc(),
    )
    .await;
}

/// **Stall** deadline for a child-thread root task linked to a durable
/// subagent invocation.
///
/// `spec.timeout_ms` is a budget of *silence*, not of work: a child is
/// failed only after going the whole budget with **no evidence of
/// work**. A child that keeps working never expires, however long it
/// runs — a six-hour build worker that commits a frame every minute is
/// as healthy as a six-second one.
///
/// This is a deliberate reversal of the original since-spawn semantics
/// (issue #299 / #360), which failed *productive* children purely for
/// being old and threw away everything they had done. Age is not
/// evidence of being stuck.
///
/// ## The two conditions
///
/// [`stall_expired`] fails a child only when **both** hold:
///
/// 1. `now >= earliest_expiry_at` (`created_at + timeout_ms`) — spawn
///    counts as the initial evidence of life, so a child younger than
///    its own budget can never be stalled out. This also still reaps a
///    child that never starts at all (wedged admission, hung before its
///    first frame).
/// 2. Its own thread committed **no event** at or after
///    `now - timeout_ms` — nothing happened for a whole budget.
///
/// Together these mean "expires roughly `timeout_ms` after the last
/// sign of life", with the spawn instant as the first such sign.
///
/// Enforced by [`heartbeat_loop`] once per tick, by the parked sweep,
/// and once up front at acquisition, so enforcement carries up to one
/// tick of slack past the nominal stall point. In a multi-node
/// deployment, clock skew between the node stamping the durable
/// timestamps and the enforcing node shifts enforcement 1:1 with the
/// skew — consistent with how wall-clock lease expiry is already
/// handled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubagentExecutionDeadline {
    /// Earliest instant the child *could* be failed — `created_at +
    /// timeout_ms`. Reaching it is necessary but **not sufficient**:
    /// the child must also have been silent for the whole budget. See
    /// [`stall_expired`].
    earliest_expiry_at: time::OffsetDateTime,
    /// The resolved spec timeout. Doubles as the silence budget and as
    /// the failure message's stated timeout.
    timeout_ms: u64,
}

/// Absolute ceiling on how long a single stall probe may run, independent
/// of the lease. On the RUNNING leg the probe is additionally bounded to
/// half the current lease (see [`heartbeat_loop`]) so a slow store can
/// never lapse the freshly-renewed lease and open a double-execution
/// window; this cap keeps a very long lease from permitting an equally long
/// probe, and bounds the parked-sweep probe (which holds no lease). A probe
/// that exceeds its budget is treated as "not expired" (fail-safe) and
/// retried on the next tick/sweep.
const STALL_PROBE_MAX: std::time::Duration = std::time::Duration::from_secs(5);

/// Hard cap on the task rows one stall probe will walk.
///
/// The walk descends only through NON-terminal rows (see
/// [`collect_subtree_activity`]), so for a stalled child its LIVE frontier
/// is a handful of rows — retained terminal history is pruned and does not
/// count against this bound. The cap is a last-resort backstop against a
/// pathological live fan-out; hitting it answers "not expired" (fail-safe)
/// and is logged, never a silent truncation.
const MAX_ACTIVITY_SUBTREE_NODES: usize = 512;

/// The activity signals of a walked subtree, in the order [`stall_expired`]
/// consults them.
#[derive(Debug, Default)]
struct SubtreeActivity {
    /// A row in the subtree carried `last_activity_at >= cutoff` — a
    /// decisive, durable sign of life. The walk short-circuits on the
    /// first one, so this is `true` iff the child is *not* stalled by the
    /// durable signal.
    found_fresh: bool,
    /// Distinct threads the fully-walked subtree spans, for the committed-
    /// event probe. More than one means a nested subagent (whose frames
    /// land on ITS thread, never the enforced child's). Only meaningful
    /// when neither `found_fresh` nor `truncated`.
    threads: Vec<agent_sdk_foundation::ThreadId>,
    /// The walk was cut short (node cap or store error); its verdict is
    /// inconclusive and must not be read as "no activity".
    truncated: bool,
}

/// Walk the task subtree rooted at `root`, stopping the instant it finds a
/// durable sign of life at or after `cutoff`.
///
/// A children-BFS over `parent_id` that ALSO hops the subagent-invocation
/// linkage (mirroring `collect_subtree` / `collect_subtree_tx`). Both hops
/// are load-bearing:
///
/// * **`parent_id`** reaches a running tool child. Its row is where a long
///   tool's live progress lands (the collector's beacon → that task's own
///   heartbeat), and while it runs the parent child root is parked with no
///   heartbeat of its own. Without this hop a child parked on ONE
///   40-minute build commits nothing and reads as silent.
/// * **the invocation linkage** reaches a nested subagent's child ROOT,
///   which is a genuine root (`parent_id` is `None`) on a NEW thread — a
///   `parent_id`-only walk misses the entire nested subtree, and a
///   productive one would be killed for its parent's silence.
///
/// **Only NON-terminal rows are expanded; terminal rows are inspected but
/// never descended into.** A terminal task's descendants are all terminal
/// too (the tree completes bottom-up) and its own `last_activity_at` is its
/// completion instant, so a completed subtree can hold nothing newer than
/// the terminal row itself. Crucially, terminal rows are inspected **inline
/// from the full `list_children` rows** — no extra read, and they never
/// enter the frontier or count against the cap. So retained terminal
/// history, however wide or deep, cannot inflate the walk: only the live
/// frontier does. That is what preserves EVENTUAL enforcement — a wedged
/// child's live frontier is small, so the walk exhausts and returns a real
/// verdict instead of a perpetual "truncated → active" — while a
/// just-completed child is still seen (as a direct child of its still-live
/// parent) and counts as a fresh sign of life for a parked parent.
async fn collect_subtree_activity(
    task_store: &dyn AgentTaskStore,
    root: &AgentTaskId,
    cutoff: time::OffsetDateTime,
) -> SubtreeActivity {
    let mut out = SubtreeActivity::default();
    // Tracks only the NON-terminal rows enqueued for expansion, so the cap
    // bounds the *live* frontier — never the retained terminal history.
    let mut expanded: std::collections::BTreeSet<AgentTaskId> = std::collections::BTreeSet::new();
    let mut frontier: std::collections::VecDeque<AgentTask> = std::collections::VecDeque::new();

    // The root — and any linkage-hopped nested root — is a genuine root task
    // not returned by a `list_children`, so it needs its own read.
    match task_store.get(root).await {
        Ok(Some(task)) => {
            if note_row_activity(&task, cutoff, &mut out) {
                return out;
            }
            if !task.status.is_terminal() {
                expanded.insert(task.id.clone());
                frontier.push_back(task);
            }
        }
        // A root that vanished is not evidence of silence.
        Ok(None) => return out,
        Err(err) => {
            warn!(
                task_id = %root,
                error = %err,
                "subagent activity probe could not read the subtree root; treating the child \
                 as active (fail-safe)",
            );
            out.truncated = true;
            return out;
        }
    }

    while let Some(task) = frontier.pop_front() {
        if expanded.len() > MAX_ACTIVITY_SUBTREE_NODES {
            warn!(
                root = %root,
                cap = MAX_ACTIVITY_SUBTREE_NODES,
                "subagent activity probe hit the live-frontier cap; treating the child as \
                 active (fail-safe: never time out on incomplete evidence)",
            );
            out.truncated = true;
            return out;
        }

        // Nested subagent: its child root is a separate root task, reached
        // by an explicit read (it is not a `parent_id` child).
        if let Some(invocation) = task.state.subagent_invocation() {
            let nested = invocation.child_root_task_id.clone();
            if expanded.insert(nested.clone()) {
                match task_store.get(&nested).await {
                    Ok(Some(nested_root)) => {
                        if note_row_activity(&nested_root, cutoff, &mut out) {
                            return out;
                        }
                        if !nested_root.status.is_terminal() {
                            frontier.push_back(nested_root);
                        }
                    }
                    Ok(None) => {}
                    Err(err) => {
                        warn!(
                            task_id = %nested,
                            error = %err,
                            "subagent activity probe could not read a nested child root; \
                             treating the child as active (fail-safe)",
                        );
                        out.truncated = true;
                        return out;
                    }
                }
            }
        }

        // Direct children come back as full rows: inspect every child's
        // activity inline (so a just-completed TERMINAL child still counts
        // as a fresh sign of life for its parked parent), but enqueue only
        // NON-terminal children for expansion.
        match task_store.list_children(&task.id).await {
            Ok(children) => {
                for child in children {
                    if note_row_activity(&child, cutoff, &mut out) {
                        return out;
                    }
                    if !child.status.is_terminal() && expanded.insert(child.id.clone()) {
                        frontier.push_back(child);
                    }
                }
            }
            Err(err) => {
                warn!(
                    task_id = %task.id,
                    error = %err,
                    "subagent activity probe could not list a subtree row's children; \
                     treating the child as active (fail-safe)",
                );
                out.truncated = true;
                return out;
            }
        }
    }
    out
}

/// Fold one subtree row's durable activity into `out`. Returns `true` — a
/// decisive early stop — when the row carries a `last_activity_at` at or
/// after `cutoff` (a fresh sign of life); otherwise records the row's thread
/// for the committed-event probe and returns `false`.
fn note_row_activity(
    task: &AgentTask,
    cutoff: time::OffsetDateTime,
    out: &mut SubtreeActivity,
) -> bool {
    if task
        .last_activity_at
        .is_some_and(|activity| activity >= cutoff)
    {
        out.found_fresh = true;
        return true;
    }
    if !out.threads.contains(&task.thread_id) {
        out.threads.push(task.thread_id.clone());
    }
    false
}

/// Whether an enforced child has gone its entire `timeout_ms` budget
/// without any evidence of work — the sole condition for failing it.
///
/// ## The evidence, and why it is not just committed events
///
/// #376 asked only one question: did the child's own thread commit an
/// event inside the budget? That is a real signal — every assistant frame,
/// tool call and tool result commits one — but it is not the *whole*
/// signal, and each gap is a way to kill a perfectly healthy child:
///
/// | Healthy child | Why its journal looks silent |
/// |---|---|
/// | parked on one 40-minute build | `ToolEventCollector::emit` buffers in memory; the tool's events commit only *after* it returns |
/// | running a nested subagent | the nested frames commit on the NESTED thread, not this child's |
/// | streaming pure tool calls | `ToolUseStart`/`ToolInputDelta` carry no text or thinking, so nothing is journalled while the stream yields |
/// | anything, under tight retention | `event_ttl_secs < timeout_ms` purges an event that fell inside the window |
///
/// So the probe asks for **any** sign of life across the child's whole
/// subtree, and committed events are one input rather than the predicate:
///
/// 1. `live` — the in-memory beacon of a RUNNING child, read straight from
///    its heartbeat loop. Exact: no durable round-trip, no tick of lag.
/// 2. the newest durable `last_activity_at` across the subtree's task rows
///    — which a tool child's own heartbeat advances while it runs, and
///    which event retention cannot purge.
/// 3. a committed event on any thread the subtree spans — the child's own,
///    plus a nested subagent's.
///
/// ## What is deliberately NOT evidence
///
/// The lease heartbeat. It renews unconditionally while the task future is
/// alive, so a child hung on a half-open connection beats forever — that is
/// precisely the failure this enforcement exists to catch. A beat proves
/// the process is alive; only the beacon proves work happened.
///
/// By the same logic a *running tool child is not itself evidence*: a tool
/// wedged on a dead socket is indistinguishable from one doing 40 minutes
/// of honest work, except that the honest one reports progress. Reaping a
/// tool that reports nothing for a whole budget is the parked sweep's
/// entire reason to exist, so a tool must EMIT to be credited.
///
/// ## Fail-safe
///
/// Any store error, or a subtree too large to walk, answers `false` (not
/// expired). Never kill a child on *unknown* activity: the next tick or
/// sweep retries, and a genuinely stalled child stays stalled.
async fn stall_expired(
    stores: &StoreRegistry,
    subject: &AgentTaskId,
    enforced: SubagentExecutionDeadline,
    live: Option<time::OffsetDateTime>,
    now: time::OffsetDateTime,
) -> bool {
    // Condition 1: spawn is the initial evidence of life, so a child
    // younger than its own budget can never be stalled out — and one that
    // never starts at all is still reaped.
    if now < enforced.earliest_expiry_at {
        return false;
    }
    let budget =
        time::Duration::milliseconds(i64::try_from(enforced.timeout_ms).unwrap_or(i64::MAX));
    let Some(cutoff) = now.checked_sub(budget) else {
        // A budget too large to subtract can never elapse.
        return false;
    };

    // Condition 2: did ANYTHING in this child's subtree happen inside the
    // budget? Cheapest and most precise signal first.
    if live.is_some_and(|live| live >= cutoff) {
        return false;
    }

    let subtree = collect_subtree_activity(stores.task_store.as_ref(), subject, cutoff).await;
    // `found_fresh`: a durable stamp inside the budget anywhere in the
    // subtree. `truncated`: the walk was inconclusive — never time out on
    // unknown activity.
    if subtree.found_fresh || subtree.truncated {
        return false;
    }

    for thread_id in &subtree.threads {
        match stores
            .event_repo
            .min_sequence_at_or_after(thread_id, cutoff)
            .await
        {
            Ok(Some(_)) => return false,
            Ok(None) => {}
            Err(err) => {
                warn!(
                    thread_id = %thread_id,
                    error = %err,
                    "subagent activity probe failed; treating the child as active \
                     (fail-safe: never time out on unknown activity)",
                );
                return false;
            }
        }
    }
    true
}

/// Per-acquisition resolution state for the subagent wall-clock
/// deadline (issue #299).
///
/// A transient store failure during linkage resolution must not
/// silently disable enforcement for the whole acquisition, so the
/// failure is kept as an explicit [`Self::Unresolved`] state that the
/// heartbeat loop retries every tick until it settles into
/// [`Self::Exempt`] or [`Self::Enforced`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum SubagentDeadlineState {
    /// Definitively not an enforcement candidate: not a child-thread
    /// root, a steering resume, or a successful lookup found no
    /// linked invocation.
    Exempt,
    /// The linkage lookup hit a transient store error; retried once
    /// per heartbeat tick until it succeeds (then enforced) or the
    /// task ends.
    Unresolved {
        /// The task's `created_at`, kept so a later successful
        /// resolution can anchor the deadline without re-reading the
        /// task row.
        created_at: time::OffsetDateTime,
    },
    /// Linked child root: enforce this deadline.
    Enforced(SubagentExecutionDeadline),
}

/// Build the stall deadline from durable anchors, or `None` when
/// `created_at + timeout_ms` overflows the datetime range
/// (practically unreachable; treated as "no deadline").
///
/// Only the *earliest* expiry is derived here. Whether the child is
/// actually stalled at that point is [`stall_expired`]'s question, and
/// it is re-asked on every tick — which is what lets a working child
/// push its expiry out indefinitely.
fn deadline_for(
    created_at: time::OffsetDateTime,
    timeout_ms: u64,
    min_budget_ms: u64,
) -> Option<SubagentExecutionDeadline> {
    // Floor the budget at the persistence cadence (see `MIN_STALL_BUDGET_HEARTBEATS`).
    let effective_ms = timeout_ms.max(min_budget_ms);
    let timeout = time::Duration::milliseconds(i64::try_from(effective_ms).unwrap_or(i64::MAX));
    let earliest_expiry_at = created_at.checked_add(timeout)?;
    Some(SubagentExecutionDeadline {
        earliest_expiry_at,
        timeout_ms: effective_ms,
    })
}

/// Multiplier for the effective stall-budget floor: the durable
/// `last_activity_at` a parked child's sweep reads is persisted at most once
/// per heartbeat interval, so a budget shorter than a couple of intervals
/// could reap a parked child whose descendant IS reporting but whose report
/// has not yet been persisted. The effective budget is therefore floored at
/// `MIN_STALL_BUDGET_HEARTBEATS × heartbeat_interval` at enforcement — where
/// the actual cadence is known (spec resolution in the SDK cannot see the
/// host's heartbeat interval). k = 2 gives one interval to persist and one
/// for the sweep to observe. bip's minute-to-hours budgets are far above
/// this and unaffected; the floor only rescues pathologically small configs.
const MIN_STALL_BUDGET_HEARTBEATS: u32 = 2;

/// The effective stall-budget floor for a given heartbeat/persistence
/// cadence — see [`MIN_STALL_BUDGET_HEARTBEATS`].
fn min_stall_budget_ms(heartbeat_interval: std::time::Duration) -> u64 {
    u64::try_from(heartbeat_interval.as_millis())
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(MIN_STALL_BUDGET_HEARTBEATS))
}

/// Derive the deadline state from a freshly-read linked invocation
/// row. Shared by the acquisition-time resolution and the heartbeat
/// loop's per-tick retry of an [`SubagentDeadlineState::Unresolved`]
/// resolution.
fn deadline_from_invocation(
    invocation: &AgentTask,
    child_created_at: time::OffsetDateTime,
    min_budget_ms: u64,
) -> SubagentDeadlineState {
    invocation
        .state
        .subagent_invocation()
        .and_then(|state| deadline_for(child_created_at, state.spec.timeout_ms, min_budget_ms))
        .map_or(
            SubagentDeadlineState::Exempt,
            SubagentDeadlineState::Enforced,
        )
}

pub(crate) struct HeartbeatLoopParams {
    pub(crate) stores: StoreRegistry,
    pub(crate) task_id: agent_server::journal::task::AgentTaskId,
    pub(crate) thread_id: agent_sdk_foundation::ThreadId,
    pub(crate) worker_id: WorkerId,
    pub(crate) lease_id: LeaseId,
    pub(crate) lease_duration: time::Duration,
    pub(crate) heartbeat_interval: std::time::Duration,
    /// Trips when the heartbeat loop itself should stop (exec finished /
    /// host shutdown).
    pub(crate) cancel: CancellationToken,
    /// Trips the running worker's turn when the lease is terminally
    /// rejected (the row is no longer ours — most commonly a
    /// `cancel_tree`), so the worker aborts the stream and enters the
    /// partial-commit-on-cancel path (seam B) instead of only failing at
    /// the final commit CAS.
    pub(crate) task_cancel: CancellationToken,
    /// Subagent wall-clock deadline state for this acquisition
    /// (issue #299). [`SubagentDeadlineState::Exempt`] for every
    /// non-linked task — no enforcement.
    pub(crate) deadline: SubagentDeadlineState,
    /// Live-progress beacon for the task this loop is beating for.
    ///
    /// Read once per tick and written through to `last_activity_at` under
    /// the same lease CAS that extends the lease — so activity costs one
    /// durable write per tick, not one per provider frame, and only the
    /// worker that owns the row can advance it.
    ///
    /// Persisted for **every** task kind, not just enforced child roots: a
    /// tool child's activity is what keeps its PARKED subagent parent
    /// alive, and that parent has no heartbeat of its own while it waits.
    pub(crate) activity: ActivityBeacon,
}

pub(crate) async fn heartbeat_loop(params: HeartbeatLoopParams) {
    let HeartbeatLoopParams {
        stores,
        task_id,
        thread_id,
        worker_id,
        lease_id,
        lease_duration,
        heartbeat_interval,
        cancel,
        task_cancel,
        mut deadline,
        activity,
    } = params;
    let mut ticker = tokio::time::interval(heartbeat_interval);
    // The stall probe runs under the lease this loop renews each tick; bound
    // it to half that lease (capped) so a slow store or large subtree can
    // never outlast the lease and open a double-execution window. Half
    // leaves ample renewal headroom before the next beat.
    let probe_timeout = (lease_duration.unsigned_abs() / 2).min(STALL_PROBE_MAX);
    // Floor the stall budget at the persistence cadence, using this loop's
    // actual heartbeat interval (see `MIN_STALL_BUDGET_HEARTBEATS`).
    let min_budget_ms = min_stall_budget_ms(heartbeat_interval);
    // Skip the immediate first tick — the lease was just set by
    // acquire_next_runnable, so the first heartbeat should fire after
    // one full interval.
    ticker.tick().await;
    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => return,
            _ = ticker.tick() => {}
        }

        let now = time::OffsetDateTime::now_utc();
        // One beacon read per tick, shared by the lease renewal (which
        // persists it) and the deadline check (which enforces against it),
        // so both see the same instant. The RUNNING leg thus consults the
        // live in-memory value rather than last tick's durable one.
        let observed_activity = activity.latest();

        // Renew the lease FIRST, before the (potentially slow) subtree
        // probe in `deadline_tick`. That probe walks the child's subtree
        // with serial store reads; awaiting it before extending the lease
        // would let a large subtree or a slow store burn the remaining
        // lease, and the expiry sweep would then requeue and reacquire the
        // row while the original provider future is still running —
        // duplicate execution, the worst failure mode on this path.
        // Renewing first guarantees a full `lease_duration` of headroom
        // before any probe runs.
        let new_expires_at = now + lease_duration;
        match stores
            .task_store
            .heartbeat_task(
                &task_id,
                &worker_id,
                &lease_id,
                new_expires_at,
                observed_activity,
                now,
            )
            .await
        {
            Ok(_) => {}
            Err(err) if heartbeat_error_is_terminal(&err) => {
                // Lease rejection (not-running / worker-mismatch /
                // lease-mismatch): the row is no longer ours, so further
                // heartbeats can only fail. Trip the per-task token so a
                // still-running worker aborts its stream and enters the
                // partial-commit-on-cancel path (seam B) — the common
                // cause is a `cancel_tree` that dropped our lease — then
                // exit cleanly.
                warn!(
                    %worker_id,
                    task_id = %task_id,
                    thread_id = %thread_id,
                    error = %err,
                    "heartbeat rejected; ticker exiting (lease no longer owned)",
                );
                task_cancel.cancel();
                return;
            }
            Err(err) => {
                // Transient store error (e.g. a DB blip). Skip the deadline
                // probe on a tick whose lease renewal failed — the row may
                // already be requeued — and retry the beat next tick before
                // the lease (which has ~3 beats of headroom) expires.
                warn!(
                    %worker_id,
                    task_id = %task_id,
                    thread_id = %thread_id,
                    error = %err,
                    "heartbeat hit a transient store error; retrying on next tick",
                );
                continue;
            }
        }

        // Subagent timeout enforcement (issue #299): the lease is now
        // freshly renewed, so the subtree probe cannot jeopardise it.
        match deadline_tick(
            &stores,
            OwnedRootTask {
                task: &task_id,
                thread: &thread_id,
                worker: &worker_id,
                lease: &lease_id,
            },
            deadline,
            observed_activity,
            probe_timeout,
            min_budget_ms,
            now,
        )
        .await
        {
            DeadlineTick::Stop => {
                // Durably failed (or ownership cleanly moved on):
                // abort the in-flight turn and stop.
                task_cancel.cancel();
                return;
            }
            DeadlineTick::Continue(next) => deadline = next,
        }
    }
}

/// Outcome of one heartbeat tick's subagent-deadline handling.
enum DeadlineTick {
    /// Keep heartbeating with this (possibly updated) state.
    Continue(SubagentDeadlineState),
    /// The timeout failure landed durably (or ownership cleanly moved
    /// on): the caller trips the per-task token and exits.
    Stop,
}

/// One heartbeat tick of subagent-deadline handling: retry an
/// [`SubagentDeadlineState::Unresolved`] resolution, then enforce an
/// expired [`SubagentDeadlineState::Enforced`] deadline.
///
/// Unlike an external `cancel_tree`, the tick's worker still owns the
/// lease, so on expiry the fail CAS lands first (waking the parent
/// invocation with a failed child outcome) and only then does the
/// caller abort the in-flight stream, letting seam B salvage the
/// completed prefix. A transient store error during the fail returns
/// [`DeadlineTick::Continue`] — the row is still Running under OUR
/// lease, and dropping it would let the expiry sweep requeue the task
/// and lose the timeout outcome; the caller keeps heartbeating and
/// this tick's work is retried on the next one.
async fn deadline_tick(
    stores: &StoreRegistry,
    owned: OwnedRootTask<'_>,
    mut deadline: SubagentDeadlineState,
    live_activity: Option<time::OffsetDateTime>,
    probe_timeout: std::time::Duration,
    min_budget_ms: u64,
    now: time::OffsetDateTime,
) -> DeadlineTick {
    // A resolution that failed at acquisition time is retried here
    // every tick — a transient store blip must not run the whole
    // acquisition without its deadline.
    if let SubagentDeadlineState::Unresolved { created_at } = deadline {
        match stores
            .task_store
            .find_subagent_invocation_for_child_root(owned.task)
            .await
        {
            Ok(Some(invocation)) => {
                deadline = deadline_from_invocation(&invocation, created_at, min_budget_ms);
            }
            Ok(None) => deadline = SubagentDeadlineState::Exempt,
            Err(err) => {
                warn!(
                    worker_id = %owned.worker,
                    task_id = %owned.task,
                    thread_id = %owned.thread,
                    error = %err,
                    "subagent linkage lookup failed; retrying deadline resolution next tick",
                );
            }
        }
    }

    let SubagentDeadlineState::Enforced(enforced) = deadline else {
        return DeadlineTick::Continue(deadline);
    };
    // Re-asked every tick, which is what makes the budget a STALL budget:
    // a child that showed ANY sign of life inside the last `timeout_ms`
    // pushes its expiry out and keeps running, no matter how long it has
    // been alive. Only sustained silence ends it.
    //
    // This is the RUNNING leg, so the beacon is read live from the loop
    // that owns this row — no durable round-trip, no tick of lag.
    //
    // The probe walks the subtree with serial store reads; bound it strictly
    // below the lease it runs under (renewed just before this call) so a slow
    // store or large subtree can never lapse the lease and requeue a still-
    // running row → duplicate execution. On timeout, treat as NOT expired
    // (never reap on an incomplete probe) and let the next tick retry.
    let expired = match tokio::time::timeout(
        probe_timeout,
        stall_expired(stores, owned.task, enforced, live_activity, now),
    )
    .await
    {
        Ok(expired) => expired,
        Err(_elapsed) => {
            warn!(
                worker_id = %owned.worker,
                task_id = %owned.task,
                thread_id = %owned.thread,
                ?probe_timeout,
                "subagent stall probe exceeded its lease-bounded budget; treating the child \
                 as active (fail-safe) and retrying next tick",
            );
            false
        }
    };
    if !expired {
        return DeadlineTick::Continue(deadline);
    }

    warn!(
        worker_id = %owned.worker,
        task_id = %owned.task,
        thread_id = %owned.thread,
        timeout_ms = enforced.timeout_ms,
        "subagent child root stalled for its whole budget; failing and aborting the turn",
    );
    match fail_timed_out_subagent_root(
        stores,
        FailTimedOutChild {
            task: owned.task,
            thread: owned.thread,
            worker: owned.worker,
            lease: owned.lease,
            timeout_ms: enforced.timeout_ms,
            // The turn's worker is guaranteed live in this process
            // (this heartbeat belongs to it): leave its open attempt
            // alone so a stream that already succeeded can still
            // commit its real usage — the worker's own abort path
            // closes attempts.
            attempt_close: AttemptClosePolicy::LeaveOpenForLiveWorker,
        },
        now,
    )
    .await
    {
        Ok(()) => DeadlineTick::Stop,
        Err(err) => {
            warn!(
                worker_id = %owned.worker,
                task_id = %owned.task,
                error = %err,
                "timed-out subagent fail hit a transient store error; \
                 keeping the lease and retrying next tick",
            );
            DeadlineTick::Continue(deadline)
        }
    }
}

/// Classify an `AgentTaskStore::heartbeat_task` error as a terminal
/// lease rejection (the row is no longer owned by this worker) versus a
/// transient store error worth retrying.
///
/// The store trait returns `anyhow::Error` (no typed error), so the
/// classification keys off the canonical `"heartbeat rejected"` marker
/// that every backend stamps on a CAS rejection — the in-memory store
/// via `.context("heartbeat rejected")` over a typed `TaskSchemaError`,
/// the SQL backends via an explicit `"heartbeat rejected: …"` message.
/// Transient errors (connection resets, commit failures) carry other
/// contexts and are treated as retryable. Using a typed error instead
/// would require changing the frozen store trait.
fn heartbeat_error_is_terminal(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.to_string().contains("heartbeat rejected"))
}

async fn execute_acquired_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
    activity: &ActivityBeacon,
) -> Result<()> {
    match task.kind {
        TaskKind::RootTurn => execute_root_task(task, stores, runtime, cancel, activity).await,
        TaskKind::ToolRuntime => execute_tool_task(task, stores, runtime, cancel, activity).await,
        // A subagent invocation task neither streams from a provider nor
        // runs a tool: it only fans results in. It has no work of its own
        // to report, and it is never the subject of a stall budget (the
        // budget is enforced on the child ROOT it spawned), so it carries
        // no beacon.
        TaskKind::Subagent => execute_subagent_task_entry(task, stores, runtime).await,
    }
}

async fn execute_root_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
    activity: &ActivityBeacon,
) -> Result<()> {
    let now = time::OffsetDateTime::now_utc();
    let error_watermark = stores
        .event_repo
        .next_sequence(&task.thread_id)
        .await
        .context("reading root-task event watermark")?;

    let outcome = Box::pin(async {
        let bootstrap =
            resolve_bootstrap_context(task.clone(), stores.definition_registry.as_ref())
                .await
                .context("resolve root-task bootstrap")?;
        let inputs = build_root_worker_inputs(
            bootstrap,
            stores.thread_store.as_ref(),
            stores.checkpoint_store.as_ref(),
            stores.message_store.as_ref(),
            now,
        )
        .await
        .context("build root-worker inputs")?;
        let provider = runtime
            .provider_resolver()
            .resolve_provider(inputs.definition())
            .await
            .context("resolve runtime provider")?;

        if task.state.is_steering_resume() {
            // R2 steering wake: a mailbox note woke this parked parent
            // early (a `ReadyToResume` row carrying a steering payload).
            // Answer with interim child results, then re-park on the
            // still-running children. The spawn selector is not
            // consulted — the re-park re-binds existing children rather
            // than spawning new ones.
            let selector = runtime.subagent_spawn_selector();
            let mut deps = stores.root_turn_deps_with_selector_and_compaction(
                selector.as_ref(),
                runtime.compaction_config(),
                runtime.compaction_config().map(|_| &provider),
            );
            deps.cancel = Some(cancel);
            deps.wakeup = runtime.wakeup_signal();
            deps.activity = Some(activity);
            resume_for_steering(inputs, &task, provider.as_ref(), &deps, now)
                .await
                .context("resume parked root task for steering wake")
        } else if matches!(task.state, TaskState::ReadyToResume { .. }) {
            let selector = runtime.subagent_spawn_selector();
            let mut deps = stores.root_turn_deps_with_selector_and_compaction(
                selector.as_ref(),
                runtime.compaction_config(),
                runtime.compaction_config().map(|_| &provider),
            );
            deps.cancel = Some(cancel);
            deps.wakeup = runtime.wakeup_signal();
            deps.activity = Some(activity);
            resume_from_children(inputs, &task, provider.as_ref(), &deps, now)
                .await
                .context("resume root task from durable child results")
        } else {
            let user_input = root_task_user_input(&task)?;
            let selector = runtime.subagent_spawn_selector();
            let mut deps = stores.root_turn_deps_with_selector_and_compaction(
                selector.as_ref(),
                runtime.compaction_config(),
                runtime.compaction_config().map(|_| &provider),
            );
            deps.cancel = Some(cancel);
            deps.wakeup = runtime.wakeup_signal();
            deps.activity = Some(activity);
            agent_server::worker::execute_root_turn(
                inputs,
                user_input,
                provider.as_ref(),
                &deps,
                now,
            )
            .await
            .context("execute fresh root task")
        }
    })
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
        Err(err) => fail_or_revert_root_task(stores, &task, &err, error_watermark, now).await,
    }
}

/// Handle a failed root-task execution.
///
/// A steering wake ([`TaskState::ReadyToResume`] with a non-empty
/// steering payload) that fails mid-exchange must NOT fail the parent:
/// its mission children are still RUNNING and `fail_task` does not
/// cascade, so failing here would strand those workers and orphan their
/// eventual results. Such a failure is reverted to the pre-wake parked
/// state (see [`revert_failed_steering_wake`]) so the ordinary fan-in
/// resumes the mission when the children finish. Any other failure — or
/// a revert that itself fails — marks the root `Failed` and promotes the
/// next queued root.
async fn fail_or_revert_root_task(
    stores: &StoreRegistry,
    task: &AgentTask,
    err: &anyhow::Error,
    error_watermark: u64,
    now: time::OffsetDateTime,
) -> Result<()> {
    if task.state.is_steering_resume() {
        match revert_failed_steering_wake(stores, task, err, error_watermark, now).await {
            // Reverted (or the lease had already moved on) — the root is
            // not terminal, so do not promote a successor.
            Ok(()) => return Ok(()),
            Err(revert_err) => {
                warn!(
                    task_id = %task.id,
                    thread_id = %task.thread_id,
                    steering_error = %err,
                    revert_error = %revert_err,
                    "steering wake revert failed; falling back to failing the root task",
                );
                // Fall through to the ordinary fail path.
            }
        }
    }
    if is_turn_slot_collision_error(err) {
        match requeue_collided_root_task(stores, task, err, now).await {
            Ok(true) => {
                let new_events =
                    newly_committed_events(stores, &task.thread_id, error_watermark).await?;
                publish_events(stores, &new_events);
                return Ok(());
            }
            Ok(false) => {
                // Budget exhausted or no longer owned — fall through to
                // the ordinary terminal path, which re-checks ownership
                // itself and skips cleanly when the row moved on.
            }
            Err(requeue_err) => {
                warn!(
                    task_id = %task.id,
                    thread_id = %task.thread_id,
                    collision_error = %err,
                    requeue_error = %requeue_err,
                    "turn-slot collision requeue failed; falling back to failing the root task",
                );
            }
        }
    }
    warn!(
        task_id = %task.id,
        thread_id = %task.thread_id,
        error = %err,
        "root task execution failed; marking task failed",
    );
    fail_root_task(stores, task, err, error_watermark, now).await?;
    promote_next_root(stores, task, now).await?;
    Ok(())
}

/// `true` if `err`'s chain bottoms out in the completed-turn slot CAS
/// rejection ([`StaleTurnCommit`]) — the one failure shape that means
/// the turn's work was fine but its slot was consumed by another
/// task's commit. Every other failure keeps the ordinary terminal
/// path.
fn is_turn_slot_collision_error(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<StaleTurnCommit>().is_some())
}

/// Requeue a promoted root whose completed-turn commit lost its slot
/// to a foreign FULL-TURN checkpoint, so it re-runs from the fresh
/// committed head instead of failing terminally (issue #354).
///
/// The worker-side shift path already absorbs the benign salvage
/// collision; the case that reaches here is a cancelled predecessor's
/// fully billed turn landing after the successor bootstrapped. The
/// successor's answer was built on a head that lacks that turn, so it
/// must be re-executed — and re-execution from an uncommitted turn is
/// exactly the crash-recovery contract the journal already promises
/// (guarded tools go through durable execution intents; the same
/// retry budget applies via [`AgentTask::release_lease`]'s
/// sweep-identical accounting).
///
/// # Foreign occupants only
///
/// A `StaleTurnCommit` whose occupying checkpoint belongs to THIS task
/// means the turn already committed — a stale-lease worker losing to
/// its own replacement's (or its own earlier) landed commit. Requeueing
/// that would re-run the submitted input on top of its own committed
/// turn and durably duplicate the conversation turn, so the same-task
/// case falls through to the ordinary terminal path (the pre-existing
/// idempotency behavior). A missing occupant checkpoint is treated the
/// same, conservatively — the worker's shift path already retried
/// through the in-memory visibility gap before surfacing the error.
///
/// The rolled-back execution's attempt row is deliberately left open
/// here: closing it from this (possibly stale) worker races a
/// replacement worker's live attempt. The shift
/// wrapper's collision exits already settled this execution's own
/// attempt with its real usage before the error reached this handler.
///
/// # Fresh-input tasks only
///
/// A colliding task executing from a stored continuation
/// (`TaskState::WaitingOnChildren` / `ReadyToResume`) is NOT
/// requeued: its continuation was captured before the
/// foreign turn landed, and the re-driven resume would overwrite the
/// recovered head with that stale state — dropping the occupant's
/// usage, cost, and metadata, the exact corruption the foreign-full-
/// turn shift refusal exists to prevent. Fresh tasks re-run from the
/// submitted input against the fresh head, which is the sound
/// crash-recovery contract; stateful resumes keep the terminal path.
///
/// Returns `Ok(true)` when the row was requeued (the dispatcher will
/// re-run it); `Ok(false)` when the collision is same-task, the task
/// carries a continuation, the store reports the row has no retry
/// budget left, or the row is no longer owned — the caller falls
/// through to its terminal path.
async fn requeue_collided_root_task(
    stores: &StoreRegistry,
    task: &AgentTask,
    err: &anyhow::Error,
    now: time::OffsetDateTime,
) -> Result<bool> {
    let (worker_id, lease_id) = running_lease(task)?;
    if !matches!(task.state, TaskState::None) {
        warn!(
            task_id = %task.id,
            thread_id = %task.thread_id,
            "turn-slot collision: task carries a continuation;              keeping the terminal path instead of requeueing",
        );
        return Ok(false);
    }
    let Some(stale) = err
        .chain()
        .find_map(|cause| cause.downcast_ref::<StaleTurnCommit>())
    else {
        return Ok(false);
    };
    let occupant = stores
        .checkpoint_store
        .get_by_turn(&task.thread_id, stale.expected_turn)
        .await
        .context("read occupant checkpoint for collision classification")?;
    let foreign_occupant = occupant
        .as_ref()
        .is_some_and(|checkpoint| checkpoint.task_id != task.id);
    if !foreign_occupant {
        warn!(
            task_id = %task.id,
            thread_id = %task.thread_id,
            expected_turn = stale.expected_turn,
            occupant_task = ?occupant.map(|checkpoint| checkpoint.task_id),
            "turn-slot collision: occupant is not a foreign commit;              keeping the terminal path instead of requeueing",
        );
        return Ok(false);
    }
    // Boundary for the abandoned streamed attempt: the collided
    // execution already persisted `Start` and streaming deltas, and
    // the re-run will emit a fresh `Start` + answer. The RECOVERABLE
    // `Error` event that terminates the abandoned attempt is
    // committed by the store ATOMICALLY with the ownership CAS and
    // the release — it lands iff this worker still owned the row, and
    // it is durable before the row is acquirable — so it can neither
    // trail a replacement's `Start` nor pollute a replacement's
    // stream from a stale snapshot. On a commit failure the row stays
    // Running under our lease and the task keeps the terminal path
    // (which emits its own non-recoverable `Error`).
    let boundary = agent_sdk_foundation::events::AgentEvent::error(
        format!("turn-slot collision: retrying from the fresh committed head ({err:#})"),
        true,
    )
    .with_emitter_task_id(task.id.as_str());
    let outcome = stores
        .task_store
        .requeue_owned_task(&task.id, &worker_id, &lease_id, Some(boundary), now)
        .await
        .context("requeue collided root task")?;
    match outcome {
        RequeueOutcome::Requeued(row) => {
            warn!(
                task_id = %task.id,
                thread_id = %task.thread_id,
                attempt = row.attempt,
                max_attempts = row.max_attempts,
                "turn-slot collision: requeued root task to re-run from the fresh committed head",
            );
            Ok(true)
        }
        RequeueOutcome::BudgetExhausted => {
            warn!(
                task_id = %task.id,
                thread_id = %task.thread_id,
                "turn-slot collision: retry budget exhausted; failing the root task",
            );
            Ok(false)
        }
        RequeueOutcome::NotOwned => Ok(false),
    }
}

/// Revert a failed steering wake back to its pre-wake parked state so
/// its still-running mission children are not stranded.
///
/// Re-reads the row first: if the lease has moved on (a sweep requeued
/// it, another worker re-acquired, or the row is no longer a steering
/// resume) this skips cleanly and leaves the terminal decision to
/// whoever owns the row — the same ownership guard [`fail_root_task`]
/// uses. On the owned path it delegates to
/// [`revert_steering_wake`](agent_server::worker::revert_steering_wake),
/// which re-parks on the original children and surfaces the failure as
/// an event, then publishes the newly committed events.
///
/// Returns `Ok(())` whether the revert ran or was skipped; the caller
/// must not promote a successor either way because the root is not
/// terminal. Propagates only genuine store errors so the caller can
/// fall back to failing the task.
async fn revert_failed_steering_wake(
    stores: &StoreRegistry,
    task: &AgentTask,
    error: &anyhow::Error,
    event_watermark: u64,
    now: time::OffsetDateTime,
) -> Result<()> {
    let (worker_id, lease_id) = running_lease(task)?;

    let current = stores
        .task_store
        .get(&task.id)
        .await
        .context("re-read steering task before revert")?;
    let still_owned = current.as_ref().is_some_and(|t| {
        t.status == TaskStatus::Running
            && t.worker_id.as_ref() == Some(&worker_id)
            && t.lease_id.as_ref() == Some(&lease_id)
            && t.state.is_steering_resume()
    });
    let Some(current) = current.filter(|_| still_owned) else {
        warn!(
            task_id = %task.id,
            thread_id = %task.thread_id,
            "skip steering revert: lease no longer owned or row is no longer a steering resume",
        );
        return Ok(());
    };

    revert_steering_wake(
        &current,
        &worker_id,
        &lease_id,
        error,
        &stores.root_turn_deps(),
        now,
    )
    .await
    .context("revert failed steering wake")?;

    let new_events = newly_committed_events(stores, &task.thread_id, event_watermark).await?;
    publish_events(stores, &new_events);
    Ok(())
}

async fn execute_tool_task(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    cancel: &CancellationToken,
    activity: &ActivityBeacon,
) -> Result<()> {
    let now = time::OffsetDateTime::now_utc();
    let (worker_id, lease_id) = running_lease(&task)?;

    // The beacon rides the bootstrap into the `ToolEventCollector`, so a
    // tool that reports progress refreshes THIS tool task's row — which is
    // the row this task's heartbeat owns.
    let bootstrap =
        match resolve_tool_bootstrap(task.clone(), stores.task_store.as_ref(), activity.clone())
            .await
        {
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
                committed_events,
                parent,
                ..
            }
            | ToolTaskOutcome::Failed {
                committed_events,
                parent,
                ..
            },
        ) => {
            publish_events(stores, &committed_events);
            // When this was the batch's last outstanding child, the
            // journal flips the parent from `WaitingOnChildren` to
            // `Pending` (runnable) in the same terminal transition. Nudge
            // a parked worker so it resumes the parent turn immediately
            // rather than waiting out the acquisition ticker. A `None`
            // parent (still other live children) or a non-runnable parent
            // leaves the poll backstop to handle it.
            if parent.is_some_and(|p| p.status.is_runnable())
                && let Some(signal) = runtime.wakeup_signal()
            {
                signal.notify_workers();
            }
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

async fn execute_subagent_task_entry(
    task: AgentTask,
    stores: &StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
) -> Result<()> {
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
            committed_events,
            parent_task,
            ..
        }) => {
            publish_events(stores, &committed_events);
            // Completing the last child of a batch flips the parent
            // root turn from `WaitingOnChildren` to `Pending` (runnable)
            // in the same terminal transition. Nudge a parked worker so
            // it resumes the parent turn immediately instead of waiting
            // out the acquisition ticker. A `None` parent (still other
            // live children) or a non-runnable parent leaves the poll
            // backstop to handle it.
            if parent_task.is_some_and(|p| p.status.is_runnable())
                && let Some(signal) = runtime.wakeup_signal()
            {
                signal.notify_workers();
            }
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
    fail_root_task_if_owned(
        stores,
        OwnedRootTask {
            task: &task.id,
            thread: &task.thread_id,
            worker: &worker_id,
            lease: &lease_id,
        },
        error,
        event_watermark,
        AttemptClosePolicy::CloseOpenAttempts,
        now,
    )
    .await
}

/// Identity of a running root task this worker believes it owns.
///
/// Bundles the four coordinates the still-owned fail path needs so
/// callers that only hold ids (the heartbeat loop) and callers that
/// hold the full acquired row ([`fail_root_task`]) share one guarded
/// transition.
struct OwnedRootTask<'a> {
    task: &'a agent_server::journal::task::AgentTaskId,
    thread: &'a agent_sdk_foundation::ThreadId,
    worker: &'a WorkerId,
    lease: &'a LeaseId,
}

/// Whether the guarded fail path may pre-close the task's open turn
/// attempts.
///
/// Closing is correct when no live worker owns the turn (the generic
/// error path, the acquisition-expiry path, the deadline sweep) —
/// stale open attempts from a crashed lease holder would otherwise
/// linger forever. It is WRONG when the turn's worker is still live in
/// this process (the heartbeat-timeout path): in the window between a
/// successful stream and `commit_completed_turn`, pre-closing the open
/// attempt as `Cancelled`/zero-tokens clobbers the real-usage close on
/// the billing-source-of-truth attempt rows and makes the
/// in-transaction close hit `AlreadyClosed`, aborting the commit.
#[derive(Clone, Copy, Debug)]
enum AttemptClosePolicy {
    CloseOpenAttempts,
    LeaveOpenForLiveWorker,
}

async fn fail_root_task_if_owned(
    stores: &StoreRegistry,
    owned: OwnedRootTask<'_>,
    error: &anyhow::Error,
    event_watermark: u64,
    attempt_close: AttemptClosePolicy,
    now: time::OffsetDateTime,
) -> Result<()> {
    let OwnedRootTask {
        task: task_id,
        thread: thread_id,
        worker: worker_id,
        lease: lease_id,
    } = owned;

    // Re-read the row before transitioning. If the lease has moved on
    // — sweep requeued the row, another worker re-acquired, or the
    // sweep already failed it closed — `fail_task` will reject the CAS
    // and propagate as `mark root task failed`. That cascading error
    // produces the noisy "task execution failed" loop seen in
    // production. Skip the transition cleanly when we no longer own
    // the row; whichever path took it over is responsible for the
    // task's terminal state.
    let current = stores
        .task_store
        .get(task_id)
        .await
        .context("re-read task before fail")?;
    let still_owned = current.as_ref().is_some_and(|t| {
        t.status == TaskStatus::Running
            && t.worker_id.as_ref() == Some(worker_id)
            && t.lease_id.as_ref() == Some(lease_id)
    });
    if !still_owned {
        let observed_status = current.as_ref().map(|t| t.status);
        warn!(
            task_id = %task_id,
            thread_id = %thread_id,
            ?observed_status,
            "skip fail_root_task: lease no longer owned by this worker (sweep or re-acquire took over)",
        );
        return Ok(());
    }

    match attempt_close {
        AttemptClosePolicy::CloseOpenAttempts => {
            fail_root_turn(
                task_id,
                worker_id,
                lease_id,
                thread_id,
                error,
                &stores.root_turn_deps(),
                now,
            )
            .await
            .context("mark root task failed")?;
        }
        AttemptClosePolicy::LeaveOpenForLiveWorker => {
            fail_root_turn_leaving_attempts_open(
                task_id,
                worker_id,
                lease_id,
                thread_id,
                error,
                &stores.root_turn_deps(),
                now,
            )
            .await
            .context("mark root task failed")?;
        }
    }

    let new_events = newly_committed_events(stores, thread_id, event_watermark).await?;
    publish_events(stores, &new_events);
    Ok(())
}

/// Store-free eligibility precheck for subagent deadline enforcement:
/// a non-steering child-thread root turn MAY carry a deadline
/// ([`SubagentDeadlineState::Unresolved`], pending the linkage
/// lookup); every other task is [`SubagentDeadlineState::Exempt`].
///
/// Kept store-free so [`run_task_with_heartbeat`] can seed the
/// heartbeat's deadline state BEFORE any store lookup runs — the
/// lookup happens only under the heartbeat's lease protection, so a
/// stalled lookup cannot outlive the acquired lease and open a
/// duplicate-execution window.
fn initial_deadline_state(task: &AgentTask) -> SubagentDeadlineState {
    if task.kind == TaskKind::RootTurn && task.is_root() && !task.state.is_steering_resume() {
        SubagentDeadlineState::Unresolved {
            created_at: task.created_at,
        }
    } else {
        SubagentDeadlineState::Exempt
    }
}

/// Resolve the wall-clock execution deadline for an acquired task.
///
/// Returns [`SubagentDeadlineState::Enforced`] only for a child-thread
/// root turn that is durably linked to a parked [`TaskKind::Subagent`]
/// invocation — every other task (plain roots, tool children,
/// invocations) resolves to [`SubagentDeadlineState::Exempt`] and runs
/// with today's unbounded-lease behavior. The deadline is anchored at
/// the child root's `created_at` plus the invocation spec's resolved
/// `timeout_ms`, mirroring the in-process SDK subagent semantics ("the
/// child took too long since spawn"), and spans the child's whole
/// durable execution: queue wait, retries, and re-acquisitions all
/// consume the same budget.
///
/// A steering-resume wake is excluded: it is a short interim exchange
/// on a parent whose own children are still running, and failing it
/// would strand those children. This means a steering exchange that
/// itself hangs is bounded by nothing until the task's next ordinary
/// acquisition — a deliberate trade against stranding live children.
///
/// A transient lookup failure resolves to
/// [`SubagentDeadlineState::Unresolved`]; the heartbeat loop retries
/// the lookup every tick so a store blip cannot silently disable
/// enforcement for the whole acquisition.
/// Resolve a task's subagent deadline against the store, distinguishing
/// enforced child roots from exempt tasks and surfacing a transient lookup
/// failure as [`SubagentDeadlineState::Unresolved`].
///
/// Test-only: the live enforcement legs derive their deadline differently —
/// the heartbeat seeds from [`initial_deadline_state`] and re-resolves an
/// `Unresolved` state per tick inside [`deadline_tick`], and the parked
/// sweep derives from the invocation row it already holds. This helper
/// gives tests one call that mirrors that resolution end to end.
#[cfg(test)]
async fn resolve_subagent_deadline(
    stores: &StoreRegistry,
    task: &AgentTask,
) -> SubagentDeadlineState {
    if matches!(initial_deadline_state(task), SubagentDeadlineState::Exempt) {
        return SubagentDeadlineState::Exempt;
    }
    match stores
        .task_store
        .find_subagent_invocation_for_child_root(&task.id)
        .await
    {
        Ok(Some(invocation)) => deadline_from_invocation(&invocation, task.created_at, 0),
        Ok(None) => SubagentDeadlineState::Exempt,
        Err(err) => {
            warn!(
                task_id = %task.id,
                thread_id = %task.thread_id,
                error = %err,
                "subagent linkage lookup failed at acquisition; will retry each heartbeat tick",
            );
            SubagentDeadlineState::Unresolved {
                created_at: task.created_at,
            }
        }
    }
}

/// Coordinates of a timed-out subagent child root about to be failed.
#[derive(Clone, Copy)]
struct FailTimedOutChild<'a> {
    task: &'a agent_server::journal::task::AgentTaskId,
    thread: &'a agent_sdk_foundation::ThreadId,
    worker: &'a WorkerId,
    lease: &'a LeaseId,
    timeout_ms: u64,
    attempt_close: AttemptClosePolicy,
}

/// Fail a subagent child-thread root that exceeded its wall-clock
/// deadline, routed through the same still-owned guard and
/// [`fail_root_turn`] machinery as any other root-task failure.
///
/// The store-side `fail_task` transition wakes the linked invocation
/// in the same write (the fan-in path a failed child already takes),
/// so the parent resumes with a `success = false` child outcome
/// carrying the timeout message — deliberately FAILED, not Cancelled,
/// so the parent LLM sees an actionable error.
async fn fail_timed_out_subagent_root(
    stores: &StoreRegistry,
    ctx: FailTimedOutChild<'_>,
    now: time::OffsetDateTime,
) -> Result<()> {
    let event_watermark = stores
        .event_repo
        .next_sequence(ctx.thread)
        .await
        .context("reading timeout event watermark")?;
    let error = anyhow!("subagent timed out after {}ms", ctx.timeout_ms);
    fail_root_task_if_owned(
        stores,
        OwnedRootTask {
            task: ctx.task,
            thread: ctx.thread,
            worker: ctx.worker,
            lease: ctx.lease,
        },
        &error,
        event_watermark,
        ctx.attempt_close,
        now,
    )
    .await
}

/// Drive a timeout failure to a durable outcome while holding the
/// lease.
///
/// A transient store error while failing the row must not drop the
/// freshly-acquired lease with the row still `Running` — the expiry
/// sweep would requeue the task (burning an attempt and losing the
/// timeout outcome, eventually replacing it with a fail-closed budget
/// message). Keep the lease alive between retries and only stop once
/// the failure landed, ownership was genuinely lost (the still-owned
/// guard's clean skip / a terminal heartbeat rejection), the bounded
/// budget ran out, or the host shuts down.
///
/// Returns `true` once the row is settled from this caller's
/// perspective (durable timeout failure landed, or ownership cleanly
/// moved to another actor who now owns the terminal decision); `false`
/// when the `max_tries` budget was exhausted or the host shut down
/// mid-retry — in both `false` cases the row is left `Running` under a
/// freshly-extended lease and converges through lease expiry.
///
/// The only caller is the parked sweep, whose shared task must not block
/// indefinitely on one wedged row (it still owes the rest of the batch
/// plus the next lease-expiry pass), so the budget is always bounded.
async fn fail_timed_out_child_holding_lease(
    stores: &StoreRegistry,
    ctx: FailTimedOutChild<'_>,
    cancel: &CancellationToken,
    lease_duration: time::Duration,
    retry_interval: std::time::Duration,
    max_tries: usize,
) -> bool {
    let mut tries = 0usize;
    loop {
        let now = time::OffsetDateTime::now_utc();
        match fail_timed_out_subagent_root(stores, ctx, now).await {
            Ok(()) => return true,
            Err(err) => {
                warn!(
                    task_id = %ctx.task,
                    thread_id = %ctx.thread,
                    error = %err,
                    "timed-out subagent fail hit a transient store error; \
                     keeping the lease and retrying",
                );
            }
        }

        tries = tries.saturating_add(1);
        if tries >= max_tries {
            return false;
        }

        // Keep the lease alive while we retry. A terminal rejection
        // means ownership genuinely moved on — whoever owns the row now
        // is responsible for its terminal state.
        let new_expires_at = now + lease_duration;
        // Lease-keepalive only, while we retry a timeout FAILURE. This row
        // is being reaped, not worked: it has no activity to report, and
        // `None` leaves `last_activity_at` untouched.
        if let Err(err) = stores
            .task_store
            .heartbeat_task(ctx.task, ctx.worker, ctx.lease, new_expires_at, None, now)
            .await
            && heartbeat_error_is_terminal(&err)
        {
            warn!(
                task_id = %ctx.task,
                error = %err,
                "lease no longer owned; abandoning timed-out subagent fail retries",
            );
            return true;
        }

        tokio::select! {
            () = cancel.cancelled() => return false,
            () = tokio::time::sleep(retry_interval) => {}
        }
    }
}

/// One sweep pass of subagent deadline enforcement for PARKED child
/// roots (issue #299, parked leg).
///
/// The heartbeat enforces deadlines only while a worker holds the
/// child root; once the root suspends into `WaitingOnChildren` (its
/// own tool children) no heartbeat exists, so a hung TOOL child would
/// wedge the parent past `timeout_ms` indefinitely. This pass runs
/// from the host's periodic sweep:
///
/// 1. Candidates come from one store-side filtered read —
///    [`AgentTaskStore::list_parked_subagent_invocations`] (status
///    index + kind predicate), so unrelated parked parents are never
///    materialized. An invocation stays `WaitingOnChildren` for the
///    child root's whole lifetime, so this covers every parked child
///    state without a full-table scan (mirroring the lease sweep's
///    indexed-batch cost model: O(parked invocations) per tick).
/// 2. Per candidate, the deadline derives from the same durable
///    anchors as the heartbeat path (child `created_at` +
///    `spec.timeout_ms`). `Running` children are skipped — see the
///    ordering invariant on the status match below — as are terminal
///    ones.
/// 3. Enforcement cancels the parked root's live descendants first
///    (`cancel_tree` per child, cascading through nested subagent
///    linkage so nothing is stranded), which flips the parked root to
///    `Pending` in the same store transition; the root is then
///    acquired and failed through the identical still-owned machinery
///    as the live path — with the same keep-lease durable-fail retry
///    the other legs have — so the parent fan-in sees
///    `success = false` with the timeout message, never a Cancelled
///    child result.
///
/// Every step is guarded by re-reads and CAS transitions, and the
/// whole pass is idempotent across ticks: losing any race (a worker
/// re-acquired the root, a concurrent cancel) is a clean skip that the
/// acquisition-expiry check or the next tick converges on.
async fn enforce_subagent_deadlines(
    stores: &StoreRegistry,
    now: time::OffsetDateTime,
    min_budget_ms: u64,
    cancel: &CancellationToken,
) -> Result<usize> {
    let parked = stores
        .task_store
        .list_parked_subagent_invocations()
        .await
        .context("list parked subagent invocations for deadline sweep")?;

    let mut enforced = 0usize;
    for invocation in parked {
        let Some(linkage) = invocation.state.subagent_invocation() else {
            continue;
        };
        let child_root_id = linkage.child_root_task_id.clone();
        let Some(deadline) = deadline_for(
            invocation.created_at,
            linkage.spec.timeout_ms,
            min_budget_ms,
        ) else {
            continue;
        };
        // Cheap pre-filter on the invocation's own creation time: the
        // child root is created in the same store transition, so it can
        // never expire BEFORE this bound (and under stall semantics it
        // usually expires much later — every committed frame pushes it
        // out). Skipping early avoids a per-candidate child-root read
        // and an activity probe on every child that cannot possibly be
        // stalled yet; the authoritative check below re-derives the
        // bound from the child root's own `created_at` and then asks
        // whether the child has actually gone silent.
        if now < deadline.earliest_expiry_at {
            continue;
        }
        let child_root = match stores.task_store.get(&child_root_id).await {
            Ok(Some(child_root)) => child_root,
            Ok(None) => continue,
            Err(err) => {
                warn!(
                    child_root = %child_root_id,
                    error = %err,
                    "subagent deadline sweep could not read a linked child root; skipping",
                );
                continue;
            }
        };
        let Some(deadline) = deadline_for(
            child_root.created_at,
            linkage.spec.timeout_ms,
            min_budget_ms,
        ) else {
            continue;
        };
        // A parked child whose descendants are still working is making
        // progress THROUGH them — this is the leg where that matters most,
        // because a child root parks precisely while a tool child or a
        // nested subagent runs, and it keeps no heartbeat of its own while
        // parked. Its evidence is therefore entirely durable, and entirely
        // in the subtree: the running descendant's `last_activity_at`
        // (advanced by that descendant's OWN heartbeat) and the events on
        // the threads the subtree spans. `live` is `None` here — no beacon
        // for this row exists in this process. The probe is bounded so one
        // pathological subtree cannot stall the whole sweep tick; a timeout
        // is treated as not-expired (fail-safe) and retried next sweep.
        let expired = match tokio::time::timeout(
            STALL_PROBE_MAX,
            stall_expired(stores, &child_root.id, deadline, None, now),
        )
        .await
        {
            Ok(expired) => expired,
            Err(_elapsed) => {
                warn!(
                    child_root = %child_root.id,
                    "parked subagent stall probe exceeded its budget; treating as active \
                     (fail-safe) and retrying next sweep",
                );
                false
            }
        };
        if !expired {
            continue;
        }
        match child_root.status {
            // ORDERING INVARIANT: skipping Running rows is only safe
            // because the expired-lease drain ran EARLIER in this same
            // sweep tick — a ghost-leased row (dead worker, expired
            // lease) has already been requeued to `Pending` by that
            // pass, so any row still Running here holds a live lease
            // and its worker's heartbeat owns enforcement. If the
            // sweep's phases are ever reordered, ghost-leased rows
            // would be skipped here AND miss the expiry pass, stalling
            // convergence by a full tick. One bounded exception: the
            // drain's `MAX_LEASE_SWEEP_ROUNDS` guard means a backlog
            // beyond ~10k expired rows can still leave ghost-Running
            // rows into THIS tick's pass — those are skipped here and
            // converge on the next tick once the drain catches up.
            TaskStatus::Running => continue,
            // Terminal rows have already woken the invocation.
            status if status.is_terminal() => continue,
            TaskStatus::Pending | TaskStatus::WaitingOnChildren => {}
            // Queued / AwaitingConfirmation never apply to
            // child-thread roots; leave anything unexpected alone.
            _ => continue,
        }
        match enforce_parked_child_deadline(stores, &child_root, deadline.timeout_ms, cancel, now)
            .await
        {
            Ok(true) => enforced += 1,
            // Lost a benign race (another worker acquired, a child
            // spawned concurrently) or exhausted the bounded fail
            // budget: the acquisition-expiry check, lease expiry, or
            // the next sweep tick converges.
            Ok(false) => {}
            Err(err) => {
                warn!(
                    child_root = %child_root.id,
                    thread_id = %child_root.thread_id,
                    error = %err,
                    "parked subagent deadline enforcement failed; retrying next sweep",
                );
            }
        }
    }
    Ok(enforced)
}

/// Lease horizon for the sweep's own acquisition of a timed-out parked
/// root, and the extension unit while its durable fail is retried.
const SWEEP_FAIL_LEASE_DURATION: time::Duration = time::Duration::seconds(30);
/// Pause between the sweep's durable-fail retries.
const SWEEP_FAIL_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);
/// Durable-fail tries per sweep pass before the row is left to lease
/// expiry — enough to absorb a store blip without wedging the shared
/// sweep task on one row.
const SWEEP_FAIL_RETRY_TRIES: usize = 3;

/// Terminate one parked, past-deadline subagent child root: cancel its
/// live descendants (so hung tool children are not stranded), then
/// acquire the now-`Pending` root and fail it with the timeout message
/// through the same still-owned machinery as the live path.
///
/// Returns `Ok(true)` when this pass durably failed the root,
/// `Ok(false)` on a benign lost race or an exhausted fail budget.
async fn enforce_parked_child_deadline(
    stores: &StoreRegistry,
    child_root: &AgentTask,
    timeout_ms: u64,
    cancel: &CancellationToken,
    now: time::OffsetDateTime,
) -> Result<bool> {
    // 1. Cancel every live descendant. `cancel_tree` cascades through
    //    nested subagent linkage, drops in-flight leases (their
    //    workers abort via heartbeat rejection → seam B), and — via
    //    the store's terminal-transition propagation — flips the
    //    parked root to `Pending` when its last live child cancels.
    let children = stores
        .task_store
        .list_children(&child_root.id)
        .await
        .context("list parked child root's descendants")?;
    for child in children {
        if child.status.is_terminal() {
            continue;
        }
        let outcome = stores
            .task_store
            .cancel_tree(&child.id, now)
            .await
            .with_context(|| format!("cancel hung descendant {} of timed-out child", child.id))?;
        // An approved Confirm-tier tool in this subtree runs on a
        // DETACHED drive (`drive_approved_confirmation`) that would
        // otherwise only notice the cancel at its next heartbeat
        // rejection — its side effect could land after the parent
        // resumed. Trip each cancelled task's live drive token now so
        // the in-flight tool aborts immediately; ids without a drive
        // are no-ops.
        for cancelled_id in &outcome.transitioned {
            stores.confirm_drive_cancels.cancel(cancelled_id);
        }
        // The cancelled descendants are tool children (no marker) or
        // subagent invocations whose cascade reaches child-thread
        // roots: those roots' terminal `Cancelled` markers were
        // committed durably on their OWN threads (never the parent's)
        // inside the cancel transaction. Wake same-process followers
        // of those child threads now; cross-host followers ride the
        // outbox advisory.
        if !outcome.markers.is_empty() {
            stores.event_notifier.notify(&outcome.markers);
        }
    }

    // 2. Re-read: the root must now be Pending (it already was for a
    //    queued/ReadyToResume root). Anything else means a racing
    //    actor owns it — leave the terminal decision to them.
    let current = stores
        .task_store
        .get(&child_root.id)
        .await
        .context("re-read parked child root after descendant cancel")?;
    let Some(current) = current else {
        return Ok(false);
    };
    if current.status != TaskStatus::Pending {
        return Ok(false);
    }

    // 3. Acquire + fail through the identical machinery as the live
    //    path — including the keep-lease durable-fail retry, so a
    //    transient store error cannot leave the row Running under this
    //    heartbeat-less sweep lease (the next tick skips Running rows,
    //    so convergence would detour through lease expiry, burn an
    //    attempt, and repeated flakes could exhaust the budget and
    //    replace the timeout message with a fail-closed one). Losing
    //    the acquire CAS is benign: whoever won runs the
    //    acquisition-expiry check against the same expired deadline.
    let worker = WorkerId::from_string(format!("deadline-sweep-{}", LeaseId::new()));
    let lease = LeaseId::new();
    let lease_expires_at = now + SWEEP_FAIL_LEASE_DURATION;
    if stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            worker.clone(),
            lease.clone(),
            lease_expires_at,
            now,
        )
        .await
        .context("acquire expired parked child root")?
        .is_none()
    {
        return Ok(false);
    }
    let settled = fail_timed_out_child_holding_lease(
        stores,
        FailTimedOutChild {
            task: &child_root.id,
            thread: &child_root.thread_id,
            worker: &worker,
            lease: &lease,
            timeout_ms,
            // No live worker owns this parked turn; stale open
            // attempts from a crashed prior lease holder are safe to
            // close.
            attempt_close: AttemptClosePolicy::CloseOpenAttempts,
        },
        cancel,
        SWEEP_FAIL_LEASE_DURATION,
        SWEEP_FAIL_RETRY_INTERVAL,
        SWEEP_FAIL_RETRY_TRIES,
    )
    .await;
    if !settled {
        warn!(
            child_root = %child_root.id,
            thread_id = %child_root.thread_id,
            tries = SWEEP_FAIL_RETRY_TRIES,
            "sweep timeout fail exhausted its retry budget; leaving the row to lease expiry",
        );
    }
    Ok(settled)
}

fn publish_events(stores: &StoreRegistry, events: &[CommittedEvent]) {
    if !events.is_empty() {
        stores.event_notifier.notify(events);
    }
}

async fn newly_committed_events(
    stores: &StoreRegistry,
    thread_id: &agent_sdk_foundation::ThreadId,
    watermark: u64,
) -> Result<Vec<CommittedEvent>> {
    committed_events_from(stores.event_repo.as_ref(), thread_id, watermark)
        .await
        .context("read committed events after failure")
}

/// Load committed events with `sequence >= start_sequence` using a
/// bounded range query rather than materialising the entire journal.
///
/// Shared by the watermark-tail publish paths ([`newly_committed_events`]
/// here and the gRPC approved-confirmation drive) so a thread whose
/// journal is thousands of events long does not pay an O(journal-length)
/// read just to publish a handful of tail events.
pub(crate) async fn committed_events_from(
    event_repo: &dyn agent_server::journal::event_repository::EventRepository,
    thread_id: &agent_sdk_foundation::ThreadId,
    start_sequence: u64,
) -> Result<Vec<CommittedEvent>> {
    match start_sequence.checked_sub(1) {
        // start_sequence >= 1: bounded range query — no full-journal
        // load. `get_events_in_range` is exclusive on its lower bound, so
        // `after = start_sequence - 1` yields `sequence >= start_sequence`.
        Some(after) => {
            let next = event_repo
                .next_sequence(thread_id)
                .await
                .context("reading event watermark")?;
            let Some(up_to) = next.checked_sub(1) else {
                return Ok(Vec::new());
            };
            if up_to <= after {
                return Ok(Vec::new());
            }
            event_repo
                .get_events_in_range(thread_id, after, up_to)
                .await
                .context("reading committed events in range")
        }
        // start_sequence == 0: the thread had no events when the
        // watermark was captured, so the journal is just the new tail —
        // a full read is cheap, and the exclusive-lower range query
        // cannot express "from sequence 0 inclusive".
        None => event_repo
            .get_events(thread_id)
            .await
            .context("reading committed events from start"),
    }
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

/// Convert the root task's typed submitted-input items into the
/// worker's [`agent_server::worker::UserInput`] payload.
///
/// The submitted-input shape (`text | image | document`) flows
/// straight from the gRPC `UserInputItem` proto through the journal,
/// and from here straight into `Message::user_with_content(blocks)`
/// at the LLM call site — image and document attachments survive the
/// daemon path end-to-end without lossy string flattening.
fn root_task_user_input(task: &AgentTask) -> Result<agent_server::worker::UserInput> {
    if task.submitted_input.is_empty() {
        bail!("root task missing submitted input");
    }
    Ok(agent_server::worker::user_input_from_submitted(
        &task.submitted_input,
    ))
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
    use agent_sdk_foundation::llm::{
        ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
    };
    use agent_sdk_providers::LlmProvider;
    use agent_server::journal::task::SubmittedInputItem;
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
            tools_fn: None,
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

    // ── root_task_user_input — gRPC binary inputs round-trip ────

    fn task_with_input(items: Vec<SubmittedInputItem>) -> AgentTask {
        let thread_id = agent_sdk_foundation::ThreadId::from_string("t-image-input");
        let now = time::OffsetDateTime::UNIX_EPOCH;
        let mut task = AgentTask::new_root_turn(thread_id, now, 3);
        task.submitted_input = items;
        task
    }

    #[test]
    fn root_task_user_input_preserves_image_attachments() -> Result<()> {
        // Regression: before this slice landed, `root_task_prompt`
        // explicitly rejected `Image` and `Document` items with
        // `"root task input item is not supported by the service host
        // yet"`, so any `submit_thread_work` call carrying a
        // `BinaryAttachment` failed at root-task entry. The new
        // `root_task_user_input` builds a typed `UserInput` whose
        // block list flows straight into
        // `Message::user_with_content(blocks)` at the LLM call site.
        let task = task_with_input(vec![
            SubmittedInputItem::Text {
                text: "what's in this picture?".into(),
            },
            SubmittedInputItem::Image {
                media_type: "image/png".into(),
                data_base64: "AAAA".into(),
            },
        ]);

        let user_input = root_task_user_input(&task)?;
        assert_eq!(user_input.blocks().len(), 2);
        assert!(matches!(
            &user_input.blocks()[0],
            agent_sdk_foundation::llm::ContentBlock::Text { text } if text == "what's in this picture?"
        ));
        assert!(matches!(
            &user_input.blocks()[1],
            agent_sdk_foundation::llm::ContentBlock::Image { source }
                if source.media_type == "image/png" && source.data == "AAAA"
        ));

        // The audit-string projection turns binary blocks into
        // `[<media_type> attachment]` markers so audit rows stay
        // descriptive without storing the raw payload twice.
        assert_eq!(
            user_input.audit_summary(),
            "what's in this picture?\n[image/png attachment]",
        );
        Ok(())
    }

    #[test]
    fn root_task_user_input_preserves_document_attachments() -> Result<()> {
        let task = task_with_input(vec![
            SubmittedInputItem::Text {
                text: "summarise".into(),
            },
            SubmittedInputItem::Document {
                media_type: "application/pdf".into(),
                data_base64: "JVBERi0xLjQK".into(),
            },
        ]);

        let user_input = root_task_user_input(&task)?;
        assert_eq!(user_input.blocks().len(), 2);
        assert!(matches!(
            &user_input.blocks()[1],
            agent_sdk_foundation::llm::ContentBlock::Document { source }
                if source.media_type == "application/pdf" && source.data == "JVBERi0xLjQK"
        ));
        Ok(())
    }

    #[test]
    fn root_task_user_input_rejects_empty_submission() {
        let task = task_with_input(Vec::new());
        let result = root_task_user_input(&task);
        assert!(result.is_err(), "empty submitted_input should error");
        assert!(format!("{:#}", result.unwrap_err()).contains("missing submitted input"),);
    }

    // ── Host construction ─────────────────────────────────────────

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

    /// A completed-turn slot collision the shift refused must REQUEUE
    /// the promoted root instead of failing it terminally — the
    /// cancellation contract promises follow-up work completes, and
    /// re-running from the fresh committed head has crash-equivalent
    /// semantics.
    #[tokio::test]
    async fn turn_slot_collision_requeues_the_root_instead_of_failing() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = agent_sdk_foundation::ThreadId::from_string("t-collision-requeue");
        let root = AgentTask::new_root_turn(thread, time::OffsetDateTime::now_utc(), 3);
        let id = root.id.clone();
        stores.task_store.submit_root_turn(root).await?;
        let worker = WorkerId::from_string("w-collided");
        let lease = LeaseId::from_string("l-collided");
        let acquired = stores
            .task_store
            .try_acquire_task(
                &id,
                worker.clone(),
                lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("acquire")?;

        // The slot was consumed by a FOREIGN task's commit — the only
        // occupant shape the requeue path may act on.
        commit_occupant_checkpoint(&stores, &acquired.thread_id, 1, "task_foreign_predecessor")
            .await?;

        let collision = anyhow::Error::new(agent_server::journal::commit::StaleTurnCommit {
            expected_turn: 1,
            committed_turns: 1,
        })
        .context("commit completed turn")
        .context("execute fresh root task");
        fail_or_revert_root_task(
            &stores,
            &acquired,
            &collision,
            0,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let row = stores.task_store.get(&id).await?.context("row")?;
        assert_eq!(
            row.status,
            TaskStatus::Pending,
            "a slot collision must requeue, not fail; got {:?} ({:?})",
            row.status,
            row.last_error,
        );
        assert!(row.worker_id.is_none() && row.lease_id.is_none());

        // The abandoned streamed attempt gets a RECOVERABLE Error
        // boundary so replay never attributes its deltas to the
        // re-run's fresh Start.
        let events = stores.event_repo.get_events(&acquired.thread_id).await?;
        let boundary = events
            .iter()
            .find_map(|committed| match &committed.event {
                agent_sdk_foundation::events::AgentEvent::Error {
                    message,
                    recoverable,
                    ..
                } => Some((committed, message.clone(), *recoverable)),
                _ => None,
            })
            .context("requeue must commit an Error boundary event")?;
        assert!(
            boundary.2,
            "the boundary must be recoverable — the task is retrying, not failed",
        );
        assert!(boundary.1.contains("turn-slot collision"));
        // The boundary names the REQUEUED task, so a follower can
        // attribute the retry edge to the run it is following instead of
        // guessing which attempt abandoned the stream.
        assert_eq!(
            boundary.0.event.emitter_task_id(),
            Some(id.as_str()),
            "the requeue boundary must be attributed to the requeued task",
        );
        Ok(())
    }

    /// The requeue inherits the sweep's retry budget: a budget-1 root
    /// that collides has nothing left to spend and takes the ordinary
    /// terminal path instead of looping forever.
    #[tokio::test]
    async fn turn_slot_collision_with_exhausted_budget_fails_terminally() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = agent_sdk_foundation::ThreadId::from_string("t-collision-capped");
        let root = AgentTask::new_root_turn(thread, time::OffsetDateTime::now_utc(), 1);
        let id = root.id.clone();
        stores.task_store.submit_root_turn(root).await?;
        let acquired = stores
            .task_store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-capped"),
                LeaseId::from_string("l-capped"),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("acquire")?;

        commit_occupant_checkpoint(&stores, &acquired.thread_id, 1, "task_foreign_predecessor")
            .await?;

        let collision = anyhow::Error::new(agent_server::journal::commit::StaleTurnCommit {
            expected_turn: 1,
            committed_turns: 1,
        })
        .context("execute fresh root task");
        fail_or_revert_root_task(
            &stores,
            &acquired,
            &collision,
            0,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let row = stores.task_store.get(&id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Failed);
        Ok(())
    }

    /// A collision whose occupant checkpoint belongs to THIS task
    /// means the turn already committed (a stale-lease worker losing
    /// to its own replacement's landed commit).
    /// Requeueing would re-run the input on top of its own committed
    /// turn and duplicate the conversation turn durably — the same-task
    /// case must keep the terminal path.
    #[tokio::test]
    async fn same_task_slot_collision_is_not_requeued() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = agent_sdk_foundation::ThreadId::from_string("t-collision-same-task");
        let root = AgentTask::new_root_turn(thread, time::OffsetDateTime::now_utc(), 3);
        let id = root.id.clone();
        stores.task_store.submit_root_turn(root).await?;
        let acquired = stores
            .task_store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-stale"),
                LeaseId::from_string("l-stale"),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("acquire")?;

        // The occupant is OUR OWN task's committed turn.
        commit_occupant_checkpoint(&stores, &acquired.thread_id, 1, acquired.id.as_str()).await?;

        let collision = anyhow::Error::new(agent_server::journal::commit::StaleTurnCommit {
            expected_turn: 1,
            committed_turns: 1,
        })
        .context("execute fresh root task");
        fail_or_revert_root_task(
            &stores,
            &acquired,
            &collision,
            0,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let row = stores.task_store.get(&id).await?.context("row")?;
        assert_eq!(
            row.status,
            TaskStatus::Failed,
            "a same-task collision must not requeue (it would duplicate the committed turn)",
        );
        Ok(())
    }

    /// Test-only: land a checkpoint at `turn` attributed to `task_id`
    /// so the collision classifier has an occupant to inspect.
    async fn commit_occupant_checkpoint(
        stores: &StoreRegistry,
        thread_id: &agent_sdk_foundation::ThreadId,
        turn: u32,
        task_id: &str,
    ) -> Result<()> {
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        stores
            .checkpoint_store
            .commit_checkpoint(NewCheckpointParams {
                thread_id: thread_id.clone(),
                turn_number: turn,
                task_id: agent_server::journal::task::AgentTaskId::from_string(task_id),
                messages: vec![],
                agent_state_snapshot: serde_json::json!({}),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                kind: CheckpointKind::FullTurn,
                now: time::OffsetDateTime::now_utc(),
            })
            .await?;
        Ok(())
    }

    /// Non-collision errors keep the ordinary terminal path — the
    /// requeue arm must not swallow genuine failures.
    #[tokio::test]
    async fn ordinary_execution_error_still_fails_the_root() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        let thread = agent_sdk_foundation::ThreadId::from_string("t-ordinary-failure");
        let root = AgentTask::new_root_turn(thread, time::OffsetDateTime::now_utc(), 3);
        let id = root.id.clone();
        stores.task_store.submit_root_turn(root).await?;
        let acquired = stores
            .task_store
            .try_acquire_task(
                &id,
                WorkerId::from_string("w-ordinary"),
                LeaseId::from_string("l-ordinary"),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("acquire")?;

        let error = anyhow::anyhow!("provider exploded").context("execute fresh root task");
        fail_or_revert_root_task(
            &stores,
            &acquired,
            &error,
            0,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let row = stores.task_store.get(&id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Failed);
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
        use agent_sdk_foundation::ThreadId;
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
        use agent_sdk_foundation::ThreadId;
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

    #[tokio::test(start_paused = true)]
    async fn heartbeat_loop_extends_lease_until_cancelled() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::task::AgentTask;

        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        // Submit and acquire a root turn so we have a Running row to
        // heartbeat against. The lease is set short on purpose — we
        // want the heartbeat to push the expiry forward each tick.
        let thread = ThreadId::from_string("t-heartbeat-extend");
        let task = AgentTask::new_root_turn(thread.clone(), time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let now = time::OffsetDateTime::now_utc();
        let worker = WorkerId::from_string("w-heartbeat");
        let lease = LeaseId::new();
        let initial_expiry = now + time::Duration::seconds(5);
        stores
            .task_store
            .try_acquire_task(&task_id, worker.clone(), lease.clone(), initial_expiry, now)
            .await?;

        let cancel = CancellationToken::new();
        let handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
            stores: stores.clone(),
            task_id: task_id.clone(),
            thread_id: thread,
            worker_id: worker,
            lease_id: lease,
            lease_duration: time::Duration::seconds(30),
            heartbeat_interval: std::time::Duration::from_secs(1),
            cancel: cancel.clone(),
            task_cancel: CancellationToken::new(),
            deadline: SubagentDeadlineState::Exempt,
            activity: ActivityBeacon::new(),
        }));

        // Auto-advance the synthetic clock past two heartbeat ticks.
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let observed = stores
            .task_store
            .get(&task_id)
            .await?
            .context("task exists")?;
        let observed_expiry = observed.lease_expires_at.context("task is leased")?;
        assert!(
            observed_expiry > initial_expiry,
            "heartbeat should have pushed lease_expires_at forward (initial={initial_expiry}, observed={observed_expiry})",
        );

        cancel.cancel();
        handle.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn heartbeat_loop_exits_when_lease_is_lost() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::task::AgentTask;

        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry(), sample_runtime()?)?;
        let stores = host.stores().clone();

        // Acquire with a real lease, then update the row to a different
        // lease behind our back. The heartbeat CAS will reject and the
        // loop must terminate on its own without us cancelling it.
        let thread = ThreadId::from_string("t-heartbeat-lost");
        let task = AgentTask::new_root_turn(thread.clone(), time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let now = time::OffsetDateTime::now_utc();
        let worker = WorkerId::from_string("w-orig");
        let real_lease = LeaseId::new();
        stores
            .task_store
            .try_acquire_task(
                &task_id,
                worker.clone(),
                real_lease,
                now + time::Duration::seconds(10),
                now,
            )
            .await?;

        // Spawn the heartbeat with a stale lease — its first CAS will fail.
        let stale_lease = LeaseId::new();
        let cancel = CancellationToken::new();
        let task_cancel = CancellationToken::new();
        let handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
            stores: stores.clone(),
            task_id,
            thread_id: thread,
            worker_id: worker,
            lease_id: stale_lease,
            lease_duration: time::Duration::seconds(30),
            heartbeat_interval: std::time::Duration::from_millis(100),
            cancel: cancel.clone(),
            task_cancel: task_cancel.clone(),
            deadline: SubagentDeadlineState::Exempt,
            activity: ActivityBeacon::new(),
        }));

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // The loop should have exited by itself; we never call cancel.
        assert!(
            handle.is_finished(),
            "heartbeat must exit on lease mismatch"
        );
        handle.await?;

        // On a terminal lease rejection the loop trips the per-task token
        // so a still-running root worker aborts its stream and enters the
        // partial-commit-on-cancel path (seam B).
        assert!(
            task_cancel.is_cancelled(),
            "lease-lost heartbeat must cancel the per-task token",
        );
        Ok(())
    }

    #[test]
    fn heartbeat_lease_rejection_is_terminal() {
        // The canonical CAS-rejection marker every backend stamps.
        let terminal = anyhow::anyhow!("heartbeat rejected: lease mismatch on task task_1");
        assert!(heartbeat_error_is_terminal(&terminal));

        // The in-memory backend wraps a typed error with `.context`.
        let wrapped = anyhow::anyhow!("worker mismatch")
            .context("heartbeat rejected")
            .context("heartbeating task");
        assert!(heartbeat_error_is_terminal(&wrapped));
    }

    #[test]
    fn heartbeat_transient_store_error_is_not_terminal() {
        // A DB blip carries store context, never the CAS-rejection
        // marker, so the loop must keep retrying rather than exit and
        // let the lease expire.
        let transient =
            anyhow::anyhow!("connection reset by peer").context("heartbeat update for task_1");
        assert!(!heartbeat_error_is_terminal(&transient));
    }

    #[test]
    fn validate_config_rejects_zero_heartbeat_interval() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                heartbeat_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        match ServiceHost::validate_config(&config) {
            Ok(()) => panic!("zero heartbeat_interval_secs must fail validation"),
            Err(err) => assert!(err.to_string().contains("heartbeat_interval_secs")),
        }
    }

    #[test]
    fn validate_config_rejects_zero_lease_duration() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                lease_duration_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        match ServiceHost::validate_config(&config) {
            Ok(()) => panic!("zero lease_duration_secs must fail validation"),
            Err(err) => assert!(err.to_string().contains("lease_duration_secs")),
        }
    }

    #[test]
    fn validate_config_rejects_lease_not_exceeding_heartbeat() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                lease_duration_secs: 10,
                heartbeat_interval_secs: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        match ServiceHost::validate_config(&config) {
            Ok(()) => panic!("lease_duration_secs <= heartbeat_interval_secs must fail validation"),
            Err(err) => assert!(err.to_string().contains("must exceed")),
        }
    }

    #[test]
    fn validate_config_accepts_default_worker_settings() -> Result<()> {
        ServiceHost::validate_config(&ServiceConfig::default())
    }

    #[tokio::test]
    async fn committed_events_from_reads_bounded_tail() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::event_repository::{EventRepository, InMemoryEventRepository};

        let repo = InMemoryEventRepository::new();
        let thread = agent_sdk_foundation::ThreadId::from_string("t-committed-from");
        let now = time::OffsetDateTime::UNIX_EPOCH;
        for i in 0..5u32 {
            repo.commit_event(&thread, AgentEvent::text(format!("msg_{i}"), "hi"), now)
                .await?;
        }
        // Events occupy sequences 0..=4.
        let tail = committed_events_from(&repo, &thread, 3).await?;
        assert_eq!(
            tail.iter().map(|event| event.sequence).collect::<Vec<_>>(),
            vec![3, 4],
        );

        // start_sequence == 0 returns the whole journal.
        assert_eq!(committed_events_from(&repo, &thread, 0).await?.len(), 5);

        // A start beyond the head returns nothing (no full-journal scan).
        assert!(committed_events_from(&repo, &thread, 99).await?.is_empty());

        // Unknown thread returns empty.
        let other = agent_sdk_foundation::ThreadId::from_string("t-committed-none");
        assert!(committed_events_from(&repo, &other, 0).await?.is_empty());
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
        use agent_sdk_foundation::ThreadId;
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
        use agent_sdk_foundation::ThreadId;
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
        use agent_sdk_foundation::ThreadId;
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
        use agent_sdk_foundation::ThreadId;
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
        use agent_sdk_foundation::ThreadId;
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
                backlog_threshold: None,
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
                backlog_threshold: None,
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
        use agent_sdk_foundation::ThreadId;
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
                backlog_threshold: None,
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
        use agent_sdk_foundation::ThreadId;
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
                backlog_threshold: None,
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

    // ── Fix 1: task-wakeup wiring (batch spawn + parent resume) ─────
    //
    // These tests exercise the in-process nudges wired by
    // `ServiceHost::run` → `ExecutionRuntime` → the worker paths:
    //   * `apply_batch_routing` fires `wake_all_now()` once a tool-child
    //     batch is durably runnable, and
    //   * `execute_tool_task` fires `notify_workers()` when the last
    //     child flips the parent to `Pending`.
    // A scripted provider drives one 2-tool batch then a text close; a
    // recording tool executor timestamps each child so the batch's
    // "start together" property is observable.

    struct ScriptedBatchProvider {
        responses: std::sync::Mutex<std::collections::VecDeque<ChatResponse>>,
    }

    impl ScriptedBatchProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses.into()),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedBatchProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            let response = {
                let mut queue = self
                    .responses
                    .lock()
                    .map_err(|_| anyhow::anyhow!("scripted responses lock poisoned"))?;
                queue
                    .pop_front()
                    .context("scripted provider ran out of responses")?
            };
            Ok(ChatOutcome::Success(response))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    struct RecordingToolExecutor {
        starts: Arc<std::sync::Mutex<Vec<std::time::Instant>>>,
    }

    impl RecordingToolExecutor {
        fn new(starts: Arc<std::sync::Mutex<Vec<std::time::Instant>>>) -> Self {
            Self { starts }
        }
    }

    #[async_trait]
    impl crate::runtime::ToolCallExecutor for RecordingToolExecutor {
        async fn execute_tool_call(
            &self,
            _bootstrap: &agent_server::worker::ToolTaskBootstrap,
            _collector: agent_server::worker::ToolEventCollector,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<agent_sdk_foundation::ToolResult> {
            if let Ok(mut starts) = self.starts.lock() {
                starts.push(std::time::Instant::now());
            }
            Ok(agent_sdk_foundation::ToolResult::success("probe ok"))
        }
    }

    fn probe_tool() -> agent_sdk_foundation::llm::Tool {
        agent_sdk_foundation::llm::Tool {
            name: "probe".into(),
            description: "A fast read-only probe".into(),
            input_schema: serde_json::json!({ "type": "object" }),
            display_name: "Probe".into(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }
    }

    fn probe_definition() -> AgentDefinition {
        let mut definition = sample_definition();
        definition.tools = vec![probe_tool()];
        definition
    }

    fn text_response(id: &str, text: &str) -> ChatResponse {
        ChatResponse {
            id: id.into(),
            content: vec![ContentBlock::Text { text: text.into() }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }
    }

    fn tool_use_batch_response(id: &str, calls: &[(&str, &str)]) -> ChatResponse {
        let content = calls
            .iter()
            .map(|(call_id, name)| ContentBlock::ToolUse {
                id: (*call_id).into(),
                name: (*name).into(),
                input: serde_json::json!({}),
                thought_signature: None,
            })
            .collect();
        ChatResponse {
            id: id.into(),
            content,
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 12,
                output_tokens: 6,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }
    }

    fn tool_runtime(
        provider: Arc<dyn LlmProvider>,
        executor: Arc<dyn crate::runtime::ToolCallExecutor>,
    ) -> Result<Arc<ExecutionRuntime>> {
        let resolver = Arc::new(StaticProviderResolver::new());
        resolver.set_fallback(provider)?;
        Ok(Arc::new(ExecutionRuntime::new(
            resolver,
            executor,
            Arc::new(AllowAllConfirmationPolicy),
        )))
    }

    #[tokio::test]
    async fn wakeup_nudges_drive_batch_and_resume_without_polling() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use std::time::{Duration, Instant};

        // A 30 s acquisition ticker and NO fallback sweep: after the
        // initial root pickup, the only thing that can move work across a
        // task hop within the assertion window is the in-process wakeup
        // nudge wired by Fix 1. If either nudge failed to fire — the
        // batch spawn's `wake_all_now` (child pickup) or the last child's
        // `notify_workers` (parent resume) — the flow would stall until
        // the 30 s ticker, far outside the 5 s window below.
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 3,
                acquisition_interval_secs: 30,
                ..Default::default()
            },
            ..Default::default()
        };

        let starts = Arc::new(std::sync::Mutex::new(Vec::<Instant>::new()));
        let provider = Arc::new(ScriptedBatchProvider::new(vec![
            tool_use_batch_response("resp_batch", &[("call_a", "probe"), ("call_b", "probe")]),
            text_response("resp_final", "all probes done"),
        ]));
        let executor = Arc::new(RecordingToolExecutor::new(Arc::clone(&starts)));
        let runtime = tool_runtime(provider, executor)?;
        let runtime_kick = Arc::clone(&runtime);
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(probe_definition()));

        let host = ServiceHost::new(config, registry, runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        let thread = ThreadId::from_string("t-wakeup-batch");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "probe twice".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let root_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Wait for `run()` to install the signal on the shared runtime,
        // then fire ONE nudge to kick the initial root pickup — standing
        // in for the production 1 s ticker / submit backstop, which is
        // not what this test exercises. Every hop AFTER this kick must be
        // driven purely by the Fix 1 nudges under test.
        let mut kicked = false;
        for _ in 0..200 {
            if let Some(signal) = runtime_kick.wakeup_signal() {
                signal.notify_workers();
                kicked = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            kicked,
            "run() must install the wakeup signal on the shared runtime",
        );

        let mut completed = false;
        for _ in 0..250 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&root_id)
                .await?
                .context("root task must still exist")?;
            if row.status == TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "root turn should complete within 5 s via wakeup nudges (30 s ticker, no fallback sweep)",
        );

        // Both batch children were dispatched together, not staggered
        // across the 30 s ticker: their execution start timestamps land
        // within a small window of each other. Snapshot out of the lock
        // so no guard is held across the shutdown await below.
        let recorded: Vec<Instant> = {
            let starts = starts
                .lock()
                .map_err(|_| anyhow::anyhow!("starts lock poisoned"))?;
            starts.clone()
        };
        assert_eq!(
            recorded.len(),
            2,
            "both probe children should have executed",
        );
        let (first, second) = if recorded[0] <= recorded[1] {
            (recorded[0], recorded[1])
        } else {
            (recorded[1], recorded[0])
        };
        let delta = second.duration_since(first);
        assert!(
            delta < Duration::from_secs(2),
            "batch children started {delta:?} apart — expected them to start together, \
             not staggered across the acquisition ticker",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn tool_flow_completes_on_acquisition_ticker_backstop() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use std::time::Duration;

        // No manual kick and no fallback sweep: every task hop here is
        // eligible to be driven solely by the 1 s acquisition ticker.
        // This proves the poll backstop still carries a full tool-call
        // turn (root → batch children → resume) to completion — the
        // wakeup nudges are a latency optimisation layered on top, never
        // a correctness requirement.
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 3,
                acquisition_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let starts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedBatchProvider::new(vec![
            tool_use_batch_response("resp_batch", &[("call_a", "probe"), ("call_b", "probe")]),
            text_response("resp_final", "done"),
        ]));
        let executor = Arc::new(RecordingToolExecutor::new(Arc::clone(&starts)));
        let runtime = tool_runtime(provider, executor)?;
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(probe_definition()));

        let host = ServiceHost::new(config, registry, runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        let thread = ThreadId::from_string("t-ticker-backstop");
        let task = AgentTask::new_root_turn_with_input(
            thread,
            vec![SubmittedInputItem::Text {
                text: "probe twice".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let root_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        let mut completed = false;
        for _ in 0..300 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(&root_id)
                .await?
                .context("root task must still exist")?;
            if row.status == TaskStatus::Completed {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "tool-call turn must complete on the acquisition-ticker backstop even without wakeup nudges",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    // ── Subagent timeout enforcement (issue #299) ───────────────────
    //
    // The heartbeat loop enforces a wall-clock deadline (child root
    // `created_at` + invocation `spec.timeout_ms`) on child-thread
    // roots linked to a durable subagent invocation:
    //   * a hung provider stream is aborted and the child fails with
    //     `subagent timed out after {timeout_ms}ms` (never Cancelled),
    //   * a child acquired past its deadline fails up front with zero
    //     LLM dispatch, and
    //   * either way the parent's fan-in resumes with the failed child
    //     outcome visible to the parent LLM.

    const HANG_CHILD_TASK: &str = "hang-child-task: stall the provider stream on purpose";
    const FAST_CHILD_TASK: &str = "fast-child-task: reply instantly";
    const HANG_CHILD_TIMEOUT_MS: u64 = 2_500;

    /// Hand-built effective spec mirroring what `resolve_subagent_spec`
    /// produces, parameterized on task text and timeout.
    fn subagent_timeout_spec(
        task: &str,
        timeout_ms: u64,
    ) -> agent_server::worker::EffectiveSubagentSpec {
        use agent_server::worker::{
            EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
            InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
        };
        use std::collections::{BTreeMap, BTreeSet};

        let capabilities: BTreeSet<String> = BTreeSet::from(["read_file".to_owned()]);
        EffectiveSubagentSpec {
            task: task.to_owned(),
            prompt: String::new(),
            model: "mock-model".into(),
            max_turns: 5,
            timeout_ms,
            depth: 1,
            max_parallel_subagents: 1,
            nickname: None,
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy::default(),
            audit_provenance: None,
            inherited_policy: InheritedSubagentPolicy {
                default_model: "mock-model".into(),
                allowed_models: BTreeSet::from(["mock-model".to_owned()]),
                default_max_turns: 5,
                max_turns: 5,
                default_timeout_ms: timeout_ms,
                max_timeout_ms: timeout_ms,
                capability_profiles: BTreeMap::from([(
                    "research".to_owned(),
                    SubagentCapabilityProfile {
                        capabilities: capabilities.clone(),
                        sandbox: SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: BTreeSet::new(),
                    },
                )]),
                allowed_capabilities: capabilities.clone(),
                max_depth: 3,
                max_parallel_subagents: 1,
                sandbox: SubagentSandboxPolicy::read_only(),
                allowed_mcp_servers: BTreeSet::new(),
                audit_provider: "mock".into(),
            },
            capabilities: EffectiveSubagentCapabilities {
                profile: "research".into(),
                allowed: capabilities,
            },
        }
    }

    fn request_contains_tool_result(request: &ChatRequest) -> bool {
        request
            .messages
            .iter()
            .any(|message| match &message.content {
                agent_sdk_foundation::llm::Content::Blocks(blocks) => blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. })),
                agent_sdk_foundation::llm::Content::Text(_) => false,
            })
    }

    fn request_flat_text(request: &ChatRequest) -> String {
        let mut flat = String::new();
        for message in &request.messages {
            match &message.content {
                agent_sdk_foundation::llm::Content::Text(text) => flat.push_str(text),
                agent_sdk_foundation::llm::Content::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => flat.push_str(text),
                            ContentBlock::ToolResult { content, .. } => flat.push_str(content),
                            _ => {}
                        }
                    }
                }
            }
            flat.push('\n');
        }
        flat
    }

    /// How the scripted provider misbehaves for the hang child.
    #[derive(Clone, Copy, Debug)]
    enum HangChildBehavior {
        /// The child's provider stream stalls forever — exercises the
        /// running-leg enforcement (heartbeat deadline).
        StallProvider,
        /// The child's turn requests a `probe` tool whose executor
        /// hangs — the child root parks in `WaitingOnChildren`,
        /// exercising the parked-leg enforcement (deadline sweep).
        RequestHungTool,
    }

    /// Scripted provider for the timeout tests, routed on request
    /// content:
    ///   * any `tool_result` block → the parent's fan-in resume: the
    ///     flattened result text is recorded (so tests can assert what
    ///     the parent LLM saw) and a text close is returned,
    ///   * the hang-child task text → counts the dispatch, then
    ///     misbehaves per [`HangChildBehavior`],
    ///   * the fast-child task text → an instant text reply,
    ///   * anything else → the parent's fresh turn: a two-call
    ///     subagent tool-use batch.
    struct SubagentScriptProvider {
        hang_child_calls: AtomicUsize,
        resume_tool_results: std::sync::Mutex<Vec<String>>,
        hang_child: HangChildBehavior,
    }

    impl SubagentScriptProvider {
        fn new() -> Self {
            Self {
                hang_child_calls: AtomicUsize::new(0),
                resume_tool_results: std::sync::Mutex::new(Vec::new()),
                hang_child: HangChildBehavior::StallProvider,
            }
        }

        fn with_hung_tool() -> Self {
            Self {
                hang_child: HangChildBehavior::RequestHungTool,
                ..Self::new()
            }
        }

        fn recorded_resume_text(&self) -> Result<String> {
            let recorded = self
                .resume_tool_results
                .lock()
                .map_err(|_| anyhow::anyhow!("resume_tool_results lock poisoned"))?;
            Ok(recorded.join("\n"))
        }
    }

    #[async_trait]
    impl LlmProvider for SubagentScriptProvider {
        async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
            if request_contains_tool_result(&request) {
                let flat = request_flat_text(&request);
                self.resume_tool_results
                    .lock()
                    .map_err(|_| anyhow::anyhow!("resume_tool_results lock poisoned"))?
                    .push(flat);
                return Ok(ChatOutcome::Success(text_response(
                    "resp_parent_final",
                    "children handled",
                )));
            }
            let flat = request_flat_text(&request);
            if flat.contains(HANG_CHILD_TASK) {
                self.hang_child_calls.fetch_add(1, Ordering::SeqCst);
                return match self.hang_child {
                    // Stalled provider stream: never yields. Only the
                    // host's deadline enforcement can unwedge this
                    // child.
                    HangChildBehavior::StallProvider => {
                        std::future::pending::<Result<ChatOutcome>>().await
                    }
                    // Request the probe tool; the executor hangs, so
                    // the child root parks on the tool child forever.
                    HangChildBehavior::RequestHungTool => Ok(ChatOutcome::Success(ChatResponse {
                        id: "resp_child_probe".into(),
                        content: vec![ContentBlock::ToolUse {
                            id: "call_probe".into(),
                            name: "probe".into(),
                            input: serde_json::json!({}),
                            thought_signature: None,
                        }],
                        model: "mock-model".into(),
                        stop_reason: Some(StopReason::ToolUse),
                        usage: Usage {
                            input_tokens: 8,
                            output_tokens: 4,
                            cached_input_tokens: 0,
                            cache_creation_input_tokens: 0,
                        },
                    })),
                };
            }
            if flat.contains(FAST_CHILD_TASK) {
                return Ok(ChatOutcome::Success(text_response(
                    "resp_fast_child",
                    "sibling done",
                )));
            }
            let content = vec![
                ContentBlock::ToolUse {
                    id: "call_hang".into(),
                    name: "subagent_hang".into(),
                    input: serde_json::json!({ "task": HANG_CHILD_TASK }),
                    thought_signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_fast".into(),
                    name: "subagent_fast".into(),
                    input: serde_json::json!({ "task": FAST_CHILD_TASK }),
                    thought_signature: None,
                },
            ];
            Ok(ChatOutcome::Success(ChatResponse {
                id: "resp_parent_spawn".into(),
                content,
                model: "mock-model".into(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 12,
                    output_tokens: 6,
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

    /// Routes `subagent_hang` / `subagent_fast` tool calls into durable
    /// subagent invocations with deterministic child-thread ids (so a
    /// retried attempt reuses the same rows and the test can find the
    /// children without scanning).
    struct DeadlineSpawnSelector;

    #[async_trait]
    impl agent_server::worker::SubagentSpawnSelector for DeadlineSpawnSelector {
        async fn decide(
            &self,
            parent_thread_id: &agent_sdk_foundation::ThreadId,
            tool_calls: &[agent_sdk_foundation::PendingToolCallInfo],
        ) -> Result<Vec<agent_server::worker::SubagentSpawnDecision>> {
            use agent_server::worker::SubagentSpawnDecision;
            use agent_server::worker::subagent_spawn_selector::SubagentSpawnPlan;
            use agent_server::worker::{SubagentCapabilityRequest, SubagentSpawnRequest};

            Ok(tool_calls
                .iter()
                .map(|call| {
                    let (task, timeout_ms, suffix) = match call.name.as_str() {
                        "subagent_hang" => (HANG_CHILD_TASK, HANG_CHILD_TIMEOUT_MS, "hang"),
                        "subagent_fast" => (FAST_CHILD_TASK, 600_000, "fast"),
                        _ => return SubagentSpawnDecision::SpawnAsTool,
                    };
                    SubagentSpawnDecision::SpawnAsSubagent {
                        plan: Box::new(SubagentSpawnPlan {
                            request: SubagentSpawnRequest::new(
                                task,
                                SubagentCapabilityRequest::new("research"),
                            ),
                            spec: subagent_timeout_spec(task, timeout_ms),
                            child_thread_id: agent_sdk_foundation::ThreadId::from_string(format!(
                                "{parent_thread_id}-{suffix}"
                            )),
                            child_root_input: Vec::new(),
                            child_caller_metadata: None,
                        }),
                    }
                })
                .collect())
        }
    }

    fn subagent_timeout_runtime(
        provider: Arc<SubagentScriptProvider>,
    ) -> Result<Arc<ExecutionRuntime>> {
        subagent_timeout_runtime_with_executor(provider, Arc::new(NoopToolExecutor))
    }

    fn subagent_timeout_runtime_with_executor(
        provider: Arc<SubagentScriptProvider>,
        executor: Arc<dyn crate::runtime::ToolCallExecutor>,
    ) -> Result<Arc<ExecutionRuntime>> {
        let resolver = Arc::new(StaticProviderResolver::new());
        resolver.set_fallback(provider)?;
        Ok(Arc::new(
            ExecutionRuntime::new(resolver, executor, Arc::new(AllowAllConfirmationPolicy))
                .with_subagent_spawn_selector(Arc::new(DeadlineSpawnSelector)),
        ))
    }

    async fn wait_for_status(
        stores: &StoreRegistry,
        task_id: &agent_server::journal::task::AgentTaskId,
        want: TaskStatus,
        max_polls: usize,
    ) -> Result<AgentTask> {
        let mut last_seen: Option<TaskStatus> = None;
        for _ in 0..max_polls {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let row = stores
                .task_store
                .get(task_id)
                .await?
                .context("polled task must still exist")?;
            if row.status == want {
                return Ok(row);
            }
            last_seen = Some(row.status);
        }
        bail!("task {task_id} never reached {want:?}; last observed {last_seen:?}");
    }

    /// Find the single root-turn task of `thread_id`.
    async fn root_task_of_thread(
        stores: &StoreRegistry,
        thread_id: &agent_sdk_foundation::ThreadId,
    ) -> Result<AgentTask> {
        let tasks = stores.task_store.list_by_thread(thread_id).await?;
        tasks
            .into_iter()
            .find(|task| task.kind == TaskKind::RootTurn)
            .with_context(|| format!("thread {thread_id} has no root turn"))
    }

    #[tokio::test]
    async fn timed_out_subagent_child_fails_and_parent_resumes() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 4,
                heartbeat_interval_secs: 1,
                acquisition_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(Arc::clone(&provider))?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        let parent_thread = ThreadId::from_string("t-subagent-timeout");
        let parent = AgentTask::new_root_turn_with_input(
            parent_thread.clone(),
            vec![SubmittedInputItem::Text {
                text: "coordinate the helpers".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let parent_id = parent.id.clone();
        stores.task_store.submit_root_turn(parent).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // The hang child wedges its provider stream forever, so the
        // ONLY way the parent completes is the heartbeat deadline
        // failing that child and waking the fan-in with a failed
        // outcome. 30 s budget for a ~3 s happy path.
        wait_for_status(&stores, &parent_id, TaskStatus::Completed, 1_500).await?;

        // The hang child: FAILED (not Cancelled) with the exact
        // timeout message, after actually dispatching the LLM call.
        let hang_thread = ThreadId::from_string("t-subagent-timeout-hang");
        let hang_root = root_task_of_thread(&stores, &hang_thread).await?;
        assert_eq!(
            hang_root.status,
            TaskStatus::Failed,
            "a timed-out child must fail, not cancel",
        );
        let hang_error = hang_root.last_error.clone().unwrap_or_default();
        assert!(
            hang_error.contains("subagent timed out after 2500ms"),
            "timed-out child must carry the timeout message, got {hang_error:?}",
        );
        assert!(
            provider.hang_child_calls.load(Ordering::SeqCst) >= 1,
            "the hang child must have dispatched its (stalled) LLM call — this exercises \
             the mid-flight heartbeat path, not the acquisition-expiry path",
        );

        // Seam B salvage: aborting the in-flight turn commits the
        // provider-valid prefix (here: the child's user prompt) even
        // though the row went FAILED, so the transcript survives.
        let salvaged = stores
            .message_store
            .get_history(&hang_thread)
            .await?
            .iter()
            .any(|message| {
                message
                    .content
                    .first_text()
                    .is_some_and(|text| text.contains(HANG_CHILD_TASK))
            });
        assert!(
            salvaged,
            "seam B must salvage the timed-out child's committed prefix into its thread history",
        );

        // The generously-budgeted sibling ran to completion unaffected.
        let fast_thread = ThreadId::from_string("t-subagent-timeout-fast");
        let fast_root = root_task_of_thread(&stores, &fast_thread).await?;
        assert_eq!(
            fast_root.status,
            TaskStatus::Completed,
            "the sibling with an unelapsed timeout must complete normally",
        );

        // Both invocations reached Completed (the timed-out child is a
        // materialized failed RESULT, not a failed invocation), and the
        // parent LLM saw the timeout message in its fan-in results.
        let invocations = stores.task_store.list_children(&parent_id).await?;
        assert_eq!(invocations.len(), 2, "one invocation per spawned child");
        for invocation in &invocations {
            assert_eq!(
                invocation.status,
                TaskStatus::Completed,
                "invocation {} must complete with a materialized result",
                invocation.id,
            );
        }
        let resume_text = provider.recorded_resume_text()?;
        assert!(
            resume_text.contains("subagent timed out after 2500ms"),
            "the parent's resume request must carry the failed child's timeout message, got \
             {resume_text:?}",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    /// A minimal suspension payload for `thread` carrying one pending
    /// Confirm-tier `tool_name` call, so downstream machinery
    /// (invocation bootstrap, parent resume) sees a well-formed
    /// continuation.
    fn pending_call_suspension(
        thread: &agent_sdk_foundation::ThreadId,
        tool_name: &str,
    ) -> agent_server::journal::task::SuspensionPayload {
        agent_server::journal::task::SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread.clone(),
                    turn: 1,
                    total_usage: agent_sdk_foundation::TokenUsage::default(),
                    turn_usage: agent_sdk_foundation::TokenUsage::default(),
                    pending_tool_calls: vec![agent_sdk_foundation::PendingToolCallInfo {
                        id: format!("call_{tool_name}"),
                        name: tool_name.to_owned(),
                        display_name: tool_name.to_owned(),
                        tier: ToolTier::Confirm,
                        input: serde_json::json!({ "task": HANG_CHILD_TASK }),
                        effective_input: serde_json::json!({ "task": HANG_CHILD_TASK }),
                        listen_context: None,
                    }],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread.clone()),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        }
    }

    /// Manually persist a parent → invocation → child-root fixture at
    /// timestamp `at`, with `timeout_ms` on the invocation spec and a
    /// well-formed pending `subagent_hang` tool call on the parent's
    /// suspension (so the invocation can materialize its result and
    /// resume the parent later). Returns `(parent, invocation,
    /// child_root)`.
    async fn persist_subagent_fixture(
        stores: &StoreRegistry,
        parent_thread: &agent_sdk_foundation::ThreadId,
        timeout_ms: u64,
        at: time::OffsetDateTime,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        use agent_server::journal::SubagentInvocationSpawn;

        stores.thread_store.get_or_create(parent_thread, at).await?;
        let parent = AgentTask::new_root_turn_with_input(
            parent_thread.clone(),
            vec![SubmittedInputItem::Text {
                text: "coordinate the helpers".into(),
            }],
            at,
            3,
        );
        let parent_id = parent.id.clone();
        stores.task_store.submit_root_turn(parent).await?;
        let worker = WorkerId::from_string("w-fixture");
        let lease = LeaseId::new();
        stores
            .task_store
            .try_acquire_task(
                &parent_id,
                worker.clone(),
                lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                at,
            )
            .await?
            .context("fixture parent must acquire before spawning")?;

        let child_thread =
            agent_sdk_foundation::ThreadId::from_string(format!("{parent_thread}-hang"));
        stores.thread_store.get_or_create(&child_thread, at).await?;

        let payload = pending_call_suspension(parent_thread, "subagent_hang");

        let (parent, invocation, child_root) = stores
            .task_store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id: child_thread,
                    spec: subagent_timeout_spec(HANG_CHILD_TASK, timeout_ms),
                    child_root_input: vec![SubmittedInputItem::Text {
                        text: HANG_CHILD_TASK.into(),
                    }],
                    spawn_index: 0,
                    child_caller_metadata: None,
                    payload,
                },
                at,
            )
            .await
            .context("persist fixture subagent invocation")?;
        Ok((parent, invocation, child_root))
    }

    /// Edge 2 (post-outage re-acquisition must not insta-reap).
    ///
    /// A child spawned long ago, then stranded by a crash/outage far longer
    /// than its budget, carries a stale (or absent) `last_activity_at`. Under
    /// the old since-spawn leg it was failed the instant a worker picked it
    /// up — before executing a single instruction — which is exactly the
    /// age/queue-wait punishment the founder's ruling forbids. Acquisition
    /// now stamps `last_activity_at` (`mark_running`), so the re-acquired
    /// child is judged healthy and gets a full fresh budget to prove itself.
    ///
    /// The companion property — a child that never gets acquired is still
    /// reaped off its `created_at` floor — is asserted first, so this test
    /// pins both halves: acquisition rescues, non-acquisition does not.
    #[tokio::test]
    async fn reacquired_aged_child_is_not_reaped_before_dispatch() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        // Spawned 5 minutes ago under a 250ms budget: hundreds of budgets
        // past its `created_at` floor before anyone touches it.
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
        let parent_thread = ThreadId::from_string("t-reacquire-aged");
        let (_parent, _invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 250, spawned_at).await?;
        let SubagentDeadlineState::Enforced(deadline) =
            resolve_subagent_deadline(&stores, &child_root).await
        else {
            bail!("linked child root must resolve an enforced deadline");
        };

        // Before acquisition: never ran, no `last_activity_at`, no events —
        // the parked sweep would (correctly) reap it off its floor.
        let now = time::OffsetDateTime::now_utc();
        assert!(
            child_root.last_activity_at.is_none(),
            "a never-acquired child has no activity stamp yet",
        );
        assert!(
            stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "a child that never started must still be reaped off its created_at floor",
        );

        // Acquire it (the post-outage requeue → re-acquire path).
        let acquired = stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-reacquire"),
                LeaseId::new(),
                now + time::Duration::seconds(30),
                now,
            )
            .await?
            .context("aged child must be acquirable")?;

        // Acquisition stamped a fresh activity instant, so the very same
        // predicate now spares it — no insta-reap, a whole budget to work.
        assert_eq!(
            acquired.last_activity_at,
            Some(now),
            "mark_running must stamp last_activity_at at acquisition",
        );
        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "a freshly re-acquired child must not be reaped before it runs, however old it is",
        );
        Ok(())
    }

    /// The founder-facing property: a child that keeps working is never
    /// timed out, no matter how old it gets.
    ///
    /// `timeout_ms` is a budget of silence, not of work. A worker that
    /// has been running for ten hours but committed a frame a moment ago
    /// is healthy — killing it would throw away ten hours of work, which
    /// is exactly what the old since-spawn deadline did.
    #[tokio::test]
    async fn active_child_survives_arbitrarily_far_past_its_stall_budget() -> Result<()> {
        use agent_sdk_foundation::{AgentEvent, ThreadId};

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let spawned_at = time::OffsetDateTime::now_utc();
        let budget_ms = 30 * 60 * 1_000; // 30 minutes, bip's default posture.
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-stall-active"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let SubagentDeadlineState::Enforced(deadline) =
            resolve_subagent_deadline(&stores, &child_root).await
        else {
            bail!("linked child root must resolve an enforced deadline");
        };

        // Ten hours in — twenty budgets past the old since-spawn deadline —
        // but the child committed a frame one minute ago.
        let now = spawned_at + time::Duration::hours(10);
        stores
            .event_repo
            .commit_event(
                &child_root.thread_id,
                AgentEvent::text("t1", "still working"),
                now - time::Duration::minutes(1),
            )
            .await?;

        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, now,).await,
            "a child that committed work inside its budget must never time out, \
             however long it has been alive",
        );
        Ok(())
    }

    /// The other half: silence still ends a child — one budget after its
    /// last sign of life, not one budget after spawn.
    #[tokio::test]
    async fn silent_child_expires_one_budget_after_its_last_frame() -> Result<()> {
        use agent_sdk_foundation::{AgentEvent, ThreadId};

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let spawned_at = time::OffsetDateTime::now_utc();
        let budget = time::Duration::minutes(30);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-stall-silent"),
            30 * 60 * 1_000,
            spawned_at,
        )
        .await?;
        let SubagentDeadlineState::Enforced(deadline) =
            resolve_subagent_deadline(&stores, &child_root).await
        else {
            bail!("linked child root must resolve an enforced deadline");
        };

        // The child worked for two hours, then went silent.
        let last_frame_at = spawned_at + time::Duration::hours(2);
        stores
            .event_repo
            .commit_event(
                &child_root.thread_id,
                AgentEvent::text("t1", "last thing I did"),
                last_frame_at,
            )
            .await?;

        // One second short of a full budget of silence: still alive. Note
        // this instant is already FOUR budgets past spawn — under the old
        // since-spawn rule the child would have been dead long ago.
        let still_alive_at = last_frame_at + budget - time::Duration::seconds(1);
        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, still_alive_at,).await,
            "expiry must be measured from the last frame, not from spawn",
        );

        // A full budget of silence after that last frame: reaped.
        let expired_at = last_frame_at + budget + time::Duration::seconds(1);
        assert!(
            stall_expired(&stores, &child_root.id, deadline, None, expired_at,).await,
            "a child silent for its whole budget must still be timed out",
        );
        Ok(())
    }

    /// A child that never produces anything is still reaped: spawn is the
    /// initial evidence of life, so the budget runs from `created_at` when
    /// no frame ever lands (wedged admission, hung before the first token).
    #[tokio::test]
    async fn child_that_never_commits_anything_still_times_out() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let spawned_at = time::OffsetDateTime::now_utc();
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-stall-never-starts"),
            1_000,
            spawned_at,
        )
        .await?;
        let SubagentDeadlineState::Enforced(deadline) =
            resolve_subagent_deadline(&stores, &child_root).await
        else {
            bail!("linked child root must resolve an enforced deadline");
        };

        assert!(
            !stall_expired(
                &stores,
                &child_root.id,
                deadline,
                None,
                spawned_at + time::Duration::milliseconds(500),
            )
            .await,
            "a child younger than its own budget can never be stalled out",
        );
        assert!(
            stall_expired(
                &stores,
                &child_root.id,
                deadline,
                None,
                spawned_at + time::Duration::milliseconds(1_500),
            )
            .await,
            "a child that never committed anything must still time out",
        );
        Ok(())
    }

    /// F1: a child parked on ONE long tool call (the founder's 40-minute
    /// build) commits nothing on its own thread until the tool returns —
    /// its live progress lands on the TOOL task's row via that task's own
    /// heartbeat. The parked child root keeps no heartbeat of its own, so
    /// its only evidence is the descendant's durable activity, reached by
    /// the subtree walk. #376's journal-only probe reaped it; the subtree
    /// max must keep it alive.
    #[tokio::test]
    async fn long_running_tool_child_keeps_its_parked_parent_alive() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-long-tool"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        // Its own thread committed nothing and its own row carries no
        // activity stamp: on the journal-only view it looks silent.
        let now = time::OffsetDateTime::now_utc();
        assert!(
            stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "sanity: with no descendant activity the parked child reads as silent",
        );

        // Its tool child reported progress a minute ago — its own heartbeat
        // persisted that onto the tool task's row.
        let mut tool =
            AgentTask::new_child(&child_root, TaskKind::ToolRuntime, child_root.created_at, 3)?;
        tool.last_activity_at = Some(now - time::Duration::minutes(1));
        stores.task_store.insert(tool).await?;

        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "a parked child whose tool descendant reported progress inside the budget must \
             survive, even though the child's own thread committed nothing",
        );
        Ok(())
    }

    /// F2: a child running a NESTED subagent commits nothing on its own
    /// thread — the nested work commits on the nested child root's OWN
    /// thread. #376's probe saw only the outer thread's silence and killed
    /// a productive subtree. The subtree walk must hop the invocation
    /// linkage to the nested thread and find its activity.
    #[tokio::test]
    async fn active_nested_subagent_keeps_the_outer_child_alive() -> Result<()> {
        use agent_sdk_foundation::{AgentEvent, ThreadId};
        use agent_server::journal::task_state::SubagentInvocationState;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, invocation, outer) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-nested-outer"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(outer.created_at, budget_ms, 0).context("deadline")?;

        // Build a nested subagent under the outer child root: a fresh child
        // root on its own thread, and an invocation task linking to it
        // (reusing the fixture invocation's resolved spec).
        let nested_thread = ThreadId::from_string("t-nested-inner");
        stores
            .thread_store
            .get_or_create(&nested_thread, spawned_at)
            .await?;
        let nested_root = AgentTask::new_root_turn(nested_thread.clone(), spawned_at, 3);
        let spec = invocation
            .state
            .subagent_invocation()
            .context("fixture invocation carries linkage")?
            .spec
            .clone();
        let nested_invocation = AgentTask::new_subagent_invocation(
            &outer,
            SubagentInvocationState {
                spec,
                child_thread_id: nested_thread.clone(),
                child_root_task_id: nested_root.id.clone(),
            },
            0,
            spawned_at,
            3,
        )?;
        stores.task_store.insert(nested_root).await?;
        stores.task_store.insert(nested_invocation).await?;

        let now = time::OffsetDateTime::now_utc();

        // With the nested subtree silent, the outer child reads as stalled.
        assert!(
            stall_expired(&stores, &outer.id, deadline, None, now).await,
            "sanity: a fully silent nested subtree leaves the outer child stalled",
        );

        // The nested child commits a frame on its OWN thread inside the
        // budget — the outer child's thread stays silent throughout.
        stores
            .event_repo
            .commit_event(
                &nested_thread,
                AgentEvent::text("n1", "nested still working"),
                now - time::Duration::minutes(1),
            )
            .await?;

        assert!(
            !stall_expired(&stores, &outer.id, deadline, None, now).await,
            "an outer child whose nested subagent is still committing must survive, though its \
             own thread is silent",
        );
        Ok(())
    }

    /// F3: a pure tool-call provider stream (`ToolUseStart` /
    /// `ToolInputDelta` / signature deltas) journals nothing while it
    /// actively yields, so the
    /// journal-only probe reads an actively-streaming turn as silent. The
    /// provider-frame refresh point bumps the live beacon on every frame;
    /// the running leg reads that beacon directly, so the child survives on
    /// the strength of the stream alone.
    #[tokio::test]
    async fn pure_tool_call_stream_keeps_a_running_child_alive() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-pure-toolcall"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        let now = time::OffsetDateTime::now_utc();

        // A live beacon bumped a moment ago by an arriving frame keeps the
        // running child alive even though nothing was journalled.
        assert!(
            !stall_expired(
                &stores,
                &child_root.id,
                deadline,
                Some(now - time::Duration::minutes(1)),
                now,
            )
            .await,
            "a running child whose live beacon shows a recent frame must survive with an empty \
             journal — a pure tool-call stream produces exactly this",
        );

        // The same child, viewed only through its (empty) journal — the
        // pre-fix predicate — reads as silent and would be killed.
        assert!(
            stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "sanity: without the live beacon the empty journal looks like silence",
        );
        Ok(())
    }

    /// F4: event retention (`event_ttl_secs < timeout_ms`) can purge an
    /// event that fell inside the stall window, so a busy child's journal
    /// query returns `None` and #376 would false-kill it. `last_activity_at`
    /// lives on the task row, which retention never touches, so the durable
    /// stamp answers even when every event is gone.
    #[tokio::test]
    async fn durable_activity_survives_event_retention_purge() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-retention"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        // Acquisition stamps a fresh `last_activity_at` on the row; then
        // imagine retention has since purged every event this child ever
        // committed, so the journal query would return `None`.
        let acquired_at = time::OffsetDateTime::now_utc();
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-retention"),
                LeaseId::new(),
                acquired_at + time::Duration::seconds(30),
                acquired_at,
            )
            .await?
            .context("child must be acquirable")?;

        // No events exist on the thread (all purged), yet the durable stamp
        // is inside the budget — the child must not be killed.
        let now = acquired_at + time::Duration::minutes(1);
        assert!(
            stores
                .event_repo
                .min_sequence_at_or_after(&child_root.thread_id, now - time::Duration::minutes(30))
                .await?
                .is_none(),
            "the journal is empty (events purged) — the pre-fix predicate would false-kill here",
        );
        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "the durable last_activity_at, which retention cannot purge, must keep the child alive",
        );
        Ok(())
    }

    /// r2 finding 4: a tool child that just completed must reset its parked
    /// parent's stall budget. The completion instant is stamped on the
    /// terminal transition itself (the row is no longer heartbeatable), so
    /// the parent's subtree probe reads the terminal child as a fresh sign
    /// of life even though the parent's own thread has been silent.
    #[tokio::test]
    async fn just_completed_tool_child_keeps_its_parked_parent_alive() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-tool-complete"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;
        let now = time::OffsetDateTime::now_utc();

        // Build a tool child that has just completed. `complete` stamps the
        // completion instant onto `last_activity_at` as part of the terminal
        // transition.
        let tool = AgentTask::new_child(&child_root, TaskKind::ToolRuntime, spawned_at, 3)?
            .mark_running(
                WorkerId::from_string("w-tool"),
                LeaseId::new(),
                now,
                spawned_at,
            )
            .map_err(|err| anyhow::anyhow!("mark_running: {err}"))?
            .complete_with_result(
                serde_json::json!({"ok": true}),
                now - time::Duration::minutes(1),
            )
            .map_err(|err| anyhow::anyhow!("complete: {err}"))?;
        assert_eq!(
            tool.last_activity_at,
            Some(now - time::Duration::minutes(1)),
            "completion must stamp last_activity_at on the terminal row",
        );
        assert!(tool.status.is_terminal());
        stores.task_store.insert(tool).await?;

        assert!(
            !stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "a parent whose tool child completed inside the budget must survive, though the \
             parent's own thread is silent",
        );
        Ok(())
    }

    /// r2 finding 3: a wedged child buried under a large history of retained
    /// terminal rows must still time out — the probe must not perpetually
    /// truncate and treat it as active. Terminal rows are inspected inline
    /// but never expanded or counted, so even a fan-out well past the node
    /// cap leaves the live frontier tiny and the walk reaches a real verdict.
    #[tokio::test]
    async fn wedged_child_under_large_terminal_history_still_times_out() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-large-history"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        // A retained history well past the node cap: every one of these is
        // terminal and long silent. The OLD walk would enqueue them all,
        // trip the cap, and truncate → "active" forever.
        let stale = spawned_at;
        for _ in 0..(MAX_ACTIVITY_SUBTREE_NODES + 16) {
            let mut done = AgentTask::new_child(&child_root, TaskKind::ToolRuntime, spawned_at, 3)?;
            done.status = TaskStatus::Completed;
            done.completed_at = Some(stale);
            done.last_activity_at = Some(stale);
            stores.task_store.insert(done).await?;
        }

        let now = time::OffsetDateTime::now_utc();
        assert!(
            stall_expired(&stores, &child_root.id, deadline, None, now).await,
            "a wedged child must still be reaped despite a retained terminal history larger \
             than the probe's node cap",
        );
        Ok(())
    }

    /// r2 finding 2: the heartbeat must renew the lease BEFORE running the
    /// (potentially slow) subtree probe. Otherwise a large subtree or slow
    /// store consumes the remaining lease, the expiry sweep requeues the
    /// row, and a second worker reacquires it while the original future is
    /// still running — duplicate execution. Here the probe's `list_children`
    /// stalls; the lease must nonetheless already be extended when it does.
    #[tokio::test]
    async fn lease_is_renewed_before_the_subtree_probe() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let start = time::OffsetDateTime::now_utc();
        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = start - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-lease-before-probe"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        // Acquire the child Running under a DELIBERATELY short lease, with a
        // stale activity stamp (acquired "in the past"), so the probe does
        // not early-exit and actually reaches `list_children`.
        let worker = WorkerId::from_string("w-lease-probe");
        let lease = LeaseId::new();
        let short_lease_expiry = start + time::Duration::seconds(1);
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                short_lease_expiry,
                spawned_at,
            )
            .await?
            .context("child must acquire")?;

        let flaky = Arc::new(
            FlakyTaskStore::new(Arc::clone(&stores.task_store)).stalling_list_children(800),
        );
        let probe_started = Arc::clone(&flaky.probe_started);
        let mut flaky_stores = stores.clone();
        flaky_stores.task_store = flaky.clone();

        let cancel = CancellationToken::new();
        let handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
            stores: flaky_stores.clone(),
            task_id: child_root.id.clone(),
            thread_id: child_root.thread_id.clone(),
            worker_id: worker.clone(),
            lease_id: lease.clone(),
            lease_duration: time::Duration::seconds(30),
            heartbeat_interval: std::time::Duration::from_millis(50),
            cancel: cancel.clone(),
            task_cancel: CancellationToken::new(),
            deadline: SubagentDeadlineState::Enforced(deadline),
            activity: agent_server::worker::ActivityBeacon::new(),
        }));

        // Wait until the probe is in-flight (its stall has begun): the beat
        // must have already run this tick.
        probe_started.notified().await;

        // Well past the short acquisition lease but far short of the renewed
        // one: the row must not be sweepable, proving the beat ran first.
        let swept = flaky
            .release_expired_leases(start + time::Duration::seconds(5))
            .await?;
        assert!(
            !swept.iter().any(|record| record.id == child_root.id),
            "the lease must be renewed before the probe runs, so the child is not sweepable \
             mid-probe; got {swept:?}",
        );
        let mid = flaky.get(&child_root.id).await?.context("child exists")?;
        assert_eq!(
            mid.status,
            TaskStatus::Running,
            "the child must still be Running under its worker mid-probe",
        );

        cancel.cancel();
        handle.await?;
        Ok(())
    }

    /// r3 finding 2: a probe that outruns its lease-bounded budget must be
    /// treated as NOT expired (fail-safe) so a slow store can never cause a
    /// false reap — the heartbeat retries on the next tick instead.
    #[tokio::test]
    async fn stall_probe_timeout_is_treated_as_active() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();

        let budget_ms: u64 = 30 * 60 * 1_000;
        let spawned_at = time::OffsetDateTime::now_utc() - time::Duration::hours(2);
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-probe-timeout"),
            budget_ms,
            spawned_at,
        )
        .await?;
        let deadline = deadline_for(child_root.created_at, budget_ms, 0).context("deadline")?;

        // Acquire Running with a stale activity stamp so the probe does not
        // early-exit and actually reaches the (stalling) `list_children`.
        let worker = WorkerId::from_string("w-probe-timeout");
        let lease = LeaseId::new();
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                spawned_at,
            )
            .await?
            .context("child must acquire")?;

        let flaky = Arc::new(
            FlakyTaskStore::new(Arc::clone(&stores.task_store)).stalling_list_children(800),
        );
        let mut flaky_stores = stores.clone();
        flaky_stores.task_store = flaky;

        // The child is genuinely silent (no activity), so an UNBOUNDED probe
        // would reap it; a probe bounded to 50ms against an 800ms store stall
        // must instead be treated as active and keep running.
        let outcome = deadline_tick(
            &flaky_stores,
            OwnedRootTask {
                task: &child_root.id,
                thread: &child_root.thread_id,
                worker: &worker,
                lease: &lease,
            },
            SubagentDeadlineState::Enforced(deadline),
            None,
            std::time::Duration::from_millis(50),
            0,
            time::OffsetDateTime::now_utc(),
        )
        .await;
        assert!(
            matches!(outcome, DeadlineTick::Continue(_)),
            "a probe that exceeds its budget must be treated as active (not reaped)",
        );
        let row = flaky_stores
            .task_store
            .get(&child_root.id)
            .await?
            .context("child exists")?;
        assert_eq!(
            row.status,
            TaskStatus::Running,
            "the child must not have been failed on a timed-out probe",
        );
        Ok(())
    }

    /// r3 finding 3: a budget below the persistence cadence is floored at
    /// enforcement, so a parked child whose descendant reported within a
    /// couple of heartbeat intervals is not reaped before its activity could
    /// even be persisted.
    #[tokio::test]
    async fn effective_budget_is_floored_at_the_persistence_cadence() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        // Pure floor: a sub-cadence budget is raised; a supra-cadence one is
        // left exactly as configured.
        let created = time::OffsetDateTime::now_utc();
        let floored = deadline_for(created, 5_000, 20_000).context("floored")?;
        assert_eq!(
            floored.timeout_ms, 20_000,
            "a 5s budget floors to the 20s cadence"
        );
        assert_eq!(
            floored.earliest_expiry_at,
            created + time::Duration::seconds(20)
        );
        let unfloored = deadline_for(created, 30_000, 20_000).context("unfloored")?;
        assert_eq!(
            unfloored.timeout_ms, 30_000,
            "a supra-cadence budget is unchanged"
        );

        // Behavioural: a child that reported 17s ago, created 25s ago, under
        // a configured 5s budget. Unfloored it looks long-silent and is
        // reaped; floored to 20s it is correctly spared.
        let host = ServiceHost::new(
            ServiceConfig::default(),
            sample_registry(),
            subagent_timeout_runtime(Arc::new(SubagentScriptProvider::new()))?,
        )?;
        let stores = host.stores().clone();
        let now = time::OffsetDateTime::now_utc();
        let (_parent, _invocation, child_root) = persist_subagent_fixture(
            &stores,
            &ThreadId::from_string("t-budget-floor"),
            5_000,
            now - time::Duration::seconds(25),
        )
        .await?;
        // Stamp last activity 17s ago via acquisition.
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                WorkerId::from_string("w-floor"),
                LeaseId::new(),
                now + time::Duration::seconds(600),
                now - time::Duration::seconds(17),
            )
            .await?
            .context("child must acquire")?;

        let floor = 20_000;
        assert!(
            stall_expired(
                &stores,
                &child_root.id,
                deadline_for(child_root.created_at, 5_000, 0).context("d")?,
                None,
                now,
            )
            .await,
            "without the floor a 5s budget reaps a child that reported 17s ago",
        );
        assert!(
            !stall_expired(
                &stores,
                &child_root.id,
                deadline_for(child_root.created_at, 5_000, floor).context("d")?,
                None,
                now,
            )
            .await,
            "the persistence-cadence floor must spare a child that reported within it",
        );
        Ok(())
    }

    #[tokio::test]
    async fn resolve_subagent_deadline_only_for_linked_child_roots() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let at = time::OffsetDateTime::now_utc();
        let parent_thread = ThreadId::from_string("t-subagent-resolve");
        let (parent, _invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 250, at).await?;

        // Linked child root: the EARLIEST possible expiry is anchored at
        // created_at + timeout. Reaching it is necessary but not
        // sufficient — `stall_expired` still has to find the child silent.
        let SubagentDeadlineState::Enforced(deadline) =
            resolve_subagent_deadline(&stores, &child_root).await
        else {
            bail!("linked child root must resolve an enforced deadline");
        };
        assert_eq!(deadline.timeout_ms, 250);
        assert_eq!(
            deadline.earliest_expiry_at,
            child_root.created_at + time::Duration::milliseconds(250),
            "the earliest-expiry floor must anchor at the child root's creation time",
        );

        // The parent root (not a linked child root) resolves no deadline.
        assert!(
            matches!(
                resolve_subagent_deadline(&stores, &parent).await,
                SubagentDeadlineState::Exempt
            ),
            "a non-child root must run without a deadline",
        );

        // A plain root on an unrelated thread resolves no deadline.
        let plain = AgentTask::new_root_turn(ThreadId::from_string("t-plain-root"), at, 3);
        let plain = stores.task_store.submit_root_turn(plain).await?;
        assert!(
            matches!(
                resolve_subagent_deadline(&stores, &plain).await,
                SubagentDeadlineState::Exempt
            ),
            "a plain root must run without a deadline",
        );
        Ok(())
    }

    // ── Parked-leg enforcement via the sweep (issue #299, round 2) ──

    /// Tool executor that hangs until its cancellation token trips
    /// (a stuck external tool), recording each start.
    struct HangingProbeExecutor {
        started: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl crate::runtime::ToolCallExecutor for HangingProbeExecutor {
        async fn execute_tool_call(
            &self,
            _bootstrap: &agent_server::worker::ToolTaskBootstrap,
            _collector: agent_server::worker::ToolEventCollector,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<agent_sdk_foundation::ToolResult> {
            self.started.fetch_add(1, Ordering::SeqCst);
            cancel.cancelled().await;
            Ok(agent_sdk_foundation::ToolResult {
                success: false,
                output: "hung probe aborted".into(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            })
        }
    }

    /// Acquire `child_root` and park it in `WaitingOnChildren` on one
    /// freshly-spawned `probe` tool child. Returns the tool child.
    async fn park_child_root_on_tool_child(
        stores: &StoreRegistry,
        child_root: &AgentTask,
        at: time::OffsetDateTime,
    ) -> Result<AgentTask> {
        use agent_server::journal::task::ChildSpawnSpec;

        let worker = WorkerId::from_string("w-child-park");
        let lease = LeaseId::new();
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                at,
            )
            .await?
            .context("child root must acquire before parking")?;
        let (_parked, children) = stores
            .task_store
            .spawn_tool_children(
                &child_root.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec { max_attempts: 3 }],
                pending_call_suspension(&child_root.thread_id, "probe"),
                None,
                at,
            )
            .await?;
        children
            .into_iter()
            .next()
            .context("exactly one spawned tool child")
    }

    /// End-to-end parked-leg enforcement: the hang child's turn
    /// requests a `probe` tool whose executor hangs forever, so the
    /// child root parks in `WaitingOnChildren` with NO live heartbeat.
    /// Only the deadline sweep can unwedge it: cancel the hung tool
    /// child, fail the root with the timeout message, and resume the
    /// parent with the failed outcome.
    #[tokio::test]
    async fn parked_subagent_child_with_hung_tool_is_failed_by_sweep() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 4,
                heartbeat_interval_secs: 1,
                acquisition_interval_secs: 1,
                sweep_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = Arc::new(SubagentScriptProvider::with_hung_tool());
        let probe_starts = Arc::new(AtomicUsize::new(0));
        let runtime = subagent_timeout_runtime_with_executor(
            Arc::clone(&provider),
            Arc::new(HangingProbeExecutor {
                started: Arc::clone(&probe_starts),
            }),
        )?;
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(probe_definition()));
        let host = ServiceHost::new(config, registry, runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        let parent_thread = ThreadId::from_string("t-subagent-parked");
        let parent = AgentTask::new_root_turn_with_input(
            parent_thread.clone(),
            vec![SubmittedInputItem::Text {
                text: "coordinate the helpers".into(),
            }],
            time::OffsetDateTime::now_utc(),
            3,
        );
        let parent_id = parent.id.clone();
        stores.task_store.submit_root_turn(parent).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // The hang child parks on a tool child whose executor never
        // returns; no heartbeat exists for the parked root, so ONLY
        // the sweep can fail it and let the parent resume.
        wait_for_status(&stores, &parent_id, TaskStatus::Completed, 1_500).await?;

        let hang_thread = ThreadId::from_string("t-subagent-parked-hang");
        let hang_root = root_task_of_thread(&stores, &hang_thread).await?;
        assert_eq!(
            hang_root.status,
            TaskStatus::Failed,
            "the parked timed-out child must fail, not cancel",
        );
        let hang_error = hang_root.last_error.clone().unwrap_or_default();
        assert!(
            hang_error.contains("subagent timed out after 2500ms"),
            "parked timed-out child must carry the timeout message, got {hang_error:?}",
        );

        // The hung tool child was cancelled (not stranded), after
        // actually starting — proving the parked leg, not a pre-tool
        // failure.
        assert!(
            probe_starts.load(Ordering::SeqCst) >= 1,
            "the hung probe must have started executing",
        );
        let tool_children = stores.task_store.list_children(&hang_root.id).await?;
        assert!(
            !tool_children.is_empty(),
            "the hang child must have spawned its probe tool child",
        );
        for tool_child in &tool_children {
            assert_eq!(
                tool_child.status,
                TaskStatus::Cancelled,
                "hung tool child {} must be cancelled by the sweep",
                tool_child.id,
            );
        }

        // The generously-budgeted sibling and the parent fan-in are
        // unaffected; the parent LLM saw the timeout message.
        let fast_root =
            root_task_of_thread(&stores, &ThreadId::from_string("t-subagent-parked-fast")).await?;
        assert_eq!(fast_root.status, TaskStatus::Completed);
        let resume_text = provider.recorded_resume_text()?;
        assert!(
            resume_text.contains("subagent timed out after 2500ms"),
            "the parent's resume request must carry the timeout message, got {resume_text:?}",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn sweep_fails_expired_parked_child_and_cancels_hung_tool_children() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let backdated = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
        let parent_thread = ThreadId::from_string("t-sweep-parked");
        let (_parent, invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 250, backdated).await?;
        let tool_child = park_child_root_on_tool_child(&stores, &child_root, backdated).await?;

        let enforced = enforce_subagent_deadlines(
            &stores,
            time::OffsetDateTime::now_utc(),
            0,
            &CancellationToken::new(),
        )
        .await?;
        assert_eq!(enforced, 1, "one parked child must be enforced");

        let tool_after = stores
            .task_store
            .get(&tool_child.id)
            .await?
            .context("tool child exists")?;
        assert_eq!(
            tool_after.status,
            TaskStatus::Cancelled,
            "the hung tool child must be cancelled, not stranded",
        );
        let root_after = stores
            .task_store
            .get(&child_root.id)
            .await?
            .context("child root exists")?;
        assert_eq!(root_after.status, TaskStatus::Failed);
        let root_error = root_after.last_error.clone().unwrap_or_default();
        assert!(
            root_error.contains("subagent timed out after 250ms"),
            "sweep-failed child must carry the timeout message, got {root_error:?}",
        );
        let invocation_after = stores
            .task_store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(
            invocation_after.status,
            TaskStatus::Pending,
            "the failed child must wake the parked invocation",
        );
        assert_eq!(invocation_after.pending_child_count, 0);
        Ok(())
    }

    #[tokio::test]
    async fn sweep_leaves_unexpired_parked_subagent_children_alone() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let at = time::OffsetDateTime::now_utc();
        let parent_thread = ThreadId::from_string("t-sweep-unexpired");
        let (_parent, invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 600_000, at).await?;
        let tool_child = park_child_root_on_tool_child(&stores, &child_root, at).await?;

        let enforced = enforce_subagent_deadlines(
            &stores,
            time::OffsetDateTime::now_utc(),
            0,
            &CancellationToken::new(),
        )
        .await?;
        assert_eq!(
            enforced, 0,
            "an unexpired parked child must not be enforced"
        );

        let tool_after = stores
            .task_store
            .get(&tool_child.id)
            .await?
            .context("tool child exists")?;
        assert_eq!(tool_after.status, TaskStatus::Pending);
        let root_after = stores
            .task_store
            .get(&child_root.id)
            .await?
            .context("child root exists")?;
        assert_eq!(root_after.status, TaskStatus::WaitingOnChildren);
        let invocation_after = stores
            .task_store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(invocation_after.status, TaskStatus::WaitingOnChildren);
        Ok(())
    }

    // ── Transient-failure resilience (issue #299, round 2) ──────────

    /// [`AgentTaskStore`] wrapper that injects a bounded number of
    /// failures into `find_subagent_invocation_for_child_root` and
    /// `fail_task`, delegating everything (and everything else) to the
    /// wrapped in-memory store. Used to prove the deadline machinery
    /// survives transient store errors.
    struct FlakyTaskStore {
        inner: Arc<dyn AgentTaskStore>,
        find_failures_remaining: AtomicUsize,
        fail_task_failures_remaining: AtomicUsize,
        fail_task_attempts: AtomicUsize,
        heartbeat_rejections_remaining: AtomicUsize,
        /// When non-zero, `list_children` (the subtree probe's expansion
        /// call) sleeps this many milliseconds before delegating, and fires
        /// `probe_started` as it begins — so a test can prove the heartbeat
        /// renewed the lease BEFORE the probe ran.
        list_children_stall_ms: AtomicUsize,
        probe_started: Arc<tokio::sync::Notify>,
    }

    impl FlakyTaskStore {
        fn new(inner: Arc<dyn AgentTaskStore>) -> Self {
            Self {
                inner,
                find_failures_remaining: AtomicUsize::new(0),
                fail_task_failures_remaining: AtomicUsize::new(0),
                fail_task_attempts: AtomicUsize::new(0),
                heartbeat_rejections_remaining: AtomicUsize::new(0),
                list_children_stall_ms: AtomicUsize::new(0),
                probe_started: Arc::new(tokio::sync::Notify::new()),
            }
        }

        /// Make every `list_children` call (the subtree probe's expansion
        /// step) sleep `ms` before delegating, firing `probe_started` as it
        /// begins — so a test can assert the heartbeat renewed the lease
        /// before the probe was even entered.
        fn stalling_list_children(self, ms: usize) -> Self {
            self.list_children_stall_ms.store(ms, Ordering::SeqCst);
            self
        }

        fn failing_finds(self, count: usize) -> Self {
            self.find_failures_remaining.store(count, Ordering::SeqCst);
            self
        }

        fn failing_fail_tasks(self, count: usize) -> Self {
            self.fail_task_failures_remaining
                .store(count, Ordering::SeqCst);
            self
        }

        /// Reject the next `count` heartbeats with the canonical
        /// terminal "heartbeat rejected" marker — simulating a lease
        /// that was revoked out from under the worker (sweep requeue,
        /// `cancel_tree`) even though the row itself was not updated.
        fn rejecting_heartbeats(self, count: usize) -> Self {
            self.heartbeat_rejections_remaining
                .store(count, Ordering::SeqCst);
            self
        }

        /// Consume one injected failure from `remaining`, returning
        /// `true` while injections are left.
        fn take_injected_failure(remaining: &AtomicUsize) -> bool {
            remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                    current.checked_sub(1)
                })
                .is_ok()
        }
    }

    use agent_sdk_foundation::{ContinuationEnvelope, ListenExecutionContext};
    use agent_server::journal::SubagentInvocationSpawn;
    use agent_server::journal::idempotency::{
        IdempotencyClaim, IdempotencyKind, IdempotencyRecord,
    };
    use agent_server::journal::recovery::RecoveryRecord;
    use agent_server::journal::store::{
        AgentTaskStore, SubmitRootTurnError, SubmitRootTurnOutcome, SubmitRootTurnParams,
    };
    use agent_server::journal::task::{AgentTaskId, ChildSpawnSpec, SuspensionPayload};

    #[async_trait]
    impl AgentTaskStore for FlakyTaskStore {
        async fn insert(&self, task: AgentTask) -> Result<()> {
            self.inner.insert(task).await
        }
        async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
            self.inner.submit_root_turn(task).await
        }
        async fn submit_root_turn_idempotent(
            &self,
            params: SubmitRootTurnParams,
        ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
            self.inner.submit_root_turn_idempotent(params).await
        }
        async fn claim_idempotency(
            &self,
            request_id: &str,
            kind: IdempotencyKind,
            fingerprint: &[u8],
        ) -> Result<IdempotencyClaim> {
            self.inner
                .claim_idempotency(request_id, kind, fingerprint)
                .await
        }
        async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
            self.inner.record_idempotency(record).await
        }
        async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
            self.inner.get(id).await
        }
        async fn update(&self, task: AgentTask) -> Result<()> {
            self.inner.update(task).await
        }
        async fn list_by_thread(
            &self,
            thread_id: &agent_sdk_foundation::ThreadId,
        ) -> Result<Vec<AgentTask>> {
            self.inner.list_by_thread(thread_id).await
        }
        async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
            let stall_ms = self.list_children_stall_ms.load(Ordering::SeqCst);
            if stall_ms > 0 {
                self.probe_started.notify_one();
                tokio::time::sleep(std::time::Duration::from_millis(stall_ms as u64)).await;
            }
            self.inner.list_children(parent_id).await
        }
        async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
            self.inner.list_by_status(status).await
        }
        async fn active_root_for_thread(
            &self,
            thread_id: &agent_sdk_foundation::ThreadId,
        ) -> Result<Option<AgentTask>> {
            self.inner.active_root_for_thread(thread_id).await
        }
        async fn list_queued_roots(
            &self,
            thread_id: &agent_sdk_foundation::ThreadId,
        ) -> Result<Vec<AgentTask>> {
            self.inner.list_queued_roots(thread_id).await
        }
        async fn requeue_owned_task(
            &self,
            id: &agent_server::AgentTaskId,
            worker: &agent_server::WorkerId,
            lease: &agent_server::LeaseId,
            boundary: Option<agent_sdk_foundation::events::AgentEvent>,
            now: time::OffsetDateTime,
        ) -> Result<RequeueOutcome> {
            self.inner
                .requeue_owned_task(id, worker, lease, boundary, now)
                .await
        }
        async fn promote_next_queued_root(
            &self,
            thread_id: &agent_sdk_foundation::ThreadId,
            now: time::OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner.promote_next_queued_root(thread_id, now).await
        }
        async fn try_acquire_task(
            &self,
            task_id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            lease_expires_at: time::OffsetDateTime,
            now: time::OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .try_acquire_task(task_id, worker, lease, lease_expires_at, now)
                .await
        }
        async fn acquire_next_runnable(
            &self,
            worker: WorkerId,
            lease: LeaseId,
            lease_expires_at: time::OffsetDateTime,
            now: time::OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .acquire_next_runnable(worker, lease, lease_expires_at, now)
                .await
        }
        async fn heartbeat_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            expires_at: time::OffsetDateTime,
            activity: Option<time::OffsetDateTime>,
            now: time::OffsetDateTime,
        ) -> Result<AgentTask> {
            if Self::take_injected_failure(&self.heartbeat_rejections_remaining) {
                bail!("heartbeat rejected: injected terminal lease rejection");
            }
            self.inner
                .heartbeat_task(task_id, worker, lease, expires_at, activity, now)
                .await
        }
        async fn release_expired_leases(
            &self,
            now: time::OffsetDateTime,
        ) -> Result<Vec<RecoveryRecord>> {
            self.inner.release_expired_leases(now).await
        }
        async fn pause_on_children(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            child_count: u32,
            payload: SuspensionPayload,
            now: time::OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .pause_on_children(task_id, worker, lease, child_count, payload, now)
                .await
        }
        async fn enqueue_steering_resume(
            &self,
            parent_id: &AgentTaskId,
            steering: Vec<agent_sdk_foundation::llm::ContentBlock>,
            now: time::OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .enqueue_steering_resume(parent_id, steering, now)
                .await
        }
        async fn repark_after_steering(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            payload: SuspensionPayload,
            reattach: Vec<AgentTaskId>,
            now: time::OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .repark_after_steering(parent_id, worker, lease, payload, reattach, now)
                .await
        }
        async fn pause_on_confirmation(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            continuation: ContinuationEnvelope,
            prepared_operation: Option<ListenExecutionContext>,
            now: time::OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .pause_on_confirmation(
                    task_id,
                    worker,
                    lease,
                    continuation,
                    prepared_operation,
                    now,
                )
                .await
        }
        // Trait-signature-mandated: every impl of this frozen trait
        // method (in-memory reference, SQLite, Postgres, conformance
        // bundle) carries the same allow because the argument list
        // cannot be reshaped without breaking the store contract.
        #[allow(clippy::too_many_arguments)]
        async fn spawn_tool_children(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            specs: Vec<ChildSpawnSpec>,
            payload: SuspensionPayload,
            child_otel_traceparent: Option<String>,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Vec<AgentTask>)> {
            self.inner
                .spawn_tool_children(
                    parent_id,
                    worker,
                    lease,
                    specs,
                    payload,
                    child_otel_traceparent,
                    now,
                )
                .await
        }
        async fn spawn_subagent_invocation(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawn: SubagentInvocationSpawn,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, AgentTask, AgentTask)> {
            self.inner
                .spawn_subagent_invocation(parent_id, worker, lease, spawn, now)
                .await
        }
        async fn spawn_subagent_batch(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawns: Vec<SubagentInvocationSpawn>,
            payload: SuspensionPayload,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>)> {
            self.inner
                .spawn_subagent_batch(parent_id, worker, lease, spawns, payload, now)
                .await
        }
        async fn spawn_mixed_children(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawn: agent_server::journal::MixedChildrenSpawn,
            now: time::OffsetDateTime,
        ) -> Result<agent_server::journal::SpawnedMixedChildren> {
            self.inner
                .spawn_mixed_children(parent_id, worker, lease, spawn, now)
                .await
        }
        async fn find_subagent_invocation_for_child_root(
            &self,
            child_root_id: &AgentTaskId,
        ) -> Result<Option<AgentTask>> {
            if Self::take_injected_failure(&self.find_failures_remaining) {
                bail!("injected transient linkage lookup failure");
            }
            self.inner
                .find_subagent_invocation_for_child_root(child_root_id)
                .await
        }
        async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>> {
            self.inner.list_parked_subagent_invocations().await
        }
        async fn complete_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.inner.complete_task(task_id, worker, lease, now).await
        }
        async fn complete_task_with_result(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            result: serde_json::Value,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.inner
                .complete_task_with_result(task_id, worker, lease, result, now)
                .await
        }
        async fn fail_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            error: String,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.fail_task_attempts.fetch_add(1, Ordering::SeqCst);
            if Self::take_injected_failure(&self.fail_task_failures_remaining) {
                bail!("injected transient fail_task failure");
            }
            self.inner
                .fail_task(task_id, worker, lease, error, now)
                .await
        }
        async fn cancel_tree(
            &self,
            root_id: &AgentTaskId,
            now: time::OffsetDateTime,
        ) -> Result<agent_server::journal::store::CancelTreeOutcome> {
            self.inner.cancel_tree(root_id, now).await
        }
        async fn resume_from_confirmation(
            &self,
            task_id: &AgentTaskId,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.inner.resume_from_confirmation(task_id, now).await
        }
        async fn approve_confirmation_and_acquire(
            &self,
            task_id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            expires_at: time::OffsetDateTime,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.inner
                .approve_confirmation_and_acquire(task_id, worker, lease, expires_at, now)
                .await
        }
        async fn reject_confirmation(
            &self,
            task_id: &AgentTaskId,
            error: String,
            now: time::OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.inner.reject_confirmation(task_id, error, now).await
        }
        async fn clear(&self) -> Result<()> {
            self.inner.clear().await
        }
    }

    /// Build a fixture whose child root is Running under `(worker,
    /// lease)`, plus a [`StoreRegistry`] clone whose task store is the
    /// given flaky wrapper. Returns `(flaky_stores, child_root,
    /// worker, lease)`.
    async fn acquired_child_with_flaky_store(
        stores: &StoreRegistry,
        flaky: Arc<FlakyTaskStore>,
        parent_thread_name: &str,
        timeout_ms: u64,
        at: time::OffsetDateTime,
    ) -> Result<(StoreRegistry, AgentTask, WorkerId, LeaseId)> {
        let parent_thread = agent_sdk_foundation::ThreadId::from_string(parent_thread_name);
        let (_parent, _invocation, child_root) =
            persist_subagent_fixture(stores, &parent_thread, timeout_ms, at).await?;
        let worker = WorkerId::from_string("w-flaky");
        let lease = LeaseId::new();
        let acquired = stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("child root must acquire for the flaky-store test")?;
        let mut flaky_stores = stores.clone();
        flaky_stores.task_store = flaky;
        Ok((flaky_stores, acquired, worker, lease))
    }

    /// Item 2: a transient linkage-lookup failure must not disable the
    /// deadline for the whole acquisition — the heartbeat retries the
    /// resolution each tick until it succeeds, then enforces.
    #[tokio::test]
    async fn deadline_resolution_retries_after_transient_lookup_errors() -> Result<()> {
        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let flaky = Arc::new(FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_finds(2));
        let (flaky_stores, child_root, worker, lease) = acquired_child_with_flaky_store(
            &stores,
            Arc::clone(&flaky),
            "t-flaky-resolve",
            100,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let cancel = CancellationToken::new();
        let task_cancel = CancellationToken::new();
        let handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
            stores: flaky_stores.clone(),
            task_id: child_root.id.clone(),
            thread_id: child_root.thread_id.clone(),
            worker_id: worker,
            lease_id: lease,
            lease_duration: time::Duration::seconds(30),
            heartbeat_interval: std::time::Duration::from_millis(50),
            cancel: cancel.clone(),
            task_cancel: task_cancel.clone(),
            // Simulates a lookup that already failed at acquisition.
            deadline: SubagentDeadlineState::Unresolved {
                created_at: child_root.created_at,
            },
            activity: ActivityBeacon::new(),
        }));

        // Two injected lookup failures burn two ticks; the third tick
        // resolves the deadline and enforces the (long-expired) budget.
        let failed =
            wait_for_status(&flaky_stores, &child_root.id, TaskStatus::Failed, 250).await?;
        let error = failed.last_error.unwrap_or_default();
        assert!(
            error.contains("subagent timed out after 100ms"),
            "deadline must be enforced once resolution finally succeeds, got {error:?}",
        );
        assert!(
            task_cancel.is_cancelled(),
            "the per-task token must trip once the timeout failure lands",
        );
        cancel.cancel();
        handle.await?;
        Ok(())
    }

    /// Item 3 (heartbeat leg): a transient store error while failing
    /// the timed-out child must keep the lease and retry — never trip
    /// the turn and walk away with the row still Running (which would
    /// hand the row to the expiry sweep and lose the timeout outcome).
    #[tokio::test]
    async fn timeout_failure_retries_until_durable_keeping_the_lease() -> Result<()> {
        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let flaky =
            Arc::new(FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_fail_tasks(1));
        let backdated = time::OffsetDateTime::now_utc() - time::Duration::seconds(10);
        let (flaky_stores, child_root, worker, lease) = acquired_child_with_flaky_store(
            &stores,
            Arc::clone(&flaky),
            "t-flaky-fail",
            100,
            backdated,
        )
        .await?;
        let Some(deadline) = deadline_for(child_root.created_at, 100, 0) else {
            bail!("fixture deadline must resolve");
        };

        // An open attempt stands in for the live worker's in-flight
        // turn: the heartbeat-driven timeout fail must leave it OPEN
        // (addendum A — the live worker owns its attempt closes).
        let open_attempt = stores
            .attempt_store
            .open_attempt(agent_server::journal::turn_attempt::OpenAttemptParams {
                task_id: child_root.id.clone(),
                attempt_number: 1,
                provenance: agent_sdk_foundation::audit::AuditProvenance::new("test", "test"),
                request_blob: serde_json::json!({ "user_prompt": "live turn" }),
                now: time::OffsetDateTime::now_utc(),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;

        let cancel = CancellationToken::new();
        let task_cancel = CancellationToken::new();
        let handle = tokio::spawn(heartbeat_loop(HeartbeatLoopParams {
            stores: flaky_stores.clone(),
            task_id: child_root.id.clone(),
            thread_id: child_root.thread_id.clone(),
            worker_id: worker,
            lease_id: lease,
            lease_duration: time::Duration::seconds(30),
            heartbeat_interval: std::time::Duration::from_millis(50),
            cancel: cancel.clone(),
            task_cancel: task_cancel.clone(),
            deadline: SubagentDeadlineState::Enforced(deadline),
            activity: ActivityBeacon::new(),
        }));

        // Tick 1's fail hits the injected error; tick 2 retries and
        // lands the durable timeout failure. Under the old behavior
        // (trip-and-exit on error) the row would stay Running forever
        // in this harness and this wait would time out.
        let failed =
            wait_for_status(&flaky_stores, &child_root.id, TaskStatus::Failed, 250).await?;
        let error = failed.last_error.unwrap_or_default();
        assert!(
            error.contains("subagent timed out after 100ms"),
            "the retried failure must carry the timeout message, got {error:?}",
        );
        assert!(
            flaky.fail_task_attempts.load(Ordering::SeqCst) >= 2,
            "the durable fail must have been retried after the injected error",
        );
        assert!(
            task_cancel.is_cancelled(),
            "the per-task token must trip only once the failure landed",
        );

        // Addendum A wiring: the heartbeat-driven fail routed through
        // `LeaveOpenForLiveWorker`, so the live worker's open attempt
        // survives (its own abort path owns the close).
        let attempts = stores.attempt_store.list_by_task(&child_root.id).await?;
        let reread = attempts
            .into_iter()
            .find(|row| row.id == open_attempt.id)
            .context("opened attempt still listed")?;
        assert!(
            !reread.is_closed(),
            "the heartbeat timeout path must not pre-close the live worker's open attempt",
        );

        cancel.cancel();
        handle.await?;
        Ok(())
    }

    /// Addendum A: the heartbeat-timeout path must NOT pre-close the
    /// live worker's open attempt (its real-usage close would be
    /// clobbered / its commit aborted); paths with no live worker
    /// (acquisition expiry, the sweep) keep the pre-close so stale
    /// attempts from crashed lease holders do not linger.
    #[tokio::test]
    async fn timeout_fail_attempt_close_policy_respects_live_worker() -> Result<()> {
        use agent_sdk_foundation::audit::AuditProvenance;
        use agent_server::journal::turn_attempt::OpenAttemptParams;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();
        let now = time::OffsetDateTime::now_utc();

        let mut open_attempt_after_fail = Vec::new();
        for (thread_name, policy) in [
            ("t-attempt-live", AttemptClosePolicy::LeaveOpenForLiveWorker),
            ("t-attempt-orphan", AttemptClosePolicy::CloseOpenAttempts),
        ] {
            let parent_thread = agent_sdk_foundation::ThreadId::from_string(thread_name);
            let (_parent, _invocation, child_root) =
                persist_subagent_fixture(&stores, &parent_thread, 600_000, now).await?;
            let worker = WorkerId::from_string("w-attempt");
            let lease = LeaseId::new();
            stores
                .task_store
                .try_acquire_task(
                    &child_root.id,
                    worker.clone(),
                    lease.clone(),
                    now + time::Duration::seconds(600),
                    now,
                )
                .await?
                .context("child root must acquire")?;
            let attempt = stores
                .attempt_store
                .open_attempt(OpenAttemptParams {
                    task_id: child_root.id.clone(),
                    attempt_number: 1,
                    provenance: AuditProvenance::new("test", "test"),
                    request_blob: serde_json::json!({ "user_prompt": "attempt-close probe" }),
                    now,
                    otel_trace_id: None,
                    otel_span_id: None,
                })
                .await?;

            fail_timed_out_subagent_root(
                &stores,
                FailTimedOutChild {
                    task: &child_root.id,
                    thread: &child_root.thread_id,
                    worker: &worker,
                    lease: &lease,
                    timeout_ms: 777,
                    attempt_close: policy,
                },
                time::OffsetDateTime::now_utc(),
            )
            .await?;

            let failed = stores
                .task_store
                .get(&child_root.id)
                .await?
                .context("child root exists")?;
            assert_eq!(failed.status, TaskStatus::Failed);

            let attempts = stores.attempt_store.list_by_task(&child_root.id).await?;
            let reread = attempts
                .into_iter()
                .find(|row| row.id == attempt.id)
                .context("opened attempt still listed")?;
            open_attempt_after_fail.push(!reread.is_closed());
        }

        assert_eq!(
            open_attempt_after_fail,
            vec![true, false],
            "the live-worker path must leave the open attempt untouched; \
             the no-live-worker path must close it",
        );
        Ok(())
    }

    // ── Round 3: sweep durable-fail retry, slot reclaim, lease-safe
    //    resolution ────────────────────────────────────────────────

    /// Item 1: the sweep leg holds its acquired lease across transient
    /// durable-fail errors, so the timeout message lands in the SAME
    /// pass — no lease-expiry requeue, no burned attempt, no eventual
    /// fail-closed message replacing the timeout one.
    #[tokio::test]
    async fn sweep_timeout_fail_retries_transient_errors_without_burning_attempts() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let backdated = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
        let parent_thread = ThreadId::from_string("t-sweep-flaky-fail");
        let (_parent, invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 250, backdated).await?;
        let tool_child = park_child_root_on_tool_child(&stores, &child_root, backdated).await?;

        let flaky =
            Arc::new(FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_fail_tasks(1));
        let mut flaky_stores = stores.clone();
        flaky_stores.task_store = Arc::clone(&flaky) as Arc<dyn AgentTaskStore>;

        let enforced = enforce_subagent_deadlines(
            &flaky_stores,
            time::OffsetDateTime::now_utc(),
            0,
            &CancellationToken::new(),
        )
        .await?;
        assert_eq!(
            enforced, 1,
            "the sweep must settle the timeout failure within the same pass",
        );

        let failed = flaky_stores
            .task_store
            .get(&child_root.id)
            .await?
            .context("child root exists")?;
        assert_eq!(failed.status, TaskStatus::Failed);
        let error = failed.last_error.unwrap_or_default();
        assert!(
            error.contains("subagent timed out after 250ms"),
            "the retried sweep failure must carry the timeout message, got {error:?}",
        );
        // ReadyToResume acquisitions consume no retry budget, so the
        // fixture's single fresh acquire is the only one on the row —
        // a lease-expiry detour would have incremented it.
        assert_eq!(
            failed.attempt, 1,
            "the sweep retry must not burn an attempt on the child root",
        );
        assert!(
            flaky.fail_task_attempts.load(Ordering::SeqCst) >= 2,
            "the durable fail must have been retried after the injected error",
        );
        let tool_after = stores
            .task_store
            .get(&tool_child.id)
            .await?
            .context("tool child exists")?;
        assert_eq!(tool_after.status, TaskStatus::Cancelled);
        let invocation_after = stores
            .task_store
            .get(&invocation.id)
            .await?
            .context("invocation exists")?;
        assert_eq!(invocation_after.status, TaskStatus::Pending);
        Ok(())
    }

    /// Provider resolver whose FIRST resolve hangs forever on a
    /// token-blind await (the exact class of hang addendum A reclaims
    /// the worker slot from); later resolves delegate normally so the
    /// parent's fan-in resume can still run.
    struct HangOnceProviderResolver {
        inner: Arc<StaticProviderResolver>,
        hung_once: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl crate::runtime::ProviderResolver for HangOnceProviderResolver {
        async fn resolve_provider(
            &self,
            definition: &AgentDefinition,
        ) -> Result<Arc<dyn LlmProvider>> {
            if !self.hung_once.swap(true, Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            self.inner.resolve_provider(definition).await
        }
    }

    /// Addendum A: a timed-out child blocked in a token-blind await
    /// (`ProviderResolver::resolve_provider` receives no cancel token)
    /// must not pin its worker-pool slot forever — after the durable
    /// failure lands and the grace window elapses, the execution
    /// future is dropped and the SINGLE worker goes on to run the
    /// invocation and the parent's fan-in resume.
    #[tokio::test]
    async fn hung_provider_resolve_frees_the_worker_slot() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 1,
                heartbeat_interval_secs: 1,
                acquisition_interval_secs: 1,
                sweep_interval_secs: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = Arc::new(SubagentScriptProvider::new());
        let static_resolver = Arc::new(StaticProviderResolver::new());
        static_resolver.set_fallback(Arc::clone(&provider) as Arc<dyn LlmProvider>)?;
        let runtime = Arc::new(
            ExecutionRuntime::new(
                Arc::new(HangOnceProviderResolver {
                    inner: static_resolver,
                    hung_once: std::sync::atomic::AtomicBool::new(false),
                }),
                Arc::new(NoopToolExecutor),
                Arc::new(AllowAllConfirmationPolicy),
            )
            .with_subagent_spawn_selector(Arc::new(DeadlineSpawnSelector)),
        );
        let runtime_kick = Arc::clone(&runtime);
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        // Budget must outlive worker pickup (kicked below) but expire
        // while the resolve hang is in flight.
        let parent_thread = ThreadId::from_string("t-hung-resolve");
        let (parent, invocation, child_root) = persist_subagent_fixture(
            &stores,
            &parent_thread,
            3_000,
            time::OffsetDateTime::now_utc(),
        )
        .await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Kick the single worker so the child is acquired well before
        // its 3s deadline (dispatching the hung resolve for real).
        for _ in 0..200 {
            if let Some(signal) = runtime_kick.wakeup_signal() {
                signal.notify_workers();
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        // The ONLY worker is wedged in the token-blind resolve; every
        // subsequent hop (invocation materialization, parent resume)
        // requires the slot to be reclaimed after the timeout failure.
        let failed = wait_for_status(&stores, &child_root.id, TaskStatus::Failed, 1_500).await?;
        let error = failed.last_error.unwrap_or_default();
        assert!(
            error.contains("subagent timed out after 3000ms"),
            "the hung child must carry the timeout message, got {error:?}",
        );
        wait_for_status(&stores, &invocation.id, TaskStatus::Completed, 1_500).await?;
        wait_for_status(&stores, &parent.id, TaskStatus::Completed, 1_500).await?;
        let resume_text = provider.recorded_resume_text()?;
        assert!(
            resume_text.contains("subagent timed out after 3000ms"),
            "the parent's resume request must carry the timeout message, got {resume_text:?}",
        );

        token.cancel();
        host_handle.await??;
        Ok(())
    }

    // ── Round 4: confirm-drive cancellation, force-drop attempt
    //    settlement, multi-batch lease drain ───────────────────────

    /// Item 1 (sweep edge): when deadline enforcement cancels a parked
    /// child's tool subtree, any registered approved-confirmation
    /// drive token for a cancelled task must trip immediately — not
    /// wait for the drive's next heartbeat rejection.
    #[tokio::test]
    async fn sweep_cancels_registered_confirm_drives() -> Result<()> {
        use agent_sdk_foundation::ThreadId;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let backdated = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
        let parent_thread = ThreadId::from_string("t-sweep-drive-cancel");
        let (_parent, _invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 250, backdated).await?;
        let tool_child = park_child_root_on_tool_child(&stores, &child_root, backdated).await?;

        // Stands in for a live `drive_approved_confirmation` executing
        // the (already approved) tool child. The guard must stay alive
        // through enforcement, like a real drive's.
        let drive_token = CancellationToken::new();
        let _drive_registration = stores
            .confirm_drive_cancels
            .register(tool_child.id.clone(), drive_token.clone());

        let enforced = enforce_subagent_deadlines(
            &stores,
            time::OffsetDateTime::now_utc(),
            0,
            &CancellationToken::new(),
        )
        .await?;
        assert_eq!(enforced, 1);
        assert!(
            drive_token.is_cancelled(),
            "cancelling the tool subtree must trip its registered confirm-drive token",
        );
        Ok(())
    }

    /// Item 2: when the execution future ignores cancellation past the
    /// abort grace and is force-dropped, any attempt the drop orphaned
    /// (the timeout fail left it open FOR the live worker) must be
    /// settled — no live worker remains to close it.
    #[tokio::test]
    async fn force_drop_settles_the_open_attempt() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_sdk_foundation::audit::AuditProvenance;
        use agent_server::journal::turn_attempt::{OpenAttemptParams, TurnAttemptOutcome};

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let static_resolver = Arc::new(StaticProviderResolver::new());
        static_resolver.set_fallback(Arc::clone(&provider) as Arc<dyn LlmProvider>)?;
        let runtime = Arc::new(
            ExecutionRuntime::new(
                Arc::new(HangOnceProviderResolver {
                    inner: static_resolver,
                    hung_once: std::sync::atomic::AtomicBool::new(false),
                }),
                Arc::new(NoopToolExecutor),
                Arc::new(AllowAllConfirmationPolicy),
            )
            .with_subagent_spawn_selector(Arc::new(DeadlineSpawnSelector)),
        );
        let host = ServiceHost::new(config, sample_registry(), runtime.clone())?;
        let stores = host.stores().clone();

        let now = time::OffsetDateTime::now_utc();
        let parent_thread = ThreadId::from_string("t-force-drop-attempt");
        let (_parent, _invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 500, now).await?;
        let worker = WorkerId::from_string("w-force-drop");
        let lease = LeaseId::new();
        let acquired = stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                now + time::Duration::seconds(600),
                now,
            )
            .await?
            .context("child root must acquire")?;

        // Simulates an attempt opened by execution before it wedged in
        // a token-blind await outside `call_llm_with_retry`.
        let attempt = stores
            .attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: child_root.id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("test", "test"),
                request_blob: serde_json::json!({ "user_prompt": "wedged turn" }),
                now,
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;

        // The 500ms deadline fires on a heartbeat tick while the
        // resolver hangs token-blind; after the 3s grace the future is
        // force-dropped and the branch must settle the attempt.
        run_task_with_heartbeat(
            acquired,
            &worker,
            &stores,
            runtime,
            &CancellationToken::new(),
            time::Duration::seconds(30),
            std::time::Duration::from_millis(50),
        )
        .await;

        let failed = stores
            .task_store
            .get(&child_root.id)
            .await?
            .context("child root exists")?;
        assert_eq!(failed.status, TaskStatus::Failed);
        let error = failed.last_error.unwrap_or_default();
        assert!(
            error.contains("subagent timed out after 500ms"),
            "timed-out child must carry the timeout message, got {error:?}",
        );

        let attempts = stores.attempt_store.list_by_task(&child_root.id).await?;
        let reread = attempts
            .into_iter()
            .find(|row| row.id == attempt.id)
            .context("opened attempt still listed")?;
        assert!(
            reread.is_closed(),
            "the force-drop branch must settle the attempt the drop orphaned",
        );
        assert_eq!(reread.outcome, Some(TurnAttemptOutcome::Cancelled));
        Ok(())
    }

    /// Item 3: one sweep tick drains an expired-lease backlog larger
    /// than the store's per-call batch, so no ghost-Running rows
    /// survive into the deadline pass that runs later in the same
    /// tick.
    #[tokio::test]
    async fn sweep_tick_drains_more_than_one_lease_batch() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::store::LEASE_RELEASE_BATCH;

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = subagent_timeout_runtime(provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime)?;
        let stores = host.stores().clone();

        let now = time::OffsetDateTime::now_utc();
        let backlog = LEASE_RELEASE_BATCH + 44;
        let mut task_ids = Vec::with_capacity(backlog);
        for index in 0..backlog {
            let task = AgentTask::new_root_turn(
                ThreadId::from_string(format!("t-drain-{index}")),
                now - time::Duration::seconds(10),
                3,
            );
            let task_id = task.id.clone();
            stores.task_store.submit_root_turn(task).await?;
            stores
                .task_store
                .try_acquire_task(
                    &task_id,
                    WorkerId::from_string(format!("w-drain-{index}")),
                    LeaseId::new(),
                    now - time::Duration::seconds(5),
                    now - time::Duration::seconds(9),
                )
                .await?
                .context("row claims")?;
            task_ids.push(task_id);
        }

        let drained = drain_expired_leases(&stores, now).await?;
        assert_eq!(
            drained, backlog,
            "one sweep tick must drain the whole backlog, not one batch",
        );
        for task_id in &task_ids {
            let row = stores
                .task_store
                .get(task_id)
                .await?
                .context("row exists")?;
            assert_eq!(
                row.status,
                TaskStatus::Pending,
                "no ghost-Running rows may survive the drain",
            );
        }
        Ok(())
    }

    /// Runtime whose FIRST provider resolve hangs forever (token
    /// blind); later resolves delegate to `provider`.
    fn hang_once_runtime(provider: &Arc<SubagentScriptProvider>) -> Result<Arc<ExecutionRuntime>> {
        let static_resolver = Arc::new(StaticProviderResolver::new());
        static_resolver.set_fallback(Arc::clone(provider) as Arc<dyn LlmProvider>)?;
        Ok(Arc::new(
            ExecutionRuntime::new(
                Arc::new(HangOnceProviderResolver {
                    inner: static_resolver,
                    hung_once: std::sync::atomic::AtomicBool::new(false),
                }),
                Arc::new(NoopToolExecutor),
                Arc::new(AllowAllConfirmationPolicy),
            )
            .with_subagent_spawn_selector(Arc::new(DeadlineSpawnSelector)),
        ))
    }

    /// Open a bare test attempt row for `task_id`.
    async fn open_test_attempt(
        stores: &StoreRegistry,
        task_id: &agent_server::journal::task::AgentTaskId,
        attempt_number: u32,
    ) -> Result<agent_server::journal::turn_attempt::TurnAttempt> {
        use agent_sdk_foundation::audit::AuditProvenance;
        use agent_server::journal::turn_attempt::OpenAttemptParams;

        stores
            .attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: task_id.clone(),
                attempt_number,
                provenance: AuditProvenance::new("test", "test"),
                request_blob: serde_json::json!({ "user_prompt": "test attempt" }),
                now: time::OffsetDateTime::now_utc(),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await
    }

    /// Spawn a wedged OLD-worker run for `child_root_id`: acquired on
    /// a short (200ms) lease, heartbeats injected to reject terminally
    /// (the per-task token trips on the first ~50ms tick), execution
    /// hung token-blind in the resolver. Returns the run's join
    /// handle; it finishes once the abort grace expires and the
    /// future is force-dropped.
    async fn spawn_wedged_old_worker_run(
        stores: &StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        child_root_id: &agent_server::journal::task::AgentTaskId,
        old_worker: WorkerId,
    ) -> Result<tokio::task::JoinHandle<()>> {
        let now = time::OffsetDateTime::now_utc();
        let acquired_old = stores
            .task_store
            .try_acquire_task(
                child_root_id,
                old_worker.clone(),
                LeaseId::new(),
                now + time::Duration::milliseconds(200),
                now,
            )
            .await?
            .context("old worker must acquire")?;
        let flaky = Arc::new(
            FlakyTaskStore::new(Arc::clone(&stores.task_store)).rejecting_heartbeats(1_000),
        );
        let mut flaky_stores = stores.clone();
        flaky_stores.task_store = flaky as Arc<dyn AgentTaskStore>;
        Ok(tokio::spawn(async move {
            run_task_with_heartbeat(
                acquired_old,
                &old_worker,
                &flaky_stores,
                runtime,
                &CancellationToken::new(),
                time::Duration::milliseconds(300),
                std::time::Duration::from_millis(50),
            )
            .await;
        }))
    }

    /// Round 5 (MAJOR): the force-drop attempt settlement must be
    /// guarded by the still-owned/terminal re-read. Double-fault
    /// scenario: the OLD worker wedges token-blind, its lease is
    /// revoked (terminal heartbeat rejection), the row requeues and a
    /// SUCCESSOR worker acquires it and opens its own attempt while
    /// the old worker is still inside its abort grace. The old
    /// worker's force-drop must NOT stamp the successor's live attempt
    /// Cancelled/zero — the successor closes it with real usage and
    /// completes the turn.
    #[tokio::test]
    async fn force_drop_skips_attempts_owned_by_a_successor() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::turn_attempt::{CloseAttemptParams, TurnAttemptOutcome};

        let config = ServiceConfig::default();
        let provider = Arc::new(SubagentScriptProvider::new());
        let runtime = hang_once_runtime(&provider)?;
        let host = ServiceHost::new(config, sample_registry(), runtime.clone())?;
        let stores = host.stores().clone();

        // Generous budget — this test is about lease loss, not the
        // deadline.
        let now = time::OffsetDateTime::now_utc();
        let parent_thread = ThreadId::from_string("t-force-drop-successor");
        let (_parent, _invocation, child_root) =
            persist_subagent_fixture(&stores, &parent_thread, 600_000, now).await?;

        // OLD worker: short real lease, heartbeats injected to reject
        // terminally (simulating the lease being revoked out from
        // under it), execution wedged token-blind in the resolver.
        let handle = spawn_wedged_old_worker_run(
            &stores,
            Arc::clone(&runtime),
            &child_root.id,
            WorkerId::from_string("w-old"),
        )
        .await?;

        // Mid-grace: the old worker's first heartbeat (~50ms) was
        // terminally rejected, tripping its per-task token; its 200ms
        // lease expires unextended. Requeue the row and hand it to a
        // successor, which opens its own attempt and starts working.
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        let swept = stores
            .task_store
            .release_expired_leases(time::OffsetDateTime::now_utc())
            .await?;
        assert!(
            swept.iter().any(|record| record.id == child_root.id),
            "the old worker's expired lease must have been requeued, got {swept:?}",
        );
        let new_worker = WorkerId::from_string("w-new");
        let new_lease = LeaseId::new();
        stores
            .task_store
            .try_acquire_task(
                &child_root.id,
                new_worker.clone(),
                new_lease.clone(),
                time::OffsetDateTime::now_utc() + time::Duration::seconds(600),
                time::OffsetDateTime::now_utc(),
            )
            .await?
            .context("successor worker must acquire the requeued row")?;
        let successor_attempt = open_test_attempt(&stores, &child_root.id, 2).await?;

        // The old worker's grace expires (~3s after its trip) and it
        // force-drops; the guarded settlement must observe the row
        // Running under the SUCCESSOR and leave its attempt alone.
        handle.await?;
        let attempts = stores.attempt_store.list_by_task(&child_root.id).await?;
        let reread = attempts
            .iter()
            .find(|row| row.id == successor_attempt.id)
            .context("successor attempt listed")?;
        assert!(
            !reread.is_closed(),
            "the old worker's force-drop must not touch the successor's live attempt",
        );

        // The successor's real-usage close and completion land — a
        // genuinely recovered turn is not failed terminally.
        let closed = stores
            .attempt_store
            .close_attempt(
                &successor_attempt.id,
                CloseAttemptParams {
                    response_blob: serde_json::json!({ "recovered": true }),
                    response_id: None,
                    response_model: None,
                    stop_reason: None,
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 111,
                    output_tokens: 222,
                    cached_input_tokens: 0,
                },
                time::OffsetDateTime::now_utc(),
            )
            .await
            .context("successor real-usage close must not hit AlreadyClosed")?;
        assert_eq!(closed.outcome, Some(TurnAttemptOutcome::Success));
        assert_eq!(closed.input_tokens, Some(111));

        let (completed, _parent_row) = stores
            .task_store
            .complete_task(
                &child_root.id,
                &new_worker,
                &new_lease,
                time::OffsetDateTime::now_utc(),
            )
            .await
            .context("successor completes the recovered turn")?;
        assert_eq!(completed.status, TaskStatus::Completed);
        Ok(())
    }
}
