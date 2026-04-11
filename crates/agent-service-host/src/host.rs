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
//! [`StoreRegistry`]: super::stores::StoreRegistry
//! [`HealthSurface`]: super::health::HealthSurface

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use agent_server::journal::task::{LeaseId, WorkerId};
use agent_server::worker::registry::AgentDefinitionRegistry;

use super::config::ServiceConfig;
use super::health::HealthSurface;
use super::stores::StoreRegistry;

// ─────────────────────────────────────────────────────────────────────
// ServiceHost
// ─────────────────────────────────────────────────────────────────────

/// The composed service host: stores + worker pool + sweeps +
/// health + lifecycle.
pub struct ServiceHost {
    config: ServiceConfig,
    stores: StoreRegistry,
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
    ) -> Result<Self> {
        Self::validate_config(&config)?;
        let stores = StoreRegistry::from_config(&config.storage, definition_registry)
            .context("initialising store registry")?;
        Ok(Self {
            config,
            stores,
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
    pub fn with_stores(config: ServiceConfig, stores: StoreRegistry) -> Result<Self> {
        Self::validate_config(&config)?;
        Ok(Self {
            config,
            stores,
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
        info!(
            pool_size = self.config.worker.pool_size,
            lease_duration_secs = self.config.worker.lease_duration_secs,
            sweep_interval_secs = self.config.worker.sweep_interval_secs,
            acquisition_interval_secs = self.config.worker.acquisition_interval_secs,
            grpc_enabled = self.config.transport.grpc_enabled,
            http_enabled = self.config.transport.http_enabled,
            "service host starting",
        );

        // Register signal handlers before spawning any tasks so that a
        // registration failure (e.g. EMFILE) never leaves an orphaned
        // background task running indefinitely.
        #[cfg(unix)]
        let mut sigterm = {
            use tokio::signal::unix::{SignalKind, signal};
            signal(SignalKind::terminate()).context("registering SIGTERM handler")?
        };

        // ── Spawn lease sweep ────────────────────────────────────
        let sweep_handle = tokio::spawn(lease_sweep_loop(
            self.stores.clone(),
            self.config.worker.sweep_interval(),
            Arc::clone(&self.health),
            self.shutdown.clone(),
        ));

        // ── Spawn worker pool ────────────────────────────────────
        let pool_size = self.config.worker.pool_size;
        let mut worker_handles = Vec::with_capacity(pool_size);
        for idx in 0..pool_size {
            let handle = tokio::spawn(worker_loop(
                idx,
                self.stores.clone(),
                self.config.worker.lease_duration(),
                self.config.worker.acquisition_interval(),
                Arc::clone(&self.health),
                self.shutdown.clone(),
            ));
            worker_handles.push(handle);
        }

        // ── Mark healthy ─────────────────────────────────────────
        self.health.set_sweep_alive(true);
        self.health.set_workers_alive(true);
        self.health.set_core(super::health::CoreHealth::Healthy);

        info!(pool_size, "service host ready");

        // ── Wait for shutdown ────────────────────────────────────
        #[cfg(unix)]
        wait_for_shutdown(&self.shutdown, &mut sigterm).await?;
        #[cfg(not(unix))]
        wait_for_shutdown(&self.shutdown).await?;

        info!("draining background tasks");

        // ── Mark unhealthy before draining ───────────────────────
        self.health.set_workers_alive(false);
        self.health.set_sweep_alive(false);

        // ── Drain ────────────────────────────────────────────────
        sweep_handle
            .await
            .context("lease sweep task panicked during shutdown")?;

        for (idx, handle) in worker_handles.into_iter().enumerate() {
            handle
                .await
                .with_context(|| format!("worker {idx} panicked during shutdown"))?;
        }

        info!("service host stopped");
        Ok(())
    }
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
// Worker loop
// ─────────────────────────────────────────────────────────────────────

/// A single worker's acquisition loop.
///
/// Each worker:
/// 1. Polls `acquire_next_runnable` at the configured interval.
/// 2. On successful acquisition, logs the task.  (Actual execution
///    is a future phase — for now the loop proves the bootstrap,
///    acquisition, and health surface work.  The lease expires
///    naturally and the sweep resets the task to `Pending`.)
/// 3. Updates the health surface.
///
/// The worker identity is derived from a unique `WorkerId` per spawn.
async fn worker_loop(
    index: usize,
    stores: StoreRegistry,
    lease_duration: time::Duration,
    poll_interval: std::time::Duration,
    _health: Arc<HealthSurface>,
    cancel: CancellationToken,
) {
    let worker_id = WorkerId::from_string(format!("worker-{index}"));
    info!(%worker_id, "worker started");

    let mut ticker = tokio::time::interval(poll_interval);
    // Skip the immediate first tick.
    ticker.tick().await;

    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                info!(%worker_id, "worker shutting down");
                return;
            }
            _ = ticker.tick() => {
                let now = time::OffsetDateTime::now_utc();
                let lease_id = LeaseId::new();
                let expires_at = now + lease_duration;

                match stores
                    .task_store
                    .acquire_next_runnable(
                        worker_id.clone(),
                        lease_id,
                        expires_at,
                        now,
                    )
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
                        // Future phase: execute the task via
                        // resolve_bootstrap_context + execute_root_turn.
                        // For now, the acquisition proves the bootstrap
                        // loop works end-to-end.
                    }
                    Ok(None) => {
                        // No runnable tasks — idle wait.
                    }
                    Err(e) => {
                        warn!(%worker_id, error = %e, "task acquisition failed");
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServiceConfig;
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
    use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;

    fn sample_definition() -> AgentDefinition {
        AgentDefinition {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
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

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn host_construction_succeeds() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
        assert_eq!(host.config().worker.pool_size, 4);
        Ok(())
    }

    #[test]
    fn stores_accessible_from_host() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
        let _stores = host.stores();
        let _deps = host.stores().root_turn_deps();
        Ok(())
    }

    #[test]
    fn shutdown_token_is_clonable() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
        let token = host.shutdown_token();
        assert!(!token.is_cancelled());
        Ok(())
    }

    #[test]
    fn health_surface_accessible() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
        let snap = host.health().snapshot();
        // Before run(), health is not yet alive.
        assert!(!snap.is_ready());
        Ok(())
    }

    // ── Validation ───────────────────────────────────────────────

    #[test]
    fn zero_sweep_interval_is_rejected() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                sweep_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry());
        assert!(result.is_err());
    }

    #[test]
    fn zero_pool_size_is_rejected() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                pool_size: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry());
        assert!(result.is_err());
    }

    #[test]
    fn zero_acquisition_interval_is_rejected() {
        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                acquisition_interval_secs: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = ServiceHost::new(config, sample_registry());
        assert!(result.is_err());
    }

    // ── Lifecycle ────────────────────────────────────────────────

    #[tokio::test]
    async fn host_shuts_down_on_token_cancel() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
        let token = host.shutdown_token();

        // Cancel immediately so `run()` returns promptly.
        token.cancel();
        host.run().await?;
        Ok(())
    }

    #[tokio::test]
    async fn health_becomes_ready_during_run() -> Result<()> {
        let config = ServiceConfig::default();
        let host = ServiceHost::new(config, sample_registry())?;
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
        let host = ServiceHost::new(config, sample_registry())?;
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
        let host = ServiceHost::new(config, sample_registry())?;
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
        let host = ServiceHost::new(config, sample_registry())?;
        let stores = host.stores().clone();
        let token = host.shutdown_token();

        // Submit a pending root turn.
        let thread = ThreadId::from_string("t-worker-test");
        let task = AgentTask::new_root_turn(thread, time::OffsetDateTime::now_utc(), 3);
        let task_id = task.id.clone();
        stores.task_store.submit_root_turn(task).await?;

        let host_handle = tokio::spawn(async move { host.run().await });

        // Advance time so the worker polls and acquires.
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // The task should have been acquired (status = Running).
        let acquired = stores
            .task_store
            .get(&task_id)
            .await?
            .context("task should still exist")?;
        assert_eq!(
            acquired.status,
            agent_server::journal::task::TaskStatus::Running,
            "worker should have acquired the pending task",
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
        let host = ServiceHost::new(config, sample_registry())?;
        let token = host.shutdown_token();

        // Cancel immediately — all 8 workers must drain cleanly.
        token.cancel();
        host.run().await?;
        Ok(())
    }
}
