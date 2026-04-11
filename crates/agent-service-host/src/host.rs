//! Service host: startup, background tasks, and graceful shutdown.
//!
//! [`ServiceHost`] composes a [`StoreRegistry`] with background sweep
//! tasks and a shutdown lifecycle.  It is the single struct a deploy
//! target creates:
//!
//! ```ignore
//! let host = ServiceHost::new(config)?;
//! host.run().await?;   // blocks until shutdown signal
//! ```
//!
//! Transport layers (gRPC, HTTP) are not started by `run()` — they
//! compose on top via [`ServiceHost::stores`]:
//!
//! ```ignore
//! let host = ServiceHost::new(config)?;
//! let grpc = GrpcTransport::new(host.stores());
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
//! | Task | Interval | Purpose |
//! |------|----------|---------|
//! | Lease sweep | [`WorkerConfig::sweep_interval`] | Release expired leases via recovery matrix |
//!
//! All background tasks respect the host's [`CancellationToken`] and
//! drain cleanly on shutdown.
//!
//! [`WorkerConfig::sweep_interval`]: super::config::WorkerConfig::sweep_interval
//! [`StoreRegistry`]: super::stores::StoreRegistry

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use agent_server::worker::registry::AgentDefinitionRegistry;

use super::config::ServiceConfig;
use super::stores::StoreRegistry;

// ─────────────────────────────────────────────────────────────────────
// ServiceHost
// ─────────────────────────────────────────────────────────────────────

/// The composed service host: stores + background tasks + lifecycle.
pub struct ServiceHost {
    config: ServiceConfig,
    stores: StoreRegistry,
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
    /// Returns an error if store initialisation fails.
    pub fn new(
        config: ServiceConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
    ) -> Result<Self> {
        let stores = StoreRegistry::from_config(&config.storage, definition_registry)
            .context("initialising store registry")?;
        Ok(Self {
            config,
            stores,
            shutdown: CancellationToken::new(),
        })
    }

    /// Build a host with a pre-built store registry.
    ///
    /// Useful in tests where stores are pre-populated.
    #[must_use]
    pub fn with_stores(config: ServiceConfig, stores: StoreRegistry) -> Self {
        Self {
            config,
            stores,
            shutdown: CancellationToken::new(),
        }
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

    /// Token that, when cancelled, triggers graceful shutdown.
    #[must_use]
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Run the service host until a shutdown signal is received.
    ///
    /// This method:
    /// 1. Logs startup configuration.
    /// 2. Spawns background sweep tasks.
    /// 3. Waits for shutdown signal (`SIGINT` / `SIGTERM` / token
    ///    cancellation).
    /// 4. Cancels background tasks and waits for them to drain.
    ///
    /// # Errors
    /// Returns an error if a background task panics or if shutdown
    /// coordination fails.
    pub async fn run(self) -> Result<()> {
        info!(
            pool_size = self.config.worker.pool_size,
            lease_duration_secs = self.config.worker.lease_duration_secs,
            sweep_interval_secs = self.config.worker.sweep_interval_secs,
            grpc_enabled = self.config.transport.grpc_enabled,
            http_enabled = self.config.transport.http_enabled,
            "service host starting",
        );

        let sweep_handle = tokio::spawn(lease_sweep_loop(
            self.stores.clone(),
            self.config.worker.sweep_interval(),
            self.shutdown.clone(),
        ));

        // Wait for shutdown signal or token cancellation.
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigterm =
                signal(SignalKind::terminate()).context("registering SIGTERM handler")?;

            tokio::select! {
                () = self.shutdown.cancelled() => {
                    info!("shutdown token cancelled");
                }
                result = tokio::signal::ctrl_c() => {
                    match result {
                        Ok(()) => info!("received SIGINT, shutting down"),
                        Err(e) => warn!(error = %e, "signal handler failed"),
                    }
                    self.shutdown.cancel();
                }
                _ = sigterm.recv() => {
                    info!("received SIGTERM, shutting down");
                    self.shutdown.cancel();
                }
            }
        }

        #[cfg(not(unix))]
        {
            tokio::select! {
                () = self.shutdown.cancelled() => {
                    info!("shutdown token cancelled");
                }
                result = tokio::signal::ctrl_c() => {
                    match result {
                        Ok(()) => info!("received SIGINT, shutting down"),
                        Err(e) => warn!(error = %e, "signal handler failed"),
                    }
                    self.shutdown.cancel();
                }
            }
        }

        info!("draining background tasks");

        // Give the sweep task a moment to finish its current iteration.
        if let Err(e) = sweep_handle.await {
            warn!(error = %e, "lease sweep task panicked during shutdown");
        }

        info!("service host stopped");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Background tasks
// ─────────────────────────────────────────────────────────────────────

/// Periodic lease-expiry sweep.
///
/// Runs [`AgentTaskStore::release_expired_leases`] at the configured
/// interval.  Respects the cancellation token for clean shutdown.
async fn lease_sweep_loop(
    stores: StoreRegistry,
    interval: std::time::Duration,
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
    async fn sweep_runs_at_least_once_before_shutdown() -> Result<()> {
        use agent_sdk_core::ThreadId;
        use agent_server::journal::task::AgentTask;

        let config = ServiceConfig {
            worker: crate::config::WorkerConfig {
                sweep_interval_secs: 1,
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
        let worker = agent_server::journal::task::WorkerId::from_string("w1");
        let lease = agent_server::journal::task::LeaseId::new();
        // Lease expires immediately.
        let expires = now - time::Duration::seconds(1);
        stores
            .task_store
            .try_acquire_task(&task_id, worker, lease, expires, now)
            .await?;

        // Run the host briefly.
        let host_handle = tokio::spawn(async move { host.run().await });

        // Give the sweep one cycle.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        token.cancel();
        host_handle.await??;

        // The expired lease should have been released — task is back to
        // Pending.
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
}
