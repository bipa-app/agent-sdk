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
use super::relay::{RelayScheduler, RelaySchedulerConfig};
use super::runtime::ExecutionRuntime;
use super::stores::StoreRegistry;
use agent_server::journal::relay::RetryBackoff;

// ─────────────────────────────────────────────────────────────────────
// ServiceHost
// ─────────────────────────────────────────────────────────────────────

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

        info!(
            storage_backend = self.stores.backend_name(),
            pool_size = self.config.worker.pool_size,
            lease_duration_secs = self.config.worker.lease_duration_secs,
            sweep_interval_secs = self.config.worker.sweep_interval_secs,
            acquisition_interval_secs = self.config.worker.acquisition_interval_secs,
            grpc_enabled = self.config.transport.grpc_enabled,
            http_enabled = self.config.transport.http_enabled,
            relay_enabled = self.config.relay.enabled,
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
                Arc::clone(&self.runtime),
                self.config.worker.lease_duration(),
                self.config.worker.acquisition_interval(),
                Arc::clone(&self.health),
                self.shutdown.clone(),
            ));
            worker_handles.push(handle);
        }

        // ── Spawn relay scheduler ────────────────────────────────
        let relay_handle = if self.config.relay.enabled {
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
            Some(scheduler.spawn(self.shutdown.clone()))
        } else {
            self.health
                .set_latency_layer(LatencyLayerHealth::NotConfigured);
            None
        };

        // ── Mark healthy ─────────────────────────────────────────
        self.health.set_sweep_alive(true);
        self.health.set_workers_alive(true);
        self.health.set_core(super::health::CoreHealth::Healthy);

        info!(
            pool_size,
            relay_enabled = self.config.relay.enabled,
            "service host ready",
        );

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

        if let Some(handle) = relay_handle
            && let Err(err) = handle.shutdown().await
        {
            warn!(error = %err, "relay scheduler exited with error");
        }

        info!("service host stopped");
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
    runtime: Arc<ExecutionRuntime>,
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
                        if let Err(err) =
                            execute_acquired_task(task, &stores, Arc::clone(&runtime), &cancel).await
                        {
                            warn!(%worker_id, error = %err, "task execution failed");
                        }
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
}
