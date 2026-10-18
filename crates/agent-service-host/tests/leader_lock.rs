//! `EpochRecovery::PostgresLeader`: one host per Postgres store runs
//! workers, a second one waits ready for the store's leader lock, and a
//! host whose lock connection dies stops.
//!
//! Runs with `TEST_DATABASE_URL` (or `DATABASE_URL`); without one the
//! tests pass without running, as the other Postgres host tests do. Each
//! test gets its own schema, so the lock key is its own too.

#![cfg(feature = "postgres")]

use std::sync::Arc;
use std::time::Duration;

use agent_sdk_foundation::ThreadId;
use agent_server::journal::task::{AgentTask, LeaseId, TaskStatus, WorkerId};
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use agent_service_host::config::{
    EpochRecovery, PostgresStorageConfig, ServiceConfig, StorageBackend, StorageConfig,
    WorkerConfig,
};
use agent_service_host::health::{HealthSurface, Leadership};
use agent_service_host::host::ServiceHost;
use agent_service_host::runtime::{
    AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
};
use anyhow::{Context, Result, ensure};
use sqlx::Connection;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

struct Schema {
    database_url: String,
    schema: String,
}

impl Schema {
    async fn create() -> Result<Option<Self>> {
        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };
        let schema = format!("leader_lock_{}", uuid::Uuid::new_v4().simple());
        let mut admin = sqlx::PgConnection::connect(&database_url).await?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await?;
        Ok(Some(Self {
            database_url,
            schema,
        }))
    }

    fn config_with(&self, epoch_recovery: EpochRecovery) -> ServiceConfig {
        let mut config = self.config();
        config.worker.epoch_recovery = epoch_recovery;
        config
    }

    async fn admin(&self) -> Result<sqlx::PgConnection> {
        Ok(sqlx::PgConnection::connect(&self.database_url).await?)
    }

    /// Rename this schema's epoch table, from outside every host.
    async fn rename_epochs(&self, from: &str, to: &str) -> Result<()> {
        let mut admin = self.admin().await?;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "ALTER TABLE {}.{from} RENAME TO {to}",
            self.schema
        )))
        .execute(&mut admin)
        .await?;
        Ok(())
    }

    async fn epoch_superseded(&self, epochs_before: i64) -> Result<bool> {
        let mut admin = self.admin().await?;
        let superseded: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
            "SELECT count(*) FROM {}.agent_sdk_host_epochs WHERE superseded_at IS NOT NULL",
            self.schema
        )))
        .fetch_one(&mut admin)
        .await?;
        Ok(superseded >= epochs_before)
    }

    fn config(&self) -> ServiceConfig {
        ServiceConfig {
            storage: StorageConfig {
                backend: StorageBackend::Postgres,
                postgres: PostgresStorageConfig {
                    database_url: Some(self.database_url.clone()),
                    schema: Some(self.schema.clone()),
                    max_connections: 4,
                },
            },
            worker: WorkerConfig {
                lease_duration_secs: 3,
                heartbeat_interval_secs: 1,
                sweep_interval_secs: 1,
                epoch_recovery: EpochRecovery::PostgresLeader,
                ..WorkerConfig::default()
            },
            ..ServiceConfig::default()
        }
    }

    /// End the backend that holds this schema's leader lock.
    async fn kill_the_lock_holder(&self) -> Result<u64> {
        let mut admin = sqlx::PgConnection::connect(&self.database_url).await?;
        let killed = sqlx::query(
            "SELECT pg_terminate_backend(pid) FROM pg_locks \
             WHERE locktype = 'advisory' AND objsubid = 2 AND granted \
               AND classid = (x'41474C44'::int)::oid \
               AND objid = ((hashtext($1)::bigint) & 4294967295)::oid",
        )
        .bind(&self.schema)
        .fetch_all(&mut admin)
        .await?;
        Ok(killed.len() as u64)
    }
}

impl Drop for Schema {
    fn drop(&mut self) {
        let url = self.database_url.clone();
        let schema = self.schema.clone();
        let _ = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                let Ok(mut conn) = sqlx::PgConnection::connect(&url).await else {
                    return;
                };
                let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                    "DROP SCHEMA IF EXISTS {schema} CASCADE"
                )))
                .execute(&mut conn)
                .await;
            });
        })
        .join();
    }
}

struct Running {
    health: Arc<HealthSurface>,
    shutdown: CancellationToken,
    run: JoinHandle<Result<()>>,
}

fn host(config: ServiceConfig) -> Result<ServiceHost> {
    let definition = AgentDefinition {
        provider: "none".into(),
        model: "none".into(),
        system_prompt: String::new(),
        max_tokens: 16,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    };
    let runtime = Arc::new(ExecutionRuntime::new(
        Arc::new(StaticProviderResolver::new()),
        Arc::new(NoopToolExecutor),
        Arc::new(AllowAllConfirmationPolicy),
    ));
    ServiceHost::new(
        config,
        Arc::new(InMemoryAgentDefinitionRegistry::new(definition)),
        runtime,
    )
}

fn start(config: ServiceConfig) -> Result<Running> {
    let host = host(config)?;
    Ok(Running {
        health: Arc::clone(host.health()),
        shutdown: host.shutdown_token(),
        run: tokio::spawn(host.run()),
    })
}

async fn leadership_becomes(health: &HealthSurface, wanted: Leadership) -> Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let now = health.snapshot().leadership;
        if now == wanted {
            return Ok(());
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "leadership stayed {now:?}, not {wanted:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_host_waits_ready_for_the_leader_and_takes_over_when_it_stops() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let first = start(schema.config())?;
    leadership_becomes(&first.health, Leadership::Leader)
        .await
        .context("the first host leads")?;

    let second = start(schema.config())?;
    leadership_becomes(&second.health, Leadership::Standby).await?;
    let standby = second.health.snapshot();
    ensure!(
        standby.is_ready() && !standby.worker_pool_alive,
        "a standby host is ready, with no workers: {standby:?}"
    );
    tokio::time::sleep(Duration::from_secs(2)).await;
    let still = second.health.snapshot();
    ensure!(
        still.leadership == Leadership::Standby && !still.worker_pool_alive,
        "a standby host took over while the first host leads: {still:?}"
    );

    first.shutdown.cancel();
    first.run.await??;
    leadership_becomes(&second.health, Leadership::Leader)
        .await
        .context("the second host leads once the first one stops")?;

    second.shutdown.cancel();
    second.run.await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_host_whose_lock_connection_dies_stops() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let host = start(schema.config())?;
    leadership_becomes(&host.health, Leadership::Leader)
        .await
        .context("the host leads")?;

    let killed = schema.kill_the_lock_holder().await?;
    ensure!(killed == 1, "expected one lock holder, ended {killed}");
    leadership_becomes(&host.health, Leadership::Lost).await?;
    ensure!(
        !host.health.snapshot().is_ready(),
        "a host that lost its lock is not ready"
    );
    let stopped = tokio::time::timeout(Duration::from_secs(20), host.run)
        .await
        .context("the host stops after it loses its lock")??;
    ensure!(
        stopped.is_err_and(|error| error.to_string().contains("lost the store's leader lock")),
        "a host that lost its lock reports it, unlike a requested stop"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_standby_host_stops_on_shutdown_without_beginning_an_epoch() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let leader = start(schema.config())?;
    leadership_becomes(&leader.health, Leadership::Leader).await?;
    let standby = start(schema.config())?;
    leadership_becomes(&standby.health, Leadership::Standby).await?;

    standby.shutdown.cancel();
    let stopped = tokio::time::timeout(Duration::from_secs(10), standby.run)
        .await
        .context("a standby host stops on shutdown")??;
    stopped.context("a standby host that is shut down stops cleanly")?;
    let mut admin = sqlx::PgConnection::connect(&schema.database_url).await?;
    let epochs: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT count(*) FROM {}.agent_sdk_host_epochs",
        schema.schema
    )))
    .fetch_one(&mut admin)
    .await?;
    ensure!(
        epochs == 1,
        "only the leader began an epoch, found {epochs}"
    );

    leader.shutdown.cancel();
    leader.run.await??;
    Ok(())
}

/// Submit a root turn and lease it for an hour through `host`'s own store,
/// so it carries `host`'s epoch and only an epoch reclaim (not expiry) can
/// take it.
async fn lease_for_an_hour(host: &ServiceHost, name: &str) -> Result<(AgentTask, LeaseId)> {
    let now = time::OffsetDateTime::now_utc();
    let root = AgentTask::new_root_turn(ThreadId::from_string(name), now, 3);
    let stores = host.stores();
    let _ = stores
        .thread_store
        .get_or_create(&root.thread_id, now)
        .await?;
    let _ = stores.task_store.submit_root_turn(root.clone()).await?;
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &root.id,
            WorkerId::from_string("seed-worker"),
            lease.clone(),
            now + time::Duration::hours(1),
            now,
        )
        .await?
        .context("lease the seeded root")?;
    Ok((root, lease))
}

/// Whether the row still holds `lease`.
async fn holds(
    host_stores: &agent_service_host::stores::StoreRegistry,
    task: &AgentTask,
    lease: &LeaseId,
) -> Result<bool> {
    let row = host_stores.task_store.get(&task.id).await?.context("row")?;
    Ok(row.status == TaskStatus::Running && row.lease_id.as_ref() == Some(lease))
}

/// B3: a host dropped after it took the lock (a failed start, an embedder
/// that bails) frees it, so the store is not left with a leader that runs
/// no workers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_host_dropped_after_it_took_the_lock_frees_it() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let first = host(schema.config())?;
    first.initialize().await?;
    first.begin_epoch().await?;
    let second = start(schema.config())?;
    leadership_becomes(&second.health, Leadership::Standby).await?;
    drop(first);
    leadership_becomes(&second.health, Leadership::Leader)
        .await
        .context("the lock of a dropped host is freed")?;
    second.shutdown.cancel();
    second.run.await??;
    Ok(())
}

/// B3: a `begin_epoch` that failed after it took the lock is retried with
/// the same lock, not with a second connection that waits on it forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retried_begin_epoch_keeps_the_lock_it_took() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let host = host(schema.config())?;
    host.initialize().await?;
    schema
        .rename_epochs("agent_sdk_host_epochs", "agent_sdk_host_epochs_away")
        .await?;
    ensure!(
        host.begin_epoch().await.is_err(),
        "the epoch cannot begin without its table"
    );
    ensure!(host.health().snapshot().leadership == Leadership::Leader);
    schema
        .rename_epochs("agent_sdk_host_epochs_away", "agent_sdk_host_epochs")
        .await?;
    tokio::time::timeout(Duration::from_secs(10), host.begin_epoch())
        .await
        .context("the retry returns")??;
    ensure!(host.health().snapshot().leadership == Leadership::Leader);
    Ok(())
}

/// B4: the successor of a dead leader takes its leases by epoch, at once
/// after the detection window, not when they expire an hour later.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_successor_reclaims_a_dead_leaders_lease_by_epoch_not_expiry() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let old = host(schema.config())?;
    old.initialize().await?;
    old.begin_epoch().await?;
    let (task, lease) = lease_for_an_hour(&old, "leader-reclaim").await?;
    let inspect = old.stores().clone();
    drop(old);

    let new = start(schema.config())?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while holds(&inspect, &task, &lease).await? {
        ensure!(
            tokio::time::Instant::now() < deadline,
            "the successor did not reclaim the dead leader's hour-long lease"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    new.shutdown.cancel();
    new.run.await??;
    Ok(())
}

/// B2: a leader does not take the live lease of a host that runs another
/// mode (a switch-over deploy): that lease waits for expiry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_leader_leaves_the_live_lease_of_a_host_in_another_mode() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let shared = host(schema.config_with(EpochRecovery::LeaseExpiry))?;
    shared.initialize().await?;
    shared.begin_epoch().await?;
    let (task, lease) = lease_for_an_hour(&shared, "leader-mixed").await?;

    let leader = start(schema.config())?;
    leadership_becomes(&leader.health, Leadership::Leader).await?;
    // Past the detection window and the reclaim.
    tokio::time::sleep(Duration::from_secs(9)).await;
    ensure!(
        holds(shared.stores(), &task, &lease).await?,
        "the leader took a live lease of a lease-expiry host"
    );
    leader.shutdown.cancel();
    leader.run.await??;
    Ok(())
}

/// B1: a leader that lost its lock cannot acquire, heartbeat or settle
/// once its successor began: the store refuses its superseded epoch,
/// whatever its own monitor has noticed yet.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_store_refuses_a_superseded_leaders_acquire_and_heartbeat() -> Result<()> {
    let Some(schema) = Schema::create().await? else {
        return Ok(());
    };
    let old = host(schema.config())?;
    old.initialize().await?;
    old.begin_epoch().await?;
    let (task, lease) = lease_for_an_hour(&old, "leader-fence").await?;
    let pending = AgentTask::new_root_turn(
        ThreadId::from_string("leader-fence-pending"),
        time::OffsetDateTime::now_utc(),
        3,
    );
    old.stores()
        .task_store
        .submit_root_turn(pending.clone())
        .await?;

    // The lock goes; the old host lives on. Its successor begins.
    ensure!(schema.kill_the_lock_holder().await? == 1);
    let new = start(schema.config())?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while !schema.epoch_superseded(1).await? {
        ensure!(
            tokio::time::Instant::now() < deadline,
            "the successor never began"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let now = time::OffsetDateTime::now_utc();
    let worker = WorkerId::from_string("seed-worker");
    let beat = old
        .stores()
        .task_store
        .heartbeat_task(
            &task.id,
            &worker,
            &lease,
            now + time::Duration::hours(1),
            None,
            now,
        )
        .await;
    ensure!(
        beat.as_ref()
            .is_err_and(|error| format!("{error:#}").contains("superseded")),
        "the superseded leader's heartbeat was accepted: {beat:?}"
    );
    let acquired = old
        .stores()
        .task_store
        .try_acquire_task(
            &pending.id,
            worker,
            LeaseId::new(),
            now + time::Duration::hours(1),
            now,
        )
        .await;
    ensure!(
        acquired
            .as_ref()
            .is_err_and(|error| format!("{error:#}").contains("superseded")),
        "the superseded leader acquired a task: {acquired:?}"
    );
    new.shutdown.cancel();
    new.run.await??;
    Ok(())
}
