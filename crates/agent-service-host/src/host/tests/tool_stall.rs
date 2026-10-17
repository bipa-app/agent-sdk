//! A stalled tool call's settle across a crash: the settle, its effect's
//! finish and its `ToolCallEnd` commit together or not at all, and a
//! restart after either outcome resumes the parent exactly once.
#![cfg(any(feature = "sqlite", all(feature = "postgres", feature = "failpoints")))]

#[cfg(feature = "failpoints")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "sqlite")]
use agent_sdk_foundation::TerminalReason;
use agent_sdk_foundation::{AgentEvent, ThreadId, ToolResult, ToolTier};
use agent_server::journal::SpawnChildrenExt;
use agent_server::journal::effect::{EffectKey, EffectState};
use agent_server::journal::task::{
    AgentTask, ChildSpawnSpec, LeaseId, SubmittedInputItem, TaskStatus, WorkerId,
};
#[cfg(feature = "sqlite")]
use agent_server::worker::ToolStallReport;
use agent_server::worker::{
    ActivityBeacon, StallSignal, StallTrip, ToolEventCollector, ToolTaskBootstrap,
    resolve_tool_bootstrap, run_tool_effect,
};

use crate::config::{StorageBackend, StorageConfig};
use crate::stores::StoreRegistry;

const TOOL: &str = "probe";

#[cfg(feature = "sqlite")]
fn storage(path: &std::path::Path) -> StorageConfig {
    StorageConfig {
        backend: StorageBackend::Sqlite {
            path: Some(path.display().to_string()),
            max_connections: None,
        },
        ..StorageConfig::default()
    }
}

async fn open(storage: &StorageConfig) -> Result<StoreRegistry> {
    let stores = StoreRegistry::from_config(storage, super::sample_registry())?;
    stores.initialize().await?;
    Ok(stores)
}

fn trip() -> StallTrip {
    StallTrip {
        budget: Duration::from_mins(15),
        silent_for: Duration::from_secs(901),
        last_progress_at: None,
    }
}

/// A root parked on one running `probe` tool child, and that child's
/// bootstrap with `signal` as its stall signal.
async fn running_probe(
    stores: &StoreRegistry,
    thread: &ThreadId,
    signal: &StallSignal,
    now: OffsetDateTime,
) -> Result<(AgentTask, ToolTaskBootstrap)> {
    let root = AgentTask::new_root_turn_with_input(
        thread.clone(),
        vec![SubmittedInputItem::Text {
            text: "run the probe".into(),
        }],
        now,
        3,
    );
    let root_id = root.id.clone();
    stores.task_store.submit_root_turn(root).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &root_id,
            worker.clone(),
            lease.clone(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the root acquires")?;
    let (parked, children) = stores
        .task_store
        .spawn_tool_children(
            &root_id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3)],
            super::pending_call_suspension_with_tier(thread, TOOL, ToolTier::Observe),
            None,
            Vec::new(),
            now,
        )
        .await?;
    let child = children.into_iter().next().context("one tool child")?;
    let running = stores
        .task_store
        .try_acquire_task(
            &child.id,
            WorkerId::new(),
            LeaseId::new(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the tool child acquires")?;
    let mut bootstrap =
        resolve_tool_bootstrap(running, stores.task_store.as_ref(), ActivityBeacon::new()).await?;
    bootstrap.stall = signal.clone();
    Ok((parked, bootstrap))
}

fn tool_call_ends(
    events: &[agent_server::journal::committed_event::CommittedEvent],
) -> Vec<ToolResult> {
    events
        .iter()
        .filter_map(|committed| {
            if let AgentEvent::ToolCallEnd { result, .. } = &committed.event {
                Some(result.clone())
            } else {
                None
            }
        })
        .collect()
}

/// A tool that stops itself as stalled while it runs, then never returns:
/// the run settles it as stalled. `arm` runs first, inside the tool, after
/// the effect's start has committed.
async fn run_stalling_probe(
    stores: &StoreRegistry,
    bootstrap: ToolTaskBootstrap,
    arm: impl FnOnce() + Send,
) -> Result<agent_server::worker::ToolTaskOutcome> {
    let signal = bootstrap.stall.clone();
    run_tool_effect(
        bootstrap,
        stores.task_store.as_ref(),
        &CancellationToken::new(),
        move |_call, _collector: ToolEventCollector| async move {
            arm();
            signal.trip(trip());
            std::future::pending::<Result<ToolResult>>().await
        },
        OffsetDateTime::now_utc(),
    )
    .await
}

/// The stalled settle commits; a process that starts afterwards finds the
/// tool child failed with its stall report, the parent runnable, and one
/// `ToolCallEnd`.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn a_committed_stall_survives_a_restart() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("stall-restart.db");
    let thread = ThreadId::from_string("t-stall-restart");
    let signal = StallSignal::new();
    let (parent, child_id) = {
        let stores = open(&storage(&path)).await?;
        let (parent, bootstrap) =
            running_probe(&stores, &thread, &signal, OffsetDateTime::now_utc()).await?;
        let child_id = bootstrap.task_id.clone();
        run_stalling_probe(&stores, bootstrap, || {}).await?;
        (parent, child_id)
    };

    let restarted = open(&storage(&path)).await?;
    let child = restarted
        .task_store
        .get(&child_id)
        .await?
        .context("the tool child row")?;
    assert_eq!(child.status, TaskStatus::Failed);
    assert_eq!(child.terminal_reason, Some(TerminalReason::WatchdogStall));
    let parent = restarted
        .task_store
        .get(&parent.id)
        .await?
        .context("the parent row")?;
    assert_eq!(
        parent.status,
        TaskStatus::Pending,
        "the parent's wait ended"
    );
    // The parent's resume reads the typed result the stalled child keeps,
    // not just its error text.
    let continuation = parent
        .state
        .continuation()
        .context("the woken parent keeps its continuation")?
        .clone()
        .unwrap_validated()
        .map_err(|error| anyhow::anyhow!("continuation: {error}"))?;
    let outcomes = agent_server::worker::aggregate_child_outcomes(
        &parent,
        &continuation,
        restarted.task_store.as_ref(),
    )
    .await?;
    let [(call_id, read)] = outcomes.as_slice() else {
        anyhow::bail!("one child outcome, got {outcomes:?}");
    };
    assert_eq!(call_id, "call_probe");
    assert!(!read.success);
    let read_report: ToolStallReport = serde_json::from_value(
        read.data
            .clone()
            .context("the parent reads the stall report")?,
    )?;
    assert_eq!(read_report.outcome, "stalled");
    let ends = tool_call_ends(&restarted.event_repo.get_events(&thread).await?);
    assert_eq!(ends.len(), 1, "{ends:?}");
    let report: ToolStallReport =
        serde_json::from_value(ends[0].data.clone().context("the stall report")?)?;
    assert_eq!(report.outcome, "stalled");
    let effect = restarted
        .task_store
        .effect(&EffectKey::tool(&child_id, "call_probe"))
        .await?
        .context("the probe's effect record")?;
    assert_eq!(
        effect.state,
        EffectState::Interrupted {
            captured_output: None
        },
        "a stalled tool may have acted: its record says interrupted",
    );
    Ok(())
}

/// Markers that fire inside a transaction right after each write of a
/// stalled settle applied.
#[cfg(feature = "failpoints")]
const SETTLE_MARKERS: [&str; 3] = [
    "transition.settle_written",
    "effect.finish_written",
    "transition.before_events",
];

/// Arm the settle markers to log into `passed`, and the failpoint after
/// every write and before `tx.commit()` to log itself and crash.
#[cfg(feature = "failpoints")]
fn arm_settle_markers(passed: &Arc<Mutex<Vec<&'static str>>>) {
    for marker in SETTLE_MARKERS {
        let passed = Arc::clone(passed);
        let _ = fail::cfg_callback(marker, move || {
            if let Ok(mut passed) = passed.lock() {
                passed.push(marker);
            }
        });
    }
    let passed = Arc::clone(passed);
    let _ = fail::cfg_callback("transition.before_commit", move || {
        if let Ok(mut passed) = passed.lock() {
            passed.push("transition.before_commit");
        }
        panic!("crash after the stalled settle's writes, before its commit");
    });
}

#[cfg(feature = "failpoints")]
fn disarm_settle_markers() {
    for marker in SETTLE_MARKERS {
        fail::remove(marker);
    }
    fail::remove("transition.before_commit");
}

/// Recover a call left `Running` by a dead process the way a host does: its
/// lease expires, the row is requeued, and a new owner acquires it under a
/// new lease and runs it. A started call that is not replay-safe is
/// interrupted, not run again.
#[cfg(feature = "failpoints")]
async fn recover_like_a_host(
    stores: &StoreRegistry,
    child_id: &agent_server::journal::task::AgentTaskId,
) -> Result<()> {
    let reclaimed_at = OffsetDateTime::now_utc() + time::Duration::hours(1);
    let records = stores
        .task_store
        .release_expired_leases(
            reclaimed_at,
            agent_server::journal::LeaseRecoveryScope::Expired,
        )
        .await?;
    let child = stores
        .task_store
        .get(child_id)
        .await?
        .context("the tool child row")?;
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(
        child.status,
        TaskStatus::Pending,
        "the expired lease requeues the call"
    );
    let running = stores
        .task_store
        .try_acquire_task(
            child_id,
            WorkerId::new(),
            LeaseId::new(),
            reclaimed_at + time::Duration::minutes(10),
            reclaimed_at,
        )
        .await?
        .context("the new owner acquires the call")?;
    let bootstrap =
        resolve_tool_bootstrap(running, stores.task_store.as_ref(), ActivityBeacon::new()).await?;
    run_tool_effect(
        bootstrap,
        stores.task_store.as_ref(),
        &CancellationToken::new(),
        |_call, _collector: ToolEventCollector| async {
            anyhow::bail!("an interrupted unsafe call must not run again")
        },
        reclaimed_at,
    )
    .await?;
    Ok(())
}

/// A crash after the stalled settle's writes and before its commit.
///
/// Markers inside the transaction prove the settle, the effect's finish and
/// the events were applied before the crash; they and the crashing
/// failpoint are armed from inside the tool, after the effect's start
/// committed, so they see only the stalled settle's transaction. Nothing of
/// the settle survives the crash. A host-style recovery then interrupts the
/// call instead of running it again, and the parent resumes once.
#[cfg(all(feature = "failpoints", feature = "sqlite"))]
#[tokio::test]
async fn tool_stall_failpoint_before_commit_leaves_nothing_and_recovers_once() -> Result<()> {
    let _scenario = fail::FailScenario::setup();
    let dir = tempfile::tempdir()?;
    assert_stall_failpoint_before_commit(&storage(&dir.path().join("stall-crash.db"))).await
}

/// [`tool_stall_failpoint_before_commit_leaves_nothing_and_recovers_once`] on
/// Postgres, in a throwaway schema, so its `transition.settle_written` marker
/// is the one that fires. Skips without `TEST_DATABASE_URL`.
#[cfg(all(feature = "failpoints", feature = "postgres"))]
#[tokio::test]
async fn tool_stall_failpoint_before_commit_on_postgres_leaves_nothing_and_recovers_once()
-> Result<()> {
    use sqlx::Connection;

    let _scenario = fail::FailScenario::setup();
    let Ok(database_url) =
        std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
    else {
        return Ok(());
    };
    let schema = format!("stall_fp_pg_{}", uuid::Uuid::new_v4().simple());
    let mut admin = sqlx::postgres::PgConnection::connect(&database_url).await?;
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
        .execute(&mut admin)
        .await?;
    let storage = StorageConfig {
        backend: StorageBackend::Postgres,
        postgres: crate::config::PostgresStorageConfig {
            database_url: Some(database_url.clone()),
            schema: Some(schema.clone()),
            max_connections: 8,
        },
    };
    let outcome = assert_stall_failpoint_before_commit(&storage).await;
    sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
        .execute(&mut admin)
        .await?;
    outcome
}

#[cfg(feature = "failpoints")]
async fn assert_stall_failpoint_before_commit(storage: &StorageConfig) -> Result<()> {
    let thread = ThreadId::from_string("t-stall-crash");
    let signal = StallSignal::new();
    let stores = open(storage).await?;
    let (parent, bootstrap) =
        running_probe(&stores, &thread, &signal, OffsetDateTime::now_utc()).await?;
    let child_id = bootstrap.task_id.clone();

    let passed = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let arm = {
        let passed = Arc::clone(&passed);
        move || arm_settle_markers(&passed)
    };
    // Bounded: a run that never reaches the stalled settle's commit would
    // otherwise hang here instead of failing.
    let crashed = tokio::time::timeout(
        Duration::from_secs(10),
        futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(run_stalling_probe(
            &stores,
            bootstrap.clone(),
            arm,
        ))),
    )
    .await;
    disarm_settle_markers();
    anyhow::ensure!(
        matches!(crashed, Ok(Err(_))),
        "the failpoint after the stalled settle's writes must fire"
    );
    let mut passed = passed
        .lock()
        .map_err(|_| anyhow::anyhow!("marker log poisoned"))?
        .clone();
    assert_eq!(
        passed.pop(),
        Some("transition.before_commit"),
        "the crash came last"
    );
    passed.sort_unstable();
    assert_eq!(
        passed,
        vec![
            "effect.finish_written",
            "transition.before_events",
            "transition.settle_written"
        ],
        "the crashed transaction applied the settle, the effect's finish and the events",
    );
    drop(stores);

    let restarted = open(storage).await?;
    let child = restarted
        .task_store
        .get(&child_id)
        .await?
        .context("the tool child row")?;
    assert_eq!(child.status, TaskStatus::Running, "the settle rolled back");
    assert_eq!(
        tool_call_ends(&restarted.event_repo.get_events(&thread).await?),
        Vec::new()
    );
    let effect = restarted
        .task_store
        .effect(&EffectKey::tool(&child_id, "call_probe"))
        .await?
        .context("the probe's effect record")?;
    assert_eq!(
        effect.state,
        EffectState::Started,
        "only the start committed"
    );

    recover_like_a_host(&restarted, &child_id).await?;
    let child = restarted
        .task_store
        .get(&child_id)
        .await?
        .context("the tool child row")?;
    assert_eq!(child.status, TaskStatus::Failed);
    let ends = tool_call_ends(&restarted.event_repo.get_events(&thread).await?);
    assert_eq!(ends.len(), 1, "{ends:?}");
    let parent = restarted
        .task_store
        .get(&parent.id)
        .await?
        .context("the parent row")?;
    assert_eq!(
        parent.status,
        TaskStatus::Pending,
        "the parent resumes once"
    );
    Ok(())
}
