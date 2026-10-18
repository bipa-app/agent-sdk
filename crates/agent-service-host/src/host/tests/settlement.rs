//! The host settles the live descendants of a terminal root: once at
//! startup, finishing whatever a previous process left half-done, and on
//! every sweep tick, right after expired leases are failed closed. A failed
//! root trips the confirmation drives of exactly the tasks it cancelled.

use std::sync::Arc;

use anyhow::{Context, Result};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use agent_sdk_foundation::{
    AgentContinuation, AgentState, ContinuationEnvelope, TerminalReason, ThreadId, TokenUsage, llm,
};
use agent_server::ChildJoinPolicy;
use agent_server::journal::SpawnChildrenExt;
use agent_server::journal::SubagentInvocationSpawn;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskStatus, WorkerId,
};

use crate::config::{ServiceConfig, WorkerConfig};
#[cfg(feature = "sqlite")]
use crate::config::{StorageBackend, StorageConfig};
use crate::host::{
    AttemptClosePolicy, OwnedRootTask, ServiceHost, drain_expired_leases, fail_root_task_if_owned,
    settle_orphaned_descendants,
};
use crate::stores::StoreRegistry;
#[cfg(feature = "sqlite")]
use crate::{config::EpochRecovery, host::begin_epoch_and_reclaim};

use super::FlakyTaskStore;

fn payload(thread: &ThreadId) -> SuspensionPayload {
    SuspensionPayload {
        continuation: ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![],
            awaiting_index: 0,
            completed_results: vec![],
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: vec![],
        }),
        suspended_messages: vec![],
        child_join_policy: ChildJoinPolicy::default(),
    }
}

struct Tree {
    root: AgentTaskId,
    children: Vec<AgentTaskId>,
}

/// A root parked on three live tool children: one waiting on an
/// unanswered confirmation, one running under a live lease, one never
/// started.
async fn park_on_live_children(
    stores: &StoreRegistry,
    thread: &ThreadId,
    max_attempts: u32,
) -> Result<Tree> {
    let now = OffsetDateTime::now_utc();
    let lease_horizon = now + time::Duration::hours(1);
    let root = AgentTask::new_root_turn(thread.clone(), now, max_attempts);
    stores.task_store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(&root.id, worker.clone(), lease.clone(), lease_horizon, now)
        .await?
        .context("root must acquire")?;
    let (_, children) = stores
        .task_store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3); 3],
            payload(thread),
            None,
            Vec::new(),
            now,
        )
        .await?;

    let confirm_worker = WorkerId::new();
    let confirm_lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &children[0].id,
            confirm_worker.clone(),
            confirm_lease.clone(),
            lease_horizon,
            now,
        )
        .await?
        .context("confirmation child must acquire")?;
    stores
        .task_store
        .pause_on_confirmation(
            &children[0].id,
            &confirm_worker,
            &confirm_lease,
            payload(thread).continuation,
            None,
            now,
        )
        .await?;
    stores
        .task_store
        .try_acquire_task(
            &children[1].id,
            WorkerId::new(),
            LeaseId::new(),
            lease_horizon,
            now,
        )
        .await?
        .context("running child must acquire")?;

    Ok(Tree {
        root: root.id,
        children: children.into_iter().map(|child| child.id).collect(),
    })
}

/// Wake the parked root early (a steering note) and lease it until
/// `lease_expires_at`, leaving its children live under a running root.
async fn resume_early(
    stores: &StoreRegistry,
    root: &AgentTaskId,
    lease_expires_at: OffsetDateTime,
) -> Result<(WorkerId, LeaseId)> {
    let now = OffsetDateTime::now_utc();
    let note = vec![llm::ContentBlock::Text {
        text: "status?".into(),
    }];
    stores
        .task_store
        .enqueue_steering_resume(root, note, now)
        .await?
        .context("parked root must wake")?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(root, worker.clone(), lease.clone(), lease_expires_at, now)
        .await?
        .context("woken root must acquire")?;
    Ok((worker, lease))
}

/// The store-level fail does not cascade, so this leaves exactly what a
/// process death between a root's failure and its settlement leaves.
async fn seed_failed_root_with_live_children(
    stores: &StoreRegistry,
    thread: &ThreadId,
) -> Result<Tree> {
    let tree = park_on_live_children(stores, thread, 3).await?;
    let horizon = OffsetDateTime::now_utc() + time::Duration::hours(1);
    let (worker, lease) = resume_early(stores, &tree.root, horizon).await?;
    stores
        .task_store
        .fail_task_with_reason(
            &tree.root,
            &worker,
            &lease,
            "steering resume failed".to_owned(),
            TerminalReason::InternalError,
            OffsetDateTime::now_utc(),
        )
        .await?;
    Ok(tree)
}

async fn task(stores: &StoreRegistry, id: &AgentTaskId) -> Result<AgentTask> {
    stores
        .task_store
        .get(id)
        .await?
        .with_context(|| format!("task {id} exists"))
}

async fn assert_children_settled(stores: &StoreRegistry, tree: &Tree) -> Result<()> {
    for child in &tree.children {
        let child = task(stores, child).await?;
        assert_eq!(child.status, TaskStatus::Cancelled, "child {}", child.id);
        assert_eq!(
            child.terminal_reason,
            Some(TerminalReason::ParentCancelled),
            "child {}",
            child.id,
        );
    }
    Ok(())
}

async fn all_children_terminal(stores: &StoreRegistry, tree: &Tree) -> Result<bool> {
    for child in &tree.children {
        if !task(stores, child).await?.status.is_terminal() {
            return Ok(false);
        }
    }
    Ok(true)
}

#[tokio::test]
async fn orphan_sweep_settles_terminal_roots_and_leaves_live_trees_alone() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let orphaned =
        seed_failed_root_with_live_children(&stores, &ThreadId::from_string("t-orphans")).await?;
    let live = park_on_live_children(&stores, &ThreadId::from_string("t-live-tree"), 3).await?;
    let drive = CancellationToken::new();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(orphaned.children[1].clone(), drive.clone());

    let settled = settle_orphaned_descendants(&stores, OffsetDateTime::now_utc()).await?;
    assert_eq!(
        settled, 3,
        "every live descendant of the failed root settles"
    );
    assert_children_settled(&stores, &orphaned).await?;
    assert!(
        drive.is_cancelled(),
        "a live approved-confirmation drive under the dead root is tripped",
    );
    let root = task(&stores, &orphaned.root).await?;
    assert_eq!(
        root.status,
        TaskStatus::Failed,
        "the root keeps its outcome"
    );

    for child in &live.children {
        assert!(
            !task(&stores, child).await?.status.is_terminal(),
            "a live tree's children are not orphans",
        );
    }
    assert_eq!(
        task(&stores, &live.root).await?.status,
        TaskStatus::WaitingOnChildren,
    );

    let again = settle_orphaned_descendants(&stores, OffsetDateTime::now_utc()).await?;
    assert_eq!(again, 0, "the sweep is idempotent");
    Ok(())
}

#[tokio::test]
async fn sweep_tick_settles_children_of_a_root_failed_closed_on_lease_expiry() -> Result<()> {
    let config = ServiceConfig {
        worker: WorkerConfig {
            sweep_interval_secs: 1,
            acquisition_interval_secs: 3_600,
            ..Default::default()
        },
        ..Default::default()
    };
    let host = ServiceHost::new(config, super::sample_registry(), super::sample_runtime()?)?;
    let stores = host.stores().clone();
    let token = host.shutdown_token();

    // A one-attempt root whose early resume dies with its worker: the
    // sweep fails it closed, and its children must not outlive it.
    let tree = park_on_live_children(&stores, &ThreadId::from_string("t-lease-orphans"), 1).await?;
    let expired = OffsetDateTime::now_utc() - time::Duration::seconds(1);
    resume_early(&stores, &tree.root, expired).await?;

    let host_handle = tokio::spawn(async move { host.run().await });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !all_children_terminal(&stores, &tree).await? {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "the sweep tick never settled the dead root's children",
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    token.cancel();
    host_handle.await??;

    let root = task(&stores, &tree.root).await?;
    assert_eq!(root.status, TaskStatus::Failed, "the root failed closed");
    assert_children_settled(&stores, &tree).await
}

/// The startup orphan settlement is exported as
/// `agent_service_host.orphans.settled`.
#[cfg(feature = "otel")]
#[tokio::test]
async fn startup_exports_the_orphans_it_settled() -> Result<()> {
    let meter = crate::observability::test_meter::TestMeter::install().await;
    let config = ServiceConfig {
        worker: WorkerConfig {
            sweep_interval_secs: 3_600,
            acquisition_interval_secs: 3_600,
            ..Default::default()
        },
        ..Default::default()
    };
    let host = ServiceHost::new(config, super::sample_registry(), super::sample_runtime()?)?;
    let stores = host.stores().clone();
    let token = host.shutdown_token();
    // A failed root over live children, as a dead process leaves it: the
    // startup pass settles them.
    let tree =
        seed_failed_root_with_live_children(&stores, &ThreadId::from_string("t-orphan-metric"))
            .await?;

    let host_handle = tokio::spawn(async move { host.run().await });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !all_children_terminal(&stores, &tree).await? {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "startup never settled"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    token.cancel();
    host_handle.await??;

    let settled = meter.counter("agent_service_host.orphans.settled", &[]);
    drop(meter);
    // A lower bound: under plain `cargo test` other tests record into the same
    // global meter (nextest runs one per process).
    assert!(
        settled >= u64::try_from(tree.children.len())?,
        "each settled child is counted"
    );
    Ok(())
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn host_startup_settles_orphans_left_by_a_previous_process() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let storage = StorageConfig {
        backend: StorageBackend::Sqlite {
            path: Some(dir.path().join("orphans.db").display().to_string()),
            max_connections: None,
        },
        ..StorageConfig::default()
    };
    let tree = {
        let previous = StoreRegistry::from_config(&storage, super::sample_registry())?;
        previous.initialize().await?;
        seed_failed_root_with_live_children(&previous, &ThreadId::from_string("t-restart")).await?
    };

    let config = ServiceConfig {
        storage,
        worker: WorkerConfig {
            sweep_interval_secs: 3_600,
            acquisition_interval_secs: 3_600,
            ..Default::default()
        },
        ..Default::default()
    };
    let host = ServiceHost::new(config, super::sample_registry(), super::sample_runtime()?)?;
    let stores = host.stores().clone();
    let token = host.shutdown_token();
    let host_handle = tokio::spawn(async move { host.run().await });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !all_children_terminal(&stores, &tree).await? {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "host startup never settled the orphans",
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    token.cancel();
    host_handle.await??;

    assert_children_settled(&stores, &tree).await?;
    assert_eq!(task(&stores, &tree.root).await?.status, TaskStatus::Failed);
    Ok(())
}

/// A root parked on one subagent whose child-thread root is itself parked on
/// a tool child awaiting an approved confirmation's drive, then woken early
/// and leased until `root_lease_expires_at`: the failing root's only direct
/// child is the invocation.
async fn running_root_over_a_child_thread_drive(
    stores: &StoreRegistry,
    thread: &ThreadId,
    max_attempts: u32,
    root_lease_expires_at: OffsetDateTime,
) -> Result<(AgentTaskId, AgentTaskId, WorkerId, LeaseId)> {
    let now = OffsetDateTime::now_utc();
    let horizon = now + time::Duration::hours(1);
    let root = AgentTask::new_root_turn(thread.clone(), now, max_attempts);
    stores.task_store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(&root.id, worker.clone(), lease.clone(), horizon, now)
        .await?
        .context("root must acquire")?;
    let child_thread = ThreadId::from_string(format!("{}-child", thread.0));
    stores
        .thread_store
        .get_or_create(&child_thread, now)
        .await?;
    let (_, _invocation, child_root) = stores
        .task_store
        .spawn_subagent_invocation(
            &root.id,
            &worker,
            &lease,
            SubagentInvocationSpawn {
                child_thread_id: child_thread.clone(),
                spec: super::subagent_timeout_spec("inspect", 60_000),
                child_root_input: vec![SubmittedInputItem::Text {
                    text: "inspect".into(),
                }],
                spawn_index: 0,
                child_caller_metadata: None,
                payload: payload(thread),
            },
            Vec::new(),
            now,
        )
        .await?;

    let child_worker = WorkerId::new();
    let child_lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            child_worker.clone(),
            child_lease.clone(),
            horizon,
            now,
        )
        .await?
        .context("child root must acquire")?;
    let (_, tools) = stores
        .task_store
        .spawn_tool_children(
            &child_root.id,
            &child_worker,
            &child_lease,
            vec![ChildSpawnSpec::new(3)],
            payload(&child_thread),
            None,
            Vec::new(),
            now,
        )
        .await?;
    let tool_worker = WorkerId::new();
    let tool_lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &tools[0].id,
            tool_worker.clone(),
            tool_lease.clone(),
            horizon,
            now,
        )
        .await?
        .context("tool child must acquire")?;
    stores
        .task_store
        .pause_on_confirmation(
            &tools[0].id,
            &tool_worker,
            &tool_lease,
            payload(&child_thread).continuation,
            None,
            now,
        )
        .await?;

    let (worker, lease) = resume_early(stores, &root.id, root_lease_expires_at).await?;
    Ok((root.id, tools[0].id.clone(), worker, lease))
}

async fn fail_owned_root(
    stores: &StoreRegistry,
    thread: &ThreadId,
    root: &AgentTaskId,
    worker: &WorkerId,
    lease: &LeaseId,
) -> Result<()> {
    fail_root_task_if_owned(
        stores,
        OwnedRootTask {
            task: root,
            thread,
            worker,
            lease,
        },
        &anyhow::anyhow!("steering resume failed"),
        TerminalReason::InternalError,
        0,
        AttemptClosePolicy::CloseOpenAttempts,
        OffsetDateTime::now_utc(),
    )
    .await
}

/// A failed root trips the confirmation drive of every task its failure
/// cancelled, however deep: here a tool child on the thread of the failed
/// root's subagent.
#[tokio::test]
async fn failing_a_root_trips_the_drives_it_cancelled_on_a_child_thread() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let thread = ThreadId::from_string("t-deep-drive");
    let horizon = OffsetDateTime::now_utc() + time::Duration::hours(1);
    let (root, tool, worker, lease) =
        running_root_over_a_child_thread_drive(&stores, &thread, 3, horizon).await?;
    let drive = CancellationToken::new();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(tool.clone(), drive.clone());

    fail_owned_root(&stores, &thread, &root, &worker, &lease).await?;

    let cancelled = task(&stores, &tool).await?;
    assert_eq!(cancelled.status, TaskStatus::Cancelled);
    assert_eq!(
        cancelled.terminal_reason,
        Some(TerminalReason::ParentCancelled)
    );
    assert!(
        drive.is_cancelled(),
        "the drive of a task the failure cancelled must be tripped",
    );
    Ok(())
}

/// A failed root whose settlement did not land trips no drive: the task
/// behind it is still live and owns its drive until the orphan sweep
/// cancels it, which trips the drive then.
#[tokio::test]
async fn failing_a_root_trips_no_drive_its_settlement_did_not_cancel() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let thread = ThreadId::from_string("t-unsettled-drive");
    let tree = park_on_live_children(&stores, &thread, 3).await?;
    let horizon = OffsetDateTime::now_utc() + time::Duration::hours(1);
    let (worker, lease) = resume_early(&stores, &tree.root, horizon).await?;
    let confirming = tree.children[0].clone();
    let drive = CancellationToken::new();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(confirming.clone(), drive.clone());

    let mut unsettled = stores.clone();
    let both_settlements = 2;
    unsettled.task_store = Arc::new(
        FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_cancels(both_settlements),
    ) as Arc<dyn AgentTaskStore>;
    fail_owned_root(&unsettled, &thread, &tree.root, &worker, &lease).await?;

    assert_eq!(task(&stores, &tree.root).await?.status, TaskStatus::Failed);
    assert_eq!(
        task(&stores, &confirming).await?.status,
        TaskStatus::AwaitingConfirmation,
        "precondition: the settlement did not land",
    );
    assert!(
        !drive.is_cancelled(),
        "a drive whose task is still live must not be tripped",
    );

    settle_orphaned_descendants(&stores, OffsetDateTime::now_utc()).await?;
    assert_children_settled(&stores, &tree).await?;
    assert!(drive.is_cancelled(), "the sweep's settlement trips it");
    Ok(())
}

/// A root failure the store refuses still trips the drives of the tasks
/// its settlement cancelled before the refusal.
#[tokio::test]
async fn a_refused_root_failure_still_trips_the_drives_it_cancelled() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let thread = ThreadId::from_string("t-refused-fail-drive");
    let tree = park_on_live_children(&stores, &thread, 3).await?;
    let horizon = OffsetDateTime::now_utc() + time::Duration::hours(1);
    let (worker, lease) = resume_early(&stores, &tree.root, horizon).await?;
    let confirming = tree.children[0].clone();
    let drive = CancellationToken::new();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(confirming.clone(), drive.clone());

    let mut refusing = stores.clone();
    refusing.task_store =
        Arc::new(FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_fail_tasks(1))
            as Arc<dyn AgentTaskStore>;
    let refused = fail_owned_root(&refusing, &thread, &tree.root, &worker, &lease).await;

    assert!(
        refused.is_err(),
        "precondition: the store refused the failure"
    );
    assert_eq!(task(&stores, &tree.root).await?.status, TaskStatus::Running);
    assert_children_settled(&stores, &tree).await?;
    assert!(
        drive.is_cancelled(),
        "the drive of a task the settlement cancelled must be tripped",
    );
    Ok(())
}

/// The lease sweep forwards what its fail-close settled: it trips the
/// drives of the tasks it cancelled and wakes the followers of the child
/// threads it closed.
#[tokio::test]
async fn the_lease_sweep_forwards_what_its_fail_close_settled() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let thread = ThreadId::from_string("t-swept-drive");
    let expired = OffsetDateTime::now_utc() - time::Duration::seconds(1);
    let (root, tool, _worker, _lease) =
        running_root_over_a_child_thread_drive(&stores, &thread, 1, expired).await?;
    let drive = CancellationToken::new();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(tool.clone(), drive.clone());
    let child_thread = task(&stores, &tool).await?.thread_id;
    let mut follower = stores.event_notifier.subscribe(&child_thread);

    drain_expired_leases(
        &stores,
        None,
        OffsetDateTime::now_utc(),
        &agent_server::journal::LeaseRecoveryScope::Expired,
    )
    .await?;

    assert_eq!(task(&stores, &root).await?.status, TaskStatus::Failed);
    let cancelled = task(&stores, &tool).await?;
    assert_eq!(cancelled.status, TaskStatus::Cancelled);
    assert!(
        drive.is_cancelled(),
        "the drive of a task the sweep cancelled must be tripped",
    );
    let marker = tokio::time::timeout(std::time::Duration::from_secs(5), follower.recv()).await??;
    assert_eq!(marker.thread_id, child_thread);
    assert!(
        matches!(
            marker.event,
            agent_sdk_foundation::AgentEvent::Cancelled { .. }
        ),
        "{:?}",
        marker.event,
    );
    Ok(())
}

/// Site 15: a subagent invocation's settle and the parent's completed
/// `SubagentProgress` commit in one transaction. A crash between the two
/// writes leaves the invocation running with no completed progress on the
/// parent thread.
#[cfg(all(feature = "sqlite", feature = "failpoints"))]
#[tokio::test]
async fn subagent_completion_failpoint_keeps_the_invocation_running_without_its_progress()
-> Result<()> {
    let _scenario = fail::FailScenario::setup();
    let dir = tempfile::tempdir()?;
    let storage = StorageConfig {
        backend: StorageBackend::Sqlite {
            path: Some(dir.path().join("subagent-settle.db").display().to_string()),
            max_connections: None,
        },
        ..StorageConfig::default()
    };
    let stores = StoreRegistry::from_config(&storage, super::sample_registry())?;
    stores.initialize().await?;
    let parent_thread = ThreadId::from_string("t-subagent-settle");
    let now = OffsetDateTime::now_utc();
    let (_, invocation, child_root) =
        super::persist_subagent_fixture(&stores, &parent_thread, 60_000, now).await?;

    let child_worker = WorkerId::new();
    let child_lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            child_worker.clone(),
            child_lease.clone(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the child root acquires")?;
    stores
        .task_store
        .fail_task(
            &child_root.id,
            &child_worker,
            &child_lease,
            "child stopped".into(),
            now,
        )
        .await?;
    let running = stores
        .task_store
        .try_acquire_task(
            &invocation.id,
            WorkerId::new(),
            LeaseId::new(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the invocation is runnable once its child root is terminal")?;
    let bootstrap =
        agent_server::worker::resolve_subagent_bootstrap(running, stores.task_store.as_ref())
            .await?;
    let deps = agent_server::worker::SubagentResultDeps {
        task_store: stores.task_store.as_ref(),
        thread_store: stores.thread_store.as_ref(),
        message_store: stores.message_store.as_ref(),
        event_repo: stores.event_repo.as_ref(),
        artifact_store: None,
        completion_hook: None,
    };

    fail::cfg("transition.before_events", "panic")
        .map_err(|error| anyhow::anyhow!("configure failpoint: {error}"))?;
    let crashed = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
        agent_server::worker::execute_subagent_task(bootstrap, &deps, now),
    ))
    .await;
    fail::remove("transition.before_events");
    anyhow::ensure!(
        crashed.is_err(),
        "the failpoint between the writes must fire"
    );

    let after = stores
        .task_store
        .get(&invocation.id)
        .await?
        .context("the invocation row")?;
    let completed_progress = stores
        .event_repo
        .get_events(&parent_thread)
        .await?
        .into_iter()
        .filter(|event| {
            matches!(
                &event.event,
                agent_sdk_foundation::AgentEvent::SubagentProgress {
                    completed: true,
                    ..
                }
            )
        })
        .count();
    anyhow::ensure!(
        after.status == TaskStatus::Running && completed_progress == 0,
        "the settle and its progress disagree after a crash: invocation {:?}, {completed_progress} \
         completed progress events",
        after.status,
    );
    Ok(())
}

/// Site 15 without a crash: when the store fails the invocation's
/// completion, `execute_subagent_task` returns the error, and neither the
/// completion nor its progress was written.
#[tokio::test]
async fn a_failed_subagent_completion_surfaces_and_writes_neither_half() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let parent_thread = ThreadId::from_string("t-subagent-store-failure");
    let now = OffsetDateTime::now_utc();
    let (_, invocation, child_root) =
        super::persist_subagent_fixture(&stores, &parent_thread, 60_000, now).await?;
    let child_worker = WorkerId::new();
    let child_lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            child_worker.clone(),
            child_lease.clone(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the child root acquires")?;
    stores
        .task_store
        .fail_task(
            &child_root.id,
            &child_worker,
            &child_lease,
            "child stopped".into(),
            now,
        )
        .await?;
    let running = stores
        .task_store
        .try_acquire_task(
            &invocation.id,
            WorkerId::new(),
            LeaseId::new(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the invocation is runnable once its child root is terminal")?;
    let failing: Arc<dyn AgentTaskStore> =
        Arc::new(FlakyTaskStore::new(Arc::clone(&stores.task_store)).failing_completes(1));
    let bootstrap =
        agent_server::worker::resolve_subagent_bootstrap(running, failing.as_ref()).await?;
    let deps = agent_server::worker::SubagentResultDeps {
        task_store: failing.as_ref(),
        thread_store: stores.thread_store.as_ref(),
        message_store: stores.message_store.as_ref(),
        event_repo: stores.event_repo.as_ref(),
        artifact_store: None,
        completion_hook: None,
    };

    let error = agent_server::worker::execute_subagent_task(bootstrap, &deps, now)
        .await
        .err()
        .context("a failed completion must surface")?;
    anyhow::ensure!(
        format!("{error:#}").contains("injected store failure completing a task"),
        "{error:#}",
    );
    let after = stores
        .task_store
        .get(&invocation.id)
        .await?
        .context("the invocation row")?;
    let completed_progress = stores
        .event_repo
        .get_events(&parent_thread)
        .await?
        .into_iter()
        .filter(|event| {
            matches!(
                &event.event,
                agent_sdk_foundation::AgentEvent::SubagentProgress {
                    completed: true,
                    ..
                }
            )
        })
        .count();
    anyhow::ensure!(
        after.status == TaskStatus::Running && completed_progress == 0,
        "a failed completion wrote half: invocation {:?}, {completed_progress} completed progress",
        after.status,
    );
    Ok(())
}

/// A completion hook that counts its runs and records each `prior` state.
#[cfg(all(feature = "sqlite", feature = "failpoints"))]
#[derive(Default)]
struct CountingCompletionHook {
    runs: std::sync::Mutex<Vec<(u32, Option<agent_server::journal::EffectState>)>>,
}

#[cfg(all(feature = "sqlite", feature = "failpoints"))]
#[async_trait::async_trait]
impl agent_server::worker::SubagentCompletionHook for CountingCompletionHook {
    fn plan(
        &self,
        _ctx: &agent_server::worker::CompletionContext,
    ) -> Option<agent_server::worker::CompletionPlan> {
        Some(agent_server::worker::CompletionPlan {
            step: "merge_back".into(),
            repo: "repo_1".into(),
            unit: "unit_1".into(),
        })
    }

    async fn run(
        &self,
        ctx: &agent_server::worker::CompletionContext,
        prior: Option<&agent_server::journal::EffectRecord>,
        _mode: agent_server::worker::RunMode,
    ) -> Result<agent_server::worker::CompletionOutput> {
        self.runs
            .lock()
            .ok()
            .context("runs lock")?
            .push((ctx.attempt, prior.map(|record| record.state.clone())));
        Ok(agent_server::worker::CompletionOutput {
            result: agent_sdk_foundation::ToolResult::success("merged unit_1"),
            state: agent_server::journal::EffectState::Completed,
            extra_writes: Vec::new(),
            park: None,
        })
    }
}

/// `subagent.completion_hook.before_settle` fires after the hook's start
/// record committed and the hook ran, before the settle. A crash there
/// leaves the record `Started`, the invocation running and the parent
/// without its result; the requeued invocation runs the hook again with
/// that record as `prior` and settles once.
#[cfg(all(feature = "sqlite", feature = "failpoints"))]
#[tokio::test]
async fn a_crash_before_the_hook_settle_keeps_its_record_started_for_the_next_attempt() -> Result<()>
{
    use agent_server::journal::EffectState;
    use agent_server::journal::effect::EffectKey;

    let _scenario = fail::FailScenario::setup();
    let dir = tempfile::tempdir()?;
    let storage = StorageConfig {
        backend: StorageBackend::Sqlite {
            path: Some(dir.path().join("hook-settle.db").display().to_string()),
            max_connections: None,
        },
        ..StorageConfig::default()
    };
    let hook = Arc::new(CountingCompletionHook::default());
    let stores = StoreRegistry::from_config(&storage, super::sample_registry())?
        .with_subagent_completion_hook(
            Arc::clone(&hook) as Arc<dyn agent_server::worker::SubagentCompletionHook>
        );
    stores.initialize().await?;
    let parent_thread = ThreadId::from_string("t-hook-settle-crash");
    let now = OffsetDateTime::now_utc();
    let (parent, invocation, child_root) =
        super::persist_subagent_fixture(&stores, &parent_thread, 60_000, now).await?;
    stores.task_store.cancel_tree(&child_root.id, now).await?;
    let acquire = |at| acquire_invocation(&stores, &invocation.id, at);
    let completed_progress = || completed_subagent_progress(&stores, &parent_thread);

    fail::cfg("subagent.completion_hook.before_settle", "panic")
        .map_err(|error| anyhow::anyhow!("configure failpoint: {error}"))?;
    let bootstrap = acquire(now).await?;
    let deps = stores.subagent_result_deps(None);
    let crashed = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
        agent_server::worker::execute_subagent_task(bootstrap, &deps, now),
    ))
    .await;
    fail::remove("subagent.completion_hook.before_settle");
    anyhow::ensure!(
        crashed.is_err(),
        "the failpoint before the settle must fire"
    );

    let key = EffectKey::git(&invocation.id, "merge_back");
    let record = stores
        .task_store
        .effect(&key)
        .await?
        .context("the start record committed before the crash")?;
    anyhow::ensure!(record.state == EffectState::Started, "{:?}", record.state);
    let runs = hook.runs.lock().ok().context("runs lock")?.clone();
    anyhow::ensure!(
        runs == vec![(1, None)],
        "the hook ran before the crash: {runs:?}"
    );
    let row = stores
        .task_store
        .get(&invocation.id)
        .await?
        .context("row")?;
    anyhow::ensure!(row.status == TaskStatus::Running, "{:?}", row.status);
    anyhow::ensure!(completed_progress().await? == 0);

    let later = now + time::Duration::minutes(1);
    stores
        .task_store
        .release_expired_leases(later, agent_server::journal::LeaseRecoveryScope::Expired)
        .await?;
    let bootstrap = acquire(later).await?;
    let outcome = agent_server::worker::execute_subagent_task(bootstrap, &deps, later).await?;
    let runs = hook.runs.lock().ok().context("runs lock")?.clone();
    anyhow::ensure!(
        runs == vec![(1, None), (2, Some(EffectState::Started))],
        "the requeued attempt continues the record: {runs:?}"
    );
    anyhow::ensure!(outcome.invocation_task.status == TaskStatus::Completed);
    let resumed = stores.task_store.get(&parent.id).await?.context("parent")?;
    anyhow::ensure!(
        resumed.status == TaskStatus::Pending,
        "{:?}",
        resumed.status
    );
    let record = stores.task_store.effect(&key).await?.context("record")?;
    anyhow::ensure!(record.state == EffectState::Completed, "{:?}", record.state);
    anyhow::ensure!(completed_progress().await? == 1);
    Ok(())
}

/// Acquire the invocation `id` at `at` with a five-second lease, and resolve
/// its bootstrap.
#[cfg(all(feature = "sqlite", feature = "failpoints"))]
async fn acquire_invocation(
    stores: &StoreRegistry,
    id: &AgentTaskId,
    at: OffsetDateTime,
) -> Result<agent_server::worker::SubagentTaskBootstrap> {
    let running = stores
        .task_store
        .try_acquire_task(
            id,
            WorkerId::new(),
            LeaseId::new(),
            at + time::Duration::seconds(5),
            at,
        )
        .await?
        .context("the invocation must acquire")?;
    agent_server::worker::resolve_subagent_bootstrap(running, stores.task_store.as_ref()).await
}

/// The completed `SubagentProgress` events on `thread`.
#[cfg(all(feature = "sqlite", feature = "failpoints"))]
async fn completed_subagent_progress(stores: &StoreRegistry, thread: &ThreadId) -> Result<usize> {
    Ok(stores
        .event_repo
        .get_events(thread)
        .await?
        .iter()
        .filter(|event| {
            matches!(
                event.event,
                agent_sdk_foundation::AgentEvent::SubagentProgress {
                    completed: true,
                    ..
                }
            )
        })
        .count())
}

/// A previous process on an `SQLite` file leased a root for an hour and
/// died. The next process begins its epoch at start; with
/// [`EpochRecovery::SingleHost`] it requeues the dead lease before any
/// worker runs, and with [`EpochRecovery::LeaseExpiry`] it leaves the
/// lease to expire.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn a_new_process_reclaims_a_dead_processs_lease_only_as_the_single_host() -> Result<()> {
    for (recovery, reclaimed) in [
        (EpochRecovery::SingleHost, true),
        (EpochRecovery::LeaseExpiry, false),
    ] {
        let dir = tempfile::tempdir()?;
        let storage = StorageConfig {
            backend: StorageBackend::Sqlite {
                path: Some(dir.path().join("epoch.db").display().to_string()),
                max_connections: None,
            },
            ..StorageConfig::default()
        };
        let now = OffsetDateTime::now_utc();
        let (root, lease) = {
            let previous = StoreRegistry::from_config(&storage, super::sample_registry())?;
            previous.initialize().await?;
            begin_epoch_and_reclaim(&previous, recovery, now).await?;
            let root = AgentTask::new_root_turn(ThreadId::from_string("t-epoch"), now, 3);
            previous.task_store.submit_root_turn(root.clone()).await?;
            let lease = LeaseId::new();
            previous
                .task_store
                .try_acquire_task(
                    &root.id,
                    WorkerId::from_string("worker-0"),
                    lease.clone(),
                    now + time::Duration::hours(1),
                    now,
                )
                .await?
                .context("the previous process leases the root")?;
            (root, lease)
        };

        let next = StoreRegistry::from_config(&storage, super::sample_registry())?;
        next.initialize().await?;
        begin_epoch_and_reclaim(&next, recovery, now + time::Duration::seconds(1)).await?;
        let row = task(&next, &root.id).await?;
        if reclaimed {
            assert_eq!(row.status, TaskStatus::Pending, "{recovery:?}");
            assert!(row.lease_id.is_none(), "{recovery:?}");
        } else {
            assert_eq!(row.status, TaskStatus::Running, "{recovery:?}");
            assert_eq!(row.lease_id, Some(lease), "{recovery:?}");
        }
    }
    Ok(())
}

/// The start-up span reaches OpenTelemetry, not only the logs: a
/// `SingleHost` begin that reclaims one earlier lease exports
/// `agent_server.epoch.begin` with the epoch id, the mode and the counts.
#[cfg(feature = "otel")]
#[tokio::test]
async fn the_epoch_begin_span_is_exported_with_what_the_reclaim_did() -> Result<()> {
    use opentelemetry::Value;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    opentelemetry::global::set_tracer_provider(provider.clone());

    let stores = StoreRegistry::from_config(
        &crate::config::StorageConfig::default(),
        super::sample_registry(),
    )?;
    stores.initialize().await?;
    let now = OffsetDateTime::now_utc();
    crate::host::begin_epoch_and_reclaim(&stores, crate::config::EpochRecovery::SingleHost, now)
        .await?;
    let root = AgentTask::new_root_turn(ThreadId::from_string("t-epoch-span"), now, 3);
    stores.task_store.submit_root_turn(root.clone()).await?;
    stores
        .task_store
        .try_acquire_task(
            &root.id,
            WorkerId::from_string("worker-0"),
            LeaseId::new(),
            now + time::Duration::hours(1),
            now,
        )
        .await?
        .context("the earlier epoch leases the root")?;
    let epoch = crate::host::begin_epoch_and_reclaim(
        &stores,
        crate::config::EpochRecovery::SingleHost,
        now + time::Duration::seconds(1),
    )
    .await?;
    provider.force_flush()?;

    let spans = exporter.get_finished_spans()?;
    let span = spans
        .iter()
        .filter(|span| span.name.as_ref() == "agent_server.epoch.begin")
        .find(|span| {
            span.attributes.iter().any(|kv| {
                kv.key.as_str() == "agent_server.epoch.id"
                    && kv.value == Value::from(epoch.as_str().to_owned())
            })
        })
        .with_context(|| format!("an agent_server.epoch.begin span for {epoch}: {spans:?}"))?;
    let attribute = |key: &str| {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.clone())
    };
    assert_eq!(
        attribute("agent_server.epoch.recovery"),
        Some(Value::from("single_host"))
    );
    assert_eq!(
        attribute("agent_server.epoch.requeued"),
        Some(Value::I64(1))
    );
    assert_eq!(
        attribute("agent_server.epoch.failed_interrupted"),
        Some(Value::I64(0))
    );
    assert_eq!(attribute("outcome"), Some(Value::from("ok")));
    Ok(())
}
