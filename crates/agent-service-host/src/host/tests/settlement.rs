//! The host settles the live descendants of a terminal root: once at
//! startup, finishing whatever a previous process left half-done, and on
//! every sweep tick, right after expired leases are failed closed.

use anyhow::{Context, Result};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use agent_sdk_foundation::{
    AgentContinuation, AgentState, ContinuationEnvelope, TerminalReason, ThreadId, TokenUsage, llm,
};
use agent_server::ChildJoinPolicy;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
};

use crate::config::{ServiceConfig, WorkerConfig};
#[cfg(feature = "sqlite")]
use crate::config::{StorageBackend, StorageConfig};
use crate::host::{ServiceHost, settle_orphaned_descendants};
use crate::stores::StoreRegistry;

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
