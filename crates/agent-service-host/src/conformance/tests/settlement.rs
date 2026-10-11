//! Settlement conformance: every cancelled or orphaned task reaches a
//! terminal state on every backend, and a parked parent's fan-in wakes no
//! matter which live state its cancelled child was in.

use anyhow::{Context, Result};

use agent_sdk_foundation::TerminalReason;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, TaskStatus, WorkerId,
};
use agent_server::journal::task_state::TaskState;
use agent_server::journal::thread_store::ThreadStore;

use super::{
    cancelled_marker_count, spawn_subagent_fixture, suspension_payload, t_plus, thread_id,
};

/// Live states a cancelled child can be caught in.
#[derive(Clone, Copy, Debug)]
enum LiveChild {
    /// Admitted, never acquired.
    Pending,
    /// Leased by a worker whose lease is still valid.
    RunningLiveLease,
    /// Leased by a worker that died: the lease has expired but no sweep
    /// has reclaimed the row yet.
    RunningDeadLease,
    /// Parked on a user confirmation nobody answered.
    AwaitingConfirmation,
}

const LIVE_CHILDREN: [LiveChild; 4] = [
    LiveChild::Pending,
    LiveChild::RunningLiveLease,
    LiveChild::RunningDeadLease,
    LiveChild::AwaitingConfirmation,
];

/// A root turn parked on `count` fresh tool children.
struct ParkedRoot {
    root: AgentTask,
    children: Vec<AgentTask>,
}

async fn park_root_on_tool_children(
    task_store: &dyn AgentTaskStore,
    tid: &str,
    count: usize,
    max_attempts: u32,
    at: i64,
) -> Result<ParkedRoot> {
    let root = AgentTask::new_root_turn(thread_id(tid), t_plus(at), 3);
    task_store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            t_plus(at + 10_000),
            t_plus(at + 1),
        )
        .await?
        .context("root must acquire")?;
    let (parent, children) = task_store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(max_attempts); count],
            suspension_payload(tid),
            None,
            Vec::new(),
            t_plus(at + 2),
        )
        .await?;
    assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
    Ok(ParkedRoot {
        root: parent,
        children,
    })
}

async fn drive_child(
    task_store: &dyn AgentTaskStore,
    tid: &str,
    child: &AgentTaskId,
    state: LiveChild,
    at: i64,
) -> Result<()> {
    let lease_expires_at = match state {
        LiveChild::Pending => return Ok(()),
        LiveChild::RunningDeadLease => t_plus(at + 1),
        LiveChild::RunningLiveLease | LiveChild::AwaitingConfirmation => t_plus(at + 10_000),
    };
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            child,
            worker.clone(),
            lease.clone(),
            lease_expires_at,
            t_plus(at),
        )
        .await?
        .context("child must acquire")?;
    if matches!(state, LiveChild::AwaitingConfirmation) {
        task_store
            .pause_on_confirmation(
                child,
                &worker,
                &lease,
                suspension_payload(tid).continuation,
                None,
                t_plus(at + 1),
            )
            .await?;
    }
    Ok(())
}

async fn complete_child(
    task_store: &dyn AgentTaskStore,
    child: &AgentTaskId,
    at: i64,
) -> Result<()> {
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            child,
            worker.clone(),
            lease.clone(),
            t_plus(at + 10_000),
            t_plus(at),
        )
        .await?
        .context("sibling must acquire")?;
    task_store
        .complete_task(child, &worker, &lease, t_plus(at + 1))
        .await?;
    Ok(())
}

/// Wake a parked root for an early (steering) resume and lease it, so it is
/// `Running` while its children are still live — the shape every early
/// resume (steering, wait-any) leaves behind.
async fn resume_root_early(
    task_store: &dyn AgentTaskStore,
    root: &AgentTaskId,
    at: i64,
) -> Result<(WorkerId, LeaseId)> {
    let note = vec![agent_sdk_foundation::llm::ContentBlock::Text {
        text: "status?".into(),
    }];
    task_store
        .enqueue_steering_resume(root, note, t_plus(at))
        .await?
        .context("parked root must wake for steering")?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            root,
            worker.clone(),
            lease.clone(),
            t_plus(at + 10_000),
            t_plus(at + 1),
        )
        .await?
        .context("woken root must acquire")?;
    Ok((worker, lease))
}

async fn fetch(task_store: &dyn AgentTaskStore, id: &AgentTaskId) -> Result<AgentTask> {
    task_store
        .get(id)
        .await?
        .with_context(|| format!("task {id} exists"))
}

fn assert_settled(task: &AgentTask, reason: &TerminalReason, label: &str) {
    assert_eq!(
        task.status,
        TaskStatus::Cancelled,
        "{label}: must be Cancelled"
    );
    assert_eq!(task.terminal_reason.as_ref(), Some(reason), "{label}");
    assert!(task.worker_id.is_none(), "{label}: lease must be dropped");
    assert!(
        matches!(task.state, TaskState::None),
        "{label}: paused state must be cleared, got {:?}",
        task.state,
    );
}

/// Cancelling a tool child wakes its parked parent whatever live state the
/// child was in — with a live worker, with a dead worker's expired lease,
/// unacquired, or parked on an unanswered confirmation.
pub(super) async fn test_cancel_child_in_every_live_state_wakes_parent(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    for (index, state) in LIVE_CHILDREN.into_iter().enumerate() {
        let tid = format!("settle-cancel-child-{index}");
        let at = 1_000 + i64::try_from(index)? * 100;
        let parked = park_root_on_tool_children(task_store, &tid, 2, 3, at).await?;
        complete_child(task_store, &parked.children[1].id, at + 3).await?;
        drive_child(task_store, &tid, &parked.children[0].id, state, at + 5).await?;

        let outcome = task_store
            .cancel_tree(&parked.children[0].id, t_plus(at + 50))
            .await?;
        assert_eq!(
            outcome.transitioned,
            vec![parked.children[0].id.clone()],
            "{state:?}",
        );
        assert!(
            outcome.markers.is_empty(),
            "{state:?}: a tool child has no marker"
        );
        let child = fetch(task_store, &parked.children[0].id).await?;
        assert_settled(&child, &TerminalReason::UserCancel, &format!("{state:?}"));

        let parent = fetch(task_store, &parked.root.id).await?;
        assert_eq!(
            parent.status,
            TaskStatus::Pending,
            "{state:?}: parent wakes"
        );
        assert_eq!(parent.pending_child_count, 0, "{state:?}");
        assert!(
            matches!(parent.state, TaskState::ReadyToResume { .. }),
            "{state:?}: parent must be ready to resume, got {:?}",
            parent.state,
        );
    }
    Ok(())
}

/// Cancelling a subagent's child-thread root — in any live state, including
/// parked behind its own tool child that waits on a confirmation — cancels
/// that subtree, closes the child thread with one marker, and wakes the
/// parent-thread invocation that was waiting on it.
pub(super) async fn test_cancel_child_thread_root_wakes_invocation(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
    event_repo: &dyn EventRepository,
) -> Result<()> {
    for (index, state) in LIVE_CHILDREN.into_iter().enumerate() {
        let tid = format!("settle-child-thread-{index}");
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, &tid).await?;
        let child_tid = child_root.thread_id.0.clone();

        // Park the child-thread root on a tool child that sits in `state`.
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        task_store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                t_plus(10_000),
                t_plus(3),
            )
            .await?
            .context("child root must acquire")?;
        let (_, tool_children) = task_store
            .spawn_tool_children(
                &child_root.id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(3)],
                suspension_payload(&child_tid),
                None,
                Vec::new(),
                t_plus(4),
            )
            .await?;
        drive_child(task_store, &child_tid, &tool_children[0].id, state, 5).await?;

        let outcome = task_store.cancel_tree(&child_root.id, t_plus(50)).await?;
        assert_eq!(outcome.transitioned.len(), 2, "{state:?}: {outcome:?}");
        assert_eq!(
            outcome.markers.len(),
            1,
            "{state:?}: the child thread closes"
        );
        assert_eq!(
            cancelled_marker_count(event_repo, &child_root.thread_id).await?,
            1,
            "{state:?}",
        );
        assert_settled(
            &fetch(task_store, &child_root.id).await?,
            &TerminalReason::UserCancel,
            &format!("{state:?} child root"),
        );
        assert_settled(
            &fetch(task_store, &tool_children[0].id).await?,
            &TerminalReason::ParentCancelled,
            &format!("{state:?} tool child"),
        );

        let woken = fetch(task_store, &invocation.id).await?;
        assert_eq!(
            woken.status,
            TaskStatus::Pending,
            "{state:?}: invocation wakes"
        );
        assert_eq!(woken.pending_child_count, 0, "{state:?}");
        let parent_after = fetch(task_store, &parent.id).await?;
        assert_eq!(
            parent_after.status,
            TaskStatus::WaitingOnChildren,
            "{state:?}: the parent waits on the still-live invocation",
        );
    }
    Ok(())
}

/// A root that ended (here: failed during an early resume) while its
/// children were still live leaves them live — the store's fail does not
/// cascade. `cancel_tree` on the terminal root is the settlement primitive:
/// it cancels exactly the live descendants (including one parked on an
/// unanswered confirmation), leaves the root's own terminal outcome alone,
/// and is idempotent.
pub(super) async fn test_cancel_tree_settles_live_descendants_of_terminal_root(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    let tid = "settle-terminal-root";
    let parked = park_root_on_tool_children(task_store, tid, 3, 3, 2_000).await?;
    drive_child(
        task_store,
        tid,
        &parked.children[0].id,
        LiveChild::AwaitingConfirmation,
        2_005,
    )
    .await?;
    drive_child(
        task_store,
        tid,
        &parked.children[1].id,
        LiveChild::RunningLiveLease,
        2_007,
    )
    .await?;
    let (worker, lease) = resume_root_early(task_store, &parked.root.id, 2_010).await?;
    task_store
        .fail_task_with_reason(
            &parked.root.id,
            &worker,
            &lease,
            "steering resume failed".to_owned(),
            TerminalReason::InternalError,
            t_plus(2_020),
        )
        .await?;
    for child in &parked.children {
        assert!(
            !fetch(task_store, &child.id).await?.status.is_terminal(),
            "precondition: the store's fail leaves child {} live",
            child.id,
        );
    }

    let outcome = task_store
        .cancel_tree(&parked.root.id, t_plus(2_030))
        .await?;
    let mut transitioned = outcome.transitioned.clone();
    transitioned.sort();
    let mut expected: Vec<AgentTaskId> = parked.children.iter().map(|c| c.id.clone()).collect();
    expected.sort();
    assert_eq!(
        transitioned, expected,
        "exactly the live descendants settle"
    );
    assert!(outcome.markers.is_empty(), "tool children carry no marker");
    for child in &parked.children {
        assert_settled(
            &fetch(task_store, &child.id).await?,
            &TerminalReason::ParentCancelled,
            child.id.as_str(),
        );
    }
    let root = fetch(task_store, &parked.root.id).await?;
    assert_eq!(
        root.status,
        TaskStatus::Failed,
        "the root keeps its outcome"
    );
    assert_eq!(root.terminal_reason, Some(TerminalReason::InternalError));

    let retry = task_store
        .cancel_tree(&parked.root.id, t_plus(2_040))
        .await?;
    assert!(retry.transitioned.is_empty(), "settlement is idempotent");
    assert!(retry.markers.is_empty());
    Ok(())
}

/// The same settlement crosses into a subagent's child thread: the live
/// invocation and its child-thread root are cancelled and the child thread
/// gets its closing marker, while the failed parent keeps its outcome.
pub(super) async fn test_cancel_tree_settles_child_thread_of_terminal_root(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
    event_repo: &dyn EventRepository,
) -> Result<()> {
    let (parent, invocation, child_root) =
        spawn_subagent_fixture(task_store, thread_store, "settle-terminal-subagent").await?;
    task_store
        .try_acquire_task(
            &child_root.id,
            WorkerId::new(),
            LeaseId::new(),
            t_plus(10_000),
            t_plus(3),
        )
        .await?
        .context("child root must acquire")?;
    let (worker, lease) = resume_root_early(task_store, &parent.id, 5).await?;
    task_store
        .fail_task_with_reason(
            &parent.id,
            &worker,
            &lease,
            "steering resume failed".to_owned(),
            TerminalReason::InternalError,
            t_plus(10),
        )
        .await?;

    let outcome = task_store.cancel_tree(&parent.id, t_plus(20)).await?;
    assert_eq!(outcome.transitioned.len(), 2, "{outcome:?}");
    assert_eq!(outcome.markers.len(), 1, "only the child thread closes");
    assert_eq!(outcome.markers[0].thread_id, child_root.thread_id);
    assert_eq!(
        cancelled_marker_count(event_repo, &parent.thread_id).await?,
        0,
        "the failed parent's thread gets no cancel marker",
    );
    assert_settled(
        &fetch(task_store, &invocation.id).await?,
        &TerminalReason::ParentCancelled,
        "invocation",
    );
    assert_settled(
        &fetch(task_store, &child_root.id).await?,
        &TerminalReason::ParentCancelled,
        "child root",
    );
    assert_eq!(
        fetch(task_store, &parent.id).await?.status,
        TaskStatus::Failed,
    );
    Ok(())
}

// ── Backend runners ──────────────────────────────────────────────────

mod in_memory {
    use anyhow::Result;

    use super::super::fresh_in_memory_stores;

    #[tokio::test]
    async fn settlement_in_memory_cancel_child_in_every_live_state_wakes_parent() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_child_in_every_live_state_wakes_parent(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_cancel_child_thread_root_wakes_invocation() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_child_thread_root_wakes_invocation(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn settlement_in_memory_terminal_root_settles_live_descendants() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_tree_settles_live_descendants_of_terminal_root(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_terminal_root_settles_child_thread() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_tree_settles_child_thread_of_terminal_root(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

}

#[cfg(feature = "sqlite")]
mod sqlite {
    use anyhow::Result;

    use crate::sqlite::SqliteDurableStore;

    async fn store() -> Result<SqliteDurableStore> {
        SqliteDurableStore::connect("sqlite::memory:").await
    }

    #[tokio::test]
    async fn settlement_sqlite_cancel_child_in_every_live_state_wakes_parent() -> Result<()> {
        let store = store().await?;
        super::test_cancel_child_in_every_live_state_wakes_parent(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_cancel_child_thread_root_wakes_invocation() -> Result<()> {
        let store = store().await?;
        super::test_cancel_child_thread_root_wakes_invocation(&store, &store, &store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_terminal_root_settles_live_descendants() -> Result<()> {
        let store = store().await?;
        super::test_cancel_tree_settles_live_descendants_of_terminal_root(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_terminal_root_settles_child_thread() -> Result<()> {
        let store = store().await?;
        super::test_cancel_tree_settles_child_thread_of_terminal_root(&store, &store, &store).await
    }

}

#[cfg(feature = "postgres")]
mod postgres {
    use anyhow::Result;

    use super::super::pg_test_store;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_child_in_every_live_state_wakes_parent() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_child_in_every_live_state_wakes_parent(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_child_thread_root_wakes_invocation() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_child_thread_root_wakes_invocation(&store, &store, &store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_terminal_root_settles_live_descendants() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_tree_settles_live_descendants_of_terminal_root(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_terminal_root_settles_child_thread() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_tree_settles_child_thread_of_terminal_root(&store, &store, &store).await
    }

}
