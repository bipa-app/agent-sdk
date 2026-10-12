//! Settlement conformance: every cancelled or orphaned task reaches a
//! terminal state on every backend, and a parked parent's fan-in wakes no
//! matter which live state its cancelled child was in.

use anyhow::{Context, Result};

use agent_sdk_foundation::TerminalReason;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::recovery::{FailureReason, RecoveryAction, RecoveryRecord};
use agent_server::journal::store::{AgentTaskStore, SubmitDisposition, SubmitRootTurnParams};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, TaskKind, TaskStatus,
    WorkerId,
};
use agent_server::journal::task_state::TaskState;
use agent_server::journal::thread::{PurgeScope, PurgeSeed};
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

/// Rows an older build that never knew `after_siblings_json` leased or
/// finished keep the wait list they carried. They must still load, on their
/// own and as children of their parent.
async fn test_stale_wait_lists_still_load(
    task_store: &dyn AgentTaskStore,
    parked: &ParkedRoot,
) -> Result<()> {
    for child in &parked.children[1..] {
        let row = task_store
            .get(&child.id)
            .await?
            .context("a row with a stale wait list must load")?;
        anyhow::ensure!(
            row.after_siblings.is_empty(),
            "a wait list on a {:?} row is ignored",
            row.status
        );
    }
    let children = task_store.list_children(&parked.root.id).await?;
    anyhow::ensure!(
        children.len() == parked.children.len(),
        "every child must list, got {}",
        children.len()
    );
    task_store.cancel_tree(&parked.root.id, t_plus(6)).await?;
    for child in task_store.list_children(&parked.root.id).await? {
        anyhow::ensure!(
            child.status.is_terminal(),
            "cancelling the tree must settle {} ({:?})",
            child.id,
            child.status
        );
    }
    Ok(())
}

/// A tool child settles: the sibling held behind it is hinted to the pool,
/// and a worker leases it by id from the hint alone.
async fn test_a_settled_child_hints_the_sibling_it_released(
    task_store: &dyn AgentTaskStore,
    tid: &str,
) -> Result<()> {
    let root = AgentTask::new_root_turn(thread_id(tid), t_plus(0), 3);
    task_store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            t_plus(10_000),
            t_plus(1),
        )
        .await?
        .context("root must acquire")?;
    let (_parent, children) = task_store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![
                ChildSpawnSpec::new(1),
                ChildSpawnSpec::new(1).after_siblings(vec![0]),
            ],
            suspension_payload(tid),
            None,
            Vec::new(),
            t_plus(2),
        )
        .await?;
    let [first, held] = children.as_slice() else {
        anyhow::bail!("two children must be spawned");
    };
    let child_worker = WorkerId::new();
    let child_lease = LeaseId::new();
    task_store
        .try_acquire_task(
            &first.id,
            child_worker.clone(),
            child_lease.clone(),
            t_plus(10_000),
            t_plus(3),
        )
        .await?
        .context("the first child must acquire")?;
    task_store
        .complete_task_with_result(
            &first.id,
            &child_worker,
            &child_lease,
            serde_json::json!({"ok": true}),
            t_plus(4),
        )
        .await?;

    let signal = agent_server::journal::WakeupSignal::default();
    crate::host::hint_released_siblings(&signal, task_store, &root.id).await;
    let leased = crate::host::acquire_hinted_task(
        task_store,
        &signal,
        &WorkerId::new(),
        time::Duration::minutes(10),
    )
    .await
    .context("the hinted sibling must be leased by id")?;
    anyhow::ensure!(
        leased.id == held.id && leased.status == TaskStatus::Running,
        "the hint must lease the released sibling, got {leased:?}"
    );
    anyhow::ensure!(!signal.has_hints(), "the hint was consumed");
    Ok(())
}

/// Claim every runnable row through the acquisition scan.
async fn claim_all_runnable(task_store: &dyn AgentTaskStore, at: i64) -> Result<Vec<AgentTask>> {
    let mut claimed = Vec::new();
    while let Some(task) = task_store
        .acquire_next_runnable(
            WorkerId::new(),
            LeaseId::new(),
            t_plus(at + 10_000),
            t_plus(at),
        )
        .await?
    {
        claimed.push(task);
        anyhow::ensure!(claimed.len() < 10_000, "the runnable scan never drained");
    }
    Ok(claimed)
}

/// `parked.children[1]` was requeued by an older build with its wait list
/// on `parked.children[0]` intact. The scan holds it while that sibling
/// runs and hands it out once the sibling finishes, with no write to it in
/// between.
async fn test_requeued_wait_list_is_acquired_once_its_siblings_finish(
    task_store: &dyn AgentTaskStore,
    parked: &ParkedRoot,
) -> Result<()> {
    let [first, requeued, _] = parked.children.as_slice() else {
        anyhow::bail!("three children must be parked");
    };
    let held = task_store
        .get(&requeued.id)
        .await?
        .context("the requeued row must load")?;
    anyhow::ensure!(
        held.status == TaskStatus::Pending && held.after_siblings == [first.id.clone()],
        "the requeued row keeps its wait list: {held:?}"
    );

    let claimed = claim_all_runnable(task_store, 5).await?;
    let first_claim = claimed
        .iter()
        .find(|task| task.id == first.id)
        .context("the sibling it waits for must be acquired")?;
    anyhow::ensure!(
        claimed.iter().all(|task| task.id != requeued.id),
        "a row whose sibling still runs must not be acquired"
    );

    task_store
        .complete_task_with_result(
            &first.id,
            first_claim.worker_id.as_ref().context("claimed worker")?,
            first_claim.lease_id.as_ref().context("claimed lease")?,
            serde_json::json!({"ok": true}),
            t_plus(6),
        )
        .await?;
    let untouched = task_store
        .get(&requeued.id)
        .await?
        .context("the requeued row must load")?;
    anyhow::ensure!(
        untouched.after_siblings == [first.id.clone()] && untouched.updated_at == held.updated_at,
        "the sibling finishing must not write the waiting row"
    );

    let claimed = claim_all_runnable(task_store, 7).await?;
    let leased = claimed
        .iter()
        .find(|task| task.id == requeued.id)
        .context("the row must be acquired once its sibling finished")?;
    anyhow::ensure!(
        leased.after_siblings.is_empty(),
        "acquisition clears the wait list it admitted"
    );
    Ok(())
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

/// Approving a confirmation continues the attempt that paused: the pickup
/// re-enters that attempt instead of charging a new one, so a child whose
/// paused execution used its whole budget can still run once approved.
/// The budget still bounds a pickup that keeps crashing.
pub(super) async fn test_approved_confirmation_pickup_keeps_the_paused_attempt(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    for max_attempts in [1_u32, 3] {
        let tid = format!("settle-approve-{max_attempts}");
        let at = 3_000 + i64::from(max_attempts) * 100;
        let parked = park_root_on_tool_children(task_store, &tid, 1, max_attempts, at).await?;
        let child_id = parked.children[0].id.clone();
        drive_child(
            task_store,
            &tid,
            &child_id,
            LiveChild::AwaitingConfirmation,
            at + 5,
        )
        .await?;
        assert_eq!(fetch(task_store, &child_id).await?.attempt, 1);

        let (approved, _prepared) = task_store
            .resume_from_confirmation(&child_id, t_plus(at + 10))
            .await?;
        assert_eq!(approved.status, TaskStatus::Pending);

        let picked_up = task_store
            .try_acquire_task(
                &child_id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(at + 12),
                t_plus(at + 11),
            )
            .await?
            .with_context(|| {
                format!("max_attempts={max_attempts}: the approved child must be acquirable")
            })?;
        assert_eq!(picked_up.status, TaskStatus::Running);
        assert_eq!(
            picked_up.attempt, 1,
            "max_attempts={max_attempts}: the pickup continues the paused attempt",
        );

        if max_attempts == 1 {
            let records = task_store.release_expired_leases(t_plus(at + 13)).await?;
            let record = records
                .iter()
                .find(|record| record.id == child_id)
                .context("the crashed pickup's lease must be reclaimed")?;
            assert!(
                matches!(
                    record.action,
                    RecoveryAction::FailClosed(FailureReason::LeaseExpiredBudgetExhausted)
                ),
                "a crashed pickup is still bounded by the budget, got {:?}",
                record.action,
            );
            let parent = fetch(task_store, &parked.root.id).await?;
            assert_eq!(parent.status, TaskStatus::Pending, "the parent fans in");
        }
    }
    Ok(())
}

/// A live row a root can still own after it ended.
#[derive(Clone, Copy, Debug)]
enum LiveDescendant {
    Tool(LiveChild),
    /// Undelivered input under an ended root, as a store from before root
    /// transitions settled their input could still hold it; written here
    /// through the structural `update`, which bypasses every transition.
    StrandedInjection,
    ParkedSubagentInvocation,
}

const TASK_STATUSES: [TaskStatus; 9] = [
    TaskStatus::Queued,
    TaskStatus::Pending,
    TaskStatus::Running,
    TaskStatus::WaitingOnChildren,
    TaskStatus::AwaitingConfirmation,
    TaskStatus::AwaitingQuestion,
    TaskStatus::Completed,
    TaskStatus::Failed,
    TaskStatus::Cancelled,
];

/// The descendant that holds `status` while live, so every live status a
/// descendant can be in is proven to be found.
const fn live_descendant_in(status: TaskStatus) -> Option<LiveDescendant> {
    match status {
        TaskStatus::Queued => Some(LiveDescendant::StrandedInjection),
        TaskStatus::Pending => Some(LiveDescendant::Tool(LiveChild::Pending)),
        TaskStatus::Running => Some(LiveDescendant::Tool(LiveChild::RunningLiveLease)),
        TaskStatus::WaitingOnChildren => Some(LiveDescendant::ParkedSubagentInvocation),
        TaskStatus::AwaitingConfirmation => {
            Some(LiveDescendant::Tool(LiveChild::AwaitingConfirmation))
        }
        // Only a root turn parks on a question.
        TaskStatus::AwaitingQuestion
        | TaskStatus::Completed
        | TaskStatus::Failed
        | TaskStatus::Cancelled => None,
    }
}

async fn running_root(
    task_store: &dyn AgentTaskStore,
    tid: &str,
    at: i64,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let root = AgentTask::new_root_turn(thread_id(tid), t_plus(at), 3);
    task_store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    let running = task_store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            t_plus(at + 10_000),
            t_plus(at + 1),
        )
        .await?
        .context("root must acquire")?;
    Ok((running, worker, lease))
}

/// Submit `text` while `tid`'s root is running, so it waits as an
/// injection for that turn's next boundary.
async fn inject(
    task_store: &dyn AgentTaskStore,
    tid: &str,
    text: &str,
    at: i64,
) -> Result<AgentTask> {
    let outcome = task_store
        .submit_root_turn_idempotent(SubmitRootTurnParams {
            task: AgentTask::new_root_turn_with_input(
                thread_id(tid),
                vec![SubmittedInputItem::Text { text: text.into() }],
                t_plus(at),
                3,
            ),
            idempotency: None,
            max_queued_depth: None,
            disposition: SubmitDisposition::InjectAtBoundary,
        })
        .await
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    assert_eq!(outcome.task.kind, TaskKind::InputInjection);
    assert_eq!(outcome.task.status, TaskStatus::Queued);
    Ok(outcome.task)
}

/// A root that ended while `descendant` was still live under it.
async fn end_root_over(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
    tid: &str,
    descendant: LiveDescendant,
    at: i64,
) -> Result<AgentTaskId> {
    match descendant {
        LiveDescendant::Tool(state) => {
            let parked = park_root_on_tool_children(task_store, tid, 2, 3, at).await?;
            for child in &parked.children {
                drive_child(task_store, tid, &child.id, state, at + 5).await?;
            }
            let (worker, lease) = resume_root_early(task_store, &parked.root.id, at + 10).await?;
            task_store
                .fail_task_with_reason(
                    &parked.root.id,
                    &worker,
                    &lease,
                    "steering resume failed".to_owned(),
                    TerminalReason::InternalError,
                    t_plus(at + 20),
                )
                .await?;
            Ok(parked.root.id)
        }
        LiveDescendant::StrandedInjection => {
            let (root, _worker, _lease) = running_root(task_store, tid, at).await?;
            inject(task_store, tid, "one more thing", at + 5).await?;
            task_store
                .update(root.clone().complete(t_plus(at + 10))?)
                .await?;
            Ok(root.id)
        }
        LiveDescendant::ParkedSubagentInvocation => {
            let (parent, _invocation, _child_root) =
                spawn_subagent_fixture(task_store, thread_store, tid).await?;
            let (worker, lease) = resume_root_early(task_store, &parent.id, at).await?;
            task_store
                .fail_task_with_reason(
                    &parent.id,
                    &worker,
                    &lease,
                    "steering resume failed".to_owned(),
                    TerminalReason::InternalError,
                    t_plus(at + 10),
                )
                .await?;
            Ok(parent.id)
        }
    }
}

/// The orphan listing names exactly the ended roots that still own a live
/// descendant, whichever live status that descendant holds, once each, in id
/// order. A live tree, an ended root whose children all finished, and a
/// childless ended root are not listed, and a settled root drops out.
pub(super) async fn test_list_terminal_roots_with_live_descendants(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
) -> Result<()> {
    let mut orphaned = Vec::new();
    for (index, status) in TASK_STATUSES.into_iter().enumerate() {
        let Some(descendant) = live_descendant_in(status) else {
            continue;
        };
        let at = 5_000 + i64::try_from(index)? * 100;
        let tid = format!("orphan-listing-{index}");
        orphaned.push(end_root_over(task_store, thread_store, &tid, descendant, at).await?);
    }

    park_root_on_tool_children(task_store, "orphan-listing-live", 2, 3, 6_000).await?;

    let finished =
        park_root_on_tool_children(task_store, "orphan-listing-done", 1, 3, 6_100).await?;
    complete_child(task_store, &finished.children[0].id, 6_105).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            &finished.root.id,
            worker.clone(),
            lease.clone(),
            t_plus(16_110),
            t_plus(6_110),
        )
        .await?
        .context("woken root must acquire")?;
    task_store
        .complete_task(&finished.root.id, &worker, &lease, t_plus(6_115))
        .await?;

    let (childless, worker, lease) = running_root(task_store, "orphan-listing-bare", 6_200).await?;
    task_store
        .complete_task(&childless.id, &worker, &lease, t_plus(6_205))
        .await?;

    orphaned.sort();
    assert_eq!(
        task_store
            .list_terminal_roots_with_live_descendants()
            .await?,
        orphaned,
    );

    for root in &orphaned {
        task_store.cancel_tree(root, t_plus(7_000)).await?;
    }
    assert!(
        task_store
            .list_terminal_roots_with_live_descendants()
            .await?
            .is_empty(),
        "a settled root is no longer listed",
    );
    Ok(())
}

/// The ways a root turn ends without a cancel.
#[derive(Clone, Copy, Debug)]
enum RootEnd {
    Completed,
    Failed,
    /// Its last lease lapsed and the sweep failed it closed.
    FailedClosed,
}

const ROOT_ENDS: [RootEnd; 3] = [RootEnd::Completed, RootEnd::Failed, RootEnd::FailedClosed];

/// Lease `id` for one attempt that `end` can finish: a lease that lapses
/// almost at once for [`RootEnd::FailedClosed`], a long one otherwise.
async fn lease_for(
    task_store: &dyn AgentTaskStore,
    id: &AgentTaskId,
    end: RootEnd,
    at: i64,
) -> Result<(WorkerId, LeaseId)> {
    let lease_expires_at = match end {
        RootEnd::Completed | RootEnd::Failed => t_plus(at + 10_000),
        RootEnd::FailedClosed => t_plus(at + 1),
    };
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    task_store
        .try_acquire_task(
            id,
            worker.clone(),
            lease.clone(),
            lease_expires_at,
            t_plus(at),
        )
        .await?
        .with_context(|| format!("{id} must acquire"))?;
    Ok((worker, lease))
}

/// End the root `id`, leased by [`lease_for`], the way `end` says.
async fn end_root(
    task_store: &dyn AgentTaskStore,
    id: &AgentTaskId,
    (worker, lease): (WorkerId, LeaseId),
    end: RootEnd,
    at: i64,
) -> Result<()> {
    match end {
        RootEnd::Completed => {
            task_store
                .complete_task(id, &worker, &lease, t_plus(at))
                .await?;
        }
        RootEnd::Failed => {
            task_store
                .fail_task_with_reason(
                    id,
                    &worker,
                    &lease,
                    "provider refused".to_owned(),
                    TerminalReason::InternalError,
                    t_plus(at),
                )
                .await?;
        }
        RootEnd::FailedClosed => {
            let mut now = at;
            loop {
                let records = task_store.release_expired_leases(t_plus(now)).await?;
                let record = records
                    .iter()
                    .find(|record| record.id == *id)
                    .with_context(|| format!("the lapsed lease of {id} must be reclaimed"))?;
                match &record.action {
                    RecoveryAction::FailClosed(_) => break,
                    RecoveryAction::Requeue => {
                        lease_for(task_store, id, RootEnd::FailedClosed, now + 1).await?;
                        now += 10;
                    }
                    RecoveryAction::NoAction => {
                        anyhow::bail!("a lapsed lease of {id} must be acted on")
                    }
                }
            }
        }
    }
    Ok(())
}

async fn root_turns_of(task_store: &dyn AgentTaskStore, tid: &str) -> Result<Vec<AgentTask>> {
    let mut turns: Vec<AgentTask> = task_store
        .list_by_thread(&thread_id(tid))
        .await?
        .into_iter()
        .filter(|task| task.kind == TaskKind::RootTurn)
        .collect();
    turns.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id)));
    Ok(turns)
}

fn text_of(task: &AgentTask) -> Result<serde_json::Value> {
    Ok(serde_json::to_value(&task.submitted_input)?)
}

/// However a top-level root ends on its own, the input injected into it that
/// no boundary delivered becomes the thread's next turns in the same step,
/// each keeping its submission time — so the thread runs it in submission
/// order with what was queued as the next turn meanwhile: input sent before
/// a queued successor runs first, input sent after it waits behind it.
pub(super) async fn test_root_end_hands_undelivered_input_over_in_submission_order(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    for (index, end) in ROOT_ENDS.into_iter().enumerate() {
        let tid = format!("settle-handover-{index}");
        let at = 8_000 + i64::try_from(index)? * 1_000;
        let max_attempts = match end {
            RootEnd::Completed | RootEnd::Failed => 3,
            RootEnd::FailedClosed => 1,
        };
        let root = AgentTask::new_root_turn(thread_id(&tid), t_plus(at), max_attempts);
        task_store.submit_root_turn(root.clone()).await?;
        let lease = lease_for(task_store, &root.id, end, at + 1).await?;
        let before = inject(task_store, &tid, "before the successor", at + 2).await?;
        let successor = task_store
            .submit_root_turn(AgentTask::new_root_turn(thread_id(&tid), t_plus(at + 3), 3))
            .await?;
        assert_eq!(successor.status, TaskStatus::Queued, "{end:?}");
        let after = inject(task_store, &tid, "after the successor", at + 4).await?;

        end_root(task_store, &root.id, lease, end, at + 10).await?;

        for injection in [&before, &after] {
            let handed_over = fetch(task_store, &injection.id).await?;
            assert_eq!(handed_over.status, TaskStatus::Completed, "{end:?}");
            assert_eq!(
                handed_over.terminal_reason,
                Some(TerminalReason::Completed),
                "{end:?}",
            );
        }
        let turns = root_turns_of(task_store, &tid).await?;
        let next: Vec<&AgentTask> = turns.iter().filter(|turn| turn.id != root.id).collect();
        assert_eq!(next.len(), 3, "{end:?}: {turns:?}");
        assert_eq!(text_of(next[0])?, text_of(&before)?, "{end:?}");
        assert_eq!(next[0].created_at, before.created_at, "{end:?}");
        assert_eq!(next[0].status, TaskStatus::Pending, "{end:?}");
        assert_eq!(next[1].id, successor.id, "{end:?}");
        assert_eq!(next[1].status, TaskStatus::Queued, "{end:?}");
        assert_eq!(text_of(next[2])?, text_of(&after)?, "{end:?}");
        assert_eq!(next[2].status, TaskStatus::Queued, "{end:?}");
        assert!(
            !task_store
                .list_terminal_roots_with_live_descendants()
                .await?
                .contains(&root.id),
            "{end:?}: the ended root holds no live input",
        );

        let first = lease_for(task_store, &next[0].id, RootEnd::Completed, at + 20).await?;
        end_root(task_store, &next[0].id, first, RootEnd::Completed, at + 21).await?;
        assert_eq!(
            fetch(task_store, &successor.id).await?.status,
            TaskStatus::Pending,
            "{end:?}: the successor runs next",
        );
        assert_eq!(
            fetch(task_store, &next[2].id).await?.status,
            TaskStatus::Queued,
            "{end:?}: the later input waits behind it",
        );
    }
    Ok(())
}

/// However a subagent's child-thread root ends on its own, the input
/// injected into it that no boundary delivered dies with it: the invocation
/// takes the root's outcome, and no turn starts on the child thread — one
/// would run unowned beside the parent, with no deadline and nobody
/// awaiting its result.
pub(super) async fn test_child_thread_root_end_discards_undelivered_input(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
) -> Result<()> {
    for (index, end) in ROOT_ENDS.into_iter().enumerate() {
        let tid = format!("settle-child-input-{index}");
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, &tid).await?;
        let child_tid = child_root.thread_id.0.clone();
        let lease = lease_for(task_store, &child_root.id, end, 3).await?;
        let injection = inject(task_store, &child_tid, "look at the logs too", 4).await?;
        assert_eq!(
            injection.parent_id.as_ref(),
            Some(&child_root.id),
            "{end:?}"
        );

        end_root(task_store, &child_root.id, lease, end, 10).await?;

        let turns = root_turns_of(task_store, &child_tid).await?;
        assert_eq!(
            turns.iter().map(|turn| &turn.id).collect::<Vec<_>>(),
            vec![&child_root.id],
            "{end:?}: no turn may start on the subagent's thread",
        );
        assert_settled(
            &fetch(task_store, &injection.id).await?,
            &TerminalReason::ParentCancelled,
            &format!("{end:?} undelivered input"),
        );
        assert_eq!(
            fetch(task_store, &invocation.id).await?.status,
            TaskStatus::Pending,
            "{end:?}: the invocation takes the child's outcome",
        );
        assert!(
            !task_store
                .list_terminal_roots_with_live_descendants()
                .await?
                .contains(&child_root.id),
            "{end:?}: the ended child root holds no live input",
        );
    }
    Ok(())
}

/// Input left undelivered by a root that ends while its thread is being
/// purged is discarded: the purge fence admits no new turn.
pub(super) async fn test_root_end_on_a_purging_thread_discards_undelivered_input(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
) -> Result<()> {
    let tid = "settle-purging-input";
    thread_store
        .get_or_create(&thread_id(tid), t_plus(9_000))
        .await?;
    let (root, worker, lease) = running_root(task_store, tid, 9_000).await?;
    let injection = inject(task_store, tid, "one more thing", 9_002).await?;
    thread_store
        .begin_purge(
            &thread_id(tid),
            PurgeSeed {
                root_thread_id: thread_id(tid),
                scope: PurgeScope::Thread,
                started_at: t_plus(9_003),
            },
        )
        .await?;

    end_root(
        task_store,
        &root.id,
        (worker, lease),
        RootEnd::Failed,
        9_010,
    )
    .await?;

    assert_settled(
        &fetch(task_store, &injection.id).await?,
        &TerminalReason::ParentCancelled,
        "undelivered input on a purging thread",
    );
    assert_eq!(
        root_turns_of(task_store, tid)
            .await?
            .iter()
            .map(|turn| &turn.id)
            .collect::<Vec<_>>(),
        vec![&root.id],
    );
    Ok(())
}

/// Rejecting a confirmation fails a tool child and lets its parent fan in.
/// A root paused on a confirmation is refused: failing it there would end
/// the root without the settlement its own terminal transitions carry.
pub(super) async fn test_reject_confirmation_refuses_a_root(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    let tid = "settle-reject-root";
    let (root, worker, lease) = running_root(task_store, tid, 10_000).await?;
    task_store
        .pause_on_confirmation(
            &root.id,
            &worker,
            &lease,
            suspension_payload(tid).continuation,
            None,
            t_plus(10_002),
        )
        .await?;

    let refused = task_store
        .reject_confirmation(&root.id, "denied".to_owned(), t_plus(10_003))
        .await;
    assert!(
        refused.is_err(),
        "a root's confirmation must not be rejected: {refused:?}"
    );
    assert_eq!(
        fetch(task_store, &root.id).await?.status,
        TaskStatus::AwaitingConfirmation,
    );
    Ok(())
}

/// A failing root's settlement cancels every live child under it, whatever
/// state each is in, but leaves the root itself running and the input
/// waiting on it queued for the root's own terminal transition. It refuses,
/// cancelling nothing, a caller that no longer holds the root.
pub(super) async fn test_cancel_descendants_spares_the_root_and_its_input(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    let tid = "settle-descendants";
    let parked = park_root_on_tool_children(task_store, tid, 3, 3, 11_000).await?;
    let children: Vec<AgentTaskId> = parked.children.iter().map(|c| c.id.clone()).collect();
    drive_child(
        task_store,
        tid,
        &children[0],
        LiveChild::AwaitingConfirmation,
        11_005,
    )
    .await?;
    drive_child(
        task_store,
        tid,
        &children[1],
        LiveChild::RunningLiveLease,
        11_007,
    )
    .await?;
    let (worker, lease) = resume_root_early(task_store, &parked.root.id, 11_010).await?;
    let pending = inject(task_store, tid, "one more thing", 11_012).await?;

    let refused = task_store
        .cancel_descendants_of_owned(&parked.root.id, &worker, &LeaseId::new(), t_plus(11_015))
        .await;
    assert!(
        refused.is_err(),
        "a stale lease must be refused: {refused:?}"
    );
    for child in &children {
        assert!(
            !fetch(task_store, child).await?.status.is_terminal(),
            "a refused settlement cancels nothing",
        );
    }

    let outcome = task_store
        .cancel_descendants_of_owned(&parked.root.id, &worker, &lease, t_plus(11_020))
        .await?;
    let mut transitioned = outcome.transitioned.clone();
    transitioned.sort();
    let mut expected = children.clone();
    expected.sort();
    assert_eq!(transitioned, expected, "exactly the live children settle");
    assert!(outcome.markers.is_empty(), "tool children carry no marker");
    for child in &children {
        assert_settled(
            &fetch(task_store, child).await?,
            &TerminalReason::ParentCancelled,
            child.as_str(),
        );
    }
    assert_eq!(
        fetch(task_store, &pending.id).await?.status,
        TaskStatus::Queued,
        "the input waits for the root's own terminal transition",
    );
    let root = fetch(task_store, &parked.root.id).await?;
    assert_eq!(
        root.status,
        TaskStatus::Running,
        "the root itself is untouched"
    );
    assert_eq!(root.lease_id.as_ref(), Some(&lease));
    Ok(())
}

/// The settlement crosses into a subagent's child thread: the invocation
/// and its child-thread root are cancelled, the child thread gets its
/// closing marker, and the failing parent's own thread gets none.
pub(super) async fn test_cancel_descendants_closes_child_threads(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
    event_repo: &dyn EventRepository,
) -> Result<()> {
    let (parent, invocation, child_root) =
        spawn_subagent_fixture(task_store, thread_store, "settle-descendants-subagent").await?;
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

    let outcome = task_store
        .cancel_descendants_of_owned(&parent.id, &worker, &lease, t_plus(10))
        .await?;
    assert_eq!(outcome.transitioned.len(), 2, "{outcome:?}");
    assert_eq!(outcome.markers.len(), 1, "only the child thread closes");
    assert_eq!(outcome.markers[0].thread_id, child_root.thread_id);
    assert_eq!(
        cancelled_marker_count(event_repo, &parent.thread_id).await?,
        0
    );
    for (task, label) in [(&invocation, "invocation"), (&child_root, "child root")] {
        assert_settled(
            &fetch(task_store, &task.id).await?,
            &TerminalReason::ParentCancelled,
            label,
        );
    }
    assert_eq!(
        fetch(task_store, &parent.id).await?.status,
        TaskStatus::Running
    );
    Ok(())
}

/// Wake `root` early whenever it is parked and lease it with a lease that
/// lapses at once, until the sweep fails it closed; return that record.
async fn fail_closed_by_lease_expiry(
    task_store: &dyn AgentTaskStore,
    root: &AgentTaskId,
    mut at: i64,
) -> Result<RecoveryRecord> {
    for _ in 0..10 {
        if fetch(task_store, root).await?.status == TaskStatus::WaitingOnChildren {
            let note = vec![agent_sdk_foundation::llm::ContentBlock::Text {
                text: "status?".into(),
            }];
            task_store
                .enqueue_steering_resume(root, note, t_plus(at))
                .await?
                .context("parked root must wake")?;
        }
        task_store
            .try_acquire_task(
                root,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(at + 2),
                t_plus(at + 1),
            )
            .await?
            .context("woken root must acquire")?;
        let records = task_store.release_expired_leases(t_plus(at + 3)).await?;
        if let Some(record) = records
            .into_iter()
            .find(|record| record.id == *root && record.action.is_fail_closed())
        {
            return Ok(record);
        }
        at += 10;
    }
    anyhow::bail!("{root} was never failed closed")
}

/// A root failed closed when its last lease lapses settles its live
/// children in that same step, before the turn queued behind it is
/// promoted, and reports what it settled.
pub(super) async fn test_fail_close_settles_children_before_the_successor(
    task_store: &dyn AgentTaskStore,
) -> Result<()> {
    let tid = "settle-fail-close";
    let parked = park_root_on_tool_children(task_store, tid, 2, 3, 12_000).await?;
    let children: Vec<AgentTaskId> = parked.children.iter().map(|c| c.id.clone()).collect();
    drive_child(
        task_store,
        tid,
        &children[0],
        LiveChild::RunningLiveLease,
        12_005,
    )
    .await?;
    let successor = AgentTask::new_root_turn(thread_id(tid), t_plus(12_006), 3);
    let successor = task_store.submit_root_turn(successor).await?;
    assert_eq!(successor.status, TaskStatus::Queued);

    let record = fail_closed_by_lease_expiry(task_store, &parked.root.id, 12_010).await?;

    for child in &children {
        assert_settled(
            &fetch(task_store, child).await?,
            &TerminalReason::ParentCancelled,
            child.as_str(),
        );
    }
    let mut settled = record.settled.transitioned.clone();
    settled.sort();
    let mut expected = children.clone();
    expected.sort();
    assert_eq!(settled, expected, "the record names every child it settled");
    assert_eq!(
        fetch(task_store, &parked.root.id).await?.status,
        TaskStatus::Failed
    );
    assert_eq!(
        fetch(task_store, &successor.id).await?.status,
        TaskStatus::Pending,
        "the successor is promoted",
    );
    assert!(
        !task_store
            .list_terminal_roots_with_live_descendants()
            .await?
            .contains(&parked.root.id),
        "nothing is left for the orphan sweep",
    );
    Ok(())
}

/// The same fail-close crosses into a subagent's child thread and reports
/// the closing marker it committed there.
pub(super) async fn test_fail_close_closes_child_threads(
    task_store: &dyn AgentTaskStore,
    thread_store: &dyn ThreadStore,
) -> Result<()> {
    let (parent, invocation, child_root) =
        spawn_subagent_fixture(task_store, thread_store, "settle-fail-close-subagent").await?;
    task_store
        .try_acquire_task(
            &child_root.id,
            WorkerId::new(),
            LeaseId::new(),
            t_plus(100_000),
            t_plus(3),
        )
        .await?
        .context("child root must acquire")?;

    let record = fail_closed_by_lease_expiry(task_store, &parent.id, 13_000).await?;

    for (task, label) in [(&invocation, "invocation"), (&child_root, "child root")] {
        assert_settled(
            &fetch(task_store, &task.id).await?,
            &TerminalReason::ParentCancelled,
            label,
        );
    }
    assert_eq!(record.settled.markers.len(), 1, "{:?}", record.settled);
    assert_eq!(record.settled.markers[0].thread_id, child_root.thread_id);
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

    #[tokio::test]
    async fn settlement_in_memory_approved_confirmation_keeps_the_paused_attempt() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_approved_confirmation_pickup_keeps_the_paused_attempt(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_lists_terminal_roots_with_live_descendants() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_list_terminal_roots_with_live_descendants(s.task.as_ref(), s.thread.as_ref())
            .await
    }

    #[tokio::test]
    async fn settlement_in_memory_root_end_hands_input_over_in_order() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_root_end_hands_undelivered_input_over_in_submission_order(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_child_thread_root_end_discards_input() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_child_thread_root_end_discards_undelivered_input(
            s.task.as_ref(),
            s.thread.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn settlement_in_memory_cancel_descendants_spares_the_root() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_descendants_spares_the_root_and_its_input(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_cancel_descendants_closes_child_threads() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_cancel_descendants_closes_child_threads(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn settlement_in_memory_fail_close_settles_children_first() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_fail_close_settles_children_before_the_successor(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_fail_close_closes_child_threads() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_fail_close_closes_child_threads(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_reject_confirmation_refuses_a_root() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_reject_confirmation_refuses_a_root(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn settlement_in_memory_root_end_on_a_purging_thread_discards_input() -> Result<()> {
        let s = fresh_in_memory_stores();
        super::test_root_end_on_a_purging_thread_discards_undelivered_input(
            s.task.as_ref(),
            s.thread.as_ref(),
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
    async fn settlement_sqlite_stale_wait_lists_from_an_older_build_still_load() -> Result<()> {
        let store = store().await?;
        let parked =
            super::park_root_on_tool_children(&store, "sqlite-stale-waits", 3, 3, 0).await?;
        let waits = serde_json::json!([parked.children[0].id.as_str()]);
        let at = super::t_plus(5);
        sqlx::query(
            "UPDATE agent_sdk_tasks SET status = 'running', worker_id = 'older-build', \
             lease_id = 'older-lease', lease_expires_at = ?2, last_heartbeat_at = ?2, \
             attempt = 1, after_siblings_json = ?3 WHERE id = ?1",
        )
        .bind(parked.children[1].id.as_str())
        .bind(at)
        .bind(&waits)
        .execute(store.pool())
        .await?;
        sqlx::query(
            "UPDATE agent_sdk_tasks SET status = 'completed', completed_at = ?2, \
             terminal_reason_json = ?3, after_siblings_json = ?4 WHERE id = ?1",
        )
        .bind(parked.children[2].id.as_str())
        .bind(at)
        .bind(serde_json::to_value(super::TerminalReason::Completed)?)
        .bind(&waits)
        .execute(store.pool())
        .await?;
        super::test_stale_wait_lists_still_load(&store, &parked).await
    }

    #[tokio::test]
    async fn settlement_sqlite_a_settled_child_hints_the_sibling_it_released() -> Result<()> {
        let store = store().await?;
        super::test_a_settled_child_hints_the_sibling_it_released(&store, "sqlite-hint").await
    }

    #[tokio::test]
    async fn settlement_sqlite_requeued_wait_list_from_an_older_build_runs_in_order() -> Result<()>
    {
        let store = store().await?;
        let parked =
            super::park_root_on_tool_children(&store, "sqlite-requeued-waits", 3, 3, 0).await?;
        sqlx::query(
            "UPDATE agent_sdk_tasks SET attempt = 1, after_siblings_json = ?2 WHERE id = ?1",
        )
        .bind(parked.children[1].id.as_str())
        .bind(serde_json::json!([parked.children[0].id.as_str()]))
        .execute(store.pool())
        .await?;
        super::test_requeued_wait_list_is_acquired_once_its_siblings_finish(&store, &parked).await
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

    #[tokio::test]
    async fn settlement_sqlite_approved_confirmation_keeps_the_paused_attempt() -> Result<()> {
        let store = store().await?;
        super::test_approved_confirmation_pickup_keeps_the_paused_attempt(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_lists_terminal_roots_with_live_descendants() -> Result<()> {
        let store = store().await?;
        super::test_list_terminal_roots_with_live_descendants(&store, &store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_root_end_hands_input_over_in_order() -> Result<()> {
        let store = store().await?;
        super::test_root_end_hands_undelivered_input_over_in_submission_order(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_child_thread_root_end_discards_input() -> Result<()> {
        let store = store().await?;
        super::test_child_thread_root_end_discards_undelivered_input(&store, &store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_cancel_descendants_spares_the_root() -> Result<()> {
        let store = store().await?;
        super::test_cancel_descendants_spares_the_root_and_its_input(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_cancel_descendants_closes_child_threads() -> Result<()> {
        let store = store().await?;
        super::test_cancel_descendants_closes_child_threads(&store, &store, &store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_fail_close_settles_children_first() -> Result<()> {
        let store = store().await?;
        super::test_fail_close_settles_children_before_the_successor(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_fail_close_closes_child_threads() -> Result<()> {
        let store = store().await?;
        super::test_fail_close_closes_child_threads(&store, &store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_reject_confirmation_refuses_a_root() -> Result<()> {
        let store = store().await?;
        super::test_reject_confirmation_refuses_a_root(&store).await
    }

    #[tokio::test]
    async fn settlement_sqlite_root_end_on_a_purging_thread_discards_input() -> Result<()> {
        let store = store().await?;
        super::test_root_end_on_a_purging_thread_discards_undelivered_input(&store, &store).await
    }
}

#[cfg(feature = "postgres")]
mod postgres {
    use anyhow::{Context, Result};
    use time::OffsetDateTime;

    use agent_sdk_foundation::{PendingToolCallInfo, TerminalReason, ToolTier};
    use agent_server::journal::store::{
        AgentTaskStore, MixedChildrenSpawn, ReattachedChild, SubmitDisposition,
        SubmitRootTurnParams, ToolChildSpawn,
    };
    use agent_server::journal::task::{
        AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
        TaskKind, TaskStatus, WorkerId,
    };

    use super::super::{pg_test_store, spawn_subagent_fixture, t_plus, thread_id};
    use crate::postgres::store::PostgresDurableStore;

    const RACE_ROUNDS: i64 = 40;

    fn injection_at(tid: &str, text: &str, at: i64) -> SubmitRootTurnParams {
        SubmitRootTurnParams {
            task: AgentTask::new_root_turn_with_input(
                thread_id(tid),
                vec![SubmittedInputItem::Text { text: text.into() }],
                t_plus(at),
                3,
            ),
            idempotency: None,
            max_queued_depth: None,
            disposition: SubmitDisposition::InjectAtBoundary,
        }
    }

    /// Rows still live under a root that has ended: nothing will ever
    /// deliver or run them.
    async fn stranded_rows(store: &PostgresDurableStore, tid: &str) -> Result<Vec<AgentTaskId>> {
        let tasks = store.list_by_thread(&thread_id(tid)).await?;
        let ended = |id: &AgentTaskId| {
            tasks
                .iter()
                .any(|task| task.id == *id && task.status.is_terminal())
        };
        Ok(tasks
            .iter()
            .filter(|task| task.parent_id.is_some() && !task.status.is_terminal())
            .filter(|task| ended(&task.root_id))
            .map(|task| task.id.clone())
            .collect())
    }

    /// The backend process of the session behind `conn`.
    async fn backend_pid(conn: &mut sqlx::PgConnection) -> Result<i32> {
        Ok(sqlx::query_scalar!(r#"SELECT pg_backend_pid() AS "pid!""#)
            .fetch_one(conn)
            .await?)
    }

    /// Block until some session waits on a lock `holder` holds. Lock waits
    /// elsewhere in the database, such as another test's, do not count.
    async fn until_blocked_by(store: &PostgresDurableStore, holder: i32) -> Result<()> {
        until_sessions_blocked_by(store, holder, 1).await
    }

    /// Block until `waiters` sessions wait on locks `holder` holds, directly
    /// or queued behind another session that does: a second waiter on a row
    /// queues behind the first, not behind the row's holder.
    async fn until_sessions_blocked_by(
        store: &PostgresDurableStore,
        holder: i32,
        waiters: i64,
    ) -> Result<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let blocked = sqlx::query_scalar!(
                r#"
WITH RECURSIVE blocked (pid) AS (
    SELECT pid FROM pg_stat_activity WHERE $1 = ANY (pg_blocking_pids(pid))
    UNION
    SELECT waiting.pid
    FROM pg_stat_activity AS waiting
    JOIN blocked ON blocked.pid = ANY (pg_blocking_pids(waiting.pid))
)
SELECT count(*) AS "blocked!" FROM blocked
"#,
                holder,
            )
            .fetch_one(store.pool())
            .await?;
            if blocked >= waiters {
                return Ok(());
            }
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "{waiters} sessions never waited on the holder's locks",
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    /// Complete a running root while a submission injects into it, round
    /// after round. Every round must commit both sides — no deadlock — and
    /// leave no input under the ended root: the input either reached the
    /// root before it ended and was handed over, or found it ended and
    /// became a turn of its own.
    async fn race_completion_against_input(pending_input: bool) -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let mut deadlocks = 0_u32;
        let mut other_errors = Vec::new();
        let mut stranded = Vec::new();
        for round in 0..RACE_ROUNDS {
            let tid = format!("pg-race-{pending_input}-{round}");
            let at = round * 100;
            let (root, worker, lease) = super::running_root(&store, &tid, at).await?;
            if pending_input {
                super::inject(&store, &tid, "sent before the race", at + 2).await?;
            }
            let completing = {
                let store = store.clone();
                let root = root.id.clone();
                tokio::spawn(async move {
                    store
                        .complete_task(&root, &worker, &lease, t_plus(at + 5))
                        .await
                        .map(|_| ())
                })
            };
            let submitting = {
                let store = store.clone();
                let params = injection_at(&tid, "sent during the race", at + 4);
                tokio::spawn(async move {
                    store
                        .submit_root_turn_idempotent(params)
                        .await
                        .map(|_| ())
                        .map_err(|error| anyhow::anyhow!("{error}"))
                })
            };
            for outcome in [completing.await?, submitting.await?] {
                if let Err(error) = outcome {
                    let message = format!("{error:#}");
                    if message.contains("deadlock") {
                        deadlocks += 1;
                    } else {
                        other_errors.push(format!("round {round}: {message}"));
                    }
                }
            }
            stranded.extend(stranded_rows(&store, &tid).await?);
        }
        assert!(
            deadlocks == 0 && other_errors.is_empty() && stranded.is_empty(),
            "over {RACE_ROUNDS} rounds: {deadlocks} deadlocked, {} stranded input, other \
             errors {other_errors:#?}",
            stranded.len(),
        );
        Ok(())
    }

    const CANCEL_RACE_ROUNDS: i64 = 100;

    /// Stop a running root while `racing` adds to its tree, round after
    /// round. However the two interleave, the stop must leave nothing live
    /// under the cancelled root: whatever reached the tree first is
    /// cancelled with it.
    async fn race_cancel_against<F>(name: &str, racing: F) -> Result<()>
    where
        F: AsyncFn(PostgresDurableStore, String, AgentTaskId, WorkerId, LeaseId, i64) -> Result<()>,
    {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let mut failures = Vec::new();
        let mut stranded = Vec::new();
        for round in 0..CANCEL_RACE_ROUNDS {
            let tid = format!("pg-cancel-{name}-{round}");
            let at = round * 100;
            let (root, worker, lease) = super::running_root(&store, &tid, at).await?;
            let cancelling = {
                let store = store.clone();
                let root = root.id.clone();
                tokio::spawn(async move { store.cancel_tree(&root, t_plus(at + 5)).await })
            };
            let adding = racing(
                store.clone(),
                tid.clone(),
                root.id.clone(),
                worker,
                lease,
                at,
            );
            let added = adding.await;
            if let Err(error) = cancelling.await? {
                failures.push(format!("round {round}: cancel: {error:#}"));
            }
            if let Err(error) = added {
                failures.push(format!("round {round}: {name}: {error:#}"));
            }
            stranded.extend(stranded_rows(&store, &tid).await?);
        }
        assert!(
            failures.is_empty() && stranded.is_empty(),
            "over {CANCEL_RACE_ROUNDS} rounds: {} rows stranded under a cancelled root, \
             failures {failures:#?}",
            stranded.len(),
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_racing_input_strands_nothing() -> Result<()> {
        race_cancel_against("inject", async |store, tid, _root, _worker, _lease, at| {
            store
                .submit_root_turn_idempotent(injection_at(&tid, "sent while stopping", at + 4))
                .await
                .map(|_| ())
                .map_err(|error| anyhow::anyhow!("{error}"))
        })
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_racing_a_tool_spawn_strands_nothing() -> Result<()> {
        race_cancel_against("spawn", async |store, tid, root, worker, lease, at| {
            let spawned = store
                .spawn_tool_children(
                    &root,
                    &worker,
                    &lease,
                    vec![ChildSpawnSpec::new(3); 2],
                    super::super::suspension_payload(&tid),
                    None,
                    Vec::new(),
                    t_plus(at + 4),
                )
                .await;
            match spawned {
                Ok(_) => Ok(()),
                Err(error) if format!("{error:#}").contains("deadlock") => Err(error),
                Err(_rejected_after_the_stop) => Ok(()),
            }
        })
        .await
    }

    /// A suspension waiting on `count` tool calls.
    fn payload_with_calls(tid: &str, count: usize) -> SuspensionPayload {
        let mut payload = super::super::suspension_payload(tid);
        payload.continuation.payload.pending_tool_calls = (0..count)
            .map(|index| PendingToolCallInfo {
                id: format!("call-{index}"),
                name: "read_file".into(),
                display_name: "Read File".into(),
                tier: ToolTier::Observe,
                concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
                input: serde_json::json!({ "path": format!("f{index}") }),
                effective_input: serde_json::json!({ "path": format!("f{index}") }),
                listen_context: None,
            })
            .collect();
        payload
    }

    /// Re-park a root, woken early over two tool children, while the first
    /// child completes, round after round. Neither side may deadlock, and
    /// the re-parked root must count exactly the children still live.
    async fn race_repark_against_child_completion<F>(name: &str, repark: F) -> Result<()>
    where
        F: AsyncFn(
            PostgresDurableStore,
            String,
            AgentTaskId,
            Vec<AgentTaskId>,
            WorkerId,
            LeaseId,
            i64,
        ) -> Result<()>,
    {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let mut failures = Vec::new();
        let mut miscounted = Vec::new();
        for round in 0..RACE_ROUNDS {
            let tid = format!("pg-repark-{name}-{round}");
            let at = round * 100;
            let parked = super::park_root_on_tool_children(&store, &tid, 2, 3, at).await?;
            let children: Vec<AgentTaskId> = parked
                .children
                .iter()
                .map(|child| child.id.clone())
                .collect();
            let child_worker = WorkerId::new();
            let child_lease = LeaseId::new();
            store
                .try_acquire_task(
                    &children[0],
                    child_worker.clone(),
                    child_lease.clone(),
                    t_plus(at + 10_000),
                    t_plus(at + 3),
                )
                .await?
                .context("child must acquire")?;
            let (worker, lease) = super::resume_root_early(&store, &parked.root.id, at + 5).await?;

            let completing = {
                let store = store.clone();
                let child = children[0].clone();
                tokio::spawn(async move {
                    store
                        .complete_task(&child, &child_worker, &child_lease, t_plus(at + 9))
                        .await
                        .map(|_| ())
                })
            };
            let reparked = repark(
                store.clone(),
                tid.clone(),
                parked.root.id.clone(),
                children,
                worker,
                lease,
                at,
            )
            .await;
            for (side, outcome) in [("child completion", completing.await?), (name, reparked)] {
                if let Err(error) = outcome {
                    failures.push(format!("round {round}: {side}: {error:#}"));
                }
            }
            let root = super::fetch(&store, &parked.root.id).await?;
            let live = store
                .list_children(&parked.root.id)
                .await?
                .iter()
                .filter(|child| !child.status.is_terminal())
                .count();
            if u32::try_from(live)? != root.pending_child_count {
                miscounted.push(format!(
                    "round {round}: {live} live children, root counts {}",
                    root.pending_child_count,
                ));
            }
        }
        assert!(
            failures.is_empty() && miscounted.is_empty(),
            "over {RACE_ROUNDS} rounds: failures {failures:#?}, miscounted {miscounted:#?}",
        );
        Ok(())
    }

    /// A root woken early over two tool children from two spawn batches:
    /// `first` was spawned before `second`, and its id sorts after it, so
    /// spawn order and id order disagree. `None` when the ids happen to
    /// agree with spawn order.
    async fn woken_over_children_ordered_against_their_ids(
        store: &PostgresDurableStore,
        tid: &str,
        at: i64,
    ) -> Result<Option<(AgentTaskId, AgentTaskId, AgentTaskId, WorkerId, LeaseId)>> {
        let parked = super::park_root_on_tool_children(store, tid, 1, 3, at).await?;
        let first = parked.children[0].id.clone();
        let (worker, lease) = super::resume_root_early(store, &parked.root.id, at + 5).await?;
        let spawned = store
            .spawn_mixed_children(
                &parked.root.id,
                &worker,
                &lease,
                MixedChildrenSpawn {
                    delivered_injection_ids: Vec::new(),
                    boundary_events: Vec::new(),
                    subagents: Vec::new(),
                    tool_children: vec![ToolChildSpawn {
                        spawn_index: 1,
                        spec: ChildSpawnSpec::new(3),
                    }],
                    reattach: vec![ReattachedChild {
                        child_id: first.clone(),
                        spawn_index: 0,
                    }],
                    payload: payload_with_calls(tid, 2),
                    child_otel_traceparent: None,
                },
                Vec::new(),
                t_plus(at + 20),
            )
            .await?;
        let second = spawned.tool_children[0].id.clone();
        if first < second {
            return Ok(None);
        }
        let (worker, lease) = super::resume_root_early(store, &parked.root.id, at + 30).await?;
        Ok(Some((parked.root.id, first, second, worker, lease)))
    }

    /// A steering re-park and a stop of the same root, staged so each takes
    /// its first child lock before reaching for the other's: with a single
    /// order over sibling rows they queue instead of deadlocking.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_steering_repark_racing_a_cancel_never_deadlocks() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let mut staged = 0;
        for round in 0..64 {
            if staged == 5 {
                break;
            }
            let tid = format!("pg-repark-cancel-{round}");
            let at = round * 100;
            let Some((root, first, second, worker, lease)) =
                woken_over_children_ordered_against_their_ids(&store, &tid, at).await?
            else {
                continue;
            };
            staged += 1;

            let mut holder = store.pool().begin().await?;
            let holder_pid = backend_pid(&mut holder).await?;
            sqlx::query_scalar!(
                "SELECT id FROM agent_sdk_tasks WHERE id = $1 FOR UPDATE",
                second.as_str(),
            )
            .fetch_one(&mut *holder)
            .await?;
            let reparking = {
                let store = store.clone();
                let root = root.clone();
                let children = vec![first.clone(), second.clone()];
                let payload = payload_with_calls(&tid, 2);
                tokio::spawn(async move {
                    store
                        .repark_after_steering(
                            &root,
                            &worker,
                            &lease,
                            payload,
                            children,
                            t_plus(at + 40),
                        )
                        .await
                        .map(|_| ())
                })
            };
            until_sessions_blocked_by(&store, holder_pid, 1).await?;
            let cancelling = {
                let store = store.clone();
                let root = root.clone();
                tokio::spawn(async move { store.cancel_tree(&root, t_plus(at + 41)).await })
            };
            until_sessions_blocked_by(&store, holder_pid, 2).await?;
            holder.commit().await?;

            let reparked = reparking.await?;
            let cancelled = cancelling.await?;
            anyhow::ensure!(
                reparked.is_ok() && cancelled.is_ok(),
                "round {round}: re-park {reparked:?}, cancel {:?}",
                cancelled.map(|_| ()),
            );
            anyhow::ensure!(
                stranded_rows(&store, &tid).await?.is_empty(),
                "round {round}: rows left live under the cancelled root",
            );
        }
        anyhow::ensure!(
            staged == 5,
            "only {staged} rounds drew ids against spawn order"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_steering_repark_racing_a_child_completion() -> Result<()> {
        race_repark_against_child_completion(
            "repark",
            async |store, tid, root, children, worker, lease, at| {
                store
                    .repark_after_steering(
                        &root,
                        &worker,
                        &lease,
                        payload_with_calls(&tid, 2),
                        children,
                        t_plus(at + 8),
                    )
                    .await
                    .map(|_| ())
            },
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_mixed_spawn_reattach_racing_a_child_completion() -> Result<()> {
        race_repark_against_child_completion(
            "mixed spawn",
            async |store, tid, root, children, worker, lease, at| {
                let reattach = children
                    .into_iter()
                    .zip(0_u32..)
                    .map(|(child_id, spawn_index)| ReattachedChild {
                        child_id,
                        spawn_index,
                    })
                    .collect();
                store
                    .spawn_mixed_children(
                        &root,
                        &worker,
                        &lease,
                        MixedChildrenSpawn {
                            delivered_injection_ids: Vec::new(),
                            boundary_events: Vec::new(),
                            subagents: Vec::new(),
                            tool_children: vec![ToolChildSpawn {
                                spawn_index: 2,
                                spec: ChildSpawnSpec::new(3),
                            }],
                            reattach,
                            payload: payload_with_calls(&tid, 3),
                            child_otel_traceparent: None,
                        },
                        Vec::new(),
                        t_plus(at + 8),
                    )
                    .await
                    .map(|_| ())
            },
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_completion_racing_input_over_pending_input() -> Result<()> {
        race_completion_against_input(true).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_completion_racing_the_first_input() -> Result<()> {
        race_completion_against_input(false).await
    }

    /// Run `transition` on `root` while another transaction holds the
    /// root's thread, then have that holder reach for the root itself, as a
    /// submission or a cancel does. With thread before root on both sides the
    /// transition queues behind the holder and both finish; locking the root
    /// first deadlocks them.
    async fn behind_a_thread_holder<T: Send + 'static>(
        store: &PostgresDurableStore,
        tid: &str,
        root: &AgentTaskId,
        transition: impl std::future::Future<Output = Result<T>> + Send + 'static,
    ) -> Result<T> {
        let mut holder = store.pool().begin().await?;
        let holder_pid = backend_pid(&mut holder).await?;
        sqlx::query_scalar!(
            "SELECT thread_id FROM agent_sdk_threads WHERE thread_id = $1 FOR UPDATE",
            tid,
        )
        .fetch_one(&mut *holder)
        .await?;
        let transition = tokio::spawn(transition);
        until_blocked_by(store, holder_pid).await?;
        sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 FOR UPDATE",
            root.as_str(),
        )
        .fetch_one(&mut *holder)
        .await
        .context("the thread's holder must still reach the root")?;
        holder.commit().await?;
        transition
            .await?
            .context("the transition must finish behind the thread's holder")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_completion_waits_behind_a_thread_holder() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let tid = "pg-thread-holder";
        let (root, worker, lease) = super::running_root(&store, tid, 0).await?;
        let pending = super::inject(&store, tid, "not yet delivered", 2).await?;

        let completing = {
            let store = store.clone();
            let root = root.id.clone();
            async move { store.complete_task(&root, &worker, &lease, t_plus(5)).await }
        };
        behind_a_thread_holder(&store, tid, &root.id, completing).await?;

        assert_eq!(
            super::fetch(&store, &root.id).await?.status,
            TaskStatus::Completed
        );
        assert_eq!(
            super::fetch(&store, &pending.id).await?.status,
            TaskStatus::Completed,
            "the pending input is handed over",
        );
        Ok(())
    }

    /// A root pausing on a question journals the question on its thread, so
    /// it too locks the thread before itself.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_question_pause_waits_behind_a_thread_holder() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let tid = "pg-question-holder";
        let (root, worker, lease) = super::running_root(&store, tid, 0).await?;

        let pausing = {
            let store = store.clone();
            let root = root.id.clone();
            async move {
                store
                    .pause_on_question(
                        &root,
                        &worker,
                        &lease,
                        super::super::question_pause(tid, 1),
                        t_plus(5),
                    )
                    .await
            }
        };
        behind_a_thread_holder(&store, tid, &root.id, pausing).await?;

        assert_eq!(
            super::fetch(&store, &root.id).await?.status,
            TaskStatus::AwaitingQuestion,
        );
        Ok(())
    }

    /// A root spawning a subagent journals the spawn on its own thread, so
    /// the spawn locks that thread before the root as well.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_subagent_spawn_waits_behind_a_thread_holder() -> Result<()> {
        use agent_server::journal::SubagentInvocationSpawn;
        use agent_server::journal::subagent_spawn_transaction::{
            AtomicSubagentSpawnCommitter, SubagentSpawnCommit, SubagentSpawnEvent,
        };

        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let tid = "pg-spawn-holder";
        let child_thread = thread_id("pg-spawn-holder-child");
        agent_server::journal::thread_store::ThreadStore::get_or_create(
            &store,
            &child_thread,
            t_plus(0),
        )
        .await?;
        let (root, worker, lease) = super::running_root(&store, tid, 0).await?;

        let spawning = {
            let store = store.clone();
            let root = root.id.clone();
            async move {
                store
                    .commit_subagent_spawn_atomic(SubagentSpawnCommit {
                        delivered_injection_ids: Vec::new(),
                        parent_id: root,
                        worker,
                        lease,
                        spawn: SubagentInvocationSpawn {
                            child_thread_id: child_thread,
                            spec: super::super::sample_subagent_spec(),
                            child_root_input: super::super::sample_subagent_input(),
                            spawn_index: 0,
                            child_caller_metadata: None,
                            payload: super::super::suspension_payload(tid),
                        },
                        event: SubagentSpawnEvent {
                            subagent_id: "call-1".into(),
                            subagent_name: "explore".into(),
                        },
                        outbox_max_attempts: 3,
                        now: t_plus(5),
                    })
                    .await
                    .map(|_| ())
            }
        };
        behind_a_thread_holder(&store, tid, &root.id, spawning).await?;

        assert_eq!(
            super::fetch(&store, &root.id).await?.status,
            TaskStatus::WaitingOnChildren,
        );
        Ok(())
    }

    /// Input submitted while its root is ending under a lock becomes a turn
    /// of its own once the root has ended, instead of waiting on a root no
    /// boundary will ever reach again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_input_never_lands_under_a_root_that_ended_meanwhile() -> Result<()>
    {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let tid = "pg-ended-meanwhile";
        let (root, _worker, _lease) = super::running_root(&store, tid, 0).await?;

        let mut ending = store.pool().begin().await?;
        let ending_pid = backend_pid(&mut ending).await?;
        sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 FOR UPDATE",
            root.id.as_str(),
        )
        .fetch_one(&mut *ending)
        .await?;
        let submitting = {
            let store = store.clone();
            let params = injection_at(tid, "sent while the root ends", 3);
            tokio::spawn(async move {
                store
                    .submit_root_turn_idempotent(params)
                    .await
                    .map_err(|error| anyhow::anyhow!("{error}"))
            })
        };
        until_blocked_by(&store, ending_pid).await?;
        let ended_at = OffsetDateTime::now_utc();
        sqlx::query!(
            r"
UPDATE agent_sdk_tasks
SET status = 'completed',
    worker_id = NULL,
    lease_id = NULL,
    lease_expires_at = NULL,
    last_heartbeat_at = NULL,
    terminal_reason_json = $2,
    completed_at = $3,
    updated_at = $3,
    last_activity_at = $3
WHERE id = $1
",
            root.id.as_str(),
            serde_json::to_value(TerminalReason::Completed)?,
            ended_at,
        )
        .execute(&mut *ending)
        .await?;
        ending.commit().await?;

        let admitted = submitting.await??.task;
        assert_eq!(
            admitted.kind,
            TaskKind::RootTurn,
            "input whose root ended meanwhile must become a turn, not wait under it",
        );
        assert_eq!(stranded_rows(&store, tid).await?, Vec::<AgentTaskId>::new());
        Ok(())
    }

    /// A child-thread root's undelivered input dies with it even while
    /// another transaction holds the subagent invocation's row, which the
    /// wake skips: the link is read, not locked.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_child_thread_input_dies_while_its_invocation_is_locked()
    -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(&store, &store, "pg-locked-invocation").await?;
        let child_tid = child_root.thread_id.0.clone();
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        store
            .try_acquire_task(
                &child_root.id,
                worker.clone(),
                lease.clone(),
                t_plus(10_000),
                t_plus(3),
            )
            .await?
            .context("child root must acquire")?;
        let pending = super::inject(&store, &child_tid, "look at the logs too", 4).await?;

        let mut holder = store.pool().begin().await?;
        sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 FOR UPDATE",
            invocation.id.as_str(),
        )
        .fetch_one(&mut *holder)
        .await?;
        store
            .complete_task(&child_root.id, &worker, &lease, t_plus(10))
            .await?;
        holder.rollback().await?;

        let turns = super::root_turns_of(&store, &child_tid).await?;
        assert_eq!(
            turns.iter().map(|turn| &turn.id).collect::<Vec<_>>(),
            vec![&child_root.id],
            "no turn may start on the subagent's thread",
        );
        super::assert_settled(
            &super::fetch(&store, &pending.id).await?,
            &TerminalReason::ParentCancelled,
            "undelivered input of the child root",
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_stale_wait_lists_from_an_older_build_still_load() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let parked = super::park_root_on_tool_children(&store, "pg-stale-waits", 3, 3, 0).await?;
        let waits = serde_json::json!([parked.children[0].id.as_str()]);
        let at = t_plus(5);
        sqlx::query(
            "UPDATE agent_sdk_tasks SET status = 'running', worker_id = 'older-build', \
             lease_id = 'older-lease', lease_expires_at = $2, last_heartbeat_at = $2, \
             attempt = 1, after_siblings_json = $3 WHERE id = $1",
        )
        .bind(parked.children[1].id.as_str())
        .bind(at)
        .bind(&waits)
        .execute(store.pool())
        .await?;
        sqlx::query(
            "UPDATE agent_sdk_tasks SET status = 'completed', completed_at = $2, \
             terminal_reason_json = $3, after_siblings_json = $4 WHERE id = $1",
        )
        .bind(parked.children[2].id.as_str())
        .bind(at)
        .bind(serde_json::to_value(TerminalReason::Completed)?)
        .bind(&waits)
        .execute(store.pool())
        .await?;
        super::test_stale_wait_lists_still_load(&store, &parked).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_a_settled_child_hints_the_sibling_it_released() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_a_settled_child_hints_the_sibling_it_released(&store, "pg-hint").await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_requeued_wait_list_from_an_older_build_runs_in_order() -> Result<()>
    {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let parked =
            super::park_root_on_tool_children(&store, "pg-requeued-waits", 3, 3, 0).await?;
        sqlx::query(
            "UPDATE agent_sdk_tasks SET attempt = 1, after_siblings_json = $2 WHERE id = $1",
        )
        .bind(parked.children[1].id.as_str())
        .bind(serde_json::json!([parked.children[0].id.as_str()]))
        .execute(store.pool())
        .await?;
        super::test_requeued_wait_list_is_acquired_once_its_siblings_finish(&store, &parked).await
    }

    /// A child's completion and a cancel of its tree, staged so the cancel
    /// holds the first child's row while the completion holds the last
    /// child's. Neither may take a lock out of the store's child-before-
    /// parent order: the completion commits, and the cancel settles the rest.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_child_completion_racing_a_cancel_never_deadlocks() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let tid = "pg-completion-vs-cancel";
        let parked = super::park_root_on_tool_children(&store, tid, 3, 3, 0).await?;
        let mut by_id = parked.children.clone();
        by_id.sort_by(|a, b| a.id.cmp(&b.id));
        let [first, middle, last] = by_id.as_slice() else {
            anyhow::bail!("three children must be parked");
        };
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        store
            .try_acquire_task(
                &last.id,
                worker.clone(),
                lease.clone(),
                t_plus(10_000),
                t_plus(5),
            )
            .await?
            .context("the last child must acquire")?;

        let mut holder = store.pool().begin().await?;
        let holder_pid = backend_pid(&mut holder).await?;
        sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 FOR UPDATE",
            middle.id.as_str(),
        )
        .fetch_one(&mut *holder)
        .await?;
        let cancelling = {
            let store = store.clone();
            let root = parked.root.id.clone();
            tokio::spawn(async move { store.cancel_tree(&root, t_plus(6)).await })
        };
        until_sessions_blocked_by(&store, holder_pid, 1).await?;

        let completing = {
            let store = store.clone();
            let child = last.id.clone();
            tokio::spawn(async move {
                store
                    .complete_task_with_result(
                        &child,
                        &worker,
                        &lease,
                        serde_json::json!({"ok": true}),
                        t_plus(7),
                    )
                    .await
                    .map(|_| ())
            })
        };
        let completed = tokio::time::timeout(std::time::Duration::from_secs(10), completing).await;
        holder.commit().await?;
        let cancelled = cancelling.await?;
        let completed = completed.context(
            "the completion waited on the cancel, which holds a parent lock out of order",
        )??;
        anyhow::ensure!(
            completed.is_ok() && cancelled.is_ok(),
            "completion {completed:?}, cancel {:?}",
            cancelled.map(|_| ()),
        );

        let status = |id: &AgentTaskId| {
            let store = store.clone();
            let id = id.clone();
            async move {
                Ok::<_, anyhow::Error>(
                    store
                        .get(&id)
                        .await?
                        .context("child row must exist")?
                        .status,
                )
            }
        };
        anyhow::ensure!(
            status(&last.id).await? == TaskStatus::Completed,
            "the completion's terminal write must land"
        );
        for child in [first, middle] {
            anyhow::ensure!(status(&child.id).await? == TaskStatus::Cancelled);
        }
        Ok(())
    }

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_approved_confirmation_keeps_the_paused_attempt() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_approved_confirmation_pickup_keeps_the_paused_attempt(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_lists_terminal_roots_with_live_descendants() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_list_terminal_roots_with_live_descendants(&store, &store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_root_end_hands_input_over_in_order() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_root_end_hands_undelivered_input_over_in_submission_order(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_child_thread_root_end_discards_input() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_child_thread_root_end_discards_undelivered_input(&store, &store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_descendants_spares_the_root() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_descendants_spares_the_root_and_its_input(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_cancel_descendants_closes_child_threads() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_cancel_descendants_closes_child_threads(&store, &store, &store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_fail_close_settles_children_first() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_fail_close_settles_children_before_the_successor(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_fail_close_closes_child_threads() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_fail_close_closes_child_threads(&store, &store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_reject_confirmation_refuses_a_root() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_reject_confirmation_refuses_a_root(&store).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn settlement_postgres_root_end_on_a_purging_thread_discards_input() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        super::test_root_end_on_a_purging_thread_discards_undelivered_input(&store, &store).await
    }
}
