//! A root turn that ends while descendants are still live settles them:
//! nothing is left waiting on a fan-in that can never run again. Input the
//! turn never delivered becomes the thread's next turn, in submission order,
//! unless the turn was a subagent's, whose input dies with it.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::journal::SpawnChildrenExt;
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::{
    AgentContinuation, AgentEvent, AgentState, ContinuationEnvelope, TerminalReason, ThreadId,
    TokenUsage, llm,
};
use anyhow::{Context, Result};
use time::{Duration, OffsetDateTime};

use super::definition::ChildJoinPolicy;
use super::root_turn::{RootTurnDeps, fail_root_turn, settle_live_descendants};
use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentPolicy, SubagentSandboxPolicy,
};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::outbox::InMemoryOutboxStore;
use crate::journal::store::{
    AgentTaskStore, CancellationMarkerSink, InMemoryAgentTaskStore,
    InMemoryAtomicSpawnEventCommitter, SubagentInvocationSpawn, SubmitDisposition,
    SubmitRootTurnParams,
};
use crate::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskKind, TaskStatus, WorkerId,
};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;

fn t_plus(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000 + secs)
}

struct Stores {
    tasks: InMemoryAgentTaskStore,
    threads: Arc<InMemoryThreadStore>,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: Arc<InMemoryEventRepository>,
    notifier: EventNotifier,
}

impl Stores {
    fn new() -> Self {
        let threads = Arc::new(InMemoryThreadStore::new());
        let events = Arc::new(InMemoryEventRepository::new());
        let outbox = Arc::new(InMemoryOutboxStore::new());
        let tasks = InMemoryAgentTaskStore::new()
            .with_cancellation_markers(CancellationMarkerSink {
                event_repo: events.clone(),
                outbox_store: outbox.clone(),
                thread_store: threads.clone(),
            })
            .with_spawn_event_committer(Arc::new(InMemoryAtomicSpawnEventCommitter::new(
                events.as_ref().clone(),
                outbox.as_ref().clone(),
            )));
        Self {
            tasks,
            threads,
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
            events,
            notifier: EventNotifier::new(),
        }
    }

    fn deps(&self) -> RootTurnDeps<'_> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: self.threads.as_ref(),
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
            event_repo: self.events.as_ref(),
            event_notifier: &self.notifier,
            subagent_spawn_selector: None,
            compaction_config: None,
            compaction_provider: None,
            compaction_artifact_store: None,
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
        }
    }
}

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

fn spec() -> EffectiveSubagentSpec {
    let names = |items: &[&str]| -> BTreeSet<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    };
    EffectiveSubagentSpec {
        task: "Inspect".into(),
        prompt: "Stay read-only.".into(),
        model: "mock-model".into(),
        max_turns: 5,
        timeout_ms: 15_000,
        depth: 1,
        max_parallel_subagents: 1,
        nickname: None,
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy {
            allowed_servers: BTreeSet::new(),
        },
        audit_provenance: Some(AuditProvenance::new("mock", "mock-model")),
        inherited_policy: InheritedSubagentPolicy {
            default_model: "mock-model".into(),
            allowed_models: names(&["mock-model"]),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: 15_000,
            max_timeout_ms: 15_000,
            capability_profiles: std::collections::BTreeMap::new(),
            allowed_capabilities: names(&["read_file"]),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: BTreeSet::new(),
            audit_provider: "mock".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: names(&["read_file"]),
        },
    }
}

async fn acquire(
    stores: &Stores,
    id: &AgentTaskId,
    at: i64,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    let task = stores
        .tasks
        .try_acquire_task(
            id,
            worker.clone(),
            lease.clone(),
            t_plus(at + 10_000),
            t_plus(at),
        )
        .await?
        .with_context(|| format!("{id} must acquire"))?;
    Ok((task, worker, lease))
}

async fn running_root(
    stores: &Stores,
    thread: &ThreadId,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let root = AgentTask::new_root_turn(thread.clone(), t_plus(0), 3);
    stores.tasks.submit_root_turn(root.clone()).await?;
    acquire(stores, &root.id, 1).await
}

/// Early resume (a steering note) leases the parked root while its
/// children are still live — the shape a failing resume leaves behind.
async fn resume_early(stores: &Stores, root: &AgentTaskId, at: i64) -> Result<(WorkerId, LeaseId)> {
    let note = vec![llm::ContentBlock::Text {
        text: "status?".into(),
    }];
    stores
        .tasks
        .enqueue_steering_resume(root, note, t_plus(at))
        .await?
        .context("parked root must wake")?;
    let (_, worker, lease) = acquire(stores, root, at + 1).await?;
    Ok((worker, lease))
}

async fn status_of(stores: &Stores, id: &AgentTaskId) -> Result<AgentTask> {
    stores
        .tasks
        .get(id)
        .await?
        .with_context(|| format!("{id} exists"))
}

fn assert_parent_cancelled(task: &AgentTask, label: &str) {
    assert_eq!(task.status, TaskStatus::Cancelled, "{label}");
    assert_eq!(
        task.terminal_reason,
        Some(TerminalReason::ParentCancelled),
        "{label}",
    );
    assert!(matches!(task.state, TaskState::None), "{label}");
}

#[tokio::test]
async fn failed_root_turn_cancels_live_tool_children() -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("settle-failed-root");
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let (_, children) = stores
        .tasks
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3); 3],
            payload(&thread),
            None,
            Vec::new(),
            t_plus(2),
        )
        .await?;

    let (_, confirm_worker, confirm_lease) = acquire(&stores, &children[0].id, 3).await?;
    stores
        .tasks
        .pause_on_confirmation(
            &children[0].id,
            &confirm_worker,
            &confirm_lease,
            payload(&thread).continuation,
            None,
            t_plus(4),
        )
        .await?;
    acquire(&stores, &children[1].id, 5).await?;

    let (resume_worker, resume_lease) = resume_early(&stores, &root.id, 6).await?;
    let failed = fail_root_turn(
        &root.id,
        &resume_worker,
        &resume_lease,
        &thread,
        &anyhow::anyhow!("build steering resume chat request: refused"),
        &stores.deps(),
        t_plus(10),
    )
    .await?;
    assert_eq!(failed.status, TaskStatus::Failed);

    for (child, label) in children
        .iter()
        .zip(["awaiting confirmation", "running", "pending"])
    {
        assert_parent_cancelled(&status_of(&stores, &child.id).await?, label);
    }
    let root_after = status_of(&stores, &root.id).await?;
    assert_eq!(
        root_after.status,
        TaskStatus::Failed,
        "the root keeps its outcome"
    );
    assert_eq!(
        root_after.terminal_reason,
        Some(TerminalReason::InternalError)
    );
    Ok(())
}

#[tokio::test]
async fn failed_root_turn_closes_a_live_child_thread_for_its_followers() -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("settle-failed-parent");
    let child_thread = ThreadId::from_string("settle-failed-parent-child");
    stores
        .threads
        .get_or_create(&child_thread, t_plus(0))
        .await?;
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let (_, invocation, child_root) = stores
        .tasks
        .spawn_subagent_invocation(
            &root.id,
            &worker,
            &lease,
            SubagentInvocationSpawn {
                child_thread_id: child_thread.clone(),
                spec: spec(),
                child_root_input: vec![SubmittedInputItem::Text {
                    text: "Inspect".into(),
                }],
                spawn_index: 0,
                child_caller_metadata: None,
                payload: payload(&thread),
            },
            Vec::new(),
            t_plus(2),
        )
        .await?;
    acquire(&stores, &child_root.id, 3).await?;

    let mut follower = stores.notifier.subscribe(&child_thread);
    let (resume_worker, resume_lease) = resume_early(&stores, &root.id, 4).await?;
    fail_root_turn(
        &root.id,
        &resume_worker,
        &resume_lease,
        &thread,
        &anyhow::anyhow!("resume failed"),
        &stores.deps(),
        t_plus(10),
    )
    .await?;

    assert_parent_cancelled(&status_of(&stores, &invocation.id).await?, "invocation");
    assert_parent_cancelled(&status_of(&stores, &child_root.id).await?, "child root");
    let closing = tokio::time::timeout(std::time::Duration::from_secs(5), follower.recv())
        .await
        .context("the child thread's live follower must be woken")??;
    assert!(
        matches!(closing.event, AgentEvent::Cancelled { .. }),
        "the follower must receive the closing marker, got {:?}",
        closing.event,
    );
    Ok(())
}

/// Submit `text` while the thread's root is running, so it waits as an
/// injection for that turn's next boundary.
async fn inject(stores: &Stores, thread: &ThreadId, text: &str, at: i64) -> Result<AgentTask> {
    let outcome = stores
        .tasks
        .submit_root_turn_idempotent(SubmitRootTurnParams {
            task: AgentTask::new_root_turn_with_input_and_caller(
                thread.clone(),
                vec![SubmittedInputItem::Text { text: text.into() }],
                serde_json::json!({ "client": "tui" }),
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

async fn root_turns(stores: &Stores, thread: &ThreadId) -> Result<Vec<AgentTask>> {
    Ok(stores
        .tasks
        .list_by_thread(thread)
        .await?
        .into_iter()
        .filter(|task| task.kind == TaskKind::RootTurn)
        .collect())
}

/// The injection was handed over, not dropped, and the thread's next turn
/// carries it: same input, same caller metadata, its place in the queue.
async fn assert_readmitted(
    stores: &Stores,
    thread: &ThreadId,
    ended_root: &AgentTaskId,
    injection: &AgentTask,
) -> Result<()> {
    let handed_over = status_of(stores, &injection.id).await?;
    assert_eq!(handed_over.status, TaskStatus::Completed);
    assert_eq!(handed_over.terminal_reason, Some(TerminalReason::Completed));

    let next_turns: Vec<AgentTask> = root_turns(stores, thread)
        .await?
        .into_iter()
        .filter(|root| root.id != *ended_root)
        .collect();
    assert_eq!(next_turns.len(), 1, "exactly one next turn: {next_turns:?}");
    let next = &next_turns[0];
    assert_eq!(next.status, TaskStatus::Pending);
    assert_eq!(
        serde_json::to_value(&next.submitted_input)?,
        serde_json::to_value(&injection.submitted_input)?,
    );
    assert_eq!(next.caller_metadata, injection.caller_metadata);
    assert_eq!(next.created_at, injection.created_at);
    Ok(())
}

#[tokio::test]
async fn completing_a_root_readmits_its_undelivered_injection() -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("settle-late-injection");
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let injection = inject(&stores, &thread, "also check the tests", 2).await?;
    stores
        .tasks
        .complete_task(&root.id, &worker, &lease, t_plus(3))
        .await?;

    assert_readmitted(&stores, &thread, &root.id, &injection).await?;
    assert_eq!(
        status_of(&stores, &root.id).await?.status,
        TaskStatus::Completed,
        "the ended root keeps its outcome",
    );

    let settled = settle_live_descendants(&root.id, &stores.deps(), t_plus(10)).await?;
    assert!(settled.is_empty(), "nothing is left to settle: {settled:?}");
    assert_eq!(root_turns(&stores, &thread).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn failed_root_turn_readmits_its_undelivered_injection() -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("fail-late-injection");
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let injection = inject(&stores, &thread, "also check the tests", 2).await?;

    fail_root_turn(
        &root.id,
        &worker,
        &lease,
        &thread,
        &anyhow::anyhow!("provider refused"),
        &stores.deps(),
        t_plus(10),
    )
    .await?;

    assert_readmitted(&stores, &thread, &root.id, &injection).await
}

/// Input injected into the failing turn runs before input submitted after
/// it as the next turn, even when that later submission was already queued
/// behind the turn when it failed.
#[tokio::test]
async fn failed_root_turn_readmits_its_input_ahead_of_a_later_submission() -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("fail-injection-before-successor");
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let injection = inject(&stores, &thread, "also check the tests", 2).await?;
    let successor = stores
        .tasks
        .submit_root_turn(AgentTask::new_root_turn(thread.clone(), t_plus(3), 3))
        .await?;
    assert_eq!(successor.status, TaskStatus::Queued);

    fail_root_turn(
        &root.id,
        &worker,
        &lease,
        &thread,
        &anyhow::anyhow!("provider refused"),
        &stores.deps(),
        t_plus(10),
    )
    .await?;

    let turns = root_turns(&stores, &thread).await?;
    let readmitted = turns
        .iter()
        .find(|turn| turn.id != root.id && turn.id != successor.id)
        .context("the injected input must become a turn")?;
    assert_eq!(readmitted.created_at, injection.created_at);
    assert_eq!(
        readmitted.status,
        TaskStatus::Pending,
        "the earlier input takes the freed slot",
    );
    assert_eq!(
        status_of(&stores, &successor.id).await?.status,
        TaskStatus::Queued,
        "the later submission waits behind it",
    );
    Ok(())
}

/// A subagent's thread whose root ends with input still undelivered: the
/// subagent's invocation takes the outcome, and the input dies with the
/// root instead of starting a turn nobody awaits beside the parent.
async fn assert_child_thread_input_dies_with_its_root(
    end: impl AsyncFnOnce(&Stores, &AgentTask, WorkerId, LeaseId) -> Result<()>,
) -> Result<()> {
    let stores = Stores::new();
    let thread = ThreadId::from_string("child-root-injection-parent");
    let child_thread = ThreadId::from_string("child-root-injection-child");
    stores
        .threads
        .get_or_create(&child_thread, t_plus(0))
        .await?;
    let (root, worker, lease) = running_root(&stores, &thread).await?;
    let (_, invocation, child_root) = stores
        .tasks
        .spawn_subagent_invocation(
            &root.id,
            &worker,
            &lease,
            SubagentInvocationSpawn {
                child_thread_id: child_thread.clone(),
                spec: spec(),
                child_root_input: vec![SubmittedInputItem::Text {
                    text: "Inspect".into(),
                }],
                spawn_index: 0,
                child_caller_metadata: None,
                payload: payload(&thread),
            },
            Vec::new(),
            t_plus(2),
        )
        .await?;
    let (_, child_worker, child_lease) = acquire(&stores, &child_root.id, 3).await?;
    let injection = inject(&stores, &child_thread, "look at the logs too", 4).await?;
    assert_eq!(injection.parent_id.as_ref(), Some(&child_root.id));

    end(&stores, &child_root, child_worker, child_lease).await?;
    let settled = settle_live_descendants(&child_root.id, &stores.deps(), t_plus(20)).await?;

    let child_turns = root_turns(&stores, &child_thread).await?;
    assert_eq!(
        child_turns.iter().map(|turn| &turn.id).collect::<Vec<_>>(),
        vec![&child_root.id],
        "no turn may start on the subagent's thread: {child_turns:?}",
    );
    let discarded = status_of(&stores, &injection.id).await?;
    assert_eq!(discarded.status, TaskStatus::Cancelled);
    assert_eq!(
        discarded.terminal_reason,
        Some(TerminalReason::ParentCancelled)
    );
    assert_eq!(
        status_of(&stores, &invocation.id).await?.status,
        TaskStatus::Pending,
        "the invocation wakes with the child's outcome",
    );
    assert!(settled.is_empty(), "nothing is left to settle: {settled:?}");
    Ok(())
}

#[tokio::test]
async fn failed_child_thread_root_discards_its_undelivered_input() -> Result<()> {
    assert_child_thread_input_dies_with_its_root(async |stores, child_root, worker, lease| {
        fail_root_turn(
            &child_root.id,
            &worker,
            &lease,
            &child_root.thread_id,
            &anyhow::anyhow!("provider refused"),
            &stores.deps(),
            t_plus(10),
        )
        .await?;
        Ok(())
    })
    .await
}

#[tokio::test]
async fn completed_child_thread_root_discards_its_undelivered_input() -> Result<()> {
    assert_child_thread_input_dies_with_its_root(async |stores, child_root, worker, lease| {
        stores
            .tasks
            .complete_task(&child_root.id, &worker, &lease, t_plus(10))
            .await?;
        Ok(())
    })
    .await
}
