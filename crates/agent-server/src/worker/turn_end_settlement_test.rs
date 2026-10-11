//! A root turn that fails while descendants are still live settles them:
//! nothing is left waiting on a fan-in that can never run again.

use std::collections::BTreeSet;
use std::sync::Arc;

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::{
    AgentContinuation, AgentEvent, AgentState, ContinuationEnvelope, TerminalReason, ThreadId,
    TokenUsage, llm,
};
use anyhow::{Context, Result};
use time::{Duration, OffsetDateTime};

use super::definition::ChildJoinPolicy;
use super::root_turn::{RootTurnDeps, fail_root_turn};
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
    InMemoryAtomicSpawnEventCommitter, SubagentInvocationSpawn,
};
use crate::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskStatus, WorkerId,
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
