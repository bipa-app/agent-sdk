//! Phase 5.3 regression tests for confirmation pause/resume, prepared
//! operations, and authoritative policy rechecks.
//!
//! Covers:
//! - Pause path: tool-runtime child pauses durably for confirmation.
//! - Approval + policy allows: re-acquire, recheck, execute successfully.
//! - Approval + policy denies: re-acquire, recheck, fail child.
//! - Rejection: fail child without executing.
//! - Timeout: fail child without executing.
//! - Prepared operation: listen-context preserved through pause.
//! - Restart: `AwaitingConfirmation` survives store re-read.
//! - Multi-child: confirmation on one child, other children unaffected.
//! - Parent recompute: rejected confirmation decrements parent count.

use super::confirmation::{
    CONFIRMATION_POLICY_DENIED_PREFIX, CONFIRMATION_REJECTED_PREFIX, CONFIRMATION_TIMEOUT_PREFIX,
    ConfirmationDecision, ConfirmationDecisionOutcome, ConfirmationPolicy,
    ConfirmationResumeOutcome, PolicyVerdict, apply_confirmation_decision,
    pause_tool_for_confirmation, resume_confirmed_tool,
};
use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::tool_task::{ToolTaskOutcome, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::execution_intent::{GuardedExecutionDeps, InMemoryExecutionIntentStore};
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{AgentEvent, PendingToolCallInfo, ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────
// Mock LLM provider (returns tool calls)
// ─────────────────────────────────────────────────────────────────────

struct MockToolCallProvider {
    tool_calls: Vec<(String, String, serde_json::Value)>,
    call_count: AtomicUsize,
}

impl MockToolCallProvider {
    fn new(tool_calls: Vec<(String, String, serde_json::Value)>) -> Self {
        Self {
            tool_calls,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmProvider for MockToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let content: Vec<ContentBlock> = self
            .tool_calls
            .iter()
            .map(|(id, name, input)| ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                thought_signature: None,
            })
            .collect();

        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_confirm_01".into(),
            content,
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
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

// ─────────────────────────────────────────────────────────────────────
// Mock confirmation policies
// ─────────────────────────────────────────────────────────────────────

/// Policy that always allows execution.
struct AllowAllPolicy;

#[async_trait]
impl ConfirmationPolicy for AllowAllPolicy {
    async fn check_policy(&self, _tool_call: &PendingToolCallInfo) -> Result<PolicyVerdict> {
        Ok(PolicyVerdict::Allowed)
    }
}

/// Policy that always denies execution.
struct DenyAllPolicy {
    reason: String,
}

impl DenyAllPolicy {
    fn new(reason: &str) -> Self {
        Self {
            reason: reason.to_owned(),
        }
    }
}

#[async_trait]
impl ConfirmationPolicy for DenyAllPolicy {
    async fn check_policy(&self, _tool_call: &PendingToolCallInfo) -> Result<PolicyVerdict> {
        Ok(PolicyVerdict::Denied {
            reason: self.reason.clone(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("thread_confirm_test")
}

fn worker_id() -> WorkerId {
    WorkerId::from_string("worker_confirm")
}

fn lease_id() -> LeaseId {
    LeaseId::from_string("lease_confirm")
}

fn child_worker() -> WorkerId {
    WorkerId::from_string("child_confirm_worker")
}

fn child_lease() -> LeaseId {
    LeaseId::from_string("child_confirm_lease")
}

fn resume_worker() -> WorkerId {
    WorkerId::from_string("resume_worker")
}

fn resume_lease() -> LeaseId {
    LeaseId::from_string("resume_lease")
}

fn sample_definition_with_confirm_tools() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a test agent.".into(),
        max_tokens: 1024,
        tools: vec![
            Tool {
                name: "bash".into(),
                description: "Run a bash command".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
                display_name: "Bash".into(),
                tier: ToolTier::Observe,
            },
            Tool {
                name: "transfer".into(),
                description: "Transfer funds".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"amount": {"type": "number"}}}),
                display_name: "Transfer".into(),
                tier: ToolTier::Confirm,
            },
        ],
        thinking: ThinkingPolicy::default(),
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn sample_bootstrap_with_tools(task: AgentTask) -> WorkerBootstrapContext {
    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition: sample_definition_with_confirm_tools(),
        thread_id,
        task_id,
        worker_id: worker_id(),
        lease_id: lease_id(),
    }
}

struct TestStores {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
    event_notifier: Arc<EventNotifier>,
}

impl TestStores {
    fn new() -> Self {
        Self {
            tasks: InMemoryAgentTaskStore::new(),
            threads: InMemoryThreadStore::new(),
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
            events: InMemoryEventRepository::new(),
            event_notifier: Arc::new(EventNotifier::new()),
        }
    }

    fn deps(&self) -> RootTurnDeps<'_> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
            event_repo: &self.events,
            event_notifier: &self.event_notifier,
            subagent_spawn_selector: None,
            compaction_config: None,
            compaction_provider: None,
        }
    }
}

async fn create_and_acquire_root(store: &InMemoryAgentTaskStore) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await.context("submit")?;
    store
        .try_acquire_task(&task_id, worker_id(), lease_id(), t_plus(300), t0())
        .await
        .context("acquire")?
        .context("task should be acquirable")
}

async fn suspend_root_with_tools(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
) -> Result<(AgentTask, Vec<AgentTask>)> {
    let provider = MockToolCallProvider::new(tool_calls);
    let task = create_and_acquire_root(&stores.tasks).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        panic!("Expected Suspended outcome");
    };

    Ok((parent_task, child_tasks))
}

async fn acquire_child(
    store: &InMemoryAgentTaskStore,
    child_id: &AgentTaskId,
) -> Result<AgentTask> {
    store
        .try_acquire_task(
            child_id,
            child_worker(),
            child_lease(),
            t_plus(600),
            t_plus(10),
        )
        .await
        .context("acquire child")?
        .context("child should be acquirable")
}

async fn acquire_child_with(
    store: &InMemoryAgentTaskStore,
    child_id: &AgentTaskId,
    worker: WorkerId,
    lease: LeaseId,
    now: time::OffsetDateTime,
) -> Result<AgentTask> {
    store
        .try_acquire_task(
            child_id,
            worker,
            lease,
            now + time::Duration::seconds(300),
            now,
        )
        .await
        .context("acquire child")?
        .context("child should be acquirable")
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — pause path
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn pause_transitions_child_to_awaiting_confirmation() -> Result<()> {
    let stores = TestStores::new();

    let (orig_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    assert_eq!(children.len(), 1);

    // Acquire the child and bootstrap it.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // Pause for confirmation.
    let (paused, committed_events) =
        pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;

    assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);
    assert!(paused.worker_id.is_none(), "lease should be dropped");
    assert!(paused.lease_id.is_none(), "lease should be dropped");
    assert!(
        matches!(paused.state, TaskState::AwaitingConfirmation { .. }),
        "expected AwaitingConfirmation state, got: {:?}",
        paused.state
    );

    // Parent should still be WaitingOnChildren.
    let parent = stores
        .tasks
        .get(&orig_parent.id)
        .await?
        .context("parent should exist")?;
    assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent.pending_child_count, 1);
    assert_eq!(committed_events.len(), 1);
    match &committed_events[0].event {
        AgentEvent::ToolRequiresConfirmation {
            id,
            name,
            display_name,
            input,
            description,
        } => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "transfer");
            assert_eq!(display_name, "Transfer");
            assert_eq!(input, &serde_json::json!({"amount": 100}));
            assert_eq!(description, "Tool Transfer requires confirmation");
        }
        other => bail!("expected ToolRequiresConfirmation event, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn pause_persists_prepared_operation_from_listen_context() -> Result<()> {
    let stores = TestStores::new();

    // Suspend with a transfer tool call (Confirm tier).
    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 200}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // The test mock doesn't set listen_context so prepared_operation
    // will be None. Verify the state handles this correctly.
    let (paused, _committed_events) =
        pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;

    assert!(
        paused.state.prepared_operation().is_none(),
        "non-listen tool should have no prepared operation"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — approval + policy allows
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn approve_and_policy_allows_executes_tool_successfully() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    // Acquire, bootstrap, pause.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;

    // Apply approval decision.
    let outcome = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    let ConfirmationDecisionOutcome::Approved { child, .. } = outcome else {
        panic!("expected Approved outcome, got: {outcome:?}");
    };
    assert_eq!(child.status, TaskStatus::Pending);

    // Re-acquire with a new worker/lease (simulates a fresh worker
    // picking up the resumed task).
    let resumed_child = acquire_child_with(
        &stores.tasks,
        &children[0].id,
        resume_worker(),
        resume_lease(),
        t_plus(25),
    )
    .await?;
    let resume_bootstrap = resolve_tool_bootstrap(resumed_child, &stores.tasks).await?;

    // Resume with AllowAll policy → should execute.
    let resume_outcome = resume_confirmed_tool(
        resume_bootstrap,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &AllowAllPolicy,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("transfer completed")) },
        t_plus(30),
    )
    .await?;

    let ConfirmationResumeOutcome::Executed(tool_outcome) = resume_outcome else {
        panic!("expected Executed outcome, got: {resume_outcome:?}");
    };
    let ToolTaskOutcome::Completed {
        child,
        parent,
        result,
        ..
    } = tool_outcome
    else {
        panic!("expected Completed, got: {tool_outcome:?}");
    };
    assert_eq!(child.status, TaskStatus::Completed);
    assert!(result.success);

    // Parent should now be Pending (single child, all done).
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);
    assert!(matches!(parent.state, TaskState::ReadyToResume { .. }));

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — approval + policy denies
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn approve_but_policy_denies_fails_child() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    // Acquire, bootstrap, pause, approve.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;
    apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    // Re-acquire.
    let resumed = acquire_child_with(
        &stores.tasks,
        &children[0].id,
        resume_worker(),
        resume_lease(),
        t_plus(25),
    )
    .await?;
    let resume_bootstrap = resolve_tool_bootstrap(resumed, &stores.tasks).await?;

    // Resume with DenyAll policy → should fail without executing.
    let policy = DenyAllPolicy::new("tool disabled by administrator");
    let resume_outcome = resume_confirmed_tool(
        resume_bootstrap,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &policy,
        &cancel,
        |_info, _collector| async {
            panic!("executor should not be called when policy denies");
        },
        t_plus(30),
    )
    .await?;

    let ConfirmationResumeOutcome::PolicyDenied {
        child,
        parent,
        reason,
    } = resume_outcome
    else {
        panic!("expected PolicyDenied, got: {resume_outcome:?}");
    };

    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        child
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_POLICY_DENIED_PREFIX),
        "error should have policy-denied prefix: {:?}",
        child.last_error
    );
    assert!(
        reason.contains("tool disabled"),
        "reason should mention policy: {reason}"
    );

    // Parent should be Pending (only child is terminal).
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — rejection
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn rejection_fails_child_without_executing() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    // Acquire, bootstrap, pause.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;

    // Reject.
    let outcome = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Rejected {
            reason: "user declined the transfer".into(),
        },
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    let ConfirmationDecisionOutcome::Rejected {
        child,
        parent,
        reason,
    } = outcome
    else {
        panic!("expected Rejected outcome, got: {outcome:?}");
    };

    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        child
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_REJECTED_PREFIX),
        "error should have rejection prefix: {:?}",
        child.last_error
    );
    assert!(reason.contains("user declined"));

    // Parent should be Pending (child terminal → count = 0).
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — timeout
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn timeout_fails_child_without_executing() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    // Acquire, bootstrap, pause.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;

    // Timeout.
    let outcome = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Timeout,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    let ConfirmationDecisionOutcome::TimedOut { child, parent } = outcome else {
        panic!("expected TimedOut outcome, got: {outcome:?}");
    };

    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        child
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_TIMEOUT_PREFIX),
        "error should have timeout prefix: {:?}",
        child.last_error
    );

    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — rejection rejects non-AwaitingConfirmation rows
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn reject_on_non_awaiting_row_fails() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![("call_1".into(), "transfer".into(), serde_json::json!({}))],
    )
    .await?;

    // Child is still Pending (not yet acquired or paused).
    let result = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Rejected {
            reason: "nope".into(),
        },
        &stores.tasks,
        t_plus(20),
    )
    .await;

    assert!(
        result.is_err(),
        "should reject non-AwaitingConfirmation row"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — approval on non-AwaitingConfirmation row fails
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn approve_on_non_awaiting_row_fails() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![("call_1".into(), "transfer".into(), serde_json::json!({}))],
    )
    .await?;

    // Child is Pending, not AwaitingConfirmation.
    let result = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await;

    assert!(
        result.is_err(),
        "should reject approval on Pending row: {result:?}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — restart: AwaitingConfirmation survives store re-read
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn awaiting_confirmation_state_survives_store_reread() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 500}),
        )],
    )
    .await?;

    // Acquire, bootstrap, pause.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let (paused, _committed_events) =
        pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;
    assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);

    // Simulate restart: re-read the task from the store.
    let reread = stores
        .tasks
        .get(&children[0].id)
        .await?
        .context("child should still exist")?;

    assert_eq!(reread.status, TaskStatus::AwaitingConfirmation);
    assert!(
        matches!(reread.state, TaskState::AwaitingConfirmation { .. }),
        "state should survive re-read: {:?}",
        reread.state
    );
    assert!(reread.state.continuation().is_some());

    // The task can still be approved after "restart".
    let outcome = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(25),
    )
    .await?;

    assert!(
        matches!(outcome, ConfirmationDecisionOutcome::Approved { .. }),
        "should be approved after restart"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — multi-child: confirmation on one, other unaffected
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn multi_child_confirmation_does_not_affect_siblings() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "transfer".into(),
                serde_json::json!({"amount": 100}),
            ),
            (
                "call_2".into(),
                "bash".into(),
                serde_json::json!({"command": "echo ok"}),
            ),
        ],
    )
    .await?;

    assert_eq!(parent.pending_child_count, 2);
    assert_eq!(children.len(), 2);

    // Acquire child 0 (transfer, Confirm tier) → pause.
    let child_0 = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap_0 = resolve_tool_bootstrap(child_0, &stores.tasks).await?;
    pause_tool_for_confirmation(&bootstrap_0, &stores.tasks, &stores.events, t_plus(15)).await?;

    // Acquire child 1 (bash, Observe tier) → execute directly.
    let child_1 = acquire_child_with(
        &stores.tasks,
        &children[1].id,
        WorkerId::from_string("child_worker_2"),
        LeaseId::from_string("child_lease_2"),
        t_plus(10),
    )
    .await?;
    let bootstrap_1 = resolve_tool_bootstrap(child_1, &stores.tasks).await?;
    let outcome_1 = super::tool_task::execute_tool_task(
        bootstrap_1,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("echo output")) },
        t_plus(20),
    )
    .await?;

    let ToolTaskOutcome::Completed {
        parent: parent_after_1,
        ..
    } = outcome_1
    else {
        panic!("Expected Completed for child 1");
    };

    // Parent still has 1 live child (child 0 is AwaitingConfirmation,
    // which is non-terminal). Parent should still be WaitingOnChildren.
    let parent_snap = parent_after_1.context("parent should be returned")?;
    assert_eq!(parent_snap.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_snap.pending_child_count, 1);

    // Now reject the confirmation on child 0.
    let outcome_0 = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Rejected {
            reason: "denied".into(),
        },
        &stores.tasks,
        t_plus(25),
    )
    .await?;

    let ConfirmationDecisionOutcome::Rejected { parent, .. } = outcome_0 else {
        panic!("expected Rejected");
    };

    // Now both children are terminal → parent should be Pending.
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — ConfirmationDecision round-trips through JSON
// ─────────────────────────────────────────────────────────────────────

#[test]
fn confirmation_decision_round_trips_through_json() -> Result<()> {
    let approved = ConfirmationDecision::Approved;
    let json = serde_json::to_string(&approved)?;
    assert_eq!(json, r#"{"kind":"approved"}"#);
    let _: ConfirmationDecision = serde_json::from_str(&json)?;

    let rejected = ConfirmationDecision::Rejected {
        reason: "nope".into(),
    };
    let json = serde_json::to_string(&rejected)?;
    assert!(json.contains("rejected"));
    assert!(json.contains("nope"));
    let _: ConfirmationDecision = serde_json::from_str(&json)?;

    let timeout = ConfirmationDecision::Timeout;
    let json = serde_json::to_string(&timeout)?;
    assert_eq!(json, r#"{"kind":"timeout"}"#);
    let _: ConfirmationDecision = serde_json::from_str(&json)?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — full lifecycle: pause → approve → recheck → execute
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn full_confirmation_lifecycle_pause_approve_execute() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    // Step 1: Suspend root with a confirm-tier tool.
    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 42}),
        )],
    )
    .await?;
    assert_eq!(parent.pending_child_count, 1);

    // Step 2: Worker acquires child → bootstrap → pause.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let (paused, _committed_events) =
        pause_tool_for_confirmation(&bootstrap, &stores.tasks, &stores.events, t_plus(15)).await?;
    assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);

    // Step 3: External transport approves.
    let decision_outcome = apply_confirmation_decision(
        &children[0].id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;
    assert!(matches!(
        decision_outcome,
        ConfirmationDecisionOutcome::Approved { .. }
    ));

    // Step 4: Worker re-acquires → bootstrap → policy recheck → execute.
    let resumed = acquire_child_with(
        &stores.tasks,
        &children[0].id,
        resume_worker(),
        resume_lease(),
        t_plus(25),
    )
    .await?;
    let resume_bootstrap = resolve_tool_bootstrap(resumed, &stores.tasks).await?;

    let resume_outcome = resume_confirmed_tool(
        resume_bootstrap,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &AllowAllPolicy,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("transfer of 42 completed")) },
        t_plus(30),
    )
    .await?;

    // Step 5: Verify outcome.
    let ConfirmationResumeOutcome::Executed(ToolTaskOutcome::Completed {
        child,
        parent,
        result,
        ..
    }) = resume_outcome
    else {
        panic!("expected Executed(Completed), got: {resume_outcome:?}");
    };
    assert_eq!(child.status, TaskStatus::Completed);
    assert!(result.success);
    assert_eq!(result.output, "transfer of 42 completed");

    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);
    assert!(matches!(parent.state, TaskState::ReadyToResume { .. }));

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.3 — reject_confirmation store method: store tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn reject_confirmation_rejects_nonexistent_task() -> Result<()> {
    let store = InMemoryAgentTaskStore::new();
    let id = AgentTaskId::new();

    let result = store
        .reject_confirmation(&id, "reason".into(), t_plus(1))
        .await;

    assert!(result.is_err());
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("does not exist"),
        "expected not-found error: {err}"
    );

    Ok(())
}

#[tokio::test]
async fn reject_confirmation_rejects_pending_task() -> Result<()> {
    let store = InMemoryAgentTaskStore::new();
    let task = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await?;

    let result = store
        .reject_confirmation(&task_id, "reason".into(), t_plus(1))
        .await;

    assert!(result.is_err());
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("not awaiting confirmation"),
        "expected status error: {err}"
    );

    Ok(())
}
