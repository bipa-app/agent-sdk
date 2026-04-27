//! Phase 5.1 regression tests for tool-runtime child-task execution.
//!
//! Covers:
//! - Happy path: suspend root → acquire child → bootstrap → execute
//!   success → child completes → parent `pending_child_count` decrements.
//! - Failure path: same setup → execute error → child fails → parent
//!   state unaffected.
//! - Cancellation: cancelled token → execution bails without
//!   completing or failing the child.
//! - Bootstrap rejections: wrong task kind, wrong status, missing
//!   parent, parent with wrong state.
//! - Positional mapping: multiple tool calls → correct tool call
//!   info resolved per child.
//! - End-to-end regression: root suspend + complete all children +
//!   verify parent becomes Pending/ReadyToResume.
use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::tool_task::{ToolTaskOutcome, execute_tool_task, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_core::{ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
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
            id: "msg_tool_01".into(),
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
// Test helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("thread_tool_test")
}

fn worker_id() -> WorkerId {
    WorkerId::from_string("worker_test")
}

fn lease_id() -> LeaseId {
    LeaseId::from_string("lease_test")
}

fn child_worker() -> WorkerId {
    WorkerId::from_string("child_worker")
}

fn child_lease() -> LeaseId {
    LeaseId::from_string("child_lease")
}

fn sample_definition_with_tools() -> AgentDefinition {
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
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                display_name: "Read File".into(),
                tier: ToolTier::Observe,
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
        definition: sample_definition_with_tools(),
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
        }
    }
}

/// Create a root task, submit it, acquire it, and return the running task.
async fn create_and_acquire_root(store: &InMemoryAgentTaskStore) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let task_id = task.id.clone();

    store.submit_root_turn(task).await.context("submit")?;

    let acquired = store
        .try_acquire_task(&task_id, worker_id(), lease_id(), t_plus(300), t0())
        .await
        .context("acquire")?
        .context("task should be acquirable")?;

    Ok(acquired)
}

/// Suspend a root turn with the given tool calls, returning the
/// parent and child tasks.
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

/// Acquire a child task and return it in Running state.
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

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — bootstrap tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bootstrap_resolves_tool_call_from_parent_continuation() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hello"}),
        )],
    )
    .await?;

    assert_eq!(children.len(), 1);

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    assert_eq!(ctx.tool_call.name, "bash");
    assert_eq!(ctx.tool_call.id, "call_1");
    assert_eq!(
        ctx.tool_call.input,
        serde_json::json!({"command": "echo hello"})
    );
    assert_eq!(ctx.child_task.kind, TaskKind::ToolRuntime);
    assert_eq!(ctx.child_task.status, TaskStatus::Running);

    Ok(())
}

#[tokio::test]
async fn bootstrap_rejects_root_turn_task() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_root(&stores.tasks).await?;

    let err = resolve_tool_bootstrap(task, &stores.tasks)
        .await
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("ToolRuntime"),
        "expected ToolRuntime rejection, got: {err:#}"
    );

    Ok(())
}

#[tokio::test]
async fn bootstrap_rejects_pending_child() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![("call_1".into(), "bash".into(), serde_json::json!({}))],
    )
    .await?;

    // Don't acquire — child is still Pending.
    let child = stores
        .tasks
        .get(&children[0].id)
        .await?
        .context("child should exist")?;
    assert_eq!(child.status, TaskStatus::Pending);

    let err = resolve_tool_bootstrap(child, &stores.tasks)
        .await
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("Running"),
        "expected Running rejection, got: {err:#}"
    );

    Ok(())
}

#[tokio::test]
async fn bootstrap_rejects_missing_parent() -> Result<()> {
    // Fabricate a ToolRuntime child whose parent_id points to a
    // nonexistent row. We can't use the real store's insert (it
    // validates parent references), so we directly construct the
    // scenario by using the store trait's `get` returning None.
    //
    // Simplest approach: acquire a real child, then ask a *different*
    // store (which has no parent) to bootstrap it.
    let stores = TestStores::new();
    let stores2 = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![("call_1".into(), "bash".into(), serde_json::json!({}))],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;

    // stores2 has never seen the parent — resolution should fail.
    let err = resolve_tool_bootstrap(child, &stores2.tasks)
        .await
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("does not exist"),
        "expected parent-not-found error, got: {err:#}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — positional mapping tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bootstrap_maps_multiple_children_to_correct_tool_calls() -> Result<()> {
    let stores = TestStores::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "bash".into(),
                serde_json::json!({"command": "ls"}),
            ),
            (
                "call_2".into(),
                "read_file".into(),
                serde_json::json!({"path": "/tmp/foo"}),
            ),
        ],
    )
    .await?;

    assert_eq!(children.len(), 2);

    // Acquire and bootstrap each child.
    let child_0 = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx_0 = resolve_tool_bootstrap(child_0, &stores.tasks).await?;

    // Need a fresh lease for the second child.
    let child_1 = stores
        .tasks
        .try_acquire_task(
            &children[1].id,
            WorkerId::from_string("child_worker_2"),
            LeaseId::from_string("child_lease_2"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire second child")?;
    let ctx_1 = resolve_tool_bootstrap(child_1, &stores.tasks).await?;

    assert_eq!(ctx_0.tool_call.id, "call_1");
    assert_eq!(ctx_0.tool_call.name, "bash");
    assert_eq!(ctx_0.tool_call.input, serde_json::json!({"command": "ls"}));

    assert_eq!(ctx_1.tool_call.id, "call_2");
    assert_eq!(ctx_1.tool_call.name, "read_file");
    assert_eq!(
        ctx_1.tool_call.input,
        serde_json::json!({"path": "/tmp/foo"})
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — happy path execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_success_completes_child_and_decrements_parent() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo ok"}),
        )],
    )
    .await?;

    assert_eq!(parent.pending_child_count, 1);
    assert_eq!(children.len(), 1);

    // Acquire and bootstrap the child.
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // Execute with a successful result.
    let outcome = execute_tool_task(
        ctx,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("command output")) },
        t_plus(20),
    )
    .await?;

    // Verify outcome.
    let ToolTaskOutcome::Completed {
        child,
        parent,
        result,
        ..
    } = outcome
    else {
        panic!("Expected Completed outcome, got: {outcome:?}");
    };

    assert_eq!(child.status, TaskStatus::Completed);
    assert!(result.success);
    assert_eq!(result.output, "command output");

    // Parent should be back to Pending (all children done).
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    // Parent state should be ReadyToResume.
    assert!(
        matches!(parent.state, TaskState::ReadyToResume { .. }),
        "expected ReadyToResume, got: {:?}",
        parent.state
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — failure path execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_failure_fails_child_without_corrupting_parent() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "exit 1"}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // Execute with an error result.
    let outcome = execute_tool_task(
        ctx,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Err(anyhow::anyhow!("command failed with exit code 1")) },
        t_plus(20),
    )
    .await?;

    let ToolTaskOutcome::Failed {
        child,
        parent,
        error,
        ..
    } = outcome
    else {
        panic!("Expected Failed outcome, got: {outcome:?}");
    };

    assert_eq!(child.status, TaskStatus::Failed);
    assert!(error.contains("command failed with exit code 1"));

    // Parent should still become Pending (failed child is terminal).
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    // Verify the parent's task_id is the root's parent_id.
    assert_eq!(Some(&parent.id), children[0].parent_id.as_ref());

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — cancellation tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_cancelled_before_tool_returns_cancelled() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo ok"}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child.clone(), &stores.tasks).await?;

    // Cancel before execution.
    cancel.cancel();

    let outcome = execute_tool_task(
        ctx,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("should not reach")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Cancelled),
        "Expected Cancelled, got: {outcome:?}"
    );

    // Child should still be Running (not driven to terminal).
    let child_after = stores
        .tasks
        .get(&child.id)
        .await?
        .context("child should still exist")?;
    assert_eq!(child_after.status, TaskStatus::Running);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — multi-child end-to-end regression
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn multi_child_complete_all_makes_parent_resumable() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "bash".into(),
                serde_json::json!({"command": "ls"}),
            ),
            (
                "call_2".into(),
                "read_file".into(),
                serde_json::json!({"path": "/tmp/foo"}),
            ),
        ],
    )
    .await?;

    assert_eq!(parent.pending_child_count, 2);
    assert_eq!(children.len(), 2);

    // Complete first child — parent should still be WaitingOnChildren.
    let child_0 = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx_0 = resolve_tool_bootstrap(child_0, &stores.tasks).await?;
    let outcome_0 = execute_tool_task(
        ctx_0,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("ls output")) },
        t_plus(20),
    )
    .await?;

    let ToolTaskOutcome::Completed {
        parent: parent_after_0,
        ..
    } = outcome_0
    else {
        panic!("Expected Completed");
    };

    let parent_0 = parent_after_0.context("parent should be returned")?;
    assert_eq!(parent_0.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_0.pending_child_count, 1);

    // Complete second child — parent should become Pending.
    let child_1 = stores
        .tasks
        .try_acquire_task(
            &children[1].id,
            WorkerId::from_string("child_worker_2"),
            LeaseId::from_string("child_lease_2"),
            t_plus(600),
            t_plus(25),
        )
        .await?
        .context("acquire second child")?;
    let ctx_1 = resolve_tool_bootstrap(child_1, &stores.tasks).await?;
    let outcome_1 = execute_tool_task(
        ctx_1,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("file contents")) },
        t_plus(30),
    )
    .await?;

    let ToolTaskOutcome::Completed {
        parent: parent_after_1,
        ..
    } = outcome_1
    else {
        panic!("Expected Completed");
    };

    let parent_1 = parent_after_1.context("parent should be returned")?;
    assert_eq!(parent_1.status, TaskStatus::Pending);
    assert_eq!(parent_1.pending_child_count, 0);
    assert!(
        matches!(parent_1.state, TaskState::ReadyToResume { .. }),
        "expected ReadyToResume, got: {:?}",
        parent_1.state
    );

    Ok(())
}

#[tokio::test]
async fn mixed_success_and_failure_still_makes_parent_resumable() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "bash".into(),
                serde_json::json!({"command": "ls"}),
            ),
            (
                "call_2".into(),
                "read_file".into(),
                serde_json::json!({"path": "/tmp/missing"}),
            ),
        ],
    )
    .await?;

    // Complete first child successfully.
    let child_0 = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx_0 = resolve_tool_bootstrap(child_0, &stores.tasks).await?;
    execute_tool_task(
        ctx_0,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("ok")) },
        t_plus(20),
    )
    .await?;

    // Fail second child.
    let child_1 = stores
        .tasks
        .try_acquire_task(
            &children[1].id,
            WorkerId::from_string("child_worker_2"),
            LeaseId::from_string("child_lease_2"),
            t_plus(600),
            t_plus(25),
        )
        .await?
        .context("acquire second child")?;
    let ctx_1 = resolve_tool_bootstrap(child_1, &stores.tasks).await?;
    let outcome = execute_tool_task(
        ctx_1,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Err(anyhow::anyhow!("file not found")) },
        t_plus(30),
    )
    .await?;

    let ToolTaskOutcome::Failed { parent, .. } = outcome else {
        panic!("Expected Failed");
    };

    // Parent should be Pending regardless of child failure.
    let parent = parent.context("parent should be returned")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.pending_child_count, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.1 — lease awareness
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_with_wrong_lease_fails_at_complete() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![("call_1".into(), "bash".into(), serde_json::json!({}))],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let mut ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // Tamper with the lease to simulate expiry + re-acquisition.
    ctx.lease_id = LeaseId::from_string("stale_lease");

    let result = execute_tool_task(
        ctx,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("ok")) },
        t_plus(20),
    )
    .await;

    // The store's CAS guard should reject the stale lease.
    assert!(result.is_err(), "expected CAS rejection, got: {result:?}");

    Ok(())
}
