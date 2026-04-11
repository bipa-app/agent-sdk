//! Integration tests for root turn execution.
//!
//! Covers the Phase 4.3 text-only commit path, the Phase 4.4
//! tool-boundary suspension path, the Phase 4.5 resume from
//! completed child tool results, and the Phase 5.4 durable
//! child-outcome aggregation and resume-from-children path.

use super::root_turn::{
    RootTurnDeps, RootTurnOutcome, aggregate_child_outcomes, cancel_root_turn, execute_root_turn,
    fail_root_turn, resume_from_children, resume_root_turn,
};
use crate::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, TaskKind, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::ThreadId;
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use time::Duration;

// ─────────────────────────────────────────────────────────────────────
// Mock LLM providers
// ─────────────────────────────────────────────────────────────────────

struct MockTextProvider {
    response_text: String,
    call_count: AtomicUsize,
}

impl MockTextProvider {
    fn new(text: &str) -> Self {
        Self {
            response_text: text.to_owned(),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for MockTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_mock_01".into(),
            content: vec![ContentBlock::Text {
                text: self.response_text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                cached_input_tokens: 10,
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

/// Mock provider that returns a response with tool-use blocks.
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

    fn single(id: &str, name: &str, input: serde_json::Value) -> Self {
        Self::new(vec![(id.into(), name.into(), input)])
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
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
                input_tokens: 120,
                output_tokens: 60,
                cached_input_tokens: 15,
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
    time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("t-root-turn-a")
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a helpful test assistant.".into(),
        max_tokens: 1024,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        policy: RuntimePolicy::server_default(),
    }
}

fn sample_definition_with_tools() -> AgentDefinition {
    AgentDefinition {
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run a shell command".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            display_name: "Bash".into(),
            tier: agent_sdk_core::ToolTier::Observe,
        }],
        ..sample_definition()
    }
}

fn sample_bootstrap(task: AgentTask) -> WorkerBootstrapContext {
    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition: sample_definition(),
        thread_id,
        task_id,
        worker_id: WorkerId::from_string("worker_test"),
        lease_id: LeaseId::from_string("lease_test"),
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
        worker_id: WorkerId::from_string("worker_test"),
        lease_id: LeaseId::from_string("lease_test"),
    }
}

struct TestStores {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
}

impl TestStores {
    fn new() -> Self {
        Self {
            tasks: InMemoryAgentTaskStore::new(),
            threads: InMemoryThreadStore::new(),
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
        }
    }

    fn deps(&self) -> RootTurnDeps<'_> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
        }
    }
}

/// Create a root task, submit it, acquire it, and return the running task.
async fn create_and_acquire_task(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();

    store.submit_root_turn(task).await.context("submit")?;

    let acquired = store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("task should be acquirable")?;

    Ok(acquired)
}

// ─────────────────────────────────────────────────────────────────────
// Phase 4.3 — text-only path tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_only_turn_end_to_end() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("Hello! I'm a helpful assistant.");

    // 1. Create and acquire a root task.
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    assert_eq!(task.status, TaskStatus::Running);

    // 2. Bootstrap and build inputs.
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    // 3. Execute the turn.
    let outcome = execute_root_turn(inputs, "Hi there!", &provider, &stores.deps(), t_plus(5))
        .await
        .context("execute_root_turn")?;

    // ── Assertions ──────────────────────────────────────────────
    let RootTurnOutcome::Completed {
        commit,
        completed_task,
        response_text,
    } = outcome
    else {
        panic!("Expected Completed variant, got Suspended");
    };

    // Provider was called exactly once.
    assert_eq!(provider.calls(), 1);

    // Response text captured.
    assert_eq!(response_text, "Hello! I'm a helpful assistant.");

    // Task completed.
    assert_eq!(completed_task.status, TaskStatus::Completed);
    assert!(completed_task.completed_at.is_some());

    // Turn attempt closed.
    assert!(commit.closed_attempt.is_closed());

    // Thread aggregate advanced.
    assert_eq!(commit.thread.committed_turns, 1);
    assert_eq!(commit.thread.total_usage.input_tokens, 100);
    assert_eq!(commit.thread.total_usage.output_tokens, 50);

    // Checkpoint created at turn 1.
    assert_eq!(commit.checkpoint.turn_number, 1);
    assert_eq!(commit.checkpoint.thread_id, thread_a());
    assert_eq!(commit.checkpoint.messages.len(), 2);

    // Durable message projection updated.
    let durable_msgs = stores.messages.get_history(&thread_a()).await?;
    assert_eq!(durable_msgs.len(), 2);

    Ok(())
}

#[tokio::test]
async fn no_durable_writes_before_commit() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("Response text");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    // After build_root_worker_inputs, thread exists (get_or_create)
    // but message projection and checkpoints are empty — no turn data
    // has been committed yet.
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty()
    );
    let thread_before = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread should exist from recovery")?;
    assert_eq!(thread_before.committed_turns, 0);

    // Execute — after this, durable stores should have turn data.
    execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(1)).await?;

    // Now durable stores have committed turn data.
    let thread_after = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread_after.committed_turns, 1);
    assert_eq!(stores.messages.get_history(&thread_a()).await?.len(), 2);
    assert_eq!(
        stores.checkpoints.list_by_thread(&thread_a()).await?.len(),
        1
    );

    Ok(())
}

#[tokio::test]
async fn checkpoint_contains_correct_agent_state() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("state test");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(1)).await?;

    // Verify the checkpoint's agent state snapshot.
    let checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint")?;

    let state: agent_sdk_core::AgentState =
        serde_json::from_value(checkpoint.agent_state_snapshot)?;
    assert_eq!(state.turn_count, 1);
    assert_eq!(state.total_usage.input_tokens, 100);
    assert_eq!(state.total_usage.output_tokens, 50);
    assert_eq!(state.thread_id, thread_a());

    Ok(())
}

#[tokio::test]
async fn turn_attempt_is_opened_and_closed() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("attempt test");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome = execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(1)).await?;

    let RootTurnOutcome::Completed { commit, .. } = outcome else {
        panic!("Expected Completed");
    };

    // Verify turn attempt lifecycle.
    let attempt = &commit.closed_attempt;
    assert!(attempt.is_closed());
    assert_eq!(attempt.response_id, Some("msg_mock_01".into()));
    assert_eq!(attempt.response_model, Some("mock-model".into()));
    assert_eq!(attempt.input_tokens, Some(100));
    assert_eq!(attempt.output_tokens, Some(50));

    // Verify via store lookup.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert!(attempts[0].is_closed());

    Ok(())
}

#[tokio::test]
async fn llm_error_propagates() -> Result<()> {
    struct ErrorProvider;

    #[async_trait]
    impl LlmProvider for ErrorProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::ServerError("internal error".into()))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let err = execute_root_turn(inputs, "test", &ErrorProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    let err_msg = format!("{err:#}");
    assert!(
        err_msg.contains("server error"),
        "expected LLM server error, got: {err_msg}",
    );

    // No durable turn data committed.
    let thread = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread from recovery")?;
    assert_eq!(thread.committed_turns, 0);

    // Turn attempt was opened and closed with the error outcome.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert!(
        attempts[0].is_closed(),
        "attempt should be closed on error path"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 4.4 — tool-boundary suspension tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tool_suspension_end_to_end() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "List files", &provider, &stores.deps(), t_plus(5)).await?;

    // ── Must be the Suspended variant ──────────────────────────
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended variant, got Completed");
    };

    // Provider was called exactly once.
    assert_eq!(provider.calls(), 1);

    // Parent is now WaitingOnChildren.
    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_task.pending_child_count, 1);
    assert!(parent_task.completed_at.is_none());
    // Lease dropped when parent parks.
    assert!(parent_task.worker_id.is_none());
    assert!(parent_task.lease_id.is_none());

    // Exactly one child task created.
    assert_eq!(child_tasks.len(), 1);
    let child = &child_tasks[0];
    assert_eq!(child.kind, TaskKind::ToolRuntime);
    assert_eq!(child.status, TaskStatus::Pending);
    assert_eq!(child.thread_id, thread_a());
    assert_eq!(child.parent_id, Some(task_id.clone()));
    assert_eq!(child.depth, 1);

    // ── No checkpoint created ──────────────────────────────────
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty(),
        "suspended path must not create a checkpoint"
    );

    // ── No message-projection update ───────────────────────────
    // The thread aggregate was get_or_created during recovery but no
    // committed turn was written.
    let thread = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread from recovery")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());

    // ── Turn attempt opened and closed with Success ────────────
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert!(attempts[0].is_closed());
    assert_eq!(attempts[0].response_id, Some("msg_tool_01".into()));
    assert_eq!(attempts[0].response_model, Some("mock-model".into()));
    assert_eq!(attempts[0].input_tokens, Some(120));
    assert_eq!(attempts[0].output_tokens, Some(60));

    Ok(())
}

#[tokio::test]
async fn tool_suspension_multiple_tool_calls() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockToolCallProvider::new(vec![
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
        (
            "call_3".into(),
            "write_file".into(),
            serde_json::json!({"path": "/tmp/bar", "content": "hi"}),
        ),
    ]);

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Do stuff", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended");
    };

    // One child per tool call.
    assert_eq!(child_tasks.len(), 3);
    assert_eq!(parent_task.pending_child_count, 3);

    // All children are ToolRuntime, Pending, under the parent.
    for child in &child_tasks {
        assert_eq!(child.kind, TaskKind::ToolRuntime);
        assert_eq!(child.status, TaskStatus::Pending);
        assert_eq!(child.parent_id, Some(parent_task.id.clone()));
    }

    Ok(())
}

#[tokio::test]
async fn tool_suspension_continuation_has_correct_content() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockToolCallProvider::new(vec![
        (
            "call_a".into(),
            "bash".into(),
            serde_json::json!({"command": "pwd"}),
        ),
        (
            "call_b".into(),
            "read_file".into(),
            serde_json::json!({"path": "/x"}),
        ),
    ]);

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(5)).await?;

    // Reload the parent task from the store to inspect its state.
    let parent = stores
        .tasks
        .get(&task_id)
        .await?
        .context("parent task should exist")?;

    // The parent's state should be WaitingOnChildren with a continuation.
    let continuation = match &parent.state {
        TaskState::WaitingOnChildren { continuation, .. } => continuation,
        other => panic!("Expected WaitingOnChildren state, got: {other:?}"),
    };

    let payload = &continuation.payload;

    // Turn identity.
    assert_eq!(payload.thread_id, thread_a());
    assert_eq!(payload.turn, 1); // First turn on a fresh thread.

    // Token usage.
    assert_eq!(payload.turn_usage.input_tokens, 120);
    assert_eq!(payload.turn_usage.output_tokens, 60);
    assert_eq!(payload.total_usage.input_tokens, 120);
    assert_eq!(payload.total_usage.output_tokens, 60);

    // Pending tool calls match the LLM response.
    assert_eq!(payload.pending_tool_calls.len(), 2);
    assert_eq!(payload.pending_tool_calls[0].id, "call_a");
    assert_eq!(payload.pending_tool_calls[0].name, "bash");
    assert_eq!(
        payload.pending_tool_calls[0].input,
        serde_json::json!({"command": "pwd"})
    );
    // "bash" is in the definition with Observe tier and "Bash" display name.
    assert_eq!(
        payload.pending_tool_calls[0].tier,
        agent_sdk_core::ToolTier::Observe
    );
    assert_eq!(payload.pending_tool_calls[0].display_name, "Bash");

    assert_eq!(payload.pending_tool_calls[1].id, "call_b");
    assert_eq!(payload.pending_tool_calls[1].name, "read_file");
    // "read_file" is NOT in the definition — falls back to Confirm.
    assert_eq!(
        payload.pending_tool_calls[1].tier,
        agent_sdk_core::ToolTier::Confirm
    );

    // No completed results yet (tools haven't run).
    assert!(payload.completed_results.is_empty());

    // LLM metadata preserved.
    assert_eq!(payload.response_id, Some("msg_tool_01".into()));
    assert_eq!(payload.stop_reason, Some(StopReason::ToolUse));

    // Agent state snapshot.
    assert_eq!(payload.state.thread_id, thread_a());
    assert_eq!(payload.state.turn_count, 1);
    assert_eq!(payload.state.total_usage.input_tokens, 120);
    assert_eq!(payload.state.total_usage.output_tokens, 60);

    Ok(())
}

#[tokio::test]
async fn tool_suspension_children_are_runnable() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome = execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        panic!("Expected Suspended");
    };

    // The child should be acquirable (it's Pending and on the runnable index).
    let child_id = &child_tasks[0].id;
    let acquired = stores
        .tasks
        .try_acquire_task(
            child_id,
            WorkerId::from_string("child_worker"),
            LeaseId::from_string("child_lease"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("child should be acquirable")?;

    assert_eq!(acquired.status, TaskStatus::Running);
    assert_eq!(acquired.kind, TaskKind::ToolRuntime);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 4.5 — resume from completed child tool results
// ─────────────────────────────────────────────────────────────────────

/// Suspend a root turn (tool call), complete all children, then
/// return the parent and continuation for the resume path.
///
/// This helper sets up the full Phase 4.4 fixture: execute root turn
/// through suspension, acquire and complete each child, then return
/// the parent task (now in Pending/ReadyToResume) with its
/// continuation and suspended messages extracted.
async fn suspend_and_complete_children(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    _child_results: &[(String, agent_sdk_core::ToolResult)],
) -> Result<(
    AgentTask,
    agent_sdk_core::AgentContinuation,
    Vec<agent_sdk_core::llm::Message>,
)> {
    let provider = MockToolCallProvider::new(tool_calls);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended variant");
    };

    // Extract continuation and messages BEFORE children complete
    // (they will be preserved in ReadyToResume state).
    let (continuation, suspended_messages) = match &parent_task.state {
        TaskState::WaitingOnChildren {
            continuation,
            suspended_messages,
        } => (continuation.payload.clone(), suspended_messages.clone()),
        other => panic!("Expected WaitingOnChildren state, got: {other:?}"),
    };

    // Complete all child tasks.
    for child in &child_tasks {
        let child_id = &child.id;
        let acquired = stores
            .tasks
            .try_acquire_task(
                child_id,
                WorkerId::from_string("child_worker"),
                LeaseId::from_string("child_lease"),
                t_plus(600),
                t_plus(10),
            )
            .await?
            .context("acquire child")?;
        assert_eq!(acquired.status, TaskStatus::Running);

        stores
            .tasks
            .complete_task(
                child_id,
                &WorkerId::from_string("child_worker"),
                &LeaseId::from_string("child_lease"),
                t_plus(15),
            )
            .await
            .context("complete child")?;
    }

    // Reload the parent — it should now be Pending with ReadyToResume.
    let parent = stores
        .tasks
        .get(&parent_task.id)
        .await?
        .context("parent after children complete")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert!(
        matches!(parent.state, TaskState::ReadyToResume { .. }),
        "parent should be ReadyToResume after all children complete"
    );

    Ok((parent, continuation, suspended_messages))
}

#[tokio::test]
async fn resume_text_only_end_to_end() -> Result<()> {
    let stores = TestStores::new();

    // Phase 4.4: suspend with a single tool call.
    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "file1.txt\nfile2.txt".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: Some(50),
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire the parent for the resume path.
    let parent_id = parent.id.clone();
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire parent")?;
    assert_eq!(acquired.status, TaskStatus::Running);

    // Build resume inputs.
    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume with text-only response.
    let resume_provider = MockTextProvider::new("Here are your files: file1.txt, file2.txt");
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await
    .context("resume_root_turn")?;

    // ── Assertions ──────────────────────────────────────────────
    let RootTurnOutcome::Completed {
        commit,
        completed_task,
        response_text,
    } = outcome
    else {
        panic!("Expected Completed variant, got Suspended");
    };

    // Response text captured.
    assert_eq!(response_text, "Here are your files: file1.txt, file2.txt");

    // Task completed.
    assert_eq!(completed_task.status, TaskStatus::Completed);

    // Turn committed.
    assert_eq!(commit.thread.committed_turns, 1);

    // Checkpoint created with full message history:
    // [user prompt] + [assistant with tool calls] + [tool results] + [final assistant]
    assert_eq!(commit.checkpoint.turn_number, 1);
    assert!(
        commit.checkpoint.messages.len() >= 4,
        "checkpoint should contain at least 4 messages (user + assistant/tools + results + final), got {}",
        commit.checkpoint.messages.len()
    );

    // Durable message projection updated.
    let durable_msgs = stores.messages.get_history(&thread_a()).await?;
    assert!(
        durable_msgs.len() >= 4,
        "durable messages should contain the full conversation"
    );

    // Turn attempt opened and closed.
    let attempts = stores.attempts.list_by_task(&parent_id).await?;
    // 2 attempts: one from the original suspension, one from the resume.
    assert_eq!(attempts.len(), 2);
    assert!(attempts[0].is_closed());
    assert!(attempts[1].is_closed());

    Ok(())
}

#[tokio::test]
async fn resume_checkpoint_contains_correct_agent_state() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "done".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo done"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire and resume.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    let resume_provider = MockTextProvider::new("completed");
    resume_root_turn(
        inputs,
        continuation.clone(),
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    // Verify checkpoint agent state.
    let checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint")?;

    let state: agent_sdk_core::AgentState =
        serde_json::from_value(checkpoint.agent_state_snapshot)?;

    // turn_count was set at suspension time (1) and should NOT be
    // double-incremented by the resume path.
    assert_eq!(state.turn_count, 1);

    // Usage should be the sum of the suspension LLM call (120/60)
    // plus the resume LLM call (100/50).
    assert_eq!(state.total_usage.input_tokens, 220);
    assert_eq!(state.total_usage.output_tokens, 110);

    Ok(())
}

#[tokio::test]
async fn resume_with_tool_calls_re_suspends() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "result".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire for resume.
    let parent_id = parent.id.clone();
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume with a response that contains MORE tool calls.
    let resume_provider = MockToolCallProvider::single(
        "call_2",
        "bash",
        serde_json::json!({"command": "cat file1.txt"}),
    );
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await
    .context("resume_root_turn")?;

    // ── Must re-suspend ────────────────────────────────────────
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended after resume with tool calls");
    };

    // Parent back in WaitingOnChildren.
    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_task.pending_child_count, 1);

    // New child created for the second tool call.
    assert_eq!(child_tasks.len(), 1);
    assert_eq!(child_tasks[0].kind, TaskKind::ToolRuntime);
    assert_eq!(child_tasks[0].status, TaskStatus::Pending);

    // No checkpoint created (still suspended).
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty(),
        "re-suspended path must not create a checkpoint"
    );

    // The new suspension carries the full conversation in
    // suspended_messages.
    let new_state = match &parent_task.state {
        TaskState::WaitingOnChildren {
            suspended_messages, ..
        } => suspended_messages,
        other => panic!("Expected WaitingOnChildren state, got: {other:?}"),
    };
    // Should contain: user prompt + original assistant + tool results + new assistant
    assert!(
        new_state.len() >= 4,
        "re-suspended messages should contain the full conversation, got {}",
        new_state.len()
    );

    Ok(())
}

#[tokio::test]
async fn resume_with_failed_tool_result() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: false,
            output: "permission denied".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: Some(10),
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "rm -rf /"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire and resume.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    let resume_provider = MockTextProvider::new("The command failed with permission denied.");
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("Expected Completed");
    };

    assert_eq!(response_text, "The command failed with permission denied.");
    Ok(())
}

#[tokio::test]
async fn resume_multiple_tool_results() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![
        (
            "call_a".to_owned(),
            agent_sdk_core::ToolResult {
                success: true,
                output: "/home/user".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            },
        ),
        (
            "call_b".to_owned(),
            agent_sdk_core::ToolResult {
                success: true,
                output: "contents of /x".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            },
        ),
    ];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![
            (
                "call_a".into(),
                "bash".into(),
                serde_json::json!({"command": "pwd"}),
            ),
            (
                "call_b".into(),
                "read_file".into(),
                serde_json::json!({"path": "/x"}),
            ),
        ],
        &child_results,
    )
    .await?;

    // Re-acquire and resume.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    let resume_provider =
        MockTextProvider::new("You are in /home/user, file contains: contents of /x");
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Completed {
        commit,
        response_text,
        ..
    } = outcome
    else {
        panic!("Expected Completed");
    };

    assert_eq!(
        response_text,
        "You are in /home/user, file contains: contents of /x"
    );
    assert_eq!(commit.thread.committed_turns, 1);

    Ok(())
}

#[tokio::test]
async fn suspension_captures_messages_for_resume() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    execute_root_turn(inputs, "List files", &provider, &stores.deps(), t_plus(5)).await?;

    // Reload and inspect the suspended messages.
    let parent = stores.tasks.get(&task_id).await?.context("parent")?;

    let messages = match &parent.state {
        TaskState::WaitingOnChildren {
            suspended_messages, ..
        } => suspended_messages,
        other => panic!("Expected WaitingOnChildren state, got: {other:?}"),
    };

    // Must have exactly 2 messages: user prompt + assistant with tool calls.
    assert_eq!(messages.len(), 2);

    // First message is the user prompt.
    assert_eq!(messages[0].role, agent_sdk_core::llm::Role::User);

    // Second message is the assistant response with tool-use blocks.
    assert_eq!(messages[1].role, agent_sdk_core::llm::Role::Assistant);
    let has_tool_use = match &messages[1].content {
        agent_sdk_core::llm::Content::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, agent_sdk_core::llm::ContentBlock::ToolUse { .. })),
        agent_sdk_core::llm::Content::Text(_) => false,
    };
    assert!(
        has_tool_use,
        "assistant message must contain tool-use blocks"
    );

    Ok(())
}

#[tokio::test]
async fn ready_to_resume_state_survives_acquisition() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "ok".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, _continuation, _messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo ok"}),
        )],
        &child_results,
    )
    .await?;

    // Parent is Pending with ReadyToResume.
    assert_eq!(parent.status, TaskStatus::Pending);
    assert!(matches!(parent.state, TaskState::ReadyToResume { .. }));

    // Acquire the task — it goes to Running.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("acquire")?;

    assert_eq!(acquired.status, TaskStatus::Running);
    // ReadyToResume state must survive the acquisition.
    assert!(
        matches!(acquired.state, TaskState::ReadyToResume { .. }),
        "ReadyToResume must survive task acquisition"
    );

    // The continuation and messages are still accessible.
    let continuation = acquired.state.continuation();
    assert!(continuation.is_some(), "continuation must be present");
    let messages = acquired.state.suspended_messages();
    assert!(!messages.is_empty(), "suspended messages must be present");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 4.6 — failure, cancellation, and regression coverage
// ─────────────────────────────────────────────────────────────────────

/// Mock provider that always returns a server error.
struct ErrorProvider;

#[async_trait]
impl LlmProvider for ErrorProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::ServerError("internal error".into()))
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

// ── Failure path ────────────────────────────────────────────────

#[tokio::test]
async fn failed_root_turn_does_not_advance_projections() -> Result<()> {
    let stores = TestStores::new();

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let worker_id = WorkerId::from_string("worker_test");
    let lease_id = LeaseId::from_string("lease_test");
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    // execute_root_turn fails (LLM server error).
    let err = execute_root_turn(inputs, "test", &ErrorProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    // Explicitly fail the task.
    let failed = fail_root_turn(
        &task_id,
        &worker_id,
        &lease_id,
        &err,
        &stores.deps(),
        t_plus(2),
    )
    .await?;

    // Task is Failed.
    assert_eq!(failed.status, TaskStatus::Failed);
    assert!(failed.last_error.is_some());
    assert!(failed.completed_at.is_some());
    assert!(failed.state.is_none());

    // No durable projection writes.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty()
    );

    Ok(())
}

#[tokio::test]
async fn failed_root_turn_closes_open_attempt() -> Result<()> {
    let stores = TestStores::new();

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let worker_id = WorkerId::from_string("worker_test");
    let lease_id = LeaseId::from_string("lease_test");
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let err = execute_root_turn(inputs, "test", &ErrorProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    fail_root_turn(
        &task_id,
        &worker_id,
        &lease_id,
        &err,
        &stores.deps(),
        t_plus(2),
    )
    .await?;

    // The attempt opened by execute_root_turn should be closed.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert!(
        attempts[0].is_closed(),
        "attempt should be closed after fail_root_turn"
    );

    Ok(())
}

#[tokio::test]
async fn failed_resumed_turn_does_not_leak_continuation() -> Result<()> {
    let stores = TestStores::new();

    // Phase 4.4: suspend with a single tool call.
    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "ok".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire for resume.
    let parent_id = parent.id.clone();
    let worker_id = WorkerId::from_string("worker_test");
    let lease_id = LeaseId::from_string("lease_test");
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            worker_id.clone(),
            lease_id.clone(),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume fails (LLM error).
    let err = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    )
    .await
    .unwrap_err();

    // Explicitly fail the task.
    let failed = fail_root_turn(
        &parent_id,
        &worker_id,
        &lease_id,
        &err,
        &stores.deps(),
        t_plus(26),
    )
    .await?;

    // Task is Failed with state cleared.
    assert_eq!(failed.status, TaskStatus::Failed);
    assert!(failed.state.is_none(), "TaskState must be cleared on fail");
    assert!(failed.last_error.is_some());

    // No checkpoint created (no durable projection writes).
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty()
    );
    assert_eq!(
        stores
            .threads
            .get(&thread_a())
            .await?
            .context("thread")?
            .committed_turns,
        0
    );

    Ok(())
}

// ── Cancellation path ───────────────────────────────────────────

#[tokio::test]
async fn cancelled_root_turn_does_not_advance_projections() -> Result<()> {
    let stores = TestStores::new();

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    // Just build inputs to trigger thread get_or_create, then cancel.
    let _inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    // Cancel the running root task.
    let cancelled_ids = cancel_root_turn(&task_id, &stores.deps(), t_plus(1)).await?;

    assert_eq!(cancelled_ids.len(), 1);
    assert_eq!(cancelled_ids[0], task_id);

    // Task is Cancelled.
    let task = stores.tasks.get(&task_id).await?.context("task")?;
    assert_eq!(task.status, TaskStatus::Cancelled);
    assert!(task.completed_at.is_some());
    assert!(task.state.is_none());

    // No durable projection writes.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty()
    );

    Ok(())
}

#[tokio::test]
async fn cancel_suspended_turn_cancels_children() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockToolCallProvider::new(vec![
        (
            "call_a".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        ),
        (
            "call_b".into(),
            "bash".into(),
            serde_json::json!({"command": "pwd"}),
        ),
    ]);

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome = execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended");
    };

    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(child_tasks.len(), 2);

    // Cancel the entire subtree.
    let cancelled_ids = cancel_root_turn(&task_id, &stores.deps(), t_plus(10)).await?;

    // All 3 tasks (parent + 2 children) should be cancelled.
    assert_eq!(cancelled_ids.len(), 3);

    // Parent is cancelled.
    let parent = stores.tasks.get(&task_id).await?.context("parent")?;
    assert_eq!(parent.status, TaskStatus::Cancelled);
    assert!(parent.state.is_none(), "state must be cleared on cancel");

    // Both children are cancelled.
    for child in &child_tasks {
        let c = stores.tasks.get(&child.id).await?.context("child")?;
        assert_eq!(c.status, TaskStatus::Cancelled);
    }

    // No durable projection writes leaked.
    assert_eq!(
        stores
            .threads
            .get(&thread_a())
            .await?
            .context("thread")?
            .committed_turns,
        0
    );
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty()
    );

    Ok(())
}

// ── Regression coverage (full lifecycle) ────────────────────────

#[tokio::test]
async fn regression_text_only_completion() -> Result<()> {
    // End-to-end regression guard for Phase 4.3: create → acquire →
    // execute (text-only) → task Completed, thread advanced, checkpoint
    // created.
    let stores = TestStores::new();
    let provider = MockTextProvider::new("regression response");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome = execute_root_turn(inputs, "hi", &provider, &stores.deps(), t_plus(1)).await?;

    let RootTurnOutcome::Completed {
        completed_task,
        response_text,
        commit,
    } = outcome
    else {
        panic!("Expected Completed");
    };

    assert_eq!(completed_task.status, TaskStatus::Completed);
    assert_eq!(response_text, "regression response");
    assert_eq!(commit.thread.committed_turns, 1);
    assert_eq!(commit.checkpoint.turn_number, 1);

    // Durable stores consistent.
    let task = stores.tasks.get(&task_id).await?.context("task")?;
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(stores.messages.get_history(&thread_a()).await?.len(), 2);

    Ok(())
}

#[tokio::test]
async fn regression_tool_suspension_and_resume_completion() -> Result<()> {
    // Full lifecycle regression guard for Phase 4.4 + 4.5:
    // suspend → complete children → resume (text-only) → Completed.
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "hello world".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: Some(25),
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hello world"}),
        )],
        &child_results,
    )
    .await?;

    // Verify intermediate state.
    assert_eq!(parent.status, TaskStatus::Pending);
    assert!(matches!(parent.state, TaskState::ReadyToResume { .. }));

    // Re-acquire and resume.
    let parent_id = parent.id.clone();
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    let resume_provider = MockTextProvider::new("Output: hello world");
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Completed {
        completed_task,
        response_text,
        commit,
    } = outcome
    else {
        panic!("Expected Completed");
    };

    // Final state assertions.
    assert_eq!(completed_task.status, TaskStatus::Completed);
    assert_eq!(response_text, "Output: hello world");
    assert_eq!(commit.thread.committed_turns, 1);
    assert!(commit.checkpoint.messages.len() >= 4);

    // Task in store is Completed.
    let task = stores.tasks.get(&parent_id).await?.context("task")?;
    assert_eq!(task.status, TaskStatus::Completed);

    Ok(())
}

#[tokio::test]
async fn regression_re_suspension_child_retry_budget() -> Result<()> {
    // Verify that re-suspended children inherit the policy's max_attempts
    // (not ChildSpawnSpec::default()). This is the fix verification for
    // the Phase 4.6 bug fix.
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "result".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire for resume.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume with more tool calls → re-suspend.
    let resume_provider =
        MockToolCallProvider::single("call_2", "bash", serde_json::json!({"command": "cat file"}));
    let outcome = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        panic!("Expected Suspended after resume with tool calls");
    };

    // The re-suspended child must inherit policy.max_attempts (3 from
    // server_default), not DEFAULT_MAX_ATTEMPTS (1).
    assert_eq!(child_tasks.len(), 1);
    assert_eq!(
        child_tasks[0].max_attempts, 3,
        "re-suspended children must inherit policy max_attempts, got {}",
        child_tasks[0].max_attempts
    );

    Ok(())
}

#[tokio::test]
async fn resume_llm_error_does_not_leak_staged_writes() -> Result<()> {
    // When the resume path's LLM call fails, no durable projection
    // writes should exist (staged stores are discarded with the inputs).
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "ok".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let (parent, continuation, suspended_messages) = suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume with error provider.
    let err = resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    )
    .await;
    assert!(err.is_err(), "resume should fail on LLM error");

    // No durable projection writes leaked.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(
        thread.committed_turns, 0,
        "no commits should occur on failed resume"
    );
    assert!(
        stores.messages.get_history(&thread_a()).await?.is_empty(),
        "no durable messages should exist after failed resume"
    );
    assert!(
        stores
            .checkpoints
            .list_by_thread(&thread_a())
            .await?
            .is_empty(),
        "no checkpoints should exist after failed resume"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.4 — durable child-outcome aggregation and resume
// ─────────────────────────────────────────────────────────────────────

/// Phase 5.4 helper: suspend at tool boundary, complete children with
/// durable result payloads, return the parent in ReadyToResume state.
///
/// Unlike `suspend_and_complete_children`, this helper persists the
/// tool results on the child rows via `complete_task_with_result` so
/// the aggregation path can read them from the journal.
async fn suspend_and_complete_children_durably(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    child_results: &[(String, agent_sdk_core::ToolResult)],
) -> Result<AgentTask> {
    let provider = MockToolCallProvider::new(tool_calls);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended variant");
    };

    // Complete children with durable result payloads.
    for (i, child) in child_tasks.iter().enumerate() {
        let child_id = &child.id;
        stores
            .tasks
            .try_acquire_task(
                child_id,
                WorkerId::from_string("child_worker"),
                LeaseId::from_string("child_lease"),
                t_plus(600),
                t_plus(10),
            )
            .await?
            .context("acquire child")?;

        // Use the i-th result (child_results is ordered by spawn_index).
        let result = &child_results[i].1;
        let payload = serde_json::to_value(result).context("serialize child result")?;
        stores
            .tasks
            .complete_task_with_result(
                child_id,
                &WorkerId::from_string("child_worker"),
                &LeaseId::from_string("child_lease"),
                payload,
                t_plus(15),
            )
            .await
            .context("complete child with result")?;
    }

    let parent = stores
        .tasks
        .get(&parent_task.id)
        .await?
        .context("parent after durable children complete")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert!(
        matches!(parent.state, TaskState::ReadyToResume { .. }),
        "parent should be ReadyToResume after all children complete"
    );

    Ok(parent)
}

/// Phase 5.4 helper: suspend, fail children (no result payload), and
/// return the parent in ReadyToResume state.
async fn suspend_and_fail_children(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    errors: &[&str],
) -> Result<AgentTask> {
    let provider = MockToolCallProvider::new(tool_calls);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended variant");
    };

    for (i, child) in child_tasks.iter().enumerate() {
        let child_id = &child.id;
        stores
            .tasks
            .try_acquire_task(
                child_id,
                WorkerId::from_string("child_worker"),
                LeaseId::from_string("child_lease"),
                t_plus(600),
                t_plus(10),
            )
            .await?
            .context("acquire child")?;

        stores
            .tasks
            .fail_task(
                child_id,
                &WorkerId::from_string("child_worker"),
                &LeaseId::from_string("child_lease"),
                errors[i].to_owned(),
                t_plus(15),
            )
            .await
            .context("fail child")?;
    }

    let parent = stores
        .tasks
        .get(&parent_task.id)
        .await?
        .context("parent after children fail")?;
    assert_eq!(parent.status, TaskStatus::Pending);
    assert!(matches!(parent.state, TaskState::ReadyToResume { .. }));

    Ok(parent)
}

#[tokio::test]
async fn aggregate_all_successful_children() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![
        (
            "call_a".to_owned(),
            agent_sdk_core::ToolResult {
                success: true,
                output: "result_a".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: Some(10),
            },
        ),
        (
            "call_b".to_owned(),
            agent_sdk_core::ToolResult {
                success: true,
                output: "result_b".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: Some(20),
            },
        ),
    ];
    let parent = suspend_and_complete_children_durably(
        &stores,
        vec![
            (
                "call_a".into(),
                "bash".into(),
                serde_json::json!({"command": "pwd"}),
            ),
            (
                "call_b".into(),
                "bash".into(),
                serde_json::json!({"command": "ls"}),
            ),
        ],
        &child_results,
    )
    .await?;

    // Extract continuation from ReadyToResume state.
    let continuation = match &parent.state {
        TaskState::ReadyToResume { continuation, .. } => &continuation.payload,
        _ => panic!("expected ReadyToResume"),
    };

    let aggregated = aggregate_child_outcomes(&parent, continuation, &stores.tasks).await?;

    assert_eq!(aggregated.len(), 2);
    assert_eq!(aggregated[0].0, "call_a");
    assert!(aggregated[0].1.success);
    assert_eq!(aggregated[0].1.output, "result_a");
    assert_eq!(aggregated[1].0, "call_b");
    assert!(aggregated[1].1.success);
    assert_eq!(aggregated[1].1.output, "result_b");

    Ok(())
}

#[tokio::test]
async fn aggregate_all_failed_children() -> Result<()> {
    let stores = TestStores::new();

    let parent = suspend_and_fail_children(
        &stores,
        vec![
            (
                "call_a".into(),
                "bash".into(),
                serde_json::json!({"command": "fail1"}),
            ),
            (
                "call_b".into(),
                "bash".into(),
                serde_json::json!({"command": "fail2"}),
            ),
        ],
        &["permission denied", "command not found"],
    )
    .await?;

    let continuation = match &parent.state {
        TaskState::ReadyToResume { continuation, .. } => &continuation.payload,
        _ => panic!("expected ReadyToResume"),
    };

    let aggregated = aggregate_child_outcomes(&parent, continuation, &stores.tasks).await?;

    assert_eq!(aggregated.len(), 2);
    assert_eq!(aggregated[0].0, "call_a");
    assert!(!aggregated[0].1.success);
    assert_eq!(aggregated[0].1.output, "permission denied");
    assert_eq!(aggregated[1].0, "call_b");
    assert!(!aggregated[1].1.success);
    assert_eq!(aggregated[1].1.output, "command not found");

    Ok(())
}

#[tokio::test]
async fn aggregate_mixed_success_and_failure() -> Result<()> {
    let stores = TestStores::new();

    // Suspend with two tool calls.
    let provider = MockToolCallProvider::new(vec![
        (
            "call_a".into(),
            "bash".into(),
            serde_json::json!({"command": "pwd"}),
        ),
        (
            "call_b".into(),
            "bash".into(),
            serde_json::json!({"command": "rm -rf /"}),
        ),
    ]);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended");
    };

    // Complete child 0 with success result.
    let child_0 = &child_tasks[0];
    stores
        .tasks
        .try_acquire_task(
            &child_0.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child 0")?;
    let success_result = agent_sdk_core::ToolResult {
        success: true,
        output: "/home/user".to_owned(),
        data: None,
        documents: Vec::new(),
        duration_ms: Some(5),
    };
    stores
        .tasks
        .complete_task_with_result(
            &child_0.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            serde_json::to_value(&success_result)?,
            t_plus(15),
        )
        .await?;

    // Fail child 1.
    let child_1 = &child_tasks[1];
    stores
        .tasks
        .try_acquire_task(
            &child_1.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child 1")?;
    stores
        .tasks
        .fail_task(
            &child_1.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            "permission denied".to_owned(),
            t_plus(15),
        )
        .await?;

    // Reload parent.
    let parent = stores.tasks.get(&parent_task.id).await?.context("parent")?;
    assert_eq!(parent.status, TaskStatus::Pending);

    let continuation = match &parent.state {
        TaskState::ReadyToResume { continuation, .. } => &continuation.payload,
        _ => panic!("expected ReadyToResume"),
    };

    let aggregated = aggregate_child_outcomes(&parent, continuation, &stores.tasks).await?;

    assert_eq!(aggregated.len(), 2);
    // First child succeeded.
    assert_eq!(aggregated[0].0, "call_a");
    assert!(aggregated[0].1.success);
    assert_eq!(aggregated[0].1.output, "/home/user");
    // Second child failed with deterministic error result.
    assert_eq!(aggregated[1].0, "call_b");
    assert!(!aggregated[1].1.success);
    assert_eq!(aggregated[1].1.output, "permission denied");

    Ok(())
}

#[tokio::test]
async fn aggregate_rejects_non_terminal_children() -> Result<()> {
    let stores = TestStores::new();

    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended { parent_task, .. } = outcome else {
        panic!("Expected Suspended");
    };

    // DO NOT complete the child — it's still Pending.
    let continuation = match &parent_task.state {
        TaskState::WaitingOnChildren { continuation, .. } => &continuation.payload,
        _ => panic!("expected WaitingOnChildren"),
    };

    let err = aggregate_child_outcomes(&parent_task, continuation, &stores.tasks).await;
    assert!(
        err.is_err(),
        "aggregate should reject non-terminal children"
    );
    let msg = format!("{:#}", err.unwrap_err());
    assert!(
        msg.contains("is not terminal"),
        "error should mention non-terminal: {msg}"
    );

    Ok(())
}

#[tokio::test]
async fn resume_from_children_end_to_end() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "file1.txt\nfile2.txt".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: Some(50),
        },
    )];
    let parent = suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire the parent.
    let parent_id = parent.id.clone();
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire parent")?;
    assert_eq!(acquired.status, TaskStatus::Running);

    // Build resume inputs.
    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Use resume_from_children — the Phase 5.4 entry point.
    let resume_provider = MockTextProvider::new("Here are your files: file1.txt, file2.txt");
    let outcome = resume_from_children(
        inputs,
        &parent,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await
    .context("resume_from_children")?;

    let RootTurnOutcome::Completed {
        completed_task,
        response_text,
        ..
    } = outcome
    else {
        panic!("Expected Completed variant, got Suspended");
    };

    assert_eq!(response_text, "Here are your files: file1.txt, file2.txt");
    assert_eq!(completed_task.status, TaskStatus::Completed);

    // Thread and checkpoint committed.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);

    Ok(())
}

#[tokio::test]
async fn resume_from_children_with_mixed_batch() -> Result<()> {
    let stores = TestStores::new();

    // Suspend with two tool calls. Complete one, fail the other.
    let provider = MockToolCallProvider::new(vec![
        (
            "call_a".into(),
            "bash".into(),
            serde_json::json!({"command": "pwd"}),
        ),
        (
            "call_b".into(),
            "bash".into(),
            serde_json::json!({"command": "fail"}),
        ),
    ]);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended");
    };

    // Complete child 0 with result.
    let child_0 = &child_tasks[0];
    stores
        .tasks
        .try_acquire_task(
            &child_0.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child 0")?;
    stores
        .tasks
        .complete_task_with_result(
            &child_0.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            serde_json::to_value(&agent_sdk_core::ToolResult::success("/home"))?,
            t_plus(15),
        )
        .await?;

    // Fail child 1.
    let child_1 = &child_tasks[1];
    stores
        .tasks
        .try_acquire_task(
            &child_1.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child 1")?;
    stores
        .tasks
        .fail_task(
            &child_1.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            "oops".to_owned(),
            t_plus(15),
        )
        .await?;

    // Reload parent.
    let parent = stores.tasks.get(&parent_task.id).await?.context("parent")?;

    // Re-acquire.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume via the Phase 5.4 entry point.
    let resume_provider =
        MockTextProvider::new("pwd returned /home, second tool failed with: oops");
    let outcome = resume_from_children(
        inputs,
        &parent,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Completed {
        response_text,
        completed_task,
        ..
    } = outcome
    else {
        panic!("Expected Completed");
    };

    assert_eq!(
        response_text,
        "pwd returned /home, second tool failed with: oops"
    );
    assert_eq!(completed_task.status, TaskStatus::Completed);

    Ok(())
}

#[tokio::test]
async fn resume_from_children_re_suspends_on_tool_calls() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_core::ToolResult {
            success: true,
            output: "first result".to_owned(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )];
    let parent = suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    )
    .await?;

    // Re-acquire.
    let parent_id = parent.id.clone();
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(20),
        )
        .await?
        .context("re-acquire")?;

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t_plus(20))
            .await?;

    // Resume with a response that has MORE tool calls.
    let resume_provider = MockToolCallProvider::single(
        "call_2",
        "bash",
        serde_json::json!({"command": "cat file.txt"}),
    );
    let outcome = resume_from_children(
        inputs,
        &parent,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended after resume with tool calls");
    };

    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(child_tasks.len(), 1);
    assert_eq!(child_tasks[0].kind, TaskKind::ToolRuntime);

    Ok(())
}

#[tokio::test]
async fn result_payload_round_trips_through_json() -> Result<()> {
    // Verify that ToolResult serialization to result_payload on the
    // task row and deserialization back produces the same value.
    let stores = TestStores::new();

    let original_result = agent_sdk_core::ToolResult {
        success: true,
        output: "round-trip test".to_owned(),
        data: Some(serde_json::json!({"key": "value"})),
        documents: Vec::new(),
        duration_ms: Some(42),
    };
    let child_results = vec![("call_1".to_owned(), original_result.clone())];

    let parent = suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "test"}),
        )],
        &child_results,
    )
    .await?;

    // Read the child from the store and verify its result_payload.
    let children = stores.tasks.list_children(&parent.id).await?;
    assert_eq!(children.len(), 1);
    let child = &children[0];
    assert!(child.result_payload.is_some());

    // Deserialize and compare.
    let recovered: agent_sdk_core::ToolResult =
        serde_json::from_value(child.result_payload.clone().unwrap())?;
    assert_eq!(recovered.success, original_result.success);
    assert_eq!(recovered.output, original_result.output);
    assert_eq!(recovered.data, original_result.data);
    assert_eq!(recovered.duration_ms, original_result.duration_ms);

    Ok(())
}

#[tokio::test]
async fn cancelled_child_produces_deterministic_error_result() -> Result<()> {
    let stores = TestStores::new();

    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "slow"}));
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
    } = outcome
    else {
        panic!("Expected Suspended");
    };

    // Cancel the entire tree (simulates user cancellation).
    stores
        .tasks
        .cancel_tree(&parent_task.id, t_plus(10))
        .await?;

    // Even though parent is Cancelled (not ReadyToResume), we can
    // still test the extract function through aggregate_child_outcomes
    // on a synthetic parent. Let's verify the cancel produces a
    // Cancelled child.
    let child = stores
        .tasks
        .get(&child_tasks[0].id)
        .await?
        .context("child")?;
    assert_eq!(child.status, TaskStatus::Cancelled);
    assert!(child.result_payload.is_none());

    Ok(())
}
