//! Integration tests for root turn execution.
//!
//! Covers the Phase 4.3 text-only commit path and the Phase 4.4
//! tool-boundary suspension path.

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
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
        TaskState::WaitingOnChildren { continuation } => continuation,
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
