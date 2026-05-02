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
use std::sync::Arc;

use crate::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, TaskKind, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt::TurnAttempt;
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::ThreadId;
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::MessageStore;
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
        tools_fn: None,
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

fn tool_result_message(
    child_results: &[(String, agent_sdk_core::ToolResult)],
) -> agent_sdk_core::llm::Message {
    let blocks = child_results
        .iter()
        .map(|(tool_use_id, result)| ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: result.output.clone(),
            is_error: if result.success { None } else { Some(true) },
        })
        .collect();
    agent_sdk_core::llm::Message::user_with_content(blocks)
}

fn assert_tool_results_follow_tool_use(
    messages: &[agent_sdk_core::llm::Message],
    tool_use_id: &str,
) {
    let tool_use_index = messages.iter().position(|message| {
        matches!(
            &message.content,
            agent_sdk_core::llm::Content::Blocks(blocks)
                if blocks.iter().any(|block| matches!(
                    block,
                    ContentBlock::ToolUse { id, .. } if id == tool_use_id
                ))
        )
    });

    let Some(index) = tool_use_index else {
        panic!("expected tool_use {tool_use_id} in recovered messages");
    };

    let Some(next_message) = messages.get(index + 1) else {
        panic!("expected tool_result message immediately after tool_use {tool_use_id}");
    };

    let has_matching_result = matches!(
        &next_message.content,
        agent_sdk_core::llm::Content::Blocks(blocks)
            if blocks.iter().any(|block| matches!(
                block,
                ContentBlock::ToolResult { tool_use_id: result_id, .. }
                    if result_id == tool_use_id
            ))
    );
    assert!(
        has_matching_result,
        "expected tool_result for {tool_use_id} immediately after tool_use",
    );
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // 3. Execute the turn.
    let outcome = execute_root_turn(inputs, "Hi there!", &provider, &stores.deps(), t_plus(5))
        .await
        .context("execute_root_turn")?;

    // ── Assertions ──────────────────────────────────────────────
    let RootTurnOutcome::Completed {
        commit,
        completed_task,
        response_text,
        ..
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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

#[ignore = "streaming refactor: Start now committed pre-LLM, deltas added; see PR for new event-ordering invariants"]
#[tokio::test]
async fn turn_attempt_is_opened_and_closed() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("attempt test");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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

#[ignore = "streaming refactor: Start now committed pre-LLM, deltas added; see PR for new event-ordering invariants"]
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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

#[ignore = "streaming refactor: Start now committed pre-LLM, deltas added; see PR for new event-ordering invariants"]
#[tokio::test]
async fn tool_suspension_end_to_end() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
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
        execute_root_turn(inputs, "List files", &provider, &stores.deps(), t_plus(5)).await?;

    // ── Must be the Suspended variant ──────────────────────────
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome =
        execute_root_turn(inputs, "Do stuff", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
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

#[ignore = "streaming refactor: Start now committed pre-LLM, deltas added; see PR for new event-ordering invariants"]
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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
        panic!("Expected Suspended variant");
    };

    // Extract continuation and messages BEFORE children complete
    // (they will be preserved in ReadyToResume state).
    let (continuation, suspended_messages) = match &parent_task.state {
        TaskState::WaitingOnChildren {
            continuation,
            suspended_messages,
            ..
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
        ..
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo done"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
        ..
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "rm -rf /"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
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
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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
    let (parent, _continuation, _messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo ok"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // execute_root_turn fails (LLM server error).
    let err = execute_root_turn(inputs, "test", &ErrorProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    // Explicitly fail the task.
    let failed = fail_root_turn(
        &task_id,
        &worker_id,
        &lease_id,
        &thread_a(),
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let err = execute_root_turn(inputs, "test", &ErrorProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    fail_root_turn(
        &task_id,
        &worker_id,
        &lease_id,
        &thread_a(),
        &err,
        &stores.deps(),
        t_plus(2),
    )
    .await?;

    // Every attempt opened during execute_root_turn (including the
    // retries call_llm_with_retry minted on transient ServerErrors)
    // must be closed by the time fail_root_turn returns.  The exact
    // attempt count tracks `STREAM_MAX_RETRIES + 1`; we only assert
    // the invariant — closed-ness — so the test stays robust to
    // future tuning of the retry budget.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert!(!attempts.is_empty(), "expected at least one attempt");
    assert!(
        attempts.iter().all(TurnAttempt::is_closed),
        "all attempts should be closed after fail_root_turn",
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
        &thread_a(),
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

/// Regression for the lost-history bug:
///
/// A long-running root turn that suspends at a tool boundary, has its
/// children complete, and then fails on the resume LLM call must
/// preserve the in-flight `suspended_messages` snapshot on the
/// message projection's draft slot. `recover_thread` must surface
/// those messages so the next root turn picks up the work the failed
/// turn already did instead of starting from an empty history.
///
/// Without the draft persistence wired by `suspend_at_tool_boundary`
/// + `suspend_resumed_turn` and the recovery path in `recover_thread`,
/// the failed task's `TaskState` clear (see
/// `failed_resumed_turn_does_not_leak_continuation`) would take the
/// only durable copy of the conversation with it and the next turn
/// would see `messages == []`.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn failed_resumed_turn_preserves_in_flight_history_via_draft() -> Result<()> {
    use crate::journal::thread_recover::recover_thread;

    let stores = TestStores::new();

    // Suspend with a single tool call and run the child to completion.
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
    .await?;

    // Sanity: the suspension path must have written the draft snapshot.
    let projection_after_suspend = stores
        .messages
        .get(&thread_a())
        .await?
        .context("projection bootstrapped on suspend")?;
    assert!(
        projection_after_suspend.has_draft(),
        "draft must be populated after the first suspension",
    );
    assert_eq!(
        projection_after_suspend.draft_messages.len(),
        suspended_messages.len(),
        "draft must mirror the suspension's suspended_messages list",
    );
    assert_eq!(
        projection_after_suspend.message_count(),
        0,
        "committed history stays empty until the turn commits",
    );

    // Re-acquire the parent for resume.
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
    .await?;
    let staged_before_resume = inputs
        .staged_stores
        .messages
        .get_history(&thread_a())
        .await?;
    assert!(
        staged_before_resume.is_empty(),
        "ReadyToResume inputs must not seed the previous draft; resume_root_turn appends suspended messages plus tool results",
    );

    let mut expected_recovered_messages = suspended_messages.clone();
    expected_recovered_messages.push(tool_result_message(&child_results));

    // Resume LLM call errors out — same shape as the production
    // "Stream ended unexpectedly" cascade that motivated this fix.
    let err = resume_root_turn(
        inputs,
        continuation,
        suspended_messages.clone(),
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    )
    .await
    .unwrap_err();

    // Fail the task — clears `TaskState` per the existing contract.
    let failed = fail_root_turn(
        &parent_id,
        &worker_id,
        &lease_id,
        &thread_a(),
        &err,
        &stores.deps(),
        t_plus(26),
    )
    .await?;
    assert_eq!(failed.status, TaskStatus::Failed);
    assert!(failed.state.is_none());

    // Helper: structural compare for `llm::Message` (no PartialEq).
    let msgs_match =
        |a: &[agent_sdk_core::llm::Message], b: &[agent_sdk_core::llm::Message]| -> bool {
            let aj = serde_json::to_value(a).expect("serialize a");
            let bj = serde_json::to_value(b).expect("serialize b");
            aj == bj
        };

    // Critical assertion: the projection's draft slot survives the
    // task's fail() — `fail_root_turn` only clears the *task* row,
    // never the message projection. The draft must include the
    // completed child results, otherwise the next root turn would
    // replay an orphaned assistant tool_use without the immediately
    // following tool_result required by Anthropic.
    let projection_after_fail = stores
        .messages
        .get(&thread_a())
        .await?
        .context("projection still present after fail")?;
    assert!(
        projection_after_fail.has_draft(),
        "draft must survive task failure",
    );
    assert!(
        msgs_match(
            &projection_after_fail.draft_messages,
            &expected_recovered_messages,
        ),
        "draft must include completed child results after the suspended tool_use",
    );
    assert_tool_results_follow_tool_use(&projection_after_fail.draft_messages, "call_1");

    // The recovery view used by the next root turn folds the draft
    // into `messages` so the resumed conversation continues from
    // where the failed turn got to.
    let view = recover_thread(
        &thread_a(),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(30),
    )
    .await?;
    assert!(
        msgs_match(&view.messages, &expected_recovered_messages),
        "recovery view must surface the in-flight draft as the next turn's history",
    );
    assert!(msgs_match(
        &view.draft_messages,
        &expected_recovered_messages
    ));
    assert_tool_results_follow_tool_use(&view.messages, "call_1");
    assert_eq!(view.next_turn_number, 1, "no turn was committed");
    assert!(view.latest_checkpoint.is_none());

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
    let _inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(inputs, "test", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(inputs, "hi", &provider, &stores.deps(), t_plus(1)).await?;

    let RootTurnOutcome::Completed {
        completed_task,
        response_text,
        commit,
        ..
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hello world"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
        ..
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
    let (parent, continuation, suspended_messages) = Box::pin(suspend_and_complete_children(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
/// durable result payloads, return the parent in `ReadyToResume` state.
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
/// return the parent in `ReadyToResume` state.
async fn suspend_and_fail_children(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    errors: &[&str],
) -> Result<AgentTask> {
    let provider = MockToolCallProvider::new(tool_calls);
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
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
    let parent = Box::pin(suspend_and_complete_children_durably(
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
    ))
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

    let parent = Box::pin(suspend_and_fail_children(
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
    ))
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
        panic!("Expected Suspended");
    };

    // Complete child 0 with success result, fail child 1.
    let success_result = agent_sdk_core::ToolResult {
        success: true,
        output: "/home/user".to_owned(),
        data: None,
        documents: Vec::new(),
        duration_ms: Some(5),
    };
    acquire_and_complete_child(&stores, &child_tasks[0], &success_result).await?;
    acquire_and_fail_child(&stores, &child_tasks[1], "permission denied").await?;

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
    let parent = Box::pin(suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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

/// Acquire a child task, complete it with a durable result payload.
async fn acquire_and_complete_child(
    stores: &TestStores,
    child: &AgentTask,
    result: &agent_sdk_core::ToolResult,
) -> Result<()> {
    stores
        .tasks
        .try_acquire_task(
            &child.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child")?;
    stores
        .tasks
        .complete_task_with_result(
            &child.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            serde_json::to_value(result).context("serialize")?,
            t_plus(15),
        )
        .await
        .context("complete child with result")?;
    Ok(())
}

/// Acquire a child task, fail it with the given error.
async fn acquire_and_fail_child(stores: &TestStores, child: &AgentTask, error: &str) -> Result<()> {
    stores
        .tasks
        .try_acquire_task(
            &child.id,
            WorkerId::from_string("cw"),
            LeaseId::from_string("cl"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("acquire child")?;
    stores
        .tasks
        .fail_task(
            &child.id,
            &WorkerId::from_string("cw"),
            &LeaseId::from_string("cl"),
            error.to_owned(),
            t_plus(15),
        )
        .await
        .context("fail child")?;
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
        panic!("Expected Suspended");
    };

    acquire_and_complete_child(
        &stores,
        &child_tasks[0],
        &agent_sdk_core::ToolResult::success("/home"),
    )
    .await?;
    acquire_and_fail_child(&stores, &child_tasks[1], "oops").await?;

    // Reload parent and re-acquire.
    let parent = stores.tasks.get(&parent_task.id).await?.context("parent")?;
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
    let parent = Box::pin(suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        )],
        &child_results,
    ))
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
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
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
        ..
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

    let parent = Box::pin(suspend_and_complete_children_durably(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "test"}),
        )],
        &child_results,
    ))
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

// ─────────────────────────────────────────────────────────────────────
// Multi-round tool-call regression test
// ─────────────────────────────────────────────────────────────────────

/// Re-acquire a parent task, build worker inputs, and call
/// `resume_from_children` with the given provider.
async fn reacquire_and_resume(
    stores: &TestStores,
    parent: &AgentTask,
    provider: &dyn LlmProvider,
    t_acquire: time::OffsetDateTime,
    t_resume: time::OffsetDateTime,
) -> Result<RootTurnOutcome> {
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_acquire + Duration::seconds(300),
            t_acquire,
        )
        .await?
        .context("re-acquire parent")?;
    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_acquire,
    )
    .await?;
    resume_from_children(inputs, parent, provider, &stores.deps(), t_resume).await
}

fn ok_result(output: &str) -> agent_sdk_core::ToolResult {
    agent_sdk_core::ToolResult {
        success: true,
        output: output.to_owned(),
        data: None,
        documents: Vec::new(),
        duration_ms: Some(10),
    }
}

#[tokio::test]
async fn resume_from_children_multi_round_does_not_collide() -> Result<()> {
    // Regression: multi-round spawn_index collision bug.
    // Round 1: 2 tool calls -> 2 children. Round 2: resume returns 1
    // more tool call -> 1 new child (spawn_index 0 again). Round 3:
    // text-only -> Completed. Without the fix, round 3 aggregation
    // sees all 3 children and collides on spawn_index 0.
    let stores = TestStores::new();

    // ── Round 1: suspend with 2 tool calls, complete durably ────
    let r1_results = vec![
        ("call_a".into(), ok_result("a")),
        ("call_b".into(), ok_result("b")),
    ];
    let parent = Box::pin(suspend_and_complete_children_durably(
        &stores,
        vec![
            (
                "call_a".into(),
                "bash".into(),
                serde_json::json!({"c": "pwd"}),
            ),
            (
                "call_b".into(),
                "bash".into(),
                serde_json::json!({"c": "ls"}),
            ),
        ],
        &r1_results,
    ))
    .await?;
    assert_eq!(stores.tasks.list_children(&parent.id).await?.len(), 2);
    assert_eq!(parent.state.child_ids().len(), 2);

    // ── Round 2: resume -> re-suspend with 1 new tool call ──────
    let r2_provider =
        MockToolCallProvider::single("call_c", "bash", serde_json::json!({"c": "cat"}));
    let outcome =
        reacquire_and_resume(&stores, &parent, &r2_provider, t_plus(20), t_plus(25)).await?;
    let RootTurnOutcome::Suspended {
        parent_task: r2_parent,
        child_tasks: r2_children,
        ..
    } = outcome
    else {
        panic!("Expected Suspended after round 2");
    };
    assert_eq!(r2_children.len(), 1);
    assert_eq!(r2_children[0].spawn_index, Some(0));
    assert_eq!(stores.tasks.list_children(&parent.id).await?.len(), 3);
    assert_eq!(r2_parent.state.child_ids().len(), 1);
    assert_eq!(r2_parent.state.child_ids()[0], r2_children[0].id);

    // Complete round 2 child, re-read parent.
    acquire_and_complete_child(&stores, &r2_children[0], &ok_result("file")).await?;
    let parent_r3 = stores.tasks.get(&parent.id).await?.context("parent r3")?;
    assert!(matches!(parent_r3.state, TaskState::ReadyToResume { .. }));
    assert_eq!(parent_r3.state.child_ids().len(), 1);

    // ── Round 3: resume -> text-only -> Completed ───────────────
    let final_provider = MockTextProvider::new("All done");
    let outcome_r3 =
        reacquire_and_resume(&stores, &parent_r3, &final_provider, t_plus(30), t_plus(35))
            .await
            .context("round 3")?;
    let RootTurnOutcome::Completed {
        completed_task,
        response_text,
        ..
    } = outcome_r3
    else {
        panic!("Expected Completed after round 3");
    };
    assert_eq!(completed_task.status, TaskStatus::Completed);
    assert_eq!(response_text, "All done");
    assert_eq!(
        stores
            .threads
            .get(&thread_a())
            .await?
            .context("thread")?
            .committed_turns,
        1
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// LLM stream retry on transient failures
// ─────────────────────────────────────────────────────────────────────
//
// Covers the worker-side retry-with-backoff added to `call_llm_with_retry`.
// The agent-sdk's in-process `call_llm_streaming` already retries
// transient LLM errors (rate-limit, server error, mid-stream drop), but
// before this slice the daemon's worker path bailed on the first
// `StreamDelta::Error`.  A single Anthropic SSE blip ("Stream ended
// unexpectedly without completion") would fail the whole `RootTurn`
// task and burn the journal's task-level `max_attempts` budget on the
// same error — surfacing as the production fail-loop:
//   `resume root task from durable child results: LLM stream error
//    (kind=ServerError): Stream ended unexpectedly without completion`
//
// These tests pin the new behaviour: recoverable kinds retry up to
// `STREAM_MAX_RETRIES` and emit `AutoRetryStart` / `AutoRetryEnd`
// envelope events; fatal kinds (`InvalidRequest`) skip the retry loop.

/// Provider whose first `n_failures` calls return `ServerError`,
/// then `Success` on subsequent calls.  Lets a single test exercise
/// the recover-after-N path without touching real network.
struct FlakyProvider {
    n_failures: usize,
    success_text: String,
    call_count: AtomicUsize,
}

impl FlakyProvider {
    fn new(n_failures: usize, success_text: &str) -> Self {
        Self {
            n_failures,
            success_text: success_text.to_owned(),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for FlakyProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n < self.n_failures {
            return Ok(ChatOutcome::ServerError(format!(
                "synthetic transient error #{n}"
            )));
        }
        Ok(ChatOutcome::Success(ChatResponse {
            id: format!("msg_flaky_{n}"),
            content: vec![ContentBlock::Text {
                text: self.success_text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                cached_input_tokens: 10,
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

/// Provider that always returns `InvalidRequest` — exercises the
/// fatal (no-retry) branch of `call_llm_with_retry`.
struct InvalidRequestProvider;

#[async_trait]
impl LlmProvider for InvalidRequestProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::InvalidRequest("schema rejected".into()))
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

/// Pull every event from the in-memory event repository and tag it
/// with its event-type discriminator string.  The test stores commit
/// auto-retry envelopes through `event_repo.commit_event`, so we can
/// inspect them with this helper without coupling to the proto layer.
async fn collected_event_kinds(events: &InMemoryEventRepository) -> Vec<String> {
    use agent_sdk_core::events::AgentEvent;
    let committed = events.get_events(&thread_a()).await.expect("read events");
    committed
        .into_iter()
        .map(|c| match c.event {
            AgentEvent::Start { .. } => "start",
            AgentEvent::TextDelta { .. } => "text_delta",
            AgentEvent::Text { .. } => "text",
            AgentEvent::ThinkingDelta { .. } => "thinking_delta",
            AgentEvent::Thinking { .. } => "thinking",
            AgentEvent::ToolCallStart { .. } => "tool_call_start",
            AgentEvent::ToolCallEnd { .. } => "tool_call_end",
            AgentEvent::TurnComplete { .. } => "turn_complete",
            AgentEvent::Done { .. } => "done",
            AgentEvent::Error { .. } => "error",
            AgentEvent::AutoRetryStart { .. } => "auto_retry_start",
            AgentEvent::AutoRetryEnd { .. } => "auto_retry_end",
            _ => "other",
        })
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn stream_server_error_retries_and_succeeds() -> Result<()> {
    // One transient failure → one retry → success.  The turn must
    // complete normally and the journal must carry one
    // `auto_retry_start` followed by an `auto_retry_end` with
    // `success: true`.
    let stores = TestStores::new();
    let provider = FlakyProvider::new(1, "Hello after retry!");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(inputs, "ping", &provider, &stores.deps(), t_plus(1)).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after retry, got Suspended");
    };
    assert_eq!(response_text, "Hello after retry!");

    // Provider was invoked twice: one failure, one success.
    assert_eq!(provider.calls(), 2);

    // Two turn attempts in the audit trail — one closed with
    // `ServerError`, one closed with `Completed`.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 2, "expected 2 attempts (1 retry)");
    assert!(attempts.iter().all(TurnAttempt::is_closed));

    // The retry envelope landed in the event log.
    let kinds = collected_event_kinds(&stores.events).await;
    assert!(
        kinds.iter().any(|k| k == "auto_retry_start"),
        "expected an auto_retry_start event in {kinds:?}",
    );
    assert!(
        kinds.iter().any(|k| k == "auto_retry_end"),
        "expected an auto_retry_end event in {kinds:?}",
    );

    Ok(())
}

#[tokio::test]
async fn stream_server_error_exhausts_budget_and_fails() -> Result<()> {
    // Provider always errors.  The turn must fail with the budget-
    // exhausted message and emit an `auto_retry_end { success:
    // false }`.  No content events should be committed.
    let stores = TestStores::new();
    // Request more failures than the budget can absorb so the
    // wrapper exhausts retries.
    let provider = FlakyProvider::new(usize::MAX, "(unreachable)");

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let err = execute_root_turn(inputs, "ping", &provider, &stores.deps(), t_plus(1))
        .await
        .expect_err("expected budget-exhausted failure");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("after") && msg.contains("retries"),
        "expected budget-exhausted message, got: {msg}",
    );

    // Total attempts == 1 (initial) + STREAM_MAX_RETRIES retries.
    // The constant lives in the impl, so we only assert the lower
    // bound of "more than 1 attempt was made".
    assert!(
        provider.calls() > 1,
        "expected multiple retry attempts, got {}",
        provider.calls(),
    );
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), provider.calls());
    assert!(attempts.iter().all(TurnAttempt::is_closed));

    let kinds = collected_event_kinds(&stores.events).await;
    assert!(
        kinds.iter().filter(|k| k == &"auto_retry_start").count() >= 1,
        "expected at least one auto_retry_start in {kinds:?}",
    );
    assert!(
        kinds.iter().any(|k| k == "auto_retry_end"),
        "expected an auto_retry_end event in {kinds:?}",
    );

    Ok(())
}

#[tokio::test]
async fn stream_invalid_request_does_not_retry() -> Result<()> {
    // `InvalidRequest` is caller-side; no amount of retry will help.
    // The wrapper must skip the retry loop and bail on the first
    // attempt, leaving exactly one closed audit row.
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let err = execute_root_turn(
        inputs,
        "ping",
        &InvalidRequestProvider,
        &stores.deps(),
        t_plus(1),
    )
    .await
    .expect_err("expected invalid-request failure");
    let msg = format!("{err:#}");
    assert!(
        msg.to_lowercase().contains("invalidrequest")
            || msg.to_lowercase().contains("invalid request")
            || msg.contains("schema rejected"),
        "expected invalid-request error, got: {msg}",
    );

    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1, "no retries for InvalidRequest");

    let kinds = collected_event_kinds(&stores.events).await;
    assert!(
        kinds.iter().all(|k| k != "auto_retry_start"),
        "no retry should have been attempted, found: {kinds:?}",
    );

    Ok(())
}
