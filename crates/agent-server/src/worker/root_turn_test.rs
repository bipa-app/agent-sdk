//! Integration tests for Phase 4.3 text-only root turn execution.
//!
//! These tests verify the end-to-end flow from task acquisition through
//! completed-turn commit using a mock LLM provider that returns text-only
//! responses.

use super::root_turn::{RootTurnDeps, execute_root_turn};
use crate::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, TaskStatus, WorkerId};
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::ThreadId;
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use time::Duration;

// ─────────────────────────────────────────────────────────────────────
// Mock LLM provider
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

    fn model(&self) -> &str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

// ────────���────────────────────────────────────────────────────────────
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
// Tests
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

    // Provider was called exactly once.
    assert_eq!(provider.calls(), 1);

    // Response text captured.
    assert_eq!(outcome.response_text, "Hello! I'm a helpful assistant.");

    // Task completed.
    assert_eq!(outcome.completed_task.status, TaskStatus::Completed);
    assert!(outcome.completed_task.completed_at.is_some());

    // Turn attempt closed.
    assert!(outcome.commit.closed_attempt.is_closed());

    // Thread aggregate advanced.
    assert_eq!(outcome.commit.thread.committed_turns, 1);
    assert_eq!(outcome.commit.thread.total_usage.input_tokens, 100);
    assert_eq!(outcome.commit.thread.total_usage.output_tokens, 50);

    // Checkpoint created at turn 1.
    assert_eq!(outcome.commit.checkpoint.turn_number, 1);
    assert_eq!(outcome.commit.checkpoint.thread_id, thread_a());
    // Checkpoint has 2 messages (user + assistant).
    assert_eq!(outcome.commit.checkpoint.messages.len(), 2);

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

    // Verify turn attempt lifecycle.
    let attempt = &outcome.commit.closed_attempt;
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
async fn tool_call_response_is_rejected() -> Result<()> {
    // A provider that returns tool calls should fail in text-only mode.
    struct ToolCallProvider;

    #[async_trait]
    impl LlmProvider for ToolCallProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_tool".into(),
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                    thought_signature: None,
                }],
                model: "mock-model".into(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                },
            }))
        }

        fn model(&self) -> &str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let err = execute_root_turn(inputs, "test", &ToolCallProvider, &stores.deps(), t_plus(1))
        .await
        .unwrap_err();

    assert!(
        err.to_string().contains("tool call"),
        "expected tool call error, got: {err}",
    );

    // No durable turn data committed (thread exists from recovery
    // but has zero committed turns).
    let thread = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread from recovery")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());

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

        fn model(&self) -> &str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
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

    Ok(())
}
