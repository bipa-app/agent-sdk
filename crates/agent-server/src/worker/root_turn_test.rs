//! Integration tests for root turn execution.
//!
//! Covers the Phase 4.3 text-only commit path, the Phase 4.4
//! tool-boundary suspension path, the Phase 4.5 resume from
//! completed child tool results, and the Phase 5.4 durable
//! child-outcome aggregation and resume-from-children path.

use super::root_turn::{
    PartialCancelCommit, RootTurnDeps, RootTurnOutcome, aggregate_child_outcomes, cancel_root_turn,
    commit_partial_turn_on_cancel, derive_reattach_tool_use_id, execute_root_turn, fail_root_turn,
    is_root_turn_cancelled, provider_valid_split, resume_for_steering, resume_from_children,
    resume_root_turn, revert_steering_wake, settle_attempt_after_lost_ownership,
};
use crate::journal::checkpoint::CheckpointKind;
use std::sync::Arc;

use crate::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use crate::journal::committed_event::CommittedEvent;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::outbox::{InMemoryOutboxStore, OutboxStore as _};
use crate::journal::outbox_message::OutboxMessageKind;
use crate::journal::store::{AgentTaskStore, CancellationMarkerSink, InMemoryAgentTaskStore};
use crate::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskKind, TaskStatus,
    WorkerId,
};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId, TurnAttemptOutcome,
};
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Role, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{
    AgentContinuation, AgentState, ContinuationEnvelope, ThreadId, TokenUsage,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_providers::streaming::{StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_tools::stores::MessageStore;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use time::Duration;
use tokio_util::sync::CancellationToken;

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
    stop_reason: StopReason,
    call_count: AtomicUsize,
}

impl MockToolCallProvider {
    fn new(tool_calls: Vec<(String, String, serde_json::Value)>) -> Self {
        Self {
            tool_calls,
            stop_reason: StopReason::ToolUse,
            call_count: AtomicUsize::new(0),
        }
    }

    fn single(id: &str, name: &str, input: serde_json::Value) -> Self {
        Self::new(vec![(id.into(), name.into(), input)])
    }

    fn with_stop_reason(mut self, stop_reason: StopReason) -> Self {
        self.stop_reason = stop_reason;
        self
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
            stop_reason: Some(self.stop_reason),
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
            tier: agent_sdk_foundation::ToolTier::Observe,
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
    child_results: &[(String, agent_sdk_foundation::ToolResult)],
) -> agent_sdk_foundation::llm::Message {
    let blocks = child_results
        .iter()
        .map(|(tool_use_id, result)| ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: result.output.clone(),
            is_error: if result.success { None } else { Some(true) },
        })
        .collect();
    agent_sdk_foundation::llm::Message::user_with_content(blocks)
}

fn assert_tool_results_follow_tool_use(
    messages: &[agent_sdk_foundation::llm::Message],
    tool_use_id: &str,
) {
    let tool_use_index = messages.iter().position(|message| {
        matches!(
            &message.content,
            agent_sdk_foundation::llm::Content::Blocks(blocks)
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
        agent_sdk_foundation::llm::Content::Blocks(blocks)
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

#[derive(Clone)]
struct TestStores {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
    outbox: InMemoryOutboxStore,
    event_notifier: Arc<EventNotifier>,
}

impl TestStores {
    fn new() -> Self {
        // Mirror the composed in-memory backend: `cancel_tree` commits
        // its terminal `Cancelled` markers through the shared event /
        // outbox / thread stores (issue #354).
        let threads = InMemoryThreadStore::new();
        let events = InMemoryEventRepository::new();
        let outbox = InMemoryOutboxStore::new();
        let tasks =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(events.clone()),
                outbox_store: Arc::new(outbox.clone()),
                thread_store: Arc::new(threads.clone()),
            });
        Self {
            tasks,
            threads,
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
            events,
            outbox,
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
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
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

    let state: agent_sdk_foundation::AgentState =
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
    // Post-streaming-refactor invariant: the worker drives `chat_stream`
    // and synthesizes the response from the `StreamAccumulator`, which
    // never reaches a provider's response-id space — so the closed
    // attempt records no `response_id` even though the underlying mock
    // returns one for the non-streaming `chat()` path.
    assert_eq!(attempt.response_id, None);
    assert_eq!(attempt.response_model, Some("mock-model".into()));
    // Usage is carried through the streamed `Usage` delta unchanged.
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
    // A caller-side `InvalidRequest` is a *fatal* stream error: the
    // retry wrapper skips the backoff loop entirely.  This is the
    // cleanest "an LLM error propagates and no durable turn is
    // committed" probe — the recoverable `ServerError` path (which
    // retries to budget exhaustion) is covered separately by
    // `stream_server_error_exhausts_budget_and_fails`.
    struct ErrorProvider;

    #[async_trait]
    impl LlmProvider for ErrorProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::InvalidRequest("internal error".into()))
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
        err_msg.contains("internal error"),
        "expected the LLM error to propagate, got: {err_msg}",
    );

    // No durable turn data committed.
    let thread = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread from recovery")?;
    assert_eq!(thread.committed_turns, 0);

    // Exactly one turn attempt, opened and closed with the error
    // outcome — the fatal kind does not retry.
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
async fn terminal_stop_reason_with_tool_blocks_commits_without_child_dispatch() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockToolCallProvider::single(
        "call_terminal",
        "bash",
        serde_json::json!({"command": "must-not-run"}),
    )
    .with_stop_reason(StopReason::EndTurn);

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

    let outcome = execute_root_turn(
        inputs,
        "finish without tools",
        &provider,
        &stores.deps(),
        t_plus(5),
    )
    .await?;

    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        panic!("terminal stop reasons must not suspend for tool blocks");
    };
    assert_eq!(provider.calls(), 1);
    assert_eq!(completed_task.status, TaskStatus::Completed);
    assert!(
        stores.tasks.list_children(&task_id).await?.is_empty(),
        "a terminal response must not create tool children"
    );

    let history = stores.messages.get_history(&thread_a()).await?;
    assert!(
        !history.iter().any(|message| matches!(
            &message.content,
            agent_sdk_foundation::llm::Content::Blocks(blocks)
                if blocks.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        )),
        "terminal tool blocks must not be persisted without results: {history:?}"
    );
    assert!(
        !agent_sdk_foundation::llm::has_unbalanced_tool_use(&history),
        "terminal response history must remain balanced: {history:?}"
    );

    Ok(())
}

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
    // Post-streaming-refactor: synthesized responses carry no provider
    // response-id, so the closed attempt records `None`.
    assert_eq!(attempts[0].response_id, None);
    assert_eq!(attempts[0].response_model, Some("mock-model".into()));
    assert_eq!(attempts[0].input_tokens, Some(120));
    assert_eq!(attempts[0].output_tokens, Some(60));

    Ok(())
}

#[tokio::test]
async fn duplicate_suspension_loser_bills_its_own_attempt_row() -> Result<()> {
    // The duplicate-suspension race: two workers both run the same fresh
    // tool-dispatch turn. Worker A wins and suspends the task to
    // WaitingOnChildren; worker B's LLM call has already completed (and been
    // billed) when it reaches the idempotency guard, which bails. B's attempt
    // row must still record the tokens its call was billed for — not zero.
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_1", "bash", serde_json::json!({"command": "ls"}));

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();

    // Both workers build their inputs while the task is still Running, so both
    // see a fresh tool-dispatch turn — the state the race requires.
    let inputs_winner = build_root_worker_inputs(
        sample_bootstrap_with_tools(task.clone()),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;
    let inputs_loser = build_root_worker_inputs(
        sample_bootstrap_with_tools(task),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Worker A wins: suspends the task and closes its own attempt with 120/60.
    let winner = execute_root_turn(
        inputs_winner,
        "List files",
        &provider,
        &stores.deps(),
        t_plus(5),
    )
    .await?;
    assert!(
        matches!(winner, RootTurnOutcome::Suspended { .. }),
        "the winning worker must suspend",
    );

    // Worker B loses: its stream completes (billed), then the guard sees
    // WaitingOnChildren and bails.
    let loser_error = execute_root_turn(
        inputs_loser,
        "List files",
        &provider,
        &stores.deps(),
        t_plus(6),
    )
    .await
    .expect_err("the duplicate suspension must bail");
    assert!(
        loser_error.to_string().contains("duplicate suspension"),
        "expected the idempotency-guard bail, got {loser_error:#}",
    );

    let mut attempts = stores.attempts.list_by_task(&task_id).await?;
    attempts.sort_by_key(|a| a.opened_at);
    assert_eq!(attempts.len(), 2, "one winner attempt + one loser attempt");

    // Winner row: unchanged, billed its own call.
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[0].input_tokens, Some(120));
    assert_eq!(attempts[0].output_tokens, Some(60));

    // Loser row: closed by the guard, but still billed the tokens its completed
    // call was charged for — the fix (was zero before).
    assert_eq!(attempts[1].outcome, Some(TurnAttemptOutcome::Cancelled));
    assert_eq!(
        attempts[1].input_tokens,
        Some(120),
        "the losing worker's completed call was billed; its row must not be zero",
    );
    assert_eq!(attempts[1].output_tokens, Some(60));

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
        agent_sdk_foundation::ToolTier::Observe
    );
    assert_eq!(payload.pending_tool_calls[0].display_name, "Bash");

    assert_eq!(payload.pending_tool_calls[1].id, "call_b");
    assert_eq!(payload.pending_tool_calls[1].name, "read_file");
    // "read_file" is NOT in the definition — falls back to Confirm.
    assert_eq!(
        payload.pending_tool_calls[1].tier,
        agent_sdk_foundation::ToolTier::Confirm
    );

    // No completed results yet (tools haven't run).
    assert!(payload.completed_results.is_empty());

    // LLM metadata preserved.  Post-streaming-refactor the synthesized
    // response carries no provider response-id, so the continuation's
    // `response_id` is `None`; the stop reason still flows through the
    // streamed `Done` delta.
    assert_eq!(payload.response_id, None);
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
    _child_results: &[(String, agent_sdk_foundation::ToolResult)],
) -> Result<(
    AgentTask,
    agent_sdk_foundation::AgentContinuation,
    Vec<agent_sdk_foundation::llm::Message>,
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
        agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
        agent_sdk_foundation::ToolResult {
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
    Box::pin(resume_root_turn(
        inputs,
        continuation.clone(),
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
    .await?;

    // Verify checkpoint agent state.
    let checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint")?;

    let state: agent_sdk_foundation::AgentState =
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
        agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
        agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
            agent_sdk_foundation::ToolResult {
                success: true,
                output: "/home/user".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            },
        ),
        (
            "call_b".to_owned(),
            agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
    assert_eq!(messages[0].role, agent_sdk_foundation::llm::Role::User);

    // Second message is the assistant response with tool-use blocks.
    assert_eq!(messages[1].role, agent_sdk_foundation::llm::Role::Assistant);
    let has_tool_use = match &messages[1].content {
        agent_sdk_foundation::llm::Content::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, agent_sdk_foundation::llm::ContentBlock::ToolUse { .. })),
        agent_sdk_foundation::llm::Content::Text(_) => false,
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
        agent_sdk_foundation::ToolResult {
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
        agent_sdk_foundation::ToolResult {
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
    let err = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    ))
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
        agent_sdk_foundation::ToolResult {
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
    let err = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages.clone(),
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    ))
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
    let msgs_match = |a: &[agent_sdk_foundation::llm::Message],
                      b: &[agent_sdk_foundation::llm::Message]|
     -> bool {
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

/// Issue #299 (round 3, addendum C): cancellation landing after
/// `open_attempt` but before the stream owns the attempt — e.g. the
/// host's subagent deadline firing between attempt open and the first
/// poll — must not leak a permanently OPEN attempt row. The host's
/// timeout fail deliberately leaves live-worker attempts open for the
/// worker's own cancel paths to close, and the task is already
/// terminal on the host side so the normal commit-path close never
/// runs; the pre-stream cancel bail therefore closes its own attempt
/// (`Cancelled`, zero usage — nothing streamed).
#[tokio::test]
async fn pre_stream_cancel_closes_the_open_attempt() -> Result<()> {
    let stores = TestStores::new();
    let provider = MockTextProvider::new("never reached");

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

    // The token is already tripped when the turn reaches the pre-call
    // cancel check — after `open_attempt`, before any streaming.
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let mut deps = stores.deps();
    deps.cancel = Some(&cancelled);

    let result = execute_root_turn(inputs, "doomed prompt", &provider, &deps, t_plus(5)).await;
    assert!(result.is_err(), "a pre-cancelled turn must not complete");

    let attempts = stores.attempts.list_by_task(&task_id).await?;
    let [attempt] = attempts.as_slice() else {
        bail!("expected exactly one attempt, got {}", attempts.len());
    };
    assert!(
        attempt.is_closed(),
        "the pre-stream cancel bail must close its own open attempt",
    );
    assert_eq!(
        attempt.outcome,
        Some(TurnAttemptOutcome::Cancelled),
        "nothing streamed, so the honest close is Cancelled",
    );
    assert_eq!(
        attempt.output_tokens,
        Some(0),
        "a never-streamed attempt must close with zero usage",
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

    // The completed prefix of the cancelled turn is committed before the
    // subtree is torn down. The suspension draft was
    // `[user, assistant+tool_use]`; its largest provider-valid prefix is
    // the bare user prompt, which commits as turn 1. The trailing
    // (unresulted) assistant tool_use is dropped from the commit and
    // re-seeded as the draft so the next turn's backfill can close it.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);

    let committed = stores.messages.get_history(&thread_a()).await?;
    assert_eq!(committed.len(), 1, "only the provider-valid prefix commits");
    assert_eq!(
        committed[0].role,
        agent_sdk_foundation::llm::Role::User,
        "the committed prefix is the user prompt",
    );

    // Exactly one checkpoint at turn 1.
    assert_eq!(
        stores.checkpoints.list_by_thread(&thread_a()).await?.len(),
        1,
    );

    // The dropped trailing assistant tool_use survives as the re-seeded
    // draft.
    let projection = stores
        .messages
        .get(&thread_a())
        .await?
        .context("projection")?;
    assert!(projection.has_draft());
    assert_eq!(projection.draft_messages.len(), 1);

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
        agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
        agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &resume_provider,
        &stores.deps(),
        t_plus(25),
    ))
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
        agent_sdk_foundation::ToolResult {
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
    let err = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &ErrorProvider,
        &stores.deps(),
        t_plus(25),
    ))
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
    child_results: &[(String, agent_sdk_foundation::ToolResult)],
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
            agent_sdk_foundation::ToolResult {
                success: true,
                output: "result_a".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: Some(10),
            },
        ),
        (
            "call_b".to_owned(),
            agent_sdk_foundation::ToolResult {
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
    let success_result = agent_sdk_foundation::ToolResult {
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
        agent_sdk_foundation::ToolResult {
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
    result: &agent_sdk_foundation::ToolResult,
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
        &agent_sdk_foundation::ToolResult::success("/home"),
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
        agent_sdk_foundation::ToolResult {
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

    let original_result = agent_sdk_foundation::ToolResult {
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
    let recovered: agent_sdk_foundation::ToolResult =
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

fn ok_result(output: &str) -> agent_sdk_foundation::ToolResult {
    agent_sdk_foundation::ToolResult {
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
    let outcome = Box::pin(reacquire_and_resume(
        &stores,
        &parent,
        &r2_provider,
        t_plus(20),
        t_plus(25),
    ))
    .await?;
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
    let outcome_r3 = Box::pin(reacquire_and_resume(
        &stores,
        &parent_r3,
        &final_provider,
        t_plus(30),
        t_plus(35),
    ))
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

/// Provider that goes offline for a scripted number of reachability probes,
/// then recovers. The first `chat_stream` call fails with a pre-connect
/// connectivity error; the worker then holds the attempt open and polls
/// `probe_connectivity`, which reports unreachable `offline_probes` times —
/// each unreachable answer forgiving the reachable-death circuit breaker —
/// before the endpoint comes back. Later calls optionally fail with server
/// errors first, proving the transient budget survives the wait untouched.
struct OfflineThenSuccessProvider {
    offline_probes: usize,
    server_failures_after_online: usize,
    call_count: AtomicUsize,
    probe_count: AtomicUsize,
}

impl OfflineThenSuccessProvider {
    fn new(offline_probes: usize) -> Self {
        Self {
            offline_probes,
            server_failures_after_online: 0,
            call_count: AtomicUsize::new(0),
            probe_count: AtomicUsize::new(0),
        }
    }

    const fn with_server_failures(mut self, failures: usize) -> Self {
        self.server_failures_after_online = failures;
        self
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for OfflineThenSuccessProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        bail!("chat is not used by this streaming provider")
    }

    async fn probe_connectivity(&self) -> bool {
        let probe = self.probe_count.fetch_add(1, Ordering::SeqCst);
        probe >= self.offline_probes
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let call = self.call_count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async_stream::stream! {
            if call == 0 {
                yield Ok(StreamDelta::Error {
                    message: "network is unreachable".to_owned(),
                    kind: StreamErrorKind::Connectivity,
                });
                return;
            }
            if call <= self.server_failures_after_online {
                yield Ok(StreamDelta::Error {
                    message: "provider 503".to_owned(),
                    kind: StreamErrorKind::ServerError,
                });
                return;
            }
            yield Ok(StreamDelta::TextDelta {
                delta: "continued after reconnect".to_owned(),
                block_index: 0,
            });
            yield Ok(StreamDelta::Usage(Usage {
                input_tokens: 10,
                output_tokens: 3,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            }));
            yield Ok(StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
            });
        })
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

/// Provider whose dispatched streams always die mid-transport while its
/// probe keeps reporting reachable — the broken-path pathology (a proxy or
/// load balancer killing every response) that the reachable-death circuit
/// breaker exists to bound.
struct ReachableButDyingProvider {
    call_count: AtomicUsize,
}

impl ReachableButDyingProvider {
    const fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for ReachableButDyingProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        bail!("chat is not used by this streaming provider")
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async_stream::stream! {
            yield Ok(StreamDelta::Error {
                message: "connection reset by peer".to_owned(),
                kind: StreamErrorKind::ConnectionLost,
            });
        })
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
    use agent_sdk_foundation::events::AgentEvent;
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

#[tokio::test(start_paused = true)]
async fn connectivity_wait_outlives_normal_retry_budget() -> Result<()> {
    let stores = TestStores::new();
    let provider = OfflineThenSuccessProvider::new(6);
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
        bail!("expected completion after connectivity recovered")
    };
    assert_eq!(response_text, "continued after reconnect");
    assert_eq!(
        provider.calls(),
        2,
        "an offline wait must probe for free, not re-dispatch billable calls"
    );
    assert_eq!(
        stores.attempts.list_by_task(&task_id).await?.len(),
        1,
        "offline probes must not manufacture zero-usage audit attempts"
    );
    let events = stores.events.get_events(&thread_a()).await?;
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event.event,
                agent_sdk_foundation::events::AgentEvent::AutoRetryStart { .. }
            ))
            .count(),
        1,
        "an unbounded connectivity wait must use one durable retry envelope"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn connectivity_wait_does_not_consume_server_retry_budget() -> Result<()> {
    let stores = TestStores::new();
    let provider = OfflineThenSuccessProvider::new(6).with_server_failures(1);
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

    let outcome = execute_root_turn(inputs, "ping", &provider, &stores.deps(), t_plus(1)).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        bail!("expected completion after connectivity and server retries")
    };
    assert_eq!(response_text, "continued after reconnect");
    assert_eq!(provider.calls(), 3);

    let events = stores.events.get_events(&thread_a()).await?;
    let retry_attempts: Vec<u32> = events
        .iter()
        .filter_map(|committed| match committed.event {
            agent_sdk_foundation::events::AgentEvent::AutoRetryStart { attempt, .. } => {
                Some(attempt)
            }
            _ => None,
        })
        .collect();
    assert_eq!(retry_attempts, vec![1, 2]);
    assert!(events.iter().any(|committed| matches!(
        committed.event,
        agent_sdk_foundation::events::AgentEvent::AutoRetryEnd {
            attempt: 2,
            success: true,
            ..
        }
    )));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn reachable_connection_deaths_trip_the_circuit_breaker() -> Result<()> {
    // Probes keep reporting reachable, yet every dispatched stream dies in
    // transit — a broken path, not an outage. The turn must fail after the
    // reachable-death bound instead of billing an attempt per backoff
    // forever, and every audit row must be settled.
    let stores = TestStores::new();
    let provider = ReachableButDyingProvider::new();
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

    let result = execute_root_turn(inputs, "ping", &provider, &stores.deps(), t_plus(1)).await;
    let Err(error) = result else {
        bail!("a reachable provider whose streams keep dying must fail the turn")
    };
    assert!(
        format!("{error:#}").contains("stayed reachable"),
        "the failure must name the circuit breaker: {error:#}"
    );
    assert_eq!(
        provider.calls(),
        4,
        "initial dispatch plus the three forgivable reachable deaths"
    );
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 4, "one audit row per dispatched call");
    assert!(
        attempts.iter().all(|attempt| !attempt.is_open()),
        "every dispatched attempt must be settled"
    );
    let events = stores.events.get_events(&thread_a()).await?;
    assert!(events.iter().any(|committed| matches!(
        committed.event,
        agent_sdk_foundation::events::AgentEvent::AutoRetryEnd { success: false, .. }
    )));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn connectivity_wait_stops_promptly_on_cancel() -> Result<()> {
    let stores = TestStores::new();
    let provider = OfflineThenSuccessProvider::new(usize::MAX);
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
    let cancel = CancellationToken::new();
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let run = execute_root_turn(inputs, "ping", &provider, &deps, t_plus(1));
    let trip_cancel = async {
        while provider.calls() == 0 {
            tokio::task::yield_now().await;
        }
        cancel.cancel();
    };
    let (result, ()) = tokio::join!(run, trip_cancel);
    let Err(error) = result else {
        bail!("cancelled connectivity wait unexpectedly completed")
    };
    assert!(is_root_turn_cancelled(&error));

    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].outcome,
        Some(TurnAttemptOutcome::Cancelled),
        "cancellation must close the held offline attempt"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn connectivity_wait_registry_tracks_the_park_and_cancel_clears_it() -> Result<()> {
    // While the worker is parked on reachability probes, the registry
    // must expose the wait — keyed by thread, carrying the failure
    // message and the journal sequence of the streak's AutoRetryStart —
    // and cancellation must retract the entry through the guard.
    let stores = TestStores::new();
    let provider = OfflineThenSuccessProvider::new(usize::MAX);
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
    let cancel = CancellationToken::new();
    let registry = super::connectivity::ConnectivityWaitRegistry::new();
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);
    deps.connectivity_waits = Some(&registry);

    let run = execute_root_turn(inputs, "ping", &provider, &deps, t_plus(1));
    let observe_then_cancel = async {
        while !registry.is_waiting(&thread_a()) {
            tokio::task::yield_now().await;
        }
        let wait = registry
            .get(&thread_a())
            .context("registry entry must exist while parked")?;
        assert_eq!(wait.error_message, "network is unreachable");
        let events = stores.events.get_events(&thread_a()).await?;
        let retry_start_sequence = events
            .iter()
            .find(|committed| {
                matches!(
                    committed.event,
                    agent_sdk_foundation::events::AgentEvent::AutoRetryStart { .. }
                )
            })
            .map(|committed| committed.sequence)
            .context("streak AutoRetryStart must be committed before the park")?;
        assert_eq!(
            wait.sequence, retry_start_sequence,
            "the registry snapshot must be orderable against the journal"
        );
        cancel.cancel();
        anyhow::Ok(())
    };
    let (result, observed) = tokio::join!(run, observe_then_cancel);
    observed?;
    let Err(error) = result else {
        bail!("cancelled connectivity wait unexpectedly completed")
    };
    assert!(is_root_turn_cancelled(&error));
    assert!(
        !registry.is_waiting(&thread_a()),
        "the guard must retract the entry when cancellation unwinds the wait"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn connectivity_wait_registry_clears_on_recovery() -> Result<()> {
    // Connectivity returns after a few probes: the run completes and the
    // registry no longer lists the thread — recovery exits through the
    // same guard as cancellation.
    let stores = TestStores::new();
    let provider = OfflineThenSuccessProvider::new(3);
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
    let registry = super::connectivity::ConnectivityWaitRegistry::new();
    let mut deps = stores.deps();
    deps.connectivity_waits = Some(&registry);

    let run = execute_root_turn(inputs, "ping", &provider, &deps, t_plus(1));
    let observe = async {
        while !registry.is_waiting(&thread_a()) {
            tokio::task::yield_now().await;
        }
        anyhow::Ok(())
    };
    let (outcome, observed) = tokio::join!(run, observe);
    observed?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome? else {
        bail!("expected completion after connectivity recovered")
    };
    assert_eq!(response_text, "continued after reconnect");
    assert!(
        !registry.is_waiting(&thread_a()),
        "a recovered wait must leave no registry entry behind"
    );
    Ok(())
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

// ─────────────────────────────────────────────────────────────────────
// Per-class retry budgets — RateLimited outlives the ServerError budget
// ─────────────────────────────────────────────────────────────────────

/// One scripted transient failure for [`ScriptedProvider`].
enum ScriptedOutcome {
    RateLimited,
    ServerError,
}

/// Provider that plays a scripted sequence of transient outcomes
/// before succeeding — lets tests interleave `RateLimited` and
/// `ServerError` failures to pin the per-class retry budgets
/// (`RATE_LIMIT_MAX_RETRIES` vs `STREAM_MAX_RETRIES`).
struct ScriptedProvider {
    script: Vec<ScriptedOutcome>,
    success_text: String,
    call_count: AtomicUsize,
}

impl ScriptedProvider {
    fn new(script: Vec<ScriptedOutcome>, success_text: &str) -> Self {
        Self {
            script,
            success_text: success_text.to_owned(),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        match self.script.get(n) {
            Some(ScriptedOutcome::RateLimited) => Ok(ChatOutcome::RateLimited(None)),
            Some(ScriptedOutcome::ServerError) => Ok(ChatOutcome::ServerError(format!(
                "synthetic transient error #{n}"
            ))),
            None => Ok(ChatOutcome::Success(ChatResponse {
                id: format!("msg_scripted_{n}"),
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
            })),
        }
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

// `start_paused` — the rate-limit schedule sleeps for minutes of
// virtual time (2s → 120s per wait); tokio's auto-advancing test
// clock makes these instant without weakening the schedule under test.
#[tokio::test(start_paused = true)]
async fn rate_limited_outlives_the_server_error_budget() -> Result<()> {
    // Five consecutive rate-limited failures exceed the server-error
    // budget (3) but sit well inside the rate-limit budget (10). The
    // turn must ride out the window and complete — this is the
    // 2026-07-16 overload-incident regression pin: a 529 window used
    // to terminally fail the turn in under 5 seconds.
    let stores = TestStores::new();
    let script = (0..5).map(|_| ScriptedOutcome::RateLimited).collect();
    let provider = ScriptedProvider::new(script, "Survived the overload window!");

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
        panic!("expected Completed after riding out the rate-limit window, got Suspended");
    };
    assert_eq!(response_text, "Survived the overload window!");
    assert_eq!(provider.calls(), 6, "5 failures + 1 success");

    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 6);
    assert!(attempts.iter().all(TurnAttempt::is_closed));

    let kinds = collected_event_kinds(&stores.events).await;
    assert_eq!(
        kinds.iter().filter(|k| *k == "auto_retry_start").count(),
        5,
        "one auto_retry_start per rate-limited failure in {kinds:?}",
    );
    assert!(
        kinds.iter().any(|k| k == "auto_retry_end"),
        "expected an auto_retry_end event in {kinds:?}",
    );

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn rate_limited_exhausts_its_own_deeper_budget() -> Result<()> {
    // A provider that never stops rate-limiting must still fail
    // eventually — after the rate-limit budget (10), not the
    // server-error budget (3), and the terminal message must name the
    // class so operators can tell weather from breakage.
    let stores = TestStores::new();
    let script = (0..32).map(|_| ScriptedOutcome::RateLimited).collect();
    let provider = ScriptedProvider::new(script, "(unreachable)");

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
        .expect_err("expected rate-limit budget exhaustion");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("after 10 retries") && msg.contains("RateLimited"),
        "expected rate-limit budget-exhausted message, got: {msg}",
    );

    // 1 initial + 10 retries, every audit row closed.
    assert_eq!(provider.calls(), 11);
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 11);
    assert!(attempts.iter().all(TurnAttempt::is_closed));

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn rate_limit_and_server_error_budgets_are_independent() -> Result<()> {
    // Five rate-limited failures followed by two server errors: the
    // rate-limit streak must not have consumed the server budget (2 of
    // 3 spent), so the turn still completes. A shared counter would
    // have failed the turn on the first server error (6th failure > 3).
    let stores = TestStores::new();
    let script = vec![
        ScriptedOutcome::RateLimited,
        ScriptedOutcome::RateLimited,
        ScriptedOutcome::RateLimited,
        ScriptedOutcome::RateLimited,
        ScriptedOutcome::RateLimited,
        ScriptedOutcome::ServerError,
        ScriptedOutcome::ServerError,
    ];
    let provider = ScriptedProvider::new(script, "Both budgets intact!");

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

    let outcome = execute_root_turn(inputs, "ping", &provider, &stores.deps(), t_plus(1)).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed with independent budgets, got Suspended");
    };
    assert_eq!(response_text, "Both budgets intact!");
    assert_eq!(provider.calls(), 8, "7 scripted failures + 1 success");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Findings #3 / #11 / #12 — per-turn (`tools_fn`) tier drives the gate
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn pending_tool_tier_uses_per_turn_tools_fn_not_static_list() -> Result<()> {
    // The host gates the confirmation pause on the pending tool call's
    // tier. `tools_fn` can harden a tool's tier per caller (e.g.
    // `Confirm` for guests where the static list says `Observe`). That
    // stricter per-turn tier MUST win, or confirmation is silently
    // skipped.
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_a", "bash", serde_json::json!({"command": "rm -rf /"}));

    let definition = AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "test".into(),
        max_tokens: 1024,
        // Static list: bash is the permissive Observe tier.
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run a shell command".into(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Bash (static)".into(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }],
        thinking: ThinkingPolicy::default(),
        // Per-turn override (consulted because the task carries
        // caller_metadata): bash is hardened to Confirm.
        tools_fn: Some(Arc::new(|_ctx| {
            vec![Tool {
                name: "bash".into(),
                description: "Run a shell command".into(),
                input_schema: serde_json::json!({"type": "object"}),
                display_name: "Bash (hardened)".into(),
                tier: agent_sdk_foundation::ToolTier::Confirm,
            }]
        })),
        policy: RuntimePolicy::server_default(),
    };

    // Task with caller_metadata so `resolve_tools` invokes `tools_fn`.
    let mut task = AgentTask::new_root_turn(thread_a(), t0(), 3);
    task.caller_metadata = Some(serde_json::json!({"role": "guest"}));
    let task_id = task.id.clone();
    stores.tasks.submit_root_turn(task).await?;
    let acquired = stores
        .tasks
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("task should be acquirable")?;

    let bootstrap = WorkerBootstrapContext {
        task: acquired,
        definition,
        thread_id: thread_a(),
        task_id: task_id.clone(),
        worker_id: WorkerId::from_string("worker_test"),
        lease_id: LeaseId::from_string("lease_test"),
    };
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(inputs, "do it", &provider, &stores.deps(), t_plus(5)).await?;
    let RootTurnOutcome::Suspended { .. } = outcome else {
        panic!("expected Suspended on a tool-use turn");
    };

    let parent = stores
        .tasks
        .get(&task_id)
        .await?
        .context("parent task should exist")?;
    let continuation = match &parent.state {
        TaskState::WaitingOnChildren { continuation, .. } => continuation,
        other => panic!("expected WaitingOnChildren, got {other:?}"),
    };
    let tool_call = &continuation.payload.pending_tool_calls[0];
    assert_eq!(
        tool_call.tier,
        agent_sdk_foundation::ToolTier::Confirm,
        "the per-turn tools_fn Confirm tier must win over the static Observe tier",
    );
    assert_eq!(tool_call.display_name, "Bash (hardened)");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Findings #6 / #7 — cooperative cancellation
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cancelled_token_aborts_root_turn_without_committing() -> Result<()> {
    // A tripped cancellation token must stop the worker cooperatively
    // (no commit, no completed turn) rather than streaming + committing
    // a cancelled thread's turn and only failing at the final CAS.
    let stores = TestStores::new();
    let provider = MockTextProvider::new("should never be committed");

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

    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let result = execute_root_turn(inputs, "hi", &provider, &deps, t_plus(5)).await;
    let Err(error) = result else {
        panic!("cancelled turn must not return a committed outcome");
    };
    assert!(
        format!("{error:#}").to_lowercase().contains("cancel"),
        "expected a cancellation error, got: {error:#}",
    );

    // The turn never committed.
    let thread = stores
        .threads
        .get(&thread_a())
        .await?
        .context("thread should exist")?;
    assert_eq!(thread.committed_turns, 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Finding #10 — selector errors fail the turn (trait contract)
// ─────────────────────────────────────────────────────────────────────

struct FailingSpawnSelector;

#[async_trait]
impl super::subagent_spawn_selector::SubagentSpawnSelector for FailingSpawnSelector {
    async fn decide(
        &self,
        _parent_thread_id: &ThreadId,
        _tool_calls: &[agent_sdk_foundation::PendingToolCallInfo],
    ) -> Result<Vec<super::subagent_spawn_selector::SubagentSpawnDecision>> {
        anyhow::bail!("synthetic selector failure")
    }
}

#[tokio::test]
async fn selector_error_fails_the_turn() -> Result<()> {
    // The SubagentSpawnSelector contract reserves `Err` for genuinely
    // unrecoverable conditions; the worker must propagate it as the turn
    // error rather than silently rerouting to spawn_tool_children.
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_a", "bash", serde_json::json!({"command": "pwd"}));

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

    let selector = FailingSpawnSelector;
    let mut deps = stores.deps();
    deps.subagent_spawn_selector = Some(&selector);

    let result = execute_root_turn(inputs, "go", &provider, &deps, t_plus(5)).await;
    let Err(error) = result else {
        panic!("a failing selector must fail the turn, not silently reroute");
    };
    assert!(
        format!("{error:#}").contains("selector failed")
            || format!("{error:#}").contains("synthetic selector failure"),
        "expected the selector error to surface, got: {error:#}",
    );

    Ok(())
}

/// Provider that records the messages of the request it receives, so a
/// test can assert the worker never ships an unbalanced `tool_use`
/// history to the model.
struct CapturingProvider {
    captured: std::sync::Mutex<Option<Vec<agent_sdk_foundation::llm::Message>>>,
}

impl CapturingProvider {
    fn new() -> Self {
        Self {
            captured: std::sync::Mutex::new(None),
        }
    }

    fn captured(&self) -> Result<Vec<agent_sdk_foundation::llm::Message>> {
        self.captured
            .lock()
            .map_err(|_| anyhow::anyhow!("captured lock poisoned"))?
            .clone()
            .context("provider was never called")
    }
}

#[async_trait]
impl LlmProvider for CapturingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        *self
            .captured
            .lock()
            .map_err(|_| anyhow::anyhow!("captured lock poisoned"))? = Some(request.messages);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_capture_01".into(),
            content: vec![ContentBlock::Text {
                text: "Understood — asking one at a time.".into(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
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

/// Collect the `tool_use_id`s closed by a synthetic "User cancelled"
/// error result across a message list.
fn cancelled_result_ids(messages: &[agent_sdk_foundation::llm::Message]) -> Vec<&str> {
    use agent_sdk_foundation::llm::{Content, USER_CANCELLED_TOOL_RESULT};
    let mut ids = Vec::new();
    for message in messages {
        if let Content::Blocks(blocks) = &message.content {
            for block in blocks {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error: Some(true),
                } = block
                    && content == USER_CANCELLED_TOOL_RESULT
                {
                    ids.push(tool_use_id.as_str());
                }
            }
        }
    }
    ids
}

/// Regression for the exact screenshot 400: a prior turn asked several
/// questions (an assistant `tool_use` turn was persisted to the
/// suspension draft) and was then abandoned/cancelled, leaving the draft
/// unanswered. When the user types a fresh follow-up, the next root turn
/// must NOT ship that orphaned `tool_use` to the provider — the worker
/// backfills "User cancelled" results durably before the request is built,
/// so the conversation continues instead of failing the provider's
/// `tool_use`/`tool_result` pairing check.
#[tokio::test]
async fn fresh_turn_backfills_orphaned_tool_use_durably() -> Result<()> {
    use agent_sdk_foundation::llm::{self, Message};

    let stores = TestStores::new();
    let thread = thread_a();

    // A prior turn's suspension left an assistant turn with four unanswered
    // questions in the durable draft (committed_turns is still 0 — it never
    // completed).
    let ask = |id: &str| ContentBlock::ToolUse {
        id: id.to_string(),
        name: "ask_user".to_string(),
        input: serde_json::json!({ "question": "?" }),
        thought_signature: None,
    };
    stores
        .messages
        .set_draft(
            &thread,
            vec![
                Message::user("plan this"),
                Message::assistant_with_content(vec![ask("q1"), ask("q2"), ask("q3"), ask("q4")]),
            ],
            t0(),
        )
        .await?;

    // A fresh root turn carrying the user's follow-up.
    let task = create_and_acquire_task(&stores.tasks, &thread).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let provider = CapturingProvider::new();
    let outcome = execute_root_turn(
        inputs,
        "I only received one of the questions",
        &provider,
        &stores.deps(),
        t_plus(5),
    )
    .await
    .context("execute_root_turn")?;
    assert!(matches!(outcome, RootTurnOutcome::Completed { .. }));

    // The request actually handed to the provider is balanced — the four
    // unanswered questions are closed with "User cancelled" results.
    let request_messages = provider.captured()?;
    assert!(
        !llm::has_unbalanced_tool_use(&request_messages),
        "request handed to the provider must be balanced",
    );
    assert_eq!(
        cancelled_result_ids(&request_messages),
        vec!["q1", "q2", "q3", "q4"],
        "every unanswered question is closed with a cancelled result",
    );

    // The repair is durable, not a transient patch: the projection draft is
    // cleared and the committed history is balanced, so the next load is
    // clean without re-synthesizing.
    let durable = stores.messages.get_history(&thread).await?;
    assert!(
        !llm::has_unbalanced_tool_use(&durable),
        "durable projection history must be balanced after the turn",
    );
    let projection = stores
        .messages
        .get(&thread)
        .await?
        .context("projection exists")?;
    assert!(
        !projection.has_draft(),
        "the orphaned draft must be cleared, not left dangling",
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.5: steering wake (R2)
// ─────────────────────────────────────────────────────────────────────

/// Mock provider that records the messages of the last chat request so
/// tests can assert the wire contract (every pending `tool_use` id
/// resolved) and returns a fixed text answer.
struct SteeringCaptureProvider {
    response_text: String,
    captured: std::sync::Mutex<Vec<agent_sdk_foundation::llm::Message>>,
    call_count: AtomicUsize,
}

impl SteeringCaptureProvider {
    fn new(text: &str) -> Self {
        Self {
            response_text: text.to_owned(),
            captured: std::sync::Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    fn captured(&self) -> Vec<agent_sdk_foundation::llm::Message> {
        self.captured.lock().expect("capture lock").clone()
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for SteeringCaptureProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        *self.captured.lock().expect("capture lock") = request.messages;
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_steer_01".into(),
            content: vec![ContentBlock::Text {
                text: self.response_text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 90,
                output_tokens: 30,
                cached_input_tokens: 5,
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

/// Find the content of the first `tool_result` block for `id` across a
/// message list.
fn find_tool_result<'a>(
    messages: &'a [agent_sdk_foundation::llm::Message],
    id: &str,
) -> Option<&'a str> {
    for message in messages {
        if let agent_sdk_foundation::llm::Content::Blocks(blocks) = &message.content {
            for block in blocks {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                    && tool_use_id == id
                {
                    return Some(content.as_str());
                }
            }
        }
    }
    None
}

fn messages_contain_text(messages: &[agent_sdk_foundation::llm::Message], needle: &str) -> bool {
    messages.iter().any(|message| {
        matches!(
            &message.content,
            agent_sdk_foundation::llm::Content::Blocks(blocks)
                if blocks.iter().any(|block| matches!(
                    block,
                    ContentBlock::Text { text } if text.contains(needle)
                ))
        )
    })
}

/// A `bash` tool-call tuple for `suspend_leaving_children_running`.
fn bash_call(id: &str) -> (String, String, serde_json::Value) {
    (id.into(), "bash".into(), serde_json::json!({"command": id}))
}

/// Suspend a fresh turn at the tool boundary and return the parked
/// parent + its (still-running, uncompleted) children.
async fn suspend_leaving_children_running(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
) -> Result<(AgentTask, Vec<AgentTask>)> {
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
        panic!("expected Suspended");
    };
    Ok((parent_task, child_tasks))
}

/// Acquire a specific child and complete it with a real result at the
/// given timestamps (kept monotonic vs. surrounding steering events).
async fn complete_child_at(
    stores: &TestStores,
    child: &AgentTask,
    result: &agent_sdk_foundation::ToolResult,
    acquire_at: time::OffsetDateTime,
    complete_at: time::OffsetDateTime,
) -> Result<()> {
    let worker = WorkerId::from_string("child_w");
    let lease = LeaseId::from_string("child_l");
    stores
        .tasks
        .try_acquire_task(
            &child.id,
            worker.clone(),
            lease.clone(),
            complete_at,
            acquire_at,
        )
        .await?
        .context("acquire child")?;
    stores
        .tasks
        .complete_task_with_result(
            &child.id,
            &worker,
            &lease,
            serde_json::to_value(result).context("serialize child result")?,
            complete_at,
        )
        .await
        .context("complete child")?;
    Ok(())
}

#[tokio::test]
async fn steering_wake_answers_and_reparks_with_wire_contract() -> Result<()> {
    let stores = TestStores::new();
    let (parent, children) = Box::pin(suspend_leaving_children_running(
        &stores,
        vec![bash_call("call_0"), bash_call("call_1")],
    ))
    .await?;
    assert_eq!(children.len(), 2);

    // Child 0 finishes during the wave; child 1 keeps running.
    complete_child_at(
        &stores,
        &children[0],
        &ok_result("done-0"),
        t_plus(8),
        t_plus(9),
    )
    .await?;
    let parked = stores.tasks.get(&parent.id).await?.context("parked")?;
    assert_eq!(parked.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parked.pending_child_count, 1);

    // Fire the wake with a steering note.
    let woken = stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "how is it going?".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("woken")?;
    assert_eq!(woken.status, TaskStatus::Pending);
    assert!(woken.state.is_steering_resume());

    // The pool acquires the Pending row and runs the exchange.
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(21),
        )
        .await?
        .context("acquire woken parent")?;
    assert_eq!(acquired.status, TaskStatus::Running);

    let bootstrap = sample_bootstrap_with_tools(acquired.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(21),
    )
    .await?;
    let provider = SteeringCaptureProvider::new("One task is done, one is still running.");
    let outcome =
        resume_for_steering(inputs, &acquired, &provider, &stores.deps(), t_plus(25)).await?;
    assert_eq!(provider.calls(), 1, "exactly one bounded LLM round");

    // Wire contract on the steering call: a tool_result for EVERY
    // pending id (real for the finished child, typed-interim for the
    // running one), plus the steering note.
    let msgs = provider.captured();
    assert_eq!(find_tool_result(&msgs, "call_0"), Some("done-0"));
    let interim = find_tool_result(&msgs, "call_1").context("interim result for call_1")?;
    assert!(interim.contains("running"), "interim payload: {interim}");
    assert!(messages_contain_text(&msgs, "how is it going?"));

    // Re-park: no new children, parent waits on the survivor under a
    // fresh binding, the finished child is dropped from tracking.
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        panic!("expected Suspended re-park");
    };
    assert!(child_tasks.is_empty(), "re-attach must spawn nothing");
    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_task.pending_child_count, 1);
    let reattach_id = derive_reattach_tool_use_id("call_1");
    match &parent_task.state {
        TaskState::WaitingOnChildren {
            continuation,
            child_ids,
            ..
        } => {
            assert_eq!(child_ids.as_slice(), std::slice::from_ref(&children[1].id));
            let pending = &continuation.payload.pending_tool_calls;
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].id, reattach_id);
            assert_eq!(pending[0].name, "bash");
        }
        other => panic!("expected WaitingOnChildren re-park, got {other:?}"),
    }
    // Survivor re-indexed to dense position 0.
    let child1 = stores.tasks.get(&children[1].id).await?.context("child1")?;
    assert_eq!(child1.spawn_index, Some(0));

    // Survivor finishes → the real result reaches the coordinator under
    // the fresh binding (nothing lost, nothing duplicated).
    assert_survivor_fanin_delivers(&stores, &parent, &children[1], &reattach_id).await
}

/// Complete the still-running survivor, drive the ordinary fan-in, and
/// assert the coordinator's final LLM history resolves the re-issued
/// binding with the survivor's REAL result.
async fn assert_survivor_fanin_delivers(
    stores: &TestStores,
    parent: &AgentTask,
    survivor: &AgentTask,
    reattach_id: &str,
) -> Result<()> {
    complete_child_at(
        stores,
        survivor,
        &ok_result("done-1"),
        t_plus(50),
        t_plus(55),
    )
    .await?;
    let ready = stores.tasks.get(&parent.id).await?.context("ready")?;
    assert_eq!(ready.status, TaskStatus::Pending);
    assert!(matches!(ready.state, TaskState::ReadyToResume { .. }));

    let final_acq = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(60),
        )
        .await?
        .context("acquire for final fan-in")?;
    let bootstrap = sample_bootstrap_with_tools(final_acq.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(60),
    )
    .await?;
    let provider = SteeringCaptureProvider::new("All finished.");
    let outcome =
        resume_from_children(inputs, &final_acq, &provider, &stores.deps(), t_plus(65)).await?;
    assert_eq!(
        find_tool_result(&provider.captured(), reattach_id),
        Some("done-1"),
        "final fan-in must resolve the re-issued binding with the real result",
    );
    assert!(matches!(outcome, RootTurnOutcome::Completed { .. }));
    Ok(())
}

#[tokio::test]
async fn steering_wake_child_completing_concurrently_does_not_double_resume() -> Result<()> {
    let stores = TestStores::new();
    let (parent, children) = Box::pin(suspend_leaving_children_running(
        &stores,
        vec![(
            "call_only".into(),
            "bash".into(),
            serde_json::json!({"command": "x"}),
        )],
    ))
    .await?;
    assert_eq!(children.len(), 1);

    // Wake the parent (→ Pending + SteeringResume, pending_child_count 0).
    let woken = stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "status?".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("woken")?;
    assert!(woken.state.is_steering_resume());

    // Race: the child completes while the parent is parked in
    // SteeringResume. The completion must NOT flip the parent to
    // ReadyToResume — the wake already owns the resume.
    complete_child_at(
        &stores,
        &children[0],
        &ok_result("done"),
        t_plus(21),
        t_plus(22),
    )
    .await?;
    let after = stores
        .tasks
        .get(&parent.id)
        .await?
        .context("after complete")?;
    assert_eq!(
        after.status,
        TaskStatus::Pending,
        "concurrent completion must not re-flip the parent",
    );
    assert!(
        after.state.is_steering_resume(),
        "parent stays in SteeringResume — no double resume",
    );

    // The single acquire runs the exchange; every child is now terminal
    // so the answer completes the turn (one resume, no orphan).
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(30),
        )
        .await?
        .context("acquire woken parent")?;
    let bootstrap = sample_bootstrap_with_tools(acquired.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(30),
    )
    .await?;
    let provider = SteeringCaptureProvider::new("Everything finished while you asked.");
    let outcome =
        resume_for_steering(inputs, &acquired, &provider, &stores.deps(), t_plus(35)).await?;

    // Wire contract still holds: the finished child's real result is in
    // the history.
    assert_eq!(
        find_tool_result(&provider.captured(), "call_only"),
        Some("done")
    );
    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        panic!("expected Completed turn when no children survive");
    };
    assert_eq!(completed_task.status, TaskStatus::Completed);

    Ok(())
}

#[tokio::test]
async fn steering_wake_restart_between_wake_and_repark_resumes_cleanly() -> Result<()> {
    let stores = TestStores::new();
    let (parent, children) = Box::pin(suspend_leaving_children_running(
        &stores,
        vec![(
            "call_0".into(),
            "bash".into(),
            serde_json::json!({"command": "a"}),
        )],
    ))
    .await?;

    // Wake, then acquire (a worker starts the exchange) ...
    stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "ping".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("woken")?;
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_crash"),
            LeaseId::from_string("lease_crash"),
            t_plus(100),
            t_plus(21),
        )
        .await?
        .context("acquire")?;
    assert_eq!(acquired.status, TaskStatus::Running);

    // ... the daemon crashes before re-park. The lease expires and the
    // recovery sweep requeues the row — SteeringResume is preserved so
    // the durable continuation stays authoritative.
    let recovered = stores.tasks.release_expired_leases(t_plus(200)).await?;
    assert_eq!(recovered.len(), 1, "the wedged steering row must requeue");
    let requeued = stores.tasks.get(&parent.id).await?.context("requeued")?;
    assert_eq!(requeued.status, TaskStatus::Pending);
    assert!(
        requeued.state.is_steering_resume(),
        "requeue must preserve the steering continuation",
    );
    // Fresh turn consumed attempt 1; the crash's lease-release bumps to
    // 2 (poisoned-resume cap). Acquiring the steering resume did NOT
    // consume budget (it is a continuation, not a new attempt).
    assert_eq!(
        requeued.attempt, 2,
        "lease-release bumps the steering-resume attempt"
    );

    // A fresh worker re-acquires and re-runs the exchange to completion.
    let reacquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(210),
        )
        .await?
        .context("re-acquire after restart")?;
    let bootstrap = sample_bootstrap_with_tools(reacquired.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(210),
    )
    .await?;
    let provider = SteeringCaptureProvider::new("Still working.");
    let outcome =
        resume_for_steering(inputs, &reacquired, &provider, &stores.deps(), t_plus(215)).await?;

    // Re-park is deterministic: the survivor re-attaches under the same
    // derived id a first run would have produced.
    let RootTurnOutcome::Suspended { parent_task, .. } = outcome else {
        panic!("expected Suspended re-park after restart");
    };
    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    match &parent_task.state {
        TaskState::WaitingOnChildren { continuation, .. } => {
            assert_eq!(
                continuation.payload.pending_tool_calls[0].id,
                derive_reattach_tool_use_id("call_0"),
            );
        }
        other => panic!("expected WaitingOnChildren, got {other:?}"),
    }
    // The survivor was never cancelled or re-spawned.
    let child = stores.tasks.get(&children[0].id).await?.context("child")?;
    assert!(!child.status.is_terminal());

    Ok(())
}

/// A steering provider that replies with prose AND a follow-up
/// `tool_use` block — models a redirect that itself spawns new work
/// ("change of plan: also do X").
struct SteeringRedirectProvider {
    text: String,
    tool_call: (String, String, serde_json::Value),
}

impl SteeringRedirectProvider {
    fn new(text: &str, id: &str, name: &str, input: serde_json::Value) -> Self {
        Self {
            text: text.to_owned(),
            tool_call: (id.to_owned(), name.to_owned(), input),
        }
    }
}

#[async_trait]
impl LlmProvider for SteeringRedirectProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_steer_redirect".into(),
            content: vec![
                ContentBlock::Text {
                    text: self.text.clone(),
                },
                ContentBlock::ToolUse {
                    id: self.tool_call.0.clone(),
                    name: self.tool_call.1.clone(),
                    input: self.tool_call.2.clone(),
                    thought_signature: None,
                },
            ],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 80,
                output_tokens: 40,
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

/// Regression (MUST-FIX #1): when every child is terminal at exchange
/// time and the steering reply itself carries `tool_use` (the model
/// acting on a redirect), the follow-up wave must be spawned — not
/// silently dropped by `build_assistant_message`'s `tool_use` filter
/// with the mission turn force-completed.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn steering_wake_all_terminal_tool_use_spawns_followup_wave_not_completes() -> Result<()> {
    let stores = TestStores::new();
    let (parent, children) = Box::pin(suspend_leaving_children_running(
        &stores,
        vec![bash_call("call_0")],
    ))
    .await?;
    assert_eq!(children.len(), 1);

    // Wake first, THEN complete the only child concurrently — the
    // parent stays in SteeringResume with every child now terminal, so
    // the exchange hits the `surviving.is_empty()` branch.
    stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "change of plan: also check the logs".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("woken")?;
    complete_child_at(
        &stores,
        &children[0],
        &ok_result("done-0"),
        t_plus(21),
        t_plus(22),
    )
    .await?;
    let after = stores.tasks.get(&parent.id).await?.context("after")?;
    assert!(
        after.state.is_steering_resume(),
        "concurrent completion must not re-flip the parent out of SteeringResume",
    );

    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_test"),
            LeaseId::from_string("lease_test"),
            t_plus(900),
            t_plus(30),
        )
        .await?
        .context("acquire woken parent")?;
    let bootstrap = sample_bootstrap_with_tools(acquired.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(30),
    )
    .await?;

    // The steering reply acts on the redirect: text + a follow-up
    // `bash` tool_use. Before the fix this tool_use was silently
    // dropped and the mission turn force-completed.
    let provider = SteeringRedirectProvider::new(
        "On it — also inspecting the logs now.",
        "followup_call",
        "bash",
        serde_json::json!({"command": "tail -n 100 app.log"}),
    );
    let outcome =
        resume_for_steering(inputs, &acquired, &provider, &stores.deps(), t_plus(35)).await?;

    // The follow-up wave is spawned instead of the turn completing.
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        panic!("expected Suspended (follow-up wave), not a force-completed turn");
    };
    assert_eq!(
        child_tasks.len(),
        1,
        "the redirect's tool_use must spawn one new child",
    );
    assert_eq!(child_tasks[0].status, TaskStatus::Pending);
    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_task.pending_child_count, 1);

    match &parent_task.state {
        TaskState::WaitingOnChildren {
            continuation,
            suspended_messages,
            child_ids,
        } => {
            // The new wave binds the steering response's tool_use id.
            let pending = &continuation.payload.pending_tool_calls;
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].id, "followup_call");
            assert_eq!(pending[0].name, "bash");
            // The new child is tracked (not the original terminal one).
            assert_eq!(
                child_ids.as_slice(),
                std::slice::from_ref(&child_tasks[0].id)
            );
            // The steering directive survives in the replay history — it
            // is NOT dropped when the follow-up wave is threaded through
            // `suspend_resumed_turn`.
            assert!(
                messages_contain_text(suspended_messages, "change of plan: also check the logs"),
                "steering note must survive in the re-parked replay history",
            );
            // The finished child's real result is in the replay too.
            assert!(
                find_tool_result(suspended_messages, "call_0").is_some(),
                "the finished child's result must be in the replay history",
            );
        }
        other => panic!("expected WaitingOnChildren, got {other:?}"),
    }

    // The original child stays terminal; the follow-up child is fresh.
    let original = stores
        .tasks
        .get(&children[0].id)
        .await?
        .context("original child")?;
    assert!(original.status.is_terminal());
    assert_ne!(child_tasks[0].id, children[0].id);

    Ok(())
}

/// Regression (MUST-FIX #2): a steering exchange that fails (e.g. a
/// provider outage during the bounded LLM round) must revert the parent
/// to its pre-wake parked state rather than failing it — failing would
/// strand the still-running mission children (`fail_task` does not
/// cascade) and orphan their results. The mission then resumes
/// uninterrupted through the ordinary fan-in.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn steering_wake_failure_reverts_to_parked_state_without_stranding_children() -> Result<()> {
    let stores = TestStores::new();
    let (parent, children) = Box::pin(suspend_leaving_children_running(
        &stores,
        vec![bash_call("call_0"), bash_call("call_1")],
    ))
    .await?;
    assert_eq!(children.len(), 2);

    // Wake the parent, then a worker acquires it (Running + SteeringResume).
    stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "how is it going?".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("woken")?;
    let worker = WorkerId::from_string("worker_test");
    let lease = LeaseId::from_string("lease_test");
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            worker.clone(),
            lease.clone(),
            t_plus(900),
            t_plus(21),
        )
        .await?
        .context("acquire")?;
    assert_eq!(acquired.status, TaskStatus::Running);
    assert!(acquired.state.is_steering_resume());

    // Simulate the bounded steering LLM round failing (provider outage
    // exactly when the user asked). This must NOT fail the parent —
    // that would strand the two still-running children.
    let error = anyhow::anyhow!("provider outage during steering exchange");
    let reparked = revert_steering_wake(
        &acquired,
        &worker,
        &lease,
        &error,
        &stores.deps(),
        t_plus(25),
    )
    .await?;

    // The parent is back on its ORIGINAL children — not Failed.
    assert_eq!(reparked.status, TaskStatus::WaitingOnChildren);
    assert_eq!(reparked.pending_child_count, 2);
    assert!(
        !reparked.state.is_steering_resume(),
        "steering note is dropped on revert — no wake→error→re-park loop",
    );
    match &reparked.state {
        TaskState::WaitingOnChildren {
            continuation,
            child_ids,
            ..
        } => {
            let pending = &continuation.payload.pending_tool_calls;
            assert_eq!(pending.len(), 2);
            assert_eq!(pending[0].id, "call_0");
            assert_eq!(pending[1].id, "call_1");
            assert_eq!(child_ids.len(), 2);
        }
        other => panic!("expected WaitingOnChildren revert, got {other:?}"),
    }

    // Neither child was cancelled or stranded — both keep running.
    for child in &children {
        let live = stores.tasks.get(&child.id).await?.context("child")?;
        assert!(
            !live.status.is_terminal(),
            "children must keep running after a steering revert",
        );
    }

    // The failure is surfaced as a non-fatal error event.
    assert!(
        collected_event_kinds(&stores.events)
            .await
            .contains(&"error".to_owned()),
        "steering failure must surface an error event",
    );

    // The mission resumes uninterrupted: completing both children fans
    // in through the ordinary resume path and completes the turn with
    // the ORIGINAL bindings resolved by the real results.
    complete_child_at(
        &stores,
        &children[0],
        &ok_result("out-0"),
        t_plus(30),
        t_plus(31),
    )
    .await?;
    complete_child_at(
        &stores,
        &children[1],
        &ok_result("out-1"),
        t_plus(32),
        t_plus(33),
    )
    .await?;
    let ready = stores.tasks.get(&parent.id).await?.context("ready")?;
    assert_eq!(ready.status, TaskStatus::Pending);
    assert!(matches!(ready.state, TaskState::ReadyToResume { .. }));
    assert!(!ready.state.is_steering_resume());

    let final_acq = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            worker.clone(),
            lease.clone(),
            t_plus(900),
            t_plus(40),
        )
        .await?
        .context("acquire for final fan-in")?;
    let bootstrap = sample_bootstrap_with_tools(final_acq.clone());
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(40),
    )
    .await?;
    let provider = SteeringCaptureProvider::new("All finished after the hiccup.");
    let outcome =
        resume_from_children(inputs, &final_acq, &provider, &stores.deps(), t_plus(45)).await?;

    assert_eq!(
        find_tool_result(&provider.captured(), "call_0"),
        Some("out-0")
    );
    assert_eq!(
        find_tool_result(&provider.captured(), "call_1"),
        Some("out-1")
    );
    assert!(matches!(outcome, RootTurnOutcome::Completed { .. }));

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Commit the completed prefix of a cancelled turn (partial commit)
// ─────────────────────────────────────────────────────────────────────

/// Flatten the text blocks of a message for content assertions.
fn message_text(message: &agent_sdk_foundation::llm::Message) -> String {
    use agent_sdk_foundation::llm::Content;
    match &message.content {
        Content::Text(text) => text.clone(),
        Content::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// An assistant message carrying a thinking block (with signature) plus a
/// `tool_use` block — the shape whose byte-verbatim preservation matters.
fn assistant_thinking_tool_use(
    thinking: &str,
    signature: &str,
    tool_id: &str,
) -> agent_sdk_foundation::llm::Message {
    use agent_sdk_foundation::llm::{Content, Message};
    Message {
        role: Role::Assistant,
        content: Content::Blocks(vec![
            ContentBlock::Thinking {
                thinking: thinking.to_owned(),
                signature: Some(signature.to_owned()),
            },
            ContentBlock::ToolUse {
                id: tool_id.to_owned(),
                name: "bash".to_owned(),
                input: serde_json::json!({"command": "ls"}),
                thought_signature: None,
            },
        ]),
    }
}

// ── provider_valid_split matrix ─────────────────────────────────────

#[test]
fn provider_valid_split_empty() {
    let (prefix, suffix) = provider_valid_split(Vec::new());
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
}

#[test]
fn provider_valid_split_user_only() {
    let (prefix, suffix) =
        provider_valid_split(vec![agent_sdk_foundation::llm::Message::user("hi")]);
    assert_eq!(prefix.len(), 1);
    assert!(suffix.is_empty());
}

#[test]
fn provider_valid_split_balanced_round_commits_in_full() {
    let messages = vec![
        agent_sdk_foundation::llm::Message::user("do it"),
        agent_sdk_foundation::llm::Message::assistant_with_tool_use(
            None,
            "call_1",
            "bash",
            serde_json::json!({"command": "ls"}),
        ),
        agent_sdk_foundation::llm::Message::tool_result("call_1", "file.txt", false),
    ];
    let (prefix, suffix) = provider_valid_split(messages);
    assert_eq!(prefix.len(), 3, "a balanced round commits in full");
    assert!(suffix.is_empty());
}

#[test]
fn provider_valid_split_unbalanced_drops_trailing_tool_use() {
    let messages = vec![
        agent_sdk_foundation::llm::Message::user("do it"),
        agent_sdk_foundation::llm::Message::assistant_with_tool_use(
            None,
            "call_1",
            "bash",
            serde_json::json!({"command": "ls"}),
        ),
    ];
    let (prefix, suffix) = provider_valid_split(messages);
    // Prefix is exactly the user prompt; the unresulted tool_use is
    // retained as the suffix.
    assert_eq!(prefix.len(), 1);
    assert_eq!(prefix[0].role, Role::User);
    assert_eq!(suffix.len(), 1);
    assert_eq!(suffix[0].role, Role::Assistant);
}

#[test]
fn provider_valid_split_multi_cycle_with_trailing_unbalanced() {
    // [u, a+tu1, tr1, a+tu2] → prefix keeps the first balanced round,
    // drops the trailing unresulted tool_use.
    let messages = vec![
        agent_sdk_foundation::llm::Message::user("q"),
        agent_sdk_foundation::llm::Message::assistant_with_tool_use(
            None,
            "call_1",
            "bash",
            serde_json::json!({"command": "ls"}),
        ),
        agent_sdk_foundation::llm::Message::tool_result("call_1", "out", false),
        agent_sdk_foundation::llm::Message::assistant_with_tool_use(
            None,
            "call_2",
            "bash",
            serde_json::json!({"command": "pwd"}),
        ),
    ];
    let (prefix, suffix) = provider_valid_split(messages);
    assert_eq!(prefix.len(), 3, "first balanced round is retained");
    assert_eq!(suffix.len(), 1, "trailing unresulted tool_use is dropped");
    assert!(!agent_sdk_foundation::llm::has_unbalanced_tool_use(&prefix));
}

#[test]
fn provider_valid_split_preserves_thinking_verbatim() {
    // Balanced round whose assistant carries a thinking block with a
    // signature: the committed prefix must be byte-identical.
    let asst = assistant_thinking_tool_use("let me think", "sig-abc123", "call_1");
    let messages = vec![
        agent_sdk_foundation::llm::Message::user("q"),
        asst.clone(),
        agent_sdk_foundation::llm::Message::tool_result("call_1", "out", false),
    ];
    let (prefix, suffix) = provider_valid_split(messages);
    assert_eq!(prefix.len(), 3);
    assert!(suffix.is_empty());
    // The thinking block (and its signature) round-tripped unchanged.
    assert_eq!(
        serde_json::to_value(&prefix[1]).unwrap(),
        serde_json::to_value(&asst).unwrap(),
        "thinking + signature must commit byte-verbatim",
    );
}

// ── commit_partial_turn_on_cancel unit behavior ─────────────────────

#[tokio::test]
async fn partial_commit_empty_candidate_is_a_strict_no_op() -> Result<()> {
    let stores = TestStores::new();
    // Seed the thread row (get_or_create).
    stores.threads.get_or_create(&thread_a(), t0()).await?;
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;

    let suffix = commit_partial_turn_on_cancel(
        PartialCancelCommit {
            thread_id: &thread_a(),
            task_id: &task.id,
            candidate: Vec::new(),
            expected_turn: 1,
            agent_state_snapshot: serde_json::Value::Null,
        },
        &stores.deps(),
        t_plus(1),
    )
    .await?;

    assert!(suffix.is_empty());
    // No attempt row, no thread mutation, no projection write.
    assert!(stores.attempts.list_by_task(&task.id).await?.is_empty());
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn partial_commit_unbalanced_candidate_commits_prefix_returns_suffix() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;

    let candidate = vec![
        agent_sdk_foundation::llm::Message::user("prompt"),
        agent_sdk_foundation::llm::Message::assistant_with_tool_use(
            None,
            "call_1",
            "bash",
            serde_json::json!({"command": "ls"}),
        ),
    ];
    let suffix = commit_partial_turn_on_cancel(
        PartialCancelCommit {
            thread_id: &thread_a(),
            task_id: &task.id,
            candidate,
            expected_turn: 1,
            agent_state_snapshot: serde_json::Value::Null,
        },
        &stores.deps(),
        t_plus(1),
    )
    .await?;

    // The user prompt committed; the tool_use came back as suffix.
    assert_eq!(suffix.len(), 1);
    assert_eq!(suffix[0].role, Role::Assistant);
    let committed = stores.messages.get_history(&thread_a()).await?;
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].role, Role::User);
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);
    // Usage is not double-billed onto the thread aggregate.
    assert_eq!(thread.total_usage.input_tokens, 0);
    assert_eq!(thread.total_usage.output_tokens, 0);
    Ok(())
}

// ── Seam B: fresh-turn mid-stream cancel ────────────────────────────

/// Streaming provider that models a real mid-stream cancel: it streams
/// one text delta, optionally flips the task's durable status to
/// `Cancelled` via `cancel_tree` (mirroring a stop/ESC that drops the
/// worker's lease), trips the shared cancellation token, streams a
/// second delta, then ends WITHOUT a `Done` marker so the worker's
/// biased cancel branch wins on the next poll. The streamed
/// "partial answer" lives only in the accumulator and must be dropped.
struct MidStreamCancelProvider {
    cancel: CancellationToken,
    cancel_tree: Option<(InMemoryAgentTaskStore, AgentTaskId, time::OffsetDateTime)>,
}

#[async_trait]
impl LlmProvider for MidStreamCancelProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        anyhow::bail!("MidStreamCancelProvider drives only the streaming path")
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let cancel = self.cancel.clone();
        let cancel_tree = self.cancel_tree.clone();
        Box::pin(async_stream::stream! {
            yield Ok(StreamDelta::TextDelta {
                delta: "partial ".to_owned(),
                block_index: 0,
            });
            if let Some((store, task_id, now)) = cancel_tree {
                let _ = store.cancel_tree(&task_id, now).await;
            }
            cancel.cancel();
            yield Ok(StreamDelta::TextDelta {
                delta: "answer".to_owned(),
                block_index: 0,
            });
        })
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

#[tokio::test]
async fn fresh_turn_mid_stream_cancel_commits_user_prompt() -> Result<()> {
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

    let cancel = CancellationToken::new();
    let provider = MidStreamCancelProvider {
        cancel: cancel.clone(),
        cancel_tree: Some((stores.tasks.clone(), task_id.clone(), t_plus(2))),
    };
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let result = execute_root_turn(
        inputs,
        "What is the capital of France?",
        &provider,
        &deps,
        t_plus(5),
    )
    .await;
    let error = result.err().context("cancelled turn must return Err")?;
    assert!(
        format!("{error:#}").to_lowercase().contains("cancel"),
        "expected a cancellation error, got: {error:#}",
    );

    // The completed prefix — exactly the user prompt — commits as turn 1.
    // The streamed "partial answer" is dropped (accumulator-only).
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);

    let committed = stores.messages.get_history(&thread_a()).await?;
    assert_eq!(committed.len(), 1, "only the user prompt commits");
    assert_eq!(committed[0].role, Role::User);
    assert!(message_text(&committed[0]).contains("capital of France"));
    assert!(
        !committed.iter().any(|m| m.role == Role::Assistant),
        "the mid-stream assistant response must not leak into the commit",
    );

    // Exactly one checkpoint at turn 1.
    assert_eq!(
        stores.checkpoints.list_by_thread(&thread_a()).await?.len(),
        1,
    );

    // A synthetic cancel-commit attempt was opened and closed Cancelled.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    let synthetic = attempts
        .iter()
        .find(|a| a.provider == "cancel-commit")
        .context("synthetic cancel-commit attempt")?;
    assert!(synthetic.is_closed());
    assert_eq!(synthetic.outcome, Some(TurnAttemptOutcome::Cancelled));

    // The next turn's recovered context contains the committed prompt.
    let view = crate::journal::thread_recover::recover_thread(
        &thread_a(),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(6),
    )
    .await?;
    assert_eq!(view.next_turn_number, 2);
    assert!(
        view.messages
            .iter()
            .any(|m| message_text(m).contains("capital of France")),
        "the rebuilt context must carry the cancelled turn's prompt",
    );

    Ok(())
}

#[tokio::test]
async fn lease_lost_mid_stream_does_not_partial_commit() -> Result<()> {
    // The token trips mid-stream but the task's durable status stays
    // Running (a lease-lost requeue, NOT a user cancel). Seam B's status
    // guard must skip the partial commit so the re-run owns turn N.
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

    let cancel = CancellationToken::new();
    let provider = MidStreamCancelProvider {
        cancel: cancel.clone(),
        // No cancel_tree: the task stays Running.
        cancel_tree: None,
    };
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let result = execute_root_turn(inputs, "a question", &provider, &deps, t_plus(5)).await;
    assert!(result.is_err(), "cancel still aborts the turn");

    // No partial commit: the task was never Cancelled.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 0);
    assert!(stores.messages.get_history(&thread_a()).await?.is_empty());
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert!(
        !attempts.iter().any(|a| a.provider == "cancel-commit"),
        "no synthetic cancel-commit attempt on a lease-lost requeue",
    );
    Ok(())
}

// ── Seam B: resume mid-stream cancel (balanced-draft-drop regression) ─

#[tokio::test]
async fn resume_mid_stream_cancel_commits_tool_results_and_clears_draft() -> Result<()> {
    let stores = TestStores::new();

    let child_results = vec![(
        "call_1".to_owned(),
        agent_sdk_foundation::ToolResult {
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

    let bootstrap = sample_bootstrap_with_tools(acquired);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(20),
    )
    .await?;

    // Cancel mid-resume: the token trips and the task's durable status
    // flips to Cancelled before the resume LLM call.
    let cancel = CancellationToken::new();
    let provider = MidStreamCancelProvider {
        cancel: cancel.clone(),
        cancel_tree: Some((stores.tasks.clone(), parent_id.clone(), t_plus(22))),
    };
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let result = Box::pin(resume_root_turn(
        inputs,
        continuation,
        suspended_messages,
        child_results,
        &provider,
        &deps,
        t_plus(25),
    ))
    .await;
    assert!(result.is_err(), "a cancelled resume must return Err");

    // The full balanced resume delta committed: user prompt + assistant
    // tool_use + tool_results — the balanced-draft-drop is fixed.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);

    let committed = stores.messages.get_history(&thread_a()).await?;
    assert!(
        committed.len() >= 3,
        "resume delta commits in full (prompt + tool_use + results), got {}",
        committed.len(),
    );
    assert!(
        !agent_sdk_foundation::llm::has_unbalanced_tool_use(&committed),
        "the committed history is provider-valid",
    );
    // The tool result survives into committed history.
    assert!(
        committed
            .iter()
            .any(|m| message_text(m).contains("file1.txt") || has_tool_result_for(m, "call_1")),
        "the completed tool results survive the cancel",
    );

    // The draft was cleared in the commit transaction.
    let projection = stores
        .messages
        .get(&thread_a())
        .await?
        .context("projection")?;
    assert!(!projection.has_draft(), "commit clears the in-flight draft");

    Ok(())
}

/// True if `message` carries a `tool_result` for `tool_use_id`.
fn has_tool_result_for(message: &agent_sdk_foundation::llm::Message, tool_use_id: &str) -> bool {
    use agent_sdk_foundation::llm::Content;
    matches!(
        &message.content,
        Content::Blocks(blocks)
            if blocks.iter().any(|b| matches!(
                b,
                ContentBlock::ToolResult { tool_use_id: id, .. } if id == tool_use_id
            ))
    )
}

// ── Seam A: parked WaitingOnChildren cancel + next-turn backfill ─────

#[tokio::test]
async fn parked_cancel_commits_prefix_and_next_turn_backfill_closes_without_dup() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_a", "bash", serde_json::json!({"command": "ls"}));

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

    // Suspend at a tool boundary → parent WaitingOnChildren, draft =
    // [user, assistant+tool_use].
    let outcome = execute_root_turn(inputs, "run ls", &provider, &stores.deps(), t_plus(5)).await?;
    assert!(matches!(outcome, RootTurnOutcome::Suspended { .. }));

    // Cancel the parked parent via cancel_root_turn (seam A).
    cancel_root_turn(&task_id, &stores.deps(), t_plus(6)).await?;

    // The user prompt committed as turn 1; the trailing tool_use is the
    // re-seeded draft.
    let committed = stores.messages.get_history(&thread_a()).await?;
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].role, Role::User);
    let projection = stores
        .messages
        .get(&thread_a())
        .await?
        .context("projection")?;
    assert_eq!(
        projection.draft_messages.len(),
        1,
        "trailing tool_use re-seeded"
    );

    // Run a fresh next turn on a NEW task. The pre-call backfill must
    // close the orphaned tool_use with a cancelled result WITHOUT
    // duplicating the already-committed user prompt.
    let next_task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let next_bootstrap = sample_bootstrap(next_task);
    let next_inputs = build_root_worker_inputs(
        next_bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t_plus(10),
    )
    .await?;
    let next_provider = MockTextProvider::new("all done");
    let next_outcome = execute_root_turn(
        next_inputs,
        "and now?",
        &next_provider,
        &stores.deps(),
        t_plus(11),
    )
    .await?;
    assert!(matches!(next_outcome, RootTurnOutcome::Completed { .. }));

    let final_history = stores.messages.get_history(&thread_a()).await?;
    assert!(
        !agent_sdk_foundation::llm::has_unbalanced_tool_use(&final_history),
        "the next turn balances the orphaned tool_use",
    );
    // The cancelled prefix's user prompt appears exactly once (no dup).
    let run_ls_count = final_history
        .iter()
        .filter(|m| message_text(m).contains("run ls"))
        .count();
    assert_eq!(run_ls_count, 1, "the committed prefix is not duplicated");

    Ok(())
}

#[tokio::test]
async fn double_cancel_root_turn_commits_prefix_once() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_a", "bash", serde_json::json!({"command": "ls"}));

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
    let outcome = execute_root_turn(inputs, "run ls", &provider, &stores.deps(), t_plus(5)).await?;
    assert!(matches!(outcome, RootTurnOutcome::Suspended { .. }));

    cancel_root_turn(&task_id, &stores.deps(), t_plus(6)).await?;
    // Second cancel finds the row already terminal (Cancelled) — no
    // longer parked — so seam A skips the commit; even if it were
    // retried, the expected_turn CAS would block a re-commit.
    let second = cancel_root_turn(&task_id, &stores.deps(), t_plus(7)).await?;
    assert!(
        second.is_empty(),
        "second cancel on a terminal tree is a no-op"
    );

    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1, "the prefix commits exactly once");
    assert_eq!(
        stores.checkpoints.list_by_thread(&thread_a()).await?.len(),
        1,
    );
    Ok(())
}

/// Open an attempt row for `task_id` so a test can observe who closes it.
async fn open_test_attempt(
    attempts: &InMemoryTurnAttemptStore,
    task_id: &AgentTaskId,
    attempt_number: u32,
    now: time::OffsetDateTime,
) -> Result<TurnAttempt> {
    attempts
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number,
            provenance: AuditProvenance::new("test-provider", "test-model"),
            request_blob: serde_json::json!({ "user_prompt": "test" }),
            now,
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await
        .context("open test attempt")
}

/// Cancelling a `Running` root must NOT pre-close its open attempt:
/// `close_attempt` is single-shot, and the live worker's own close (the
/// stream-abort `Cancelled` close, or a suspension / commit close that
/// carries the round's real token usage) would be rejected against a
/// zero-usage row pinned here.
#[tokio::test]
async fn cancel_of_running_root_leaves_attempt_close_to_its_worker() -> Result<()> {
    let stores = TestStores::new();

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let attempt = open_test_attempt(&stores.attempts, &task_id, 1, t0()).await?;

    let cancelled = cancel_root_turn(&task_id, &stores.deps(), t_plus(1)).await?;
    assert_eq!(cancelled, vec![task_id.clone()]);
    let row = stores.tasks.get(&task_id).await?.context("task")?;
    assert_eq!(row.status, TaskStatus::Cancelled);

    let reread = stores.attempts.get(&attempt.id).await?.context("attempt")?;
    assert!(
        !reread.is_closed(),
        "a running root's attempt stays open for its live worker to close",
    );
    Ok(())
}

/// A parked root has no live worker, so the cancel seam still owns the
/// attempt-close hygiene: an attempt left open on a `WaitingOnChildren`
/// row (e.g. a steering exchange whose process died mid-flight) is
/// closed `Cancelled` by `cancel_root_turn`.
#[tokio::test]
async fn cancel_of_parked_root_closes_open_attempts() -> Result<()> {
    let stores = TestStores::new();
    let provider =
        MockToolCallProvider::single("call_a", "bash", serde_json::json!({"command": "ls"}));

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
    let outcome = execute_root_turn(inputs, "run ls", &provider, &stores.deps(), t_plus(5)).await?;
    assert!(matches!(outcome, RootTurnOutcome::Suspended { .. }));
    let parked = stores.tasks.get(&task_id).await?.context("parked task")?;
    assert_eq!(parked.status, TaskStatus::WaitingOnChildren);

    // Simulate an attempt left open on the parked row.
    let existing = stores.attempts.list_by_task(&task_id).await?;
    let next_number = u32::try_from(existing.len()).context("attempt count")? + 1;
    let orphan = open_test_attempt(&stores.attempts, &task_id, next_number, t_plus(6)).await?;

    cancel_root_turn(&task_id, &stores.deps(), t_plus(7)).await?;

    let reread = stores.attempts.get(&orphan.id).await?.context("attempt")?;
    assert!(
        reread.is_closed(),
        "a parked root's open attempt is closed by the cancel seam",
    );
    assert_eq!(reread.outcome, Some(TurnAttemptOutcome::Cancelled));
    Ok(())
}

/// Count the `AgentEvent::Cancelled` markers committed on `thread`.
async fn cancelled_event_count(
    events: &InMemoryEventRepository,
    thread: &ThreadId,
) -> Result<usize> {
    use agent_sdk_foundation::events::AgentEvent;
    Ok(events
        .get_events(thread)
        .await?
        .iter()
        .filter(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
        .count())
}

/// An effective cancel commits exactly one terminal `Cancelled` event
/// (delivered to live subscribers via the notifier), and an idempotent
/// retry on the already-terminal tree emits nothing.
#[tokio::test]
async fn cancel_root_turn_commits_single_terminal_cancelled_event() -> Result<()> {
    use agent_sdk_foundation::events::AgentEvent;

    let stores = TestStores::new();
    let task = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let task_id = task.id.clone();
    stores.tasks.submit_root_turn(task).await?;

    let mut live_rx = stores.event_notifier.subscribe(&thread_a());

    let cancelled = cancel_root_turn(&task_id, &stores.deps(), t_plus(1)).await?;
    assert_eq!(cancelled.len(), 1);
    assert_eq!(cancelled_event_count(&stores.events, &thread_a()).await?, 1);

    // The notifier woke live followers with the terminal marker.
    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), live_rx.recv())
        .await
        .context("timed out waiting for the notified cancelled event")?
        .context("notifier channel closed")?;
    let AgentEvent::Cancelled { turn, usage, .. } = &delivered.event else {
        bail!(
            "live followers must receive the terminal marker, got {:?}",
            delivered.event,
        );
    };
    // A plain Pending root on a fresh thread has no durable usage
    // anywhere — the honest report is zero at turn zero.
    assert_eq!(*turn, 0);
    assert_eq!(usage, &TokenUsage::default());

    // The marker's cross-host advisory landed in the outbox alongside
    // the committed event (issue #354: remote followers are woken by
    // the relay, not this process's notifier).
    let advisory_rows = stores
        .outbox
        .claim_pending("marker-relay-probe", 16, t_plus(2))
        .await?;
    assert_eq!(
        advisory_rows
            .iter()
            .filter(|row| row.kind == OutboxMessageKind::ThreadEventsAvailable
                && row.thread_id == thread_a())
            .count(),
        1,
        "exactly one thread_events_available advisory for the marker, got {advisory_rows:?}",
    );

    // Idempotent retry: nothing transitioned, so no second marker.
    let second = cancel_root_turn(&task_id, &stores.deps(), t_plus(2)).await?;
    assert!(second.is_empty());
    assert_eq!(cancelled_event_count(&stores.events, &thread_a()).await?, 1);
    Ok(())
}

/// Event repo that parks the first `Cancelled` commit on a [`Notify`]
/// until the test releases it — forcing the exact interleaving the
/// in-lock marker commit must exclude.
#[derive(Clone)]
struct StallingMarkerEventRepo {
    inner: InMemoryEventRepository,
    /// Signalled when the marker commit has parked (the cancel holds the
    /// task-store write lock at that point).
    parked: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl EventRepository for StallingMarkerEventRepo {
    async fn commit_event(
        &self,
        thread_id: &ThreadId,
        event: agent_sdk_foundation::events::AgentEvent,
        now: time::OffsetDateTime,
    ) -> Result<CommittedEvent> {
        if matches!(
            event,
            agent_sdk_foundation::events::AgentEvent::Cancelled { .. }
        ) {
            self.parked.notify_one();
            self.release.notified().await;
        }
        self.inner.commit_event(thread_id, event, now).await
    }
    async fn commit_event_batch(
        &self,
        thread_id: &ThreadId,
        events: Vec<agent_sdk_foundation::events::AgentEvent>,
        now: time::OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        self.inner.commit_event_batch(thread_id, events, now).await
    }
    async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
        self.inner.next_sequence(thread_id).await
    }
    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
        self.inner.get_events(thread_id).await
    }
    async fn get_events_in_range(
        &self,
        thread_id: &ThreadId,
        after_sequence: u64,
        up_to_sequence: u64,
    ) -> Result<Vec<CommittedEvent>> {
        self.inner
            .get_events_in_range(thread_id, after_sequence, up_to_sequence)
            .await
    }
    async fn threads_with_events_before(
        &self,
        cutoff: time::OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        self.inner.threads_with_events_before(cutoff, limit).await
    }
    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: time::OffsetDateTime,
    ) -> Result<Option<u64>> {
        self.inner.max_sequence_before(thread_id, cutoff).await
    }
    async fn min_sequence_at_or_after(
        &self,
        thread_id: &ThreadId,
        cutoff: time::OffsetDateTime,
    ) -> Result<Option<u64>> {
        self.inner.min_sequence_at_or_after(thread_id, cutoff).await
    }
}

/// The in-memory sink commits markers while the task-store write lock
/// is still held, so a promoted successor cannot be acquired — and thus
/// cannot journal its first event — before the marker's sequence. The
/// stalling repo parks the marker commit; if the lock were dropped
/// before the sink commit, the successor's acquire would succeed during
/// the stall and its events would precede the marker.
#[tokio::test]
async fn cancel_marker_sequence_precedes_promoted_successor_events() -> Result<()> {
    let threads = InMemoryThreadStore::new();
    let events = InMemoryEventRepository::new();
    let outbox = InMemoryOutboxStore::new();
    let parked = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let tasks = InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
        event_repo: Arc::new(StallingMarkerEventRepo {
            inner: events.clone(),
            parked: Arc::clone(&parked),
            release: Arc::clone(&release),
        }),
        outbox_store: Arc::new(outbox.clone()),
        thread_store: Arc::new(threads.clone()),
    });

    // Active blocking root + queued successor on the same thread.
    let active = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let active_id = active.id.clone();
    tasks.submit_root_turn(active).await?;
    let queued = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let queued_id = queued.id.clone();
    tasks.submit_root_turn(queued).await?;

    // Cancel in the background; it parks inside the marker commit while
    // still holding the task-store write lock.
    let cancelling_tasks = tasks.clone();
    let cancel_id = active_id.clone();
    let cancel =
        tokio::spawn(async move { cancelling_tasks.cancel_tree(&cancel_id, t_plus(1)).await });

    // Wait until the marker commit is provably parked — the cancel holds
    // the task-store write lock from here until the release.
    tokio::time::timeout(std::time::Duration::from_secs(5), parked.notified())
        .await
        .context("the marker commit must park in the stalling repo")?;

    // While the marker commit is parked, the promoted successor must NOT
    // be acquirable: the write lock is held across the sink commit.
    let blocked_acquire = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        tasks.try_acquire_task(
            &queued_id,
            WorkerId::from_string("w-race"),
            LeaseId::from_string("l-race"),
            t_plus(60),
            t_plus(1),
        ),
    )
    .await;
    assert!(
        blocked_acquire.is_err(),
        "the successor must not be acquirable while the marker commit holds the lock",
    );

    // Release the marker commit; the cancel completes with the marker.
    release.notify_one();
    let outcome = cancel.await.context("cancel task join")??;
    assert_eq!(outcome.markers.len(), 1);

    // Now the successor acquires and journals its first event — strictly
    // after the marker's sequence.
    let acquired = tasks
        .try_acquire_task(
            &queued_id,
            WorkerId::from_string("w-race"),
            LeaseId::from_string("l-race"),
            t_plus(60),
            t_plus(2),
        )
        .await?;
    let Some(_) = acquired else {
        bail!("the promoted successor must be acquirable after the cancel completes");
    };
    let successor_event = events
        .commit_event(
            &thread_a(),
            agent_sdk_foundation::events::AgentEvent::text("s", "successor start"),
            t_plus(2),
        )
        .await?;

    let journal = events.get_events(&thread_a()).await?;
    let marker_seq = journal
        .iter()
        .find(|e| {
            matches!(
                e.event,
                agent_sdk_foundation::events::AgentEvent::Cancelled { .. }
            )
        })
        .map(|e| e.sequence)
        .context("the marker must be in the journal")?;
    assert!(
        marker_seq < successor_event.sequence,
        "the marker (seq {marker_seq}) must precede the successor's first event (seq {})",
        successor_event.sequence,
    );
    Ok(())
}

/// Cancelling a QUEUED root parked behind a live active root must NOT
/// commit a thread-terminal `Cancelled` marker — that would close every
/// follower mid-stream while the active root keeps producing events.
/// The queued cancel stays observable via the returned ids / `GetTask`.
#[tokio::test]
async fn queued_root_cancel_emits_no_thread_terminal_marker() -> Result<()> {
    let stores = TestStores::new();

    // First root takes the active slot; the second queues behind it.
    let active = AgentTask::new_root_turn(thread_a(), t0(), 3);
    let active_id = active.id.clone();
    stores.tasks.submit_root_turn(active).await?;
    let queued = AgentTask::new_root_turn(thread_a(), t_plus(1), 3);
    let queued_id = queued.id.clone();
    stores.tasks.submit_root_turn(queued).await?;
    let queued_row = stores.tasks.get(&queued_id).await?.context("queued row")?;
    assert_eq!(queued_row.status, TaskStatus::Queued, "precondition");

    let cancelled = cancel_root_turn(&queued_id, &stores.deps(), t_plus(2)).await?;
    assert_eq!(cancelled, vec![queued_id]);

    // No thread-terminal marker, and the active root is undisturbed.
    assert_eq!(cancelled_event_count(&stores.events, &thread_a()).await?, 0);
    let active_row = stores.tasks.get(&active_id).await?.context("active row")?;
    assert_eq!(active_row.status, TaskStatus::Pending);
    Ok(())
}

/// Attempt store that simulates the successor-slot race in the exact
/// window that would leak the salvage's synthetic attempt: the first
/// `open_attempt` advances the thread's turn counter AFTER the
/// salvage's stale-turn pre-check has passed, so the salvage's
/// `commit_completed_turn` loses the stale-turn guard while the
/// synthetic attempt sits open.
struct TurnStealingAttemptStore {
    inner: InMemoryTurnAttemptStore,
    threads: InMemoryThreadStore,
    thread_id: ThreadId,
    fired: AtomicBool,
}

#[async_trait]
impl TurnAttemptStore for TurnStealingAttemptStore {
    async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt> {
        if !self.fired.swap(true, Ordering::SeqCst) {
            self.threads
                .commit_turn(&self.thread_id, 1, &TokenUsage::default(), params.now)
                .await
                .context("steal the turn slot")?;
        }
        self.inner.open_attempt(params).await
    }

    async fn close_attempt(
        &self,
        id: &TurnAttemptId,
        params: CloseAttemptParams,
        now: time::OffsetDateTime,
    ) -> Result<TurnAttempt> {
        self.inner.close_attempt(id, params, now).await
    }

    async fn get(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        self.inner.get(id).await
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>> {
        self.inner.list_by_task(task_id).await
    }
}

/// A salvage commit that fails (here: the stale-turn guard losing to a
/// racing successor) must close the synthetic cancel-commit attempt it
/// opened — otherwise the attempt leaks open forever on a terminal
/// task, invisible to the cancel seam's snapshot-scoped close.
#[tokio::test]
async fn failed_salvage_commit_closes_its_synthetic_attempt() -> Result<()> {
    let stores = TestStores::new();
    let thread = thread_a();
    stores.threads.get_or_create(&thread, t0()).await?;

    let stealing = TurnStealingAttemptStore {
        inner: stores.attempts.clone(),
        threads: stores.threads.clone(),
        thread_id: thread.clone(),
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &stealing;

    let task = AgentTask::new_root_turn(thread.clone(), t0(), 3);
    let task_id = task.id.clone();

    let result = commit_partial_turn_on_cancel(
        PartialCancelCommit {
            thread_id: &thread,
            task_id: &task_id,
            candidate: vec![agent_sdk_foundation::llm::Message::user("salvage me")],
            expected_turn: 1,
            agent_state_snapshot: serde_json::json!({}),
        },
        &deps,
        t_plus(1),
    )
    .await;
    let error = result
        .err()
        .context("a stolen turn slot must fail the salvage commit")?;
    assert!(
        format!("{error:#}").contains("stale turn commit"),
        "expected the stale-turn guard to reject, got: {error:#}",
    );

    // The synthetic attempt must not leak open.
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1, "exactly the synthetic attempt");
    assert!(
        attempts[0].is_closed(),
        "the failed salvage must close its own synthetic attempt",
    );
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Cancelled));
    Ok(())
}

/// The terminal `Cancelled` event derives its usage from the parked
/// turn's durable continuation (`total_usage` — the cumulative usage
/// accumulated so far, matching the SDK's own `Cancelled` emission),
/// not a zero default.
#[tokio::test]
async fn cancelled_event_derives_usage_from_parked_continuation() -> Result<()> {
    use agent_sdk_foundation::events::AgentEvent;

    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();

    let seeded_usage = TokenUsage {
        input_tokens: 111,
        output_tokens: 22,
        cached_input_tokens: 3,
        cache_creation_input_tokens: 4,
    };
    let continuation = ContinuationEnvelope::wrap(AgentContinuation {
        thread_id: thread_a(),
        turn: 1,
        total_usage: seeded_usage.clone(),
        turn_usage: TokenUsage::default(),
        pending_tool_calls: Vec::new(),
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: AgentState::new(thread_a()),
        response_id: None,
        stop_reason: None,
        response_content: Vec::new(),
    });
    stores
        .tasks
        .spawn_tool_children(
            &task_id,
            &WorkerId::from_string("worker_test"),
            &LeaseId::from_string("lease_test"),
            vec![ChildSpawnSpec::new(2)],
            SuspensionPayload {
                continuation,
                suspended_messages: Vec::new(),
            },
            None,
            t_plus(1),
        )
        .await
        .context("park the root on a tool child")?;

    cancel_root_turn(&task_id, &stores.deps(), t_plus(2)).await?;

    let events = stores.events.get_events(&thread_a()).await?;
    let cancelled = events
        .iter()
        .find(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
        .context("terminal cancelled event")?;
    let AgentEvent::Cancelled { usage, .. } = &cancelled.event else {
        bail!("filtered to a cancelled event above");
    };
    assert_eq!(
        usage, &seeded_usage,
        "usage must come from the parked continuation's total_usage",
    );
    Ok(())
}

/// Streaming provider that models the real RPC-cancel-while-running
/// sequence: it streams one delta, runs the full `cancel_root_turn`
/// lifecycle (which commits the terminal `Cancelled` marker), trips the
/// worker's token, then streams once more so the biased cancel branch
/// aborts the turn and seam B salvages the prefix.
struct MidStreamCancelRootTurnProvider {
    cancel: CancellationToken,
    stores: TestStores,
    task_id: AgentTaskId,
    cancel_at: time::OffsetDateTime,
}

#[async_trait]
impl LlmProvider for MidStreamCancelRootTurnProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        anyhow::bail!("MidStreamCancelRootTurnProvider drives only the streaming path")
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let cancel = self.cancel.clone();
        let stores = self.stores.clone();
        let task_id = self.task_id.clone();
        let cancel_at = self.cancel_at;
        Box::pin(async_stream::stream! {
            yield Ok(StreamDelta::TextDelta {
                delta: "partial ".to_owned(),
                block_index: 0,
            });
            let _ = cancel_root_turn(&task_id, &stores.deps(), cancel_at).await;
            cancel.cancel();
            yield Ok(StreamDelta::TextDelta {
                delta: "answer".to_owned(),
                block_index: 0,
            });
        })
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

/// The RPC-cancel-then-worker-abort sequence commits exactly ONE
/// terminal `Cancelled` marker: `cancel_root_turn` emits it, and the
/// running worker's abort path (seam B) salvages the prefix without
/// adding a second lifecycle event.
#[tokio::test]
async fn running_abort_after_cancel_commits_single_cancelled_event() -> Result<()> {
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

    let cancel = CancellationToken::new();
    let provider = MidStreamCancelRootTurnProvider {
        cancel: cancel.clone(),
        stores: stores.clone(),
        task_id: task_id.clone(),
        cancel_at: t_plus(2),
    };
    let mut deps = stores.deps();
    deps.cancel = Some(&cancel);

    let result = execute_root_turn(
        inputs,
        "What is the capital of France?",
        &provider,
        &deps,
        t_plus(5),
    )
    .await;
    let error = result.err().context("cancelled turn must return Err")?;
    assert!(
        format!("{error:#}").to_lowercase().contains("cancel"),
        "expected a cancellation error, got: {error:#}",
    );

    // Seam B still salvaged the prefix (the user prompt) as turn 1.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);

    // Exactly one terminal marker: the cancel's, not a second from the
    // worker's abort path.
    assert_eq!(cancelled_event_count(&stores.events, &thread_a()).await?, 1);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Issue #354 — successor turn vs cancelled salvage slot race
// ─────────────────────────────────────────────────────────────────────

/// Minimal close params for the simulated salvage commit.
fn salvage_close_params() -> CloseAttemptParams {
    CloseAttemptParams {
        response_blob: serde_json::json!({"salvage": true}),
        response_id: None,
        response_model: Some("mock-model".into()),
        stop_reason: None,
        outcome: TurnAttemptOutcome::Cancelled,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        route_provider: None,
        thinking_adaptive: false,
        resolved_effort: None,
    }
}

/// Attempt store that reproduces the successor-slot race
/// deterministically: when the SUCCESSOR opens its turn attempt (i.e.
/// after it bootstrapped with `next_turn_number = 1`), a cancelled
/// predecessor's seam-B salvage lands one or more REAL completed-turn
/// commits — checkpoints included — for a DIFFERENT task starting on
/// the successor's slot. Optionally cancels a task first so the
/// negative test can pin that a dead root never shifts.
struct SalvageRacingAttemptStore {
    inner: InMemoryTurnAttemptStore,
    stores: TestStores,
    salvager_task: AgentTaskId,
    /// Task to cancel via `cancel_tree` before the salvage commit
    /// lands (models an RPC cancel racing the running worker).
    cancel_first: Option<AgentTaskId>,
    /// The commits the race lands, in slot order starting at turn 1:
    /// `(turn_usage, checkpoint_kind)` per commit. A zero-usage
    /// `CancelSalvage` entry models the synthetic seam-B salvage; a
    /// `FullTurn` entry models a live predecessor's completion landing
    /// after the successor bootstrapped. Multiple entries model
    /// back-to-back late commits leaving consecutive occupied slots.
    racing_commits: Vec<(TokenUsage, CheckpointKind)>,
    fired: AtomicBool,
}

impl SalvageRacingAttemptStore {
    async fn land_racing_salvage(&self, now: time::OffsetDateTime) -> Result<()> {
        if let Some(cancel_id) = &self.cancel_first {
            self.stores
                .tasks
                .cancel_tree(cancel_id, now)
                .await
                .context("cancel racing task")?;
        }
        for (index, (usage, kind)) in self.racing_commits.iter().enumerate() {
            let slot = u32::try_from(index).context("racing slot overflow")? + 1;
            let salvage_attempt = self
                .inner
                .open_attempt(OpenAttemptParams {
                    task_id: self.salvager_task.clone(),
                    attempt_number: slot,
                    provenance: AuditProvenance::new("mock", "mock-model"),
                    request_blob: serde_json::json!({}),
                    now,
                    otel_trace_id: None,
                    otel_span_id: None,
                })
                .await
                .context("open salvage attempt")?;
            crate::journal::commit::commit_completed_turn(
                crate::journal::commit::CompletedTurnCommit {
                    checkpoint_kind: *kind,
                    thread_id: thread_a(),
                    task_id: self.salvager_task.clone(),
                    expected_turn: slot,
                    turn_attempt_id: salvage_attempt.id,
                    close_attempt_params: salvage_close_params(),
                    messages: vec![agent_sdk_foundation::llm::Message::assistant(
                        "salvaged prefix",
                    )],
                    turn_usage: usage.clone(),
                    agent_state_snapshot: serde_json::json!({}),
                    events: Vec::new(),
                    outbox_max_attempts: 3,
                    owner_guard: None,
                    now,
                },
                &self.stores.threads,
                &self.stores.messages,
                &self.inner,
                &self.stores.checkpoints,
                &self.stores.events,
            )
            .await
            .context("racing salvage commit")?;
        }
        Ok(())
    }
}

#[async_trait]
impl TurnAttemptStore for SalvageRacingAttemptStore {
    async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt> {
        let attempt = self.inner.open_attempt(params).await?;
        if !self.fired.swap(true, Ordering::SeqCst) {
            self.land_racing_salvage(t_plus(1)).await?;
        }
        Ok(attempt)
    }

    async fn close_attempt(
        &self,
        id: &TurnAttemptId,
        params: CloseAttemptParams,
        now: time::OffsetDateTime,
    ) -> Result<TurnAttempt> {
        self.inner.close_attempt(id, params, now).await
    }

    async fn get(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        self.inner.get(id).await
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>> {
        self.inner.list_by_task(task_id).await
    }
}

/// A successor turn that loses its bootstrapped slot to a cancelled
/// predecessor's salvage commit must not fail terminally — the commit
/// detects the cross-task collision, shifts to the next turn number,
/// and completes, with the turn-indexed lifecycle events remapped to
/// the landed turn.
#[tokio::test]
async fn successor_turn_shifts_past_cancelled_salvage_slot_collision() -> Result<()> {
    use agent_sdk_foundation::events::AgentEvent;

    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    // The successor root turn: acquired, bootstrapped at turn 1.
    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let successor_id = successor.id.clone();
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;
    assert_eq!(inputs.recovery_view.next_turn_number, 1, "precondition");

    // The cancelled predecessor whose salvage steals turn 1 after the
    // successor bootstrapped.
    let salvager_id = AgentTaskId::from_string("task_cancelled_salvager");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: salvager_id.clone(),
        cancel_first: None,
        racing_commits: vec![(TokenUsage::default(), CheckpointKind::CancelSalvage)],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("successor answer");
    let outcome = execute_root_turn(inputs, "run the successor", &provider, &deps, t_plus(5))
        .await
        .context("successor turn must not fail on the stolen slot")?;
    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        bail!("expected Completed, got a suspension");
    };
    assert_eq!(completed_task.status, TaskStatus::Completed);

    // The salvage kept turn 1; the successor landed on turn 2.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 2);
    let salvage_checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("salvage checkpoint at turn 1")?;
    assert_eq!(salvage_checkpoint.task_id, salvager_id);
    let successor_checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 2)
        .await?
        .context("successor checkpoint at turn 2")?;
    assert_eq!(successor_checkpoint.task_id, successor_id);
    // The shifted checkpoint's state snapshot must carry the LANDED
    // turn count — recovery seeds staged state from it, and a stale
    // count leaves every later turn one behind.
    assert_eq!(
        successor_checkpoint
            .agent_state_snapshot
            .get("turn_count")
            .and_then(serde_json::Value::as_u64),
        Some(2),
        "shifted snapshot must carry the landed turn count",
    );

    // Turn-indexed lifecycle events were remapped to the landed turn.
    let events = stores.events.get_events(&thread_a()).await?;
    let turn_complete = events
        .iter()
        .find_map(|committed| match &committed.event {
            AgentEvent::TurnComplete { turn, .. } => Some(*turn),
            _ => None,
        })
        .context("TurnComplete event")?;
    assert_eq!(turn_complete, 2, "TurnComplete must carry the landed turn");
    let done_turns = events
        .iter()
        .find_map(|committed| match &committed.event {
            AgentEvent::Done { total_turns, .. } => Some(*total_turns),
            _ => None,
        })
        .context("Done event")?;
    assert_eq!(done_turns, 2, "Done must carry the landed turn");

    // The shift rewrites turn indices only: emitter attribution names
    // the task that actually executed the turn, which the slot number it
    // landed on cannot change.
    for committed in &events {
        let stamped = matches!(
            committed.event,
            AgentEvent::Start { .. } | AgentEvent::TurnComplete { .. } | AgentEvent::Done { .. }
        );
        if stamped {
            assert_eq!(
                committed.event.emitter_task_id(),
                Some(successor_id.as_str()),
                "the shifted turn's lifecycle events must stay attributed to their emitter: {:?}",
                committed.event,
            );
        }
    }
    Ok(())
}

/// A foreign FULL-TURN commit occupying the slot must
/// refuse the shift — the successor's snapshot and `Done` totals were
/// built before the predecessor's fully billed turn landed, so shifting
/// would durably drop that usage/cost. The collision surfaces instead
/// (the host's collision handler requeues the task to re-run from the
/// fresh committed head). Only a `CancelSalvage` occupant is
/// state-compatible with a shift.
#[tokio::test]
async fn billed_foreign_commit_refuses_the_shift() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // The racing commit is a live predecessor's FULL turn: real usage.
    let predecessor_id = AgentTaskId::from_string("task_billed_predecessor");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: predecessor_id.clone(),
        cancel_first: None,
        racing_commits: vec![(
            TokenUsage {
                input_tokens: 111,
                output_tokens: 222,
                ..Default::default()
            },
            CheckpointKind::FullTurn,
        )],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("built on a stale head");
    let error = execute_root_turn(inputs, "stale successor", &provider, &deps, t_plus(5))
        .await
        .err()
        .context("a billed foreign occupant must refuse the shift")?;
    assert!(
        format!("{error:#}").contains("already committed"),
        "expected the slot-collision rejection, got: {error:#}",
    );

    // The billed turn is intact and nothing was spliced after it.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);
    let occupant = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint at turn 1")?;
    assert_eq!(occupant.task_id, predecessor_id);
    assert!(
        stores
            .checkpoints
            .get_by_turn(&thread_a(), 2)
            .await?
            .is_none(),
        "the stale successor must not have committed a shifted turn 2",
    );
    Ok(())
}

/// A root that was CANCELLED while running must not shift — its late
/// full-turn commit loses to the successor's committed slot and
/// propagates the collision error.
/// This test exercises the shift-eligibility guard (the caller-side
/// re-read); the guard→retry pair is not atomic, so the authoritative
/// enforcement on the durable backends is the `owner_guard` the
/// shifted retry carries into the commit transaction (pinned by the
/// `conformance_{sqlite,postgres}_owner_guarded_commit_rejects_lost_ownership`
/// tests). On this in-memory backend the eligibility re-read is the
/// strongest available check — the residual in-process window is
/// documented on `CompletedTurnCommit::owner_guard`.
#[tokio::test]
async fn cancelled_root_commit_never_shifts_past_a_successor() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    // The soon-cancelled root, bootstrapped at turn 1.
    let doomed = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let doomed_id = doomed.id.clone();
    let bootstrap = sample_bootstrap(doomed);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Mid-turn: an external cancel lands on the doomed root, then a
    // successor's commit takes turn 1.
    let successor_id = AgentTaskId::from_string("task_fast_successor");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: successor_id.clone(),
        cancel_first: Some(doomed_id.clone()),
        racing_commits: vec![(TokenUsage::default(), CheckpointKind::CancelSalvage)],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("too late");
    let error = execute_root_turn(inputs, "doomed turn", &provider, &deps, t_plus(5))
        .await
        .err()
        .context("a cancelled root's late commit must fail, not shift")?;
    assert!(
        format!("{error:#}").contains("already committed"),
        "expected the slot-collision rejection, got: {error:#}",
    );

    // The successor's turn is the only committed turn; the dead root
    // spliced nothing after it.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);
    let occupant = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint at turn 1")?;
    assert_eq!(occupant.task_id, successor_id);
    assert!(
        stores
            .checkpoints
            .get_by_turn(&thread_a(), 2)
            .await?
            .is_none(),
        "the cancelled root must not have committed a shifted turn 2",
    );

    // The dead root's no-shift exit must settle its
    // own attempt — the cancel seam left the live worker's attempt
    // open and the host skips rows it no longer owns, so without the
    // settle it would leak open forever.
    let attempts = stores.attempts.list_by_task(&doomed_id).await?;
    assert!(!attempts.is_empty(), "the doomed turn opened an attempt");
    assert!(
        attempts.iter().all(TurnAttempt::is_closed),
        "a cancelled root's collision exit must leave no attempt open",
    );
    Ok(())
}

/// Token usage is NOT a salvage signature: a provider
/// that omits its usage delta produces a real completion with all-zero
/// counters; if that full turn occupies the slot, the shift must still
/// refuse — the successor bootstrapped before it and would overwrite its
/// state with a stale snapshot. Eligibility keys off the checkpoint's
/// durable `kind`, which only the cancel path stamps `CancelSalvage`.
#[tokio::test]
async fn zero_usage_full_turn_refuses_the_shift() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // A real full-turn completion whose provider reported zero usage.
    let predecessor_id = AgentTaskId::from_string("task_zero_usage_predecessor");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: predecessor_id.clone(),
        cancel_first: None,
        racing_commits: vec![(TokenUsage::default(), CheckpointKind::FullTurn)],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("built on a stale head");
    let error = execute_root_turn(inputs, "stale successor", &provider, &deps, t_plus(5))
        .await
        .err()
        .context("a zero-usage FULL turn occupant must refuse the shift")?;
    assert!(
        format!("{error:#}").contains("already committed"),
        "expected the slot-collision rejection, got: {error:#}",
    );

    // The full turn is intact and nothing was spliced after it.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 1);
    let occupant = stores
        .checkpoints
        .get_by_turn(&thread_a(), 1)
        .await?
        .context("checkpoint at turn 1")?;
    assert_eq!(occupant.task_id, predecessor_id);
    assert_eq!(occupant.kind, CheckpointKind::FullTurn);
    assert!(
        stores
            .checkpoints
            .get_by_turn(&thread_a(), 2)
            .await?
            .is_none(),
        "the stale successor must not have committed a shifted turn 2",
    );
    Ok(())
}

/// The shift advances ONE slot per collision so every
/// intervening occupant is validated. Two late commits land back-to-back
/// — turn 1 is shiftable salvage, but turn 2 is a BILLED full turn. A
/// head-jumping shift would validate only turn 1 and splice the
/// successor's stale state past the billed turn; single-stepping
/// re-collides at turn 2, inspects THAT occupant, and refuses.
#[tokio::test]
async fn shift_refuses_when_an_intervening_occupant_is_a_full_turn() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let predecessor_id = AgentTaskId::from_string("task_two_late_commits");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: predecessor_id.clone(),
        cancel_first: None,
        racing_commits: vec![
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
            (
                TokenUsage {
                    input_tokens: 333,
                    output_tokens: 444,
                    ..Default::default()
                },
                CheckpointKind::FullTurn,
            ),
        ],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("built on a stale head");
    let error = execute_root_turn(inputs, "stale successor", &provider, &deps, t_plus(5))
        .await
        .err()
        .context("an intervening billed occupant must stop the walk")?;
    assert!(
        format!("{error:#}").contains("already committed"),
        "expected the slot-collision rejection, got: {error:#}",
    );

    // Both late commits are intact; the successor spliced nothing.
    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 2);
    assert!(
        stores
            .checkpoints
            .get_by_turn(&thread_a(), 3)
            .await?
            .is_none(),
        "the stale successor must not have committed a shifted turn 3",
    );
    Ok(())
}

/// Exhausting the shift budget on a genuine
/// collision must not leak the attempt — the max-shift exit settles it
/// exactly like the no-shift exit (ownership may have been lost
/// concurrently, and no later path owns the attempt then).
#[tokio::test]
async fn exhausted_shift_budget_settles_the_attempt() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let successor_id = successor.id.clone();
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Four back-to-back salvage occupants: the walk absorbs three
    // shifts and gives up on the fourth collision.
    let salvager_id = AgentTaskId::from_string("task_quad_salvager");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: salvager_id.clone(),
        cancel_first: None,
        racing_commits: vec![
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
        ],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("cannot land");
    let error = execute_root_turn(inputs, "walk too far", &provider, &deps, t_plus(5))
        .await
        .err()
        .context("a fourth collision must exhaust the walk")?;
    assert!(
        format!("{error:#}").contains("already committed"),
        "expected the slot-collision rejection, got: {error:#}",
    );

    // The successor's own attempts are all settled; nothing leaks open.
    let attempts = stores.attempts.list_by_task(&successor_id).await?;
    assert!(!attempts.is_empty(), "the walk opened an attempt");
    assert!(
        attempts.iter().all(TurnAttempt::is_closed),
        "the max-shift exit must leave no attempt open",
    );
    Ok(())
}

/// Positive counterpart of the single-step walk: when EVERY intervening
/// occupant is shiftable salvage, the walk crosses them one slot at a
/// time and the successor lands on the first free slot.
#[tokio::test]
async fn shift_walks_across_consecutive_salvage_occupants() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;

    let successor = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let successor_id = successor.id.clone();
    let bootstrap = sample_bootstrap(successor);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let salvager_id = AgentTaskId::from_string("task_double_salvager");
    let racing = SalvageRacingAttemptStore {
        inner: stores.attempts.clone(),
        stores: stores.clone(),
        salvager_task: salvager_id.clone(),
        cancel_first: None,
        racing_commits: vec![
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
            (TokenUsage::default(), CheckpointKind::CancelSalvage),
        ],
        fired: AtomicBool::new(false),
    };
    let mut deps = stores.deps();
    deps.attempt_store = &racing;

    let provider = MockTextProvider::new("successor answer");
    let outcome = execute_root_turn(inputs, "run the successor", &provider, &deps, t_plus(5))
        .await
        .context("the walk must cross two salvage slots")?;
    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        bail!("expected Completed, got a suspension");
    };
    assert_eq!(completed_task.status, TaskStatus::Completed);

    let thread = stores.threads.get(&thread_a()).await?.context("thread")?;
    assert_eq!(thread.committed_turns, 3);
    let successor_checkpoint = stores
        .checkpoints
        .get_by_turn(&thread_a(), 3)
        .await?
        .context("successor checkpoint at turn 3")?;
    assert_eq!(successor_checkpoint.task_id, successor_id);
    assert_eq!(successor_checkpoint.kind, CheckpointKind::FullTurn);
    Ok(())
}

/// A leftover OPEN attempt from a predecessor
/// execution is LEFT OPEN by the next execution — the old worker may
/// still be live after a lease expiry, and closing its attempt from
/// here would permanently record zero usage where its real billed
/// usage belongs. The new execution numbers its own attempt past the
/// leftover and completes normally.
#[tokio::test]
async fn next_execution_leaves_a_leftover_open_attempt_alone() -> Result<()> {
    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();

    // The predecessor execution's attempt, left open by a rolled-back
    // commit.
    let leftover = stores
        .attempts
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("mock", "mock-model"),
            request_blob: serde_json::json!({}),
            now: t0(),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;

    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;
    let deps = stores.deps();
    let provider = MockTextProvider::new("re-run answer");
    let outcome = execute_root_turn(inputs, "re-run", &provider, &deps, t_plus(5))
        .await
        .context("re-driven execution")?;
    let RootTurnOutcome::Completed { .. } = outcome else {
        bail!("expected Completed");
    };

    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 2, "leftover + fresh attempt");
    let leftover_row = attempts
        .iter()
        .find(|attempt| attempt.id == leftover.id)
        .context("leftover attempt")?;
    assert!(
        !leftover_row.is_closed(),
        "the predecessor's attempt must be left alone — its (possibly \
         still live) owner settles it with real usage",
    );
    let fresh = attempts
        .iter()
        .find(|attempt| attempt.id != leftover.id)
        .context("fresh attempt")?;
    assert!(
        fresh.is_closed(),
        "the new execution's own attempt closes with its commit",
    );
    Ok(())
}

/// An owner-guard rejection
/// (`LostCommitOwnership`) rolls the in-transaction attempt close
/// back, and no later path owns that attempt — the shift wrapper
/// settles it best-effort, preserving the REAL usage on the
/// billing-source-of-truth attempt row under the `Cancelled` outcome.
#[tokio::test]
async fn lost_ownership_rejection_settles_the_open_attempt() -> Result<()> {
    use crate::journal::commit::LostCommitOwnership;

    let stores = TestStores::new();
    stores.threads.get_or_create(&thread_a(), t0()).await?;
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let attempt = stores
        .attempts
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("mock", "mock-model"),
            request_blob: serde_json::json!({}),
            now: t0(),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;

    let params = crate::journal::commit::CompletedTurnCommit {
        checkpoint_kind: CheckpointKind::FullTurn,
        thread_id: thread_a(),
        task_id: task_id.clone(),
        expected_turn: 1,
        turn_attempt_id: attempt.id.clone(),
        close_attempt_params: CloseAttemptParams {
            response_blob: serde_json::json!({}),
            response_id: None,
            response_model: Some("mock-model".into()),
            stop_reason: None,
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 77,
            output_tokens: 33,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_adaptive: false,
            resolved_effort: None,
        },
        messages: vec![],
        turn_usage: TokenUsage::default(),
        agent_state_snapshot: serde_json::json!({}),
        events: Vec::new(),
        outbox_max_attempts: 3,
        owner_guard: None,
        now: t_plus(2),
    };
    let deps = stores.deps();

    // A non-ownership error must NOT settle: the attempt stays open
    // for whichever path owns the failure.
    let unrelated = anyhow::anyhow!("provider exploded");
    settle_attempt_after_lost_ownership(&unrelated, &params, &deps).await;
    let row = stores.attempts.get(&attempt.id).await?.context("attempt")?;
    assert!(!row.is_closed(), "unrelated errors must not settle");

    // The owner-guard rejection settles with REAL usage + Cancelled.
    let rejection = anyhow::Error::new(LostCommitOwnership {
        task_id: task_id.clone(),
    })
    .context("commit completed turn");
    settle_attempt_after_lost_ownership(&rejection, &params, &deps).await;
    let row = stores.attempts.get(&attempt.id).await?.context("attempt")?;
    assert!(row.is_closed());
    assert_eq!(row.outcome, Some(TurnAttemptOutcome::Cancelled));
    assert_eq!(row.input_tokens, Some(77));
    assert_eq!(row.output_tokens, Some(33));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Mixed batches — subagent spawns + tool children in the same turn
// ─────────────────────────────────────────────────────────────────────

/// Routes every `subagent_*` call to a durable spawn and everything else
/// to the regular tool path — the shape a coordinator host wires when its
/// LLM emits subagent calls alongside ordinary tools in one response.
struct MixedSpawnSelector;

impl MixedSpawnSelector {
    fn plan(task: &str) -> super::subagent_spawn_selector::SubagentSpawnPlan {
        use crate::worker::subagent::{
            EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
            InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentCapabilityRequest,
            SubagentSandboxPolicy, SubagentSpawnRequest,
        };
        let capabilities: std::collections::BTreeSet<String> =
            std::iter::once("read_file".to_owned()).collect();
        let inherited_policy = InheritedSubagentPolicy {
            default_model: "mock-model".to_owned(),
            allowed_models: std::iter::once("mock-model".to_owned()).collect(),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: 30_000,
            max_timeout_ms: 30_000,
            capability_profiles: std::collections::BTreeMap::from([(
                "research".to_owned(),
                SubagentCapabilityProfile {
                    capabilities: capabilities.clone(),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: std::collections::BTreeSet::new(),
                },
            )]),
            allowed_capabilities: capabilities.clone(),
            max_depth: 3,
            max_parallel_subagents: 2,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: std::collections::BTreeSet::new(),
            audit_provider: "mock".to_owned(),
        };
        super::subagent_spawn_selector::SubagentSpawnPlan {
            request: SubagentSpawnRequest::new(task, SubagentCapabilityRequest::new("research")),
            spec: EffectiveSubagentSpec {
                task: task.to_owned(),
                prompt: String::new(),
                model: "mock-model".to_owned(),
                max_turns: 5,
                timeout_ms: 30_000,
                depth: 1,
                max_parallel_subagents: 0,
                nickname: None,
                sandbox: SubagentSandboxPolicy::read_only(),
                mcp: EffectiveSubagentMcpPolicy::default(),
                audit_provenance: None,
                inherited_policy,
                capabilities: EffectiveSubagentCapabilities {
                    profile: "research".to_owned(),
                    allowed: capabilities,
                },
            },
            child_thread_id: ThreadId::new(),
            child_root_input: Vec::new(),
            child_caller_metadata: None,
        }
    }
}

#[async_trait]
impl super::subagent_spawn_selector::SubagentSpawnSelector for MixedSpawnSelector {
    async fn decide(
        &self,
        _parent_thread_id: &ThreadId,
        tool_calls: &[agent_sdk_foundation::PendingToolCallInfo],
    ) -> Result<Vec<super::subagent_spawn_selector::SubagentSpawnDecision>> {
        Ok(tool_calls
            .iter()
            .map(|call| {
                if call.name.starts_with("subagent_") {
                    super::subagent_spawn_selector::SubagentSpawnDecision::SpawnAsSubagent {
                        plan: Box::new(Self::plan(&call.name)),
                    }
                } else {
                    super::subagent_spawn_selector::SubagentSpawnDecision::SpawnAsTool
                }
            })
            .collect())
    }
}

/// A definition exposing one Confirm-tier subagent tool (the spawn path
/// requires Confirm tier) alongside the ordinary Observe-tier `bash`.
fn sample_definition_with_subagent_tools() -> AgentDefinition {
    AgentDefinition {
        tools: vec![
            Tool {
                name: "subagent_explore".into(),
                description: "Spawn an exploring subagent".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"task": {"type": "string"}}}),
                display_name: "Subagent: Explore".into(),
                tier: agent_sdk_foundation::ToolTier::Confirm,
            },
            Tool {
                name: "bash".into(),
                description: "Run a shell command".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
                display_name: "Bash".into(),
                tier: agent_sdk_foundation::ToolTier::Observe,
            },
        ],
        ..sample_definition()
    }
}

/// The two tool calls of a mixed turn, subagent first (slot 0) and the
/// ordinary tool second (slot 1).
fn subagent_then_tool_calls() -> Vec<(String, String, serde_json::Value)> {
    vec![
        (
            "call_sub".into(),
            "subagent_explore".into(),
            serde_json::json!({"task": "map the crate"}),
        ),
        (
            "call_bash".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        ),
    ]
}

/// The same two calls interleaved the other way: the ordinary tool takes
/// slot 0 and the subagent slot 1, so the batch's children are NOT
/// grouped by kind in slot order.
fn tool_then_subagent_calls() -> Vec<(String, String, serde_json::Value)> {
    vec![
        (
            "call_bash".into(),
            "bash".into(),
            serde_json::json!({"command": "ls"}),
        ),
        (
            "call_sub".into(),
            "subagent_explore".into(),
            serde_json::json!({"task": "map the crate"}),
        ),
    ]
}

/// Run one turn whose LLM response mixes a `subagent_explore` call with
/// a `bash` call, routed through [`MixedSpawnSelector`]. Returns the
/// parked parent and its children.
async fn suspend_on_mixed_batch(
    stores: &TestStores,
    tool_calls: Vec<(String, String, serde_json::Value)>,
) -> Result<(AgentTask, Vec<AgentTask>)> {
    let provider = MockToolCallProvider::new(tool_calls);

    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let mut bootstrap = sample_bootstrap_with_tools(task);
    bootstrap.definition = sample_definition_with_subagent_tools();
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let selector = MixedSpawnSelector;
    let mut deps = stores.deps();
    deps.subagent_spawn_selector = Some(&selector);

    let outcome = execute_root_turn(inputs, "go", &provider, &deps, t_plus(5)).await?;
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        panic!("a mixed batch must suspend the parent on its children");
    };
    Ok((parent_task, child_tasks))
}

#[tokio::test]
async fn mixed_batch_spawns_subagent_slot_and_tool_child_in_one_turn() -> Result<()> {
    // A decision vector mixing SpawnAsSubagent with SpawnAsTool spawns
    // the subagent slot AND routes the remaining call as a tool child in
    // the same turn — no degradation to an all-tools batch, no retry
    // round-trip.
    let stores = TestStores::new();
    let (parent_task, child_tasks) =
        suspend_on_mixed_batch(&stores, subagent_then_tool_calls()).await?;

    assert_eq!(parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(parent_task.pending_child_count, 2);
    assert_eq!(child_tasks.len(), 2);

    // Children come back in tool-call order: the subagent invocation for
    // slot 0, the tool-runtime child for slot 1.
    let invocation = child_tasks
        .first()
        .context("mixed batch must persist the subagent invocation")?;
    assert_eq!(invocation.kind, TaskKind::Subagent);
    assert_eq!(invocation.spawn_index, Some(0));
    let tool_child = child_tasks
        .get(1)
        .context("mixed batch must persist the tool child")?;
    assert_eq!(tool_child.kind, TaskKind::ToolRuntime);
    assert_eq!(tool_child.spawn_index, Some(1));
    assert_eq!(tool_child.status, TaskStatus::Pending);
    assert_eq!(tool_child.thread_id, thread_a());

    // The subagent half is durably linked to a fresh child-thread root.
    let linkage = invocation
        .state
        .subagent_invocation()
        .context("invocation must carry durable linkage")?;
    let child_root = stores
        .tasks
        .get(&linkage.child_root_task_id)
        .await?
        .context("child root must be persisted")?;
    assert_eq!(child_root.kind, TaskKind::RootTurn);
    assert_eq!(child_root.status, TaskStatus::Pending);
    assert_eq!(child_root.thread_id, linkage.child_thread_id);

    Ok(())
}

#[tokio::test]
async fn mixed_batch_tool_child_executes_and_fans_in() -> Result<()> {
    // The tool half of a mixed batch is an ordinary runnable child: it
    // executes in this turn and fans in, leaving the parent waiting only
    // on the subagent it spawned alongside it.
    let stores = TestStores::new();
    let (parent_task, child_tasks) =
        suspend_on_mixed_batch(&stores, subagent_then_tool_calls()).await?;
    let tool_child = child_tasks
        .get(1)
        .context("mixed batch must persist the tool child")?;

    let acquired = stores
        .tasks
        .try_acquire_task(
            &tool_child.id,
            WorkerId::from_string("w-tool"),
            LeaseId::from_string("l-tool"),
            t_plus(600),
            t_plus(10),
        )
        .await?
        .context("tool child must be runnable")?;
    let ctx = crate::worker::tool_task::resolve_tool_bootstrap(
        acquired,
        &stores.tasks,
        crate::worker::activity::ActivityBeacon::default(),
    )
    .await?;
    assert_eq!(
        ctx.tool_call.name, "bash",
        "the tool child must resolve its own slot in the continuation",
    );

    let cancel = CancellationToken::new();
    let executed = crate::worker::tool_task::execute_tool_task(
        ctx,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_call, _collector| async {
            Ok(agent_sdk_foundation::ToolResult {
                success: true,
                output: "ok".to_owned(),
                data: None,
                documents: Vec::new(),
                duration_ms: Some(1),
            })
        },
        t_plus(11),
    )
    .await?;
    let crate::worker::tool_task::ToolTaskOutcome::Completed { child, result, .. } = executed
    else {
        panic!("the tool child must run to completion in this turn");
    };
    assert_eq!(child.status, TaskStatus::Completed);
    assert!(result.success);

    let parent_after = stores
        .tasks
        .get(&parent_task.id)
        .await?
        .context("parent must still exist")?;
    assert_eq!(parent_after.status, TaskStatus::WaitingOnChildren);
    assert_eq!(
        parent_after.pending_child_count, 1,
        "the executed tool child must fan in, leaving only the subagent pending",
    );

    Ok(())
}

#[tokio::test]
async fn interleaved_mixed_batch_survives_a_steering_revert_with_original_bindings() -> Result<()> {
    // `repark_after_steering` re-derives every re-attached child's
    // spawn_index from its POSITION in the parent's child-id vector. For
    // an interleaved mixed batch (tool at slot 0, subagent at slot 1),
    // a vector grouped by child kind would hand the subagent slot 0 and
    // the tool child slot 1 — each would then resolve, execute against,
    // and report a tool call that was never its own. The bindings must
    // come back out of a revert exactly as they went in.
    let stores = TestStores::new();
    let (parent, children) =
        Box::pin(suspend_on_mixed_batch(&stores, tool_then_subagent_calls())).await?;
    assert_eq!(children.len(), 2);

    let tool_child_id = children
        .first()
        .context("slot 0 is the tool child")?
        .id
        .clone();
    let invocation_id = children
        .get(1)
        .context("slot 1 is the subagent invocation")?
        .id
        .clone();

    // Wake the parent with a steering note, then acquire it.
    stores
        .tasks
        .enqueue_steering_resume(
            &parent.id,
            vec![ContentBlock::Text {
                text: "how is it going?".into(),
            }],
            t_plus(20),
        )
        .await?
        .context("steering wake")?;
    let worker = WorkerId::from_string("worker_test");
    let lease = LeaseId::from_string("lease_test");
    let acquired = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            worker.clone(),
            lease.clone(),
            t_plus(900),
            t_plus(21),
        )
        .await?
        .context("acquire steering wake")?;
    assert!(acquired.state.is_steering_resume());

    // The bounded steering LLM round fails, so the wake reverts.
    let error = anyhow::anyhow!("provider outage during steering exchange");
    let reparked = revert_steering_wake(
        &acquired,
        &worker,
        &lease,
        &error,
        &stores.deps(),
        t_plus(25),
    )
    .await?;
    assert_eq!(reparked.status, TaskStatus::WaitingOnChildren);
    assert_eq!(reparked.pending_child_count, 2);

    // Each child still resolves the tool call it was spawned for.
    let tool_child = stores
        .tasks
        .get(&tool_child_id)
        .await?
        .context("tool child survives the revert")?;
    assert_eq!(tool_child.kind, TaskKind::ToolRuntime);
    assert_eq!(
        tool_child.spawn_index,
        Some(0),
        "the tool child must stay bound to the bash call at slot 0",
    );
    let invocation = stores
        .tasks
        .get(&invocation_id)
        .await?
        .context("subagent invocation survives the revert")?;
    assert_eq!(invocation.kind, TaskKind::Subagent);
    assert_eq!(
        invocation.spawn_index,
        Some(1),
        "the subagent invocation must stay bound to the subagent call at slot 1",
    );

    // The re-parked continuation still names the original calls in order.
    match &reparked.state {
        TaskState::WaitingOnChildren {
            continuation,
            child_ids,
            ..
        } => {
            let pending = &continuation.payload.pending_tool_calls;
            assert_eq!(pending[0].id, "call_bash");
            assert_eq!(pending[1].id, "call_sub");
            assert_eq!(child_ids, &vec![tool_child_id, invocation_id]);
        }
        other => panic!("expected WaitingOnChildren revert, got {other:?}"),
    }

    Ok(())
}
