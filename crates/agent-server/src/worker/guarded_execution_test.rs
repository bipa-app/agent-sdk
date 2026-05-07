//! Phase 5.2 regression tests for guarded tool execution.
//!
//! Covers:
//! - Replay-safe tools bypass intent persistence.
//! - Side-effecting tools persist intent before execution.
//! - Fail-closed: side-effecting tool is blocked when intent store fails.
//! - Retry safety: completed intents prevent duplicate execution.
//! - Retry safety: in-flight intents for side-effecting tools fail-closed.
//! - Retry safety: failed-before-start intents allow re-execution.
//! - Intent status transitions through the full lifecycle.
use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::tool_task::{ToolTaskOutcome, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::execution_intent::{
    ExecutionIntent, ExecutionIntentStore, GuardedExecutionDeps, InMemoryExecutionIntentStore,
    IntentStatus, OperationId, ToolEffectClass, guarded_tool_execution,
};
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
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
// Mock LLM provider
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
            id: "msg_guard_01".into(),
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
// Failing intent store for fail-closed tests
// ─────────────────────────────────────────────────────────────────────

/// An intent store that always fails on persist.
struct FailingIntentStore;

#[async_trait]
impl ExecutionIntentStore for FailingIntentStore {
    async fn persist_intent(&self, _intent: &ExecutionIntent) -> anyhow::Result<()> {
        anyhow::bail!("simulated storage failure")
    }

    async fn update_intent(&self, _intent: &ExecutionIntent) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_intent(
        &self,
        _operation_id: &OperationId,
    ) -> anyhow::Result<Option<ExecutionIntent>> {
        Ok(None)
    }

    async fn get_intent_by_task(
        &self,
        _child_task_id: &AgentTaskId,
    ) -> anyhow::Result<Option<ExecutionIntent>> {
        Ok(None)
    }
}

// ───────────────────────────────────────────────────────────���─────────
// Test helpers (mirror tool_task_test patterns)
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("thread_guard_test")
}

fn worker_id() -> WorkerId {
    WorkerId::from_string("worker_guard")
}

fn lease_id() -> LeaseId {
    LeaseId::from_string("lease_guard")
}

fn child_worker() -> WorkerId {
    WorkerId::from_string("child_guard_worker")
}

fn child_lease() -> LeaseId {
    LeaseId::from_string("child_guard_lease")
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

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — replay-safe bypass
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn replay_safe_tool_bypasses_intent_persistence() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    // Bash is Observe tier → ReplaySafe.
    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hello"}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::ReplaySafe,
        |_info, _collector| async { Ok(ToolResult::success("hello")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed, got: {outcome:?}"
    );

    // No intent should have been persisted.
    let intent = intent_store.get_intent(&op_id).await?;
    assert!(intent.is_none(), "replay-safe should not persist intent");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — side-effecting happy path
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn side_effecting_tool_persists_intent_and_completes() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    // Transfer is Confirm tier → SideEffecting.
    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async { Ok(ToolResult::success("transfer done")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed, got: {outcome:?}"
    );

    // Intent should exist and be Completed.
    let intent = intent_store
        .get_intent(&op_id)
        .await?
        .expect("intent should be persisted");
    assert_eq!(intent.status, IntentStatus::Completed);
    assert_eq!(intent.effect_class, ToolEffectClass::SideEffecting);
    assert_eq!(intent.tool_name, "transfer");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — fail-closed when intent store fails
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fail_closed_when_intent_persist_fails() -> Result<()> {
    let stores = TestStores::new();
    let failing_store = FailingIntentStore;
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let child_id = child.id.clone();
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &failing_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async {
            panic!("executor should never be called when intent persist fails");
        },
        t_plus(20),
    )
    .await?;

    // Should return Failed (not an Err — the guard handles it cleanly).
    let ToolTaskOutcome::Failed { child, error, .. } = outcome else {
        panic!("expected Failed outcome, got: {outcome:?}");
    };
    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        error.contains("fail-closed"),
        "error should mention fail-closed: {error}"
    );
    assert!(
        error.contains("simulated storage failure"),
        "error should contain root cause: {error}"
    );

    // Verify child is durably Failed in the store.
    let child_after = stores
        .tasks
        .get(&child_id)
        .await?
        .expect("child should still exist");
    assert_eq!(child_after.status, TaskStatus::Failed);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — retry: completed intent blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_intent_already_completed() -> Result<()> {
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Pre-seed a Completed intent to simulate a previous successful run.
    let tool_call = ctx.tool_call.clone();
    let mut prior = ExecutionIntent::new(
        op_id.clone(),
        ToolEffectClass::SideEffecting,
        &tool_call,
        ctx.task_id.clone(),
        t_plus(10),
    );
    prior.mark_started(t_plus(11));
    prior.mark_completed(t_plus(12));
    intent_store.persist_intent(&prior).await?;
    intent_store.update_intent(&prior).await?;

    // Attempt guarded execution — should fail with "already completed".
    let result = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async {
            panic!("executor should not be called for already-completed operation");
        },
        t_plus(20),
    )
    .await;

    assert!(result.is_err(), "expected error for completed retry");
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("already completed"),
        "error should mention already completed: {err}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — retry: in-flight intent blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_intent_ambiguous_in_flight() -> Result<()> {
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Pre-seed a Started intent to simulate a crashed mid-execution.
    let tool_call = ctx.tool_call.clone();
    let mut prior = ExecutionIntent::new(
        op_id.clone(),
        ToolEffectClass::SideEffecting,
        &tool_call,
        ctx.task_id.clone(),
        t_plus(10),
    );
    prior.mark_started(t_plus(11));
    intent_store.persist_intent(&prior).await?;
    intent_store.update_intent(&prior).await?;

    // Attempt guarded execution — should fail with "ambiguous in-flight".
    let result = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async {
            panic!("executor should not be called for ambiguous in-flight");
        },
        t_plus(20),
    )
    .await;

    assert!(result.is_err(), "expected error for ambiguous retry");
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("ambiguous in-flight"),
        "error should mention ambiguous: {err}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — retry: failed side-effecting intent blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_side_effecting_intent_failed() -> Result<()> {
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Pre-seed a Failed intent — the executor ran and returned an error,
    // so it may have caused partial side effects.
    let tool_call = ctx.tool_call.clone();
    let mut prior = ExecutionIntent::new(
        op_id.clone(),
        ToolEffectClass::SideEffecting,
        &tool_call,
        ctx.task_id.clone(),
        t_plus(10),
    );
    prior.mark_failed("network timeout after partial debit", t_plus(11));
    intent_store.persist_intent(&prior).await?;
    intent_store.update_intent(&prior).await?;

    // Attempt guarded execution — should fail with "ambiguous in-flight".
    let result = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async {
            panic!("executor should not be called for failed side-effecting operation");
        },
        t_plus(20),
    )
    .await;

    assert!(
        result.is_err(),
        "expected error for failed side-effecting retry"
    );
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("ambiguous in-flight"),
        "error should mention ambiguous: {err}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — retry: failed replay-safe intent allows re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_allowed_when_replay_safe_intent_failed() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hello"}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Pre-seed a Failed intent for a replay-safe tool — safe to retry
    // since replay-safe tools have no side effects.
    let tool_call = ctx.tool_call.clone();
    let mut prior = ExecutionIntent::new(
        op_id.clone(),
        ToolEffectClass::ReplaySafe,
        &tool_call,
        ctx.task_id.clone(),
        t_plus(10),
    );
    prior.mark_failed("transient error", t_plus(11));
    intent_store.persist_intent(&prior).await?;
    intent_store.update_intent(&prior).await?;

    // Attempt guarded execution — should succeed because ReplaySafe
    // bypasses intent persistence entirely.
    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::ReplaySafe,
        |_info, _collector| async { Ok(ToolResult::success("retry succeeded")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed after retry, got: {outcome:?}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — intent lifecycle: failure records error
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn side_effecting_failure_records_intent_as_failed() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 999}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async { Err(anyhow::anyhow!("insufficient funds")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Failed { .. }),
        "expected Failed, got: {outcome:?}"
    );

    // Intent should be Failed with the error message.
    let intent = intent_store
        .get_intent(&op_id)
        .await?
        .expect("intent should be persisted");
    assert_eq!(intent.status, IntentStatus::Failed);
    assert!(
        intent
            .error
            .as_deref()
            .unwrap()
            .contains("insufficient funds"),
        "intent error should contain root cause: {:?}",
        intent.error
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5.2 — cancellation records intent as failed
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cancellation_records_intent_as_failed() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 50}),
        )],
    )
    .await?;

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Cancel before execution.
    cancel.cancel();

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_info, _collector| async { Ok(ToolResult::success("should not reach")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Cancelled),
        "expected Cancelled, got: {outcome:?}"
    );

    // guarded_tool_execution persists intent *before* delegating to
    // execute_tool_task, so the intent must exist even when the
    // cancel token was pre-set. The post-execution handler updates
    // it to Failed.
    let intent = intent_store
        .get_intent(&op_id)
        .await?
        .expect("intent must be persisted for SideEffecting tool");
    assert_eq!(intent.status, IntentStatus::Failed);

    Ok(())
}
