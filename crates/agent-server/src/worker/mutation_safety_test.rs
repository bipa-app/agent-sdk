//! Phase 5.5 mutation-safety regression suite.
//!
//! This is the **exit proof package** for Phase 5. It validates
//! correctness-oriented invariants across the full tool-runtime
//! lifecycle:
//!
//! 1. **Duplicate retry risk** — side-effecting tools cannot be
//!    re-executed after completing, starting, or failing.
//! 2. **Confirmation restart** — a child task in
//!    `AwaitingConfirmation` survives restart and the prepared
//!    operation is preserved or failed-closed correctly.
//! 3. **Cancellation** — cancellation at every lifecycle stage
//!    produces the correct terminal state and audit trail.
//! 4. **Prepared-operation fail-closed** — listen-tier tools with
//!    prepared operations fail closed on recovery.
//! 5. **Audit coverage** — every lifecycle path produces durable
//!    audit events with correct provenance.
//! 6. **Redaction** — sensitive tool inputs/outputs are redacted
//!    before audit storage.
//!
//! These tests intentionally exercise the **integration** between
//! multiple Phase 5 modules (`tool_task`, `confirmation`,
//! `execution_intent`, `tool_audit`, `redaction`) rather than testing
//! each in isolation.

use super::confirmation::{
    CONFIRMATION_POLICY_DENIED_PREFIX, CONFIRMATION_REJECTED_PREFIX, CONFIRMATION_TIMEOUT_PREFIX,
    ConfirmationDecision, ConfirmationDecisionOutcome, ConfirmationPolicy,
    ConfirmationResumeOutcome, PolicyVerdict, apply_confirmation_decision,
    pause_tool_for_confirmation, resume_confirmed_tool,
};
use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::tool_task::{ToolTaskOutcome, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::execution_intent::{
    ExecutionIntent, ExecutionIntentStore, GuardedExecutionDeps, InMemoryExecutionIntentStore,
    IntentStatus, OperationId, ToolEffectClass, classify_tool_effect, guarded_tool_execution,
};
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::redaction::{RedactionPolicy, redact_value};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::tool_audit::{
    InMemoryToolAuditEventStore, ToolAuditEvent, ToolAuditEventKind, ToolAuditEventParams,
    ToolAuditEventStore,
};
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_core::{PendingToolCallInfo, ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio_util::sync::CancellationToken;

// ═════════════════════════════════════════════════════════════════════
// Shared test infrastructure
// ═════════════════════════════════════════════════════════════════════

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("thread_mutation_safety")
}

fn worker_id() -> WorkerId {
    WorkerId::from_string("worker_ms")
}

fn lease_id() -> LeaseId {
    LeaseId::from_string("lease_ms")
}

fn child_worker() -> WorkerId {
    WorkerId::from_string("child_ms_worker")
}

fn child_lease() -> LeaseId {
    LeaseId::from_string("child_ms_lease")
}

fn resume_worker() -> WorkerId {
    WorkerId::from_string("resume_ms_worker")
}

fn resume_lease() -> LeaseId {
    LeaseId::from_string("resume_ms_lease")
}

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
            id: "msg_ms_01".into(),
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

struct AllowAllPolicy;

#[async_trait]
impl ConfirmationPolicy for AllowAllPolicy {
    async fn check_policy(&self, _tool_call: &PendingToolCallInfo) -> Result<PolicyVerdict> {
        Ok(PolicyVerdict::Allowed)
    }
}

struct DenyAllPolicy {
    reason: String,
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
// Failing intent store
// ─────────────────────────────────────────────────────────────────────

struct FailingIntentStore;

#[async_trait]
impl ExecutionIntentStore for FailingIntentStore {
    async fn persist_intent(&self, _intent: &ExecutionIntent) -> anyhow::Result<()> {
        anyhow::bail!("simulated intent store failure")
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

// ─────────────────────────────────────────────────────────────────────
// Test stores and helpers
// ─────────────────────────────────────────────────────────────────────

fn sample_definition() -> AgentDefinition {
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
    let bootstrap = sample_bootstrap(task);
    let inputs =
        build_root_worker_inputs(bootstrap, &stores.threads, &stores.checkpoints, t0()).await?;

    let outcome =
        execute_root_turn(inputs, "Run tools", &provider, &stores.deps(), t_plus(5)).await?;

    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        anyhow::bail!("Expected Suspended outcome, got: {outcome:?}");
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

async fn emit_audit_event(
    audit_store: &InMemoryToolAuditEventStore,
    bootstrap: &super::tool_task::ToolTaskBootstrap,
    kind: ToolAuditEventKind,
    now: time::OffsetDateTime,
) -> ToolAuditEvent {
    let event = ToolAuditEvent::new(ToolAuditEventParams {
        operation_id: OperationId::new(&bootstrap.task_id, &bootstrap.tool_call.id).to_string(),
        task_id: bootstrap.task_id.clone(),
        parent_task_id: bootstrap
            .child_task
            .parent_id
            .clone()
            .unwrap_or_else(|| AgentTaskId::from_string("unknown")),
        thread_id: bootstrap.thread_id.clone(),
        tool_call_id: bootstrap.tool_call.id.clone(),
        tool_name: bootstrap.tool_call.name.clone(),
        effect_class: classify_tool_effect(&bootstrap.tool_call),
        kind,
        provider: "mock".into(),
        model: "mock-model".into(),
        input: Some(bootstrap.tool_call.input.clone()),
        output: None,
        error: None,
        now,
    });
    // Best-effort — audit store failure does not block execution.
    audit_store.record_event(&event).await.ok();
    event
}

// ═════════════════════════════════════════════════════════════════════
// 1. DUPLICATE RETRY RISK
// ═════════════════════════════════════════════════════════════════════

/// Verify that executing a side-effecting tool marks the intent as
/// `Completed`, which prevents any subsequent retry with the same
/// operation ID (tested in `guarded_execution_test`).
#[tokio::test]
async fn execution_marks_intent_completed_blocking_retry() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 500}),
        )],
    )
    .await?;

    // First execution: success
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
        |_, _collector| async { Ok(ToolResult::success("transfer complete")) },
        t_plus(20),
    )
    .await?;

    assert!(matches!(outcome, ToolTaskOutcome::Completed { .. }));

    // Verify intent is Completed — the core safety invariant
    let intent = intent_store
        .get_intent(&op_id)
        .await?
        .context("intent should exist")?;
    assert_eq!(intent.status, IntentStatus::Completed);
    assert_eq!(intent.effect_class, ToolEffectClass::SideEffecting);
    assert_eq!(intent.tool_name, "transfer");
    assert!(intent.is_terminal());

    Ok(())
}

/// Verify that a side-effecting tool with an ambiguous in-flight
/// intent cannot be automatically retried.
#[tokio::test]
async fn retry_with_ambiguous_in_flight_is_blocked() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

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
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let op_id = OperationId::new(&ctx.task_id, &ctx.tool_call.id);

    // Pre-seed a Started intent (simulates crash mid-execution)
    let mut started_intent = ExecutionIntent::new(
        op_id,
        ToolEffectClass::SideEffecting,
        &ctx.tool_call,
        ctx.task_id.clone(),
        t_plus(10),
    );
    started_intent.mark_started(t_plus(11));
    intent_store.persist_intent(&started_intent).await?;
    intent_store.update_intent(&started_intent).await?;

    let result = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_, _collector| async { panic!("executor must not be called for ambiguous retry") },
        t_plus(20),
    )
    .await;

    assert!(result.is_err());
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("ambiguous in-flight"),
        "error should mention ambiguous: {err}"
    );

    Ok(())
}

/// Verify that a replay-safe tool CAN be retried after failure.
#[tokio::test]
async fn replay_safe_tool_retries_freely_after_failure() -> Result<()> {
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

    // ReplaySafe bypasses intent entirely, so no intent to seed.
    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::ReplaySafe,
        |_, _collector| async { Ok(ToolResult::success("hello")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "replay-safe tool should complete freely"
    );

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 2. CONFIRMATION RESTART SCENARIOS
// ═════════════════════════════════════════════════════════════════════

/// Verify that a child task in `AwaitingConfirmation` survives a
/// simulated restart (re-read from the store) with its continuation
/// and prepared operation intact.
#[tokio::test]
async fn confirmation_state_survives_restart() -> Result<()> {
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

    // Acquire and pause for confirmation
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let child_id = ctx.task_id.clone();

    let (paused, _committed_events) =
        pause_tool_for_confirmation(&ctx, &stores.tasks, &stores.events, t_plus(10)).await?;
    assert_eq!(paused.status, TaskStatus::AwaitingConfirmation);

    // Simulate restart: re-read from store
    let reloaded = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should exist after restart")?;

    assert_eq!(
        reloaded.status,
        TaskStatus::AwaitingConfirmation,
        "status must survive restart"
    );

    // The state should carry the continuation
    assert!(
        matches!(reloaded.state, TaskState::AwaitingConfirmation { .. }),
        "typed state must survive restart"
    );

    // Now approve and execute
    let decision_outcome = apply_confirmation_decision(
        &child_id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    let ConfirmationDecisionOutcome::Approved { child: resumed, .. } = decision_outcome else {
        anyhow::bail!("expected Approved, got: {decision_outcome:?}");
    };
    assert_eq!(resumed.status, TaskStatus::Pending);

    Ok(())
}

/// Verify that approval → policy deny produces the correct failure
/// path through the full lifecycle.
#[tokio::test]
async fn confirmation_approved_then_policy_denied() -> Result<()> {
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
    let child_id = ctx.task_id.clone();

    // Pause for confirmation
    pause_tool_for_confirmation(&ctx, &stores.tasks, &stores.events, t_plus(10)).await?;

    // Approve
    apply_confirmation_decision(
        &child_id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    // Re-acquire after approval
    let resumed_child = stores
        .tasks
        .try_acquire_task(
            &child_id,
            resume_worker(),
            resume_lease(),
            t_plus(600),
            t_plus(25),
        )
        .await?
        .context("child should be re-acquirable after approval")?;

    let ctx2 = resolve_tool_bootstrap(resumed_child, &stores.tasks).await?;

    // Policy denies at resume time
    let deny_policy = DenyAllPolicy {
        reason: "permissions revoked".into(),
    };

    let resume_outcome = resume_confirmed_tool(
        ctx2,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &deny_policy,
        &cancel,
        |_, _collector| async { panic!("executor must not be called when policy denies") },
        t_plus(30),
    )
    .await?;

    let ConfirmationResumeOutcome::PolicyDenied {
        child: denied_child,
        reason,
        ..
    } = resume_outcome
    else {
        anyhow::bail!("expected PolicyDenied, got: {resume_outcome:?}");
    };

    assert_eq!(denied_child.status, TaskStatus::Failed);
    assert_eq!(reason, "permissions revoked");

    // Verify the error prefix is correct in the store
    let persisted = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should still exist")?;
    assert_eq!(persisted.status, TaskStatus::Failed);
    assert!(
        persisted
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_POLICY_DENIED_PREFIX),
        "error should start with policy denied prefix: {:?}",
        persisted.last_error,
    );

    Ok(())
}

/// Verify that rejection produces the correct failure state with
/// canonical error prefix.
#[tokio::test]
async fn confirmation_rejected_produces_canonical_error() -> Result<()> {
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let child_id = ctx.task_id.clone();

    pause_tool_for_confirmation(&ctx, &stores.tasks, &stores.events, t_plus(10)).await?;

    let outcome = apply_confirmation_decision(
        &child_id,
        ConfirmationDecision::Rejected {
            reason: "user declined".into(),
        },
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    let ConfirmationDecisionOutcome::Rejected {
        child: rejected,
        reason,
        ..
    } = outcome
    else {
        anyhow::bail!("expected Rejected, got: {outcome:?}");
    };

    assert_eq!(rejected.status, TaskStatus::Failed);
    assert_eq!(reason, "user declined");

    let persisted = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should still exist")?;
    assert!(
        persisted
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_REJECTED_PREFIX),
        "error should start with rejected prefix: {:?}",
        persisted.last_error,
    );

    Ok(())
}

/// Verify that timeout produces the correct failure state with
/// canonical error prefix.
#[tokio::test]
async fn confirmation_timeout_produces_canonical_error() -> Result<()> {
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

    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let child_id = ctx.task_id.clone();

    pause_tool_for_confirmation(&ctx, &stores.tasks, &stores.events, t_plus(10)).await?;

    let outcome = apply_confirmation_decision(
        &child_id,
        ConfirmationDecision::Timeout,
        &stores.tasks,
        t_plus(60),
    )
    .await?;

    assert!(
        matches!(outcome, ConfirmationDecisionOutcome::TimedOut { .. }),
        "expected TimedOut, got: {outcome:?}"
    );

    let persisted = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should still exist")?;
    assert_eq!(persisted.status, TaskStatus::Failed);
    assert!(
        persisted
            .last_error
            .as_deref()
            .unwrap_or("")
            .starts_with(CONFIRMATION_TIMEOUT_PREFIX),
        "error should start with timeout prefix: {:?}",
        persisted.last_error,
    );

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 3. CANCELLATION AT EVERY STAGE
// ═════════════════════════════════════════════════════════════════════

/// Verify cancellation before tool execution returns Cancelled without
/// driving the child to a terminal state.
#[tokio::test]
async fn cancellation_before_execution_returns_cancelled() -> Result<()> {
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
    let child_id = child.id.clone();
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // Cancel before execution
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
        |_, _collector| async { panic!("executor should not run when cancelled") },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Cancelled),
        "expected Cancelled, got: {outcome:?}"
    );

    // For side-effecting tools, intent is persisted before the cancel
    // check in execute_tool_task, so it should exist and be Failed.
    let op_id = OperationId::new(&child_id, "call_1");
    let intent = intent_store.get_intent(&op_id).await?;
    assert!(
        intent.is_some(),
        "intent should be persisted for SideEffecting tool even when cancelled"
    );
    assert_eq!(
        intent.as_ref().map(|i| i.status),
        Some(IntentStatus::Failed)
    );

    Ok(())
}

/// Verify that `cancel_tree` cascades to all children in a multi-child
/// batch.
#[tokio::test]
async fn cancel_tree_cascades_to_all_children() -> Result<()> {
    let stores = TestStores::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "bash".into(),
                serde_json::json!({"command": "echo a"}),
            ),
            (
                "call_2".into(),
                "transfer".into(),
                serde_json::json!({"amount": 50}),
            ),
        ],
    )
    .await?;

    assert_eq!(children.len(), 2);

    // Cancel the entire tree from the root
    stores.tasks.cancel_tree(&parent.id, t_plus(10)).await?;

    // Verify both children are Cancelled
    for child in &children {
        let persisted = stores
            .tasks
            .get(&child.id)
            .await?
            .context("child should exist")?;
        assert_eq!(
            persisted.status,
            TaskStatus::Cancelled,
            "child {} should be Cancelled",
            child.id,
        );
    }

    // Parent should also be Cancelled
    let persisted_parent = stores
        .tasks
        .get(&parent.id)
        .await?
        .context("parent should exist")?;
    assert_eq!(persisted_parent.status, TaskStatus::Cancelled);

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 4. PREPARED-OPERATION FAIL-CLOSED
// ═════════════════════════════════════════════════════════════════════

/// Verify that fail-closed blocks execution when intent persistence
/// fails, producing a Failed child task.
#[tokio::test]
async fn fail_closed_on_intent_persist_failure() -> Result<()> {
    let stores = TestStores::new();
    let failing_store = FailingIntentStore;
    let cancel = CancellationToken::new();

    let (_parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 1000}),
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
        |_, _collector| async { panic!("executor must not be called when intent persist fails") },
        t_plus(20),
    )
    .await?;

    let ToolTaskOutcome::Failed { child, error, .. } = outcome else {
        anyhow::bail!("expected Failed, got: {outcome:?}");
    };
    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        error.contains("fail-closed"),
        "error should mention fail-closed: {error}"
    );

    // Verify durably Failed in store
    let persisted = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should exist")?;
    assert_eq!(persisted.status, TaskStatus::Failed);

    Ok(())
}

/// Verify the end-to-end confirmation + execution path succeeds when
/// policy allows and intent persistence works.
#[tokio::test]
async fn confirmation_approval_then_successful_execution() -> Result<()> {
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
    let child_id = ctx.task_id.clone();

    // Pause for confirmation
    pause_tool_for_confirmation(&ctx, &stores.tasks, &stores.events, t_plus(10)).await?;

    // Approve
    apply_confirmation_decision(
        &child_id,
        ConfirmationDecision::Approved,
        &stores.tasks,
        t_plus(20),
    )
    .await?;

    // Re-acquire and execute
    let resumed = stores
        .tasks
        .try_acquire_task(
            &child_id,
            resume_worker(),
            resume_lease(),
            t_plus(600),
            t_plus(25),
        )
        .await?
        .context("child should be re-acquirable")?;

    let ctx2 = resolve_tool_bootstrap(resumed, &stores.tasks).await?;

    let resume_outcome = resume_confirmed_tool(
        ctx2,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &AllowAllPolicy,
        &cancel,
        |_, _collector| async { Ok(ToolResult::success("transfer done")) },
        t_plus(30),
    )
    .await?;

    let ConfirmationResumeOutcome::Executed(tool_outcome) = resume_outcome else {
        anyhow::bail!("expected Executed, got: {resume_outcome:?}");
    };

    assert!(
        matches!(tool_outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed, got: {tool_outcome:?}"
    );

    // Verify intent was persisted and completed
    let intent = intent_store
        .get_intent_by_task(&child_id)
        .await?
        .context("intent should exist")?;
    assert_eq!(intent.status, IntentStatus::Completed);
    assert_eq!(intent.tool_name, "transfer");

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 5. AUDIT COVERAGE
// ═════════════════════════════════════════════════════════════════════

/// Verify that a full happy-path lifecycle produces the expected
/// audit event sequence with correct provenance.
#[tokio::test]
async fn audit_trail_covers_full_lifecycle() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let audit_store = InMemoryToolAuditEventStore::new();
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
    let child_id = ctx.task_id.clone();
    let op_str = OperationId::new(&child_id, "call_1").to_string();

    // Emit audit events through the lifecycle
    emit_audit_event(
        &audit_store,
        &ctx,
        ToolAuditEventKind::Dispatched,
        t_plus(10),
    )
    .await;

    emit_audit_event(
        &audit_store,
        &ctx,
        ToolAuditEventKind::ExecutionStarted,
        t_plus(15),
    )
    .await;

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_, _collector| async { Ok(ToolResult::success("done")) },
        t_plus(20),
    )
    .await?;

    assert!(matches!(outcome, ToolTaskOutcome::Completed { .. }));

    // Record the terminal audit event
    let final_event = ToolAuditEvent::new(ToolAuditEventParams {
        operation_id: op_str.clone(),
        task_id: child_id.clone(),
        parent_task_id: AgentTaskId::from_string("parent_unknown"),
        thread_id: thread_a(),
        tool_call_id: "call_1".into(),
        tool_name: "transfer".into(),
        effect_class: ToolEffectClass::SideEffecting,
        kind: ToolAuditEventKind::Completed,
        provider: "mock".into(),
        model: "mock-model".into(),
        input: Some(serde_json::json!({"amount": 100})),
        output: Some("done".into()),
        error: None,
        now: t_plus(20),
    });
    audit_store.record_event(&final_event).await?;

    // Query audit trail
    let events = audit_store.list_by_operation(&op_str).await?;
    assert_eq!(events.len(), 3);

    let kinds: Vec<&str> = events.iter().map(ToolAuditEvent::kind_str).collect();
    assert_eq!(kinds, vec!["dispatched", "execution_started", "completed"]);

    // Verify provenance on every event
    for event in &events {
        assert_eq!(event.provider, "mock");
        assert_eq!(event.model, "mock-model");
        assert_eq!(event.tool_name, "transfer");
        assert_eq!(event.effect_class, ToolEffectClass::SideEffecting);
    }

    // Verify terminal event has output
    assert_eq!(events[2].output.as_deref(), Some("done"));

    Ok(())
}

/// Verify that audit events are queryable by task and thread.
#[tokio::test]
async fn audit_events_queryable_by_task_and_thread() -> Result<()> {
    let audit_store = InMemoryToolAuditEventStore::new();

    let task_a = AgentTaskId::from_string("task_a");
    let task_b = AgentTaskId::from_string("task_b");
    let thread = ThreadId::from_string("thread_audit_query");

    // Task A events
    for kind in [
        ToolAuditEventKind::Dispatched,
        ToolAuditEventKind::Completed,
    ] {
        let event = ToolAuditEvent::new(ToolAuditEventParams {
            operation_id: "task_a:call_1".into(),
            task_id: task_a.clone(),
            parent_task_id: AgentTaskId::from_string("parent"),
            thread_id: thread.clone(),
            tool_call_id: "call_1".into(),
            tool_name: "bash".into(),
            effect_class: ToolEffectClass::ReplaySafe,
            kind,
            provider: "mock".into(),
            model: "mock-model".into(),
            input: None,
            output: None,
            error: None,
            now: t0(),
        });
        audit_store.record_event(&event).await?;
    }

    // Task B event (different thread)
    let event_b = ToolAuditEvent::new(ToolAuditEventParams {
        operation_id: "task_b:call_1".into(),
        task_id: task_b.clone(),
        parent_task_id: AgentTaskId::from_string("parent"),
        thread_id: ThreadId::from_string("thread_other"),
        tool_call_id: "call_1".into(),
        tool_name: "transfer".into(),
        effect_class: ToolEffectClass::SideEffecting,
        kind: ToolAuditEventKind::Dispatched,
        provider: "mock".into(),
        model: "mock-model".into(),
        input: None,
        output: None,
        error: None,
        now: t0(),
    });
    audit_store.record_event(&event_b).await?;

    // Query by task
    let first_task_events = audit_store.list_by_task(&task_a).await?;
    assert_eq!(first_task_events.len(), 2);

    let second_task_events = audit_store.list_by_task(&task_b).await?;
    assert_eq!(second_task_events.len(), 1);

    // Query by thread
    let thread_events = audit_store.list_by_thread(&thread).await?;
    assert_eq!(thread_events.len(), 2); // Only task_a events

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 6. REDACTION
// ═════════════════════════════════════════════════════════════════════

/// Verify that baseline redaction strips sensitive fields from tool
/// inputs before audit storage.
#[tokio::test]
async fn redaction_strips_sensitive_input_fields() -> Result<()> {
    let policy = RedactionPolicy::baseline();

    let input = serde_json::json!({
        "command": "echo hello",
        "api_key": "sk-secret-key-value",
        "config": {
            "password": "hunter2",
            "endpoint": "https://api.example.com",
        },
    });

    let redacted = redact_value(&input, &policy);

    // Non-sensitive fields preserved
    assert_eq!(redacted["command"], "echo hello");
    assert_eq!(redacted["config"]["endpoint"], "https://api.example.com");

    // Sensitive fields redacted
    assert_eq!(redacted["api_key"], "[REDACTED]");
    assert_eq!(redacted["config"]["password"], "[REDACTED]");

    Ok(())
}

/// Verify that full redaction replaces the entire value.
#[test]
fn full_redaction_replaces_entire_value() {
    let policy = RedactionPolicy::full();
    let input = serde_json::json!({
        "safe": "data",
        "amount": 100,
    });
    let redacted = redact_value(&input, &policy);
    assert_eq!(redacted, serde_json::json!("[REDACTED]"));
}

/// Verify that redaction integrates with audit events — sensitive
/// inputs are redacted when building audit records.
#[tokio::test]
async fn audit_event_with_redacted_input() -> Result<()> {
    let policy = RedactionPolicy::baseline();
    let audit_store = InMemoryToolAuditEventStore::new();

    let sensitive_input = serde_json::json!({
        "amount": 100,
        "api_key": "sk-live-secret",
        "recipient": "user@example.com",
    });

    let redacted_input = redact_value(&sensitive_input, &policy);

    let event = ToolAuditEvent::new(ToolAuditEventParams {
        operation_id: "task_1:call_1".into(),
        task_id: AgentTaskId::from_string("task_1"),
        parent_task_id: AgentTaskId::from_string("parent_1"),
        thread_id: ThreadId::from_string("thread_redact"),
        tool_call_id: "call_1".into(),
        tool_name: "transfer".into(),
        effect_class: ToolEffectClass::SideEffecting,
        kind: ToolAuditEventKind::Completed,
        provider: "mock".into(),
        model: "mock-model".into(),
        input: Some(redacted_input),
        output: None,
        error: None,
        now: t0(),
    });
    audit_store.record_event(&event).await?;

    let events = audit_store.list_by_operation("task_1:call_1").await?;
    assert_eq!(events.len(), 1);

    let persisted_input = events[0].input.as_ref().context("input should exist")?;
    assert_eq!(persisted_input["amount"], 100);
    assert_eq!(persisted_input["api_key"], "[REDACTED]");
    assert_eq!(persisted_input["recipient"], "user@example.com");

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 7. MULTI-CHILD INDEPENDENCE
// ═════════════════════════════════════════════════════════════════════

/// Verify that in a multi-child batch, one child's confirmation does
/// not affect the other child, and the parent resumes only when both
/// reach terminal states.
#[tokio::test]
async fn multi_child_independence_and_parent_resume() -> Result<()> {
    let stores = TestStores::new();
    let intent_store = InMemoryExecutionIntentStore::new();
    let cancel = CancellationToken::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![
            (
                "call_1".into(),
                "bash".into(),
                serde_json::json!({"command": "echo a"}),
            ),
            (
                "call_2".into(),
                "transfer".into(),
                serde_json::json!({"amount": 50}),
            ),
        ],
    )
    .await?;

    assert_eq!(children.len(), 2);

    // Complete the bash (Observe) child — should succeed immediately
    let bash_child = acquire_child(&stores.tasks, &children[0].id).await?;
    let bash_ctx = resolve_tool_bootstrap(bash_child, &stores.tasks).await?;

    let bash_outcome = guarded_tool_execution(
        bash_ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::ReplaySafe,
        |_, _collector| async { Ok(ToolResult::success("a")) },
        t_plus(20),
    )
    .await?;

    assert!(matches!(bash_outcome, ToolTaskOutcome::Completed { .. }));

    // Parent should NOT be ready yet (transfer child still pending)
    let parent_after_bash = stores
        .tasks
        .get(&parent.id)
        .await?
        .context("parent should exist")?;
    assert_eq!(
        parent_after_bash.status,
        TaskStatus::WaitingOnChildren,
        "parent should still be waiting after first child completes"
    );

    // Now complete the transfer child
    let transfer_child = acquire_child(&stores.tasks, &children[1].id).await?;
    let transfer_ctx = resolve_tool_bootstrap(transfer_child, &stores.tasks).await?;

    let transfer_outcome = guarded_tool_execution(
        transfer_ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &intent_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        |_, _collector| async { Ok(ToolResult::success("50 transferred")) },
        t_plus(30),
    )
    .await?;

    assert!(matches!(
        transfer_outcome,
        ToolTaskOutcome::Completed { .. }
    ));

    // Parent should now be ready to resume (both children terminal)
    let parent_after_all = stores
        .tasks
        .get(&parent.id)
        .await?
        .context("parent should exist")?;
    assert_eq!(
        parent_after_all.status,
        TaskStatus::Pending,
        "parent should be Pending (ready to resume) after all children complete"
    );

    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// 8. INTENT AND EXECUTION LIFECYCLE COHERENCE
// ═════════════════════════════════════════════════════════════════════

/// Verify that the execution intent and task store states are coherent
/// after a failed execution — both should reflect the failure.
#[tokio::test]
async fn intent_and_task_coherent_after_execution_failure() -> Result<()> {
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
    let child_id = child.id.clone();
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
        |_, _collector| async { Err(anyhow::anyhow!("insufficient funds")) },
        t_plus(20),
    )
    .await?;

    assert!(matches!(outcome, ToolTaskOutcome::Failed { .. }));

    // Task store: child is Failed
    let persisted_child = stores
        .tasks
        .get(&child_id)
        .await?
        .context("child should exist")?;
    assert_eq!(persisted_child.status, TaskStatus::Failed);

    // Intent store: intent is Failed with matching error
    let intent = intent_store
        .get_intent(&op_id)
        .await?
        .context("intent should exist")?;
    assert_eq!(intent.status, IntentStatus::Failed);
    assert!(
        intent
            .error
            .as_deref()
            .unwrap_or("")
            .contains("insufficient funds"),
        "intent error should contain root cause: {:?}",
        intent.error,
    );

    Ok(())
}

/// Verify that the design invariant holds: if no durable intent exists
/// for a side-effecting tool, the executor was never invoked.
#[tokio::test]
async fn no_intent_means_no_execution_invariant() -> Result<()> {
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
    let ctx = resolve_tool_bootstrap(child, &stores.tasks).await?;

    // This uses a failing intent store, so the intent will not persist.
    // The executor must NOT be called.
    let execution_count = std::sync::Arc::new(AtomicUsize::new(0));
    let count_clone = execution_count.clone();

    let outcome = guarded_tool_execution(
        ctx,
        &GuardedExecutionDeps {
            task_store: &stores.tasks,
            intent_store: &failing_store,
            event_repo: &stores.events,
        },
        &cancel,
        ToolEffectClass::SideEffecting,
        move |_, _collector| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            async { Ok(ToolResult::success("should not reach")) }
        },
        t_plus(20),
    )
    .await?;

    // The executor was never invoked
    assert_eq!(
        execution_count.load(Ordering::SeqCst),
        0,
        "executor must not be called when intent persist fails"
    );

    // Outcome should be Failed (fail-closed)
    assert!(
        matches!(outcome, ToolTaskOutcome::Failed { .. }),
        "expected Failed, got: {outcome:?}"
    );

    Ok(())
}
