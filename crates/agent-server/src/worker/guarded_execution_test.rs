//! Regression tests for tool calls run as effects ([`run_tool_effect`]).
//!
//! Covers:
//! - Replay-safe tools write no effect record.
//! - Side-effecting tools commit their record before execution.
//! - Fail-closed: a side-effecting tool is blocked when its record cannot
//!   be claimed.
//! - Retry safety: a completed record prevents duplicate execution.
//! - Retry safety: a started or failed record settles the call as
//!   interrupted without running it.
//! - Retry safety: a replay-safe call runs again whatever its record says.
//! - A worker that lost its lease neither claims nor settles.
use crate::worker::activity::ActivityBeacon;
use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::tool_effect::{run_tool_effect, tool_intent};
use super::tool_task::{ToolTaskBootstrap, ToolTaskOutcome, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::effect::{
    EffectKey, EffectRecord, EffectState, EffectSubject, EffectWrite, TOOL_INTERRUPTED_PREFIX,
};
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::transition::{Committed, DurableStore, Guard, Rejected, Transition, Write};
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{ThreadId, ToolReplay, ToolResult, ToolTier};
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
                served_speed: None,
                input_tokens: 100,
                output_tokens: 50,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
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
// A store that refuses every effect start, for fail-closed tests
// ─────────────────────────────────────────────────────────────────────

/// Commits through `inner`, except a transition that starts an effect,
/// which fails as a store error would.
struct FailingEffectStart<'a> {
    inner: &'a InMemoryAgentTaskStore,
}

#[async_trait]
impl DurableStore for FailingEffectStart<'_> {
    async fn commit(
        &self,
        transition: Transition,
        now: time::OffsetDateTime,
    ) -> std::result::Result<Committed, Rejected> {
        if transition
            .effect_writes()
            .any(|write| matches!(write, EffectWrite::Start(_)))
        {
            return Err(Rejected::Store(anyhow::anyhow!(
                "simulated storage failure"
            )));
        }
        self.inner.commit(transition, now).await
    }

    async fn effect(&self, key: &EffectKey) -> Result<Option<EffectRecord>> {
        self.inner.effect(key).await
    }

    async fn effects_for_task(&self, owner: &AgentTaskId) -> Result<Vec<EffectRecord>> {
        self.inner.effects_for_task(owner).await
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
                concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
                replay: agent_sdk_foundation::ToolReplay::Unsafe,
            },
            Tool {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                display_name: "Read".into(),
                tier: ToolTier::Observe,
                concurrency: agent_sdk_foundation::ToolConcurrency::ReadOnly,
                replay: ToolReplay::Safe,
            },
            Tool {
                name: "transfer".into(),
                description: "Transfer funds".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"amount": {"type": "number"}}}),
                display_name: "Transfer".into(),
                tier: ToolTier::Confirm,
                concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
                replay: agent_sdk_foundation::ToolReplay::Unsafe,
            },
        ],
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
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
        let threads = InMemoryThreadStore::new();
        let events = InMemoryEventRepository::new();
        let outbox = crate::journal::outbox::InMemoryOutboxStore::new();
        let attempts = InMemoryTurnAttemptStore::new();
        Self {
            tasks: InMemoryAgentTaskStore::new()
                .with_cancellation_markers(crate::journal::CancellationMarkerSink {
                    event_repo: Arc::new(events.clone()),
                    outbox_store: Arc::new(outbox.clone()),
                    thread_store: Arc::new(threads.clone()),
                })
                .with_spawn_event_committer(Arc::new(
                    crate::journal::store::InMemoryAtomicSpawnEventCommitter::new(
                        events.clone(),
                        outbox,
                    ),
                ))
                .with_attempt_store(Arc::new(attempts.clone())),
            threads,
            messages: InMemoryMessageProjectionStore::new(),
            attempts,
            checkpoints: InMemoryCheckpointStore::new(),
            events,
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
            compaction_artifact_store: None,
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
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

/// Commit the record a previous attempt of `ctx`'s call left: started,
/// then ended with `end` when given.
async fn seed_effect(
    store: &InMemoryAgentTaskStore,
    ctx: &ToolTaskBootstrap,
    replay: ToolReplay,
    end: Option<EffectState>,
) -> Result<EffectKey> {
    let mut intent = tool_intent(ctx);
    intent.replay = replay;
    let key = intent.key.clone();
    store
        .commit(
            Transition {
                guards: vec![
                    Guard::TaskIs {
                        task: ctx.task_id.clone(),
                        status: TaskStatus::Running,
                        lease: Some((ctx.worker_id.clone(), ctx.lease_id.clone())),
                    },
                    Guard::EffectFree { key: key.clone() },
                ],
                writes: vec![Write::Effect(EffectWrite::Start(intent))],
            },
            t_plus(11),
        )
        .await?;
    if let Some(state) = end {
        store
            .commit(
                Transition {
                    guards: Vec::new(),
                    writes: vec![Write::Effect(EffectWrite::Finish {
                        key: key.clone(),
                        owner: ctx.task_id.clone(),
                        state,
                    })],
                },
                t_plus(12),
            )
            .await?;
    }
    Ok(key)
}

/// Suspend a root on one call of `tool` and acquire its child.
async fn child_bootstrap(
    stores: &TestStores,
    tool: &str,
    input: serde_json::Value,
) -> Result<ToolTaskBootstrap> {
    let (_parent, children) =
        suspend_root_with_tools(stores, vec![("call_1".into(), tool.into(), input)]).await?;
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await
}

// ─────────────────────────────────────────────────────────────────────
// Replay-safe bypass
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn replay_safe_tool_bypasses_intent_persistence() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "read", serde_json::json!({"path": "a.txt"})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);

    let outcome = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("hello")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed, got: {outcome:?}"
    );
    assert!(
        stores.tasks.effect(&key).await?.is_none(),
        "replay-safe should not persist intent"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Side-effecting happy path
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn side_effecting_tool_persists_intent_and_completes() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);
    let tasks = &stores.tasks;
    let probe_key = key.clone();

    let outcome = run_tool_effect(
        ctx,
        tasks,
        &cancel,
        |_info, _collector| async move {
            let record = tasks
                .effect(&probe_key)
                .await?
                .context("the record commits before the tool runs")?;
            anyhow::ensure!(record.state == EffectState::Started);
            Ok(ToolResult::success("transfer done"))
        },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Completed { .. }),
        "expected Completed, got: {outcome:?}"
    );
    let intent = stores
        .tasks
        .effect(&key)
        .await?
        .context("intent should be persisted")?;
    assert_eq!(intent.state, EffectState::Completed);
    assert_eq!(intent.replay, ToolReplay::Unsafe);
    assert!(
        matches!(&intent.subject, EffectSubject::Tool { name, .. } if name == "transfer"),
        "{:?}",
        intent.subject
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Fail-closed when the record cannot be claimed
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fail_closed_when_intent_persist_fails() -> Result<()> {
    let stores = TestStores::new();
    let failing_store = FailingEffectStart {
        inner: &stores.tasks,
    };
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    let child_id = ctx.task_id.clone();

    let outcome = run_tool_effect(
        ctx,
        &failing_store,
        &cancel,
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
        .context("child should still exist")?;
    assert_eq!(child_after.status, TaskStatus::Failed);
    Ok(())
}

/// I5 under the lease: a worker whose lease was taken over neither claims
/// the record nor runs the tool.
#[tokio::test]
async fn a_worker_that_lost_its_lease_cannot_claim() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);
    stores
        .tasks
        .release_expired_leases(t_plus(10_000), crate::journal::LeaseRecoveryScope::Expired)
        .await?;
    stores
        .tasks
        .try_acquire_task(
            &ctx.task_id,
            WorkerId::from_string("other_worker"),
            LeaseId::from_string("other_lease"),
            t_plus(20_000),
            t_plus(10_001),
        )
        .await?
        .context("the other worker takes the child over")?;

    let result = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async {
            panic!("a stale worker must not run the tool");
        },
        t_plus(10_002),
    )
    .await;

    assert!(result.is_err(), "the stale worker's settle is refused too");
    assert!(
        stores.tasks.effect(&key).await?.is_none(),
        "a stale worker writes no record"
    );
    Ok(())
}

/// The outcome commits with the settle: when the settle is refused (the
/// lease moved while the tool ran), the record stays `Started` and the next
/// owner settles the call as interrupted instead of running it again.
#[tokio::test]
async fn a_refused_settle_leaves_the_record_started() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);
    let child_id = ctx.task_id.clone();
    let tasks = &stores.tasks;

    let result = run_tool_effect(
        ctx,
        tasks,
        &cancel,
        |_info, _collector| async move {
            tasks
                .release_expired_leases(t_plus(10_000), crate::journal::LeaseRecoveryScope::Expired)
                .await?;
            Ok(ToolResult::success("transfer done"))
        },
        t_plus(20),
    )
    .await;
    assert!(
        result.is_err(),
        "the settle under the lost lease is refused"
    );
    let record = stores.tasks.effect(&key).await?.context("record")?;
    assert_eq!(record.state, EffectState::Started);

    let child = stores
        .tasks
        .try_acquire_task(
            &child_id,
            WorkerId::from_string("next_worker"),
            LeaseId::from_string("next_lease"),
            t_plus(20_000),
            t_plus(10_001),
        )
        .await?
        .context("the child is acquirable again")?;
    let next = resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;
    let result = run_tool_effect(
        next,
        &stores.tasks,
        &cancel,
        |_info, _collector| async {
            panic!("the call already ran once");
        },
        t_plus(10_002),
    )
    .await;
    assert_interrupted(result)?;
    let record = stores.tasks.effect(&key).await?.context("record")?;
    assert!(
        matches!(record.state, EffectState::Interrupted { .. }),
        "{:?}",
        record.state
    );
    Ok(())
}

/// The review's probe: the tree is cancelled while an unsafe call runs. The
/// cancel ends the child, so it also abandons the child's started record in
/// that transaction; the worker's end transition is refused, and nothing
/// can run the call again because a terminal child is never acquired.
#[tokio::test]
async fn cancelling_the_tree_mid_call_abandons_the_record() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let (parent, children) = suspend_root_with_tools(
        &stores,
        vec![(
            "call_1".into(),
            "transfer".into(),
            serde_json::json!({"amount": 100}),
        )],
    )
    .await?;
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let ctx = resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);
    let child_id = ctx.task_id.clone();
    let tasks = &stores.tasks;
    let parent_id = parent.id.clone();

    let result = run_tool_effect(
        ctx,
        tasks,
        &cancel,
        |_info, _collector| async move {
            tasks.cancel_tree(&parent_id, t_plus(30)).await?;
            Ok(ToolResult::success("transfer done"))
        },
        t_plus(20),
    )
    .await;

    assert!(
        result.is_err(),
        "the end transition under a cancelled child is refused"
    );
    let child = stores.tasks.get(&child_id).await?.context("child")?;
    assert_eq!(child.status, TaskStatus::Cancelled);
    let record = stores.tasks.effect(&key).await?.context("record")?;
    assert_eq!(record.state, EffectState::Abandoned);
    assert!(
        stores
            .tasks
            .try_acquire_task(
                &child_id,
                WorkerId::from_string("next_worker"),
                LeaseId::from_string("next_lease"),
                t_plus(20_000),
                t_plus(10_001),
            )
            .await?
            .is_none(),
        "a cancelled child is never acquired, so the call never runs again"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Retry: completed record blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_intent_already_completed() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    seed_effect(
        &stores.tasks,
        &ctx,
        ToolReplay::Unsafe,
        Some(EffectState::Completed),
    )
    .await?;

    let result = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async {
            panic!("executor should not be called for already-completed operation");
        },
        t_plus(20),
    )
    .await;

    let Err(err) = result else {
        anyhow::bail!("expected error for completed retry");
    };
    let err = format!("{err:#}");
    assert!(
        err.contains("already completed"),
        "error should mention already completed: {err}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Retry: in-flight record blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_intent_ambiguous_in_flight() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    // A started record: the previous process crashed mid-execution.
    let key = seed_effect(&stores.tasks, &ctx, ToolReplay::Unsafe, None).await?;

    let result = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async {
            panic!("executor should not be called for ambiguous in-flight");
        },
        t_plus(20),
    )
    .await;

    let error = assert_interrupted(result)?;
    assert!(
        error.contains("transfer"),
        "the result names the tool: {error}"
    );
    let intent = stores
        .tasks
        .effect(&key)
        .await?
        .context("intent should exist")?;
    assert_eq!(
        intent.state,
        EffectState::Interrupted {
            captured_output: None
        },
        "the started intent is closed as interrupted"
    );
    assert_eq!(intent.state.columns().status, "failed");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Retry: failed side-effecting record blocks re-execution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_blocked_when_side_effecting_intent_failed() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 100})).await?;
    // A failed record: the executor ran and returned an error, so it may
    // have caused partial side effects.
    let failed = EffectState::Failed {
        error: "network timeout after partial debit".into(),
    };
    let key = seed_effect(
        &stores.tasks,
        &ctx,
        ToolReplay::Unsafe,
        Some(failed.clone()),
    )
    .await?;

    let result = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async {
            panic!("executor should not be called for failed side-effecting operation");
        },
        t_plus(20),
    )
    .await;

    let error = assert_interrupted(result)?;
    assert!(
        error.contains("transfer"),
        "the result names the tool: {error}"
    );
    let intent = stores
        .tasks
        .effect(&key)
        .await?
        .context("intent should exist")?;
    assert_eq!(intent.state, failed, "the failed intent stays failed");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Retry: a replay-safe call runs again
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_allowed_when_replay_safe_intent_failed() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "read", serde_json::json!({"path": "a.txt"})).await?;
    // A failed record for a replay-safe call, as an older release could
    // leave: safe to retry, since the call has no side effects.
    seed_effect(
        &stores.tasks,
        &ctx,
        ToolReplay::Safe,
        Some(EffectState::Failed {
            error: "transient error".into(),
        }),
    )
    .await?;

    let outcome = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
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
// Lifecycle: failure records the error
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn side_effecting_failure_records_intent_as_failed() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 999})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);

    let outcome = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async { Err(anyhow::anyhow!("insufficient funds")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Failed { .. }),
        "expected Failed, got: {outcome:?}"
    );
    let intent = stores
        .tasks
        .effect(&key)
        .await?
        .context("intent should be persisted")?;
    let EffectState::Failed { error } = &intent.state else {
        anyhow::bail!("expected a failed record, got {:?}", intent.state);
    };
    assert!(
        error.contains("insufficient funds"),
        "intent error should contain root cause: {error}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Cancellation before execution claims nothing
// ─────────────────────────────────────────────────────────────────────

/// A worker cancelled before the executor runs claims no record: the call
/// never ran, so the next attempt may run it. (Before effects, the guard
/// wrote the intent first and left it `Failed`, which made every later
/// attempt settle the call as interrupted.)
#[tokio::test]
async fn cancellation_before_execution_claims_nothing() -> Result<()> {
    let stores = TestStores::new();
    let cancel = CancellationToken::new();

    let ctx = child_bootstrap(&stores, "transfer", serde_json::json!({"amount": 50})).await?;
    let key = EffectKey::tool(&ctx.task_id, &ctx.tool_call.id);
    cancel.cancel();

    let outcome = run_tool_effect(
        ctx,
        &stores.tasks,
        &cancel,
        |_info, _collector| async { Ok(ToolResult::success("should not reach")) },
        t_plus(20),
    )
    .await?;

    assert!(
        matches!(outcome, ToolTaskOutcome::Cancelled),
        "expected Cancelled, got: {outcome:?}"
    );
    assert!(stores.tasks.effect(&key).await?.is_none());
    Ok(())
}

/// Assert `result` settled the child as interrupted without running it:
/// the child is `Failed` with the interrupted prefix, and one
/// `ToolCallEnd` closes the call.
fn assert_interrupted(result: Result<ToolTaskOutcome>) -> Result<String> {
    let ToolTaskOutcome::Failed {
        child,
        error,
        committed_events,
        ..
    } = result?
    else {
        anyhow::bail!("an unsafe in-flight retry must settle as interrupted");
    };
    assert_eq!(child.status, TaskStatus::Failed);
    assert!(
        error.starts_with(TOOL_INTERRUPTED_PREFIX),
        "error should be an interrupted result: {error}"
    );
    assert_eq!(child.last_error.as_deref(), Some(error.as_str()));
    let [end] = committed_events.as_slice() else {
        anyhow::bail!("expected exactly one ToolCallEnd, got {committed_events:?}");
    };
    assert!(
        matches!(&end.event, AgentEvent::ToolCallEnd { result, .. } if !result.success && result.output == error),
        "expected a failed ToolCallEnd carrying the interrupted result: {end:?}"
    );
    Ok(error)
}
