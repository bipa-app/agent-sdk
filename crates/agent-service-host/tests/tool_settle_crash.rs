#![cfg(all(feature = "sqlite", feature = "failpoints"))]

use std::process::Command;
use std::time::Duration;

use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
};
use agent_sdk_foundation::{AgentEvent, ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{
    ActivityBeacon, RootTurnDeps, RootTurnOutcome, ToolTaskBootstrap, execute_root_turn,
    execute_tool_task, resolve_tool_bootstrap,
};
use agent_service_host::fail;
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde_json::json;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

const CHILD_ENV: &str = "AGENT_SDK_TOOL_SETTLE_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_TOOL_SETTLE_CRASH_DB";
const READY_ENV: &str = "AGENT_SDK_TOOL_SETTLE_CRASH_READY";
const FAILPOINT: &str = "transition.before_events";
const CALL_ID: &str = "settle-call";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "crash-test".into(),
        model: "crash-test-model".into(),
        system_prompt: "Run one tool.".into(),
        max_tokens: 256,
        tools: vec![llm::Tool {
            name: "probe".into(),
            description: "Probes something".into(),
            input_schema: json!({"type": "object"}),
            display_name: "Probe".into(),
            tier: ToolTier::Observe,
            concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
            replay: agent_sdk_foundation::ToolReplay::Unsafe,
        }],
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

struct OneToolCallProvider;

#[async_trait]
impl LlmProvider for OneToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "settle-response".into(),
            content: vec![ContentBlock::ToolUse {
                id: CALL_ID.into(),
                name: "probe".into(),
                input: json!({}),
                thought_signature: None,
            }],
            model: "crash-test-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                served_speed: None,
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "crash-test-model"
    }

    fn provider(&self) -> &'static str {
        "crash-test"
    }
}

const THREAD: &str = "tool-settle-thread";

fn root_deps<'a>(store: &'a SqliteDurableStore, notifier: &'a EventNotifier) -> RootTurnDeps<'a> {
    RootTurnDeps {
        task_store: store,
        thread_store: store,
        message_store: store,
        attempt_store: store,
        checkpoint_store: store,
        event_repo: store,
        event_notifier: notifier,
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

/// Run a fresh root turn whose model calls one tool, up to its suspension.
async fn suspend_root_on_one_tool_call(
    store: &SqliteDurableStore,
) -> Result<(AgentTaskId, RootTurnOutcome)> {
    let thread_id = ThreadId::from_string(THREAD);
    ThreadStore::get_or_create(store, &thread_id, at(0)).await?;
    MessageProjectionStore::get_or_create(store, &thread_id, at(0)).await?;
    let root = AgentTask::new_root_turn(thread_id.clone(), at(0), 3);
    let root_id = root.id.clone();
    store.submit_root_turn(root).await?;
    let worker = WorkerId::from_string("root-worker");
    let lease = LeaseId::from_string("root-lease");
    let root = store
        .try_acquire_task(&root_id, worker.clone(), lease.clone(), at(60), at(0))
        .await?
        .context("root was not acquired")?;
    let bootstrap = WorkerBootstrapContext {
        thread_id,
        task_id: root_id.clone(),
        task: root,
        worker_id: worker,
        lease_id: lease,
        definition: definition(),
    };
    let inputs = build_root_worker_inputs(bootstrap, store, store, store, at(1)).await?;
    let notifier = EventNotifier::new();
    let deps = root_deps(store, &notifier);
    let outcome =
        execute_root_turn(inputs, "Run the tool", &OneToolCallProvider, &deps, at(2)).await?;
    Ok((root_id, outcome))
}

/// A tool child of a fresh root turn, leased and ready to run.
async fn leased_tool_child(store: &SqliteDurableStore) -> Result<ToolTaskBootstrap> {
    let (_, outcome) = suspend_root_on_one_tool_call(store).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        bail!("the root turn did not suspend on its tool call");
    };
    let [child] = child_tasks.as_slice() else {
        bail!("expected one tool child, got {}", child_tasks.len());
    };
    let child = store
        .try_acquire_task(
            &child.id,
            WorkerId::from_string("tool-worker"),
            LeaseId::from_string("tool-lease"),
            at(600),
            at(10),
        )
        .await?
        .context("tool child was not acquired")?;
    resolve_tool_bootstrap(child, store, ActivityBeacon::default()).await
}

async fn run_tool(store: &SqliteDurableStore, bootstrap: ToolTaskBootstrap) -> Result<()> {
    execute_tool_task(
        bootstrap,
        store,
        &CancellationToken::new(),
        |_call, _collector| async { Ok(ToolResult::success("done")) },
        at(11),
    )
    .await
    .map(drop)
}

/// The child's row and its `ToolCallEnd` must agree after a crash: either
/// both committed or neither did.
async fn assert_neither_half(store: &SqliteDurableStore, child: &AgentTaskId) -> Result<()> {
    assert_unchanged(store, child, TaskStatus::Running).await
}

/// After a crash inside a transition the task is still `status` and its
/// tool call has no `ToolCallEnd`.
async fn assert_unchanged(
    store: &SqliteDurableStore,
    child: &AgentTaskId,
    status: TaskStatus,
) -> Result<()> {
    let task = AgentTaskStore::get(store, child)
        .await?
        .context("the tool child is gone")?;
    let ends = EventRepository::get_events(store, &task.thread_id)
        .await?
        .into_iter()
        .filter(|event| matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == CALL_ID))
        .count();
    ensure!(
        task.status == status && ends == 0,
        "the settle and its ToolCallEnd disagree after a crash: child {:?}, {ends} ToolCallEnd",
        task.status,
    );
    Ok(())
}

#[tokio::test]
async fn tool_settle_failpoint_rolls_back_the_child_with_its_tool_call_end() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("settle.db").display()
    );
    let store = SqliteDurableStore::connect(&url).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();

    let _scenario = fail::FailScenario::setup();
    fail::cfg(FAILPOINT, "panic").map_err(|error| anyhow::anyhow!("{error}"))?;
    let crashed = std::panic::AssertUnwindSafe(run_tool(&store, bootstrap));
    let result = futures::FutureExt::catch_unwind(crashed).await;
    fail::remove(FAILPOINT);
    ensure!(
        result.is_err(),
        "the failpoint between the writes must fire"
    );

    assert_neither_half(&store, &child).await
}

/// Child-process half of the kill test: pause inside the settle transaction
/// and wait to be killed there.
#[tokio::test]
async fn tool_settle_failpoint_kill_child() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    fail::cfg_callback(FAILPOINT, move || {
        let _ = std::fs::write(&ready, child.as_str());
        std::thread::sleep(Duration::from_mins(1));
    })
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    run_tool(&store, bootstrap).await?;
    bail!("the child was not killed inside the settle transaction")
}

#[tokio::test]
async fn kill_nine_inside_the_tool_settle_failpoint_leaves_neither_half() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let db = temp.path().join("settle-kill.db");
    let ready = temp.path().join("in-settle");
    let url = format!("sqlite://{}?mode=rwc", db.display());

    let mut child = Command::new(std::env::current_exe()?)
        .arg("tool_settle_failpoint_kill_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, &url)
        .env(READY_ENV, &ready)
        .spawn()
        .context("spawn the settle child")?;
    let mut task_id = None;
    for _ in 0..400 {
        if let Ok(id) = std::fs::read_to_string(&ready)
            && !id.is_empty()
        {
            task_id = Some(AgentTaskId::from_string(id.trim()));
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let Some(task_id) = task_id else {
        let _ = child.kill();
        let output = child.wait_with_output()?;
        bail!(
            "the child never reached the settle transaction: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    let killed = Command::new("kill")
        .arg("-9")
        .arg(child.id().to_string())
        .status()
        .context("kill -9 the settle child")?;
    ensure!(killed.success(), "kill -9 failed with {killed}");
    let _ = child.wait()?;

    let store = SqliteDurableStore::connect(&url).await?;
    assert_neither_half(&store, &task_id).await
}

/// Run `crashing` with the transition failpoint armed to panic, and require
/// that it fired.
async fn crash_in_transition<F: std::future::Future>(crashing: F) -> Result<()> {
    fail::cfg(FAILPOINT, "panic").map_err(|error| anyhow::anyhow!("{error}"))?;
    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(crashing)).await;
    fail::remove(FAILPOINT);
    ensure!(
        result.is_err(),
        "the failpoint between the writes must fire"
    );
    Ok(())
}

async fn paused_on_confirmation(store: &SqliteDurableStore) -> Result<ToolTaskBootstrap> {
    let bootstrap = leased_tool_child(store).await?;
    agent_server::worker::pause_tool_for_confirmation(&bootstrap, store, at(12)).await?;
    Ok(bootstrap)
}

async fn fresh_store(temp: &tempfile::TempDir) -> Result<SqliteDurableStore> {
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("settle.db").display()
    );
    SqliteDurableStore::connect(&url).await
}

#[tokio::test]
async fn reject_failpoint_keeps_the_confirmation_parked_without_its_tool_call_end() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let child = paused_on_confirmation(&store).await?.task_id;
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(agent_server::worker::apply_confirmation_decision(
        &child,
        agent_server::worker::ConfirmationDecision::Rejected {
            reason: "no".into(),
        },
        &store,
        at(13),
    ))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::AwaitingConfirmation).await
}

#[tokio::test]
async fn pause_failpoint_keeps_the_child_running_without_its_prompt() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(agent_server::worker::pause_tool_for_confirmation(
        &bootstrap,
        &store,
        at(12),
    ))
    .await?;
    let task = AgentTaskStore::get(&store, &bootstrap.task_id)
        .await?
        .context("the tool child is gone")?;
    let prompts = EventRepository::get_events(&store, &task.thread_id)
        .await?
        .into_iter()
        .filter(|event| {
            matches!(&event.event, AgentEvent::ToolRequiresConfirmation { id, .. } if id == CALL_ID)
        })
        .count();
    ensure!(
        task.status == TaskStatus::Running && prompts == 0,
        "the pause and its prompt disagree after a crash: child {:?}, {prompts} prompts",
        task.status,
    );
    Ok(())
}

/// Site 8: a root turn parks on its tool children and commits its
/// suspension (content and `ToolCallStart`) in one transaction. A crash
/// inside that transaction leaves the root running with no children and no
/// suspension event.
#[tokio::test]
async fn suspension_failpoint_leaves_the_root_running_without_children_or_events() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let _scenario = fail::FailScenario::setup();
    fail::cfg("atomic_spawn_batch.after_rows", "panic")
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let crashed = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
        suspend_root_on_one_tool_call(&store),
    ))
    .await;
    fail::remove("atomic_spawn_batch.after_rows");
    ensure!(
        crashed.is_err(),
        "the failpoint inside the spawn transaction must fire"
    );

    let thread = ThreadId::from_string(THREAD);
    let roots = AgentTaskStore::list_by_thread(&store, &thread).await?;
    let [root] = roots.as_slice() else {
        bail!("one root and no children expected, got {}", roots.len());
    };
    let starts = EventRepository::get_events(&store, &thread)
        .await?
        .into_iter()
        .filter(|event| matches!(event.event, AgentEvent::ToolCallStart { .. }))
        .count();
    ensure!(
        root.status == TaskStatus::Running && starts == 0,
        "the spawn and its suspension events disagree after a crash: root {:?}, {starts} \
         ToolCallStart",
        root.status,
    );
    Ok(())
}

/// The suspension's events commit with the spawn: after a clean suspend,
/// the root's `ToolCallStart` is in the journal.
#[tokio::test]
async fn a_suspension_commits_its_tool_call_start_with_the_spawn() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let (_, outcome) = suspend_root_on_one_tool_call(&store).await?;
    let RootTurnOutcome::Suspended {
        committed_events, ..
    } = outcome
    else {
        bail!("the root turn did not suspend");
    };
    ensure!(
        committed_events
            .iter()
            .any(|event| matches!(event.event, AgentEvent::ToolCallStart { .. })),
        "the outcome carries the suspension's ToolCallStart",
    );
    Ok(())
}

async fn error_events(store: &SqliteDurableStore) -> Result<usize> {
    Ok(
        EventRepository::get_events(store, &ThreadId::from_string(THREAD))
            .await?
            .iter()
            .filter(|event| matches!(event.event, AgentEvent::Error { .. }))
            .count(),
    )
}

/// Site 6: a failing root turn and its terminal error event commit in one
/// transaction. A crash between the writes leaves the root running with
/// no error event.
#[tokio::test]
async fn root_failure_failpoint_leaves_the_root_running_without_its_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let thread_id = ThreadId::from_string(THREAD);
    ThreadStore::get_or_create(&store, &thread_id, at(0)).await?;
    let root = AgentTask::new_root_turn(thread_id.clone(), at(0), 3);
    let root_id = root.id.clone();
    store.submit_root_turn(root).await?;
    let worker = WorkerId::from_string("root-worker");
    let lease = LeaseId::from_string("root-lease");
    store
        .try_acquire_task(&root_id, worker.clone(), lease.clone(), at(60), at(0))
        .await?
        .context("root was not acquired")?;
    let notifier = EventNotifier::new();
    let deps = root_deps(&store, &notifier);
    let error = anyhow::anyhow!("the provider stopped");
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(agent_server::worker::fail_root_turn_with_reason(
        agent_server::worker::FailRootTurnParams {
            task_id: &root_id,
            worker_id: &worker,
            lease_id: &lease,
            thread_id: &thread_id,
            error: &error,
            reason: agent_sdk_foundation::TerminalReason::InternalError,
        },
        &deps,
        at(5),
    ))
    .await?;
    let after = AgentTaskStore::get(&store, &root_id)
        .await?
        .context("the root")?;
    let errors = error_events(&store).await?;
    ensure!(
        after.status == TaskStatus::Running && errors == 0,
        "the failure and its error disagree after a crash: root {:?}, {errors} error events",
        after.status,
    );
    Ok(())
}

/// Site 7: reverting a failed steering wake re-parks the parent and
/// commits its error event in one transaction. A crash between the writes
/// leaves the steering exchange running with no error event.
#[tokio::test]
async fn steering_revert_failpoint_leaves_the_exchange_running_without_its_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let (root_id, _) = suspend_root_on_one_tool_call(&store).await?;
    store
        .enqueue_steering_resume(
            &root_id,
            vec![agent_sdk_foundation::llm::ContentBlock::Text {
                text: "also check the logs".into(),
            }],
            at(20),
        )
        .await?
        .context("the parked root takes the steering note")?;
    let worker = WorkerId::from_string("steering-worker");
    let lease = LeaseId::from_string("steering-lease");
    let parent = store
        .try_acquire_task(&root_id, worker.clone(), lease.clone(), at(600), at(21))
        .await?
        .context("the woken root acquires")?;
    let notifier = EventNotifier::new();
    let deps = root_deps(&store, &notifier);
    let error = anyhow::anyhow!("the steering exchange failed");
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(agent_server::worker::revert_steering_wake(
        &parent,
        &worker,
        &lease,
        &error,
        &deps,
        at(22),
    ))
    .await?;
    let after = AgentTaskStore::get(&store, &root_id)
        .await?
        .context("the root")?;
    let errors = error_events(&store).await?;
    ensure!(
        after.status == TaskStatus::Running && errors == 0,
        "the re-park and its error disagree after a crash: root {:?}, {errors} error events",
        after.status,
    );
    Ok(())
}

struct DenyAll;

#[async_trait]
impl agent_server::worker::ConfirmationPolicy for DenyAll {
    async fn check_policy(
        &self,
        _tool_call: &agent_sdk_foundation::PendingToolCallInfo,
    ) -> Result<agent_server::worker::PolicyVerdict> {
        Ok(agent_server::worker::PolicyVerdict::Denied {
            reason: "policy changed".into(),
        })
    }
}

#[tokio::test]
async fn policy_denial_failpoint_keeps_the_child_running_without_its_tool_call_end() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let child = paused_on_confirmation(&store).await?.task_id;
    agent_server::worker::apply_confirmation_decision(
        &child,
        agent_server::worker::ConfirmationDecision::Approved,
        &store,
        at(13),
    )
    .await?;
    let resumed = store
        .try_acquire_task(
            &child,
            WorkerId::from_string("resume-worker"),
            LeaseId::from_string("resume-lease"),
            at(700),
            at(14),
        )
        .await?
        .context("the approved child was not acquired")?;
    let bootstrap = resolve_tool_bootstrap(resumed, &store, ActivityBeacon::default()).await?;
    let deps = agent_server::journal::execution_intent::GuardedExecutionDeps {
        task_store: &store,
        intent_store: &store,
        event_repo: &store,
    };
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(Box::pin(agent_server::worker::resume_confirmed_tool(
        bootstrap,
        &deps,
        &DenyAll,
        &CancellationToken::new(),
        |_call, _collector| async { Ok(ToolResult::success("must not run")) },
        at(15),
    )))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::Running).await
}

#[tokio::test]
async fn interrupt_failpoint_keeps_the_child_and_its_intent_unsettled() -> Result<()> {
    use agent_server::journal::execution_intent::{
        ExecutionIntent, ExecutionIntentStore, GuardedExecutionDeps, IntentStatus, OperationId,
        ToolEffectClass, guarded_tool_execution,
    };
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    let operation = OperationId::new(&child, &bootstrap.tool_call.id);
    let mut started = ExecutionIntent::new(
        operation.clone(),
        ToolEffectClass::SideEffecting,
        &bootstrap.tool_call,
        child.clone(),
        at(11),
    );
    store.persist_intent(&started).await?;
    started.mark_started(at(11));
    store.update_intent(&started).await?;

    let deps = GuardedExecutionDeps {
        task_store: &store,
        intent_store: &store,
        event_repo: &store,
    };
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(guarded_tool_execution(
        bootstrap,
        &deps,
        &CancellationToken::new(),
        ToolEffectClass::SideEffecting,
        |_call, _collector| async { Ok(ToolResult::success("must not run")) },
        at(12),
    ))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::Running).await?;
    let intent = store
        .get_intent(&operation)
        .await?
        .context("the intent is gone")?;
    ensure!(
        intent.status == IntentStatus::Started,
        "the intent's Failed status must roll back with the settle, got {:?}",
        intent.status,
    );
    Ok(())
}

struct ClaimFails;

#[async_trait]
impl agent_server::journal::execution_intent::ExecutionIntentStore for ClaimFails {
    async fn persist_intent(
        &self,
        _intent: &agent_server::journal::execution_intent::ExecutionIntent,
    ) -> Result<()> {
        bail!("intent store unavailable")
    }

    async fn update_intent(
        &self,
        _intent: &agent_server::journal::execution_intent::ExecutionIntent,
    ) -> Result<()> {
        bail!("intent store unavailable")
    }

    async fn get_intent(
        &self,
        _operation_id: &agent_server::journal::execution_intent::OperationId,
    ) -> Result<Option<agent_server::journal::execution_intent::ExecutionIntent>> {
        Ok(None)
    }

    async fn get_intent_by_task(
        &self,
        _child_task_id: &AgentTaskId,
    ) -> Result<Option<agent_server::journal::execution_intent::ExecutionIntent>> {
        Ok(None)
    }
}

#[tokio::test]
async fn claim_failure_failpoint_keeps_the_child_running_without_its_tool_call_end() -> Result<()> {
    use agent_server::journal::execution_intent::{
        GuardedExecutionDeps, ToolEffectClass, guarded_tool_execution,
    };
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    let deps = GuardedExecutionDeps {
        task_store: &store,
        intent_store: &ClaimFails,
        event_repo: &store,
    };
    let _scenario = fail::FailScenario::setup();
    crash_in_transition(guarded_tool_execution(
        bootstrap,
        &deps,
        &CancellationToken::new(),
        ToolEffectClass::SideEffecting,
        |_call, _collector| async { Ok(ToolResult::success("must not run")) },
        at(12),
    ))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::Running).await
}
