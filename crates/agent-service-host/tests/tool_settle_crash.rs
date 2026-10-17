#![cfg(all(feature = "sqlite", feature = "failpoints"))]

use std::process::Command;
use std::time::Duration;

use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
};
use agent_sdk_foundation::{AgentEvent, ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::effect::{EffectKey, EffectRecord, EffectState, EffectWrite};
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::{Committed, DurableStore, Guard, Rejected, Transition, Write};
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{
    ActivityBeacon, RootTurnDeps, RootTurnOutcome, ToolTaskBootstrap, execute_root_turn,
    execute_tool_task, resolve_tool_bootstrap, run_tool_effect, tool_intent,
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
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("settle.db").display()
    );
    let store = SqliteDurableStore::connect(&url).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();

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
    let _scenario = serial_failpoints();
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
    let _scenario = serial_failpoints();
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

/// Every test here shares fail-rs's process-global registry, and the setup
/// of one passes through the failpoint another has armed. Holding the
/// scenario for the whole test serializes them under a plain `cargo test`.
fn serial_failpoints() -> fail::FailScenario<'static> {
    fail::FailScenario::setup()
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
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let child = paused_on_confirmation(&store).await?.task_id;
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
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
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
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
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
    let _scenario = serial_failpoints();
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
    let _scenario = serial_failpoints();
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
    let _scenario = serial_failpoints();
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
    let _scenario = serial_failpoints();
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
    crash_in_transition(Box::pin(agent_server::worker::resume_confirmed_tool(
        bootstrap,
        &store,
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
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    let key = seed_started_effect(&store, &bootstrap).await?;

    crash_in_transition(run_tool_effect(
        bootstrap,
        &store,
        &CancellationToken::new(),
        |_call, _collector| async { Ok(ToolResult::success("must not run")) },
        at(12),
    ))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::Running).await?;
    let intent = store.effect(&key).await?.context("the intent is gone")?;
    ensure!(
        intent.state == EffectState::Started,
        "the intent's interrupted state must roll back with the settle, got {:?}",
        intent.state,
    );
    Ok(())
}

/// Commits through the store, except a transition that starts an effect,
/// which fails as an unavailable store would.
struct ClaimFails<'a> {
    inner: &'a SqliteDurableStore,
}

#[async_trait]
impl DurableStore for ClaimFails<'_> {
    async fn commit(
        &self,
        transition: Transition,
        now: OffsetDateTime,
    ) -> std::result::Result<Committed, Rejected> {
        if transition
            .effect_writes()
            .any(|write| matches!(write, EffectWrite::Start(_)))
        {
            return Err(Rejected::Store(anyhow::anyhow!("intent store unavailable")));
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

#[tokio::test]
async fn claim_failure_failpoint_keeps_the_child_running_without_its_tool_call_end() -> Result<()> {
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    crash_in_transition(run_tool_effect(
        bootstrap,
        &ClaimFails { inner: &store },
        &CancellationToken::new(),
        |_call, _collector| async { Ok(ToolResult::success("must not run")) },
        at(12),
    ))
    .await?;
    assert_unchanged(&store, &child, TaskStatus::Running).await
}

/// The record a crashed earlier attempt of `bootstrap`'s call left.
async fn seed_started_effect(
    store: &SqliteDurableStore,
    bootstrap: &ToolTaskBootstrap,
) -> Result<EffectKey> {
    let intent = tool_intent(bootstrap);
    let key = intent.key.clone();
    store
        .commit(
            Transition {
                guards: vec![
                    Guard::TaskIs {
                        task: bootstrap.task_id.clone(),
                        status: TaskStatus::Running,
                        lease: Some((bootstrap.worker_id.clone(), bootstrap.lease_id.clone())),
                    },
                    Guard::EffectFree { key: key.clone() },
                ],
                writes: vec![Write::Effect(EffectWrite::Start(intent))],
            },
            at(11),
        )
        .await?;
    Ok(key)
}

/// Crash `run_tool_effect` at `failpoint` and return how often the tool
/// ran, the child and the call's record.
async fn crash_effect_at(
    failpoint: &str,
) -> Result<(usize, AgentTask, Option<EffectRecord>, tempfile::TempDir)> {
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let bootstrap = leased_tool_child(&store).await?;
    let child = bootstrap.task_id.clone();
    let key = EffectKey::tool(&child, &bootstrap.tool_call.id);
    let runs = std::sync::atomic::AtomicUsize::new(0);
    let cancel = CancellationToken::new();

    fail::cfg(failpoint, "panic").map_err(|error| anyhow::anyhow!("{error}"))?;
    let crashed = std::panic::AssertUnwindSafe(run_tool_effect(
        bootstrap,
        &store,
        &cancel,
        |_call, _collector| async {
            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolResult::success("ran"))
        },
        at(12),
    ));
    let result = futures::FutureExt::catch_unwind(crashed).await;
    fail::remove(failpoint);
    ensure!(result.is_err(), "the failpoint {failpoint} must fire");

    let task = AgentTaskStore::get(&store, &child)
        .await?
        .context("the tool child is gone")?;
    let record = store.effect(&key).await?;
    Ok((
        runs.load(std::sync::atomic::Ordering::SeqCst),
        task,
        record,
        temp,
    ))
}

/// I5: the record commits before the tool runs. A crash right after the
/// start leaves a `Started` record and a tool that never ran, so recovery
/// settles the call as interrupted rather than guessing.
#[tokio::test]
async fn effect_after_start_failpoint_leaves_a_started_record_and_no_run() -> Result<()> {
    let _scenario = serial_failpoints();
    let (runs, child, record, _temp) = crash_effect_at("effect.after_start").await?;
    ensure!(runs == 0, "the tool ran {runs} times");
    ensure!(child.status == TaskStatus::Running, "{:?}", child.status);
    let record = record.context("the start committed before the crash")?;
    ensure!(record.state == EffectState::Started, "{:?}", record.state);
    Ok(())
}

/// The claim commits whole or not at all. `effect.start_written` fires
/// inside the start transaction right after the record's INSERT succeeded
/// (`crash_effect_at` requires it to fire), so the crash lands after the
/// write it protects: the record must be gone and the tool unrun.
#[tokio::test]
async fn effect_start_written_failpoint_rolls_back_the_claim() -> Result<()> {
    let _scenario = serial_failpoints();
    let (runs, child, record, _temp) = crash_effect_at("effect.start_written").await?;
    ensure!(runs == 0, "the tool ran {runs} times");
    ensure!(child.status == TaskStatus::Running, "{:?}", child.status);
    ensure!(record.is_none(), "a rolled-back claim left {record:?}");
    Ok(())
}

/// The outcome, the child's settle and its `ToolCallEnd` commit in one
/// transaction. Each crash point is required to fire (`crash_effect_at`),
/// and each comes after a write it protects:
/// - `effect.before_finish`: after the tool ran, before the end commit
///   (the mutation that commits the finish in its own transaction after the
///   settle turns this case red);
/// - `effect.finish_written`: inside the end transaction, right after the
///   record's UPDATE to its outcome succeeded (the finish is its first
///   write);
/// - `transition.before_events`: inside the end transaction, in the events
///   write, after the finish and the child's settle were written.
///
/// Every one must leave the child running and the record started.
#[tokio::test]
async fn effect_end_failpoints_never_settle_the_child_without_its_record() -> Result<()> {
    let _scenario = serial_failpoints();
    for failpoint in ["effect.before_finish", "effect.finish_written", FAILPOINT] {
        let (runs, child, record, temp) = crash_effect_at(failpoint).await?;
        ensure!(runs == 1, "{failpoint}: the tool ran {runs} times");
        ensure!(
            child.status == TaskStatus::Running,
            "{failpoint}: the child settled without its record: {:?}",
            child.status
        );
        let record = record.with_context(|| format!("{failpoint}: no record"))?;
        ensure!(
            record.state == EffectState::Started,
            "{failpoint}: the record ended without the settle: {:?}",
            record.state
        );
        let store = fresh_store(&temp).await?;
        assert_unchanged(&store, &child.id, TaskStatus::Running).await?;
    }
    Ok(())
}

struct Summarizer;

#[async_trait]
impl LlmProvider for Summarizer {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "summary".into(),
            content: vec![ContentBlock::Text {
                text: "[summary]".into(),
            }],
            model: "crash-test-model".into(),
            stop_reason: Some(StopReason::EndTurn),
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

/// Site 10: a compaction and its `ContextCompacted` event commit in one
/// transaction. A crash between the writes leaves the message projection
/// uncompacted with no `ContextCompacted` and no advisory row.
#[tokio::test]
async fn compaction_failpoint_leaves_the_projection_uncompacted_without_its_event() -> Result<()> {
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let thread_id = ThreadId::from_string(THREAD);
    ThreadStore::get_or_create(&store, &thread_id, at(0)).await?;
    let earlier: Vec<llm::Message> = (0..6)
        .map(|turn| {
            let text = format!("turn {turn}: {}", "context ".repeat(400));
            if turn % 2 == 0 {
                llm::Message::user(text)
            } else {
                llm::Message::assistant(text)
            }
        })
        .collect();
    MessageProjectionStore::commit_messages(&store, &thread_id, earlier.clone(), at(0)).await?;
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
        thread_id: thread_id.clone(),
        task_id: root_id,
        task: root,
        worker_id: worker,
        lease_id: lease,
        definition: definition(),
    };
    let inputs = build_root_worker_inputs(bootstrap, &store, &store, &store, at(1)).await?;
    let config = agent_sdk::context::CompactionConfig::default()
        .with_threshold_tokens(1)
        .with_min_messages(2)
        .with_retain_recent(0);
    let summarizer: std::sync::Arc<dyn LlmProvider> = std::sync::Arc::new(Summarizer);
    let notifier = EventNotifier::new();
    let deps = RootTurnDeps {
        compaction_config: Some(&config),
        compaction_provider: Some(&summarizer),
        ..root_deps(&store, &notifier)
    };
    crash_in_transition(execute_root_turn(
        inputs,
        "Carry on",
        &OneToolCallProvider,
        &deps,
        at(2),
    ))
    .await?;

    let projection = MessageProjectionStore::get(&store, &thread_id)
        .await?
        .context("the projection")?;
    let compacted = EventRepository::get_events(&store, &thread_id)
        .await?
        .iter()
        .filter(|event| matches!(event.event, AgentEvent::ContextCompacted { .. }))
        .count();
    let advisories = agent_server::journal::outbox::OutboxStore::list_by_thread(&store, &thread_id)
        .await?
        .iter()
        .filter(|row| {
            row.kind
                == agent_server::journal::outbox_message::OutboxMessageKind::ThreadEventsAvailable
        })
        .count();
    ensure!(
        projection.compactions.is_empty() && compacted == 0 && advisories == 0,
        "the compaction and its event disagree after a crash: {} compactions, \
         {compacted} ContextCompacted, {advisories} advisories",
        projection.compactions.len(),
    );
    ensure!(
        projection.messages.len() == earlier.len(),
        "the raw transcript is unchanged"
    );
    Ok(())
}

fn empty_close() -> agent_server::journal::turn_attempt::CloseAttemptParams {
    use agent_server::journal::turn_attempt::{CloseAttemptParams, TurnAttemptOutcome};

    CloseAttemptParams {
        response_blob: json!({}),
        response_id: None,
        response_model: None,
        stop_reason: None,
        outcome: TurnAttemptOutcome::Success,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        route_provider: None,
        thinking_mode: None,
        thinking_budget_tokens: None,
        thinking_effort: None,
    }
}

/// Design I2: the completed turn and its root's terminal row commit together.
/// `commit.completed_turn.settled` fires inside the turn's transaction only
/// once the root's settle has been written (and is required to fire), so a
/// crash there must leave neither: the root running, the thread at no
/// committed turn, the attempt open.
#[tokio::test]
async fn completed_turn_settle_failpoint_rolls_back_the_turn_with_its_root() -> Result<()> {
    use agent_sdk_foundation::TokenUsage;
    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_server::journal::Settle;
    use agent_server::journal::checkpoint::CheckpointKind;
    use agent_server::journal::commit::{
        CommitOwnerGuard, CompletedTurnCommit, commit_completed_turn,
    };
    use agent_server::journal::turn_attempt::OpenAttemptParams;
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;

    const SETTLED: &str = "commit.completed_turn.settled";
    let _scenario = serial_failpoints();
    let temp = tempfile::tempdir()?;
    let store = fresh_store(&temp).await?;
    let thread = ThreadId::from_string("completed-turn-settle");
    ThreadStore::get_or_create(&store, &thread, at(1)).await?;
    let root = AgentTask::new_root_turn(thread.clone(), at(1), 3);
    store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = (WorkerId::from_string("w"), LeaseId::from_string("l"));
    store
        .try_acquire_task(&root.id, worker.clone(), lease.clone(), at(600), at(2))
        .await?
        .context("acquire the root")?;
    let attempt = TurnAttemptStore::open_attempt(
        &store,
        OpenAttemptParams {
            task_id: root.id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-5-5"),
            request_blob: json!({}),
            now: at(3),
            otel_trace_id: None,
            otel_span_id: None,
        },
    )
    .await?;
    let params = CompletedTurnCommit {
        extra_writes: vec![Write::SettleTask {
            task: root.id.clone(),
            outcome: Settle::CompleteWithoutResult,
        }],
        delivered_injection_ids: Vec::new(),
        checkpoint_kind: CheckpointKind::FullTurn,
        thread_id: thread.clone(),
        task_id: root.id.clone(),
        expected_turn: 1,
        turn_attempt_id: attempt.id.clone(),
        close_attempt_params: empty_close(),
        messages: vec![llm::Message::assistant("done")],
        turn_usage: TokenUsage::default(),
        agent_state_snapshot: json!({}),
        events: vec![AgentEvent::text("m1", "done")],
        outbox_max_attempts: 3,
        owner_guard: Some(CommitOwnerGuard {
            worker_id: worker,
            lease_id: lease,
        }),
        now: at(4),
    };

    fail::cfg(SETTLED, "panic").map_err(|error| anyhow::anyhow!("{error}"))?;
    let crashed = std::panic::AssertUnwindSafe(commit_completed_turn(params, &store));
    let result = futures::FutureExt::catch_unwind(crashed).await;
    fail::remove(SETTLED);
    ensure!(
        result.is_err(),
        "the failpoint after the root's settle must fire"
    );

    drop(store);
    let store = fresh_store(&temp).await?;
    let root = AgentTaskStore::get(&store, &root.id)
        .await?
        .context("root")?;
    ensure!(root.status == TaskStatus::Running, "{:?}", root.status);
    let committed = ThreadStore::get(&store, &thread)
        .await?
        .map_or(0, |thread| thread.committed_turns);
    ensure!(
        committed == 0,
        "the turn landed without its root: {committed}"
    );
    let attempt = TurnAttemptStore::get(&store, &attempt.id)
        .await?
        .context("attempt")?;
    ensure!(
        attempt.closed_at.is_none(),
        "the attempt closed without the turn"
    );
    Ok(())
}
