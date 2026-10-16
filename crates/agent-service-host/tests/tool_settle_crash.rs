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

/// A tool child of a fresh root turn, leased and ready to run.
async fn leased_tool_child(store: &SqliteDurableStore) -> Result<ToolTaskBootstrap> {
    let thread_id = ThreadId::from_string("tool-settle-thread");
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
        task_id: root_id,
        task: root,
        worker_id: worker,
        lease_id: lease,
        definition: definition(),
    };
    let inputs = build_root_worker_inputs(bootstrap, store, store, store, at(1)).await?;
    let notifier = EventNotifier::new();
    let deps = RootTurnDeps {
        task_store: store,
        thread_store: store,
        message_store: store,
        attempt_store: store,
        checkpoint_store: store,
        event_repo: store,
        event_notifier: &notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };
    let outcome =
        execute_root_turn(inputs, "Run the tool", &OneToolCallProvider, &deps, at(2)).await?;
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
    let task = AgentTaskStore::get(store, child)
        .await?
        .context("the tool child is gone")?;
    let ends = EventRepository::get_events(store, &task.thread_id)
        .await?
        .into_iter()
        .filter(|event| matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == CALL_ID))
        .count();
    ensure!(
        task.status == TaskStatus::Running && ends == 0,
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
