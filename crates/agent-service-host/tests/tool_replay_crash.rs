#![cfg(feature = "sqlite")]

use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use agent_sdk::builtin_tools::{BuiltinToolsConfig, register_builtin_tools};
use agent_sdk::todo::TodoState;
use agent_sdk::user_interaction::AskUserQuestionTool;
use agent_sdk::{
    AgentCapabilities, DynamicToolName, InMemoryFileSystem, Tool, ToolContext, ToolRegistry,
};
use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
};
use agent_sdk_foundation::{AgentEvent, ThreadId, ToolReplay, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::DurableStore;
use agent_server::journal::effect::{EffectKey, EffectState, TOOL_INTERRUPTED_PREFIX};
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::recovery::RecoveryAction;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{
    ActivityBeacon, RootTurnDeps, RootTurnOutcome, ToolTaskOutcome, execute_root_turn,
    resolve_tool_bootstrap, run_tool_effect,
};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};
use time::OffsetDateTime;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

const CHILD_ENV: &str = "AGENT_SDK_TOOL_REPLAY_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_TOOL_REPLAY_CRASH_DB";
const TOOL_ENV: &str = "AGENT_SDK_TOOL_REPLAY_CRASH_TOOL";
const READY_ENV: &str = "AGENT_SDK_TOOL_REPLAY_CRASH_READY";

const UNDECLARED_TOOL: &str = "deploy";
const CALL_ID: &str = "crash-tool-call";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

/// A host tool that declares nothing about replay.
struct DeployTool;

impl Tool<()> for DeployTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(UNDECLARED_TOOL)
    }

    fn display_name(&self) -> &'static str {
        "Deploy"
    }

    fn description(&self) -> &'static str {
        "Deploys the service"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
        Ok(ToolResult::success("deployed"))
    }
}

/// The tools a host advertises when its policy skips every prompt, as
/// bip's default `AutoApprove` does: each tool is lowered to `Observe`,
/// and keeps the replay value it declares.
fn auto_approved_tools() -> Vec<llm::Tool> {
    let mut registry = ToolRegistry::<()>::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: Arc::new(InMemoryFileSystem::new("/workspace")),
            capabilities: AgentCapabilities::full_access(),
            todo_state: Some(Arc::new(RwLock::new(TodoState::new()))),
            link_fetch: true,
        },
    );
    registry.register(DeployTool);
    registry.register(AskUserQuestionTool::with_channels(1).0);
    registry
        .to_llm_tools()
        .into_iter()
        .map(|tool| llm::Tool {
            tier: ToolTier::Observe,
            ..tool
        })
        .collect()
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "crash-test".into(),
        model: "crash-test-model".into(),
        system_prompt: "Run one tool.".into(),
        max_tokens: 256,
        tools: auto_approved_tools(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

struct OneToolCallProvider {
    tool: String,
}

#[async_trait]
impl LlmProvider for OneToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "crash-tool-response".into(),
            content: vec![ContentBlock::ToolUse {
                id: CALL_ID.into(),
                name: self.tool.clone(),
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

fn root_turn_deps<'a>(
    store: &'a SqliteDurableStore,
    notifier: &'a EventNotifier,
) -> RootTurnDeps<'a> {
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

/// Run a root turn whose model calls `tool` once, and return the tool
/// child it spawned.
async fn spawn_tool_child(store: &SqliteDurableStore, tool: &str) -> Result<AgentTask> {
    let thread_id = ThreadId::from_string("kill-nine-tool-thread");
    ThreadStore::get_or_create(store, &thread_id, at(0)).await?;
    MessageProjectionStore::get_or_create(store, &thread_id, at(0)).await?;
    let root = AgentTask::new_root_turn(thread_id.clone(), at(0), 3);
    let root_id = root.id.clone();
    store.submit_root_turn(root).await?;
    let worker = WorkerId::from_string("seed-worker");
    let lease = LeaseId::from_string("seed-lease");
    let root = store
        .try_acquire_task(&root_id, worker.clone(), lease.clone(), at(60), at(0))
        .await?
        .context("seed root was not acquired")?;
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
    let provider = OneToolCallProvider {
        tool: tool.to_owned(),
    };
    let outcome = execute_root_turn(
        inputs,
        "Run the tool",
        &provider,
        &root_turn_deps(store, &notifier),
        at(2),
    )
    .await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        bail!("the root turn did not suspend on its tool call");
    };
    let [child] = child_tasks.as_slice() else {
        bail!("expected one tool child, got {}", child_tasks.len());
    };
    Ok(child.clone())
}

/// Child-process half: start the tool call through the guarded path,
/// publish the child id from inside the tool, then wait to be killed
/// while the tool runs.
#[tokio::test]
async fn tool_replay_crash_child() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let tool = std::env::var(TOOL_ENV).context("missing crash tool")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    let child = spawn_tool_child(&store, &tool).await?;
    let child = store
        .try_acquire_task(
            &child.id,
            WorkerId::from_string("crashed-worker"),
            LeaseId::from_string("crashed-lease"),
            at(600),
            at(10),
        )
        .await?
        .context("tool child was not acquired")?;
    let bootstrap = resolve_tool_bootstrap(child, &store, ActivityBeacon::default()).await?;
    let child_id = bootstrap.task_id.clone();
    run_tool_effect(
        bootstrap,
        &store,
        &CancellationToken::new(),
        move |_call, _collector| async move {
            std::fs::write(&ready, child_id.as_str())?;
            tokio::time::sleep(Duration::from_mins(1)).await;
            bail!("child was not killed while the tool ran")
        },
        at(11),
    )
    .await?;
    bail!("child was not killed while the tool ran")
}

/// Spawn the child, wait until its tool is running, then `kill -9` it.
async fn crash_while_tool_runs(
    url: &str,
    tool: &str,
    ready_path: &std::path::Path,
) -> Result<AgentTaskId> {
    let executable = std::env::current_exe()?;
    let mut child = Command::new(executable)
        .arg("tool_replay_crash_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, url)
        .env(TOOL_ENV, tool)
        .env(READY_ENV, ready_path)
        .spawn()
        .context("spawn tool child process")?;

    for _ in 0..200 {
        if let Ok(task_id) = std::fs::read_to_string(ready_path)
            && !task_id.is_empty()
        {
            let kill_status = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("send kill -9 to tool child process")?;
            ensure!(kill_status.success(), "kill -9 failed with {kill_status}");
            let _ = child.wait()?;
            return Ok(AgentTaskId::from_string(task_id.trim()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "tool child process did not start the tool: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Recover the crashed call the way a booting host does: the lease
/// sweep requeues the expired tool child, a new worker acquires it, and
/// the guarded path decides whether the tool runs again.
async fn recover(
    store: &SqliteDurableStore,
    task_id: &AgentTaskId,
    runs: &AtomicUsize,
) -> Result<ToolTaskOutcome> {
    let records = store.release_expired_leases(at(601)).await?;
    ensure!(
        records
            .iter()
            .any(|record| record.id == *task_id && record.action == RecoveryAction::Requeue),
        "the lease sweep did not requeue the crashed tool child: {records:?}"
    );
    let child = store
        .try_acquire_task(
            task_id,
            WorkerId::from_string("boot-worker"),
            LeaseId::from_string("boot-lease"),
            at(1200),
            at(602),
        )
        .await?
        .context("boot did not reacquire the crashed tool child")?;
    let bootstrap = resolve_tool_bootstrap(child, store, ActivityBeacon::default()).await?;
    run_tool_effect(
        bootstrap,
        store,
        &CancellationToken::new(),
        |_call, _collector| async {
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::success("ran again"))
        },
        at(603),
    )
    .await
}

async fn crash_and_recover(
    tool: &str,
) -> Result<(tempfile::TempDir, SqliteDurableStore, AgentTask, usize)> {
    let temp = tempfile::tempdir()?;
    let db_path = temp.path().join("tool-replay-crash.db");
    let ready_path = temp.path().join("tool-started");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let task_id = crash_while_tool_runs(&url, tool, &ready_path).await?;

    let store = SqliteDurableStore::connect(&url).await?;
    let runs = AtomicUsize::new(0);
    recover(&store, &task_id, &runs).await?;
    let child = AgentTaskStore::get(&store, &task_id)
        .await?
        .context("boot lost the tool child")?;
    Ok((temp, store, child, runs.load(Ordering::SeqCst)))
}

#[tokio::test]
async fn kill_nine_during_an_unsafe_tool_interrupts_it_instead_of_running_it_again() -> Result<()> {
    for tool in ["bash", "write", "edit", "link_fetch", UNDECLARED_TOOL] {
        let (_temp, store, child, runs) = crash_and_recover(tool).await?;
        ensure!(
            runs == 0,
            "{tool}: recovery ran the tool again ({runs} runs)"
        );
        ensure!(
            child.status == TaskStatus::Failed,
            "{tool}: the tool child is {:?}, not Failed",
            child.status
        );
        let error = child.last_error.as_deref().unwrap_or_default();
        ensure!(
            error.starts_with(TOOL_INTERRUPTED_PREFIX),
            "{tool}: the tool result is not an interrupted result: {error}"
        );

        let intent = store
            .effect(&EffectKey::tool(&child.id, CALL_ID))
            .await?
            .with_context(|| format!("{tool}: no effect record was written before the run"))?;
        ensure!(
            matches!(intent.state, EffectState::Interrupted { .. }),
            "{tool}: the interrupted effect is {:?}",
            intent.state
        );

        let ends: Vec<ToolResult> = EventRepository::get_events(&store, &child.thread_id)
            .await?
            .into_iter()
            .filter_map(|event| match event.event {
                AgentEvent::ToolCallEnd { id, result, .. } if id == CALL_ID => Some(result),
                _ => None,
            })
            .collect();
        let [end] = ends.as_slice() else {
            bail!("{tool}: expected one ToolCallEnd, got {ends:?}");
        };
        ensure!(
            !end.success && end.output == error,
            "{tool}: the ToolCallEnd does not carry the interrupted result: {end:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn kill_nine_during_a_safe_tool_runs_it_again() -> Result<()> {
    let advertised = auto_approved_tools();
    for tool in ["read", "ask_user"] {
        let declared = advertised
            .iter()
            .find(|candidate| candidate.name == tool)
            .with_context(|| format!("{tool} is not advertised"))?;
        ensure!(
            declared.replay == ToolReplay::Safe,
            "{tool} is not declared Safe"
        );

        let (_temp, _store, child, runs) = crash_and_recover(tool).await?;
        ensure!(
            runs == 1,
            "{tool}: recovery ran the safe tool {runs} times, not once"
        );
        ensure!(
            child.status == TaskStatus::Completed,
            "{tool}: the safe tool child is {:?}, not Completed",
            child.status
        );
    }
    Ok(())
}
