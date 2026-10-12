//! A turn parked across a store reopen must resume with its real tool
//! result: the reopen's repair sweep must not close the parked call.

#![cfg(feature = "sqlite")]

use std::sync::Mutex;

use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use agent_server::journal::task_state::TaskState;
use agent_server::journal::thread_store::ThreadStore;
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{
    RootTurnDeps, RootTurnOutcome, execute_root_turn, resume_from_children,
};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use time::OffsetDateTime;

const PARKED_CALL: &str = "parked-build-call";
const REAL_OUTPUT: &str = "build ok";

const fn usage() -> Usage {
    Usage {
        served_speed: None,
        input_tokens: 1,
        output_tokens: 1,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        request_input_tokens: Some(1),
    }
}

fn at(seconds: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(seconds)
}

struct ToolCallProvider;

#[async_trait]
impl LlmProvider for ToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "park-response".into(),
            content: vec![ContentBlock::ToolUse {
                id: PARKED_CALL.into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "cargo build"}),
                thought_signature: None,
            }],
            model: "park-test-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: usage(),
        }))
    }

    fn model(&self) -> &'static str {
        "park-test-model"
    }

    fn provider(&self) -> &'static str {
        "park-test"
    }
}

struct RecordingProvider {
    requests: Mutex<Vec<Vec<llm::Message>>>,
}

impl RecordingProvider {
    const fn new() -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Result<Vec<Vec<llm::Message>>> {
        Ok(self
            .requests
            .lock()
            .ok()
            .context("recording provider lock poisoned")?
            .clone())
    }
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.requests
            .lock()
            .ok()
            .context("recording provider lock poisoned")?
            .push(request.messages);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "resume-response".into(),
            content: vec![ContentBlock::Text {
                text: "the build passed".into(),
            }],
            model: "park-test-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: usage(),
        }))
    }

    fn model(&self) -> &'static str {
        "park-test-model"
    }

    fn provider(&self) -> &'static str {
        "park-test"
    }
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "park-test".into(),
        model: "park-test-model".into(),
        system_prompt: "Run the build.".into(),
        max_tokens: 256,
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run a shell command".into(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Bash".into(),
            tier: ToolTier::Observe,
        }],
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn deps<'a>(store: &'a SqliteDurableStore, notifier: &'a EventNotifier) -> RootTurnDeps<'a> {
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

fn worker() -> WorkerId {
    WorkerId::from_string("park-worker")
}

fn lease() -> LeaseId {
    LeaseId::from_string("park-lease")
}

async fn acquire(store: &SqliteDurableStore, task_id: &AgentTaskId, now: i64) -> Result<AgentTask> {
    store
        .try_acquire_task(task_id, worker(), lease(), at(now + 600), at(now))
        .await?
        .with_context(|| format!("acquire {task_id}"))
}

async fn root_inputs(
    store: &SqliteDurableStore,
    task: AgentTask,
    now: i64,
) -> Result<agent_server::journal::execution_context::RootWorkerInputs> {
    let bootstrap = WorkerBootstrapContext {
        thread_id: task.thread_id.clone(),
        task_id: task.id.clone(),
        task,
        worker_id: worker(),
        lease_id: lease(),
        definition: definition(),
    };
    build_root_worker_inputs(bootstrap, store, store, store, at(now)).await
}

/// Run a real root turn until the worker parks it behind its tool child,
/// exactly as production does (including the suspension-draft snapshot).
async fn park_behind_tool_child(store: &SqliteDurableStore) -> Result<(AgentTaskId, AgentTaskId)> {
    let thread_id = ThreadId::from_string("parked-across-reopen");
    ThreadStore::get_or_create(store, &thread_id, at(0)).await?;
    MessageProjectionStore::get_or_create(store, &thread_id, at(0)).await?;
    let root = AgentTask::new_root_turn(thread_id, at(0), 3);
    let root_id = root.id.clone();
    store.submit_root_turn(root).await?;
    let acquired = acquire(store, &root_id, 1).await?;
    let inputs = root_inputs(store, acquired, 1).await?;
    let notifier = EventNotifier::new();
    let outcome = execute_root_turn(
        inputs,
        "Run the build",
        &ToolCallProvider,
        &deps(store, &notifier),
        at(2),
    )
    .await?;
    let RootTurnOutcome::Suspended {
        parent_task,
        child_tasks,
        ..
    } = outcome
    else {
        bail!("the tool call did not park the root turn");
    };
    ensure!(parent_task.status == TaskStatus::WaitingOnChildren);
    let [child] = child_tasks.as_slice() else {
        bail!("expected one tool child, got {}", child_tasks.len());
    };
    Ok((root_id, child.id.clone()))
}

fn tool_use_count(messages: &[llm::Message], id: &str) -> usize {
    blocks(messages)
        .filter(|block| matches!(block, ContentBlock::ToolUse { id: use_id, .. } if use_id == id))
        .count()
}

fn tool_result_contents<'a>(messages: &'a [llm::Message], id: &str) -> Vec<&'a str> {
    blocks(messages)
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } if tool_use_id == id => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

fn blocks(messages: &[llm::Message]) -> impl Iterator<Item = &ContentBlock> {
    messages.iter().flat_map(|message| match &message.content {
        Content::Blocks(blocks) => blocks.as_slice(),
        Content::Text(_) => &[],
    })
}

#[tokio::test]
async fn turn_parked_across_a_store_reopen_resumes_with_its_real_tool_result() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("parked-turn.db").display()
    );

    let store = SqliteDurableStore::connect(&url).await?;
    let (root_id, child_id) = park_behind_tool_child(&store).await?;
    let thread_id = ThreadId::from_string("parked-across-reopen");
    let parked = MessageProjectionStore::get(&store, &thread_id)
        .await?
        .context("projection after park")?;
    ensure!(
        tool_use_count(&parked.draft_messages, PARKED_CALL) == 1,
        "the park must snapshot the pending tool_use into the draft"
    );
    drop(store);

    // Reopening runs the startup repair sweep, as a daemon restart does.
    let store = SqliteDurableStore::connect(&url).await?;
    let reopened = MessageProjectionStore::get(&store, &thread_id)
        .await?
        .context("projection after reopen")?;
    assert_eq!(
        serde_json::to_value(&reopened)?,
        serde_json::to_value(&parked)?,
        "the reopen sweep rewrote a parked turn's projection"
    );

    acquire(&store, &child_id, 10).await?;
    store
        .complete_task_with_result(
            &child_id,
            &worker(),
            &lease(),
            serde_json::to_value(ToolResult::success(REAL_OUTPUT))?,
            at(11),
        )
        .await?;
    let ready = AgentTaskStore::get(&store, &root_id)
        .await?
        .context("parent after child completion")?;
    ensure!(
        matches!(ready.state, TaskState::ReadyToResume { .. }),
        "the drained parent must be ready to resume: {:?}",
        ready.status
    );
    let acquired = acquire(&store, &root_id, 12).await?;
    let inputs = root_inputs(&store, acquired, 12).await?;
    let provider = RecordingProvider::new();
    let notifier = EventNotifier::new();
    let outcome =
        resume_from_children(inputs, &ready, &provider, &deps(&store, &notifier), at(13)).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "the resumed turn did not complete"
    );

    let requests = provider.requests()?;
    let [request] = requests.as_slice() else {
        bail!("expected one resume request, got {}", requests.len());
    };
    assert!(llm::is_provider_valid_tool_sequence(request));
    assert_eq!(
        tool_use_count(request, PARKED_CALL),
        1,
        "the resume request duplicated the parked tool_use"
    );
    assert_eq!(
        tool_result_contents(request, PARKED_CALL),
        vec![REAL_OUTPUT],
        "the resume request must carry exactly the real tool result"
    );

    let committed = MessageProjectionStore::get(&store, &thread_id)
        .await?
        .context("projection after resume")?;
    assert_eq!(
        tool_use_count(&committed.messages, PARKED_CALL),
        1,
        "the committed transcript duplicated the parked tool_use"
    );
    assert_eq!(
        tool_result_contents(&committed.messages, PARKED_CALL),
        vec![REAL_OUTPUT],
        "the committed transcript must record only the real tool result"
    );
    assert!(llm::is_provider_valid_tool_sequence(
        &committed.context_history()
    ));
    Ok(())
}
