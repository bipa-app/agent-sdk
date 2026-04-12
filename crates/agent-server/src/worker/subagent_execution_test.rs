//! End-to-end durable subagent execution tests.

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn, resume_from_children};
use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentSpec, SpawnedSubagentInvocation,
    SubagentInvocationDeps, SubagentResult, SubagentResultDeps, SubagentSummary,
    SubagentTaskOutcome, execute_subagent_task, resolve_subagent_bootstrap,
    spawn_subagent_invocation,
};
use super::tool_task::{ToolTaskOutcome, execute_tool_task, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, SubmittedInputItem, SuspensionPayload, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_core::{
    AgentContinuation, AgentState, PendingToolCallInfo, ThreadId, TokenUsage, ToolResult, ToolTier,
    events::AgentEvent, llm,
};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use time::{Duration, OffsetDateTime};
use tokio_util::sync::CancellationToken;

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
}

#[async_trait]
impl LlmProvider for MockTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_text".into(),
            content: vec![ContentBlock::Text {
                text: self.response_text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                cached_input_tokens: 0,
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

struct MockToolCallProvider {
    tool_calls: Vec<(String, String, serde_json::Value)>,
}

impl MockToolCallProvider {
    fn single(id: &str, name: &str, input: serde_json::Value) -> Self {
        Self {
            tool_calls: vec![(id.to_owned(), name.to_owned(), input)],
        }
    }
}

#[async_trait]
impl LlmProvider for MockToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_tool".into(),
            content: self
                .tool_calls
                .iter()
                .map(|(id, name, input)| ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    thought_signature: None,
                })
                .collect(),
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 120,
                output_tokens: 60,
                cached_input_tokens: 0,
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

struct FailingEventRepository;

#[async_trait]
impl EventRepository for FailingEventRepository {
    async fn commit_event(
        &self,
        _thread_id: &ThreadId,
        _event: AgentEvent,
        _now: OffsetDateTime,
    ) -> Result<crate::journal::CommittedEvent> {
        anyhow::bail!("synthetic event commit failure");
    }

    async fn commit_event_batch(
        &self,
        _thread_id: &ThreadId,
        _events: Vec<AgentEvent>,
        _now: OffsetDateTime,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        anyhow::bail!("synthetic event batch commit failure");
    }

    async fn next_sequence(&self, _thread_id: &ThreadId) -> Result<u64> {
        Ok(0)
    }

    async fn get_events(
        &self,
        _thread_id: &ThreadId,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
    }

    async fn get_events_in_range(
        &self,
        _thread_id: &ThreadId,
        _after_sequence: u64,
        _up_to_sequence: u64,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
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

    fn root_turn_deps(&self) -> RootTurnDeps<'_> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
            event_repo: &self.events,
        }
    }

    fn subagent_deps(&self) -> SubagentResultDeps<'_> {
        SubagentResultDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            event_repo: &self.events,
        }
    }
}

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a durable test assistant.".into(),
        max_tokens: 1024,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        policy: RuntimePolicy::server_default(),
    }
}

fn child_definition_with_tools() -> AgentDefinition {
    AgentDefinition {
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run shell commands".into(),
            input_schema: json!({"type":"object","properties":{"command":{"type":"string"}}}),
            display_name: "Bash".into(),
            tier: ToolTier::Observe,
        }],
        ..sample_definition()
    }
}

fn bootstrap(task: AgentTask, definition: AgentDefinition) -> Result<WorkerBootstrapContext> {
    let worker_id = task
        .worker_id
        .clone()
        .context("running task missing worker_id")?;
    let lease_id = task
        .lease_id
        .clone()
        .context("running task missing lease_id")?;
    Ok(WorkerBootstrapContext {
        thread_id: task.thread_id.clone(),
        task_id: task.id.clone(),
        worker_id,
        lease_id,
        task,
        definition,
    })
}

fn parent_suspension_payload(thread_id: &ThreadId) -> SuspensionPayload {
    let tool_call = PendingToolCallInfo {
        id: "call_subagent".into(),
        name: "subagent_researcher".into(),
        display_name: "Subagent: Researcher".into(),
        tier: ToolTier::Confirm,
        input: json!({"task":"Investigate durable reuse"}),
        effective_input: json!({"task":"Investigate durable reuse"}),
        listen_context: None,
    };
    SuspensionPayload {
        continuation: agent_sdk_core::ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread_id.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![tool_call],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread_id.clone()),
            response_id: None,
            stop_reason: Some(StopReason::ToolUse),
            response_content: vec![ContentBlock::ToolUse {
                id: "call_subagent".into(),
                name: "subagent_researcher".into(),
                input: json!({"task":"Investigate durable reuse"}),
                thought_signature: None,
            }],
        }),
        suspended_messages: vec![
            llm::Message::user("Investigate durable reuse"),
            llm::Message::assistant_with_tool_use(
                None,
                "call_subagent",
                "subagent_researcher",
                json!({"task":"Investigate durable reuse"}),
            ),
        ],
    }
}

fn subagent_spec() -> EffectiveSubagentSpec {
    EffectiveSubagentSpec {
        task: "Investigate durable reuse".into(),
        prompt: "Stay in read-only mode.".into(),
        model: "mock-model".into(),
        max_turns: 4,
        timeout_ms: 30_000,
        nickname: Some("Scout".into()),
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["read_file", "rg"]),
        },
    }
}

fn submitted_text(task: &AgentTask) -> Result<String> {
    task.submitted_input
        .iter()
        .map(|item| match item {
            SubmittedInputItem::Text { text } => Ok(text.clone()),
            other => anyhow::bail!("unsupported submitted input in test: {other:?}"),
        })
        .collect::<Result<Vec<_>>>()
        .map(|parts| parts.join("\n"))
}

async fn create_running_parent_root(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let parent = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let parent_id = parent.id.clone();
    store.submit_root_turn(parent).await?;
    let worker = WorkerId::from_string("w-parent");
    let lease = LeaseId::from_string("l-parent");
    let claimed = store
        .try_acquire_task(
            &parent_id,
            worker.clone(),
            lease.clone(),
            t_plus(60),
            t_plus(1),
        )
        .await?
        .context("claim parent root")?;
    Ok((claimed, worker, lease))
}

fn child_spawn(thread_id: &ThreadId) -> crate::journal::SubagentInvocationSpawn {
    crate::journal::SubagentInvocationSpawn {
        child_thread_id: ThreadId::new(),
        spec: subagent_spec(),
        child_root_input: vec![SubmittedInputItem::Text {
            text: "Stay in read-only mode.\n\nInvestigate durable reuse".into(),
        }],
        spawn_index: 0,
        payload: parent_suspension_payload(thread_id),
    }
}

async fn suspend_child_root_on_tool_call(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<Vec<AgentTask>> {
    let child_running = stores
        .tasks
        .try_acquire_task(
            &spawned.child_root_task.id,
            WorkerId::from_string("w-child-root"),
            LeaseId::from_string("l-child-root"),
            t_plus(60),
            t_plus(3),
        )
        .await?
        .context("claim child root")?;
    let child_inputs = build_root_worker_inputs(
        bootstrap(child_running.clone(), child_definition_with_tools())?,
        &stores.threads,
        &stores.checkpoints,
        t_plus(3),
    )
    .await?;
    let child_prompt = submitted_text(&child_running)?;
    let child_first = execute_root_turn(
        child_inputs,
        &child_prompt,
        &MockToolCallProvider::single("call_child_bash", "bash", json!({"command":"ls"})),
        &stores.root_turn_deps(),
        t_plus(4),
    )
    .await?;
    let RootTurnOutcome::Suspended {
        child_tasks: tool_tasks,
        ..
    } = child_first
    else {
        anyhow::bail!("expected child root to suspend on tool call");
    };
    assert_eq!(tool_tasks.len(), 1);

    Ok(tool_tasks)
}

async fn run_child_until_ready_to_resume(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<AgentTask> {
    let tool_tasks = suspend_child_root_on_tool_call(stores, spawned).await?;

    let tool_running = stores
        .tasks
        .try_acquire_task(
            &tool_tasks[0].id,
            WorkerId::from_string("w-child-tool"),
            LeaseId::from_string("l-child-tool"),
            t_plus(60),
            t_plus(5),
        )
        .await?
        .context("claim child tool")?;
    let tool_bootstrap = resolve_tool_bootstrap(tool_running, &stores.tasks).await?;
    let tool_outcome = execute_tool_task(
        tool_bootstrap,
        &stores.tasks,
        &stores.events,
        &CancellationToken::new(),
        |_tool_call, _collector| async { Ok(ToolResult::success("child ls output")) },
        t_plus(6),
    )
    .await?;
    let ToolTaskOutcome::Completed {
        parent: child_after_tool,
        ..
    } = tool_outcome
    else {
        anyhow::bail!("expected child tool task to complete successfully");
    };
    let child_after_tool =
        child_after_tool.context("child root should be returned after tool completion")?;
    assert_eq!(child_after_tool.id, spawned.child_root_task.id);
    assert_eq!(
        child_after_tool.status,
        crate::journal::task::TaskStatus::Pending
    );
    assert!(matches!(
        child_after_tool.state,
        crate::journal::task_state::TaskState::ReadyToResume { .. }
    ));

    let persisted_child_root = stores
        .tasks
        .get(&spawned.child_root_task.id)
        .await?
        .context("load child root after tool completion")?;
    assert_eq!(
        persisted_child_root.status,
        crate::journal::task::TaskStatus::Pending
    );
    assert!(matches!(
        persisted_child_root.state,
        crate::journal::task_state::TaskState::ReadyToResume { .. }
    ));

    Ok(persisted_child_root)
}

async fn resume_child_root_to_completion(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
    persisted_child_root: &AgentTask,
) -> Result<()> {
    let child_resumable = stores
        .tasks
        .try_acquire_task(
            &spawned.child_root_task.id,
            WorkerId::from_string("w-child-resume"),
            LeaseId::from_string("l-child-resume"),
            t_plus(60),
            t_plus(7),
        )
        .await?
        .with_context(|| {
            format!(
                "reclaim child root from status {:?}",
                persisted_child_root.status
            )
        })?;
    let child_resume_inputs = build_root_worker_inputs(
        bootstrap(child_resumable.clone(), child_definition_with_tools())?,
        &stores.threads,
        &stores.checkpoints,
        t_plus(7),
    )
    .await?;
    let child_resume = resume_from_children(
        child_resume_inputs,
        &child_resumable,
        &MockTextProvider::new("child final response"),
        &stores.root_turn_deps(),
        t_plus(8),
    )
    .await?;
    let RootTurnOutcome::Completed { completed_task, .. } = child_resume else {
        anyhow::bail!("expected child root to complete after resume");
    };
    assert_eq!(completed_task.id, spawned.child_root_task.id);

    let invocation_pending = stores
        .tasks
        .get(&spawned.invocation_task.id)
        .await?
        .context("load invocation after child completion")?;
    assert_eq!(
        invocation_pending.status,
        crate::journal::task::TaskStatus::Pending
    );
    assert!(invocation_pending.state.subagent_invocation().is_some());

    Ok(())
}

async fn execute_subagent_invocation(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<SubagentTaskOutcome> {
    let invocation_running = stores
        .tasks
        .try_acquire_task(
            &spawned.invocation_task.id,
            WorkerId::from_string("w-subagent"),
            LeaseId::from_string("l-subagent"),
            t_plus(60),
            t_plus(9),
        )
        .await?
        .context("claim invocation task")?;
    let subagent_bootstrap = resolve_subagent_bootstrap(invocation_running, &stores.tasks).await?;
    execute_subagent_task(subagent_bootstrap, &stores.subagent_deps(), t_plus(10)).await
}

async fn materialize_subagent_result(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<SubagentTaskOutcome> {
    let subagent_outcome = execute_subagent_invocation(stores, spawned).await?;
    assert_eq!(
        subagent_outcome.invocation_task.status,
        crate::journal::task::TaskStatus::Completed
    );
    assert!(subagent_outcome.tool_result.success);
    assert_eq!(subagent_outcome.tool_result.output, "child final response");

    let result_data = subagent_outcome
        .tool_result
        .data
        .clone()
        .context("subagent tool result data missing")?;
    let structured: SubagentResult =
        serde_json::from_value(result_data).context("decode SubagentResult")?;
    assert_eq!(structured.final_response, "child final response");
    assert_eq!(structured.summary.total_turns, 1);
    assert_eq!(structured.summary.tool_count, 1);
    assert!(structured.summary.success);
    assert_eq!(structured.child_thread_id, spawned.child_thread.thread_id);
    assert_eq!(structured.child_root_task_id, spawned.child_root_task.id);
    assert_eq!(structured.subagent_task_id, spawned.invocation_task.id);
    assert_parent_summary_progress_events(stores, spawned, &structured.summary).await?;
    assert_persisted_invocation_result(stores, spawned).await?;

    Ok(subagent_outcome)
}

struct ExpectedSubagentProgress<'a> {
    spawned: &'a SpawnedSubagentInvocation,
    subagent_name: &'a str,
    completed: bool,
    success: bool,
    current_turn: u32,
    tool_count: u32,
    total_tokens: u64,
}

async fn assert_parent_summary_progress_events(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
    summary: &SubagentSummary,
) -> Result<()> {
    let parent_events = stores
        .events
        .get_events(&spawned.parent_task.thread_id)
        .await?;
    assert_eq!(parent_events.len(), 2);
    assert!(
        parent_events
            .iter()
            .all(|event| matches!(event.event, AgentEvent::SubagentProgress { .. }))
    );

    assert_subagent_progress_event(
        &parent_events[0].event,
        "spawn",
        &ExpectedSubagentProgress {
            spawned,
            subagent_name: "researcher",
            completed: false,
            success: false,
            current_turn: 0,
            tool_count: 0,
            total_tokens: 0,
        },
    )?;
    assert_subagent_progress_event(
        &parent_events[1].event,
        "completion",
        &ExpectedSubagentProgress {
            spawned,
            subagent_name: "researcher",
            completed: true,
            success: true,
            current_turn: summary.total_turns,
            tool_count: summary.tool_count,
            total_tokens: u64::from(summary.total_usage.input_tokens)
                + u64::from(summary.total_usage.output_tokens),
        },
    )?;

    Ok(())
}

fn assert_subagent_progress_event(
    event: &AgentEvent,
    phase: &str,
    expected: &ExpectedSubagentProgress<'_>,
) -> Result<()> {
    let child_root_task_id = expected.spawned.child_root_task.id.to_string();
    let subagent_task_id = expected.spawned.invocation_task.id.to_string();

    match event {
        AgentEvent::SubagentProgress {
            subagent_name,
            child_thread_id,
            child_root_task_id: actual_child_root_task_id,
            subagent_task_id: actual_subagent_task_id,
            tool_name,
            completed,
            success,
            current_turn,
            tool_count,
            total_tokens,
            ..
        } => {
            assert_eq!(subagent_name, expected.subagent_name);
            assert_eq!(
                child_thread_id.as_ref(),
                Some(&expected.spawned.child_thread.thread_id)
            );
            assert_eq!(
                actual_child_root_task_id.as_deref(),
                Some(child_root_task_id.as_str())
            );
            assert_eq!(
                actual_subagent_task_id.as_deref(),
                Some(subagent_task_id.as_str())
            );
            assert_eq!(tool_name, expected.subagent_name);
            assert_eq!(*completed, expected.completed);
            assert_eq!(*success, expected.success);
            assert_eq!(*current_turn, Some(expected.current_turn));
            assert_eq!(*tool_count, expected.tool_count);
            assert_eq!(*total_tokens, expected.total_tokens);
            Ok(())
        }
        other => anyhow::bail!("expected {phase} SubagentProgress, got {other:?}"),
    }
}

async fn assert_persisted_invocation_result(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<()> {
    let persisted_invocation = stores
        .tasks
        .get(&spawned.invocation_task.id)
        .await?
        .context("load completed invocation")?;
    let persisted_payload = persisted_invocation
        .result_payload
        .context("invocation result_payload missing")?;
    let persisted_tool_result: ToolResult = serde_json::from_value(persisted_payload)?;
    assert_eq!(persisted_tool_result.output, "child final response");

    Ok(())
}

#[tokio::test]
async fn cancelled_child_thread_does_not_count_unexecuted_tool_tasks() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread_id = ThreadId::from_string("t-parent-subagent-cancel");
    let (parent, worker, lease) =
        create_running_parent_root(&stores.tasks, &parent_thread_id).await?;

    let spawned: SpawnedSubagentInvocation = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        child_spawn(&parent_thread_id),
        &SubagentInvocationDeps {
            task_store: &stores.tasks,
            thread_store: &stores.threads,
            event_repo: &stores.events,
        },
        t_plus(2),
    )
    .await?;

    let tool_tasks = suspend_child_root_on_tool_call(&stores, &spawned).await?;
    let transitioned = stores
        .tasks
        .cancel_tree(&spawned.child_root_task.id, t_plus(5))
        .await?;
    assert!(transitioned.contains(&spawned.child_root_task.id));
    assert!(transitioned.contains(&tool_tasks[0].id));

    let subagent_outcome = execute_subagent_invocation(&stores, &spawned).await?;
    assert!(!subagent_outcome.tool_result.success);

    let result_data = subagent_outcome
        .tool_result
        .data
        .context("subagent tool result data missing")?;
    let structured: SubagentResult =
        serde_json::from_value(result_data).context("decode cancelled SubagentResult")?;
    assert_eq!(structured.summary.tool_count, 0);
    assert!(!structured.summary.success);
    assert_eq!(
        structured.error_details.as_deref(),
        Some("subagent execution was cancelled")
    );

    Ok(())
}

#[tokio::test]
async fn completion_tolerates_parent_progress_commit_failures() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread_id = ThreadId::from_string("t-parent-subagent-progress-failure");
    let (parent, worker, lease) =
        create_running_parent_root(&stores.tasks, &parent_thread_id).await?;

    let spawned: SpawnedSubagentInvocation = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        child_spawn(&parent_thread_id),
        &SubagentInvocationDeps {
            task_store: &stores.tasks,
            thread_store: &stores.threads,
            event_repo: &stores.events,
        },
        t_plus(2),
    )
    .await?;

    let persisted_child_root = run_child_until_ready_to_resume(&stores, &spawned).await?;
    resume_child_root_to_completion(&stores, &spawned, &persisted_child_root).await?;

    let invocation_running = stores
        .tasks
        .try_acquire_task(
            &spawned.invocation_task.id,
            WorkerId::from_string("w-subagent-failing-progress"),
            LeaseId::from_string("l-subagent-failing-progress"),
            t_plus(60),
            t_plus(9),
        )
        .await?
        .context("claim invocation task")?;
    let subagent_bootstrap = resolve_subagent_bootstrap(invocation_running, &stores.tasks).await?;
    let failing_events = FailingEventRepository;
    let subagent_outcome = execute_subagent_task(
        subagent_bootstrap,
        &SubagentResultDeps {
            task_store: &stores.tasks,
            thread_store: &stores.threads,
            message_store: &stores.messages,
            event_repo: &failing_events,
        },
        t_plus(10),
    )
    .await?;

    assert!(subagent_outcome.tool_result.success);
    assert!(subagent_outcome.committed_events.is_empty());

    Ok(())
}

async fn resume_parent_after_subagent(
    stores: &TestStores,
    parent: &AgentTask,
) -> Result<AgentTask> {
    let parent_running = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("w-parent-resume"),
            LeaseId::from_string("l-parent-resume"),
            t_plus(60),
            t_plus(11),
        )
        .await?
        .context("claim parent root for resume")?;
    let parent_inputs = build_root_worker_inputs(
        bootstrap(parent_running.clone(), sample_definition())?,
        &stores.threads,
        &stores.checkpoints,
        t_plus(11),
    )
    .await?;
    let parent_resume = resume_from_children(
        parent_inputs,
        &parent_running,
        &MockTextProvider::new("parent final response"),
        &stores.root_turn_deps(),
        t_plus(12),
    )
    .await?;
    let RootTurnOutcome::Completed {
        response_text,
        completed_task,
        ..
    } = parent_resume
    else {
        anyhow::bail!("expected parent root to complete after subagent result submission");
    };
    assert_eq!(response_text, "parent final response");
    assert_eq!(
        completed_task.status,
        crate::journal::task::TaskStatus::Completed
    );

    Ok(completed_task)
}

#[tokio::test]
async fn child_thread_reuses_root_turn_and_tool_runtime_before_materializing_parent_result()
-> Result<()> {
    let stores = TestStores::new();
    let parent_thread_id = ThreadId::from_string("t-parent-subagent-exec");
    let (parent, worker, lease) =
        create_running_parent_root(&stores.tasks, &parent_thread_id).await?;

    let spawned: SpawnedSubagentInvocation = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        child_spawn(&parent_thread_id),
        &SubagentInvocationDeps {
            task_store: &stores.tasks,
            thread_store: &stores.threads,
            event_repo: &stores.events,
        },
        t_plus(2),
    )
    .await?;

    let persisted_child_root = run_child_until_ready_to_resume(&stores, &spawned).await?;
    resume_child_root_to_completion(&stores, &spawned, &persisted_child_root).await?;
    let subagent_outcome = materialize_subagent_result(&stores, &spawned).await?;
    let child_events = stores
        .events
        .get_events(&spawned.child_thread.thread_id)
        .await?;
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event.event, AgentEvent::ToolCallStart { .. }))
    );
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event.event, AgentEvent::ToolCallEnd { .. }))
    );
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event.event, AgentEvent::Done { .. }))
    );
    let parent_ready = subagent_outcome
        .parent_task
        .clone()
        .context("parent root should be ready after invocation completion")?;
    assert_eq!(
        parent_ready.status,
        crate::journal::task::TaskStatus::Pending
    );
    assert!(matches!(
        parent_ready.state,
        crate::journal::task_state::TaskState::ReadyToResume { .. }
    ));
    let completed_parent = resume_parent_after_subagent(&stores, &parent).await?;
    assert_eq!(completed_parent.id, parent.id);

    Ok(())
}
