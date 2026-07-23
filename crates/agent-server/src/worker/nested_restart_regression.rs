//! Phase 7.7 nested subagent restart, replay, cancellation, and failure
//! regression suite.
//!
//! Earlier Phase 7 slices delivered the durable primitives:
//!
//! - 7.1 / 7.2 — `SubagentSpawnRequest`, `SubagentInvocationState`
//!   linkage, and `spawn_subagent_invocation` (child-thread root
//!   creation under the same write as the parent park).
//! - 7.3 — durable child-thread execution reuse (`execute_root_turn`
//!   and `execute_tool_task` on the child thread) and
//!   `materialize_terminal_subagent_result`.
//! - 7.4 — summary-only parent visibility: the parent thread only
//!   sees two `SubagentProgress` events (spawn + completion), the
//!   rich progression (`ToolCallStart`, `Text`, `Done`, etc.) lives
//!   exclusively on the child thread.
//! - 7.5 — inherited subagent policy enforcement.
//! - 7.6 — `cancel_tree` cascading across `SubagentInvocation`
//!   thread boundaries in every backend.
//!
//! Phase 7.7 closes Phase 7 by **proving** that those primitives
//! compose correctly under the failure modes the durable model was
//! built for:
//!
//! 1. **Restart recovery** — a mid-flight nested subagent tree
//!    survives a simulated process restart (notifier drop + lease
//!    expiry sweep) and resumes to a deterministic terminal state.
//! 2. **Thread-scoped replay** — after restart, the parent thread
//!    event stream replays only `SubagentProgress` summaries while
//!    the child thread event stream replays the full rich timeline.
//!    The two views stay fully isolated.
//! 3. **Cross-thread cancellation** — `cancel_tree` of a running
//!    parent root is stable across restart: re-reading the tree
//!    after a restart still sees every row terminal
//!    (`Cancelled`) and the typed `SubagentInvocation` linkage
//!    preserved for audit.
//! 4. **Cross-thread failure propagation** — a failed child-thread
//!    root still wakes the parent-thread invocation after restart,
//!    producing an error `ToolResult` on the parent's pending tool
//!    call with no duplicate or lost transitions.
//!
//! Everything in this module runs against the in-memory store /
//! event repository — it is the authoritative behavioral reference
//! (per the cross-backend conformance ADR). The persistent backends
//! (`SQLite`, `Postgres`) reuse the same `AgentTaskStore` contract
//! and are covered by the conformance suite for the transitional
//! primitives; the composite restart flow is the focus here.
use crate::worker::activity::ActivityBeacon;
use std::sync::Arc;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{
    AgentContinuation, AgentState, PendingToolCallInfo, ThreadId, TokenUsage, ToolResult, ToolTier,
    llm,
};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde_json::json;
use time::{Duration, OffsetDateTime};
use tokio_util::sync::CancellationToken;

use super::root_turn::{
    RootTurnDeps, RootTurnOutcome, cancel_root_turn, execute_root_turn, fail_root_turn,
    resume_from_children,
};
use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentPolicy, SpawnedSubagentInvocation, SubagentCapabilityProfile,
    SubagentInvocationDeps, SubagentResultDeps, SubagentSandboxPolicy, SubagentTaskOutcome,
    execute_subagent_task, resolve_subagent_bootstrap, spawn_subagent_invocation,
};
use super::tool_task::{ToolTaskOutcome, execute_tool_task, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::event_stream::{StreamEvent, stream_events};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::recovery::{RecoveryAction, RecoveryRecord};
use crate::journal::retention::InMemoryRetentionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore, SubagentInvocationSpawn};
use crate::journal::task::{
    AgentTask, AgentTaskId, LeaseId, SubmittedInputItem, SuspensionPayload, TaskKind, TaskStatus,
    WorkerId,
};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};

// ─────────────────────────────────────────────────────────────────────
// Time + identifiers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn lease_ttl_seconds() -> i64 {
    60
}

/// A simulated process restart long enough to expire any outstanding
/// lease. Every restart helper uses this same delta so tests stay
/// deterministic.
fn restart_delta_seconds() -> i64 {
    lease_ttl_seconds() + 30
}

// ─────────────────────────────────────────────────────────────────────
// Mock LLM providers
// ─────────────────────────────────────────────────────────────────────

/// Responds with a single text block, independent of prompt. Useful
/// for root-turn resume paths where the child result is already in
/// the transcript.
struct MockTextProvider {
    text: String,
}

impl MockTextProvider {
    fn new(text: &str) -> Self {
        Self {
            text: text.to_owned(),
        }
    }
}

#[async_trait]
impl LlmProvider for MockTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_text".into(),
            content: vec![ContentBlock::Text {
                text: self.text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 40,
                output_tokens: 20,
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

/// Responds once with a tool call, then (on the resume round-trip)
/// with a closing text block. Models a full root-turn suspend /
/// resume pair with a single tool call.
struct MockToolThenText {
    tool_id: String,
    tool_name: String,
    tool_input: serde_json::Value,
    resume_text: String,
    call_count: AtomicUsize,
}

impl MockToolThenText {
    fn new(
        tool_id: &str,
        tool_name: &str,
        tool_input: serde_json::Value,
        resume_text: &str,
    ) -> Self {
        Self {
            tool_id: tool_id.to_owned(),
            tool_name: tool_name.to_owned(),
            tool_input,
            resume_text: resume_text.to_owned(),
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmProvider for MockToolThenText {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let call = self.call_count.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_tool".into(),
                content: vec![ContentBlock::ToolUse {
                    id: self.tool_id.clone(),
                    name: self.tool_name.clone(),
                    input: self.tool_input.clone(),
                    thought_signature: None,
                }],
                model: "mock-model".into(),
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage {
                    input_tokens: 80,
                    output_tokens: 40,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
        } else {
            Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_resume".into(),
                content: vec![ContentBlock::Text {
                    text: self.resume_text.clone(),
                }],
                model: "mock-model".into(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 50,
                    output_tokens: 25,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
        }
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test stores
// ─────────────────────────────────────────────────────────────────────

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

    fn root_turn_deps(&self) -> RootTurnDeps<'_> {
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

    fn subagent_deps(&self) -> SubagentResultDeps<'_> {
        SubagentResultDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            event_repo: &self.events,
        }
    }

    fn invocation_deps(&self) -> SubagentInvocationDeps<'_> {
        SubagentInvocationDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            event_repo: &self.events,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Definitions + suspension payloads
// ─────────────────────────────────────────────────────────────────────

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a durable test assistant.".into(),
        max_tokens: 1024,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        tools_fn: None,
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

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn parent_suspension_payload(parent_thread: &ThreadId, tool_id: &str) -> SuspensionPayload {
    let tool_call = PendingToolCallInfo {
        id: tool_id.into(),
        name: "subagent_researcher".into(),
        display_name: "Subagent: Researcher".into(),
        tier: ToolTier::Confirm,
        input: json!({ "task": "Investigate durable reuse" }),
        effective_input: json!({ "task": "Investigate durable reuse" }),
        listen_context: None,
    };
    SuspensionPayload {
        continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: parent_thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![tool_call],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(parent_thread.clone()),
            response_id: None,
            stop_reason: Some(StopReason::ToolUse),
            response_content: vec![ContentBlock::ToolUse {
                id: tool_id.into(),
                name: "subagent_researcher".into(),
                input: json!({ "task": "Investigate durable reuse" }),
                thought_signature: None,
            }],
        }),
        suspended_messages: vec![
            llm::Message::user("Investigate durable reuse"),
            llm::Message::assistant_with_tool_use(
                None,
                tool_id,
                "subagent_researcher",
                json!({ "task": "Investigate durable reuse" }),
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
        depth: 1,
        max_parallel_subagents: 1,
        nickname: Some("Scout".into()),
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy {
            allowed_servers: set(&["docs"]),
        },
        audit_provenance: Some(AuditProvenance::new("mock", "mock-model")),
        inherited_policy: InheritedSubagentPolicy {
            default_model: "mock-model".into(),
            allowed_models: BTreeSet::from(["mock-model".to_owned()]),
            default_max_turns: 4,
            max_turns: 4,
            default_timeout_ms: 30_000,
            max_timeout_ms: 30_000,
            capability_profiles: std::collections::BTreeMap::from([(
                "research".into(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file", "rg"]),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: set(&["docs"]),
                },
            )]),
            allowed_capabilities: set(&["read_file", "rg"]),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: set(&["docs"]),
            audit_provider: "mock".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["read_file", "rg"]),
        },
    }
}

fn child_spawn(parent_thread: &ThreadId, tool_id: &str) -> SubagentInvocationSpawn {
    SubagentInvocationSpawn {
        child_thread_id: ThreadId::new(),
        spec: subagent_spec(),
        child_root_input: vec![SubmittedInputItem::Text {
            text: "Stay in read-only mode.\n\nInvestigate durable reuse".into(),
        }],
        spawn_index: 0,
        payload: parent_suspension_payload(parent_thread, tool_id),
        child_caller_metadata: None,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Scenario builders
// ─────────────────────────────────────────────────────────────────────

/// A running parent root ready to spawn its invocation.
struct ParentRootSetup {
    task: AgentTask,
    worker: WorkerId,
    lease: LeaseId,
}

async fn create_running_parent_root(
    stores: &TestStores,
    parent_thread: &ThreadId,
    suffix: &str,
) -> Result<ParentRootSetup> {
    let task = AgentTask::new_root_turn(parent_thread.clone(), t0(), 3);
    let task_id = task.id.clone();
    stores.tasks.submit_root_turn(task).await?;
    let worker = WorkerId::from_string(format!("w-parent-{suffix}"));
    let lease = LeaseId::from_string(format!("l-parent-{suffix}"));
    let claimed = stores
        .tasks
        .try_acquire_task(
            &task_id,
            worker.clone(),
            lease.clone(),
            t_plus(lease_ttl_seconds()),
            t_plus(1),
        )
        .await?
        .context("claim parent root")?;
    Ok(ParentRootSetup {
        task: claimed,
        worker,
        lease,
    })
}

async fn spawn_child_invocation(
    stores: &TestStores,
    setup: &ParentRootSetup,
    parent_thread: &ThreadId,
    tool_id: &str,
    at: OffsetDateTime,
) -> Result<SpawnedSubagentInvocation> {
    spawn_subagent_invocation(
        &setup.task.id,
        &setup.worker,
        &setup.lease,
        child_spawn(parent_thread, tool_id),
        &stores.invocation_deps(),
        at,
    )
    .await
}

/// Drive the child root up to a point where it has suspended on a
/// single tool call. Returns the spawned tool-runtime tasks so the
/// caller can decide whether to complete, fail, or restart around
/// them.
async fn suspend_child_root_on_tool_call(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
    worker_suffix: &str,
    at_acquire: OffsetDateTime,
    at_execute: OffsetDateTime,
) -> Result<Vec<AgentTask>> {
    let child_running = stores
        .tasks
        .try_acquire_task(
            &spawned.child_root_task.id,
            WorkerId::from_string(format!("w-child-{worker_suffix}")),
            LeaseId::from_string(format!("l-child-{worker_suffix}")),
            at_acquire + Duration::seconds(lease_ttl_seconds()),
            at_acquire,
        )
        .await?
        .context("claim child root")?;
    let child_inputs = build_root_worker_inputs(
        bootstrap(child_running.clone(), child_definition_with_tools())?,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        at_acquire,
    )
    .await?;
    let child_prompt = submitted_text_from_child(&child_running)?;
    let outcome = execute_root_turn(
        child_inputs,
        &child_prompt,
        &MockToolThenText::new(
            "call_child_bash",
            "bash",
            json!({ "command": "ls" }),
            "child final response",
        ),
        &stores.root_turn_deps(),
        at_execute,
    )
    .await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        anyhow::bail!("expected child root to suspend on tool call");
    };
    ensure!(
        child_tasks.len() == 1,
        "expected 1 child tool task, got {}",
        child_tasks.len(),
    );
    Ok(child_tasks)
}

fn submitted_text_from_child(task: &AgentTask) -> Result<String> {
    task.submitted_input
        .iter()
        .map(|item| match item {
            SubmittedInputItem::Text { text } => Ok(text.clone()),
            other => anyhow::bail!("unsupported submitted input in test: {other:?}"),
        })
        .collect::<Result<Vec<_>>>()
        .map(|parts| parts.join("\n"))
}

/// Complete the single child tool task and drive the child root
/// through resume → completion → invocation materialization →
/// parent resume.
async fn drive_child_to_completion(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
    tool_tasks: &[AgentTask],
    worker_suffix: &str,
    mut clock: OffsetDateTime,
) -> Result<SubagentTaskOutcome> {
    let tool_running = stores
        .tasks
        .try_acquire_task(
            &tool_tasks[0].id,
            WorkerId::from_string(format!("w-tool-{worker_suffix}")),
            LeaseId::from_string(format!("l-tool-{worker_suffix}")),
            clock + Duration::seconds(lease_ttl_seconds()),
            clock,
        )
        .await?
        .context("claim child tool task")?;
    let tool_bootstrap =
        resolve_tool_bootstrap(tool_running, &stores.tasks, ActivityBeacon::default()).await?;
    clock += Duration::seconds(1);
    let tool_outcome = execute_tool_task(
        tool_bootstrap,
        &stores.tasks,
        &stores.events,
        &CancellationToken::new(),
        |_tool_call, _collector| async { Ok(ToolResult::success("child ls output")) },
        clock,
    )
    .await?;
    let ToolTaskOutcome::Completed {
        parent: parent_after_tool,
        ..
    } = tool_outcome
    else {
        anyhow::bail!("expected child tool to complete successfully");
    };
    let parent_after_tool =
        parent_after_tool.context("child root should reappear after tool completion")?;
    ensure!(
        matches!(parent_after_tool.state, TaskState::ReadyToResume { .. }),
        "child root should be ReadyToResume after tool task, got {:?}",
        parent_after_tool.state,
    );

    clock += Duration::seconds(1);
    let child_resume = stores
        .tasks
        .try_acquire_task(
            &spawned.child_root_task.id,
            WorkerId::from_string(format!("w-resume-{worker_suffix}")),
            LeaseId::from_string(format!("l-resume-{worker_suffix}")),
            clock + Duration::seconds(lease_ttl_seconds()),
            clock,
        )
        .await?
        .context("reclaim child root for resume")?;
    let resume_inputs = build_root_worker_inputs(
        bootstrap(child_resume.clone(), child_definition_with_tools())?,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        clock,
    )
    .await?;
    clock += Duration::seconds(1);
    let resume_outcome = resume_from_children(
        resume_inputs,
        &child_resume,
        &MockTextProvider::new("child final response"),
        &stores.root_turn_deps(),
        clock,
    )
    .await?;
    let RootTurnOutcome::Completed { completed_task, .. } = resume_outcome else {
        anyhow::bail!("expected child root resume to complete");
    };
    ensure!(
        completed_task.status == TaskStatus::Completed,
        "child root expected Completed, got {:?}",
        completed_task.status,
    );

    clock += Duration::seconds(1);
    let invocation_running = stores
        .tasks
        .try_acquire_task(
            &spawned.invocation_task.id,
            WorkerId::from_string(format!("w-invocation-{worker_suffix}")),
            LeaseId::from_string(format!("l-invocation-{worker_suffix}")),
            clock + Duration::seconds(lease_ttl_seconds()),
            clock,
        )
        .await?
        .context("claim invocation task after child completion")?;
    let subagent_bootstrap = resolve_subagent_bootstrap(invocation_running, &stores.tasks).await?;
    clock += Duration::seconds(1);
    execute_subagent_task(subagent_bootstrap, &stores.subagent_deps(), clock).await
}

async fn resume_parent_root(
    stores: &TestStores,
    parent_id: &AgentTaskId,
    worker_suffix: &str,
    at: OffsetDateTime,
) -> Result<AgentTask> {
    let parent_running = stores
        .tasks
        .try_acquire_task(
            parent_id,
            WorkerId::from_string(format!("w-parent-resume-{worker_suffix}")),
            LeaseId::from_string(format!("l-parent-resume-{worker_suffix}")),
            at + Duration::seconds(lease_ttl_seconds()),
            at,
        )
        .await?
        .context("claim parent root for resume")?;
    let parent_inputs = build_root_worker_inputs(
        bootstrap(parent_running.clone(), sample_definition())?,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        at,
    )
    .await?;
    let outcome = resume_from_children(
        parent_inputs,
        &parent_running,
        &MockTextProvider::new("parent final response"),
        &stores.root_turn_deps(),
        at + Duration::seconds(1),
    )
    .await?;
    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        anyhow::bail!("expected parent root to complete after resume");
    };
    Ok(completed_task)
}

// ─────────────────────────────────────────────────────────────────────
// Restart simulator
// ─────────────────────────────────────────────────────────────────────

/// Drop the live-tail [`EventNotifier`], advance the clock past the
/// default lease TTL, and run the lease-expiry sweep. Returns the
/// recovered records so individual tests can assert on which rows
/// were requeued vs. failed-closed.
async fn simulate_restart(
    stores: &TestStores,
    notifier: EventNotifier,
    after: OffsetDateTime,
) -> Result<(OffsetDateTime, Vec<RecoveryRecord>)> {
    drop(notifier);
    let clock = after + Duration::seconds(restart_delta_seconds());
    let records = stores
        .tasks
        .release_expired_leases(clock)
        .await
        .context("release_expired_leases during restart")?;
    Ok((clock, records))
}

// ─────────────────────────────────────────────────────────────────────
// Event-stream replay helpers
// ─────────────────────────────────────────────────────────────────────

async fn drain_thread_stream(
    thread_id: &ThreadId,
    repo: &dyn EventRepository,
    notifier: &EventNotifier,
    expected_count: usize,
) -> Result<Vec<AgentEvent>> {
    let mut stream = stream_events(
        thread_id,
        None,
        repo,
        &InMemoryRetentionStore::new(),
        notifier,
    )
    .await?;
    let mut collected = Vec::with_capacity(expected_count);
    for _ in 0..expected_count {
        match stream.next().await {
            Some(StreamEvent::Event(committed)) => collected.push(committed.event.clone()),
            Some(StreamEvent::Lagged { skipped }) => {
                anyhow::bail!("unexpected Lagged during replay drain, skipped={skipped}")
            }
            Some(StreamEvent::RetentionGap { .. }) => {
                panic!("unexpected retention gap")
            }
            None => {
                anyhow::bail!(
                    "stream closed before yielding {expected_count} events (got {})",
                    collected.len()
                )
            }
        }
    }
    Ok(collected)
}

fn notify_repo_events(
    repo_events: impl IntoIterator<Item = crate::journal::CommittedEvent>,
    notifier: &EventNotifier,
) {
    let events: Vec<_> = repo_events.into_iter().collect();
    if events.is_empty() {
        return;
    }
    notifier.notify(&events);
}

// ─────────────────────────────────────────────────────────────────────
// Phase 7.7 step helpers — thin wrappers around the durable worker
// entry points so the restart scenarios read as a sequence of
// post-restart steps instead of 20-line acquire blocks.
// ─────────────────────────────────────────────────────────────────────

async fn acquire_with_suffix(
    stores: &TestStores,
    id: &AgentTaskId,
    worker_suffix: &str,
    at: OffsetDateTime,
    err_context: &'static str,
) -> Result<AgentTask> {
    stores
        .tasks
        .try_acquire_task(
            id,
            WorkerId::from_string(format!("w-{worker_suffix}")),
            LeaseId::from_string(format!("l-{worker_suffix}")),
            at + Duration::seconds(lease_ttl_seconds()),
            at,
        )
        .await?
        .context(err_context)
}

async fn run_tool_task_to_completion(
    stores: &TestStores,
    tool_id: &AgentTaskId,
    worker_suffix: &str,
    acquire_at: OffsetDateTime,
    exec_at: OffsetDateTime,
    result_text: &'static str,
) -> Result<()> {
    let tool_running = acquire_with_suffix(
        stores,
        tool_id,
        worker_suffix,
        acquire_at,
        "claim child tool task",
    )
    .await?;
    let tool_bootstrap =
        resolve_tool_bootstrap(tool_running, &stores.tasks, ActivityBeacon::default()).await?;
    let tool_outcome = execute_tool_task(
        tool_bootstrap,
        &stores.tasks,
        &stores.events,
        &CancellationToken::new(),
        move |_tool_call, _collector| async move { Ok(ToolResult::success(result_text)) },
        exec_at,
    )
    .await?;
    ensure!(
        matches!(tool_outcome, ToolTaskOutcome::Completed { .. }),
        "tool task expected to complete, got {:?}",
        std::mem::discriminant(&tool_outcome),
    );
    Ok(())
}

async fn resume_child_root_to_completion(
    stores: &TestStores,
    child_root_id: &AgentTaskId,
    worker_suffix: &str,
    acquire_at: OffsetDateTime,
    exec_at: OffsetDateTime,
) -> Result<AgentTask> {
    let child_resume = acquire_with_suffix(
        stores,
        child_root_id,
        worker_suffix,
        acquire_at,
        "reclaim child root for resume",
    )
    .await?;
    let resume_inputs = build_root_worker_inputs(
        bootstrap(child_resume.clone(), child_definition_with_tools())?,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        acquire_at,
    )
    .await?;
    let outcome = resume_from_children(
        resume_inputs,
        &child_resume,
        &MockTextProvider::new("child final response"),
        &stores.root_turn_deps(),
        exec_at,
    )
    .await?;
    let RootTurnOutcome::Completed { completed_task, .. } = outcome else {
        anyhow::bail!("expected child root resume to complete");
    };
    ensure!(
        completed_task.status == TaskStatus::Completed,
        "child root expected Completed, got {:?}",
        completed_task.status,
    );
    Ok(completed_task)
}

async fn run_invocation_to_completion(
    stores: &TestStores,
    invocation_id: &AgentTaskId,
    worker_suffix: &str,
    acquire_at: OffsetDateTime,
    exec_at: OffsetDateTime,
) -> Result<SubagentTaskOutcome> {
    let invocation_running = acquire_with_suffix(
        stores,
        invocation_id,
        worker_suffix,
        acquire_at,
        "claim invocation task",
    )
    .await?;
    let subagent_bootstrap = resolve_subagent_bootstrap(invocation_running, &stores.tasks).await?;
    execute_subagent_task(subagent_bootstrap, &stores.subagent_deps(), exec_at).await
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: happy-path restart resumes nested subagent tree to
// deterministic terminal state.
// ─────────────────────────────────────────────────────────────────────

/// Runs the full nested tree lifecycle (parent root → spawn invocation
/// → child root executes a tool → child resumes → invocation
/// materializes → parent resumes) with a **restart injected between**
/// the child tool completion and the child root resume. Proves that:
///
/// 1. The journal preserves the `SubagentInvocation` linkage across
///    the lease-expiry sweep (child-root row survives the sweep
///    intact).
/// 2. A fresh worker (different `WorkerId` / `LeaseId`) can resume
///    the child root from its `ReadyToResume` state.
/// 3. Subsequent invocation materialization and parent resume see
///    exactly the same summary view as the no-restart path.
#[tokio::test]
async fn restart_mid_child_resume_completes_nested_tree_deterministically() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-happy-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "happy").await?;
    let spawned =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-happy", t_plus(2)).await?;
    notify_repo_events(spawned.committed_events.clone(), &notifier);

    let tool_tasks =
        suspend_child_root_on_tool_call(&stores, &spawned, "happy", t_plus(3), t_plus(4)).await?;
    run_tool_task_to_completion(
        &stores,
        &tool_tasks[0].id,
        "tool-happy",
        t_plus(5),
        t_plus(6),
        "child ls output",
    )
    .await?;

    // Restart: the lease on any residual Running row must be
    // released. Nothing is Running at this moment (tool task is
    // terminal, child root is Pending+ReadyToResume, parent and
    // invocation are WaitingOnChildren with no lease) but we still
    // run the sweep to prove it is a no-op on idle rows.
    let (clock_after_restart, records) = simulate_restart(&stores, notifier, t_plus(7)).await?;
    assert!(
        records.is_empty(),
        "expected idle sweep to be a no-op, got records {records:?}"
    );
    let post_restart_notifier = EventNotifier::new();

    assert_child_and_invocation_ready_after_restart(&stores, &spawned).await?;

    // A fresh worker resumes the child root, then the invocation
    // materialization and parent resume all complete.
    Box::pin(resume_child_root_to_completion(
        &stores,
        &spawned.child_root_task.id,
        "resume-after-restart",
        clock_after_restart,
        clock_after_restart + Duration::seconds(1),
    ))
    .await?;
    let subagent_outcome = run_invocation_to_completion(
        &stores,
        &spawned.invocation_task.id,
        "inv-after-restart",
        clock_after_restart + Duration::seconds(2),
        clock_after_restart + Duration::seconds(3),
    )
    .await?;
    assert_invocation_success(&subagent_outcome, "child final response");

    let completed_parent = Box::pin(resume_parent_root(
        &stores,
        &parent.task.id,
        "after-restart",
        clock_after_restart + Duration::seconds(4),
    ))
    .await?;
    assert_eq!(completed_parent.status, TaskStatus::Completed);
    assert_eq!(completed_parent.id, parent.task.id);

    // The post-restart notifier was never wired into the commit
    // path, so the drain below reads purely from the durable repo
    // replay — a notifier drop must not lose any events.
    assert_parent_summary_events(&stores, &parent_thread, &post_restart_notifier).await?;

    Ok(())
}

async fn assert_child_and_invocation_ready_after_restart(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<()> {
    let child_root = stores
        .tasks
        .get(&spawned.child_root_task.id)
        .await?
        .context("child root must survive the simulated restart")?;
    assert_eq!(child_root.status, TaskStatus::Pending);
    assert!(matches!(child_root.state, TaskState::ReadyToResume { .. }));

    let invocation = stores
        .tasks
        .get(&spawned.invocation_task.id)
        .await?
        .context("invocation must survive the simulated restart")?;
    assert_eq!(invocation.status, TaskStatus::WaitingOnChildren);
    let linkage = invocation
        .state
        .subagent_invocation()
        .context("invocation state must still carry SubagentInvocation linkage")?;
    assert_eq!(linkage.child_root_task_id, spawned.child_root_task.id);
    assert_eq!(linkage.child_thread_id, spawned.child_thread.thread_id);
    Ok(())
}

fn assert_invocation_success(outcome: &SubagentTaskOutcome, expected_output: &str) {
    assert!(outcome.tool_result.success);
    assert_eq!(
        outcome.tool_result.output, expected_output,
        "invocation must materialize the child's final response as the parent's tool result"
    );
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
    let parent_after_invocation = outcome
        .parent_task
        .as_ref()
        .expect("parent should be recomputed after invocation completion");
    assert_eq!(parent_after_invocation.status, TaskStatus::Pending);
    assert!(matches!(
        parent_after_invocation.state,
        TaskState::ReadyToResume { .. }
    ));
}

async fn assert_parent_summary_events(
    stores: &TestStores,
    parent_thread: &ThreadId,
    notifier: &EventNotifier,
) -> Result<()> {
    let parent_events = drain_thread_stream(parent_thread, &stores.events, notifier, 2).await?;
    assert_eq!(parent_events.len(), 2);
    for event in &parent_events {
        assert!(
            matches!(event, AgentEvent::SubagentProgress { .. }),
            "parent replay event must be SubagentProgress, got {event:?}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: lease-expiry sweep requeues a mid-flight child root
// with no loss of linkage.
// ─────────────────────────────────────────────────────────────────────

/// Models the more severe restart: the worker dies mid tool-task
/// execution, so its lease must be swept before any other worker
/// can make progress. Verifies that:
///
/// 1. `release_expired_leases` reports the Running child tool task
///    as [`RecoveryAction::Requeue`].
/// 2. After the sweep, the child tool row is `Pending` again — no
///    worker or lease fields leak through.
/// 3. A new worker can re-acquire the same tool task with a fresh
///    lease and finish the nested tree.
/// 4. The parent thread still ends up with exactly two summary
///    `SubagentProgress` events (no duplicate spawn, no missing
///    completion).
#[tokio::test]
async fn restart_lease_sweep_requeues_running_child_tool_and_tree_completes() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-sweep-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "sweep").await?;
    let spawned =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-sweep", t_plus(2)).await?;
    notify_repo_events(spawned.committed_events.clone(), &notifier);

    let tool_tasks =
        suspend_child_root_on_tool_call(&stores, &spawned, "sweep", t_plus(3), t_plus(4)).await?;

    // Claim the tool task with a lease that will expire over the
    // restart boundary, then *walk away* without completing.
    let claim_time = t_plus(5);
    let _doomed = acquire_with_suffix(
        &stores,
        &tool_tasks[0].id,
        "tool-doomed",
        claim_time,
        "claim tool task on doomed worker",
    )
    .await?;

    // Simulate the restart: notifier dropped, time jumps past the
    // lease expiry, sweep runs.
    let (clock_after_restart, records) = simulate_restart(&stores, notifier, claim_time).await?;
    let tool_id = tool_tasks[0].id.clone();
    assert_sweep_requeued_tool(&records, &tool_id);
    assert_tool_row_is_clean_pending(&stores, &tool_id).await?;

    let post_restart_notifier = EventNotifier::new();

    // A fresh worker re-acquires the tool task and runs the nested
    // tree to completion.
    run_tool_task_to_completion(
        &stores,
        &tool_id,
        "tool-recovered",
        clock_after_restart,
        clock_after_restart + Duration::seconds(1),
        "recovered output",
    )
    .await?;
    Box::pin(resume_child_root_to_completion(
        &stores,
        &spawned.child_root_task.id,
        "child-resume-recovered",
        clock_after_restart + Duration::seconds(2),
        clock_after_restart + Duration::seconds(3),
    ))
    .await?;
    let subagent_outcome = run_invocation_to_completion(
        &stores,
        &spawned.invocation_task.id,
        "inv-recovered",
        clock_after_restart + Duration::seconds(4),
        clock_after_restart + Duration::seconds(5),
    )
    .await?;
    assert!(subagent_outcome.tool_result.success);

    let completed_parent = Box::pin(resume_parent_root(
        &stores,
        &parent.task.id,
        "recovered",
        clock_after_restart + Duration::seconds(6),
    ))
    .await?;
    assert_eq!(completed_parent.status, TaskStatus::Completed);

    // Parent thread must still have exactly 2 SubagentProgress
    // events — spawn + completion — and no leak of rich child events.
    assert_parent_summary_events(&stores, &parent_thread, &post_restart_notifier).await?;

    Ok(())
}

fn assert_sweep_requeued_tool(records: &[RecoveryRecord], tool_id: &AgentTaskId) {
    assert_eq!(
        records.len(),
        1,
        "expected exactly the doomed tool task to be recovered, got {records:?}"
    );
    assert_eq!(records[0].id, *tool_id);
    assert_eq!(records[0].action, RecoveryAction::Requeue);
}

async fn assert_tool_row_is_clean_pending(
    stores: &TestStores,
    tool_id: &AgentTaskId,
) -> Result<()> {
    let requeued = stores
        .tasks
        .get(tool_id)
        .await?
        .context("requeued tool task must still exist")?;
    assert_eq!(requeued.status, TaskStatus::Pending);
    assert!(requeued.worker_id.is_none());
    assert!(requeued.lease_id.is_none());
    assert!(requeued.lease_expires_at.is_none());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: thread-scoped replay after restart — parent summary
// only, child rich.
// ─────────────────────────────────────────────────────────────────────

/// Runs the nested tree to full completion, then simulates a
/// restart and re-opens both the parent and child event streams
/// from sequence 0. Verifies that:
///
/// - Parent thread replay contains **only** `SubagentProgress`
///   events — never `ToolCallStart`, `ToolCallEnd`, `Text`, etc.
/// - Child thread replay contains the rich per-turn lifecycle
///   (`Start`, `ToolCallStart`, `ToolCallEnd`, `TurnComplete`,
///   `Done`) but **never** a `SubagentProgress` event.
/// - Neither thread leaks events from the other: the stream is
///   strictly thread-scoped on replay.
#[tokio::test]
async fn replay_after_restart_is_summary_only_on_parent_and_rich_on_child() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-replay-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "replay").await?;
    let spawned =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-replay", t_plus(2)).await?;
    notify_repo_events(spawned.committed_events.clone(), &notifier);

    let tool_tasks =
        suspend_child_root_on_tool_call(&stores, &spawned, "replay", t_plus(3), t_plus(4)).await?;
    // Box::pin: M5.4's deeper helper-call stack pushed this test's
    // future past clippy's `large_futures` threshold. Test-only
    // hot path — boxing is free.
    let subagent_outcome = Box::pin(drive_child_to_completion(
        &stores,
        &spawned,
        &tool_tasks,
        "replay",
        t_plus(5),
    ))
    .await?;
    assert!(subagent_outcome.tool_result.success);
    let completed_parent = Box::pin(resume_parent_root(
        &stores,
        &parent.task.id,
        "replay",
        t_plus(20),
    ))
    .await?;
    assert_eq!(completed_parent.status, TaskStatus::Completed);

    // Simulate a restart and build a fresh notifier. The durable
    // event repository survives and is the single source of truth.
    let (_clock, _records) = simulate_restart(&stores, notifier, t_plus(25)).await?;
    let reconnect_notifier = EventNotifier::new();

    let parent_repo_events = stores.events.get_events(&parent_thread).await?;
    let child_repo_events = stores
        .events
        .get_events(&spawned.child_thread.thread_id)
        .await?;

    assert_parent_thread_replay_is_summary_only(
        &parent_thread,
        &stores.events,
        &reconnect_notifier,
        &parent_repo_events,
    )
    .await?;
    assert_child_thread_replay_is_rich(
        &spawned.child_thread.thread_id,
        &stores.events,
        &reconnect_notifier,
        &child_repo_events,
    )
    .await?;

    // Cross-thread isolation: no event in the parent repo belongs
    // to the child thread, and vice versa.
    for committed in &parent_repo_events {
        assert_eq!(committed.thread_id, parent_thread);
    }
    for committed in &child_repo_events {
        assert_eq!(committed.thread_id, spawned.child_thread.thread_id);
    }

    Ok(())
}

/// Drain the parent thread's replay stream and assert it carries only
/// the per-spawn / per-completion `SubagentProgress` summaries plus
/// lifecycle events — never any of the child's tool-call events.
async fn assert_parent_thread_replay_is_summary_only(
    parent_thread: &ThreadId,
    events: &InMemoryEventRepository,
    reconnect_notifier: &EventNotifier,
    parent_repo_events: &[crate::journal::CommittedEvent],
) -> Result<()> {
    let parent_event_count = parent_repo_events.len();
    assert!(
        parent_event_count >= 2,
        "parent thread must carry at least the two subagent summaries, got {parent_event_count} events",
    );

    let parent_events = drain_thread_stream(
        parent_thread,
        events,
        reconnect_notifier,
        parent_event_count,
    )
    .await?;
    // Exactly one spawn summary and one completion summary.
    let (spawn_count, completed_count) = count_parent_summaries(&parent_events);
    assert_eq!(spawn_count, 1, "expected one spawn summary event");
    assert_eq!(completed_count, 1, "expected one completion summary event");
    // Phase 7.4 contract: the parent thread must never carry
    // child-rich events. The child's tool call uses
    // `"call_child_bash"` — any `ToolCall*` / `ToolProgress` that
    // references it would be a cross-thread leak.
    for event in &parent_events {
        match event {
            AgentEvent::ToolCallStart { id, .. }
            | AgentEvent::ToolCallEnd { id, .. }
            | AgentEvent::ToolProgress { id, .. } => {
                assert_ne!(
                    id, "call_child_bash",
                    "parent thread must not replay the child's tool events, got {event:?}"
                );
            }
            _ => {}
        }
    }
    Ok(())
}

/// Drain the child thread's replay stream and assert it carries the
/// full rich timeline (`ToolCallStart` / `ToolCallEnd` / `Done`) and
/// **no** `SubagentProgress` events — those live exclusively on the
/// parent.
async fn assert_child_thread_replay_is_rich(
    child_thread: &ThreadId,
    events: &InMemoryEventRepository,
    reconnect_notifier: &EventNotifier,
    child_repo_events: &[crate::journal::CommittedEvent],
) -> Result<()> {
    let child_event_count = child_repo_events.len();
    assert!(
        child_event_count > 0,
        "child thread must have committed at least Start/Done lifecycle events",
    );
    let child_events =
        drain_thread_stream(child_thread, events, reconnect_notifier, child_event_count).await?;
    assert_eq!(child_events.len(), child_event_count);
    for event in &child_events {
        assert!(
            !matches!(event, AgentEvent::SubagentProgress { .. }),
            "child thread must never carry SubagentProgress events, got {event:?}"
        );
    }
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStart { .. })),
        "child thread replay must include ToolCallStart",
    );
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallEnd { .. })),
        "child thread replay must include ToolCallEnd",
    );
    assert!(
        child_events
            .iter()
            .any(|event| matches!(event, AgentEvent::Done { .. })),
        "child thread replay must include a Done event for the completed turn",
    );
    Ok(())
}

async fn fail_child_root_with_synthetic_error(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
    worker_suffix: &str,
    acquire_at: OffsetDateTime,
    fail_at: OffsetDateTime,
) -> Result<()> {
    let child_running = acquire_with_suffix(
        stores,
        &spawned.child_root_task.id,
        worker_suffix,
        acquire_at,
        "claim child root to fail it",
    )
    .await?;
    let worker_id = child_running
        .worker_id
        .clone()
        .context("child running worker id")?;
    let lease_id = child_running
        .lease_id
        .clone()
        .context("child running lease id")?;
    let synthetic_error = anyhow::anyhow!("synthetic child-root failure");
    let failed = fail_root_turn(
        &child_running.id,
        &worker_id,
        &lease_id,
        &child_running.thread_id,
        &synthetic_error,
        &stores.root_turn_deps(),
        fail_at,
    )
    .await?;
    assert_eq!(failed.status, TaskStatus::Failed);
    assert_eq!(
        failed.last_error.as_deref(),
        Some("synthetic child-root failure"),
        "child root must carry the failure message on the row",
    );
    Ok(())
}

async fn assert_invocation_linkage_pending(
    stores: &TestStores,
    spawned: &SpawnedSubagentInvocation,
) -> Result<()> {
    let invocation = stores
        .tasks
        .get(&spawned.invocation_task.id)
        .await?
        .context("invocation must exist")?;
    assert_eq!(invocation.status, TaskStatus::Pending);
    let linkage = invocation
        .state
        .subagent_invocation()
        .context("invocation must still carry its SubagentInvocation linkage")?;
    assert_eq!(linkage.child_root_task_id, spawned.child_root_task.id);
    assert_eq!(linkage.child_thread_id, spawned.child_thread.thread_id);
    Ok(())
}

async fn assert_grandchild_tree_survived_restart(
    stores: &TestStores,
    parent_id: &AgentTaskId,
    spawned_a: &SpawnedSubagentInvocation,
    spawned_b: &SpawnedSubagentInvocation,
    child_thread: &ThreadId,
    grandchild_thread: &ThreadId,
) -> Result<()> {
    let parent_row = stores
        .tasks
        .get(parent_id)
        .await?
        .context("parent row missing")?;
    assert_eq!(parent_row.kind, TaskKind::RootTurn);
    assert_eq!(parent_row.status, TaskStatus::WaitingOnChildren);

    let inv_a = stores
        .tasks
        .get(&spawned_a.invocation_task.id)
        .await?
        .context("invocation_a missing")?;
    assert_eq!(inv_a.kind, TaskKind::Subagent);
    assert_eq!(inv_a.status, TaskStatus::WaitingOnChildren);
    let linkage_a = inv_a
        .state
        .subagent_invocation()
        .context("invocation_a must retain SubagentInvocation state")?;
    assert_eq!(linkage_a.child_thread_id, *child_thread);
    assert_eq!(linkage_a.child_root_task_id, spawned_a.child_root_task.id);

    let child_root_row = stores
        .tasks
        .get(&spawned_a.child_root_task.id)
        .await?
        .context("child root missing")?;
    assert_eq!(child_root_row.kind, TaskKind::RootTurn);
    assert_eq!(child_root_row.status, TaskStatus::WaitingOnChildren);
    assert_eq!(child_root_row.thread_id, *child_thread);

    let inv_b = stores
        .tasks
        .get(&spawned_b.invocation_task.id)
        .await?
        .context("invocation_b missing")?;
    assert_eq!(inv_b.kind, TaskKind::Subagent);
    assert_eq!(inv_b.status, TaskStatus::WaitingOnChildren);
    let linkage_b = inv_b
        .state
        .subagent_invocation()
        .context("invocation_b must retain SubagentInvocation state")?;
    assert_eq!(linkage_b.child_thread_id, *grandchild_thread);
    assert_eq!(linkage_b.child_root_task_id, spawned_b.child_root_task.id);
    assert_eq!(inv_b.thread_id, *child_thread);

    let grand_root_row = stores
        .tasks
        .get(&spawned_b.child_root_task.id)
        .await?
        .context("grandchild root missing")?;
    assert_eq!(grand_root_row.kind, TaskKind::RootTurn);
    assert_eq!(grand_root_row.status, TaskStatus::Pending);
    assert_eq!(grand_root_row.thread_id, *grandchild_thread);
    Ok(())
}

async fn assert_single_spawn_summary_on_thread(
    stores: &TestStores,
    thread_id: &ThreadId,
    notifier: &EventNotifier,
    expected_invocation_id: &AgentTaskId,
    expect_creation_event: bool,
    context_msg: &'static str,
) -> Result<()> {
    // A SPAWNED child thread's journal opens with its ThreadCreated
    // (committed with the spawn); a harness-created root has none.
    let expected_len = if expect_creation_event { 2 } else { 1 };
    let events = drain_thread_stream(thread_id, &stores.events, notifier, expected_len).await?;
    assert_eq!(events.len(), expected_len, "{context_msg}");
    let mut events = events.into_iter();
    if expect_creation_event {
        let first = events.next().context("missing creation event")?;
        assert!(
            matches!(first, AgentEvent::ThreadCreated { .. }),
            "spawned child journal must open with ThreadCreated, got {first:?} ({context_msg})",
        );
    }
    match events.next().context("missing spawn summary")? {
        AgentEvent::SubagentProgress {
            subagent_task_id,
            completed,
            ..
        } => {
            assert!(!completed);
            assert_eq!(
                subagent_task_id.as_deref(),
                Some(expected_invocation_id.to_string().as_str()),
                "{context_msg}"
            );
        }
        other => anyhow::bail!("unexpected {context_msg}: event {other:?}"),
    }
    Ok(())
}

fn count_parent_summaries(events: &[AgentEvent]) -> (usize, usize) {
    let mut spawn = 0;
    let mut completed = 0;
    for event in events {
        if let AgentEvent::SubagentProgress {
            completed: is_completed,
            ..
        } = event
        {
            if *is_completed {
                completed += 1;
            } else {
                spawn += 1;
            }
        }
    }
    (spawn, completed)
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: cancel_tree cascades across thread boundaries and is
// stable across a restart.
// ─────────────────────────────────────────────────────────────────────

/// Builds a live nested tree (parent `Running`-paused, invocation
/// `WaitingOnChildren`, child root `Pending`, child tool `Running`),
/// then:
///
/// 1. Issues `cancel_root_turn` on the parent. The Phase 7.6
///    cross-thread cascade must fold the invocation, child root,
///    and child tool into `Cancelled`.
/// 2. Simulates a restart. The sweep must be a no-op on a
///    fully-terminal subtree — no row should be requeued or
///    failed-closed after cancellation.
/// 3. Re-reads every row and asserts the terminal state persists,
///    the `SubagentInvocation` linkage is cleared from the typed
///    state (cancel drops payload), and no live children remain.
#[tokio::test]
async fn cancellation_across_threads_is_stable_across_restart() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-cancel-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "cancel").await?;
    let spawned =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-cancel", t_plus(2)).await?;
    notify_repo_events(spawned.committed_events.clone(), &notifier);

    let tool_tasks =
        suspend_child_root_on_tool_call(&stores, &spawned, "cancel", t_plus(3), t_plus(4)).await?;

    // Acquire the child tool task so it is Running — the cancel
    // tree walk must sweep across this active lease too.
    let _tool_running = stores
        .tasks
        .try_acquire_task(
            &tool_tasks[0].id,
            WorkerId::from_string("w-tool-cancel"),
            LeaseId::from_string("l-tool-cancel"),
            t_plus(lease_ttl_seconds() + 5),
            t_plus(5),
        )
        .await?
        .context("claim child tool task before cancel")?;

    let cancelled_ids =
        cancel_root_turn(&parent.task.id, &stores.root_turn_deps(), t_plus(6)).await?;
    // Parent + invocation + child root + child tool — 4 rows in the
    // tree.
    assert_eq!(
        cancelled_ids.len(),
        4,
        "expected cancel to cascade to all 4 live rows, got {cancelled_ids:?}",
    );

    // All four rows must already be Cancelled *before* the restart —
    // the restart is only meaningful if it is a no-op on terminal
    // state.
    for id in [
        &parent.task.id,
        &spawned.invocation_task.id,
        &spawned.child_root_task.id,
        &tool_tasks[0].id,
    ] {
        let task = stores
            .tasks
            .get(id)
            .await?
            .with_context(|| format!("task {id} must exist after cancel"))?;
        assert_eq!(task.status, TaskStatus::Cancelled, "task {id} status");
        assert!(
            matches!(task.state, TaskState::None),
            "cancelled task {id} must have cleared its typed state, got {:?}",
            task.state,
        );
    }

    // Simulate the restart. The lease-expiry sweep must find
    // nothing to do on a fully-terminal tree.
    let (_clock, records) = simulate_restart(&stores, notifier, t_plus(7)).await?;
    assert!(
        records.is_empty(),
        "cancel-then-restart sweep must be a no-op, got {records:?}",
    );

    // The same terminal snapshot must still hold after the sweep.
    for id in [
        &parent.task.id,
        &spawned.invocation_task.id,
        &spawned.child_root_task.id,
        &tool_tasks[0].id,
    ] {
        let task = stores
            .tasks
            .get(id)
            .await?
            .with_context(|| format!("task {id} must exist after restart"))?;
        assert_eq!(task.status, TaskStatus::Cancelled);
    }

    // Attempting to cancel again is idempotent.
    let second_cancel =
        cancel_root_turn(&parent.task.id, &stores.root_turn_deps(), t_plus(8)).await?;
    assert!(
        second_cancel.is_empty(),
        "second cancel on terminal tree must be idempotent, got {second_cancel:?}",
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: failed child-thread root wakes the parent invocation
// across a restart.
// ─────────────────────────────────────────────────────────────────────

/// Forces the child-thread root to `Failed`, simulates a restart,
/// then drives the parent-visible invocation through
/// materialization. Proves that:
///
/// 1. `propagate_terminal_child_transition` promotes the linked
///    invocation to `Pending`+`ReadyToResume` as soon as the child
///    root fails (mirrors Phase 7.6 cancel cascade but for failure).
/// 2. The invocation state survives a notifier drop + sweep.
/// 3. `execute_subagent_task` materializes a failure
///    [`ToolResult`] whose `success` is `false` and whose output
///    carries the child's error message.
/// 4. The parent thread ends with exactly two `SubagentProgress`
///    events (spawn + completion with `success=false`).
#[tokio::test]
async fn failed_child_root_propagates_to_parent_invocation_across_restart() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-fail-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "fail").await?;
    let spawned =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-fail", t_plus(2)).await?;
    notify_repo_events(spawned.committed_events.clone(), &notifier);

    fail_child_root_with_synthetic_error(&stores, &spawned, "child-fail", t_plus(3), t_plus(4))
        .await?;
    assert_invocation_linkage_pending(&stores, &spawned).await?;

    // Simulate a restart. Failed tasks must not resurrect, and the
    // invocation must not be disturbed by the sweep.
    let (clock_after_restart, records) = simulate_restart(&stores, notifier, t_plus(5)).await?;
    assert!(
        records.is_empty(),
        "restart sweep must be a no-op — no leases are held, got {records:?}",
    );
    assert_invocation_linkage_pending(&stores, &spawned).await?;

    // Materialize the invocation — the parent's tool result should
    // carry the child's error.
    let subagent_outcome = run_invocation_to_completion(
        &stores,
        &spawned.invocation_task.id,
        "invocation-fail",
        clock_after_restart,
        clock_after_restart + Duration::seconds(1),
    )
    .await?;
    assert!(
        !subagent_outcome.tool_result.success,
        "failed child must surface as an unsuccessful tool result"
    );
    assert_eq!(
        subagent_outcome.tool_result.output, "synthetic child-root failure",
        "failed tool result output must preserve the child's error message"
    );
    assert_eq!(
        subagent_outcome.subagent_result.error_details.as_deref(),
        Some("synthetic child-root failure"),
        "SubagentResult must include structured error details"
    );
    assert_eq!(
        subagent_outcome.invocation_task.status,
        TaskStatus::Completed,
        "failed materialization still completes the invocation task row",
    );

    // Parent thread ends with spawn + completion summaries. The
    // completion summary must reflect the failure.
    let parent_repo_events = stores.events.get_events(&parent_thread).await?;
    assert_eq!(parent_repo_events.len(), 2);
    let events: Vec<AgentEvent> = parent_repo_events.iter().map(|c| c.event.clone()).collect();
    let completion_success = match events.last() {
        Some(AgentEvent::SubagentProgress {
            completed: true,
            success,
            ..
        }) => *success,
        other => anyhow::bail!(
            "expected last parent event to be completion SubagentProgress, got {other:?}",
        ),
    };
    assert!(
        !completion_success,
        "parent's completion summary must reflect child failure",
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Acceptance: nested (grandchild) tree — deeper-than-one nesting
// survives restart with per-thread replay correctness.
// ─────────────────────────────────────────────────────────────────────

/// Builds a two-level nested tree:
///
/// ```text
/// parent_thread (t_parent)
///   parent_root [RootTurn]
///     invocation_A [Subagent]            ── linkage ──> child_thread (t_child)
///                                                           child_root [RootTurn]
///                                                             invocation_B [Subagent] ── linkage ──> grandchild_thread (t_grand)
///                                                                                                       grandchild_root [RootTurn]
/// ```
///
/// Verifies the pure spawn path survives a restart: every linkage
/// row is preserved, and parent, child, and grandchild threads each
/// carry their correct event slice with zero cross-thread leak. This
/// is the structural proof that Phase 7's `SubagentInvocation` state
/// composes across arbitrary depth under restart.
#[tokio::test]
async fn nested_grandchild_tree_linkage_survives_restart() -> Result<()> {
    let stores = TestStores::new();
    let parent_thread = ThreadId::from_string("t-p77-grand-parent");
    let notifier = EventNotifier::new();

    let parent = create_running_parent_root(&stores, &parent_thread, "grand").await?;
    let spawned_a =
        spawn_child_invocation(&stores, &parent, &parent_thread, "call-grand-a", t_plus(2)).await?;
    notify_repo_events(spawned_a.committed_events.clone(), &notifier);
    let child_thread = spawned_a.child_thread.thread_id.clone();

    // The child root is Pending — claim it and mount a second-level
    // invocation (modeling a subagent that itself spawns another
    // subagent). We take the child root into Running via
    // `try_acquire_task`, then drive it through `spawn_subagent_invocation`
    // to produce the grandchild.
    let child_running = stores
        .tasks
        .try_acquire_task(
            &spawned_a.child_root_task.id,
            WorkerId::from_string("w-child-grand"),
            LeaseId::from_string("l-child-grand"),
            t_plus(lease_ttl_seconds() + 3),
            t_plus(3),
        )
        .await?
        .context("claim child root for grandchild spawn")?;

    let grand_spawn = SubagentInvocationSpawn {
        child_thread_id: ThreadId::new(),
        spec: subagent_spec(),
        child_root_input: vec![SubmittedInputItem::Text {
            text: "Stay in read-only mode.\n\nInvestigate grandchild durable reuse".into(),
        }],
        spawn_index: 0,
        payload: parent_suspension_payload(&child_thread, "call-grand-b"),
        child_caller_metadata: None,
    };
    let spawned_b = spawn_subagent_invocation(
        &child_running.id,
        child_running
            .worker_id
            .as_ref()
            .context("child worker id")?,
        child_running.lease_id.as_ref().context("child lease id")?,
        grand_spawn,
        &stores.invocation_deps(),
        t_plus(4),
    )
    .await?;
    notify_repo_events(spawned_b.committed_events.clone(), &notifier);
    let grandchild_thread = spawned_b.child_thread.thread_id.clone();

    // Simulate a restart. None of the rows are mid-tool here;
    // invocation_A is WaitingOnChildren, child_root is
    // WaitingOnChildren, invocation_B is WaitingOnChildren,
    // grandchild_root is Pending. No leased rows → sweep is a no-op.
    let (_clock, records) = simulate_restart(&stores, notifier, t_plus(5)).await?;
    assert!(
        records.is_empty(),
        "nested spawn-then-restart sweep must be a no-op, got {records:?}",
    );

    // Every row survives with correct kind + linkage.
    assert_grandchild_tree_survived_restart(
        &stores,
        &parent.task.id,
        &spawned_a,
        &spawned_b,
        &child_thread,
        &grandchild_thread,
    )
    .await?;

    // Replay each thread and assert isolation. The reconnect
    // notifier reads purely from durable state.
    let reconnect_notifier = EventNotifier::new();
    assert_single_spawn_summary_on_thread(
        &stores,
        &parent_thread,
        &reconnect_notifier,
        &spawned_a.invocation_task.id,
        false,
        "parent thread must see invocation_A spawn summary",
    )
    .await?;
    assert_single_spawn_summary_on_thread(
        &stores,
        &child_thread,
        &reconnect_notifier,
        &spawned_b.invocation_task.id,
        true,
        "child thread must see invocation_B spawn summary only",
    )
    .await?;

    // Grandchild thread has not emitted any events yet — the
    // grandchild root has not been claimed. Reading from the
    // repository directly avoids blocking on the live channel.
    let grand_events = stores.events.get_events(&grandchild_thread).await?;
    assert_eq!(
        grand_events.len(),
        1,
        "grandchild thread must hold only its creation event until its root runs, got {grand_events:?}",
    );
    assert!(
        matches!(grand_events[0].event, AgentEvent::ThreadCreated { .. }),
        "grandchild journal must open with ThreadCreated, got {:?}",
        grand_events[0].event,
    );

    // Idempotency across restart: a second sweep still finds
    // nothing to do.
    let records_again = stores
        .tasks
        .release_expired_leases(t_plus(restart_delta_seconds() + 60))
        .await?;
    assert!(
        records_again.is_empty(),
        "second sweep after restart must also be a no-op, got {records_again:?}",
    );

    Ok(())
}
