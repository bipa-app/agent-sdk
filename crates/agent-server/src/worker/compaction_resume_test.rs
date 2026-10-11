//! Regression: compacting a tool-using turn between its tool rounds must
//! never re-append that turn's in-flight messages to the durable transcript.
//!
//! A long turn suspends at every tool boundary and resumes from the task
//! row's `suspended_messages`. Compaction folds the in-flight draft into the
//! raw transcript, so a later suspension that still carries the folded
//! prefix makes the next resume re-stage it and the next compaction fold it
//! again: one extra copy of the turn per compaction, until the provider
//! rejects the prompt. These tests drive several resume rounds through the
//! real worker entry points, with proactive and overflow-recovery
//! compaction, for every engine.

use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_sdk::context::{CompactionConfig, CompactionEngine};
use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Message, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use time::{Duration, OffsetDateTime};

use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message::MessageProjection;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::outbox::InMemoryOutboxStore;
use crate::journal::store::{
    AgentTaskStore, CancellationMarkerSink, InMemoryAgentTaskStore,
    InMemoryAtomicSpawnEventCommitter, RequeueOutcome,
};
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use crate::journal::task_state::TaskState;
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use crate::worker::root_turn::{
    RootTurnDeps, RootTurnOutcome, execute_root_turn, resume_from_children,
};

const RESUME_ROUNDS: usize = 3;
const PROMPT_TOO_LONG: &str = "prompt is too long: 1010596 tokens > 1000000 maximum";

#[derive(Clone, Copy, Debug)]
enum Trigger {
    Threshold,
    Overflow,
}

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn at(seconds: i64) -> OffsetDateTime {
    t0() + Duration::seconds(seconds)
}

fn thread_id() -> ThreadId {
    ThreadId::from_string("t-compaction-resume")
}

fn bulk(tag: &str) -> String {
    format!("{tag} {}", "very/deep/path ".repeat(400))
}

fn usage() -> Usage {
    Usage {
        served_speed: None,
        input_tokens: 100,
        output_tokens: 50,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        request_input_tokens: Some(100),
    }
}

fn tool_call(id: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: format!("msg_{id}"),
        content: vec![ContentBlock::ToolUse {
            id: id.to_owned(),
            name: "bash".to_owned(),
            input: serde_json::json!({ "command": bulk(id) }),
            thought_signature: None,
        }],
        model: "mock-model".into(),
        stop_reason: Some(StopReason::ToolUse),
        usage: usage(),
    })
}

fn text(reply: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: "msg_final".into(),
        content: vec![ContentBlock::Text {
            text: reply.to_owned(),
        }],
        model: "mock-model".into(),
        stop_reason: Some(StopReason::EndTurn),
        usage: usage(),
    })
}

/// Main-model double: pops one scripted outcome per call and records every
/// request it was sent.
struct ScriptedMainProvider {
    responses: Mutex<VecDeque<ChatOutcome>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedMainProvider {
    fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, outcome: ChatOutcome) -> Result<()> {
        self.responses
            .lock()
            .map_err(|_| anyhow::anyhow!("response queue poisoned"))?
            .push_back(outcome);
        Ok(())
    }

    fn requests(&self) -> Result<Vec<ChatRequest>> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .map_err(|_| anyhow::anyhow!("request log poisoned"))
    }
}

#[async_trait]
impl LlmProvider for ScriptedMainProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?
            .push(request);
        self.responses
            .lock()
            .map_err(|_| anyhow::anyhow!("response queue poisoned"))?
            .pop_front()
            .context("main provider script exhausted")
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }

    fn supports_historical_image_blocks(&self) -> bool {
        true
    }
}

/// Summarizer double for the LLM-backed engines.
struct SummaryProvider {
    calls: AtomicUsize,
}

#[async_trait]
impl LlmProvider for SummaryProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(text(&format!("[summary {call}]")))
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

struct Harness {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
    event_notifier: Arc<EventNotifier>,
    config: CompactionConfig,
    summarizer: Arc<dyn LlmProvider>,
    artifact_store: Arc<agent_sdk::ArtifactStore>,
    _artifact_dir: tempfile::TempDir,
}

impl Harness {
    fn new(engine: CompactionEngine, trigger: Trigger) -> Result<Self> {
        let threads = InMemoryThreadStore::new();
        let events = InMemoryEventRepository::new();
        let outbox = InMemoryOutboxStore::new();
        let tasks = InMemoryAgentTaskStore::new()
            .with_cancellation_markers(CancellationMarkerSink {
                event_repo: Arc::new(events.clone()),
                outbox_store: Arc::new(outbox.clone()),
                thread_store: Arc::new(threads.clone()),
            })
            .with_spawn_event_committer(Arc::new(InMemoryAtomicSpawnEventCommitter::new(
                events.clone(),
                outbox,
            )));
        let threshold = match trigger {
            Trigger::Threshold => 1,
            Trigger::Overflow => usize::MAX,
        };
        let artifact_dir = tempfile::tempdir().context("artifact tempdir")?;
        let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(
            artifact_dir.path().join("t-compaction-resume"),
        ));
        Ok(Self {
            tasks,
            threads,
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
            events,
            event_notifier: Arc::new(EventNotifier::new()),
            config: CompactionConfig::default()
                .with_engine(engine)
                .with_threshold_tokens(threshold)
                .with_min_messages(2)
                .with_retain_recent(0),
            summarizer: Arc::new(SummaryProvider {
                calls: AtomicUsize::new(0),
            }),
            artifact_store,
            _artifact_dir: artifact_dir,
        })
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
            compaction_config: Some(&self.config),
            compaction_provider: Some(&self.summarizer),
            compaction_artifact_store: Some(&self.artifact_store),
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
        }
    }

    async fn projection(&self) -> Result<MessageProjection> {
        self.messages
            .get(&thread_id())
            .await?
            .context("message projection")
    }

    async fn acquire(
        &self,
        task_id: &crate::journal::task::AgentTaskId,
        now: i64,
    ) -> Result<AgentTask> {
        self.tasks
            .try_acquire_task(
                task_id,
                WorkerId::from_string("worker_resume"),
                LeaseId::from_string("lease_resume"),
                at(now + 300),
                at(now),
            )
            .await?
            .context("task must be acquirable")
    }

    async fn complete_children(&self, children: &[AgentTask], now: i64) -> Result<()> {
        for child in children {
            self.tasks
                .try_acquire_task(
                    &child.id,
                    WorkerId::from_string("child_worker"),
                    LeaseId::from_string("child_lease"),
                    at(now + 300),
                    at(now),
                )
                .await?
                .context("child must be acquirable")?;
            self.tasks
                .complete_task_with_result(
                    &child.id,
                    &WorkerId::from_string("child_worker"),
                    &LeaseId::from_string("child_lease"),
                    serde_json::to_value(ToolResult::success(bulk("output")))?,
                    at(now + 1),
                )
                .await
                .context("complete child")?;
        }
        Ok(())
    }

    async fn inputs(
        &self,
        task: AgentTask,
        now: i64,
    ) -> Result<crate::journal::execution_context::RootWorkerInputs> {
        let thread_id = task.thread_id.clone();
        let task_id = task.id.clone();
        build_root_worker_inputs(
            WorkerBootstrapContext {
                task,
                definition: definition(),
                thread_id,
                task_id,
                worker_id: WorkerId::from_string("worker_resume"),
                lease_id: LeaseId::from_string("lease_resume"),
            },
            &self.threads,
            &self.checkpoints,
            &self.messages,
            at(now),
        )
        .await
    }
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a helpful test assistant.".into(),
        max_tokens: 1024,
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run a shell command".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } }
            }),
            display_name: "Bash".into(),
            tier: ToolTier::Observe,
        }],
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn tool_use_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| match &message.content {
            Content::Blocks(blocks) => blocks.as_slice(),
            Content::Text(_) => &[],
        })
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn ensure_unique_tool_use_ids(messages: &[Message], label: &str) -> Result<()> {
    let ids = tool_use_ids(messages);
    let mut seen = BTreeSet::new();
    for id in &ids {
        ensure!(
            seen.insert(id.as_str()),
            "{label}: tool_use {id} appears more than once in {ids:?}"
        );
    }
    Ok(())
}

fn ensure_disjoint_from_transcript(
    in_flight: &[Message],
    transcript: &[Message],
    label: &str,
) -> Result<()> {
    let committed: BTreeSet<String> = tool_use_ids(transcript).into_iter().collect();
    for id in tool_use_ids(in_flight) {
        ensure!(
            !committed.contains(&id),
            "{label}: tool_use {id} is already in the durable transcript, so the next \
             compaction would fold it a second time",
        );
    }
    Ok(())
}

fn suspended_parts(outcome: RootTurnOutcome) -> Result<(AgentTask, Vec<AgentTask>)> {
    match outcome {
        RootTurnOutcome::Suspended {
            parent_task,
            child_tasks,
            ..
        } => Ok((parent_task, child_tasks)),
        RootTurnOutcome::Completed { .. } => bail!("expected the turn to suspend on a tool call"),
    }
}

fn parked_messages(task: &AgentTask) -> Result<&[Message]> {
    match &task.state {
        TaskState::WaitingOnChildren {
            suspended_messages, ..
        }
        | TaskState::ReadyToResume {
            suspended_messages, ..
        } => Ok(suspended_messages),
        other => bail!("expected a parked tool suspension, got {other:?}"),
    }
}

async fn compaction_counts(harness: &Harness) -> Result<Vec<(usize, usize)>> {
    Ok(harness
        .events
        .get_events(&thread_id())
        .await?
        .into_iter()
        .filter_map(|committed| match committed.event {
            AgentEvent::ContextCompacted {
                original_count,
                new_count,
                ..
            } => Some((original_count, new_count)),
            _ => None,
        })
        .collect())
}

async fn start_tool_turn(
    harness: &Harness,
    main: &ScriptedMainProvider,
) -> Result<(AgentTaskId, Vec<AgentTask>)> {
    main.push(tool_call("call_1"))?;
    let task = AgentTask::new_root_turn(thread_id(), t0(), 3);
    let task_id = task.id.clone();
    harness.tasks.submit_root_turn(task).await?;
    let task = harness.acquire(&task_id, 0).await?;
    let inputs = harness.inputs(task, 0).await?;
    let outcome = execute_root_turn(inputs, "Run tools", main, &harness.deps(), at(1)).await?;
    let (_, children) = suspended_parts(outcome)?;
    Ok((task_id, children))
}

fn ensure_requests_have_unique_tool_use_ids(
    main: &ScriptedMainProvider,
    label: &str,
) -> Result<()> {
    for (index, request) in main.requests()?.iter().enumerate() {
        ensure_unique_tool_use_ids(
            &request.messages,
            &format!("{label}: provider request {index}"),
        )?;
    }
    Ok(())
}

struct ResumeRound<'a> {
    round: usize,
    parked_len: usize,
    before: &'a MessageProjection,
    after: &'a MessageProjection,
}

fn ensure_round_folded_only_unfolded_messages(check: &ResumeRound<'_>) -> Result<()> {
    let ResumeRound {
        round,
        parked_len,
        before,
        after,
    } = *check;
    ensure!(
        after.compactions.len() == before.compactions.len() + 1,
        "round {round}: the resume must compact exactly once (before {}, after {})",
        before.compactions.len(),
        after.compactions.len(),
    );
    let entry = after
        .compactions
        .last()
        .context("compaction entry for this round")?;
    let expected_source = before.context_history().len() + parked_len + 1;
    ensure!(
        entry.source_message_count == expected_source,
        "round {round}: compaction source must be the effective view plus the parked \
         messages and their tool results (expected {expected_source}, got {})",
        entry.source_message_count,
    );
    let final_reply = usize::from(round == RESUME_ROUNDS);
    ensure!(
        after.messages.len() == before.messages.len() + parked_len + 1 + final_reply,
        "round {round}: the compaction boundary may append only the parked messages and \
         their tool results (plus the final reply on commit): transcript grew from {} to {}",
        before.messages.len(),
        after.messages.len(),
    );
    ensure!(
        entry.draft_fold_start == before.messages.len() && entry.draft_fold_len == parked_len + 1,
        "round {round}: the entry must record the raw range it folded from the draft",
    );
    ensure_unique_tool_use_ids(&after.messages, &format!("round {round}: raw transcript"))?;
    ensure_unique_tool_use_ids(
        &after.context_history(),
        &format!("round {round}: effective view"),
    )
}

async fn ensure_turn_committed_each_message_once(harness: &Harness) -> Result<()> {
    let projection = harness.projection().await?;
    let expected_ids: Vec<String> = (1..=RESUME_ROUNDS)
        .map(|round| format!("call_{round}"))
        .collect();
    ensure!(
        tool_use_ids(&projection.messages) == expected_ids,
        "the committed transcript must hold every tool call exactly once, in order: {:?}",
        tool_use_ids(&projection.messages),
    );
    let expected_raw_len = 2 + RESUME_ROUNDS * 2;
    ensure!(
        projection.messages.len() == expected_raw_len,
        "prompt + one (tool_use, tool_result) pair per round + the final reply is \
         {expected_raw_len} messages; the transcript has {}",
        projection.messages.len(),
    );
    ensure!(projection.draft_messages.is_empty());

    let counts = compaction_counts(harness).await?;
    ensure!(counts.len() == RESUME_ROUNDS);
    for (index, (original, new)) in counts.iter().enumerate() {
        ensure!(
            new <= original,
            "compaction {index} grew the effective view: {original} -> {new}"
        );
    }
    let (first_original, _) = counts.first().context("first compaction")?;
    let (last_original, _) = counts.last().context("last compaction")?;
    ensure!(
        last_original <= first_original,
        "compaction sources must not grow across rounds of one turn: first {first_original}, \
         last {last_original}",
    );
    Ok(())
}

/// Runs one turn that calls a tool, then resumes it `RESUME_ROUNDS` times
/// with a compaction in every resume. Each resume but the last asks for one
/// more tool; the last answers in text and commits the turn.
async fn run_compacted_tool_turn(engine: CompactionEngine, trigger: Trigger) -> Result<()> {
    let harness = Harness::new(engine, trigger)?;
    let main = ScriptedMainProvider::new();
    let (task_id, mut children) = start_tool_turn(&harness, &main).await?;

    for round in 1..=RESUME_ROUNDS {
        let now = i64::try_from(round)? * 1_000;
        harness.complete_children(&children, now).await?;
        let parent = harness.acquire(&task_id, now + 10).await?;
        ensure!(
            matches!(parent.state, TaskState::ReadyToResume { .. }),
            "round {round}: parent must be ready to resume"
        );

        if matches!(trigger, Trigger::Overflow) {
            main.push(ChatOutcome::InvalidRequest(PROMPT_TOO_LONG.into()))?;
        }
        if round < RESUME_ROUNDS {
            main.push(tool_call(&format!("call_{}", round + 1)))?;
        } else {
            main.push(text("all done"))?;
        }

        let before = harness.projection().await?;
        ensure_disjoint_from_transcript(
            parked_messages(&parent)?,
            &before.messages,
            &format!("round {round}: parked suspended_messages"),
        )?;
        let inputs = harness.inputs(parent.clone(), now + 10).await?;
        let outcome =
            resume_from_children(inputs, &parent, &main, &harness.deps(), at(now + 20)).await?;
        let after = harness.projection().await?;
        ensure_round_folded_only_unfolded_messages(&ResumeRound {
            round,
            parked_len: parked_messages(&parent)?.len(),
            before: &before,
            after: &after,
        })?;
        ensure_requests_have_unique_tool_use_ids(&main, &format!("round {round}"))?;

        match outcome {
            RootTurnOutcome::Suspended { child_tasks, .. } => {
                ensure!(
                    round < RESUME_ROUNDS,
                    "round {round}: the last resume must commit"
                );
                ensure_disjoint_from_transcript(
                    &after.draft_messages,
                    &after.messages,
                    &format!("round {round}: projection draft"),
                )?;
                children = child_tasks;
            }
            RootTurnOutcome::Completed { completed_task, .. } => {
                ensure!(
                    round == RESUME_ROUNDS,
                    "round {round}: turn completed early"
                );
                ensure!(completed_task.status == TaskStatus::Completed);
            }
        }
    }
    ensure_turn_committed_each_message_once(&harness).await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::Legacy, Trigger::Threshold).await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::PruneFirst, Trigger::Threshold).await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::Snapcompact, Trigger::Threshold).await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::Legacy, Trigger::Overflow).await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::PruneFirst, Trigger::Overflow).await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(CompactionEngine::Snapcompact, Trigger::Overflow).await
}

/// A resume whose compaction already folded the turn into the transcript can
/// still die afterwards (provider weather, lease loss, a lost turn slot); the
/// host then re-runs it from the same parked `suspended_messages`. The rerun
/// must recognize the folded messages instead of staging them a second time.
#[tokio::test]
async fn resume_rerun_after_a_compacting_attempt_died_does_not_restage_the_folded_turn()
-> Result<()> {
    let harness = Harness::new(CompactionEngine::Legacy, Trigger::Threshold)?;
    let main = ScriptedMainProvider::new();
    let (task_id, children) = start_tool_turn(&harness, &main).await?;
    harness.complete_children(&children, 1_000).await?;

    main.push(ChatOutcome::InvalidRequest(
        "upstream rejected the request".into(),
    ))?;
    let parent = harness.acquire(&task_id, 1_010).await?;
    let inputs = harness.inputs(parent.clone(), 1_010).await?;
    let died = resume_from_children(inputs, &parent, &main, &harness.deps(), at(1_020)).await;
    ensure!(
        died.is_err(),
        "the first resume attempt must die after compacting"
    );
    let folded = harness.projection().await?;
    ensure!(
        folded.compactions.len() == 1 && folded.draft_messages.is_empty(),
        "precondition: the dead attempt folded the in-flight turn into the transcript",
    );
    ensure!(tool_use_ids(&folded.messages) == ["call_1"]);

    let requeued = harness
        .tasks
        .requeue_owned_task(
            &task_id,
            &WorkerId::from_string("worker_resume"),
            &LeaseId::from_string("lease_resume"),
            None,
            at(1_030),
        )
        .await?;
    ensure!(matches!(requeued, RequeueOutcome::Requeued(_)));

    main.push(text("all done"))?;
    let parent = harness.acquire(&task_id, 2_000).await?;
    let inputs = harness.inputs(parent.clone(), 2_000).await?;
    let outcome = resume_from_children(inputs, &parent, &main, &harness.deps(), at(2_010)).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "the rerun must commit the turn"
    );

    let projection = harness.projection().await?;
    ensure_unique_tool_use_ids(&projection.messages, "raw transcript after the rerun")?;
    ensure_unique_tool_use_ids(
        &projection.context_history(),
        "effective view after the rerun",
    )?;
    ensure_requests_have_unique_tool_use_ids(&main, "rerun")?;
    ensure!(
        projection.messages.len() == 4,
        "prompt, tool call, tool result and the final reply, each once; the transcript has {}",
        projection.messages.len(),
    );
    Ok(())
}
