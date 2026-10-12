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
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Message, StopReason, Tool,
    Usage, provider_tool_sequence_error_index,
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
    RootTurnDeps, RootTurnOutcome, cancel_root_turn, execute_root_turn, resume_for_steering,
    resume_from_children, revert_steering_wake,
};

const RESUME_ROUNDS: usize = 3;
const PROMPT_TOO_LONG: &str = "prompt is too long: 1010596 tokens > 1000000 maximum";

#[derive(Clone, Copy, Debug)]
enum Trigger {
    Threshold,
    Overflow,
}

/// How much of the recent raw transcript each compaction keeps verbatim.
/// With `Default`, the folded in-flight messages stay in the effective view,
/// so a duplicate shows there as well as in the raw transcript.
#[derive(Clone, Copy, Debug)]
enum Retention {
    Nothing,
    Default,
}

const SEEDED_TURNS: usize = 6;

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
    tool_calls(&[id])
}

fn tool_calls(ids: &[&str]) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: format!("msg_{}", ids.join("_")),
        content: ids
            .iter()
            .map(|id| ContentBlock::ToolUse {
                id: (*id).to_owned(),
                name: "bash".to_owned(),
                input: serde_json::json!({ "command": bulk(id) }),
                thought_signature: None,
            })
            .collect(),
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
    fn new(engine: CompactionEngine, trigger: Trigger, retention: Retention) -> Result<Self> {
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
        let config = CompactionConfig::default()
            .with_engine(engine)
            .with_threshold_tokens(threshold)
            .with_min_messages(2);
        let config = match retention {
            Retention::Nothing => config.with_retain_recent(0),
            Retention::Default => config,
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
            config,
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

    /// Commit `turns` prior exchanges straight into the projection, so a
    /// compaction at default retention has older history to summarize.
    async fn seed_history(&self, turns: usize) -> Result<usize> {
        let messages: Vec<Message> = (0..turns)
            .flat_map(|turn| {
                [
                    Message::user(bulk(&format!("earlier question {turn}"))),
                    Message::assistant(bulk(&format!("earlier answer {turn}"))),
                ]
            })
            .collect();
        let seeded = messages.len();
        self.messages
            .commit_messages(&thread_id(), messages, t0())
            .await?;
        Ok(seeded)
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
        request_context_fn: None,
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

fn ensure_provider_valid(messages: &[Message], label: &str) -> Result<()> {
    match provider_tool_sequence_error_index(messages) {
        None => Ok(()),
        Some(index) => bail!(
            "{label}: tool_use/tool_result sequence breaks at message {index}: {:?}",
            tool_use_ids(messages),
        ),
    }
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
    start_turn_calling(harness, main, &["call_1"]).await
}

async fn start_turn_calling(
    harness: &Harness,
    main: &ScriptedMainProvider,
    ids: &[&str],
) -> Result<(AgentTaskId, Vec<AgentTask>)> {
    main.push(tool_calls(ids))?;
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

/// The transcript and compaction lineage a tool turn started from: seeded
/// history, plus whatever the fresh call compacted before the first resume.
struct TurnBaseline {
    seeded: usize,
    compactions: usize,
}

async fn ensure_turn_committed_each_message_once(
    harness: &Harness,
    baseline: &TurnBaseline,
) -> Result<()> {
    let projection = harness.projection().await?;
    let expected_ids: Vec<String> = (1..=RESUME_ROUNDS)
        .map(|round| format!("call_{round}"))
        .collect();
    ensure!(
        tool_use_ids(&projection.messages) == expected_ids,
        "the committed transcript must hold every tool call exactly once, in order: {:?}",
        tool_use_ids(&projection.messages),
    );
    let expected_raw_len = baseline.seeded + 2 + RESUME_ROUNDS * 2;
    ensure!(
        projection.messages.len() == expected_raw_len,
        "seeded history + prompt + one (tool_use, tool_result) pair per round + the final \
         reply is {expected_raw_len} messages; the transcript has {}",
        projection.messages.len(),
    );
    ensure!(projection.draft_messages.is_empty());
    ensure_provider_valid(&projection.messages, "raw transcript")?;
    ensure_provider_valid(&projection.context_history(), "effective view")?;

    let counts = compaction_counts(harness).await?;
    let counts = counts
        .get(baseline.compactions..)
        .context("compactions since the fresh call")?;
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
async fn run_compacted_tool_turn(
    engine: CompactionEngine,
    trigger: Trigger,
    retention: Retention,
) -> Result<()> {
    let harness = Harness::new(engine, trigger, retention)?;
    let seeded = match retention {
        Retention::Nothing => 0,
        Retention::Default => harness.seed_history(SEEDED_TURNS).await?,
    };
    let main = ScriptedMainProvider::new();
    let (task_id, mut children) = start_tool_turn(&harness, &main).await?;
    let baseline = TurnBaseline {
        seeded,
        compactions: compaction_counts(&harness).await?.len(),
    };

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
        if matches!(retention, Retention::Default) {
            ensure!(
                tool_use_ids(&after.context_history()).contains(&format!("call_{round}")),
                "round {round}: default retention must keep this round's tool call in the \
                 effective view, where a re-staged copy would show",
            );
        }

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
    ensure_turn_committed_each_message_once(&harness, &baseline).await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::PruneFirst,
        Trigger::Threshold,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn threshold_compaction_between_tool_rounds_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Snapcompact,
        Trigger::Threshold,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Legacy,
        Trigger::Overflow,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::PruneFirst,
        Trigger::Overflow,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_between_tool_rounds_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Snapcompact,
        Trigger::Overflow,
        Retention::Nothing,
    )
    .await
}

#[tokio::test]
async fn threshold_compaction_at_default_retention_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Default,
    )
    .await
}

#[tokio::test]
async fn threshold_compaction_at_default_retention_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::PruneFirst,
        Trigger::Threshold,
        Retention::Default,
    )
    .await
}

#[tokio::test]
async fn threshold_compaction_at_default_retention_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Snapcompact,
        Trigger::Threshold,
        Retention::Default,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_at_default_retention_never_refolds_legacy() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Legacy,
        Trigger::Overflow,
        Retention::Default,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_at_default_retention_never_refolds_prune_first() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::PruneFirst,
        Trigger::Overflow,
        Retention::Default,
    )
    .await
}

#[tokio::test]
async fn overflow_compaction_at_default_retention_never_refolds_snapcompact() -> Result<()> {
    run_compacted_tool_turn(
        CompactionEngine::Snapcompact,
        Trigger::Overflow,
        Retention::Default,
    )
    .await
}

/// Starts a tool turn, completes its child and runs a resume that compacts,
/// folding the in-flight turn into the transcript, and then dies on its
/// provider call. The host then requeues the row with its `ReadyToResume`
/// state intact, as it does on provider weather, lease loss or a lost turn
/// slot: the parked `suspended_messages` predate the fold.
async fn fold_then_die(harness: &Harness, main: &ScriptedMainProvider) -> Result<AgentTaskId> {
    let (task_id, children) = start_tool_turn(harness, main).await?;
    harness.complete_children(&children, 1_000).await?;

    main.push(ChatOutcome::InvalidRequest(
        "upstream rejected the request".into(),
    ))?;
    let parent = harness.acquire(&task_id, 1_010).await?;
    let before = harness.projection().await?;
    let inputs = harness.inputs(parent.clone(), 1_010).await?;
    let died = resume_from_children(inputs, &parent, main, &harness.deps(), at(1_020)).await;
    ensure!(
        died.is_err(),
        "the resume attempt must die after compacting"
    );
    let folded = harness.projection().await?;
    ensure!(
        folded.compactions.len() == before.compactions.len() + 1
            && folded.draft_messages.is_empty(),
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
    Ok(task_id)
}

/// The host re-runs a resume that died after its compaction from the same
/// parked `suspended_messages`; the rerun must recognize the folded messages
/// instead of staging them a second time.
#[tokio::test]
async fn resume_rerun_after_a_compacting_attempt_died_does_not_restage_the_folded_turn()
-> Result<()> {
    let harness = Harness::new(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Nothing,
    )?;
    let main = ScriptedMainProvider::new();
    let task_id = fold_then_die(&harness, &main).await?;

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

/// Cancelling the requeued row commits its parked `suspended_messages`
/// (`[prompt, call_1]`), all of which the dead attempt's fold already holds
/// together with `call_1`'s real result. Committing them again would leave
/// `call_1` in the transcript twice, once with a synthetic cancelled result.
async fn cancel_after_a_folded_resume_died(retention: Retention) -> Result<()> {
    let harness = Harness::new(CompactionEngine::Legacy, Trigger::Threshold, retention)?;
    if matches!(retention, Retention::Default) {
        harness.seed_history(SEEDED_TURNS).await?;
    }
    let main = ScriptedMainProvider::new();
    let task_id = fold_then_die(&harness, &main).await?;
    let parked = harness.tasks.get(&task_id).await?.context("parked row")?;
    ensure!(
        parked.status == TaskStatus::Pending
            && matches!(parked.state, TaskState::ReadyToResume { .. }),
        "precondition: the requeued row is parked ready to resume",
    );
    let folded = harness.projection().await?;

    let cancelled = cancel_root_turn(&task_id, &harness.deps(), at(1_040)).await?;
    ensure!(cancelled.contains(&task_id));

    let projection = harness.projection().await?;
    ensure!(
        projection.messages.len() == folded.messages.len(),
        "the fold already holds the whole parked turn, so the cancel commits nothing: \
         the transcript grew from {} to {} messages",
        folded.messages.len(),
        projection.messages.len(),
    );
    ensure!(
        tool_use_ids(&projection.messages) == ["call_1"],
        "raw transcript after the cancel: {:?}",
        tool_use_ids(&projection.messages),
    );
    ensure_provider_valid(&projection.messages, "raw transcript after the cancel")?;
    let effective = projection.context_history();
    ensure_unique_tool_use_ids(&effective, "effective view after the cancel")?;
    ensure_provider_valid(&effective, "effective view after the cancel")?;
    if matches!(retention, Retention::Default) {
        ensure!(
            tool_use_ids(&effective) == ["call_1"],
            "default retention keeps the folded call in the effective view: {:?}",
            tool_use_ids(&effective),
        );
    }
    Ok(())
}

#[tokio::test]
async fn parked_cancel_after_a_folded_resume_died_commits_nothing_twice() -> Result<()> {
    cancel_after_a_folded_resume_died(Retention::Nothing).await
}

#[tokio::test]
async fn parked_cancel_after_a_folded_resume_died_at_default_retention() -> Result<()> {
    cancel_after_a_folded_resume_died(Retention::Default).await
}

async fn wake_for_steering(harness: &Harness, task_id: &AgentTaskId, now: i64) -> Result<()> {
    harness
        .tasks
        .enqueue_steering_resume(
            task_id,
            vec![ContentBlock::Text {
                text: "how is it going?".into(),
            }],
            at(now),
        )
        .await?
        .context("the parked parent must wake for steering")?;
    Ok(())
}

async fn ensure_turn_holds_each_tool_call_once(
    harness: &Harness,
    main: &ScriptedMainProvider,
    expected: &[String],
    label: &str,
) -> Result<()> {
    let projection = harness.projection().await?;
    ensure!(
        tool_use_ids(&projection.messages) == expected,
        "{label}: the transcript must hold every tool call exactly once, in order: {:?}",
        tool_use_ids(&projection.messages),
    );
    ensure!(
        projection.draft_messages.is_empty(),
        "{label}: the committed turn leaves no draft",
    );
    ensure_provider_valid(&projection.messages, &format!("{label}: raw transcript"))?;
    let effective = projection.context_history();
    ensure_unique_tool_use_ids(&effective, &format!("{label}: effective view"))?;
    ensure_provider_valid(&effective, &format!("{label}: effective view"))?;
    ensure_requests_have_unique_tool_use_ids(main, label)
}

/// Runs a steering wake while one of two tool children is still running and
/// checks that the steering exchange compacted, folding the parked turn and
/// the steering message, before its reply.
async fn compacting_steering_wake(
    harness: &Harness,
    main: &ScriptedMainProvider,
    reply: ChatOutcome,
    finish_survivor_first: bool,
) -> Result<(AgentTaskId, AgentTask, RootTurnOutcome)> {
    let (task_id, children) = start_turn_calling(harness, main, &["call_1", "call_2"]).await?;
    let [finished, survivor] = children.as_slice() else {
        bail!("expected two tool children, got {}", children.len());
    };
    harness
        .complete_children(std::slice::from_ref(finished), 1_000)
        .await?;
    wake_for_steering(harness, &task_id, 1_010).await?;
    if finish_survivor_first {
        harness
            .complete_children(std::slice::from_ref(survivor), 1_015)
            .await?;
    }

    main.push(reply)?;
    let parent = harness.acquire(&task_id, 1_020).await?;
    ensure!(parent.state.is_steering_resume());
    let before = harness.projection().await?;
    let inputs = harness.inputs(parent.clone(), 1_020).await?;
    let outcome = resume_for_steering(inputs, &parent, main, &harness.deps(), at(1_030)).await?;
    let after = harness.projection().await?;
    ensure!(
        after.compactions.len() == before.compactions.len() + 1,
        "precondition: the steering exchange compacted",
    );
    let entry = after.compactions.last().context("steering compaction")?;
    ensure!(
        entry.draft_fold_len == parked_messages(&parent)?.len() + 1,
        "the steering fold holds the parked turn and the steering message",
    );
    Ok((task_id, survivor.clone(), outcome))
}

async fn ensure_reparked_only_unfolded(task: &AgentTask, harness: &Harness) -> Result<()> {
    let parked = parked_messages(task)?;
    let projection = harness.projection().await?;
    ensure_disjoint_from_transcript(parked, &projection.messages, "steering re-park")?;
    ensure!(
        parked.len() == 1,
        "the steering exchange folded the parked turn, so the re-park holds only its new \
         assistant message; it holds {}",
        parked.len(),
    );
    Ok(())
}

async fn finish_turn(
    harness: &Harness,
    main: &ScriptedMainProvider,
    task_id: &AgentTaskId,
    children: &[AgentTask],
) -> Result<()> {
    harness.complete_children(children, 2_000).await?;
    main.push(text("all done"))?;
    let parent = harness.acquire(task_id, 2_010).await?;
    let inputs = harness.inputs(parent.clone(), 2_010).await?;
    let outcome = resume_from_children(inputs, &parent, main, &harness.deps(), at(2_020)).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "the fan-in must commit the turn"
    );
    Ok(())
}

/// A steering wake whose exchange compacts re-parks on the survivor through
/// `prepare_steering_repark`. The fold already holds the parked turn and the
/// steering message, so the re-park must carry only the steering answer; the
/// survivor's fan-in then commits each call once.
#[tokio::test]
async fn steering_wake_that_compacts_reparks_only_the_unfolded_turn() -> Result<()> {
    let harness = Harness::new(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Nothing,
    )?;
    let main = ScriptedMainProvider::new();
    let (task_id, survivor, outcome) = compacting_steering_wake(
        &harness,
        &main,
        text("call_1 is done, call_2 is still running"),
        false,
    )
    .await?;
    let (reparked, spawned) = suspended_parts(outcome)?;
    ensure!(
        spawned.is_empty(),
        "the re-park re-attaches the survivor instead of spawning"
    );
    ensure_reparked_only_unfolded(&reparked, &harness).await?;

    finish_turn(&harness, &main, &task_id, std::slice::from_ref(&survivor)).await?;
    ensure_turn_holds_each_tool_call_once(
        &harness,
        &main,
        &["call_1".to_owned(), "call_2".to_owned()],
        "steering re-park",
    )
    .await
}

/// A steering wake that finds every child finished and answers with a new
/// tool call re-suspends through the resume path; with the parked turn and
/// the steering message folded, only the new call may be parked.
#[tokio::test]
async fn steering_wake_that_compacts_and_calls_a_tool_parks_only_the_new_call() -> Result<()> {
    let harness = Harness::new(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Nothing,
    )?;
    let main = ScriptedMainProvider::new();
    let (task_id, _, outcome) =
        compacting_steering_wake(&harness, &main, tool_call("call_3"), true).await?;
    let (suspended, spawned) = suspended_parts(outcome)?;
    ensure!(spawned.len() == 1, "the new tool call spawns one child");
    ensure_reparked_only_unfolded(&suspended, &harness).await?;

    finish_turn(&harness, &main, &task_id, &spawned).await?;
    ensure_turn_holds_each_tool_call_once(
        &harness,
        &main,
        &[
            "call_1".to_owned(),
            "call_2".to_owned(),
            "call_3".to_owned(),
        ],
        "steering tool call",
    )
    .await
}

/// A steering exchange that compacted and then failed is reverted to its
/// pre-wake parked state, but its fold already holds that whole parked turn
/// (and the steering message that answered it). The revert must re-park none
/// of it, and a cancel of the reverted row must commit none of it again.
#[tokio::test]
async fn cancel_after_a_compacted_steering_wake_was_reverted_commits_nothing_twice() -> Result<()> {
    let harness = Harness::new(
        CompactionEngine::Legacy,
        Trigger::Threshold,
        Retention::Nothing,
    )?;
    let main = ScriptedMainProvider::new();
    let (task_id, children) = start_turn_calling(&harness, &main, &["call_1", "call_2"]).await?;
    let finished = children.first().context("first tool child")?;
    harness
        .complete_children(std::slice::from_ref(finished), 1_000)
        .await?;
    wake_for_steering(&harness, &task_id, 1_010).await?;

    main.push(ChatOutcome::InvalidRequest(
        "upstream rejected the request".into(),
    ))?;
    let parent = harness.acquire(&task_id, 1_020).await?;
    let inputs = harness.inputs(parent.clone(), 1_020).await?;
    let Err(error) = resume_for_steering(inputs, &parent, &main, &harness.deps(), at(1_030)).await
    else {
        bail!("the steering exchange must fail after compacting");
    };
    let folded = harness.projection().await?;
    ensure!(
        folded.compactions.len() == 1 && folded.draft_messages.is_empty(),
        "precondition: the failed exchange folded the parked turn into the transcript",
    );
    ensure!(tool_use_ids(&folded.messages) == ["call_1", "call_2"]);

    let reverted = revert_steering_wake(
        &parent,
        &WorkerId::from_string("worker_resume"),
        &LeaseId::from_string("lease_resume"),
        &error,
        &harness.deps(),
        at(1_040),
    )
    .await?;
    ensure!(reverted.status == TaskStatus::WaitingOnChildren);
    ensure!(
        parked_messages(&reverted)?.is_empty(),
        "the revert must not re-park messages the failed exchange folded: {:?}",
        tool_use_ids(parked_messages(&reverted)?),
    );

    cancel_root_turn(&task_id, &harness.deps(), at(1_050)).await?;
    let projection = harness.projection().await?;
    ensure!(
        projection.messages.len() == folded.messages.len(),
        "the cancel must commit nothing the fold holds: the transcript grew from {} to {}",
        folded.messages.len(),
        projection.messages.len(),
    );
    ensure_provider_valid(&projection.messages, "raw transcript after the cancel")?;
    ensure_unique_tool_use_ids(
        &projection.context_history(),
        "effective view after the cancel",
    )
}
