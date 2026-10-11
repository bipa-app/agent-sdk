//! Integration tests for daemon-side auto-compaction.
//!
//! Companion of [`super::compaction`] — exercises both wired
//! call-sites in [`super::root_turn`]:
//!
//! 1. **Pre-call threshold trigger** — when the staged history
//!    exceeds [`CompactionConfig::threshold_tokens`] before
//!    [`super::root_turn::execute_root_turn`] starts streaming, the
//!    worker rewrites the durable projection + staged buffer and
//!    then sends the compacted history to the LLM.
//! 2. **Post-failure prompt-too-long recovery** — when the provider
//!    rejects a turn with `InvalidRequest("prompt is too long…")`,
//!    [`super::root_turn::call_llm_with_retry`] runs an emergency
//!    compaction and retries with the rewritten history instead of
//!    failing the turn.
//!
//! These two tests guard the user-visible regression that motivated
//! M7.5: a long-running thread that crossed Anthropic's 1M
//! cap surfaced
//! `LLM stream error (kind=InvalidRequest): "prompt is too long: …"`
//! to the user with no recovery path. The fixtures below use
//! deterministic mock providers — no live network — so they run in
//! the default `nextest` set.

use std::io::Read;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration as StdDuration;

use agent_sdk::context::{CompactionConfig, CompactionEngine};
use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Message, StopReason, Usage,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::MessageStore;
use anyhow::{Context, Result};
use async_trait::async_trait;
use time::Duration;
use time::OffsetDateTime;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt::TurnAttemptOutcome;
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::activity::ActivityBeacon;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use crate::worker::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};

// ─────────────────────────────────────────────────────────────────────
// Mock provider — sequential canned responses.
//
// Each call consumes the next response from the queue. Lets a single
// test simulate the realistic three-call shape the compaction flow
// produces:
//   1. Original LLM call → fails or compaction-threshold is checked
//      against the staged history
//   2. Compactor's summarisation call → returns a synthetic summary
//   3. Retry of the original turn → returns the actual reply
// ─────────────────────────────────────────────────────────────────────

struct ScriptedProvider {
    responses: Mutex<Vec<ChatOutcome>>,
    call_count: AtomicUsize,
    supports_historical_images: bool,
    requests: Mutex<Vec<ChatRequest>>,
    provider_name: &'static str,
    model_name: &'static str,
}

impl ScriptedProvider {
    fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: Mutex::new(responses),
            call_count: AtomicUsize::new(0),
            supports_historical_images: false,
            requests: Mutex::new(Vec::new()),
            provider_name: "mock",
            model_name: "mock-model",
        }
    }

    fn with_identity(provider_name: &'static str, model_name: &'static str) -> Self {
        Self {
            provider_name,
            model_name,
            ..Self::new(vec![ok_response("[summary]")])
        }
    }

    fn image_capable(responses: Vec<ChatOutcome>) -> Self {
        Self {
            supports_historical_images: true,
            ..Self::new(responses)
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    fn prompts(&self) -> Result<Vec<String>> {
        let requests = self
            .requests
            .lock()
            .map_err(|_| anyhow::anyhow!("ScriptedProvider request mutex poisoned"))?;
        Ok(requests
            .iter()
            .map(|request| {
                request
                    .messages
                    .iter()
                    .find_map(|message| match &message.content {
                        Content::Text(text) => Some(text.clone()),
                        Content::Blocks(_) => None,
                    })
                    .unwrap_or_default()
            })
            .collect())
    }

    fn captured_requests(&self) -> Result<Vec<ChatRequest>> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .map_err(|_| anyhow::anyhow!("ScriptedProvider request mutex poisoned"))
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.requests
            .lock()
            .map_err(|_| anyhow::anyhow!("ScriptedProvider request mutex poisoned"))?
            .push(request);
        let mut responses = self
            .responses
            .lock()
            .map_err(|_| anyhow::anyhow!("ScriptedProvider mutex poisoned"))?;
        if responses.is_empty() {
            anyhow::bail!("ScriptedProvider response queue exhausted");
        }
        Ok(responses.remove(0))
    }

    fn model(&self) -> &str {
        self.model_name
    }

    fn provider(&self) -> &'static str {
        self.provider_name
    }

    fn supports_historical_image_blocks(&self) -> bool {
        self.supports_historical_images
    }
}

struct BlockingProvider {
    call_count: AtomicUsize,
    started: Notify,
}

impl BlockingProvider {
    fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            started: Notify::new(),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for BlockingProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        std::future::pending::<Result<ChatOutcome>>().await
    }

    fn model(&self) -> &'static str {
        "blocking-model"
    }

    fn provider(&self) -> &'static str {
        "blocking"
    }
}

struct CancelOnResponseProvider {
    cancel: CancellationToken,
    call_count: AtomicUsize,
}

impl CancelOnResponseProvider {
    fn new(cancel: CancellationToken) -> Self {
        Self {
            cancel,
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for CancelOnResponseProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.cancel.cancel();
        Ok(ok_response("[summary completed before cancellation fence]"))
    }

    fn model(&self) -> &'static str {
        "cancel-on-response-model"
    }

    fn provider(&self) -> &'static str {
        "cancel-on-response"
    }
}

fn ok_response(text: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: "msg_mock".into(),
        content: vec![ContentBlock::Text {
            text: text.to_owned(),
        }],
        model: "mock-model".into(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            served_speed: None,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            request_input_tokens: Some(100),
        },
    })
}

fn zero_usage_response(text: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: "msg_mock_zero_usage".into(),
        content: vec![ContentBlock::Text {
            text: text.to_owned(),
        }],
        model: "mock-model".into(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            served_speed: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        },
    })
}

// ─────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn thread_id() -> ThreadId {
    ThreadId::from_string("t-compaction-integration")
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a helpful test assistant.".into(),
        max_tokens: 1024,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn sample_bootstrap(task: AgentTask) -> WorkerBootstrapContext {
    let task_id = task.id.clone();
    let thread_id = task.thread_id.clone();
    WorkerBootstrapContext {
        task,
        definition: sample_definition(),
        thread_id,
        task_id,
        worker_id: WorkerId::from_string("worker_compaction"),
        lease_id: LeaseId::from_string("lease_compaction"),
    }
}

struct Fixtures {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
    event_notifier: Arc<EventNotifier>,
}

impl Fixtures {
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

    fn deps_with_compaction<'a>(
        &'a self,
        config: &'a CompactionConfig,
        provider: &'a Arc<dyn LlmProvider>,
    ) -> RootTurnDeps<'a> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
            event_repo: &self.events,
            event_notifier: &self.event_notifier,
            subagent_spawn_selector: None,
            compaction_config: Some(config),
            compaction_provider: Some(provider),
            compaction_artifact_store: None,
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
        }
    }
}

async fn create_and_acquire_root_task(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await?;
    let acquired = store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_compaction"),
            LeaseId::from_string("lease_compaction"),
            t0() + Duration::seconds(300),
            t0(),
        )
        .await?
        .expect("task should be acquirable");
    Ok(acquired)
}

/// Seed the durable projection with `count` user/assistant turns so
/// the staged store later picks up enough messages to cross
/// `min_messages_for_compaction`. Each message carries `text` so the
/// estimator returns a non-zero token count.
///
/// We seed via `set_draft` rather than `replace_history` because
/// `recover_thread` only includes the projection's committed
/// `messages` when `thread.committed_turns > 0` and a checkpoint is
/// present — both of which require a more involved fixture. The
/// `draft_messages` field is included in `view.messages` even on
/// fresh threads, which is exactly what we want: messages flow into
/// the staged buffer the worker then consults for the compaction
/// threshold check.
async fn seed_projection_history(
    store: &InMemoryMessageProjectionStore,
    thread_id: &ThreadId,
    count: usize,
    text: &str,
) -> Result<()> {
    let mut messages = Vec::with_capacity(count * 2);
    for i in 0..count {
        messages.push(Message::user(format!("user-{i}: {text}")));
        messages.push(Message::assistant(format!("assistant-{i}: {text}")));
    }
    store
        .set_draft(thread_id, messages, t0())
        .await
        .map_err(|e| anyhow::anyhow!("seed projection draft: {e}"))?;
    Ok(())
}

fn read_artifact_bytes(store: &agent_sdk::ArtifactStore, artifact_id: u64) -> Result<Vec<u8>> {
    let mut file = store
        .resolve(artifact_id)
        .with_context(|| format!("resolve artifact {artifact_id}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read artifact {artifact_id}"))?;
    Ok(bytes)
}

/// Audit shape for the local Snapcompact turn: a zero-usage
/// `context_compaction` attempt followed by the billed main call.
async fn assert_snapcompact_audit_attempts(
    fixtures: &Fixtures,
    task_id: &crate::journal::task::AgentTaskId,
) -> Result<()> {
    let attempts = fixtures.attempts.list_by_task(task_id).await?;
    assert_eq!(
        attempts.len(),
        2,
        "local compaction and the main call must keep distinct audit attempts",
    );
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[0].input_tokens, Some(0));
    assert_eq!(attempts[0].output_tokens, Some(0));
    assert_eq!(attempts[0].cached_input_tokens, Some(0));
    assert_eq!(attempts[0].cache_creation_input_tokens, Some(0));
    assert_eq!(
        attempts[0]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert_eq!(attempts[1].input_tokens, Some(100));
    assert_eq!(attempts[1].output_tokens, Some(50));
    Ok(())
}

/// The raw committed projection must preserve every seeded message
/// byte-for-byte and record exactly one compaction entry covering them.
async fn assert_snapcompact_promoted_projection(
    fixtures: &Fixtures,
    seeded_history: &[Message],
) -> Result<()> {
    let seeded_len = seeded_history.len();
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after Snapcompact root turn")?;
    assert_eq!(
        projection.messages.len(),
        seeded_len + 2,
        "raw committed history must preserve every old message and append the fresh turn",
    );
    assert_eq!(
        &projection.messages[..seeded_len],
        seeded_history,
        "compaction must promote the seeded draft without rewriting any raw message",
    );
    assert!(matches!(
        &projection.messages[0].content,
        Content::Text(text) if text.contains("user-0: old-transcript-body")
    ));
    assert!(matches!(
        &projection.messages[seeded_len - 1].content,
        Content::Text(text) if text.contains("assistant-5: old-transcript-body")
    ));
    assert_eq!(
        projection.draft_messages,
        [] as [agent_sdk_foundation::Message; 0]
    );
    assert_eq!(projection.compactions.len(), 1);
    let entry = &projection.compactions[0];
    assert_eq!(entry.compacted_start, 0);
    assert_eq!(entry.compacted_end, seeded_len);
    assert_eq!(entry.source_message_count, seeded_len);
    assert!(entry.generated_summary);
    Ok(())
}

/// The effective checkpoint must persist exact artifact URIs (source +
/// PNG frames), and the captured main provider request must carry those
/// frames hydrated back to inline base64. Returns the effective history
/// for follow-up staging checks.
async fn assert_snapcompact_checkpoint_and_hydration(
    fixtures: &Fixtures,
    artifact_store: &agent_sdk::ArtifactStore,
    scripted: &ScriptedProvider,
) -> Result<Vec<Message>> {
    use agent_sdk_foundation::llm::canonical_snapcompact_checkpoint;
    use base64::Engine as _;

    let effective = fixtures.messages.get_history(&thread_id()).await?;
    let checkpoint = effective
        .first()
        .context("effective projection must start with a Snapcompact checkpoint")?;
    let metadata = canonical_snapcompact_checkpoint(checkpoint)
        .context("effective projection did not start with a canonical Snapcompact checkpoint")?;
    assert!(
        metadata.frame_count > 0,
        "large repeated history must render at least one frame",
    );
    let Content::Blocks(stored_blocks) = &checkpoint.content else {
        anyhow::bail!("canonical Snapcompact checkpoint must use blocks");
    };
    let ContentBlock::CompactionSummary { artifact_ids, .. } = &stored_blocks[0] else {
        anyhow::bail!("canonical Snapcompact checkpoint must start with its summary");
    };
    assert_eq!(
        artifact_ids.len(),
        metadata.frame_count as usize + 1,
        "checkpoint must retain exactly its source and frame artifacts",
    );
    assert!(artifact_ids.contains(&metadata.source_artifact_id));

    let source_bytes = read_artifact_bytes(artifact_store, metadata.source_artifact_id)?;
    let source_text = String::from_utf8(source_bytes)?;
    assert!(source_text.contains("user-0: old-transcript-body"));
    assert!(source_text.contains("assistant-5: old-transcript-body"));

    let mut stored_frame_ids = Vec::new();
    for block in stored_blocks {
        let ContentBlock::Image { source } = block else {
            continue;
        };
        let artifact_id = source
            .data
            .strip_prefix("artifact://")
            .context("stored Snapcompact frame must be an exact artifact URI")?
            .parse::<u64>()?;
        assert_eq!(source.data, format!("artifact://{artifact_id}"));
        assert!(!source.data.contains("base64"));
        assert!(artifact_ids.contains(&artifact_id));
        let frame = read_artifact_bytes(artifact_store, artifact_id)?;
        assert!(frame.starts_with(b"\x89PNG\r\n\x1a\n"));
        stored_frame_ids.push(artifact_id);
    }
    assert_eq!(stored_frame_ids.len(), metadata.frame_count as usize);

    let requests = scripted.captured_requests()?;
    assert_eq!(requests.len(), 1);
    let provider_checkpoint = requests[0]
        .messages
        .iter()
        .find(|message| {
            matches!(
                &message.content,
                Content::Blocks(blocks)
                    if blocks.iter().any(|block| matches!(
                        block,
                        ContentBlock::CompactionSummary {
                            snapcompact: Some(_),
                            ..
                        }
                    ))
            )
        })
        .context("main provider request must contain the Snapcompact checkpoint")?;
    let Content::Blocks(provider_blocks) = &provider_checkpoint.content else {
        anyhow::bail!("provider checkpoint must use blocks");
    };
    let provider_frames: Vec<_> = provider_blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Image { source } => Some(source),
            _ => None,
        })
        .collect();
    assert_eq!(provider_frames.len(), stored_frame_ids.len());
    for (source, artifact_id) in provider_frames.iter().zip(&stored_frame_ids) {
        assert!(!source.data.starts_with("artifact://"));
        let decoded = base64::engine::general_purpose::STANDARD.decode(source.data.as_bytes())?;
        assert!(decoded.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(decoded, read_artifact_bytes(artifact_store, *artifact_id)?);
    }
    Ok(effective)
}

/// Freshly measured main-call usage must prevent an immediate
/// Snapcompact re-trigger against the just-compacted history.
async fn assert_no_immediate_snapcompact_retrigger(
    deps: &RootTurnDeps<'_>,
    fixtures: &Fixtures,
    scripted: &ScriptedProvider,
    effective: Vec<Message>,
) -> Result<()> {
    let staged = crate::journal::staged::StagedMessageStore::new(thread_id(), effective);
    let immediate = super::compaction::maybe_compact_staged_history(
        deps,
        scripted,
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(2),
    )
    .await?;
    assert!(
        !immediate.completed && !immediate.applied,
        "fresh measured main usage must prevent an immediate Snapcompact loop",
    );
    assert_eq!(immediate.llm_usage.input_tokens, 0);
    assert_eq!(immediate.llm_usage.output_tokens, 0);
    assert_eq!(scripted.calls(), 1);
    assert_eq!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection after immediate trigger check")?
            .compactions
            .len(),
        1,
    );
    Ok(())
}

#[tokio::test]
async fn snapcompact_auto_compaction_persists_uris_and_hydrates_main_request() -> Result<()> {
    const SEEDED_TURNS: usize = 6;
    const THRESHOLD_TOKENS: usize = 20_000;

    let fixtures = Fixtures::new();
    let old_body = "old-transcript-body ".repeat(4_000);
    seed_projection_history(&fixtures.messages, &thread_id(), SEEDED_TURNS, &old_body).await?;
    let seeded_history = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("seeded projection")?
        .draft_messages;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Snapcompact)
        .with_threshold_tokens(THRESHOLD_TOKENS)
        .with_retain_recent(0)
        .with_min_messages(1);
    let scripted = Arc::new(ScriptedProvider::image_capable(vec![ok_response(
        "main response after local compaction",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(
        artifact_dir.path().join("t-compaction-integration"),
    ));
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;
    let fresh_prompt = format!("current-request-marker {}", "n".repeat(100_000));

    let outcome = execute_root_turn(
        inputs,
        &fresh_prompt,
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;
    let RootTurnOutcome::Completed {
        response_text,
        commit,
        ..
    } = outcome
    else {
        anyhow::bail!("Snapcompact root turn did not complete");
    };
    assert_eq!(response_text, "main response after local compaction");
    assert_eq!(
        scripted.calls(),
        1,
        "local Snapcompact must not spend a summarizer provider call",
    );

    assert_snapcompact_audit_attempts(&fixtures, &task_id).await?;
    assert_eq!(commit.thread.total_usage.input_tokens, 100);
    assert_eq!(commit.thread.total_usage.output_tokens, 50);

    assert_snapcompact_promoted_projection(&fixtures, &seeded_history).await?;

    let effective =
        assert_snapcompact_checkpoint_and_hydration(&fixtures, &artifact_store, &scripted).await?;

    assert_no_immediate_snapcompact_retrigger(&deps, &fixtures, &scripted, effective).await?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

/// Pre-call path: the staged history alone is already over the
/// configured token threshold (and message count), so the worker
/// should compact, rewrite the projection, and then send the
/// compacted history (plus the fresh user prompt) to the LLM.
///
/// The provider script verifies the call order: the first call is
/// the compactor's summarisation request, and the second is the
/// turn's actual LLM call.
#[tokio::test]
async fn pre_call_threshold_triggers_compaction() -> Result<()> {
    let fixtures = Fixtures::new();

    // 12 turns × 2 messages = 24 messages, well over the default
    // `min_messages_for_compaction = 20` and the threshold below.
    seed_projection_history(
        &fixtures.messages,
        &thread_id(),
        12,
        &"x".repeat(200), // bump tokens above the tiny threshold
    )
    .await?;

    // Tiny threshold so the seeded history is guaranteed over budget.
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(10);

    // Provider script:
    //   call 1 → compactor summarisation
    //   call 2 → original turn
    let scripted = Arc::new(ScriptedProvider::new(vec![
        ok_response("[summary] previous 24 messages folded"),
        ok_response("Hello after compaction"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();

    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(
        inputs,
        "Hi after compaction!",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let RootTurnOutcome::Completed {
        response_text,
        commit,
        ..
    } = outcome
    else {
        panic!("expected Completed turn");
    };
    assert_eq!(response_text, "Hello after compaction");

    // Provider was called twice: once by the compactor, once by the
    // turn. The compactor consumed the summarisation slot first.
    assert_eq!(scripted.calls(), 2);
    assert!(
        scripted
            .prompts()?
            .first()
            .is_some_and(|prompt| prompt.contains("Compaction purpose: pre-spawn.")),
        "auto compaction must select the pre-spawn prompt",
    );
    let attempts = fixtures.attempts.list_by_task(&task_id).await?;
    assert_eq!(
        attempts.len(),
        2,
        "compaction and chat each own one attempt"
    );
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[0].input_tokens, Some(100));
    assert_eq!(attempts[0].output_tokens, Some(50));
    assert_eq!(
        attempts[0]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert_eq!(attempts[1].input_tokens, Some(100));
    assert_eq!(attempts[1].output_tokens, Some(50));
    assert_eq!(commit.thread.total_usage.input_tokens, 200);
    assert_eq!(commit.thread.total_usage.output_tokens, 100);

    assert_precall_compacted_projection(&fixtures, &commit).await?;

    assert_precall_compaction_events(&fixtures).await?;

    Ok(())
}

/// The pre-call trigger must shrink the effective context, keep the raw
/// projection append-only, record one compaction entry, and hand the
/// compacted view to checkpoint/fork consumers.
async fn assert_precall_compacted_projection(
    fixtures: &Fixtures,
    commit: &crate::journal::CommitOutcome,
) -> Result<()> {
    let context = fixtures.messages.get_history(&thread_id()).await?;
    assert!(
        context.len() < 24,
        "expected effective context to shrink after compaction, found {} messages",
        context.len(),
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection should exist after compaction")?;
    assert_eq!(
        projection.messages.len(),
        26,
        "append-only compaction must preserve 24 originals before appending the fresh turn",
    );
    assert_eq!(projection.compactions.len(), 1);
    assert!(
        projection.draft_messages.is_empty(),
        "persisted draft must be folded and cleared by append_compaction",
    );
    let entry = &projection.compactions[0];
    assert_eq!(
        entry.source_message_count, 24,
        "the compactor must see the exact projection source including all 24 draft messages",
    );
    assert_eq!(entry.compacted_start, 0);
    assert!(entry.compacted_end > entry.compacted_start);
    assert!(entry.compacted_end <= projection.messages.len());
    assert_eq!(
        serde_json::to_value(&commit.checkpoint.messages)?,
        serde_json::to_value(&context)?,
        "checkpoint and fork consumers must inherit the effective compacted view",
    );
    Ok(())
}

/// A `ContextCompacted` event was committed so subscribers (TUI,
/// desktop) can render the compaction in their transcripts, and
/// reopening after compaction must not duplicate lifecycle events.
async fn assert_precall_compaction_events(fixtures: &Fixtures) -> Result<()> {
    let events = fixtures.events.get_events(&thread_id()).await?;
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ContextCompacted { .. })),
        "expected ContextCompacted event, got events: {:?}",
        events
            .iter()
            .map(|e| event_kind(&e.event))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.event, AgentEvent::UserInput { .. }))
            .count(),
        1,
        "reopening after compaction must not duplicate user input",
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.event, AgentEvent::Start { .. }))
            .count(),
        1,
        "reopening after compaction must not duplicate turn start",
    );
    Ok(())
}

#[tokio::test]
async fn zero_usage_completed_compaction_keeps_distinct_attempts() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"q".repeat(200)).await?;
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    let scripted = Arc::new(ScriptedProvider::new(vec![
        zero_usage_response("[summary with omitted usage]"),
        ok_response("main response"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(
        inputs,
        "main request",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;
    let RootTurnOutcome::Completed { commit, .. } = outcome else {
        panic!("zero-usage compaction should still continue to the main call");
    };
    assert_eq!(scripted.calls(), 2);

    let mut attempts = fixtures.attempts.list_by_task(&task_id).await?;
    attempts.sort_by_key(|attempt| attempt.attempt_number);
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[0].input_tokens, Some(0));
    assert_eq!(attempts[0].output_tokens, Some(0));
    assert_eq!(
        attempts[0]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert!(
        attempts[1]
            .response_blob
            .as_ref()
            .is_some_and(|blob| blob.get("operation").is_none()),
        "main response must close a distinct non-compaction attempt",
    );
    assert_eq!(attempts[1].input_tokens, Some(100));
    assert_eq!(attempts[1].output_tokens, Some(50));
    assert_eq!(commit.thread.total_usage.input_tokens, 100);
    assert_eq!(commit.thread.total_usage.output_tokens, 50);
    Ok(())
}

#[tokio::test]
async fn cancellation_during_summarization_leaves_compaction_state_unchanged() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let id = thread_id();
    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("user-{index}: {}", "x".repeat(200))));
        history.push(Message::assistant(format!(
            "assistant-{index}: {}",
            "x".repeat(200)
        )));
    }
    fixtures
        .messages
        .commit_messages(&id, history.clone(), t0())
        .await?;
    let staged = StagedMessageStore::new(id.clone(), history.clone());
    let projection_before = fixtures
        .messages
        .get(&id)
        .await?
        .context("projection should exist before cancellation")?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    {
        let blocking = Arc::new(BlockingProvider::new());
        let provider: Arc<dyn LlmProvider> = blocking.clone();
        let cancel = CancellationToken::new();
        let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
        deps.cancel = Some(&cancel);
        let envelope = sample_envelope();

        let compaction = super::compaction::maybe_compact_staged_history(
            &deps,
            provider.as_ref(),
            &staged,
            &id,
            0,
            &envelope,
            t0() + Duration::seconds(1),
        );
        tokio::pin!(compaction);
        tokio::select! {
            () = blocking.started.notified() => {}
            result = &mut compaction => {
                panic!(
                    "pre-call compaction completed before the blocking provider was cancelled: \
                     {result:?}"
                );
            }
        }

        cancel.cancel();
        tokio::time::timeout(StdDuration::from_millis(250), &mut compaction)
            .await
            .context("cancelled pre-call compaction did not stop promptly")??;
        assert_eq!(blocking.calls(), 1);
    }

    {
        let blocking = Arc::new(BlockingProvider::new());
        let provider: Arc<dyn LlmProvider> = blocking.clone();
        let cancel = CancellationToken::new();
        let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
        deps.cancel = Some(&cancel);

        let compaction = super::compaction::compact_after_overflow(
            &deps,
            provider.as_ref(),
            sample_definition().max_tokens,
            &staged,
            &id,
            t0() + Duration::seconds(2),
        );
        tokio::pin!(compaction);
        tokio::select! {
            () = blocking.started.notified() => {}
            result = &mut compaction => {
                panic!(
                    "overflow compaction completed before the blocking provider was cancelled: \
                     {result:?}"
                );
            }
        }

        cancel.cancel();
        let applied = tokio::time::timeout(StdDuration::from_millis(250), &mut compaction)
            .await
            .context("cancelled overflow compaction did not stop promptly")??;
        assert!(
            !applied.applied,
            "cancelled overflow compaction must report no recovery"
        );
        assert_eq!(blocking.calls(), 1);
    }
    assert_cancelled_compaction_state_unchanged(
        &fixtures,
        &staged,
        &id,
        &history,
        &projection_before,
    )
    .await?;

    Ok(())
}

/// A cancelled compaction must leave every durable and staged surface
/// untouched: staged history, durable messages, draft, compaction
/// entries, and the event log.
async fn assert_cancelled_compaction_state_unchanged(
    fixtures: &Fixtures,
    staged: &crate::journal::staged::StagedMessageStore,
    id: &ThreadId,
    history: &[Message],
    projection_before: &crate::MessageProjection,
) -> Result<()> {
    assert_eq!(
        serde_json::to_value(staged.get_history(id).await?)?,
        serde_json::to_value(history)?,
        "cancelled compaction must not rewrite staged history",
    );

    let projection_after = fixtures
        .messages
        .get(id)
        .await?
        .context("projection should remain after cancellation")?;
    assert_eq!(
        serde_json::to_value(&projection_after.messages)?,
        serde_json::to_value(&projection_before.messages)?,
        "cancelled compaction must not mutate durable messages",
    );
    assert_eq!(
        serde_json::to_value(&projection_after.draft_messages)?,
        serde_json::to_value(&projection_before.draft_messages)?,
        "cancelled compaction must not fold or clear the durable draft",
    );
    assert_eq!(
        projection_after.compactions.len(),
        projection_before.compactions.len(),
        "cancelled compaction must not append a durable compaction entry",
    );

    let events = fixtures.events.get_events(&thread_id()).await?;
    assert!(
        events
            .iter()
            .all(|event| !matches!(event.event, AgentEvent::ContextCompacted { .. })),
        "cancelled compaction must not commit ContextCompacted",
    );
    Ok(())
}

#[tokio::test]
async fn completed_compaction_cancel_race_audits_usage_without_applying() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"r".repeat(200)).await?;
    let projection_before = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection before cancellation race")?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    let cancel = CancellationToken::new();
    let scripted = Arc::new(CancelOnResponseProvider::new(cancel.clone()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let activity = ActivityBeacon::new();
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.cancel = Some(&cancel);
    deps.activity = Some(&activity);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;
    let error = execute_root_turn(
        inputs,
        "cancel while summarizing",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await
    .err()
    .context("cancellation race should stop the turn")?;
    assert!(
        super::root_turn::is_root_turn_cancelled(&error),
        "completed-response cancellation must retain the typed cancel marker: {error:#}",
    );
    assert_eq!(scripted.calls(), 1);

    let attempts = fixtures.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[0].input_tokens, Some(100));
    assert_eq!(attempts[0].output_tokens, Some(50));
    assert_eq!(
        attempts[0]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("applied"))
            .and_then(serde_json::Value::as_bool),
        Some(false),
    );
    let live_usage = activity.usage();
    assert_eq!(live_usage.input_tokens, 100);
    assert_eq!(live_usage.output_tokens, 50);

    let projection_after = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after cancellation race")?;
    assert_eq!(
        serde_json::to_value(&projection_after.messages)?,
        serde_json::to_value(&projection_before.messages)?,
    );
    assert_eq!(
        serde_json::to_value(&projection_after.draft_messages)?,
        serde_json::to_value(&projection_before.draft_messages)?,
    );
    assert_eq!(
        projection_after.compactions.len(),
        projection_before.compactions.len(),
    );
    let events = fixtures.events.get_events(&thread_id()).await?;
    assert!(
        events
            .iter()
            .all(|event| !matches!(event.event, AgentEvent::ContextCompacted { .. })),
        "cancellation fence must suppress ContextCompacted",
    );
    Ok(())
}

#[tokio::test]
async fn no_progress_compaction_is_skipped_and_the_turn_proceeds() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"n".repeat(200)).await?;
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    // The summarizer returns an oversized summary, so the assembled view is
    // no smaller than the source — a no-progress compaction. The doctrine
    // (ENG-9651 follow-up): this must SKIP the compaction and let the turn
    // proceed with the uncompacted history, never fail the turn.
    // The first queued response feeds the compaction summarizer (oversized
    // → no progress → skip); the second feeds the turn's own LLM call, which
    // then completes against the uncompacted history.
    let provider = Arc::new(ScriptedProvider::new(vec![
        ok_response(&"z".repeat(20_000)),
        ok_response("turn completed without compacting"),
    ]));
    let turn_provider: Arc<dyn LlmProvider> = provider.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &turn_provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;
    let outcome = execute_root_turn(
        inputs,
        "trigger no progress",
        turn_provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    // The turn COMPLETES — the no-progress compaction was skipped, not fatal.
    let RootTurnOutcome::Completed { .. } = outcome else {
        panic!("a no-progress compaction must not fail the turn: {outcome:?}");
    };

    // No compaction boundary was recorded.
    assert_eq!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection after skipped compaction")?
            .compactions
            .len(),
        0,
        "a skipped compaction records no boundary",
    );
    Ok(())
}

/// A summarizer response carrying only a thinking block (no text) — what an
/// adaptive-thinking model returns when its visible text is empty.
fn thinking_only_response() -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: "msg_mock_thinking".into(),
        content: vec![agent_sdk::llm::ContentBlock::Thinking {
            thinking: "reasoning with no visible answer".into(),
            signature: None,
        }],
        model: "mock-model".into(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            served_speed: None,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        },
    })
}

#[tokio::test]
async fn thinking_only_summarization_is_skipped_and_the_turn_proceeds() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"n".repeat(200)).await?;
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    // The summarizer returns a thinking-only response (no text) — the
    // adaptive-thinking failure mode. The doctrine: this must SKIP the
    // compaction and let the turn proceed uncompacted, never fail it.
    // First queued response feeds the summarizer (thinking-only → skip);
    // the second feeds the turn's own call.
    let provider = Arc::new(ScriptedProvider::new(vec![
        thinking_only_response(),
        ok_response("turn completed without a summary"),
    ]));
    let turn_provider: Arc<dyn LlmProvider> = provider.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &turn_provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;
    let outcome = execute_root_turn(
        inputs,
        "trigger thinking-only summarization",
        turn_provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let RootTurnOutcome::Completed { .. } = outcome else {
        panic!("a thinking-only summarization must not fail the turn: {outcome:?}");
    };
    assert_eq!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection after skipped compaction")?
            .compactions
            .len(),
        0,
        "a skipped compaction records no boundary",
    );
    Ok(())
}

/// Finding #4: when compaction folds history into the committed
/// projection, the in-flight draft must be cleared — otherwise a turn
/// that later fails (so the commit path never clears the draft) leaves
/// recovery folding the same messages in twice (compacted projection +
/// raw draft). Here the turn's LLM call fails *after* the pre-call
/// compaction; the draft must still be empty.
#[tokio::test]
async fn compaction_clears_draft_even_when_turn_then_fails() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"x".repeat(200)).await?;

    // Precondition: the draft slot is populated.
    let before = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection should exist after seeding")?;
    assert!(before.has_draft(), "precondition: draft seeded");

    let cfg = CompactionConfig::default().with_threshold_tokens(10);
    // call 1 → compactor summarisation; call 2 → turn LLM, which fails
    // with a non-retryable InvalidRequest so the turn never commits.
    let scripted = Arc::new(ScriptedProvider::new(vec![
        ok_response("[summary]"),
        ChatOutcome::InvalidRequest("bad request".into()),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let result = execute_root_turn(
        inputs,
        "go",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await;
    assert!(result.is_err(), "the InvalidRequest turn must fail");

    // The pre-call compaction cleared the draft; the failed turn never
    // ran the commit path, so this is the *only* thing that could have
    // cleared it.
    let after = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection should still exist")?;
    assert!(
        !after.has_draft(),
        "compaction must clear the in-flight draft so recovery does not double-fold it",
    );

    Ok(())
}

/// Post-failure path: the provider returns
/// `InvalidRequest("prompt is too long: …")` on the first attempt,
/// the worker should run emergency compaction and retry with the
/// rewritten history. The second attempt succeeds.
#[tokio::test]
async fn prompt_too_long_triggers_emergency_compaction_and_retry() -> Result<()> {
    let fixtures = Fixtures::new();

    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"y".repeat(200)).await?;

    // High threshold so the pre-call check does NOT fire — we want
    // the post-failure path to be the one that runs compaction.
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(usize::MAX);

    let scripted = Arc::new(ScriptedProvider::new(vec![
        // Original turn: provider rejects with the exact Anthropic
        // 1M-cap error shape that surfaced the user-visible bug.
        ChatOutcome::InvalidRequest("prompt is too long: 1010596 tokens > 1000000 maximum".into()),
        // Compactor's summarisation call.
        ok_response("[emergency summary]"),
        // Retry of the original turn after compaction.
        ok_response("Hello after recovery"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();

    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(
        inputs,
        "Tell me a joke",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let RootTurnOutcome::Completed {
        response_text,
        commit,
        ..
    } = outcome
    else {
        panic!("expected Completed turn after emergency compaction");
    };
    assert_eq!(response_text, "Hello after recovery");

    // 3 calls: rejection, compactor, retry.
    assert_eq!(scripted.calls(), 3);
    assert!(
        scripted
            .prompts()?
            .iter()
            .any(|prompt| prompt.contains("Compaction purpose: overflow recovery.")),
        "overflow recovery must select the overflow prompt",
    );
    let mut attempts = fixtures.attempts.list_by_task(&task_id).await?;
    attempts.sort_by_key(|attempt| attempt.attempt_number);
    assert_eq!(
        attempts.len(),
        3,
        "overflow rejection, compaction, and retry each own one attempt",
    );
    assert_eq!(
        attempts[0].outcome,
        Some(TurnAttemptOutcome::InvalidRequest),
    );
    assert_eq!(attempts[0].input_tokens, Some(0));
    assert_eq!(attempts[1].outcome, Some(TurnAttemptOutcome::Success));
    assert_eq!(attempts[1].input_tokens, Some(100));
    assert_eq!(attempts[1].output_tokens, Some(50));
    assert_eq!(
        attempts[1]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert_eq!(attempts[2].input_tokens, Some(100));
    assert_eq!(attempts[2].output_tokens, Some(50));
    assert_eq!(commit.thread.total_usage.input_tokens, 200);
    assert_eq!(commit.thread.total_usage.output_tokens, 100);

    // Compaction event committed.
    let events = fixtures.events.get_events(&thread_id()).await?;
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ContextCompacted { .. })),
        "expected ContextCompacted event after emergency compaction",
    );

    // The retry succeeded so a `Done` event landed; an `Error` would
    // mean the recovery branch never engaged.
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. })),
        "expected Done event from successful retry",
    );
    assert!(
        events
            .iter()
            .all(|e| !matches!(e.event, AgentEvent::Error { .. })),
        "no Error event should be emitted when emergency compaction succeeds",
    );

    Ok(())
}

/// The request rebuilt after emergency compaction carries the definition's
/// request-only context again, while the summarization call and the compacted
/// projection never see it.
#[tokio::test]
async fn prompt_too_long_retry_reapplies_request_context() -> Result<()> {
    const MARKER: &str = "REQUEST-ONLY-OVERFLOW-91c2";
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"y".repeat(200)).await?;
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(usize::MAX);
    let scripted = Arc::new(ScriptedProvider::new(vec![
        ChatOutcome::InvalidRequest("prompt is too long: 1010596 tokens > 1000000 maximum".into()),
        ok_response("[emergency summary]"),
        ok_response("Hello after recovery"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let mut bootstrap = sample_bootstrap(task);
    bootstrap.definition.request_context_fn = Some(Arc::new(|_| Some(MARKER.to_owned())));
    let inputs = build_root_worker_inputs(
        bootstrap,
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;
    let outcome = execute_root_turn(
        inputs,
        "Tell me a joke",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;
    let RootTurnOutcome::Completed { .. } = outcome else {
        anyhow::bail!("expected Completed turn after emergency compaction");
    };

    let requests = scripted.captured_requests()?;
    let [rejected, summarization, retry] = requests.as_slice() else {
        anyhow::bail!("expected rejection, summarization, and retry requests");
    };
    let ends_with_context = |request: &ChatRequest| {
        matches!(
            request.messages.last().map(|message| &message.content),
            Some(Content::Blocks(blocks)) if matches!(
                blocks.last(),
                Some(ContentBlock::Text { text })
                    if text == &format!("<system-reminder>\n{MARKER}\n</system-reminder>")
            )
        )
    };
    assert!(ends_with_context(rejected));
    assert!(ends_with_context(retry));
    assert!(retry.messages.len() < rejected.messages.len());
    assert!(!format!("{:?}", summarization.messages).contains(MARKER));

    let projection = fixtures.messages.get(&thread_id()).await?;
    assert!(!format!("{projection:?}").contains(MARKER));
    let events = fixtures.events.get_events(&thread_id()).await?;
    assert!(!format!("{events:?}").contains(MARKER));
    Ok(())
}

/// `OpenAI` Responses regression: a seeded history whose older turns carry
/// `OpaqueReasoning` blocks (encrypted provider scratchpad) must still
/// recover when the provider rejects the turn with the Responses-API
/// overflow prose. The widened matcher engages the emergency compaction,
/// the summarized prefix's opaque blocks are dropped with it, and the
/// retry succeeds on the rewritten projection.
#[tokio::test]
async fn prompt_too_long_recovers_history_carrying_opaque_reasoning() -> Result<()> {
    let fixtures = Fixtures::new();

    // Seed 12 turns (24 messages ≥ the default min_messages); the FIRST
    // assistant message carries an opaque reasoning block like every
    // assistant turn in an OpenAI Responses thread. It lands in the
    // summarized prefix (default retain_recent=10 ⇒ split at 14).
    let mut messages = Vec::new();
    for i in 0..12 {
        messages.push(Message::user(format!("user-{i}: {}", "y".repeat(200))));
        if i == 0 {
            messages.push(Message::assistant_with_content(vec![
                ContentBlock::OpaqueReasoning {
                    provider: "openai-responses".to_owned(),
                    data: serde_json::json!({"id": "rs_0", "encrypted_content": "ciphertext"}),
                },
                ContentBlock::Text {
                    text: format!("assistant-{i}: {}", "y".repeat(200)),
                },
            ]));
        } else {
            messages.push(Message::assistant(format!(
                "assistant-{i}: {}",
                "y".repeat(200)
            )));
        }
    }
    fixtures
        .messages
        .set_draft(&thread_id(), messages, t0())
        .await
        .map_err(|e| anyhow::anyhow!("seed projection draft: {e}"))?;

    // High threshold so the pre-call check does NOT fire — only the
    // post-failure overflow path may compact.
    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(usize::MAX);

    let scripted = Arc::new(ScriptedProvider::new(vec![
        // The exact prose the Responses API returns on context overflow.
        ChatOutcome::InvalidRequest(
            "Your input exceeds the context window of this model. \
             Please adjust your input and try again."
                .into(),
        ),
        // Compactor's summarisation call.
        ok_response("[emergency summary]"),
        // Retry of the original turn after compaction.
        ok_response("Hello after recovery"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();

    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let outcome = execute_root_turn(
        inputs,
        "Tell me a joke",
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed turn after emergency compaction over opaque history");
    };
    assert_eq!(response_text, "Hello after recovery");
    assert_eq!(scripted.calls(), 3);

    // The rewritten projection no longer carries the summarized prefix's
    // opaque reasoning.
    let durable = fixtures.messages.get_history(&thread_id()).await?;
    let opaque_messages = durable
        .iter()
        .filter(|message| match &message.content {
            agent_sdk_foundation::llm::Content::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::OpaqueReasoning { .. })),
            agent_sdk_foundation::llm::Content::Text(_) => false,
        })
        .count();
    assert_eq!(
        opaque_messages, 0,
        "emergency compaction must drop the summarized prefix's opaque reasoning"
    );

    Ok(())
}

/// Negative case: when no `compaction_config` is wired, the provider
/// rejecting with `prompt is too long` must fail the turn fatally —
/// preserving the pre-PR behaviour for hosts that haven't opted in.
#[tokio::test]
async fn prompt_too_long_without_config_still_goes_fatal() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 4, "abc").await?;

    let provider = ScriptedProvider::new(vec![ChatOutcome::InvalidRequest(
        "prompt is too long: 1010596 tokens > 1000000 maximum".into(),
    )]);

    // Default deps — no compaction wired.
    let deps = RootTurnDeps {
        task_store: &fixtures.tasks,
        thread_store: &fixtures.threads,
        message_store: &fixtures.messages,
        attempt_store: &fixtures.attempts,
        checkpoint_store: &fixtures.checkpoints,
        event_repo: &fixtures.events,
        event_notifier: &fixtures.event_notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    let err = execute_root_turn(
        inputs,
        "ping",
        &provider,
        &deps,
        t0() + Duration::seconds(1),
    )
    .await
    .expect_err("expected fatal failure without compaction wired");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("prompt is too long"),
        "expected prompt-too-long surfaced as fatal error, got: {msg}",
    );
    // Exactly one provider call — no compaction-driven retry.
    assert_eq!(provider.calls(), 1);

    Ok(())
}

fn sample_envelope() -> ChatRequest {
    let definition = sample_definition();
    ChatRequest {
        system: definition.system_prompt,
        messages: Vec::new(),
        tools: None,
        max_tokens: definition.max_tokens,
        max_tokens_explicit: true,
        session_id: None,
        cached_content: None,
        thinking: None,
        tool_choice: None,
        response_format: None,
        cache: None,
    }
}

fn sample_envelope_fingerprint() -> Result<String> {
    super::compaction::request_envelope_fingerprint("mock", "mock-model", &sample_envelope())
}

/// Aggregate billing columns are set larger than any stamped
/// `request_input_tokens` so a leak of the aggregate into the trigger fails.
async fn seed_attempt_row(
    fixtures: &Fixtures,
    task_id: &crate::journal::task::AgentTaskId,
    attempt_number: u32,
    response_blob: serde_json::Value,
    aggregate_input_tokens: u32,
    opened_at: OffsetDateTime,
) -> Result<()> {
    use crate::journal::turn_attempt::{CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome};
    use crate::journal::turn_attempt_store::TurnAttemptStore;
    use agent_sdk_foundation::audit::AuditProvenance;

    let attempt = fixtures
        .attempts
        .open_attempt(OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number,
            provenance: AuditProvenance::new("mock", "mock-model"),
            request_blob: serde_json::json!({ "user_prompt": "seeded" }),
            now: opened_at,
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;
    fixtures
        .attempts
        .close_attempt(
            &attempt.id,
            CloseAttemptParams {
                response_blob,
                response_id: None,
                response_model: None,
                stop_reason: Some(StopReason::EndTurn),
                outcome: TurnAttemptOutcome::Success,
                input_tokens: aggregate_input_tokens,
                output_tokens: 10,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                route_provider: None,
                thinking_mode: None,
                thinking_budget_tokens: None,
                thinking_effort: None,
            },
            opened_at + Duration::seconds(1),
        )
        .await?;
    Ok(())
}

async fn seed_anchor_attempt(
    fixtures: &Fixtures,
    task_id: &crate::journal::task::AgentTaskId,
    attempt_number: u32,
    request_input_tokens: u64,
    request_message_count: u64,
    history_generation: u64,
    opened_at: OffsetDateTime,
) -> Result<()> {
    let blob = serde_json::json!({
        "model": "mock-model",
        "request_input_tokens": request_input_tokens,
        "request_message_count": request_message_count,
        "request_fingerprint": sample_envelope_fingerprint()?,
        "history_generation": history_generation,
    });
    let aggregate = u32::try_from(
        request_input_tokens
            .saturating_mul(2)
            .min(u64::from(u32::MAX)),
    )
    .unwrap_or(u32::MAX);
    seed_attempt_row(
        fixtures,
        task_id,
        attempt_number,
        blob,
        aggregate,
        opened_at,
    )
    .await
}

#[tokio::test]
async fn legacy_engine_flag_off_uses_estimated_trigger_and_generic_prompt() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        100,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!(
            "legacy-user-{index}: {}",
            "x".repeat(30_000)
        )));
        history.push(Message::assistant(format!(
            "legacy-assistant-{index}: {}",
            "y".repeat(30_000)
        )));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response("[legacy summary]")]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(scripted.calls(), 1);
    assert!(
        scripted
            .prompts()?
            .first()
            .is_some_and(|prompt| !prompt.contains("Compaction purpose:")),
        "flag-off fallback must retain the generic legacy prompt",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection")?;
    assert_eq!(projection.compactions.len(), 1);
    Ok(())
}

#[tokio::test]
async fn snapcompact_measured_trigger_wins_over_large_estimate() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        100,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!(
            "measured-user-{index}: {}",
            "x".repeat(30_000),
        )));
        history.push(Message::assistant(format!(
            "measured-assistant-{index}: {}",
            "y".repeat(30_000),
        )));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Snapcompact)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::image_capable(Vec::new()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(
        artifact_dir.path().join("measured-snapcompact"),
    ));
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(
        !outcome.completed && !outcome.applied,
        "Snapcompact must trust fresh measured usage instead of the larger estimate",
    );
    assert_eq!(scripted.calls(), 0);
    assert!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection after measured Snapcompact check")?
            .compactions
            .is_empty(),
    );
    Ok(())
}

#[tokio::test]
async fn stale_measured_usage_before_compaction_does_not_retrigger() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;

    // Billed Success at t+5, long before the compaction boundary at t+10.
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        90_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0())
        .await?;
    fixtures
        .messages
        .append_compaction(
            &thread_id(),
            history.clone(),
            history.len(),
            0,
            t0() + Duration::seconds(10),
        )
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(Vec::new()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        0,
        "a stale pre-compaction measurement must not buy a billed summarization",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection")?;
    assert_eq!(projection.compactions.len(), 1);
    Ok(())
}

/// History buffered AFTER the last billed attempt (wait-any child results,
/// steering injections) must count toward the proactive trigger: the anchor
/// alone reads stale-low and would defer to one wasted provider round-trip in
/// overflow recovery.
#[tokio::test]
async fn history_appended_after_measured_attempt_triggers_proactively() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;

    // Billed Success covering only the first 2 staged messages, well under
    // the threshold.
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        1_000,
        2,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = vec![Message::user("u-0"), Message::assistant("a-0")];
    for index in 0..22 {
        history.push(Message::user(format!(
            "buffered-child-result-{index}: {}",
            "y".repeat(30_000)
        )));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[summary] buffered child results folded",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        1,
        "freshly appended history must trigger proactive compaction, not overflow recovery",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection")?;
    assert_eq!(projection.compactions.len(), 1);
    Ok(())
}

#[tokio::test]
async fn cache_and_output_usage_fields_cannot_inflate_the_measured_trigger() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_attempt_row(
        &fixtures,
        &task.id,
        1,
        serde_json::json!({
            "model": "mock-model",
            "request_input_tokens": 500_000,
            "request_message_count": 24,
            "request_fingerprint": sample_envelope_fingerprint()?,
            "history_generation": 0,
        }),
        1_000_000,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(600_000);
    let scripted = Arc::new(ScriptedProvider::new(Vec::new()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(
        !outcome.completed && !outcome.applied,
        "500k measured input must not cross a 600k threshold; summing output \
         and cache subsets would have read 1.1M and fired",
    );
    assert_eq!(scripted.calls(), 0);
    assert!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection after inflated-usage check")?
            .compactions
            .is_empty(),
    );
    Ok(())
}

#[tokio::test]
async fn later_compaction_audit_attempt_is_ignored_for_the_anchor() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        90_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;
    seed_attempt_row(
        &fixtures,
        &task.id,
        2,
        serde_json::json!({
            "operation": "context_compaction",
            "applied": false,
            "request_input_tokens": 1_000,
            "request_message_count": 24,
            "request_fingerprint": sample_envelope_fingerprint()?,
            "history_generation": 0,
        }),
        1_000,
        t0() + Duration::seconds(10),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[summary] anchored on the main attempt",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        1,
        "the 90k main attempt must anchor the trigger; the later audit \
         attempt's 1k summarizer usage must be ignored",
    );
    Ok(())
}

#[tokio::test]
async fn single_response_appended_after_measured_attempt_adds_estimator_delta() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        40_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(25);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    history.push(Message::assistant(format!(
        "post-measure-response: {}",
        "z".repeat(80_000)
    )));
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[summary] appended response folded",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        1,
        "the uncovered response's estimator delta must lift 40k over the \
         50k threshold",
    );
    Ok(())
}

#[tokio::test]
async fn appended_suffix_counts_exactly_once() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        40_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(25);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    history.push(Message::assistant("z".repeat(80_000)));
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(70_000);
    let scripted = Arc::new(ScriptedProvider::new(Vec::new()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(
        !outcome.completed,
        "40k anchored + ~20k suffix stays under 70k; counting the suffix twice would fire",
    );
    assert_eq!(scripted.calls(), 0);
    Ok(())
}

#[tokio::test]
async fn success_close_stamps_the_measured_request_facts() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 2, "abc").await?;

    let provider = ScriptedProvider::new(vec![ok_response("done")]);
    let deps = RootTurnDeps {
        task_store: &fixtures.tasks,
        thread_store: &fixtures.threads,
        message_store: &fixtures.messages,
        attempt_store: &fixtures.attempts,
        checkpoint_store: &fixtures.checkpoints,
        event_repo: &fixtures.events,
        event_notifier: &fixtures.event_notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    execute_root_turn(
        inputs,
        "ping",
        &provider,
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let attempts = fixtures.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1);
    let blob = attempts[0]
        .response_blob
        .as_ref()
        .context("closed attempt has a response blob")?;
    assert_eq!(
        blob.get("request_message_count")
            .and_then(serde_json::Value::as_u64),
        Some(5),
        "close must stamp staged history (4) + fresh user message (1)",
    );
    assert_eq!(
        blob.get("request_input_tokens")
            .and_then(serde_json::Value::as_u64),
        Some(100),
        "close must stamp the provider's physical request total",
    );
    assert_eq!(
        blob.get("request_fingerprint")
            .and_then(serde_json::Value::as_str),
        Some(sample_envelope_fingerprint()?.as_str()),
        "close must stamp the dispatched envelope's fingerprint",
    );
    assert_eq!(
        blob.get("history_generation")
            .and_then(serde_json::Value::as_u64),
        Some(0),
        "close must stamp the staged source's history generation",
    );
    assert!(
        attempts[0]
            .request_blob
            .get("request_message_count")
            .is_none(),
        "no scalar may annotate the request blob at open time",
    );
    Ok(())
}

#[tokio::test]
async fn fresh_user_prompt_counts_toward_the_proactive_trigger() -> Result<()> {
    let fixtures = Fixtures::new();
    // Near-empty bodies would make compaction a no-progress skip.
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"q".repeat(200)).await?;

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    seed_anchor_attempt(
        &fixtures,
        &task_id,
        1,
        79_000,
        24,
        0,
        t0() - Duration::seconds(5),
    )
    .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(80_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![
        ok_response("[summary] prompt counted"),
        ok_response("main reply"),
    ]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);

    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    // ~5k estimated tokens: 79k anchored + 5k pending crosses 80k.
    let prompt = "p".repeat(20_000);
    execute_root_turn(
        inputs,
        &prompt,
        provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        2,
        "summarizer + main call — the prompt's estimate must fire the trigger proactively",
    );
    let requests = scripted.captured_requests()?;
    assert!(
        !requests[0].messages.iter().any(|message| {
            matches!(&message.content, Content::Text(text) if text.contains("ppppp"))
                || matches!(
                    &message.content,
                    Content::Blocks(blocks) if blocks.iter().any(|block| matches!(
                        block,
                        ContentBlock::Text { text } if text.contains("ppppp")
                    ))
                )
        }),
        "the fresh prompt must never be folded into the compaction summary source",
    );
    assert_eq!(
        fixtures
            .messages
            .get(&thread_id())
            .await?
            .context("projection")?
            .compactions
            .len(),
        1,
    );

    assert_compaction_row_precedes_main_row(&fixtures, &task_id).await?;
    assert_single_user_input_before_start(&fixtures).await
}

async fn assert_compaction_row_precedes_main_row(
    fixtures: &Fixtures,
    task_id: &crate::journal::task::AgentTaskId,
) -> Result<()> {
    let mut attempts = fixtures.attempts.list_by_task(task_id).await?;
    attempts.sort_by_key(|attempt| attempt.attempt_number);
    assert_eq!(attempts.len(), 3);
    assert_eq!(
        attempts[1]
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert_eq!(attempts[1].input_tokens, Some(100));
    assert_eq!(attempts[1].output_tokens, Some(50));
    let final_history_len = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after turn")?
        .context_history()
        .len();
    let main_blob = attempts[2]
        .response_blob
        .as_ref()
        .context("main attempt response blob")?;
    assert_eq!(
        main_blob
            .get("request_message_count")
            .and_then(serde_json::Value::as_u64),
        u64::try_from(final_history_len - 1).ok(),
        "post-compaction main request: rewritten staged history + fresh prompt",
    );
    assert_eq!(
        main_blob
            .get("history_generation")
            .and_then(serde_json::Value::as_u64),
        Some(1),
        "the main dispatch is stamped with the POST-compaction lineage",
    );

    Ok(())
}

async fn assert_single_user_input_before_start(fixtures: &Fixtures) -> Result<()> {
    let events = fixtures.events.get_events(&thread_id()).await?;
    let user_inputs: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, committed)| matches!(committed.event, AgentEvent::UserInput { .. }))
        .map(|(index, _)| index)
        .collect();
    let starts: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, committed)| matches!(committed.event, AgentEvent::Start { .. }))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(user_inputs.len(), 1, "exactly one UserInput event");
    assert_eq!(starts.len(), 1, "exactly one Start event");
    assert!(
        user_inputs[0] < starts[0],
        "UserInput precedes Start in the journal",
    );
    Ok(())
}

#[tokio::test]
async fn user_input_event_is_not_duplicated_on_task_retry() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 2, "abc").await?;

    let provider = ScriptedProvider::new(vec![ok_response("done")]);
    let deps = RootTurnDeps {
        task_store: &fixtures.tasks,
        thread_store: &fixtures.threads,
        message_store: &fixtures.messages,
        attempt_store: &fixtures.attempts,
        checkpoint_store: &fixtures.checkpoints,
        event_repo: &fixtures.events,
        event_notifier: &fixtures.event_notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };

    let first = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = first.id.clone();
    let prior = AgentEvent::user_input(
        thread_id(),
        vec![ContentBlock::Text {
            text: "ping".into(),
        }],
    )
    .with_emitter_task_id(task_id.as_str());
    fixtures
        .events
        .commit_event(&thread_id(), prior, t0())
        .await?;
    fixtures
        .tasks
        .release_expired_leases(t0() + Duration::seconds(301))
        .await?;
    let retried = fixtures
        .tasks
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_compaction"),
            LeaseId::from_string("lease_compaction"),
            t0() + Duration::seconds(900),
            t0() + Duration::seconds(302),
        )
        .await?
        .context("expired task should be re-acquirable")?;
    assert_eq!(retried.attempt, 2);

    let inputs = build_root_worker_inputs(
        sample_bootstrap(retried),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0() + Duration::seconds(302),
    )
    .await?;
    execute_root_turn(
        inputs,
        "ping",
        &provider,
        &deps,
        t0() + Duration::seconds(303),
    )
    .await?;

    let user_inputs = fixtures
        .events
        .get_events(&thread_id())
        .await?
        .iter()
        .filter(|committed| matches!(committed.event, AgentEvent::UserInput { .. }))
        .count();
    assert_eq!(
        user_inputs, 1,
        "retry must not duplicate the UserInput event"
    );
    Ok(())
}

#[tokio::test]
async fn compaction_and_main_spend_bill_on_distinct_truthful_rows() -> Result<()> {
    let fixtures = Fixtures::new();
    seed_projection_history(&fixtures.messages, &thread_id(), 12, &"q".repeat(200)).await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::Legacy)
        .with_threshold_tokens(10);
    let summarizer = Arc::new(ScriptedProvider::with_identity("openai", "gpt-summarizer"));
    let compaction_provider: Arc<dyn LlmProvider> = summarizer.clone();
    let main_scripted = Arc::new(ScriptedProvider::new(vec![ok_response("main response")]));
    let main_provider: Arc<dyn LlmProvider> = main_scripted.clone();
    let mut deps = fixtures.deps_with_compaction(&cfg, &main_provider);
    deps.compaction_provider = Some(&compaction_provider);

    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    let task_id = task.id.clone();
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &fixtures.threads,
        &fixtures.checkpoints,
        &fixtures.messages,
        t0(),
    )
    .await?;

    execute_root_turn(
        inputs,
        "main request",
        main_provider.as_ref(),
        &deps,
        t0() + Duration::seconds(1),
    )
    .await?;

    let mut attempts = fixtures.attempts.list_by_task(&task_id).await?;
    attempts.sort_by_key(|attempt| attempt.attempt_number);
    assert_eq!(attempts.len(), 2, "compaction row + main row");

    let compaction_row = &attempts[0];
    assert_eq!(
        compaction_row
            .response_blob
            .as_ref()
            .and_then(|blob| blob.get("operation"))
            .and_then(serde_json::Value::as_str),
        Some("context_compaction"),
    );
    assert_eq!(compaction_row.provider, "openai");
    assert_eq!(compaction_row.requested_model, "gpt-summarizer");
    assert_eq!(
        compaction_row.response_model.as_deref(),
        Some("mock-model"),
        "summarizer ChatResponse model rides the compaction close",
    );
    assert_eq!(compaction_row.input_tokens, Some(100));
    assert_eq!(compaction_row.output_tokens, Some(50));

    let main_row = &attempts[1];
    assert_eq!(main_row.provider, "mock");
    assert_eq!(main_row.requested_model, "mock-model");
    assert_eq!(main_row.input_tokens, Some(100));
    assert_eq!(main_row.output_tokens, Some(50));

    let total_input: u32 = attempts.iter().filter_map(|a| a.input_tokens).sum();
    let total_output: u32 = attempts.iter().filter_map(|a| a.output_tokens).sum();
    assert_eq!((total_input, total_output), (200, 100));
    Ok(())
}

#[tokio::test]
async fn compactor_budgets_follow_the_model_aware_threshold() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let read = |id: &str| Message {
        role: agent_sdk_foundation::llm::Role::Assistant,
        content: Content::Blocks(vec![ContentBlock::ToolUse {
            id: id.into(),
            name: "read".into(),
            input: serde_json::json!({ "path": "src/lib.rs" }),
            thought_signature: None,
        }]),
    };
    let history = vec![
        Message::user("start"),
        read("read_old"),
        Message::tool_result("read_old", "y".repeat(800_000), false),
        read("read_new"),
        Message::tool_result("read_new", "x".repeat(40_000), false),
        Message::assistant("done"),
    ];
    let fixtures = Fixtures::new();
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0())
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(1_000)
        .with_min_messages(1)
        .with_model_aware_threshold(true)
        .with_model_context_window_tokens(200_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[summary] must not be needed",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(1),
    )
    .await?;

    assert!(outcome.applied, "{outcome:?}");
    assert_eq!(
        scripted.calls(),
        0,
        "the pruned view fits the model-aware threshold, so no summary is needed",
    );
    Ok(())
}

#[tokio::test]
async fn unstable_route_uses_fixed_threshold_and_full_estimate() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    struct UnstableRouteProvider {
        inner: ScriptedProvider,
    }

    #[async_trait]
    impl LlmProvider for UnstableRouteProvider {
        async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
            self.inner.chat(request).await
        }

        fn model(&self) -> &'static str {
            "claude-fable-5"
        }

        fn provider(&self) -> &'static str {
            "anthropic"
        }

        fn has_stable_request_route(&self) -> bool {
            false
        }
    }

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        90_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}: {}", "x".repeat(30_000))));
        history.push(Message::assistant(format!(
            "a-{index}: {}",
            "y".repeat(30_000)
        )));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_model_aware_threshold(true)
        .with_threshold_tokens(50_000);
    let unstable = Arc::new(UnstableRouteProvider {
        inner: ScriptedProvider::new(vec![ok_response("[summary] unstable route")]),
    });
    let provider: Arc<dyn LlmProvider> = unstable.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        unstable.inner.calls(),
        1,
        "fixed threshold + full estimate must fire; honoring the 1M \
         representative window or the 90k anchor would have skipped",
    );
    Ok(())
}

#[tokio::test]
async fn post_compaction_main_attempt_anchors_the_next_turn() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0())
        .await?;
    fixtures
        .messages
        .append_compaction(
            &thread_id(),
            history.clone(),
            history.len(),
            0,
            t0() + Duration::seconds(10),
        )
        .await?;

    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        500_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        2,
        9_000,
        24,
        1,
        t0() + Duration::seconds(10),
    )
    .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(Vec::new()));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        0,
        "the generation-1 attempt (9k) must anchor — rejecting it (or \
         anchoring the stale 500k generation-0 row) re-triggers a billed \
         summarization on an already-compacted history",
    );
    Ok(())
}

#[tokio::test]
async fn anchor_selection_is_causal_by_covered_prefix() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let task = create_and_acquire_root_task(&fixtures.tasks, &thread_id()).await?;

    seed_anchor_attempt(
        &fixtures,
        &task.id,
        1,
        1_000,
        2,
        0,
        t0() + Duration::seconds(15),
    )
    .await?;
    seed_anchor_attempt(
        &fixtures,
        &task.id,
        2,
        90_000,
        24,
        0,
        t0() + Duration::seconds(5),
    )
    .await?;

    let mut history = Vec::with_capacity(24);
    for index in 0..12 {
        history.push(Message::user(format!("u-{index}")));
        history.push(Message::assistant(format!("a-{index}")));
    }
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0() + Duration::seconds(6))
        .await?;

    let cfg = CompactionConfig::default()
        .with_engine(CompactionEngine::PruneFirst)
        .with_threshold_tokens(50_000);
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[summary] causal anchor",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let deps = fixtures.deps_with_compaction(&cfg, &provider);
    let staged = StagedMessageStore::new(thread_id(), history);

    super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert_eq!(
        scripted.calls(),
        1,
        "the count-24/90k attempt is the causal latest and must anchor; \
         picking the later-opened count-2/1k row would silently skip",
    );
    Ok(())
}

fn event_kind(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::Start { .. } => "start",
        AgentEvent::Text { .. } => "text",
        AgentEvent::TextDelta { .. } => "text_delta",
        AgentEvent::Thinking { .. } => "thinking",
        AgentEvent::ThinkingDelta { .. } => "thinking_delta",
        AgentEvent::ToolCallStart { .. } => "tool_call_start",
        AgentEvent::ToolCallEnd { .. } => "tool_call_end",
        AgentEvent::TurnComplete { .. } => "turn_complete",
        AgentEvent::Done { .. } => "done",
        AgentEvent::Error { .. } => "error",
        AgentEvent::ContextCompacted { .. } => "context_compacted",
        AgentEvent::AutoRetryStart { .. } => "auto_retry_start",
        AgentEvent::AutoRetryEnd { .. } => "auto_retry_end",
        _ => "other",
    }
}

const RECLAIM_TEST_PNG: &[u8] = b"\x89PNG\r\n\x1a\nreclaim-regression-bytes";

fn published_artifact_count(store: &agent_sdk::ArtifactStore) -> Result<usize> {
    let mut count = 0;
    for entry in std::fs::read_dir(store.dir())? {
        let file_name = entry?.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.ends_with(".log")
            && file_name.as_bytes().first().is_some_and(u8::is_ascii_digit)
        {
            count += 1;
        }
    }
    Ok(count)
}

/// Snapcompact publishes exactly one attachment artifact for this history;
/// the fixture providers lack historical-image support, so the summary
/// comes from the LLM.
fn attachment_bearing_history(marker: &str) -> Vec<Message> {
    use base64::Engine as _;

    let mut history = vec![Message::user_with_content(vec![
        ContentBlock::Text {
            text: format!("{marker}\n{}", "x".repeat(120_000)),
        },
        ContentBlock::Image {
            source: agent_sdk_foundation::llm::ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(RECLAIM_TEST_PNG),
            ),
        },
    ])];
    for index in 0..3 {
        history.push(Message::assistant(format!("{marker}-assistant-{index}")));
        history.push(Message::user(format!("{marker}-user-{index}")));
    }
    history.push(Message::assistant(format!("{marker}-final")));
    history
}

fn reclaim_test_config() -> CompactionConfig {
    CompactionConfig::default()
        .with_engine(CompactionEngine::Snapcompact)
        .with_threshold_tokens(10_000)
        .with_min_messages(1)
        .with_retain_recent(2)
}

/// Cancels from inside the summarization call, i.e. after the attachment
/// batch was published.
struct CancelAfterPublishProvider {
    cancel: CancellationToken,
    artifact_dir: std::path::PathBuf,
    artifacts_at_call: AtomicUsize,
    call_count: AtomicUsize,
}

impl CancelAfterPublishProvider {
    fn new(cancel: CancellationToken, artifact_dir: std::path::PathBuf) -> Self {
        Self {
            cancel,
            artifact_dir,
            artifacts_at_call: AtomicUsize::new(0),
            call_count: AtomicUsize::new(0),
        }
    }

    fn count_artifacts(&self) -> usize {
        std::fs::read_dir(&self.artifact_dir).map_or(0, |entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|entry| {
                    let file_name = entry.file_name();
                    let file_name = file_name.to_string_lossy();
                    file_name.ends_with(".log")
                        && file_name.as_bytes().first().is_some_and(u8::is_ascii_digit)
                })
                .count()
        })
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for CancelAfterPublishProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.artifacts_at_call
            .store(self.count_artifacts(), Ordering::SeqCst);
        self.cancel.cancel();
        Ok(ok_response("[summary after publish, before apply]"))
    }

    fn model(&self) -> &'static str {
        "cancel-after-publish-model"
    }

    fn provider(&self) -> &'static str {
        "cancel-after-publish"
    }
}

#[tokio::test]
async fn cancellation_after_compactor_return_reclaims_published_artifacts() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let history = attachment_bearing_history("cancel-reclaim");
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0())
        .await?;

    let cfg = reclaim_test_config();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(artifact_dir.path()));
    let cancel = CancellationToken::new();
    let scripted = Arc::new(CancelAfterPublishProvider::new(
        cancel.clone(),
        artifact_store.dir().to_path_buf(),
    ));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);
    deps.cancel = Some(&cancel);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(
        !outcome.applied,
        "a cancelled-after-return compaction must not apply"
    );
    assert_eq!(scripted.calls(), 1, "the summarizer must have been called");
    assert!(
        scripted.artifacts_at_call.load(Ordering::SeqCst) > 0,
        "the attachment batch must have been published before the cancellation",
    );
    assert_eq!(
        published_artifact_count(&artifact_store)?,
        0,
        "cancellation after the compactor returned must reclaim the published batch",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after cancelled compaction")?;
    assert!(
        projection.compactions.is_empty(),
        "no durable compaction entry may exist"
    );
    Ok(())
}

#[tokio::test]
async fn tool_sequence_refusal_reclaims_published_artifacts() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let staged_history = attachment_bearing_history("refusal-reclaim");

    // Same message count, but the retained boundary cuts a
    // tool_use/tool_result pair, so `append_compaction` refuses.
    let mut projection_history = staged_history.clone();
    projection_history[5] = Message {
        role: agent_sdk_foundation::llm::Role::Assistant,
        content: Content::Blocks(vec![ContentBlock::ToolUse {
            id: "call_cut".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
            thought_signature: None,
        }]),
    };
    projection_history[6] = Message {
        role: agent_sdk_foundation::llm::Role::User,
        content: Content::Blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "call_cut".into(),
            content: "ok".into(),
            artifact: None,
            is_error: None,
        }]),
    };
    fixtures
        .messages
        .commit_messages(&thread_id(), projection_history, t0())
        .await?;

    let cfg = reclaim_test_config();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(artifact_dir.path()));
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[refused summary]",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);
    let staged = StagedMessageStore::new(thread_id(), staged_history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(
        !outcome.applied,
        "a refused compaction must be skipped, not applied"
    );
    assert_eq!(scripted.calls(), 1, "the summarizer must have been called");
    assert_eq!(
        published_artifact_count(&artifact_store)?,
        0,
        "a tool-sequence refusal must reclaim the published batch",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after refused compaction")?;
    assert!(
        projection.compactions.is_empty(),
        "the refused append must leave no compaction entry"
    );
    Ok(())
}

#[tokio::test]
async fn append_failure_reclaims_published_artifacts() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let staged_history = attachment_bearing_history("append-failure-reclaim");

    // Fewer durable messages than staged: `CompactionSourceMismatch`.
    fixtures
        .messages
        .commit_messages(
            &thread_id(),
            vec![Message::user("short"), Message::assistant("projection")],
            t0(),
        )
        .await?;

    let cfg = reclaim_test_config();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(artifact_dir.path()));
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[appendless summary]",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);
    let staged = StagedMessageStore::new(thread_id(), staged_history);

    let error = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await
    .err()
    .context("a failed durable append must fail the compaction")?;
    assert!(
        format!("{error:?}").contains("append durable compaction entry"),
        "unexpected failure shape: {error:?}",
    );
    assert_eq!(scripted.calls(), 1, "the summarizer must have been called");
    assert_eq!(
        published_artifact_count(&artifact_store)?,
        0,
        "an append failure must reclaim the published batch",
    );
    Ok(())
}

#[tokio::test]
async fn applied_compaction_retains_published_artifacts() -> Result<()> {
    use crate::journal::staged::StagedMessageStore;

    let fixtures = Fixtures::new();
    let history = attachment_bearing_history("retain-on-apply");
    fixtures
        .messages
        .commit_messages(&thread_id(), history.clone(), t0())
        .await?;

    let cfg = reclaim_test_config();
    let artifact_dir = tempfile::tempdir()?;
    let artifact_store = Arc::new(agent_sdk::ArtifactStore::new(artifact_dir.path()));
    let scripted = Arc::new(ScriptedProvider::new(vec![ok_response(
        "[applied summary]",
    )]));
    let provider: Arc<dyn LlmProvider> = scripted.clone();
    let mut deps = fixtures.deps_with_compaction(&cfg, &provider);
    deps.compaction_artifact_store = Some(&artifact_store);
    let staged = StagedMessageStore::new(thread_id(), history);

    let outcome = super::compaction::maybe_compact_staged_history(
        &deps,
        provider.as_ref(),
        &staged,
        &thread_id(),
        0,
        &sample_envelope(),
        t0() + Duration::seconds(20),
    )
    .await?;

    assert!(outcome.applied, "the healthy compaction must apply");
    assert!(
        published_artifact_count(&artifact_store)? > 0,
        "the applied compaction's published artifacts must be retained",
    );
    let projection = fixtures
        .messages
        .get(&thread_id())
        .await?
        .context("projection after applied compaction")?;
    assert_eq!(
        projection.compactions.len(),
        1,
        "exactly one durable compaction entry must exist"
    );
    Ok(())
}
