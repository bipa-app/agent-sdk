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

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_sdk::context::CompactionConfig;
use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Message, StopReason, Usage,
};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use time::Duration;
use time::OffsetDateTime;

use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
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
}

impl ScriptedProvider {
    fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: Mutex::new(responses),
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn chat(&self, _: ChatRequest) -> Result<ChatOutcome> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut responses = self
            .responses
            .lock()
            .map_err(|_| anyhow::anyhow!("ScriptedProvider mutex poisoned"))?;
        if responses.is_empty() {
            anyhow::bail!("ScriptedProvider response queue exhausted");
        }
        Ok(responses.remove(0))
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
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
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
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
        tools_fn: None,
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
            cancel: None,
            wakeup: None,
            activity: None,
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
    let cfg = CompactionConfig::default().with_threshold_tokens(10);

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

    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed turn");
    };
    assert_eq!(response_text, "Hello after compaction");

    // Provider was called twice: once by the compactor, once by the
    // turn. The compactor consumed the summarisation slot first.
    assert_eq!(scripted.calls(), 2);

    // The durable projection got rewritten by the pre-call compaction
    // (1 summary + 0..N retained recent messages). The exact summary
    // length depends on the compactor's retain logic, so we only
    // assert the *committed* history shrank from the seeded 24-message
    // shape. The fresh turn's user prompt + assistant reply land in
    // `draft_messages` rather than the committed projection because
    // this fixture's thread never advances `committed_turns` (no
    // checkpoint is created until a real `commit_completed_turn`),
    // which is the path under exercise here.
    let durable = fixtures.messages.get_history(&thread_id()).await?;
    assert!(
        durable.len() < 24,
        "expected committed projection to shrink after compaction, found {} messages",
        durable.len(),
    );

    // A `ContextCompacted` event was committed so subscribers (TUI,
    // desktop) can render the compaction in their transcripts.
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
    let cfg = CompactionConfig::default().with_threshold_tokens(usize::MAX);

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
        panic!("expected Completed turn after emergency compaction");
    };
    assert_eq!(response_text, "Hello after recovery");

    // 3 calls: rejection, compactor, retry.
    assert_eq!(scripted.calls(), 3);

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
        cancel: None,
        wakeup: None,
        activity: None,
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
