//! ENG-9510 integration coverage: the durable worker publishes a live
//! context estimate at LLM dispatch — before the provider yields any
//! frame — and re-anchors it on billed usage when the call settles.
//!
//! The estimator's state-machine semantics (anchor / tail / pending
//! relative-error bounds) live in [`crate::context_estimate`]'s unit
//! tests; this file proves the worker actually drives the registry at
//! the right lifecycle points during a real `execute_root_turn`.

use std::sync::{Arc, Mutex};

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest};
use agent_sdk_providers::LlmProvider;
use agent_sdk_providers::streaming::StreamBox;
use anyhow::{Context, Result};
use async_trait::async_trait;
use time::Duration;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::test_support::{StreamingScriptedProvider, TurnScript};
use crate::context_estimate::{self, EstimatePhase, LiveContextEstimate};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::InMemoryEventRepository;
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

/// Wraps the scripted provider and snapshots the registry's reading
/// for the thread at the exact moment `chat_stream` is entered — i.e.
/// after dispatch bookkeeping, before any streamed frame exists.
struct SnoopingProvider {
    inner: StreamingScriptedProvider,
    thread_id: ThreadId,
    at_dispatch: Arc<Mutex<Option<LiveContextEstimate>>>,
}

#[async_trait]
impl LlmProvider for SnoopingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.inner.chat(request).await
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        if let Ok(mut slot) = self.at_dispatch.lock() {
            *slot = context_estimate::live().live_estimate(&self.thread_id);
        }
        self.inner.chat_stream(request)
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &'static str {
        self.inner.provider()
    }
}

#[tokio::test]
async fn worker_publishes_estimate_at_dispatch_and_anchors_on_settle() -> Result<()> {
    // Unique thread id: the registry under test is process-global.
    let thread_id = ThreadId::from_string("t-ctx-estimate-worker");

    let tasks = InMemoryAgentTaskStore::new();
    let threads = InMemoryThreadStore::new();
    let messages = InMemoryMessageProjectionStore::new();
    let attempts = InMemoryTurnAttemptStore::new();
    let checkpoints = InMemoryCheckpointStore::new();
    let events = InMemoryEventRepository::new();
    let event_notifier = Arc::new(EventNotifier::new());

    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();
    tasks.submit_root_turn(task).await.context("submit")?;
    let acquired = tasks
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_ctx_est"),
            LeaseId::from_string("lease_ctx_est"),
            t0() + Duration::seconds(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("task should be acquirable")?;

    let definition = AgentDefinition {
        provider: "anthropic".into(),
        model: "mock-model".into(),
        system_prompt: "You are a test assistant.".into(),
        tools: Vec::new(),
        max_tokens: 1024,
        thinking: ThinkingPolicy::Disabled,
        thinking_display: None,
        tools_fn: None,
        policy: RuntimePolicy::default(),
    };
    let bootstrap = WorkerBootstrapContext {
        thread_id: acquired.thread_id.clone(),
        task_id: acquired.id.clone(),
        task: acquired,
        definition,
        worker_id: WorkerId::from_string("worker_ctx_est"),
        lease_id: LeaseId::from_string("lease_ctx_est"),
    };
    let inputs = build_root_worker_inputs(bootstrap, &threads, &checkpoints, &messages, t0())
        .await
        .context("build inputs")?;

    let at_dispatch = Arc::new(Mutex::new(None));
    let provider = SnoopingProvider {
        inner: StreamingScriptedProvider::single(TurnScript::text("hello")),
        thread_id: thread_id.clone(),
        at_dispatch: Arc::clone(&at_dispatch),
    };

    let deps = RootTurnDeps {
        task_store: &tasks,
        thread_store: &threads,
        message_store: &messages,
        attempt_store: &attempts,
        checkpoint_store: &checkpoints,
        event_repo: &events,
        event_notifier: &event_notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };

    let outcome = execute_root_turn(inputs, "hi", &provider, &deps, t0()).await?;
    assert!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "scripted text turn should complete",
    );

    // Layer 3 (pending snapshot): the registry answered with an
    // in-flight estimate BEFORE the provider produced a single frame.
    let dispatch_reading = at_dispatch
        .lock()
        .map_err(|_| anyhow::anyhow!("snoop lock poisoned"))?
        .context("chat_stream must observe a published reading")?;
    assert_eq!(dispatch_reading.phase, EstimatePhase::InFlight);
    assert!(
        dispatch_reading.prompt_tokens > 0,
        "system prompt + user prompt must estimate above zero",
    );

    // Layer 1 (anchor): after the call settled, the reading converges
    // to the scripted provider's billed usage (DEFAULT_USAGE input =
    // 100 — exact, zero relative error at the turn boundary).
    let settled = context_estimate::live()
        .live_estimate(&thread_id)
        .context("anchored reading after settle")?;
    assert_eq!(settled.phase, EstimatePhase::Anchored);
    assert_eq!(settled.prompt_tokens, 100);

    context_estimate::live().forget(&thread_id);
    Ok(())
}
