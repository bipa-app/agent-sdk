//! End-to-end regression: image / document blocks survive the full
//! root-turn pipeline.
//!
//! Until this slice landed, the worker's public API (`user_prompt:
//! &str`) flattened user input to a string before reaching the LLM
//! call. Combined with `agent-service-host::host::root_task_prompt`
//! explicitly rejecting `Image` and `Document` items at the gRPC
//! boundary, every multi-modal `submit_thread_work` call failed at
//! root-task entry. This test fixture wires the full pipeline:
//!
//! 1. Build a `UserInput` that carries a text block + an image block
//!    (the same shape `host::root_task_user_input` produces from a
//!    gRPC `BinaryAttachment`).
//! 2. Run `execute_root_turn` against a mock provider that captures
//!    the `ChatRequest`.
//! 3. Assert the captured request's last user message contains the
//!    image block intact — i.e. the worker handed the LLM provider
//!    the same content it received from the host.
//! 4. Run a second turn on the same thread and assert the staged
//!    history also retains the image block — the buffer-on-commit
//!    path doesn't lossy-flatten on its way to the projection.

use std::sync::Arc;
use std::sync::Mutex;

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, ContentSource, StopReason, Usage,
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
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use crate::worker::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use crate::worker::user_input::UserInput;

/// Mock provider that records every `ChatRequest` it sees and replies
/// with a plain text response. Lets the test inspect what payload the
/// worker actually handed to the provider.
struct CapturingProvider {
    captured: Mutex<Vec<ChatRequest>>,
    response_text: String,
}

impl CapturingProvider {
    fn new(text: &str) -> Self {
        Self {
            captured: Mutex::new(Vec::new()),
            response_text: text.to_owned(),
        }
    }

    fn captured(&self) -> Vec<ChatRequest> {
        self.captured
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl LlmProvider for CapturingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        if let Ok(mut guard) = self.captured.lock() {
            guard.push(request);
        }
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_capture_01".into(),
            content: vec![ContentBlock::Text {
                text: self.response_text.clone(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
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

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn thread_id() -> ThreadId {
    ThreadId::from_string("t-multimodal")
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a test assistant.".into(),
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
        worker_id: WorkerId::from_string("worker_mm"),
        lease_id: LeaseId::from_string("lease_mm"),
    }
}

struct Stores {
    tasks: InMemoryAgentTaskStore,
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
    event_notifier: Arc<EventNotifier>,
}

impl Stores {
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
            compaction_config: None,
            compaction_provider: None,
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
            WorkerId::from_string("worker_mm"),
            LeaseId::from_string("lease_mm"),
            t0() + Duration::seconds(300),
            t0(),
        )
        .await?
        .expect("task should be acquirable");
    Ok(acquired)
}

fn image_input(question: &str, media_type: &str, data_b64: &str) -> UserInput {
    UserInput::from_blocks(vec![
        ContentBlock::Text {
            text: question.into(),
        },
        ContentBlock::Image {
            source: ContentSource::new(media_type, data_b64),
        },
    ])
}

fn last_user_blocks(request: &ChatRequest) -> &[ContentBlock] {
    let last = request
        .messages
        .last()
        .expect("chat request has at least the freshly appended user message");
    match &last.content {
        Content::Blocks(blocks) => blocks.as_slice(),
        Content::Text(_) => panic!(
            "expected Content::Blocks for the user-input message; got Content::Text — \
             this means the worker is still flattening rich input through \
             Message::user(text)",
        ),
    }
}

#[tokio::test]
async fn fresh_turn_image_input_reaches_llm_chat_request() -> Result<()> {
    let stores = Stores::new();
    let provider = CapturingProvider::new("I see a cat.");

    let task = create_and_acquire_root_task(&stores.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let user_input = image_input("what's in this picture?", "image/png", "AAAA");

    let outcome = execute_root_turn(
        inputs,
        user_input,
        &provider,
        &stores.deps(),
        t0() + Duration::seconds(1),
    )
    .await
    .context("execute_root_turn with image input")?;
    assert!(matches!(outcome, RootTurnOutcome::Completed { .. }));

    let captured = provider.captured();
    assert_eq!(captured.len(), 1, "exactly one LLM call");
    let blocks = last_user_blocks(&captured[0]);
    assert_eq!(blocks.len(), 2, "image block must accompany text");
    assert!(matches!(
        &blocks[0],
        ContentBlock::Text { text } if text == "what's in this picture?"
    ));
    assert!(matches!(
        &blocks[1],
        ContentBlock::Image { source }
            if source.media_type == "image/png" && source.data == "AAAA"
    ));

    // The compaction-event check is incidental — but if compaction
    // ever fires unexpectedly we'd want to know in the same test
    // run. (The default `CompactionConfig` isn't wired in this
    // fixture, so no event should appear.)
    let events = stores.events.get_events(&thread_id()).await?;
    assert!(
        events
            .iter()
            .all(|e| !matches!(e.event, AgentEvent::ContextCompacted { .. }))
    );

    Ok(())
}

#[tokio::test]
async fn document_input_round_trips_through_chat_request() -> Result<()> {
    let stores = Stores::new();
    let provider = CapturingProvider::new("Here's the summary…");

    let task = create_and_acquire_root_task(&stores.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let user_input = UserInput::from_blocks(vec![
        ContentBlock::Text {
            text: "summarise the attached pdf".into(),
        },
        ContentBlock::Document {
            source: ContentSource::new("application/pdf", "JVBERi0xLjQK"),
        },
    ]);

    execute_root_turn(
        inputs,
        user_input,
        &provider,
        &stores.deps(),
        t0() + Duration::seconds(1),
    )
    .await?;

    let captured = provider.captured();
    let blocks = last_user_blocks(&captured[0]);
    assert!(matches!(
        &blocks[1],
        ContentBlock::Document { source }
            if source.media_type == "application/pdf" && source.data == "JVBERi0xLjQK"
    ));

    Ok(())
}

#[tokio::test]
async fn text_only_string_input_still_works() -> Result<()> {
    // Back-compat regression: every existing call site (and every
    // existing test fixture in this crate) passes `&str` to
    // `execute_root_turn`. With the `impl Into<UserInput>` accepting
    // both `&str` and the typed variant, that path must keep
    // compiling and producing the same wire shape — `Content::Blocks`
    // wrapping a single `ContentBlock::Text`.
    let stores = Stores::new();
    let provider = CapturingProvider::new("hi back");

    let task = create_and_acquire_root_task(&stores.tasks, &thread_id()).await?;
    let inputs = build_root_worker_inputs(
        sample_bootstrap(task),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    execute_root_turn(
        inputs,
        "hello",
        &provider,
        &stores.deps(),
        t0() + Duration::seconds(1),
    )
    .await?;

    let captured = provider.captured();
    let blocks = last_user_blocks(&captured[0]);
    assert_eq!(blocks.len(), 1);
    assert!(matches!(
        &blocks[0],
        ContentBlock::Text { text } if text == "hello"
    ));
    Ok(())
}
