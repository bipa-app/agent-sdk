//! Phase 6.5 replay coverage tests.
//!
//! Verifies that the full public event surface — root-turn content
//! events (`Start`, `Thinking`, `Text`), tool-runtime progress events
//! (`ToolProgress`), and lifecycle edges — is durably committed and
//! replays in the correct order across interleaved root and tool
//! task activity on the same thread.
use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn, resume_from_children};
use super::tool_task::{ToolTaskOutcome, execute_tool_task, resolve_tool_bootstrap};
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
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use time::Duration;
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn thread_replay() -> ThreadId {
    ThreadId::from_string("t-replay-coverage")
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "anthropic".into(),
        model: "mock-model".into(),
        system_prompt: "You are a test assistant.".into(),
        tools: Vec::new(),
        max_tokens: 1024,
        thinking: ThinkingPolicy::Disabled,
        tools_fn: None,
        policy: RuntimePolicy::default(),
    }
}

fn definition_with_tools() -> AgentDefinition {
    AgentDefinition {
        tools: vec![Tool {
            name: "bash".into(),
            description: "Run a shell command".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            display_name: "Bash".into(),
            tier: ToolTier::Observe,
        }],
        ..sample_definition()
    }
}

fn definition_with_thinking() -> AgentDefinition {
    AgentDefinition {
        thinking: ThinkingPolicy::Enabled {
            budget_tokens: 2048,
        },
        ..sample_definition()
    }
}

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

fn bootstrap(task: AgentTask, definition: AgentDefinition) -> WorkerBootstrapContext {
    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition,
        thread_id,
        task_id,
        worker_id: WorkerId::from_string("worker_replay"),
        lease_id: LeaseId::from_string("lease_replay"),
    }
}

async fn create_and_acquire(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await.context("submit")?;
    let acquired = store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_replay"),
            LeaseId::from_string("lease_replay"),
            t_plus(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("task should be acquirable")?;
    Ok(acquired)
}

// ─────────────────────────────────────────────────────────────────────
// Mock providers
// ─────────────────────────────────────────────────────────────────────

use agent_sdk_providers::streaming::{StreamBox, StreamDelta};

const fn replay_usage() -> Usage {
    Usage {
        input_tokens: 200,
        output_tokens: 100,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
    }
}

/// Stream a `thinking` block (with signature) on block 0 followed by a
/// `text` block on block 1, then `Usage` + `Done{EndTurn}`.  Empty
/// `thinking` / `text` are skipped so callers can request text-only or
/// tool-only shapes.  This mirrors what a real provider's SSE decoder
/// emits and exercises the per-delta journal-commit path.
fn thinking_text_stream(thinking: String, signature: String, text: String) -> StreamBox<'static> {
    Box::pin(async_stream::stream! {
        if !thinking.is_empty() {
            yield Ok(StreamDelta::ThinkingDelta { delta: thinking, block_index: 0 });
            yield Ok(StreamDelta::SignatureDelta { delta: signature, block_index: 0 });
        }
        if !text.is_empty() {
            yield Ok(StreamDelta::TextDelta { delta: text, block_index: 1 });
        }
        yield Ok(StreamDelta::Usage(replay_usage()));
        yield Ok(StreamDelta::Done { stop_reason: Some(StopReason::EndTurn) });
    })
}

struct ThinkingTextProvider {
    thinking: String,
    response_text: String,
}

#[async_trait]
impl LlmProvider for ThinkingTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_thinking".into(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: self.thinking.clone(),
                    signature: Some("sig_test".into()),
                },
                ContentBlock::Text {
                    text: self.response_text.clone(),
                },
            ],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: replay_usage(),
        }))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        thinking_text_stream(
            self.thinking.clone(),
            "sig_test".to_owned(),
            self.response_text.clone(),
        )
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

struct ThinkingToolCallProvider {
    thinking: String,
    tool_calls: Vec<(String, String, serde_json::Value)>,
    call_count: AtomicUsize,
    resume_thinking: String,
    resume_text: String,
}

#[async_trait]
impl LlmProvider for ThinkingToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
        if call_num > 0 {
            // Resume call returns thinking + text.
            return Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_resume".into(),
                content: vec![
                    ContentBlock::Thinking {
                        thinking: self.resume_thinking.clone(),
                        signature: Some("sig_resume".into()),
                    },
                    ContentBlock::Text {
                        text: self.resume_text.clone(),
                    },
                ],
                model: "mock-model".into(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 50,
                    output_tokens: 25,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }));
        }
        // First call returns thinking + tool calls.
        let mut content = vec![ContentBlock::Thinking {
            thinking: self.thinking.clone(),
            signature: Some("sig_first".into()),
        }];
        for (id, name, input) in &self.tool_calls {
            content.push(ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                thought_signature: None,
            });
        }
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_tool".into(),
            content,
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: replay_usage(),
        }))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
        if call_num > 0 {
            // Resume turn: thinking (optional) + text.
            return thinking_text_stream(
                self.resume_thinking.clone(),
                "sig_resume".to_owned(),
                self.resume_text.clone(),
            );
        }
        // First turn: optional thinking on block 0, then a tool call per
        // subsequent block, then `Usage` + `Done{ToolUse}`.
        let thinking = self.thinking.clone();
        let tool_calls = self.tool_calls.clone();
        Box::pin(async_stream::stream! {
            // Optional thinking occupies block 0; tool calls then start
            // at block 1.  With no thinking they start at block 0.
            let has_thinking = !thinking.is_empty();
            let first_tool_block = usize::from(has_thinking);
            if has_thinking {
                yield Ok(StreamDelta::ThinkingDelta { delta: thinking, block_index: 0 });
                yield Ok(StreamDelta::SignatureDelta {
                    delta: "sig_first".to_owned(),
                    block_index: 0,
                });
            }
            for (offset, (id, name, input)) in tool_calls.into_iter().enumerate() {
                let block_index = first_tool_block + offset;
                yield Ok(StreamDelta::ToolUseStart {
                    id: id.clone(),
                    name,
                    block_index,
                    thought_signature: None,
                });
                yield Ok(StreamDelta::ToolInputDelta {
                    id,
                    delta: input.to_string(),
                    block_index,
                });
            }
            yield Ok(StreamDelta::Usage(replay_usage()));
            yield Ok(StreamDelta::Done { stop_reason: Some(StopReason::ToolUse) });
        })
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

/// Streamed text-with-thinking turn.  Post-streaming-refactor journal
/// order: `UserInput → Start → ThinkingDelta → Thinking → TextDelta →
/// Text → TurnComplete → Done`.  The streamed `ThinkingDelta` /
/// `TextDelta` precede their consolidated `Thinking` / `Text` blocks.
#[tokio::test]
async fn text_only_with_thinking_replays_in_order() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_replay()).await?;
    let ctx = bootstrap(task, definition_with_thinking());
    let inputs = build_root_worker_inputs(
        ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let provider = ThinkingTextProvider {
        thinking: "Let me think about this...".into(),
        response_text: "Here is my answer".into(),
    };
    execute_root_turn(inputs, "question", &provider, &stores.deps(), t0()).await?;

    // The turn streams both deltas (thinking then text) before
    // committing both consolidated blocks at turn close:
    //   UserInput → Start → ThinkingDelta → TextDelta
    //            → Thinking → Text → TurnComplete → Done
    let events = stores.events.get_events(&thread_replay()).await?;
    assert_eq!(
        events.len(),
        8,
        "UserInput + Start + ThinkingDelta + TextDelta + Thinking + Text + TurnComplete + Done"
    );

    assert!(matches!(&events[0].event, AgentEvent::UserInput { .. }));
    assert!(matches!(&events[1].event, AgentEvent::Start { .. }));
    assert!(matches!(&events[2].event, AgentEvent::ThinkingDelta { .. }));
    assert!(matches!(&events[3].event, AgentEvent::TextDelta { .. }));
    assert!(matches!(&events[4].event, AgentEvent::Thinking { .. }));
    assert!(matches!(&events[5].event, AgentEvent::Text { .. }));
    assert!(matches!(&events[6].event, AgentEvent::TurnComplete { .. }));
    assert!(matches!(&events[7].event, AgentEvent::Done { .. }));

    // Verify content fidelity of the consolidated blocks.
    if let AgentEvent::Thinking { text, .. } = &events[4].event {
        assert_eq!(text, "Let me think about this...");
    } else {
        panic!("expected Thinking");
    }
    if let AgentEvent::Text { text, .. } = &events[5].event {
        assert_eq!(text, "Here is my answer");
    } else {
        panic!("expected Text");
    }

    // Sequences are contiguous.
    for (i, evt) in events.iter().enumerate() {
        assert_eq!(evt.sequence, i as u64, "sequence gap at index {i}");
    }

    Ok(())
}

/// Tool progress events emitted by the executor are durably committed
/// between `ToolCallStart` and `ToolCallEnd`.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn tool_progress_events_are_durable() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_replay()).await?;
    let ctx = bootstrap(task, definition_with_tools());
    let inputs = build_root_worker_inputs(
        ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Suspend to create a child tool task.
    let provider = ThinkingToolCallProvider {
        thinking: String::new(),
        tool_calls: vec![(
            "tc_prog".into(),
            "bash".into(),
            serde_json::json!({"command": "echo progress"}),
        )],
        call_count: AtomicUsize::new(0),
        resume_thinking: String::new(),
        resume_text: "done".into(),
    };
    let outcome =
        execute_root_turn(inputs, "run with progress", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended"),
    };

    // Execute the child tool, emitting progress events.
    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("w_child"),
            LeaseId::from_string("l_child"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire child")?;
    let child_bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;

    let cancel = CancellationToken::new();
    let tool_outcome = execute_tool_task(
        child_bootstrap,
        &stores.tasks,
        &stores.events,
        &cancel,
        |tc, collector| async move {
            // Emit two progress events during execution.
            collector.emit(AgentEvent::tool_progress(
                &tc.id,
                &tc.name,
                &tc.display_name,
                "running",
                "Executing command...",
                None,
            ));
            collector.emit(AgentEvent::tool_progress(
                &tc.id,
                &tc.name,
                &tc.display_name,
                "complete",
                "Command finished",
                Some(serde_json::json!({"exit_code": 0})),
            ));
            Ok(ToolResult::success("output"))
        },
        t0(),
    )
    .await?;

    let ToolTaskOutcome::Completed {
        committed_events: tool_events,
        ..
    } = tool_outcome
    else {
        panic!("expected Completed")
    };

    // Tool should have 3 events: 2 progress + 1 ToolCallEnd.
    assert_eq!(tool_events.len(), 3, "2 ToolProgress + 1 ToolCallEnd");
    assert!(matches!(
        &tool_events[0].event,
        AgentEvent::ToolProgress { stage, .. } if stage == "running"
    ));
    assert!(matches!(
        &tool_events[1].event,
        AgentEvent::ToolProgress { stage, .. } if stage == "complete"
    ));
    assert!(matches!(
        &tool_events[2].event,
        AgentEvent::ToolCallEnd { .. }
    ));

    // Verify from the event repository: full thread replay order.
    // The first turn streams a tool call with no thinking, so the
    // journal leads with the `UserInput` admission record, then `Start`:
    //   UserInput → Start → ToolCallStart → ToolProgress(running)
    //            → ToolProgress(complete) → ToolCallEnd
    let all = stores.events.get_events(&thread_replay()).await?;
    assert!(matches!(&all[0].event, AgentEvent::UserInput { .. }));
    assert!(matches!(&all[1].event, AgentEvent::Start { .. }));
    assert!(matches!(&all[2].event, AgentEvent::ToolCallStart { .. }));
    assert!(matches!(
        &all[3].event,
        AgentEvent::ToolProgress { stage, .. } if stage == "running"
    ));
    assert!(matches!(
        &all[4].event,
        AgentEvent::ToolProgress { stage, .. } if stage == "complete"
    ));
    assert!(matches!(&all[5].event, AgentEvent::ToolCallEnd { .. }));

    // Sequences are contiguous.
    for (i, evt) in all.iter().enumerate() {
        assert_eq!(evt.sequence, i as u64);
    }

    Ok(())
}

/// Execute a child tool task and resume the parent through the journal.
async fn execute_child_and_resume(
    stores: &TestStores,
    child_tasks: &[AgentTask],
    provider: &dyn LlmProvider,
) -> Result<()> {
    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("w_child"),
            LeaseId::from_string("l_child"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire child")?;
    let child_bootstrap = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let cancel = CancellationToken::new();
    execute_tool_task(
        child_bootstrap,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_tc, _collector| async { Ok(ToolResult::success("hi")) },
        t0(),
    )
    .await?;

    let parent_id = child_tasks[0]
        .parent_id
        .as_ref()
        .context("child has no parent_id")?;
    let parent = stores.tasks.get(parent_id).await?.context("parent")?;
    let parent_acq = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("w_resume"),
            LeaseId::from_string("l_resume"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("parent acquirable")?;
    let resume_ctx = WorkerBootstrapContext {
        task: parent_acq.clone(),
        definition: definition_with_tools(),
        thread_id: thread_replay(),
        task_id: parent_acq.id.clone(),
        worker_id: WorkerId::from_string("w_resume"),
        lease_id: parent_acq
            .lease_id
            .clone()
            .context("parent has no lease_id")?,
    };
    let resume_inputs = build_root_worker_inputs(
        resume_ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;
    resume_from_children(resume_inputs, &parent_acq, provider, &stores.deps(), t0()).await?;
    Ok(())
}

/// Full streamed lifecycle across a root suspend, a child tool task,
/// and a streamed resume.  The streamed `ThinkingDelta` / `TextDelta`
/// precede their consolidated blocks, and each root turn leads with its
/// `UserInput` admission record before `Start`.
#[tokio::test]
async fn full_lifecycle_with_thinking_replays_across_root_and_tool() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_replay()).await?;
    let ctx = bootstrap(task, definition_with_tools());
    let inputs = build_root_worker_inputs(
        ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let provider = ThinkingToolCallProvider {
        thinking: "I should run this command".into(),
        tool_calls: vec![(
            "tc_full".into(),
            "bash".into(),
            serde_json::json!({"command": "echo hi"}),
        )],
        call_count: AtomicUsize::new(0),
        resume_thinking: "The command succeeded".into(),
        resume_text: "The output was: hi".into(),
    };

    // 1. Execute root turn → suspend with thinking.
    let outcome = execute_root_turn(inputs, "say hi", &provider, &stores.deps(), t0()).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        panic!("expected Suspended")
    };

    // 2–3. Execute child tool + resume parent.
    Box::pin(execute_child_and_resume(&stores, &child_tasks, &provider)).await?;

    // 4. Verify full replay order from the repository.
    let events = stores.events.get_events(&thread_replay()).await?;

    let types: Vec<&str> = events
        .iter()
        .map(|e| match &e.event {
            AgentEvent::UserInput { .. } => "UserInput",
            AgentEvent::Start { .. } => "Start",
            AgentEvent::Thinking { .. } => "Thinking",
            AgentEvent::ThinkingDelta { .. } => "ThinkingDelta",
            AgentEvent::Text { .. } => "Text",
            AgentEvent::TextDelta { .. } => "TextDelta",
            AgentEvent::ToolCallStart { .. } => "ToolCallStart",
            AgentEvent::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEvent::ToolProgress { .. } => "ToolProgress",
            AgentEvent::TurnComplete { .. } => "TurnComplete",
            AgentEvent::Done { .. } => "Done",
            other => panic!("unexpected event type: {other:?}"),
        })
        .collect();

    // Suspend turn streams thinking then a tool call; the resume turn
    // streams thinking then text.  Streamed deltas precede their
    // consolidated blocks.
    // The resume turn streams *both* deltas (thinking then text) before
    // committing *both* consolidated blocks at turn close, so the resume
    // tail is `ThinkingDelta → TextDelta → Thinking → Text`.
    assert_eq!(
        types,
        vec![
            "UserInput",
            "Start",
            "ThinkingDelta", // streamed (first LLM call)
            "Thinking",      // consolidated (first LLM call)
            "ToolCallStart",
            "ToolCallEnd",
            "ThinkingDelta", // streamed (resume LLM call)
            "TextDelta",     // streamed (resume LLM call)
            "Thinking",      // consolidated (resume LLM call)
            "Text",          // consolidated (resume LLM call)
            "TurnComplete",
            "Done",
        ],
    );

    // Sequences are contiguous across root and tool boundaries.
    for (i, evt) in events.iter().enumerate() {
        assert_eq!(
            evt.sequence, i as u64,
            "sequence gap at index {i}: expected {i}, got {}",
            evt.sequence,
        );
    }

    // All events belong to the same thread.
    for evt in &events {
        assert_eq!(evt.thread_id, thread_replay());
    }

    Ok(())
}

/// Two interleaved tool tasks produce correctly ordered events within
/// the shared thread event stream.
#[tokio::test]
async fn multiple_tool_tasks_interleave_correctly() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_replay()).await?;

    // Definition with two different tools.
    let def = AgentDefinition {
        tools: vec![
            Tool {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({}),
                display_name: "Read".into(),
                tier: ToolTier::Observe,
            },
            Tool {
                name: "write".into(),
                description: "Write a file".into(),
                input_schema: serde_json::json!({}),
                display_name: "Write".into(),
                tier: ToolTier::Observe,
            },
        ],
        ..sample_definition()
    };
    let ctx = bootstrap(task, def.clone());
    let inputs = build_root_worker_inputs(
        ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Provider that returns two tool calls on first call, text on resume.
    let provider = ThinkingToolCallProvider {
        thinking: String::new(),
        tool_calls: vec![
            (
                "tc_read".into(),
                "read".into(),
                serde_json::json!({"path": "/tmp/a"}),
            ),
            (
                "tc_write".into(),
                "write".into(),
                serde_json::json!({"path": "/tmp/b"}),
            ),
        ],
        call_count: AtomicUsize::new(0),
        resume_thinking: String::new(),
        resume_text: "both done".into(),
    };

    let outcome = execute_root_turn(inputs, "do both", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended"),
    };
    assert_eq!(child_tasks.len(), 2);

    // Execute both children sequentially.
    for (i, child_task) in child_tasks.iter().enumerate() {
        let child = stores
            .tasks
            .try_acquire_task(
                &child_task.id,
                WorkerId::from_string(format!("w_{i}")),
                LeaseId::from_string(format!("l_{i}")),
                t_plus(300),
                t0(),
            )
            .await?
            .context("acquire child")?;
        let cb = resolve_tool_bootstrap(child, &stores.tasks).await?;
        let cancel = CancellationToken::new();
        execute_tool_task(
            cb,
            &stores.tasks,
            &stores.events,
            &cancel,
            |_tc, _collector| async { Ok(ToolResult::success("ok")) },
            t0(),
        )
        .await?;
    }

    // Verify event stream (no thinking, so no `ThinkingDelta`):
    //   UserInput → Start → ToolCallStart(read) → ToolCallStart(write)
    //            → ToolCallEnd → ToolCallEnd
    let events = stores.events.get_events(&thread_replay()).await?;

    // UserInput + Start + 2 ToolCallStart + 2 ToolCallEnd = 6 before resume.
    assert!(events.len() >= 6);
    assert!(matches!(&events[0].event, AgentEvent::UserInput { .. }));
    assert!(matches!(&events[1].event, AgentEvent::Start { .. }));
    assert!(matches!(
        &events[2].event,
        AgentEvent::ToolCallStart { name, .. } if name == "read"
    ));
    assert!(matches!(
        &events[3].event,
        AgentEvent::ToolCallStart { name, .. } if name == "write"
    ));
    // ToolCallEnd events follow (order depends on execution order).
    assert!(matches!(&events[4].event, AgentEvent::ToolCallEnd { .. }));
    assert!(matches!(&events[5].event, AgentEvent::ToolCallEnd { .. }));

    // Sequences are contiguous.
    for (i, evt) in events.iter().enumerate() {
        assert_eq!(evt.sequence, i as u64);
    }

    Ok(())
}

/// Collector with no progress events produces no extra committed events.
#[tokio::test]
async fn empty_collector_adds_no_events() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_replay()).await?;
    let ctx = bootstrap(task, definition_with_tools());
    let inputs = build_root_worker_inputs(
        ctx,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let provider = ThinkingToolCallProvider {
        thinking: String::new(),
        tool_calls: vec![(
            "tc_empty".into(),
            "bash".into(),
            serde_json::json!({"command": "true"}),
        )],
        call_count: AtomicUsize::new(0),
        resume_thinking: String::new(),
        resume_text: "ok".into(),
    };
    let outcome = execute_root_turn(inputs, "run", &provider, &stores.deps(), t0()).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        panic!("expected Suspended")
    };

    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("w"),
            LeaseId::from_string("l"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire")?;
    let cb = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let cancel = CancellationToken::new();

    let tool_outcome = execute_tool_task(
        cb,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_tc, _collector| async {
            // No progress events emitted.
            Ok(ToolResult::success("done"))
        },
        t0(),
    )
    .await?;

    let ToolTaskOutcome::Completed {
        committed_events: tool_events,
        ..
    } = tool_outcome
    else {
        panic!("expected Completed")
    };

    // Only the ToolCallEnd event — no extra progress events.
    assert_eq!(tool_events.len(), 1);
    assert!(matches!(
        &tool_events[0].event,
        AgentEvent::ToolCallEnd { .. }
    ));

    Ok(())
}
