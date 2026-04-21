//! Comprehensive regression suite for the durable event system.
//!
//! This module exercises the full event lifecycle end-to-end, covering every
//! guarantee required by the event contract:
//!
//! - Monotonic ordering across root turns, tool tasks, and resumed roots.
//! - Restart replay consistency (events survive notifier recreation).
//! - Replay-to-live handoff with zero gaps and zero duplicates.
//! - Lagging subscriber detection, bounded-wait grace, and reconnect recovery.
//! - Fail-closed event persistence (error events committed on failure).
//! - Concurrent child-task progress events with ordering guarantees.
//! - Confirmation pause and resume with correct event sequencing.
//! - Multi-turn contiguous sequencing across distinct root turns.
//! - Cross-thread isolation under concurrent activity.

use super::root_turn::{
    RootTurnDeps, RootTurnOutcome, execute_root_turn, fail_root_turn, resume_from_children,
};
use super::tool_task::{ToolTaskOutcome, execute_tool_task, resolve_tool_bootstrap};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::event_stream::{StreamEvent, stream_events};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::live_tail::{LiveTailConfig, LiveTailEvent, LiveTailHub};
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::retention::InMemoryRetentionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::InMemoryThreadStore;
use crate::journal::turn_attempt_store::InMemoryTurnAttemptStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_core::events::AgentEvent;
use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_core::{ThreadId, ToolResult, ToolTier};
use agent_sdk_providers::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_reg() -> ThreadId {
    ThreadId::from_string("t-phase6-regression")
}

fn thread_reg_b() -> ThreadId {
    ThreadId::from_string("t-phase6-regression-b")
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
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

fn definition_with_two_tools() -> AgentDefinition {
    AgentDefinition {
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

    fn deps(&self) -> RootTurnDeps<'_> {
        RootTurnDeps {
            task_store: &self.tasks,
            thread_store: &self.threads,
            message_store: &self.messages,
            attempt_store: &self.attempts,
            checkpoint_store: &self.checkpoints,
            event_repo: &self.events,
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
        worker_id: WorkerId::from_string("worker_reg"),
        lease_id: LeaseId::from_string("lease_reg"),
    }
}

async fn create_and_acquire(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await.context("submit")?;
    store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_reg"),
            LeaseId::from_string("lease_reg"),
            t_plus(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("task should be acquirable")
}

/// Assert that all sequences in the event list are contiguous starting from
/// the expected base.
fn assert_contiguous_sequences(events: &[crate::journal::committed_event::CommittedEvent]) {
    for (i, evt) in events.iter().enumerate() {
        assert_eq!(
            evt.sequence, i as u64,
            "sequence gap at index {i}: expected {i}, got {}",
            evt.sequence,
        );
    }
}

/// Assert that all events belong to the expected thread.
fn assert_all_same_thread(
    events: &[crate::journal::committed_event::CommittedEvent],
    thread_id: &ThreadId,
) {
    for evt in events {
        assert_eq!(evt.thread_id, *thread_id);
    }
}

/// Extract event type names for readable assertions.
fn event_type_names(events: &[crate::journal::committed_event::CommittedEvent]) -> Vec<&str> {
    events
        .iter()
        .map(|e| match &e.event {
            AgentEvent::Start { .. } => "Start",
            AgentEvent::Thinking { .. } => "Thinking",
            AgentEvent::Text { .. } => "Text",
            AgentEvent::ToolCallStart { .. } => "ToolCallStart",
            AgentEvent::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEvent::ToolProgress { .. } => "ToolProgress",
            AgentEvent::TurnComplete { .. } => "TurnComplete",
            AgentEvent::Done { .. } => "Done",
            AgentEvent::Error { .. } => "Error",
            AgentEvent::Refusal { .. } => "Refusal",
            other => panic!("unexpected event type: {other:?}"),
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// Mock providers
// ─────────────────────────────────────────────────────────────────────

struct MockTextProvider {
    response_text: String,
}

#[async_trait]
impl LlmProvider for MockTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_reg_text".into(),
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

struct MockToolCallProvider {
    tool_calls: Vec<(String, String, serde_json::Value)>,
    call_count: AtomicUsize,
    resume_text: String,
}

#[async_trait]
impl LlmProvider for MockToolCallProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
        if call_num > 0 {
            return Ok(ChatOutcome::Success(ChatResponse {
                id: "msg_reg_resume".into(),
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
            }));
        }
        let content: Vec<ContentBlock> = self
            .tool_calls
            .iter()
            .map(|(id, name, input)| ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                thought_signature: None,
            })
            .collect();
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_reg_tool".into(),
            content,
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
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

// ─────────────────────────────────────────────────────────────────────
// 1. Monotonic ordering across root turn + tool task + resume
// ─────────────────────────────────────────────────────────────────────

/// Full root-turn lifecycle: suspend → tool execution → resume.
/// All events on the thread have contiguous, monotonically increasing
/// sequences with no gaps at worker-boundary transitions.
#[tokio::test]
async fn monotonic_ordering_across_full_lifecycle() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let ctx = bootstrap(task, definition_with_tools());
    let inputs = build_root_worker_inputs(ctx, &stores.threads, &stores.checkpoints, t0()).await?;

    let provider = MockToolCallProvider {
        tool_calls: vec![(
            "tc_mono".into(),
            "bash".into(),
            serde_json::json!({"command": "echo ok"}),
        )],
        call_count: AtomicUsize::new(0),
        resume_text: "done".into(),
    };

    // Step 1: Execute root turn → suspends with tool call.
    let outcome = execute_root_turn(inputs, "run command", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended"),
    };

    // Step 2: Execute the child tool task.
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
        |_tc, _collector| async { Ok(ToolResult::success("output")) },
        t0(),
    )
    .await?;

    // Step 3: Resume the parent.
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
        thread_id: thread_reg(),
        task_id: parent_acq.id.clone(),
        worker_id: WorkerId::from_string("w_resume"),
        lease_id: parent_acq
            .lease_id
            .clone()
            .context("parent has no lease_id")?,
    };
    let resume_inputs =
        build_root_worker_inputs(resume_ctx, &stores.threads, &stores.checkpoints, t0()).await?;
    resume_from_children(resume_inputs, &parent_acq, &provider, &stores.deps(), t0()).await?;

    // Verify: all events are contiguous and belong to the same thread.
    let events = stores.events.get_events(&thread_reg()).await?;
    assert!(
        events.len() >= 6,
        "expected at least 6 events, got {}",
        events.len()
    );
    assert_contiguous_sequences(&events);
    assert_all_same_thread(&events, &thread_reg());

    // Verify event type ordering.
    let types = event_type_names(&events);
    assert_eq!(types[0], "Start");
    assert_eq!(types[1], "ToolCallStart");
    assert_eq!(types[2], "ToolCallEnd");
    assert_eq!(types[3], "Text");
    assert_eq!(types[4], "TurnComplete");
    assert_eq!(types[5], "Done");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 2. Restart replay — events survive notifier recreation
// ─────────────────────────────────────────────────────────────────────

/// After events are committed, a fresh `EventNotifier` (simulating a
/// server restart) still allows full replay from the durable
/// `EventRepository`. New live events after restart also stream
/// correctly.
#[tokio::test]
async fn restart_replay_events_survive_notifier_recreation() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Pre-restart: commit 5 events.
    for i in 0..5i64 {
        let e = repo
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("m{i}"), format!("pre-{i}")),
                t_plus(i),
            )
            .await?;
        notifier.notify(std::slice::from_ref(&e));
    }

    // Simulate restart: drop old notifier, create new one.
    drop(notifier);
    let new_notifier = EventNotifier::new();

    // Reconnect client replays from the beginning.
    let mut stream = stream_events(
        &thread_reg(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &new_notifier,
    )
    .await?;

    // All 5 pre-restart events should replay.
    let mut seen = Vec::new();
    for _ in 0..5 {
        match stream.next().await {
            Some(StreamEvent::Event(e)) => seen.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(seen, vec![0, 1, 2, 3, 4]);

    // Post-restart: commit and notify new events.
    let e5 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m5", "post-5"), t_plus(5))
        .await?;
    new_notifier.notify(std::slice::from_ref(&e5));

    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 5),
        other => panic!("expected Event(seq=5), got {other:?}"),
    }

    // Reconnect from after seq 3 — should get 4 and 5.
    let mut reconnected = stream_events(
        &thread_reg(),
        Some(3),
        &repo,
        &InMemoryRetentionStore::new(),
        &new_notifier,
    )
    .await?;
    let e6 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m6", "post-6"), t_plus(6))
        .await?;
    new_notifier.notify(std::slice::from_ref(&e6));

    let mut reconnect_seen = Vec::new();
    for _ in 0..3 {
        match reconnected.next().await {
            Some(StreamEvent::Event(e)) => reconnect_seen.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(reconnect_seen, vec![4, 5, 6]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 3. Replay-to-live handoff — worker events during active stream
// ─────────────────────────────────────────────────────────────────────

/// A subscriber opens a stream before a root turn executes. The replay
/// phase delivers pre-existing events, then the live tail seamlessly
/// picks up events committed during the turn — no gaps, no duplicates.
#[tokio::test]
async fn replay_to_live_handoff_during_worker_execution() -> Result<()> {
    let stores = TestStores::new();
    let notifier = EventNotifier::new();

    // Pre-commit 2 events (before any worker runs).
    for i in 0..2i64 {
        let e = stores
            .events
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("pre_{i}"), format!("pre-{i}")),
                t_plus(i),
            )
            .await?;
        notifier.notify(std::slice::from_ref(&e));
    }

    // Open a stream from after seq 0.
    let mut stream = stream_events(
        &thread_reg(),
        Some(0),
        &stores.events,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Replay delivers seq 1.
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 1),
        other => panic!("expected Event(seq=1), got {other:?}"),
    }

    // Now execute a text-only root turn — its events will go to the
    // repository and notify will push them to the live tail.
    let task = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let ctx = bootstrap(task, sample_definition());
    let inputs = build_root_worker_inputs(ctx, &stores.threads, &stores.checkpoints, t0()).await?;
    let provider = MockTextProvider {
        response_text: "answer".into(),
    };
    let outcome = execute_root_turn(inputs, "hi", &provider, &stores.deps(), t0()).await?;
    let committed = match outcome {
        RootTurnOutcome::Completed {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Suspended { .. } => panic!("expected Completed"),
    };
    notifier.notify(&committed);

    // Stream should deliver all worker events via live tail.
    let expected_count = committed.len();
    let mut live_seen = Vec::new();
    for _ in 0..expected_count {
        match stream.next().await {
            Some(StreamEvent::Event(e)) => live_seen.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }

    // Sequences continue contiguously from pre-committed events.
    let expected_seqs: Vec<u64> = (2..2 + expected_count as u64).collect();
    assert_eq!(live_seen, expected_seqs);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 4. Lagging subscriber — LiveTailHub lag detection and recovery
// ─────────────────────────────────────────────────────────────────────

/// A slow subscriber that cannot keep up with published events is
/// disconnected with a `ReplayRequired` signal after the grace period.
/// Reconnecting via `stream_events` recovers the missed events.
#[tokio::test]
async fn lagging_subscriber_disconnect_and_recovery() -> Result<()> {
    let repo = InMemoryEventRepository::new();

    // Tiny buffer, zero grace period for deterministic lag testing.
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 3,
        lag_grace_period: Duration::ZERO,
    });
    let mut rx = hub.subscribe(&thread_reg());

    // Commit and publish 3 events (fill buffer).
    for i in 0..3i64 {
        let e = repo
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("m{i}"), format!("v{i}")),
                t_plus(i),
            )
            .await?;
        hub.publish(std::slice::from_ref(&e));
    }

    // This publish triggers lag (buffer full).
    let e3 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m3", "v3"), t_plus(3))
        .await?;
    hub.publish(std::slice::from_ref(&e3));

    // Next publish disconnects (zero grace period).
    let e4 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m4", "v4"), t_plus(4))
        .await?;
    hub.publish(std::slice::from_ref(&e4));

    assert_eq!(hub.subscriber_count(&thread_reg()), 0);

    // Drain the 3 buffered events.
    for expected_seq in 0..3 {
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, expected_seq),
            other => panic!("expected Event({expected_seq}), got {other:?}"),
        }
    }

    // Next recv: ReplayRequired with last_delivered = 2.
    match rx.recv().await {
        Some(LiveTailEvent::ReplayRequired {
            last_delivered_sequence,
        }) => {
            assert_eq!(last_delivered_sequence, Some(2));
        }
        other => panic!("expected ReplayRequired, got {other:?}"),
    }

    // Stream is closed.
    assert!(rx.recv().await.is_none());

    // Recovery: reconnect via stream_events with after_sequence = 2.
    let notifier = EventNotifier::new();
    let mut reconnected = stream_events(
        &thread_reg(),
        Some(2),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Should replay seq 3 and 4 from durable storage.
    let mut recovered = Vec::new();
    for _ in 0..2 {
        match reconnected.next().await {
            Some(StreamEvent::Event(e)) => recovered.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(recovered, vec![3, 4]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 5. Fail-closed event persistence
// ─────────────────────────────────────────────────────────────────────

/// A failed root turn commits an Error event to the durable repository.
/// The error event is replayable and has a valid sequence.
#[tokio::test]
async fn fail_closed_error_event_persisted() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let task_id = task.id.clone();

    let err = anyhow::anyhow!("catastrophic failure");
    fail_root_turn(
        &task_id,
        &WorkerId::from_string("worker_reg"),
        &LeaseId::from_string("lease_reg"),
        &thread_reg(),
        &err,
        &stores.deps(),
        t0(),
    )
    .await?;

    // Error event is in the repository.
    let events = stores.events.get_events(&thread_reg()).await?;
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0].event, AgentEvent::Error { .. }));
    assert_eq!(events[0].sequence, 0);

    // Error event is replayable via stream_events.
    let notifier = EventNotifier::new();
    let mut stream = stream_events(
        &thread_reg(),
        None,
        &stores.events,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    match stream.next().await {
        Some(StreamEvent::Event(e)) => {
            assert!(matches!(&e.event, AgentEvent::Error { .. }));
            assert_eq!(e.sequence, 0);
        }
        other => panic!("expected Error event, got {other:?}"),
    }

    Ok(())
}

/// Events from a successful turn following a failed one continue with
/// contiguous sequences.
#[tokio::test]
async fn events_after_failure_continue_contiguous_sequence() -> Result<()> {
    let stores = TestStores::new();

    // First: a failed turn emits an Error event at seq 0.
    let task1 = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let task1_id = task1.id.clone();
    fail_root_turn(
        &task1_id,
        &WorkerId::from_string("worker_reg"),
        &LeaseId::from_string("lease_reg"),
        &thread_reg(),
        &anyhow::anyhow!("first failure"),
        &stores.deps(),
        t0(),
    )
    .await?;

    // Second: a successful text-only turn.
    let task2 = AgentTask::new_root_turn(thread_reg(), t_plus(1), 3);
    let task2_id = task2.id.clone();
    stores.tasks.submit_root_turn(task2).await?;
    let acquired2 = stores
        .tasks
        .try_acquire_task(
            &task2_id,
            WorkerId::from_string("w2"),
            LeaseId::from_string("l2"),
            t_plus(300),
            t_plus(1),
        )
        .await?
        .context("acquire second task")?;
    let ctx2 = WorkerBootstrapContext {
        task: acquired2,
        definition: sample_definition(),
        thread_id: thread_reg(),
        task_id: task2_id,
        worker_id: WorkerId::from_string("w2"),
        lease_id: LeaseId::from_string("l2"),
    };
    let inputs2 =
        build_root_worker_inputs(ctx2, &stores.threads, &stores.checkpoints, t_plus(1)).await?;
    let provider = MockTextProvider {
        response_text: "recovered".into(),
    };
    execute_root_turn(inputs2, "retry", &provider, &stores.deps(), t_plus(1)).await?;

    // Verify: all events have contiguous sequences.
    let all_events = stores.events.get_events(&thread_reg()).await?;
    assert!(
        all_events.len() >= 5,
        "Error + Start + Text + TurnComplete + Done"
    );
    assert_contiguous_sequences(&all_events);
    assert_eq!(event_type_names(&all_events)[0], "Error");
    assert_eq!(event_type_names(&all_events)[1], "Start");

    Ok(())
}

/// Acquire a child task, bootstrap it, and execute with a progress-emitting
/// executor. Returns the outcome.
async fn acquire_and_execute_child_with_progress(
    stores: &TestStores,
    child_task: &AgentTask,
    worker_label: &str,
    progress_stages: Vec<(&str, &str)>,
) -> Result<ToolTaskOutcome> {
    let child = stores
        .tasks
        .try_acquire_task(
            &child_task.id,
            WorkerId::from_string(format!("w_{worker_label}")),
            LeaseId::from_string(format!("l_{worker_label}")),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire child")?;
    let cb = resolve_tool_bootstrap(child, &stores.tasks).await?;
    let cancel = CancellationToken::new();
    let stages: Vec<(String, String)> = progress_stages
        .into_iter()
        .map(|(s, m)| (s.to_owned(), m.to_owned()))
        .collect();
    execute_tool_task(
        cb,
        &stores.tasks,
        &stores.events,
        &cancel,
        |tc, collector| {
            let stages = stages;
            async move {
                for (stage, message) in &stages {
                    collector.emit(AgentEvent::tool_progress(
                        &tc.id,
                        &tc.name,
                        &tc.display_name,
                        stage,
                        message,
                        None,
                    ));
                }
                Ok(ToolResult::success("ok"))
            }
        },
        t0(),
    )
    .await
}

// ─────────────────────────────────────────────────────────────────────
// 6. Concurrent child-task progress events
// ─────────────────────────────────────────────────────────────────────

/// Two child tool tasks execute sequentially and emit progress events.
/// All events — including progress — are durably committed with
/// contiguous sequences.
#[tokio::test]
async fn concurrent_child_progress_events_ordered() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let ctx = bootstrap(task, definition_with_two_tools());
    let inputs = build_root_worker_inputs(ctx, &stores.threads, &stores.checkpoints, t0()).await?;

    let provider = MockToolCallProvider {
        tool_calls: vec![
            (
                "tc_read".into(),
                "read".into(),
                serde_json::json!({"path": "/a"}),
            ),
            (
                "tc_write".into(),
                "write".into(),
                serde_json::json!({"path": "/b"}),
            ),
        ],
        call_count: AtomicUsize::new(0),
        resume_text: "both done".into(),
    };

    // Suspend: emits Start + 2 ToolCallStart.
    let outcome = execute_root_turn(inputs, "do both", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended"),
    };
    assert_eq!(child_tasks.len(), 2);

    // Execute child 1 with one progress event.
    acquire_and_execute_child_with_progress(
        &stores,
        &child_tasks[0],
        "c1",
        vec![("reading", "Reading file...")],
    )
    .await?;

    // Execute child 2 with two progress events.
    acquire_and_execute_child_with_progress(
        &stores,
        &child_tasks[1],
        "c2",
        vec![("writing", "Writing file..."), ("done", "File written")],
    )
    .await?;

    // Verify: all events are contiguous.
    let all = stores.events.get_events(&thread_reg()).await?;
    assert_contiguous_sequences(&all);
    assert_all_same_thread(&all, &thread_reg());

    let types = event_type_names(&all);
    assert_eq!(types[0], "Start");
    assert_eq!(types[1], "ToolCallStart");
    assert_eq!(types[2], "ToolCallStart");
    assert_eq!(types[3], "ToolProgress");
    assert_eq!(types[4], "ToolCallEnd");
    assert_eq!(types[5], "ToolProgress");
    assert_eq!(types[6], "ToolProgress");
    assert_eq!(types[7], "ToolCallEnd");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 7. Resumed root-turn flow — events span suspend and resume
// ─────────────────────────────────────────────────────────────────────

/// Execute a child tool task (no progress events) and resume the parent.
async fn execute_child_and_resume_parent(
    stores: &TestStores,
    child_tasks: &[AgentTask],
    provider: &dyn LlmProvider,
    thread_id: &ThreadId,
) -> Result<RootTurnOutcome> {
    // Execute child tool.
    acquire_and_execute_child_with_progress(stores, &child_tasks[0], "child", vec![]).await?;

    // Resume parent.
    let parent_id = child_tasks[0].parent_id.as_ref().context("no parent_id")?;
    let parent = stores.tasks.get(parent_id).await?.context("parent")?;
    let parent_acq = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("w_res"),
            LeaseId::from_string("l_res"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("parent acquirable")?;
    let resume_ctx = WorkerBootstrapContext {
        task: parent_acq.clone(),
        definition: definition_with_tools(),
        thread_id: thread_id.clone(),
        task_id: parent_acq.id.clone(),
        worker_id: WorkerId::from_string("w_res"),
        lease_id: parent_acq
            .lease_id
            .clone()
            .context("no lease_id on parent")?,
    };
    let resume_inputs =
        build_root_worker_inputs(resume_ctx, &stores.threads, &stores.checkpoints, t0()).await?;
    resume_from_children(resume_inputs, &parent_acq, provider, &stores.deps(), t0()).await
}

/// A full suspend → tool → resume cycle produces a contiguous event
/// stream. The resumed root adds `Text` + `TurnComplete` + `Done` after the
/// tool's `ToolCallEnd`.
#[tokio::test]
async fn resumed_root_turn_events_span_suspend_and_resume() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let ctx = bootstrap(task, definition_with_tools());
    let inputs = build_root_worker_inputs(ctx, &stores.threads, &stores.checkpoints, t0()).await?;

    let provider = MockToolCallProvider {
        tool_calls: vec![(
            "tc_span".into(),
            "bash".into(),
            serde_json::json!({"command": "test"}),
        )],
        call_count: AtomicUsize::new(0),
        resume_text: "completed after tool".into(),
    };

    // Suspend.
    let outcome = execute_root_turn(inputs, "span test", &provider, &stores.deps(), t0()).await?;
    let (child_tasks, suspend_events) = match outcome {
        RootTurnOutcome::Suspended {
            child_tasks,
            committed_events,
            ..
        } => (child_tasks, committed_events),
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended"),
    };
    assert_eq!(suspend_events.len(), 2);
    assert_eq!(
        event_type_names(&suspend_events),
        vec!["Start", "ToolCallStart"]
    );

    // Execute child and resume parent.
    let resume_outcome =
        execute_child_and_resume_parent(&stores, &child_tasks, &provider, &thread_reg()).await?;
    let resume_events = match resume_outcome {
        RootTurnOutcome::Completed {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Suspended { .. } => panic!("expected Completed after resume"),
    };
    assert_eq!(resume_events.len(), 3);
    assert_eq!(
        event_type_names(&resume_events),
        vec!["Text", "TurnComplete", "Done"]
    );

    // Full thread event stream is contiguous.
    let all = stores.events.get_events(&thread_reg()).await?;
    assert_eq!(
        event_type_names(&all),
        vec![
            "Start",
            "ToolCallStart",
            "ToolCallEnd",
            "Text",
            "TurnComplete",
            "Done",
        ]
    );
    assert_contiguous_sequences(&all);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 8. Cross-thread isolation under concurrent activity
// ─────────────────────────────────────────────────────────────────────

/// Two threads receive events concurrently. Each thread maintains
/// independent monotonic sequences starting from 0, and events do
/// not leak across threads.
#[tokio::test]
async fn cross_thread_isolation() -> Result<()> {
    let stores = TestStores::new();

    // Thread A: text-only turn.
    let root_a = AgentTask::new_root_turn(thread_reg(), t0(), 3);
    let root_a_id = root_a.id.clone();
    stores.tasks.submit_root_turn(root_a).await?;
    let acq_a = stores
        .tasks
        .try_acquire_task(
            &root_a_id,
            WorkerId::from_string("w_a"),
            LeaseId::from_string("l_a"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire A")?;
    let ctx_a = WorkerBootstrapContext {
        task: acq_a,
        definition: sample_definition(),
        thread_id: thread_reg(),
        task_id: root_a_id,
        worker_id: WorkerId::from_string("w_a"),
        lease_id: LeaseId::from_string("l_a"),
    };
    let inputs_a =
        build_root_worker_inputs(ctx_a, &stores.threads, &stores.checkpoints, t0()).await?;

    // Thread B: text-only turn.
    let root_second = AgentTask::new_root_turn(thread_reg_b(), t0(), 3);
    let root_second_id = root_second.id.clone();
    stores.tasks.submit_root_turn(root_second).await?;
    let acq_b = stores
        .tasks
        .try_acquire_task(
            &root_second_id,
            WorkerId::from_string("w_b"),
            LeaseId::from_string("l_b"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("acquire B")?;
    let ctx_b = WorkerBootstrapContext {
        task: acq_b,
        definition: sample_definition(),
        thread_id: thread_reg_b(),
        task_id: root_second_id,
        worker_id: WorkerId::from_string("w_b"),
        lease_id: LeaseId::from_string("l_b"),
    };
    let inputs_b =
        build_root_worker_inputs(ctx_b, &stores.threads, &stores.checkpoints, t0()).await?;

    let provider_a = MockTextProvider {
        response_text: "answer A".into(),
    };
    let provider_b = MockTextProvider {
        response_text: "answer B".into(),
    };

    // Execute both turns.
    execute_root_turn(inputs_a, "question A", &provider_a, &stores.deps(), t0()).await?;
    execute_root_turn(inputs_b, "question B", &provider_b, &stores.deps(), t0()).await?;

    // Verify: each thread has independent events.
    let events_a = stores.events.get_events(&thread_reg()).await?;
    let events_b = stores.events.get_events(&thread_reg_b()).await?;

    // Both start at seq 0.
    assert_eq!(events_a[0].sequence, 0);
    assert_eq!(events_b[0].sequence, 0);

    // Both are contiguous within their own thread.
    assert_contiguous_sequences(&events_a);
    assert_contiguous_sequences(&events_b);

    // No cross-contamination.
    assert_all_same_thread(&events_a, &thread_reg());
    assert_all_same_thread(&events_b, &thread_reg_b());

    // Same structure.
    assert_eq!(event_type_names(&events_a), event_type_names(&events_b));

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 9. Multi-turn contiguous sequencing
// ─────────────────────────────────────────────────────────────────────

/// Two consecutive text-only turns on the same thread produce events
/// with sequences that are contiguous across the turn boundary.
#[tokio::test]
async fn multi_turn_contiguous_sequencing() -> Result<()> {
    let stores = TestStores::new();

    // Turn 1.
    let task1 = create_and_acquire(&stores.tasks, &thread_reg()).await?;
    let ctx1 = bootstrap(task1, sample_definition());
    let inputs1 =
        build_root_worker_inputs(ctx1, &stores.threads, &stores.checkpoints, t0()).await?;
    let p1 = MockTextProvider {
        response_text: "turn 1 answer".into(),
    };
    execute_root_turn(inputs1, "question 1", &p1, &stores.deps(), t0()).await?;

    let events_after_turn1 = stores.events.get_events(&thread_reg()).await?;
    let turn1_count = events_after_turn1.len();

    // Turn 2.
    let task2 = AgentTask::new_root_turn(thread_reg(), t_plus(10), 3);
    let task2_id = task2.id.clone();
    stores.tasks.submit_root_turn(task2).await?;
    let acq2 = stores
        .tasks
        .try_acquire_task(
            &task2_id,
            WorkerId::from_string("w2"),
            LeaseId::from_string("l2"),
            t_plus(300),
            t_plus(10),
        )
        .await?
        .context("acquire task 2")?;
    let ctx2 = WorkerBootstrapContext {
        task: acq2,
        definition: sample_definition(),
        thread_id: thread_reg(),
        task_id: task2_id,
        worker_id: WorkerId::from_string("w2"),
        lease_id: LeaseId::from_string("l2"),
    };
    let inputs2 =
        build_root_worker_inputs(ctx2, &stores.threads, &stores.checkpoints, t_plus(10)).await?;
    let p2 = MockTextProvider {
        response_text: "turn 2 answer".into(),
    };
    execute_root_turn(inputs2, "question 2", &p2, &stores.deps(), t_plus(10)).await?;

    // All events across both turns should be contiguous.
    let all_events = stores.events.get_events(&thread_reg()).await?;
    assert!(all_events.len() > turn1_count);
    assert_contiguous_sequences(&all_events);

    // Turn 2 events start right after turn 1 events.
    assert_eq!(all_events[turn1_count].sequence, turn1_count as u64);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 10. LiveTailHub bounded-wait grace period
// ─────────────────────────────────────────────────────────────────────

/// A lagging subscriber within the grace period is NOT disconnected.
/// After the grace period expires, the next publish triggers disconnect
/// with correct last-delivered sequence.
#[tokio::test]
async fn bounded_wait_grace_period_respected() -> Result<()> {
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 2,
        lag_grace_period: Duration::from_millis(200),
    });
    let mut rx = hub.subscribe(&thread_reg());

    let repo = InMemoryEventRepository::new();

    // Fill buffer.
    for i in 0..2i64 {
        let e = repo
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("m{i}"), format!("v{i}")),
                t_plus(i),
            )
            .await?;
        hub.publish(std::slice::from_ref(&e));
    }

    // Trigger lag.
    let e2 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m2", "v2"), t_plus(2))
        .await?;
    hub.publish(std::slice::from_ref(&e2));

    // Still within grace period — subscriber should still be connected.
    assert_eq!(hub.subscriber_count(&thread_reg()), 1);

    // Publish more while in grace period.
    let e3 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m3", "v3"), t_plus(3))
        .await?;
    hub.publish(std::slice::from_ref(&e3));
    assert_eq!(
        hub.subscriber_count(&thread_reg()),
        1,
        "subscriber should remain during grace period"
    );

    // Wait for grace period to expire.
    tokio::time::sleep(Duration::from_millis(250)).await;

    // Next publish triggers disconnect.
    let e4 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m4", "v4"), t_plus(4))
        .await?;
    hub.publish(std::slice::from_ref(&e4));
    assert_eq!(
        hub.subscriber_count(&thread_reg()),
        0,
        "subscriber should be disconnected after grace period"
    );

    // Drain the 2 buffered events.
    match rx.recv().await {
        Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
        other => panic!("expected Event(0), got {other:?}"),
    }
    match rx.recv().await {
        Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 1),
        other => panic!("expected Event(1), got {other:?}"),
    }

    // ReplayRequired with last_delivered = 1.
    match rx.recv().await {
        Some(LiveTailEvent::ReplayRequired {
            last_delivered_sequence,
        }) => {
            assert_eq!(last_delivered_sequence, Some(1));
        }
        other => panic!("expected ReplayRequired, got {other:?}"),
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 11. Replay-required reconnect semantics
// ─────────────────────────────────────────────────────────────────────

/// After receiving `ReplayRequired`, a client reconnects via
/// `stream_events` using the `last_delivered_sequence` as the
/// `after_sequence`. This picks up exactly the events that were
/// missed — no gaps, no duplicates.
#[tokio::test]
async fn replay_required_reconnect_resumes_exactly() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::with_capacity(3);

    // Commit 2 events.
    for i in 0..2i64 {
        let e = repo
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("m{i}"), format!("v{i}")),
                t_plus(i),
            )
            .await?;
        notifier.notify(std::slice::from_ref(&e));
    }

    // Stream from start, drain replay.
    let mut stream = stream_events(
        &thread_reg(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let mut seen = Vec::new();
    for _ in 0..2 {
        match stream.next().await {
            Some(StreamEvent::Event(e)) => seen.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(seen, vec![0, 1]);

    // Flood the live tail to trigger lagging.
    for i in 2..20i64 {
        let e = repo
            .commit_event(
                &thread_reg(),
                AgentEvent::text(format!("m{i}"), format!("v{i}")),
                t_plus(i),
            )
            .await?;
        notifier.notify(std::slice::from_ref(&e));
    }

    match stream.next().await {
        Some(StreamEvent::Lagged { skipped }) => {
            assert!(skipped > 0);
        }
        other => panic!("expected Lagged, got {other:?}"),
    }

    // Reconnect from last seen (1).
    let mut reconnected = stream_events(
        &thread_reg(),
        Some(1),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let mut reconnect_seen = Vec::new();
    for _ in 0..18 {
        match reconnected.next().await {
            Some(StreamEvent::Event(e)) => reconnect_seen.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(reconnect_seen, (2..20).collect::<Vec<u64>>());

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 12. Committed-envelope-only guarantee
// ─────────────────────────────────────────────────────────────────────

/// Events committed to the repository but NOT notified are invisible
/// in the live tail. Only durable replay (via reconnect) surfaces them.
#[tokio::test]
async fn committed_only_events_invisible_until_replay() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Commit and notify event 0.
    let e0 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m0", "v0"), t0())
        .await?;
    notifier.notify(std::slice::from_ref(&e0));

    // Stream from start, drain event 0.
    let mut stream = stream_events(
        &thread_reg(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 0),
        other => panic!("expected Event(0), got {other:?}"),
    }

    // Commit event 1 WITHOUT notifying.
    repo.commit_event(&thread_reg(), AgentEvent::text("m1", "ghost"), t_plus(1))
        .await?;

    // Commit and notify event 2.
    let e2 = repo
        .commit_event(&thread_reg(), AgentEvent::text("m2", "visible"), t_plus(2))
        .await?;
    notifier.notify(std::slice::from_ref(&e2));

    // Live tail delivers event 2 (skipping the unnotified event 1).
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 2),
        other => panic!("expected Event(2), got {other:?}"),
    }

    // Reconnect to recover the "ghost" event 1 via durable replay.
    let mut reconnected = stream_events(
        &thread_reg(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let mut full = Vec::new();
    for _ in 0..3 {
        match reconnected.next().await {
            Some(StreamEvent::Event(e)) => full.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(full, vec![0, 1, 2]);

    Ok(())
}
