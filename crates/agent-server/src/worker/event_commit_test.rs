//! Phase 6.2 regression tests for atomic event commit rules.
//!
//! Verifies that lifecycle events are committed after the state
//! transitions they describe, returned in outcome types, and follow
//! monotonic thread-scoped sequencing.

use super::root_turn::{
    RootTurnDeps, RootTurnOutcome, execute_root_turn, fail_root_turn, resume_from_children,
};
use crate::worker::activity::ActivityBeacon;
use std::sync::Arc;

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
use agent_sdk_foundation::llm::{StopReason, Tool};
use agent_sdk_foundation::{ThreadId, ToolResult, ToolTier};
use anyhow::{Context, Result};
use time::Duration;
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────
// Shared test helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("t-event-commit-a")
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

fn sample_definition_with_tools() -> AgentDefinition {
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

fn sample_bootstrap(task: AgentTask) -> WorkerBootstrapContext {
    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition: sample_definition(),
        thread_id,
        task_id,
        worker_id: WorkerId::from_string("worker_evt"),
        lease_id: LeaseId::from_string("lease_evt"),
    }
}

fn sample_bootstrap_with_tools(task: AgentTask) -> WorkerBootstrapContext {
    let thread_id = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition: sample_definition_with_tools(),
        thread_id,
        task_id,
        worker_id: WorkerId::from_string("worker_evt"),
        lease_id: LeaseId::from_string("lease_evt"),
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
            cancel: None,
            wakeup: None,
            activity: None,
            connectivity_waits: None,
        }
    }
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + Duration::seconds(secs)
}

async fn create_and_acquire_task(
    store: &InMemoryAgentTaskStore,
    thread_id: &ThreadId,
) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
    let task_id = task.id.clone();

    store.submit_root_turn(task).await.context("submit")?;

    let acquired = store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_evt"),
            LeaseId::from_string("lease_evt"),
            t_plus(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("task should be acquirable")?;
    Ok(acquired)
}

// ─────────────────────────────────────────────────────────────────────
// Scripted providers
// ─────────────────────────────────────────────────────────────────────
//
// These tests now drive the *real* streaming commit path through the
// shared `StreamingScriptedProvider` (see `worker::test_support`).  A
// chat-only mock would route through `LlmProvider::chat_stream`'s
// default single-shot adapter and never exercise per-delta commits.

use crate::worker::test_support::{StreamingScriptedProvider, TurnScript};

/// A single text turn (`UserInput → Start → TextDelta → Text →
/// TurnComplete → Done`).
fn text_provider(text: &str) -> StreamingScriptedProvider {
    StreamingScriptedProvider::single(TurnScript::text(text))
}

/// A single text turn ending with [`StopReason::Refusal`].
fn refusal_provider(text: &str) -> StreamingScriptedProvider {
    StreamingScriptedProvider::single(TurnScript::text_with_stop(text, StopReason::Refusal))
}

/// A turn that emits the given tool calls and suspends.  The provider
/// queue also carries a follow-up `text("done")` turn so the same
/// instance can drive the post-tool resume call.
fn tool_call_provider(tool_calls: &[(&str, &str, serde_json::Value)]) -> StreamingScriptedProvider {
    StreamingScriptedProvider::new(vec![
        TurnScript::tool_calls(tool_calls),
        TurnScript::text("done"),
    ])
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_only_turn_emits_turn_complete_and_done() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;

    let provider = text_provider("hello");
    let outcome = execute_root_turn(inputs, "hi", &provider, &stores.deps(), t0()).await?;

    let committed_events = match outcome {
        RootTurnOutcome::Completed {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Suspended { .. } => panic!("expected Completed, got Suspended"),
    };

    // Post-streaming-refactor invariant for the *outcome* vec:
    // `UserInput → Start → Text → TurnComplete → Done` (5 events).
    // The streamed `TextDelta` is committed to the journal during
    // streaming but is *not* threaded back into the outcome vec (it is
    // committed out-of-band by the streaming delta-batch flush), so the
    // outcome carries only the consolidated content + lifecycle edges.
    assert_eq!(committed_events.len(), 5, "expected 5 outcome events");
    assert!(
        matches!(&committed_events[0].event, AgentEvent::UserInput { .. }),
        "event[0] should be UserInput, got {:?}",
        committed_events[0].event,
    );
    assert!(
        matches!(&committed_events[1].event, AgentEvent::Start { .. }),
        "event[1] should be Start, got {:?}",
        committed_events[1].event,
    );
    assert!(
        matches!(&committed_events[2].event, AgentEvent::Text { .. }),
        "event[2] should be Text, got {:?}",
        committed_events[2].event,
    );
    assert!(
        matches!(&committed_events[3].event, AgentEvent::TurnComplete { .. }),
        "event[3] should be TurnComplete, got {:?}",
        committed_events[3].event,
    );
    assert!(
        matches!(&committed_events[4].event, AgentEvent::Done { .. }),
        "event[4] should be Done, got {:?}",
        committed_events[4].event,
    );

    // The journal additionally carries the streamed `TextDelta` between
    // `Start` and the consolidated `Text`, so the full replay is
    // `UserInput → Start → TextDelta → Text → TurnComplete → Done`
    // (6 events) with contiguous sequences.
    let repo_events = stores.events.get_events(&thread_a()).await?;
    assert_eq!(repo_events.len(), 6, "expected 6 journal events");
    assert!(matches!(
        &repo_events[0].event,
        AgentEvent::UserInput { .. }
    ));
    assert!(matches!(&repo_events[1].event, AgentEvent::Start { .. }));
    assert!(
        matches!(&repo_events[2].event, AgentEvent::TextDelta { .. }),
        "event[2] should be the streamed TextDelta, got {:?}",
        repo_events[2].event,
    );
    assert!(matches!(&repo_events[3].event, AgentEvent::Text { .. }));
    assert!(matches!(
        &repo_events[4].event,
        AgentEvent::TurnComplete { .. }
    ));
    assert!(matches!(&repo_events[5].event, AgentEvent::Done { .. }));
    for (i, evt) in repo_events.iter().enumerate() {
        assert_eq!(evt.sequence, i as u64);
    }

    // The streamed `TextDelta` and the consolidated `Text` share the
    // same lazily-allocated message id (`content_ids`), so a live
    // observer can stitch the delta onto the final block.
    let (
        AgentEvent::TextDelta {
            message_id: delta_id,
            ..
        },
        AgentEvent::Text {
            message_id: text_id,
            ..
        },
    ) = (&repo_events[2].event, &repo_events[3].event)
    else {
        panic!("expected TextDelta then Text");
    };
    assert_eq!(
        delta_id, text_id,
        "delta and consolidated Text must share a message id"
    );

    Ok(())
}

#[tokio::test]
async fn refusal_turn_emits_refusal_event() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;

    let provider = refusal_provider("I cannot do that");
    let outcome =
        execute_root_turn(inputs, "do something bad", &provider, &stores.deps(), t0()).await?;

    let committed_events = match outcome {
        RootTurnOutcome::Completed {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Suspended { .. } => panic!("expected Completed, got Suspended"),
    };

    // UserInput + Start + Text + Refusal + TurnComplete + Done (6 events).
    assert_eq!(committed_events.len(), 6, "expected 6 events");
    assert!(
        matches!(&committed_events[0].event, AgentEvent::UserInput { .. }),
        "event[0] should be UserInput, got {:?}",
        committed_events[0].event,
    );
    assert!(
        matches!(&committed_events[1].event, AgentEvent::Start { .. }),
        "event[1] should be Start, got {:?}",
        committed_events[1].event,
    );
    assert!(
        matches!(&committed_events[2].event, AgentEvent::Text { .. }),
        "event[2] should be Text, got {:?}",
        committed_events[2].event,
    );
    assert!(
        matches!(&committed_events[3].event, AgentEvent::Refusal { .. }),
        "event[3] should be Refusal, got {:?}",
        committed_events[3].event,
    );
    assert!(
        matches!(&committed_events[4].event, AgentEvent::TurnComplete { .. }),
        "event[4] should be TurnComplete, got {:?}",
        committed_events[4].event,
    );
    assert!(
        matches!(&committed_events[5].event, AgentEvent::Done { .. }),
        "event[5] should be Done, got {:?}",
        committed_events[5].event,
    );

    Ok(())
}

#[tokio::test]
async fn suspension_emits_tool_call_start_per_tool() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;

    let provider = tool_call_provider(&[
        ("tc_1", "bash", serde_json::json!({"command": "ls"})),
        ("tc_2", "bash", serde_json::json!({"command": "pwd"})),
    ]);
    let outcome =
        execute_root_turn(inputs, "run commands", &provider, &stores.deps(), t0()).await?;

    let committed_events = match outcome {
        RootTurnOutcome::Suspended {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended, got Completed"),
    };

    // UserInput + Start + one ToolCallStart per tool call (4 events).
    assert_eq!(committed_events.len(), 4, "expected 4 suspension events");
    assert!(
        matches!(&committed_events[0].event, AgentEvent::UserInput { .. }),
        "event[0] should be UserInput, got {:?}",
        committed_events[0].event,
    );
    assert!(
        matches!(&committed_events[1].event, AgentEvent::Start { .. }),
        "event[1] should be Start, got {:?}",
        committed_events[1].event,
    );
    assert!(
        matches!(&committed_events[2].event, AgentEvent::ToolCallStart { .. }),
        "event[2] should be ToolCallStart, got {:?}",
        committed_events[2].event,
    );
    assert!(
        matches!(&committed_events[3].event, AgentEvent::ToolCallStart { .. }),
        "event[3] should be ToolCallStart, got {:?}",
        committed_events[3].event,
    );

    // Sequences should be contiguous.
    for (i, evt) in committed_events.iter().enumerate() {
        assert_eq!(evt.sequence, i as u64);
    }

    Ok(())
}

#[tokio::test]
async fn tool_completion_emits_tool_call_end() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;

    // Suspend to create child tasks.
    let provider =
        tool_call_provider(&[("tc_1", "bash", serde_json::json!({"command": "echo hi"}))]);
    let outcome = execute_root_turn(inputs, "run", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended, got Completed"),
    };

    // Acquire and bootstrap the child.
    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("worker_child"),
            LeaseId::from_string("lease_child"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("child task should be acquirable")?;
    let child_bootstrap =
        resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;

    // Execute the tool.
    let cancel = CancellationToken::new();
    let tool_outcome = execute_tool_task(
        child_bootstrap,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_tc, _collector| async {
            Ok(ToolResult {
                success: true,
                output: "hi".into(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            })
        },
        t0(),
    )
    .await?;

    let committed_events = match tool_outcome {
        ToolTaskOutcome::Completed {
            committed_events, ..
        } => committed_events,
        ToolTaskOutcome::Failed { .. } | ToolTaskOutcome::Cancelled => {
            panic!("expected Completed")
        }
    };

    // Should have exactly one ToolCallEnd event.
    assert_eq!(committed_events.len(), 1);
    assert!(
        matches!(&committed_events[0].event, AgentEvent::ToolCallEnd { .. }),
        "expected ToolCallEnd, got {:?}",
        committed_events[0].event,
    );

    Ok(())
}

#[tokio::test]
async fn tool_failure_emits_tool_call_end_with_error() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;

    let provider =
        tool_call_provider(&[("tc_fail", "bash", serde_json::json!({"command": "false"}))]);
    let outcome = execute_root_turn(inputs, "run", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended, got Completed"),
    };

    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("worker_child"),
            LeaseId::from_string("lease_child"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("child task should be acquirable")?;
    let child_bootstrap =
        resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;

    let cancel = CancellationToken::new();
    let tool_outcome = execute_tool_task(
        child_bootstrap,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_tc, _collector| async { anyhow::bail!("tool execution failed") },
        t0(),
    )
    .await?;

    let committed_events = match tool_outcome {
        ToolTaskOutcome::Failed {
            committed_events, ..
        } => committed_events,
        ToolTaskOutcome::Completed { .. } | ToolTaskOutcome::Cancelled => {
            panic!("expected Failed")
        }
    };

    assert_eq!(committed_events.len(), 1);
    match &committed_events[0].event {
        AgentEvent::ToolCallEnd { result, .. } => {
            assert!(!result.success, "failed tool should have success=false");
        }
        event => panic!("expected ToolCallEnd, got {event:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn fail_root_turn_emits_error_event() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let task_id = task.id.clone();
    let worker_id = WorkerId::from_string("worker_evt");
    let lease_id = LeaseId::from_string("lease_evt");

    let err = anyhow::anyhow!("something went wrong");
    fail_root_turn(
        &task_id,
        &worker_id,
        &lease_id,
        &thread_a(),
        &err,
        &stores.deps(),
        t0(),
    )
    .await?;

    let repo_events = stores.events.get_events(&thread_a()).await?;
    assert_eq!(repo_events.len(), 1, "expected 1 Error event");
    assert!(
        matches!(&repo_events[0].event, AgentEvent::Error { .. }),
        "expected Error, got {:?}",
        repo_events[0].event,
    );

    Ok(())
}

/// Execute a child tool task and resume the parent, collecting all
/// lifecycle events along the way.
async fn execute_child_and_resume(
    stores: &TestStores,
    child_tasks: &[AgentTask],
    provider: &StreamingScriptedProvider,
) -> Result<()> {
    let child = stores
        .tasks
        .try_acquire_task(
            &child_tasks[0].id,
            WorkerId::from_string("worker_child"),
            LeaseId::from_string("lease_child"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("child task should be acquirable")?;
    let child_bootstrap =
        resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;
    let cancel = CancellationToken::new();
    execute_tool_task(
        child_bootstrap,
        &stores.tasks,
        &stores.events,
        &cancel,
        |_tc, _collector| async {
            Ok(ToolResult {
                success: true,
                output: "ok".into(),
                data: None,
                documents: Vec::new(),
                duration_ms: None,
            })
        },
        t0(),
    )
    .await?;

    let parent = stores
        .tasks
        .get(&child_tasks[0].parent_id.as_ref().unwrap().clone())
        .await?
        .context("parent")?;

    let parent_acq = stores
        .tasks
        .try_acquire_task(
            &parent.id,
            WorkerId::from_string("worker_resume"),
            LeaseId::from_string("lease_resume"),
            t_plus(300),
            t0(),
        )
        .await?
        .context("parent task should be acquirable")?;

    let resume_bootstrap = WorkerBootstrapContext {
        task: parent_acq.clone(),
        definition: sample_definition_with_tools(),
        thread_id: thread_a(),
        task_id: parent_acq.id.clone(),
        worker_id: WorkerId::from_string("worker_resume"),
        lease_id: parent_acq.lease_id.clone().unwrap(),
    };
    let resume_inputs = build_root_worker_inputs(
        resume_bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    resume_from_children(resume_inputs, &parent_acq, provider, &stores.deps(), t0()).await?;
    Ok(())
}

#[tokio::test]
async fn event_sequences_monotonic_across_lifecycle() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap_with_tools(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    // Step 1: suspend (emits ToolCallStart).
    let provider =
        tool_call_provider(&[("tc_seq", "bash", serde_json::json!({"command": "echo"}))]);
    let outcome = execute_root_turn(inputs, "run", &provider, &stores.deps(), t0()).await?;
    let child_tasks = match outcome {
        RootTurnOutcome::Suspended { child_tasks, .. } => child_tasks,
        RootTurnOutcome::Completed { .. } => panic!("expected Suspended, got Completed"),
    };

    // Steps 2-3: execute child tool + resume parent.
    Box::pin(execute_child_and_resume(&stores, &child_tasks, &provider)).await?;

    // Every event on the thread — across the root suspend, the child
    // tool task, and the streamed resume — has a contiguous, strictly
    // monotonic sequence with no gaps at worker-boundary transitions.
    let all_events = stores.events.get_events(&thread_a()).await?;
    for i in 1..all_events.len() {
        assert_eq!(
            all_events[i].sequence,
            all_events[i - 1].sequence + 1,
            "events at index {}/{} have non-contiguous sequences: {} vs {}",
            i - 1,
            i,
            all_events[i - 1].sequence,
            all_events[i].sequence,
        );
    }

    // Post-streaming-refactor journal ordering across root + tool task
    // boundaries.  The resume turn now streams, so the resume `Text` is
    // preceded by its `TextDelta`:
    //   UserInput → Start → ToolCallStart → ToolCallEnd
    //            → TextDelta → Text → TurnComplete → Done
    // Each root turn leads with `UserInput` (the durable admission
    // record) before `Start`, and there is a single `Start` for the
    // turn — resume does not re-emit `Start`.
    let kinds: Vec<&str> = all_events
        .iter()
        .map(|e| match &e.event {
            AgentEvent::UserInput { .. } => "UserInput",
            AgentEvent::Start { .. } => "Start",
            AgentEvent::ToolCallStart { .. } => "ToolCallStart",
            AgentEvent::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEvent::TextDelta { .. } => "TextDelta",
            AgentEvent::Text { .. } => "Text",
            AgentEvent::TurnComplete { .. } => "TurnComplete",
            AgentEvent::Done { .. } => "Done",
            other => panic!("unexpected event in lifecycle: {other:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "UserInput",
            "Start",
            "ToolCallStart",
            "ToolCallEnd",
            "TextDelta",
            "Text",
            "TurnComplete",
            "Done",
        ],
        "full-lifecycle journal order changed",
    );

    Ok(())
}

#[tokio::test]
async fn committed_events_returned_in_outcome_types() -> Result<()> {
    let stores = TestStores::new();
    let task = create_and_acquire_task(&stores.tasks, &thread_a()).await?;
    let bootstrap = sample_bootstrap(task);
    let inputs = build_root_worker_inputs(
        bootstrap,
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await?;

    let provider = text_provider("result");
    let outcome = execute_root_turn(inputs, "test", &provider, &stores.deps(), t0()).await?;

    let outcome_events = match outcome {
        RootTurnOutcome::Completed {
            committed_events, ..
        } => committed_events,
        RootTurnOutcome::Suspended { .. } => panic!("expected Completed, got Suspended"),
    };

    // Post-streaming-refactor invariant: the outcome's `committed_events`
    // is the consolidated/lifecycle projection of the turn and does *not*
    // include the per-delta `TextDelta` events committed out-of-band by
    // the streaming delta-batch flush.  So the journal carries exactly one more
    // event (the streamed `TextDelta`) than the outcome, and the outcome
    // events are a *subsequence* of the journal — matched by event_id —
    // with a sequence gap where the delta was dropped.
    let repo_events = stores.events.get_events(&thread_a()).await?;
    assert_eq!(
        outcome_events.len() + 1,
        repo_events.len(),
        "journal must carry exactly one extra (streamed TextDelta) event",
    );

    // Every outcome event is present in the journal, in order, identified
    // by its stable event_id and sequence.
    let repo_by_id: std::collections::HashMap<_, _> =
        repo_events.iter().map(|e| (e.event_id, e)).collect();
    let mut last_seq: Option<u64> = None;
    for outcome_evt in &outcome_events {
        let repo_evt = repo_by_id
            .get(&outcome_evt.event_id)
            .context("outcome event missing from journal")?;
        assert_eq!(outcome_evt.sequence, repo_evt.sequence);
        if let Some(prev) = last_seq {
            assert!(
                outcome_evt.sequence > prev,
                "outcome events must stay in journal order",
            );
        }
        last_seq = Some(outcome_evt.sequence);
    }

    // The one journal event absent from the outcome is the streamed
    // `TextDelta`.
    let outcome_ids: std::collections::HashSet<_> =
        outcome_events.iter().map(|e| e.event_id).collect();
    let extra: Vec<_> = repo_events
        .iter()
        .filter(|e| !outcome_ids.contains(&e.event_id))
        .collect();
    assert_eq!(extra.len(), 1, "exactly one journal-only event");
    assert!(
        matches!(&extra[0].event, AgentEvent::TextDelta { .. }),
        "the journal-only event is the streamed TextDelta, got {:?}",
        extra[0].event,
    );

    Ok(())
}
