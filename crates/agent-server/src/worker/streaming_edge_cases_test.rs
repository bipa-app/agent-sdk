//! Streaming edge-case regression tests (plan §6.1 / §A2.4).
//!
//! These drive the worker's real `chat_stream` + `StreamAccumulator`
//! commit path through the shared [`StreamingScriptedProvider`] over
//! deliberately awkward scripted delta sequences:
//!
//! * interleaved text + tool-use deltas on different blocks,
//! * out-of-order `block_index`,
//! * a mid-stream `ServerError` / `RateLimited` that triggers a retry,
//! * a stream that ends without a `Done` delta,
//! * a stream truncated after `ToolUseStart` but before any
//!   `ToolInputDelta`.
//!
//! Every case asserts the committed event order *and* that the
//! synthesized turn produces a balanced, API-valid history (every
//! `tool_use` is well-formed; text/tool blocks land in `block_index`
//! order; no half-parsed tool input leaks through).
//!
//! Retry timing is virtual: the recovery cases use
//! `#[tokio::test(start_paused = true)]` so the backoff `sleep`
//! completes instantly and deterministically — no real wall-clock
//! delay, no flakiness.

use std::sync::Arc;

use super::root_turn::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use super::test_support::{StreamingScriptedProvider, TurnScript};
use crate::journal::checkpoint_store::InMemoryCheckpointStore;
use crate::journal::event_notifier::EventNotifier;
use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::execution_context::build_root_worker_inputs;
use crate::journal::message_store::InMemoryMessageProjectionStore;
use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use crate::journal::task::{AgentTask, LeaseId, WorkerId};
use crate::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use crate::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{StopReason, Tool};
use agent_sdk_foundation::{ThreadId, ToolTier};
use agent_sdk_providers::streaming::{StreamDelta, StreamErrorKind};
use anyhow::{Context, Result};
use time::Duration;

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> time::OffsetDateTime {
    time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> time::OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn thread_id() -> ThreadId {
    ThreadId::from_string("t-streaming-edge")
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "You are a test assistant.".into(),
        tools: vec![Tool {
            name: "get_weather".into(),
            description: "Get the weather".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }),
            display_name: "Weather".into(),
            tier: ToolTier::Observe,
        }],
        max_tokens: 1024,
        thinking: ThinkingPolicy::Disabled,
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
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
            cancel: None,
            wakeup: None,
            activity: None,
        }
    }
}

async fn acquire_root(stores: &Stores) -> Result<AgentTask> {
    let task = AgentTask::new_root_turn(thread_id(), t0(), 3);
    let task_id = task.id.clone();
    stores
        .tasks
        .submit_root_turn(task)
        .await
        .context("submit")?;
    stores
        .tasks
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("worker_edge"),
            LeaseId::from_string("lease_edge"),
            t_plus(300),
            t0(),
        )
        .await
        .context("acquire")?
        .context("acquirable")
}

fn bootstrap(task: AgentTask) -> WorkerBootstrapContext {
    let thread = task.thread_id.clone();
    let task_id = task.id.clone();
    WorkerBootstrapContext {
        task,
        definition: definition(),
        thread_id: thread,
        task_id,
        worker_id: WorkerId::from_string("worker_edge"),
        lease_id: LeaseId::from_string("lease_edge"),
    }
}

/// Run a single scripted root turn end-to-end.
async fn run_turn(
    stores: &Stores,
    provider: &StreamingScriptedProvider,
) -> Result<RootTurnOutcome> {
    let task = acquire_root(stores).await?;
    let inputs = build_root_worker_inputs(
        bootstrap(task),
        &stores.threads,
        &stores.checkpoints,
        &stores.messages,
        t0(),
    )
    .await
    .context("build inputs")?;
    execute_root_turn(inputs, "edge", provider, &stores.deps(), t0()).await
}

fn event_kinds(events: &[crate::journal::committed_event::CommittedEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|e| match &e.event {
            AgentEvent::UserInput { .. } => "UserInput",
            AgentEvent::Start { .. } => "Start",
            AgentEvent::TextDelta { .. } => "TextDelta",
            AgentEvent::Text { .. } => "Text",
            AgentEvent::ThinkingDelta { .. } => "ThinkingDelta",
            AgentEvent::Thinking { .. } => "Thinking",
            AgentEvent::ToolCallStart { .. } => "ToolCallStart",
            AgentEvent::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEvent::TurnComplete { .. } => "TurnComplete",
            AgentEvent::Done { .. } => "Done",
            AgentEvent::AutoRetryStart { .. } => "AutoRetryStart",
            AgentEvent::AutoRetryEnd { .. } => "AutoRetryEnd",
            AgentEvent::Error { .. } => "Error",
            other => panic!("unexpected event kind: {other:?}"),
        })
        .collect()
}

fn assert_contiguous(events: &[crate::journal::committed_event::CommittedEvent]) {
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.sequence, i as u64, "sequence gap at index {i}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 1 — interleaved text + tool-use deltas
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn interleaved_text_and_tool_deltas_suspend_with_balanced_history() -> Result<()> {
    let stores = Stores::new();
    // Text on block 0 arrives interleaved with the tool-use blocks 1 & 2:
    // text chunk, tool-1 start, text chunk, tool-1 input, tool-2 start,
    // tool-2 input — exactly the kind of interleaving a real SSE stream
    // produces when the model narrates between tool calls.
    let provider = StreamingScriptedProvider::single(TurnScript::from_deltas(vec![
        StreamDelta::TextDelta {
            delta: "Let me ".into(),
            block_index: 0,
        },
        StreamDelta::ToolUseStart {
            id: "tc_1".into(),
            name: "get_weather".into(),
            block_index: 1,
            thought_signature: None,
        },
        StreamDelta::TextDelta {
            delta: "check both.".into(),
            block_index: 0,
        },
        StreamDelta::ToolInputDelta {
            id: "tc_1".into(),
            delta: r#"{"city":"Lisbon"}"#.into(),
            block_index: 1,
        },
        StreamDelta::ToolUseStart {
            id: "tc_2".into(),
            name: "get_weather".into(),
            block_index: 2,
            thought_signature: None,
        },
        StreamDelta::ToolInputDelta {
            id: "tc_2".into(),
            delta: r#"{"city":"Porto"}"#.into(),
            block_index: 2,
        },
        StreamDelta::Usage(agent_sdk_foundation::llm::Usage {
            input_tokens: 30,
            output_tokens: 20,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }),
        StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        },
    ]));

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Suspended {
        child_tasks,
        committed_events,
        ..
    } = outcome
    else {
        panic!("expected Suspended with two tool calls");
    };

    // Two child tool tasks spawned, in block order.
    assert_eq!(child_tasks.len(), 2);

    // The suspend outcome carries the admission + narration text + a
    // ToolCallStart per tool: UserInput, Start, Text, ToolCallStart×2.
    assert_eq!(
        event_kinds(&committed_events),
        vec![
            "UserInput",
            "Start",
            "Text",
            "ToolCallStart",
            "ToolCallStart"
        ],
    );

    // The journal additionally carries the two streamed TextDelta
    // chunks (block 0 arrived in two fragments, interleaved with the
    // tool blocks) before the single consolidated Text; sequences are
    // contiguous.
    let journal = stores.events.get_events(&thread_id()).await?;
    assert_eq!(
        event_kinds(&journal),
        vec![
            "UserInput",
            "Start",
            "TextDelta",
            "TextDelta",
            "Text",
            "ToolCallStart",
            "ToolCallStart",
        ],
    );
    assert_contiguous(&journal);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 2 — out-of-order block_index
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn out_of_order_block_index_sorts_into_block_order() -> Result<()> {
    let stores = Stores::new();
    // Deltas arrive with block 1's tool before block 0's text. The
    // accumulator must sort the synthesized content back into block
    // order so the persisted history is API-valid.
    let provider = StreamingScriptedProvider::single(TurnScript::from_deltas(vec![
        StreamDelta::ToolUseStart {
            id: "tc_late".into(),
            name: "get_weather".into(),
            block_index: 1,
            thought_signature: None,
        },
        StreamDelta::ToolInputDelta {
            id: "tc_late".into(),
            delta: r#"{"city":"Madrid"}"#.into(),
            block_index: 1,
        },
        StreamDelta::TextDelta {
            delta: "Checking Madrid.".into(),
            block_index: 0,
        },
        StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        },
    ]));

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Suspended {
        child_tasks,
        committed_events,
        ..
    } = outcome
    else {
        panic!("expected Suspended");
    };
    assert_eq!(child_tasks.len(), 1);

    // Despite out-of-order arrival the suspend events stay in block
    // order: the text (block 0) precedes the ToolCallStart (block 1).
    assert_eq!(
        event_kinds(&committed_events),
        vec!["UserInput", "Start", "Text", "ToolCallStart"],
    );
    assert_contiguous(&stores.events.get_events(&thread_id()).await?);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 3 — mid-stream ServerError / RateLimited trigger a retry
// ─────────────────────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn mid_stream_server_error_retries_then_succeeds() -> Result<()> {
    let stores = Stores::new();
    // First attempt: stream one text delta, then die with a 5xx-class
    // error mid-stream. Second attempt: succeed cleanly. The recoverable
    // error must drive exactly one retry; the turn then completes.
    let first = TurnScript::from_deltas(vec![
        StreamDelta::TextDelta {
            delta: "partial".into(),
            block_index: 0,
        },
        StreamDelta::TextDelta {
            delta: " more".into(),
            block_index: 0,
        },
        StreamDelta::Done {
            stop_reason: Some(StopReason::EndTurn),
        },
    ])
    .fail_after(1, StreamErrorKind::ServerError, "upstream 503 mid-stream");
    let second = TurnScript::text("recovered answer");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after retry");
    };
    assert_eq!(response_text, "recovered answer");

    // Two attempts recorded (one failed mid-stream, one succeeded) and
    // an auto-retry envelope landed in the journal.
    let kinds = event_kinds(&stores.events.get_events(&thread_id()).await?);
    assert!(
        kinds.contains(&"AutoRetryStart"),
        "expected AutoRetryStart in {kinds:?}",
    );
    assert!(
        kinds.contains(&"AutoRetryEnd"),
        "expected AutoRetryEnd in {kinds:?}",
    );
    // Final, recovered turn closes cleanly.
    assert_eq!(kinds.last(), Some(&"Done"));

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn mid_stream_rate_limit_retries_then_succeeds() -> Result<()> {
    let stores = Stores::new();
    let first = TurnScript::text("ignored").fail_after(
        0,
        StreamErrorKind::RateLimited(None),
        "429 before any token",
    );
    let second = TurnScript::text("after backoff");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after rate-limit retry");
    };
    assert_eq!(response_text, "after backoff");

    let attempts = stores
        .attempts
        .list_by_task(&acquire_existing_task_id(&stores).await?)
        .await?;
    assert_eq!(attempts.len(), 2, "one rate-limited attempt + one success");

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn failed_attempt_usage_is_billed_and_folded_into_the_turn() -> Result<()> {
    // Attempt 1 streams a text delta and its usage (100/50) and THEN hits a
    // recoverable rate limit; attempt 2 succeeds with its own usage (100/50).
    // The provider billed both, so:
    //   * attempt 1's audit row must carry its real 100/50 (not zero),
    //   * attempt 2's row bills only its own 100/50 (no double-count), and
    //   * the committed turn total must be 200/100 (both attempts).
    let stores = Stores::new();
    // `text` appends Usage(DEFAULT_USAGE=100/50) then Done; failing after 2
    // deltas injects the error right after that usage — the exact
    // usage-before-error shape this PR made providers emit.
    let first = TurnScript::text("partial").fail_after(
        2,
        StreamErrorKind::RateLimited(None),
        "429 after usage was reported",
    );
    let second = TurnScript::text("after backoff");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed {
        response_text,
        commit,
        ..
    } = outcome
    else {
        panic!("expected Completed after rate-limit retry");
    };
    assert_eq!(response_text, "after backoff");

    // Turn total (thread cumulative) includes BOTH attempts.
    assert_eq!(
        commit.thread.total_usage.input_tokens, 200,
        "turn total must bill both attempts' input tokens",
    );
    assert_eq!(
        commit.thread.total_usage.output_tokens, 100,
        "turn total must bill both attempts' output tokens",
    );

    // Per-attempt audit rows: each bills its OWN tokens exactly once.
    let attempts = stores
        .attempts
        .list_by_task(&acquire_existing_task_id(&stores).await?)
        .await?;
    assert_eq!(attempts.len(), 2, "one rate-limited attempt + one success");
    assert_eq!(
        attempts[0].input_tokens,
        Some(100),
        "the failed attempt's row must carry the tokens it streamed, not zero",
    );
    assert_eq!(attempts[0].output_tokens, Some(50));
    assert_eq!(
        attempts[1].input_tokens,
        Some(100),
        "the successful attempt's row bills only its own tokens (no fold)",
    );
    assert_eq!(attempts[1].output_tokens, Some(50));

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn mid_stream_rate_limit_waits_for_the_provider_hint() -> Result<()> {
    // The exponential schedule caps at 8s; a 30s server hint must win, so the
    // worker does not retry into a guaranteed second 429.
    let stores = Stores::new();
    let first = TurnScript::text("ignored").fail_after(
        0,
        StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(30))),
        "429 with Retry-After: 30",
    );
    let second = TurnScript::text("after the hint");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let started = tokio::time::Instant::now();
    let outcome = run_turn(&stores, &provider).await?;
    let waited = started.elapsed();

    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after rate-limit retry");
    };
    assert_eq!(response_text, "after the hint");
    assert!(
        waited >= std::time::Duration::from_secs(30),
        "the retry must wait out the server's hint, waited {waited:?}"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 3b — a silent stream trips the stall budgets and retries
// ─────────────────────────────────────────────────────────────────────

// Regression for the 2026-07-11 incident: a request written to a
// half-open connection yields no events, no error, and no journal
// activity — before STREAM_FIRST_EVENT_TIMEOUT the poll loop hung
// forever and an external watchdog eventually killed the whole task
// tree. The stall must instead surface as a recoverable error that the
// normal retry path absorbs.
#[tokio::test(start_paused = true)]
async fn first_event_stall_is_retried_then_succeeds() -> Result<()> {
    let stores = Stores::new();
    // Attempt 1: the script sleeps 400s (virtual) before its first
    // delta — past the 330s first-event stall budget, so nothing is
    // ever delivered. Attempt 2: clean success.
    let first = TurnScript::text("never delivered").with_delay(std::time::Duration::from_secs(400));
    let second = TurnScript::text("recovered answer");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after first-event stall retry");
    };
    assert_eq!(response_text, "recovered answer");

    let attempts = stores
        .attempts
        .list_by_task(&acquire_existing_task_id(&stores).await?)
        .await?;
    assert_eq!(attempts.len(), 2, "one stalled attempt + one success");

    // The stall rides the standard recoverable path: an auto-retry
    // envelope lands in the journal (this commit is also what keeps
    // consumer-side stall watchdogs from declaring the thread dead).
    let kinds = event_kinds(&stores.events.get_events(&thread_id()).await?);
    assert!(
        kinds.contains(&"AutoRetryStart"),
        "expected AutoRetryStart in {kinds:?}",
    );
    assert_eq!(kinds.last(), Some(&"Done"));

    Ok(())
}

#[tokio::test(start_paused = true)]
async fn mid_stream_stall_is_retried_then_succeeds() -> Result<()> {
    let stores = Stores::new();
    // Attempt 1: every delta is gated behind a uniform 200s delay. The
    // first delta lands (200s < 330s first-event budget) and latches
    // the tighter inter-event budget; the gap before the second delta
    // (another 200s > 120s) then trips the mid-stream stall. Attempt 2:
    // clean success.
    let first = TurnScript::from_deltas(vec![
        StreamDelta::TextDelta {
            delta: "partial".into(),
            block_index: 0,
        },
        StreamDelta::TextDelta {
            delta: " never lands".into(),
            block_index: 0,
        },
        StreamDelta::Done {
            stop_reason: Some(StopReason::EndTurn),
        },
    ])
    .with_delay(std::time::Duration::from_secs(200));
    let second = TurnScript::text("recovered answer");
    let provider = StreamingScriptedProvider::new(vec![first, second]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after mid-stream stall retry");
    };
    assert_eq!(response_text, "recovered answer");

    let attempts = stores
        .attempts
        .list_by_task(&acquire_existing_task_id(&stores).await?)
        .await?;
    assert_eq!(attempts.len(), 2, "one stalled attempt + one success");

    Ok(())
}

/// The single root task's id (there is exactly one in these tests).
async fn acquire_existing_task_id(stores: &Stores) -> Result<crate::journal::task::AgentTaskId> {
    let tasks = stores.tasks.list_by_thread(&thread_id()).await?;
    tasks
        .into_iter()
        .next()
        .map(|t| t.id)
        .context("a root task should exist")
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 4 — stream ends without a Done delta
// ─────────────────────────────────────────────────────────────────────

// Regression for finding #8: a stream that ends without a completion
// marker is a *truncated* response, not a successful turn. The worker
// must NOT commit it as Completed — it retries, and a clean follow-up
// attempt completes the turn.
#[tokio::test(start_paused = true)]
async fn stream_ends_without_done_is_retried_then_completes() -> Result<()> {
    let stores = Stores::new();
    let provider = StreamingScriptedProvider::new(vec![
        // Attempt 1: text + usage but the connection ends before a
        // `Done`. No stop_reason → treated as truncated → retried.
        TurnScript::from_deltas(vec![
            StreamDelta::TextDelta {
                delta: "no done marker".into(),
                block_index: 0,
            },
            StreamDelta::Usage(agent_sdk_foundation::llm::Usage {
                input_tokens: 5,
                output_tokens: 4,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            }),
            // no StreamDelta::Done
        ]),
        // Attempt 2: a complete, well-formed text turn.
        TurnScript::text("complete answer"),
    ]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed after the truncated stream is retried");
    };
    assert_eq!(response_text, "complete answer");

    // The retry envelope is visible in the journal, and the final turn
    // closes cleanly with the second (complete) attempt's content.
    let journal = stores.events.get_events(&thread_id()).await?;
    let kinds = event_kinds(&journal);
    assert!(
        kinds.contains(&"AutoRetryStart"),
        "expected an AutoRetryStart from the truncated-stream retry: {kinds:?}"
    );
    assert!(
        kinds.contains(&"AutoRetryEnd"),
        "expected an AutoRetryEnd after the retry succeeded: {kinds:?}"
    );
    assert_eq!(
        kinds.last().copied(),
        Some("Done"),
        "the completed turn must still close with Done: {kinds:?}"
    );
    assert_contiguous(&journal);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 5 — stream truncated after ToolUseStart, before ToolInputDelta
// ─────────────────────────────────────────────────────────────────────

// Regression for finding #8: a stream truncated after `ToolUseStart`
// (before any `ToolInputDelta` or `Done`) has no completion marker, so
// the worker must retry rather than spawn a child task with an empty
// `{}` input. A clean follow-up attempt then suspends correctly.
#[tokio::test(start_paused = true)]
async fn truncated_tool_stream_is_retried_then_suspends() -> Result<()> {
    let stores = Stores::new();
    let provider = StreamingScriptedProvider::new(vec![
        // Attempt 1: announces a tool use then dies before any input
        // JSON and before `Done`. No stop_reason → truncated → retried.
        TurnScript::from_deltas(vec![StreamDelta::ToolUseStart {
            id: "tc_trunc".into(),
            name: "get_weather".into(),
            block_index: 0,
            thought_signature: None,
        }]),
        // Attempt 2: a complete tool call with real input JSON.
        TurnScript::tool_calls(&[("tc_ok", "get_weather", serde_json::json!({"city": "SF"}))]),
    ]);

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Suspended {
        child_tasks,
        committed_events,
        ..
    } = outcome
    else {
        panic!("expected Suspended after the truncated tool stream is retried");
    };
    assert_eq!(child_tasks.len(), 1);

    // The spawned child carries the second (complete) attempt's
    // well-formed input — never the truncated attempt's empty `{}`.
    let start = committed_events
        .iter()
        .find_map(|e| match &e.event {
            AgentEvent::ToolCallStart {
                id, input, name, ..
            } => Some((id.clone(), name.clone(), input.clone())),
            _ => None,
        })
        .context("a ToolCallStart event")?;
    assert_eq!(start.0, "tc_ok");
    assert_eq!(start.1, "get_weather");
    assert_eq!(start.2, serde_json::json!({"city": "SF"}));

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 6 — per-delta delay under virtual time stays deterministic
// ─────────────────────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn chunked_text_with_per_delta_delay_commits_in_order() -> Result<()> {
    let stores = Stores::new();
    // Multiple text chunks on one block, each gated behind a per-delta
    // delay. Under `start_paused` virtual time auto-advances, so the
    // turn completes instantly yet the per-chunk commit ordering is
    // exercised exactly as it would be against a slow real stream.
    let provider = StreamingScriptedProvider::single(
        TurnScript::text_chunked(&["Strea", "ming ", "answer"])
            .with_delay(std::time::Duration::from_millis(25)),
    );

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed");
    };
    assert_eq!(response_text, "Streaming answer");

    // Each non-empty chunk is committed as its own TextDelta before the
    // single consolidated Text block.
    let journal = stores.events.get_events(&thread_id()).await?;
    assert_eq!(
        event_kinds(&journal),
        vec![
            "UserInput",
            "Start",
            "TextDelta",
            "TextDelta",
            "TextDelta",
            "Text",
            "TurnComplete",
            "Done",
        ],
    );
    assert_contiguous(&journal);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 7 — streamed thinking + text and thinking + tool
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn streamed_thinking_then_text_commits_both_blocks() -> Result<()> {
    let stores = Stores::new();
    let provider = StreamingScriptedProvider::single(TurnScript::thinking_text(
        "weighing options",
        "sig-abc",
        "final answer",
    ));

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Completed { response_text, .. } = outcome else {
        panic!("expected Completed");
    };
    assert_eq!(response_text, "final answer");

    let journal = stores.events.get_events(&thread_id()).await?;
    // Both deltas stream before both consolidated blocks.
    assert_eq!(
        event_kinds(&journal),
        vec![
            "UserInput",
            "Start",
            "ThinkingDelta",
            "TextDelta",
            "Thinking",
            "Text",
            "TurnComplete",
            "Done",
        ],
    );
    assert_contiguous(&journal);

    Ok(())
}

#[tokio::test]
async fn streamed_thinking_then_tool_suspends_in_block_order() -> Result<()> {
    let stores = Stores::new();
    let provider = StreamingScriptedProvider::single(TurnScript::thinking_tool_calls(
        "I should look this up",
        "sig-think",
        &[(
            "tc_think",
            "get_weather",
            serde_json::json!({"city": "Oslo"}),
        )],
    ));

    let outcome = run_turn(&stores, &provider).await?;
    let RootTurnOutcome::Suspended {
        child_tasks,
        committed_events,
        ..
    } = outcome
    else {
        panic!("expected Suspended");
    };
    assert_eq!(child_tasks.len(), 1);
    // Consolidated thinking precedes the tool-call boundary.
    assert_eq!(
        event_kinds(&committed_events),
        vec!["UserInput", "Start", "Thinking", "ToolCallStart"],
    );
    assert_contiguous(&stores.events.get_events(&thread_id()).await?);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Edge case 8 — leading fatal stream error does not retry
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn leading_invalid_request_error_fails_without_retry() -> Result<()> {
    let stores = Stores::new();
    // A fatal (caller-side) error on the very first delta: no retry, no
    // durable turn, exactly one closed audit attempt.
    let provider = StreamingScriptedProvider::single(TurnScript::error(
        StreamErrorKind::InvalidRequest,
        "schema rejected",
    ));

    let err = run_turn(&stores, &provider)
        .await
        .expect_err("a fatal stream error must surface as a turn failure");
    assert!(
        format!("{err:#}").contains("schema rejected"),
        "error should propagate the provider message: {err:#}",
    );

    let task_id = acquire_existing_task_id(&stores).await?;
    let attempts = stores.attempts.list_by_task(&task_id).await?;
    assert_eq!(attempts.len(), 1, "fatal error opens exactly one attempt");

    let thread = stores
        .threads
        .get(&thread_id())
        .await?
        .context("thread row")?;
    assert_eq!(
        thread.committed_turns, 0,
        "no durable turn on a fatal error"
    );

    Ok(())
}
