//! The SDK-owned run loop: consumes a backend event stream and drives one
//! ACP prompt to its stop reason.
//!
//! Everything correctness-sensitive lives HERE, once, for every backend
//! (review finding #9 in the satoshi design docs):
//!
//! - **Cursor conversion (contract C-c)** — [`AcpRunHandle`] carries the
//!   sequence the turn's first event WILL receive; the stream contract is
//!   strictly-greater-than. The `checked_sub(1)` below is the only place in
//!   the codebase that arithmetic happens. A handle whose first event is
//!   sequence 0 opens the stream from the beginning (`None`).
//! - **Lag-reopen (C-c)** — a `Lagged` stream is reopened from the last
//!   yielded sequence: gapless by the strictly-greater contract, duplicate
//!   free by the suppression below. A `RetentionGap` is fatal instead: the
//!   journal pruned events this turn needed, so continuity cannot be
//!   proven. Backends REPORT both; recovery policy lives here.
//! - **Task-scoped completion (C-d)** — the turn is a two-phase machine:
//!   `AwaitingStart` (nothing streams; only OUR task's `Start` advances)
//!   then `Streaming`. A terminal event resolves the prompt only when its
//!   `emitter_task_id` is OUR task; foreign terminals (a stale
//!   predecessor's late `Done`, a cancelled root's salvage commit) are
//!   ignored, and UNATTRIBUTED terminals (pre-attribution journals) are
//!   reconciled against the backend's task status before they may close
//!   anything. One deliberate asymmetry: an OUR-attributed terminal
//!   resolves even in `AwaitingStart` — a queued turn cancelled before it
//!   ever started emits `Cancelled` without a `Start`, and per the
//!   attribution rules an event naming our task can never be a
//!   predecessor's.
//! - **Stall reconciliation (C-d)** — a task can die without committing a
//!   terminal event (worker crash after status write, journal loss). A
//!   bounded stall poll asks the backend for OUR task's durable status
//!   whenever the stream goes quiet; two consecutive terminal readings
//!   (grace for an in-flight journal commit) resolve the prompt from the
//!   STATUS — including `Failed` with the task's recorded error.
//! - **Duplicate suppression** — replay/live/reopen handoffs may overlap;
//!   events at or below the last yielded sequence are dropped.
//! - **Consolidated-event dedupe** — `Text` following its own `TextDelta`s
//!   is skipped, or every message would render twice.
//! - **Cancel forwarding** — `session/cancel` cancels the token; the loop
//!   signals the backend once and keeps draining until OUR terminal
//!   commits. Cancellation is an edge, not an exit.
//! - **Single resolution, by construction** — the prompt's outcome IS this
//!   function's return value. There is no out-of-band resolution channel,
//!   callback, or shared cell that could fire twice; every terminal path
//!   is a `return`.
//!
//! Milestone scope: text deltas + terminals. Thinking, tool calls,
//! subagent lifecycle, and plan synthesis land in M2 (ENG-9404…9406).

use std::collections::HashSet;
use std::time::Duration;

use agent_sdk_foundation::AgentEvent;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::backend::{AcpBackend, AcpRunHandle, BackendTaskStatus, RunStreamItem};
use crate::server::{PromptError, UpdateSink};
use crate::wire::StopReason;

/// How often the loop probes the backend for OUR task's durable status
/// while the event stream is quiet. Cheap (one store read) and only load
/// bearing when a task dies without committing a terminal event.
const STALL_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Consecutive terminal status readings required before the loop resolves
/// from STATUS instead of a journal event. The gap between readings is the
/// grace window for a just-committed terminal event still in flight on the
/// stream — progress on the stream resets the streak.
const TERMINAL_STATUS_CONFIRMATIONS: u32 = 2;

/// Backoff before reopening a lagged stream, so a hot broadcast channel
/// cannot spin the loop through instant lag-reopen cycles.
const LAG_REOPEN_DELAY: Duration = Duration::from_millis(200);

/// Outcome of mapping a single committed event.
#[derive(Debug, PartialEq, Eq)]
enum Mapped {
    /// Emit a `session/update` payload.
    Update(Value),
    /// The turn is over with this stop reason.
    Terminal(StopReason),
    /// The turn failed; resolve the prompt as a JSON-RPC error.
    Fail(String),
    /// Not mapped in this milestone.
    Ignore,
}

/// The turn's completion machine (contract C-d). Resolution is not a
/// state: resolving IS returning from [`run_prompt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Before OUR task's `Start`: late predecessor traffic may still be
    /// flowing on the thread, so nothing streams and unattributed
    /// terminals are categorically stale.
    AwaitingStart,
    /// After OUR task's `Start`: content streams; terminals resolve per
    /// the attribution rules.
    Streaming,
}

/// Map one event to its wire effect.
///
/// `delta_seen` tracks message ids that streamed as deltas, so their
/// consolidated `Text` form is not re-emitted.
fn map_event(event: &AgentEvent, delta_seen: &mut HashSet<String>) -> Mapped {
    match event {
        AgentEvent::TextDelta { message_id, delta } => {
            delta_seen.insert(message_id.clone());
            Mapped::Update(chunk(delta))
        }
        AgentEvent::Text { message_id, text } => {
            if delta_seen.contains(message_id) {
                Mapped::Ignore
            } else {
                Mapped::Update(chunk(text))
            }
        }
        AgentEvent::Done { .. } => Mapped::Terminal(StopReason::EndTurn),
        AgentEvent::Cancelled { .. } => Mapped::Terminal(StopReason::Cancelled),
        AgentEvent::Refusal { .. } => Mapped::Terminal(StopReason::Refusal),
        AgentEvent::BudgetExceeded { .. } => Mapped::Terminal(StopReason::MaxTokens),
        AgentEvent::Error { message, .. } => Mapped::Fail(message.clone()),
        // Thinking/tool/subagent/plan/etc.: deliberately unmapped until M2.
        _ => Mapped::Ignore,
    }
}

fn chunk(text: &str) -> Value {
    json!({
        "sessionUpdate": "agent_message_chunk",
        "content": { "type": "text", "text": text },
    })
}

/// Resolve the prompt from the task's durable STATUS (no journal event).
fn resolve_from_status(status: BackendTaskStatus) -> Result<StopReason, PromptError> {
    match status {
        BackendTaskStatus::Completed => Ok(StopReason::EndTurn),
        BackendTaskStatus::Cancelled => Ok(StopReason::Cancelled),
        BackendTaskStatus::Failed { error } => {
            Err(PromptError::new(error.unwrap_or_else(|| {
                "task failed without a recorded error".to_owned()
            })))
        }
        BackendTaskStatus::Running => Err(PromptError::new(
            "resolve_from_status called with a non-terminal status (loop bug)",
        )),
    }
}

/// An UNATTRIBUTED terminal (or error) arrived while streaming: only OUR
/// task's durable status may close the turn (contract C-d). `None` means
/// the event was not ours — keep streaming.
async fn reconcile_unattributed<B: AcpBackend + ?Sized>(
    backend: &B,
    handle: &AcpRunHandle,
) -> Option<Result<StopReason, PromptError>> {
    match backend
        .task_status(&handle.thread_id, &handle.task_id)
        .await
    {
        Ok(BackendTaskStatus::Running) => None,
        Ok(status) => Some(resolve_from_status(status)),
        Err(e) => {
            log::warn!(
                "acp: status probe after an unattributed terminal failed \
                 (stall poll will retry): {e}"
            );
            None
        }
    }
}

/// Drive one submitted turn to completion.
// One linear select loop: splitting it further would scatter the C-c/C-d
// state machine across helpers (same call as the provider impls).
#[allow(clippy::too_many_lines)]
pub(crate) async fn run_prompt<B: AcpBackend + ?Sized>(
    backend: &B,
    handle: &AcpRunHandle,
    updates: &UpdateSink,
    cancel: &CancellationToken,
) -> Result<StopReason, PromptError> {
    // Contract C-c: the ONLY cursor arithmetic in the transport.
    let initial_after = handle.first_event_sequence.checked_sub(1);
    let mut events = backend
        .open_events(&handle.thread_id, initial_after)
        .await
        .map_err(|e| PromptError::new(e.message))?;

    let mut last_yielded: Option<u64> = None;
    let mut delta_seen: HashSet<String> = HashSet::new();
    let mut cancel_forwarded = false;
    let mut phase = Phase::AwaitingStart;
    let mut terminal_status_streak: u32 = 0;

    let mut stall = tokio::time::interval(STALL_POLL_INTERVAL);
    stall.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    stall.tick().await; // interval's first tick is immediate — skip it.

    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled(), if !cancel_forwarded => {
                cancel_forwarded = true;
                if let Err(e) = backend.cancel(&handle.thread_id, &handle.task_id).await {
                    log::warn!("acp: backend cancel failed (draining anyway): {e}");
                }
                // Keep draining: OUR terminal `Cancelled` event closes the turn.
            }
            item = events.next() => {
                let Some(item) = item else {
                    return Err(PromptError::new(
                        "event stream ended without a terminal event",
                    ));
                };
                match item {
                    RunStreamItem::Lagged => {
                        // C-c: reopen strictly after the last event we
                        // yielded (or the turn's original cursor when
                        // nothing has streamed yet) — gapless by the
                        // stream contract, duplicate-free by suppression.
                        let after = last_yielded.or(initial_after);
                        tokio::time::sleep(LAG_REOPEN_DELAY).await;
                        events = backend
                            .open_events(&handle.thread_id, after)
                            .await
                            .map_err(|e| PromptError::new(format!(
                                "reopen after lag failed: {}", e.message
                            )))?;
                        log::info!(
                            "acp: event stream lagged; reopened after sequence {after:?}"
                        );
                    }
                    RunStreamItem::RetentionGap => {
                        return Err(PromptError::new(
                            "journal retention pruned events this turn needed — \
                             continuity cannot be proven, failing the prompt",
                        ));
                    }
                    RunStreamItem::Event(ev) => {
                        if last_yielded.is_some_and(|last| ev.sequence <= last) {
                            continue;
                        }
                        last_yielded = Some(ev.sequence);
                        // Stream progress: any in-flight journal commit
                        // clearly still flows, so restart the status grace.
                        terminal_status_streak = 0;

                        // Contract C-d: attribution decides everything.
                        let is_ours = ev.event.emitter_task_id() == Some(handle.task_id.as_str());
                        let is_foreign =
                            ev.event.emitter_task_id().is_some_and(|t| t != handle.task_id);

                        if matches!(ev.event, AgentEvent::Start { .. }) {
                            if is_ours && phase == Phase::AwaitingStart {
                                phase = Phase::Streaming;
                            }
                            continue;
                        }

                        match map_event(&ev.event, &mut delta_seen) {
                            Mapped::Update(update) => {
                                // Content events carry no attribution;
                                // the phase gate is what keeps a late
                                // predecessor's text out of OUR prompt.
                                if phase == Phase::Streaming {
                                    updates
                                        .session_update(update)
                                        .await
                                        .map_err(|e| PromptError::new(e.to_string()))?;
                                }
                            }
                            Mapped::Terminal(reason) => {
                                if is_ours {
                                    return Ok(reason);
                                }
                                if is_foreign || phase == Phase::AwaitingStart {
                                    // A predecessor's late terminal (or a
                                    // salvage commit): never ours to act on.
                                    continue;
                                }
                                if let Some(resolution) =
                                    reconcile_unattributed(backend, handle).await
                                {
                                    return resolution;
                                }
                            }
                            Mapped::Fail(message) => {
                                if is_ours {
                                    return Err(PromptError::new(message));
                                }
                                if is_foreign || phase == Phase::AwaitingStart {
                                    continue;
                                }
                                if let Some(resolution) =
                                    reconcile_unattributed(backend, handle).await
                                {
                                    return resolution;
                                }
                            }
                            Mapped::Ignore => {}
                        }
                    }
                }
            }
            _ = stall.tick() => {
                // Bounded stall reconciliation: a task can reach a terminal
                // STATUS without a terminal EVENT (crash between the status
                // write and the journal commit). Status is inherently
                // task-scoped, so this is safe in either phase.
                match backend.task_status(&handle.thread_id, &handle.task_id).await {
                    Ok(BackendTaskStatus::Running) => terminal_status_streak = 0,
                    Ok(status) => {
                        terminal_status_streak += 1;
                        if terminal_status_streak >= TERMINAL_STATUS_CONFIRMATIONS {
                            log::warn!(
                                "acp: task {} reached terminal status with no terminal \
                                 event on the stream — resolving from status",
                                handle.task_id
                            );
                            return resolve_from_status(status);
                        }
                    }
                    Err(e) => log::warn!("acp: stall status probe failed (will retry): {e}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::{ThreadId, TokenUsage};
    use std::time::Duration;

    fn done() -> AgentEvent {
        AgentEvent::Done {
            thread_id: ThreadId::from_string("t".to_owned()),
            total_turns: 1,
            total_usage: TokenUsage::default(),
            duration: Duration::from_millis(1),
            estimated_cost_usd: None,
            emitter_task_id: None,
        }
    }

    #[test]
    fn golden_text_delta_maps_to_agent_message_chunk() {
        let mut seen = HashSet::new();
        let mapped = map_event(&AgentEvent::text_delta("m1", "hel"), &mut seen);
        let Mapped::Update(update) = mapped else {
            panic!("expected update");
        };
        assert_eq!(update["sessionUpdate"], json!("agent_message_chunk"));
        assert_eq!(update["content"]["text"], json!("hel"));
        assert!(seen.contains("m1"));
    }

    #[test]
    fn golden_consolidated_text_dedupes_against_its_deltas() {
        let mut seen = HashSet::new();
        let _ = map_event(&AgentEvent::text_delta("m1", "a"), &mut seen);
        assert_eq!(
            map_event(&AgentEvent::text("m1", "ab"), &mut seen),
            Mapped::Ignore,
            "consolidated Text after its deltas must not re-emit"
        );
        // A message that never streamed deltas emits its full text once.
        let Mapped::Update(update) = map_event(&AgentEvent::text("m2", "whole"), &mut seen) else {
            panic!("expected update");
        };
        assert_eq!(update["content"]["text"], json!("whole"));
    }

    #[test]
    fn golden_terminals_map_to_their_stop_reasons() {
        let mut seen = HashSet::new();
        assert_eq!(
            map_event(&done(), &mut seen),
            Mapped::Terminal(StopReason::EndTurn)
        );
        assert_eq!(
            map_event(
                &AgentEvent::Cancelled {
                    turn: 1,
                    usage: TokenUsage::default(),
                    reason: None,
                    emitter_task_id: None,
                },
                &mut seen
            ),
            Mapped::Terminal(StopReason::Cancelled)
        );
        assert_eq!(
            map_event(&AgentEvent::refusal("m1", None), &mut seen),
            Mapped::Terminal(StopReason::Refusal)
        );
        assert_eq!(
            map_event(
                &AgentEvent::BudgetExceeded {
                    thread_id: ThreadId::from_string("t".to_owned()),
                    total_turns: 1,
                    total_usage: TokenUsage::default(),
                    duration: Duration::from_millis(1),
                    estimated_cost_usd: None,
                    limit: agent_sdk_foundation::types::BudgetLimitKind::TotalTokens,
                    emitter_task_id: None,
                },
                &mut seen
            ),
            Mapped::Terminal(StopReason::MaxTokens)
        );
        assert_eq!(
            map_event(&AgentEvent::error("boom", false), &mut seen),
            Mapped::Fail("boom".to_owned())
        );
    }

    #[test]
    fn golden_out_of_scope_variants_are_ignored_in_m0() {
        let mut seen = HashSet::new();
        assert_eq!(
            map_event(&AgentEvent::thinking_delta("m1", "hmm"), &mut seen),
            Mapped::Ignore,
            "thinking mapping is M2 (ENG-9404)"
        );
        assert_eq!(
            map_event(
                &AgentEvent::tool_call_start(
                    "t1",
                    "grep",
                    "Grep",
                    json!({}),
                    agent_sdk_foundation::ToolTier::Observe,
                ),
                &mut seen
            ),
            Mapped::Ignore,
            "tool mapping is M2 (ENG-9404)"
        );
    }

    #[test]
    fn status_resolution_maps_every_terminal_shape() {
        assert!(matches!(
            resolve_from_status(BackendTaskStatus::Completed),
            Ok(StopReason::EndTurn)
        ));
        assert!(matches!(
            resolve_from_status(BackendTaskStatus::Cancelled),
            Ok(StopReason::Cancelled)
        ));
        let err = resolve_from_status(BackendTaskStatus::Failed {
            error: Some("boom".to_owned()),
        })
        .expect_err("failed status is a prompt error");
        assert!(err.message.contains("boom"));
        let err = resolve_from_status(BackendTaskStatus::Failed { error: None })
            .expect_err("failed status is a prompt error");
        assert!(err.message.contains("without a recorded error"));
    }
}
