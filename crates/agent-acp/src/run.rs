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
//! - **Duplicate suppression** — replay/live handoffs may overlap; events at
//!   or below the last yielded sequence are dropped.
//! - **Consolidated-event dedupe** — `Text` following its own `TextDelta`s
//!   is skipped, or every message would render twice.
//! - **Cancel forwarding** — `session/cancel` cancels the token; the loop
//!   signals the backend once and keeps draining until the terminal
//!   `Cancelled` event commits. Cancellation is an edge, not an exit.
//!
//! Milestone scope (M0.2): text deltas + terminals only. Thinking, tool
//! calls, subagent lifecycle, and plan synthesis land in M2 (ENG-9404…9406);
//! lag-reopen lands in M1.4 (ENG-9402) — until then `Lagged` fails the
//! prompt loudly rather than skipping events silently.

use std::collections::HashSet;

use agent_sdk_foundation::AgentEvent;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::backend::{AcpBackend, AcpRunHandle, RunStreamItem};
use crate::server::{PromptError, UpdateSink};
use crate::wire::StopReason;

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

/// Map one event under the M0.2 subset.
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

/// Drive one submitted turn to completion.
pub(crate) async fn run_prompt<B: AcpBackend + ?Sized>(
    backend: &B,
    handle: &AcpRunHandle,
    updates: &UpdateSink,
    cancel: &CancellationToken,
) -> Result<StopReason, PromptError> {
    // Contract C-c: the ONLY cursor arithmetic in the transport.
    let after_sequence = handle.first_event_sequence.checked_sub(1);
    let mut events = backend
        .open_events(&handle.thread_id, after_sequence)
        .await
        .map_err(|e| PromptError::new(e.message))?;

    let mut last_yielded: Option<u64> = None;
    let mut delta_seen: HashSet<String> = HashSet::new();
    let mut cancel_forwarded = false;

    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled(), if !cancel_forwarded => {
                cancel_forwarded = true;
                if let Err(e) = backend.cancel(&handle.thread_id, &handle.task_id).await {
                    log::warn!("acp: backend cancel failed (draining anyway): {e}");
                }
                // Keep draining: the terminal `Cancelled` event closes the turn.
            }
            item = events.next() => {
                let Some(item) = item else {
                    return Err(PromptError::new(
                        "event stream ended without a terminal event",
                    ));
                };
                match item {
                    RunStreamItem::Lagged => {
                        return Err(PromptError::new(
                            "event stream lagged; reopen lands in ENG-9402 — failing loudly \
                             rather than skipping events",
                        ));
                    }
                    RunStreamItem::Event(ev) => {
                        if last_yielded.is_some_and(|last| ev.sequence <= last) {
                            continue;
                        }
                        last_yielded = Some(ev.sequence);
                        match map_event(&ev.event, &mut delta_seen) {
                            Mapped::Update(update) => {
                                updates
                                    .session_update(update)
                                    .await
                                    .map_err(|e| PromptError::new(e.to_string()))?;
                            }
                            Mapped::Terminal(reason) => return Ok(reason),
                            Mapped::Fail(message) => return Err(PromptError::new(message)),
                            Mapped::Ignore => {}
                        }
                    }
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
}
