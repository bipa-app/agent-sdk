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
//! - **Content attribution (ENG-9422)** — content events stamped with a
//!   foreign `emitter_task_id` are dropped: a cancelled predecessor's
//!   salvage flush can land deltas after OUR `Start`, and without the
//!   attribution check they would render inside our answer. Unattributed
//!   content (pre-attribution journals) still streams.
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
//! - **Write-keyed keepalives** — the keepalive deadline resets only when
//!   an outbound frame is written to the client. Stream items that map to
//!   nothing (drop-listed, foreign, pre-`Start`) leave it alone; buzz-acp
//!   times out on stdout idle, so read-side activity proves nothing.
//! - **Single resolution, by construction** — the prompt's outcome IS this
//!   function's return value. There is no out-of-band resolution channel,
//!   callback, or shared cell that could fire twice; every terminal path
//!   is a `return`.
//!
//! Mapper scope: text/thinking deltas, consolidated-content dedupe, tool
//! lifecycle, turn usage, keepalives, and terminals. Subagent, plan, and
//! permission mapping land in later slices.

use std::pin::Pin;
use std::time::Duration;

use agent_sdk_foundation::AgentEvent;
use futures::StreamExt;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::backend::{AcpBackend, AcpRunHandle, BackendTaskStatus, RunStreamItem};
use crate::mapper::{EventMapper, Mapped};
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

/// Emit activity well inside buzz-acp's idle window while the CLIENT sees
/// nothing. Only writing an outbound frame postpones this deadline: buzz-acp
/// kills turns on stdout idle, so stream items that map to nothing
/// (drop-listed retries, foreign traffic, pre-`Start` events) must not push
/// it back — a chatty-but-unmapped backend would otherwise starve stdout.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);

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

/// Resolve the prompt from the task's durable STATUS (no journal event).
fn resolve_from_status(status: BackendTaskStatus) -> Result<StopReason, PromptError> {
    match status {
        BackendTaskStatus::Completed => Ok(StopReason::EndTurn),
        BackendTaskStatus::Cancelled => Ok(StopReason::Cancelled),
        BackendTaskStatus::Failed { error } => resolve_failed_status(error),
        BackendTaskStatus::Running => Err(PromptError::new(
            "resolve_from_status called with a non-terminal status (loop bug)",
        )),
    }
}

fn resolve_failed_status(error: Option<String>) -> Result<StopReason, PromptError> {
    let Some(error) = error else {
        return Err(PromptError::new("task failed without a recorded error"));
    };
    Err(PromptError::new(error))
}

/// An unattributed terminal arrived while streaming. Only the submitted
/// task's durable status may close the turn; `None` keeps streaming.
async fn reconcile_unattributed<B: AcpBackend + ?Sized>(
    backend: &B,
    handle: &AcpRunHandle,
) -> Result<Option<StopReason>, PromptError> {
    match backend
        .task_status(&handle.thread_id, &handle.task_id)
        .await
    {
        Ok(BackendTaskStatus::Running) => Ok(None),
        Ok(status) => resolve_from_status(status).map(Some),
        Err(error) => {
            log::warn!(
                "acp: status probe after an unattributed terminal failed \
                 (stall poll will retry): {error}"
            );
            Ok(None)
        }
    }
}

async fn handle_event<B: AcpBackend + ?Sized>(
    backend: &B,
    handle: &AcpRunHandle,
    updates: &UpdateSink,
    keepalive: Pin<&mut tokio::time::Sleep>,
    phase: &mut Phase,
    mapper: &mut EventMapper,
    event: &AgentEvent,
) -> Result<Option<StopReason>, PromptError> {
    let is_ours = event.emitter_task_id() == Some(handle.task_id.as_str());
    let is_foreign = event
        .emitter_task_id()
        .is_some_and(|task_id| task_id != handle.task_id);

    if matches!(event, AgentEvent::Start { .. }) {
        if is_ours && *phase == Phase::AwaitingStart {
            *phase = Phase::Streaming;
        }
        return Ok(None);
    }

    // Drop foreign content before mapping so its message id cannot poison the
    // consolidated-content dedupe state for the submitted task.
    if is_foreign {
        return Ok(None);
    }

    match mapper.map(event) {
        Mapped::Update(update) => {
            if *phase == Phase::Streaming {
                updates
                    .session_update(update)
                    .await
                    .map_err(|error| PromptError::new(error.to_string()))?;
                // An outbound frame reached the client — THAT is what
                // postpones the keepalive, not reading a stream item.
                reset_keepalive(keepalive);
            }
            Ok(None)
        }
        Mapped::Terminal(reason) => {
            if is_ours {
                return Ok(Some(reason));
            }
            if *phase == Phase::AwaitingStart {
                return Ok(None);
            }
            reconcile_unattributed(backend, handle).await
        }
        Mapped::Fail(message) => {
            if is_ours {
                return Err(PromptError::new(message));
            }
            if *phase == Phase::AwaitingStart {
                return Ok(None);
            }
            reconcile_unattributed(backend, handle).await
        }
        Mapped::Ignore => Ok(None),
    }
}

fn reset_keepalive(keepalive: Pin<&mut tokio::time::Sleep>) {
    keepalive.reset(tokio::time::Instant::now() + KEEPALIVE_INTERVAL);
}

async fn emit_keepalive(updates: &UpdateSink) -> Result<(), PromptError> {
    updates
        .session_update(json!({ "sessionUpdate": "keepalive" }))
        .await
        .map_err(|error| PromptError::new(error.to_string()))
}

/// Drive one submitted turn to completion.
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
    let mut mapper = EventMapper::default();
    let mut cancel_forwarded = false;
    let mut phase = Phase::AwaitingStart;
    let mut terminal_status_streak: u32 = 0;

    let mut stall = tokio::time::interval(STALL_POLL_INTERVAL);
    stall.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    stall.tick().await; // interval's first tick is immediate — skip it.

    let keepalive = tokio::time::sleep(KEEPALIVE_INTERVAL);
    tokio::pin!(keepalive);

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
                    RunStreamItem::Event(event) => {
                        if last_yielded.is_some_and(|last| event.sequence <= last) {
                            continue;
                        }
                        last_yielded = Some(event.sequence);
                        // Stream progress resets the grace for an in-flight
                        // terminal journal commit.
                        terminal_status_streak = 0;
                        if let Some(reason) = handle_event(
                            backend,
                            handle,
                            updates,
                            keepalive.as_mut(),
                            &mut phase,
                            &mut mapper,
                            &event.event,
                        )
                        .await?
                        {
                            return Ok(reason);
                        }
                    }
                }
            }
            () = &mut keepalive => {
                emit_keepalive(updates).await?;
                reset_keepalive(keepalive.as_mut());
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
        let error = resolve_from_status(BackendTaskStatus::Failed {
            error: Some("boom".to_owned()),
        });
        let Err(error) = error else {
            panic!("failed status must be a prompt error");
        };
        assert!(error.message.contains("boom"));

        let error = resolve_from_status(BackendTaskStatus::Failed { error: None });
        let Err(error) = error else {
            panic!("failed status must be a prompt error");
        };
        assert!(error.message.contains("without a recorded error"));
    }
}
