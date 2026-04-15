//! Cross-instance thread-event watch-handler contract.
//!
//! Phase 8.4 (ENG-7968) consumes the `thread_events_available` advisory
//! messages that Phase 8.1 writes and Phase 8.2 publishes.  Every
//! advisory is **only a hint** that a thread advanced; the authoritative
//! stream is [`EventRepository`] and the committed-event table behind
//! it.  A handler never trusts the broker payload as event data — it
//! only carries `{ thread_id, last_sequence }`, and the handler
//! resolves the actual events by replaying the durable log.
//!
//! # The contract in one picture
//!
//! ```text
//!   ┌──────────── broker delivery ────────────┐
//!   │ { thread_id: ..., last_sequence: ... } │   (advisory only)
//!   └──────────────────┬──────────────────────┘
//!                      │ handle_payload()
//!                      ▼
//!             ┌────────────────────────┐
//!             │ EventRepository.get*   │   ← authoritative replay
//!             └──────────┬─────────────┘
//!                        │
//!                        ▼
//!            ┌────────────────────────┐
//!            │ EventNotifier.notify   │   ← nudge local subscribers
//!            └────────────────────────┘
//! ```
//!
//! The nudge itself just feeds the durable events back into the
//! thread's in-process broadcast channel so any
//! [`EventStream`](super::event_stream::EventStream) currently parked on
//! the live tail wakes up and resumes.  Subscribers filter duplicates
//! against their own `last_yielded` cursor, so an advisory that overlaps
//! with a client's own replay cannot produce a duplicate event on the
//! wire.
//!
//! # Duplicate / out-of-order safety
//!
//! A handler keeps a per-thread high-water mark of the highest
//! sequence it has already forwarded.  Three things fall out of it:
//!
//! 1. **Duplicate advisory** (same `last_sequence` twice) — the second
//!    delivery lands with `last_sequence <= high_water` and is reported
//!    as [`ThreadEventsWatchOutcome::AlreadyCurrent`] without touching
//!    the notifier.
//! 2. **Out-of-order advisory** (`last_sequence = 5` arriving after
//!    `last_sequence = 9`) — same path; the late advisory observes a
//!    higher high-water and short-circuits.
//! 3. **Replication lag** — if the durable replica cannot yet see
//!    events up to `last_sequence`, the handler emits only what it
//!    finds and leaves the high-water at the actual read, so a later
//!    redelivery (after replication catches up) still fires.  The
//!    journal is always the authority.
//!
//! # Unknown / pruned threads
//!
//! An advisory for a thread that returns zero events from the
//! repository (never existed, retention-pruned, not yet visible on
//! this replica) is reported as
//! [`ThreadEventsWatchOutcome::UnknownThread`].  Callers ack these
//! without requeue — the broker is permitted to republish and the
//! repository is always the authority.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::event_notifier::EventNotifier;
use super::event_repository::EventRepository;
use super::outbox_message::ThreadEventsAvailablePayload;

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// What a re-check of the durable journal found for a
/// `thread_events_available` advisory.
///
/// Returned from [`ThreadEventsWatchHandler::handle_payload`] so
/// callers can make observability decisions (logging, metrics,
/// dead-letter handling) without taking a second round trip to the
/// repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThreadEventsWatchOutcome {
    /// The handler read one or more events from the durable repository
    /// and forwarded them to the local
    /// [`EventNotifier`].  Subscribers are responsible for de-duping
    /// against their own cursor — that guarantee is owned by
    /// [`EventStream`](super::event_stream::EventStream), not by the
    /// handler.
    Forwarded {
        /// Number of events pushed through the notifier.
        emitted_count: u32,
        /// Highest sequence pushed through the notifier.
        emitted_up_to: u64,
    },
    /// The advisory's `last_sequence` is already covered by an earlier
    /// delivery on this instance, so no replay was required.  Covers
    /// both exact duplicates and out-of-order redeliveries.
    AlreadyCurrent {
        /// High-water already emitted on this instance (matches the
        /// most recently seen `last_sequence`).
        high_water: u64,
    },
    /// The durable repository returned no events at or below
    /// `last_sequence` for this thread.  Treat as a benign duplicate —
    /// the thread may have been retention-pruned or the advisory may
    /// have arrived ahead of replication.
    UnknownThread,
}

impl ThreadEventsWatchOutcome {
    /// `true` when the handler actually forwarded events through the
    /// notifier.  Useful for tests and metrics.
    #[must_use]
    pub const fn forwarded(&self) -> bool {
        matches!(self, Self::Forwarded { .. })
    }

    /// Number of events the handler pushed through the notifier, if
    /// any.
    #[must_use]
    pub const fn emitted_count(&self) -> u32 {
        match self {
            Self::Forwarded { emitted_count, .. } => *emitted_count,
            Self::AlreadyCurrent { .. } | Self::UnknownThread => 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Handler contract
// ─────────────────────────────────────────────────────────────────────

/// Narrow boundary every `thread_events_available` transport shares.
///
/// Consumers pull advisory payloads from somewhere (AMQP queue,
/// in-memory channel, test driver) and hand each one to the handler.
/// The handler performs the journal re-check and produces a
/// [`ThreadEventsWatchOutcome`] the consumer can log / ack / nack.
#[async_trait]
pub trait ThreadEventsWatchHandler: Send + Sync {
    /// Re-check the durable event repository for the referenced thread
    /// and optionally forward new events into the local notifier.
    ///
    /// # Errors
    /// Returns an error only when the durable repository itself cannot
    /// be queried.  Missing threads and duplicate / out-of-order
    /// advisories are returned as
    /// [`ThreadEventsWatchOutcome::UnknownThread`] /
    /// [`ThreadEventsWatchOutcome::AlreadyCurrent`] — **not** errors —
    /// so the consumer can ack the broker message with a clean
    /// conscience.
    async fn handle_payload(
        &self,
        payload: &ThreadEventsAvailablePayload,
        now: OffsetDateTime,
    ) -> Result<ThreadEventsWatchOutcome>;
}

// ─────────────────────────────────────────────────────────────────────
// Default journal-backed handler
// ─────────────────────────────────────────────────────────────────────

/// Default [`ThreadEventsWatchHandler`] that reads committed events
/// from an [`EventRepository`] and nudges a shared [`EventNotifier`].
///
/// Carries a small in-memory high-water map so duplicate and
/// out-of-order advisories short-circuit without touching the
/// notifier.  The map is intentionally not durable: on restart the
/// first advisory per thread replays from the beginning of the
/// committed log, which subscribers filter against their own
/// `last_yielded` cursor.
#[derive(Clone)]
pub struct NotifierThreadEventsWatchHandler {
    event_repo: Arc<dyn EventRepository>,
    notifier: Arc<EventNotifier>,
    /// Per-thread highest sequence already forwarded through the
    /// notifier.  Absence means "nothing forwarded yet on this
    /// instance".
    high_water: Arc<Mutex<HashMap<String, u64>>>,
}

impl NotifierThreadEventsWatchHandler {
    /// Construct a handler that looks events up in `event_repo` and
    /// nudges subscribers through `notifier`.
    #[must_use]
    pub fn new(event_repo: Arc<dyn EventRepository>, notifier: Arc<EventNotifier>) -> Self {
        Self {
            event_repo,
            notifier,
            high_water: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Access the underlying notifier.  Handy when the caller wants to
    /// share it with the in-process commit path.
    #[must_use]
    pub const fn notifier(&self) -> &Arc<EventNotifier> {
        &self.notifier
    }

    fn read_high_water(&self, thread_id: &ThreadId) -> Option<u64> {
        self.high_water
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&thread_id.0)
            .copied()
    }

    fn bump_high_water(&self, thread_id: &ThreadId, sequence: u64) {
        self.high_water
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(thread_id.0.clone())
            .and_modify(|v| *v = (*v).max(sequence))
            .or_insert(sequence);
    }

    async fn load_replay_window(
        &self,
        thread_id: &ThreadId,
        from_inclusive: u64,
        up_to: u64,
    ) -> Result<Vec<CommittedEvent>> {
        // `get_events_in_range` uses a strictly-greater-than lower
        // bound.  For `from_inclusive == 0` we can't express
        // "seq > -1" on a `u64`, so fall back to a full replay filter.
        if from_inclusive == 0 {
            let all = self
                .event_repo
                .get_events(thread_id)
                .await
                .with_context(|| {
                    format!(
                        "load events for thread {} during watch re-check",
                        thread_id.0
                    )
                })?;
            Ok(all.into_iter().filter(|e| e.sequence <= up_to).collect())
        } else {
            self.event_repo
                .get_events_in_range(thread_id, from_inclusive - 1, up_to)
                .await
                .with_context(|| {
                    format!(
                        "load events in range ({}..={}] for thread {} during watch re-check",
                        from_inclusive - 1,
                        up_to,
                        thread_id.0,
                    )
                })
        }
    }
}

#[async_trait]
impl ThreadEventsWatchHandler for NotifierThreadEventsWatchHandler {
    async fn handle_payload(
        &self,
        payload: &ThreadEventsAvailablePayload,
        _now: OffsetDateTime,
    ) -> Result<ThreadEventsWatchOutcome> {
        if let Some(high_water) = self.read_high_water(&payload.thread_id)
            && payload.last_sequence <= high_water
        {
            log::debug!(
                thread_id = &payload.thread_id.0,
                last_sequence = payload.last_sequence,
                high_water;
                "thread_events_available advisory already covered by earlier delivery; \
                 treating as benign duplicate",
            );
            return Ok(ThreadEventsWatchOutcome::AlreadyCurrent { high_water });
        }

        let baseline = self
            .read_high_water(&payload.thread_id)
            .map_or(0, |hw| hw + 1);
        let events = self
            .load_replay_window(&payload.thread_id, baseline, payload.last_sequence)
            .await?;

        if events.is_empty() {
            log::debug!(
                thread_id = &payload.thread_id.0,
                last_sequence = payload.last_sequence,
                baseline;
                "thread_events_available re-check found no events for replay window; \
                 treating as unknown-thread duplicate",
            );
            return Ok(ThreadEventsWatchOutcome::UnknownThread);
        }

        let emitted_count_usize = events.len();
        let emitted_up_to = events
            .last()
            .context("non-empty events vec must have a last element")?
            .sequence;
        self.notifier.notify(&events);
        self.bump_high_water(&payload.thread_id, emitted_up_to);
        let emitted_count = u32::try_from(emitted_count_usize).unwrap_or(u32::MAX);

        log::debug!(
            thread_id = &payload.thread_id.0,
            last_sequence = payload.last_sequence,
            emitted_count,
            emitted_up_to;
            "thread_events_available re-check forwarded durable events to notifier",
        );

        Ok(ThreadEventsWatchOutcome::Forwarded {
            emitted_count,
            emitted_up_to,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helper: run one-shot payload dispatch
// ─────────────────────────────────────────────────────────────────────

/// Convenience wrapper that boxes the payload dispatch so consumers
/// can log around the handler without duplicating the invocation
/// shape.
///
/// # Errors
/// Propagates handler errors unchanged.
pub async fn dispatch_thread_events_payload(
    handler: &(dyn ThreadEventsWatchHandler + '_),
    payload: &ThreadEventsAvailablePayload,
    now: OffsetDateTime,
) -> Result<ThreadEventsWatchOutcome> {
    let outcome = handler.handle_payload(payload, now).await?;
    match &outcome {
        ThreadEventsWatchOutcome::Forwarded {
            emitted_count,
            emitted_up_to,
        } => {
            log::debug!(
                thread_id = &payload.thread_id.0,
                last_sequence = payload.last_sequence,
                emitted_count = *emitted_count,
                emitted_up_to = *emitted_up_to;
                "thread_events_available advisory forwarded",
            );
        }
        ThreadEventsWatchOutcome::AlreadyCurrent { high_water } => {
            log::debug!(
                thread_id = &payload.thread_id.0,
                last_sequence = payload.last_sequence,
                high_water = *high_water;
                "thread_events_available advisory already current",
            );
        }
        ThreadEventsWatchOutcome::UnknownThread => {
            log::debug!(
                thread_id = &payload.thread_id.0,
                last_sequence = payload.last_sequence;
                "thread_events_available advisory references unknown thread",
            );
        }
    }
    Ok(outcome)
}

// ─────────────────────────────────────────────────────────────────────
// In-memory capture handler (test double)
// ─────────────────────────────────────────────────────────────────────

/// Test double that records every handled payload and lets the test
/// control what outcome the handler reports.
///
/// Keeping this helper alongside the trait keeps external test
/// harnesses (broker consumers) from having to duplicate the same mock
/// across crates.
#[derive(Default)]
pub struct CapturingThreadEventsWatchHandler {
    inner: tokio::sync::Mutex<CapturingInner>,
}

#[derive(Default)]
struct CapturingInner {
    payloads: Vec<ThreadEventsAvailablePayload>,
    outcome: Option<ThreadEventsWatchOutcome>,
    error: Option<String>,
}

impl CapturingThreadEventsWatchHandler {
    /// Build an empty capturing handler.  Uses
    /// [`ThreadEventsWatchOutcome::UnknownThread`] as the default reply
    /// so a test that forgets to configure the outcome never
    /// accidentally looks like a successful nudge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot every payload delivered to the handler in arrival
    /// order.
    pub async fn payloads(&self) -> Vec<ThreadEventsAvailablePayload> {
        self.inner.lock().await.payloads.clone()
    }

    /// Override the outcome every future call returns.
    pub async fn reply_with(&self, outcome: ThreadEventsWatchOutcome) {
        let mut inner = self.inner.lock().await;
        inner.outcome = Some(outcome);
        inner.error = None;
    }

    /// Force every future call to fail with this error string.
    pub async fn fail_with(&self, message: impl Into<String>) {
        let mut inner = self.inner.lock().await;
        inner.error = Some(message.into());
        inner.outcome = None;
    }
}

#[async_trait]
impl ThreadEventsWatchHandler for CapturingThreadEventsWatchHandler {
    async fn handle_payload(
        &self,
        payload: &ThreadEventsAvailablePayload,
        _now: OffsetDateTime,
    ) -> Result<ThreadEventsWatchOutcome> {
        let mut inner = self.inner.lock().await;
        inner.payloads.push(payload.clone());
        if let Some(err) = inner.error.clone() {
            anyhow::bail!(err);
        }
        Ok(inner
            .outcome
            .clone()
            .unwrap_or(ThreadEventsWatchOutcome::UnknownThread))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::event_repository::InMemoryEventRepository;
    use agent_sdk_core::events::AgentEvent;
    use std::time::Duration as StdDuration;
    use time::Duration as TimeDuration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + TimeDuration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + TimeDuration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-watch-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-watch-b")
    }

    async fn seed(
        repo: &InMemoryEventRepository,
        thread: &ThreadId,
        count: u64,
    ) -> Result<Vec<CommittedEvent>> {
        let mut out = Vec::new();
        for i in 0..count {
            let committed = repo
                .commit_event(
                    thread,
                    AgentEvent::text(format!("msg_{i}"), format!("payload-{i}")),
                    t_plus(i64::try_from(i).context("seq fits in i64")?),
                )
                .await?;
            out.push(committed);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn forwards_committed_events_from_start() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 3).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx = notifier.subscribe(&thread_a());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        let outcome = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 2,
                },
                t0(),
            )
            .await?;
        assert_eq!(
            outcome,
            ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 3,
                emitted_up_to: 2,
            }
        );

        for expected in 0..3u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
            assert_eq!(received.thread_id, thread_a());
        }
        Ok(())
    }

    #[tokio::test]
    async fn duplicate_advisory_is_benign() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 4).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx = notifier.subscribe(&thread_a());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        let payload = ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 3,
        };

        let first = handler.handle_payload(&payload, t0()).await?;
        assert!(matches!(
            first,
            ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 4,
                emitted_up_to: 3
            }
        ));
        for expected in 0..4u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }

        let second = handler.handle_payload(&payload, t0()).await?;
        assert_eq!(
            second,
            ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 3 }
        );
        // The notifier must not have been touched a second time.
        let no_more = tokio::time::timeout(StdDuration::from_millis(50), rx.recv()).await;
        assert!(
            no_more.is_err(),
            "duplicate advisory must not emit any extra events; got {no_more:?}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn out_of_order_advisory_is_benign() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 6).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx = notifier.subscribe(&thread_a());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        // Forward the high advisory first.
        let high = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 5,
                },
                t0(),
            )
            .await?;
        assert!(matches!(high, ThreadEventsWatchOutcome::Forwarded { .. }));
        for expected in 0..6u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }

        // Now deliver an older advisory — high-water already covers it.
        let late = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 2,
                },
                t0(),
            )
            .await?;
        assert_eq!(
            late,
            ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 5 }
        );
        let no_more = tokio::time::timeout(StdDuration::from_millis(50), rx.recv()).await;
        assert!(
            no_more.is_err(),
            "out-of-order advisory must not re-emit earlier events; got {no_more:?}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn unknown_thread_advisory_acks_without_emit() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        let notifier = Arc::new(EventNotifier::new());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        let outcome = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: ThreadId::from_string("t-does-not-exist"),
                    last_sequence: 0,
                },
                t0(),
            )
            .await?;
        assert_eq!(outcome, ThreadEventsWatchOutcome::UnknownThread);
        Ok(())
    }

    #[tokio::test]
    async fn partial_emit_when_replica_lags_behind_advisory() -> Result<()> {
        // The durable repo only has 2 events; the advisory claims 4.
        // The handler must emit what it can find and leave high-water
        // at the actual committed maximum so a later redelivery can
        // pick up the catch-up once replication completes.
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 2).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx = notifier.subscribe(&thread_a());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        let outcome = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 3,
                },
                t0(),
            )
            .await?;
        assert_eq!(
            outcome,
            ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 2,
                emitted_up_to: 1,
            }
        );
        for expected in 0..2u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }

        // Replication catches up — events 2 and 3 land.
        seed(&repo, &thread_a(), 0).await?; // no-op seed helper contract
        // Commit the missing tail directly to simulate replication
        // catching up.  The in-memory repo assigns the next sequences
        // automatically.
        repo.commit_event(&thread_a(), AgentEvent::text("msg_2", "late-2"), t_plus(2))
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("msg_3", "late-3"), t_plus(3))
            .await?;

        // Re-deliver the advisory — the handler must now emit 2 and 3.
        let catch_up = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 3,
                },
                t0(),
            )
            .await?;
        assert_eq!(
            catch_up,
            ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 2,
                emitted_up_to: 3,
            }
        );
        for expected in 2..4u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn incremental_advisories_forward_tail_only() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 2).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx = notifier.subscribe(&thread_a());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 1,
                },
                t0(),
            )
            .await?;
        for expected in 0..2u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }

        // Remote instance commits two more events and advisories fire.
        repo.commit_event(&thread_a(), AgentEvent::text("msg_2", "b"), t_plus(2))
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("msg_3", "c"), t_plus(3))
            .await?;
        let outcome = handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 3,
                },
                t0(),
            )
            .await?;
        assert_eq!(
            outcome,
            ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 2,
                emitted_up_to: 3,
            }
        );
        for expected in 2..4u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn high_water_is_thread_scoped() -> Result<()> {
        let repo = Arc::new(InMemoryEventRepository::new());
        seed(&repo, &thread_a(), 3).await?;
        seed(&repo, &thread_b(), 2).await?;

        let notifier = Arc::new(EventNotifier::new());
        let mut rx_a = notifier.subscribe(&thread_a());
        let mut rx_b = notifier.subscribe(&thread_b());
        let handler = NotifierThreadEventsWatchHandler::new(
            Arc::clone(&repo) as Arc<dyn EventRepository>,
            Arc::clone(&notifier),
        );

        // Advance thread_a only.
        handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_a(),
                    last_sequence: 2,
                },
                t0(),
            )
            .await?;
        for expected in 0..3u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx_a.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
        }

        // Thread B must still see a full initial replay on its own
        // advisory — the watermark is per-thread.
        handler
            .handle_payload(
                &ThreadEventsAvailablePayload {
                    thread_id: thread_b(),
                    last_sequence: 1,
                },
                t0(),
            )
            .await?;
        for expected in 0..2u64 {
            let received = tokio::time::timeout(StdDuration::from_millis(100), rx_b.recv())
                .await
                .context("notifier did not deliver event in time")??;
            assert_eq!(received.sequence, expected);
            assert_eq!(received.thread_id, thread_b());
        }
        Ok(())
    }

    #[tokio::test]
    async fn capturing_handler_records_and_replies() -> Result<()> {
        let handler = CapturingThreadEventsWatchHandler::new();
        handler
            .reply_with(ThreadEventsWatchOutcome::Forwarded {
                emitted_count: 1,
                emitted_up_to: 0,
            })
            .await;
        let payload = ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 0,
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert!(outcome.forwarded());

        let calls = handler.payloads().await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], payload);
        Ok(())
    }

    #[tokio::test]
    async fn capturing_handler_surfaces_errors_for_retry_testing() -> Result<()> {
        let handler = CapturingThreadEventsWatchHandler::new();
        handler.fail_with("simulated repo failure").await;
        let payload = ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 0,
        };
        let Err(err) = handler.handle_payload(&payload, t0()).await else {
            anyhow::bail!("expected handler to fail, got Ok");
        };
        let rendered = format!("{err:#}");
        assert!(rendered.contains("simulated repo failure"));
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_helper_forwards_outcome() -> Result<()> {
        let handler = CapturingThreadEventsWatchHandler::new();
        handler
            .reply_with(ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 7 })
            .await;
        let payload = ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 7,
        };
        let outcome = dispatch_thread_events_payload(&handler, &payload, t0()).await?;
        assert_eq!(
            outcome,
            ThreadEventsWatchOutcome::AlreadyCurrent { high_water: 7 }
        );
        Ok(())
    }
}
