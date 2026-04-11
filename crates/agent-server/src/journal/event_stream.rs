//! Race-free replay-to-live event stream.
//!
//! [`stream_events`] is the public replay surface for Phase 6.3.  A
//! client that disconnects can restart from `after_sequence` and
//! receive the exact committed event stream without gaps or duplicates,
//! even while workers are actively producing new events.
//!
//! # Handoff protocol
//!
//! The replay-to-live handoff is a three-step process that closes the
//! race window at the watermark boundary:
//!
//! 1. **Subscribe** — acquire a live tail receiver from the
//!    [`EventNotifier`] *before* reading durable state.  This ensures
//!    that any event committed after the subscribe call will be
//!    delivered through the receiver.
//!
//! 2. **Capture watermark** — read the current committed high-water
//!    mark from the [`EventRepository`] (the highest committed
//!    sequence, or `None` if the thread has no events).  Because the
//!    subscription was established in step 1, any event committed
//!    after this read will arrive through the live tail.
//!
//! 3. **Replay + switch** — yield durable events with
//!    `sequence > after_sequence && sequence <= watermark` from the
//!    repository, then switch to the live tail for
//!    `sequence > watermark`.  The live receiver may deliver events
//!    that overlap with the replay window (committed between steps 1
//!    and 2), so we filter duplicates by tracking the last yielded
//!    sequence.
//!
//! # Guarantees
//!
//! - **No gaps**: the subscribe-before-read ordering ensures every
//!   committed event is either in the durable replay or the live tail.
//! - **No duplicates**: the `last_yielded` tracking skips any live
//!   event whose sequence was already emitted during replay.
//! - **No unpublished events**: clients never see in-memory events
//!   that have not been durably committed — the live tail is fed
//!   exclusively by [`EventNotifier::notify`], which is called only
//!   after durable commit.
//! - **Thread-scoped**: the stream is bound to a single thread, with
//!   no hidden per-connection cursor state.

use super::committed_event::CommittedEvent;
use super::event_notifier::{EventNotifier, EventReceiver};
use super::event_repository::EventRepository;
use agent_sdk_core::ThreadId;
use anyhow::{Context, Result};
use tokio::sync::broadcast;

// ─────────────────────────────────────────────────────────────────────
// StreamEvents API
// ─────────────────────────────────────────────────────────────────────

/// A stream of committed events for a single thread, with race-free
/// replay-to-live handoff.
///
/// Created by [`stream_events`].  Call [`next`](EventStream::next) in
/// a loop to receive events in sequence order.
pub struct EventStream {
    /// Events buffered from the durable replay phase.
    replay_buffer: Vec<CommittedEvent>,
    /// Current index into `replay_buffer`.
    replay_index: usize,
    /// Live tail receiver (post-watermark events).
    live_rx: EventReceiver,
    /// The highest sequence yielded so far.  Used to skip duplicates
    /// during the handoff window.
    last_yielded: Option<u64>,
    /// Whether the replay phase has been fully drained.
    replay_drained: bool,
}

/// Outcome of [`EventStream::next`].
#[derive(Debug)]
pub enum StreamEvent {
    /// A committed event in sequence order.
    Event(Box<CommittedEvent>),
    /// The subscriber fell behind the live broadcast buffer and must
    /// re-establish the stream from durable storage.  The `skipped`
    /// count indicates how many events were lost.
    Lagged { skipped: u64 },
}

impl EventStream {
    /// Receive the next committed event from the stream.
    ///
    /// During the replay phase, events are yielded from the durable
    /// buffer.  After replay is exhausted, events come from the live
    /// tail.
    ///
    /// Returns `None` when the live tail channel is closed (all
    /// notifiers dropped).
    pub async fn next(&mut self) -> Option<StreamEvent> {
        // Phase 1: drain the replay buffer.
        if !self.replay_drained {
            if let Some(event) = self.next_from_replay() {
                return Some(StreamEvent::Event(Box::new(event)));
            }
            self.replay_drained = true;
        }

        // Phase 2: live tail.
        self.next_from_live().await
    }

    fn next_from_replay(&mut self) -> Option<CommittedEvent> {
        if self.replay_index < self.replay_buffer.len() {
            let event = self.replay_buffer[self.replay_index].clone();
            self.replay_index += 1;
            self.last_yielded = Some(event.sequence);
            return Some(event);
        }
        None
    }

    async fn next_from_live(&mut self) -> Option<StreamEvent> {
        loop {
            match self.live_rx.recv().await {
                Ok(event) => {
                    // Skip events already yielded during replay.
                    if self.last_yielded.is_some_and(|last| event.sequence <= last) {
                        continue;
                    }
                    self.last_yielded = Some(event.sequence);
                    return Some(StreamEvent::Event(Box::new(event)));
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    return Some(StreamEvent::Lagged { skipped: n });
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return None;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────

/// Create a replay-to-live event stream for a thread.
///
/// The returned [`EventStream`] yields all committed events with
/// `sequence > after_sequence` in order, seamlessly transitioning
/// from durable replay to live tail without gaps or duplicates.
///
/// Pass `after_sequence = None` to replay from the beginning of the
/// thread's event history.
///
/// # Errors
///
/// Returns an error if the durable replay query fails.
pub async fn stream_events(
    thread_id: &ThreadId,
    after_sequence: Option<u64>,
    event_repo: &dyn EventRepository,
    notifier: &EventNotifier,
) -> Result<EventStream> {
    // Step 1: subscribe BEFORE reading durable state.
    let live_rx = notifier.subscribe(thread_id);

    // Step 2: capture the committed watermark.
    //
    // `next_sequence` returns the next-to-be-assigned sequence.
    // The highest committed sequence is `watermark - 1`, or the
    // thread has no events if `watermark == 0`.
    let watermark = event_repo
        .next_sequence(thread_id)
        .await
        .context("stream_events: read watermark")?;

    // Step 3: replay durable events in the window
    // `(after_sequence, high_water]`.
    //
    // For "replay from start" (`after_sequence = None`) we load all
    // events up to the watermark via `get_events` + filter, because
    // `get_events_in_range` uses a strictly-greater-than lower bound
    // and would skip sequence 0.
    let replay_buffer = if watermark == 0 {
        Vec::new()
    } else {
        let high_water = watermark - 1;
        match after_sequence {
            None => {
                // Replay from the very beginning.
                let all = event_repo
                    .get_events(thread_id)
                    .await
                    .context("stream_events: replay from start")?;
                all.into_iter()
                    .filter(|e| e.sequence <= high_water)
                    .collect()
            }
            Some(after) if after < high_water => event_repo
                .get_events_in_range(thread_id, after, high_water)
                .await
                .context("stream_events: replay")?,
            Some(_) => Vec::new(),
        }
    };

    // `last_yielded` seeds the dedup guard so live-tail events that
    // overlap with the replay window are skipped.
    let initial_last_yielded = after_sequence;

    Ok(EventStream {
        replay_buffer,
        replay_index: 0,
        live_rx,
        last_yielded: initial_last_yielded,
        replay_drained: false,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::event_repository::InMemoryEventRepository;
    use super::*;
    use agent_sdk_core::events::AgentEvent;
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-stream-a")
    }

    // ── Basic replay ────────────────────────────────────────────────

    #[tokio::test]
    async fn replay_all_events_from_start() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Commit 3 events.
        repo.commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("m2", "b"), t_plus(1))
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("m3", "c"), t_plus(2))
            .await?;

        let mut stream = stream_events(&thread_a(), None, &repo, &notifier).await?;

        // Should get all 3 from replay.
        for expected_seq in 0..3 {
            match stream.next().await {
                Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, expected_seq),
                other => panic!("expected Event(seq={expected_seq}), got {other:?}"),
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn replay_after_sequence_skips_earlier() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        repo.commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("m2", "b"), t_plus(1))
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("m3", "c"), t_plus(2))
            .await?;

        // Replay from after sequence 1 — should get seq 2 only.
        let mut stream = stream_events(&thread_a(), Some(1), &repo, &notifier).await?;

        match stream.next().await {
            Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 2),
            other => panic!("expected Event(seq=2), got {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn replay_empty_thread() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        let stream = stream_events(&thread_a(), None, &repo, &notifier).await?;
        // Replay buffer should be empty.
        assert!(stream.replay_buffer.is_empty());
        Ok(())
    }

    // ── Live tail ───────────────────────────────────────────────────

    #[tokio::test]
    async fn live_events_after_replay() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Pre-commit 1 event.
        repo.commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;

        let mut stream = stream_events(&thread_a(), None, &repo, &notifier).await?;

        // Drain replay.
        match stream.next().await {
            Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(seq=0), got {other:?}"),
        }

        // Now commit a new event and notify.
        let new_events = repo
            .commit_event(&thread_a(), AgentEvent::text("m2", "b"), t_plus(1))
            .await?;
        notifier.notify(&[new_events]);

        // Should receive from live tail.
        match stream.next().await {
            Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 1),
            other => panic!("expected Event(seq=1), got {other:?}"),
        }
        Ok(())
    }

    // ── Handoff dedup ───────────────────────────────────────────────

    #[tokio::test]
    async fn handoff_deduplicates_overlap() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Pre-commit 2 events.
        repo.commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;
        let e1 = repo
            .commit_event(&thread_a(), AgentEvent::text("m2", "b"), t_plus(1))
            .await?;

        let mut stream = stream_events(&thread_a(), None, &repo, &notifier).await?;

        // Before draining replay, "notify" the live tail with an event
        // that's already in the replay window (simulates a commit that
        // happened between subscribe and watermark capture).
        notifier.notify(&[e1]);

        // Also notify a genuinely new event.
        let e2 = repo
            .commit_event(&thread_a(), AgentEvent::text("m3", "c"), t_plus(2))
            .await?;
        notifier.notify(&[e2]);

        // Drain: should get seq 0, 1 from replay, then 2 from live.
        // Seq 1 should NOT appear twice.
        let mut seen_sequences = Vec::new();
        for _ in 0..3 {
            match stream.next().await {
                Some(StreamEvent::Event(e)) => seen_sequences.push(e.sequence),
                other => panic!("expected Event, got {other:?}"),
            }
        }
        assert_eq!(seen_sequences, vec![0, 1, 2]);
        Ok(())
    }

    // ── Concurrent production during replay ─────────────────────────

    #[tokio::test]
    async fn concurrent_production_no_gaps() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Pre-commit 5 events.
        for i in 0..5 {
            repo.commit_event(
                &thread_a(),
                AgentEvent::text(format!("m{i}"), format!("msg-{i}")),
                t_plus(i),
            )
            .await?;
        }

        // Start stream from after sequence 2.
        let mut stream = stream_events(&thread_a(), Some(2), &repo, &notifier).await?;

        // Concurrently commit more events.
        for i in 5..8 {
            let e = repo
                .commit_event(
                    &thread_a(),
                    AgentEvent::text(format!("m{i}"), format!("msg-{i}")),
                    t_plus(i),
                )
                .await?;
            notifier.notify(&[e]);
        }

        // Should get seq 3, 4 from replay, then 5, 6, 7 from live.
        let mut seen = Vec::new();
        for _ in 0..5 {
            match stream.next().await {
                Some(StreamEvent::Event(e)) => seen.push(e.sequence),
                other => panic!("expected Event, got {other:?}"),
            }
        }
        assert_eq!(seen, vec![3, 4, 5, 6, 7]);
        Ok(())
    }

    // ── Thread isolation ────────────────────────────────────────────

    #[tokio::test]
    async fn streams_are_thread_scoped() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();
        let thread_b = ThreadId::from_string("t-stream-b");

        repo.commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;
        repo.commit_event(&thread_b, AgentEvent::text("m1", "b"), t0())
            .await?;

        let stream_a = stream_events(&thread_a(), None, &repo, &notifier).await?;
        let stream_b = stream_events(&thread_b, None, &repo, &notifier).await?;

        assert_eq!(stream_a.replay_buffer.len(), 1);
        assert_eq!(stream_a.replay_buffer[0].thread_id, thread_a());
        assert_eq!(stream_b.replay_buffer.len(), 1);
        assert_eq!(stream_b.replay_buffer[0].thread_id, thread_b);
        Ok(())
    }

    // ── Reconnect after restart ─────────────────────────────────────

    #[tokio::test]
    async fn reconnect_after_restart_gets_exact_stream() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Commit 5 events (simulating pre-restart state).
        for i in 0..5 {
            repo.commit_event(
                &thread_a(),
                AgentEvent::text(format!("m{i}"), format!("pre-{i}")),
                t_plus(i),
            )
            .await?;
        }

        // Client reconnects after seeing sequence 2.
        let mut stream = stream_events(&thread_a(), Some(2), &repo, &notifier).await?;

        // Should replay seq 3 and 4.
        let mut replayed = Vec::new();
        for _ in 0..2 {
            match stream.next().await {
                Some(StreamEvent::Event(e)) => replayed.push(e.sequence),
                other => panic!("expected Event, got {other:?}"),
            }
        }
        assert_eq!(replayed, vec![3, 4]);

        // New events after restart.
        let e5 = repo
            .commit_event(&thread_a(), AgentEvent::text("m5", "post-5"), t_plus(5))
            .await?;
        notifier.notify(&[e5]);

        match stream.next().await {
            Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 5),
            other => panic!("expected Event(seq=5), got {other:?}"),
        }
        Ok(())
    }

    // ── after_sequence at head (nothing to replay) ──────────────────

    #[tokio::test]
    async fn after_sequence_at_head_goes_straight_to_live() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Commit 3 events.
        for i in 0..3 {
            repo.commit_event(
                &thread_a(),
                AgentEvent::text(format!("m{i}"), "x"),
                t_plus(i),
            )
            .await?;
        }

        // Client already has everything (after_sequence = 2, which is
        // the last committed sequence).
        let mut stream = stream_events(&thread_a(), Some(2), &repo, &notifier).await?;
        assert!(stream.replay_buffer.is_empty());

        // Only live events should arrive.
        let e3 = repo
            .commit_event(&thread_a(), AgentEvent::text("m3", "new"), t_plus(3))
            .await?;
        notifier.notify(&[e3]);

        match stream.next().await {
            Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 3),
            other => panic!("expected Event(seq=3), got {other:?}"),
        }
        Ok(())
    }

    // ── Batch commit and notify ─────────────────────────────────────

    #[tokio::test]
    async fn batch_commit_during_replay() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let notifier = EventNotifier::new();

        // Pre-commit 2 events.
        repo.commit_event(&thread_a(), AgentEvent::text("m0", "a"), t0())
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("m1", "b"), t_plus(1))
            .await?;

        let mut stream = stream_events(&thread_a(), None, &repo, &notifier).await?;

        // Batch commit 3 more events.
        let batch = repo
            .commit_event_batch(
                &thread_a(),
                vec![
                    AgentEvent::text("m2", "c"),
                    AgentEvent::text("m3", "d"),
                    AgentEvent::text("m4", "e"),
                ],
                t_plus(2),
            )
            .await?;
        notifier.notify(&batch);

        // Drain all 5.
        let mut seen = Vec::new();
        for _ in 0..5 {
            match stream.next().await {
                Some(StreamEvent::Event(e)) => seen.push(e.sequence),
                other => panic!("expected Event, got {other:?}"),
            }
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        Ok(())
    }
}
