//! Phase 6.4 – Live tail hub with per-subscriber bounded buffers,
//! lag detection, and replay-required disconnect.
//!
//! [`LiveTailHub`] is the same-process live event fanout surface for
//! committed thread-scoped envelopes.  Each subscriber receives its own
//! bounded queue.  When a subscriber falls behind (queue full), the hub
//! enters a bounded-wait grace period.  If the subscriber remains behind
//! after the grace period expires, the hub disconnects the subscriber
//! with a replay-required signal that includes the last delivered durable
//! sequence.
//!
//! # Design
//!
//! - Each subscriber gets an independent `mpsc::Sender` with bounded
//!   capacity — no shared ring buffer, no cross-subscriber interference.
//! - [`LiveTailHub::publish`] is non-blocking: workers call it after
//!   durable commit and are never stalled by slow subscribers.
//! - Lag detection transitions a subscriber from `Healthy` to `Lagging`
//!   when `try_send` fails with a full buffer.
//! - Once lag is detected, the subscriber is **doomed** — it will be
//!   disconnected after the grace period expires.  During the grace
//!   period the hub stops delivering events to the lagging subscriber
//!   (events are durable and recoverable via replay), giving the
//!   subscriber time to drain its existing buffer.
//! - On disconnection, the hub signals `replay_required` through shared
//!   out-of-band state and drops the sender half.  The receiver drains
//!   any remaining buffered events, then returns
//!   [`LiveTailEvent::ReplayRequired`] with the last successfully
//!   delivered sequence.
//! - The subscriber reconnects via
//!   [`stream_events`](super::event_stream::stream_events) with
//!   `after_sequence` set to the last-delivered sequence, seamlessly
//!   recovering the missed committed envelopes.
//!
//! # Workers never block
//!
//! [`LiveTailHub::publish`] holds a `std::sync::Mutex` for the minimum
//! time required to iterate subscribers and call `try_send`.  No async
//! `.await` point is reached under the lock.  A slow or dead subscriber
//! never stalls the producer.

use super::committed_event::CommittedEvent;
use agent_sdk_core::ThreadId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// ─────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────

/// Default per-subscriber buffer capacity.
const DEFAULT_BUFFER_CAPACITY: usize = 256;

/// Default lag grace period before disconnect.
const DEFAULT_LAG_GRACE_PERIOD: Duration = Duration::from_secs(5);

/// Configuration for the live tail hub.
#[derive(Clone, Copy, Debug)]
pub struct LiveTailConfig {
    /// Maximum events buffered per subscriber before lag detection.
    pub buffer_capacity: usize,
    /// Time to wait after lag detection before disconnecting.
    pub lag_grace_period: Duration,
}

impl Default for LiveTailConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
            lag_grace_period: DEFAULT_LAG_GRACE_PERIOD,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Subscriber types
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for a subscriber within a single thread's fanout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SubscriberId(u64);

/// Items delivered to a live tail subscriber.
#[derive(Debug)]
pub enum LiveTailEvent {
    /// A committed event.
    Event(Box<CommittedEvent>),
    /// The subscriber fell behind and must reconnect via replay.
    /// `last_delivered_sequence` is the highest sequence that was
    /// successfully buffered — the subscriber should reconnect with
    /// `after_sequence` set to this value.
    ReplayRequired {
        last_delivered_sequence: Option<u64>,
    },
}

/// A subscriber's handle to receive live events.
///
/// Created by [`LiveTailHub::subscribe`].  Call
/// [`recv`](LiveTailReceiver::recv) in a loop to consume events.
pub struct LiveTailReceiver {
    rx: mpsc::Receiver<CommittedEvent>,
    shared: Arc<SubscriberShared>,
    subscriber_id: SubscriberId,
    /// Ensures `ReplayRequired` is returned exactly once.
    replay_signaled: bool,
}

impl LiveTailReceiver {
    /// Receive the next item from the live tail.
    ///
    /// Returns [`LiveTailEvent::Event`] for normal delivery.
    /// Returns [`LiveTailEvent::ReplayRequired`] **exactly once** when
    /// the hub disconnects this subscriber due to lag.
    /// Returns `None` when the hub is dropped (normal shutdown).
    pub async fn recv(&mut self) -> Option<LiveTailEvent> {
        if let Some(event) = self.rx.recv().await {
            return Some(LiveTailEvent::Event(Box::new(event)));
        }

        if self.replay_signaled {
            return None;
        }
        if self.shared.replay_required.load(Ordering::Acquire) {
            self.replay_signaled = true;
            let last_seq = *self
                .shared
                .last_delivered_sequence
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Some(LiveTailEvent::ReplayRequired {
                last_delivered_sequence: last_seq,
            })
        } else {
            None
        }
    }

    /// This subscriber's unique identifier.
    #[must_use]
    pub const fn subscriber_id(&self) -> SubscriberId {
        self.subscriber_id
    }
}

/// Shared out-of-band state between the hub and a subscriber's
/// receiver.  Written by the hub at disconnect time, read by the
/// receiver after the mpsc channel closes.
struct SubscriberShared {
    /// The highest sequence successfully delivered to this subscriber.
    /// Written by the hub at disconnect time (under the hub mutex).
    last_delivered_sequence: Mutex<Option<u64>>,
    /// Set by the hub when disconnecting due to lag.  The Release
    /// store synchronises with the receiver's Acquire load to
    /// guarantee visibility of `last_delivered_sequence`.
    replay_required: AtomicBool,
}

// ─────────────────────────────────────────────────────────────────────
// Hub internals
// ─────────────────────────────────────────────────────────────────────

/// Per-subscriber lag state.
enum LagState {
    Healthy,
    /// The subscriber's buffer is full.  `since` is the instant lag
    /// was first detected.  Once `grace_period` elapses from `since`,
    /// the subscriber will be disconnected.
    Lagging {
        since: Instant,
    },
}

/// Per-subscriber handle held by the hub.
struct SubscriberHandle {
    tx: mpsc::Sender<CommittedEvent>,
    shared: Arc<SubscriberShared>,
    lag_state: LagState,
    /// Hub-side tracking of the last successfully delivered sequence.
    /// Copied to `shared.last_delivered_sequence` at disconnect time.
    last_delivered_sequence: Option<u64>,
}

/// Per-thread fan-out state.
struct ThreadFanout {
    subscribers: HashMap<SubscriberId, SubscriberHandle>,
    next_id: u64,
}

struct LiveTailHubInner {
    threads: HashMap<String, ThreadFanout>,
}

// ─────────────────────────────────────────────────────────────────────
// LiveTailHub
// ─────────────────────────────────────────────────────────────────────

/// Thread-scoped live tail hub with per-subscriber bounded buffers,
/// lag detection, and replay-required disconnect.
///
/// Cloning shares the same underlying state.
#[derive(Clone)]
pub struct LiveTailHub {
    inner: Arc<Mutex<LiveTailHubInner>>,
    config: LiveTailConfig,
}

impl LiveTailHub {
    /// Create a new hub with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LiveTailConfig::default())
    }

    /// Create a new hub with custom configuration.
    ///
    /// `buffer_capacity` is clamped to a minimum of 1 because Tokio's
    /// `mpsc::channel` requires a non-zero buffer size.
    #[must_use]
    pub fn with_config(config: LiveTailConfig) -> Self {
        let config = LiveTailConfig {
            buffer_capacity: config.buffer_capacity.max(1),
            ..config
        };
        Self {
            inner: Arc::new(Mutex::new(LiveTailHubInner {
                threads: HashMap::new(),
            })),
            config,
        }
    }

    /// Subscribe to live committed events for a thread.
    ///
    /// The returned [`LiveTailReceiver`] receives events published
    /// **after** this call.  Events committed before the subscribe
    /// call must be obtained via replay.
    #[must_use]
    pub fn subscribe(&self, thread_id: &ThreadId) -> LiveTailReceiver {
        let (tx, rx) = mpsc::channel(self.config.buffer_capacity);
        let shared = Arc::new(SubscriberShared {
            last_delivered_sequence: Mutex::new(None),
            replay_required: AtomicBool::new(false),
        });

        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let fanout = guard
            .threads
            .entry(thread_id.0.clone())
            .or_insert_with(|| ThreadFanout {
                subscribers: HashMap::new(),
                next_id: 0,
            });

        let id = SubscriberId(fanout.next_id);
        fanout.next_id += 1;

        fanout.subscribers.insert(
            id,
            SubscriberHandle {
                tx,
                shared: Arc::clone(&shared),
                lag_state: LagState::Healthy,
                last_delivered_sequence: None,
            },
        );

        drop(guard);

        LiveTailReceiver {
            rx,
            shared,
            subscriber_id: id,
            replay_signaled: false,
        }
    }

    /// Publish committed events to all subscribers of the events'
    /// thread.
    ///
    /// Call this after events have been durably committed to the
    /// [`EventRepository`](super::event_repository::EventRepository).
    /// This method **never blocks** on subscriber backpressure.
    ///
    /// All events in the slice must belong to the same thread.
    pub fn publish(&self, events: &[CommittedEvent]) {
        if events.is_empty() {
            return;
        }

        debug_assert!(
            events.iter().all(|e| e.thread_id == events[0].thread_id),
            "publish: all events must belong to the same thread"
        );

        let thread_key = &events[0].thread_id.0;
        let now = Instant::now();
        let grace = self.config.lag_grace_period;

        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let Some(fanout) = guard.threads.get_mut(thread_key) else {
            return;
        };

        let mut to_remove = Vec::new();

        for (&sub_id, handle) in &mut fanout.subscribers {
            // ── Lagging subscribers: check grace period ─────────
            if let LagState::Lagging { since } = handle.lag_state {
                if handle.tx.is_closed() {
                    // Receiver was dropped — clean up immediately.
                    to_remove.push(sub_id);
                } else if now.duration_since(since) >= grace {
                    // Grace period expired — disconnect.
                    *handle
                        .shared
                        .last_delivered_sequence
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        handle.last_delivered_sequence;
                    handle.shared.replay_required.store(true, Ordering::Release);
                    to_remove.push(sub_id);
                }
                // Whether disconnecting or still within grace period,
                // skip event delivery — events are durable.
                continue;
            }

            // ── Healthy subscribers: deliver events ─────────────
            for event in events {
                match handle.tx.try_send(event.clone()) {
                    Ok(()) => {
                        handle.last_delivered_sequence = Some(event.sequence);
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        handle.lag_state = LagState::Lagging { since: now };
                        break;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        to_remove.push(sub_id);
                        break;
                    }
                }
            }
        }

        for id in &to_remove {
            fanout.subscribers.remove(id);
        }

        if fanout.subscribers.is_empty() {
            guard.threads.remove(thread_key);
        }

        drop(guard);
    }

    /// Number of active subscribers for a thread.
    #[must_use]
    pub fn subscriber_count(&self, thread_id: &ThreadId) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .threads
            .get(&thread_id.0)
            .map_or(0, |f| f.subscribers.len())
    }
}

impl Default for LiveTailHub {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::events::AgentEvent;
    use time::OffsetDateTime;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1_700_000_000)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-livetail-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-livetail-b")
    }

    fn sample_committed(thread_id: &ThreadId, seq: u64) -> CommittedEvent {
        CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: thread_id.clone(),
            sequence: seq,
            timestamp: t0(),
            event: AgentEvent::text(format!("msg_{seq}"), format!("payload-{seq}")),
        }
    }

    fn zero_grace_hub(capacity: usize) -> LiveTailHub {
        LiveTailHub::with_config(LiveTailConfig {
            buffer_capacity: capacity,
            lag_grace_period: Duration::ZERO,
        })
    }

    // ── Basic delivery ─────────────────────────────────────────────

    #[tokio::test]
    async fn subscriber_receives_published_events() -> anyhow::Result<()> {
        let hub = LiveTailHub::new();
        let mut rx = hub.subscribe(&thread_a());

        hub.publish(&[
            sample_committed(&thread_a(), 0),
            sample_committed(&thread_a(), 1),
        ]);

        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(0), got {other:?}"),
        }
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 1),
            other => panic!("expected Event(1), got {other:?}"),
        }
        Ok(())
    }

    // ── Thread isolation ───────────────────────────────────────────

    #[tokio::test]
    async fn subscribers_are_thread_scoped() -> anyhow::Result<()> {
        let hub = LiveTailHub::new();
        let mut rx_a = hub.subscribe(&thread_a());
        let mut rx_b = hub.subscribe(&thread_b());

        hub.publish(&[sample_committed(&thread_a(), 0)]);
        hub.publish(&[sample_committed(&thread_b(), 0)]);

        match rx_a.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.thread_id, thread_a()),
            other => panic!("expected Event for thread_a, got {other:?}"),
        }
        match rx_b.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.thread_id, thread_b()),
            other => panic!("expected Event for thread_b, got {other:?}"),
        }
        Ok(())
    }

    // ── Multiple subscribers ───────────────────────────────────────

    #[tokio::test]
    async fn multiple_subscribers_same_thread() -> anyhow::Result<()> {
        let hub = LiveTailHub::new();
        let mut rx1 = hub.subscribe(&thread_a());
        let mut rx2 = hub.subscribe(&thread_a());

        hub.publish(&[sample_committed(&thread_a(), 42)]);

        match rx1.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 42),
            other => panic!("expected Event(42), got {other:?}"),
        }
        match rx2.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 42),
            other => panic!("expected Event(42), got {other:?}"),
        }
        Ok(())
    }

    // ── No-op edge cases ───────────────────────────────────────────

    #[tokio::test]
    async fn publish_without_subscribers_is_noop() {
        let hub = LiveTailHub::new();
        hub.publish(&[sample_committed(&thread_a(), 0)]);
    }

    #[tokio::test]
    async fn empty_publish_is_noop() {
        let hub = LiveTailHub::new();
        let _rx = hub.subscribe(&thread_a());
        hub.publish(&[]);
    }

    // ── Workers never block ────────────────────────────────────────

    #[tokio::test]
    async fn publish_never_blocks_on_slow_subscriber() {
        let hub = LiveTailHub::with_config(LiveTailConfig {
            buffer_capacity: 5,
            lag_grace_period: Duration::from_mins(1),
        });
        let _rx = hub.subscribe(&thread_a()); // Subscribe but never read.

        // Publish 100 events — must not block.
        let start = Instant::now();
        for seq in 0..100 {
            hub.publish(&[sample_committed(&thread_a(), seq)]);
        }
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "publish took too long: {elapsed:?}"
        );
    }

    // ── Lag detection and disconnect ───────────────────────────────

    #[tokio::test]
    async fn overflow_triggers_lag_and_disconnect() -> anyhow::Result<()> {
        let hub = zero_grace_hub(3);
        let mut rx = hub.subscribe(&thread_a());

        assert_eq!(hub.subscriber_count(&thread_a()), 1);

        // Fill the buffer (capacity = 3).
        for seq in 0..3 {
            hub.publish(&[sample_committed(&thread_a(), seq)]);
        }

        // This publish triggers lag (buffer full).
        hub.publish(&[sample_committed(&thread_a(), 3)]);

        // Grace period = 0, so next publish disconnects.
        hub.publish(&[sample_committed(&thread_a(), 4)]);

        assert_eq!(hub.subscriber_count(&thread_a()), 0);

        // Drain the 3 buffered events.
        for expected_seq in 0..3 {
            match rx.recv().await {
                Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, expected_seq),
                other => panic!("expected Event({expected_seq}), got {other:?}"),
            }
        }

        // Next recv should return ReplayRequired.
        match rx.recv().await {
            Some(LiveTailEvent::ReplayRequired {
                last_delivered_sequence,
            }) => {
                assert_eq!(last_delivered_sequence, Some(2));
            }
            other => panic!("expected ReplayRequired, got {other:?}"),
        }

        // Subsequent recv returns None (stream closed).
        assert!(rx.recv().await.is_none());

        Ok(())
    }

    // ── Grace period delays disconnect ─────────────────────────────

    #[tokio::test]
    async fn grace_period_delays_disconnect() -> anyhow::Result<()> {
        let hub = LiveTailHub::with_config(LiveTailConfig {
            buffer_capacity: 2,
            lag_grace_period: Duration::from_millis(200),
        });
        let mut rx = hub.subscribe(&thread_a());

        // Fill the buffer.
        hub.publish(&[
            sample_committed(&thread_a(), 0),
            sample_committed(&thread_a(), 1),
        ]);

        // Trigger lag.
        hub.publish(&[sample_committed(&thread_a(), 2)]);

        // Immediately publish more — within grace period.
        hub.publish(&[sample_committed(&thread_a(), 3)]);
        assert_eq!(
            hub.subscriber_count(&thread_a()),
            1,
            "subscriber should still be connected during grace period"
        );

        // Wait for grace period to expire.
        tokio::time::sleep(Duration::from_millis(250)).await;

        // Next publish triggers the grace-period check.
        hub.publish(&[sample_committed(&thread_a(), 4)]);
        assert_eq!(
            hub.subscriber_count(&thread_a()),
            0,
            "subscriber should be disconnected after grace period"
        );

        // Drain buffered events.
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(0), got {other:?}"),
        }
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 1),
            other => panic!("expected Event(1), got {other:?}"),
        }

        // Replay-required with last delivered = 1.
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

    // ── Partial batch delivery ─────────────────────────────────────

    #[tokio::test]
    async fn batch_publish_partial_delivery() -> anyhow::Result<()> {
        let hub = zero_grace_hub(2);
        let mut rx = hub.subscribe(&thread_a());

        // Publish a batch of 4 into a capacity-2 buffer.
        // Events 0 and 1 should be delivered; 2 triggers lag.
        hub.publish(&[
            sample_committed(&thread_a(), 0),
            sample_committed(&thread_a(), 1),
            sample_committed(&thread_a(), 2),
            sample_committed(&thread_a(), 3),
        ]);

        // Subscriber is now Lagging. Next publish disconnects (grace=0).
        hub.publish(&[sample_committed(&thread_a(), 4)]);

        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(0), got {other:?}"),
        }
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 1),
            other => panic!("expected Event(1), got {other:?}"),
        }
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

    // ── Receiver dropped cleans up ─────────────────────────────────

    #[tokio::test]
    async fn receiver_dropped_cleans_up_healthy_subscriber() {
        let hub = LiveTailHub::new();
        let rx = hub.subscribe(&thread_a());
        assert_eq!(hub.subscriber_count(&thread_a()), 1);

        drop(rx);

        // Publish triggers cleanup of the closed channel.
        hub.publish(&[sample_committed(&thread_a(), 0)]);
        assert_eq!(hub.subscriber_count(&thread_a()), 0);
    }

    #[tokio::test]
    async fn receiver_dropped_cleans_up_lagging_subscriber() {
        let hub = LiveTailHub::with_config(LiveTailConfig {
            buffer_capacity: 2,
            lag_grace_period: Duration::from_mins(1),
        });
        let rx = hub.subscribe(&thread_a());

        // Fill buffer and trigger lag.
        hub.publish(&[
            sample_committed(&thread_a(), 0),
            sample_committed(&thread_a(), 1),
        ]);
        hub.publish(&[sample_committed(&thread_a(), 2)]);
        assert_eq!(hub.subscriber_count(&thread_a()), 1);

        // Drop the receiver while lagging.
        drop(rx);

        // Next publish detects the closed channel and cleans up
        // immediately — does not wait for the 60s grace period.
        hub.publish(&[sample_committed(&thread_a(), 3)]);
        assert_eq!(hub.subscriber_count(&thread_a()), 0);
    }

    // ── Hub dropped ────────────────────────────────────────────────

    #[tokio::test]
    async fn hub_dropped_receiver_gets_none() {
        let hub = LiveTailHub::new();
        let mut rx = hub.subscribe(&thread_a());

        drop(hub);

        // Normal shutdown — no replay_required flag.
        assert!(rx.recv().await.is_none());
    }

    // ── Clone shares state ─────────────────────────────────────────

    #[tokio::test]
    async fn clone_shares_state() -> anyhow::Result<()> {
        let hub = LiveTailHub::new();
        let clone = hub.clone();

        let mut rx = hub.subscribe(&thread_a());
        clone.publish(&[sample_committed(&thread_a(), 7)]);

        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 7),
            other => panic!("expected Event(7), got {other:?}"),
        }
        Ok(())
    }

    // ── subscriber_count tracking ──────────────────────────────────

    #[tokio::test]
    async fn subscriber_count_tracks_subscriptions() {
        let hub = LiveTailHub::new();
        assert_eq!(hub.subscriber_count(&thread_a()), 0);

        let rx1 = hub.subscribe(&thread_a());
        assert_eq!(hub.subscriber_count(&thread_a()), 1);

        let _rx2 = hub.subscribe(&thread_a());
        assert_eq!(hub.subscriber_count(&thread_a()), 2);

        drop(rx1);
        hub.publish(&[sample_committed(&thread_a(), 0)]);
        assert_eq!(hub.subscriber_count(&thread_a()), 1);
    }

    // ── Fast subscriber unaffected by slow one ─────────────────────

    #[tokio::test]
    async fn fast_subscriber_unaffected_by_slow() -> anyhow::Result<()> {
        let hub = zero_grace_hub(2);

        let _slow_rx = hub.subscribe(&thread_a()); // Never reads.
        let mut fast_rx = hub.subscribe(&thread_a());

        // Publish events beyond slow subscriber's capacity.
        for seq in 0..5 {
            hub.publish(&[sample_committed(&thread_a(), seq)]);
            // Fast subscriber drains immediately.
            match fast_rx.recv().await {
                Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, seq),
                other => panic!("expected Event({seq}), got {other:?}"),
            }
        }

        Ok(())
    }

    // ── Subscriber ID is unique ────────────────────────────────────

    #[tokio::test]
    async fn subscriber_ids_are_unique() {
        let hub = LiveTailHub::new();
        let rx1 = hub.subscribe(&thread_a());
        let rx2 = hub.subscribe(&thread_a());
        let rx3 = hub.subscribe(&thread_b());

        assert_ne!(rx1.subscriber_id(), rx2.subscriber_id());
        // IDs across threads may overlap (scoped to ThreadFanout),
        // which is fine — they're only meaningful within a thread.
        let _ = rx3;
    }

    // ── Zero buffer capacity is clamped ─────────────────────────────

    #[tokio::test]
    async fn zero_buffer_capacity_is_clamped_to_one() -> anyhow::Result<()> {
        let hub = LiveTailHub::with_config(LiveTailConfig {
            buffer_capacity: 0,
            lag_grace_period: Duration::ZERO,
        });
        let mut rx = hub.subscribe(&thread_a());

        hub.publish(&[sample_committed(&thread_a(), 0)]);

        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(0), got {other:?}"),
        }
        Ok(())
    }

    // ── Replay-required returned exactly once ──────────────────────

    #[tokio::test]
    async fn replay_required_returned_exactly_once() -> anyhow::Result<()> {
        let hub = zero_grace_hub(1);
        let mut rx = hub.subscribe(&thread_a());

        // Fill buffer.
        hub.publish(&[sample_committed(&thread_a(), 0)]);
        // Trigger lag.
        hub.publish(&[sample_committed(&thread_a(), 1)]);
        // Disconnect (grace = 0).
        hub.publish(&[sample_committed(&thread_a(), 2)]);

        // Drain buffered event.
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 0),
            other => panic!("expected Event(0), got {other:?}"),
        }

        // First call after drain: ReplayRequired.
        assert!(matches!(
            rx.recv().await,
            Some(LiveTailEvent::ReplayRequired { .. })
        ));

        // Second call: None (replay already signaled).
        assert!(rx.recv().await.is_none());

        Ok(())
    }
}
