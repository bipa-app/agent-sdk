//! Thread-scoped live event notification hub.
//!
//! [`EventNotifier`] provides the same-process live tail for committed
//! events. Workers call [`EventNotifier::notify`] after persisting a
//! batch of events to the [`EventRepository`](super::event_repository::EventRepository); subscribers receive
//! clones of the committed events through a per-thread broadcast
//! channel.
//!
//! The notifier is **not** the source of truth — the
//! [`EventRepository`](super::event_repository::EventRepository) is. The notifier is a best-effort fan-out
//! channel for same-process consumers. A slow subscriber that falls
//! behind the broadcast buffer receives a `Lagged` error and must
//! re-replay from durable storage to recover the gap.
//!
//! # Design
//!
//! - Each thread gets an independent `broadcast::Sender` created
//!   lazily on first subscribe or notify.
//! - The broadcast buffer size is configurable at construction time.
//! - [`EventNotifier::subscribe`] returns an [`EventReceiver`] that
//!   wraps `broadcast::Receiver<CommittedEvent>`.
//! - [`EventNotifier::notify`] sends each committed event to the
//!   thread's broadcast channel. If no subscribers exist, the send
//!   is a no-op (events are already durable).

use super::committed_event::CommittedEvent;
use agent_sdk_core::ThreadId;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::broadcast;

/// Default broadcast channel capacity per thread.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// A wrapper around a `broadcast::Receiver<CommittedEvent>`.
///
/// Consumers call [`recv`](EventReceiver::recv) in a loop to receive
/// live committed events. A `Lagged` error means the subscriber fell
/// behind and must re-replay from durable storage.
pub struct EventReceiver {
    rx: broadcast::Receiver<CommittedEvent>,
}

impl EventReceiver {
    /// Receive the next committed event from the live tail.
    ///
    /// # Errors
    ///
    /// Returns `Err(RecvError::Lagged(n))` if the subscriber fell
    /// behind by `n` events, or `Err(RecvError::Closed)` if all
    /// senders have been dropped.
    pub async fn recv(&mut self) -> Result<CommittedEvent, broadcast::error::RecvError> {
        self.rx.recv().await
    }
}

/// Thread-scoped live event notification hub.
///
/// Cloning shares the same underlying channel state.
#[derive(Clone)]
pub struct EventNotifier {
    inner: std::sync::Arc<Mutex<EventNotifierInner>>,
    capacity: usize,
}

struct EventNotifierInner {
    channels: HashMap<String, broadcast::Sender<CommittedEvent>>,
}

impl EventNotifier {
    /// Create a new notifier with the default channel capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a new notifier with a custom per-thread channel capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(EventNotifierInner {
                channels: HashMap::new(),
            })),
            capacity,
        }
    }

    /// Subscribe to live committed events for a thread.
    ///
    /// The returned [`EventReceiver`] will receive all events notified
    /// **after** this call. Events committed before the subscribe call
    /// must be obtained via [`EventRepository`](super::event_repository::EventRepository) replay.
    #[must_use]
    pub fn subscribe(&self, thread_id: &ThreadId) -> EventReceiver {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tx = guard
            .channels
            .entry(thread_id.0.clone())
            .or_insert_with(|| broadcast::channel(self.capacity).0);
        let rx = tx.subscribe();
        drop(guard);
        EventReceiver { rx }
    }

    /// Notify subscribers of newly committed events.
    ///
    /// Call this after the events have been durably committed to the
    /// [`EventRepository`](super::event_repository::EventRepository). Events are sent in order. If no
    /// subscribers exist for the thread, this is a no-op.
    pub fn notify(&self, events: &[CommittedEvent]) {
        if events.is_empty() {
            return;
        }
        let thread_key = &events[0].thread_id.0;

        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(tx) = guard.channels.get(thread_key) {
            for event in events {
                // If send fails (no receivers), that's fine — events
                // are already durable.
                let _ = tx.send(event.clone());
            }
        }
        drop(guard);
    }
}

impl Default for EventNotifier {
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
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-notifier-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-notifier-b")
    }

    fn sample_committed(thread_id: &ThreadId, seq: u64) -> CommittedEvent {
        CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: thread_id.clone(),
            sequence: seq,
            timestamp: t0(),
            event: AgentEvent::text("msg_1", "hello"),
        }
    }

    #[tokio::test]
    async fn subscriber_receives_notified_events() -> anyhow::Result<()> {
        let notifier = EventNotifier::new();
        let mut rx = notifier.subscribe(&thread_a());

        let events = vec![
            sample_committed(&thread_a(), 0),
            sample_committed(&thread_a(), 1),
        ];
        notifier.notify(&events);

        let e0 = rx.recv().await?;
        let e1 = rx.recv().await?;
        assert_eq!(e0.sequence, 0);
        assert_eq!(e1.sequence, 1);
        Ok(())
    }

    #[tokio::test]
    async fn subscribers_are_thread_scoped() -> anyhow::Result<()> {
        let notifier = EventNotifier::new();
        let mut rx_a = notifier.subscribe(&thread_a());
        let mut rx_b = notifier.subscribe(&thread_b());

        notifier.notify(&[sample_committed(&thread_a(), 0)]);
        notifier.notify(&[sample_committed(&thread_b(), 0)]);

        let a_event = rx_a.recv().await?;
        let b_event = rx_b.recv().await?;
        assert_eq!(a_event.thread_id, thread_a());
        assert_eq!(b_event.thread_id, thread_b());
        Ok(())
    }

    #[tokio::test]
    async fn notify_without_subscribers_is_noop() {
        let notifier = EventNotifier::new();
        notifier.notify(&[sample_committed(&thread_a(), 0)]);
    }

    #[tokio::test]
    async fn multiple_subscribers_same_thread() -> anyhow::Result<()> {
        let notifier = EventNotifier::new();
        let mut rx1 = notifier.subscribe(&thread_a());
        let mut rx2 = notifier.subscribe(&thread_a());

        notifier.notify(&[sample_committed(&thread_a(), 5)]);

        let e1 = rx1.recv().await?;
        let e2 = rx2.recv().await?;
        assert_eq!(e1.sequence, 5);
        assert_eq!(e2.sequence, 5);
        Ok(())
    }

    #[tokio::test]
    async fn empty_notify_is_noop() {
        let notifier = EventNotifier::new();
        let _rx = notifier.subscribe(&thread_a());
        notifier.notify(&[]);
    }

    #[tokio::test]
    async fn lagged_subscriber_gets_error() -> anyhow::Result<()> {
        // Tiny buffer to force lagging.
        let notifier = EventNotifier::with_capacity(2);
        let mut rx = notifier.subscribe(&thread_a());

        // Send 5 events into a capacity-2 buffer.
        for seq in 0..5 {
            notifier.notify(&[sample_committed(&thread_a(), seq)]);
        }

        // First recv should report lagged.
        let result = rx.recv().await;
        assert!(
            matches!(result, Err(broadcast::error::RecvError::Lagged(_))),
            "expected Lagged, got {result:?}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn clone_shares_state() -> anyhow::Result<()> {
        let notifier = EventNotifier::new();
        let clone = notifier.clone();

        let mut rx = notifier.subscribe(&thread_a());
        clone.notify(&[sample_committed(&thread_a(), 42)]);

        let event = rx.recv().await?;
        assert_eq!(event.sequence, 42);
        Ok(())
    }
}
