//! Broker boundary: the narrow trait every broker transport implements.
//!
//! Phase 8.1 splits the relay path into four layers so each can be
//! tested and replaced in isolation:
//!
//! ```text
//!     OutboxStore  ─►  RelayWorker  ─►  Publisher  ─►  BrokerAdapter
//! ```
//!
//! [`BrokerAdapter`] is the lowest layer.  An AMQP-specific
//! implementation lives outside this crate (Phase 8.2 will land it
//! inside `agent-service-host`).  Tests use [`InMemoryBrokerAdapter`].
//!
//! # Contract
//!
//! 1. **At-least-once.** A successful `publish` call MUST eventually
//!    result in the message being delivered to subscribers.  The
//!    broker may republish on retry; consumers must be idempotent.
//! 2. **No ordering guarantees across kinds.**  A
//!    [`OutboxMessageKind::TaskWakeup`] message published after a
//!    [`OutboxMessageKind::ThreadEventsAvailable`] message may arrive
//!    before it.  Consumers MUST NOT rely on broker order.
//! 3. **Advisory only.** The payload carried by an [`OutboxMessage`]
//!    is a reference to durable state (see
//!    [`super::outbox_message`]); the broker has no authority over
//!    that state.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::RwLock;

use super::outbox_message::{OutboxMessage, OutboxMessageKind};

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Narrow boundary every broker transport implements.
///
/// Implementations should be cheap to clone (typically `Arc`-wrapped
/// internally) so they can be passed to multiple publishers concurrently.
#[async_trait]
pub trait BrokerAdapter: Send + Sync {
    /// Publish one advisory message to the broker.
    ///
    /// The adapter is responsible for translating [`OutboxMessage`]
    /// into the broker's wire format (e.g. an AMQP basic.publish call
    /// with the kind embedded in a header).
    ///
    /// # Errors
    /// Returns an error if the broker rejects the publish or the
    /// adapter cannot reach the broker.  Callers (typically a
    /// [`super::relay::Publisher`]) treat errors as transient — the
    /// outbox row stays in `Pending` for retry.
    async fn publish(&self, message: &OutboxMessage) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory bounded adapter
// ─────────────────────────────────────────────────────────────────────

/// Default cap on the in-memory adapter's recording buffer.
///
/// The host wires this adapter as the default broker, so an unbounded
/// buffer would leak memory proportional to total outbox throughput on
/// a long-running deploy.  The buffer is a bounded ring: once full it
/// drops the oldest recorded message (and warns), keeping memory flat
/// while still letting tests assert on recent publishes.
const DEFAULT_RECORD_CAPACITY: usize = 1024;

#[derive(Default)]
struct InMemoryBrokerAdapterInner {
    published: VecDeque<OutboxMessage>,
    dropped: u64,
}

/// In-memory adapter that records recent published messages in a
/// bounded ring buffer.
///
/// Cloning shares the underlying buffer so a publisher driven from
/// one task and assertions made from another see the same state.
/// Suitable both as a test double and as the host's default
/// no-broker-configured adapter: the default-capacity cap bounds memory
/// regardless of throughput.
#[derive(Clone)]
pub struct InMemoryBrokerAdapter {
    inner: Arc<RwLock<InMemoryBrokerAdapterInner>>,
    capacity: usize,
}

impl Default for InMemoryBrokerAdapter {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_RECORD_CAPACITY)
    }
}

impl InMemoryBrokerAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an adapter whose recording buffer retains at most
    /// `capacity` of the most recently published messages (clamped to a
    /// minimum of 1).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InMemoryBrokerAdapterInner::default())),
            capacity: capacity.max(1),
        }
    }

    /// Snapshot of the retained messages, oldest first.
    ///
    /// Only the most recent [`capacity`](Self::with_capacity) messages
    /// are retained; older ones may have been dropped (see
    /// [`dropped_count`](Self::dropped_count)).
    pub async fn published(&self) -> Vec<OutboxMessage> {
        self.inner.read().await.published.iter().cloned().collect()
    }

    /// Number of retained messages (never exceeds the capacity).
    pub async fn published_count(&self) -> usize {
        self.inner.read().await.published.len()
    }

    /// Number of retained messages of a specific kind.
    pub async fn published_count_of(&self, kind: OutboxMessageKind) -> usize {
        self.inner
            .read()
            .await
            .published
            .iter()
            .filter(|message| message.kind() == kind)
            .count()
    }

    /// Number of messages evicted from the ring buffer because it was
    /// at capacity when a newer message arrived.
    pub async fn dropped_count(&self) -> u64 {
        self.inner.read().await.dropped
    }
}

#[async_trait]
impl BrokerAdapter for InMemoryBrokerAdapter {
    async fn publish(&self, message: &OutboxMessage) -> Result<()> {
        let mut inner = self.inner.write().await;
        if inner.published.len() >= self.capacity {
            inner.published.pop_front();
            inner.dropped += 1;
            log::warn!(
                capacity = self.capacity as u64;
                "InMemoryBrokerAdapter recording buffer full; dropping oldest message",
            );
        }
        inner.published.push_back(message.clone());
        drop(inner);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::outbox_message::{TaskWakeupPayload, ThreadEventsAvailablePayload};
    use crate::journal::task::AgentTaskId;
    use agent_sdk_foundation::ThreadId;

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-broker-a")
    }

    fn task_message() -> OutboxMessage {
        OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id: AgentTaskId::from_string("task_broker_a"),
            thread_id: thread_a(),
        })
    }

    fn events_message(seq: u64) -> OutboxMessage {
        OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: seq,
        })
    }

    #[tokio::test]
    async fn in_memory_adapter_records_publish_order() -> Result<()> {
        let adapter = InMemoryBrokerAdapter::new();
        adapter.publish(&task_message()).await?;
        adapter.publish(&events_message(0)).await?;
        adapter.publish(&events_message(1)).await?;

        let published = adapter.published().await;
        assert_eq!(published.len(), 3);
        assert_eq!(published[0].kind(), OutboxMessageKind::TaskWakeup);
        assert_eq!(
            published[1].kind(),
            OutboxMessageKind::ThreadEventsAvailable
        );
        assert_eq!(
            published[2].kind(),
            OutboxMessageKind::ThreadEventsAvailable
        );
        Ok(())
    }

    #[tokio::test]
    async fn in_memory_adapter_counts_by_kind() -> Result<()> {
        let adapter = InMemoryBrokerAdapter::new();
        adapter.publish(&task_message()).await?;
        adapter.publish(&task_message()).await?;
        adapter.publish(&events_message(0)).await?;

        assert_eq!(adapter.published_count().await, 3);
        assert_eq!(
            adapter
                .published_count_of(OutboxMessageKind::TaskWakeup)
                .await,
            2,
        );
        assert_eq!(
            adapter
                .published_count_of(OutboxMessageKind::ThreadEventsAvailable)
                .await,
            1,
        );
        Ok(())
    }

    #[tokio::test]
    async fn in_memory_adapter_handles_concurrent_publishes() -> Result<()> {
        let adapter = InMemoryBrokerAdapter::new();
        let mut handles = Vec::new();

        for n in 0..32u64 {
            let adapter = adapter.clone();
            handles.push(tokio::spawn(async move {
                adapter.publish(&events_message(n)).await
            }));
        }

        for handle in handles {
            handle.await??;
        }

        assert_eq!(adapter.published_count().await, 32);
        // We don't assert ordering — concurrent publishers may
        // interleave.  We do assert no message was dropped.
        Ok(())
    }

    #[tokio::test]
    async fn adapter_can_be_used_through_dyn_trait() -> Result<()> {
        let adapter: Arc<dyn BrokerAdapter> = Arc::new(InMemoryBrokerAdapter::new());
        adapter.publish(&task_message()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn recording_buffer_is_bounded_and_drops_oldest() -> Result<()> {
        // Capacity 2: publishing 5 messages keeps only the last 2 and
        // counts the 3 evictions, so memory stays flat under sustained
        // throughput (this adapter is the host's default broker).
        let adapter = InMemoryBrokerAdapter::with_capacity(2);
        for seq in 0..5u64 {
            adapter.publish(&events_message(seq)).await?;
        }

        assert_eq!(adapter.published_count().await, 2);
        assert_eq!(adapter.dropped_count().await, 3);

        let retained = adapter.published().await;
        let retained_seqs: Vec<u64> = retained
            .iter()
            .map(|m| match m {
                OutboxMessage::ThreadEventsAvailable(p) => p.last_sequence,
                OutboxMessage::TaskWakeup(_) => panic!("unexpected kind"),
            })
            .collect();
        assert_eq!(retained_seqs, vec![3, 4]);
        Ok(())
    }
}
