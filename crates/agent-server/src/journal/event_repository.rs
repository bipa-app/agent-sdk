//! Append-only event repository with thread-scoped sequencing.
//!
//! [`EventRepository`] is the authoritative commit surface for durable
//! events in the server.  Every event committed through this trait
//! receives a server-owned `event_id` (UUID v7), a monotonic per-thread
//! `sequence`, and a commit-time `timestamp`.  The `(thread_id,
//! sequence)` pair is unique and enforced by the repository.
//!
//! Workers submit raw [`AgentEvent`]s; the repository allocates
//! metadata and persists the committed record atomically.  This
//! replaces SDK-local [`SequenceCounter`](agent_sdk_core::events::SequenceCounter)
//! semantics for the authoritative server path.
//!
//! # Ownership rule
//!
//! Event sequence allocation and persistence happen under the same
//! write lock, so no two callers can observe the same sequence for a
//! thread.  A database backend would use a serialisable transaction
//! with `SELECT … FOR UPDATE` on the thread's sequence high-water mark.
//!
//! # Batched append
//!
//! [`EventRepository::commit_event_batch`] assigns a contiguous range
//! of sequences to a batch of events in input order.  Workers use this
//! to commit an entire turn's event stream atomically, preventing
//! interleaved sequences from concurrent turns on the same thread.

use super::committed_event::CommittedEvent;
use super::event_outbox_transaction::AtomicEventOutboxCommitter;
use agent_sdk_core::ThreadId;
use agent_sdk_core::events::AgentEvent;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::sync::RwLock;

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Append-only repository for committed events with thread-scoped
/// sequencing.
///
/// Every mutation allocates server-owned metadata (UUID v7 `event_id`,
/// monotonic `sequence`, commit-time `timestamp`) atomically with
/// persistence.  The `(thread_id, sequence)` pair is unique.
///
/// # Errors
///
/// All methods return `anyhow::Result` so implementations can surface
/// storage-specific errors with context.
#[async_trait]
pub trait EventRepository: Send + Sync {
    /// Optional backend-specific hook for atomically committing events
    /// and the matching coalesced advisory outbox row in one
    /// transaction.
    ///
    /// In-memory stores leave this as `None`; durable backends such as
    /// `Postgres` and `SQLite` override it to surface the
    /// [`commit_events_with_outbox`](super::event_outbox_transaction::AtomicEventOutboxCommitter::commit_events_with_outbox)
    /// unit of work.  Phase 8.1: every call writes exactly one
    /// [`OutboxMessageKind::ThreadEventsAvailable`](super::outbox_message::OutboxMessageKind::ThreadEventsAvailable)
    /// outbox row carrying an advisory `{thread_id, last_sequence}`
    /// payload.
    #[must_use]
    fn atomic_event_outbox_committer(&self) -> Option<&dyn AtomicEventOutboxCommitter> {
        None
    }

    /// Commit a single event to the given thread.
    ///
    /// Allocates a UUID v7 `event_id`, the next monotonic `sequence`,
    /// and sets the commit `timestamp` to `now`.
    async fn commit_event(
        &self,
        thread_id: &ThreadId,
        event: AgentEvent,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent>;

    /// Commit a batch of events to the given thread.
    ///
    /// Events are assigned contiguous sequence numbers in input order.
    /// The batch is persisted atomically — either all events commit or
    /// none do.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty.
    async fn commit_event_batch(
        &self,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>>;

    /// Get the next sequence number that would be assigned for a thread.
    ///
    /// Returns 0 if no events have been committed for this thread.
    async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64>;

    /// Retrieve all committed events for a thread in sequence order.
    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>>;

    /// Retrieve committed events for a thread with `sequence > after_sequence`
    /// and `sequence <= up_to_sequence`, in sequence order.
    ///
    /// This is the replay query surface: a reconnecting client passes its
    /// last-seen sequence as `after_sequence` and the captured watermark
    /// as `up_to_sequence` to get exactly the committed events it missed.
    async fn get_events_in_range(
        &self,
        thread_id: &ThreadId,
        after_sequence: u64,
        up_to_sequence: u64,
    ) -> Result<Vec<CommittedEvent>>;

    /// Return distinct thread IDs that have at least one committed event
    /// with `committed_at < cutoff`, limited to `limit` results.
    ///
    /// Used by the retention janitor to identify threads with events
    /// eligible for TTL-based purge.
    async fn threads_with_events_before(
        &self,
        cutoff: OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>>;

    /// Return the highest sequence number among events with
    /// `committed_at < cutoff` for a given thread.
    ///
    /// Returns `None` if no events match.  The retention janitor uses
    /// this as the time-based candidate retention floor.
    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct InMemoryEventRepositoryInner {
    /// Events keyed by `thread_id`, stored in append order (which is
    /// also sequence order).
    events: HashMap<String, Vec<CommittedEvent>>,
}

impl InMemoryEventRepositoryInner {
    fn next_sequence(&self, thread_id: &ThreadId) -> u64 {
        self.events
            .get(&thread_id.0)
            .and_then(|events| events.last())
            .map_or(0, |last| last.sequence + 1)
    }
}

/// In-memory reference implementation of [`EventRepository`].
///
/// Sequence allocation and persistence happen under a single write
/// lock, guaranteeing `(thread_id, sequence)` uniqueness by
/// construction.
///
/// Cloning this type shares the same underlying event journal.
#[derive(Clone, Default)]
pub struct InMemoryEventRepository {
    inner: Arc<RwLock<InMemoryEventRepositoryInner>>,
}

impl InMemoryEventRepository {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EventRepository for InMemoryEventRepository {
    async fn commit_event(
        &self,
        thread_id: &ThreadId,
        event: AgentEvent,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        let mut inner = self.inner.write().await;
        let seq = inner.next_sequence(thread_id);

        let committed = CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: thread_id.clone(),
            sequence: seq,
            timestamp: now,
            event,
        };

        inner
            .events
            .entry(thread_id.0.clone())
            .or_default()
            .push(committed.clone());
        drop(inner);

        Ok(committed)
    }

    async fn commit_event_batch(
        &self,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        ensure!(!events.is_empty(), "cannot commit an empty event batch");

        let mut inner = self.inner.write().await;
        let start_seq = inner.next_sequence(thread_id);

        let committed: Vec<CommittedEvent> = events
            .into_iter()
            .zip(start_seq..)
            .map(|(event, seq)| CommittedEvent {
                event_id: uuid::Uuid::now_v7(),
                thread_id: thread_id.clone(),
                sequence: seq,
                timestamp: now,
                event,
            })
            .collect();

        inner
            .events
            .entry(thread_id.0.clone())
            .or_default()
            .extend(committed.clone());
        drop(inner);

        Ok(committed)
    }

    async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
        let inner = self.inner.read().await;
        Ok(inner.next_sequence(thread_id))
    }

    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
        let inner = self.inner.read().await;
        Ok(inner.events.get(&thread_id.0).cloned().unwrap_or_default())
    }

    async fn get_events_in_range(
        &self,
        thread_id: &ThreadId,
        after_sequence: u64,
        up_to_sequence: u64,
    ) -> Result<Vec<CommittedEvent>> {
        let inner = self.inner.read().await;
        let result = inner
            .events
            .get(&thread_id.0)
            .map(|evts| {
                evts.iter()
                    .filter(|e| e.sequence > after_sequence && e.sequence <= up_to_sequence)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        drop(inner);
        Ok(result)
    }

    async fn threads_with_events_before(
        &self,
        cutoff: OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        let inner = self.inner.read().await;
        let threads: Vec<ThreadId> = inner
            .events
            .iter()
            .filter(|(_, evts)| evts.iter().any(|e| e.timestamp < cutoff))
            .map(|(tid, _)| ThreadId::from_string(tid.clone()))
            .take(limit as usize)
            .collect();
        drop(inner);
        Ok(threads)
    }

    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        let inner = self.inner.read().await;
        let max_seq = inner.events.get(&thread_id.0).and_then(|evts| {
            evts.iter()
                .filter(|e| e.timestamp < cutoff)
                .map(|e| e.sequence)
                .max()
        });
        drop(inner);
        Ok(max_seq)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-event-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-event-b")
    }

    fn sample_event() -> AgentEvent {
        AgentEvent::text("msg_test", "hello")
    }

    // ── single-event commit ─────────────────────────────────────────

    #[tokio::test]
    async fn commit_single_event_assigns_sequence_zero() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let committed = repo.commit_event(&thread_a(), sample_event(), t0()).await?;

        assert_eq!(committed.sequence, 0);
        assert_eq!(committed.thread_id, thread_a());
        assert_eq!(committed.timestamp, t0());
        Ok(())
    }

    #[tokio::test]
    async fn commit_single_event_assigns_uuid_v7() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let committed = repo.commit_event(&thread_a(), sample_event(), t0()).await?;

        assert_eq!(
            committed.event_id.get_version(),
            Some(uuid::Version::SortRand),
        );
        Ok(())
    }

    #[tokio::test]
    async fn commit_single_event_preserves_payload() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let committed = repo
            .commit_event(&thread_a(), AgentEvent::text("msg_42", "content"), t0())
            .await?;

        match &committed.event {
            AgentEvent::Text { message_id, text } => {
                assert_eq!(message_id, "msg_42");
                assert_eq!(text, "content");
            }
            other => panic!("expected Text, got {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn commit_multiple_events_yields_monotonic_sequences() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let e0 = repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        let e1 = repo
            .commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;
        let e2 = repo
            .commit_event(&thread_a(), sample_event(), t_plus(2))
            .await?;

        assert_eq!(e0.sequence, 0);
        assert_eq!(e1.sequence, 1);
        assert_eq!(e2.sequence, 2);
        Ok(())
    }

    #[tokio::test]
    async fn commit_events_have_unique_event_ids() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let mut ids = Vec::new();

        for i in 0..10 {
            let committed = repo
                .commit_event(&thread_a(), sample_event(), t_plus(i))
                .await?;
            ids.push(committed.event_id);
        }

        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
        Ok(())
    }

    // ── batched commit ──────────────────────────────────────────────

    #[tokio::test]
    async fn batch_commit_assigns_contiguous_sequences() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let events = vec![sample_event(), sample_event(), sample_event()];
        let committed = repo.commit_event_batch(&thread_a(), events, t0()).await?;

        assert_eq!(committed.len(), 3);
        assert_eq!(committed[0].sequence, 0);
        assert_eq!(committed[1].sequence, 1);
        assert_eq!(committed[2].sequence, 2);
        Ok(())
    }

    #[tokio::test]
    async fn batch_commit_preserves_input_order() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let events = vec![
            AgentEvent::text("msg_a", "first"),
            AgentEvent::text("msg_b", "second"),
            AgentEvent::text("msg_c", "third"),
        ];
        let committed = repo.commit_event_batch(&thread_a(), events, t0()).await?;

        match &committed[0].event {
            AgentEvent::Text { text, .. } => assert_eq!(text, "first"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &committed[1].event {
            AgentEvent::Text { text, .. } => assert_eq!(text, "second"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &committed[2].event {
            AgentEvent::Text { text, .. } => assert_eq!(text, "third"),
            other => panic!("expected Text, got {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn batch_continues_from_last_sequence() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        // Commit 2 individual events first.
        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;

        // Then a batch of 3 — sequences must continue.
        let batch = repo
            .commit_event_batch(
                &thread_a(),
                vec![sample_event(), sample_event(), sample_event()],
                t_plus(2),
            )
            .await?;

        assert_eq!(batch[0].sequence, 2);
        assert_eq!(batch[1].sequence, 3);
        assert_eq!(batch[2].sequence, 4);
        Ok(())
    }

    #[tokio::test]
    async fn batch_commit_assigns_unique_event_ids() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let committed = repo
            .commit_event_batch(
                &thread_a(),
                vec![sample_event(), sample_event(), sample_event()],
                t0(),
            )
            .await?;

        let ids: HashSet<_> = committed.iter().map(|e| e.event_id).collect();
        assert_eq!(ids.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn batch_commit_assigns_consistent_timestamp() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let committed = repo
            .commit_event_batch(
                &thread_a(),
                vec![sample_event(), sample_event()],
                t_plus(99),
            )
            .await?;

        for event in &committed {
            assert_eq!(event.timestamp, t_plus(99));
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_batch_is_rejected() -> anyhow::Result<()> {
        let repo = InMemoryEventRepository::new();
        let result = repo.commit_event_batch(&thread_a(), vec![], t0()).await;

        assert!(result.is_err(), "expected empty batch to be rejected");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty event batch"));
        Ok(())
    }

    // ── thread isolation ────────────────────────────────────────────

    #[tokio::test]
    async fn different_threads_have_independent_sequences() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let a0 = repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        let b0 = repo
            .commit_event(&thread_b(), sample_event(), t_plus(1))
            .await?;
        let a1 = repo
            .commit_event(&thread_a(), sample_event(), t_plus(2))
            .await?;
        let b1 = repo
            .commit_event(&thread_b(), sample_event(), t_plus(3))
            .await?;

        // Both threads start at 0 independently.
        assert_eq!(a0.sequence, 0);
        assert_eq!(b0.sequence, 0);
        assert_eq!(a1.sequence, 1);
        assert_eq!(b1.sequence, 1);
        Ok(())
    }

    #[tokio::test]
    async fn get_events_isolated_between_threads() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;
        repo.commit_event(&thread_b(), sample_event(), t_plus(2))
            .await?;

        let a_events = repo.get_events(&thread_a()).await?;
        let b_events = repo.get_events(&thread_b()).await?;
        assert_eq!(a_events.len(), 2);
        assert_eq!(b_events.len(), 1);
        Ok(())
    }

    // ── retrieval ───────────────────────────────────────────────────

    #[tokio::test]
    async fn get_events_returns_in_sequence_order() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        repo.commit_event(&thread_a(), AgentEvent::text("msg_1", "first"), t0())
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("msg_2", "second"), t_plus(1))
            .await?;
        repo.commit_event(&thread_a(), AgentEvent::text("msg_3", "third"), t_plus(2))
            .await?;

        let events = repo.get_events(&thread_a()).await?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[1].sequence, 1);
        assert_eq!(events[2].sequence, 2);
        Ok(())
    }

    #[tokio::test]
    async fn get_events_returns_empty_for_unknown_thread() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let events = repo.get_events(&thread_a()).await?;
        assert!(events.is_empty());
        Ok(())
    }

    // ── next_sequence ───────────────────────────────────────────────

    #[tokio::test]
    async fn next_sequence_starts_at_zero() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        assert_eq!(repo.next_sequence(&thread_a()).await?, 0);
        Ok(())
    }

    #[tokio::test]
    async fn next_sequence_advances_after_commits() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        assert_eq!(repo.next_sequence(&thread_a()).await?, 1);

        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;
        assert_eq!(repo.next_sequence(&thread_a()).await?, 2);
        Ok(())
    }

    #[tokio::test]
    async fn next_sequence_advances_after_batch() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        repo.commit_event_batch(
            &thread_a(),
            vec![sample_event(), sample_event(), sample_event()],
            t0(),
        )
        .await?;

        assert_eq!(repo.next_sequence(&thread_a()).await?, 3);
        Ok(())
    }

    #[tokio::test]
    async fn next_sequence_independent_per_thread() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;

        assert_eq!(repo.next_sequence(&thread_a()).await?, 2);
        assert_eq!(repo.next_sequence(&thread_b()).await?, 0);
        Ok(())
    }

    // ── (thread_id, sequence) uniqueness ────────────────────────────
    //
    // In the in-memory implementation, uniqueness is guaranteed by
    // construction (sequences are monotonically allocated under a
    // write lock).  This test verifies the invariant holds across
    // mixed single + batch operations.

    #[tokio::test]
    async fn no_duplicate_sequences_across_mixed_operations() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        // Single commits.
        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;

        // Batch commit.
        repo.commit_event_batch(&thread_a(), vec![sample_event(), sample_event()], t_plus(2))
            .await?;

        // More single commits.
        repo.commit_event(&thread_a(), sample_event(), t_plus(3))
            .await?;

        let events = repo.get_events(&thread_a()).await?;
        let sequences: Vec<u64> = events.iter().map(|e| e.sequence).collect();
        assert_eq!(sequences, vec![0, 1, 2, 3, 4]);
        Ok(())
    }

    // ── clone shares state ──────────────────────────────────────────

    #[tokio::test]
    async fn cloned_repository_shares_state() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let clone = repo.clone();

        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        let events = clone.get_events(&thread_a()).await?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 0);

        // Sequence continues from clone's perspective.
        let committed = clone
            .commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;
        assert_eq!(committed.sequence, 1);
        Ok(())
    }

    // ── all event_ids are UUID v7 ───────────────────────────────────

    #[tokio::test]
    async fn all_committed_events_have_v7_uuids() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        // Single events.
        let single = repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        assert_eq!(single.event_id.get_version(), Some(uuid::Version::SortRand),);

        // Batch events.
        let batch = repo
            .commit_event_batch(&thread_a(), vec![sample_event(), sample_event()], t_plus(1))
            .await?;

        for event in &batch {
            assert_eq!(event.event_id.get_version(), Some(uuid::Version::SortRand),);
        }
        Ok(())
    }

    // ── get_events_in_range ────────────────────────────────────────

    #[tokio::test]
    async fn get_events_in_range_returns_correct_window() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        // Commit 5 events: sequences 0..4.
        for i in 0..5 {
            repo.commit_event(&thread_a(), sample_event(), t_plus(i))
                .await?;
        }

        // Range (1, 3] → sequences 2, 3.
        let events = repo.get_events_in_range(&thread_a(), 1, 3).await?;
        let seqs: Vec<u64> = events.iter().map(|e| e.sequence).collect();
        assert_eq!(seqs, vec![2, 3]);
        Ok(())
    }

    #[tokio::test]
    async fn get_events_in_range_empty_when_after_equals_up_to() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;

        let events = repo.get_events_in_range(&thread_a(), 1, 1).await?;
        assert!(events.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn get_events_in_range_empty_for_unknown_thread() -> Result<()> {
        let repo = InMemoryEventRepository::new();
        let events = repo.get_events_in_range(&thread_a(), 0, 10).await?;
        assert!(events.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn get_events_in_range_full_range() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        for i in 0..3 {
            repo.commit_event(&thread_a(), sample_event(), t_plus(i))
                .await?;
        }

        // Range (0, 2] → sequences 1, 2. Excludes 0.
        let events = repo.get_events_in_range(&thread_a(), 0, 2).await?;
        let seqs: Vec<u64> = events.iter().map(|e| e.sequence).collect();
        assert_eq!(seqs, vec![1, 2]);
        Ok(())
    }

    #[tokio::test]
    async fn get_events_in_range_thread_isolated() -> Result<()> {
        let repo = InMemoryEventRepository::new();

        repo.commit_event(&thread_a(), sample_event(), t0()).await?;
        repo.commit_event(&thread_a(), sample_event(), t_plus(1))
            .await?;
        repo.commit_event(&thread_b(), sample_event(), t0()).await?;

        let a_events = repo.get_events_in_range(&thread_a(), 0, 1).await?;
        assert_eq!(a_events.len(), 1);
        assert_eq!(a_events[0].thread_id, thread_a());

        let b_events = repo.get_events_in_range(&thread_b(), 0, 10).await?;
        assert!(b_events.is_empty());
        Ok(())
    }
}
