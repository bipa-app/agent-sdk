//! Optional atomic completed-turn commit hook for durable backends.
//!
//! The in-memory stores satisfy [`super::commit::commit_completed_turn`]
//! by calling the per-store trait methods sequentially because each
//! individual mutation is already atomic under a single write lock.
//! Durable SQL backends need a stronger contract: the attempt close,
//! thread aggregate advance, message projection update, raw message
//! batch append, checkpoint insert, **and** the turn's lifecycle events
//! plus their coalesced advisory outbox row must share one database
//! transaction so recovery never observes a partial turn commit and a
//! committed turn never lacks its persisted events (Phase 10 · D).

use anyhow::Result;
use async_trait::async_trait;

use super::commit::{CommitOutcome, CompletedTurnCommit};

/// Backend-specific hook for atomically committing a completed turn.
///
/// Implement this on a durable backend that can coordinate the
/// completed-turn write set inside one transaction. The generic
/// [`super::commit::commit_completed_turn`] helper will delegate to
/// this hook when the active [`super::thread_store::ThreadStore`]
/// exposes it.
#[async_trait]
pub trait AtomicCompletedTurnCommitter: Send + Sync {
    /// Commit the completed-turn state projections, the turn's
    /// lifecycle events, and the coalesced advisory outbox row
    /// atomically.
    ///
    /// Implementations commit the durable state projections (attempt
    /// close, thread aggregate, message batch/head, checkpoint) and —
    /// when `params.events` is non-empty — the contiguous event batch
    /// plus exactly one
    /// [`OutboxMessageKind::ThreadEventsAvailable`](super::outbox_message::OutboxMessageKind::ThreadEventsAvailable)
    /// row inside the same SQL transaction (Phase 10 · D). The returned
    /// [`CommitOutcome::committed_events`] carries the server-assigned
    /// metadata for those events. An empty `params.events` skips the
    /// event/outbox writes and returns an empty `committed_events`.
    async fn commit_completed_turn_atomic(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome>;
}
