//! Atomic event + outbox commit hook for durable backends.
//!
//! The in-memory stores satisfy event and outbox writes independently
//! because each mutation is already atomic under its own write lock.
//! Durable SQL backends need a stronger contract: committed events and
//! their corresponding outbox row must share one database transaction
//! so a crash between the two writes never leaves committed events
//! without a matching outbox advisory.
//!
//! # Phase 8.1 contract: one advisory row per commit batch
//!
//! Each call writes **exactly one**
//! [`OutboxMessageKind::ThreadEventsAvailable`](super::outbox_message::OutboxMessageKind::ThreadEventsAvailable)
//! row whose payload carries `{ thread_id, last_sequence }` — the
//! highest sequence committed in this batch.  Consumers receive a
//! coalesced hint instead of N per-event rows; they replay any suffix
//! they care about by reading
//! [`agent_sdk_committed_events`](super::event_repository::EventRepository)
//! using their own cursor.  This is the only path that may write a
//! `thread_events_available` row.
//!
//! This hook follows the same pattern as
//! [`AtomicCompletedTurnCommitter`](super::completed_turn_transaction::AtomicCompletedTurnCommitter).

use agent_sdk_core::ThreadId;
use agent_sdk_core::events::AgentEvent;
use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::outbox::OutboxRow;

// ─────────────────────────────────────────────────────────────────────
// Params
// ─────────────────────────────────────────────────────────────────────

/// Parameters for an atomic event + outbox commit.
pub struct EventOutboxCommit {
    /// Thread to commit events on.
    pub thread_id: ThreadId,
    /// Events to commit (assigned contiguous sequences atomically).
    pub events: Vec<AgentEvent>,
    /// Maximum relay attempts for the coalesced outbox row.
    pub outbox_max_attempts: u32,
    /// Server commit timestamp.
    pub now: OffsetDateTime,
}

/// Outcome of an atomic event + outbox commit.
///
/// `outbox_row` is `None` only if the input batch is empty (which the
/// implementation rejects, so in practice `outbox_row` is always
/// `Some`).  We expose it as `Option` rather than `OutboxRow` so a
/// future variant that opts out of relay (e.g. internal-only events)
/// can land without breaking callers.
pub struct EventOutboxCommitOutcome {
    /// The committed events with server-assigned metadata.
    pub committed_events: Vec<CommittedEvent>,
    /// The single advisory outbox row created for this batch.
    pub outbox_row: Option<OutboxRow>,
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Backend-specific hook for atomically committing events and the
/// matching advisory outbox row in one transaction.
///
/// Implement this on a durable backend that can coordinate the event
/// insert and outbox insert inside one SQL transaction.  The
/// [`EventRepository`](super::event_repository::EventRepository) can
/// expose this hook so callers that need the outbox guarantee use the
/// atomic path.
#[async_trait]
pub trait AtomicEventOutboxCommitter: Send + Sync {
    /// Commit events and write one coalesced
    /// [`OutboxMessageKind::ThreadEventsAvailable`](super::outbox_message::OutboxMessageKind::ThreadEventsAvailable)
    /// row atomically.
    ///
    /// Both the committed-event inserts and the single outbox-row
    /// insert execute inside one SQL transaction.  If the transaction
    /// commits, both exist; if it rolls back, neither does.
    async fn commit_events_with_outbox(
        &self,
        params: EventOutboxCommit,
    ) -> Result<EventOutboxCommitOutcome>;
}
