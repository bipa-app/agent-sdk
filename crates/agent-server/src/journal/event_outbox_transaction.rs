//! Atomic event + outbox commit hook for durable backends.
//!
//! The in-memory stores satisfy event and outbox writes independently
//! because each mutation is already atomic under its own write lock.
//! Durable SQL backends need a stronger contract: committed events and
//! their corresponding outbox rows must share one database transaction
//! so a crash between the two writes never leaves committed events
//! without matching outbox rows.
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
    /// Maximum relay attempts for each outbox row.
    pub outbox_max_attempts: u32,
    /// Server commit timestamp.
    pub now: OffsetDateTime,
}

/// Outcome of an atomic event + outbox commit.
pub struct EventOutboxCommitOutcome {
    /// The committed events with server-assigned metadata.
    pub committed_events: Vec<CommittedEvent>,
    /// The outbox rows created in the same transaction.
    pub outbox_rows: Vec<OutboxRow>,
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Backend-specific hook for atomically committing events and outbox
/// rows in one transaction.
///
/// Implement this on a durable backend that can coordinate the event
/// insert and outbox insert inside one SQL transaction.  The
/// [`EventRepository`](super::event_repository::EventRepository) can
/// expose this hook so callers that need the outbox guarantee can use
/// the atomic path.
#[async_trait]
pub trait AtomicEventOutboxCommitter: Send + Sync {
    /// Commit events and their corresponding outbox rows atomically.
    ///
    /// Both the committed-event inserts and the outbox-row inserts
    /// execute inside a single SQL transaction.  If the transaction
    /// commits, both exist; if it rolls back, neither does.
    async fn commit_events_with_outbox(
        &self,
        params: EventOutboxCommit,
    ) -> Result<EventOutboxCommitOutcome>;
}
