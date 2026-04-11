//! Server-owned committed event record.
//!
//! A [`CommittedEvent`] is the durable record produced by the server's
//! event commit path.  Unlike [`AgentEventEnvelope`] which may be
//! allocated by SDK-local code, a `CommittedEvent` always carries
//! server-authoritative metadata:
//!
//! - `event_id` — UUID v7 (time-ordered, server-allocated)
//! - `sequence` — monotonic per-thread, contiguous within batches
//! - `timestamp` — server commit-time
//!
//! The `(thread_id, sequence)` pair uniquely identifies a committed
//! event and is enforced by the
//! [`EventRepository`](super::event_repository::EventRepository).

use agent_sdk_core::ThreadId;
use agent_sdk_core::events::{AgentEvent, AgentEventEnvelope};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

// ─────────────────────────────────────────────────────────────────────
// CommittedEvent
// ─────────────────────────────────────────────────────────────────────

/// A committed event record with server-authoritative metadata.
///
/// The `(thread_id, sequence)` pair is the durable unique key.  The
/// `event_id` (UUID v7) provides a globally unique, time-ordered
/// identifier.  The `timestamp` is the server commit-time.
///
/// Call [`into_envelope`](CommittedEvent::into_envelope) to convert to
/// an [`AgentEventEnvelope`] for client wire transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommittedEvent {
    /// Globally unique identifier (UUID v7, time-ordered).
    pub event_id: uuid::Uuid,
    /// Thread this event belongs to.
    pub thread_id: ThreadId,
    /// Monotonically increasing sequence within the thread.
    pub sequence: u64,
    /// Server commit-time (UTC).
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// The event payload.
    pub event: AgentEvent,
}

impl CommittedEvent {
    /// Consume and convert to an [`AgentEventEnvelope`] for wire
    /// transmission.
    ///
    /// The envelope carries `event_id`, `sequence`, `timestamp`, and
    /// the inner event — everything a client needs for idempotent
    /// consumption and ordering.  `thread_id` is dropped because the
    /// envelope's delivery channel already implies it.
    #[must_use]
    pub fn into_envelope(self) -> AgentEventEnvelope {
        AgentEventEnvelope {
            event_id: self.event_id,
            sequence: self.sequence,
            timestamp: self.timestamp,
            event: self.event,
        }
    }

    /// Borrow-convert to an [`AgentEventEnvelope`].
    #[must_use]
    pub fn to_envelope(&self) -> AgentEventEnvelope {
        AgentEventEnvelope {
            event_id: self.event_id,
            sequence: self.sequence,
            timestamp: self.timestamp,
            event: self.event.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn sample_committed(seq: u64) -> CommittedEvent {
        CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: ThreadId::from_string("t-event-test"),
            sequence: seq,
            timestamp: t0(),
            event: AgentEvent::text("msg_1", "hello"),
        }
    }

    // ── envelope conversion ─────────────────────────────────────────

    #[test]
    fn into_envelope_preserves_fields() {
        let committed = sample_committed(42);
        let event_id = committed.event_id;
        let timestamp = committed.timestamp;
        let envelope = committed.into_envelope();

        assert_eq!(envelope.event_id, event_id);
        assert_eq!(envelope.sequence, 42);
        assert_eq!(envelope.timestamp, timestamp);
        match &envelope.event {
            AgentEvent::Text { message_id, text } => {
                assert_eq!(message_id, "msg_1");
                assert_eq!(text, "hello");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn to_envelope_preserves_fields() {
        let committed = sample_committed(7);
        let envelope = committed.to_envelope();

        assert_eq!(envelope.event_id, committed.event_id);
        assert_eq!(envelope.sequence, 7);
        assert_eq!(envelope.timestamp, committed.timestamp);
    }

    #[test]
    fn into_envelope_serializes_flat() {
        let committed = sample_committed(0);
        let envelope = committed.into_envelope();
        let json: serde_json::Value = serde_json::to_value(&envelope).expect("serialize");

        // Flattened: type at top level, no nested "event" key
        assert!(json.get("type").is_some());
        assert!(json.get("event").is_none());
    }

    // ── serialization ───────────────────────────────────────────────

    #[test]
    fn committed_event_round_trips_through_json() -> anyhow::Result<()> {
        let committed = sample_committed(5);
        let json = serde_json::to_string(&committed)?;
        let restored: CommittedEvent = serde_json::from_str(&json)?;

        assert_eq!(restored.event_id, committed.event_id);
        assert_eq!(restored.thread_id, committed.thread_id);
        assert_eq!(restored.sequence, committed.sequence);
        assert_eq!(restored.timestamp, committed.timestamp);
        Ok(())
    }

    #[test]
    fn committed_event_wire_format_keys_are_stable() -> anyhow::Result<()> {
        let committed = sample_committed(0);
        let value = serde_json::to_value(&committed)?;

        for key in ["event_id", "thread_id", "sequence", "timestamp", "event"] {
            assert!(
                value.get(key).is_some(),
                "CommittedEvent wire format lost key `{key}` — this breaks the server contract"
            );
        }
        Ok(())
    }

    #[test]
    fn committed_event_nests_event_payload() -> anyhow::Result<()> {
        let committed = sample_committed(0);
        let value = serde_json::to_value(&committed)?;

        // The event is nested (not flattened) in the storage format.
        let event_obj = value
            .get("event")
            .expect("event key must exist")
            .as_object()
            .expect("event must be an object");
        assert_eq!(event_obj.get("type").and_then(|v| v.as_str()), Some("text"),);
        Ok(())
    }

    #[test]
    fn committed_event_uuid_is_v7() {
        let committed = sample_committed(0);
        assert_eq!(
            committed.event_id.get_version(),
            Some(uuid::Version::SortRand),
        );
    }
}
