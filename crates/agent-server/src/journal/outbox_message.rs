//! Logical outbox message kinds and their advisory payload shapes.
//!
//! Phase 8.1 of the rewrite locks the contract for what may be written
//! into `agent_sdk_outbox`.  The outbox is the **only** path through
//! which the server hands work to a broker — there is no direct publish
//! from worker code.  This module defines what those handoffs are
//! allowed to mean.
//!
//! # Two logical kinds
//!
//! ```text
//!   ┌────────────────────────────┐  durable trigger
//!   │ task_wakeup                │◀─ task journal mutation that makes
//!   │   { task_id, thread_id }   │   a task runnable (admit, promote,
//!   └────────────────────────────┘   release after suspension)
//!
//!   ┌────────────────────────────┐  durable trigger
//!   │ thread_events_available    │◀─ committed-event batch landing on a
//!   │   { thread_id,             │   thread (one row per *batch*, not
//!   │     last_sequence }        │   per event)
//!   └────────────────────────────┘
//! ```
//!
//! # Payloads are advisory, not authoritative
//!
//! A queue payload carries only the durable references a consumer needs
//! to look the canonical record up itself.  It carries no event body,
//! no message content, no auth context, no rendered text.  The contract
//! is:
//!
//! 1. Treat every advisory as a hint, not as data.
//! 2. Resolve `task_id` against
//!    [`agent_sdk_tasks`](super::store::AgentTaskStore) before acting.
//! 3. Resolve `(thread_id, sequence)` against
//!    [`agent_sdk_committed_events`](super::event_repository::EventRepository)
//!    before acting.
//! 4. Treat a missing or out-of-date row as a benign duplicate; the
//!    relay is at-least-once and the broker may republish.
//!
//! Callers that violate any of these will eventually corrupt downstream
//! state when the broker drops, replays, reorders, or coalesces a
//! message — all of which are permitted by the contract.
//!
//! # Wire format
//!
//! Each payload serialises to a JSON object whose top-level keys are
//! the fields of the matching `*Payload` struct.  The `kind` itself is
//! stored as a separate column on the outbox row so the relay worker
//! can route without parsing the payload first.

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::task::AgentTaskId;

// ─────────────────────────────────────────────────────────────────────
// Kind discriminator
// ─────────────────────────────────────────────────────────────────────

/// Logical kind of an outbox message.
///
/// Phase 8.1 limits the contract to exactly two kinds.  Adding a new
/// kind requires updating:
///
/// - this enum,
/// - the [`OutboxMessage`] variant set,
/// - the `agent_sdk_outbox_kind_check` SQL constraint on every backend,
/// - the relay worker's routing table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboxMessageKind {
    /// A task became runnable; consumers should poke their worker
    /// scheduler so it picks up the new work without polling.
    TaskWakeup,
    /// A new committed-event batch landed for a thread; consumers
    /// should advance their replay cursor to at least `last_sequence`.
    ThreadEventsAvailable,
}

impl OutboxMessageKind {
    /// Wire-format string for durable persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TaskWakeup => "task_wakeup",
            Self::ThreadEventsAvailable => "thread_events_available",
        }
    }

    /// Parse a wire-format string back into the enum.
    ///
    /// We deliberately do NOT implement [`std::str::FromStr`] here:
    /// callers always have the kind from a `kind` column or a broker
    /// header, never from arbitrary user input, and a freestanding
    /// inherent method keeps the call sites concrete and obvious.
    ///
    /// # Errors
    /// Returns an error if `s` is not one of the canonical strings
    /// returned by [`Self::as_str`].
    pub fn parse_wire(s: &str) -> Result<Self> {
        match s {
            "task_wakeup" => Ok(Self::TaskWakeup),
            "thread_events_available" => Ok(Self::ThreadEventsAvailable),
            other => anyhow::bail!("unknown outbox message kind: {other}"),
        }
    }
}

impl std::fmt::Display for OutboxMessageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Advisory payloads
// ─────────────────────────────────────────────────────────────────────

/// Advisory payload for a [`OutboxMessageKind::TaskWakeup`].
///
/// Carries only the references the consumer needs to look the task up
/// in [`agent_sdk_tasks`](super::store::AgentTaskStore).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskWakeupPayload {
    /// Task whose state changed in a way that warrants a wake.
    pub task_id: AgentTaskId,
    /// Thread the task belongs to (carried for cheap routing).
    pub thread_id: ThreadId,
}

/// Advisory payload for a [`OutboxMessageKind::ThreadEventsAvailable`].
///
/// Carries the thread reference and the highest committed sequence the
/// consumer is guaranteed to be able to read from
/// [`agent_sdk_committed_events`](super::event_repository::EventRepository)
/// after the triggering transaction.  The consumer can replay any
/// suffix from its own cursor up to (and beyond) `last_sequence`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadEventsAvailablePayload {
    /// Thread that received the event batch.
    pub thread_id: ThreadId,
    /// Highest sequence committed in the triggering transaction.
    pub last_sequence: u64,
}

// ─────────────────────────────────────────────────────────────────────
// Typed envelope
// ─────────────────────────────────────────────────────────────────────

/// Typed advisory message that travels through the relay.
///
/// The on-the-wire representation is the bare payload object — the
/// `kind` is carried alongside the payload (in the outbox `kind` column
/// or the broker message header) so the relay layer never has to peek
/// at the JSON to decide how to dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutboxMessage {
    /// Task became runnable.
    TaskWakeup(TaskWakeupPayload),
    /// Thread received new committed events.
    ThreadEventsAvailable(ThreadEventsAvailablePayload),
}

impl OutboxMessage {
    /// Logical kind of this message.
    #[must_use]
    pub const fn kind(&self) -> OutboxMessageKind {
        match self {
            Self::TaskWakeup(_) => OutboxMessageKind::TaskWakeup,
            Self::ThreadEventsAvailable(_) => OutboxMessageKind::ThreadEventsAvailable,
        }
    }

    /// Serialise the payload to a JSON value suitable for the
    /// `payload_json` column of `agent_sdk_outbox`.
    ///
    /// # Errors
    /// Returns an error if serialisation fails (it should not under
    /// normal conditions; the payload structs are pure data).
    pub fn to_payload_json(&self) -> Result<serde_json::Value> {
        match self {
            Self::TaskWakeup(p) => serde_json::to_value(p).context("serialise task_wakeup payload"),
            Self::ThreadEventsAvailable(p) => {
                serde_json::to_value(p).context("serialise thread_events_available payload")
            }
        }
    }

    /// Reconstruct a message from a kind tag and a serialised payload.
    ///
    /// # Errors
    /// Returns an error if `payload` does not deserialise as the shape
    /// expected for `kind`.
    pub fn from_payload_json(kind: OutboxMessageKind, payload: serde_json::Value) -> Result<Self> {
        match kind {
            OutboxMessageKind::TaskWakeup => {
                let payload = serde_json::from_value::<TaskWakeupPayload>(payload)
                    .context("deserialise task_wakeup payload")?;
                Ok(Self::TaskWakeup(payload))
            }
            OutboxMessageKind::ThreadEventsAvailable => {
                let payload = serde_json::from_value::<ThreadEventsAvailablePayload>(payload)
                    .context("deserialise thread_events_available payload")?;
                Ok(Self::ThreadEventsAvailable(payload))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-outbox-msg-a")
    }

    fn task_a() -> AgentTaskId {
        AgentTaskId::from_string("task_outbox_msg_a")
    }

    #[test]
    fn kind_round_trips_through_string() -> Result<()> {
        for kind in [
            OutboxMessageKind::TaskWakeup,
            OutboxMessageKind::ThreadEventsAvailable,
        ] {
            assert_eq!(OutboxMessageKind::parse_wire(kind.as_str())?, kind);
            assert_eq!(format!("{kind}"), kind.as_str());
        }
        Ok(())
    }

    #[test]
    fn kind_rejects_unknown_strings() {
        assert!(OutboxMessageKind::parse_wire("nope").is_err());
        // Empty string must not silently parse to a default variant.
        assert!(OutboxMessageKind::parse_wire("").is_err());
    }

    #[test]
    fn task_wakeup_payload_round_trips() -> Result<()> {
        let message = OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id: task_a(),
            thread_id: thread_a(),
        });

        let json = message.to_payload_json()?;
        let object = json.as_object().context("expected JSON object")?;
        assert_eq!(
            object.get("task_id").and_then(serde_json::Value::as_str),
            Some("task_outbox_msg_a"),
        );
        assert_eq!(
            object.get("thread_id").and_then(serde_json::Value::as_str),
            Some("t-outbox-msg-a"),
        );
        // Payload must NOT contain the kind tag — the kind lives in the
        // outbox column / broker header.
        assert!(object.get("kind").is_none());

        let restored = OutboxMessage::from_payload_json(OutboxMessageKind::TaskWakeup, json)?;
        assert_eq!(restored, message);
        assert_eq!(restored.kind(), OutboxMessageKind::TaskWakeup);
        Ok(())
    }

    #[test]
    fn thread_events_available_payload_round_trips() -> Result<()> {
        let message = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 42,
        });

        let json = message.to_payload_json()?;
        let object = json.as_object().context("expected JSON object")?;
        assert_eq!(
            object.get("thread_id").and_then(serde_json::Value::as_str),
            Some("t-outbox-msg-a"),
        );
        assert_eq!(
            object
                .get("last_sequence")
                .and_then(serde_json::Value::as_u64),
            Some(42),
        );

        let restored =
            OutboxMessage::from_payload_json(OutboxMessageKind::ThreadEventsAvailable, json)?;
        assert_eq!(restored, message);
        assert_eq!(restored.kind(), OutboxMessageKind::ThreadEventsAvailable);
        Ok(())
    }

    #[test]
    fn from_payload_json_rejects_mismatched_shape() {
        // A task_wakeup payload deserialised under the
        // thread_events_available kind must fail loudly.
        let task_payload = serde_json::json!({
            "task_id": "task_x",
            "thread_id": "t-x",
        });
        let result = OutboxMessage::from_payload_json(
            OutboxMessageKind::ThreadEventsAvailable,
            task_payload,
        );
        assert!(result.is_err(), "mismatched payload should not deserialise");
    }

    #[test]
    fn kind_is_stable_snake_case() {
        // The wire format is part of the durable contract — these
        // strings are persisted in `agent_sdk_outbox.kind` and embedded
        // in broker headers.  Changing them is a breaking change.
        assert_eq!(OutboxMessageKind::TaskWakeup.as_str(), "task_wakeup");
        assert_eq!(
            OutboxMessageKind::ThreadEventsAvailable.as_str(),
            "thread_events_available"
        );
    }
}
