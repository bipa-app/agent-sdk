//! Durable message projection schema and invariants.
//!
//! The `message_projection` is the single place where the committed
//! per-thread conversation history lives. The committed history is
//! updated **only** at turn-completion time, so a crash can never leave
//! the committed projection ahead of the latest completed checkpoint.
//!
//! # Draft history
//!
//! Alongside the committed history, the projection carries an optional
//! `draft_messages` list — the in-flight conversation accumulated since
//! the last committed turn. Drafts exist so a turn that fails
//! mid-stream (e.g. a transient LLM transport error after several
//! suspension/resume cycles) can still surface its accumulated work to
//! the next turn instead of vanishing entirely.
//!
//! Draft semantics:
//! - Written at every tool-boundary suspension via
//!   [`MessageProjection::set_draft`]: the suspension path snapshots
//!   its full `suspended_messages` list into the draft slot.
//! - Cleared on every successful turn commit via
//!   [`MessageProjection::clear_draft`]: the committed history now
//!   subsumes the draft, so the slot must be empty for the next
//!   turn.
//! - Survives task failure: `fail_root_turn` clears the failed task's
//!   transient `TaskState` payload, but the projection's draft is
//!   preserved on a separate row, so [`super::thread_recover`] can fold
//!   the in-flight messages into the recovery view.
//!
//! # Mutation paths
//!
//! | Transition | What it does | Guard |
//! |------------|--------------|-------|
//! | [`MessageProjection::append_committed`] | Extends committed history, bumps version | Non-empty input |
//! | [`MessageProjection::replace_history`] | Atomic swap of committed history, bumps version | — |
//! | [`MessageProjection::set_draft`] | Replace the in-flight draft, bumps version | — |
//! | [`MessageProjection::clear_draft`] | Drop the in-flight draft, bumps version | — |
//!
//! All transitions consume `self` and return a new `MessageProjection`,
//! following the same pure-transition pattern as
//! [`super::thread::Thread::apply_committed_turn`].
//!
//! # Version field
//!
//! Every mutation increments `version` by one. This lets future
//! database-backed implementations use optimistic concurrency control
//! (compare-and-swap on `version`) without changing the schema layer.
//!
//! # Wire format
//!
//! Messages are stored in insertion order as `Vec<llm::Message>`.
//! The outer projection row serializes with `snake_case` keys to
//! match every other journal type. `draft_messages` defaults to an
//! empty list so older serialized rows still deserialize cleanly.

use agent_sdk_foundation::{ThreadId, llm};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors for [`MessageProjection`] transitions.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MessageProjectionError {
    #[error("cannot commit an empty message batch")]
    EmptyCommit,
}

// ─────────────────────────────────────────────────────────────────────
// MessageProjection
// ─────────────────────────────────────────────────────────────────────

/// One row in the `message_projection` table.
///
/// Holds the committed message history for a single thread. The
/// history is modified exclusively through [`Self::append_committed`]
/// and [`Self::replace_history`], both of which are called by the
/// store's entry points under a write lock. No other code path may
/// modify the messages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageProjection {
    /// The thread this projection belongs to.
    pub thread_id: ThreadId,
    /// Ordered committed message history.
    pub messages: Vec<llm::Message>,
    /// In-flight conversation accumulated since the last committed
    /// turn.
    ///
    /// Populated by [`Self::set_draft`] at every tool-boundary
    /// suspension and cleared by [`Self::clear_draft`] at turn
    /// commit. Empty when no turn is in flight.
    ///
    /// `#[serde(default)]` keeps older serialized rows compatible —
    /// they decode with an empty draft slot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_messages: Vec<llm::Message>,
    /// Monotonically increasing version, bumped on every mutation.
    pub version: u64,
    /// When this projection row was first created.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Last time this row was modified.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

impl MessageProjection {
    /// Create a fresh, empty projection for the given thread.
    #[must_use]
    pub const fn new(thread_id: ThreadId, now: OffsetDateTime) -> Self {
        Self {
            thread_id,
            messages: Vec::new(),
            draft_messages: Vec::new(),
            version: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Append committed messages to the history.
    ///
    /// This is the append-only mutation path used at turn completion.
    /// Each call extends the existing history and bumps the version.
    ///
    /// # Errors
    /// Returns [`MessageProjectionError::EmptyCommit`] if `new_messages`
    /// is empty — callers should not commit zero messages.
    pub fn append_committed(
        mut self,
        new_messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<Self, MessageProjectionError> {
        if new_messages.is_empty() {
            return Err(MessageProjectionError::EmptyCommit);
        }
        self.messages.extend(new_messages);
        self.version += 1;
        self.updated_at = now;
        Ok(self)
    }

    /// Replace the entire message history atomically.
    ///
    /// Used for context compaction: the caller replaces all existing
    /// messages with a condensed summary. This is a full swap — the
    /// old history is discarded and the new one takes its place.
    ///
    /// Unlike [`Self::append_committed`], an empty replacement is
    /// allowed (clears the history).
    #[must_use]
    pub fn replace_history(mut self, messages: Vec<llm::Message>, now: OffsetDateTime) -> Self {
        self.messages = messages;
        self.version += 1;
        self.updated_at = now;
        self
    }

    /// Replace the in-flight draft messages.
    ///
    /// Called at every tool-boundary suspension with the full
    /// `suspended_messages` list captured at that point. The draft
    /// is overwritten (not appended) because each suspension carries
    /// the complete in-flight history through that boundary.
    ///
    /// An empty `messages` argument is allowed and produces the same
    /// observable state as [`Self::clear_draft`], but
    /// [`Self::clear_draft`] is the canonical entry point for the
    /// turn-commit path.
    #[must_use]
    pub fn set_draft(mut self, messages: Vec<llm::Message>, now: OffsetDateTime) -> Self {
        self.draft_messages = messages;
        self.version += 1;
        self.updated_at = now;
        self
    }

    /// Drop the in-flight draft messages.
    ///
    /// Called after a successful turn commit so the next turn starts
    /// with no stale draft. No-op when the draft is already empty
    /// (still bumps the version so writers can observe the
    /// transition).
    #[must_use]
    pub fn clear_draft(mut self, now: OffsetDateTime) -> Self {
        self.draft_messages = Vec::new();
        self.version += 1;
        self.updated_at = now;
        self
    }

    /// Number of messages in the committed history.
    #[must_use]
    pub const fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// `true` when an in-flight draft is currently held.
    #[must_use]
    pub const fn has_draft(&self) -> bool {
        !self.draft_messages.is_empty()
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

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_id() -> ThreadId {
        ThreadId::from_string("t-msg-test")
    }

    // ── construction ──────────────────────────────────────────────

    #[test]
    fn new_projection_is_empty_with_version_zero() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.thread_id, thread_id());
        assert!(p.messages.is_empty());
        assert_eq!(p.version, 0);
        assert_eq!(p.message_count(), 0);
    }

    // ── append_committed ──────────────────────────────────────────

    #[test]
    fn append_committed_extends_history() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(
                vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
                t_plus(1),
            )
            .expect("non-empty commit succeeds");
        assert_eq!(p.message_count(), 2);
        assert_eq!(p.version, 1);
        assert_eq!(p.updated_at, t_plus(1));
    }

    #[test]
    fn append_committed_accumulates_across_turns() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("turn 1")], t_plus(1))
            .unwrap();
        let p = p
            .append_committed(vec![llm::Message::assistant("reply 1")], t_plus(2))
            .unwrap();
        let p = p
            .append_committed(
                vec![
                    llm::Message::user("turn 2"),
                    llm::Message::assistant("reply 2"),
                ],
                t_plus(3),
            )
            .unwrap();
        assert_eq!(p.message_count(), 4);
        assert_eq!(p.version, 3);
    }

    #[test]
    fn append_committed_rejects_empty_batch() {
        let p = MessageProjection::new(thread_id(), t0());
        let err = p.append_committed(vec![], t_plus(1)).unwrap_err();
        assert_eq!(err, MessageProjectionError::EmptyCommit);
    }

    #[test]
    fn append_committed_preserves_order() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("first")], t_plus(1))
            .unwrap();
        let p = p
            .append_committed(vec![llm::Message::user("second")], t_plus(2))
            .unwrap();
        assert_eq!(p.messages[0].role, llm::Role::User);
        assert_eq!(p.messages[1].role, llm::Role::User);
        // Verify actual content ordering
        let texts: Vec<_> = p
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                llm::Content::Text(t) => Some(t.as_str()),
                llm::Content::Blocks(_) => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second"]);
    }

    // ── replace_history ───────────────────────────────────────────

    #[test]
    fn replace_history_swaps_entire_history() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(
                vec![
                    llm::Message::user("old 1"),
                    llm::Message::assistant("old 2"),
                    llm::Message::user("old 3"),
                ],
                t_plus(1),
            )
            .unwrap();
        assert_eq!(p.message_count(), 3);

        let p = p.replace_history(
            vec![
                llm::Message::user("[Summary]"),
                llm::Message::assistant("Continuing..."),
            ],
            t_plus(2),
        );
        assert_eq!(p.message_count(), 2);
        assert_eq!(p.version, 2); // append bumped to 1, replace bumps to 2
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn replace_history_allows_empty_replacement() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("data")], t_plus(1))
            .unwrap();
        let p = p.replace_history(vec![], t_plus(2));
        assert_eq!(p.message_count(), 0);
        assert_eq!(p.version, 2);
    }

    #[test]
    fn replace_history_bumps_version() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.version, 0);
        let p = p.replace_history(vec![llm::Message::user("fresh")], t_plus(1));
        assert_eq!(p.version, 1);
        let p = p.replace_history(vec![llm::Message::user("newer")], t_plus(2));
        assert_eq!(p.version, 2);
    }

    // ── version monotonicity ──────────────────────────────────────

    #[test]
    fn version_increases_monotonically_across_mixed_operations() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.version, 0);

        let p = p
            .append_committed(vec![llm::Message::user("a")], t_plus(1))
            .unwrap();
        assert_eq!(p.version, 1);

        let p = p.replace_history(vec![llm::Message::user("b")], t_plus(2));
        assert_eq!(p.version, 2);

        let p = p
            .append_committed(vec![llm::Message::user("c")], t_plus(3))
            .unwrap();
        assert_eq!(p.version, 3);
    }

    // ── identity ──────────────────────────────────────────────────

    #[test]
    fn thread_id_and_created_at_survive_mutations() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("a")], t_plus(1))
            .unwrap();
        let p = p.replace_history(vec![], t_plus(2));
        assert_eq!(p.thread_id, thread_id());
        assert_eq!(p.created_at, t0());
    }

    // ── set_draft / clear_draft ───────────────────────────────────

    #[test]
    fn new_projection_has_no_draft() {
        let p = MessageProjection::new(thread_id(), t0());
        assert!(!p.has_draft());
        assert!(p.draft_messages.is_empty());
    }

    #[test]
    fn set_draft_stores_in_flight_messages_without_touching_committed() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("committed")], t_plus(1))
            .unwrap();
        assert_eq!(p.message_count(), 1);
        assert_eq!(p.version, 1);

        let p = p.set_draft(
            vec![
                llm::Message::user("turn 2 prompt"),
                llm::Message::assistant("calling tool"),
            ],
            t_plus(2),
        );
        // Committed history is untouched.
        assert_eq!(p.message_count(), 1);
        // Draft now carries the in-flight messages.
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 2);
        assert_eq!(p.version, 2);
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn set_draft_overwrites_prior_draft() {
        // Each suspension passes the full accumulated suspended_messages,
        // so set_draft must replace the slot rather than append.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("first suspension")], t_plus(1));
        let p = p.set_draft(
            vec![
                llm::Message::user("first suspension"),
                llm::Message::assistant("second tool boundary"),
            ],
            t_plus(2),
        );
        assert_eq!(p.draft_messages.len(), 2);
        assert_eq!(p.version, 2);
    }

    #[test]
    fn clear_draft_drops_in_flight_messages() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("in flight")], t_plus(1));
        assert!(p.has_draft());
        let p = p.clear_draft(t_plus(2));
        assert!(!p.has_draft());
        assert_eq!(p.version, 2);
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn clear_draft_on_empty_slot_still_bumps_version() {
        // The store path may call clear_draft unconditionally on every
        // commit — calling it with an already-empty slot must remain
        // observable so optimistic-concurrency writers don't miss the
        // transition.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.clear_draft(t_plus(1));
        assert!(!p.has_draft());
        assert_eq!(p.version, 1);
    }

    #[test]
    fn draft_survives_commit_history_extensions_until_cleared() {
        // Sanity: append_committed and replace_history operate on the
        // committed slot only. The draft must persist verbatim until
        // an explicit clear_draft / set_draft.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("draft")], t_plus(1));

        let p = p
            .append_committed(vec![llm::Message::user("committed-1")], t_plus(2))
            .unwrap();
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 1);

        let p = p.replace_history(vec![llm::Message::user("[summary]")], t_plus(3));
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 1);
    }

    #[test]
    fn draft_round_trips_through_json() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(
            vec![
                llm::Message::user("draft prompt"),
                llm::Message::assistant("draft asst"),
            ],
            t_plus(1),
        );
        let json = serde_json::to_string(&p)?;
        let recovered: MessageProjection = serde_json::from_str(&json)?;
        assert!(recovered.has_draft());
        assert_eq!(recovered.draft_messages.len(), 2);
        assert_eq!(recovered.version, p.version);
        Ok(())
    }

    #[test]
    fn legacy_json_without_draft_field_decodes_empty() -> anyhow::Result<()> {
        // Older daemons serialized rows before draft_messages existed.
        // The field is `#[serde(default)]` so legacy payloads must
        // still round-trip with an empty draft slot.
        let legacy = serde_json::json!({
            "thread_id": "t-msg-test",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "version": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        });
        let recovered: MessageProjection = serde_json::from_value(legacy)?;
        assert!(!recovered.has_draft());
        assert_eq!(recovered.messages.len(), 1);
        Ok(())
    }

    // ── wire format ───────────────────────────────────────────────

    #[test]
    fn projection_round_trips_through_json() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.append_committed(
            vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
            t_plus(1),
        )?;
        let json = serde_json::to_string(&p)?;
        let recovered: MessageProjection = serde_json::from_str(&json)?;
        assert_eq!(recovered.thread_id, p.thread_id);
        assert_eq!(recovered.version, p.version);
        assert_eq!(recovered.messages.len(), p.messages.len());
        Ok(())
    }

    #[test]
    fn wire_format_keys_are_stable() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("test")], t_plus(1))
            .unwrap();
        let value = serde_json::to_value(&p)?;
        for key in [
            "thread_id",
            "messages",
            "version",
            "created_at",
            "updated_at",
        ] {
            assert!(
                value.get(key).is_some(),
                "MessageProjection wire format lost key `{key}` — this breaks the server contract"
            );
        }
        Ok(())
    }
}
