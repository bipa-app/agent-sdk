//! Thread projection schema and invariants.
//!
//! The `threads` projection is the single place where committed
//! conversation-level counters and status live. Aggregates
//! (`committed_turns`, `total_usage`) are updated **only** through
//! the [`Thread::apply_committed_turn`] transition, which is called
//! by [`super::thread_store::ThreadStore::commit_turn`]. There is no
//! generic `update()` path that touches counters — this prevents
//! split ownership of thread-level aggregates.
//!
//! # Ownership rule
//!
//! Thread aggregates are written by the completed-turn commit path
//! and nowhere else. A worker that completes a turn drives a single
//! `commit_turn` call on the store, which:
//!
//! 1. Loads or creates the [`Thread`] row.
//! 2. Calls [`Thread::apply_committed_turn`] with the turn's
//!    [`TokenUsage`] delta.
//! 3. Persists the updated row atomically.
//!
//! No other code path may modify `committed_turns` or
//! `total_usage`. This is the guarantee later recovery and billing
//! phases will rely on.
//!
//! # Wire format
//!
//! [`ThreadStatus`] uses `#[serde(rename_all = "snake_case")]` so
//! durable rows serialize as `"active"` / `"completed"`, matching
//! the `snake_case` convention every other journal enum follows.

use agent_sdk_core::{ThreadId, TokenUsage};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

// ─────────────────────────────────────────────────────────────────────
// ThreadStatus
// ─────────────────────────────────────────────────────────────────────

/// Lifecycle status of a thread projection row.
///
/// A thread starts [`ThreadStatus::Active`] on first creation and
/// stays active while turns are being committed. A caller may
/// transition it to [`ThreadStatus::Completed`] once the conversation
/// is done, after which no further turns may be committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatus {
    /// The thread is open for new turns.
    Active,
    /// The thread is closed — no further turns may be committed.
    Completed,
}

impl ThreadStatus {
    /// `true` if the thread is still open for turn commits.
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// `true` if the thread is closed.
    #[must_use]
    pub const fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors for [`Thread`] rows.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ThreadSchemaError {
    #[error("cannot commit a turn on a completed thread")]
    CommitOnCompletedThread,

    #[error("thread is already completed")]
    AlreadyCompleted,

    #[error("committed_turns must be > 0 for a completed thread")]
    CompletedWithZeroTurns,
}

// ─────────────────────────────────────────────────────────────────────
// Thread
// ─────────────────────────────────────────────────────────────────────

/// One row in the `threads` projection.
///
/// A `Thread` holds the durable aggregate state for a single
/// conversation. The counters (`committed_turns`, `total_usage`) are
/// modified exclusively by [`Thread::apply_committed_turn`], which
/// is the pure transition the store's `commit_turn` entry point
/// calls. No other code path touches the aggregates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thread {
    /// The thread's identity.
    pub thread_id: ThreadId,
    /// Lifecycle status.
    pub status: ThreadStatus,
    /// Number of turns that have been committed to this thread.
    /// Incremented exclusively by [`Self::apply_committed_turn`].
    pub committed_turns: u32,
    /// Cumulative token usage across every committed turn.
    /// Accumulated exclusively by [`Self::apply_committed_turn`].
    pub total_usage: TokenUsage,
    /// When this thread row was first created.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Last time this row was modified.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

impl Thread {
    /// Create a fresh thread row for the given id.
    ///
    /// The thread starts [`ThreadStatus::Active`] with zero committed
    /// turns and zero usage. Used by the store's `get_or_create` path.
    #[must_use]
    pub fn new(thread_id: ThreadId, now: OffsetDateTime) -> Self {
        Self {
            thread_id,
            status: ThreadStatus::Active,
            committed_turns: 0,
            total_usage: TokenUsage::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Apply a committed turn's token usage to this thread's
    /// aggregates.
    ///
    /// This is the **only** mutation path for `committed_turns` and
    /// `total_usage`. The store's `commit_turn` entry point calls
    /// this under its write lock so aggregates are always consistent
    /// with the set of committed turns.
    ///
    /// # Errors
    /// Returns [`ThreadSchemaError::CommitOnCompletedThread`] if the
    /// thread is already completed.
    pub fn apply_committed_turn(
        mut self,
        turn_usage: &TokenUsage,
        now: OffsetDateTime,
    ) -> Result<Self, ThreadSchemaError> {
        if self.status.is_completed() {
            return Err(ThreadSchemaError::CommitOnCompletedThread);
        }
        self.committed_turns = self.committed_turns.saturating_add(1);
        self.total_usage.add(turn_usage);
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Transition the thread to [`ThreadStatus::Completed`].
    ///
    /// A completed thread rejects further [`Self::apply_committed_turn`]
    /// calls. The caller is responsible for ensuring the conversation
    /// is actually done (the schema layer does not reason about task
    /// lifecycle).
    ///
    /// # Errors
    /// - [`ThreadSchemaError::AlreadyCompleted`] if the thread is
    ///   already completed.
    /// - [`ThreadSchemaError::CompletedWithZeroTurns`] if no turns
    ///   have been committed yet.
    pub fn mark_completed(mut self, now: OffsetDateTime) -> Result<Self, ThreadSchemaError> {
        if self.status.is_completed() {
            return Err(ThreadSchemaError::AlreadyCompleted);
        }
        if self.committed_turns == 0 {
            return Err(ThreadSchemaError::CompletedWithZeroTurns);
        }
        self.status = ThreadStatus::Completed;
        self.updated_at = now;
        self.validate()?;
        Ok(self)
    }

    /// Check every structural invariant on this row.
    ///
    /// # Errors
    /// Returns the first violated invariant.
    pub const fn validate(&self) -> Result<(), ThreadSchemaError> {
        if self.status.is_completed() && self.committed_turns == 0 {
            return Err(ThreadSchemaError::CompletedWithZeroTurns);
        }
        Ok(())
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
        ThreadId::from_string("t-thread-test")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
        }
    }

    // ── construction ──────────────────────────────────────────────

    #[test]
    fn new_thread_is_active_with_zero_counters() {
        let thread = Thread::new(thread_id(), t0());
        assert_eq!(thread.status, ThreadStatus::Active);
        assert_eq!(thread.committed_turns, 0);
        assert_eq!(thread.total_usage, TokenUsage::default());
        assert!(thread.validate().is_ok());
    }

    // ── apply_committed_turn ──────────────────────────────────────

    #[test]
    fn apply_committed_turn_increments_counters() {
        let thread = Thread::new(thread_id(), t0());
        let updated = thread
            .apply_committed_turn(&usage(100, 50), t_plus(1))
            .expect("commit succeeds");
        assert_eq!(updated.committed_turns, 1);
        assert_eq!(updated.total_usage, usage(100, 50));
        assert_eq!(updated.updated_at, t_plus(1));
    }

    #[test]
    fn apply_committed_turn_accumulates_across_multiple_turns() {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(100, 50), t_plus(1)).unwrap();
        let t = t.apply_committed_turn(&usage(200, 80), t_plus(2)).unwrap();
        let t = t.apply_committed_turn(&usage(50, 20), t_plus(3)).unwrap();
        assert_eq!(t.committed_turns, 3);
        assert_eq!(t.total_usage, usage(350, 150));
    }

    #[test]
    fn apply_committed_turn_rejects_completed_thread() {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(10, 5), t_plus(1)).unwrap();
        let t = t.mark_completed(t_plus(2)).unwrap();
        let err = t
            .apply_committed_turn(&usage(10, 5), t_plus(3))
            .unwrap_err();
        assert_eq!(err, ThreadSchemaError::CommitOnCompletedThread);
    }

    // ── mark_completed ────────────────────────────────────────────

    #[test]
    fn mark_completed_transitions_active_thread() {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(10, 5), t_plus(1)).unwrap();
        let t = t.mark_completed(t_plus(2)).unwrap();
        assert_eq!(t.status, ThreadStatus::Completed);
        assert_eq!(t.updated_at, t_plus(2));
    }

    #[test]
    fn mark_completed_rejects_already_completed() {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(10, 5), t_plus(1)).unwrap();
        let t = t.mark_completed(t_plus(2)).unwrap();
        let err = t.mark_completed(t_plus(3)).unwrap_err();
        assert_eq!(err, ThreadSchemaError::AlreadyCompleted);
    }

    #[test]
    fn mark_completed_rejects_zero_turns() {
        let t = Thread::new(thread_id(), t0());
        let err = t.mark_completed(t_plus(1)).unwrap_err();
        assert_eq!(err, ThreadSchemaError::CompletedWithZeroTurns);
    }

    // ── validate ──────────────────────────────────────────────────

    #[test]
    fn validate_rejects_completed_with_zero_turns() {
        let mut t = Thread::new(thread_id(), t0());
        t.status = ThreadStatus::Completed;
        assert_eq!(t.validate(), Err(ThreadSchemaError::CompletedWithZeroTurns));
    }

    // ── wire format ───────────────────────────────────────────────

    #[test]
    fn thread_round_trips_through_json() -> anyhow::Result<()> {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(100, 50), t_plus(1))?;
        let json = serde_json::to_string(&t)?;
        let recovered: Thread = serde_json::from_str(&json)?;
        assert_eq!(recovered, t);
        Ok(())
    }

    #[test]
    fn thread_status_wire_format_is_snake_case() -> anyhow::Result<()> {
        assert_eq!(serde_json::to_string(&ThreadStatus::Active)?, "\"active\"");
        assert_eq!(
            serde_json::to_string(&ThreadStatus::Completed)?,
            "\"completed\""
        );
        // Round-trip
        let recovered: ThreadStatus = serde_json::from_str("\"active\"")?;
        assert_eq!(recovered, ThreadStatus::Active);
        let recovered: ThreadStatus = serde_json::from_str("\"completed\"")?;
        assert_eq!(recovered, ThreadStatus::Completed);
        Ok(())
    }

    #[test]
    fn thread_wire_format_keys_are_stable() -> anyhow::Result<()> {
        let t = Thread::new(thread_id(), t0());
        let t = t.apply_committed_turn(&usage(100, 50), t_plus(1))?;
        let value = serde_json::to_value(&t)?;
        for key in [
            "thread_id",
            "status",
            "committed_turns",
            "total_usage",
            "created_at",
            "updated_at",
        ] {
            assert!(
                value.get(key).is_some(),
                "Thread wire format lost key `{key}` — this breaks the server contract"
            );
        }
        assert_eq!(value["status"], serde_json::json!("active"));
        Ok(())
    }

    // ── ThreadStatus classification ───────────────────────────────

    #[test]
    fn thread_status_classification_is_stable() {
        assert!(ThreadStatus::Active.is_active());
        assert!(!ThreadStatus::Active.is_completed());
        assert!(!ThreadStatus::Completed.is_active());
        assert!(ThreadStatus::Completed.is_completed());
    }
}
