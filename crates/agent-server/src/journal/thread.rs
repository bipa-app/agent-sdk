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
//! durable rows serialize as `"active"` / `"completed"` / `"deleting"` /
//! `"deleted"`, matching the `snake_case` convention every other journal enum
//! follows.

use agent_sdk_foundation::{ThreadId, TokenUsage};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

// ─────────────────────────────────────────────────────────────────────
// ThreadStatus
// ─────────────────────────────────────────────────────────────────────

/// Lifecycle status of a thread projection row.
///
/// A thread starts [`ThreadStatus::Active`] on first creation. Purge first
/// installs a durable [`ThreadStatus::Deleting`] fence and only then advances
/// it to [`ThreadStatus::Deleted`]. Both fence states are permanent admission
/// barriers; the row remains as a tombstone so no bootstrap path can recreate
/// the thread. [`ThreadStatus::Completed`] is retained for backwards
/// compatibility with the existing close-with-history contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatus {
    /// The thread is open for new turns.
    Active,
    /// The thread is closed — no further turns may be committed.
    Completed,
    /// Purge has started and new work is durably fenced out.
    Deleting,
    /// Purge completed; this tombstone is permanent.
    Deleted,
}

impl std::fmt::Display for ThreadStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Deleting => "deleting",
            Self::Deleted => "deleted",
        };
        f.write_str(status)
    }
}

impl ThreadStatus {
    /// `true` if the thread is still open for turn commits and task admission.
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// `true` if the thread is closed through the legacy completion path.
    #[must_use]
    pub const fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    /// `true` once the durable deletion fence has been installed.
    #[must_use]
    pub const fn is_purge_fenced(self) -> bool {
        matches!(self, Self::Deleting | Self::Deleted)
    }
}

/// Store operation guarded by the thread lifecycle fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadOperation {
    Create,
    Fork,
    Submit,
    Spawn,
    Promote,
    Acquire,
    Commit,
}

impl std::fmt::Display for ThreadOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operation = match self {
            Self::Create => "create",
            Self::Fork => "fork",
            Self::Submit => "submit",
            Self::Spawn => "spawn",
            Self::Promote => "promote",
            Self::Acquire => "acquire",
            Self::Commit => "commit",
        };
        f.write_str(operation)
    }
}

/// Typed rejection returned by every admission surface for a non-active
/// thread.
///
/// Callers can downcast `anyhow::Error` to this type and inspect the exact
/// lifecycle state rather than parsing an error string.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("thread {thread_id} is {status}; {operation} is fenced")]
pub struct ThreadStateConflict {
    pub thread_id: ThreadId,
    pub status: ThreadStatus,
    pub operation: ThreadOperation,
}

/// Purge reach: only the named thread, or its durable subagent invocation
/// descendants as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PurgeScope {
    Thread,
    InvocationTree,
}

/// Durable, idempotent proof of one completed purge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PurgeSeed {
    pub root_thread_id: ThreadId,
    pub scope: PurgeScope,
    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
}

/// Everything a completed purge recorded, retained on the tombstone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PurgeReceipt {
    pub root_thread_id: ThreadId,
    pub scope: PurgeScope,
    pub purged_thread_ids: Vec<ThreadId>,
    pub cancelled_task_ids: Vec<String>,
    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub completed_at: OffsetDateTime,
}

/// The one durable purge column, in either lifecycle shape.
///
/// Serialized untagged: a completed receipt is a strict superset of a
/// seed, so decode tries [`PurgeRecord::Receipt`] first and a seed can
/// never masquerade as one. Pre-seed tombstones decode unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PurgeRecord {
    Receipt(PurgeReceipt),
    Seed(PurgeSeed),
}

/// Typed purge failure for an id that has no live row or tombstone.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("thread {thread_id} does not exist")]
pub struct ThreadNotFound {
    pub thread_id: ThreadId,
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

    #[error("a deleting thread must carry its purge seed")]
    DeletingWithoutSeed,

    #[error("a deleted thread must carry its purge receipt")]
    DeletedWithoutReceipt,

    #[error("purge evidence is only valid on deleting/deleted rows, one kind each")]
    StrayPurgeRecord,
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
    /// Completed purge receipt. Present only on the permanent tombstone and
    /// retained so retries return byte-for-byte equivalent evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purge_receipt: Option<PurgeReceipt>,
    /// Fence-time purge identity, persisted with the `Deleting` fence so a
    /// crash-retry resumes with the same `started_at` and scope. Cleared
    /// when [`Self::finish_purge`] stores the completed receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purge_seed: Option<PurgeSeed>,
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
            purge_receipt: None,
            purge_seed: None,
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

    /// Install the durable deletion fence and persist its identity.
    ///
    /// Active and legacy-completed rows enter deletion carrying `seed` — the
    /// purge's durable identity (root, scope, `started_at`). A deleting row
    /// is returned unchanged, keeping its ORIGINAL seed, so an interrupted
    /// purge resumes under the first attempt's identity. A deleted row must
    /// be handled as an idempotent receipt replay by the store.
    #[must_use]
    pub fn begin_purge(mut self, seed: PurgeSeed) -> Self {
        match self.status {
            ThreadStatus::Active | ThreadStatus::Completed => {
                self.status = ThreadStatus::Deleting;
                self.updated_at = seed.started_at;
                self.purge_seed = Some(seed);
            }
            ThreadStatus::Deleting | ThreadStatus::Deleted => {}
        }
        self
    }

    /// Advance a fenced row to its permanent tombstone and retain the receipt.
    #[must_use]
    pub fn finish_purge(mut self, receipt: PurgeReceipt) -> Self {
        self.status = ThreadStatus::Deleted;
        self.updated_at = receipt.completed_at;
        self.purge_receipt = Some(receipt);
        self.purge_seed = None;
        self
    }

    /// The durable purge column's current content, whichever shape the
    /// lifecycle is in. Stores persist exactly this value.
    #[must_use]
    pub fn purge_record(&self) -> Option<PurgeRecord> {
        match (&self.purge_receipt, &self.purge_seed) {
            (Some(receipt), _) => Some(PurgeRecord::Receipt(receipt.clone())),
            (None, Some(seed)) => Some(PurgeRecord::Seed(seed.clone())),
            (None, None) => None,
        }
    }

    /// Split a decoded purge column back into the two domain fields.
    #[must_use]
    pub fn split_purge_record(
        record: Option<PurgeRecord>,
    ) -> (Option<PurgeReceipt>, Option<PurgeSeed>) {
        match record {
            Some(PurgeRecord::Receipt(receipt)) => (Some(receipt), None),
            Some(PurgeRecord::Seed(seed)) => (None, Some(seed)),
            None => (None, None),
        }
    }

    /// Return a typed error unless this row accepts `operation`.
    ///
    /// Legacy-completed rows remain readable through create replay/recovery and
    /// may still be forked, preserving the pre-purge contract. All other
    /// admission requires an active row. Deleting and deleted rows reject every
    /// operation.
    ///
    /// # Errors
    /// Returns [`ThreadStateConflict`] with the exact durable state when the
    /// operation is fenced.
    pub fn require_active(&self, operation: ThreadOperation) -> Result<(), ThreadStateConflict> {
        let allowed = self.status.is_active()
            || (self.status.is_completed()
                && matches!(operation, ThreadOperation::Create | ThreadOperation::Fork));
        if allowed {
            return Ok(());
        }
        Err(ThreadStateConflict {
            thread_id: self.thread_id.clone(),
            status: self.status,
            operation,
        })
    }

    /// Check every structural invariant on this row.
    ///
    /// # Errors
    /// Returns the first violated invariant.
    pub const fn validate(&self) -> Result<(), ThreadSchemaError> {
        if self.status.is_completed() && self.committed_turns == 0 {
            return Err(ThreadSchemaError::CompletedWithZeroTurns);
        }
        match self.status {
            ThreadStatus::Deleting => {
                if self.purge_seed.is_none() {
                    return Err(ThreadSchemaError::DeletingWithoutSeed);
                }
                if self.purge_receipt.is_some() {
                    return Err(ThreadSchemaError::StrayPurgeRecord);
                }
            }
            ThreadStatus::Deleted => {
                if self.purge_receipt.is_none() {
                    return Err(ThreadSchemaError::DeletedWithoutReceipt);
                }
                if self.purge_seed.is_some() {
                    return Err(ThreadSchemaError::StrayPurgeRecord);
                }
            }
            ThreadStatus::Active | ThreadStatus::Completed => {
                if self.purge_receipt.is_some() || self.purge_seed.is_some() {
                    return Err(ThreadSchemaError::StrayPurgeRecord);
                }
            }
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
            ..Default::default()
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

    #[test]
    fn method_state_fence_matrix_is_exhaustive() {
        // Every named admission operation × every thread status, as a
        // plain nested-loop table (randomizing the id added nothing —
        // the property never depended on it). `Completed` is in the
        // matrix so its Create/Fork carve-out is pinned rather than
        // dead-coded, and `Commit` so a committed-turn write on a
        // fenced thread is provably refused.
        let operations = [
            ThreadOperation::Create,
            ThreadOperation::Fork,
            ThreadOperation::Submit,
            ThreadOperation::Spawn,
            ThreadOperation::Promote,
            ThreadOperation::Acquire,
            ThreadOperation::Commit,
        ];
        let statuses = [
            ThreadStatus::Active,
            ThreadStatus::Completed,
            ThreadStatus::Deleting,
            ThreadStatus::Deleted,
        ];

        for operation in operations {
            for status in statuses {
                let mut thread = Thread::new(ThreadId::from_string("matrix"), t0());
                if status == ThreadStatus::Completed {
                    thread.committed_turns = 1;
                }
                thread.status = status;
                let expected_ok = match status {
                    ThreadStatus::Active => true,
                    ThreadStatus::Completed => {
                        matches!(operation, ThreadOperation::Create | ThreadOperation::Fork)
                    }
                    ThreadStatus::Deleting | ThreadStatus::Deleted => false,
                };
                match thread.require_active(operation) {
                    Ok(()) => assert!(expected_ok, "{status:?} unexpectedly accepted {operation}"),
                    Err(error) => {
                        assert!(
                            !expected_ok,
                            "{status:?} unexpectedly refused {operation}: {error}",
                        );
                        assert_eq!(error.thread_id, thread.thread_id);
                        assert_eq!(error.status, status);
                        assert_eq!(error.operation, operation);
                    }
                }
            }
        }
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
