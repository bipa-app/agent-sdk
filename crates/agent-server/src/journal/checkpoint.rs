//! Completed-turn checkpoint schema and invariants.
//!
//! A [`Checkpoint`] is an immutable snapshot of the conversation state
//! at the moment a turn commits successfully. It stores the full
//! message history and an opaque agent-state blob so v1 recovery can
//! restore a thread to any committed turn without replaying from
//! epoch.
//!
//! # Uniqueness rule
//!
//! There is **exactly one** checkpoint per `(thread_id, turn_number)`.
//! The store enforces this with a partial-unique index and rejects
//! duplicate inserts.
//!
//! # Lifecycle
//!
//! Checkpoints are **immutable** — once created they are never
//! modified or deleted. This matches the append-only audit philosophy
//! of [`super::turn_attempt::TurnAttempt`].
//!
//! # What this table does **not** own
//!
//! - Thread aggregate counters → [`super::thread::Thread`]
//! - Committed message projection → [`super::message::MessageProjection`]
//! - Turn-attempt audit records → [`super::turn_attempt::TurnAttempt`]
//! - Recovery loaders → [`super::thread_recover`]

use agent_sdk_core::{ThreadId, TokenUsage, llm};
use serde::{Deserialize, Serialize};
use std::fmt;
use time::OffsetDateTime;
use uuid::Uuid;

use super::task::AgentTaskId;

// ─────────────────────────────────────────────────────────────────────
// Identity
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for a checkpoint row.
///
/// Formatted as `checkpoint_<uuid>` to be visually distinct from task
/// IDs (`task_<uuid>`), lease IDs (`lease_<uuid>`), and attempt IDs
/// (`attempt_<uuid>`) in logs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CheckpointId(pub String);

impl CheckpointId {
    /// Allocate a fresh checkpoint ID.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("checkpoint_{}", Uuid::new_v4()))
    }

    /// Wrap an existing string as a checkpoint ID (used by stores
    /// when rehydrating rows from durable storage).
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CheckpointId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors for [`Checkpoint`] rows.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointSchemaError {
    /// `turn_number` must be ≥ 1 (turns are 1-indexed).
    #[error("turn_number must be >= 1")]
    ZeroTurnNumber,
}

// ─────────────────────────────────────────────────────────────────────
// Checkpoint
// ─────────────────────────────────────────────────────────────────────

/// One row in the `checkpoints` table.
///
/// A checkpoint captures the complete conversation state at the
/// instant a turn commits successfully. The row is created by the
/// atomic commit path ([`super::commit::commit_completed_turn`]) and
/// is never modified afterward.
///
/// # Fields
///
/// | Group | Fields |
/// |-------|--------|
/// | Identity | `id`, `thread_id`, `turn_number`, `task_id` |
/// | Snapshot | `messages`, `agent_state_snapshot` |
/// | Usage | `turn_usage` |
/// | Timing | `created_at` |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique row identity.
    pub id: CheckpointId,

    /// The thread this checkpoint belongs to.
    pub thread_id: ThreadId,

    /// 1-indexed turn number within the thread.
    ///
    /// Matches `Thread::committed_turns` after the turn commits.
    /// The `(thread_id, turn_number)` pair is the uniqueness key.
    pub turn_number: u32,

    /// The task that produced this turn.
    pub task_id: AgentTaskId,

    /// Full message history at this turn.
    ///
    /// This is a snapshot of the committed message projection at the
    /// moment the turn committed, not a delta. Recovery can restore
    /// the thread to this turn without replaying prior turns.
    pub messages: Vec<llm::Message>,

    /// Opaque agent-state blob for v1 recovery.
    ///
    /// Stored as a JSON value so the checkpoint schema does not need
    /// to understand the agent-state format. The recovery loader
    /// (future phase) deserializes this into the appropriate type.
    pub agent_state_snapshot: serde_json::Value,

    /// Token usage for this specific turn.
    pub turn_usage: TokenUsage,

    /// When this checkpoint was created.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

impl Checkpoint {
    /// Create a new checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointSchemaError::ZeroTurnNumber`] if
    /// `params.turn_number` is zero.
    pub fn new(params: NewCheckpointParams) -> Result<Self, CheckpointSchemaError> {
        let checkpoint = Self {
            id: CheckpointId::new(),
            thread_id: params.thread_id,
            turn_number: params.turn_number,
            task_id: params.task_id,
            messages: params.messages,
            agent_state_snapshot: params.agent_state_snapshot,
            turn_usage: params.turn_usage,
            created_at: params.now,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Check every structural invariant on this row.
    ///
    /// # Invariants
    ///
    /// - `turn_number` must be ≥ 1 (turns are 1-indexed).
    ///
    /// # Errors
    /// Returns the first violated invariant.
    pub const fn validate(&self) -> Result<(), CheckpointSchemaError> {
        if self.turn_number == 0 {
            return Err(CheckpointSchemaError::ZeroTurnNumber);
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Param structs
// ─────────────────────────────────────────────────────────────────────

/// Arguments for [`Checkpoint::new`].
///
/// Named fields prevent positional confusion for a struct that lands
/// in the durable checkpoint table.
#[derive(Clone, Debug)]
pub struct NewCheckpointParams {
    /// The thread this checkpoint belongs to.
    pub thread_id: ThreadId,
    /// 1-indexed turn number within the thread.
    pub turn_number: u32,
    /// The task that produced this turn.
    pub task_id: AgentTaskId,
    /// Full message history at this turn.
    pub messages: Vec<llm::Message>,
    /// Opaque agent-state snapshot for v1 recovery.
    pub agent_state_snapshot: serde_json::Value,
    /// Token usage for this specific turn.
    pub turn_usage: TokenUsage,
    /// Current wall-clock time.
    pub now: OffsetDateTime,
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn sample_params() -> NewCheckpointParams {
        NewCheckpointParams {
            thread_id: ThreadId::from_string("t-ckpt-test"),
            turn_number: 1,
            task_id: AgentTaskId::from_string("task_test-1"),
            messages: vec![llm::Message::user("hello")],
            agent_state_snapshot: serde_json::json!({
                "thread_id": "t-ckpt-test",
                "turn_count": 1,
            }),
            turn_usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            now: t0(),
        }
    }

    // ── Identity ─────────────────────────────────────────────────

    #[test]
    fn checkpoint_id_has_prefix() {
        let id = CheckpointId::new();
        assert!(
            id.as_str().starts_with("checkpoint_"),
            "expected checkpoint_ prefix, got: {id}",
        );
    }

    #[test]
    fn checkpoint_id_display_matches_inner() {
        let id = CheckpointId::from_string("checkpoint_abc");
        assert_eq!(format!("{id}"), "checkpoint_abc");
    }

    #[test]
    fn checkpoint_id_round_trips_through_json() -> Result<()> {
        let id = CheckpointId::from_string("checkpoint_rnd");
        let json = serde_json::to_string(&id).context("serialize")?;
        let back: CheckpointId = serde_json::from_str(&json).context("deserialize")?;
        assert_eq!(back, id);
        Ok(())
    }

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn new_checkpoint_validates_and_is_immutable() -> Result<()> {
        let ckpt = Checkpoint::new(sample_params()).context("new")?;

        assert_eq!(ckpt.thread_id, ThreadId::from_string("t-ckpt-test"));
        assert_eq!(ckpt.turn_number, 1);
        assert_eq!(ckpt.task_id, AgentTaskId::from_string("task_test-1"));
        assert_eq!(ckpt.messages.len(), 1);
        assert_eq!(ckpt.turn_usage.input_tokens, 100);
        assert_eq!(ckpt.turn_usage.output_tokens, 50);
        assert_eq!(ckpt.created_at, t0());
        ckpt.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn new_checkpoint_rejects_zero_turn_number() {
        let params = NewCheckpointParams {
            turn_number: 0,
            ..sample_params()
        };
        let err = Checkpoint::new(params).unwrap_err();
        assert_eq!(err, CheckpointSchemaError::ZeroTurnNumber);
    }

    #[test]
    fn checkpoint_with_turn_number_one_is_valid() -> Result<()> {
        let ckpt = Checkpoint::new(sample_params()).context("new")?;
        assert_eq!(ckpt.turn_number, 1);
        ckpt.validate().context("validate")?;
        Ok(())
    }

    // ── JSON round-trip ──────────────────────────────────────────

    #[test]
    fn checkpoint_round_trips_through_json() -> Result<()> {
        let ckpt = Checkpoint::new(sample_params()).context("new")?;
        let json = serde_json::to_string(&ckpt).context("serialize")?;
        let back: Checkpoint = serde_json::from_str(&json).context("deserialize")?;

        assert_eq!(back.id, ckpt.id);
        assert_eq!(back.thread_id, ckpt.thread_id);
        assert_eq!(back.turn_number, ckpt.turn_number);
        assert_eq!(back.task_id, ckpt.task_id);
        assert_eq!(back.messages.len(), ckpt.messages.len());
        assert_eq!(back.turn_usage, ckpt.turn_usage);
        back.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn checkpoint_json_contains_expected_fields() -> Result<()> {
        let ckpt = Checkpoint::new(sample_params()).context("new")?;
        let json = serde_json::to_value(&ckpt).context("serialize")?;

        for key in [
            "id",
            "thread_id",
            "turn_number",
            "task_id",
            "messages",
            "agent_state_snapshot",
            "turn_usage",
            "created_at",
        ] {
            assert!(
                json.get(key).is_some(),
                "Checkpoint wire format lost key `{key}` — this breaks the server contract"
            );
        }
        assert_eq!(json["turn_number"], 1);
        assert_eq!(json["thread_id"], "t-ckpt-test");
        Ok(())
    }

    // ── Multiple turns ───────────────────────────────────────────

    #[test]
    fn checkpoints_for_different_turns_have_distinct_ids() -> Result<()> {
        let c1 = Checkpoint::new(sample_params()).context("turn 1")?;
        let c2 = Checkpoint::new(NewCheckpointParams {
            turn_number: 2,
            ..sample_params()
        })
        .context("turn 2")?;

        assert_ne!(c1.id, c2.id);
        assert_eq!(c1.turn_number, 1);
        assert_eq!(c2.turn_number, 2);
        Ok(())
    }

    // ── Validation edge cases ────────────────────────────────────

    #[test]
    fn validate_rejects_zero_turn_number_on_existing_row() {
        let mut ckpt = Checkpoint::new(sample_params()).unwrap();
        ckpt.turn_number = 0;
        assert_eq!(ckpt.validate(), Err(CheckpointSchemaError::ZeroTurnNumber),);
    }

    #[test]
    fn checkpoint_with_empty_messages_is_valid() -> Result<()> {
        let params = NewCheckpointParams {
            messages: vec![],
            ..sample_params()
        };
        let ckpt = Checkpoint::new(params).context("new")?;
        ckpt.validate().context("validate")?;
        assert!(ckpt.messages.is_empty());
        Ok(())
    }
}
