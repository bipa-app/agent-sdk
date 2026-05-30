//! Durable idempotency records for at-least-once client requests.
//!
//! Phase 10 · E. Clients that retry mutating control-plane calls
//! (`CreateThread`, `SubmitThreadWork`, `ForkThread`,
//! `DecideConfirmation`) carry a caller-supplied `request_id`. Before
//! this module the dedup table lived in process memory
//! (`Arc<Mutex<IdempotencyState>>` in the gRPC layer), so a retry that
//! arrived after a host restart was *not* deduped — the same
//! `request_id` produced a duplicate root turn, a double fork, or a
//! double-applied decision (the classic at-least-once footgun).
//!
//! The records here are durable: they survive restart and, for the
//! submission path, are claimed **inside the same transaction** as task
//! admission so there is no time-of-check / time-of-use window.
//!
//! # Shape
//!
//! Every record is keyed by `request_id` and carries:
//! - the [`IdempotencyKind`] the key was first used for (so a retry that
//!   re-uses a key for a *different* operation kind is rejected),
//! - a `fingerprint` of the request payload (so a retry that re-uses a
//!   key with a *different* payload is rejected with a conflict rather
//!   than silently returning the original result), and
//! - the durable references to the original effect (thread id, task id,
//!   …) so a retry can reconstruct the original response without
//!   re-running the effect.

use serde::{Deserialize, Serialize};

/// Which control-plane operation a `request_id` was first claimed for.
///
/// A retry that re-uses a `request_id` for a *different* kind is a
/// caller bug and must be surfaced, not silently aliased onto an
/// unrelated effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdempotencyKind {
    /// `CreateThread`.
    CreateThread,
    /// `SubmitThreadWork`.
    SubmitWork,
    /// `ForkThread`.
    ForkThread,
    /// `DecideConfirmation`.
    DecideConfirmation,
}

impl IdempotencyKind {
    /// Stable wire string used as the persisted discriminator.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CreateThread => "create_thread",
            Self::SubmitWork => "submit_work",
            Self::ForkThread => "fork_thread",
            Self::DecideConfirmation => "decide_confirmation",
        }
    }

    /// Parse a persisted discriminator back into the typed kind.
    #[must_use]
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "create_thread" => Some(Self::CreateThread),
            "submit_work" => Some(Self::SubmitWork),
            "fork_thread" => Some(Self::ForkThread),
            "decide_confirmation" => Some(Self::DecideConfirmation),
            _ => None,
        }
    }
}

/// A durable idempotency record keyed by `request_id`.
///
/// The `result_json` blob is operation-specific: the gRPC layer encodes
/// the durable references it needs to reconstruct the original response
/// (e.g. the minted thread id, the admitted task id). The journal layer
/// treats it as an opaque payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    /// Caller-supplied idempotency key.
    pub request_id: String,
    /// The operation kind this key was first claimed for.
    pub kind: IdempotencyKind,
    /// Fingerprint of the original request payload.
    pub fingerprint: Vec<u8>,
    /// Operation-specific durable references to the original effect.
    pub result_json: serde_json::Value,
}

/// Outcome of claiming a `request_id` against the durable idempotency
/// table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdempotencyClaim {
    /// No record existed; the caller now owns the key and must run the
    /// effect.
    Fresh,
    /// A record already exists with a matching kind + fingerprint. The
    /// caller should reconstruct the original response from the stored
    /// references rather than re-running the effect.
    Replay(Box<IdempotencyRecord>),
    /// A record exists under this key but for a different kind or
    /// payload fingerprint — a caller contract violation.
    Conflict,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_wire_round_trips() {
        for kind in [
            IdempotencyKind::CreateThread,
            IdempotencyKind::SubmitWork,
            IdempotencyKind::ForkThread,
            IdempotencyKind::DecideConfirmation,
        ] {
            assert_eq!(IdempotencyKind::from_wire(kind.as_str()), Some(kind));
        }
        assert_eq!(IdempotencyKind::from_wire("bogus"), None);
    }

    #[test]
    fn record_json_round_trips() -> anyhow::Result<()> {
        let record = IdempotencyRecord {
            request_id: "req-1".into(),
            kind: IdempotencyKind::SubmitWork,
            fingerprint: vec![1, 2, 3],
            result_json: serde_json::json!({ "task_id": "task-7" }),
        };
        let encoded = serde_json::to_string(&record)?;
        let decoded: IdempotencyRecord = serde_json::from_str(&encoded)?;
        assert_eq!(decoded, record);
        Ok(())
    }
}
