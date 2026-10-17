//! The marker a store attaches to a transition it stopped retrying after
//! lock conflicts, so callers and telemetry can tell it from any other
//! store failure.

/// A store gave up retrying this transition after lock conflicts.
///
/// It retried after deadlocks or serialization failures until its retry
/// budget ran out. Attached as `anyhow` context to
/// the [`Rejected::Store`](super::Rejected::Store) error; find it with
/// `error.downcast_ref::<LockConflict>()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the transition still hit a lock conflict after {retries} retries")]
pub struct LockConflict {
    /// How many times the store retried before giving up.
    pub retries: u32,
}
