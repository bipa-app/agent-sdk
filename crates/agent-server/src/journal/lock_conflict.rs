//! The marker a store attaches to a transition that failed on a lock
//! conflict it could not wait out.
//!
//! Callers and telemetry use it to tell such a failure from any other store
//! failure. Postgres attaches it after its deadlock and
//! serialization-failure retries run out; SQLite attaches it to a commit
//! that failed `SQLITE_BUSY`/`SQLITE_LOCKED` after its busy timeout. The
//! in-memory store cannot conflict.

/// A store gave up retrying this transition after lock conflicts.
///
/// It retried after deadlocks or serialization failures until its retry
/// budget ran out. Attached as `anyhow` context to
/// the [`Rejected::Store`](super::Rejected::Store) error; find it with
/// `error.downcast_ref::<LockConflict>()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the transition still hit a lock conflict after {retries} retries")]
pub struct LockConflict {
    /// How many times the store itself retried before giving up. 0 on
    /// SQLite, where `PRAGMA busy_timeout` does the waiting and the store
    /// does not retry.
    pub retries: u32,
}
