//! Optional atomic completed-turn commit hook for durable backends.
//!
//! The in-memory stores satisfy [`super::commit::commit_completed_turn`]
//! by calling the per-store trait methods sequentially because each
//! individual mutation is already atomic under a single write lock.
//! Durable SQL backends need a stronger contract: the attempt close,
//! thread aggregate advance, message projection update, raw message
//! batch append, and checkpoint insert must share one database
//! transaction so recovery never observes a partial turn commit.

use anyhow::Result;
use async_trait::async_trait;

use super::commit::{CommitOutcome, CompletedTurnCommit};

/// Backend-specific hook for atomically committing a completed turn.
///
/// Implement this on a durable backend that can coordinate the
/// completed-turn write set inside one transaction. The generic
/// [`super::commit::commit_completed_turn`] helper will delegate to
/// this hook when the active [`super::thread_store::ThreadStore`]
/// exposes it.
#[async_trait]
pub trait AtomicCompletedTurnCommitter: Send + Sync {
    /// Commit the completed-turn state projections atomically.
    ///
    /// Implementations should commit the durable state projections only
    /// (attempt close, thread aggregate, message batch/head, and
    /// checkpoint). Lifecycle events remain outside this hook because
    /// the current durable-core Postgres contract intentionally leaves
    /// the public event journal for a later track.
    async fn commit_completed_turn_atomic(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome>;
}
