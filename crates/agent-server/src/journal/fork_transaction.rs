//! Optional atomic fork commit hook for durable backends.
//!
//! `agent_service_host::grpc::GrpcControlService::fork_thread`
//! produces a destination thread by reading the source's committed
//! state through a chosen turn boundary and writing the equivalent
//! state under a freshly-minted `thread_id`. The in-memory store
//! satisfies that contract by calling each per-store trait method
//! sequentially — each individual mutation is atomic under its own
//! write lock and the test harness doesn't care about partial
//! recovery.
//!
//! Durable backends need a stronger contract. Without one, a crash
//! mid-fork can leave a half-built thread behind (aggregate row
//! exists, projection rewritten, but the events / checkpoint never
//! committed) — and a retry under the same `request_id` would still
//! either run into a `(thread_id, turn_number)` uniqueness conflict
//! on the half-written checkpoint, or silently double-append events.
//! Wrapping the entire write set in one transaction makes the fork
//! all-or-nothing: a partial fork is impossible to observe, and a
//! retry after a transport failure either sees an empty destination
//! (operation did not start) or a complete one (operation already
//! durably succeeded; the `request_id` replay catches it).
//!
//! The hook is optional: backends that can't transact across stores
//! return `None` and the caller falls back to sequential writes
//! with the documented partial-state caveats.

use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};

use super::checkpoint::NewCheckpointParams;
use super::thread_store::{ThreadCreation, ThreadCreationOutcome};

/// Authoritative input for a single atomic fork-state copy.
///
/// Built by `GrpcControlService::fork_thread` after reading the
/// source's committed state. Every field corresponds 1:1 with one
/// mutation a sequential implementation would otherwise have to do
/// in isolation; carrying them as a single bundle is what lets the
/// backend wrap them in one transaction.
#[derive(Clone, Debug)]
pub struct ForkCommitParams {
    /// Server-minted destination thread id. The hook is responsible
    /// for inserting the thread aggregate row keyed on this id.
    pub new_thread_id: ThreadId,
    /// Caller-addressed creation identity. `None` preserves the legacy
    /// server-minted path; `Some` enables primary-key get-or-create.
    pub creation: Option<ThreadCreation>,
    /// Wall-clock timestamp the new thread aggregate, projection,
    /// checkpoint, and events should record as their `created_at` /
    /// `updated_at` / `committed_at`. Held outside individual store
    /// methods so the entire transaction observes one consistent
    /// timestamp.
    pub now: OffsetDateTime,
    /// Number of `commit_turn` calls to fold into the destination
    /// thread aggregate. Mirrors the source's `committed_turns`
    /// count at the fork boundary so `recover_thread`'s
    /// `committed_turns == checkpoint.turn_number` guard accepts
    /// the destination on the next worker pickup.
    pub committed_turns: u32,
    /// Cumulative `total_usage` to seed on the destination's
    /// thread aggregate, extracted from the source's
    /// `agent_state_snapshot.total_usage` at the fork boundary.
    /// This makes the fork's `total_usage` match what the source
    /// reported at that checkpoint instead of starting at zero —
    /// the source's accumulated cost paid for every message we're
    /// inheriting, so the fork's running total should reflect
    /// that on day one.
    pub cumulative_total_usage: TokenUsage,
    /// Seed messages for the destination's projection
    /// (`agent_sdk_message_heads`). Empty for a turn-zero fork.
    pub messages: Vec<llm::Message>,
    /// Seed checkpoint for the destination, with the
    /// `agent_state_snapshot.thread_id` already rewritten to
    /// `new_thread_id`. `None` for a turn-zero fork (no source
    /// checkpoint to mirror).
    pub checkpoint: Option<NewCheckpointParams>,
    /// Full destination journal seed: `events[0]` is the destination's
    /// `ThreadCreated` (built once by the params constructor — the ONE
    /// choke point for that invariant), followed by the source events
    /// re-committed under the destination. Fresh sequences are assigned
    /// inside the transaction; committers write this vector verbatim.
    pub events: Vec<AgentEvent>,
    /// Maximum relay attempts for the coalesced outbox advisory.
    pub outbox_max_attempts: u32,
}

/// Backend-specific hook for atomically committing a fork.
///
/// Implement this on a durable backend that can coordinate the fork
/// write set inside one transaction. The hook is queried via
/// [`super::thread_store::ThreadStore::atomic_fork_committer`]; when
/// the active backend exposes it, the gRPC handler delegates the
/// state copy to this method instead of running the per-store
/// mutations sequentially.
#[async_trait]
pub trait AtomicForkCommitter: Send + Sync {
    /// Atomically commit the fork's state to the destination thread.
    ///
    /// Implementations must satisfy:
    ///
    /// 1. Either every mutation in `params` is durably visible after
    ///    `Ok(())` returns, or none of them is. A crash or returned
    ///    error must leave the destination thread in the
    ///    not-created state.
    /// 2. Sequence numbers for `params.events` are assigned inside
    ///    the same transaction so the new thread's
    ///    `EventRepository::next_sequence` is consistent with the
    ///    committed events the moment the transaction commits.
    /// 3. The thread aggregate's `committed_turns` ends at the
    ///    `params.committed_turns` value; in-memory backends that
    ///    don't implement this hook achieve the same end state by
    ///    looping `commit_turn` outside the transaction.
    /// Returns whether this call created the destination or found the exact
    /// same caller-addressed fork already committed.
    async fn commit_fork_atomic(&self, params: ForkCommitParams) -> Result<ThreadCreationOutcome>;
}
