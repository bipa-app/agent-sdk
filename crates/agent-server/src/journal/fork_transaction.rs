//! The write set of one fork, committed by
//! [`super::thread_store::ThreadStore::commit_fork`].
//!
//! `agent_service_host::grpc::GrpcControlService::fork_thread` reads the
//! source's committed state through a chosen turn boundary and hands the
//! store everything the destination needs. The store writes it as one unit:
//! a crash or error leaves the destination not created, never half-built,
//! so a retry under the same `request_id` sees either nothing or the whole
//! fork.

use time::OffsetDateTime;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};

use super::checkpoint::NewCheckpointParams;
use super::thread_store::ThreadCreation;

/// Authoritative input for a single atomic fork-state copy.
///
/// Built by `GrpcControlService::fork_thread` after reading the
/// source's committed state. Every field corresponds 1:1 with one
/// mutation a sequential implementation would otherwise have to do
/// in isolation; carrying them as a single bundle is what lets the
/// backend wrap them in one transaction.
#[derive(Clone, Debug)]
pub struct ForkCommitParams {
    /// The thread the fork copies. The store refuses the fork when it is
    /// purged ([`Fence::FORK_SOURCE`](super::transition::Fence::FORK_SOURCE)).
    pub source_thread_id: ThreadId,
    /// Destination thread id; the store inserts its aggregate row.
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
