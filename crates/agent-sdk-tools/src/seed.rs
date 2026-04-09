//! Durable reconstruction types for worker-context recovery.
//!
//! [`ToolContextSeed`] captures the durable state a worker needs to
//! reconstruct a [`crate::tools::ToolContext`] deterministically.  Host-provided runtime
//! dependencies are represented by [`HostDependencies`], and the
//! [`ExecutionContextFactory`] trait abstracts how a host combines these
//! to build a ready-to-use context.
//!
//! ## Design rationale
//!
//! The server must rebuild `ToolContext` from two sources:
//!
//! 1. **Durable task state** — thread identity, turn number, event-sequence
//!    offset, user-defined metadata.  These survive restarts and are stored
//!    in the task / thread record.  [`ToolContextSeed`] models this.
//!
//! 2. **Host-provided runtime deps** — event store, event authority,
//!    cancellation tokens, concurrency limiters.  These are created fresh
//!    by the orchestration layer each time a worker is started.
//!    [`HostDependencies`] models this.
//!
//! By keeping these separate the Phase 4 root worker and Phase 5
//! tool-runtime worker can both depend on this contract directly, and the
//! server never has to infer context shape from SDK internals.

use crate::stores::EventStore;
use agent_sdk_core::types::ThreadId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Durable reconstruction payload
// ---------------------------------------------------------------------------

/// Durable inputs needed to reconstruct a [`crate::tools::ToolContext`].
///
/// Every field is recoverable from the task / thread record in durable
/// storage.  The server serialises this when a task is created and
/// deserialises it when a worker starts (or restarts).
///
/// This is the *stable* reconstruction contract — later phases depend on
/// its shape, so changing it requires a version bump.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolContextSeed {
    /// Thread identity for the current conversation.
    pub thread_id: ThreadId,
    /// Turn number within the thread (1-based).
    pub turn: usize,
    /// Sequence offset for event ordering continuity across turns.
    ///
    /// The event authority uses this to resume numbering where the
    /// previous turn left off, so events within a thread form a
    /// single monotonic sequence even across worker restarts.
    pub sequence_offset: u64,
    /// Arbitrary key-value metadata forwarded to the tool context.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolContextSeed {
    /// Create a seed for a fresh first turn with no prior state.
    #[must_use]
    pub fn first_turn(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            turn: 1,
            sequence_offset: 0,
            metadata: HashMap::new(),
        }
    }

    /// Builder-style setter for metadata.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Host-provided runtime dependencies
// ---------------------------------------------------------------------------

/// Runtime dependencies created by the host when a worker is started.
///
/// These are *not* durable — they are constructed fresh each time and do
/// not survive worker restarts.  The orchestration layer is responsible
/// for providing them.
///
/// Note: the event authority is intentionally absent — [`crate::tools::ToolContext::from_seed`]
/// constructs it internally from [`ToolContextSeed::sequence_offset`] to
/// guarantee monotonic sequencing.
pub struct HostDependencies {
    /// Authoritative event store for persisting tool-originated events.
    pub event_store: Arc<dyn EventStore>,
    /// Token for cooperative cancellation.
    pub cancel_token: CancellationToken,
    /// Optional concurrency limiter for subagent spawning.
    pub subagent_semaphore: Option<Arc<tokio::sync::Semaphore>>,
}

// ---------------------------------------------------------------------------
// Factory trait
// ---------------------------------------------------------------------------

/// Abstraction for how hosts build a [`crate::tools::ToolContext`] from
/// durable seeds and fresh runtime dependencies.
///
/// The SDK provides [`crate::tools::ToolContext::from_seed`] as the default
/// implementation — hosts can call it directly or implement this trait to
/// add additional ambient state (e.g. database connections wrapped in the
/// application context `Ctx`).
pub trait ExecutionContextFactory<Ctx>: Send + Sync {
    /// Build a ready-to-use `ToolContext` from the durable seed, the
    /// application context, and host-provided runtime dependencies.
    fn build(
        &self,
        seed: &ToolContextSeed,
        app: Ctx,
        deps: HostDependencies,
    ) -> crate::tools::ToolContext<Ctx>;
}

/// Default factory that delegates to [`crate::tools::ToolContext::from_seed`].
///
/// Hosts that do not need custom ambient injection can use this directly.
pub struct DefaultContextFactory;

impl<Ctx> ExecutionContextFactory<Ctx> for DefaultContextFactory {
    fn build(
        &self,
        seed: &ToolContextSeed,
        app: Ctx,
        deps: HostDependencies,
    ) -> crate::tools::ToolContext<Ctx> {
        crate::tools::ToolContext::from_seed(seed, app, deps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_first_turn_defaults() {
        let seed = ToolContextSeed::first_turn(ThreadId::from_string("t-1"));
        assert_eq!(seed.thread_id, ThreadId::from_string("t-1"));
        assert_eq!(seed.turn, 1);
        assert_eq!(seed.sequence_offset, 0);
        assert!(seed.metadata.is_empty());
    }

    #[test]
    fn seed_with_metadata() {
        let seed = ToolContextSeed::first_turn(ThreadId::new())
            .with_metadata("user_id", serde_json::json!("u-42"));
        assert_eq!(
            seed.metadata.get("user_id"),
            Some(&serde_json::json!("u-42"))
        );
    }

    #[test]
    fn seed_round_trips_through_json() -> anyhow::Result<()> {
        let original = ToolContextSeed {
            thread_id: ThreadId::from_string("t-round-trip"),
            turn: 5,
            sequence_offset: 42,
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".into(), serde_json::json!("value"));
                m
            },
        };
        let json = serde_json::to_string(&original)?;
        let recovered: ToolContextSeed = serde_json::from_str(&json)?;
        assert_eq!(recovered, original);
        Ok(())
    }
}
