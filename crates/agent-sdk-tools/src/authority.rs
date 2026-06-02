//! Event envelope authority — governs how raw events are wrapped in
//! [`AgentEventEnvelope`]s with unique IDs, sequence numbers, and timestamps.
//!
//! In local/CLI mode the SDK creates a [`LocalEventAuthority`] that starts
//! sequencing at 0 for each run.  In server mode the orchestration layer
//! injects its own authority (or seeds the offset) so ordering is continuous
//! across turns within the same thread.

use agent_sdk_foundation::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};

/// Authority that governs how events are wrapped in envelopes.
///
/// In local/CLI mode a fresh [`SequenceCounter`] starts at 0 per run.
/// In server mode the authority seeds sequences from durable storage
/// so ordering is continuous across turns within a thread.
pub trait EventAuthority: Send + Sync {
    /// Wrap a raw event into an authoritative envelope.
    fn wrap(&self, event: AgentEvent) -> AgentEventEnvelope;
}

/// Default event authority for local/CLI execution.
///
/// Creates envelopes using a [`SequenceCounter`] that starts at 0 (or a
/// caller-supplied offset).  This reproduces the existing behaviour — it is
/// simply extracted behind the [`EventAuthority`] trait so the server can
/// substitute its own implementation.
pub struct LocalEventAuthority {
    seq: SequenceCounter,
}

impl LocalEventAuthority {
    /// Create an authority that starts sequencing at 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seq: SequenceCounter::new(),
        }
    }

    /// Create an authority that starts sequencing at the given offset.
    ///
    /// Used by server mode to resume sequencing where the previous turn
    /// left off.
    #[must_use]
    pub fn with_offset(start: u64) -> Self {
        Self {
            seq: SequenceCounter::with_offset(start),
        }
    }
}

impl Default for LocalEventAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl EventAuthority for LocalEventAuthority {
    fn wrap(&self, event: AgentEvent) -> AgentEventEnvelope {
        AgentEventEnvelope::wrap(event, &self.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> AgentEvent {
        AgentEvent::text("msg_test", "hello")
    }

    #[test]
    fn local_authority_sequences_from_zero() {
        let auth = LocalEventAuthority::new();
        let e0 = auth.wrap(sample_event());
        let e1 = auth.wrap(sample_event());
        let e2 = auth.wrap(sample_event());
        assert_eq!(e0.sequence, 0);
        assert_eq!(e1.sequence, 1);
        assert_eq!(e2.sequence, 2);
    }

    #[test]
    fn local_authority_with_offset_resumes_sequencing() {
        let auth = LocalEventAuthority::with_offset(100);
        let e0 = auth.wrap(sample_event());
        let e1 = auth.wrap(sample_event());
        assert_eq!(e0.sequence, 100);
        assert_eq!(e1.sequence, 101);
    }

    #[test]
    fn local_authority_default_same_as_new() {
        let auth = LocalEventAuthority::default();
        assert_eq!(auth.wrap(sample_event()).sequence, 0);
    }

    #[test]
    fn local_authority_assigns_unique_event_ids() {
        let auth = LocalEventAuthority::new();
        let ids: Vec<_> = (0..100)
            .map(|_| auth.wrap(sample_event()).event_id)
            .collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }
}
