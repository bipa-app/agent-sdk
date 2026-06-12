//! Storage traits for message history and agent state.
//!
//! Re-exported from [`agent_sdk_tools::stores`]. The optional [`sqlite`] module
//! adds a durable, single-file [`SqliteStore`](sqlite::SqliteStore) that
//! implements all four store traits.

pub use agent_sdk_tools::stores::*;

/// Embedded, durable, single-file SQLite-backed session store.
///
/// Behind the `sqlite` cargo feature. See [`SqliteStore`](sqlite::SqliteStore).
#[cfg(feature = "sqlite")]
#[cfg_attr(docsrs, doc(cfg(feature = "sqlite")))]
pub mod sqlite;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentEventEnvelope;
    use agent_sdk_foundation::events::{AgentEvent, SequenceCounter};
    use agent_sdk_foundation::types::ThreadId;
    use anyhow::Result;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The `ObservingEventStore` decorator is reachable through the façade and
    /// forwards every appended envelope to the closure before delegating
    /// persistence to the inner store.
    #[tokio::test]
    async fn observing_event_store_forwards_each_append_via_facade() -> Result<()> {
        let seen = Arc::new(AtomicUsize::new(0));
        let seen_in_cb = Arc::clone(&seen);
        let store = ObservingEventStore::new(InMemoryEventStore::new(), move |_envelope| {
            seen_in_cb.fetch_add(1, Ordering::SeqCst);
        });
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("m1", "a"), &seq),
            )
            .await?;
        store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("m2", "b"), &seq),
            )
            .await?;

        assert_eq!(
            seen.load(Ordering::SeqCst),
            2,
            "observer must run once per appended envelope"
        );
        assert_eq!(
            store.get_events(&thread_id).await?.len(),
            2,
            "inner store must persist every appended envelope"
        );
        Ok(())
    }
}
