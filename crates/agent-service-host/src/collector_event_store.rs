//! [`EventStore`] adapter that funnels tool-emitted events into a
//! [`ToolEventCollector`].
//!
//! The v2 durable tool runtime (see
//! [`agent_server::worker::tool_task::execute_tool_task`]) buffers
//! progress events in a [`ToolEventCollector`] during execution and
//! commits them to the event repository **after** the CAS-guarded
//! state transition succeeds.  This preserves the "no orphaned events
//! from stale-lease workers" invariant.
//!
//! A [`ToolContext<Ctx>`] built for inline execution, however, routes
//! [`ToolContext::emit_event`] calls through an [`EventStore`] +
//! [`EventAuthority`].  Plugging a real [`EventStore`] into the v2
//! executor would bypass the collector and double-write events.
//!
//! [`CollectorEventStore`] bridges the two worlds:  it implements
//! [`EventStore::append`] by unwrapping the envelope and pushing the
//! inner [`AgentEvent`](agent_sdk_foundation::events::AgentEvent) into the
//! collector; [`commit_tool_events`] re-wraps them with the worker's
//! authoritative sequence numbers after the state transition commits.
//!
//! The store is **write-only** for the tool-execution path.  Read
//! methods return empty results because tool code has no legitimate
//! reason to read from its own emit sink during a single call; a tool
//! that wants to replay history should be passed a separate event
//! store in its application context.
//!
//! [`ToolContext<Ctx>`]: agent_sdk_tools::ToolContext
//! [`ToolContext::emit_event`]: agent_sdk_tools::ToolContext::emit_event
//! [`EventAuthority`]: agent_sdk_tools::EventAuthority
//! [`commit_tool_events`]: agent_server::worker::tool_task

use agent_sdk_foundation::events::AgentEventEnvelope;
use agent_sdk_foundation::types::ThreadId;
use agent_sdk_tools::stores::{EventStore, StoredTurnEvents};
use agent_server::worker::ToolEventCollector;
use anyhow::Result;
use async_trait::async_trait;

/// Write-only [`EventStore`] adapter that forwards appended events to
/// a [`ToolEventCollector`].
///
/// Intended to be installed as the [`HostDependencies::event_store`]
/// of a [`ToolContext<Ctx>`] constructed by a v2 [`RegistryToolExecutor`].
///
/// [`HostDependencies::event_store`]: agent_sdk_tools::seed::HostDependencies::event_store
/// [`ToolContext<Ctx>`]: agent_sdk_tools::ToolContext
/// [`RegistryToolExecutor`]: crate::registry_tool_executor::RegistryToolExecutor
pub struct CollectorEventStore {
    collector: ToolEventCollector,
}

impl CollectorEventStore {
    #[must_use]
    pub const fn new(collector: ToolEventCollector) -> Self {
        Self { collector }
    }
}

#[async_trait]
impl EventStore for CollectorEventStore {
    async fn append(
        &self,
        _thread_id: &ThreadId,
        _turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()> {
        // The envelope's sequence/id/timestamp are discarded: the worker's
        // commit path re-wraps each event with its authoritative sequence
        // number before persisting to the event repository.
        self.collector.emit(envelope.event);
        Ok(())
    }

    async fn finish_turn(&self, _thread_id: &ThreadId, _turn: usize) -> Result<()> {
        // Turn finishing is owned by the worker, not the tool.
        Ok(())
    }

    async fn get_turn(
        &self,
        _thread_id: &ThreadId,
        _turn: usize,
    ) -> Result<Option<StoredTurnEvents>> {
        Ok(None)
    }

    async fn get_turns(&self, _thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
        Ok(Vec::new())
    }

    async fn clear(&self, _thread_id: &ThreadId) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::events::{AgentEvent, SequenceCounter};

    #[tokio::test]
    async fn append_forwards_event_to_collector() -> Result<()> {
        let collector = ToolEventCollector::new();
        let store = CollectorEventStore::new(collector.clone());
        let thread = ThreadId::new();
        let seq = SequenceCounter::new();

        let envelope = AgentEventEnvelope::wrap(AgentEvent::text("msg-1", "hello"), &seq);
        store.append(&thread, 1, envelope).await?;
        store
            .append(
                &thread,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("msg-2", "world"), &seq),
            )
            .await?;

        let drained = collector.drain();
        assert_eq!(drained.len(), 2);
        match &drained[0] {
            AgentEvent::Text { text, .. } => assert_eq!(text, "hello"),
            other => panic!("unexpected event: {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn read_methods_return_empty() -> Result<()> {
        let collector = ToolEventCollector::new();
        let store = CollectorEventStore::new(collector);
        let thread = ThreadId::new();

        assert!(store.get_turn(&thread, 1).await?.is_none());
        assert!(store.get_turns(&thread).await?.is_empty());
        store.finish_turn(&thread, 1).await?;
        store.clear(&thread).await?;
        Ok(())
    }
}
