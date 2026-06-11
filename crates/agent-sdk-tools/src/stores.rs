//! Storage traits for message history, agent state, and event persistence.
//!
//! The SDK uses three storage abstractions:
//!
//! - [`MessageStore`] - Stores conversation message history per thread
//! - [`StateStore`] - Stores agent state checkpoints for recovery
//! - [`EventStore`] - Stores turn-scoped event envelopes for retrieval
//!
//! # Built-in Implementation
//!
//! [`InMemoryStore`] implements the message/state traits and is suitable for
//! testing and single-process deployments. [`InMemoryEventStore`] provides the
//! corresponding in-memory event journal. For production, implement custom
//! stores backed by your database (e.g., Postgres, Redis).

use agent_sdk_foundation::events::AgentEventEnvelope;
use agent_sdk_foundation::llm;
use agent_sdk_foundation::types::{AgentState, ThreadId, ToolExecution};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::RwLock as AsyncRwLock;

/// Trait for storing and retrieving conversation messages.
/// Implement this trait to persist messages to your storage backend.
#[async_trait]
pub trait MessageStore: Send + Sync {
    /// Append a message to the thread's history
    ///
    /// # Errors
    /// Returns an error if the message cannot be stored.
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()>;

    /// Get all messages for a thread
    ///
    /// # Errors
    /// Returns an error if the history cannot be retrieved.
    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>>;

    /// Clear all messages for a thread
    ///
    /// # Errors
    /// Returns an error if the messages cannot be cleared.
    async fn clear(&self, thread_id: &ThreadId) -> Result<()>;

    /// Get the message count for a thread
    ///
    /// # Errors
    /// Returns an error if the count cannot be retrieved.
    async fn count(&self, thread_id: &ThreadId) -> Result<usize> {
        Ok(self.get_history(thread_id).await?.len())
    }

    /// Replace the entire message history for a thread.
    /// Used for context compaction to replace old messages with a summary.
    ///
    /// # Errors
    /// Returns an error if the history cannot be replaced.
    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
    ) -> Result<()>;
}

/// Trait for storing agent state checkpoints.
/// Implement this to enable conversation recovery and resume.
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Save the current agent state
    ///
    /// # Errors
    /// Returns an error if the state cannot be saved.
    async fn save(&self, state: &AgentState) -> Result<()>;

    /// Load the most recent state for a thread
    ///
    /// # Errors
    /// Returns an error if the state cannot be loaded.
    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>>;

    /// Delete state for a thread
    ///
    /// # Errors
    /// Returns an error if the state cannot be deleted.
    async fn delete(&self, thread_id: &ThreadId) -> Result<()>;
}

/// Stored event data for a single turn.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct StoredTurnEvents {
    /// Turn number (1-based once execution starts).
    pub turn: usize,
    /// Events emitted for this turn.
    pub events: Vec<AgentEventEnvelope>,
    /// Whether `finish_turn()` has completed for this turn.
    pub finished: bool,
}

/// Trait for storing and retrieving turn-scoped event streams.
///
/// Event writes are split into two phases:
/// 1. [`append`](EventStore::append) records individual envelopes
/// 2. [`finish_turn`](EventStore::finish_turn) marks the authoritative close barrier
#[async_trait]
pub trait EventStore: Send + Sync {
    /// Append an event envelope for the given thread and turn.
    ///
    /// # Errors
    /// Returns an error if the event cannot be persisted.
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()>;

    /// Mark the given turn as finished and flush any buffered writes.
    ///
    /// # Errors
    /// Returns an error if the store cannot durably close the turn.
    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()>;

    /// Retrieve the stored data for a single turn.
    ///
    /// # Errors
    /// Returns an error if the turn cannot be retrieved.
    async fn get_turn(&self, thread_id: &ThreadId, turn: usize)
    -> Result<Option<StoredTurnEvents>>;

    /// Retrieve all stored turns for the given thread in ascending turn order.
    ///
    /// # Errors
    /// Returns an error if the thread history cannot be retrieved.
    async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>>;

    /// Retrieve all event envelopes for the given thread across every stored turn.
    ///
    /// # Errors
    /// Returns an error if the thread history cannot be retrieved.
    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<AgentEventEnvelope>> {
        let turns = self.get_turns(thread_id).await?;
        Ok(turns
            .into_iter()
            .flat_map(|turn| turn.events.into_iter())
            .collect())
    }

    /// Count the stored events for `thread_id` without materializing them.
    ///
    /// The default falls back to the length of
    /// [`get_events`](EventStore::get_events) (which clones the whole history);
    /// stores that can answer cheaply should override this. Callers that only
    /// need a baseline count — e.g. to read just the new events after a turn —
    /// should prefer this over `get_events(..).len()`.
    ///
    /// # Errors
    /// Returns an error if the count cannot be retrieved.
    async fn event_count(&self, thread_id: &ThreadId) -> Result<usize> {
        Ok(self.get_events(thread_id).await?.len())
    }

    /// Retrieve event envelopes for `thread_id` from `offset` onward, in overall
    /// append order, skipping the earlier ones.
    ///
    /// Lets incremental readers avoid re-cloning the whole history each call.
    /// The default slices [`get_events`](EventStore::get_events); stores with a
    /// cheaper access path should override.
    ///
    /// # Errors
    /// Returns an error if the events cannot be retrieved.
    async fn get_events_since(
        &self,
        thread_id: &ThreadId,
        offset: usize,
    ) -> Result<Vec<AgentEventEnvelope>> {
        Ok(self
            .get_events(thread_id)
            .await?
            .into_iter()
            .skip(offset)
            .collect())
    }

    /// Clear all events for the given thread.
    ///
    /// # Errors
    /// Returns an error if the thread cannot be cleared.
    async fn clear(&self, thread_id: &ThreadId) -> Result<()>;
}

/// Store for tracking tool executions (idempotency).
///
/// This trait enables write-ahead execution tracking to ensure tool idempotency.
/// The pattern is:
/// 1. Record execution intent BEFORE calling the tool (`record_execution`)
/// 2. Update with result AFTER completion (`update_execution`)
/// 3. On retry, check if execution exists and return cached result
#[async_trait]
pub trait ToolExecutionStore: Send + Sync {
    /// Get an execution by `tool_call_id`.
    ///
    /// # Errors
    /// Returns an error if the execution cannot be retrieved.
    async fn get_execution(&self, tool_call_id: &str) -> Result<Option<ToolExecution>>;

    /// Record a new execution (write-ahead, before calling tool).
    ///
    /// # Errors
    /// Returns an error if the execution cannot be recorded.
    async fn record_execution(&self, execution: ToolExecution) -> Result<()>;

    /// Update an existing execution (after completion or to set `operation_id`).
    ///
    /// # Errors
    /// Returns an error if the execution cannot be updated.
    async fn update_execution(&self, execution: ToolExecution) -> Result<()>;

    /// Get execution by `operation_id` (for async tool resume).
    ///
    /// # Errors
    /// Returns an error if the execution cannot be retrieved.
    async fn get_execution_by_operation_id(
        &self,
        operation_id: &str,
    ) -> Result<Option<ToolExecution>>;
}

#[derive(Default)]
struct InMemoryStoreInner {
    messages: RwLock<HashMap<String, Vec<llm::Message>>>,
    states: RwLock<HashMap<String, AgentState>>,
}

/// In-memory implementation of `MessageStore` and `StateStore`.
/// Useful for testing and simple use cases.
///
/// Cloning shares the same underlying message/state maps (mirroring
/// [`InMemoryEventStore`]'s shared-journal semantics). This matters because the
/// agent builder takes its stores **by value**: hand the builder a clone and
/// keep the original, and the kept handle still observes everything the agent
/// records. Without shared handles, history written through the builder's copy
/// would be permanently unreachable to the caller.
#[derive(Clone, Default)]
pub struct InMemoryStore {
    inner: Arc<InMemoryStoreInner>,
}

impl InMemoryStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct InMemoryEventStoreInner {
    turns: AsyncRwLock<HashMap<String, BTreeMap<usize, StoredTurnEvents>>>,
}

/// In-memory implementation of [`EventStore`].
///
/// Cloning this type shares the same underlying event journal.
#[derive(Clone, Default)]
pub struct InMemoryEventStore {
    inner: Arc<InMemoryEventStoreInner>,
}

impl InMemoryEventStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    async fn update_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        update: impl FnOnce(&mut StoredTurnEvents) -> Result<()>,
    ) -> Result<()> {
        let mut turns = self.inner.turns.write().await;
        let stored_turn = turns
            .entry(thread_id.0.clone())
            .or_default()
            .entry(turn)
            .or_insert_with(|| StoredTurnEvents {
                turn,
                events: Vec::new(),
                finished: false,
            });
        let result = update(stored_turn);
        drop(turns);
        result
    }
}

#[async_trait]
impl MessageStore for InMemoryStore {
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()> {
        self.inner
            .messages
            .write()
            .ok()
            .context("lock poisoned")?
            .entry(thread_id.0.clone())
            .or_default()
            .push(message);
        Ok(())
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        let messages = self.inner.messages.read().ok().context("lock poisoned")?;
        Ok(messages.get(&thread_id.0).cloned().unwrap_or_default())
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner
            .messages
            .write()
            .ok()
            .context("lock poisoned")?
            .remove(&thread_id.0);
        Ok(())
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
    ) -> Result<()> {
        self.inner
            .messages
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(thread_id.0.clone(), messages);
        Ok(())
    }
}

#[async_trait]
impl StateStore for InMemoryStore {
    async fn save(&self, state: &AgentState) -> Result<()> {
        self.inner
            .states
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(state.thread_id.0.clone(), state.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        let states = self.inner.states.read().ok().context("lock poisoned")?;
        Ok(states.get(&thread_id.0).cloned())
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner
            .states
            .write()
            .ok()
            .context("lock poisoned")?
            .remove(&thread_id.0);
        Ok(())
    }
}

// Blanket impls so a shared `Arc<Store>` is itself a `MessageStore` /
// `StateStore`. This lets callers keep a readable handle after handing the
// store to the agent builder (which takes stores by value), without forcing
// every store type to be `Clone`.
#[async_trait]
impl<T: MessageStore + ?Sized> MessageStore for Arc<T> {
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()> {
        (**self).append(thread_id, message).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        (**self).get_history(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        (**self).clear(thread_id).await
    }

    async fn count(&self, thread_id: &ThreadId) -> Result<usize> {
        (**self).count(thread_id).await
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
    ) -> Result<()> {
        (**self).replace_history(thread_id, messages).await
    }
}

#[async_trait]
impl<T: StateStore + ?Sized> StateStore for Arc<T> {
    async fn save(&self, state: &AgentState) -> Result<()> {
        (**self).save(state).await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        (**self).load(thread_id).await
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        (**self).delete(thread_id).await
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()> {
        self.update_turn(thread_id, turn, |stored_turn| {
            anyhow::ensure!(
                !stored_turn.finished,
                "cannot append to finished turn {turn}"
            );
            stored_turn.events.push(envelope);
            Ok(())
        })
        .await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
        self.update_turn(thread_id, turn, |stored_turn| {
            anyhow::ensure!(!stored_turn.finished, "turn {turn} is already finished");
            stored_turn.finished = true;
            Ok(())
        })
        .await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> Result<Option<StoredTurnEvents>> {
        let turns = self.inner.turns.read().await;
        Ok(turns
            .get(&thread_id.0)
            .and_then(|thread_turns| thread_turns.get(&turn).cloned()))
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
        let turns = self.inner.turns.read().await;
        Ok(turns
            .get(&thread_id.0)
            .map(|thread_turns| thread_turns.values().cloned().collect())
            .unwrap_or_default())
    }

    async fn event_count(&self, thread_id: &ThreadId) -> Result<usize> {
        // Sum the per-turn lengths under the read lock — no envelope is cloned.
        let turns = self.inner.turns.read().await;
        Ok(turns.get(&thread_id.0).map_or(0, |thread_turns| {
            thread_turns.values().map(|turn| turn.events.len()).sum()
        }))
    }

    async fn get_events_since(
        &self,
        thread_id: &ThreadId,
        offset: usize,
    ) -> Result<Vec<AgentEventEnvelope>> {
        // Only the requested tail is cloned, not the whole history.
        let turns = self.inner.turns.read().await;
        Ok(turns
            .get(&thread_id.0)
            .map(|thread_turns| {
                thread_turns
                    .values()
                    .flat_map(|turn| turn.events.iter())
                    .skip(offset)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        {
            let mut turns = self.inner.turns.write().await;
            turns.remove(&thread_id.0);
        }
        Ok(())
    }
}

/// An [`EventStore`] decorator that invokes a callback on every appended
/// envelope, then delegates all storage to an inner store.
///
/// This is the reusable, blessed way to "stream to stdout" (or to any live
/// observer) with the SDK: the agent loop writes every [`AgentEventEnvelope`]
/// through the configured event store, so wrapping a store lets you watch
/// events as they happen — printing `TextDelta`s, forwarding to a UI channel —
/// without hand-rolling the full five-method [`EventStore`] surface or wiring an
/// in-process channel. The callback runs before the inner store records the
/// envelope.
///
/// # Example
///
/// ```
/// use agent_sdk_tools::stores::{InMemoryEventStore, ObservingEventStore};
/// use agent_sdk_foundation::events::AgentEvent;
///
/// let _store = ObservingEventStore::new(InMemoryEventStore::new(), |envelope| {
///     if let AgentEvent::TextDelta { delta, .. } = &envelope.event {
///         print!("{delta}");
///     }
/// });
/// ```
pub struct ObservingEventStore<S, F> {
    inner: S,
    observer: F,
}

impl<S, F> ObservingEventStore<S, F>
where
    S: EventStore,
    F: Fn(&AgentEventEnvelope) + Send + Sync,
{
    /// Wrap `inner`, calling `observer` on every appended envelope before it is
    /// persisted.
    #[must_use]
    pub const fn new(inner: S, observer: F) -> Self {
        Self { inner, observer }
    }

    /// Borrow the wrapped inner store (e.g. to read back persisted history).
    #[must_use]
    pub const fn inner(&self) -> &S {
        &self.inner
    }
}

#[async_trait]
impl<S, F> EventStore for ObservingEventStore<S, F>
where
    S: EventStore,
    F: Fn(&AgentEventEnvelope) + Send + Sync,
{
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()> {
        (self.observer)(&envelope);
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
        self.inner.finish_turn(thread_id, turn).await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> Result<Option<StoredTurnEvents>> {
        self.inner.get_turn(thread_id, turn).await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
        self.inner.get_turns(thread_id).await
    }

    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<AgentEventEnvelope>> {
        self.inner.get_events(thread_id).await
    }

    async fn event_count(&self, thread_id: &ThreadId) -> Result<usize> {
        self.inner.event_count(thread_id).await
    }

    async fn get_events_since(
        &self,
        thread_id: &ThreadId,
        offset: usize,
    ) -> Result<Vec<AgentEventEnvelope>> {
        self.inner.get_events_since(thread_id, offset).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner.clear(thread_id).await
    }
}

/// In-memory implementation of `ToolExecutionStore`.
///
/// Useful for testing and simple use cases where durability is not required.
/// For production, implement a custom store backed by a database.
#[derive(Default)]
pub struct InMemoryExecutionStore {
    /// Executions indexed by `tool_call_id`
    executions: RwLock<HashMap<String, ToolExecution>>,
    /// Index from `operation_id` to `tool_call_id` for async tool lookup
    operation_index: RwLock<HashMap<String, String>>,
}

impl InMemoryExecutionStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ToolExecutionStore for InMemoryExecutionStore {
    async fn get_execution(&self, tool_call_id: &str) -> Result<Option<ToolExecution>> {
        let executions = self.executions.read().ok().context("lock poisoned")?;
        Ok(executions.get(tool_call_id).cloned())
    }

    async fn record_execution(&self, execution: ToolExecution) -> Result<()> {
        let tool_call_id = execution.tool_call_id.clone();
        let operation_id = execution.operation_id.clone();

        // Hold the executions write lock for the whole insert. Readers acquire
        // executions first (the global executions -> operation_index lock
        // order), so they cannot observe a half-written record. Indexing the
        // operation_id here (not only in `update_execution`) means a write-ahead
        // record is resolvable by `get_execution_by_operation_id` immediately.
        let mut executions = self.executions.write().ok().context("lock poisoned")?;
        if let Some(op_id) = operation_id {
            self.operation_index
                .write()
                .ok()
                .context("lock poisoned")?
                .insert(op_id, tool_call_id.clone());
        }
        executions.insert(tool_call_id, execution);
        drop(executions);
        Ok(())
    }

    async fn update_execution(&self, execution: ToolExecution) -> Result<()> {
        let tool_call_id = execution.tool_call_id.clone();
        let new_operation_id = execution.operation_id.clone();

        // Hold the executions write lock across the whole update so a concurrent
        // reader (which gates on executions first) can never observe the new
        // index entry against a stale execution.
        let mut executions = self.executions.write().ok().context("lock poisoned")?;

        // Drop a superseded operation_id index entry when the id changes, so a
        // stale id stops resolving instead of pointing forever at this call.
        let stale_op_id = executions
            .get(&tool_call_id)
            .and_then(|prev| prev.operation_id.clone())
            .filter(|prev| Some(prev) != new_operation_id.as_ref());
        if stale_op_id.is_some() || new_operation_id.is_some() {
            let mut op_index = self.operation_index.write().ok().context("lock poisoned")?;
            if let Some(stale) = stale_op_id {
                op_index.remove(&stale);
            }
            if let Some(op_id) = new_operation_id {
                op_index.insert(op_id, tool_call_id.clone());
            }
        }
        executions.insert(tool_call_id, execution);
        drop(executions);
        Ok(())
    }

    async fn get_execution_by_operation_id(
        &self,
        operation_id: &str,
    ) -> Result<Option<ToolExecution>> {
        // Acquire executions first (the global executions -> operation_index
        // lock order) and hold it while resolving the id, so this reader can
        // neither deadlock against nor observe a partial write from a concurrent
        // record/update.
        let executions = self.executions.read().ok().context("lock poisoned")?;
        let tool_call_id = {
            let op_index = self.operation_index.read().ok().context("lock poisoned")?;
            op_index.get(operation_id).cloned()
        };
        let Some(tool_call_id) = tool_call_id else {
            return Ok(None);
        };
        Ok(executions.get(&tool_call_id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
    use agent_sdk_foundation::llm::Message;
    use agent_sdk_foundation::types::ToolResult;

    #[tokio::test]
    async fn test_in_memory_message_store() -> Result<()> {
        let store = InMemoryStore::new();
        let thread_id = ThreadId::new();

        // Initially empty
        let history = store.get_history(&thread_id).await?;
        assert!(history.is_empty());

        // Add messages
        store.append(&thread_id, Message::user("Hello")).await?;
        store
            .append(&thread_id, Message::assistant("Hi there!"))
            .await?;

        // Retrieve messages
        let history = store.get_history(&thread_id).await?;
        assert_eq!(history.len(), 2);

        // Count
        let count = store.count(&thread_id).await?;
        assert_eq!(count, 2);

        // Clear
        store.clear(&thread_id).await?;
        let history = store.get_history(&thread_id).await?;
        assert!(history.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_replace_history() -> Result<()> {
        let store = InMemoryStore::new();
        let thread_id = ThreadId::new();

        // Add some messages
        store.append(&thread_id, Message::user("Hello")).await?;
        store
            .append(&thread_id, Message::assistant("Hi there!"))
            .await?;
        store
            .append(&thread_id, Message::user("How are you?"))
            .await?;

        // Verify original messages
        let history = store.get_history(&thread_id).await?;
        assert_eq!(history.len(), 3);

        // Replace with compacted history
        let new_history = vec![
            Message::user("[Summary] Previous conversation about greetings"),
            Message::assistant("I understand the context. Continuing..."),
        ];
        store.replace_history(&thread_id, new_history).await?;

        // Verify replaced history
        let history = store.get_history(&thread_id).await?;
        assert_eq!(history.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_state_store() -> Result<()> {
        let store = InMemoryStore::new();
        let thread_id = ThreadId::new();

        // Initially none
        let state = store.load(&thread_id).await?;
        assert!(state.is_none());

        // Save state
        let state = AgentState::new(thread_id.clone());
        store.save(&state).await?;

        // Load state
        let loaded = store.load(&thread_id).await?;
        assert!(loaded.is_some());
        if let Some(loaded_state) = loaded {
            assert_eq!(loaded_state.thread_id, thread_id);
        }

        // Delete state
        store.delete(&thread_id).await?;
        let state = store.load(&thread_id).await?;
        assert!(state.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_event_store_tracks_turns_and_finish_barrier() -> Result<()> {
        let store = InMemoryEventStore::new();
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("msg_1", "hello"), &seq),
            )
            .await?;
        store
            .append(
                &thread_id,
                2,
                AgentEventEnvelope::wrap(AgentEvent::text("msg_2", "world"), &seq),
            )
            .await?;

        let turn_1 = store
            .get_turn(&thread_id, 1)
            .await?
            .context("missing turn 1")?;
        assert_eq!(turn_1.turn, 1);
        assert_eq!(turn_1.events.len(), 1);
        assert!(!turn_1.finished);

        store.finish_turn(&thread_id, 1).await?;
        store.finish_turn(&thread_id, 2).await?;

        let turn_1 = store
            .get_turn(&thread_id, 1)
            .await?
            .context("missing finished turn 1")?;
        let turn_2 = store
            .get_turn(&thread_id, 2)
            .await?
            .context("missing finished turn 2")?;
        assert!(turn_1.finished);
        assert!(turn_2.finished);

        let turns = store.get_turns(&thread_id).await?;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].turn, 1);
        assert_eq!(turns[1].turn, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_event_store_finish_turn_without_events_creates_finished_turn()
    -> Result<()> {
        let store = InMemoryEventStore::new();
        let thread_id = ThreadId::new();

        store.finish_turn(&thread_id, 3).await?;

        let turn = store
            .get_turn(&thread_id, 3)
            .await?
            .context("missing empty finished turn")?;
        assert_eq!(turn.turn, 3);
        assert!(turn.events.is_empty());
        assert!(turn.finished);

        store.clear(&thread_id).await?;
        assert!(store.get_turns(&thread_id).await?.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_event_store_rejects_append_after_finish() -> Result<()> {
        let store = InMemoryEventStore::new();
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        store.finish_turn(&thread_id, 1).await?;

        let error = store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("msg_1", "late"), &seq),
            )
            .await
            .expect_err("append after finish should fail");

        assert!(error.to_string().contains("cannot append to finished turn"));
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_event_store_rejects_duplicate_finish() -> Result<()> {
        let store = InMemoryEventStore::new();
        let thread_id = ThreadId::new();

        store.finish_turn(&thread_id, 1).await?;

        let error = store
            .finish_turn(&thread_id, 1)
            .await
            .expect_err("duplicate finish should fail");

        assert!(error.to_string().contains("already finished"));
        Ok(())
    }

    #[tokio::test]
    async fn test_execution_store_basic_operations() -> Result<()> {
        let store = InMemoryExecutionStore::new();
        let thread_id = ThreadId::new();

        // Initially none
        let execution = store.get_execution("tool_call_123").await?;
        assert!(execution.is_none());

        // Record execution
        let execution = ToolExecution::new_in_flight(
            "tool_call_123",
            thread_id.clone(),
            "my_tool",
            "My Tool",
            serde_json::json!({"param": "value"}),
            time::OffsetDateTime::now_utc(),
        );
        store.record_execution(execution).await?;

        // Retrieve execution
        let loaded = store.get_execution("tool_call_123").await?;
        assert!(loaded.is_some());
        let loaded = loaded.expect("execution should exist");
        assert_eq!(loaded.tool_call_id, "tool_call_123");
        assert_eq!(loaded.tool_name, "my_tool");
        assert!(loaded.is_in_flight());

        Ok(())
    }

    #[tokio::test]
    async fn test_execution_store_complete_execution() -> Result<()> {
        let store = InMemoryExecutionStore::new();
        let thread_id = ThreadId::new();

        // Record in-flight execution
        let mut execution = ToolExecution::new_in_flight(
            "tool_call_456",
            thread_id.clone(),
            "my_tool",
            "My Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        store.record_execution(execution.clone()).await?;

        // Complete the execution
        execution.complete(ToolResult::success("Done!"));
        store.update_execution(execution).await?;

        // Verify it's completed
        let loaded = store.get_execution("tool_call_456").await?;
        let loaded = loaded.expect("execution should exist");
        assert!(loaded.is_completed());
        assert!(loaded.result.is_some());
        assert!(loaded.result.as_ref().is_some_and(|r| r.success));

        Ok(())
    }

    #[tokio::test]
    async fn test_execution_store_operation_id_lookup() -> Result<()> {
        let store = InMemoryExecutionStore::new();
        let thread_id = ThreadId::new();

        // Record execution with operation_id
        let mut execution = ToolExecution::new_in_flight(
            "tool_call_789",
            thread_id.clone(),
            "async_tool",
            "Async Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        execution.set_operation_id("op_abc123");
        store.record_execution(execution.clone()).await?;
        store.update_execution(execution).await?;

        // Lookup by operation_id
        let loaded = store.get_execution_by_operation_id("op_abc123").await?;
        assert!(loaded.is_some());
        let loaded = loaded.expect("execution should exist");
        assert_eq!(loaded.tool_call_id, "tool_call_789");
        assert_eq!(loaded.operation_id, Some("op_abc123".to_string()));

        // Non-existent operation_id
        let not_found = store.get_execution_by_operation_id("nonexistent").await?;
        assert!(not_found.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn in_memory_store_clone_shares_history() -> Result<()> {
        // A clone handed to the builder shares state with the kept handle, so
        // history written by the agent stays reachable to the caller.
        let store = InMemoryStore::new();
        let handle = store.clone();
        let thread_id = ThreadId::new();

        store.append(&thread_id, Message::user("hello")).await?;

        let history = handle.get_history(&thread_id).await?;
        assert_eq!(
            history.len(),
            1,
            "clone must observe appends via the original"
        );
        Ok(())
    }

    #[tokio::test]
    async fn arc_store_blanket_impls_forward() -> Result<()> {
        let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
        let thread_id = ThreadId::new();

        // `Arc<InMemoryStore>` is itself a `MessageStore` and `StateStore`.
        MessageStore::append(&store, &thread_id, Message::user("hi")).await?;
        assert_eq!(MessageStore::count(&store, &thread_id).await?, 1);

        let state = AgentState::new(thread_id.clone());
        StateStore::save(&store, &state).await?;
        assert!(StateStore::load(&store, &thread_id).await?.is_some());

        // The kept Arc handle still sees everything.
        assert_eq!(store.get_history(&thread_id).await?.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn event_count_and_get_events_since_are_incremental() -> Result<()> {
        let store = InMemoryEventStore::new();
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        assert_eq!(store.event_count(&thread_id).await?, 0);

        for (turn, (id, text)) in [(1, ("m1", "a")), (1, ("m2", "b")), (2, ("m3", "c"))] {
            store
                .append(
                    &thread_id,
                    turn,
                    AgentEventEnvelope::wrap(AgentEvent::text(id, text), &seq),
                )
                .await?;
        }

        assert_eq!(store.event_count(&thread_id).await?, 3);

        let tail = store.get_events_since(&thread_id, 1).await?;
        assert_eq!(tail.len(), 2, "should skip the first event");
        // Consistent with the full read.
        let all = store.get_events(&thread_id).await?;
        assert_eq!(all.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn record_execution_indexes_operation_id_immediately() -> Result<()> {
        let store = InMemoryExecutionStore::new();
        let thread_id = ThreadId::new();

        let mut execution = ToolExecution::new_in_flight(
            "call_1",
            thread_id,
            "async_tool",
            "Async Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        execution.set_operation_id("op_1");
        // Write-ahead record only — no `update_execution` call.
        store.record_execution(execution).await?;

        let loaded = store.get_execution_by_operation_id("op_1").await?;
        assert_eq!(
            loaded
                .context("write-ahead operation_id must resolve")?
                .tool_call_id,
            "call_1"
        );
        Ok(())
    }

    #[tokio::test]
    async fn update_execution_removes_stale_operation_id() -> Result<()> {
        let store = InMemoryExecutionStore::new();
        let thread_id = ThreadId::new();

        let mut execution = ToolExecution::new_in_flight(
            "call_2",
            thread_id,
            "async_tool",
            "Async Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        execution.set_operation_id("op_old");
        store.record_execution(execution.clone()).await?;

        // Re-point the execution at a new operation id.
        execution.set_operation_id("op_new");
        store.update_execution(execution).await?;

        assert!(
            store
                .get_execution_by_operation_id("op_old")
                .await?
                .is_none(),
            "superseded operation_id must stop resolving"
        );
        let loaded = store.get_execution_by_operation_id("op_new").await?;
        assert_eq!(
            loaded
                .context("new operation_id must resolve")?
                .tool_call_id,
            "call_2"
        );
        Ok(())
    }

    #[tokio::test]
    async fn observing_event_store_invokes_callback_and_delegates() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let seen = Arc::new(AtomicUsize::new(0));
        let seen_for_cb = Arc::clone(&seen);
        let store = ObservingEventStore::new(InMemoryEventStore::new(), move |_envelope| {
            seen_for_cb.fetch_add(1, Ordering::SeqCst);
        });
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("m1", "hi"), &seq),
            )
            .await?;
        store
            .append(
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("m2", "yo"), &seq),
            )
            .await?;

        assert_eq!(seen.load(Ordering::SeqCst), 2, "observer runs per append");
        // Delegation: the inner store actually persisted both events.
        assert_eq!(store.get_events(&thread_id).await?.len(), 2);
        assert_eq!(store.inner().get_events(&thread_id).await?.len(), 2);
        Ok(())
    }
}
