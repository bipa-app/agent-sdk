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

use agent_sdk_core::events::AgentEventEnvelope;
use agent_sdk_core::llm;
use agent_sdk_core::types::{AgentState, ThreadId, ToolExecution};
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

/// In-memory implementation of `MessageStore` and `StateStore`.
/// Useful for testing and simple use cases.
#[derive(Default)]
pub struct InMemoryStore {
    messages: RwLock<HashMap<String, Vec<llm::Message>>>,
    states: RwLock<HashMap<String, AgentState>>,
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
}

#[async_trait]
impl MessageStore for InMemoryStore {
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()> {
        self.messages
            .write()
            .ok()
            .context("lock poisoned")?
            .entry(thread_id.0.clone())
            .or_default()
            .push(message);
        Ok(())
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        let messages = self.messages.read().ok().context("lock poisoned")?;
        Ok(messages.get(&thread_id.0).cloned().unwrap_or_default())
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.messages
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
        self.messages
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
        self.states
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(state.thread_id.0.clone(), state.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        let states = self.states.read().ok().context("lock poisoned")?;
        Ok(states.get(&thread_id.0).cloned())
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        self.states
            .write()
            .ok()
            .context("lock poisoned")?
            .remove(&thread_id.0);
        Ok(())
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
        self.inner
            .turns
            .write()
            .await
            .entry(thread_id.0.clone())
            .or_default()
            .entry(turn)
            .or_insert_with(|| StoredTurnEvents {
                turn,
                events: Vec::new(),
                finished: false,
            })
            .events
            .push(envelope);
        Ok(())
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
        self.inner
            .turns
            .write()
            .await
            .entry(thread_id.0.clone())
            .or_default()
            .entry(turn)
            .or_insert_with(|| StoredTurnEvents {
                turn,
                events: Vec::new(),
                finished: false,
            })
            .finished = true;
        Ok(())
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

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        {
            let mut turns = self.inner.turns.write().await;
            turns.remove(&thread_id.0);
        }
        Ok(())
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
        self.executions
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(tool_call_id, execution);
        Ok(())
    }

    async fn update_execution(&self, execution: ToolExecution) -> Result<()> {
        let tool_call_id = execution.tool_call_id.clone();

        // Update operation_id index if present
        if let Some(ref op_id) = execution.operation_id {
            self.operation_index
                .write()
                .ok()
                .context("lock poisoned")?
                .insert(op_id.clone(), tool_call_id.clone());
        }

        self.executions
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(tool_call_id, execution);
        Ok(())
    }

    async fn get_execution_by_operation_id(
        &self,
        operation_id: &str,
    ) -> Result<Option<ToolExecution>> {
        // Get tool_call_id and drop lock before acquiring another
        let tool_call_id = {
            let op_index = self.operation_index.read().ok().context("lock poisoned")?;
            op_index.get(operation_id).cloned()
        };

        let Some(tool_call_id) = tool_call_id else {
            return Ok(None);
        };

        let executions = self.executions.read().ok().context("lock poisoned")?;
        Ok(executions.get(&tool_call_id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::llm::Message;
    use agent_sdk_core::types::ToolResult;

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
}
