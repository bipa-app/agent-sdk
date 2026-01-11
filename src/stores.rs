use crate::llm;
use crate::types::{AgentState, ThreadId};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;

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
}
