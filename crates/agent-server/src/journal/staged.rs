//! Staged message and state adapters for buffered turn execution.
//!
//! During a turn the root worker must **not** write durable projections
//! — Phase 3 invariants require that message and state mutations only
//! become visible at commit time through
//! [`super::commit::commit_completed_turn`].
//!
//! This module provides two lightweight adapters that implement the SDK's
//! [`MessageStore`] and [`StateStore`] traits while keeping all mutations
//! in memory:
//!
//! - [`StagedMessageStore`] — seeded from checkpoint messages, buffers
//!   appends and replace-history calls until commit.
//! - [`StagedStateStore`] — seeded from the checkpoint's agent-state
//!   snapshot (deserialized into [`AgentState`]), buffers saves until
//!   commit.
//!
//! Both adapters expose a `drain_*` method that moves the buffered data
//! out for the commit path to consume. After draining the adapter is
//! empty and must not be reused.
//!
//! # Seeding
//!
//! [`StagedStores::from_recovery_view`] constructs both adapters from a
//! [`ThreadRecoveryView`]:
//!
//! - **Existing thread** (has checkpoint): messages and agent-state come
//!   from the latest completed checkpoint.
//! - **Fresh thread** (no checkpoint): messages start empty and
//!   agent-state is a fresh [`AgentState`] for the thread.
//!
//! # Design properties
//!
//! 1. **No durable mid-turn writes** — the adapters never touch a
//!    durable store. Reads return the seed + buffered mutations; writes
//!    accumulate in memory only.
//! 2. **Crash safety** — if the worker crashes before commit, the
//!    staged data is lost and the thread resumes from the last committed
//!    checkpoint on the next attempt.
//! 3. **Single-turn scope** — each adapter is constructed for exactly
//!    one turn attempt. There is no cross-turn reuse.

use agent_sdk_core::llm;
use agent_sdk_core::types::{AgentState, ThreadId};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::RwLock;

use agent_sdk_tools::stores::{MessageStore, StateStore};

use super::thread_recover::ThreadRecoveryView;

// ─────────────────────────────────────────────────────────────────────
// StagedMessageStore
// ─────────────────────────────────────────────────────────────────────

/// In-memory [`MessageStore`] that buffers all mutations during a turn.
///
/// Seeded with the committed message history from the latest checkpoint
/// (or empty for a fresh thread). Appends and replacements accumulate
/// in memory and never touch durable storage.
///
/// The commit path calls [`Self::drain_messages`] to consume the
/// buffered history.
pub struct StagedMessageStore {
    thread_id: ThreadId,
    messages: RwLock<Vec<llm::Message>>,
    /// Number of seed messages from the checkpoint. `drain_messages`
    /// returns only messages appended *after* the seed so the commit
    /// path can safely append the delta to the durable projection.
    seed_len: usize,
}

impl StagedMessageStore {
    /// Create a new staged store seeded with the given messages.
    #[must_use]
    pub const fn new(thread_id: ThreadId, seed_messages: Vec<llm::Message>) -> Self {
        let seed_len = seed_messages.len();
        Self {
            thread_id,
            messages: RwLock::new(seed_messages),
            seed_len,
        }
    }

    /// Drain only the **newly appended** messages for the commit path.
    ///
    /// The returned vec excludes the seed messages that were provided
    /// at construction time, so the caller can safely pass the result
    /// to [`super::commit::commit_completed_turn`] which *appends* to
    /// the durable projection.
    ///
    /// After this call the internal buffer is empty and the store
    /// should not be reused.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn drain_messages(&self) -> Result<Vec<llm::Message>> {
        let all = std::mem::take(&mut *self.messages.write().ok().context("lock poisoned")?);
        Ok(all.into_iter().skip(self.seed_len).collect())
    }

    /// Snapshot the current buffered messages without consuming them.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn snapshot_messages(&self) -> Result<Vec<llm::Message>> {
        let guard = self.messages.read().ok().context("lock poisoned")?;
        Ok(guard.clone())
    }
}

#[async_trait]
impl MessageStore for StagedMessageStore {
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged message store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        self.messages
            .write()
            .ok()
            .context("lock poisoned")?
            .push(message);
        Ok(())
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged message store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        let guard = self.messages.read().ok().context("lock poisoned")?;
        Ok(guard.clone())
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged message store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        self.messages.write().ok().context("lock poisoned")?.clear();
        Ok(())
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
    ) -> Result<()> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged message store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        *self.messages.write().ok().context("lock poisoned")? = messages;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// StagedStateStore
// ─────────────────────────────────────────────────────────────────────

/// In-memory [`StateStore`] that buffers all mutations during a turn.
///
/// Seeded with the agent-state snapshot from the latest checkpoint
/// (deserialized into [`AgentState`]) or a fresh state for new threads.
/// Saves accumulate in memory and never touch durable storage.
///
/// The commit path calls [`Self::drain_state`] to consume the buffered
/// state for the checkpoint's `agent_state_snapshot` field.
pub struct StagedStateStore {
    thread_id: ThreadId,
    state: RwLock<Option<AgentState>>,
}

impl StagedStateStore {
    /// Create a new staged store seeded with the given agent state.
    #[must_use]
    pub const fn new(thread_id: ThreadId, seed_state: Option<AgentState>) -> Self {
        Self {
            thread_id,
            state: RwLock::new(seed_state),
        }
    }

    /// Drain the buffered agent state for the commit path.
    ///
    /// Returns the latest saved state (or the seed if no saves
    /// occurred). After this call the internal buffer is `None` and
    /// the store should not be reused.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn drain_state(&self) -> Result<Option<AgentState>> {
        let mut guard = self.state.write().ok().context("lock poisoned")?;
        Ok(guard.take())
    }

    /// Snapshot the current buffered state without consuming it.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn snapshot_state(&self) -> Result<Option<AgentState>> {
        let guard = self.state.read().ok().context("lock poisoned")?;
        Ok(guard.clone())
    }
}

#[async_trait]
impl StateStore for StagedStateStore {
    async fn save(&self, state: &AgentState) -> Result<()> {
        anyhow::ensure!(
            state.thread_id == self.thread_id,
            "staged state store bound to thread {}, got {}",
            self.thread_id,
            state.thread_id,
        );
        *self.state.write().ok().context("lock poisoned")? = Some(state.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged state store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        let guard = self.state.read().ok().context("lock poisoned")?;
        Ok(guard.clone())
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        anyhow::ensure!(
            thread_id == &self.thread_id,
            "staged state store bound to thread {}, got {}",
            self.thread_id,
            thread_id,
        );
        *self.state.write().ok().context("lock poisoned")? = None;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// StagedStores — convenience bundle
// ─────────────────────────────────────────────────────────────────────

/// Paired staged message and state stores for a single turn attempt.
///
/// Constructed via [`Self::from_recovery_view`] which seeds both
/// adapters from the [`ThreadRecoveryView`] produced by Phase 3.5's
/// [`super::thread_recover::recover_thread`].
pub struct StagedStores {
    /// Staged message store seeded from checkpoint history.
    pub messages: StagedMessageStore,
    /// Staged state store seeded from checkpoint agent-state snapshot.
    pub state: StagedStateStore,
}

impl StagedStores {
    /// Construct staged stores seeded from a thread recovery view.
    ///
    /// # Seeding rules
    ///
    /// - **Messages**: seeded from `view.messages` (the committed
    ///   history from the latest checkpoint, or empty for a fresh
    ///   thread).
    /// - **Agent state**: if the view has a non-null
    ///   `agent_state_snapshot`, it is deserialized into [`AgentState`].
    ///   Otherwise a fresh [`AgentState`] is created for the thread
    ///   (fresh thread path).
    ///
    /// # Errors
    ///
    /// Returns an error if the `agent_state_snapshot` cannot be
    /// deserialized into [`AgentState`].
    pub fn from_recovery_view(view: &ThreadRecoveryView) -> Result<Self> {
        Self::from_recovery_view_with_messages(view, view.messages.clone())
    }

    /// Construct staged stores from a recovery view, but seed messages
    /// only from the latest committed checkpoint.
    ///
    /// This is used when resuming an already-suspended root task. The
    /// task state supplies `suspended_messages` and completed child
    /// results explicitly; including recovery draft messages in the
    /// seed would duplicate the assistant `tool_use` before the
    /// matching `tool_result` is appended.
    ///
    /// # Errors
    ///
    /// Returns an error if the `agent_state_snapshot` cannot be
    /// deserialized into [`AgentState`].
    pub fn from_recovery_view_committed_only(view: &ThreadRecoveryView) -> Result<Self> {
        let messages = view
            .latest_checkpoint
            .as_ref()
            .map_or_else(Vec::new, |checkpoint| checkpoint.messages.clone());
        Self::from_recovery_view_with_messages(view, messages)
    }

    fn from_recovery_view_with_messages(
        view: &ThreadRecoveryView,
        messages: Vec<llm::Message>,
    ) -> Result<Self> {
        let thread_id = view.thread.thread_id.clone();

        // Seed agent state from the checkpoint snapshot or create a
        // fresh one for new threads.
        let seed_state = if view.agent_state_snapshot.is_null() {
            AgentState {
                thread_id: thread_id.clone(),
                turn_count: 0,
                total_usage: agent_sdk_core::TokenUsage::default(),
                metadata: std::collections::HashMap::new(),
                created_at: view.thread.created_at,
            }
        } else {
            serde_json::from_value(view.agent_state_snapshot.clone())
                .context("deserialize agent_state_snapshot from checkpoint")?
        };

        Ok(Self {
            messages: StagedMessageStore::new(thread_id.clone(), messages),
            state: StagedStateStore::new(thread_id, Some(seed_state)),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::TokenUsage;

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-staged-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-staged-b")
    }

    fn sample_messages() -> Vec<llm::Message> {
        vec![
            llm::Message::user("hello"),
            llm::Message::assistant("hi there"),
        ]
    }

    // ── StagedMessageStore ──────────────────────────────────────

    #[tokio::test]
    async fn staged_messages_seeded_and_appendable() -> Result<()> {
        let store = StagedMessageStore::new(thread_a(), sample_messages());

        // Seed messages visible via get_history.
        let history = store.get_history(&thread_a()).await?;
        assert_eq!(history.len(), 2);

        // Append buffers in memory.
        store
            .append(&thread_a(), llm::Message::user("follow-up"))
            .await?;
        let history = store.get_history(&thread_a()).await?;
        assert_eq!(history.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn staged_messages_replace_history() -> Result<()> {
        let store = StagedMessageStore::new(thread_a(), sample_messages());

        let replacement = vec![llm::Message::user("compacted summary")];
        store.replace_history(&thread_a(), replacement).await?;

        let history = store.get_history(&thread_a()).await?;
        assert_eq!(history.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn staged_messages_drain_consumes() -> Result<()> {
        let store = StagedMessageStore::new(thread_a(), sample_messages());
        store
            .append(&thread_a(), llm::Message::user("extra"))
            .await?;

        // drain_messages returns only the delta (appended after seed).
        let drained = store.drain_messages()?;
        assert_eq!(drained.len(), 1);

        // After drain, store is empty.
        let history = store.get_history(&thread_a()).await?;
        assert!(history.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn staged_messages_rejects_wrong_thread() {
        let store = StagedMessageStore::new(thread_a(), vec![]);
        let err = store
            .append(&thread_b(), llm::Message::user("wrong"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("bound to thread"));
    }

    #[tokio::test]
    async fn staged_messages_clear() -> Result<()> {
        let store = StagedMessageStore::new(thread_a(), sample_messages());
        store.clear(&thread_a()).await?;
        let history = store.get_history(&thread_a()).await?;
        assert!(history.is_empty());
        Ok(())
    }

    // ── StagedStateStore ────────────────────────────────────────

    #[tokio::test]
    async fn staged_state_seeded_and_saveable() -> Result<()> {
        let seed = AgentState::new(thread_a());
        let store = StagedStateStore::new(thread_a(), Some(seed.clone()));

        // Seed visible via load.
        let loaded = store.load(&thread_a()).await?;
        assert!(loaded.is_some());
        assert_eq!(loaded.as_ref().map(|s| &s.thread_id), Some(&thread_a()));

        // Save overwrites in memory.
        let mut updated = seed;
        updated.turn_count = 5;
        updated.total_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        store.save(&updated).await?;

        let loaded = store.load(&thread_a()).await?;
        let loaded = loaded.context("should be Some")?;
        assert_eq!(loaded.turn_count, 5);
        assert_eq!(loaded.total_usage.input_tokens, 100);

        Ok(())
    }

    #[tokio::test]
    async fn staged_state_drain_consumes() -> Result<()> {
        let seed = AgentState::new(thread_a());
        let store = StagedStateStore::new(thread_a(), Some(seed));

        let drained = store.drain_state()?;
        assert!(drained.is_some());

        // After drain, load returns None.
        let loaded = store.load(&thread_a()).await?;
        assert!(loaded.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn staged_state_rejects_wrong_thread() {
        let store = StagedStateStore::new(thread_a(), None);
        let wrong_state = AgentState::new(thread_b());
        let err = store.save(&wrong_state).await.unwrap_err();
        assert!(err.to_string().contains("bound to thread"));
    }

    #[tokio::test]
    async fn staged_state_delete_clears_buffer() -> Result<()> {
        let seed = AgentState::new(thread_a());
        let store = StagedStateStore::new(thread_a(), Some(seed));
        store.delete(&thread_a()).await?;
        let loaded = store.load(&thread_a()).await?;
        assert!(loaded.is_none());
        Ok(())
    }

    // ── StagedStores from recovery view ─────────────────────────

    #[tokio::test]
    async fn staged_stores_from_fresh_thread_view() -> Result<()> {
        let view = ThreadRecoveryView {
            thread: super::super::thread::Thread::new(thread_a(), time::OffsetDateTime::now_utc()),
            messages: Vec::new(),
            agent_state_snapshot: serde_json::Value::Null,
            latest_checkpoint: None,
            draft_messages: Vec::new(),
            next_turn_number: 1,
        };

        let staged = StagedStores::from_recovery_view(&view)?;

        // Messages start empty.
        let msgs = staged.messages.get_history(&thread_a()).await?;
        assert!(msgs.is_empty());

        // State is a fresh AgentState for the thread.
        let state = staged.state.load(&thread_a()).await?;
        let state = state.context("should be Some")?;
        assert_eq!(state.thread_id, thread_a());
        assert_eq!(state.turn_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn staged_stores_from_checkpoint_view() -> Result<()> {
        let seed_state = AgentState {
            thread_id: thread_a(),
            turn_count: 3,
            total_usage: TokenUsage {
                input_tokens: 500,
                output_tokens: 200,
                ..Default::default()
            },
            metadata: std::collections::HashMap::default(),
            created_at: time::OffsetDateTime::now_utc(),
        };
        let snapshot = serde_json::to_value(&seed_state)?;

        let view = ThreadRecoveryView {
            thread: super::super::thread::Thread::new(thread_a(), time::OffsetDateTime::now_utc()),
            messages: sample_messages(),
            agent_state_snapshot: snapshot,
            latest_checkpoint: None,
            draft_messages: Vec::new(),
            next_turn_number: 4,
        };

        let staged = StagedStores::from_recovery_view(&view)?;

        // Messages seeded from checkpoint.
        let msgs = staged.messages.get_history(&thread_a()).await?;
        assert_eq!(msgs.len(), 2);

        // State deserialized from snapshot.
        let state = staged.state.load(&thread_a()).await?;
        let state = state.context("should be Some")?;
        assert_eq!(state.turn_count, 3);
        assert_eq!(state.total_usage.input_tokens, 500);

        Ok(())
    }

    #[tokio::test]
    async fn committed_only_view_excludes_draft_messages() -> Result<()> {
        let seed_state = AgentState {
            thread_id: thread_a(),
            turn_count: 1,
            total_usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            metadata: std::collections::HashMap::default(),
            created_at: time::OffsetDateTime::now_utc(),
        };
        let snapshot = serde_json::to_value(&seed_state)?;
        let committed = vec![llm::Message::user("committed")];
        let draft = vec![
            llm::Message::user("draft user"),
            llm::Message::assistant_with_tool_use(
                None,
                "call_1",
                "bash",
                serde_json::json!({"command": "pwd"}),
            ),
        ];
        let checkpoint = super::super::checkpoint::Checkpoint::new(
            super::super::checkpoint::NewCheckpointParams {
                thread_id: thread_a(),
                turn_number: 1,
                task_id: super::super::task::AgentTaskId::from_string("task_committed"),
                messages: committed.clone(),
                agent_state_snapshot: snapshot.clone(),
                turn_usage: TokenUsage::default(),
                now: time::OffsetDateTime::now_utc(),
            },
        )?;

        let mut view_messages = committed.clone();
        view_messages.extend(draft.clone());
        let view = ThreadRecoveryView {
            thread: super::super::thread::Thread::new(thread_a(), time::OffsetDateTime::now_utc()),
            messages: view_messages,
            agent_state_snapshot: snapshot,
            latest_checkpoint: Some(checkpoint),
            draft_messages: draft,
            next_turn_number: 2,
        };

        let staged = StagedStores::from_recovery_view_committed_only(&view)?;
        let messages = staged.messages.get_history(&thread_a()).await?;
        assert_eq!(messages.len(), 1);
        assert_eq!(
            serde_json::to_value(messages)?,
            serde_json::to_value(committed)?
        );

        Ok(())
    }

    #[tokio::test]
    async fn staged_stores_mutations_do_not_affect_seed() -> Result<()> {
        let view = ThreadRecoveryView {
            thread: super::super::thread::Thread::new(thread_a(), time::OffsetDateTime::now_utc()),
            messages: sample_messages(),
            agent_state_snapshot: serde_json::Value::Null,
            latest_checkpoint: None,
            draft_messages: Vec::new(),
            next_turn_number: 1,
        };

        let staged = StagedStores::from_recovery_view(&view)?;

        // Mutate the staged stores.
        staged
            .messages
            .append(&thread_a(), llm::Message::user("new"))
            .await?;
        let mut new_state = AgentState::new(thread_a());
        new_state.turn_count = 99;
        staged.state.save(&new_state).await?;

        // The original view is unchanged (since staged stores clone on
        // construction, mutations are isolated).
        assert_eq!(view.messages.len(), 2);
        assert_eq!(view.agent_state_snapshot, serde_json::Value::Null);

        // Staged stores reflect the mutations.
        let msgs = staged.messages.get_history(&thread_a()).await?;
        assert_eq!(msgs.len(), 3);
        let state = staged.state.load(&thread_a()).await?.context("Some")?;
        assert_eq!(state.turn_count, 99);

        Ok(())
    }

    #[tokio::test]
    async fn snapshot_does_not_consume() -> Result<()> {
        let store = StagedMessageStore::new(thread_a(), sample_messages());
        let snap1 = store.snapshot_messages()?;
        let snap2 = store.snapshot_messages()?;
        assert_eq!(snap1.len(), snap2.len());

        let state_store = StagedStateStore::new(thread_a(), Some(AgentState::new(thread_a())));
        let s1 = state_store.snapshot_state()?;
        let s2 = state_store.snapshot_state()?;
        assert!(s1.is_some());
        assert!(s2.is_some());

        Ok(())
    }
}
