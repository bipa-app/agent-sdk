//! Durable-friendly storage trait for [`Checkpoint`] rows.
//!
//! The [`CheckpointStore`] trait is the sole write surface for
//! completed-turn checkpoints. Its key design properties:
//!
//! 1. **Append-only** — checkpoints are inserted via
//!    [`CheckpointStore::commit_checkpoint`] and are never modified or
//!    deleted.
//! 2. **Thread-scoped uniqueness** — at most one checkpoint exists
//!    per `(thread_id, turn_number)`. Duplicate inserts are rejected.
//! 3. **Immutable after creation** — there is no `update()` or
//!    `delete()` because checkpoints are durable snapshots.
//!
//! [`InMemoryCheckpointStore`] is the reference implementation,
//! following the same `Arc<RwLock<Inner>>` pattern as
//! [`super::thread_store::InMemoryThreadStore`].
//!
//! # Entry point summary
//!
//! | Entry point | What it does | Guard |
//! |-------------|-------------|-------|
//! | [`CheckpointStore::commit_checkpoint`] | Inserts a new checkpoint | `(thread_id, turn_number)` unique |
//! | [`CheckpointStore::get`] | Reads a single row by id | — |
//! | [`CheckpointStore::get_by_turn`] | Reads a row by `(thread_id, turn_number)` | — |
//! | [`CheckpointStore::list_by_thread`] | Lists all checkpoints for a thread | Ordered by `turn_number` |

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_core::ThreadId;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tokio::sync::RwLock;

use super::checkpoint::{Checkpoint, CheckpointId, NewCheckpointParams};

/// Storage trait for [`Checkpoint`] rows.
///
/// The trait surface is deliberately narrow: `commit_checkpoint` is
/// the only write path. There is no `update()` or `delete()` because
/// checkpoints are immutable snapshots of committed conversation
/// state.
///
/// Implementations must guarantee that the `(thread_id, turn_number)`
/// uniqueness constraint is enforced, and that `list_by_thread`
/// returns rows ordered by `turn_number` ascending.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Create a new checkpoint for a completed turn.
    ///
    /// The store enforces the `(thread_id, turn_number)` uniqueness
    /// constraint: a second call with the same thread and turn number
    /// returns an error.
    ///
    /// # Errors
    /// - `duplicate checkpoint` if a checkpoint already exists for
    ///   this `(thread_id, turn_number)`.
    /// - [`super::checkpoint::CheckpointSchemaError`] if the params
    ///   fail validation.
    /// - Store-level write errors.
    async fn commit_checkpoint(&self, params: NewCheckpointParams) -> Result<Checkpoint>;

    /// Look up a single checkpoint by id.
    ///
    /// Returns `None` if the id does not exist.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, id: &CheckpointId) -> Result<Option<Checkpoint>>;

    /// Look up a checkpoint by `(thread_id, turn_number)`.
    ///
    /// Returns `None` if no checkpoint exists for this turn.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get_by_turn(
        &self,
        thread_id: &ThreadId,
        turn_number: u32,
    ) -> Result<Option<Checkpoint>>;

    /// List all checkpoints for a thread, ordered by `turn_number`
    /// ascending.
    ///
    /// Returns an empty vec if no checkpoints exist for the thread.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<Checkpoint>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
#[allow(clippy::struct_field_names)]
struct Inner {
    /// Primary key index.
    by_id: HashMap<CheckpointId, Checkpoint>,
    /// Secondary index: thread → checkpoint ids (sorted at query time
    /// by `turn_number`).
    by_thread: HashMap<ThreadId, Vec<CheckpointId>>,
    /// Uniqueness index: `(thread_id, turn_number)` → checkpoint id.
    by_thread_turn: HashMap<(ThreadId, u32), CheckpointId>,
}

/// In-memory reference implementation of [`CheckpointStore`].
///
/// Follows the same `Arc<RwLock<Inner>>` pattern as
/// [`super::thread_store::InMemoryThreadStore`]. Designed for tests
/// and single-process usage.
#[derive(Clone, Default)]
pub struct InMemoryCheckpointStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryCheckpointStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn commit_checkpoint(&self, params: NewCheckpointParams) -> Result<Checkpoint> {
        let checkpoint = Checkpoint::new(params)?;

        let mut inner = self.inner.write().await;
        let key = (checkpoint.thread_id.clone(), checkpoint.turn_number);
        if inner.by_thread_turn.contains_key(&key) {
            return Err(anyhow!(
                "duplicate checkpoint for thread {} turn {}",
                checkpoint.thread_id,
                checkpoint.turn_number,
            ));
        }

        inner.by_thread_turn.insert(key, checkpoint.id.clone());
        inner
            .by_thread
            .entry(checkpoint.thread_id.clone())
            .or_default()
            .push(checkpoint.id.clone());
        inner
            .by_id
            .insert(checkpoint.id.clone(), checkpoint.clone());
        drop(inner);

        Ok(checkpoint)
    }

    async fn get(&self, id: &CheckpointId) -> Result<Option<Checkpoint>> {
        let inner = self.inner.read().await;
        let result = inner.by_id.get(id).cloned();
        drop(inner);
        Ok(result)
    }

    async fn get_by_turn(
        &self,
        thread_id: &ThreadId,
        turn_number: u32,
    ) -> Result<Option<Checkpoint>> {
        let inner = self.inner.read().await;
        let key = (thread_id.clone(), turn_number);
        let result = inner
            .by_thread_turn
            .get(&key)
            .and_then(|id| inner.by_id.get(id))
            .cloned();
        drop(inner);
        Ok(result)
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<Checkpoint>> {
        let inner = self.inner.read().await;
        let Some(ids) = inner.by_thread.get(thread_id) else {
            return Ok(Vec::new());
        };
        let mut checkpoints: Vec<Checkpoint> = ids
            .iter()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        checkpoints.sort_by_key(|c| c.turn_number);
        Ok(checkpoints)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::task::AgentTaskId;
    use super::*;
    use agent_sdk_core::{TokenUsage, llm};
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-ckpt-store-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-ckpt-store-b")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
        }
    }

    fn sample_params(thread_id: ThreadId, turn: u32) -> NewCheckpointParams {
        NewCheckpointParams {
            thread_id,
            turn_number: turn,
            task_id: AgentTaskId::from_string(format!("task_turn-{turn}")),
            messages: vec![llm::Message::user(format!("turn {turn}"))],
            agent_state_snapshot: serde_json::json!({ "turn": turn }),
            turn_usage: usage(100, 50),
            now: t0(),
        }
    }

    // ── commit_checkpoint ─────────────────────────────────────────

    #[tokio::test]
    async fn commit_creates_checkpoint() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let ckpt = store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("commit")?;

        assert_eq!(ckpt.thread_id, thread_a());
        assert_eq!(ckpt.turn_number, 1);
        assert!(ckpt.id.as_str().starts_with("checkpoint_"));
        Ok(())
    }

    #[tokio::test]
    async fn commit_rejects_duplicate_turn_number() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("first")?;

        let err = store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("duplicate"),
            "expected duplicate error, got: {err}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn commit_rejects_zero_turn_number() {
        let store = InMemoryCheckpointStore::new();
        let err = store
            .commit_checkpoint(sample_params(thread_a(), 0))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("turn_number"),
            "expected validation error, got: {err}",
        );
    }

    #[tokio::test]
    async fn commit_allows_same_turn_number_on_different_threads() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("thread a")?;
        store
            .commit_checkpoint(sample_params(thread_b(), 1))
            .await
            .context("thread b")?;
        Ok(())
    }

    // ── get ─────────────────────────────────────────────���─────────

    #[tokio::test]
    async fn get_returns_existing_checkpoint() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let ckpt = store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("commit")?;

        let loaded = store
            .get(&ckpt.id)
            .await
            .context("get")?
            .context("not found")?;
        assert_eq!(loaded.id, ckpt.id);
        assert_eq!(loaded.turn_number, 1);
        Ok(())
    }

    #[tokio::test]
    async fn get_returns_none_for_unknown_id() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let result = store
            .get(&CheckpointId::from_string("checkpoint_unknown"))
            .await
            .context("get")?;
        assert!(result.is_none());
        Ok(())
    }

    // ── get_by_turn ──────────────────────────────────────────────

    #[tokio::test]
    async fn get_by_turn_returns_matching_checkpoint() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let ckpt = store
            .commit_checkpoint(sample_params(thread_a(), 3))
            .await
            .context("commit")?;

        let loaded = store
            .get_by_turn(&thread_a(), 3)
            .await
            .context("get_by_turn")?
            .context("not found")?;
        assert_eq!(loaded.id, ckpt.id);
        assert_eq!(loaded.turn_number, 3);
        Ok(())
    }

    #[tokio::test]
    async fn get_by_turn_returns_none_for_unknown_turn() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("commit")?;

        let result = store
            .get_by_turn(&thread_a(), 99)
            .await
            .context("get_by_turn")?;
        assert!(result.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn get_by_turn_returns_none_for_unknown_thread() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let result = store
            .get_by_turn(&thread_a(), 1)
            .await
            .context("get_by_turn")?;
        assert!(result.is_none());
        Ok(())
    }

    // ── list_by_thread ───────────────────────────────────────────

    #[tokio::test]
    async fn list_by_thread_returns_ordered_by_turn_number() -> Result<()> {
        let store = InMemoryCheckpointStore::new();

        // Insert out of order to verify sorting.
        store
            .commit_checkpoint(sample_params(thread_a(), 3))
            .await
            .context("turn 3")?;
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("turn 1")?;
        store
            .commit_checkpoint(sample_params(thread_a(), 2))
            .await
            .context("turn 2")?;

        let list = store.list_by_thread(&thread_a()).await.context("list")?;
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].turn_number, 1);
        assert_eq!(list[1].turn_number, 2);
        assert_eq!(list[2].turn_number, 3);
        Ok(())
    }

    #[tokio::test]
    async fn list_by_thread_returns_empty_for_unknown_thread() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let list = store.list_by_thread(&thread_a()).await.context("list")?;
        assert!(list.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn list_by_thread_isolates_threads() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("a-1")?;
        store
            .commit_checkpoint(sample_params(thread_a(), 2))
            .await
            .context("a-2")?;
        store
            .commit_checkpoint(sample_params(thread_b(), 1))
            .await
            .context("b-1")?;

        let a_list = store.list_by_thread(&thread_a()).await.context("a")?;
        let b_list = store.list_by_thread(&thread_b()).await.context("b")?;
        assert_eq!(a_list.len(), 2);
        assert_eq!(b_list.len(), 1);
        Ok(())
    }

    // ── uniqueness regression ────────────────────────────────────

    #[tokio::test]
    async fn uniqueness_is_per_thread_turn_pair() -> Result<()> {
        let store = InMemoryCheckpointStore::new();

        // Same turn number on two different threads: OK.
        store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .context("a-1")?;
        store
            .commit_checkpoint(sample_params(thread_b(), 1))
            .await
            .context("b-1")?;

        // Different turn numbers on the same thread: OK.
        store
            .commit_checkpoint(sample_params(thread_a(), 2))
            .await
            .context("a-2")?;

        // Same (thread, turn) again: rejected.
        let err = store
            .commit_checkpoint(sample_params(thread_a(), 1))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
        Ok(())
    }

    // ── snapshot fidelity ────────────────────────────────────────

    #[tokio::test]
    async fn checkpoint_preserves_messages_and_snapshot() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let params = NewCheckpointParams {
            messages: vec![
                llm::Message::user("hello"),
                llm::Message::assistant("hi there"),
            ],
            agent_state_snapshot: serde_json::json!({
                "thread_id": "t-ckpt-store-a",
                "turn_count": 1,
                "metadata": { "key": "value" },
            }),
            ..sample_params(thread_a(), 1)
        };

        let ckpt = store.commit_checkpoint(params).await.context("commit")?;

        let loaded = store
            .get(&ckpt.id)
            .await
            .context("get")?
            .context("not found")?;

        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.agent_state_snapshot["metadata"]["key"], "value",);
        Ok(())
    }

    // ── timing ───────────────────────────────────────────────────

    #[tokio::test]
    async fn checkpoint_records_created_at() -> Result<()> {
        let store = InMemoryCheckpointStore::new();
        let params = NewCheckpointParams {
            now: t_plus(42),
            ..sample_params(thread_a(), 1)
        };
        let ckpt = store.commit_checkpoint(params).await.context("commit")?;
        assert_eq!(ckpt.created_at, t_plus(42));
        Ok(())
    }
}
