//! Durable-friendly storage trait for [`MessageProjection`] rows.
//!
//! The [`MessageProjectionStore`] trait is the sole write surface for
//! the per-thread message projection. Its key design properties:
//!
//! 1. **No raw `update()`** — all mutations flow through named entry
//!    points (`commit_messages`, `replace_history`, `set_draft`,
//!    `clear_draft`) so callers cannot bypass the projection's
//!    transition guards.
//! 2. **Committed history is turn-bounded** — `commit_messages` and
//!    `replace_history` are called only at turn completion, keeping
//!    the committed projection consistent with the latest completed
//!    checkpoint.
//! 3. **Drafts capture in-flight conversation** — `set_draft` /
//!    `clear_draft` operate on a separate slot reserved for
//!    suspended-but-not-yet-committed history. A failed turn leaves
//!    the slot populated so [`super::thread_recover`] can fold the
//!    in-flight messages into the recovery view; a successful commit
//!    clears the slot.
//! 4. **All mutations are atomic** — every entry point performs its
//!    swap under the write lock so readers never observe a partial
//!    state.
//!
//! [`InMemoryMessageProjectionStore`] is the reference implementation,
//! following the same `Arc<RwLock<Inner>>` pattern as
//! [`super::thread_store::InMemoryThreadStore`].
//!
//! # Entry point summary
//!
//! | Entry point | What it mutates | Guard |
//! |-------------|----------------|-------|
//! | [`MessageProjectionStore::commit_messages`] | Appends committed messages, bumps version | Non-empty batch |
//! | [`MessageProjectionStore::replace_history`] | Swaps entire committed history, bumps version | — |
//! | [`MessageProjectionStore::set_draft`] | Replaces the in-flight draft, bumps version | — |
//! | [`MessageProjectionStore::clear_draft`] | Drops the in-flight draft, bumps version | — |
//! | [`MessageProjectionStore::get_or_create`] | Creates row with empty history and empty draft | Idempotent |
//!
//! No other entry point modifies the projection's stored state.

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_core::{ThreadId, llm};
use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::RwLock;

use super::message::MessageProjection;

/// Storage trait for [`MessageProjection`] rows.
///
/// The trait surface is deliberately narrow: `commit_messages` and
/// `replace_history` are the only mutation paths, enforcing the rule
/// that message history is written exclusively at turn completion.
///
/// Implementations must guarantee that concurrent mutations on the
/// same thread serialize, and that `get_or_create` is idempotent.
#[async_trait]
pub trait MessageProjectionStore: Send + Sync {
    /// Return the projection row, creating it if it does not exist.
    ///
    /// Idempotent: if the row already exists, returns it unchanged.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be written.
    async fn get_or_create(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection>;

    /// Look up a projection by thread id.
    ///
    /// Returns `None` if the projection has never been created.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, thread_id: &ThreadId) -> Result<Option<MessageProjection>>;

    /// Return the committed message history for a thread.
    ///
    /// Convenience method: returns `Vec::new()` if the thread has no
    /// projection yet.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>>;

    /// Append committed messages at turn completion.
    ///
    /// This is the append-only commit path. A successful call
    /// atomically:
    ///
    /// 1. Loads or creates the projection row.
    /// 2. Applies [`MessageProjection::append_committed`] with the
    ///    given messages.
    /// 3. Persists the updated row.
    ///
    /// Returns the projection as persisted.
    ///
    /// # Errors
    /// - [`super::message::MessageProjectionError::EmptyCommit`] if
    ///   `messages` is empty.
    /// - Store-level write errors.
    async fn commit_messages(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection>;

    /// Replace the entire committed history atomically.
    ///
    /// Used for context compaction: the old history is discarded and
    /// the new one takes its place. The swap happens under the write
    /// lock so readers never see partial state.
    ///
    /// Returns the projection as persisted.
    ///
    /// # Errors
    /// Store-level write errors.
    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection>;

    /// Replace the in-flight draft messages for `thread_id`.
    ///
    /// Called by the worker at every tool-boundary suspension with
    /// the full `suspended_messages` list captured at that point.
    /// The draft is overwritten (not appended) because each
    /// suspension carries the complete in-flight history through
    /// that boundary.
    ///
    /// Behavior:
    /// 1. Loads or bootstraps the projection row for `thread_id`.
    /// 2. Applies [`MessageProjection::set_draft`] with the given
    ///    messages.
    /// 3. Persists the updated row atomically.
    ///
    /// Returns the projection as persisted.
    ///
    /// # Errors
    /// Store-level write errors.
    async fn set_draft(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection>;

    /// Drop the in-flight draft for `thread_id`.
    ///
    /// Called as the final step of a successful turn commit so the
    /// next turn starts with a clean draft slot. Idempotent: calling
    /// this on a thread that has no draft is a no-op for observers,
    /// but still bumps the version and `updated_at` so the
    /// projection's monotonic timeline remains intact.
    ///
    /// Returns the projection as persisted, or [`None`] when the
    /// projection row does not exist yet (e.g. the very first commit
    /// short-circuits a clear because there was nothing to clear in
    /// the first place).
    ///
    /// # Errors
    /// Store-level write errors.
    async fn clear_draft(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<MessageProjection>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Inner {
    by_thread: HashMap<ThreadId, MessageProjection>,
}

/// In-memory reference implementation of [`MessageProjectionStore`].
///
/// Follows the same `Arc<RwLock<Inner>>` pattern as
/// [`super::thread_store::InMemoryThreadStore`]. Designed for tests
/// and single-process usage.
#[derive(Clone, Default)]
pub struct InMemoryMessageProjectionStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryMessageProjectionStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MessageProjectionStore for InMemoryMessageProjectionStore {
    async fn get_or_create(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut inner = self.inner.write().await;
        let projection = inner
            .by_thread
            .entry(thread_id.clone())
            .or_insert_with(|| MessageProjection::new(thread_id.clone(), now))
            .clone();
        drop(inner);
        Ok(projection)
    }

    async fn get(&self, thread_id: &ThreadId) -> Result<Option<MessageProjection>> {
        let inner = self.inner.read().await;
        let result = inner.by_thread.get(thread_id).cloned();
        drop(inner);
        Ok(result)
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        let inner = self.inner.read().await;
        let result = inner
            .by_thread
            .get(thread_id)
            .map(|p| p.messages.clone())
            .unwrap_or_default();
        drop(inner);
        Ok(result)
    }

    async fn commit_messages(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut inner = self.inner.write().await;
        let projection = inner
            .by_thread
            .entry(thread_id.clone())
            .or_insert_with(|| MessageProjection::new(thread_id.clone(), now));
        let updated = projection.clone().append_committed(messages, now)?;
        *projection = updated.clone();
        drop(inner);
        Ok(updated)
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut inner = self.inner.write().await;
        let projection = inner
            .by_thread
            .entry(thread_id.clone())
            .or_insert_with(|| MessageProjection::new(thread_id.clone(), now));
        let updated = projection.clone().replace_history(messages, now);
        *projection = updated.clone();
        drop(inner);
        Ok(updated)
    }

    async fn set_draft(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut inner = self.inner.write().await;
        let projection = inner
            .by_thread
            .entry(thread_id.clone())
            .or_insert_with(|| MessageProjection::new(thread_id.clone(), now));
        let updated = projection.clone().set_draft(messages, now);
        *projection = updated.clone();
        drop(inner);
        Ok(updated)
    }

    async fn clear_draft(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<MessageProjection>> {
        let mut inner = self.inner.write().await;
        let Some(projection) = inner.by_thread.get(thread_id).cloned() else {
            // No projection row exists yet — there's nothing to clear.
            // Returning `None` lets the commit path skip an unnecessary
            // bootstrap when the very first turn never wrote a draft.
            return Ok(None);
        };
        let updated = projection.clear_draft(now);
        inner.by_thread.insert(thread_id.clone(), updated.clone());
        drop(inner);
        Ok(Some(updated))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-msg-store-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-msg-store-b")
    }

    // ── get_or_create ─────────────────────────────────────────────

    #[tokio::test]
    async fn get_or_create_bootstraps_empty_projection() {
        let store = InMemoryMessageProjectionStore::new();
        let p = store.get_or_create(&thread_a(), t0()).await.unwrap();
        assert_eq!(p.thread_id, thread_a());
        assert!(p.messages.is_empty());
        assert_eq!(p.version, 0);
    }

    #[tokio::test]
    async fn get_or_create_is_idempotent() {
        let store = InMemoryMessageProjectionStore::new();
        store.get_or_create(&thread_a(), t0()).await.unwrap();

        // Commit some messages to change the row.
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("hi")], t_plus(1))
            .await
            .unwrap();

        // get_or_create must return the existing row, not a fresh one.
        let p = store.get_or_create(&thread_a(), t_plus(2)).await.unwrap();
        assert_eq!(p.message_count(), 1);
        assert_eq!(p.version, 1);
    }

    // ── get ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_returns_none_for_unknown_thread() {
        let store = InMemoryMessageProjectionStore::new();
        let result = store.get(&thread_a()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn get_returns_existing_projection() {
        let store = InMemoryMessageProjectionStore::new();
        store.get_or_create(&thread_a(), t0()).await.unwrap();
        let result = store.get(&thread_a()).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().thread_id, thread_a());
    }

    // ── get_history ───────────────────────────────────────────────

    #[tokio::test]
    async fn get_history_returns_empty_for_unknown_thread() {
        let store = InMemoryMessageProjectionStore::new();
        let history = store.get_history(&thread_a()).await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn get_history_returns_committed_messages() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(
                &thread_a(),
                vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
                t0(),
            )
            .await
            .unwrap();
        let history = store.get_history(&thread_a()).await.unwrap();
        assert_eq!(history.len(), 2);
    }

    // ─��� commit_messages ───────────────────────────────────────────

    #[tokio::test]
    async fn commit_messages_creates_projection_on_first_call() {
        let store = InMemoryMessageProjectionStore::new();
        let p = store
            .commit_messages(&thread_a(), vec![llm::Message::user("hello")], t0())
            .await
            .unwrap();
        assert_eq!(p.message_count(), 1);
        assert_eq!(p.version, 1);
    }

    #[tokio::test]
    async fn commit_messages_accumulates_across_calls() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("turn 1")], t0())
            .await
            .unwrap();
        store
            .commit_messages(
                &thread_a(),
                vec![llm::Message::assistant("reply 1")],
                t_plus(1),
            )
            .await
            .unwrap();
        let p = store
            .commit_messages(
                &thread_a(),
                vec![
                    llm::Message::user("turn 2"),
                    llm::Message::assistant("reply 2"),
                ],
                t_plus(2),
            )
            .await
            .unwrap();
        assert_eq!(p.message_count(), 4);
        assert_eq!(p.version, 3);
    }

    #[tokio::test]
    async fn commit_messages_rejects_empty_batch() {
        let store = InMemoryMessageProjectionStore::new();
        let err = store
            .commit_messages(&thread_a(), vec![], t0())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty-commit error, got: {err}"
        );
    }

    #[tokio::test]
    async fn commit_messages_isolates_threads() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("a")], t0())
            .await
            .unwrap();
        store
            .commit_messages(
                &thread_b(),
                vec![llm::Message::user("b1"), llm::Message::user("b2")],
                t0(),
            )
            .await
            .unwrap();

        let a = store.get(&thread_a()).await.unwrap().unwrap();
        let b = store.get(&thread_b()).await.unwrap().unwrap();
        assert_eq!(a.message_count(), 1);
        assert_eq!(b.message_count(), 2);
    }

    // ── replace_history ───────────────────────────────────────────

    #[tokio::test]
    async fn replace_history_swaps_messages() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(
                &thread_a(),
                vec![
                    llm::Message::user("old 1"),
                    llm::Message::assistant("old 2"),
                    llm::Message::user("old 3"),
                ],
                t0(),
            )
            .await
            .unwrap();

        let p = store
            .replace_history(
                &thread_a(),
                vec![
                    llm::Message::user("[Summary]"),
                    llm::Message::assistant("Continuing..."),
                ],
                t_plus(1),
            )
            .await
            .unwrap();
        assert_eq!(p.message_count(), 2);
        assert_eq!(p.version, 2);
    }

    #[tokio::test]
    async fn replace_history_creates_projection_if_absent() {
        let store = InMemoryMessageProjectionStore::new();
        let p = store
            .replace_history(&thread_a(), vec![llm::Message::user("bootstrapped")], t0())
            .await
            .unwrap();
        assert_eq!(p.message_count(), 1);
        assert_eq!(p.version, 1);
    }

    #[tokio::test]
    async fn replace_history_allows_empty_replacement() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("data")], t0())
            .await
            .unwrap();
        let p = store
            .replace_history(&thread_a(), vec![], t_plus(1))
            .await
            .unwrap();
        assert_eq!(p.message_count(), 0);
        assert_eq!(p.version, 2);
    }

    #[tokio::test]
    async fn replace_history_does_not_affect_other_threads() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("a")], t0())
            .await
            .unwrap();
        store
            .commit_messages(&thread_b(), vec![llm::Message::user("b")], t0())
            .await
            .unwrap();

        store
            .replace_history(&thread_a(), vec![], t_plus(1))
            .await
            .unwrap();

        let a = store.get(&thread_a()).await.unwrap().unwrap();
        let b = store.get(&thread_b()).await.unwrap().unwrap();
        assert_eq!(a.message_count(), 0);
        assert_eq!(b.message_count(), 1); // untouched
    }

    // ── set_draft / clear_draft ───────────────────────────────────

    #[tokio::test]
    async fn set_draft_creates_projection_when_absent() {
        let store = InMemoryMessageProjectionStore::new();
        let p = store
            .set_draft(&thread_a(), vec![llm::Message::user("draft")], t0())
            .await
            .unwrap();
        // Bootstrapping the row + applying set_draft both bump the
        // version, so the first persisted version observed by callers
        // is `1` rather than `0`.
        assert_eq!(p.version, 1);
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 1);
        assert_eq!(p.message_count(), 0);
    }

    #[tokio::test]
    async fn set_draft_overwrites_prior_draft() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .set_draft(&thread_a(), vec![llm::Message::user("first")], t0())
            .await
            .unwrap();
        let p = store
            .set_draft(
                &thread_a(),
                vec![
                    llm::Message::user("first"),
                    llm::Message::assistant("second"),
                ],
                t_plus(1),
            )
            .await
            .unwrap();
        // Two suspensions overwrote the draft slot — committed history
        // remains empty and the slot now carries the latest two-message
        // snapshot.
        assert_eq!(p.draft_messages.len(), 2);
        assert_eq!(p.message_count(), 0);
    }

    #[tokio::test]
    async fn set_draft_does_not_modify_committed_history() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .commit_messages(&thread_a(), vec![llm::Message::user("turn 1")], t0())
            .await
            .unwrap();
        let p = store
            .set_draft(
                &thread_a(),
                vec![llm::Message::user("turn 2 in flight")],
                t_plus(1),
            )
            .await
            .unwrap();
        assert_eq!(p.message_count(), 1, "committed history untouched");
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 1);
    }

    #[tokio::test]
    async fn clear_draft_returns_none_when_projection_absent() {
        let store = InMemoryMessageProjectionStore::new();
        let result = store.clear_draft(&thread_a(), t0()).await.unwrap();
        // No row to clear — the commit path can skip a needless
        // bootstrap on the first-turn happy path.
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn clear_draft_drops_in_flight_messages() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .set_draft(&thread_a(), vec![llm::Message::user("draft")], t0())
            .await
            .unwrap();
        let p = store
            .clear_draft(&thread_a(), t_plus(1))
            .await
            .unwrap()
            .expect("clear returns the projection row");
        assert!(!p.has_draft());
    }

    #[tokio::test]
    async fn draft_isolates_threads() {
        let store = InMemoryMessageProjectionStore::new();
        store
            .set_draft(&thread_a(), vec![llm::Message::user("a-draft")], t0())
            .await
            .unwrap();
        store
            .set_draft(&thread_b(), vec![llm::Message::user("b-draft")], t0())
            .await
            .unwrap();
        store.clear_draft(&thread_a(), t_plus(1)).await.unwrap();

        let a = store.get(&thread_a()).await.unwrap().unwrap();
        let b = store.get(&thread_b()).await.unwrap().unwrap();
        assert!(!a.has_draft());
        assert!(
            b.has_draft(),
            "thread b's draft must survive thread a's clear"
        );
        assert_eq!(b.draft_messages.len(), 1);
    }

    // ── version monotonicity ──────────────────────────────────────

    #[tokio::test]
    async fn version_increases_monotonically() {
        let store = InMemoryMessageProjectionStore::new();
        let p = store
            .commit_messages(&thread_a(), vec![llm::Message::user("a")], t0())
            .await
            .unwrap();
        assert_eq!(p.version, 1);

        let p = store
            .replace_history(&thread_a(), vec![llm::Message::user("b")], t_plus(1))
            .await
            .unwrap();
        assert_eq!(p.version, 2);

        let p = store
            .commit_messages(&thread_a(), vec![llm::Message::user("c")], t_plus(2))
            .await
            .unwrap();
        assert_eq!(p.version, 3);
    }

    // ── identity stability ────────────────────────────────────────

    #[tokio::test]
    async fn identity_survives_mutations() {
        let store = InMemoryMessageProjectionStore::new();
        let p1 = store
            .commit_messages(&thread_a(), vec![llm::Message::user("a")], t0())
            .await
            .unwrap();
        let p2 = store
            .replace_history(&thread_a(), vec![llm::Message::user("b")], t_plus(1))
            .await
            .unwrap();
        let p3 = store
            .commit_messages(&thread_a(), vec![llm::Message::user("c")], t_plus(2))
            .await
            .unwrap();
        // Identity is stable across all mutations.
        assert_eq!(p1.thread_id, p2.thread_id);
        assert_eq!(p2.thread_id, p3.thread_id);
        assert_eq!(p1.created_at, p3.created_at);
    }
}
