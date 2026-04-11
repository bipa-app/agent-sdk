//! Durable-friendly storage trait for [`Thread`] projection rows.
//!
//! The [`ThreadStore`] trait is the sole write surface for thread
//! aggregates. Its key design property is that **there is no raw
//! `update()` that touches `committed_turns` or `total_usage`** —
//! aggregates flow exclusively through [`ThreadStore::commit_turn`],
//! which delegates to [`Thread::apply_committed_turn`] under the
//! store's write lock. This prevents split ownership: no worker can
//! mutate thread counters outside the completed-turn commit path.
//!
//! [`InMemoryThreadStore`] is the reference implementation, used in
//! tests and single-process usage, following the same
//! `Arc<RwLock<Inner>>` pattern as
//! [`super::store::InMemoryAgentTaskStore`].
//!
//! # Aggregate ownership contract
//!
//! | Entry point | What it mutates | Guard |
//! |-------------|----------------|-------|
//! | [`ThreadStore::commit_turn`] | `committed_turns`, `total_usage`, `updated_at` | Status = Active |
//! | [`ThreadStore::mark_completed`] | `status`, `updated_at` | Status = Active, `committed_turns > 0` |
//! | [`ThreadStore::get_or_create`] | Creates row with zero counters | Idempotent: no-op if exists |
//!
//! No other entry point modifies aggregate counters.

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_core::{ThreadId, TokenUsage};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::RwLock;

use super::completed_turn_transaction::AtomicCompletedTurnCommitter;
use super::thread::Thread;

/// Storage trait for [`Thread`] projection rows.
///
/// The trait surface is deliberately narrow: `commit_turn` is the
/// only aggregate-mutation path, enforcing single ownership of
/// thread-level counters. There is no `update(&self, thread: Thread)`
/// because exposing one would let callers mutate counters outside the
/// commit path.
///
/// Implementations must guarantee that concurrent `commit_turn` calls
/// on the same thread serialize (only one succeeds per atomic scope),
/// and that `get_or_create` is idempotent.
#[async_trait]
pub trait ThreadStore: Send + Sync {
    /// Optional backend-specific hook that can commit the completed
    /// turn projections inside one durable transaction.
    ///
    /// In-memory stores leave this as `None`; durable backends such as
    /// Postgres override it to surface an atomic commit boundary to
    /// [`super::commit::commit_completed_turn`].
    #[must_use]
    fn atomic_completed_turn_committer(&self) -> Option<&dyn AtomicCompletedTurnCommitter> {
        None
    }

    /// Return the thread row, creating it if it does not exist.
    ///
    /// Idempotent: if the row already exists, returns it unchanged.
    /// Used by the commit path to bootstrap a thread on first use
    /// without requiring a separate "create thread" ceremony.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be written.
    async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread>;

    /// Look up a thread by id.
    ///
    /// Returns `None` if the thread has never been created.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, thread_id: &ThreadId) -> Result<Option<Thread>>;

    /// Commit a completed turn's token usage to the thread's
    /// aggregates.
    ///
    /// This is the **only** mutation path for `committed_turns` and
    /// `total_usage`. A successful call atomically:
    ///
    /// 1. Loads or creates the thread row.
    /// 2. Applies [`Thread::apply_committed_turn`] with the given
    ///    `turn_usage`.
    /// 3. Persists the updated row.
    ///
    /// Returns the thread as persisted.
    ///
    /// # Errors
    /// - [`super::thread::ThreadSchemaError::CommitOnCompletedThread`] if the thread
    ///   has already been completed.
    /// - Store-level write errors.
    async fn commit_turn(
        &self,
        thread_id: &ThreadId,
        turn_usage: &TokenUsage,
        now: OffsetDateTime,
    ) -> Result<Thread>;

    /// Close the thread so no further turns can be committed.
    ///
    /// The thread must be active and have at least one committed turn.
    ///
    /// # Errors
    /// - `thread does not exist` if the thread has never been created.
    /// - [`super::thread::ThreadSchemaError::AlreadyCompleted`] if already closed.
    /// - [`super::thread::ThreadSchemaError::CompletedWithZeroTurns`] if no turns
    ///   have been committed.
    async fn mark_completed(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread>;

    /// List all threads in the store.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list(&self) -> Result<Vec<Thread>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Inner {
    by_id: HashMap<ThreadId, Thread>,
}

/// In-memory reference implementation of [`ThreadStore`].
///
/// Follows the same `Arc<RwLock<Inner>>` pattern as
/// [`super::store::InMemoryAgentTaskStore`]. Designed for tests and
/// single-process usage.
#[derive(Clone, Default)]
pub struct InMemoryThreadStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryThreadStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ThreadStore for InMemoryThreadStore {
    async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut inner = self.inner.write().await;
        let thread = inner
            .by_id
            .entry(thread_id.clone())
            .or_insert_with(|| Thread::new(thread_id.clone(), now))
            .clone();
        drop(inner);
        Ok(thread)
    }

    async fn get(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
        let inner = self.inner.read().await;
        let result = inner.by_id.get(thread_id).cloned();
        drop(inner);
        Ok(result)
    }

    async fn commit_turn(
        &self,
        thread_id: &ThreadId,
        turn_usage: &TokenUsage,
        now: OffsetDateTime,
    ) -> Result<Thread> {
        let mut inner = self.inner.write().await;
        let thread = inner
            .by_id
            .entry(thread_id.clone())
            .or_insert_with(|| Thread::new(thread_id.clone(), now));
        let updated = thread.clone().apply_committed_turn(turn_usage, now)?;
        *thread = updated.clone();
        drop(inner);
        Ok(updated)
    }

    async fn mark_completed(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut inner = self.inner.write().await;
        let thread = inner
            .by_id
            .get(thread_id)
            .ok_or_else(|| anyhow!("thread {thread_id} does not exist"))?;
        let completed = thread.clone().mark_completed(now)?;
        inner.by_id.insert(thread_id.clone(), completed.clone());
        drop(inner);
        Ok(completed)
    }

    async fn list(&self) -> Result<Vec<Thread>> {
        let inner = self.inner.read().await;
        let result = inner.by_id.values().cloned().collect();
        drop(inner);
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::thread::ThreadStatus;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-store-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-store-b")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
        }
    }

    // ── get_or_create ─────────────────────────────────────────────

    #[tokio::test]
    async fn get_or_create_bootstraps_thread_on_first_call() {
        let store = InMemoryThreadStore::new();
        let thread = store.get_or_create(&thread_a(), t0()).await.unwrap();
        assert_eq!(thread.thread_id, thread_a());
        assert_eq!(thread.status, ThreadStatus::Active);
        assert_eq!(thread.committed_turns, 0);
    }

    #[tokio::test]
    async fn get_or_create_is_idempotent() {
        let store = InMemoryThreadStore::new();
        let first = store.get_or_create(&thread_a(), t0()).await.unwrap();

        // Commit a turn to change the row.
        store
            .commit_turn(&thread_a(), &usage(10, 5), t_plus(1))
            .await
            .unwrap();

        // get_or_create must return the existing row, not a fresh one.
        let second = store.get_or_create(&thread_a(), t_plus(2)).await.unwrap();
        assert_eq!(second.committed_turns, 1);
        assert_eq!(first.thread_id, second.thread_id);
    }

    // ── get ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_returns_none_for_unknown_thread() {
        let store = InMemoryThreadStore::new();
        let result = store.get(&thread_a()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn get_returns_existing_thread() {
        let store = InMemoryThreadStore::new();
        store.get_or_create(&thread_a(), t0()).await.unwrap();
        let result = store.get(&thread_a()).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().thread_id, thread_a());
    }

    // ── commit_turn ───────────────────────────────────────────────

    #[tokio::test]
    async fn commit_turn_creates_thread_on_first_call() {
        let store = InMemoryThreadStore::new();
        let thread = store
            .commit_turn(&thread_a(), &usage(100, 50), t0())
            .await
            .unwrap();
        assert_eq!(thread.committed_turns, 1);
        assert_eq!(thread.total_usage, usage(100, 50));
    }

    #[tokio::test]
    async fn commit_turn_accumulates_across_calls() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(100, 50), t0())
            .await
            .unwrap();
        store
            .commit_turn(&thread_a(), &usage(200, 80), t_plus(1))
            .await
            .unwrap();
        let thread = store
            .commit_turn(&thread_a(), &usage(50, 20), t_plus(2))
            .await
            .unwrap();
        assert_eq!(thread.committed_turns, 3);
        assert_eq!(thread.total_usage, usage(350, 150));
    }

    #[tokio::test]
    async fn commit_turn_rejects_completed_thread() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(10, 5), t0())
            .await
            .unwrap();
        store.mark_completed(&thread_a(), t_plus(1)).await.unwrap();
        let err = store
            .commit_turn(&thread_a(), &usage(10, 5), t_plus(2))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("completed"),
            "expected completed error, got: {err}"
        );
    }

    #[tokio::test]
    async fn commit_turn_isolates_threads() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(100, 50), t0())
            .await
            .unwrap();
        store
            .commit_turn(&thread_b(), &usage(200, 80), t0())
            .await
            .unwrap();

        let a = store.get(&thread_a()).await.unwrap().unwrap();
        let b = store.get(&thread_b()).await.unwrap().unwrap();
        assert_eq!(a.committed_turns, 1);
        assert_eq!(b.committed_turns, 1);
        assert_eq!(a.total_usage, usage(100, 50));
        assert_eq!(b.total_usage, usage(200, 80));
    }

    // ── mark_completed ────────────────────────────────────────────

    #[tokio::test]
    async fn mark_completed_transitions_thread() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(10, 5), t0())
            .await
            .unwrap();
        let thread = store.mark_completed(&thread_a(), t_plus(1)).await.unwrap();
        assert_eq!(thread.status, ThreadStatus::Completed);
    }

    #[tokio::test]
    async fn mark_completed_rejects_unknown_thread() {
        let store = InMemoryThreadStore::new();
        let err = store.mark_completed(&thread_a(), t0()).await.unwrap_err();
        assert!(
            err.to_string().contains("does not exist"),
            "expected not-found, got: {err}"
        );
    }

    #[tokio::test]
    async fn mark_completed_rejects_already_completed() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(10, 5), t0())
            .await
            .unwrap();
        store.mark_completed(&thread_a(), t_plus(1)).await.unwrap();
        let err = store
            .mark_completed(&thread_a(), t_plus(2))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("already completed"),
            "expected already-completed, got: {err}"
        );
    }

    #[tokio::test]
    async fn mark_completed_rejects_zero_turn_thread() {
        let store = InMemoryThreadStore::new();
        store.get_or_create(&thread_a(), t0()).await.unwrap();
        let err = store
            .mark_completed(&thread_a(), t_plus(1))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("committed_turns"),
            "expected zero-turns error, got: {err}"
        );
    }

    // ── list ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn list_returns_all_threads() {
        let store = InMemoryThreadStore::new();
        store
            .commit_turn(&thread_a(), &usage(10, 5), t0())
            .await
            .unwrap();
        store
            .commit_turn(&thread_b(), &usage(20, 10), t0())
            .await
            .unwrap();
        let threads = store.list().await.unwrap();
        assert_eq!(threads.len(), 2);
    }

    #[tokio::test]
    async fn list_returns_empty_for_empty_store() {
        let store = InMemoryThreadStore::new();
        let threads = store.list().await.unwrap();
        assert!(threads.is_empty());
    }

    // ── aggregate ownership regression ────────────────────────────
    //
    // Prove that thread aggregates are only reachable through the
    // commit_turn path. The trait has no `update()` method that
    // accepts a `Thread`, so the only way to modify
    // `committed_turns` or `total_usage` is through `commit_turn`.
    // These tests verify the invariant from the consumer's
    // perspective.

    #[tokio::test]
    async fn aggregates_are_monotonically_increasing() {
        let store = InMemoryThreadStore::new();
        let mut prev_turns = 0;
        let mut prev_input = 0;
        for i in 1..=5 {
            let thread = store
                .commit_turn(&thread_a(), &usage(10, 5), t_plus(i))
                .await
                .unwrap();
            assert!(thread.committed_turns > prev_turns);
            assert!(thread.total_usage.input_tokens > prev_input);
            prev_turns = thread.committed_turns;
            prev_input = thread.total_usage.input_tokens;
        }
    }

    #[tokio::test]
    async fn thread_identity_survives_multiple_commits() {
        let store = InMemoryThreadStore::new();
        let t1 = store
            .commit_turn(&thread_a(), &usage(10, 5), t0())
            .await
            .unwrap();
        let t2 = store
            .commit_turn(&thread_a(), &usage(20, 10), t_plus(1))
            .await
            .unwrap();
        // Identity fields are stable across commits.
        assert_eq!(t1.thread_id, t2.thread_id);
        assert_eq!(t1.created_at, t2.created_at);
        // Aggregates advanced.
        assert_eq!(t2.committed_turns, 2);
        assert_eq!(t2.total_usage, usage(30, 15));
    }
}
