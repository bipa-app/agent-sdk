//! Retention-floor tracking for committed events.
//!
//! The retention floor is the lowest `sequence` number that is
//! **guaranteed to still exist** in the committed-events journal for a
//! given thread.  Events below the floor may have been purged by the
//! janitor process.
//!
//! # Purpose
//!
//! - **Replay safety**: a reconnecting client that asks for events
//!   below the retention floor receives a `RetentionGap` control frame
//!   instead of silently missing data.
//! - **Garbage collection**: the janitor advances the floor and deletes
//!   rows below it, keeping the event table bounded.
//! - **Operational visibility**: operators can inspect the per-thread
//!   retention posture before GA hardening begins.
//!
//! # Lifecycle
//!
//! 1. A thread is created → retention floor defaults to 0 (all events
//!    are retained).
//! 2. The janitor evaluates a retention policy (time-based, count-based,
//!    or both) and computes a new floor.
//! 3. The janitor advances the floor and deletes events below it in a
//!    single transaction.
//! 4. Replay queries consult the floor to detect gaps.
//!
//! The janitor implementation itself is **out of scope** for this slice;
//! only the storage shape and query surface are defined here.

use agent_sdk_core::ThreadId;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::sync::RwLock;

// ─────────────────────────────────────────────────────────────────────
// Cursor
// ──��──────────────────────────────────────────────────────────────────

/// Per-thread retention watermark.
///
/// Events with `sequence < retention_floor` may have been purged and
/// are not guaranteed to be available for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionCursor {
    /// Thread this cursor belongs to.
    pub thread_id: ThreadId,
    /// Lowest sequence guaranteed to exist.  Events below this value
    /// may have been garbage-collected.
    pub retention_floor: u64,
    /// When the floor was last advanced.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Retention-floor store for committed-event garbage collection.
///
/// The janitor uses this surface to advance retention floors and the
/// replay path uses it to detect retention gaps.
///
/// # Contract
///
/// - The floor only moves forward (monotonically increasing).
/// - A thread with no cursor has an implicit floor of 0 (all events
///   retained).
/// - Advancing the floor and purging events below it must happen
///   atomically in a single transaction.
#[async_trait]
pub trait RetentionStore: Send + Sync {
    /// Get the retention cursor for a thread.
    ///
    /// Returns `None` if no cursor has been set (implicit floor = 0).
    async fn get_cursor(&self, thread_id: &ThreadId) -> Result<Option<RetentionCursor>>;

    /// Advance the retention floor for a thread.
    ///
    /// The new floor must be greater than or equal to the current floor.
    /// In the Postgres backend, this also deletes committed events below
    /// the new floor in the same transaction.
    ///
    /// Returns the updated cursor.
    async fn advance_floor(
        &self,
        thread_id: &ThreadId,
        new_floor: u64,
        now: OffsetDateTime,
    ) -> Result<RetentionCursor>;

    /// Get the effective retention floor for a thread.
    ///
    /// Returns 0 if no cursor exists (all events retained).
    async fn effective_floor(&self, thread_id: &ThreadId) -> Result<u64> {
        Ok(self
            .get_cursor(thread_id)
            .await?
            .map_or(0, |c| c.retention_floor))
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct InMemoryRetentionStoreInner {
    cursors: HashMap<String, RetentionCursor>,
}

/// In-memory reference implementation of [`RetentionStore`].
///
/// Cloning shares the same underlying state.
#[derive(Clone, Default)]
pub struct InMemoryRetentionStore {
    inner: Arc<RwLock<InMemoryRetentionStoreInner>>,
}

impl InMemoryRetentionStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl RetentionStore for InMemoryRetentionStore {
    async fn get_cursor(&self, thread_id: &ThreadId) -> Result<Option<RetentionCursor>> {
        let inner = self.inner.read().await;
        let result = inner.cursors.get(&thread_id.0).cloned();
        drop(inner);
        Ok(result)
    }

    async fn advance_floor(
        &self,
        thread_id: &ThreadId,
        new_floor: u64,
        now: OffsetDateTime,
    ) -> Result<RetentionCursor> {
        let mut inner = self.inner.write().await;
        let current_floor = inner
            .cursors
            .get(&thread_id.0)
            .map_or(0, |c| c.retention_floor);

        anyhow::ensure!(
            new_floor >= current_floor,
            "retention floor can only advance: current {current_floor}, requested {new_floor}",
        );

        let cursor = RetentionCursor {
            thread_id: thread_id.clone(),
            retention_floor: new_floor,
            updated_at: now,
        };
        inner.cursors.insert(thread_id.0.clone(), cursor.clone());
        drop(inner);
        Ok(cursor)
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
        ThreadId::from_string("t-retention-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-retention-b")
    }

    #[tokio::test]
    async fn no_cursor_returns_none() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        assert!(store.get_cursor(&thread_a()).await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn effective_floor_defaults_to_zero() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        assert_eq!(store.effective_floor(&thread_a()).await?, 0);
        Ok(())
    }

    #[tokio::test]
    async fn advance_floor_creates_cursor() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        let cursor = store.advance_floor(&thread_a(), 10, t0()).await?;

        assert_eq!(cursor.thread_id, thread_a());
        assert_eq!(cursor.retention_floor, 10);
        assert_eq!(cursor.updated_at, t0());
        Ok(())
    }

    #[tokio::test]
    async fn advance_floor_is_monotonic() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        store.advance_floor(&thread_a(), 10, t0()).await?;
        store.advance_floor(&thread_a(), 20, t_plus(1)).await?;

        let cursor = store.get_cursor(&thread_a()).await?.expect("cursor exists");
        assert_eq!(cursor.retention_floor, 20);
        Ok(())
    }

    #[tokio::test]
    async fn advance_floor_rejects_backward_move() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        store.advance_floor(&thread_a(), 10, t0()).await?;

        let result = store.advance_floor(&thread_a(), 5, t_plus(1)).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn advance_floor_allows_same_value() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        store.advance_floor(&thread_a(), 10, t0()).await?;
        store.advance_floor(&thread_a(), 10, t_plus(1)).await?;

        let cursor = store.get_cursor(&thread_a()).await?.expect("cursor exists");
        assert_eq!(cursor.retention_floor, 10);
        Ok(())
    }

    #[tokio::test]
    async fn threads_have_independent_cursors() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        store.advance_floor(&thread_a(), 10, t0()).await?;
        store.advance_floor(&thread_b(), 50, t0()).await?;

        assert_eq!(store.effective_floor(&thread_a()).await?, 10);
        assert_eq!(store.effective_floor(&thread_b()).await?, 50);
        Ok(())
    }

    #[tokio::test]
    async fn effective_floor_reflects_advance() -> Result<()> {
        let store = InMemoryRetentionStore::new();
        assert_eq!(store.effective_floor(&thread_a()).await?, 0);

        store.advance_floor(&thread_a(), 42, t0()).await?;
        assert_eq!(store.effective_floor(&thread_a()).await?, 42);
        Ok(())
    }
}
