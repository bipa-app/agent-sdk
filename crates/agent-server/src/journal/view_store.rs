//! Read access to the session views the stores maintain.
//!
//! A store with [`ViewConfig::enabled`](super::view::ViewConfig) folds every
//! committed event into its thread's view in the same transaction as the
//! event. A thread's view is materialized on its first read: the store folds
//! the thread's retained events under the thread lock and keeps the view
//! current from then on. A view that an older binary left behind (events
//! with no view update) is caught up the same way, by the next write or read.

use super::view::{IndexedItem, ThreadView, ThreadViewStatus};
use agent_sdk_foundation::ThreadId;
use anyhow::Result;
use async_trait::async_trait;

/// The newest items of a view and its status, at one sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadViewSnapshot {
    pub status: ThreadViewStatus,
    /// The index the next new item gets.
    pub next_index: u64,
    /// Items in timeline order: the newest `item_limit` of them.
    pub items: Vec<(String, IndexedItem)>,
    /// Whether the view has items older than the first one returned.
    pub has_older: bool,
}

impl ThreadViewSnapshot {
    /// The snapshot of a whole in-memory view, newest `item_limit` items.
    #[must_use]
    pub fn of(view: &ThreadView, item_limit: usize) -> Self {
        let mut items: Vec<(String, IndexedItem)> = view
            .items
            .iter()
            .map(|(key, indexed)| (key.clone(), indexed.clone()))
            .collect();
        items.sort_by_key(|(_, indexed)| indexed.index);
        let has_older = items.len() > item_limit;
        let items = items.split_off(items.len().saturating_sub(item_limit));
        Self {
            status: view.status.clone(),
            next_index: view.next_index,
            items,
            has_older,
        }
    }
}

/// The store keeps no views: [`ViewConfig::enabled`](super::view::ViewConfig)
/// is false on this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewsDisabled;

impl std::fmt::Display for ViewsDisabled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread views are disabled on this host")
    }
}

impl std::error::Error for ViewsDisabled {}

/// Read the session view of a thread.
///
/// Both methods return [`ViewsDisabled`] (as the error's root cause) when
/// the store keeps no views, and `None` for a thread with no events or a
/// thread being purged.
#[async_trait]
pub trait ThreadViewStore: Send + Sync {
    /// The view's status and its newest `item_limit` items, materializing
    /// or catching up the view first when it is missing or behind.
    async fn thread_view(
        &self,
        thread_id: &ThreadId,
        item_limit: usize,
    ) -> Result<Option<ThreadViewSnapshot>>;

    /// Up to `limit` items with an index below `before_index`, in timeline
    /// order, for paging back from a snapshot.
    async fn thread_view_items(
        &self,
        thread_id: &ThreadId,
        before_index: u64,
        limit: usize,
    ) -> Result<Option<Vec<(String, IndexedItem)>>>;
}

/// The items of `snapshot_items` (timeline order) below `before_index`, the
/// last `limit` of them.
#[must_use]
pub fn items_before(
    mut items: Vec<(String, IndexedItem)>,
    before_index: u64,
    limit: usize,
) -> Vec<(String, IndexedItem)> {
    items.retain(|(_, indexed)| indexed.index < before_index);
    items.sort_by_key(|(_, indexed)| indexed.index);
    items.split_off(items.len().saturating_sub(limit))
}
