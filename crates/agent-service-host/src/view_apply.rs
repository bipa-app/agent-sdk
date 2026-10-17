//! The part of a view update that both SQL backends share.
//!
//! A backend loads the view's status row, then for each event folds it,
//! loads the items the ops address that it has not loaded yet, and applies
//! the ops to a partial [`ThreadView`]. At the end it writes back the items
//! the ops touched and the status row, all in the event's transaction.

use std::collections::{BTreeMap, BTreeSet};

use agent_server::journal::CommittedEvent;
use agent_server::journal::view::{
    IndexedItem, ThreadView, ThreadViewStatus, ViewConfig, ViewOp, fold,
};

/// A view with only the items the current update reads or writes.
pub struct PartialView {
    view: ThreadView,
    loaded: BTreeSet<String>,
    touched: BTreeMap<String, u64>,
}

impl PartialView {
    pub const fn new(status: ThreadViewStatus, next_index: u64) -> Self {
        Self {
            view: ThreadView {
                status,
                items: BTreeMap::new(),
                next_index,
            },
            loaded: BTreeSet::new(),
            touched: BTreeMap::new(),
        }
    }

    /// Fold `event` into the status alone. Returns the ops; the caller
    /// loads [`Self::unloaded_keys`] and then calls [`Self::apply`].
    pub fn fold(&self, event: &CommittedEvent, config: &ViewConfig) -> Vec<ViewOp> {
        fold(
            &self.view.status,
            event.sequence,
            event.timestamp,
            &event.event,
            config,
        )
        .1
    }

    /// Keys the ops address that the view has not loaded yet.
    pub fn unloaded_keys(&self, ops: &[ViewOp]) -> Vec<String> {
        let mut keys: Vec<String> = ops
            .iter()
            .filter_map(op_key)
            .filter(|key| !self.loaded.contains(*key))
            .map(str::to_owned)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Record a loaded item; `None` when the store has no row for `key`.
    pub fn loaded(&mut self, key: String, item: Option<IndexedItem>) {
        if let Some(item) = item {
            self.view.items.insert(key.clone(), item);
        }
        self.loaded.insert(key);
    }

    pub fn apply(&mut self, sequence: u64, ops: &[ViewOp]) {
        for op in ops {
            if let Some(key) = op_key(op) {
                self.touched.insert(key.to_owned(), sequence);
            }
            self.view.apply(op);
        }
    }

    pub const fn status(&self) -> &ThreadViewStatus {
        &self.view.status
    }

    pub const fn next_index(&self) -> u64 {
        self.view.next_index
    }

    /// The items to write back: key, item, and the last event that touched
    /// it.
    pub fn touched(&self) -> impl Iterator<Item = (&str, &IndexedItem, u64)> {
        self.touched.iter().filter_map(|(key, sequence)| {
            self.view
                .items
                .get(key)
                .map(|item| (key.as_str(), item, *sequence))
        })
    }
}

fn op_key(op: &ViewOp) -> Option<&str> {
    match op {
        ViewOp::SetStatus { .. } => None,
        ViewOp::Put { key, .. } | ViewOp::AppendText { key, .. } | ViewOp::Patch { key, .. } => {
            Some(key)
        }
    }
}

/// A whole view folded from `events` in memory, for a first materialization.
pub fn materialize<'a>(
    events: impl IntoIterator<Item = &'a CommittedEvent>,
    config: &ViewConfig,
) -> ThreadView {
    ThreadView::from_events(
        events
            .into_iter()
            .map(|event| (event.sequence, event.timestamp, &event.event)),
        config,
    )
}
