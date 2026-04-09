//! Durable-friendly storage traits for [`AgentTask`] rows.
//!
//! This module defines the [`AgentTaskStore`] trait and a reference
//! [`InMemoryAgentTaskStore`] implementation. The trait surface deliberately
//! mirrors the indexes a production store must expose (see the module-level
//! documentation on [`super`]), so Phase 2.2+ can depend on the semantics
//! without re-designing the API.
//!
//! Every `insert` / `update` calls [`AgentTask::validate`] before committing
//! the row, so no invariant-violating task can ever be reached through the
//! public API — not even in-memory during tests. The in-memory store also
//! enforces the **partial-unique "one active root per thread"** invariant,
//! which is the single most important durability guarantee the acquisition
//! API (Phase 2.3) will rely on.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use tokio::sync::RwLock;

use super::task::{AgentTask, AgentTaskId, TaskKind, TaskStatus};

/// Persistent store for [`AgentTask`] rows.
///
/// Implementations are required to expose the index surface documented on
/// the [`super`] module. The reference in-memory implementation below is
/// used in tests and also acts as the semantic spec for any future SQL- or
/// Redis-backed store.
#[async_trait]
pub trait AgentTaskStore: Send + Sync {
    /// Insert a new task row.
    ///
    /// The row must pass [`AgentTask::validate`] and, if it is a
    /// [`TaskKind::RootTurn`] in a non-terminal status, must not collide
    /// with an existing active root on the same thread.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, if a row with the
    /// same `id` already exists, or if the active-root invariant would
    /// be violated.
    async fn insert(&self, task: AgentTask) -> Result<()>;

    /// Look up a task by id.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>>;

    /// Replace a task row with an updated version.
    ///
    /// The incoming row must pass [`AgentTask::validate`] and must refer
    /// to an `id` that already exists in the store.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, does not exist, or
    /// if the active-root invariant would be violated after the update.
    async fn update(&self, task: AgentTask) -> Result<()>;

    /// List every task (root and descendants) bound to a thread.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>>;

    /// List the direct children of a task.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>>;

    /// List every task currently in the given status.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>>;

    /// Return the currently active [`TaskKind::RootTurn`] for a thread, if
    /// one exists.
    ///
    /// "Active" means the row is in any non-terminal status. This is the
    /// primary consumer of the partial-unique "one active root per thread"
    /// invariant.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>>;

    /// Remove every stored task. Used by tests.
    ///
    /// # Errors
    /// Returns an error if the store cannot be cleared.
    async fn clear(&self) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Inner {
    /// Primary key index.
    by_id: HashMap<AgentTaskId, AgentTask>,
    /// `(thread_id)` index: every task row scoped to a thread.
    by_thread: HashMap<ThreadId, Vec<AgentTaskId>>,
    /// `(parent_id)` index: direct children of a parent task.
    by_parent: HashMap<AgentTaskId, Vec<AgentTaskId>>,
    /// `(status)` index: rows currently in each lifecycle state.
    by_status: HashMap<TaskStatus, BTreeSet<AgentTaskId>>,
    /// Partial unique index on `(thread_id)` where
    /// `kind = root_turn AND status NOT IN (completed, failed, cancelled)`.
    active_root_by_thread: HashMap<ThreadId, AgentTaskId>,
}

/// In-memory reference implementation of [`AgentTaskStore`].
///
/// Designed for tests and single-process usage. The implementation takes
/// the index surface seriously: every required index from the module-level
/// doc is kept in sync on every write, and the partial-unique "one active
/// root per thread" constraint is enforced on both `insert` and `update`.
#[derive(Clone, Default)]
pub struct InMemoryAgentTaskStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryAgentTaskStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Inner {
    fn add_to_indexes(&mut self, task: &AgentTask) {
        self.by_thread
            .entry(task.thread_id.clone())
            .or_default()
            .push(task.id.clone());

        if let Some(parent_id) = &task.parent_id {
            self.by_parent
                .entry(parent_id.clone())
                .or_default()
                .push(task.id.clone());
        }

        self.by_status
            .entry(task.status)
            .or_default()
            .insert(task.id.clone());

        if task.kind == TaskKind::RootTurn && task.status.is_active() {
            self.active_root_by_thread
                .insert(task.thread_id.clone(), task.id.clone());
        }
    }

    fn remove_from_status_index(&mut self, task: &AgentTask) {
        if let Some(set) = self.by_status.get_mut(&task.status) {
            set.remove(&task.id);
            if set.is_empty() {
                self.by_status.remove(&task.status);
            }
        }
    }

    fn remove_active_root_if_match(&mut self, task: &AgentTask) {
        if task.kind != TaskKind::RootTurn {
            return;
        }
        if let Some(current) = self.active_root_by_thread.get(&task.thread_id)
            && *current == task.id
        {
            self.active_root_by_thread.remove(&task.thread_id);
        }
    }
}

#[async_trait]
impl AgentTaskStore for InMemoryAgentTaskStore {
    async fn insert(&self, task: AgentTask) -> Result<()> {
        task.validate()
            .context("insert rejected: task failed schema validation")?;

        let mut inner = self.inner.write().await;

        if inner.by_id.contains_key(&task.id) {
            return Err(anyhow!(
                "insert rejected: task id {} already exists",
                task.id
            ));
        }

        // Partial-unique: one active root per thread.
        if task.kind == TaskKind::RootTurn
            && task.status.is_active()
            && let Some(existing) = inner.active_root_by_thread.get(&task.thread_id)
        {
            return Err(anyhow!(
                "insert rejected: thread {} already has active root task {}",
                task.thread_id,
                existing
            ));
        }

        inner.add_to_indexes(&task);
        inner.by_id.insert(task.id.clone(), task);
        drop(inner);
        Ok(())
    }

    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
        let inner = self.inner.read().await;
        let result = inner.by_id.get(id).cloned();
        drop(inner);
        Ok(result)
    }

    async fn update(&self, task: AgentTask) -> Result<()> {
        task.validate()
            .context("update rejected: task failed schema validation")?;

        let mut inner = self.inner.write().await;

        let old = inner
            .by_id
            .get(&task.id)
            .cloned()
            .ok_or_else(|| anyhow!("update rejected: task {} does not exist", task.id))?;

        // Partial-unique check: if the new row is still an active root, make
        // sure no *other* row already holds the active-root slot on the same
        // thread.
        if task.kind == TaskKind::RootTurn
            && task.status.is_active()
            && let Some(current) = inner.active_root_by_thread.get(&task.thread_id)
            && *current != task.id
        {
            return Err(anyhow!(
                "update rejected: thread {} already has a different active root task {}",
                task.thread_id,
                current
            ));
        }

        // Drop the old row from every secondary index; the primary-key
        // entry will be overwritten below.
        inner.remove_from_status_index(&old);
        if old.kind == TaskKind::RootTurn && !task.status.is_active() {
            inner.remove_active_root_if_match(&old);
        }

        // Thread / parent indexes are append-only and keyed by `id`, so the
        // old entries stay in place and still point at the (now-updated)
        // row in `by_id`. That matches how a SQL backend would behave —
        // `(thread_id)` / `(parent_id)` are invariants of the row, never
        // mutated in-place.

        inner
            .by_status
            .entry(task.status)
            .or_default()
            .insert(task.id.clone());

        if task.kind == TaskKind::RootTurn {
            if task.status.is_active() {
                inner
                    .active_root_by_thread
                    .insert(task.thread_id.clone(), task.id.clone());
            } else {
                inner.remove_active_root_if_match(&old);
            }
        }

        inner.by_id.insert(task.id.clone(), task);
        drop(inner);
        Ok(())
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_thread
            .get(thread_id)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_parent
            .get(parent_id)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .by_status
            .get(&status)
            .into_iter()
            .flatten()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
        let inner = self.inner.read().await;
        let result = inner
            .active_root_by_thread
            .get(thread_id)
            .and_then(|id| inner.by_id.get(id).cloned());
        drop(inner);
        Ok(result)
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        *inner = Inner::default();
        drop(inner);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::{LeaseId, WorkerId};
    use time::OffsetDateTime;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("valid ts")
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000 + secs).expect("valid ts")
    }

    fn thread(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t0(), 3)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_and_get_round_trip() {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.expect("insert");
        let got = store.get(&id).await.expect("get").expect("exists");
        assert_eq!(got, task);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_invalid_task() {
        let store = InMemoryAgentTaskStore::new();
        let mut task = fresh_root("t1");
        task.status = TaskStatus::Running; // invalid: missing lease fields
        let err = store.insert(task).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_invalid_task() {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.expect("insert");

        // Load and corrupt the row.
        let mut bad = store.get(&id).await.expect("get").expect("exists");
        bad.status = TaskStatus::Running; // still no lease fields -> invalid
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_children_returns_only_direct_children() {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.expect("insert root");

        let tool_a = AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).expect("a");
        let tool_b = AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).expect("b");
        store.insert(tool_a.clone()).await.expect("insert a");
        store.insert(tool_b.clone()).await.expect("insert b");

        let children = store.list_children(&root.id).await.expect("children");
        assert_eq!(children.len(), 2);
        let ids: std::collections::HashSet<_> = children.iter().map(|c| c.id.clone()).collect();
        assert!(ids.contains(&tool_a.id));
        assert!(ids.contains(&tool_b.id));

        // Tool runtimes have no children.
        let empty = store.list_children(&tool_a.id).await.expect("empty");
        assert!(empty.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_thread_returns_root_and_descendants() {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.expect("root");

        let tool = AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).expect("tool");
        store.insert(tool.clone()).await.expect("tool");

        let other = fresh_root("t2");
        store.insert(other.clone()).await.expect("other");

        let t1 = store.list_by_thread(&thread("t1")).await.expect("list t1");
        let ids: std::collections::HashSet<_> = t1.iter().map(|t| t.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&root.id));
        assert!(ids.contains(&tool.id));

        let t2 = store.list_by_thread(&thread("t2")).await.expect("list t2");
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, other.id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_status_reflects_current_state() {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.expect("insert");

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .expect("pending");
        assert_eq!(pending.len(), 1);

        // Promote to running.
        let running = root
            .clone()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .expect("running");
        store.update(running.clone()).await.expect("update");

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .expect("pending after");
        assert!(pending.is_empty());
        let running_list = store
            .list_by_status(TaskStatus::Running)
            .await
            .expect("running list");
        assert_eq!(running_list.len(), 1);
        assert_eq!(running_list[0].id, running.id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn active_root_for_thread_returns_only_non_terminal_root() {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.expect("insert");

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .expect("active");
        assert_eq!(active.map(|t| t.id), Some(root.id.clone()));

        // Complete the root via the legal transition path.
        let running = root
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .expect("running");
        store.update(running.clone()).await.expect("update running");
        let done = running.complete(t_plus(2)).expect("complete");
        store.update(done).await.expect("update done");

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .expect("active after");
        assert!(active.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_rejects_second_active_root_on_same_thread() {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first).await.expect("first");
        let second = fresh_root("t1");
        let err = store.insert(second).await.unwrap_err();
        assert!(
            err.to_string().contains("already has active root"),
            "unexpected: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_new_root_after_previous_root_completes() {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first.clone()).await.expect("first");

        // Walk the first root to Completed through the legal path.
        let running = first
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .expect("running");
        store.update(running.clone()).await.expect("update running");
        let done = running.complete(t_plus(2)).expect("complete");
        store.update(done).await.expect("update done");

        // A brand-new root on the same thread must now be admissible.
        let second = fresh_root("t1");
        store.insert(second.clone()).await.expect("second");

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .expect("active");
        assert_eq!(active.map(|t| t.id), Some(second.id));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_tool_runtime_children_while_root_is_running() {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.expect("root");

        // Transition the root to Running.
        let running = root
            .clone()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .expect("running");
        store.update(running.clone()).await.expect("update running");

        // Insert a tool-runtime child under the still-active root.
        let tool =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 1).expect("tool");
        store.insert(tool.clone()).await.expect("tool insert");

        let children = store.list_children(&running.id).await.expect("children");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, tool.id);

        // The active root must still be reachable.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .expect("active")
            .expect("still active");
        assert_eq!(active.id, running.id);
        assert_eq!(active.status, TaskStatus::Running);
    }
}
