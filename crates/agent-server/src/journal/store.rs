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
//! enforces the **partial-unique "one blocking root per thread"** invariant,
//! which is the single most important durability guarantee the acquisition
//! API (Phase 2.3) will rely on.
//!
//! # Root admission and FIFO queueing (Phase 2.2)
//!
//! Phase 2.2 splits the "one active root per thread" rule along a finer
//! boundary than Phase 2.1's `is_active`: the slot is held only by
//! [`TaskStatus`]es that satisfy [`TaskStatus::blocks_root_admission`], which
//! excludes the `Queued` state. Concretely:
//!
//! - At most one [`TaskKind::RootTurn`] per thread may sit in
//!   `Pending | Running | WaitingOnChildren | AwaitingConfirmation`.
//! - Any number of additional root turns may coexist in [`TaskStatus::Queued`]
//!   while that slot is held. They promote in deterministic FIFO order
//!   (`created_at`, tiebroken by `id`) as the slot frees up.
//! - Tool-runtime children are never gated by this slot — they are leaf
//!   tasks under whatever root currently holds the slot.
//!
//! The Phase 2.2 admission entry points are [`AgentTaskStore::submit_root_turn`]
//! and [`AgentTaskStore::promote_next_queued_root`]. Callers should use
//! `submit_root_turn` instead of `insert` for every externally-submitted
//! root turn so queueing is applied automatically.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use time::OffsetDateTime;
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
    /// [`TaskKind::RootTurn`] in a status that satisfies
    /// [`TaskStatus::blocks_root_admission`], must not collide with an
    /// existing blocking root on the same thread.
    ///
    /// # Errors
    /// Returns an error if the row fails validation, if a row with the
    /// same `id` already exists, or if the blocking-root invariant would
    /// be violated.
    ///
    /// Prefer [`AgentTaskStore::submit_root_turn`] when inserting
    /// externally-submitted root turns: it applies the FIFO queue rule
    /// automatically instead of rejecting the submission when the slot
    /// is busy.
    async fn insert(&self, task: AgentTask) -> Result<()>;

    /// Durably admit a new root turn on a thread, respecting the
    /// same-thread FIFO queue.
    ///
    /// The supplied task must be a freshly-constructed
    /// [`TaskKind::RootTurn`] in [`TaskStatus::Pending`] — i.e. the output
    /// of [`AgentTask::new_root_turn`] (possibly with a custom retry budget
    /// but not yet mutated by any other state-transition helper). The store
    /// decides, atomically, whether to admit it as `Pending` (if the thread
    /// has no blocking root yet) or convert it to `Queued` (if a blocking
    /// root already holds the slot).
    ///
    /// Returns the admitted row **as persisted**, so callers can observe the
    /// final status without a second `get`. This is how Phase 2.2's
    /// admission contract is expressed in the trait surface:
    ///
    /// - If `task.thread_id` has no [`TaskStatus::blocks_root_admission`]
    ///   root, the row lands in [`TaskStatus::Pending`].
    /// - Otherwise, the row lands in [`TaskStatus::Queued`] behind every
    ///   previously queued root on the same thread. `created_at` (and, as a
    ///   tie-breaker, `id`) define the FIFO order.
    ///
    /// # Errors
    /// Returns an error if the task is not a freshly-constructed root turn,
    /// if the row fails validation, or if the underlying store write fails.
    async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask>;

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
    /// if the blocking-root invariant would be violated after the update.
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

    /// Return the currently blocking [`TaskKind::RootTurn`] for a thread,
    /// if one exists.
    ///
    /// "Blocking" means the row is a root turn whose status satisfies
    /// [`TaskStatus::blocks_root_admission`] (i.e. `Pending`, `Running`,
    /// `WaitingOnChildren`, or `AwaitingConfirmation`). This is the primary
    /// consumer of the partial-unique "one blocking root per thread"
    /// invariant. Queued roots are explicitly **not** returned here — use
    /// [`AgentTaskStore::list_queued_roots`] to see them.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>>;

    /// Return the queued root turns on a thread in strict FIFO order.
    ///
    /// Rows are ordered by `created_at` ascending, tiebroken by `id` so the
    /// order is deterministic even at sub-microsecond time resolution. The
    /// returned vector is empty if the thread has no queued roots.
    ///
    /// # Errors
    /// Returns an error if the store cannot be queried.
    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>>;

    /// Promote the FIFO head of the thread's queued-root list into
    /// [`TaskStatus::Pending`], if the thread has no blocking root.
    ///
    /// Returns `Some(task)` with the newly-promoted row when a promotion
    /// fired. Returns `None` if the thread has no queued roots to promote,
    /// or if the thread's active-root slot is still held by a blocking root
    /// (so no promotion is legal yet).
    ///
    /// This method is a no-op idempotent operation: callers may invoke it
    /// every time a root completes, fails, or cancels, without needing to
    /// know whether another queued root is waiting. Retries of the active
    /// root that release the lease back to `Pending` do **not** fire a
    /// promotion, because the active-root slot is still held — queued
    /// roots cannot overtake a retry.
    ///
    /// # Errors
    /// Returns an error if the promotion transition or the store write
    /// fails.
    async fn promote_next_queued_root(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>>;

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
    /// `kind = root_turn AND status.blocks_root_admission()`. Holds the
    /// id of whichever root currently occupies the thread's single
    /// active-root slot, if any. Queued roots are **not** stored here —
    /// they live in `queued_roots_by_thread` instead.
    active_root_by_thread: HashMap<ThreadId, AgentTaskId>,
    /// Per-thread FIFO queue of root turns waiting behind the active-root
    /// slot. Entries are stored as `(created_at, id)` so iteration is
    /// deterministic even if two roots share a timestamp. A `BTreeSet`
    /// gives ordered insert / iterate / remove in `O(log n)` without
    /// needing a custom priority queue.
    queued_roots_by_thread: HashMap<ThreadId, BTreeSet<(OffsetDateTime, AgentTaskId)>>,
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

        if task.kind == TaskKind::RootTurn {
            if task.status.blocks_root_admission() {
                self.active_root_by_thread
                    .insert(task.thread_id.clone(), task.id.clone());
            } else if task.status == TaskStatus::Queued {
                self.queued_roots_by_thread
                    .entry(task.thread_id.clone())
                    .or_default()
                    .insert((task.created_at, task.id.clone()));
            }
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

    /// Remove a task from the per-thread queued-root FIFO index, if it is
    /// a queued root.
    fn remove_queued_root_if_match(&mut self, task: &AgentTask) {
        if task.kind != TaskKind::RootTurn {
            return;
        }
        if let Some(queue) = self.queued_roots_by_thread.get_mut(&task.thread_id) {
            queue.remove(&(task.created_at, task.id.clone()));
            if queue.is_empty() {
                self.queued_roots_by_thread.remove(&task.thread_id);
            }
        }
    }

    /// Is the thread's active-root slot currently held by a blocking root?
    fn thread_has_blocking_root(&self, thread_id: &ThreadId) -> bool {
        self.active_root_by_thread.contains_key(thread_id)
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

        // Cross-row invariants for non-root tasks: the parent must already
        // exist in the store, and the child's `thread_id`, `root_id`, and
        // `depth` must match the parent row. These are enforced here (and
        // not by `AgentTask::validate`) because `validate` has no access to
        // the parent. Bypassing these checks would let a mutated or
        // deserialized child slip into the wrong `by_thread` / `by_parent`
        // bucket and silently corrupt list queries and the active-root
        // invariant.
        if let Some(parent_id) = &task.parent_id {
            let parent = inner.by_id.get(parent_id).ok_or_else(|| {
                anyhow!("insert rejected: child task references unknown parent {parent_id}")
            })?;
            if parent.kind.is_leaf() {
                let parent_kind = parent.kind;
                return Err(anyhow!(
                    "insert rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
                ));
            }
            if parent.thread_id != task.thread_id {
                let child_thread = &task.thread_id;
                let parent_thread = &parent.thread_id;
                return Err(anyhow!(
                    "insert rejected: child thread_id {child_thread} does not match parent thread_id {parent_thread}"
                ));
            }
            if parent.root_id != task.root_id {
                let child_root = &task.root_id;
                let parent_root = &parent.root_id;
                return Err(anyhow!(
                    "insert rejected: child root_id {child_root} does not match parent root_id {parent_root}"
                ));
            }
            let expected_depth = parent.depth.saturating_add(1);
            if task.depth != expected_depth {
                let child_depth = task.depth;
                let parent_depth = parent.depth;
                return Err(anyhow!(
                    "insert rejected: child depth {child_depth} must be parent.depth + 1 ({parent_depth} + 1 = {expected_depth})"
                ));
            }
        }

        // Partial-unique: one *blocking* root per thread.
        //
        // `blocks_root_admission()` excludes [`TaskStatus::Queued`], so a
        // fresh root turn may be inserted in [`TaskStatus::Queued`] even
        // when another root already holds the slot — that is how
        // [`AgentTaskStore::submit_root_turn`] persists queued
        // submissions. A caller that bypasses `submit_root_turn` and
        // tries to insert a *blocking* root while the slot is already
        // held is rejected here.
        if task.kind == TaskKind::RootTurn
            && task.status.blocks_root_admission()
            && let Some(existing) = inner.active_root_by_thread.get(&task.thread_id)
        {
            let thread_id = &task.thread_id;
            return Err(anyhow!(
                "insert rejected: thread {thread_id} already has active root task {existing}"
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

        let old = inner.by_id.get(&task.id).cloned().ok_or_else(|| {
            let id = &task.id;
            anyhow!("update rejected: task {id} does not exist")
        })?;

        // Row invariants: `kind`, `parent_id`, `root_id`, `depth`, `thread_id`,
        // `created_at`, and `max_attempts` are fixed at construction time and
        // must never change across updates. Silently accepting a mutation here
        // would corrupt the secondary indexes (`by_thread`, `by_parent`,
        // `active_root_by_thread`) because those are keyed by the old values.
        if old.kind != task.kind {
            let old_kind = old.kind;
            let new_kind = task.kind;
            return Err(anyhow!(
                "update rejected: task kind is immutable (was {old_kind:?}, got {new_kind:?})"
            ));
        }
        if old.parent_id != task.parent_id {
            let old_parent = &old.parent_id;
            let new_parent = &task.parent_id;
            return Err(anyhow!(
                "update rejected: parent_id is immutable (was {old_parent:?}, got {new_parent:?})"
            ));
        }
        if old.root_id != task.root_id {
            let old_root = &old.root_id;
            let new_root = &task.root_id;
            return Err(anyhow!(
                "update rejected: root_id is immutable (was {old_root}, got {new_root})"
            ));
        }
        if old.depth != task.depth {
            let old_depth = old.depth;
            let new_depth = task.depth;
            return Err(anyhow!(
                "update rejected: depth is immutable (was {old_depth}, got {new_depth})"
            ));
        }
        if old.thread_id != task.thread_id {
            let old_thread = &old.thread_id;
            let new_thread = &task.thread_id;
            return Err(anyhow!(
                "update rejected: thread_id is immutable (was {old_thread}, got {new_thread})"
            ));
        }
        if old.created_at != task.created_at {
            return Err(anyhow!("update rejected: created_at is immutable"));
        }
        if old.max_attempts != task.max_attempts {
            let old_max = old.max_attempts;
            let new_max = task.max_attempts;
            return Err(anyhow!(
                "update rejected: max_attempts is immutable (was {old_max}, got {new_max})"
            ));
        }

        // Partial-unique check: if the new row is a blocking root, make
        // sure no *other* row already holds the active-root slot on the
        // same thread. Queued roots skip this check because they don't
        // occupy the slot.
        if task.kind == TaskKind::RootTurn
            && task.status.blocks_root_admission()
            && let Some(current) = inner.active_root_by_thread.get(&task.thread_id)
            && *current != task.id
        {
            let thread_id = &task.thread_id;
            return Err(anyhow!(
                "update rejected: thread {thread_id} already has a different active root task {current}"
            ));
        }

        // Drop the old row from every secondary index; the primary-key
        // entry will be overwritten below.
        inner.remove_from_status_index(&old);

        // Thread / parent indexes are append-only and keyed by `id`, so the
        // old entries stay in place and still point at the (now-updated)
        // row in `by_id`. This is safe because `thread_id` / `parent_id` are
        // row invariants enforced above and never mutate across updates.

        inner
            .by_status
            .entry(task.status)
            .or_default()
            .insert(task.id.clone());

        // Rebalance the per-thread root indexes. The invariants we need:
        //
        // * `active_root_by_thread[thread]` is set ⇔ the new row is a
        //   root turn in a `blocks_root_admission` state.
        // * `queued_roots_by_thread[thread]` contains `(created_at, id)`
        //   ⇔ the new row is a queued root turn.
        //
        // `kind` and `thread_id` are row invariants (guarded above), so
        // the old row can only mutate between root-turn statuses here.
        if task.kind == TaskKind::RootTurn {
            // Drop the old classification first so we don't leave a stale
            // entry behind on any transition.
            if old.status.blocks_root_admission() {
                inner.remove_active_root_if_match(&old);
            } else if old.status == TaskStatus::Queued {
                inner.remove_queued_root_if_match(&old);
            }

            // Re-register under the new classification.
            if task.status.blocks_root_admission() {
                inner
                    .active_root_by_thread
                    .insert(task.thread_id.clone(), task.id.clone());
            } else if task.status == TaskStatus::Queued {
                inner
                    .queued_roots_by_thread
                    .entry(task.thread_id.clone())
                    .or_default()
                    .insert((task.created_at, task.id.clone()));
            }
            // Terminal statuses simply drop out of both indexes.
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

    async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
        // Shape gate: only Phase 2.2 root admission flows through here,
        // and only with freshly-constructed rows. Everything else must
        // go through `insert` / `update` and pays for the tighter
        // per-call invariants there.
        if task.kind != TaskKind::RootTurn {
            let kind = task.kind;
            return Err(anyhow!(
                "submit_root_turn rejected: expected root_turn, got {kind:?}"
            ));
        }
        if task.status != TaskStatus::Pending {
            let status = task.status;
            return Err(anyhow!(
                "submit_root_turn rejected: new root must start in Pending (got {status:?})"
            ));
        }
        if task.attempt != 0 {
            let attempt = task.attempt;
            return Err(anyhow!(
                "submit_root_turn rejected: new root must have attempt == 0 (got {attempt})"
            ));
        }
        if !task.is_root() {
            return Err(anyhow!("submit_root_turn rejected: task must be a root"));
        }
        task.validate()
            .context("submit_root_turn rejected: task failed schema validation")?;

        let mut inner = self.inner.write().await;

        if inner.by_id.contains_key(&task.id) {
            let id = &task.id;
            return Err(anyhow!(
                "submit_root_turn rejected: task id {id} already exists"
            ));
        }

        // Decide the admission target atomically under the write lock so
        // two concurrent submissions on the same thread always serialize
        // and end up in a deterministic FIFO order.
        let admitted = if inner.thread_has_blocking_root(&task.thread_id) {
            // Convert the incoming row from `Pending` to `Queued` so it
            // lives on the same-thread FIFO queue behind the blocking
            // root. Pass `created_at` as `now` so the queued row's
            // `updated_at` equals its `created_at`, preserving the
            // original submission timestamp in the audit field rather
            // than recording the queuing-decision time. FIFO ordering is
            // unaffected by this choice — the queue index is keyed on
            // `(created_at, id)` and `created_at` is immutable.
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")?
        } else {
            task
        };

        inner.add_to_indexes(&admitted);
        inner.by_id.insert(admitted.id.clone(), admitted.clone());
        drop(inner);
        Ok(admitted)
    }

    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let inner = self.inner.read().await;
        let result: Vec<AgentTask> = inner
            .queued_roots_by_thread
            .get(thread_id)
            .into_iter()
            .flatten()
            .filter_map(|(_, id)| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        Ok(result)
    }

    async fn promote_next_queued_root(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut inner = self.inner.write().await;

        // If the slot is still held, the active root is retrying or
        // waiting on something and no promotion may fire — retries of
        // the active root must never be overtaken by queued roots.
        if inner.thread_has_blocking_root(thread_id) {
            return Ok(None);
        }

        // Pop the FIFO head. `BTreeSet` iterates in ascending key order,
        // and the key is `(created_at, id)` so the earliest submission
        // wins, with `id` breaking ties deterministically.
        let head_key = inner
            .queued_roots_by_thread
            .get(thread_id)
            .and_then(|q| q.iter().next().cloned());
        let Some((created_at, id)) = head_key else {
            return Ok(None);
        };

        // Load the row, run the pure promotion transition, and commit.
        let queued_row = inner
            .by_id
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("promote rejected: queue head {id} missing from by_id"))?;
        let promoted = queued_row
            .promote_to_pending(now)
            .context("promote rejected: promotion transition failed")?;

        // Reindex: drop from the queued index, add to the active-root
        // slot, swap the status bucket.
        if let Some(queue) = inner.queued_roots_by_thread.get_mut(thread_id) {
            queue.remove(&(created_at, id.clone()));
            if queue.is_empty() {
                inner.queued_roots_by_thread.remove(thread_id);
            }
        }
        if let Some(set) = inner.by_status.get_mut(&TaskStatus::Queued) {
            set.remove(&id);
            if set.is_empty() {
                inner.by_status.remove(&TaskStatus::Queued);
            }
        }
        inner
            .by_status
            .entry(TaskStatus::Pending)
            .or_default()
            .insert(id.clone());
        inner
            .active_root_by_thread
            .insert(thread_id.clone(), id.clone());
        inner.by_id.insert(id, promoted.clone());
        drop(inner);
        Ok(Some(promoted))
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
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t0(), 3)
    }

    /// Build a fresh root turn on `thread` with `created_at` at `t0 +
    /// secs` seconds. Used by Phase 2.2 FIFO tests that need two roots
    /// with deterministic, distinct wall-clock timestamps.
    fn fresh_root_at(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread(name), t_plus(secs), 3)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_and_get_round_trip() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.context("insert")?;
        let got = store
            .get(&id)
            .await
            .context("get")?
            .context("task exists")?;
        assert_eq!(got, task);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_invalid_task() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let mut task = fresh_root("t1");
        task.status = TaskStatus::Running; // invalid: missing lease fields
        let err = store.insert(task).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_invalid_task() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let task = fresh_root("t1");
        let id = task.id.clone();
        store.insert(task.clone()).await.context("insert")?;

        // Load and corrupt the row.
        let mut bad = store
            .get(&id)
            .await
            .context("get")?
            .context("task exists")?;
        bad.status = TaskStatus::Running; // still no lease fields -> invalid
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_children_returns_only_direct_children() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert root")?;

        let tool_a =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("a")?;
        let tool_b =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("b")?;
        store.insert(tool_a.clone()).await.context("insert a")?;
        store.insert(tool_b.clone()).await.context("insert b")?;

        let children = store.list_children(&root.id).await.context("children")?;
        assert_eq!(children.len(), 2);
        let ids: std::collections::HashSet<_> = children.iter().map(|c| c.id.clone()).collect();
        assert!(ids.contains(&tool_a.id));
        assert!(ids.contains(&tool_b.id));

        // Tool runtimes have no children.
        let empty = store.list_children(&tool_a.id).await.context("empty")?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_thread_returns_root_and_descendants() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        let tool =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("tool")?;
        store.insert(tool.clone()).await.context("tool")?;

        let other = fresh_root("t2");
        store.insert(other.clone()).await.context("other")?;

        let t1 = store
            .list_by_thread(&thread("t1"))
            .await
            .context("list t1")?;
        let ids: std::collections::HashSet<_> = t1.iter().map(|t| t.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&root.id));
        assert!(ids.contains(&tool.id));

        let t2 = store
            .list_by_thread(&thread("t2"))
            .await
            .context("list t2")?;
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, other.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_by_status_reflects_current_state() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .context("pending")?;
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
            .context("running")?;
        store.update(running.clone()).await.context("update")?;

        let pending = store
            .list_by_status(TaskStatus::Pending)
            .await
            .context("pending after")?;
        assert!(pending.is_empty());
        let running_list = store
            .list_by_status(TaskStatus::Running)
            .await
            .context("running list")?;
        assert_eq!(running_list.len(), 1);
        assert_eq!(running_list[0].id, running.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn active_root_for_thread_returns_only_non_terminal_root() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert_eq!(active.map(|t| t.id), Some(root.id.clone()));

        // Complete the root via the legal transition path.
        let running = root
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let done = running.complete(t_plus(2)).context("complete")?;
        store.update(done).await.context("update done")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active after")?;
        assert!(active.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_rejects_second_active_root_on_same_thread() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first).await.context("first")?;
        let second = fresh_root("t1");
        let err = store.insert(second).await.unwrap_err();
        assert!(
            err.to_string().contains("already has active root"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_new_root_after_previous_root_completes() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store.insert(first.clone()).await.context("first")?;

        // Walk the first root to Completed through the legal path.
        let running = first
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;
        let done = running.complete(t_plus(2)).context("complete")?;
        store.update(done).await.context("update done")?;

        // A brand-new root on the same thread must now be admissible.
        let second = fresh_root("t1");
        store.insert(second.clone()).await.context("second")?;

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert_eq!(active.map(|t| t.id), Some(second.id));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_allows_tool_runtime_children_while_root_is_running() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        // Transition the root to Running.
        let running = root
            .clone()
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store
            .update(running.clone())
            .await
            .context("update running")?;

        // Insert a tool-runtime child under the still-active root.
        let tool =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 1).context("tool")?;
        store.insert(tool.clone()).await.context("tool insert")?;

        let children = store.list_children(&running.id).await.context("children")?;
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, tool.id);

        // The active root must still be reachable.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("still active")?;
        assert_eq!(active.id, running.id);
        assert_eq!(active.status, TaskStatus::Running);
        Ok(())
    }

    // ── regression tests for row-invariant guards ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_kind_mutation_even_when_status_is_active() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        // Mutate kind: RootTurn -> ToolRuntime while keeping status active.
        // Without the guard this would corrupt active_root_by_thread: the
        // cleanup branch only fires when !task.status.is_active(), and the
        // re-registration branch skips non-RootTurn kinds, leaving a stale
        // root-id pointer on the thread.
        let mut mutated = root.clone();
        mutated.kind = TaskKind::ToolRuntime;
        // Give the "child" the shape a ToolRuntime would have so only the
        // kind mutation is under test (parent_id/root_id still satisfy
        // validate() because this is still the root row).
        let err = store.update(mutated).await.unwrap_err();
        assert!(
            err.to_string().contains("kind is immutable"),
            "unexpected: {err}"
        );

        // Regression check: the active-root pointer and its kind are intact.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("still active")?;
        assert_eq!(active.id, root.id);
        assert_eq!(active.kind, TaskKind::RootTurn);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rejects_mutation_of_any_row_invariant() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("insert")?;

        // thread_id
        let mut bad = root.clone();
        bad.thread_id = thread("t-other");
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("thread_id is immutable"),
            "unexpected: {err}"
        );

        // parent_id (root has None; setting Some must be rejected)
        let mut bad = root.clone();
        bad.parent_id = Some(AgentTaskId::new());
        let err = store.update(bad).await.unwrap_err();
        // parent_id guard fires before validate() on the store path,
        // but validate() would also reject this as RootHasParent. Accept
        // either wording — the important thing is the row is not written.
        assert!(
            err.to_string().contains("parent_id is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // root_id
        let mut bad = root.clone();
        bad.root_id = AgentTaskId::new();
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("root_id is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // depth
        let mut bad = root.clone();
        bad.depth = 1;
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("depth is immutable")
                || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // created_at
        let mut bad = root.clone();
        bad.created_at = t_plus(9999);
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("created_at is immutable"),
            "unexpected: {err}"
        );

        // max_attempts
        let mut bad = root;
        bad.max_attempts = 99;
        let err = store.update(bad).await.unwrap_err();
        assert!(
            err.to_string().contains("max_attempts is immutable"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_unknown_parent() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let ghost_root = fresh_root("t1");
        // Build a child referencing ghost_root without ever inserting it.
        let child = AgentTask::new_child(&ghost_root, TaskKind::ToolRuntime, t_plus(1), 1)
            .context("child")?;
        let err = store.insert(child).await.unwrap_err();
        assert!(
            err.to_string().contains("unknown parent"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_mutated_thread_id() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.thread_id = thread("t-other");
        let err = store.insert(child).await.unwrap_err();
        assert!(err.to_string().contains("thread_id"), "unexpected: {err}");

        // Regression check: the wrong thread must not now bucket a child.
        let other = store
            .list_by_thread(&thread("t-other"))
            .await
            .context("list other")?;
        assert!(other.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_child_with_mutated_root_id_or_depth() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;

        // Mutated root_id
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.root_id = AgentTaskId::new();
        // validate() already catches this as ChildRootIdMismatch is not
        // the only path; the store's cross-row check fires first.
        let err = store.insert(child).await.unwrap_err();
        assert!(
            err.to_string().contains("root_id") || err.to_string().contains("schema validation"),
            "unexpected: {err}"
        );

        // Mutated depth
        let mut child =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("child")?;
        child.depth = 42;
        let err = store.insert(child).await.unwrap_err();
        assert!(err.to_string().contains("depth"), "unexpected: {err}");
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 2.2 — Root submission queue and FIFO promotion (ENG-7916)
    // ──────────────────────────────────────────────────────────────
    //
    // These tests exercise the queue admission contract across every
    // blocking active-root state (`Pending`, `Running`,
    // `WaitingOnChildren`, `AwaitingConfirmation`), FIFO promotion order,
    // retry safety, and tool-child isolation. They form the acceptance
    // suite for Phase 2.2 and should cover every rule in the Linear
    // issue.

    /// Helper: mark a task Running via the legal transition path and
    /// persist the resulting row. Used to put a thread's active root
    /// into a blocking non-Pending state in tests.
    async fn running_root(store: &InMemoryAgentTaskStore, root: AgentTask) -> Result<AgentTask> {
        let running = root
            .mark_running(
                WorkerId::from_string("w1"),
                LeaseId::from_string("l1"),
                t_plus(60),
                t_plus(1),
            )
            .context("running")?;
        store.update(running.clone()).await.context("update")?;
        Ok(running)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_admits_first_root_as_pending() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        let admitted = store
            .submit_root_turn(first.clone())
            .await
            .context("submit")?;
        assert_eq!(admitted.status, TaskStatus::Pending);
        assert_eq!(admitted.id, first.id);

        // The active-root slot is now held and queued list is empty.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);
        assert_eq!(active.status, TaskStatus::Pending);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert!(queued.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_queues_second_root_behind_pending() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        let admitted = store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        assert_eq!(admitted.status, TaskStatus::Queued);
        assert_eq!(admitted.id, second.id);

        // The active-root slot is still held by the first root.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);
        assert_eq!(queued[0].status, TaskStatus::Queued);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_queues_behind_every_blocking_state() -> Result<()> {
        for &(label, setup) in &[
            ("Pending", 0i64),
            ("Running", 1),
            ("WaitingOnChildren", 2),
            ("AwaitingConfirmation", 3),
        ] {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            store
                .submit_root_turn(first.clone())
                .await
                .context("first")?;

            // Drive `first` into the blocking state under test. Each
            // branch ends with the active root in a non-Pending blocking
            // status, so a second submission must queue.
            match setup {
                0 => {
                    // Already Pending — no-op.
                }
                1 => {
                    running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                }
                2 => {
                    let running = running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                    let waiting = running
                        .wait_on_children(2, t_plus(2))
                        .context("wait_on_children")?;
                    store.update(waiting).await.context("update waiting")?;
                }
                3 => {
                    let running = running_root(&store, first.clone())
                        .await
                        .context("drive running")?;
                    let awaiting = running
                        .await_confirmation(t_plus(2))
                        .context("await_confirmation")?;
                    store.update(awaiting).await.context("update awaiting")?;
                }
                _ => unreachable!(),
            }

            let second = fresh_root_at("t1", 10);
            let admitted = store
                .submit_root_turn(second.clone())
                .await
                .context("second submit")?;
            assert_eq!(
                admitted.status,
                TaskStatus::Queued,
                "second root did not queue behind blocking state {label}",
            );

            let queued = store
                .list_queued_roots(&thread("t1"))
                .await
                .context("queued")?;
            assert_eq!(queued.len(), 1, "queue length wrong for state {label}");
            assert_eq!(
                queued[0].id, second.id,
                "queue head wrong for state {label}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_promote_in_fifo_order_after_active_completes() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);
        let third = fresh_root_at("t1", 2);

        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        store
            .submit_root_turn(third.clone())
            .await
            .context("third")?;

        // Walk the first root to Completed.
        let running = running_root(&store, first.clone())
            .await
            .context("drive running")?;
        let done = running.complete(t_plus(10)).context("complete")?;
        store.update(done).await.context("update done")?;

        // Queued roots still in FIFO order in `list_queued_roots`.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 2);
        assert_eq!(queued[0].id, second.id);
        assert_eq!(queued[1].id, third.id);

        // Promote the head: it must be the *oldest* queued root.
        let promoted = store
            .promote_next_queued_root(&thread("t1"), t_plus(11))
            .await
            .context("promote")?
            .context("promotion fired")?;
        assert_eq!(promoted.id, second.id);
        assert_eq!(promoted.status, TaskStatus::Pending);

        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, second.id);

        // The second promotion must not fire while the slot is held.
        let no_promotion = store
            .promote_next_queued_root(&thread("t1"), t_plus(12))
            .await
            .context("promote blocked")?;
        assert!(no_promotion.is_none());

        // Walk `second` to Completed, then promote `third`.
        let running2 = running_root(&store, promoted.clone())
            .await
            .context("drive running 2")?;
        let done2 = running2.complete(t_plus(20)).context("complete 2")?;
        store.update(done2).await.context("update done 2")?;

        let promoted2 = store
            .promote_next_queued_root(&thread("t1"), t_plus(21))
            .await
            .context("promote 2")?
            .context("promotion 2 fired")?;
        assert_eq!(promoted2.id, third.id);

        // Queue is now drained.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued after")?;
        assert!(queued.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_is_noop_when_queue_is_empty() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;

        // Complete without any queued roots behind.
        let running = running_root(&store, first.clone())
            .await
            .context("drive running")?;
        let done = running.complete(t_plus(10)).context("complete")?;
        store.update(done).await.context("update done")?;

        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(11))
            .await
            .context("promote empty")?;
        assert!(nothing.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_is_noop_while_active_root_is_still_blocking() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        store.submit_root_turn(first).await.context("first")?;
        store.submit_root_turn(second).await.context("second")?;

        // Slot still held by `first` in Pending: promotion must refuse.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(2))
            .await
            .context("promote while blocking")?;
        assert!(nothing.is_none());

        // Queued list is still intact.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retry_of_active_root_does_not_let_queued_roots_overtake() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);

        let admitted_first = store
            .submit_root_turn(first.clone())
            .await
            .context("first")?;
        store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;

        // Lease the first, then release it back to Pending (a retry).
        // `release_lease` keeps `attempt` at 1 but clears the lease.
        let running = running_root(&store, admitted_first.clone())
            .await
            .context("drive running")?;
        let released = running.release_lease(t_plus(5)).context("release")?;
        assert_eq!(released.status, TaskStatus::Pending);
        assert_eq!(released.attempt, 1);
        store.update(released).await.context("update released")?;

        // The active-root slot is still held by the first root — the
        // promotion attempt must be a no-op so queued roots cannot
        // overtake a retry.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(6))
            .await
            .context("promote during retry")?;
        assert!(nothing.is_none());

        // And the second root is still queued, not promoted.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, first.id);
        assert_eq!(active.status, TaskStatus::Pending);

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);

        // Running → complete the first; now the second must promote.
        let running2 = running_root(&store, active)
            .await
            .context("drive running again")?;
        let done = running2.complete(t_plus(7)).context("complete")?;
        store.update(done).await.context("update done")?;

        let promoted = store
            .promote_next_queued_root(&thread("t1"), t_plus(8))
            .await
            .context("promote after complete")?
            .context("promotion fired")?;
        assert_eq!(promoted.id, second.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_promote_after_failure_and_cancellation() -> Result<()> {
        // Failure.
        {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            let second = fresh_root_at("t1", 1);
            store.submit_root_turn(first.clone()).await.context("f1")?;
            store.submit_root_turn(second.clone()).await.context("q1")?;

            let running = running_root(&store, first.clone())
                .await
                .context("running")?;
            let failed = running.fail("boom".into(), t_plus(10)).context("fail")?;
            store.update(failed).await.context("update failed")?;

            let promoted = store
                .promote_next_queued_root(&thread("t1"), t_plus(11))
                .await
                .context("promote after fail")?
                .context("promotion fired")?;
            assert_eq!(promoted.id, second.id);
        }

        // Cancellation (straight from Pending).
        {
            let store = InMemoryAgentTaskStore::new();
            let first = fresh_root_at("t1", 0);
            let second = fresh_root_at("t1", 1);
            store.submit_root_turn(first.clone()).await.context("f2")?;
            store.submit_root_turn(second.clone()).await.context("q2")?;

            let cancelled = first.cancel(t_plus(10)).context("cancel")?;
            store.update(cancelled).await.context("update cancelled")?;

            let promoted = store
                .promote_next_queued_root(&thread("t1"), t_plus(11))
                .await
                .context("promote after cancel")?
                .context("promotion fired")?;
            assert_eq!(promoted.id, second.id);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_runtime_children_never_block_root_admission() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let root = fresh_root("t1");
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit root")?;

        // Drive root to Running and spawn a tool-runtime child under it.
        let running = running_root(&store, root.clone())
            .await
            .context("drive running")?;
        let tool =
            AgentTask::new_child(&running, TaskKind::ToolRuntime, t_plus(2), 1).context("tool")?;
        store.insert(tool.clone()).await.context("insert tool")?;

        // A second root submission must queue behind `running`, not the
        // tool child — tool children do not participate in root
        // admission.
        let second = fresh_root_at("t1", 3);
        let admitted = store
            .submit_root_turn(second.clone())
            .await
            .context("second")?;
        assert_eq!(admitted.status, TaskStatus::Queued);

        // The active-root slot is exactly the root, not the tool.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?
            .context("slot held")?;
        assert_eq!(active.id, running.id);
        assert_eq!(active.kind, TaskKind::RootTurn);

        // Now complete the tool child. Tool children completing must
        // not by themselves promote queued roots — the root is still
        // running. We drive the tool to Running + Completed to prove
        // the full tool lifecycle doesn't leak into root admission.
        let tool_running = tool
            .mark_running(
                WorkerId::from_string("w-tool"),
                LeaseId::from_string("l-tool"),
                t_plus(60),
                t_plus(3),
            )
            .context("tool running")?;
        store
            .update(tool_running.clone())
            .await
            .context("update tool running")?;
        let tool_done = tool_running.complete(t_plus(4)).context("tool done")?;
        store.update(tool_done).await.context("update tool done")?;

        // Queued list still holds the second root; the active slot is
        // still held by the running root.
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, second.id);

        // Promotion attempt still a no-op while the root itself is
        // blocking.
        let nothing = store
            .promote_next_queued_root(&thread("t1"), t_plus(5))
            .await
            .context("promote blocked")?;
        assert!(nothing.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_roots_do_not_cross_thread_boundaries() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        // Thread A: blocking root + one queued root.
        let a_first = fresh_root_at("a", 0);
        let a_second = fresh_root_at("a", 1);
        store
            .submit_root_turn(a_first.clone())
            .await
            .context("a1")?;
        store
            .submit_root_turn(a_second.clone())
            .await
            .context("a2")?;

        // Thread B: its first root must admit as Pending, not queue
        // behind thread A's root.
        let b_first = fresh_root_at("b", 0);
        let b_admitted = store
            .submit_root_turn(b_first.clone())
            .await
            .context("b1")?;
        assert_eq!(b_admitted.status, TaskStatus::Pending);

        // Each thread sees only its own queue.
        let a_queued = store
            .list_queued_roots(&thread("a"))
            .await
            .context("a queue")?;
        assert_eq!(a_queued.len(), 1);
        assert_eq!(a_queued[0].id, a_second.id);

        let b_queued = store
            .list_queued_roots(&thread("b"))
            .await
            .context("b queue")?;
        assert!(b_queued.is_empty());

        // Completing thread B's root must not fire a promotion on A.
        let b_running = running_root(&store, b_first.clone())
            .await
            .context("b running")?;
        let b_done = b_running.complete(t_plus(10)).context("b done")?;
        store.update(b_done).await.context("update b done")?;

        let a_active = store
            .active_root_for_thread(&thread("a"))
            .await
            .context("a active")?
            .context("still held")?;
        assert_eq!(a_active.id, a_first.id);
        let a_queued = store
            .list_queued_roots(&thread("a"))
            .await
            .context("a queue after")?;
        assert_eq!(a_queued.len(), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_root_turn_rejects_non_root_or_non_pending_input() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();

        // Wrong kind: tool runtime.
        let root = fresh_root("t1");
        store.insert(root.clone()).await.context("root")?;
        let tool =
            AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 1).context("tool")?;
        let err = store.submit_root_turn(tool).await.unwrap_err();
        assert!(
            err.to_string().contains("expected root_turn"),
            "unexpected: {err}"
        );

        // Wrong status: Queued (caller trying to pre-queue).
        let mut queued = fresh_root_at("t2", 0);
        queued.status = TaskStatus::Queued;
        queued.updated_at = queued.created_at;
        let err = store.submit_root_turn(queued).await.unwrap_err();
        assert!(
            err.to_string().contains("must start in Pending"),
            "unexpected: {err}"
        );

        // Wrong attempt counter: pretending to re-submit.
        let mut retried = fresh_root("t3");
        retried.attempt = 1;
        let err = store.submit_root_turn(retried).await.unwrap_err();
        assert!(
            err.to_string().contains("attempt == 0"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_queued_roots_is_deterministic_even_with_same_timestamp() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let blocker = fresh_root_at("t1", 0);
        store.submit_root_turn(blocker).await.context("blocker")?;

        // Submit two roots with the exact same `created_at` — FIFO
        // order must still be deterministic, tie-broken by `id`.
        let a = fresh_root_at("t1", 5);
        let b = fresh_root_at("t1", 5);
        store.submit_root_turn(a.clone()).await.context("a")?;
        store.submit_root_turn(b.clone()).await.context("b")?;

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 2);
        // Order must match the ascending `id` order for reproducible
        // scheduling at sub-microsecond tiebreaks.
        let (lo, hi) = if a.id <= b.id { (&a, &b) } else { (&b, &a) };
        assert_eq!(queued[0].id, lo.id);
        assert_eq!(queued[1].id, hi.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_root_cancellation_removes_it_from_queue() -> Result<()> {
        let store = InMemoryAgentTaskStore::new();
        let blocker = fresh_root_at("t1", 0);
        let queued = fresh_root_at("t1", 1);
        store.submit_root_turn(blocker).await.context("blocker")?;
        let admitted = store
            .submit_root_turn(queued.clone())
            .await
            .context("queued")?;
        assert_eq!(admitted.status, TaskStatus::Queued);

        // Cancel the queued row directly: the queue index must drop it.
        let cancelled = admitted.cancel(t_plus(2)).context("cancel")?;
        store
            .update(cancelled.clone())
            .await
            .context("update cancelled")?;

        let remaining = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queue after cancel")?;
        assert!(remaining.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_rejects_adding_another_blocking_root_bypassing_submit() -> Result<()> {
        // `submit_root_turn` is the FIFO admission entry point, but the
        // raw `insert` path must still enforce the partial-unique
        // invariant so callers that bypass `submit_root_turn` can't
        // accidentally push two blocking roots onto the same thread.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root("t1");
        store
            .submit_root_turn(first)
            .await
            .context("submit first")?;

        let second = fresh_root("t1");
        let err = store.insert(second).await.unwrap_err();
        assert!(
            err.to_string().contains("already has active root"),
            "unexpected: {err}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_allows_raw_queued_root_next_to_blocking_root() -> Result<()> {
        // Even outside `submit_root_turn`, a caller with a pre-queued
        // row must be able to insert it alongside a blocking root,
        // because `Queued` does not occupy the slot. This is how
        // recovery paths will rehydrate queued roots from durable
        // storage after a crash.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        store
            .submit_root_turn(first)
            .await
            .context("submit first")?;

        // Build a queued row by hand (mirrors the shape after
        // `admit_as_queued`): Pending → Queued.
        let raw_second = fresh_root_at("t1", 1);
        let queued_second = raw_second
            .admit_as_queued(t_plus(1))
            .context("admit queued")?;
        store
            .insert(queued_second.clone())
            .await
            .context("insert queued")?;

        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("list queue")?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, queued_second.id);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_rebalances_indexes_across_queued_and_pending() -> Result<()> {
        // Cover every transition that mutates a root-turn row's
        // classification across the two Phase 2.2 indexes:
        //
        //   1. Pending (active slot) → Completed (drops both)
        //   2. Queued → Pending (via update, after slot freed)
        //   3. Pending → WaitingOnChildren → Pending (slot stays held)
        //
        // This is a smoke test for the update() rebalancing logic; the
        // higher-level tests above already prove the queue behaviour
        // end-to-end.
        let store = InMemoryAgentTaskStore::new();
        let first = fresh_root_at("t1", 0);
        let second = fresh_root_at("t1", 1);
        store
            .submit_root_turn(first.clone())
            .await
            .context("submit first")?;
        let admitted_second = store
            .submit_root_turn(second.clone())
            .await
            .context("submit second")?;
        assert_eq!(admitted_second.status, TaskStatus::Queued);

        // Drive the first to Completed.
        let running = running_root(&store, first.clone())
            .await
            .context("running")?;
        let done = running.complete(t_plus(5)).context("complete")?;
        store.update(done).await.context("update done")?;

        // The active-root slot is now free; the queued index still holds
        // the second root. Queued rows persist until promotion is
        // explicitly triggered (via `promote_next_queued_root` or via
        // `update` rebalancing in this test).
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active")?;
        assert!(active.is_none());
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued")?;
        assert_eq!(queued.len(), 1);

        // Now manually promote the queued row via `update` (rather than
        // `promote_next_queued_root`). This exercises the same index
        // rebalancing that the dedicated promotion path uses.
        let promoted_row = admitted_second
            .promote_to_pending(t_plus(6))
            .context("promote to pending")?;
        store
            .update(promoted_row.clone())
            .await
            .context("update promoted")?;

        // After the manual promotion, the slot must be held and the
        // queue must be empty.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active after")?
            .context("slot held")?;
        assert_eq!(active.id, second.id);
        let queued = store
            .list_queued_roots(&thread("t1"))
            .await
            .context("queued after")?;
        assert!(queued.is_empty());

        // Exercise Pending → WaitingOnChildren → Pending.
        let running2 = running_root(&store, active).await.context("running 2")?;
        let waiting = running2.wait_on_children(1, t_plus(7)).context("wait")?;
        store
            .update(waiting.clone())
            .await
            .context("update waiting")?;

        // The slot is still held (WaitingOnChildren is blocking).
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active waiting")?
            .context("still held")?;
        assert_eq!(active.status, TaskStatus::WaitingOnChildren);

        let resolved = waiting.child_resolved(t_plus(8)).context("resolved")?;
        assert_eq!(resolved.status, TaskStatus::Pending);
        store.update(resolved).await.context("update resolved")?;

        // And the slot is *still* held — Pending blocks admission.
        let active = store
            .active_root_for_thread(&thread("t1"))
            .await
            .context("active final")?
            .context("still held")?;
        assert_eq!(active.status, TaskStatus::Pending);
        Ok(())
    }
}
