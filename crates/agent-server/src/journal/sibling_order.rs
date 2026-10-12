//! Durable execution order for the tool children of one batch.
//!
//! The worker derives each tool child's wait list from
//! [`agent_sdk_foundation::batch_order`] and records it on
//! [`ChildSpawnSpec::after`](super::task::ChildSpawnSpec::after). Every
//! store backend spawns the whole batch at once, as before, and writes each
//! child's wait list to [`AgentTask::after_siblings`].
//!
//! Readiness is decided at acquisition, never by a write: a pending child is
//! acquirable when every sibling it waits for is terminal (or gone), and
//! acquisition clears the list as it leases the row. No terminal write takes
//! an extra lock or touches a sibling, a sibling finishing twice or out of
//! order changes nothing, and a row an older build requeued with its list
//! intact is acquired as soon as its siblings have finished.
//!
//! Every child row exists from the spawn on, so cancellation, early resume,
//! re-attachment, and fan-in see a waiting child exactly like a child that
//! has not been leased yet.
//!
//! A held row can stay held for as long as a sibling runs or awaits a
//! decision, so acquisition never walks held rows to find ready work. The
//! FIFO scan reads only rows that wait on nothing. A held row is reached in
//! one of two ways:
//!
//! - When a tool child settles, the host asks the store for the held
//!   siblings under the same parent that are now ready
//!   ([`AgentTaskStore::ready_held_children`](super::store::AgentTaskStore::ready_held_children))
//!   and wakes a worker for each, which leases it by id.
//! - As a backstop for a wake that never came (another host settled the
//!   sibling, or this one crashed first), [`HeldProbe`] examines a bounded
//!   window of held rows from a rotating cursor when the ready scan finds
//!   nothing, and every [`HELD_PROBE_FIRST_EVERY`] acquisitions before it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result, ensure};

use super::store::ToolChildSpawn;
use super::task::{AgentTask, AgentTaskId, ChildSpawnSpec, TaskKind, TaskStatus};

/// Hold each freshly built tool child behind the siblings its spec names.
///
/// `waits` pairs a child's `spawn_index` with the spawn indices of the
/// siblings it waits for. A named sibling is either a tool child of this
/// same spawn with a lower index, or one of `reattached`, the still-live
/// children of an earlier batch this spawn carries over; both come before
/// the child, so wait lists can never form a cycle. A re-attached subagent
/// invocation is skipped, because subagents never hold anything back.
/// Siblings that are already terminal (a result settled before spawn) are
/// dropped too, and a child left waiting for nothing is runnable at once.
///
/// # Errors
/// Returns an error when a wait list names a sibling outside this spawn,
/// a new sibling that does not come earlier, or when the hold transition
/// fails.
pub fn hold_tool_children_behind_siblings(
    children: &mut [AgentTask],
    reattached: &[AgentTask],
    waits: &[(u32, Vec<u32>)],
) -> Result<()> {
    let spawned: BTreeMap<u32, &AgentTask> = children
        .iter()
        .filter_map(|child| child.spawn_index.map(|slot| (slot, child)))
        .collect();
    let carried: BTreeMap<u32, &AgentTask> = reattached
        .iter()
        .filter_map(|child| child.spawn_index.map(|slot| (slot, child)))
        .collect();
    let mut holds = Vec::new();
    for (slot, after_slots) in waits {
        let mut after = Vec::with_capacity(after_slots.len());
        for earlier in after_slots {
            let sibling = if let Some(child) = spawned.get(earlier) {
                ensure!(
                    earlier < slot,
                    "spawn rejected: child {slot} may only wait for earlier siblings, not {earlier}"
                );
                *child
            } else {
                let child = carried.get(earlier).with_context(|| {
                    format!("spawn rejected: child {slot} waits for {earlier}, which is not a child of this spawn")
                })?;
                if child.kind != TaskKind::ToolRuntime {
                    continue;
                }
                *child
            };
            if !sibling.status.is_terminal() {
                after.push(sibling.id.clone());
            }
        }
        if !after.is_empty() {
            holds.push((*slot, after));
        }
    }
    for (slot, after) in holds {
        let child = children
            .iter_mut()
            .find(|child| child.spawn_index == Some(slot))
            .with_context(|| format!("spawn rejected: no tool child at spawn index {slot}"))?;
        *child = child
            .clone()
            .wait_on_siblings(after)
            .context("spawn rejected: hold child behind its siblings")?;
    }
    Ok(())
}

/// Wait lists for a pure tool spawn, whose spawn indices are spec positions.
///
/// # Errors
/// Returns an error when a batch is too large for a `u32` spawn index.
pub fn pure_spawn_waits(specs: &[ChildSpawnSpec]) -> Result<Vec<(u32, Vec<u32>)>> {
    specs
        .iter()
        .enumerate()
        .map(|(slot, spec)| {
            let slot =
                u32::try_from(slot).context("spawn rejected: batch index exceeds u32::MAX")?;
            Ok((slot, spec.after.clone()))
        })
        .collect()
}

/// Wait lists for the tool children of a mixed spawn.
#[must_use]
pub fn mixed_spawn_waits(tool_children: &[ToolChildSpawn]) -> Vec<(u32, Vec<u32>)> {
    tool_children
        .iter()
        .map(|tool| (tool.spawn_index, tool.spec.after.clone()))
        .collect()
}

/// The ids of `tasks` that are not yet terminal, the set
/// [`AgentTask::waits_on_any`] and [`AgentTask::admit_after_siblings`]
/// check a wait list against.
pub fn unfinished_ids<'a>(tasks: impl IntoIterator<Item = &'a AgentTask>) -> BTreeSet<AgentTaskId> {
    tasks
        .into_iter()
        .filter(|task| !task.status.is_terminal())
        .map(|task| task.id.clone())
        .collect()
}

/// The wait list a stored row carries, as this build reads it.
///
/// A wait list only means something on a pending row. An older build that
/// does not know the column can lease or finish a waiting row and leave the
/// list behind; on any other status it is ignored rather than failing the
/// row's decode.
#[must_use]
pub fn decoded_wait_list(status: TaskStatus, stored: Vec<AgentTaskId>) -> Vec<AgentTaskId> {
    match status {
        TaskStatus::Pending => stored,
        TaskStatus::Queued
        | TaskStatus::Running
        | TaskStatus::WaitingOnChildren
        | TaskStatus::AwaitingConfirmation
        | TaskStatus::AwaitingQuestion
        | TaskStatus::Completed
        | TaskStatus::Failed
        | TaskStatus::Cancelled => Vec::new(),
    }
}

/// How many held rows one backstop probe examines.
pub const HELD_PROBE_WINDOW: usize = 32;

/// Every this many acquisitions, the backstop probe runs before the ready
/// scan, so a held row whose wake was missed is reached even while ready
/// work never runs out.
pub const HELD_PROBE_FIRST_EVERY: u64 = 16;

/// One held row a backstop probe examined. `key` is the row's position in
/// the `(created_at, id)` order the probe walks, as the backend compares it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldRow<K> {
    pub key: K,
    pub id: AgentTaskId,
    /// Every sibling the row waits for is terminal or gone.
    pub ready: bool,
}

/// Where a store's backstop probe resumes, shared by every clone of the
/// store.
#[derive(Debug)]
pub struct HeldProbe<K> {
    state: Arc<HeldProbeState<K>>,
}

#[derive(Debug)]
struct HeldProbeState<K> {
    cursor: Mutex<Option<K>>,
    acquisitions: AtomicU64,
}

impl<K> Clone for HeldProbe<K> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<K> Default for HeldProbe<K> {
    fn default() -> Self {
        Self {
            state: Arc::new(HeldProbeState {
                cursor: Mutex::new(None),
                acquisitions: AtomicU64::new(0),
            }),
        }
    }
}

impl<K: Clone> HeldProbe<K> {
    /// Count one acquisition, and say whether it probes held rows before
    /// the ready scan.
    #[must_use]
    pub fn probes_first(&self) -> bool {
        let count = self.state.acquisitions.fetch_add(1, Ordering::Relaxed);
        count % HELD_PROBE_FIRST_EVERY == HELD_PROBE_FIRST_EVERY - 1
    }

    /// The row after which the next probe starts, `None` for the start.
    #[must_use]
    pub fn cursor(&self) -> Option<K> {
        self.state
            .cursor
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record one probe's reads (see [`plan_held_window`]) and return the
    /// ready rows it found, in the order it examined them.
    pub fn advance(
        &self,
        had_cursor: bool,
        after_cursor: Vec<HeldRow<K>>,
        from_start: Vec<HeldRow<K>>,
    ) -> Vec<AgentTaskId> {
        let (ready, next) =
            plan_held_window(had_cursor, after_cursor, from_start, HELD_PROBE_WINDOW);
        *self
            .state
            .cursor
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = next;
        ready
    }
}

/// Combine one backstop probe's reads.
///
/// `after_cursor` holds up to `window` held rows after the cursor (from
/// the start when there is none). When that read ran short after a cursor,
/// the probe wraps: `from_start` holds up to the rest of the window, read
/// again from the start. Returns the ready rows in the order examined, and
/// where the next probe resumes: after the last row examined, or `None`
/// once every held row has been examined.
#[must_use]
pub fn plan_held_window<K: Clone>(
    had_cursor: bool,
    after_cursor: Vec<HeldRow<K>>,
    from_start: Vec<HeldRow<K>>,
    window: usize,
) -> (Vec<AgentTaskId>, Option<K>) {
    let rest = window.saturating_sub(after_cursor.len());
    let next = if rest == 0 {
        after_cursor.last().map(|row| row.key.clone())
    } else if had_cursor && from_start.len() >= rest {
        from_start.last().map(|row| row.key.clone())
    } else {
        None
    };
    let wrapped = if had_cursor && rest > 0 {
        from_start
    } else {
        Vec::new()
    };
    let mut seen = BTreeSet::new();
    let ready = after_cursor
        .into_iter()
        .chain(wrapped)
        .filter(|row| seen.insert(row.id.clone()))
        .filter(|row| row.ready)
        .map(|row| row.id)
        .collect();
    (ready, next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::{LeaseId, TaskSchemaError, WorkerId};
    use agent_sdk_foundation::ThreadId;
    use time::OffsetDateTime;

    fn now() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + time::Duration::hours(1)
    }

    fn batch(size: u32) -> Result<Vec<AgentTask>> {
        let root = AgentTask::new_root_turn(ThreadId::from_string("t-order"), now(), 3)
            .mark_running(
                WorkerId::from_string("w"),
                LeaseId::from_string("l"),
                now() + time::Duration::minutes(1),
                now(),
            )?;
        (0..size)
            .map(|slot| {
                let mut child = AgentTask::new_child(&root, TaskKind::ToolRuntime, now(), 1)?;
                child.spawn_index = Some(slot);
                Ok(child)
            })
            .collect()
    }

    fn lease(child: &AgentTask) -> Result<AgentTask, TaskSchemaError> {
        child.clone().mark_running(
            WorkerId::from_string("w"),
            LeaseId::from_string("l"),
            now() + time::Duration::minutes(1),
            now(),
        )
    }

    fn finish(child: &AgentTask) -> Result<AgentTask> {
        Ok(lease(child)?.complete(now())?)
    }

    #[test]
    fn a_held_child_runs_only_after_every_sibling_it_waits_for() -> Result<()> {
        let mut children = batch(3)?;
        hold_tool_children_behind_siblings(&mut children, &[], &[(1, vec![0]), (2, vec![0, 1])])?;
        let unfinished = unfinished_ids(&children);
        assert!(!children[0].waits_on_any(&unfinished));
        assert!(children[1].waits_on_any(&unfinished));
        assert!(children[2].waits_on_any(&unfinished));
        assert_eq!(
            children[1].clone().admit_after_siblings(&unfinished).err(),
            Some(TaskSchemaError::HeldBySiblings)
        );

        children[0] = finish(&children[0])?;
        let unfinished = unfinished_ids(&children);
        assert!(!children[1].waits_on_any(&unfinished));
        assert!(children[2].waits_on_any(&unfinished));
        let admitted = children[1].clone().admit_after_siblings(&unfinished)?;
        assert_eq!(admitted.after_siblings, Vec::<AgentTaskId>::new());
        assert!(lease(&admitted).is_ok());
        Ok(())
    }

    #[test]
    fn a_child_waiting_only_on_settled_siblings_is_not_held() -> Result<()> {
        let mut children = batch(2)?;
        children[0] = finish(&children[0])?;
        hold_tool_children_behind_siblings(&mut children, &[], &[(1, vec![0])])?;
        assert_eq!(children[1].after_siblings, Vec::<AgentTaskId>::new());
        Ok(())
    }

    #[test]
    fn wait_lists_must_name_earlier_siblings_of_the_same_spawn() -> Result<()> {
        let mut children = batch(2)?;
        assert!(hold_tool_children_behind_siblings(&mut children, &[], &[(0, vec![1])]).is_err());
        assert!(hold_tool_children_behind_siblings(&mut children, &[], &[(1, vec![7])]).is_err());
        Ok(())
    }

    #[test]
    fn a_new_child_may_wait_for_a_reattached_survivor() -> Result<()> {
        let mut survivors = batch(1)?;
        survivors[0].spawn_index = Some(5);
        let mut children = batch(1)?;
        hold_tool_children_behind_siblings(&mut children, &survivors, &[(0, vec![5])])?;
        assert_eq!(children[0].after_siblings, [survivors[0].id.clone()]);
        Ok(())
    }

    #[test]
    fn a_held_child_cannot_be_leased_before_admission() -> Result<()> {
        let mut children = batch(2)?;
        hold_tool_children_behind_siblings(&mut children, &[], &[(1, vec![0])])?;
        assert_eq!(
            lease(&children[1]).err(),
            Some(TaskSchemaError::HeldBySiblings)
        );
        Ok(())
    }

    fn held(key: u32, ready: bool) -> HeldRow<u32> {
        HeldRow {
            key,
            id: AgentTaskId::from_string(format!("task_{key}")),
            ready,
        }
    }

    fn ids(keys: &[u32]) -> Vec<AgentTaskId> {
        keys.iter()
            .map(|key| AgentTaskId::from_string(format!("task_{key}")))
            .collect()
    }

    #[test]
    fn a_full_window_resumes_after_its_last_row() {
        let rows = vec![held(1, false), held(2, true), held(3, false)];
        assert_eq!(
            plan_held_window(false, rows, Vec::new(), 3),
            (ids(&[2]), Some(3))
        );
    }

    #[test]
    fn a_short_window_without_a_cursor_covered_every_held_row() {
        let rows = vec![held(1, true), held(2, false)];
        assert_eq!(
            plan_held_window(false, rows, Vec::new(), 3),
            (ids(&[1]), None)
        );
    }

    #[test]
    fn a_short_window_after_a_cursor_wraps_to_the_start() {
        let after = vec![held(8, false), held(9, true)];
        let from_start = vec![held(1, true), held(2, false)];
        assert_eq!(
            plan_held_window(true, after, from_start, 4),
            (ids(&[9, 1]), Some(2)),
            "the wrapped rows fill the window, and the next probe resumes after them"
        );
    }

    #[test]
    fn a_wrap_that_runs_out_examined_every_held_row_once() {
        let after = vec![held(2, true)];
        let from_start = vec![held(1, true), held(2, true)];
        assert_eq!(
            plan_held_window(true, after, from_start, 4),
            (ids(&[2, 1]), None)
        );
    }

    #[test]
    fn no_held_rows_restart_from_the_start() {
        assert_eq!(
            plan_held_window::<u32>(true, Vec::new(), Vec::new(), 4),
            (Vec::new(), None)
        );
    }

    #[test]
    fn a_probe_cursor_is_shared_by_clones() {
        let probe = HeldProbe::<u32>::default();
        let clone = probe.clone();
        let full: Vec<_> = (1..=u32::try_from(HELD_PROBE_WINDOW).unwrap_or(u32::MAX))
            .map(|key| held(key, false))
            .collect();
        let last = full.last().map(|row| row.key);
        assert!(clone.advance(false, full, Vec::new()).is_empty());
        assert_eq!(probe.cursor(), last);
    }

    #[test]
    fn every_nth_acquisition_probes_held_rows_first() {
        let probe = HeldProbe::<u32>::default();
        let first: Vec<bool> = (0..HELD_PROBE_FIRST_EVERY * 2)
            .map(|_| probe.probes_first())
            .collect();
        assert_eq!(first.iter().filter(|probes| **probes).count(), 2);
        assert!(first[usize::try_from(HELD_PROBE_FIRST_EVERY - 1).unwrap_or(0)]);
    }

    #[test]
    fn a_stale_wait_list_is_ignored_off_a_pending_row() {
        let stale = vec![AgentTaskId::from_string("task_sibling")];
        assert_eq!(decoded_wait_list(TaskStatus::Pending, stale.clone()), stale);
        assert_eq!(
            decoded_wait_list(TaskStatus::Running, stale.clone()),
            Vec::<AgentTaskId>::new()
        );
        assert_eq!(
            decoded_wait_list(TaskStatus::Completed, stale),
            Vec::<AgentTaskId>::new()
        );
    }
}
