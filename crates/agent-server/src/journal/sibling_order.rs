//! Durable execution order for the tool children of one batch.
//!
//! The worker derives each tool child's wait list from
//! [`agent_sdk_foundation::batch_order`] and records it on
//! [`ChildSpawnSpec::after`](super::task::ChildSpawnSpec::after). Every
//! store backend spawns the whole batch at once, as before, but a child
//! with a wait list is written `Pending` with
//! [`AgentTask::after_siblings`] set, which acquisition skips. When a child
//! reaches a terminal state, the store releases, in the same write, every
//! held sibling whose wait list is now fully terminal.
//!
//! Every child row exists from the spawn on, so cancellation, early resume,
//! re-attachment, and fan-in see held children exactly like children that
//! have not been leased yet. Only the spawn and the release live here,
//! shared by all backends, so they cannot drift apart.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use time::OffsetDateTime;

use super::store::ToolChildSpawn;
use super::task::{AgentTask, AgentTaskId, ChildSpawnSpec, TaskStatus};

/// Hold each freshly built tool child behind the earlier siblings its spec
/// names.
///
/// `waits` pairs a child's `spawn_index` with the spawn indices of the
/// siblings it waits for. Every named sibling must be a tool child of this
/// same spawn with a lower index, so wait lists can never form a cycle.
/// Siblings that are already terminal (a result settled before spawn) are
/// dropped, and a child left waiting for nothing stays runnable.
///
/// # Errors
/// Returns an error when a wait list names a sibling outside this spawn,
/// one that does not come earlier, or when the hold transition fails.
pub fn hold_tool_children_behind_siblings(
    children: &mut [AgentTask],
    waits: &[(u32, Vec<u32>)],
) -> Result<()> {
    let slots: BTreeMap<u32, (AgentTaskId, bool)> = children
        .iter()
        .filter_map(|child| {
            child
                .spawn_index
                .map(|slot| (slot, (child.id.clone(), child.status.is_terminal())))
        })
        .collect();
    for (slot, after_slots) in waits {
        let mut after = Vec::with_capacity(after_slots.len());
        for earlier in after_slots {
            ensure!(
                earlier < slot,
                "spawn rejected: child {slot} may only wait for earlier siblings, not {earlier}"
            );
            let (id, terminal) = slots.get(earlier).with_context(|| {
                format!("spawn rejected: child {slot} waits for {earlier}, which is not a tool child of this spawn")
            })?;
            if !terminal {
                after.push(id.clone());
            }
        }
        if after.is_empty() {
            continue;
        }
        let child = children
            .iter_mut()
            .find(|child| child.spawn_index == Some(*slot))
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

/// The held children of one parent that can run now.
///
/// `held` are the parent's children waiting on siblings, and `unfinished`
/// is every child of the parent that is not yet terminal, as both stand
/// after a terminal transition. A held child whose wait list names no
/// unfinished sibling is returned released, for the caller to write in the
/// same transaction. A sibling that no longer exists counts as finished,
/// so a row never waits on something that cannot finish.
///
/// # Errors
/// Returns an error when a release transition fails.
pub fn release_ready_siblings(
    held: &[AgentTask],
    unfinished: &BTreeSet<AgentTaskId>,
    now: OffsetDateTime,
) -> Result<Vec<AgentTask>> {
    let mut released = Vec::new();
    for child in held {
        if child.after_siblings.is_empty()
            || child.status != TaskStatus::Pending
            || child
                .after_siblings
                .iter()
                .any(|id| unfinished.contains(id))
        {
            continue;
        }
        released.push(
            child
                .clone()
                .release_from_siblings(now)
                .with_context(|| format!("release child {} from its siblings", child.id))?,
        );
    }
    Ok(released)
}

/// Split one parent's children into the held ones and the ids of those not
/// yet terminal, the two inputs of [`release_ready_siblings`].
#[must_use]
pub fn held_and_unfinished(children: &[AgentTask]) -> (Vec<AgentTask>, BTreeSet<AgentTaskId>) {
    let held = children
        .iter()
        .filter(|child| !child.after_siblings.is_empty())
        .cloned()
        .collect();
    let unfinished = children
        .iter()
        .filter(|child| !child.status.is_terminal())
        .map(|child| child.id.clone())
        .collect();
    (held, unfinished)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::{LeaseId, TaskKind, WorkerId};
    use agent_sdk_foundation::ThreadId;
    use anyhow::Context;

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

    fn finish(child: &AgentTask) -> Result<AgentTask> {
        Ok(child
            .clone()
            .mark_running(
                WorkerId::from_string("w"),
                LeaseId::from_string("l"),
                now() + time::Duration::minutes(1),
                now(),
            )?
            .complete(now())?)
    }

    fn ready(children: &[AgentTask]) -> Result<Vec<AgentTask>> {
        let (held, unfinished) = held_and_unfinished(children);
        release_ready_siblings(&held, &unfinished, now())
    }

    #[test]
    fn a_held_child_runs_only_after_every_sibling_it_waits_for() -> Result<()> {
        let mut children = batch(3)?;
        hold_tool_children_behind_siblings(&mut children, &[(1, vec![0]), (2, vec![0, 1])])?;
        assert!(children[0].is_leasable());
        assert!(!children[1].is_leasable());
        assert!(!children[2].is_leasable());
        assert!(ready(&children)?.is_empty());

        children[0] = finish(&children[0])?;
        let released = ready(&children)?;
        assert_eq!(released.len(), 1);
        let edit2 = released.first().context("second edit released")?;
        assert_eq!(edit2.id, children[1].id);
        assert!(edit2.is_leasable());
        children[1] = edit2.clone();

        assert!(ready(&children)?.is_empty());
        children[1] = finish(&children[1])?;
        let released = ready(&children)?;
        assert_eq!(
            released.iter().map(|row| &row.id).collect::<Vec<_>>(),
            [&children[2].id]
        );
        Ok(())
    }

    #[test]
    fn a_child_waiting_only_on_settled_siblings_is_not_held() -> Result<()> {
        let mut children = batch(2)?;
        children[0] = finish(&children[0])?;
        hold_tool_children_behind_siblings(&mut children, &[(1, vec![0])])?;
        assert!(children[1].is_leasable());
        Ok(())
    }

    #[test]
    fn wait_lists_must_name_earlier_siblings_of_the_same_spawn() -> Result<()> {
        let mut children = batch(2)?;
        assert!(hold_tool_children_behind_siblings(&mut children, &[(0, vec![1])]).is_err());
        assert!(hold_tool_children_behind_siblings(&mut children, &[(1, vec![7])]).is_err());
        Ok(())
    }

    #[test]
    fn a_held_child_cannot_be_leased() -> Result<()> {
        let mut children = batch(2)?;
        hold_tool_children_behind_siblings(&mut children, &[(1, vec![0])])?;
        let held = children[1].clone();
        assert!(
            held.mark_running(
                WorkerId::from_string("w"),
                LeaseId::from_string("l"),
                now() + time::Duration::minutes(1),
                now(),
            )
            .is_err()
        );
        Ok(())
    }
}
