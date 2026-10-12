//! Execution order for the tool calls of one model response.
//!
//! A model that emits several tool calls in one response expects them to
//! take effect in the order it wrote them: `[edit, edit, bash "cargo
//! test"]` means "test the edited source". Running every call at once lets
//! the test race the edits. Only calls that cannot change anything can
//! safely overlap.
//!
//! # The rule
//!
//! Walk the batch in model order:
//!
//! - Consecutive [`Shared`](BatchSlot::Shared) calls form one wave and run
//!   concurrently.
//! - Each [`Exclusive`](BatchSlot::Exclusive) call is a wave of its own.
//! - A wave starts only after every call in every earlier wave has
//!   finished, so a call never starts before an earlier call it could
//!   affect or observe.
//! - [`Detached`](BatchSlot::Detached) calls take no part: they neither
//!   wait nor make anything wait.
//!
//! `[read, read, edit, read]` therefore runs as `{read, read}`, then
//! `{edit}`, then `{read}`.
//!
//! A call is `Shared` only when its tool declares
//! [`ToolConcurrency::ReadOnly`](crate::ToolConcurrency::ReadOnly).
//! Everything else, including shell commands, file writes, and unknown or
//! MCP tools, is `Exclusive`. Each runtime maps its calls onto slots:
//!
//! - The in-process agent loop resolves confirmation inline through
//!   hooks, so a call that needs confirmation is `Exclusive` there even
//!   when its tool is read-only.
//! - The durable worker marks subagent spawns `Detached`, so they fan out
//!   together as before, and marks calls whose result was settled before
//!   spawn `Detached`, because they never run.

/// How one call of a batch takes part in the batch's execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchSlot {
    /// May run alongside adjacent `Shared` calls.
    Shared,
    /// Runs alone, after every earlier call and before every later one.
    Exclusive,
    /// Neither waits for nor holds back any other call.
    Detached,
}

/// Partition a batch into waves of call indices, in execution order.
///
/// Every non-detached index appears in exactly one wave. Detached
/// indices appear in none.
#[must_use]
pub fn execution_waves(slots: &[BatchSlot]) -> Vec<Vec<usize>> {
    let mut waves: Vec<Vec<usize>> = Vec::new();
    let mut open_shared_wave = false;
    for (index, slot) in slots.iter().enumerate() {
        match slot {
            BatchSlot::Detached => {}
            BatchSlot::Shared => {
                match waves.last_mut() {
                    Some(wave) if open_shared_wave => wave.push(index),
                    _ => waves.push(vec![index]),
                }
                open_shared_wave = true;
            }
            BatchSlot::Exclusive => {
                waves.push(vec![index]);
                open_shared_wave = false;
            }
        }
    }
    waves
}

/// For every call, the earlier calls that must finish before it starts.
///
/// The result is indexed like `slots`. A call in the first wave, and
/// every detached call, waits for nothing.
#[must_use]
pub fn wait_lists(slots: &[BatchSlot]) -> Vec<Vec<usize>> {
    let mut after = vec![Vec::new(); slots.len()];
    let mut finished_before: Vec<usize> = Vec::new();
    for wave in execution_waves(slots) {
        for &index in &wave {
            after[index].clone_from(&finished_before);
        }
        finished_before.extend(wave);
    }
    after
}

#[cfg(test)]
mod tests {
    use super::BatchSlot::{Detached, Exclusive, Shared};
    use super::{execution_waves, wait_lists};

    #[test]
    fn mutations_and_commands_each_run_alone_in_model_order() {
        let slots = [Exclusive, Exclusive, Exclusive];
        assert_eq!(execution_waves(&slots), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(wait_lists(&slots), vec![vec![], vec![0], vec![0, 1]]);
    }

    #[test]
    fn consecutive_reads_share_one_wave() {
        let slots = [Shared, Shared, Shared];
        assert_eq!(execution_waves(&slots), vec![vec![0, 1, 2]]);
        assert_eq!(wait_lists(&slots), vec![Vec::<usize>::new(); 3]);
    }

    #[test]
    fn a_barrier_splits_reads_into_waves_around_it() {
        let slots = [Shared, Shared, Exclusive, Shared];
        assert_eq!(execution_waves(&slots), vec![vec![0, 1], vec![2], vec![3]]);
        assert_eq!(
            wait_lists(&slots),
            vec![vec![], vec![], vec![0, 1], vec![0, 1, 2]]
        );
    }

    #[test]
    fn detached_calls_neither_wait_nor_split_a_wave() {
        let slots = [Shared, Detached, Shared, Exclusive, Detached];
        assert_eq!(execution_waves(&slots), vec![vec![0, 2], vec![3]]);
        assert_eq!(
            wait_lists(&slots),
            vec![vec![], vec![], vec![], vec![0, 2], vec![]]
        );
    }
}
