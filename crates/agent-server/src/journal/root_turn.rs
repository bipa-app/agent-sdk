//! The terminal measurement of a root turn: `agent_server.root_turn.duration`
//! and `agent_server.root_turn.outcome`.
//!
//! Every store records a root turn once, after the transaction that made it
//! terminal commits: a settle (through
//! [`Committed::settled`](super::transition::Committed::settled) or a
//! completed turn's [`CommitOutcome::settled`](super::commit::CommitOutcome)),
//! `complete_task`, a cancel, and a fail-closed lease sweep, epoch reclaim or
//! acquisition. `reject_confirmation` ends only tool children, so it records
//! none.
//! A root turn is any [`TaskKind::RootTurn`] row, including the root of a
//! subagent's child thread; tool children, invocations and input injections
//! record nothing.

use super::task::{AgentTask, TaskKind, TaskStatus};

/// How a root turn ended: the `outcome` label of `agent_server.root_turn.*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootTurnOutcome {
    /// [`TaskStatus::Completed`].
    Success,
    /// [`TaskStatus::Failed`].
    Error,
    /// [`TaskStatus::Cancelled`].
    Cancelled,
}

impl RootTurnOutcome {
    /// The metric label: `success`, `error` or `cancelled`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// One root turn that reached a terminal status, measured from its row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalRootTurn {
    /// Admission (`created_at`) to terminal (`completed_at`), in seconds.
    pub duration_secs: f64,
    /// How it ended.
    pub outcome: RootTurnOutcome,
}

impl TerminalRootTurn {
    /// The measurement of `task` as committed, or `None` when it is not a
    /// terminal root turn. A terminal row with no `completed_at`, or one
    /// before `created_at`, is skipped with a warning.
    #[must_use]
    pub fn of(task: &AgentTask) -> Option<Self> {
        if task.kind != TaskKind::RootTurn {
            return None;
        }
        let outcome = match task.status {
            TaskStatus::Completed => RootTurnOutcome::Success,
            TaskStatus::Failed => RootTurnOutcome::Error,
            TaskStatus::Cancelled => RootTurnOutcome::Cancelled,
            TaskStatus::Queued
            | TaskStatus::Pending
            | TaskStatus::Running
            | TaskStatus::WaitingOnChildren
            | TaskStatus::AwaitingConfirmation
            | TaskStatus::AwaitingQuestion => return None,
        };
        let Some(terminal_at) = task.completed_at else {
            log::warn!(
                "terminal root turn {} has no completed_at; skipping agent_server.root_turn",
                task.id
            );
            return None;
        };
        if terminal_at < task.created_at {
            log::warn!(
                "root turn {} ended before it was admitted; skipping agent_server.root_turn",
                task.id
            );
            return None;
        }
        Some(Self {
            duration_secs: (terminal_at - task.created_at).as_seconds_f64(),
            outcome,
        })
    }

    /// Record this turn on `agent_server.root_turn.*`. A no-op without the
    /// `otel` feature.
    pub fn record(self) {
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_root_turn(self);
    }

    /// Record each of `turns`.
    pub fn record_each(turns: impl IntoIterator<Item = Self>) {
        for turn in turns {
            turn.record();
        }
    }

    /// Record every terminal root turn among `tasks` (see [`Self::of`]).
    pub fn record_tasks<'a>(tasks: impl IntoIterator<Item = &'a AgentTask>) {
        for turn in tasks.into_iter().filter_map(Self::of) {
            turn.record();
        }
    }

    /// Record the terminal root turns among a commit's settled tasks (the
    /// first of each pair; the second is the recomputed parent).
    pub fn record_settled(settled: &[(AgentTask, Option<AgentTask>)]) {
        Self::record_tasks(settled.iter().map(|(task, _)| task));
    }
}

#[cfg(test)]
mod tests {
    use agent_sdk_foundation::ThreadId;
    use time::{Duration, OffsetDateTime};

    use super::{RootTurnOutcome, TerminalRootTurn};
    use crate::journal::task::{AgentTask, TaskKind, TaskStatus};

    fn root(status: TaskStatus, elapsed: Option<Duration>) -> AgentTask {
        let created = OffsetDateTime::UNIX_EPOCH + Duration::hours(1);
        let mut task = AgentTask::new_root_turn(ThreadId::from_string("t"), created, 3);
        task.status = status;
        task.completed_at = elapsed.map(|elapsed| created + elapsed);
        task
    }

    #[test]
    fn a_terminal_root_turn_is_measured_from_admission() {
        for (status, outcome) in [
            (TaskStatus::Completed, RootTurnOutcome::Success),
            (TaskStatus::Failed, RootTurnOutcome::Error),
            (TaskStatus::Cancelled, RootTurnOutcome::Cancelled),
        ] {
            assert_eq!(
                TerminalRootTurn::of(&root(status, Some(Duration::milliseconds(1500)))),
                Some(TerminalRootTurn {
                    duration_secs: 1.5,
                    outcome
                })
            );
        }
    }

    #[test]
    fn a_live_root_another_kind_or_a_bad_clock_is_not_measured() {
        assert_eq!(TerminalRootTurn::of(&root(TaskStatus::Running, None)), None);
        assert_eq!(
            TerminalRootTurn::of(&root(TaskStatus::Completed, None)),
            None
        );
        assert_eq!(
            TerminalRootTurn::of(&root(TaskStatus::Completed, Some(Duration::seconds(-1)))),
            None
        );
        let mut tool = root(TaskStatus::Completed, Some(Duration::seconds(1)));
        tool.kind = TaskKind::ToolRuntime;
        assert_eq!(TerminalRootTurn::of(&tool), None);
    }
}
