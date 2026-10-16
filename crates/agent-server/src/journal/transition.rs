//! One state change and the events that report it, applied in one
//! transaction.
//!
//! A [`Transition`] is a value: guards that must hold, then writes. A
//! [`DurableStore`] applies it all or not at all. The guards are checked
//! before the first write, and the events are sequenced and given their
//! `thread_events_available` outbox row inside the same transaction, so a
//! crash can never leave a state change without the event that reports it
//! (design invariants I1–I4, bip `docs/durable-transitions-design-2026-10-12.md`).

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{
    ContinuationEnvelope, ListenExecutionContext, TerminalReason, ThreadId,
};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::execution_intent::ExecutionIntent;
use super::task::{AgentTask, AgentTaskId, LeaseId, SuspensionPayload, TaskStatus, WorkerId};

/// Guards and writes applied in one transaction.
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub guards: Vec<Guard>,
    pub writes: Vec<Write>,
}

/// A condition every guard checks before the first write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Guard {
    /// The task exists, is in `status`, and when `lease` is given is held
    /// by exactly that worker and lease.
    TaskIs {
        task: AgentTaskId,
        status: TaskStatus,
        lease: Option<(WorkerId, LeaseId)>,
    },
}

/// One write of a transition, applied in order.
#[derive(Clone, Debug)]
pub enum Write {
    /// Drive a running task to a terminal status. The backend also
    /// recomputes the parent and releases held siblings, in the same
    /// transaction. Needs a [`Guard::TaskIs`] with a lease on the same task.
    SettleTask { task: AgentTaskId, outcome: Settle },
    /// Fail a task parked on a confirmation (rejected or timed out) and
    /// recompute its parent. Needs a [`Guard::TaskIs`] for
    /// [`TaskStatus::AwaitingConfirmation`] on the same task.
    RejectConfirmation { task: AgentTaskId, error: String },
    /// Park a running task on a confirmation: it drops its lease and keeps
    /// the parent's continuation and the call's prepared operation. Needs a
    /// [`Guard::TaskIs`] with a lease on the same task.
    PauseOnConfirmation {
        task: AgentTaskId,
        continuation: Box<ContinuationEnvelope>,
        prepared_operation: Option<ListenExecutionContext>,
    },
    /// Re-park a steering exchange's parent on `reattach`, its still-live
    /// children, under `(worker, lease)`. Unlike the other task writes it
    /// takes no guard: the backend checks the parent's ownership itself,
    /// after locking the children, so the store's child-before-parent lock
    /// order holds.
    ReparkAfterSteering {
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        payload: Box<SuspensionPayload>,
        reattach: Vec<AgentTaskId>,
    },
    /// Store an execution intent's new status.
    RecordIntent(Box<ExecutionIntent>),
    /// Events on `thread`, without sequences: the backend allocates them
    /// and writes one coalesced outbox row per thread.
    AppendEvents {
        thread: ThreadId,
        events: Vec<AgentEvent>,
    },
}

/// How [`Write::SettleTask`] ends a task.
#[derive(Clone, Debug)]
pub enum Settle {
    Complete {
        result: serde_json::Value,
    },
    Fail {
        error: String,
    },
    FailWithReason {
        error: String,
        reason: TerminalReason,
    },
}

/// What a committed transition produced, in write order.
#[derive(Clone, Debug, Default)]
pub struct Committed {
    /// For each [`Write::SettleTask`] and [`Write::RejectConfirmation`]:
    /// the terminal task and its parent
    /// after the parent was recomputed (`None` for a root).
    pub settled: Vec<(AgentTask, Option<AgentTask>)>,
    /// For each [`Write::PauseOnConfirmation`]: the parked task.
    pub paused: Vec<AgentTask>,
    /// For each [`Write::ReparkAfterSteering`]: the re-parked parent.
    pub reparked: Vec<AgentTask>,
    /// Every event the transition appended, with its sequence.
    pub events: Vec<CommittedEvent>,
}

/// Why a transition wrote nothing.
#[derive(Debug, thiserror::Error)]
pub enum Rejected {
    /// The guard at `index` did not hold.
    #[error("transition rejected by guard {index}: {reason}")]
    Guard { index: usize, reason: String },
    /// The transition is malformed, or the store failed; nothing was
    /// written.
    #[error("transition failed: {0:#}")]
    Store(#[from] anyhow::Error),
}

/// A store that applies a [`Transition`] in one transaction.
#[async_trait]
pub trait DurableStore: Send + Sync {
    /// Check every guard, then apply every write, in one transaction.
    ///
    /// # Errors
    ///
    /// [`Rejected::Guard`] when a guard does not hold, and
    /// [`Rejected::Store`] when the transition is malformed or the store
    /// fails. Either way nothing was written.
    async fn commit(
        &self,
        transition: Transition,
        now: OffsetDateTime,
    ) -> Result<Committed, Rejected>;
}

impl Transition {
    /// Settle a running, leased task and append `events`, one transaction.
    #[must_use]
    pub fn settle(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        outcome: Settle,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![
                Write::SettleTask { task, outcome },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Settle a running, leased task with no events.
    #[must_use]
    pub fn settle_only(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        outcome: Settle,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![Write::SettleTask { task, outcome }],
        }
    }

    /// Fail a task parked on a confirmation and append `events`, one
    /// transaction.
    #[must_use]
    pub fn reject_confirmation(
        task: AgentTaskId,
        error: String,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::AwaitingConfirmation,
                lease: None,
            }],
            writes: vec![
                Write::RejectConfirmation { task, error },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Park a running, leased task on a confirmation and append `events`,
    /// one transaction.
    #[must_use]
    pub fn pause_on_confirmation(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        let mut transition =
            Self::pause_only(task, worker, lease, continuation, prepared_operation);
        transition
            .writes
            .push(Write::AppendEvents { thread, events });
        transition
    }

    /// Park a running, leased task on a confirmation with no events.
    #[must_use]
    pub fn pause_only(
        task: AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
    ) -> Self {
        Self {
            guards: vec![Guard::TaskIs {
                task: task.clone(),
                status: TaskStatus::Running,
                lease: Some((worker, lease)),
            }],
            writes: vec![Write::PauseOnConfirmation {
                task,
                continuation: Box::new(continuation),
                prepared_operation,
            }],
        }
    }

    /// Re-park a steering exchange's parent on its still-live children and
    /// append `events`, one transaction.
    #[must_use]
    pub fn repark_after_steering(
        task: AgentTaskId,
        (worker, lease): (WorkerId, LeaseId),
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        thread: ThreadId,
        events: Vec<AgentEvent>,
    ) -> Self {
        Self {
            guards: Vec::new(),
            writes: vec![
                Write::ReparkAfterSteering {
                    task,
                    worker,
                    lease,
                    payload: Box::new(payload),
                    reattach,
                },
                Write::AppendEvents { thread, events },
            ],
        }
    }

    /// Whether a guard holds `task` in `status`.
    #[must_use]
    pub fn guards_status(&self, task: &AgentTaskId, status: TaskStatus) -> bool {
        self.guards.iter().any(|guard| match guard {
            Guard::TaskIs {
                task: guarded,
                status: guarded_status,
                ..
            } => guarded == task && *guarded_status == status,
        })
    }

    /// The (worker, lease) a guard holds on `task`, which a
    /// [`Write::SettleTask`] on that task settles under.
    #[must_use]
    pub fn lease_for(&self, task: &AgentTaskId) -> Option<(&WorkerId, &LeaseId)> {
        self.guards.iter().find_map(|guard| match guard {
            Guard::TaskIs {
                task: guarded,
                lease: Some((worker, lease)),
                ..
            } if guarded == task => Some((worker, lease)),
            Guard::TaskIs { .. } => None,
        })
    }

    /// Whether any write appends at least one event.
    #[must_use]
    pub fn appends_events(&self) -> bool {
        self.writes.iter().any(|write| match write {
            Write::AppendEvents { events, .. } => !events.is_empty(),
            Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::RecordIntent(_) => false,
        })
    }

    /// Whether any write records an execution intent.
    #[must_use]
    pub fn records_intents(&self) -> bool {
        self.writes
            .iter()
            .any(|write| matches!(write, Write::RecordIntent(_)))
    }

    /// Check the shape every backend relies on before it opens a
    /// transaction: each settle has a leased guard on its task, each pause
    /// a leased guard that its task is running, and each confirmation
    /// reject a guard that it is awaiting confirmation.
    ///
    /// # Errors
    ///
    /// [`Rejected::Store`] naming the first malformed write.
    pub fn validate(&self) -> Result<(), Rejected> {
        for write in &self.writes {
            match write {
                Write::SettleTask { task, .. } => {
                    if self.lease_for(task).is_none() {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "settling task {task} needs a TaskIs guard with its lease"
                        )));
                    }
                }
                Write::PauseOnConfirmation { task, .. } => {
                    if self.lease_for(task).is_none()
                        || !self.guards_status(task, TaskStatus::Running)
                    {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "pausing task {task} needs a TaskIs guard that it is running \
                             under its lease"
                        )));
                    }
                }
                Write::RejectConfirmation { task, .. } => {
                    if !self.guards_status(task, TaskStatus::AwaitingConfirmation) {
                        return Err(Rejected::Store(anyhow::anyhow!(
                            "rejecting the confirmation of task {task} needs a TaskIs guard \
                             that it is awaiting confirmation"
                        )));
                    }
                }
                Write::ReparkAfterSteering { .. }
                | Write::RecordIntent(_)
                | Write::AppendEvents { .. } => {}
            }
        }
        Ok(())
    }
}

/// How a guard names the status it expects, in the words the store's own
/// rejections have always used.
const fn status_words(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Queued => "queued",
        TaskStatus::Pending => "pending",
        TaskStatus::Running => "running",
        TaskStatus::WaitingOnChildren => "waiting on children",
        TaskStatus::AwaitingConfirmation => "awaiting confirmation",
        TaskStatus::AwaitingQuestion => "awaiting a question",
        TaskStatus::Completed => "completed",
        TaskStatus::Failed => "failed",
        TaskStatus::Cancelled => "cancelled",
    }
}

/// The error a store method that wraps [`DurableStore::commit`] returns
/// for `rejected`: a guard rejection reads `{method} rejected: {reason}`,
/// the text that method returned before it became a wrapper.
#[must_use]
pub fn wrapper_error(method: &str, rejected: Rejected) -> anyhow::Error {
    match rejected {
        Rejected::Guard { reason, .. } => anyhow::anyhow!("{method} rejected: {reason}"),
        Rejected::Store(error) => error,
    }
}

impl Guard {
    /// Check this guard against the stored row of its task.
    ///
    /// # Errors
    ///
    /// [`Rejected::Guard`] with `index` when the guard does not hold.
    pub fn check(&self, index: usize, stored: Option<&AgentTask>) -> Result<(), Rejected> {
        match self {
            Self::TaskIs {
                task,
                status,
                lease,
            } => {
                let reject = |reason: String| Rejected::Guard { index, reason };
                let Some(stored) = stored else {
                    return Err(reject(format!("task {task} does not exist")));
                };
                if stored.status != *status {
                    return Err(reject(format!(
                        "task {task} is not {} (status {:?})",
                        status_words(*status),
                        stored.status
                    )));
                }
                if let Some((worker, lease)) = lease {
                    if stored.worker_id.as_ref() != Some(worker) {
                        return Err(reject(format!("worker mismatch on task {task}")));
                    }
                    if stored.lease_id.as_ref() != Some(lease) {
                        return Err(reject(format!("lease mismatch on task {task}")));
                    }
                }
                Ok(())
            }
        }
    }

    /// The task this guard reads.
    #[must_use]
    pub const fn task(&self) -> &AgentTaskId {
        match self {
            Self::TaskIs { task, .. } => task,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(worker: &str, lease: &str) -> AgentTask {
        let mut task =
            AgentTask::new_root_turn(ThreadId::from_string("t"), OffsetDateTime::UNIX_EPOCH, 3);
        task.status = TaskStatus::Running;
        task.worker_id = Some(WorkerId::from_string(worker));
        task.lease_id = Some(LeaseId::from_string(lease));
        task
    }

    fn guard_on(task: &AgentTask, worker: &str, lease: &str) -> Guard {
        Guard::TaskIs {
            task: task.id.clone(),
            status: TaskStatus::Running,
            lease: Some((WorkerId::from_string(worker), LeaseId::from_string(lease))),
        }
    }

    #[test]
    fn a_guard_holds_only_for_its_status_and_lease() {
        let task = running("w1", "l1");
        assert!(guard_on(&task, "w1", "l1").check(0, Some(&task)).is_ok());
        for guard in [guard_on(&task, "w2", "l1"), guard_on(&task, "w1", "l2")] {
            assert!(matches!(
                guard.check(3, Some(&task)),
                Err(Rejected::Guard { index: 3, .. })
            ));
        }
        let mut done = task.clone();
        done.status = TaskStatus::Completed;
        assert!(guard_on(&task, "w1", "l1").check(0, Some(&done)).is_err());
        assert!(guard_on(&task, "w1", "l1").check(0, None).is_err());
    }

    #[test]
    fn a_settle_needs_a_leased_guard_on_its_task() {
        let task = running("w1", "l1");
        let mut transition = Transition::settle(
            task.id.clone(),
            WorkerId::from_string("w1"),
            LeaseId::from_string("l1"),
            Settle::Fail {
                error: "boom".into(),
            },
            task.thread_id,
            Vec::new(),
        );
        assert!(transition.validate().is_ok());
        transition.guards.clear();
        assert!(matches!(transition.validate(), Err(Rejected::Store(_))));
    }
}
