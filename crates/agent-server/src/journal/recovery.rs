//! Phase 2.5 retry budget, failure handling, and stale-task recovery matrix.
//!
//! Phase 2.3 (ENG-7917) landed the acquisition / lease / heartbeat /
//! expiry-sweep primitives and deliberately deferred retry-budget
//! enforcement to Phase 2.5. Phase 2.4 (ENG-7918) layered typed
//! durable pause state on top, which introduced the one variant
//! ([`TaskState::AwaitingConfirmation`] with a
//! [`ListenExecutionContext`] prepared operation) that can carry a
//! staged external side-effect across a crash. Phase 2.5 closes
//! both gaps by making recovery deterministic for every row the
//! journal can see:
//!
//! 1. **Retry exhaustion is terminal.** Before 2.5 an exhausted row
//!    would either poison an `acquire_next_runnable` scan or bounce
//!    back and forth through `release_expired_leases` / Pending
//!    forever. Phase 2.5 classifies exhausted rows as
//!    [`RecoveryAction::FailClosed`] so the sweep and the acquire
//!    path both write them to [`TaskStatus::Failed`] instead of
//!    requeuing blindly.
//! 2. **Unsafe prepared operations fail closed.** A tool-runtime
//!    task that persisted a [`TaskState::AwaitingConfirmation`]
//!    with a `Some(prepared_operation)` has staged an external
//!    side-effect that the Phase 2.5 rewrite has no safe resume
//!    contract for — a blind requeue would let a fresh worker
//!    double-execute the staged operation. The recovery matrix
//!    classifies that exact row as
//!    [`FailureReason::UnsafePreparedOperationRecovery`] so the
//!    journal fails closed instead of re-running the tool.
//! 3. **Duplicate ownership is impossible across requeue + retry.**
//!    The `(worker_id, lease_id)` CAS from Phase 2.3 still guards
//!    every heartbeat and every targeted acquisition; Phase 2.5
//!    adds explicit regression coverage in the store tests so an
//!    old worker can never heartbeat a row that has been requeued
//!    or failed closed.
//!
//! # Matrix
//!
//! [`classify_recovery`] is the single pure entry point every
//! Phase 2.5 call site uses. Its decision depends on the row's
//! [`TaskKind`] × [`TaskStatus`] × retry-budget × prepared-operation
//! bit, plus a [`RecoveryContext`] flag that distinguishes
//! acquisition-time decisions from expiry-sweep decisions:
//!
//! | Context          | Status / kind                                     | Budget    | Prepared op | Action                                                   |
//! |------------------|---------------------------------------------------|-----------|-------------|----------------------------------------------------------|
//! | Acquisition      | `Pending`, budget ok                              | ok        | n/a         | [`RecoveryAction::NoAction`]                             |
//! | Acquisition      | `Pending`, budget exhausted                       | exhausted | n/a         | [`RecoveryAction::FailClosed`] `RetryBudgetExhausted`    |
//! | `ExpiredLease`   | `Running`, budget ok                              | ok        | n/a         | [`RecoveryAction::Requeue`]                              |
//! | `ExpiredLease`   | `Running`, budget exhausted                       | exhausted | n/a         | [`RecoveryAction::FailClosed`] `LeaseExpiredBudgetExhausted` |
//! | Any              | `tool_runtime` `AwaitingConfirmation` w/ prep op  | any       | Some        | [`RecoveryAction::FailClosed`] `UnsafePreparedOperationRecovery` |
//! | Any              | `root_turn` `AwaitingConfirmation`                | any       | any         | [`RecoveryAction::NoAction`]                             |
//! | Any              | `WaitingOnChildren`                               | any       | n/a         | [`RecoveryAction::NoAction`]                             |
//! | Any              | `Queued`                                          | any       | n/a         | [`RecoveryAction::NoAction`]                             |
//! | Any              | `Completed` / `Failed` / `Cancelled`              | any       | n/a         | [`RecoveryAction::NoAction`]                             |
//!
//! The table is locked by the
//! `classify_recovery_matrix_is_exhaustive` test in this module so
//! any future status / kind addition forces an explicit
//! classification rather than silently inheriting `NoAction`.
//!
//! # Integration with the store
//!
//! The matrix is consumed by three call sites in
//! [`super::store::InMemoryAgentTaskStore`]:
//!
//! - [`super::AgentTaskStore::try_acquire_task`] — classifies the
//!   row before transitioning to `Running`. Exhausted rows are
//!   failed-closed and the call returns `Ok(None)` instead of
//!   bubbling up an `AttemptExceedsMax` error.
//! - [`super::AgentTaskStore::acquire_next_runnable`] — classifies
//!   the scan head and, on `FailClosed`, fails the row and keeps
//!   scanning so a single exhausted row can never poison the
//!   whole worker pool.
//! - [`super::AgentTaskStore::release_expired_leases`] — runs the
//!   classification on every row coming off the expiry index and
//!   writes either a `Requeue` (the Phase 2.3 behavior) or a
//!   `FailClosed`, returning a [`RecoveryRecord`] per sweep row so
//!   callers can log and react.
//!
//! The call sites keep the write-lock discipline from Phase 2.3 —
//! every classification and its matching transition run under the
//! same store write lock so a row can never be observed in an
//! intermediate state.
//!
//! [`TaskState::AwaitingConfirmation`]: super::TaskState::AwaitingConfirmation
//! [`ListenExecutionContext`]: agent_sdk_core::ListenExecutionContext
//! [`TaskKind`]: super::TaskKind
//! [`TaskStatus`]: super::TaskStatus

use serde::{Deserialize, Serialize};

use super::task::{AgentTask, AgentTaskId, TaskKind, TaskStatus};

/// Canonical fail-closed reasons written on Phase 2.5 recovery
/// transitions.
///
/// The variants are stable `snake_case` identifiers on the wire so
/// downstream systems (dashboards, alerting, replay tooling) can
/// pattern-match on the `last_error` prefix without reparsing free
/// text. [`FailureReason::error_prefix`] returns the exact string
/// used at the head of [`AgentTask::last_error`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureReason {
    /// A row in [`TaskStatus::Pending`] whose `attempt == max_attempts`
    /// was seen by an acquisition path. The row is failed closed
    /// instead of being leased for a doomed attempt that would
    /// immediately overflow the budget.
    RetryBudgetExhausted,
    /// A row in [`TaskStatus::Running`] whose lease expired and whose
    /// `attempt == max_attempts`. Phase 2.3 would blindly requeue this
    /// row; Phase 2.5 fails it closed so the journal never enters an
    /// endless requeue loop.
    LeaseExpiredBudgetExhausted,
    /// A [`TaskKind::ToolRuntime`] row in [`TaskStatus::AwaitingConfirmation`]
    /// with a `Some(prepared_operation)`. Tool-runtime tasks have no
    /// safe resume contract for staged listen/execute operations in
    /// Phase 2.5, so blind recovery would risk double-executing the
    /// external operation. The journal fails the row closed and
    /// leaves the prepared operation for an operator to reconcile
    /// out-of-band.
    UnsafePreparedOperationRecovery,
}

impl FailureReason {
    /// The stable `snake_case` prefix written at the head of
    /// [`AgentTask::last_error`] whenever Phase 2.5 fails a row
    /// closed.
    ///
    /// Wire-format locked by the `failure_reason_error_prefix_is_stable`
    /// test in this module.
    #[must_use]
    pub const fn error_prefix(self) -> &'static str {
        match self {
            Self::RetryBudgetExhausted => "retry_budget_exhausted",
            Self::LeaseExpiredBudgetExhausted => "lease_expired_budget_exhausted",
            Self::UnsafePreparedOperationRecovery => "unsafe_prepared_operation_recovery",
        }
    }

    /// A canonical `last_error` message for this reason, including
    /// the row's id, attempt counter, and max-attempt budget.
    ///
    /// Callers should prefer this helper over hand-rolling an error
    /// string so the wire format stays stable across call sites.
    #[must_use]
    pub fn error_message(self, task: &AgentTask) -> String {
        let prefix = self.error_prefix();
        let id = &task.id;
        let attempt = task.attempt;
        let max = task.max_attempts;
        let kind = task.kind;
        let status = task.status;
        format!(
            "{prefix}: task {id} ({kind} {status:?}) exhausted recovery budget (attempt={attempt}, max={max})"
        )
    }
}

/// Where in the journal lifecycle a recovery classification is
/// being made.
///
/// The matrix only needs to distinguish acquisition-time decisions
/// (is this row safe to lease right now?) from expiry-sweep decisions
/// (the row's worker just disappeared — do we requeue or fail?), so
/// two variants are enough. A future phase may add more contexts (e.g.
/// a subagent-resume path) without disrupting the acquisition /
/// sweep wiring that already exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryContext {
    /// A worker is trying to lease the row right now via
    /// [`super::AgentTaskStore::try_acquire_task`] or
    /// [`super::AgentTaskStore::acquire_next_runnable`].
    AcquisitionAttempt,
    /// The expiry-sweep in
    /// [`super::AgentTaskStore::release_expired_leases`] is about
    /// to release the row's lease and has to decide whether to
    /// requeue it or fail it closed.
    ExpiredLease,
}

/// What Phase 2.5 has decided to do with a row after consulting
/// the recovery matrix.
///
/// Every variant is exhaustive and deterministic — two different
/// callers running [`classify_recovery`] on the same row in the
/// same context always agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// The row is already in a safe state for the caller's context.
    /// The caller should proceed with its normal path (acquire the
    /// row, leave the row alone, etc).
    NoAction,
    /// Return the row to [`TaskStatus::Pending`]. Used only by the
    /// expiry sweep — the acquisition paths never produce this
    /// action because they only ever see already-`Pending` rows.
    Requeue,
    /// Transition the row to [`TaskStatus::Failed`] with the given
    /// reason. Acquisition paths skip the row and return `None`;
    /// the sweep records the failure in the [`RecoveryRecord`] it
    /// returns to its caller.
    FailClosed(FailureReason),
}

impl RecoveryAction {
    /// `true` if this action terminates the row's lifecycle.
    #[must_use]
    pub const fn is_fail_closed(self) -> bool {
        matches!(self, Self::FailClosed(_))
    }

    /// `true` if this action requeues the row to [`TaskStatus::Pending`].
    #[must_use]
    pub const fn is_requeue(self) -> bool {
        matches!(self, Self::Requeue)
    }
}

/// One row's Phase 2.5 recovery outcome, returned from
/// [`super::AgentTaskStore::release_expired_leases`].
///
/// Phase 2.3's sweep returned a flat `Vec<AgentTaskId>` of the rows
/// it released. Phase 2.5 widens that to a `Vec<RecoveryRecord>` so
/// the caller can distinguish "requeued for another attempt" from
/// "failed closed because the budget was used up" without a second
/// round trip to `get()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryRecord {
    /// The id of the row the sweep touched.
    pub id: AgentTaskId,
    /// What Phase 2.5 did with the row.
    pub action: RecoveryAction,
}

impl RecoveryRecord {
    /// Construct a record for a row that was requeued.
    #[must_use]
    pub const fn requeued(id: AgentTaskId) -> Self {
        Self {
            id,
            action: RecoveryAction::Requeue,
        }
    }

    /// Construct a record for a row that was failed closed with
    /// the given reason.
    #[must_use]
    pub const fn failed_closed(id: AgentTaskId, reason: FailureReason) -> Self {
        Self {
            id,
            action: RecoveryAction::FailClosed(reason),
        }
    }
}

/// Apply the Phase 2.5 recovery matrix to `task` in `context`.
///
/// This is the single pure entry point every Phase 2.5 call site
/// shares. The decision depends only on the row's schema fields
/// (kind, status, attempt, `max_attempts`, state) and the caller's
/// context — never on external state — so two workers that read
/// the row through `get()` always agree on the classification.
///
/// See the module-level documentation on [`super::recovery`] for
/// the full decision table.
#[must_use]
pub fn classify_recovery(task: &AgentTask, context: RecoveryContext) -> RecoveryAction {
    // Step 1: unsafe prepared-operation recovery is the highest-priority
    // fail-closed rule because it protects against double-executing a
    // staged external side-effect. It applies regardless of the
    // recovery context: any worker that looks at a tool-runtime
    // awaiting-confirmation row with a prepared op must fail it
    // closed, even if a phantom lease is somehow still alive.
    if task.kind == TaskKind::ToolRuntime
        && task.status == TaskStatus::AwaitingConfirmation
        && task.has_prepared_operation()
    {
        return RecoveryAction::FailClosed(FailureReason::UnsafePreparedOperationRecovery);
    }

    // Step 2: context-specific classification.
    match (context, task.status) {
        // Acquisition-time rules.
        //
        // Only `Pending` rows are runnable. `can_be_leased` already
        // encodes that predicate, so every other status (queued,
        // waiting, running, terminal) silently classifies as
        // NoAction — the acquisition path will see `None` via the
        // existing CAS check regardless.
        (RecoveryContext::AcquisitionAttempt, TaskStatus::Pending) => {
            if task.is_budget_exhausted() {
                RecoveryAction::FailClosed(FailureReason::RetryBudgetExhausted)
            } else {
                RecoveryAction::NoAction
            }
        }

        // Expired-lease rules. The sweep only ever observes `Running`
        // rows — the lease-expiry index is populated exclusively by
        // [`crate::journal::store::Inner::register_runnable_lease_indexes`]
        // from `Running` transitions. Anything else is a bookkeeping bug
        // and the store surfaces it as an error, not a recovery decision.
        (RecoveryContext::ExpiredLease, TaskStatus::Running) => {
            if task.is_budget_exhausted() {
                RecoveryAction::FailClosed(FailureReason::LeaseExpiredBudgetExhausted)
            } else {
                RecoveryAction::Requeue
            }
        }

        // Every other combination is safe for the caller's context —
        // the call site was going to leave the row alone anyway.
        _ => RecoveryAction::NoAction,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::{LeaseId, SuspensionPayload, WorkerId};
    use crate::journal::task_state::TaskState;
    use agent_sdk_core::{
        AgentContinuation, AgentState, ContinuationEnvelope, ListenExecutionContext, ThreadId,
        TokenUsage,
    };
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread() -> ThreadId {
        ThreadId::from_string("t-recovery")
    }

    fn sample_continuation() -> ContinuationEnvelope {
        let thread = thread();
        ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        })
    }

    fn sample_prepared_op() -> ListenExecutionContext {
        ListenExecutionContext {
            operation_id: "op-recovery".into(),
            revision: 1,
            snapshot: serde_json::json!({"preview": true}),
            expires_at: None,
        }
    }

    fn fresh_root(budget: u32) -> AgentTask {
        AgentTask::new_root_turn(thread(), t0(), budget)
    }

    /// Drive a fresh root up to `status` via the real transition
    /// helpers so the row passes `validate()` end-to-end. This is the
    /// shared factory every table-driven test uses to make sure the
    /// matrix is never evaluated on a hand-crafted invariant violation.
    fn root_in_status(status: TaskStatus, budget: u32, attempt: u32) -> Result<AgentTask> {
        let mut root = fresh_root(budget);
        for i in 0..attempt {
            let offset = i64::from(i);
            let running = root.mark_running(
                WorkerId::from_string(format!("w-{i}")),
                LeaseId::from_string(format!("l-{i}")),
                t_plus(60 + offset),
                t_plus(1 + offset),
            )?;
            root = running.release_lease(t_plus(2 + offset))?;
        }
        match status {
            TaskStatus::Pending => Ok(root),
            TaskStatus::Queued => root.admit_as_queued(t_plus(100)),
            TaskStatus::Running => root.mark_running(
                WorkerId::from_string("w-run"),
                LeaseId::from_string("l-run"),
                t_plus(200),
                t_plus(150),
            ),
            TaskStatus::WaitingOnChildren => {
                let running = root.mark_running(
                    WorkerId::from_string("w-run"),
                    LeaseId::from_string("l-run"),
                    t_plus(200),
                    t_plus(150),
                )?;
                running.wait_on_children(
                    1,
                    SuspensionPayload {
                        continuation: sample_continuation(),
                        suspended_messages: Vec::new(),
                    },
                    Vec::new(),
                    t_plus(160),
                )
            }
            TaskStatus::AwaitingConfirmation => {
                let running = root.mark_running(
                    WorkerId::from_string("w-run"),
                    LeaseId::from_string("l-run"),
                    t_plus(200),
                    t_plus(150),
                )?;
                running.await_confirmation(sample_continuation(), None, t_plus(160))
            }
            TaskStatus::Completed => {
                let running = root.mark_running(
                    WorkerId::from_string("w-run"),
                    LeaseId::from_string("l-run"),
                    t_plus(200),
                    t_plus(150),
                )?;
                running.complete(t_plus(170))
            }
            TaskStatus::Failed => {
                let running = root.mark_running(
                    WorkerId::from_string("w-run"),
                    LeaseId::from_string("l-run"),
                    t_plus(200),
                    t_plus(150),
                )?;
                running.fail("boom".into(), t_plus(170))
            }
            TaskStatus::Cancelled => root.cancel(t_plus(180)),
        }
        .map_err(Into::into)
    }

    /// Drive a fresh tool-runtime child to [`TaskStatus::AwaitingConfirmation`]
    /// with the given prepared-operation choice. Children are leaves
    /// so the factory only needs this one shape.
    fn tool_child_in_awaiting_confirmation(prepared: Option<ListenExecutionContext>) -> AgentTask {
        let root = fresh_root(5);
        let child = AgentTask::new_child(&root, TaskKind::ToolRuntime, t_plus(1), 3)
            .expect("tool child constructs");
        let running = child
            .mark_running(
                WorkerId::from_string("w-tool"),
                LeaseId::from_string("l-tool"),
                t_plus(60),
                t_plus(2),
            )
            .expect("tool child runs");
        running
            .await_confirmation(sample_continuation(), prepared, t_plus(3))
            .expect("tool child awaits")
    }

    // ── FailureReason wire-format lock ────────────────────────────

    #[test]
    fn failure_reason_error_prefix_is_stable() {
        assert_eq!(
            FailureReason::RetryBudgetExhausted.error_prefix(),
            "retry_budget_exhausted",
        );
        assert_eq!(
            FailureReason::LeaseExpiredBudgetExhausted.error_prefix(),
            "lease_expired_budget_exhausted",
        );
        assert_eq!(
            FailureReason::UnsafePreparedOperationRecovery.error_prefix(),
            "unsafe_prepared_operation_recovery",
        );
    }

    #[test]
    fn failure_reason_error_message_includes_id_and_budget() {
        let mut task = fresh_root(3);
        task.attempt = 3;
        let message = FailureReason::RetryBudgetExhausted.error_message(&task);
        assert!(
            message.starts_with("retry_budget_exhausted:"),
            "unexpected: {message}"
        );
        assert!(message.contains(task.id.as_str()), "missing id: {message}");
        assert!(message.contains("attempt=3"), "missing attempt: {message}");
        assert!(message.contains("max=3"), "missing max: {message}");
    }

    #[test]
    fn failure_reason_wire_format_is_snake_case() -> Result<()> {
        let cases = [
            (
                FailureReason::RetryBudgetExhausted,
                "\"retry_budget_exhausted\"",
            ),
            (
                FailureReason::LeaseExpiredBudgetExhausted,
                "\"lease_expired_budget_exhausted\"",
            ),
            (
                FailureReason::UnsafePreparedOperationRecovery,
                "\"unsafe_prepared_operation_recovery\"",
            ),
        ];
        for (reason, wire) in cases {
            let encoded = serde_json::to_string(&reason).context("serialize")?;
            assert_eq!(encoded, wire, "wire drift for {reason:?}");
            let decoded: FailureReason = serde_json::from_str(&encoded).context("deserialize")?;
            assert_eq!(decoded, reason);
        }
        Ok(())
    }

    // ── Recovery matrix, table-driven ─────────────────────────────

    #[test]
    fn classify_recovery_matrix_is_exhaustive() -> Result<()> {
        // Acquisition context: Pending rows are the only ones that
        // participate; every other status is NoAction.
        let pending_ok = root_in_status(TaskStatus::Pending, 3, 0)?;
        assert_eq!(
            classify_recovery(&pending_ok, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
            "pending w/ budget must be NoAction"
        );

        let pending_exhausted = root_in_status(TaskStatus::Pending, 2, 2)?;
        assert_eq!(
            classify_recovery(&pending_exhausted, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::FailClosed(FailureReason::RetryBudgetExhausted),
            "pending w/ attempt == max must fail closed"
        );

        // Exhausted Pending under the sweep context is a NoAction —
        // sweep never sees Pending rows, and the matrix makes that
        // path an explicit no-op rather than an error.
        assert_eq!(
            classify_recovery(&pending_exhausted, RecoveryContext::ExpiredLease),
            RecoveryAction::NoAction,
        );

        // ExpiredLease context: Running rows are the only ones the
        // sweep pulls off the index.
        let running_ok = root_in_status(TaskStatus::Running, 3, 0)?;
        assert_eq!(
            classify_recovery(&running_ok, RecoveryContext::ExpiredLease),
            RecoveryAction::Requeue,
            "running w/ budget must requeue"
        );

        // A row that has already burned all its attempts is one more
        // expired lease away from the cap: attempt == max_attempts
        // means the worker holding this lease is already on its final
        // attempt and any sweep must fail it closed.
        let running_exhausted = root_in_status(TaskStatus::Running, 1, 0)?;
        assert_eq!(running_exhausted.attempt, 1);
        assert_eq!(running_exhausted.max_attempts, 1);
        assert_eq!(
            classify_recovery(&running_exhausted, RecoveryContext::ExpiredLease),
            RecoveryAction::FailClosed(FailureReason::LeaseExpiredBudgetExhausted),
        );

        // Unsafe prepared operation: tool-runtime children with a
        // staged listen/execute op must fail closed regardless of the
        // recovery context, to protect against double-execution.
        let unsafe_child = tool_child_in_awaiting_confirmation(Some(sample_prepared_op()));
        assert_eq!(
            classify_recovery(&unsafe_child, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::FailClosed(FailureReason::UnsafePreparedOperationRecovery),
        );
        assert_eq!(
            classify_recovery(&unsafe_child, RecoveryContext::ExpiredLease),
            RecoveryAction::FailClosed(FailureReason::UnsafePreparedOperationRecovery),
        );

        // Same shape minus the prepared operation: safe again.
        let safe_child = tool_child_in_awaiting_confirmation(None);
        assert_eq!(
            classify_recovery(&safe_child, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
        );

        // Root turns awaiting confirmation are out of scope for the
        // unsafe-prepared-operation rule — the transport owns their
        // resume, so Phase 2.5 leaves them alone.
        let safe_root_await = root_in_status(TaskStatus::AwaitingConfirmation, 3, 0)?;
        assert_eq!(
            classify_recovery(&safe_root_await, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
        );

        // Waiting-on-children and queued rows always classify as
        // NoAction; the matrix does not touch them.
        let waiting = root_in_status(TaskStatus::WaitingOnChildren, 3, 0)?;
        assert_eq!(
            classify_recovery(&waiting, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
        );
        let queued = root_in_status(TaskStatus::Queued, 3, 0)?;
        assert_eq!(
            classify_recovery(&queued, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
        );

        // Terminal rows are always NoAction — the row is already
        // closed out.
        for status in [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            let terminal = root_in_status(status, 3, 0)?;
            assert_eq!(
                classify_recovery(&terminal, RecoveryContext::AcquisitionAttempt),
                RecoveryAction::NoAction,
                "terminal {status:?} must be NoAction"
            );
            assert_eq!(
                classify_recovery(&terminal, RecoveryContext::ExpiredLease),
                RecoveryAction::NoAction,
            );
        }

        Ok(())
    }

    #[test]
    fn recovery_action_helpers_classify_every_variant() {
        assert!(!RecoveryAction::NoAction.is_fail_closed());
        assert!(!RecoveryAction::NoAction.is_requeue());

        assert!(!RecoveryAction::Requeue.is_fail_closed());
        assert!(RecoveryAction::Requeue.is_requeue());

        let fc = RecoveryAction::FailClosed(FailureReason::RetryBudgetExhausted);
        assert!(fc.is_fail_closed());
        assert!(!fc.is_requeue());
    }

    #[test]
    fn recovery_record_constructors_set_expected_shape() {
        let id = AgentTaskId::from_string("task_abc");
        let requeued = RecoveryRecord::requeued(id.clone());
        assert_eq!(requeued.id, id);
        assert!(requeued.action.is_requeue());

        let failed = RecoveryRecord::failed_closed(id.clone(), FailureReason::RetryBudgetExhausted);
        assert_eq!(failed.id, id);
        assert_eq!(
            failed.action,
            RecoveryAction::FailClosed(FailureReason::RetryBudgetExhausted)
        );
    }

    // ── Regression: the recovery matrix is strictly deterministic ──

    #[test]
    fn classification_is_deterministic_for_identical_rows() -> Result<()> {
        let row = root_in_status(TaskStatus::Running, 2, 1)?;
        assert_eq!(
            classify_recovery(&row, RecoveryContext::ExpiredLease),
            classify_recovery(&row, RecoveryContext::ExpiredLease),
        );
        Ok(())
    }

    #[test]
    fn unsafe_prepared_operation_check_is_orthogonal_to_task_state_none() -> Result<()> {
        // Sanity: a root turn in AwaitingConfirmation with a prepared
        // op does NOT trigger the unsafe rule. Only tool-runtime
        // children do, because roots defer their resume to the
        // confirmation transport.
        let root = fresh_root(3);
        let running = root.mark_running(
            WorkerId::from_string("w-root"),
            LeaseId::from_string("l-root"),
            t_plus(60),
            t_plus(1),
        )?;
        let awaiting = running.await_confirmation(
            sample_continuation(),
            Some(sample_prepared_op()),
            t_plus(2),
        )?;
        assert!(matches!(
            awaiting.state,
            TaskState::AwaitingConfirmation { .. }
        ));
        assert_eq!(
            classify_recovery(&awaiting, RecoveryContext::AcquisitionAttempt),
            RecoveryAction::NoAction,
        );
        Ok(())
    }
}
