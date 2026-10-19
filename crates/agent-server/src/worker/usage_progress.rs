//! The worker's close of an LLM attempt of a root turn, with its live-usage
//! event.
//!
//! The attempt row's close and its [`AgentEvent::UsageProgress`] commit in one
//! transition under the turn's lease, so a reader that folds the events sees
//! exactly the closed attempts. When the lease is gone the transition is
//! refused and writes nothing; the worker then closes its own row the old,
//! unguarded way, because attempt rows are the billing record and the
//! recovery that would otherwise close the row records zero tokens.

use agent_sdk_foundation::{AgentEvent, ThreadId, TokenUsage};
use anyhow::{Context, Result};
use time::OffsetDateTime;

use super::root_turn::RootTurnDeps;
use crate::journal::CommittedEvent;
use crate::journal::commit::{CommitOutcome, CompletedTurnCommit};
use crate::journal::task::{AgentTaskId, LeaseId, WorkerId};
use crate::journal::thread::ThreadStateConflict;
use crate::journal::turn_attempt::{CloseAttemptParams, TurnAttempt};
use crate::journal::{Rejected, Transition};

/// The root turn that closes an attempt: its task, the lease it runs under,
/// and the thread its events go to.
#[derive(Clone, Debug)]
pub struct AttemptOwner {
    pub task: AgentTaskId,
    pub worker: WorkerId,
    pub lease: LeaseId,
    pub thread: ThreadId,
}

impl AttemptOwner {
    pub(crate) fn of(bootstrap: &super::bootstrap::WorkerBootstrapContext) -> Self {
        Self {
            task: bootstrap.task_id.clone(),
            worker: bootstrap.worker_id.clone(),
            lease: bootstrap.lease_id.clone(),
            thread: bootstrap.thread_id.clone(),
        }
    }
}

/// The [`AgentEvent::UsageProgress`] for closing `attempt` with `params`.
///
/// `turn_usage` sums `params` and every closed row of `rows` other than
/// `attempt` (the task's attempt rows as read before the close).
#[must_use]
pub fn usage_progress_event(
    attempt: &TurnAttempt,
    params: &CloseAttemptParams,
    rows: &[TurnAttempt],
    context_tokens: Option<u32>,
) -> AgentEvent {
    let usage = TokenUsage {
        input_tokens: params.input_tokens,
        output_tokens: params.output_tokens,
        cached_input_tokens: params.cached_input_tokens,
        cache_creation_input_tokens: params.cache_creation_input_tokens,
    };
    let mut turn_usage = usage.clone();
    for row in rows
        .iter()
        .filter(|row| row.id != attempt.id && row.is_closed())
    {
        turn_usage.add(&TokenUsage {
            input_tokens: row.input_tokens.unwrap_or(0),
            output_tokens: row.output_tokens.unwrap_or(0),
            cached_input_tokens: row.cached_input_tokens.unwrap_or(0),
            cache_creation_input_tokens: row.cache_creation_input_tokens.unwrap_or(0),
        });
    }
    AgentEvent::UsageProgress {
        task_id: attempt.task_id.to_string(),
        attempt_id: attempt.id.to_string(),
        attempt_number: attempt.attempt_number,
        provider: attempt.provider.clone(),
        model: attempt.requested_model.clone(),
        response_model: params.response_model.clone(),
        route_provider: params.route_provider.clone(),
        usage,
        turn_usage,
        context_tokens,
    }
}

/// The transition that closes `attempt` with `params` and appends `event`
/// to the owner's thread, guarded on the owner's lease.
#[must_use]
pub fn close_attempt_transition(
    owner: &AttemptOwner,
    attempt: &TurnAttempt,
    params: CloseAttemptParams,
    event: AgentEvent,
) -> Transition {
    Transition::close_attempt_owned(
        (
            owner.task.clone(),
            owner.worker.clone(),
            owner.lease.clone(),
        ),
        attempt.id.clone(),
        params,
        owner.thread.clone(),
        vec![event],
    )
}

/// Close `attempt` with `params` and commit its `UsageProgress` with it.
///
/// Errors as [`TurnAttemptStore::close_attempt`] does: an attempt already
/// closed (by recovery) surfaces as
/// [`TurnAttemptSchemaError::AlreadyClosed`](crate::journal::TurnAttemptSchemaError::AlreadyClosed).
///
/// [`TurnAttemptStore::close_attempt`]: crate::journal::TurnAttemptStore::close_attempt
pub(crate) async fn close_attempt_reporting_usage(
    deps: &RootTurnDeps<'_>,
    owner: &AttemptOwner,
    attempt: &TurnAttempt,
    params: CloseAttemptParams,
    context_tokens: Option<u32>,
    now: OffsetDateTime,
) -> Result<()> {
    let rows = match deps.attempt_store.list_by_task(&attempt.task_id).await {
        Ok(rows) => rows,
        Err(error) => {
            log::warn!(
                "read the attempts of task {} to report attempt {}'s usage: {error:#}; \
                 closing it without the event",
                attempt.task_id,
                attempt.id,
            );
            record_refused(Refused::Store);
            return close_unreported(deps, attempt, params, now).await;
        }
    };
    let event = usage_progress_event(attempt, &params, &rows, context_tokens);
    let transition = close_attempt_transition(owner, attempt, params.clone(), event);
    let error = match deps.task_store.commit(transition, now).await {
        Ok(committed) => {
            deps.event_notifier.notify(&committed.events);
            record_reported();
            return Ok(());
        }
        Err(error) => error,
    };
    let refused = match error {
        Rejected::Guard { reason, .. } => (Refused::LeaseLost, reason),
        Rejected::Store(store) if store.downcast_ref::<ThreadStateConflict>().is_some() => {
            (Refused::Purged, format!("{store:#}"))
        }
        // The callers read an already-closed row (recovery closed it) off the
        // error's top level, as `close_attempt` returns it.
        Rejected::Store(store)
            if store
                .downcast_ref::<crate::journal::TurnAttemptSchemaError>()
                .is_some() =>
        {
            return Err(store);
        }
        Rejected::Store(store) => {
            record_refused(Refused::Store);
            return Err(store)
                .with_context(|| format!("close attempt {} with its usage event", attempt.id));
        }
    };
    let (refused, reason) = refused;
    record_refused(refused);
    log::info!(
        "attempt {} of task {} closes without its usage event: {reason}",
        attempt.id,
        attempt.task_id,
    );
    close_unreported(deps, attempt, params, now).await
}

/// Commit a completed turn (see `commit_completed_turn_shifting_slot`) with
/// the `UsageProgress` of the attempt its commit closes.
///
/// `attempt` is the closing row with the call's last request prompt size.
/// The event joins the commit's events, so the close and the event commit
/// together.
pub(crate) async fn commit_completed_turn_reporting_usage(
    mut commit: CompletedTurnCommit,
    (attempt, context_tokens): (&TurnAttempt, Option<u32>),
    (worker, lease): (&WorkerId, &LeaseId),
    deps: &RootTurnDeps<'_>,
) -> Result<CommitOutcome> {
    add_closing_event(
        &mut commit.events,
        deps,
        attempt,
        &commit.close_attempt_params,
        context_tokens,
    )
    .await;
    let outcome =
        super::root_turn::commit_completed_turn_shifting_slot(commit, worker, lease, deps).await?;
    record_committed(&outcome.committed_events);
    Ok(outcome)
}

/// The `UsageProgress` of an attempt that a completed turn's commit closes
/// itself, inserted before the turn's `TurnComplete` in `events`. Without
/// the task's attempt rows the turn commits without it.
async fn add_closing_event(
    events: &mut Vec<AgentEvent>,
    deps: &RootTurnDeps<'_>,
    attempt: &TurnAttempt,
    params: &CloseAttemptParams,
    context_tokens: Option<u32>,
) {
    let rows = match deps.attempt_store.list_by_task(&attempt.task_id).await {
        Ok(rows) => rows,
        Err(error) => {
            log::warn!(
                "read the attempts of task {} to report attempt {}'s usage: {error:#}; \
                 the turn commits without the event",
                attempt.task_id,
                attempt.id,
            );
            record_refused(Refused::Store);
            return;
        }
    };
    let event = usage_progress_event(attempt, params, &rows, context_tokens);
    let at = events
        .iter()
        .position(|event| matches!(event, AgentEvent::TurnComplete { .. }))
        .unwrap_or(events.len());
    events.insert(at, event);
}

/// Count the `UsageProgress` a completed turn's commit wrote.
fn record_committed(events: &[CommittedEvent]) {
    if events
        .iter()
        .any(|committed| matches!(committed.event, AgentEvent::UsageProgress { .. }))
    {
        record_reported();
    }
}

/// Close the worker's own attempt row with its billed usage and no event.
async fn close_unreported(
    deps: &RootTurnDeps<'_>,
    attempt: &TurnAttempt,
    params: CloseAttemptParams,
    now: OffsetDateTime,
) -> Result<()> {
    deps.attempt_store
        .close_attempt(&attempt.id, params, now)
        .await
        .map(drop)
}

/// Why a usage close wrote no event.
#[derive(Clone, Copy, Debug)]
enum Refused {
    LeaseLost,
    Purged,
    Store,
}

#[cfg(feature = "otel")]
fn record_reported() {
    crate::observability::ServerMetrics::global()
        .record_progress_event(crate::observability::attrs::PROGRESS_USAGE);
}

#[cfg(not(feature = "otel"))]
const fn record_reported() {}

#[cfg(feature = "otel")]
fn record_refused(refused: Refused) {
    use crate::observability::attrs;
    let reason = match refused {
        Refused::LeaseLost => attrs::PROGRESS_LEASE_LOST,
        Refused::Purged => attrs::PROGRESS_PURGED,
        Refused::Store => attrs::PROGRESS_STORE,
    };
    crate::observability::ServerMetrics::global().record_progress_refused(reason);
}

#[cfg(not(feature = "otel"))]
const fn record_refused(_refused: Refused) {}
