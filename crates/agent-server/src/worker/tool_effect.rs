//! A tool child run as an effect ([`crate::journal::effect`]).

use std::future::Future;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{PendingToolCallInfo, ToolReplay, ToolResult};
use anyhow::{Context, bail};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::effect::{
    EffectIntent, EffectKey, EffectRecord, EffectRun, EffectState, EffectSubject, EffectWrite,
    TOOL_INTERRUPTED_PREFIX, run_effect,
};
use crate::journal::task::TaskStatus;
use crate::journal::transition::{DurableStore, Guard, Rejected, Settle, Transition, Write};
use crate::worker::tool_task::{
    ToolEventCollector, ToolRun, ToolTaskBootstrap, ToolTaskOutcome, execute_on, instrumented,
    run_tool_executor, settle_transition, settled_outcome,
};

/// Run a tool-runtime child task as an effect.
///
/// A call declared [`ToolReplay::Safe`] runs as
/// [`execute_tool_task`](crate::worker::tool_task::execute_tool_task) does,
/// with no effect record: running it again is its recovery. Any other call
/// is recovered from its record first:
///
/// - no record: it never ran. Its `Started` record commits under the
///   child's lease, then the executor runs, then the child's settle, its
///   `ToolCallEnd` and the record's outcome commit in one transition. A
///   record that cannot be claimed (another worker holds it, or the store
///   refused it) fails the child without running the tool.
/// - `Started`: it may have run. The child fails with a
///   [`TOOL_INTERRUPTED_PREFIX`] result and the record is finished
///   `Interrupted`, in one transition; the tool is not run again.
/// - `Completed`: it ran to the end and the child should already be
///   settled; running it again is refused with an error.
/// - any other outcome: the child fails as interrupted.
///
/// # Errors
/// A store error reading the record, a refused duplicate of a completed
/// call, or a failed settle (for instance after the child lost its lease).
pub async fn run_tool_effect<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn DurableStore,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    if bootstrap.tool_call.replay == ToolReplay::Safe {
        return execute_on(bootstrap, task_store, cancel, executor, now).await;
    }

    let key = EffectKey::tool(&bootstrap.task_id, &bootstrap.tool_call.id);
    let prior = task_store
        .effect(&key)
        .await
        .context("failed to check prior intent")?;
    if let Some(prior) = prior {
        if prior.state == EffectState::Completed {
            bail!(
                "operation {key} already completed at {}; refusing duplicate execution",
                prior.updated_at,
            );
        }
        return interrupt_unsafe_replay(&bootstrap, task_store, &prior, now).await;
    }

    if cancel.is_cancelled() {
        return Ok(ToolTaskOutcome::Cancelled);
    }
    instrumented(Box::pin(run_claimed(
        bootstrap, task_store, executor, key, now,
    )))
    .await
}

async fn run_claimed<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn DurableStore,
    executor: F,
    key: EffectKey,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    let intent = tool_intent(&bootstrap);
    let start = Transition {
        guards: vec![Guard::TaskIs {
            task: bootstrap.task_id.clone(),
            status: TaskStatus::Running,
            lease: Some((bootstrap.worker_id.clone(), bootstrap.lease_id.clone())),
        }],
        writes: Vec::new(),
    };
    let run = run_effect(
        task_store,
        intent,
        start,
        || run_tool_executor(&bootstrap, executor),
        |run: &ToolRun| Ok((settle_transition(&bootstrap, run)?, end_state(run))),
        now,
    )
    .await
    .with_context(|| format!("settle tool child {}", bootstrap.task_id))?;
    match run {
        EffectRun::Ran { value, committed } => settled_outcome(value, committed),
        EffectRun::Refused(rejected) => {
            fail_closed_on_claim_error(&bootstrap, task_store, &key, &rejected, now).await
        }
    }
}

/// The effect a tool child's call starts.
#[must_use]
pub fn tool_intent(bootstrap: &ToolTaskBootstrap) -> EffectIntent {
    EffectIntent {
        key: EffectKey::tool(&bootstrap.task_id, &bootstrap.tool_call.id),
        replay: bootstrap.tool_call.replay,
        owner: bootstrap.task_id.clone(),
        thread: bootstrap.thread_id.clone(),
        subject: EffectSubject::Tool {
            call_id: bootstrap.tool_call.id.clone(),
            name: bootstrap.tool_call.name.clone(),
            input: bootstrap.tool_call.input.clone(),
        },
        attempt: 1,
    }
}

fn end_state(run: &ToolRun) -> EffectState {
    match &run.result {
        Ok(_) => EffectState::Completed,
        Err(err) => EffectState::Failed {
            error: format!("{err:#}"),
        },
    }
}

/// Settle a tool call whose earlier attempt started and never finished,
/// without running it again.
///
/// The tool may have changed state before the process stopped, and it
/// is not declared replay-safe, so the model gets an interrupted result
/// instead of a second run.
async fn interrupt_unsafe_replay(
    bootstrap: &ToolTaskBootstrap,
    task_store: &dyn DurableStore,
    prior: &EffectRecord,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome> {
    let error_msg = format!(
        "{TOOL_INTERRUPTED_PREFIX} tool '{}' started before the process stopped and did not \
         finish (operation {}, last status {}). It was not run again because it is not \
         declared replay-safe. Check its effects before you call it again.",
        bootstrap.tool_call.name,
        prior.key,
        status_words(&prior.state),
    );
    let mut transition = Transition::settle(
        bootstrap.task_id.clone(),
        bootstrap.worker_id.clone(),
        bootstrap.lease_id.clone(),
        Settle::Fail {
            error: error_msg.clone(),
        },
        bootstrap.thread_id.clone(),
        vec![failure_tool_call_end(bootstrap, &error_msg)],
    );
    if prior.state.is_started() {
        transition.writes.insert(
            0,
            Write::Effect(EffectWrite::Finish {
                key: prior.key.clone(),
                owner: prior.owner.clone(),
                state: EffectState::Interrupted {
                    captured_output: None,
                },
            }),
        );
    }
    let outcome = commit_failed_settle(task_store, transition, error_msg, now)
        .await
        .with_context(|| format!("fail interrupted tool task {}", bootstrap.task_id))?;
    if prior.state.is_started() {
        crate::journal::effect::record_effect(prior.kind(), "reclaimed");
    }
    log::info!(
        effect_key = prior.key.as_str(),
        task_id = bootstrap.task_id.as_str(),
        prior_state = status_words(&prior.state);
        "an unsafe tool call that already started was settled as interrupted, not run again"
    );
    Ok(outcome)
}

const fn status_words(state: &EffectState) -> &'static str {
    match state {
        EffectState::Started => "Started",
        EffectState::Completed => "Completed",
        EffectState::Failed { .. } => "Failed",
        EffectState::Interrupted { .. } => "Interrupted",
        EffectState::Salvaged { .. } => "Salvaged",
        EffectState::Abandoned => "Abandoned",
    }
}

/// Fail the child task without executing when its effect could not be
/// claimed (another worker's record, a lost lease or a store failure).
async fn fail_closed_on_claim_error(
    bootstrap: &ToolTaskBootstrap,
    task_store: &dyn DurableStore,
    key: &EffectKey,
    rejected: &Rejected,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome> {
    let error_msg = format!(
        "fail-closed: could not claim execution intent for tool '{}' (operation {key}): \
         {rejected:#}",
        bootstrap.tool_call.name,
    );
    let transition = Transition::settle(
        bootstrap.task_id.clone(),
        bootstrap.worker_id.clone(),
        bootstrap.lease_id.clone(),
        Settle::Fail {
            error: error_msg.clone(),
        },
        bootstrap.thread_id.clone(),
        vec![failure_tool_call_end(bootstrap, &error_msg)],
    );
    commit_failed_settle(task_store, transition, error_msg, now)
        .await
        .with_context(|| {
            format!(
                "fail_task after intent-claim failure for {}",
                bootstrap.task_id
            )
        })
}

/// The `ToolCallEnd` that closes a failed tool call with `error`.
fn failure_tool_call_end(bootstrap: &ToolTaskBootstrap, error: &str) -> AgentEvent {
    AgentEvent::tool_call_end(
        &bootstrap.tool_call.id,
        &bootstrap.tool_call.name,
        &bootstrap.tool_call.display_name,
        ToolResult {
            success: false,
            output: error.to_owned(),
            artifact: None,
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        },
    )
}

async fn commit_failed_settle(
    task_store: &dyn DurableStore,
    transition: Transition,
    error: String,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome> {
    let committed = task_store
        .commit(transition, now)
        .await
        .map_err(anyhow::Error::from)?;
    let (child, parent) = committed
        .settled
        .into_iter()
        .next()
        .context("a settle transition returns its settled task")?;
    Ok(ToolTaskOutcome::Failed {
        child,
        parent,
        error,
        committed_events: committed.events,
    })
}
