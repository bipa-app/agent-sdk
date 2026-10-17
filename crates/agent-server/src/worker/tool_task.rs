//! Phase 5.1 tool-runtime worker: bootstrap, execution, and lifecycle
//! for [`TaskKind::ToolRuntime`] child tasks.
//!
//! When a root turn suspends at the tool boundary (Phase 4.4), it
//! spawns one `ToolRuntime` child task per tool call. This module
//! provides the worker path that:
//!
//! 1. **Bootstraps** the child task by reading the parent's durable
//!    [`TaskState`] to reconstruct the [`PendingToolCallInfo`] for
//!    exactly one tool call.
//! 2. **Executes** the tool through a caller-supplied async callback
//!    (the tool registry and dispatch live outside `agent-server`).
//! 3. **Completes or fails** the child task through the journal,
//!    letting the store recompute the parent's `pending_child_count`
//!    so the parent can resume when every child reaches a terminal
//!    state.
//!
//! # Positional mapping
//!
//! Child tasks are spawned in the same order as the parent's
//! [`AgentContinuation::pending_tool_calls`](agent_sdk_foundation::AgentContinuation::pending_tool_calls) array. Each child
//! carries a `spawn_index` field set during
//! [`SpawnChildrenExt::spawn_tool_children`](crate::journal::SpawnChildrenExt::spawn_tool_children) that records its position
//! within the batch. The bootstrap step reads this index directly
//! instead of relying on sibling sort order, making the mapping
//! correct regardless of the store's `list_children` return order.
//!
//! # Cancellation and lease awareness
//!
//! The worker checks the supplied [`CancellationToken`] **before**
//! tool execution. If cancelled at that point, it returns early
//! without driving the child to a terminal state — the
//! [`AgentTaskStore::cancel_tree`] sweep handles cleanup.
//!
//! After the executor returns, the result is **always** committed
//! to the journal regardless of cancellation state, because the
//! tool's side effects have already been applied and dropping the
//! result would cause the recovery matrix to re-execute the tool.
//! Lease expiry is caught by the store's CAS guards on
//! `complete_task` / `fail_task`.
//!
//! [`TaskState`]: crate::journal::task_state::TaskState
//! [`PendingToolCallInfo`]: agent_sdk_foundation::PendingToolCallInfo
//! [`AgentContinuation`]: agent_sdk_foundation::AgentContinuation
//! [`AgentTaskStore::cancel_tree`]: crate::journal::store::AgentTaskStore::cancel_tree
//! [`CancellationToken`]: tokio_util::sync::CancellationToken

use std::future::Future;
use std::sync::{Arc, Mutex};

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{PendingToolCallInfo, ThreadId, ToolResult};
use anyhow::{Context, bail, ensure};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::journal::committed_event::CommittedEvent;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};
use crate::journal::transition::{Committed, DurableStore, Settle, Transition};
use crate::worker::activity::ActivityBeacon;
use crate::worker::tool_stall::{StallSignal, StallTrip, ToolStallReport};
use agent_sdk_foundation::events::TerminalReason;

// ─────────────────────────────────────────────────────────────────────
// Tool event collector
// ─────────────────────────────────────────────────────────────────────

/// Collects [`AgentEvent`]s emitted by a tool executor during execution.
///
/// The executor receives a `ToolEventCollector` and calls [`emit`](Self::emit)
/// to record progress events (e.g. [`AgentEvent::ToolProgress`]).  After the
/// executor returns, [`execute_tool_task`] drains the collected events and
/// commits them to the event repository before the `ToolCallEnd` event.
///
/// The collector is `Clone + Send + Sync` so it can be shared across
/// spawned tasks within the executor.
#[derive(Clone, Default)]
pub struct ToolEventCollector {
    events: Arc<Mutex<Vec<AgentEvent>>>,
    /// Live progress signal for the subagent stall budget.
    ///
    /// Buffered events do not become durable until the executor returns,
    /// so a child parked on ONE long tool call (a 40-minute build) commits
    /// nothing at all mid-flight and reads as silent. Bumping the beacon
    /// on every [`emit`](Self::emit) makes that in-flight progress visible
    /// to the stall probe *while the tool is still running*, via the tool
    /// task's own heartbeat rather than via the journal.
    activity: ActivityBeacon,
}

impl ToolEventCollector {
    /// Create a new empty collector whose progress is not reported to any
    /// stall budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a collector that reports live progress to `activity`.
    ///
    /// The beacon must belong to the **tool task's own** heartbeat — that
    /// is the actor holding this row's lease, and therefore the only one
    /// that can persist what the beacon sees.
    #[must_use]
    pub fn with_activity(activity: ActivityBeacon) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            activity,
        }
    }

    /// Record a progress event. This is a non-blocking operation.
    ///
    /// Also marks the tool as *alive right now* for the stall budget: this
    /// is the only signal a long-running tool produces before it returns.
    pub fn emit(&self, event: AgentEvent) {
        self.activity.bump(OffsetDateTime::now_utc());
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Mark the tool alive without reporting an event: evidence of work
    /// for the stall budget, such as output bytes or process activity that
    /// the host reports some other way.
    pub fn note_progress(&self) {
        self.activity.bump(OffsetDateTime::now_utc());
    }

    /// Drain all collected events, leaving the collector empty.
    ///
    /// Used by the worker to flush events at commit time, and by
    /// downstream crates that wrap the collector in an adapter (such
    /// as `agent-service-host`'s `CollectorEventStore`) to verify
    /// forwarding behaviour.
    #[must_use]
    pub fn drain(&self) -> Vec<AgentEvent> {
        self.events
            .lock()
            .map(|mut e| std::mem::take(&mut *e))
            .unwrap_or_default()
    }
}
use crate::journal::task_state::TaskState;

// ─────────────────────────────────────────────────────────────────────
// Bootstrap context
// ─────────────────────────────────────────────────────────────────────

/// Trusted bootstrapping inputs for a tool-runtime child task.
///
/// Built by [`resolve_tool_bootstrap`], which validates the child
/// row, reads the parent's durable state, and resolves exactly one
/// [`PendingToolCallInfo`] from the parent's continuation.
#[derive(Clone, Debug)]
pub struct ToolTaskBootstrap {
    /// The acquired child task row.
    pub child_task: AgentTask,
    /// The parent task row (read-only snapshot for context).
    pub parent_task: AgentTask,
    /// Thread the task family is bound to.
    pub thread_id: ThreadId,
    /// The child task's durable identity.
    pub task_id: AgentTaskId,
    /// The worker that owns the lease.
    pub worker_id: WorkerId,
    /// The current lease identifier.
    pub lease_id: LeaseId,
    /// The tool call this child is responsible for executing.
    pub tool_call: PendingToolCallInfo,
    /// This tool task's own live-progress beacon.
    ///
    /// Handed to the [`ToolEventCollector`] so every progress emission
    /// marks the tool alive *while it is still running*, and read once per
    /// tick by the heartbeat that holds this row's lease — the only actor
    /// that can persist it. A [`Default`] beacon (nobody reads it) is
    /// harmless: the tool simply reports no live progress, exactly as
    /// before this field existed.
    pub activity: ActivityBeacon,
    /// Stops the tool when its heartbeat finds it silent for its stall
    /// budget. A [`Default`] signal never trips.
    pub stall: StallSignal,
}

// ─────────────────────────────────────────────────────────────────────
// Bootstrap resolution
// ─────────────────────────────────────────────────────────────────────

/// Validate a running tool-runtime child task and resolve its
/// execution inputs from the parent's durable state.
///
/// # Preconditions
///
/// - `child.kind` must be [`TaskKind::ToolRuntime`].
/// - `child.status` must be [`TaskStatus::Running`].
/// - `child.worker_id` and `child.lease_id` must be present.
/// - `child.parent_id` must point to an existing parent task.
/// - The parent's [`TaskState`] must carry a continuation with a
///   `pending_tool_calls` entry at this child's positional index.
///
/// # Positional index
///
/// The child's `spawn_index` field determines which
/// `PendingToolCallInfo` it owns. This index is set during
/// [`SpawnChildrenExt::spawn_tool_children`](crate::journal::SpawnChildrenExt::spawn_tool_children) and read directly,
/// making the mapping independent of `list_children` ordering.
/// # Errors
///
/// Returns an error if any precondition fails, the parent task
/// cannot be read, or the positional tool-call index is out of
/// bounds.
pub async fn resolve_tool_bootstrap(
    child: AgentTask,
    task_store: &dyn AgentTaskStore,
    activity: ActivityBeacon,
) -> anyhow::Result<ToolTaskBootstrap> {
    // ── Precondition: tool-runtime kind ──────────────────────────
    ensure!(
        child.kind == TaskKind::ToolRuntime,
        "tool bootstrap requires a ToolRuntime task, got {:?}",
        child.kind,
    );

    // ── Precondition: Running status ─────────────────────────────
    ensure!(
        child.status == TaskStatus::Running,
        "tool bootstrap requires a Running task, got {:?}",
        child.status,
    );

    // ── Extract lease fields ─────────────────────────────────────
    let worker_id = child
        .worker_id
        .clone()
        .context("Running task missing worker_id")?;
    let lease_id = child
        .lease_id
        .clone()
        .context("Running task missing lease_id")?;

    // ── Read the parent ─────────────────────────────────���────────
    let parent_id = child
        .parent_id
        .as_ref()
        .context("ToolRuntime task missing parent_id")?;

    let parent = task_store
        .get(parent_id)
        .await
        .context("failed to read parent task")?
        .with_context(|| format!("parent task {parent_id} does not exist"))?;

    // ── Extract continuation from parent state ───────────────────
    let continuation = match &parent.state {
        TaskState::WaitingOnChildren { continuation, .. }
        | TaskState::ReadyToResume { continuation, .. } => &continuation.payload,
        other => {
            bail!("parent task {parent_id} has unexpected state for tool bootstrap: {other:?}")
        }
    };

    // ── Resolve positional index via spawn_index ─────────────────
    let spawn_index = child
        .spawn_index
        .context("ToolRuntime child missing spawn_index")?;

    let child_index = usize::try_from(spawn_index).context("spawn_index exceeds usize")?;

    ensure!(
        child_index < continuation.pending_tool_calls.len(),
        "child spawn_index {child_index} out of bounds for {} pending tool calls on parent {parent_id}",
        continuation.pending_tool_calls.len(),
    );

    let tool_call = continuation.pending_tool_calls[child_index].clone();
    let thread_id = child.thread_id.clone();
    let task_id = child.id.clone();

    Ok(ToolTaskBootstrap {
        child_task: child,
        parent_task: parent,
        thread_id,
        task_id,
        worker_id,
        lease_id,
        tool_call,
        activity,
        stall: StallSignal::default(),
    })
}

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// Result of [`execute_tool_task`].
#[derive(Debug)]
pub enum ToolTaskOutcome {
    /// Tool executed successfully. The child task is now `Completed`
    /// and the parent's `pending_child_count` has been recomputed.
    Completed {
        child: AgentTask,
        parent: Option<AgentTask>,
        result: ToolResult,
        /// `ToolCallEnd` event committed after the state transition.
        committed_events: Vec<CommittedEvent>,
    },
    /// Tool execution failed. The child task is now `Failed` and the
    /// parent's `pending_child_count` has been recomputed.
    Failed {
        child: AgentTask,
        parent: Option<AgentTask>,
        error: String,
        /// `ToolCallEnd` event committed after the state transition.
        committed_events: Vec<CommittedEvent>,
    },
    /// The worker was cancelled before the executor ran. The child
    /// task was **not** driven to a terminal state.
    Cancelled,
}

// ─────────────────────────────────────────────────────────────────────
// Execution
// ─────────────────────────────────────────────────────────────────────

/// Execute a single tool-runtime child task through the journal.
///
/// The `executor` callback receives the resolved [`PendingToolCallInfo`]
/// and produces the [`ToolResult`]. The worker owns the lifecycle:
///
/// 1. Check cancellation (bail early if already cancelled).
/// 2. Run `executor(tool_call)`.
/// 3. On success → [`AgentTaskStore::complete_task_with_result`].
/// 4. On failure → [`AgentTaskStore::fail_task`].
///
/// The callback signature is intentionally minimal — the caller
/// (which has access to the tool registry, hooks, audit sink, etc.)
/// provides whatever execution logic it needs. This keeps
/// `agent-server` independent of `agent-sdk`'s runtime.
///
/// [`AgentTaskStore::complete_task_with_result`]: crate::journal::store::AgentTaskStore::complete_task_with_result
/// [`AgentTaskStore::fail_task`]: crate::journal::store::AgentTaskStore::fail_task
/// # Errors
///
/// Returns an error if the store's `complete_task_with_result` or
/// `fail_task` CAS check fails (e.g. lease expired or wrong worker).
pub async fn execute_tool_task<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    // Box-pin the inner future so the outer function's state
    // machine stays small enough that clippy::large_futures does
    // not fire on every test that drives tool execution.
    execute_on(bootstrap, task_store, cancel, executor, now).await
}

/// [`execute_tool_task`] against any [`DurableStore`]: the settle is its
/// only write.
pub(crate) async fn execute_on<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    store: &dyn DurableStore,
    cancel: &CancellationToken,
    executor: F,
    now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    instrumented(Box::pin(execute_tool_task_inner(
        bootstrap, store, cancel, executor, now,
    )))
    .await
}

/// Record a tool execution's outcome and duration around `run`.
pub(crate) async fn instrumented(
    run: impl Future<Output = anyhow::Result<ToolTaskOutcome>>,
) -> anyhow::Result<ToolTaskOutcome> {
    #[cfg(feature = "otel")]
    let started_at = std::time::Instant::now();

    let result = run.await;

    #[cfg(feature = "otel")]
    {
        let metrics = crate::observability::ServerMetrics::global();
        let outcome = match &result {
            Ok(ToolTaskOutcome::Completed { .. }) => crate::observability::attrs::OUTCOME_DONE,
            Ok(ToolTaskOutcome::Cancelled) => crate::observability::attrs::OUTCOME_CANCELLED,
            Ok(ToolTaskOutcome::Failed { .. }) | Err(_) => {
                crate::observability::attrs::OUTCOME_ERROR
            }
        };
        metrics.record_task_execution(
            crate::observability::attrs::KIND_TOOL,
            outcome,
            started_at.elapsed().as_secs_f64(),
        );
    }

    result
}

/// Inner body of [`execute_tool_task`]. Kept separate so the outer
/// function can wrap the entire call in a metric-recording shim
/// without threading a stopwatch through every early return.
async fn execute_tool_task_inner<F, Fut>(
    bootstrap: ToolTaskBootstrap,
    task_store: &dyn DurableStore,
    cancel: &CancellationToken,
    executor: F,
    // The instant the task was acquired, kept only for signature symmetry;
    // the terminal transition and its events are stamped with the ACTUAL
    // completion instant captured after the tool returns.
    _entry_now: OffsetDateTime,
) -> anyhow::Result<ToolTaskOutcome>
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    // ── Pre-execution cancellation check ─────────────────────────
    if cancel.is_cancelled() {
        return Ok(ToolTaskOutcome::Cancelled);
    }

    let run = run_tool_executor(&bootstrap, executor).await;
    let transition = settle_transition(&bootstrap, &run)?;
    let committed = task_store
        .commit(transition, run.finished_at)
        .await
        .map_err(anyhow::Error::from)
        .with_context(|| run.settle_failure(&bootstrap.task_id))?;
    settled_outcome(run, committed)
}

/// A tool call's executor run, before the transition that settles it.
pub(crate) struct ToolRun {
    pub(crate) result: anyhow::Result<ToolResult>,
    /// Set when the stall signal stopped the tool before it returned.
    pub(crate) stalled: Option<ToolStallReport>,
    progress_events: Vec<AgentEvent>,
    /// When the executor returned.
    pub(crate) finished_at: OffsetDateTime,
}

impl ToolRun {
    fn settle_failure(&self, task_id: &AgentTaskId) -> String {
        match &self.result {
            Ok(_) => format!("complete_task_with_result failed for child {task_id}"),
            Err(_) => format!("fail_task failed for child {task_id}"),
        }
    }
}

/// Run the tool call's executor and collect what it emitted.
pub(crate) async fn run_tool_executor<F, Fut>(bootstrap: &ToolTaskBootstrap, executor: F) -> ToolRun
where
    F: FnOnce(PendingToolCallInfo, ToolEventCollector) -> Fut,
    Fut: Future<Output = anyhow::Result<ToolResult>>,
{
    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global()
        .record_task_acquired(crate::observability::attrs::KIND_TOOL);

    // The collector carries this task's activity beacon: nothing the tool
    // emits reaches the journal until it returns (see the deferred commit
    // in the settle transition), so the beacon is the ONLY way a tool that
    // runs for longer than a stall budget can prove it is alive while it
    // runs.
    let collector = ToolEventCollector::with_activity(bootstrap.activity.clone());
    // A tool that stalls is dropped mid-flight; whatever it did is unknown,
    // which is what its result and its effect record then say.
    let outcome = tokio::select! {
        biased;
        result = executor(bootstrap.tool_call.clone(), collector.clone()) => Ok(result),
        trip = bootstrap.stall.tripped() => Err(trip),
    };

    // The tool may have run for a long time. Drive the terminal transition
    // and its committed events with the ACTUAL completion instant, not the
    // entry time captured before the await — otherwise a long child would
    // stamp a stale `last_activity_at` that a parent's stall probe reads as
    // silence the moment the child finishes. (`AgentTask`'s monotonic setter
    // is the backstop against a rewind; the correct instant belongs here.)
    let finished_at = OffsetDateTime::now_utc();

    // NOTE: No post-execution cancellation check here. Once the
    // executor has returned, side effects have already been applied
    // and the result must be committed to the journal. Dropping it
    // would leave the child in Running state, causing the recovery
    // matrix to re-execute the tool on restart (double execution of
    // non-idempotent operations).

    // Drain collected progress events now, but defer committing until
    // the CAS-guarded settle commits. This prevents orphaned progress
    // events when a stale-lease worker's CAS fails and another worker
    // re-executes the tool.
    let progress_events = collector.drain();
    match outcome {
        Ok(result) => ToolRun {
            result,
            stalled: None,
            progress_events,
            finished_at,
        },
        Err(trip) => ToolRun::stalled(bootstrap, trip, progress_events, finished_at),
    }
}

impl ToolRun {
    /// The run of a tool stopped as stalled, keeping what it reported.
    #[must_use]
    pub(crate) fn stalled(
        bootstrap: &ToolTaskBootstrap,
        trip: StallTrip,
        progress_events: Vec<AgentEvent>,
        finished_at: OffsetDateTime,
    ) -> Self {
        let report = ToolStallReport::new(trip, &progress_events);
        Self {
            result: Err(anyhow::anyhow!(report.message(&bootstrap.tool_call.name))),
            stalled: Some(report),
            progress_events,
            finished_at,
        }
    }
}

/// The settle of a stalled tool call, for the heartbeat to commit when the
/// tool's run has ignored the stall signal for the abort grace.
///
/// It carries no partial output: the progress the tool reported is held by
/// the run.
///
/// # Errors
/// Serialising the stall report fails.
pub fn stalled_settle_transition(
    bootstrap: &ToolTaskBootstrap,
    trip: StallTrip,
    now: OffsetDateTime,
) -> anyhow::Result<Transition> {
    settle_transition(
        bootstrap,
        &ToolRun::stalled(bootstrap, trip, Vec::new(), now),
    )
}

/// The transition that settles the tool child from its run: the child's
/// terminal row, its progress events and its `ToolCallEnd`, in one
/// transaction, so the row and the event that reports it commit together
/// or not at all. The lease guard runs first: a worker that lost its lease
/// writes neither.
pub(crate) fn settle_transition(
    bootstrap: &ToolTaskBootstrap,
    run: &ToolRun,
) -> anyhow::Result<Transition> {
    let (outcome, end_result) = match (&run.result, &run.stalled) {
        (_, Some(report)) => {
            let result = ToolResult {
                success: false,
                output: report.message(&bootstrap.tool_call.name),
                artifact: None,
                data: Some(
                    serde_json::to_value(report).context("serialize the tool stall report")?,
                ),
                documents: Vec::new(),
                duration_ms: None,
            };
            (
                Settle::FailWithReason {
                    error: result.output.clone(),
                    reason: TerminalReason::WatchdogStall,
                    result: Some(
                        serde_json::to_value(&result)
                            .context("serialize the stalled tool result")?,
                    ),
                },
                result,
            )
        }
        (Ok(result), None) => (
            Settle::Complete {
                result: serde_json::to_value(result)
                    .context("serialize tool result for durable storage")?,
            },
            result.clone(),
        ),
        (Err(err), None) => {
            let error_msg = format!("{err:#}");
            (
                Settle::Fail {
                    error: error_msg.clone(),
                },
                ToolResult {
                    success: false,
                    output: error_msg,
                    artifact: None,
                    data: None,
                    documents: Vec::new(),
                    duration_ms: None,
                },
            )
        }
    };
    let mut events = run.progress_events.clone();
    events.push(AgentEvent::tool_call_end(
        &bootstrap.tool_call.id,
        &bootstrap.tool_call.name,
        &bootstrap.tool_call.display_name,
        end_result,
    ));
    Ok(Transition::settle(
        bootstrap.task_id.clone(),
        bootstrap.worker_id.clone(),
        bootstrap.lease_id.clone(),
        outcome,
        bootstrap.thread_id.clone(),
        events,
    ))
}

/// The outcome of a run whose [`settle_transition`] committed.
///
/// No beacon bump here: the settle has already made the row terminal, so a
/// beacon-driven heartbeat write would be CAS-rejected. The completion
/// instant is instead recorded durably on the terminal transition itself
/// (`AgentTask::complete`), which is what a parent's stall probe reads.
pub(crate) fn settled_outcome(
    run: ToolRun,
    committed: Committed,
) -> anyhow::Result<ToolTaskOutcome> {
    let (child, parent) = committed
        .settled
        .into_iter()
        .next()
        .context("a settle transition returns its settled task")?;
    let committed_events = committed.events;
    Ok(match run.result {
        Ok(result) => ToolTaskOutcome::Completed {
            child,
            parent,
            result,
            committed_events,
        },
        Err(err) => ToolTaskOutcome::Failed {
            child,
            parent,
            error: format!("{err:#}"),
            committed_events,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F1 seam: `emit` must mark the tool alive, not merely buffer the
    /// event. A tool that runs longer than a stall budget commits nothing
    /// until it returns (the events are drained and committed afterwards),
    /// so the beacon it bumps on every progress emission is the ONLY sign
    /// of life the parked stall probe can see mid-flight.
    #[test]
    fn emit_marks_the_tool_alive_via_the_beacon() {
        let beacon = ActivityBeacon::new();
        assert!(beacon.latest().is_none(), "beacon starts empty");

        let collector = ToolEventCollector::with_activity(beacon.clone());
        collector.emit(AgentEvent::text("m1", "progress"));

        assert!(
            beacon.latest().is_some(),
            "emit must advance the activity beacon so a long-running tool that only reports \
             progress is not reaped as silent",
        );
        assert_eq!(
            collector.drain().len(),
            1,
            "emit must still buffer the event for the deferred post-CAS commit",
        );
    }

    /// A collector with no wired beacon (the default) still works — it just
    /// reports no activity, exactly as before the beacon existed.
    #[test]
    fn a_default_collector_buffers_without_a_shared_beacon() {
        let collector = ToolEventCollector::new();
        collector.emit(AgentEvent::text("m1", "progress"));
        assert_eq!(collector.drain().len(), 1);
    }
}
