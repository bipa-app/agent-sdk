//! The tool-child leg of the stall budget: a tool call ends when its tool
//! shows no evidence of work for its budget, so a parent waiting on it
//! cannot wait forever (bip #756).
//!
//! The tool child's own heartbeat enforces it, because that loop holds the
//! row's lease and reads the live [`ActivityBeacon`](agent_server::worker::ActivityBeacon).
//! It trips the call's [`StallSignal`]; the tool's run then drops the tool
//! and settles the call as stalled with what the tool had reported. If the
//! run ignores the signal for [`EXECUTION_ABORT_GRACE`], the heartbeat
//! settles the call itself. If the heartbeat stops, the lease expires and
//! lease recovery applies; while it beats, it enforces.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use agent_server::journal::Rejected;
use agent_server::worker::tool_task::stalled_settle_transition;
use agent_server::worker::{StallBudget, StallSignal, StallTrip, ToolTaskBootstrap};
use time::OffsetDateTime;
use tracing::warn;

use super::{EXECUTION_ABORT_GRACE, publish_events, settle_tool_child};
use crate::runtime::ExecutionRuntime;
use crate::stores::StoreRegistry;

/// The stall budget of one tool child's acquisition, shared by its
/// heartbeat and its run.
#[derive(Clone, Debug, Default)]
pub struct ToolStallWatch {
    default_budget: Option<Duration>,
    signal: StallSignal,
    armed: Arc<OnceLock<Armed>>,
}

#[derive(Clone)]
struct Armed {
    budget: Duration,
    bootstrap: ToolTaskBootstrap,
    since: OffsetDateTime,
    runtime: Arc<ExecutionRuntime>,
}

impl std::fmt::Debug for Armed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Armed")
            .field("budget", &self.budget)
            .field("task_id", &self.bootstrap.task_id)
            .field("since", &self.since)
            .finish_non_exhaustive()
    }
}

impl ToolStallWatch {
    pub fn new(default_budget: Option<Duration>) -> Self {
        Self {
            default_budget,
            ..Self::default()
        }
    }

    /// Start enforcing the budget `declared` resolves to on the call
    /// `bootstrap` is about to run, and hand that call the signal that
    /// stops it.
    pub fn arm(
        &self,
        bootstrap: &mut ToolTaskBootstrap,
        declared: StallBudget,
        runtime: &Arc<ExecutionRuntime>,
        now: OffsetDateTime,
    ) {
        bootstrap.stall = self.signal.clone();
        let Some(budget) = declared.resolve(self.default_budget) else {
            #[cfg(feature = "otel")]
            agent_server::observability::ServerMetrics::global()
                .record_tool_stall_unbounded(&stall_signal(bootstrap, 0, 0));
            return;
        };
        // One watch per acquisition, armed once by the run that owns it.
        let first = self.armed.set(Armed {
            budget,
            bootstrap: bootstrap.clone(),
            since: now,
            runtime: Arc::clone(runtime),
        });
        debug_assert!(first.is_ok(), "a tool stall watch is armed once");
    }
}

/// What the heartbeat does after one tick of the tool stall leg.
pub enum ToolStallTick {
    Continue,
    /// The call is settled: trip the per-task token and stop beating.
    Stop,
}

/// The heartbeat's side of a [`ToolStallWatch`].
pub struct ToolStallLeg {
    watch: ToolStallWatch,
    tripped_at: Option<Instant>,
}

impl ToolStallLeg {
    pub const fn new(watch: ToolStallWatch) -> Self {
        Self {
            watch,
            tripped_at: None,
        }
    }

    /// One tick, after the lease was renewed. `observed` is the beacon read
    /// for this tick; `min_budget` the floor the persistence cadence sets.
    pub async fn tick(
        &mut self,
        stores: &StoreRegistry,
        observed: Option<OffsetDateTime>,
        min_budget: Duration,
        now: OffsetDateTime,
    ) -> ToolStallTick {
        let Some(armed) = self.watch.armed.get() else {
            return ToolStallTick::Continue;
        };
        match (self.watch.signal.tripped_with(), self.tripped_at) {
            (Some(trip), Some(tripped_at)) if tripped_at.elapsed() >= EXECUTION_ABORT_GRACE => {
                settle_ignored_stall(stores, armed, trip, now).await
            }
            (Some(_), _) => ToolStallTick::Continue,
            (None, _) => {
                let budget = armed.budget.max(min_budget);
                let last_progress_at = observed.filter(|at| *at > armed.since);
                let evidence = last_progress_at.unwrap_or(armed.since);
                let silent_for = Duration::try_from(now - evidence).unwrap_or_default();
                if silent_for < budget {
                    return ToolStallTick::Continue;
                }
                let trip = StallTrip {
                    budget,
                    silent_for,
                    last_progress_at,
                };
                warn!(
                    task_id = %armed.bootstrap.task_id,
                    thread_id = %armed.bootstrap.thread_id,
                    tool_name = %armed.bootstrap.tool_call.name,
                    budget_secs = budget.as_secs(),
                    silent_for_secs = silent_for.as_secs(),
                    "tool showed no progress for its stall budget; stopping the call",
                );
                record(&armed.bootstrap, STALL_TRIPPED, trip);
                self.watch.signal.trip(trip);
                self.tripped_at = Some(Instant::now());
                ToolStallTick::Continue
            }
        }
    }
}

/// Values of `agent_server.tool.stalls`'s `outcome` attribute.
const STALL_TRIPPED: &str = "tripped";
const STALL_FALLBACK_SETTLED: &str = "fallback_settled";

fn record(bootstrap: &ToolTaskBootstrap, outcome: &'static str, trip: StallTrip) {
    #[cfg(feature = "otel")]
    agent_server::observability::ServerMetrics::global().record_tool_stall(
        outcome,
        &stall_signal(
            bootstrap,
            u64::try_from(trip.budget.as_millis()).unwrap_or(u64::MAX),
            u64::try_from(trip.silent_for.as_millis()).unwrap_or(u64::MAX),
        ),
    );
    #[cfg(not(feature = "otel"))]
    let _ = (bootstrap, outcome, trip);
}

#[cfg(feature = "otel")]
fn stall_signal(
    bootstrap: &ToolTaskBootstrap,
    budget_ms: u64,
    silent_for_ms: u64,
) -> agent_server::observability::ToolStallSignal<'_> {
    agent_server::observability::ToolStallSignal {
        tool_name: &bootstrap.tool_call.name,
        task_id: bootstrap.task_id.as_str(),
        traceparent: bootstrap.child_task.otel_traceparent.as_deref(),
        budget_ms,
        silent_for_ms,
    }
}

/// Settle a stalled call whose run ignored the stop. The guard on this
/// lease decides a race with a run that settles at the same moment: one
/// settle commits, the other is refused.
async fn settle_ignored_stall(
    stores: &StoreRegistry,
    armed: &Armed,
    trip: StallTrip,
    now: OffsetDateTime,
) -> ToolStallTick {
    let bootstrap = &armed.bootstrap;
    let transition = match stalled_settle_transition(bootstrap, trip, now) {
        Ok(transition) => transition,
        Err(error) => {
            warn!(task_id = %bootstrap.task_id, error = %error, "build the stalled tool settle");
            return ToolStallTick::Continue;
        }
    };
    match stores.task_store.commit(transition, now).await {
        Ok(committed) => {
            warn!(
                task_id = %bootstrap.task_id,
                tool_name = %bootstrap.tool_call.name,
                grace_secs = EXECUTION_ABORT_GRACE.as_secs(),
                "stalled tool ignored the stop past the abort grace; settled its call from the \
                 heartbeat",
            );
            record(bootstrap, STALL_FALLBACK_SETTLED, trip);
            publish_events(stores, &committed.events);
            settle_tool_child(
                armed.runtime.wakeup_signal(),
                stores.task_store.as_ref(),
                bootstrap.child_task.parent_id.as_ref(),
            )
            .await;
            ToolStallTick::Stop
        }
        // The row is no longer this lease's: the run settled it, or another
        // actor ended it. Either way the call is over.
        Err(Rejected::Guard { .. }) => ToolStallTick::Stop,
        Err(Rejected::Store(error)) => {
            warn!(
                task_id = %bootstrap.task_id,
                error = %error,
                "stalled tool settle hit a store error; retrying next tick",
            );
            ToolStallTick::Continue
        }
    }
}
