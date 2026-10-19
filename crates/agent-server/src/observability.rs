//! `agent_server.*` metric instruments.
//!
//! Compiled only with `feature = "otel"`. The module owns a small,
//! lazily-initialised [`ServerMetrics`] singleton that the rest of
//! `agent-server` reaches via [`ServerMetrics::global`]. Every
//! histogram declared here pins its own bucket boundaries — we never
//! accept the SDK default boundaries because they are tuned for HTTP
//! timings, not the journal/relay/wakeup ranges this crate cares
//! about.
//!
//! ## Lifecycle
//!
//! Instruments are bound to whichever meter provider is current at
//! the first call to [`ServerMetrics::global`] / [`ServerMetrics::init`].
//! Tests that swap in a fresh `opentelemetry_sdk::metrics::SdkMeterProvider`
//! between cases must call [`ServerMetrics::reset_for_testing`] so
//! the next lookup rebuilds against the new provider.
//!
//! ## Naming
//!
//! Every instrument is namespaced under `agent_server.*` so
//! dashboards and alerts can correlate a metric with the durable
//! `agent-server` codepaths without translation. The `gen_ai.*`
//! namespace is reserved for the LLM client and is never
//! touched here.
//!
//! ## Workers and the host boundary
//!
//! `agent-server` does not own a worker pool — the actual worker
//! lifecycle (spawn, register, shutdown) lives in the
//! `agent-service-host` crate and the host application's shim. The
//! [`ServerMetrics::worker_started`] and [`ServerMetrics::worker_stopped`]
//! helpers are exposed here so those host crates can adjust the
//! shared `agent_server.workers.active` `UpDownCounter` without
//! redeclaring it. The metric itself lives next to its peers because
//! the dashboard story for "active workers vs queued tasks vs
//! lease-expired tasks" requires them all to come from the same
//! meter scope.
//!
//! ## Outbox-depth gauge
//!
//! [`ServerMetrics::register_outbox_depth_callback`] wires a
//! caller-supplied `Fn() -> u64` onto an `ObservableGauge` so each
//! metric export pulls the latest depth. Owners of the closure are
//! responsible for any TTL caching needed to keep durable backends
//! (typically Postgres) from being stampeded by the meter export
//! interval. The in-memory test path uses a trivial closure over a
//! `HashMap` walk; the host wiring passes a Postgres-backed closure
//! with a short TTL.

use std::sync::{Arc, RwLock};

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram, ObservableGauge, UpDownCounter};

use crate::journal::recovery::{FailureReason, RecoveryAction, RecoveryRecord};
use crate::journal::relay::RelayTick;
use crate::journal::tool_audit::ToolAuditEvent;
use crate::journal::tool_audit::ToolEffectClass;

// ─────────────────────────────────────────────────────────────────────
// Bucket boundaries
// ─────────────────────────────────────────────────────────────────────

/// Bucket boundaries for `agent_server.tasks.execution.duration`
/// (seconds). Spans an LLM-call's typical 10 ms .. multi-minute range
/// in the same coarse log-2 cadence the SDK already uses for LLM
/// metrics.
const TASK_DURATION_BUCKETS_S: &[f64] = &[0.01, 0.04, 0.16, 0.64, 2.56, 10.24, 40.96];

/// Bucket boundaries for `agent_server.root_turn.duration` (seconds):
/// log-2 from 10 ms to about 87 minutes, the boundaries of the
/// `bip.turn.duration` histogram this metric replaces.
const ROOT_TURN_DURATION_BUCKETS_S: &[f64] = &[
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92, 163.84,
    327.68, 655.36, 1_310.72, 2_621.44, 5_242.88,
];

/// Bucket boundaries for `agent_server.journal.commit.duration`
/// (seconds). Journal commits are dominated by Postgres round-trips
/// so the boundaries are tighter at the short end.
const COMMIT_DURATION_BUCKETS_S: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

/// Bucket boundaries for `agent_server.tool.phase.duration` (seconds).
/// A store round-trip lands in the first buckets; the tail reaches past the
/// 900-second tool stall budget, where a stalled phase shows up.
const TOOL_PHASE_BUCKETS_S: &[f64] =
    &[0.001, 0.005, 0.025, 0.1, 0.5, 2.5, 10.0, 60.0, 300.0, 900.0];

/// Bucket boundaries for
/// `agent_server.thread_events_watch.lag_ms`. Tuned for AMQP
/// advisory consumer lag; we expect the bulk of healthy lag to land
/// under 100 ms, with stragglers flagged at 2 s and pathological
/// cases at 10 s.
const WATCH_LAG_BUCKETS_MS: &[f64] = &[1.0, 5.0, 25.0, 100.0, 500.0, 2_000.0, 10_000.0];

/// Bucket boundaries for `agent_server.view.folded_events`: the committed
/// events one read-time view fold covered.
const VIEW_FOLDED_BUCKETS: &[f64] = &[16.0, 128.0, 1_024.0, 8_192.0, 65_536.0, 524_288.0];

/// Bucket boundaries for `agent_server.view.fold.duration` (seconds).
const VIEW_FOLD_BUCKETS_S: &[f64] = &[0.001, 0.005, 0.025, 0.1, 0.5, 2.5, 10.0];

// ─────────────────────────────────────────────────────────────────────
// Attribute keys and stable values
// ─────────────────────────────────────────────────────────────────────

/// Stable string keys and values used as metric attributes.
///
/// Centralised here so the dashboard / alert configuration in the
/// `agent-service-host` crate and the host application can reference the same
/// constants instead of re-typing magic strings.
pub mod attrs {
    /// Task kind: `root` / `subagent` / `tool` / `listen`.
    pub const KIND: &str = "kind";
    /// Task outcome: `done` / `error` / `cancelled` / `awaiting_confirmation` / `suspended`.
    pub const OUTCOME: &str = "outcome";
    /// Journal commit step: `atomic` / `non_atomic`.
    pub const EVENT_KIND: &str = "event_kind";
    /// Relay tick outcome: `claimed` / `delivered` / `failed` / `expired`.
    pub const RELAY_OUTCOME: &str = "outcome";
    /// Lease-expiry reason: `requeued` / `<failure_reason>`.
    pub const REASON: &str = "reason";
    /// What a lease reclaim did with a row: `requeued` or
    /// `failed_interrupted` (failed closed).
    pub const ACTION: &str = "action";
    pub const ACTION_REQUEUED: &str = "requeued";
    pub const ACTION_FAILED_INTERRUPTED: &str = "failed_interrupted";
    /// Tool audit event outcome: discriminant of [`ToolAuditEventKind`](crate::journal::tool_audit::ToolAuditEventKind).
    pub const AUDIT_OUTCOME: &str = "outcome";
    /// Tool effect class on audit events.
    pub const EFFECT_CLASS: &str = "effect_class";
    /// A write the purge fence refused: the label of a [`FencedWriter`](crate::journal::thread::FencedWriter).
    pub const WRITER: &str = "writer";
    /// Thread lifecycle status: the label of a [`ThreadStatus`](crate::journal::thread::ThreadStatus).
    pub const THREAD_STATUS: &str = "thread_status";
    /// What happened to an effect (`agent_server.effects`).
    pub const TRANSITION: &str = "transition";
    /// A host's leader-lock state (`agent_server.leader`): `waiting`,
    /// `acquired`, `lost` or `released`.
    pub const LEADER_STATE: &str = "state";
    /// Span-only: the kind of the effect an `effect.run` span runs.
    pub const SPAN_EFFECT_KIND: &str = "agent_server.effect.kind";
    /// Span-only: the state an `effect.run` span's effect ended in.
    pub const SPAN_EFFECT_STATE: &str = "agent_server.effect.state";
    /// Span-only: an effect's key.
    pub const SPAN_EFFECT_ID: &str = "agent_server.effect.id";
    /// Span-only: the owner task of an effect.
    pub const SPAN_TASK_ID: &str = "agent_server.task.id";
    /// Span-only: a task's kind (`subagent`, ...).
    pub const SPAN_TASK_KIND: &str = "agent_server.task.kind";
    /// Span-only: the task's acquisition count, from 1.
    pub const SPAN_TASK_ATTEMPT: &str = "agent_server.task.attempt";
    /// Span-only: the task's retry budget.
    pub const SPAN_TASK_MAX_ATTEMPTS: &str = "agent_server.task.max_attempts";
    /// A subagent invocation's acquisition count: `1`, `2`, `3` or `4+`.
    pub const ATTEMPT: &str = "attempt";
    /// What a prompt step did to a subagent invocation: `parked` / `woken`.
    pub const STEP: &str = "step";
    pub const STEP_PARKED: &str = "parked";
    pub const STEP_WOKEN: &str = "woken";
    /// The tool a stalled call ran (span-only: tool names are unbounded).
    pub const SPAN_TOOL_NAME: &str = "agent_server.tool.name";
    /// A tool call's store-side phase, on `agent_server.tool.phase` spans:
    /// `lookup` / `claim` / `settlement` (bip #980's `bip.tool.phase` values).
    pub const SPAN_TOOL_PHASE: &str = "agent_server.tool.phase";
    /// The same phase as a label of `agent_server.tool.phase.duration`.
    pub const PHASE: &str = "phase";
    /// `outcome` of a tool phase: the boundary completed.
    pub const OUTCOME_SUCCESS: &str = "success";
    /// `outcome` of a tool phase: its future was dropped before it ended.
    pub const OUTCOME_INTERRUPTED: &str = "interrupted";
    /// `outcome` of `agent_server.tool.stalls`: the heartbeat found the tool
    /// silent for its budget and stopped it.
    pub const STALL_TRIPPED: &str = "tripped";
    /// `outcome` of `agent_server.tool.stalls`: the tool's run ignored the
    /// stop, so the heartbeat settled the call.
    pub const STALL_FALLBACK_SETTLED: &str = "fallback_settled";
    /// `kind` of `agent_server.progress.events`: a periodic subagent
    /// progress frame on the parent's thread.
    pub const PROGRESS_SUBAGENT: &str = "subagent_progress";
    /// `kind` of `agent_server.progress.events`: a root turn's stall notice.
    pub const PROGRESS_ROOT_STALL_NOTICE: &str = "root_stall_notice";
    /// `kind` of `agent_server.progress.events`: a root turn's
    /// `UsageProgress`, committed with its attempt's close.
    pub const PROGRESS_USAGE: &str = "usage_progress";
    /// `reason` of `agent_server.progress.refused`: the emitter no longer
    /// holds the lease it writes under.
    pub const PROGRESS_LEASE_LOST: &str = "lease_lost";
    /// `reason` of `agent_server.progress.refused`: the target thread is
    /// being purged or was purged.
    pub const PROGRESS_PURGED: &str = "purged";
    /// `reason` of `agent_server.progress.refused`: the store failed.
    pub const PROGRESS_STORE: &str = "store";
    pub const SPAN_STALL_BUDGET_MS: &str = "agent_server.tool.stall_budget_ms";
    pub const SPAN_STALL_SILENT_FOR_MS: &str = "agent_server.tool.silent_for_ms";
    /// Span-only: the `RunMode` of a subagent completion hook run
    /// (`execute` / `inspect`).
    pub const SPAN_COMPLETION_HOOK_MODE: &str = "agent_server.completion_hook.mode";
    /// Span-only: how a subagent completion hook run ended: the effect's
    /// end state (`completed`, `failed`, `salvaged`, ...), `no_effect`,
    /// `refused` or `error`.
    pub const SPAN_COMPLETION_HOOK_OUTCOME: &str = "agent_server.completion_hook.outcome";
    /// Which leg of the subagent stall budget failed a child root
    /// (`agent_server.subagent.stalls`).
    pub const LEG: &str = "leg";
    /// `leg`: the child root's own heartbeat, while it runs.
    pub const SUBAGENT_STALL_RUNNING: &str = "running";
    /// `leg`: the parked sweep, while the child root waits on its children.
    pub const SUBAGENT_STALL_PARKED: &str = "parked";
    /// Span-only: the subagent invocation a stalled child root answers.
    pub const SPAN_SUBAGENT_INVOCATION_ID: &str = "agent_server.subagent.invocation_id";

    pub const KIND_ROOT: &str = "root";
    pub const KIND_SUBAGENT: &str = "subagent";
    pub const KIND_TOOL: &str = "tool";
    /// Reserved variant. The current journal does not model a
    /// dedicated "listen" task kind separately from
    /// [`crate::journal::TaskKind::ToolRuntime`]. Kept here so a
    /// future split does not need a migration.
    pub const KIND_LISTEN: &str = "listen";

    pub const OUTCOME_DONE: &str = "done";
    pub const OUTCOME_ERROR: &str = "error";
    pub const OUTCOME_CANCELLED: &str = "cancelled";
    pub const OUTCOME_AWAITING_CONFIRMATION: &str = "awaiting_confirmation";
    pub const OUTCOME_SUSPENDED: &str = "suspended";

    pub const RELAY_CLAIMED: &str = "claimed";
    pub const RELAY_DELIVERED: &str = "delivered";
    pub const RELAY_FAILED: &str = "failed";
    pub const RELAY_EXPIRED: &str = "expired";

    pub const REASON_REQUEUED: &str = "requeued";

    /// Store backend of a transition: `memory` / `sqlite` / `postgres`.
    pub const BACKEND: &str = "backend";
    /// Document scope kind: `host` / `thread` / `task`; on lease reclaims,
    /// which leases were taken: `expired` (the sweep) or `foreign_epoch`
    /// (a single host's start-up reclaim).
    pub const SCOPE: &str = "scope";
    /// Document operation as committed: `put`, or `retire` (a retire of a
    /// document that was not live included).
    pub const OP: &str = "op";

    pub const BACKEND_MEMORY: &str = "memory";
    pub const BACKEND_SQLITE: &str = "sqlite";
    pub const BACKEND_POSTGRES: &str = "postgres";

    /// Transition outcome: committed.
    pub const OUTCOME_COMMITTED: &str = "committed";
    /// Transition outcome: a task guard did not hold.
    pub const OUTCOME_GUARD: &str = "guard";
    /// Transition outcome: a write's own expectation (a document's
    /// `DocExpect`) did not hold.
    pub const OUTCOME_EXPECTATION: &str = "expectation";
    /// Transition outcome: malformed, refused, or the store failed.
    pub const OUTCOME_STORE: &str = "store";
    /// A store error that is a lock conflict the store could not wait out
    /// ([`crate::journal::LockConflict`]): Postgres after its retries,
    /// SQLite after its busy timeout.
    pub const OUTCOME_LOCK_CONFLICT: &str = "lock_conflict";

    /// Steering wake outcome: `woken` / `declined` (the parent was no
    /// longer parked).
    pub const OUTCOME_WOKEN: &str = "woken";
    pub const OUTCOME_DECLINED: &str = "declined";
    /// A queued steering wake's read of its queue.
    ///
    /// `taken` (woke the parent), `empty` (nothing queued), `not_parked`
    /// or `no_extension` (refused: the store has no extension to take it).
    pub const OUTCOME_TAKEN: &str = "taken";
    pub const OUTCOME_EMPTY: &str = "empty";
    pub const OUTCOME_NOT_PARKED: &str = "not_parked";
    pub const OUTCOME_NO_EXTENSION: &str = "no_extension";
    /// Where a transition extension ran: `commit` / `completed_turn`.
    pub const EXTENSION_POINT: &str = "extension.point";
    /// The kind of a write a transition extension was refused
    /// ([`crate::journal::Write::kind_name`]): one of the `Write` variants.
    pub const WRITE_KIND: &str = "write.kind";
    pub const EXTENSION_POINT_COMMIT: &str = "commit";
    pub const EXTENSION_POINT_COMPLETED_TURN: &str = "completed_turn";

    pub const OP_PUT: &str = "put";
    pub const OP_RETIRE: &str = "retire";

    /// A view stream's lifecycle step, or a refused view RPC.
    pub const VIEW_STREAM_EVENT: &str = "view.stream.event";
    /// Prompt row owner: `host` / `task`.
    pub const PROMPT_OWNER: &str = "owner";
    /// Prompt row step: [`crate::journal::prompt::PromptStep::as_str`].
    pub const PROMPT_STEP: &str = "step";

    /// Prompt handler outcome: closed the prompt.
    pub const OUTCOME_CLOSED: &str = "closed";
    /// Prompt handler outcome: kept the prompt answered for a later retry.
    pub const OUTCOME_PENDING: &str = "pending";

    pub const COMMIT_KIND_ATOMIC: &str = "atomic";
    pub const COMMIT_KIND_NON_ATOMIC: &str = "non_atomic";
}

/// The ids a tool-stall signal carries on its span (never on a metric).
#[derive(Clone, Copy, Debug)]
pub struct ToolStallSignal<'a> {
    pub tool_name: &'a str,
    pub task_id: &'a str,
    /// The task's stored W3C `traceparent`, which the span nests under.
    pub traceparent: Option<&'a str>,
    pub budget_ms: u64,
    pub silent_for_ms: u64,
}

impl ToolStallSignal<'_> {
    fn emit(&self, event: &'static str, attributes: Vec<KeyValue>) {
        record_on_own_span(
            "agent_server.tool.stall",
            self.traceparent,
            vec![
                KeyValue::new(attrs::SPAN_TOOL_NAME, self.tool_name.to_owned()),
                KeyValue::new(attrs::SPAN_TASK_ID, self.task_id.to_owned()),
            ],
            event,
            attributes,
        );
    }
}

/// The ids a subagent-stall signal carries on its span (never on a metric).
#[derive(Clone, Copy, Debug)]
pub struct SubagentStallSignal<'a> {
    /// The stalled child root.
    pub task_id: &'a str,
    /// The invocation the child root answers, when it was read.
    pub invocation_id: Option<&'a str>,
    /// The W3C `traceparent` the span nests under.
    pub traceparent: Option<&'a str>,
    pub budget_ms: u64,
}

/// Record `event` on a short span named `span_name`. Paths that run under no
/// span of their own (a heartbeat, a sweep) use it; the span is parented on
/// `traceparent` when it decodes. `ids` go on both the span and the event.
fn record_on_own_span(
    span_name: &'static str,
    traceparent: Option<&str>,
    ids: Vec<KeyValue>,
    event: &'static str,
    mut attributes: Vec<KeyValue>,
) {
    use agent_sdk::observability::loop_instrument::context_from_traceparent;
    use opentelemetry::trace::{Span, Tracer};

    let parent = traceparent
        .and_then(context_from_traceparent)
        .unwrap_or_default();
    let mut span = global::tracer("agent-server").start_with_context(span_name, &parent);
    for id in &ids {
        span.set_attribute(id.clone());
    }
    attributes.extend(ids);
    span.add_event(event, attributes);
    span.end();
}

// ─────────────────────────────────────────────────────────────────────
// ServerMetrics
// ─────────────────────────────────────────────────────────────────────

/// Container for every metric instrument the `agent-server` crate
/// records.
///
/// Held behind an `Arc` so call sites that need frequent recording
/// (per task, per tick) can clone the handle into an async future
/// without re-walking a `Mutex`.
pub struct ServerMetrics {
    pub(crate) workers_active: UpDownCounter<i64>,
    pub(crate) tasks_acquired: Counter<u64>,
    pub(crate) tasks_execution_duration: Histogram<f64>,
    pub(crate) tasks_lease_expired: Counter<u64>,
    pub(crate) epoch_reclaims: Counter<u64>,
    pub(crate) journal_commit_duration: Histogram<f64>,
    pub(crate) relay_tick: Counter<u64>,
    pub(crate) wakeup_fallback_sweep: Counter<u64>,
    pub(crate) thread_events_watch_lag_ms: Histogram<f64>,
    pub(crate) tool_audit_outcome: Counter<u64>,
    pub(crate) transition_duration: Histogram<f64>,
    pub(crate) transition_lock_conflict_retries: Counter<u64>,
    pub(crate) documents_written: Counter<u64>,
    /// `agent_server.journal.steering_resumes`: steering wakes committed,
    /// by backend and outcome.
    pub(crate) steering_resumes: Counter<u64>,
    /// `agent_server.journal.extension.{writes,refused}` and
    /// `agent_server.journal.steering_queue.takes`.
    pub(crate) extension: ExtensionInstruments,
    /// `agent_server.effects`: effect transitions, by kind.
    pub(crate) effects: Counter<u64>,
    /// `agent_server.effect.duration`: started to finished, by kind and
    /// the state it finished in.
    pub(crate) effect_duration: Histogram<f64>,
    /// `agent_server.journal.completed_turn.commits`: completed-turn commits
    /// by outcome.
    pub(crate) completed_turn_commits: Counter<u64>,
    /// `agent_server.subagent.invocation.attempts`: subagent invocation
    /// runs, by attempt.
    pub(crate) invocation_attempts: Counter<u64>,
    /// `agent_server.subagent.invocation.failed_closed`: invocations the
    /// lease sweep failed closed at their retry budget, by reason.
    pub(crate) invocation_failed_closed: Counter<u64>,
    pub(crate) view_fold_duration: Histogram<f64>,
    pub(crate) view_folded_events: Histogram<f64>,
    pub(crate) view_stream: Counter<u64>,
    pub(crate) prompts: Counter<u64>,
    /// `agent_server.tool.stalls`: tool calls stopped for silence, by
    /// outcome.
    pub(crate) tool_stalls: Counter<u64>,
    /// `agent_server.tool.stall_budget.unbounded`: tool calls that run
    /// with no stall budget.
    pub(crate) tool_stall_unbounded: Counter<u64>,
    /// `agent_server.tool.phase.duration`: a tool call's store-side phase
    /// (lookup, claim, settlement), by phase and outcome.
    pub(crate) tool_phase_duration: Histogram<f64>,
    /// `agent_server.fence.refused`: writes outside a transition the purge
    /// fence refused, by writer and thread status.
    pub(crate) fence_refused: Counter<u64>,
    /// `agent_server.progress.events`: progress events the worker appended
    /// under a lease, by kind.
    pub(crate) progress_events: Counter<u64>,
    /// `agent_server.progress.refused`: progress appends refused, by reason.
    pub(crate) progress_refused: Counter<u64>,
    /// `agent_server.subagent.completion_hook.refused`: hook outputs the
    /// SDK refused, by `reason`.
    pub(crate) completion_hook_refused: Counter<u64>,
    /// `agent_server.subagent.invocation.prompts`: subagent invocations
    /// parked on a host prompt and woken from one, by backend and step.
    pub(crate) invocation_prompts: Counter<u64>,
    /// `agent_server.subagent.stalls`: subagent child roots failed for
    /// silence, by leg.
    pub(crate) subagent_stalls: Counter<u64>,
    /// `agent_server.leader`: leader-lock transitions, by state.
    pub(crate) leader: Counter<u64>,
    /// `agent_server.leader.is_leader`: hosts that hold their store's
    /// leader lock now. Summed across hosts, more than 1 per store is two
    /// leaders.
    pub(crate) is_leader: UpDownCounter<i64>,
    /// `agent_server.root_turn.duration`: a root turn's admission to its
    /// terminal status, by outcome.
    pub(crate) root_turn_duration: Histogram<f64>,
    /// `agent_server.root_turn.outcome`: root turns that reached a terminal
    /// status, by outcome.
    pub(crate) root_turn_outcome: Counter<u64>,
    /// Shared meter handle reused by `register_outbox_depth_callback`
    /// callers so they bind their gauge to the same provider as
    /// every other instrument on this struct.
    meter: opentelemetry::metrics::Meter,
}

impl std::fmt::Debug for ServerMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerMetrics").finish_non_exhaustive()
    }
}

static METRICS: RwLock<Option<Arc<ServerMetrics>>> = RwLock::new(None);

impl ServerMetrics {
    /// Build instruments under the supplied meter scope and cache
    /// the resulting handle for subsequent [`ServerMetrics::global`]
    /// callers.
    ///
    /// If the cache is already populated the supplied `name` is
    /// ignored and the cached handle is returned — the first caller
    /// in the process wins. Tests that rotate the global meter
    /// provider between cases must call
    /// [`ServerMetrics::reset_for_testing`] beforehand so the next
    /// `init` rebuilds against the fresh provider.
    #[must_use]
    pub fn init(name: &'static str) -> Arc<Self> {
        if let Some(existing) = read_cached() {
            return existing;
        }

        let built = Arc::new(Self::build(name));
        write_cached(Arc::clone(&built));
        built
    }

    /// Convenience wrapper that initialises the singleton under the
    /// `agent-server` meter scope.
    #[must_use]
    pub fn global() -> Arc<Self> {
        Self::init(env!("CARGO_PKG_NAME"))
    }

    /// Drop the cached singleton.
    ///
    /// Test-only escape hatch. Production code never calls this —
    /// the instruments are bound to whichever meter provider was
    /// current when [`init`](Self::init) ran, and rebuilding them
    /// against a different provider mid-run would silently lose
    /// data points.
    pub fn reset_for_testing() {
        let mut guard = match METRICS.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
    }

    /// Increment `agent_server.workers.active`.
    ///
    /// Call this from the host worker loop when a worker registers.
    /// The matching [`worker_stopped`](Self::worker_stopped) keeps
    /// the gauge balanced.
    pub fn worker_started(&self) {
        self.workers_active.add(1, &[]);
    }

    /// Decrement `agent_server.workers.active`.
    ///
    /// Call this when a worker exits its loop (graceful shutdown,
    /// panic-recovery, lease loss, …).
    pub fn worker_stopped(&self) {
        self.workers_active.add(-1, &[]);
    }

    /// Record a single `agent_server.tasks.execution.duration`
    /// sample.
    ///
    /// `kind` and `outcome` are intentionally `&'static str` so the
    /// caller cannot accidentally pass an empty string — every
    /// sample is guaranteed to carry both attributes.
    pub fn record_task_execution(
        &self,
        kind: &'static str,
        outcome: &'static str,
        duration_secs: f64,
    ) {
        self.tasks_execution_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::KIND, kind),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
    }

    /// Record a single `agent_server.tasks.acquired` increment.
    pub fn record_task_acquired(&self, kind: &'static str) {
        self.tasks_acquired
            .add(1, &[KeyValue::new(attrs::KIND, kind)]);
    }

    /// Record one `agent_server.tasks.lease_expired` increment per
    /// row in `records`, attributing each by its sweep outcome.
    ///
    /// `Requeue` rows carry `reason="requeued"`; `FailClosed` rows
    /// carry the failure-reason discriminant
    /// (`error_prefix` from [`FailureReason`]). `NoAction` rows are
    /// not counted because the sweep does not actually touch them.
    pub fn record_lease_expiry_outcomes(&self, records: &[RecoveryRecord]) {
        self.record_lease_reclaim(records, &crate::journal::LeaseRecoveryScope::Expired);
    }

    /// [`Self::record_lease_expiry_outcomes`] for the rows one
    /// `release_expired_leases(.., scope)` call reclaimed, with the
    /// `scope` label (`expired` or `foreign_epoch`).
    pub fn record_lease_reclaim(
        &self,
        records: &[RecoveryRecord],
        scope: &crate::journal::LeaseRecoveryScope,
    ) {
        for record in records {
            let reason = match record.action {
                RecoveryAction::Requeue => attrs::REASON_REQUEUED,
                RecoveryAction::FailClosed(reason) => failure_reason_label(reason),
                RecoveryAction::NoAction => continue,
            };
            self.tasks_lease_expired.add(
                1,
                &[
                    KeyValue::new(attrs::REASON, reason),
                    KeyValue::new(attrs::SCOPE, scope.as_str()),
                ],
            );
            let action = match record.action {
                RecoveryAction::Requeue => attrs::ACTION_REQUEUED,
                RecoveryAction::FailClosed(_) => attrs::ACTION_FAILED_INTERRUPTED,
                RecoveryAction::NoAction => continue,
            };
            self.epoch_reclaims.add(
                1,
                &[
                    KeyValue::new(attrs::SCOPE, scope.as_str()),
                    KeyValue::new(attrs::ACTION, action),
                ],
            );
        }
    }

    /// Count one leader-lock transition (`waiting`, `acquired`, `lost`,
    /// `released`).
    ///
    /// `acquired` raises `agent_server.leader.is_leader` and `released`
    /// lowers it. A host that loses its lock counts as a leader until it
    /// has drained and released, which is the window to watch.
    pub fn record_leader(&self, state: &'static str) {
        self.leader
            .add(1, &[KeyValue::new(attrs::LEADER_STATE, state)]);
        match state {
            "acquired" => self.is_leader.add(1, &[]),
            "released" => self.is_leader.add(-1, &[]),
            _ => {}
        }
    }

    /// Record one `agent_server.fence.refused` increment for `writer`
    /// refused on `conflict`'s thread status. The thread id is not a label
    /// (unbounded cardinality).
    pub fn record_fence_refusal(
        &self,
        writer: crate::journal::thread::FencedWriter,
        conflict: &crate::journal::thread::ThreadStateConflict,
    ) {
        self.fence_refused.add(
            1,
            &[
                KeyValue::new(attrs::WRITER, writer.as_str()),
                KeyValue::new(attrs::THREAD_STATUS, conflict.status.as_str()),
            ],
        );
    }

    /// Record a single `agent_server.journal.commit.duration` sample
    /// for one of the two commit paths.
    pub fn record_journal_commit(&self, event_kind: &'static str, duration_secs: f64) {
        self.journal_commit_duration.record(
            duration_secs,
            &[KeyValue::new(attrs::EVENT_KIND, event_kind)],
        );
    }

    /// Record one [`DurableStore::commit`](crate::journal::DurableStore::commit):
    /// its duration and outcome on `agent_server.journal.transition.duration`,
    /// and each committed document write on
    /// `agent_server.journal.documents.written`. `guard_count` is the
    /// transition's number of guards, which tells a guard rejection from a
    /// write's own failed expectation.
    pub fn record_transition(
        &self,
        backend: &'static str,
        guard_count: usize,
        result: &Result<crate::journal::Committed, crate::journal::Rejected>,
        duration_secs: f64,
    ) {
        let outcome = Self::commit_outcome(guard_count, result);
        self.transition_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::BACKEND, backend),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
        let Ok(committed) = result else {
            return;
        };
        for resumed in &committed.steering_resumed {
            let outcome = if resumed.is_some() {
                attrs::OUTCOME_WOKEN
            } else {
                attrs::OUTCOME_DECLINED
            };
            self.steering_resumes.add(
                1,
                &[
                    KeyValue::new(attrs::BACKEND, backend),
                    KeyValue::new(attrs::OUTCOME, outcome),
                ],
            );
        }
        for invocation in &committed.invocations {
            let step = if invocation.status == crate::journal::TaskStatus::WaitingOnChildren {
                attrs::STEP_PARKED
            } else {
                attrs::STEP_WOKEN
            };
            self.invocation_prompts.add(
                1,
                &[
                    KeyValue::new(attrs::BACKEND, backend),
                    KeyValue::new(attrs::STEP, step),
                ],
            );
        }
        self.record_documents(backend, &committed.documents);
    }

    /// Count what a transition extension did on one commit: the writes it
    /// added, or a refusal of a write it may not add.
    pub fn record_extension(
        &self,
        point: &'static str,
        added: usize,
        refused: Option<&'static str>,
    ) {
        let attributes = [KeyValue::new(attrs::EXTENSION_POINT, point)];
        if let Some(kind) = refused {
            self.extension.refused.add(
                1,
                &[
                    KeyValue::new(attrs::EXTENSION_POINT, point),
                    KeyValue::new(attrs::WRITE_KIND, kind),
                ],
            );
        } else if added > 0 {
            self.extension
                .writes
                .add(u64::try_from(added).unwrap_or(u64::MAX), &attributes);
        }
    }

    /// Count one read of a queued steering wake's queue on
    /// `agent_server.journal.steering_queue.takes`. It is counted inside
    /// the transaction, so a retried or rolled-back transaction counts
    /// again; `agent_server.journal.steering_resumes` counts the commits.
    pub fn record_steering_queue(&self, backend: &'static str, outcome: &'static str) {
        self.extension.steering_queue_takes.add(
            1,
            &[
                KeyValue::new(attrs::BACKEND, backend),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
    }

    /// Count each committed document write on
    /// `agent_server.journal.documents.written`, from a transition or from a
    /// completed turn's extra writes.
    pub fn record_documents(
        &self,
        backend: &'static str,
        documents: &[crate::journal::DocumentOutcome],
    ) {
        for document in documents {
            let op = if document.revision.is_some() {
                attrs::OP_PUT
            } else {
                attrs::OP_RETIRE
            };
            self.documents_written.add(
                1,
                &[
                    KeyValue::new(attrs::BACKEND, backend),
                    KeyValue::new(attrs::SCOPE, document.key.scope.kind_str()),
                    KeyValue::new(attrs::OP, op),
                ],
            );
        }
    }

    /// Count one prompt row step on `agent_server.prompts`. `owner` is
    /// [`attrs::PROMPT_OWNER`]'s value, `step` a
    /// [`crate::journal::prompt::PromptStep`] and `outcome` a transition outcome
    /// (`committed` / `guard` / `expectation` / `store`) or, for
    /// `handle`, `closed` / `pending` / `error`.
    pub fn record_prompt(
        &self,
        owner: &'static str,
        step: crate::journal::prompt::PromptStep,
        outcome: &'static str,
    ) {
        self.prompts.add(
            1,
            &[
                KeyValue::new(attrs::PROMPT_OWNER, owner),
                KeyValue::new(attrs::PROMPT_STEP, step.as_str()),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
    }

    /// The outcome label of a commit result, as
    /// [`Self::record_transition`] attributes it.
    #[must_use]
    pub fn commit_outcome(
        guard_count: usize,
        result: &Result<crate::journal::Committed, crate::journal::Rejected>,
    ) -> &'static str {
        match result {
            Ok(_) => attrs::OUTCOME_COMMITTED,
            Err(crate::journal::Rejected::Guard { index, .. }) if *index >= guard_count => {
                attrs::OUTCOME_EXPECTATION
            }
            Err(crate::journal::Rejected::Guard { .. }) => attrs::OUTCOME_GUARD,
            Err(crate::journal::Rejected::Store(error))
                if error
                    .downcast_ref::<crate::journal::LockConflict>()
                    .is_some() =>
            {
                attrs::OUTCOME_LOCK_CONFLICT
            }
            Err(crate::journal::Rejected::Store(_)) => attrs::OUTCOME_STORE,
        }
    }

    /// Count one transition a store retried after a lock conflict.
    pub fn record_transition_lock_conflict_retry(&self, backend: &'static str) {
        self.transition_lock_conflict_retries
            .add(1, &[KeyValue::new(attrs::BACKEND, backend)]);
    }

    /// Fan out a [`RelayTick`]'s counters onto
    /// `agent_server.relay.tick`.
    ///
    /// Each non-zero counter contributes one `add()` call with the
    /// matching outcome attribute and the counter's value.
    pub fn record_relay_tick(&self, tick: &RelayTick) {
        let entries: [(usize, &'static str); 4] = [
            (tick.claimed, attrs::RELAY_CLAIMED),
            (tick.delivered, attrs::RELAY_DELIVERED),
            (tick.failed, attrs::RELAY_FAILED),
            (tick.expired, attrs::RELAY_EXPIRED),
        ];
        for (count, outcome) in entries {
            if count == 0 {
                continue;
            }
            let attr = [KeyValue::new(attrs::RELAY_OUTCOME, outcome)];
            self.relay_tick.add(count as u64, &attr);
        }
    }

    /// Record a single `agent_server.wakeup.fallback_sweep`
    /// increment. Called once per sweep tick.
    pub fn record_fallback_sweep(&self) {
        self.wakeup_fallback_sweep.add(1, &[]);
    }

    /// Record one `agent_server.thread_events_watch.lag_ms` sample.
    pub fn record_thread_events_watch_lag(&self, lag_ms: f64) {
        self.thread_events_watch_lag_ms.record(lag_ms, &[]);
    }

    /// Record one read-time view fold: `agent_server.view.fold.duration`
    /// and the events it covered, `agent_server.view.folded_events`.
    pub fn record_view_fold(&self, events: usize, duration_secs: f64) {
        self.view_fold_duration.record(duration_secs, &[]);
        self.view_folded_events
            .record(f64::from(u32::try_from(events).unwrap_or(u32::MAX)), &[]);
    }

    /// Record one `agent_server.view.stream` step.
    pub fn record_view_stream(&self, event: &'static str) {
        self.view_stream
            .add(1, &[KeyValue::new(attrs::VIEW_STREAM_EVENT, event)]);
    }

    /// Record one `agent_server.tool_audit.outcome` increment for
    /// the given tool-audit event.
    /// Record one root turn that reached a terminal status on
    /// `agent_server.root_turn.duration` and `agent_server.root_turn.outcome`
    /// (see [`crate::journal::TerminalRootTurn`]). The stores call it after
    /// the transaction commits.
    pub fn record_root_turn(&self, turn: crate::journal::TerminalRootTurn) {
        let labels = [KeyValue::new(attrs::OUTCOME, turn.outcome.as_str())];
        self.root_turn_duration.record(turn.duration_secs, &labels);
        self.root_turn_outcome.add(1, &labels);
    }

    /// Count one completed-turn commit with its bounded outcome (see
    /// [`crate::journal::commit::commit_outcome_label`]).
    pub fn record_completed_turn_commit(&self, outcome: &'static str) {
        self.completed_turn_commits
            .add(1, &[KeyValue::new(attrs::OUTCOME, outcome)]);
    }

    /// Count one effect transition (`started`, `refused`, `finished`,
    /// `reclaimed`, `abandoned`, `continued`) on `agent_server.effects`.
    pub fn record_effect(
        &self,
        kind: crate::journal::effect::EffectKind,
        transition: &'static str,
    ) {
        let attrs = [
            KeyValue::new(attrs::KIND, crate::journal::effect::kind_wire(kind)),
            KeyValue::new(attrs::TRANSITION, transition),
        ];
        self.effects.add(1, &attrs);
    }

    /// Count one run of a subagent invocation on
    /// `agent_server.subagent.invocation.attempts{attempt}`. A run with
    /// `attempt` above 1 is a retry after a crash or an expired lease.
    pub fn record_invocation_attempt(&self, attempt: u32) {
        self.invocation_attempts
            .add(1, &[KeyValue::new(attrs::ATTEMPT, attempt_label(attempt))]);
    }

    /// Count one run of a tool call on `agent_server.tool.attempts{attempt}`
    /// and, for a run above attempt 1 (its task was requeued after a crash
    /// or an expired lease, and a `Safe` tool runs again), emit the span
    /// `agent_server.tool.retry` with the task's id, attempt and budget and
    /// the tool's name. The counter is built on use (the SDK returns the
    /// same instrument), once per tool call.
    pub fn record_tool_attempt(&self, task: &crate::journal::AgentTask, tool_name: &str) {
        use opentelemetry::trace::{Span, Tracer};

        self.meter
            .u64_counter("agent_server.tool.attempts")
            .with_description("Tool call runs, by the task's attempt.")
            .build()
            .add(
                1,
                &[KeyValue::new(attrs::ATTEMPT, attempt_label(task.attempt))],
            );
        if task.attempt <= 1 {
            return;
        }
        let mut span = global::tracer("agent-server").start("agent_server.tool.retry");
        span.set_attribute(KeyValue::new(
            attrs::SPAN_TASK_ID,
            task.id.as_str().to_owned(),
        ));
        span.set_attribute(KeyValue::new(attrs::SPAN_TASK_KIND, attrs::KIND_TOOL));
        span.set_attribute(KeyValue::new(
            attrs::SPAN_TASK_ATTEMPT,
            i64::from(task.attempt),
        ));
        span.set_attribute(KeyValue::new(
            attrs::SPAN_TASK_MAX_ATTEMPTS,
            i64::from(task.max_attempts),
        ));
        span.set_attribute(KeyValue::new(attrs::SPAN_TOOL_NAME, tool_name.to_owned()));
        span.end();
    }

    /// Count a subagent invocation the lease sweep failed closed at its
    /// retry budget, by its failure reason.
    pub fn record_invocation_failed_closed(&self, reason: FailureReason) {
        self.invocation_failed_closed.add(
            1,
            &[KeyValue::new(attrs::REASON, failure_reason_label(reason))],
        );
    }

    /// Record one tool call phase on `agent_server.tool.phase.duration`.
    /// The ids stay on the phase's span; the labels are bounded.
    pub(crate) fn record_tool_phase(
        &self,
        phase: &'static str,
        outcome: &'static str,
        duration_secs: f64,
    ) {
        self.tool_phase_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::PHASE, phase),
                KeyValue::new(attrs::OUTCOME, outcome),
            ],
        );
    }

    /// Count a subagent completion hook output the SDK refused, by `reason`:
    /// `extra_write` (a write other than `Document` or `AppendEvents`),
    /// `park` (unsupported in this release) or `state` (a `Started` finish).
    pub fn record_completion_hook_refused(&self, reason: &'static str) {
        self.completion_hook_refused
            .add(1, &[KeyValue::new(attrs::REASON, reason)]);
    }

    /// Record how long an effect ran, started to finished, with the state it
    /// finished in.
    pub fn record_effect_duration(
        &self,
        kind: crate::journal::effect::EffectKind,
        outcome: &'static str,
        duration_secs: f64,
    ) {
        let attrs = [
            KeyValue::new(attrs::KIND, crate::journal::effect::kind_wire(kind)),
            KeyValue::new(attrs::OUTCOME, outcome),
        ];
        self.effect_duration.record(duration_secs, &attrs);
    }

    /// Count an effect abandoned because its owner became terminal, and add
    /// the `agent_server.effect.abandoned` event to the active span, which
    /// alone carries the ids.
    pub fn record_effect_abandoned(
        &self,
        kind: crate::journal::effect::EffectKind,
        reason: &'static str,
        effect_key: &str,
        task_id: &str,
    ) {
        self.record_effect(kind, "abandoned");
        let event = vec![
            KeyValue::new(attrs::REASON, reason),
            KeyValue::new(
                attrs::SPAN_EFFECT_KIND,
                crate::journal::effect::kind_wire(kind),
            ),
            KeyValue::new(attrs::SPAN_EFFECT_ID, effect_key.to_owned()),
            KeyValue::new(attrs::SPAN_TASK_ID, task_id.to_owned()),
        ];
        opentelemetry::trace::get_active_span(|span| {
            span.add_event("agent_server.effect.abandoned", event);
        });
    }

    /// Count a tool call stopped for silence with `outcome`
    /// ([`attrs::STALL_TRIPPED`] or [`attrs::STALL_FALLBACK_SETTLED`]), and
    /// record the `agent_server.tool.stalled` event, which alone carries the
    /// tool and task, on a short `agent_server.tool.stall` span. The tool
    /// child's heartbeat runs under no span of its own, so the span is
    /// parented on the task's stored `traceparent` when it has one.
    pub fn record_tool_stall(&self, outcome: &'static str, stall: &ToolStallSignal<'_>) {
        self.tool_stalls
            .add(1, &[KeyValue::new(attrs::OUTCOME, outcome)]);
        stall.emit(
            "agent_server.tool.stalled",
            vec![
                KeyValue::new(attrs::OUTCOME, outcome),
                KeyValue::new(
                    attrs::SPAN_STALL_BUDGET_MS,
                    i64::try_from(stall.budget_ms).unwrap_or(i64::MAX),
                ),
                KeyValue::new(
                    attrs::SPAN_STALL_SILENT_FOR_MS,
                    i64::try_from(stall.silent_for_ms).unwrap_or(i64::MAX),
                ),
            ],
        );
    }

    /// Count a progress event appended under a lease, by `kind`
    /// ([`attrs::PROGRESS_SUBAGENT`], [`attrs::PROGRESS_ROOT_STALL_NOTICE`] or
    /// [`attrs::PROGRESS_USAGE`]).
    pub fn record_progress_event(&self, kind: &'static str) {
        self.progress_events
            .add(1, &[KeyValue::new(attrs::KIND, kind)]);
    }

    /// Count a refused progress append, by `reason`
    /// ([`attrs::PROGRESS_LEASE_LOST`], [`attrs::PROGRESS_PURGED`] or
    /// [`attrs::PROGRESS_STORE`]).
    pub fn record_progress_refused(&self, reason: &'static str) {
        self.progress_refused
            .add(1, &[KeyValue::new(attrs::REASON, reason)]);
    }

    /// Count a tool call that starts with no stall budget, and record the
    /// `agent_server.tool.stall_budget_unbounded` event on a short
    /// `agent_server.tool.stall` span.
    pub fn record_tool_stall_unbounded(&self, stall: &ToolStallSignal<'_>) {
        self.tool_stall_unbounded.add(1, &[]);
        stall.emit("agent_server.tool.stall_budget_unbounded", Vec::new());
    }

    /// Count a subagent child root failed for silence on `leg`
    /// ([`attrs::SUBAGENT_STALL_RUNNING`] or [`attrs::SUBAGENT_STALL_PARKED`]),
    /// and record the `agent_server.subagent.stalled` event, which alone
    /// carries the ids, on a short `agent_server.subagent.stall` span.
    pub fn record_subagent_stall(&self, leg: &'static str, stall: &SubagentStallSignal<'_>) {
        self.subagent_stalls
            .add(1, &[KeyValue::new(attrs::LEG, leg)]);
        let mut ids = vec![KeyValue::new(attrs::SPAN_TASK_ID, stall.task_id.to_owned())];
        if let Some(invocation_id) = stall.invocation_id {
            ids.push(KeyValue::new(
                attrs::SPAN_SUBAGENT_INVOCATION_ID,
                invocation_id.to_owned(),
            ));
        }
        record_on_own_span(
            "agent_server.subagent.stall",
            stall.traceparent,
            ids,
            "agent_server.subagent.stalled",
            vec![
                KeyValue::new(attrs::LEG, leg),
                KeyValue::new(
                    attrs::SPAN_STALL_BUDGET_MS,
                    i64::try_from(stall.budget_ms).unwrap_or(i64::MAX),
                ),
            ],
        );
    }

    pub fn record_tool_audit(&self, event: &ToolAuditEvent) {
        let attrs = [
            KeyValue::new(attrs::AUDIT_OUTCOME, event.kind.as_str()),
            KeyValue::new(attrs::EFFECT_CLASS, effect_class_label(event.effect_class)),
        ];
        self.tool_audit_outcome.add(1, &attrs);
    }

    /// Register a callback that publishes
    /// `agent_server.journal.outbox_depth` on every metric export.
    ///
    /// The returned [`ObservableGauge`] is held by the caller for
    /// the lifetime of the registration; dropping it deregisters
    /// the callback. Pass an `Arc<AtomicU64>` (or any cheap shared
    /// state) into the closure so the gauge keeps reading the
    /// caller's latest value.
    ///
    /// Owners are responsible for caching the underlying value
    /// (typically a 5 s TTL) to avoid stampeding durable backends
    /// on every export.
    #[must_use]
    pub fn register_outbox_depth_callback(
        &self,
        provider: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> ObservableGauge<u64> {
        self.meter
            .u64_observable_gauge("agent_server.journal.outbox_depth")
            .with_description("Pending outbox rows awaiting relay.")
            .with_callback(move |observer| observer.observe(provider(), &[]))
            .build()
    }

    fn build(scope: &'static str) -> Self {
        let meter = global::meter(scope);

        let tasks_acquired = meter
            .u64_counter("agent_server.tasks.acquired")
            .with_description("Tasks acquired by a worker for execution.")
            .build();
        let tasks_execution_duration = meter
            .f64_histogram("agent_server.tasks.execution.duration")
            .with_unit("s")
            .with_description("Wall-clock duration of a single task execution.")
            .with_boundaries(TASK_DURATION_BUCKETS_S.to_vec())
            .build();
        let journal_commit_duration = meter
            .f64_histogram("agent_server.journal.commit.duration")
            .with_unit("s")
            .with_description("Duration of an atomic completed-turn commit.")
            .with_boundaries(COMMIT_DURATION_BUCKETS_S.to_vec())
            .build();
        let relay_tick = meter
            .u64_counter("agent_server.relay.tick")
            .with_description("Outbox relay rows observed per tick, attributed by outcome.")
            .build();
        let wakeup_fallback_sweep = meter
            .u64_counter("agent_server.wakeup.fallback_sweep")
            .with_description("Times the fallback wakeup sweep fired a pulse.")
            .build();
        let thread_events_watch_lag_ms = meter
            .f64_histogram("agent_server.thread_events_watch.lag_ms")
            .with_unit("ms")
            .with_description("Time spent dispatching a thread_events_available advisory.")
            .with_boundaries(WATCH_LAG_BUCKETS_MS.to_vec())
            .build();
        let tool_audit_outcome = meter
            .u64_counter("agent_server.tool_audit.outcome")
            .with_description("Tool audit events recorded, attributed by lifecycle outcome.")
            .build();
        let (transition_duration, transition_lock_conflict_retries) =
            transition_instruments(&meter);
        let steering_resumes = steering_resumes_counter(&meter);
        let documents_written = documents_written_counter(&meter);
        let extension = extension_instruments(&meter);
        let prompts = prompts_counter(&meter);

        let (effects, effect_duration, completed_turn_commits) = durability_instruments(&meter);
        let invocation = invocation_instruments(&meter);
        let (tool_stalls, tool_stall_unbounded, subagent_stalls) =
            Self::build_stall_instruments(&meter);
        let tool_phase_duration = meter
            .f64_histogram("agent_server.tool.phase.duration")
            .with_unit("s")
            .with_description(
                "A tool call's store-side phase: lookup (the prior-effect read), claim (the \
                 effect-start commit) or settlement (the settle commit), by phase and outcome \
                 (success, error, interrupted).",
            )
            .with_boundaries(TOOL_PHASE_BUCKETS_S.to_vec())
            .build();
        let fence_refused = fence_refused_counter(&meter);
        let (progress_events, progress_refused) = progress_instruments(&meter);

        let (view_fold_duration, view_folded_events, view_stream) = view_instruments(&meter);
        let (root_turn_duration, root_turn_outcome) = root_turn_instruments(&meter);

        Self {
            workers_active: workers_active_counter(&meter),
            tasks_acquired,
            tasks_execution_duration,
            tasks_lease_expired: lease_expired_counter(&meter),
            epoch_reclaims: epoch_reclaims_counter(&meter),
            journal_commit_duration,
            relay_tick,
            wakeup_fallback_sweep,
            thread_events_watch_lag_ms,
            tool_audit_outcome,
            transition_duration,
            transition_lock_conflict_retries,
            documents_written,
            steering_resumes,
            extension,
            effects,
            effect_duration,
            completed_turn_commits,
            invocation_attempts: invocation.attempts,
            invocation_failed_closed: invocation.failed_closed,
            view_fold_duration,
            view_folded_events,
            view_stream,
            prompts,
            tool_stalls,
            tool_stall_unbounded,
            tool_phase_duration,
            fence_refused,
            progress_events,
            progress_refused,
            completion_hook_refused: invocation.completion_hook_refused,
            invocation_prompts: invocation.prompts,
            subagent_stalls,
            leader: leader_counter(&meter),
            is_leader: is_leader_gauge(&meter),
            root_turn_duration,
            root_turn_outcome,
            meter,
        }
    }

    fn build_stall_instruments(
        meter: &opentelemetry::metrics::Meter,
    ) -> (Counter<u64>, Counter<u64>, Counter<u64>) {
        let tool_stalls = meter
            .u64_counter("agent_server.tool.stalls")
            .with_description(
                "Tool calls stopped because the tool showed no progress for its stall budget, \
                 by outcome: tripped (the heartbeat stopped it), fallback_settled (the tool \
                 ignored the stop and the heartbeat settled the call).",
            )
            .build();
        let tool_stall_unbounded = meter
            .u64_counter("agent_server.tool.stall_budget.unbounded")
            .with_description("Tool calls started with no stall budget.")
            .build();
        let subagent_stalls = meter
            .u64_counter("agent_server.subagent.stalls")
            .with_description(
                "Subagent child roots failed because their subtree showed no progress for the \
                 stall budget, by leg: running (the child root's heartbeat), parked (the sweep, \
                 while the child root waits on its children). A fail the still-owned guard \
                 skipped because another actor took the row also counts.",
            )
            .build();
        (tool_stalls, tool_stall_unbounded, subagent_stalls)
    }
}

/// `agent_server.tasks.lease_expired`.
fn lease_expired_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.tasks.lease_expired")
        .with_description(
            "Leases the expiry sweep or a start-up epoch reclaim took back, by outcome and scope.",
        )
        .build()
}

/// `agent_server.epoch.reclaims`.
fn epoch_reclaims_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.epoch.reclaims")
        .with_description(
            "Leases a lease sweep or a start-up epoch reclaim took back, attributed by \
             scope and what recovery did with the row.",
        )
        .build()
}

/// `agent_server.journal.transition.duration` and
/// `agent_server.journal.transition.lock_conflict_retries`.
fn transition_instruments(meter: &opentelemetry::metrics::Meter) -> (Histogram<f64>, Counter<u64>) {
    let duration = meter
        .f64_histogram("agent_server.journal.transition.duration")
        .with_unit("s")
        .with_description("Duration of DurableStore::commit, attributed by backend and outcome.")
        .with_boundaries(COMMIT_DURATION_BUCKETS_S.to_vec())
        .build();
    let lock_conflict_retries = meter
        .u64_counter("agent_server.journal.transition.lock_conflict_retries")
        .with_description(
            "Transitions the store retried after a deadlock or serialization failure.",
        )
        .build();
    (duration, lock_conflict_retries)
}

/// `agent_server.root_turn.duration` and `agent_server.root_turn.outcome`.
fn root_turn_instruments(meter: &opentelemetry::metrics::Meter) -> (Histogram<f64>, Counter<u64>) {
    let duration = meter
        .f64_histogram("agent_server.root_turn.duration")
        .with_unit("s")
        .with_description(
            "A root turn's admission to its terminal status, by outcome (success, error, \
             cancelled).",
        )
        .with_boundaries(ROOT_TURN_DURATION_BUCKETS_S.to_vec())
        .build();
    let outcome = meter
        .u64_counter("agent_server.root_turn.outcome")
        .with_description(
            "Root turns that reached a terminal status, by outcome (success, error, cancelled).",
        )
        .build();
    (duration, outcome)
}

/// `agent_server.progress.events` and `agent_server.progress.refused`.
fn progress_instruments(meter: &opentelemetry::metrics::Meter) -> (Counter<u64>, Counter<u64>) {
    let events = meter
        .u64_counter("agent_server.progress.events")
        .with_description(
            "Progress events the worker appended under the lease of the task it reports on, \
             by kind: subagent_progress or root_stall_notice.",
        )
        .build();
    let refused = meter
        .u64_counter("agent_server.progress.refused")
        .with_description(
            "Progress appends refused, by reason: lease_lost (the emitter no longer holds its \
             lease), purged (the thread is purged) or store (the store failed).",
        )
        .build();
    (events, refused)
}

/// `agent_server.workers.active`.
fn workers_active_counter(meter: &opentelemetry::metrics::Meter) -> UpDownCounter<i64> {
    meter
        .i64_up_down_counter("agent_server.workers.active")
        .with_description("Active worker processes ready to acquire tasks.")
        .build()
}

/// `agent_server.leader.is_leader`.
fn is_leader_gauge(meter: &opentelemetry::metrics::Meter) -> UpDownCounter<i64> {
    meter
        .i64_up_down_counter("agent_server.leader.is_leader")
        .with_description("Hosts that hold their store's leader lock now.")
        .build()
}

/// `agent_server.leader`.
fn leader_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.leader")
        .with_description(
            "Leader-lock transitions of a host under EpochRecovery::PostgresLeader: \
             waiting, acquired, lost, released.",
        )
        .build()
}

/// `agent_server.fence.refused`.
fn fence_refused_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.fence.refused")
        .with_description(
            "Writes outside a transition the purge fence refused, attributed by writer and \
             thread status.",
        )
        .build()
}

/// The subagent invocation instruments: runs by attempt, invocations the
/// lease sweep failed closed at their budget, and completion hook outputs
/// the SDK refused.
fn invocation_instruments(meter: &opentelemetry::metrics::Meter) -> InvocationInstruments {
    let attempts = meter
        .u64_counter("agent_server.subagent.invocation.attempts")
        .with_description(
            "Subagent invocation runs by attempt (1, 2, 3, 4+). A run above attempt 1 is a \
             retry after a crash or an expired lease.",
        )
        .build();
    let failed_closed = meter
        .u64_counter("agent_server.subagent.invocation.failed_closed")
        .with_description(
            "Subagent invocations the lease sweep failed closed at their retry budget, by reason.",
        )
        .build();
    let completion_hook_refused = meter
        .u64_counter("agent_server.subagent.completion_hook.refused")
        .with_description(
            "Subagent completion hook outputs refused before the settle, by reason: \
             extra_write, park or state. The invocation then fails closed.",
        )
        .build();
    let prompts = meter
        .u64_counter("agent_server.subagent.invocation.prompts")
        .with_description(
            "Subagent invocations parked on a host prompt (step=parked) and woken from one \
             (step=woken), by backend.",
        )
        .build();
    InvocationInstruments {
        attempts,
        failed_closed,
        completion_hook_refused,
        prompts,
    }
}

/// The counters [`invocation_instruments`] builds.
struct InvocationInstruments {
    attempts: Counter<u64>,
    failed_closed: Counter<u64>,
    completion_hook_refused: Counter<u64>,
    prompts: Counter<u64>,
}

/// The crash-safety instruments: effect transitions, effect run time and
/// completed-turn commit outcomes.
fn durability_instruments(
    meter: &opentelemetry::metrics::Meter,
) -> (Counter<u64>, Histogram<f64>, Counter<u64>) {
    let completed_turn_commits = meter
        .u64_counter("agent_server.journal.completed_turn.commits")
        .with_description(
            "Completed-turn commits by outcome: ok, lost_ownership (the worker no longer \
             owned the task, so the finished turn was discarded), stale_turn (another \
             commit took the turn slot) or error.",
        )
        .build();
    let effects = meter
        .u64_counter("agent_server.effects")
        .with_description(
            "Effect transitions by kind: started (its record committed), refused (the \
             claim did not commit), continued (a requeued owner continues its started \
             record), finished, reclaimed (recovery ended it without \
             running it again), abandoned (its owner became terminal first), salvaged (a cut-off LLM response committed with the turn that continues).",
        )
        .build();
    let effect_duration = meter
        .f64_histogram("agent_server.effect.duration")
        .with_unit("s")
        .with_description("Effect run time, started to finished, by kind and outcome.")
        .build();
    (effects, effect_duration, completed_turn_commits)
}

fn view_instruments(
    meter: &opentelemetry::metrics::Meter,
) -> (Histogram<f64>, Histogram<f64>, Counter<u64>) {
    let fold_duration = meter
        .f64_histogram("agent_server.view.fold.duration")
        .with_unit("s")
        .with_description("Duration of one read-time fold of a thread's events into its view.")
        .with_boundaries(VIEW_FOLD_BUCKETS_S.to_vec())
        .build();
    let folded_events = meter
        .f64_histogram("agent_server.view.folded_events")
        .with_description("Committed events one read-time view fold covered.")
        .with_boundaries(VIEW_FOLDED_BUCKETS.to_vec())
        .build();
    let stream = meter
        .u64_counter("agent_server.view.stream")
        .with_description("Session view stream steps and refused view RPCs.")
        .build();
    (fold_duration, folded_events, stream)
}

/// `agent_server.journal.documents.written`: document writes committed, by
/// backend, scope and operation.
fn documents_written_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.journal.documents.written")
        .with_description("Document writes committed, attributed by backend, scope and operation.")
        .build()
}

/// `agent_server.journal.steering_resumes`: steering wakes committed, by
/// backend and outcome.
fn steering_resumes_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.journal.steering_resumes")
        .with_description(
            "Steering wakes committed, attributed by backend and outcome (woken or declined).",
        )
        .build()
}

/// The transition extension's counters: the writes it added, the commits
/// refused for a write it may not add, and the reads of a queued steering
/// wake's queue.
fn extension_instruments(meter: &opentelemetry::metrics::Meter) -> ExtensionInstruments {
    let writes = meter
        .u64_counter("agent_server.journal.extension.writes")
        .with_description("Writes a transition extension added, attributed by extension point.")
        .build();
    let refused = meter
        .u64_counter("agent_server.journal.extension.refused")
        .with_description(
            "Commits refused because the transition extension returned a write it may not add, \
             attributed by extension point and write kind.",
        )
        .build();
    let steering_queue_takes = meter
        .u64_counter("agent_server.journal.steering_queue.takes")
        .with_description(
            "Reads of a queued steering wake's queue inside its transaction, attributed by \
             backend and outcome (taken, empty, not_parked, no_extension).",
        )
        .build();
    ExtensionInstruments {
        writes,
        refused,
        steering_queue_takes,
    }
}

/// The transition extension's counters.
#[derive(Clone, Debug)]
pub(crate) struct ExtensionInstruments {
    /// `agent_server.journal.extension.writes`: the writes it added.
    pub(crate) writes: Counter<u64>,
    /// `agent_server.journal.extension.refused`: commits refused for a
    /// write it may not add.
    pub(crate) refused: Counter<u64>,
    /// `agent_server.journal.steering_queue.takes`: reads of a queued
    /// steering wake's queue, by backend and outcome.
    pub(crate) steering_queue_takes: Counter<u64>,
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────

fn prompts_counter(meter: &opentelemetry::metrics::Meter) -> Counter<u64> {
    meter
        .u64_counter("agent_server.prompts")
        .with_description(
            "Prompt row steps (answer, close, withdraw, handle, redrive), attributed by owner \
             and outcome.",
        )
        .build()
}

fn read_cached() -> Option<Arc<ServerMetrics>> {
    let guard = match METRICS.read() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.as_ref().map(Arc::clone)
}

fn write_cached(value: Arc<ServerMetrics>) {
    let mut guard = match METRICS.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(value);
    }
    // If a concurrent caller raced us in, drop the freshly-built
    // handle on the floor. Both copies are functionally equivalent
    // (same scope, same global meter), so the duplicate is
    // harmless.
}

/// The bounded `attempt` label: `1`, `2`, `3` or `4+`.
const fn attempt_label(attempt: u32) -> &'static str {
    match attempt {
        0 | 1 => "1",
        2 => "2",
        3 => "3",
        _ => "4+",
    }
}

const fn failure_reason_label(reason: FailureReason) -> &'static str {
    reason.error_prefix()
}

const fn effect_class_label(class: ToolEffectClass) -> &'static str {
    match class {
        ToolEffectClass::ReplaySafe => "replay_safe",
        ToolEffectClass::SideEffecting => "side_effecting",
        ToolEffectClass::Resumable => "resumable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_reason_label_matches_error_prefix() {
        // The label IS the error prefix — keep them coupled so a
        // dashboard query like `reason="retry_budget_exhausted"`
        // exactly matches the wire format written by the journal.
        for r in [
            FailureReason::RetryBudgetExhausted,
            FailureReason::LeaseExpiredBudgetExhausted,
            FailureReason::UnsafePreparedOperationRecovery,
        ] {
            assert_eq!(failure_reason_label(r), r.error_prefix());
        }
    }

    #[test]
    fn effect_class_label_is_snake_case_serde() {
        // The label is the same snake_case discriminant the
        // `ToolEffectClass` serde representation uses, keeping the
        // metric attribute aligned with the audit-row JSON value.
        for (class, expected) in [
            (ToolEffectClass::ReplaySafe, "replay_safe"),
            (ToolEffectClass::SideEffecting, "side_effecting"),
            (ToolEffectClass::Resumable, "resumable"),
        ] {
            assert_eq!(effect_class_label(class), expected);
        }
    }
}
