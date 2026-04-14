//! Task-wakeup consumer contract.
//!
//! Phase 8.3 (ENG-7967) builds on top of Phase 8.1's
//! [`crate::journal::outbox_message::TaskWakeupPayload`] and Phase 8.2's
//! AMQP relay to close the "broker as nudge, journal as authority" loop.
//! The broker path is a **latency optimisation**, never a source of
//! truth.  This module defines the small surface every consumer — AMQP,
//! in-process fallback sweep, or a bespoke test double — must share so
//! the worker pool can be nudged without ever letting a queue payload
//! drive execution.
//!
//! # The contract in one picture
//!
//! ```text
//!   ┌─────────── broker delivery ──────────┐
//!   │ { task_id: ..., thread_id: ... }    │  (advisory only)
//!   └──────────────────┬──────────────────┘
//!                      │ handle_payload()
//!                      ▼
//!             ┌────────────────────┐
//!             │ AgentTaskStore.get │   ← authoritative re-check
//!             └────────┬───────────┘
//!                      │
//!                      ▼
//!       ┌──────────────┴──────────────┐
//!       │                             │
//!       ▼                             ▼
//!   Pending / Queued /           Terminal / Missing:
//!   Waiting pair?                 noop + observe
//!       │
//!       ▼
//!   WakeupSignal::notify_workers()   ← nudge; workers CAS on acquire
//! ```
//!
//! The nudge itself is just a [`tokio::sync::Notify`] — it buffers one
//! permit and wakes at most one idle worker.  Every worker that receives
//! the nudge runs
//! [`AgentTaskStore::acquire_next_runnable`],
//! which is the single place the
//! `Pending → Running` CAS lives.  Two consumers that deliver the same
//! wakeup therefore resolve into **exactly one execution**: whichever
//! worker wins the CAS, plus any number of losers that observe
//! `Ok(None)` and go back to idle.
//!
//! # Duplicate safety
//!
//! There are three independent reasons a duplicate wakeup cannot
//! produce a duplicate execution:
//!
//! 1. The consumer never executes the task; it only signals.
//! 2. The signal is a permit-style notify — N duplicate calls resolve
//!    to a single worker awakening (any extra permits fold into the
//!    same one).
//! 3. The `Pending → Running` CAS in
//!    [`AgentTaskStore::acquire_next_runnable`]
//!    is serialised by the store write lock — two workers racing on
//!    the same row observe exactly one `Some(task)` winner.
//!
//! These three properties are independent so the system tolerates
//! broker, consumer, or worker misbehaviour without ever double-running
//! work.
//!
//! # Fallback sweeps
//!
//! The worker pool's acquisition ticker (see
//! `agent_service_host::config::WorkerConfig::acquisition_interval`)
//! runs every `acquisition_interval_secs` and calls
//! [`AgentTaskStore::acquire_next_runnable`]
//! unconditionally.  If the broker is unreachable or a wakeup is lost
//! in-flight, the ticker still fires and the work still moves.
//! `WakeupSignal` is therefore purely a latency optimisation — every
//! correctness property is already enforced by the journal.
//!
//! A [`FallbackWakeupSweep`] is provided for deployments that want an
//! extra periodic signal that is independent of both the consumer and
//! the per-worker ticker.  It exists so tests (and operators who want
//! belt-and-suspenders) can prove that a silent broker still makes
//! progress.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::outbox_message::TaskWakeupPayload;
use super::store::AgentTaskStore;
use super::task::{AgentTask, TaskStatus};

// ─────────────────────────────────────────────────────────────────────
// Signal
// ─────────────────────────────────────────────────────────────────────

/// Lock-free "wake up, there is work" nudge between consumers and
/// workers.
///
/// Backed by a [`tokio::sync::Notify`] so:
///
/// - `notify_workers` is cheap and non-blocking.
/// - A call that precedes the first `wait_for_nudge().await` is
///   buffered as one permit (never lost if a worker is about to park).
/// - Multiple `notify_workers` calls collapse to at most one extra
///   wake-up per worker, which matches the "broker is a nudge" model:
///   duplicates fold naturally, the journal is what decides whether
///   there is actually work.
#[derive(Debug)]
pub struct WakeupSignal {
    notify: Notify,
}

impl Default for WakeupSignal {
    fn default() -> Self {
        Self {
            notify: Notify::new(),
        }
    }
}

impl WakeupSignal {
    /// Allocate a fresh signal wrapped in an [`Arc`] so producers and
    /// consumers can share it.
    #[must_use]
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Notify one waiting worker that there may be work to claim.
    ///
    /// If no worker is currently parked the notify is stored as a
    /// single permit; whichever worker parks next consumes it.  Any
    /// additional calls before the permit is consumed are folded into
    /// the same permit.
    pub fn notify_workers(&self) {
        self.notify.notify_one();
    }

    /// Wake every currently-parked worker at once.
    ///
    /// This does **not** buffer: calls that precede the first waiter
    /// are lost.  Used by [`FallbackWakeupSweep`] to fan out a
    /// time-based pulse and by shutdown paths that want to flush
    /// every worker out of the park in one go.
    pub fn wake_all_now(&self) {
        self.notify.notify_waiters();
    }

    /// Park until someone calls `notify_workers` (or a buffered permit
    /// is available).
    ///
    /// Callers are expected to race this future with their own
    /// periodic ticker so a lost nudge can never stall progress; see
    /// the service-host worker loop for the canonical pattern.
    pub async fn wait_for_nudge(&self) {
        self.notify.notified().await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// What a re-check of the durable journal found for an advisory wakeup
/// payload.
///
/// Returned from [`TaskWakeupHandler::handle_payload`] so callers can
/// make observability decisions (logging, metrics, dead-letter
/// handling) without taking a second round trip to the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskWakeupOutcome {
    /// The referenced task is [`TaskStatus::Pending`] or
    /// [`TaskStatus::Queued`] — the nudge was forwarded to the worker
    /// pool.
    Nudged {
        /// Canonical status as persisted in the journal at re-check
        /// time.
        status: TaskStatus,
    },
    /// The referenced task exists but is not in a state where a worker
    /// would acquire it (waiting, running, terminal).  No nudge was
    /// sent; duplicates of this outcome are expected and benign.
    NotRunnable {
        /// Canonical status as persisted in the journal at re-check
        /// time.
        status: TaskStatus,
    },
    /// The referenced task is not in the store.  This can happen when
    /// the broker republishes a wakeup for a row that has since been
    /// cancelled, or when retention has already removed the task.
    /// Callers should treat this as a benign duplicate of an earlier
    /// wakeup and ack the broker message.
    Missing,
}

impl TaskWakeupOutcome {
    /// `true` when the handler actually forwarded a nudge to the
    /// worker pool.  Useful for tests.
    #[must_use]
    pub const fn nudged(&self) -> bool {
        matches!(self, Self::Nudged { .. })
    }

    /// Canonical status at re-check time, if any.
    #[must_use]
    pub const fn observed_status(&self) -> Option<TaskStatus> {
        match self {
            Self::Nudged { status } | Self::NotRunnable { status } => Some(*status),
            Self::Missing => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Handler contract
// ─────────────────────────────────────────────────────────────────────

/// Narrow boundary every wakeup transport implements.
///
/// A consumer is responsible for pulling advisory payloads from
/// somewhere (AMQP queue, in-memory channel, periodic sweep) and
/// handing them to the handler.  The handler does the journal re-check
/// and produces a [`TaskWakeupOutcome`] the caller can log / ack / nack.
#[async_trait]
pub trait TaskWakeupHandler: Send + Sync {
    /// Re-check durable state for the referenced task and optionally
    /// nudge the worker pool.
    ///
    /// # Errors
    /// Returns an error only when the durable store itself cannot be
    /// queried.  Missing or non-runnable tasks are returned as
    /// [`TaskWakeupOutcome::Missing`] / [`TaskWakeupOutcome::NotRunnable`]
    /// — **not** errors — so the consumer can ack the broker message
    /// with a clean conscience.
    async fn handle_payload(
        &self,
        payload: &TaskWakeupPayload,
        now: OffsetDateTime,
    ) -> Result<TaskWakeupOutcome>;
}

// ─────────────────────────────────────────────────────────────────────
// Default journal-backed handler
// ─────────────────────────────────────────────────────────────────────

/// Default [`TaskWakeupHandler`] that looks the task up in an
/// [`AgentTaskStore`] and nudges a shared [`WakeupSignal`].
///
/// Carries no state beyond the two `Arc`s, so it is cheap to clone and
/// can be spawned inside multiple consumers (AMQP + fallback sweep +
/// anything else) without contention.
#[derive(Clone)]
pub struct JournalTaskWakeupHandler {
    store: Arc<dyn AgentTaskStore>,
    signal: Arc<WakeupSignal>,
}

impl JournalTaskWakeupHandler {
    /// Construct a handler that looks tasks up in `store` and nudges
    /// workers through `signal`.
    #[must_use]
    pub fn new(store: Arc<dyn AgentTaskStore>, signal: Arc<WakeupSignal>) -> Self {
        Self { store, signal }
    }

    /// Access the underlying signal.  Used by the host layer to share
    /// the same signal with its worker pool.
    #[must_use]
    pub const fn signal(&self) -> &Arc<WakeupSignal> {
        &self.signal
    }
}

#[async_trait]
impl TaskWakeupHandler for JournalTaskWakeupHandler {
    async fn handle_payload(
        &self,
        payload: &TaskWakeupPayload,
        _now: OffsetDateTime,
    ) -> Result<TaskWakeupOutcome> {
        let maybe_task = self
            .store
            .get(&payload.task_id)
            .await
            .with_context(|| format!("load task {} during wakeup re-check", payload.task_id))?;

        Ok(maybe_task.map_or_else(
            || {
                log::debug!(
                    task_id = payload.task_id.as_str(),
                    thread_id = &payload.thread_id.0;
                    "wakeup payload references task that does not exist; treating as benign duplicate",
                );
                TaskWakeupOutcome::Missing
            },
            |task| self.classify(&task, payload),
        ))
    }
}

impl JournalTaskWakeupHandler {
    fn classify(&self, task: &AgentTask, payload: &TaskWakeupPayload) -> TaskWakeupOutcome {
        let status = task.status;
        if task.thread_id != payload.thread_id {
            // Advisory payloads are just hints; a thread-id mismatch
            // shouldn't happen in practice but if it does we still
            // nudge the pool so the journal can serve as the final
            // arbiter.
            log::warn!(
                task_id = task.id.as_str(),
                payload_thread = &payload.thread_id.0,
                stored_thread = &task.thread_id.0;
                "wakeup payload thread mismatch with durable task row; nudging anyway",
            );
        }

        if status.is_runnable() || status == TaskStatus::Queued {
            log::debug!(
                task_id = task.id.as_str(),
                thread_id = &task.thread_id.0,
                status = format!("{status:?}");
                "wakeup re-check found runnable/queued task; nudging worker pool",
            );
            self.signal.notify_workers();
            TaskWakeupOutcome::Nudged { status }
        } else {
            log::debug!(
                task_id = task.id.as_str(),
                thread_id = &task.thread_id.0,
                status = format!("{status:?}");
                "wakeup re-check found non-runnable task; skipping nudge",
            );
            TaskWakeupOutcome::NotRunnable { status }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fallback sweep
// ─────────────────────────────────────────────────────────────────────

/// Simple fallback that nudges the worker pool on a fixed cadence.
///
/// The worker pool's own acquisition ticker is already a fallback
/// against a silent broker, but a separate sweep gives deployments an
/// explicit knob to tighten the guarantee: even if every worker is
/// busy and every broker delivery is lost, a `FallbackWakeupSweep` run
/// on a short interval ensures that [`WakeupSignal::wake_all_now`]
/// fires and any worker that completes in between tick wakes up
/// immediately.
///
/// The sweep does not consult the store — it is a pure time-based
/// pulse.  All correctness still lives in the CAS the worker runs
/// after being woken.
pub struct FallbackWakeupSweep {
    signal: Arc<WakeupSignal>,
    interval: std::time::Duration,
}

impl FallbackWakeupSweep {
    /// Construct a sweep that notifies every `interval` once [`run`] is
    /// driven.
    ///
    /// [`run`]: Self::run
    #[must_use]
    pub const fn new(signal: Arc<WakeupSignal>, interval: std::time::Duration) -> Self {
        Self { signal, interval }
    }

    /// Drive the sweep until `cancel` fires.  Cheap to spawn on a
    /// dedicated tokio task.
    pub async fn run(&self, cancel: CancellationToken) {
        log::info!(
            interval_secs = self.interval.as_secs();
            "fallback wakeup sweep starting",
        );
        let mut ticker = tokio::time::interval(self.interval);
        // Skip the immediate first tick so startup does not produce
        // a duplicate nudge on top of whatever the consumer / backfill
        // is already doing.
        ticker.tick().await;

        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    log::info!("fallback wakeup sweep shutting down");
                    return;
                }
                _ = ticker.tick() => {
                    log::debug!("fallback wakeup sweep firing pulse");
                    self.signal.wake_all_now();
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helper: run one-shot payload dispatch
// ─────────────────────────────────────────────────────────────────────

/// Convenience wrapper that boxes the payload dispatch so consumers can
/// log around the handler without duplicating the invocation shape.
///
/// # Errors
/// Propagates handler errors unchanged.
pub async fn dispatch_payload(
    handler: &(dyn TaskWakeupHandler + '_),
    payload: &TaskWakeupPayload,
    now: OffsetDateTime,
) -> Result<TaskWakeupOutcome> {
    let outcome = handler.handle_payload(payload, now).await?;
    match &outcome {
        TaskWakeupOutcome::Nudged { status } => {
            log::debug!(
                task_id = payload.task_id.as_str(),
                thread_id = &payload.thread_id.0,
                status = format!("{status:?}");
                "wakeup dispatched",
            );
        }
        TaskWakeupOutcome::NotRunnable { status } => {
            log::debug!(
                task_id = payload.task_id.as_str(),
                thread_id = &payload.thread_id.0,
                status = format!("{status:?}");
                "wakeup observed non-runnable task",
            );
        }
        TaskWakeupOutcome::Missing => {
            log::debug!(
                task_id = payload.task_id.as_str(),
                thread_id = &payload.thread_id.0;
                "wakeup observed missing task",
            );
        }
    }
    Ok(outcome)
}

// ─────────────────────────────────────────────────────────────────────
// In-memory capture handler (test double)
// ─────────────────────────────────────────────────────────────────────

/// Test double that records every handled payload and lets the test
/// control what outcome the handler reports.
///
/// Keeping this helper alongside the trait keeps external test
/// harnesses (broker consumers, worker loops) from having to duplicate
/// the same mock across crates.
#[derive(Default)]
pub struct CapturingTaskWakeupHandler {
    inner: tokio::sync::Mutex<CapturingInner>,
}

#[derive(Default)]
struct CapturingInner {
    payloads: Vec<TaskWakeupPayload>,
    outcome: Option<TaskWakeupOutcome>,
    error: Option<String>,
}

impl CapturingTaskWakeupHandler {
    /// Build an empty capturing handler.  Uses
    /// [`TaskWakeupOutcome::Missing`] as the default reply so a test
    /// that forgets to configure the outcome never accidentally looks
    /// like a successful nudge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot every payload delivered to the handler in arrival
    /// order.
    pub async fn payloads(&self) -> Vec<TaskWakeupPayload> {
        self.inner.lock().await.payloads.clone()
    }

    /// Override the outcome every future call returns.
    pub async fn reply_with(&self, outcome: TaskWakeupOutcome) {
        let mut inner = self.inner.lock().await;
        inner.outcome = Some(outcome);
        inner.error = None;
    }

    /// Force every future call to fail with this error string.
    pub async fn fail_with(&self, message: impl Into<String>) {
        let mut inner = self.inner.lock().await;
        inner.error = Some(message.into());
        inner.outcome = None;
    }
}

#[async_trait]
impl TaskWakeupHandler for CapturingTaskWakeupHandler {
    async fn handle_payload(
        &self,
        payload: &TaskWakeupPayload,
        _now: OffsetDateTime,
    ) -> Result<TaskWakeupOutcome> {
        let mut inner = self.inner.lock().await;
        inner.payloads.push(payload.clone());
        if let Some(err) = inner.error.clone() {
            anyhow::bail!(err);
        }
        Ok(inner.outcome.clone().unwrap_or(TaskWakeupOutcome::Missing))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::store::InMemoryAgentTaskStore;
    use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
    use agent_sdk_core::ThreadId;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration as StdDuration;
    use time::Duration as TimeDuration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + TimeDuration::seconds(1_700_000_000)
    }

    fn sample_thread() -> ThreadId {
        ThreadId::from_string("t-wakeup")
    }

    async fn seed_pending_root(
        store: &InMemoryAgentTaskStore,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let task = AgentTask::new_root_turn(sample_thread(), now, 3);
        let admitted = store.submit_root_turn(task).await?;
        Ok(admitted)
    }

    async fn set_task_running(
        store: &InMemoryAgentTaskStore,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let acquired = store
            .try_acquire_task(
                id,
                WorkerId::from_string("test-worker"),
                LeaseId::new(),
                now + TimeDuration::seconds(30),
                now,
            )
            .await?
            .context("task must be acquirable")?;
        Ok(acquired)
    }

    #[tokio::test]
    async fn signal_buffers_a_single_permit_across_producer_first() {
        let signal = WakeupSignal::shared();
        signal.notify_workers();
        // A waiter that parks AFTER the notify still completes.
        let result =
            tokio::time::timeout(StdDuration::from_millis(100), signal.wait_for_nudge()).await;
        assert!(result.is_ok(), "buffered notify should wake a late waiter");
    }

    #[tokio::test]
    async fn signal_collapses_duplicate_notifies_to_one_wake_up() {
        let signal = WakeupSignal::shared();
        signal.notify_workers();
        signal.notify_workers();
        signal.notify_workers();

        // First wait completes immediately.
        signal.wait_for_nudge().await;
        // Second wait must NOT complete — the three notifies fold into
        // a single permit.
        let second =
            tokio::time::timeout(StdDuration::from_millis(50), signal.wait_for_nudge()).await;
        assert!(
            second.is_err(),
            "duplicate notify_workers calls must collapse to one permit",
        );
    }

    #[tokio::test]
    async fn handler_nudges_for_pending_task() -> Result<()> {
        let store: Arc<InMemoryAgentTaskStore> = Arc::new(InMemoryAgentTaskStore::new());
        let admitted = seed_pending_root(&store, t0()).await?;

        let signal = WakeupSignal::shared();
        let handler = JournalTaskWakeupHandler::new(store.clone(), Arc::clone(&signal));
        let payload = TaskWakeupPayload {
            task_id: admitted.id.clone(),
            thread_id: sample_thread(),
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert_eq!(
            outcome,
            TaskWakeupOutcome::Nudged {
                status: TaskStatus::Pending
            }
        );
        // Nudge buffered — a subsequent wait must complete immediately.
        tokio::time::timeout(StdDuration::from_millis(100), signal.wait_for_nudge())
            .await
            .context("signal did not fire after nudge")?;
        Ok(())
    }

    #[tokio::test]
    async fn handler_does_not_nudge_running_task() -> Result<()> {
        let store: Arc<InMemoryAgentTaskStore> = Arc::new(InMemoryAgentTaskStore::new());
        let admitted = seed_pending_root(&store, t0()).await?;
        let _running = set_task_running(&store, &admitted.id, t0()).await?;

        let signal = WakeupSignal::shared();
        let handler = JournalTaskWakeupHandler::new(store.clone(), Arc::clone(&signal));
        let payload = TaskWakeupPayload {
            task_id: admitted.id,
            thread_id: sample_thread(),
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert_eq!(
            outcome,
            TaskWakeupOutcome::NotRunnable {
                status: TaskStatus::Running
            }
        );
        // No nudge should have fired.
        let res = tokio::time::timeout(StdDuration::from_millis(50), signal.wait_for_nudge()).await;
        assert!(res.is_err(), "nudge fired despite running task");
        Ok(())
    }

    #[tokio::test]
    async fn handler_reports_missing_for_unknown_task() -> Result<()> {
        let store: Arc<InMemoryAgentTaskStore> = Arc::new(InMemoryAgentTaskStore::new());
        let signal = WakeupSignal::shared();
        let handler = JournalTaskWakeupHandler::new(store.clone(), Arc::clone(&signal));

        let payload = TaskWakeupPayload {
            task_id: AgentTaskId::from_string("task_does_not_exist"),
            thread_id: sample_thread(),
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert_eq!(outcome, TaskWakeupOutcome::Missing);
        let res = tokio::time::timeout(StdDuration::from_millis(50), signal.wait_for_nudge()).await;
        assert!(res.is_err(), "nudge fired despite missing task");
        Ok(())
    }

    #[tokio::test]
    async fn duplicate_handle_payload_still_nudges_only_once() -> Result<()> {
        let store: Arc<InMemoryAgentTaskStore> = Arc::new(InMemoryAgentTaskStore::new());
        let admitted = seed_pending_root(&store, t0()).await?;

        let signal = WakeupSignal::shared();
        let handler = JournalTaskWakeupHandler::new(store.clone(), Arc::clone(&signal));
        let payload = TaskWakeupPayload {
            task_id: admitted.id,
            thread_id: sample_thread(),
        };
        handler.handle_payload(&payload, t0()).await?;
        handler.handle_payload(&payload, t0()).await?;
        handler.handle_payload(&payload, t0()).await?;

        // One permit released — at most one waiter wakes.
        signal.wait_for_nudge().await;
        let second =
            tokio::time::timeout(StdDuration::from_millis(50), signal.wait_for_nudge()).await;
        assert!(
            second.is_err(),
            "duplicate wakeups must not accumulate extra nudges",
        );
        Ok(())
    }

    #[tokio::test]
    async fn handler_queues_kind_is_nudged_too() -> Result<()> {
        // Two roots on the same thread: the first is admitted as
        // Pending and holds the active-root slot; the second ends up
        // Queued.  A wakeup for the queued one must still produce a
        // nudge so the worker pool eventually promotes it.
        let store: Arc<InMemoryAgentTaskStore> = Arc::new(InMemoryAgentTaskStore::new());
        let first = seed_pending_root(&store, t0()).await?;
        assert_eq!(first.status, TaskStatus::Pending);
        let second_task = AgentTask::new_root_turn(sample_thread(), t0(), 3);
        let second = store.submit_root_turn(second_task).await?;
        assert_eq!(second.status, TaskStatus::Queued);

        let signal = WakeupSignal::shared();
        let handler = JournalTaskWakeupHandler::new(store.clone(), Arc::clone(&signal));
        let payload = TaskWakeupPayload {
            task_id: second.id,
            thread_id: sample_thread(),
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert_eq!(
            outcome,
            TaskWakeupOutcome::Nudged {
                status: TaskStatus::Queued
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn capturing_handler_records_and_overrides_outcome() -> Result<()> {
        let handler = CapturingTaskWakeupHandler::new();
        handler
            .reply_with(TaskWakeupOutcome::Nudged {
                status: TaskStatus::Pending,
            })
            .await;
        let payload = TaskWakeupPayload {
            task_id: AgentTaskId::new(),
            thread_id: sample_thread(),
        };
        let outcome = handler.handle_payload(&payload, t0()).await?;
        assert!(outcome.nudged());

        let calls = handler.payloads().await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].task_id, payload.task_id);
        Ok(())
    }

    #[tokio::test]
    async fn capturing_handler_surfaces_errors_for_retry_testing() -> Result<()> {
        let handler = CapturingTaskWakeupHandler::new();
        handler.fail_with("simulated lookup failure").await;
        let payload = TaskWakeupPayload {
            task_id: AgentTaskId::new(),
            thread_id: sample_thread(),
        };
        let Err(err) = handler.handle_payload(&payload, t0()).await else {
            anyhow::bail!("expected handler to fail, got Ok");
        };
        let rendered = format!("{err:#}");
        assert!(rendered.contains("simulated lookup failure"));
        Ok(())
    }

    #[tokio::test]
    async fn fallback_sweep_fires_periodically_until_cancelled() -> Result<()> {
        let signal = WakeupSignal::shared();
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = Arc::clone(&fired);
        let waiter_signal = Arc::clone(&signal);
        let waiter_cancel = CancellationToken::new();
        let waiter_cancel_clone = waiter_cancel.clone();
        let waiter = tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = waiter_cancel_clone.cancelled() => return,
                    () = waiter_signal.wait_for_nudge() => {
                        fired_clone.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });

        let sweep = FallbackWakeupSweep::new(Arc::clone(&signal), StdDuration::from_millis(20));
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let sweep_handle = tokio::spawn(async move { sweep.run(cancel_clone).await });

        tokio::time::sleep(StdDuration::from_millis(100)).await;
        cancel.cancel();
        sweep_handle.await?;
        waiter_cancel.cancel();
        let _ = waiter.await;
        assert!(
            fired.load(Ordering::SeqCst) >= 2,
            "fallback sweep should have fired at least twice in 100 ms",
        );
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_payload_forwards_outcome() -> Result<()> {
        let handler = CapturingTaskWakeupHandler::new();
        handler.reply_with(TaskWakeupOutcome::Missing).await;
        let payload = TaskWakeupPayload {
            task_id: AgentTaskId::new(),
            thread_id: sample_thread(),
        };
        let outcome = dispatch_payload(&handler, &payload, t0()).await?;
        assert_eq!(outcome, TaskWakeupOutcome::Missing);
        Ok(())
    }
}
