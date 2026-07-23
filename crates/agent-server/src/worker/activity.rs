//! In-memory "last evidence of work" clock for one running task.
//!
//! The subagent stall budget (`spec.timeout_ms`) fails a child only after
//! it has gone a whole budget with no evidence of work. Committed events
//! alone cannot answer that question — a child parked on one long tool
//! call commits nothing until the tool returns, a pure tool-call provider
//! stream journals nothing while it is actively yielding frames, and
//! event retention can purge an event that fell inside the window.
//!
//! So work is also recorded out-of-band, here. Execution bumps the beacon
//! at every sign of life; the task's own lease heartbeat reads it once per
//! tick and persists it to `agent_sdk_tasks.last_activity_at`. That keeps
//! the durable write rate at one per tick instead of one per provider
//! frame, at the cost of the durable value lagging the beacon by at most a
//! tick — irrelevant against a budget measured in minutes, and the
//! heartbeat path reads the beacon directly, so enforcement for a RUNNING
//! task is exact.
//!
//! ## This is NOT the heartbeat
//!
//! The lease heartbeat renews unconditionally while the task future is
//! alive, so a child hung on a half-open connection heartbeats forever —
//! that is precisely the failure the stall budget exists to catch. A
//! heartbeat proves the process is alive; the beacon only advances when
//! the task actually *did* something.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage};

use crate::journal::committed_event::CommittedEvent;
use crate::journal::event_outbox_transaction::AtomicEventOutboxCommitter;
use crate::journal::event_repository::EventRepository;

/// A shared, monotonic clock of the newest sign of work on one task.
///
/// Cloneable and cheap: every clone observes the same underlying instant,
/// so the executor, the tool collector and the heartbeat loop can each
/// hold one.
///
/// A [`Default`] beacon has never been bumped and reads as `None`, which
/// is why the collector and the worker can hold one unconditionally —
/// callers that do not care about activity pay one relaxed atomic store
/// per bump and nothing else.
#[derive(Clone, Debug, Default)]
pub struct ActivityBeacon {
    /// Unix nanoseconds of the newest observed sign of work. `0` means
    /// "never bumped" — the epoch itself is not a representable activity
    /// instant in any live system, so it doubles as the sentinel.
    latest_unix_nanos: Arc<AtomicI64>,
    /// Usage streamed since the last turn commit — see
    /// [`ActivityBeacon::record_usage`] for why this is *uncommitted*
    /// usage rather than a running total.
    uncommitted_usage: Arc<AtomicTokenUsage>,
}

/// The four [`TokenUsage`] counters as lock-free atomics.
///
/// Exists so [`ActivityBeacon`] holds one named aggregate instead of four
/// loose atomics: the counters are only ever read, added, and cleared
/// together, and a fifth counter should be one edit, not five.
///
/// Atomics rather than a `Mutex<TokenUsage>` because `add` runs on the
/// provider-frame path, which must never block. The four fields are not
/// updated as one atomic unit, so a concurrent [`Self::snapshot`] can
/// observe a torn mid-update value; that is deliberate and harmless — the
/// snapshot feeds a progress event, and the next tick reconciles.
#[derive(Debug, Default)]
struct AtomicTokenUsage {
    input: AtomicU32,
    output: AtomicU32,
    cache_read: AtomicU32,
    cache_creation: AtomicU32,
}

impl AtomicTokenUsage {
    /// Add one provider call's usage to the running total.
    ///
    /// Saturating: a counter that would wrap past `u32::MAX` pins there
    /// instead of restarting at zero, so the progress stream can never
    /// report a token count that fell off a cliff.
    fn add(&self, usage: &TokenUsage) {
        for (counter, delta) in [
            (&self.input, usage.input_tokens),
            (&self.output, usage.output_tokens),
            (&self.cache_read, usage.cached_input_tokens),
            (&self.cache_creation, usage.cache_creation_input_tokens),
        ] {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(delta))
            });
        }
    }

    fn snapshot(&self) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input.load(Ordering::Relaxed),
            output_tokens: self.output.load(Ordering::Relaxed),
            cached_input_tokens: self.cache_read.load(Ordering::Relaxed),
            cache_creation_input_tokens: self.cache_creation.load(Ordering::Relaxed),
        }
    }

    fn clear(&self) {
        self.input.store(0, Ordering::Relaxed);
        self.output.store(0, Ordering::Relaxed);
        self.cache_read.store(0, Ordering::Relaxed);
        self.cache_creation.store(0, Ordering::Relaxed);
    }
}

impl ActivityBeacon {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sign of work observed at `at`.
    ///
    /// Monotonic: the beacon never moves backwards, so an out-of-order or
    /// skewed observation cannot retire a newer one. Non-blocking, and
    /// safe to call on every provider frame.
    pub fn bump(&self, at: OffsetDateTime) {
        let Ok(nanos) = i64::try_from(at.unix_timestamp_nanos()) else {
            // Unrepresentable instant (year > 2262). Recording nothing is
            // the fail-safe direction: the durable fallbacks (committed
            // events, the spawn floor) still answer the stall question.
            return;
        };
        self.latest_unix_nanos.fetch_max(nanos, Ordering::Relaxed);
    }

    /// The newest sign of work, or `None` if this task has not produced
    /// one yet.
    #[must_use]
    pub fn latest(&self) -> Option<OffsetDateTime> {
        let nanos = self.latest_unix_nanos.load(Ordering::Relaxed);
        if nanos == 0 {
            return None;
        }
        OffsetDateTime::from_unix_timestamp_nanos(i128::from(nanos)).ok()
    }

    /// Add one provider call's usage to this turn's uncommitted total.
    ///
    /// The caller hands over a **per-call** `TokenUsage` built fresh from
    /// that call's `llm::Usage` — not a restated running total — so the
    /// reducer must be addition. Summing is also what makes the counters
    /// monotonic within a turn: each call only ever adds.
    ///
    /// What accumulates here is strictly the usage the thread aggregate
    /// does **not** know about yet. [`Self::take_usage`] hands it back at
    /// the turn-commit boundary, where `thread.total_usage` takes
    /// ownership of it; a reader that adds `thread.total_usage` to
    /// [`Self::usage`] therefore counts every token exactly once.
    pub fn record_usage(&self, usage: &TokenUsage) {
        self.uncommitted_usage.add(usage);
    }

    /// Snapshot the usage streamed since the last turn commit.
    #[must_use]
    pub fn usage(&self) -> TokenUsage {
        self.uncommitted_usage.snapshot()
    }

    /// Clear the uncommitted counters, returning what they held.
    ///
    /// Called once per successful turn commit, because that commit folds
    /// the same tokens into the thread's durable `total_usage`. Skipping
    /// it is a double-count that never self-corrects: every later progress
    /// tick reports `total_usage + the same tokens again`.
    #[must_use]
    pub fn take_usage(&self) -> TokenUsage {
        let taken = self.uncommitted_usage.snapshot();
        self.uncommitted_usage.clear();
        taken
    }
}

/// An [`EventRepository`] that records **every** successful commit on a task's
/// [`ActivityBeacon`] — refresh point (c) of the stall contract (ADR-0003).
///
/// ## Why this is a decorator and not a call-site bump
///
/// It used to be a call-site bump, and call sites get forgotten. The streaming
/// -delta flush bumped the beacon; the auto-retry start/end events, the
/// compaction event, the turn-start and user-input events, and the suspension
/// batches did **not**. Each omission is the same bug: a child commits real
/// work, event retention later purges the evidence, and the stall probe — which
/// falls back to the durable beacon precisely because retention can purge
/// events — reads the child as silent and reaps it. A child in retry backoff or
/// finishing a compaction is exactly the kind of child this kills.
///
/// Wrapping the repository makes the bump **unmissable**: on this path a commit
/// that does not record activity is now unrepresentable, including for call
/// sites that do not exist yet. That is the difference between a rule and a
/// mechanism.
///
/// Only *successful* commits count — the `?` short-circuits on error, so a
/// failed write is not evidence of work.
///
/// ## Known boundary
///
/// [`EventRepository::atomic_event_outbox_committer`] hands out a raw committer
/// that writes events **without** passing through [`Self::commit_event`], so a
/// caller using it would bypass the beacon. It is delegated unchanged (there is
/// no production caller on any commit path today, and suppressing it would
/// silently drop the backends' atomic event+outbox unit of work). If a commit
/// path ever adopts that hook it must bump the beacon itself — see ADR-0003's
/// Consequences.
pub struct ActivityTrackingEventRepo<'a> {
    inner: &'a dyn EventRepository,
    activity: ActivityBeacon,
}

impl<'a> ActivityTrackingEventRepo<'a> {
    #[must_use]
    pub fn new(inner: &'a dyn EventRepository, activity: ActivityBeacon) -> Self {
        Self { inner, activity }
    }
}

#[async_trait]
impl EventRepository for ActivityTrackingEventRepo<'_> {
    fn atomic_event_outbox_committer(&self) -> Option<&dyn AtomicEventOutboxCommitter> {
        self.inner.atomic_event_outbox_committer()
    }

    async fn commit_event(
        &self,
        thread_id: &ThreadId,
        event: AgentEvent,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        let committed = self.inner.commit_event(thread_id, event, now).await?;
        self.activity.bump(now);
        Ok(committed)
    }

    async fn commit_event_batch(
        &self,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        let committed = self
            .inner
            .commit_event_batch(thread_id, events, now)
            .await?;
        self.activity.bump(now);
        Ok(committed)
    }

    async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
        self.inner.next_sequence(thread_id).await
    }

    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
        self.inner.get_events(thread_id).await
    }

    async fn get_events_in_range(
        &self,
        thread_id: &ThreadId,
        after_sequence: u64,
        up_to_sequence: u64,
    ) -> Result<Vec<CommittedEvent>> {
        self.inner
            .get_events_in_range(thread_id, after_sequence, up_to_sequence)
            .await
    }

    async fn threads_with_events_before(
        &self,
        cutoff: OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        self.inner.threads_with_events_before(cutoff, limit).await
    }

    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        self.inner.max_sequence_before(thread_id, cutoff).await
    }

    async fn min_sequence_at_or_after(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        self.inner.min_sequence_at_or_after(thread_id, cutoff).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: i64) -> OffsetDateTime {
        // Infallible by construction: offsetting the epoch saturates
        // rather than erroring, so no `expect`/`unwrap` in test code.
        OffsetDateTime::UNIX_EPOCH.saturating_add(time::Duration::seconds(secs))
    }

    #[test]
    fn unbumped_beacon_reads_none() {
        assert_eq!(ActivityBeacon::new().latest(), None);
    }

    #[test]
    fn bump_is_observed_by_every_clone() {
        let beacon = ActivityBeacon::new();
        let observer = beacon.clone();
        beacon.bump(at(1_000));
        assert_eq!(observer.latest(), Some(at(1_000)));
    }

    fn usage(input: u32, output: u32, cache_read: u32, cache_creation: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cached_input_tokens: cache_read,
            cache_creation_input_tokens: cache_creation,
        }
    }

    /// Per-call usage ACCUMULATES; it is not a running total to max over.
    ///
    /// `note_stream_usage` builds a fresh `TokenUsage` from each provider
    /// call's own `llm::Usage`, so a turn of N calls hands over N
    /// independent values. The reducer used to be `fetch_max`, which keeps
    /// only the single largest call and silently drops the rest — a
    /// five-call turn reported roughly one call's tokens. The counters
    /// below are chosen so max-of-two and sum-of-two cannot coincide on
    /// any field.
    #[test]
    fn record_usage_sums_every_call_rather_than_keeping_the_largest() {
        let beacon = ActivityBeacon::new();

        beacon.record_usage(&usage(100, 20, 7, 3));
        beacon.record_usage(&usage(30, 50, 1, 9));

        assert_eq!(
            beacon.usage(),
            usage(130, 70, 8, 12),
            "each provider call's usage must add; keeping the max would give (100, 50, 7, 9)",
        );
    }

    /// The beacon carries only UNCOMMITTED usage, so a live reader can add
    /// it to the thread total without double-counting.
    ///
    /// Replays the exact defect: turn 1 streams U, the turn commits and
    /// folds U into `thread.total_usage`, and the next progress tick reads
    /// `total_usage + beacon.usage()`. With no hand-off at the commit
    /// boundary that tick reports 2U — and stays inflated for the rest of
    /// the subagent's life, because nothing ever subtracts.
    #[test]
    fn take_usage_hands_the_turn_off_so_a_committed_turn_is_not_counted_twice() {
        let beacon = ActivityBeacon::new();
        beacon.record_usage(&usage(100, 20, 7, 3));

        let committed = beacon.take_usage();
        assert_eq!(
            committed,
            usage(100, 20, 7, 3),
            "the commit boundary must receive exactly what the turn streamed",
        );

        // The thread aggregate now owns those tokens.
        let mut thread_total = TokenUsage::default();
        thread_total.add(&committed);

        assert_eq!(
            beacon.usage(),
            TokenUsage::default(),
            "a committed turn must leave nothing behind on the beacon",
        );

        let mut live_total = thread_total.clone();
        live_total.add(&beacon.usage());
        assert_eq!(
            live_total,
            usage(100, 20, 7, 3),
            "the post-commit progress tick must report the true total, not double it",
        );

        // Turn 2 streams on the same beacon: only the NEW usage is live.
        beacon.record_usage(&usage(5, 6, 0, 0));
        let mut mid_turn_two = thread_total;
        mid_turn_two.add(&beacon.usage());
        assert_eq!(
            mid_turn_two,
            usage(105, 26, 7, 3),
            "turn 2's live total is turn 1 committed plus turn 2 so far",
        );
    }

    #[test]
    fn beacon_never_moves_backwards() {
        let beacon = ActivityBeacon::new();
        beacon.bump(at(2_000));
        beacon.bump(at(1_000));
        assert_eq!(
            beacon.latest(),
            Some(at(2_000)),
            "an older observation must not retire a newer one",
        );
    }

    /// ADR-0003 refresh point (c) — EVERY successful commit is evidence of
    /// work, not just the streaming-delta flush.
    ///
    /// The bump used to live at call sites, and the call sites that were not
    /// the delta flush never got one: auto-retry start/end, the compaction
    /// event, turn-start / user-input, and the suspension batches. A child in
    /// retry backoff or finishing a compaction commits real work, has that
    /// work purged by event retention, and is then reaped as silent — the
    /// durable beacon is the ONLY thing retention cannot purge, and it was not
    /// being written.
    ///
    /// Wrapping the repository makes the bump unmissable. Both commit
    /// primitives are exercised here, plus the negative: a FAILED commit is
    /// not work.
    #[tokio::test]
    async fn every_successful_commit_through_the_tracked_repo_marks_the_task_alive()
    -> anyhow::Result<()> {
        use crate::journal::event_repository::InMemoryEventRepository;

        let thread = ThreadId::from_string("t-tracked-commit");
        let inner = InMemoryEventRepository::new();
        let beacon = ActivityBeacon::new();
        let repo = ActivityTrackingEventRepo::new(&inner, beacon.clone());

        assert_eq!(beacon.latest(), None, "beacon starts empty");

        // A single non-delta commit — the shape auto-retry and compaction use.
        repo.commit_event(&thread, AgentEvent::text("m1", "retrying"), at(1_000))
            .await?;
        assert_eq!(
            beacon.latest(),
            Some(at(1_000)),
            "a single event commit must mark the task alive",
        );

        // A batch commit — the shape the suspension/content flushes use.
        repo.commit_event_batch(&thread, vec![AgentEvent::text("m2", "done")], at(2_000))
            .await?;
        assert_eq!(
            beacon.latest(),
            Some(at(2_000)),
            "a batch commit must mark the task alive",
        );

        // The negative: an empty batch is rejected by the repository, and a
        // FAILED commit is not evidence of work — the beacon must not move.
        assert!(
            repo.commit_event_batch(&thread, Vec::new(), at(3_000))
                .await
                .is_err(),
            "an empty batch is a rejected commit",
        );
        assert_eq!(
            beacon.latest(),
            Some(at(2_000)),
            "a commit that FAILED is not evidence of work",
        );

        Ok(())
    }
}
