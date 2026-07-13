//! Built-in retention janitor policy.
//!
//! [`run_janitor_cycle`] is a stateless function that evaluates
//! configurable retention policies against the store layer and
//! executes the resulting cleanup.  It is called periodically by the
//! service host's background loop.
//!
//! # Safety invariants
//!
//! 1. The janitor never deletes events referenced by unpublished
//!    (pending / claimed) outbox rows.
//! 2. The janitor always preserves at least the latest checkpoint for
//!    every thread.
//! 3. The retention floor only advances — never backwards.

use anyhow::{Context, Result};
use time::OffsetDateTime;

use super::checkpoint_store::CheckpointStore;
use super::event_repository::EventRepository;
use super::outbox::OutboxStore;
use super::retention::RetentionStore;

// ─────────────────────────────────────────────────────────────────────
// Policy
// ─────────────────────────────────────────────────────────────────────

/// Configurable retention policy evaluated by each janitor cycle.
#[derive(Clone, Debug)]
pub struct RetentionPolicy {
    /// Events older than this are eligible for purge.  `None` = keep
    /// forever (the janitor skips event cleanup).
    pub event_ttl: Option<std::time::Duration>,
    /// Maximum checkpoints per thread.  `None` = no limit.
    pub checkpoint_max_per_thread: Option<u32>,
    /// Maximum threads processed per janitor cycle.  The event-TTL
    /// and checkpoint-limit passes draw from this same shared budget,
    /// so the combined number of threads touched per cycle never
    /// exceeds this value.
    pub batch_size: u32,
}

// ─────────────────────────────────────────────────────────────────────
// Dependencies
// ─────────────────────────────────────────────────────────────────────

/// Store references needed by the janitor.
pub struct RetentionJanitorDeps<'a> {
    pub event_repo: &'a dyn EventRepository,
    pub retention_store: &'a dyn RetentionStore,
    pub outbox_store: &'a dyn OutboxStore,
    pub checkpoint_store: &'a dyn CheckpointStore,
}

// ─────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────

/// Summary of a single janitor sweep cycle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JanitorCycleReport {
    /// Threads on which the cycle performed useful work (a floor
    /// advance and/or a standalone checkpoint prune).
    pub threads_scanned: u32,
    /// Number of sequences the retention floor advanced past this cycle
    /// (summed across threads).
    ///
    /// This is a **floor-advance delta**, not a count of physically
    /// deleted rows.  On durable backends the delta equals the rows
    /// deleted in the same transaction; against the cursor-only
    /// [`InMemoryRetentionStore`](super::retention::InMemoryRetentionStore)
    /// no rows are actually removed, so the count reflects logical
    /// reclamation only.
    pub events_purged: u64,
    pub checkpoints_pruned: u64,
    pub floors_advanced: u32,
}

// ─────────────────────────────────────────────────────────────────────
// Cycle
// ─────────────────────────────────────────────────────────────────────

/// Run one janitor sweep cycle.
///
/// Returns a report summarising the work performed.  The caller
/// (the host background loop) decides whether to log or act on the
/// report.
///
/// # Errors
///
/// Returns an error if any store query or mutation fails.
pub async fn run_janitor_cycle(
    policy: &RetentionPolicy,
    deps: &RetentionJanitorDeps<'_>,
    now: OffsetDateTime,
) -> Result<JanitorCycleReport> {
    let mut report = JanitorCycleReport::default();

    // Shared per-cycle thread budget: both passes draw from the same
    // pool so the combined thread count never exceeds `batch_size`,
    // matching the documented per-cycle contract.
    let mut remaining_budget = policy.batch_size;

    // ── Pass 1: event-TTL sweep ─────────────────────────────────
    //
    // Threads with events older than the TTL get their retention
    // floor advanced and old events purged.  Checkpoint pruning is
    // also performed on these threads as a side-effect when the
    // limit is configured.
    let mut event_swept_threads = std::collections::HashSet::new();

    if let Some(event_ttl) = policy.event_ttl {
        let ttl_secs = i64::try_from(event_ttl.as_secs()).unwrap_or(i64::MAX);
        let cutoff = now - time::Duration::seconds(ttl_secs);

        let threads = deps
            .event_repo
            .threads_with_events_before(cutoff, remaining_budget)
            .await
            .context("list threads needing retention")?;

        for thread_id in &threads {
            // ── Compute safe floor ──────────────────────────────
            //
            // We do NOT charge budget or mark the thread as swept
            // until after a useful advance completes below: threads
            // that exit via an early `continue` (TOCTOU where events
            // vanished, or outbox rows pinning the floor) must remain
            // eligible for Pass 2's standalone checkpoint sweep and
            // must not consume batch budget on no-op work.
            let candidate_seq = deps
                .event_repo
                .max_sequence_before(thread_id, cutoff)
                .await
                .context("read max sequence before cutoff")?;

            let Some(max_seq) = candidate_seq else {
                continue;
            };

            // Cap the floor to the FIRST live (non-expired) sequence,
            // if any.  This guarantees `advance_floor`, which deletes
            // everything below the floor regardless of per-row
            // `committed_at`, never destroys a live event — even when
            // `committed_at` is non-monotonic (clock skew, concurrent
            // writers, NTP step corrections produce a later sequence
            // with an earlier timestamp).
            let first_live_seq = deps
                .event_repo
                .min_sequence_at_or_after(thread_id, cutoff)
                .await
                .context("read min sequence at or after cutoff")?;

            let candidate_floor = first_live_seq.map_or(max_seq + 1, |live| live.min(max_seq + 1));

            let current_floor = deps
                .retention_store
                .effective_floor(thread_id)
                .await
                .context("read current retention floor")?;

            // Read the outbox safety bound LAST — immediately before the
            // advance decision — so any unpublished row that became
            // visible *after* the event-sequence reads above still pins
            // the floor.  Reading it first opens a TOCTOU window: under
            // clock skew a worker can commit a batch (with a caller-
            // supplied `now` older than the cutoff) plus its Pending
            // outbox row between the stale bound read and the sequence
            // reads, so `max_sequence_before` includes those events while
            // the bound fails to pin them — and the advance would purge
            // events still referenced by an unpublished advisory.
            // Durable backends should additionally re-check this bound
            // inside the `advance_floor` transaction.
            let outbox_bound = deps
                .outbox_store
                .min_unpublished_sequence(thread_id)
                .await
                .context("read outbox safety bound")?;

            let safe_floor =
                outbox_bound.map_or(candidate_floor, |min_unpub| candidate_floor.min(min_unpub));

            if safe_floor <= current_floor {
                continue;
            }

            // ── Advance floor (atomically deletes events) ───────
            deps.retention_store
                .advance_floor(thread_id, safe_floor, now)
                .await
                .with_context(|| format!("advance retention floor for {thread_id}"))?;

            report.floors_advanced += 1;
            let purged = safe_floor.saturating_sub(current_floor);
            report.events_purged += purged;

            // ── Prune checkpoints (piggy-back on event sweep) ───
            if let Some(limit) = policy.checkpoint_max_per_thread {
                let keep = limit.max(1);
                let pruned = deps
                    .checkpoint_store
                    .delete_checkpoints_beyond_limit(thread_id, keep)
                    .await
                    .with_context(|| format!("prune checkpoints for {thread_id}"))?;
                report.checkpoints_pruned += pruned;
            }

            // ── Charge budget and mark as swept (useful work done)
            report.threads_scanned += 1;
            event_swept_threads.insert(thread_id.clone());
            remaining_budget = remaining_budget.saturating_sub(1);
        }
    }

    // ── Pass 2: standalone checkpoint pruning ───────────────────
    //
    // Threads that were NOT already processed in pass 1 but have
    // more checkpoints than the configured limit.  This ensures
    // checkpoint_max_per_thread works independently of event_ttl.
    // Uses the remaining portion of the shared `batch_size` budget
    // so the total thread count across both passes never exceeds it.
    if remaining_budget > 0
        && let Some(limit) = policy.checkpoint_max_per_thread
    {
        let keep = limit.max(1);
        let threads = deps
            .checkpoint_store
            .threads_exceeding_checkpoint_count(keep, remaining_budget)
            .await
            .context("list threads with excess checkpoints")?;

        for thread_id in &threads {
            if event_swept_threads.contains(thread_id) {
                continue;
            }
            if remaining_budget == 0 {
                break;
            }
            report.threads_scanned += 1;
            remaining_budget = remaining_budget.saturating_sub(1);
            let pruned = deps
                .checkpoint_store
                .delete_checkpoints_beyond_limit(thread_id, keep)
                .await
                .with_context(|| format!("prune checkpoints for {thread_id}"))?;
            report.checkpoints_pruned += pruned;
        }
    }

    Ok(report)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::checkpoint::CheckpointKind;
    use crate::journal::checkpoint_store::InMemoryCheckpointStore;
    use crate::journal::event_repository::InMemoryEventRepository;
    use crate::journal::outbox::InMemoryOutboxStore;
    use crate::journal::retention::InMemoryRetentionStore;
    use agent_sdk_foundation::ThreadId;
    use agent_sdk_foundation::events::AgentEvent;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-janitor-a")
    }

    fn thread_b() -> ThreadId {
        ThreadId::from_string("t-janitor-b")
    }

    fn deps<'a>(
        events: &'a InMemoryEventRepository,
        retention: &'a InMemoryRetentionStore,
        outbox: &'a InMemoryOutboxStore,
        checkpoints: &'a InMemoryCheckpointStore,
    ) -> RetentionJanitorDeps<'a> {
        RetentionJanitorDeps {
            event_repo: events,
            retention_store: retention,
            outbox_store: outbox,
            checkpoint_store: checkpoints,
        }
    }

    fn policy_with_ttl(ttl_secs: u64) -> RetentionPolicy {
        RetentionPolicy {
            event_ttl: Some(std::time::Duration::from_secs(ttl_secs)),
            checkpoint_max_per_thread: None,
            batch_size: 100,
        }
    }

    // ── Behaviour tests ─────────────────────────────────────────

    #[tokio::test]
    async fn janitor_noop_when_policies_are_none() -> Result<()> {
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;

        let policy = RetentionPolicy {
            event_ttl: None,
            checkpoint_max_per_thread: None,
            batch_size: 100,
        };

        let report = run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(3600),
        )
        .await?;
        assert_eq!(report, JanitorCycleReport::default());
        Ok(())
    }

    #[tokio::test]
    async fn janitor_skips_when_no_expired_events() -> Result<()> {
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "a"), t0())
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(100),
        )
        .await?;

        assert_eq!(report.threads_scanned, 0);
        assert_eq!(report.floors_advanced, 0);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_advances_floor_for_expired_events() -> Result<()> {
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        // Old events at t0, new event at t0+7200
        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old"), t0())
            .await?;
        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "old"), t_plus(1))
            .await?;
        events
            .commit_event(&thread_a(), AgentEvent::text("m2", "new"), t_plus(7200))
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(7200),
        )
        .await?;

        assert_eq!(report.threads_scanned, 1);
        assert_eq!(report.floors_advanced, 1);
        assert_eq!(report.events_purged, 2);

        let floor = retention.effective_floor(&thread_a()).await?;
        assert_eq!(floor, 2);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_respects_outbox_safety_bound() -> Result<()> {
        use crate::journal::outbox::{NewOutboxRow, OutboxStore};
        use crate::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        // Commit 3 old events
        for i in 0..3u64 {
            events
                .commit_event(&thread_a(), AgentEvent::text(format!("m{i}"), "old"), t0())
                .await?;
        }

        // Insert an unpublished outbox row referencing sequence 1
        let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_a(),
            last_sequence: 1,
        })
        .to_payload_json()?;
        outbox
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_a(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(1),
                payload_json: payload,
                max_attempts: 3,
                now: t0(),
            }])
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(7200),
        )
        .await?;

        // Floor should be capped at 1 (the min unpublished sequence)
        let floor = retention.effective_floor(&thread_a()).await?;
        assert_eq!(floor, 1);
        assert_eq!(report.events_purged, 1);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_prunes_excess_checkpoints() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_foundation::{TokenUsage, llm};

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        // Create events and checkpoints
        for i in 1..=5u32 {
            events
                .commit_event(&thread_a(), AgentEvent::text(format!("m{i}"), "old"), t0())
                .await?;
            checkpoint_store
                .commit_checkpoint(NewCheckpointParams {
                    kind: CheckpointKind::FullTurn,
                    thread_id: thread_a(),
                    turn_number: i,
                    task_id: AgentTaskId::from_string(format!("task-{i}")),
                    messages: vec![llm::Message::user(format!("turn {i}"))],
                    agent_state_snapshot: serde_json::json!({}),
                    turn_usage: TokenUsage::default(),
                    now: t0(),
                })
                .await?;
        }

        let policy = RetentionPolicy {
            event_ttl: Some(std::time::Duration::from_hours(1)),
            checkpoint_max_per_thread: Some(2),
            batch_size: 100,
        };

        let report = run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoint_store),
            t_plus(7200),
        )
        .await?;

        assert_eq!(report.checkpoints_pruned, 3);
        let remaining = checkpoint_store.list_by_thread(&thread_a()).await?;
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].turn_number, 4);
        assert_eq!(remaining[1].turn_number, 5);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_preserves_latest_checkpoint() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_foundation::{TokenUsage, llm};

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "old"), t0())
            .await?;
        checkpoint_store
            .commit_checkpoint(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: thread_a(),
                turn_number: 1,
                task_id: AgentTaskId::from_string("task-1"),
                messages: vec![llm::Message::user("turn 1")],
                agent_state_snapshot: serde_json::json!({}),
                turn_usage: TokenUsage::default(),
                now: t0(),
            })
            .await?;

        let policy = RetentionPolicy {
            event_ttl: Some(std::time::Duration::from_hours(1)),
            checkpoint_max_per_thread: Some(1),
            batch_size: 100,
        };

        run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoint_store),
            t_plus(7200),
        )
        .await?;

        let remaining = checkpoint_store.list_by_thread(&thread_a()).await?;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].turn_number, 1);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_is_idempotent() -> Result<()> {
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old"), t0())
            .await?;
        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "new"), t_plus(7200))
            .await?;

        let d = deps(&events, &retention, &outbox, &checkpoints);
        let policy = policy_with_ttl(3600);
        let now = t_plus(7200);

        let r1 = run_janitor_cycle(&policy, &d, now).await?;
        assert_eq!(r1.floors_advanced, 1);

        let r2 = run_janitor_cycle(&policy, &d, now).await?;
        assert_eq!(r2.floors_advanced, 0);
        assert_eq!(r2.events_purged, 0);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_handles_multiple_threads() -> Result<()> {
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old"), t0())
            .await?;
        events
            .commit_event(&thread_b(), AgentEvent::text("m0", "old"), t0())
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(7200),
        )
        .await?;

        assert_eq!(report.threads_scanned, 2);
        assert_eq!(report.floors_advanced, 2);

        assert_eq!(retention.effective_floor(&thread_a()).await?, 1);
        assert_eq!(retention.effective_floor(&thread_b()).await?, 1);
        Ok(())
    }

    #[tokio::test]
    async fn batch_size_is_shared_budget_across_passes() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_foundation::{TokenUsage, llm};

        // Two disjoint populations:
        //   - thread_a: has expired events (Pass 1 eligible).
        //   - thread_b: has excess checkpoints only (Pass 2 eligible).
        //
        // With batch_size = 1, the documented contract says at most
        // ONE thread may be touched per cycle. Previously batch_size
        // was applied independently to each pass, so up to 2 threads
        // could be touched.  This test guards against that regression.
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old"), t0())
            .await?;

        for i in 1..=3u32 {
            checkpoint_store
                .commit_checkpoint(NewCheckpointParams {
                    kind: CheckpointKind::FullTurn,
                    thread_id: thread_b(),
                    turn_number: i,
                    task_id: AgentTaskId::from_string(format!("task-b-{i}")),
                    messages: vec![llm::Message::user(format!("turn {i}"))],
                    agent_state_snapshot: serde_json::json!({}),
                    turn_usage: TokenUsage::default(),
                    now: t0(),
                })
                .await?;
        }

        let policy = RetentionPolicy {
            event_ttl: Some(std::time::Duration::from_hours(1)),
            checkpoint_max_per_thread: Some(1),
            batch_size: 1,
        };

        let report = run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoint_store),
            t_plus(7200),
        )
        .await?;

        assert_eq!(
            report.threads_scanned, 1,
            "shared budget violated: {} threads touched with batch_size=1",
            report.threads_scanned
        );
        Ok(())
    }

    #[tokio::test]
    async fn janitor_protects_entire_multi_event_batch_via_outbox_bound() -> Result<()> {
        use crate::journal::outbox::{NewOutboxRow, OutboxStore};
        use crate::journal::outbox_message::OutboxMessageKind;

        // Regression: when a worker commits events [3, 4, 5] in one
        // batch and the resulting outbox row pins the safety bound,
        // the janitor must not delete events 3 or 4 while the row is
        // unpublished.  The outbox row is seeded with sequence=3 —
        // the FIRST event of the batch — matching the contract
        // implemented in `insert_thread_events_outbox_row_tx`.
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        for i in 0..=2u64 {
            let name = format!("m{i}");
            events
                .commit_event(&thread_a(), AgentEvent::text(&name, "prefix"), t0())
                .await?;
        }
        // Batch [3, 4, 5] committed atomically (all same timestamp).
        for i in 3..=5u64 {
            let name = format!("m{i}");
            events
                .commit_event(&thread_a(), AgentEvent::text(&name, "batch"), t0())
                .await?;
        }

        // Advance past the prefix first so the safety bound is what
        // keeps the pending batch alive.
        retention.advance_floor(&thread_a(), 3, t_plus(1)).await?;

        outbox
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_a(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(3),
                payload_json: serde_json::json!({}),
                max_attempts: 3,
                now: t0(),
            }])
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(7200),
        )
        .await?;

        assert_eq!(
            retention.effective_floor(&thread_a()).await?,
            3,
            "floor must stay at 3; seqs 3..=5 are in the pending batch",
        );
        assert_eq!(report.events_purged, 0);
        assert_eq!(report.floors_advanced, 0);
        Ok(())
    }

    #[tokio::test]
    async fn floor_never_advances_past_live_event_with_earlier_timestamp() -> Result<()> {
        // Regression: non-monotonic `committed_at` (clock skew,
        // concurrent writers) must not trick the janitor into
        // deleting a live event.  Setup: seq 0, 1 expired; seq 2
        // live (within TTL); seq 3 has an *older* timestamp than
        // seq 2 (expired).  `max_sequence_before` returns 3 →
        // candidate_floor would be 4 and destroy seq 2 if we used
        // only `max_sequence_before`.  The min-live-sequence cap
        // pins candidate_floor to 2, keeping seq 2 alive.
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        // TTL = 1h, now = t0() + 2h → cutoff = t0() + 1h.
        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old0"), t0())
            .await?;
        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "old1"), t0())
            .await?;
        // Live: timestamp at cutoff+10 min (well within TTL).
        events
            .commit_event(&thread_a(), AgentEvent::text("m2", "live"), t_plus(4200))
            .await?;
        // Non-monotonic: seq 3 timestamp before seq 2's.
        events
            .commit_event(&thread_a(), AgentEvent::text("m3", "old3"), t0())
            .await?;

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &deps(&events, &retention, &outbox, &checkpoints),
            t_plus(7200),
        )
        .await?;

        assert_eq!(
            retention.effective_floor(&thread_a()).await?,
            2,
            "floor must stop at the first live sequence (2), not seq 4",
        );
        assert_eq!(
            report.events_purged, 2,
            "only seq 0 and 1 may be purged; seq 2 is live",
        );
        Ok(())
    }

    #[tokio::test]
    async fn outbox_blocked_pass1_thread_leaves_pass2_budget_intact() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::outbox::{NewOutboxRow, OutboxStore};
        use crate::journal::outbox_message::OutboxMessageKind;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_foundation::{TokenUsage, llm};

        // Regression: a Pass 1 thread whose floor cannot advance (the
        // outbox pins `safe_floor <= current_floor`) must NOT burn
        // `batch_size` budget or mark itself as swept — otherwise a
        // Pass 2 thread with excess checkpoints gets silently skipped.
        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        // thread_a: expired events + pending outbox row at seq 0
        // (pins safe_floor to 0, which is <= current_floor 0).
        events
            .commit_event(&thread_a(), AgentEvent::text("m0", "old"), t0())
            .await?;
        outbox
            .insert_batch(vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id: thread_a(),
                event_id: Some(uuid::Uuid::now_v7()),
                sequence: Some(0),
                payload_json: serde_json::json!({}),
                max_attempts: 3,
                now: t0(),
            }])
            .await?;

        // thread_b: excess checkpoints (Pass 2 eligible only).
        for i in 1..=3u32 {
            checkpoint_store
                .commit_checkpoint(NewCheckpointParams {
                    kind: CheckpointKind::FullTurn,
                    thread_id: thread_b(),
                    turn_number: i,
                    task_id: AgentTaskId::from_string(format!("task-b-{i}")),
                    messages: vec![llm::Message::user(format!("turn {i}"))],
                    agent_state_snapshot: serde_json::json!({}),
                    turn_usage: TokenUsage::default(),
                    now: t0(),
                })
                .await?;
        }

        let policy = RetentionPolicy {
            event_ttl: Some(std::time::Duration::from_hours(1)),
            checkpoint_max_per_thread: Some(1),
            batch_size: 2,
        };

        let report = run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoint_store),
            t_plus(7200),
        )
        .await?;

        assert_eq!(
            report.floors_advanced, 0,
            "thread_a's floor must stay pinned by the outbox row",
        );
        assert_eq!(
            report.checkpoints_pruned, 2,
            "Pass 2 must still prune thread_b's excess checkpoints",
        );
        assert_eq!(
            report.threads_scanned, 1,
            "only thread_b counts as scanned (thread_a did no useful work)",
        );
        let remaining = checkpoint_store.list_by_thread(&thread_b()).await?;
        assert_eq!(remaining.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn janitor_prunes_checkpoints_without_event_ttl() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_foundation::{TokenUsage, llm};

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        for i in 1..=5u32 {
            checkpoint_store
                .commit_checkpoint(NewCheckpointParams {
                    kind: CheckpointKind::FullTurn,
                    thread_id: thread_a(),
                    turn_number: i,
                    task_id: AgentTaskId::from_string(format!("task-{i}")),
                    messages: vec![llm::Message::user(format!("turn {i}"))],
                    agent_state_snapshot: serde_json::json!({}),
                    turn_usage: TokenUsage::default(),
                    now: t0(),
                })
                .await?;
        }

        let policy = RetentionPolicy {
            event_ttl: None,
            checkpoint_max_per_thread: Some(2),
            batch_size: 100,
        };

        let report = run_janitor_cycle(
            &policy,
            &deps(&events, &retention, &outbox, &checkpoint_store),
            t_plus(100),
        )
        .await?;

        assert_eq!(report.checkpoints_pruned, 3);
        assert_eq!(report.events_purged, 0);
        assert_eq!(report.floors_advanced, 0);

        let remaining = checkpoint_store.list_by_thread(&thread_a()).await?;
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].turn_number, 4);
        assert_eq!(remaining[1].turn_number, 5);
        Ok(())
    }

    // Wrapper that injects an outbox row during the janitor's final
    // event-sequence read (`min_sequence_at_or_after`), simulating a
    // worker that commits a batch plus its Pending outbox row mid-cycle.
    // Hoisted to module scope so each method stays small.
    struct RacingEventRepo {
        inner: std::sync::Arc<InMemoryEventRepository>,
        outbox: std::sync::Arc<InMemoryOutboxStore>,
        injected: std::sync::atomic::AtomicBool,
    }

    #[async_trait::async_trait]
    impl EventRepository for RacingEventRepo {
        fn atomic_event_outbox_committer(
            &self,
        ) -> Option<&dyn crate::journal::event_outbox_transaction::AtomicEventOutboxCommitter>
        {
            None
        }

        async fn commit_event(
            &self,
            thread_id: &ThreadId,
            event: AgentEvent,
            now: OffsetDateTime,
        ) -> Result<crate::journal::committed_event::CommittedEvent> {
            self.inner.commit_event(thread_id, event, now).await
        }

        async fn commit_event_batch(
            &self,
            thread_id: &ThreadId,
            events: Vec<AgentEvent>,
            now: OffsetDateTime,
        ) -> Result<Vec<crate::journal::committed_event::CommittedEvent>> {
            self.inner.commit_event_batch(thread_id, events, now).await
        }

        async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
            self.inner.next_sequence(thread_id).await
        }

        async fn get_events(
            &self,
            thread_id: &ThreadId,
        ) -> Result<Vec<crate::journal::committed_event::CommittedEvent>> {
            self.inner.get_events(thread_id).await
        }

        async fn get_events_in_range(
            &self,
            thread_id: &ThreadId,
            after_sequence: u64,
            up_to_sequence: u64,
        ) -> Result<Vec<crate::journal::committed_event::CommittedEvent>> {
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
            use crate::journal::outbox::{NewOutboxRow, OutboxStore};
            use crate::journal::outbox_message::{
                OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
            };
            use std::sync::atomic::Ordering;
            if !self.injected.swap(true, Ordering::SeqCst) {
                let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                    thread_id: thread_id.clone(),
                    last_sequence: 1,
                })
                .to_payload_json()?;
                self.outbox
                    .insert_batch(vec![NewOutboxRow {
                        kind: OutboxMessageKind::ThreadEventsAvailable,
                        thread_id: thread_id.clone(),
                        event_id: Some(uuid::Uuid::now_v7()),
                        sequence: Some(1),
                        payload_json: payload,
                        max_attempts: 3,
                        now: t0(),
                    }])
                    .await?;
            }
            self.inner.min_sequence_at_or_after(thread_id, cutoff).await
        }
    }

    // Regression for the outbox-bound TOCTOU: the safety bound is read
    // last, so a row that becomes visible *during* the event reads still
    // pins the floor.  Under the old ordering (bound read first) the
    // injected row would be invisible and the floor would advance over
    // it, purging an event still referenced by an unpublished advisory.
    #[tokio::test]
    async fn outbox_bound_inserted_after_event_reads_still_pins_floor() -> Result<()> {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let inner = Arc::new(InMemoryEventRepository::new());
        let outbox = Arc::new(InMemoryOutboxStore::new());
        let retention = InMemoryRetentionStore::new();
        let checkpoints = InMemoryCheckpointStore::new();

        // Three expired events at t0 (sequences 0, 1, 2).
        for i in 0..3u64 {
            inner
                .commit_event(&thread_a(), AgentEvent::text(format!("m{i}"), "old"), t0())
                .await?;
        }

        let events = RacingEventRepo {
            inner: Arc::clone(&inner),
            outbox: Arc::clone(&outbox),
            injected: AtomicBool::new(false),
        };

        let report = run_janitor_cycle(
            &policy_with_ttl(3600),
            &RetentionJanitorDeps {
                event_repo: &events,
                retention_store: &retention,
                outbox_store: outbox.as_ref(),
                checkpoint_store: &checkpoints,
            },
            t_plus(7200),
        )
        .await?;

        // The row injected mid-cycle (seq 1) must pin the floor at 1.
        assert_eq!(retention.effective_floor(&thread_a()).await?, 1);
        assert_eq!(report.events_purged, 1);
        Ok(())
    }
}
