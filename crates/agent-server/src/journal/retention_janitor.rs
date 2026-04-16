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
    /// Maximum threads processed per janitor cycle.
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
    pub threads_scanned: u32,
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
            .threads_with_events_before(cutoff, policy.batch_size)
            .await
            .context("list threads needing retention")?;

        for thread_id in &threads {
            report.threads_scanned += 1;
            event_swept_threads.insert(thread_id.clone());

            // ── Compute safe floor ──────────────────────────────
            let outbox_bound = deps
                .outbox_store
                .min_unpublished_sequence(thread_id)
                .await
                .context("read outbox safety bound")?;

            let candidate_seq = deps
                .event_repo
                .max_sequence_before(thread_id, cutoff)
                .await
                .context("read max sequence before cutoff")?;

            let Some(max_seq) = candidate_seq else {
                continue;
            };

            let candidate_floor = max_seq + 1;

            let safe_floor =
                outbox_bound.map_or(candidate_floor, |min_unpub| candidate_floor.min(min_unpub));

            let current_floor = deps
                .retention_store
                .effective_floor(thread_id)
                .await
                .context("read current retention floor")?;

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
        }
    }

    // ── Pass 2: standalone checkpoint pruning ───────────────────
    //
    // Threads that were NOT already processed in pass 1 but have
    // more checkpoints than the configured limit.  This ensures
    // checkpoint_max_per_thread works independently of event_ttl.
    if let Some(limit) = policy.checkpoint_max_per_thread {
        let keep = limit.max(1);
        let threads = deps
            .checkpoint_store
            .threads_exceeding_checkpoint_count(keep, policy.batch_size)
            .await
            .context("list threads with excess checkpoints")?;

        for thread_id in &threads {
            if event_swept_threads.contains(thread_id) {
                continue;
            }
            report.threads_scanned += 1;
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
    use crate::journal::checkpoint_store::InMemoryCheckpointStore;
    use crate::journal::event_repository::InMemoryEventRepository;
    use crate::journal::outbox::InMemoryOutboxStore;
    use crate::journal::retention::InMemoryRetentionStore;
    use agent_sdk_core::ThreadId;
    use agent_sdk_core::events::AgentEvent;
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
        use agent_sdk_core::{TokenUsage, llm};

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
            event_ttl: Some(std::time::Duration::from_secs(3600)),
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
        use agent_sdk_core::{TokenUsage, llm};

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        events
            .commit_event(&thread_a(), AgentEvent::text("m1", "old"), t0())
            .await?;
        checkpoint_store
            .commit_checkpoint(NewCheckpointParams {
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
            event_ttl: Some(std::time::Duration::from_secs(3600)),
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
    async fn janitor_prunes_checkpoints_without_event_ttl() -> Result<()> {
        use crate::journal::checkpoint::NewCheckpointParams;
        use crate::journal::checkpoint_store::CheckpointStore;
        use crate::journal::task::AgentTaskId;
        use agent_sdk_core::{TokenUsage, llm};

        let events = InMemoryEventRepository::new();
        let retention = InMemoryRetentionStore::new();
        let outbox = InMemoryOutboxStore::new();
        let checkpoint_store = InMemoryCheckpointStore::new();

        for i in 1..=5u32 {
            checkpoint_store
                .commit_checkpoint(NewCheckpointParams {
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
}
