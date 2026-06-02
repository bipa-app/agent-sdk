//! Phase 6.3 replay correctness regression tests.
//!
//! These tests verify the replay API and race-free handoff across
//! scenarios that the unit tests in `event_stream` and `event_notifier`
//! do not cover individually:
//!
//! - Concurrent event production during active replay consumption.
//! - Watermark capture timing under interleaved commits.
//! - Multiple reconnects to the same thread at different offsets.
//! - Integration with `EventRepository` commit paths (single + batch).
//! - Unpublished events never leak to subscribers.

use super::committed_event::CommittedEvent;
use super::event_notifier::EventNotifier;
use super::event_repository::{EventRepository, InMemoryEventRepository};
use super::event_stream::{StreamEvent, stream_events};
use super::retention::InMemoryRetentionStore;
use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use anyhow::Result;
use time::{Duration, OffsetDateTime};

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("t-replay-a")
}

/// Helper: commit an event and notify in one step.
async fn commit_and_notify(
    repo: &InMemoryEventRepository,
    notifier: &EventNotifier,
    thread_id: &ThreadId,
    seq_label: &str,
    time_offset: i64,
) -> Result<CommittedEvent> {
    let event = repo
        .commit_event(
            thread_id,
            AgentEvent::text(seq_label, seq_label),
            t_plus(time_offset),
        )
        .await?;
    notifier.notify(std::slice::from_ref(&event));
    Ok(event)
}

/// Helper: commit a batch and notify in one step.
async fn commit_batch_and_notify(
    repo: &InMemoryEventRepository,
    notifier: &EventNotifier,
    thread_id: &ThreadId,
    count: usize,
    time_offset: i64,
) -> Result<Vec<CommittedEvent>> {
    let events: Vec<AgentEvent> = (0..count)
        .map(|i| AgentEvent::text(format!("batch-{i}"), format!("batch-{i}")))
        .collect();
    let committed = repo
        .commit_event_batch(thread_id, events, t_plus(time_offset))
        .await?;
    notifier.notify(&committed);
    Ok(committed)
}

/// Helper: drain `count` events from a stream and collect sequences.
async fn drain_sequences(stream: &mut super::event_stream::EventStream, count: usize) -> Vec<u64> {
    let mut seqs = Vec::with_capacity(count);
    for _ in 0..count {
        match stream.next().await {
            Some(StreamEvent::Event(e)) => seqs.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    seqs
}

// ─────────────────────────────────────────────────────────────────────
// Concurrent production during active consumption
// ─────────────────────────────────────────────────────────────────────

/// A producer commits events while a consumer is actively draining the
/// replay buffer. The consumer must see all events in order with no
/// gaps.
#[tokio::test]
async fn producer_during_replay_drain_yields_contiguous_sequence() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Pre-commit 10 events.
    for i in 0..10i64 {
        repo.commit_event(
            &thread_a(),
            AgentEvent::text(format!("m{i}"), "x"),
            t_plus(i),
        )
        .await?;
    }

    // Open stream from the beginning.
    let mut stream = stream_events(
        &thread_a(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Drain the first 5 from replay.
    let first_5 = drain_sequences(&mut stream, 5).await;
    assert_eq!(first_5, vec![0, 1, 2, 3, 4]);

    // While draining, the producer commits 5 more.
    for i in 10..15i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // Drain the remaining replay (5..9) and then live (10..14).
    let rest = drain_sequences(&mut stream, 10).await;
    assert_eq!(rest, vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Interleaved batch commits during handoff
// ─────────────────────────────────────────────────────────────────────

/// Batch commits that straddle the watermark boundary are handled
/// correctly: events within the replay window come from durable
/// replay, and events beyond the watermark arrive through the live
/// tail.
#[tokio::test]
async fn batch_commit_straddles_watermark() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Pre-commit 3 events.
    for i in 0..3i64 {
        repo.commit_event(
            &thread_a(),
            AgentEvent::text(format!("m{i}"), "x"),
            t_plus(i),
        )
        .await?;
    }

    // Open stream (watermark captures seq 2 as high-water).
    let mut stream = stream_events(
        &thread_a(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Now batch-commit 4 more events (sequences 3..6).
    commit_batch_and_notify(&repo, &notifier, &thread_a(), 4, 3).await?;

    // Drain all 7 events.
    let all = drain_sequences(&mut stream, 7).await;
    assert_eq!(all, vec![0, 1, 2, 3, 4, 5, 6]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Multiple reconnects at different offsets
// ─────────────────────────────────────────────────────────────────────

/// Two clients reconnect to the same thread at different offsets and
/// each receives the correct suffix.
#[tokio::test]
async fn multiple_reconnects_different_offsets() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Commit 8 events.
    for i in 0..8i64 {
        repo.commit_event(
            &thread_a(),
            AgentEvent::text(format!("m{i}"), "x"),
            t_plus(i),
        )
        .await?;
    }

    // Client A reconnects after seq 2.
    let mut stream_a = stream_events(
        &thread_a(),
        Some(2),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    // Client B reconnects after seq 5.
    let mut stream_b = stream_events(
        &thread_a(),
        Some(5),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    let a_seqs = drain_sequences(&mut stream_a, 5).await;
    let b_seqs = drain_sequences(&mut stream_b, 2).await;

    assert_eq!(a_seqs, vec![3, 4, 5, 6, 7]);
    assert_eq!(b_seqs, vec![6, 7]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Unpublished events never leak
// ─────────────────────────────────────────────────────────────────────

/// Events committed to the repository but NOT notified are never
/// delivered through the live tail. Only durable replay surfaces
/// them on reconnect.
#[tokio::test]
async fn unnotified_events_invisible_in_live_tail() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Commit 2 events (notified).
    commit_and_notify(&repo, &notifier, &thread_a(), "m0", 0).await?;
    commit_and_notify(&repo, &notifier, &thread_a(), "m1", 1).await?;

    // Open stream after seq 1 (fully caught up).
    let mut stream = stream_events(
        &thread_a(),
        Some(1),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Commit event 2 WITHOUT notifying (simulates a bug or a
    // separate code path that forgot to notify).
    repo.commit_event(&thread_a(), AgentEvent::text("m2", "sneaky"), t_plus(2))
        .await?;

    // Commit event 3 WITH notification.
    commit_and_notify(&repo, &notifier, &thread_a(), "m3", 3).await?;

    // The live tail should deliver seq 3, skipping seq 2 because
    // it was never notified. A reconnecting client would pick up
    // seq 2 on the next replay.
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 3),
        other => panic!("expected Event(seq=3), got {other:?}"),
    }

    // Reconnect to verify seq 2 is available through durable replay.
    let mut reconnected = stream_events(
        &thread_a(),
        Some(1),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let seqs = drain_sequences(&mut reconnected, 2).await;
    assert_eq!(seqs, vec![2, 3]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Empty thread replay
// ─────────────────────────────────────────────────────────────────────

/// Streaming from an empty thread goes directly to live tail.
#[tokio::test]
async fn empty_thread_replay_goes_to_live() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    let mut stream = stream_events(
        &thread_a(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Commit the first event.
    commit_and_notify(&repo, &notifier, &thread_a(), "m0", 0).await?;

    match stream.next().await {
        Some(StreamEvent::Event(e)) => {
            assert_eq!(e.sequence, 0);
            assert_eq!(e.thread_id, thread_a());
        }
        other => panic!("expected Event(seq=0), got {other:?}"),
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Sequence continuity across single + batch commits
// ─────────────────────────────────────────────────────────────────────

/// Mixed single and batch commits produce a contiguous sequence stream
/// through the replay API.
#[tokio::test]
async fn mixed_single_and_batch_commits_contiguous() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Single commit.
    commit_and_notify(&repo, &notifier, &thread_a(), "s0", 0).await?;

    // Batch of 3.
    commit_batch_and_notify(&repo, &notifier, &thread_a(), 3, 1).await?;

    // Single commit.
    commit_and_notify(&repo, &notifier, &thread_a(), "s4", 4).await?;

    // Batch of 2.
    commit_batch_and_notify(&repo, &notifier, &thread_a(), 2, 5).await?;

    // Replay from start.
    let mut stream = stream_events(
        &thread_a(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let seqs = drain_sequences(&mut stream, 7).await;
    assert_eq!(seqs, vec![0, 1, 2, 3, 4, 5, 6]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Handoff overlap dedup under heavy concurrent writes
// ─────────────────────────────────────────────────────────────────────

/// Simulates the worst-case handoff race: events are committed
/// between subscribe (step 1) and watermark capture (step 2) by
/// directly writing to the repo+notifier before the stream is
/// constructed. We verify no duplicates.
#[tokio::test]
async fn handoff_overlap_with_many_concurrent_writes() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Pre-commit 5 events.
    for i in 0..5i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // Open stream from after seq 3. Watermark will be at seq 4.
    let mut stream = stream_events(
        &thread_a(),
        Some(3),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Concurrently commit 5 more events while replay hasn't drained.
    for i in 5..10i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // Drain all events from the stream.
    let all = drain_sequences(&mut stream, 6).await;
    // seq 4 from replay, seq 5..9 from live.
    assert_eq!(all, vec![4, 5, 6, 7, 8, 9]);

    // Verify no duplicates.
    let unique: std::collections::HashSet<u64> = all.iter().copied().collect();
    assert_eq!(all.len(), unique.len(), "duplicate sequences detected");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// after_sequence beyond committed range
// ─────────────────────────────────────────────────────────────────────

/// When `after_sequence` is beyond the last committed event, the replay
/// buffer is empty and the stream goes directly to live.
#[tokio::test]
async fn after_sequence_beyond_committed_range() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();

    // Commit 3 events (seq 0, 1, 2).
    for i in 0..3i64 {
        repo.commit_event(
            &thread_a(),
            AgentEvent::text(format!("m{i}"), "x"),
            t_plus(i),
        )
        .await?;
    }

    // after_sequence = 10 (beyond anything committed).
    let mut stream = stream_events(
        &thread_a(),
        Some(10),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Commit a new event.
    commit_and_notify(&repo, &notifier, &thread_a(), "m3", 3).await?;

    // The live tail should NOT deliver this because its sequence (3)
    // is <= after_sequence (10). Only events with seq > 10 should
    // arrive.
    commit_and_notify(&repo, &notifier, &thread_a(), "m4", 4).await?;

    // Neither seq 3 nor seq 4 should be delivered because both are
    // <= 10. We need seq 11+ to come through. Let's verify by
    // committing enough events to get past seq 10.
    for i in 5..12i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // Now seq 11 should arrive (it's the first with sequence > 10).
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 11),
        other => panic!("expected Event(seq=11), got {other:?}"),
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Lagged subscriber recovery
// ─────────────────────────────────────────────────────────────────────

/// A slow consumer that falls behind the broadcast buffer gets a
/// Lagged notification and can reconnect via a fresh stream.
#[tokio::test]
async fn lagged_subscriber_can_reconnect() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    // Small buffer to force lagging.
    let notifier = EventNotifier::with_capacity(4);

    // Pre-commit 2 events.
    for i in 0..2i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    let mut stream = stream_events(
        &thread_a(),
        None,
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;

    // Drain replay.
    let replay_seqs = drain_sequences(&mut stream, 2).await;
    assert_eq!(replay_seqs, vec![0, 1]);

    // Flood the live tail buffer.
    for i in 2..20i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // The next recv should report lagged.
    match stream.next().await {
        Some(StreamEvent::Lagged { skipped }) => {
            assert!(skipped > 0, "expected positive skip count");
        }
        other => panic!("expected Lagged, got {other:?}"),
    }

    // Reconnect from the last known good sequence (1) and verify
    // we get all events.
    let mut reconnected = stream_events(
        &thread_a(),
        Some(1),
        &repo,
        &InMemoryRetentionStore::new(),
        &notifier,
    )
    .await?;
    let seqs = drain_sequences(&mut reconnected, 18).await;
    assert_eq!(seqs, (2..20).collect::<Vec<u64>>());

    Ok(())
}
