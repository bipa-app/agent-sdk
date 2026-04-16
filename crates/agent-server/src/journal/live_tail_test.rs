//! Phase 6.4 live tail hub integration and regression tests.
//!
//! These tests verify the live tail hub's interaction with the durable
//! event repository and the replay API.  They cover scenarios that the
//! unit tests in [`live_tail`](super::live_tail) do not:
//!
//! - Slow consumer reconnecting via durable replay after lag disconnect.
//! - Fast and slow subscribers coexisting on the same thread without
//!   interference.
//! - Mixed single and batch commits flowing through the live tail.
//! - Subscriber reconnect using [`stream_events`] for full recovery.

use super::committed_event::CommittedEvent;
use super::event_notifier::EventNotifier;
use super::event_repository::{EventRepository, InMemoryEventRepository};
use super::event_stream::{StreamEvent, stream_events};
use super::live_tail::{LiveTailConfig, LiveTailEvent, LiveTailHub};
use agent_sdk_core::ThreadId;
use agent_sdk_core::events::AgentEvent;
use anyhow::{Context, Result};
use std::time::Duration;
use time::OffsetDateTime;

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + time::Duration::seconds(secs)
}

fn thread_a() -> ThreadId {
    ThreadId::from_string("t-livetail-int-a")
}

/// Helper: commit an event, publish to the live tail hub, and return
/// the committed event.
async fn commit_and_publish(
    repo: &InMemoryEventRepository,
    hub: &LiveTailHub,
    thread_id: &ThreadId,
    time_offset: i64,
) -> Result<CommittedEvent> {
    let event = repo
        .commit_event(
            thread_id,
            AgentEvent::text(format!("m{time_offset}"), format!("payload-{time_offset}")),
            t_plus(time_offset),
        )
        .await?;
    hub.publish(std::slice::from_ref(&event));
    Ok(event)
}

// ─────────────────────────────────────────────────────────────────────
// Slow consumer recovers via durable replay
// ─────────────────────────────────────────────────────────────────────

/// A slow subscriber that falls behind the live tail hub can reconnect
/// via the durable event repository and recover exactly the missed
/// committed envelopes.
#[tokio::test]
async fn slow_consumer_recovers_via_durable_replay() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 3,
        lag_grace_period: Duration::ZERO,
    });

    let mut rx = hub.subscribe(&thread_a());

    // Commit and publish 3 events (fills subscriber's buffer).
    for i in 0..3 {
        commit_and_publish(&repo, &hub, &thread_a(), i).await?;
    }

    // Next commit triggers lag (buffer full, subscriber hasn't read).
    commit_and_publish(&repo, &hub, &thread_a(), 3).await?;

    // Another publish disconnects (grace period = 0).
    commit_and_publish(&repo, &hub, &thread_a(), 4).await?;

    // Commit a few more events after the disconnect.
    commit_and_publish(&repo, &hub, &thread_a(), 5).await?;
    commit_and_publish(&repo, &hub, &thread_a(), 6).await?;

    // Drain the receiver: 3 buffered events + ReplayRequired.
    let mut delivered_seqs = Vec::new();
    let last_delivered = loop {
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => delivered_seqs.push(e.sequence),
            Some(LiveTailEvent::ReplayRequired {
                last_delivered_sequence,
            }) => break last_delivered_sequence,
            None => panic!("expected ReplayRequired before channel close"),
        }
    };

    assert_eq!(delivered_seqs, vec![0, 1, 2]);
    assert_eq!(last_delivered, Some(2));

    // ── Reconnect via durable replay ───────────────────────────────
    // The subscriber uses `last_delivered_sequence` to fetch exactly
    // the missed events from the event repository.
    //
    // `get_events_in_range` uses a strictly-greater-than lower bound,
    // so `None` (no events delivered) must use `get_events` instead
    // to avoid skipping sequence 0.
    let watermark = repo.next_sequence(&thread_a()).await? - 1;
    let missed = match last_delivered {
        None => {
            let all = repo.get_events(&thread_a()).await?;
            all.into_iter()
                .filter(|e| e.sequence <= watermark)
                .collect()
        }
        Some(seq) => {
            repo.get_events_in_range(&thread_a(), seq, watermark)
                .await?
        }
    };

    let missed_seqs: Vec<u64> = missed.iter().map(|e| e.sequence).collect();
    assert_eq!(missed_seqs, vec![3, 4, 5, 6]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Reconnect via stream_events recovers cleanly
// ─────────────────────────────────────────────────────────────────────

/// After a lag disconnect, a subscriber can reconnect using
/// [`stream_events`] with `after_sequence` set to the last-delivered
/// sequence, seamlessly resuming from durable replay into the live
/// tail (via `EventNotifier`).
#[tokio::test]
async fn reconnect_via_stream_events_recovers_cleanly() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let notifier = EventNotifier::new();
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 3,
        lag_grace_period: Duration::ZERO,
    });

    let mut rx = hub.subscribe(&thread_a());

    // Commit 3 events (fills buffer).
    for i in 0..3 {
        let e = commit_and_publish(&repo, &hub, &thread_a(), i).await?;
        notifier.notify(std::slice::from_ref(&e));
    }

    // Trigger lag + disconnect.
    let e3 = commit_and_publish(&repo, &hub, &thread_a(), 3).await?;
    notifier.notify(std::slice::from_ref(&e3));
    let e4 = commit_and_publish(&repo, &hub, &thread_a(), 4).await?;
    notifier.notify(std::slice::from_ref(&e4));

    // Drain receiver to get ReplayRequired.
    let last_delivered = loop {
        match rx.recv().await {
            Some(LiveTailEvent::Event(_)) => {}
            Some(LiveTailEvent::ReplayRequired {
                last_delivered_sequence,
            }) => break last_delivered_sequence,
            None => panic!("expected ReplayRequired"),
        }
    };
    assert_eq!(last_delivered, Some(2));

    // Reconnect using stream_events.
    let mut stream = stream_events(&thread_a(), last_delivered, &repo, &notifier).await?;

    // Should get events 3 and 4 from durable replay.
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 3),
        other => panic!("expected Event(3), got {other:?}"),
    }
    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 4),
        other => panic!("expected Event(4), got {other:?}"),
    }

    // New events committed after reconnect arrive via live tail.
    let e5 = repo
        .commit_event(&thread_a(), AgentEvent::text("m5", "payload-5"), t_plus(5))
        .await?;
    notifier.notify(std::slice::from_ref(&e5));

    match stream.next().await {
        Some(StreamEvent::Event(e)) => assert_eq!(e.sequence, 5),
        other => panic!("expected Event(5), got {other:?}"),
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Fast and slow subscribers coexist
// ─────────────────────────────────────────────────────────────────────

/// A fast subscriber continues to receive events normally even when a
/// slow subscriber on the same thread is lagging and eventually
/// disconnected.
#[tokio::test]
async fn fast_subscriber_unaffected_by_slow_disconnect() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 3,
        lag_grace_period: Duration::ZERO,
    });

    let mut slow_rx = hub.subscribe(&thread_a()); // Never reads.
    let mut fast_rx = hub.subscribe(&thread_a());

    assert_eq!(hub.subscriber_count(&thread_a()), 2);

    // Commit 5 events.  Fast subscriber drains between each publish.
    for i in 0i64..5 {
        commit_and_publish(&repo, &hub, &thread_a(), i).await?;
        let expected_seq = u64::try_from(i).context("loop index")?;
        match fast_rx.recv().await {
            Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, expected_seq),
            other => panic!("fast subscriber: expected Event({i}), got {other:?}"),
        }
    }

    // By now, slow subscriber has lagged and been disconnected.
    // The fast subscriber's count should still be 1.
    // (Trigger a publish to force cleanup.)
    commit_and_publish(&repo, &hub, &thread_a(), 5).await?;
    match fast_rx.recv().await {
        Some(LiveTailEvent::Event(e)) => assert_eq!(e.sequence, 5),
        other => panic!("fast subscriber: expected Event(5), got {other:?}"),
    }

    // Slow subscriber gets ReplayRequired.
    let mut slow_seqs = Vec::new();
    loop {
        match slow_rx.recv().await {
            Some(LiveTailEvent::Event(e)) => slow_seqs.push(e.sequence),
            Some(LiveTailEvent::ReplayRequired {
                last_delivered_sequence,
            }) => {
                assert_eq!(last_delivered_sequence, Some(2));
                break;
            }
            None => panic!("expected ReplayRequired for slow subscriber"),
        }
    }
    // Slow subscriber buffered 3 events before lag.
    assert_eq!(slow_seqs, vec![0, 1, 2]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Batch commit flows through live tail
// ─────────────────────────────────────────────────────────────────────

/// Batch-committed events published through the live tail hub are
/// delivered in order and maintain sequence continuity.
#[tokio::test]
async fn batch_commit_flows_through_live_tail() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let hub = LiveTailHub::new();
    let mut rx = hub.subscribe(&thread_a());

    // Single commit.
    let e0 = commit_and_publish(&repo, &hub, &thread_a(), 0).await?;
    let _ = e0;

    // Batch commit of 3.
    let batch = repo
        .commit_event_batch(
            &thread_a(),
            vec![
                AgentEvent::text("m1", "a"),
                AgentEvent::text("m2", "b"),
                AgentEvent::text("m3", "c"),
            ],
            t_plus(1),
        )
        .await?;
    hub.publish(&batch);

    // Single commit.
    commit_and_publish(&repo, &hub, &thread_a(), 4).await?;

    // Drain all 5 events.
    let mut seqs = Vec::new();
    for _ in 0..5 {
        match rx.recv().await {
            Some(LiveTailEvent::Event(e)) => seqs.push(e.sequence),
            other => panic!("expected Event, got {other:?}"),
        }
    }
    assert_eq!(seqs, vec![0, 1, 2, 3, 4]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Workers never blocked during overflow
// ─────────────────────────────────────────────────────────────────────

/// Even with many slow subscribers, committing and publishing events
/// completes without blocking.  This validates the acceptance criterion
/// that slow subscribers do not stall task execution.
#[tokio::test]
async fn workers_never_blocked_during_overflow() -> Result<()> {
    let repo = InMemoryEventRepository::new();
    let hub = LiveTailHub::with_config(LiveTailConfig {
        buffer_capacity: 5,
        lag_grace_period: Duration::from_mins(1),
    });

    // Create 10 subscribers that never read.
    let _subscribers: Vec<_> = (0..10).map(|_| hub.subscribe(&thread_a())).collect();

    let start = std::time::Instant::now();
    for i in 0..50 {
        commit_and_publish(&repo, &hub, &thread_a(), i).await?;
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "commit+publish took too long with slow subscribers: {elapsed:?}",
    );

    Ok(())
}
