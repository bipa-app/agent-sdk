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
use std::sync::atomic::{AtomicI64, Ordering};

use time::OffsetDateTime;

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
}
