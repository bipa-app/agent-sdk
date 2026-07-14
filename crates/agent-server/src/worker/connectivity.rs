//! In-memory registry of root turns parked in an offline connectivity wait.
//!
//! When a streaming LLM call fails with a connectivity-class error
//! ([`crate::worker::root_turn`]'s `wait_out_connectivity_loss`), the worker
//! parks on free reachability probes until the provider answers again. That
//! park produces **no provider frames and no journal commits** — to any
//! outside observer measuring silence (a host's stall sweep, a watchdog, a
//! reconnecting frontend) it is indistinguishable from a wedged task.
//!
//! This registry is the runtime's own, authoritative answer: a thread has an
//! entry **exactly while its worker is inside the reachability wait loop**,
//! inserted before the first probe and removed by RAII guard when the wait
//! ends — through recovery, cancellation, an error, or the task future being
//! dropped outright. Because the wait can only exist inside a live worker, a
//! process restart empties the registry *and* kills every wait it described;
//! recovery re-enters the wait and re-registers. Nothing about this state
//! needs to be durable.
//!
//! Hosts hand the registry to their sweeps/watchdogs (spare a silent thread
//! that is deliberately waiting) and to their status RPCs (render "waiting
//! for connectivity" to a client that attached mid-outage). Keyed by
//! [`ThreadId`], so a subagent child's wait is visible under the child
//! thread its supervisor already tracks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use agent_sdk_foundation::ThreadId;

/// One parked wait, as observers see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectivityWait {
    /// Journal sequence of the `AutoRetryStart` envelope that opened the
    /// current connectivity streak (`0` when that commit failed). Lets
    /// stream-following clients order a registry snapshot against live
    /// journal events they have already applied.
    pub sequence: u64,
    /// Human-readable reason from the failure that entered this wait.
    pub error_message: String,
}

/// Shared handle to the live wait set. Cheap to clone — every clone
/// observes the same underlying map, so the host runtime, its sweeps, and
/// its RPC layer can each hold one.
#[derive(Clone, Debug, Default)]
pub struct ConnectivityWaitRegistry {
    inner: Arc<Mutex<HashMap<ThreadId, ConnectivityWait>>>,
}

impl ConnectivityWaitRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `thread_id` as parked in a connectivity wait until the returned
    /// guard drops. Guard-scoped on purpose: every exit from the wait —
    /// recovery, cancellation, error propagation, or the whole task future
    /// being dropped by the host's abort grace — runs the removal, so a
    /// stale "waiting" entry for a dead worker is unrepresentable.
    pub fn enter(
        &self,
        thread_id: ThreadId,
        sequence: u64,
        error_message: String,
    ) -> ConnectivityWaitGuard {
        self.lock().insert(
            thread_id.clone(),
            ConnectivityWait {
                sequence,
                error_message,
            },
        );
        ConnectivityWaitGuard {
            registry: self.clone(),
            thread_id,
        }
    }

    /// The wait `thread_id` is currently parked in, if any.
    #[must_use]
    pub fn get(&self, thread_id: &ThreadId) -> Option<ConnectivityWait> {
        self.lock().get(thread_id).cloned()
    }

    /// `true` while `thread_id`'s worker is parked waiting for
    /// connectivity.
    #[must_use]
    pub fn is_waiting(&self, thread_id: &ThreadId) -> bool {
        self.lock().contains_key(thread_id)
    }

    fn exit(&self, thread_id: &ThreadId) {
        self.lock().remove(thread_id);
    }

    /// A poisoned lock only means another thread panicked mid-operation on
    /// the map; the map itself (insert/remove of `Clone` values) cannot be
    /// left logically torn, so recover the guard rather than propagate.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<ThreadId, ConnectivityWait>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Removes the wait entry on drop — see [`ConnectivityWaitRegistry::enter`].
#[must_use = "the wait entry is removed when this guard drops"]
#[derive(Debug)]
pub struct ConnectivityWaitGuard {
    registry: ConnectivityWaitRegistry,
    thread_id: ThreadId,
}

impl Drop for ConnectivityWaitGuard {
    fn drop(&mut self) {
        self.registry.exit(&self.thread_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    #[test]
    fn entry_lives_exactly_as_long_as_the_guard() {
        let registry = ConnectivityWaitRegistry::new();
        let guard = registry.enter(thread("t1"), 7, "offline".to_owned());
        assert!(registry.is_waiting(&thread("t1")));
        assert_eq!(
            registry.get(&thread("t1")),
            Some(ConnectivityWait {
                sequence: 7,
                error_message: "offline".to_owned(),
            })
        );
        drop(guard);
        assert!(!registry.is_waiting(&thread("t1")));
        assert_eq!(registry.get(&thread("t1")), None);
    }

    #[test]
    fn clones_observe_the_same_map() {
        let registry = ConnectivityWaitRegistry::new();
        let observer = registry.clone();
        let _guard = registry.enter(thread("t1"), 1, "offline".to_owned());
        assert!(observer.is_waiting(&thread("t1")));
    }

    #[test]
    fn dropping_the_owning_future_clears_the_entry() {
        // The host drops execution futures that ignore cancellation past
        // its abort grace; the guard inside the dropped future must still
        // clean up. Model that by dropping a pinned future mid-flight.
        let registry = ConnectivityWaitRegistry::new();
        let waiting = registry.clone();
        let future = async move {
            let _guard = waiting.enter(thread("t1"), 1, "offline".to_owned());
            std::future::pending::<()>().await;
        };
        let mut pinned = Box::pin(future);
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(waker);
        assert!(
            std::future::Future::poll(pinned.as_mut(), &mut context).is_pending(),
            "future must park after entering the wait"
        );
        assert!(registry.is_waiting(&thread("t1")));
        drop(pinned);
        assert!(!registry.is_waiting(&thread("t1")));
    }
}
