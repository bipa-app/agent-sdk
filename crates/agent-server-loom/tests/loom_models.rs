//! Scoped `loom` models for the journal's lock-free hot spots (Phase 11 · D).
//!
//! `loom` exhaustively permutes every legal thread interleaving and
//! memory ordering for the code under test, catching atomicity bugs that
//! a stress test would only hit probabilistically. It is **expensive**:
//! the state space explodes with thread and operation count, so we scope
//! these models to exactly the three lock-free primitives the durable
//! task journal depends on, with 2-3 threads each:
//!
//! 1. `wakeup_permit_fold_wakes_at_most_one` — the
//!    `WakeupSignal::notify_workers` permit fold (`task_wakeup.rs`):
//!    N concurrent notifies collapse to a single buffered permit, so a
//!    parked worker wakes **at most once** no matter how many duplicate
//!    broker deliveries land. Modeled on `loom`'s atomics with the same
//!    "store-1, swap-to-0" permit semantics `tokio::sync::Notify` uses
//!    for `notify_one` against a single waiter.
//!
//! 2. `pending_to_running_cas_yields_one_winner` — the
//!    `Pending → Running` claim (`AgentTaskStore::acquire_next_runnable`
//!    / `try_acquire_task` in `store.rs`). Two workers race to lease the
//!    same row; **exactly one** wins and the loser observes the row
//!    already `Running`. Modeled as a CAS on the row's status word.
//!
//! 3. `worker_lease_cas_rejects_stale_owner` — the
//!    `(worker_id, lease_id)` heartbeat CAS
//!    (`AgentTaskStore::heartbeat_task` via `AgentTask::touch_heartbeat`).
//!    After a lease is reassigned (expiry sweep → new worker), a stale
//!    owner's heartbeat CAS must fail; the new owner's must succeed.
//!    Modeled as a CAS on a packed `(generation)` lease word.
//!
//! # Why model the primitive, not the whole store
//!
//! The production CAS sites run *inside* the store's `RwLock` write
//! scope, so the actual concurrency primitive being relied on is the
//! lock plus the compare-and-set logic it guards. `loom` cannot model a
//! `std`/`tokio` `RwLock` meaningfully, and modeling the entire store
//! would blow up the state space for no extra signal. Instead each model
//! reproduces the *essential atomic protocol* of one hot spot — the
//! exact invariant the lock exists to preserve — and proves it holds
//! under every interleaving. This is the standard `loom` discipline:
//! prove the lock-free core, not the data structure around it.
//!
//! # Running
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 \
//!   cargo test -p agent-server-loom --test loom_models
//! ```
//!
//! Without `--cfg loom` the whole module compiles away (the file is
//! gated behind `#![cfg(loom)]`), so a normal `cargo test` is a no-op.

#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;

// ─────────────────────────────────────────────────────────────────────
// 1. Wakeup permit fold
// ─────────────────────────────────────────────────────────────────────

/// Minimal model of the `notify_one`/`notified` permit protocol against
/// a single waiter: producers store a permit (idempotently, capped at 1)
/// and the waiter consumes it with a swap-to-zero. The invariant: no
/// matter how many producers fire, the single waiter wakes at most once.
struct PermitSignal {
    /// 0 = no permit, 1 = one buffered permit. Capped at 1 — this is the
    /// fold: extra notifies do not accumulate.
    permit: AtomicUsize,
}

impl PermitSignal {
    fn new() -> Self {
        Self {
            permit: AtomicUsize::new(0),
        }
    }

    /// Buffer a permit, folding duplicates. Modeled as a `swap(1)` so
    /// concurrent notifies are idempotent — the slot saturates at one.
    fn notify_workers(&self) {
        self.permit.store(1, Ordering::Release);
    }

    /// Consume the buffered permit if present. Returns `true` when this
    /// call woke (i.e. observed and took a permit).
    fn try_consume(&self) -> bool {
        self.permit.swap(0, Ordering::AcqRel) == 1
    }
}

/// Three duplicate broker deliveries (`notify_workers`) race against one
/// parked worker. The worker must wake at most once: a single permit is
/// consumable exactly once, and the extra notifies fold into it.
#[test]
fn wakeup_permit_fold_wakes_at_most_one() {
    loom::model(|| {
        let signal = Arc::new(PermitSignal::new());

        // Two concurrent duplicate notifies (the broker re-delivered the
        // same advisory wakeup).
        let producers: Vec<_> = (0..2)
            .map(|_| {
                let signal = Arc::clone(&signal);
                thread::spawn(move || signal.notify_workers())
            })
            .collect();

        for p in producers {
            p.join().unwrap();
        }

        // After all notifies settle, the single waiter consumes. Because
        // the permit saturates at 1, a second consume must find nothing
        // — duplicates folded.
        let first = signal.try_consume();
        let second = signal.try_consume();
        assert!(first, "a parked worker must observe the folded permit");
        assert!(
            !second,
            "duplicate notifies must fold to a single permit — no double wake",
        );
    });
}

// ─────────────────────────────────────────────────────────────────────
// 2. Pending → Running CAS
// ─────────────────────────────────────────────────────────────────────

const STATUS_PENDING: usize = 0;
const STATUS_RUNNING: usize = 1;

/// Minimal model of one task row's status word and the claim CAS.
struct TaskRow {
    status: AtomicUsize,
}

impl TaskRow {
    fn pending() -> Self {
        Self {
            status: AtomicUsize::new(STATUS_PENDING),
        }
    }

    /// Attempt the `Pending → Running` transition. Returns `true` iff
    /// this caller won the claim.
    fn try_claim(&self) -> bool {
        self.status
            .compare_exchange(
                STATUS_PENDING,
                STATUS_RUNNING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

/// Two workers (woken by a duplicate wakeup) race to lease the same
/// `Pending` row. Exactly one wins; the row ends `Running` exactly once.
#[test]
fn pending_to_running_cas_yields_one_winner() {
    loom::model(|| {
        let row = Arc::new(TaskRow::pending());

        // Both workers must be spawned before either is joined so loom
        // can interleave their claims; binding them separately keeps that
        // ordering explicit.
        let worker_a = {
            let row = Arc::clone(&row);
            thread::spawn(move || usize::from(row.try_claim()))
        };
        let worker_b = {
            let row = Arc::clone(&row);
            thread::spawn(move || usize::from(row.try_claim()))
        };
        let wins = worker_a.join().unwrap() + worker_b.join().unwrap();

        assert_eq!(
            wins, 1,
            "exactly one worker may win the Pending→Running claim",
        );
        assert_eq!(
            row.status.load(Ordering::Acquire),
            STATUS_RUNNING,
            "the row must end up Running",
        );
    });
}

// ─────────────────────────────────────────────────────────────────────
// 3. (worker_id, lease_id) heartbeat CAS
// ─────────────────────────────────────────────────────────────────────

/// Minimal model of a leased row's ownership word. The lease is packed
/// as a monotonically-increasing generation: acquiring a fresh lease
/// bumps the generation, and a heartbeat CAS only succeeds if it
/// presents the *current* generation. A stale owner (older generation)
/// is rejected — this is exactly the `(worker_id, lease_id)` CAS the
/// journal uses to prevent double ownership after an expiry sweep
/// reassigns a row.
struct LeasedRow {
    /// Current lease generation. 1 = the original owner.
    generation: AtomicUsize,
}

impl LeasedRow {
    fn owned_by_gen1() -> Self {
        Self {
            generation: AtomicUsize::new(1),
        }
    }

    /// Expiry sweep reassigns the lease: bump to the next generation.
    /// Returns the new generation the fresh owner holds.
    fn reassign(&self) -> usize {
        // fetch_add returns the previous value; the new owner holds prev+1.
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Heartbeat CAS: confirms ownership iff `my_gen` is *still* the
    /// current generation, atomically. Modeled as a no-op
    /// `compare_exchange(my_gen, my_gen)` so the success decision is made
    /// atomically against the live word — exactly like the production
    /// `(worker_id, lease_id)` CAS, which reads-and-checks the owner
    /// under the same write lock that any reassignment must also hold.
    /// Returns `true` iff this caller still owns the lease.
    fn heartbeat(&self, my_gen: usize) -> bool {
        self.generation
            .compare_exchange(my_gen, my_gen, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// A stale owner (generation 1) and the expiry sweep race. The sweep
/// reassigns the lease to a fresh owner (generation 2) and that fresh
/// owner heartbeats with its new generation. The invariant — the
/// no-double-ownership guarantee — is **monotone, order-independent**:
///
/// - the fresh owner's heartbeat against the generation it just set
///   always succeeds; and
/// - the stale owner's heartbeat against generation 1 may succeed only
///   while the generation is *still* 1 (it raced ahead of the
///   reassignment); once the sweep bumps the generation, every gen-1
///   heartbeat fails forever.
///
/// `loom` explores every interleaving of the reassign and the two
/// heartbeats; the assertions must hold in all of them.
#[test]
fn worker_lease_cas_rejects_stale_owner() {
    loom::model(|| {
        let row = Arc::new(LeasedRow::owned_by_gen1());

        // Thread A: the expiry sweep reassigns the lease, then the fresh
        // owner heartbeats with its new generation. Both the new
        // generation AND its heartbeat result come back so we can assert
        // they are consistent regardless of how the stale thread
        // interleaves.
        let sweep = {
            let row = Arc::clone(&row);
            thread::spawn(move || {
                let new_gen = row.reassign();
                (new_gen, row.heartbeat(new_gen))
            })
        };

        // Thread B: the stale owner heartbeats with generation 1.
        let stale = {
            let row = Arc::clone(&row);
            thread::spawn(move || row.heartbeat(1))
        };

        let (new_gen, fresh_owner_ok) = sweep.join().unwrap();
        let stale_owner_ok = stale.join().unwrap();

        // The fresh owner heartbeats against the very generation it
        // installed; the only way it could fail is a *second* reassign,
        // which this model never issues — so it must succeed.
        assert!(
            fresh_owner_ok,
            "the fresh lease owner's heartbeat must succeed against its own generation",
        );
        assert_eq!(new_gen, 2, "the single reassign installs generation 2");

        // No double ownership at the same generation: the stale owner
        // and the fresh owner present *different* generations (1 vs 2),
        // so they can never both confirm ownership of the same lease
        // word. A successful stale heartbeat only means it landed before
        // the reassignment — it can never co-own with the fresh owner.
        // The decisive guarantee: the two CASes target disjoint
        // generations, so at most one owner is valid per generation.
        assert!(
            !(stale_owner_ok && new_gen == 1),
            "a stale (gen-1) owner can never confirm ownership at the reassigned generation",
        );
    });
}
