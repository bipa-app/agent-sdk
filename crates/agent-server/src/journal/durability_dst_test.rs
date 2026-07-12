//! Seeded Deterministic Simulation Testing (DST) loop for the journal
//! task store (Phase 11 · D).
//!
//! A single-process, single-seed scheduler that interleaves `M`
//! simulated workers over one [`InMemoryAgentTaskStore`], injecting the
//! four fault classes the durable layer must tolerate —
//!
//! - **lease expiry** (a worker's heartbeat lapsed; the sweep requeues
//!   or fails the row),
//! - **duplicate wakeups** (the broker re-delivered an advisory nudge),
//! - **cancellation** (a caller cancelled a thread mid-flight),
//! - **crash** (a worker vanished without releasing its lease) —
//!
//! across `K` operations, asserting the journal's global invariants
//! after **every** step. All nondeterminism (which worker acts, which
//! action it takes, which fault fires, virtual-clock advance) is drawn
//! from one seeded PRNG, so a failing run is fully reproducible: the
//! seed is printed on failure and a dedicated regression test replays
//! it.
//!
//! # Why no real concurrency
//!
//! [`InMemoryAgentTaskStore`]'s methods take the store's write lock,
//! mutate, and return without ever awaiting a yield point — they are
//! effectively synchronous critical sections. Real OS-thread
//! interleaving would therefore add no coverage the lock does not
//! already serialize, and would destroy reproducibility. The DST loop
//! instead models the interleaving explicitly: each "step" is one
//! worker's atomic store operation, and the *order* of steps is the
//! schedule the seed controls. This is the standard single-threaded-DST
//! technique (`FoundationDB` / `TigerBeetle` style): control the
//! schedule, not the threads.
//!
//! # Invariants asserted every step
//!
//! 1. **Single blocking root per thread** — at most one root in a
//!    blocking status (`Pending`/`Running`/waiting) per thread.
//! 2. **No double ownership** — no two `Running` rows share a
//!    `(worker_id, lease_id)`, and every `Running` row has both set.
//! 3. **Lease/state coherence** — non-`Running` rows carry no lease;
//!    every row passes its own structural `validate()`.
//! 4. **Cancel propagates** — once a thread is cancelled, the whole
//!    tree is terminal and stays terminal.
//! 5. **Fail-closed child wakes parent** — a `WaitingOnChildren` parent
//!    is never left blocked once all its children are terminal.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result};
use time::{Duration, OffsetDateTime};

use super::store::{AgentTaskStore, InMemoryAgentTaskStore};
use super::task::{AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId};
use agent_sdk_foundation::ThreadId;

// ─────────────────────────────────────────────────────────────────────
// Deterministic PRNG (SplitMix64) — no external `rand` dependency
// ─────────────────────────────────────────────────────────────────────

/// A tiny seeded PRNG. `SplitMix64` is fast, has good statistical
/// properties for a simulation driver, and — crucially — is fully
/// reproducible from a single `u64` seed with no platform dependence.
struct Prng {
    state: u64,
}

impl Prng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`. `n` must be > 0.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A uniform index in `[0, len)`. `len` must be > 0. Returns a
    /// `usize` without a lossy cast: the draw is reduced modulo `len`
    /// (as a `u64`), so the result always fits.
    fn index(&mut self, len: usize) -> usize {
        let len_u64 = len as u64;
        // `below` returns a value strictly less than `len`, so the
        // conversion back to `usize` is always exact.
        usize::try_from(self.below(len_u64)).unwrap_or(0)
    }

    fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Simulated worker bookkeeping
// ─────────────────────────────────────────────────────────────────────

/// What one simulated worker is currently holding. The scheduler keeps
/// this mirror so it can drive valid follow-up operations (heartbeat,
/// complete) without re-reading the store on every decision — but every
/// store call is still re-validated against the live row, and the
/// invariant pass reads the store as the single source of truth.
#[derive(Clone)]
struct HeldLease {
    task_id: AgentTaskId,
    worker: WorkerId,
    lease_id: LeaseId,
}

struct Simulation {
    store: InMemoryAgentTaskStore,
    prng: Prng,
    clock: OffsetDateTime,
    /// Threads in play. A bounded set keeps same-thread contention high
    /// so the single-blocking-root invariant is exercised hard.
    threads: Vec<ThreadId>,
    /// Per simulated worker: the lease it currently holds, if any.
    held: BTreeMap<u32, Option<HeldLease>>,
    /// Task ids whose subtree a caller has cancelled via `cancel_tree`.
    /// Each such row — and every descendant the cancel returned — must
    /// stay terminal forever. Tracked per-row (not per-thread) because
    /// `cancel_tree` cancels exactly one root's subtree, leaving
    /// independent queued roots on the same thread untouched.
    cancelled_task_ids: HashSet<AgentTaskId>,
    /// Monotonic id source so spawned task / lease / worker names never
    /// collide within one run.
    seq: u64,
}

impl Simulation {
    fn new(seed: u64, thread_count: usize, worker_count: u32) -> Self {
        let store = InMemoryAgentTaskStore::new();
        let threads = (0..thread_count)
            .map(|i| ThreadId::from_string(format!("dst-thread-{i}")))
            .collect();
        let held = (0..worker_count).map(|w| (w, None)).collect();
        Self {
            store,
            prng: Prng::new(seed),
            clock: OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000),
            threads,
            held,
            cancelled_task_ids: HashSet::new(),
            seq: 0,
        }
    }

    fn fresh_id(&mut self, prefix: &str) -> String {
        self.seq += 1;
        format!("{prefix}-{}", self.seq)
    }

    fn pick_thread(&mut self) -> ThreadId {
        let idx = self.prng.index(self.threads.len());
        self.threads[idx].clone()
    }

    fn advance_clock(&mut self) {
        // Virtual time only — never a real sleep. Advancing by a small
        // jittered amount keeps lease-expiry timing interesting. The draw
        // is in `[0, 5)`, so `i64::try_from` is always exact.
        let jitter = i64::try_from(self.prng.below(5)).unwrap_or(0);
        self.clock += Duration::seconds(1 + jitter);
    }

    // ── One scheduler step ────────────────────────────────────────────

    /// Execute exactly one scheduled action and then advance virtual
    /// time. Returns `Ok(())` on success; store errors propagate so a
    /// genuine bug surfaces rather than being swallowed.
    async fn step(&mut self) -> Result<()> {
        // Weighted action menu. The weights bias toward forward progress
        // (submit / acquire / complete) while still injecting every fault
        // class frequently enough to stress recovery.
        let action = self.prng.below(100);
        match action {
            0..=24 => self.act_submit_root().await?,
            25..=49 => self.act_acquire().await?,
            50..=59 => self.act_heartbeat().await?,
            60..=74 => self.act_complete().await?,
            75..=82 => self.act_crash(),
            83..=89 => self.act_expire_leases().await?,
            90..=95 => self.act_duplicate_wakeup().await?,
            _ => self.act_cancel_thread().await?,
        }
        self.advance_clock();
        Ok(())
    }

    /// Submit a fresh root turn. The store decides Pending vs Queued; the
    /// single-blocking-root invariant is what we later assert.
    async fn act_submit_root(&mut self) -> Result<()> {
        let thread = self.pick_thread();
        // A fresh root mints a brand-new id, so it is never in the
        // cancelled set — a new root after a cancellation is a legitimate
        // fresh start that must be admitted as Pending or Queued.
        let root = AgentTask::new_root_turn(thread, self.clock, 3);
        // A duplicate id submission is rejected by the store; we always
        // mint fresh ids so this is a clean admission.
        let _ = self.fresh_id("root");
        // A submission rejection (e.g. a thread at its queue cap) is a
        // legitimate store outcome, not an invariant violation — tolerate
        // it. Only the per-step invariant pass can fail the run.
        let _ = self.store.submit_root_turn(root).await;
        Ok(())
    }

    /// An idle worker scans for a runnable row and claims it.
    async fn act_acquire(&mut self) -> Result<()> {
        let Some(worker_idx) = self.pick_idle_worker() else {
            return Ok(());
        };
        let worker = WorkerId::from_string(self.fresh_id("w"));
        let lease = LeaseId::from_string(self.fresh_id("l"));
        let expires = self.clock + Duration::seconds(30);
        let claimed = self
            .store
            .acquire_next_runnable(worker.clone(), lease.clone(), expires, self.clock)
            .await
            .context("dst: acquire_next_runnable")?;
        if let Some(task) = claimed {
            self.held.insert(
                worker_idx,
                Some(HeldLease {
                    task_id: task.id,
                    worker,
                    lease_id: lease,
                }),
            );
        }
        Ok(())
    }

    /// A worker heartbeats the lease it holds, extending expiry.
    async fn act_heartbeat(&mut self) -> Result<()> {
        let Some((worker_idx, held)) = self.pick_busy_worker() else {
            return Ok(());
        };
        let expires = self.clock + Duration::seconds(30);
        // The CAS may legitimately reject (the row was cancelled or its
        // lease expired and was requeued). A rejection means the worker
        // lost the row; drop our mirror and move on — that is precisely
        // the no-double-ownership guard doing its job.
        match self
            .store
            .heartbeat_task(
                &held.task_id,
                &held.worker,
                &held.lease_id,
                expires,
                self.clock,
            )
            .await
        {
            Ok(_) => {}
            Err(_) => {
                self.held.insert(worker_idx, None);
            }
        }
        Ok(())
    }

    /// A worker completes the leaf it holds (roots have no children in
    /// this model, so completion is always valid for a held leaf root).
    async fn act_complete(&mut self) -> Result<()> {
        let Some((worker_idx, held)) = self.pick_busy_worker() else {
            return Ok(());
        };
        // Either outcome drops our mirror: on Ok the row is terminal; on
        // a CAS rejection the lease was already stolen / cancelled.
        let _ = self
            .store
            .complete_task(&held.task_id, &held.worker, &held.lease_id, self.clock)
            .await;
        self.held.insert(worker_idx, None);
        Ok(())
    }

    /// Crash: a worker vanishes without releasing its lease. We simply
    /// forget the lease in our mirror; the store row stays `Running`
    /// with a live lease until the expiry sweep reclaims it. This is the
    /// realistic crash model — no clean teardown.
    fn act_crash(&mut self) {
        if let Some((worker_idx, _)) = self.pick_busy_worker() {
            self.held.insert(worker_idx, None);
        }
    }

    /// Run the lease-expiry sweep at the current virtual time. Rows whose
    /// lease lapsed are requeued or failed-closed by the recovery matrix.
    async fn act_expire_leases(&mut self) -> Result<()> {
        // Jump the clock forward far enough to expire some live leases,
        // then sweep. Any worker whose row was swept loses it on its next
        // CAS, which the heartbeat/complete handlers already tolerate.
        if self.prng.one_in(2) {
            self.clock += Duration::seconds(45);
        }
        let _records = self
            .store
            .release_expired_leases(self.clock)
            .await
            .context("dst: release_expired_leases")?;
        Ok(())
    }

    /// A duplicate broker wakeup re-runs the runnable scan. This must
    /// never create or double-run work — at most it claims one more
    /// runnable row for an idle worker, which is identical to a normal
    /// acquire. We model it as an extra acquire attempt.
    async fn act_duplicate_wakeup(&mut self) -> Result<()> {
        // A duplicate wakeup that nudges an already-busy pool is a no-op;
        // a duplicate that nudges an idle worker just acquires. Either
        // way the (Pending→Running) CAS guarantees single execution.
        self.act_acquire().await
    }

    /// A caller cancels a whole thread's tree.
    async fn act_cancel_thread(&mut self) -> Result<()> {
        let thread = self.pick_thread();
        let Some(active) = self
            .store
            .active_root_for_thread(&thread)
            .await
            .context("dst: active_root_for_thread")?
        else {
            return Ok(());
        };
        let cancelled = self
            .store
            .cancel_tree(&active.id, self.clock)
            .await
            .context("dst: cancel_tree")?
            .transitioned;
        // Record the active root AND every descendant the cancel touched
        // so the invariant pass can assert they stay terminal. Sibling
        // queued roots on the same thread are NOT part of this subtree,
        // so they are deliberately not recorded.
        self.cancelled_task_ids.insert(active.id);
        self.cancelled_task_ids.extend(cancelled);
        // Any worker holding a row in this tree loses it on its next CAS;
        // we don't eagerly clear mirrors, the CAS-rejection handlers do.
        Ok(())
    }

    // ── Worker selection helpers ──────────────────────────────────────

    fn pick_idle_worker(&mut self) -> Option<u32> {
        let idle: Vec<u32> = self
            .held
            .iter()
            .filter_map(|(w, lease)| lease.is_none().then_some(*w))
            .collect();
        if idle.is_empty() {
            return None;
        }
        let idx = self.prng.index(idle.len());
        Some(idle[idx])
    }

    fn pick_busy_worker(&mut self) -> Option<(u32, HeldLease)> {
        let busy: Vec<(u32, HeldLease)> = self
            .held
            .iter()
            .filter_map(|(w, lease)| lease.clone().map(|l| (*w, l)))
            .collect();
        if busy.is_empty() {
            return None;
        }
        let idx = self.prng.index(busy.len());
        Some(busy[idx].clone())
    }

    // ── Invariant checks (read the store as source of truth) ──────────

    /// Assert every global invariant against the live store. Returns the
    /// failing invariant's description so the caller can print the seed.
    async fn check_invariants(&self) -> Result<()> {
        let mut all_rows: Vec<AgentTask> = Vec::new();
        for thread in &self.threads {
            let rows = self
                .store
                .list_by_thread(thread)
                .await
                .context("dst: list_by_thread for invariant check")?;
            all_rows.extend(rows);
        }

        // Invariant 3: every row passes its own structural validation.
        for row in &all_rows {
            row.validate().with_context(|| {
                format!("invariant: row {} failed structural validate()", row.id)
            })?;
        }

        // Invariant 2: no two Running rows share a (worker, lease), and
        // every Running row has both lease fields populated; no
        // non-Running row carries a lease.
        let mut owners: BTreeMap<(String, String), AgentTaskId> = BTreeMap::new();
        for row in &all_rows {
            if row.status == TaskStatus::Running {
                let worker = row
                    .worker_id
                    .as_ref()
                    .with_context(|| format!("invariant: Running row {} has no worker", row.id))?;
                let lease = row
                    .lease_id
                    .as_ref()
                    .with_context(|| format!("invariant: Running row {} has no lease", row.id))?;
                let key = (worker.as_str().to_owned(), lease.as_str().to_owned());
                if let Some(other) = owners.insert(key, row.id.clone()) {
                    anyhow::bail!(
                        "invariant: double ownership — rows {} and {} share (worker, lease)",
                        other,
                        row.id,
                    );
                }
            } else {
                anyhow::ensure!(
                    row.worker_id.is_none() && row.lease_id.is_none(),
                    "invariant: non-Running row {} ({:?}) still carries a lease",
                    row.id,
                    row.status,
                );
            }
        }

        // Invariant 1: at most one blocking ROOT per thread.
        for thread in &self.threads {
            let blocking_roots = all_rows
                .iter()
                .filter(|r| {
                    r.thread_id == *thread
                        && r.kind == TaskKind::RootTurn
                        && r.status.blocks_root_admission()
                })
                .count();
            anyhow::ensure!(
                blocking_roots <= 1,
                "invariant: thread {thread} has {blocking_roots} blocking roots (>1)",
            );
        }

        // Invariant 4: every row whose subtree was cancelled is terminal
        // and stays terminal — cancellation is monotone.
        for row in &all_rows {
            if self.cancelled_task_ids.contains(&row.id) {
                anyhow::ensure!(
                    row.status.is_terminal(),
                    "invariant: cancelled row {} reverted to non-terminal status {:?}",
                    row.id,
                    row.status,
                );
            }
        }

        // Invariant 5: a WaitingOnChildren parent must still have at
        // least one non-terminal child (else it would be stuck forever).
        let by_parent: BTreeMap<AgentTaskId, Vec<&AgentTask>> = {
            let mut map: BTreeMap<AgentTaskId, Vec<&AgentTask>> = BTreeMap::new();
            for row in &all_rows {
                if let Some(parent) = &row.parent_id {
                    map.entry(parent.clone()).or_default().push(row);
                }
            }
            map
        };
        for row in &all_rows {
            if row.status == TaskStatus::WaitingOnChildren
                && let Some(children) = by_parent.get(&row.id)
            {
                let live = children.iter().any(|c| !c.status.is_terminal());
                anyhow::ensure!(
                    live,
                    "invariant: parent {} is WaitingOnChildren but all children are terminal \
                     (fail-closed child must wake parent)",
                    row.id,
                );
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Driver
// ─────────────────────────────────────────────────────────────────────

/// Run one full simulation for `ops` steps under `seed`, asserting all
/// invariants after every step. On any failure the seed is included in
/// the error so the exact run can be replayed by
/// [`replay_seed`].
async fn run_simulation(seed: u64, ops: usize) -> Result<()> {
    let mut sim = Simulation::new(seed, 4, 6);
    for op in 0..ops {
        sim.step()
            .await
            .with_context(|| format!("DST step {op} failed under seed {seed}"))?;
        sim.check_invariants().await.with_context(|| {
            format!(
                "DST invariant violated at step {op} under seed {seed} \
                 — replay with run_simulation({seed}, {ops})"
            )
        })?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

/// The headline DST loop: a handful of independent seeds, each running a
/// few thousand operations. Every invariant is checked after every step;
/// the seed is printed on failure for a deterministic replay.
#[tokio::test]
async fn seeded_dst_loop_holds_all_invariants() -> Result<()> {
    // A spread of fixed seeds — deterministic across runs and machines.
    // Each does 2_000 ops, so the suite covers ~16k scheduled operations
    // without being slow (the in-memory store ops are microseconds).
    let seeds: [u64; 8] = [
        1,
        7,
        42,
        1_337,
        0xDEAD_BEEF,
        0x0BAD_F00D,
        0xC0FF_EE00,
        0x1234_5678_9ABC_DEF0,
    ];
    for seed in seeds {
        run_simulation(seed, 2_000)
            .await
            .with_context(|| format!("seeded DST loop failed; replay with seed {seed}"))?;
    }
    Ok(())
}

/// Replay a single known-good seed for a longer horizon. If a future
/// change ever breaks an invariant under this seed, this test pins the
/// exact reproduction.
#[tokio::test]
async fn replay_seed_42_long_horizon() -> Result<()> {
    run_simulation(42, 4_000)
        .await
        .context("long-horizon replay of seed 42 violated an invariant")
}

/// Randomized exploration lane (finding #30).
///
/// The fixed-seed loop above replays the *same* ~16k schedules on every
/// run, so it can only catch regressions on schedules it already covered
/// — it can never *discover* a new interleaving bug. This lane lets a
/// nightly / on-demand run explore fresh schedules while staying
/// **deterministic-by-default** for the PR lane:
///
/// - With no env var set, the lane is a no-op (so a normal `cargo nextest
///   run` only executes the fixed-seed baseline above — same coverage,
///   same speed, fully reproducible).
/// - `AGENT_DST_SEED=<u64>` runs exactly that one seed. This is the
///   **replay** entry point: when the lane below prints a failing seed,
///   re-run with this var to reproduce it deterministically.
/// - `AGENT_DST_RANDOM=<N>` draws `N` fresh seeds (mixed from the wall
///   clock so each invocation explores new schedules) and prints **every
///   seed before running it**, so any failure names a seed the operator
///   can feed back through `AGENT_DST_SEED` for a deterministic replay.
/// - `AGENT_DST_OPS=<K>` overrides the per-seed operation count
///   (default 2000, matching the fixed baseline).
///
/// The seed generator is the same `SplitMix64` `Prng` the simulation
/// uses, so the only nondeterminism introduced is the *initial* clock
/// mix — and that is always printed, collapsing back to a fixed seed.
#[tokio::test]
async fn randomized_dst_exploration_lane() -> Result<()> {
    let ops = std::env::var("AGENT_DST_OPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(2_000);

    // Explicit single-seed replay path — highest priority so a printed
    // failing seed reproduces with one env var and nothing else.
    if let Some(seed) = std::env::var("AGENT_DST_SEED")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
    {
        println!("DST replay: AGENT_DST_SEED={seed} ops={ops}");
        return run_simulation(seed, ops)
            .await
            .with_context(|| format!("DST replay failed; reproduce with AGENT_DST_SEED={seed}"));
    }

    let Some(count) = std::env::var("AGENT_DST_RANDOM")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|count| *count > 0)
    else {
        // Deterministic-by-default: no env var means no exploration, so
        // the PR lane stays fast and reproducible on the fixed seeds.
        return Ok(());
    };

    // Mix the wall clock into a base seed so each invocation explores a
    // new region of the schedule space; the drawn seeds are printed so a
    // failure is always reproducible via AGENT_DST_SEED.
    let base = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_nanos() % u128::from(u64::MAX)).ok())
        .unwrap_or(0x1234_5678_9ABC_DEF0);
    let mut seed_source = Prng::new(base ^ 0x9E37_79B9_7F4A_7C15);
    for run in 0..count {
        let seed = seed_source.next_u64();
        println!(
            "DST exploration run {run}/{count}: seed={seed} ops={ops} (replay: AGENT_DST_SEED={seed})"
        );
        run_simulation(seed, ops).await.with_context(|| {
            format!("randomized DST lane failed; reproduce with AGENT_DST_SEED={seed} AGENT_DST_OPS={ops}")
        })?;
    }
    Ok(())
}

/// The PRNG is deterministic: the same seed yields the same sequence,
/// which is what makes "print the seed on failure" a real replay
/// guarantee and not a hopeful suggestion.
#[test]
fn prng_is_reproducible_from_seed() {
    let mut a = Prng::new(12_345);
    let mut b = Prng::new(12_345);
    for _ in 0..1_000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    // A different seed diverges immediately.
    let mut c = Prng::new(12_346);
    let mut d = Prng::new(12_345);
    assert_ne!(c.next_u64(), d.next_u64());
}
