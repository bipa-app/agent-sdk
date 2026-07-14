# ADR-0003: The Subagent Activity Contract

| Field       | Value                                                       |
|-------------|-------------------------------------------------------------|
| **Status**  | Accepted                                                     |
| **Date**    | 2026-07-13                                                   |
| **Parent**  | ENG-9192 — subagent timeout is a stall budget, not a deadline |

## Context

`spec.timeout_ms` is a budget of **silence**, not of work. A subagent is
failed only after it has gone the whole budget with no evidence of work;
a child that keeps working never expires, however long it runs.

> "Its a timeout of stallness and not a timeout of work. Some subagents
> work may take a long time and we dont want to kill the session and lose
> all work because of a stupid timeout config."

That inverts the question the enforcement path asks. It no longer asks
*"how old is this child"* (a value read from one column) but *"when did
anything in this child's subtree last make progress"* — a value
**assembled** from an in-memory beacon, a durable per-task column, a
subtree walk, and a committed-event probe, written by four refresh points
and read by three enforcement legs across three storage backends.

Every one of those sites can silently break the answer, and each break
looks like the same bug: **a healthy child is killed and its work thrown
away.** Four review rounds each found *new* instances of that one bug
class (4 → 5 → 3 → 5 findings) because the invariants were enforced
piecemeal at whichever call site a reviewer happened to look at.

This ADR states the invariants explicitly so they can be enforced at
**choke points** and proved by test, instead of rediscovered one call
site at a time.

**The bias, everywhere: never kill on unknown.** Every ambiguity in this
document resolves toward "the child is alive". A store error, a truncated
walk, a probe that overran its budget, a missing row — all answer *not
expired*. A genuinely stalled child stays stalled and dies on the next
tick; a healthy one killed on bad evidence is unrecoverable. The costs
are not symmetric.

---

## The invariants

### I1 — Effective activity

> Every task row has `effective_activity = last_activity_at ?? created_at`.
> No reader may treat `None` as "stale forever". A row's creation **is**
> its initial activity.

`last_activity_at` is `NULL` on any row that has not yet been acquired
(`new_child`, `new_root_turn`, `new_subagent_invocation` all start it
`None`) and on every row that predates migration `0012`. A reader that
folds `None` into "no evidence" throws away the one thing it does know:
the row exists, and it was created at `created_at`.

The failure this prevents: a parent parks on children it *just spawned*;
those children are still `Queued`, so they carry no activity; under tight
retention the spawn events have aged out of the journal — and the parent
is reaped as silent moments after it demonstrably worked.

Note this does **not** weaken the reap. A child that never starts has an
old `created_at`, so its effective activity is old, and it still times
out — which is exactly what `created_at` already meant on the subject's
own floor (see I-floor below).

**Choke point:** `note_row_activity` (`host.rs`) — the single place a task
row's durable activity is folded into a probe.

---

### I2 — Monotonicity at the storage layer

> No write path — Rust setter, SQL `UPDATE`, any backend — may move
> `last_activity_at` backward. Every comparison must be **chronological**.

Monotonicity is not a nicety: a rewind is indistinguishable from silence.
A terminal transition stamped with an instant captured *before* a long
execution, or a comparison that picks the older of two timestamps, hands
the probe a subtree that "went quiet" at the exact moment a long child
finished — and a parked ancestor is reaped.

The trap is that "compare two timestamps" is **not** a safe default in
SQL. SQLite binds `OffsetDateTime` as RFC3339 **TEXT** (sqlx →
`format(&Rfc3339)`), and `time` trims trailing zeros, so the stored
precision is **variable**. Lexical order is therefore *not* chronological:

| a | b | lexical `MAX` | chronological max |
|---|---|---|---|
| `…T10:00:00Z` | `…T10:00:00.5Z` | **`…:00Z`** ❌ | `…:00.5Z` |
| `…T10:00:00.1Z` | `…T10:00:00.12Z` | **`…:00.1Z`** ❌ | `…:00.12Z` |
| `…T10:00:00.5Z` | `…T10:00:00.55Z` | **`…:00.5Z`** ❌ | `…:00.55Z` |

`'Z'` (0x5A) sorts after `'.'` (0x2E), so a whole-second timestamp beats a
sub-second one in the same second; and any fraction that is a lexical
*prefix* of another wins over the longer, later one. The comparison must
run on a parsed/numeric form (`julianday()`), never on the text.

**Choke points:**

| Backend | Activity write | Guard |
|---|---|---|
| in-memory | `AgentTask::advance_last_activity_at` | Rust `>` on `OffsetDateTime` |
| Postgres | `heartbeat_task` | `GREATEST(...)` on `timestamptz` |
| SQLite | `heartbeat_task` | `julianday(...)` comparison, **not** `MAX()` on TEXT |

**Carve-out (deliberate):** `AgentTaskStore::update` and the stores'
private `update_task_tx` write the row **verbatim**, with no monotonic
guard. They are *not* activity write paths: `update_task_tx` is the write
half of a read-modify-write whose read is taken under a row lock
(`load_task_tx(.., lock = true)` in Postgres; the write-lock / snapshot
upgrade in SQLite and in-memory), so the value it writes back was itself
produced by the monotonic setter and can never be older than what is
stored. `update` has **no production caller** — it is a test seeding
primitive, and seeding an arbitrary row must stay possible.

---

### I3 — No activity left behind on transitions

> Every transition that **drops a lease or ends a heartbeat** must persist
> the transition instant durably as part of that transition's CAS.

Activity is persisted by the heartbeat, at most once per tick. A
transition that ends the heartbeat therefore *discards* every bump since
the last tick — and for a row that parks, that discarded instant is the
**last** activity it will ever record until re-acquisition. The row's
durable clock is left behind reality by up to a full tick, and the parked
sweep reads exactly that clock.

Stamping the **transition instant** (`now`) rather than the beacon is both
simpler and strictly stronger: every beacon bump was observed *during* the
execution that is now ending, so `now ≥ beacon.latest()` always. The
transition instant dominates.

> **The transition instant must be captured AT the transition.** The pure
> helpers stamp whatever `now` they are handed — so a caller that bound one
> `now` at task *entry* and reused it after a long execution backdates the
> row by the whole run. `execute_tool_task`, `execute_subagent_task_entry`
> and the detached Confirm-tier drive all did this, and the Confirm-tier
> drive is *designed* to outlive its lease. The monotonic setter (I2) stops
> it rewinding below heartbeat-persisted activity, but for a task whose
> beacon never fired it stamps the row hours in the past — and `completed_at`
> with it. The helpers stay pure (the determinism the rest of these tests
> depend on); the discipline is at the caller, and the entry binding is
> named `started_at` so reusing it reads as wrong.

The transitions, and why each does or does not stamp — this table is the
contract, and a new transition must be added to it:

| Transition | Ends a heartbeat? | Stamps | Why |
|---|---|---|---|
| `mark_running` | acquires | **yes** | acquisition is evidence of life; gives a re-acquired child a fresh budget instead of reaping it off a pre-crash timestamp |
| `touch_heartbeat` | no | **yes** | the ordinary once-per-tick write-through |
| `complete` / `complete_with_result` | yes | **yes** | terminal rows are read *inline* by the probe — a just-completed child is a fresh sign of life for its parked parent |
| `fail` / `fail_with_reason` | yes | **yes** | same |
| `cancel` | yes | **yes** | same — the third terminal outcome; its absence was pure asymmetry |
| `wait_on_children` | yes | **yes** | park: the turn that produced this park *was* the work |
| `repark_after_steering` | yes | **yes** | park |
| `await_confirmation` | yes | **yes** | park |
| `release_lease` | yes | **no** | the *sweeper* reclaiming a **dead worker's** row. Nothing worked; crediting it would launder a crash into evidence of life. The row is re-stamped by `mark_running` when someone actually picks it up. |
| `begin_steering_resume` | no (already parked) | **no** | holds no lease, so no beacon is in flight; `mark_running` stamps on acquisition |
| `resume_from_confirmation` | no (already parked) | **no** | same |
| `child_resolved` / `recompute_pending_children` | no | **no** | the child's own terminal stamp is the signal; double-recording it on the parent adds nothing |
| `admit_as_queued` / `promote_to_pending` | no | **no** | I1 makes `created_at` the initial activity |

**Choke point:** the pure `AgentTask` transition helpers in
`journal/task.rs`. All three backends — in-memory, SQLite, Postgres —
route **every** status transition through them (verified: no backend
hand-writes a task status `UPDATE`), so the stamp propagates to all three
from one edit.

---

### I4 — The floor covers every persistence cadence

> The enforcement floor must be **≥ 2 × the slowest cadence that can carry
> activity for the probed subtree**.

A parked child's evidence is entirely durable, and the durable value is
written at most once per heartbeat tick. A budget shorter than the
persistence cadence can therefore reap a child whose descendant **is**
reporting but whose report has not yet been written. The effective budget
is floored at `MIN_STALL_BUDGET_HEARTBEATS × heartbeat_interval` (k = 2:
one interval to persist, one for the sweep to observe) — **at
enforcement**, where the cadence is known, because spec resolution in the
SDK cannot see the host's heartbeat interval.

A configured `timeout_ms` below that floor is **not honored literally**;
enforcement uses the floor. `timeout_ms` is unchanged on the wire.

The trap is the word *slowest*. The floor is only sound if **every**
heartbeat that can persist activity for a probed subtree beats at the
cadence the floor is derived from. There were two:

| Cadence | Site | Interval |
|---|---|---|
| worker pool | `run_task_with_heartbeat` | `worker.heartbeat_interval` |
| detached Confirm-tier tool drive | `drive_approved_confirmation` (grpc) | ~~`lease_duration / 3`~~ |

A Confirm-tier tool's row is precisely what keeps its **parked** ancestor
alive, and that ancestor's floor was derived from the *worker's* interval.
With a 60s lease and a 5s worker heartbeat, the tool persisted every 20s
against a 10s floor — a false reap of a tool that was actively emitting.

Rather than teach every enforcement site a second cadence — which
re-scatters the invariant and silently under-covers again the next time
someone spawns a heartbeat — **there is now exactly one cadence.** The
detached drive beats at the configured `worker.heartbeat_interval`, so
`2 × heartbeat_interval` is correct *by construction* at every enforcement
site. (At the default 30s lease / 10s heartbeat the derived value was
already 10s, so default deployments are unaffected.)

**Choke point:** `HeartbeatLoopParams::heartbeat_interval`. **Every**
`heartbeat_loop` must be spawned with the configured worker cadence. A
loop that beats slower silently under-covers the floor.

---

### I5 — The probe is bounded, and a bounded-out probe means "alive"

> Every stall probe runs under a timeout that cannot outlast the lease it
> holds, and a probe that exceeds it answers **not expired**.

The probe walks a subtree with serial store reads. On the **running** leg
it runs under the lease the heartbeat just renewed: a slow store or a
large subtree that outlasts the lease lets the expiry sweep requeue and
re-acquire a row whose provider future is still running — **duplicate
execution**, the worst failure on this path. Bounding the probe strictly
below half the freshly-renewed lease closes that window.

| Leg | Bound |
|---|---|
| running (`deadline_tick`) | `timeout(min(lease / 2, STALL_PROBE_MAX))` |
| parked sweep (`enforce_subagent_deadlines`) | `timeout(STALL_PROBE_MAX)` (holds no lease) |
| acquisition | *deleted* — acquisition is evidence of work (I3), so a freshly-acquired child is never stalled at dispatch |

Both bounds are fail-safe: a timeout is `false` (not expired), logged, and
retried next tick/sweep. Both are **parameters**, not constants read in
place, so both fail-safes are provable without a sleep.

**A timeout is not enough — the READS must be bounded too.** A fail-safe
that fires is not free: a probe that can never finish makes a wedged child
**immortal**, because every timeout answers *alive*. The walk expands only
non-terminal rows, but it used to *read* every child (`list_children`) and
discard the terminal ones in memory — and terminal task rows are retained.
A parent with thousands of finished tool calls therefore materialized
thousands of rows on **every** probe of **every** parked ancestor, until no
probe could finish inside its bound. `AgentTaskStore::probe_children`
replaces that with two bounded, indexed reads — an `EXISTS` for a fresh
sign of life across all children, and a **capped** live frontier — so the
cost of a probe is independent of how much history the parent has
accumulated, and enforcement can actually converge.

---

### The floor of last resort: `created_at`

Underneath all five, `stall_expired` keeps its first condition: a child
younger than its own budget (`now < created_at + timeout_ms`) can **never**
be stalled out. Spawn is the initial evidence of life. This is also what
still reaps a child that never starts at all — a wedged admission that
hangs before its first frame produces no beacon, no durable stamp, and no
event, and is reaped off its `created_at` floor by the parked sweep.

---

## The commit path — refresh point (c), enumerated

An event commit is evidence of work (refresh point (c)). The beacon is the
*retention-proof* half of that signal: the committed-event probe can be
defeated by `event_ttl_secs < timeout_ms`, and only the durable
`last_activity_at` survives a purge. So **every** commit must bump.

It didn't. The bump lived at call sites, and only one call site had it:

| Commit site | Bumped before? |
|---|---|
| streaming-delta flush (`flush_streaming_deltas`) | **yes** |
| `emit_auto_retry_start` / `emit_auto_retry_end` | no |
| `compaction::apply_compaction` | no |
| turn-start / user-input commits | no |
| suspension + content batches | no |
| tool-result batch (`commit_tool_events`) | no — but covered twice over, by `ToolEventCollector::emit` and by the terminal stamp (I3) |

A child in retry backoff, or finishing a compaction, commits real work,
has it purged, and is reaped as silent.

The fix is a **mechanism, not a rule**:
`ActivityTrackingEventRepo` decorates the `EventRepository`, so a commit
that does not record activity is unrepresentable on this path — including
for call sites that do not exist yet. `RootTurnDeps::wire_activity` sets
the beacon and the decorated repository **together**, because wiring one
without the other is precisely the bug.

**Known boundary:** `EventRepository::atomic_event_outbox_committer` hands
out a raw committer that writes events *without* passing through
`commit_event`. It is delegated unchanged (no production commit path uses
it today, and suppressing it would drop the backends' atomic event+outbox
unit of work). A commit path that adopts that hook must bump the beacon
itself.

---

## Consequences

* A new transition that drops a lease must be added to the I3 table and
  must stamp, or explain in that table why it does not.
* A new `heartbeat_loop` spawn site must use the configured worker
  cadence (I4), or the floor must be re-derived.
* A new storage backend must guard its activity write with a
  **chronological** comparison (I2) — the in-memory setter's `>` is the
  reference semantics; text order is not a substitute. Its
  `probe_children` must likewise be **bounded** and must agree with
  `ChildProbe::from_rows`.
* Any new probe leg must be bounded and fail-safe (I5).
* A commit path that bypasses `EventRepository::commit_event*` (i.e. uses
  the atomic outbox committer) must bump the beacon itself.
* **Anything the probe cannot read is `truncated`, never "no activity".**
  A missing row, a store error, a capped walk — all mean *alive*. The
  failure this closes: a dangling subagent linkage, or a subtree root that
  vanished, returned an empty-but-successful walk that read as total
  silence and killed a live ancestor. "Absent" and "unreadable" are not the
  same answer.
* **Bounded reads cut both ways.** An unbounded read is not just slow: past
  its timeout the probe is treated as alive, so a wedged child becomes
  *immortal*. Enforcement can only converge if every read the probe makes
  is bounded independently of retained history.
