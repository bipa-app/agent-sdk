# ADR-0001: Local Durability — SQLite Embedded Storage Backend

| Field       | Value                                                |
|-------------|------------------------------------------------------|
| **Status**  | Proposed                                             |
| **Date**    | 2026-04-11                                           |
| **Issue**   | [ENG-7987](https://linear.app/bipa/issue/ENG-7987)   |
| **Parent**  | Track A — Concrete Durability Backends               |

## Context

The agent-sdk now has a concrete server-side durability backend
(`PostgresDurableStore`) that implements all 10+ journal store traits
across 9 database tables.  The next required backend is for **local
usage**: the desktop app and CLI daemon that run on a single machine
with no external database server.

The local backend must survive process restarts, support replay and
recovery, and integrate with the existing `StoreRegistry` / config
model.  We need to decide:

1. Which embedded storage engine to use.
2. How locking and concurrency work on a single machine.
3. Where local semantics may intentionally differ from PostgreSQL.
4. How the local database file is configured, located, and cleaned up.

### Candidates evaluated

| Engine | Transactions | SQL | Maturity | Rust ecosystem | Notes |
|--------|-------------|-----|----------|----------------|-------|
| **SQLite (via `sqlx`)** | Full ACID, WAL mode | Yes | 25+ years, billions of deployments | `sqlx` already in workspace | Schema reuse from Postgres; same query layer |
| **SQLite (via `rusqlite`)** | Full ACID, WAL mode | Yes | 25+ years | Mature, sync API | Needs `spawn_blocking`; different query style |
| **redb** | ACID, MVCC | No (key-value) | ~3 years | Growing | Would require hand-rolled query/index layer |
| **sled** | Partial (no full ACID) | No (key-value) | Unmaintained since 2022 | Stalled | Unsuitable for production |
| **Raw files (JSON/bincode)** | No | No | N/A | N/A | No atomicity; corruption risk on crash |

## Decision

**Use SQLite via `sqlx` in WAL mode as the local durable storage
backend.**

SQLite is chosen because:

1. **Schema reuse.** The 9 Postgres tables can be expressed in SQLite
   SQL with minimal dialect translation (see [Schema Parity](#schema-parity)
   below).  This eliminates the need for a parallel data model.

2. **`sqlx` unification.** The workspace already depends on `sqlx`.
   Adding the `sqlite` feature to the existing dependency means the
   local backend uses the same `sqlx::query!` / `sqlx::Pool` /
   `sqlx::migrate!` patterns as the Postgres backend.  One query
   style, two backends.

3. **ACID guarantees.** SQLite in WAL mode provides full crash
   recovery.  A process crash mid-transaction loses only uncommitted
   work — exactly the same guarantee as Postgres.

4. **Proven at scale.** SQLite is deployed on billions of devices.
   Its local-file model is a natural fit for desktop and CLI processes
   that own a single data directory.

5. **No operational burden.** No server to install, no connection
   string, no Docker container.  The backend creates its database file
   on first use.

### Alternatives rejected

- **`rusqlite`**: Would require a different query interface
  (`spawn_blocking` wrappers, `params![]` macro, manual row mapping)
  diverging from the `sqlx`-based Postgres backend.  The maintenance
  cost of two SQL dialects × two driver APIs is not justified.

- **redb / sled**: Key-value stores would require hand-building every
  index, ordering guarantee, and multi-table transaction that SQL
  gives for free.  The store traits assume relational semantics (e.g.
  `list_by_thread`, `list_children`, `acquire_next_runnable` with
  ordered scans and skip-locked patterns).

- **Raw files**: No crash-safe atomicity.  A power loss during a
  multi-step commit (close attempt + advance counters + append
  messages + update head + insert checkpoint) would corrupt state.

## Detailed design

### Schema parity

The 9 Postgres tables map to SQLite with the following dialect
changes.  No structural differences — column names, types, and
constraints are preserved.

| Postgres construct | SQLite equivalent | Notes |
|--------------------|-------------------|-------|
| `TEXT PRIMARY KEY` | `TEXT PRIMARY KEY` | Identical |
| `JSONB` | `TEXT` (JSON stored as text) | SQLite `json()` functions available; no binary JSONB |
| `TIMESTAMPTZ` | `TEXT` | ISO 8601 strings; `sqlx` `time::OffsetDateTime` codec handles this |
| `BIGINT` | `INTEGER` | SQLite `INTEGER` is 64-bit signed |
| `BOOLEAN` | `INTEGER` (0/1) | Standard SQLite convention |
| `CHECK (col IN (...))` | `CHECK (col IN (...))` | Identical |
| `FOREIGN KEY ... DEFERRABLE` | `FOREIGN KEY` + `PRAGMA foreign_keys = ON` | SQLite FKs are opt-in at connection level |
| `CREATE INDEX ... WHERE` | `CREATE INDEX ... WHERE` | Partial indexes supported since SQLite 3.8.0 |

A separate `migrations/sqlite/` directory will hold SQLite-dialect
migration files mirroring the Postgres migrations.  The two sets
evolve independently but are tested for structural parity.

### Concurrency and locking model

#### Single-owner assumption

The local backend assumes **one process owns the database file** at a
time.  This is the desktop daemon or CLI process.  The owner may run
multiple async tasks and worker threads within that process.

Multiple external processes must **not** open the same SQLite file
concurrently.  The local backend enforces this with a POSIX advisory
lock on a companion `.lock` file at startup:

```
$DATA_DIR/agent-sdk.db       # SQLite database (WAL mode)
$DATA_DIR/agent-sdk.db-wal   # WAL file (managed by SQLite)
$DATA_DIR/agent-sdk.db-shm   # shared-memory file (managed by SQLite)
$DATA_DIR/agent-sdk.lock     # process-level advisory lock
```

If the lock cannot be acquired, the process exits with a clear error
rather than risk concurrent writes.

#### In-process concurrency

SQLite in WAL mode allows **concurrent reads** but **serialises
writes** at the database level.  This matches the local use case:

- **Reads** (get task, list threads, get history): proceed without
  blocking, even during an active write transaction.
- **Writes** (insert task, commit turn, advance retention): serialised
  by SQLite's internal writer lock.  `BEGIN IMMEDIATE` is used for all
  write transactions to fail-fast on contention rather than
  deadlocking.

Since the local backend is single-process, the Postgres `FOR UPDATE`
and `SKIP LOCKED` patterns translate naturally:

| Postgres pattern | Local SQLite equivalent | Rationale |
|------------------|------------------------|-----------|
| `SELECT ... FOR UPDATE` | `BEGIN IMMEDIATE` + plain `SELECT` | Database-level write lock replaces row-level lock |
| `SELECT ... FOR UPDATE SKIP LOCKED` | Not needed | Single-process workers use in-memory task dispatch; no lease contention |
| `CAS` via `WHERE worker_id = $1 AND lease_id = $2` | Same `WHERE` clause | CAS guards still useful for correctness even without multi-process contention |

#### Why row-level locking is unnecessary

The Postgres backend uses `FOR UPDATE` and `SKIP LOCKED` because
multiple service-host processes compete for tasks via the database.
In local mode there is exactly one process, so:

- **Worker acquisition** uses an in-process queue (the existing
  `InMemory` task dispatch) rather than `acquire_next_runnable` SQL
  scans.  The SQLite store persists the task state for restart
  recovery but does not need contention-resistant polling.

- **Lease expiry sweeps** still run to recover from unclean shutdown
  (e.g. `SIGKILL`), but they never race with another process.

### Recovery and restart model

Recovery after a crash or unclean shutdown follows this sequence:

1. **SQLite WAL replay.** SQLite automatically replays the WAL on
   open if the previous connection did not shut down cleanly.  This
   restores the database to the last committed transaction.

2. **Lease sweep.** The worker startup sweep
   (`release_expired_leases`) marks any `running` tasks whose leases
   expired during downtime back to `pending`.  This is the same logic
   as the Postgres backend.

3. **Outbox drain.** Pending outbox entries are re-claimed and
   delivered.  For local-only mode (no external event consumers), the
   outbox may be configured as a no-op.

4. **Intent reconciliation.** `ExecutionIntentStore` entries in
   `claimed` state on startup indicate tool calls that began but
   never completed.  The existing fail-closed intent reconciliation
   logic applies unchanged.

No manual intervention is required.  The local backend is
self-healing on restart.

### Intentional semantic differences from PostgreSQL

| Area | PostgreSQL backend | Local SQLite backend | Rationale |
|------|-------------------|----------------------|-----------|
| **Multi-process access** | Designed for it (connection pool, row locks) | Single-process only; advisory file lock | Desktop/CLI owns the machine |
| **Worker acquisition** | `SKIP LOCKED` poll | In-process queue + persisted state | No contention; lower latency |
| **Outbox relay** | Delivers to external subscribers | Optional; may be no-op for local-only | No external consumers in local mode |
| **Connection pooling** | `PgPool` with configurable pool size | Single `SqlitePool` (WAL mode handles reader concurrency) | One machine, one process |
| **Schema migrations** | `sqlx::migrate!` from `migrations/postgres/` | `sqlx::migrate!` from `migrations/sqlite/` | Separate dialect files; tested for structural parity |
| **Timestamp storage** | `TIMESTAMPTZ` (binary) | `TEXT` ISO 8601 | `sqlx` codec abstracts the difference |
| **JSON storage** | `JSONB` (binary, indexable) | `TEXT` (JSON string) | No JSON path index queries in current schema |

### Required parity (must not differ)

These behaviours **must** be identical between backends to ensure
agents behave the same in local and server modes:

1. **Task lifecycle state machine.** Status transitions, admission
   constraints, and parent–child invariants are identical.
2. **Turn commit atomicity.** The 5-table `commit_completed_turn`
   transaction must be all-or-nothing in both backends.
3. **Event ordering.** Committed events within a thread have
   contiguous, monotonically increasing sequence numbers.
4. **Checkpoint immutability.** Once committed, a checkpoint is
   never mutated or rewritten.
5. **Retention semantics.** `advance_retention_floor` deletes events
   below the floor and advances the cursor atomically.
6. **Execution intent fail-closed guard.** An intent must be
   persisted before a side-effecting tool runs, and reconciled on
   restart.
7. **Message projection.** `replace_history` and `commit_messages`
   produce the same observable message stream.
8. **CAS guards.** Optimistic version checks on message heads and
   heartbeat CAS on leases behave identically (even though
   contention is unlikely in local mode).

### Configuration model

The `StorageBackend` enum gains a `Sqlite` variant:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageBackend {
    #[default]
    InMemory,
    Sqlite {
        /// Path to the database file.
        /// Default: `$XDG_DATA_HOME/agent-sdk/agent-sdk.db`
        /// (macOS: `~/Library/Application Support/agent-sdk/agent-sdk.db`)
        path: Option<String>,
    },
    // Postgres { url: String },  // future
}
```

YAML configuration:

```yaml
# Minimal — uses platform-default data directory
storage:
  backend: !sqlite
    path: null

# Explicit path
storage:
  backend: !sqlite
    path: "/home/user/.local/share/agent-sdk/agent-sdk.db"
```

#### Default data directory resolution

| Platform | Default path |
|----------|-------------|
| Linux | `$XDG_DATA_HOME/agent-sdk/agent-sdk.db` (fallback: `~/.local/share/agent-sdk/`) |
| macOS | `~/Library/Application Support/agent-sdk/agent-sdk.db` |
| Windows | `%LOCALAPPDATA%\agent-sdk\agent-sdk.db` |

The `dirs` crate (or `dirs-next`) provides platform-correct resolution.
The directory is created on first use with `0700` permissions.

#### Cleanup and retention

The existing `RetentionConfig` applies unchanged:

```yaml
retention:
  event_ttl_secs: 604800    # 7 days
  checkpoint_max_per_thread: 50
```

Additionally, the local backend respects an optional maximum database
size.  When exceeded, the oldest completed tasks and their associated
data are purged:

```yaml
storage:
  backend:
    sqlite:
      path: null           # use platform default
      max_size_mb: 512     # optional; null = unlimited
```

### Implementation structure

The SQLite backend is implemented as `SqliteDurableStore` in the
`agent-service-host` crate, behind a `sqlite` cargo feature flag:

```
crates/agent-service-host/
├── Cargo.toml                          # [features] sqlite = ["sqlx/sqlite"]
├── src/
│   ├── config.rs                       # StorageBackend::Sqlite variant
│   ├── stores.rs                       # StoreRegistry::from_config match arm
│   ├── sqlite.rs                       # module root + integration tests
│   └── sqlite/
│       ├── store.rs                    # SqliteDurableStore: all trait impls
│       └── migrations.rs              # sqlx::migrate! from sqlite dir
└── migrations/
    ├── postgres/
    │   ├── 0001_durable_core.sql
    │   └── 0002_event_journal_outbox.sql
    └── sqlite/
        ├── 0001_durable_core.sql       # SQLite-dialect mirror
        └── 0002_event_journal_outbox.sql
```

The `SqliteDurableStore` struct mirrors `PostgresDurableStore`:

```rust
#[derive(Clone)]
pub struct SqliteDurableStore {
    pool: SqlitePool,
}
```

It implements the same 10 store traits.  Query strings differ where
dialect requires (e.g. `RETURNING` syntax, `json_extract` vs `->>`),
but the trait method signatures and semantics are identical.

### Testing strategy

1. **Structural parity tests.** A test compares the column names,
   types, and constraints of the SQLite schema against the
   reference Postgres schema metadata (similar to the existing
   `test_schema_alignment_*` tests in `postgres.rs`).

2. **Shared trait conformance suite.** The existing `InMemory*` tests
   are extracted into a backend-agnostic test harness parameterised
   over `&dyn Store`.  Both Postgres and SQLite backends run the same
   suite.

3. **Crash recovery tests.** Tests that kill the `SqlitePool` mid-
   transaction and re-open the database to verify WAL replay
   restores the last committed state.

4. **Advisory lock tests.** Verify that a second process attempting
   to open the same database file receives a clear error.

## Consequences

### Positive

- **Desktop and CLI have crash-safe, zero-ops persistence** out of
  the box.
- **Schema and query reuse** reduces the maintenance surface compared
  to a non-SQL backend.
- **Transport planning is unblocked.** The local daemon's restart,
  replay, and concurrency semantics are defined.
- **Feature-flag isolation** (`sqlite`) keeps the SQLite dependency
  out of server builds that only need Postgres.

### Negative

- **Two SQL dialects to maintain.** Migration files and a subset of
  queries will diverge where Postgres and SQLite syntax differs.
  Mitigated by structural parity tests.
- **No row-level locking.** Write-heavy workloads on large datasets
  will serialise at the database level.  Acceptable for local
  single-user usage; not suitable for a multi-tenant server.
- **JSON queries are weaker.** If future features require JSONB path
  indexing, the SQLite backend will need application-level workarounds
  or additional indexed columns.

### Risks

- **`sqlx` SQLite support maturity.**  `sqlx`'s SQLite driver is
  less exercised than its Postgres driver.  Mitigated by integration
  tests and by keeping the local workload modest (single user, not
  high-throughput).
- **WAL mode + NFS.**  SQLite WAL mode does not work correctly on
  network filesystems.  The advisory lock and documentation must warn
  against placing the data directory on NFS/SMB mounts.
