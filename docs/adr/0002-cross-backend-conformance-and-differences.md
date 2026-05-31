# ADR-0002: Cross-Backend Conformance and Operational Differences

| Field       | Value                                                |
|-------------|------------------------------------------------------|
| **Status**  | Accepted                                             |
| **Date**    | 2026-04-12                                           |
| **Parent**  | Track A — Concrete Durability Backends               |

## Context

The agent-sdk supports three storage backends:

1. **In-memory** — used for tests and ephemeral development.
2. **PostgreSQL** — production server deployments.
3. **SQLite** — local desktop and CLI deployments.

All three implement the same set of store traits (`AgentTaskStore`,
`ThreadStore`, `MessageProjectionStore`, `TurnAttemptStore`,
`CheckpointStore`, `EventRepository`, `OutboxStore`, `RetentionStore`),
but their operational characteristics differ. This ADR documents those
differences and describes the conformance suite that proves semantic
equivalence across the critical correctness paths.

## Conformance Suite

The conformance suite (`crates/agent-service-host/src/conformance.rs`)
runs identical test logic against each backend, covering:

| # | Invariant | What it tests |
|---|-----------|---------------|
| 1 | Root admission | At most one blocking root per thread |
| 2 | FIFO queueing | Second root gets `Queued`; FIFO order preserved |
| 3 | Lease acquisition | `try_acquire_task` moves `Pending` → `Running` |
| 4 | Heartbeat | Heartbeat extends lease expiry |
| 5 | Lease expiry sweep | Expired leases are requeued |
| 6 | Completed-turn commit | Atomic thread advance + checkpoint creation |
| 7 | Child spawn + resume | Parent pauses; completing child resumes parent |
| 8 | Cancel tree | Cancellation propagates to all descendants |
| 9 | Queued root promotion | Completed root promotes FIFO head |
| 10 | Retry exhaustion | Budget-exhausted row is failed closed |
| 11 | Fail-closed child wakes parent | Recovery-path fail-close propagates to `WaitingOnChildren` parent |
| 12 | Clear with parent-child | `clear()` wipes parent/child chains despite `ON DELETE RESTRICT` self-FKs |

All tests use the same `async fn` test functions parameterised by
`&dyn AgentTaskStore` (and related trait objects), ensuring no backend
gets special-cased test logic.

## Backend-Specific Operational Differences

### Locking model

| Backend | Write serialisation | Row-level locking |
|---------|--------------------|--------------------|
| **In-memory** | `tokio::sync::RwLock` on `Inner` struct | N/A — single in-process lock |
| **SQLite** | Database-level write lock (`BEGIN IMMEDIATE` via WAL mode) | Not available — one writer at a time |
| **PostgreSQL** | Transaction-level via `BEGIN` | `FOR UPDATE` / `FOR UPDATE SKIP LOCKED` per row |

**Implications:** SQLite serialises *all* writers at the database level,
which is acceptable for the single-process desktop/CLI use case. The
in-memory backend serialises via a Tokio `RwLock`, which is equivalent
for single-process. Only PostgreSQL supports true multi-writer
concurrency with row-level locking.

### `acquire_next_runnable` behaviour

| Backend | Scan strategy | Contention handling |
|---------|--------------|---------------------|
| **In-memory** | `BTreeSet` index scan | Lock held for duration of scan |
| **SQLite** | `ORDER BY created_at, id LIMIT 1` full scan | Database-level lock; no contention with self |
| **PostgreSQL** | `ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED` | Skips rows locked by other workers |

**Implications:** `SKIP LOCKED` is a PostgreSQL-only optimisation for
multi-worker pools. The SQLite backend does not need it because only one
process writes at a time.

### Persistence

| Backend | Survives restart | Data location |
|---------|-----------------|---------------|
| **In-memory** | No | Process heap |
| **SQLite** | Yes (file-backed) | Local filesystem (`~/Library/Application Support/agent-sdk/` on macOS, `~/.local/share/agent-sdk/` on Linux) |
| **PostgreSQL** | Yes | External database server |

### Durability surface report

Each backend reports which of the 10 store surfaces are durable
(survive restart) via `StoreRegistry::durability_report()`:

| Surface | In-memory | SQLite | PostgreSQL |
|---------|-----------|--------|------------|
| `task_store` | ❌ | ✅ | ✅ |
| `thread_store` | ❌ | ✅ | ✅ |
| `message_store` | ❌ | ✅ | ✅ |
| `attempt_store` | ❌ | ✅ | ✅ |
| `checkpoint_store` | ❌ | ✅ | ✅ |
| `event_repo` | ❌ | ✅ | ✅ |
| `execution_intent_store` | ❌ | ❌ (in-memory fallback) | ❌ (in-memory fallback) |
| `tool_audit_store` | ❌ | ❌ (in-memory fallback) | ❌ (in-memory fallback) |
| `outbox_store` | ❌ | ✅ | ✅ |
| `retention_store` | ❌ | ✅ | ✅ |

The `execution_intent_store` and `tool_audit_store` remain in-memory
across all backends until dedicated durable implementations are added.

### Clear / reset behaviour

| Backend | `AgentTaskStore::clear()` implementation |
|---------|------------------------------------------|
| **In-memory** | `inner.write().await` + clear all hashmaps |
| **SQLite** | `PRAGMA foreign_keys = OFF`, `DELETE FROM` each table, `PRAGMA foreign_keys = ON` (FK enforcement is suspended for the wipe because `agent_sdk_tasks`'s self-referential `ON DELETE RESTRICT` FKs are checked per-row) |
| **PostgreSQL** | `TRUNCATE TABLE ... CASCADE` |

### Outbox claim behaviour

| Backend | `OutboxStore::claim_pending()` implementation |
|---------|-----------------------------------------------|
| **In-memory** | Walk sorted pending rows, atomically flip under `RwLock` |
| **SQLite** | `SELECT` pending IDs → `UPDATE` each → `SELECT` claimed rows |
| **PostgreSQL** | `UPDATE ... WHERE id IN (SELECT ... FOR UPDATE SKIP LOCKED) RETURNING ...` |

**Implications:** PostgreSQL can claim rows without blocking concurrent
claimers. SQLite serialises all claims through the database-level write
lock, which is fine for single-process but would bottleneck under
multi-process access.

## Design Principle

Backend-specific behaviour is **explicit** in this document and in the
code (`stores.rs` surfaces, SQL dialect comments in `sqlite/store.rs`),
never hidden behind a shared abstraction that silently differs. The
conformance suite proves that the *semantics* are equivalent even when
the *mechanisms* differ.
