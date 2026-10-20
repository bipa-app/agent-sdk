//! `sqlx`-backed `SQLite` implementation of the durable-core stores.
//!
//! This backend mirrors the [`crate::postgres::store::PostgresDurableStore`]
//! semantics with `SQLite` dialect adjustments:
//!
//! - **No row-level locking.** `SQLite` serialises all writes at the
//!   database level. `BEGIN IMMEDIATE` replaces `FOR UPDATE` / `SKIP
//!   LOCKED`.
//! - **Compile-time checked writes.** INSERT / UPDATE / DELETE use
//!   `sqlx::query!()` macros for compile-time SQL validation.  SELECT
//!   queries use runtime `sqlx::query_as::<_, Record>()` with
//!   `#[derive(FromRow)]` because `SQLite`'s weak type system requires
//!   verbose column-level type annotations in `query_as!()`.
//! - **TEXT for timestamps and JSON.** `SQLite` stores `OffsetDateTime`
//!   as ISO 8601 TEXT and JSON values as TEXT.

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{
    ContinuationEnvelope, ListenExecutionContext, QuestionAnswer, TerminalReason, ThreadId,
    TokenUsage, llm,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{FromRow, Sqlite, SqlitePool, Transaction};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use time::OffsetDateTime;

use agent_server::journal::checkpoint::{
    Checkpoint, CheckpointId, CheckpointKind, NewCheckpointParams,
};
use agent_server::journal::checkpoint_store::CheckpointStore;
use agent_server::journal::commit::{
    CommitOutcome, CommitOwnerGuard, CompletedTurnCommit, DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
    LostCommitOwnership, StaleTurnCommit,
};
use agent_server::journal::committed_event::CommittedEvent;
use agent_server::journal::effect::{EffectKey, EffectRecord, EffectRow, EffectState, EffectWrite};
use agent_server::journal::epoch::{EpochId, LeaseRecoveryScope};
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::fork_transaction::ForkCommitParams;
use agent_server::journal::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};
use agent_server::journal::message::{CompactionEntry, MessageProjection};
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::outbox::{
    NewOutboxRow, OutboxRow, OutboxRowId, OutboxStatus, OutboxStore, kind_payload_invariants_hold,
};
use agent_server::journal::outbox_message::{
    OutboxMessage, OutboxMessageKind, TaskWakeupPayload, ThreadEventsAvailablePayload,
};
use agent_server::journal::recovery::{
    FailureReason, RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
use agent_server::journal::relay::{
    TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS, TaskWakeupEmitter, TaskWakeupTrigger,
};
use agent_server::journal::retention::{RetentionCursor, RetentionStore};
use agent_server::journal::root_turn::TerminalRootTurn;
use agent_server::journal::sibling_order::{
    HELD_PROBE_WINDOW, HeldProbe, HeldRow, decoded_wait_list, hold_tool_children_behind_siblings,
    mixed_spawn_waits, pure_spawn_waits,
};
use agent_server::journal::store::{
    AgentTaskStore, CancelScope, CancelTreeOutcome, ChildProbe, MixedChildrenSpawn,
    QuestionAnswerApplied, QuestionPause, ReattachedChild, RequeueOutcome, SubagentBatchSpawn,
    SubagentInvocationSpawn, SubmitDisposition, SubmitRootTurnError, SubmitRootTurnOutcome,
    SubmitRootTurnParams, ToolChildSpawn, apply_question_answer, apply_question_pause,
    cancel_reason, has_precompleted_tool_child, is_owned_by, live_mixed_child_count,
    mixed_child_ids_in_slot_order, new_mixed_tool_child, publish_transactional_mixed_parent,
    release_undelivered_input, settles_undelivered_input, submitted_task_disposition,
};
use agent_server::journal::subagent_spawn_transaction::{
    SubagentSpawnEvent, started_event_from_invocation, subagent_started_event,
};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, SuspensionPayload, TaskKind, TaskStatus, WorkerId,
};
use agent_server::journal::task_state::{SubagentInvocationState, TaskState};
use agent_server::journal::thread::{
    PurgeReceipt, PurgeSeed, Thread, ThreadNotFound, ThreadOperation, ThreadStatus,
};
use agent_server::journal::thread_creation_transaction::{
    ThreadCreationCommit, ThreadCreationCommitted, ThreadCreationRows, ThreadNaming,
};
use agent_server::journal::thread_store::{
    BeginPurge, ThreadCreation, ThreadCreationOutcome, ThreadIdConflict, ThreadStore,
    creation_identity_key,
};
use agent_server::journal::tool_audit::{ToolAuditEvent, ToolAuditEventId, ToolAuditEventStore};
use agent_server::journal::transition::{
    ChildSpawn, Lock, SpawnedChildren, SubagentSpawn, ThreadTier, ToolChildrenSpawn,
};
use agent_server::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId,
};
use agent_server::journal::turn_attempt_store::TurnAttemptStore;

use super::migrations::apply_durable_core_migrations;

/// A completed turn's settle (when it carries one) and the documents its
/// extra writes put.
type CompletedTurnExtra = (
    Option<(AgentTask, Option<AgentTask>)>,
    Vec<agent_server::journal::DocumentOutcome>,
);

mod documents;

// ─────────────────────────────────────────────────────────────────────
// Store struct
// ─────────────────────────────────────────────────────────────────────

/// `sqlx`-backed `SQLite` durable-core store.
///
/// A single instance implements all 10 store traits via the shared
/// connection pool, just like [`crate::postgres::store::PostgresDurableStore`]
/// does for `PostgreSQL`.
#[derive(Clone)]
pub struct SqliteDurableStore {
    pool: SqlitePool,
    held_probe: HeldProbe<HeldKey>,
    /// The epoch the newest committed `Write::BeginEpoch` began; every
    /// lease this store grants after it records it.
    current_epoch: Arc<std::sync::RwLock<Option<EpochId>>>,
    extension: Option<std::sync::Arc<dyn agent_server::journal::TransitionExtension>>,
    /// Write transactions [`Self::begin`] retried after the database stayed
    /// locked past the busy timeout.
    lock_conflict_retries: Arc<AtomicU64>,
}

/// A held row's position in the backstop probe's `(created_at, id)` order,
/// as the stored text so it compares exactly like the scan's `ORDER BY`.
type HeldKey = (String, String);

impl SqliteDurableStore {
    /// Construct the store from an existing pool.
    ///
    /// # Safety contract
    ///
    /// The pool **must** have been configured with `after_connect` hooks
    /// that execute at minimum:
    ///
    /// - `PRAGMA foreign_keys = ON`
    /// - `PRAGMA busy_timeout = 5000` (or equivalent)
    ///
    /// Passing a plain unconfigured pool will silently disable foreign-key
    /// enforcement. Prefer [`connect`](Self::connect) for production use.
    #[must_use]
    pub(crate) fn from_pool(pool: SqlitePool) -> Self {
        Self {
            pool,
            held_probe: HeldProbe::default(),
            current_epoch: Arc::default(),
            extension: None,
            lock_conflict_retries: Arc::default(),
        }
    }

    /// Default pooled connections for a file-backed database when the
    /// caller does not size the pool explicitly.
    ///
    /// Deliberately small: it predates configurable sizing and stays the
    /// compatibility default for one-off handles (CLI helpers, auxiliary
    /// readers). A `ServiceHost` deployment MUST size the pool to its
    /// worker fan-out via [`Self::connect_with_max_connections`] — with a
    /// pool far below the worker count, long journal reads queue every
    /// other caller at pool-acquire, and a starved lease heartbeat that
    /// waits out the acquire queue expires the lease under a live
    /// execution (observed in production as `sqlx` slow-acquire warnings
    /// followed by `lease_expired_budget_exhausted` task failures).
    pub const DEFAULT_MAX_CONNECTIONS: u32 = 4;

    /// Connect to a `SQLite` database and apply the durable-core migrations.
    ///
    /// The `database_url` is a `SQLite` connection string. Use
    /// `"sqlite::memory:"` for an ephemeral in-memory database (useful
    /// for tests) or a file path like `"sqlite:///path/to/agent-sdk.db"`.
    /// The pool is sized at [`Self::DEFAULT_MAX_CONNECTIONS`]; hosts with
    /// a worker pool use [`Self::connect_with_max_connections`].
    ///
    /// For file-backed databases, WAL mode is enabled at connection time
    /// for concurrent read access. In-memory databases do not support WAL
    /// and use their native journal mode instead.
    ///
    /// `PRAGMA foreign_keys = ON` is enforced on every connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot be created, WAL mode cannot be
    /// activated on a file-backed database, or migrations fail.
    pub async fn connect(database_url: &str) -> Result<Self> {
        Self::connect_with_max_connections(database_url, Self::DEFAULT_MAX_CONNECTIONS).await
    }

    /// [`Self::connect`] with an explicit pool ceiling for file-backed
    /// databases.
    ///
    /// Under WAL mode readers run concurrently and only the single
    /// writer serialises (bounded by `PRAGMA busy_timeout`), so the pool
    /// ceiling should track the number of concurrent CALLERS — worker
    /// tasks, sweeps, transports — not a write-parallelism estimate.
    /// An in-memory database always uses one connection regardless of
    /// `max_connections` (`SQLite` in-memory databases are per-connection).
    ///
    /// # Errors
    ///
    /// Returns an error if `max_connections` is zero, the pool cannot be
    /// created, WAL mode cannot be activated on a file-backed database,
    /// or migrations fail.
    pub async fn connect_with_max_connections(
        database_url: &str,
        max_connections: u32,
    ) -> Result<Self> {
        ensure!(max_connections > 0, "sqlite max_connections must be > 0");
        let is_memory = database_url.contains(":memory:");

        let pool = SqlitePoolOptions::new()
            // In-memory databases are per-connection in SQLite. Use a
            // single connection so migrations and queries share the
            // same schema. File-backed databases can use multiple
            // connections under WAL mode.
            .max_connections(if is_memory { 1 } else { max_connections })
            .after_connect(move |conn, _meta| {
                Box::pin(async move {
                    // WAL mode: only attempt on file-backed databases.
                    if !is_memory {
                        let row = sqlx::query("PRAGMA journal_mode = WAL")
                            .fetch_one(&mut *conn)
                            .await?;
                        let mode: String = sqlx::Row::get(&row, 0);
                        if mode != "wal" {
                            return Err(sqlx::Error::Protocol(format!(
                                "failed to enable WAL mode: got '{mode}'"
                            )));
                        }
                    }
                    sqlx::query("PRAGMA foreign_keys = ON")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("PRAGMA busy_timeout = 5000")
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                })
            })
            .connect(database_url)
            .await
            .context("connect sqlite durable store")?;
        let store = Self::from_pool(pool);
        store.migrate().await?;
        // Offline data repair (ENG-9651): run the one-shot, ledgered
        // history-repair sweep after the schema migrations (the ledger
        // table must exist). Every open runs it, possibly while another
        // process is serving turns, so it defers threads with live tasks.
        let report =
            super::repair_sweep::run_startup_repair_sweep(&store.pool, OffsetDateTime::now_utc())
                .await
                .context("startup history-repair sweep")?;
        if report.repaired > 0 || report.unrepairable > 0 {
            tracing::warn!(
                "startup history-repair sweep: scanned={}, ledgered={}, repaired={}, deferred={}, unrepairable={}",
                report.scanned,
                report.ledgered,
                report.repaired,
                report.deferred,
                report.unrepairable
            );
        } else if report.deferred > 0 {
            tracing::info!(
                "startup history-repair sweep: scanned={}, deferred={} (threads whose live turn owns their open tool calls)",
                report.scanned,
                report.deferred
            );
        }
        Ok(store)
    }

    /// Apply the durable-core schema migrations against the pool.
    ///
    /// # Errors
    ///
    /// Returns an error if `sqlx` fails to apply or validate the
    /// embedded migrations.
    pub async fn migrate(&self) -> Result<()> {
        apply_durable_core_migrations(&self.pool).await
    }

    /// Borrow the underlying connection pool.
    #[must_use]
    pub const fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Install the host's transition extension: it runs once per
    /// [`commit`](agent_server::journal::DurableStore::commit) and per
    /// completed turn, and its writes commit in the same transaction.
    #[must_use]
    pub fn with_transition_extension(
        mut self,
        extension: std::sync::Arc<dyn agent_server::journal::TransitionExtension>,
    ) -> Self {
        self.extension = Some(extension);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────

// Defined as a macro (not a `const &str`) so the column list can be
// spliced into `concat!(...)` at every query site, producing a real
// `&'static str` literal that satisfies sqlx 0.9's `SqlSafeStr` bound
// without an `AssertSqlSafe` escape hatch.
macro_rules! task_columns {
    () => {
        "id, kind, status, parent_id, root_id, depth, thread_id, \
         submitted_input_json, caller_metadata_json, worker_id, lease_id, \
         lease_expires_at, last_heartbeat_at, state_json, attempt, max_attempts, \
         last_error, terminal_reason_json, pending_child_count, spawn_index, result_payload, \
         otel_traceparent, created_at, updated_at, completed_at, last_activity_at, \
         after_siblings_json"
    };
}

impl SqliteDurableStore {
    /// `cancel_tree` over `scope`, refused unless `owner` (when given)
    /// still holds `root_id`.
    async fn cancel_scoped(
        &self,
        root_id: &AgentTaskId,
        scope: CancelScope,
        owner: Option<(&WorkerId, &LeaseId)>,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome> {
        let mut tx = self.begin().await?;
        let Some(top) = Self::load_task_tx(&mut tx, root_id).await? else {
            return Err(anyhow!(
                "cancel_tree rejected: task {root_id} does not exist"
            ));
        };
        if let Some((worker, lease)) = owner
            && !is_owned_by(&top, worker, lease)
        {
            return Err(anyhow!(
                "cancel rejected: task {root_id} is not running under this worker and lease"
            ));
        }
        let (outcome, root_turns) = Self::cancel_scoped_tx(&mut tx, root_id, scope, now).await?;
        tx.commit().await.context("commit cancel_tree")?;
        TerminalRootTurn::record_each(root_turns);
        Ok(outcome)
    }

    /// The body of `cancel_scoped`, inside the caller's transaction, with
    /// the root turns it cancelled (recorded once the caller commits).
    async fn cancel_scoped_tx(
        tx: &mut Transaction<'_, Sqlite>,
        root_id: &AgentTaskId,
        scope: CancelScope,
        now: OffsetDateTime,
    ) -> Result<(CancelTreeOutcome, Vec<TerminalRootTurn>)> {
        // Collect the given task + descendants (mirror in-memory
        // `collect_subtree`: children-BFS, root-first, following
        // SubagentInvocation linkage across threads). A mid-tree id
        // cancels only its own subtree instead of no-opping on a
        // `root_id = $1` scan. SQLite serializes writers via
        // BEGIN IMMEDIATE, so there is no lock-order deadlock to avoid;
        // cancelling in BFS order keeps the returned `transitioned` slice
        // identical to the in-memory reference store.
        //
        // Unlike Postgres (READ COMMITTED row locks), `begin()` here
        // takes the database write lock up front (`BEGIN IMMEDIATE`),
        // so no concurrent writer can settle any of these rows between
        // this snapshot and the UPDATEs below — the snapshot IS the
        // transaction's consistent view. Two racing `cancel_tree`
        // calls fully serialize: the loser's snapshot already sees
        // terminal rows and it transitions / emits nothing. A
        // Postgres-style locked re-read would re-read identical data,
        // so this backend intentionally uses snapshot-driven
        // transitions (pinned by the
        // `conformance_sqlite_concurrent_cancels_single_marker` race
        // test).
        let all_tasks = Self::collect_subtree_tx(tx, root_id).await?;
        let candidate_threads: std::collections::HashSet<ThreadId> = all_tasks
            .iter()
            .map(|task| task.thread_id.clone())
            .collect();
        let mut fenced_threads = std::collections::HashSet::new();
        for thread_id in candidate_threads {
            if Self::get_thread_tx(tx, &thread_id)
                .await?
                .is_some_and(|thread| thread.status.is_purge_fenced())
            {
                fenced_threads.insert(thread_id);
            }
        }

        let mut transitioned = Vec::with_capacity(all_tasks.len());
        // Track cancelled root-turn roots (and their threads) so we can
        // wake their linked invocations and promote queued successors.
        let mut cancelled_root_ids: Vec<AgentTaskId> = Vec::new();
        let mut cancelled_root_threads: Vec<ThreadId> = Vec::new();
        // Terminal `Cancelled` markers: one per blocking root this
        // call transitions (pre-cancel occupant of its thread's
        // active-root slot), on that root's OWN thread —
        // cascade-cancelled child-thread roots included, queued roots
        // never. Atomic with the cancel transitions: a crash can
        // neither lose the marker nor emit it without the
        // cancellation, and an idempotent retry (terminal tree) emits
        // nothing.
        let mut markers: Vec<CommittedEvent> = Vec::new();
        let mut root_turns = Vec::new();
        for row in all_tasks {
            let Some(reason) = cancel_reason(scope, root_id, &row) else {
                continue;
            };
            let is_root_turn_root = row.kind == TaskKind::RootTurn && row.is_root();
            let thread_id = row.thread_id.clone();
            if is_root_turn_root && row.status.blocks_root_admission() {
                let continuation_usage = row
                    .state
                    .continuation()
                    .map(|continuation| continuation.payload.total_usage.clone());
                let committed = Self::insert_cancelled_marker_tx(
                    tx,
                    &thread_id,
                    &row.id,
                    continuation_usage,
                    reason.clone(),
                    now,
                )
                .await?;
                markers.push(committed);
            }
            let cancelled = row
                .cancel_with_reason(reason, now)
                .context("cancel_tree: cancel transition failed")?;
            Self::update_task_tx(tx, &cancelled).await?;
            root_turns.extend(TerminalRootTurn::of(&cancelled));
            if is_root_turn_root {
                cancelled_root_ids.push(cancelled.id.clone());
                cancelled_root_threads.push(thread_id);
            }
            transitioned.push(cancelled.id);
        }

        // Phase 7.6: wake linked SubagentInvocation tasks that were
        // WaitingOnChildren on a now-cancelled child root. This
        // mirrors the in-memory store's resume_linked_subagent_invocation.
        // The woken invocation transitions to Pending (not Cancelled),
        // so it must NOT be added to the transitioned vec.
        for cancelled_child_root in &cancelled_root_ids {
            Self::resume_linked_subagent_invocation_tx(tx, cancelled_child_root, now).await?;
        }

        // Each cancelled root frees its thread's active-root slot; promote
        // the next queued root in the same transaction (emitting its
        // durable wakeup) so cancelling the active root never strands its
        // queued successors.
        for thread_id in &cancelled_root_threads {
            if !fenced_threads.contains(thread_id) {
                Self::promote_next_queued_root_tx(tx, thread_id, now).await?;
            }
        }

        // Mid-tree cancel: when the cancelled subtree hangs under a
        // parent OUTSIDE the subtree (e.g. the host's subagent-deadline
        // sweep cancelling a parked root's tool children), recompute
        // that parked parent's pending_child_count so it resumes —
        // mirroring the in-memory store, whose per-row cancel
        // propagates every terminal transition to the row's parent.
        // Only the subtree's TOP row can have an out-of-subtree parent
        // (every deeper row's parent lies inside the subtree and is
        // itself cancelled), so one propagation suffices.
        if transitioned.first().is_some_and(|id| id == root_id)
            && let Some(cancelled_top) = Self::load_task_tx(tx, root_id).await?
            && cancelled_top.parent_id.is_some()
        {
            Self::propagate_terminal_to_parent_tx(tx, &cancelled_top, now, "cancel_tree").await?;
        }

        Ok((
            CancelTreeOutcome {
                transitioned,
                markers,
            },
            root_turns,
        ))
    }

    /// How many times [`Self::begin`] retried a write transaction that
    /// found the database locked.
    #[cfg(test)]
    pub(crate) fn lock_conflict_retries(&self) -> u64 {
        self.lock_conflict_retries.load(Ordering::Relaxed)
    }

    /// Begin a write transaction, retrying a `BEGIN IMMEDIATE` that found
    /// the database locked past the busy timeout ([`BEGIN_BUSY_BACKOFF`]):
    /// nothing has run yet, so a retry is safe for every caller. The last
    /// failure carries [`agent_server::journal::LockConflict`] with the
    /// number of retries.
    async fn begin(&self) -> Result<Transaction<'_, Sqlite>> {
        // BEGIN IMMEDIATE acquires the RESERVED lock up front so that
        // under WAL mode we fail fast with plain SQLITE_BUSY (which
        // busy_timeout retries) rather than SQLITE_BUSY_SNAPSHOT on
        // upgrade from a deferred read snapshot to a write (which
        // busy_timeout does not retry).  See the module-level docs.
        retry_busy(
            &self.lock_conflict_retries,
            "begin a write transaction",
            || self.pool.begin_with("BEGIN IMMEDIATE"),
        )
        .await
        .context("begin sqlite durable-core transaction")
    }

    async fn insert_thread_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<bool> {
        let bootstrap = Thread::new(thread_id.clone(), now);
        let thread_id_key = thread_key(&bootstrap.thread_id);
        let status_wire = enum_to_wire(&bootstrap.status)?;
        let committed_turns = i64::from(bootstrap.committed_turns);
        let input_tokens = i64::from(bootstrap.total_usage.input_tokens);
        let output_tokens = i64::from(bootstrap.total_usage.output_tokens);
        let purge_receipt_json: Option<String> = None;
        let inserted = sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at,
    purge_receipt_json
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT (thread_id) DO NOTHING
",
            thread_id_key,
            status_wire,
            committed_turns,
            input_tokens,
            output_tokens,
            bootstrap.created_at,
            bootstrap.updated_at,
            purge_receipt_json,
        )
        .execute(&mut **tx)
        .await
        .context("insert thread row")?;
        Ok(inserted.rows_affected() == 1)
    }

    async fn append_compaction_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        result_messages: Vec<llm::Message>,
        source_message_count: usize,
        retained_message_count: usize,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        Self::bootstrap_appendable_thread_tx(tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(tx, thread_id, now).await?;
        let updated = projection.append_compaction(
            result_messages,
            source_message_count,
            retained_message_count,
            now,
        )?;
        Self::upsert_message_head_tx(tx, &updated).await?;
        Ok(updated)
    }

    async fn bootstrap_thread_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<()> {
        let _inserted = Self::insert_thread_row_tx(tx, thread_id, now).await?;
        Ok(())
    }

    /// The epoch new leases record, if one has begun.
    fn current_epoch(&self) -> Option<EpochId> {
        self.current_epoch.read().map_or_else(
            |poisoned| poisoned.into_inner().clone(),
            |epoch| epoch.clone(),
        )
    }

    fn set_current_epoch(&self, epoch: EpochId) {
        let mut current = self
            .current_epoch
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *current = Some(epoch);
    }

    async fn insert_epoch_tx(
        tx: &mut Transaction<'_, Sqlite>,
        record: &agent_server::journal::EpochRecord,
    ) -> Result<()> {
        let (id, label, recovery) = (
            record.id.as_str(),
            record.label.as_deref(),
            record.recovery.as_deref(),
        );
        sqlx::query!(
            "INSERT INTO agent_sdk_host_epochs (epoch_id, started_at, label, recovery) \
             VALUES (?1, ?2, ?3, ?4)",
            id,
            record.started_at,
            label,
            recovery,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("begin epoch {}", record.id))?;
        if record.is_exclusive() {
            let [single_host, postgres_leader] = agent_server::journal::EXCLUSIVE_RECOVERY;
            let id = record.id.as_str();
            sqlx::query!(
                "UPDATE agent_sdk_host_epochs SET superseded_at = ?1 \
                 WHERE epoch_id <> ?2 AND superseded_at IS NULL AND recovery IN (?3, ?4)",
                record.started_at,
                id,
                single_host,
                postgres_leader,
            )
            .execute(&mut **tx)
            .await
            .with_context(|| format!("supersede the exclusive epochs before {}", record.id))?;
        }
        Ok(())
    }

    /// Whether `epoch` was superseded by a newer exclusive epoch.
    async fn epoch_superseded_tx(
        tx: &mut Transaction<'_, Sqlite>,
        epoch: &EpochId,
    ) -> Result<bool> {
        let id = epoch.as_str();
        let superseded = sqlx::query_scalar!(
            r#"SELECT superseded_at IS NOT NULL AS "superseded!: bool" FROM agent_sdk_host_epochs WHERE epoch_id = ?1"#,
            id,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("load whether epoch {epoch} is superseded"))?;
        Ok(superseded.unwrap_or(false))
    }

    /// Whether `task`'s lease was granted under a superseded epoch.
    async fn lease_superseded_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTaskId,
    ) -> Result<bool> {
        let id = task.as_str();
        let superseded = sqlx::query_scalar!(
            r#"SELECT e.superseded_at IS NOT NULL AS "superseded!: bool" FROM agent_sdk_tasks t
               JOIN agent_sdk_host_epochs e ON e.epoch_id = t.lease_epoch WHERE t.id = ?1"#,
            id,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("load whether the lease epoch of {task} is superseded"))?;
        Ok(superseded.unwrap_or(false))
    }

    /// Record that `task`'s new lease was granted under `epoch`. Refused,
    /// rolling the grant back, when a newer exclusive epoch superseded
    /// `epoch`: the process no longer leads its store.
    async fn stamp_lease_epoch_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTaskId,
        epoch: Option<&EpochId>,
    ) -> Result<()> {
        if let Some(epoch) = epoch
            && Self::epoch_superseded_tx(tx, epoch).await?
        {
            return Err(agent_server::journal::EpochSuperseded::new(epoch).into());
        }
        let (epoch, id) = (epoch.map(EpochId::as_str), task.as_str());
        sqlx::query!(
            "UPDATE agent_sdk_tasks SET lease_epoch = ?1 WHERE id = ?2",
            epoch,
            id,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("stamp the lease epoch of {task}"))?;
        Ok(())
    }

    async fn bootstrap_appendable_thread_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<()> {
        Self::bootstrap_thread_row_tx(tx, thread_id, now).await?;
        crate::fence_signal::fence_refused(
            agent_server::journal::thread::FencedWriter::MessageHead,
            Self::require_active_thread_tx(tx, thread_id, ThreadOperation::Append).await,
        )
    }

    async fn require_appendable_thread_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        writer: agent_server::journal::thread::FencedWriter,
    ) -> Result<()> {
        let thread = Self::get_thread_tx(tx, thread_id).await?;
        crate::fence_signal::fence_refused(
            writer,
            agent_server::journal::Fence::APPEND.check(thread_id, thread.as_ref()),
        )
    }

    async fn claim_thread_creation_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        creation: &ThreadCreation,
        now: OffsetDateTime,
    ) -> Result<ThreadCreationOutcome> {
        let inserted_thread = Self::insert_thread_row_tx(tx, thread_id, now).await?;
        let identity_key = creation_identity_key(thread_id);
        let kind = creation.idempotency_kind();
        let fingerprint = creation.fingerprint();
        let result_json = serde_json::json!({ "thread_id": thread_id.0 });
        let result_text = serde_json::to_string(&result_json)
            .context("encode caller-minted thread identity result")?;

        if inserted_thread {
            let inserted_identity = sqlx::query!(
                r#"INSERT INTO agent_sdk_idempotency
                       (request_id, kind, fingerprint, result_json, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5)
                   ON CONFLICT (request_id) DO NOTHING"#,
                identity_key,
                kind.as_str(),
                fingerprint,
                result_text,
                now,
            )
            .execute(&mut **tx)
            .await
            .with_context(|| format!("claim creation identity for {thread_id}"))?;
            if inserted_identity.rows_affected() == 1 {
                return Ok(ThreadCreationOutcome::Created);
            }
            return Err(anyhow::Error::new(ThreadIdConflict {
                thread_id: thread_id.clone(),
            }));
        }

        let row = sqlx::query!(
            r#"SELECT kind, fingerprint, result_json
               FROM agent_sdk_idempotency WHERE request_id = ?1"#,
            identity_key,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("load creation identity for {thread_id}"))?;
        if let Some(row) = row {
            let stored_result: serde_json::Value = serde_json::from_str(&row.result_json)
                .context("decode caller-minted thread identity result")?;
            if row.kind == kind.as_str()
                && row.fingerprint == fingerprint
                && stored_result == result_json
            {
                return Ok(ThreadCreationOutcome::Existing);
            }
        }
        Err(anyhow::Error::new(ThreadIdConflict {
            thread_id: thread_id.clone(),
        }))
    }

    async fn get_thread_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<Option<Thread>> {
        let record = sqlx::query_as::<_, ThreadRecord>(
            r"
SELECT thread_id, status, committed_turns,
       total_input_tokens, total_output_tokens, created_at, updated_at,
       purge_receipt_json
FROM agent_sdk_threads
WHERE thread_id = ?1
",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("get thread {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn require_active_thread_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        operation: ThreadOperation,
    ) -> Result<()> {
        let thread = Self::get_thread_tx(tx, thread_id).await?.ok_or_else(|| {
            anyhow::Error::new(ThreadNotFound {
                thread_id: thread_id.clone(),
            })
        })?;
        thread.require_active(operation)?;
        Ok(())
    }

    async fn get_thread_pool(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
        let record = sqlx::query_as::<_, ThreadRecord>(
            r"
SELECT thread_id, status, committed_turns,
       total_input_tokens, total_output_tokens, created_at, updated_at,
       purge_receipt_json
FROM agent_sdk_threads
WHERE thread_id = ?1
",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get thread {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn upsert_thread_tx(tx: &mut Transaction<'_, Sqlite>, thread: &Thread) -> Result<()> {
        let thread_id_key = thread_key(&thread.thread_id);
        let status_wire = enum_to_wire(&thread.status)?;
        let committed_turns = i64::from(thread.committed_turns);
        let input_tokens = i64::from(thread.total_usage.input_tokens);
        let output_tokens = i64::from(thread.total_usage.output_tokens);
        let purge_receipt_json = thread
            .purge_record()
            .map(|record| serde_json::to_string(&record))
            .transpose()
            .context("encode purge record")?;
        sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at,
    purge_receipt_json
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT (thread_id) DO UPDATE SET
    status = excluded.status,
    committed_turns = excluded.committed_turns,
    total_input_tokens = excluded.total_input_tokens,
    total_output_tokens = excluded.total_output_tokens,
    updated_at = excluded.updated_at,
    purge_receipt_json = excluded.purge_receipt_json
",
            thread_id_key,
            status_wire,
            committed_turns,
            input_tokens,
            output_tokens,
            thread.created_at,
            thread.updated_at,
            purge_receipt_json,
        )
        .execute(&mut **tx)
        .await
        .context("upsert thread")?;
        Ok(())
    }

    async fn get_message_head_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        if let Some(projection) = read_message_head(&mut **tx, thread_id).await? {
            Ok(projection)
        } else {
            let fresh = MessageProjection::new(thread_id.clone(), now);
            // Use INSERT … ON CONFLICT DO NOTHING (not the destructive
            // upsert) so a concurrent commit that already wrote a real
            // projection is never overwritten with an empty bootstrap row.
            // Matches the Postgres lock_message_head_tx behaviour.
            let thread_id_key = thread_key(thread_id);
            let history_json = json_to_value(&fresh.messages, "message head bootstrap history")?;
            let history_generation = i64_from_u64(
                fresh.history_generation,
                "message head bootstrap history generation",
            )?;
            let version = i64_from_u64(fresh.version, "message head bootstrap version")?;
            // Bootstrap the row with `draft_messages_json = NULL` —
            // a fresh thread has no in-flight turn. The full upsert
            // path (`upsert_message_head_tx`) carries the actual
            // draft once a suspension boundary fires.
            sqlx::query(
                r"
INSERT INTO agent_sdk_message_heads (thread_id, history_json, draft_messages_json, compactions_json, history_generation, version, created_at, updated_at)
VALUES (?1, ?2, NULL, NULL, ?3, ?4, ?5, ?6)
ON CONFLICT (thread_id) DO NOTHING
",
            )
            .bind(thread_id_key)
            .bind(history_json)
            .bind(history_generation)
            .bind(version)
            .bind(fresh.created_at)
            .bind(fresh.updated_at)
            .execute(&mut **tx)
            .await
            .context("bootstrap message head")?;
            Ok(fresh)
        }
    }

    async fn get_message_head_pool(
        &self,
        thread_id: &ThreadId,
    ) -> Result<Option<MessageProjection>> {
        read_message_head(&self.pool, thread_id).await
    }

    async fn upsert_message_head_tx(
        tx: &mut Transaction<'_, Sqlite>,
        projection: &MessageProjection,
    ) -> Result<()> {
        let thread_id_key = thread_key(&projection.thread_id);
        let history_json = json_to_value(&projection.messages, "message head history")?;
        // Persist `draft_messages` as NULL when empty so the column
        // distinguishes "no in-flight turn" from "explicit empty
        // draft" — matches the intent the migration documents.
        let draft_messages_json = if projection.draft_messages.is_empty() {
            None
        } else {
            Some(json_to_value(
                &projection.draft_messages,
                "message head draft messages",
            )?)
        };
        let compactions_json = if projection.compactions.is_empty() {
            None
        } else {
            Some(json_to_value(
                &projection.compactions,
                "message head compactions",
            )?)
        };
        let history_generation = i64_from_u64(
            projection.history_generation,
            "message head history generation",
        )?;
        let version = i64_from_u64(projection.version, "message head version")?;
        sqlx::query(
            r"
INSERT INTO agent_sdk_message_heads (thread_id, history_json, draft_messages_json, compactions_json, history_generation, version, created_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = excluded.history_json,
    draft_messages_json = excluded.draft_messages_json,
    compactions_json = excluded.compactions_json,
    history_generation = excluded.history_generation,
    version = excluded.version,
    updated_at = excluded.updated_at
",
        )
        .bind(thread_id_key)
        .bind(history_json)
        .bind(draft_messages_json)
        .bind(compactions_json)
        .bind(history_generation)
        .bind(version)
        .bind(projection.created_at)
        .bind(projection.updated_at)
        .execute(&mut **tx)
        .await
        .context("upsert message head")?;
        Ok(())
    }

    async fn insert_message_commit_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        turn_number: u32,
        task_id: &AgentTaskId,
        version: u64,
        messages: &[llm::Message],
        now: OffsetDateTime,
    ) -> Result<()> {
        let thread_id_key = thread_key(thread_id);
        let turn_number_i64 = i64::from(turn_number);
        let task_id_str = task_id.as_str();
        let head_version_after = i64_from_u64(version, "message commit version")?;
        let batch_json = json_to_value(messages, "message commit messages")?;
        sqlx::query!(
            r"
INSERT INTO agent_sdk_message_commits
    (thread_id, turn_number, task_id, head_version_after, batch_json, committed_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
",
            thread_id_key,
            turn_number_i64,
            task_id_str,
            head_version_after,
            batch_json,
            now,
        )
        .execute(&mut **tx)
        .await
        .context("insert message commit")?;
        Ok(())
    }

    async fn get_attempt_pool(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        let record = sqlx::query_as::<_, TurnAttemptRecord>(
            r"
SELECT id, task_id, attempt_number, provider, requested_model,
       request_blob, response_blob, response_id, response_model,
       stop_reason, outcome, input_tokens, output_tokens,
       cached_input_tokens, cache_creation_input_tokens, route_provider,
       thinking_mode, thinking_budget_tokens, thinking_effort, opened_at, closed_at, duration_ms,
       otel_trace_id, otel_span_id
FROM agent_sdk_turn_attempts
WHERE id = ?1
",
        )
        .bind(id.as_str())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get attempt {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_attempt_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &TurnAttemptId,
    ) -> Result<Option<TurnAttempt>> {
        let record = sqlx::query_as::<_, TurnAttemptRecord>(
            r"
SELECT id, task_id, attempt_number, provider, requested_model,
       request_blob, response_blob, response_id, response_model,
       stop_reason, outcome, input_tokens, output_tokens,
       cached_input_tokens, cache_creation_input_tokens, route_provider,
       thinking_mode, thinking_budget_tokens, thinking_effort, opened_at, closed_at, duration_ms,
       otel_trace_id, otel_span_id
FROM agent_sdk_turn_attempts
WHERE id = ?1
",
        )
        .bind(id.as_str())
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("lock attempt {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn insert_attempt_tx(
        tx: &mut Transaction<'_, Sqlite>,
        attempt: &TurnAttempt,
    ) -> Result<()> {
        let id = attempt.id.as_str();
        let task_id = attempt.task_id.as_str();
        let attempt_number = i64::from(attempt.attempt_number);
        let response_id = attempt.response_id.as_deref();
        let response_model = attempt.response_model.as_deref();
        let stop_reason = optional_enum_to_wire(attempt.stop_reason.as_ref())?;
        let outcome = optional_enum_to_wire(attempt.outcome.as_ref())?;
        let input_tokens = attempt.input_tokens.map(i64::from);
        let output_tokens = attempt.output_tokens.map(i64::from);
        let cached_input_tokens = attempt.cached_input_tokens.map(i64::from);
        let cache_creation_input_tokens = attempt.cache_creation_input_tokens.map(i64::from);
        let route_provider = attempt.route_provider.as_deref();
        let thinking_mode = optional_enum_to_wire(attempt.thinking_mode.as_ref())?;
        let thinking_budget_tokens = attempt.thinking_budget_tokens.map(i64::from);
        let thinking_effort = optional_enum_to_wire(attempt.thinking_effort.as_ref())?;
        let duration_ms = attempt
            .duration_ms
            .map(|v| i64::try_from(v).context("duration_ms exceeds i64::MAX"))
            .transpose()?;
        let otel_trace_id = attempt.otel_trace_id.as_deref();
        let otel_span_id = attempt.otel_span_id.as_deref();
        sqlx::query!(
            r"
INSERT INTO agent_sdk_turn_attempts (
    id, task_id, attempt_number, provider, requested_model,
    request_blob, response_blob, response_id, response_model,
    stop_reason, outcome, input_tokens, output_tokens,
    cached_input_tokens, cache_creation_input_tokens, route_provider,
    thinking_mode, thinking_budget_tokens, thinking_effort, opened_at, closed_at, duration_ms,
    otel_trace_id, otel_span_id
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
",
            id,
            task_id,
            attempt_number,
            attempt.provider,
            attempt.requested_model,
            attempt.request_blob,
            attempt.response_blob,
            response_id,
            response_model,
            stop_reason,
            outcome,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            route_provider,
            thinking_mode,
            thinking_budget_tokens,
            thinking_effort,
            attempt.opened_at,
            attempt.closed_at,
            duration_ms,
            otel_trace_id,
            otel_span_id,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert attempt {}", attempt.id))?;
        Ok(())
    }

    async fn update_attempt_tx(
        tx: &mut Transaction<'_, Sqlite>,
        attempt: &TurnAttempt,
    ) -> Result<()> {
        let id = attempt.id.as_str();
        let response_id = attempt.response_id.as_deref();
        let response_model = attempt.response_model.as_deref();
        let stop_reason = optional_enum_to_wire(attempt.stop_reason.as_ref())?;
        let outcome = optional_enum_to_wire(attempt.outcome.as_ref())?;
        let input_tokens = attempt.input_tokens.map(i64::from);
        let output_tokens = attempt.output_tokens.map(i64::from);
        let cached_input_tokens = attempt.cached_input_tokens.map(i64::from);
        let cache_creation_input_tokens = attempt.cache_creation_input_tokens.map(i64::from);
        let route_provider = attempt.route_provider.as_deref();
        let thinking_mode = optional_enum_to_wire(attempt.thinking_mode.as_ref())?;
        let thinking_budget_tokens = attempt.thinking_budget_tokens.map(i64::from);
        let thinking_effort = optional_enum_to_wire(attempt.thinking_effort.as_ref())?;
        let duration_ms = attempt
            .duration_ms
            .map(|v| i64::try_from(v).context("duration_ms exceeds i64::MAX"))
            .transpose()?;
        let otel_trace_id = attempt.otel_trace_id.as_deref();
        let otel_span_id = attempt.otel_span_id.as_deref();
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_turn_attempts SET
    response_blob = ?2, response_id = ?3, response_model = ?4,
    stop_reason = ?5, outcome = ?6, input_tokens = ?7, output_tokens = ?8,
    cached_input_tokens = ?9, cache_creation_input_tokens = ?10,
    route_provider = ?11, thinking_mode = ?12, thinking_budget_tokens = ?13,
    thinking_effort = ?14, closed_at = ?15, duration_ms = ?16,
    otel_trace_id = ?17, otel_span_id = ?18
WHERE id = ?1
",
            id,
            attempt.response_blob,
            response_id,
            response_model,
            stop_reason,
            outcome,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            route_provider,
            thinking_mode,
            thinking_budget_tokens,
            thinking_effort,
            attempt.closed_at,
            duration_ms,
            otel_trace_id,
            otel_span_id,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("update attempt {}", attempt.id))?;
        ensure!(
            result.rows_affected() == 1,
            "update attempt affected {} rows",
            result.rows_affected()
        );
        Ok(())
    }

    async fn get_checkpoint_pool(&self, id: &CheckpointId) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as::<_, CheckpointRecord>(
            r"
SELECT id, thread_id, turn_number, task_id, messages_json,
       agent_state_snapshot, turn_input_tokens, turn_output_tokens, kind, created_at
FROM agent_sdk_turn_checkpoints
WHERE id = ?1
",
        )
        .bind(id.as_str())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get checkpoint {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn insert_checkpoint_tx(
        tx: &mut Transaction<'_, Sqlite>,
        checkpoint: &Checkpoint,
    ) -> Result<()> {
        let id = checkpoint.id.as_str();
        let thread_id_key = thread_key(&checkpoint.thread_id);
        let turn_number = i64::from(checkpoint.turn_number);
        let task_id = checkpoint.task_id.as_str();
        let messages_json = json_to_value(&checkpoint.messages, "checkpoint messages")?;
        let turn_input_tokens = i64::from(checkpoint.turn_usage.input_tokens);
        let turn_output_tokens = i64::from(checkpoint.turn_usage.output_tokens);
        let kind = checkpoint.kind.as_str();
        sqlx::query!(
            r"
INSERT INTO agent_sdk_turn_checkpoints (
    id, thread_id, turn_number, task_id, messages_json,
    agent_state_snapshot, turn_input_tokens, turn_output_tokens, kind, created_at
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
",
            id,
            thread_id_key,
            turn_number,
            task_id,
            messages_json,
            checkpoint.agent_state_snapshot,
            turn_input_tokens,
            turn_output_tokens,
            kind,
            checkpoint.created_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert checkpoint {}", checkpoint.id))?;
        Ok(())
    }

    // ── Task helpers ─────────────────────────────────────────────────

    async fn load_task_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE id = ?1",
        ))
        .bind(id.as_str())
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("load task {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_task_pool(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE id = ?1",
        ))
        .bind(id.as_str())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get task {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_task_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE id = ?1",
        ))
        .bind(id.as_str())
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("get task {id} (tx)"))?;
        record.map(TryInto::try_into).transpose()
    }

    /// Count the queued (not active/blocking) root turns on a thread,
    /// inside the supplied transaction.
    async fn queued_root_count_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<u32> {
        let key = thread_key(thread_id);
        let count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status = 'queued'",
            key,
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("count queued roots for {thread_id}"))?;
        Ok(u32::try_from(count).unwrap_or(u32::MAX))
    }

    /// Look up an existing `SubmitWork` idempotency record inside the
    /// admission transaction and, if it matches, build the replay
    /// outcome. Returns `Ok(None)` when the key is unused and
    /// [`SubmitRootTurnError::IdempotencyConflict`] on a mismatch.
    async fn try_replay_submit_tx(
        tx: &mut Transaction<'_, Sqlite>,
        claim: &agent_server::journal::store::SubmitRootIdempotency,
    ) -> std::result::Result<Option<SubmitRootTurnOutcome>, SubmitRootTurnError> {
        let existing = sqlx::query!(
            r#"SELECT kind, fingerprint, result_json FROM agent_sdk_idempotency WHERE request_id = ?1"#,
            claim.request_id,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("idempotency lookup for {}", claim.request_id))
        .map_err(SubmitRootTurnError::Other)?;

        let Some(row) = existing else {
            return Ok(None);
        };
        if row.kind != IdempotencyKind::SubmitWork.as_str() || row.fingerprint != claim.fingerprint
        {
            return Err(SubmitRootTurnError::IdempotencyConflict);
        }
        let result_json: serde_json::Value = serde_json::from_str(&row.result_json)
            .context("decode stored idempotency result_json")
            .map_err(SubmitRootTurnError::Other)?;
        let task_id = result_json
            .get("task_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("submit idempotency record missing task_id reference"))
            .map_err(SubmitRootTurnError::Other)?;
        let admitted = Self::get_task_tx(tx, &AgentTaskId::from_string(task_id))
            .await
            .map_err(SubmitRootTurnError::Other)?
            .ok_or_else(|| anyhow!("idempotent submit record points at missing task"))
            .map_err(SubmitRootTurnError::Other)?;
        let queued_depth = Self::queued_root_count_tx(tx, &admitted.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;
        Ok(Some(SubmitRootTurnOutcome {
            disposition: submitted_task_disposition(&admitted),
            task: admitted,
            replayed: true,
            replayed_result: Some(result_json),
            queued_depth,
        }))
    }

    /// Insert the `SubmitWork` idempotency record inside the admission
    /// transaction (`result_json` stored as TEXT per the `SQLite` dialect).
    async fn claim_submit_idempotency_tx(
        tx: &mut Transaction<'_, Sqlite>,
        claim: &agent_server::journal::store::SubmitRootIdempotency,
        created_at: OffsetDateTime,
    ) -> Result<()> {
        let kind_wire = IdempotencyKind::SubmitWork.as_str();
        let result_text =
            serde_json::to_string(&claim.result_json).context("encode idempotency result_json")?;
        let fingerprint = claim.fingerprint.clone();
        let request_id = claim.request_id.clone();
        sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES (?1, ?2, ?3, ?4, ?5)"#,
            request_id,
            kind_wire,
            fingerprint,
            result_text,
            created_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert idempotency record for {request_id}"))?;
        Ok(())
    }

    /// Admit a fresh root turn, claim its idempotency key, and commit
    /// the transaction. `would_queue` is the admission decision computed
    /// under the `BEGIN IMMEDIATE` write lock.
    async fn commit_fresh_admission_tx(
        mut tx: Transaction<'_, Sqlite>,
        task: AgentTask,
        injection_parent: Option<AgentTask>,
        would_queue: bool,
        idempotency: Option<agent_server::journal::store::SubmitRootIdempotency>,
    ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
        let admitted = if let Some(parent) = injection_parent {
            task.into_input_injection(&parent)
                .context("submit injection rejected: invalid parent linkage")
                .map_err(SubmitRootTurnError::Other)?
        } else if would_queue {
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")
                .map_err(SubmitRootTurnError::Other)?
        } else {
            task
        };

        Self::insert_task_tx(&mut tx, &admitted)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        // Phase 10 · D: emit a durable `task_wakeup` advisory row in the
        // SAME transaction when the new root is immediately runnable, so a
        // worker (even in another process) is nudged to call
        // `acquire_next_runnable` even after this host dies. A queued root
        // is parked and is nudged on promotion instead. This mirrors the
        // non-idempotent `submit_root_turn` path; the transport calls this
        // idempotent variant, so the wakeup emit must live here too.
        if admitted.status == TaskStatus::Pending {
            Self::insert_task_wakeup_outbox_row_tx(
                &mut tx,
                &admitted.id,
                &admitted.thread_id,
                TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS,
                admitted.created_at,
            )
            .await
            .map_err(SubmitRootTurnError::Other)?;
        }

        if let Some(claim) = idempotency {
            Self::claim_submit_idempotency_tx(&mut tx, &claim, admitted.created_at)
                .await
                .map_err(SubmitRootTurnError::Other)?;
        }

        let queued_depth = Self::queued_root_count_tx(&mut tx, &admitted.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        tx.commit()
            .await
            .context("commit submit_root_turn_idempotent")
            .map_err(SubmitRootTurnError::Other)?;
        Ok(SubmitRootTurnOutcome {
            disposition: submitted_task_disposition(&admitted),
            task: admitted,
            replayed: false,
            replayed_result: None,
            queued_depth,
        })
    }

    async fn insert_task_tx(tx: &mut Transaction<'_, Sqlite>, task: &AgentTask) -> Result<()> {
        let id = task.id.as_str();
        let kind = enum_to_wire(&task.kind)?;
        let status = enum_to_wire(&task.status)?;
        let parent_id = task.parent_id.as_ref().map(AgentTaskId::as_str);
        let root_id = task.root_id.as_str();
        let depth = i64::from(task.depth);
        let thread_id_key = thread_key(&task.thread_id);
        let submitted_input_json = json_to_value(&task.submitted_input, "task submitted input")?;
        let caller_metadata_json = task.caller_metadata.clone();
        let worker_id = task.worker_id.as_ref().map(WorkerId::as_str);
        let lease_id = task.lease_id.as_ref().map(LeaseId::as_str);
        let state_json = json_to_value(&task.state, "task state")?;
        let attempt = i64::from(task.attempt);
        let max_attempts = i64::from(task.max_attempts);
        let terminal_reason = task
            .terminal_reason
            .as_ref()
            .map(|reason| json_to_value(reason, "task terminal reason"))
            .transpose()?;
        let pending_child_count = i64::from(task.pending_child_count);
        let spawn_index = task.spawn_index.map(i64::from);
        let after_siblings_json = after_siblings_json(task)?;
        sqlx::query!(
            r"
INSERT INTO agent_sdk_tasks (
    id, kind, status, parent_id, root_id, depth, thread_id,
    submitted_input_json, caller_metadata_json, worker_id, lease_id, lease_expires_at,
    last_heartbeat_at, state_json, attempt, max_attempts, last_error, terminal_reason_json,
    pending_child_count, spawn_index, result_payload,
    created_at, updated_at, completed_at, otel_traceparent, last_activity_at,
    after_siblings_json
) VALUES (
    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
    ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
)
",
            id,
            kind,
            status,
            parent_id,
            root_id,
            depth,
            thread_id_key,
            submitted_input_json,
            caller_metadata_json,
            worker_id,
            lease_id,
            task.lease_expires_at,
            task.last_heartbeat_at,
            state_json,
            attempt,
            max_attempts,
            task.last_error,
            terminal_reason,
            pending_child_count,
            spawn_index,
            task.result_payload,
            task.created_at,
            task.updated_at,
            task.completed_at,
            task.otel_traceparent,
            task.last_activity_at,
            after_siblings_json,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert task {}", task.id))?;
        Ok(())
    }

    /// The siblings `task` waits for that exist and are not yet terminal,
    /// the same check the acquisition scan makes.
    async fn load_unfinished_siblings_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTask,
    ) -> Result<BTreeSet<AgentTaskId>> {
        let Some(waited) = after_siblings_json(task)? else {
            return Ok(BTreeSet::new());
        };
        Ok(sqlx::query_scalar!(
            r"SELECT sibling.id AS id FROM json_each(?1) AS waited JOIN agent_sdk_tasks AS sibling ON sibling.id = waited.value WHERE sibling.status NOT IN ('completed', 'failed', 'cancelled')",
            waited,
        )
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("load the unfinished siblings {} waits for", task.id))?
        .into_iter()
        .flatten()
        .map(AgentTaskId::from_string)
        .collect())
    }

    /// Lease the oldest pending row that waits on nothing.
    async fn acquire_ready_head(
        &self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        // SQLite has no SKIP LOCKED; the single-process model means
        // database-level locking is sufficient. We loop to skip
        // fail-closed rows, same as the Postgres backend.
        loop {
            let mut tx = self.begin().await?;
            // Threads mid-purge are excluded IN the scan: one fenced
            // thread's oldest pending row must never stall every other
            // thread's acquires (its rows are cancelled by the purge).
            // SQLite has no statistics to choose the ready index over
            // agent_sdk_tasks_by_status_idx, which would walk every held
            // row, so the scan names it (see migration 0026).
            let record = sqlx::query_as::<_, TaskRecord>(concat!(
                "SELECT ",
                task_columns!(),
                " FROM agent_sdk_tasks INDEXED BY agent_sdk_tasks_ready_fifo_idx \
                 WHERE status >= 'pending' AND status <= 'pending' \
                 AND after_siblings_json IS NULL \
                 AND thread_id NOT IN (SELECT thread_id FROM agent_sdk_threads \
                 WHERE status IN ('deleting', 'deleted')) \
                 ORDER BY created_at, id LIMIT 1",
            ))
            .fetch_optional(&mut *tx)
            .await
            .context("load runnable head")?;
            let Some(record) = record else {
                return Ok(None);
            };
            let old = AgentTask::try_from(record)?;
            if !old.status.can_be_leased() {
                let status = old.status;
                return Err(anyhow!(
                    "acquire_next_runnable: runnable index held non-pending row {} in status {status:?}",
                    old.id
                ));
            }

            let unfinished = Self::load_unfinished_siblings_tx(&mut tx, &old).await?;
            match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
                RecoveryAction::NoAction => {
                    let claimed = old
                        .clone()
                        .admit_after_siblings(&unfinished)
                        .and_then(|admitted| {
                            admitted.mark_running(worker.clone(), lease.clone(), expires_at, now)
                        })
                        .context(
                            "acquire_next_runnable rejected: mark_running transition failed",
                        )?;
                    Self::update_task_tx(&mut tx, &claimed).await?;
                    Self::stamp_lease_epoch_tx(&mut tx, &claimed.id, self.current_epoch().as_ref())
                        .await?;
                    tx.commit().await.context("commit acquire_next_runnable")?;
                    return Ok(Some(claimed));
                }
                RecoveryAction::FailClosed(reason) => {
                    let (_settled, root_turns) =
                        Self::fail_closed_tx(&mut tx, &old, reason, now, "acquire_next_runnable")
                            .await?;
                    tx.commit()
                        .await
                        .context("commit fail-closed runnable head")?;
                    TerminalRootTurn::record_each(root_turns);
                }
                RecoveryAction::Requeue => {
                    return Err(anyhow!(
                        "acquire_next_runnable: recovery matrix produced Requeue for acquisition-time row {}",
                        old.id
                    ));
                }
            }
        }
    }

    /// Lease a held row whose siblings have all finished, examining one
    /// bounded window of held rows from the probe's cursor.
    async fn acquire_ready_held(
        &self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let cursor = self.held_probe.cursor();
        let after_cursor = self
            .load_held_window(cursor.as_ref(), HELD_PROBE_WINDOW)
            .await?;
        let from_start = if cursor.is_some() && after_cursor.len() < HELD_PROBE_WINDOW {
            self.load_held_window(None, HELD_PROBE_WINDOW - after_cursor.len())
                .await?
        } else {
            Vec::new()
        };
        for id in self
            .held_probe
            .advance(cursor.is_some(), after_cursor, from_start)
        {
            match AgentTaskStore::try_acquire_task(
                self,
                &id,
                worker.clone(),
                lease.clone(),
                expires_at,
                now,
            )
            .await
            {
                Ok(Some(task)) => return Ok(Some(task)),
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    task_id = %id,
                    error = %format!("{error:#}"),
                    "held probe skipped a row it could not lease"
                ),
            }
        }
        Ok(None)
    }

    /// Up to `limit` held rows after `after` (from the start when `None`)
    /// in `(created_at, id)` order, each with whether every sibling it
    /// waits for is terminal or gone. Rows of threads mid-purge are left
    /// out, as in the ready scan.
    async fn load_held_window(
        &self,
        after: Option<&HeldKey>,
        limit: usize,
    ) -> Result<Vec<HeldRow<HeldKey>>> {
        let limit = i64::try_from(limit).context("held probe window exceeds i64")?;
        let rows: Vec<(String, String, bool)> = match after {
            Some((created_at, id)) => sqlx::query!(
                r#"SELECT held.id AS "id!", held.created_at AS "created_at!", NOT EXISTS (SELECT 1 FROM json_each(held.after_siblings_json) AS waited JOIN agent_sdk_tasks AS sibling ON sibling.id = waited.value WHERE sibling.status NOT IN ('completed', 'failed', 'cancelled')) AS "ready!: bool" FROM agent_sdk_tasks AS held INDEXED BY agent_sdk_tasks_held_fifo_idx WHERE held.status >= 'pending' AND held.status <= 'pending' AND held.after_siblings_json IS NOT NULL AND held.kind = 'tool_runtime' AND (held.created_at, held.id) > (?1, ?2) AND held.thread_id NOT IN (SELECT thread_id FROM agent_sdk_threads WHERE status IN ('deleting', 'deleted')) ORDER BY held.created_at, held.id LIMIT ?3"#,
                created_at,
                id,
                limit,
            )
            .fetch_all(&self.pool)
            .await
            .context("load held rows after the probe cursor")?
            .into_iter()
            .map(|row| (row.id, row.created_at, row.ready))
            .collect(),
            None => sqlx::query!(
                r#"SELECT held.id AS "id!", held.created_at AS "created_at!", NOT EXISTS (SELECT 1 FROM json_each(held.after_siblings_json) AS waited JOIN agent_sdk_tasks AS sibling ON sibling.id = waited.value WHERE sibling.status NOT IN ('completed', 'failed', 'cancelled')) AS "ready!: bool" FROM agent_sdk_tasks AS held INDEXED BY agent_sdk_tasks_held_fifo_idx WHERE held.status >= 'pending' AND held.status <= 'pending' AND held.after_siblings_json IS NOT NULL AND held.kind = 'tool_runtime' AND held.thread_id NOT IN (SELECT thread_id FROM agent_sdk_threads WHERE status IN ('deleting', 'deleted')) ORDER BY held.created_at, held.id LIMIT ?1"#,
                limit,
            )
            .fetch_all(&self.pool)
            .await
            .context("load held rows from the start")?
            .into_iter()
            .map(|row| (row.id, row.created_at, row.ready))
            .collect(),
        };
        Ok(rows
            .into_iter()
            .map(|(id, created_at, ready)| HeldRow {
                id: AgentTaskId::from_string(id.clone()),
                key: (created_at, id),
                ready,
            })
            .collect())
    }

    async fn update_task_tx(tx: &mut Transaction<'_, Sqlite>, task: &AgentTask) -> Result<()> {
        let id = task.id.as_str();
        let kind = enum_to_wire(&task.kind)?;
        let status = enum_to_wire(&task.status)?;
        let parent_id = task.parent_id.as_ref().map(AgentTaskId::as_str);
        let root_id = task.root_id.as_str();
        let depth = i64::from(task.depth);
        let thread_id_key = thread_key(&task.thread_id);
        let submitted_input_json = json_to_value(&task.submitted_input, "task submitted input")?;
        let caller_metadata_json = task.caller_metadata.clone();
        let worker_id = task.worker_id.as_ref().map(WorkerId::as_str);
        let lease_id = task.lease_id.as_ref().map(LeaseId::as_str);
        let state_json = json_to_value(&task.state, "task state")?;
        let attempt = i64::from(task.attempt);
        let max_attempts = i64::from(task.max_attempts);
        let terminal_reason = task
            .terminal_reason
            .as_ref()
            .map(|reason| json_to_value(reason, "task terminal reason"))
            .transpose()?;
        let pending_child_count = i64::from(task.pending_child_count);
        let spawn_index = task.spawn_index.map(i64::from);
        let after_siblings_json = after_siblings_json(task)?;
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_tasks SET
    kind = ?2, status = ?3, parent_id = ?4, root_id = ?5,
    depth = ?6, thread_id = ?7, submitted_input_json = ?8,
    worker_id = ?9, lease_id = ?10, lease_expires_at = ?11,
    last_heartbeat_at = ?12, state_json = ?13, attempt = ?14,
    max_attempts = ?15, last_error = ?16, terminal_reason_json = ?17,
    pending_child_count = ?18,
    spawn_index = ?19, result_payload = ?20,
    created_at = ?21, updated_at = ?22, completed_at = ?23,
    caller_metadata_json = ?24,
    otel_traceparent = ?25,
    last_activity_at = ?26,
    after_siblings_json = ?27
WHERE id = ?1
",
            id,
            kind,
            status,
            parent_id,
            root_id,
            depth,
            thread_id_key,
            submitted_input_json,
            worker_id,
            lease_id,
            task.lease_expires_at,
            task.last_heartbeat_at,
            state_json,
            attempt,
            max_attempts,
            task.last_error,
            terminal_reason,
            pending_child_count,
            spawn_index,
            task.result_payload,
            task.created_at,
            task.updated_at,
            task.completed_at,
            caller_metadata_json,
            task.otel_traceparent,
            task.last_activity_at,
            after_siblings_json,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("update task {}", task.id))?;
        ensure!(
            result.rows_affected() == 1,
            "update task affected {} rows for {}",
            result.rows_affected(),
            task.id
        );
        if task.status.is_terminal() {
            Self::abandon_started_effects_tx(tx, task).await?;
        }
        Ok(())
    }

    /// Close the still-open attempt row an abandoned LLM request names, in
    /// the transaction that abandons the request: its task ended before the
    /// request finished, so nothing will close it later (crash matrix S3).
    /// Apply a transition's attempt write inside its transaction. A close
    /// refuses an unknown attempt, one of another task, and one already
    /// closed.
    async fn apply_attempt_write_tx(
        tx: &mut Transaction<'_, Sqlite>,
        write: agent_server::journal::TurnAttemptWrite,
        now: OffsetDateTime,
    ) -> Result<()> {
        match write {
            agent_server::journal::TurnAttemptWrite::Open(attempt) => {
                Self::insert_attempt_tx(tx, &attempt).await
            }
            agent_server::journal::TurnAttemptWrite::Close {
                attempt,
                task,
                params,
            } => {
                let open = Self::get_attempt_tx(tx, &attempt)
                    .await?
                    .ok_or_else(|| anyhow!("attempt not found: {attempt}"))?;
                anyhow::ensure!(
                    open.task_id == task,
                    "attempt {attempt} belongs to task {}, not {task}",
                    open.task_id
                );
                let closed = open.close(*params, now)?;
                Self::update_attempt_tx(tx, &closed).await?;
                agent_server::fail_point!("transition.attempt_closed");
                Ok(())
            }
        }
    }

    async fn close_abandoned_attempt_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &TurnAttemptId,
        now: OffsetDateTime,
    ) -> Result<()> {
        let Some(attempt) = Self::get_attempt_tx(tx, id).await? else {
            return Ok(());
        };
        if attempt.outcome.is_some() {
            return Ok(());
        }
        let closed = attempt.close(CloseAttemptParams::abandoned(), now)?;
        Self::update_attempt_tx(tx, &closed).await
    }

    /// End the still-`Started` effects of a task that just became
    /// terminal as `Abandoned`, in the same transaction: a `Started`
    /// record's owner is live. Every write of a task's status goes through
    /// [`Self::update_task_tx`], so no path that ends an owner skips this.
    async fn abandon_started_effects_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTask,
    ) -> Result<()> {
        let columns = EffectState::Abandoned.columns();
        let abandoned = sqlx::query!(
            r#"
UPDATE agent_sdk_execution_intents
SET status     = ?1,
    error      = ?2,
    resolution = ?3,
    updated_at = ?4
WHERE child_task_id = ?5
  AND status IN ('pending', 'started')
RETURNING operation_id AS "operation_id!", effect_kind AS "effect_kind!",
          tool_call_id AS "tool_call_id!"
"#,
            columns.status,
            columns.error,
            columns.resolution,
            task.updated_at,
            task.id.as_str(),
        )
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("abandon the started effects of task {}", task.id))?;
        let reason = agent_server::journal::effect::abandon_reason(task);
        for row in &abandoned {
            tracing::info!(
                effect_key = %row.operation_id,
                effect_kind = %row.effect_kind,
                task_id = %task.id,
                reason,
                "an effect was abandoned because its owner ended first; it is never run again"
            );
            #[cfg(feature = "otel")]
            if let Some(kind) = agent_server::journal::effect::kind_from_wire(&row.effect_kind) {
                agent_server::observability::ServerMetrics::global().record_effect_abandoned(
                    kind,
                    reason,
                    &row.operation_id,
                    task.id.as_str(),
                );
            }
            if row.effect_kind
                == agent_server::journal::effect::kind_wire(
                    agent_server::journal::effect::EffectKind::LlmRequest,
                )
            {
                Self::close_abandoned_attempt_tx(
                    tx,
                    &TurnAttemptId::from_string(row.tool_call_id.clone()),
                    task.updated_at,
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn enforce_insert_cross_row_invariants_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTask,
    ) -> Result<()> {
        task.validate()
            .context("insert rejected: task failed schema validation")?;
        Self::bootstrap_thread_row_tx(tx, &task.thread_id, task.created_at).await?;

        if let Some(parent_id) = &task.parent_id {
            let parent = Self::load_task_tx(tx, parent_id).await?.ok_or_else(|| {
                anyhow!("insert rejected: child task references unknown parent {parent_id}")
            })?;
            if parent.kind.is_leaf() {
                let parent_kind = parent.kind;
                return Err(anyhow!(
                    "insert rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
                ));
            }
            if parent.thread_id != task.thread_id {
                return Err(anyhow!(
                    "insert rejected: child thread_id {} does not match parent thread_id {}",
                    task.thread_id,
                    parent.thread_id
                ));
            }
            if parent.root_id != task.root_id {
                return Err(anyhow!(
                    "insert rejected: child root_id {} does not match parent root_id {}",
                    task.root_id,
                    parent.root_id
                ));
            }
            let expected_depth = parent.depth.saturating_add(1);
            if task.depth != expected_depth {
                return Err(anyhow!(
                    "insert rejected: child depth {} must be parent.depth + 1 ({} + 1 = {})",
                    task.depth,
                    parent.depth,
                    expected_depth
                ));
            }
        }

        if task.kind == TaskKind::RootTurn && task.status.blocks_root_admission() {
            let thread_id_key = thread_key(&task.thread_id);
            let task_id_str = task.id.as_str();
            let existing = sqlx::query_scalar!(
                r"
SELECT id FROM agent_sdk_tasks
WHERE thread_id = ?1 AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question')
  AND id <> ?2
LIMIT 1
",
                thread_id_key,
                task_id_str,
            )
            .fetch_optional(&mut **tx)
            .await
            .with_context(|| format!("check blocking root slot for thread {}", task.thread_id))?
            .flatten();
            if let Some(existing_id) = existing {
                return Err(anyhow!(
                    "insert rejected: thread {} already has active root task {}",
                    task.thread_id,
                    existing_id
                ));
            }
        }

        Ok(())
    }

    async fn validate_update_row_invariants_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task: &AgentTask,
    ) -> Result<AgentTask> {
        task.validate()
            .context("update rejected: task failed schema validation")?;
        let old = Self::load_task_tx(tx, &task.id)
            .await?
            .ok_or_else(|| anyhow!("update rejected: task {} does not exist", task.id))?;

        if old.kind != task.kind {
            return Err(anyhow!(
                "update rejected: task kind is immutable (was {:?}, got {:?})",
                old.kind,
                task.kind
            ));
        }
        if old.parent_id != task.parent_id {
            return Err(anyhow!(
                "update rejected: parent_id is immutable (was {:?}, got {:?})",
                old.parent_id,
                task.parent_id
            ));
        }
        if old.root_id != task.root_id {
            return Err(anyhow!(
                "update rejected: root_id is immutable (was {}, got {})",
                old.root_id,
                task.root_id
            ));
        }
        if old.depth != task.depth {
            return Err(anyhow!(
                "update rejected: depth is immutable (was {}, got {})",
                old.depth,
                task.depth
            ));
        }
        if old.thread_id != task.thread_id {
            return Err(anyhow!(
                "update rejected: thread_id is immutable (was {}, got {})",
                old.thread_id,
                task.thread_id
            ));
        }
        if old.created_at != task.created_at {
            return Err(anyhow!("update rejected: created_at is immutable"));
        }
        if old.max_attempts != task.max_attempts {
            return Err(anyhow!(
                "update rejected: max_attempts is immutable (was {}, got {})",
                old.max_attempts,
                task.max_attempts
            ));
        }

        if task.kind == TaskKind::RootTurn && task.status.blocks_root_admission() {
            let thread_id_key = thread_key(&task.thread_id);
            let task_id_str = task.id.as_str();
            let current = sqlx::query_scalar!(
                r"
SELECT id FROM agent_sdk_tasks
WHERE thread_id = ?1 AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question')
  AND id <> ?2
LIMIT 1
",
                thread_id_key,
                task_id_str,
            )
            .fetch_optional(&mut **tx)
            .await
            .with_context(|| format!("check competing root slot for thread {}", task.thread_id))?
            .flatten();
            if let Some(current_id) = current {
                return Err(anyhow!(
                    "update rejected: thread {} already has a different active root task {}",
                    task.thread_id,
                    current_id
                ));
            }
        }
        Ok(old)
    }

    async fn load_live_child_count_tx(
        tx: &mut Transaction<'_, Sqlite>,
        parent_id: &AgentTaskId,
    ) -> Result<u32> {
        let parent_id_str = parent_id.as_str();
        let record = sqlx::query!(
            r"SELECT COUNT(*) AS cnt FROM agent_sdk_tasks WHERE parent_id = ?1 AND status NOT IN ('completed', 'failed', 'cancelled')",
            parent_id_str,
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("count live children for {parent_id}"))?;
        let live: i64 = record.cnt;
        u32_from_i64(live, "live child count")
    }

    /// Re-park a steering exchange's parent inside the caller's
    /// transaction, checking its ownership after locking its children.
    async fn repark_after_steering_tx(
        tx: &mut Transaction<'_, Sqlite>,
        parent_id: &AgentTaskId,
        (worker, lease): (&WorkerId, &LeaseId),
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let old = Self::load_task_tx(tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("repark rejected: task {parent_id} does not exist"))?;
        if old.status != TaskStatus::Running {
            let status = old.status;
            return Err(anyhow!(
                "repark rejected: task {parent_id} is not running (status {status:?})"
            ));
        }
        match &old.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "repark rejected: worker mismatch on task {parent_id}"
                ));
            }
        }
        match &old.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "repark rejected: lease mismatch on task {parent_id}"
                ));
            }
        }
        if !matches!(old.state, TaskState::ReadyToResume { .. }) {
            return Err(anyhow!(
                "repark rejected: task {parent_id} is not a child resume"
            ));
        }

        // Re-index the re-attach children to dense spawn_index
        // positions matching the new continuation's tool-use order.
        let mut live = 0_u32;
        for (idx, child_id) in reattach.iter().enumerate() {
            let mut child = Self::load_task_tx(tx, child_id)
                .await?
                .ok_or_else(|| anyhow!("repark rejected: re-attach child {child_id} missing"))?;
            if child.parent_id.as_ref() != Some(parent_id) {
                return Err(anyhow!(
                    "repark rejected: re-attach child {child_id} is not a child of {parent_id}"
                ));
            }
            child.spawn_index =
                Some(u32::try_from(idx).context("repark rejected: reattach index exceeds u32")?);
            child.updated_at = now;
            child
                .validate()
                .context("repark rejected: re-attach child validate failed")?;
            if !child.status.is_terminal() {
                live = live
                    .checked_add(1)
                    .context("repark rejected: live child count exceeds u32")?;
            }
            Self::update_task_tx(tx, &child).await?;
        }

        // A re-attach child that finished during the LLM call is already
        // terminal here and excluded.
        let reparked = old
            .clone()
            .repark_after_steering(live, payload, reattach, now)
            .context("repark rejected: repark_after_steering transition failed")?;
        Self::update_task_tx(tx, &reparked).await?;
        Ok(reparked)
    }

    /// Park a running task on a confirmation inside the caller's
    /// transaction. The caller's guard has checked its status and lease.
    async fn pause_on_confirmation_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &AgentTaskId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let old = Self::load_task_tx(tx, id)
            .await?
            .ok_or_else(|| anyhow!("pause rejected: task {id} does not exist"))?;
        let paused = old
            .await_confirmation(continuation, prepared_operation, now)
            .context("pause rejected: await_confirmation transition failed")?;
        Self::update_task_tx(tx, &paused).await?;
        Ok(paused)
    }

    /// Commit `events` on `thread_id` with their outbox row, inside the
    /// caller's transaction; nothing for an empty list.
    async fn commit_parent_events_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let start_seq = Self::next_event_sequence_tx(tx, thread_id).await?;
        let committed = Self::insert_events_tx(tx, thread_id, events, start_seq, now).await?;
        Self::insert_thread_events_outbox_row_tx(
            tx,
            &committed,
            DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        )
        .await?;
        Ok(committed)
    }

    /// Fail a task parked on a confirmation and recompute its waiting
    /// parent, inside the caller's transaction.
    async fn reject_confirmation_tx(
        tx: &mut Transaction<'_, Sqlite>,
        id: &AgentTaskId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let old = Self::load_task_tx(tx, id)
            .await?
            .ok_or_else(|| anyhow!("reject_confirmation rejected: task {id} does not exist"))?;
        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "reject_confirmation rejected: task {id} is not awaiting confirmation (status {status:?})"
            ));
        }
        let failed = old
            .clone()
            .reject_confirmation(error, now)
            .context("reject_confirmation: fail transition failed")?;
        Self::update_task_tx(tx, &failed).await?;

        let parent = if let Some(parent_id) = &failed.parent_id {
            let old_parent = Self::load_task_tx(tx, parent_id).await?.ok_or_else(|| {
                anyhow!("reject_confirmation: child {id} references missing parent {parent_id}")
            })?;
            if old_parent.status == TaskStatus::WaitingOnChildren {
                let live = Self::load_live_child_count_tx(tx, parent_id).await?;
                let new_parent = old_parent
                    .clone()
                    .recompute_pending_children(live, now)
                    .context("reject_confirmation: recompute_pending_children failed")?;
                Self::update_task_tx(tx, &new_parent).await?;
                Some(new_parent)
            } else {
                Some(old_parent)
            }
        } else {
            None
        };

        Ok((failed, parent))
    }

    /// Insert an effect's `Started` row inside the caller's transaction.
    /// The transition's `EffectFree` guard has already found the key free
    /// under the owner's lock, so a conflict here is a store error.
    async fn insert_effect_tx(
        tx: &mut Transaction<'_, Sqlite>,
        record: &EffectRecord,
    ) -> Result<()> {
        let row = record.to_row()?;
        let result = sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at,
    effect_kind, attempt, resolution, captured_output, journal_anchor, thread_id
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
ON CONFLICT (operation_id) DO NOTHING
",
            row.operation_id,
            row.effect_class,
            row.tool_call_id,
            row.child_task_id,
            row.tool_name,
            row.input,
            row.status,
            row.error,
            row.created_at,
            row.updated_at,
            row.effect_kind,
            row.attempt,
            row.resolution,
            row.captured_output,
            row.journal_anchor,
            row.thread_id,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert effect {}", record.key))?;
        ensure!(
            result.rows_affected() == 1,
            "effect {} already has a record",
            record.key
        );
        Ok(())
    }

    /// End a `Started` effect of `owner` inside the caller's transaction.
    async fn finish_effect_tx(
        tx: &mut Transaction<'_, Sqlite>,
        key: &EffectKey,
        owner: &AgentTaskId,
        state: &EffectState,
        now: OffsetDateTime,
    ) -> Result<()> {
        let columns = state.columns();
        let key_str = key.as_str();
        let owner_str = owner.as_str();
        let finished = sqlx::query!(
            r#"
UPDATE agent_sdk_execution_intents
SET status          = ?3,
    error           = ?4,
    resolution      = ?5,
    captured_output = ?6,
    updated_at      = ?7
WHERE operation_id = ?1
  AND child_task_id = ?2
  AND (status IN ('pending', 'started') OR (resolution = 'abandoned' AND ?5 = 'salvaged'))
RETURNING effect_kind AS "effect_kind!", tool_call_id AS "tool_call_id!"
"#,
            key_str,
            owner_str,
            columns.status,
            columns.error,
            columns.resolution,
            columns.captured_output,
            now,
        )
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("finish effect {key}"))?;
        let [row] = finished.as_slice() else {
            anyhow::bail!("finish effect {key} affected {} rows", finished.len());
        };
        // An LLM request that ends closes its attempt if it is still open
        // (provisionally, as an abandoned close: a live worker's own close
        // replaces it).
        if row.effect_kind
            == agent_server::journal::effect::kind_wire(
                agent_server::journal::effect::EffectKind::LlmRequest,
            )
        {
            Self::close_abandoned_attempt_tx(
                tx,
                &TurnAttemptId::from_string(row.tool_call_id.clone()),
                now,
            )
            .await?;
        }
        Ok(())
    }

    async fn load_effect<'e, E>(executor: E, key: &EffectKey) -> Result<Option<EffectRecord>>
    where
        E: sqlx::Executor<'e, Database = Sqlite>,
    {
        let record = sqlx::query_as::<_, EffectRecordRow>(
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at,
    effect_kind, attempt, resolution, captured_output, journal_anchor, thread_id
FROM agent_sdk_execution_intents
WHERE operation_id = ?1
",
        )
        .bind(key.as_str())
        .fetch_optional(executor)
        .await
        .with_context(|| format!("get effect {key}"))?;
        record.map(EffectRecordRow::into_record).transpose()
    }

    /// Check the transition's effect expectations against the rows as this
    /// transaction sees them.
    async fn check_effects_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
    ) -> Result<(), agent_server::journal::Rejected> {
        let mut stored = BTreeMap::new();
        for key in transition.effect_keys_read() {
            if let Some(record) = Self::load_effect(&mut **tx, key).await? {
                stored.insert(key.clone(), record);
            }
        }
        transition.check_effects(|key| stored.get(key))
    }

    async fn apply_task_terminal_transition_tx(
        tx: &mut Transaction<'_, Sqlite>,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
        error_prefix: &'static str,
        transition: impl FnOnce(AgentTask) -> Result<AgentTask, agent_server::journal::TaskSchemaError>,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let old_child = Self::load_task_tx(tx, child_id)
            .await?
            .ok_or_else(|| anyhow!("{error_prefix} rejected: task {child_id} does not exist"))?;
        if old_child.status != TaskStatus::Running {
            let status = old_child.status;
            return Err(anyhow!(
                "{error_prefix} rejected: task {child_id} is not running (status {status:?})"
            ));
        }
        match &old_child.worker_id {
            Some(current) if current == worker => {}
            _ => {
                return Err(anyhow!(
                    "{error_prefix} rejected: worker mismatch on task {child_id}"
                ));
            }
        }
        match &old_child.lease_id {
            Some(current) if current == lease => {}
            _ => {
                return Err(anyhow!(
                    "{error_prefix} rejected: lease mismatch on task {child_id}"
                ));
            }
        }

        let new_child = transition(old_child.clone())
            .with_context(|| format!("{error_prefix}: terminal transition failed"))?;
        Self::update_task_tx(tx, &new_child).await?;

        let parent = if let Some(parent_id) = &new_child.parent_id {
            let old_parent = Self::load_task_tx(tx, parent_id).await?.ok_or_else(|| {
                anyhow!("{error_prefix}: child {child_id} references missing parent {parent_id}")
            })?;
            if old_parent.status == TaskStatus::WaitingOnChildren {
                let live = Self::load_live_child_count_tx(tx, parent_id).await?;
                let new_parent = old_parent
                    .clone()
                    .recompute_pending_children(live, now)
                    .with_context(|| {
                        format!("{error_prefix}: recompute_pending_children transition failed")
                    })?;
                Self::update_task_tx(tx, &new_parent).await?;
                Some(new_parent)
            } else {
                Some(old_parent)
            }
        } else if new_child.kind == TaskKind::RootTurn && new_child.is_root() {
            // Phase 7.6 / M5.4 follow-up: a child-thread
            // root turn has no `parent_id` but is logically linked
            // to a parent-thread `Subagent` invocation task via
            // `state.subagent_invocation.child_root_task_id`. The
            // in-memory `InMemoryAgentTaskStore` resumes that
            // invocation here (see `journal::store.rs` —
            // `resume_linked_subagent_invocation`); without the
            // mirror call here the SQLite-backed daemon leaves the
            // invocation stuck in `WaitingOnChildren` after the
            // child thread completes, which means
            // `execute_subagent_task` never runs and the parent
            // thread's `SubagentProgress { completed: true }` event
            // never fires. UI surfaces stay in `Running` forever.
            Self::release_undelivered_input_tx(tx, &new_child, now).await?;
            let resumed =
                Self::resume_linked_subagent_invocation_tx(tx, &new_child.id, now).await?;
            // Crash-safe queued-root promotion: a terminal root frees the
            // thread's active-root slot, so promote the FIFO head in the
            // SAME transaction rather than relying solely on the host's
            // post-commit promotion (which a crash could skip).
            Self::promote_next_queued_root_tx(tx, &new_child.thread_id, now).await?;
            resumed
        } else {
            None
        };

        Ok((new_child, parent))
    }

    /// After a child task transitions to a terminal state, wake its
    /// `WaitingOnChildren` parent (if any) by recomputing
    /// `pending_child_count`.  Without this, a parent with a single
    /// fail-closed child stays in `WaitingOnChildren` forever
    /// because no future `complete_task`/`fail_task` call will fire
    /// to decrement the counter — a liveness deadlock.
    async fn propagate_terminal_to_parent_tx(
        tx: &mut Transaction<'_, Sqlite>,
        child: &AgentTask,
        now: OffsetDateTime,
        error_prefix: &str,
    ) -> Result<()> {
        let Some(parent_id) = &child.parent_id else {
            // Parentless terminal row. A child-thread root turn has no
            // `parent_id` but is logically linked to a parent-thread
            // `Subagent` invocation via
            // `state.subagent_invocation.child_root_task_id`. Mirror the
            // in-memory store and `apply_task_terminal_transition_tx`:
            // wake that invocation so a fail-closed child root (e.g. a
            // recovery sweep) does not leave the parent stuck in
            // `WaitingOnChildren` forever (a durable liveness deadlock).
            // Also promote this thread's next queued root in the same
            // transaction so a sweep-failed root never strands its
            // queued successors.
            if child.kind == TaskKind::RootTurn && child.is_root() {
                Self::release_undelivered_input_tx(tx, child, now).await?;
                Self::resume_linked_subagent_invocation_tx(tx, &child.id, now).await?;
                Self::promote_next_queued_root_tx(tx, &child.thread_id, now).await?;
            }
            return Ok(());
        };
        let old_parent = Self::load_task_tx(tx, parent_id).await?.ok_or_else(|| {
            anyhow!(
                "{error_prefix}: child {child_id} references missing parent {parent_id}",
                child_id = child.id,
            )
        })?;
        if old_parent.status != TaskStatus::WaitingOnChildren {
            return Ok(());
        }
        let live = Self::load_live_child_count_tx(tx, parent_id).await?;
        let new_parent = old_parent
            .recompute_pending_children(live, now)
            .with_context(|| {
                format!("{error_prefix}: recompute_pending_children transition failed")
            })?;
        Self::update_task_tx(tx, &new_parent).await?;
        Ok(())
    }

    /// Fail `old` closed for `reason` inside the caller's transaction:
    /// the terminal transition, the terminal event that closes a root's
    /// turn (see [`FailureReason::terminal_event`]), and the parent /
    /// successor wake-up land together. Also returns the root turns it
    /// ended, for the caller to record once it commits.
    async fn fail_closed_tx(
        tx: &mut Transaction<'_, Sqlite>,
        old: &AgentTask,
        reason: FailureReason,
        now: OffsetDateTime,
        error_prefix: &str,
    ) -> Result<(CancelTreeOutcome, Vec<TerminalRootTurn>)> {
        let (settled, mut root_turns) = if old.kind == TaskKind::RootTurn && old.is_root() {
            Self::cancel_scoped_tx(tx, &old.id, CancelScope::Descendants, now).await?
        } else {
            (CancelTreeOutcome::default(), Vec::new())
        };
        let failed = old
            .clone()
            .fail_with_reason(reason, now)
            .with_context(|| format!("{error_prefix}: fail-closed transition failed"))?;
        Self::update_task_tx(tx, &failed).await?;
        if let Some(event) = reason.terminal_event(old) {
            let start_seq = Self::next_event_sequence_tx(tx, &old.thread_id).await?;
            let committed =
                Self::insert_events_tx(tx, &old.thread_id, vec![event], start_seq, now).await?;
            Self::insert_thread_events_outbox_row_tx(
                tx,
                &committed,
                DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                now,
            )
            .await?;
        }
        Self::propagate_terminal_to_parent_tx(tx, &failed, now, error_prefix).await?;
        root_turns.extend(TerminalRootTurn::of(&failed));
        Ok((settled, root_turns))
    }

    /// Settle the input injections `root` never delivered, inside the
    /// transaction that ended it and before its thread's queue head is
    /// promoted (see [`release_undelivered_input`]).
    async fn release_undelivered_input_tx(
        tx: &mut Transaction<'_, Sqlite>,
        root: &AgentTask,
        now: OffsetDateTime,
    ) -> Result<()> {
        if !settles_undelivered_input(root) {
            return Ok(());
        }
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE parent_id = ?1 AND kind = 'input_injection' AND status = 'queued' \
             ORDER BY created_at, id",
        ))
        .bind(root.id.as_str())
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("load undelivered input of {}", root.id))?;
        if records.is_empty() {
            return Ok(());
        }
        let thread_open = Self::get_thread_tx(tx, &root.thread_id)
            .await?
            .is_none_or(|thread| thread.require_active(ThreadOperation::Submit).is_ok());
        let hand_over = thread_open && !Self::links_child_thread_root_tx(tx, &root.id).await?;
        for record in records {
            let released = release_undelivered_input(AgentTask::try_from(record)?, hand_over, now)?;
            Self::update_task_tx(tx, &released.injection).await?;
            if let Some(next_turn) = &released.next_turn {
                Self::insert_task_tx(tx, next_turn).await?;
            }
        }
        Ok(())
    }

    /// Whether `root` is a subagent's child-thread root: some invocation
    /// links it, whatever that invocation's status (see
    /// [`release_undelivered_input`]).
    async fn links_child_thread_root_tx(
        tx: &mut Transaction<'_, Sqlite>,
        root: &AgentTaskId,
    ) -> Result<bool> {
        let root_id = root.as_str();
        let linked = sqlx::query_file_scalar!("src/sqlite/links_child_thread_root.sql", root_id)
            .fetch_one(&mut **tx)
            .await
            .with_context(|| format!("look up the invocation linking {root}"))?;
        Ok(linked)
    }

    /// Phase 7.6: find a `Subagent` invocation task that is
    /// `WaitingOnChildren` and whose `SubagentInvocation` state
    /// links to the given `child_root_id`, then wake it to `Pending`
    /// via `recompute_pending_children(0, now)`.
    ///
    /// This mirrors the in-memory store's
    /// `resume_linked_subagent_invocation` and ensures that when a
    /// child-thread root reaches a terminal state (cancelled, failed,
    /// completed), the parent-thread invocation is unblocked.
    async fn resume_linked_subagent_invocation_tx(
        tx: &mut Transaction<'_, Sqlite>,
        child_root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let maybe_record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE kind = 'subagent' \
               AND status = 'waiting_on_children' \
               AND json_extract(state_json, '$.invocation.child_root_task_id') = ?1",
        ))
        .bind(child_root_id.as_str())
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| {
            format!("resume_linked_subagent_invocation: lookup for child_root {child_root_id}")
        })?;

        let Some(record) = maybe_record else {
            return Ok(None);
        };
        let old_invocation: AgentTask = record.try_into()?;
        let new_invocation = old_invocation
            .recompute_pending_children(0, now)
            .context("cancel_tree: subagent invocation resume transition failed")?;
        Self::update_task_tx(tx, &new_invocation).await?;
        Ok(Some(new_invocation))
    }

    /// Promote the FIFO head of `thread_id`'s queued-root list to
    /// `Pending` inside `tx`, if the thread has no blocking root, and
    /// emit the promoted root's durable `task_wakeup` advisory outbox row
    /// (Phase 10 · D — so a worker in any process is nudged even after
    /// host death).
    ///
    /// Shared by [`AgentTaskStore::promote_next_queued_root`] and every
    /// in-transaction root-turn terminal path (terminal CAS, recovery
    /// fail-close, cancel) so a queued root is never stranded behind a
    /// root that reached a terminal state without the host's post-commit
    /// promotion firing.
    async fn promote_next_queued_root_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let blocking_check_key = thread_key(thread_id);
        let blocking = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question') LIMIT 1", blocking_check_key)
            .fetch_optional(&mut **tx)
            .await
            .with_context(|| format!("check blocking root for {thread_id}"))?
            .flatten();
        if blocking.is_some() {
            return Ok(None);
        }

        let queued = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE thread_id = ?1 AND kind = 'root_turn' AND status = 'queued' \
             ORDER BY created_at, id LIMIT 1",
        ))
        .bind(thread_key(thread_id))
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("load queue head for {thread_id}"))?;
        let Some(queued) = queued else {
            return Ok(None);
        };
        let queued = AgentTask::try_from(queued)?;
        let promoted = queued
            .clone()
            .promote_to_pending(now)
            .context("promote rejected: promotion transition failed")?;
        Self::update_task_tx(tx, &promoted).await?;

        // The promoted root is now runnable — emit its durable wakeup in
        // the same transaction so the promotion survives host death.
        Self::insert_task_wakeup_outbox_row_tx(
            tx,
            &promoted.id,
            &promoted.thread_id,
            TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS,
            now,
        )
        .await?;
        Ok(Some(promoted))
    }

    /// Collect `start_id` plus every descendant, mirroring the in-memory
    /// `collect_subtree`: a children-BFS (following `parent_id`) that also
    /// follows `SubagentInvocation` linkage across thread boundaries.
    ///
    /// Unlike a `root_id = $1` scan, this walks from the *given* id, so a
    /// mid-tree task id cancels its own subtree (matching the in-memory
    /// reference store) instead of silently no-opping. Reads are
    /// non-locking; the cancel path takes its row locks at update time in
    /// a deterministic deepest-first order.
    async fn collect_subtree_tx(
        tx: &mut Transaction<'_, Sqlite>,
        start_id: &AgentTaskId,
    ) -> Result<Vec<AgentTask>> {
        use std::collections::{BTreeSet, VecDeque};
        let mut visited: BTreeSet<AgentTaskId> = BTreeSet::new();
        let mut out: Vec<AgentTask> = Vec::new();
        let mut frontier: VecDeque<AgentTaskId> = VecDeque::new();
        frontier.push_back(start_id.clone());
        while let Some(id) = frontier.pop_front() {
            if !visited.insert(id.clone()) {
                continue;
            }
            let Some(task) = Self::load_task_tx(tx, &id).await? else {
                continue;
            };
            if let Some(invocation) = task.state.subagent_invocation() {
                let child_root = invocation.child_root_task_id.clone();
                if !visited.contains(&child_root) {
                    frontier.push_back(child_root);
                }
            }
            let id_str = id.as_str();
            let child_ids = sqlx::query_scalar!(
                "SELECT id FROM agent_sdk_tasks WHERE parent_id = ?1 ORDER BY created_at, id",
                id_str,
            )
            .fetch_all(&mut **tx)
            .await
            .with_context(|| format!("load children of {id}"))?;
            // SQLite infers `id` as nullable; flatten the always-present
            // PK values.
            for child in child_ids.into_iter().flatten() {
                let child_id = AgentTaskId::from_string(child);
                if !visited.contains(&child_id) {
                    frontier.push_back(child_id);
                }
            }
            out.push(task);
        }
        Ok(out)
    }

    // ── Event helpers ────────────────────────────────────────────────

    async fn next_event_sequence_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<u64> {
        let thread_id_key = thread_key(thread_id);
        // Derive the next sequence from both the committed-events max
        // AND the retention floor so that sequences never regress
        // after the janitor purges events. Without the retention
        // floor, a thread whose entire history has been purged would
        // see MAX()=NULL and start re-assigning sequences from 0,
        // making those events invisible to any subscriber seeded with
        // `last_yielded = floor - 1`.
        let record = sqlx::query!(
            r"SELECT COALESCE(MAX(sequence) + 1, 0) AS next_seq FROM agent_sdk_committed_events WHERE thread_id = ?1",
            thread_id_key,
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("next event sequence (tx) for {thread_id}"))?;
        let from_events =
            u64::try_from(record.next_seq).context("event next_sequence (tx) out of range")?;

        let floor_record = sqlx::query!(
            r"SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = ?1",
            thread_id_key,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("read retention floor (tx) for {thread_id}"))?;
        let floor = floor_record
            .map(|r| u64::try_from(r.retention_floor))
            .transpose()
            .context("retention_floor (tx) out of range")?
            .unwrap_or(0);

        Ok(from_events.max(floor))
    }

    /// Insert a spawned child thread's `ThreadCreated` and its
    /// events-available advisory inside the caller's transaction.
    async fn insert_child_created_tx(
        tx: &mut Transaction<'_, Sqlite>,
        child_thread: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<()> {
        let seq = Self::next_event_sequence_tx(tx, child_thread).await?;
        let committed = Self::insert_events_tx(
            tx,
            child_thread,
            vec![AgentEvent::thread_created(child_thread.clone(), None, None)],
            seq,
            now,
        )
        .await?;
        Self::insert_thread_events_outbox_row_tx(
            tx,
            &committed,
            DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        )
        .await?;
        Ok(())
    }

    /// Commit each new subagent's `SubagentProgress` start event on the
    /// parent journal and each new child thread's `ThreadCreated` event
    /// inside the caller's spawn transaction. Attached tool-only batches
    /// have neither and therefore commit no event batch or advisory.
    async fn insert_spawn_started_events_tx(
        tx: &mut Transaction<'_, Sqlite>,
        parent_thread: &ThreadId,
        prepared: &[(AgentTask, AgentTask)],
        events: Vec<SubagentSpawnEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        let mut started = Vec::with_capacity(events.len());
        for (event, (invocation, child_root)) in events.into_iter().zip(prepared) {
            started.push(started_event_from_invocation(
                event, invocation, child_root,
            )?);
        }
        let committed = if started.is_empty() {
            Vec::new()
        } else {
            let start_seq = Self::next_event_sequence_tx(tx, parent_thread).await?;
            let committed =
                Self::insert_events_tx(tx, parent_thread, started, start_seq, now).await?;
            Self::insert_thread_events_outbox_row_tx(
                tx,
                &committed,
                DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                now,
            )
            .await?;
            committed
        };
        for (_, child_root) in prepared {
            Self::insert_child_created_tx(tx, &child_root.thread_id, now).await?;
        }
        Ok(committed)
    }

    async fn insert_events_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        start_seq: u64,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        let mut committed = Vec::with_capacity(events.len());
        for (idx, event) in events.into_iter().enumerate() {
            let seq = start_seq + idx as u64;
            let event_id = uuid::Uuid::now_v7();
            let event_json = json_to_value(&event, "committed event batch payload")?;

            let event_id_str = event_id.to_string();
            let thread_id_key = thread_key(thread_id);
            let sequence = i64_from_u64(seq, "event batch sequence")?;
            sqlx::query!(
                r"INSERT INTO agent_sdk_committed_events (event_id, thread_id, sequence, event_json, committed_at)
VALUES (?1, ?2, ?3, ?4, ?5)",
                event_id_str,
                thread_id_key,
                sequence,
                event_json,
                now,
            )
            .execute(&mut **tx)
            .await
            .with_context(|| format!("insert committed event seq {seq} for {thread_id}"))?;

            committed.push(CommittedEvent {
                event_id,
                thread_id: thread_id.clone(),
                sequence: seq,
                timestamp: now,
                event,
            });
        }
        Ok(committed)
    }

    /// Insert the coalesced advisory outbox row for a freshly
    /// committed event batch.  See the Postgres analogue for the
    /// Phase 8.1 contract — exactly one
    /// `OutboxMessageKind::ThreadEventsAvailable` row per batch with
    /// payload `{thread_id, last_sequence}`.
    ///
    /// The row's `sequence` / `event_id` columns store the FIRST
    /// event of the batch so that `min_unpublished_sequence` acts as
    /// a retention-floor safety bound for every event in the batch
    /// (not just the last one).  The advisory payload still carries
    /// `last_sequence` so subscribers know how far to replay.
    async fn insert_thread_events_outbox_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        committed: &[CommittedEvent],
        max_attempts: u32,
        now: OffsetDateTime,
    ) -> Result<OutboxRow> {
        let first = committed
            .first()
            .context("event batch must contain at least one event")?;
        let last = committed
            .last()
            .context("event batch must contain at least one event")?;

        let id = OutboxRowId::new();
        let payload_message = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: last.thread_id.clone(),
            last_sequence: last.sequence,
        });
        let payload_json = payload_message
            .to_payload_json()
            .context("serialise thread_events_available advisory payload")?;
        let id_str = id.as_str();
        let thread_id_key = thread_key(&last.thread_id);
        let event_id_str = first.event_id.to_string();
        let sequence_i64 = i64_from_u64(first.sequence, "outbox sequence")?;
        let max_attempts_i64 = i64::from(max_attempts);
        let kind_str = OutboxMessageKind::ThreadEventsAvailable.as_str();

        sqlx::query!(
            r"
INSERT INTO agent_sdk_outbox
    (id, kind, thread_id, event_id, sequence, status, payload_json,
     created_at, next_attempt_at, attempt_count, max_attempts)
VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6, ?7, ?7, 0, ?8)
",
            id_str,
            kind_str,
            thread_id_key,
            event_id_str,
            sequence_i64,
            payload_json,
            now,
            max_attempts_i64,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| {
            format!(
                "insert thread_events_available outbox row for batch [{}..={}]",
                first.sequence, last.sequence,
            )
        })?;

        Ok(OutboxRow {
            id,
            kind: OutboxMessageKind::ThreadEventsAvailable,
            thread_id: last.thread_id.clone(),
            event_id: Some(first.event_id),
            sequence: Some(first.sequence),
            status: OutboxStatus::Pending,
            payload_json,
            created_at: now,
            next_attempt_at: now,
            attempt_count: 0,
            max_attempts,
            last_error: None,
            claimed_by: None,
            claimed_at: None,
            delivered_at: None,
        })
    }

    /// Insert a durable `task_wakeup` advisory outbox row inside an
    /// existing transaction.  See the Postgres analogue
    /// (`insert_task_wakeup_outbox_row_tx`) for the Phase 10 · D
    /// contract — the row carries no `event_id`/`sequence`, only a
    /// `{task_id, thread_id}` advisory payload, and becomes visible iff
    /// the surrounding admission transaction commits.
    async fn insert_task_wakeup_outbox_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task_id: &AgentTaskId,
        thread_id: &ThreadId,
        max_attempts: u32,
        now: OffsetDateTime,
    ) -> Result<OutboxRowId> {
        let id = OutboxRowId::new();
        let payload_json = OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id: task_id.clone(),
            thread_id: thread_id.clone(),
        })
        .to_payload_json()
        .context("serialise task_wakeup advisory payload")?;

        let id_str = id.as_str();
        let thread_id_key = thread_key(thread_id);
        let kind_str = OutboxMessageKind::TaskWakeup.as_str();
        let max_attempts_i64 = i64::from(max_attempts);

        sqlx::query!(
            r"
INSERT INTO agent_sdk_outbox
    (id, kind, thread_id, event_id, sequence, status, payload_json,
     created_at, next_attempt_at, attempt_count, max_attempts)
VALUES (?1, ?2, ?3, NULL, NULL, 'pending', ?4, ?5, ?5, 0, ?6)
",
            id_str,
            kind_str,
            thread_id_key,
            payload_json,
            now,
            max_attempts_i64,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert task_wakeup outbox row for task {task_id}"))?;

        Ok(id)
    }

    /// Commit the terminal `Cancelled` marker for one cancelled
    /// blocking root inside `cancel_tree`'s transaction: one committed
    /// event on the root's own thread plus its coalesced
    /// `thread_events_available` outbox advisory (issue #354). See the
    /// Postgres analogue for the full contract; `SQLite` serializes
    /// writers via `BEGIN IMMEDIATE`, so no lock ordering applies.
    async fn insert_cancelled_marker_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        task_id: &AgentTaskId,
        continuation_usage: Option<TokenUsage>,
        reason: TerminalReason,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        Self::bootstrap_thread_row_tx(tx, thread_id, now).await?;
        let start_seq = Self::next_event_sequence_tx(tx, thread_id).await?;
        let thread = Self::get_thread_tx(tx, thread_id)
            .await?
            .with_context(|| format!("thread {thread_id} missing after bootstrap"))?;

        let turn = usize::try_from(thread.committed_turns).unwrap_or(0);
        let usage = continuation_usage.unwrap_or(thread.total_usage);
        let mut committed = Self::insert_events_tx(
            tx,
            thread_id,
            vec![
                AgentEvent::cancelled_with_reason(turn, usage, reason)
                    .with_emitter_task_id(task_id.as_str()),
            ],
            start_seq,
            now,
        )
        .await?;
        Self::insert_thread_events_outbox_row_tx(
            tx,
            &committed,
            DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        )
        .await?;
        committed.pop().context("cancelled marker batch was empty")
    }

    async fn insert_subagent_spawn_event_tx(
        tx: &mut Transaction<'_, Sqlite>,
        subagent: &SubagentSpawn,
        parent_task: &AgentTask,
        (invocation, child_root): (&AgentTask, &AgentTask),
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        let event = subagent_started_event(
            subagent.event.clone(),
            &subagent.spawn,
            invocation,
            child_root,
        );
        let start_seq = Self::next_event_sequence_tx(tx, &parent_task.thread_id).await?;
        let mut committed =
            Self::insert_events_tx(tx, &parent_task.thread_id, vec![event], start_seq, now).await?;
        Self::insert_thread_events_outbox_row_tx(tx, &committed, subagent.outbox_max_attempts, now)
            .await?;
        committed.pop().context("spawn event batch was empty")
    }

    async fn commit_thread_creation_atomic_inner(
        &self,
        params: ThreadCreationCommit,
    ) -> Result<ThreadCreationCommitted> {
        let mut tx = self.begin().await?;
        match params.naming {
            ThreadNaming::Minted => {
                Self::bootstrap_thread_row_tx(&mut tx, &params.thread_id, params.now).await?;
            }
            ThreadNaming::CallerNamed => {
                let claim = Self::claim_thread_creation_tx(
                    &mut tx,
                    &params.thread_id,
                    &ThreadCreation::Create,
                    params.now,
                )
                .await?;
                match claim {
                    ThreadCreationOutcome::Created => {}
                    ThreadCreationOutcome::Existing => {
                        tx.commit()
                            .await
                            .context("commit idempotent thread creation lookup")?;
                        return Ok(ThreadCreationCommitted::Existing);
                    }
                }
            }
        }
        let thread = Self::get_thread_tx(&mut tx, &params.thread_id)
            .await?
            .context("created thread missing after bootstrap")?;
        Self::get_message_head_tx(&mut tx, &params.thread_id, params.now).await?;

        agent_server::fail_point!("atomic_create.after_rows");

        let start_seq = Self::next_event_sequence_tx(&mut tx, &params.thread_id).await?;
        let mut committed = Self::insert_events_tx(
            &mut tx,
            &params.thread_id,
            vec![params.event],
            start_seq,
            params.now,
        )
        .await?;
        Self::insert_thread_events_outbox_row_tx(
            &mut tx,
            &committed,
            params.outbox_max_attempts,
            params.now,
        )
        .await?;
        let committed_event = committed
            .pop()
            .context("thread creation event batch was empty")?;
        tx.commit()
            .await
            .context("commit sqlite thread creation transaction")?;
        Ok(ThreadCreationCommitted::Created(Box::new(
            ThreadCreationRows {
                thread,
                committed_event,
            },
        )))
    }

    /// Atomic fork transaction for `SQLite`.
    ///
    /// Wraps the entire write set the fork RPC handler hands us
    /// (thread aggregate bootstrap + projection rewrite +
    /// `committed_turns` mirroring + checkpoint insert + event
    /// re-commit) inside one transaction. A crash or rolled-back
    /// transaction leaves the destination thread in the
    /// not-created state — never partially-built — so the gRPC
    /// handler's idempotency replay can safely retry under the
    /// same `request_id` after a transport blip without seeing a
    /// half-finished fork.
    async fn commit_fork_atomic_inner(
        &self,
        params: ForkCommitParams,
    ) -> Result<ThreadCreationOutcome> {
        use agent_server::journal::message::MessageProjection;

        let mut tx = self.begin().await?;

        let source = Self::get_thread_tx(&mut tx, &params.source_thread_id).await?;
        crate::fence_signal::fence_refused(
            agent_server::journal::thread::FencedWriter::ForkSource,
            agent_server::journal::Fence::FORK_SOURCE
                .check(&params.source_thread_id, source.as_ref()),
        )?;

        // Claim caller-minted destinations by primary key inside the same
        // transaction as the fork copy. Matching retries are no-ops; a
        // conflicting origin is typed and rolls back without mutation.
        let creation_outcome = if let Some(creation) = params.creation.as_ref() {
            Self::claim_thread_creation_tx(&mut tx, &params.new_thread_id, creation, params.now)
                .await?
        } else {
            Self::bootstrap_thread_row_tx(&mut tx, &params.new_thread_id, params.now).await?;
            ThreadCreationOutcome::Created
        };
        if creation_outcome == ThreadCreationOutcome::Existing {
            tx.commit()
                .await
                .context("commit idempotent sqlite fork lookup")?;
            return Ok(ThreadCreationOutcome::Existing);
        }

        // 1. The claim above inserted the destination aggregate row AND
        //    its creation-identity idempotency row (unlike
        //    `bootstrap_thread_row_tx`, which only writes the thread row).

        // 2. Mirror `committed_turns` by repeatedly applying
        //    `Thread::apply_committed_turn`. Each call advances
        //    `committed_turns` by one and folds the supplied
        //    `TokenUsage` into the aggregate's running
        //    `total_usage`. Only the final iteration carries the
        //    full `cumulative_total_usage` from the source's
        //    snapshot at the fork boundary so the destination
        //    lands at exactly that total without distributing
        //    arbitrary per-turn usage values along the way (the
        //    per-turn usage is already preserved on the source's
        //    checkpoint chain — the fork only needs the cumulative
        //    end state to keep cost reporting honest).
        let mut thread_row = Self::get_thread_tx(&mut tx, &params.new_thread_id)
            .await?
            .context("forked thread row missing after bootstrap")?;
        if params.committed_turns > 0 {
            let zero = TokenUsage::default();
            for turn_index in 0..params.committed_turns {
                let usage_for_this_turn = if turn_index + 1 == params.committed_turns {
                    &params.cumulative_total_usage
                } else {
                    &zero
                };
                thread_row = thread_row
                    .apply_committed_turn(usage_for_this_turn, params.now)
                    .context("advancing forked thread aggregate inside sqlite fork transaction")?;
            }
            Self::upsert_thread_tx(&mut tx, &thread_row).await?;
        }

        // 3. Seed the projection. `replace_history` builds a fresh
        //    `MessageProjection` keyed on `new_thread_id` so the
        //    bumped `version` is consistent with what
        //    `recover_thread`'s draft / committed merge expects.
        // Read (and thereby CREATE, on the empty-messages path too) the
        // destination's projection row — the get-or-create side effect
        // is load-bearing for turn-zero forks, which must still leave a
        // projection row behind.
        let projection_before =
            Self::get_message_head_tx(&mut tx, &params.new_thread_id, params.now).await?;
        if !params.messages.is_empty() {
            let updated_projection =
                MessageProjection::replace_history(projection_before, params.messages, params.now);
            Self::upsert_message_head_tx(&mut tx, &updated_projection).await?;
        }

        // 4. Mirror the source's checkpoint at the fork-point turn.
        if let Some(checkpoint_params) = params.checkpoint {
            let checkpoint = agent_server::journal::checkpoint::Checkpoint::new(checkpoint_params)
                .context("constructing forked checkpoint")?;
            Self::insert_checkpoint_tx(&mut tx, &checkpoint).await?;
        }

        agent_server::fail_point!("atomic_fork.after_rows");

        // 5. Commit the destination journal seed (`events[0]` is the
        //    ThreadCreated built by the params constructor) plus the
        //    coalesced outbox advisory — same transaction as every row
        //    above, which is the point of this committer.
        let start_seq = Self::next_event_sequence_tx(&mut tx, &params.new_thread_id).await?;
        let committed = Self::insert_events_tx(
            &mut tx,
            &params.new_thread_id,
            params.events,
            start_seq,
            params.now,
        )
        .await?;
        Self::insert_thread_events_outbox_row_tx(
            &mut tx,
            &committed,
            params.outbox_max_attempts,
            params.now,
        )
        .await?;

        tx.commit()
            .await
            .context("commit sqlite fork transaction")?;
        Ok(creation_outcome)
    }

    /// Enforce [`CompletedTurnCommit::owner_guard`] on the task row
    /// inside the completed-turn transaction (issue #354): the worker
    /// passes it on every root completion, so — under `SQLite`'s exclusive
    /// write transaction — a cancellation or lease loss since the worker
    /// last checked rejects the whole turn here, with its settle, instead
    /// of committing a turn under a task the worker no longer owns. A
    /// `None` guard (callers that complete no task) is a no-op.
    async fn enforce_commit_owner_guard_tx(
        tx: &mut Transaction<'_, Sqlite>,
        task_id: &AgentTaskId,
        owner_guard: Option<&CommitOwnerGuard>,
    ) -> Result<()> {
        let Some(guard) = owner_guard else {
            return Ok(());
        };
        let current = Self::load_task_tx(tx, task_id).await?.ok_or_else(|| {
            anyhow::Error::new(LostCommitOwnership {
                task_id: task_id.clone(),
            })
        })?;
        if current.status != TaskStatus::Running
            || current.worker_id.as_ref() != Some(&guard.worker_id)
            || current.lease_id.as_ref() != Some(&guard.lease_id)
        {
            return Err(anyhow::Error::new(LostCommitOwnership {
                task_id: task_id.clone(),
            }));
        }
        Ok(())
    }

    /// Atomic completed-turn transaction for `SQLite`.
    /// Complete boundary-injection rows Queued→Completed inside `tx`, so
    /// delivery and the enclosing durable fact commit or roll back
    /// together. Rows already completed (a replayed commit) are skipped.
    async fn complete_injections_tx(
        tx: &mut Transaction<'_, Sqlite>,
        ids: &[AgentTaskId],
        now: OffsetDateTime,
    ) -> Result<()> {
        for id in ids {
            let Some(old) = Self::load_task_tx(tx, id).await? else {
                continue;
            };
            if old.kind != TaskKind::InputInjection || old.status != TaskStatus::Queued {
                continue;
            }
            let completed = old
                .complete_input_injection(now)
                .context("complete input injection transition failed")?;
            Self::update_task_tx(tx, &completed).await?;
        }
        Ok(())
    }

    async fn commit_completed_turn_atomic_inner(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome> {
        params.validate_extra_writes()?;
        let extra = params.extra_transition();
        let mut tx = self.begin().await?;

        Self::bootstrap_thread_row_tx(&mut tx, &params.thread_id, params.now).await?;
        let old_attempt = Self::get_attempt_tx(&mut tx, &params.turn_attempt_id)
            .await?
            .ok_or_else(|| anyhow!("attempt not found: {}", params.turn_attempt_id))?;
        let closed_attempt = old_attempt
            .close(params.close_attempt_params, params.now)
            .context("close attempt inside sqlite completed-turn transaction")?;
        Self::update_attempt_tx(&mut tx, &closed_attempt).await?;

        let old_thread = Self::get_thread_tx(&mut tx, &params.thread_id)
            .await?
            .context("thread missing after bootstrap")?;
        old_thread.require_active(ThreadOperation::Commit)?;
        // Stale turn double-commit guard. The transaction holds the
        // write lock on the thread row; assert it is still positioned to
        // produce `expected_turn` before incrementing. A stale-lease
        // worker that lost the race to another worker fails here instead
        // of durably double-committing the turn (pure in-Rust comparison
        // on the already-loaded row — no extra query).
        if old_thread.committed_turns.saturating_add(1) != params.expected_turn {
            return Err(anyhow::Error::new(StaleTurnCommit {
                expected_turn: params.expected_turn,
                committed_turns: old_thread.committed_turns,
            }));
        }

        Self::enforce_commit_owner_guard_tx(&mut tx, &params.task_id, params.owner_guard.as_ref())
            .await?;
        let thread = old_thread
            .apply_committed_turn(&params.turn_usage, params.now)
            .context("advance thread aggregate inside sqlite completed-turn transaction")?;
        Self::upsert_thread_tx(&mut tx, &thread).await?;

        let projection_before =
            Self::get_message_head_tx(&mut tx, &params.thread_id, params.now).await?;
        // Append the committed turn's messages AND clear any
        // in-flight draft in the same transaction. Without the
        // draft clear, a crash window between this transaction and
        // an out-of-band `clear_draft` call would let the next
        // recovery view see committed turn N's messages duplicated
        // through the still-populated draft slot.
        let updated_projection = projection_before
            .append_committed(params.messages.clone(), params.now)
            .context("append committed messages inside sqlite completed-turn transaction")?
            .clear_draft(params.now);

        Self::insert_message_commit_tx(
            &mut tx,
            &params.thread_id,
            thread.committed_turns,
            &params.task_id,
            updated_projection.version,
            &params.messages,
            params.now,
        )
        .await?;
        Self::upsert_message_head_tx(&mut tx, &updated_projection).await?;

        let checkpoint = Checkpoint::new(NewCheckpointParams {
            kind: params.checkpoint_kind,
            thread_id: params.thread_id.clone(),
            turn_number: thread.committed_turns,
            task_id: params.task_id,
            messages: updated_projection.messages,
            agent_state_snapshot: params.agent_state_snapshot,
            turn_usage: params.turn_usage,
            now: params.now,
        })?;
        Self::insert_checkpoint_tx(&mut tx, &checkpoint).await?;

        // Phase 10 · D: commit the turn's lifecycle events AND the single
        // coalesced advisory outbox row inside this SAME transaction so a
        // crash can never leave a committed turn with zero persisted
        // events (or a committed-but-unpublished event batch).
        let committed_events = if params.events.is_empty() {
            Vec::new()
        } else {
            let start_seq = Self::next_event_sequence_tx(&mut tx, &params.thread_id).await?;
            let committed = Self::insert_events_tx(
                &mut tx,
                &params.thread_id,
                params.events,
                start_seq,
                params.now,
            )
            .await
            .context("insert lifecycle events inside sqlite completed-turn transaction")?;
            Self::insert_thread_events_outbox_row_tx(
                &mut tx,
                &committed,
                params.outbox_max_attempts,
                params.now,
            )
            .await
            .context("insert advisory outbox row inside sqlite completed-turn transaction")?;
            committed
        };

        // Failpoint (11 · A): simulate a crash at the true atomic-commit
        // boundary — the events and the coalesced advisory outbox row are
        // staged in `tx` but `tx.commit()` has not yet run, so the whole
        // turn (state + events) will roll back. Recovery must replay
        // idempotently. No-op (and not compiled) without the `failpoints`
        // feature.
        agent_server::fail_point!("commit.before_event_commit");

        Self::complete_injections_tx(&mut tx, &params.delivered_injection_ids, params.now).await?;
        let (settled, extra_documents) =
            Self::settle_completed_turn_tx(&mut tx, &extra, params.now).await?;

        tx.commit()
            .await
            .context("commit sqlite completed-turn transaction")?;
        Self::record_turn_documents(extra_documents);

        Ok(CommitOutcome {
            closed_attempt,
            thread,
            checkpoint,
            committed_events,
            settled,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// AgentTaskStore
// ─────────────────────────────────────────────────────────────────────

/// The thread's active (non-queued, non-terminal) root turn, if any.
///
/// Generic over the executor so the admission transaction and the pooled
/// trait method share one statement — including the `ORDER BY created_at, id`
/// that makes "the active root" the same row here as in the Postgres backend
/// rather than whatever the query plan happens to return first.
///
/// Runtime `query_as` per the `SQLite` carve-out in `CLAUDE.md`: this is a complex
/// SELECT mapping onto the `TaskRecord` `FromRow` type.
async fn active_root_for_thread_exec<'e, E>(
    executor: E,
    thread_id: &ThreadId,
) -> Result<Option<AgentTask>>
where
    E: sqlx::SqliteExecutor<'e>,
{
    let record = sqlx::query_as::<_, TaskRecord>(concat!(
        "SELECT ",
        task_columns!(),
        " FROM agent_sdk_tasks \
         WHERE thread_id = ?1 \
           AND kind = 'root_turn' \
           AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question') \
         ORDER BY created_at, id LIMIT 1",
    ))
    .bind(thread_key(thread_id))
    .fetch_optional(executor)
    .await
    .with_context(|| format!("load active root for {thread_id}"))?;
    record.map(TryInto::try_into).transpose()
}

fn validate_subagent_spawn_parent(
    parent: &AgentTask,
    parent_id: &AgentTaskId,
    worker: &WorkerId,
    lease: &LeaseId,
) -> Result<()> {
    if parent.status != TaskStatus::Running {
        let status = parent.status;
        return Err(anyhow!(
            "spawn rejected: task {parent_id} is not running (status {status:?})"
        ));
    }
    match &parent.worker_id {
        Some(current) if current == worker => {}
        _ => {
            return Err(anyhow!(
                "spawn rejected: worker mismatch on task {parent_id}"
            ));
        }
    }
    match &parent.lease_id {
        Some(current) if current == lease => {}
        _ => {
            return Err(anyhow!(
                "spawn rejected: lease mismatch on task {parent_id}"
            ));
        }
    }
    if parent.kind.is_leaf() {
        let parent_kind = parent.kind;
        return Err(anyhow!(
            "spawn rejected: parent {parent_id} is a leaf kind ({parent_kind:?}) and cannot spawn children"
        ));
    }
    Ok(())
}

/// Materialize the `(invocation, child_root)` pair for every subagent
/// entry in a batch, validating cross-entry thread uniqueness and
/// per-row availability inside the caller's transaction.
///
/// Shared by [`SpawnChildrenExt::spawn_subagent_batch`](agent_server::journal::SpawnChildrenExt::spawn_subagent_batch) and
/// [`SpawnChildrenExt::spawn_mixed_children`](agent_server::journal::SpawnChildrenExt::spawn_mixed_children) so the two paths cannot
/// drift on invocation-row construction.
///
/// Nothing is written here: the caller commits the rows only after the
/// parent transition succeeds, so a rejection on entry K leaves the
/// transaction free of partial fan-out.
async fn prepare_subagent_batch_rows_tx(
    tx: &mut Transaction<'_, Sqlite>,
    old_parent: &AgentTask,
    spawns: Vec<SubagentInvocationSpawn>,
    now: OffsetDateTime,
) -> Result<Vec<(AgentTask, AgentTask)>> {
    let mut seen_thread_ids: std::collections::HashSet<&ThreadId> =
        std::collections::HashSet::with_capacity(spawns.len());
    for spawn in &spawns {
        if !seen_thread_ids.insert(&spawn.child_thread_id) {
            return Err(anyhow!(
                "spawn rejected: duplicate child_thread_id {} in batch",
                spawn.child_thread_id
            ));
        }
    }

    let mut prepared: Vec<(AgentTask, AgentTask)> = Vec::with_capacity(spawns.len());
    for spawn in spawns {
        let SubagentInvocationSpawn {
            child_thread_id,
            spec,
            child_root_input,
            payload: _per_entry_payload,
            spawn_index,
            child_caller_metadata,
        } = spawn;

        let child_thread = SqliteDurableStore::get_thread_tx(tx, &child_thread_id).await?;
        if child_thread.is_none() {
            return Err(anyhow!(
                "spawn rejected: child thread {child_thread_id} does not exist"
            ));
        }
        let child_thread_key = thread_key(&child_thread_id);
        let existing_task = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 LIMIT 1",
            child_thread_key
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("check existing tasks for child thread {child_thread_id}"))?
        .flatten();
        if existing_task.is_some() {
            return Err(anyhow!(
                "spawn rejected: child thread {child_thread_id} already has tasks"
            ));
        }

        let child_root = AgentTask::new_root_turn_with_optional_caller(
            child_thread_id.clone(),
            child_root_input,
            child_caller_metadata,
            now,
            // Root-turn budget: see `spawn_subagent_batch_tx`'s child root.
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
        );
        let existing = SqliteDurableStore::load_task_tx(tx, &child_root.id).await?;
        if existing.is_some() {
            return Err(anyhow!(
                "spawn rejected: child root id {} already exists",
                child_root.id
            ));
        }

        let invocation = AgentTask::new_subagent_invocation(
            old_parent,
            SubagentInvocationState {
                spec,
                child_thread_id,
                child_root_task_id: child_root.id.clone(),
                awaiting_prompt: None,
            },
            spawn_index,
            now,
            AgentTask::SUBAGENT_INVOCATION_MAX_ATTEMPTS,
        )
        .context("spawn rejected: new_subagent_invocation failed")?;
        let existing = SqliteDurableStore::load_task_tx(tx, &invocation.id).await?;
        if existing.is_some() {
            return Err(anyhow!(
                "spawn rejected: invocation id {} already exists",
                invocation.id
            ));
        }

        prepared.push((invocation, child_root));
    }
    Ok(prepared)
}

/// Shared by [`SpawnChildrenExt::spawn_mixed_children`](agent_server::journal::SpawnChildrenExt::spawn_mixed_children): builds the new
/// tool-runtime rows and resolves the re-attached survivors.
///
/// Nothing is written here: the caller commits the rows only after the
/// parent transition succeeds, so a rejection on any entry leaves the
/// transaction free of partial fan-out.
async fn prepare_mixed_tool_and_reattach_rows_tx(
    tx: &mut Transaction<'_, Sqlite>,
    old_parent: &AgentTask,
    parent_id: &AgentTaskId,
    tool_children: &[ToolChildSpawn],
    reattach: Vec<ReattachedChild>,
    child_otel_traceparent: Option<&str>,
    now: OffsetDateTime,
) -> Result<(Vec<AgentTask>, Vec<AgentTask>)> {
    let mut tool_rows: Vec<AgentTask> = Vec::with_capacity(tool_children.len());
    for tool in tool_children {
        let child = new_mixed_tool_child(old_parent, tool, child_otel_traceparent, now)?;
        let existing = SqliteDurableStore::load_task_tx(tx, &child.id).await?;
        if existing.is_some() {
            return Err(anyhow!(
                "spawn rejected: child id {} already exists",
                child.id
            ));
        }
        tool_rows.push(child);
    }

    let mut reattached_rows = Vec::with_capacity(reattach.len());
    let mut reattached_ids = BTreeSet::new();
    for attached in reattach {
        ensure!(
            reattached_ids.insert(attached.child_id.clone()),
            "spawn rejected: duplicate re-attach child {}",
            attached.child_id
        );
        let mut child = SqliteDurableStore::load_task_tx(tx, &attached.child_id)
            .await?
            .with_context(|| {
                format!(
                    "spawn rejected: re-attach child {} missing",
                    attached.child_id
                )
            })?;
        ensure!(
            child.parent_id.as_ref() == Some(parent_id),
            "spawn rejected: re-attach child {} is not a child of {parent_id}",
            attached.child_id
        );
        child.spawn_index = Some(attached.spawn_index);
        child.updated_at = now;
        child
            .validate()
            .context("spawn rejected: re-attach child validate failed")?;
        reattached_rows.push(child);
    }
    hold_tool_children_behind_siblings(
        &mut tool_rows,
        &reattached_rows,
        &mixed_spawn_waits(tool_children),
    )?;
    Ok((tool_rows, reattached_rows))
}

/// Applies a transition in one database transaction: the guards are read
/// first, then the writes in order, then each thread's coalesced outbox
/// row. Any error drops the transaction, so nothing was written.
/// Marks a commit that failed because the database stayed locked past the
/// busy timeout with [`agent_server::journal::LockConflict`] (`retries: 0`:
/// SQLite's busy handler waited, the store does not retry), so telemetry
/// counts it as `outcome=lock_conflict`, as on Postgres.
fn mark_busy(
    result: Result<agent_server::journal::Committed, agent_server::journal::Rejected>,
) -> Result<agent_server::journal::Committed, agent_server::journal::Rejected> {
    match result {
        Err(agent_server::journal::Rejected::Store(error))
            if is_busy(&error)
                && error
                    .downcast_ref::<agent_server::journal::LockConflict>()
                    .is_none() =>
        {
            Err(agent_server::journal::Rejected::Store(error.context(
                agent_server::journal::LockConflict { retries: 0 },
            )))
        }
        result => result,
    }
}

/// How long [`SqliteDurableStore::begin`] waits before each retry of a
/// `BEGIN IMMEDIATE` that found the database locked, on top of the
/// connection's own `busy_timeout` wait.
const BEGIN_BUSY_BACKOFF: [std::time::Duration; 3] = [
    std::time::Duration::from_millis(50),
    std::time::Duration::from_millis(200),
    std::time::Duration::from_millis(800),
];

/// Run one database operation, retrying it after the database stayed locked
/// past the connection's busy timeout: [`BEGIN_BUSY_BACKOFF`] sleeps, each
/// counted in `retries_seen` and on
/// `agent_server.journal.transition.lock_conflict_retries`. Only for an
/// operation that is safe to repeat: one that found the database locked ran
/// nothing (a `BEGIN`, or a single autocommit statement). The last failure
/// carries [`agent_server::journal::LockConflict`] with the retry count.
pub(super) async fn retry_busy<T, Fut>(
    retries_seen: &AtomicU64,
    what: &'static str,
    mut op: impl FnMut() -> Fut,
) -> Result<T>
where
    Fut: std::future::Future<Output = std::result::Result<T, sqlx::Error>>,
{
    let started = std::time::Instant::now();
    let mut retries = 0;
    loop {
        match op().await.map_err(anyhow::Error::from) {
            Err(error) if is_busy(&error) => {
                let Some(backoff) = BEGIN_BUSY_BACKOFF.get(retries) else {
                    tracing::warn!(
                        what,
                        retries,
                        waited_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                        %error,
                        "a sqlite write stayed locked through every retry"
                    );
                    return Err(error.context(agent_server::journal::LockConflict {
                        retries: u32::try_from(retries).unwrap_or(u32::MAX),
                    }));
                };
                retries += 1;
                retries_seen.fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "otel")]
                agent_server::observability::ServerMetrics::global()
                    .record_transition_lock_conflict_retry(
                        agent_server::observability::attrs::BACKEND_SQLITE,
                    );
                tracing::warn!(what, retries, %error, "retrying a sqlite write that found the database locked");
                tokio::time::sleep(*backoff).await;
            }
            result => return result,
        }
    }
}

/// Whether `error` is `SQLITE_BUSY` (5) or `SQLITE_LOCKED` (6), in any of
/// their extended forms (the primary code is the low byte).
fn is_busy(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<sqlx::Error>()
            .and_then(sqlx::Error::as_database_error)
            .and_then(sqlx::error::DatabaseError::code)
            .and_then(|code| code.parse::<i32>().ok())
            .is_some_and(|code| matches!(code & 0xff, 5 | 6))
    })
}

#[async_trait]
impl agent_server::journal::DurableStore for SqliteDurableStore {
    async fn commit(
        &self,
        transition: agent_server::journal::Transition,
        now: OffsetDateTime,
    ) -> Result<agent_server::journal::Committed, agent_server::journal::Rejected> {
        #[cfg(feature = "otel")]
        let (started, guard_count) = (std::time::Instant::now(), transition.guards.len());
        let result = mark_busy(self.commit_transition(transition, now).await);
        #[cfg(feature = "otel")]
        agent_server::observability::ServerMetrics::global().record_transition(
            agent_server::observability::attrs::BACKEND_SQLITE,
            guard_count,
            &result,
            started.elapsed().as_secs_f64(),
        );
        if let Ok(committed) = &result {
            TerminalRootTurn::record_settled(&committed.settled);
        }
        result
    }

    async fn effect(&self, key: &EffectKey) -> Result<Option<EffectRecord>> {
        Self::load_effect(&self.pool, key).await
    }

    async fn effects_for_task(&self, owner: &AgentTaskId) -> Result<Vec<EffectRecord>> {
        let records = sqlx::query_as::<_, EffectRecordRow>(
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at,
    effect_kind, attempt, resolution, captured_output, journal_anchor, thread_id
FROM agent_sdk_execution_intents
WHERE child_task_id = ?1
ORDER BY created_at, operation_id
",
        )
        .bind(owner.as_str())
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list effects of task {owner}"))?;
        records
            .into_iter()
            .map(EffectRecordRow::into_record)
            .collect()
    }
}

impl SqliteDurableStore {
    async fn commit_transition(
        &self,
        transition: agent_server::journal::Transition,
        now: OffsetDateTime,
    ) -> Result<agent_server::journal::Committed, agent_server::journal::Rejected> {
        let transition = transition.extended(self.extension.as_deref())?;
        transition.validate()?;
        let mut tx = self.begin().await?;
        Self::check_declared_threads_tx(&mut tx, &transition, true, now).await?;
        for (index, guard) in transition.guards.iter().enumerate() {
            if let Some(task) = guard.task() {
                let stored = Self::load_task_tx(&mut tx, task).await?;
                guard.check(index, stored.as_ref())?;
                if guard.leased() && Self::lease_superseded_tx(&mut tx, task).await? {
                    return Err(agent_server::journal::Rejected::Guard {
                        index,
                        reason: agent_server::journal::EpochSuperseded::of_lease(task).to_string(),
                    });
                }
            }
        }
        Self::check_effects_tx(&mut tx, &transition).await?;
        Self::check_declared_threads_tx(&mut tx, &transition, false, now).await?;
        let mut committed = agent_server::journal::Committed::default();
        let mut fan_ins = agent_server::journal::FanIns::default();
        for (index, write) in agent_server::journal::writes_in_apply_order(&transition) {
            Self::apply_write_tx(
                &mut tx,
                &transition,
                write,
                self.extension.as_deref(),
                now,
                (&mut committed, &mut fan_ins),
            )
            .await
            .map_err(|error| {
                agent_server::journal::Rejected::from_write_error(
                    error,
                    transition.guards.len(),
                    index,
                )
            })?;
        }
        agent_server::fail_point!("transition.before_commit");
        tx.commit().await.context("commit transition")?;
        if let Some(record) = transition.begun_epochs().last() {
            self.set_current_epoch(record.id.clone());
        }
        Ok(committed)
    }
}

impl SqliteDurableStore {
    /// Count the documents a committed turn wrote.
    fn record_turn_documents(documents: Vec<agent_server::journal::DocumentOutcome>) {
        #[cfg(feature = "otel")]
        agent_server::observability::ServerMetrics::global().record_documents(
            agent_server::observability::attrs::BACKEND_SQLITE,
            &documents,
        );
        drop(documents);
    }

    /// Apply a completed turn's extra writes inside its transaction,
    /// through the same write arms as a transition, and return its settle
    /// and the documents it wrote.
    async fn settle_completed_turn_tx(
        tx: &mut Transaction<'_, Sqlite>,
        extra: &agent_server::journal::Transition,
        now: OffsetDateTime,
    ) -> Result<CompletedTurnExtra> {
        let mut committed = agent_server::journal::Committed::default();
        for write in extra.writes.iter().cloned() {
            Self::apply_write_tx(
                tx,
                extra,
                write,
                None,
                now,
                (
                    &mut committed,
                    &mut agent_server::journal::FanIns::default(),
                ),
            )
            .await?;
        }
        if !committed.settled.is_empty() {
            agent_server::fail_point!("commit.completed_turn.settled");
        }
        // After every extra write, before the commit: a crash here must
        // leave none of them.
        agent_server::fail_point!("commit.completed_turn.extra_written");
        Ok((committed.settled.pop(), committed.documents))
    }

    /// Apply one effect write of a transition inside its transaction.
    async fn apply_effect_tx(
        tx: &mut Transaction<'_, Sqlite>,
        effect: EffectWrite,
        now: OffsetDateTime,
    ) -> Result<()> {
        match effect {
            EffectWrite::Start(intent) => {
                let anchor = Self::next_event_sequence_tx(tx, &intent.thread).await?;
                let record = EffectRecord::started(intent, Some(anchor), now);
                Self::insert_effect_tx(tx, &record).await?;
                agent_server::fail_point!("effect.start_written");
            }
            EffectWrite::Finish { key, owner, state } => {
                Self::finish_effect_tx(tx, &key, &owner, &state, now).await?;
                agent_server::fail_point!("effect.finish_written");
            }
        }
        Ok(())
    }

    /// Apply a [`Write::ParkInvocation`](agent_server::journal::Write::ParkInvocation)
    /// or [`Write::WakeInvocation`](agent_server::journal::Write::WakeInvocation)
    /// inside the transaction. A wake whose invocation no longer waits on its
    /// prompt is an expectation failure: the transition is refused whole.
    async fn invocation_step_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
        write: &agent_server::journal::Write,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        use agent_server::journal::Write;

        let (invocation, prompt) = match write {
            Write::ParkInvocation { invocation, prompt }
            | Write::WakeInvocation { invocation, prompt } => (invocation, prompt),
            other => return Err(anyhow!("not an invocation prompt step: {other:?}")),
        };
        let old = Self::load_task_tx(tx, invocation)
            .await?
            .ok_or_else(|| anyhow!("invocation {invocation} does not exist"))?;
        if matches!(write, Write::ParkInvocation { .. }) {
            let (worker, lease) = transition
                .lease_for(invocation)
                .ok_or_else(|| anyhow!("park rejected: no lease guard on {invocation}"))?;
            let parked = old
                .park_on_prompt(worker, lease, prompt.clone(), now)
                .with_context(|| format!("park rejected for invocation {invocation}"))?;
            Self::update_task_tx(tx, &parked).await?;
            agent_server::fail_point!("invocation.park_written");
            return Ok(parked);
        }
        if old.state.awaited_prompt() != Some(prompt.as_str()) {
            return Err(agent_server::journal::document::ExpectationFailed {
                reason: format!(
                    "invocation {invocation} does not wait on prompt {prompt} ({:?})",
                    old.state.awaited_prompt()
                ),
            }
            .into());
        }
        let woken = old
            .wake_from_prompt(prompt, now)
            .with_context(|| format!("wake rejected for invocation {invocation}"))?;
        Self::update_task_tx(tx, &woken).await?;
        agent_server::fail_point!("invocation.wake_written");
        Ok(woken)
    }

    /// Apply a [`Write::PauseOnConfirmation`](agent_server::journal::Write::PauseOnConfirmation).
    async fn apply_pause_tx(
        tx: &mut Transaction<'_, Sqlite>,
        write: agent_server::journal::Write,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let agent_server::journal::Write::PauseOnConfirmation {
            task,
            continuation,
            prepared_operation,
        } = write
        else {
            return Err(anyhow!("not a pause: {write:?}"));
        };
        Self::pause_on_confirmation_tx(tx, &task, *continuation, prepared_operation, now).await
    }

    /// Apply one write of a transition inside its transaction. `fan_ins`
    /// records the parents its settles fan in, for its queued steering.
    async fn apply_write_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
        write: agent_server::journal::Write,
        extension: Option<&dyn agent_server::journal::TransitionExtension>,
        now: OffsetDateTime,
        (committed, fan_ins): (
            &mut agent_server::journal::Committed,
            &mut agent_server::journal::FanIns,
        ),
    ) -> Result<()> {
        use agent_server::journal::Write;

        match write {
            Write::SettleTask { task, outcome } => {
                let parked = Self::parked_watched_parents_tx(tx, transition).await?;
                let settled = Self::apply_settle_tx(tx, transition, &task, outcome, now).await?;
                fan_ins.record(&parked, &settled);
                committed.settled.push(settled);
                agent_server::fail_point!("transition.settle_written");
            }
            Write::RejectConfirmation { task, error } => committed
                .settled
                .push(Self::reject_confirmation_tx(tx, &task, error, now).await?),
            step @ (Write::ParkInvocation { .. } | Write::WakeInvocation { .. }) => committed
                .invocations
                .push(Self::invocation_step_tx(tx, transition, &step, now).await?),
            pause @ Write::PauseOnConfirmation { .. } => committed
                .paused
                .push(Self::apply_pause_tx(tx, pause, now).await?),
            Write::ReparkAfterSteering {
                task,
                worker,
                lease,
                payload,
                reattach,
            } => {
                let reparked = Self::repark_after_steering_tx(
                    tx,
                    &task,
                    (&worker, &lease),
                    *payload,
                    reattach,
                    now,
                )
                .await?;
                committed.reparked.push(reparked);
            }
            Write::Effect(effect) => Self::apply_effect_tx(tx, effect, now).await?,
            Write::BeginEpoch(record) => Self::insert_epoch_tx(tx, &record).await?,
            Write::TurnAttempt(attempt) => Self::apply_attempt_write_tx(tx, attempt, now).await?,
            Write::AppendCompaction {
                thread,
                messages,
                source_message_count,
                retained_message_count,
            } => {
                let compacted = Self::append_compaction_tx(
                    tx,
                    &thread,
                    messages,
                    source_message_count,
                    retained_message_count,
                    now,
                )
                .await?;
                committed.compacted.push(compacted);
            }
            Write::AppendEvents { thread, events } => {
                if events.is_empty() {
                    return Ok(());
                }
                agent_server::fail_point!("transition.before_events");
                let start = Self::next_event_sequence_tx(tx, &thread).await?;
                let batch = Self::insert_events_tx(tx, &thread, events, start, now).await?;
                Self::insert_thread_events_outbox_row_tx(
                    tx,
                    &batch,
                    DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                    now,
                )
                .await?;
                committed.events.extend(batch);
            }
            Write::SpawnChildren { parent, spawn } => {
                let spawned = Self::apply_spawn_tx(tx, transition, &parent, *spawn, now).await?;
                committed.events.extend(spawned.events.iter().cloned());
                committed.spawned.push(spawned);
            }
            Write::Document(document) => {
                let outcome = Self::apply_document_tx(tx, document, now).await?;
                committed.documents.push(outcome);
            }
            Write::EnqueueSteeringResume { parent, steering } => {
                Self::apply_steering_tx(
                    tx,
                    extension,
                    &parent,
                    steering,
                    now,
                    (committed, fan_ins),
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Apply a [`Write::EnqueueSteeringResume`](agent_server::journal::Write::EnqueueSteeringResume)
    /// and report it: the woken parent, and a folded child in place of the
    /// row its settle reported.
    async fn apply_steering_tx(
        tx: &mut Transaction<'_, Sqlite>,
        extension: Option<&dyn agent_server::journal::TransitionExtension>,
        parent: &AgentTaskId,
        steering: agent_server::journal::SteeringInput,
        now: OffsetDateTime,
        (committed, fan_ins): (
            &mut agent_server::journal::Committed,
            &agent_server::journal::FanIns,
        ),
    ) -> Result<()> {
        let (woken, folded) = Self::enqueue_steering_resume_tx(
            tx,
            extension,
            parent,
            fan_ins.child_of(parent),
            steering,
            now,
        )
        .await?;
        committed.steering_resumed.push(woken);
        if let Some(folded) = folded
            && let Some((row, _)) = committed
                .settled
                .iter_mut()
                .find(|(row, _)| row.id == folded.id)
        {
            *row = folded;
        }
        Ok(())
    }

    /// The parents of `transition`'s queued steering writes that wait on
    /// their children now, read before a settle so it can tell a fan-in.
    async fn parked_watched_parents_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
    ) -> Result<Vec<AgentTaskId>> {
        let mut parked = Vec::new();
        for parent in agent_server::journal::FanIns::watched(transition) {
            if Self::load_task_tx(tx, parent)
                .await?
                .is_some_and(|row| row.status == TaskStatus::WaitingOnChildren)
            {
                parked.push(parent.clone());
            }
        }
        Ok(parked)
    }

    /// Fold the steering queued for `parent` into the stored result of
    /// `child`, whose settle in this transaction fanned it in: the queue at
    /// the revision it was read at, then the child's result. Returns the
    /// rewritten child, or `None` with nothing written.
    async fn fold_queued_steering_tx(
        tx: &mut Transaction<'_, Sqlite>,
        extension: Option<&dyn agent_server::journal::TransitionExtension>,
        parent: &AgentTask,
        child: &AgentTaskId,
        queue: agent_server::journal::DocumentKey,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let settled = Self::load_task_tx(tx, child)
            .await?
            .ok_or_else(|| anyhow!("steering fold rejected: task {child} does not exist"))?;
        let stored = Self::get_document_on(tx, &queue).await?;
        let Some((result, write)) = agent_server::journal::fold_queued_steering(
            "sqlite",
            extension,
            parent,
            &settled,
            queue,
            stored.as_ref(),
        )?
        else {
            return Ok(None);
        };
        Self::apply_document_tx(tx, write, now).await?;
        let mut folded = settled;
        folded.result_payload = Some(result);
        Self::update_task_tx(tx, &folded).await?;
        agent_server::fail_point!("transition.steering_folded");
        Ok(Some(folded))
    }

    /// Wake a parent parked on its children for a steering exchange. Only
    /// a parent still waiting on its children is woken; any other status
    /// declines with `None`, so a concurrent fan-in never races the wake.
    /// A [`SteeringInput::Queued`](agent_server::journal::SteeringInput::Queued)
    /// reads its queue in this transaction, after the park, and writes the
    /// taken queue back in it; after a settle that fanned the parent in
    /// (`fanned_in` names that child) it folds the queue into the child's
    /// result instead. Returns the woken parent and the folded child.
    async fn enqueue_steering_resume_tx(
        tx: &mut Transaction<'_, Sqlite>,
        extension: Option<&dyn agent_server::journal::TransitionExtension>,
        parent_id: &AgentTaskId,
        fanned_in: Option<&AgentTaskId>,
        steering: agent_server::journal::SteeringInput,
        now: OffsetDateTime,
    ) -> Result<(Option<AgentTask>, Option<AgentTask>)> {
        let old = Self::load_task_tx(tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("steering wake rejected: task {parent_id} does not exist"))?;
        let steering = match steering {
            agent_server::journal::SteeringInput::Blocks(steering) => steering,
            queued @ agent_server::journal::SteeringInput::Queued { .. } => {
                let queue = queued
                    .queue_on(&old.thread_id)
                    .context("a queued steering input names its queue")?;
                if old.status != TaskStatus::WaitingOnChildren
                    && let Some(child) = fanned_in
                {
                    let folded =
                        Self::fold_queued_steering_tx(tx, extension, &old, child, queue, now)
                            .await?;
                    return Ok((None, folded));
                }
                let stored = Self::get_document_on(tx, &queue).await?;
                let Some((steering, write)) = agent_server::journal::take_queued_steering(
                    "sqlite",
                    extension,
                    &old,
                    queue,
                    stored.as_ref(),
                )?
                else {
                    return Ok((None, None));
                };
                Self::apply_document_tx(tx, write, now).await?;
                steering
            }
        };
        if old.status != TaskStatus::WaitingOnChildren {
            return Ok((None, None));
        }
        let woken = old
            .begin_steering_resume(steering, now)
            .context("steering wake rejected: begin_steering_resume transition failed")?;
        Self::update_task_tx(tx, &woken).await?;
        Ok((Some(woken), None))
    }

    /// Settle a leased task with the existing terminal-transition helper.
    async fn apply_settle_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
        task: &AgentTaskId,
        outcome: agent_server::journal::Settle,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        use agent_server::journal::Settle;

        let (worker, lease) = transition
            .lease_for(task)
            .map(|(worker, lease)| (worker.clone(), lease.clone()))
            .ok_or_else(|| anyhow!("settling task {task} needs a leased guard"))?;
        match outcome {
            Settle::Complete { result } => {
                Self::apply_task_terminal_transition_tx(
                    tx,
                    task,
                    &worker,
                    &lease,
                    now,
                    "commit",
                    move |child| child.complete_with_result(result, now),
                )
                .await
            }
            Settle::Fail { error } => {
                Self::apply_task_terminal_transition_tx(
                    tx,
                    task,
                    &worker,
                    &lease,
                    now,
                    "commit",
                    move |child| child.fail(error, now),
                )
                .await
            }
            Settle::CompleteWithoutResult => {
                Self::apply_task_terminal_transition_tx(
                    tx,
                    task,
                    &worker,
                    &lease,
                    now,
                    "commit",
                    move |child| child.complete(now),
                )
                .await
            }
            Settle::FailWithReason {
                error,
                reason,
                result,
            } => {
                Self::apply_task_terminal_transition_tx(
                    tx,
                    task,
                    &worker,
                    &lease,
                    now,
                    "commit",
                    move |child| {
                        child
                            .fail_with_terminal_reason(error, reason, now)
                            .map(|failed| failed.with_failure_result(result))
                    },
                )
                .await
            }
        }
    }
}

impl SqliteDurableStore {
    /// Check the purge fences the transition declares for one side of its
    /// guards, creating the thread rows it declares with `create` there.
    /// Before the guards: a spawn's parent thread. After them: a spawn's
    /// child threads, and every append's thread (see
    /// [`agent_server::journal::Fence::after_guards`]). So a spawn checks
    /// the parent's thread, the parent's lease, then each child thread, the
    /// order the spawn methods used, and an append reports a stale lease
    /// before a purge.
    async fn check_declared_threads_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
        before_guards: bool,
        now: OffsetDateTime,
    ) -> Result<()> {
        let checked_now = |tier: ThreadTier, fence: Option<&agent_server::journal::Fence>| {
            let after = fence.is_some_and(agent_server::journal::Fence::after_guards)
                || tier == ThreadTier::Spawned;
            after != before_guards
        };
        for lock in transition.locks() {
            match lock {
                Lock::Thread {
                    thread,
                    tier,
                    fence,
                    create,
                } => {
                    if !checked_now(tier, fence.as_ref()) {
                        continue;
                    }
                    if create {
                        Self::bootstrap_thread_row_tx(tx, &thread, now).await?;
                    }
                    if let Some(fence) = fence {
                        let stored = Self::get_thread_tx(tx, &thread).await?;
                        fence.check(&thread, stored.as_ref())?;
                    }
                }
                Lock::ThreadOf {
                    task,
                    fence: Some(fence),
                } => {
                    if !checked_now(ThreadTier::Own, Some(&fence)) {
                        continue;
                    }
                    if let Some(task) = Self::load_task_tx(tx, &task).await? {
                        let stored = Self::get_thread_tx(tx, &task.thread_id).await?;
                        fence.check(&task.thread_id, stored.as_ref())?;
                    }
                }
                Lock::ThreadOf { fence: None, .. } | Lock::Task(_) | Lock::Document(_) => {}
            }
        }
        Ok(())
    }

    /// Apply a [`agent_server::journal::Write::SpawnChildren`]: the guard
    /// held the parent running under its lease, and the purge fence was
    /// checked before it.
    async fn apply_spawn_tx(
        tx: &mut Transaction<'_, Sqlite>,
        transition: &agent_server::journal::Transition,
        parent_id: &AgentTaskId,
        spawn: ChildSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedChildren> {
        let (worker, lease) = transition
            .lease_for(parent_id)
            .map(|(worker, lease)| (worker.clone(), lease.clone()))
            .ok_or_else(|| anyhow!("spawning under {parent_id} needs a leased guard"))?;
        let old_parent = Self::load_task_tx(tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, &worker, &lease)?;
        match spawn {
            ChildSpawn::Tools(tools) => {
                Self::spawn_tool_children_tx(tx, old_parent, tools, now).await
            }
            ChildSpawn::SubagentRows {
                spawn,
                delivered_injection_ids,
            } => {
                let batch = SubagentBatchSpawn {
                    delivered_injection_ids,
                    payload: spawn.payload.clone(),
                    spawns: vec![spawn],
                    events: Vec::new(),
                };
                Self::spawn_subagent_batch_tx(tx, old_parent, batch, now).await
            }
            ChildSpawn::Subagent(subagent) => {
                Self::spawn_subagent_tx(tx, old_parent, subagent, now).await
            }
            ChildSpawn::SubagentBatch(batch) => {
                Self::spawn_subagent_batch_tx(tx, old_parent, batch, now).await
            }
            ChildSpawn::Mixed { spawn, events } => {
                Self::spawn_mixed_children_tx(tx, old_parent, spawn, events, now).await
            }
        }
    }

    async fn spawn_tool_children_tx(
        tx: &mut Transaction<'_, Sqlite>,
        old_parent: AgentTask,
        ToolChildrenSpawn {
            specs,
            payload,
            child_otel_traceparent,
            delivered_injection_ids,
        }: ToolChildrenSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedChildren> {
        let waits = pure_spawn_waits(&specs)?;
        let mut children = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.into_iter().enumerate() {
            let mut child =
                AgentTask::new_child(&old_parent, TaskKind::ToolRuntime, now, spec.max_attempts)
                    .context("spawn rejected: new_child failed")?;
            child.spawn_index =
                Some(u32::try_from(idx).context("spawn rejected: batch index exceeds u32::MAX")?);
            child.otel_traceparent.clone_from(&child_otel_traceparent);
            let existing = Self::load_task_tx(tx, &child.id).await?;
            if existing.is_some() || children.iter().any(|e: &AgentTask| e.id == child.id) {
                return Err(anyhow!(
                    "spawn rejected: child id {} already exists",
                    child.id
                ));
            }
            children.push(child);
        }
        hold_tool_children_behind_siblings(&mut children, &[], &waits)?;
        let child_count = u32::try_from(children.len())
            .context("spawn rejected: child count exceeds u32::MAX")?;
        let child_ids = children.iter().map(|c| c.id.clone()).collect();
        let new_parent = old_parent
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(tx, &new_parent).await?;
        for child in &children {
            Self::insert_task_tx(tx, child).await?;
        }
        Self::complete_injections_tx(tx, &delivered_injection_ids, now).await?;
        Ok(SpawnedChildren {
            parent: new_parent,
            subagents: Vec::new(),
            tool_children: children,
            child_threads: Vec::new(),
            events: Vec::new(),
        })
    }

    async fn spawn_subagent_tx(
        tx: &mut Transaction<'_, Sqlite>,
        old_parent: AgentTask,
        subagent: SubagentSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedChildren> {
        let child_thread_id = subagent.spawn.child_thread_id.clone();
        let child_thread = Self::get_thread_tx(tx, &child_thread_id)
            .await?
            .context("spawned child thread missing after bootstrap")?;
        let child_thread_key = thread_key(&child_thread_id);
        let existing_task = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 LIMIT 1",
            child_thread_key
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("check existing tasks for child thread {child_thread_id}"))?
        .flatten();
        ensure!(
            existing_task.is_none(),
            "spawn rejected: child thread {child_thread_id} already has tasks",
        );

        let child_root = AgentTask::new_root_turn_with_optional_caller(
            child_thread_id.clone(),
            subagent.spawn.child_root_input.clone(),
            subagent.spawn.child_caller_metadata.clone(),
            now,
            // Child roots are ROOT TURNS and take the root-turn budget —
            // parity with the in-memory reference store, which spawns them
            // with `RuntimePolicy::server_default().max_attempts` (== this
            // constant, locked by a test). With `DEFAULT_MAX_ATTEMPTS` (1)
            // a single expired lease — one heartbeat starved past the
            // lease TTL under store contention — terminally killed a
            // running worker tree (`lease_expired_budget_exhausted`).
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
        );
        ensure!(
            Self::load_task_tx(tx, &child_root.id).await?.is_none(),
            "spawn rejected: child root id {} already exists",
            child_root.id,
        );
        let invocation = AgentTask::new_subagent_invocation(
            &old_parent,
            SubagentInvocationState {
                spec: subagent.spawn.spec.clone(),
                child_thread_id: child_thread_id.clone(),
                child_root_task_id: child_root.id.clone(),
                awaiting_prompt: None,
            },
            subagent.spawn.spawn_index,
            now,
            AgentTask::SUBAGENT_INVOCATION_MAX_ATTEMPTS,
        )
        .context("spawn rejected: new_subagent_invocation failed")?;
        ensure!(
            Self::load_task_tx(tx, &invocation.id).await?.is_none(),
            "spawn rejected: invocation id {} already exists",
            invocation.id,
        );

        let parent_task = old_parent
            .wait_on_children(
                1,
                subagent.spawn.payload.clone(),
                vec![invocation.id.clone()],
                now,
            )
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(tx, &parent_task).await?;
        Self::insert_task_tx(tx, &invocation).await?;
        Self::insert_task_tx(tx, &child_root).await?;
        Self::complete_injections_tx(tx, &subagent.delivered_injection_ids, now).await?;

        agent_server::fail_point!("atomic_spawn.after_rows");

        let committed_event = Self::insert_subagent_spawn_event_tx(
            tx,
            &subagent,
            &parent_task,
            (&invocation, &child_root),
            now,
        )
        .await?;
        Self::insert_child_created_tx(tx, &child_root.thread_id, now).await?;
        Ok(SpawnedChildren {
            parent: parent_task,
            subagents: vec![(invocation, child_root)],
            tool_children: Vec::new(),
            child_threads: vec![child_thread],
            events: vec![committed_event],
        })
    }

    async fn spawn_subagent_batch_tx(
        tx: &mut Transaction<'_, Sqlite>,
        old_parent: AgentTask,
        SubagentBatchSpawn {
            delivered_injection_ids,
            spawns,
            payload,
            events,
        }: SubagentBatchSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedChildren> {
        // Build (invocation, child_root) per entry, validating against
        // existing rows under the same transaction so the whole batch
        // is rejected atomically if any single entry would conflict.
        let prepared = prepare_subagent_batch_rows_tx(tx, &old_parent, spawns, now).await?;
        let child_ids: Vec<AgentTaskId> = prepared
            .iter()
            .map(|(invocation, _)| invocation.id.clone())
            .collect();
        let child_count =
            u32::try_from(prepared.len()).context("spawn rejected: child count exceeds u32")?;
        let new_parent = old_parent
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(tx, &new_parent).await?;
        for (invocation, child_root) in &prepared {
            Self::insert_task_tx(tx, invocation).await?;
            Self::insert_task_tx(tx, child_root).await?;
        }
        let committed = if events.is_empty() {
            Vec::new()
        } else {
            agent_server::fail_point!("atomic_spawn_batch.after_rows");
            Self::insert_spawn_started_events_tx(tx, &new_parent.thread_id, &prepared, events, now)
                .await?
        };
        Self::complete_injections_tx(tx, &delivered_injection_ids, now).await?;
        Ok(SpawnedChildren {
            parent: new_parent,
            subagents: prepared,
            tool_children: Vec::new(),
            child_threads: Vec::new(),
            events: committed,
        })
    }

    async fn spawn_mixed_children_tx(
        tx: &mut Transaction<'_, Sqlite>,
        old_parent: AgentTask,
        spawn: MixedChildrenSpawn,
        events: Vec<SubagentSpawnEvent>,
        now: OffsetDateTime,
    ) -> Result<SpawnedChildren> {
        let MixedChildrenSpawn {
            delivered_injection_ids,
            boundary_events,
            trailing_events,
            subagents,
            tool_children,
            reattach,
            payload,
            child_otel_traceparent,
        } = spawn;
        let has_precompleted = has_precompleted_tool_child(&tool_children);
        let parent_id = old_parent.id.clone();

        let prepared = prepare_subagent_batch_rows_tx(tx, &old_parent, subagents, now).await?;
        let (tool_rows, reattached_rows) = prepare_mixed_tool_and_reattach_rows_tx(
            tx,
            &old_parent,
            &parent_id,
            &tool_children,
            reattach,
            child_otel_traceparent.as_deref(),
            now,
        )
        .await?;
        let child_ids = mixed_child_ids_in_slot_order(&prepared, &tool_rows, &reattached_rows)?;
        let child_count =
            u32::try_from(child_ids.len()).context("spawn rejected: child count exceeds u32")?;
        let live_children = live_mixed_child_count(&prepared, &tool_rows, &reattached_rows)?;
        let waiting_parent = old_parent
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        let new_parent = publish_transactional_mixed_parent(
            waiting_parent,
            has_precompleted,
            live_children,
            now,
        )?;

        Self::update_task_tx(tx, &new_parent).await?;
        for (invocation, child_root) in &prepared {
            Self::insert_task_tx(tx, invocation).await?;
            Self::insert_task_tx(tx, child_root).await?;
        }
        for child in &tool_rows {
            Self::insert_task_tx(tx, child).await?;
        }
        for child in &reattached_rows {
            Self::update_task_tx(tx, child).await?;
        }

        agent_server::fail_point!("atomic_spawn_batch.after_rows");

        let mut committed =
            Self::commit_parent_events_tx(tx, &new_parent.thread_id, boundary_events, now).await?;
        committed.extend(
            Self::insert_spawn_started_events_tx(tx, &new_parent.thread_id, &prepared, events, now)
                .await?,
        );
        committed.extend(
            Self::commit_parent_events_tx(tx, &new_parent.thread_id, trailing_events, now).await?,
        );
        Self::complete_injections_tx(tx, &delivered_injection_ids, now).await?;
        Ok(SpawnedChildren {
            parent: new_parent,
            subagents: prepared,
            tool_children: tool_rows,
            child_threads: Vec::new(),
            events: committed,
        })
    }
}

#[async_trait]
impl AgentTaskStore for SqliteDurableStore {
    async fn commit_completed_turn(&self, params: CompletedTurnCommit) -> Result<CommitOutcome> {
        let params = params.extended(self.extension.as_deref())?;
        let outcome = self.commit_completed_turn_atomic_inner(params).await?;
        TerminalRootTurn::record_tasks(outcome.settled.iter().map(|(task, _)| task));
        Ok(outcome)
    }

    async fn insert(&self, task: AgentTask) -> Result<()> {
        let mut tx = self.begin().await?;
        Self::enforce_insert_cross_row_invariants_tx(&mut tx, &task).await?;
        Self::insert_task_tx(&mut tx, &task).await?;
        tx.commit().await.context("commit task insert")?;
        Ok(())
    }

    async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
        if task.kind != TaskKind::RootTurn {
            let kind = task.kind;
            return Err(anyhow!(
                "submit_root_turn rejected: expected root_turn, got {kind:?}"
            ));
        }
        if task.status != TaskStatus::Pending {
            let status = task.status;
            return Err(anyhow!(
                "submit_root_turn rejected: new root must start in Pending (got {status:?})"
            ));
        }
        if task.attempt != 0 {
            let attempt = task.attempt;
            return Err(anyhow!(
                "submit_root_turn rejected: new root must have attempt == 0 (got {attempt})"
            ));
        }
        if !task.is_root() {
            return Err(anyhow!("submit_root_turn rejected: task must be a root"));
        }
        task.validate()
            .context("submit_root_turn rejected: task failed schema validation")?;

        let mut tx = self.begin().await?;
        // Fence BEFORE bootstrap: a deleting/deleted tombstone must be
        // seen before the get-or-create could touch the thread row, so
        // the fence — not the bootstrap's ON CONFLICT — is what refuses
        // resurrection.
        if let Some(existing) = Self::get_thread_tx(&mut tx, &task.thread_id).await? {
            existing.require_active(ThreadOperation::Submit)?;
        }
        Self::bootstrap_thread_row_tx(&mut tx, &task.thread_id, task.created_at).await?;

        let task_id_str = task.id.as_str();
        let id_exists = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = ?1 LIMIT 1",
            task_id_str
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("check existing task {}", task.id))?
        .flatten();
        if id_exists.is_some() {
            return Err(anyhow!(
                "submit_root_turn rejected: task id {} already exists",
                task.id
            ));
        }

        let blocking_check_key = thread_key(&task.thread_id);
        let thread_has_blocking_root: bool = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question') LIMIT 1", blocking_check_key)
            .fetch_optional(&mut *tx)
            .await
            .with_context(|| format!("check active root slot for {}", task.thread_id))?
            .flatten()
            .is_some();

        let queued_check_key = thread_key(&task.thread_id);
        let thread_has_queued_roots: bool = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status = 'queued' LIMIT 1", queued_check_key)
            .fetch_optional(&mut *tx)
            .await
            .with_context(|| format!("check queued roots for {}", task.thread_id))?
            .flatten()
            .is_some();

        let admitted = if thread_has_blocking_root || thread_has_queued_roots {
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")?
        } else {
            task
        };

        Self::insert_task_tx(&mut tx, &admitted).await?;

        // Phase 10 · D: emit a durable `task_wakeup` advisory row in the
        // SAME transaction when the new root is immediately runnable, so
        // a worker (even in another process) is nudged to call
        // `acquire_next_runnable`. A queued root is parked behind an
        // active/queued root and is nudged when promoted instead.
        if admitted.status == TaskStatus::Pending {
            Self::insert_task_wakeup_outbox_row_tx(
                &mut tx,
                &admitted.id,
                &admitted.thread_id,
                TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS,
                admitted.created_at,
            )
            .await?;
        }

        tx.commit().await.context("commit submit_root_turn")?;
        Ok(admitted)
    }

    async fn submit_root_turn_idempotent(
        &self,
        params: SubmitRootTurnParams,
    ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
        let SubmitRootTurnParams {
            task,
            idempotency,
            max_queued_depth,
            disposition,
        } = params;
        agent_server::journal::store::validate_submit_root_shape(&task)?;

        // BEGIN IMMEDIATE serializes writers, so the idempotency claim,
        // queue-depth check, and admission insert below are atomic.
        let mut tx = self.begin().await.map_err(SubmitRootTurnError::Other)?;
        Self::bootstrap_thread_row_tx(&mut tx, &task.thread_id, task.created_at)
            .await
            .map_err(SubmitRootTurnError::Other)?;
        Self::require_active_thread_tx(&mut tx, &task.thread_id, ThreadOperation::Submit)
            .await
            .map_err(SubmitRootTurnError::from)?;

        // 1. Idempotency replay / conflict, inside the same transaction.
        // Always explicitly commit (replay) or roll back (conflict /
        // rejection) so the BEGIN IMMEDIATE write lock is released
        // promptly rather than waiting on `Drop` (a dropped-but-open tx
        // would hold the database-level write lock and stall writers).
        if let Some(claim) = &idempotency {
            match Self::try_replay_submit_tx(&mut tx, claim).await {
                Ok(Some(outcome)) => {
                    tx.commit()
                        .await
                        .context("commit idempotent submit replay")
                        .map_err(SubmitRootTurnError::Other)?;
                    return Ok(outcome);
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = tx.rollback().await;
                    return Err(error);
                }
            }
        }

        let task_id_str = task.id.as_str();
        let id_exists = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = ?1 LIMIT 1",
            task_id_str
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("check existing task {}", task.id))
        .map_err(SubmitRootTurnError::Other)?
        .flatten();
        if id_exists.is_some() {
            let _ = tx.rollback().await;
            return Err(SubmitRootTurnError::Other(anyhow!(
                "submit_root_turn rejected: task id {} already exists",
                task.id
            )));
        }

        let blocking_check_key = thread_key(&task.thread_id);
        // Mirrors the Postgres split: a `NextTurn` submit only needs to know
        // *whether* an active root exists, so it keeps main's cheap id probe;
        // only an injection request, which inspects the root's status to decide
        // parentage, reads the full ordered row.
        let (thread_has_blocking_root, active_root) = match disposition {
            SubmitDisposition::NextTurn => (
                sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation', 'awaiting_question') LIMIT 1", blocking_check_key)
                    .fetch_optional(&mut *tx)
                    .await
                    .with_context(|| format!("check active root slot for {}", task.thread_id))
                    .map_err(SubmitRootTurnError::Other)?
                    .flatten()
                    .is_some(),
                None,
            ),
            SubmitDisposition::InjectAtBoundary => {
                let root = active_root_for_thread_exec(&mut *tx, &task.thread_id)
                    .await
                    .map_err(SubmitRootTurnError::Other)?;
                (root.is_some(), root)
            }
        };
        let injection_parent = agent_server::journal::store::resolve_injection_parent(
            disposition,
            active_root.as_ref(),
        )
        .cloned();

        let current_queued = Self::queued_root_count_tx(&mut tx, &task.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        // Back-pressure applies only when the request remains a future root.
        let would_queue =
            injection_parent.is_none() && (thread_has_blocking_root || current_queued > 0);
        if let Some(cap) = max_queued_depth
            && would_queue
            && current_queued >= cap
        {
            let _ = tx.rollback().await;
            return Err(SubmitRootTurnError::QueueDepthExceeded {
                cap,
                current_depth: current_queued,
            });
        }

        // 3. Admit, claim the idempotency key, and commit — all atomic.
        Self::commit_fresh_admission_tx(tx, task, injection_parent, would_queue, idempotency).await
    }

    async fn claim_idempotency(
        &self,
        request_id: &str,
        kind: IdempotencyKind,
        fingerprint: &[u8],
    ) -> Result<IdempotencyClaim> {
        // Atomic reservation: insert a placeholder (`result_json` =
        // serialized JSON null) if absent. A 1-row insert means we won
        // the claim; 0 rows means a concurrent retry got there first, so
        // it cannot also observe `Fresh`.
        let now = OffsetDateTime::now_utc();
        let kind_wire = kind.as_str();
        let inserted = retry_busy(&self.lock_conflict_retries, "claim an idempotency key", || {
            sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES (?1, ?2, ?3, 'null', ?4)
               ON CONFLICT (request_id) DO NOTHING"#,
            request_id,
            kind_wire,
            fingerprint,
            now,
        )
                .execute(&self.pool)
        })
        .await
        .with_context(|| format!("idempotency claim for {request_id}"))?;

        if inserted.rows_affected() == 1 {
            return Ok(IdempotencyClaim::Fresh);
        }

        // Lost the race (or a prior row exists): inspect the existing row.
        let row = sqlx::query!(
            r#"SELECT kind, fingerprint, result_json FROM agent_sdk_idempotency WHERE request_id = ?1"#,
            request_id,
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("idempotency lookup for {request_id}"))?
        .ok_or_else(|| anyhow!("idempotency row for {request_id} vanished after claim conflict"))?;

        if row.kind != kind.as_str() || row.fingerprint != fingerprint {
            return Ok(IdempotencyClaim::Conflict);
        }
        let result_json: serde_json::Value = serde_json::from_str(&row.result_json)
            .context("decode stored idempotency result_json")?;
        if result_json.is_null() {
            return Ok(IdempotencyClaim::InFlight);
        }
        let stored_kind = IdempotencyKind::from_wire(&row.kind)
            .ok_or_else(|| anyhow!("unknown idempotency kind {} stored", row.kind))?;
        Ok(IdempotencyClaim::Replay(Box::new(IdempotencyRecord {
            request_id: request_id.to_owned(),
            kind: stored_kind,
            fingerprint: row.fingerprint,
            result_json,
        })))
    }

    async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
        let kind_wire = record.kind.as_str();
        let result_text =
            serde_json::to_string(&record.result_json).context("encode idempotency result_json")?;
        let created_at = OffsetDateTime::now_utc();
        // Fill the reservation placeholder written by `claim_idempotency`.
        // The `WHERE result_json = 'null'` guard fills only the
        // placeholder and never overwrites a real recorded result, so a
        // racing retry cannot clobber it. With no prior claim row the
        // INSERT lands the full record (defensive path).
        retry_busy(&self.lock_conflict_retries, "record an idempotency result", || {
            sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES (?1, ?2, ?3, ?4, ?5)
               ON CONFLICT (request_id) DO UPDATE
                   SET result_json = excluded.result_json
                   WHERE result_json = 'null'"#,
            record.request_id,
            kind_wire,
            record.fingerprint,
            result_text,
            created_at,
        )
                .execute(&self.pool)
        })
        .await
        .with_context(|| format!("record idempotency result for {}", record.request_id))?;
        Ok(())
    }

    async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
        self.get_task_pool(id).await
    }

    async fn update(&self, task: AgentTask) -> Result<()> {
        let mut tx = self.begin().await?;
        let _old = Self::validate_update_row_invariants_tx(&mut tx, &task).await?;
        Self::update_task_tx(&mut tx, &task).await?;
        tx.commit().await.context("commit task update")?;
        Ok(())
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE thread_id = ?1 ORDER BY created_at, id",
        ))
        .bind(thread_key(thread_id))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tasks for thread {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE parent_id = ?1 ORDER BY created_at, id",
        ))
        .bind(parent_id.as_str())
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list children for {parent_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn complete_input_injection(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut tx = self.begin().await?;
        let Some(old) = Self::load_task_tx(&mut tx, id).await? else {
            return Ok(None);
        };
        if old.kind != TaskKind::InputInjection || old.status != TaskStatus::Queued {
            return Ok(None);
        }
        let completed = old
            .complete_input_injection(now)
            .context("complete input injection transition failed")?;
        Self::update_task_tx(&mut tx, &completed).await?;
        tx.commit()
            .await
            .context("commit input injection delivery")?;
        Ok(Some(completed))
    }

    /// ADR-0003 I5 — bounded. Two indexed reads on `parent_id`, neither of
    /// which can grow with retained terminal history:
    ///
    /// 1. an `EXISTS` over ALL children for a fresh sign of life, and
    /// 2. at most `live_limit` NON-terminal rows for the walk to expand.
    ///
    /// The freshness test is `julianday(...)`, not a text comparison: this
    /// column is variable-precision RFC3339 TEXT, whose lexical order is NOT
    /// chronological (see `heartbeat_task`). It is also `COALESCE`d to
    /// `created_at` — a never-acquired child carries no activity stamp, and its
    /// creation is its initial evidence of life (I1).
    async fn probe_children(
        &self,
        parent_id: &AgentTaskId,
        cutoff: OffsetDateTime,
        live_limit: usize,
    ) -> Result<ChildProbe> {
        let fresh_activity: bool = sqlx::query_scalar::<_, i64>(
            "SELECT EXISTS( \
                 SELECT 1 FROM agent_sdk_tasks \
                 WHERE parent_id = ?1 \
                   AND julianday(COALESCE(last_activity_at, created_at)) >= julianday(?2))",
        )
        .bind(parent_id.as_str())
        .bind(cutoff)
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("probe child activity for {parent_id}"))?
            != 0;

        let limit = i64::try_from(live_limit).unwrap_or(i64::MAX);
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE parent_id = ?1 AND status NOT IN ('completed', 'failed', 'cancelled') \
             ORDER BY created_at, id LIMIT ?2",
        ))
        .bind(parent_id.as_str())
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("probe live children for {parent_id}"))?;
        let live: Vec<AgentTask> = records
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_>>()?;

        Ok(ChildProbe {
            live,
            fresh_activity,
        })
    }

    async fn count_by_status(&self, status: TaskStatus) -> Result<u64> {
        let status_wire = enum_to_wire(&status)?;
        let count = sqlx::query_scalar!(
            r#"SELECT COUNT(*) AS "count!: i64" FROM agent_sdk_tasks WHERE status = ?1"#,
            status_wire,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("count tasks in status {status:?}"))?;
        Ok(u64::try_from(count).unwrap_or_default())
    }

    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let status_wire = enum_to_wire(&status)?;
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks WHERE status = ?1 ORDER BY created_at, id",
        ))
        .bind(status_wire)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tasks in status {status:?}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
        active_root_for_thread_exec(&self.pool, thread_id).await
    }

    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE thread_id = ?1 AND kind = 'root_turn' AND status = 'queued' \
             ORDER BY created_at, id",
        ))
        .bind(thread_key(thread_id))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list queued roots for {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn promote_next_queued_root(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        Self::require_active_thread_tx(&mut tx, thread_id, ThreadOperation::Promote).await?;
        let promoted = Self::promote_next_queued_root_tx(&mut tx, thread_id, now).await?;
        tx.commit()
            .await
            .context("commit promote_next_queued_root")?;
        Ok(promoted)
    }

    async fn try_acquire_task(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let mut tx = self.begin().await?;
        let Some(old) = Self::load_task_tx(&mut tx, id).await? else {
            return Ok(None);
        };
        Self::require_active_thread_tx(&mut tx, &old.thread_id, ThreadOperation::Acquire).await?;
        if !old.status.can_be_leased() {
            return Ok(None);
        }
        let unfinished = Self::load_unfinished_siblings_tx(&mut tx, &old).await?;
        if old.waits_on_any(&unfinished) {
            return Ok(None);
        }
        match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
            RecoveryAction::NoAction => {}
            RecoveryAction::FailClosed(reason) => {
                let (_settled, root_turns) =
                    Self::fail_closed_tx(&mut tx, &old, reason, now, "try_acquire_task").await?;
                tx.commit().await.context("commit fail-closed acquire")?;
                TerminalRootTurn::record_each(root_turns);
                return Ok(None);
            }
            RecoveryAction::Requeue => {
                return Err(anyhow!(
                    "try_acquire_task: recovery matrix produced Requeue for acquisition-time row {id}"
                ));
            }
        }
        let claimed = old
            .clone()
            .admit_after_siblings(&unfinished)
            .and_then(|admitted| admitted.mark_running(worker, lease, expires_at, now))
            .context("try_acquire_task rejected: mark_running transition failed")?;
        Self::update_task_tx(&mut tx, &claimed).await?;
        Self::stamp_lease_epoch_tx(&mut tx, &claimed.id, self.current_epoch().as_ref()).await?;
        tx.commit().await.context("commit try_acquire_task")?;
        Ok(Some(claimed))
    }

    async fn acquire_next_runnable(
        &self,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        if self.held_probe.probes_first()
            && let Some(task) = self
                .acquire_ready_held(worker.clone(), lease.clone(), expires_at, now)
                .await?
        {
            return Ok(Some(task));
        }
        if let Some(task) = self
            .acquire_ready_head(worker.clone(), lease.clone(), expires_at, now)
            .await?
        {
            return Ok(Some(task));
        }
        self.acquire_ready_held(worker, lease, expires_at, now)
            .await
    }

    async fn ready_held_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTaskId>> {
        let parent_key = parent_id.as_str();
        Ok(sqlx::query_scalar!(
            r#"SELECT held.id AS "id!" FROM agent_sdk_tasks AS held INDEXED BY agent_sdk_tasks_held_by_parent_idx WHERE held.parent_id = ?1 AND held.status >= 'pending' AND held.status <= 'pending' AND held.after_siblings_json IS NOT NULL AND NOT EXISTS (SELECT 1 FROM json_each(held.after_siblings_json) AS waited JOIN agent_sdk_tasks AS sibling ON sibling.id = waited.value WHERE sibling.status NOT IN ('completed', 'failed', 'cancelled')) ORDER BY held.created_at, held.id"#,
            parent_key,
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("load the ready held children of {parent_id}"))?
        .into_iter()
        .map(AgentTaskId::from_string)
        .collect())
    }

    async fn heartbeat_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        expires_at: OffsetDateTime,
        activity: Option<OffsetDateTime>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut tx = self.begin().await?;
        if Self::lease_superseded_tx(&mut tx, id).await? {
            anyhow::bail!(
                "heartbeat rejected: {}",
                agent_server::journal::EpochSuperseded::of_lease(id)
            );
        }
        // Single guarded UPDATE of only the lease columns — never reload
        // and rewrite the whole 25-column row (re-serialising the
        // submitted_input / state JSON blobs) on a per-tick heartbeat.
        // `RETURNING` hands back the refreshed row in one round trip.
        //
        // ADR-0003 I2 — the activity column advances but never retreats, and
        // the comparison that decides "advance" must be CHRONOLOGICAL.
        //
        // It cannot be the text comparison it looks like it could be. sqlx
        // binds `OffsetDateTime` as RFC3339 TEXT (`format(&Rfc3339)`) and
        // `time` trims trailing zeros, so the stored fractional precision is
        // VARIABLE — and lexicographic order is then NOT chronological:
        //
        //     MAX('..T10:00:00Z',   '..T10:00:00.5Z')  = '..:00Z'    (older!)
        //     MAX('..T10:00:00.1Z', '..T10:00:00.12Z') = '..:00.1Z'  (older!)
        //
        // 'Z' (0x5A) sorts after '.' (0x2E), so a whole-second stamp beats a
        // sub-second one inside the same second; and any fraction that is a
        // lexical PREFIX of a later one wins over it. A scalar `MAX()` here
        // therefore REWINDS `last_activity_at` — and a rewind is
        // indistinguishable from silence, which reaps a healthy child.
        //
        // `julianday()` parses both operands to a numeric instant, restoring
        // chronological order. Its ~50µs resolution is six orders of
        // magnitude finer than the enforcement floor (>= 2 heartbeats), and a
        // tie keeps the stored value, so it can never rewind. The NULL arms
        // come first: a beat carrying no work (?6 IS NULL) leaves the column
        // exactly as it was, and the first beat to carry work adopts it.
        let worker_str = worker.as_str();
        let lease_str = lease.as_str();
        let updated = sqlx::query_as::<_, TaskRecord>(concat!(
            "UPDATE agent_sdk_tasks \
             SET lease_expires_at = ?4, last_heartbeat_at = ?5, updated_at = ?5, \
                 last_activity_at = CASE \
                     WHEN ?6 IS NULL THEN last_activity_at \
                     WHEN last_activity_at IS NULL THEN ?6 \
                     WHEN julianday(?6) > julianday(last_activity_at) THEN ?6 \
                     ELSE last_activity_at \
                 END \
             WHERE id = ?1 AND status = 'running' AND worker_id = ?2 AND lease_id = ?3 \
             RETURNING ",
            task_columns!(),
        ))
        .bind(id.as_str())
        .bind(worker_str)
        .bind(lease_str)
        .bind(expires_at)
        .bind(now)
        .bind(activity)
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("heartbeat update for {id}"))?;

        if let Some(record) = updated {
            let refreshed = AgentTask::try_from(record)?;
            tx.commit().await.context("commit heartbeat_task")?;
            return Ok(refreshed);
        }

        // CAS rejection: one diagnostic SELECT to produce a precise error.
        let existing = Self::load_task_tx(&mut tx, id).await?;
        tx.rollback().await.context("rollback rejected heartbeat")?;
        Err(match existing {
            None => anyhow!("heartbeat rejected: task {id} does not exist"),
            Some(task) if task.status != TaskStatus::Running => {
                let status = task.status;
                anyhow!("heartbeat rejected: task {id} is not running (status {status:?})")
            }
            Some(task) if task.worker_id.as_ref() != Some(worker) => {
                anyhow!("heartbeat rejected: worker mismatch on task {id}")
            }
            Some(_) => anyhow!("heartbeat rejected: lease mismatch on task {id}"),
        })
    }

    async fn release_expired_leases(
        &self,
        now: OffsetDateTime,
        scope: LeaseRecoveryScope,
    ) -> Result<Vec<RecoveryRecord>> {
        if let LeaseRecoveryScope::ForeignEpoch(epoch) = &scope {
            ensure!(
                self.current_epoch().as_ref() == Some(epoch),
                "release_expired_leases: epoch {epoch} is not the epoch this store began"
            );
        }
        let mut tx = self.begin().await?;
        // Bound the sweep to a batch so a mass worker outage cannot turn
        // one sweep into a giant transaction that locks every expired
        // task at once; the host's drain loop (and any other caller)
        // keeps calling until a pass returns fewer than the batch.
        let batch =
            i64::try_from(agent_server::journal::store::LEASE_RELEASE_BATCH).unwrap_or(i64::MAX);
        let expired = match &scope {
            LeaseRecoveryScope::Expired => sqlx::query_as::<_, TaskRecord>(concat!(
                "SELECT ",
                task_columns!(),
                " FROM agent_sdk_tasks \
                 WHERE status = 'running' AND lease_expires_at <= ?1 \
                 ORDER BY lease_expires_at, id \
                 LIMIT ?2",
            ))
            .bind(now)
            .bind(batch)
            .fetch_all(&mut *tx)
            .await
            .context("load expired leases")?,
            LeaseRecoveryScope::ForeignEpoch(epoch) => sqlx::query_as::<_, TaskRecord>(concat!(
                "SELECT ",
                task_columns!(),
                " FROM agent_sdk_tasks \
                 WHERE status = 'running' AND lease_epoch IS NOT ?1 \
                 ORDER BY lease_expires_at, id \
                 LIMIT ?2",
            ))
            .bind(epoch.as_str())
            .bind(batch)
            .fetch_all(&mut *tx)
            .await
            .context("load leases of older epochs")?,
            LeaseRecoveryScope::SupersededEpochs => sqlx::query_as::<_, TaskRecord>(concat!(
                "SELECT ",
                task_columns!(),
                " FROM agent_sdk_tasks \
                 WHERE status = 'running' AND lease_epoch IN \
                   (SELECT epoch_id FROM agent_sdk_host_epochs WHERE superseded_at IS NOT NULL) \
                 ORDER BY lease_expires_at, id \
                 LIMIT ?1",
            ))
            .bind(batch)
            .fetch_all(&mut *tx)
            .await
            .context("load leases of superseded epochs")?,
        };

        let mut released = Vec::with_capacity(expired.len());
        let mut root_turns = Vec::new();
        for record in expired {
            let Some(old) = Self::load_task_tx(&mut tx, &AgentTask::try_from(record)?.id).await?
            else {
                continue;
            };
            if old.status.is_terminal() {
                continue;
            }
            if old.status != TaskStatus::Running {
                let status = old.status;
                return Err(anyhow!(
                    "release_expired_leases: expiry index held non-running row {} in status {status:?}",
                    old.id
                ));
            }
            let record = match classify_recovery(&old, scope.context()) {
                RecoveryAction::Requeue => {
                    let released_row = old
                        .clone()
                        .release_lease(now)
                        .context("release_expired_leases: release transition failed")?;
                    Self::update_task_tx(&mut tx, &released_row).await?;
                    RecoveryRecord::requeued(old.id.clone())
                }
                RecoveryAction::FailClosed(reason) => {
                    let (settled, ended) =
                        Self::fail_closed_tx(&mut tx, &old, reason, now, "release_expired_leases")
                            .await?;
                    root_turns.extend(ended);
                    RecoveryRecord::failed_closed(old.id.clone(), reason).with_settled(settled)
                }
                RecoveryAction::NoAction => {
                    return Err(anyhow!(
                        "release_expired_leases: recovery matrix produced NoAction for expired row {}",
                        old.id
                    ));
                }
            };
            released.push(record);
        }

        tx.commit().await.context("commit release_expired_leases")?;
        TerminalRootTurn::record_each(root_turns);
        Ok(released)
    }

    async fn requeue_owned_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        boundary: Option<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<RequeueOutcome> {
        let mut tx = self.begin().await?;
        let Some(old) = Self::load_task_tx(&mut tx, id).await? else {
            return Ok(RequeueOutcome::NotOwned);
        };
        let still_owned = old.status == TaskStatus::Running
            && old.worker_id.as_ref() == Some(worker)
            && old.lease_id.as_ref() == Some(lease);
        if !still_owned {
            return Ok(RequeueOutcome::NotOwned);
        }
        if old.is_budget_exhausted() {
            return Ok(RequeueOutcome::BudgetExhausted);
        }
        // Boundary event + advisory in the SAME transaction as the
        // ownership CAS and the release (the cancel_tree marker
        // pattern): it lands iff this caller still owned the row, and
        // it is durable before the row is acquirable.
        if let Some(event) = boundary {
            Self::require_appendable_thread_tx(
                &mut tx,
                &old.thread_id,
                agent_server::journal::thread::FencedWriter::RequeueOwnedTask,
            )
            .await?;
            let start_seq = Self::next_event_sequence_tx(&mut tx, &old.thread_id).await?;
            let committed =
                Self::insert_events_tx(&mut tx, &old.thread_id, vec![event], start_seq, now)
                    .await?;
            Self::insert_thread_events_outbox_row_tx(
                &mut tx,
                &committed,
                DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
                now,
            )
            .await?;
        }
        let released_row = old
            .release_lease(now)
            .context("requeue_owned_task: release transition failed")?;
        Self::update_task_tx(&mut tx, &released_row).await?;
        tx.commit().await.context("commit requeue_owned_task")?;
        Ok(RequeueOutcome::Requeued(Box::new(released_row)))
    }

    async fn pause_on_children(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        child_count: u32,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("pause rejected: task {id} does not exist"))?;
        if old.status != TaskStatus::Running {
            let status = old.status;
            return Err(anyhow!(
                "pause rejected: task {id} is not running (status {status:?})"
            ));
        }
        match &old.worker_id {
            Some(current) if current == worker => {}
            _ => return Err(anyhow!("pause rejected: worker mismatch on task {id}")),
        }
        match &old.lease_id {
            Some(current) if current == lease => {}
            _ => return Err(anyhow!("pause rejected: lease mismatch on task {id}")),
        }
        let paused = old
            .clone()
            .wait_on_children(child_count, payload, Vec::new(), now)
            .context("pause rejected: wait_on_children transition failed")?;
        Self::update_task_tx(&mut tx, &paused).await?;
        tx.commit().await.context("commit pause_on_children")?;
        Ok(paused)
    }

    async fn enqueue_steering_resume(
        &self,
        parent_id: &AgentTaskId,
        steering: Vec<agent_sdk_foundation::llm::ContentBlock>,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition::enqueue_steering_resume(parent_id.clone(), steering),
            now,
        )
        .await
        .map_err(|rejected| {
            agent_server::journal::transition::wrapper_error("steering wake", rejected)
        })?;
        Ok(committed.steering_resumed.into_iter().next().flatten())
    }

    async fn repark_after_steering(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        payload: SuspensionPayload,
        reattach: Vec<AgentTaskId>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition {
                guards: Vec::new(),
                writes: vec![agent_server::journal::Write::ReparkAfterSteering {
                    task: parent_id.clone(),
                    worker: worker.clone(),
                    lease: lease.clone(),
                    payload: Box::new(payload),
                    reattach,
                }],
            },
            now,
        )
        .await
        .map_err(|rejected| agent_server::journal::transition::wrapper_error("repark", rejected))?;
        committed
            .reparked
            .into_iter()
            .next()
            .context("a repark transition returns its parent")
    }

    async fn pause_on_confirmation(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        continuation: ContinuationEnvelope,
        prepared_operation: Option<ListenExecutionContext>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition::pause_only(
                id.clone(),
                worker.clone(),
                lease.clone(),
                continuation,
                prepared_operation,
            ),
            now,
        )
        .await
        .map_err(|rejected| agent_server::journal::transition::wrapper_error("pause", rejected))?;
        committed
            .paused
            .into_iter()
            .next()
            .context("a pause transition returns its parked task")
    }

    async fn pause_on_question(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        pause: QuestionPause,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<CommittedEvent>)> {
        let QuestionPause {
            delivered_injection_ids,
            payload,
            questions,
            mut events,
        } = pause;
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("question pause rejected: task {id} does not exist"))?;
        let paused = apply_question_pause(&old, worker, lease, payload, questions.clone(), now)?;
        Self::require_appendable_thread_tx(
            &mut tx,
            &paused.thread_id,
            agent_server::journal::thread::FencedWriter::PauseOnQuestion,
        )
        .await?;
        Self::update_task_tx(&mut tx, &paused).await?;
        events.push(AgentEvent::question_asked(id.as_str(), questions));
        let start_seq = Self::next_event_sequence_tx(&mut tx, &paused.thread_id).await?;
        let committed =
            Self::insert_events_tx(&mut tx, &paused.thread_id, events, start_seq, now).await?;
        Self::insert_thread_events_outbox_row_tx(
            &mut tx,
            &committed,
            DEFAULT_TURN_OUTBOX_MAX_ATTEMPTS,
            now,
        )
        .await?;
        Self::complete_injections_tx(&mut tx, &delivered_injection_ids, now).await?;
        tx.commit().await.context("commit pause_on_question")?;
        Ok((paused, committed))
    }

    async fn answer_question(
        &self,
        id: &AgentTaskId,
        receipt_id: &str,
        answers: Vec<QuestionAnswer>,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("answer_question rejected: task {id} does not exist"))?;
        let answered = match apply_question_answer(&old, receipt_id, &answers, now)? {
            QuestionAnswerApplied::Replay(task) => {
                tx.commit().await.context("commit answer_question replay")?;
                return Ok(task);
            }
            QuestionAnswerApplied::Answered(task) => task,
        };
        Self::update_task_tx(&mut tx, &answered).await?;
        tx.commit().await.context("commit answer_question")?;
        Ok(answered)
    }

    async fn find_subagent_invocation_for_child_root(
        &self,
        child_root_id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        // Read-only counterpart of `resume_linked_subagent_invocation_tx`:
        // same linkage predicate, no transition, pool-level read.
        let record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE kind = 'subagent' \
               AND status = 'waiting_on_children' \
               AND json_extract(state_json, '$.invocation.child_root_task_id') = ?1",
        ))
        .bind(child_root_id.as_str())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| {
            format!("find_subagent_invocation_for_child_root: lookup for {child_root_id}")
        })?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE kind = 'subagent' AND status = 'waiting_on_children' \
             ORDER BY created_at, id",
        ))
        .fetch_all(&self.pool)
        .await
        .context("list parked subagent invocations")?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_terminal_roots_with_live_descendants(&self) -> Result<Vec<AgentTaskId>> {
        let roots = sqlx::query_file_scalar!("src/sqlite/terminal_roots_with_live_descendants.sql")
            .fetch_all(&self.pool)
            .await
            .context("list terminal roots with live descendants")?;
        Ok(roots.into_iter().map(AgentTaskId::from_string).collect())
    }

    async fn complete_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut tx = self.begin().await?;
        let result = Self::apply_task_terminal_transition_tx(
            &mut tx,
            child_id,
            worker,
            lease,
            now,
            "complete_task",
            |child| child.complete(now),
        )
        .await?;
        tx.commit().await.context("commit complete_task")?;
        TerminalRootTurn::record_tasks([&result.0]);
        Ok(result)
    }

    async fn complete_task_with_result(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        result_payload: serde_json::Value,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition::settle_only(
                child_id.clone(),
                worker.clone(),
                lease.clone(),
                agent_server::journal::Settle::Complete {
                    result: result_payload,
                },
            ),
            now,
        )
        .await
        .map_err(|rejected| {
            agent_server::journal::transition::wrapper_error("complete_task_with_result", rejected)
        })?;
        committed
            .settled
            .into_iter()
            .next()
            .context("a settle transition returns its settled task")
    }

    async fn fail_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition::settle_only(
                child_id.clone(),
                worker.clone(),
                lease.clone(),
                agent_server::journal::Settle::Fail { error },
            ),
            now,
        )
        .await
        .map_err(|rejected| {
            agent_server::journal::transition::wrapper_error("fail_task", rejected)
        })?;
        committed
            .settled
            .into_iter()
            .next()
            .context("a settle transition returns its settled task")
    }

    async fn fail_task_with_reason(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        error: String,
        reason: TerminalReason,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let committed = agent_server::journal::DurableStore::commit(
            self,
            agent_server::journal::Transition::settle_only(
                child_id.clone(),
                worker.clone(),
                lease.clone(),
                agent_server::journal::Settle::FailWithReason {
                    error,
                    reason,
                    result: None,
                },
            ),
            now,
        )
        .await
        .map_err(|rejected| {
            agent_server::journal::transition::wrapper_error("fail_task_with_reason", rejected)
        })?;
        committed
            .settled
            .into_iter()
            .next()
            .context("a settle transition returns its settled task")
    }

    async fn cancel_tree(
        &self,
        root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome> {
        self.cancel_scoped(root_id, CancelScope::Tree, None, now)
            .await
    }

    async fn cancel_descendants_of_owned(
        &self,
        parent: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome> {
        self.cancel_scoped(parent, CancelScope::Descendants, Some((worker, lease)), now)
            .await
    }

    async fn resume_from_confirmation(
        &self,
        id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("resume rejected: task {id} does not exist"))?;
        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "resume rejected: task {id} is not awaiting confirmation (status {status:?})"
            ));
        }
        let (resumed, prepared_operation) = old
            .clone()
            .approve_confirmation_for_pickup(now)
            .context("resume rejected: approve_confirmation_for_pickup transition failed")?;
        Self::update_task_tx(&mut tx, &resumed).await?;
        tx.commit()
            .await
            .context("commit resume_from_confirmation")?;
        Ok((resumed, prepared_operation))
    }

    async fn approve_confirmation_and_acquire(
        &self,
        id: &AgentTaskId,
        worker: WorkerId,
        lease: LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("approve rejected: task {id} does not exist"))?;
        if old.status != TaskStatus::AwaitingConfirmation {
            let status = old.status;
            return Err(anyhow!(
                "approve rejected: task {id} is not awaiting confirmation (status {status:?})"
            ));
        }
        let (claimed, prepared_operation) = old
            .clone()
            .approve_confirmation(worker, lease, expires_at, now)
            .context("approve rejected: approve_confirmation transition failed")?;
        Self::update_task_tx(&mut tx, &claimed).await?;
        // The one lease grant outside the worker pool: stamp it like
        // `try_acquire_task`, or the next `ForeignEpoch` reclaim would take
        // it back from its live drive.
        Self::stamp_lease_epoch_tx(&mut tx, &claimed.id, self.current_epoch().as_ref()).await?;
        tx.commit()
            .await
            .context("commit approve_confirmation_and_acquire")?;
        Ok((claimed, prepared_operation))
    }

    async fn reject_confirmation(
        &self,
        id: &AgentTaskId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut tx = self.begin().await?;
        let result = Self::reject_confirmation_tx(&mut tx, id, error, now).await?;
        tx.commit().await.context("commit reject_confirmation")?;
        Ok(result)
    }

    async fn clear(&self) -> Result<()> {
        // SQLite does not support TRUNCATE CASCADE, and
        // `agent_sdk_tasks` carries a self-referential FK with
        // ON DELETE RESTRICT — RESTRICT is enforced per-row even when
        // `defer_foreign_keys` is on, so bulk DELETEs fail even when
        // the set is internally consistent.  Disable FK enforcement
        // for the duration of the wipe.  Connection PRAGMAs cannot
        // run inside a transaction and are scoped to the connection,
        // so we acquire one and run every statement on it.
        //
        // Critical: `PRAGMA foreign_keys = OFF` is connection-level
        // session state.  If we returned the connection to the pool
        // with FKs still off, the next acquirer would silently
        // bypass referential integrity.  We therefore re-enable FKs
        // unconditionally — even when the DELETE sequence errors —
        // by collecting the deletion result and running the restore
        // PRAGMA before propagating any error.
        let mut conn = self
            .pool
            .acquire()
            .await
            .context("acquire sqlite connection for clear")?;
        sqlx::query("PRAGMA foreign_keys = OFF")
            .execute(&mut *conn)
            .await
            .context("disable sqlite foreign_keys for clear")?;

        let delete_result = Self::clear_tables(&mut conn).await;

        let restore_result = sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&mut *conn)
            .await
            .context("re-enable sqlite foreign_keys after clear");

        delete_result?;
        restore_result?;
        Ok(())
    }
}

impl SqliteDurableStore {
    async fn clear_tables(conn: &mut sqlx::SqliteConnection) -> Result<()> {
        sqlx::query!("DELETE FROM agent_sdk_turn_checkpoints")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_message_commits")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_turn_attempts")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_message_heads")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_committed_events")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_outbox")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_retention_cursors")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_execution_intents")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_tool_audit_events")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_documents")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_tasks")
            .execute(&mut *conn)
            .await?;
        sqlx::query!("DELETE FROM agent_sdk_threads")
            .execute(&mut *conn)
            .await?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// ThreadStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl ThreadStore for SqliteDurableStore {
    async fn commit_thread_creation(
        &self,
        params: ThreadCreationCommit,
    ) -> Result<ThreadCreationCommitted> {
        self.commit_thread_creation_atomic_inner(params).await
    }

    async fn commit_fork(&self, params: ForkCommitParams) -> Result<ThreadCreationOutcome> {
        self.commit_fork_atomic_inner(params).await
    }

    async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let thread = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .context("thread missing after bootstrap")?;
        thread.require_active(ThreadOperation::Create)?;
        tx.commit().await.context("commit get_or_create thread")?;
        Ok(thread)
    }

    async fn get_or_create_for_creation(
        &self,
        thread_id: &ThreadId,
        creation: &ThreadCreation,
        now: OffsetDateTime,
    ) -> Result<ThreadCreationOutcome> {
        let mut tx = self.begin().await?;
        let outcome = Self::claim_thread_creation_tx(&mut tx, thread_id, creation, now).await?;
        let _projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        tx.commit()
            .await
            .context("commit caller-addressed thread get-or-create")?;
        Ok(outcome)
    }

    async fn get(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
        self.get_thread_pool(thread_id).await
    }

    async fn commit_turn(
        &self,
        thread_id: &ThreadId,
        expected_turn: u32,
        turn_usage: &TokenUsage,
        now: OffsetDateTime,
    ) -> Result<Thread> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let old = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .context("thread missing after bootstrap")?;
        old.require_active(ThreadOperation::Commit)?;
        if old.committed_turns.saturating_add(1) != expected_turn {
            return Err(anyhow::Error::new(StaleTurnCommit {
                expected_turn,
                committed_turns: old.committed_turns,
            }));
        }
        let thread = old.apply_committed_turn(turn_usage, now)?;
        Self::upsert_thread_tx(&mut tx, &thread).await?;
        tx.commit().await.context("commit thread turn")?;
        Ok(thread)
    }

    async fn mark_completed(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut tx = self.begin().await?;
        let old = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread {thread_id} does not exist"))?;
        let completed = old.mark_completed(now)?;
        Self::upsert_thread_tx(&mut tx, &completed).await?;
        tx.commit().await.context("commit mark_completed thread")?;
        Ok(completed)
    }

    async fn begin_purge(&self, thread_id: &ThreadId, seed: PurgeSeed) -> Result<BeginPurge> {
        let mut tx = self.begin().await?;
        let old = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .ok_or_else(|| {
                anyhow::Error::new(ThreadNotFound {
                    thread_id: thread_id.clone(),
                })
            })?;
        if old.status == ThreadStatus::Deleted {
            return old
                .purge_receipt
                .map(BeginPurge::AlreadyDeleted)
                .ok_or_else(|| anyhow!("deleted thread {thread_id} is missing its purge receipt"));
        }
        let deleting = old.begin_purge(seed);
        let effective = deleting
            .purge_seed
            .clone()
            .ok_or_else(|| anyhow!("deleting thread {thread_id} is missing its purge seed"))?;
        Self::upsert_thread_tx(&mut tx, &deleting).await?;
        tx.commit().await.context("commit begin thread purge")?;
        Ok(BeginPurge::Fenced(effective))
    }

    async fn finish_purge(
        &self,
        thread_id: &ThreadId,
        receipt: &PurgeReceipt,
    ) -> Result<PurgeReceipt> {
        let mut tx = self.begin().await?;
        let old = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .ok_or_else(|| {
                anyhow::Error::new(ThreadNotFound {
                    thread_id: thread_id.clone(),
                })
            })?;
        if old.status == ThreadStatus::Deleted {
            return old
                .purge_receipt
                .ok_or_else(|| anyhow!("deleted thread {thread_id} is missing its purge receipt"));
        }
        let deleted = old.finish_purge(receipt.clone());
        Self::upsert_thread_tx(&mut tx, &deleted).await?;
        Self::retire_thread_documents_tx(&mut tx, thread_key(thread_id), deleted.updated_at)
            .await?;
        tx.commit().await.context("commit finish thread purge")?;
        Ok(receipt.clone())
    }

    async fn list(&self) -> Result<Vec<Thread>> {
        let records = sqlx::query_as::<_, ThreadRecord>(
            "SELECT thread_id, status, committed_turns, total_input_tokens, total_output_tokens, created_at, updated_at, purge_receipt_json FROM agent_sdk_threads ORDER BY created_at, thread_id",
        )
        .fetch_all(&self.pool)
        .await
        .context("list threads")?;
        records.into_iter().map(TryInto::try_into).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// MessageProjectionStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl MessageProjectionStore for SqliteDurableStore {
    async fn get_or_create(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        tx.commit()
            .await
            .context("commit get_or_create message head")?;
        Ok(projection)
    }

    async fn get(&self, thread_id: &ThreadId) -> Result<Option<MessageProjection>> {
        self.get_message_head_pool(thread_id).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        Ok(self
            .get_message_head_pool(thread_id)
            .await?
            .map(|projection| projection.context_history())
            .unwrap_or_default())
    }

    async fn commit_messages(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_appendable_thread_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        let updated = projection.append_committed(messages, now)?;
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit message head append")?;
        Ok(updated)
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_appendable_thread_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        let updated = projection.replace_history(messages, now);
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit replace_history")?;
        Ok(updated)
    }

    async fn append_compaction(
        &self,
        thread_id: &ThreadId,
        result_messages: Vec<llm::Message>,
        source_message_count: usize,
        retained_message_count: usize,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        let updated = Self::append_compaction_tx(
            &mut tx,
            thread_id,
            result_messages,
            source_message_count,
            retained_message_count,
            now,
        )
        .await?;
        tx.commit().await.context("commit append_compaction")?;
        Ok(updated)
    }

    async fn append_repair(
        &self,
        thread_id: &ThreadId,
        repair_messages: Vec<llm::Message>,
        balanced_messages: Vec<llm::Message>,
        source_message_count: usize,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_appendable_thread_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        let updated = projection.append_repair(
            repair_messages,
            balanced_messages,
            source_message_count,
            now,
        )?;
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit append_repair")?;
        Ok(updated)
    }

    async fn set_draft(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_appendable_thread_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        let updated = projection.set_draft(messages, now);
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit set_draft")?;
        Ok(updated)
    }

    async fn clear_draft(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<MessageProjection>> {
        let mut tx = self.begin().await?;
        Self::require_appendable_thread_tx(
            &mut tx,
            thread_id,
            agent_server::journal::thread::FencedWriter::MessageHead,
        )
        .await?;
        let Some(projection) = read_message_head(&mut *tx, thread_id).await? else {
            // No projection row exists yet — nothing to clear. The
            // commit path treats this as a no-op so first-turn
            // happy-paths don't bootstrap rows just to clear them.
            return Ok(None);
        };
        let updated = projection.clear_draft(now);
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit clear_draft")?;
        Ok(Some(updated))
    }
}

// ─────────────────────────────────────────────────────────────────────
// TurnAttemptStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl TurnAttemptStore for SqliteDurableStore {
    async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt> {
        let attempt = TurnAttempt::open(params);
        let mut tx = self.begin().await?;
        Self::insert_attempt_tx(&mut tx, &attempt).await?;
        tx.commit().await.context("commit open_attempt")?;
        Ok(attempt)
    }

    async fn close_attempt(
        &self,
        id: &TurnAttemptId,
        params: CloseAttemptParams,
        now: OffsetDateTime,
    ) -> Result<TurnAttempt> {
        let mut tx = self.begin().await?;
        let old = Self::get_attempt_tx(&mut tx, id)
            .await?
            .ok_or_else(|| anyhow!("attempt not found: {id}"))?;
        let closed = old.close(params, now)?;
        Self::update_attempt_tx(&mut tx, &closed).await?;
        tx.commit().await.context("commit close_attempt")?;
        Ok(closed)
    }

    async fn get(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        self.get_attempt_pool(id).await
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>> {
        let records = sqlx::query_as::<_, TurnAttemptRecord>(
            r"
SELECT id, task_id, attempt_number, provider, requested_model,
       request_blob, response_blob, response_id, response_model,
       stop_reason, outcome, input_tokens, output_tokens,
       cached_input_tokens, cache_creation_input_tokens, route_provider,
       thinking_mode, thinking_budget_tokens, thinking_effort, opened_at, closed_at, duration_ms,
       otel_trace_id, otel_span_id
FROM agent_sdk_turn_attempts WHERE task_id = ?1 ORDER BY attempt_number
",
        )
        .bind(task_id.as_str())
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list attempts for task {task_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// CheckpointStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl CheckpointStore for SqliteDurableStore {
    async fn commit_checkpoint(&self, params: NewCheckpointParams) -> Result<Checkpoint> {
        let checkpoint = Checkpoint::new(params)?;
        let mut tx = self.begin().await?;
        // Duplicate check
        let dup_thread_key = thread_key(&checkpoint.thread_id);
        let dup_turn_number = i64::from(checkpoint.turn_number);
        let dup = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 AND turn_number = ?2",
            dup_thread_key,
            dup_turn_number,
        )
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        if dup.is_some() {
            return Err(anyhow!(
                "duplicate checkpoint for thread {} turn {}",
                checkpoint.thread_id,
                checkpoint.turn_number
            ));
        }
        Self::insert_checkpoint_tx(&mut tx, &checkpoint).await?;
        tx.commit().await.context("commit checkpoint insert")?;
        Ok(checkpoint)
    }

    async fn get(&self, id: &CheckpointId) -> Result<Option<Checkpoint>> {
        self.get_checkpoint_pool(id).await
    }

    async fn get_by_turn(
        &self,
        thread_id: &ThreadId,
        turn_number: u32,
    ) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as::<_, CheckpointRecord>(
            r"SELECT id, thread_id, turn_number, task_id, messages_json, agent_state_snapshot, turn_input_tokens, turn_output_tokens, kind, created_at FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 AND turn_number = ?2",
        )
        .bind(thread_key(thread_id))
        .bind(i64::from(turn_number))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get checkpoint {thread_id} turn {turn_number}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_latest_by_thread(&self, thread_id: &ThreadId) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as::<_, CheckpointRecord>(
            r"SELECT id, thread_id, turn_number, task_id, messages_json, agent_state_snapshot, turn_input_tokens, turn_output_tokens, kind, created_at FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 ORDER BY turn_number DESC LIMIT 1",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get latest checkpoint for {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<Checkpoint>> {
        let records = sqlx::query_as::<_, CheckpointRecord>(
            r"SELECT id, thread_id, turn_number, task_id, messages_json, agent_state_snapshot, turn_input_tokens, turn_output_tokens, kind, created_at FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 ORDER BY turn_number",
        )
        .bind(thread_key(thread_id))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list checkpoints for {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn threads_exceeding_checkpoint_count(
        &self,
        threshold: u32,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        let threshold_i64 = i64::from(threshold);
        let limit_i64 = i64::from(limit);
        let rows = sqlx::query_scalar!(
            r"SELECT thread_id FROM agent_sdk_turn_checkpoints GROUP BY thread_id HAVING COUNT(*) > ?1 ORDER BY thread_id LIMIT ?2",
            threshold_i64,
            limit_i64,
        )
        .fetch_all(&self.pool)
        .await
        .context("threads_exceeding_checkpoint_count")?;
        Ok(rows.into_iter().map(ThreadId::from_string).collect())
    }

    async fn delete_checkpoints_beyond_limit(
        &self,
        thread_id: &ThreadId,
        keep_latest_n: u32,
    ) -> Result<u64> {
        ensure!(keep_latest_n >= 1, "keep_latest_n must be at least 1");
        let tid = thread_key(thread_id);
        let keep = i64::from(keep_latest_n);
        let result = retry_busy(&self.lock_conflict_retries, "delete old checkpoints", || {
            sqlx::query!(
            r"DELETE FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 AND id NOT IN (SELECT id FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 ORDER BY turn_number DESC LIMIT ?2)",
            tid,
            keep,
        )
                .execute(&self.pool)
        })
        .await
        .with_context(|| format!("delete checkpoints beyond limit for {thread_id}"))?;
        Ok(result.rows_affected())
    }
}

// ─────────────────────────────────────────────────────────────────────
// EventRepository
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl EventRepository for SqliteDurableStore {
    async fn commit_event(
        &self,
        thread_id: &ThreadId,
        event: AgentEvent,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        let mut committed = self.commit_event_batch(thread_id, vec![event], now).await?;
        Ok(committed.remove(0))
    }

    async fn commit_event_batch(
        &self,
        thread_id: &ThreadId,
        events: Vec<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<Vec<CommittedEvent>> {
        ensure!(!events.is_empty(), "cannot commit an empty event batch");
        let mut tx = self.begin().await?;
        if let Some(thread) = Self::get_thread_tx(&mut tx, thread_id).await? {
            thread.require_active(ThreadOperation::Append)?;
        }
        let start_seq = Self::next_event_sequence_tx(&mut tx, thread_id).await?;
        let committed = Self::insert_events_tx(&mut tx, thread_id, events, start_seq, now).await?;
        tx.commit().await.context("commit event batch insert")?;
        Ok(committed)
    }

    async fn next_sequence(&self, thread_id: &ThreadId) -> Result<u64> {
        let thread_id_key = thread_key(thread_id);
        let record = sqlx::query!("SELECT COALESCE(MAX(sequence) + 1, 0) AS next_seq FROM agent_sdk_committed_events WHERE thread_id = ?1", thread_id_key)
            .fetch_one(&self.pool)
            .await
            .with_context(|| format!("next event sequence for {thread_id}"))?;
        let from_events =
            u64::try_from(record.next_seq).context("event next_sequence out of range")?;

        // When the janitor has purged all events, MAX() is NULL and
        // `from_events` falls back to 0. The retention floor records
        // the highest sequence already assigned, so clamping to it
        // prevents sequence regression after a full purge.
        let floor_record = sqlx::query!(
            r"SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = ?1",
            thread_id_key,
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("read retention floor for {thread_id}"))?;
        let floor = floor_record
            .map(|r| u64::try_from(r.retention_floor))
            .transpose()
            .context("retention_floor out of range")?
            .unwrap_or(0);

        Ok(from_events.max(floor))
    }

    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<CommittedEvent>> {
        let records = sqlx::query_as::<_, CommittedEventRecord>(
            "SELECT event_id, thread_id, sequence, event_json, committed_at FROM agent_sdk_committed_events WHERE thread_id = ?1 ORDER BY sequence",
        )
        .bind(thread_key(thread_id))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("get events for {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn get_events_in_range(
        &self,
        thread_id: &ThreadId,
        after_sequence: u64,
        up_to_sequence: u64,
    ) -> Result<Vec<CommittedEvent>> {
        let records = sqlx::query_as::<_, CommittedEventRecord>(
            "SELECT event_id, thread_id, sequence, event_json, committed_at FROM agent_sdk_committed_events WHERE thread_id = ?1 AND sequence > ?2 AND sequence <= ?3 ORDER BY sequence",
        )
        .bind(thread_key(thread_id))
        .bind(i64_from_u64(after_sequence, "after_sequence")?)
        .bind(i64_from_u64(up_to_sequence, "up_to_sequence")?)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("get events in range ({after_sequence}, {up_to_sequence}] for {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn threads_with_events_before(
        &self,
        cutoff: OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        let limit_i64 = i64::from(limit);
        let rows: Vec<String> = sqlx::query_scalar!(
            r"SELECT DISTINCT thread_id FROM agent_sdk_committed_events WHERE committed_at < ?1 ORDER BY thread_id LIMIT ?2",
            cutoff,
            limit_i64,
        )
        .fetch_all(&self.pool)
        .await
        .context("threads_with_events_before")?;
        Ok(rows.into_iter().map(ThreadId::from_string).collect())
    }

    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        let tid = thread_key(thread_id);
        let record = sqlx::query!(
            r#"SELECT MAX(sequence) AS "max_seq: i64" FROM agent_sdk_committed_events WHERE thread_id = ?1 AND committed_at < ?2"#,
            tid,
            cutoff,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("max_sequence_before for {thread_id}"))?;
        match record.max_seq {
            Some(v) => Ok(Some(u64_from_i64(v, "max_sequence_before")?)),
            None => Ok(None),
        }
    }

    async fn min_sequence_at_or_after(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        let tid = thread_key(thread_id);
        let record = sqlx::query!(
            r#"SELECT MIN(sequence) AS "min_seq: i64" FROM agent_sdk_committed_events WHERE thread_id = ?1 AND committed_at >= ?2"#,
            tid,
            cutoff,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("min_sequence_at_or_after for {thread_id}"))?;
        match record.min_seq {
            Some(v) => Ok(Some(u64_from_i64(v, "min_sequence_at_or_after")?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
impl SqliteDurableStore {
    /// Commit `events` and their coalesced advisory in one transaction, for
    /// the outbox tests that need a real `thread_events_available` row.
    async fn seed_events_with_outbox(
        &self,
        params: tests::SeedEvents,
    ) -> Result<tests::SeededEvents> {
        ensure!(
            !params.events.is_empty(),
            "cannot commit an empty event batch"
        );
        let mut tx = self.begin().await?;
        let start_seq = Self::next_event_sequence_tx(&mut tx, &params.thread_id).await?;
        let committed = Self::insert_events_tx(
            &mut tx,
            &params.thread_id,
            params.events,
            start_seq,
            params.now,
        )
        .await?;
        let outbox_row = Self::insert_thread_events_outbox_row_tx(
            &mut tx,
            &committed,
            params.outbox_max_attempts,
            params.now,
        )
        .await?;
        tx.commit()
            .await
            .context("commit atomic events + outbox transaction")?;
        Ok(tests::SeededEvents {
            committed_events: committed,
            outbox_row,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// TaskWakeupEmitter (Phase 10 · D — durable cross-process wakeup)
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl TaskWakeupEmitter for SqliteDurableStore {
    /// Insert a durable `task_wakeup` advisory row in its own
    /// transaction.  The admission paths use
    /// `Self::insert_task_wakeup_outbox_row_tx` directly so the
    /// wakeup is atomic with the journal mutation; this standalone
    /// hook covers callers that emit a wakeup without a surrounding
    /// journal write.
    async fn emit_in_transaction(&self, trigger: TaskWakeupTrigger) -> Result<OutboxRowId> {
        ensure!(
            trigger.max_attempts >= 1,
            "task_wakeup max_attempts must be at least 1"
        );
        let mut tx = self.begin().await?;
        let id = Self::insert_task_wakeup_outbox_row_tx(
            &mut tx,
            &trigger.task_id,
            &trigger.thread_id,
            trigger.max_attempts,
            trigger.now,
        )
        .await?;
        tx.commit()
            .await
            .context("commit task_wakeup emit transaction")?;
        Ok(id)
    }
}

// ─────────────────────────────────────────────────────────────────────
// OutboxStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl OutboxStore for SqliteDurableStore {
    async fn insert_batch(&self, rows: Vec<NewOutboxRow>) -> Result<Vec<OutboxRow>> {
        ensure!(!rows.is_empty(), "cannot insert an empty outbox batch");
        let mut tx = self.begin().await?;
        let mut result = Vec::with_capacity(rows.len());
        for params in rows {
            ensure!(
                kind_payload_invariants_hold(params.kind, params.event_id, params.sequence),
                "outbox row of kind {} has incompatible event_id/sequence",
                params.kind,
            );

            let id = OutboxRowId::new();
            let id_str: &str = id.as_str();
            let kind_str = params.kind.as_str();
            let thread_key = thread_key(&params.thread_id);
            let event_id_str = params.event_id.map(|uuid| uuid.to_string());
            let sequence_i64 = params
                .sequence
                .map(|seq| i64_from_u64(seq, "outbox sequence"))
                .transpose()?;
            let max_attempts = i64::from(params.max_attempts);
            sqlx::query!(
                r"INSERT INTO agent_sdk_outbox (id, kind, thread_id, event_id, sequence, status, payload_json, created_at, next_attempt_at, attempt_count, max_attempts) VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6, ?7, ?7, 0, ?8)",
                id_str,
                kind_str,
                thread_key,
                event_id_str,
                sequence_i64,
                params.payload_json,
                params.now,
                max_attempts,
            )
            .execute(&mut *tx)
            .await
            .with_context(|| format!("insert outbox row {id}"))?;
            result.push(OutboxRow {
                id,
                kind: params.kind,
                thread_id: params.thread_id,
                event_id: params.event_id,
                sequence: params.sequence,
                status: OutboxStatus::Pending,
                payload_json: params.payload_json,
                created_at: params.now,
                next_attempt_at: params.now,
                attempt_count: 0,
                max_attempts: params.max_attempts,
                last_error: None,
                claimed_by: None,
                claimed_at: None,
                delivered_at: None,
            });
        }
        tx.commit().await.context("commit outbox batch insert")?;
        Ok(result)
    }

    async fn claim_pending(
        &self,
        worker_id: &str,
        limit: u32,
        now: OffsetDateTime,
    ) -> Result<Vec<OutboxRow>> {
        // SQLite has no SKIP LOCKED / UPDATE RETURNING, so we SELECT
        // pending IDs in the desired order, UPDATE them, then SELECT
        // the claimed rows back.
        let mut tx = self.begin().await?;
        let limit_i64 = i64::from(limit);
        let ids: Vec<String> = sqlx::query_scalar!("SELECT id FROM agent_sdk_outbox WHERE status = 'pending' AND next_attempt_at <= ?1 ORDER BY next_attempt_at, id LIMIT ?2", now, limit_i64)
            .fetch_all(&mut *tx)
            .await
            .context("select pending outbox rows")?
            .into_iter()
            .flatten()
            .collect();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        for id in &ids {
            let id_str = id.as_str();
            sqlx::query!("UPDATE agent_sdk_outbox SET status = 'claimed', claimed_by = ?2, claimed_at = ?3 WHERE id = ?1", id_str, worker_id, now)
                .execute(&mut *tx)
                .await
                .with_context(|| format!("claim outbox row {id}"))?;
        }
        let mut rows = Vec::with_capacity(ids.len());
        for id in &ids {
            let record = sqlx::query_as::<_, OutboxRecord>(
                "SELECT id, kind, thread_id, event_id, sequence, status, payload_json, created_at, next_attempt_at, attempt_count, max_attempts, last_error, claimed_by, claimed_at, delivered_at FROM agent_sdk_outbox WHERE id = ?1",
            )
            .bind(id.as_str())
            .fetch_one(&mut *tx)
            .await
            .with_context(|| format!("reload claimed outbox row {id}"))?;
            rows.push(OutboxRow::try_from(record)?);
        }
        tx.commit().await.context("commit claim_pending")?;
        rows.sort_by_key(|r| (r.next_attempt_at, r.id.clone()));
        Ok(rows)
    }

    async fn mark_delivered(
        &self,
        id: &OutboxRowId,
        worker_id: &str,
        now: OffsetDateTime,
    ) -> Result<()> {
        let id_str = id.as_str();
        // Guard on `claimed_by`: only the worker still holding the claim may
        // ack. A reclaimed or cascade-deleted row matches zero rows — a safe
        // no-op (the current owner acks it; acking here would double-process).
        retry_busy(&self.lock_conflict_retries, "mark an outbox row delivered", || {
            sqlx::query!("UPDATE agent_sdk_outbox SET status = 'delivered', delivered_at = ?2 WHERE id = ?1 AND claimed_by = ?3", id_str, now, worker_id)
                .execute(&self.pool)
        })
        .await
            .with_context(|| format!("mark outbox row {id} delivered"))?;
        Ok(())
    }

    async fn mark_failed(
        &self,
        id: &OutboxRowId,
        worker_id: &str,
        error: &str,
        next_attempt_at: OffsetDateTime,
        _now: OffsetDateTime,
    ) -> Result<()> {
        let id_str = id.as_str();
        // `claimed_by = ?4` guard: a reclaimed or cascade-deleted row matches
        // zero rows — a safe no-op (the current owner handles it).
        retry_busy(
            &self.lock_conflict_retries,
            "mark an outbox row failed",
            || {
                sqlx::query!(
                    r"
UPDATE agent_sdk_outbox SET
    attempt_count = attempt_count + 1,
    status = CASE WHEN attempt_count + 1 >= max_attempts THEN 'expired' ELSE 'pending' END,
    last_error = CASE WHEN attempt_count + 1 >= max_attempts THEN ?2 ELSE NULL END,
    next_attempt_at = CASE WHEN attempt_count + 1 >= max_attempts THEN next_attempt_at ELSE ?3 END,
    claimed_by = CASE WHEN attempt_count + 1 >= max_attempts THEN claimed_by ELSE NULL END,
    claimed_at = CASE WHEN attempt_count + 1 >= max_attempts THEN claimed_at ELSE NULL END
WHERE id = ?1 AND claimed_by = ?4
",
                    id_str,
                    error,
                    next_attempt_at,
                    worker_id,
                )
                .execute(&self.pool)
            },
        )
        .await
        .with_context(|| format!("mark outbox row {id} failed"))?;
        Ok(())
    }

    async fn reclaim_expired_claims(
        &self,
        now: OffsetDateTime,
        claim_lease: time::Duration,
    ) -> Result<u64> {
        let threshold = now - claim_lease;
        let result = retry_busy(&self.lock_conflict_retries, "reclaim expired outbox claims", || {
            sqlx::query!(
            r"UPDATE agent_sdk_outbox SET status = 'pending', claimed_by = NULL, claimed_at = NULL, next_attempt_at = ?1 WHERE status = 'claimed' AND claimed_at IS NOT NULL AND claimed_at <= ?2",
            now,
            threshold,
        )
                .execute(&self.pool)
        })
        .await
        .context("reclaim expired outbox claims")?;
        Ok(result.rows_affected())
    }

    async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>> {
        let record = sqlx::query_as::<_, OutboxRecord>(
            "SELECT id, kind, thread_id, event_id, sequence, status, payload_json, created_at, next_attempt_at, attempt_count, max_attempts, last_error, claimed_by, claimed_at, delivered_at FROM agent_sdk_outbox WHERE id = ?1",
        )
        .bind(id.as_str())
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get outbox row {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>> {
        let records = sqlx::query_as::<_, OutboxRecord>(
            "SELECT id, kind, thread_id, event_id, sequence, status, payload_json, created_at, next_attempt_at, attempt_count, max_attempts, last_error, claimed_by, claimed_at, delivered_at FROM agent_sdk_outbox WHERE thread_id = ?1 ORDER BY CASE WHEN sequence IS NULL THEN 1 ELSE 0 END, sequence, id",
        )
        .bind(thread_key(thread_id))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list outbox rows for {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
        let thread_id_key = thread_key(thread_id);
        let record = sqlx::query!("SELECT COUNT(*) AS cnt FROM agent_sdk_outbox WHERE thread_id = ?1 AND status IN ('pending', 'claimed')", thread_id_key)
            .fetch_one(&self.pool)
            .await
            .with_context(|| format!("count pending outbox rows for {thread_id}"))?;
        let count: i64 = record.cnt;
        u64::try_from(count).context("outbox pending count out of range")
    }

    async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>> {
        let tid = thread_key(thread_id);
        let record = sqlx::query!(
            r#"SELECT MIN(sequence) AS "min_seq: i64" FROM agent_sdk_outbox WHERE thread_id = ?1 AND status IN ('pending', 'claimed') AND sequence IS NOT NULL"#,
            tid,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("min_unpublished_sequence for {thread_id}"))?;
        match record.min_seq {
            Some(v) => Ok(Some(u64_from_i64(v, "min_unpublished_sequence")?)),
            None => Ok(None),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// RetentionStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl RetentionStore for SqliteDurableStore {
    async fn get_cursor(&self, thread_id: &ThreadId) -> Result<Option<RetentionCursor>> {
        let record = sqlx::query_as::<_, RetentionCursorRecord>(
            "SELECT thread_id, retention_floor, updated_at FROM agent_sdk_retention_cursors WHERE thread_id = ?1",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get retention cursor for {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn advance_floor(
        &self,
        thread_id: &ThreadId,
        new_floor: u64,
        now: OffsetDateTime,
    ) -> Result<RetentionCursor> {
        let mut tx = self.begin().await?;
        let new_floor_i64 = i64_from_u64(new_floor, "retention floor")?;

        let retention_key = thread_key(thread_id);
        let current = sqlx::query!(
            "SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = ?1",
            retention_key,
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("read retention floor for {thread_id}"))?;
        if let Some(row) = current {
            let current_floor: i64 = row.retention_floor;
            ensure!(
                new_floor_i64 >= current_floor,
                "retention floor can only advance: current {current_floor}, requested {new_floor}"
            );
        }

        let upsert_key = thread_key(thread_id);
        sqlx::query!("INSERT INTO agent_sdk_retention_cursors (thread_id, retention_floor, updated_at) VALUES (?1, ?2, ?3) ON CONFLICT (thread_id) DO UPDATE SET retention_floor = excluded.retention_floor, updated_at = excluded.updated_at", upsert_key, new_floor_i64, now)
            .execute(&mut *tx)
            .await
            .with_context(|| format!("advance retention floor for {thread_id}"))?;

        let purge_key = thread_key(thread_id);
        sqlx::query!(
            "DELETE FROM agent_sdk_committed_events WHERE thread_id = ?1 AND sequence < ?2",
            purge_key,
            new_floor_i64,
        )
        .execute(&mut *tx)
        .await
        .with_context(|| format!("purge events below floor {new_floor} for {thread_id}"))?;

        tx.commit()
            .await
            .context("commit retention floor advance")?;
        Ok(RetentionCursor {
            thread_id: thread_id.clone(),
            retention_floor: new_floor,
            updated_at: now,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ToolAuditEventStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl ToolAuditEventStore for SqliteDurableStore {
    async fn record_event(&self, event: &ToolAuditEvent) -> Result<()> {
        let kind_payload = json_to_value(&event.kind, "tool audit event kind_payload")
            .with_context(|| format!("serialize tool audit event {} kind", event.id.as_str()))?;
        let kind_str = event.kind.as_str();
        let effect_class_wire = enum_to_wire(&event.effect_class)?;
        let id = event.id.as_str();
        let task_id = event.task_id.as_str();
        let parent_task_id = event.parent_task_id.as_str();
        let thread_id = thread_key(&event.thread_id);

        retry_busy(
            &self.lock_conflict_retries,
            "record a tool audit event",
            || {
                sqlx::query!(
                    r"
INSERT INTO agent_sdk_tool_audit_events (
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind, kind_payload,
    provider, model, input, output, error, recorded_at
) VALUES (
    ?1, ?2, ?3, ?4, ?5,
    ?6, ?7, ?8, ?9, ?10,
    ?11, ?12, ?13, ?14, ?15, ?16
)
",
                    id,
                    event.operation_id,
                    task_id,
                    parent_task_id,
                    thread_id,
                    event.tool_call_id,
                    event.tool_name,
                    effect_class_wire,
                    kind_str,
                    kind_payload,
                    event.provider,
                    event.model,
                    event.input,
                    event.output,
                    event.error,
                    event.recorded_at,
                )
                .execute(&self.pool)
            },
        )
        .await
        .with_context(|| format!("record tool audit event {id}"))?;
        Ok(())
    }

    async fn list_by_operation(&self, operation_id: &str) -> Result<Vec<ToolAuditEvent>> {
        let rows = sqlx::query_as::<_, ToolAuditEventRecord>(
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE operation_id = ?1
ORDER BY seq ASC
",
        )
        .bind(operation_id)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by operation {operation_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<ToolAuditEvent>> {
        let key = task_id.as_str();
        let rows = sqlx::query_as::<_, ToolAuditEventRecord>(
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE task_id = ?1
ORDER BY seq ASC
",
        )
        .bind(key)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by task {task_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<ToolAuditEvent>> {
        let key = thread_key(thread_id);
        let rows = sqlx::query_as::<_, ToolAuditEventRecord>(
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE thread_id = ?1
ORDER BY seq ASC
",
        )
        .bind(key)
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by thread {thread_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Record types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, FromRow)]
struct TaskRecord {
    id: String,
    kind: String,
    status: String,
    parent_id: Option<String>,
    root_id: String,
    depth: i64,
    thread_id: String,
    submitted_input_json: serde_json::Value,
    caller_metadata_json: Option<serde_json::Value>,
    worker_id: Option<String>,
    lease_id: Option<String>,
    lease_expires_at: Option<OffsetDateTime>,
    last_heartbeat_at: Option<OffsetDateTime>,
    state_json: serde_json::Value,
    attempt: i64,
    max_attempts: i64,
    last_error: Option<String>,
    terminal_reason_json: Option<serde_json::Value>,
    pending_child_count: i64,
    spawn_index: Option<i64>,
    result_payload: Option<serde_json::Value>,
    otel_traceparent: Option<String>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
    completed_at: Option<OffsetDateTime>,
    last_activity_at: Option<OffsetDateTime>,
    after_siblings_json: Option<serde_json::Value>,
}

impl TryFrom<TaskRecord> for AgentTask {
    type Error = anyhow::Error;
    fn try_from(r: TaskRecord) -> Result<Self> {
        let status = enum_from_wire(&r.status, "task status")?;
        let task = Self {
            id: AgentTaskId::from_string(r.id),
            kind: enum_from_wire(&r.kind, "task kind")?,
            status,
            parent_id: r.parent_id.map(AgentTaskId::from_string),
            root_id: AgentTaskId::from_string(r.root_id),
            depth: u32_from_i64(r.depth, "task depth")?,
            thread_id: ThreadId::from_string(r.thread_id),
            submitted_input: json_from_value(r.submitted_input_json, "task submitted_input")?,
            caller_metadata: r.caller_metadata_json,
            worker_id: r.worker_id.map(WorkerId::from_string),
            lease_id: r.lease_id.map(LeaseId::from_string),
            lease_expires_at: r.lease_expires_at,
            last_heartbeat_at: r.last_heartbeat_at,
            last_activity_at: r.last_activity_at,
            state: json_from_value(r.state_json, "task state")?,
            attempt: u32_from_i64(r.attempt, "task attempt")?,
            max_attempts: u32_from_i64(r.max_attempts, "task max_attempts")?,
            last_error: r.last_error,
            terminal_reason: r
                .terminal_reason_json
                .map(|value| json_from_value(value, "task terminal_reason_json"))
                .transpose()?,
            pending_child_count: u32_from_i64(r.pending_child_count, "task pending_child_count")?,
            spawn_index: r
                .spawn_index
                .map(|v| u32_from_i64(v, "task spawn_index"))
                .transpose()?,
            after_siblings: decoded_wait_list(
                status,
                r.after_siblings_json
                    .map(|value| json_from_value(value, "task after_siblings_json"))
                    .transpose()?
                    .unwrap_or_default(),
            ),
            result_payload: r.result_payload,
            otel_traceparent: r.otel_traceparent,
            created_at: r.created_at,
            updated_at: r.updated_at,
            completed_at: r.completed_at,
        };
        task.validate()
            .context("rehydrate task from sqlite row validation")?;
        Ok(task)
    }
}

#[derive(Debug, FromRow)]
struct ThreadRecord {
    thread_id: String,
    status: String,
    committed_turns: i64,
    total_input_tokens: i64,
    total_output_tokens: i64,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
    purge_receipt_json: Option<String>,
}

impl TryFrom<ThreadRecord> for Thread {
    type Error = anyhow::Error;
    fn try_from(r: ThreadRecord) -> Result<Self> {
        let status = enum_from_wire(&r.status, "thread status")?;
        let thread = Self {
            thread_id: ThreadId::from_string(r.thread_id),
            status,
            committed_turns: u32_from_i64(r.committed_turns, "thread committed_turns")?,
            total_usage: TokenUsage {
                input_tokens: u32_from_i64(r.total_input_tokens, "thread input tokens")?,
                output_tokens: u32_from_i64(r.total_output_tokens, "thread output tokens")?,
                ..Default::default()
            },
            created_at: r.created_at,
            updated_at: r.updated_at,
            purge_receipt: None,
            purge_seed: None,
        };
        let record_value = r
            .purge_receipt_json
            .map(|json| serde_json::from_str(&json).context("decode purge record"))
            .transpose()?;
        let (purge_receipt, purge_seed) = Self::split_purge_record(record_value);
        let thread = Self {
            purge_receipt,
            purge_seed,
            ..thread
        };
        thread
            .validate()
            .context("rehydrate thread from sqlite row validation")?;
        Ok(thread)
    }
}

/// Load and decode one message head exactly as every store read does.
pub(super) async fn read_message_head<'e, E>(
    executor: E,
    thread_id: &ThreadId,
) -> Result<Option<MessageProjection>>
where
    E: sqlx::Executor<'e, Database = Sqlite>,
{
    let record = sqlx::query_as::<_, MessageHeadRecord>(
        r"
SELECT thread_id, history_json, draft_messages_json, compactions_json, history_generation, version, created_at, updated_at
FROM agent_sdk_message_heads
WHERE thread_id = ?1
",
    )
    .bind(thread_key(thread_id))
    .fetch_optional(executor)
    .await
    .with_context(|| format!("get message head {thread_id}"))?;
    record.map(TryInto::try_into).transpose()
}

#[derive(Debug, FromRow)]
struct MessageHeadRecord {
    thread_id: String,
    history_json: serde_json::Value,
    /// In-flight draft snapshot, NULL when no turn is suspended.
    /// Populated by the worker at every tool-boundary suspension and
    /// cleared atomically by `commit_completed_turn_atomic_inner`.
    draft_messages_json: Option<serde_json::Value>,
    compactions_json: Option<serde_json::Value>,
    history_generation: i64,
    version: i64,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
}

impl TryFrom<MessageHeadRecord> for MessageProjection {
    type Error = anyhow::Error;
    fn try_from(r: MessageHeadRecord) -> Result<Self> {
        let messages: Vec<llm::Message> = json_from_value(r.history_json, "message head history")?;
        let draft_messages = match r.draft_messages_json {
            Some(value) => json_from_value(value, "message head draft messages")?,
            None => Vec::new(),
        };
        let mut compactions: Vec<CompactionEntry> = match r.compactions_json {
            Some(value) => json_from_value(value, "message head compactions")?,
            None => Vec::new(),
        };
        if !compaction_lineage_matches_history(&compactions, messages.len()) {
            compactions = Vec::new();
        }
        Ok(Self {
            thread_id: ThreadId::from_string(r.thread_id),
            messages,
            draft_messages,
            compactions,
            history_generation: u64_from_i64(
                r.history_generation,
                "message head history generation",
            )?,
            version: u64_from_i64(r.version, "message head version")?,
            created_at: r.created_at,
            updated_at: r.updated_at,
        })
    }
}

fn compaction_lineage_matches_history(compactions: &[CompactionEntry], history_len: usize) -> bool {
    let mut expected_start = 0;
    compactions.iter().all(|entry| {
        let matches = entry.compacted_start == expected_start
            && entry.compacted_end >= entry.compacted_start
            && entry.compacted_end <= history_len;
        expected_start = entry.compacted_end;
        matches
    })
}

#[derive(Debug, FromRow)]
struct TurnAttemptRecord {
    id: String,
    task_id: String,
    attempt_number: i64,
    provider: String,
    requested_model: String,
    request_blob: serde_json::Value,
    response_blob: Option<serde_json::Value>,
    response_id: Option<String>,
    response_model: Option<String>,
    stop_reason: Option<String>,
    outcome: Option<String>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cached_input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
    route_provider: Option<String>,
    thinking_mode: Option<String>,
    thinking_budget_tokens: Option<i64>,
    thinking_effort: Option<String>,
    opened_at: OffsetDateTime,
    closed_at: Option<OffsetDateTime>,
    duration_ms: Option<i64>,
    otel_trace_id: Option<String>,
    otel_span_id: Option<String>,
}

impl TryFrom<TurnAttemptRecord> for TurnAttempt {
    type Error = anyhow::Error;
    fn try_from(r: TurnAttemptRecord) -> Result<Self> {
        let attempt = Self {
            id: TurnAttemptId::from_string(r.id),
            task_id: AgentTaskId::from_string(r.task_id),
            attempt_number: u32_from_i64(r.attempt_number, "attempt_number")?,
            provider: r.provider,
            requested_model: r.requested_model,
            request_blob: r.request_blob,
            response_blob: r.response_blob,
            response_id: r.response_id,
            response_model: r.response_model,
            stop_reason: r
                .stop_reason
                .map(|v| enum_from_wire(&v, "turn attempt stop_reason"))
                .transpose()?,
            outcome: r
                .outcome
                .map(|v| enum_from_wire(&v, "turn attempt outcome"))
                .transpose()?,
            input_tokens: r
                .input_tokens
                .map(|v| u32_from_i64(v, "turn attempt input_tokens"))
                .transpose()?,
            output_tokens: r
                .output_tokens
                .map(|v| u32_from_i64(v, "turn attempt output_tokens"))
                .transpose()?,
            cached_input_tokens: r
                .cached_input_tokens
                .map(|v| u32_from_i64(v, "turn attempt cached_input_tokens"))
                .transpose()?,
            cache_creation_input_tokens: r
                .cache_creation_input_tokens
                .map(|v| u32_from_i64(v, "turn attempt cache_creation_input_tokens"))
                .transpose()?,
            route_provider: r.route_provider,
            thinking_mode: r
                .thinking_mode
                .map(|v| enum_from_wire(&v, "turn attempt thinking_mode"))
                .transpose()?,
            thinking_budget_tokens: r
                .thinking_budget_tokens
                .map(|v| u32_from_i64(v, "turn attempt thinking_budget_tokens"))
                .transpose()?,
            thinking_effort: r
                .thinking_effort
                .map(|v| enum_from_wire(&v, "turn attempt thinking_effort"))
                .transpose()?,
            opened_at: r.opened_at,
            closed_at: r.closed_at,
            duration_ms: r
                .duration_ms
                .map(|v| u64_from_i64(v, "turn attempt duration_ms"))
                .transpose()?,
            otel_trace_id: r.otel_trace_id,
            otel_span_id: r.otel_span_id,
        };
        attempt
            .validate()
            .context("rehydrate turn attempt from sqlite row validation")?;
        Ok(attempt)
    }
}

#[derive(Debug, FromRow)]
struct CheckpointRecord {
    id: String,
    thread_id: String,
    turn_number: i64,
    task_id: String,
    messages_json: serde_json::Value,
    agent_state_snapshot: serde_json::Value,
    turn_input_tokens: i64,
    turn_output_tokens: i64,
    kind: String,
    created_at: OffsetDateTime,
}

impl TryFrom<CheckpointRecord> for Checkpoint {
    type Error = anyhow::Error;
    fn try_from(r: CheckpointRecord) -> Result<Self> {
        let checkpoint = Self {
            id: CheckpointId::from_string(r.id),
            thread_id: ThreadId::from_string(r.thread_id),
            turn_number: u32_from_i64(r.turn_number, "checkpoint turn_number")?,
            task_id: AgentTaskId::from_string(r.task_id),
            messages: json_from_value(r.messages_json, "checkpoint messages")?,
            agent_state_snapshot: r.agent_state_snapshot,
            turn_usage: TokenUsage {
                input_tokens: u32_from_i64(r.turn_input_tokens, "checkpoint turn_input_tokens")?,
                output_tokens: u32_from_i64(r.turn_output_tokens, "checkpoint turn_output_tokens")?,
                ..Default::default()
            },
            kind: CheckpointKind::parse(&r.kind)
                .context("rehydrate checkpoint kind from sqlite row")?,
            created_at: r.created_at,
        };
        checkpoint
            .validate()
            .context("rehydrate checkpoint from sqlite row validation")?;
        Ok(checkpoint)
    }
}

#[derive(Debug, FromRow)]
struct CommittedEventRecord {
    event_id: String,
    thread_id: String,
    sequence: i64,
    event_json: serde_json::Value,
    committed_at: OffsetDateTime,
}

impl TryFrom<CommittedEventRecord> for CommittedEvent {
    type Error = anyhow::Error;
    fn try_from(r: CommittedEventRecord) -> Result<Self> {
        Ok(Self {
            event_id: uuid::Uuid::parse_str(&r.event_id).context("parse committed event UUID")?,
            thread_id: ThreadId::from_string(r.thread_id),
            sequence: u64_from_i64(r.sequence, "committed event sequence")?,
            timestamp: r.committed_at,
            event: json_from_value(r.event_json, "committed event payload")?,
        })
    }
}

#[derive(Debug, FromRow)]
struct OutboxRecord {
    id: String,
    kind: String,
    thread_id: String,
    event_id: Option<String>,
    sequence: Option<i64>,
    status: String,
    payload_json: serde_json::Value,
    created_at: OffsetDateTime,
    next_attempt_at: OffsetDateTime,
    attempt_count: i64,
    max_attempts: i64,
    last_error: Option<String>,
    claimed_by: Option<String>,
    claimed_at: Option<OffsetDateTime>,
    delivered_at: Option<OffsetDateTime>,
}

impl TryFrom<OutboxRecord> for OutboxRow {
    type Error = anyhow::Error;
    fn try_from(r: OutboxRecord) -> Result<Self> {
        let event_id = r
            .event_id
            .as_deref()
            .map(uuid::Uuid::parse_str)
            .transpose()
            .context("parse outbox event UUID")?;
        let sequence = r
            .sequence
            .map(|seq| u64_from_i64(seq, "outbox sequence"))
            .transpose()?;
        Ok(Self {
            id: OutboxRowId::from_string(r.id),
            kind: OutboxMessageKind::parse_wire(&r.kind).context("parse outbox kind")?,
            thread_id: ThreadId::from_string(r.thread_id),
            event_id,
            sequence,
            status: enum_from_wire(&r.status, "outbox status")?,
            payload_json: r.payload_json,
            created_at: r.created_at,
            next_attempt_at: r.next_attempt_at,
            attempt_count: u32_from_i64(r.attempt_count, "outbox attempt_count")?,
            max_attempts: u32_from_i64(r.max_attempts, "outbox max_attempts")?,
            last_error: r.last_error,
            claimed_by: r.claimed_by,
            claimed_at: r.claimed_at,
            delivered_at: r.delivered_at,
        })
    }
}

#[derive(Debug, FromRow)]
struct RetentionCursorRecord {
    thread_id: String,
    retention_floor: i64,
    updated_at: OffsetDateTime,
}

impl TryFrom<RetentionCursorRecord> for RetentionCursor {
    type Error = anyhow::Error;
    fn try_from(r: RetentionCursorRecord) -> Result<Self> {
        Ok(Self {
            thread_id: ThreadId::from_string(r.thread_id),
            retention_floor: u64_from_i64(r.retention_floor, "retention floor")?,
            updated_at: r.updated_at,
        })
    }
}

#[derive(Debug, FromRow)]
struct EffectRecordRow {
    operation_id: String,
    effect_class: String,
    tool_call_id: String,
    child_task_id: String,
    tool_name: String,
    input: serde_json::Value,
    status: String,
    error: Option<String>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
    effect_kind: String,
    attempt: i64,
    resolution: Option<String>,
    captured_output: Option<String>,
    journal_anchor: Option<i64>,
    thread_id: Option<String>,
}

impl EffectRecordRow {
    fn into_record(self) -> Result<EffectRecord> {
        EffectRecord::from_row(EffectRow {
            operation_id: self.operation_id,
            effect_class: self.effect_class,
            tool_call_id: self.tool_call_id,
            child_task_id: self.child_task_id,
            tool_name: self.tool_name,
            input: self.input,
            status: self.status,
            error: self.error,
            effect_kind: self.effect_kind,
            attempt: self.attempt,
            resolution: self.resolution,
            captured_output: self.captured_output,
            journal_anchor: self.journal_anchor,
            thread_id: self.thread_id,
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }
}

#[derive(Debug, FromRow)]
struct ToolAuditEventRecord {
    id: String,
    operation_id: String,
    task_id: String,
    parent_task_id: String,
    thread_id: String,
    tool_call_id: String,
    tool_name: String,
    effect_class: String,
    kind_payload: serde_json::Value,
    provider: String,
    model: String,
    input: Option<serde_json::Value>,
    output: Option<String>,
    error: Option<String>,
    recorded_at: OffsetDateTime,
}

impl TryFrom<ToolAuditEventRecord> for ToolAuditEvent {
    type Error = anyhow::Error;
    fn try_from(r: ToolAuditEventRecord) -> Result<Self> {
        Ok(Self {
            id: ToolAuditEventId::from_string(r.id),
            operation_id: r.operation_id,
            task_id: AgentTaskId::from_string(r.task_id),
            parent_task_id: AgentTaskId::from_string(r.parent_task_id),
            thread_id: ThreadId::from_string(r.thread_id),
            tool_call_id: r.tool_call_id,
            tool_name: r.tool_name,
            effect_class: enum_from_wire(&r.effect_class, "tool audit effect_class")?,
            kind: json_from_value(r.kind_payload, "tool audit event kind")?,
            provider: r.provider,
            model: r.model,
            input: r.input,
            output: r.output,
            error: r.error,
            recorded_at: r.recorded_at,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Wire format helpers
// ─────────────────────────────────────────────────────────────────────

const fn thread_key(thread_id: &ThreadId) -> &str {
    thread_id.0.as_str()
}

/// The persisted form of a task's wait list: `NULL` when it waits on
/// nothing, so acquisition can filter on `IS NULL`.
fn after_siblings_json(task: &AgentTask) -> Result<Option<serde_json::Value>> {
    if task.after_siblings.is_empty() {
        return Ok(None);
    }
    json_to_value(&task.after_siblings, "task after_siblings").map(Some)
}

fn json_to_value<T: Serialize + ?Sized>(value: &T, label: &str) -> Result<serde_json::Value> {
    serde_json::to_value(value).with_context(|| format!("serialize {label} to JSON"))
}

fn json_from_value<T: DeserializeOwned>(value: serde_json::Value, label: &str) -> Result<T> {
    serde_json::from_value(value).with_context(|| format!("deserialize {label} from JSON"))
}

fn enum_to_wire<T: Serialize>(value: &T) -> Result<String> {
    let v = serde_json::to_value(value).context("serialize enum to wire string")?;
    let wire = v
        .as_str()
        .context("enum wire value did not serialize as a string")?;
    Ok(wire.to_owned())
}

fn optional_enum_to_wire<T: Serialize>(value: Option<&T>) -> Result<Option<String>> {
    value.map(enum_to_wire).transpose()
}

fn enum_from_wire<T: DeserializeOwned>(wire: &str, label: &str) -> Result<T> {
    serde_json::from_value(serde_json::Value::String(wire.to_owned()))
        .with_context(|| format!("deserialize {label} from wire string {wire:?}"))
}

fn u32_from_i64(value: i64, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} out of range for u32: {value}"))
}

fn u64_from_i64(value: i64, label: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{label} out of range for u64: {value}"))
}

fn i64_from_u64(value: u64, label: &str) -> Result<i64> {
    i64::try_from(value).with_context(|| format!("{label} out of range for i64: {value}"))
}

#[cfg(test)]
mod tests {
    pub(super) struct SeedEvents {
        pub(super) thread_id: agent_sdk_foundation::ThreadId,
        pub(super) events: Vec<agent_sdk_foundation::events::AgentEvent>,
        pub(super) outbox_max_attempts: u32,
        pub(super) now: time::OffsetDateTime,
    }

    pub(super) struct SeededEvents {
        pub(super) committed_events: Vec<agent_server::journal::CommittedEvent>,
        pub(super) outbox_row: agent_server::journal::outbox::OutboxRow,
    }

    use agent_server::journal::BeginPurge;
    use agent_server::journal::thread::PurgeSeed;
    use anyhow::{Context, Result};
    use time::Duration;

    use agent_sdk_foundation::events::AgentEvent;
    use agent_sdk_foundation::{
        AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, QuestionAnswer,
        QuestionPayload, ThreadId, TokenUsage, ToolReplay, ToolTier,
    };
    use agent_server::journal::effect::{
        EffectIntent, EffectKey, EffectState, EffectSubject, EffectWrite,
    };
    use agent_server::journal::store::{AgentTaskStore, QuestionPause};
    use agent_server::journal::task::{
        AgentTask, AgentTaskId, LeaseId, SuspensionPayload, WorkerId,
    };
    use agent_server::journal::task_state::TaskState;
    use agent_server::journal::thread_store::ThreadStore;
    use agent_server::journal::tool_audit::ToolEffectClass;
    use agent_server::journal::turn_attempt::{TurnAttempt, TurnAttemptId};
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;
    use agent_server::journal::{Guard, Transition, Write};

    use super::SqliteDurableStore;

    /// A commit that cannot get the write lock through the busy timeout and
    /// every retry is marked a lock conflict with its retry count; a commit
    /// that fails for any other reason is not.
    #[tokio::test]
    async fn a_commit_that_stays_locked_is_a_lock_conflict() -> Result<()> {
        use sqlx::Connection as _;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("busy.db");
        let url = format!("sqlite://{}?mode=rwc", path.display());
        let store = SqliteDurableStore::connect(&url).await?;
        let thread_id = agent_sdk_foundation::ThreadId::from_string("stays-locked");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;
        let append = |thread: &agent_sdk_foundation::ThreadId| Transition {
            guards: Vec::new(),
            writes: vec![Write::AppendEvents {
                thread: thread.clone(),
                events: vec![AgentEvent::text("m1", "waits")],
            }],
        };

        // Another writer holds the lock; this store waits 50 ms, not 5 s.
        let mut holder = sqlx::SqliteConnection::connect(&url).await?;
        sqlx::query("BEGIN IMMEDIATE").execute(&mut holder).await?;
        let impatient = SqliteDurableStore::from_pool(
            sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(1)
                .after_connect(|conn, _meta| {
                    Box::pin(async move {
                        sqlx::query("PRAGMA busy_timeout = 50")
                            .execute(conn)
                            .await?;
                        Ok(())
                    })
                })
                .connect(&url)
                .await?,
        );
        let locked =
            agent_server::journal::DurableStore::commit(&impatient, append(&thread_id), t0()).await;
        let Err(agent_server::journal::Rejected::Store(error)) = locked else {
            anyhow::bail!("expected a store error, got {locked:?}");
        };
        assert_eq!(
            error.downcast_ref::<agent_server::journal::LockConflict>(),
            Some(&agent_server::journal::LockConflict {
                retries: u32::try_from(super::BEGIN_BUSY_BACKOFF.len())?,
            }),
            "{error:#}"
        );
        sqlx::query("ROLLBACK").execute(&mut holder).await?;

        // Unlocked, a commit that fails otherwise (a missing thread) is not
        // a lock conflict.
        let missing = agent_sdk_foundation::ThreadId::from_string("never-created");
        let refused =
            agent_server::journal::DurableStore::commit(&impatient, append(&missing), t0()).await;
        let Err(agent_server::journal::Rejected::Store(error)) = refused else {
            anyhow::bail!("expected a store error, got {refused:?}");
        };
        assert_eq!(
            error.downcast_ref::<agent_server::journal::LockConflict>(),
            None,
            "{error:#}"
        );
        Ok(())
    }

    /// Eight sessions claim effects on one SQLite file while another writer
    /// holds the write lock for longer than the busy timeout: each write
    /// transaction waits and retries instead of failing, and every claim
    /// commits (dogfood D8: an effect claim failed with "database is
    /// locked" at `BEGIN`, so its tool never ran).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn effect_claims_survive_a_writer_that_holds_the_lock_past_the_busy_timeout() -> Result<()>
    {
        use agent_server::journal::DurableStore as _;
        use sqlx::Connection as _;

        const SESSIONS: usize = 8;
        let dir = tempfile::tempdir()?;
        let url = format!(
            "sqlite://{}?mode=rwc",
            dir.path().join("contended.db").display()
        );
        drop(SqliteDurableStore::connect(&url).await?);
        let store = SqliteDurableStore::from_pool(
            sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(u32::try_from(SESSIONS)?)
                .after_connect(|conn, _meta| {
                    Box::pin(async move {
                        sqlx::query("PRAGMA busy_timeout = 200")
                            .execute(conn)
                            .await?;
                        Ok(())
                    })
                })
                .connect(&url)
                .await?,
        );

        let mut holder = sqlx::SqliteConnection::connect(&url).await?;
        sqlx::query("BEGIN IMMEDIATE").execute(&mut holder).await?;
        let claims: Vec<_> = (0..SESSIONS)
            .map(|session| {
                let store = store.clone();
                tokio::spawn(async move {
                    seed_started_effect(&store, &format!("contended-{session}")).await
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        sqlx::query("ROLLBACK").execute(&mut holder).await?;

        for claim in claims {
            let (_task, key) = claim.await??;
            let record = store.effect(&key).await?.context("the claimed effect")?;
            assert!(record.state.is_started(), "{key}: {:?}", record.state);
        }
        assert!(
            store.lock_conflict_retries() > 0,
            "the writer held the lock, so some transaction must have retried"
        );
        Ok(())
    }

    /// The autocommit writes retry too: eight requests claim and record an
    /// idempotency key (the first write of `CreateThread`, `ForkThread`,
    /// `DecideConfirmation` and `AnswerQuestion`) while another writer holds
    /// the lock longer than the busy timeout. Every claim is `Fresh` and every
    /// record lands.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idempotency_claims_survive_a_writer_that_holds_the_lock_past_the_busy_timeout()
    -> Result<()> {
        use agent_server::journal::idempotency::{
            IdempotencyClaim, IdempotencyKind, IdempotencyRecord,
        };
        use sqlx::Connection as _;

        const REQUESTS: usize = 8;
        let dir = tempfile::tempdir()?;
        let url = format!(
            "sqlite://{}?mode=rwc",
            dir.path().join("claims.db").display()
        );
        drop(SqliteDurableStore::connect(&url).await?);
        let store = SqliteDurableStore::from_pool(
            sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(u32::try_from(REQUESTS)?)
                .after_connect(|conn, _meta| {
                    Box::pin(async move {
                        sqlx::query("PRAGMA busy_timeout = 200")
                            .execute(conn)
                            .await?;
                        Ok(())
                    })
                })
                .connect(&url)
                .await?,
        );

        let mut holder = sqlx::SqliteConnection::connect(&url).await?;
        sqlx::query("BEGIN IMMEDIATE").execute(&mut holder).await?;
        let requests: Vec<_> = (0..REQUESTS)
            .map(|request| {
                let store = store.clone();
                tokio::spawn(async move {
                    let request_id = format!("req-{request}");
                    let claim = store
                        .claim_idempotency(&request_id, IdempotencyKind::CreateThread, b"print")
                        .await?;
                    anyhow::ensure!(
                        matches!(claim, IdempotencyClaim::Fresh),
                        "{request_id}: {claim:?}"
                    );
                    store
                        .record_idempotency(IdempotencyRecord {
                            request_id: request_id.clone(),
                            kind: IdempotencyKind::CreateThread,
                            fingerprint: b"print".to_vec(),
                            result_json: serde_json::json!({ "thread": request }),
                        })
                        .await?;
                    Ok::<_, anyhow::Error>(request_id)
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        sqlx::query("ROLLBACK").execute(&mut holder).await?;

        for request in requests {
            let request_id = request.await??;
            let replay = store
                .claim_idempotency(&request_id, IdempotencyKind::CreateThread, b"print")
                .await?;
            assert!(
                matches!(replay, IdempotencyClaim::Replay(_)),
                "{request_id} was recorded: {replay:?}"
            );
        }
        assert!(
            store.lock_conflict_retries() > 0,
            "the writer held the lock, so some write must have retried"
        );
        Ok(())
    }

    fn t0() -> time::OffsetDateTime {
        time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> time::OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    /// Durable coverage of `claim_thread_creation_tx`: the caller-minted
    /// claim must replay across calls AND across a real reopen, and a
    /// conflicting creation must stay typed after restart.
    #[tokio::test]
    async fn caller_minted_creation_claim_survives_reopen_and_conflicts() -> Result<()> {
        use agent_server::journal::thread_store::{
            ThreadCreation, ThreadCreationOutcome, ThreadIdConflict, ThreadStore,
        };

        let db_path = std::env::temp_dir().join(format!(
            "agent_sdk_creation_claim_{}.db",
            uuid::Uuid::new_v4().simple()
        ));
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let thread_id = ThreadId::from_string("00000000-0000-4000-8000-00000000c1a1");
        let source_id = ThreadId::from_string("00000000-0000-4000-8000-00000000c1a2");
        let fork = ThreadCreation::Fork {
            source_thread_id: source_id,
            fork_after_committed_turns: 0,
        };

        {
            let store = SqliteDurableStore::connect(&url).await?;
            let first = store
                .get_or_create_for_creation(&thread_id, &ThreadCreation::Create, t0())
                .await?;
            assert_eq!(first, ThreadCreationOutcome::Created);
            let second = store
                .get_or_create_for_creation(&thread_id, &ThreadCreation::Create, t_plus(1))
                .await?;
            assert_eq!(second, ThreadCreationOutcome::Existing);
            let conflict = store
                .get_or_create_for_creation(&thread_id, &fork, t_plus(2))
                .await
                .expect_err("a fork claim on a created id must conflict");
            assert!(
                conflict.downcast_ref::<ThreadIdConflict>().is_some(),
                "expected ThreadIdConflict, got {conflict:#}",
            );
        }

        let reopened = SqliteDurableStore::connect(&url).await?;
        let replay = reopened
            .get_or_create_for_creation(&thread_id, &ThreadCreation::Create, t_plus(3))
            .await?;
        assert_eq!(
            replay,
            ThreadCreationOutcome::Existing,
            "the creation claim must survive a restart",
        );
        let conflict = reopened
            .get_or_create_for_creation(&thread_id, &fork, t_plus(4))
            .await
            .expect_err("the conflict must stay typed after restart");
        assert!(
            conflict.downcast_ref::<ThreadIdConflict>().is_some(),
            "expected ThreadIdConflict after reopen, got {conflict:#}",
        );

        drop(reopened);
        let _ = std::fs::remove_file(&db_path);
        Ok(())
    }

    #[tokio::test]
    async fn purge_record_check_binds_record_presence_to_lifecycle() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        for (suffix, status, record) in [
            ("deleting_null", "'deleting'", "NULL"),
            ("deleted_null", "'deleted'", "NULL"),
            (
                "active_with_record",
                "'active'",
                "'{\"root_thread_id\":\"x\"}'",
            ),
        ] {
            let sql = format!(
                "INSERT INTO agent_sdk_threads (
                    thread_id, status, committed_turns, total_input_tokens,
                    total_output_tokens, created_at, updated_at, purge_receipt_json
                 ) VALUES ('t-{suffix}', {status}, 0, 0, 0, '2026-01-01', '2026-01-01', {record})"
            );
            let result = sqlx::query(sqlx::AssertSqlSafe(sql))
                .execute(store.pool())
                .await;
            let error = result.expect_err("lifecycle-incoherent purge record must not persist");
            assert!(
                error.to_string().to_ascii_lowercase().contains("check"),
                "unexpected error for {suffix}: {error}",
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn begin_purge_persists_the_seed_on_the_deleting_row() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-seeded-fence");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let seed = PurgeSeed {
            root_thread_id: ThreadId::from_string("t-tree-root"),
            scope: agent_server::journal::PurgeScope::InvocationTree,
            started_at: t_plus(1),
        };
        let outcome = ThreadStore::begin_purge(&store, &thread_id, seed.clone()).await?;
        assert_eq!(outcome, BeginPurge::Fenced(seed.clone()));

        let fenced = ThreadStore::get(&store, &thread_id)
            .await?
            .context("fenced row")?;
        assert_eq!(fenced.purge_seed, Some(seed.clone()));

        let retry = ThreadStore::begin_purge(
            &store,
            &thread_id,
            PurgeSeed {
                root_thread_id: thread_id.clone(),
                scope: agent_server::journal::PurgeScope::Thread,
                started_at: t_plus(60),
            },
        )
        .await?;
        assert_eq!(
            retry,
            BeginPurge::Fenced(seed),
            "a retry resumes under the FIRST fence's stored seed",
        );
        Ok(())
    }

    #[tokio::test]
    async fn thinking_effort_with_mode_off_is_rejected_by_the_check_constraint() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let task = AgentTask::new_root_turn(ThreadId::from_string("off-with-effort"), t0(), 3);
        AgentTaskStore::submit_root_turn(&store, task.clone()).await?;

        let result = sqlx::query(
            r"
INSERT INTO agent_sdk_turn_attempts (
    id, task_id, attempt_number, provider, requested_model, request_blob,
    opened_at, thinking_mode, thinking_effort
) VALUES (?1, ?2, 1, 'anthropic', 'claude-sonnet-4-6', ?3, ?4, 'off', 'high')
",
        )
        .bind("attempt_off_with_effort")
        .bind(task.id.as_str())
        .bind(serde_json::json!({"messages": []}))
        .bind(t0())
        .execute(store.pool())
        .await;

        let error = result.expect_err("an off-mode row carrying an effort must not persist");
        assert!(
            error.to_string().to_ascii_lowercase().contains("check"),
            "unexpected error: {error}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn terminal_reason_triggers_survive_the_task_table_rebuild() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let task = AgentTask::new_root_turn(ThreadId::from_string("trigger-survival"), t0(), 3);
        AgentTaskStore::submit_root_turn(&store, task.clone()).await?;

        let result = sqlx::query(
            "INSERT INTO agent_sdk_tasks (
                id, kind, status, root_id, depth, thread_id, attempt, max_attempts,
                created_at, updated_at, completed_at
             ) VALUES ('task_no_reason', 'root_turn', 'completed', 'task_no_reason', 0,
                'trigger-survival', 0, 1, '2026-01-01', '2026-01-01', '2026-01-01')",
        )
        .execute(store.pool())
        .await;

        let error = result.expect_err("a terminal row without a reason must be rejected");
        assert!(
            error
                .to_string()
                .contains("terminal_reason_json must be set exactly"),
            "unexpected error: {error}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn budget_mode_and_tokens_must_travel_together_in_the_check_constraint() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let task = AgentTask::new_root_turn(ThreadId::from_string("budget-coherence"), t0(), 3);
        AgentTaskStore::submit_root_turn(&store, task.clone()).await?;

        for (suffix, mode, tokens) in [
            ("missing_tokens", "'budget'", "NULL"),
            ("stray_tokens", "'adaptive'", "9000"),
        ] {
            let sql = format!(
                "INSERT INTO agent_sdk_turn_attempts (
                    id, task_id, attempt_number, provider, requested_model, request_blob,
                    opened_at, thinking_mode, thinking_budget_tokens
                ) VALUES (?1, ?2, 1, 'anthropic', 'claude-sonnet-4-6', ?3, ?4, {mode}, {tokens})"
            );
            let result = sqlx::query(sqlx::AssertSqlSafe(sql))
                .bind(format!("attempt_budget_{suffix}"))
                .bind(task.id.as_str())
                .bind(serde_json::json!({"messages": []}))
                .bind(t0())
                .execute(store.pool())
                .await;
            let error = result.expect_err("incoherent budget evidence must not persist");
            assert!(
                error.to_string().to_ascii_lowercase().contains("check"),
                "unexpected error for {suffix}: {error}",
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn evidence_migration_preserves_legacy_attempt_as_null_read_fields() -> Result<()> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await?;
        let migrations = crate::sqlite::migrations::durable_core_migrations();
        // Locate 0013 by version, never by position: this batch stacks
        // 0013–0018 with a reserved slot in play, and a renumber that shifted
        // the ladder would silently turn a hardcoded index into a test that
        // asserts something else entirely instead of failing.
        let evidence_index = migrations
            .iter()
            .position(|migration| migration.version == "0013")
            .context("migration 0013 missing from the reviewable bundle")?;
        for migration in migrations.iter().take(evidence_index) {
            sqlx::raw_sql(migration.sql).execute(&pool).await?;
        }

        let store = SqliteDurableStore::from_pool(pool.clone());
        let task_id = "task_legacy_evidence";
        sqlx::query(
            "INSERT INTO agent_sdk_threads (
                thread_id, status, committed_turns, total_input_tokens, total_output_tokens,
                created_at, updated_at
             ) VALUES ('legacy-evidence', 'active', 0, 0, 0, '2026-01-01', '2026-01-01')",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "INSERT INTO agent_sdk_tasks (
                id, kind, status, root_id, depth, thread_id, attempt, max_attempts,
                created_at, updated_at
             ) VALUES (?, 'root_turn', 'queued', ?, 0, 'legacy-evidence', 0, 3,
                '2026-01-01', '2026-01-01')",
        )
        .bind(task_id)
        .bind(task_id)
        .execute(&pool)
        .await?;
        let attempt_id = TurnAttemptId::from_string("attempt_legacy_evidence");
        let request_blob = serde_json::json!({"messages": []});
        sqlx::query!(
            r"
INSERT INTO agent_sdk_turn_attempts (
    id, task_id, attempt_number, provider, requested_model, request_blob, opened_at
) VALUES (?1, ?2, 1, 'anthropic', 'claude-sonnet-4-6', ?3, ?4)
",
            attempt_id.as_str(),
            task_id,
            request_blob,
            t0(),
        )
        .execute(&pool)
        .await?;

        let evidence_migration = &migrations[evidence_index];
        sqlx::raw_sql(evidence_migration.sql).execute(&pool).await?;

        let legacy: TurnAttempt = TurnAttemptStore::get(&store, &attempt_id)
            .await?
            .context("legacy attempt missing after migration")?;
        assert!(legacy.cache_creation_input_tokens.is_none());
        assert!(legacy.route_provider.is_none());
        assert!(
            legacy.thinking_mode.is_none(),
            "pre-migration rows must read as evidence-not-captured",
        );
        assert!(legacy.thinking_budget_tokens.is_none());
        assert!(legacy.thinking_effort.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn acquire_skips_the_fenced_head_and_claims_the_next_thread() -> Result<()> {
        use agent_server::journal::PurgeScope;
        use agent_server::journal::store::AgentTaskStore as _;
        use agent_server::journal::task::{AgentTask, LeaseId, WorkerId};
        use agent_server::journal::thread_store::ThreadStore as _;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let fenced_thread = ThreadId::from_string("t-acq-fenced");
        let live_thread = ThreadId::from_string("t-acq-live");
        store.get_or_create(&fenced_thread, t0()).await?;
        store.get_or_create(&live_thread, t0()).await?;
        let fenced_root = AgentTask::new_root_turn(fenced_thread.clone(), t0(), 3);
        let fenced_root_id = fenced_root.id.clone();
        let live_root = AgentTask::new_root_turn(live_thread.clone(), t_plus(1), 3);
        let live_root_id = live_root.id.clone();
        store.submit_root_turn(fenced_root).await?;
        store.submit_root_turn(live_root).await?;
        let _ = store
            .begin_purge(
                &fenced_thread,
                PurgeSeed {
                    root_thread_id: fenced_thread.clone(),
                    scope: PurgeScope::Thread,
                    started_at: t_plus(2),
                },
            )
            .await?;

        let claimed = store
            .acquire_next_runnable(
                WorkerId::from_string("w-skip"),
                LeaseId::from_string("l-skip"),
                t_plus(600),
                t_plus(3),
            )
            .await?
            .context("acquire must skip the fenced head and claim the live root")?;
        assert_eq!(claimed.id, live_root_id);

        let none = store
            .acquire_next_runnable(
                WorkerId::from_string("w-skip-2"),
                LeaseId::from_string("l-skip-2"),
                t_plus(600),
                t_plus(4),
            )
            .await?;
        assert!(none.is_none(), "only the fenced row remains: {none:?}");
        let fenced_after =
            agent_server::journal::store::AgentTaskStore::get(&store, &fenced_root_id)
                .await?
                .context("fenced root survives")?;
        assert_eq!(
            fenced_after.status,
            agent_server::journal::TaskStatus::Pending
        );
        Ok(())
    }

    /// A running task on `{name}` with one started effect, keyed by its
    /// first tool call.
    async fn seed_started_effect(
        store: &SqliteDurableStore,
        name: &str,
    ) -> Result<(AgentTask, EffectKey)> {
        use agent_server::journal::DurableStore;

        let thread = ThreadId::from_string(name);
        ThreadStore::get_or_create(store, &thread, t0()).await?;
        let task = AgentTask::new_root_turn(thread.clone(), t0(), 3);
        let task_id = task.id.clone();
        store.submit_root_turn(task).await?;
        let (worker, lease) = (WorkerId::new(), LeaseId::new());
        let task = store
            .try_acquire_task(
                &task_id,
                worker.clone(),
                lease.clone(),
                t_plus(600),
                t_plus(1),
            )
            .await?
            .context("acquire the effect owner")?;
        let key = EffectKey::tool(&task.id, "call_1");
        store
            .commit(
                Transition {
                    guards: vec![
                        Guard::TaskIs {
                            task: task.id.clone(),
                            status: agent_server::journal::TaskStatus::Running,
                            lease: Some((worker, lease)),
                        },
                        Guard::EffectFree { key: key.clone() },
                    ],
                    writes: vec![Write::Effect(EffectWrite::Start(EffectIntent {
                        key: key.clone(),
                        replay: ToolReplay::Unsafe,
                        owner: task.id.clone(),
                        thread,
                        subject: EffectSubject::Tool {
                            call_id: "call_1".into(),
                            name: "test_tool".into(),
                            input: serde_json::json!({"key": "value"}),
                        },
                        attempt: 1,
                    }))],
                },
                t_plus(2),
            )
            .await?;
        Ok((task, key))
    }

    #[tokio::test]
    async fn effect_rows_survive_reopen() -> Result<()> {
        use agent_server::journal::DurableStore;

        let db_path = std::env::temp_dir().join(format!(
            "agent_sdk_effect_restart_{}.db",
            uuid::Uuid::new_v4().simple()
        ));
        let url = format!("sqlite://{}?mode=rwc", db_path.display());

        let store = SqliteDurableStore::connect(&url).await?;
        let (task, key) = seed_started_effect(&store, "effect-restart").await?;
        drop(store);

        let reopened = SqliteDurableStore::connect(&url).await?;
        let loaded = reopened
            .effect(&key)
            .await?
            .context("the record survives a reopen")?;
        assert_eq!(loaded.state, EffectState::Started);
        assert_eq!(reopened.effects_for_task(&task.id).await?, vec![loaded]);

        drop(reopened);
        let _ = std::fs::remove_file(&db_path);
        Ok(())
    }

    /// 0027's checks hold on SQLite: an unknown effect kind and a
    /// resolution on a row that is not `failed` are refused.
    #[tokio::test]
    async fn effect_checks_refuse_unknown_kinds_and_stray_resolutions() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let now = t0();
        let unknown_kind = sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at, effect_kind
) VALUES ('op_kind', 'side_effecting', 'call', 'task', 'tool', '{}', 'started', NULL, ?1, ?1, 'shell_out')
",
            now,
        )
        .execute(&store.pool)
        .await;
        assert!(
            unknown_kind.is_err(),
            "an unknown effect_kind must be refused"
        );

        let stray_resolution = sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at, resolution
) VALUES ('op_resolution', 'side_effecting', 'call', 'task', 'tool', '{}', 'started', NULL, ?1, ?1, 'interrupted')
",
            now,
        )
        .execute(&store.pool)
        .await;
        assert!(
            stray_resolution.is_err(),
            "only a failed row carries a resolution"
        );
        Ok(())
    }

    /// A row an earlier release wrote, with none of 0027's columns set,
    /// reads as a started, unsafe tool call: recovery settles it as
    /// interrupted, as that release would have.
    #[tokio::test]
    async fn legacy_intent_rows_read_as_started_tool_calls() -> Result<()> {
        use agent_server::journal::DurableStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let now = t0();
        sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
) VALUES ('task_legacy:call_1', 'resumable', 'call_1', 'task_legacy', 'bash', '{}', 'pending', NULL, ?1, ?1)
",
            now,
        )
        .execute(&store.pool)
        .await?;

        let record = store
            .effect(&EffectKey::from_stored("task_legacy:call_1"))
            .await?
            .context("the legacy row reads")?;
        assert_eq!(record.state, EffectState::Started);
        assert_eq!(record.replay, ToolReplay::Unsafe);
        assert_eq!(record.thread, None);
        assert_eq!(record.journal_anchor, None);
        assert!(matches!(record.subject, EffectSubject::Tool { ref name, .. } if name == "bash"));
        Ok(())
    }

    #[tokio::test]
    async fn clear_removes_effect_rows() -> Result<()> {
        use agent_server::journal::DurableStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let (task, key) = seed_started_effect(&store, "effect-clear").await?;

        AgentTaskStore::clear(&store).await?;

        assert!(store.effect(&key).await?.is_none());
        assert_eq!(store.effects_for_task(&task.id).await?, Vec::new());
        Ok(())
    }

    /// Seed a store lifetime with a submitted, acquired, parked, and
    /// answered question root, so the restart test's subject is the
    /// reopen boundary rather than a long setup.
    async fn seed_answered_question(url: &str, thread_id: &ThreadId) -> Result<AgentTaskId> {
        let store = SqliteDurableStore::connect(url).await?;
        store.get_or_create(thread_id, t0()).await?;
        let task = AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
        let task_id = task.id.clone();
        store.submit_root_turn(task).await?;
        let worker = WorkerId::from_string("question-worker");
        let lease = LeaseId::from_string("question-lease");
        store
            .try_acquire_task(
                &task_id,
                worker.clone(),
                lease.clone(),
                t_plus(60),
                t_plus(1),
            )
            .await?
            .context("acquire question task")?;
        let tool_call = PendingToolCallInfo {
            id: "question-call".into(),
            name: "ask_user".into(),
            display_name: "Ask User".into(),
            tier: ToolTier::Observe,
            concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
            replay: agent_sdk_foundation::ToolReplay::Unsafe,
            input: serde_json::json!({"question": "Where?"}),
            effective_input: serde_json::json!({"question": "Where?"}),
            listen_context: None,
        };
        let (_paused, committed) = store
            .pause_on_question(
                &task_id,
                &worker,
                &lease,
                QuestionPause {
                    delivered_injection_ids: Vec::new(),
                    payload: SuspensionPayload {
                        continuation: ContinuationEnvelope::wrap(AgentContinuation {
                            thread_id: thread_id.clone(),
                            turn: 1,
                            total_usage: TokenUsage::default(),
                            turn_usage: TokenUsage::default(),
                            pending_tool_calls: vec![tool_call],
                            awaiting_index: 0,
                            completed_results: Vec::new(),
                            state: AgentState::new(thread_id.clone()),
                            response_id: None,
                            stop_reason: None,
                            response_content: Vec::new(),
                        }),
                        suspended_messages: Vec::new(),
                        child_join_policy: agent_server::ChildJoinPolicy::default(),
                    },
                    questions: vec![QuestionPayload {
                        tool_call_id: "question-call".into(),
                        question: "Where?".into(),
                        header: None,
                        options: Vec::new(),
                        multi_select: false,
                    }],
                    events: Vec::new(),
                },
                t_plus(2),
            )
            .await?;
        assert!(
            committed
                .iter()
                .any(|event| matches!(event.event, AgentEvent::QuestionAsked { .. })),
            "pause_on_question must journal QuestionAsked in the same transaction",
        );
        store
            .answer_question(
                &task_id,
                "restart-receipt",
                vec![QuestionAnswer {
                    tool_call_id: "question-call".into(),
                    answer: "Staging".into(),
                }],
                t_plus(3),
            )
            .await?;
        Ok(task_id)
    }

    #[tokio::test]
    async fn answered_question_survives_sqlite_restart_and_reacquire() -> Result<()> {
        let db_path = std::env::temp_dir().join(format!(
            "agent_sdk_answered_question_restart_{}.db",
            uuid::Uuid::new_v4().simple()
        ));
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let thread_id = ThreadId::from_string("question-restart-thread");
        let task_id = seed_answered_question(&url, &thread_id).await?;

        let reopened = SqliteDurableStore::connect(&url).await?;
        let loaded = AgentTaskStore::get(&reopened, &task_id)
            .await?
            .context("answered question must survive restart")?;
        let (receipt_id, answers) = loaded
            .state
            .question_answers()
            .with_context(|| format!("restarted task lost AnsweredQuestion: {:?}", loaded.state))?;
        assert_eq!(receipt_id, "restart-receipt");
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].tool_call_id, "question-call");
        assert_eq!(answers[0].answer, "Staging");
        let acquired = reopened
            .try_acquire_task(
                &task_id,
                WorkerId::from_string("boot-worker"),
                LeaseId::from_string("boot-lease"),
                t_plus(120),
                t_plus(4),
            )
            .await?
            .context("boot did not reacquire answered question")?;
        assert!(matches!(acquired.state, TaskState::AnsweredQuestion { .. }));

        drop(reopened);
        let _ = std::fs::remove_file(&db_path);
        Ok(())
    }

    // ── ToolAuditEventStore ──────────────────────────────────────

    fn tool_audit_event_fixture(
        operation_id: &str,
        task: &str,
        thread: &str,
        kind: agent_server::journal::tool_audit::ToolAuditEventKind,
        secs: i64,
    ) -> agent_server::journal::tool_audit::ToolAuditEvent {
        use agent_server::journal::tool_audit::{ToolAuditEvent, ToolAuditEventParams};

        ToolAuditEvent::new(ToolAuditEventParams {
            operation_id: operation_id.into(),
            task_id: AgentTaskId::from_string(format!("task_{task}")),
            parent_task_id: AgentTaskId::from_string(format!("task_{task}_parent")),
            thread_id: ThreadId::from_string(format!("thread_{thread}")),
            tool_call_id: format!("call_{task}"),
            tool_name: "transfer".into(),
            effect_class: ToolEffectClass::SideEffecting,
            kind,
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            input: Some(serde_json::json!({"amount": 100, "api_key": "sk-1234"})),
            output: None,
            error: None,
            now: t_plus(secs),
        })
    }

    #[tokio::test]
    async fn connect_with_max_connections_sizes_the_file_pool() -> Result<()> {
        let tmp = tempfile::NamedTempFile::new()?;
        let url = format!("sqlite:{}?mode=rwc", tmp.path().display());

        let store = SqliteDurableStore::connect_with_max_connections(&url, 16).await?;
        assert_eq!(store.pool().options().get_max_connections(), 16);

        // The compatibility constructor keeps the conservative default.
        let default_store = SqliteDurableStore::connect(&url).await?;
        assert_eq!(
            default_store.pool().options().get_max_connections(),
            SqliteDurableStore::DEFAULT_MAX_CONNECTIONS,
        );
        Ok(())
    }

    #[tokio::test]
    async fn connect_with_max_connections_keeps_memory_single_connection() -> Result<()> {
        // SQLite in-memory databases are per-connection; a wider pool
        // would split the schema across invisible databases.
        let store = SqliteDurableStore::connect_with_max_connections("sqlite::memory:", 16).await?;
        assert_eq!(store.pool().options().get_max_connections(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn connect_with_zero_max_connections_is_rejected() {
        let result = SqliteDurableStore::connect_with_max_connections("sqlite::memory:", 0).await;
        assert!(result.is_err(), "zero-sized pool must be refused");
    }

    #[tokio::test]
    async fn tool_audit_lifecycle_ordering_survives_restart() -> Result<()> {
        use agent_server::journal::tool_audit::{ToolAuditEventKind, ToolAuditEventStore};

        let tmp = tempfile::NamedTempFile::new()?;
        let url = format!("sqlite:{}?mode=rwc", tmp.path().display());

        let store = SqliteDurableStore::connect(&url).await?;

        let operation = "task_lifecycle:call_1";
        let lifecycle = [
            (ToolAuditEventKind::Dispatched, 10),
            (ToolAuditEventKind::ConfirmationRequested, 11),
            (ToolAuditEventKind::ConfirmationApproved, 12),
            (ToolAuditEventKind::ExecutionStarted, 13),
            (ToolAuditEventKind::Completed, 14),
        ];
        for (kind, secs) in lifecycle {
            let event = tool_audit_event_fixture(operation, "life", "life", kind, secs);
            ToolAuditEventStore::record_event(&store, &event).await?;
        }

        drop(store);
        let reopened = SqliteDurableStore::connect(&url).await?;
        let events = ToolAuditEventStore::list_by_operation(&reopened, operation).await?;
        assert_eq!(events.len(), 5);
        let kinds: Vec<&'static str> = events.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "dispatched",
                "confirmation_requested",
                "confirmation_approved",
                "execution_started",
                "completed",
            ],
        );

        Ok(())
    }

    #[tokio::test]
    async fn tool_audit_queries_follow_durable_sequence_when_timestamps_skew() -> Result<()> {
        use agent_server::journal::tool_audit::{ToolAuditEventKind, ToolAuditEventStore};

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let dispatched = tool_audit_event_fixture(
            "task_skew:call_1",
            "skew",
            "skew",
            ToolAuditEventKind::Dispatched,
            20,
        );
        let completed = tool_audit_event_fixture(
            "task_skew:call_1",
            "skew",
            "skew",
            ToolAuditEventKind::Completed,
            10,
        );

        ToolAuditEventStore::record_event(&store, &dispatched).await?;
        ToolAuditEventStore::record_event(&store, &completed).await?;

        let op_kinds: Vec<_> = store
            .list_by_operation(&dispatched.operation_id)
            .await?
            .into_iter()
            .map(|event| event.kind.as_str())
            .collect();
        assert_eq!(op_kinds, vec!["dispatched", "completed"]);

        let task_kinds: Vec<_> = ToolAuditEventStore::list_by_task(&store, &dispatched.task_id)
            .await?
            .into_iter()
            .map(|event| event.kind.as_str())
            .collect();
        assert_eq!(task_kinds, vec!["dispatched", "completed"]);

        let thread_kinds: Vec<_> =
            ToolAuditEventStore::list_by_thread(&store, &dispatched.thread_id)
                .await?
                .into_iter()
                .map(|event| event.kind.as_str())
                .collect();
        assert_eq!(thread_kinds, vec!["dispatched", "completed"]);

        Ok(())
    }

    #[tokio::test]
    async fn tool_audit_provenance_persists_across_query_paths() -> Result<()> {
        use agent_server::journal::tool_audit::{ToolAuditEventKind, ToolAuditEventStore};

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let event = tool_audit_event_fixture(
            "task_prov:call_1",
            "prov",
            "prov",
            ToolAuditEventKind::Dispatched,
            20,
        );
        ToolAuditEventStore::record_event(&store, &event).await?;

        let op_events = store.list_by_operation(&event.operation_id).await?;
        assert_eq!(op_events.len(), 1);
        let loaded = &op_events[0];
        assert_eq!(loaded.id, event.id);
        assert_eq!(loaded.provider, "anthropic");
        assert_eq!(loaded.model, "claude-sonnet-4-5-20250929");
        assert_eq!(loaded.parent_task_id, event.parent_task_id);

        let by_task = ToolAuditEventStore::list_by_task(&store, &event.task_id).await?;
        assert_eq!(by_task.len(), 1);
        assert_eq!(by_task[0].id, event.id);

        let by_thread = ToolAuditEventStore::list_by_thread(&store, &event.thread_id).await?;
        assert_eq!(by_thread.len(), 1);
        assert_eq!(by_thread[0].id, event.id);

        Ok(())
    }

    #[tokio::test]
    async fn tool_audit_query_isolates_by_task_and_thread() -> Result<()> {
        use agent_server::journal::tool_audit::{ToolAuditEventKind, ToolAuditEventStore};

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let e_a = tool_audit_event_fixture(
            "task_a:call_1",
            "a",
            "shared",
            ToolAuditEventKind::Dispatched,
            30,
        );
        let e_b = tool_audit_event_fixture(
            "task_b:call_1",
            "b",
            "shared",
            ToolAuditEventKind::Dispatched,
            31,
        );
        let e_other = tool_audit_event_fixture(
            "task_c:call_1",
            "c",
            "elsewhere",
            ToolAuditEventKind::Dispatched,
            32,
        );
        ToolAuditEventStore::record_event(&store, &e_a).await?;
        ToolAuditEventStore::record_event(&store, &e_b).await?;
        ToolAuditEventStore::record_event(&store, &e_other).await?;

        let by_task = ToolAuditEventStore::list_by_task(&store, &e_a.task_id).await?;
        assert_eq!(by_task.len(), 1);

        let by_thread = ToolAuditEventStore::list_by_thread(&store, &e_a.thread_id).await?;
        assert_eq!(by_thread.len(), 2, "two events on shared thread");

        let elsewhere = ToolAuditEventStore::list_by_thread(&store, &e_other.thread_id).await?;
        assert_eq!(elsewhere.len(), 1);

        let unknown = store.list_by_operation("task_none:call_none").await?;
        assert!(unknown.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn tool_audit_redaction_decorator_scrubs_before_durable_write() -> Result<()> {
        use agent_server::journal::redaction::REDACTED_MARKER;
        use agent_server::journal::tool_audit::{
            RedactingToolAuditEventStore, ToolAuditEvent, ToolAuditEventKind, ToolAuditEventParams,
            ToolAuditEventStore,
        };
        use std::sync::Arc;

        let store = Arc::new(SqliteDurableStore::connect("sqlite::memory:").await?);
        let decorator = RedactingToolAuditEventStore::baseline(store.clone());

        let params = ToolAuditEventParams {
            operation_id: "task_redact:call_1".into(),
            task_id: AgentTaskId::from_string("task_redact"),
            parent_task_id: AgentTaskId::from_string("task_redact_parent"),
            thread_id: ThreadId::from_string("thread_redact"),
            tool_call_id: "call_1".into(),
            tool_name: "transfer".into(),
            effect_class: ToolEffectClass::SideEffecting,
            kind: ToolAuditEventKind::Failed {
                error: "Bearer provider-secret".into(),
            },
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            input: Some(serde_json::json!({
                "command": "transfer",
                "api_key": "sk-secret",
                "normal": "ok",
            })),
            output: Some("sk-output-token".into()),
            error: Some("sk-top-level-error".into()),
            now: t_plus(40),
        };
        let event = ToolAuditEvent::new(params);
        decorator.record_event(&event).await?;

        let stored = store.list_by_operation(&event.operation_id).await?;
        assert_eq!(stored.len(), 1);
        let stored = &stored[0];
        let input = stored.input.as_ref().context("input present")?;
        assert_eq!(input["api_key"], REDACTED_MARKER);
        assert_eq!(input["normal"], "ok");
        assert_eq!(stored.output.as_deref(), Some(REDACTED_MARKER));
        assert_eq!(stored.error.as_deref(), Some(REDACTED_MARKER));
        match &stored.kind {
            ToolAuditEventKind::Failed { error } => assert_eq!(error, REDACTED_MARKER),
            other => anyhow::bail!("expected Failed audit kind, got {other:?}"),
        }

        Ok(())
    }

    // ── Phase 8.1: outbox advisory contract ─────────────────────────

    #[tokio::test]
    async fn an_event_batch_gets_one_thread_events_advisory() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-coalesce");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![
                    AgentEvent::text("msg_a", "first"),
                    AgentEvent::text("msg_b", "second"),
                    AgentEvent::text("msg_c", "third"),
                ],
                outbox_max_attempts: 5,
                now: t0(),
            })
            .await?;

        assert_eq!(outcome.committed_events.len(), 3);
        let row = outcome.outbox_row;
        assert_eq!(row.kind, OutboxMessageKind::ThreadEventsAvailable);
        assert_eq!(row.thread_id, thread_id);
        // The row columns reference the FIRST event of the batch
        // (retention safety bound); the payload carries the LAST.
        assert_eq!(row.sequence, Some(0));
        assert_eq!(row.event_id, Some(outcome.committed_events[0].event_id));

        let message = OutboxMessage::from_payload_json(row.kind, row.payload_json.clone())?;
        assert_eq!(
            message,
            OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                thread_id: thread_id.clone(),
                last_sequence: 2,
            }),
        );

        let rows = OutboxStore::list_by_thread(&store, &thread_id).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, OutboxMessageKind::ThreadEventsAvailable);

        Ok(())
    }

    #[tokio::test]
    async fn outbox_insert_batch_accepts_task_wakeup_with_null_event_refs() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::outbox::{NewOutboxRow, OutboxStore};
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, TaskWakeupPayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-task-wakeup");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let task_id = AgentTaskId::new();
        let payload = OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id: task_id.clone(),
            thread_id: thread_id.clone(),
        })
        .to_payload_json()?;

        let rows = OutboxStore::insert_batch(
            &store,
            vec![NewOutboxRow {
                kind: OutboxMessageKind::TaskWakeup,
                thread_id: thread_id.clone(),
                event_id: None,
                sequence: None,
                payload_json: payload.clone(),
                max_attempts: 3,
                now: t0(),
            }],
        )
        .await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, OutboxMessageKind::TaskWakeup);
        assert!(rows[0].event_id.is_none());
        assert!(rows[0].sequence.is_none());

        let stored = OutboxStore::get(&store, &rows[0].id)
            .await?
            .context("row missing after insert")?;
        assert_eq!(stored.kind, OutboxMessageKind::TaskWakeup);
        assert!(stored.event_id.is_none());
        assert!(stored.sequence.is_none());
        assert_eq!(stored.payload_json, payload);

        Ok(())
    }

    #[tokio::test]
    async fn outbox_insert_batch_rejects_thread_events_without_sequence() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_server::journal::outbox::{NewOutboxRow, OutboxStore};
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-thread-events-bad");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let payload = OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_id.clone(),
            last_sequence: 0,
        })
        .to_payload_json()?;

        let result = OutboxStore::insert_batch(
            &store,
            vec![NewOutboxRow {
                kind: OutboxMessageKind::ThreadEventsAvailable,
                thread_id,
                event_id: Some(uuid::Uuid::now_v7()),
                // Phase 8.1 contract: thread_events_available rows
                // MUST carry both event_id AND sequence references.
                sequence: None,
                payload_json: payload,
                max_attempts: 3,
                now: t0(),
            }],
        )
        .await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn outbox_mark_delivered_is_noop_when_row_was_cascade_deleted() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-delivered-missing");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            })
            .await?;
        let row_id = outcome.outbox_row.id;

        RetentionStore::advance_floor(&store, &thread_id, 1, t_plus(1)).await?;
        assert!(OutboxStore::get(&store, &row_id).await?.is_none());

        OutboxStore::mark_delivered(&store, &row_id, "relay-worker", t_plus(2)).await?;
        Ok(())
    }

    #[tokio::test]
    async fn outbox_mark_failed_is_noop_when_row_was_cascade_deleted() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-failed-missing");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            })
            .await?;
        let row_id = outcome.outbox_row.id;

        RetentionStore::advance_floor(&store, &thread_id, 1, t_plus(1)).await?;
        assert!(OutboxStore::get(&store, &row_id).await?.is_none());

        OutboxStore::mark_failed(
            &store,
            &row_id,
            "relay-worker",
            "timeout",
            t_plus(3),
            t_plus(2),
        )
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn event_sequence_does_not_regress_after_full_retention_purge() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-seq-no-regress");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        // Commit 3 events — sequences 0, 1, 2.
        for i in 0..3 {
            EventRepository::commit_event(
                &store,
                &thread_id,
                AgentEvent::text(format!("m{i}"), "hello"),
                t_plus(i),
            )
            .await?;
        }

        // Janitor purges every event by advancing the floor past the
        // last assigned sequence.
        RetentionStore::advance_floor(&store, &thread_id, 3, t_plus(3)).await?;
        assert_eq!(
            RetentionStore::effective_floor(&store, &thread_id).await?,
            3,
        );

        // A new event must land at sequence >= 3, not reuse 0.
        let fresh = EventRepository::commit_event(
            &store,
            &thread_id,
            AgentEvent::text("m3", "post-purge"),
            t_plus(4),
        )
        .await?;
        assert!(
            fresh.sequence >= 3,
            "sequence regressed after purge: {} (expected >= 3)",
            fresh.sequence,
        );
        Ok(())
    }

    #[tokio::test]
    async fn outbox_list_by_thread_orders_task_wakeups_after_thread_events() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::{NewOutboxRow, OutboxStore};
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, TaskWakeupPayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-order");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            })
            .await?;

        let task_id = AgentTaskId::new();
        let payload = OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id,
            thread_id: thread_id.clone(),
        })
        .to_payload_json()?;
        OutboxStore::insert_batch(
            &store,
            vec![NewOutboxRow {
                kind: OutboxMessageKind::TaskWakeup,
                thread_id: thread_id.clone(),
                event_id: None,
                sequence: None,
                payload_json: payload,
                max_attempts: 3,
                now: t_plus(1),
            }],
        )
        .await?;

        let rows = OutboxStore::list_by_thread(&store, &thread_id).await?;
        let kinds: Vec<OutboxMessageKind> = rows.iter().map(|row| row.kind).collect();
        assert_eq!(
            kinds,
            vec![
                OutboxMessageKind::ThreadEventsAvailable,
                OutboxMessageKind::TaskWakeup,
            ]
        );

        Ok(())
    }

    // ── Phase 8.2: reclaim for crash recovery ────────────────────────

    #[tokio::test]
    async fn outbox_reclaim_returns_stale_claimed_rows_to_pending() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::{OutboxStatus, OutboxStore};
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-reclaim-stale");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            })
            .await?;

        let claimed = OutboxStore::claim_pending(&store, "worker-a", 10, t_plus(1)).await?;
        assert_eq!(claimed.len(), 1);
        let id = claimed[0].id.clone();

        let reclaimed =
            OutboxStore::reclaim_expired_claims(&store, t_plus(30), time::Duration::seconds(10))
                .await?;
        assert_eq!(reclaimed, 1);

        let row = OutboxStore::get(&store, &id)
            .await?
            .context("row should still exist")?;
        assert_eq!(row.status, OutboxStatus::Pending);
        assert!(row.claimed_by.is_none());
        assert!(row.claimed_at.is_none());
        assert_eq!(row.attempt_count, 0);

        let reclaimed_rows = OutboxStore::claim_pending(&store, "worker-b", 10, t_plus(31)).await?;
        assert_eq!(reclaimed_rows.len(), 1);
        assert_eq!(reclaimed_rows[0].id, id);

        Ok(())
    }

    #[tokio::test]
    async fn outbox_reclaim_skips_live_claims() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-reclaim-live");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        store
            .seed_events_with_outbox(SeedEvents {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_live", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            })
            .await?;

        OutboxStore::claim_pending(&store, "worker-live", 10, t_plus(1)).await?;

        let reclaimed =
            OutboxStore::reclaim_expired_claims(&store, t_plus(6), time::Duration::seconds(30))
                .await?;
        assert_eq!(reclaimed, 0);

        Ok(())
    }

    /// `SQLite` end-to-end of the lost-history fix:
    ///
    /// `set_draft` round-trips through the new `draft_messages_json`
    /// column, persists across reconnects (the migration applies
    /// cleanly to a fresh DB), and the atomic completed-turn
    /// transaction wipes the slot atomically with the message
    /// commit so a recovery between the two never sees committed +
    /// draft duplicated.
    #[tokio::test]
    async fn draft_messages_persist_until_atomic_commit_clears_them() -> Result<()> {
        use agent_sdk_foundation::llm;
        use agent_sdk_foundation::{TokenUsage, audit::AuditProvenance};
        use agent_server::journal::checkpoint::CheckpointKind;
        use agent_server::journal::commit::CompletedTurnCommit;
        use agent_server::journal::message_store::MessageProjectionStore;
        use agent_server::journal::task::AgentTask;
        use agent_server::journal::turn_attempt::{
            CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome,
        };
        use agent_server::journal::turn_attempt_store::TurnAttemptStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-draft");

        // 1. Suspension snapshot — the worker writes the
        //    accumulated `suspended_messages` to the projection draft
        //    after every tool boundary.
        let suspended = vec![
            llm::Message::user("user prompt"),
            llm::Message::assistant("calling tool"),
        ];
        let projection_after_suspend =
            MessageProjectionStore::set_draft(&store, &thread_id, suspended.clone(), t_plus(1))
                .await?;
        assert!(projection_after_suspend.has_draft());
        assert_eq!(projection_after_suspend.draft_messages.len(), 2);

        // 2. Reload from disk to prove the new column round-trips.
        let reloaded = MessageProjectionStore::get(&store, &thread_id)
            .await?
            .context("projection persisted")?;
        assert!(reloaded.has_draft());
        assert_eq!(reloaded.draft_messages.len(), 2);

        // 3. Submit a real root task — `agent_sdk_message_commits`
        //    has a FK to `agent_sdk_tasks` so the commit transaction
        //    needs an existing task row to reference.
        let task = AgentTask::new_root_turn(thread_id.clone(), t_plus(2), 3);
        let task_id = task.id.clone();
        AgentTaskStore::submit_root_turn(&store, task).await?;

        // 4. Open an attempt for the upcoming commit. The
        //    completed-turn transaction validates the attempt id, so
        //    we need a real one.
        let attempt = TurnAttemptStore::open_attempt(
            &store,
            OpenAttemptParams {
                task_id: task_id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(2),
                otel_trace_id: None,
                otel_span_id: None,
            },
        )
        .await?;

        // 5. Atomically commit the turn. The transaction must clear
        //    the draft as the same write that appends committed
        //    history, so a crash-and-reload between the two would be
        //    impossible.
        let final_messages = vec![
            llm::Message::user("user prompt"),
            llm::Message::assistant("final reply"),
        ];
        let commit = AgentTaskStore::commit_completed_turn(
            &store,
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                delivered_injection_ids: Vec::new(),
                thread_id: thread_id.clone(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: CloseAttemptParams {
                    response_blob: serde_json::json!({"id": "msg_01"}),
                    response_id: Some("msg_01".into()),
                    response_model: Some("claude-sonnet-4-5-20250929".into()),
                    stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 10,
                    output_tokens: 20,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    route_provider: None,
                    thinking_mode: None,
                    thinking_budget_tokens: None,
                    thinking_effort: None,
                },
                messages: final_messages,
                turn_usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                    ..Default::default()
                },
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: Vec::new(),
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(3),
            },
        )
        .await?;
        assert_eq!(commit.checkpoint.turn_number, 1);

        // 6. Final state: committed history present, draft cleared.
        let projection_after_commit = MessageProjectionStore::get(&store, &thread_id)
            .await?
            .context("projection persisted")?;
        assert!(
            !projection_after_commit.has_draft(),
            "atomic commit must wipe the draft as part of the same transaction",
        );
        assert_eq!(
            projection_after_commit.message_count(),
            2,
            "committed history reflects the turn's final messages",
        );

        Ok(())
    }

    fn atomic_commit_close_params() -> agent_server::journal::turn_attempt::CloseAttemptParams {
        agent_server::journal::turn_attempt::CloseAttemptParams {
            response_blob: serde_json::json!({"id": "msg_01"}),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
            outcome: agent_server::journal::turn_attempt::TurnAttemptOutcome::Success,
            input_tokens: 10,
            output_tokens: 20,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_mode: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
        }
    }

    /// Phase 10 · D regression: on `SQLite` a committed turn's lifecycle
    /// events and the coalesced advisory outbox row land in the SAME
    /// transaction as the state projections. "Events exist iff the turn
    /// committed": a successful commit yields both the turn AND its
    /// events; an empty-events commit yields neither events nor an
    /// advisory row.
    #[tokio::test]
    async fn events_and_outbox_commit_atomically_with_turn_sqlite() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_sdk_foundation::llm;
        use agent_sdk_foundation::{TokenUsage, audit::AuditProvenance};
        use agent_server::journal::checkpoint::CheckpointKind;
        use agent_server::journal::commit::CompletedTurnCommit;
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::OutboxMessageKind;
        use agent_server::journal::task::{AgentTask, TaskStatus};
        use agent_server::journal::thread_store::ThreadStore;
        use agent_server::journal::turn_attempt::OpenAttemptParams;
        use agent_server::journal::turn_attempt_store::TurnAttemptStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-events-iff");

        let task = AgentTask::new_root_turn(thread_id.clone(), t_plus(1), 3);
        let task_id = task.id.clone();
        let admitted = AgentTaskStore::submit_root_turn(&store, task).await?;

        // The runnable admission must have emitted a durable task_wakeup
        // advisory row in the same transaction.
        assert_eq!(admitted.status, TaskStatus::Pending);
        let rows = OutboxStore::list_by_thread(&store, &thread_id).await?;
        let wakeup_rows = rows
            .iter()
            .filter(|r| r.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(
            wakeup_rows, 1,
            "submit_root_turn must emit exactly one durable task_wakeup row"
        );

        let attempt = TurnAttemptStore::open_attempt(
            &store,
            OpenAttemptParams {
                task_id: task_id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(2),
                otel_trace_id: None,
                otel_span_id: None,
            },
        )
        .await?;

        let outcome = AgentTaskStore::commit_completed_turn(
            &store,
            CompletedTurnCommit {
                extra_writes: Vec::new(),
                checkpoint_kind: CheckpointKind::FullTurn,
                delivered_injection_ids: Vec::new(),
                thread_id: thread_id.clone(),
                task_id,
                expected_turn: 1,
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: atomic_commit_close_params(),
                messages: vec![llm::Message::user("hi"), llm::Message::assistant("there")],
                turn_usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                    ..Default::default()
                },
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: vec![AgentEvent::start(thread_id.clone(), 1)],
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(3),
            },
        )
        .await?;

        // Turn committed → events exist, returned and persisted.
        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.committed_events.len(), 1);
        let events = EventRepository::get_events(&store, &thread_id).await?;
        assert_eq!(
            events.len(),
            1,
            "a committed turn must carry its persisted events"
        );

        // The coalesced advisory outbox row landed in the SAME tx.
        let rows = OutboxStore::list_by_thread(&store, &thread_id).await?;
        let thread_events_rows = rows
            .iter()
            .filter(|r| r.kind == OutboxMessageKind::ThreadEventsAvailable)
            .count();
        assert_eq!(
            thread_events_rows, 1,
            "exactly one thread_events_available advisory row per committed turn"
        );

        // Confirm the thread aggregate also advanced.
        let thread = ThreadStore::get(&store, &thread_id)
            .await?
            .context("thread row")?;
        assert_eq!(thread.committed_turns, 1);

        Ok(())
    }

    /// Phase 10 · D: the standalone durable `emit_in_transaction` hook
    /// persists a `task_wakeup` advisory row that a consumer can decode
    /// back into its `{task_id, thread_id}` payload.
    #[tokio::test]
    async fn task_wakeup_emit_in_transaction_persists_durable_row_sqlite() -> Result<()> {
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::{OutboxMessage, OutboxMessageKind};
        use agent_server::journal::relay::{TaskWakeupEmitter, TaskWakeupTrigger};
        use agent_server::journal::task::AgentTaskId;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-wakeup-emit");
        // The outbox row carries a FK to agent_sdk_threads; the thread is
        // always bootstrapped before a task on it becomes runnable.
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;
        let task_id = AgentTaskId::new();

        let id = TaskWakeupEmitter::emit_in_transaction(
            &store,
            TaskWakeupTrigger {
                task_id: task_id.clone(),
                thread_id: thread_id.clone(),
                now: t0(),
                max_attempts: 3,
            },
        )
        .await?;

        let row = OutboxStore::get(&store, &id).await?.context("wakeup row")?;
        assert_eq!(row.kind, OutboxMessageKind::TaskWakeup);
        assert_eq!(row.thread_id, thread_id);
        assert!(row.event_id.is_none());
        assert!(row.sequence.is_none());
        let payload = OutboxMessage::from_payload_json(row.kind, row.payload_json)?;
        let OutboxMessage::TaskWakeup(p) = payload else {
            anyhow::bail!("expected task_wakeup payload, got {payload:?}");
        };
        assert_eq!(p.task_id, task_id);
        assert_eq!(p.thread_id, thread_id);
        Ok(())
    }

    /// Phase 10 · D + E regression: the transport-facing
    /// `submit_root_turn_idempotent` path on `SQLite` must also emit the
    /// durable `task_wakeup` advisory row inside the admission
    /// transaction when the root lands runnable (`Pending`), exactly like
    /// the non-idempotent `submit_root_turn`. A queued (parked) root must
    /// not emit one.
    #[tokio::test]
    async fn idempotent_submit_emits_durable_task_wakeup_when_runnable_sqlite() -> Result<()> {
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::OutboxMessageKind;
        use agent_server::journal::store::{
            AgentTaskStore, SubmitRootIdempotency, SubmitRootTurnParams,
        };
        use agent_server::journal::task::{AgentTask, TaskStatus};

        fn params(task: AgentTask, request_id: &str) -> SubmitRootTurnParams {
            let result_json = serde_json::json!({ "task_id": task.id.to_string() });
            SubmitRootTurnParams {
                task,
                idempotency: Some(SubmitRootIdempotency {
                    request_id: request_id.to_owned(),
                    fingerprint: b"fp".to_vec(),
                    result_json,
                }),
                max_queued_depth: None,
                disposition: agent_server::journal::store::SubmitDisposition::NextTurn,
            }
        }

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-idem-wakeup");

        // First root lands Pending → exactly one durable task_wakeup row.
        let first = AgentTask::new_root_turn(thread_id.clone(), t_plus(1), 3);
        let outcome = store
            .submit_root_turn_idempotent(params(first, "w-1"))
            .await
            .map_err(|error| anyhow::anyhow!("first submit: {error}"))?;
        assert_eq!(outcome.task.status, TaskStatus::Pending);

        let wakeups_after_first = OutboxStore::list_by_thread(&store, &thread_id)
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(
            wakeups_after_first, 1,
            "idempotent admission of a runnable root must emit exactly one durable task_wakeup"
        );

        // Second root lands Queued (parked) → no additional wakeup.
        let queued = AgentTask::new_root_turn(thread_id.clone(), t_plus(2), 3);
        let queued_outcome = store
            .submit_root_turn_idempotent(params(queued, "w-2"))
            .await
            .map_err(|error| anyhow::anyhow!("queued submit: {error}"))?;
        assert_eq!(queued_outcome.task.status, TaskStatus::Queued);

        let wakeups_after_queued = OutboxStore::list_by_thread(&store, &thread_id)
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(
            wakeups_after_queued, 1,
            "a queued (parked) root must not emit an extra task_wakeup"
        );

        Ok(())
    }

    /// `ThreadStore::commit_fork` on `SQLite` must commit thread aggregate
    /// + projection + checkpoint + events as one transaction. This
    /// covers the happy path: every store reflects the committed
    /// fork state after a single `commit_fork_atomic` call.
    #[tokio::test]
    async fn fork_atomic_commits_full_state_in_one_transaction() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_sdk_foundation::llm::Message;
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        use agent_server::journal::checkpoint_store::CheckpointStore;
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::fork_transaction::ForkCommitParams;
        use agent_server::journal::message_store::MessageProjectionStore;
        use agent_server::journal::store::AgentTaskStore;
        use agent_server::journal::task::AgentTask;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let now = t0();

        // The source thread + task have to actually exist on disk so
        // the fork's mirrored checkpoint can carry the source's
        // task_id without violating the
        // agent_sdk_message_commits.task_id FK.
        let source_thread_id = ThreadId::from_string("source-thread");
        ThreadStore::get_or_create(&store, &source_thread_id, now).await?;
        let source_task = AgentTask::new_root_turn(source_thread_id.clone(), now, 3);
        let source_task_id = source_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, source_task).await?;

        let new_thread_id = ThreadId::from_string("forked-thread-id");
        let messages = vec![
            Message::user("hi from source"),
            Message::assistant("hello from source"),
        ];
        // Per the ForkCommitParams contract, events[0] is the
        // destination's ThreadCreated — built by the caller, committed
        // verbatim by the store.
        let events = vec![
            AgentEvent::thread_created(
                new_thread_id.clone(),
                Some(source_thread_id.clone()),
                Some(1),
            ),
            AgentEvent::start(source_thread_id.clone(), 1),
            AgentEvent::done(
                source_thread_id.clone(),
                1,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            ),
        ];
        let params = ForkCommitParams {
            source_thread_id: source_thread_id.clone(),
            new_thread_id: new_thread_id.clone(),
            creation: None,
            now,
            committed_turns: 1,
            cumulative_total_usage: agent_sdk_foundation::TokenUsage::default(),
            messages: messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: source_task_id,
                messages: messages.clone(),
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 1,
                    "total_usage": agent_sdk_foundation::TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                now,
            }),
            events: events.clone(),
            outbox_max_attempts: 3,
        };

        store.commit_fork(params).await?;

        let thread = ThreadStore::get(&store, &new_thread_id)
            .await?
            .context("forked thread aggregate must exist after commit")?;
        assert_eq!(thread.committed_turns, 1);

        let projection = MessageProjectionStore::get_history(&store, &new_thread_id).await?;
        assert_eq!(projection.len(), 2);

        let checkpoint = CheckpointStore::get_by_turn(&store, &new_thread_id, 1)
            .await?
            .context("forked checkpoint must exist")?;
        assert_eq!(checkpoint.messages.len(), 2);

        let committed = EventRepository::get_events(&store, &new_thread_id).await?;
        assert_eq!(committed.len(), 3);
        let sequences: Vec<_> = committed.iter().map(|c| c.sequence).collect();
        assert_eq!(
            sequences,
            vec![0, 1, 2],
            "copied events and creation event committed with fresh sequences"
        );

        Ok(())
    }

    /// `ForkCommitParams::cumulative_total_usage` must land on the
    /// destination's thread aggregate `total_usage` so cost reporting
    /// on the fork picks up where the source left off.  The earlier
    /// per-turn iterations contribute zero usage; only the final
    /// `apply_committed_turn` carries the carryover, which means the
    /// fork's running total ends at exactly `cumulative_total_usage`.
    #[tokio::test]
    async fn fork_atomic_carries_cumulative_total_usage() -> Result<()> {
        use agent_sdk_foundation::TokenUsage;
        use agent_sdk_foundation::llm::Message;
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        use agent_server::journal::fork_transaction::ForkCommitParams;
        use agent_server::journal::store::AgentTaskStore;
        use agent_server::journal::task::AgentTask;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let now = t0();

        let source_thread_id = ThreadId::from_string("usage-source");
        ThreadStore::get_or_create(&store, &source_thread_id, now).await?;
        let source_task = AgentTask::new_root_turn(source_thread_id.clone(), now, 3);
        let source_task_id = source_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, source_task).await?;

        let new_thread_id = ThreadId::from_string("usage-fork");
        let cumulative = TokenUsage {
            input_tokens: 1234,
            output_tokens: 567,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let messages = vec![Message::user("u"), Message::assistant("a")];
        let params = ForkCommitParams {
            source_thread_id: source_thread_id.clone(),
            new_thread_id: new_thread_id.clone(),
            creation: None,
            now,
            committed_turns: 3,
            cumulative_total_usage: cumulative.clone(),
            messages: messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 3,
                task_id: source_task_id,
                messages: messages.clone(),
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 3,
                    "total_usage": cumulative,
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: TokenUsage::default(),
                now,
            }),
            events: vec![agent_sdk_foundation::events::AgentEvent::thread_created(
                new_thread_id.clone(),
                Some(source_thread_id.clone()),
                Some(3),
            )],
            outbox_max_attempts: 3,
        };
        store.commit_fork(params).await?;

        let thread = ThreadStore::get(&store, &new_thread_id)
            .await?
            .context("fork must exist after commit")?;
        assert_eq!(thread.committed_turns, 3);
        assert_eq!(
            thread.total_usage, cumulative,
            "fork's total_usage must mirror the source's cumulative usage at the fork boundary",
        );

        Ok(())
    }

    /// A failing fork must leave zero observable state on the
    /// destination thread. We trigger a failure by passing a
    /// checkpoint whose `task_id` collides with an already-inserted
    /// row in another thread's table; the unique constraint on
    /// `(thread_id, turn_number)` will reject the second commit
    /// inside the transaction, the rollback fires, and the
    /// destination must come up empty.
    #[tokio::test]
    async fn fork_atomic_rolls_back_on_failure() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_sdk_foundation::llm::Message;
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::fork_transaction::ForkCommitParams;
        use agent_server::journal::message_store::MessageProjectionStore;
        use agent_server::journal::store::AgentTaskStore;
        use agent_server::journal::task::AgentTask;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let new_thread_id = ThreadId::from_string("forked-rollback");
        let now = t0();

        // Both the pre-existing checkpoint and the fork checkpoint
        // need real task rows behind them — the
        // agent_sdk_message_commits → agent_sdk_tasks FK doesn't care
        // which thread the task lives on, just that it exists.
        let pre_thread_id = ThreadId::from_string("pre-thread");
        ThreadStore::get_or_create(&store, &pre_thread_id, now).await?;
        let pre_task = AgentTask::new_root_turn(pre_thread_id.clone(), now, 3);
        let pre_task_id = pre_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, pre_task).await?;

        let source_thread_id = ThreadId::from_string("source-thread");
        ThreadStore::get_or_create(&store, &source_thread_id, now).await?;
        let source_task = AgentTask::new_root_turn(source_thread_id.clone(), now, 3);
        let source_task_id = source_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, source_task).await?;

        // Pre-insert a checkpoint at turn_number = 1 on the
        // destination thread id. The fork's checkpoint insert hits
        // the `(thread_id, turn_number)` unique constraint and rolls
        // back the entire transaction.
        ThreadStore::get_or_create(&store, &new_thread_id, now).await?;
        agent_server::journal::checkpoint_store::CheckpointStore::commit_checkpoint(
            &store,
            NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: pre_task_id,
                messages: vec![],
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 0,
                    "total_usage": agent_sdk_foundation::TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                now,
            },
        )
        .await?;

        // Now run the fork. Its checkpoint insert at turn 1 will
        // collide and the transaction rolls back.
        let fresh_messages = vec![Message::user("seeded")];
        let params = ForkCommitParams {
            source_thread_id: source_thread_id.clone(),
            new_thread_id: new_thread_id.clone(),
            creation: None,
            now,
            committed_turns: 1,
            cumulative_total_usage: agent_sdk_foundation::TokenUsage::default(),
            messages: fresh_messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: source_task_id,
                messages: fresh_messages.clone(),
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 1,
                    "total_usage": agent_sdk_foundation::TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                now,
            }),
            events: vec![AgentEvent::start(source_thread_id.clone(), 1)],
            outbox_max_attempts: 3,
        };
        let err = store
            .commit_fork(params)
            .await
            .expect_err("checkpoint collision must fail the fork transaction");
        let _ = err; // payload-shape doesn't matter; rollback is what we're checking.

        // Crucially, the rolled-back transaction must NOT leave the
        // fresh projection/event mutations behind. The pre-existing
        // checkpoint is still there (that wasn't part of the failed
        // tx), but the destination's projection must still be empty
        // and the event log must have no entries.
        let projection = MessageProjectionStore::get_history(&store, &new_thread_id).await?;
        assert!(
            projection.is_empty(),
            "rolled-back fork must not leave seeded messages on the destination"
        );
        let committed = EventRepository::get_events(&store, &new_thread_id).await?;
        assert!(
            committed.is_empty(),
            "rolled-back fork must not leave events on the destination"
        );

        Ok(())
    }

    /// ADR-0003 I5 — the stall probe's child read must be BOUNDED, and its SQL
    /// must agree with the Rust reference semantics.
    ///
    /// The probe used to `list_children` and filter terminal rows in memory.
    /// Terminal task rows are RETAINED, so a parent that has run thousands of
    /// tool calls materialized thousands of rows on every probe of every parked
    /// ancestor — until the probe could no longer finish inside its timeout,
    /// and a timed-out probe is treated as ALIVE. That is the inverse failure:
    /// unbounded database work AND a wedged child that can never be reaped.
    ///
    /// This pins all three properties the bounded query has to hold:
    ///   * the live frontier is capped and contains no terminal rows;
    ///   * a TERMINAL child's freshness is still seen (it is what keeps a
    ///     parked parent alive the moment a long tool finishes);
    ///   * the SQL agrees with [`ChildProbe::from_rows`], including the I1
    ///     `COALESCE(last_activity_at, created_at)` fallback and the I2
    ///     `julianday` (NOT lexical) comparison.
    #[tokio::test]
    async fn probe_children_is_bounded_and_matches_the_reference_semantics() -> Result<()> {
        use agent_server::journal::store::ChildProbe;
        use agent_server::journal::task::{AgentTask, LeaseId, TaskKind, WorkerId};
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-probe-children");
        let base = t0();
        ThreadStore::get_or_create(&store, &thread_id, base).await?;

        let parent = AgentTask::new_root_turn(thread_id.clone(), base, 3);
        let parent_id = parent.id.clone();
        AgentTaskStore::submit_root_turn(&store, parent.clone()).await?;
        let parent = AgentTaskStore::get(&store, &parent_id)
            .await?
            .context("parent")?;

        // 40 RETAINED TERMINAL children, all long finished.
        for _ in 0..40 {
            let child = AgentTask::new_child(&parent, TaskKind::ToolRuntime, base, 3)?;
            let id = child.id.clone();
            AgentTaskStore::insert(&store, child).await?;
            let claimed = AgentTaskStore::get(&store, &id).await?.context("child")?;
            let done = claimed
                .mark_running(
                    WorkerId::from_string("w"),
                    LeaseId::new(),
                    base + Duration::seconds(60),
                    base,
                )?
                .complete(base + Duration::seconds(1))?;
            AgentTaskStore::update(&store, done).await?;
        }
        // 5 LIVE children, never acquired (so `last_activity_at` is NULL —
        // the I1 fallback must read their `created_at`).
        for _ in 0..5 {
            let child = AgentTask::new_child(&parent, TaskKind::ToolRuntime, base, 3)?;
            AgentTaskStore::insert(&store, child).await?;
        }

        // Cutoff AFTER everything: nothing is fresh, and the live frontier is
        // capped below the live count.
        let cutoff = base + Duration::seconds(600);
        let probe = AgentTaskStore::probe_children(&store, &parent_id, cutoff, 3).await?;
        assert_eq!(probe.live.len(), 3, "the live frontier must honour the cap");
        assert!(
            probe.live.iter().all(|t| !t.status.is_terminal()),
            "retained TERMINAL rows must never enter the frontier",
        );
        assert!(
            !probe.fresh_activity,
            "with every row older than the cutoff, nothing is fresh",
        );

        // Cutoff BEFORE the terminal children settled: a just-finished child is
        // a fresh sign of life for its parked parent, even though it is
        // terminal and therefore never returned in `live`.
        let probe = AgentTaskStore::probe_children(&store, &parent_id, base, 3).await?;
        assert!(
            probe.fresh_activity,
            "a TERMINAL child that settled inside the window must still count as life",
        );

        // The SQL must agree with the Rust reference on the SAME rows —
        // including the `COALESCE(last_activity_at, created_at)` fallback for
        // the never-acquired live children and the chronological comparison.
        for cutoff in [
            base,
            base + Duration::seconds(1),
            base + Duration::seconds(600),
        ] {
            let all = AgentTaskStore::list_children(&store, &parent_id).await?;
            let reference = ChildProbe::from_rows(all, cutoff, 3);
            let actual = AgentTaskStore::probe_children(&store, &parent_id, cutoff, 3).await?;
            assert_eq!(
                actual.fresh_activity, reference.fresh_activity,
                "SQL and the Rust reference must agree on freshness at cutoff {cutoff}",
            );
            assert_eq!(
                actual.live.len(),
                reference.live.len(),
                "SQL and the Rust reference must agree on the live frontier at cutoff {cutoff}",
            );
        }

        Ok(())
    }

    /// ADR-0003 I2 — the activity comparison must be CHRONOLOGICAL, never
    /// lexical.
    ///
    /// sqlx binds `OffsetDateTime` as RFC3339 TEXT (`format(&Rfc3339)`) and
    /// `time` trims trailing zeros, so the stored fractional precision is
    /// VARIABLE. The scalar `MAX()` compares that text with the BINARY
    /// collation, and `'Z'` (0x5A) sorts after `'.'` (0x2E) — so a whole-second
    /// stamp beats a sub-second one inside the same second, and any fraction
    /// that is a lexical PREFIX of a later one beats the later one:
    ///
    /// ```text
    /// MAX('..T10:00:00Z',   '..T10:00:00.5Z')  = '..:00Z'    (the OLDER one!)
    /// MAX('..T10:00:00.1Z', '..T10:00:00.12Z') = '..:00.1Z'  (the OLDER one!)
    /// ```
    ///
    /// A heartbeat carrying genuinely newer work would then be silently
    /// dropped, freezing `last_activity_at` — and frozen activity is
    /// indistinguishable from silence, so a working child gets reaped.
    ///
    /// Every pair below is one where the LEXICALLY larger string is the
    /// chronologically OLDER instant. `julianday()` restores real order.
    #[tokio::test]
    async fn heartbeat_activity_advances_chronologically_not_lexically() -> Result<()> {
        use agent_server::journal::task::{AgentTask, LeaseId, WorkerId};
        use agent_server::journal::thread_store::ThreadStore;

        // `t0()` is a whole second (zero nanos), so it formats with NO
        // fractional part — the operand that wins lexically but loses in time.
        let base = t0();
        let cases = [
            (
                "whole-second vs +500ms",
                base,
                base + Duration::milliseconds(500),
            ),
            (
                "fraction that is a lexical prefix of a later one (.1 vs .12)",
                base + Duration::milliseconds(100),
                base + Duration::milliseconds(120),
            ),
            (
                ".5 vs .55",
                base + Duration::milliseconds(500),
                base + Duration::milliseconds(550),
            ),
        ];

        for (label, older, newer) in cases {
            assert!(newer > older, "fixture sanity for {label}");

            let store = SqliteDurableStore::connect("sqlite::memory:").await?;
            let thread_id = ThreadId::from_string("t-activity-order");
            ThreadStore::get_or_create(&store, &thread_id, base).await?;
            let task = AgentTask::new_root_turn(thread_id.clone(), base, 3);
            let id = task.id.clone();
            AgentTaskStore::submit_root_turn(&store, task).await?;

            let worker = WorkerId::from_string("w-order");
            let lease = LeaseId::new();
            let expires = base + Duration::seconds(600);
            AgentTaskStore::try_acquire_task(
                &store,
                &id,
                worker.clone(),
                lease.clone(),
                expires,
                base,
            )
            .await?
            .context("acquire")?;

            // Persist the older instant, then beat again carrying the newer.
            AgentTaskStore::heartbeat_task(
                &store,
                &id,
                &worker,
                &lease,
                expires,
                Some(older),
                base,
            )
            .await?;
            let advanced = AgentTaskStore::heartbeat_task(
                &store,
                &id,
                &worker,
                &lease,
                expires,
                Some(newer),
                base,
            )
            .await?;
            assert_eq!(
                advanced.last_activity_at,
                Some(newer),
                "{label}: a chronologically newer observation must advance the column \
                 (a lexical MAX keeps the older string and drops this beat's work)",
            );

            // ...and the reverse must never rewind it.
            let held = AgentTaskStore::heartbeat_task(
                &store,
                &id,
                &worker,
                &lease,
                expires,
                Some(older),
                base,
            )
            .await?;
            assert_eq!(
                held.last_activity_at,
                Some(newer),
                "{label}: an older observation must never retire a newer one",
            );

            // A beat that carried no work leaves the column exactly as it was.
            let idle =
                AgentTaskStore::heartbeat_task(&store, &id, &worker, &lease, expires, None, base)
                    .await?;
            assert_eq!(
                idle.last_activity_at,
                Some(newer),
                "{label}: an idle beat must neither advance nor clear activity",
            );
        }

        Ok(())
    }

    async fn seed_compacted_sqlite_thread(
        store: &SqliteDurableStore,
        thread_id: &ThreadId,
    ) -> Result<Vec<agent_sdk_foundation::llm::Message>> {
        use agent_sdk_foundation::llm;
        use agent_server::journal::message_store::MessageProjectionStore;

        let raw_messages = vec![
            llm::Message::user("first"),
            llm::Message::assistant("second"),
            llm::Message::user("third"),
        ];
        MessageProjectionStore::commit_messages(store, thread_id, raw_messages.clone(), t0())
            .await?;
        let compacted = MessageProjectionStore::append_compaction(
            store,
            thread_id,
            vec![llm::Message::assistant("summary"), raw_messages[2].clone()],
            raw_messages.len(),
            1,
            t_plus(1),
        )
        .await?;
        assert_eq!(compacted.compactions.len(), 1);
        Ok(raw_messages)
    }

    #[tokio::test]
    async fn explicit_replace_history_invalidates_compaction_lineage() -> Result<()> {
        use agent_sdk_foundation::llm;
        use agent_server::journal::message_store::MessageProjectionStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-compaction-explicit-replace");
        let raw_messages = seed_compacted_sqlite_thread(&store, &thread_id).await?;

        let replacement = vec![
            llm::Message::user("replacement"),
            llm::Message::assistant("replacement answer"),
            llm::Message::user("replacement tail"),
        ];
        assert_eq!(
            replacement.len(),
            raw_messages.len(),
            "the explicit replace regression must not rely on the shrink trigger",
        );
        let replaced = MessageProjectionStore::replace_history(
            &store,
            &thread_id,
            replacement.clone(),
            t_plus(2),
        )
        .await?;
        assert!(
            replaced.compactions.is_empty(),
            "the explicit destructive API must clear lineage in its row write",
        );
        let reloaded = MessageProjectionStore::get(&store, &thread_id)
            .await?
            .context("replaced projection")?;
        assert!(reloaded.compactions.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn legacy_shrink_upsert_clears_stale_compaction_lineage() -> Result<()> {
        use agent_sdk_foundation::llm;
        use agent_server::journal::message_store::MessageProjectionStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-compaction-upgrade-fence");
        seed_compacted_sqlite_thread(&store, &thread_id).await?;

        let appended = MessageProjectionStore::commit_messages(
            &store,
            &thread_id,
            vec![llm::Message::user("new tail")],
            t_plus(3),
        )
        .await?;
        let compacted = MessageProjectionStore::append_compaction(
            &store,
            &thread_id,
            vec![
                llm::Message::assistant("new summary"),
                appended.messages[3].clone(),
            ],
            appended.context_history().len(),
            1,
            t_plus(4),
        )
        .await?;
        assert_eq!(compacted.compactions.len(), 2);
        let lineage_preserved = MessageProjectionStore::commit_messages(
            &store,
            &thread_id,
            vec![llm::Message::assistant("post-compaction tail")],
            t_plus(5),
        )
        .await?;
        assert_eq!(lineage_preserved.compactions.len(), 2);
        let stale_compactions_json = serde_json::to_value(&lineage_preserved.compactions)
            .context("encode stale compaction fixture")?;

        // This is the pre-compaction writer's UPSERT shape: it shortens
        // history but does not know that compactions_json exists.
        let legacy_replacement = vec![llm::Message::user("legacy replacement")];
        sqlx::query(
            r"
INSERT INTO agent_sdk_message_heads (
    thread_id, history_json, draft_messages_json, version, created_at, updated_at
) VALUES (?1, ?2, NULL, ?3, ?4, ?5)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = excluded.history_json,
    draft_messages_json = excluded.draft_messages_json,
    version = excluded.version,
    updated_at = excluded.updated_at
",
        )
        .bind(super::thread_key(&thread_id))
        .bind(serde_json::to_value(&legacy_replacement)?)
        .bind(99_i64)
        .bind(t0())
        .bind(t_plus(6))
        .execute(&store.pool)
        .await
        .context("simulate legacy sqlite message-head upsert")?;

        let persisted_compactions: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT compactions_json FROM agent_sdk_message_heads WHERE thread_id = ?1",
        )
        .bind(super::thread_key(&thread_id))
        .fetch_one(&store.pool)
        .await
        .context("read sqlite compaction lineage after legacy shrink")?;
        assert!(
            persisted_compactions.is_none(),
            "the migration trigger must clear stale lineage atomically",
        );

        // Rows left stale before the fence was installed must still be
        // recoverable: the decoder discards an out-of-range lineage and
        // exposes the intact raw replacement.
        sqlx::query(
            "UPDATE agent_sdk_message_heads SET compactions_json = ?1 WHERE thread_id = ?2",
        )
        .bind(stale_compactions_json)
        .bind(super::thread_key(&thread_id))
        .execute(&store.pool)
        .await
        .context("restore stale sqlite lineage fixture")?;
        let safely_decoded = MessageProjectionStore::get(&store, &thread_id)
            .await?
            .context("decode sqlite head with stale lineage")?;
        assert!(safely_decoded.compactions.is_empty());
        assert_eq!(safely_decoded.messages.len(), legacy_replacement.len());
        assert_eq!(
            safely_decoded.context_history().len(),
            legacy_replacement.len(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod input_injection_migration_test {
    use super::*;
    use agent_server::journal::task::SubmittedInputItem;

    fn t_plus(seconds: i64) -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1_700_000_000 + seconds)
    }

    #[tokio::test]
    async fn sqlite_persists_and_cancels_input_injection_rows() -> anyhow::Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread = ThreadId::from_string("sqlite-injection");
        let root = store
            .submit_root_turn_idempotent(SubmitRootTurnParams {
                task: AgentTask::new_root_turn_with_input(
                    thread.clone(),
                    vec![SubmittedInputItem::Text {
                        text: "root".into(),
                    }],
                    t_plus(1),
                    3,
                ),
                idempotency: None,
                max_queued_depth: None,
                disposition: SubmitDisposition::NextTurn,
            })
            .await
            .map_err(|error| anyhow::anyhow!("root submit: {error}"))?;
        let running = store
            .try_acquire_task(
                &root.task.id,
                WorkerId::from_string("worker"),
                LeaseId::from_string("lease"),
                t_plus(100),
                t_plus(2),
            )
            .await?
            .context("acquire root")?;
        let injection = store
            .submit_root_turn_idempotent(SubmitRootTurnParams {
                task: AgentTask::new_root_turn_with_input(
                    thread,
                    vec![SubmittedInputItem::Text {
                        text: "inject".into(),
                    }],
                    t_plus(3),
                    3,
                ),
                idempotency: None,
                max_queued_depth: None,
                disposition: SubmitDisposition::InjectAtBoundary,
            })
            .await
            .map_err(|error| anyhow::anyhow!("injection submit: {error}"))?;
        assert_eq!(injection.task.kind, TaskKind::InputInjection);
        assert_eq!(injection.task.status, TaskStatus::Queued);

        store.cancel_tree(&running.id, t_plus(4)).await?;
        let cancelled = AgentTaskStore::get(&store, &injection.task.id)
            .await?
            .context("injection survives as cancelled journal row")?;
        assert_eq!(cancelled.status, TaskStatus::Cancelled);
        Ok(())
    }
}

#[cfg(test)]
mod orphan_listing_plan_test {
    use super::*;

    /// Retained history dwarfing the live rows: 400 finished tasks and one
    /// live child under a finished root.
    const SKEWED_HISTORY: &str = r#"
INSERT INTO agent_sdk_threads (thread_id, status, committed_turns, total_input_tokens, total_output_tokens, created_at, updated_at)
VALUES ('plan', 'active', 0, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 199)
INSERT INTO agent_sdk_tasks (id, kind, status, parent_id, root_id, depth, thread_id, attempt, max_attempts, terminal_reason_json, created_at, updated_at, completed_at)
SELECT printf('root-%04d', i), 'root_turn', 'completed', NULL, printf('root-%04d', i), 0, 'plan', 1, 3, '{"kind":"completed"}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z' FROM n;
WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 199)
INSERT INTO agent_sdk_tasks (id, kind, status, parent_id, root_id, depth, thread_id, attempt, max_attempts, terminal_reason_json, created_at, updated_at, completed_at)
SELECT printf('done-%04d', i), 'tool_runtime', 'completed', printf('root-%04d', i), printf('root-%04d', i), 1, 'plan', 1, 3, '{"kind":"completed"}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z' FROM n;
INSERT INTO agent_sdk_tasks (id, kind, status, parent_id, root_id, depth, thread_id, attempt, max_attempts, created_at, updated_at)
VALUES ('live-0007', 'tool_runtime', 'pending', 'root-0007', 'root-0007', 1, 'plan', 0, 3, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
"#;

    async fn plan(store: &SqliteDurableStore) -> Result<Vec<String>> {
        let steps: Vec<(i64, i64, i64, String)> = sqlx::query_as(concat!(
            "EXPLAIN QUERY PLAN ",
            include_str!("terminal_roots_with_live_descendants.sql"),
        ))
        .fetch_all(&store.pool)
        .await?;
        Ok(steps.into_iter().map(|(_, _, _, detail)| detail).collect())
    }

    fn assert_reads_only_live_rows(plan: &[String], label: &str) {
        assert!(
            plan.iter()
                .any(|step| step.starts_with("SEARCH descendant") && step.contains("(status=?)")),
            "{label}: the listing must look the live rows up by status, not scan the history: \
             {plan:?}",
        );
    }

    /// The child-thread link a root's undelivered input is classified by
    /// is read through an index, never a scan of every task.
    #[tokio::test]
    async fn child_thread_link_is_looked_up_through_an_index() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let steps: Vec<(i64, i64, i64, String)> = sqlx::query_as(concat!(
            "EXPLAIN QUERY PLAN ",
            include_str!("links_child_thread_root.sql"),
        ))
        .fetch_all(&store.pool)
        .await?;
        let plan: Vec<String> = steps.into_iter().map(|(_, _, _, detail)| detail).collect();
        assert!(
            plan.iter()
                .any(|step| step.starts_with("SEARCH agent_sdk_tasks USING")),
            "the link must be searched through an index, not scanned: {plan:?}",
        );
        Ok(())
    }

    #[tokio::test]
    async fn orphan_listing_reads_only_the_live_rows_with_and_without_statistics() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        sqlx::raw_sql(SKEWED_HISTORY).execute(&store.pool).await?;
        assert_reads_only_live_rows(&plan(&store).await?, "without statistics");

        sqlx::query("ANALYZE").execute(&store.pool).await?;
        assert_reads_only_live_rows(&plan(&store).await?, "after ANALYZE");

        sqlx::raw_sql("DELETE FROM sqlite_stat4; ANALYZE sqlite_schema;")
            .execute(&store.pool)
            .await?;
        assert_reads_only_live_rows(&plan(&store).await?, "with averaged statistics only");

        assert_eq!(
            store.list_terminal_roots_with_live_descendants().await?,
            vec![AgentTaskId::from_string("root-0007")],
        );
        Ok(())
    }
}
