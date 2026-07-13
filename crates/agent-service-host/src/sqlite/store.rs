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
    ContinuationEnvelope, ListenExecutionContext, ThreadId, TokenUsage, llm,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{FromRow, Sqlite, SqlitePool, Transaction};
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
use agent_server::journal::completed_turn_transaction::AtomicCompletedTurnCommitter;
use agent_server::journal::event_outbox_transaction::{
    AtomicEventOutboxCommitter, EventOutboxCommit, EventOutboxCommitOutcome,
};
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_intent::{ExecutionIntent, ExecutionIntentStore, OperationId};
use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};
use agent_server::journal::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};
use agent_server::journal::message::MessageProjection;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::outbox::{
    NewOutboxRow, OutboxRow, OutboxRowId, OutboxStatus, OutboxStore, kind_payload_invariants_hold,
};
use agent_server::journal::outbox_message::{
    OutboxMessage, OutboxMessageKind, TaskWakeupPayload, ThreadEventsAvailablePayload,
};
use agent_server::journal::recovery::{
    RecoveryAction, RecoveryContext, RecoveryRecord, classify_recovery,
};
use agent_server::journal::relay::{
    TASK_WAKEUP_OUTBOX_MAX_ATTEMPTS, TaskWakeupEmitter, TaskWakeupTrigger,
};
use agent_server::journal::retention::{RetentionCursor, RetentionStore};
use agent_server::journal::store::{
    AgentTaskStore, CancelTreeOutcome, RequeueOutcome, SubagentInvocationSpawn,
    SubmitRootTurnError, SubmitRootTurnOutcome, SubmitRootTurnParams,
};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskKind, TaskStatus,
    WorkerId,
};
use agent_server::journal::task_state::SubagentInvocationState;
use agent_server::journal::thread::Thread;
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::tool_audit::{ToolAuditEvent, ToolAuditEventId, ToolAuditEventStore};
use agent_server::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId,
};
use agent_server::journal::turn_attempt_store::TurnAttemptStore;

use super::migrations::apply_durable_core_migrations;

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
}

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
    pub(crate) const fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Connect to a `SQLite` database and apply the durable-core migrations.
    ///
    /// The `database_url` is a `SQLite` connection string. Use
    /// `"sqlite::memory:"` for an ephemeral in-memory database (useful
    /// for tests) or a file path like `"sqlite:///path/to/agent-sdk.db"`.
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
        let is_memory = database_url.contains(":memory:");

        let pool = SqlitePoolOptions::new()
            // In-memory databases are per-connection in SQLite. Use a
            // single connection so migrations and queries share the
            // same schema. File-backed databases can use multiple
            // connections under WAL mode.
            .max_connections(if is_memory { 1 } else { 4 })
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
         last_error, pending_child_count, spawn_index, result_payload, \
         otel_traceparent, created_at, updated_at, completed_at"
    };
}

impl SqliteDurableStore {
    async fn begin(&self) -> Result<Transaction<'_, Sqlite>> {
        // BEGIN IMMEDIATE acquires the RESERVED lock up front so that
        // under WAL mode we fail fast with plain SQLITE_BUSY (which
        // busy_timeout retries) rather than SQLITE_BUSY_SNAPSHOT on
        // upgrade from a deferred read snapshot to a write (which
        // busy_timeout does not retry).  See the module-level docs.
        self.pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .context("begin sqlite durable-core transaction")
    }

    async fn bootstrap_thread_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<()> {
        let bootstrap = Thread::new(thread_id.clone(), now);
        let thread_id_key = thread_key(&bootstrap.thread_id);
        let status_wire = enum_to_wire(&bootstrap.status)?;
        let committed_turns = i64::from(bootstrap.committed_turns);
        let input_tokens = i64::from(bootstrap.total_usage.input_tokens);
        let output_tokens = i64::from(bootstrap.total_usage.output_tokens);
        sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT (thread_id) DO NOTHING
",
            thread_id_key,
            status_wire,
            committed_turns,
            input_tokens,
            output_tokens,
            bootstrap.created_at,
            bootstrap.updated_at,
        )
        .execute(&mut **tx)
        .await
        .context("bootstrap thread row")?;
        Ok(())
    }

    async fn get_thread_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<Option<Thread>> {
        let record = sqlx::query_as::<_, ThreadRecord>(
            r"
SELECT thread_id, status, committed_turns,
       total_input_tokens, total_output_tokens, created_at, updated_at
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

    async fn get_thread_pool(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
        let record = sqlx::query_as::<_, ThreadRecord>(
            r"
SELECT thread_id, status, committed_turns,
       total_input_tokens, total_output_tokens, created_at, updated_at
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
        sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT (thread_id) DO UPDATE SET
    status = excluded.status,
    committed_turns = excluded.committed_turns,
    total_input_tokens = excluded.total_input_tokens,
    total_output_tokens = excluded.total_output_tokens,
    updated_at = excluded.updated_at
",
            thread_id_key,
            status_wire,
            committed_turns,
            input_tokens,
            output_tokens,
            thread.created_at,
            thread.updated_at,
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
        let record = sqlx::query_as::<_, MessageHeadRecord>(
            r"
SELECT thread_id, history_json, draft_messages_json, version, created_at, updated_at
FROM agent_sdk_message_heads
WHERE thread_id = ?1
",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("get message head for {thread_id}"))?;

        if let Some(r) = record {
            r.try_into()
        } else {
            let fresh = MessageProjection::new(thread_id.clone(), now);
            // Use INSERT … ON CONFLICT DO NOTHING (not the destructive
            // upsert) so a concurrent commit that already wrote a real
            // projection is never overwritten with an empty bootstrap row.
            // Matches the Postgres lock_message_head_tx behaviour.
            let thread_id_key = thread_key(thread_id);
            let history_json = json_to_value(&fresh.messages, "message head bootstrap history")?;
            let version = i64_from_u64(fresh.version, "message head bootstrap version")?;
            // Bootstrap the row with `draft_messages_json = NULL` —
            // a fresh thread has no in-flight turn. The full upsert
            // path (`upsert_message_head_tx`) carries the actual
            // draft once a suspension boundary fires.
            sqlx::query(
                r"
INSERT INTO agent_sdk_message_heads (thread_id, history_json, draft_messages_json, version, created_at, updated_at)
VALUES (?1, ?2, NULL, ?3, ?4, ?5)
ON CONFLICT (thread_id) DO NOTHING
",
            )
            .bind(thread_id_key)
            .bind(history_json)
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
        let record = sqlx::query_as::<_, MessageHeadRecord>(
            r"
SELECT thread_id, history_json, draft_messages_json, version, created_at, updated_at
FROM agent_sdk_message_heads
WHERE thread_id = ?1
",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get message head {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
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
        let version = i64_from_u64(projection.version, "message head version")?;
        sqlx::query!(
            r"
INSERT INTO agent_sdk_message_heads (thread_id, history_json, draft_messages_json, version, created_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = excluded.history_json,
    draft_messages_json = excluded.draft_messages_json,
    version = excluded.version,
    updated_at = excluded.updated_at
",
            thread_id_key,
            history_json,
            draft_messages_json,
            version,
            projection.created_at,
            projection.updated_at,
        )
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
       cached_input_tokens, opened_at, closed_at, duration_ms,
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
       cached_input_tokens, opened_at, closed_at, duration_ms,
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
    cached_input_tokens, opened_at, closed_at, duration_ms,
    otel_trace_id, otel_span_id
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
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
    cached_input_tokens = ?9, closed_at = ?10, duration_ms = ?11,
    otel_trace_id = ?12, otel_span_id = ?13
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
        would_queue: bool,
        idempotency: Option<agent_server::journal::store::SubmitRootIdempotency>,
    ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
        let admitted = if would_queue {
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
        let pending_child_count = i64::from(task.pending_child_count);
        let spawn_index = task.spawn_index.map(i64::from);
        sqlx::query!(
            r"
INSERT INTO agent_sdk_tasks (
    id, kind, status, parent_id, root_id, depth, thread_id,
    submitted_input_json, caller_metadata_json, worker_id, lease_id, lease_expires_at,
    last_heartbeat_at, state_json, attempt, max_attempts, last_error,
    pending_child_count, spawn_index, result_payload,
    created_at, updated_at, completed_at, otel_traceparent
) VALUES (
    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
    ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24
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
            pending_child_count,
            spawn_index,
            task.result_payload,
            task.created_at,
            task.updated_at,
            task.completed_at,
            task.otel_traceparent,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert task {}", task.id))?;
        Ok(())
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
        let pending_child_count = i64::from(task.pending_child_count);
        let spawn_index = task.spawn_index.map(i64::from);
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_tasks SET
    kind = ?2, status = ?3, parent_id = ?4, root_id = ?5,
    depth = ?6, thread_id = ?7, submitted_input_json = ?8,
    worker_id = ?9, lease_id = ?10, lease_expires_at = ?11,
    last_heartbeat_at = ?12, state_json = ?13, attempt = ?14,
    max_attempts = ?15, last_error = ?16, pending_child_count = ?17,
    spawn_index = ?18, result_payload = ?19,
    created_at = ?20, updated_at = ?21, completed_at = ?22,
    caller_metadata_json = ?23,
    otel_traceparent = ?24
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
            pending_child_count,
            spawn_index,
            task.result_payload,
            task.created_at,
            task.updated_at,
            task.completed_at,
            caller_metadata_json,
            task.otel_traceparent,
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
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
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
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
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
        let blocking = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation') LIMIT 1", blocking_check_key)
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
        continuation_usage: Option<TokenUsage>,
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
            vec![AgentEvent::cancelled(turn, usage)],
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
    async fn commit_fork_atomic_inner(&self, params: ForkCommitParams) -> Result<()> {
        use agent_server::journal::message::MessageProjection;

        let mut tx = self.begin().await?;

        // 1. Bootstrap the destination thread aggregate. Same call
        //    `bootstrap_thread_row_tx` makes for `commit_completed_turn`,
        //    which means the row layout (status, created_at, etc.)
        //    matches what `CreateThread` would have produced — the
        //    fork's destination is indistinguishable from a fresh
        //    thread the moment the transaction commits.
        Self::bootstrap_thread_row_tx(&mut tx, &params.new_thread_id, params.now).await?;

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
        if !params.messages.is_empty() {
            let projection_before =
                Self::get_message_head_tx(&mut tx, &params.new_thread_id, params.now).await?;
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

        // 5. Re-commit events under the new thread id with fresh
        //    sequences. The starting sequence is always 0 because
        //    a freshly-bootstrapped thread has no prior events;
        //    `next_event_sequence_tx` returns 0 in that case.
        if !params.events.is_empty() {
            let start_seq = Self::next_event_sequence_tx(&mut tx, &params.new_thread_id).await?;
            Self::insert_events_tx(
                &mut tx,
                &params.new_thread_id,
                params.events,
                start_seq,
                params.now,
            )
            .await?;
        }

        tx.commit()
            .await
            .context("commit sqlite fork transaction")?;
        Ok(())
    }

    /// Enforce [`CompletedTurnCommit::owner_guard`] on the task row
    /// inside the completed-turn transaction (issue #354, residual 5
    /// hardening): the slot-shift retry validates — under `SQLite`'s
    /// exclusive write transaction — that the presenting worker still
    /// owns a live `Running` row, so a cancellation / lease loss
    /// between the caller's shift-eligibility check and the retry
    /// rejects here instead of splicing a dead root's turn into the
    /// shifted slot. A `None` guard (every first, non-shifted commit)
    /// is a no-op.
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
    async fn commit_completed_turn_atomic_inner(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome> {
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

        // Owner-guarded commit (issue #354, residual 5 hardening).
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

        tx.commit()
            .await
            .context("commit sqlite completed-turn transaction")?;

        Ok(CommitOutcome {
            closed_attempt,
            thread,
            checkpoint,
            committed_events,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// AgentTaskStore
// ─────────────────────────────────────────────────────────────────────

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

#[async_trait]
impl AgentTaskStore for SqliteDurableStore {
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
        let thread_has_blocking_root: bool = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation') LIMIT 1", blocking_check_key)
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
        } = params;
        agent_server::journal::store::validate_submit_root_shape(&task)?;

        // BEGIN IMMEDIATE serializes writers, so the idempotency claim,
        // queue-depth check, and admission insert below are atomic.
        let mut tx = self.begin().await.map_err(SubmitRootTurnError::Other)?;
        Self::bootstrap_thread_row_tx(&mut tx, &task.thread_id, task.created_at)
            .await
            .map_err(SubmitRootTurnError::Other)?;

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
        let thread_has_blocking_root: bool = sqlx::query_scalar!("SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation') LIMIT 1", blocking_check_key)
            .fetch_optional(&mut *tx)
            .await
            .with_context(|| format!("check active root slot for {}", task.thread_id))
            .map_err(SubmitRootTurnError::Other)?
            .flatten()
            .is_some();

        let current_queued = Self::queued_root_count_tx(&mut tx, &task.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        // 2. Back-pressure before any write.
        let would_queue = thread_has_blocking_root || current_queued > 0;
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
        Self::commit_fresh_admission_tx(tx, task, would_queue, idempotency).await
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
        let inserted = sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES (?1, ?2, ?3, 'null', ?4)
               ON CONFLICT (request_id) DO NOTHING"#,
            request_id,
            kind_wire,
            fingerprint,
            now,
        )
        .execute(&self.pool)
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
            // Reserved by a concurrent in-flight claim that hasn't
            // recorded its result yet — fail closed.
            return Ok(IdempotencyClaim::Conflict);
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
        let record = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ", task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE thread_id = ?1 \
               AND kind = 'root_turn' \
               AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation') \
             ORDER BY created_at, id LIMIT 1",
        ))
        .bind(thread_key(thread_id))
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("load active root for {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
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
        if !old.status.can_be_leased() {
            return Ok(None);
        }
        match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
            RecoveryAction::NoAction => {}
            RecoveryAction::FailClosed(reason) => {
                let failed = old
                    .clone()
                    .fail_with_reason(reason, now)
                    .context("try_acquire_task: fail-closed transition failed")?;
                Self::update_task_tx(&mut tx, &failed).await?;
                Self::propagate_terminal_to_parent_tx(&mut tx, &failed, now, "try_acquire_task")
                    .await?;
                tx.commit().await.context("commit fail-closed acquire")?;
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
            .mark_running(worker, lease, expires_at, now)
            .context("try_acquire_task rejected: mark_running transition failed")?;
        Self::update_task_tx(&mut tx, &claimed).await?;
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
        // SQLite has no SKIP LOCKED; the single-process model means
        // database-level locking is sufficient. We loop to skip
        // fail-closed rows, same as the Postgres backend.
        loop {
            let mut tx = self.begin().await?;
            let record = sqlx::query_as::<_, TaskRecord>(concat!(
                "SELECT ",
                task_columns!(),
                " FROM agent_sdk_tasks WHERE status = 'pending' \
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

            match classify_recovery(&old, RecoveryContext::AcquisitionAttempt) {
                RecoveryAction::NoAction => {
                    let claimed = old
                        .clone()
                        .mark_running(worker.clone(), lease.clone(), expires_at, now)
                        .context(
                            "acquire_next_runnable rejected: mark_running transition failed",
                        )?;
                    Self::update_task_tx(&mut tx, &claimed).await?;
                    tx.commit().await.context("commit acquire_next_runnable")?;
                    return Ok(Some(claimed));
                }
                RecoveryAction::FailClosed(reason) => {
                    let failed = old
                        .clone()
                        .fail_with_reason(reason, now)
                        .context("acquire_next_runnable: fail-closed transition failed")?;
                    Self::update_task_tx(&mut tx, &failed).await?;
                    Self::propagate_terminal_to_parent_tx(
                        &mut tx,
                        &failed,
                        now,
                        "acquire_next_runnable",
                    )
                    .await?;
                    tx.commit()
                        .await
                        .context("commit fail-closed runnable head")?;
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

    async fn heartbeat_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        expires_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<AgentTask> {
        let mut tx = self.begin().await?;
        // Single guarded UPDATE of only the lease columns — never reload
        // and rewrite the whole 24-column row (re-serialising the
        // submitted_input / state JSON blobs) on a per-tick heartbeat.
        // `RETURNING` hands back the refreshed row in one round trip.
        let worker_str = worker.as_str();
        let lease_str = lease.as_str();
        let updated = sqlx::query_as::<_, TaskRecord>(concat!(
            "UPDATE agent_sdk_tasks \
             SET lease_expires_at = ?4, last_heartbeat_at = ?5, updated_at = ?5 \
             WHERE id = ?1 AND status = 'running' AND worker_id = ?2 AND lease_id = ?3 \
             RETURNING ",
            task_columns!(),
        ))
        .bind(id.as_str())
        .bind(worker_str)
        .bind(lease_str)
        .bind(expires_at)
        .bind(now)
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

    async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>> {
        let mut tx = self.begin().await?;
        // Bound the sweep to a batch so a mass worker outage cannot turn
        // one sweep into a giant transaction that locks every expired
        // task at once; the host's drain loop (and any other caller)
        // keeps calling until a pass returns fewer than the batch.
        let expired = sqlx::query_as::<_, TaskRecord>(concat!(
            "SELECT ",
            task_columns!(),
            " FROM agent_sdk_tasks \
             WHERE status = 'running' AND lease_expires_at <= ?1 \
             ORDER BY lease_expires_at, id \
             LIMIT ?2",
        ))
        .bind(now)
        .bind(i64::try_from(agent_server::journal::store::LEASE_RELEASE_BATCH).unwrap_or(i64::MAX))
        .fetch_all(&mut *tx)
        .await
        .context("load expired leases")?;

        let mut released = Vec::with_capacity(expired.len());
        for record in expired {
            let old = AgentTask::try_from(record)?;
            if old.status != TaskStatus::Running {
                let status = old.status;
                return Err(anyhow!(
                    "release_expired_leases: expiry index held non-running row {} in status {status:?}",
                    old.id
                ));
            }
            let record = match classify_recovery(&old, RecoveryContext::ExpiredLease) {
                RecoveryAction::Requeue => {
                    let released_row = old
                        .clone()
                        .release_lease(now)
                        .context("release_expired_leases: release transition failed")?;
                    Self::update_task_tx(&mut tx, &released_row).await?;
                    RecoveryRecord::requeued(old.id.clone())
                }
                RecoveryAction::FailClosed(reason) => {
                    let failed = old
                        .clone()
                        .fail_with_reason(reason, now)
                        .context("release_expired_leases: fail-closed transition failed")?;
                    Self::update_task_tx(&mut tx, &failed).await?;
                    Self::propagate_terminal_to_parent_tx(
                        &mut tx,
                        &failed,
                        now,
                        "release_expired_leases",
                    )
                    .await?;
                    RecoveryRecord::failed_closed(old.id.clone(), reason)
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
        Ok(released)
    }

    async fn requeue_owned_task(
        &self,
        id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
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
        // A steering wake is distinguished from an ordinary fan-in only
        // by a non-empty steering payload (both share the ReadyToResume
        // kind). An empty payload would create a ReadyToResume whose
        // children are still running — a broken fan-in — so reject it.
        if steering.is_empty() {
            return Err(anyhow!(
                "steering wake rejected: steering content must be non-empty"
            ));
        }
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("steering wake rejected: task {parent_id} does not exist"))?;

        // Idempotent CAS: decline unless the parent is still parked.
        if old.status != TaskStatus::WaitingOnChildren {
            return Ok(None);
        }

        let woken = old
            .clone()
            .begin_steering_resume(steering, now)
            .context("steering wake rejected: begin_steering_resume transition failed")?;
        Self::update_task_tx(&mut tx, &woken).await?;
        tx.commit()
            .await
            .context("commit enqueue_steering_resume")?;
        Ok(Some(woken))
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
        let mut tx = self.begin().await?;
        let old = Self::load_task_tx(&mut tx, parent_id)
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
        if !old.state.is_steering_resume() {
            return Err(anyhow!(
                "repark rejected: task {parent_id} is not a steering resume"
            ));
        }

        // Re-index the re-attach children to dense spawn_index
        // positions matching the new continuation's tool-use order.
        for (idx, child_id) in reattach.iter().enumerate() {
            let mut child = Self::load_task_tx(&mut tx, child_id)
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
            Self::update_task_tx(&mut tx, &child).await?;
        }

        // Journal-authoritative live count (matches the completion
        // path): a re-attach child that finished during the steering
        // LLM call is already terminal and excluded.
        let live = Self::load_live_child_count_tx(&mut tx, parent_id).await?;
        let reparked = old
            .clone()
            .repark_after_steering(live, payload, reattach, now)
            .context("repark rejected: repark_after_steering transition failed")?;
        Self::update_task_tx(&mut tx, &reparked).await?;
        tx.commit().await.context("commit repark_after_steering")?;
        Ok(reparked)
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
            .await_confirmation(continuation, prepared_operation, now)
            .context("pause rejected: await_confirmation transition failed")?;
        Self::update_task_tx(&mut tx, &paused).await?;
        tx.commit().await.context("commit pause_on_confirmation")?;
        Ok(paused)
    }

    #[allow(clippy::too_many_arguments)]
    async fn spawn_tool_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        specs: Vec<ChildSpawnSpec>,
        payload: SuspensionPayload,
        child_otel_traceparent: Option<String>,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<AgentTask>)> {
        if specs.is_empty() {
            return Err(anyhow!("spawn rejected: specs must be non-empty"));
        }
        let mut tx = self.begin().await?;
        let old_parent = Self::load_task_tx(&mut tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;

        let mut children = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.into_iter().enumerate() {
            let mut child =
                AgentTask::new_child(&old_parent, TaskKind::ToolRuntime, now, spec.max_attempts)
                    .context("spawn rejected: new_child failed")?;
            child.spawn_index =
                Some(u32::try_from(idx).context("spawn rejected: batch index exceeds u32::MAX")?);
            child.otel_traceparent.clone_from(&child_otel_traceparent);
            let existing = Self::load_task_tx(&mut tx, &child.id).await?;
            if existing.is_some() || children.iter().any(|e: &AgentTask| e.id == child.id) {
                return Err(anyhow!(
                    "spawn rejected: child id {} already exists",
                    child.id
                ));
            }
            children.push(child);
        }

        let child_count = u32::try_from(children.len())
            .context("spawn rejected: child count exceeds u32::MAX")?;
        let child_ids = children.iter().map(|c| c.id.clone()).collect();
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(&mut tx, &new_parent).await?;
        for child in &children {
            Self::insert_task_tx(&mut tx, child).await?;
        }
        tx.commit().await.context("commit spawn_tool_children")?;
        Ok((new_parent, children))
    }

    async fn spawn_subagent_invocation(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: SubagentInvocationSpawn,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let mut tx = self.begin().await?;
        let SubagentInvocationSpawn {
            child_thread_id,
            spec,
            child_root_input,
            payload,
            spawn_index,
            child_caller_metadata,
        } = spawn;

        let old_parent = Self::load_task_tx(&mut tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;

        // Verify child thread exists
        let child_thread = Self::get_thread_tx(&mut tx, &child_thread_id).await?;
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
        .fetch_optional(&mut *tx)
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
            AgentTask::DEFAULT_MAX_ATTEMPTS,
        );
        let existing = Self::load_task_tx(&mut tx, &child_root.id).await?;
        if existing.is_some() {
            return Err(anyhow!(
                "spawn rejected: child root id {} already exists",
                child_root.id
            ));
        }

        let invocation = AgentTask::new_subagent_invocation(
            &old_parent,
            SubagentInvocationState {
                spec,
                child_thread_id,
                child_root_task_id: child_root.id.clone(),
            },
            spawn_index,
            now,
            AgentTask::DEFAULT_MAX_ATTEMPTS,
        )
        .context("spawn rejected: new_subagent_invocation failed")?;
        let existing = Self::load_task_tx(&mut tx, &invocation.id).await?;
        if existing.is_some() {
            return Err(anyhow!(
                "spawn rejected: invocation id {} already exists",
                invocation.id
            ));
        }

        let new_parent = old_parent
            .clone()
            .wait_on_children(1, payload, vec![invocation.id.clone()], now)
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(&mut tx, &new_parent).await?;
        Self::insert_task_tx(&mut tx, &invocation).await?;
        Self::insert_task_tx(&mut tx, &child_root).await?;
        tx.commit()
            .await
            .context("commit spawn_subagent_invocation")?;
        Ok((new_parent, invocation, child_root))
    }

    async fn spawn_subagent_batch(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawns: Vec<SubagentInvocationSpawn>,
        payload: SuspensionPayload,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>)> {
        if spawns.is_empty() {
            return Err(anyhow!("spawn rejected: spawns must be non-empty"));
        }
        let mut tx = self.begin().await?;

        let old_parent = Self::load_task_tx(&mut tx, parent_id)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;

        // Cross-entry uniqueness — same rule as the InMemory impl.
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

        // Build (invocation, child_root) per entry, validating against
        // existing rows under the same transaction so the whole batch
        // is rejected atomically if any single entry would conflict.
        let mut prepared: Vec<(AgentTask, AgentTask)> = Vec::with_capacity(spawns.len());
        let mut child_ids: Vec<AgentTaskId> = Vec::with_capacity(spawns.len());
        for spawn in spawns {
            let SubagentInvocationSpawn {
                child_thread_id,
                spec,
                child_root_input,
                payload: _per_entry_payload,
                spawn_index,
                child_caller_metadata,
            } = spawn;

            // Verify child thread exists.
            let child_thread = Self::get_thread_tx(&mut tx, &child_thread_id).await?;
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
            .fetch_optional(&mut *tx)
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
                AgentTask::DEFAULT_MAX_ATTEMPTS,
            );
            let existing = Self::load_task_tx(&mut tx, &child_root.id).await?;
            if existing.is_some() {
                return Err(anyhow!(
                    "spawn rejected: child root id {} already exists",
                    child_root.id
                ));
            }

            let invocation = AgentTask::new_subagent_invocation(
                &old_parent,
                SubagentInvocationState {
                    spec,
                    child_thread_id,
                    child_root_task_id: child_root.id.clone(),
                },
                spawn_index,
                now,
                AgentTask::DEFAULT_MAX_ATTEMPTS,
            )
            .context("spawn rejected: new_subagent_invocation failed")?;
            let existing = Self::load_task_tx(&mut tx, &invocation.id).await?;
            if existing.is_some() {
                return Err(anyhow!(
                    "spawn rejected: invocation id {} already exists",
                    invocation.id
                ));
            }
            child_ids.push(invocation.id.clone());
            prepared.push((invocation, child_root));
        }

        let child_count =
            u32::try_from(prepared.len()).context("spawn rejected: child count exceeds u32")?;
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;

        Self::update_task_tx(&mut tx, &new_parent).await?;
        for (invocation, child_root) in &prepared {
            Self::insert_task_tx(&mut tx, invocation).await?;
            Self::insert_task_tx(&mut tx, child_root).await?;
        }
        tx.commit().await.context("commit spawn_subagent_batch")?;
        Ok((new_parent, prepared))
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
        let mut tx = self.begin().await?;
        let result = Self::apply_task_terminal_transition_tx(
            &mut tx,
            child_id,
            worker,
            lease,
            now,
            "complete_task_with_result",
            move |child| child.complete_with_result(result_payload, now),
        )
        .await?;
        tx.commit()
            .await
            .context("commit complete_task_with_result")?;
        Ok(result)
    }

    async fn fail_task(
        &self,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        error: String,
        now: OffsetDateTime,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let mut tx = self.begin().await?;
        let result = Self::apply_task_terminal_transition_tx(
            &mut tx,
            child_id,
            worker,
            lease,
            now,
            "fail_task",
            move |child| child.fail(error, now),
        )
        .await?;
        tx.commit().await.context("commit fail_task")?;
        Ok(result)
    }

    async fn cancel_tree(
        &self,
        root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<CancelTreeOutcome> {
        let mut tx = self.begin().await?;
        let Some(_) = Self::load_task_tx(&mut tx, root_id).await? else {
            return Err(anyhow!(
                "cancel_tree rejected: task {root_id} does not exist"
            ));
        };

        // Collect the given task + descendants (mirror in-memory
        // `collect_subtree`: children-BFS, root-first, following
        // SubagentInvocation linkage across threads). A mid-tree id
        // cancels only its own subtree instead of no-opping on a
        // `root_id = $1` scan. SQLite serializes writers via
        // BEGIN IMMEDIATE, so there is no lock-order deadlock to avoid;
        // cancelling in BFS order keeps the returned `transitioned` slice
        // identical to the in-memory reference store.
        //
        // Round-2 F1 audit: unlike Postgres (READ COMMITTED row locks),
        // `begin()` here takes the database write lock up front
        // (`BEGIN IMMEDIATE`), so no concurrent writer can settle any
        // of these rows between this snapshot and the UPDATEs below —
        // the snapshot IS the transaction's consistent view. Two
        // racing `cancel_tree` calls fully serialize: the loser's
        // snapshot already sees terminal rows and it transitions /
        // emits nothing. A Postgres-style locked re-read would re-read
        // identical data, so this backend intentionally keeps
        // snapshot-driven transitions (pinned by the
        // `conformance_sqlite_concurrent_cancels_single_marker` race
        // test).
        let all_tasks = Self::collect_subtree_tx(&mut tx, root_id).await?;

        let mut transitioned = Vec::with_capacity(all_tasks.len());
        // Track cancelled root-turn roots (and their threads) so we can
        // wake their linked invocations and promote queued successors.
        let mut cancelled_root_ids: Vec<AgentTaskId> = Vec::new();
        let mut cancelled_root_threads: Vec<ThreadId> = Vec::new();
        // Terminal `Cancelled` markers (issue #354): one per blocking
        // root this call transitions (pre-cancel occupant of its
        // thread's active-root slot), on that root's OWN thread —
        // cascade-cancelled child-thread roots included, queued roots
        // never. Atomic with the cancel transitions: a crash can
        // neither lose the marker nor emit it without the
        // cancellation, and an idempotent retry (terminal tree) emits
        // nothing.
        let mut markers: Vec<CommittedEvent> = Vec::new();
        for row in all_tasks {
            if row.status.is_terminal() {
                continue;
            }
            let is_root_turn_root = row.kind == TaskKind::RootTurn && row.is_root();
            let thread_id = row.thread_id.clone();
            if is_root_turn_root && row.status.blocks_root_admission() {
                let continuation_usage = row
                    .state
                    .continuation()
                    .map(|continuation| continuation.payload.total_usage.clone());
                let committed =
                    Self::insert_cancelled_marker_tx(&mut tx, &thread_id, continuation_usage, now)
                        .await?;
                markers.push(committed);
            }
            let cancelled = row
                .cancel(now)
                .context("cancel_tree: cancel transition failed")?;
            Self::update_task_tx(&mut tx, &cancelled).await?;
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
            Self::resume_linked_subagent_invocation_tx(&mut tx, cancelled_child_root, now).await?;
        }

        // Each cancelled root frees its thread's active-root slot; promote
        // the next queued root in the same transaction (emitting its
        // durable wakeup) so cancelling the active root never strands its
        // queued successors.
        for thread_id in &cancelled_root_threads {
            Self::promote_next_queued_root_tx(&mut tx, thread_id, now).await?;
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
            && let Some(cancelled_top) = Self::load_task_tx(&mut tx, root_id).await?
            && cancelled_top.parent_id.is_some()
        {
            Self::propagate_terminal_to_parent_tx(&mut tx, &cancelled_top, now, "cancel_tree")
                .await?;
        }

        tx.commit().await.context("commit cancel_tree")?;
        Ok(CancelTreeOutcome {
            transitioned,
            markers,
        })
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
            .resume_from_confirmation(now)
            .context("resume rejected: resume_from_confirmation transition failed")?;
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
        let (resumed, prepared_operation) = old
            .clone()
            .resume_from_confirmation(now)
            .context("approve rejected: resume_from_confirmation transition failed")?;
        let claimed = resumed
            .mark_running(worker, lease, expires_at, now)
            .context("approve rejected: mark_running transition failed")?;
        Self::update_task_tx(&mut tx, &claimed).await?;
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
        let old = Self::load_task_tx(&mut tx, id)
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
            .fail(error, now)
            .context("reject_confirmation: fail transition failed")?;
        Self::update_task_tx(&mut tx, &failed).await?;

        let parent = if let Some(parent_id) = &failed.parent_id {
            let old_parent = Self::load_task_tx(&mut tx, parent_id)
                .await?
                .ok_or_else(|| {
                    anyhow!("reject_confirmation: child {id} references missing parent {parent_id}")
                })?;
            if old_parent.status == TaskStatus::WaitingOnChildren {
                let live = Self::load_live_child_count_tx(&mut tx, parent_id).await?;
                let new_parent = old_parent
                    .clone()
                    .recompute_pending_children(live, now)
                    .context("reject_confirmation: recompute_pending_children failed")?;
                Self::update_task_tx(&mut tx, &new_parent).await?;
                Some(new_parent)
            } else {
                Some(old_parent)
            }
        } else {
            None
        };

        tx.commit().await.context("commit reject_confirmation")?;
        Ok((failed, parent))
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
    fn atomic_completed_turn_committer(&self) -> Option<&dyn AtomicCompletedTurnCommitter> {
        Some(self)
    }

    fn atomic_fork_committer(&self) -> Option<&dyn AtomicForkCommitter> {
        Some(self)
    }

    async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let thread = Self::get_thread_tx(&mut tx, thread_id)
            .await?
            .context("thread missing after bootstrap")?;
        tx.commit().await.context("commit get_or_create thread")?;
        Ok(thread)
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

    async fn list(&self) -> Result<Vec<Thread>> {
        let records = sqlx::query_as::<_, ThreadRecord>(
            "SELECT thread_id, status, committed_turns, total_input_tokens, total_output_tokens, created_at, updated_at FROM agent_sdk_threads ORDER BY created_at, thread_id",
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
            .map(|p| p.messages)
            .unwrap_or_default())
    }

    async fn commit_messages(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
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
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let projection = Self::get_message_head_tx(&mut tx, thread_id, now).await?;
        let updated = projection.replace_history(messages, now);
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit replace_history")?;
        Ok(updated)
    }

    async fn set_draft(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
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
        let Some(projection) = self.get_message_head_pool(thread_id).await? else {
            // No projection row exists yet — nothing to clear. The
            // commit path treats this as a no-op so first-turn
            // happy-paths don't bootstrap rows just to clear them.
            return Ok(None);
        };
        let updated = projection.clear_draft(now);
        let mut tx = self.begin().await?;
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
       cached_input_tokens, opened_at, closed_at, duration_ms,
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
        let result = sqlx::query!(
            r"DELETE FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 AND id NOT IN (SELECT id FROM agent_sdk_turn_checkpoints WHERE thread_id = ?1 ORDER BY turn_number DESC LIMIT ?2)",
            tid,
            keep,
        )
        .execute(&self.pool)
        .await
        .with_context(|| format!("delete checkpoints beyond limit for {thread_id}"))?;
        Ok(result.rows_affected())
    }
}

// ─────────────────────────────────────────────────────────────────────
// AtomicCompletedTurnCommitter
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl AtomicCompletedTurnCommitter for SqliteDurableStore {
    async fn commit_completed_turn_atomic(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome> {
        self.commit_completed_turn_atomic_inner(params).await
    }
}

#[async_trait]
impl AtomicForkCommitter for SqliteDurableStore {
    async fn commit_fork_atomic(&self, params: ForkCommitParams) -> Result<()> {
        self.commit_fork_atomic_inner(params).await
    }
}

// ─────────────────────────────────────────────────────────────────────
// EventRepository
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl EventRepository for SqliteDurableStore {
    fn atomic_event_outbox_committer(&self) -> Option<&dyn AtomicEventOutboxCommitter> {
        Some(self)
    }

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

// ─────────────────────────────────────────────────────────────────────
// AtomicEventOutboxCommitter
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl AtomicEventOutboxCommitter for SqliteDurableStore {
    async fn commit_events_with_outbox(
        &self,
        params: EventOutboxCommit,
    ) -> Result<EventOutboxCommitOutcome> {
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
        Ok(EventOutboxCommitOutcome {
            committed_events: committed,
            outbox_row: Some(outbox_row),
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
        sqlx::query!("UPDATE agent_sdk_outbox SET status = 'delivered', delivered_at = ?2 WHERE id = ?1 AND claimed_by = ?3", id_str, now, worker_id)
            .execute(&self.pool)
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
        let result = sqlx::query!(
            r"UPDATE agent_sdk_outbox SET status = 'pending', claimed_by = NULL, claimed_at = NULL, next_attempt_at = ?1 WHERE status = 'claimed' AND claimed_at IS NOT NULL AND claimed_at <= ?2",
            now,
            threshold,
        )
        .execute(&self.pool)
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
// ExecutionIntentStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl ExecutionIntentStore for SqliteDurableStore {
    async fn persist_intent(&self, intent: &ExecutionIntent) -> Result<()> {
        let effect_class_wire = enum_to_wire(&intent.effect_class)?;
        let status_wire = enum_to_wire(&intent.status)?;
        let input_json = json_to_value(&intent.input, "execution intent input")?;
        let op_id = intent.operation_id.as_str();
        let child_task = intent.child_task_id.as_str();

        // Atomic claim: insert-if-absent. `ON CONFLICT DO NOTHING` keeps
        // an existing in-flight record intact, and the affected-row count
        // distinguishes the winning claim (1 row) from a concurrent loser
        // (0 rows). This is the database half of the fail-closed
        // double-execution guard — a blind upsert here would let a stale
        // worker downgrade a `Started` record back to `Pending`. A
        // conflict (0 rows) returns an error so `guarded_tool_execution`
        // fails the child closed.
        let result = sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
ON CONFLICT (operation_id) DO NOTHING
",
            op_id,
            effect_class_wire,
            intent.tool_call_id,
            child_task,
            intent.tool_name,
            input_json,
            status_wire,
            intent.error,
            intent.created_at,
            intent.updated_at,
        )
        .execute(&self.pool)
        .await
        .with_context(|| format!("persist execution intent {}", intent.operation_id))?;

        ensure!(
            result.rows_affected() == 1,
            "execution intent for operation {} already claimed (conflict)",
            intent.operation_id,
        );
        Ok(())
    }

    async fn update_intent(&self, intent: &ExecutionIntent) -> Result<()> {
        let status_wire = enum_to_wire(&intent.status)?;
        let op_id = intent.operation_id.as_str();

        let result = sqlx::query!(
            r"
UPDATE agent_sdk_execution_intents
SET status     = ?2,
    error      = ?3,
    updated_at = ?4
WHERE operation_id = ?1
",
            op_id,
            status_wire,
            intent.error,
            intent.updated_at,
        )
        .execute(&self.pool)
        .await
        .with_context(|| format!("update execution intent {}", intent.operation_id))?;

        ensure!(
            result.rows_affected() == 1,
            "update execution intent affected {} rows for {}",
            result.rows_affected(),
            intent.operation_id
        );
        Ok(())
    }

    async fn get_intent(&self, operation_id: &OperationId) -> Result<Option<ExecutionIntent>> {
        let op_id = operation_id.as_str();
        let record = sqlx::query_as::<_, ExecutionIntentRecord>(
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
FROM agent_sdk_execution_intents
WHERE operation_id = ?1
",
        )
        .bind(op_id)
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get execution intent {operation_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_intent_by_task(
        &self,
        child_task_id: &AgentTaskId,
    ) -> Result<Option<ExecutionIntent>> {
        let task_id = child_task_id.as_str();
        let record = sqlx::query_as::<_, ExecutionIntentRecord>(
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
FROM agent_sdk_execution_intents
WHERE child_task_id = ?1
",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get execution intent by task {child_task_id}"))?;
        record.map(TryInto::try_into).transpose()
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
    pending_child_count: i64,
    spawn_index: Option<i64>,
    result_payload: Option<serde_json::Value>,
    otel_traceparent: Option<String>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
    completed_at: Option<OffsetDateTime>,
}

impl TryFrom<TaskRecord> for AgentTask {
    type Error = anyhow::Error;
    fn try_from(r: TaskRecord) -> Result<Self> {
        let task = Self {
            id: AgentTaskId::from_string(r.id),
            kind: enum_from_wire(&r.kind, "task kind")?,
            status: enum_from_wire(&r.status, "task status")?,
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
            state: json_from_value(r.state_json, "task state")?,
            attempt: u32_from_i64(r.attempt, "task attempt")?,
            max_attempts: u32_from_i64(r.max_attempts, "task max_attempts")?,
            last_error: r.last_error,
            pending_child_count: u32_from_i64(r.pending_child_count, "task pending_child_count")?,
            spawn_index: r
                .spawn_index
                .map(|v| u32_from_i64(v, "task spawn_index"))
                .transpose()?,
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
}

impl TryFrom<ThreadRecord> for Thread {
    type Error = anyhow::Error;
    fn try_from(r: ThreadRecord) -> Result<Self> {
        let thread = Self {
            thread_id: ThreadId::from_string(r.thread_id),
            status: enum_from_wire(&r.status, "thread status")?,
            committed_turns: u32_from_i64(r.committed_turns, "thread committed_turns")?,
            total_usage: TokenUsage {
                input_tokens: u32_from_i64(r.total_input_tokens, "thread input tokens")?,
                output_tokens: u32_from_i64(r.total_output_tokens, "thread output tokens")?,
                ..Default::default()
            },
            created_at: r.created_at,
            updated_at: r.updated_at,
        };
        thread
            .validate()
            .context("rehydrate thread from sqlite row validation")?;
        Ok(thread)
    }
}

#[derive(Debug, FromRow)]
struct MessageHeadRecord {
    thread_id: String,
    history_json: serde_json::Value,
    /// In-flight draft snapshot, NULL when no turn is suspended.
    /// Populated by the worker at every tool-boundary suspension and
    /// cleared atomically by `commit_completed_turn_atomic_inner`.
    draft_messages_json: Option<serde_json::Value>,
    version: i64,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
}

impl TryFrom<MessageHeadRecord> for MessageProjection {
    type Error = anyhow::Error;
    fn try_from(r: MessageHeadRecord) -> Result<Self> {
        let draft_messages = match r.draft_messages_json {
            Some(value) => json_from_value(value, "message head draft messages")?,
            None => Vec::new(),
        };
        Ok(Self {
            thread_id: ThreadId::from_string(r.thread_id),
            messages: json_from_value(r.history_json, "message head history")?,
            draft_messages,
            version: u64_from_i64(r.version, "message head version")?,
            created_at: r.created_at,
            updated_at: r.updated_at,
        })
    }
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
struct ExecutionIntentRecord {
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
}

impl TryFrom<ExecutionIntentRecord> for ExecutionIntent {
    type Error = anyhow::Error;
    fn try_from(r: ExecutionIntentRecord) -> Result<Self> {
        Ok(Self {
            operation_id: OperationId(r.operation_id),
            effect_class: enum_from_wire(&r.effect_class, "execution intent effect_class")?,
            tool_call_id: r.tool_call_id,
            child_task_id: AgentTaskId::from_string(r.child_task_id),
            tool_name: r.tool_name,
            input: r.input,
            status: enum_from_wire(&r.status, "execution intent status")?,
            error: r.error,
            created_at: r.created_at,
            updated_at: r.updated_at,
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
    use anyhow::{Context, Result};
    use time::Duration;

    use agent_sdk_foundation::ThreadId;
    use agent_server::journal::execution_intent::{
        ExecutionIntent, ExecutionIntentStore, IntentStatus, OperationId, ToolEffectClass,
    };
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::task::AgentTaskId;

    use super::SqliteDurableStore;

    fn t0() -> time::OffsetDateTime {
        time::OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> time::OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn make_test_intent(task_name: &str, secs: i64) -> ExecutionIntent {
        let task_id = AgentTaskId::from_string(format!("task_{task_name}"));
        let op_id = OperationId::new(&task_id, "call_1");
        ExecutionIntent {
            operation_id: op_id,
            effect_class: ToolEffectClass::SideEffecting,
            tool_call_id: "call_1".into(),
            child_task_id: task_id,
            tool_name: "test_tool".into(),
            input: serde_json::json!({"key": "value"}),
            status: IntentStatus::Pending,
            error: None,
            created_at: t_plus(secs),
            updated_at: t_plus(secs),
        }
    }

    #[tokio::test]
    async fn execution_intent_persist_and_get_by_operation_id() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let intent = make_test_intent("persist_op", 10);
        store.persist_intent(&intent).await?;

        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("intent should be present after persist")?;
        assert_eq!(loaded.operation_id, intent.operation_id);
        assert_eq!(loaded.tool_name, "test_tool");
        assert_eq!(loaded.status, IntentStatus::Pending);

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_persist_and_get_by_task() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let intent = make_test_intent("persist_task", 11);
        store.persist_intent(&intent).await?;

        let loaded = store
            .get_intent_by_task(&intent.child_task_id)
            .await?
            .context("intent should be present by task id")?;
        assert_eq!(loaded.child_task_id, intent.child_task_id);
        assert_eq!(loaded.operation_id, intent.operation_id);

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_update_transitions_status() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let mut intent = make_test_intent("update_status", 12);
        store.persist_intent(&intent).await?;

        intent.mark_started(t_plus(13));
        store.update_intent(&intent).await?;
        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("should exist after update to started")?;
        assert_eq!(loaded.status, IntentStatus::Started);

        intent.mark_completed(t_plus(14));
        store.update_intent(&intent).await?;
        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("should exist after update to completed")?;
        assert_eq!(loaded.status, IntentStatus::Completed);

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_failed_stores_error() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let mut intent = make_test_intent("failed_error", 15);
        store.persist_intent(&intent).await?;

        intent.mark_started(t_plus(16));
        store.update_intent(&intent).await?;

        intent.mark_failed("something went wrong", t_plus(17));
        store.update_intent(&intent).await?;

        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("should exist after failed")?;
        assert_eq!(loaded.status, IntentStatus::Failed);
        assert_eq!(loaded.error.as_deref(), Some("something went wrong"));

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_restart_recovery() -> Result<()> {
        // Use a file-backed database so the data persists across connections.
        let db_path = std::env::temp_dir().join(format!(
            "agent_sdk_intent_restart_{}.db",
            uuid::Uuid::new_v4().simple()
        ));
        let url = format!("sqlite://{}?mode=rwc", db_path.display());

        let store = SqliteDurableStore::connect(&url).await?;
        let intent = make_test_intent("restart", 20);
        store.persist_intent(&intent).await?;
        drop(store);

        // Reconnect — simulates restart.
        let store2 = SqliteDurableStore::connect(&url).await?;
        let loaded = store2
            .get_intent(&intent.operation_id)
            .await?
            .context("intent should survive reconnection")?;
        assert_eq!(loaded.operation_id, intent.operation_id);

        let loaded_by_task = store2
            .get_intent_by_task(&intent.child_task_id)
            .await?
            .context("intent should be findable by task after reconnect")?;
        assert_eq!(loaded_by_task.operation_id, intent.operation_id);

        drop(store2);
        let _ = std::fs::remove_file(&db_path);

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_duplicate_persist_is_insert_if_absent() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let mut intent = make_test_intent("claim", 25);
        store.persist_intent(&intent).await?;

        // A second claim on the same operation id must lose with a
        // conflict error and must NOT clobber the in-flight record back
        // to a weaker status. Transitions go through `update_intent`.
        intent.mark_started(t_plus(26));
        let err = store.persist_intent(&intent).await.unwrap_err();
        assert!(
            err.to_string().contains("conflict"),
            "expected claim conflict, got: {err}"
        );

        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("should exist after claim")?;
        assert_eq!(loaded.status, IntentStatus::Pending);

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_get_nonexistent_returns_none() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let op_id = OperationId("nonexistent:call".into());
        assert!(store.get_intent(&op_id).await?.is_none());
        assert!(
            store
                .get_intent_by_task(&AgentTaskId::from_string("no_such_task"))
                .await?
                .is_none()
        );

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_clear_removes_rows() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;

        let intent = make_test_intent("clear", 30);
        store.persist_intent(&intent).await?;

        AgentTaskStore::clear(&store).await?;

        assert!(store.get_intent(&intent.operation_id).await?.is_none());
        assert!(
            store
                .get_intent_by_task(&intent.child_task_id)
                .await?
                .is_none()
        );

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
    async fn commit_events_with_outbox_emits_one_thread_events_advisory_per_batch() -> Result<()> {
        use agent_sdk_foundation::ThreadId;
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-coalesce");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![
                    AgentEvent::text("msg_a", "first"),
                    AgentEvent::text("msg_b", "second"),
                    AgentEvent::text("msg_c", "third"),
                ],
                outbox_max_attempts: 5,
                now: t0(),
            },
        )
        .await?;

        assert_eq!(outcome.committed_events.len(), 3);
        let row = outcome
            .outbox_row
            .as_ref()
            .context("Phase 8.1 contract: outbox_row must be present")?;
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
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-delivered-missing");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            },
        )
        .await?;
        let row_id = outcome
            .outbox_row
            .context("outbox row should exist for committed event batch")?
            .id;

        RetentionStore::advance_floor(&store, &thread_id, 1, t_plus(1)).await?;
        assert!(OutboxStore::get(&store, &row_id).await?.is_none());

        OutboxStore::mark_delivered(&store, &row_id, "relay-worker", t_plus(2)).await?;
        Ok(())
    }

    #[tokio::test]
    async fn outbox_mark_failed_is_noop_when_row_was_cascade_deleted() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-failed-missing");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let outcome = AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            },
        )
        .await?;
        let row_id = outcome
            .outbox_row
            .context("outbox row should exist for committed event batch")?
            .id;

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
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::{NewOutboxRow, OutboxStore};
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, TaskWakeupPayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-outbox-order");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            },
        )
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
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::{OutboxStatus, OutboxStore};
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-reclaim-stale");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_a", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            },
        )
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
        use agent_server::journal::event_outbox_transaction::{
            AtomicEventOutboxCommitter, EventOutboxCommit,
        };
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::thread_store::ThreadStore;

        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("t-sqlite-reclaim-live");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        AtomicEventOutboxCommitter::commit_events_with_outbox(
            &store,
            EventOutboxCommit {
                thread_id: thread_id.clone(),
                events: vec![AgentEvent::text("msg_live", "first")],
                outbox_max_attempts: 3,
                now: t0(),
            },
        )
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
        use agent_server::journal::completed_turn_transaction::AtomicCompletedTurnCommitter;
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
        let commit = AtomicCompletedTurnCommitter::commit_completed_turn_atomic(
            &store,
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
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
        use agent_server::journal::completed_turn_transaction::AtomicCompletedTurnCommitter;
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::OutboxMessageKind;
        use agent_server::journal::task::{AgentTask, TaskStatus};
        use agent_server::journal::thread_store::ThreadStore;
        use agent_server::journal::turn_attempt::{
            CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome,
        };
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

        let outcome = AtomicCompletedTurnCommitter::commit_completed_turn_atomic(
            &store,
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
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
                },
                messages: vec![llm::Message::user("hi"), llm::Message::assistant("there")],
                turn_usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                    ..Default::default()
                },
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: vec![AgentEvent::Start {
                    thread_id: thread_id.clone(),
                    turn: 1,
                }],
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

    /// `AtomicForkCommitter` on `SQLite` must commit thread aggregate
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
        use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};
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
        let events = vec![
            AgentEvent::start(source_thread_id.clone(), 1),
            AgentEvent::done(
                source_thread_id.clone(),
                1,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            ),
        ];
        let params = ForkCommitParams {
            new_thread_id: new_thread_id.clone(),
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
        };

        store.commit_fork_atomic(params).await?;

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
        assert_eq!(committed.len(), 2);
        let sequences: Vec<_> = committed.iter().map(|c| c.sequence).collect();
        assert_eq!(
            sequences,
            vec![0, 1],
            "events committed with fresh sequences"
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
        use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};
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
            new_thread_id: new_thread_id.clone(),
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
            events: Vec::new(),
        };
        store.commit_fork_atomic(params).await?;

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
        use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};
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
            new_thread_id: new_thread_id.clone(),
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
        };
        let err = store
            .commit_fork_atomic(params)
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
}
