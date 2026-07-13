//! `sqlx`-backed Postgres implementation of the durable-core stores.
//!
//! This backend keeps the durable semantics aligned with the in-memory
//! reference model by reusing the existing pure Rust state-transition
//! helpers (`AgentTask::mark_running`, `Thread::apply_committed_turn`,
//! `TurnAttempt::close`, and so on) and letting Postgres provide row
//! locking, uniqueness constraints, and transaction scope.
//!
//! The same concrete type implements the task, thread, message,
//! turn-attempt, and checkpoint store traits. That lets the thread
//! store surface a backend-specific
//! [`agent_server::journal::AtomicCompletedTurnCommitter`] hook so
//! `commit_completed_turn` can collapse the attempt close, thread
//! aggregate advance, message head/raw-batch write, and checkpoint
//! insert into one SQL transaction.

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{
    ContinuationEnvelope, ListenExecutionContext, ThreadId, TokenUsage, llm,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
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
use agent_server::journal::outbox::kind_payload_invariants_hold;
use agent_server::journal::outbox::{
    NewOutboxRow, OutboxRow, OutboxRowId, OutboxStatus, OutboxStore,
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
    AgentTaskStore, CancelTreeOutcome, MixedChildrenSpawn, RequeueOutcome, SpawnedMixedChildren,
    SubagentInvocationSpawn, SubmitRootIdempotency, SubmitRootTurnError, SubmitRootTurnOutcome,
    SubmitRootTurnParams, mixed_child_ids_in_slot_order, new_mixed_tool_child,
    validate_mixed_children_spawn,
};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskKind, TaskStatus, WorkerId,
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

/// `sqlx`-backed durable-core store.
#[derive(Clone)]
pub struct PostgresDurableStore {
    pool: PgPool,
}

impl PostgresDurableStore {
    /// Construct the store from an existing pool.
    #[must_use]
    pub const fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Build a lazily connecting Postgres pool.
    ///
    /// The returned store does not perform network I/O until first
    /// use. Call [`Self::migrate`] during startup to verify the
    /// database is reachable and the durable-core schema is ready.
    ///
    /// # Errors
    ///
    /// Returns an error if the database URL is invalid or the schema
    /// name is malformed.
    pub fn connect_lazy(
        database_url: &str,
        max_connections: u32,
        schema: Option<&str>,
    ) -> Result<Self> {
        ensure!(max_connections > 0, "postgres max_connections must be > 0");

        let connect_options: PgConnectOptions = database_url
            .parse()
            .context("parse postgres durable store connection string")?;

        let mut pool_options = PgPoolOptions::new().max_connections(max_connections);

        if let Some(schema_name) = schema {
            validate_schema_name(schema_name)?;
            let schema_name = schema_name.to_owned();
            pool_options = pool_options.after_connect(move |conn, _meta| {
                let schema_name = schema_name.clone();
                Box::pin(async move {
                    // Time the per-connection setup
                    // (search_path config) so the host can publish
                    // `db.client.connections.create_time`. sqlx 0.8
                    // does not expose a true `before_connect` hook;
                    // this captures the post-handshake setup latency,
                    // which is the closest signal an operator can act
                    // on without forking sqlx.
                    #[cfg(feature = "otel")]
                    let started_at = std::time::Instant::now();

                    let _ = sqlx::query_scalar!(
                        "SELECT pg_catalog.set_config('search_path', $1, false)",
                        schema_name,
                    )
                    .fetch_one(conn)
                    .await?;

                    #[cfg(feature = "otel")]
                    crate::observability::HostMetrics::global()
                        .record_db_client_connection_create_time(
                            "agent_service_host.postgres",
                            started_at.elapsed().as_secs_f64(),
                        );
                    Ok(())
                })
            });
        } else {
            // Even when no schema override is configured we still
            // want a `db.client.connections.create_time` sample on
            // every fresh connection. Wrap a no-op probe query (a
            // single-row `SELECT 1` is cheaper than `set_config` and
            // does not allocate on the server).
            #[cfg(feature = "otel")]
            {
                pool_options = pool_options.after_connect(|conn, _meta| {
                    Box::pin(async move {
                        let started_at = std::time::Instant::now();
                        // Probe the connection so the timer captures
                        // a real round-trip, not just future setup.
                        let _ = sqlx::query_scalar::<_, i32>("SELECT 1")
                            .fetch_one(conn)
                            .await?;
                        crate::observability::HostMetrics::global()
                            .record_db_client_connection_create_time(
                                "agent_service_host.postgres",
                                started_at.elapsed().as_secs_f64(),
                            );
                        Ok(())
                    })
                });
            }
        }

        Ok(Self::from_pool(
            pool_options.connect_lazy_with(connect_options),
        ))
    }

    /// Connect to Postgres and apply the durable-core migrations.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot be created or migrations fail.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let store = Self::connect_lazy(database_url, 8, None)?;
        store
            .pool
            .acquire()
            .await
            .context("connect postgres durable store")?;
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
    pub const fn pool(&self) -> &PgPool {
        &self.pool
    }

    async fn begin(&self) -> Result<Transaction<'_, Postgres>> {
        self.pool
            .begin()
            .await
            .context("begin postgres durable-core transaction")
    }

    async fn bootstrap_thread_row_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<()> {
        let bootstrap = Thread::new(thread_id.clone(), now);
        sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id,
    status,
    committed_turns,
    total_input_tokens,
    total_output_tokens,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (thread_id) DO NOTHING
",
            thread_key(&bootstrap.thread_id),
            enum_to_wire(&bootstrap.status)?,
            i64::from(bootstrap.committed_turns),
            i64::from(bootstrap.total_usage.input_tokens),
            i64::from(bootstrap.total_usage.output_tokens),
            bootstrap.created_at,
            bootstrap.updated_at,
        )
        .execute(&mut **tx)
        .await
        .context("bootstrap thread row")?;
        Ok(())
    }

    async fn lock_thread_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
    ) -> Result<Option<Thread>> {
        let record = sqlx::query_as!(
            ThreadRecord,
            r"
SELECT
    thread_id,
    status,
    committed_turns,
    total_input_tokens,
    total_output_tokens,
    created_at,
    updated_at
FROM agent_sdk_threads
WHERE thread_id = $1
FOR UPDATE
",
            thread_key(thread_id),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("lock thread {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_thread_pool(&self, thread_id: &ThreadId) -> Result<Option<Thread>> {
        let record = sqlx::query_as!(
            ThreadRecord,
            r"
SELECT
    thread_id,
    status,
    committed_turns,
    total_input_tokens,
    total_output_tokens,
    created_at,
    updated_at
FROM agent_sdk_threads
WHERE thread_id = $1
",
            thread_key(thread_id),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get thread {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn upsert_thread_tx(tx: &mut Transaction<'_, Postgres>, thread: &Thread) -> Result<()> {
        sqlx::query!(
            r"
INSERT INTO agent_sdk_threads (
    thread_id,
    status,
    committed_turns,
    total_input_tokens,
    total_output_tokens,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (thread_id) DO UPDATE SET
    status = EXCLUDED.status,
    committed_turns = EXCLUDED.committed_turns,
    total_input_tokens = EXCLUDED.total_input_tokens,
    total_output_tokens = EXCLUDED.total_output_tokens,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at
",
            thread_key(&thread.thread_id),
            enum_to_wire(&thread.status)?,
            i64::from(thread.committed_turns),
            i64::from(thread.total_usage.input_tokens),
            i64::from(thread.total_usage.output_tokens),
            thread.created_at,
            thread.updated_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("upsert thread {}", thread.thread_id))?;
        Ok(())
    }

    async fn lock_message_head_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        // Bootstrap with `draft_messages_json = NULL` — a fresh
        // thread has no in-flight turn. The full upsert path below
        // populates the column once a suspension boundary fires.
        sqlx::query!(
            r"
INSERT INTO agent_sdk_message_heads (
    thread_id,
    history_json,
    draft_messages_json,
    version,
    created_at,
    updated_at
) VALUES ($1, $2, NULL, $3, $4, $5)
ON CONFLICT (thread_id) DO NOTHING
",
            thread_key(thread_id),
            json_to_value(&Vec::<llm::Message>::new(), "empty message history")?,
            0_i64,
            now,
            now,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("bootstrap message head {thread_id}"))?;

        let record = sqlx::query_as!(
            MessageHeadRecord,
            r"
SELECT
    thread_id,
    history_json,
    draft_messages_json,
    version,
    created_at,
    updated_at
FROM agent_sdk_message_heads
WHERE thread_id = $1
FOR UPDATE
",
            thread_key(thread_id),
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("lock message head {thread_id}"))?;
        record.try_into()
    }

    async fn get_message_head_pool(
        &self,
        thread_id: &ThreadId,
    ) -> Result<Option<MessageProjection>> {
        let record = sqlx::query_as!(
            MessageHeadRecord,
            r"
SELECT
    thread_id,
    history_json,
    draft_messages_json,
    version,
    created_at,
    updated_at
FROM agent_sdk_message_heads
WHERE thread_id = $1
",
            thread_key(thread_id),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get message head {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn upsert_message_head_tx(
        tx: &mut Transaction<'_, Postgres>,
        projection: &MessageProjection,
    ) -> Result<()> {
        // Persist `draft_messages` as NULL when empty so the column
        // distinguishes "no in-flight turn" from "explicit empty
        // draft" — matches the intent the migration documents and
        // mirrors the SQLite path.
        let draft_messages_json = if projection.draft_messages.is_empty() {
            None
        } else {
            Some(json_to_value(
                &projection.draft_messages,
                "message head draft messages",
            )?)
        };
        sqlx::query!(
            r"
INSERT INTO agent_sdk_message_heads (
    thread_id,
    history_json,
    draft_messages_json,
    version,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = EXCLUDED.history_json,
    draft_messages_json = EXCLUDED.draft_messages_json,
    version = EXCLUDED.version,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at
",
            thread_key(&projection.thread_id),
            json_to_value(&projection.messages, "message head history")?,
            draft_messages_json,
            i64_from_u64(projection.version, "message head version")?,
            projection.created_at,
            projection.updated_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("upsert message head {}", projection.thread_id))?;
        Ok(())
    }

    async fn insert_message_commit_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        turn_number: u32,
        task_id: &AgentTaskId,
        head_version_after: u64,
        batch: &[llm::Message],
        committed_at: OffsetDateTime,
    ) -> Result<()> {
        sqlx::query!(
            r"
INSERT INTO agent_sdk_message_commits (
    thread_id,
    turn_number,
    task_id,
    head_version_after,
    batch_json,
    committed_at
) VALUES ($1, $2, $3, $4, $5, $6)
",
            thread_key(thread_id),
            i64::from(turn_number),
            task_id.as_str(),
            i64_from_u64(head_version_after, "message commit head_version_after")?,
            json_to_value(batch, "committed message batch")?,
            committed_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| {
            format!("insert raw message batch for thread {thread_id} turn {turn_number}")
        })?;
        Ok(())
    }

    async fn lock_attempt_tx(
        tx: &mut Transaction<'_, Postgres>,
        id: &TurnAttemptId,
    ) -> Result<Option<TurnAttempt>> {
        let record = sqlx::query_as!(
            TurnAttemptRecord,
            r"
SELECT
    id,
    task_id,
    attempt_number,
    provider,
    requested_model,
    request_blob,
    response_blob,
    response_id,
    response_model,
    stop_reason,
    outcome,
    input_tokens,
    output_tokens,
    cached_input_tokens,
    opened_at,
    closed_at,
    duration_ms,
    otel_trace_id,
    otel_span_id
FROM agent_sdk_turn_attempts
WHERE id = $1
FOR UPDATE
",
            id.as_str(),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("lock turn attempt {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_attempt_pool(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        let record = sqlx::query_as!(
            TurnAttemptRecord,
            r"
SELECT
    id,
    task_id,
    attempt_number,
    provider,
    requested_model,
    request_blob,
    response_blob,
    response_id,
    response_model,
    stop_reason,
    outcome,
    input_tokens,
    output_tokens,
    cached_input_tokens,
    opened_at,
    closed_at,
    duration_ms,
    otel_trace_id,
    otel_span_id
FROM agent_sdk_turn_attempts
WHERE id = $1
",
            id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get turn attempt {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn insert_attempt_tx(
        tx: &mut Transaction<'_, Postgres>,
        attempt: &TurnAttempt,
    ) -> Result<()> {
        sqlx::query!(
            r"
INSERT INTO agent_sdk_turn_attempts (
    id,
    task_id,
    attempt_number,
    provider,
    requested_model,
    request_blob,
    response_blob,
    response_id,
    response_model,
    stop_reason,
    outcome,
    input_tokens,
    output_tokens,
    cached_input_tokens,
    opened_at,
    closed_at,
    duration_ms,
    otel_trace_id,
    otel_span_id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9,
    $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
)
",
            attempt.id.as_str(),
            attempt.task_id.as_str(),
            i64::from(attempt.attempt_number),
            attempt.provider,
            attempt.requested_model,
            attempt.request_blob.clone(),
            attempt.response_blob.clone(),
            attempt.response_id.clone(),
            attempt.response_model.clone(),
            optional_enum_to_wire(attempt.stop_reason.as_ref())?,
            optional_enum_to_wire(attempt.outcome.as_ref())?,
            optional_u32_to_i64(attempt.input_tokens),
            optional_u32_to_i64(attempt.output_tokens),
            optional_u32_to_i64(attempt.cached_input_tokens),
            attempt.opened_at,
            attempt.closed_at,
            optional_u64_to_i64(attempt.duration_ms, "attempt duration_ms")?,
            attempt.otel_trace_id.clone(),
            attempt.otel_span_id.clone(),
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert turn attempt {}", attempt.id))?;
        Ok(())
    }

    async fn update_attempt_tx(
        tx: &mut Transaction<'_, Postgres>,
        attempt: &TurnAttempt,
    ) -> Result<()> {
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_turn_attempts
SET
    task_id = $2,
    attempt_number = $3,
    provider = $4,
    requested_model = $5,
    request_blob = $6,
    response_blob = $7,
    response_id = $8,
    response_model = $9,
    stop_reason = $10,
    outcome = $11,
    input_tokens = $12,
    output_tokens = $13,
    cached_input_tokens = $14,
    opened_at = $15,
    closed_at = $16,
    duration_ms = $17,
    otel_trace_id = $18,
    otel_span_id = $19
WHERE id = $1
",
            attempt.id.as_str(),
            attempt.task_id.as_str(),
            i64::from(attempt.attempt_number),
            attempt.provider,
            attempt.requested_model,
            attempt.request_blob.clone(),
            attempt.response_blob.clone(),
            attempt.response_id.clone(),
            attempt.response_model.clone(),
            optional_enum_to_wire(attempt.stop_reason.as_ref())?,
            optional_enum_to_wire(attempt.outcome.as_ref())?,
            optional_u32_to_i64(attempt.input_tokens),
            optional_u32_to_i64(attempt.output_tokens),
            optional_u32_to_i64(attempt.cached_input_tokens),
            attempt.opened_at,
            attempt.closed_at,
            optional_u64_to_i64(attempt.duration_ms, "attempt duration_ms")?,
            attempt.otel_trace_id.clone(),
            attempt.otel_span_id.clone(),
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("update turn attempt {}", attempt.id))?;

        ensure!(
            result.rows_affected() == 1,
            "update turn attempt affected {} rows for {}",
            result.rows_affected(),
            attempt.id
        );
        Ok(())
    }

    async fn lock_checkpoint_by_turn_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        turn_number: u32,
    ) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as!(
            CheckpointRecord,
            r"
SELECT
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
FROM agent_sdk_turn_checkpoints
WHERE thread_id = $1
  AND turn_number = $2
FOR UPDATE
",
            thread_key(thread_id),
            i64::from(turn_number),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("lock checkpoint {thread_id} turn {turn_number}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_checkpoint_pool(&self, id: &CheckpointId) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as!(
            CheckpointRecord,
            r"
SELECT
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
FROM agent_sdk_turn_checkpoints
WHERE id = $1
",
            id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get checkpoint {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn insert_checkpoint_tx(
        tx: &mut Transaction<'_, Postgres>,
        checkpoint: &Checkpoint,
    ) -> Result<()> {
        sqlx::query!(
            r"
INSERT INTO agent_sdk_turn_checkpoints (
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
",
            checkpoint.id.as_str(),
            thread_key(&checkpoint.thread_id),
            i64::from(checkpoint.turn_number),
            checkpoint.task_id.as_str(),
            json_to_value(&checkpoint.messages, "checkpoint messages")?,
            checkpoint.agent_state_snapshot.clone(),
            i64::from(checkpoint.turn_usage.input_tokens),
            i64::from(checkpoint.turn_usage.output_tokens),
            checkpoint.kind.as_str(),
            checkpoint.created_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| {
            format!(
                "insert checkpoint for thread {} turn {}",
                checkpoint.thread_id, checkpoint.turn_number
            )
        })?;
        Ok(())
    }

    async fn load_task_tx(
        tx: &mut Transaction<'_, Postgres>,
        id: &AgentTaskId,
        lock: bool,
    ) -> Result<Option<AgentTask>> {
        let record = if lock {
            sqlx::query_as!(
                TaskRecord,
                r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE id = $1
FOR UPDATE
",
                id.as_str(),
            )
            .fetch_optional(&mut **tx)
            .await
        } else {
            sqlx::query_as!(
                TaskRecord,
                r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE id = $1
",
                id.as_str(),
            )
            .fetch_optional(&mut **tx)
            .await
        }
        .with_context(|| format!("load task {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_task_pool(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE id = $1
",
            id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get task {id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_task_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE id = $1
",
            id.as_str(),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("get task {id} (tx)"))?;
        record.map(TryInto::try_into).transpose()
    }

    /// Count the queued (not active/blocking) root turns on a thread,
    /// inside the supplied transaction.
    async fn queued_root_count_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
    ) -> Result<u32> {
        let count = sqlx::query_scalar!(
            r#"
SELECT COUNT(*) AS "count!"
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status = 'queued'
"#,
            thread_key(thread_id),
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("count queued roots for {thread_id}"))?;
        Ok(u32::try_from(count).unwrap_or(u32::MAX))
    }

    /// Look up an existing `SubmitWork` idempotency record inside the
    /// admission transaction and, if it matches, build the replay
    /// outcome (committing the read transaction). Returns `Ok(None)`
    /// when the key is unused (the caller proceeds to admit) and
    /// [`SubmitRootTurnError::IdempotencyConflict`] on a kind/fingerprint
    /// mismatch.
    async fn try_replay_submit_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        claim: &agent_server::journal::store::SubmitRootIdempotency,
    ) -> std::result::Result<Option<SubmitRootTurnOutcome>, SubmitRootTurnError> {
        let existing = sqlx::query!(
            r#"SELECT kind, fingerprint, result_json FROM agent_sdk_idempotency WHERE request_id = $1"#,
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
        let task_id = row
            .result_json
            .get("task_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("submit idempotency record missing task_id reference"))
            .map_err(SubmitRootTurnError::Other)?;
        let admitted = self
            .get_task_tx(tx, &AgentTaskId::from_string(task_id))
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
            replayed_result: Some(row.result_json),
            queued_depth,
        }))
    }

    /// Insert the `SubmitWork` idempotency record inside the admission
    /// transaction.
    async fn claim_submit_idempotency_tx(
        tx: &mut Transaction<'_, Postgres>,
        claim: &SubmitRootIdempotency,
        created_at: OffsetDateTime,
    ) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES ($1, $2, $3, $4, $5)"#,
            claim.request_id,
            IdempotencyKind::SubmitWork.as_str(),
            claim.fingerprint,
            claim.result_json,
            created_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert idempotency record for {}", claim.request_id))?;
        Ok(())
    }

    /// Admit a fresh root turn, claim its idempotency key, and commit
    /// the transaction. `would_queue` is the admission decision computed
    /// under the thread lock.
    async fn commit_fresh_admission_tx(
        mut tx: Transaction<'_, Postgres>,
        task: AgentTask,
        would_queue: bool,
        idempotency: Option<SubmitRootIdempotency>,
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

        // Phase 10 · D: when the new root is immediately runnable
        // (`Pending`, not parked behind an active/queued root), emit a
        // durable `task_wakeup` advisory row in the SAME transaction so a
        // worker in any process is nudged to run `acquire_next_runnable`
        // even after this host dies. A queued root is parked, so its
        // eventual promotion emits the wakeup instead. This mirrors the
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

    async fn insert_task_tx(tx: &mut Transaction<'_, Postgres>, task: &AgentTask) -> Result<()> {
        sqlx::query!(
            r"
INSERT INTO agent_sdk_tasks (
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
    $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
)
",
            task.id.as_str(),
            enum_to_wire(&task.kind)?,
            enum_to_wire(&task.status)?,
            task.parent_id.as_ref().map(AgentTaskId::as_str),
            task.root_id.as_str(),
            i64::from(task.depth),
            thread_key(&task.thread_id),
            json_to_value(&task.submitted_input, "task submitted input")?,
            task.caller_metadata.clone(),
            task.worker_id.as_ref().map(WorkerId::as_str),
            task.lease_id.as_ref().map(LeaseId::as_str),
            task.lease_expires_at,
            task.last_heartbeat_at,
            json_to_value(&task.state, "task state")?,
            i64::from(task.attempt),
            i64::from(task.max_attempts),
            task.last_error.clone(),
            i64::from(task.pending_child_count),
            task.spawn_index.map(i64::from),
            task.result_payload.clone(),
            task.otel_traceparent.clone(),
            task.created_at,
            task.updated_at,
            task.completed_at,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert task {}", task.id))?;
        Ok(())
    }

    async fn update_task_tx(tx: &mut Transaction<'_, Postgres>, task: &AgentTask) -> Result<()> {
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_tasks
SET
    kind = $2,
    status = $3,
    parent_id = $4,
    root_id = $5,
    depth = $6,
    thread_id = $7,
    submitted_input_json = $8,
    worker_id = $9,
    lease_id = $10,
    lease_expires_at = $11,
    last_heartbeat_at = $12,
    state_json = $13,
    attempt = $14,
    max_attempts = $15,
    last_error = $16,
    pending_child_count = $17,
    spawn_index = $18,
    result_payload = $19,
    created_at = $20,
    updated_at = $21,
    completed_at = $22,
    caller_metadata_json = $23,
    otel_traceparent = $24
WHERE id = $1
",
            task.id.as_str(),
            enum_to_wire(&task.kind)?,
            enum_to_wire(&task.status)?,
            task.parent_id.as_ref().map(AgentTaskId::as_str),
            task.root_id.as_str(),
            i64::from(task.depth),
            thread_key(&task.thread_id),
            json_to_value(&task.submitted_input, "task submitted input")?,
            task.worker_id.as_ref().map(WorkerId::as_str),
            task.lease_id.as_ref().map(LeaseId::as_str),
            task.lease_expires_at,
            task.last_heartbeat_at,
            json_to_value(&task.state, "task state")?,
            i64::from(task.attempt),
            i64::from(task.max_attempts),
            task.last_error.clone(),
            i64::from(task.pending_child_count),
            task.spawn_index.map(i64::from),
            task.result_payload.clone(),
            task.created_at,
            task.updated_at,
            task.completed_at,
            task.caller_metadata.clone(),
            task.otel_traceparent.clone(),
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

    async fn lock_parent_for_child_tx(
        tx: &mut Transaction<'_, Postgres>,
        task: &AgentTask,
    ) -> Result<Option<AgentTask>> {
        let Some(parent_id) = &task.parent_id else {
            return Ok(None);
        };
        Self::load_task_tx(tx, parent_id, true).await
    }

    async fn enforce_insert_cross_row_invariants_tx(
        tx: &mut Transaction<'_, Postgres>,
        task: &AgentTask,
    ) -> Result<()> {
        task.validate()
            .context("insert rejected: task failed schema validation")?;

        Self::bootstrap_thread_row_tx(tx, &task.thread_id, task.created_at).await?;

        if let Some(parent) = Self::lock_parent_for_child_tx(tx, task).await? {
            if parent.kind.is_leaf() {
                let parent_id = task.parent_id.as_ref().context("parent id")?;
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
        } else if task.parent_id.is_some() {
            let parent_id = task.parent_id.as_ref().context("parent id")?;
            return Err(anyhow!(
                "insert rejected: child task references unknown parent {parent_id}"
            ));
        }

        if task.kind == TaskKind::RootTurn {
            Self::bootstrap_thread_row_tx(tx, &task.thread_id, task.created_at).await?;
            let _ = Self::lock_thread_tx(tx, &task.thread_id).await?;
            if task.status.blocks_root_admission() {
                let existing = sqlx::query_scalar!(
                    r"
SELECT id
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
  AND id <> $2
LIMIT 1
FOR UPDATE
",
                    thread_key(&task.thread_id),
                    task.id.as_str(),
                )
                .fetch_optional(&mut **tx)
                .await
                .with_context(|| {
                    format!("check blocking root slot for thread {}", task.thread_id)
                })?;
                if let Some(existing) = existing {
                    return Err(anyhow!(
                        "insert rejected: thread {} already has active root task {}",
                        task.thread_id,
                        existing
                    ));
                }
            }
        }

        Ok(())
    }

    /// Enforce [`CompletedTurnCommit::owner_guard`] on the locked task
    /// row inside the completed-turn transaction: the slot-shift retry
    /// validates that the presenting worker still owns a live `Running`
    /// row, so a cancellation / lease loss that landed between the
    /// caller's shift-eligibility check and the retry rejects here —
    /// inside the transaction — instead of splicing a dead root's turn
    /// into the shifted slot. A `None` guard (every first, non-shifted
    /// commit) is a no-op.
    async fn enforce_commit_owner_guard_tx(
        tx: &mut Transaction<'_, Postgres>,
        task_id: &AgentTaskId,
        owner_guard: Option<&CommitOwnerGuard>,
    ) -> Result<()> {
        let Some(guard) = owner_guard else {
            return Ok(());
        };
        let current = Self::load_task_tx(tx, task_id, true)
            .await?
            .ok_or_else(|| {
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

    /// # WARNING: inverted lock order — do not extend
    ///
    /// This helper locks the TASK row first (`load_task_tx(.., true)`
    /// below) and only then, for root turns, the THREAD row — the
    /// reverse of the thread→task order every production write path
    /// observes (`submit_root_turn`, `cancel_tree`, the owner-guarded
    /// completed-turn commit). It is reachable only through
    /// [`AgentTaskStore::update`], the structural rehydration / test
    /// primitive that worker code must never use for lifecycle
    /// transitions, so the inversion is not on any concurrent
    /// production path today. Do NOT route new runtime paths through
    /// `update`, and do not add thread-row locking after task-row
    /// locking elsewhere: pairing this order with any thread→task
    /// path on live traffic is a 40P01 deadlock. Fixing the order
    /// here would require locking the thread row before knowing the
    /// row's kind (or re-reading), which is not worth it for a
    /// test-only surface — hence this fence instead.
    async fn validate_update_row_invariants_tx(
        tx: &mut Transaction<'_, Postgres>,
        task: &AgentTask,
    ) -> Result<AgentTask> {
        task.validate()
            .context("update rejected: task failed schema validation")?;

        let old = Self::load_task_tx(tx, &task.id, true)
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

        if task.kind == TaskKind::RootTurn {
            let _ = Self::lock_thread_tx(tx, &task.thread_id).await?;
            if task.status.blocks_root_admission() {
                let current = sqlx::query_scalar!(
                    r"
SELECT id
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
  AND id <> $2
LIMIT 1
FOR UPDATE
",
                    thread_key(&task.thread_id),
                    task.id.as_str(),
                )
                .fetch_optional(&mut **tx)
                .await
                .with_context(|| {
                    format!("check competing root slot for thread {}", task.thread_id)
                })?;
                if let Some(current) = current {
                    return Err(anyhow!(
                        "update rejected: thread {} already has a different active root task {}",
                        task.thread_id,
                        current
                    ));
                }
            }
        }

        Ok(old)
    }

    async fn load_live_child_count_tx(
        tx: &mut Transaction<'_, Postgres>,
        parent_id: &AgentTaskId,
    ) -> Result<u32> {
        let live = sqlx::query_scalar!(
            r#"
SELECT COUNT(*) AS "count!"
FROM agent_sdk_tasks
WHERE parent_id = $1
  AND status NOT IN ('completed', 'failed', 'cancelled')
"#,
            parent_id.as_str(),
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("count live children for {parent_id}"))?;
        u32_from_i64(live, "live child count")
    }

    /// After a child task transitions to a terminal state, wake its
    /// `WaitingOnChildren` parent (if any) by recomputing
    /// `pending_child_count`.  Without this, a parent with a single
    /// fail-closed child stays in `WaitingOnChildren` forever
    /// because no future `complete_task`/`fail_task` call will fire
    /// to decrement the counter — a liveness deadlock.
    async fn propagate_terminal_to_parent_tx(
        tx: &mut Transaction<'_, Postgres>,
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
            // recovery sweep / lease-expiry fail-close) does not leave
            // the parent stuck in `WaitingOnChildren` forever (a durable
            // liveness deadlock on the very crash-recovery path the
            // durable backend exists for). Also promote this thread's
            // next queued root in the same transaction so a sweep-failed
            // root never strands its queued successors.
            if child.kind == TaskKind::RootTurn && child.is_root() {
                Self::resume_linked_subagent_invocation_tx(tx, &child.id, now).await?;
                Self::promote_next_queued_root_tx(tx, &child.thread_id, now).await?;
            }
            return Ok(());
        };
        let old_parent = Self::load_task_tx(tx, parent_id, true)
            .await?
            .ok_or_else(|| {
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

    async fn apply_task_terminal_transition_tx(
        tx: &mut Transaction<'_, Postgres>,
        child_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        now: OffsetDateTime,
        error_prefix: &'static str,
        transition: impl FnOnce(AgentTask) -> Result<AgentTask, agent_server::journal::TaskSchemaError>,
    ) -> Result<(AgentTask, Option<AgentTask>)> {
        let old_child = Self::load_task_tx(tx, child_id, true)
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
            let old_parent = Self::load_task_tx(tx, parent_id, true)
                .await?
                .ok_or_else(|| {
                    anyhow!(
                        "{error_prefix}: child {child_id} references missing parent {parent_id}"
                    )
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
            // `state.subagent_invocation.child_root_task_id`. Mirror
            // the in-memory store's `resume_linked_subagent_invocation`
            // call here — without it the durable daemon leaves the
            // invocation stuck in `WaitingOnChildren` after the child
            // thread completes, `execute_subagent_task` never runs,
            // and the parent thread's `SubagentProgress { completed:
            // true }` event never fires.
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

    /// Phase 7.6: find a `Subagent` invocation task that is
    /// `WaitingOnChildren` and whose `SubagentInvocation` state
    /// links to the given `child_root_id`, then wake it to `Pending`
    /// via `recompute_pending_children(0, now)`.
    ///
    /// This mirrors the in-memory store's
    /// `resume_linked_subagent_invocation` and ensures that when a
    /// child-thread root reaches a terminal state (cancelled, failed,
    /// completed), the parent-thread invocation is unblocked.
    ///
    /// Uses `FOR UPDATE SKIP LOCKED` to prevent the circular wait that
    /// would otherwise occur when `cancel_tree(parent_root)` and
    /// `cancel_tree(child_root)` run concurrently: the parent's BFS
    /// locks the invocation row first and then follows the linkage to
    /// the child-root, while the child's cancel path locks the
    /// child-root first and then reaches for the invocation row. If
    /// the invocation is already locked by a concurrent parent
    /// cancellation, that transaction will cancel the invocation
    /// outright, so skipping the wakeup here is semantically
    /// equivalent and cannot leave the invocation stuck.
    async fn resume_linked_subagent_invocation_tx(
        tx: &mut Transaction<'_, Postgres>,
        child_root_id: &AgentTaskId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let maybe_record = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE kind = 'subagent'
  AND status = 'waiting_on_children'
  AND state_json -> 'invocation' ->> 'child_root_task_id' = $1
FOR UPDATE SKIP LOCKED
",
            child_root_id.as_str(),
        )
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
    /// (Phase 10 · D — a cross-process worker is nudged even after host
    /// death).
    ///
    /// Shared by [`AgentTaskStore::promote_next_queued_root`] and every
    /// in-transaction root-turn terminal path (terminal CAS, recovery
    /// fail-close, cancel) so a queued root is never stranded behind a
    /// root that reached a terminal state without the host's post-commit
    /// promotion firing.
    async fn promote_next_queued_root_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<AgentTask>> {
        let blocking_root = sqlx::query_scalar!(
            r"
SELECT id
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
LIMIT 1
FOR UPDATE
",
            thread_key(thread_id),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("check blocking root for {thread_id}"))?;
        if blocking_root.is_some() {
            return Ok(None);
        }

        let queued = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status = 'queued'
ORDER BY created_at, id
LIMIT 1
FOR UPDATE
",
            thread_key(thread_id),
        )
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
    /// reference store) instead of silently no-opping.
    ///
    /// Reads are **non-locking** (no `FOR UPDATE`): the cancel path takes
    /// its per-row locks at `UPDATE` time in a deterministic deepest-first
    /// order, so it acquires child locks before parent locks — the same
    /// order as the `complete_task` / `fail_task` terminal paths. That
    /// consistent ordering removes the lock-order inversion that could
    /// deadlock (Postgres SQLSTATE 40P01) a concurrent
    /// `cancel_tree(root)` + `complete_task(child)`.
    async fn collect_subtree_tx(
        tx: &mut Transaction<'_, Postgres>,
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
            let Some(task) = Self::load_task_tx(tx, &id, false).await? else {
                continue;
            };
            if let Some(invocation) = task.state.subagent_invocation() {
                let child_root = invocation.child_root_task_id.clone();
                if !visited.contains(&child_root) {
                    frontier.push_back(child_root);
                }
            }
            let child_ids = sqlx::query_scalar!(
                "SELECT id FROM agent_sdk_tasks WHERE parent_id = $1 ORDER BY created_at, id",
                id.as_str(),
            )
            .fetch_all(&mut **tx)
            .await
            .with_context(|| format!("load children of {id}"))?;
            for child in child_ids {
                let child_id = AgentTaskId::from_string(child);
                if !visited.contains(&child_id) {
                    frontier.push_back(child_id);
                }
            }
            out.push(task);
        }
        Ok(out)
    }

    #[cfg(test)]
    async fn commit_completed_turn_atomic_with_failure(
        &self,
        params: CompletedTurnCommit,
        fail_after: Option<InjectedCommitFailure>,
    ) -> Result<CommitOutcome> {
        self.commit_completed_turn_atomic_inner(params, fail_after)
            .await
    }

    /// Atomic fork transaction for Postgres (mirror of
    /// `SqliteDurableStore::commit_fork_atomic_inner`).
    ///
    /// Wraps the entire fork write set (thread aggregate bootstrap +
    /// `committed_turns` mirroring + projection rewrite + checkpoint
    /// insert + event re-commit) inside one transaction. A crash or
    /// rolled-back transaction leaves the destination thread in the
    /// not-created state — never partially-built — so the gRPC handler's
    /// idempotency replay can safely retry under the same `request_id`
    /// without seeing a half-finished fork (which would otherwise inflate
    /// `committed_turns` / `total_usage` on replay or hit a checkpoint
    /// uniqueness conflict).
    async fn commit_fork_atomic_inner(&self, params: ForkCommitParams) -> Result<()> {
        let mut tx = self.begin().await?;

        // 1. Bootstrap the destination thread aggregate, exactly as
        //    `CreateThread` / `commit_completed_turn` would.
        Self::bootstrap_thread_row_tx(&mut tx, &params.new_thread_id, params.now).await?;

        // 2. Mirror `committed_turns` by repeatedly applying
        //    `Thread::apply_committed_turn`. Only the final iteration
        //    carries the full `cumulative_total_usage` so the
        //    destination lands at exactly the source's snapshot total.
        let mut thread_row = Self::lock_thread_tx(&mut tx, &params.new_thread_id)
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
                    .context(
                        "advancing forked thread aggregate inside postgres fork transaction",
                    )?;
            }
            Self::upsert_thread_tx(&mut tx, &thread_row).await?;
        }

        // 3. Seed the projection via `replace_history` so the bumped
        //    `version` is consistent with `recover_thread`'s expectations.
        if !params.messages.is_empty() {
            let projection_before =
                Self::lock_message_head_tx(&mut tx, &params.new_thread_id, params.now).await?;
            let updated_projection =
                MessageProjection::replace_history(projection_before, params.messages, params.now);
            Self::upsert_message_head_tx(&mut tx, &updated_projection).await?;
        }

        // 4. Mirror the source's checkpoint at the fork-point turn.
        if let Some(checkpoint_params) = params.checkpoint {
            let checkpoint =
                Checkpoint::new(checkpoint_params).context("constructing forked checkpoint")?;
            Self::insert_checkpoint_tx(&mut tx, &checkpoint).await?;
        }

        // 5. Re-commit events under the new thread id with fresh
        //    sequences (a freshly-bootstrapped thread starts at 0).
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
            .context("commit postgres fork transaction")?;
        Ok(())
    }

    async fn commit_completed_turn_atomic_inner(
        &self,
        params: CompletedTurnCommit,
        fail_after: Option<InjectedCommitFailure>,
    ) -> Result<CommitOutcome> {
        let mut tx = self.begin().await?;

        Self::bootstrap_thread_row_tx(&mut tx, &params.thread_id, params.now).await?;
        let old_attempt = Self::lock_attempt_tx(&mut tx, &params.turn_attempt_id)
            .await?
            .ok_or_else(|| anyhow!("attempt not found: {}", params.turn_attempt_id))?;
        let closed_attempt = old_attempt
            .close(params.close_attempt_params, params.now)
            .context("close attempt inside postgres completed-turn transaction")?;
        Self::update_attempt_tx(&mut tx, &closed_attempt).await?;
        maybe_inject_failure(fail_after, InjectedCommitFailure::AttemptClose)?;

        let old_thread = Self::lock_thread_tx(&mut tx, &params.thread_id)
            .await?
            .context("thread missing after bootstrap")?;
        // Stale turn double-commit guard. The thread row is locked
        // FOR UPDATE; assert it is still positioned to produce
        // `expected_turn` before incrementing. A stale-lease worker that
        // lost the race to another worker fails here instead of durably
        // double-committing the turn (pure in-Rust comparison on the
        // already-locked row — no extra query).
        if old_thread.committed_turns.saturating_add(1) != params.expected_turn {
            return Err(anyhow::Error::new(StaleTurnCommit {
                expected_turn: params.expected_turn,
                committed_turns: old_thread.committed_turns,
            }));
        }

        // Owner-guarded commit, enforced after the thread lock so the
        // transaction's lock order stays thread→task, consistent with
        // `submit_root_turn` and `cancel_tree`.
        Self::enforce_commit_owner_guard_tx(&mut tx, &params.task_id, params.owner_guard.as_ref())
            .await?;
        let thread = old_thread
            .apply_committed_turn(&params.turn_usage, params.now)
            .context("advance thread aggregate inside postgres completed-turn transaction")?;
        Self::upsert_thread_tx(&mut tx, &thread).await?;
        maybe_inject_failure(fail_after, InjectedCommitFailure::ThreadAdvance)?;

        let projection_before =
            Self::lock_message_head_tx(&mut tx, &params.thread_id, params.now).await?;
        // Append the committed turn's messages AND clear any
        // in-flight draft in the same transaction. The
        // worker-level suspension paths populated the draft slot
        // at every tool-boundary suspension; once the turn fully
        // commits, the draft is subsumed by the committed history
        // and must be wiped before the next turn starts. Doing the
        // clear here (rather than as a follow-up call) closes the
        // crash window where a recovery between commit and clear
        // would surface duplicated messages through the next-turn
        // view.
        let updated_projection = projection_before
            .append_committed(params.messages.clone(), params.now)
            .context("append committed messages inside postgres completed-turn transaction")?
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
        maybe_inject_failure(fail_after, InjectedCommitFailure::MessageCommit)?;

        let checkpoint = Checkpoint::new(NewCheckpointParams {
            kind: params.checkpoint_kind,
            thread_id: params.thread_id.clone(),
            turn_number: thread.committed_turns,
            task_id: params.task_id,
            messages: updated_projection.messages,
            agent_state_snapshot: params.agent_state_snapshot,
            turn_usage: params.turn_usage,
            now: params.now,
        })
        .context("build checkpoint inside postgres completed-turn transaction")?;
        Self::insert_checkpoint_tx(&mut tx, &checkpoint).await?;
        maybe_inject_failure(fail_after, InjectedCommitFailure::CheckpointInsert)?;

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
            .context("insert lifecycle events inside postgres completed-turn transaction")?;
            Self::insert_thread_events_outbox_row_tx(
                &mut tx,
                &committed,
                params.outbox_max_attempts,
                params.now,
            )
            .await
            .context("insert advisory outbox row inside postgres completed-turn transaction")?;
            committed
        };
        maybe_inject_failure(fail_after, InjectedCommitFailure::EventCommit)?;

        // Failpoint (11 · A): simulate a crash at the true atomic-commit
        // boundary — the events and the coalesced advisory outbox row are
        // staged in `tx` but `tx.commit()` has not yet run, so the whole
        // turn (state + events) will roll back. Recovery must replay
        // idempotently. Complements the `InjectedCommitFailure` test hook
        // above (a deterministic test double); this is the runtime
        // `fail-rs` failpoint. No-op (and not compiled) without the
        // `failpoints` feature.
        agent_server::fail_point!("commit.before_event_commit");

        tx.commit()
            .await
            .context("commit postgres completed-turn transaction")?;

        Ok(CommitOutcome {
            thread,
            checkpoint,
            closed_attempt,
            committed_events,
        })
    }
}

fn validate_schema_name(schema: &str) -> Result<()> {
    ensure!(!schema.is_empty(), "postgres schema cannot be empty");
    ensure!(
        schema
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_'),
        "postgres schema must contain only ASCII letters, digits, or underscores",
    );
    ensure!(
        schema
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_'),
        "postgres schema must start with an ASCII letter or underscore",
    );
    Ok(())
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

async fn ensure_child_thread_available_for_spawn_tx(
    tx: &mut Transaction<'_, Postgres>,
    child_thread_id: &ThreadId,
) -> Result<()> {
    let child_thread = PostgresDurableStore::lock_thread_tx(tx, child_thread_id).await?;
    if child_thread.is_none() {
        return Err(anyhow!(
            "spawn rejected: child thread {child_thread_id} does not exist"
        ));
    }

    let existing_child_thread_task = sqlx::query_scalar!(
        r"
SELECT id
FROM agent_sdk_tasks
WHERE thread_id = $1
LIMIT 1
",
        thread_key(child_thread_id),
    )
    .fetch_optional(tx.as_mut())
    .await
    .with_context(|| format!("check existing tasks for child thread {child_thread_id}"))?;
    if existing_child_thread_task.is_some() {
        return Err(anyhow!(
            "spawn rejected: child thread id {child_thread_id} already has tasks"
        ));
    }

    Ok(())
}

async fn ensure_spawn_task_id_available_tx(
    tx: &mut Transaction<'_, Postgres>,
    task_id: &AgentTaskId,
    label: &str,
) -> Result<()> {
    let existing_task = PostgresDurableStore::load_task_tx(tx, task_id, false).await?;
    if existing_task.is_some() {
        return Err(anyhow!(
            "spawn rejected: {label} task id {task_id} already exists"
        ));
    }
    Ok(())
}

/// Materialize the `(invocation, child_root)` pair for every subagent
/// entry in a batch, validating cross-entry thread uniqueness and
/// per-row availability inside the caller's transaction.
///
/// Shared by [`AgentTaskStore::spawn_subagent_batch`] and
/// [`AgentTaskStore::spawn_mixed_children`] so the two paths cannot
/// drift on invocation-row construction or on the lock order (parent
/// task row first, then each child thread row).
///
/// Nothing is written here: the caller commits the rows only after the
/// parent transition succeeds, so a rejection on entry K leaves the
/// transaction free of partial fan-out.
async fn prepare_subagent_batch_rows_tx(
    tx: &mut Transaction<'_, Postgres>,
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
        ensure_child_thread_available_for_spawn_tx(tx, &child_thread_id).await?;

        let child_root = AgentTask::new_root_turn_with_optional_caller(
            child_thread_id.clone(),
            child_root_input,
            child_caller_metadata,
            now,
            AgentTask::DEFAULT_MAX_ATTEMPTS,
        );
        ensure_spawn_task_id_available_tx(tx, &child_root.id, "child root").await?;

        let invocation = AgentTask::new_subagent_invocation(
            old_parent,
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
        ensure_spawn_task_id_available_tx(tx, &invocation.id, "invocation").await?;

        prepared.push((invocation, child_root));
    }
    Ok(prepared)
}

#[async_trait]
impl AgentTaskStore for PostgresDurableStore {
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
        let _ = Self::lock_thread_tx(&mut tx, &task.thread_id).await?;

        let id_exists = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 LIMIT 1",
            task.id.as_str(),
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("check existing task {}", task.id))?;
        if id_exists.is_some() {
            return Err(anyhow!(
                "submit_root_turn rejected: task id {} already exists",
                task.id
            ));
        }

        let thread_has_blocking_root = sqlx::query_scalar!(
            r#"
SELECT EXISTS (
    SELECT 1
    FROM agent_sdk_tasks
    WHERE thread_id = $1
      AND kind = 'root_turn'
      AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
) AS "exists!"
"#,
            thread_key(&task.thread_id),
        )
        .fetch_one(&mut *tx)
        .await
        .with_context(|| format!("check active root slot for {}", task.thread_id))?;

        let thread_has_queued_roots = sqlx::query_scalar!(
            r#"
SELECT EXISTS (
    SELECT 1
    FROM agent_sdk_tasks
    WHERE thread_id = $1
      AND kind = 'root_turn'
      AND status = 'queued'
) AS "exists!"
"#,
            thread_key(&task.thread_id),
        )
        .fetch_one(&mut *tx)
        .await
        .with_context(|| format!("check queued roots for {}", task.thread_id))?;

        let admitted = if thread_has_blocking_root || thread_has_queued_roots {
            let created_at = task.created_at;
            task.admit_as_queued(created_at)
                .context("submit_root_turn rejected: cannot admit as queued")?
        } else {
            task
        };

        Self::insert_task_tx(&mut tx, &admitted).await?;

        // Phase 10 · D: when the new root is immediately runnable
        // (`Pending`, not parked behind an active/queued root), emit a
        // durable `task_wakeup` advisory row in the SAME transaction so
        // a worker in any process — including one on another host after
        // this one dies — is nudged to run `acquire_next_runnable`. A
        // queued root is parked, so its eventual promotion emits the
        // wakeup instead.
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

        let mut tx = self.begin().await.map_err(SubmitRootTurnError::Other)?;
        Self::bootstrap_thread_row_tx(&mut tx, &task.thread_id, task.created_at)
            .await
            .map_err(SubmitRootTurnError::Other)?;
        // Serialize concurrent submissions on the same thread so the
        // idempotency claim, queue-depth check, and admission insert are
        // atomic with respect to one another.
        let _ = Self::lock_thread_tx(&mut tx, &task.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        // 1. Idempotency replay / conflict, inside the same transaction.
        // Both the replay and the conflict path must release the
        // FOR UPDATE thread lock promptly — a dropped-but-uncommitted
        // tx parks the connection `idle in transaction`, holding the
        // lock and stalling other writers. So we always explicitly
        // commit (replay) or roll back (conflict / rejection) rather
        // than relying on `Drop`.
        if let Some(claim) = &idempotency {
            match self.try_replay_submit_tx(&mut tx, claim).await {
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

        let id_exists = sqlx::query_scalar!(
            "SELECT id FROM agent_sdk_tasks WHERE id = $1 LIMIT 1",
            task.id.as_str(),
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("check existing task {}", task.id))
        .map_err(SubmitRootTurnError::Other)?;
        if id_exists.is_some() {
            let _ = tx.rollback().await;
            return Err(SubmitRootTurnError::Other(anyhow!(
                "submit_root_turn rejected: task id {} already exists",
                task.id
            )));
        }

        let thread_has_blocking_root = sqlx::query_scalar!(
            r#"
SELECT EXISTS (
    SELECT 1
    FROM agent_sdk_tasks
    WHERE thread_id = $1
      AND kind = 'root_turn'
      AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
) AS "exists!"
"#,
            thread_key(&task.thread_id),
        )
        .fetch_one(&mut *tx)
        .await
        .with_context(|| format!("check active root slot for {}", task.thread_id))
        .map_err(SubmitRootTurnError::Other)?;

        let current_queued = Self::queued_root_count_tx(&mut tx, &task.thread_id)
            .await
            .map_err(SubmitRootTurnError::Other)?;

        // 2. Back-pressure: reject a submission that would be queued and
        // push the queued depth past the cap before any write.
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
        // Atomic reservation: insert a placeholder (`result_json = JSON
        // null`) if absent. `RETURNING` yields a row only when *this*
        // statement inserted, so a non-empty result means we won the
        // claim and a concurrent retry cannot also observe `Fresh`.
        let inserted = sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES ($1, $2, $3, 'null'::jsonb, $4)
               ON CONFLICT (request_id) DO NOTHING
               RETURNING request_id"#,
            request_id,
            kind.as_str(),
            fingerprint,
            OffsetDateTime::now_utc(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("idempotency claim for {request_id}"))?;

        if inserted.is_some() {
            return Ok(IdempotencyClaim::Fresh);
        }

        // Lost the race (or a prior row exists): inspect the existing row.
        let row = sqlx::query!(
            r#"SELECT kind, fingerprint, result_json FROM agent_sdk_idempotency WHERE request_id = $1"#,
            request_id,
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("idempotency lookup for {request_id}"))?
        .ok_or_else(|| anyhow!("idempotency row for {request_id} vanished after claim conflict"))?;

        if row.kind != kind.as_str() || row.fingerprint != fingerprint {
            return Ok(IdempotencyClaim::Conflict);
        }
        if row.result_json.is_null() {
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
            result_json: row.result_json,
        })))
    }

    async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
        // Fill the reservation placeholder written by `claim_idempotency`.
        // The `WHERE result_json = 'null'::jsonb` guard fills only the
        // placeholder and never overwrites a real recorded result, so a
        // racing retry cannot clobber it. With no prior claim row the
        // INSERT lands the full record (defensive path).
        sqlx::query!(
            r#"INSERT INTO agent_sdk_idempotency (request_id, kind, fingerprint, result_json, created_at)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (request_id) DO UPDATE
                   SET result_json = EXCLUDED.result_json
                   WHERE agent_sdk_idempotency.result_json = 'null'::jsonb"#,
            record.request_id,
            record.kind.as_str(),
            record.fingerprint,
            record.result_json,
            OffsetDateTime::now_utc(),
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
        let records = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE thread_id = $1
ORDER BY created_at, id
",
            thread_key(thread_id),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tasks for thread {thread_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE parent_id = $1
ORDER BY created_at, id
",
            parent_id.as_str(),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list children for {parent_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let status_wire = enum_to_wire(&status)?;
        let records = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE status = $1
ORDER BY created_at, id
",
            status_wire,
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tasks in status {status:?}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
        let record = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')
ORDER BY created_at, id
LIMIT 1
",
            thread_key(thread_id),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("load active root for {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE thread_id = $1
  AND kind = 'root_turn'
  AND status = 'queued'
ORDER BY created_at, id
",
            thread_key(thread_id),
        )
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
        let _ = Self::lock_thread_tx(&mut tx, thread_id).await?;
        let promoted = Self::promote_next_queued_root_tx(&mut tx, thread_id, now).await?;
        // Explicit terminal disposition (never leave a FOR UPDATE
        // transaction parked idle-in-transaction): commit on a real
        // promotion, roll back otherwise.
        match &promoted {
            Some(_) => tx
                .commit()
                .await
                .context("commit promote_next_queued_root")?,
            None => tx
                .rollback()
                .await
                .context("rollback promote_next_queued_root")?,
        }
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
        let Some(old) = Self::load_task_tx(&mut tx, id, true).await? else {
            tx.rollback().await.context("rollback missing acquire")?;
            return Ok(None);
        };
        if !old.status.can_be_leased() {
            tx.rollback()
                .await
                .context("rollback non-runnable acquire")?;
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
                    "try_acquire_task: recovery matrix produced Requeue for acquisition-time row {id}",
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
        loop {
            let mut tx = self.begin().await?;
            let record = sqlx::query_as!(
                TaskRecord,
                r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE status = 'pending'
ORDER BY created_at, id
LIMIT 1
FOR UPDATE SKIP LOCKED
",
            )
            .fetch_optional(&mut *tx)
            .await
            .context("load runnable head")?;
            let Some(record) = record else {
                tx.rollback()
                    .await
                    .context("rollback empty runnable scan")?;
                return Ok(None);
            };
            let old = AgentTask::try_from(record)?;
            if !old.status.can_be_leased() {
                let status = old.status;
                tx.rollback()
                    .await
                    .context("rollback corrupt runnable head")?;
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
                    tx.rollback()
                        .await
                        .context("rollback invalid acquire_next_runnable recovery")?;
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
        // (FOR UPDATE) and rewrite the whole 24-column row (re-serialising
        // submitted_input / state JSON blobs) on a per-tick heartbeat.
        // `RETURNING` hands back the refreshed row in one round trip.
        let updated = sqlx::query_as!(
            TaskRecord,
            r"
UPDATE agent_sdk_tasks
SET lease_expires_at = $4, last_heartbeat_at = $5, updated_at = $5
WHERE id = $1 AND status = 'running' AND worker_id = $2 AND lease_id = $3
RETURNING
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
",
            id.as_str(),
            worker.as_str(),
            lease.as_str(),
            expires_at,
            now,
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("heartbeat update for {id}"))?;

        if let Some(record) = updated {
            let refreshed = AgentTask::try_from(record)?;
            tx.commit().await.context("commit heartbeat_task")?;
            return Ok(refreshed);
        }

        // CAS rejection: one diagnostic SELECT to produce a precise error.
        let existing = Self::load_task_tx(&mut tx, id, false).await?;
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
        let expired = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE status = 'running'
  AND lease_expires_at <= $1
ORDER BY lease_expires_at, id
LIMIT $2
FOR UPDATE SKIP LOCKED
",
            now,
            i64::try_from(agent_server::journal::store::LEASE_RELEASE_BATCH).unwrap_or(i64::MAX),
        )
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
        boundary: Option<AgentEvent>,
        now: OffsetDateTime,
    ) -> Result<RequeueOutcome> {
        let mut tx = self.begin().await?;
        // Lock order: the global hierarchy is
        // thread → task (cancel_tree locks thread rows in phase 1
        // before re-reading tasks). When a boundary event is present
        // this method needs BOTH locks, so it must not take the task
        // lock first — read the row unlocked to learn the thread id,
        // lock the thread, then lock and revalidate the task.
        let Some(screened) = Self::load_task_tx(&mut tx, id, false).await? else {
            return Ok(RequeueOutcome::NotOwned);
        };
        if boundary.is_some() {
            Self::bootstrap_thread_row_tx(&mut tx, &screened.thread_id, now).await?;
            Self::lock_thread_tx(&mut tx, &screened.thread_id)
                .await?
                .with_context(|| {
                    format!("thread {} missing after bootstrap", screened.thread_id)
                })?;
        }
        let Some(old) = Self::load_task_tx(&mut tx, id, true).await? else {
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
        // it is durable before the row is acquirable. The sequence
        // allocation re-locks the thread row held above — no new lock
        // edge.
        if let Some(event) = boundary {
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
        let old = Self::load_task_tx(&mut tx, id, true)
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
        let old = Self::load_task_tx(&mut tx, parent_id, true)
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
        let old = Self::load_task_tx(&mut tx, parent_id, true)
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
            let mut child = Self::load_task_tx(&mut tx, child_id, true)
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
        let old = Self::load_task_tx(&mut tx, id, true)
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
        let old_parent = Self::load_task_tx(&mut tx, parent_id, true)
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
            let existing = Self::load_task_tx(&mut tx, &child.id, false).await?;
            if existing.is_some()
                || children
                    .iter()
                    .any(|existing: &AgentTask| existing.id == child.id)
            {
                return Err(anyhow!(
                    "spawn rejected: child id {} already exists",
                    child.id
                ));
            }
            children.push(child);
        }

        let child_count = u32::try_from(children.len())
            .context("spawn rejected: child count exceeds u32::MAX")?;
        let child_ids = children.iter().map(|child| child.id.clone()).collect();
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

        let old_parent = Self::load_task_tx(&mut tx, parent_id, true)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;
        ensure_child_thread_available_for_spawn_tx(&mut tx, &child_thread_id).await?;

        let child_root = AgentTask::new_root_turn_with_optional_caller(
            child_thread_id.clone(),
            child_root_input,
            child_caller_metadata,
            now,
            AgentTask::DEFAULT_MAX_ATTEMPTS,
        );
        ensure_spawn_task_id_available_tx(&mut tx, &child_root.id, "child root").await?;

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
        ensure_spawn_task_id_available_tx(&mut tx, &invocation.id, "invocation").await?;

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

        let old_parent = Self::load_task_tx(&mut tx, parent_id, true)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;

        let prepared = prepare_subagent_batch_rows_tx(&mut tx, &old_parent, spawns, now).await?;
        let child_ids: Vec<AgentTaskId> = prepared
            .iter()
            .map(|(invocation, _)| invocation.id.clone())
            .collect();

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

    async fn spawn_mixed_children(
        &self,
        parent_id: &AgentTaskId,
        worker: &WorkerId,
        lease: &LeaseId,
        spawn: MixedChildrenSpawn,
        now: OffsetDateTime,
    ) -> Result<SpawnedMixedChildren> {
        validate_mixed_children_spawn(&spawn)?;
        let MixedChildrenSpawn {
            subagents,
            tool_children,
            payload,
            child_otel_traceparent,
        } = spawn;

        let mut tx = self.begin().await?;

        // Lock order matches `spawn_subagent_batch`: the parent task row
        // first, then each child thread row. Diverging here would let a
        // mixed spawn and a pure fan-out deadlock against each other.
        let old_parent = Self::load_task_tx(&mut tx, parent_id, true)
            .await?
            .ok_or_else(|| anyhow!("spawn rejected: task {parent_id} does not exist"))?;
        validate_subagent_spawn_parent(&old_parent, parent_id, worker, lease)?;

        let prepared = prepare_subagent_batch_rows_tx(&mut tx, &old_parent, subagents, now).await?;

        let mut tool_rows: Vec<AgentTask> = Vec::with_capacity(tool_children.len());
        for tool in &tool_children {
            let child =
                new_mixed_tool_child(&old_parent, tool, child_otel_traceparent.as_deref(), now)?;
            ensure_spawn_task_id_available_tx(&mut tx, &child.id, "child").await?;
            tool_rows.push(child);
        }

        let child_ids = mixed_child_ids_in_slot_order(&prepared, &tool_rows)?;

        let child_count =
            u32::try_from(child_ids.len()).context("spawn rejected: child count exceeds u32")?;
        let new_parent = old_parent
            .clone()
            .wait_on_children(child_count, payload, child_ids, now)
            .context("spawn rejected: wait_on_children transition failed")?;
        Self::update_task_tx(&mut tx, &new_parent).await?;
        for (invocation, child_root) in &prepared {
            Self::insert_task_tx(&mut tx, invocation).await?;
            Self::insert_task_tx(&mut tx, child_root).await?;
        }
        for child in &tool_rows {
            Self::insert_task_tx(&mut tx, child).await?;
        }
        tx.commit().await.context("commit spawn_mixed_children")?;
        Ok(SpawnedMixedChildren {
            parent: new_parent,
            subagents: prepared,
            tool_children: tool_rows,
        })
    }

    async fn find_subagent_invocation_for_child_root(
        &self,
        child_root_id: &AgentTaskId,
    ) -> Result<Option<AgentTask>> {
        // Read-only counterpart of `resume_linked_subagent_invocation_tx`:
        // same linkage predicate, no transition, no row lock.
        let record = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE kind = 'subagent'
  AND status = 'waiting_on_children'
  AND state_json -> 'invocation' ->> 'child_root_task_id' = $1
",
            child_root_id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| {
            format!("find_subagent_invocation_for_child_root: lookup for {child_root_id}")
        })?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>> {
        let records = sqlx::query_as!(
            TaskRecord,
            r"
SELECT
    id,
    kind,
    status,
    parent_id,
    root_id,
    depth,
    thread_id,
    submitted_input_json,
    caller_metadata_json,
    worker_id,
    lease_id,
    lease_expires_at,
    last_heartbeat_at,
    state_json,
    attempt,
    max_attempts,
    last_error,
    pending_child_count,
    spawn_index,
    result_payload,
    otel_traceparent,
    created_at,
    updated_at,
    completed_at
FROM agent_sdk_tasks
WHERE kind = 'subagent'
  AND status = 'waiting_on_children'
ORDER BY created_at, id
",
        )
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
        let Some(_) = Self::load_task_tx(&mut tx, root_id, false).await? else {
            return Err(anyhow!(
                "cancel_tree rejected: task {root_id} does not exist"
            ));
        };

        // Collect the given task + descendants (mirror in-memory
        // `collect_subtree`: children-BFS, root-first, following
        // SubagentInvocation linkage across threads). A mid-tree id
        // cancels only its own subtree instead of no-opping on a
        // `root_id = $1` scan.
        //
        // The snapshot only shapes the WALK (ids, depths, kinds,
        // thread ids — all row-invariant). Statuses in it are advisory:
        // under READ COMMITTED a concurrent cancel_tree / terminal CAS
        // can settle any of these rows between this read and our
        // UPDATEs, so every transition below is gated on a locked
        // re-read — otherwise two racing cancels of the same root
        // could both emit a marker and both report the transition, and
        // a stale snapshot could clobber a concurrently-Completed root
        // with Cancelled.
        let all_tasks = Self::collect_subtree_tx(&mut tx, root_id).await?;

        // Phase 1 — thread-row locks, FIRST. Every thread that MIGHT
        // need a terminal marker is bootstrapped and locked before any
        // task-row lock is taken, keeping this transaction's lock
        // order (thread rows → task rows) consistent with
        // `submit_root_turn` so a concurrent submit cannot deadlock
        // the cancel. The candidate set is every root-turn root in the
        // snapshot regardless of its snapshot status — status can
        // change before the locked re-read below (e.g. a queued root
        // promoted to Pending by a concurrent terminal transition),
        // but kind / is_root / thread_id cannot. BFS order keeps the
        // thread-lock order deterministic (parent thread before its
        // descendants' threads) for concurrent overlapping cancels.
        let mut marker_threads_locked: std::collections::HashSet<ThreadId> =
            std::collections::HashSet::new();
        for row in &all_tasks {
            if row.kind == TaskKind::RootTurn
                && row.is_root()
                && marker_threads_locked.insert(row.thread_id.clone())
            {
                Self::bootstrap_thread_row_tx(&mut tx, &row.thread_id, now).await?;
                let _ = Self::lock_thread_tx(&mut tx, &row.thread_id).await?;
            }
        }

        // Phase 2 — CAS cancel transitions. Apply the cancel UPDATEs
        // deepest-first so the per-row locks are taken in the same
        // child-then-parent order as the terminal transition paths
        // (`complete_task` / `fail_task`), removing the lock-order
        // inversion that could deadlock (40P01) a concurrent
        // `cancel_tree(root)` + `complete_task(child-of-root)`. Each
        // row is re-read FOR UPDATE and skipped if it settled
        // concurrently — the transition, the returned `transitioned`
        // slice, and the marker gating below all derive from the
        // locked re-read, never from the snapshot.
        let mut cancel_order: Vec<&AgentTask> = all_tasks.iter().collect();
        cancel_order.sort_by_key(|t| std::cmp::Reverse(t.depth));

        let mut cancelled_ids: std::collections::BTreeSet<AgentTaskId> =
            std::collections::BTreeSet::new();
        // Track cancelled root-turn roots (and their threads) so we can
        // wake their linked invocations and promote queued successors.
        let mut cancelled_root_ids: Vec<AgentTaskId> = Vec::new();
        let mut cancelled_root_threads: Vec<ThreadId> = Vec::new();
        // Blocking roots actually transitioned by THIS call, keyed by
        // id so phase 3 can emit their markers in BFS order. The
        // continuation usage is captured from the locked re-read
        // before `cancel` clears the typed state.
        let mut marker_candidates: std::collections::BTreeMap<
            AgentTaskId,
            (ThreadId, Option<TokenUsage>),
        > = std::collections::BTreeMap::new();
        for row in cancel_order {
            let Some(current) = Self::load_task_tx(&mut tx, &row.id, true).await? else {
                continue;
            };
            if current.status.is_terminal() {
                continue;
            }
            let is_root_turn_root = current.kind == TaskKind::RootTurn && current.is_root();
            let thread_id = current.thread_id.clone();
            if is_root_turn_root && current.status.blocks_root_admission() {
                marker_candidates.insert(
                    current.id.clone(),
                    (
                        thread_id.clone(),
                        current
                            .state
                            .continuation()
                            .map(|continuation| continuation.payload.total_usage.clone()),
                    ),
                );
            }
            let cancelled = current
                .cancel(now)
                .context("cancel_tree: cancel transition failed")?;
            Self::update_task_tx(&mut tx, &cancelled).await?;
            cancelled_ids.insert(cancelled.id.clone());
            if is_root_turn_root {
                cancelled_root_ids.push(cancelled.id.clone());
                cancelled_root_threads.push(thread_id);
            }
        }
        // Build the returned slice in BFS (root-first) order for parity.
        let transitioned: Vec<AgentTaskId> = all_tasks
            .iter()
            .filter(|task| cancelled_ids.contains(&task.id))
            .map(|task| task.id.clone())
            .collect();

        // Phase 3 — terminal `Cancelled` markers: exactly
        // one per blocking root this call ACTUALLY transitioned (the
        // pre-cancel occupant of its thread's active-root slot), on
        // that root's OWN thread — cascade-cancelled child-thread
        // roots included, queued roots never. Atomic with the cancel
        // transitions above, so a crash can neither lose a marker nor
        // emit one without its cancellation, and a racing duplicate
        // cancel (whose re-read saw the row already terminal) emits
        // nothing. The thread rows were locked in phase 1, so the
        // sequence allocation here re-locks held rows (no new lock
        // edges). Emitted in BFS order.
        let mut markers: Vec<CommittedEvent> = Vec::new();
        for row in &all_tasks {
            let Some((thread_id, continuation_usage)) = marker_candidates.remove(&row.id) else {
                continue;
            };
            let committed =
                Self::insert_cancelled_marker_tx(&mut tx, &thread_id, continuation_usage, now)
                    .await?;
            markers.push(committed);
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
        // itself cancelled), so one propagation suffices. Taking the
        // parent lock after the child updates preserves the
        // child-then-parent lock order the deepest-first cancel loop
        // establishes.
        if transitioned.first().is_some_and(|id| id == root_id)
            && let Some(cancelled_top) = Self::load_task_tx(&mut tx, root_id, false).await?
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
        let old = Self::load_task_tx(&mut tx, id, true)
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
        let old = Self::load_task_tx(&mut tx, id, true)
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
        let old = Self::load_task_tx(&mut tx, id, true)
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
            let old_parent = Self::load_task_tx(&mut tx, parent_id, true)
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
        sqlx::query!(
            r"
TRUNCATE TABLE
    agent_sdk_turn_attempts,
    agent_sdk_turn_checkpoints,
    agent_sdk_message_commits,
    agent_sdk_message_heads,
    agent_sdk_execution_intents,
    agent_sdk_tool_audit_events,
    agent_sdk_tasks,
    agent_sdk_threads
CASCADE
",
        )
        .execute(&self.pool)
        .await
        .context("clear postgres durable-core tables")?;
        Ok(())
    }
}

#[async_trait]
impl ThreadStore for PostgresDurableStore {
    fn atomic_completed_turn_committer(&self) -> Option<&dyn AtomicCompletedTurnCommitter> {
        Some(self)
    }

    fn atomic_fork_committer(&self) -> Option<&dyn AtomicForkCommitter> {
        Some(self)
    }

    async fn get_or_create(&self, thread_id: &ThreadId, now: OffsetDateTime) -> Result<Thread> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let thread = Self::lock_thread_tx(&mut tx, thread_id)
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
        let old = Self::lock_thread_tx(&mut tx, thread_id)
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
        let old = Self::lock_thread_tx(&mut tx, thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread {thread_id} does not exist"))?;
        let completed = old.mark_completed(now)?;
        Self::upsert_thread_tx(&mut tx, &completed).await?;
        tx.commit().await.context("commit mark_completed thread")?;
        Ok(completed)
    }

    async fn list(&self) -> Result<Vec<Thread>> {
        let records = sqlx::query_as!(
            ThreadRecord,
            r"
SELECT
    thread_id,
    status,
    committed_turns,
    total_input_tokens,
    total_output_tokens,
    created_at,
    updated_at
FROM agent_sdk_threads
ORDER BY created_at, thread_id
",
        )
        .fetch_all(&self.pool)
        .await
        .context("list threads")?;
        records.into_iter().map(TryInto::try_into).collect()
    }
}

#[async_trait]
impl MessageProjectionStore for PostgresDurableStore {
    async fn get_or_create(
        &self,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<MessageProjection> {
        let mut tx = self.begin().await?;
        Self::bootstrap_thread_row_tx(&mut tx, thread_id, now).await?;
        let projection = Self::lock_message_head_tx(&mut tx, thread_id, now).await?;
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
            .map(|projection| projection.messages)
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
        let projection = Self::lock_message_head_tx(&mut tx, thread_id, now).await?;
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
        let projection = Self::lock_message_head_tx(&mut tx, thread_id, now).await?;
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
        let projection = Self::lock_message_head_tx(&mut tx, thread_id, now).await?;
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
            // No projection row exists yet — nothing to clear.
            // Mirror the SQLite path: the commit helper short-
            // circuits on first-turn happy paths.
            return Ok(None);
        };
        let updated = projection.clear_draft(now);
        let mut tx = self.begin().await?;
        Self::upsert_message_head_tx(&mut tx, &updated).await?;
        tx.commit().await.context("commit clear_draft")?;
        Ok(Some(updated))
    }
}

#[async_trait]
impl TurnAttemptStore for PostgresDurableStore {
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
        let old = Self::lock_attempt_tx(&mut tx, id)
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
        let records = sqlx::query_as!(
            TurnAttemptRecord,
            r"
SELECT
    id,
    task_id,
    attempt_number,
    provider,
    requested_model,
    request_blob,
    response_blob,
    response_id,
    response_model,
    stop_reason,
    outcome,
    input_tokens,
    output_tokens,
    cached_input_tokens,
    opened_at,
    closed_at,
    duration_ms,
    otel_trace_id,
    otel_span_id
FROM agent_sdk_turn_attempts
WHERE task_id = $1
ORDER BY attempt_number
",
            task_id.as_str(),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list attempts for task {task_id}"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }
}

#[async_trait]
impl CheckpointStore for PostgresDurableStore {
    async fn commit_checkpoint(&self, params: NewCheckpointParams) -> Result<Checkpoint> {
        let checkpoint = Checkpoint::new(params)?;
        let mut tx = self.begin().await?;
        let duplicate = Self::lock_checkpoint_by_turn_tx(
            &mut tx,
            &checkpoint.thread_id,
            checkpoint.turn_number,
        )
        .await?;
        if duplicate.is_some() {
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
        let record = sqlx::query_as!(
            CheckpointRecord,
            r"
SELECT
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
FROM agent_sdk_turn_checkpoints
WHERE thread_id = $1
  AND turn_number = $2
",
            thread_key(thread_id),
            i64::from(turn_number),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get checkpoint {thread_id} turn {turn_number}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_latest_by_thread(&self, thread_id: &ThreadId) -> Result<Option<Checkpoint>> {
        let record = sqlx::query_as!(
            CheckpointRecord,
            r"
SELECT
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
FROM agent_sdk_turn_checkpoints
WHERE thread_id = $1
ORDER BY turn_number DESC
LIMIT 1
",
            thread_key(thread_id),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get latest checkpoint for {thread_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<Checkpoint>> {
        let records = sqlx::query_as!(
            CheckpointRecord,
            r"
SELECT
    id,
    thread_id,
    turn_number,
    task_id,
    messages_json,
    agent_state_snapshot,
    turn_input_tokens,
    turn_output_tokens,
    kind,
    created_at
FROM agent_sdk_turn_checkpoints
WHERE thread_id = $1
ORDER BY turn_number
",
            thread_key(thread_id),
        )
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
        let rows = sqlx::query!(
            r"SELECT thread_id FROM agent_sdk_turn_checkpoints GROUP BY thread_id HAVING COUNT(*) > $1 ORDER BY thread_id LIMIT $2",
            threshold_i64,
            limit_i64,
        )
        .fetch_all(&self.pool)
        .await
        .context("threads_exceeding_checkpoint_count")?;
        Ok(rows
            .into_iter()
            .map(|r| ThreadId::from_string(r.thread_id))
            .collect())
    }

    async fn delete_checkpoints_beyond_limit(
        &self,
        thread_id: &ThreadId,
        keep_latest_n: u32,
    ) -> Result<u64> {
        ensure!(keep_latest_n >= 1, "keep_latest_n must be at least 1");

        let result = sqlx::query!(
            r"
DELETE FROM agent_sdk_turn_checkpoints
WHERE thread_id = $1
  AND id NOT IN (
      SELECT id FROM agent_sdk_turn_checkpoints
      WHERE thread_id = $1
      ORDER BY turn_number DESC
      LIMIT $2
  )
",
            thread_key(thread_id),
            i64::from(keep_latest_n),
        )
        .execute(&self.pool)
        .await
        .with_context(|| format!("delete checkpoints beyond limit for {thread_id}"))?;

        Ok(result.rows_affected())
    }
}

#[async_trait]
impl AtomicCompletedTurnCommitter for PostgresDurableStore {
    async fn commit_completed_turn_atomic(
        &self,
        params: CompletedTurnCommit,
    ) -> Result<CommitOutcome> {
        self.commit_completed_turn_atomic_inner(params, None).await
    }
}

#[async_trait]
impl AtomicForkCommitter for PostgresDurableStore {
    async fn commit_fork_atomic(&self, params: ForkCommitParams) -> Result<()> {
        self.commit_fork_atomic_inner(params).await
    }
}

// ─────────────────────────────────────────────────────────────────────
// EventRepository
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl EventRepository for PostgresDurableStore {
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
        let record = sqlx::query!(
            r"SELECT COALESCE(MAX(sequence) + 1, 0) AS next_seq FROM agent_sdk_committed_events WHERE thread_id = $1",
            thread_key(thread_id),
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("next event sequence for {thread_id}"))?;

        let from_events = u64::try_from(record.next_seq.unwrap_or(0))
            .context("event next_sequence out of range")?;

        // Clamp to the retention floor: when the janitor has purged
        // all events, MAX() is NULL and `from_events` would fall back
        // to 0, letting new events reuse sequences already "claimed"
        // by purged history and silently missing every subscriber.
        let floor_record = sqlx::query!(
            r"SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = $1",
            thread_key(thread_id),
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
        let records = sqlx::query_as!(
            CommittedEventRecord,
            r"
SELECT event_id, thread_id, sequence, event_json, committed_at
FROM agent_sdk_committed_events
WHERE thread_id = $1
ORDER BY sequence
",
            thread_key(thread_id),
        )
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
        let records = sqlx::query_as!(
            CommittedEventRecord,
            r"
SELECT event_id, thread_id, sequence, event_json, committed_at
FROM agent_sdk_committed_events
WHERE thread_id = $1
  AND sequence > $2
  AND sequence <= $3
ORDER BY sequence
",
            thread_key(thread_id),
            i64_from_u64(after_sequence, "after_sequence")?,
            i64_from_u64(up_to_sequence, "up_to_sequence")?,
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| {
            format!("get events in range ({after_sequence}, {up_to_sequence}] for {thread_id}")
        })?;

        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn threads_with_events_before(
        &self,
        cutoff: OffsetDateTime,
        limit: u32,
    ) -> Result<Vec<ThreadId>> {
        let records = sqlx::query!(
            r"
SELECT DISTINCT thread_id
FROM agent_sdk_committed_events
WHERE committed_at < $1
ORDER BY thread_id
LIMIT $2
",
            cutoff,
            i64::from(limit),
        )
        .fetch_all(&self.pool)
        .await
        .context("threads with events before cutoff")?;

        Ok(records.into_iter().map(|r| ThreadId(r.thread_id)).collect())
    }

    async fn max_sequence_before(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        let record = sqlx::query!(
            r"
SELECT MAX(sequence) AS max_seq
FROM agent_sdk_committed_events
WHERE thread_id = $1
  AND committed_at < $2
",
            thread_key(thread_id),
            cutoff,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("max sequence before cutoff for {thread_id}"))?;

        record
            .max_seq
            .map(|v| u64_from_i64(v, "max_sequence_before"))
            .transpose()
    }

    async fn min_sequence_at_or_after(
        &self,
        thread_id: &ThreadId,
        cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        let record = sqlx::query!(
            r"
SELECT MIN(sequence) AS min_seq
FROM agent_sdk_committed_events
WHERE thread_id = $1
  AND committed_at >= $2
",
            thread_key(thread_id),
            cutoff,
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("min sequence at or after cutoff for {thread_id}"))?;

        record
            .min_seq
            .map(|v| u64_from_i64(v, "min_sequence_at_or_after"))
            .transpose()
    }
}

impl PostgresDurableStore {
    /// Allocate the next event sequence for a thread inside an existing
    /// transaction.
    ///
    /// Postgres forbids `FOR UPDATE` on aggregate queries, so we lock
    /// the owning `agent_sdk_threads` row instead to serialise concurrent
    /// writers on the same thread, then read the max sequence without a
    /// locking clause.
    async fn next_event_sequence_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
    ) -> Result<u64> {
        // Lock the thread row to serialise concurrent event writers.
        sqlx::query!(
            r"SELECT thread_id FROM agent_sdk_threads WHERE thread_id = $1 FOR UPDATE",
            thread_key(thread_id),
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("lock thread row for event sequence allocation on {thread_id}"))?
        .ok_or_else(|| anyhow::anyhow!("thread not found: {thread_id}"))?;

        let record = sqlx::query!(
            r"SELECT COALESCE(MAX(sequence) + 1, 0) AS next_seq FROM agent_sdk_committed_events WHERE thread_id = $1",
            thread_key(thread_id),
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("next event sequence (tx) for {thread_id}"))?;

        let from_events = u64::try_from(record.next_seq.unwrap_or(0))
            .context("event next_sequence (tx) out of range")?;

        // Derived from the retention floor so sequences never regress
        // after the janitor purges events. See `next_sequence` for
        // the full rationale.
        let floor_record = sqlx::query!(
            r"SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = $1",
            thread_key(thread_id),
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

    /// Insert committed events into an existing transaction.
    async fn insert_events_tx(
        tx: &mut Transaction<'_, Postgres>,
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

            sqlx::query!(
                r"
INSERT INTO agent_sdk_committed_events (event_id, thread_id, sequence, event_json, committed_at)
VALUES ($1, $2, $3, $4, $5)
",
                event_id.to_string(),
                thread_key(thread_id),
                i64_from_u64(seq, "event batch sequence")?,
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
    /// committed event batch.
    ///
    /// Phase 8.1 contract: exactly one
    /// `OutboxMessageKind::ThreadEventsAvailable` row per batch.
    ///
    /// The row's `sequence` / `event_id` columns reference the FIRST
    /// event of the batch so `min_unpublished_sequence` acts as a
    /// retention-floor safety bound over the entire batch range
    /// (otherwise the janitor could delete earlier events of a
    /// multi-event batch).  The FK to `agent_sdk_committed_events`
    /// stays sound either way.  The advisory payload still carries
    /// `last_sequence` so subscribers know how far to replay.
    async fn insert_thread_events_outbox_row_tx(
        tx: &mut Transaction<'_, Postgres>,
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
        let first_sequence_i64 = i64_from_u64(first.sequence, "outbox sequence")?;

        sqlx::query!(
            r"
INSERT INTO agent_sdk_outbox
    (id, kind, thread_id, event_id, sequence, status, payload_json,
     created_at, next_attempt_at, attempt_count, max_attempts)
VALUES ($1, 'thread_events_available', $2, $3, $4, 'pending', $5, $6, $6, 0, $7)
",
            id.as_str(),
            thread_key(&last.thread_id),
            first.event_id.to_string(),
            first_sequence_i64,
            payload_json,
            now,
            i64::from(max_attempts),
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
    /// existing transaction.
    ///
    /// Phase 10 · D: the row carries no `event_id`/`sequence` (those are
    /// reserved for `thread_events_available`); its advisory payload is
    /// `{task_id, thread_id}`.  Inserting it inside the admission
    /// transaction guarantees the wakeup becomes visible iff the
    /// task-journal mutation that made the task runnable also commits —
    /// the durable analogue of the in-process `WakeupSignal` nudge so
    /// another process's worker pool can be woken even after this host
    /// dies.
    async fn insert_task_wakeup_outbox_row_tx(
        tx: &mut Transaction<'_, Postgres>,
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

        sqlx::query!(
            r"
INSERT INTO agent_sdk_outbox
    (id, kind, thread_id, event_id, sequence, status, payload_json,
     created_at, next_attempt_at, attempt_count, max_attempts)
VALUES ($1, 'task_wakeup', $2, NULL, NULL, 'pending', $3, $4, $4, 0, $5)
",
            id.as_str(),
            thread_key(thread_id),
            payload_json,
            now,
            i64::from(max_attempts),
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("insert task_wakeup outbox row for task {task_id}"))?;

        Ok(id)
    }

    /// Commit the terminal `Cancelled` marker for one cancelled
    /// blocking root inside `cancel_tree`'s transaction: one committed
    /// event on the root's own thread plus its coalesced
    /// `thread_events_available` outbox advisory.
    ///
    /// Locks the thread row first (bootstrap + `FOR UPDATE` via the
    /// sequence allocator), which keeps the transaction's lock order —
    /// thread rows before task rows — consistent with
    /// `submit_root_turn`, so a concurrent submit on the same thread
    /// cannot deadlock against the cancel.
    ///
    /// `continuation_usage` is the cumulative usage captured from the
    /// root's parked continuation before the cancel clears its typed
    /// state; when absent the thread aggregate is reported instead.
    async fn insert_cancelled_marker_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &ThreadId,
        continuation_usage: Option<TokenUsage>,
        now: OffsetDateTime,
    ) -> Result<CommittedEvent> {
        // A blocking root's thread row always exists (submit
        // bootstraps it), but bootstrap defensively so a marker on an
        // exotic fixture cannot abort the whole cancellation.
        Self::bootstrap_thread_row_tx(tx, thread_id, now).await?;
        let start_seq = Self::next_event_sequence_tx(tx, thread_id).await?;
        let thread = Self::lock_thread_tx(tx, thread_id)
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
}

// ─────────────────────────────────────────────────────────────────────
// AtomicEventOutboxCommitter
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl AtomicEventOutboxCommitter for PostgresDurableStore {
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
impl TaskWakeupEmitter for PostgresDurableStore {
    /// Insert a durable `task_wakeup` advisory row in its own
    /// transaction.
    ///
    /// The admission paths (`submit_root_turn`, queue promotion) call
    /// `Self::insert_task_wakeup_outbox_row_tx` inside their own
    /// transaction so the wakeup is atomic with the journal mutation
    /// that made the task runnable.  This standalone hook serves
    /// callers that need to emit a wakeup independently (e.g. a
    /// re-publish path) where there is no surrounding journal write.
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
impl OutboxStore for PostgresDurableStore {
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
            let event_id_str = params.event_id.map(|uuid| uuid.to_string());
            let sequence_i64 = params
                .sequence
                .map(|seq| i64_from_u64(seq, "outbox sequence"))
                .transpose()?;

            sqlx::query!(
                r"
INSERT INTO agent_sdk_outbox
    (id, kind, thread_id, event_id, sequence, status, payload_json,
     created_at, next_attempt_at, attempt_count, max_attempts)
VALUES ($1, $2, $3, $4, $5, 'pending', $6, $7, $7, 0, $8)
",
                id.as_str(),
                params.kind.as_str(),
                thread_key(&params.thread_id),
                event_id_str.as_deref(),
                sequence_i64,
                params.payload_json,
                params.now,
                i64::from(params.max_attempts),
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
        let records = sqlx::query_as!(
            OutboxRecord,
            r"
UPDATE agent_sdk_outbox
SET status = 'claimed', claimed_by = $1, claimed_at = $2
WHERE id IN (
    SELECT id FROM agent_sdk_outbox
    WHERE status = 'pending' AND next_attempt_at <= $2
    ORDER BY next_attempt_at, id
    LIMIT $3
    FOR UPDATE SKIP LOCKED
)
RETURNING id, kind, thread_id, event_id, sequence, status, payload_json,
          created_at, next_attempt_at, attempt_count, max_attempts,
          last_error, claimed_by, claimed_at, delivered_at
",
            worker_id,
            now,
            i64::from(limit),
        )
        .fetch_all(&self.pool)
        .await
        .context("claim pending outbox rows")?;

        let mut rows: Vec<OutboxRow> = records
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_>>()?;
        rows.sort_by_key(|r| (r.next_attempt_at, r.id.clone()));
        Ok(rows)
    }

    async fn mark_delivered(
        &self,
        id: &OutboxRowId,
        worker_id: &str,
        now: OffsetDateTime,
    ) -> Result<()> {
        // Guard on `claimed_by`: only the worker that still holds the claim
        // may ack the row. A claim reclaimed by another worker (or a
        // cascade-deleted row) matches zero rows here — a safe no-op, because
        // the current owner will ack it and acking here would double-process.
        // The guard is what prevents a stale worker from double-acking.
        sqlx::query!(
            r"
UPDATE agent_sdk_outbox
SET status = 'delivered', delivered_at = $2
WHERE id = $1 AND claimed_by = $3
",
            id.as_str(),
            now,
            worker_id,
        )
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
        // last_error must be NULL for pending rows per the
        // agent_sdk_outbox_error_check constraint.
        // `claimed_by = $4` guard: a reclaimed or cascade-deleted row matches
        // zero rows — a safe no-op (the current owner handles it).
        sqlx::query!(
            r"
UPDATE agent_sdk_outbox
SET
    attempt_count = attempt_count + 1,
    status = CASE
        WHEN attempt_count + 1 >= max_attempts THEN 'expired'
        ELSE 'pending'
    END,
    last_error = CASE
        WHEN attempt_count + 1 >= max_attempts THEN $2
        ELSE NULL
    END,
    next_attempt_at = CASE
        WHEN attempt_count + 1 >= max_attempts THEN next_attempt_at
        ELSE $3
    END,
    claimed_by = CASE
        WHEN attempt_count + 1 >= max_attempts THEN claimed_by
        ELSE NULL
    END,
    claimed_at = CASE
        WHEN attempt_count + 1 >= max_attempts THEN claimed_at
        ELSE NULL
    END
WHERE id = $1 AND claimed_by = $4
",
            id.as_str(),
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
        // Compute the threshold outside SQL so the query stays portable
        // and we can pass a single timestamp.  Rows whose claim started
        // at or before `threshold` are considered stale.
        let threshold = now - claim_lease;
        let result = sqlx::query!(
            r"
UPDATE agent_sdk_outbox
SET
    status = 'pending',
    claimed_by = NULL,
    claimed_at = NULL,
    next_attempt_at = $1
WHERE status = 'claimed'
  AND claimed_at IS NOT NULL
  AND claimed_at <= $2
",
            now,
            threshold,
        )
        .execute(&self.pool)
        .await
        .context("reclaim expired outbox claims")?;

        Ok(result.rows_affected())
    }

    async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>> {
        let record = sqlx::query_as!(
            OutboxRecord,
            r"
SELECT id, kind, thread_id, event_id, sequence, status, payload_json,
       created_at, next_attempt_at, attempt_count, max_attempts,
       last_error, claimed_by, claimed_at, delivered_at
FROM agent_sdk_outbox
WHERE id = $1
",
            id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get outbox row {id}"))?;

        record.map(TryInto::try_into).transpose()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>> {
        let records = sqlx::query_as!(
            OutboxRecord,
            r"
SELECT id, kind, thread_id, event_id, sequence, status, payload_json,
       created_at, next_attempt_at, attempt_count, max_attempts,
       last_error, claimed_by, claimed_at, delivered_at
FROM agent_sdk_outbox
WHERE thread_id = $1
ORDER BY sequence NULLS LAST, id
",
            thread_key(thread_id),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list outbox rows for {thread_id}"))?;

        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
        let record = sqlx::query!(
            r"SELECT COUNT(*) AS cnt FROM agent_sdk_outbox WHERE thread_id = $1 AND status IN ('pending', 'claimed')",
            thread_key(thread_id),
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("count pending outbox rows for {thread_id}"))?;

        let count = record.cnt.unwrap_or(0);
        u64::try_from(count).context("outbox pending count out of range")
    }

    async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>> {
        let record = sqlx::query!(
            r"
SELECT MIN(sequence) AS min_seq
FROM agent_sdk_outbox
WHERE thread_id = $1
  AND status IN ('pending', 'claimed')
  AND sequence IS NOT NULL
",
            thread_key(thread_id),
        )
        .fetch_one(&self.pool)
        .await
        .with_context(|| format!("min unpublished sequence for {thread_id}"))?;

        record
            .min_seq
            .map(|v| u64_from_i64(v, "min_unpublished_sequence"))
            .transpose()
    }
}

// ─────────────────────────────────────────────────────────────────────
// RetentionStore
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl RetentionStore for PostgresDurableStore {
    async fn get_cursor(&self, thread_id: &ThreadId) -> Result<Option<RetentionCursor>> {
        let record = sqlx::query_as!(
            RetentionCursorRecord,
            r"
SELECT thread_id, retention_floor, updated_at
FROM agent_sdk_retention_cursors
WHERE thread_id = $1
",
            thread_key(thread_id),
        )
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

        // Read the current floor under a row lock so we can reject
        // backward moves with a clear error, matching the in-memory
        // implementation's contract.
        let current = sqlx::query!(
            r"SELECT retention_floor FROM agent_sdk_retention_cursors WHERE thread_id = $1 FOR UPDATE",
            thread_key(thread_id),
        )
        .fetch_optional(&mut *tx)
        .await
        .with_context(|| format!("read retention floor for {thread_id}"))?;

        if let Some(row) = current {
            ensure!(
                new_floor_i64 >= row.retention_floor,
                "retention floor can only advance: current {}, requested {new_floor}",
                row.retention_floor,
            );
        }

        sqlx::query!(
            r"
INSERT INTO agent_sdk_retention_cursors (thread_id, retention_floor, updated_at)
VALUES ($1, $2, $3)
ON CONFLICT (thread_id) DO UPDATE
SET retention_floor = EXCLUDED.retention_floor,
    updated_at = EXCLUDED.updated_at
",
            thread_key(thread_id),
            new_floor_i64,
            now,
        )
        .execute(&mut *tx)
        .await
        .with_context(|| format!("advance retention floor for {thread_id}"))?;

        sqlx::query!(
            r"DELETE FROM agent_sdk_committed_events WHERE thread_id = $1 AND sequence < $2",
            thread_key(thread_id),
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
impl ExecutionIntentStore for PostgresDurableStore {
    async fn persist_intent(&self, intent: &ExecutionIntent) -> Result<()> {
        let effect_class_wire = enum_to_wire(&intent.effect_class)?;
        let status_wire = enum_to_wire(&intent.status)?;

        // Atomic claim: insert-if-absent. `ON CONFLICT DO NOTHING` keeps
        // an existing in-flight record intact, and the affected-row count
        // distinguishes the winning claim (1 row) from a concurrent loser
        // (0 rows). A blind upsert here would let a stale worker downgrade
        // a `Started` record back to `Pending` and double-execute the
        // side-effecting tool — the exact harm this guard prevents. A
        // conflict (0 rows) returns an error so `guarded_tool_execution`
        // fails the child closed.
        let result = sqlx::query!(
            r"
INSERT INTO agent_sdk_execution_intents (
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (operation_id) DO NOTHING
",
            intent.operation_id.as_str(),
            effect_class_wire,
            intent.tool_call_id,
            intent.child_task_id.as_str(),
            intent.tool_name,
            intent.input,
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

        let result = sqlx::query!(
            r"
UPDATE agent_sdk_execution_intents
SET status     = $2,
    error      = $3,
    updated_at = $4
WHERE operation_id = $1
",
            intent.operation_id.as_str(),
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
        let record = sqlx::query_as!(
            ExecutionIntentRecord,
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
FROM agent_sdk_execution_intents
WHERE operation_id = $1
",
            operation_id.as_str(),
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get execution intent {operation_id}"))?;
        record.map(TryInto::try_into).transpose()
    }

    async fn get_intent_by_task(
        &self,
        child_task_id: &AgentTaskId,
    ) -> Result<Option<ExecutionIntent>> {
        let record = sqlx::query_as!(
            ExecutionIntentRecord,
            r"
SELECT
    operation_id, effect_class, tool_call_id, child_task_id,
    tool_name, input, status, error, created_at, updated_at
FROM agent_sdk_execution_intents
WHERE child_task_id = $1
",
            child_task_id.as_str(),
        )
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
impl ToolAuditEventStore for PostgresDurableStore {
    async fn record_event(&self, event: &ToolAuditEvent) -> Result<()> {
        let kind_payload = json_to_value(&event.kind, "tool audit event kind_payload")
            .with_context(|| format!("serialize tool audit event {} kind", event.id.as_str()))?;
        let kind_str = event.kind.as_str();
        let effect_class_wire = enum_to_wire(&event.effect_class)?;

        sqlx::query!(
            r"
INSERT INTO agent_sdk_tool_audit_events (
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind, kind_payload,
    provider, model, input, output, error, recorded_at
) VALUES (
    $1, $2, $3, $4, $5,
    $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16
)
",
            event.id.as_str(),
            event.operation_id,
            event.task_id.as_str(),
            event.parent_task_id.as_str(),
            thread_key(&event.thread_id),
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
        .with_context(|| format!("record tool audit event {}", event.id.as_str()))?;
        Ok(())
    }

    async fn list_by_operation(&self, operation_id: &str) -> Result<Vec<ToolAuditEvent>> {
        let rows = sqlx::query_as!(
            ToolAuditEventRecord,
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE operation_id = $1
ORDER BY seq ASC
",
            operation_id,
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by operation {operation_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<ToolAuditEvent>> {
        let rows = sqlx::query_as!(
            ToolAuditEventRecord,
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE task_id = $1
ORDER BY seq ASC
",
            task_id.as_str(),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by task {task_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<ToolAuditEvent>> {
        let rows = sqlx::query_as!(
            ToolAuditEventRecord,
            r"
SELECT
    id, operation_id, task_id, parent_task_id, thread_id,
    tool_call_id, tool_name, effect_class, kind_payload,
    provider, model, input, output, error, recorded_at
FROM agent_sdk_tool_audit_events
WHERE thread_id = $1
ORDER BY seq ASC
",
            thread_key(thread_id),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list tool audit events by thread {thread_id}"))?;
        rows.into_iter().map(TryInto::try_into).collect()
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

    fn try_from(record: ToolAuditEventRecord) -> Result<Self> {
        Ok(Self {
            id: ToolAuditEventId::from_string(record.id),
            operation_id: record.operation_id,
            task_id: AgentTaskId::from_string(record.task_id),
            parent_task_id: AgentTaskId::from_string(record.parent_task_id),
            thread_id: ThreadId::from_string(record.thread_id),
            tool_call_id: record.tool_call_id,
            tool_name: record.tool_name,
            effect_class: enum_from_wire(&record.effect_class, "tool audit effect_class")?,
            kind: json_from_value(record.kind_payload, "tool audit event kind")?,
            provider: record.provider,
            model: record.model,
            input: record.input,
            output: record.output,
            error: record.error,
            recorded_at: record.recorded_at,
        })
    }
}

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

    fn try_from(record: TaskRecord) -> Result<Self> {
        let task = Self {
            id: AgentTaskId::from_string(record.id),
            kind: enum_from_wire(&record.kind, "task kind")?,
            status: enum_from_wire(&record.status, "task status")?,
            parent_id: record.parent_id.map(AgentTaskId::from_string),
            root_id: AgentTaskId::from_string(record.root_id),
            depth: u32_from_i64(record.depth, "task depth")?,
            thread_id: ThreadId::from_string(record.thread_id),
            submitted_input: json_from_value::<Vec<SubmittedInputItem>>(
                record.submitted_input_json,
                "task submitted_input",
            )?,
            caller_metadata: record.caller_metadata_json,
            worker_id: record.worker_id.map(WorkerId::from_string),
            lease_id: record.lease_id.map(LeaseId::from_string),
            lease_expires_at: record.lease_expires_at,
            last_heartbeat_at: record.last_heartbeat_at,
            state: json_from_value(record.state_json, "task state")?,
            attempt: u32_from_i64(record.attempt, "task attempt")?,
            max_attempts: u32_from_i64(record.max_attempts, "task max_attempts")?,
            last_error: record.last_error,
            pending_child_count: u32_from_i64(
                record.pending_child_count,
                "task pending_child_count",
            )?,
            spawn_index: record
                .spawn_index
                .map(|value| u32_from_i64(value, "task spawn_index"))
                .transpose()?,
            result_payload: record.result_payload,
            otel_traceparent: record.otel_traceparent,
            created_at: record.created_at,
            updated_at: record.updated_at,
            completed_at: record.completed_at,
        };
        task.validate()
            .context("rehydrate task from postgres row validation")?;
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

    fn try_from(record: ThreadRecord) -> Result<Self> {
        let thread = Self {
            thread_id: ThreadId::from_string(record.thread_id),
            status: enum_from_wire(&record.status, "thread status")?,
            committed_turns: u32_from_i64(record.committed_turns, "thread committed_turns")?,
            total_usage: TokenUsage {
                input_tokens: u32_from_i64(record.total_input_tokens, "thread input tokens")?,
                output_tokens: u32_from_i64(record.total_output_tokens, "thread output tokens")?,
                ..Default::default()
            },
            created_at: record.created_at,
            updated_at: record.updated_at,
        };
        thread
            .validate()
            .context("rehydrate thread from postgres row validation")?;
        Ok(thread)
    }
}

#[derive(Debug, FromRow)]
struct MessageHeadRecord {
    thread_id: String,
    history_json: serde_json::Value,
    /// In-flight draft snapshot, NULL when no turn is suspended.
    /// Populated by the worker at every tool-boundary suspension and
    /// cleared atomically by the completed-turn transaction.
    draft_messages_json: Option<serde_json::Value>,
    version: i64,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
}

impl TryFrom<MessageHeadRecord> for MessageProjection {
    type Error = anyhow::Error;

    fn try_from(record: MessageHeadRecord) -> Result<Self> {
        let draft_messages = match record.draft_messages_json {
            Some(value) => json_from_value(value, "message head draft messages")?,
            None => Vec::new(),
        };
        Ok(Self {
            thread_id: ThreadId::from_string(record.thread_id),
            messages: json_from_value(record.history_json, "message head history")?,
            draft_messages,
            version: u64_from_i64(record.version, "message head version")?,
            created_at: record.created_at,
            updated_at: record.updated_at,
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

    fn try_from(record: TurnAttemptRecord) -> Result<Self> {
        let attempt = Self {
            id: TurnAttemptId::from_string(record.id),
            task_id: AgentTaskId::from_string(record.task_id),
            attempt_number: u32_from_i64(record.attempt_number, "attempt_number")?,
            provider: record.provider,
            requested_model: record.requested_model,
            request_blob: record.request_blob,
            response_blob: record.response_blob,
            response_id: record.response_id,
            response_model: record.response_model,
            stop_reason: record
                .stop_reason
                .map(|value| enum_from_wire(&value, "turn attempt stop_reason"))
                .transpose()?,
            outcome: record
                .outcome
                .map(|value| enum_from_wire(&value, "turn attempt outcome"))
                .transpose()?,
            input_tokens: record
                .input_tokens
                .map(|value| u32_from_i64(value, "turn attempt input_tokens"))
                .transpose()?,
            output_tokens: record
                .output_tokens
                .map(|value| u32_from_i64(value, "turn attempt output_tokens"))
                .transpose()?,
            cached_input_tokens: record
                .cached_input_tokens
                .map(|value| u32_from_i64(value, "turn attempt cached_input_tokens"))
                .transpose()?,
            opened_at: record.opened_at,
            closed_at: record.closed_at,
            duration_ms: record
                .duration_ms
                .map(|value| u64_from_i64(value, "turn attempt duration_ms"))
                .transpose()?,
            otel_trace_id: record.otel_trace_id,
            otel_span_id: record.otel_span_id,
        };
        attempt
            .validate()
            .context("rehydrate turn attempt from postgres row validation")?;
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

    fn try_from(record: CheckpointRecord) -> Result<Self> {
        let checkpoint = Self {
            id: CheckpointId::from_string(record.id),
            thread_id: ThreadId::from_string(record.thread_id),
            turn_number: u32_from_i64(record.turn_number, "checkpoint turn_number")?,
            task_id: AgentTaskId::from_string(record.task_id),
            messages: json_from_value(record.messages_json, "checkpoint messages")?,
            agent_state_snapshot: record.agent_state_snapshot,
            turn_usage: TokenUsage {
                input_tokens: u32_from_i64(
                    record.turn_input_tokens,
                    "checkpoint turn_input_tokens",
                )?,
                output_tokens: u32_from_i64(
                    record.turn_output_tokens,
                    "checkpoint turn_output_tokens",
                )?,
                ..Default::default()
            },
            kind: CheckpointKind::parse(&record.kind)
                .context("rehydrate checkpoint kind from postgres row")?,
            created_at: record.created_at,
        };
        checkpoint
            .validate()
            .context("rehydrate checkpoint from postgres row validation")?;
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

    fn try_from(record: CommittedEventRecord) -> Result<Self> {
        Ok(Self {
            event_id: uuid::Uuid::parse_str(&record.event_id)
                .context("parse committed event UUID")?,
            thread_id: ThreadId::from_string(record.thread_id),
            sequence: u64_from_i64(record.sequence, "committed event sequence")?,
            timestamp: record.committed_at,
            event: json_from_value(record.event_json, "committed event payload")?,
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

    fn try_from(record: OutboxRecord) -> Result<Self> {
        let event_id = record
            .event_id
            .as_deref()
            .map(uuid::Uuid::parse_str)
            .transpose()
            .context("parse outbox event UUID")?;
        let sequence = record
            .sequence
            .map(|seq| u64_from_i64(seq, "outbox sequence"))
            .transpose()?;
        Ok(Self {
            id: OutboxRowId::from_string(record.id),
            kind: OutboxMessageKind::parse_wire(&record.kind).context("parse outbox kind")?,
            thread_id: ThreadId::from_string(record.thread_id),
            event_id,
            sequence,
            status: enum_from_wire(&record.status, "outbox status")?,
            payload_json: record.payload_json,
            created_at: record.created_at,
            next_attempt_at: record.next_attempt_at,
            attempt_count: u32_from_i64(record.attempt_count, "outbox attempt_count")?,
            max_attempts: u32_from_i64(record.max_attempts, "outbox max_attempts")?,
            last_error: record.last_error,
            claimed_by: record.claimed_by,
            claimed_at: record.claimed_at,
            delivered_at: record.delivered_at,
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

    fn try_from(record: RetentionCursorRecord) -> Result<Self> {
        Ok(Self {
            thread_id: ThreadId::from_string(record.thread_id),
            retention_floor: u64_from_i64(record.retention_floor, "retention floor")?,
            updated_at: record.updated_at,
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

    fn try_from(record: ExecutionIntentRecord) -> Result<Self> {
        Ok(Self {
            operation_id: OperationId(record.operation_id),
            effect_class: enum_from_wire(&record.effect_class, "execution intent effect_class")?,
            tool_call_id: record.tool_call_id,
            child_task_id: AgentTaskId::from_string(record.child_task_id),
            tool_name: record.tool_name,
            input: record.input,
            status: enum_from_wire(&record.status, "execution intent status")?,
            error: record.error,
            created_at: record.created_at,
            updated_at: record.updated_at,
        })
    }
}

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
    let value = serde_json::to_value(value).context("serialize enum to wire string")?;
    let wire = value
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

fn optional_u32_to_i64(value: Option<u32>) -> Option<i64> {
    value.map(i64::from)
}

fn optional_u64_to_i64(value: Option<u64>, label: &str) -> Result<Option<i64>> {
    value.map(|value| i64_from_u64(value, label)).transpose()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InjectedCommitFailure {
    AttemptClose,
    ThreadAdvance,
    MessageCommit,
    CheckpointInsert,
    EventCommit,
}

fn maybe_inject_failure(
    configured: Option<InjectedCommitFailure>,
    current: InjectedCommitFailure,
) -> Result<()> {
    if configured == Some(current) {
        return Err(anyhow!(
            "injected postgres completed-turn failure at {current:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::events::AgentEvent;
    use anyhow::{Context, Result};
    use sqlx::Connection;
    use sqlx::PgConnection;
    use time::Duration;
    use uuid::Uuid;

    use agent_server::journal::checkpoint_store::CheckpointStore;
    use agent_server::journal::commit::commit_completed_turn;
    use agent_server::journal::event_repository::InMemoryEventRepository;
    use agent_server::journal::execution_intent::ExecutionIntentStore;
    use agent_server::journal::message_store::MessageProjectionStore;
    use agent_server::journal::recovery::FailureReason;
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::thread_store::ThreadStore;
    use agent_server::journal::turn_attempt::{CloseAttemptParams, OpenAttemptParams};
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;
    use agent_server::journal::{AgentTask, LeaseId, TaskStatus, TurnAttemptOutcome, WorkerId};

    use super::*;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_id(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread_id(name), t_plus(secs), 3)
    }

    fn close_params() -> CloseAttemptParams {
        CloseAttemptParams {
            response_blob: serde_json::json!({"id": "msg_pg_1"}),
            response_id: Some("msg_pg_1".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 120,
            output_tokens: 60,
            cached_input_tokens: 12,
        }
    }

    fn open_params(task_id: &AgentTaskId) -> OpenAttemptParams {
        OpenAttemptParams {
            task_id: task_id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            request_blob: serde_json::json!({"messages": []}),
            now: t0(),
            otel_trace_id: None,
            otel_span_id: None,
        }
    }

    fn drop_test_schema(database_url: String, schema: String) {
        let _ = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                let Ok(mut conn) = PgConnection::connect(&database_url).await else {
                    return;
                };
                let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                    "DROP SCHEMA IF EXISTS {schema} CASCADE"
                )))
                .execute(&mut conn)
                .await;
            });
        })
        .join();
    }

    struct PostgresTestSchema {
        schema: String,
        database_url: String,
    }

    impl Drop for PostgresTestSchema {
        fn drop(&mut self) {
            drop_test_schema(self.database_url.clone(), self.schema.clone());
        }
    }

    async fn test_store() -> Result<Option<(PostgresDurableStore, PostgresTestSchema)>> {
        let Ok(database_url) = env::var("TEST_DATABASE_URL").or_else(|_| env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };

        let schema = format!("eng_7985_{}", Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&database_url)
            .await
            .context("connect postgres admin for tests")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create test schema {schema}"))?;
        drop(admin);

        let search_path = schema.clone();
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .after_connect(move |conn, _meta| {
                let sql = format!("SET search_path TO {search_path}");
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&database_url)
            .await
            .context("connect postgres test pool")?;
        let store = PostgresDurableStore::from_pool(pool);
        store
            .migrate()
            .await
            .context("migrate postgres test store")?;
        Ok(Some((
            store,
            PostgresTestSchema {
                schema,
                database_url,
            },
        )))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_submits_on_same_thread_serialize_with_exactly_one_pending() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let count = 10usize;
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                let root = AgentTask::new_root_turn(
                    thread_id("t-pg-race"),
                    t_plus(i64::try_from(idx).context("idx to i64")?),
                    3,
                );
                store.submit_root_turn(root).await
            }));
        }

        let mut results = Vec::with_capacity(count);
        for handle in handles {
            results.push(handle.await.context("join submit")??);
        }

        let pending = results
            .iter()
            .filter(|task| task.status == TaskStatus::Pending)
            .count();
        let queued = results
            .iter()
            .filter(|task| task.status == TaskStatus::Queued)
            .count();
        assert_eq!(pending, 1);
        assert_eq!(queued, count - 1);

        let active = store
            .active_root_for_thread(&thread_id("t-pg-race"))
            .await?
            .context("active root")?;
        assert_eq!(active.status, TaskStatus::Pending);

        let queue = store.list_queued_roots(&thread_id("t-pg-race")).await?;
        assert_eq!(queue.len(), count - 1);
        for pair in queue.windows(2) {
            assert!(pair[0].created_at <= pair[1].created_at);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_promotes_fire_exactly_once_after_slot_frees() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let first = fresh_root("t-pg-promo", 0);
        let second = fresh_root("t-pg-promo", 1);
        let second_id = second.id.clone();
        store.submit_root_turn(first.clone()).await?;
        store.submit_root_turn(second).await?;

        let running = store
            .try_acquire_task(
                &first.id,
                WorkerId::from_string("w-pg-promo"),
                LeaseId::from_string("l-pg-promo"),
                t_plus(60),
                t_plus(2),
            )
            .await?
            .context("claimed first root")?;
        store.update(running.complete(t_plus(3))?).await?;

        let count = 8usize;
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .promote_next_queued_root(&thread_id("t-pg-promo"), t_plus(4))
                    .await
            }));
        }

        let mut promoted = 0usize;
        for handle in handles {
            if handle.await.context("join promote")??.is_some() {
                promoted += 1;
            }
        }
        assert_eq!(promoted, 1);

        let active = store
            .active_root_for_thread(&thread_id("t-pg-promo"))
            .await?
            .context("promoted root")?;
        assert_eq!(active.id, second_id);
        assert_eq!(active.status, TaskStatus::Pending);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn try_acquire_task_is_exclusive_across_two_workers() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let root = fresh_root("t-pg-acquire", 0);
        let admitted = store.submit_root_turn(root).await?;
        let id = admitted.id.clone();

        let store_a = store.clone();
        let store_b = store.clone();

        let a = tokio::spawn(async move {
            store_a
                .try_acquire_task(
                    &id,
                    WorkerId::from_string("w-a"),
                    LeaseId::from_string("l-a"),
                    t_plus(60),
                    t_plus(1),
                )
                .await
        });
        let b = tokio::spawn(async move {
            store_b
                .try_acquire_task(
                    &admitted.id,
                    WorkerId::from_string("w-b"),
                    LeaseId::from_string("l-b"),
                    t_plus(60),
                    t_plus(1),
                )
                .await
        });

        let r1 = a.await.context("join a")??;
        let r2 = b.await.context("join b")??;
        let winners = usize::from(r1.is_some()) + usize::from(r2.is_some());
        assert_eq!(winners, 1);

        let winner = r1.or(r2).context("winner")?;
        assert_eq!(winner.status, TaskStatus::Running);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn acquire_next_runnable_commits_fail_closed_heads_before_returning_none() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let mut ids = Vec::new();
        for (idx, name) in ["a", "b", "c"].iter().enumerate() {
            let secs = i64::try_from(idx).context("enumerate fits i64")? * 10;
            let root = AgentTask::new_root_turn(
                thread_id(&format!("t-pg-exhaust-{name}")),
                t_plus(secs),
                1,
            );
            let id = root.id.clone();
            store
                .submit_root_turn(root)
                .await
                .context("submit exhausted root")?;
            let claimed = store
                .try_acquire_task(
                    &id,
                    WorkerId::from_string(format!("w-{name}")),
                    LeaseId::from_string(format!("l-{name}")),
                    t_plus(secs + 50),
                    t_plus(secs + 1),
                )
                .await
                .context("acquire exhausted root")?
                .context("exhausted root claimed")?;
            store
                .update(
                    claimed
                        .release_lease(t_plus(secs + 2))
                        .context("release exhausted root")?,
                )
                .await
                .context("persist released exhausted root")?;
            ids.push(id);
        }

        let result = store
            .acquire_next_runnable(
                WorkerId::from_string("w-scan"),
                LeaseId::from_string("l-scan"),
                t_plus(1_000),
                t_plus(900),
            )
            .await
            .context("scan exhausted queue")?;
        assert!(result.is_none(), "every head must be failed closed");

        for id in &ids {
            let row = AgentTaskStore::get(&store, id)
                .await
                .context("get exhausted row")?
                .context("row exists")?;
            assert_eq!(
                row.status,
                TaskStatus::Failed,
                "expected Failed for {id}, got {:?}",
                row.status
            );
            let message = row
                .last_error
                .as_deref()
                .context("failed rows carry last_error")?;
            assert!(
                message.starts_with(FailureReason::RetryBudgetExhausted.error_prefix()),
                "unexpected fail-closed reason for {id}: {message}"
            );
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn completed_turn_transaction_rolls_back_partial_writes_on_error() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let root = fresh_root("t-pg-commit", 0);
        let admitted = store.submit_root_turn(root.clone()).await?;
        let running = store
            .try_acquire_task(
                &admitted.id,
                WorkerId::from_string("w-commit"),
                LeaseId::from_string("l-commit"),
                t_plus(60),
                t_plus(1),
            )
            .await?
            .context("acquire running task")?;
        let attempt = store.open_attempt(open_params(&running.id)).await?;

        let err = store
            .commit_completed_turn_atomic_with_failure(
                CompletedTurnCommit {
                    checkpoint_kind: CheckpointKind::FullTurn,
                    thread_id: running.thread_id.clone(),
                    task_id: running.id.clone(),
                    expected_turn: 1,
                    turn_attempt_id: attempt.id.clone(),
                    close_attempt_params: close_params(),
                    messages: vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
                    turn_usage: TokenUsage {
                        input_tokens: 120,
                        output_tokens: 60,
                        ..Default::default()
                    },
                    agent_state_snapshot: serde_json::json!({"turn": 1}),
                    events: Vec::new(),
                    outbox_max_attempts: 3,
                    owner_guard: None,
                    now: t_plus(2),
                },
                Some(InjectedCommitFailure::ThreadAdvance),
            )
            .await
            .err()
            .context("fault injection should fail")?;
        assert!(format!("{err:#}").contains("injected postgres completed-turn failure"));

        let attempt_after = TurnAttemptStore::get(&store, &attempt.id)
            .await?
            .context("attempt after")?;
        assert!(
            attempt_after.is_open(),
            "attempt close should have rolled back"
        );

        let thread = ThreadStore::get(&store, &running.thread_id)
            .await?
            .context("thread row should exist from task bootstrap")?;
        assert_eq!(thread.committed_turns, 0);
        assert_eq!(thread.total_usage, TokenUsage::default());

        let projection = MessageProjectionStore::get(&store, &running.thread_id).await?;
        assert!(projection.is_none(), "message head must roll back");

        let checkpoints = CheckpointStore::list_by_thread(&store, &running.thread_id).await?;
        assert!(checkpoints.is_empty(), "checkpoint insert must roll back");

        Ok(())
    }

    /// Phase 10 · D regression: events exist **iff** the turn committed.
    ///
    /// A crash injected *after* the state projections but *before* the
    /// transaction commits (at the `EventCommit` step) must roll the
    /// whole turn back — leaving zero committed turns AND zero events.
    /// A clean commit must then persist both the turn and its events
    /// (plus the coalesced advisory outbox row) atomically. This is the
    /// invariant the old split-transaction design could violate: a
    /// committed turn with no persisted events.
    #[tokio::test(flavor = "multi_thread")]
    async fn events_exist_iff_turn_committed_postgres() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let root = fresh_root("t-pg-events-iff", 0);
        let admitted = store.submit_root_turn(root.clone()).await?;
        let running = store
            .try_acquire_task(
                &admitted.id,
                WorkerId::from_string("w-events-iff"),
                LeaseId::from_string("l-events-iff"),
                t_plus(60),
                t_plus(1),
            )
            .await?
            .context("acquire running task")?;
        let attempt = store.open_attempt(open_params(&running.id)).await?;

        let thread_id = running.thread_id.clone();
        let commit_params = |now| CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_id.clone(),
            task_id: running.id.clone(),
            expected_turn: 1,
            turn_attempt_id: attempt.id.clone(),
            close_attempt_params: close_params(),
            messages: vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
            turn_usage: TokenUsage {
                input_tokens: 120,
                output_tokens: 60,
                ..Default::default()
            },
            agent_state_snapshot: serde_json::json!({"turn": 1}),
            events: vec![AgentEvent::Start {
                thread_id: thread_id.clone(),
                turn: 1,
            }],
            outbox_max_attempts: 3,
            owner_guard: None,
            now,
        };

        // ── Phase 1: crash inside the commit transaction ─────────────
        let err = store
            .commit_completed_turn_atomic_with_failure(
                commit_params(t_plus(2)),
                Some(InjectedCommitFailure::EventCommit),
            )
            .await
            .err()
            .context("event-commit fault injection should fail")?;
        assert!(format!("{err:#}").contains("injected postgres completed-turn failure"));

        // The turn must NOT have committed, and ZERO events may persist —
        // never a committed turn with no events (the old design's bug).
        let thread = ThreadStore::get(&store, &running.thread_id)
            .await?
            .context("thread row exists from task bootstrap")?;
        assert_eq!(thread.committed_turns, 0, "turn must roll back");
        let events = EventRepository::get_events(&store, &running.thread_id).await?;
        assert!(events.is_empty(), "no events when the turn rolled back");
        let checkpoints = CheckpointStore::list_by_thread(&store, &running.thread_id).await?;
        assert!(checkpoints.is_empty(), "checkpoint must roll back too");

        // ── Phase 2: a clean commit persists turn + events together ──
        let outcome = store
            .commit_completed_turn_atomic(commit_params(t_plus(3)))
            .await
            .context("clean commit")?;
        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.committed_events.len(), 1, "returns committed event");

        // Turn committed → events exist, and the coalesced advisory
        // outbox row landed in the SAME transaction.
        let events = EventRepository::get_events(&store, &running.thread_id).await?;
        assert_eq!(events.len(), 1, "committed turn has its persisted events");
        let outbox_rows = OutboxStore::list_by_thread(&store, &running.thread_id).await?;
        let thread_events_rows = outbox_rows
            .iter()
            .filter(|r| r.kind == OutboxMessageKind::ThreadEventsAvailable)
            .count();
        assert_eq!(thread_events_rows, 1, "one advisory row per committed turn");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn commit_completed_turn_uses_atomic_postgres_hook() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };

        let root = fresh_root("t-pg-hook", 0);
        let admitted = store.submit_root_turn(root.clone()).await?;
        let running = store
            .try_acquire_task(
                &admitted.id,
                WorkerId::from_string("w-hook"),
                LeaseId::from_string("l-hook"),
                t_plus(60),
                t_plus(1),
            )
            .await?
            .context("acquire hook task")?;
        let attempt = store.open_attempt(open_params(&running.id)).await?;
        let event_repo = InMemoryEventRepository::new();

        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: running.thread_id.clone(),
                task_id: running.id.clone(),
                expected_turn: 1,
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: close_params(),
                messages: vec![llm::Message::user("hi"), llm::Message::assistant("there")],
                turn_usage: TokenUsage {
                    input_tokens: 120,
                    output_tokens: 60,
                    ..Default::default()
                },
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events: vec![AgentEvent::Start {
                    thread_id: running.thread_id.clone(),
                    turn: 1,
                }],
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(2),
            },
            &store,
            &store,
            &store,
            &store,
            &event_repo,
        )
        .await?;

        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.checkpoint.turn_number, 1);
        assert!(outcome.closed_attempt.is_closed());

        let history = MessageProjectionStore::get_history(&store, &running.thread_id).await?;
        assert_eq!(history.len(), 2);

        Ok(())
    }

    // ── ExecutionIntentStore ─────────────────────────────────────

    fn make_test_intent(task_name: &str, secs: i64) -> ExecutionIntent {
        use agent_server::journal::execution_intent::{IntentStatus, ToolEffectClass};

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
        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let intent = make_test_intent("persist_op", 10);
        store.persist_intent(&intent).await?;

        let loaded = store
            .get_intent(&intent.operation_id)
            .await?
            .context("intent should be present after persist")?;
        assert_eq!(loaded.operation_id, intent.operation_id);
        assert_eq!(loaded.tool_name, "test_tool");
        assert_eq!(
            loaded.status,
            agent_server::journal::execution_intent::IntentStatus::Pending
        );

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_persist_and_get_by_task() -> Result<()> {
        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

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
        use agent_server::journal::execution_intent::IntentStatus;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

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
        use agent_server::journal::execution_intent::IntentStatus;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

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
        let Some((store, guard)) = test_store().await? else {
            return Ok(());
        };

        let intent = make_test_intent("restart", 20);
        store.persist_intent(&intent).await?;

        // Simulate restart by creating a new store from the same pool.
        let store2 = PostgresDurableStore::from_pool(store.pool().clone());

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

        drop(guard);
        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_duplicate_persist_is_insert_if_absent() -> Result<()> {
        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let mut intent = make_test_intent("claim", 25);
        store.persist_intent(&intent).await?;

        // A second claim on the same operation id must lose with a
        // conflict error and must NOT clobber the in-flight record.
        // Status transitions go through `update_intent`.
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
        assert_eq!(
            loaded.status,
            agent_server::journal::execution_intent::IntentStatus::Pending
        );

        Ok(())
    }

    #[tokio::test]
    async fn execution_intent_get_nonexistent_returns_none() -> Result<()> {
        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

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
        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

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

    fn tool_audit_event(
        operation_id: &str,
        task: &str,
        thread: &str,
        kind: agent_server::journal::tool_audit::ToolAuditEventKind,
        secs: i64,
    ) -> agent_server::journal::tool_audit::ToolAuditEvent {
        use agent_server::journal::execution_intent::ToolEffectClass;
        use agent_server::journal::tool_audit::{ToolAuditEvent, ToolAuditEventParams};

        ToolAuditEvent::new(ToolAuditEventParams {
            operation_id: operation_id.into(),
            task_id: AgentTaskId::from_string(format!("task_{task}")),
            parent_task_id: AgentTaskId::from_string(format!("task_{task}_parent")),
            thread_id: thread_id(thread),
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

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let operation = "task_lifecycle:call_1";
        let lifecycle = vec![
            (ToolAuditEventKind::Dispatched, 10),
            (ToolAuditEventKind::ConfirmationRequested, 11),
            (ToolAuditEventKind::ConfirmationApproved, 12),
            (ToolAuditEventKind::ExecutionStarted, 13),
            (ToolAuditEventKind::Completed, 14),
        ];

        for (kind, secs) in lifecycle {
            let event = tool_audit_event(operation, "lifecycle", "tool-audit-life", kind, secs);
            ToolAuditEventStore::record_event(&store, &event).await?;
        }

        let reopened = PostgresDurableStore::from_pool(store.pool().clone());
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

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let dispatched = tool_audit_event(
            "task_skew:call_1",
            "skew",
            "tool-audit-skew",
            ToolAuditEventKind::Dispatched,
            20,
        );
        let completed = tool_audit_event(
            "task_skew:call_1",
            "skew",
            "tool-audit-skew",
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
    async fn tool_audit_provenance_and_query_paths_persist() -> Result<()> {
        use agent_server::journal::tool_audit::{ToolAuditEventKind, ToolAuditEventStore};

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let event = tool_audit_event(
            "task_prov:call_prov",
            "prov",
            "tool-audit-prov",
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
        assert_eq!(loaded.tool_name, "transfer");
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

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let e_a = tool_audit_event(
            "task_a:call_1",
            "a",
            "tool-audit-isolate",
            ToolAuditEventKind::Dispatched,
            30,
        );
        let e_b = tool_audit_event(
            "task_b:call_1",
            "b",
            "tool-audit-isolate",
            ToolAuditEventKind::Dispatched,
            31,
        );
        let e_other_thread = tool_audit_event(
            "task_c:call_1",
            "c",
            "tool-audit-elsewhere",
            ToolAuditEventKind::Dispatched,
            32,
        );
        ToolAuditEventStore::record_event(&store, &e_a).await?;
        ToolAuditEventStore::record_event(&store, &e_b).await?;
        ToolAuditEventStore::record_event(&store, &e_other_thread).await?;

        let by_a = ToolAuditEventStore::list_by_task(&store, &e_a.task_id).await?;
        assert_eq!(by_a.len(), 1);
        assert_eq!(by_a[0].task_id, e_a.task_id);

        let by_thread = ToolAuditEventStore::list_by_thread(&store, &e_a.thread_id).await?;
        assert_eq!(
            by_thread.len(),
            2,
            "two events on tool-audit-isolate thread"
        );

        let elsewhere =
            ToolAuditEventStore::list_by_thread(&store, &e_other_thread.thread_id).await?;
        assert_eq!(elsewhere.len(), 1);

        let unknown_op = store.list_by_operation("task_none:call_none").await?;
        assert!(unknown_op.is_empty());

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

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let decorator = RedactingToolAuditEventStore::baseline(Arc::new(store.clone()));

        let params = ToolAuditEventParams {
            operation_id: "task_redact:call_1".into(),
            task_id: AgentTaskId::from_string("task_redact"),
            parent_task_id: AgentTaskId::from_string("task_redact_parent"),
            thread_id: thread_id("tool-audit-redact"),
            tool_call_id: "call_1".into(),
            tool_name: "transfer".into(),
            effect_class: agent_server::journal::execution_intent::ToolEffectClass::SideEffecting,
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
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, ThreadEventsAvailablePayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-outbox-coalesce");
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
        assert_eq!(row.event_id, Some(outcome.committed_events[0].event_id),);

        // Advisory payload must round-trip back to the typed message.
        let message = OutboxMessage::from_payload_json(row.kind, row.payload_json.clone())?;
        assert_eq!(
            message,
            OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
                thread_id: thread_id.clone(),
                last_sequence: 2,
            }),
        );

        // Exactly one outbox row should exist for the thread — not three.
        let rows = OutboxStore::list_by_thread(&store, &thread_id).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, OutboxMessageKind::ThreadEventsAvailable);

        Ok(())
    }

    #[tokio::test]
    async fn outbox_insert_batch_accepts_task_wakeup_with_null_event_refs() -> Result<()> {
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, TaskWakeupPayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-task-wakeup");
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
        let row = &rows[0];
        assert_eq!(row.kind, OutboxMessageKind::TaskWakeup);
        assert!(row.event_id.is_none());
        assert!(row.sequence.is_none());

        let stored = OutboxStore::get(&store, &row.id)
            .await?
            .context("row missing after insert")?;
        assert_eq!(stored.kind, OutboxMessageKind::TaskWakeup);
        assert!(stored.event_id.is_none());
        assert!(stored.sequence.is_none());
        assert_eq!(stored.payload_json, payload);

        Ok(())
    }

    #[tokio::test]
    async fn outbox_insert_batch_rejects_task_wakeup_with_event_id() -> Result<()> {
        use agent_server::journal::outbox_message::{OutboxMessageKind, TaskWakeupPayload};
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-task-wakeup-bad");
        ThreadStore::get_or_create(&store, &thread_id, t0()).await?;

        let task_id = AgentTaskId::new();
        let payload = serde_json::to_value(&TaskWakeupPayload {
            task_id,
            thread_id: thread_id.clone(),
        })?;

        let result = OutboxStore::insert_batch(
            &store,
            vec![NewOutboxRow {
                kind: OutboxMessageKind::TaskWakeup,
                thread_id,
                // Phase 8.1 contract: TaskWakeup rows must NOT carry
                // event references; the in-memory invariant rejects
                // this before it reaches Postgres.
                event_id: Some(uuid::Uuid::now_v7()),
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
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-outbox-delivered-missing");
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
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-outbox-failed-missing");
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
    async fn outbox_list_by_thread_orders_task_wakeups_after_thread_events() -> Result<()> {
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::outbox_message::{
            OutboxMessage, OutboxMessageKind, TaskWakeupPayload,
        };
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-outbox-order");
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
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-reclaim-stale");
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

        // Worker-A claims at t+1 then "crashes" before marking.
        let claimed = OutboxStore::claim_pending(&store, "worker-a", 10, t_plus(1)).await?;
        assert_eq!(claimed.len(), 1);
        let id = claimed[0].id.clone();

        // Lease = 10s.  At t+30 the claim is 29s old, so stale.
        let reclaimed =
            OutboxStore::reclaim_expired_claims(&store, t_plus(30), time::Duration::seconds(10))
                .await?;
        assert_eq!(reclaimed, 1);

        let row = OutboxStore::get(&store, &id)
            .await?
            .context("row should still exist")?;
        assert_eq!(
            row.status,
            agent_server::journal::outbox::OutboxStatus::Pending
        );
        assert!(row.claimed_by.is_none());
        assert!(row.claimed_at.is_none());
        assert_eq!(row.attempt_count, 0);

        // A fresh worker can now re-claim the row — duplicate republish on recovery.
        let reclaimed_rows = OutboxStore::claim_pending(&store, "worker-b", 10, t_plus(31)).await?;
        assert_eq!(reclaimed_rows.len(), 1);
        assert_eq!(reclaimed_rows[0].id, id);

        Ok(())
    }

    #[tokio::test]
    async fn event_sequence_does_not_regress_after_full_retention_purge() -> Result<()> {
        use agent_server::journal::event_repository::EventRepository;
        use agent_server::journal::retention::RetentionStore;
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-seq-no-regress");
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
        // last assigned sequence — MAX(sequence) falls back to NULL
        // on the empty table.
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
    async fn outbox_reclaim_skips_live_claims_and_terminal_rows() -> Result<()> {
        use agent_server::journal::event_outbox_transaction::EventOutboxCommit;
        use agent_server::journal::thread_store::ThreadStore;

        let Some((store, _guard)) = test_store().await? else {
            return Ok(());
        };

        let thread_id = thread_id("t-pg-reclaim-live");
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

        // 5s after claim, lease of 30s still live — should reclaim 0.
        let reclaimed =
            OutboxStore::reclaim_expired_claims(&store, t_plus(6), time::Duration::seconds(30))
                .await?;
        assert_eq!(reclaimed, 0);

        Ok(())
    }

    // ── Phase 10 · E: durable idempotency (transactional with admission) ──

    fn submit_params(
        task: AgentTask,
        request_id: &str,
        fingerprint: &[u8],
        max_queued_depth: Option<u32>,
    ) -> SubmitRootTurnParams {
        let result_json = serde_json::json!({ "task_id": task.id.to_string() });
        SubmitRootTurnParams {
            task,
            idempotency: Some(SubmitRootIdempotency {
                request_id: request_id.to_owned(),
                fingerprint: fingerprint.to_vec(),
                result_json,
            }),
            max_queued_depth,
        }
    }

    #[tokio::test]
    async fn idempotent_submit_dedups_and_replays() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };
        let tid = thread_id("pg-idem");
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let first_id = first.id.clone();
        let outcome = store
            .submit_root_turn_idempotent(submit_params(first, "pg-req", b"fp", None))
            .await
            .map_err(|error| anyhow!("first submit failed: {error}"))?;
        assert!(!outcome.replayed);

        let retry = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let replay = store
            .submit_root_turn_idempotent(submit_params(retry, "pg-req", b"fp", None))
            .await
            .map_err(|error| anyhow!("retry submit failed: {error}"))?;
        assert!(replay.replayed);
        assert_eq!(replay.task.id, first_id);

        let tasks = AgentTaskStore::list_by_thread(&store, &tid).await?;
        assert_eq!(tasks.iter().filter(|t| t.is_root()).count(), 1);
        Ok(())
    }

    /// Concurrent retries under the same `request_id` admit exactly one
    /// root — the strongest proof the dedup is transactional with
    /// admission (no TOCTOU). The `FOR UPDATE` thread lock serializes
    /// the racing claims so only the first commits the insert.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_idempotent_submits_admit_exactly_one_root() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };
        let tid = thread_id("pg-idem-race");
        let count = 8usize;
        let mut handles = Vec::with_capacity(count);
        for idx in 0..count {
            let store = store.clone();
            let tid = tid.clone();
            handles.push(tokio::spawn(async move {
                let root =
                    AgentTask::new_root_turn(tid, t_plus(i64::try_from(idx).context("idx")?), 3);
                store
                    .submit_root_turn_idempotent(submit_params(root, "race-req", b"fp", None))
                    .await
                    .map_err(|error| anyhow!("submit: {error}"))
            }));
        }

        let mut admitted = 0usize;
        let mut replayed = 0usize;
        for handle in handles {
            let outcome = handle.await.context("join submit")??;
            if outcome.replayed {
                replayed += 1;
            } else {
                admitted += 1;
            }
        }
        assert_eq!(admitted, 1, "exactly one racing retry admitted a root");
        assert_eq!(replayed, count - 1, "the rest replayed the original");

        let tasks = AgentTaskStore::list_by_thread(&store, &tid).await?;
        assert_eq!(tasks.iter().filter(|t| t.is_root()).count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn idempotent_submit_queue_depth_cap_rejects() -> Result<()> {
        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };
        let tid = thread_id("pg-queue-cap");
        let active = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        store
            .submit_root_turn_idempotent(submit_params(active, "c-active", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow!("active: {error}"))?;
        let queued = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let outcome = store
            .submit_root_turn_idempotent(submit_params(queued, "c-1", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow!("queued: {error}"))?;
        assert_eq!(outcome.queued_depth, 1);

        let overflow = AgentTask::new_root_turn(tid.clone(), t_plus(12), 3);
        let result = store
            .submit_root_turn_idempotent(submit_params(overflow, "c-2", b"fp", Some(1)))
            .await;
        assert!(matches!(
            result,
            Err(SubmitRootTurnError::QueueDepthExceeded {
                cap: 1,
                current_depth: 1
            })
        ));
        Ok(())
    }

    /// Phase 10 · D + E regression: the transport-facing
    /// `submit_root_turn_idempotent` path must also emit the durable
    /// `task_wakeup` advisory row inside the admission transaction when
    /// the root lands runnable (`Pending`) — exactly like the
    /// non-idempotent `submit_root_turn`. A root that lands `Queued`
    /// (parked behind the active slot) must NOT emit a wakeup; it is
    /// nudged on promotion instead.
    #[tokio::test]
    async fn idempotent_submit_emits_durable_task_wakeup_when_runnable() -> Result<()> {
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::OutboxMessageKind;

        let Some((store, _schema_guard)) = test_store().await? else {
            return Ok(());
        };
        let tid = thread_id("pg-idem-wakeup");

        // First root lands Pending → exactly one durable task_wakeup row.
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let outcome = store
            .submit_root_turn_idempotent(submit_params(first, "w-1", b"fp", None))
            .await
            .map_err(|error| anyhow!("first submit: {error}"))?;
        assert_eq!(outcome.task.status, TaskStatus::Pending);

        let wakeups_after_first = OutboxStore::list_by_thread(&store, &tid)
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(
            wakeups_after_first, 1,
            "idempotent admission of a runnable root must emit exactly one durable task_wakeup"
        );

        // Second root lands Queued (parked) → no additional wakeup.
        let queued = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let queued_outcome = store
            .submit_root_turn_idempotent(submit_params(queued, "w-2", b"fp", None))
            .await
            .map_err(|error| anyhow!("queued submit: {error}"))?;
        assert_eq!(queued_outcome.task.status, TaskStatus::Queued);

        let wakeups_after_queued = OutboxStore::list_by_thread(&store, &tid)
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
}
