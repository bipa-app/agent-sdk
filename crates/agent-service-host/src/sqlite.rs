//! `SQLite` embedded storage backend for local durability.
//!
//! This module provides [`SqliteDurableStore`] — a single struct that
//! implements all 10 store traits the journal and worker layers consume,
//! backed by an embedded `SQLite` database in WAL mode.
//!
//! # Architecture
//!
//! The `SQLite` backend mirrors the `PostgreSQL` backend's architecture:
//!
//! - **Same store traits.** Every trait in [`crate::stores::StoreRegistry`]
//!   is implemented by the same concrete type.
//! - **Same domain transitions.** All state-machine logic uses the shared
//!   pure-Rust helpers (`AgentTask::mark_running`, `Thread::apply_committed_turn`,
//!   etc.) — the `SQLite` backend only provides persistence and locking.
//! - **Separate migrations.** `SQLite`-dialect SQL lives in `migrations/sqlite/`,
//!   independently versioned from the `PostgreSQL` migrations but tested for
//!   structural parity.
//!
//! # Locking model
//!
//! `SQLite` in WAL mode serialises writes at the database level. The
//! `PostgreSQL` `FOR UPDATE` / `SKIP LOCKED` row-level locking patterns are
//! replaced by `BEGIN IMMEDIATE`, which acquires the write lock at
//! transaction start. This is acceptable for the single-process local
//! use case.
//!
//! # Feature flag
//!
//! This entire module is gated behind the `sqlite` cargo feature so
//! server builds that only need `PostgreSQL` do not pull in the `SQLite`
//! dependency.

pub mod migrations;
pub mod store;

pub use store::SqliteDurableStore;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use anyhow::{Context, Result, ensure};
    use sqlx::sqlite::SqlitePoolOptions;

    use super::migrations::{
        DURABLE_CORE_MIGRATOR, INPUT_INJECTION_KIND_MIGRATION_VERSION,
        TASK_TERMINAL_REASON_MIGRATION_VERSION, THREAD_PURGE_LIFECYCLE_MIGRATION_VERSION,
        durable_core_migrations, outbox_message_kind_migration,
    };

    #[test]
    fn sqlite_migrations_cover_every_expected_table() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|m| m.sql)
            .collect::<Vec<_>>()
            .join("\n");

        let expected_tables = [
            "agent_sdk_tasks",
            "agent_sdk_threads",
            "agent_sdk_message_heads",
            "agent_sdk_message_commits",
            "agent_sdk_turn_attempts",
            "agent_sdk_turn_checkpoints",
            "agent_sdk_committed_events",
            "agent_sdk_outbox",
            "agent_sdk_retention_cursors",
            "agent_sdk_execution_intents",
            "agent_sdk_tool_audit_events",
        ];

        for table in expected_tables {
            ensure!(
                sql_bundle.contains(&format!("CREATE TABLE {table}")),
                "missing CREATE TABLE for {table}",
            );
        }

        Ok(())
    }

    /// The bundle is ordered, gap-tolerant, and contains this PR's
    /// migration.
    ///
    /// Deliberately NOT a hardcoded version list. Sibling PRs each own a
    /// reserved version number and merge in an arbitrary order, so
    /// pinning the exact set makes every one of them break the others
    /// until rebased — a merge-order coupling that says nothing about
    /// whether the migrator is correct. What actually matters is that
    /// versions apply in a strictly increasing order (sqlx applies them
    /// in slice order, so a descending pair would run out of sequence)
    /// and that this PR's own migration is present.
    #[test]
    fn sqlite_executable_migration_bundle_is_ordered_and_contains_this_prs_migration() -> Result<()>
    {
        let versions: Vec<i64> = DURABLE_CORE_MIGRATOR
            .migrations
            .iter()
            .map(|migration| migration.version)
            .collect();

        ensure!(
            versions.windows(2).all(|pair| pair[0] < pair[1]),
            "migration versions must be strictly increasing (ordered and unique), got {versions:?}",
        );
        ensure!(
            versions.contains(&TASK_TERMINAL_REASON_MIGRATION_VERSION),
            "bundle is missing this PR's migration \
             {TASK_TERMINAL_REASON_MIGRATION_VERSION}, got {versions:?}",
        );
        ensure!(
            versions.contains(&INPUT_INJECTION_KIND_MIGRATION_VERSION),
            "bundle is missing this PR's migration \
             {INPUT_INJECTION_KIND_MIGRATION_VERSION}, got {versions:?}",
        );
        ensure!(
            versions.contains(&THREAD_PURGE_LIFECYCLE_MIGRATION_VERSION),
            "bundle is missing this PR's migration \
             {THREAD_PURGE_LIFECYCLE_MIGRATION_VERSION}, got {versions:?}",
        );
        Ok(())
    }

    #[test]
    fn sqlite_migration_uses_json_type_not_jsonb_typeof() -> Result<()> {
        // Strip SQL comments before checking for Postgres-only constructs.
        let sql_bundle: String = durable_core_migrations()
            .iter()
            .flat_map(|m| m.sql.lines())
            .filter(|line| !line.trim_start().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");

        ensure!(
            !sql_bundle.contains("jsonb_typeof"),
            "SQLite migrations must use json_type(), not jsonb_typeof()",
        );
        ensure!(
            !sql_bundle.contains("JSONB"),
            "SQLite migrations must use TEXT, not JSONB",
        );
        ensure!(
            !sql_bundle.contains("TIMESTAMPTZ"),
            "SQLite migrations must use TEXT, not TIMESTAMPTZ",
        );
        ensure!(
            !sql_bundle.contains("DEFERRABLE"),
            "SQLite does not support DEFERRABLE foreign keys",
        );
        Ok(())
    }

    #[test]
    fn sqlite_migration_preserves_all_constraint_names() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|m| m.sql)
            .collect::<Vec<_>>()
            .join("\n");

        // Spot-check critical constraints from both migrations.
        let required_constraints = [
            "agent_sdk_tasks_kind_check",
            "agent_sdk_tasks_status_check",
            "agent_sdk_tasks_root_identity_check",
            "agent_sdk_tasks_depth_kind_check",
            "agent_sdk_tasks_running_lease_check",
            "agent_sdk_tasks_waiting_state_check",
            "agent_sdk_threads_status_check",
            "agent_sdk_message_heads_version_check",
            "agent_sdk_committed_events_sequence_check",
            "agent_sdk_outbox_status_check",
            "agent_sdk_retention_cursors_floor_check",
        ];

        for name in required_constraints {
            ensure!(sql_bundle.contains(name), "missing constraint: {name}");
        }
        Ok(())
    }

    #[test]
    fn sqlite_outbox_kind_migration_rewrites_legacy_payloads_to_advisory_shape() -> Result<()> {
        let sql = outbox_message_kind_migration();
        ensure!(
            sql.contains("json_object('thread_id', thread_id, 'last_sequence', sequence)"),
            "SQLite outbox kind migration must rebuild legacy payload_json values",
        );
        Ok(())
    }

    #[test]
    fn sqlite_migration_preserves_all_index_names() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|m| m.sql)
            .collect::<Vec<_>>()
            .join("\n");

        let required_indexes = [
            "agent_sdk_tasks_by_thread_idx",
            "agent_sdk_tasks_by_parent_idx",
            "agent_sdk_tasks_by_status_idx",
            "agent_sdk_tasks_root_admission_slot_idx",
            "agent_sdk_tasks_queued_roots_fifo_idx",
            "agent_sdk_tasks_runnable_fifo_idx",
            "agent_sdk_tasks_running_lease_expiry_idx",
            "agent_sdk_tasks_root_tree_idx",
            "agent_sdk_tasks_subagent_child_root_waiting_idx",
            "agent_sdk_threads_active_idx",
            "agent_sdk_turn_checkpoints_latest_by_thread_idx",
            "agent_sdk_committed_events_replay_idx",
            "agent_sdk_outbox_relay_scan_idx",
            "agent_sdk_outbox_claimed_sweep_idx",
        ];

        for name in required_indexes {
            ensure!(sql_bundle.contains(name), "missing index: {name}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn terminal_reason_migration_backfills_all_terminal_rows() -> Result<()> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await?;
        let migrations = durable_core_migrations();
        let terminal_reason_index = migrations
            .iter()
            .position(|migration| migration.version == "0014")
            .context("terminal-reason migration must be registered as 0014")?;

        for migration in &migrations[..terminal_reason_index] {
            sqlx::raw_sql(migration.sql).execute(&pool).await?;
        }

        sqlx::query(
            "INSERT INTO agent_sdk_threads (
                thread_id, status, committed_turns, total_input_tokens, total_output_tokens,
                created_at, updated_at
             ) VALUES ('thread-1', 'active', 0, 0, 0, '2026-01-01', '2026-01-01')",
        )
        .execute(&pool)
        .await?;

        for (id, status, last_error, completed_at) in [
            ("completed-task", "completed", None, Some("2026-01-01")),
            (
                "failed-task",
                "failed",
                Some("provider failed"),
                Some("2026-01-01"),
            ),
            ("cancelled-task", "cancelled", None, Some("2026-01-01")),
            ("queued-task", "queued", None, None),
        ] {
            sqlx::query(
                "INSERT INTO agent_sdk_tasks (
                    id, kind, status, root_id, depth, thread_id, attempt, max_attempts,
                    last_error, created_at, updated_at, completed_at
                 ) VALUES (?, 'root_turn', ?, ?, 0, 'thread-1', 0, 1, ?, '2026-01-01',
                    '2026-01-01', ?)",
            )
            .bind(id)
            .bind(status)
            .bind(id)
            .bind(last_error)
            .bind(completed_at)
            .execute(&pool)
            .await?;
        }

        sqlx::raw_sql(migrations[terminal_reason_index].sql)
            .execute(&pool)
            .await?;

        let terminal_rows_without_reason: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM agent_sdk_tasks
             WHERE status IN ('completed', 'failed', 'cancelled')
               AND terminal_reason_json IS NULL",
        )
        .fetch_one(&pool)
        .await?;
        ensure!(
            terminal_rows_without_reason.0 == 0,
            "terminal rows must be backfilled before the migration completes",
        );

        let rows: Vec<(String, Option<String>)> = sqlx::query_as(
            "SELECT status, terminal_reason_json FROM agent_sdk_tasks ORDER BY status",
        )
        .fetch_all(&pool)
        .await?;
        ensure!(
            rows == vec![
                (
                    "cancelled".to_owned(),
                    Some(r#"{"reason":"user_cancel"}"#.to_owned()),
                ),
                (
                    "completed".to_owned(),
                    Some(r#"{"reason":"completed"}"#.to_owned()),
                ),
                (
                    "failed".to_owned(),
                    Some(r#"{"reason":"internal_error"}"#.to_owned()),
                ),
                ("queued".to_owned(), None),
            ],
            "unexpected terminal reason backfill: {rows:?}",
        );

        Ok(())
    }

    #[tokio::test]
    async fn sqlite_migrations_apply_to_in_memory_database() -> Result<()> {
        let store = super::SqliteDurableStore::connect("sqlite::memory:").await?;

        // Verify every durable-core table exists by querying sqlite_master.
        let tables: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'agent_sdk_%' ORDER BY name",
        )
        .fetch_all(store.pool())
        .await?;

        let table_names: BTreeSet<String> = tables.into_iter().map(|r| r.0).collect();
        let expected: BTreeSet<String> = [
            "agent_sdk_tasks",
            "agent_sdk_threads",
            "agent_sdk_message_heads",
            "agent_sdk_message_commits",
            "agent_sdk_turn_attempts",
            "agent_sdk_turn_checkpoints",
            "agent_sdk_committed_events",
            "agent_sdk_outbox",
            "agent_sdk_retention_cursors",
            "agent_sdk_execution_intents",
            "agent_sdk_tool_audit_events",
            "agent_sdk_idempotency",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        ensure!(
            table_names == expected,
            "table mismatch: expected {expected:?}, got {table_names:?}",
        );

        Ok(())
    }
}
