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

    use anyhow::{Result, ensure};

    use super::migrations::{
        DURABLE_CORE_MIGRATOR, durable_core_migrations, outbox_message_kind_migration,
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

    #[test]
    fn sqlite_executable_migration_bundle_contains_all_migrations() -> Result<()> {
        let migrations = &DURABLE_CORE_MIGRATOR.migrations;
        ensure!(
            migrations.len() == 9,
            "expected 9 executable migrations, got {:?}",
            migrations.iter().map(|m| m.version).collect::<Vec<_>>(),
        );
        for (idx, expected) in [1_i64, 2, 3, 4, 5, 6, 7, 8, 9].iter().enumerate() {
            ensure!(
                migrations[idx].version == *expected,
                "expected version {expected}, got {}",
                migrations[idx].version,
            );
        }
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
