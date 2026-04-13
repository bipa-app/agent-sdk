//! `PostgreSQL` backend for the current durable core.
//!
//! This module keeps the contract-first pieces reviewable while also
//! housing the runtime SQL implementation for the currently supported
//! durable surfaces:
//!
//! - `sqlx`-managed migration SQL for the current Phase 2-4 durable core,
//! - explicit table / constraint / index metadata, and
//! - repository boundaries aligned to the existing journal store traits,
//! - the `sqlx` durable-core store used by the host when
//!   `storage.backend=postgres`.
//!
//! The remaining host surfaces without a Postgres implementation
//! (execution intents and tool audit) stay explicit in the store
//! registry as in-memory fallbacks until follow-up work lands.

pub mod migrations;
pub mod repository;
pub mod schema;
pub mod store;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use anyhow::{Context, Result, ensure};

    use super::migrations::{
        DURABLE_CORE_MIGRATOR, durable_core_migrations, event_journal_outbox_migration,
    };
    use super::repository::{
        completed_turn_units_of_work, event_journal_repository_boundaries,
        event_journal_units_of_work, repository_boundaries,
    };
    use super::schema::{durable_core_tables, event_journal_outbox_tables};

    #[test]
    fn durable_core_migrations_cover_every_declared_table_constraint_and_index() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|migration| migration.sql)
            .collect::<Vec<_>>()
            .join("\n");

        for table in durable_core_tables() {
            ensure!(
                sql_bundle.contains(&format!("CREATE TABLE {}", table.name)),
                "missing CREATE TABLE for {}",
                table.name,
            );

            for constraint in table.constraints {
                ensure!(
                    sql_bundle.contains(constraint.name),
                    "missing constraint {} for {}",
                    constraint.name,
                    table.name,
                );
            }

            for index in table.indexes {
                ensure!(
                    sql_bundle.contains(index.name),
                    "missing index {} for {}",
                    index.name,
                    table.name,
                );
            }
        }

        Ok(())
    }

    #[test]
    fn executable_migration_bundle_contains_all_migrations() -> Result<()> {
        let migrations = &DURABLE_CORE_MIGRATOR.migrations;
        ensure!(
            migrations.len() == 3,
            "expected 3 executable migrations (durable core + event journal + execution intents), got {:?}",
            migrations
                .iter()
                .map(|migration| migration.version)
                .collect::<Vec<_>>(),
        );
        ensure!(
            migrations[0].version == 1,
            "expected durable core migration version 1, got {}",
            migrations[0].version,
        );
        ensure!(
            migrations[1].version == 2,
            "expected event journal migration version 2, got {}",
            migrations[1].version,
        );
        ensure!(
            migrations[2].version == 3,
            "expected execution intents migration version 3, got {}",
            migrations[2].version,
        );
        Ok(())
    }

    #[test]
    fn executable_migration_bundle_avoids_duplicate_unique_backing_indexes() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|migration| migration.sql)
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(
            !sql_bundle.contains("agent_sdk_message_commits_by_thread_turn_idx"),
            "message commits should rely on the primary-key backing index",
        );
        ensure!(
            !sql_bundle.contains("agent_sdk_turn_attempts_by_task_attempt_idx"),
            "turn attempts should rely on the unique-constraint backing index",
        );
        Ok(())
    }

    #[test]
    fn waiting_state_check_allows_queued_none_and_rejects_queued_ready_to_resume() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|migration| migration.sql)
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(
            sql_bundle.contains("AND state_json ->> 'kind' IS NOT NULL"),
            "waiting-state check must reject JSON null kind values",
        );
        ensure!(
            sql_bundle.contains("'subagent_invocation'"),
            "waiting-state check must accept the subagent_invocation state kind",
        );
        ensure!(
            sql_bundle.contains(
                "state_json ->> 'kind' = 'none'\n                    AND status IN ('queued', 'pending', 'running')"
            ),
            "waiting-state check must allow queued rows with the default none state",
        );
        ensure!(
            sql_bundle.contains(
                "state_json ->> 'kind' = 'subagent_invocation'\n                    AND (\n                        (\n                            status = 'waiting_on_children'\n                            AND pending_child_count > 0\n                        )\n                        OR (\n                            status IN ('pending', 'running')\n                            AND pending_child_count = 0\n                        )\n                    )"
            ),
            "waiting-state check must allow subagent_invocation rows to become runnable after their child thread drains",
        );
        ensure!(
            sql_bundle.contains(
                "state_json ->> 'kind' = 'ready_to_resume'\n                    AND status IN ('pending', 'running')"
            ),
            "waiting-state check must allow queued rows with the default none state",
        );
        ensure!(
            sql_bundle.contains(
                "state_json ->> 'kind' = 'ready_to_resume'\n                    AND status IN ('pending', 'running')"
            ),
            "waiting-state check must allow ready_to_resume only on pending/running rows",
        );
        Ok(())
    }

    #[test]
    fn depth_kind_check_is_biconditional() -> Result<()> {
        let sql_bundle = durable_core_migrations()
            .iter()
            .map(|migration| migration.sql)
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(
            sql_bundle.contains("CHECK ((depth = 0) = (kind = 'root_turn'))"),
            "depth/kind check must enforce the full depth-zero iff root_turn invariant",
        );
        Ok(())
    }

    #[test]
    fn repository_contracts_cover_the_current_durable_core_traits() -> Result<()> {
        let actual = repository_boundaries()
            .iter()
            .map(|boundary| boundary.store_trait)
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            "agent_server::journal::store::AgentTaskStore",
            "agent_server::journal::thread_store::ThreadStore",
            "agent_server::journal::message_store::MessageProjectionStore",
            "agent_server::journal::turn_attempt_store::TurnAttemptStore",
            "agent_server::journal::checkpoint_store::CheckpointStore",
        ]);
        ensure!(
            actual == expected,
            "repository traits mismatch: expected {expected:?}, got {actual:?}",
        );

        let completed_turn = completed_turn_units_of_work()
            .iter()
            .find(|unit| unit.name == "commit_completed_turn")
            .context("missing commit_completed_turn unit of work")?;
        ensure!(
            completed_turn.tables
                == [
                    "agent_sdk_turn_attempts",
                    "agent_sdk_threads",
                    "agent_sdk_message_heads",
                    "agent_sdk_message_commits",
                    "agent_sdk_turn_checkpoints",
                ],
            "commit_completed_turn tables drifted: {:?}",
            completed_turn.tables,
        );

        let message_boundary = repository_boundaries()
            .iter()
            .find(|boundary| {
                boundary.store_trait
                    == "agent_server::journal::message_store::MessageProjectionStore"
            })
            .context("missing message projection repository boundary")?;
        ensure!(
            message_boundary.tables == ["agent_sdk_message_heads", "agent_sdk_message_commits"],
            "message repository tables drifted: {:?}",
            message_boundary.tables,
        );
        ensure!(
            message_boundary.reads == ["get", "get_history"],
            "message repository reads drifted: {:?}",
            message_boundary.reads,
        );
        ensure!(
            message_boundary.writes == ["get_or_create", "commit_messages", "replace_history"],
            "message repository writes drifted: {:?}",
            message_boundary.writes,
        );

        Ok(())
    }

    #[test]
    fn event_journal_migration_covers_every_declared_table_constraint_and_index() -> Result<()> {
        let sql = event_journal_outbox_migration();

        for table in event_journal_outbox_tables() {
            ensure!(
                sql.contains(&format!("CREATE TABLE {}", table.name)),
                "missing CREATE TABLE for {}",
                table.name,
            );

            for constraint in table.constraints {
                ensure!(
                    sql.contains(constraint.name),
                    "missing constraint {} for {}",
                    constraint.name,
                    table.name,
                );
            }

            for index in table.indexes {
                ensure!(
                    sql.contains(index.name),
                    "missing index {} for {}",
                    index.name,
                    table.name,
                );
            }
        }

        Ok(())
    }

    #[test]
    fn event_journal_repository_contracts_cover_new_traits() -> Result<()> {
        let boundaries = event_journal_repository_boundaries();
        let actual: BTreeSet<_> = boundaries.iter().map(|b| b.store_trait).collect();
        let expected = BTreeSet::from([
            "agent_server::journal::event_repository::EventRepository",
            "agent_server::journal::outbox::OutboxStore",
            "agent_server::journal::retention::RetentionStore",
        ]);
        ensure!(
            actual == expected,
            "event journal repository traits mismatch: expected {expected:?}, got {actual:?}",
        );
        Ok(())
    }

    #[test]
    fn event_journal_units_of_work_cover_cross_table_transactions() -> Result<()> {
        let units = event_journal_units_of_work();

        let commit_with_outbox = units
            .iter()
            .find(|u| u.name == "commit_events_with_outbox")
            .context("missing commit_events_with_outbox unit of work")?;
        ensure!(
            commit_with_outbox.tables == ["agent_sdk_committed_events", "agent_sdk_outbox"],
            "commit_events_with_outbox tables drifted: {:?}",
            commit_with_outbox.tables,
        );

        let retention = units
            .iter()
            .find(|u| u.name == "advance_retention_floor")
            .context("missing advance_retention_floor unit of work")?;
        ensure!(
            retention.tables == ["agent_sdk_retention_cursors", "agent_sdk_committed_events"],
            "advance_retention_floor tables drifted: {:?}",
            retention.tables,
        );

        Ok(())
    }

    #[test]
    fn committed_events_table_enforces_thread_sequence_uniqueness() -> Result<()> {
        let sql = event_journal_outbox_migration();
        ensure!(
            sql.contains("agent_sdk_committed_events_thread_sequence_key"),
            "committed_events must enforce (thread_id, sequence) uniqueness",
        );
        ensure!(
            sql.contains("UNIQUE (thread_id, sequence)"),
            "committed_events UNIQUE constraint must cover (thread_id, sequence)",
        );
        Ok(())
    }

    #[test]
    fn outbox_is_transactionally_tied_to_committed_events() -> Result<()> {
        let sql = event_journal_outbox_migration();
        ensure!(
            sql.contains("agent_sdk_outbox_event_fk"),
            "outbox must have a FK to committed_events",
        );
        ensure!(
            sql.contains("REFERENCES agent_sdk_committed_events(event_id)"),
            "outbox FK must reference committed_events.event_id",
        );
        Ok(())
    }
}
