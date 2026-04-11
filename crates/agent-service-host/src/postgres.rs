//! Contract-first `PostgreSQL` backend shape for the current durable core.
//!
//! ENG-7984 is intentionally **not** the runtime SQL implementation.
//! Instead this module fixes the durable contract that implementation
//! work will target:
//!
//! - `sqlx`-managed migration SQL for the current Phase 2-4 durable core,
//! - explicit table / constraint / index metadata, and
//! - repository boundaries aligned to the existing journal store traits.
//!
//! The service host still only instantiates the in-memory backend today.
//! This module exists so the future `sqlx`-backed Postgres backend can land against a
//! stable, already-reviewed schema contract instead of reopening the
//! data model during implementation.

pub mod migrations;
pub mod repository;
pub mod schema;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use anyhow::{Context, Result, ensure};

    use super::migrations::{
        DURABLE_CORE_MIGRATOR, durable_core_migrations, future_event_outbox_notes,
    };
    use super::repository::{completed_turn_units_of_work, repository_boundaries};
    use super::schema::durable_core_tables;

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
    fn executable_migration_bundle_excludes_future_notes() -> Result<()> {
        let migrations = &DURABLE_CORE_MIGRATOR.migrations;
        ensure!(
            migrations.len() == 1,
            "expected only executable durable-core migrations, got {:?}",
            migrations
                .iter()
                .map(|migration| migration.version)
                .collect::<Vec<_>>(),
        );
        ensure!(
            migrations[0].version == 1,
            "expected durable core executable migration version 1, got {}",
            migrations[0].version,
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
    fn waiting_state_check_rejects_json_null_kind_and_queued_ready_to_resume() -> Result<()> {
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
            sql_bundle.contains(
                "state_json ->> 'kind' IN ('none', 'ready_to_resume')\n                    AND status NOT IN (\n                        'queued',"
            ),
            "waiting-state check must exclude queued rows from ready_to_resume",
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
    fn future_notes_call_out_event_repository_and_outbox_follow_up() -> Result<()> {
        let notes = future_event_outbox_notes();
        ensure!(
            notes.contains("committed_events"),
            "future notes must mention committed_events",
        );
        ensure!(notes.contains("outbox"), "future notes must mention outbox",);
        ensure!(
            notes.contains("thread-scoped sequence"),
            "future notes must preserve event ordering semantics",
        );
        Ok(())
    }
}
