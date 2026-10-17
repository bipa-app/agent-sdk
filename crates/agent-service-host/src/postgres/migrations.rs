//! Reviewable migration bundle for the current durable core.
//!
//! The migration SQL here is the durable contract for the first
//! Postgres-backed server implementation. It intentionally covers only
//! the current runtime spine:
//!
//! - `agent_sdk_tasks`
//! - `agent_sdk_threads`
//! - `agent_sdk_message_heads`
//! - `agent_sdk_message_commits`
//! - `agent_sdk_turn_attempts`
//! - `agent_sdk_turn_checkpoints`
//!
//! Event persistence and outbox relay tables are called out separately
//! in follow-up notes under `notes/` because this migration bundle is
//! not implementing that behavior yet, and those notes must not be
//! picked up by `sqlx::migrate!`.
//!
//! The bundle is embedded through [`sqlx::migrate!`] so the future
//! backend can use `sqlx`'s built-in migration table and startup
//! application flow.

use anyhow::{Context, Result};
use sqlx::PgPool;
use sqlx::migrate::Migrator;

/// A single reviewable migration artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostgresMigration {
    /// Monotonic migration version string.
    pub version: &'static str,
    /// Short human-readable purpose.
    pub summary: &'static str,
    /// SQL payload.
    pub sql: &'static str,
}

const DURABLE_CORE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0001_durable_core.sql"
));
const EVENT_JOURNAL_OUTBOX_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0002_event_journal_outbox.sql"
));
const EXECUTION_INTENTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0003_execution_intents.sql"
));
const TOOL_AUDIT_EVENTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0004_tool_audit_events.sql"
));
const OUTBOX_MESSAGE_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0005_outbox_message_kind.sql"
));
const TASK_CALLER_METADATA_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0006_task_caller_metadata.sql"
));
const MESSAGE_HEAD_DRAFTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0007_message_head_drafts.sql"
));
const TURN_ATTEMPT_OTEL_IDS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0008_turn_attempt_otel_ids.sql"
));
const IDEMPOTENCY_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0009_idempotency.sql"
));
const TASK_OTEL_TRACEPARENT_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0010_task_otel_traceparent.sql"
));
const CHECKPOINT_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0011_checkpoint_kind.sql"
));
const TASK_LAST_ACTIVITY_AT_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0012_task_last_activity_at.sql"
));
const TURN_ATTEMPT_EVIDENCE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0013_turn_attempt_evidence.sql"
));

const TASK_TERMINAL_REASON_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0014_task_terminal_reason.sql"
));

const INPUT_INJECTION_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0016_input_injection_kind.sql"
));

const THREAD_PURGE_LIFECYCLE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0017_thread_purge_lifecycle.sql"
));

const DURABLE_AWAITING_QUESTION_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0018_durable_awaiting_question.sql"
));

const PURGE_SEED_AT_FENCE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0019_purge_seed_at_fence.sql"
));
const APPEND_ONLY_COMPACTIONS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0020_append_only_compactions.sql"
));
const MESSAGE_HEAD_HISTORY_GENERATION_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0022_message_head_history_generation.sql"
));
const VALIDATE_MESSAGE_HEAD_HISTORY_GENERATION_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0023_validate_message_head_history_generation.sql"
));
const SUBAGENT_CHILD_ROOT_LINK_INDEX_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0024_subagent_child_root_link_index.sql"
));
const DROP_PARKED_SUBAGENT_CHILD_ROOT_INDEX_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0025_drop_parked_subagent_child_root_index.sql"
));
const TASK_AFTER_SIBLINGS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0026_task_after_siblings.sql"
));
const EFFECTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0027_effects.sql"
));
const VALIDATE_EFFECTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0029_validate_effects.sql"
));

/// `sqlx`-managed migration bundle for the Postgres durable contract.
pub static DURABLE_CORE_MIGRATOR: Migrator = sqlx::migrate!("migrations/postgres");

const MIGRATIONS: [PostgresMigration; 26] = [
    PostgresMigration {
        version: "0001",
        summary: "current durable core tables, constraints, and indexes",
        sql: DURABLE_CORE_SQL,
    },
    PostgresMigration {
        version: "0002",
        summary: "event journal, transactional outbox, and retention cursors",
        sql: EVENT_JOURNAL_OUTBOX_SQL,
    },
    PostgresMigration {
        version: "0003",
        summary: "durable execution intent records for guarded tool execution",
        sql: EXECUTION_INTENTS_SQL,
    },
    PostgresMigration {
        version: "0004",
        summary: "durable tool audit events for child-task execution lifecycle",
        sql: TOOL_AUDIT_EVENTS_SQL,
    },
    PostgresMigration {
        version: "0005",
        summary: "Phase 8.1 outbox message kind discriminator and advisory payload contract",
        sql: OUTBOX_MESSAGE_KIND_SQL,
    },
    PostgresMigration {
        version: "0006",
        summary: "per-turn caller metadata captured on tasks at submission time",
        sql: TASK_CALLER_METADATA_SQL,
    },
    PostgresMigration {
        version: "0007",
        summary: "in-flight draft message slots on message heads for resume-buffer surfacing",
        sql: MESSAGE_HEAD_DRAFTS_SQL,
    },
    PostgresMigration {
        version: "0008",
        summary: "Phase 9 A7 turn attempt OTel trace + span ids for replay-link emission",
        sql: TURN_ATTEMPT_OTEL_IDS_SQL,
    },
    PostgresMigration {
        version: "0009",
        summary: "Phase 10 E durable at-least-once idempotency records",
        sql: IDEMPOTENCY_SQL,
    },
    PostgresMigration {
        version: "0010",
        summary: "per-task OTel traceparent for distributed-trace continuation",
        sql: TASK_OTEL_TRACEPARENT_SQL,
    },
    PostgresMigration {
        version: "0011",
        summary: "explicit checkpoint kind discriminating cancel-salvage from full turns",
        sql: CHECKPOINT_KIND_SQL,
    },
    PostgresMigration {
        version: "0012",
        summary: "durable per-task last_activity_at anchoring the subagent stall budget",
        sql: TASK_LAST_ACTIVITY_AT_SQL,
    },
    PostgresMigration {
        version: "0013",
        summary: "turn-attempt cache creation, route, and resolved effort evidence",
        sql: TURN_ATTEMPT_EVIDENCE_SQL,
    },
    PostgresMigration {
        version: "0014",
        summary: "typed reason required on every terminal task row",
        sql: TASK_TERMINAL_REASON_SQL,
    },
    PostgresMigration {
        version: "0016",
        summary: "input-injection task kind for durable boundary delivery",
        sql: INPUT_INJECTION_KIND_SQL,
    },
    PostgresMigration {
        version: "0017",
        summary: "resurrection-proof thread purge lifecycle and durable receipts",
        sql: THREAD_PURGE_LIFECYCLE_SQL,
    },
    PostgresMigration {
        version: "0018",
        summary: "durable AwaitingQuestion state and AnswerQuestion idempotency",
        sql: DURABLE_AWAITING_QUESTION_SQL,
    },
    PostgresMigration {
        version: "0019",
        summary: "purge seed persists at fence time for stable crash-retry identity",
        sql: PURGE_SEED_AT_FENCE_SQL,
    },
    PostgresMigration {
        version: "0020",
        summary: "append-only compaction entries on message heads",
        sql: APPEND_ONLY_COMPACTIONS_SQL,
    },
    PostgresMigration {
        version: "0022",
        summary: "monotonic history rewrite lineage on message heads",
        sql: MESSAGE_HEAD_HISTORY_GENERATION_SQL,
    },
    PostgresMigration {
        version: "0023",
        summary: "validate the history rewrite lineage check outside the column migration",
        sql: VALIDATE_MESSAGE_HEAD_HISTORY_GENERATION_SQL,
    },
    PostgresMigration {
        version: "0024",
        summary: "index every subagent invocation's child-root link, whatever its status",
        sql: SUBAGENT_CHILD_ROOT_LINK_INDEX_SQL,
    },
    PostgresMigration {
        version: "0025",
        summary: "drop the parked-invocation child-root index 0024 supersedes",
        sql: DROP_PARKED_SUBAGENT_CHILD_ROOT_INDEX_SQL,
    },
    PostgresMigration {
        version: "0026",
        summary: "earlier batch siblings a pending tool child waits for",
        sql: TASK_AFTER_SIBLINGS_SQL,
    },
    PostgresMigration {
        version: "0027",
        summary: "execution intents record every effect kind",
        sql: EFFECTS_SQL,
    },
    PostgresMigration {
        version: "0029",
        summary: "validate the effect checks outside the column migration",
        sql: VALIDATE_EFFECTS_SQL,
    },
];

/// Version of the migration this PR introduces: the typed terminal
/// reason column plus its enforcement.
///
/// Named so the bundle test can assert "my migration is in the bundle"
/// without pinning the versions its sibling PRs own.
#[cfg(test)]
pub(crate) const TASK_TERMINAL_REASON_MIGRATION_VERSION: i64 = 14;

/// Version of the migration this PR introduces: the durable
/// input-injection task kind.
#[cfg(test)]
pub(crate) const INPUT_INJECTION_KIND_MIGRATION_VERSION: i64 = 16;

/// Version of the migration this PR introduces: the thread purge
/// lifecycle fence.
#[cfg(test)]
pub(crate) const THREAD_PURGE_LIFECYCLE_MIGRATION_VERSION: i64 = 17;

/// Version of the migration this PR introduces: the durable
/// awaiting-question state.
#[cfg(test)]
pub(crate) const AWAITING_QUESTION_MIGRATION_VERSION: i64 = 18;

/// Version of the migration this PR introduces: the fence-time purge seed.
#[cfg(test)]
pub(crate) const PURGE_SEED_MIGRATION_VERSION: i64 = 19;

/// Version of the migration this PR introduces: the monotonic
/// `history_generation` rewrite-lineage column on message heads.
#[cfg(test)]
pub(crate) const HISTORY_GENERATION_MIGRATION_VERSION: i64 = 22;

/// Version of the migration that validates the `history_generation` check
/// outside the column migration's ACCESS EXCLUSIVE lock.
#[cfg(test)]
pub(crate) const HISTORY_GENERATION_VALIDATION_MIGRATION_VERSION: i64 = 23;

/// Version of the migration that records the earlier batch siblings a
/// pending tool child waits for.
#[cfg(test)]
pub(crate) const TASK_AFTER_SIBLINGS_MIGRATION_VERSION: i64 = 26;

/// Version of the migration that makes the execution intents the record
/// of every effect kind.
#[cfg(test)]
pub(crate) const EFFECTS_MIGRATION_VERSION: i64 = 27;

/// Version of the migration that validates the effect checks outside the
/// column migration's ACCESS EXCLUSIVE lock.
#[cfg(test)]
pub(crate) const EFFECTS_VALIDATION_MIGRATION_VERSION: i64 = 29;

/// The reviewable executable migration bundle for the current durable core.
#[must_use]
pub const fn durable_core_migrations() -> &'static [PostgresMigration] {
    &MIGRATIONS
}

/// Apply the embedded `sqlx` migration bundle to a live Postgres pool.
///
/// This is the runtime entry point the future Postgres backend should
/// call at startup before instantiating repositories.
///
/// # Errors
///
/// Returns an error if `sqlx` fails to apply or validate the embedded
/// migrations.
pub async fn apply_durable_core_migrations(pool: &PgPool) -> Result<()> {
    DURABLE_CORE_MIGRATOR
        .run(pool)
        .await
        .context("apply sqlx postgres durable-core migrations")
}

/// The reviewable event journal and outbox migration SQL.
#[must_use]
pub const fn event_journal_outbox_migration() -> &'static str {
    EVENT_JOURNAL_OUTBOX_SQL
}

/// The reviewable execution intents migration SQL.
#[must_use]
pub const fn execution_intents_migration() -> &'static str {
    EXECUTION_INTENTS_SQL
}

/// The reviewable tool audit events migration SQL.
#[must_use]
pub const fn tool_audit_events_migration() -> &'static str {
    TOOL_AUDIT_EVENTS_SQL
}

/// The reviewable Phase 8.1 outbox message-kind migration SQL.
#[must_use]
pub const fn outbox_message_kind_migration() -> &'static str {
    OUTBOX_MESSAGE_KIND_SQL
}

/// The reviewable per-turn caller-metadata migration SQL.
#[must_use]
pub const fn task_caller_metadata_migration() -> &'static str {
    TASK_CALLER_METADATA_SQL
}

/// The reviewable message-head drafts migration SQL.
#[must_use]
pub const fn message_head_drafts_migration() -> &'static str {
    MESSAGE_HEAD_DRAFTS_SQL
}

/// The reviewable per-task `OTel` traceparent migration SQL.
#[must_use]
pub const fn task_otel_traceparent_migration() -> &'static str {
    TASK_OTEL_TRACEPARENT_SQL
}

/// The reviewable Phase 9 A7 turn-attempt OTel-ids migration SQL.
#[must_use]
pub const fn turn_attempt_otel_ids_migration() -> &'static str {
    TURN_ATTEMPT_OTEL_IDS_SQL
}

/// The reviewable Phase 10 E idempotency migration SQL.
#[must_use]
pub const fn idempotency_migration() -> &'static str {
    IDEMPOTENCY_SQL
}

/// The reviewable checkpoint-kind migration SQL.
#[must_use]
pub const fn checkpoint_kind_migration() -> &'static str {
    CHECKPOINT_KIND_SQL
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result, ensure};
    use sqlx::Connection;
    use sqlx::postgres::PgConnection;

    use super::{
        DURABLE_CORE_MIGRATOR, EFFECTS_SQL, PostgresMigration, TASK_AFTER_SIBLINGS_SQL,
        TURN_ATTEMPT_EVIDENCE_SQL, durable_core_migrations,
    };

    async fn history_generation_check_is_validated(conn: &mut PgConnection) -> Result<bool> {
        let (validated,): (bool,) = sqlx::query_as(
            "SELECT convalidated FROM pg_constraint
             WHERE conname = 'agent_sdk_message_heads_history_generation_check'
               AND conrelid = 'agent_sdk_message_heads'::regclass",
        )
        .fetch_one(conn)
        .await
        .context("read the history_generation check")?;
        Ok(validated)
    }

    async fn assert_history_generation_check_validates_separately(
        conn: &mut PgConnection,
    ) -> Result<()> {
        let (through_column, later): (Vec<&PostgresMigration>, Vec<&PostgresMigration>) =
            durable_core_migrations()
                .iter()
                .partition(|migration| migration.version <= "0022");
        for migration in through_column {
            sqlx::raw_sql(migration.sql)
                .execute(&mut *conn)
                .await
                .with_context(|| format!("apply migration {}", migration.version))?;
        }
        ensure!(
            !history_generation_check_is_validated(conn).await?,
            "0022 must add the check NOT VALID: validating it there scans every \
             message head under the ACCESS EXCLUSIVE lock its ADD COLUMN holds \
             until commit",
        );
        for migration in later {
            sqlx::raw_sql(migration.sql)
                .execute(&mut *conn)
                .await
                .with_context(|| format!("apply migration {}", migration.version))?;
        }
        ensure!(
            history_generation_check_is_validated(conn).await?,
            "a later migration must validate the history_generation check",
        );
        Ok(())
    }

    async fn effect_checks_validated(conn: &mut PgConnection) -> Result<Vec<bool>> {
        let rows: Vec<(bool,)> = sqlx::query_as(
            "SELECT convalidated FROM pg_constraint
             WHERE conname IN ('agent_sdk_execution_intents_kind_check',
                               'agent_sdk_execution_intents_resolution_check',
                               'agent_sdk_execution_intents_attempt_check')
               AND conrelid = 'agent_sdk_execution_intents'::regclass
             ORDER BY conname",
        )
        .fetch_all(conn)
        .await
        .context("read the effect checks")?;
        Ok(rows.into_iter().map(|(validated,)| validated).collect())
    }

    async fn assert_effect_checks_validate_separately(conn: &mut PgConnection) -> Result<()> {
        let (through_columns, later): (Vec<&PostgresMigration>, Vec<&PostgresMigration>) =
            durable_core_migrations()
                .iter()
                .partition(|migration| migration.version <= "0027");
        for migration in through_columns {
            sqlx::raw_sql(migration.sql)
                .execute(&mut *conn)
                .await
                .with_context(|| format!("apply migration {}", migration.version))?;
        }
        ensure!(
            effect_checks_validated(conn).await? == [false; 3],
            "0027 must add its three checks NOT VALID: validating them there scans every \
             intent under the ACCESS EXCLUSIVE lock its ALTERs hold until commit",
        );
        for migration in later {
            sqlx::raw_sql(migration.sql)
                .execute(&mut *conn)
                .await
                .with_context(|| format!("apply migration {}", migration.version))?;
        }
        ensure!(
            effect_checks_validated(conn).await? == [true; 3],
            "a later migration must validate the three effect checks",
        );
        Ok(())
    }

    #[tokio::test]
    async fn effect_checks_are_validated_outside_the_column_migration() -> Result<()> {
        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(());
        };
        let schema = format!("effect_checks_{}", uuid::Uuid::new_v4().simple());
        let mut conn = PgConnection::connect(&database_url)
            .await
            .context("connect postgres for the migration test")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut conn)
            .await
            .with_context(|| format!("create test schema {schema}"))?;
        sqlx::query(sqlx::AssertSqlSafe(format!("SET search_path TO {schema}")))
            .execute(&mut conn)
            .await
            .context("scope the connection to the test schema")?;

        let outcome = assert_effect_checks_validate_separately(&mut conn).await;
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
            .execute(&mut conn)
            .await
            .with_context(|| format!("drop test schema {schema}"))?;
        outcome
    }

    #[tokio::test]
    async fn history_generation_check_is_validated_outside_the_column_migration() -> Result<()> {
        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(());
        };
        let schema = format!("history_generation_{}", uuid::Uuid::new_v4().simple());
        let mut conn = PgConnection::connect(&database_url)
            .await
            .context("connect postgres for the migration test")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut conn)
            .await
            .with_context(|| format!("create test schema {schema}"))?;
        sqlx::query(sqlx::AssertSqlSafe(format!("SET search_path TO {schema}")))
            .execute(&mut conn)
            .await
            .context("scope the connection to the test schema")?;

        let outcome = assert_history_generation_check_validates_separately(&mut conn).await;
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
            .execute(&mut conn)
            .await
            .with_context(|| format!("drop test schema {schema}"))?;
        outcome
    }

    /// Every replica migrates at startup, and the migrator serializes them
    /// on a session advisory lock. Two replicas migrating one fresh schema
    /// at once must both succeed and leave the child-root link index valid:
    /// a migration that waits on other sessions' snapshots while a second
    /// replica waits for the lock deadlocks against it.
    #[tokio::test]
    async fn two_replicas_migrating_at_once_both_succeed() -> Result<()> {
        use sqlx::postgres::PgPoolOptions;

        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(());
        };
        let schema = format!("two_replicas_{}", uuid::Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&database_url)
            .await
            .context("connect postgres for the migration test")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create test schema {schema}"))?;

        let replica = || {
            let search_path = format!("SET search_path TO {schema}");
            PgPoolOptions::new()
                .max_connections(2)
                .after_connect(move |conn, _meta| {
                    let sql = search_path.clone();
                    Box::pin(async move {
                        sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                        Ok(())
                    })
                })
                .connect(&database_url)
        };
        let (first, second) = (replica().await?, replica().await?);
        let (first_run, second_run) = tokio::join!(
            DURABLE_CORE_MIGRATOR.run(&first),
            DURABLE_CORE_MIGRATOR.run(&second),
        );
        let link_index_valid = sqlx::query_scalar!(
            r#"
SELECT indisvalid AS "valid!"
FROM pg_index
WHERE indexrelid = to_regclass('agent_sdk_tasks_subagent_child_root_idx')
"#
        )
        .fetch_optional(&first)
        .await;
        first.close().await;
        second.close().await;
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("drop test schema {schema}"))?;

        first_run.context("the first replica's migration")?;
        second_run.context("the second replica's migration")?;
        ensure!(
            link_index_valid? == Some(true),
            "the child-root link index must exist and be valid",
        );
        Ok(())
    }

    /// Guard against the finding-#20 regression: the reviewable
    /// [`durable_core_migrations`] array silently omitted 0006/0007/0010,
    /// so the schema-contract tests (which concatenate only the reviewable
    /// bundle) skipped the columns and constraints those migrations
    /// introduce. Pin the reviewable bundle to the executable migrator so
    /// a future migration can never again drop out of the contract checks
    /// undetected.
    #[test]
    fn reviewable_bundle_covers_every_executable_migration() {
        let reviewable = durable_core_migrations().len();
        let executable = DURABLE_CORE_MIGRATOR.migrations.len();
        assert_eq!(
            reviewable, executable,
            "reviewable MIGRATIONS array ({reviewable}) must list every executable migration \
             ({executable}); a missing entry hides that migration's schema from the contract tests",
        );
    }

    /// The reviewable bundle's declared versions must match the executable
    /// migrator's versions one-for-one and in order, so the summaries stay
    /// aligned with the SQL the migrator actually applies.
    #[test]
    fn reviewable_versions_match_executable_versions_in_order() {
        let reviewable: Vec<String> = durable_core_migrations()
            .iter()
            .map(|migration| migration.version.to_owned())
            .collect();
        let executable: Vec<String> = DURABLE_CORE_MIGRATOR
            .migrations
            .iter()
            .map(|migration| format!("{:04}", migration.version))
            .collect();
        assert_eq!(
            reviewable, executable,
            "reviewable migration versions must match the executable migrator's, in order",
        );
    }

    /// The child-root link index is built and its predecessor dropped
    /// inside the migration's own transaction, plainly: the migrator's
    /// advisory lock serializes replicas, and a concurrent build would wait
    /// on a replica blocked on that lock (see
    /// `two_replicas_migrating_at_once_both_succeed`).
    #[test]
    fn task_index_migrations_run_inside_their_transaction() -> Result<()> {
        for version in ["0024", "0025"] {
            let executable = DURABLE_CORE_MIGRATOR
                .migrations
                .iter()
                .find(|migration| format!("{:04}", migration.version) == version)
                .with_context(|| format!("migration {version} must be in the bundle"))?;
            ensure!(!executable.no_tx, "{version} must run inside a transaction");
            let reviewable = durable_core_migrations()
                .iter()
                .find(|migration| migration.version == version)
                .with_context(|| format!("migration {version} must be reviewable"))?;
            ensure!(
                !reviewable.sql.contains("CONCURRENTLY"),
                "{version} must not build or drop concurrently",
            );
        }
        Ok(())
    }

    /// 0027's `SET LOCAL lock_timeout` lasts only as long as the
    /// migration's own transaction: outside one it would warn and do
    /// nothing, and its ALTERs could queue every intent query behind a
    /// waiting ACCESS EXCLUSIVE lock.
    #[test]
    fn effects_migrations_run_inside_their_transaction() -> Result<()> {
        for version in ["0027", "0029"] {
            let executable = DURABLE_CORE_MIGRATOR
                .migrations
                .iter()
                .find(|migration| format!("{:04}", migration.version) == version)
                .with_context(|| format!("migration {version} must be in the bundle"))?;
            ensure!(!executable.no_tx, "{version} must run inside a transaction");
        }
        ensure!(
            EFFECTS_SQL.contains("SET LOCAL lock_timeout"),
            "0027 must bound how long it waits for its ACCESS EXCLUSIVE lock",
        );
        ensure!(
            EFFECTS_SQL.matches(") NOT VALID").count() == 3,
            "0027 must add its three checks NOT VALID",
        );
        Ok(())
    }

    #[test]
    fn turn_attempt_evidence_migration_is_additive_and_nullable() {
        crate::migration_contract::assert_additive_nullable_migration(
            TURN_ATTEMPT_EVIDENCE_SQL,
            "agent_sdk_turn_attempts",
            5,
        );
    }

    #[test]
    fn task_after_siblings_migration_is_additive_and_nullable() {
        crate::migration_contract::assert_additive_nullable_migration(
            TASK_AFTER_SIBLINGS_SQL,
            "agent_sdk_tasks",
            1,
        );
    }
}
