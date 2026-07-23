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

/// `sqlx`-managed migration bundle for the Postgres durable contract.
pub static DURABLE_CORE_MIGRATOR: Migrator = sqlx::migrate!("migrations/postgres");

const MIGRATIONS: [PostgresMigration; 16] = [
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
    use super::{DURABLE_CORE_MIGRATOR, TURN_ATTEMPT_EVIDENCE_SQL, durable_core_migrations};

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

    #[test]
    fn turn_attempt_evidence_migration_is_additive_and_nullable() {
        crate::migration_contract::assert_additive_nullable_migration(
            TURN_ATTEMPT_EVIDENCE_SQL,
            "agent_sdk_turn_attempts",
            5,
        );
    }
}
