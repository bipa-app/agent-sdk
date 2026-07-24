//! `sqlx`-managed migration bundle for the `SQLite` durable contract.
//!
//! Mirrors the `PostgreSQL` migration bundle in
//! [`crate::postgres::migrations`] but loads from the `migrations/sqlite/`
//! directory. The two migration sets evolve independently but are tested
//! for structural parity.

use anyhow::{Context, Result};
use sqlx::SqlitePool;
use sqlx::migrate::Migrator;

/// A single reviewable migration artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqliteMigration {
    /// Monotonic migration version string.
    pub version: &'static str,
    /// Short human-readable purpose.
    pub summary: &'static str,
    /// SQL payload.
    pub sql: &'static str,
}

const DURABLE_CORE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0001_durable_core.sql"
));
const EVENT_JOURNAL_OUTBOX_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0002_event_journal_outbox.sql"
));
const EXECUTION_INTENTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0003_execution_intents.sql"
));
const TOOL_AUDIT_EVENTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0004_tool_audit_events.sql"
));
const OUTBOX_MESSAGE_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0005_outbox_message_kind.sql"
));
const TASK_CALLER_METADATA_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0006_task_caller_metadata.sql"
));
const MESSAGE_HEAD_DRAFTS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0007_message_head_drafts.sql"
));
const TURN_ATTEMPT_OTEL_IDS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0008_turn_attempt_otel_ids.sql"
));
const IDEMPOTENCY_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0009_idempotency.sql"
));
const TASK_OTEL_TRACEPARENT_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0010_task_otel_traceparent.sql"
));
const CHECKPOINT_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0011_checkpoint_kind.sql"
));
const TASK_LAST_ACTIVITY_AT_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0012_task_last_activity_at.sql"
));
const TURN_ATTEMPT_EVIDENCE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0013_turn_attempt_evidence.sql"
));

const TASK_TERMINAL_REASON_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0014_task_terminal_reason.sql"
));

const INPUT_INJECTION_KIND_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0016_input_injection_kind.sql"
));

const THREAD_PURGE_LIFECYCLE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0017_thread_purge_lifecycle.sql"
));

const DURABLE_AWAITING_QUESTION_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0018_durable_awaiting_question.sql"
));

const PURGE_SEED_AT_FENCE_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/sqlite/0019_purge_seed_at_fence.sql"
));

/// `sqlx`-managed migration bundle for the `SQLite` durable contract.
pub static DURABLE_CORE_MIGRATOR: Migrator = sqlx::migrate!("migrations/sqlite");

const MIGRATIONS: [SqliteMigration; 18] = [
    SqliteMigration {
        version: "0001",
        summary: "current durable core tables, constraints, and indexes",
        sql: DURABLE_CORE_SQL,
    },
    SqliteMigration {
        version: "0002",
        summary: "event journal, transactional outbox, and retention cursors",
        sql: EVENT_JOURNAL_OUTBOX_SQL,
    },
    SqliteMigration {
        version: "0003",
        summary: "durable execution intent records for guarded tool execution",
        sql: EXECUTION_INTENTS_SQL,
    },
    SqliteMigration {
        version: "0004",
        summary: "durable tool audit events for child-task execution lifecycle",
        sql: TOOL_AUDIT_EVENTS_SQL,
    },
    SqliteMigration {
        version: "0005",
        summary: "Phase 8.1 outbox message kind discriminator and advisory payload contract",
        sql: OUTBOX_MESSAGE_KIND_SQL,
    },
    SqliteMigration {
        version: "0006",
        summary: "per-turn caller metadata captured on tasks at submission time",
        sql: TASK_CALLER_METADATA_SQL,
    },
    SqliteMigration {
        version: "0007",
        summary: "in-flight draft message slots on message heads for resume-buffer surfacing",
        sql: MESSAGE_HEAD_DRAFTS_SQL,
    },
    SqliteMigration {
        version: "0008",
        summary: "Phase 9 A7 turn attempt OTel trace + span ids for replay-link emission",
        sql: TURN_ATTEMPT_OTEL_IDS_SQL,
    },
    SqliteMigration {
        version: "0009",
        summary: "Phase 10 E durable at-least-once idempotency records",
        sql: IDEMPOTENCY_SQL,
    },
    SqliteMigration {
        version: "0010",
        summary: "per-task OTel traceparent for distributed-trace continuation",
        sql: TASK_OTEL_TRACEPARENT_SQL,
    },
    SqliteMigration {
        version: "0011",
        summary: "explicit checkpoint kind discriminating cancel-salvage from full turns",
        sql: CHECKPOINT_KIND_SQL,
    },
    SqliteMigration {
        version: "0012",
        summary: "durable per-task last_activity_at anchoring the subagent stall budget",
        sql: TASK_LAST_ACTIVITY_AT_SQL,
    },
    SqliteMigration {
        version: "0013",
        summary: "turn-attempt cache creation, route, and resolved effort evidence",
        sql: TURN_ATTEMPT_EVIDENCE_SQL,
    },
    SqliteMigration {
        version: "0014",
        summary: "typed reason required on every terminal task row",
        sql: TASK_TERMINAL_REASON_SQL,
    },
    SqliteMigration {
        version: "0016",
        summary: "input-injection task kind for durable boundary delivery",
        sql: INPUT_INJECTION_KIND_SQL,
    },
    SqliteMigration {
        version: "0017",
        summary: "resurrection-proof thread purge lifecycle and durable receipts",
        sql: THREAD_PURGE_LIFECYCLE_SQL,
    },
    SqliteMigration {
        version: "0018",
        summary: "durable AwaitingQuestion state and AnswerQuestion idempotency",
        sql: DURABLE_AWAITING_QUESTION_SQL,
    },
    SqliteMigration {
        version: "0019",
        summary: "purge seed persists at fence time for stable crash-retry identity",
        sql: PURGE_SEED_AT_FENCE_SQL,
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

/// The reviewable executable migration bundle for the `SQLite` durable core.
#[must_use]
pub const fn durable_core_migrations() -> &'static [SqliteMigration] {
    &MIGRATIONS
}

/// Apply the embedded `sqlx` migration bundle to a live `SQLite` pool.
///
/// This is the runtime entry point the `SQLite` backend calls at startup
/// before instantiating store trait implementations.
///
/// # Errors
///
/// Returns an error if `sqlx` fails to apply or validate the embedded
/// migrations.
pub async fn apply_durable_core_migrations(pool: &SqlitePool) -> Result<()> {
    DURABLE_CORE_MIGRATOR
        .run(pool)
        .await
        .context("apply sqlx sqlite durable-core migrations")
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

    /// Mirror of the Postgres finding-#20 guard: the schema-contract
    /// checks concatenate only the reviewable bundle, so a migration
    /// missing from it has its columns silently skipped by those
    /// checks. Pin the reviewable bundle to the executable migrator so
    /// a migration can never drop out of the contract checks
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

    /// The reviewable bundle's declared versions must match the
    /// executable migrator's versions one-for-one and in order.
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
        assert_eq!(reviewable, executable);
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
