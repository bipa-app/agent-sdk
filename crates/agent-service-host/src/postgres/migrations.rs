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
//! in follow-up notes under `notes/` because ENG-7984 is not
//! implementing that behavior yet, and those notes must not be picked
//! up by `sqlx::migrate!`.
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
const TURN_ATTEMPT_OTEL_IDS_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0008_turn_attempt_otel_ids.sql"
));
const IDEMPOTENCY_SQL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/postgres/0009_idempotency.sql"
));

/// `sqlx`-managed migration bundle for the Postgres durable contract.
pub static DURABLE_CORE_MIGRATOR: Migrator = sqlx::migrate!("migrations/postgres");

const MIGRATIONS: [PostgresMigration; 7] = [
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
        version: "0008",
        summary: "Phase 9 A7 turn attempt OTel trace + span ids for replay-link emission",
        sql: TURN_ATTEMPT_OTEL_IDS_SQL,
    },
    PostgresMigration {
        version: "0009",
        summary: "Phase 10 E durable at-least-once idempotency records",
        sql: IDEMPOTENCY_SQL,
    },
];

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
