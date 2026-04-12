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

/// `sqlx`-managed migration bundle for the `SQLite` durable contract.
pub static DURABLE_CORE_MIGRATOR: Migrator = sqlx::migrate!("migrations/sqlite");

const MIGRATIONS: [SqliteMigration; 2] = [
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
];

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
