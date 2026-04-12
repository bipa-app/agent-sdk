//! `sqlx`-backed `SQLite` implementation of the durable-core stores.
//!
//! This backend mirrors the [`crate::postgres::store::PostgresDurableStore`]
//! semantics with `SQLite` dialect adjustments. It uses `BEGIN IMMEDIATE`
//! for all write transactions instead of row-level `FOR UPDATE` locks,
//! since the local backend assumes single-process ownership.
//!
//! All trait implementations will be added in follow-up PRs:
//!   - ENG-8001: `AgentTaskStore`, `ThreadStore`, `MessageProjectionStore`
//!   - ENG-8002: `EventRepository`, `OutboxStore`, `RetentionStore`
//!   - ENG-8003: `TurnAttemptStore`, `CheckpointStore`, `StoreRegistry` wiring

use anyhow::{Context, Result};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

use super::migrations::apply_durable_core_migrations;

/// `sqlx`-backed `SQLite` durable-core store.
///
/// A single instance implements all 10 store traits via the shared
/// connection pool, just like [`crate::postgres::store::PostgresDurableStore`]
/// does for `PostgreSQL`.
#[derive(Clone)]
pub struct SqliteDurableStore {
    pool: SqlitePool,
}

impl SqliteDurableStore {
    /// Construct the store from an existing pool.
    ///
    /// # Safety contract
    ///
    /// The pool **must** have been configured with `after_connect` hooks
    /// that execute at minimum:
    ///
    /// - `PRAGMA foreign_keys = ON`
    /// - `PRAGMA busy_timeout = 5000` (or equivalent)
    ///
    /// Passing a plain unconfigured pool will silently disable foreign-key
    /// enforcement. Prefer [`connect`](Self::connect) for production use.
    #[must_use]
    pub(crate) const fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Connect to a `SQLite` database and apply the durable-core migrations.
    ///
    /// The `database_url` is a `SQLite` connection string. Use
    /// `"sqlite::memory:"` for an ephemeral in-memory database (useful
    /// for tests) or a file path like `"sqlite:///path/to/agent-sdk.db"`.
    ///
    /// For file-backed databases, WAL mode is enabled at connection time
    /// for concurrent read access. In-memory databases do not support WAL
    /// and use their native journal mode instead.
    ///
    /// `PRAGMA foreign_keys = ON` is enforced on every connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot be created, WAL mode cannot be
    /// activated on a file-backed database, or migrations fail.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let is_memory = database_url.contains(":memory:");

        let pool = SqlitePoolOptions::new()
            // In-memory databases are per-connection in SQLite. Use a
            // single connection so migrations and queries share the
            // same schema. File-backed databases can use multiple
            // connections under WAL mode.
            .max_connections(if is_memory { 1 } else { 4 })
            .after_connect(move |conn, _meta| {
                Box::pin(async move {
                    // WAL mode: only attempt on file-backed databases.
                    // In-memory databases do not support WAL and
                    // silently return 'memory' — that is expected.
                    if !is_memory {
                        let row = sqlx::query("PRAGMA journal_mode = WAL")
                            .fetch_one(&mut *conn)
                            .await?;
                        let mode: String = row.get(0);
                        if mode != "wal" {
                            return Err(sqlx::Error::Protocol(format!(
                                "failed to enable WAL mode: got '{mode}'"
                            )));
                        }
                    }
                    sqlx::query("PRAGMA foreign_keys = ON")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("PRAGMA busy_timeout = 5000")
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                })
            })
            .connect(database_url)
            .await
            .context("connect sqlite durable store")?;
        let store = Self::from_pool(pool);
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
    pub const fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
