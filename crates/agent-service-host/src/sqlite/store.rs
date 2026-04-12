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
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;

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
    #[must_use]
    pub const fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Connect to a `SQLite` database and apply the durable-core migrations.
    ///
    /// The `database_url` is a `SQLite` connection string. Use
    /// `"sqlite::memory:"` for an ephemeral in-memory database (useful
    /// for tests) or a file path like `"sqlite:///path/to/agent-sdk.db"`.
    ///
    /// WAL mode is enabled at connection time for concurrent read access.
    /// `PRAGMA foreign_keys = ON` is enforced on every connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot be created or migrations fail.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    sqlx::query("PRAGMA journal_mode = WAL")
                        .execute(&mut *conn)
                        .await?;
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
