//! Runs the reusable journal/store conformance battery
//! ([`agent_server::journal::conformance::run_journal_store_conformance`])
//! against the durable backends owned by this crate: `SQLite` (in
//! `:memory:`) and `PostgreSQL` (when a `TEST_DATABASE_URL` is set).
//!
//! The in-memory arm is exercised inside `agent-server` itself, next to
//! the battery. These arms close the loop on the plan's
//! one-spec-many-backends requirement (§A2.1 / §6.1): the *same* battery
//! is swapped onto each backend, never forked per backend.
//!
//! The Postgres arm mirrors the established gating in
//! [`crate::postgres::store`]: each test allocates a fresh schema so
//! parallel runs cannot collide, and the whole arm is a no-op when no
//! database URL is configured (so a keyless `cargo nextest run` stays
//! green). With a local Postgres up, run it for real with
//! `TEST_DATABASE_URL` pointed at it.

#[cfg(test)]
mod tests {
    use agent_server::journal::conformance::{ConformanceReport, run_journal_store_conformance};
    use anyhow::Result;
    // `Context` is only used by the Postgres-gated setup helpers below.
    #[cfg(feature = "postgres")]
    use anyhow::Context;

    /// Assert the report shape every backend must satisfy: every
    /// mandatory case ran, and the only recorded skip is the
    /// genuinely-optional atomic event+outbox committer (which the
    /// durable backends actually provide, so for them the skip list is
    /// empty and the present-arm ran instead).
    fn assert_full_battery(report: &ConformanceReport) {
        assert!(
            report
                .passed
                .contains(&"latest_by_insertion_order_not_lexicographic_id".to_owned()),
            "the LangGraph #6821 case must run on every backend",
        );
        assert!(
            report
                .passed
                .contains(&"concurrent_appends_serialize".to_owned()),
            "the concurrent-append case must run on every backend",
        );
        assert!(
            report.passed.len() >= 16,
            "no mandatory conformance case may be silently skipped, got {:?}",
            report.passed,
        );
    }

    // ── SQLite backend ────────────────────────────────────────────────

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_passes_full_journal_conformance() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        let report = run_journal_store_conformance(&store).await?;
        assert_full_battery(&report);
        // The durable SQLite backend provides the atomic event+outbox
        // committer, so the optional arm runs rather than skips.
        assert!(
            report.skipped.is_empty(),
            "SQLite provides every optional method, got skips: {:?}",
            report.skipped,
        );
        assert!(
            report
                .passed
                .contains(&"optional_atomic_outbox_committer_present".to_owned()),
            "SQLite must report the optional outbox committer as present",
        );
        Ok(())
    }

    // ── Postgres backend ──────────────────────────────────────────────

    /// Allocate a throwaway Postgres schema and a migrated store on it,
    /// or `None` when no database URL is configured. Mirrors the gating
    /// used across `crate::postgres::store`'s own tests so the arm is a
    /// keyless no-op in CI without a database.
    #[cfg(feature = "postgres")]
    async fn pg_test_store() -> Result<
        Option<(
            crate::postgres::store::PostgresDurableStore,
            PostgresTestSchema,
        )>,
    > {
        use sqlx::Connection;
        use sqlx::postgres::{PgConnection, PgPoolOptions};
        use uuid::Uuid;

        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };

        let schema = format!("eng_8709_{}", Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&database_url)
            .await
            .context("connect postgres admin for conformance tests")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create test schema {schema}"))?;
        drop(admin);

        let search_path = schema.clone();
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .after_connect(move |conn, _meta| {
                let sql = format!("SET search_path TO {search_path}");
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&database_url)
            .await
            .context("connect postgres conformance test pool")?;
        let store = crate::postgres::store::PostgresDurableStore::from_pool(pool);
        store
            .migrate()
            .await
            .context("migrate postgres conformance test store")?;
        Ok(Some((
            store,
            PostgresTestSchema {
                schema,
                database_url,
            },
        )))
    }

    /// Drops the throwaway schema on `Drop` so a real-database run leaves
    /// nothing behind.
    #[cfg(feature = "postgres")]
    struct PostgresTestSchema {
        schema: String,
        database_url: String,
    }

    #[cfg(feature = "postgres")]
    impl Drop for PostgresTestSchema {
        fn drop(&mut self) {
            let database_url = self.database_url.clone();
            let schema = self.schema.clone();
            let _ = std::thread::spawn(move || {
                use sqlx::Connection;
                use sqlx::postgres::PgConnection;
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    let Ok(mut conn) = PgConnection::connect(&database_url).await else {
                        return;
                    };
                    let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                        "DROP SCHEMA IF EXISTS {schema} CASCADE"
                    )))
                    .execute(&mut conn)
                    .await;
                });
            })
            .join();
        }
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn postgres_passes_full_journal_conformance() -> Result<()> {
        let Some((store, _schema_guard)) = pg_test_store().await? else {
            // No database configured — keyless no-op so the arm does not
            // fail a database-less CI lane.
            return Ok(());
        };
        let report = run_journal_store_conformance(&store).await?;
        assert_full_battery(&report);
        // The durable Postgres backend provides the atomic event+outbox
        // committer, so the optional arm runs rather than skips.
        assert!(
            report.skipped.is_empty(),
            "Postgres provides every optional method, got skips: {:?}",
            report.skipped,
        );
        assert!(
            report
                .passed
                .contains(&"optional_atomic_outbox_committer_present".to_owned()),
            "Postgres must report the optional outbox committer as present",
        );
        Ok(())
    }
}
