//! Session view conformance: every backend's stored view equals the fold of
//! its committed events, through every event write path, on a fresh thread,
//! after an older binary appended without the view, and around a purge.

use std::sync::Arc;
use std::time::Duration as StdDuration;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, ToolResult, ToolTier};
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::AgentTask;
use agent_server::journal::thread::{PurgeReceipt, PurgeScope, PurgeSeed};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::view::{ThreadView, ViewConfig};
use agent_server::journal::view_store::{ThreadViewStore, ViewsDisabled};
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use anyhow::{Context, Result, ensure};
use time::{Duration, OffsetDateTime};

use crate::stores::{StoreOptions, StoreRegistry};
use crate::view_apply::materialize;

const STALL: &str = "__view_stall";
const ALL: usize = 10_000;

fn enabled() -> ViewConfig {
    ViewConfig {
        enabled: true,
        stall_tool_name: Some(STALL.to_owned()),
        visible_document_kinds: Vec::new(),
    }
}

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::days(20_000) + Duration::seconds(secs)
}

fn text_delta(message_id: &str, delta: &str) -> AgentEvent {
    AgentEvent::TextDelta {
        message_id: message_id.to_owned(),
        delta: delta.to_owned(),
        emitter_task_id: Some("root".to_owned()),
    }
}

fn tool_start(id: &str) -> AgentEvent {
    AgentEvent::ToolCallStart {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        input: serde_json::json!({"command": "ls"}),
        tier: ToolTier::Observe,
    }
}

fn tool_end(id: &str) -> AgentEvent {
    AgentEvent::ToolCallEnd {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        result: ToolResult::success("ok"),
    }
}

/// A turn in batches, the way the worker commits it: the start, delta
/// batches, a tool with a confirmation, the closing text, the end.
fn turn(thread: &ThreadId, n: usize) -> Vec<Vec<AgentEvent>> {
    let m = format!("m{n}");
    let t = format!("t{n}");
    vec![
        vec![AgentEvent::start(thread.clone(), n).with_emitter_task_id("root")],
        vec![text_delta(&m, "hel"), text_delta(&m, "lo")],
        vec![
            AgentEvent::Text {
                message_id: m.clone(),
                text: "hello".to_owned(),
                emitter_task_id: Some("root".to_owned()),
            },
            tool_start(&t),
        ],
        vec![AgentEvent::ToolRequiresConfirmation {
            id: t.clone(),
            name: "bash".to_owned(),
            display_name: "Bash".to_owned(),
            input: serde_json::json!({"command": "ls"}),
            description: "list".to_owned(),
        }],
        vec![
            AgentEvent::ToolProgress {
                id: "stall".to_owned(),
                name: STALL.to_owned(),
                display_name: STALL.to_owned(),
                stage: "stalled".to_owned(),
                message: "quiet".to_owned(),
                data: None,
            },
            tool_end(&t),
        ],
        vec![
            AgentEvent::TurnComplete {
                turn: n,
                usage: TokenUsage::default(),
                emitter_task_id: Some("root".to_owned()),
            },
            AgentEvent::Done {
                thread_id: thread.clone(),
                total_turns: n,
                total_usage: TokenUsage::default(),
                duration: StdDuration::from_millis(3),
                estimated_cost_usd: None,
                emitter_task_id: Some("root".to_owned()),
            },
        ],
    ]
}

/// The view a store returns for `thread`, as a whole [`ThreadView`].
async fn stored(views: &dyn ThreadViewStore, thread: &ThreadId) -> Result<ThreadView> {
    let snapshot = views
        .thread_view(thread, ALL)
        .await?
        .with_context(|| format!("no view for {thread}"))?;
    ensure!(!snapshot.has_older, "a whole view fits in {ALL} items");
    Ok(ThreadView {
        status: snapshot.status,
        items: snapshot.items.into_iter().collect(),
        next_index: snapshot.next_index,
    })
}

async fn folded(events: &dyn EventRepository, thread: &ThreadId) -> Result<ThreadView> {
    let committed = events.get_events(thread).await?;
    Ok(materialize(&committed, &enabled()))
}

async fn expect_current(
    views: &dyn ThreadViewStore,
    events: &dyn EventRepository,
    thread: &ThreadId,
    when: &str,
) -> Result<()> {
    let stored = stored(views, thread).await?;
    let folded = folded(events, thread).await?;
    ensure!(
        stored == folded,
        "{when}: the stored view differs from the fold of the events\nstored: {stored:#?}\nfolded: {folded:#?}"
    );
    Ok(())
}

/// Every batch through the split append path and a cancel marker through the
/// cancel transaction land in the view, which equals the fold of the events
/// after each write.
async fn view_follows_every_write(
    events: &dyn EventRepository,
    threads: &dyn ThreadStore,
    tasks: &dyn AgentTaskStore,
    views: &dyn ThreadViewStore,
    name: &str,
) -> Result<()> {
    let thread = ThreadId::from_string(name);
    let _ = threads.get_or_create(&thread, at(0)).await?;
    let mut second = 0;
    for (n, batches) in [turn(&thread, 1), turn(&thread, 2)].into_iter().enumerate() {
        for (i, batch) in batches.into_iter().enumerate() {
            second += 1;
            let _ = events
                .commit_event_batch(&thread, batch, at(second))
                .await?;
            expect_current(views, events, &thread, &format!("turn {n} batch {i}")).await?;
        }
    }

    let root = AgentTask::new_root_turn(thread.clone(), at(100), 3);
    let _ = tasks.submit_root_turn(root.clone()).await?;
    let _ = tasks.cancel_tree(&root.id, at(101)).await?;
    expect_current(views, events, &thread, "after a cancel").await?;
    let view = stored(views, &thread).await?;
    ensure!(
        view.status.last_outcome.is_some(),
        "the cancel marker reached the view: {:?}",
        view.status
    );
    Ok(())
}

/// A view off by default writes nothing and refuses to be read.
async fn disabled_views_refuse(
    events: &dyn EventRepository,
    threads: &dyn ThreadStore,
    views: &dyn ThreadViewStore,
    name: &str,
) -> Result<()> {
    let thread = ThreadId::from_string(name);
    let _ = threads.get_or_create(&thread, at(0)).await?;
    for batch in turn(&thread, 1) {
        let _ = events.commit_event_batch(&thread, batch, at(1)).await?;
    }
    let refused = views.thread_view(&thread, ALL).await;
    ensure!(
        refused
            .as_ref()
            .err()
            .is_some_and(|error| error.root_cause().downcast_ref::<ViewsDisabled>().is_some()),
        "a disabled store refuses to read a view: {refused:?}"
    );
    Ok(())
}

/// A purged thread's view is gone and is not rebuilt.
async fn purge_removes_the_view(
    events: &dyn EventRepository,
    threads: &dyn ThreadStore,
    views: &dyn ThreadViewStore,
    name: &str,
) -> Result<()> {
    let thread = ThreadId::from_string(name);
    let _ = threads.get_or_create(&thread, at(0)).await?;
    for batch in turn(&thread, 1) {
        let _ = events.commit_event_batch(&thread, batch, at(1)).await?;
    }
    ensure!(views.thread_view(&thread, ALL).await?.is_some());
    let _ = threads
        .begin_purge(
            &thread,
            PurgeSeed {
                root_thread_id: thread.clone(),
                scope: PurgeScope::Thread,
                started_at: at(2),
            },
        )
        .await?;
    ensure!(
        views.thread_view(&thread, ALL).await?.is_none(),
        "a deleting thread has no view"
    );
    let _ = threads
        .finish_purge(
            &thread,
            &PurgeReceipt {
                root_thread_id: thread.clone(),
                scope: PurgeScope::Thread,
                purged_thread_ids: vec![thread.clone()],
                cancelled_task_ids: Vec::new(),
                started_at: at(2),
                completed_at: at(3),
            },
        )
        .await?;
    ensure!(
        views.thread_view(&thread, ALL).await?.is_none(),
        "a deleted thread has no view"
    );
    Ok(())
}

/// The stall flag and the tool lifecycle reach the stored view: a sanity
/// check that the store passes its `ViewConfig` to the fold.
async fn the_store_folds_with_its_config(
    events: &dyn EventRepository,
    threads: &dyn ThreadStore,
    views: &dyn ThreadViewStore,
    name: &str,
) -> Result<()> {
    let thread = ThreadId::from_string(name);
    let _ = threads.get_or_create(&thread, at(0)).await?;
    let batches = turn(&thread, 1);
    for batch in batches.into_iter().take(4) {
        let _ = events.commit_event_batch(&thread, batch, at(1)).await?;
    }
    let _ = events
        .commit_event_batch(
            &thread,
            vec![AgentEvent::ToolProgress {
                id: "stall".to_owned(),
                name: STALL.to_owned(),
                display_name: STALL.to_owned(),
                stage: "stalled".to_owned(),
                message: "quiet".to_owned(),
                data: None,
            }],
            at(2),
        )
        .await?;
    let view = stored(views, &thread).await?;
    ensure!(view.status.stalled, "the stall notice set the flag");
    ensure!(view.status.pending_confirmations.len() == 1);
    Ok(())
}

fn registry_definitions() -> Arc<dyn agent_server::worker::registry::AgentDefinitionRegistry> {
    Arc::new(InMemoryAgentDefinitionRegistry::new(AgentDefinition {
        provider: "anthropic".into(),
        model: "claude-sonnet-5-5".into(),
        system_prompt: "test".into(),
        max_tokens: 1024,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }))
}

fn in_memory(view: ViewConfig) -> StoreRegistry {
    StoreRegistry::in_memory_with(registry_definitions(), StoreOptions { view })
}

#[tokio::test]
async fn memory_view_follows_every_write() -> Result<()> {
    let s = in_memory(enabled());
    view_follows_every_write(
        s.event_repo.as_ref(),
        s.thread_store.as_ref(),
        s.task_store.as_ref(),
        s.view_store.as_ref(),
        "view-mem",
    )
    .await
}

#[tokio::test]
async fn memory_disabled_views_refuse() -> Result<()> {
    let s = in_memory(ViewConfig::default());
    disabled_views_refuse(
        s.event_repo.as_ref(),
        s.thread_store.as_ref(),
        s.view_store.as_ref(),
        "view-mem-off",
    )
    .await
}

#[tokio::test]
async fn memory_purge_removes_the_view() -> Result<()> {
    let s = in_memory(enabled());
    purge_removes_the_view(
        s.event_repo.as_ref(),
        s.thread_store.as_ref(),
        s.view_store.as_ref(),
        "view-mem-purge",
    )
    .await
}

#[tokio::test]
async fn memory_store_folds_with_its_config() -> Result<()> {
    let s = in_memory(enabled());
    the_store_folds_with_its_config(
        s.event_repo.as_ref(),
        s.thread_store.as_ref(),
        s.view_store.as_ref(),
        "view-mem-cfg",
    )
    .await
}

#[cfg(feature = "sqlite")]
mod sqlite {
    use super::*;
    use crate::sqlite::SqliteDurableStore;

    async fn store(view: ViewConfig) -> Result<SqliteDurableStore> {
        Ok(SqliteDurableStore::connect("sqlite::memory:")
            .await?
            .with_view_config(view))
    }

    async fn rows(store: &SqliteDurableStore) -> Result<(i64, i64)> {
        let pool = store.pool_for_view_test();
        let views: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_sdk_thread_views")
            .fetch_one(pool)
            .await?;
        let items: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_sdk_thread_view_items")
            .fetch_one(pool)
            .await?;
        Ok((views, items))
    }

    #[tokio::test]
    async fn sqlite_view_follows_every_write() -> Result<()> {
        let s = store(enabled()).await?;
        view_follows_every_write(&s, &s, &s, &s, "view-sq").await?;
        let thread = ThreadId::from_string("view-sq");
        let _ = s
            .commit_event_batch(&thread, vec![tool_start("after-reads")], at(200))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == Some(folded(&s, &thread).await?),
            "the write path alone keeps the rows current"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sqlite_disabled_views_write_no_rows_and_refuse() -> Result<()> {
        let s = store(ViewConfig::default()).await?;
        disabled_views_refuse(&s, &s, &s, "view-sq-off").await?;
        ensure!(
            rows(&s).await? == (0, 0),
            "a disabled store writes no view rows"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sqlite_purge_removes_the_view() -> Result<()> {
        let s = store(enabled()).await?;
        purge_removes_the_view(&s, &s, &s, "view-sq-purge").await?;
        ensure!(
            rows(&s).await? == (0, 0),
            "finish_purge deleted the view rows"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sqlite_store_folds_with_its_config() -> Result<()> {
        let s = store(enabled()).await?;
        the_store_folds_with_its_config(&s, &s, &s, "view-sq-cfg").await
    }

    /// An older binary (views off) appends to the same database: the next
    /// write folds the gap, and a thread it wrote alone is materialized on
    /// its first read. A view at another schema is rebuilt.
    #[tokio::test]
    async fn sqlite_view_catches_up_after_an_older_binary() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let url = format!(
            "sqlite://{}?mode=rwc",
            dir.path().join("views.db").display()
        );
        let new = SqliteDurableStore::connect(&url)
            .await?
            .with_view_config(enabled());
        let old = SqliteDurableStore::connect(&url).await?;

        let thread = ThreadId::from_string("view-sq-skew");
        let _ = new.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        for (store, label) in [(&new, "new"), (&old, "old"), (&old, "old"), (&new, "new")] {
            let batch = batches.next().context("batch")?;
            let _ = store.commit_event_batch(&thread, batch, at(1)).await?;
            if label == "new" {
                let seen = new
                    .stored_view_for_test(&thread)
                    .await?
                    .context("view row")?;
                ensure!(
                    seen == folded(&new, &thread).await?,
                    "the write path folded the events the old binary appended"
                );
            }
        }

        let alone = ThreadId::from_string("view-sq-old-only");
        let _ = old.get_or_create(&alone, at(0)).await?;
        for batch in turn(&alone, 1) {
            let _ = old.commit_event_batch(&alone, batch, at(2)).await?;
        }
        ensure!(rows_for(&new, &alone).await? == 0, "no view until a read");
        expect_current(&new, &new, &alone, "first read of an old thread").await?;

        sqlx::query("UPDATE agent_sdk_thread_views SET view_schema = 0, status_json = '{}'")
            .execute(new.pool_for_view_test())
            .await?;
        expect_current(&new, &new, &thread, "a view at another schema").await?;
        Ok(())
    }

    async fn rows_for(store: &SqliteDurableStore, thread: &ThreadId) -> Result<i64> {
        Ok(
            sqlx::query_scalar("SELECT COUNT(*) FROM agent_sdk_thread_views WHERE thread_id = ?1")
                .bind(thread.0.as_str())
                .fetch_one(store.pool_for_view_test())
                .await?,
        )
    }

    /// A batch of deltas alone leaves the stored view where it was; a read
    /// folds the deltas before it answers, and the next other write folds
    /// them into the rows.
    #[tokio::test]
    async fn sqlite_a_delta_batch_defers_the_view_to_the_next_read_or_write() -> Result<()> {
        let s = store(enabled()).await?;
        let thread = ThreadId::from_string("view-sqlite-defer");
        let _ = s.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        let _ = s
            .commit_event_batch(&thread, batches.next().context("start")?, at(1))
            .await?;
        let before = s.stored_view_for_test(&thread).await?;
        let _ = s
            .commit_event_batch(&thread, batches.next().context("deltas")?, at(2))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == before,
            "the deltas did not touch the view rows"
        );
        expect_current(&s, &s, &thread, "a read after deltas").await?;
        let _ = s
            .commit_event_batch(&thread, batches.next().context("text")?, at(3))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == Some(folded(&s, &thread).await?),
            "the next write folded the deltas into the rows"
        );
        Ok(())
    }

    /// A crash after the events are written and before the view is leaves
    /// neither: the view write is in the events' transaction.
    #[cfg(feature = "failpoints")]
    #[tokio::test]
    async fn sqlite_view_failpoint_writes_neither_events_nor_view() -> Result<()> {
        let _scenario = crate::fail::FailScenario::setup();
        let s = store(enabled()).await?;
        let thread = ThreadId::from_string("view-sq-crash");
        let _ = s.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        for _ in 0..2 {
            let _ = s
                .commit_event_batch(&thread, batches.next().context("batch")?, at(1))
                .await?;
        }
        let before = s.stored_view_for_test(&thread).await?;

        crate::fail::cfg("view.before_apply", "panic")
            .map_err(|error| anyhow::anyhow!("configure view failpoint: {error}"))?;
        let crashed = std::panic::AssertUnwindSafe(s.commit_event_batch(
            &thread,
            batches.next().context("text")?,
            at(2),
        ));
        let result = futures::FutureExt::catch_unwind(crashed).await;
        crate::fail::remove("view.before_apply");
        ensure!(result.is_err(), "the view failpoint fired");

        ensure!(
            s.get_events(&thread).await?.len() == 3,
            "the text batch rolled back"
        );
        ensure!(
            s.stored_view_for_test(&thread).await? == before,
            "the view rolled back with them"
        );
        Ok(())
    }
}

#[cfg(feature = "postgres")]
use postgres::stores as postgres_stores;

#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use crate::postgres::store::PostgresDurableStore;
    use sqlx::Connection;
    use sqlx::postgres::{PgConnection, PgPoolOptions};

    /// A store on a throwaway schema; `None` without `TEST_DATABASE_URL`.
    pub(super) struct Schema {
        name: String,
        url: String,
    }

    impl Drop for Schema {
        fn drop(&mut self) {
            let url = self.url.clone();
            let name = self.name.clone();
            let _joined = std::thread::spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    if let Ok(mut conn) = PgConnection::connect(&url).await {
                        let _dropped = sqlx::query(sqlx::AssertSqlSafe(format!(
                            "DROP SCHEMA IF EXISTS {name} CASCADE"
                        )))
                        .execute(&mut conn)
                        .await;
                    }
                });
            })
            .join();
        }
    }

    pub(super) async fn stores(
        views: &[ViewConfig],
    ) -> Result<Option<(Vec<PostgresDurableStore>, Schema)>> {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            return Ok(None);
        };
        let name = format!("view_pg_{}", uuid::Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&url).await?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {name}")))
            .execute(&mut admin)
            .await?;
        drop(admin);
        let schema = Schema {
            name: name.clone(),
            url: url.clone(),
        };
        let mut out = Vec::new();
        for view in views {
            let search_path = name.clone();
            let pool = PgPoolOptions::new()
                .max_connections(8)
                .after_connect(move |conn, _meta| {
                    let sql = format!("SET search_path TO {search_path}");
                    Box::pin(async move {
                        sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                        Ok(())
                    })
                })
                .connect(&url)
                .await?;
            let store = PostgresDurableStore::from_pool(pool).with_view_config(view.clone());
            store.migrate().await?;
            out.push(store);
        }
        Ok(Some((out, schema)))
    }

    async fn rows(store: &PostgresDurableStore) -> Result<(i64, i64)> {
        let pool = store.pool_for_test();
        let views: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_sdk_thread_views")
            .fetch_one(pool)
            .await?;
        let items: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_sdk_thread_view_items")
            .fetch_one(pool)
            .await?;
        Ok((views, items))
    }

    #[tokio::test]
    async fn pg_view_follows_every_write() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled()]).await? else {
            return Ok(());
        };
        let s = &stores[0];
        view_follows_every_write(s, s, s, s, "view-pg").await?;
        let thread = ThreadId::from_string("view-pg");
        let _ = s
            .commit_event_batch(&thread, vec![tool_start("after-reads")], at(200))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == Some(folded(s, &thread).await?),
            "the write path alone keeps the rows current"
        );
        Ok(())
    }

    /// The satoshi shape: views off. A full turn writes no view row, and a
    /// read refuses.
    #[tokio::test]
    async fn pg_disabled_views_write_no_rows_and_refuse() -> Result<()> {
        let Some((stores, _schema)) = stores(&[ViewConfig::default()]).await? else {
            return Ok(());
        };
        let s = &stores[0];
        disabled_views_refuse(s, s, s, "view-pg-off").await?;
        ensure!(
            rows(s).await? == (0, 0),
            "a disabled store writes no view rows"
        );
        Ok(())
    }

    #[tokio::test]
    async fn pg_purge_removes_the_view() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled()]).await? else {
            return Ok(());
        };
        let s = &stores[0];
        purge_removes_the_view(s, s, s, "view-pg-purge").await?;
        ensure!(
            rows(s).await? == (0, 0),
            "finish_purge deleted the view rows"
        );
        Ok(())
    }

    #[tokio::test]
    async fn pg_store_folds_with_its_config() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled()]).await? else {
            return Ok(());
        };
        let s = &stores[0];
        the_store_folds_with_its_config(s, s, s, "view-pg-cfg").await
    }

    /// The rolling-deploy shape: an older pod (views off) appends beside a
    /// newer one. The newer pod's next write and read fold what it missed.
    #[tokio::test]
    async fn pg_view_catches_up_after_an_older_binary() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled(), ViewConfig::default()]).await? else {
            return Ok(());
        };
        let (new, old) = (&stores[0], &stores[1]);
        let thread = ThreadId::from_string("view-pg-skew");
        let _ = new.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        for store in [new, old, old, new, old] {
            let batch = batches.next().context("batch")?;
            let _ = store.commit_event_batch(&thread, batch, at(1)).await?;
        }
        expect_current(new, new, &thread, "after an older pod appended last").await?;

        let alone = ThreadId::from_string("view-pg-old-only");
        let _ = old.get_or_create(&alone, at(0)).await?;
        for batch in turn(&alone, 1) {
            let _ = old.commit_event_batch(&alone, batch, at(2)).await?;
        }
        expect_current(new, new, &alone, "first read of an old thread").await
    }

    /// Writers and readers of one thread at once: the thread row lock
    /// serializes them, nothing deadlocks, and the view ends equal to the
    /// fold.
    #[tokio::test]
    async fn pg_concurrent_writes_and_reads_keep_the_view_current() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled(), enabled()]).await? else {
            return Ok(());
        };
        let thread = ThreadId::from_string("view-pg-race");
        let _ = stores[0].get_or_create(&thread, at(0)).await?;
        let _ = stores[0]
            .commit_event_batch(
                &thread,
                vec![AgentEvent::start(thread.clone(), 1).with_emitter_task_id("root")],
                at(0),
            )
            .await?;
        let mut work = Vec::new();
        for writer in 0..8u8 {
            let store = stores[usize::from(writer % 2)].clone();
            let thread = thread.clone();
            work.push(tokio::spawn(async move {
                for i in 0..16 {
                    let id = format!("w{writer}-{i}");
                    store
                        .commit_event_batch(&thread, vec![tool_start(&id)], at(1))
                        .await?;
                    store
                        .commit_event_batch(&thread, vec![tool_end(&id)], at(2))
                        .await?;
                    let _ = store.thread_view(&thread, 10).await?;
                }
                anyhow::Ok(())
            }));
        }
        for joined in work {
            joined.await??;
        }
        expect_current(&stores[1], &stores[0], &thread, "after the race").await?;
        let view = stored(&stores[0], &thread).await?;
        ensure!(view.status.open_tools.is_empty());
        ensure!(view.items.len() == 8 * 16);
        Ok(())
    }

    /// A batch of deltas alone leaves the stored view where it was; a read
    /// folds the deltas before it answers, and the next other write folds
    /// them into the rows.
    #[tokio::test]
    async fn pg_a_delta_batch_defers_the_view_to_the_next_read_or_write() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled()]).await? else {
            return Ok(());
        };
        let s = &stores[0];
        let thread = ThreadId::from_string("view-pg-defer");
        let _ = s.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        let _ = s
            .commit_event_batch(&thread, batches.next().context("start")?, at(1))
            .await?;
        let before = s.stored_view_for_test(&thread).await?;
        let _ = s
            .commit_event_batch(&thread, batches.next().context("deltas")?, at(2))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == before,
            "the deltas did not touch the view rows"
        );
        expect_current(s, s, &thread, "a read after deltas").await?;
        let _ = s
            .commit_event_batch(&thread, batches.next().context("text")?, at(3))
            .await?;
        ensure!(
            s.stored_view_for_test(&thread).await? == Some(folded(s, &thread).await?),
            "the next write folded the deltas into the rows"
        );
        Ok(())
    }

    #[cfg(feature = "failpoints")]
    #[tokio::test]
    async fn pg_view_failpoint_writes_neither_events_nor_view() -> Result<()> {
        let Some((stores, _schema)) = stores(&[enabled()]).await? else {
            return Ok(());
        };
        let _scenario = crate::fail::FailScenario::setup();
        let s = &stores[0];
        let thread = ThreadId::from_string("view-pg-crash");
        let _ = s.get_or_create(&thread, at(0)).await?;
        let mut batches = turn(&thread, 1).into_iter();
        for _ in 0..2 {
            let _ = s
                .commit_event_batch(&thread, batches.next().context("batch")?, at(1))
                .await?;
        }
        let (views_before, items_before) = rows(s).await?;
        crate::fail::cfg("view.before_apply", "panic")
            .map_err(|error| anyhow::anyhow!("configure view failpoint: {error}"))?;
        let crashed = std::panic::AssertUnwindSafe(s.commit_event_batch(
            &thread,
            batches.next().context("text")?,
            at(2),
        ));
        let result = futures::FutureExt::catch_unwind(crashed).await;
        crate::fail::remove("view.before_apply");
        ensure!(result.is_err(), "the view failpoint fired");
        ensure!(
            s.get_events(&thread).await?.len() == 3,
            "the text batch rolled back"
        );
        ensure!(
            rows(s).await? == (views_before, items_before),
            "no view write"
        );
        Ok(())
    }

    /// Reconciliation C5: a database that applied 0035 before a later build
    /// adds lower-numbered migrations still migrates. The probe migration
    /// stands for Phase 2/3's 0027–0034.
    #[tokio::test]
    async fn pg_a_lower_migration_after_0035_applies() -> Result<()> {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            return Ok(());
        };
        out_of_order::check_postgres(&url).await
    }
}

mod out_of_order {
    use super::*;
    use std::path::Path;

    /// The highest version below 35 that `source` does not use: a
    /// stand-in for a later build's lower-numbered migration.
    pub fn probe_version(source: &Path) -> Result<i64> {
        let mut used = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(source)? {
            let name = entry?.file_name();
            if let Some(version) = name.to_str().and_then(|n| n.get(..4)?.parse::<i64>().ok()) {
                used.insert(version);
            }
        }
        (1..35)
            .rev()
            .find(|version| !used.contains(version))
            .context("no free version below 35")
    }

    /// Copy `source` into `into`, plus the probe migration when `with_probe`.
    pub fn bundle(source: &Path, into: &Path, with_probe: bool) -> Result<()> {
        std::fs::create_dir_all(into)?;
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            std::fs::copy(entry.path(), into.join(entry.file_name()))?;
        }
        if with_probe {
            let version = probe_version(source)?;
            std::fs::write(
                into.join(format!("{version:04}_out_of_order_probe.sql")),
                "CREATE TABLE agent_sdk_out_of_order_probe (id INTEGER PRIMARY KEY);\n",
            )?;
        }
        Ok(())
    }

    fn migrations(backend: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("migrations")
            .join(backend)
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_a_lower_migration_after_0035_applies() -> Result<()> {
        use sqlx::migrate::Migrator;
        let dir = tempfile::tempdir()?;
        let (first, later) = (dir.path().join("first"), dir.path().join("later"));
        bundle(&migrations("sqlite"), &first, false)?;
        bundle(&migrations("sqlite"), &later, true)?;
        let probe = probe_version(&migrations("sqlite"))?;
        let url = format!("sqlite://{}?mode=rwc", dir.path().join("m.db").display());
        let pool = sqlx::SqlitePool::connect(&url).await?;
        let applied = || sqlx::query_scalar::<_, i64>("SELECT version FROM _sqlx_migrations");
        Migrator::new(first.as_path()).await?.run(&pool).await?;
        let before: Vec<i64> = applied().fetch_all(&pool).await?;
        ensure!(
            before.contains(&35) && !before.contains(&probe),
            "{before:?}"
        );
        Migrator::new(later.as_path()).await?.run(&pool).await?;
        let after: Vec<i64> = applied().fetch_all(&pool).await?;
        ensure!(
            after.contains(&35) && after.contains(&probe),
            "{probe} applied after 0035: {after:?}"
        );
        Ok(())
    }

    #[cfg(feature = "postgres")]
    pub async fn check_postgres(url: &str) -> Result<()> {
        use sqlx::Connection;
        use sqlx::migrate::Migrator;
        use sqlx::postgres::{PgConnection, PgPoolOptions};
        let dir = tempfile::tempdir()?;
        let (first, later) = (dir.path().join("first"), dir.path().join("later"));
        bundle(&migrations("postgres"), &first, false)?;
        bundle(&migrations("postgres"), &later, true)?;
        let probe = probe_version(&migrations("postgres"))?;
        let name = format!("view_mig_{}", uuid::Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(url).await?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {name}")))
            .execute(&mut admin)
            .await?;
        let search_path = name.clone();
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .after_connect(move |conn, _meta| {
                let sql = format!("SET search_path TO {search_path}");
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(url)
            .await?;
        let outcome = async {
            let applied = || sqlx::query_scalar::<_, i64>("SELECT version FROM _sqlx_migrations");
            Migrator::new(first.as_path()).await?.run(&pool).await?;
            let before: Vec<i64> = applied().fetch_all(&pool).await?;
            ensure!(
                before.contains(&35) && !before.contains(&probe),
                "{before:?}"
            );
            Migrator::new(later.as_path()).await?.run(&pool).await?;
            let after: Vec<i64> = applied().fetch_all(&pool).await?;
            ensure!(
                after.contains(&35) && after.contains(&probe),
                "{probe} applied after 0035: {after:?}"
            );
            anyhow::Ok(())
        }
        .await;
        pool.close().await;
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {name} CASCADE")))
            .execute(&mut admin)
            .await?;
        outcome
    }
}

/// Write-path cost of the view, run only with `AGENT_SDK_VIEW_BENCH=1`
/// (`--nocapture` prints the numbers): per-batch latency of the worker's
/// delta flush shape (16 deltas), with views off and on, the first read of
/// a long thread an older binary wrote, and a snapshot read.
mod bench {
    use super::*;
    use std::time::Instant;

    fn wanted() -> bool {
        std::env::var("AGENT_SDK_VIEW_BENCH").is_ok_and(|value| value == "1")
    }

    fn percentile(samples: &mut [f64], percent: usize) -> f64 {
        samples.sort_by(f64::total_cmp);
        let rank = samples.len().saturating_sub(1) * percent / 100;
        samples.get(rank).copied().unwrap_or_default()
    }

    async fn delta_batches(
        events: &dyn EventRepository,
        threads: &dyn ThreadStore,
        name: &str,
        batches: usize,
    ) -> Result<(f64, f64, f64)> {
        let thread = ThreadId::from_string(name);
        let _ = threads.get_or_create(&thread, at(0)).await?;
        let _ = events
            .commit_event_batch(
                &thread,
                vec![AgentEvent::start(thread.clone(), 1).with_emitter_task_id("root")],
                at(0),
            )
            .await?;
        let mut samples = Vec::with_capacity(batches);
        let mut closes = Vec::new();
        for batch in 0..batches {
            let message = format!("m{}", batch / 50);
            let deltas = (0..16).map(|_| text_delta(&message, "token ")).collect();
            let started = Instant::now();
            let _ = events.commit_event_batch(&thread, deltas, at(1)).await?;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            if batch % 50 == 49 {
                let close = vec![AgentEvent::Text {
                    message_id: message.clone(),
                    text: "token ".repeat(800),
                    emitter_task_id: Some("root".to_owned()),
                }];
                let started = Instant::now();
                let _ = events.commit_event_batch(&thread, close, at(1)).await?;
                closes.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        Ok((
            percentile(&mut samples, 50),
            percentile(&mut samples, 95),
            percentile(&mut closes, 50),
        ))
    }

    async fn long_thread(
        writer: &dyn EventRepository,
        threads: &dyn ThreadStore,
        reader: &dyn ThreadViewStore,
        name: &str,
        events: usize,
    ) -> Result<(f64, f64)> {
        let thread = ThreadId::from_string(name);
        let _ = threads.get_or_create(&thread, at(0)).await?;
        let mut written = 0;
        let mut n = 0;
        while written < events {
            n += 1;
            for batch in turn(&thread, n) {
                written += batch.len();
                let _ = writer.commit_event_batch(&thread, batch, at(1)).await?;
            }
        }
        let started = Instant::now();
        let _ = reader.thread_view(&thread, 200).await?;
        let first = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let _ = reader.thread_view(&thread, 200).await?;
        Ok((first, started.elapsed().as_secs_f64() * 1000.0))
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_view_write_cost() -> Result<()> {
        use crate::sqlite::SqliteDurableStore;
        if !wanted() {
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let url = format!(
            "sqlite://{}?mode=rwc",
            dir.path().join("bench.db").display()
        );
        let off = SqliteDurableStore::connect(&url).await?;
        let on = SqliteDurableStore::connect(&url)
            .await?
            .with_view_config(enabled());
        let (off50, off95, offc) = delta_batches(&off, &off, "bench-off", 2000).await?;
        let (on50, on95, onc) = delta_batches(&on, &on, "bench-on", 2000).await?;
        println!(
            "sqlite 16-delta batch ms: off p50 {off50:.3} p95 {off95:.3} | on p50 {on50:.3} p95 {on95:.3}; closing text after 800 deltas p50: off {offc:.3} on {onc:.3}"
        );
        for events in [1_000, 50_000] {
            let (first, again) =
                long_thread(&off, &off, &on, &format!("bench-long-{events}"), events).await?;
            println!(
                "sqlite first read of a {events}-event thread: {first:.1} ms; current read {again:.2} ms"
            );
        }
        Ok(())
    }

    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn pg_view_write_cost() -> Result<()> {
        if !wanted() {
            return Ok(());
        }
        let Some((stores, _schema)) =
            super::postgres_stores(&[ViewConfig::default(), enabled()]).await?
        else {
            return Ok(());
        };
        let (off, on) = (&stores[0], &stores[1]);
        let (off50, off95, offc) = delta_batches(off, off, "bench-off", 2000).await?;
        let (on50, on95, onc) = delta_batches(on, on, "bench-on", 2000).await?;
        println!(
            "postgres 16-delta batch ms: off p50 {off50:.3} p95 {off95:.3} | on p50 {on50:.3} p95 {on95:.3}; closing text after 800 deltas p50: off {offc:.3} on {onc:.3}"
        );
        for events in [1_000, 50_000] {
            let (first, again) =
                long_thread(off, off, on, &format!("bench-long-{events}"), events).await?;
            println!(
                "postgres first read of a {events}-event thread: {first:.1} ms; current read {again:.2} ms"
            );
        }
        Ok(())
    }
}
