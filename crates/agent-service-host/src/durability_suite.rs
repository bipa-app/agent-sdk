//! Durability proof suite for the durable SQL backends (Phase 11 · D).
//!
//! Every restart/recovery test in the crate today runs against
//! `sqlite::memory:` or the in-memory store, both of which survive a
//! "restart" for free — the data never actually leaves the process. And
//! a "crash" is simulated by skipping a commit step, so the atomicity
//! claim rests on code structure, not on an injected fault during a real
//! database write. This module closes both gaps:
//!
//! 1. [`tests::sqlite_on_disk_reopen_resumes_midflight_tree`] — writes a
//!    mid-flight turn/tool tree to a real **on-disk** `SQLite` file
//!    (tempfile), drops the store **and** its connection pool to model a
//!    process death, reopens from the same file, runs the lease-expiry
//!    sweep + `recover_thread`, and asserts the thread resumes to a
//!    deterministic terminal state with no orphan `tool_use` and no
//!    duplicate results.
//!
//! 2. [`tests::sqlite_crash_at_commit_failpoint_rolls_back_whole_turn`]
//!    and [`tests::sqlite_crash_replays_idempotently_after_reopen`]
//!    (both `#[cfg(feature = "failpoints")]`) — arm the
//!    `commit.before_event_commit` failpoint, which fires inside the
//!    atomic completed-turn transaction *after* the events and outbox
//!    row are staged but *before* `tx.commit()`. The injected panic
//!    aborts during a real DB write; after reopen the suite asserts no
//!    lost or duplicated work and no torn projection (events exist iff
//!    the turn committed).
//!
//! The Postgres mirrors run only when `TEST_DATABASE_URL` (or
//! `DATABASE_URL`) is set, exactly like the existing Postgres store
//! tests, and create an isolated schema per test so they never collide.
//!
//! # Determinism
//!
//! All timestamps are virtual (the `t_plus` helper); there are no real
//! sleeps and no scripted provider is needed because the journal layer
//! is driven directly. The lease-expiry sweep is run at an explicit
//! virtual time past every staged lease, so its outcome is fixed.

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    use agent_sdk_foundation::events::AgentEvent;
    use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
    use agent_server::journal::commit::{CompletedTurnCommit, commit_completed_turn};
    use agent_server::journal::recovery::RecoveryAction;
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::task::{
        AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
    };
    use agent_server::journal::thread_recover::recover_thread;
    use agent_server::journal::turn_attempt::{
        CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome,
    };
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;

    // Only the failpoint crash tests read durable state back through the
    // `ThreadStore` / `CheckpointStore` / `EventRepository` traits with
    // fully-qualified syntax; the reopen-resume test goes through
    // `recover_thread`, which takes the stores as `&dyn`.
    #[cfg(feature = "failpoints")]
    use agent_server::journal::checkpoint_store::CheckpointStore;
    #[cfg(feature = "failpoints")]
    use agent_server::journal::event_repository::EventRepository;
    #[cfg(feature = "failpoints")]
    use agent_server::journal::thread_store::ThreadStore;
    // The `fail-rs` registry, re-exported by the host crate under the
    // `failpoints` feature (forwards to `agent-server/failpoints`).
    #[cfg(feature = "failpoints")]
    use crate::fail;

    #[cfg(feature = "sqlite")]
    use crate::sqlite::SqliteDurableStore;

    /// One durable backend that implements every store trait the
    /// completed-turn commit path touches. Both `SqliteDurableStore` and
    /// `PostgresDurableStore` satisfy it, so the shared helpers are
    /// generic over `S: DurableStore`. The blanket impl below means no
    /// backend has to opt in by hand.
    use agent_server::journal::checkpoint_store::CheckpointStore as CheckpointStoreTrait;
    use agent_server::journal::event_repository::EventRepository as EventRepositoryTrait;
    use agent_server::journal::message_store::MessageProjectionStore;
    use agent_server::journal::thread_store::ThreadStore as ThreadStoreTrait;

    trait DurableStore:
        AgentTaskStore
        + ThreadStoreTrait
        + MessageProjectionStore
        + TurnAttemptStore
        + CheckpointStoreTrait
        + EventRepositoryTrait
    {
    }

    impl<S> DurableStore for S where
        S: AgentTaskStore
            + ThreadStoreTrait
            + MessageProjectionStore
            + TurnAttemptStore
            + CheckpointStoreTrait
            + EventRepositoryTrait
    {
    }

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn suspension_payload(thread: &ThreadId) -> SuspensionPayload {
        SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread.clone(),
                    turn: 2,
                    total_usage: TokenUsage::default(),
                    turn_usage: TokenUsage::default(),
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread.clone()),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        }
    }

    /// Commit one completed turn through the generic atomic commit path
    /// (the `SqliteDurableStore` exposes the atomic committer, so this
    /// routes through one DB transaction), then close out the root so the
    /// thread's active-root slot is free for the next turn.
    ///
    /// The full lifecycle is exercised so the durable FKs are satisfied:
    /// submit the root → acquire it (`Running`) → open + close the turn
    /// attempt via the atomic commit → complete the root. The durable
    /// side effects (thread aggregate, checkpoint, events) are what the
    /// reopen / crash paths later verify.
    ///
    /// Generic over the concrete durable store so the `SQLite` and
    /// `Postgres` suites share one body — both backends implement every
    /// store trait and route the commit through the same atomic committer
    /// + failpoint.
    async fn commit_one_turn<S>(
        store: &S,
        thread: &ThreadId,
        task_id: &AgentTaskId,
        messages: Vec<llm::Message>,
        events: Vec<AgentEvent>,
        at: OffsetDateTime,
    ) -> Result<()>
    where
        S: DurableStore,
    {
        let (worker, lease) = prepare_running_root(store, thread, task_id, at).await?;
        commit_turn_attempt(store, thread, task_id, 1, messages, events, at).await?;
        // Close the root so the thread's active-root slot frees for the
        // next turn. A worker-driven complete uses the same lease the
        // commit ran under.
        store
            .complete_task(task_id, &worker, &lease, at + Duration::seconds(1))
            .await
            .context("complete root after committed turn")?;
        Ok(())
    }

    /// Submit a root for `task_id` on `thread` and lease it to `Running`,
    /// satisfying the durable FKs the turn-commit path depends on.
    /// Returns the `(worker, lease)` the row was claimed under so the
    /// caller can later complete it.
    async fn prepare_running_root<S>(
        store: &S,
        thread: &ThreadId,
        task_id: &AgentTaskId,
        at: OffsetDateTime,
    ) -> Result<(WorkerId, LeaseId)>
    where
        S: DurableStore,
    {
        let root = AgentTask::new_root_turn(thread.clone(), at, 3);
        // Pin a stable, test-controlled id so the reopen path can
        // reference the row after the crash. A root's `root_id` must
        // equal its `id` (schema invariant), so set both.
        let root = AgentTask {
            id: task_id.clone(),
            root_id: task_id.clone(),
            ..root
        };
        store
            .submit_root_turn(root)
            .await
            .context("submit root for committed turn")?;
        let worker = WorkerId::from_string(format!("w-commit-{task_id}"));
        let lease = LeaseId::from_string(format!("l-commit-{task_id}"));
        store
            .try_acquire_task(
                task_id,
                worker.clone(),
                lease.clone(),
                at + Duration::seconds(600),
                at,
            )
            .await
            .context("acquire root for committed turn")?
            .context("root must acquire for committed turn")?;
        Ok((worker, lease))
    }

    /// Build a durable mid-flight tree on `store`: one *committed* prior
    /// turn (so recovery has a checkpoint to rebuild from), then an
    /// in-flight turn whose root is `Running`, has spawned one tool child
    /// (parent → `WaitingOnChildren`), and that child is itself leased
    /// (`Running`) — the exact shape a host death would freeze on disk.
    /// Returns the child's id so the caller can drive its post-restart
    /// resume. Shared verbatim by the `SQLite` and `Postgres` reopen
    /// tests.
    async fn build_midflight_tree<S>(
        store: &S,
        thread: &ThreadId,
        root_id: &AgentTaskId,
        prior_task: &AgentTaskId,
    ) -> Result<AgentTaskId>
    where
        S: DurableStore,
    {
        commit_one_turn(
            store,
            thread,
            prior_task,
            vec![
                llm::Message::user("first question"),
                llm::Message::assistant("first answer"),
            ],
            vec![AgentEvent::text("prior-evt", "turn 1 done")],
            t_plus(5),
        )
        .await?;

        let (worker, lease) = prepare_running_root(store, thread, root_id, t_plus(10)).await?;
        let (parent, children) = store
            .spawn_tool_children(
                root_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec { max_attempts: 3 }],
                suspension_payload(thread),
                None,
                t_plus(12),
            )
            .await
            .context("spawn tool child for mid-flight tree")?;
        anyhow::ensure!(
            parent.status == TaskStatus::WaitingOnChildren,
            "parent must park on children",
        );
        anyhow::ensure!(children.len() == 1, "exactly one child spawned");
        let child_id = children[0].id.clone();

        // The child is mid-flight: a worker leased it, then the host died.
        // Its lease is still live on disk at crash time.
        store
            .try_acquire_task(
                &child_id,
                WorkerId::from_string("w-child"),
                LeaseId::from_string("l-child"),
                t_plus(40),
                t_plus(13),
            )
            .await
            .context("acquire child for mid-flight tree")?
            .context("child must acquire")?;
        Ok(child_id)
    }

    /// Open and commit one turn attempt for an already-`Running` root.
    /// This is the call that exercises the atomic completed-turn
    /// transaction — and therefore the `commit.before_event_commit`
    /// failpoint. Separated from [`prepare_running_root`] so a crash +
    /// retry can re-run *just* the commit without re-submitting the root.
    async fn commit_turn_attempt<S>(
        store: &S,
        thread: &ThreadId,
        task_id: &AgentTaskId,
        attempt_number: u32,
        messages: Vec<llm::Message>,
        events: Vec<AgentEvent>,
        at: OffsetDateTime,
    ) -> Result<()>
    where
        S: DurableStore,
    {
        let attempt = store
            .open_attempt(OpenAttemptParams {
                task_id: task_id.clone(),
                attempt_number,
                provenance: agent_sdk_foundation::audit::AuditProvenance::new(
                    "anthropic",
                    "claude-sonnet-4-5-20250929",
                ),
                request_blob: serde_json::json!({"messages": []}),
                now: at,
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await
            .context("open attempt for committed turn")?;
        commit_completed_turn(
            CompletedTurnCommit {
                thread_id: thread.clone(),
                task_id: task_id.clone(),
                turn_attempt_id: attempt.id,
                close_attempt_params: CloseAttemptParams {
                    response_blob: serde_json::json!({"id": "msg", "content": []}),
                    response_id: Some("msg".into()),
                    response_model: Some("claude-sonnet-4-5-20250929".into()),
                    stop_reason: Some(llm::StopReason::EndTurn),
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                },
                messages,
                turn_usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    ..Default::default()
                },
                agent_state_snapshot: serde_json::json!({"turn": 1}),
                events,
                outbox_max_attempts: 3,
                now: at,
            },
            store,
            store,
            store,
            store,
            store,
        )
        .await
        .context("commit completed turn")?;
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // 1. Real on-disk SQLite reopen-and-resume
    // ─────────────────────────────────────────────────────────────────

    /// Write a mid-flight turn/tool tree to a real on-disk `SQLite` file,
    /// drop the store + pool (process death), reopen from the same file,
    /// run the lease-expiry sweep + `recover_thread`, and assert a
    /// deterministic terminal state with no orphan / no duplicate.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_on_disk_reopen_resumes_midflight_tree() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db_path = dir.path().join("durability-reopen.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let thread = ThreadId::from_string("reopen-midflight");
        let root_id = AgentTaskId::from_string("task_reopen-root");
        let child_id: AgentTaskId;

        // ── Before the crash: build the mid-flight tree ───────────────
        {
            let store = SqliteDurableStore::connect(&url).await?;
            let prior_task = AgentTaskId::from_string("task_reopen-prior");
            child_id = build_midflight_tree(&store, &thread, &root_id, &prior_task).await?;
            // Crash: drop the store and its pool. The file remains.
            drop(store);
        }

        // ── After the crash: reopen the SAME file ─────────────────────
        let store = SqliteDurableStore::connect(&url).await?;

        // The durable tree survived the reopen.
        let recovered_root = AgentTaskStore::get(&store, &root_id)
            .await?
            .context("root must survive reopen")?;
        assert_eq!(recovered_root.status, TaskStatus::WaitingOnChildren);
        let recovered_child = AgentTaskStore::get(&store, &child_id)
            .await?
            .context("child must survive reopen")?;
        assert_eq!(
            recovered_child.status,
            TaskStatus::Running,
            "the child's Running state + lease survived the crash",
        );

        // Run the lease-expiry sweep at a virtual time past every lease.
        // The orphaned child lease must be reclaimed: the recovery matrix
        // requeues it (budget remains) so a fresh worker can finish it.
        let records = store
            .release_expired_leases(t_plus(100))
            .await
            .context("lease-expiry sweep after reopen")?;
        let child_record = records
            .iter()
            .find(|r| r.id == child_id)
            .context("sweep must touch the orphaned child")?;
        assert_eq!(
            child_record.action,
            RecoveryAction::Requeue,
            "the orphaned child must be requeued, not lost or failed",
        );

        // After the sweep the child is runnable again with NO stale lease
        // (no double-ownership window) — exactly one runnable copy.
        let swept_child = AgentTaskStore::get(&store, &child_id)
            .await?
            .context("child still present after sweep")?;
        assert_eq!(swept_child.status, TaskStatus::Pending);
        assert!(
            swept_child.worker_id.is_none() && swept_child.lease_id.is_none(),
            "requeued child must carry no stale lease",
        );

        // Drive the resume to a deterministic terminal state: a fresh
        // worker acquires the requeued child and completes it, which
        // wakes the parent.
        let cw = WorkerId::from_string("w-child-2");
        let cl = LeaseId::from_string("l-child-2");
        store
            .try_acquire_task(&child_id, cw.clone(), cl.clone(), t_plus(140), t_plus(101))
            .await?
            .context("fresh worker re-acquires requeued child")?;
        let (completed_child, resumed_parent) = store
            .complete_task(&child_id, &cw, &cl, t_plus(102))
            .await?;
        assert_eq!(completed_child.status, TaskStatus::Completed);
        let resumed = resumed_parent.context("completing the last child wakes the parent")?;
        assert_eq!(
            resumed.status,
            TaskStatus::Pending,
            "the parent root resumes runnable once its child is terminal",
        );

        // recover_thread must rebuild the conversation from the committed
        // checkpoint, with no orphaned tool_use and no duplicate results.
        let view = recover_thread(&thread, &store, &store, &store, t_plus(103))
            .await
            .context("recover_thread after reopen + resume")?;
        assert_eq!(
            view.thread.committed_turns, 1,
            "only the prior turn committed; the in-flight turn 2 never did",
        );
        assert_eq!(
            view.next_turn_number, 2,
            "recovery targets turn 2 — the in-flight turn resumes from the checkpoint",
        );
        assert_eq!(
            view.messages.len(),
            2,
            "exactly the two committed messages — no duplicate, no orphan tool_use",
        );
        // The in-flight turn never committed, so no draft duplicates the
        // committed history.
        assert!(
            view.draft_messages.is_empty(),
            "no in-flight draft was committed, so the recovery view has no draft duplication",
        );

        // Determinism: a second recover yields the identical view.
        let view2 = recover_thread(&thread, &store, &store, &store, t_plus(104)).await?;
        assert_eq!(view.messages.len(), view2.messages.len());
        assert_eq!(view.next_turn_number, view2.next_turn_number);

        drop(store);
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // Postgres mirrors (run only when TEST_DATABASE_URL is set)
    // ─────────────────────────────────────────────────────────────────

    /// A per-test Postgres schema. Dropped on `Drop` so a test never
    /// leaks state into the shared database. Mirrors the pattern in
    /// `postgres::store`'s own test suite, but lives here so the
    /// durability suite is self-contained.
    #[cfg(feature = "postgres")]
    struct PgSchema {
        database_url: String,
        schema: String,
    }

    #[cfg(feature = "postgres")]
    impl PgSchema {
        /// Allocate a fresh isolated schema, or `None` when no test
        /// database is configured (so the test skips cleanly).
        async fn create() -> Result<Option<Self>> {
            use sqlx::Connection;
            let Ok(database_url) =
                std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
            else {
                return Ok(None);
            };
            let schema = format!("eng_8711_{}", uuid::Uuid::new_v4().simple());
            let mut admin = sqlx::postgres::PgConnection::connect(&database_url)
                .await
                .context("connect postgres admin for durability suite")?;
            sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
                .execute(&mut admin)
                .await
                .with_context(|| format!("create durability test schema {schema}"))?;
            Ok(Some(Self {
                database_url,
                schema,
            }))
        }

        /// Open a fresh migrated store scoped to this schema. Calling it
        /// twice — with a `drop` of the first store in between — models a
        /// host restart against the same durable database.
        async fn open_store(&self) -> Result<crate::postgres::store::PostgresDurableStore> {
            let search_path = self.schema.clone();
            let pool = sqlx::postgres::PgPoolOptions::new()
                .max_connections(4)
                .after_connect(move |conn, _meta| {
                    let sql = format!("SET search_path TO {search_path}");
                    Box::pin(async move {
                        sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                        Ok(())
                    })
                })
                .connect(&self.database_url)
                .await
                .context("connect schema-scoped postgres pool")?;
            let store = crate::postgres::store::PostgresDurableStore::from_pool(pool);
            store
                .migrate()
                .await
                .context("migrate postgres durability store")?;
            Ok(store)
        }
    }

    #[cfg(feature = "postgres")]
    impl Drop for PgSchema {
        fn drop(&mut self) {
            use sqlx::Connection;
            let url = self.database_url.clone();
            let schema = self.schema.clone();
            // Drop the schema on a throwaway blocking thread so cleanup
            // does not depend on the async runtime still being alive.
            let _ = std::thread::spawn(move || {
                let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                rt.block_on(async move {
                    if let Ok(mut conn) = sqlx::postgres::PgConnection::connect(&url).await {
                        let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                            "DROP SCHEMA IF EXISTS {schema} CASCADE"
                        )))
                        .execute(&mut conn)
                        .await;
                    }
                });
            })
            .join();
        }
    }

    /// Postgres mirror of the on-disk reopen-resume test: write a
    /// mid-flight tree, drop the store + pool (restart), reopen against
    /// the same schema, sweep + recover, assert a deterministic terminal
    /// state with no orphan / no duplicate.
    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn postgres_reopen_resumes_midflight_tree() -> Result<()> {
        let Some(schema) = PgSchema::create().await? else {
            return Ok(());
        };
        let thread = ThreadId::from_string("pg-reopen-midflight");
        let root_id = AgentTaskId::from_string("task_pg-reopen-root");
        let child_id: AgentTaskId;

        // ── Before the restart ────────────────────────────────────────
        {
            let store = schema.open_store().await?;
            let prior_task = AgentTaskId::from_string("task_pg-reopen-prior");
            child_id = build_midflight_tree(&store, &thread, &root_id, &prior_task).await?;
            drop(store);
        }

        // ── After the restart: reopen the same schema ─────────────────
        let store = schema.open_store().await?;
        let recovered_child = AgentTaskStore::get(&store, &child_id)
            .await?
            .context("child must survive restart")?;
        assert_eq!(recovered_child.status, TaskStatus::Running);

        let records = store.release_expired_leases(t_plus(100)).await?;
        let child_record = records
            .iter()
            .find(|r| r.id == child_id)
            .context("sweep must touch the orphaned child")?;
        assert_eq!(child_record.action, RecoveryAction::Requeue);

        let cw = WorkerId::from_string("w-child-2");
        let cl = LeaseId::from_string("l-child-2");
        store
            .try_acquire_task(&child_id, cw.clone(), cl.clone(), t_plus(140), t_plus(101))
            .await?
            .context("fresh worker re-acquires requeued child")?;
        let (_, resumed_parent) = store
            .complete_task(&child_id, &cw, &cl, t_plus(102))
            .await?;
        assert_eq!(
            resumed_parent
                .context("completing the last child wakes the parent")?
                .status,
            TaskStatus::Pending,
        );

        let view = recover_thread(&thread, &store, &store, &store, t_plus(103)).await?;
        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.next_turn_number, 2);
        assert_eq!(view.messages.len(), 2);
        assert!(view.draft_messages.is_empty());

        drop(store);
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // 2. fail-rs crash-between-steps
    // ─────────────────────────────────────────────────────────────────

    /// The injected panic fires inside the atomic completed-turn
    /// transaction (after events + outbox are staged, before
    /// `tx.commit()`). The whole turn must roll back: zero committed
    /// turns, zero events, zero checkpoints — no torn projection.
    ///
    /// MUST run under nextest (one process per test): the `fail-rs`
    /// registry is process-global. The `failpoints` nextest test-group
    /// serializes it.
    #[cfg(all(feature = "sqlite", feature = "failpoints"))]
    #[tokio::test]
    async fn sqlite_crash_at_commit_failpoint_rolls_back_whole_turn() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db_path = dir.path().join("durability-crash.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let thread = ThreadId::from_string("crash-rollback");
        let task_id = AgentTaskId::from_string("task_crash-rollback");

        let _scenario = fail::FailScenario::setup();
        // Arm a panic at the true atomic-commit boundary.
        fail::cfg("commit.before_event_commit", "panic")
            .map_err(|e| anyhow::anyhow!("configure failpoint: {e}"))?;

        let store = SqliteDurableStore::connect(&url).await?;

        // The committed turn carries lifecycle events, so the failpoint
        // path runs after they are staged. The panic aborts the commit.
        let crashed = std::panic::AssertUnwindSafe(commit_one_turn(
            &store,
            &thread,
            &task_id,
            vec![llm::Message::user("doomed turn")],
            vec![AgentEvent::text("doomed-evt", "should never persist")],
            t_plus(5),
        ));
        let result = futures::FutureExt::catch_unwind(crashed).await;
        assert!(
            result.is_err(),
            "the armed failpoint must abort the commit with a panic",
        );

        // Disarm so the post-crash assertions run cleanly.
        fail::remove("commit.before_event_commit");

        // No torn projection: the whole turn rolled back.
        let thread_row = ThreadStore::get(&store, &thread).await?;
        if let Some(t) = thread_row {
            assert_eq!(
                t.committed_turns, 0,
                "the doomed turn must roll back — zero committed turns",
            );
        }
        let events = EventRepository::get_events(&store, &thread).await?;
        assert!(events.is_empty(), "no events when the turn rolled back");
        let checkpoints = CheckpointStore::list_by_thread(&store, &thread).await?;
        assert!(
            checkpoints.is_empty(),
            "no checkpoint when the turn rolled back",
        );

        drop(store);
        Ok(())
    }

    /// After the crash rolls back the turn, a retry under the *same*
    /// inputs (post-reopen, failpoint disarmed) commits cleanly and
    /// exactly once — idempotent replay, no lost or duplicated work.
    ///
    /// MUST run under nextest (process-global `fail-rs` registry).
    #[cfg(all(feature = "sqlite", feature = "failpoints"))]
    #[tokio::test]
    async fn sqlite_crash_replays_idempotently_after_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db_path = dir.path().join("durability-replay.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let thread = ThreadId::from_string("crash-replay");
        let task_id = AgentTaskId::from_string("task_crash-replay");

        let _scenario = fail::FailScenario::setup();

        // ── First attempt: crash mid-commit ──────────────────────────
        // Submit + acquire the root (durable, separate transactions that
        // survive the crash), then arm the failpoint and crash inside the
        // atomic commit transaction.
        {
            let store = SqliteDurableStore::connect(&url).await?;
            prepare_running_root(&store, &thread, &task_id, t_plus(5)).await?;

            fail::cfg("commit.before_event_commit", "panic")
                .map_err(|e| anyhow::anyhow!("configure failpoint: {e}"))?;
            let crashed = std::panic::AssertUnwindSafe(commit_turn_attempt(
                &store,
                &thread,
                &task_id,
                1,
                vec![llm::Message::user("replay me")],
                vec![AgentEvent::text("replay-evt", "turn body")],
                t_plus(6),
            ));
            let result = futures::FutureExt::catch_unwind(crashed).await;
            assert!(result.is_err(), "first attempt must crash at the failpoint");
            drop(store);
        }

        // Disarm and reopen the SAME file — the host has restarted.
        fail::remove("commit.before_event_commit");
        let store = SqliteDurableStore::connect(&url).await?;

        // Nothing leaked from the crashed commit: the root row survived
        // (submitted before the crash) but the turn never committed.
        if let Some(t) = ThreadStore::get(&store, &thread).await? {
            assert_eq!(
                t.committed_turns, 0,
                "crashed attempt left no committed turn"
            );
        }
        assert!(
            EventRepository::get_events(&store, &thread)
                .await?
                .is_empty(),
            "crashed attempt left no events",
        );

        // ── Retry: a fresh attempt on the same root now commits cleanly,
        //    exactly once. The root row is still durable from before the
        //    crash, so the retry only re-runs the commit (attempt #2).
        commit_turn_attempt(
            &store,
            &thread,
            &task_id,
            2,
            vec![llm::Message::user("replay me")],
            vec![AgentEvent::text("replay-evt", "turn body")],
            t_plus(20),
        )
        .await
        .context("retry after reopen must commit cleanly")?;

        let committed = ThreadStore::get(&store, &thread)
            .await?
            .context("thread must exist after the successful retry")?;
        assert_eq!(
            committed.committed_turns, 1,
            "exactly one turn committed across crash + retry — no duplicate",
        );
        let events = EventRepository::get_events(&store, &thread).await?;
        assert_eq!(
            events.len(),
            1,
            "exactly one lifecycle event — the crashed attempt's event never persisted",
        );
        let checkpoints = CheckpointStore::list_by_thread(&store, &thread).await?;
        assert_eq!(checkpoints.len(), 1, "exactly one checkpoint after replay");

        // recover_thread reconstructs the single committed turn — no
        // orphan, no duplicate.
        let view = recover_thread(&thread, &store, &store, &store, t_plus(21)).await?;
        assert_eq!(view.thread.committed_turns, 1);
        assert_eq!(view.messages.len(), 1);

        drop(store);
        Ok(())
    }

    /// Postgres mirror of the crash-then-replay test: crash inside the
    /// atomic commit transaction (failpoint), reopen the same schema, and
    /// assert no lost / duplicated work and no torn projection, then a
    /// clean idempotent retry. Runs only when `TEST_DATABASE_URL` is set.
    ///
    /// MUST run under nextest (process-global `fail-rs` registry).
    #[cfg(all(feature = "postgres", feature = "failpoints"))]
    #[tokio::test]
    async fn postgres_crash_replays_idempotently_after_reopen() -> Result<()> {
        let Some(schema) = PgSchema::create().await? else {
            return Ok(());
        };
        let thread = ThreadId::from_string("pg-crash-replay");
        let task_id = AgentTaskId::from_string("task_pg-crash-replay");

        let _scenario = fail::FailScenario::setup();

        // ── First attempt: crash inside the atomic commit ────────────
        {
            let store = schema.open_store().await?;
            prepare_running_root(&store, &thread, &task_id, t_plus(5)).await?;

            fail::cfg("commit.before_event_commit", "panic")
                .map_err(|e| anyhow::anyhow!("configure failpoint: {e}"))?;
            let crashed = std::panic::AssertUnwindSafe(commit_turn_attempt(
                &store,
                &thread,
                &task_id,
                1,
                vec![llm::Message::user("replay me")],
                vec![AgentEvent::text("replay-evt", "turn body")],
                t_plus(6),
            ));
            let result = futures::FutureExt::catch_unwind(crashed).await;
            assert!(result.is_err(), "first attempt must crash at the failpoint");
            drop(store);
        }

        // Disarm and reopen the same schema — the host restarted.
        fail::remove("commit.before_event_commit");
        let store = schema.open_store().await?;

        // No torn projection from the crashed commit.
        if let Some(t) = ThreadStore::get(&store, &thread).await? {
            assert_eq!(
                t.committed_turns, 0,
                "crashed attempt left no committed turn"
            );
        }
        assert!(
            EventRepository::get_events(&store, &thread)
                .await?
                .is_empty(),
            "crashed attempt left no events",
        );

        // ── Retry commits cleanly, exactly once ───────────────────────
        commit_turn_attempt(
            &store,
            &thread,
            &task_id,
            2,
            vec![llm::Message::user("replay me")],
            vec![AgentEvent::text("replay-evt", "turn body")],
            t_plus(20),
        )
        .await
        .context("retry after reopen must commit cleanly")?;

        let committed = ThreadStore::get(&store, &thread)
            .await?
            .context("thread must exist after the successful retry")?;
        assert_eq!(committed.committed_turns, 1, "exactly one committed turn");
        assert_eq!(
            EventRepository::get_events(&store, &thread).await?.len(),
            1,
            "exactly one lifecycle event survives across crash + retry",
        );
        assert_eq!(
            CheckpointStore::list_by_thread(&store, &thread)
                .await?
                .len(),
            1,
            "exactly one checkpoint after replay",
        );

        drop(store);
        Ok(())
    }
}
