//! One-shot, ledgered startup repair of durably corrupted message
//! projections (ENG-9651).
//!
//! The write seams (cancel commit closes started tool calls; the
//! compaction storage seam refuses pair-cutting boundaries; the
//! confirmation lifecycle closes gated calls on decision) prevent new
//! corruption. This sweep is the offline migration for rows written by
//! pre-fix builds: it runs on every store open, in the daemon and in any
//! other process that opens the same file, and records each repaired thread in `agent_sdk_applied_repairs` so a
//! thread is repaired exactly once per corruption class. A thread that
//! re-corrupts after being repaired is repaired exactly once more (the
//! ledger keys on the corruption marker, not a global "done" flag), and
//! the sweep logs loudly — recurrence means a producer regressed.
//!
//! A thread with any non-terminal task is never repaired: its in-flight
//! turn owns the trailing draft, and a parked turn (waiting on children,
//! a confirmation, or a question answer, or ready to resume) legitimately
//! ends in an unanswered `tool_use` whose real result is still coming.
//! Every store open runs the sweep, possibly while another process is
//! running that turn, so the ownership check runs inside the repair's
//! write transaction. Such a thread is deferred (no write, no ledger row)
//! and revisited by a later sweep once its turn has settled.
//!
//! Repair is mechanical and lossless-by-construction: the sweep operates
//! on the provider view (`MessageProjection::provider_view`: the
//! compaction-selected context history plus any in-flight draft), closes
//! dangling `tool_use` blocks with a synthetic "User cancelled"
//! `tool_result`, drops orphan results, and persists through
//! `MessageProjection::append_repair`, which keeps the raw committed
//! transcript untouched and records the corrected view as an append-only
//! compaction entry. The raw transcript of a repaired thread therefore
//! stays invalid forever; only the provider view is ever validated.
//! Signature-bound (thinking) messages are never edited — the repair
//! library only inserts synthetic user messages around them or removes a
//! wholesale broken tail.

use anyhow::{Context, Result};
use sqlx::{Sqlite, SqlitePool, Transaction};
use time::OffsetDateTime;

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::llm;

use super::store::read_message_head;

/// Ledger key for the ENG-9651 tool-sequence repair class. Bumping this
/// re-runs the sweep against every thread (idempotent — a valid thread
/// validates clean and is skipped).
const REPAIR_KEY_TOOL_SEQUENCE_V1: &str = "tool_sequence_v1";

/// Outcome of one startup sweep.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RepairSweepReport {
    /// Threads examined.
    pub scanned: u64,
    /// Threads already repaired under the current key.
    pub ledgered: u64,
    /// Threads this run repaired.
    pub repaired: u64,
    /// Invalid threads left untouched because a non-terminal task still
    /// owns their in-flight turn.
    pub deferred: u64,
    /// Threads whose corruption the mechanical repair could not fix —
    /// left untouched and logged for manual inspection.
    pub unrepairable: u64,
}

/// Run the one-shot repair sweep against an open pool.
///
/// Must be called after the schema migrations (the ledger table must
/// exist). Idempotent and safe to re-run: a valid thread is skipped, and a
/// thread with a live task is deferred, so the sweep is safe even while
/// another process serves the same database.
///
/// # Errors
///
/// Returns an error only on a storage-level failure (the sweep could not
/// list the message heads). Per-thread read and repair failures are
/// logged and counted in [`RepairSweepReport::unrepairable`] rather than
/// aborting the sweep — one poisoned thread must not brick startup for
/// every healthy thread.
pub async fn run_startup_repair_sweep(
    pool: &SqlitePool,
    now: OffsetDateTime,
) -> Result<RepairSweepReport> {
    let mut report = RepairSweepReport::default();

    let threads: Vec<(String, i64)> = sqlx::query_as(
        r"
SELECT h.thread_id,
       EXISTS(
           SELECT 1 FROM agent_sdk_applied_repairs r
           WHERE r.thread_id = h.thread_id AND r.repair_key = ?1
       ) AS already_repaired
FROM agent_sdk_message_heads h
",
    )
    .bind(REPAIR_KEY_TOOL_SEQUENCE_V1)
    .fetch_all(pool)
    .await
    .context("scan message heads for repair sweep")?;

    for (thread_key, already_repaired) in threads {
        report.scanned += 1;
        let already_repaired = already_repaired != 0;
        let thread_id = ThreadId::from_string(thread_key.clone());

        // Validate BEFORE consulting the ledger: a thread that re-corrupts
        // after being repaired must be repaired again, not skipped because
        // a stale ledger row exists.
        let projection = match read_message_head(pool, &thread_id).await {
            Ok(Some(projection)) => projection,
            Ok(None) => continue,
            Err(error) => {
                report.unrepairable += 1;
                tracing::error!(
                    "startup repair sweep could not read thread {thread_key}: {error:#}"
                );
                continue;
            }
        };
        if llm::is_provider_valid_tool_sequence(&projection.provider_view()) {
            continue;
        }

        match repair_one_thread(pool, &thread_id, now).await {
            Ok(ThreadRepair::Repaired) => {
                report.repaired += 1;
                note_recorruption(&mut report, already_repaired, &thread_key);
                tracing::warn!(
                    "startup repair sweep repaired corrupted tool sequence on thread {thread_key}"
                );
            }
            Ok(ThreadRepair::Deferred { live_task }) => {
                report.deferred += 1;
                tracing::info!(
                    "startup repair sweep deferred thread {thread_key}: live task {live_task} \
                     owns its in-flight tool calls"
                );
            }
            Ok(ThreadRepair::AlreadyValid | ThreadRepair::Vanished) => {}
            Ok(ThreadRepair::Unrepairable) => {
                report.unrepairable += 1;
                note_recorruption(&mut report, already_repaired, &thread_key);
                tracing::error!(
                    "startup repair sweep could not mechanically repair thread {thread_key}; \
                     left untouched for manual inspection"
                );
            }
            Err(error) => {
                report.unrepairable += 1;
                tracing::error!("startup repair sweep failed on thread {thread_key}: {error:#}");
            }
        }
    }

    Ok(report)
}

fn note_recorruption(report: &mut RepairSweepReport, already_repaired: bool, thread_key: &str) {
    if already_repaired {
        report.ledgered += 1;
        tracing::warn!(
            "startup repair sweep: thread {thread_key} re-corrupted after a prior repair \
             (a producer regressed or a kill left a dangling call)"
        );
    }
}

/// What [`repair_one_thread`] did with a thread the scan found invalid.
enum ThreadRepair {
    Repaired,
    /// A non-terminal task owns the thread's in-flight turn.
    Deferred {
        live_task: String,
    },
    /// The row changed since the scan and now validates.
    AlreadyValid,
    /// The row was deleted since the scan.
    Vanished,
    /// The mechanical repair still does not validate; the row is untouched.
    Unrepairable,
}

/// Repair one thread's projection and record the ledger entry in the same
/// transaction, unless a live task owns the thread.
///
/// The scan is only a prefilter: ownership and corruption are both
/// re-established from rows read under the write lock, because the store
/// may be open in another process that is running this thread's turn.
async fn repair_one_thread(
    pool: &SqlitePool,
    thread_id: &ThreadId,
    now: OffsetDateTime,
) -> Result<ThreadRepair> {
    let mut tx = pool
        .begin_with("BEGIN IMMEDIATE")
        .await
        .context("begin repair transaction")?;

    if let Some(live_task) = live_task_on_thread(&mut tx, thread_id).await? {
        return Ok(ThreadRepair::Deferred { live_task });
    }

    let Some(projection) = read_message_head(&mut *tx, thread_id)
        .await
        .with_context(|| format!("re-read projection for {thread_id}"))?
    else {
        return Ok(ThreadRepair::Vanished);
    };

    let sequence = projection.provider_view();
    if llm::is_provider_valid_tool_sequence(&sequence) {
        return Ok(ThreadRepair::AlreadyValid);
    }
    let repaired = llm::repair_tool_sequence_in_place(&sequence, llm::USER_CANCELLED_TOOL_RESULT);
    if !llm::is_provider_valid_tool_sequence(&repaired) {
        return Ok(ThreadRepair::Unrepairable);
    }

    // Never the repaired view's tail: a repair can insert mid-sequence, so
    // its tail may be an original message.
    let evidence = llm::orphaned_tool_result_message(&sequence, llm::USER_CANCELLED_TOOL_RESULT)
        .unwrap_or_else(|| llm::Message::user(llm::USER_CANCELLED_TOOL_RESULT));
    let updated = projection
        .append_repair(vec![evidence], repaired, sequence.len(), now)
        .context("append repair entry")?;

    persist_projection_tx(&mut tx, &updated).await?;

    sqlx::query(
        r"
INSERT INTO agent_sdk_applied_repairs (thread_id, repair_key, applied_at)
VALUES (?1, ?2, ?3)
ON CONFLICT (thread_id, repair_key) DO NOTHING
",
    )
    .bind(thread_id.0.as_str())
    .bind(REPAIR_KEY_TOOL_SEQUENCE_V1)
    .bind(now)
    .execute(&mut *tx)
    .await
    .context("record repair ledger entry")?;

    tx.commit().await.context("commit repair transaction")?;
    Ok(ThreadRepair::Repaired)
}

async fn live_task_on_thread(
    tx: &mut Transaction<'_, Sqlite>,
    thread_id: &ThreadId,
) -> Result<Option<String>> {
    let key = thread_id.0.as_str();
    Ok(sqlx::query_scalar!(
        "SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1 AND status NOT IN ('completed', 'failed', 'cancelled') LIMIT 1",
        key,
    )
    .fetch_optional(&mut **tx)
    .await
    .with_context(|| format!("check live tasks on {thread_id}"))?
    .flatten())
}

async fn persist_projection_tx(
    tx: &mut Transaction<'_, Sqlite>,
    projection: &agent_server::journal::MessageProjection,
) -> Result<()> {
    let history_json =
        serde_json::to_string(&projection.messages).context("serialize repaired history")?;
    let draft_json = if projection.draft_messages.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&projection.draft_messages).context("serialize draft")?)
    };
    let compactions_json = if projection.compactions.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&projection.compactions).context("serialize compactions")?)
    };
    let history_generation =
        i64::try_from(projection.history_generation).context("history generation overflow")?;
    let version = i64::try_from(projection.version).context("version overflow")?;
    sqlx::query(
        r"
INSERT INTO agent_sdk_message_heads
    (thread_id, history_json, draft_messages_json, compactions_json, history_generation, version, created_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = excluded.history_json,
    draft_messages_json = excluded.draft_messages_json,
    compactions_json = excluded.compactions_json,
    history_generation = excluded.history_generation,
    version = excluded.version,
    updated_at = excluded.updated_at
",
    )
    .bind(projection.thread_id.0.as_str())
    .bind(history_json)
    .bind(draft_json)
    .bind(compactions_json)
    .bind(history_generation)
    .bind(version)
    .bind(projection.created_at)
    .bind(projection.updated_at)
    .execute(&mut **tx)
    .await
    .context("persist repaired projection")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::store::SqliteDurableStore;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("t0")
    }

    fn seed_projection() -> Vec<llm::Message> {
        // A cancelled turn: committed user prompt plus a dangling assistant
        // tool_use that was never answered (the founder's bricked shape).
        vec![
            llm::Message::user("run the plan"),
            llm::Message::assistant_with_tool_use(
                None,
                "call-1",
                "linear_plan_apply",
                serde_json::json!({"plan_id": "p"}),
            ),
        ]
    }

    async fn seed_corrupted_thread(store: &SqliteDurableStore, key: &str) -> Result<ThreadId> {
        let thread_id = ThreadId::from_string(key);
        agent_server::journal::thread_store::ThreadStore::get_or_create(store, &thread_id, t0())
            .await?;
        agent_server::journal::message_store::MessageProjectionStore::replace_history(
            store,
            &thread_id,
            seed_projection(),
            t0(),
        )
        .await?;
        Ok(thread_id)
    }

    #[tokio::test]
    async fn sweep_repairs_a_cancelled_turn_dangling_tool_use() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = seed_corrupted_thread(&store, "sweep-repairs").await?;

        let before =
            agent_server::journal::message_store::MessageProjectionStore::get(&store, &thread_id)
                .await?
                .context("projection")?;
        assert!(
            !llm::is_provider_valid_tool_sequence(&before.messages),
            "seeded projection must be invalid before the sweep"
        );

        let report = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(report.repaired, 1, "one thread repaired");
        assert_eq!(report.unrepairable, 0);

        let after =
            agent_server::journal::message_store::MessageProjectionStore::get(&store, &thread_id)
                .await?
                .context("projection after")?;
        let view = after.context_history();
        assert!(
            llm::is_provider_valid_tool_sequence(&view),
            "context history is provider-valid after the sweep"
        );
        assert!(
            !llm::has_unbalanced_tool_use(&view),
            "no dangling tool_use remains"
        );
        assert!(
            after
                .messages
                .iter()
                .any(|m| m.content.first_text() == Some("run the plan")),
            "raw transcript preserved"
        );
        assert_eq!(after.compactions.len(), 1, "one append-only repair entry");
        Ok(())
    }

    #[tokio::test]
    async fn an_unreadable_thread_never_stops_the_sweep() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let poisoned = seed_corrupted_thread(&store, "sweep-poisoned").await?;
        sqlx::query("UPDATE agent_sdk_message_heads SET history_json = '[1]' WHERE thread_id = ?1")
            .bind(poisoned.to_string())
            .execute(store.pool())
            .await?;
        let healthy = seed_corrupted_thread(&store, "sweep-healthy").await?;

        let report = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(report.unrepairable, 1, "the unreadable thread is counted");
        assert_eq!(report.repaired, 1, "the next thread is still repaired");

        let after =
            agent_server::journal::message_store::MessageProjectionStore::get(&store, &healthy)
                .await?
                .context("projection after")?;
        assert!(llm::is_provider_valid_tool_sequence(
            &after.context_history()
        ));
        Ok(())
    }

    #[tokio::test]
    async fn sweep_is_idempotent_via_the_ledger() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        seed_corrupted_thread(&store, "sweep-idempotent").await?;

        let first = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(first.repaired, 1);
        let second = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(
            second.repaired, 0,
            "second run repairs nothing — the repaired thread is now valid"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sweep_re_repairs_a_thread_that_re_corrupts() -> Result<()> {
        // A daemon kill mid-turn can leave a NEW dangling tool_use on a thread
        // the sweep already repaired. The validity check runs before the
        // ledger skip, so a re-corrupted thread is repaired again.
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = seed_corrupted_thread(&store, "sweep-recorrupt").await?;

        let first = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(first.repaired, 1);

        // Re-corrupt: drop a fresh dangling tool_use into the draft.
        agent_server::journal::message_store::MessageProjectionStore::set_draft(
            &store,
            &thread_id,
            vec![
                llm::Message::user("again"),
                llm::Message::assistant_with_tool_use(
                    None,
                    "call-recorrupt",
                    "bash",
                    serde_json::json!({"command": "ls"}),
                ),
            ],
            t0(),
        )
        .await?;

        let second = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(
            second.repaired, 1,
            "a thread that re-corrupts after a prior repair must be repaired again"
        );
        assert_eq!(second.ledgered, 1, "the re-corruption is flagged");
        Ok(())
    }

    #[tokio::test]
    async fn sweep_skips_valid_threads() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = ThreadId::from_string("sweep-valid");
        agent_server::journal::thread_store::ThreadStore::get_or_create(&store, &thread_id, t0())
            .await?;
        agent_server::journal::message_store::MessageProjectionStore::replace_history(
            &store,
            &thread_id,
            vec![llm::Message::user("hi"), llm::Message::assistant("hello")],
            t0(),
        )
        .await?;

        let report = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(report.repaired, 0, "valid thread untouched");
        assert_eq!(report.unrepairable, 0);
        Ok(())
    }

    async fn seed_history(
        store: &SqliteDurableStore,
        key: &str,
        history: Vec<llm::Message>,
    ) -> Result<ThreadId> {
        let thread_id = ThreadId::from_string(key);
        agent_server::journal::thread_store::ThreadStore::get_or_create(store, &thread_id, t0())
            .await?;
        agent_server::journal::message_store::MessageProjectionStore::replace_history(
            store,
            &thread_id,
            history,
            t0(),
        )
        .await?;
        Ok(thread_id)
    }

    async fn load(
        store: &SqliteDurableStore,
        thread_id: &ThreadId,
    ) -> Result<agent_server::journal::MessageProjection> {
        agent_server::journal::message_store::MessageProjectionStore::get(store, thread_id)
            .await?
            .context("projection")
    }

    #[tokio::test]
    async fn a_thread_repaired_by_dropping_an_orphan_result_is_never_repaired_again() -> Result<()>
    {
        // The repair strips the orphan result in place, so the view keeps its
        // length while the raw transcript keeps the orphan and gains the
        // cancellation marker: raw 5, view 4.
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = seed_history(
            &store,
            "orphan-result",
            vec![
                llm::Message::user("first prompt"),
                llm::Message::assistant("first reply"),
                llm::Message::tool_result("call-ghost", "late output", false),
                llm::Message::assistant("second reply"),
            ],
        )
        .await?;

        let first = run_startup_repair_sweep(store.pool(), t0()).await?;
        anyhow::ensure!(first.repaired == 1, "first sweep: {first:?}");
        let repaired = load(&store, &thread_id).await?;
        anyhow::ensure!(repaired.messages.len() == 5 && repaired.provider_view().len() == 4);
        anyhow::ensure!(!llm::is_provider_valid_tool_sequence(&repaired.messages));
        let repaired = serde_json::to_value(repaired)?;

        match repair_one_thread(store.pool(), &thread_id, t0()).await {
            Ok(ThreadRepair::AlreadyValid) => {}
            Ok(
                ThreadRepair::Repaired
                | ThreadRepair::Deferred { .. }
                | ThreadRepair::Vanished
                | ThreadRepair::Unrepairable,
            ) => anyhow::bail!("a valid provider view must read as already valid"),
            Err(error) => return Err(error.context("re-repairing a valid provider view")),
        }
        let second = run_startup_repair_sweep(store.pool(), t0()).await?;
        anyhow::ensure!(
            second.repaired == 0 && second.unrepairable == 0,
            "second sweep: {second:?}"
        );
        anyhow::ensure!(projection_json(&store, &thread_id).await? == repaired);
        Ok(())
    }

    #[tokio::test]
    async fn a_mid_history_dangling_call_is_repaired_once_without_duplicating_messages()
    -> Result<()> {
        // Closing a call mid-history inserts a message, so the view and the
        // raw transcript keep equal lengths after the repair.
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let thread_id = seed_history(
            &store,
            "mid-history-call",
            vec![
                llm::Message::user("first prompt"),
                llm::Message::assistant_with_tool_use(
                    None,
                    "call-mid",
                    "bash",
                    serde_json::json!({"command": "ls"}),
                ),
                llm::Message::user("second prompt"),
                llm::Message::assistant("second reply"),
            ],
        )
        .await?;

        let first = run_startup_repair_sweep(store.pool(), t0()).await?;
        anyhow::ensure!(first.repaired == 1, "first sweep: {first:?}");
        let repaired = load(&store, &thread_id).await?;
        let view = repaired.provider_view();
        anyhow::ensure!(llm::is_provider_valid_tool_sequence(&view));
        let replies = |messages: &[llm::Message]| {
            messages
                .iter()
                .filter(|message| message.content.first_text() == Some("second reply"))
                .count()
        };
        anyhow::ensure!(
            replies(&view) == 1 && replies(&repaired.messages) == 1,
            "the repair duplicated a reply: view {}, raw {}",
            replies(&view),
            replies(&repaired.messages)
        );
        anyhow::ensure!(
            repaired.messages.last()
                == Some(&llm::Message::tool_result(
                    "call-mid",
                    llm::USER_CANCELLED_TOOL_RESULT,
                    true,
                )),
            "the raw evidence must be the synthetic result for the closed call"
        );
        let repaired = serde_json::to_value(repaired)?;

        for round in 0..2 {
            let report = run_startup_repair_sweep(store.pool(), t0()).await?;
            anyhow::ensure!(
                report.repaired == 0 && report.unrepairable == 0,
                "sweep {round} after the repair: {report:?}"
            );
        }
        anyhow::ensure!(projection_json(&store, &thread_id).await? == repaired);
        Ok(())
    }

    const LIVE_CALL: &str = "live-call";

    /// Durable shapes in which a root turn legitimately owns a trailing,
    /// still-unanswered `tool_use` in its suspension draft.
    #[derive(Clone, Copy, Debug)]
    enum LiveTurn {
        WaitingOnChildren,
        ChildAwaitingConfirmation,
        AwaitingQuestion,
        AnsweredQuestion,
        ReadyToResume,
        Resuming,
        RunningBeforePark,
    }

    impl LiveTurn {
        const fn tool_name(self) -> &'static str {
            match self {
                Self::AwaitingQuestion | Self::AnsweredQuestion => "ask_user",
                Self::WaitingOnChildren
                | Self::ChildAwaitingConfirmation
                | Self::ReadyToResume
                | Self::Resuming
                | Self::RunningBeforePark => "bash",
            }
        }
    }

    struct SeededTurn {
        thread_id: ThreadId,
        root_id: agent_server::journal::task::AgentTaskId,
    }

    fn worker() -> agent_server::journal::task::WorkerId {
        agent_server::journal::task::WorkerId::from_string("sweep-worker")
    }

    fn lease() -> agent_server::journal::task::LeaseId {
        agent_server::journal::task::LeaseId::from_string("sweep-lease")
    }

    fn lease_expiry() -> OffsetDateTime {
        t0() + time::Duration::minutes(10)
    }

    fn live_call(tool_name: &str) -> agent_sdk_foundation::PendingToolCallInfo {
        agent_sdk_foundation::PendingToolCallInfo {
            id: LIVE_CALL.into(),
            name: tool_name.into(),
            display_name: tool_name.into(),
            tier: agent_sdk_foundation::ToolTier::Observe,
            concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
            input: serde_json::json!({"question": "Which target?", "command": "cargo build"}),
            effective_input: serde_json::json!({"question": "Which target?", "command": "cargo build"}),
            listen_context: None,
        }
    }

    fn live_continuation(
        thread_id: &ThreadId,
        tool_name: &str,
    ) -> agent_sdk_foundation::ContinuationEnvelope {
        agent_sdk_foundation::ContinuationEnvelope::wrap(agent_sdk_foundation::AgentContinuation {
            thread_id: thread_id.clone(),
            turn: 1,
            total_usage: agent_sdk_foundation::TokenUsage::default(),
            turn_usage: agent_sdk_foundation::TokenUsage::default(),
            pending_tool_calls: vec![live_call(tool_name)],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: agent_sdk_foundation::AgentState::new(thread_id.clone()),
            response_id: None,
            stop_reason: Some(llm::StopReason::ToolUse),
            response_content: Vec::new(),
        })
    }

    /// The suspension draft `snapshot_suspension_draft` writes at a tool
    /// boundary: the turn's prompt and its trailing assistant `tool_use`.
    fn suspension_draft(tool_name: &str) -> Vec<llm::Message> {
        vec![
            llm::Message::user("run the build"),
            llm::Message::assistant_with_tool_use(
                None,
                LIVE_CALL,
                tool_name,
                serde_json::json!({"question": "Which target?", "command": "cargo build"}),
            ),
        ]
    }

    fn suspension_payload(
        thread_id: &ThreadId,
        tool_name: &str,
    ) -> agent_server::journal::task::SuspensionPayload {
        agent_server::journal::task::SuspensionPayload {
            continuation: live_continuation(thread_id, tool_name),
            suspended_messages: suspension_draft(tool_name),
            child_join_policy: agent_server::ChildJoinPolicy::default(),
        }
    }

    async fn park_on_tool_child(
        store: &SqliteDurableStore,
        seeded: &SeededTurn,
    ) -> Result<agent_server::journal::task::AgentTask> {
        let (_parent, children) =
            agent_server::journal::store::AgentTaskStore::spawn_tool_children(
                store,
                &seeded.root_id,
                &worker(),
                &lease(),
                vec![agent_server::journal::task::ChildSpawnSpec::new(3)],
                suspension_payload(&seeded.thread_id, "bash"),
                None,
                Vec::new(),
                t0(),
            )
            .await?;
        children.into_iter().next().context("spawned tool child")
    }

    async fn acquire(
        store: &SqliteDurableStore,
        task_id: &agent_server::journal::task::AgentTaskId,
    ) -> Result<agent_server::journal::task::AgentTask> {
        agent_server::journal::store::AgentTaskStore::try_acquire_task(
            store,
            task_id,
            worker(),
            lease(),
            lease_expiry(),
            t0(),
        )
        .await?
        .with_context(|| format!("acquire {task_id}"))
    }

    async fn complete_tool_child(
        store: &SqliteDurableStore,
        child: &agent_server::journal::task::AgentTask,
    ) -> Result<()> {
        acquire(store, &child.id).await?;
        agent_server::journal::store::AgentTaskStore::complete_task_with_result(
            store,
            &child.id,
            &worker(),
            &lease(),
            serde_json::to_value(agent_sdk_foundation::ToolResult::success("build ok"))?,
            t0(),
        )
        .await?;
        Ok(())
    }

    async fn park_on_question(store: &SqliteDurableStore, seeded: &SeededTurn) -> Result<()> {
        agent_server::journal::store::AgentTaskStore::pause_on_question(
            store,
            &seeded.root_id,
            &worker(),
            &lease(),
            agent_server::journal::store::QuestionPause {
                delivered_injection_ids: Vec::new(),
                payload: suspension_payload(&seeded.thread_id, "ask_user"),
                questions: vec![agent_sdk_foundation::QuestionPayload {
                    tool_call_id: LIVE_CALL.into(),
                    question: "Which target?".into(),
                    header: None,
                    options: Vec::new(),
                    multi_select: false,
                }],
                events: Vec::new(),
            },
            t0(),
        )
        .await?;
        Ok(())
    }

    /// Seed a thread whose committed history is clean and whose root turn
    /// is live in `turn`, with the suspension draft the production park
    /// path writes (`snapshot_suspension_draft`).
    async fn seed_live_turn(
        store: &SqliteDurableStore,
        key: &str,
        turn: LiveTurn,
    ) -> Result<SeededTurn> {
        let thread_id = ThreadId::from_string(key);
        agent_server::journal::thread_store::ThreadStore::get_or_create(store, &thread_id, t0())
            .await?;
        agent_server::journal::message_store::MessageProjectionStore::replace_history(
            store,
            &thread_id,
            vec![
                llm::Message::user("earlier turn"),
                llm::Message::assistant("earlier reply"),
            ],
            t0(),
        )
        .await?;
        let root =
            agent_server::journal::task::AgentTask::new_root_turn(thread_id.clone(), t0(), 3);
        let seeded = SeededTurn {
            thread_id,
            root_id: root.id.clone(),
        };
        agent_server::journal::store::AgentTaskStore::submit_root_turn(store, root).await?;
        acquire(store, &seeded.root_id).await?;

        match turn {
            LiveTurn::RunningBeforePark => {}
            LiveTurn::WaitingOnChildren => {
                park_on_tool_child(store, &seeded).await?;
            }
            LiveTurn::ChildAwaitingConfirmation => {
                let child = park_on_tool_child(store, &seeded).await?;
                acquire(store, &child.id).await?;
                agent_server::journal::store::AgentTaskStore::pause_on_confirmation(
                    store,
                    &child.id,
                    &worker(),
                    &lease(),
                    live_continuation(&seeded.thread_id, "bash"),
                    None,
                    t0(),
                )
                .await?;
            }
            LiveTurn::ReadyToResume => {
                let child = park_on_tool_child(store, &seeded).await?;
                complete_tool_child(store, &child).await?;
            }
            LiveTurn::Resuming => {
                let child = park_on_tool_child(store, &seeded).await?;
                complete_tool_child(store, &child).await?;
                acquire(store, &seeded.root_id).await?;
            }
            LiveTurn::AwaitingQuestion => {
                park_on_question(store, &seeded).await?;
            }
            LiveTurn::AnsweredQuestion => {
                park_on_question(store, &seeded).await?;
                agent_server::journal::store::AgentTaskStore::answer_question(
                    store,
                    &seeded.root_id,
                    "sweep-receipt",
                    vec![agent_sdk_foundation::QuestionAnswer {
                        tool_call_id: LIVE_CALL.into(),
                        answer: "staging".into(),
                    }],
                    t0(),
                )
                .await?;
            }
        }

        agent_server::journal::message_store::MessageProjectionStore::set_draft(
            store,
            &seeded.thread_id,
            suspension_draft(turn.tool_name()),
            t0(),
        )
        .await?;
        Ok(seeded)
    }

    async fn ledger_rows(store: &SqliteDurableStore, thread_id: &ThreadId) -> Result<i64> {
        Ok(sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM agent_sdk_applied_repairs WHERE thread_id = ?1",
        )
        .bind(thread_id.0.as_str())
        .fetch_one(store.pool())
        .await?)
    }

    async fn projection_json(
        store: &SqliteDurableStore,
        thread_id: &ThreadId,
    ) -> Result<serde_json::Value> {
        let projection =
            agent_server::journal::message_store::MessageProjectionStore::get(store, thread_id)
                .await?
                .context("projection")?;
        Ok(serde_json::to_value(projection)?)
    }

    async fn assert_sweep_leaves_live_turn_untouched(turn: LiveTurn) -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let seeded = seed_live_turn(&store, &format!("live-{turn:?}"), turn).await?;

        let before = agent_server::journal::message_store::MessageProjectionStore::get(
            &store,
            &seeded.thread_id,
        )
        .await?
        .context("projection before")?;
        let mut provider_view = before.context_history();
        provider_view.extend(before.draft_messages.iter().cloned());
        assert!(
            !llm::is_provider_valid_tool_sequence(&provider_view),
            "{turn:?}: the parked draft must carry the trailing tool_use the sweep flags"
        );
        let before = serde_json::to_value(before)?;

        let report = run_startup_repair_sweep(store.pool(), t0()).await?;

        assert_eq!(
            report.repaired, 0,
            "{turn:?}: a live turn's pending call was repaired"
        );
        assert_eq!(
            report.deferred, 1,
            "{turn:?}: the live turn must be deferred"
        );
        assert_eq!(report.unrepairable, 0, "{turn:?}");
        assert_eq!(
            projection_json(&store, &seeded.thread_id).await?,
            before,
            "{turn:?}: the sweep mutated a live turn's projection"
        );
        assert_eq!(
            ledger_rows(&store, &seeded.thread_id).await?,
            0,
            "{turn:?}: a live turn must not be ledgered as repaired"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sweep_leaves_a_turn_waiting_on_children_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::WaitingOnChildren).await
    }

    #[tokio::test]
    async fn sweep_leaves_a_turn_awaiting_tool_confirmation_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::ChildAwaitingConfirmation).await
    }

    #[tokio::test]
    async fn sweep_leaves_a_turn_awaiting_a_question_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::AwaitingQuestion).await
    }

    #[tokio::test]
    async fn sweep_leaves_an_answered_question_turn_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::AnsweredQuestion).await
    }

    #[tokio::test]
    async fn sweep_leaves_a_turn_ready_to_resume_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::ReadyToResume).await
    }

    #[tokio::test]
    async fn sweep_leaves_a_resuming_turn_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::Resuming).await
    }

    #[tokio::test]
    async fn sweep_leaves_a_turn_running_into_its_park_untouched() -> Result<()> {
        assert_sweep_leaves_live_turn_untouched(LiveTurn::RunningBeforePark).await
    }

    #[tokio::test]
    async fn sweep_repairs_the_same_draft_once_its_turn_is_cancelled() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        let seeded = seed_live_turn(&store, "cancelled-turn", LiveTurn::WaitingOnChildren).await?;
        agent_server::journal::store::AgentTaskStore::cancel_tree(&store, &seeded.root_id, t0())
            .await?;

        let report = run_startup_repair_sweep(store.pool(), t0()).await?;

        assert_eq!(
            report.repaired, 1,
            "a dangling call no live task owns is genuine corruption"
        );
        assert_eq!(report.deferred, 0);
        assert_eq!(ledger_rows(&store, &seeded.thread_id).await?, 1);
        let after = agent_server::journal::message_store::MessageProjectionStore::get(
            &store,
            &seeded.thread_id,
        )
        .await?
        .context("projection after")?;
        assert!(llm::is_provider_valid_tool_sequence(
            &after.context_history()
        ));
        assert!(
            after.draft_messages.is_empty(),
            "the repair folds the draft"
        );
        Ok(())
    }
}
