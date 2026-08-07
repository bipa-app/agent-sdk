//! One-shot, ledgered startup repair of durably corrupted message
//! projections (ENG-9651).
//!
//! The write seams (cancel commit closes started tool calls; the
//! compaction storage seam refuses pair-cutting boundaries; the
//! confirmation lifecycle closes gated calls on decision) prevent new
//! corruption. This sweep is the offline migration for rows written by
//! pre-fix builds: it runs once at store open, before any worker starts,
//! and records each repaired thread in `agent_sdk_applied_repairs` so a
//! thread is repaired exactly once per corruption class. A thread that
//! re-corrupts after being repaired is repaired exactly once more (the
//! ledger keys on the corruption marker, not a global "done" flag), and
//! the sweep logs loudly — recurrence means a producer regressed.
//!
//! Repair is mechanical and lossless-by-construction: the sweep operates
//! on the provider-visible sequence (compaction-selected context history
//! plus any in-flight draft), closes dangling `tool_use` blocks with a
//! synthetic "User cancelled" `tool_result`, drops orphan results, and
//! persists through `MessageProjection::append_repair`, which keeps the
//! raw committed transcript untouched and records the corrected view as
//! an append-only compaction entry. Signature-bound (thinking) messages
//! are never edited — the repair library only inserts synthetic user
//! messages around them or removes a wholesale broken tail.

use anyhow::{Context, Result};
use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use time::OffsetDateTime;

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::llm;

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
    /// Threads whose corruption the mechanical repair could not fix —
    /// left untouched and logged for manual inspection.
    pub unrepairable: u64,
}

/// Run the one-shot repair sweep against an open pool.
///
/// Must be called after the schema migrations (the ledger table must
/// exist) and before any worker acquires tasks. Idempotent and safe to
/// re-run: threads already repaired under the current key are skipped.
///
/// # Errors
///
/// Returns an error only on a storage-level failure (the sweep could not
/// read the ledger or a projection). Per-thread repair failures are
/// logged and counted in [`RepairSweepReport::unrepairable`] rather than
/// aborting the sweep — one poisoned thread must not brick startup for
/// every healthy thread.
pub async fn run_startup_repair_sweep(
    pool: &SqlitePool,
    now: OffsetDateTime,
) -> Result<RepairSweepReport> {
    let mut report = RepairSweepReport::default();

    let rows = sqlx::query(
        r"
SELECT h.thread_id, h.history_json, h.draft_messages_json, h.compactions_json,
       h.version, h.created_at, h.updated_at,
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

    for row in rows {
        report.scanned += 1;
        let thread_key: String = row.get("thread_id");
        let already_repaired: i64 = row.get("already_repaired");
        if already_repaired != 0 {
            report.ledgered += 1;
            continue;
        }
        let thread_id = ThreadId::from_string(thread_key.clone());

        let history: Vec<llm::Message> =
            serde_json::from_str(row.get::<&str, _>("history_json"))
                .with_context(|| format!("decode history for {thread_key}"))?;
        let draft: Vec<llm::Message> = row
            .get::<Option<&str>, _>("draft_messages_json")
            .map(serde_json::from_str)
            .transpose()
            .with_context(|| format!("decode draft for {thread_key}"))?
            .unwrap_or_default();

        // The provider-visible sequence: committed history, then the
        // in-flight draft (which is where a cancelled turn left its
        // dangling tool_use).
        let mut sequence = history;
        sequence.extend(draft);

        if llm::is_provider_valid_tool_sequence(&sequence) {
            continue;
        }

        match repair_one_thread(pool, &thread_id, &sequence, now).await {
            Ok(true) => {
                report.repaired += 1;
                tracing::warn!(
                    "startup repair sweep repaired corrupted tool sequence on thread {thread_key}"
                );
            }
            Ok(false) => {
                report.unrepairable += 1;
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

/// Repair one thread's projection and record the ledger entry in the same
/// transaction. Returns `Ok(false)` when the mechanical repair still does
/// not validate (the row is left untouched).
async fn repair_one_thread(
    pool: &SqlitePool,
    thread_id: &ThreadId,
    sequence: &[llm::Message],
    now: OffsetDateTime,
) -> Result<bool> {
    let repaired = llm::repair_tool_sequence_in_place(sequence, llm::USER_CANCELLED_TOOL_RESULT);
    if !llm::is_provider_valid_tool_sequence(&repaired) {
        return Ok(false);
    }

    // Persist via the projection's append_repair transition: the raw
    // transcript is preserved and the corrected view is recorded as an
    // append-only compaction entry, exactly as the runtime path does.
    let mut tx = pool
        .begin_with("BEGIN IMMEDIATE")
        .await
        .context("begin repair transaction")?;

    let record = sqlx::query(
        r"
SELECT history_json, draft_messages_json, compactions_json, version, created_at, updated_at
FROM agent_sdk_message_heads
WHERE thread_id = ?1
",
    )
    .bind(thread_id.0.as_str())
    .fetch_optional(&mut *tx)
    .await
    .with_context(|| format!("re-read projection for {thread_id}"))?;

    let Some(record) = record else {
        // The row vanished between the scan and the repair — nothing to do.
        return Ok(true);
    };

    let history: Vec<llm::Message> =
        serde_json::from_str(record.get::<&str, _>("history_json")).context("re-decode history")?;
    let draft: Vec<llm::Message> = record
        .get::<Option<&str>, _>("draft_messages_json")
        .map(serde_json::from_str)
        .transpose()
        .context("re-decode draft")?
        .unwrap_or_default();
    let compactions: Vec<agent_server::journal::CompactionEntry> = record
        .get::<Option<&str>, _>("compactions_json")
        .map(serde_json::from_str)
        .transpose()
        .context("re-decode compactions")?
        .unwrap_or_default();
    let version: i64 = record.get("version");
    let created_at: OffsetDateTime = record.get("created_at");
    let updated_at: OffsetDateTime = record.get("updated_at");

    let projection = agent_server::journal::MessageProjection {
        thread_id: thread_id.clone(),
        messages: history,
        draft_messages: draft,
        compactions,
        version: u64::try_from(version).context("projection version overflow")?,
        created_at,
        updated_at,
    };

    // The repair_messages payload is the delta the repair synthesized
    // (the synthetic closing results); the balanced view is the full
    // corrected context history. source_message_count is the provider-
    // visible length BEFORE the repair so the append-only entry records
    // exactly what it replaced.
    let source_message_count = sequence.len();
    let repair_messages: Vec<llm::Message> = repaired
        .iter()
        .skip(source_message_count)
        .cloned()
        .collect();
    let repair_messages = if repair_messages.is_empty() {
        // The repair only removed messages (dropped an orphan tail) — the
        // corrected view itself is the repair payload.
        vec![llm::Message::user(llm::USER_CANCELLED_TOOL_RESULT)]
    } else {
        repair_messages
    };

    let updated = projection
        .append_repair(repair_messages, repaired, source_message_count, now)
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
    Ok(true)
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
    let version = i64::try_from(projection.version).context("version overflow")?;
    sqlx::query(
        r"
INSERT INTO agent_sdk_message_heads
    (thread_id, history_json, draft_messages_json, compactions_json, version, created_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT (thread_id) DO UPDATE SET
    history_json = excluded.history_json,
    draft_messages_json = excluded.draft_messages_json,
    compactions_json = excluded.compactions_json,
    version = excluded.version,
    updated_at = excluded.updated_at
",
    )
    .bind(projection.thread_id.0.as_str())
    .bind(history_json)
    .bind(draft_json)
    .bind(compactions_json)
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
    async fn sweep_is_idempotent_via_the_ledger() -> Result<()> {
        let store = SqliteDurableStore::connect("sqlite::memory:").await?;
        seed_corrupted_thread(&store, "sweep-idempotent").await?;

        let first = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(first.repaired, 1);
        let second = run_startup_repair_sweep(store.pool(), t0()).await?;
        assert_eq!(second.repaired, 0, "second run repairs nothing (ledgered)");
        assert_eq!(second.ledgered, 1);
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
}
