//! Live verification for the ENG-9651 startup repair sweep against a real
//! (copied) founder database. Ignored by default — it requires
//! `ENG9651_VERIFY_DB` to point at a `SQLite` file with the durable schema.
//!
//! Run:
//!   ENG9651_VERIFY_DB=/path/to/agent-runtime.sqlite3 \
//!     cargo test -p agent-service-host --all-features --test `repair_sweep_live` -- --ignored --nocapture

#![cfg(feature = "sqlite")]

use anyhow::{Context, Result};

use agent_sdk_foundation::llm;

#[tokio::test]
#[ignore = "requires ENG9651_VERIFY_DB pointing at a real durable SQLite file"]
async fn sweep_repairs_every_invalid_thread_in_a_real_database() -> Result<()> {
    let path = std::env::var("ENG9651_VERIFY_DB")
        .context("set ENG9651_VERIFY_DB to a copied durable SQLite file")?;
    let url = format!("sqlite:{path}");

    // Count provider-invalid threads BEFORE opening through the store
    // (which runs the sweep).
    let pre_pool = sqlx::SqlitePool::connect(&url).await?;
    let heads: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT thread_id, history_json, draft_messages_json FROM agent_sdk_message_heads",
    )
    .fetch_all(&pre_pool)
    .await?;
    let mut invalid_before = 0usize;
    for (_tid, history, draft) in &heads {
        let mut messages: Vec<llm::Message> = serde_json::from_str(history)?;
        if let Some(d) = draft {
            messages.extend(serde_json::from_str::<Vec<llm::Message>>(d)?);
        }
        if !llm::is_provider_valid_tool_sequence(&messages) {
            invalid_before += 1;
        }
    }
    pre_pool.close().await;
    println!("threads scanned: {}", heads.len());
    println!("invalid before sweep: {invalid_before}");

    // Open through the store: runs schema migrations + the repair sweep.
    let store = agent_service_host::sqlite::store::SqliteDurableStore::connect(&url).await?;

    // Every thread's provider-visible view (compaction-selected context
    // history + draft) must now validate. The repair writes the corrected
    // view as an append-only compaction entry and leaves the raw committed
    // history untouched, so we must reconstruct the view, not read raw
    // history_json.
    let heads_after: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT thread_id, history_json, draft_messages_json, compactions_json FROM agent_sdk_message_heads",
    )
    .fetch_all(store.pool())
    .await?;
    let mut invalid_after = 0usize;
    let mut still_dangling = Vec::new();
    for (tid, history, draft, compactions) in &heads_after {
        let raw: Vec<llm::Message> = serde_json::from_str(history)?;
        // Reconstruct MessageProjection::context_history(): with a
        // compaction entry, the view is the last entry's
        // replacement_messages followed by raw history after its
        // compacted_end boundary.
        let mut view: Vec<llm::Message> = match compactions {
            Some(c) => {
                let entries: Vec<serde_json::Value> = serde_json::from_str(c)?;
                match entries.last() {
                    Some(last) => {
                        let end = usize::try_from(
                            last.get("compacted_end")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0),
                        )
                        .unwrap_or(0);
                        let mut v: Vec<llm::Message> = serde_json::from_value(
                            last.get("replacement_messages")
                                .cloned()
                                .unwrap_or_default(),
                        )?;
                        v.extend(raw.into_iter().skip(end));
                        v
                    }
                    None => raw,
                }
            }
            None => raw,
        };
        if let Some(d) = draft {
            view.extend(serde_json::from_str::<Vec<llm::Message>>(d)?);
        }
        if !llm::is_provider_valid_tool_sequence(&view) {
            invalid_after += 1;
            still_dangling.push(tid.clone());
        }
    }
    println!("invalid after sweep (provider view): {invalid_after}");

    let ledgered: i64 = sqlx::query_scalar("SELECT count(*) FROM agent_sdk_applied_repairs")
        .fetch_one(store.pool())
        .await?;
    println!("ledger entries: {ledgered}");

    assert!(
        invalid_before > 0,
        "the copied DB must contain corruption for this test to be meaningful"
    );
    assert_eq!(
        invalid_after, 0,
        "every thread must be provider-valid after the sweep; still invalid: {still_dangling:?}"
    );
    assert!(
        ledgered > 0,
        "the sweep must record ledger entries for repaired threads"
    );
    Ok(())
}
