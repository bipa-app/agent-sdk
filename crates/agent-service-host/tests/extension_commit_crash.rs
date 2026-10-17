#![cfg(all(feature = "sqlite", feature = "failpoints"))]

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
use agent_server::journal::checkpoint::CheckpointKind;
use agent_server::journal::commit::{CompletedTurnCommit, commit_completed_turn};
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttemptId, TurnAttemptOutcome,
};
use agent_server::journal::turn_attempt_store::TurnAttemptStore;
use agent_server::journal::{
    DocExpect, DocKind, DocScope, DocumentKey, DocumentStore, DocumentWrite, TransitionExtension,
    TransitionView, Write,
};
use agent_service_host::fail;
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use time::OffsetDateTime;

const CHILD_ENV: &str = "AGENT_SDK_EXTENSION_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_EXTENSION_CRASH_DB";
const READY_ENV: &str = "AGENT_SDK_EXTENSION_CRASH_READY";
const FAILPOINT: &str = "commit.before_event_commit";
const THREAD: &str = "extension-crash";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

fn booked(thread: &ThreadId) -> Result<DocumentKey> {
    Ok(DocumentKey::new(
        DocScope::Thread(thread.clone()),
        DocKind::new("crash.booked_turns")?,
        "",
    ))
}

/// Books every completed turn into a document on its thread, the way a
/// host books cost.
struct BookTurns;

impl TransitionExtension for BookTurns {
    fn extend(&self, view: TransitionView<'_>) -> Vec<Write> {
        let TransitionView::CompletedTurn(commit) = view else {
            return Vec::new();
        };
        booked(&commit.thread_id)
            .map(|key| {
                vec![Write::Document(DocumentWrite::put(
                    key,
                    1,
                    DocExpect::Any,
                    serde_json::json!({ "turn": commit.expected_turn }),
                ))]
            })
            .unwrap_or_default()
    }
}

async fn store(url: &str) -> Result<SqliteDurableStore> {
    Ok(SqliteDurableStore::connect(url)
        .await?
        .with_transition_extension(Arc::new(BookTurns)))
}

/// Submit and lease a root and open its attempt; return the task and
/// attempt ids its first turn commits with.
async fn start_first_turn(store: &SqliteDurableStore) -> Result<(AgentTaskId, TurnAttemptId)> {
    let thread = ThreadId::from_string(THREAD);
    let root = AgentTask::new_root_turn(thread.clone(), at(1), 3);
    let _ = store.submit_root_turn(root.clone()).await?;
    let _ = store
        .try_acquire_task(&root.id, WorkerId::new(), LeaseId::new(), at(60), at(2))
        .await?;
    let attempt = store
        .open_attempt(OpenAttemptParams {
            task_id: root.id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("crash-test", "crash-test-model"),
            request_blob: serde_json::json!({}),
            now: at(3),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;
    Ok((root.id, attempt.id))
}

fn first_turn(task_id: AgentTaskId, turn_attempt_id: TurnAttemptId) -> CompletedTurnCommit {
    CompletedTurnCommit {
        thread_id: ThreadId::from_string(THREAD),
        task_id,
        expected_turn: 1,
        delivered_injection_ids: Vec::new(),
        turn_attempt_id,
        close_attempt_params: CloseAttemptParams {
            response_blob: serde_json::json!({}),
            response_id: None,
            response_model: None,
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 10,
            output_tokens: 5,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_mode: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
        },
        messages: vec![llm::Message::assistant("done")],
        turn_usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        },
        checkpoint_kind: CheckpointKind::FullTurn,
        agent_state_snapshot: serde_json::json!({}),
        events: Vec::new(),
        outbox_max_attempts: 3,
        owner_guard: None,
        extra_writes: Vec::new(),
        now: at(4),
    }
}

/// Child-process half of the kill test: pause inside the completed turn's
/// transaction, after the extension's document is staged, and wait to be
/// killed there.
#[tokio::test]
async fn extension_commit_kill_child() -> Result<()> {
    let _scenario = fail::FailScenario::setup();
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = store(&url).await?;
    let (task, attempt) = start_first_turn(&store).await?;
    let staged = format!("{} {}", task.as_str(), attempt.as_str());
    fail::cfg_callback(FAILPOINT, move || {
        let _ = std::fs::write(&ready, &staged);
        std::thread::sleep(Duration::from_mins(1));
    })
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    commit_completed_turn(first_turn(task, attempt), &store).await?;
    bail!("the child was not killed inside the completed turn's transaction")
}

/// `kill -9` while a completed turn and its extension document are staged
/// in one `SQLite` transaction: a new process sees neither. It then commits
/// the same turn through the extension, which books the document at
/// revision 1: a revision past 1 would mean the killed attempt left its
/// document behind, and a missing document that the extension's write
/// never ran.
#[tokio::test]
async fn kill_nine_inside_a_completed_turn_leaves_neither_the_turn_nor_its_document() -> Result<()>
{
    let _scenario = fail::FailScenario::setup();
    let temp = tempfile::tempdir()?;
    let db = temp.path().join("extension-kill.db");
    let ready = temp.path().join("staged");
    let url = format!("sqlite://{}?mode=rwc", db.display());

    let mut child = Command::new(std::env::current_exe()?)
        .arg("extension_commit_kill_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, &url)
        .env(READY_ENV, &ready)
        .spawn()
        .context("spawn the completed-turn child")?;
    let mut staged = None;
    for _ in 0..400 {
        if let Ok(text) = std::fs::read_to_string(&ready)
            && let Some((task, attempt)) = text.split_once(' ')
        {
            staged = Some((
                AgentTaskId::from_string(task),
                TurnAttemptId::from_string(attempt),
            ));
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let Some((task, attempt)) = staged else {
        let _ = child.kill();
        let output = child.wait_with_output()?;
        bail!(
            "the child never reached the completed turn's transaction: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    let killed = Command::new("kill")
        .arg("-9")
        .arg(child.id().to_string())
        .status()
        .context("kill -9 the completed-turn child")?;
    ensure!(killed.success(), "kill -9 failed with {killed}");
    let _ = child.wait()?;

    let store = SqliteDurableStore::connect(&url).await?;
    let thread = ThreadId::from_string(THREAD);
    let committed = ThreadStore::get(&store, &thread)
        .await?
        .map_or(0, |thread| thread.committed_turns);
    ensure!(
        committed == 0,
        "the killed turn committed: {committed} turns"
    );
    ensure!(
        store.get_document(&booked(&thread)?).await?.is_none(),
        "the killed turn's document committed"
    );

    let store = self::store(&url).await?;
    let outcome = commit_completed_turn(first_turn(task, attempt), &store).await?;
    ensure!(outcome.thread.committed_turns == 1);
    let booked = store
        .get_document(&booked(&thread)?)
        .await?
        .context("the retried turn books its document")?;
    ensure!(
        booked.revision == 1,
        "the retried turn's document is at revision {}: the killed attempt left one behind",
        booked.revision
    );
    Ok(())
}
