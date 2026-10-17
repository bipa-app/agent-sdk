//! Crash-matrix S6: a human prompt survives `kill -9`. A question, a tool
//! confirmation and a host prompt row are parked by a child process that is
//! then killed; after the restart all three are pending, and the host
//! prompt is answered once. A second child answers the host prompt and is
//! killed before its close; the restart hands the answer to the handler
//! once and closes it once.
#![cfg(feature = "sqlite")]

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use agent_sdk_foundation::llm::{ContentBlock, StopReason};
use agent_sdk_foundation::{
    AgentContinuation, AgentEvent, AgentState, ContinuationEnvelope, PendingToolCallInfo,
    QuestionPayload, ThreadId, TokenUsage, ToolResult, ToolTier,
};
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::prompt::{get_prompt, pending_prompts};
use agent_server::journal::store::{AgentTaskStore, QuestionPause};
use agent_server::journal::task::{
    AgentTask, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::{
    DurableStore, Guard, NewPrompt, PendingPrompt, PromptAnswer, PromptCloseFrame, PromptDecision,
    PromptFrame, PromptOwner, PromptRecord, PromptState, SpawnChildrenExt, StoredPrompt,
    Transition,
};
use agent_service_host::prompts::{
    AnswerError, Answered, PromptHandler, PromptResolution, answer_prompt, redrive_answered_prompts,
};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use time::OffsetDateTime;

const PARK_CHILD_ENV: &str = "AGENT_SDK_PROMPT_PARK_CRASH_CHILD";
const ANSWER_CHILD_ENV: &str = "AGENT_SDK_PROMPT_ANSWER_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_PROMPT_CRASH_DB";
const READY_ENV: &str = "AGENT_SDK_PROMPT_CRASH_READY";

const QUESTION_THREAD: &str = "kill-nine-prompt-question";
const TOOL_THREAD: &str = "kill-nine-prompt-tool";
const HOST_THREAD: &str = "kill-nine-prompt-host";
const HOST_PROMPT: &str = "merge-kill-nine";

fn at(seconds: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(seconds)
}

fn call(id: &str, name: &str) -> PendingToolCallInfo {
    PendingToolCallInfo {
        id: id.into(),
        name: name.into(),
        display_name: name.into(),
        tier: ToolTier::Confirm,
        concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
        replay: agent_sdk_foundation::ToolReplay::Unsafe,
        input: serde_json::json!({}),
        effective_input: serde_json::json!({}),
        listen_context: None,
    }
}

fn continuation(thread: &ThreadId, calls: Vec<PendingToolCallInfo>) -> ContinuationEnvelope {
    ContinuationEnvelope::wrap(AgentContinuation {
        thread_id: thread.clone(),
        turn: 1,
        total_usage: TokenUsage::default(),
        turn_usage: TokenUsage::default(),
        pending_tool_calls: calls,
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: AgentState::new(thread.clone()),
        response_id: None,
        stop_reason: Some(StopReason::ToolUse),
        response_content: Vec::new(),
    })
}

/// A thread with a root turn that `(worker, lease)` holds.
async fn running_root(
    store: &SqliteDurableStore,
    thread: &str,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let thread_id = ThreadId::from_string(thread);
    ThreadStore::get_or_create(store, &thread_id, at(0)).await?;
    MessageProjectionStore::get_or_create(store, &thread_id, at(0)).await?;
    let root = AgentTask::new_root_turn(thread_id, at(0), 3);
    store.submit_root_turn(root.clone()).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    let root = store
        .try_acquire_task(&root.id, worker.clone(), lease.clone(), at(600), at(1))
        .await?
        .context("the root is acquirable")?;
    Ok((root, worker, lease))
}

async fn park_question(store: &SqliteDurableStore) -> Result<()> {
    let (root, worker, lease) = running_root(store, QUESTION_THREAD).await?;
    let thread = root.thread_id.clone();
    store
        .pause_on_question(
            &root.id,
            &worker,
            &lease,
            QuestionPause {
                delivered_injection_ids: Vec::new(),
                payload: SuspensionPayload {
                    continuation: continuation(&thread, vec![call("question-call", "ask_user")]),
                    suspended_messages: vec![
                        agent_sdk_foundation::llm::Message::user("Deploy"),
                        agent_sdk_foundation::llm::Message::assistant_with_content(vec![
                            ContentBlock::ToolUse {
                                id: "question-call".into(),
                                name: "ask_user".into(),
                                input: serde_json::json!({}),
                                thought_signature: None,
                            },
                        ]),
                    ],
                    child_join_policy: agent_server::ChildJoinPolicy::default(),
                },
                questions: vec![QuestionPayload {
                    tool_call_id: "question-call".into(),
                    question: "Which target?".into(),
                    header: None,
                    options: Vec::new(),
                    multi_select: false,
                }],
                events: Vec::new(),
            },
            at(2),
        )
        .await?;
    Ok(())
}

async fn park_tool_confirmation(store: &SqliteDurableStore) -> Result<()> {
    let (root, worker, lease) = running_root(store, TOOL_THREAD).await?;
    let thread = root.thread_id.clone();
    let calls = vec![call("tool-call", "deploy")];
    let (_, children) = store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3)],
            SuspensionPayload {
                continuation: continuation(&thread, calls.clone()),
                suspended_messages: Vec::new(),
                child_join_policy: agent_server::ChildJoinPolicy::default(),
            },
            None,
            Vec::new(),
            at(2),
        )
        .await?;
    let child = children.into_iter().next().context("one tool child")?;
    let child_worker = WorkerId::new();
    let child_lease = LeaseId::new();
    store
        .try_acquire_task(
            &child.id,
            child_worker.clone(),
            child_lease.clone(),
            at(600),
            at(3),
        )
        .await?
        .context("the tool child is acquirable")?;
    store
        .commit(
            Transition::pause_on_confirmation(
                child.id.clone(),
                child_worker,
                child_lease,
                continuation(&thread, calls),
                None,
                thread,
                vec![AgentEvent::tool_requires_confirmation(
                    "tool-call",
                    "deploy",
                    "deploy",
                    serde_json::json!({}),
                    "Tool deploy requires confirmation",
                )],
            ),
            at(4),
        )
        .await
        .map_err(anyhow::Error::from)?;
    Ok(())
}

fn host_prompt() -> PromptRecord {
    PromptRecord::open(
        NewPrompt {
            id: HOST_PROMPT.into(),
            surface_thread: ThreadId::from_string(HOST_THREAD),
            owner: PromptOwner::Host {
                name: "crash.merge_gate".into(),
            },
            owner_task: None,
            frame: PromptFrame {
                name: "apply_worker_merge".into(),
                display_name: "Merge worker changes".into(),
                input: serde_json::json!({"branch": "bip/thread/x"}),
                description: "merge?".into(),
            },
            close: PromptCloseFrame {
                name: "apply_worker_merge".into(),
                display_name: "Merge worker changes".into(),
                input: serde_json::json!({"branch": "bip/thread/x"}),
            },
            must_reach_human: false,
            host_state: serde_json::json!({"commit": "abc"}),
        },
        at(5),
    )
}

/// Open the host prompt in the same transaction as a guard on its thread's
/// running root, as a host opens one beside its own task write.
async fn open_host_prompt(store: &SqliteDurableStore) -> Result<()> {
    let (root, worker, lease) = running_root(store, HOST_THREAD).await?;
    store
        .commit(
            Transition {
                guards: vec![Guard::TaskIs {
                    task: root.id,
                    status: TaskStatus::Running,
                    lease: Some((worker, lease)),
                }],
                writes: host_prompt().open_writes(true)?,
            },
            at(6),
        )
        .await
        .map_err(anyhow::Error::from)?;
    Ok(())
}

fn answer(request: &str) -> PromptAnswer {
    PromptAnswer {
        decision: PromptDecision::Approved,
        request_id: request.into(),
        automatic: false,
        answered_at: at(10),
    }
}

#[tokio::test]
async fn prompt_park_crash_child() -> Result<()> {
    if std::env::var_os(PARK_CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    park_question(&store).await?;
    park_tool_confirmation(&store).await?;
    open_host_prompt(&store).await?;
    std::fs::write(ready, b"parked")?;
    tokio::time::sleep(Duration::from_mins(1)).await;
    bail!("child was not killed while its prompts were parked")
}

#[tokio::test]
async fn prompt_answer_crash_child() -> Result<()> {
    if std::env::var_os(ANSWER_CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    let answered = answer_prompt(
        &store,
        &store,
        &ThreadId::from_string(HOST_THREAD),
        HOST_PROMPT,
        answer("kill-nine-answer"),
        at(10),
    )
    .await
    .map_err(anyhow::Error::from)?;
    ensure!(
        matches!(answered, Answered::Fresh(_)),
        "the child answers first"
    );
    std::fs::write(ready, b"answered")?;
    tokio::time::sleep(Duration::from_mins(1)).await;
    bail!("child was not killed after committing the answer")
}

/// Run `child_test` in a child process and `kill -9` it once it writes
/// the ready file.
async fn run_then_kill(
    child_test: &str,
    env: &str,
    url: &str,
    ready: &std::path::Path,
) -> Result<()> {
    let mut child = Command::new(std::env::current_exe()?)
        .arg(child_test)
        .arg("--exact")
        .arg("--nocapture")
        .env(env, "1")
        .env(DB_ENV, url)
        .env(READY_ENV, ready)
        .spawn()
        .with_context(|| format!("spawn {child_test}"))?;
    for _ in 0..200 {
        if ready.exists() {
            let status = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("send kill -9")?;
            ensure!(status.success(), "kill -9 failed with {status}");
            let _ = child.wait()?;
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "{child_test} did not reach its boundary: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

struct CountingHandler {
    calls: AtomicUsize,
}

#[async_trait]
impl PromptHandler for CountingHandler {
    async fn on_answered(&self, prompt: &StoredPrompt) -> Result<PromptResolution> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(PromptResolution::Close {
            result: ToolResult::success(format!("merged {}", prompt.record.id)),
            retain: false,
            extra: Transition::default(),
        })
    }
}

fn kinds(pending: &[PendingPrompt]) -> Vec<String> {
    pending
        .iter()
        .map(|prompt| match prompt {
            PendingPrompt::ToolConfirmation { call, .. } => format!("tool:{}", call.id),
            PendingPrompt::Question { questions, .. } => format!(
                "question:{}",
                questions.first().map_or("", |q| q.tool_call_id.as_str())
            ),
            PendingPrompt::Row(row) => format!("row:{}", row.record.id),
        })
        .collect()
}

/// S6: every parked prompt is pending after `kill -9`, and an answer that
/// committed before the kill is applied once and closed once at restart.
#[tokio::test]
async fn kill_nine_keeps_every_prompt_and_applies_an_answer_once() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("prompts.sqlite3").display()
    );
    run_then_kill(
        "prompt_park_crash_child",
        PARK_CHILD_ENV,
        &url,
        &temp.path().join("parked"),
    )
    .await?;

    let boot = SqliteDurableStore::connect(&url).await?;
    assert_eq!(
        kinds(&pending_prompts(&boot, &boot, &ThreadId::from_string(QUESTION_THREAD)).await?),
        ["question:question-call"]
    );
    assert_eq!(
        kinds(&pending_prompts(&boot, &boot, &ThreadId::from_string(TOOL_THREAD)).await?),
        ["tool:tool-call"]
    );
    assert_eq!(
        kinds(&pending_prompts(&boot, &boot, &ThreadId::from_string(HOST_THREAD)).await?),
        [format!("row:{HOST_PROMPT}")]
    );
    drop(boot);

    run_then_kill(
        "prompt_answer_crash_child",
        ANSWER_CHILD_ENV,
        &url,
        &temp.path().join("answered"),
    )
    .await?;

    let boot = SqliteDurableStore::connect(&url).await?;
    let host_thread = ThreadId::from_string(HOST_THREAD);
    let row = get_prompt(&boot, &host_thread, HOST_PROMPT)
        .await?
        .context("the answered row survived the kill")?;
    ensure!(
        matches!(&row.record.state, PromptState::Answered { answer } if answer.request_id == "kill-nine-answer"),
        "the answer committed before the kill: {:?}",
        row.record.state
    );
    let conflict = answer_prompt(
        &boot,
        &boot,
        &host_thread,
        HOST_PROMPT,
        answer("after-restart"),
        at(11),
    )
    .await;
    ensure!(
        matches!(conflict, Err(AnswerError::AlreadyAnswered { .. })),
        "a second answer is refused: {conflict:?}"
    );

    let handler = CountingHandler {
        calls: AtomicUsize::new(0),
    };
    assert_eq!(
        redrive_answered_prompts(&boot, &boot, &handler, at(12)).await?,
        1
    );
    assert_eq!(
        redrive_answered_prompts(&boot, &boot, &handler, at(13)).await?,
        0
    );
    assert_eq!(handler.calls.load(Ordering::SeqCst), 1);
    assert!(
        get_prompt(&boot, &host_thread, HOST_PROMPT)
            .await?
            .is_none()
    );

    let events = boot.get_events(&host_thread).await?;
    let opens = events
        .iter()
        .filter(|event| matches!(&event.event, AgentEvent::ToolRequiresConfirmation { id, .. } if id == HOST_PROMPT))
        .count();
    let closes = events
        .iter()
        .filter(
            |event| matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == HOST_PROMPT),
        )
        .count();
    assert_eq!((opens, closes), (1, 1));
    Ok(())
}
