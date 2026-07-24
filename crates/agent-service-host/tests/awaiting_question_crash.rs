#![cfg(feature = "sqlite")]

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, StopReason, Tool, Usage,
};
use agent_sdk_foundation::{
    AgentContinuation, AgentEvent, AgentState, ContinuationEnvelope, PendingToolCallInfo,
    QuestionAnswer, QuestionPayload, ThreadId, TokenUsage, ToolTier,
};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::store::{AgentTaskStore, QuestionPause};
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, SuspensionPayload, WorkerId};
use agent_server::journal::task_state::TaskState;
use agent_server::journal::thread_store::ThreadStore;
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{RootTurnDeps, RootTurnOutcome, resume_from_question};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use time::OffsetDateTime;

const CHILD_ENV: &str = "AGENT_SDK_QUESTION_CRASH_CHILD";
const PARK_CHILD_ENV: &str = "AGENT_SDK_QUESTION_PARK_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_QUESTION_CRASH_DB";
const TASK_ENV: &str = "AGENT_SDK_QUESTION_CRASH_TASK";
const READY_ENV: &str = "AGENT_SDK_QUESTION_CRASH_READY";

fn question_payload() -> QuestionPayload {
    QuestionPayload {
        tool_call_id: "crash-question-call".into(),
        question: "Which target?".into(),
        header: Some("Deploy".into()),
        options: Vec::new(),
        multi_select: false,
    }
}

struct AnswerCheckingProvider {
    saw_persisted_answer: AtomicBool,
}

impl AnswerCheckingProvider {
    const fn new() -> Self {
        Self {
            saw_persisted_answer: AtomicBool::new(false),
        }
    }

    fn saw_persisted_answer(&self) -> bool {
        self.saw_persisted_answer.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for AnswerCheckingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let answer_applied = request.messages.iter().any(|message| {
            matches!(
                &message.content,
                Content::Blocks(blocks)
                    if blocks.iter().any(|block| matches!(
                        block,
                        ContentBlock::ToolResult { tool_use_id, content, .. }
                            if tool_use_id == "crash-question-call"
                                && content == "User answered: Staging"
                    ))
            )
        });
        ensure!(
            answer_applied,
            "boot resume did not apply the persisted question answer"
        );
        self.saw_persisted_answer.store(true, Ordering::SeqCst);
        Ok(ChatOutcome::Success(ChatResponse {
            id: "crash-resume-response".into(),
            content: vec![ContentBlock::Text {
                text: "deploying to staging".into(),
            }],
            model: "crash-test-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "crash-test-model"
    }

    fn provider(&self) -> &'static str {
        "crash-test"
    }
}

fn crash_test_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "crash-test".into(),
        model: "crash-test-model".into(),
        system_prompt: "Resume after a durable answer.".into(),
        max_tokens: 256,
        tools: vec![Tool {
            name: "ask_user".into(),
            description: "Ask the user a question".into(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Ask User".into(),
            tier: ToolTier::Observe,
        }],
        thinking: ThinkingPolicy::default(),
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn suspended_messages() -> Vec<agent_sdk_foundation::llm::Message> {
    vec![
        agent_sdk_foundation::llm::Message::user("Deploy the service"),
        agent_sdk_foundation::llm::Message::assistant_with_content(vec![ContentBlock::ToolUse {
            id: "crash-question-call".into(),
            name: "ask_user".into(),
            input: serde_json::json!({"question": "Which target?"}),
            thought_signature: None,
        }]),
    ]
}

async fn seed_awaiting_question(store: &SqliteDurableStore) -> Result<AgentTaskId> {
    let now = OffsetDateTime::UNIX_EPOCH;
    let thread_id = ThreadId::from_string("kill-nine-question-thread");
    ThreadStore::get_or_create(store, &thread_id, now).await?;
    MessageProjectionStore::get_or_create(store, &thread_id, now).await?;
    let task = AgentTask::new_root_turn(thread_id.clone(), now, 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await?;
    let worker = WorkerId::from_string("seed-worker");
    let lease = LeaseId::from_string("seed-lease");
    store
        .try_acquire_task(
            &task_id,
            worker.clone(),
            lease.clone(),
            now + time::Duration::minutes(1),
            now,
        )
        .await?
        .context("seed task was not acquired")?;
    let call = PendingToolCallInfo {
        id: "crash-question-call".into(),
        name: "ask_user".into(),
        display_name: "Ask User".into(),
        tier: ToolTier::Observe,
        input: serde_json::json!({"question": "Which target?"}),
        effective_input: serde_json::json!({"question": "Which target?"}),
        listen_context: None,
    };
    // The park commits the state row AND the QuestionAsked journal
    // event in one store transaction — no separate commit_event call.
    store
        .pause_on_question(
            &task_id,
            &worker,
            &lease,
            QuestionPause {
                delivered_injection_ids: Vec::new(),
                payload: SuspensionPayload {
                    continuation: ContinuationEnvelope::wrap(AgentContinuation {
                        thread_id: thread_id.clone(),
                        turn: 1,
                        total_usage: TokenUsage::default(),
                        turn_usage: TokenUsage::default(),
                        pending_tool_calls: vec![call],
                        awaiting_index: 0,
                        completed_results: Vec::new(),
                        state: AgentState::new(thread_id.clone()),
                        response_id: None,
                        stop_reason: Some(StopReason::ToolUse),
                        response_content: Vec::new(),
                    }),
                    suspended_messages: suspended_messages(),
                },
                questions: vec![question_payload()],
                events: Vec::new(),
            },
            now,
        )
        .await?;
    Ok(task_id)
}

#[tokio::test]
async fn question_crash_child() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let task_id =
        AgentTaskId::from_string(std::env::var(TASK_ENV).context("missing crash task id")?);
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    store
        .answer_question(
            &task_id,
            "kill-nine-receipt",
            vec![QuestionAnswer {
                tool_call_id: "crash-question-call".into(),
                answer: "Staging".into(),
            }],
            OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1),
        )
        .await?;
    std::fs::write(ready, b"answer committed")?;
    tokio::time::sleep(Duration::from_mins(1)).await;
    bail!("child was not killed after committing the answer")
}

/// Child-process half of the crash-then-answer test: park on the
/// question, publish the task id, then wait to be killed while parked.
#[tokio::test]
async fn question_park_crash_child() -> Result<()> {
    if std::env::var_os(PARK_CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;
    let task_id = seed_awaiting_question(&store).await?;
    std::fs::write(ready, task_id.as_str())?;
    tokio::time::sleep(Duration::from_mins(1)).await;
    bail!("child was not killed while parked on the question")
}

/// Spawn the park child, wait for it to publish the parked task id,
/// then `kill -9` it while the task is still `AwaitingQuestion`.
async fn park_then_kill(url: &str, ready_path: &std::path::Path) -> Result<AgentTaskId> {
    let executable = std::env::current_exe()?;
    let mut child = Command::new(executable)
        .arg("question_park_crash_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(PARK_CHILD_ENV, "1")
        .env(DB_ENV, url)
        .env(READY_ENV, ready_path)
        .spawn()
        .context("spawn park child")?;

    for _ in 0..100 {
        if ready_path.exists() {
            let task_id = std::fs::read_to_string(ready_path)?;
            let kill_status = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("send kill -9 to park child")?;
            ensure!(kill_status.success(), "kill -9 failed with {kill_status}");
            let _ = child.wait()?;
            return Ok(AgentTaskId::from_string(task_id.trim()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "park child did not reach the park boundary: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

async fn answer_then_kill(
    url: &str,
    task_id: &AgentTaskId,
    ready_path: &std::path::Path,
) -> Result<()> {
    let executable = std::env::current_exe()?;
    let mut child = Command::new(executable)
        .arg("question_crash_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, url)
        .env(TASK_ENV, task_id.as_str())
        .env(READY_ENV, ready_path)
        .spawn()
        .context("spawn answer child")?;

    for _ in 0..100 {
        if ready_path.exists() {
            let kill_status = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("send kill -9 to answer child")?;
            ensure!(kill_status.success(), "kill -9 failed with {kill_status}");
            let _ = child.wait()?;
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "answer child did not reach commit boundary: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

async fn assert_boot_resumes_answer(
    url: &str,
    task_id: &AgentTaskId,
    expected_receipt: &str,
) -> Result<()> {
    let boot = SqliteDurableStore::connect(url).await?;
    let loaded = AgentTaskStore::get(&boot, task_id)
        .await?
        .context("boot lost answered task")?;
    let (receipt_id, answers) = loaded
        .state
        .question_answers()
        .with_context(|| format!("boot did not recover AnsweredQuestion: {:?}", loaded.state))?;
    assert_eq!(receipt_id, expected_receipt);
    let [answer] = answers else {
        bail!("expected exactly one persisted answer, got {answers:?}");
    };
    assert_eq!(answer.tool_call_id, "crash-question-call");
    assert_eq!(answer.answer, "Staging");
    assert_eq!(
        loaded.state.questions().context("questions retained")?,
        std::slice::from_ref(&question_payload()),
    );

    let acquired = boot
        .try_acquire_task(
            task_id,
            WorkerId::from_string("boot-worker"),
            LeaseId::from_string("boot-lease"),
            OffsetDateTime::UNIX_EPOCH + time::Duration::minutes(2),
            OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(2),
        )
        .await?
        .context("boot did not make answered task runnable")?;
    assert!(matches!(acquired.state, TaskState::AnsweredQuestion { .. }));

    let bootstrap = WorkerBootstrapContext {
        thread_id: acquired.thread_id.clone(),
        task_id: acquired.id.clone(),
        task: acquired.clone(),
        worker_id: WorkerId::from_string("boot-worker"),
        lease_id: LeaseId::from_string("boot-lease"),
        definition: crash_test_definition(),
    };
    let resume_at = OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(3);
    let inputs = build_root_worker_inputs(bootstrap, &boot, &boot, &boot, resume_at).await?;
    let notifier = EventNotifier::new();
    let deps = RootTurnDeps {
        task_store: &boot,
        thread_store: &boot,
        message_store: &boot,
        attempt_store: &boot,
        checkpoint_store: &boot,
        event_repo: &boot,
        event_notifier: &notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };
    let provider = AnswerCheckingProvider::new();
    let outcome = resume_from_question(inputs, &acquired, &provider, &deps, resume_at).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "boot did not complete the answered turn"
    );
    ensure!(
        provider.saw_persisted_answer(),
        "resume provider never received the persisted answer"
    );
    let question_events = EventRepository::get_events(&boot, &acquired.thread_id)
        .await?
        .into_iter()
        .filter(|event| matches!(event.event, AgentEvent::QuestionAsked { .. }))
        .count();
    ensure!(question_events == 1, "boot resume re-asked the question");
    Ok(())
}

#[tokio::test]
async fn kill_nine_after_answer_boot_reacquires_persisted_answer() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let db_path = temp.path().join("question-crash.db");
    let ready_path = temp.path().join("answer-committed");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    let seed_store = SqliteDurableStore::connect(&url).await?;
    let task_id = seed_awaiting_question(&seed_store).await?;
    drop(seed_store);

    answer_then_kill(&url, &task_id, &ready_path).await?;
    assert_boot_resumes_answer(&url, &task_id, "kill-nine-receipt").await
}

/// Crash-then-answer: the process dies while the task is PARKED in
/// `AwaitingQuestion`; a later boot must find the question durable
/// (state + `QuestionAsked` event, committed in one transaction),
/// accept the answer, and resume without re-asking.
#[tokio::test]
async fn kill_nine_while_parked_boot_answers_and_resumes() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let db_path = temp.path().join("question-park-crash.db");
    let ready_path = temp.path().join("question-parked");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let task_id = park_then_kill(&url, &ready_path).await?;

    let boot = SqliteDurableStore::connect(&url).await?;
    let parked = AgentTaskStore::get(&boot, &task_id)
        .await?
        .context("boot lost the parked task")?;
    ensure!(
        matches!(parked.state, TaskState::AwaitingQuestion { .. }),
        "boot did not recover AwaitingQuestion: {:?}",
        parked.state,
    );
    let question_events = EventRepository::get_events(&boot, &parked.thread_id)
        .await?
        .into_iter()
        .filter(|event| matches!(event.event, AgentEvent::QuestionAsked { .. }))
        .count();
    ensure!(
        question_events == 1,
        "the park must journal exactly one QuestionAsked atomically with the state \
         commit, found {question_events}",
    );
    boot.answer_question(
        &task_id,
        "park-crash-receipt",
        vec![QuestionAnswer {
            tool_call_id: "crash-question-call".into(),
            answer: "Staging".into(),
        }],
        OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1),
    )
    .await?;
    drop(boot);

    assert_boot_resumes_answer(&url, &task_id, "park-crash-receipt").await
}
