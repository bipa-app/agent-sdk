//! A root turn's tool round on the durable backends (crash safety Phase 2,
//! P2-3b2): the consumed LLM request finishes in the spawn's transaction,
//! and a fresh turn closes the calls an ended root left unanswered.
//!
//! The crash cases MUST run under nextest (process-global `fail-rs`
//! registry), like the rest of this suite.

#[cfg(feature = "failpoints")]
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(feature = "failpoints")]
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage,
};
use agent_sdk_foundation::{ThreadId, ToolConcurrency, ToolReplay, ToolTier};
use agent_sdk_providers::LlmProvider;
use agent_server::journal::checkpoint_store::CheckpointStore;
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::execution_context::build_root_worker_inputs;
use agent_server::journal::message_store::MessageProjectionStore;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::turn_attempt_store::TurnAttemptStore;
use agent_server::worker::bootstrap::WorkerBootstrapContext;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::{RootTurnDeps, RootTurnOutcome, execute_root_turn};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use time::OffsetDateTime;

#[cfg(feature = "failpoints")]
use crate::fail;

/// A store that hosts a whole root turn.
pub(super) trait Store:
    AgentTaskStore
    + ThreadStore
    + MessageProjectionStore
    + TurnAttemptStore
    + CheckpointStore
    + EventRepository
{
}

impl<T> Store for T where
    T: AgentTaskStore
        + ThreadStore
        + MessageProjectionStore
        + TurnAttemptStore
        + CheckpointStore
        + EventRepository
{
}

const THREAD: &str = "tool-round";
const TOOL: &str = "probe";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "crash-test".into(),
        model: "crash-test-model".into(),
        system_prompt: "Answer.".into(),
        max_tokens: 256,
        tools: vec![llm::Tool {
            name: TOOL.into(),
            description: "Probes".into(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Probe".into(),
            tier: ToolTier::Observe,
            concurrency: ToolConcurrency::Exclusive,
            replay: ToolReplay::Unsafe,
        }],
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

/// Calls the probe tool on its first `tool_rounds` requests (a fresh call
/// id each), then answers with text. Records every request's messages.
struct ToolRounds {
    tool_rounds: usize,
    requests: Mutex<Vec<Vec<llm::Message>>>,
}

impl ToolRounds {
    const fn new(tool_rounds: usize) -> Self {
        Self {
            tool_rounds,
            requests: Mutex::new(Vec::new()),
        }
    }

    fn last_request(&self) -> Result<Vec<llm::Message>> {
        self.requests
            .lock()
            .map_err(|_| anyhow::anyhow!("requests lock poisoned"))?
            .last()
            .cloned()
            .context("no request was sent")
    }
}

#[async_trait]
impl LlmProvider for ToolRounds {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let call = {
            let mut requests = self
                .requests
                .lock()
                .map_err(|_| anyhow::anyhow!("requests lock poisoned"))?;
            requests.push(request.messages);
            requests.len()
        };
        let (content, stop) = if call <= self.tool_rounds {
            (
                vec![ContentBlock::ToolUse {
                    id: format!("call-{call}"),
                    name: TOOL.into(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            )
        } else {
            (
                vec![ContentBlock::Text {
                    text: "the answer".into(),
                }],
                StopReason::EndTurn,
            )
        };
        Ok(ChatOutcome::Success(ChatResponse {
            id: format!("r-{call}"),
            content,
            model: "crash-test-model".into(),
            stop_reason: Some(stop),
            usage: Usage {
                served_speed: None,
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
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

fn deps<'a, S: Store>(store: &'a S, notifier: &'a EventNotifier) -> RootTurnDeps<'a> {
    RootTurnDeps {
        task_store: store,
        thread_store: store,
        message_store: store,
        attempt_store: store,
        checkpoint_store: store,
        event_repo: store,
        event_notifier: notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    }
}

fn lease_of(name: &str) -> (WorkerId, LeaseId) {
    (
        WorkerId::from_string(format!("w-{name}")),
        LeaseId::from_string(format!("l-{name}")),
    )
}

/// Submit a root on the thread and lease it to `name`'s worker.
async fn leased_root<S: Store>(store: &S, name: &str) -> Result<AgentTaskId> {
    let thread = ThreadId::from_string(THREAD);
    ThreadStore::get_or_create(store, &thread, at(0)).await?;
    MessageProjectionStore::get_or_create(store, &thread, at(0)).await?;
    let root = AgentTask::new_root_turn(thread, at(0), 3);
    store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = lease_of(name);
    store
        .try_acquire_task(&root.id, worker, lease, at(600), at(1))
        .await?
        .context("acquire")?;
    Ok(root.id)
}

/// Run one turn of `root` under `name`'s lease.
async fn run_turn<S: Store>(
    store: &S,
    root: &AgentTaskId,
    name: &str,
    prompt: &str,
    provider: &ToolRounds,
) -> Result<RootTurnOutcome> {
    let task = AgentTaskStore::get(store, root).await?.context("root")?;
    let (worker_id, lease_id) = lease_of(name);
    let bootstrap = WorkerBootstrapContext {
        thread_id: ThreadId::from_string(THREAD),
        task_id: root.clone(),
        task,
        worker_id,
        lease_id,
        definition: definition(),
    };
    let inputs = build_root_worker_inputs(bootstrap, store, store, store, at(2)).await?;
    let notifier = EventNotifier::new();
    execute_root_turn(inputs, prompt, provider, &deps(store, &notifier), at(3)).await
}

/// The ids of the calls `messages` closes as cancelled.
fn cancelled_closures(messages: &[llm::Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match &message.content {
            llm::Content::Blocks(blocks) => Some(blocks),
            llm::Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } if content == llm::USER_CANCELLED_TOOL_RESULT => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect()
}

/// A root parked on `call-1`, then cancelled without its cancel commit
/// (the process died during the cancel): its draft ends in the call.
async fn cancelled_mid_tool_batch<S: Store>(store: &S) -> Result<()> {
    let first = leased_root(store, "first").await?;
    let outcome = run_turn(store, &first, "first", "first prompt", &ToolRounds::new(1)).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Suspended { .. }),
        "the first turn must park on its tool call"
    );
    store.cancel_tree(&first, at(4)).await?;
    Ok(())
}

/// The committed transcript and the draft of the thread.
async fn transcript<S: Store>(store: &S) -> Result<(Vec<llm::Message>, Vec<llm::Message>)> {
    let projection = MessageProjectionStore::get(store, &ThreadId::from_string(THREAD))
        .await?
        .context("projection")?;
    Ok((projection.messages, projection.draft_messages))
}

/// Review of #534, B2, without a store reopen: the next root's request is
/// provider-valid, closing the cancelled root's call once, and the
/// closure commits with the turn.
pub(super) async fn the_next_turn_closes_a_cancelled_roots_call<S: Store>(store: &S) -> Result<()> {
    cancelled_mid_tool_batch(store).await?;
    let second = leased_root(store, "second").await?;
    let provider = ToolRounds::new(0);
    let outcome = run_turn(store, &second, "second", "second prompt", &provider).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "the second turn must complete"
    );
    let sent = provider.last_request()?;
    ensure!(
        llm::provider_tool_sequence_error_index(&sent).is_none(),
        "the next request is not provider-valid: {sent:?}"
    );
    ensure!(
        cancelled_closures(&sent) == ["call-1"],
        "the request closes the call once: {sent:?}"
    );
    let (committed, draft) = transcript(store).await?;
    ensure!(draft.is_empty(), "the draft must clear: {draft:?}");
    ensure!(
        cancelled_closures(&committed) == ["call-1"],
        "the closure commits once: {committed:?}"
    );
    Ok(())
}

/// Count each hit of `failpoint`, and crash (panic) at it when `crash`.
#[cfg(feature = "failpoints")]
fn count_hits(failpoint: &'static str, crash: bool) -> Result<Arc<AtomicUsize>> {
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&hits);
    fail::cfg_callback(failpoint, move || {
        seen.fetch_add(1, Ordering::SeqCst);
        assert!(!crash, "crash at {failpoint}");
    })
    .map_err(anyhow::Error::msg)?;
    Ok(hits)
}

/// Run `turn`, which must crash at a failpoint (a panic).
#[cfg(feature = "failpoints")]
async fn crashes<T>(what: &str, turn: impl std::future::Future<Output = Result<T>>) -> Result<()> {
    let crashed = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(turn)).await;
    ensure!(crashed.is_err(), "{what}: the failpoint must fire");
    Ok(())
}

/// The states of `root`'s LLM requests.
#[cfg(feature = "failpoints")]
async fn llm_requests<S: Store>(
    store: &S,
    root: &AgentTaskId,
) -> Result<Vec<agent_server::journal::effect::EffectState>> {
    use agent_server::journal::DurableStore;
    use agent_server::journal::effect::EffectKind;
    Ok(DurableStore::effects_for_task(store, root)
        .await?
        .into_iter()
        .filter(|record| record.kind() == EffectKind::LlmRequest)
        .map(|record| record.state)
        .collect())
}

/// The failpoints after the consumed request's finish is written, inside
/// the spawn transaction: the finish itself, and the spawn's rows (an
/// ordinary tool round spawns with its events, so through the mixed
/// spawn).
#[cfg(feature = "failpoints")]
pub(super) const AFTER_SPAWN_FINISH: &[&str] =
    &["effect.finish_written", "atomic_spawn_batch.after_rows"];

/// Crash a tool round at `failpoint`, inside its spawn transaction, then
/// reopen with `reopen` and retry. The crash leaves no child, the root
/// `Running` and its request `Started` (its attempt closed `Success`, the
/// close's own commit): the finish rolled back with the spawn. The retry
/// parks the root on its child with its own request `Completed`; the
/// crashed one stays `Started` for the root's settle to abandon (crash
/// matrix S11).
#[cfg(feature = "failpoints")]
pub(super) async fn crash_in_the_spawn_then_retry<S: Store>(
    failpoint: &'static str,
    store: S,
    reopen: impl std::future::Future<Output = Result<S>>,
) -> Result<()> {
    use agent_server::journal::TurnAttemptOutcome;
    use agent_server::journal::effect::EffectState;
    use agent_server::journal::task::TaskStatus;

    let root = leased_root(&store, "root").await?;
    let finishes = count_hits(
        "effect.finish_written",
        failpoint == "effect.finish_written",
    )?;
    if failpoint != "effect.finish_written" {
        fail::cfg(failpoint, "panic").map_err(anyhow::Error::msg)?;
    }
    let provider = ToolRounds::new(usize::MAX);
    let turn = run_turn(&store, &root, "root", "a prompt", &provider);
    let crashed = crashes(failpoint, turn).await;
    fail::remove("effect.finish_written");
    fail::remove(failpoint);
    crashed?;
    // The request's finish was written in the crashed transaction.
    let reached = finishes.load(Ordering::SeqCst);
    ensure!(reached == 1, "{failpoint}: finishes written {reached}");
    drop(store);

    let store = reopen.await?;
    let row = AgentTaskStore::get(&store, &root).await?.context("root")?;
    ensure!(
        row.status == TaskStatus::Running,
        "{failpoint}: {:?}",
        row.status
    );
    let children = store.list_children(&root).await?;
    ensure!(children.is_empty(), "{failpoint}: children {children:?}");
    let crashed_requests = llm_requests(&store, &root).await?;
    ensure!(
        crashed_requests == [EffectState::Started],
        "{failpoint}: requests after the crash {crashed_requests:?}"
    );
    let attempts = TurnAttemptStore::list_by_task(&store, &root).await?;
    ensure!(
        attempts.len() == 1 && attempts[0].outcome == Some(TurnAttemptOutcome::Success),
        "{failpoint}: the crash comes after the close: {attempts:?}"
    );

    let outcome = run_turn(&store, &root, "root", "a prompt", &provider).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        anyhow::bail!("{failpoint}: the retry must park on its tool call");
    };
    ensure!(child_tasks.len() == 1, "{failpoint}: {child_tasks:?}");
    let row = AgentTaskStore::get(&store, &root).await?.context("root")?;
    ensure!(
        row.status == TaskStatus::WaitingOnChildren,
        "{failpoint}: {:?}",
        row.status
    );
    let retried = llm_requests(&store, &root).await?;
    ensure!(
        retried.len() == 2
            && retried.contains(&EffectState::Completed)
            && retried.contains(&EffectState::Started),
        "{failpoint}: the retry's request finished with its spawn: {retried:?}"
    );
    Ok(())
}

/// Park `root` on its first tool call, finish the call's child, and lease
/// the woken root to its worker again, ready to resume.
#[cfg(feature = "failpoints")]
async fn woken_after_one_round<S: Store>(
    store: &S,
    root: &AgentTaskId,
    provider: &ToolRounds,
) -> Result<()> {
    let outcome = run_turn(store, root, "root", "a prompt", provider).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        anyhow::bail!("the first round must park on its tool call");
    };
    let [child] = child_tasks.as_slice() else {
        anyhow::bail!("one child per call: {child_tasks:?}");
    };
    let (worker, lease) = lease_of("child");
    store
        .try_acquire_task(&child.id, worker.clone(), lease.clone(), at(600), at(4))
        .await?
        .context("acquire the child")?;
    let result = agent_sdk_foundation::ToolResult {
        success: true,
        output: "probed".into(),
        artifact: None,
        data: None,
        documents: Vec::new(),
        duration_ms: None,
    };
    store
        .complete_task_with_result(
            &child.id,
            &worker,
            &lease,
            serde_json::to_value(result)?,
            at(5),
        )
        .await?;
    let (worker, lease) = lease_of("root");
    store
        .try_acquire_task(root, worker, lease, at(600), at(6))
        .await?
        .context("acquire the woken root")?;
    Ok(())
}

/// Resume the woken `root` from its children under its worker's lease.
#[cfg(feature = "failpoints")]
async fn resume<S: Store>(
    store: &S,
    root: &AgentTaskId,
    provider: &ToolRounds,
) -> Result<RootTurnOutcome> {
    let task = AgentTaskStore::get(store, root).await?.context("root")?;
    let (worker_id, lease_id) = lease_of("root");
    let bootstrap = WorkerBootstrapContext {
        thread_id: ThreadId::from_string(THREAD),
        task_id: root.clone(),
        task: task.clone(),
        worker_id,
        lease_id,
        definition: definition(),
    };
    let inputs = build_root_worker_inputs(bootstrap, store, store, store, at(7)).await?;
    let notifier = EventNotifier::new();
    agent_server::worker::resume_from_children(
        inputs,
        &task,
        provider,
        &deps(store, &notifier),
        at(8),
    )
    .await
}

/// Review of #548, B1: crash a resumed turn whose response calls a tool
/// again at `failpoint`, inside the spawn transaction that re-parks the
/// root (`suspend_resumed_turn`), then reopen with `reopen` and retry the
/// resume. The crash leaves no new child, the root `Running` and the
/// resumed request `Started`: its finish rolled back with the spawn. The
/// retry re-parks the root with its own request `Completed`.
#[cfg(feature = "failpoints")]
pub(super) async fn crash_in_the_resumed_spawn_then_retry<S: Store>(
    failpoint: &'static str,
    store: S,
    reopen: impl std::future::Future<Output = Result<S>>,
) -> Result<()> {
    use agent_server::journal::effect::EffectState;
    use agent_server::journal::task::TaskStatus;

    let root = leased_root(&store, "root").await?;
    let provider = ToolRounds::new(usize::MAX);
    woken_after_one_round(&store, &root, &provider).await?;
    let finishes = count_hits(
        "effect.finish_written",
        failpoint == "effect.finish_written",
    )?;
    if failpoint != "effect.finish_written" {
        fail::cfg(failpoint, "panic").map_err(anyhow::Error::msg)?;
    }
    let crashed = crashes(failpoint, resume(&store, &root, &provider)).await;
    fail::remove("effect.finish_written");
    fail::remove(failpoint);
    crashed?;
    // The resumed request's finish was written in the crashed transaction.
    let reached = finishes.load(Ordering::SeqCst);
    ensure!(reached == 1, "{failpoint}: finishes written {reached}");
    drop(store);

    let store = reopen.await?;
    let row = AgentTaskStore::get(&store, &root).await?.context("root")?;
    ensure!(
        row.status == TaskStatus::Running,
        "{failpoint}: {:?}",
        row.status
    );
    let children = store.list_children(&root).await?;
    ensure!(children.len() == 1, "{failpoint}: children {children:?}");
    let completed = |states: &[EffectState]| {
        states
            .iter()
            .filter(|state| **state == EffectState::Completed)
            .count()
    };
    let crashed_requests = llm_requests(&store, &root).await?;
    ensure!(
        crashed_requests.len() == 2 && completed(&crashed_requests) == 1,
        "{failpoint}: requests after the crash {crashed_requests:?}"
    );

    let outcome = resume(&store, &root, &provider).await?;
    let RootTurnOutcome::Suspended { child_tasks, .. } = outcome else {
        anyhow::bail!("{failpoint}: the retry must re-park on its tool call");
    };
    ensure!(child_tasks.len() == 1, "{failpoint}: {child_tasks:?}");
    let row = AgentTaskStore::get(&store, &root).await?.context("root")?;
    ensure!(
        row.status == TaskStatus::WaitingOnChildren,
        "{failpoint}: {:?}",
        row.status
    );
    let retried = llm_requests(&store, &root).await?;
    ensure!(
        retried.len() == 3 && completed(&retried) == 2 && retried.contains(&EffectState::Started),
        "{failpoint}: the retry's request finished with its re-park: {retried:?}"
    );
    Ok(())
}

/// The closure of a cancelled root's call is never doubled by a crash: a
/// crash inside the next turn's completed-turn transaction (after its
/// root's settle is written) commits no turn and leaves the one closure in
/// the draft; the retry commits the turn with that closure, once.
#[cfg(feature = "failpoints")]
pub(super) async fn crash_in_the_closing_turn_then_retry<S: Store>(
    store: S,
    reopen: impl std::future::Future<Output = Result<S>>,
) -> Result<()> {
    use agent_server::journal::task::TaskStatus;

    const SETTLED: &str = "commit.completed_turn.settled";
    cancelled_mid_tool_batch(&store).await?;
    let second = leased_root(&store, "second").await?;
    let settles = count_hits(SETTLED, true)?;
    let provider = ToolRounds::new(0);
    let turn = run_turn(&store, &second, "second", "second prompt", &provider);
    let crashed = crashes(SETTLED, turn).await;
    fail::remove(SETTLED);
    crashed?;
    let reached = settles.load(Ordering::SeqCst);
    ensure!(reached == 1, "settles written {reached}");
    drop(store);

    let store = reopen.await?;
    let row = AgentTaskStore::get(&store, &second)
        .await?
        .context("root")?;
    ensure!(row.status == TaskStatus::Running, "{:?}", row.status);
    let turns = ThreadStore::get(&store, &ThreadId::from_string(THREAD))
        .await?
        .map_or(0, |thread| thread.committed_turns);
    ensure!(turns == 0, "a turn landed: {turns}");
    let (committed, draft) = transcript(&store).await?;
    ensure!(committed.is_empty(), "{committed:?}");
    ensure!(
        cancelled_closures(&draft) == ["call-1"],
        "the draft holds the one closure: {draft:?}"
    );

    let provider = ToolRounds::new(0);
    let outcome = run_turn(&store, &second, "second", "second prompt", &provider).await?;
    ensure!(
        matches!(outcome, RootTurnOutcome::Completed { .. }),
        "the retry must complete"
    );
    let sent = provider.last_request()?;
    ensure!(
        cancelled_closures(&sent) == ["call-1"]
            && llm::provider_tool_sequence_error_index(&sent).is_none(),
        "the retry's request: {sent:?}"
    );
    let (committed, draft) = transcript(&store).await?;
    ensure!(draft.is_empty(), "{draft:?}");
    ensure!(
        cancelled_closures(&committed) == ["call-1"],
        "the closure commits once: {committed:?}"
    );
    Ok(())
}
