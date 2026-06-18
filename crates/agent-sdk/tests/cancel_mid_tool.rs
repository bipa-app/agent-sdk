//! Graceful cancellation mid-tool integration test.
//!
//! Locks in the SDK-side contract for the user-facing scenario:
//!
//! 1. A run starts. The LLM responds with `tool_use`. The SDK
//!    persists the assistant message and dispatches the tool.
//! 2. The user cancels mid-tool by signalling the
//!    [`CancellationToken`] tied to the run.
//! 3. **The tool itself ignores the cancel token** (this mirrors
//!    `bash` / `cargo check`, which block on a subprocess and do
//!    not observe `ToolContext::cancel_token()`).
//! 4. The SDK boundary nonetheless cancels the in-flight tool
//!    future and synthesises a successful
//!    [`ToolResult`] whose content is `"Cancelled by user"`.
//!    See `agent_loop/tool_execution.rs::run_with_cancel`.
//! 5. That synthesised result flows through the normal
//!    `append_tool_results` path — no new run-loop branches, no
//!    crash-recovery synthesis.
//! 6. The run returns [`AgentRunState::Cancelled`].
//! 7. The next run on the same `thread_id` sees a clean,
//!    internally consistent history. The Anthropic API never
//!    receives an unbalanced `tool_use` / `tool_result` pair, so
//!    the bug
//!
//!    ```text
//!    messages.N: `tool_use` ids were found without `tool_result`
//!    blocks immediately after: toolu_…
//!    ```
//!
//!    cannot reproduce on this path.
//!
//! Scope guard: this test exercises the SDK-side cancellation race
//! in `complete_sync_tool_call`. The "hard abort" path
//! (`JoinHandle::abort()`) is the same shape as a process crash
//! from the SDK's perspective and is intentionally out of scope —
//! that's the edge case of an edge case and is the territory of the
//! `recover_orphaned_tool_use` load-time backfill, which this test
//! deliberately does not exercise.

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, Usage,
};
use agent_sdk::{
    AgentInput, AgentRunState, AgentState, AllowAllHooks, CancellationToken, DynamicToolName,
    InMemoryEventStore, InMemoryStore, MessageStore, StateStore, ThreadId, Tool, ToolContext,
    ToolRegistry, ToolResult, ToolTier, builder,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use tokio::sync::{Mutex, oneshot};

/// Marker the SDK puts in the synthesised user message when the
/// orphan-recovery path (`recover_orphaned_tool_use`) runs.
/// Cooperative cancellation should never produce this string — the
/// test asserts on its absence.
const ORPHAN_RECOVERY_MARKER: &str = agent_sdk::llm::USER_CANCELLED_TOOL_RESULT;

/// LLM provider whose `chat()` returns the next pre-scripted outcome
/// and records every request it received so the test can inspect the
/// payload the SDK assembled for each turn.
struct RecordingProvider {
    responses: RwLock<Vec<ChatOutcome>>,
    requests: Arc<RwLock<Vec<ChatRequest>>>,
}

impl RecordingProvider {
    fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
            requests: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn request_handle(&self) -> Arc<RwLock<Vec<ChatRequest>>> {
        Arc::clone(&self.requests)
    }
}

fn read_requests(handle: &Arc<RwLock<Vec<ChatRequest>>>) -> Result<Vec<ChatRequest>> {
    handle
        .read()
        .ok()
        .context("requests lock poisoned")
        .map(|r| r.clone())
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        self.requests
            .write()
            .ok()
            .context("requests lock poisoned")?
            .push(request);
        let mut responses = self
            .responses
            .write()
            .ok()
            .context("responses lock poisoned")?;
        if responses.is_empty() {
            Err(anyhow!("RecordingProvider script exhausted"))
        } else {
            Ok(responses.remove(0))
        }
    }

    fn model(&self) -> &'static str {
        "test-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

/// Tool that signals "started" through `started_tx`, then parks on a
/// oneshot we never resolve. **Deliberately ignores
/// `ToolContext::cancel_token()`** to mirror the production
/// behaviour of `bash` / `cargo check`: the subprocess does not
/// observe the SDK's cancellation token, so the tool body itself
/// would block forever.
///
/// The SDK's `complete_sync_tool_call` now races the tool's future
/// against the run's cancel token at the SDK boundary (see
/// `agent_loop/tool_execution.rs::run_with_cancel`), so this test
/// asserts the SDK-side fix without any tool-level cooperation.
struct NonCooperativeTool {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    work_rx: Mutex<Option<oneshot::Receiver<()>>>,
}

impl NonCooperativeTool {
    fn new(started_tx: oneshot::Sender<()>, work_rx: oneshot::Receiver<()>) -> Self {
        Self {
            started_tx: Mutex::new(Some(started_tx)),
            work_rx: Mutex::new(Some(work_rx)),
        }
    }
}

impl Tool<()> for NonCooperativeTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("blocking_bash")
    }

    fn display_name(&self) -> &'static str {
        "Blocking Bash"
    }

    fn description(&self) -> &'static str {
        "Stub of a long-running bash invocation that does NOT observe \
         ToolContext::cancel_token() — the SDK boundary must cancel it."
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "command": { "type": "string" } } })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
        // Hand the "tool started" signal to the test exactly once.
        // Bind the awaited guard into its own variable so the
        // `MutexGuard` drops before the `if let` body runs.
        let started_tx = self.started_tx.lock().await.take();
        if let Some(sender) = started_tx {
            let _ = sender.send(());
        }
        // Park forever. We do NOT consult `ctx.cancel_token()` — the
        // SDK boundary must cancel us. If the SDK fix regresses, this
        // future just hangs and the test hits its `tokio::test`
        // default timeout.
        let work_rx = self
            .work_rx
            .lock()
            .await
            .take()
            .ok_or_else(|| anyhow!("NonCooperativeTool work receiver already taken"))?;
        let _ = work_rx.await;
        Ok(ToolResult::success("work finished (unexpected)"))
    }
}

/// Shared (clonable) in-memory store so the first and second runs
/// see the same conversation history.
#[derive(Clone, Default)]
struct SharedStore(Arc<InMemoryStore>);

impl SharedStore {
    fn new() -> Self {
        Self(Arc::new(InMemoryStore::new()))
    }
}

#[async_trait]
impl MessageStore for SharedStore {
    async fn append(&self, thread_id: &ThreadId, message: Message) -> Result<()> {
        self.0.append(thread_id, message).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<Message>> {
        self.0.get_history(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.0.clear(thread_id).await
    }

    async fn replace_history(&self, thread_id: &ThreadId, messages: Vec<Message>) -> Result<()> {
        self.0.replace_history(thread_id, messages).await
    }
}

#[async_trait]
impl StateStore for SharedStore {
    async fn save(&self, state: &AgentState) -> Result<()> {
        self.0.save(state).await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        self.0.load(thread_id).await
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        self.0.delete(thread_id).await
    }
}

fn tool_use_response(id: &str, name: &str, input: Value) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: format!("resp_{id}"),
        content: vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
            thought_signature: None,
        }],
        model: "test-model".to_string(),
        stop_reason: Some(StopReason::ToolUse),
        usage: Usage {
            input_tokens: 5,
            output_tokens: 5,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
    })
}

fn text_response(text: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: format!("resp_{text}"),
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        model: "test-model".to_string(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            input_tokens: 5,
            output_tokens: 5,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
    })
}

/// IDs of `tool_use` blocks that have no matching `tool_result`
/// block in the following user message. Empty means consistent.
fn orphan_tool_use_ids(history: &[Message]) -> Vec<String> {
    let mut orphans = Vec::new();
    for (idx, message) in history.iter().enumerate() {
        if message.role != Role::Assistant {
            continue;
        }
        let Content::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            if let ContentBlock::ToolUse { id, .. } = block {
                let satisfied = history.get(idx + 1).is_some_and(|next| {
                    let Content::Blocks(next_blocks) = &next.content else {
                        return false;
                    };
                    next_blocks.iter().any(|b| match b {
                        ContentBlock::ToolResult { tool_use_id, .. } => tool_use_id == id,
                        _ => false,
                    })
                });
                if !satisfied {
                    orphans.push(id.clone());
                }
            }
        }
    }
    orphans
}

/// Collect every `tool_result` content string in the history that
/// targets `tool_use_id`. A clean cancellation produces exactly one;
/// crash recovery would produce a second.
fn tool_results_for(history: &[Message], tool_use_id: &str) -> Vec<String> {
    history
        .iter()
        .filter_map(|m| match &m.content {
            Content::Blocks(blocks) => Some(blocks),
            Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|b| match b {
            ContentBlock::ToolResult {
                tool_use_id: id,
                content,
                ..
            } if id == tool_use_id => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// Output of [`run_first_with_cancellation`] — the final state plus
/// a handle to the provider's recorded requests so the caller can
/// inspect them later.
struct FirstRunOutcome {
    final_state: AgentRunState,
}

async fn run_first_with_cancellation(
    store: &SharedStore,
    thread_id: &ThreadId,
    tool_call_id: &str,
) -> Result<FirstRunOutcome> {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    // The work_tx half is intentionally kept alive — dropping it
    // would close the work-arm receiver and let the `select!`
    // resolve via a closed channel instead of the cancellation arm
    // we want to exercise. Naming it `keep_alive_work_tx` makes the
    // intent loud rather than borrowing the underscore convention.
    let (keep_alive_work_tx, work_rx) = oneshot::channel::<()>();
    let mut tools = ToolRegistry::new();
    tools.register(NonCooperativeTool::new(started_tx, work_rx));

    let provider = RecordingProvider::new(vec![tool_use_response(
        tool_call_id,
        "blocking_bash",
        json!({ "command": "cargo check" }),
    )]);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .build_with_stores();

    let cancel = CancellationToken::new();
    let final_state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("please run cargo check".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    // Synchronise: wait until the tool is parked on `work_rx.await`
    // before flipping the cancel token, so the SDK cancellation race
    // fires while the tool is actually in flight.
    started_rx
        .await
        .context("NonCooperativeTool never signalled start")?;
    cancel.cancel();
    // Keep the work-arm sender alive until after the run resolves so
    // the tool's `work_rx.await` cannot resolve via a closed-channel
    // signal — the SDK boundary must be what cancels it.
    let final_state = final_state_rx.await.context("agent state channel closed")?;
    drop(keep_alive_work_tx);

    Ok(FirstRunOutcome { final_state })
}

async fn run_second_after_cancel(
    store: &SharedStore,
    thread_id: &ThreadId,
) -> Result<(AgentRunState, Arc<RwLock<Vec<ChatRequest>>>)> {
    let provider = RecordingProvider::new(vec![text_response("All good")]);
    let requests_handle = provider.request_handle();

    let agent = builder::<()>()
        .provider(provider)
        .tools(ToolRegistry::new())
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .build_with_stores();

    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("never mind, what's the time?".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("second run final-state channel closed")?;
    Ok((final_state, requests_handle))
}

fn assert_no_orphan_recovery_marker(history: &[Message]) {
    for message in history {
        let Content::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            if let ContentBlock::ToolResult { content, .. } = block {
                assert!(
                    !content.contains(ORPHAN_RECOVERY_MARKER),
                    "graceful cancellation must not borrow the crash-recovery \
                     synth string; got tool_result content {content:?}",
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_cancels_non_cooperative_tool_and_persists_clean_history() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let tool_call_id = "toolu_cancel_test_1";

    // ── Run 1: SDK cancels the tool boundary → "Cancelled by user" ──
    let FirstRunOutcome { final_state } =
        run_first_with_cancellation(&store, &thread_id, tool_call_id).await?;
    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "first run must end as Cancelled; got {final_state:?}",
    );

    // ── Assertion 1: history is internally consistent right now ─────
    //
    // Critically: this is asserted **immediately after the first run
    // ends**, before any subsequent run touches the store. The SDK
    // must commit the tool_result during the cancelled run itself —
    // not on the next run as a recovery synth.
    let history_after_cancel = store.get_history(&thread_id).await?;
    let orphans_after_cancel = orphan_tool_use_ids(&history_after_cancel);
    assert!(
        orphans_after_cancel.is_empty(),
        "graceful cancellation must leave zero orphan tool_use blocks; \
         got {orphans_after_cancel:?} in history {history_after_cancel:#?}",
    );

    // ── Assertion 2: the tool_result is the cancellation marker ─────
    let results = tool_results_for(&history_after_cancel, tool_call_id);
    assert_eq!(
        results,
        vec!["Cancelled by user".to_string()],
        "the SDK must commit exactly one tool_result for the cancelled \
         tool_use, with content 'Cancelled by user'; got {results:?}",
    );

    // ── Assertion 3: orphan-recovery marker never appears ──────────
    //
    // If this assertion fires, the SDK has routed the cooperative
    // cancellation through `recover_orphaned_tool_use`, which is the
    // wrong code path — that backfill is reserved for orphaned
    // tool_use blocks recovered on the *next* run.
    assert_no_orphan_recovery_marker(&history_after_cancel);

    // ── Run 2: user sends a new message; SDK loads clean history ────
    let (final_state_2, provider_2_requests) = run_second_after_cancel(&store, &thread_id).await?;
    assert!(
        matches!(final_state_2, AgentRunState::Done { .. }),
        "second run must complete cleanly; got {final_state_2:?}",
    );

    // ── Assertion 4: still consistent and still exactly one result ──
    //
    // If the SDK silently doubled the tool_result via the recovery
    // path (e.g. because it treated the cancellation as a crash),
    // this would now hold two entries. Holding one entry proves the
    // first run already committed the result and the second run did
    // not need to synthesise anything.
    let final_history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&final_history).is_empty(),
        "second run must not introduce orphans; got history {final_history:#?}",
    );
    let results_final = tool_results_for(&final_history, tool_call_id);
    assert_eq!(
        results_final,
        vec!["Cancelled by user".to_string()],
        "the second run must NOT synthesise an extra tool_result for \
         the already-completed tool_use; got {results_final:?}",
    );

    // ── Assertion 5: outbound request is well-formed ────────────────
    //
    // Direct lock-in for the user's bug repro: the messages the SDK
    // sent to the provider for the second turn must NOT have an
    // orphan tool_use. If they do, the Anthropic API would respond
    // with `messages.N: tool_use ids were found without tool_result
    // blocks immediately after: toolu_…` — exactly the failure the
    // user reported.
    let requests = read_requests(&provider_2_requests)?;
    let final_request = requests
        .last()
        .context("second provider received no requests")?;
    let request_orphans = orphan_tool_use_ids(&final_request.messages);
    assert!(
        request_orphans.is_empty(),
        "the request the SDK assembled for the second turn must not carry \
         orphan tool_use blocks; got {request_orphans:?} in messages \
         {:#?}",
        final_request.messages,
    );

    Ok(())
}
