//! Cancellation + per-tool-timeout completeness for the long-running
//! tool kinds (Phase 10 · B, ENG-8704).
//!
//! The sync-tool cancellation contract is locked in by
//! `cancel_mid_tool.rs`. This file extends the same SDK-boundary
//! guarantees to the paths where cancellation matters most:
//!
//! 1. **Async tools** — cancelling while the async tool is parked in its
//!    `check_status` poll loop must drop the in-flight future and commit
//!    exactly one balanced `"Cancelled by user"` `tool_result`.
//! 2. **Listen tools** — cancelling while the listen tool's `execute()`
//!    is in flight must do the same.
//! 3. **Per-tool timeout** — a non-cooperative tool that exceeds
//!    `AgentConfig::tool_timeout_ms` must be stopped at the boundary and
//!    produce a balanced `"Tool timed out"` `tool_result`.
//!
//! In every case the run must end with balanced history (zero orphan
//! `tool_use` blocks) and must NOT borrow the crash-recovery synth
//! string — cancellation/timeout is not a crash.

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, Usage,
};
use agent_sdk::{
    AgentConfig, AgentInput, AgentRunState, AgentState, AllowAllHooks, AsyncTool,
    CancellationToken, DynamicToolName, InMemoryEventStore, InMemoryStore, ListenExecuteTool,
    ListenToolUpdate, MessageStore, ProgressStage, StateStore, ThreadId, Tool, ToolContext,
    ToolOutcome, ToolRegistry, ToolResult, ToolStatus, ToolTier, builder,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use tokio::sync::{Mutex, oneshot};

const CRASH_RECOVERY_MARKER: &str = "Tool execution was interrupted by a crash. Please retry.";

// ── Scripted provider ────────────────────────────────────────────────

struct RecordingProvider {
    responses: RwLock<Vec<ChatOutcome>>,
}

impl RecordingProvider {
    const fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
        }
    }
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
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

// ── Shared store ─────────────────────────────────────────────────────

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

// ── History helpers (mirrors cancel_mid_tool.rs) ─────────────────────

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

fn assert_no_crash_recovery_marker(history: &[Message]) {
    for message in history {
        let Content::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            if let ContentBlock::ToolResult { content, .. } = block {
                assert!(
                    !content.contains(CRASH_RECOVERY_MARKER),
                    "cancellation/timeout must not borrow the crash-recovery synth string; \
                     got tool_result content {content:?}",
                );
            }
        }
    }
}

// ── Test tools ───────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct NoStage;
impl ProgressStage for NoStage {}

/// Async tool that returns `InProgress` immediately, then parks in its
/// `check_status` poll loop forever. It does NOT observe the cancel
/// token — the SDK boundary must cancel it.
struct ParkingAsyncTool {
    started_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl AsyncTool<()> for ParkingAsyncTool {
    type Name = DynamicToolName;
    type Stage = NoStage;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("parking_async")
    }
    fn display_name(&self) -> &'static str {
        "Parking Async"
    }
    fn description(&self) -> &'static str {
        "Async tool that never completes its poll loop"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolOutcome> {
        Ok(ToolOutcome::in_progress("op_async_1", "started"))
    }

    fn check_status(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
    ) -> impl Stream<Item = ToolStatus<NoStage>> + Send {
        let started_tx = Arc::clone(&self.started_tx);
        futures::stream::once(async move {
            // Signal that the poll loop is live, then park forever.
            // Bind the awaited guard's `take()` into its own variable so
            // the `MutexGuard` drops before the `if let` body runs.
            let sender = started_tx.lock().await.take();
            if let Some(sender) = sender {
                let _ = sender.send(());
            }
            std::future::pending::<ToolStatus<NoStage>>().await
        })
    }
}

/// Listen tool whose `listen()` is immediately `Ready`, then whose
/// `execute()` parks forever without observing the cancel token.
struct ParkingListenTool {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
}

impl ListenExecuteTool<()> for ParkingListenTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("parking_listen")
    }
    fn display_name(&self) -> &'static str {
        "Parking Listen"
    }
    fn description(&self) -> &'static str {
        "Listen tool whose execute() never returns"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn listen(
        &self,
        _ctx: &ToolContext<()>,
        _input: Value,
    ) -> impl Stream<Item = ListenToolUpdate> + Send {
        futures::stream::iter(vec![ListenToolUpdate::Ready {
            operation_id: "op_listen_1".to_string(),
            revision: 1,
            message: "ready".to_string(),
            snapshot: json!({ "ok": true }),
            expires_at: None,
        }])
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
        _expected_revision: u64,
    ) -> Result<ToolResult> {
        let sender = self.started_tx.lock().await.take();
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
        std::future::pending::<()>().await;
        Ok(ToolResult::success("executed (unexpected)"))
    }
}

/// Sync tool that parks forever and ignores the cancel token. Used to
/// exercise the per-tool timeout boundary.
struct ParkingSyncTool;

impl Tool<()> for ParkingSyncTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("parking_sync")
    }
    fn display_name(&self) -> &'static str {
        "Parking Sync"
    }
    fn description(&self) -> &'static str {
        "Sync tool that never returns; used for timeout tests"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
        std::future::pending::<()>().await;
        Ok(ToolResult::success("done (unexpected)"))
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_cancels_async_tool_poll_loop_and_commits_one_result() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let tool_call_id = "toolu_async_cancel";

    let (started_tx, started_rx) = oneshot::channel::<()>();
    let mut tools = ToolRegistry::new();
    tools.register_async(ParkingAsyncTool {
        started_tx: Arc::new(Mutex::new(Some(started_tx))),
    });

    let provider = RecordingProvider::new(vec![tool_use_response(
        tool_call_id,
        "parking_async",
        json!({}),
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
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("kick off async".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    started_rx
        .await
        .context("async poll loop never signalled start")?;
    cancel.cancel();
    let final_state = state_rx.await.context("agent state channel closed")?;

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "run must end Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "async cancel must leave zero orphan tool_use; history {history:#?}",
    );
    assert_eq!(
        tool_results_for(&history, tool_call_id),
        vec!["Cancelled by user".to_string()],
        "exactly one 'Cancelled by user' tool_result expected",
    );
    assert_no_crash_recovery_marker(&history);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_cancels_listen_tool_execute_and_commits_one_result() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let tool_call_id = "toolu_listen_cancel";

    let (started_tx, started_rx) = oneshot::channel::<()>();
    let mut tools = ToolRegistry::new();
    tools.register_listen(ParkingListenTool {
        started_tx: Mutex::new(Some(started_tx)),
    });

    let provider = RecordingProvider::new(vec![tool_use_response(
        tool_call_id,
        "parking_listen",
        json!({}),
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
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("kick off listen".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    started_rx
        .await
        .context("listen execute never signalled start")?;
    cancel.cancel();
    let final_state = state_rx.await.context("agent state channel closed")?;

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "run must end Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "listen cancel must leave zero orphan tool_use; history {history:#?}",
    );
    assert_eq!(
        tool_results_for(&history, tool_call_id),
        vec!["Cancelled by user".to_string()],
        "exactly one 'Cancelled by user' tool_result expected",
    );
    assert_no_crash_recovery_marker(&history);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_tool_timeout_stops_non_cooperative_tool_with_balanced_history() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let tool_call_id = "toolu_timeout";

    let mut tools = ToolRegistry::new();
    tools.register(ParkingSyncTool);

    // After the parked tool is stopped by the timeout, the SDK commits a
    // balanced timeout result and loops; the second scripted response
    // ends the turn so the run completes `Done` cleanly.
    let provider = RecordingProvider::new(vec![
        tool_use_response(tool_call_id, "parking_sync", json!({})),
        text_response("done"),
    ]);

    // Tight timeout so the parked tool is stopped at the boundary.
    let config = AgentConfig {
        tool_timeout_ms: Some(50),
        ..AgentConfig::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .config(config)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .build_with_stores();

    // Cancel token is never fired: the boundary timeout must be what
    // stops the tool. Wait on the run with a generous outer timeout so a
    // regression hangs the test deterministically rather than forever.
    let final_state = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.run(
            thread_id.clone(),
            AgentInput::Text("run the parked tool".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        ),
    )
    .await
    .context("run did not resolve within 5s — timeout boundary did not fire")?
    .context("agent state channel closed")?;

    // The run was not cancelled: the boundary timeout produced a balanced
    // tool_result, the loop continued, and the follow-up turn ended the
    // run cleanly.
    assert!(
        matches!(final_state, AgentRunState::Done { .. }),
        "run must complete Done after the tool timed out; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "timeout must leave zero orphan tool_use; history {history:#?}",
    );
    assert_eq!(
        tool_results_for(&history, tool_call_id),
        vec!["Tool timed out".to_string()],
        "exactly one 'Tool timed out' tool_result expected",
    );
    assert_no_crash_recovery_marker(&history);
    Ok(())
}
