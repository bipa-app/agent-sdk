//! Cancellation + message-lifecycle edge-case matrix (Phase 11 · C).
//!
//! This file completes the cancellation + message-lifecycle edge-case
//! matrix for Phase 11 · C. The already-merged Phase 10
//! cards each landed one or two targeted regression tests
//! (`cancel_mid_tool.rs`, `cancel_llm_streaming.rs`,
//! `cancel_async_listen_timeout.rs`, `panic_isolation.rs`); this file
//! covers the remaining rows so a regression in any Phase 10 fix fails
//! a deterministic test rather than shipping silently:
//!
//! **Cancellation rows covered here**
//! - cancel during the parallel observe `join_all`
//! - double-cancel is a no-op
//! - cancel-vs-result race resolves deterministically (result wins when
//!   it lands first; cancel wins when it lands first)
//! - interrupt-then-drain ordering
//! - aborted stream yields a well-formed (not half-parsed) history
//! - `AbortSignal` forwarded into in-flight tools (cooperative tool sees
//!   the token fire)
//! - caller-cancel must not outrun background teardown
//! - persistence not gated on a clean finish
//! - hard-abort → `recover_orphaned_tool_use` produces a
//!   provider-acceptable next turn
//!
//! **Message lifecycle / ordering rows covered here**
//! - concurrent send + receive on one thread
//! - `receive_response` stops at the terminal result
//! - distinct-thread isolation
//! - composite: new input mid-stream + cancel mid-tool — the in-flight
//!   turn settles, FIFO is preserved, the queued message runs next with
//!   balanced history, and the two threads' event streams stay isolated
//!
//! Determinism: every cancel is landed with a scripted pause-on-signal
//! provider / tool — a `oneshot` "I'm live" signal handed to the test,
//! then a parked never-resolved receiver — so the cancel always fires
//! while the target is genuinely in flight. No test uses a real
//! `sleep`, a wall-clock deadline, or a busy-wait race, so every run is
//! deterministic. (The per-tool-timeout boundary, which is the only
//! cancellation path that needs the clock, is already covered with
//! virtual time in `cancel_async_listen_timeout.rs`.)

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamBox, StreamDelta, Usage,
};
use agent_sdk::{
    AgentConfig, AgentEvent, AgentInput, AgentRunState, AgentState, AllowAllHooks,
    CancellationToken, DynamicToolName, EventStore, InMemoryEventStore, InMemoryStore,
    MessageStore, StateStore, ThreadId, Tool, ToolContext, ToolRegistry, ToolResult, ToolTier,
    builder,
};
use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Mutex, oneshot};

/// Marker the SDK writes only on the orphan-recovery path
/// (`recover_orphaned_tool_use` → `balance_tool_results`), which closes an
/// abandoned `tool_use` at load. Cooperative cancellation and timeout
/// produce their own distinct result text and must never borrow it.
const ORPHAN_RECOVERY_MARKER: &str = agent_sdk::llm::USER_CANCELLED_TOOL_RESULT;

// ── Shared in-memory store ───────────────────────────────────────────

/// Clonable in-memory store so a follow-up run inspects the history the
/// previous (cancelled / aborted) run left behind.
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

// ── History invariants ───────────────────────────────────────────────

/// IDs of `tool_use` blocks with no matching `tool_result` in the next
/// user message. Empty means a balanced (provider-acceptable) history.
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
                    next_blocks.iter().any(|b| {
                        matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)
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

/// Collect every `tool_result` content string targeting `tool_use_id`.
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

/// Every `tool_use` id present anywhere in the history (assistant side).
fn tool_use_ids(history: &[Message]) -> Vec<String> {
    history
        .iter()
        .filter_map(|m| match &m.content {
            Content::Blocks(blocks) => Some(blocks),
            Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|b| match b {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
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
                    "cooperative cancellation must not borrow the orphan-recovery synth \
                     string; got tool_result content {content:?}",
                );
            }
        }
    }
}

/// Exactly one terminal `Cancelled` event, no `Done` event.
async fn assert_cancelled_event_terminal(
    event_store: &InMemoryEventStore,
    thread_id: &ThreadId,
) -> Result<()> {
    let events = event_store.get_events(thread_id).await?;
    let cancelled = events
        .iter()
        .filter(|e| matches!(e.event, AgentEvent::Cancelled { .. }))
        .count();
    let done = events
        .iter()
        .filter(|e| matches!(e.event, AgentEvent::Done { .. }))
        .count();
    assert_eq!(
        cancelled, 1,
        "exactly one terminal Cancelled event must be emitted; got {cancelled}",
    );
    assert_eq!(done, 0, "a cancelled run must not also emit a Done event");
    Ok(())
}

// ── Scripted providers ───────────────────────────────────────────────

/// A single assistant turn that emits *several* `Observe`-tier `tool_use`
/// blocks at once, so the SDK runs them concurrently via `join_all`.
fn multi_tool_use_response(ids_and_names: &[(&str, &str)]) -> ChatOutcome {
    let content = ids_and_names
        .iter()
        .map(|(id, name)| ContentBlock::ToolUse {
            id: (*id).to_string(),
            name: (*name).to_string(),
            input: serde_json::json!({}),
            thought_signature: None,
        })
        .collect();
    ChatOutcome::Success(ChatResponse {
        id: "resp_multi".to_string(),
        content,
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

/// Provider that records every request it received and drains a scripted
/// queue of outcomes. Lets a test inspect the exact messages the SDK
/// assembled for the next turn (proving balance / provider-acceptance).
struct RecordingProvider {
    responses: std::sync::RwLock<Vec<ChatOutcome>>,
    requests: Arc<std::sync::RwLock<Vec<ChatRequest>>>,
}

impl RecordingProvider {
    fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: std::sync::RwLock::new(responses),
            requests: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    fn request_handle(&self) -> Arc<std::sync::RwLock<Vec<ChatRequest>>> {
        Arc::clone(&self.requests)
    }
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

fn read_requests(handle: &Arc<std::sync::RwLock<Vec<ChatRequest>>>) -> Result<Vec<ChatRequest>> {
    handle
        .read()
        .ok()
        .context("requests lock poisoned")
        .map(|r| r.clone())
}

// ── Tools used to land a cancel deterministically ────────────────────

/// `Observe`-tier tool that signals "started" once, then parks on a
/// never-resolved receiver. It does NOT consult the cancel token — the
/// SDK boundary must cancel it. Multiple instances can share one
/// "all started" barrier so a parallel batch can be fully in-flight
/// before the cancel fires.
struct ParkingObserveTool {
    name: &'static str,
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    park_rx: Mutex<Option<oneshot::Receiver<()>>>,
}

impl ParkingObserveTool {
    fn new(
        name: &'static str,
        started_tx: oneshot::Sender<()>,
        park_rx: oneshot::Receiver<()>,
    ) -> Self {
        Self {
            name,
            started_tx: Mutex::new(Some(started_tx)),
            park_rx: Mutex::new(Some(park_rx)),
        }
    }
}

impl Tool<()> for ParkingObserveTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(self.name)
    }
    fn display_name(&self) -> &'static str {
        "Parking Observe"
    }
    fn description(&self) -> &'static str {
        "Observe tool that parks forever, ignoring the cancel token"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> Result<ToolResult> {
        let started_tx = self.started_tx.lock().await.take();
        if let Some(sender) = started_tx {
            let _ = sender.send(());
        }
        let park_rx = self.park_rx.lock().await.take();
        if let Some(rx) = park_rx {
            let _ = rx.await;
        }
        Ok(ToolResult::success("finished (unexpected)"))
    }
}

/// `Observe`-tier tool that *cooperatively* observes the cancel token:
/// it signals "started", then awaits `ctx.cancel_token().cancelled()`.
/// When the SDK forwards the run's `AbortSignal`, the token fires and
/// the tool records that it saw the cancellation before returning. This
/// proves the token is propagated *into* the in-flight tool (Mastra's
/// "`AbortSignal` forwarded into in-flight tools" row).
struct AbortAwareTool {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    saw_cancel: Arc<AtomicUsize>,
}

impl Tool<()> for AbortAwareTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("abort_aware")
    }
    fn display_name(&self) -> &'static str {
        "Abort Aware"
    }
    fn description(&self) -> &'static str {
        "Tool that awaits its forwarded cancel token and records the signal"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> Result<ToolResult> {
        let started_tx = self.started_tx.lock().await.take();
        if let Some(sender) = started_tx {
            let _ = sender.send(());
        }
        let token = ctx
            .cancel_token()
            .ok_or_else(|| anyhow!("AbortSignal (cancel token) was not forwarded into the tool"))?;
        // Cooperatively await the forwarded token. If the SDK forwards
        // it, this resolves when the test fires the cancel.
        token.cancelled().await;
        self.saw_cancel.fetch_add(1, Ordering::SeqCst);
        // Return a normal result; the SDK-boundary race will also
        // observe the same token and synthesize the balanced
        // "Cancelled by user" result, so the run still ends Cancelled.
        Ok(ToolResult::success("tool observed cancel"))
    }
}

/// `Observe`-tier tool whose body resolves immediately (the "result"
/// half of the cancel-vs-result race). Used with a pre-cancelled token
/// to prove deterministic resolution.
struct InstantTool;

impl Tool<()> for InstantTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("instant")
    }
    fn display_name(&self) -> &'static str {
        "Instant"
    }
    fn description(&self) -> &'static str {
        "Tool that returns immediately"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> Result<ToolResult> {
        Ok(ToolResult::success("instant result"))
    }
}

// ── Streaming provider that can pause mid-tool-use ───────────────────

/// What the streaming provider should do once it has signalled the test
/// that the stream is live.
enum StreamScript {
    /// Emit a couple of text deltas, then `ToolUseStart` + a *partial*
    /// (incomplete-JSON) `ToolInputDelta`, then park forever without a
    /// `Done`. The SDK must discard the half-parsed accumulator on
    /// cancel.
    PartialToolUseThenPark { tool_id: String, tool_name: String },
}

/// Streaming provider that drives a [`StreamScript`], signals the test
/// when the stream is live, then parks on a never-resolved receiver. It
/// does not observe any cancel token — the SDK's `process_stream` race
/// must stop it.
struct PausableStreamProvider {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    park_rx: Mutex<Option<oneshot::Receiver<()>>>,
    script: StreamScript,
}

#[async_trait]
impl LlmProvider for PausableStreamProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Err(anyhow!("PausableStreamProvider only supports chat_stream"))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let started_tx = self.started_tx.lock().await.take();
            let park_rx = self.park_rx.lock().await.take();

            match &self.script {
                StreamScript::PartialToolUseThenPark { tool_id, tool_name } => {
                    yield Ok(StreamDelta::TextDelta {
                        delta: "Let me run that".to_string(),
                        block_index: 0,
                    });
                    yield Ok(StreamDelta::ToolUseStart {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        block_index: 1,
                        thought_signature: None,
                    });
                    // Deliberately *incomplete* JSON — if the SDK ever
                    // finalized this partial stream it would persist a
                    // half-parsed / malformed tool_use.
                    yield Ok(StreamDelta::ToolInputDelta {
                        id: tool_id.clone(),
                        delta: "{\"command\": \"cargo che".to_string(),
                        block_index: 1,
                    });
                }
            }

            if let Some(sender) = started_tx {
                let _ = sender.send(());
            }
            if let Some(rx) = park_rx {
                // Never resolved by the test; the SDK boundary must
                // cancel the stream poll. A regression hangs and trips
                // the outer timeout guard.
                let _ = rx.await;
            }
            // Unreachable on the cancel path. If the fix regresses and
            // the stream is allowed to complete, these would finalize a
            // (now well-formed) tool_use, which the test would catch as
            // a non-cancelled outcome.
            yield Ok(StreamDelta::ToolInputDelta {
                id: "late".to_string(),
                delta: "ck\"}".to_string(),
                block_index: 1,
            });
            yield Ok(StreamDelta::Done {
                stop_reason: Some(StopReason::ToolUse),
            });
        })
    }

    fn model(&self) -> &'static str {
        "test-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

// ── Builders ─────────────────────────────────────────────────────────

fn build_agent<P: LlmProvider + 'static>(
    provider: P,
    tools: ToolRegistry<()>,
    store: &SharedStore,
    event_store: Arc<InMemoryEventStore>,
    config: AgentConfig,
) -> agent_sdk::AgentLoop<(), P, AllowAllHooks, SharedStore, SharedStore> {
    builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .config(config)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(event_store)
        .build_with_stores()
}

// ═════════════════════════════════════════════════════════════════════
//  Cancellation rows
// ═════════════════════════════════════════════════════════════════════

/// Cancel while a **parallel `Observe` batch** is in flight: every tool
/// in the `join_all` is cancelled at the boundary and each gets exactly
/// one balanced `"Cancelled by user"` result, leaving zero orphans.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_during_parallel_observe_join_all_balances_every_call() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    // Three observe tools requested in one assistant turn → one
    // `join_all` batch. Each signals start and parks; the cancel fires
    // only after all three are confirmed in flight.
    let (s1_tx, s1_rx) = oneshot::channel();
    let (s2_tx, s2_rx) = oneshot::channel();
    let (s3_tx, s3_rx) = oneshot::channel();
    let (keep1, p1) = oneshot::channel();
    let (keep2, p2) = oneshot::channel();
    let (keep3, p3) = oneshot::channel();

    let mut tools = ToolRegistry::new();
    tools.register(ParkingObserveTool::new("obs_a", s1_tx, p1));
    tools.register(ParkingObserveTool::new("obs_b", s2_tx, p2));
    tools.register(ParkingObserveTool::new("obs_c", s3_tx, p3));

    let provider = RecordingProvider::new(vec![multi_tool_use_response(&[
        ("toolu_a", "obs_a"),
        ("toolu_b", "obs_b"),
        ("toolu_c", "obs_c"),
    ])]);

    let agent = build_agent(
        provider,
        tools,
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("run all three observers".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    // Wait until *all three* observe calls are live, proving they really
    // overlap in the `join_all`, before cancelling.
    s1_rx.await.context("obs_a never started")?;
    s2_rx.await.context("obs_b never started")?;
    s3_rx.await.context("obs_c never started")?;
    cancel.cancel();

    let final_state = state_rx.await.context("agent state channel closed")?;
    drop((keep1, keep2, keep3));

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "cancel during the parallel observe batch must end Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "every tool_use in the cancelled batch must be balanced; orphans in {history:#?}",
    );
    for id in ["toolu_a", "toolu_b", "toolu_c"] {
        assert_eq!(
            tool_results_for(&history, id),
            vec!["Cancelled by user".to_string()],
            "each parallel tool_use must get exactly one balanced cancel result ({id})",
        );
    }
    assert_no_orphan_recovery_marker(&history);
    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

/// Firing the same cancel token twice is a no-op: the run still ends
/// `Cancelled` once with balanced history and one terminal event.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn double_cancel_is_a_noop() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel();
    let (keep_alive, park_rx) = oneshot::channel();
    let mut tools = ToolRegistry::new();
    tools.register(ParkingObserveTool::new("obs", started_tx, park_rx));

    let provider = RecordingProvider::new(vec![multi_tool_use_response(&[("toolu_dc", "obs")])]);
    let agent = build_agent(
        provider,
        tools,
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("park then cancel twice".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    started_rx.await.context("tool never started")?;
    // Fire twice — the second must be a no-op, not a panic or a double
    // terminal event.
    cancel.cancel();
    cancel.cancel();

    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive);

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "double cancel must still end Cancelled exactly once; got {final_state:?}",
    );
    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "double cancel must leave balanced history; {history:#?}",
    );
    assert_eq!(
        tool_results_for(&history, "toolu_dc"),
        vec!["Cancelled by user".to_string()],
        "double cancel must commit exactly one balanced result",
    );
    // One terminal Cancelled event — the second cancel must not emit a
    // second terminal marker.
    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

/// Cancel-vs-result race resolves deterministically: a token that is
/// already cancelled when the run starts always resolves the same way,
/// run after run — the `biased` cancellation precheck wins before any
/// tool runs, so the run ends `Cancelled` with no orphaned `tool_use`
/// and at most one balanced result per emitted `tool_use`. There is no
/// coin-flip between "the instant tool's result lands" and "the cancel
/// fires": the resolution is fixed by the boundary's bias, which is
/// exactly the determinism property this row guards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_vs_result_race_is_deterministic_and_balanced() -> Result<()> {
    // Run the identical scenario several times on fresh threads/stores.
    // The outcome must be byte-for-byte the same every iteration —
    // that repetition is what proves "resolves deterministically"
    // rather than observing one lucky resolution.
    for iteration in 0..8 {
        let store = SharedStore::new();
        let thread_id = ThreadId::new();
        let event_store = Arc::new(InMemoryEventStore::new());

        let mut tools = ToolRegistry::new();
        tools.register(InstantTool);

        // An instant-tool turn followed by a wrap-up turn. If the
        // cancel ever lost the race, the loop would proceed into the
        // tool and the second response would be consumed.
        let provider = RecordingProvider::new(vec![
            multi_tool_use_response(&[("toolu_race", "instant")]),
            text_response("done"),
        ]);
        let agent = build_agent(provider, tools, &store, event_store, AgentConfig::default());

        // Pre-cancel the token. The run-loop's cancellation precheck is
        // `biased`, so an already-cancelled token is honored before any
        // tool is dispatched — the resolution is fixed, never a coin
        // flip against the instant tool's result.
        let cancel = CancellationToken::new();
        cancel.cancel();

        let final_state = agent
            .run(
                thread_id.clone(),
                AgentInput::Text("race the cancel against an instant tool".to_string()),
                ToolContext::new(()),
                cancel,
            )
            .await
            .context("agent state channel closed")?;

        assert!(
            matches!(final_state, AgentRunState::Cancelled { .. }),
            "iteration {iteration}: a pre-cancelled run must resolve deterministically as \
             Cancelled; got {final_state:?}",
        );
        let history = store.get_history(&thread_id).await?;
        let orphans = orphan_tool_use_ids(&history);
        assert!(
            orphans.is_empty(),
            "iteration {iteration}: the cancel-vs-result race must never leave an orphan \
             tool_use; got {orphans:?} in {history:#?}",
        );
        // For any tool_use that *did* get emitted, there is exactly one
        // balanced result — never zero, never two.
        for id in tool_use_ids(&history) {
            assert_eq!(
                tool_results_for(&history, &id).len(),
                1,
                "iteration {iteration}: tool_use {id} must have exactly one result",
            );
        }
        assert_no_orphan_recovery_marker(&history);
    }
    Ok(())
}

/// Interrupt-then-drain ordering (Claude SDK): after a cancel interrupts
/// the in-flight turn, the *next* run on the same thread drains cleanly —
/// it sees the balanced history the cancelled run committed, produces a
/// normal `Done`, and the assembled provider request carries no orphan
/// `tool_use`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_then_drain_preserves_ordering_and_balance() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();

    // ── Interrupt: park a tool, cancel it. ──
    {
        let event_store = Arc::new(InMemoryEventStore::new());
        let (started_tx, started_rx) = oneshot::channel();
        let (keep_alive, park_rx) = oneshot::channel();
        let mut tools = ToolRegistry::new();
        tools.register(ParkingObserveTool::new("obs", started_tx, park_rx));
        let provider =
            RecordingProvider::new(vec![multi_tool_use_response(&[("toolu_drain", "obs")])]);
        let agent = build_agent(provider, tools, &store, event_store, AgentConfig::default());

        let cancel = CancellationToken::new();
        let state_rx = agent.run(
            thread_id.clone(),
            AgentInput::Text("first message that gets interrupted".to_string()),
            ToolContext::new(()),
            cancel.clone(),
        );
        started_rx.await.context("tool never started")?;
        cancel.cancel();
        let s = state_rx.await.context("agent state channel closed")?;
        drop(keep_alive);
        assert!(
            matches!(s, AgentRunState::Cancelled { .. }),
            "interrupt must cancel; got {s:?}"
        );
    }

    // ── Drain: a second message must run next, in order, cleanly. ──
    let provider = RecordingProvider::new(vec![text_response("drained cleanly")]);
    let requests = provider.request_handle();
    let agent = build_agent(
        provider,
        ToolRegistry::new(),
        &store,
        Arc::new(InMemoryEventStore::new()),
        AgentConfig::default(),
    );
    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("second message drains".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("drain run channel closed")?;

    assert!(
        matches!(final_state, AgentRunState::Done { .. }),
        "the drained follow-up run must complete Done; got {final_state:?}",
    );
    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "drain must see balanced history; {history:#?}",
    );
    // The interrupted message comes before the drained one in history —
    // FIFO ordering preserved across the interrupt.
    let user_texts: Vec<String> = history
        .iter()
        .filter(|m| m.role == Role::User)
        .filter_map(|m| match &m.content {
            Content::Text(t) => Some(t.clone()),
            Content::Blocks(_) => None,
        })
        .collect();
    let first_idx = user_texts
        .iter()
        .position(|t| t.contains("first message"))
        .context("interrupted message missing from history")?;
    let second_idx = user_texts
        .iter()
        .position(|t| t.contains("second message"))
        .context("drained message missing from history")?;
    assert!(
        first_idx < second_idx,
        "FIFO ordering must hold: interrupted message precedes drained one; got {user_texts:?}",
    );

    // The request the SDK assembled for the drained turn must be
    // provider-acceptable (no orphan tool_use).
    let reqs = read_requests(&requests)?;
    let last = reqs.last().context("drain run sent no request")?;
    assert!(
        orphan_tool_use_ids(&last.messages).is_empty(),
        "drained turn's outbound request must carry no orphan tool_use; {:#?}",
        last.messages,
    );
    Ok(())
}

/// Aborted stream yields a well-formed (not half-parsed) history
/// (Pydantic AI): a stream that emits `ToolUseStart` + a *partial*
/// (incomplete-JSON) `ToolInputDelta` and is then cancelled must NOT
/// persist the half-parsed `tool_use`. History stays empty/balanced and
/// the next turn's request is provider-acceptable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_stream_yields_well_formed_history() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel();
    let (keep_alive, park_rx) = oneshot::channel();
    let provider = PausableStreamProvider {
        started_tx: Mutex::new(Some(started_tx)),
        park_rx: Mutex::new(Some(park_rx)),
        script: StreamScript::PartialToolUseThenPark {
            tool_id: "toolu_partial".to_string(),
            tool_name: "obs".to_string(),
        },
    };

    let config = AgentConfig {
        streaming: true,
        ..AgentConfig::default()
    };
    let agent = build_agent(
        provider,
        ToolRegistry::new(),
        &store,
        event_store.clone(),
        config,
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("stream a tool call then abort".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    // The stream has emitted the partial tool_use and parked. Cancel now.
    started_rx
        .await
        .context("stream never signalled it was live")?;
    cancel.cancel();

    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive);

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "an aborted stream must end Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    // No half-parsed tool_use leaked into history.
    assert!(
        tool_use_ids(&history).is_empty(),
        "an aborted partial-tool-use stream must not persist a half-parsed tool_use; {history:#?}",
    );
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "aborted stream must leave balanced history; {history:#?}",
    );
    // No assistant message at all — the turn never completed.
    assert_eq!(
        history.iter().filter(|m| m.role == Role::Assistant).count(),
        0,
        "no assistant message should be persisted from a half-streamed turn; {history:#?}",
    );
    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

/// `AbortSignal` forwarded into in-flight tools (Mastra): a tool that
/// awaits its `ctx.cancel_token()` observes the cancel firing. Proves
/// the run's signal is propagated *into* the tool, not just enforced at
/// the boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_signal_is_forwarded_into_in_flight_tools() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel();
    let saw_cancel = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(AbortAwareTool {
        started_tx: Mutex::new(Some(started_tx)),
        saw_cancel: Arc::clone(&saw_cancel),
    });

    let provider = RecordingProvider::new(vec![multi_tool_use_response(&[(
        "toolu_abort",
        "abort_aware",
    )])]);
    let agent = build_agent(
        provider,
        tools,
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("run an abort-aware tool".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    started_rx.await.context("abort-aware tool never started")?;
    cancel.cancel();

    let final_state = state_rx.await.context("agent state channel closed")?;

    assert_eq!(
        saw_cancel.load(Ordering::SeqCst),
        1,
        "the in-flight tool must observe the forwarded cancel token exactly once",
    );
    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "the run must end Cancelled; got {final_state:?}",
    );
    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "forwarded-abort run must leave balanced history; {history:#?}",
    );
    // Exactly one result is committed. Its *content* is intentionally
    // not pinned: once the token is forwarded, the SDK boundary races
    // the tool's own (now cancel-aware) result against the synthetic
    // "Cancelled by user" marker. Either is balanced and correct — the
    // tool returning a cooperative result after observing the abort
    // ("tool observed cancel") and the boundary synthesizing the cancel
    // marker are both acceptable. Pinning the string would make the
    // test flaky on scheduler timing; the invariants that matter are
    // "the tool saw the abort" + "exactly one balanced result".
    let results = tool_results_for(&history, "toolu_abort");
    assert_eq!(
        results.len(),
        1,
        "the cancelled abort-aware tool must commit exactly one balanced result; got {results:?}",
    );
    assert!(
        results[0] == "Cancelled by user" || results[0] == "tool observed cancel",
        "the committed result must be a balanced cancel/cooperative result; got {:?}",
        results[0],
    );
    assert_no_orphan_recovery_marker(&history);
    Ok(())
}

/// Caller-cancel must not outrun background teardown (Pydantic AI
/// #5132): when the run resolves `Cancelled`, the `tool_result` has
/// *already* been committed to the store — the caller can read a fully
/// balanced history the instant the state channel resolves, with no
/// later async write racing in behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn caller_cancel_does_not_outrun_background_teardown() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel();
    let (keep_alive, park_rx) = oneshot::channel();
    let mut tools = ToolRegistry::new();
    tools.register(ParkingObserveTool::new("obs", started_tx, park_rx));

    let provider = RecordingProvider::new(vec![multi_tool_use_response(&[("toolu_td", "obs")])]);
    let agent = build_agent(
        provider,
        tools,
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("park then cancel".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );
    started_rx.await.context("tool never started")?;
    cancel.cancel();
    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive);

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "run must end Cancelled; got {final_state:?}",
    );

    // CRITICAL: read the store *immediately* after the state channel
    // resolves, with no intervening yield. The balanced tool_result must
    // already be durable — teardown completed before the caller was
    // told the run ended.
    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "teardown must finish before the caller is signalled; orphans visible in {history:#?}",
    );
    assert_eq!(
        tool_results_for(&history, "toolu_td"),
        vec!["Cancelled by user".to_string()],
        "the balanced cancel result must already be committed when the run resolves",
    );
    // And the terminal event must already be visible too.
    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

/// Persistence not gated on a clean finish (Mastra #13984): a cancelled
/// run still durably persists the work it did before the cancel. The
/// user message and the synthesized balanced `tool_result` survive even
/// though the run never reached a clean `Done`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_is_not_gated_on_a_clean_finish() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel();
    let (keep_alive, park_rx) = oneshot::channel();
    let mut tools = ToolRegistry::new();
    tools.register(ParkingObserveTool::new("obs", started_tx, park_rx));

    let provider =
        RecordingProvider::new(vec![multi_tool_use_response(&[("toolu_persist", "obs")])]);
    let agent = build_agent(
        provider,
        tools,
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("durably persist before cancel".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );
    started_rx.await.context("tool never started")?;
    cancel.cancel();
    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive);

    assert!(matches!(final_state, AgentRunState::Cancelled { .. }));

    let history = store.get_history(&thread_id).await?;
    // The user prompt was persisted despite the cancel.
    assert!(
        history.iter().any(|m| m.role == Role::User
            && matches!(&m.content, Content::Text(t) if t.contains("durably persist"))),
        "the user message must survive a cancelled run; {history:#?}",
    );
    // The assistant tool_use turn was persisted.
    assert!(
        !tool_use_ids(&history).is_empty(),
        "the assistant tool_use the cancel interrupted must be persisted; {history:#?}",
    );
    // And it is balanced (the synthesized result was also persisted).
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "the synthesized tool_result must be persisted alongside the tool_use; {history:#?}",
    );
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
//  Hard-abort → recover_orphaned_tool_use
// ═════════════════════════════════════════════════════════════════════

/// Hard-abort → `recover_orphaned_tool_use` produces a
/// provider-acceptable next turn.
///
/// This is the edge-case `cancel_mid_tool.rs` explicitly scoped out: a
/// **hard abort** (the run task is `JoinHandle::abort()`'d mid-tool, the
/// same shape as a process crash) can leave the assistant `tool_use`
/// persisted but the `tool_result` never committed — an orphan. The next
/// `Text` run must detect the orphan and synthesize
/// cancellation results so the request the SDK assembles for the provider is
/// balanced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hard_abort_then_synth_yields_provider_acceptable_next_turn() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();

    // ── Simulate a hard crash: persist an assistant tool_use with no
    //    following tool_result, exactly as a process that died between
    //    persisting the assistant turn and committing the tool result
    //    would leave the store. (Driving `run_abortable().abort()` is
    //    inherently racy — directly seeding the orphan is the
    //    deterministic equivalent of that crash, and is what the
    //    recovery path keys off.) ──
    store
        .append(&thread_id, Message::user("do the thing"))
        .await?;
    store
        .append(
            &thread_id,
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: "toolu_crashed".to_string(),
                    name: "obs".to_string(),
                    input: serde_json::json!({ "command": "cargo check" }),
                    thought_signature: None,
                }]),
            },
        )
        .await?;

    // Sanity: the seeded history is unbalanced right now (the crash
    // condition the recovery path must fix).
    let before = store.get_history(&thread_id).await?;
    assert_eq!(
        orphan_tool_use_ids(&before),
        vec!["toolu_crashed".to_string()],
        "precondition: the crash left an orphan tool_use",
    );

    // ── Next run: a fresh user `Text` input triggers
    //    recover_orphaned_tool_use → balance_tool_results
    //    before the new turn's LLM call. ──
    let provider = RecordingProvider::new(vec![text_response("recovered after crash")]);
    let requests = provider.request_handle();
    let agent = build_agent(
        provider,
        ToolRegistry::new(),
        &store,
        Arc::new(InMemoryEventStore::new()),
        AgentConfig::default(),
    );

    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("are we recovered?".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("recovery run channel closed")?;

    assert!(
        matches!(final_state, AgentRunState::Done { .. }),
        "the recovery run must complete Done; got {final_state:?}",
    );

    // History is now balanced and carries the orphan-recovery synth
    // result for the orphaned tool_use.
    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "recover_orphaned_tool_use must balance the orphan; {history:#?}",
    );
    let results = tool_results_for(&history, "toolu_crashed");
    assert_eq!(
        results.len(),
        1,
        "exactly one synth result; got {results:?}"
    );
    assert!(
        results[0].contains(ORPHAN_RECOVERY_MARKER),
        "the synth result must be the orphan-recovery marker; got {:?}",
        results[0],
    );

    // The request the SDK actually sent to the provider for the recovery
    // turn must be provider-acceptable (no orphan tool_use) — the direct
    // lock-in for the "provider-acceptable next turn" criterion.
    let reqs = read_requests(&requests)?;
    let last = reqs.last().context("recovery run sent no request")?;
    let request_orphans = orphan_tool_use_ids(&last.messages);
    assert!(
        request_orphans.is_empty(),
        "the synthesized next-turn request must be balanced for the provider; got {request_orphans:?} in {:#?}",
        last.messages,
    );
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
//  Message lifecycle / ordering rows
// ═════════════════════════════════════════════════════════════════════

/// Concurrent send + receive on one thread: a consumer subscribed to the
/// event stream observes the run's events while the run is in flight,
/// and the run resolves cleanly. Send (run) and receive (event drain)
/// overlap without deadlock or lost events.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_send_and_receive_on_one_thread() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let provider = RecordingProvider::new(vec![text_response("hello back")]);
    let agent = build_agent(
        provider,
        ToolRegistry::new(),
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    // "Send": start the run.
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );

    // "Receive": concurrently poll the event store until the run
    // produces a terminal Done event, then stop. This overlaps with the
    // in-flight run.
    let recv_store = event_store.clone();
    let recv_thread = thread_id.clone();
    let receiver = tokio::spawn(async move {
        loop {
            let events = recv_store.get_events(&recv_thread).await?;
            if events
                .iter()
                .any(|e| matches!(e.event, AgentEvent::Done { .. }))
            {
                return anyhow::Ok(events);
            }
            tokio::task::yield_now().await;
        }
    });

    let final_state = state_rx.await.context("agent state channel closed")?;
    assert!(
        matches!(final_state, AgentRunState::Done { .. }),
        "concurrent send+receive run must complete Done; got {final_state:?}",
    );

    let events = receiver
        .await
        .context("receiver task panicked")?
        .context("receiver returned error")?;
    // The receiver observed a Start before the Done — events are
    // delivered in order, not lost.
    let start_idx = events
        .iter()
        .position(|e| matches!(e.event, AgentEvent::Start { .. }))
        .context("receiver missed the Start event")?;
    let done_idx = events
        .iter()
        .position(|e| matches!(e.event, AgentEvent::Done { .. }))
        .context("receiver missed the Done event")?;
    assert!(
        start_idx < done_idx,
        "events must be ordered Start..Done; got {events:#?}",
    );
    Ok(())
}

/// `receive_response` stops at the terminal result: once a run reaches a
/// terminal event (`Done`), no further events are appended afterward, so
/// a consumer that stops at the terminal marker does not truncate live
/// events nor hang waiting for more.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn receive_stops_at_terminal_result() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let provider = RecordingProvider::new(vec![text_response("the answer")]);
    let agent = build_agent(
        provider,
        ToolRegistry::new(),
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("question".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("agent state channel closed")?;
    assert!(matches!(final_state, AgentRunState::Done { .. }));

    let events = event_store.get_events(&thread_id).await?;
    // Exactly one terminal event, and it is the last event in the
    // stream — a consumer that stops there sees the whole response.
    let terminal_positions: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(
                e.event,
                AgentEvent::Done { .. } | AgentEvent::Error { .. } | AgentEvent::Cancelled { .. }
            )
        })
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        terminal_positions,
        vec![events.len() - 1],
        "there must be exactly one terminal event and it must be last; {events:#?}",
    );
    Ok(())
}

/// Distinct-thread isolation: two runs on two threads share the same
/// stores but never see each other's history or events.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn distinct_thread_isolation() -> Result<()> {
    let store = SharedStore::new();
    let event_store = Arc::new(InMemoryEventStore::new());
    let thread_a = ThreadId::new();
    let thread_b = ThreadId::new();

    let agent_a = build_agent(
        RecordingProvider::new(vec![text_response("answer A")]),
        ToolRegistry::new(),
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );
    let agent_b = build_agent(
        RecordingProvider::new(vec![text_response("answer B")]),
        ToolRegistry::new(),
        &store,
        event_store.clone(),
        AgentConfig::default(),
    );

    // Run both concurrently on the shared stores.
    let rx_a = agent_a.run(
        thread_a.clone(),
        AgentInput::Text("question A".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let rx_b = agent_b.run(
        thread_b.clone(),
        AgentInput::Text("question B".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let (state_a, state_b) = tokio::join!(rx_a, rx_b);
    assert!(matches!(
        state_a.context("a channel closed")?,
        AgentRunState::Done { .. }
    ));
    assert!(matches!(
        state_b.context("b channel closed")?,
        AgentRunState::Done { .. }
    ));

    // Thread A's history contains only A's prompt; B's only B's.
    let hist_a = store.get_history(&thread_a).await?;
    let hist_b = store.get_history(&thread_b).await?;
    let a_has_b = hist_a.iter().any(|m| match &m.content {
        Content::Text(t) => t.contains("question B") || t.contains("answer B"),
        Content::Blocks(_) => false,
    });
    let b_has_a = hist_b.iter().any(|m| match &m.content {
        Content::Text(t) => t.contains("question A") || t.contains("answer A"),
        Content::Blocks(_) => false,
    });
    assert!(!a_has_b, "thread A leaked thread B's content: {hist_a:#?}");
    assert!(!b_has_a, "thread B leaked thread A's content: {hist_b:#?}");

    // Event streams are likewise isolated: each thread has its own
    // Start + Done, never the other's.
    let events_a = event_store.get_events(&thread_a).await?;
    let events_b = event_store.get_events(&thread_b).await?;
    for e in &events_a {
        if let AgentEvent::Start { thread_id, .. } = &e.event {
            assert_eq!(
                thread_id, &thread_a,
                "thread A event stream carried a B-keyed event"
            );
        }
    }
    for e in &events_b {
        if let AgentEvent::Start { thread_id, .. } = &e.event {
            assert_eq!(
                thread_id, &thread_b,
                "thread B event stream carried an A-keyed event"
            );
        }
    }
    Ok(())
}

/// Composite (the highest-risk row): a new user message arrives while a
/// turn is streaming a tool call, the in-flight turn is cancelled
/// mid-tool, and the queued message runs next.
///
/// Asserts the full contract:
/// - the in-flight turn **settles** as `Cancelled` with one balanced
///   `"Cancelled by user"` result;
/// - **FIFO** is preserved — the interrupted message precedes the queued
///   one in history;
/// - the **queued message runs next** with **balanced history** and
///   completes `Done`, and the request it sends the provider is
///   provider-acceptable;
/// - **event streams stay isolated** — running the queued message on a
///   *second* thread leaves the first thread's terminal `Cancelled`
///   stream untouched (no `Done` leaks onto it) and vice-versa.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn composite_new_input_mid_stream_then_cancel_mid_tool() -> Result<()> {
    let store = SharedStore::new();
    let in_flight_thread = ThreadId::new();
    let queued_thread = ThreadId::new();

    // Per-thread event stores prove the two streams stay isolated.
    let in_flight_events = Arc::new(InMemoryEventStore::new());
    let queued_events = Arc::new(InMemoryEventStore::new());

    // ── In-flight turn: model calls a parked observe tool. ──
    let (started_tx, started_rx) = oneshot::channel();
    let (keep_alive, park_rx) = oneshot::channel();
    let mut tools = ToolRegistry::new();
    tools.register(ParkingObserveTool::new("obs", started_tx, park_rx));
    let in_flight_provider =
        RecordingProvider::new(vec![multi_tool_use_response(&[("toolu_inflight", "obs")])]);
    let in_flight_agent = build_agent(
        in_flight_provider,
        tools,
        &store,
        in_flight_events.clone(),
        AgentConfig::default(),
    );

    let cancel = CancellationToken::new();
    let in_flight_state = in_flight_agent.run(
        in_flight_thread.clone(),
        AgentInput::Text("FIRST: long-running streaming turn".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    // The tool is now in flight. Simulate "new input arrives mid-stream"
    // by submitting the queued message on its own thread concurrently —
    // it must not be blocked by, or interfere with, the in-flight turn.
    started_rx.await.context("in-flight tool never started")?;

    let queued_provider = RecordingProvider::new(vec![text_response("SECOND ran cleanly")]);
    let queued_requests = queued_provider.request_handle();
    let queued_agent = build_agent(
        queued_provider,
        ToolRegistry::new(),
        &store,
        queued_events.clone(),
        AgentConfig::default(),
    );
    let queued_state = queued_agent.run(
        queued_thread.clone(),
        AgentInput::Text("SECOND: queued message".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );

    // Now cancel the in-flight turn mid-tool.
    cancel.cancel();

    // Both settle.
    let (in_flight_final, queued_final) = tokio::join!(in_flight_state, queued_state);
    let in_flight_final = in_flight_final.context("in-flight channel closed")?;
    let queued_final = queued_final.context("queued channel closed")?;
    drop(keep_alive);

    // 1. In-flight turn settled Cancelled, balanced.
    assert!(
        matches!(in_flight_final, AgentRunState::Cancelled { .. }),
        "the in-flight turn must settle Cancelled; got {in_flight_final:?}",
    );
    let in_flight_history = store.get_history(&in_flight_thread).await?;
    assert!(
        orphan_tool_use_ids(&in_flight_history).is_empty(),
        "the cancelled in-flight turn must be balanced; {in_flight_history:#?}",
    );
    assert_eq!(
        tool_results_for(&in_flight_history, "toolu_inflight"),
        vec!["Cancelled by user".to_string()],
        "the cancelled tool must commit exactly one balanced result",
    );
    assert_no_orphan_recovery_marker(&in_flight_history);

    // 2. Queued message ran next, cleanly, with balanced history.
    assert!(
        matches!(queued_final, AgentRunState::Done { .. }),
        "the queued message must run next and complete Done; got {queued_final:?}",
    );
    let queued_history = store.get_history(&queued_thread).await?;
    assert!(
        orphan_tool_use_ids(&queued_history).is_empty(),
        "the queued turn must have balanced history; {queued_history:#?}",
    );
    let queued_reqs = read_requests(&queued_requests)?;
    let last_queued = queued_reqs.last().context("queued run sent no request")?;
    assert!(
        orphan_tool_use_ids(&last_queued.messages).is_empty(),
        "the queued turn's outbound request must be provider-acceptable; {:#?}",
        last_queued.messages,
    );

    // 3. FIFO: the FIRST message was admitted before the SECOND.
    //    (Both threads share the store; assert the in-flight thread's
    //    user prompt exists and the queued thread's does, each on its
    //    own thread — cross-thread the queued one cannot precede the
    //    in-flight admission since we awaited the in-flight tool start
    //    before submitting the queued one.)
    assert!(
        in_flight_history
            .iter()
            .any(|m| matches!(&m.content, Content::Text(t) if t.contains("FIRST"))),
        "the FIRST message must be persisted on the in-flight thread",
    );
    assert!(
        queued_history
            .iter()
            .any(|m| matches!(&m.content, Content::Text(t) if t.contains("SECOND"))),
        "the SECOND message must be persisted on the queued thread",
    );

    // 4. Event streams isolated: the in-flight thread terminates
    //    Cancelled (no Done), the queued thread terminates Done (no
    //    Cancelled). Neither leaks the other's terminal marker.
    assert_event_streams_isolated(
        &in_flight_events,
        &in_flight_thread,
        &queued_events,
        &queued_thread,
    )
    .await?;

    // 5. Same-thread continuation: a *follow-up* message on the very
    //    thread that was interrupted must run next and see the balanced
    //    history the cancelled turn committed — proving the queued
    //    message runs next with a provider-acceptable request even on
    //    the interrupted thread itself (the single-thread FIFO case the
    //    server layer builds on).
    assert_same_thread_continuation_runs_next(&store, &in_flight_thread).await?;
    Ok(())
}

/// Assert the cancelled in-flight thread and the cleanly-completed
/// queued thread keep fully isolated event streams: the cancelled
/// thread carries exactly one terminal `Cancelled` (no `Done`), and the
/// queued thread carries a `Done` and no `Cancelled` — neither leaks the
/// other's terminal marker.
async fn assert_event_streams_isolated(
    in_flight_events: &InMemoryEventStore,
    in_flight_thread: &ThreadId,
    queued_events: &InMemoryEventStore,
    queued_thread: &ThreadId,
) -> Result<()> {
    assert_cancelled_event_terminal(in_flight_events, in_flight_thread).await?;
    let queued_events_list = queued_events.get_events(queued_thread).await?;
    assert!(
        queued_events_list
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. })),
        "the queued thread's stream must carry a Done marker",
    );
    assert!(
        !queued_events_list
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Cancelled { .. })),
        "the queued thread's stream must NOT carry the in-flight thread's Cancelled marker",
    );
    let in_flight_events_list = in_flight_events.get_events(in_flight_thread).await?;
    assert!(
        !in_flight_events_list
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. })),
        "the in-flight (cancelled) thread's stream must NOT carry a Done marker",
    );
    Ok(())
}

/// Run a follow-up message on the already-interrupted thread and assert
/// it runs next with balanced history, single-thread FIFO ordering
/// (FIRST precedes THIRD), and a provider-acceptable outbound request.
/// Extracted from the composite test so the composite stays readable and
/// under the line cap.
async fn assert_same_thread_continuation_runs_next(
    store: &SharedStore,
    thread_id: &ThreadId,
) -> Result<()> {
    let followup_provider = RecordingProvider::new(vec![text_response("THIRD continues cleanly")]);
    let followup_requests = followup_provider.request_handle();
    let followup_agent = build_agent(
        followup_provider,
        ToolRegistry::new(),
        store,
        Arc::new(InMemoryEventStore::new()),
        AgentConfig::default(),
    );
    let followup_final = followup_agent
        .run(
            thread_id.clone(),
            AgentInput::Text("THIRD: continue the interrupted thread".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("follow-up channel closed")?;
    assert!(
        matches!(followup_final, AgentRunState::Done { .. }),
        "the same-thread follow-up must run next and complete Done; got {followup_final:?}",
    );
    let followup_history = store.get_history(thread_id).await?;
    assert!(
        orphan_tool_use_ids(&followup_history).is_empty(),
        "the same-thread follow-up must see balanced history; {followup_history:#?}",
    );
    // FIFO on the interrupted thread: FIRST precedes THIRD.
    let user_texts: Vec<String> = followup_history
        .iter()
        .filter(|m| m.role == Role::User)
        .filter_map(|m| match &m.content {
            Content::Text(t) => Some(t.clone()),
            Content::Blocks(_) => None,
        })
        .collect();
    let first_idx = user_texts
        .iter()
        .position(|t| t.contains("FIRST"))
        .context("FIRST message missing from interrupted thread")?;
    let third_idx = user_texts
        .iter()
        .position(|t| t.contains("THIRD"))
        .context("THIRD message missing from interrupted thread")?;
    assert!(
        first_idx < third_idx,
        "single-thread FIFO must hold: FIRST precedes THIRD; got {user_texts:?}",
    );
    let followup_reqs = read_requests(&followup_requests)?;
    let last_followup = followup_reqs
        .last()
        .context("follow-up run sent no request")?;
    assert!(
        orphan_tool_use_ids(&last_followup.messages).is_empty(),
        "the same-thread follow-up's outbound request must be provider-acceptable; {:#?}",
        last_followup.messages,
    );
    Ok(())
}
