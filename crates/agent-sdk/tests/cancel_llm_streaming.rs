//! Cancellation completeness for the LLM call and context compaction.
//!
//! Companion to `cancel_mid_tool.rs`. That test locks in the
//! tool-boundary cancel race; this one locks in the gaps that were
//! still open after it (Phase 10 · A):
//!
//! 1. **Cancel mid-stream** — a cancel issued while the model is still
//!    streaming stops further model events promptly (it does not wait
//!    for the full response), the run ends [`AgentRunState::Cancelled`],
//!    history stays balanced, and a terminal [`AgentEvent::Cancelled`]
//!    is emitted.
//! 2. **Cancel before the first token** — a cancel that lands before
//!    the first delta ends the run `Cancelled` with no half-written
//!    assistant message and no orphan `tool_use`.
//! 3. **Cancel during compaction** — a cancel while the (slow,
//!    destructive) compaction summary is in flight prevents
//!    `replace_history` from ever running, so history is left
//!    untouched.
//! 4. **Terminal event** — every cancellation path emits exactly one
//!    `Cancelled` event so a streaming consumer receives a closing
//!    marker and does not hang waiting for `Done`.
//!
//! Throughout, the contract from `cancel_mid_tool.rs` holds: the run
//! returns `Cancelled` and the persisted history has no orphan
//! `tool_use` blocks.

use agent_sdk::context::{CompactionResult, ContextCompactor};
use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, StreamBox, StreamDelta, Usage,
};
use agent_sdk::{
    AgentEvent, AgentInput, AgentRunState, AgentState, AllowAllHooks, CancellationToken,
    EventStore, InMemoryEventStore, InMemoryStore, MessageStore, StateStore, ThreadId, TokenUsage,
    ToolContext, ToolRegistry, builder,
};
use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::sync::oneshot;

/// Shared (clonable) in-memory store so a follow-up run can inspect the
/// history the cancelled run left behind.
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

/// IDs of `tool_use` blocks that have no matching `tool_result` block in
/// the following user message. Empty means a balanced history.
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

/// Whether the run emitted exactly one terminal `Cancelled` event and
/// no `Done` event (the run did not pretend to finish normally).
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
    assert_eq!(
        done, 0,
        "a cancelled run must not emit a Done event as if it finished",
    );
    Ok(())
}

// ── 1 & 2: streaming-cancel provider ────────────────────────────────

/// Provider whose stream emits a few text deltas, signals the test that
/// the stream is live, then parks forever on a oneshot. It does **not**
/// observe any cancel token — the SDK's `process_stream` race must be
/// what stops it. A `cancel_before_first_token` flag makes the very
/// first poll park before any delta is emitted.
struct BlockingStreamProvider {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    park_rx: Mutex<Option<oneshot::Receiver<()>>>,
    cancel_before_first_token: bool,
}

impl BlockingStreamProvider {
    fn new(started_tx: oneshot::Sender<()>, park_rx: oneshot::Receiver<()>, before: bool) -> Self {
        Self {
            started_tx: Mutex::new(Some(started_tx)),
            park_rx: Mutex::new(Some(park_rx)),
            cancel_before_first_token: before,
        }
    }
}

#[async_trait]
impl LlmProvider for BlockingStreamProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Err(anyhow!("BlockingStreamProvider only supports chat_stream"))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let before = self.cancel_before_first_token;
        Box::pin(async_stream::stream! {
            let started_tx = self.started_tx.lock().await.take();
            let park_rx = self.park_rx.lock().await.take();

            if !before {
                // Emit a couple of content deltas so the consumer is
                // mid-stream when the cancel arrives.
                yield Ok(StreamDelta::TextDelta { delta: "Thinking".to_string(), block_index: 0 });
                yield Ok(StreamDelta::TextDelta { delta: " out loud".to_string(), block_index: 0 });
            }

            // Tell the test the stream is live, then park. Once-only.
            if let Some(sender) = started_tx {
                let _ = sender.send(());
            }
            if let Some(rx) = park_rx {
                // Never resolved by the test; the SDK boundary must
                // cancel the stream poll. If the fix regresses this
                // hangs and the tokio test default timeout fires.
                let _ = rx.await;
            }
            yield Ok(StreamDelta::TextDelta { delta: " (unexpected)".to_string(), block_index: 0 });
            yield Ok(StreamDelta::Done { stop_reason: Some(StopReason::EndTurn) });
        })
    }

    fn model(&self) -> &'static str {
        "test-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

async fn run_streaming_cancel(
    before_first_token: bool,
) -> Result<(
    AgentRunState,
    SharedStore,
    ThreadId,
    Arc<InMemoryEventStore>,
)> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    let (started_tx, started_rx) = oneshot::channel::<()>();
    let (keep_alive_park_tx, park_rx) = oneshot::channel::<()>();
    let provider = BlockingStreamProvider::new(started_tx, park_rx, before_first_token);

    let config = agent_sdk::AgentConfig {
        streaming: true,
        ..agent_sdk::AgentConfig::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(ToolRegistry::new())
        .hooks(AllowAllHooks)
        .config(config)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(event_store.clone())
        .build_with_stores();

    let cancel = CancellationToken::new();
    if before_first_token {
        // Cancel before the run even starts the stream so the first
        // poll sees an already-cancelled token.
        cancel.cancel();
    }

    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("stream me a long answer".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    if !before_first_token {
        started_rx
            .await
            .context("stream never signalled that it was live")?;
        cancel.cancel();
    }

    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive_park_tx);

    Ok((final_state, store, thread_id, event_store))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_mid_stream_ends_cancelled_with_balanced_history() -> Result<()> {
    let (final_state, store, thread_id, event_store) = run_streaming_cancel(false).await?;

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "cancel mid-stream must end the run as Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    let orphans = orphan_tool_use_ids(&history);
    assert!(
        orphans.is_empty(),
        "cancel mid-stream must leave zero orphan tool_use blocks; got {orphans:?} in {history:#?}",
    );

    // No half-written assistant message should have been persisted from
    // the partial stream (the turn never completed).
    let assistant_msgs = history.iter().filter(|m| m.role == Role::Assistant).count();
    assert_eq!(
        assistant_msgs, 0,
        "cancel mid-stream must not persist a half-written assistant message; history {history:#?}",
    );

    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_before_first_token_ends_cancelled_clean() -> Result<()> {
    let (final_state, store, thread_id, event_store) = run_streaming_cancel(true).await?;

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "cancel before the first token must end the run as Cancelled; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;
    assert!(
        orphan_tool_use_ids(&history).is_empty(),
        "cancel before first token must leave no orphan tool_use; history {history:#?}",
    );
    let assistant_msgs = history.iter().filter(|m| m.role == Role::Assistant).count();
    assert_eq!(
        assistant_msgs, 0,
        "cancel before first token must not persist any assistant message; history {history:#?}",
    );

    assert_cancelled_event_terminal(&event_store, &thread_id).await?;
    Ok(())
}

// ── 3: compaction-cancel ─────────────────────────────────────────────

/// Non-streaming provider that returns a plain text response. Only the
/// compaction-cancel test uses it; compaction runs in `load_turn_messages`
/// *before* this provider is ever called, so its body is unreachable in
/// the cancellation path.
struct SimpleTextProvider;

#[async_trait]
impl LlmProvider for SimpleTextProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "resp".to_string(),
            content: vec![ContentBlock::Text {
                text: "done".to_string(),
            }],
            model: "test-model".to_string(),
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
        "test-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

/// Compactor that always wants to compact, signals the test that
/// `compact_history` is live, then parks forever. If it ever reaches the
/// (destructive) `replace_history` write, it flips `replace_ran` — the
/// test asserts that never happens once the cancel fires.
struct BlockingCompactor {
    started_tx: Mutex<Option<oneshot::Sender<()>>>,
    park_rx: Mutex<Option<oneshot::Receiver<()>>>,
    replace_observed: Arc<AtomicBool>,
}

#[async_trait]
impl ContextCompactor for BlockingCompactor {
    async fn compact(&self, _messages: &[Message]) -> Result<String> {
        Err(anyhow!("unused"))
    }

    fn estimate_tokens(&self, _messages: &[Message]) -> usize {
        1_000_000
    }

    fn needs_compaction(&self, _messages: &[Message]) -> bool {
        true
    }

    async fn compact_history(&self, messages: Vec<Message>) -> Result<CompactionResult> {
        // Bind the awaited guards into their own variables so the
        // `MutexGuard` drops before the `if let` body runs.
        let started_tx = self.started_tx.lock().await.take();
        if let Some(sender) = started_tx {
            let _ = sender.send(());
        }
        let park_rx = self.park_rx.lock().await.take();
        if let Some(rx) = park_rx {
            // Park until cancelled. The SDK must race this against the
            // cancel token and never proceed to replace_history.
            let _ = rx.await;
        }
        // Reaching here would mean the SDK did NOT honor the cancel and
        // is about to perform the destructive history rewrite.
        self.replace_observed.store(true, Ordering::SeqCst);
        let original_count = messages.len();
        Ok(CompactionResult {
            messages: vec![Message::user("[summary]")],
            original_count,
            new_count: 1,
            original_tokens: 1_000_000,
            new_tokens: 10,
            llm_usage: TokenUsage::default(),
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_during_compaction_prevents_replace_history() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let event_store = Arc::new(InMemoryEventStore::new());

    // Seed a non-trivial history so there is something to "compact".
    for i in 0..4 {
        store
            .append(&thread_id, Message::user(format!("message {i}")))
            .await?;
    }

    let (started_tx, started_rx) = oneshot::channel::<()>();
    let (keep_alive_park_tx, park_rx) = oneshot::channel::<()>();
    let replace_observed = Arc::new(AtomicBool::new(false));
    let compactor = BlockingCompactor {
        started_tx: Mutex::new(Some(started_tx)),
        park_rx: Mutex::new(Some(park_rx)),
        replace_observed: Arc::clone(&replace_observed),
    };

    let agent = builder::<()>()
        .provider(SimpleTextProvider)
        .tools(ToolRegistry::new())
        .hooks(AllowAllHooks)
        .with_custom_compactor(compactor)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(event_store.clone())
        .build_with_stores();

    let cancel = CancellationToken::new();
    let state_rx = agent.run(
        thread_id.clone(),
        AgentInput::Text("continue the work".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    started_rx
        .await
        .context("compactor never signalled that compact_history was live")?;
    cancel.cancel();

    let final_state = state_rx.await.context("agent state channel closed")?;
    drop(keep_alive_park_tx);

    assert!(
        matches!(final_state, AgentRunState::Cancelled { .. }),
        "cancel during compaction must end the run as Cancelled; got {final_state:?}",
    );
    assert!(
        !replace_observed.load(Ordering::SeqCst),
        "the destructive replace_history must NOT run after a cancel during compaction",
    );

    // The destructive rewrite never landed: the compactor's `[summary]`
    // marker must be absent and all four seeded messages must survive.
    let history_after = store.get_history(&thread_id).await?;
    let has_summary = history_after.iter().any(|m| match &m.content {
        Content::Text(text) => text.contains("[summary]"),
        Content::Blocks(_) => false,
    });
    assert!(
        !has_summary,
        "compaction cancel must not write the summarized history; got {history_after:#?}",
    );
    let seeded_survived = (0..4).all(|i| {
        history_after.iter().any(|m| match &m.content {
            Content::Text(text) => text == &format!("message {i}"),
            Content::Blocks(_) => false,
        })
    });
    assert!(
        seeded_survived,
        "compaction cancel must leave the original messages intact; got {history_after:#?}",
    );

    assert_cancelled_event_terminal(&event_store, &thread_id).await?;

    // Belt-and-suspenders: give any leaked compaction task a moment; the
    // replace flag must still be unset.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !replace_observed.load(Ordering::SeqCst),
        "replace_history must never run even after the run resolves",
    );

    Ok(())
}
