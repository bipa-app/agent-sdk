use super::test_utils::*;
use super::*;
use crate::context::{CompactionConfig, CompactionResult, ContextCompactor};
use crate::events::{AgentEvent, AgentEventEnvelope};
use crate::hooks::{AgentHooks, AllowAllHooks, RequestDecision, ToolDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Message, Role, StopReason,
    StreamDelta, StreamErrorKind, Usage,
};
use crate::reminders::{ReminderConfig, ToolReminder};
use crate::stores::{
    EventStore, InMemoryEventStore, InMemoryStore, MessageStore, StateStore, StoredTurnEvents,
};
use crate::tools::{ListenToolUpdate, ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentInput, AgentRunState, BudgetLimitKind, ContinuationEnvelope, ToolInvocation,
    ToolResult, ToolTier, TurnOptions, TurnOutcome, UsageLimits,
};
use crate::types::{AgentState, RetryConfig};
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn expected_turn_for_input(input: &AgentInput, existing_turns: &[StoredTurnEvents]) -> usize {
    match input {
        AgentInput::Resume { continuation, .. } => continuation.payload.turn,
        AgentInput::SubmitToolResults { continuation, .. } => {
            continuation.payload.turn.saturating_add(1)
        }
        _ => existing_turns
            .last()
            .map_or(1, |turn| turn.turn.saturating_add(1)),
    }
}

async fn load_turn_events(
    store: &dyn EventStore,
    thread_id: &ThreadId,
    turn: usize,
) -> anyhow::Result<Vec<AgentEventEnvelope>> {
    Ok(store
        .get_turn(thread_id, turn)
        .await?
        .map_or_else(Vec::new, |stored_turn| stored_turn.events))
}

async fn run_turn_recorded<Ctx, P, H, M, S>(
    agent: &AgentLoop<Ctx, P, H, M, S>,
    thread_id: ThreadId,
    input: AgentInput,
    tool_context: ToolContext<Ctx>,
    options: TurnOptions,
) -> anyhow::Result<(TurnOutcome, Vec<AgentEventEnvelope>)>
where
    Ctx: Send + Sync + Clone + 'static,
    P: crate::llm::LlmProvider + 'static,
    H: crate::hooks::AgentHooks + 'static,
    M: crate::stores::MessageStore + 'static,
    S: crate::stores::StateStore + 'static,
{
    let existing_turns = agent.event_store.get_turns(&thread_id).await?;
    let expected_turn = expected_turn_for_input(&input, &existing_turns);
    // Box the `run_turn` future so this helper's own future stays under
    // the `clippy::large_futures` threshold across its ~60 call sites.
    let outcome = Box::pin(agent.run_turn(
        thread_id.clone(),
        input,
        tool_context,
        CancellationToken::new(),
        options,
    ))
    .await;
    let events = load_turn_events(agent.event_store.as_ref(), &thread_id, expected_turn).await?;
    Ok((outcome, events))
}

async fn run_recorded<Ctx, P, H, M, S>(
    agent: &AgentLoop<Ctx, P, H, M, S>,
    thread_id: ThreadId,
    input: AgentInput,
    tool_context: ToolContext<Ctx>,
) -> anyhow::Result<(AgentRunState, Vec<AgentEventEnvelope>)>
where
    Ctx: Send + Sync + Clone + 'static,
    P: crate::llm::LlmProvider + 'static,
    H: crate::hooks::AgentHooks + 'static,
    M: crate::stores::MessageStore + 'static,
    S: crate::stores::StateStore + 'static,
{
    let state_rx = agent.run(
        thread_id.clone(),
        input,
        tool_context,
        CancellationToken::new(),
    );
    let state = state_rx.await?;
    let events = load_events(agent.event_store.as_ref(), &thread_id).await?;
    Ok((state, events))
}

#[derive(Clone, Copy)]
enum EventStoreFailureMode {
    Append,
    FinishTurn,
    /// Fail both `append` and `finish_turn` — models the common case where
    /// the same store outage that rejected the event append also rejects the
    /// turn-close barrier.
    Both,
}

#[derive(Clone)]
struct FailingEventStore {
    inner: Arc<InMemoryEventStore>,
    failure_mode: EventStoreFailureMode,
}

impl FailingEventStore {
    fn new(failure_mode: EventStoreFailureMode) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(InMemoryEventStore::new()),
            failure_mode,
        })
    }
}

#[async_trait]
impl EventStore for FailingEventStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> anyhow::Result<()> {
        if matches!(
            self.failure_mode,
            EventStoreFailureMode::Append | EventStoreFailureMode::Both
        ) {
            anyhow::bail!("append failure");
        }
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> anyhow::Result<()> {
        if matches!(
            self.failure_mode,
            EventStoreFailureMode::FinishTurn | EventStoreFailureMode::Both
        ) {
            anyhow::bail!("finish failure");
        }
        self.inner.finish_turn(thread_id, turn).await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> anyhow::Result<Option<StoredTurnEvents>> {
        self.inner.get_turn(thread_id, turn).await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<StoredTurnEvents>> {
        self.inner.get_turns(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> anyhow::Result<()> {
        self.inner.clear(thread_id).await
    }
}

#[derive(Clone, Default)]
struct FailingMessageStore;

#[async_trait]
impl MessageStore for FailingMessageStore {
    async fn append(
        &self,
        _thread_id: &ThreadId,
        _message: crate::llm::Message,
    ) -> anyhow::Result<()> {
        anyhow::bail!("message store unavailable");
    }

    async fn get_history(&self, _thread_id: &ThreadId) -> anyhow::Result<Vec<crate::llm::Message>> {
        Ok(Vec::new())
    }

    async fn clear(&self, _thread_id: &ThreadId) -> anyhow::Result<()> {
        Ok(())
    }

    async fn replace_history(
        &self,
        _thread_id: &ThreadId,
        _messages: Vec<crate::llm::Message>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ConfirmAllHooks;

#[async_trait]
impl AgentHooks for ConfirmAllHooks {
    async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
        ToolDecision::RequiresConfirmation(format!("Confirm {}?", invocation.tool_name))
    }
}

// ===================
// Builder Tests
// ===================

#[test]
fn test_builder_creates_agent_loop() {
    let provider = MockProvider::new(vec![]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    assert_eq!(agent.config.max_turns, None);
    assert_eq!(agent.config.max_tokens, None);
}

#[test]
fn test_builder_with_custom_config() {
    let provider = MockProvider::new(vec![]);
    let config = AgentConfig {
        max_turns: Some(5),
        max_tokens: Some(2048),
        system_prompt: "Custom prompt".to_string(),
        model: "custom-model".to_string(),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    assert_eq!(agent.config.max_turns, Some(5));
    assert_eq!(agent.config.max_tokens, Some(2048));
    assert_eq!(agent.config.system_prompt, "Custom prompt");
}

#[test]
fn test_builder_with_tools() {
    let provider = MockProvider::new(vec![]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    assert_eq!(agent.tools.len(), 1);
}

#[test]
fn test_builder_with_custom_stores() {
    let provider = MockProvider::new(vec![]);
    let message_store = InMemoryStore::new();
    let state_store = InMemoryStore::new();

    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(message_store)
        .state_store(state_store)
        .event_store(new_event_store())
        .build_with_stores();

    // Just verify it builds without panicking
    assert_eq!(agent.config.max_turns, None);
}

// ===================
// Run Loop Tests
// ===================

#[tokio::test]
async fn test_simple_text_response() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello, user!")]);

    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have: Start, Text, Done
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Text { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. }))
    );

    Ok(())
}

#[tokio::test]
async fn test_tool_execution() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        // First call: request tool use
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "test"})),
        // Second call: respond with text
        MockProvider::text_response("Tool executed successfully"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Run echo".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have tool call events
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ToolCallStart { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ToolCallEnd { .. }))
    );

    Ok(())
}

/// A read-only tool whose `execute` blocks on a shared [`tokio::sync::Barrier`]
/// until `party_size` concurrent invocations have arrived. Used to prove the
/// agent loop fans out `ToolTier::Observe` calls in parallel: a serial loop
/// would deadlock the barrier (only one party ever arrives) and the test's
/// surrounding `tokio::time::timeout` would fail.
struct BarrierTool {
    barrier: Arc<tokio::sync::Barrier>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum BarrierToolName {
    Barrier,
}

impl crate::tools::ToolName for BarrierToolName {}

impl crate::tools::Tool<()> for BarrierTool {
    type Name = BarrierToolName;

    fn name(&self) -> BarrierToolName {
        BarrierToolName::Barrier
    }

    fn display_name(&self) -> &'static str {
        "Barrier"
    }

    fn description(&self) -> &'static str {
        "Waits on a shared barrier; used to prove parallel observe-tier execution."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        self.barrier.wait().await;
        Ok(ToolResult::success("through".to_string()))
    }
}

/// With serial tool execution (the pre-patch behavior) this test deadlocks on
/// the first `barrier.wait()` and the outer timeout fires. With parallel
/// observe-tier batching, all three calls arrive at the barrier concurrently
/// and complete immediately.
///
/// We also assert that results appear in the same order the model emitted
/// them, because parallel fan-out must not reshuffle the tool-result stream.
#[tokio::test]
async fn test_observe_tools_run_in_parallel() -> anyhow::Result<()> {
    const PARTY: usize = 3;

    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("call_a", "barrier", json!({})),
            ("call_b", "barrier", json!({})),
            ("call_c", "barrier", json!({})),
        ]),
        MockProvider::text_response("done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(BarrierTool {
        barrier: Arc::new(tokio::sync::Barrier::new(PARTY)),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        run_recorded(
            &agent,
            thread_id,
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
        ),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "tool calls ran serially: the barrier deadlocked because only \
             one of {PARTY} parties ever arrived. parallel observe-tier \
             execution is not active."
        )
    })??;

    // All three tool calls finished successfully. We do NOT assert event
    // order: with parallel execution, `ToolCallEnd` events arrive in
    // completion order, which is what a streaming UI wants. The model-facing
    // `tool_results` vector is separately input-ordered by `join_all`'s
    // ordering guarantee — see `execute_pending_tool_calls_for_turn`.
    let mut tool_end_ids: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.event {
            AgentEvent::ToolCallEnd { id, result, .. } => Some((id.clone(), result.success)),
            _ => None,
        })
        .collect();
    tool_end_ids.sort();

    assert_eq!(
        tool_end_ids,
        vec![
            ("call_a".to_string(), true),
            ("call_b".to_string(), true),
            ("call_c".to_string(), true),
        ],
        "expected all three barrier calls to complete successfully, got {tool_end_ids:?}"
    );

    Ok(())
}

#[tokio::test]
async fn test_max_turns_limit() -> anyhow::Result<()> {
    // Provider that always requests a tool
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "1"})),
        MockProvider::tool_use_response("tool_2", "echo", json!({"message": "2"})),
        MockProvider::tool_use_response("tool_3", "echo", json!({"message": "3"})),
        MockProvider::tool_use_response("tool_4", "echo", json!({"message": "4"})),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let config = AgentConfig {
        max_turns: Some(2),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Loop".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have an error about max turns
    assert!(events.iter().any(|e| {
        matches!(&e.event, AgentEvent::Error { message, .. } if message.contains("Maximum turns"))
    }));

    Ok(())
}

#[tokio::test]
async fn test_unknown_tool_handling() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        // Request unknown tool
        MockProvider::tool_use_response("tool_1", "nonexistent_tool", json!({})),
        // LLM gets tool error and ends conversation
        MockProvider::text_response("I couldn't find that tool."),
    ]);

    // Empty tool registry
    let tools = ToolRegistry::new();

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Call unknown".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Unknown tool errors are returned to the LLM (not emitted as ToolCallEnd)
    // The conversation should complete successfully with a Done event
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. }))
    );

    // The LLM's response about the missing tool should be in the events
    assert!(events.iter().any(|e| {
        matches!(&e.event, AgentEvent::Text { text, .. } if text.contains("couldn't find"))
    }));

    Ok(())
}

#[tokio::test]
async fn test_rate_limit_handling() -> anyhow::Result<()> {
    // Provide enough RateLimited responses to exhaust all retries (max_retries + 1)
    let provider = MockProvider::new(vec![
        ChatOutcome::RateLimited(None),
        ChatOutcome::RateLimited(None),
        ChatOutcome::RateLimited(None),
        ChatOutcome::RateLimited(None),
        ChatOutcome::RateLimited(None),
        ChatOutcome::RateLimited(None), // 6th attempt exceeds max_retries (5)
    ]);

    // Use fast retry config for faster tests
    let config = AgentConfig {
        retry: crate::types::RetryConfig::fast(),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have rate limit error after exhausting retries
    assert!(events.iter().any(|e| {
        matches!(&e.event, AgentEvent::Error { message, recoverable: true } if message.contains("Rate limited"))
    }));

    Ok(())
}

#[tokio::test]
async fn test_rate_limit_recovery() -> anyhow::Result<()> {
    // Rate limited once, then succeeds
    let provider = MockProvider::new(vec![
        ChatOutcome::RateLimited(None),
        MockProvider::text_response("Recovered after rate limit"),
    ]);

    // Use fast retry config for faster tests
    let config = AgentConfig {
        retry: crate::types::RetryConfig::fast(),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have successful completion after retry
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. }))
    );

    Ok(())
}

#[tokio::test]
async fn test_server_error_handling() -> anyhow::Result<()> {
    // Provide enough ServerError responses to exhaust all retries (max_retries + 1)
    let provider = MockProvider::new(vec![
        ChatOutcome::ServerError("Internal error".to_string()),
        ChatOutcome::ServerError("Internal error".to_string()),
        ChatOutcome::ServerError("Internal error".to_string()),
        ChatOutcome::ServerError("Internal error".to_string()),
        ChatOutcome::ServerError("Internal error".to_string()),
        ChatOutcome::ServerError("Internal error".to_string()), // 6th attempt exceeds max_retries
    ]);

    // Use fast retry config for faster tests
    let config = AgentConfig {
        retry: crate::types::RetryConfig::fast(),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have server error after exhausting retries
    assert!(events.iter().any(|e| {
        matches!(&e.event, AgentEvent::Error { message, recoverable: true } if message.contains("Server error"))
    }));

    Ok(())
}

#[tokio::test]
async fn test_server_error_recovery() -> anyhow::Result<()> {
    // Server error once, then succeeds
    let provider = MockProvider::new(vec![
        ChatOutcome::ServerError("Temporary error".to_string()),
        MockProvider::text_response("Recovered after server error"),
    ]);

    // Use fast retry config for faster tests
    let config = AgentConfig {
        retry: crate::types::RetryConfig::fast(),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    // Should have successful completion after retry
    assert!(
        events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Done { .. }))
    );

    Ok(())
}

// ================================
// Event Envelope Idempotency Tests
// ================================

#[tokio::test]
async fn test_envelope_event_ids_are_unique() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    let mut ids = std::collections::HashSet::new();
    for envelope in events {
        assert!(
            ids.insert(envelope.event_id),
            "duplicate event_id: {}",
            envelope.event_id
        );
    }
    assert!(ids.len() >= 3, "expected at least Start+Text+Done events");

    Ok(())
}

#[tokio::test]
async fn test_envelope_sequences_are_strictly_increasing() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, envelopes) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    for pair in envelopes.windows(2) {
        assert!(
            pair[1].sequence > pair[0].sequence,
            "sequence not strictly increasing: {} -> {}",
            pair[0].sequence,
            pair[1].sequence,
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_envelope_sequences_start_at_zero() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    let first = events.first().context("should have at least one event")?;
    assert_eq!(first.sequence, 0);

    Ok(())
}

#[tokio::test]
async fn test_envelope_sequences_have_no_gaps() -> anyhow::Result<()> {
    // Use a tool call to generate more events (Start, ToolCallStart, ToolCallEnd, Text, TurnComplete, Done, etc.)
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("t1", "echo", json!({"message": "test"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, sequences_events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Go".into()),
        ToolContext::new(()),
    )
    .await?;

    let sequences: Vec<u64> = sequences_events
        .into_iter()
        .map(|envelope| envelope.sequence)
        .collect();

    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(
        sequences, expected,
        "sequences should be 0, 1, 2, ... with no gaps"
    );

    Ok(())
}

#[tokio::test]
async fn test_envelope_timestamps_are_non_decreasing() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, envelopes) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    for pair in envelopes.windows(2) {
        assert!(
            pair[1].timestamp >= pair[0].timestamp,
            "timestamp went backwards: {:?} -> {:?}",
            pair[0].timestamp,
            pair[1].timestamp,
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_separate_runs_have_independent_sequences() -> anyhow::Result<()> {
    let provider_a = MockProvider::new(vec![MockProvider::text_response("A")]);
    let provider_b = MockProvider::new(vec![MockProvider::text_response("B")]);

    let agent_a = builder::<()>()
        .provider(provider_a)
        .event_store(new_event_store())
        .build();
    let agent_b = builder::<()>()
        .provider(provider_b)
        .event_store(new_event_store())
        .build();

    let thread_id_a = ThreadId::new();
    let thread_id_b = ThreadId::new();
    let (_, events_a) = run_recorded(
        &agent_a,
        thread_id_a,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;
    let (_, events_b) = run_recorded(
        &agent_b,
        thread_id_b,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    let first_a = events_a.first().context("run A should emit events")?;
    let first_b = events_b.first().context("run B should emit events")?;

    // Both runs start at sequence 0
    assert_eq!(first_a.sequence, 0);
    assert_eq!(first_b.sequence, 0);

    // But event_ids are different
    assert_ne!(first_a.event_id, first_b.event_id);

    Ok(())
}

#[tokio::test]
async fn test_envelope_event_ids_are_valid_uuid_v4() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    )
    .await?;

    for envelope in events {
        assert_eq!(
            envelope.event_id.get_version(),
            Some(uuid::Version::Random),
            "event_id should be UUID v4"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_envelope_with_tool_calls_maintains_invariants() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("t1", "echo", json!({"message": "a"})),
        MockProvider::tool_use_response("t2", "echo", json!({"message": "b"})),
        MockProvider::text_response("All done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (_, envelopes) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Go".into()),
        ToolContext::new(()),
    )
    .await?;

    // All event_ids unique
    let ids: std::collections::HashSet<uuid::Uuid> = envelopes.iter().map(|e| e.event_id).collect();
    assert_eq!(ids.len(), envelopes.len(), "all event_ids must be unique");

    // Sequences: 0, 1, 2, ... no gaps
    let expected: Vec<u64> = (0..envelopes.len() as u64).collect();
    let actual: Vec<u64> = envelopes.iter().map(|e| e.sequence).collect();
    assert_eq!(actual, expected, "sequences must be contiguous from 0");

    // Timestamps non-decreasing
    for pair in envelopes.windows(2) {
        assert!(pair[1].timestamp >= pair[0].timestamp);
    }

    // Should contain tool call events wrapped in envelopes
    assert!(
        envelopes
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ToolCallStart { .. }))
    );
    assert!(
        envelopes
            .iter()
            .any(|e| matches!(e.event, AgentEvent::ToolCallEnd { .. }))
    );

    Ok(())
}

#[tokio::test]
async fn test_listen_tool_confirmation_flow() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("Listen flow complete"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ListenEchoTool {
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    // Turn 1: reaches awaiting confirmation after listen pre-runtime
    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    // Turn 2: confirm and execute
    let (outcome_2, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

    // Turn 3: continue and finish
    let (outcome_3, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Continue,
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_3, TurnOutcome::Done { .. }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 0);

    Ok(())
}

#[tokio::test]
async fn test_run_turn_handles_multiple_confirmation_resumes_on_same_turn() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("tool_1", "echo", json!({"message": "first"})),
            ("tool_2", "echo", json!({"message": "second"})),
        ]),
        MockProvider::text_response("All confirmations complete"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .hooks(ConfirmAllHooks)
        .tools(tools)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::new();

    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run two tools".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (continuation_1, tool_call_id_1) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected first confirmation, got {other:?}"),
    };
    assert_eq!(continuation_1.payload.turn, 1);

    let (outcome_2, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation: continuation_1,
            tool_call_id: tool_call_id_1,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (continuation_2, tool_call_id_2) = match outcome_2 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected second confirmation, got {other:?}"),
    };
    assert_eq!(continuation_2.payload.turn, 1);

    let (outcome_3, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation: continuation_2,
            tool_call_id: tool_call_id_2,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_3, TurnOutcome::NeedsMoreTurns { .. }));

    let (outcome_4, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Continue,
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_4, TurnOutcome::Done { .. }));

    Ok(())
}

#[tokio::test]
async fn test_listen_tool_rejection_cancels_operation() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("Rejected flow complete"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ListenEchoTool {
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let _ = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: false,
            rejection_reason: Some("nope".to_string()),
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    assert_eq!(cancel_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn test_listen_tool_invalidated_stream_returns_error_result() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("After invalidation"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates: vec![ListenToolUpdate::Invalidated {
            operation_id: "listen-op-1".to_string(),
            message: "quote expired".to_string(),
            recoverable: true,
        }],
        execute_error: None,
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    )
    .await?;

    assert!(events.iter().any(|event| {
        matches!(
            &event.event,
            AgentEvent::ToolProgress { stage, .. } if stage == "listen_invalidated"
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(
            &event.event,
            AgentEvent::ToolCallEnd { result, .. }
                if !result.success && result.output.contains("invalidated")
        )
    }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn test_listen_tool_stream_end_before_ready_is_reported() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("After stream end"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates: vec![ListenToolUpdate::Listening {
            operation_id: "listen-op-1".to_string(),
            revision: 1,
            message: "still preparing".to_string(),
            snapshot: None,
            expires_at: None,
        }],
        execute_error: None,
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    )
    .await?;

    assert!(events.iter().any(|event| {
        matches!(
            &event.event,
            AgentEvent::ToolCallEnd { result, .. }
                if !result.success && result.output.contains("ended before operation became ready")
        )
    }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn test_listen_tool_max_updates_exceeded_is_reported() -> anyhow::Result<()> {
    use super::types::MAX_LISTEN_UPDATES;

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("After update cap"),
    ]);

    let updates = (0..=MAX_LISTEN_UPDATES)
        .map(|revision| ListenToolUpdate::Listening {
            operation_id: "listen-op-1".to_string(),
            revision: revision as u64,
            message: format!("update-{revision}"),
            snapshot: None,
            expires_at: None,
        })
        .collect();

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates,
        execute_error: None,
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let (_, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    )
    .await?;

    assert!(events.iter().any(|event| {
        matches!(
            &event.event,
            AgentEvent::ToolCallEnd { result, .. }
                if !result.success && result.output.contains("exceeded max updates")
        )
    }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn test_listen_tool_stream_disconnect_triggers_cancel() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "listen_echo",
        json!({"message": "test"}),
    )]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates: vec![ListenToolUpdate::Listening {
            operation_id: "listen-op-1".to_string(),
            revision: 1,
            message: "still preparing".to_string(),
            snapshot: None,
            expires_at: None,
        }],
        execute_error: None,
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome, TurnOutcome::NeedsMoreTurns { .. }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn test_listen_execute_error_after_confirmation_is_reported() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "test"})),
        MockProvider::text_response("After execute error"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates: vec![ListenToolUpdate::Ready {
            operation_id: "listen-op-1".to_string(),
            revision: 1,
            message: "Ready to execute".to_string(),
            snapshot: json!({ "preview": "v1" }),
            expires_at: None,
        }],
        execute_error: Some("execute failed".to_string()),
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let (outcome_2, events_2) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));
    assert!(events_2.iter().any(|event| {
        matches!(
            &event.event,
            AgentEvent::ToolCallEnd { result, .. }
                if !result.success && result.output.contains("Listen execute error")
        )
    }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn test_mixed_listen_and_sync_tool_calls_in_one_turn() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("tool_listen", "listen_echo", json!({"message": "listen"})),
            ("tool_echo", "echo", json!({"message": "sync"})),
        ]),
        MockProvider::text_response("Mixed tool flow complete"),
    ]);

    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    tools.register_listen(ListenEchoTool {
        cancel_calls: cancel_calls.clone(),
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run mixed tools".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let (outcome_2, events_2) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

    assert!(events_2.iter().any(|event| {
        matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == "tool_listen")
    }));
    assert!(events_2.iter().any(|event| {
        matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == "tool_echo")
    }));

    let (outcome_3, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Continue,
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    assert!(matches!(outcome_3, TurnOutcome::Done { .. }));
    assert_eq!(cancel_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn test_multi_tool_results_batched_into_single_message() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        // First call: LLM requests two tools at once
        MockProvider::tool_uses_response(vec![
            ("tool_1", "echo", json!({"message": "first"})),
            ("tool_2", "echo", json!({"message": "second"})),
        ]),
        // Second call: text response
        MockProvider::text_response("Both tools done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let message_store = Arc::new(InMemoryStore::new());
    let message_store_ref = Arc::clone(&message_store);

    let agent = AgentLoop {
        provider: Arc::new(provider),
        tools: Arc::new(tools),
        hooks: Arc::new(AllowAllHooks),
        message_store,
        state_store: Arc::new(InMemoryStore::new()),
        event_store: new_event_store(),
        event_authority: None,
        config: AgentConfig::default(),
        compaction_config: None,
        compactor: None,
        execution_store: None,
        audit_sink: Arc::new(crate::hooks::NoopAuditSink),
        reminder_config: None,
        #[cfg(feature = "otel")]
        observability_store: None,
    };

    let thread_id = ThreadId::new();
    let _ = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run both tools".to_string()),
        ToolContext::new(()),
    )
    .await?;

    let history = message_store_ref.get_history(&thread_id).await?;

    // Find user messages that contain ToolResult blocks
    let tool_result_messages: Vec<_> = history
        .iter()
        .filter(|msg| {
            if let Content::Blocks(blocks) = &msg.content {
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            } else {
                false
            }
        })
        .collect();

    // There should be exactly ONE user message with tool results (batched)
    assert_eq!(
        tool_result_messages.len(),
        1,
        "Expected exactly 1 batched tool_result message, got {}",
        tool_result_messages.len()
    );

    // That single message should contain both tool results
    if let Content::Blocks(blocks) = &tool_result_messages[0].content {
        let tool_result_count = blocks
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
            .count();
        assert_eq!(
            tool_result_count, 2,
            "Expected 2 ToolResult blocks in the batched message, got {tool_result_count}"
        );
    } else {
        panic!("Expected Blocks content in tool_result message");
    }

    Ok(())
}

// ===================
// Server Boundary Tests
// ===================

#[tokio::test]
async fn test_run_turn_is_direct_async_no_spawn() -> anyhow::Result<()> {
    // run_turn should execute directly in the caller's task (no tokio::spawn)
    // and only resolve after the event store turn has been finished.
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (outcome, events) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    let stored_turn = agent
        .event_store
        .get_turn(&thread_id, 1)
        .await?
        .context("expected persisted turn")?;
    assert!(matches!(outcome, TurnOutcome::Done { .. }));
    assert!(
        !events.is_empty(),
        "Expected events to reach the event store"
    );
    assert!(
        stored_turn.finished,
        "Expected finish_turn barrier to complete"
    );
    Ok(())
}

#[tokio::test]
async fn test_run_confirmation_resume_keeps_turn_open_until_resume_completes() -> anyhow::Result<()>
{
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("tool_1", "echo", json!({"message": "first"})),
            ("tool_2", "echo", json!({"message": "second"})),
        ]),
        MockProvider::text_response("Run complete"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .hooks(ConfirmAllHooks)
        .tools(tools)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::new();

    let state_1 = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("Need confirmation twice".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    let (continuation_1, tool_call_id_1) = match state_1 {
        AgentRunState::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected first run confirmation, got {other:?}"),
    };
    let first_turn = agent
        .event_store
        .get_turn(&thread_id, 1)
        .await?
        .context("missing open turn after first confirmation")?;
    assert!(!first_turn.finished);

    let state_2 = agent
        .run(
            thread_id.clone(),
            AgentInput::Resume {
                continuation: continuation_1,
                tool_call_id: tool_call_id_1,
                confirmed: true,
                rejection_reason: None,
            },
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    let (continuation_2, tool_call_id_2) = match state_2 {
        AgentRunState::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected second run confirmation, got {other:?}"),
    };
    let second_turn = agent
        .event_store
        .get_turn(&thread_id, 1)
        .await?
        .context("missing open turn after second confirmation")?;
    assert!(!second_turn.finished);

    let state_3 = agent
        .run(
            thread_id.clone(),
            AgentInput::Resume {
                continuation: continuation_2,
                tool_call_id: tool_call_id_2,
                confirmed: true,
                rejection_reason: None,
            },
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    assert!(matches!(state_3, AgentRunState::Done { .. }));
    let completed_turn = agent
        .event_store
        .get_turn(&thread_id, 1)
        .await?
        .context("missing completed turn after run finishes")?;
    assert!(completed_turn.finished);

    Ok(())
}

#[tokio::test]
async fn test_run_marks_done_turn_finished() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    assert!(matches!(state, AgentRunState::Done { .. }));
    assert!(
        agent
            .event_store
            .get_turn(&thread_id, 1)
            .await?
            .context("missing persisted run turn")?
            .finished
    );

    Ok(())
}

#[tokio::test]
async fn test_run_turn_returns_error_when_event_append_fails() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(FailingEventStore::new(EventStoreFailureMode::Append))
        .build();

    let outcome = Box::pin(agent.run_turn(
        ThreadId::new(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        TurnOptions::default(),
    ))
    .await;

    match outcome {
        TurnOutcome::Error(error) => {
            assert!(error.message.contains("Failed to append event"));
        }
        other => panic!("Expected append failure, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn test_run_returns_error_when_event_append_fails() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(FailingEventStore::new(EventStoreFailureMode::Append))
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    match state {
        AgentRunState::Error(error) => {
            assert!(error.message.contains("Failed to append event"));
        }
        other => panic!("Expected append failure, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn test_run_turn_returns_error_when_finish_turn_fails() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(FailingEventStore::new(EventStoreFailureMode::FinishTurn))
        .build();

    let outcome = Box::pin(agent.run_turn(
        ThreadId::new(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        TurnOptions::default(),
    ))
    .await;

    match outcome {
        TurnOutcome::Error(error) => {
            assert!(error.message.contains("Failed to finish turn event store"));
        }
        other => panic!("Expected finish failure, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn test_run_returns_error_when_finish_turn_fails() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(FailingEventStore::new(EventStoreFailureMode::FinishTurn))
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    match state {
        AgentRunState::Error(error) => {
            assert!(error.message.contains("Failed to finish turn event store"));
        }
        other => panic!("Expected finish failure, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn test_run_turn_repeated_init_failures_preserve_original_error() -> anyhow::Result<()> {
    let event_store = new_event_store();
    let thread_id = ThreadId::new();
    let agent = builder::<()>()
        .provider(MockProvider::new(vec![]))
        .hooks(AllowAllHooks)
        .message_store(FailingMessageStore)
        .state_store(InMemoryStore::new())
        .event_store(event_store.clone())
        .build_with_stores();

    for _ in 0..2 {
        let outcome = Box::pin(agent.run_turn(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
            TurnOptions::default(),
        ))
        .await;

        match outcome {
            TurnOutcome::Error(error) => {
                assert!(error.message.contains("message store unavailable"));
            }
            other => panic!("Expected initialization failure, got {other:?}"),
        }
    }

    assert!(event_store.get_turn(&thread_id, 0).await?.is_none());

    Ok(())
}

#[tokio::test]
async fn test_external_tool_runtime_returns_pending_tool_calls() -> anyhow::Result<()> {
    use crate::types::ToolRuntime;

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "test"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let options = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        options,
    )
    .await?;

    match outcome {
        TurnOutcome::PendingToolCalls {
            tool_calls,
            continuation,
            ..
        } => {
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].name, "echo");
            assert_eq!(tool_calls[0].id, "tool_1");
            // Continuation should reference the same thread
            assert!(!continuation.payload.thread_id.to_string().is_empty());
        }
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_external_tool_runtime_no_tools_returns_done() -> anyhow::Result<()> {
    use crate::types::ToolRuntime;

    // When there are no tool calls, External mode should behave like inline.
    let provider = MockProvider::new(vec![MockProvider::text_response("Just text")]);

    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let options = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        options,
    )
    .await?;

    assert!(
        matches!(outcome, TurnOutcome::Done { .. }),
        "Expected Done when no tool calls, got {outcome:?}"
    );
    Ok(())
}

#[tokio::test]
async fn terminal_stop_reason_ignores_tool_blocks_before_execution_and_persistence()
-> anyhow::Result<()> {
    use crate::llm::{ChatResponse, StopReason, Usage};

    let executions = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let provider = MockProvider::new(vec![ChatOutcome::Success(ChatResponse {
        id: "msg_terminal_tool".to_string(),
        content: vec![
            ContentBlock::Text {
                text: "I cannot continue.".to_string(),
            },
            ContentBlock::ToolUse {
                id: "tool_should_not_run".to_string(),
                name: "counter".to_string(),
                input: json!({ "tag": "unexpected" }),
                thought_signature: None,
            },
        ],
        model: "mock-model".to_string(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            input_tokens: 10,
            output_tokens: 20,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
    })]);

    let mut tools = ToolRegistry::new();
    tools.register(CountingTool {
        executions: Arc::clone(&executions),
    });
    let message_store = InMemoryStore::new();
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .tools(tools)
        .message_store(message_store.clone())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::new();

    let (outcome, events) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("do the thing".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    assert!(
        matches!(outcome, TurnOutcome::Done { .. }),
        "a terminal response must not hand off tools, got {outcome:?}"
    );
    {
        let executions = executions
            .lock()
            .map_err(|_| anyhow::anyhow!("execution counter lock poisoned"))?;
        assert!(
            executions.is_empty(),
            "a terminal response must not execute its tool blocks"
        );
        drop(executions);
    }
    assert!(
        !events.iter().any(|event| matches!(
            event.event,
            AgentEvent::ToolCallStart { .. } | AgentEvent::ToolCallEnd { .. }
        )),
        "a terminal response must not emit tool lifecycle events"
    );

    let history = message_store.get_history(&thread_id).await?;
    assert!(
        !history.iter().any(|message| matches!(
            &message.content,
            Content::Blocks(blocks)
                if blocks.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        )),
        "ignored tool blocks must not be persisted without results: {history:?}"
    );
    assert!(
        !crate::llm::has_unbalanced_tool_use(&history),
        "terminal response history must remain balanced: {history:?}"
    );

    Ok(())
}

#[tokio::test]
async fn test_strict_durability_saves_state_checkpoints() -> anyhow::Result<()> {
    // Verify strict durability mode runs without errors.
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "test"})),
        MockProvider::text_response("Done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let options = TurnOptions {
        tool_runtime: crate::types::ToolRuntime::Inline,
        strict_durability: true,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run with durability".to_string()),
        ToolContext::new(()),
        options,
    )
    .await?;

    // Should complete normally with NeedsMoreTurns (tool was executed inline).
    assert!(
        matches!(outcome, TurnOutcome::NeedsMoreTurns { .. }),
        "Expected NeedsMoreTurns, got {outcome:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_run_still_works_as_convenience_wrapper() -> anyhow::Result<()> {
    // Verify run() still works after the refactor.
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello from run()!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (state, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    let mut got_text = false;
    for envelope in events {
        if matches!(envelope.event, AgentEvent::Text { .. }) {
            got_text = true;
        }
    }
    assert!(got_text, "Expected a text event from run()");

    assert!(matches!(state, crate::types::AgentRunState::Done { .. }));

    Ok(())
}

// =========================================================================
// External tool handoff: SubmitToolResults regression tests
// =========================================================================

#[tokio::test]
async fn test_submit_tool_results_round_trip() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    // Turn 1: LLM requests a tool call → PendingToolCalls
    // Turn 2: We submit the result → LLM sees it → Done
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hello"})),
        MockProvider::text_response("Got the echo result"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let external_opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    // Turn 1: get PendingToolCalls
    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        external_opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls {
            tool_calls,
            continuation,
            ..
        } => {
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "tool_1");
            assert_eq!(tool_calls[0].name, "echo");
            continuation
        }
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Turn 2: submit external results → should produce Done
    let (outcome2, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("echo: hello"),
            }],
        },
        ToolContext::new(()),
        external_opts,
    )
    .await?;

    assert!(
        matches!(outcome2, TurnOutcome::Done { .. }),
        "Expected Done after submitting tool results, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_batch() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    // LLM requests two tool calls in one turn → submit both results
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("call_a", "echo", json!({"message": "first"})),
            ("call_b", "echo", json!({"message": "second"})),
        ]),
        MockProvider::text_response("Both results received"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run two tools".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls {
            tool_calls,
            continuation,
            ..
        } => {
            assert_eq!(tool_calls.len(), 2);
            continuation
        }
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit results (deliberately in reverse order to prove ordering doesn't matter)
    let (outcome2, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![
                ExternalToolResult {
                    tool_call_id: "call_b".to_string(),
                    result: ToolResult::success("second result"),
                },
                ExternalToolResult {
                    tool_call_id: "call_a".to_string(),
                    result: ToolResult::success("first result"),
                },
            ],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    assert!(
        matches!(outcome2, TurnOutcome::Done { .. }),
        "Expected Done after batch submit, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_thread_id_mismatch() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "test"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit on a DIFFERENT thread_id → should error
    let wrong_thread = ThreadId::new();
    let outcome2 = Box::pin(agent.run_turn(
        wrong_thread,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("echo"),
            }],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    ))
    .await;

    assert!(
        matches!(outcome2, TurnOutcome::Error(_)),
        "Expected Error for thread mismatch, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_missing_result() -> anyhow::Result<()> {
    use crate::types::ToolRuntime;

    let provider = MockProvider::new(vec![MockProvider::tool_uses_response(vec![
        ("call_a", "echo", json!({"message": "first"})),
        ("call_b", "echo", json!({"message": "second"})),
    ])]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run two tools".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit only one result for two tool calls → should error
    let outcome2 = Box::pin(agent.run_turn(
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![crate::types::ExternalToolResult {
                tool_call_id: "call_a".to_string(),
                result: crate::types::ToolResult::success("only one"),
            }],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    ))
    .await;

    assert!(
        matches!(outcome2, TurnOutcome::Error(_)),
        "Expected Error for missing result, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_unknown_tool_call_id() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "test"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit with wrong tool_call_id → should error
    let outcome2 = Box::pin(agent.run_turn(
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "unknown_id".to_string(),
                result: ToolResult::success("echo"),
            }],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    ))
    .await;

    assert!(
        matches!(outcome2, TurnOutcome::Error(_)),
        "Expected Error for unknown tool call ID, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_continuation_serializes() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    // Verify the continuation round-trips through JSON — the durable contract.
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "test"})),
        MockProvider::text_response("Got it"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Round-trip through JSON (simulates durable persistence)
    let json = serde_json::to_string(&continuation)?;
    let recovered: Box<ContinuationEnvelope> = serde_json::from_str(&json)?;
    assert_eq!(recovered.payload.thread_id, continuation.payload.thread_id);
    assert_eq!(recovered.payload.turn, continuation.payload.turn);
    assert_eq!(recovered.payload.pending_tool_calls.len(), 1);

    // Use the deserialized continuation to resume
    let (outcome2, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation: recovered,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("echo: test"),
            }],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    assert!(
        matches!(outcome2, TurnOutcome::Done { .. }),
        "Expected Done after deserialized continuation resume, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_with_failed_tool() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    // External runtime may report a tool failure — the SDK should still pass
    // it through to the LLM as an error result.
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "crash"})),
        MockProvider::text_response("I see the tool failed"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit a *failed* tool result
    let (outcome2, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::error("tool crashed"),
            }],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    assert!(
        matches!(outcome2, TurnOutcome::Done { .. }),
        "Expected Done even with failed tool, got {outcome2:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_submit_tool_results_duplicate_tool_call_id() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "test"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run tool".to_string()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Submit the same tool_call_id twice → should error
    let outcome2 = Box::pin(agent.run_turn(
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![
                ExternalToolResult {
                    tool_call_id: "tool_1".to_string(),
                    result: ToolResult::success("first"),
                },
                ExternalToolResult {
                    tool_call_id: "tool_1".to_string(),
                    result: ToolResult::success("second"),
                },
            ],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    ))
    .await;

    assert!(
        matches!(outcome2, TurnOutcome::Error(_)),
        "Expected Error for duplicate tool call ID, got {outcome2:?}"
    );
    Ok(())
}

// ===================
// Phase 1.5 Tests: ToolInvocation, Versioned Continuations, Resume-time Policy
// ===================

/// Verify that `ToolInvocation` carries all expected fields to the hooks.
#[tokio::test]
#[allow(clippy::significant_drop_tightening)]
async fn test_tool_invocation_carries_full_context() -> anyhow::Result<()> {
    use std::sync::Mutex;

    /// Hook that captures the `ToolInvocation` it receives.
    #[derive(Clone)]
    struct CapturingHooks {
        captured: Arc<Mutex<Vec<ToolInvocation>>>,
    }

    #[async_trait]
    impl AgentHooks for CapturingHooks {
        async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
            self.captured.lock().unwrap().push(invocation.clone());
            ToolDecision::Allow
        }
    }

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let hooks = CapturingHooks {
        captured: Arc::new(Mutex::new(Vec::new())),
    };
    let captured_ref = Arc::clone(&hooks.captured);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let _ = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run tool".into()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let captured = captured_ref.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let inv = &captured[0];
    assert_eq!(inv.tool_call_id, "tool_1");
    assert_eq!(inv.tool_name, "echo");
    assert_eq!(inv.display_name, "Echo");
    assert_eq!(inv.tier, ToolTier::Observe);
    assert_eq!(inv.requested_input, json!({"message": "hi"}));
    assert_eq!(inv.effective_input, json!({"message": "hi"}));
    assert!(inv.listen_context.is_none());
    Ok(())
}

/// Verify that continuations round-trip through JSON with version.
#[tokio::test]
async fn test_continuation_envelope_round_trip() -> anyhow::Result<()> {
    use crate::types::CONTINUATION_VERSION;

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "hi"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let opts = TurnOptions {
        tool_runtime: crate::types::ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run tool".into()),
        ToolContext::new(()),
        opts,
    )
    .await?;

    let envelope = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Verify version is set
    assert_eq!(envelope.version, CONTINUATION_VERSION);

    // Round-trip through JSON
    let json = serde_json::to_string(&envelope)?;
    let recovered: ContinuationEnvelope = serde_json::from_str(&json)?;
    assert_eq!(recovered.version, CONTINUATION_VERSION);
    assert_eq!(recovered.payload.pending_tool_calls.len(), 1);
    assert_eq!(recovered.payload.pending_tool_calls[0].name, "echo");

    // Validate unwrap succeeds
    let payload = recovered.unwrap_validated();
    assert!(payload.is_ok());
    Ok(())
}

/// Verify that an unknown continuation version is rejected at resume time.
#[tokio::test]
async fn test_unknown_continuation_version_rejected() -> anyhow::Result<()> {
    use crate::types::{
        AgentContinuation, AgentState, CONTINUATION_VERSION, PendingToolCallInfo, TokenUsage,
    };

    let provider = MockProvider::new(vec![MockProvider::text_response("Done")]);

    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let thread = ThreadId::new();

    // Construct an envelope with a future version
    let bad_envelope = ContinuationEnvelope {
        version: CONTINUATION_VERSION + 1,
        payload: AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_1".into(),
                name: "echo".into(),
                display_name: "Echo".into(),
                tier: ToolTier::Observe,
                input: serde_json::json!({}),
                effective_input: serde_json::json!({}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        },
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread,
        AgentInput::Resume {
            continuation: Box::new(bad_envelope),
            tool_call_id: "call_1".into(),
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    match outcome {
        TurnOutcome::Error(e) => {
            assert!(
                e.message.contains("Unsupported continuation version"),
                "Expected version error, got: {}",
                e.message,
            );
        }
        other => panic!("Expected Error for bad version, got {other:?}"),
    }
    Ok(())
}

/// Verify that resume-time Block from hooks authoritatively prevents execution.
#[tokio::test]
async fn test_resume_time_block_is_authoritative() -> anyhow::Result<()> {
    use std::sync::atomic::AtomicBool;

    /// Hooks that allow the first call (to get past the initial tool eval)
    /// then block on resume.
    struct BlockOnResumeHooks {
        first_call_done: AtomicBool,
    }

    #[async_trait]
    impl AgentHooks for BlockOnResumeHooks {
        async fn pre_tool_use(&self, _invocation: &ToolInvocation) -> ToolDecision {
            if self.first_call_done.swap(true, Ordering::SeqCst) {
                // Second call (resume) -> block
                ToolDecision::Block("policy changed".into())
            } else {
                // First call -> require confirmation (so we get a continuation)
                ToolDecision::RequiresConfirmation("confirm?".into())
            }
        }
    }

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let hooks = BlockOnResumeHooks {
        first_call_done: AtomicBool::new(false),
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let thread_id = ThreadId::new();

    // First turn: should yield AwaitingConfirmation
    let (outcome1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run echo".into()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (continuation, tool_call_id) = match outcome1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    // Resume with confirmation — but hooks now return Block
    let (outcome2, events) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    // The turn should still complete (the blocked tool result is fed back to LLM)
    // but the tool result should indicate it was blocked.
    match outcome2 {
        TurnOutcome::Done { .. } | TurnOutcome::NeedsMoreTurns { .. } => {
            // Check that a ToolCallEnd event was emitted with a failure
            let has_blocked_result = events.iter().any(|e| {
                matches!(&e.event, AgentEvent::ToolCallEnd { result, .. } if !result.success && result.output.contains("Blocked at resume"))
            });
            assert!(
                has_blocked_result,
                "Expected a ToolCallEnd with 'Blocked at resume' message"
            );
        }
        other => panic!("Expected Done/NeedsMoreTurns after blocked resume, got {other:?}"),
    }
    Ok(())
}

/// Verify that `effective_input` on `PendingToolCallInfo` defaults to match `input`.
#[tokio::test]
async fn test_effective_input_defaults_to_requested() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "test"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let opts = TurnOptions {
        tool_runtime: crate::types::ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Run tool".into()),
        ToolContext::new(()),
        opts,
    )
    .await?;

    let tool_calls = match outcome {
        TurnOutcome::PendingToolCalls { tool_calls, .. } => tool_calls,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].input, tool_calls[0].effective_input);
    assert_eq!(tool_calls[0].input, json!({"message": "test"}));
    Ok(())
}

// ==========================================================================
// Phase 1.6 Tests: ToolAuditSink and Full Lifecycle Audit Records
//
// Each test drives the agent loop through a specific lifecycle transition
// and asserts that the recording audit sink captured exactly the expected
// outcome variant(s). Tests favor explicit assertions on `outcome_kind()`
// so a regression anywhere in the lifecycle wiring surfaces as a clean
// failure.
// ==========================================================================

use agent_sdk_foundation::audit::{ToolAuditOutcome, ToolAuditRecord};
use tokio::sync::Mutex;

/// Test-only audit sink that captures every record in memory.
///
/// Uses `tokio::sync::Mutex` so `.lock().await` is infallible. Per
/// `CLAUDE.md` no `.unwrap()` is allowed, even in tests.
#[derive(Default)]
struct RecordingAuditSink {
    records: Mutex<Vec<ToolAuditRecord>>,
}

#[async_trait]
impl crate::hooks::ToolAuditSink for RecordingAuditSink {
    async fn record(&self, record: ToolAuditRecord) {
        self.records.lock().await.push(record);
    }
}

impl RecordingAuditSink {
    async fn kinds(&self) -> Vec<&'static str> {
        self.records
            .lock()
            .await
            .iter()
            .map(ToolAuditRecord::outcome_kind)
            .collect()
    }

    async fn records(&self) -> Vec<ToolAuditRecord> {
        self.records.lock().await.clone()
    }
}

/// Hook that returns a fixed decision for every invocation.
struct FixedDecisionHooks {
    decision: Mutex<ToolDecision>,
}

impl FixedDecisionHooks {
    fn new(decision: ToolDecision) -> Self {
        Self {
            decision: Mutex::new(decision),
        }
    }
}

#[async_trait]
impl AgentHooks for FixedDecisionHooks {
    async fn pre_tool_use(&self, _invocation: &ToolInvocation) -> ToolDecision {
        self.decision.lock().await.clone()
    }
}

/// Message store that fails the Nth call to `append`, then succeeds.
///
/// Used by audit-durability regression tests that need to exercise a
/// transient message-store failure at a specific call site (for
/// example, "fail only the external tool-result append").
struct FailAtAppendStore {
    inner: Arc<crate::stores::InMemoryStore>,
    appends_seen: std::sync::atomic::AtomicUsize,
    fail_on: usize,
}

impl FailAtAppendStore {
    fn new(fail_on: usize) -> Self {
        Self {
            inner: Arc::new(crate::stores::InMemoryStore::new()),
            appends_seen: std::sync::atomic::AtomicUsize::new(0),
            fail_on,
        }
    }
}

#[async_trait]
impl MessageStore for FailAtAppendStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        message: crate::llm::Message,
    ) -> anyhow::Result<()> {
        let idx = self.appends_seen.fetch_add(1, Ordering::SeqCst);
        if idx == self.fail_on {
            anyhow::bail!("synthetic message store failure at append #{idx}");
        }
        self.inner.append(thread_id, message).await
    }
    async fn get_history(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<crate::llm::Message>> {
        self.inner.get_history(thread_id).await
    }
    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<crate::llm::Message>,
    ) -> anyhow::Result<()> {
        self.inner.replace_history(thread_id, messages).await
    }
    async fn clear(&self, thread_id: &ThreadId) -> anyhow::Result<()> {
        self.inner.clear(thread_id).await
    }
}

#[tokio::test]
async fn test_audit_emits_completed_on_local_sync_tool_success() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let _ = run_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
    )
    .await?;

    let kinds = sink.kinds().await;
    assert_eq!(
        kinds,
        vec!["completed"],
        "Local sync tool success should emit exactly one Completed record"
    );
    let records = sink.records().await;
    let record = &records[0];
    assert_eq!(record.tool_call_id, "tool_1");
    assert_eq!(record.tool_name, "echo");
    assert_eq!(record.provenance.provider, "mock");
    assert_eq!(record.provenance.model, "mock-model");
    match &record.outcome {
        ToolAuditOutcome::Completed { result } => assert!(result.success),
        other => panic!("Expected Completed outcome, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_blocked_when_policy_blocks_tool() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());
    let hooks = FixedDecisionHooks::new(ToolDecision::Block("policy denied".into()));

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build_with_stores();

    let _ = run_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
    )
    .await?;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"blocked"),
        "Expected a Blocked record, got {kinds:?}"
    );
    let records = sink.records().await;
    let blocked = records
        .iter()
        .find(|r| matches!(r.outcome, ToolAuditOutcome::Blocked { .. }))
        .expect("missing blocked record");
    match &blocked.outcome {
        ToolAuditOutcome::Blocked { reason } => assert_eq!(reason, "policy denied"),
        other => panic!("Expected Blocked, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_requires_confirmation_when_policy_yields() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "hi"}),
    )]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());
    let hooks = FixedDecisionHooks::new(ToolDecision::RequiresConfirmation("please?".into()));

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build_with_stores();

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    assert!(
        matches!(outcome, TurnOutcome::AwaitingConfirmation { .. }),
        "Expected AwaitingConfirmation, got {outcome:?}"
    );
    let kinds = sink.kinds().await;
    assert_eq!(
        kinds,
        vec!["requires_confirmation"],
        "Expected exactly one RequiresConfirmation record"
    );
    let records = sink.records().await;
    match &records[0].outcome {
        ToolAuditOutcome::RequiresConfirmation { description, .. } => {
            assert_eq!(description, "please?");
        }
        other => panic!("Expected RequiresConfirmation, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_cached_on_idempotent_replay() -> anyhow::Result<()> {
    // Pre-seed the execution store with a completed record for tool_1 so
    // that the next turn's execution short-circuits via the cached path.
    use crate::stores::InMemoryExecutionStore;
    use crate::types::ToolExecution;

    let execution_store = Arc::new(InMemoryExecutionStore::new());
    let thread_id = ThreadId::new();
    let mut execution = ToolExecution::new_in_flight(
        "tool_1",
        thread_id.clone(),
        "echo",
        "Echo",
        json!({"message": "hi"}),
        time::OffsetDateTime::now_utc(),
    );
    execution.complete(ToolResult::success("cached output"));
    execution_store.record_execution(execution).await?;

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .execution_store_shared(
            Arc::clone(&execution_store) as Arc<dyn crate::stores::ToolExecutionStore>
        )
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let _ = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("run".into()),
        ToolContext::new(()),
    )
    .await?;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"cached"),
        "Expected a Cached record, got {kinds:?}"
    );
    let records = sink.records().await;
    let cached = records
        .iter()
        .find(|r| matches!(r.outcome, ToolAuditOutcome::Cached { .. }))
        .expect("missing cached record");
    match &cached.outcome {
        ToolAuditOutcome::Cached { result } => assert_eq!(result.output, "cached output"),
        other => panic!("Expected Cached, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_invalidated_on_listen_tool_invalidation() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "listen_echo", json!({"message": "q"})),
        MockProvider::text_response("After invalidation"),
    ]);
    let cancel_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register_listen(ScenarioListenTool {
        updates: vec![ListenToolUpdate::Invalidated {
            operation_id: "listen-op-1".to_string(),
            message: "quote expired".to_string(),
            recoverable: true,
        }],
        execute_error: None,
        cancel_calls: cancel_calls.clone(),
    });
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let _ = run_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("run listen".into()),
        ToolContext::new(()),
    )
    .await?;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"invalidated"),
        "Expected Invalidated record, got {kinds:?}"
    );
    let records = sink.records().await;
    let invalidated = records
        .iter()
        .find(|r| matches!(r.outcome, ToolAuditOutcome::Invalidated { .. }))
        .expect("missing invalidated record");
    match &invalidated.outcome {
        ToolAuditOutcome::Invalidated { reason } => assert!(reason.contains("invalidated")),
        other => panic!("Expected Invalidated, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_completed_on_external_tool_submission() -> anyhow::Result<()> {
    use crate::types::{ExternalToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Got the echo result"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    let _ = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("external"),
            }],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"completed"),
        "External submission should emit Completed, got {kinds:?}"
    );
    let records = sink.records().await;
    let completed = records
        .iter()
        .find(|r| {
            matches!(
                &r.outcome,
                ToolAuditOutcome::Completed { result } if result.output == "external"
            )
        })
        .expect("missing completed record for external tool");
    assert_eq!(completed.tool_call_id, "tool_1");
    assert_eq!(completed.tool_name, "echo");
    // Tier must be the echo tool's actual tier (Observe), not a
    // hardcoded fallback. Previously this path always emitted Observe
    // regardless of the tool's real tier.
    assert_eq!(completed.tier, ToolTier::Observe);
    Ok(())
}

/// Regression test for the critical bug where
/// `emit_external_tool_audit` hardcoded `ToolTier::Observe` for every
/// external-runtime audit record.
///
/// A confirm-tier tool flows through the external submit path and the
/// resulting `Completed` record MUST carry `ToolTier::Confirm` — not
/// a silently downgraded `Observe` — so that servers enforcing
/// tier-based compliance invariants see the authoritative tier.
#[tokio::test]
async fn test_audit_external_submission_preserves_confirm_tier() -> anyhow::Result<()> {
    use crate::tools::Tool;
    use crate::types::{ExternalToolResult, ToolRuntime};

    /// Confirm-tier tool for regression testing the external-path tier
    /// propagation. Not shared via `test_utils` because no other test
    /// needs a confirm-tier tool.
    struct WriteFileTool;

    impl Tool<()> for WriteFileTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::Echo
        }

        fn display_name(&self) -> &'static str {
            "Write File"
        }

        fn description(&self) -> &'static str {
            "Confirm-tier tool for tier-propagation regression tests"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn tier(&self) -> ToolTier {
            ToolTier::Confirm
        }

        async fn execute(
            &self,
            _ctx: &ToolContext<()>,
            _input: serde_json::Value,
        ) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success("wrote"))
        }
    }

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"path": "/tmp/x"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(WriteFileTool);
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;
    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    let _ = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("wrote"),
            }],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    let records = sink.records().await;
    let completed = records
        .iter()
        .find(|r| matches!(&r.outcome, ToolAuditOutcome::Completed { .. }))
        .expect("missing completed record");
    assert_eq!(
        completed.tier,
        ToolTier::Confirm,
        "External path must preserve the tool's actual tier, got {:?}",
        completed.tier
    );
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_replayed_on_duplicate_external_submission() -> anyhow::Result<()> {
    use crate::stores::InMemoryExecutionStore;
    use crate::types::{ExternalToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done A"),
        MockProvider::text_response("Done B"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());
    let execution_store = Arc::new(InMemoryExecutionStore::new());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .execution_store_shared(
            Arc::clone(&execution_store) as Arc<dyn crate::stores::ToolExecutionStore>
        )
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    // Turn 1: LLM requests tool → pending
    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;
    let continuation = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Turn 2: submit result (first time → Completed)
    let _ = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::SubmitToolResults {
            continuation: continuation.clone(),
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("external"),
            }],
        },
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;

    // Turn 3: re-submit the same continuation + result → should be detected
    // as a replay via the execution store.
    let _ = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("external"),
            }],
        },
        ToolContext::new(()),
        opts,
    )
    .await?;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"replayed"),
        "Duplicate external submission should emit Replayed, got {kinds:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_blocked_at_resume_when_policy_rejects() -> anyhow::Result<()> {
    use std::sync::atomic::AtomicBool;

    struct BlockOnResumeHooks {
        first_call_done: AtomicBool,
    }

    #[async_trait]
    impl AgentHooks for BlockOnResumeHooks {
        async fn pre_tool_use(&self, _invocation: &ToolInvocation) -> ToolDecision {
            if self.first_call_done.swap(true, Ordering::SeqCst) {
                ToolDecision::Block("policy changed".into())
            } else {
                ToolDecision::RequiresConfirmation("confirm?".into())
            }
        }
    }

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let hooks = BlockOnResumeHooks {
        first_call_done: AtomicBool::new(false),
    };
    let sink = Arc::new(RecordingAuditSink::default());

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build_with_stores();

    let thread_id = ThreadId::new();
    let (outcome1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    let (continuation, tool_call_id) = match outcome1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let _ = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let records = sink.records().await;
    // We expect at least two lifecycle transitions for tool_1:
    //   1. RequiresConfirmation (first turn)
    //   2. Blocked (second turn, at resume time)
    let kinds_for_tool_1: Vec<_> = records
        .iter()
        .filter(|r| r.tool_call_id == "tool_1")
        .map(ToolAuditRecord::outcome_kind)
        .collect();
    assert!(
        kinds_for_tool_1.contains(&"requires_confirmation"),
        "Expected RequiresConfirmation record, got {kinds_for_tool_1:?}"
    );
    assert!(
        kinds_for_tool_1.contains(&"blocked"),
        "Expected Blocked record at resume, got {kinds_for_tool_1:?}"
    );
    let blocked = records
        .iter()
        .find(|r| matches!(r.outcome, ToolAuditOutcome::Blocked { .. }))
        .expect("missing blocked record");
    match &blocked.outcome {
        ToolAuditOutcome::Blocked { reason } => assert_eq!(reason, "policy changed"),
        other => panic!("Expected Blocked, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn test_audit_emits_persistence_failed_when_tool_event_append_fails() -> anyhow::Result<()> {
    // A custom event store that lets the first N appends through and
    // then fails. With N=1 the turn-start event succeeds but the
    // first tool-related append (`tool_call_start`) fails, exercising
    // the audit `PersistenceFailed` path inside the local tool runtime.
    use std::sync::atomic::AtomicUsize;

    struct FailAfterAppendStore {
        inner: Arc<InMemoryEventStore>,
        appends_seen: AtomicUsize,
        appends_allowed: usize,
    }

    #[async_trait]
    impl EventStore for FailAfterAppendStore {
        async fn append(
            &self,
            thread_id: &ThreadId,
            turn: usize,
            envelope: AgentEventEnvelope,
        ) -> anyhow::Result<()> {
            let prior = self.appends_seen.fetch_add(1, Ordering::SeqCst);
            if prior >= self.appends_allowed {
                anyhow::bail!("synthetic append failure on call #{}", prior + 1);
            }
            self.inner.append(thread_id, turn, envelope).await
        }
        async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> anyhow::Result<()> {
            self.inner.finish_turn(thread_id, turn).await
        }
        async fn get_turn(
            &self,
            thread_id: &ThreadId,
            turn: usize,
        ) -> anyhow::Result<Option<StoredTurnEvents>> {
            self.inner.get_turn(thread_id, turn).await
        }
        async fn get_turns(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<StoredTurnEvents>> {
            self.inner.get_turns(thread_id).await
        }
        async fn clear(&self, thread_id: &ThreadId) -> anyhow::Result<()> {
            self.inner.clear(thread_id).await
        }
    }

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());

    // Allow exactly 1 append (turn start), then fail. The next append is
    // `tool_call_start`, which surfaces as an audit `PersistenceFailed`.
    let event_store: Arc<dyn EventStore> = Arc::new(FailAfterAppendStore {
        inner: Arc::new(InMemoryEventStore::new()),
        appends_seen: AtomicUsize::new(0),
        appends_allowed: 1,
    });

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(event_store)
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build();

    let _ = run_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
    )
    .await;

    let kinds = sink.kinds().await;
    assert!(
        kinds.contains(&"persistence_failed"),
        "Failing tool-event append should emit PersistenceFailed, got {kinds:?}"
    );
    let records = sink.records().await;
    let pf = records
        .iter()
        .find(|r| matches!(r.outcome, ToolAuditOutcome::PersistenceFailed { .. }))
        .expect("missing persistence_failed record");
    match &pf.outcome {
        ToolAuditOutcome::PersistenceFailed { error, .. } => {
            assert!(error.contains("synthetic append failure"), "{error}");
        }
        other => panic!("Expected PersistenceFailed, got {other:?}"),
    }
    Ok(())
}

/// Regression test for the critical durability-ordering bug: if the
/// message store `append_tool_results` fails on the external runtime
/// path, a retry with the same continuation must re-evaluate every
/// submission as fresh (Completed) rather than short-circuit as
/// Replayed — otherwise the conversation is permanently broken because
/// the message store would never get the tool result.
#[tokio::test]
async fn test_audit_external_append_failure_does_not_mark_execution_store() -> anyhow::Result<()> {
    use crate::stores::InMemoryExecutionStore;
    use crate::types::{ExternalToolResult, ToolRuntime};

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let sink = Arc::new(RecordingAuditSink::default());
    let execution_store = Arc::new(InMemoryExecutionStore::new());

    // append sequence on the external path:
    //   0: Text("run") on turn 1
    //   1: assistant ToolUse message (persisted by agent loop)
    //   2: tool results batch on SubmitToolResults (this is the one we fail)
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(FailAtAppendStore::new(2))
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .execution_store_shared(
            Arc::clone(&execution_store) as Arc<dyn crate::stores::ToolExecutionStore>
        )
        .audit_sink_shared(Arc::clone(&sink) as Arc<dyn crate::hooks::ToolAuditSink>)
        .build_with_stores();

    let thread_id = ThreadId::new();
    let opts = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: false,
    };

    // Turn 1: LLM requests tool → pending
    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("run".into()),
        ToolContext::new(()),
        opts.clone(),
    )
    .await?;
    let continuation: Box<ContinuationEnvelope> = match outcome {
        TurnOutcome::PendingToolCalls { continuation, .. } => continuation,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // Snapshot the continuation so we can retry the same logical
    // submission twice without the envelope being consumed.
    let retry_continuation = continuation.clone();

    // Turn 2: submit results — the first message-store append fails,
    // which should emit Completed + PersistenceFailed audit records and
    // leave the execution store clean.
    let outcome2 = Box::pin(agent.run_turn(
        thread_id.clone(),
        AgentInput::SubmitToolResults {
            continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("external"),
            }],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts.clone(),
    ))
    .await;
    assert!(
        matches!(outcome2, TurnOutcome::Error(_)),
        "Expected Error from failed append, got {outcome2:?}"
    );

    // The execution store must NOT have been marked completed, so a
    // retry is re-evaluated as fresh.
    let exec =
        <dyn crate::stores::ToolExecutionStore>::get_execution(&*execution_store, "tool_1").await?;
    assert!(
        !exec.is_some_and(|e| e.is_completed()),
        "Execution store must not mark tool_1 as completed before append succeeds"
    );

    // Retry turn: the same continuation and same result, but this
    // time the message store succeeds. The audit sink must emit a
    // *fresh* Completed record (not Replayed) because the first
    // attempt never persisted.
    let _ = Box::pin(agent.run_turn(
        thread_id,
        AgentInput::SubmitToolResults {
            continuation: retry_continuation,
            results: vec![ExternalToolResult {
                tool_call_id: "tool_1".to_string(),
                result: ToolResult::success("external"),
            }],
        },
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    ))
    .await;

    let kinds = sink.kinds().await;
    // Expected sequence over both attempts:
    //   attempt 1: Completed, PersistenceFailed
    //   attempt 2: Completed (NOT Replayed)
    let completed_count = kinds.iter().filter(|k| **k == "completed").count();
    let replayed_count = kinds.iter().filter(|k| **k == "replayed").count();
    assert!(
        completed_count >= 2,
        "Expected at least two Completed records across attempts, got {kinds:?}"
    );
    assert_eq!(
        replayed_count, 0,
        "Retry after failure must not emit Replayed, got {kinds:?}"
    );
    Ok(())
}

// ===================
// TurnSummary Regression Suite (Phase 1.7)
// ===================
//
// These tests close Phase 1 by exercising the full server-facing
// `TurnSummary` contract on top of the earlier server-boundary work.
// Each test builds an end-to-end `run_turn` call
// with a `MockProvider` whose response fields are known, then asserts
// the resulting `TurnSummary` carries the provenance, response id,
// stop reason, usage, duration, tool-call count, and option flags that
// later server phases depend on.

#[tokio::test]
async fn test_turn_summary_done_populates_provenance_and_response_id() -> anyhow::Result<()> {
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![MockProvider::text_response("Done!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let options = TurnOptions {
        tool_runtime: ToolRuntime::Inline,
        strict_durability: false,
    };
    let thread_id = ThreadId::from_string("t-summary-done");

    let (outcome, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        options,
    )
    .await?;

    let summary: TurnSummary = match outcome {
        TurnOutcome::Done { summary, .. } => summary,
        other => panic!("Expected Done, got {other:?}"),
    };

    assert_eq!(summary.thread_id, thread_id);
    assert_eq!(summary.provenance.provider, "mock");
    assert_eq!(summary.provenance.model, "mock-model");
    assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
    assert_eq!(summary.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(summary.tool_call_count, 0);
    assert_eq!(summary.turn_usage.input_tokens, 10);
    assert_eq!(summary.turn_usage.output_tokens, 20);
    assert_eq!(summary.total_usage.input_tokens, 10);
    assert_eq!(summary.total_usage.output_tokens, 20);
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);
    assert!(!summary.strict_durability);
    assert!(summary.turn >= 1);
    assert!(summary.total_turns >= 1);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_needs_more_turns_reports_tool_call_count() -> anyhow::Result<()> {
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    // Two LLM calls: first asks for an echo tool, second wraps up.
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("All done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Echo please".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let summary: TurnSummary = match outcome {
        TurnOutcome::NeedsMoreTurns { summary, .. } => summary,
        other => panic!("Expected NeedsMoreTurns, got {other:?}"),
    };

    // Phase 1 contract: the tool-call count must match the LLM's request
    // and the stop reason must reflect the turn-closing LLM call.
    assert_eq!(summary.tool_call_count, 1);
    assert_eq!(summary.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
    assert_eq!(summary.provenance.provider, "mock");
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_pending_tool_calls_carries_external_runtime() -> anyhow::Result<()> {
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![MockProvider::tool_uses_response(vec![
        ("tool_1", "echo", json!({"message": "one"})),
        ("tool_2", "echo", json!({"message": "two"})),
    ])]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .event_store(new_event_store())
        .build();

    let options = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: true,
    };

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::from_string("t-pending-summary"),
        AgentInput::Text("Run two echos".to_string()),
        ToolContext::new(()),
        options,
    )
    .await?;

    let summary: TurnSummary = match outcome {
        TurnOutcome::PendingToolCalls { summary, .. } => summary,
        other => panic!("Expected PendingToolCalls, got {other:?}"),
    };

    // The summary for an external-runtime handoff must carry every
    // field later phases use to build a durable turn row.
    assert_eq!(summary.tool_call_count, 2);
    assert_eq!(summary.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(summary.tool_runtime, ToolRuntime::External);
    assert!(summary.strict_durability);
    assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
    assert_eq!(summary.provenance.model, "mock-model");
    assert_eq!(summary.turn_usage.input_tokens, 10);
    assert_eq!(summary.turn_usage.output_tokens, 20);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_is_serializable_for_durable_persistence() -> anyhow::Result<()> {
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![MockProvider::text_response("Hello")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let summary: TurnSummary = match outcome {
        TurnOutcome::Done { summary, .. } => summary,
        other => panic!("Expected Done, got {other:?}"),
    };

    // Round-trip through JSON because the server persists summaries as
    // part of durable turn rows.
    let json = serde_json::to_string(&summary)?;
    let recovered: TurnSummary = serde_json::from_str(&json)?;
    assert_eq!(recovered, summary);

    // Deserialize via serde_json::Value to make sure the wire contract
    // uses the stable snake_case discriminants later phases will match on.
    let value: serde_json::Value = serde_json::from_str(&json)?;
    assert_eq!(value["tool_runtime"], serde_json::json!("inline"));
    assert_eq!(value["stop_reason"], serde_json::json!("end_turn"));
    assert_eq!(value["provenance"]["provider"], serde_json::json!("mock"));

    // Sanity: this summary was produced by an inline-runtime turn.
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_awaiting_confirmation_carries_provenance() -> anyhow::Result<()> {
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![MockProvider::tool_use_response(
        "tool_1",
        "echo",
        json!({"message": "confirm me"}),
    )]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(ConfirmAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::from_string("t-await-summary"),
        AgentInput::Text("Echo please".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let summary: TurnSummary = match outcome {
        TurnOutcome::AwaitingConfirmation { summary, .. } => summary,
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    // The yield path still needs provenance for durable confirmation rows.
    assert_eq!(summary.provenance.provider, "mock");
    assert_eq!(summary.provenance.model, "mock-model");
    assert_eq!(summary.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_cancelled_before_llm_still_records_provenance() -> anyhow::Result<()> {
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![MockProvider::text_response("unused")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    // Pre-cancel the token so the turn trips the early-cancellation path.
    let cancel = CancellationToken::new();
    cancel.cancel();

    let outcome = Box::pin(agent.run_turn(
        ThreadId::from_string("t-cancelled-summary"),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        cancel,
        TurnOptions::default(),
    ))
    .await;

    let summary: TurnSummary = match outcome {
        TurnOutcome::Cancelled { summary, .. } => summary,
        other => panic!("Expected Cancelled, got {other:?}"),
    };

    // Even the early-cancellation path has to emit a summary so the
    // server can record a durable row for the cancelled turn.
    assert_eq!(summary.provenance.provider, "mock");
    assert_eq!(summary.provenance.model, "mock-model");
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);
    assert!(summary.response_id.is_none());
    assert!(summary.stop_reason.is_none());
    assert_eq!(summary.tool_call_count, 0);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_duration_ms_reflects_turn_wall_clock() -> anyhow::Result<()> {
    use crate::types::TurnSummary;

    let provider = MockProvider::new(vec![MockProvider::text_response("quick")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let start = std::time::Instant::now();
    let (outcome, _) = run_turn_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("hello".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;
    let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let summary: TurnSummary = match outcome {
        TurnOutcome::Done { summary, .. } => summary,
        other => panic!("Expected Done, got {other:?}"),
    };

    // Allow a generous upper bound for flaky CI; the important
    // property is that duration is recorded and bounded, not exact.
    assert!(
        summary.duration_ms <= elapsed_ms + 50,
        "summary duration {} ms must not exceed observed elapsed {} ms",
        summary.duration_ms,
        elapsed_ms,
    );
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_resume_after_confirmation_carries_summary() -> anyhow::Result<()> {
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("tool_1", "echo", json!({"message": "resume"})),
        MockProvider::text_response("done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(ConfirmAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let thread_id = ThreadId::from_string("t-resume-summary");

    // Turn 1: yields for confirmation
    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Echo please".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (tool_call_id, continuation, yield_summary) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            tool_call_id,
            continuation,
            summary,
            ..
        } => {
            // Yield summary should carry provenance and the real LLM
            // metadata from the turn-closing call that asked for the
            // echo tool.
            assert_eq!(summary.provenance.provider, "mock");
            assert_eq!(summary.tool_call_count, 1);
            assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
            assert_eq!(summary.stop_reason, Some(StopReason::ToolUse));
            (tool_call_id, continuation, summary)
        }
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    // Turn 2: resume with confirmation → NeedsMoreTurns (tool ran inline)
    let (outcome_2, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let summary: TurnSummary = match outcome_2 {
        TurnOutcome::NeedsMoreTurns { summary, .. } => summary,
        other => panic!("Expected NeedsMoreTurns, got {other:?}"),
    };

    // The resume summary still has to carry provenance so server
    // phases can join it to the original turn row.
    assert_eq!(summary.provenance.provider, "mock");
    assert_eq!(summary.provenance.model, "mock-model");
    assert_eq!(summary.tool_runtime, ToolRuntime::Inline);

    // Phase 1.7 regression guard: the resume-side summary
    // must carry the same turn-closing LLM metadata as the pre-pause
    // summary for the same turn. These fields used to be fabricated
    // as `None` / `0` on the resume path because the handler built a
    // synthetic `TurnContext` instead of threading real data through
    // `process_resume`.
    assert_eq!(
        summary.response_id, yield_summary.response_id,
        "resume summary must carry the pre-pause response id",
    );
    assert_eq!(
        summary.stop_reason, yield_summary.stop_reason,
        "resume summary must carry the pre-pause stop reason",
    );
    assert_eq!(
        summary.tool_call_count, yield_summary.tool_call_count,
        "resume summary must report the same tool call count as the pre-pause summary",
    );
    assert_eq!(summary.response_id.as_deref(), Some("msg_1"));
    assert_eq!(summary.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(summary.tool_call_count, 1);
    Ok(())
}

#[tokio::test]
async fn test_turn_summary_resume_nested_confirmation_preserves_llm_metadata() -> anyhow::Result<()>
{
    use crate::llm::StopReason;
    use crate::types::{ToolRuntime, TurnSummary};

    // Single LLM call produces two tool uses, both requiring
    // confirmation. Each confirmation round reports a `TurnSummary`
    // that must carry the same turn-closing LLM metadata as the
    // initial yield summary — this guards the nested
    // `AwaitingConfirmation` branch of `process_resume`.
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("tool_1", "echo", json!({"message": "one"})),
            ("tool_2", "echo", json!({"message": "two"})),
        ]),
        MockProvider::text_response("done"),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(ConfirmAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::from_string("t-resume-nested-summary");

    // Turn 1: first LLM call yields on tool_1.
    let (outcome_1, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("Run two tools".to_string()),
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (tool_call_id_1, continuation_1, yield_summary_1) =
        expect_awaiting_confirmation(outcome_1, "first AwaitingConfirmation");
    assert_eq!(yield_summary_1.tool_call_count, 2);
    assert_eq!(yield_summary_1.response_id.as_deref(), Some("msg_1"));
    assert_eq!(yield_summary_1.stop_reason, Some(StopReason::ToolUse));

    // Turn 1 resume: confirming tool_1 must yield again on tool_2.
    let (outcome_2, _) = run_turn_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Resume {
            continuation: continuation_1,
            tool_call_id: tool_call_id_1,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let (tool_call_id_2, continuation_2, nested_summary) =
        expect_awaiting_confirmation(outcome_2, "second AwaitingConfirmation");

    // Phase 1.7 regression guard: the nested-resume
    // summary must carry the pre-pause LLM metadata that was snapshotted
    // into the continuation on the first yield.
    assert_summary_llm_metadata_matches(&nested_summary, &yield_summary_1, "nested resume");
    assert_eq!(nested_summary.tool_runtime, ToolRuntime::Inline);
    assert_eq!(nested_summary.provenance.provider, "mock");

    // Turn 1 final resume: confirming tool_2 completes the tool phase.
    let (outcome_3, _) = run_turn_recorded(
        &agent,
        thread_id,
        AgentInput::Resume {
            continuation: continuation_2,
            tool_call_id: tool_call_id_2,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
        TurnOptions::default(),
    )
    .await?;

    let completed_summary: TurnSummary = match outcome_3 {
        TurnOutcome::NeedsMoreTurns { summary, .. } => summary,
        other => panic!("Expected NeedsMoreTurns, got {other:?}"),
    };
    assert_summary_llm_metadata_matches(&completed_summary, &yield_summary_1, "completed resume");
    Ok(())
}

/// Destructure a [`TurnOutcome::AwaitingConfirmation`] into its
/// interesting pieces for resume-path tests.
///
/// Panics with `context` if the outcome is any other variant — tests
/// use it as a one-line assertion + extraction helper.
fn expect_awaiting_confirmation(
    outcome: crate::types::TurnOutcome,
    context: &str,
) -> (
    String,
    Box<crate::types::ContinuationEnvelope>,
    crate::types::TurnSummary,
) {
    match outcome {
        crate::types::TurnOutcome::AwaitingConfirmation {
            tool_call_id,
            continuation,
            summary,
            ..
        } => (tool_call_id, continuation, summary),
        other => panic!("Expected {context}, got {other:?}"),
    }
}

/// Assert that two [`TurnSummary`]s carry identical turn-closing LLM
/// metadata. Used by the resume-path regression suite to guard the
/// pause/resume round-trip against regressions.
fn assert_summary_llm_metadata_matches(
    got: &crate::types::TurnSummary,
    expected: &crate::types::TurnSummary,
    label: &str,
) {
    assert_eq!(
        got.response_id, expected.response_id,
        "{label} must preserve the turn-closing response id",
    );
    assert_eq!(
        got.stop_reason, expected.stop_reason,
        "{label} must preserve the turn-closing stop reason",
    );
    assert_eq!(
        got.tool_call_count, expected.tool_call_count,
        "{label} must report the same tool call count as the pre-pause summary",
    );
}

// =====================================================================
// core-loop regression fixtures
// =====================================================================

/// A context compactor that replaces the whole history with a single short
/// summary message and never calls the provider — so overflow-recovery tests
/// don't have to interleave a summarization round-trip into the mock script.
struct ShrinkCompactor;

#[async_trait]
impl ContextCompactor for ShrinkCompactor {
    async fn compact(&self, _messages: &[Message]) -> anyhow::Result<String> {
        Ok("[summary]".to_string())
    }

    fn estimate_tokens(&self, _messages: &[Message]) -> usize {
        0
    }

    fn needs_compaction(&self, _messages: &[Message]) -> bool {
        // Threshold compaction never fires; only the explicit overflow path
        // (which calls `compact_history` directly) exercises this compactor.
        false
    }

    async fn compact_history(&self, messages: Vec<Message>) -> anyhow::Result<CompactionResult> {
        let original_count = messages.len();
        Ok(CompactionResult {
            messages: vec![Message::user("[summary]")],
            original_count,
            new_count: 1,
            original_tokens: 1_000,
            new_tokens: 10,
        })
    }
}

/// A state store whose `save` always fails — drives the strict-durability
/// hard-error path.
#[derive(Clone, Default)]
struct FailingStateStore;

#[async_trait]
impl StateStore for FailingStateStore {
    async fn save(&self, _state: &AgentState) -> anyhow::Result<()> {
        anyhow::bail!("state store unavailable");
    }

    async fn load(&self, _thread_id: &ThreadId) -> anyhow::Result<Option<AgentState>> {
        Ok(None)
    }

    async fn delete(&self, _thread_id: &ThreadId) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A hook that requires confirmation only for a specific `tool_call_id` and
/// allows everything else — used to pause one element of a parallel batch.
struct ConfirmToolCallHook {
    target: &'static str,
}

#[async_trait]
impl AgentHooks for ConfirmToolCallHook {
    async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
        if invocation.tool_call_id == self.target {
            ToolDecision::RequiresConfirmation("confirm".to_string())
        } else {
            ToolDecision::Allow
        }
    }
}

/// An Observe-tier tool that records the `tag` of every `execute` call so a
/// test can detect double execution.
struct CountingTool {
    executions: Arc<std::sync::Mutex<Vec<String>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum CounterToolName {
    Counter,
}

impl crate::tools::ToolName for CounterToolName {}

impl crate::tools::Tool<()> for CountingTool {
    type Name = CounterToolName;

    fn name(&self) -> CounterToolName {
        CounterToolName::Counter
    }

    fn display_name(&self) -> &'static str {
        "Counter"
    }

    fn description(&self) -> &'static str {
        "Records each execution by tag."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": { "tag": { "type": "string" } } })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        input: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        let tag = input
            .get("tag")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?")
            .to_string();
        if let Ok(mut guard) = self.executions.lock() {
            guard.push(tag.clone());
        }
        Ok(ToolResult::success(format!("counted {tag}")))
    }
}

/// Poll the event store until `turn` is recorded as finished, yielding to the
/// spawned run task between checks. Deterministic (no wall-clock sleeps).
async fn wait_for_turn_finished(
    store: &dyn EventStore,
    thread_id: &ThreadId,
    turn: usize,
) -> anyhow::Result<()> {
    for _ in 0..100_000 {
        let turns = store.get_turns(thread_id).await?;
        if turns.iter().any(|t| t.turn == turn && t.finished) {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
    anyhow::bail!("turn {turn} never finished")
}

// --- #3: second looping run on a thread keys under the next turn ------

#[tokio::test]
async fn second_text_run_on_thread_continues_turn_keys() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::text_response("first reply"),
        MockProvider::text_response("second reply"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (state_1, _) = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("one".to_string()),
        ToolContext::new(()),
    )
    .await?;
    assert!(matches!(state_1, AgentRunState::Done { .. }));

    // The second Text run on the SAME thread must succeed — before the fix it
    // failed with "turn 1 is already finished" because the turn key restarted
    // at 0 and collided with the prior run's finished turn.
    let (state_2, _) = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("two".to_string()),
        ToolContext::new(()),
    )
    .await?;
    assert!(
        matches!(state_2, AgentRunState::Done { .. }),
        "second run on the thread should succeed, got {state_2:?}"
    );

    let turns = agent.event_store.get_turns(&thread_id).await?;
    let mut turn_numbers: Vec<usize> = turns.iter().map(|t| t.turn).collect();
    turn_numbers.sort_unstable();
    assert_eq!(
        turn_numbers,
        vec![1, 2],
        "the two runs must occupy consecutive turns"
    );
    assert!(
        turns.iter().all(|t| t.finished),
        "both turns should be finished"
    );
    Ok(())
}

// --- #2: AgentInput::Message recovers an orphaned tool_use ------------

#[tokio::test]
async fn message_input_recovers_orphaned_tool_use() -> anyhow::Result<()> {
    let message_store = InMemoryStore::new();
    let thread_id = ThreadId::new();

    // Seed an orphaned assistant tool_use with no following tool_result
    // (a prior turn that was cancelled/abandoned mid-flight).
    message_store
        .append(
            &thread_id,
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: "orphan_1".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "message": "x" }),
                    thought_signature: None,
                }]),
            },
        )
        .await?;

    let provider = MockProvider::new(vec![MockProvider::text_response("recovered")]);
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(message_store.clone())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let (state, _) = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Message(vec![ContentBlock::Text {
            text: "continue".to_string(),
        }]),
        ToolContext::new(()),
    )
    .await?;
    assert!(matches!(state, AgentRunState::Done { .. }));

    // Message inputs run orphan recovery at load, durably closing the
    // unanswered tool_use with a "User cancelled" error result before the
    // user message is appended — so the history handed to the model is
    // balanced and the conversation continues.
    let history = message_store.get_history(&thread_id).await?;
    let has_recovery = history.iter().any(|m| match &m.content {
        Content::Blocks(blocks) => blocks.iter().any(|b| {
            matches!(
                b,
                ContentBlock::ToolResult { is_error: Some(true), content, .. }
                    if content == crate::llm::USER_CANCELLED_TOOL_RESULT
            )
        }),
        Content::Text(_) => false,
    });
    assert!(
        has_recovery,
        "Message input must trigger orphaned tool_use recovery; history={history:?}"
    );
    assert!(
        !crate::llm::has_unbalanced_tool_use(&history),
        "history must be balanced after recovery; history={history:?}"
    );
    Ok(())
}

// --- #5: overflow -> compaction -> EndTurn completes ------------------

#[tokio::test]
async fn context_overflow_then_compaction_completes() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::context_window_exceeded(),
        MockProvider::text_response("recovered"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .with_compaction(CompactionConfig::default())
        .with_custom_compactor(ShrinkCompactor)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let (state, _) = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
    )
    .await?;
    // Before the fix this returned Error("turn 1 is already finished") because
    // the recovery decremented the turn counter and replayed a closed key.
    assert!(
        matches!(state, AgentRunState::Done { .. }),
        "overflow recovery should complete the run, got {state:?}"
    );

    let turns = agent.event_store.get_turns(&thread_id).await?;
    let mut turn_numbers: Vec<usize> = turns.iter().map(|t| t.turn).collect();
    turn_numbers.sort_unstable();
    assert_eq!(
        turn_numbers,
        vec![1, 2],
        "overflow retry must run as the next monotonic turn"
    );
    assert!(turns.iter().all(|t| t.finished));
    Ok(())
}

// --- #4: compaction_retries resets after a successful turn ------------

#[tokio::test]
async fn non_consecutive_overflows_recover_after_retry_reset() -> anyhow::Result<()> {
    // Four overflows, each separated by a successful turn. The retry budget
    // must reset between episodes so none of them hard-fails.
    let provider = MockProvider::new(vec![
        MockProvider::context_window_exceeded(),
        MockProvider::tool_use_response("t1", "echo", json!({ "message": "a" })),
        MockProvider::context_window_exceeded(),
        MockProvider::tool_use_response("t2", "echo", json!({ "message": "b" })),
        MockProvider::context_window_exceeded(),
        MockProvider::tool_use_response("t3", "echo", json!({ "message": "c" })),
        MockProvider::context_window_exceeded(),
        MockProvider::text_response("all done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .with_compaction(CompactionConfig::default())
        .with_custom_compactor(ShrinkCompactor)
        .event_store(new_event_store())
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    assert!(
        matches!(state, AgentRunState::Done { .. }),
        "non-consecutive overflows must each recover after the retry reset, got {state:?}"
    );
    Ok(())
}

#[tokio::test]
async fn consecutive_overflows_exhaust_retry_budget() -> anyhow::Result<()> {
    // Four consecutive overflows with no intervening success: the consecutive
    // retry budget (MAX_COMPACTION_RETRIES = 3) is exhausted on the fourth.
    let provider = MockProvider::new(vec![
        MockProvider::context_window_exceeded(),
        MockProvider::context_window_exceeded(),
        MockProvider::context_window_exceeded(),
        MockProvider::context_window_exceeded(),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .with_compaction(CompactionConfig::default())
        .with_custom_compactor(ShrinkCompactor)
        .event_store(new_event_store())
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    assert!(
        matches!(state, AgentRunState::Error(_)),
        "four consecutive overflows should exhaust the retry budget, got {state:?}"
    );
    Ok(())
}

// --- #16: strict durability checkpoint failure is a hard error --------

#[tokio::test]
async fn strict_durability_checkpoint_failure_errors() {
    let provider = MockProvider::new(vec![MockProvider::text_response("hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(FailingStateStore)
        .event_store(new_event_store())
        .build_with_stores();

    let outcome = Box::pin(agent.run_turn(
        ThreadId::new(),
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        TurnOptions {
            strict_durability: true,
            ..Default::default()
        },
    ))
    .await;
    assert!(
        matches!(outcome, TurnOutcome::Error(_)),
        "a failed strict-durability checkpoint must abort the turn, got {outcome:?}"
    );
}

#[tokio::test]
async fn non_strict_checkpoint_failure_does_not_error() {
    // Without strict durability, a failed state checkpoint is only a warning.
    let provider = MockProvider::new(vec![MockProvider::text_response("hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(FailingStateStore)
        .event_store(new_event_store())
        .build_with_stores();

    let outcome = Box::pin(agent.run_turn(
        ThreadId::new(),
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        TurnOptions::default(),
    ))
    .await;
    assert!(
        matches!(outcome, TurnOutcome::Done { .. }),
        "non-strict checkpoint failures must not abort the turn, got {outcome:?}"
    );
}

// --- #12: finish_turn failure preserves the original turn error -------

#[tokio::test]
async fn finish_turn_failure_preserves_original_error() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("hi")]);
    let event_store = FailingEventStore::new(EventStoreFailureMode::Both);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(event_store)
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    match state {
        AgentRunState::Error(error) => assert!(
            error.message.contains("Failed to append event"),
            "the original append error must survive a finish_turn failure, got: {}",
            error.message
        ),
        other => panic!("expected Error, got {other:?}"),
    }
    Ok(())
}

// --- #15: parallel batch confirmation drains completed siblings -------

#[tokio::test]
async fn parallel_batch_confirmation_does_not_re_execute_siblings() -> anyhow::Result<()> {
    let executions = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("t1", "counter", json!({ "tag": "a" })),
            ("t2", "counter", json!({ "tag": "b" })),
            ("t3", "counter", json!({ "tag": "c" })),
        ]),
        MockProvider::text_response("done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(CountingTool {
        executions: Arc::clone(&executions),
    });
    let agent = builder::<()>()
        .provider(provider)
        .hooks(ConfirmToolCallHook { target: "t2" })
        .tools(tools)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::new();

    let state_1 = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    let (continuation, tool_call_id) = match state_1 {
        AgentRunState::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("expected AwaitingConfirmation, got {other:?}"),
    };
    assert_eq!(
        tool_call_id, "t2",
        "the middle tool should be the one paused"
    );

    let state_2 = agent
        .run(
            thread_id.clone(),
            AgentInput::Resume {
                continuation,
                tool_call_id,
                confirmed: true,
                rejection_reason: None,
            },
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    assert!(matches!(state_2, AgentRunState::Done { .. }));

    let mut execs = executions
        .lock()
        .ok()
        .context("executions mutex poisoned")?
        .clone();
    execs.sort();
    // Each tool runs exactly once: the already-completed sibling `c` must not
    // be re-executed on resume (the bug double-ran it).
    assert_eq!(
        execs,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "each tool must execute exactly once; got {execs:?}"
    );
    Ok(())
}

// --- #8: streaming retry isolates deltas under a fresh message_id -----

#[tokio::test]
async fn streaming_retry_uses_fresh_message_id_per_attempt() -> anyhow::Result<()> {
    let provider = StreamScriptProvider::new(vec![
        StreamScriptStep::Frames(vec![
            StreamDelta::TextDelta {
                delta: "Hello ".to_string(),
                block_index: 0,
            },
            StreamDelta::Error {
                message: "transient blip".to_string(),
                kind: StreamErrorKind::ServerError,
            },
        ]),
        StreamScriptStep::Frames(vec![
            StreamDelta::TextDelta {
                delta: "Hello ".to_string(),
                block_index: 0,
            },
            StreamDelta::TextDelta {
                delta: "world".to_string(),
                block_index: 0,
            },
            StreamDelta::Done {
                stop_reason: Some(crate::llm::StopReason::EndTurn),
            },
        ]),
    ]);
    let config = AgentConfig {
        streaming: true,
        // Zero-delay retry: exercises the recoverable retry path with no
        // real backoff sleep.
        retry: RetryConfig {
            max_retries: 2,
            base_delay_ms: 0,
            max_delay_ms: 0,
        },
        ..Default::default()
    };
    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let (state, events) = run_recorded(
        &agent,
        ThreadId::new(),
        AgentInput::Text("hi".to_string()),
        ToolContext::new(()),
    )
    .await?;
    assert!(matches!(state, AgentRunState::Done { .. }));

    let deltas: Vec<(String, String)> = events
        .iter()
        .filter_map(|e| match &e.event {
            AgentEvent::TextDelta { message_id, delta } => {
                Some((message_id.clone(), delta.clone()))
            }
            _ => None,
        })
        .collect();
    let distinct: std::collections::BTreeSet<&String> = deltas.iter().map(|(id, _)| id).collect();
    assert_eq!(
        distinct.len(),
        2,
        "each stream attempt must use a distinct message_id; got {deltas:?}"
    );

    let final_text_id = events
        .iter()
        .find_map(|e| match &e.event {
            AgentEvent::Text { message_id, .. } => Some(message_id.clone()),
            _ => None,
        })
        .context("expected a final Text event")?;
    // The surviving (successful) attempt's deltas are the ones correlated with
    // the final assembled Text — and the failed attempt's partial is isolated
    // under its own id rather than duplicated under the surviving id.
    let surviving: Vec<&String> = deltas
        .iter()
        .filter(|(id, _)| *id == final_text_id)
        .map(|(_, d)| d)
        .collect();
    assert_eq!(
        surviving,
        vec!["Hello ", "world"],
        "the surviving message must contain only the successful attempt's deltas"
    );
    let abandoned = deltas.iter().filter(|(id, _)| *id != final_text_id).count();
    assert_eq!(
        abandoned, 1,
        "the failed attempt's partial delta must stay isolated under its own id"
    );
    Ok(())
}

// --- #10: stalled stream surfaces an inactivity timeout ---------------

#[tokio::test]
async fn stalled_stream_times_out_and_errors() -> anyhow::Result<()> {
    // A stream that never yields a frame must not hang the run forever; the
    // per-frame inactivity timeout (reduced to 20ms under cfg(test)) surfaces a
    // recoverable error, and with no retries the run ends in error rather than
    // stalling. The stall is permanent, so the timeout firing is deterministic.
    let provider = StreamScriptProvider::new(vec![StreamScriptStep::FramesThenStall(vec![])]);
    let config = AgentConfig {
        streaming: true,
        retry: RetryConfig::no_retry(),
        ..Default::default()
    };
    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("go".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;
    assert!(
        matches!(state, AgentRunState::Error(_)),
        "a stalled stream must end the run in error, got {state:?}"
    );
    Ok(())
}

// --- #1 / #11: run_persistent behavioral coverage ---------------------

#[tokio::test]
async fn run_persistent_processes_injected_turns_then_done_on_drop() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::text_response("first"),
        MockProvider::text_response("second"),
        MockProvider::text_response("third"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let handle = agent.run_persistent(
        thread_id.clone(),
        AgentInput::Text("first".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let AgentHandle {
        input_tx, state_rx, ..
    } = handle;

    input_tx
        .send(AgentInput::Text("second".to_string()))
        .await?;
    input_tx.send(AgentInput::Text("third".to_string())).await?;
    drop(input_tx);

    let state = state_rx.await?;
    assert!(
        matches!(state, AgentRunState::Done { .. }),
        "dropping input_tx should end the run as Done, got {state:?}"
    );

    let turns = agent.event_store.get_turns(&thread_id).await?;
    let mut turn_numbers: Vec<usize> = turns.iter().map(|t| t.turn).collect();
    turn_numbers.sort_unstable();
    assert_eq!(
        turn_numbers,
        vec![1, 2, 3],
        "initial + two injected turns should occupy three consecutive turns"
    );
    assert!(
        turns.iter().all(|t| t.finished),
        "every persistent turn must be finished"
    );

    // Sequences are globally continuous across the persistent turns.
    let mut sequences: Vec<u64> = agent
        .event_store
        .get_events(&thread_id)
        .await?
        .into_iter()
        .map(|envelope| envelope.sequence)
        .collect();
    sequences.sort_unstable();
    for pair in sequences.windows(2) {
        assert_eq!(
            pair[1],
            pair[0] + 1,
            "persistent-mode event sequences must be continuous: {sequences:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn run_persistent_cancel_between_turns_is_cancelled() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("first")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let cancel = CancellationToken::new();

    let handle = agent.run_persistent(
        thread_id.clone(),
        AgentInput::Text("first".to_string()),
        ToolContext::new(()),
        cancel.clone(),
    );

    // Wait until the agent has finished turn 1 and is parked on the input
    // channel, then cancel.
    wait_for_turn_finished(agent.event_store.as_ref(), &thread_id, 1).await?;
    cancel.cancel();

    let state = handle.state_rx.await?;
    assert!(
        matches!(state, AgentRunState::Cancelled { .. }),
        "cancelling while parked should yield Cancelled, got {state:?}"
    );

    // The terminal Cancelled event is keyed under turn+1 (turn 1 was finished),
    // and that turn is itself finished.
    let turn_2 = agent
        .event_store
        .get_turn(&thread_id, 2)
        .await?
        .context("expected a turn 2 carrying the terminal Cancelled event")?;
    assert!(turn_2.finished, "the cancel turn must be finished");
    assert!(
        turn_2
            .events
            .iter()
            .any(|e| matches!(e.event, AgentEvent::Cancelled { .. })),
        "turn 2 must carry the Cancelled event"
    );
    Ok(())
}

#[tokio::test]
async fn run_persistent_unsupported_input_errors() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("first")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let handle = agent.run_persistent(
        thread_id,
        AgentInput::Text("first".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let AgentHandle {
        input_tx, state_rx, ..
    } = handle;

    // `Continue` is not a valid persistent-channel input — it must end the run
    // with Error rather than silently reporting Done.
    input_tx.send(AgentInput::Continue).await?;
    drop(input_tx);

    let state = state_rx.await?;
    assert!(
        matches!(state, AgentRunState::Error(_)),
        "injecting Continue should error the run, got {state:?}"
    );
    Ok(())
}

// ===========================================================================
// AGENT-LOOP feature cluster: budgets, parallel cap, guardrails, reminders,
// run_stream.
// ===========================================================================

/// Provider that reports a priced provider/model (`openai` / `gpt-4o`) so the
/// run can estimate a non-`None` cost. Always returns a terminal text turn
/// with a fixed usage so the cost is deterministic.
struct PricedProvider;

#[async_trait]
impl crate::llm::LlmProvider for PricedProvider {
    async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "msg_priced".to_string(),
            content: vec![ContentBlock::Text {
                text: "all done".to_string(),
            }],
            model: "gpt-4o".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 2_000,
                output_tokens: 1_000,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "gpt-4o"
    }

    fn provider(&self) -> &'static str {
        "openai"
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProbeToolName {
    Probe,
}

impl crate::tools::ToolName for ProbeToolName {}

/// Observe-tier tool that records the peak number of concurrently-executing
/// invocations. Yields twice mid-execution so that, if two invocations are
/// allowed to overlap, the in-flight counter is observed at 2.
struct ConcurrencyProbeTool {
    in_flight: Arc<AtomicUsize>,
    max_in_flight: Arc<AtomicUsize>,
}

impl crate::tools::Tool<()> for ConcurrencyProbeTool {
    type Name = ProbeToolName;

    fn name(&self) -> ProbeToolName {
        ProbeToolName::Probe
    }

    fn display_name(&self) -> &'static str {
        "Probe"
    }

    fn description(&self) -> &'static str {
        "Records peak concurrency of observe-tier execution."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": { "message": { "type": "string" } } })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        input: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(current, Ordering::SeqCst);
        // Yield so an overlapping invocation would be observed in-flight.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        Ok(ToolResult::success(format!("probe:{message}")))
    }
}

#[tokio::test]
async fn test_usage_budget_total_tokens_breach_stops_run() -> anyhow::Result<()> {
    // The first (tool-use) turn accrues 30 tokens (10 in + 20 out), over the
    // 25-token limit, so the run stops with BudgetExceeded instead of looping.
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("call_1", "echo", json!({"message": "x"})),
        MockProvider::text_response("should not be reached"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let config = AgentConfig {
        usage_limits: Some(UsageLimits {
            max_total_tokens: Some(25),
            ..Default::default()
        }),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (state, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
    )
    .await?;

    match state {
        AgentRunState::BudgetExceeded { limit, .. } => {
            assert_eq!(limit, BudgetLimitKind::TotalTokens);
        }
        other => anyhow::bail!("expected BudgetExceeded, got {other:?}"),
    }

    assert!(
        events.iter().any(|e| matches!(
            &e.event,
            AgentEvent::BudgetExceeded {
                limit: BudgetLimitKind::TotalTokens,
                ..
            }
        )),
        "a BudgetExceeded event must be emitted",
    );

    Ok(())
}

#[tokio::test]
async fn test_under_budget_done_reports_estimated_cost() -> anyhow::Result<()> {
    // gpt-4o pricing for 2000 in / 1000 out = $0.0025 + $0.005 = $0.0075.
    let agent = builder::<()>()
        .provider(PricedProvider)
        .config(AgentConfig {
            usage_limits: Some(UsageLimits {
                max_cost_usd: Some(10.0),
                ..Default::default()
            }),
            ..Default::default()
        })
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (state, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    match state {
        AgentRunState::Done {
            estimated_cost_usd: Some(cost),
            ..
        } => {
            assert!(
                (cost - 0.0075).abs() < 1e-9,
                "unexpected estimated cost: {cost}"
            );
        }
        other => anyhow::bail!("expected Done with Some(cost), got {other:?}"),
    }

    // The Done event carries the same cost on the wire.
    let done_cost = events.iter().find_map(|e| match &e.event {
        AgentEvent::Done {
            estimated_cost_usd, ..
        } => Some(*estimated_cost_usd),
        _ => None,
    });
    assert_eq!(done_cost, Some(Some(0.0075)));

    Ok(())
}

#[tokio::test]
async fn test_max_parallel_tools_one_runs_sequentially() -> anyhow::Result<()> {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));

    let provider = MockProvider::new(vec![
        MockProvider::tool_uses_response(vec![
            ("call_a", "probe", json!({"message": "a"})),
            ("call_b", "probe", json!({"message": "b"})),
            ("call_c", "probe", json!({"message": "c"})),
        ]),
        MockProvider::text_response("done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ConcurrencyProbeTool {
        in_flight: Arc::clone(&in_flight),
        max_in_flight: Arc::clone(&max_in_flight),
    });

    let config = AgentConfig {
        max_parallel_tools: Some(1),
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .config(config)
        .event_store(new_event_store())
        .build();

    let thread_id = ThreadId::new();
    let (state, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
    )
    .await?;

    assert!(matches!(state, AgentRunState::Done { .. }));
    assert_eq!(
        max_in_flight.load(Ordering::SeqCst),
        1,
        "Some(1) must run observe-tier tools strictly sequentially",
    );

    // Result order is preserved: ToolCallEnd ids appear in input order.
    let tool_end_ids: Vec<String> = events
        .iter()
        .filter_map(|e| match &e.event {
            AgentEvent::ToolCallEnd { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_end_ids,
        vec![
            "call_a".to_string(),
            "call_b".to_string(),
            "call_c".to_string()
        ],
        "sequential execution must preserve input ordering",
    );

    Ok(())
}

struct BlockingRequestHooks;

#[async_trait]
impl AgentHooks for BlockingRequestHooks {
    async fn pre_llm_request(&self, _request: &ChatRequest) -> RequestDecision {
        RequestDecision::Block("policy violation".to_string())
    }
}

#[tokio::test]
async fn test_pre_llm_request_block_ends_run() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("never sent")]);

    let agent = builder::<()>()
        .provider(provider)
        .hooks(BlockingRequestHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();

    let thread_id = ThreadId::new();
    let (state, events) = run_recorded(
        &agent,
        thread_id,
        AgentInput::Text("hi".to_string()),
        ToolContext::new(()),
    )
    .await?;

    match state {
        AgentRunState::Error(error) => {
            assert!(
                error.message.contains("blocked by guardrail")
                    && error.message.contains("policy violation"),
                "unexpected error message: {}",
                error.message
            );
        }
        other => anyhow::bail!("expected Error from guardrail block, got {other:?}"),
    }

    assert!(
        events.iter().any(|e| matches!(
            &e.event,
            AgentEvent::Error { message, .. } if message.contains("blocked by guardrail")
        )),
        "a guardrail-block error event must be emitted",
    );

    Ok(())
}

#[tokio::test]
async fn test_tool_reminder_fires_after_trigger() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![
        MockProvider::tool_use_response("call_1", "echo", json!({"message": "hi"})),
        MockProvider::text_response("done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let reminders = ReminderConfig::new()
        .with_tool_reminder("echo", ToolReminder::always("REMEMBER_TO_VERIFY_OUTPUT"));

    let message_store = InMemoryStore::new();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(message_store)
        .state_store(InMemoryStore::new())
        .with_reminders(reminders)
        .event_store(new_event_store())
        .build_with_stores();

    let thread_id = ThreadId::new();
    let _ = run_recorded(
        &agent,
        thread_id.clone(),
        AgentInput::Text("go".to_string()),
        ToolContext::new(()),
    )
    .await?;

    let history = agent.message_store.get_history(&thread_id).await?;
    let reminded = history.iter().any(|m| {
        let Content::Blocks(blocks) = &m.content else {
            return false;
        };
        blocks.iter().any(|b| {
            matches!(
                b,
                ContentBlock::ToolResult { content, .. }
                    if content.contains("REMEMBER_TO_VERIFY_OUTPUT")
                        && content.contains("<system-reminder>")
            )
        })
    });
    assert!(
        reminded,
        "the configured tool reminder must be appended to the echo tool result",
    );

    Ok(())
}

#[tokio::test]
async fn test_run_stream_yields_same_events_as_store() -> anyhow::Result<()> {
    use futures::StreamExt as _;

    let provider = MockProvider::new(vec![MockProvider::text_response("streamed hello")]);
    let store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .event_store(store.clone())
        .build();

    let thread_id = ThreadId::new();
    let stream = agent.run_stream(
        thread_id.clone(),
        AgentInput::Text("hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let streamed: Vec<AgentEvent> = stream.collect().await;

    assert!(!streamed.is_empty(), "run_stream must yield events");

    let stored: Vec<AgentEvent> = store
        .get_events(&thread_id)
        .await?
        .into_iter()
        .map(|envelope| envelope.event)
        .collect();

    // AgentEvent is not PartialEq; compare the JSON wire forms, which is the
    // durable contract the stream is meant to mirror.
    let streamed_json: anyhow::Result<Vec<serde_json::Value>> = streamed
        .iter()
        .map(|e| serde_json::to_value(e).map_err(anyhow::Error::from))
        .collect();
    let stored_json: anyhow::Result<Vec<serde_json::Value>> = stored
        .iter()
        .map(|e| serde_json::to_value(e).map_err(anyhow::Error::from))
        .collect();
    assert_eq!(streamed_json?, stored_json?);

    Ok(())
}
