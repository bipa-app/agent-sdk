use super::test_utils::*;
use super::*;
use crate::events::{AgentEvent, AgentEventEnvelope};
use crate::hooks::AllowAllHooks;
use crate::llm::{ChatOutcome, Content, ContentBlock};
use crate::stores::{
    EventStore, InMemoryEventStore, InMemoryStore, MessageStore, StoredTurnEvents,
};
use crate::tools::{ListenToolUpdate, ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, AgentRunState, TurnOptions, TurnOutcome};
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn expected_turn_for_input(input: &AgentInput, existing_turns: &[StoredTurnEvents]) -> usize {
    match input {
        AgentInput::Resume { continuation, .. } => continuation.turn,
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
    let outcome = agent
        .run_turn(
            thread_id.clone(),
            input,
            tool_context,
            CancellationToken::new(),
            options,
        )
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
        if matches!(self.failure_mode, EventStoreFailureMode::Append) {
            anyhow::bail!("append failure");
        }
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> anyhow::Result<()> {
        if matches!(self.failure_mode, EventStoreFailureMode::FinishTurn) {
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
        ChatOutcome::RateLimited,
        ChatOutcome::RateLimited,
        ChatOutcome::RateLimited,
        ChatOutcome::RateLimited,
        ChatOutcome::RateLimited,
        ChatOutcome::RateLimited, // 6th attempt exceeds max_retries (5)
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
        ChatOutcome::RateLimited,
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
        config: AgentConfig::default(),
        compaction_config: None,
        compactor: None,
        execution_store: None,
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
// Server Boundary Tests (ENG-7908)
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
async fn test_run_turn_returns_error_when_event_append_fails() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(FailingEventStore::new(EventStoreFailureMode::Append))
        .build();

    let outcome = agent
        .run_turn(
            ThreadId::new(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
            TurnOptions::default(),
        )
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

    let outcome = agent
        .run_turn(
            ThreadId::new(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
            TurnOptions::default(),
        )
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
            assert!(!continuation.thread_id.to_string().is_empty());
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
        "Expected Done when no tool calls, got {:?}",
        outcome
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
        "Expected NeedsMoreTurns, got {:?}",
        outcome
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
