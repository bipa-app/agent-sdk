use super::test_utils::*;
use super::*;
use crate::events::AgentEvent;
use crate::hooks::AllowAllHooks;
use crate::llm::{ChatOutcome, Content, ContentBlock};
use crate::stores::InMemoryStore;
use crate::stores::MessageStore;
use crate::tools::{ListenToolUpdate, ToolContext, ToolRegistry};
use crate::types::{AgentConfig, AgentInput, TurnOutcome};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ===================
// Builder Tests
// ===================

#[test]
fn test_builder_creates_agent_loop() {
    let provider = MockProvider::new(vec![]);
    let agent = builder::<()>().provider(provider).build();

    assert_eq!(agent.config.max_turns, 10);
    assert_eq!(agent.config.max_tokens, 4096);
}

#[test]
fn test_builder_with_custom_config() {
    let provider = MockProvider::new(vec![]);
    let config = AgentConfig {
        max_turns: 5,
        max_tokens: 2048,
        system_prompt: "Custom prompt".to_string(),
        model: "custom-model".to_string(),
        ..Default::default()
    };

    let agent = builder::<()>().provider(provider).config(config).build();

    assert_eq!(agent.config.max_turns, 5);
    assert_eq!(agent.config.max_tokens, 2048);
    assert_eq!(agent.config.system_prompt, "Custom prompt");
}

#[test]
fn test_builder_with_tools() {
    let provider = MockProvider::new(vec![]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>().provider(provider).tools(tools).build();

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
        .build_with_stores();

    // Just verify it builds without panicking
    assert_eq!(agent.config.max_turns, 10);
}

// ===================
// Run Loop Tests
// ===================

#[tokio::test]
async fn test_simple_text_response() -> anyhow::Result<()> {
    let provider = MockProvider::new(vec![MockProvider::text_response("Hello, user!")]);

    let agent = builder::<()>().provider(provider).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).tools(tools).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(
        thread_id,
        AgentInput::Text("Run echo".to_string()),
        tool_ctx,
    );

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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
        max_turns: 2,
        ..Default::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .config(config)
        .build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) =
        agent.run(thread_id, AgentInput::Text("Loop".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).tools(tools).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(
        thread_id,
        AgentInput::Text("Call unknown".to_string()),
        tool_ctx,
    );

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).config(config).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).config(config).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).config(config).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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

    let agent = builder::<()>().provider(provider).config(config).build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());
    let (mut rx, _final_state) = agent.run(thread_id, AgentInput::Text("Hi".to_string()), tool_ctx);

    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

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
    let agent = builder::<()>().provider(provider).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    let mut ids = std::collections::HashSet::new();
    while let Some(envelope) = rx.recv().await {
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
    let agent = builder::<()>().provider(provider).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    let mut envelopes = Vec::new();
    while let Some(envelope) = rx.recv().await {
        envelopes.push(envelope);
    }

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
    let agent = builder::<()>().provider(provider).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    let first = rx.recv().await.expect("should have at least one event");
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
    let agent = builder::<()>().provider(provider).tools(tools).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Go".into()),
        ToolContext::new(()),
    );

    let mut sequences = Vec::new();
    while let Some(envelope) = rx.recv().await {
        sequences.push(envelope.sequence);
    }

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
    let agent = builder::<()>().provider(provider).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    let mut envelopes = Vec::new();
    while let Some(envelope) = rx.recv().await {
        envelopes.push(envelope);
    }

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

    let agent_a = builder::<()>().provider(provider_a).build();
    let agent_b = builder::<()>().provider(provider_b).build();

    let (mut rx_a, _) = agent_a.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );
    let (mut rx_b, _) = agent_b.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    let first_a = rx_a.recv().await.expect("run A should emit events");
    let first_b = rx_b.recv().await.expect("run B should emit events");

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
    let agent = builder::<()>().provider(provider).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hi".into()),
        ToolContext::new(()),
    );

    while let Some(envelope) = rx.recv().await {
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
    let agent = builder::<()>().provider(provider).tools(tools).build();

    let (mut rx, _) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Go".into()),
        ToolContext::new(()),
    );

    let mut envelopes = Vec::new();
    while let Some(envelope) = rx.recv().await {
        envelopes.push(envelope);
    }

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let thread_id = ThreadId::new();

    // Turn 1: reaches awaiting confirmation after listen pre-runtime
    let (_events_1, outcome_rx_1) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let outcome_1 = outcome_rx_1.await?;

    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    // Turn 2: confirm and execute
    let (_events_2, outcome_rx_2) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
    );
    let outcome_2 = outcome_rx_2.await?;
    assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

    // Turn 3: continue and finish
    let (_events_3, outcome_rx_3) =
        agent.run_turn(thread_id, AgentInput::Continue, ToolContext::new(()));
    let outcome_3 = outcome_rx_3.await?;
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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let thread_id = ThreadId::new();

    let (_events_1, outcome_rx_1) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let outcome_1 = outcome_rx_1.await?;

    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let (_events_2, outcome_rx_2) = agent.run_turn(
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: false,
            rejection_reason: Some("nope".to_string()),
        },
        ToolContext::new(()),
    );
    let _ = outcome_rx_2.await?;

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let (rx, _state_rx) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let events = drain_events(rx).await;

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let (rx, _state_rx) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let events = drain_events(rx).await;

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let (rx, _state_rx) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let events = drain_events(rx).await;

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let (events_rx, outcome_rx) = agent.run_turn(
        ThreadId::new(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    drop(events_rx);

    let outcome = outcome_rx.await?;
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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let thread_id = ThreadId::new();

    let (_events_1, outcome_rx_1) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("Run listen tool".to_string()),
        ToolContext::new(()),
    );
    let outcome_1 = outcome_rx_1.await?;
    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let (events_2, outcome_rx_2) = agent.run_turn(
        thread_id,
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
    );
    let outcome_2 = outcome_rx_2.await?;
    let events_2 = drain_events(events_2).await;

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

    let agent = builder::<()>().provider(provider).tools(tools).build();
    let thread_id = ThreadId::new();

    let (_events_1, outcome_rx_1) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("Run mixed tools".to_string()),
        ToolContext::new(()),
    );
    let outcome_1 = outcome_rx_1.await?;
    let (continuation, tool_call_id) = match outcome_1 {
        TurnOutcome::AwaitingConfirmation {
            continuation,
            tool_call_id,
            ..
        } => (continuation, tool_call_id),
        other => panic!("Expected AwaitingConfirmation, got {other:?}"),
    };

    let (events_2, outcome_rx_2) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Resume {
            continuation,
            tool_call_id,
            confirmed: true,
            rejection_reason: None,
        },
        ToolContext::new(()),
    );
    let outcome_2 = outcome_rx_2.await?;
    let events_2 = drain_events(events_2).await;
    assert!(matches!(outcome_2, TurnOutcome::NeedsMoreTurns { .. }));

    assert!(events_2.iter().any(|event| {
        matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == "tool_listen")
    }));
    assert!(events_2.iter().any(|event| {
        matches!(&event.event, AgentEvent::ToolCallEnd { id, .. } if id == "tool_echo")
    }));

    let (_events_3, outcome_rx_3) =
        agent.run_turn(thread_id, AgentInput::Continue, ToolContext::new(()));
    let outcome_3 = outcome_rx_3.await?;
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
        config: AgentConfig::default(),
        compaction_config: None,
        execution_store: None,
    };

    let thread_id = ThreadId::new();
    let (mut rx, _final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Run both tools".to_string()),
        ToolContext::new(()),
    );

    while rx.recv().await.is_some() {}

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
