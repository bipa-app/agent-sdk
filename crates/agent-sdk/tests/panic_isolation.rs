//! Panic-isolation integration tests (Phase 10 · C).
//!
//! Locks in the SDK contract that a panic inside a tool or LLM/stream
//! future is isolated into an *observable, structured* outcome instead
//! of unwinding the whole spawned run task. Before this fix there was
//! no `catch_unwind` anywhere: a panicking tool unwound the run task,
//! dropped `state_tx` (so the caller saw an opaque `RecvError`), and
//! orphaned the assistant `tool_use` — reintroducing the unbalanced
//! `tool_use` / `tool_result` history bug the cancel fix closed.
//!
//! The two scenarios asserted here:
//!
//! 1. **A tool that panics** → the SDK commits a structured error
//!    `tool_result` ("Tool panicked: …") via the normal completion
//!    path, the run still ends as `Done`/`Error` (the channel is never
//!    dropped), and history stays balanced (no orphan `tool_use`).
//! 2. **A panicking LLM call** → the run ends as `AgentRunState::Error`
//!    carrying a structured message, *not* a dropped channel /
//!    `RecvError`.

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, Message, Role,
    StopReason, Usage,
};
use agent_sdk::{
    AgentInput, AgentRunState, AllowAllHooks, CancellationToken, DynamicToolName,
    InMemoryEventStore, InMemoryStore, MessageStore, ThreadId, Tool, ToolContext, ToolRegistry,
    ToolResult, ToolTier, builder,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};

/// LLM provider that returns the next pre-scripted outcome.
struct ScriptedProvider {
    responses: RwLock<Vec<ChatOutcome>>,
}

impl ScriptedProvider {
    const fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
        }
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let mut responses = self
            .responses
            .write()
            .ok()
            .ok_or_else(|| anyhow!("responses lock poisoned"))?;
        if responses.is_empty() {
            Err(anyhow!("ScriptedProvider script exhausted"))
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

/// LLM provider whose `chat()` panics — stands in for a buggy provider
/// adapter (or compaction call) that unwinds mid-flight.
struct PanickingProvider;

#[async_trait]
impl LlmProvider for PanickingProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        panic!("provider blew up");
    }

    fn model(&self) -> &'static str {
        "panic-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

/// Tool that always panics when executed. Deliberately panics with a
/// formatted `String` payload to exercise the `String` downcast branch
/// of the panic-message extractor.
struct PanickingTool;

impl Tool<()> for PanickingTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("boom")
    }

    fn display_name(&self) -> &'static str {
        "Boom"
    }

    fn description(&self) -> &'static str {
        "A tool that panics on execution."
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
        panic!("tool exploded: {}", "kaboom");
    }
}

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
impl agent_sdk::StateStore for SharedStore {
    async fn save(&self, state: &agent_sdk::AgentState) -> Result<()> {
        self.0.save(state).await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<agent_sdk::AgentState>> {
        self.0.load(thread_id).await
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        self.0.delete(thread_id).await
    }
}

fn tool_use_response(id: &str, name: &str) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: format!("resp_{id}"),
        content: vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({}),
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

/// IDs of `tool_use` blocks with no matching `tool_result` in the next
/// user message. Empty means balanced history.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_panic_becomes_structured_result_and_history_stays_balanced() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    let tool_call_id = "toolu_panic_test_1";

    let mut tools = ToolRegistry::new();
    tools.register(PanickingTool);

    // Turn 1: model calls the panicking tool. Turn 2: after the
    // structured error result is fed back, the model wraps up.
    let provider = ScriptedProvider::new(vec![
        tool_use_response(tool_call_id, "boom"),
        text_response("Recovered from the tool failure."),
    ]);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .build_with_stores();

    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("run the boom tool".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        // The channel must resolve — a dropped `state_tx` (the pre-fix
        // bug) would surface here as a `RecvError`.
        .map_err(|e| anyhow!("run state channel dropped (RecvError): {e}"))?;

    // The run continues past the panic and completes normally; the
    // tool panic is just a failed tool result, not a run-ending event.
    assert!(
        matches!(final_state, AgentRunState::Done { .. }),
        "run must end as Done after the tool panic is absorbed; got {final_state:?}",
    );

    let history = store.get_history(&thread_id).await?;

    // Balanced history: no orphan tool_use.
    let orphans = orphan_tool_use_ids(&history);
    assert!(
        orphans.is_empty(),
        "a tool panic must not leave an orphan tool_use; got {orphans:?} in {history:#?}",
    );

    // Exactly one structured error tool_result, carrying the panic
    // message (proving it was caught, not swallowed).
    let results = tool_results_for(&history, tool_call_id);
    assert_eq!(
        results.len(),
        1,
        "exactly one tool_result for the panicking tool_use; got {results:?}",
    );
    assert!(
        results[0].contains("Tool panicked"),
        "tool_result must be the structured panic error; got {:?}",
        results[0],
    );
    assert!(
        results[0].contains("tool exploded: kaboom"),
        "structured error should carry the original panic message; got {:?}",
        results[0],
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llm_panic_surfaces_as_structured_run_error_not_recv_error() -> Result<()> {
    let store = SharedStore::new();
    let thread_id = ThreadId::new();

    let agent = builder::<()>()
        .provider(PanickingProvider)
        .tools(ToolRegistry::new())
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .build_with_stores();

    let final_state = agent
        .run(
            thread_id.clone(),
            AgentInput::Text("hello".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        // Before the run-loop catch_unwind boundary, a panic in the
        // provider call dropped `state_tx` and the caller saw this as
        // an opaque RecvError. Now the channel resolves with an Error.
        .map_err(|e| anyhow!("run state channel dropped (RecvError) on LLM panic: {e}"))?;

    match final_state {
        AgentRunState::Error(err) => {
            assert!(
                err.message.contains("panicked"),
                "LLM panic must surface as a structured run error mentioning the panic; got {:?}",
                err.message,
            );
            assert!(
                err.message.contains("provider blew up"),
                "structured run error should carry the original panic message; got {:?}",
                err.message,
            );
        }
        other => panic!("LLM panic must end the run as Error; got {other:?}"),
    }

    Ok(())
}
