//! End-to-end coverage for Phase 13 · B typed tool-arg validation.
//!
//! Drives a [`TypedTool`] through the real agent loop with a scripted
//! provider and asserts the runtime contract:
//!
//! * a model tool call whose arguments don't match `Input` is turned into a
//!   structured validation-error `tool_result` (model self-corrects on the
//!   next turn), and the tool's `execute` is **never** reached;
//! * the next turn's well-formed call deserializes and runs;
//! * `tool_use` / `tool_result` history stays balanced on both paths
//!   (no orphan `tool_use`);
//! * the run terminates cleanly under a bounded `max_turns`, even when every
//!   turn emits invalid args (no infinite loop).

#[path = "support/stub_provider.rs"]
mod stub_provider;

use agent_sdk::llm::{Content, ContentBlock, Message};
use agent_sdk::{
    AgentConfig, AgentInput, AllowAllHooks, CancellationToken, InMemoryEventStore, InMemoryStore,
    MessageStore, ThreadId, ToolContext, ToolRegistry, ToolResult, TypedTool, builder,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use stub_provider::StubProvider;

/// Shared message store so the test can both pass it to the builder and read
/// the persisted history back afterwards.
#[derive(Clone)]
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

#[derive(Debug, Serialize, Deserialize)]
struct GreetArgs {
    name: String,
    greeting: String,
}

/// Typed tool that counts how many times `execute` actually runs, so tests
/// can assert the validation boundary never reaches it with bad arguments.
struct GreetTool {
    executions: Arc<AtomicUsize>,
}

impl TypedTool<()> for GreetTool {
    type Input = GreetArgs;

    fn name(&self) -> &'static str {
        "greet"
    }

    fn description(&self) -> &'static str {
        "Greet someone by name"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "greeting": { "type": "string" }
            },
            "required": ["name", "greeting"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: GreetArgs) -> Result<ToolResult> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(format!(
            "{}, {}!",
            input.greeting, input.name
        )))
    }
}

/// Count `tool_use` and `tool_result` blocks across the whole history.
///
/// A balanced turn has exactly one `tool_result` for every `tool_use`.
fn count_tool_blocks(history: &[Message]) -> (usize, usize) {
    let mut uses = 0;
    let mut results = 0;
    for msg in history {
        if let Content::Blocks(blocks) = &msg.content {
            for block in blocks {
                match block {
                    ContentBlock::ToolUse { .. } => uses += 1,
                    ContentBlock::ToolResult { .. } => results += 1,
                    _ => {}
                }
            }
        }
    }
    (uses, results)
}

/// Find the first `tool_result` block whose `tool_use_id` matches `id`.
fn tool_result_for<'a>(history: &'a [Message], id: &str) -> Option<&'a ContentBlock> {
    history.iter().find_map(|msg| {
        let Content::Blocks(blocks) = &msg.content else {
            return None;
        };
        blocks.iter().find(|block| {
            matches!(
                block,
                ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id
            )
        })
    })
}

#[tokio::test]
async fn invalid_args_self_correct_then_succeed_without_executing_on_bad_call() -> Result<()> {
    let executions = Arc::new(AtomicUsize::new(0));

    let mut tools = ToolRegistry::new();
    tools.register_typed(GreetTool {
        executions: executions.clone(),
    });

    // Turn 1: model omits the required `greeting` field -> validation error.
    // Turn 2: model self-corrects with well-formed args -> execute runs.
    // Turn 3: model wraps up.
    let provider = StubProvider::new(vec![
        StubProvider::tool_use_response("call_bad", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response(
            "call_good",
            "greet",
            json!({ "name": "Ada", "greeting": "Hello" }),
        ),
        StubProvider::text_response("All done."),
    ]);

    let store = SharedStore::new();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(InMemoryStore::new())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .config(AgentConfig {
            max_turns: Some(8),
            ..AgentConfig::default()
        })
        .build_with_stores();

    let thread_id = ThreadId::new();
    agent
        .run(
            thread_id.clone(),
            AgentInput::Text("greet Ada".into()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    // execute() ran exactly once — only for the well-formed call.
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "execute must run only for the valid call, never for the invalid one"
    );

    let history = store.get_history(&thread_id).await?;

    // Balanced history: one tool_result per tool_use, including the
    // synthesised validation-error result (no orphan tool_use).
    let (uses, results) = count_tool_blocks(&history);
    assert_eq!(uses, 2, "two tool_use blocks were scripted");
    assert_eq!(results, 2, "every tool_use has a matching tool_result");

    // The bad call's result is the structured self-correction error.
    let bad = tool_result_for(&history, "call_bad").context("missing result for call_bad")?;
    let ContentBlock::ToolResult {
        content, is_error, ..
    } = bad
    else {
        anyhow::bail!("expected a tool_result block for call_bad");
    };
    assert_eq!(
        *is_error,
        Some(true),
        "validation failure is an error result"
    );
    assert!(
        content.contains("Invalid arguments for tool `greet`"),
        "validation error must identify the tool: {content}"
    );
    assert!(
        content.contains("greeting"),
        "validation error must surface the missing field name: {content}"
    );

    // The good call actually executed.
    let good = tool_result_for(&history, "call_good").context("missing result for call_good")?;
    let ContentBlock::ToolResult {
        content, is_error, ..
    } = good
    else {
        anyhow::bail!("expected a tool_result block for call_good");
    };
    assert_ne!(*is_error, Some(true), "valid call should succeed");
    assert_eq!(content, "Hello, Ada!");

    Ok(())
}

#[tokio::test]
async fn repeated_invalid_args_terminate_cleanly_under_max_turns() -> Result<()> {
    let executions = Arc::new(AtomicUsize::new(0));

    let mut tools = ToolRegistry::new();
    tools.register_typed(GreetTool {
        executions: executions.clone(),
    });

    // The model keeps emitting invalid args every turn. If the script runs
    // dry the StubProvider returns a terminal text response, but the bound we
    // actually rely on is `max_turns`: a stubborn model can never loop
    // forever.
    let provider = StubProvider::new(vec![
        StubProvider::tool_use_response("call_1", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response("call_2", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response("call_3", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response("call_4", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response("call_5", "greet", json!({ "name": "Ada" })),
        StubProvider::tool_use_response("call_6", "greet", json!({ "name": "Ada" })),
    ]);

    let store = SharedStore::new();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(InMemoryStore::new())
        .event_store(Arc::new(InMemoryEventStore::new()))
        .config(AgentConfig {
            max_turns: Some(3),
            ..AgentConfig::default()
        })
        .build_with_stores();

    let thread_id = ThreadId::new();
    // The run must return (not hang) despite never getting a valid call.
    agent
        .run(
            thread_id.clone(),
            AgentInput::Text("greet Ada".into()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    // execute() never ran with invalid args.
    assert_eq!(
        executions.load(Ordering::SeqCst),
        0,
        "execute must never run when every call is invalid"
    );

    // Even bounded, history stays balanced: no orphan tool_use.
    let history = store.get_history(&thread_id).await?;
    let (uses, results) = count_tool_blocks(&history);
    assert_eq!(
        uses, results,
        "balanced tool_use/tool_result even when truncated by max_turns"
    );

    Ok(())
}
