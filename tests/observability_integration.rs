#![cfg(feature = "otel")]

//! Integration tests for the observability instrumentation.

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Message, StopReason, Usage,
};
use agent_sdk::observability::attrs;
use agent_sdk::{
    AgentEvent, AgentEventEnvelope, AgentInput, AgentState, AllowAllHooks, CancellationToken,
    DynamicToolName, InMemoryStore, LlmProvider, MessageStore, StateStore, ThreadId, Tool,
    ToolContext, ToolRegistry, ToolResult, ToolTier, builder,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use opentelemetry::global;
use opentelemetry::trace::{Status, TraceId};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use tokio::sync::{Mutex, MutexGuard};

/// Tests share the global tracer provider; serialize them.
static TEST_LOCK: Mutex<()> = Mutex::const_new(());

struct TestProvider {
    responses: RwLock<Vec<ChatOutcome>>,
}

impl TestProvider {
    const fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
        }
    }

    fn text_response(text: &str) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp_1".to_string(),
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_input_tokens: 0,
            },
        })
    }

    fn tool_use_response(tool_id: &str, tool_name: &str, input: Value) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp_2".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: tool_id.to_string(),
                name: tool_name.to_string(),
                input,
                thought_signature: None,
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
                cached_input_tokens: 0,
            },
        })
    }
}

#[async_trait]
impl LlmProvider for TestProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let mut responses = self
            .responses
            .write()
            .map_err(|_| anyhow!("lock poisoned"))?;
        if responses.is_empty() {
            Ok(Self::text_response("default"))
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

struct EchoTool;

impl Tool<()> for EchoTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("echo")
    }

    fn display_name(&self) -> &'static str {
        "Echo"
    }

    fn description(&self) -> &'static str {
        "Echoes input"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {"text": {"type": "string"}}})
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let text = input
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("no text");
        Ok(ToolResult::success(text))
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

async fn acquire_test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

fn setup_tracer() -> (SdkTracerProvider, InMemorySpanExporter) {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    global::set_tracer_provider(provider.clone());
    (provider, exporter)
}

fn get_spans(exporter: &InMemorySpanExporter) -> Result<Vec<SpanData>> {
    exporter
        .get_finished_spans()
        .context("failed to read finished spans")
}

fn root_span_for_thread<'a>(spans: &'a [SpanData], thread_id: &ThreadId) -> Result<&'a SpanData> {
    let conversation_id = thread_id.to_string();
    spans
        .iter()
        .find(|span| {
            span.name.as_ref() == "invoke_agent"
                && get_attr(span, attrs::GEN_AI_CONVERSATION_ID).as_deref()
                    == Some(conversation_id.as_str())
        })
        .with_context(|| format!("missing invoke_agent span for thread {conversation_id}"))
}

fn spans_in_trace(spans: &[SpanData], trace_id: TraceId) -> Vec<&SpanData> {
    spans
        .iter()
        .filter(|span| span.span_context.trace_id() == trace_id)
        .collect()
}

fn find_span_in_trace<'a>(spans: &[&'a SpanData], name: &str) -> Result<&'a SpanData> {
    spans
        .iter()
        .copied()
        .find(|span| span.name.as_ref() == name)
        .with_context(|| format!("missing {name} span in trace"))
}

fn get_attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| format!("{}", kv.value))
}

async fn drain_events(mut rx: tokio::sync::mpsc::Receiver<AgentEventEnvelope>) {
    while let Some(envelope) = rx.recv().await {
        if matches!(
            envelope.event,
            AgentEvent::Done { .. } | AgentEvent::Error { .. }
        ) {
            break;
        }
    }
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
}

async fn seed_compaction_history(store: &SharedStore, thread_id: &ThreadId) -> Result<()> {
    store
        .append(thread_id, Message::user("Previous request"))
        .await?;
    store
        .append(thread_id, Message::assistant("Previous response"))
        .await?;
    Ok(())
}

#[tokio::test]
async fn root_span_emitted_for_simple_run() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hello!")]);
    let agent = builder::<()>().provider(provider).build();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    assert_eq!(
        get_attr(root, attrs::GEN_AI_OPERATION_NAME).as_deref(),
        Some("invoke_agent")
    );
    assert_eq!(get_attr(root, attrs::SDK_RUN_MODE).as_deref(), Some("loop"));
    assert_eq!(get_attr(root, attrs::SDK_OUTCOME).as_deref(), Some("done"));
    assert_eq!(
        get_attr(root, attrs::GEN_AI_PROVIDER_NAME).as_deref(),
        Some("anthropic")
    );
    assert_eq!(
        get_attr(root, attrs::SDK_INPUT_KIND).as_deref(),
        Some("text")
    );

    Ok(())
}

#[tokio::test]
async fn turn_span_emitted() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Done")]);
    let agent = builder::<()>().provider(provider).build();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;

    assert_eq!(get_attr(turn, attrs::SDK_TURN_NUMBER).as_deref(), Some("1"));

    Ok(())
}

#[tokio::test]
async fn context_compaction_span_is_child_of_root_span() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    seed_compaction_history(&store, &thread_id).await?;

    let provider = TestProvider::new(vec![
        TestProvider::text_response("Conversation summary"),
        TestProvider::text_response("Done"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .with_compaction(
            agent_sdk::context::CompactionConfig::new()
                .with_threshold_tokens(1)
                .with_min_messages(1)
                .with_retain_recent(1),
        )
        .build_with_stores();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Follow up".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let compaction = find_span_in_trace(&trace_spans, "agent.context_compaction")?;

    assert_eq!(compaction.parent_span_id, root.span_context.span_id());
    assert_eq!(
        compaction.span_context.trace_id(),
        root.span_context.trace_id()
    );
    assert!(!compaction.parent_span_is_remote);
    assert_eq!(
        get_attr(compaction, attrs::SDK_COMPACTION_TRIGGER).as_deref(),
        Some("threshold")
    );
    assert_eq!(
        get_attr(compaction, attrs::SDK_COMPACTION_ORIGINAL_COUNT).as_deref(),
        Some("3")
    );
    assert_eq!(
        get_attr(compaction, attrs::SDK_COMPACTION_NEW_COUNT).as_deref(),
        Some("3")
    );
    assert_eq!(
        get_attr(compaction, attrs::SDK_OUTCOME).as_deref(),
        Some("success")
    );

    Ok(())
}

#[tokio::test]
async fn context_compaction_failure_sets_error_status() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    seed_compaction_history(&store, &thread_id).await?;

    let provider = TestProvider::new(vec![
        ChatOutcome::ServerError("summary backend unavailable".to_string()),
        TestProvider::text_response("Done"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .with_compaction(
            agent_sdk::context::CompactionConfig::new()
                .with_threshold_tokens(1)
                .with_min_messages(1)
                .with_retain_recent(1),
        )
        .build_with_stores();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Follow up".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let compaction = find_span_in_trace(&trace_spans, "agent.context_compaction")?;

    assert_eq!(compaction.parent_span_id, root.span_context.span_id());
    assert_eq!(
        get_attr(compaction, attrs::SDK_COMPACTION_TRIGGER).as_deref(),
        Some("threshold")
    );
    assert_eq!(
        get_attr(compaction, attrs::ERROR_TYPE).as_deref(),
        Some("context_compaction_failed")
    );
    assert_eq!(
        get_attr(compaction, attrs::SDK_OUTCOME).as_deref(),
        Some("error")
    );
    assert!(matches!(&compaction.status, Status::Error { .. }));

    Ok(())
}

#[tokio::test]
async fn llm_span_emitted_with_model_name() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>().provider(provider).build();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert_eq!(
        get_attr(llm, attrs::GEN_AI_OPERATION_NAME).as_deref(),
        Some("chat")
    );
    assert_eq!(
        get_attr(llm, attrs::GEN_AI_RESPONSE_MODEL).as_deref(),
        Some("test-model")
    );
    assert!(get_attr(llm, attrs::GEN_AI_USAGE_INPUT_TOKENS).is_some());
    assert!(get_attr(llm, attrs::GEN_AI_USAGE_OUTPUT_TOKENS).is_some());

    Ok(())
}

#[tokio::test]
async fn tool_span_emitted_with_tool_name() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hello"})),
        TestProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .build_with_stores();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Echo something".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;

    assert_eq!(
        get_attr(tool, attrs::GEN_AI_TOOL_NAME).as_deref(),
        Some("echo")
    );
    assert_eq!(
        get_attr(tool, attrs::GEN_AI_TOOL_CALL_ID).as_deref(),
        Some("call_1")
    );
    assert_eq!(
        get_attr(tool, attrs::SDK_TOOL_OUTCOME).as_deref(),
        Some("success")
    );
    assert_eq!(
        get_attr(tool, attrs::SDK_TOOL_TIER).as_deref(),
        Some("observe")
    );
    assert_eq!(
        get_attr(tool, attrs::SDK_TOOL_KIND).as_deref(),
        Some("sync")
    );

    Ok(())
}

#[tokio::test]
async fn unknown_tool_span_has_error_type() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "nonexistent", json!({})),
        TestProvider::text_response("Done"),
    ]);

    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .build_with_stores();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Use nonexistent tool".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;

    assert_eq!(
        get_attr(tool, attrs::ERROR_TYPE).as_deref(),
        Some("unknown_tool")
    );
    assert_eq!(
        get_attr(tool, attrs::SDK_TOOL_OUTCOME).as_deref(),
        Some("error")
    );

    Ok(())
}

#[tokio::test]
async fn provider_name_normalized_on_root_span() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>().provider(provider).build();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    assert_eq!(
        get_attr(root, attrs::GEN_AI_PROVIDER_NAME).as_deref(),
        Some("anthropic")
    );
    assert_eq!(
        get_attr(root, attrs::SDK_PROVIDER_ID).as_deref(),
        Some("anthropic")
    );

    Ok(())
}

#[tokio::test]
async fn single_turn_mode_sets_run_mode() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>().provider(provider).build();
    let thread_id = ThreadId::new();
    let (rx, outcome_rx) = agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = outcome_rx.await.context("turn outcome channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    assert_eq!(
        get_attr(root, attrs::SDK_RUN_MODE).as_deref(),
        Some("single_turn")
    );

    Ok(())
}

#[tokio::test]
async fn all_span_types_present_for_tool_call_flow() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hello"})),
        TestProvider::text_response("Final answer"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .build_with_stores();
    let thread_id = ThreadId::new();
    let (rx, final_state) = agent.run(
        thread_id.clone(),
        AgentInput::Text("Test".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    drain_events(rx).await;
    let _ = final_state.await.context("agent state channel closed")?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let span_names: Vec<&str> = trace_spans.iter().map(|span| span.name.as_ref()).collect();

    assert!(
        span_names.contains(&"invoke_agent"),
        "missing invoke_agent: {span_names:?}"
    );
    assert!(
        span_names.contains(&"agent.turn"),
        "missing agent.turn: {span_names:?}"
    );
    assert!(
        span_names.iter().any(|name| name.starts_with("chat ")),
        "missing chat span: {span_names:?}"
    );
    assert!(
        span_names.contains(&"execute_tool"),
        "missing execute_tool: {span_names:?}"
    );

    Ok(())
}
