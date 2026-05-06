#![cfg(feature = "otel")]

//! Integration tests for the observability instrumentation.

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Message, StopReason, Usage,
};
use agent_sdk::observability::{
    CaptureDecision, CaptureResult, ObservabilityStore, PayloadBundle, attrs, langfuse,
};
use agent_sdk::{
    AgentInput, AgentState, AllowAllHooks, CancellationToken, DynamicToolName, InMemoryEventStore,
    InMemoryStore, LlmProvider, MessageStore, StateStore, ThreadId, Tool, ToolContext,
    ToolRegistry, ToolResult, ToolTier, TurnOptions, builder,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use opentelemetry::global;
use opentelemetry::trace::{Status, TraceId};
use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider, SpanData};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
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
                cache_creation_input_tokens: 0,
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
                cache_creation_input_tokens: 0,
            },
        })
    }

    fn text_response_with_usage(text: &str, usage: Usage) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp_cached".to_string(),
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage,
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

struct InlinePayloadStore;

#[async_trait]
impl ObservabilityStore for InlinePayloadStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> Result<CaptureResult> {
        Ok(CaptureResult {
            system_instructions: CaptureDecision::Inline,
            input_messages: CaptureDecision::Inline,
            output_messages: CaptureDecision::Inline,
        })
    }
}

struct FailingPayloadStore;

#[async_trait]
impl ObservabilityStore for FailingPayloadStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> Result<CaptureResult> {
        Err(anyhow!("payload capture failed"))
    }
}

struct CountingPayloadStore {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ObservabilityStore for CountingPayloadStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> Result<CaptureResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CaptureResult {
            system_instructions: CaptureDecision::Omit,
            input_messages: CaptureDecision::Omit,
            output_messages: CaptureDecision::Omit,
        })
    }
}

async fn acquire_test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

fn setup_tracer() -> (SdkTracerProvider, InMemorySpanExporter) {
    setup_tracer_with_sampler(Sampler::AlwaysOn)
}

fn setup_tracer_with_sampler(sampler: Sampler) -> (SdkTracerProvider, InMemorySpanExporter) {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_sampler(sampler)
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

fn get_observation_type(span: &SpanData) -> Option<String> {
    get_attr(span, langfuse::LANGFUSE_OBSERVATION_TYPE)
}

fn parse_json_attr(span: &SpanData, key: &str) -> Result<Value> {
    let raw = get_attr(span, key).with_context(|| format!("missing {key} attribute"))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {key}: {raw}"))
}

fn span_has_event(span: &SpanData, event_name: &str) -> bool {
    span.events
        .iter()
        .any(|event| event.name.as_ref() == event_name)
}

fn new_event_store() -> Arc<InMemoryEventStore> {
    Arc::new(InMemoryEventStore::new())
}

async fn wait_for_run(
    final_state: tokio::sync::oneshot::Receiver<agent_sdk::AgentRunState>,
) -> Result<()> {
    let _ = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    Ok(())
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
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(event_store)
        .with_compaction(
            agent_sdk::context::CompactionConfig::new()
                .with_threshold_tokens(1)
                .with_min_messages(1)
                .with_retain_recent(1),
        )
        .build_with_stores();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Follow up".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    assert_eq!(
        get_observation_type(compaction).as_deref(),
        Some("chain"),
        "agent.context_compaction must be tagged ObservationType::Chain"
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
    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(event_store)
        .with_compaction(
            agent_sdk::context::CompactionConfig::new()
                .with_threshold_tokens(1)
                .with_min_messages(1)
                .with_retain_recent(1),
        )
        .build_with_stores();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Follow up".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
async fn llm_span_emits_cached_token_attributes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response_with_usage(
        "cached",
        Usage {
            input_tokens: 180,
            output_tokens: 50,
            cached_input_tokens: 20,
            cache_creation_input_tokens: 10,
        },
    )]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert_eq!(
        get_attr(llm, attrs::GEN_AI_USAGE_INPUT_TOKENS).as_deref(),
        Some("180"),
    );
    assert_eq!(
        get_attr(llm, attrs::GEN_AI_USAGE_OUTPUT_TOKENS).as_deref(),
        Some("50"),
    );
    assert_eq!(
        get_attr(llm, attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS).as_deref(),
        Some("20"),
    );
    assert_eq!(
        get_attr(llm, attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS).as_deref(),
        Some("10"),
    );

    Ok(())
}

#[tokio::test]
async fn root_span_emits_aggregated_cached_token_attributes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response_with_usage(
        "cached",
        Usage {
            input_tokens: 180,
            output_tokens: 50,
            cached_input_tokens: 20,
            cache_creation_input_tokens: 10,
        },
    )]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    assert_eq!(
        get_attr(root, attrs::GEN_AI_USAGE_INPUT_TOKENS).as_deref(),
        Some("180"),
    );
    assert_eq!(
        get_attr(root, attrs::GEN_AI_USAGE_OUTPUT_TOKENS).as_deref(),
        Some("50"),
    );
    assert_eq!(
        get_attr(root, attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS).as_deref(),
        Some("20"),
    );
    assert_eq!(
        get_attr(root, attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS).as_deref(),
        Some("10"),
    );

    Ok(())
}

#[tokio::test]
async fn turn_span_emits_cached_token_attributes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response_with_usage(
        "cached",
        Usage {
            input_tokens: 180,
            output_tokens: 50,
            cached_input_tokens: 20,
            cache_creation_input_tokens: 10,
        },
    )]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;

    assert_eq!(
        get_attr(turn, attrs::SDK_TURN_INPUT_TOKENS).as_deref(),
        Some("180"),
    );
    assert_eq!(
        get_attr(turn, attrs::SDK_TURN_OUTPUT_TOKENS).as_deref(),
        Some("50"),
    );
    assert_eq!(
        get_attr(turn, attrs::SDK_TURN_CACHE_READ_INPUT_TOKENS).as_deref(),
        Some("20"),
    );
    assert_eq!(
        get_attr(turn, attrs::SDK_TURN_CACHE_CREATION_INPUT_TOKENS).as_deref(),
        Some("10"),
    );

    Ok(())
}

#[tokio::test]
async fn inline_payload_store_records_input_and_output_messages() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi there")]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(InlinePayloadStore)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    let input_messages = parse_json_attr(llm, attrs::GEN_AI_INPUT_MESSAGES)?;
    let output_messages = parse_json_attr(llm, attrs::GEN_AI_OUTPUT_MESSAGES)?;
    let input = input_messages
        .as_array()
        .and_then(|messages| messages.first())
        .context("missing first input message")?;
    let output = output_messages
        .as_array()
        .and_then(|messages| messages.first())
        .context("missing first output message")?;

    assert_eq!(input["role"], "user");
    assert_eq!(input["content"][0]["text"], "Hello");
    assert_eq!(output["role"], "assistant");
    assert_eq!(output["content"][0]["text"], "Hi there");

    Ok(())
}

#[tokio::test]
async fn failing_payload_store_does_not_fail_agent() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Still works")]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(FailingPayloadStore)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let state = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    assert!(matches!(state, agent_sdk::AgentRunState::Done { .. }));

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert!(span_has_event(llm, "payload_capture_failed"));
    assert!(get_attr(llm, attrs::GEN_AI_INPUT_MESSAGES).is_none());
    assert!(get_attr(llm, attrs::GEN_AI_OUTPUT_MESSAGES).is_none());

    Ok(())
}

#[tokio::test]
async fn payload_store_is_called_for_non_recording_spans() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, _exporter) = setup_tracer_with_sampler(Sampler::AlwaysOff);

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(CountingPayloadStore {
            calls: Arc::clone(&calls),
        })
        .event_store(new_event_store())
        .build();
    let final_state = agent.run(
        ThreadId::new(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    assert_eq!(calls.load(Ordering::SeqCst), 1);

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

    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(event_store)
        .build_with_stores();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Echo something".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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

    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(event_store)
        .build_with_stores();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Use nonexistent tool".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let _ = agent
        .run_turn(
            thread_id.clone(),
            AgentInput::Text("Hello".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
            TurnOptions::default(),
        )
        .await;
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

    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(event_store)
        .build_with_stores();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Test".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
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

    // Langfuse observation-type tagging (A4): every span the SDK emits
    // must carry the documented `langfuse.observation.type` so a
    // vanilla consumer pointing OTel at Langfuse gets the right icons
    // (Agent / Generation / Tool) without writing any glue.
    let invoke = find_span_in_trace(&trace_spans, "invoke_agent")?;
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;
    assert_eq!(
        get_observation_type(invoke).as_deref(),
        Some("agent"),
        "invoke_agent must be tagged ObservationType::Agent"
    );
    assert_eq!(
        get_observation_type(chat).as_deref(),
        Some("generation"),
        "chat span must be tagged ObservationType::Generation"
    );
    assert_eq!(
        get_observation_type(tool).as_deref(),
        Some("tool"),
        "execute_tool span must be tagged ObservationType::Tool"
    );

    Ok(())
}

// ── Baggage propagation (A3) ─────────────────────────────────────────

use agent_sdk::observability::baggage as obs_baggage;
use opentelemetry::Context as OtelContext;
use opentelemetry::baggage::BaggageExt;

const ALL_BAGGAGE: &[(&str, &str)] = &[
    (obs_baggage::BAGGAGE_USER_ID, "user-42"),
    (obs_baggage::BAGGAGE_SESSION_ID, "session-7"),
    (obs_baggage::BAGGAGE_LANGFUSE_USER_ID, "lf-user-42"),
    (obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID, "lf-session-7"),
    (obs_baggage::BAGGAGE_DEPLOYMENT_ENVIRONMENT, "test"),
];

fn baggage_context(entries: &[(&'static str, &'static str)]) -> OtelContext {
    let kvs: Vec<opentelemetry::KeyValue> = entries
        .iter()
        .map(|(k, v)| opentelemetry::KeyValue::new(*k, *v))
        .collect();
    OtelContext::current_with_baggage(kvs)
}

/// Run a future under the supplied otel context. Local `use` of
/// `FutureExt` keeps the rest of the file's `with_context` calls
/// (which target `anyhow::Context`) unambiguous.
async fn run_with_baggage<F, T>(cx: OtelContext, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    use opentelemetry::trace::FutureExt;
    fut.with_context(cx).await
}

fn assert_attr_eq(span: &SpanData, key: &str, expected: &str) {
    assert_eq!(
        get_attr(span, key).as_deref(),
        Some(expected),
        "expected {key}={expected} on span {:?}",
        span.name
    );
}

fn assert_attr_absent(span: &SpanData, key: &str) {
    assert!(
        get_attr(span, key).is_none(),
        "expected {key} to be absent on span {:?}, got {:?}",
        span.name,
        get_attr(span, key)
    );
}

#[tokio::test]
async fn baggage_attributes_copied_to_every_span() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        TestProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let event_store = new_event_store();
    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(event_store)
        .build_with_stores();
    let thread_id = ThreadId::new();
    let cx = baggage_context(ALL_BAGGAGE);

    run_with_baggage(cx, async {
        let final_state = agent.run(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        anyhow::Ok(())
    })
    .await?;

    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    let names_to_check = [
        "invoke_agent",
        "agent.turn",
        "chat test-model",
        "execute_tool",
    ];
    for name in names_to_check {
        let span = find_span_in_trace(&trace_spans, name)?;
        for (key, value) in ALL_BAGGAGE {
            assert_attr_eq(span, key, value);
        }
    }

    Ok(())
}

#[tokio::test]
async fn baggage_session_id_mirrored_to_gen_ai_conversation_id() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let session_value = "lf-session-mirror";
    let cx = baggage_context(&[(obs_baggage::BAGGAGE_SESSION_ID, session_value)]);

    run_with_baggage(cx, async {
        let final_state = agent.run(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        anyhow::Ok(())
    })
    .await?;

    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;

    for span in [llm, turn] {
        assert_attr_eq(span, obs_baggage::BAGGAGE_SESSION_ID, session_value);
        assert_attr_eq(span, attrs::GEN_AI_CONVERSATION_ID, session_value);
    }

    Ok(())
}

#[tokio::test]
async fn baggage_partial_only_user_id() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let cx = baggage_context(&[(obs_baggage::BAGGAGE_USER_ID, "only-user")]);

    run_with_baggage(cx, async {
        let final_state = agent.run(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        anyhow::Ok(())
    })
    .await?;

    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    for name in ["invoke_agent", "agent.turn", "chat test-model"] {
        let span = find_span_in_trace(&trace_spans, name)?;
        assert_attr_eq(span, obs_baggage::BAGGAGE_USER_ID, "only-user");
        assert_attr_absent(span, obs_baggage::BAGGAGE_SESSION_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_LANGFUSE_USER_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_DEPLOYMENT_ENVIRONMENT);
    }

    Ok(())
}

#[tokio::test]
async fn baggage_absent_no_attributes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    for name in ["invoke_agent", "agent.turn", "chat test-model"] {
        let span = find_span_in_trace(&trace_spans, name)?;
        assert_attr_absent(span, obs_baggage::BAGGAGE_USER_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_SESSION_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_LANGFUSE_USER_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID);
        assert_attr_absent(span, obs_baggage::BAGGAGE_DEPLOYMENT_ENVIRONMENT);
    }

    Ok(())
}

#[tokio::test]
async fn baggage_survives_tokio_spawn() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    // The agent spawns a tokio task internally for `run()`. The baggage we
    // attach in this test task must survive that spawn (the SDK wraps the
    // spawned future with `FutureExt::with_context(parent_cx)`). If that
    // contract regresses, the user.id attribute will go missing on every
    // span that was emitted from the spawned task.
    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let cx = baggage_context(&[(obs_baggage::BAGGAGE_USER_ID, "spawn-user")]);

    // The spawned task does the actual work — wait for it inside the
    // contextualised future so the baggage is in scope until the agent
    // has emitted every span.
    run_with_baggage(cx, async {
        let final_state = agent.run(
            thread_id.clone(),
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        anyhow::Ok(())
    })
    .await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    // Spans below the root were emitted from inside the spawned task.
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert_attr_eq(root, obs_baggage::BAGGAGE_USER_ID, "spawn-user");
    assert_attr_eq(turn, obs_baggage::BAGGAGE_USER_ID, "spawn-user");
    assert_attr_eq(llm, obs_baggage::BAGGAGE_USER_ID, "spawn-user");

    Ok(())
}

#[tokio::test]
async fn baggage_helpers_attach_and_preserve_existing_entries() -> Result<()> {
    // The `with_user_id` / `with_session_id` helpers must not clobber
    // unrelated baggage entries already on the context.
    let cx = OtelContext::current_with_baggage([opentelemetry::KeyValue::new("trace.tag", "v1")]);
    let cx = obs_baggage::with_user_id(&cx, "alice");
    let cx = obs_baggage::with_session_id(&cx, "session-1");

    assert_eq!(
        cx.baggage().get("trace.tag").map(ToString::to_string),
        Some("v1".to_string())
    );
    assert_eq!(
        cx.baggage()
            .get(obs_baggage::BAGGAGE_USER_ID)
            .map(ToString::to_string),
        Some("alice".to_string())
    );
    assert_eq!(
        cx.baggage()
            .get(obs_baggage::BAGGAGE_SESSION_ID)
            .map(ToString::to_string),
        Some("session-1".to_string())
    );

    Ok(())
}

// ── RunOptions / Langfuse trace metadata (A5) ────────────────────────

use agent_sdk::RunOptions;

fn run_options_with_session(session: &str) -> RunOptions {
    RunOptions {
        session_id: Some(session.to_string()),
        ..RunOptions::default()
    }
}

#[tokio::test]
async fn run_options_stamp_langfuse_trace_metadata_on_root_span() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hello!")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "release_channel".to_string(),
        Value::String("beta".to_string()),
    );
    metadata.insert("user_count".to_string(), json!(42));

    let opts = RunOptions {
        session_id: Some("session-A5".to_string()),
        user_id: Some("user-A5".to_string()),
        trace_name: Some("a5.test.run".to_string()),
        trace_tags: vec!["mobile.android".to_string(), "experiment.b".to_string()],
        trace_metadata: metadata,
        release: Some("1.2.3".to_string()),
        environment: Some("staging".to_string()),
        trace_text_max_chars: None,
    };

    let final_state = agent.run_with_options(
        thread_id.clone(),
        AgentInput::Text("Hi from A5".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    // Trace name is stamped verbatim.
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_TRACE_NAME).as_deref(),
        Some("a5.test.run"),
    );

    // Tags become a comma-joined string (the OTel SDK stringifies the
    // string-array attribute consistently for `assert_eq!`).
    assert!(
        get_attr(root, langfuse::LANGFUSE_TRACE_TAGS)
            .is_some_and(|v| v.contains("mobile.android") && v.contains("experiment.b")),
        "trace tags missing or malformed: {:?}",
        get_attr(root, langfuse::LANGFUSE_TRACE_TAGS),
    );

    // Each metadata entry lands under `langfuse.trace.metadata.<key>`.
    assert_eq!(
        get_attr(
            root,
            &format!(
                "{}{}",
                langfuse::LANGFUSE_TRACE_METADATA_PREFIX,
                "release_channel"
            ),
        )
        .as_deref(),
        Some("beta"),
    );
    assert_eq!(
        get_attr(
            root,
            &format!(
                "{}{}",
                langfuse::LANGFUSE_TRACE_METADATA_PREFIX,
                "user_count"
            ),
        )
        .as_deref(),
        Some("42"),
    );

    // `release` and `environment` map to the canonical Langfuse attrs.
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_RELEASE).as_deref(),
        Some("1.2.3"),
    );
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_ENVIRONMENT).as_deref(),
        Some("staging"),
    );

    // Session/user become baggage entries that the existing
    // `copy_baggage_to_active_span` helper mirrors onto the span.
    assert_eq!(
        get_attr(root, obs_baggage::BAGGAGE_SESSION_ID).as_deref(),
        Some("session-A5"),
    );
    assert_eq!(
        get_attr(root, obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID).as_deref(),
        Some("session-A5"),
    );
    assert_eq!(
        get_attr(root, obs_baggage::BAGGAGE_USER_ID).as_deref(),
        Some("user-A5"),
    );
    assert_eq!(
        get_attr(root, obs_baggage::BAGGAGE_LANGFUSE_USER_ID).as_deref(),
        Some("user-A5"),
    );

    // Trace input mirrors the `AgentInput` summary.
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_TRACE_INPUT).as_deref(),
        Some("Hi from A5"),
    );

    Ok(())
}

#[tokio::test]
async fn run_options_default_omits_caller_supplied_trace_metadata() -> Result<()> {
    // Backwards compatibility: `agent.run(...)` (no options) must
    // continue to emit the same span surface as before A5 — the
    // caller-supplied Langfuse fields (`trace.name`, `trace.tags`,
    // `release`, `environment`, and the metadata prefix) MUST stay
    // absent.
    //
    // `langfuse.trace.input` and `langfuse.trace.output` are
    // populated unconditionally because the SDK now lifts that
    // computation away from consumers — that's covered by the
    // dedicated `_accumulate_trace_output_from_events` test.
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hello".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    assert_attr_absent(root, langfuse::LANGFUSE_TRACE_NAME);
    assert_attr_absent(root, langfuse::LANGFUSE_TRACE_TAGS);
    assert_attr_absent(root, langfuse::LANGFUSE_RELEASE);
    assert_attr_absent(root, langfuse::LANGFUSE_ENVIRONMENT);
    assert_attr_absent(
        root,
        &format!("{}{}", langfuse::LANGFUSE_TRACE_METADATA_PREFIX, "anything"),
    );

    Ok(())
}

#[tokio::test]
async fn run_options_accumulate_trace_output_from_events() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hello"})),
        TestProvider::text_response("All done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .build_with_stores();
    let thread_id = ThreadId::new();

    let final_state = agent.run_with_options(
        thread_id.clone(),
        AgentInput::Text("Echo and finish".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        run_options_with_session("trace-output-session"),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    let trace_output = get_attr(root, langfuse::LANGFUSE_TRACE_OUTPUT)
        .context("missing langfuse.trace.output on root span")?;

    // The accumulator labels every chunk it ingests. We expect at
    // minimum: the assistant text, the tool call summary, and the
    // tool result body.
    assert!(
        trace_output.contains("[Assistant]"),
        "expected [Assistant] label, got: {trace_output}",
    );
    assert!(
        trace_output.contains("All done"),
        "expected assistant text, got: {trace_output}",
    );
    assert!(
        trace_output.contains("[Tool Call]"),
        "expected [Tool Call] label, got: {trace_output}",
    );
    assert!(
        trace_output.contains("echo"),
        "expected tool name in trace output, got: {trace_output}",
    );
    assert!(
        trace_output.contains("[Tool Result]"),
        "expected [Tool Result] label, got: {trace_output}",
    );
    assert!(
        trace_output.contains("hello"),
        "expected tool result body, got: {trace_output}",
    );

    Ok(())
}

#[tokio::test]
async fn run_turn_with_options_stamps_metadata_in_single_turn_mode() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![TestProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    let mut metadata = serde_json::Map::new();
    metadata.insert("flag".to_string(), Value::String("enabled".to_string()));
    let opts = RunOptions {
        session_id: Some("turn-session".to_string()),
        user_id: None,
        trace_name: Some("single-turn".to_string()),
        trace_tags: Vec::new(),
        trace_metadata: metadata,
        release: None,
        environment: None,
        trace_text_max_chars: None,
    };

    let _ = agent
        .run_turn_with_options(
            thread_id.clone(),
            AgentInput::Text("One-shot".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
            TurnOptions::default(),
            opts,
        )
        .await;

    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    assert_eq!(
        get_attr(root, attrs::SDK_RUN_MODE).as_deref(),
        Some("single_turn"),
    );
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_TRACE_NAME).as_deref(),
        Some("single-turn"),
    );
    assert_eq!(
        get_attr(
            root,
            &format!("{}{}", langfuse::LANGFUSE_TRACE_METADATA_PREFIX, "flag"),
        )
        .as_deref(),
        Some("enabled"),
    );
    assert_eq!(
        get_attr(root, langfuse::LANGFUSE_TRACE_INPUT).as_deref(),
        Some("One-shot"),
    );
    assert_eq!(
        get_attr(root, obs_baggage::BAGGAGE_SESSION_ID).as_deref(),
        Some("turn-session"),
    );

    Ok(())
}

#[tokio::test]
async fn run_options_truncate_trace_text_at_caller_supplied_max() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let long: String = "x".repeat(200);
    let provider = TestProvider::new(vec![TestProvider::text_response(&long)]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();

    // Cap trace text at a small budget so we can observe the
    // truncation marker on the accumulated `langfuse.trace.output`.
    let opts = RunOptions {
        session_id: None,
        user_id: None,
        trace_name: None,
        trace_tags: Vec::new(),
        trace_metadata: serde_json::Map::new(),
        release: None,
        environment: None,
        trace_text_max_chars: Some(32),
    };

    let final_state = agent.run_with_options(
        thread_id.clone(),
        AgentInput::Text("Long output".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    let trace_output = get_attr(root, langfuse::LANGFUSE_TRACE_OUTPUT)
        .context("missing langfuse.trace.output on root span")?;
    assert!(
        trace_output.chars().count() <= 32,
        "trace output exceeded budget ({} chars): {trace_output}",
        trace_output.chars().count(),
    );
    assert!(
        trace_output.ends_with('…'),
        "expected ellipsis truncation marker, got: {trace_output}",
    );

    Ok(())
}

// ── Span events (A6) ─────────────────────────────────────────────────

use agent_sdk::llm::{StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk::{AgentConfig, AgentRunState};
use async_stream::stream;
use std::time::Duration;

/// LLM provider that emits a scripted streaming sequence.
///
/// Each chunk in `script` is yielded after a tiny sleep so the
/// test exercises the live streaming path (first-chunk → completed
/// or first-chunk → dropped) rather than the synthetic fallback that
/// `LlmProvider::chat_stream` synthesises from `chat()`.
struct ScriptedStreamProvider {
    script: RwLock<Vec<StreamDelta>>,
}

impl ScriptedStreamProvider {
    const fn new(script: Vec<StreamDelta>) -> Self {
        Self {
            script: RwLock::new(script),
        }
    }
}

#[async_trait]
impl LlmProvider for ScriptedStreamProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Err(anyhow!(
            "ScriptedStreamProvider.chat() should not be called"
        ))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let script = self
            .script
            .write()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        Box::pin(stream! {
            for delta in script {
                tokio::time::sleep(Duration::from_millis(1)).await;
                yield Ok(delta);
            }
        })
    }

    fn model(&self) -> &'static str {
        "test-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

fn streaming_config() -> AgentConfig {
    AgentConfig {
        streaming: true,
        ..AgentConfig::default()
    }
}

#[tokio::test]
async fn llm_span_emits_stream_lifecycle_events() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = ScriptedStreamProvider::new(vec![
        StreamDelta::TextDelta {
            delta: "hi".to_string(),
            block_index: 0,
        },
        StreamDelta::TextDelta {
            delta: " there".to_string(),
            block_index: 0,
        },
        StreamDelta::Usage(Usage {
            input_tokens: 1,
            output_tokens: 2,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }),
        StreamDelta::Done {
            stop_reason: Some(StopReason::EndTurn),
        },
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .config(streaming_config())
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert!(
        span_has_event(llm, "llm.stream.first_chunk"),
        "expected llm.stream.first_chunk on chat span, got events: {:?}",
        llm.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );
    assert!(
        span_has_event(llm, "llm.stream.completed"),
        "expected llm.stream.completed on chat span, got events: {:?}",
        llm.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );

    Ok(())
}

#[tokio::test]
async fn llm_span_emits_dropped_event_when_stream_aborts() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    // No retries so the fatal stream error short-circuits cleanly.
    let mut config = streaming_config();
    config.retry = agent_sdk::RetryConfig::no_retry();

    let provider = ScriptedStreamProvider::new(vec![
        StreamDelta::TextDelta {
            delta: "partial".to_string(),
            block_index: 0,
        },
        StreamDelta::Error {
            message: "boom".to_string(),
            kind: StreamErrorKind::InvalidRequest,
        },
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .config(config)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let state = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tp.force_flush()
        .context("failed to flush tracer provider")?;
    assert!(matches!(state, AgentRunState::Error(_)));

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let llm = find_span_in_trace(&trace_spans, "chat test-model")?;

    assert!(
        span_has_event(llm, "llm.stream.dropped"),
        "expected llm.stream.dropped on chat span, got events: {:?}",
        llm.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );
    let dropped = llm
        .events
        .iter()
        .find(|e| e.name.as_ref() == "llm.stream.dropped")
        .context("dropped event")?;
    let attrs: std::collections::HashMap<String, String> = dropped
        .attributes
        .iter()
        .map(|kv| (kv.key.to_string(), format!("{}", kv.value)))
        .collect();
    assert_eq!(
        attrs
            .get(attrs::SDK_LLM_STREAM_DROP_REASON)
            .map(String::as_str),
        Some("fatal_error"),
    );
    assert_eq!(
        attrs.get(attrs::ERROR_TYPE).map(String::as_str),
        Some("invalid_request"),
    );

    Ok(())
}

#[tokio::test]
async fn root_span_emits_max_turns_reached_event() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    // max_turns=1 + a tool-use response forces the next iteration to
    // hit `begin_turn`'s `ctx.turn > max_turns` branch on turn 2.
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        // Padding entries — never reached because max_turns aborts first.
        TestProvider::text_response("Done"),
    ]);

    let config = AgentConfig {
        max_turns: Some(1),
        ..AgentConfig::default()
    };

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .event_store(new_event_store())
        .config(config)
        .build_with_stores();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Trigger max turns".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let state = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tp.force_flush()
        .context("failed to flush tracer provider")?;
    assert!(matches!(state, AgentRunState::Error(_)));

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    assert!(
        span_has_event(root, "agent.max_turns_reached"),
        "expected agent.max_turns_reached on root span, got events: {:?}",
        root.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );

    Ok(())
}

#[tokio::test]
async fn root_span_emits_context_window_exceeded_event() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![ChatOutcome::Success(ChatResponse {
        id: "resp_ctx".to_string(),
        content: vec![ContentBlock::Text {
            text: "too big".to_string(),
        }],
        model: "test-model".to_string(),
        stop_reason: Some(StopReason::ModelContextWindowExceeded),
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
    })]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Hi".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let state = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tp.force_flush()
        .context("failed to flush tracer provider")?;
    assert!(matches!(state, AgentRunState::Error(_)));

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    assert!(
        span_has_event(root, "agent.context_window_exceeded"),
        "expected agent.context_window_exceeded on root span, got events: {:?}",
        root.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );

    Ok(())
}

// Hook that always asks for confirmation — used by the
// `tool_span_emits_confirmation_required_event` test.
use agent_sdk::{AgentHooks, ToolDecision, ToolInvocation};

#[derive(Default)]
struct ConfirmAllHooks;

#[async_trait]
impl AgentHooks for ConfirmAllHooks {
    async fn pre_tool_use(&self, _invocation: &ToolInvocation) -> ToolDecision {
        ToolDecision::RequiresConfirmation("Confirm please".to_string())
    }
}

#[tokio::test]
async fn tool_span_emits_confirmation_required_event() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let provider = TestProvider::new(vec![
        TestProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        TestProvider::text_response("Done"),
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
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("Echo".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let state = final_state.await.context("agent state channel closed")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tp.force_flush()
        .context("failed to flush tracer provider")?;
    assert!(matches!(state, AgentRunState::AwaitingConfirmation { .. }));

    let spans = get_spans(&exporter)?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;
    assert!(
        span_has_event(tool, "tool.confirmation_required"),
        "expected tool.confirmation_required on tool span, got events: {:?}",
        tool.events
            .iter()
            .map(|e| e.name.as_ref())
            .collect::<Vec<_>>(),
    );

    Ok(())
}
// ── Metrics (B1) ─────────────────────────────────────────────────────

mod metrics {
    use super::*;

    use agent_sdk::observability::metrics::Metrics;
    use opentelemetry::KeyValue as MetricKv;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    /// Install a fresh meter provider + in-memory exporter and reset
    /// the cached `Metrics` singleton so the next `Metrics::global()`
    /// call rebuilds against this provider. Tests share a process so
    /// every metric case must call this before recording.
    fn setup_meter() -> (SdkMeterProvider, InMemoryMetricExporter) {
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        opentelemetry::global::set_meter_provider(provider.clone());
        Metrics::reset_for_testing();
        (provider, exporter)
    }

    fn collected(exporter: &InMemoryMetricExporter) -> Result<Vec<ResourceMetrics>> {
        exporter
            .get_finished_metrics()
            .context("failed to read collected metrics")
    }

    /// Walk the per-test `ResourceMetrics` snapshot and return every
    /// histogram data point recorded for `metric_name` under any
    /// scope. Bypasses the scope-name layer because tests don't care
    /// which meter scope produced the record — they only assert on
    /// the metric name + attributes.
    fn collect_histogram_attrs(
        snapshots: &[ResourceMetrics],
        metric_name: &str,
    ) -> Vec<Vec<(String, String)>> {
        let mut out = Vec::new();
        for resource in snapshots {
            for scope in resource.scope_metrics() {
                for metric in scope.metrics() {
                    if metric.name() != metric_name {
                        continue;
                    }
                    match metric.data() {
                        AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                            for dp in h.data_points() {
                                out.push(kv_pairs(dp.attributes()));
                            }
                        }
                        AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                            for dp in h.data_points() {
                                out.push(kv_pairs(dp.attributes()));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        out
    }

    /// Sum-instrument equivalent of [`collect_histogram_attrs`].
    /// Counters land in the `Sum` variant; we fold all attribute
    /// vectors so the test asserts on label combinations the
    /// recorder produced.
    fn collect_counter_attrs(
        snapshots: &[ResourceMetrics],
        metric_name: &str,
    ) -> Vec<Vec<(String, String)>> {
        let mut out = Vec::new();
        for resource in snapshots {
            for scope in resource.scope_metrics() {
                for metric in scope.metrics() {
                    if metric.name() != metric_name {
                        continue;
                    }
                    match metric.data() {
                        AggregatedMetrics::U64(MetricData::Sum(sum)) => {
                            for dp in sum.data_points() {
                                out.push(kv_pairs(dp.attributes()));
                            }
                        }
                        AggregatedMetrics::F64(MetricData::Sum(sum)) => {
                            for dp in sum.data_points() {
                                out.push(kv_pairs(dp.attributes()));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        out
    }

    fn kv_pairs<'a>(iter: impl Iterator<Item = &'a MetricKv>) -> Vec<(String, String)> {
        iter.map(|kv| (kv.key.as_str().to_string(), format!("{}", kv.value)))
            .collect()
    }

    fn has_label(set: &[(String, String)], key: &str, value: &str) -> bool {
        set.iter()
            .any(|(k, v)| k.as_str() == key && v.as_str() == value)
    }

    fn matches_all(set: &[(String, String)], expected: &[(&str, &str)]) -> bool {
        expected.iter().all(|(k, v)| has_label(set, k, v))
    }

    #[tokio::test]
    async fn metrics_records_token_usage_per_type() -> Result<()> {
        let _guard = acquire_test_lock().await;
        let (_tp, _exporter) = setup_tracer();
        let (mp, mexporter) = setup_meter();

        let provider = TestProvider::new(vec![TestProvider::text_response("Done")]);
        let agent = builder::<()>()
            .provider(provider)
            .event_store(new_event_store())
            .build();
        let thread_id = ThreadId::new();
        let final_state = agent.run(
            thread_id,
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        mp.force_flush().context("force_flush meter provider")?;

        let snapshots = collected(&mexporter)?;
        let points = collect_histogram_attrs(&snapshots, "gen_ai.client.token.usage");
        assert!(
            !points.is_empty(),
            "gen_ai.client.token.usage produced no data points"
        );
        assert!(
            points.iter().any(|p| matches_all(
                p,
                &[
                    ("gen_ai.operation.name", "chat"),
                    ("gen_ai.provider.name", "anthropic"),
                    ("gen_ai.token.type", "input"),
                    ("gen_ai.request.model", "test-model"),
                ],
            )),
            "missing input-token data point: {points:?}"
        );
        assert!(
            points.iter().any(|p| matches_all(
                p,
                &[
                    ("gen_ai.token.type", "output"),
                    ("gen_ai.operation.name", "chat")
                ],
            )),
            "missing output-token data point: {points:?}"
        );
        // `cache_read` / `cache_creation` are only recorded when
        // > 0 — `text_response` has both at zero, so they must NOT
        // appear. The "never as a single combined value" rule from
        // the card boils down to these per-type separations.
        assert!(
            !points
                .iter()
                .any(|p| has_label(p, "gen_ai.token.type", "cache_read")),
            "unexpected cache_read data point: {points:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn metrics_records_operation_duration_with_error_type_on_failure() -> Result<()> {
        let _guard = acquire_test_lock().await;
        let (_tp, _exporter) = setup_tracer();
        let (mp, mexporter) = setup_meter();

        // Force the rate-limit error path. `call_llm_with_retry`
        // exhausts the retry budget and surfaces a "Rate limited
        // after N retries" `AgentError`, which `classify_llm_error`
        // maps to `error.type=rate_limited`.
        let provider = TestProvider::new(vec![
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
            ChatOutcome::RateLimited,
        ]);
        let agent = builder::<()>()
            .provider(provider)
            .event_store(new_event_store())
            .config(agent_sdk::AgentConfig {
                retry: agent_sdk::RetryConfig {
                    max_retries: 1,
                    base_delay_ms: 1,
                    max_delay_ms: 1,
                },
                ..Default::default()
            })
            .build();
        let thread_id = ThreadId::new();
        let final_state = agent.run(
            thread_id,
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        mp.force_flush().context("force_flush meter provider")?;

        let snapshots = collected(&mexporter)?;
        let points = collect_histogram_attrs(&snapshots, "gen_ai.client.operation.duration");
        assert!(
            points.iter().any(|p| matches_all(
                p,
                &[
                    ("gen_ai.operation.name", "chat"),
                    ("gen_ai.provider.name", "anthropic"),
                    ("error.type", "rate_limited"),
                ],
            )),
            "expected operation.duration data point with error.type=rate_limited; got {points:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn metrics_records_tool_execution_with_outcome_success() -> Result<()> {
        let _guard = acquire_test_lock().await;
        let (_tp, _exporter) = setup_tracer();
        let (mp, mexporter) = setup_meter();

        let provider = TestProvider::new(vec![
            TestProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
            TestProvider::text_response("Final"),
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);
        let agent = builder::<()>()
            .provider(provider)
            .tools(tools)
            .hooks(AllowAllHooks)
            .message_store(InMemoryStore::new())
            .state_store(InMemoryStore::new())
            .event_store(new_event_store())
            .build_with_stores();
        let thread_id = ThreadId::new();
        let final_state = agent.run(
            thread_id,
            AgentInput::Text("Use echo".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        mp.force_flush().context("force_flush meter provider")?;

        let snapshots = collected(&mexporter)?;
        let count_points = collect_counter_attrs(&snapshots, "agent_sdk.tools.execution.count");
        assert!(
            count_points.iter().any(|p| matches_all(
                p,
                &[
                    ("gen_ai.tool.name", "echo"),
                    ("agent_sdk.tool.kind", "sync"),
                    ("agent_sdk.tool.outcome", "success"),
                ],
            )),
            "missing tools.execution.count success record: {count_points:?}"
        );
        let dur_points = collect_histogram_attrs(&snapshots, "agent_sdk.tools.execution.duration");
        assert!(
            dur_points.iter().any(|p| matches_all(
                p,
                &[
                    ("gen_ai.tool.name", "echo"),
                    ("agent_sdk.tool.outcome", "success"),
                ],
            )),
            "missing tools.execution.duration success record: {dur_points:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn metrics_records_run_outcome_counter() -> Result<()> {
        let _guard = acquire_test_lock().await;
        let (_tp, _exporter) = setup_tracer();
        let (mp, mexporter) = setup_meter();

        let provider = TestProvider::new(vec![TestProvider::text_response("Done")]);
        let agent = builder::<()>()
            .provider(provider)
            .event_store(new_event_store())
            .build();
        let thread_id = ThreadId::new();
        let final_state = agent.run(
            thread_id,
            AgentInput::Text("Hi".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        mp.force_flush().context("force_flush meter provider")?;

        let snapshots = collected(&mexporter)?;
        let outcome_points = collect_counter_attrs(&snapshots, "agent_sdk.runs.outcome");
        assert!(
            outcome_points
                .iter()
                .any(|p| matches_all(p, &[("agent_sdk.outcome", "done")],)),
            "missing runs.outcome=done record: {outcome_points:?}"
        );
        let turns_points = collect_histogram_attrs(&snapshots, "agent_sdk.turns.duration");
        assert!(
            turns_points.iter().any(|p| matches_all(
                p,
                &[
                    ("agent_sdk.outcome", "done"),
                    ("agent_sdk.input.kind", "text"),
                    ("gen_ai.provider.name", "anthropic"),
                ],
            )),
            "missing turns.duration record: {turns_points:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn metrics_records_context_compaction() -> Result<()> {
        let _guard = acquire_test_lock().await;
        let (_tp, _exporter) = setup_tracer();
        let (mp, mexporter) = setup_meter();

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
            .event_store(new_event_store())
            .with_compaction(
                agent_sdk::context::CompactionConfig::new()
                    .with_threshold_tokens(1)
                    .with_min_messages(1)
                    .with_retain_recent(1),
            )
            .build_with_stores();
        let final_state = agent.run(
            thread_id,
            AgentInput::Text("Follow up".to_string()),
            ToolContext::new(()),
            CancellationToken::new(),
        );
        wait_for_run(final_state).await?;
        mp.force_flush().context("force_flush meter provider")?;

        let snapshots = collected(&mexporter)?;
        let compaction_points = collect_counter_attrs(&snapshots, "agent_sdk.context.compaction");
        assert!(
            compaction_points
                .iter()
                .any(|p| matches_all(p, &[("agent_sdk.compaction.trigger", "threshold")],)),
            "missing context.compaction trigger=threshold record: {compaction_points:?}"
        );
        let saved_points =
            collect_histogram_attrs(&snapshots, "agent_sdk.context.compaction.tokens_saved");
        assert!(
            !saved_points.is_empty(),
            "expected at least one tokens_saved data point"
        );
        Ok(())
    }
}

// ── Span links (A7) ─────────────────────────────────────────────────

use agent_sdk::EventStore;
use agent_sdk::observability::spans;
use agent_sdk::subagent::{SubagentConfig, SubagentTool};
use opentelemetry::trace::{Span, SpanId as OtelSpanId, TraceContextExt, TraceId as OtelTraceId};

/// `Clone`-able LLM provider used by the subagent link test.
///
/// The subagent tool requires `LlmProvider: Clone` because it
/// instantiates a fresh `AgentLoop` per invocation; the main
/// `TestProvider` in this file deliberately is not `Clone` so it can
/// drain its scripted responses across calls. This wrapper hosts a
/// minimal scripted provider for the link test alone.
#[derive(Clone)]
struct CloneableTestProvider {
    responses: Arc<Mutex<Vec<ChatOutcome>>>,
}

impl CloneableTestProvider {
    fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl LlmProvider for CloneableTestProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            Ok(ChatOutcome::Success(ChatResponse {
                id: "resp_default".to_string(),
                content: vec![ContentBlock::Text {
                    text: "default".to_string(),
                }],
                model: "test-model".to_string(),
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
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

#[tokio::test]
async fn link_to_replay_origin_attaches_attributes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    // Reference values used by the OTel docs in
    // https://www.w3.org/TR/trace-context/#examples-of-http-traceparent-headers.
    let original_trace = "4bf92f3577b34da6a3ce929d0e0e4736";
    let original_span = "00f067aa0ba902b7";

    let mut span = spans::start_internal_span("agent.replay_test", vec![]);
    spans::link_to_replay_origin(&mut span, original_trace, original_span, 2);
    span.end();
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let target = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent.replay_test")
        .context("missing agent.replay_test span")?;
    assert_eq!(
        target.links.links.len(),
        1,
        "expected exactly one replay link",
    );
    let link = target
        .links
        .links
        .first()
        .context("missing first replay link")?;
    let expected_trace_id =
        OtelTraceId::from_hex(original_trace).context("parse original trace id hex")?;
    let expected_span_id =
        OtelSpanId::from_hex(original_span).context("parse original span id hex")?;
    assert_eq!(link.span_context.trace_id(), expected_trace_id);
    assert_eq!(link.span_context.span_id(), expected_span_id);

    let attrs: std::collections::HashMap<String, String> = link
        .attributes
        .iter()
        .map(|kv| (kv.key.to_string(), format!("{}", kv.value)))
        .collect();
    assert_eq!(
        attrs
            .get(attrs::AGENT_REPLAY_ORIGINAL_TRACE_ID)
            .map(String::as_str),
        Some(original_trace),
    );
    assert_eq!(
        attrs
            .get(attrs::AGENT_REPLAY_ORIGINAL_SPAN_ID)
            .map(String::as_str),
        Some(original_span),
    );
    assert_eq!(
        attrs
            .get(attrs::AGENT_REPLAY_ATTEMPT_INDEX)
            .map(String::as_str),
        Some("2"),
    );

    Ok(())
}

#[tokio::test]
async fn link_to_replay_origin_drops_malformed_ids() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    let mut span = spans::start_internal_span("agent.replay_malformed", vec![]);
    // "ZZZZ" is not valid hex. The helper must silently drop the link
    // rather than panic.
    spans::link_to_replay_origin(&mut span, "ZZZZ", "ZZZZ", 1);
    span.end();
    tp.force_flush()
        .context("failed to flush tracer provider")?;

    let spans = get_spans(&exporter)?;
    let target = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent.replay_malformed")
        .context("missing agent.replay_malformed span")?;
    assert!(
        target.links.links.is_empty(),
        "expected no links on malformed input, got {} links",
        target.links.links.len(),
    );

    Ok(())
}

#[tokio::test]
async fn subagent_invoke_agent_links_to_parent_turn() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (tp, exporter) = setup_tracer();

    // 1. Build a parent span manually; the subagent emit captures
    //    `Context::current().span().span_context()` so we just need a
    //    real span to be active when the subagent's emit hook runs.
    let parent = spans::start_internal_span("invoke_agent_parent_test", vec![]);
    let parent_trace_id = parent.span_context().trace_id();
    let parent_span_id = parent.span_context().span_id();
    let cx = OtelContext::current_with_span(parent);

    // 2. Run the subagent under the parent context. The subagent
    //    internally calls `Context::current().span().span_context()`
    //    to capture the parent.
    let provider = Arc::new(CloneableTestProvider::new(vec![ChatOutcome::Success(
        ChatResponse {
            id: "resp_subagent".to_string(),
            content: vec![ContentBlock::Text {
                text: "Subagent done".to_string(),
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 5,
                output_tokens: 10,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        },
    )]));
    let tools = Arc::new(ToolRegistry::<()>::new());
    let event_store_factory = || -> Arc<dyn EventStore> { Arc::new(InMemoryEventStore::new()) };
    let subagent = SubagentTool::new(
        SubagentConfig::new("worker"),
        provider,
        tools,
        event_store_factory,
    );

    let parent_ctx_for_run = ToolContext::new(());
    let attached = cx.attach();
    let result = <SubagentTool<CloneableTestProvider> as Tool<()>>::execute(
        &subagent,
        &parent_ctx_for_run,
        json!({ "task": "Inspect the repo" }),
    )
    .await?;
    drop(attached);
    assert!(result.success, "subagent should succeed");

    tp.force_flush()
        .context("failed to flush tracer provider")?;

    // 3. Find the synthetic subagent invoke_agent span.
    let spans = get_spans(&exporter)?;
    let subagent_span = spans
        .iter()
        .find(|span| {
            span.name.as_ref() == "invoke_agent"
                && get_attr(span, attrs::GEN_AI_AGENT_NAME).as_deref() == Some("worker")
        })
        .with_context(|| "missing subagent invoke_agent span".to_string())?;

    assert_eq!(
        subagent_span.links.links.len(),
        1,
        "expected exactly one parent-turn link on subagent span",
    );
    let link = subagent_span
        .links
        .links
        .first()
        .with_context(|| "missing parent-turn link".to_string())?;
    assert_eq!(link.span_context.trace_id(), parent_trace_id);
    assert_eq!(link.span_context.span_id(), parent_span_id);
    // `link_to_parent_turn` doesn't carry attributes — the relationship
    // is implicit in the linked SpanContext.
    assert!(link.attributes.is_empty());

    Ok(())
}
