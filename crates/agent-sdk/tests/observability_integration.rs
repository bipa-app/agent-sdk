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
