//! Phase 9 · F1 — Conformance + privacy integration tests.
//!
//! Single-file, top-down assertion of the Track A/B/C contract:
//!
//! 1. **Span shape** — every SDK boundary emits the documented span
//!    name + attributes + Langfuse `observation.type`.
//! 2. **Baggage** — parent context with the five baseline keys
//!    produces every nested span carrying them as attributes.
//! 3. **`RunOptions` landing** — caller-supplied trace metadata
//!    appears on the root span exactly where the docs say.
//! 4. **Metrics** — every metric defined in B1 + B3 records at least
//!    one labelled sample under the right scenario.
//! 5. **Privacy (C1/C2)** — the redactor masks PII before payloads
//!    leave the observability boundary, and the operator-level
//!    capture gate (C2) holds in both directions.
//! 6. **Replay link (A7)** — the `spans::link_to_replay_origin`
//!    helper attaches a `SpanLink` whose `SpanContext` matches the
//!    supplied trace/span ids and carries the documented attributes.
//! 7. **Outbound baggage filter (C3)** — `install_global_provider`'s
//!    composite propagator drops every non-allow-listed entry on
//!    outbound serialisation while preserving inbound entries.
//!
//! All tests run against in-memory exporters — no live `OTel`
//! backend, no Postgres, no AMQP.

// The conformance battery asserts the full SDK boundary contract, including
// MCP request span/metric emission, so it requires the `mcp` tool feature in
// addition to `otel`. Both are enabled under `--all-features` in CI.
#![cfg(all(feature = "otel", feature = "mcp"))]

use agent_sdk::context::CompactionConfig;
use agent_sdk::llm::Message;
use agent_sdk::llm::Usage;
use agent_sdk::mcp::McpClient;
use agent_sdk::observability::baggage as obs_baggage;
use agent_sdk::observability::payload::PayloadRedactor;
use agent_sdk::observability::types::CaptureDecision;
use agent_sdk::observability::types::CaptureResult;
use agent_sdk::observability::types::ObservabilityStore;
use agent_sdk::observability::types::PayloadBundle;
use agent_sdk::observability::{attrs, langfuse, spans};
use agent_sdk::subagent::{SubagentConfig, SubagentTool};
use agent_sdk::{
    AgentConfig, AgentInput, AllowAllHooks, CancellationToken, EventStore, InMemoryEventStore,
    InMemoryStore, MessageStore, RetryConfig, RunOptions, ThreadId, Tool, ToolContext,
    ToolRegistry, builder,
};
use agent_sdk_core::privacy::BaselineDetector;
use agent_sdk_otel::{OtelConfig, SamplerKind, install_global_provider};
use anyhow::{Context, Result};
use async_trait::async_trait;
use opentelemetry::Context as OtelContext;
use opentelemetry::KeyValue;
use opentelemetry::baggage::BaggageExt;
use opentelemetry::global;
use opentelemetry::trace::{Span, SpanId, TraceId};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

#[path = "support/harness.rs"]
mod harness;
#[path = "support/stub_provider.rs"]
mod stub_provider;

use harness::{
    CaptureGateGuard, acquire_test_lock, assert_metric_counter_sample,
    assert_metric_histogram_sample, assert_span_attribute, assert_span_attribute_absent,
    assert_span_attribute_present, find_span_in_trace, get_attr, root_span_for_thread,
    setup_in_memory_provider, spans_in_trace, wait_for_run,
};
use stub_provider::{CloneableStubProvider, EchoTool, ScriptedMcpTransport, StubProvider};

// ── Shared fixtures ──────────────────────────────────────────────────

fn new_event_store() -> Arc<InMemoryEventStore> {
    Arc::new(InMemoryEventStore::new())
}

async fn run_tool_use_flow(harness: &harness::InMemoryHarness) -> Result<ThreadId> {
    let provider = StubProvider::new(vec![
        StubProvider::tool_use_response("call_1", "echo", json!({"text": "hello"})),
        StubProvider::text_response("Final answer"),
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
        thread_id.clone(),
        AgentInput::Text("Test the echo".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    harness.force_flush_all()?;
    Ok(thread_id)
}

// ── 1. Span-tree shape ───────────────────────────────────────────────

#[tokio::test]
async fn test_run_emits_full_span_tree() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    // Seed history so the compaction span has something to summarise.
    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    seed_compaction_history(&store, &thread_id).await?;

    let provider = StubProvider::new(vec![
        StubProvider::text_response("Summary"),
        StubProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        StubProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(new_event_store())
        .with_compaction(
            CompactionConfig::new()
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
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    let invoke = find_span_in_trace(&trace_spans, "invoke_agent")?;
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;
    let compaction = find_span_in_trace(&trace_spans, "agent.context_compaction")?;

    // Every child belongs to the same trace.
    assert_eq!(turn.span_context.trace_id(), invoke.span_context.trace_id());
    assert_eq!(chat.span_context.trace_id(), invoke.span_context.trace_id());
    assert_eq!(tool.span_context.trace_id(), invoke.span_context.trace_id());
    assert_eq!(
        compaction.span_context.trace_id(),
        invoke.span_context.trace_id(),
    );

    // Compaction is a direct child of the root.
    assert_eq!(compaction.parent_span_id, invoke.span_context.span_id());

    Ok(())
}

// ── 2. Attribute completeness ────────────────────────────────────────

#[tokio::test]
async fn test_attribute_completeness() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();
    let thread_id = run_tool_use_flow(&harness).await?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let turn = find_span_in_trace(&trace_spans, "agent.turn")?;
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;
    let tool = find_span_in_trace(&trace_spans, "execute_tool")?;

    // Root span — every required attribute is present and non-empty.
    for key in [
        attrs::GEN_AI_OPERATION_NAME,
        attrs::GEN_AI_PROVIDER_NAME,
        attrs::GEN_AI_CONVERSATION_ID,
        attrs::SDK_PROVIDER_ID,
        attrs::SDK_RUN_MODE,
        attrs::SDK_INPUT_KIND,
        attrs::SDK_OUTCOME,
    ] {
        assert_span_attribute_present(root, key);
    }

    // Turn span — number + cumulative token usage.
    assert_span_attribute_present(turn, attrs::SDK_TURN_NUMBER);
    assert_span_attribute_present(turn, attrs::SDK_TURN_INPUT_TOKENS);
    assert_span_attribute_present(turn, attrs::SDK_TURN_OUTPUT_TOKENS);

    // Chat span — GenAI semconv.
    for key in [
        attrs::GEN_AI_OPERATION_NAME,
        attrs::GEN_AI_PROVIDER_NAME,
        attrs::GEN_AI_REQUEST_MODEL,
        attrs::GEN_AI_RESPONSE_MODEL,
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
    ] {
        assert_span_attribute_present(chat, key);
    }

    // Tool span — name + outcome + tier + kind.
    for key in [
        attrs::GEN_AI_TOOL_NAME,
        attrs::GEN_AI_TOOL_CALL_ID,
        attrs::SDK_TOOL_OUTCOME,
        attrs::SDK_TOOL_TIER,
        attrs::SDK_TOOL_KIND,
    ] {
        assert_span_attribute_present(tool, key);
    }

    Ok(())
}

// ── 3. Langfuse observation-type tagging ─────────────────────────────

#[tokio::test]
async fn test_observation_type_tagging() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    let store = SharedStore::new();
    let thread_id = ThreadId::new();
    seed_compaction_history(&store, &thread_id).await?;

    let provider = StubProvider::new(vec![
        StubProvider::text_response("Summary"),
        StubProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        StubProvider::text_response("Done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool);

    let agent = builder::<()>()
        .provider(provider)
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(store.clone())
        .state_store(store.clone())
        .event_store(new_event_store())
        .with_compaction(
            CompactionConfig::new()
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
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    assert_span_attribute(root, langfuse::LANGFUSE_OBSERVATION_TYPE, "agent");
    assert_span_attribute(
        find_span_in_trace(&trace_spans, "chat test-model")?,
        langfuse::LANGFUSE_OBSERVATION_TYPE,
        "generation",
    );
    assert_span_attribute(
        find_span_in_trace(&trace_spans, "execute_tool")?,
        langfuse::LANGFUSE_OBSERVATION_TYPE,
        "tool",
    );
    assert_span_attribute(
        find_span_in_trace(&trace_spans, "agent.context_compaction")?,
        langfuse::LANGFUSE_OBSERVATION_TYPE,
        "chain",
    );

    Ok(())
}

// ── 4. Baggage propagation ───────────────────────────────────────────

const ALL_BAGGAGE: &[(&str, &str)] = &[
    (obs_baggage::BAGGAGE_USER_ID, "user-42"),
    (obs_baggage::BAGGAGE_SESSION_ID, "session-7"),
    (obs_baggage::BAGGAGE_LANGFUSE_USER_ID, "lf-user-42"),
    (obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID, "lf-session-7"),
    (obs_baggage::BAGGAGE_DEPLOYMENT_ENVIRONMENT, "test"),
];

fn baggage_context(entries: &[(&'static str, &'static str)]) -> OtelContext {
    let kvs: Vec<KeyValue> = entries.iter().map(|(k, v)| KeyValue::new(*k, *v)).collect();
    OtelContext::current_with_baggage(kvs)
}

async fn run_with_baggage<F, T>(cx: OtelContext, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    use opentelemetry::trace::FutureExt;
    fut.with_context(cx).await
}

#[tokio::test]
async fn test_baggage_propagation() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    let provider = StubProvider::new(vec![
        StubProvider::tool_use_response("call_1", "echo", json!({"text": "hi"})),
        StubProvider::text_response("Done"),
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

    harness.force_flush_all()?;
    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());

    for name in [
        "invoke_agent",
        "agent.turn",
        "chat test-model",
        "execute_tool",
    ] {
        let span = find_span_in_trace(&trace_spans, name)?;
        for (key, value) in ALL_BAGGAGE {
            assert_span_attribute(span, key, value);
        }
    }

    Ok(())
}

// ── 5. RunOptions landing ────────────────────────────────────────────

#[tokio::test]
async fn test_run_options_landing() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    let provider = StubProvider::new(vec![StubProvider::text_response("Hi back")]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .build();

    let mut metadata = serde_json::Map::new();
    metadata.insert("release_channel".to_string(), json!("beta"));
    metadata.insert("user_count".to_string(), json!(42));
    let opts = RunOptions {
        session_id: Some("session-F1".to_string()),
        user_id: Some("user-F1".to_string()),
        trace_name: Some("f1.test.run".to_string()),
        trace_tags: vec!["mobile.android".to_string(), "experiment.f1".to_string()],
        trace_metadata: metadata,
        release: Some("9.0.0".to_string()),
        environment: Some("staging".to_string()),
        trace_text_max_chars: None,
    };

    let thread_id = ThreadId::new();
    let final_state = agent.run_with_options(
        thread_id.clone(),
        AgentInput::Text("Hello F1".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        opts,
    );
    wait_for_run(final_state).await?;
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;

    assert_span_attribute(root, langfuse::LANGFUSE_TRACE_NAME, "f1.test.run");
    // Tags land as a string-array; the stringified rendering depends on
    // the OTel SDK so we assert on content, not exact format.
    let tags = get_attr(root, langfuse::LANGFUSE_TRACE_TAGS)
        .context("missing langfuse.trace.tags on root span")?;
    assert!(
        tags.contains("mobile.android") && tags.contains("experiment.f1"),
        "trace.tags missing entries: {tags}",
    );
    assert_span_attribute(
        root,
        &format!(
            "{}{}",
            langfuse::LANGFUSE_TRACE_METADATA_PREFIX,
            "release_channel"
        ),
        "beta",
    );
    assert_span_attribute(
        root,
        &format!(
            "{}{}",
            langfuse::LANGFUSE_TRACE_METADATA_PREFIX,
            "user_count"
        ),
        "42",
    );
    assert_span_attribute(root, langfuse::LANGFUSE_RELEASE, "9.0.0");
    assert_span_attribute(root, langfuse::LANGFUSE_ENVIRONMENT, "staging");
    assert_span_attribute(root, obs_baggage::BAGGAGE_SESSION_ID, "session-F1");
    assert_span_attribute(root, obs_baggage::BAGGAGE_LANGFUSE_SESSION_ID, "session-F1");
    assert_span_attribute(root, obs_baggage::BAGGAGE_USER_ID, "user-F1");
    assert_span_attribute(root, obs_baggage::BAGGAGE_LANGFUSE_USER_ID, "user-F1");
    assert_span_attribute(root, langfuse::LANGFUSE_TRACE_INPUT, "Hello F1");

    Ok(())
}

// ── 6. Metrics conformance (B1: client + run-level) ──────────────────

#[tokio::test]
async fn test_metrics_emitted_b1() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    let provider = StubProvider::new(vec![StubProvider::text_response("Done")]);
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
    harness.force_flush_all()?;

    let snapshots = harness.metrics()?;

    assert_metric_histogram_sample(
        &snapshots,
        "gen_ai.client.token.usage",
        &[
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.provider.name", "anthropic"),
            ("gen_ai.token.type", "input"),
            ("gen_ai.request.model", "test-model"),
        ],
    );
    assert_metric_histogram_sample(
        &snapshots,
        "gen_ai.client.token.usage",
        &[
            ("gen_ai.token.type", "output"),
            ("gen_ai.operation.name", "chat"),
        ],
    );
    assert_metric_histogram_sample(
        &snapshots,
        "gen_ai.client.operation.duration",
        &[
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.provider.name", "anthropic"),
        ],
    );
    assert_metric_counter_sample(
        &snapshots,
        "agent_sdk.runs.outcome",
        &[("agent_sdk.outcome", "done")],
    );
    assert_metric_histogram_sample(
        &snapshots,
        "agent_sdk.turns.duration",
        &[
            ("agent_sdk.outcome", "done"),
            ("agent_sdk.input.kind", "text"),
            ("gen_ai.provider.name", "anthropic"),
        ],
    );

    Ok(())
}

// ── 7. Metrics conformance (B3: subagent + MCP + retries) ────────────

#[tokio::test]
async fn test_metrics_emitted_b3() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    // 7a — subagent invocation records counter + parent-context token usage.
    let subagent_provider = Arc::new(CloneableStubProvider::new(vec![
        agent_sdk::llm::ChatOutcome::Success(agent_sdk::llm::ChatResponse {
            id: "resp_sub".to_string(),
            content: vec![agent_sdk::llm::ContentBlock::Text {
                text: "Subagent done".to_string(),
            }],
            model: "test-model".to_string(),
            stop_reason: Some(agent_sdk::llm::StopReason::EndTurn),
            usage: Usage {
                input_tokens: 5,
                output_tokens: 10,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }),
    ]));
    let subagent_tools = Arc::new(ToolRegistry::<()>::new());
    let subagent_factory = || -> Arc<dyn EventStore> { Arc::new(InMemoryEventStore::new()) };
    let subagent = SubagentTool::new(
        SubagentConfig::new("worker"),
        subagent_provider,
        subagent_tools,
        subagent_factory,
    );
    let parent_ctx = ToolContext::new(());
    let result = <SubagentTool<CloneableStubProvider> as Tool<()>>::execute(
        &subagent,
        &parent_ctx,
        json!({ "task": "Inspect" }),
    )
    .await?;
    assert!(result.success, "subagent should succeed");

    // 7b — MCP call duration records on tools/list + tools/call.
    let transport = Arc::new(ScriptedMcpTransport::new(vec![
        ScriptedMcpTransport::tools_call_text("ok"),
        ScriptedMcpTransport::list_tools_empty(),
    ]));
    let mcp_client = McpClient::new_uninitialized(transport, "f1-mcp".to_string());
    let _ = mcp_client.list_tools().await?;
    let _ = mcp_client.call_tool("noop", json!({})).await?;

    // 7c — LLM retries record on the rate-limit retry path.
    let provider = StubProvider::new(vec![
        agent_sdk::llm::ChatOutcome::RateLimited,
        StubProvider::text_response("Done"),
    ]);
    let agent = builder::<()>()
        .provider(provider)
        .event_store(new_event_store())
        .config(AgentConfig {
            retry: RetryConfig {
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
    harness.force_flush_all()?;

    let snapshots = harness.metrics()?;
    assert_metric_counter_sample(
        &snapshots,
        "agent_sdk.subagent.invocations",
        &[
            ("gen_ai.agent.name", "worker"),
            ("agent_sdk.outcome", "done"),
        ],
    );
    assert_metric_histogram_sample(
        &snapshots,
        "agent_sdk.mcp.requests.duration",
        &[("mcp.method", "tools/list"), ("mcp.server.name", "f1-mcp")],
    );
    assert_metric_histogram_sample(
        &snapshots,
        "agent_sdk.mcp.requests.duration",
        &[("mcp.method", "tools/call"), ("mcp.server.name", "f1-mcp")],
    );
    assert_metric_counter_sample(
        &snapshots,
        "agent_sdk.llm.retries",
        &[
            ("gen_ai.provider.name", "anthropic"),
            ("error.type", "rate_limited"),
        ],
    );

    Ok(())
}

// ── 8. PII redactor coverage ─────────────────────────────────────────

/// `Inline` for every artifact, attesting PII safety AND carrying a
/// `BaselineDetector`-backed redactor. This is the only safe shape for
/// inline capture in production.
struct AttestingBaselineStore {
    redactor: PayloadRedactor,
}

impl AttestingBaselineStore {
    fn new() -> Result<Self> {
        Ok(Self {
            redactor: PayloadRedactor::new(Arc::new(
                BaselineDetector::new().context("baseline detector")?,
            )),
        })
    }
}

#[async_trait]
impl ObservabilityStore for AttestingBaselineStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> Result<CaptureResult> {
        Ok(CaptureResult {
            system_instructions: CaptureDecision::Inline,
            input_messages: CaptureDecision::Inline,
            output_messages: CaptureDecision::Inline,
        })
    }

    fn redactor(&self) -> &PayloadRedactor {
        &self.redactor
    }

    fn acknowledge_pii_redaction(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn test_no_pii_in_spans() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let _gate = CaptureGateGuard::set(true);
    let harness = setup_in_memory_provider();

    // Assistant text echoes a CPF + email so output-message redaction
    // is exercised on the same span as input redaction.
    let provider = StubProvider::new(vec![StubProvider::text_response(
        "Customer CPF 111.444.777-35 and email ana@bipa.exchange were processed.",
    )]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(AttestingBaselineStore::new()?)
        .event_store(new_event_store())
        .build();
    let thread_id = ThreadId::new();
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text(
            "My CPF is 111.444.777-35 and email ana@bipa.exchange — \
             API key sk-abcdefghijklmnopqrstuv"
                .to_string(),
        ),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    wait_for_run(final_state).await?;
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;

    let input_messages = get_attr(chat, attrs::GEN_AI_INPUT_MESSAGES)
        .context("input messages must land when both prongs of C2 are satisfied")?;
    let output_messages = get_attr(chat, attrs::GEN_AI_OUTPUT_MESSAGES)
        .context("output messages must land when both prongs of C2 are satisfied")?;

    // Raw PII never reaches the span.
    for raw in [
        "111.444.777-35",
        "ana@bipa.exchange",
        "sk-abcdefghijklmnopqrstuv",
    ] {
        assert!(
            !input_messages.contains(raw),
            "raw {raw} leaked to gen_ai.input.messages: {input_messages}",
        );
        assert!(
            !output_messages.contains(raw),
            "raw {raw} leaked to gen_ai.output.messages: {output_messages}",
        );
    }
    // Masking markers are present.
    assert!(input_messages.contains("[REDACTED:cpf]"));
    assert!(input_messages.contains("[REDACTED:email]"));
    assert!(input_messages.contains("[REDACTED:secret]"));
    assert!(output_messages.contains("[REDACTED:cpf]"));
    assert!(output_messages.contains("[REDACTED:email]"));

    Ok(())
}

// ── 9. Capture gate — default deny ───────────────────────────────────

struct InlineNoAttestationStore;

#[async_trait]
impl ObservabilityStore for InlineNoAttestationStore {
    async fn capture(&self, _bundle: &PayloadBundle) -> Result<CaptureResult> {
        Ok(CaptureResult {
            system_instructions: CaptureDecision::Inline,
            input_messages: CaptureDecision::Inline,
            output_messages: CaptureDecision::Inline,
        })
    }
}

#[tokio::test]
async fn test_capture_payloads_default_deny() -> Result<()> {
    let _guard = acquire_test_lock().await;
    // Gate is open at the operator level …
    let _gate = CaptureGateGuard::set(true);
    let harness = setup_in_memory_provider();

    // … but the store has not attested PII redaction, so the gate
    // must still downgrade every `Inline` to `Omit`.
    let provider = StubProvider::new(vec![StubProvider::text_response("Hi")]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(InlineNoAttestationStore)
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
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;
    assert_span_attribute_absent(chat, attrs::GEN_AI_INPUT_MESSAGES);
    assert_span_attribute_absent(chat, attrs::GEN_AI_OUTPUT_MESSAGES);

    Ok(())
}

// ── 10. Capture gate — opt-in path ───────────────────────────────────

#[tokio::test]
async fn test_capture_payloads_with_attestation() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let _gate = CaptureGateGuard::set(true);
    let harness = setup_in_memory_provider();

    let provider = StubProvider::new(vec![StubProvider::text_response("Hi back")]);
    let agent = builder::<()>()
        .provider(provider)
        .observability_store(AttestingBaselineStore::new()?)
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
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let root = root_span_for_thread(&spans, &thread_id)?;
    let trace_spans = spans_in_trace(&spans, root.span_context.trace_id());
    let chat = find_span_in_trace(&trace_spans, "chat test-model")?;

    let input = get_attr(chat, attrs::GEN_AI_INPUT_MESSAGES)
        .context("input messages must land when both prongs are satisfied")?;
    let output = get_attr(chat, attrs::GEN_AI_OUTPUT_MESSAGES)
        .context("output messages must land when both prongs are satisfied")?;
    assert!(input.contains("Hello"), "input: {input}");
    assert!(output.contains("Hi back"), "output: {output}");

    Ok(())
}

// ── 11. Replay link contract ─────────────────────────────────────────

#[tokio::test]
async fn test_replay_link() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let harness = setup_in_memory_provider();

    // W3C trace-context spec example values — guaranteed valid hex.
    let original_trace = "4bf92f3577b34da6a3ce929d0e0e4736";
    let original_span = "00f067aa0ba902b7";

    let mut span = spans::start_internal_span("agent.replay_conformance", vec![]);
    spans::link_to_replay_origin(&mut span, original_trace, original_span, 3);
    span.end();
    harness.force_flush_all()?;

    let spans = harness.spans()?;
    let target = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent.replay_conformance")
        .context("missing agent.replay_conformance span")?;

    assert_eq!(target.links.links.len(), 1);
    let link = target.links.links.first().context("missing replay link")?;
    let expected_trace_id = TraceId::from_hex(original_trace).context("parse trace id hex")?;
    let expected_span_id = SpanId::from_hex(original_span).context("parse span id hex")?;
    assert_eq!(link.span_context.trace_id(), expected_trace_id);
    assert_eq!(link.span_context.span_id(), expected_span_id);

    let link_attrs: HashMap<String, String> = link
        .attributes
        .iter()
        .map(|kv| (kv.key.to_string(), format!("{}", kv.value)))
        .collect();
    assert_eq!(
        link_attrs
            .get(attrs::AGENT_REPLAY_ORIGINAL_TRACE_ID)
            .map(String::as_str),
        Some(original_trace),
    );
    assert_eq!(
        link_attrs
            .get(attrs::AGENT_REPLAY_ORIGINAL_SPAN_ID)
            .map(String::as_str),
        Some(original_span),
    );
    assert_eq!(
        link_attrs
            .get(attrs::AGENT_REPLAY_ATTEMPT_INDEX)
            .map(String::as_str),
        Some("3"),
    );

    Ok(())
}

// ── 12. Outbound baggage allow-list (C3) ─────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_baggage_outbound_filter() -> Result<()> {
    let _guard = acquire_test_lock().await;

    let cfg = OtelConfig::builder("agent-sdk-f1-c3")
        .otlp_endpoint(None)
        .sampler(SamplerKind::AlwaysOff)
        .propagated_baggage_keys(Vec::new()) // empty → baseline allow-list
        .build();
    let guard = install_global_provider(&cfg)?;

    let cx = OtelContext::current_with_baggage([
        KeyValue::new("user.id", "alice"),
        KeyValue::new("session.id", "sess-1"),
        KeyValue::new("password", "hunter2"),
        KeyValue::new("api_key", "shhh"),
    ]);
    let mut headers: HashMap<String, String> = HashMap::new();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&cx, &mut headers);
    });

    let header = headers
        .get("baggage")
        .context("baggage header should carry user.id + session.id")?;
    assert!(header.contains("user.id=alice"), "got: {header}");
    assert!(header.contains("session.id=sess-1"), "got: {header}");
    assert!(!header.contains("password"), "got: {header}");
    assert!(!header.contains("hunter2"), "got: {header}");
    assert!(!header.contains("api_key"), "got: {header}");

    // Extract round-trip preserves whatever upstream hands us — the
    // allow-list only filters the egress.
    let mut inbound: HashMap<String, String> = HashMap::new();
    inbound.insert(
        "baggage".to_string(),
        "user.id=bob,password=lookatme".to_string(),
    );
    let extracted = global::get_text_map_propagator(|p| p.extract(&inbound));
    let baggage = extracted.baggage();
    assert_eq!(
        baggage.get("user.id").map(ToString::to_string),
        Some("bob".to_string()),
    );
    assert_eq!(
        baggage.get("password").map(ToString::to_string),
        Some("lookatme".to_string()),
    );

    guard.shutdown()?;
    Ok(())
}

// ── Shared compaction store ──────────────────────────────────────────

/// Cloneable wrapper around [`InMemoryStore`] used by the compaction
/// tests so the message and state stores can share state through
/// `Arc`. Mirrors the helper in `observability_integration.rs`.
#[derive(Clone, Default)]
struct SharedStore(Arc<InMemoryStore>);

impl SharedStore {
    fn new() -> Self {
        Self(Arc::new(InMemoryStore::new()))
    }
}

#[async_trait]
impl MessageStore for SharedStore {
    async fn append(&self, thread_id: &ThreadId, message: agent_sdk::llm::Message) -> Result<()> {
        self.0.append(thread_id, message).await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<agent_sdk::llm::Message>> {
        self.0.get_history(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.0.clear(thread_id).await
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<agent_sdk::llm::Message>,
    ) -> Result<()> {
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

/// Seed two-message history so the threshold compactor fires on the
/// first turn. `bulky_user` is intentionally large so the summariser
/// produces a token saving rather than tripping the B3 expanded-skip
/// guard.
async fn seed_compaction_history(store: &SharedStore, thread_id: &ThreadId) -> Result<()> {
    let bulky_user = "Investigate why the dashboard times out under load. ".repeat(40);
    store.append(thread_id, Message::user(bulky_user)).await?;
    store
        .append(thread_id, Message::assistant("Acknowledged."))
        .await?;
    Ok(())
}
