//! `OpenTelemetry` configuration example.
//!
//! Demonstrates the production-style bootstrap: a single
//! `agent_sdk_otel::install_global_provider` call wires up tracer + meter
//! providers and a W3C `TraceContext` + Baggage propagator, and the returned
//! `OtelGuard` flushes pending exports on shutdown.
//!
//! Run with:
//! ```bash
//! cargo run --example otel --features otel
//! ```
//!
//! Set `OTEL_EXPORTER_OTLP_ENDPOINT` to point at an `OTel` collector
//! (e.g. `http://localhost:4317`); leave it unset to run in no-op mode.

use agent_sdk::llm::{ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage};
use agent_sdk::observability::{CaptureDecision, CaptureResult, ObservabilityStore, PayloadBundle};
use agent_sdk::{
    AgentInput, CancellationToken, EventStore, InMemoryEventStore, LlmProvider, RunOptions,
    ThreadId, ToolContext, builder,
};
use agent_sdk_otel::{OtelConfig, install_global_provider};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;

struct DemoProvider;

#[async_trait]
impl LlmProvider for DemoProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(ChatResponse {
            id: "resp_demo".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello from the instrumented agent!".to_string(),
            }],
            model: "demo-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 8,
                output_tokens: 12,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "demo-model"
    }

    fn provider(&self) -> &'static str {
        "anthropic"
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

#[tokio::main]
async fn main() -> Result<()> {
    // Read OTel config from the environment. Override `service.name`
    // unless the operator has supplied `OTEL_SERVICE_NAME` themselves.
    let mut cfg = OtelConfig::from_env()?;
    if std::env::var_os("OTEL_SERVICE_NAME").is_none() {
        cfg.service_name = "agent-sdk-otel-example".to_string();
    }
    if cfg.service_version.is_none() {
        cfg.service_version = Some(env!("CARGO_PKG_VERSION").to_string());
    }
    let guard = install_global_provider(&cfg)?;

    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(DemoProvider)
        .observability_store(InlinePayloadStore)
        .event_store(event_store.clone())
        .build();

    let thread_id = ThreadId::new();

    // A5: pass per-run trace metadata through `RunOptions` so the
    // SDK populates `langfuse.trace.{name,session.id,user.id,
    // metadata.*}` and the running `langfuse.trace.{input,output}`
    // accumulator without any consumer-side glue. Leave any field
    // at its default (e.g. `RunOptions::default()`) when you don't
    // need that piece of metadata.
    let mut trace_metadata = serde_json::Map::new();
    trace_metadata.insert(
        "client.platform".to_string(),
        serde_json::Value::String("example".to_string()),
    );
    let run_options = RunOptions {
        session_id: Some(thread_id.to_string()),
        user_id: Some("demo-user".to_string()),
        trace_name: Some("agent-sdk.otel-example".to_string()),
        trace_tags: vec!["example".to_string(), "demo".to_string()],
        trace_metadata,
        release: Some(env!("CARGO_PKG_VERSION").to_string()),
        environment: Some("local".to_string()),
        trace_text_max_chars: None,
    };

    let final_state = agent.run_with_options(
        thread_id.clone(),
        AgentInput::Text("Say hello in one sentence.".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
        run_options,
    );
    let state = final_state.await.context("agent state channel closed")?;
    let event_count = event_store.get_events(&thread_id).await?.len();

    println!("Final state: {state:?}");
    println!("Persisted {event_count} events");
    println!(
        "OTel pipeline shutting down — set OTEL_EXPORTER_OTLP_ENDPOINT \
         to push spans/metrics to a collector; leave it unset for no-op mode."
    );

    guard.shutdown()?;
    Ok(())
}
