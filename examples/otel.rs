//! OpenTelemetry configuration example.
//!
//! Run with:
//! ```bash
//! cargo run --example otel --features otel
//! ```

use agent_sdk::llm::{ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, Usage};
use agent_sdk::observability::{CaptureDecision, CaptureResult, ObservabilityStore, PayloadBundle};
use agent_sdk::{AgentInput, CancellationToken, LlmProvider, ThreadId, ToolContext, builder};
use anyhow::{Context, Result};
use async_trait::async_trait;
use opentelemetry::global;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

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
            },
        }))
    }

    fn model(&self) -> &str {
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
    let exporter = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    global::set_tracer_provider(tracer_provider.clone());

    let agent = builder::<()>()
        .provider(DemoProvider)
        .observability_store(InlinePayloadStore)
        .build();

    let (mut events, final_state) = agent.run(
        ThreadId::new(),
        AgentInput::Text("Say hello in one sentence.".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );

    while events.recv().await.is_some() {}

    let state = final_state.await.context("agent state channel closed")?;
    tracer_provider
        .force_flush()
        .context("failed to flush tracer provider")?;
    let spans = exporter
        .get_finished_spans()
        .context("failed to read finished spans")?;

    println!("Final state: {state:?}");
    println!("Exported {} spans:", spans.len());
    for span in spans {
        println!("- {}", span.name);
    }

    Ok(())
}
