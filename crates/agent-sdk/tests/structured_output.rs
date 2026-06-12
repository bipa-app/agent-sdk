//! End-to-end structured-output runner tests through the public `agent-sdk`
//! facade.
//!
//! These exercise the bounded re-prompt / validate / typed-error path with a
//! deterministic scripted provider — one happy path per provider *strategy*
//! (native JSON mode vs the tool-forcing fallback), a mismatch→retry→success
//! flow, and retry exhaustion terminating in a typed error.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, LlmProvider, StopReason, StreamBox,
    StreamDelta, StreamErrorKind, Usage,
};
use agent_sdk::{ResponseFormat, StructuredConfig, StructuredOutputError, run_structured};
use agent_sdk_providers::StructuredOutputSupport;
use anyhow::Result;
use async_trait::async_trait;

const RESPOND_TOOL_NAME: &str = "respond";

struct ScriptedProvider {
    provider_name: &'static str,
    model: String,
    support: StructuredOutputSupport,
    outcomes: Mutex<std::collections::VecDeque<ChatOutcome>>,
    calls: AtomicUsize,
}

impl ScriptedProvider {
    fn new(
        provider_name: &'static str,
        support: StructuredOutputSupport,
        outcomes: Vec<ChatOutcome>,
    ) -> Self {
        Self {
            provider_name,
            model: "scripted-model".to_owned(),
            support,
            outcomes: Mutex::new(outcomes.into()),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .outcomes
            .lock()
            .expect("outcomes lock")
            .pop_front()
            .expect("scripted outcomes exhausted"))
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            yield Ok(StreamDelta::Error {
                message: "unused".to_owned(),
                kind: StreamErrorKind::ServerError,
            });
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        self.provider_name
    }

    fn structured_output_support(&self) -> StructuredOutputSupport {
        self.support
    }
}

fn ticket_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "priority": { "type": "string", "enum": ["low", "high"] }
        },
        "required": ["title", "priority"],
        "additionalProperties": false
    })
}

fn request() -> ChatRequest {
    ChatRequest {
        system: String::new(),
        messages: vec![agent_sdk::llm::Message::user("Open a ticket.")],
        tools: None,
        max_tokens: 256,
        max_tokens_explicit: true,
        session_id: None,
        cached_content: None,
        thinking: None,
        tool_choice: None,
        response_format: Some(ResponseFormat::new("ticket", ticket_schema())),
        cache: None,
    }
}

fn ok(content: Vec<ContentBlock>) -> ChatOutcome {
    ChatOutcome::Success(ChatResponse {
        id: "r".to_owned(),
        content,
        model: "scripted-model".to_owned(),
        stop_reason: Some(StopReason::EndTurn),
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
    })
}

fn text(t: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text { text: t.to_owned() }]
}

fn respond_tool(input: serde_json::Value) -> Vec<ContentBlock> {
    vec![ContentBlock::ToolUse {
        id: "call".to_owned(),
        name: RESPOND_TOOL_NAME.to_owned(),
        input,
        thought_signature: None,
    }]
}

#[tokio::test]
async fn native_provider_happy_path() -> Result<()> {
    let provider = ScriptedProvider::new(
        "openai",
        StructuredOutputSupport::Native,
        vec![ok(text(r#"{"title": "Bug", "priority": "high"}"#))],
    );

    let out = run_structured(&provider, request(), StructuredConfig::default()).await?;
    assert_eq!(out.value["title"], "Bug");
    assert_eq!(out.value["priority"], "high");
    assert_eq!(out.retries, 0);
    Ok(())
}

#[tokio::test]
async fn tool_forcing_provider_happy_path() -> Result<()> {
    let provider = ScriptedProvider::new(
        "anthropic",
        StructuredOutputSupport::ToolForcing,
        vec![ok(respond_tool(
            serde_json::json!({"title": "Bug", "priority": "low"}),
        ))],
    );

    let out = run_structured(&provider, request(), StructuredConfig::default()).await?;
    assert_eq!(out.value["priority"], "low");
    Ok(())
}

#[tokio::test]
async fn mismatch_retry_success() -> Result<()> {
    let provider = ScriptedProvider::new(
        "openai",
        StructuredOutputSupport::Native,
        vec![
            // `priority` not in the enum → schema violation.
            ok(text(r#"{"title": "Bug", "priority": "urgent"}"#)),
            ok(text(r#"{"title": "Bug", "priority": "high"}"#)),
        ],
    );

    let out = run_structured(&provider, request(), StructuredConfig { max_retries: 2 }).await?;
    assert_eq!(out.value["priority"], "high");
    assert_eq!(out.retries, 1);
    assert_eq!(provider.call_count(), 2);
    Ok(())
}

#[tokio::test]
async fn retry_exhaustion_typed_error() -> Result<()> {
    let provider = ScriptedProvider::new(
        "openai",
        StructuredOutputSupport::Native,
        vec![
            ok(text(r#"{"title": 1}"#)),
            ok(text(r#"{"title": 2}"#)),
            ok(text(r#"{"title": 3}"#)),
        ],
    );

    let err = run_structured(&provider, request(), StructuredConfig { max_retries: 2 })
        .await
        .expect_err("schema never satisfied");

    assert!(matches!(
        err,
        StructuredOutputError::RetriesExhausted { attempts: 3, .. }
    ));
    assert_eq!(provider.call_count(), 3);
    Ok(())
}
