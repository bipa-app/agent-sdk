//! Stub fixtures shared across the conformance tests.
//!
//! The stubs here intentionally cover only what the conformance suite
//! exercises:
//!
//! * [`StubProvider`] — sequential drain of pre-scripted `ChatOutcome`s,
//!   used by every non-streaming, non-subagent test. Mirrors the
//!   `TestProvider` in `tests/observability_integration.rs` but lives
//!   here so the conformance file has no incidental coupling to its
//!   older sibling.
//! * [`CloneableStubProvider`] — the same contract behind an
//!   `Arc<Mutex<_>>`, required by [`agent_sdk::subagent::SubagentTool`]
//!   which clones the provider into a fresh `AgentLoop` per call.
//! * [`ScriptedStreamProvider`] — yields pre-scripted streaming deltas
//!   so the streaming-only assertions (TTFC/TPOC, A6 events) can run
//!   without a live model.
//! * [`EchoTool`] — `Observe` / `sync` tool whose body returns the
//!   input `text` verbatim. Used by every test that wants a tool span
//!   in the tree.
//! * [`ScriptedMcpTransport`] — drains a LIFO list of pre-baked
//!   JSON-RPC responses for the MCP metric assertions.
//!
//! No production code lives here.

#![allow(dead_code)] // Each conformance test pulls a different subset.

use agent_sdk::DynamicToolName;
use agent_sdk::Tool;
use agent_sdk::ToolContext;
use agent_sdk::ToolResult;
use agent_sdk::ToolTier;
use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, LlmProvider, StopReason, StreamBox,
    StreamDelta, Usage,
};
use agent_sdk::mcp::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};
use agent_sdk::mcp::transport::McpTransport;
use anyhow::{Result, anyhow};
use async_stream::stream;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::Mutex;

/// Sequential, non-clonable LLM stub.
///
/// Drains the scripted responses in order. If the script runs dry, a
/// canned `text_response("default")` is returned so accidental
/// over-consumption manifests as a stable string rather than a panic.
pub struct StubProvider {
    responses: RwLock<Vec<ChatOutcome>>,
}

impl StubProvider {
    pub const fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
        }
    }

    pub fn text_response(text: &str) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp_text".to_string(),
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

    pub fn tool_use_response(tool_id: &str, tool_name: &str, input: Value) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp_tool".to_string(),
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
}

#[async_trait]
impl LlmProvider for StubProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let mut responses = self
            .responses
            .write()
            .map_err(|_| anyhow!("StubProvider lock poisoned"))?;
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

/// `Clone`-able sibling of [`StubProvider`].
///
/// `SubagentTool` requires `LlmProvider: Clone` because it
/// instantiates a fresh `AgentLoop` per invocation. The shared
/// `Arc<Mutex<_>>` body keeps every clone draining from the same
/// scripted queue.
#[derive(Clone)]
pub struct CloneableStubProvider {
    responses: Arc<Mutex<Vec<ChatOutcome>>>,
}

impl CloneableStubProvider {
    pub fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl LlmProvider for CloneableStubProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            Ok(StubProvider::text_response("default"))
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

/// LLM provider that yields a pre-scripted streaming sequence.
///
/// Used only by the `metrics_emitted_b1` / streaming-path assertions.
/// Each delta lands after a 1 ms `sleep` so the SDK's first-chunk
/// recorder and the per-chunk timer get distinguishable timestamps.
pub struct ScriptedStreamProvider {
    script: RwLock<Vec<StreamDelta>>,
}

impl ScriptedStreamProvider {
    pub const fn new(script: Vec<StreamDelta>) -> Self {
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
        // A poisoned `RwLock` here means the test process is already
        // in a bad state; surface a fatal error rather than silently
        // yielding an empty stream. The script is consumed exactly
        // once per call by design.
        let script = match self.script.write() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(_) => {
                return Box::pin(stream! {
                    yield Err(anyhow!("ScriptedStreamProvider lock poisoned"));
                });
            }
        };
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

/// `Observe`-tier, `sync`-kind tool that echoes the supplied `text`.
///
/// Used by every conformance test that needs a tool span in the tree.
pub struct EchoTool;

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
            .ok_or_else(|| anyhow!("missing `text` argument"))?;
        Ok(ToolResult::success(text))
    }
}

/// LIFO stub MCP transport.
///
/// Tests push responses in reverse order so the next `send()` pops
/// the back of the vector and binds the caller's request id into the
/// returned envelope.
pub struct ScriptedMcpTransport {
    responses: Mutex<Vec<JsonRpcResponse>>,
}

impl ScriptedMcpTransport {
    pub fn new(responses: Vec<JsonRpcResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }

    /// Build a successful `tools/list` response carrying no tools.
    pub fn list_tools_empty() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({ "tools": [] })),
            error: None,
            id: RequestId::Number(0),
        }
    }

    /// Build a successful `tools/call` response with a single text part.
    pub fn tools_call_text(text: &str) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({
                "content": [{"type": "text", "text": text}],
                "isError": false,
            })),
            error: None,
            id: RequestId::Number(0),
        }
    }
}

#[async_trait]
impl McpTransport for ScriptedMcpTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let next = {
            let mut responses = self.responses.lock().await;
            responses
                .pop()
                .ok_or_else(|| anyhow!("ScriptedMcpTransport ran out of responses"))?
        };
        Ok(JsonRpcResponse {
            id: request.id,
            ..next
        })
    }

    async fn send_notification(&self, _request: JsonRpcRequest) -> Result<()> {
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
