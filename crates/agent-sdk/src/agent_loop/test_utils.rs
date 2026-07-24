use crate::events::AgentEventEnvelope;
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, StopReason, StreamBox, StreamDelta, Usage,
};
use crate::stores::{EventStore, InMemoryEventStore};
use crate::tools::{ListenExecuteTool, ListenStopReason, ListenToolUpdate, Tool, ToolContext};
use crate::types::{ThreadId, ToolResult, ToolTier};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

// ===================
// Mock LLM Provider
// ===================

pub struct MockProvider {
    responses: RwLock<Vec<ChatOutcome>>,
    call_count: AtomicUsize,
    requests: RwLock<Vec<ChatRequest>>,
}

impl MockProvider {
    pub fn new(responses: Vec<ChatOutcome>) -> Self {
        Self {
            responses: RwLock::new(responses),
            call_count: AtomicUsize::new(0),
            requests: RwLock::new(Vec::new()),
        }
    }

    /// Number of `chat` calls served so far.
    pub fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Every `ChatRequest` received so far, in call order.
    pub fn recorded_requests(&self) -> anyhow::Result<Vec<ChatRequest>> {
        use anyhow::Context as _;
        Ok(self
            .requests
            .read()
            .ok()
            .context("requests lock poisoned")?
            .clone())
    }

    pub fn text_response(text: &str) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            model: "mock-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    pub fn tool_use_response(
        tool_id: &str,
        tool_name: &str,
        input: serde_json::Value,
    ) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: tool_id.to_string(),
                name: tool_name.to_string(),
                input,
                thought_signature: None,
            }],
            model: "mock-model".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    /// A successful response whose stop reason signals the model's context
    /// window was exceeded. Drives the overflow-recovery / compaction-retry
    /// path in `handle_turn_stop_reason`.
    pub fn context_window_exceeded() -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "msg_overflow".to_string(),
            content: vec![],
            model: "mock-model".to_string(),
            stop_reason: Some(StopReason::ModelContextWindowExceeded),
            usage: Usage {
                input_tokens: 5,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    pub fn tool_uses_response(tool_uses: Vec<(&str, &str, serde_json::Value)>) -> ChatOutcome {
        let content = tool_uses
            .into_iter()
            .map(|(id, name, input)| ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
                thought_signature: None,
            })
            .collect();

        ChatOutcome::Success(ChatResponse {
            id: "msg_1".to_string(),
            content,
            model: "mock-model".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }
}

#[async_trait]
impl crate::llm::LlmProvider for MockProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        use anyhow::Context as _;
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        self.requests
            .write()
            .ok()
            .context("requests lock poisoned")?
            .push(request);
        let responses = self.responses.read().ok().context("lock poisoned")?;
        if idx < responses.len() {
            Ok(responses[idx].clone())
        } else {
            // Default: end conversation
            Ok(Self::text_response("Done"))
        }
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

/// One `chat_stream` attempt's scripted behaviour for [`StreamScriptProvider`].
#[derive(Clone)]
pub enum StreamScriptStep {
    /// Yield these frames (as `Ok`) in order, then end the stream.
    Frames(Vec<StreamDelta>),
    /// Yield these frames, then never complete — simulates a half-open
    /// connection that stalls mid-stream so the inactivity timeout fires.
    FramesThenStall(Vec<StreamDelta>),
}

/// A streaming provider whose `chat_stream` returns a different scripted
/// sequence of [`StreamDelta`]s per call. Lets streaming-retry and
/// inactivity-timeout behaviour be driven deterministically.
pub struct StreamScriptProvider {
    steps: RwLock<Vec<StreamScriptStep>>,
    call_count: AtomicUsize,
    probe_results: RwLock<std::collections::VecDeque<bool>>,
}

impl StreamScriptProvider {
    pub fn new(steps: Vec<StreamScriptStep>) -> Self {
        Self {
            steps: RwLock::new(steps),
            call_count: AtomicUsize::new(0),
            probe_results: RwLock::new(std::collections::VecDeque::new()),
        }
    }

    /// Script the answers `probe_connectivity` returns, in order. Once the
    /// script is exhausted (or when none is set) probes report reachable, so
    /// scripts only need to spell out the offline stretch.
    #[must_use]
    pub fn with_probe_script(self, probes: Vec<bool>) -> Self {
        if let Ok(mut scripted) = self.probe_results.write() {
            scripted.extend(probes);
        }
        self
    }
}

#[async_trait]
impl crate::llm::LlmProvider for StreamScriptProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
        // Streaming tests set `config.streaming = true`, so `chat` is only a
        // fallback (e.g. if a non-streaming path is reached unexpectedly).
        Ok(MockProvider::text_response("Done"))
    }

    async fn probe_connectivity(&self) -> bool {
        self.probe_results
            .write()
            .ok()
            .and_then(|mut probes| probes.pop_front())
            .unwrap_or(true)
    }

    fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let step = self
            .steps
            .read()
            .ok()
            .and_then(|steps| steps.get(idx).cloned());
        match step {
            Some(StreamScriptStep::Frames(frames)) => {
                Box::pin(futures::stream::iter(frames.into_iter().map(Ok)))
            }
            Some(StreamScriptStep::FramesThenStall(frames)) => Box::pin(
                futures::stream::iter(frames.into_iter().map(Ok))
                    .chain(futures::stream::pending::<Result<StreamDelta>>()),
            ),
            None => Box::pin(futures::stream::iter(std::iter::once(Ok(
                StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: None,
                },
            )))),
        }
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

pub fn new_event_store() -> Arc<InMemoryEventStore> {
    Arc::new(InMemoryEventStore::new())
}

pub async fn load_events(
    store: &dyn EventStore,
    thread_id: &ThreadId,
) -> Result<Vec<AgentEventEnvelope>> {
    store.get_events(thread_id).await
}

// ===================
// Mock Tool
// ===================

pub struct EchoTool;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestToolName {
    Echo,
    ListenEcho,
}

impl crate::tools::ToolName for TestToolName {}

impl Tool<()> for EchoTool {
    type Name = TestToolName;

    fn name(&self) -> TestToolName {
        TestToolName::Echo
    }

    fn display_name(&self) -> &'static str {
        "Echo"
    }

    fn description(&self) -> &'static str {
        "Echo the input message"
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        input: serde_json::Value,
    ) -> Result<ToolResult> {
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("no message");
        Ok(ToolResult::success(format!("Echo: {message}")))
    }
}

pub struct ListenEchoTool {
    pub cancel_calls: std::sync::Arc<AtomicUsize>,
}

impl ListenExecuteTool<()> for ListenEchoTool {
    type Name = TestToolName;

    fn name(&self) -> TestToolName {
        TestToolName::ListenEcho
    }

    fn display_name(&self) -> &'static str {
        "Listen Echo"
    }

    fn description(&self) -> &'static str {
        "Listen/execute tool used for confirmation flow tests"
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    fn listen(
        &self,
        _ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> impl futures::Stream<Item = ListenToolUpdate> + Send {
        futures::stream::iter(vec![
            ListenToolUpdate::Listening {
                operation_id: "listen-op-1".to_string(),
                revision: 1,
                message: "Preparing operation".to_string(),
                snapshot: Some(json!({ "preview": "v1" })),
                expires_at: None,
            },
            ListenToolUpdate::Ready {
                operation_id: "listen-op-1".to_string(),
                revision: 2,
                message: "Ready to execute".to_string(),
                snapshot: json!({ "preview": "v2" }),
                expires_at: None,
            },
        ])
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
        _expected_revision: u64,
    ) -> Result<ToolResult> {
        Ok(ToolResult::success("Listen execute complete"))
    }

    async fn cancel(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
        _reason: ListenStopReason,
    ) -> Result<()> {
        self.cancel_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

pub struct ScenarioListenTool {
    pub updates: Vec<ListenToolUpdate>,
    pub execute_error: Option<String>,
    pub cancel_calls: std::sync::Arc<AtomicUsize>,
}

impl ListenExecuteTool<()> for ScenarioListenTool {
    type Name = TestToolName;

    fn name(&self) -> TestToolName {
        TestToolName::ListenEcho
    }

    fn display_name(&self) -> &'static str {
        "Scenario Listen Tool"
    }

    fn description(&self) -> &'static str {
        "Configurable listen tool for edge-case tests"
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    fn listen(
        &self,
        _ctx: &ToolContext<()>,
        _input: serde_json::Value,
    ) -> impl futures::Stream<Item = ListenToolUpdate> + Send {
        futures::stream::iter(self.updates.clone())
    }

    async fn execute(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
        _expected_revision: u64,
    ) -> Result<ToolResult> {
        self.execute_error.as_ref().map_or_else(
            || Ok(ToolResult::success("Scenario execute complete")),
            |message| Err(anyhow::anyhow!(message.clone())),
        )
    }

    async fn cancel(
        &self,
        _ctx: &ToolContext<()>,
        _operation_id: &str,
        _reason: ListenStopReason,
    ) -> Result<()> {
        self.cancel_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
