//! `OpenAI` Responses API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Responses API (`/v1/responses`). This API is required for models like
//! `gpt-5.2-codex` that are optimized for agentic workflows.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason,
    StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

// GPT-5.2-Codex (agentic coding model, Responses API only)
pub const MODEL_GPT52_CODEX: &str = "gpt-5.2-codex";

/// Reasoning effort level for the model.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
    /// Extra-high reasoning for complex problems
    #[serde(rename = "xhigh")]
    XHigh,
}

/// `OpenAI` Responses API provider.
///
/// This provider uses the `/v1/responses` endpoint which supports models like
/// `gpt-5.2-codex` that are designed for agentic coding workflows.
#[derive(Clone)]
pub struct OpenAIResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    reasoning_effort: Option<ReasoningEffort>,
}

impl OpenAIResponsesProvider {
    /// Create a new `OpenAI` Responses API provider.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
            reasoning_effort: None,
        }
    }

    /// Create a provider with a custom base URL.
    #[must_use]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
            reasoning_effort: None,
        }
    }

    /// Create a provider using GPT-5.2-Codex (optimized for agentic coding).
    #[must_use]
    pub fn codex(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_CODEX.to_owned())
    }

    /// Set the reasoning effort level.
    #[must_use]
    pub const fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

#[async_trait]
impl LlmProvider for OpenAIResponsesProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let input = build_api_input(&request);
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());

        let api_request = ApiResponsesRequest {
            model: &self.model,
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens: Some(request.max_tokens),
            reasoning: self.reasoning_effort.map(|e| ApiReasoning { effort: e }),
        };

        tracing::debug!(
            model = %self.model,
            max_tokens = request.max_tokens,
            "OpenAI Responses API request"
        );

        let response = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&api_request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;

        tracing::debug!(
            status = %status,
            body_len = bytes.len(),
            "OpenAI Responses API response"
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::error!(status = %status, body = %body, "OpenAI Responses server error");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::warn!(status = %status, body = %body, "OpenAI Responses client error");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        let content = build_content_blocks(&api_response.output);

        // Determine stop reason based on output content
        let has_tool_calls = content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        let stop_reason = if has_tool_calls {
            Some(StopReason::ToolUse)
        } else {
            api_response.status.map(|s| match s {
                ApiStatus::Completed => StopReason::EndTurn,
                ApiStatus::Incomplete => StopReason::MaxTokens,
                ApiStatus::Failed => StopReason::StopSequence,
            })
        };

        Ok(ChatOutcome::Success(ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: api_response.usage.map_or(
                Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
                |u| Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                },
            ),
        }))
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let input = build_api_input(&request);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| ts.into_iter().map(convert_tool).collect());

            let api_request = ApiResponsesRequestStreaming {
                model: &self.model,
                input: &input,
                tools: tools.as_deref(),
                max_output_tokens: Some(request.max_tokens),
                reasoning: self.reasoning_effort.map(|e| ApiReasoning { effort: e }),
                stream: true,
            };

            tracing::debug!(
                model = %self.model,
                max_tokens = request.max_tokens,
                "OpenAI Responses API streaming request"
            );

            let Ok(response) = self.client
                .post(format!("{}/responses", self.base_url))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&api_request)
                .send()
                .await
            else {
                yield Err(anyhow::anyhow!("request failed"));
                return;
            };

            let status = response.status();

            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let recoverable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                tracing::warn!(status = %status, body = %body, "OpenAI Responses error");
                yield Ok(StreamDelta::Error { message: body, recoverable });
                return;
            }

            let mut buffer = String::new();
            let mut stream = response.bytes_stream();
            let mut usage: Option<Usage> = None;
            let mut tool_calls: std::collections::HashMap<String, ToolCallAccumulator> =
                std::collections::HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error"));
                    return;
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() { continue; }

                    let Some(data) = line.strip_prefix("data: ") else { continue; };

                    if data == "[DONE]" {
                        // Emit any accumulated tool calls
                        for acc in tool_calls.values() {
                            yield Ok(StreamDelta::ToolUseStart {
                                id: acc.id.clone(),
                                name: acc.name.clone(),
                                block_index: 1,
                            });
                            yield Ok(StreamDelta::ToolInputDelta {
                                id: acc.id.clone(),
                                delta: acc.arguments.clone(),
                                block_index: 1,
                            });
                        }

                        if let Some(u) = usage.take() {
                            yield Ok(StreamDelta::Usage(u));
                        }

                        let stop_reason = if tool_calls.is_empty() {
                            StopReason::EndTurn
                        } else {
                            StopReason::ToolUse
                        };
                        yield Ok(StreamDelta::Done { stop_reason: Some(stop_reason) });
                        return;
                    }

                    // Parse streaming event
                    if let Ok(event) = serde_json::from_str::<ApiStreamEvent>(data) {
                        match event.r#type.as_str() {
                            "response.output_text.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::TextDelta { delta, block_index: 0 });
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                if let (Some(call_id), Some(delta)) = (event.call_id, event.delta) {
                                    let acc = tool_calls.entry(call_id.clone()).or_insert_with(|| {
                                        ToolCallAccumulator {
                                            id: call_id,
                                            name: event.name.unwrap_or_default(),
                                            arguments: String::new(),
                                        }
                                    });
                                    acc.arguments.push_str(&delta);
                                }
                            }
                            "response.completed" => {
                                if let Some(resp) = event.response
                                    && let Some(u) = resp.usage
                                {
                                    usage = Some(Usage {
                                        input_tokens: u.input_tokens,
                                        output_tokens: u.output_tokens,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Stream ended without [DONE]
            if let Some(u) = usage {
                yield Ok(StreamDelta::Usage(u));
            }
            yield Ok(StreamDelta::Done { stop_reason: Some(StopReason::EndTurn) });
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "openai-responses"
    }
}

// ============================================================================
// Input building
// ============================================================================

fn build_api_input(request: &ChatRequest) -> Vec<ApiInputItem> {
    let mut items = Vec::new();

    // Add system message if present
    if !request.system.is_empty() {
        items.push(ApiInputItem::Message(ApiMessage {
            role: ApiRole::System,
            content: ApiMessageContent::Text(request.system.clone()),
        }));
    }

    // Convert messages
    for msg in &request.messages {
        match &msg.content {
            Content::Text(text) => {
                items.push(ApiInputItem::Message(ApiMessage {
                    role: match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    },
                    content: ApiMessageContent::Text(text.clone()),
                }));
            }
            Content::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            items.push(ApiInputItem::Message(ApiMessage {
                                role: match msg.role {
                                    crate::llm::Role::User => ApiRole::User,
                                    crate::llm::Role::Assistant => ApiRole::Assistant,
                                },
                                content: ApiMessageContent::Text(text.clone()),
                            }));
                        }
                        ContentBlock::Thinking { .. } => {
                            // Thinking blocks are ephemeral - not sent back to API
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            // Tool use from assistant becomes a function_call in output history
                            items.push(ApiInputItem::FunctionCall(ApiFunctionCall::new(
                                id.clone(),
                                name.clone(),
                                serde_json::to_string(input).unwrap_or_default(),
                            )));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            // Tool result becomes function_call_output
                            items.push(ApiInputItem::FunctionCallOutput(
                                ApiFunctionCallOutput::new(tool_use_id.clone(), content.clone()),
                            ));
                        }
                    }
                }
            }
        }
    }

    items
}

/// Recursively fix a JSON schema for `OpenAI` strict mode.
/// Adds `additionalProperties: false` and ensures all properties are required.
fn fix_schema_for_strict_mode(schema: &mut serde_json::Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    // Check if this is an object type schema
    let is_object_type = obj
        .get("type")
        .is_some_and(|t| t.as_str() == Some("object"));

    if is_object_type {
        // Add additionalProperties: false
        obj.insert(
            "additionalProperties".to_owned(),
            serde_json::Value::Bool(false),
        );

        // Ensure all properties are marked as required
        if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
            let all_keys: Vec<serde_json::Value> = props
                .keys()
                .map(|k| serde_json::Value::String(k.clone()))
                .collect();
            obj.insert("required".to_owned(), serde_json::Value::Array(all_keys));
        }
    }

    // Recursively process nested schemas
    if let Some(props) = obj.get_mut("properties")
        && let Some(props_obj) = props.as_object_mut()
    {
        for prop_schema in props_obj.values_mut() {
            fix_schema_for_strict_mode(prop_schema);
        }
    }

    // Process array items
    if let Some(items) = obj.get_mut("items") {
        fix_schema_for_strict_mode(items);
    }

    // Process anyOf/oneOf/allOf
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(arr) = obj.get_mut(key)
            && let Some(arr_items) = arr.as_array_mut()
        {
            for item in arr_items {
                fix_schema_for_strict_mode(item);
            }
        }
    }
}

fn convert_tool(tool: crate::llm::Tool) -> ApiTool {
    // The Responses API with strict: true requires:
    // 1. additionalProperties: false on all object schemas
    // 2. All properties must be in the required array
    // These requirements apply recursively to nested schemas
    let mut schema = tool.input_schema;
    fix_schema_for_strict_mode(&mut schema);

    ApiTool {
        r#type: "function".to_owned(),
        name: tool.name,
        description: Some(tool.description),
        parameters: Some(schema),
        strict: Some(true),
    }
}

fn build_content_blocks(output: &[ApiOutputItem]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for item in output {
        match item {
            ApiOutputItem::Message { content, .. } => {
                for c in content {
                    if let ApiOutputContent::Text { text } = c
                        && !text.is_empty()
                    {
                        blocks.push(ContentBlock::Text { text: text.clone() });
                    }
                }
            }
            ApiOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input = serde_json::from_str(arguments).unwrap_or(serde_json::Value::Null);
                blocks.push(ContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                    thought_signature: None,
                });
            }
            ApiOutputItem::Unknown => {
                // Skip unknown output types
            }
        }
    }

    blocks
}

// ============================================================================
// Streaming helpers
// ============================================================================

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
struct ApiResponsesRequest<'a> {
    model: &'a str,
    input: &'a [ApiInputItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
}

#[derive(Serialize)]
struct ApiResponsesRequestStreaming<'a> {
    model: &'a str,
    input: &'a [ApiInputItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiReasoning {
    effort: ReasoningEffort,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiInputItem {
    Message(ApiMessage),
    FunctionCall(ApiFunctionCall),
    FunctionCallOutput(ApiFunctionCallOutput),
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    System,
    User,
    Assistant,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
}

#[derive(Serialize)]
struct ApiFunctionCall {
    r#type: &'static str,
    call_id: String,
    name: String,
    arguments: String,
}

impl ApiFunctionCall {
    const fn new(call_id: String, name: String, arguments: String) -> Self {
        Self {
            r#type: "function_call",
            call_id,
            name,
            arguments,
        }
    }
}

#[derive(Serialize)]
struct ApiFunctionCallOutput {
    r#type: &'static str,
    call_id: String,
    output: String,
}

impl ApiFunctionCallOutput {
    const fn new(call_id: String, output: String) -> Self {
        Self {
            r#type: "function_call_output",
            call_id,
            output,
        }
    }
}

#[derive(Serialize)]
struct ApiTool {
    r#type: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
struct ApiResponse {
    id: String,
    model: String,
    output: Vec<ApiOutputItem>,
    #[serde(default)]
    status: Option<ApiStatus>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ApiStatus {
    Completed,
    Incomplete,
    Failed,
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(rename = "role")]
        _role: String,
        content: Vec<ApiOutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiOutputContent {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(other)]
    Unknown,
}

// ============================================================================
// Streaming Types
// ============================================================================

#[derive(Deserialize)]
struct ApiStreamEvent {
    r#type: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    response: Option<ApiStreamResponse>,
}

#[derive(Deserialize)]
struct ApiStreamResponse {
    #[serde(default)]
    usage: Option<ApiUsage>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_constant() {
        assert_eq!(MODEL_GPT52_CODEX, "gpt-5.2-codex");
    }

    #[test]
    fn test_codex_factory() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT52_CODEX);
        assert_eq!(provider.provider(), "openai-responses");
    }

    #[test]
    fn test_reasoning_effort_serialization() {
        let low = serde_json::to_string(&ReasoningEffort::Low).unwrap();
        assert_eq!(low, "\"low\"");

        let xhigh = serde_json::to_string(&ReasoningEffort::XHigh).unwrap();
        assert_eq!(xhigh, "\"xhigh\"");
    }

    #[test]
    fn test_with_reasoning_effort() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string())
            .with_reasoning_effort(ReasoningEffort::High);
        assert!(provider.reasoning_effort.is_some());
    }

    #[test]
    fn test_api_tool_serialization() {
        let tool = ApiTool {
            r#type: "function".to_owned(),
            name: "get_weather".to_owned(),
            description: Some("Get weather".to_owned()),
            parameters: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("\"strict\":true"));
    }

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "id": "resp_123",
            "model": "gpt-5.2-codex",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ]
                }
            ],
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp_123");
        assert_eq!(response.model, "gpt-5.2-codex");
        assert_eq!(response.output.len(), 1);
    }

    #[test]
    fn test_api_response_with_function_call() {
        let json = r#"{
            "id": "resp_456",
            "model": "gpt-5.2-codex",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": "{\"path\": \"test.txt\"}"
                }
            ],
            "status": "completed"
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.output.len(), 1);

        match &response.output[0] {
            ApiOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(name, "read_file");
                assert!(arguments.contains("test.txt"));
            }
            _ => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_build_content_blocks_text() {
        let output = vec![ApiOutputItem::Message {
            _role: "assistant".to_owned(),
            content: vec![ApiOutputContent::Text {
                text: "Hello!".to_owned(),
            }],
        }];

        let blocks = build_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello!"));
    }

    #[test]
    fn test_build_content_blocks_function_call() {
        let output = vec![ApiOutputItem::FunctionCall {
            call_id: "call_123".to_owned(),
            name: "test_tool".to_owned(),
            arguments: r#"{"key": "value"}"#.to_owned(),
        }];

        let blocks = build_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        assert!(
            matches!(&blocks[0], ContentBlock::ToolUse { id, name, .. } if id == "call_123" && name == "test_tool")
        );
    }
}
