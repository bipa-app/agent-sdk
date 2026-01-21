//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls. Supports both streaming and
//! non-streaming responses.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason,
    StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const API_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

pub const MODEL_HAIKU_35: &str = "claude-3-5-haiku-20241022";
pub const MODEL_SONNET_35: &str = "claude-3-5-sonnet-20241022";
pub const MODEL_SONNET_4: &str = "claude-sonnet-4-20250514";
pub const MODEL_OPUS_4: &str = "claude-opus-4-20250514";

/// Anthropic LLM provider using the Messages API.
#[derive(Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Create a provider using Claude 3.5 Haiku.
    #[must_use]
    pub fn haiku(api_key: String) -> Self {
        Self::new(api_key, MODEL_HAIKU_35.to_owned())
    }

    /// Create a provider using Claude Sonnet 4.
    #[must_use]
    pub fn sonnet(api_key: String) -> Self {
        Self::new(api_key, MODEL_SONNET_4.to_owned())
    }

    /// Create a provider using Claude Opus 4.
    #[must_use]
    pub fn opus(api_key: String) -> Self {
        Self::new(api_key, MODEL_OPUS_4.to_owned())
    }

    /// Build API messages from the chat request.
    fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
        request
            .messages
            .iter()
            .map(|m| ApiMessage {
                role: match m.role {
                    crate::llm::Role::User => ApiRole::User,
                    crate::llm::Role::Assistant => ApiRole::Assistant,
                },
                content: match &m.content {
                    Content::Text(s) => ApiMessageContent::Text(s.clone()),
                    Content::Blocks(blocks) => ApiMessageContent::Blocks(
                        blocks
                            .iter()
                            .map(|b| match b {
                                ContentBlock::Text { text } => {
                                    ApiContentBlockInput::Text { text: text.clone() }
                                }
                                ContentBlock::ToolUse { id, name, input } => {
                                    ApiContentBlockInput::ToolUse {
                                        id: id.clone(),
                                        name: name.clone(),
                                        input: input.clone(),
                                    }
                                }
                                ContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    is_error,
                                } => ApiContentBlockInput::ToolResult {
                                    tool_use_id: tool_use_id.clone(),
                                    content: content.clone(),
                                    is_error: *is_error,
                                },
                            })
                            .collect(),
                    ),
                },
            })
            .collect()
    }

    /// Build API tools from the chat request.
    fn build_api_tools(request: &ChatRequest) -> Option<Vec<ApiTool>> {
        request.tools.clone().map(|ts| {
            ts.into_iter()
                .map(|t| ApiTool {
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema,
                })
                .collect()
        })
    }
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let messages = Self::build_api_messages(&request);
        let tools = Self::build_api_tools(&request);

        let api_request = ApiMessagesRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: &request.system,
            messages: &messages,
            tools: tools.as_deref(),
            stream: false,
        };

        tracing::debug!(
            model = %self.model,
            max_tokens = request.max_tokens,
            "Anthropic LLM request"
        );

        let response = self
            .client
            .post(format!("{API_BASE_URL}/v1/messages"))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
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
            "Anthropic LLM response"
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::error!(status = %status, body = %body, "Anthropic server error");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::warn!(status = %status, body = %body, "Anthropic client error");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        let content: Vec<ContentBlock> = api_response
            .content
            .into_iter()
            .map(|b| match b {
                ApiResponseContentBlock::Text { text } => ContentBlock::Text { text },
                ApiResponseContentBlock::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id, name, input }
                }
            })
            .collect();

        let stop_reason = api_response.stop_reason.map(|r| match r {
            ApiStopReason::EndTurn => StopReason::EndTurn,
            ApiStopReason::ToolUse => StopReason::ToolUse,
            ApiStopReason::MaxTokens => StopReason::MaxTokens,
            ApiStopReason::StopSequence => StopReason::StopSequence,
        });

        Ok(ChatOutcome::Success(ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: Usage {
                input_tokens: api_response.usage.input_tokens,
                output_tokens: api_response.usage.output_tokens,
            },
        }))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let messages = Self::build_api_messages(&request);
            let tools = Self::build_api_tools(&request);

            let api_request = ApiMessagesRequest {
                model: &self.model,
                max_tokens: request.max_tokens,
                system: &request.system,
                messages: &messages,
                tools: tools.as_deref(),
                stream: true,
            };

            tracing::debug!(
                model = %self.model,
                max_tokens = request.max_tokens,
                "Anthropic streaming LLM request"
            );

            let response = match self
                .client
                .post(format!("{API_BASE_URL}/v1/messages"))
                .header("Content-Type", "application/json")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .json(&api_request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("request failed: {e}"));
                    return;
                }
            };

            let status = response.status();

            if status == StatusCode::TOO_MANY_REQUESTS {
                yield Ok(StreamDelta::Error {
                    message: "Rate limited".to_string(),
                    recoverable: true,
                });
                return;
            }

            if status.is_server_error() {
                let body = response.text().await.unwrap_or_default();
                tracing::error!(status = %status, body = %body, "Anthropic server error");
                yield Ok(StreamDelta::Error {
                    message: body,
                    recoverable: true,
                });
                return;
            }

            if status.is_client_error() {
                let body = response.text().await.unwrap_or_default();
                tracing::warn!(status = %status, body = %body, "Anthropic client error");
                yield Ok(StreamDelta::Error {
                    message: body,
                    recoverable: false,
                });
                return;
            }

            // Process SSE stream
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(anyhow::anyhow!("stream error: {e}"));
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (separated by double newlines)
                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // Parse SSE event
                    if let Some(delta) = parse_sse_event(&event_block, &mut input_tokens, &mut output_tokens) {
                        yield Ok(delta);
                    }
                }
            }
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }
}

/// Parse an SSE event block and return the corresponding `StreamDelta`.
fn parse_sse_event(
    event_block: &str,
    input_tokens: &mut u32,
    output_tokens: &mut u32,
) -> Option<StreamDelta> {
    let mut event_type = None;
    let mut data = None;

    for line in event_block.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            event_type = Some(value.trim());
        } else if let Some(value) = line.strip_prefix("data: ") {
            data = Some(value);
        }
    }

    let data = data?;

    match event_type {
        Some("message_start") => {
            // Extract input tokens from message_start
            if let Ok(event) = serde_json::from_str::<SseMessageStart>(data) {
                *input_tokens = event.message.usage.input_tokens;
            }
            None
        }
        Some("content_block_start") => {
            // Handle tool_use block start
            if let Ok(event) = serde_json::from_str::<SseContentBlockStart>(data)
                && let SseContentBlock::ToolUse { id, name } = event.content_block
            {
                return Some(StreamDelta::ToolUseStart {
                    id,
                    name,
                    block_index: event.index,
                });
            }
            None
        }
        Some("content_block_delta") => {
            if let Ok(event) = serde_json::from_str::<SseContentBlockDelta>(data) {
                match event.delta {
                    SseDelta::TextDelta { text } => {
                        return Some(StreamDelta::TextDelta {
                            delta: text,
                            block_index: event.index,
                        });
                    }
                    SseDelta::InputJsonDelta { partial_json } => {
                        // We need the tool ID from the content_block_start event
                        // For now, use a placeholder - the accumulator tracks by block_index
                        return Some(StreamDelta::ToolInputDelta {
                            id: String::new(), // Will be filled by accumulator via block_index
                            delta: partial_json,
                            block_index: event.index,
                        });
                    }
                }
            }
            None
        }
        Some("message_delta") => {
            if let Ok(event) = serde_json::from_str::<SseMessageDelta>(data) {
                *output_tokens = event.usage.output_tokens;
                let stop_reason = event.delta.stop_reason.map(|r| match r {
                    ApiStopReason::EndTurn => StopReason::EndTurn,
                    ApiStopReason::ToolUse => StopReason::ToolUse,
                    ApiStopReason::MaxTokens => StopReason::MaxTokens,
                    ApiStopReason::StopSequence => StopReason::StopSequence,
                });
                // Emit usage and done
                return Some(StreamDelta::Done { stop_reason });
            }
            None
        }
        Some("message_stop") => {
            // Final event - emit usage
            Some(StreamDelta::Usage(Usage {
                input_tokens: *input_tokens,
                output_tokens: *output_tokens,
            }))
        }
        _ => None,
    }
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
struct ApiMessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: &'a [ApiMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Blocks(Vec<ApiContentBlockInput>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ApiContentBlockInput {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    User,
    Assistant,
}

// ============================================================================
// API Response Types (non-streaming)
// ============================================================================

#[derive(Deserialize)]
struct ApiResponse {
    id: String,
    content: Vec<ApiResponseContentBlock>,
    model: String,
    stop_reason: Option<ApiStopReason>,
    usage: ApiUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ApiStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ============================================================================
// SSE Streaming Types
// ============================================================================

#[derive(Deserialize)]
struct SseMessageStart {
    message: SseMessageStartMessage,
}

#[derive(Deserialize)]
struct SseMessageStartMessage {
    usage: SseMessageStartUsage,
}

#[derive(Deserialize)]
struct SseMessageStartUsage {
    input_tokens: u32,
}

#[derive(Deserialize)]
struct SseContentBlockStart {
    index: usize,
    content_block: SseContentBlock,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum SseContentBlock {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize)]
struct SseContentBlockDelta {
    index: usize,
    delta: SseDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum SseDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize)]
struct SseMessageDelta {
    delta: SseMessageDeltaData,
    usage: SseMessageDeltaUsage,
}

#[derive(Deserialize)]
struct SseMessageDeltaData {
    stop_reason: Option<ApiStopReason>,
}

#[derive(Deserialize)]
struct SseMessageDeltaUsage {
    output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // Constructor Tests
    // ===================

    #[test]
    fn test_new_creates_provider_with_custom_model() {
        let provider =
            AnthropicProvider::new("test-api-key".to_string(), "custom-model".to_string());

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_haiku_factory_creates_haiku_provider() {
        let provider = AnthropicProvider::haiku("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_HAIKU_35);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_sonnet_factory_creates_sonnet_provider() {
        let provider = AnthropicProvider::sonnet("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_SONNET_4);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_opus_factory_creates_opus_provider() {
        let provider = AnthropicProvider::opus("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_OPUS_4);
        assert_eq!(provider.provider(), "anthropic");
    }

    // ===================
    // Model Constants Tests
    // ===================

    #[test]
    fn test_model_constants_have_expected_values() {
        assert!(MODEL_HAIKU_35.contains("haiku"));
        assert!(MODEL_SONNET_35.contains("sonnet"));
        assert!(MODEL_SONNET_4.contains("sonnet"));
        assert!(MODEL_OPUS_4.contains("opus"));
    }

    // ===================
    // Clone Tests
    // ===================

    #[test]
    fn test_provider_is_cloneable() {
        let provider = AnthropicProvider::new("test-api-key".to_string(), "test-model".to_string());
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }

    // ===================
    // API Type Serialization Tests
    // ===================

    #[test]
    fn test_api_role_serialization() {
        let user_role = ApiRole::User;
        let assistant_role = ApiRole::Assistant;

        let user_json = serde_json::to_string(&user_role).unwrap();
        let assistant_json = serde_json::to_string(&assistant_role).unwrap();

        assert_eq!(user_json, "\"user\"");
        assert_eq!(assistant_json, "\"assistant\"");
    }

    #[test]
    fn test_api_content_block_text_serialization() {
        let block = ApiContentBlockInput::Text {
            text: "Hello, world!".to_string(),
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello, world!\""));
    }

    #[test]
    fn test_api_content_block_tool_use_serialization() {
        let block = ApiContentBlockInput::ToolUse {
            id: "tool_123".to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({"path": "/test.txt"}),
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"id\":\"tool_123\""));
        assert!(json.contains("\"name\":\"read_file\""));
    }

    #[test]
    fn test_api_content_block_tool_result_serialization() {
        let block = ApiContentBlockInput::ToolResult {
            tool_use_id: "tool_123".to_string(),
            content: "File contents here".to_string(),
            is_error: None,
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(json.contains("\"tool_use_id\":\"tool_123\""));
        assert!(json.contains("\"content\":\"File contents here\""));
        // is_error should be skipped when None
        assert!(!json.contains("is_error"));
    }

    #[test]
    fn test_api_content_block_tool_result_with_error_serialization() {
        let block = ApiContentBlockInput::ToolResult {
            tool_use_id: "tool_123".to_string(),
            content: "Error occurred".to_string(),
            is_error: Some(true),
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"is_error\":true"));
    }

    #[test]
    fn test_api_tool_serialization() {
        let tool = ApiTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
        assert!(json.contains("input_schema"));
    }

    #[test]
    fn test_api_request_with_stream() {
        let messages = vec![];
        let request = ApiMessagesRequest {
            model: "claude-3-5-sonnet",
            max_tokens: 1024,
            system: "You are helpful.",
            messages: &messages,
            tools: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
    }

    // ===================
    // API Type Deserialization Tests
    // ===================

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "id": "msg_123",
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "msg_123");
        assert_eq!(response.model, "claude-3-5-sonnet");
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
    }

    #[test]
    fn test_api_response_with_tool_use_deserialization() {
        let json = r#"{
            "id": "msg_456",
            "content": [
                {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "test.txt"}}
            ],
            "model": "claude-3-5-sonnet",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 150,
                "output_tokens": 30
            }
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            ApiResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tool_1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "test.txt");
            }
            ApiResponseContentBlock::Text { .. } => panic!("Expected ToolUse content block"),
        }
    }

    #[test]
    fn test_api_stop_reason_deserialization() {
        let end_turn: ApiStopReason = serde_json::from_str("\"end_turn\"").unwrap();
        let tool_use: ApiStopReason = serde_json::from_str("\"tool_use\"").unwrap();
        let max_tokens: ApiStopReason = serde_json::from_str("\"max_tokens\"").unwrap();
        let stop_sequence: ApiStopReason = serde_json::from_str("\"stop_sequence\"").unwrap();

        assert!(matches!(end_turn, ApiStopReason::EndTurn));
        assert!(matches!(tool_use, ApiStopReason::ToolUse));
        assert!(matches!(max_tokens, ApiStopReason::MaxTokens));
        assert!(matches!(stop_sequence, ApiStopReason::StopSequence));
    }

    #[test]
    fn test_api_response_mixed_content_deserialization() {
        let json = r#"{
            "id": "msg_789",
            "content": [
                {"type": "text", "text": "Let me help you."},
                {"type": "tool_use", "id": "tool_2", "name": "write_file", "input": {"path": "out.txt", "content": "data"}}
            ],
            "model": "claude-3-5-sonnet",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 100
            }
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.content.len(), 2);
        assert!(
            matches!(&response.content[0], ApiResponseContentBlock::Text { text } if text == "Let me help you.")
        );
        assert!(
            matches!(&response.content[1], ApiResponseContentBlock::ToolUse { name, .. } if name == "write_file")
        );
    }

    // ===================
    // SSE Parsing Tests
    // ===================

    #[test]
    fn test_sse_text_delta_parsing() {
        let event = r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(matches!(
            delta,
            Some(StreamDelta::TextDelta { delta, block_index }) if delta == "Hello" && block_index == 0
        ));
    }

    #[test]
    fn test_sse_tool_use_start_parsing() {
        let event = r#"event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"read_file"}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(matches!(
            delta,
            Some(StreamDelta::ToolUseStart { id, name, block_index })
            if id == "toolu_123" && name == "read_file" && block_index == 1
        ));
    }

    #[test]
    fn test_sse_input_json_delta_parsing() {
        let event = r#"event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(matches!(
            delta,
            Some(StreamDelta::ToolInputDelta { delta, block_index, .. })
            if delta == "{\"path\":" && block_index == 1
        ));
    }

    #[test]
    fn test_sse_message_start_captures_input_tokens() {
        let event = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet","usage":{"input_tokens":150}}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(delta.is_none());
        assert_eq!(input_tokens, 150);
    }

    #[test]
    fn test_sse_message_delta_parsing() {
        let event = r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(matches!(
            delta,
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn)
            })
        ));
        assert_eq!(output_tokens, 42);
    }

    #[test]
    fn test_sse_message_stop_emits_usage() {
        let event = r#"event: message_stop
data: {"type":"message_stop"}"#;

        let mut input_tokens = 100;
        let mut output_tokens = 50;
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens);

        assert!(matches!(
            delta,
            Some(StreamDelta::Usage(Usage {
                input_tokens: 100,
                output_tokens: 50
            }))
        ));
    }

    #[test]
    fn test_sse_content_block_types_deserialization() {
        let text_block: SseContentBlock = serde_json::from_str(r#"{"type":"text"}"#).unwrap();
        assert!(matches!(text_block, SseContentBlock::Text));

        let tool_block: SseContentBlock =
            serde_json::from_str(r#"{"type":"tool_use","id":"123","name":"test"}"#).unwrap();
        assert!(matches!(tool_block, SseContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_sse_delta_types_deserialization() {
        let text_delta: SseDelta =
            serde_json::from_str(r#"{"type":"text_delta","text":"Hello"}"#).unwrap();
        assert!(matches!(text_delta, SseDelta::TextDelta { text } if text == "Hello"));

        let json_delta: SseDelta =
            serde_json::from_str(r#"{"type":"input_json_delta","partial_json":"{}"}"#).unwrap();
        assert!(
            matches!(json_delta, SseDelta::InputJsonDelta { partial_json } if partial_json == "{}")
        );
    }
}
