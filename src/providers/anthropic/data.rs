//! Shared Anthropic API types, conversion functions, and SSE stream parser.
//!
//! Used by both the `AnthropicProvider` (direct API key auth) and `VertexProvider`
//! (`OAuth2` Bearer auth for Claude models on Vertex AI) since they share the same
//! request/response format.

use crate::llm::{
    ChatRequest, Content, ContentBlock, ContentSource, StopReason, StreamDelta, Usage,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
pub struct ApiMessagesRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    pub max_tokens: u32,
    pub system: &'a str,
    pub messages: &'a [ApiMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<&'a [ApiTool]>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ApiThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<ApiOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_version: Option<&'a str>,
}

/// Configuration for extended thinking in the API request.
#[derive(Serialize)]
#[serde(untagged)]
pub enum ApiThinkingConfig {
    Enabled {
        #[serde(rename = "type")]
        config_type: &'static str,
        budget_tokens: u32,
    },
    Adaptive {
        #[serde(rename = "type")]
        config_type: &'static str,
    },
}

impl ApiThinkingConfig {
    pub const fn from_thinking_config(config: &crate::llm::ThinkingConfig) -> Self {
        match &config.mode {
            crate::llm::ThinkingMode::Enabled { budget_tokens } => Self::Enabled {
                config_type: "enabled",
                budget_tokens: *budget_tokens,
            },
            crate::llm::ThinkingMode::Adaptive => Self::Adaptive {
                config_type: "adaptive",
            },
        }
    }
}

/// Output configuration for effort level.
#[derive(Serialize)]
pub struct ApiOutputConfig {
    pub effort: crate::llm::Effort,
}

#[derive(Serialize)]
pub struct ApiTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Serialize)]
pub struct ApiMessage {
    pub role: ApiRole,
    pub content: ApiMessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum ApiMessageContent {
    Text(String),
    Blocks(Vec<ApiContentBlockInput>),
}

#[derive(Serialize)]
pub struct ApiSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: String,
    data: String,
}

impl ApiSource {
    pub fn from_content_source(source: &ContentSource) -> Self {
        Self {
            source_type: "base64",
            media_type: source.media_type.clone(),
            data: source.data.clone(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ApiContentBlockInput {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
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
    #[serde(rename = "image")]
    Image { source: ApiSource },
    #[serde(rename = "document")]
    Document { source: ApiSource },
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiRole {
    User,
    Assistant,
}

// ============================================================================
// API Response Types (non-streaming)
// ============================================================================

#[derive(Deserialize)]
pub struct ApiResponse {
    pub id: String,
    pub content: Vec<ApiResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<ApiStopReason>,
    pub usage: ApiUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ApiResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    ModelContextWindowExceeded,
}

#[derive(Deserialize)]
pub struct ApiUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ============================================================================
// SSE Streaming Types
// ============================================================================

#[derive(Deserialize)]
pub struct SseMessageStart {
    pub message: SseMessageStartMessage,
}

#[derive(Deserialize)]
pub struct SseMessageStartMessage {
    pub usage: SseMessageStartUsage,
}

#[derive(Deserialize)]
pub struct SseMessageStartUsage {
    pub input_tokens: u32,
}

#[derive(Deserialize)]
pub struct SseContentBlockStart {
    pub index: usize,
    pub content_block: SseContentBlock,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum SseContentBlock {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "thinking")]
    Thinking,
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize)]
pub struct SseContentBlockDelta {
    pub index: usize,
    pub delta: SseDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum SseDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Deserialize)]
pub struct SseMessageDelta {
    pub delta: SseMessageDeltaData,
    pub usage: SseMessageDeltaUsage,
}

#[derive(Deserialize)]
pub struct SseMessageDeltaData {
    pub stop_reason: Option<ApiStopReason>,
}

#[derive(Deserialize)]
pub struct SseMessageDeltaUsage {
    pub output_tokens: u32,
}

// ============================================================================
// Conversion Functions
// ============================================================================

/// Build API messages from the chat request.
pub fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
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
                            ContentBlock::Thinking {
                                thinking,
                                signature,
                                ..
                            } => ApiContentBlockInput::Thinking {
                                thinking: thinking.clone(),
                                signature: signature.clone(),
                            },
                            ContentBlock::RedactedThinking { data } => {
                                ApiContentBlockInput::RedactedThinking { data: data.clone() }
                            }
                            ContentBlock::ToolUse {
                                id, name, input, ..
                            } => ApiContentBlockInput::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            },
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => ApiContentBlockInput::ToolResult {
                                tool_use_id: tool_use_id.clone(),
                                content: content.clone(),
                                is_error: *is_error,
                            },
                            ContentBlock::Image { source } => ApiContentBlockInput::Image {
                                source: ApiSource::from_content_source(source),
                            },
                            ContentBlock::Document { source } => ApiContentBlockInput::Document {
                                source: ApiSource::from_content_source(source),
                            },
                        })
                        .collect(),
                ),
            },
        })
        .collect()
}

/// Build API tools from the chat request.
pub fn build_api_tools(request: &ChatRequest) -> Option<Vec<ApiTool>> {
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

/// Map an `ApiStopReason` to a `StopReason`.
pub const fn map_stop_reason(reason: &ApiStopReason) -> StopReason {
    match reason {
        ApiStopReason::EndTurn => StopReason::EndTurn,
        ApiStopReason::ToolUse => StopReason::ToolUse,
        ApiStopReason::MaxTokens => StopReason::MaxTokens,
        ApiStopReason::StopSequence => StopReason::StopSequence,
        ApiStopReason::Refusal => StopReason::Refusal,
        ApiStopReason::ModelContextWindowExceeded => StopReason::ModelContextWindowExceeded,
    }
}

/// Map `ApiResponseContentBlock`s to `ContentBlock`s.
pub fn map_content_blocks(blocks: Vec<ApiResponseContentBlock>) -> Vec<ContentBlock> {
    blocks
        .into_iter()
        .map(|b| match b {
            ApiResponseContentBlock::Text { text } => ContentBlock::Text { text },
            ApiResponseContentBlock::Thinking {
                thinking,
                signature,
            } => ContentBlock::Thinking {
                thinking,
                signature,
            },
            ApiResponseContentBlock::RedactedThinking { data } => {
                ContentBlock::RedactedThinking { data }
            }
            ApiResponseContentBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature: None,
            },
        })
        .collect()
}

/// Parse an SSE event block and return the corresponding `StreamDelta`.
pub fn parse_sse_event(
    event_block: &str,
    input_tokens: &mut u32,
    output_tokens: &mut u32,
    tool_ids: &mut std::collections::HashMap<usize, String>,
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
            if let Ok(event) = serde_json::from_str::<SseContentBlockStart>(data) {
                match event.content_block {
                    SseContentBlock::ToolUse { id, name } => {
                        // Store the tool ID for later input deltas
                        tool_ids.insert(event.index, id.clone());
                        return Some(StreamDelta::ToolUseStart {
                            id,
                            name,
                            block_index: event.index,
                            thought_signature: None,
                        });
                    }
                    SseContentBlock::RedactedThinking { data } => {
                        return Some(StreamDelta::RedactedThinking {
                            data,
                            block_index: event.index,
                        });
                    }
                    SseContentBlock::Text | SseContentBlock::Thinking => {}
                }
            }
            None
        }
        Some("content_block_delta") => {
            if let Ok(event) = serde_json::from_str::<SseContentBlockDelta>(data) {
                match event.delta {
                    SseDelta::Text { text } => {
                        return Some(StreamDelta::TextDelta {
                            delta: text,
                            block_index: event.index,
                        });
                    }
                    SseDelta::Thinking { thinking } => {
                        return Some(StreamDelta::ThinkingDelta {
                            delta: thinking,
                            block_index: event.index,
                        });
                    }
                    SseDelta::Signature { signature } => {
                        return Some(StreamDelta::SignatureDelta {
                            delta: signature,
                            block_index: event.index,
                        });
                    }
                    SseDelta::InputJson { partial_json } => {
                        // Look up the tool ID from the content_block_start event
                        let id = tool_ids.get(&event.index).cloned().unwrap_or_default();
                        return Some(StreamDelta::ToolInputDelta {
                            id,
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
                let stop_reason = event.delta.stop_reason.as_ref().map(map_stop_reason);
                // Emit final events
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
            model: Some("claude-3-5-sonnet"),
            max_tokens: 1024,
            system: "You are helpful.",
            messages: &messages,
            tools: None,
            stream: true,
            thinking: None,
            output_config: None,
            anthropic_version: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"model\":\"claude-3-5-sonnet\""));
        // anthropic_version should be skipped when None
        assert!(!json.contains("anthropic_version"));
    }

    #[test]
    fn test_api_request_without_model() {
        let messages = vec![];
        let request = ApiMessagesRequest {
            model: None,
            max_tokens: 1024,
            system: "You are helpful.",
            messages: &messages,
            tools: None,
            stream: false,
            thinking: None,
            output_config: None,
            anthropic_version: Some("vertex-2023-10-16"),
        };

        let json = serde_json::to_string(&request).unwrap();
        // model should be skipped when None
        assert!(!json.contains("\"model\""));
        assert!(json.contains("\"anthropic_version\":\"vertex-2023-10-16\""));
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
            _ => {
                panic!("Expected ToolUse content block")
            }
        }
    }

    #[test]
    fn test_api_stop_reason_deserialization() {
        let end_turn: ApiStopReason = serde_json::from_str("\"end_turn\"").unwrap();
        let tool_use: ApiStopReason = serde_json::from_str("\"tool_use\"").unwrap();
        let max_tokens: ApiStopReason = serde_json::from_str("\"max_tokens\"").unwrap();
        let stop_sequence: ApiStopReason = serde_json::from_str("\"stop_sequence\"").unwrap();
        let refusal: ApiStopReason = serde_json::from_str("\"refusal\"").unwrap();
        let ctx_exceeded: ApiStopReason =
            serde_json::from_str("\"model_context_window_exceeded\"").unwrap();

        assert!(matches!(end_turn, ApiStopReason::EndTurn));
        assert!(matches!(tool_use, ApiStopReason::ToolUse));
        assert!(matches!(max_tokens, ApiStopReason::MaxTokens));
        assert!(matches!(stop_sequence, ApiStopReason::StopSequence));
        assert!(matches!(refusal, ApiStopReason::Refusal));
        assert!(matches!(
            ctx_exceeded,
            ApiStopReason::ModelContextWindowExceeded
        ));
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
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

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
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

        assert!(matches!(
            delta,
            Some(StreamDelta::ToolUseStart { id, name, block_index, thought_signature: None })
            if id == "toolu_123" && name == "read_file" && block_index == 1
        ));
        // Verify tool ID is stored for later input deltas
        assert_eq!(tool_ids.get(&1), Some(&"toolu_123".to_string()));
    }

    #[test]
    fn test_sse_input_json_delta_parsing() {
        let event = r#"event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        // Pre-populate tool_ids as if we received the tool_use_start event
        let mut tool_ids = std::collections::HashMap::new();
        tool_ids.insert(1, "toolu_123".to_string());

        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

        // Verify the tool ID is correctly looked up
        assert!(matches!(
            delta,
            Some(StreamDelta::ToolInputDelta { id, delta, block_index })
            if id == "toolu_123" && delta == "{\"path\":" && block_index == 1
        ));
    }

    #[test]
    fn test_sse_message_start_captures_input_tokens() {
        let event = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet","usage":{"input_tokens":150}}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

        assert!(delta.is_none());
        assert_eq!(input_tokens, 150);
    }

    #[test]
    fn test_sse_message_delta_parsing() {
        let event = r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

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
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

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
        assert!(matches!(text_delta, SseDelta::Text { text } if text == "Hello"));

        let json_delta: SseDelta =
            serde_json::from_str(r#"{"type":"input_json_delta","partial_json":"{}"}"#).unwrap();
        assert!(matches!(json_delta, SseDelta::InputJson { partial_json } if partial_json == "{}"));
    }

    #[test]
    fn test_map_stop_reason() {
        assert_eq!(
            map_stop_reason(&ApiStopReason::EndTurn),
            StopReason::EndTurn
        );
        assert_eq!(
            map_stop_reason(&ApiStopReason::ToolUse),
            StopReason::ToolUse
        );
        assert_eq!(
            map_stop_reason(&ApiStopReason::MaxTokens),
            StopReason::MaxTokens
        );
        assert_eq!(
            map_stop_reason(&ApiStopReason::StopSequence),
            StopReason::StopSequence
        );
        assert_eq!(
            map_stop_reason(&ApiStopReason::Refusal),
            StopReason::Refusal
        );
        assert_eq!(
            map_stop_reason(&ApiStopReason::ModelContextWindowExceeded),
            StopReason::ModelContextWindowExceeded
        );
    }

    #[test]
    fn test_map_content_blocks() {
        let api_blocks = vec![
            ApiResponseContentBlock::Text {
                text: "Hello".to_string(),
            },
            ApiResponseContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "test.txt"}),
            },
        ];

        let blocks = map_content_blocks(api_blocks);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello"));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "read_file"));
    }
}
