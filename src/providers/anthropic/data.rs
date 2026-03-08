//! Shared Anthropic API types, conversion functions, and SSE stream parser.
//!
//! Used by both the `AnthropicProvider` (direct API key auth) and `VertexProvider`
//! (`OAuth2` Bearer auth for Claude models on Vertex AI) since they share the same
//! request/response format.

use crate::llm::{
    ChatRequest, Content, ContentBlock, ContentSource, Message, Role, StopReason, StreamDelta,
    Usage,
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
    /// Catch-all for unknown content block types (future API additions).
    #[serde(other)]
    Unknown,
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
    /// Catch-all for unknown delta types (future API additions).
    #[serde(other)]
    Unknown,
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
///
/// Anthropic requires every `thinking` block to include its opaque signature
/// when that assistant content is sent back on a later turn. Older persisted
/// threads or interrupted streaming turns may contain thinking text without the
/// corresponding signature. In that case we drop the invalid thinking block
/// instead of sending a request the API will reject.
pub fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
    request
        .messages
        .iter()
        .filter_map(build_api_message)
        .collect()
}

fn build_api_message(message: &Message) -> Option<ApiMessage> {
    let role = map_api_role(message.role);
    let content = build_api_message_content(&message.content, role_label(message.role))?;
    Some(ApiMessage { role, content })
}

const fn map_api_role(role: Role) -> ApiRole {
    match role {
        Role::User => ApiRole::User,
        Role::Assistant => ApiRole::Assistant,
    }
}

const fn role_label(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn build_api_message_content(content: &Content, role_label: &str) -> Option<ApiMessageContent> {
    match content {
        Content::Text(s) => Some(ApiMessageContent::Text(s.clone())),
        Content::Blocks(blocks) => {
            let api_blocks = blocks
                .iter()
                .filter_map(|block| build_api_content_block(block, role_label))
                .collect::<Vec<_>>();

            if api_blocks.is_empty() {
                log::warn!(
                    "Skipping Anthropic {role_label} message because all content blocks were removed"
                );
                None
            } else {
                Some(ApiMessageContent::Blocks(api_blocks))
            }
        }
    }
}

fn build_api_content_block(block: &ContentBlock, role_label: &str) -> Option<ApiContentBlockInput> {
    match block {
        ContentBlock::Text { text } => Some(ApiContentBlockInput::Text { text: text.clone() }),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            let signature = signature.clone().filter(|signature| !signature.is_empty());
            if signature.is_none() {
                log::warn!("Skipping Anthropic {role_label} thinking block without signature");
                return None;
            }

            Some(ApiContentBlockInput::Thinking {
                thinking: thinking.clone(),
                signature,
            })
        }
        ContentBlock::RedactedThinking { data } => {
            Some(ApiContentBlockInput::RedactedThinking { data: data.clone() })
        }
        ContentBlock::ToolUse {
            id, name, input, ..
        } => Some(ApiContentBlockInput::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => Some(ApiContentBlockInput::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        }),
        ContentBlock::Image { source } => Some(ApiContentBlockInput::Image {
            source: ApiSource::from_content_source(source),
        }),
        ContentBlock::Document { source } => Some(ApiContentBlockInput::Document {
            source: ApiSource::from_content_source(source),
        }),
    }
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

/// Map `ApiResponseContentBlock`s to `ContentBlock`.
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

#[derive(Deserialize)]
struct SseTypeOnly {
    #[serde(rename = "type")]
    event_type: String,
}

fn preview_sse_data(data: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 200;
    let mut preview = data.chars().take(MAX_PREVIEW_CHARS).collect::<String>();
    if data.chars().count() > MAX_PREVIEW_CHARS {
        preview.push('…');
    }
    preview
}

fn log_sse_parse_error(event_type: &str, data: &str, error: &serde_json::Error) {
    log::warn!(
        "Failed to parse Anthropic SSE event type={event_type} error={error} data_preview={}",
        preview_sse_data(data)
    );
}

fn normalized_sse_event_block(event_block: &str) -> String {
    event_block.replace("\r\n", "\n").replace('\r', "\n")
}

fn parse_sse_fields(event_block: &str) -> (Option<String>, Option<String>) {
    let normalized = normalized_sse_event_block(event_block);
    let mut event_type = None;
    let mut data_lines = Vec::new();

    for line in normalized.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            event_type = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            data_lines.push(value.to_string());
        }
    }

    let data = (!data_lines.is_empty()).then(|| data_lines.join("\n"));
    let inferred_event_type = data.as_deref().and_then(|data| {
        serde_json::from_str::<SseTypeOnly>(data)
            .ok()
            .map(|event| event.event_type)
    });

    (event_type.or(inferred_event_type), data)
}

pub fn take_next_sse_event(buffer: &mut String) -> Option<String> {
    const SEPARATORS: [&str; 5] = ["\r\n\r\n", "\n\n", "\r\r", "\n\r\n", "\r\n\n"];

    let (start, separator_len) = SEPARATORS
        .into_iter()
        .filter_map(|separator| buffer.find(separator).map(|idx| (idx, separator.len())))
        .min_by_key(|(idx, _)| *idx)?;

    let event_block = buffer[..start].to_string();
    buffer.drain(..start + separator_len);
    Some(event_block)
}

pub fn is_message_stop_event(event_block: &str) -> bool {
    matches!(
        parse_sse_fields(event_block).0.as_deref(),
        Some("message_stop")
    )
}

/// Parse an SSE event block and return the corresponding `StreamDelta`.
pub fn parse_sse_event(
    event_block: &str,
    input_tokens: &mut u32,
    output_tokens: &mut u32,
    tool_ids: &mut std::collections::HashMap<usize, String>,
) -> Option<StreamDelta> {
    let (event_type, data) = parse_sse_fields(event_block);
    let event_type = event_type?;
    let data = data?;

    match event_type.as_str() {
        "message_start" => {
            // Extract input tokens from message_start
            match serde_json::from_str::<SseMessageStart>(&data) {
                Ok(event) => {
                    *input_tokens = event.message.usage.input_tokens;
                }
                Err(error) => log_sse_parse_error(&event_type, &data, &error),
            }
            None
        }
        "content_block_start" => {
            match serde_json::from_str::<SseContentBlockStart>(&data) {
                Ok(event) => match event.content_block {
                    SseContentBlock::ToolUse { id, name } => {
                        // Store the tool ID for later input deltas
                        tool_ids.insert(event.index, id.clone());
                        Some(StreamDelta::ToolUseStart {
                            id,
                            name,
                            block_index: event.index,
                            thought_signature: None,
                        })
                    }
                    SseContentBlock::RedactedThinking { data } => {
                        Some(StreamDelta::RedactedThinking {
                            data,
                            block_index: event.index,
                        })
                    }
                    SseContentBlock::Text
                    | SseContentBlock::Thinking
                    | SseContentBlock::Unknown => None,
                },
                Err(error) => {
                    log_sse_parse_error(&event_type, &data, &error);
                    None
                }
            }
        }
        "content_block_delta" => match serde_json::from_str::<SseContentBlockDelta>(&data) {
            Ok(event) => match event.delta {
                SseDelta::Text { text } => Some(StreamDelta::TextDelta {
                    delta: text,
                    block_index: event.index,
                }),
                SseDelta::Thinking { thinking } => Some(StreamDelta::ThinkingDelta {
                    delta: thinking,
                    block_index: event.index,
                }),
                SseDelta::Signature { signature } => Some(StreamDelta::SignatureDelta {
                    delta: signature,
                    block_index: event.index,
                }),
                SseDelta::InputJson { partial_json } => {
                    // Look up the tool ID from the content_block_start event
                    let id = tool_ids.get(&event.index).cloned().unwrap_or_default();
                    Some(StreamDelta::ToolInputDelta {
                        id,
                        delta: partial_json,
                        block_index: event.index,
                    })
                }
                SseDelta::Unknown => None,
            },
            Err(error) => {
                log_sse_parse_error(&event_type, &data, &error);
                None
            }
        },
        "message_delta" => {
            match serde_json::from_str::<SseMessageDelta>(&data) {
                Ok(event) => {
                    *output_tokens = event.usage.output_tokens;
                    let stop_reason = event.delta.stop_reason.as_ref().map(map_stop_reason);
                    // Emit final events
                    Some(StreamDelta::Done { stop_reason })
                }
                Err(error) => {
                    log_sse_parse_error(&event_type, &data, &error);
                    None
                }
            }
        }
        "message_stop" => {
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

    #[test]
    fn test_build_api_messages_preserves_signed_thinking_blocks() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Let me reason about this".to_string(),
                        signature: Some("sig_123".to_string()),
                    },
                    ContentBlock::Text {
                        text: "Done.".to_string(),
                    },
                ]),
            }],
            tools: None,
            max_tokens: 1024,
            thinking: None,
        };

        let messages = build_api_messages(&request);
        let json = serde_json::to_value(&messages).unwrap();

        assert_eq!(json[0]["content"][0]["type"], "thinking");
        assert_eq!(
            json[0]["content"][0]["thinking"],
            "Let me reason about this"
        );
        assert_eq!(json[0]["content"][0]["signature"], "sig_123");
        assert_eq!(json[0]["content"][1]["type"], "text");
        assert_eq!(json[0]["content"][1]["text"], "Done.");
    }

    #[test]
    fn test_build_api_messages_skips_unsigned_thinking_blocks() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Hidden reasoning".to_string(),
                        signature: None,
                    },
                    ContentBlock::Text {
                        text: "Visible answer".to_string(),
                    },
                ]),
            }],
            tools: None,
            max_tokens: 1024,
            thinking: None,
        };

        let messages = build_api_messages(&request);
        let json = serde_json::to_value(&messages).unwrap();

        assert_eq!(json[0]["content"].as_array().map(Vec::len), Some(1));
        assert_eq!(json[0]["content"][0]["type"], "text");
        assert_eq!(json[0]["content"][0]["text"], "Visible answer");
    }

    #[test]
    fn test_build_api_messages_drops_message_with_only_unsigned_thinking() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: Content::Blocks(vec![ContentBlock::Thinking {
                        thinking: "Hidden reasoning".to_string(),
                        signature: None,
                    }]),
                },
                Message::user("Continue"),
            ],
            tools: None,
            max_tokens: 1024,
            thinking: None,
        };

        let messages = build_api_messages(&request);
        let json = serde_json::to_value(&messages).unwrap();

        assert_eq!(json.as_array().map(Vec::len), Some(1));
        assert_eq!(json[0]["role"], "user");
        assert_eq!(json[0]["content"], "Continue");
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
    fn test_take_next_sse_event_handles_crlf_separator() {
        let mut buffer =
            "event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n".to_string();

        let event = take_next_sse_event(&mut buffer).unwrap();

        assert!(is_message_stop_event(&event));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_sse_signature_delta_parsing_with_multiline_data_and_crlf() {
        let event = "event: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\r\ndata: \"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_123\"}}";

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

        assert!(matches!(
            delta,
            Some(StreamDelta::SignatureDelta { delta, block_index })
            if delta == "sig_123" && block_index == 0
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
    fn test_sse_delta_unknown_type_does_not_fail() {
        // Future API additions should deserialize as Unknown rather than
        // causing the entire content_block_delta event to fail.
        let unknown: SseDelta =
            serde_json::from_str(r#"{"type":"citations_delta","citations":[]}"#).unwrap();
        assert!(matches!(unknown, SseDelta::Unknown));
    }

    #[test]
    fn test_sse_content_block_unknown_type_does_not_fail() {
        let unknown: SseContentBlock = serde_json::from_str(
            r#"{"type":"server_tool_use","id":"st_1","name":"web_search","input":{}}"#,
        )
        .unwrap();
        assert!(matches!(unknown, SseContentBlock::Unknown));
    }

    #[test]
    fn test_sse_signature_delta_parsing() {
        let event = r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_123"}}"#;

        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut tool_ids = std::collections::HashMap::new();
        let delta = parse_sse_event(event, &mut input_tokens, &mut output_tokens, &mut tool_ids);

        assert!(matches!(
            delta,
            Some(StreamDelta::SignatureDelta { delta, block_index })
            if delta == "sig_123" && block_index == 0
        ));
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
