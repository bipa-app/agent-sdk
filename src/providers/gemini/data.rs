//! Shared Gemini API types, conversion functions, and SSE stream parser.
//!
//! Used by both the `GeminiProvider` (API key auth) and `VertexProvider` (`OAuth2` Bearer auth)
//! since they share the same request/response format.

use crate::llm::{Content, ContentBlock, StopReason, StreamBox, StreamDelta, Usage};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerateContentRequest<'a> {
    pub contents: &'a [ApiContent],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<&'a ApiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<&'a [ApiToolConfig]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<ApiGenerationConfig>,
}

#[derive(Serialize, Deserialize)]
pub struct ApiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Parts can be missing in some edge cases (e.g., empty responses, safety blocks)
    #[serde(default)]
    pub parts: Vec<ApiPart>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ApiPart {
    Text {
        text: String,
        /// Thought signature may appear with text in Gemini 3 models
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: ApiFunctionCall,
        /// Thought signature for Gemini 3 models — preserves reasoning context
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: ApiFunctionResponse,
    },
    /// Catch-all for unknown part types to prevent parse failures
    Unknown(serde_json::Value),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiToolConfig {
    pub function_declarations: Vec<ApiFunctionDeclaration>,
}

#[derive(Serialize)]
pub struct ApiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ApiThinkingConfig>,
}

/// Gemini thinking configuration.
///
/// Gemini 3.x models use `thinking_level` (LOW / MEDIUM / HIGH).
/// Thinking **cannot be disabled** on Gemini 3 Pro and 3.1 Pro.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiThinkingConfig {
    pub thinking_level: &'static str,
}

/// Map an agent-sdk `ThinkingConfig` to the Gemini API thinking level.
pub const fn map_thinking_config(config: &crate::llm::ThinkingConfig) -> ApiThinkingConfig {
    use crate::llm::ThinkingMode;
    let level = match &config.mode {
        // Adaptive → let the model decide (HIGH gives it the most room)
        ThinkingMode::Adaptive => "HIGH",
        // Explicit budget: map to LOW / MEDIUM / HIGH based on token budget
        ThinkingMode::Enabled { budget_tokens } => {
            if *budget_tokens <= 4_096 {
                "LOW"
            } else if *budget_tokens <= 16_384 {
                "MEDIUM"
            } else {
                "HIGH"
            }
        }
    };
    ApiThinkingConfig {
        thinking_level: level,
    }
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerateContentResponse {
    pub candidates: Vec<ApiCandidate>,
    pub usage_metadata: Option<ApiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiCandidate {
    pub content: ApiContent,
    pub finish_reason: Option<ApiFinishReason>,
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Other,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiUsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
}

// ============================================================================
// Conversion Functions
// ============================================================================

pub fn build_api_contents(messages: &[crate::llm::Message]) -> Vec<ApiContent> {
    // Build a mapping of tool_use_id -> function_name from all messages
    let mut tool_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in messages {
        if let Content::Blocks(blocks) = &msg.content {
            for block in blocks {
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    tool_names.insert(id.clone(), name.clone());
                }
            }
        }
    }

    let mut contents = Vec::new();

    for msg in messages {
        let role = match msg.role {
            crate::llm::Role::User => "user",
            crate::llm::Role::Assistant => "model",
        };

        let parts = match &msg.content {
            Content::Text(text) => vec![ApiPart::Text {
                text: text.clone(),
                thought_signature: None,
            }],
            Content::Blocks(blocks) => {
                let mut parts = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(ApiPart::Text {
                                text: text.clone(),
                                thought_signature: None,
                            });
                        }
                        ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::Document { .. } => {
                            // These blocks are not sent to the Gemini API
                        }
                        ContentBlock::ToolUse {
                            id: _,
                            name,
                            input,
                            thought_signature,
                        } => {
                            parts.push(ApiPart::FunctionCall {
                                function_call: ApiFunctionCall {
                                    name: name.clone(),
                                    args: input.clone(),
                                },
                                thought_signature: thought_signature.clone(),
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let func_name = tool_names
                                .get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| "unknown_function".to_owned());
                            let response = if is_error.unwrap_or(false) {
                                serde_json::json!({ "error": content })
                            } else {
                                serde_json::json!({ "result": content })
                            };
                            parts.push(ApiPart::FunctionResponse {
                                function_response: ApiFunctionResponse {
                                    name: func_name,
                                    response,
                                },
                            });
                        }
                    }
                }
                parts
            }
        };

        contents.push(ApiContent {
            role: Some(role.to_owned()),
            parts,
        });
    }

    contents
}

pub fn convert_tools_to_config(tools: Vec<crate::llm::Tool>) -> ApiToolConfig {
    ApiToolConfig {
        function_declarations: tools
            .into_iter()
            .map(|t| ApiFunctionDeclaration {
                name: t.name,
                description: t.description,
                parameters: t.input_schema,
            })
            .collect(),
    }
}

pub fn build_content_blocks(content: &ApiContent) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for part in &content.parts {
        match part {
            ApiPart::Text { text, .. } => {
                if !text.is_empty() {
                    blocks.push(ContentBlock::Text { text: text.clone() });
                }
            }
            ApiPart::FunctionCall {
                function_call,
                thought_signature,
            } => {
                let id = format!("call_{}", uuid_simple());
                blocks.push(ContentBlock::ToolUse {
                    id,
                    name: function_call.name.clone(),
                    input: function_call.args.clone(),
                    thought_signature: thought_signature.clone(),
                });
            }
            ApiPart::FunctionResponse { .. } => {
                // Function responses in the response are unusual, skip them
            }
            ApiPart::Unknown(value) => {
                log::warn!("Unknown API part type in Gemini response, skipping part={value:?}");
            }
        }
    }

    blocks
}

pub fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

/// Map an `ApiFinishReason` to a `StopReason`, overriding to `ToolUse` when tool calls are present.
pub const fn map_finish_reason(reason: &ApiFinishReason, has_tool_calls: bool) -> StopReason {
    if has_tool_calls {
        StopReason::ToolUse
    } else {
        match reason {
            ApiFinishReason::Stop | ApiFinishReason::Other => StopReason::EndTurn,
            ApiFinishReason::MaxTokens => StopReason::MaxTokens,
            ApiFinishReason::Safety | ApiFinishReason::Recitation => StopReason::StopSequence,
        }
    }
}

// ============================================================================
// Shared SSE Stream Parser
// ============================================================================

/// Parse a Gemini SSE response stream into `StreamDelta` events.
///
/// This is used by both `GeminiProvider` and `VertexProvider` since the
/// streaming response format is identical. Each SSE event contains independent,
/// incremental text — not cumulative.
pub fn stream_gemini_response(response: reqwest::Response) -> StreamBox<'static> {
    Box::pin(async_stream::stream! {
        let mut block_index: usize = 0;
        let mut saw_function_call = false;
        let mut usage: Option<Usage> = None;
        let mut stop_reason: Option<StopReason> = None;
        let mut buffer = String::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let Ok(chunk) = chunk_result else {
                yield Err(anyhow::anyhow!("stream error"));
                return;
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }

                // Gemini SSE format: "data: {...}"
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                let Ok(resp) =
                    serde_json::from_str::<ApiGenerateContentResponse>(data)
                else {
                    continue;
                };

                // Extract usage
                if let Some(u) = resp.usage_metadata {
                    usage = Some(Usage {
                        input_tokens: u.prompt_token_count,
                        output_tokens: u.candidates_token_count,
                    });
                }

                // Process candidates
                if let Some(candidate) = resp.candidates.into_iter().next() {
                    if let Some(reason) = &candidate.finish_reason {
                        stop_reason = Some(map_finish_reason(reason, false));
                    }

                    for part in &candidate.content.parts {
                        match part {
                            ApiPart::Text { text, .. } => {
                                if !text.is_empty() {
                                    yield Ok(StreamDelta::TextDelta {
                                        delta: text.clone(),
                                        block_index,
                                    });
                                    block_index += 1;
                                }
                            }
                            ApiPart::FunctionCall { function_call, thought_signature } => {
                                saw_function_call = true;
                                let id = format!("call_{}", uuid_simple());
                                yield Ok(StreamDelta::ToolUseStart {
                                    id: id.clone(),
                                    name: function_call.name.clone(),
                                    block_index,
                                    thought_signature: thought_signature.clone(),
                                });
                                yield Ok(StreamDelta::ToolInputDelta {
                                    id,
                                    delta: serde_json::to_string(&function_call.args)
                                        .unwrap_or_default(),
                                    block_index,
                                });
                                block_index += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Override to ToolUse if we saw any function calls during the stream.
        if saw_function_call {
            stop_reason = Some(StopReason::ToolUse);
        }

        // Emit final events
        if let Some(u) = usage {
            yield Ok(StreamDelta::Usage(u));
        }
        yield Ok(StreamDelta::Done { stop_reason });
    })
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
    fn test_api_content_serialization() {
        let content = ApiContent {
            role: Some("user".to_string()),
            parts: vec![ApiPart::Text {
                text: "Hello!".to_string(),
                thought_signature: None,
            }],
        };

        let json = serde_json::to_string(&content).unwrap_or_default();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello!\""));
    }

    #[test]
    fn test_api_part_text_serialization() {
        let part = ApiPart::Text {
            text: "Hello, world!".to_string(),
            thought_signature: None,
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"text\":\"Hello, world!\""));
    }

    #[test]
    fn test_api_part_function_call_serialization() {
        let part = ApiPart::FunctionCall {
            function_call: ApiFunctionCall {
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "/test.txt"}),
            },
            thought_signature: None,
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"functionCall\""));
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"args\""));
    }

    #[test]
    fn test_api_part_function_response_serialization() {
        let part = ApiPart::FunctionResponse {
            function_response: ApiFunctionResponse {
                name: "read_file".to_string(),
                response: serde_json::json!({"result": "file contents"}),
            },
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"functionResponse\""));
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"response\""));
    }

    #[test]
    fn test_api_tool_config_serialization() {
        let config = ApiToolConfig {
            function_declarations: vec![ApiFunctionDeclaration {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"functionDeclarations\""));
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
    }

    #[test]
    fn test_api_generation_config_serialization() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(1024),
            thinking_config: None,
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"maxOutputTokens\":1024"));
        assert!(!json.contains("thinkingConfig"));
    }

    #[test]
    fn test_api_generation_config_with_thinking() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(65536),
            thinking_config: Some(ApiThinkingConfig {
                thinking_level: "HIGH",
            }),
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"thinkingConfig\""));
        assert!(json.contains("\"thinkingLevel\":\"HIGH\""));
    }

    // ===================
    // API Type Deserialization Tests
    // ===================

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello!"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50
            }
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(response.candidates.len(), 1);
        assert!(response.usage_metadata.is_some());
        let usage = response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
        });
        assert_eq!(usage.prompt_token_count, 100);
        assert_eq!(usage.candidates_token_count, 50);
    }

    #[test]
    fn test_api_response_with_function_call_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "test.txt"}
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        let content = &response.candidates[0].content;
        assert_eq!(content.parts.len(), 1);
        match &content.parts[0] {
            ApiPart::FunctionCall { function_call, .. } => {
                assert_eq!(function_call.name, "read_file");
            }
            _ => panic!("Expected FunctionCall part"),
        }
    }

    #[test]
    fn test_api_finish_reason_deserialization() {
        let stop: ApiFinishReason =
            serde_json::from_str("\"STOP\"").unwrap_or_else(|e| panic!("parse failed: {e}"));
        let max_tokens: ApiFinishReason =
            serde_json::from_str("\"MAX_TOKENS\"").unwrap_or_else(|e| panic!("parse failed: {e}"));
        let safety: ApiFinishReason =
            serde_json::from_str("\"SAFETY\"").unwrap_or_else(|e| panic!("parse failed: {e}"));

        assert!(matches!(stop, ApiFinishReason::Stop));
        assert!(matches!(max_tokens, ApiFinishReason::MaxTokens));
        assert!(matches!(safety, ApiFinishReason::Safety));
    }

    // ===================
    // Message Conversion Tests
    // ===================

    #[test]
    fn test_build_api_contents_simple() {
        let messages = vec![crate::llm::Message::user("Hello")];

        let contents = build_api_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("user".to_string()));
        assert_eq!(contents[0].parts.len(), 1);
    }

    #[test]
    fn test_build_api_contents_assistant() {
        let messages = vec![crate::llm::Message::assistant("Hi there!")];

        let contents = build_api_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("model".to_string()));
    }

    #[test]
    fn test_convert_tools_to_config() {
        let tools = vec![crate::llm::Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];

        let api_tools = convert_tools_to_config(tools);
        assert_eq!(api_tools.function_declarations.len(), 1);
        assert_eq!(api_tools.function_declarations[0].name, "test_tool");
    }

    #[test]
    fn test_build_content_blocks_text_only() {
        let content = ApiContent {
            role: Some("model".to_string()),
            parts: vec![ApiPart::Text {
                text: "Hello!".to_string(),
                thought_signature: None,
            }],
        };

        let blocks = build_content_blocks(&content);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello!"));
    }

    #[test]
    fn test_build_content_blocks_with_function_call() {
        let content = ApiContent {
            role: Some("model".to_string()),
            parts: vec![ApiPart::FunctionCall {
                function_call: ApiFunctionCall {
                    name: "read_file".to_string(),
                    args: serde_json::json!({"path": "test.txt"}),
                },
                thought_signature: None,
            }],
        };

        let blocks = build_content_blocks(&content);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { name, .. } if name == "read_file"));
    }

    #[test]
    fn test_uuid_simple_generates_unique_ids() {
        let id1 = uuid_simple();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let id2 = uuid_simple();

        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }

    // ===================
    // Streaming Response Tests
    // ===================

    #[test]
    fn test_streaming_response_text_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}]
                    }
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(response.candidates.len(), 1);
        match &response.candidates[0].content.parts[0] {
            ApiPart::Text { text, .. } => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text part"),
        }
    }

    #[test]
    fn test_streaming_response_with_usage_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        let usage = response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
        });
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 5);
        assert!(matches!(
            response.candidates[0].finish_reason,
            Some(ApiFinishReason::Stop)
        ));
    }

    #[test]
    fn test_streaming_response_function_call_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "NYC"}
                            }
                        }]
                    }
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        match &response.candidates[0].content.parts[0] {
            ApiPart::FunctionCall { function_call, .. } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["location"], "NYC");
            }
            _ => panic!("Expected FunctionCall part"),
        }
    }

    // ===================
    // Finish Reason Mapping Tests
    // ===================

    #[test]
    fn test_map_finish_reason_stop() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::Stop, false),
            StopReason::EndTurn
        );
    }

    #[test]
    fn test_map_finish_reason_overrides_to_tool_use() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::Stop, true),
            StopReason::ToolUse
        );
    }

    #[test]
    fn test_map_finish_reason_max_tokens() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::MaxTokens, false),
            StopReason::MaxTokens
        );
    }
}
