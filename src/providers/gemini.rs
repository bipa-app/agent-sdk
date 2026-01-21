//! Google Gemini API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Google Gemini
//! API (`generativelanguage.googleapis.com`).

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason,
    StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// Gemini 3 series (latest, Dec 2025)
pub const MODEL_GEMINI_3_FLASH: &str = "gemini-3.0-flash";
pub const MODEL_GEMINI_3_PRO: &str = "gemini-3.0-pro";

// Gemini 2.5 series
pub const MODEL_GEMINI_25_FLASH: &str = "gemini-2.5-flash";
pub const MODEL_GEMINI_25_PRO: &str = "gemini-2.5-pro";

// Gemini 2.0 series
pub const MODEL_GEMINI_2_FLASH: &str = "gemini-2.0-flash";
pub const MODEL_GEMINI_2_FLASH_LITE: &str = "gemini-2.0-flash-lite";

/// Google Gemini LLM provider.
#[derive(Clone)]
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Create a provider using Gemini 3.0 Flash (fast and capable, current default).
    #[must_use]
    pub fn flash(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_3_FLASH.to_owned())
    }

    /// Create a provider using Gemini 2.0 Flash Lite (fastest, most cost-effective).
    #[must_use]
    pub fn flash_lite(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_2_FLASH_LITE.to_owned())
    }

    /// Create a provider using Gemini 3.0 Pro (most capable).
    #[must_use]
    pub fn pro(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_3_PRO.to_owned())
    }
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for GeminiProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let contents = build_api_contents(&request.messages);
        let tools = request.tools.map(convert_tools_to_config);
        let system_instruction = if request.system.is_empty() {
            None
        } else {
            Some(ApiContent {
                role: None,
                parts: vec![ApiPart::Text {
                    text: request.system.clone(),
                    thought_signature: None,
                }],
            })
        };

        let api_request = ApiGenerateContentRequest {
            contents: &contents,
            system_instruction: system_instruction.as_ref(),
            tools: tools.as_ref().map(std::slice::from_ref),
            generation_config: Some(ApiGenerationConfig {
                max_output_tokens: Some(request.max_tokens),
            }),
        };

        tracing::debug!(
            model = %self.model,
            max_tokens = request.max_tokens,
            "Gemini LLM request"
        );

        let response = self
            .client
            .post(format!(
                "{API_BASE_URL}/models/{}:generateContent",
                self.model
            ))
            .header("Content-Type", "application/json")
            .query(&[("key", &self.api_key)])
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
            "Gemini LLM response"
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::error!(status = %status, body = %body, "Gemini server error");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::warn!(status = %status, body = %body, "Gemini client error");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiGenerateContentResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        let candidate = api_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no candidates in response"))?;

        let content = build_content_blocks(&candidate.content);

        // Warn if parts were returned but no content blocks were built (possible unknown part types)
        if content.is_empty() && !candidate.content.parts.is_empty() {
            tracing::warn!(raw_parts = ?candidate.content.parts, "Gemini parts not converted to content blocks");
        }

        // Gemini returns STOP for both natural endings and function calls.
        // We need to check if there are tool calls and override to ToolUse.
        let has_tool_calls = content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        let stop_reason = candidate.finish_reason.map(|r| {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                match r {
                    ApiFinishReason::Stop | ApiFinishReason::Other => StopReason::EndTurn,
                    ApiFinishReason::MaxTokens => StopReason::MaxTokens,
                    ApiFinishReason::Safety | ApiFinishReason::Recitation => {
                        StopReason::StopSequence
                    }
                }
            }
        });

        let usage = api_response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
        });

        Ok(ChatOutcome::Success(ChatResponse {
            id: String::new(), // Gemini doesn't provide a response ID
            content,
            model: self.model.clone(),
            stop_reason,
            usage: Usage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count,
            },
        }))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let contents = build_api_contents(&request.messages);
            let tools = request.tools.map(convert_tools_to_config);
            let system_instruction = if request.system.is_empty() {
                None
            } else {
                Some(ApiContent { role: None, parts: vec![ApiPart::Text { text: request.system.clone(), thought_signature: None }] })
            };

            let api_request = ApiGenerateContentRequest {
                contents: &contents,
                system_instruction: system_instruction.as_ref(),
                tools: tools.as_ref().map(std::slice::from_ref),
                generation_config: Some(ApiGenerationConfig { max_output_tokens: Some(request.max_tokens) }),
            };

            tracing::debug!(model = %self.model, max_tokens = request.max_tokens, "Gemini streaming LLM request");

            let Ok(response) = self.client
                .post(format!("{API_BASE_URL}/models/{}:streamGenerateContent", self.model))
                .header("Content-Type", "application/json")
                .query(&[("key", &self.api_key), ("alt", &"sse".to_string())])
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
                tracing::warn!(status = %status, body = %body, "Gemini error");
                yield Ok(StreamDelta::Error { message: body, recoverable });
                return;
            }

            let mut prev_text_len = 0usize;
            let mut prev_func_count = 0usize;
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
                    if line.is_empty() { continue; }

                    // Gemini SSE format: "data: {...}"
                    let Some(data) = line.strip_prefix("data: ") else { continue; };
                    let Ok(resp) = serde_json::from_str::<ApiGenerateContentResponse>(data) else { continue; };

                    // Extract usage
                    if let Some(u) = resp.usage_metadata {
                        usage = Some(Usage { input_tokens: u.prompt_token_count, output_tokens: u.candidates_token_count });
                    }

                    // Process candidates
                    if let Some(candidate) = resp.candidates.into_iter().next() {
                        // Check finish reason (we'll adjust for tool calls after processing content)
                        if let Some(reason) = candidate.finish_reason {
                            stop_reason = Some(match reason {
                                ApiFinishReason::Stop | ApiFinishReason::Other => StopReason::EndTurn,
                                ApiFinishReason::MaxTokens => StopReason::MaxTokens,
                                ApiFinishReason::Safety | ApiFinishReason::Recitation => StopReason::StopSequence,
                            });
                        }

                        // Emit deltas for new content
                        for (i, part) in candidate.content.parts.iter().enumerate() {
                            match part {
                                ApiPart::Text { text, .. } => {
                                    if text.len() > prev_text_len {
                                        let delta = &text[prev_text_len..];
                                        yield Ok(StreamDelta::TextDelta { delta: delta.to_string(), block_index: 0 });
                                        prev_text_len = text.len();
                                    }
                                }
                                ApiPart::FunctionCall { function_call, .. } if i >= prev_func_count => {
                                    let id = format!("call_{}", uuid_simple());
                                    yield Ok(StreamDelta::ToolUseStart { id: id.clone(), name: function_call.name.clone(), block_index: i + 1 });
                                    yield Ok(StreamDelta::ToolInputDelta { id, delta: serde_json::to_string(&function_call.args).unwrap_or_default(), block_index: i + 1 });
                                    prev_func_count = i + 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Gemini returns STOP for both natural endings and function calls.
            // Override to ToolUse if we saw any function calls during the stream.
            if prev_func_count > 0 {
                stop_reason = Some(StopReason::ToolUse);
            }

            // Emit final events
            if let Some(u) = usage { yield Ok(StreamDelta::Usage(u)); }
            yield Ok(StreamDelta::Done { stop_reason });
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "gemini"
    }
}

fn build_api_contents(messages: &[crate::llm::Message]) -> Vec<ApiContent> {
    // First, build a mapping of tool_use_id -> function_name from all messages
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
                            // Look up the function name from our mapping
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

fn convert_tools_to_config(tools: Vec<crate::llm::Tool>) -> ApiToolConfig {
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

fn build_content_blocks(content: &ApiContent) -> Vec<ContentBlock> {
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
                // Generate a unique ID for the tool call
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
                tracing::warn!(part = ?value, "Unknown API part type in Gemini response, skipping");
            }
        }
    }

    blocks
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiGenerateContentRequest<'a> {
    contents: &'a [ApiContent],
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<&'a ApiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiToolConfig]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<ApiGenerationConfig>,
}

#[derive(Serialize, Deserialize)]
struct ApiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    /// Parts can be missing in some edge cases (e.g., empty responses, safety blocks)
    #[serde(default)]
    parts: Vec<ApiPart>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum ApiPart {
    Text {
        text: String,
        /// Thought signature may appear with text in Gemini 3 models
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: ApiFunctionCall,
        /// Thought signature for Gemini 3 models - preserves reasoning context
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
struct ApiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct ApiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiToolConfig {
    function_declarations: Vec<ApiFunctionDeclaration>,
}

#[derive(Serialize)]
struct ApiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiGenerateContentResponse {
    candidates: Vec<ApiCandidate>,
    usage_metadata: Option<ApiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiCandidate {
    content: ApiContent,
    finish_reason: Option<ApiFinishReason>,
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ApiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Other,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // Constructor Tests
    // ===================

    #[test]
    fn test_new_creates_provider_with_custom_model() {
        let provider = GeminiProvider::new("test-api-key".to_string(), "custom-model".to_string());

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_flash_factory_creates_flash_provider() {
        let provider = GeminiProvider::flash("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_3_FLASH);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_flash_lite_factory_creates_flash_lite_provider() {
        let provider = GeminiProvider::flash_lite("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_2_FLASH_LITE);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_pro_factory_creates_pro_provider() {
        let provider = GeminiProvider::pro("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_3_PRO);
        assert_eq!(provider.provider(), "gemini");
    }

    // ===================
    // Model Constants Tests
    // ===================

    #[test]
    fn test_model_constants_have_expected_values() {
        assert_eq!(MODEL_GEMINI_3_FLASH, "gemini-3.0-flash");
        assert_eq!(MODEL_GEMINI_3_PRO, "gemini-3.0-pro");
        assert_eq!(MODEL_GEMINI_25_FLASH, "gemini-2.5-flash");
        assert_eq!(MODEL_GEMINI_25_PRO, "gemini-2.5-pro");
        assert_eq!(MODEL_GEMINI_2_FLASH, "gemini-2.0-flash");
        assert_eq!(MODEL_GEMINI_2_FLASH_LITE, "gemini-2.0-flash-lite");
    }

    // ===================
    // Clone Tests
    // ===================

    #[test]
    fn test_provider_is_cloneable() {
        let provider = GeminiProvider::new("test-api-key".to_string(), "test-model".to_string());
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }

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

        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello!\""));
    }

    #[test]
    fn test_api_part_text_serialization() {
        let part = ApiPart::Text {
            text: "Hello, world!".to_string(),
            thought_signature: None,
        };

        let json = serde_json::to_string(&part).unwrap();
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

        let json = serde_json::to_string(&part).unwrap();
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

        let json = serde_json::to_string(&part).unwrap();
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

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"functionDeclarations\""));
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
    }

    #[test]
    fn test_api_generation_config_serialization() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(1024),
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"maxOutputTokens\":1024"));
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

        let response: ApiGenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert!(response.usage_metadata.is_some());
        let usage = response.usage_metadata.unwrap();
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

        let response: ApiGenerateContentResponse = serde_json::from_str(json).unwrap();
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
        let stop: ApiFinishReason = serde_json::from_str("\"STOP\"").unwrap();
        let max_tokens: ApiFinishReason = serde_json::from_str("\"MAX_TOKENS\"").unwrap();
        let safety: ApiFinishReason = serde_json::from_str("\"SAFETY\"").unwrap();

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

        // IDs should be non-empty
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

        let response: ApiGenerateContentResponse = serde_json::from_str(json).unwrap();
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

        let response: ApiGenerateContentResponse = serde_json::from_str(json).unwrap();
        let usage = response.usage_metadata.unwrap();
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

        let response: ApiGenerateContentResponse = serde_json::from_str(json).unwrap();
        match &response.candidates[0].content.parts[0] {
            ApiPart::FunctionCall { function_call, .. } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["location"], "NYC");
            }
            _ => panic!("Expected FunctionCall part"),
        }
    }
}
