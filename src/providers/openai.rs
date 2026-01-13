//! `OpenAI` API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Chat Completions API. It also supports `OpenAI`-compatible APIs (Ollama, vLLM, etc.)
//! via the `with_base_url` constructor.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub const MODEL_GPT4O: &str = "gpt-4o";
pub const MODEL_GPT4O_MINI: &str = "gpt-4o-mini";
pub const MODEL_GPT4_TURBO: &str = "gpt-4-turbo";
pub const MODEL_O1: &str = "o1";
pub const MODEL_O1_MINI: &str = "o1-mini";

/// `OpenAI` LLM provider using the Chat Completions API.
///
/// Also supports `OpenAI`-compatible APIs (Ollama, vLLM, Azure `OpenAI`, etc.)
/// via the `with_base_url` constructor.
#[derive(Clone)]
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIProvider {
    /// Create a new `OpenAI` provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Create a new provider with a custom base URL for OpenAI-compatible APIs.
    #[must_use]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
        }
    }

    /// Create a provider using GPT-4o.
    #[must_use]
    pub fn gpt4o(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT4O.to_owned())
    }

    /// Create a provider using GPT-4o-mini (fast and cost-effective).
    #[must_use]
    pub fn gpt4o_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT4O_MINI.to_owned())
    }

    /// Create a provider using GPT-4 Turbo.
    #[must_use]
    pub fn gpt4_turbo(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT4_TURBO.to_owned())
    }

    /// Create a provider using o1 (reasoning model).
    #[must_use]
    pub fn o1(api_key: String) -> Self {
        Self::new(api_key, MODEL_O1.to_owned())
    }

    /// Create a provider using o1-mini (fast reasoning model).
    #[must_use]
    pub fn o1_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_O1_MINI.to_owned())
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let messages = build_api_messages(&request);
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());

        let api_request = ApiChatRequest {
            model: &self.model,
            messages: &messages,
            max_completion_tokens: Some(request.max_tokens),
            tools: tools.as_deref(),
        };

        tracing::debug!(
            model = %self.model,
            max_tokens = request.max_tokens,
            "OpenAI LLM request"
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
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
            "OpenAI LLM response"
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::error!(status = %status, body = %body, "OpenAI server error");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            tracing::warn!(status = %status, body = %body, "OpenAI client error");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiChatResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no choices in response"))?;

        let content = build_content_blocks(&choice.message);

        let stop_reason = choice.finish_reason.map(|r| match r {
            ApiFinishReason::Stop => StopReason::EndTurn,
            ApiFinishReason::ToolCalls => StopReason::ToolUse,
            ApiFinishReason::Length => StopReason::MaxTokens,
            ApiFinishReason::ContentFilter => StopReason::StopSequence,
        });

        Ok(ChatOutcome::Success(ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: Usage {
                input_tokens: api_response.usage.prompt_tokens,
                output_tokens: api_response.usage.completion_tokens,
            },
        }))
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "openai"
    }
}

fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
    let mut messages = Vec::new();

    // Add system message first (OpenAI uses a separate message for system prompt)
    if !request.system.is_empty() {
        messages.push(ApiMessage {
            role: ApiRole::System,
            content: Some(request.system.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert SDK messages to OpenAI format
    for msg in &request.messages {
        match &msg.content {
            Content::Text(text) => {
                messages.push(ApiMessage {
                    role: match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    },
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Content::Blocks(blocks) => {
                // Handle mixed content blocks
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(ApiToolCall {
                                id: id.clone(),
                                r#type: "function".to_owned(),
                                function: ApiFunctionCall {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_owned()),
                                },
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            // Tool results are separate messages in OpenAI
                            messages.push(ApiMessage {
                                role: ApiRole::Tool,
                                content: Some(content.clone()),
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id.clone()),
                            });
                        }
                    }
                }

                // Add assistant message with text and/or tool calls
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    let role = match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    };

                    // Only add if it's an assistant message or has text content
                    if role == ApiRole::Assistant || !text_parts.is_empty() {
                        messages.push(ApiMessage {
                            role,
                            content: if text_parts.is_empty() {
                                None
                            } else {
                                Some(text_parts.join("\n"))
                            },
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls)
                            },
                            tool_call_id: None,
                        });
                    }
                }
            }
        }
    }

    messages
}

fn convert_tool(t: crate::llm::Tool) -> ApiTool {
    ApiTool {
        r#type: "function".to_owned(),
        function: ApiFunction {
            name: t.name,
            description: t.description,
            parameters: t.input_schema,
        },
    }
}

fn build_content_blocks(message: &ApiResponseMessage) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    // Add text content if present
    if let Some(content) = &message.content
        && !content.is_empty()
    {
        blocks.push(ContentBlock::Text {
            text: content.clone(),
        });
    }

    // Add tool calls if present
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    blocks
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
struct ApiChatRequest<'a> {
    model: &'a str,
    messages: &'a [ApiMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize)]
struct ApiToolCall {
    id: String,
    r#type: String,
    function: ApiFunctionCall,
}

#[derive(Serialize)]
struct ApiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ApiTool {
    r#type: String,
    function: ApiFunction,
}

#[derive(Serialize)]
struct ApiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
struct ApiChatResponse {
    id: String,
    choices: Vec<ApiChoice>,
    model: String,
    usage: ApiUsage,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiResponseMessage,
    finish_reason: Option<ApiFinishReason>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiResponseToolCall>>,
}

#[derive(Deserialize)]
struct ApiResponseToolCall {
    id: String,
    function: ApiResponseFunctionCall,
}

#[derive(Deserialize)]
struct ApiResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ApiFinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // Constructor Tests
    // ===================

    #[test]
    fn test_new_creates_provider_with_custom_model() {
        let provider = OpenAIProvider::new("test-api-key".to_string(), "custom-model".to_string());

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "openai");
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_with_base_url_creates_provider_with_custom_url() {
        let provider = OpenAIProvider::with_base_url(
            "test-api-key".to_string(),
            "llama3".to_string(),
            "http://localhost:11434/v1".to_string(),
        );

        assert_eq!(provider.model(), "llama3");
        assert_eq!(provider.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_gpt4o_factory_creates_gpt4o_provider() {
        let provider = OpenAIProvider::gpt4o("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT4O);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt4o_mini_factory_creates_gpt4o_mini_provider() {
        let provider = OpenAIProvider::gpt4o_mini("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT4O_MINI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_o1_factory_creates_o1_provider() {
        let provider = OpenAIProvider::o1("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_O1);
        assert_eq!(provider.provider(), "openai");
    }

    // ===================
    // Model Constants Tests
    // ===================

    #[test]
    fn test_model_constants_have_expected_values() {
        assert_eq!(MODEL_GPT4O, "gpt-4o");
        assert_eq!(MODEL_GPT4O_MINI, "gpt-4o-mini");
        assert_eq!(MODEL_GPT4_TURBO, "gpt-4-turbo");
        assert_eq!(MODEL_O1, "o1");
        assert_eq!(MODEL_O1_MINI, "o1-mini");
    }

    // ===================
    // Clone Tests
    // ===================

    #[test]
    fn test_provider_is_cloneable() {
        let provider = OpenAIProvider::new("test-api-key".to_string(), "test-model".to_string());
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
        assert_eq!(provider.base_url, cloned.base_url);
    }

    // ===================
    // API Type Serialization Tests
    // ===================

    #[test]
    fn test_api_role_serialization() {
        let system_role = ApiRole::System;
        let user_role = ApiRole::User;
        let assistant_role = ApiRole::Assistant;
        let tool_role = ApiRole::Tool;

        assert_eq!(serde_json::to_string(&system_role).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&user_role).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&assistant_role).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&tool_role).unwrap(), "\"tool\"");
    }

    #[test]
    fn test_api_message_serialization_simple() {
        let message = ApiMessage {
            role: ApiRole::User,
            content: Some("Hello, world!".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello, world!\""));
        // Optional fields should be omitted
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));
    }

    #[test]
    fn test_api_message_serialization_with_tool_calls() {
        let message = ApiMessage {
            role: ApiRole::Assistant,
            content: Some("Let me help.".to_string()),
            tool_calls: Some(vec![ApiToolCall {
                id: "call_123".to_string(),
                r#type: "function".to_string(),
                function: ApiFunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{\"path\": \"/test.txt\"}".to_string(),
                },
            }]),
            tool_call_id: None,
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"id\":\"call_123\""));
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"read_file\""));
    }

    #[test]
    fn test_api_tool_message_serialization() {
        let message = ApiMessage {
            role: ApiRole::Tool,
            content: Some("File contents here".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"tool\""));
        assert!(json.contains("\"tool_call_id\":\"call_123\""));
        assert!(json.contains("\"content\":\"File contents here\""));
    }

    #[test]
    fn test_api_tool_serialization() {
        let tool = ApiTool {
            r#type: "function".to_string(),
            function: ApiFunction {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string"}
                    }
                }),
            },
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
        assert!(json.contains("\"parameters\""));
    }

    // ===================
    // API Type Deserialization Tests
    // ===================

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        }"#;

        let response: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.usage.prompt_tokens, 100);
        assert_eq!(response.usage.completion_tokens, 50);
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_api_response_with_tool_calls_deserialization() {
        let json = r#"{
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\": \"test.txt\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 30
            }
        }"#;

        let response: ApiChatResponse = serde_json::from_str(json).unwrap();
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn test_api_finish_reason_deserialization() {
        let stop: ApiFinishReason = serde_json::from_str("\"stop\"").unwrap();
        let tool_calls: ApiFinishReason = serde_json::from_str("\"tool_calls\"").unwrap();
        let length: ApiFinishReason = serde_json::from_str("\"length\"").unwrap();
        let content_filter: ApiFinishReason = serde_json::from_str("\"content_filter\"").unwrap();

        assert!(matches!(stop, ApiFinishReason::Stop));
        assert!(matches!(tool_calls, ApiFinishReason::ToolCalls));
        assert!(matches!(length, ApiFinishReason::Length));
        assert!(matches!(content_filter, ApiFinishReason::ContentFilter));
    }

    // ===================
    // Message Conversion Tests
    // ===================

    #[test]
    fn test_build_api_messages_with_system() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![crate::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
        };

        let api_messages = build_api_messages(&request);
        assert_eq!(api_messages.len(), 2);
        assert_eq!(api_messages[0].role, ApiRole::System);
        assert_eq!(
            api_messages[0].content,
            Some("You are helpful.".to_string())
        );
        assert_eq!(api_messages[1].role, ApiRole::User);
        assert_eq!(api_messages[1].content, Some("Hello".to_string()));
    }

    #[test]
    fn test_build_api_messages_empty_system() {
        let request = ChatRequest {
            system: String::new(),
            messages: vec![crate::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
        };

        let api_messages = build_api_messages(&request);
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, ApiRole::User);
    }

    #[test]
    fn test_convert_tool() {
        let tool = crate::llm::Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };

        let api_tool = convert_tool(tool);
        assert_eq!(api_tool.r#type, "function");
        assert_eq!(api_tool.function.name, "test_tool");
        assert_eq!(api_tool.function.description, "A test tool");
    }

    #[test]
    fn test_build_content_blocks_text_only() {
        let message = ApiResponseMessage {
            content: Some("Hello!".to_string()),
            tool_calls: None,
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello!"));
    }

    #[test]
    fn test_build_content_blocks_with_tool_calls() {
        let message = ApiResponseMessage {
            content: Some("Let me help.".to_string()),
            tool_calls: Some(vec![ApiResponseToolCall {
                id: "call_123".to_string(),
                function: ApiResponseFunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{\"path\": \"test.txt\"}".to_string(),
                },
            }]),
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me help."));
        assert!(
            matches!(&blocks[1], ContentBlock::ToolUse { id, name, .. } if id == "call_123" && name == "read_file")
        );
    }
}
