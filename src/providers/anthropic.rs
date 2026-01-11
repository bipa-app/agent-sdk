//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
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
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let messages: Vec<ApiMessage> = request
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
            .collect();

        let tools: Option<Vec<ApiTool>> = request.tools.map(|ts| {
            ts.into_iter()
                .map(|t| ApiTool {
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema,
                })
                .collect()
        });

        let api_request = ApiMessagesRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: &request.system,
            messages: &messages,
            tools: tools.as_deref(),
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

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "anthropic"
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
// API Response Types
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
