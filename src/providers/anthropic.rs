//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls. Supports both streaming and
//! non-streaming responses.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, ContentSource, LlmProvider,
    StopReason, StreamBox, StreamDelta, Usage,
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

pub const MODEL_HAIKU_45: &str = "claude-haiku-4-5-20251001";
pub const MODEL_SONNET_45: &str = "claude-sonnet-4-5-20250929";
pub const MODEL_OPUS_46: &str = "claude-opus-4-6";

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
        // Configure client with appropriate timeouts for streaming
        // - No overall timeout (streaming can take a long time)
        // - 30 second connect timeout
        // - TCP keepalive to prevent connection drops
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            client,
            api_key,
            model,
        }
    }

    /// Create a provider using Claude Haiku 4.5.
    #[must_use]
    pub fn haiku(api_key: String) -> Self {
        Self::new(api_key, MODEL_HAIKU_45.to_owned())
    }

    /// Create a provider using Claude Sonnet 4.5.
    #[must_use]
    pub fn sonnet(api_key: String) -> Self {
        Self::new(api_key, MODEL_SONNET_45.to_owned())
    }

    /// Create a provider using Claude Opus 4.6.
    #[must_use]
    pub fn opus(api_key: String) -> Self {
        Self::new(api_key, MODEL_OPUS_46.to_owned())
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
                                ContentBlock::Document { source } => {
                                    ApiContentBlockInput::Document {
                                        source: ApiSource::from_content_source(source),
                                    }
                                }
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
        let thinking = request
            .thinking
            .as_ref()
            .map(ApiThinkingConfig::from_thinking_config);
        let output_config = request
            .thinking
            .as_ref()
            .and_then(|t| t.effort)
            .map(|effort| ApiOutputConfig { effort });

        let api_request = ApiMessagesRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: &request.system,
            messages: &messages,
            tools: tools.as_deref(),
            stream: false,
            thinking,
            output_config,
        };

        log::debug!(
            "Anthropic LLM request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        // Log full request payload for debugging
        if log::log_enabled!(log::Level::Debug) {
            match serde_json::to_string_pretty(&api_request) {
                Ok(json) => log::debug!("Anthropic API request payload:\n{json}"),
                Err(e) => log::debug!("Failed to serialize request for logging: {e}"),
            }
        }

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

        log::debug!(
            "Anthropic LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("Anthropic server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("Anthropic client error status={status} body={body}");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        // Log the full response for debugging
        log::debug!(
            "Anthropic API response: id={} model={} stop_reason={:?} usage={{input_tokens={}, output_tokens={}}} content_blocks={}",
            api_response.id,
            api_response.model,
            api_response.stop_reason,
            api_response.usage.input_tokens,
            api_response.usage.output_tokens,
            api_response.content.len()
        );

        let content: Vec<ContentBlock> = api_response
            .content
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
            .collect();

        let stop_reason = api_response.stop_reason.map(|r| match r {
            ApiStopReason::EndTurn => StopReason::EndTurn,
            ApiStopReason::ToolUse => StopReason::ToolUse,
            ApiStopReason::MaxTokens => StopReason::MaxTokens,
            ApiStopReason::StopSequence => StopReason::StopSequence,
            ApiStopReason::Refusal => StopReason::Refusal,
            ApiStopReason::ModelContextWindowExceeded => StopReason::ModelContextWindowExceeded,
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
            let thinking = request
                .thinking
                .as_ref()
                .map(ApiThinkingConfig::from_thinking_config);
            let output_config = request
                .thinking
                .as_ref()
                .and_then(|t| t.effort)
                .map(|effort| ApiOutputConfig { effort });

            let api_request = ApiMessagesRequest {
                model: &self.model,
                max_tokens: request.max_tokens,
                system: &request.system,
                messages: &messages,
                tools: tools.as_deref(),
                stream: true,
                thinking,
                output_config,
            };

            log::debug!("Anthropic streaming LLM request model={} max_tokens={}", self.model, request.max_tokens);

            // Log full request payload for debugging
            if log::log_enabled!(log::Level::Debug) {
                match serde_json::to_string_pretty(&api_request) {
                    Ok(json) => log::debug!("Anthropic streaming API request payload:\n{json}"),
                    Err(e) => log::debug!("Failed to serialize streaming request for logging: {e}"),
                }
            }

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
                log::error!("Anthropic server error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    recoverable: true,
                });
                return;
            }

            if status.is_client_error() {
                let body = response.text().await.unwrap_or_default();
                log::warn!("Anthropic client error status={status} body={body}");
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
            // Track tool IDs by block index for correlating input deltas
            let mut tool_ids: std::collections::HashMap<usize, String> =
                std::collections::HashMap::new();

            let mut received_message_stop = false;
            let mut chunk_count: u64 = 0;
            let mut total_bytes: u64 = 0;

            // Drop guard to detect if the stream is dropped before completion
            struct StreamDropGuard {
                completed: bool,
                chunk_count: u64,
            }
            impl Drop for StreamDropGuard {
                fn drop(&mut self) {
                    if !self.completed {
                        // Use eprintln as a last resort since log might not be available during task cancellation
                        eprintln!(
                            "[agent-sdk] CRITICAL: SSE stream DROPPED at chunk_count={} - task was cancelled!",
                            self.chunk_count
                        );
                        log::error!(
                            "SSE stream was DROPPED before completion at chunk_count={} - the consuming task was likely cancelled",
                            self.chunk_count
                        );
                    }
                }
            }
            let mut drop_guard = StreamDropGuard { completed: false, chunk_count: 0 };

            log::debug!("Starting SSE stream processing");

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        log::error!("Stream error while reading chunk error={e} chunk_count={chunk_count} total_bytes={total_bytes}");
                        yield Err(anyhow::anyhow!("stream error: {e}"));
                        return;
                    }
                };

                chunk_count += 1;
                total_bytes += chunk.len() as u64;
                drop_guard.chunk_count = chunk_count;

                // Log progress every 10 chunks to show HTTP stream is alive
                if chunk_count.is_multiple_of(10) {
                    log::debug!("SSE chunk progress: chunk_count={chunk_count} total_bytes={total_bytes}");
                }
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (separated by double newlines)
                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // Track if we received message_stop
                    if event_block.contains("event: message_stop") {
                        log::debug!("Received message_stop event chunk_count={chunk_count} total_bytes={total_bytes}");
                        received_message_stop = true;
                    }

                    // Parse SSE event
                    if let Some(delta) = parse_sse_event(
                        &event_block,
                        &mut input_tokens,
                        &mut output_tokens,
                        &mut tool_ids,
                    ) {
                        yield Ok(delta);
                    }
                }
            }

            log::debug!(
                "SSE stream ended chunk_count={chunk_count} total_bytes={total_bytes} buffer_remaining={} received_message_stop={received_message_stop}",
                buffer.len()
            );

            // Process any remaining buffer content (handles incomplete final chunk)
            let remaining = buffer.trim();
            if !remaining.is_empty() {
                log::debug!(
                    "Processing remaining buffer content remaining_len={} remaining_preview={}",
                    remaining.len(),
                    remaining.chars().take(100).collect::<String>()
                );

                // Track if remaining buffer contains message_stop
                if remaining.contains("event: message_stop") {
                    received_message_stop = true;
                }

                if let Some(delta) = parse_sse_event(
                    remaining,
                    &mut input_tokens,
                    &mut output_tokens,
                    &mut tool_ids,
                ) {
                    yield Ok(delta);
                }
            }

            // Mark stream as properly completed
            drop_guard.completed = true;

            // If stream ended without message_stop, emit a recoverable error
            if !received_message_stop {
                log::warn!(
                    "SSE stream ended without message_stop event - stream may have been interrupted chunk_count={chunk_count} total_bytes={total_bytes}"
                );
                yield Ok(StreamDelta::Error {
                    message: "Stream ended unexpectedly without completion".to_string(),
                    recoverable: true,
                });
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
                let stop_reason = event.delta.stop_reason.map(|r| match r {
                    ApiStopReason::EndTurn => StopReason::EndTurn,
                    ApiStopReason::ToolUse => StopReason::ToolUse,
                    ApiStopReason::MaxTokens => StopReason::MaxTokens,
                    ApiStopReason::StopSequence => StopReason::StopSequence,
                    ApiStopReason::Refusal => StopReason::Refusal,
                    ApiStopReason::ModelContextWindowExceeded => {
                        StopReason::ModelContextWindowExceeded
                    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ApiThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<ApiOutputConfig>,
}

/// Configuration for extended thinking in the API request.
#[derive(Serialize)]
#[serde(untagged)]
enum ApiThinkingConfig {
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
    const fn from_thinking_config(config: &crate::llm::ThinkingConfig) -> Self {
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
struct ApiOutputConfig {
    effort: crate::llm::Effort,
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
struct ApiSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: String,
    data: String,
}

impl ApiSource {
    fn from_content_source(source: &ContentSource) -> Self {
        Self {
            source_type: "base64",
            media_type: source.media_type.clone(),
            data: source.data.clone(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ApiContentBlockInput {
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
enum ApiStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    ModelContextWindowExceeded,
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
    #[serde(rename = "thinking")]
    Thinking,
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
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
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
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

        assert_eq!(provider.model(), MODEL_HAIKU_45);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_sonnet_factory_creates_sonnet_provider() {
        let provider = AnthropicProvider::sonnet("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_SONNET_45);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_opus_factory_creates_opus_provider() {
        let provider = AnthropicProvider::opus("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_OPUS_46);
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
            thinking: None,
            output_config: None,
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
}
