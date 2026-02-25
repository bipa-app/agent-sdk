//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls. Supports both streaming and
//! non-streaming responses.

pub(crate) mod data;

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use data::{
    ApiMessagesRequest, ApiOutputConfig, ApiThinkingConfig, build_api_messages, build_api_tools,
    map_content_blocks, map_stop_reason, parse_sse_event,
};
use futures::StreamExt;
use reqwest::StatusCode;

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
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();

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
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let messages = build_api_messages(&request);
        let tools = build_api_tools(&request);
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
            model: Some(&self.model),
            max_tokens: request.max_tokens,
            system: &request.system,
            messages: &messages,
            tools: tools.as_deref(),
            stream: false,
            thinking,
            output_config,
            anthropic_version: None,
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

        let api_response: data::ApiResponse = serde_json::from_slice(&bytes)
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

        let content = map_content_blocks(api_response.content);
        let stop_reason = api_response.stop_reason.as_ref().map(map_stop_reason);

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
            let messages = build_api_messages(&request);
            let tools = build_api_tools(&request);
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
                model: Some(&self.model),
                max_tokens: request.max_tokens,
                system: &request.system,
                messages: &messages,
                tools: tools.as_deref(),
                stream: true,
                thinking,
                output_config,
                anthropic_version: None,
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
}
