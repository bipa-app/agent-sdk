//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls. Supports both streaming and
//! non-streaming responses.

pub(crate) mod data;

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, LlmProvider, StreamBox, StreamDelta,
    ThinkingConfig, Usage,
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
const CLAUDE_CODE_VERSION: &str = "2.1.62";

pub const MODEL_HAIKU_35: &str = "claude-3-5-haiku-20241022";
pub const MODEL_SONNET_35: &str = "claude-3-5-sonnet-20241022";
pub const MODEL_SONNET_4: &str = "claude-sonnet-4-20250514";
pub const MODEL_OPUS_4: &str = "claude-opus-4-20250514";

pub const MODEL_HAIKU_45: &str = "claude-haiku-4-5-20251001";
pub const MODEL_SONNET_45: &str = "claude-sonnet-4-5-20250929";
pub const MODEL_SONNET_46: &str = "claude-sonnet-4-6";
pub const MODEL_OPUS_46: &str = "claude-opus-4-6";

/// Claude Code tool name mappings for OAuth mode.
///
/// When using OAuth tokens, tool names must match Claude Code's exact casing.
const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "WebFetch",
    "WebSearch",
];

/// Maps a tool name to Claude Code's canonical casing (case-insensitive match).
fn to_claude_code_name(name: &str) -> String {
    let lower = name.to_lowercase();
    for cc_name in CLAUDE_CODE_TOOLS {
        if cc_name.to_lowercase() == lower {
            return (*cc_name).to_string();
        }
    }
    name.to_string()
}

/// Maps a Claude Code tool name back to the original tool name.
fn from_claude_code_name(name: &str, original_names: &[String]) -> String {
    let lower = name.to_lowercase();
    for original in original_names {
        if original.to_lowercase() == lower {
            return original.clone();
        }
    }
    name.to_string()
}

/// Returns true if the API key is an OAuth token (`sk-ant-oat-*`).
#[must_use]
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// Authentication mode for the Anthropic provider.
#[derive(Clone, Debug)]
enum AuthMode {
    /// Standard API key authentication (x-api-key header).
    ApiKey,
    /// OAuth token authentication (Bearer header + Claude Code identity).
    OAuth,
}

/// Anthropic LLM provider using the Messages API.
#[derive(Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    auth_mode: AuthMode,
    /// Original tool names for reverse mapping in OAuth mode (reserved for future use).
    #[allow(dead_code)]
    original_tool_names: Vec<String>,
    thinking: Option<ThinkingConfig>,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the specified API key and model.
    ///
    /// Automatically detects OAuth tokens (`sk-ant-oat-*`) and switches to
    /// Bearer auth with Claude Code identity headers.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        let auth_mode = if is_oauth_token(&api_key) {
            AuthMode::OAuth
        } else {
            AuthMode::ApiKey
        };

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
            auth_mode,
            original_tool_names: Vec::new(),
            thinking: None,
        }
    }

    /// Returns whether this provider is using OAuth authentication.
    #[must_use]
    pub const fn is_oauth(&self) -> bool {
        matches!(self.auth_mode, AuthMode::OAuth)
    }

    /// Applies authentication headers to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.auth_mode {
            AuthMode::ApiKey => builder
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION),
            AuthMode::OAuth => builder
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("anthropic-version", API_VERSION)
                .header(
                    "anthropic-beta",
                    "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
                )
                .header("user-agent", format!("claude-cli/{CLAUDE_CODE_VERSION}"))
                .header("x-app", "cli"),
        }
    }

    /// Wraps the system prompt for OAuth mode (prepends Claude Code identity).
    fn wrap_system_prompt<'a>(&self, system: &'a str) -> std::borrow::Cow<'a, str> {
        match self.auth_mode {
            AuthMode::ApiKey => std::borrow::Cow::Borrowed(system),
            AuthMode::OAuth => {
                let identity = "You are Claude Code, Anthropic's official CLI for Claude.";
                if system.is_empty() {
                    std::borrow::Cow::Owned(identity.to_string())
                } else {
                    std::borrow::Cow::Owned(format!("{identity}\n\n{system}"))
                }
            }
        }
    }

    /// Create a provider using Claude Haiku 4.5.
    #[must_use]
    pub fn haiku(api_key: String) -> Self {
        Self::new(api_key, MODEL_HAIKU_45.to_owned())
    }

    /// Create a provider using Claude Sonnet 4.6.
    #[must_use]
    pub fn sonnet(api_key: String) -> Self {
        Self::new(api_key, MODEL_SONNET_46.to_owned())
    }

    /// Create a provider using Claude Sonnet 4.5.
    #[must_use]
    pub fn sonnet_45(api_key: String) -> Self {
        Self::new(api_key, MODEL_SONNET_45.to_owned())
    }

    /// Create a provider using Claude Sonnet 4.6.
    #[must_use]
    pub fn sonnet_46(api_key: String) -> Self {
        Self::new(api_key, MODEL_SONNET_46.to_owned())
    }

    /// Create a provider using Claude Opus 4.6.
    #[must_use]
    pub fn opus(api_key: String) -> Self {
        Self::new(api_key, MODEL_OPUS_46.to_owned())
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
            Ok(thinking) => thinking,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        let messages = build_api_messages(&request);
        let tools = if self.is_oauth() {
            build_api_tools(&request).map(|tools| {
                tools
                    .into_iter()
                    .map(|mut t| {
                        t.name = to_claude_code_name(&t.name);
                        t
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            build_api_tools(&request)
        };
        let thinking = thinking_config
            .as_ref()
            .map(ApiThinkingConfig::from_thinking_config);
        let output_config = thinking_config
            .as_ref()
            .and_then(|t| t.effort)
            .map(|effort| ApiOutputConfig { effort });

        let system_prompt = self.wrap_system_prompt(&request.system);

        let api_request = ApiMessagesRequest {
            model: Some(&self.model),
            max_tokens: request.max_tokens,
            system: &system_prompt,
            messages: &messages,
            tools: tools.as_deref(),
            stream: false,
            thinking,
            output_config,
            anthropic_version: None,
        };

        log::debug!(
            "Anthropic LLM request model={} max_tokens={} oauth={}",
            self.model,
            request.max_tokens,
            self.is_oauth()
        );

        // Log full request payload for debugging
        if log::log_enabled!(log::Level::Debug) {
            match serde_json::to_string_pretty(&api_request) {
                Ok(json) => log::debug!("Anthropic API request payload:\n{json}"),
                Err(e) => log::debug!("Failed to serialize request for logging: {e}"),
            }
        }

        let builder = self
            .client
            .post(format!("{API_BASE_URL}/v1/messages"))
            .header("Content-Type", "application/json");
        let response = self
            .apply_auth(builder)
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

        let mut content = map_content_blocks(api_response.content);

        // Reverse-map tool names from Claude Code casing back to original names
        if self.is_oauth() {
            let original_names: Vec<String> = request
                .tools
                .as_ref()
                .map(|ts| ts.iter().map(|t| t.name.clone()).collect())
                .unwrap_or_default();
            for block in &mut content {
                if let ContentBlock::ToolUse { name, .. } = block {
                    *name = from_claude_code_name(name, &original_names);
                }
            }
        }

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
            let is_oauth = self.is_oauth();
            let original_tool_names: Vec<String> = request
                .tools
                .as_ref()
                .map(|ts| ts.iter().map(|t| t.name.clone()).collect())
                .unwrap_or_default();

            let messages = build_api_messages(&request);
            let tools = if is_oauth {
                build_api_tools(&request).map(|tools| {
                    tools
                        .into_iter()
                        .map(|mut t| {
                            t.name = to_claude_code_name(&t.name);
                            t
                        })
                        .collect::<Vec<_>>()
                })
            } else {
                build_api_tools(&request)
            };
            let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
                Ok(thinking) => thinking,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        recoverable: false,
                    });
                    return;
                }
            };
            let thinking = thinking_config
                .as_ref()
                .map(ApiThinkingConfig::from_thinking_config);
            let output_config = thinking_config
                .as_ref()
                .and_then(|t| t.effort)
                .map(|effort| ApiOutputConfig { effort });

            let system_prompt = self.wrap_system_prompt(&request.system);

            let api_request = ApiMessagesRequest {
                model: Some(&self.model),
                max_tokens: request.max_tokens,
                system: &system_prompt,
                messages: &messages,
                tools: tools.as_deref(),
                stream: true,
                thinking,
                output_config,
                anthropic_version: None,
            };

            log::debug!("Anthropic streaming LLM request model={} max_tokens={} oauth={}", self.model, request.max_tokens, is_oauth);

            // Log full request payload for debugging
            if log::log_enabled!(log::Level::Debug) {
                match serde_json::to_string_pretty(&api_request) {
                    Ok(json) => log::debug!("Anthropic streaming API request payload:\n{json}"),
                    Err(e) => log::debug!("Failed to serialize streaming request for logging: {e}"),
                }
            }

            let builder = self
                .client
                .post(format!("{API_BASE_URL}/v1/messages"))
                .header("Content-Type", "application/json");
            let response = match self
                .apply_auth(builder)
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
                    if let Some(mut delta) = parse_sse_event(
                        &event_block,
                        &mut input_tokens,
                        &mut output_tokens,
                        &mut tool_ids,
                    ) {
                        // Reverse-map tool names from Claude Code casing
                        if is_oauth
                            && let StreamDelta::ToolUseStart { ref mut name, .. } = delta
                        {
                            *name = from_claude_code_name(name, &original_tool_names);
                        }
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

                if let Some(mut delta) = parse_sse_event(
                    remaining,
                    &mut input_tokens,
                    &mut output_tokens,
                    &mut tool_ids,
                ) {
                    if is_oauth
                        && let StreamDelta::ToolUseStart { ref mut name, .. } = delta
                    {
                        *name = from_claude_code_name(name, &original_tool_names);
                    }
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

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
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
    fn test_only_anthropic_46_models_accept_adaptive_thinking() {
        let sonnet_46 = AnthropicProvider::sonnet_46("test-api-key".to_string());
        assert!(
            sonnet_46
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );

        let sonnet_45 = AnthropicProvider::sonnet_45("test-api-key".to_string());
        let error = sonnet_45
            .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("adaptive thinking is not supported")
        );
    }

    #[test]
    fn test_sonnet_factory_creates_sonnet_provider() {
        let provider = AnthropicProvider::sonnet("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_SONNET_46);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_sonnet_45_factory_creates_sonnet_provider() {
        let provider = AnthropicProvider::sonnet_45("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_SONNET_45);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_sonnet_46_factory_creates_sonnet_provider() {
        let provider = AnthropicProvider::sonnet_46("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_SONNET_46);
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
        assert!(MODEL_SONNET_46.contains("sonnet"));
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
