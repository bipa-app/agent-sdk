//! Google Vertex AI provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Google Vertex AI
//! platform. It supports both Gemini models (using the Gemini API format) and
//! Claude models (using the Anthropic Messages API format via `rawPredict`).
//!
//! Publisher detection is automatic based on the model name:
//! - `claude-*` models route to `publishers/anthropic` using `rawPredict`
//! - All other models route to `publishers/google` using `generateContent`

use crate::attachments::validate_request_attachments;
use crate::impls::anthropic::{
    MODEL_FABLE_5, MODEL_OPUS_46, MODEL_OPUS_47, MODEL_OPUS_48, MODEL_SONNET_5, MODEL_SONNET_46,
    data as anthropic_data,
};
use crate::impls::gemini::data::{
    ApiContent, ApiFunctionCallingConfig, ApiGenerateContentRequest, ApiGenerateContentResponse,
    ApiGenerationConfig, ApiPart, ApiUsageMetadata, build_api_contents, build_content_blocks,
    convert_tools_to_config, gemini_response_schema, map_finish_reason, map_thinking_config,
    stream_gemini_response,
};
use crate::provider::{LlmProvider, thinking_for_forced_tool};
use crate::streaming::{StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ResponseFormat, ThinkingConfig, ThinkingMode, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;

pub const MODEL_GEMINI_3_FLASH: &str = "gemini-3-flash-preview";
pub const MODEL_GEMINI_31_PRO: &str = "gemini-3.1-pro-preview";

// Legacy Gemini 3.0 Pro model kept for explicit opt-in.
pub const MODEL_GEMINI_3_PRO: &str = "gemini-3.0-pro";

/// The Anthropic API version used for Claude models on Vertex AI.
const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
const DEFAULT_SAFE_MAX_OUTPUT_TOKENS: u32 = 32_000;

/// Connect timeout for the HTTP client (matches Anthropic).
const CONNECT_TIMEOUT_SECS: u64 = 30;
/// TCP keepalive interval to keep long streaming connections from dropping.
const TCP_KEEPALIVE_SECS: u64 = 30;
/// Per-request read timeout for the **non-streaming** chat paths. Bounds a
/// black-holed endpoint so a single turn cannot hang the agent loop forever.
/// Streaming requests intentionally have no overall timeout.
const CHAT_READ_TIMEOUT_SECS: u64 = 300;

const fn vertex_cache_control() -> anthropic_data::ApiCacheControl {
    anthropic_data::ApiCacheControl::ephemeral_with_ttl(None)
}

/// Build the Claude-on-Vertex tool list, caching the tool prefix with the
/// Vertex ephemeral breakpoint.
fn build_vertex_claude_tools(request: &ChatRequest) -> Option<Vec<anthropic_data::ApiTool>> {
    anthropic_data::build_api_tools_with_cache(request, Some(vertex_cache_control()))
}

/// Google Vertex AI LLM provider.
///
/// Uses the same Gemini request/response format as `GeminiProvider` but
/// authenticates via `OAuth2` Bearer tokens and routes through the Vertex AI
/// regional endpoint.
///
/// Claude models are also supported — the provider detects the publisher from
/// the model name and uses the appropriate API format automatically.
#[derive(Clone)]
pub struct VertexProvider {
    client: reqwest::Client,
    access_token: String,
    project_id: String,
    region: String,
    model: String,
    thinking: Option<ThinkingConfig>,
}

impl VertexProvider {
    /// Create a new Vertex AI provider with full control over all parameters.
    #[must_use]
    pub fn new(access_token: String, project_id: String, region: String, model: String) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .tcp_keepalive(std::time::Duration::from_secs(TCP_KEEPALIVE_SECS))
            .build()
            .unwrap_or_else(|error| {
                log::warn!(
                    "failed to build Vertex HTTP client with timeouts ({error}); using default client"
                );
                reqwest::Client::new()
            });

        Self {
            client,
            access_token,
            project_id,
            region,
            model,
            thinking: None,
        }
    }

    /// Create a provider using Gemini 3 Flash Preview on Vertex AI.
    #[must_use]
    pub fn flash(access_token: String, project_id: String, region: String) -> Self {
        Self::new(
            access_token,
            project_id,
            region,
            MODEL_GEMINI_3_FLASH.to_owned(),
        )
    }

    /// Create a provider using Gemini 3.1 Pro Preview on Vertex AI.
    #[must_use]
    pub fn pro(access_token: String, project_id: String, region: String) -> Self {
        Self::new(
            access_token,
            project_id,
            region,
            MODEL_GEMINI_31_PRO.to_owned(),
        )
    }

    /// Detect whether the model is a Claude model (Anthropic publisher).
    fn is_claude_model(&self) -> bool {
        self.model.starts_with("claude-")
    }

    /// Build the base URL for the given publisher and model.
    ///
    /// For the `global` location the domain is `aiplatform.googleapis.com`
    /// (no region prefix). Regional locations use `{region}-aiplatform.googleapis.com`.
    fn base_url(&self, publisher: &str) -> String {
        let domain = if self.region == "global" {
            "aiplatform.googleapis.com".to_owned()
        } else {
            format!("{}-aiplatform.googleapis.com", self.region)
        };
        format!(
            "https://{domain}/v1/projects/{project}/locations/{region}/publishers/{publisher}/models/{model}",
            domain = domain,
            region = self.region,
            project = self.project_id,
            publisher = publisher,
            model = self.model,
        )
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    fn requires_anthropic_adaptive_thinking(&self) -> bool {
        matches!(
            self.model.as_str(),
            MODEL_SONNET_46
                | MODEL_SONNET_5
                | MODEL_OPUS_46
                | MODEL_OPUS_47
                | MODEL_OPUS_48
                | MODEL_FABLE_5
        )
    }

    fn build_cached_vertex_claude_messages(
        request: &ChatRequest,
    ) -> Vec<anthropic_data::ApiMessage> {
        let mut messages = anthropic_data::build_api_messages(request);
        anthropic_data::apply_cache_control_to_last_user_message(
            &mut messages,
            vertex_cache_control(),
        );
        messages
    }

    fn build_vertex_claude_system_prompt(
        system: &str,
    ) -> Option<anthropic_data::ApiSystemPrompt<'_>> {
        anthropic_data::build_api_system_prompt(system, Some(vertex_cache_control()))
    }

    /// Effective output-token budget for a request.
    ///
    /// Mirrors the Anthropic provider: an implicit budget falls back to the
    /// provider/model default ([`default_max_tokens`](LlmProvider::default_max_tokens),
    /// which clamps to Vertex's safe ceiling) instead of silently capping at
    /// `ChatRequest::DEFAULT_MAX_TOKENS`.
    fn effective_max_tokens(&self, request: &ChatRequest) -> u32 {
        if request.max_tokens_explicit {
            request.max_tokens
        } else {
            self.default_max_tokens()
        }
    }

    fn map_claude_response(api_response: anthropic_data::ApiResponse) -> ChatResponse {
        let content = anthropic_data::map_content_blocks(api_response.content);
        let stop_reason = api_response
            .stop_reason
            .as_ref()
            .map(anthropic_data::map_stop_reason);

        ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: Usage {
                input_tokens: api_response.usage.total_input_tokens(),
                output_tokens: api_response.usage.output,
                cached_input_tokens: api_response.usage.cached_input_tokens(),
                cache_creation_input_tokens: api_response.usage.cache_creation_input_tokens(),
            },
        }
    }
}

#[async_trait]
impl LlmProvider for VertexProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        if self.is_claude_model() {
            return self.chat_claude(request).await;
        }
        self.chat_gemini(request).await
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        if self.is_claude_model() {
            return self.chat_stream_claude(request);
        }
        self.chat_stream_gemini(request)
    }

    fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
        let Some(thinking) = thinking else {
            return Ok(());
        };

        if self
            .capabilities()
            .is_some_and(|caps| !caps.supports_thinking)
        {
            return Err(anyhow::anyhow!(
                "thinking is not supported for provider={} model={}",
                self.provider(),
                self.model()
            ));
        }

        if matches!(thinking.mode, ThinkingMode::Adaptive)
            && !self
                .capabilities()
                .is_some_and(|caps| caps.supports_adaptive_thinking)
        {
            return Err(anyhow::anyhow!(
                "adaptive thinking is not supported for provider={} model={}",
                self.provider(),
                self.model()
            ));
        }

        if self.is_claude_model()
            && self.requires_anthropic_adaptive_thinking()
            && matches!(thinking.mode, ThinkingMode::Enabled { .. })
        {
            return Err(anyhow::anyhow!(
                "budget_tokens thinking is deprecated for provider={} model={}; use ThinkingConfig::adaptive() instead",
                self.provider(),
                self.model()
            ));
        }

        Ok(())
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "vertex"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }

    fn default_max_tokens(&self) -> u32 {
        let provider = if self.is_claude_model() {
            "anthropic"
        } else {
            "gemini"
        };
        let model_max = self
            .capabilities()
            .and_then(|caps| caps.max_output_tokens)
            .or_else(|| {
                crate::model_capabilities::default_max_output_tokens(provider, self.model())
            })
            .unwrap_or(4096);
        model_max.clamp(4096, DEFAULT_SAFE_MAX_OUTPUT_TOKENS)
    }
}

// ============================================================================
// Gemini path (publishers/google)
// ============================================================================

impl VertexProvider {
    #[allow(clippy::too_many_lines)]
    async fn chat_gemini(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let thinking = match self.resolve_thinking_config(request.thinking.as_ref()) {
            Ok(thinking) => thinking,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let contents = build_api_contents(&request.messages);
        let tools = request
            .tools
            .as_ref()
            .map(|t| convert_tools_to_config(t.clone()));
        let tool_config = request
            .tool_choice
            .as_ref()
            .map(ApiFunctionCallingConfig::from_tool_choice);
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

        let thinking_config = thinking.as_ref().map(map_thinking_config);
        let (response_mime_type, response_schema) =
            request.response_format.as_ref().map_or((None, None), |rf| {
                (
                    Some("application/json"),
                    Some(gemini_response_schema(&rf.schema)),
                )
            });

        let max_tokens = self.effective_max_tokens(&request);
        let api_request = ApiGenerateContentRequest {
            contents: &contents,
            system_instruction: system_instruction.as_ref(),
            tools: tools.as_ref().map(std::slice::from_ref),
            tool_config,
            generation_config: Some(ApiGenerationConfig {
                max_output_tokens: Some(max_tokens),
                thinking_config,
                response_mime_type,
                response_schema,
            }),
            cached_content: request.cached_content.as_deref(),
        };

        log::debug!(
            "Vertex AI LLM request model={} max_tokens={}",
            self.model,
            max_tokens
        );

        let url = format!("{}:generateContent", self.base_url("google"));

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(CHAT_READ_TIMEOUT_SECS))
            .bearer_auth(&self.access_token)
            .json(&api_request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;

        let status = response.status();
        // Read `Retry-After` off the 429 response before the body is consumed.
        let retry_after = if status == StatusCode::TOO_MANY_REQUESTS {
            crate::http::retry_after_from_headers(response.headers())
        } else {
            None
        };
        let bytes = response
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;

        log::debug!(
            "Vertex AI LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited(retry_after));
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("Vertex AI server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("Vertex AI client error status={status} body={body}");
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

        if content.is_empty() && !candidate.content.parts.is_empty() {
            log::warn!(
                "Vertex AI parts not converted to content blocks raw_parts={:?}",
                candidate.content.parts
            );
        }

        let has_tool_calls = content
            .iter()
            .any(|b| matches!(b, agent_sdk_foundation::llm::ContentBlock::ToolUse { .. }));

        let stop_reason = candidate
            .finish_reason
            .as_ref()
            .map(|r| map_finish_reason(r, has_tool_calls));

        let usage = api_response
            .usage_metadata
            .unwrap_or(ApiUsageMetadata {
                prompt: 0,
                candidates: 0,
                cached_content: 0,
            })
            .into_usage();

        Ok(ChatOutcome::Success(ChatResponse {
            id: String::new(),
            content,
            model: self.model.clone(),
            stop_reason,
            usage,
        }))
    }

    fn chat_stream_gemini(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let thinking = match self.resolve_thinking_config(request.thinking.as_ref()) {
                Ok(thinking) => thinking,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };
            if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }

            let contents = build_api_contents(&request.messages);
            let tools = request
                .tools
                .as_ref()
                .map(|t| convert_tools_to_config(t.clone()));
            let tool_config = request
                .tool_choice
                .as_ref()
                .map(ApiFunctionCallingConfig::from_tool_choice);
            let system_instruction = build_gemini_system_instruction(&request.system);
            let thinking_config = thinking.as_ref().map(map_thinking_config);
            let (response_mime_type, response_schema) =
                gemini_response_format(request.response_format.as_ref());

            let max_tokens = self.effective_max_tokens(&request);
            let api_request = ApiGenerateContentRequest {
                contents: &contents,
                system_instruction: system_instruction.as_ref(),
                tools: tools.as_ref().map(std::slice::from_ref),
                tool_config,
                generation_config: Some(ApiGenerationConfig {
                    max_output_tokens: Some(max_tokens),
                    thinking_config,
                    response_mime_type,
                    response_schema,
                }),
                cached_content: request.cached_content.as_deref(),
            };

            log::debug!(
                "Vertex AI streaming LLM request model={} max_tokens={}",
                self.model,
                max_tokens
            );

            let url = format!("{}:streamGenerateContent?alt=sse", self.base_url("google"));

            let response = match self.send_gemini_stream_request(&url, &api_request).await {
                Ok(response) => response,
                Err(item) => {
                    yield item;
                    return;
                }
            };

            let mut inner = stream_gemini_response(response);
            while let Some(item) = futures::StreamExt::next(&mut inner).await {
                yield item;
            }
        })
    }

    /// Issue the Vertex Gemini streaming request, returning the raw response on
    /// success or a ready-to-`yield` stream item describing the failure.
    ///
    /// The `Err` payload is the exact `StreamBox` item to `yield`: an
    /// `anyhow::Error` for a transport failure, or a classified
    /// [`StreamDelta::Error`] for a non-success HTTP status. This keeps the
    /// generator's failure handling to a single `yield`.
    async fn send_gemini_stream_request(
        &self,
        url: &str,
        api_request: &ApiGenerateContentRequest<'_>,
    ) -> Result<reqwest::Response, anyhow::Result<StreamDelta>> {
        let response = match self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.access_token)
            .json(api_request)
            .send()
            .await
        {
            Ok(response) => response,
            // Include the cause so 401 detection / diagnostics survive.
            Err(e) => return Err(Err(anyhow::anyhow!("request failed: {e}"))),
        };

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let kind = if status == StatusCode::TOO_MANY_REQUESTS {
                StreamErrorKind::RateLimited
            } else if status.is_server_error() {
                StreamErrorKind::ServerError
            } else {
                StreamErrorKind::InvalidRequest
            };
            log::warn!("Vertex AI error status={status} body={body}");
            return Err(Ok(StreamDelta::Error {
                message: body,
                kind,
            }));
        }

        Ok(response)
    }
}

/// Build the Gemini `system_instruction` content from the request system prompt,
/// or `None` when the prompt is empty.
fn build_gemini_system_instruction(system: &str) -> Option<ApiContent> {
    if system.is_empty() {
        None
    } else {
        Some(ApiContent {
            role: None,
            parts: vec![ApiPart::Text {
                text: system.to_owned(),
                thought_signature: None,
            }],
        })
    }
}

/// Map an optional response format into Gemini's `(responseMimeType,
/// responseSchema)` pair, sanitizing the schema to the subset Gemini accepts.
fn gemini_response_format(
    response_format: Option<&ResponseFormat>,
) -> (Option<&'static str>, Option<serde_json::Value>) {
    response_format.map_or((None, None), |rf| {
        (
            Some("application/json"),
            Some(gemini_response_schema(&rf.schema)),
        )
    })
}

// ============================================================================
// Claude path (publishers/anthropic)
// ============================================================================

impl VertexProvider {
    async fn chat_claude(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
            // Forcing a specific tool is incompatible with extended thinking on
            // Claude (the API 400s), so drop thinking at the wire boundary even
            // when it was resurrected from the provider-configured default.
            Ok(thinking) => thinking_for_forced_tool(thinking, request.tool_choice.as_ref()),
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let messages = Self::build_cached_vertex_claude_messages(&request);
        let tools = build_vertex_claude_tools(&request);
        let thinking = thinking_config
            .as_ref()
            .map(anthropic_data::ApiThinkingConfig::from_thinking_config);
        let output_config = thinking_config
            .as_ref()
            .and_then(|t| t.effort)
            .map(|effort| anthropic_data::ApiOutputConfig { effort });
        let system = Self::build_vertex_claude_system_prompt(&request.system);
        let tool_choice = request
            .tool_choice
            .as_ref()
            .map(anthropic_data::ApiToolChoice::from_tool_choice);

        let max_tokens = self.effective_max_tokens(&request);
        let api_request = anthropic_data::ApiMessagesRequest {
            model: None, // model is in the URL for Vertex
            max_tokens,
            system,
            messages: &messages,
            tools: tools.as_deref(),
            tool_choice,
            stream: false,
            thinking,
            output_config,
            anthropic_version: Some(VERTEX_ANTHROPIC_VERSION),
        };

        log::debug!(
            "Vertex AI (Claude) LLM request model={} max_tokens={}",
            self.model,
            max_tokens
        );

        if log::log_enabled!(log::Level::Debug) {
            match serde_json::to_string_pretty(&api_request) {
                Ok(json) => log::debug!("Vertex AI (Claude) request payload:\n{json}"),
                Err(e) => log::debug!("Failed to serialize request for logging: {e}"),
            }
        }

        let url = format!("{}:rawPredict", self.base_url("anthropic"));

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(CHAT_READ_TIMEOUT_SECS))
            .bearer_auth(&self.access_token)
            .json(&api_request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;

        let status = response.status();
        // Read `Retry-After` off the 429 response before the body is consumed.
        let retry_after = if status == StatusCode::TOO_MANY_REQUESTS {
            crate::http::retry_after_from_headers(response.headers())
        } else {
            None
        };
        let bytes = response
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;

        log::debug!(
            "Vertex AI (Claude) response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited(retry_after));
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("Vertex AI (Claude) server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("Vertex AI (Claude) client error status={status} body={body}");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: anthropic_data::ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        log::debug!(
            "Vertex AI (Claude) response: id={} model={} stop_reason={:?} usage={{input_tokens={}, output_tokens={}}} content_blocks={}",
            api_response.id,
            api_response.model,
            api_response.stop_reason,
            api_response.usage.total_input_tokens(),
            api_response.usage.output,
            api_response.content.len()
        );

        Ok(ChatOutcome::Success(Self::map_claude_response(
            api_response,
        )))
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream_claude(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
                // Forcing a specific tool is incompatible with extended thinking
                // on Claude (the API 400s), so drop thinking at the wire boundary
                // even when it was resurrected from the provider-configured
                // default.
                Ok(thinking) => thinking_for_forced_tool(thinking, request.tool_choice.as_ref()),
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };
            if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            let messages = Self::build_cached_vertex_claude_messages(&request);
            let tools = build_vertex_claude_tools(&request);
            let thinking = thinking_config
                .as_ref()
                .map(anthropic_data::ApiThinkingConfig::from_thinking_config);
            let output_config = thinking_config
                .as_ref()
                .and_then(|t| t.effort)
                .map(|effort| anthropic_data::ApiOutputConfig { effort });
            let system = Self::build_vertex_claude_system_prompt(&request.system);
            let tool_choice = request
                .tool_choice
                .as_ref()
                .map(anthropic_data::ApiToolChoice::from_tool_choice);

            let max_tokens = self.effective_max_tokens(&request);
            let api_request = anthropic_data::ApiMessagesRequest {
                model: None, // model is in the URL for Vertex
                max_tokens,
                system,
                messages: &messages,
                tools: tools.as_deref(),
                tool_choice,
                stream: true,
                thinking,
                output_config,
                anthropic_version: Some(VERTEX_ANTHROPIC_VERSION),
            };

            log::debug!(
                "Vertex AI (Claude) streaming request model={} max_tokens={}",
                self.model,
                max_tokens
            );

            if log::log_enabled!(log::Level::Debug) {
                match serde_json::to_string_pretty(&api_request) {
                    Ok(json) => log::debug!("Vertex AI (Claude) streaming request payload:\n{json}"),
                    Err(e) => log::debug!("Failed to serialize request for logging: {e}"),
                }
            }

            let url = format!("{}:streamRawPredict", self.base_url("anthropic"));

            let response = match self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .bearer_auth(&self.access_token)
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
                    kind: StreamErrorKind::RateLimited,
                });
                return;
            }

            if status.is_server_error() {
                let body = response.text().await.unwrap_or_default();
                log::error!("Vertex AI (Claude) server error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    kind: StreamErrorKind::ServerError,
                });
                return;
            }

            if status.is_client_error() {
                let body = response.text().await.unwrap_or_default();
                log::warn!("Vertex AI (Claude) client error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }

            // Process SSE stream using the Anthropic SSE parser
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;
            let mut cached_input_tokens: u32 = 0;
            let mut cache_creation_input_tokens: u32 = 0;
            let mut tool_ids: std::collections::HashMap<usize, String> =
                std::collections::HashMap::new();
            let mut received_message_stop = false;
            let mut pending_stop_reason: Option<agent_sdk_foundation::llm::StopReason> = None;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        // Include the cause so 401 detection / diagnostics survive.
                        yield Err(anyhow::anyhow!("stream error: {e}"));
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (terminated by a blank line)
                while let Some(event_block) = anthropic_data::take_next_sse_event(&mut buffer) {
                    if anthropic_data::is_message_stop_event(&event_block) {
                        received_message_stop = true;
                    }

                    if let Some(delta) = anthropic_data::parse_sse_event(
                        &event_block,
                        &mut input_tokens,
                        &mut output_tokens,
                        &mut cached_input_tokens,
                        &mut cache_creation_input_tokens,
                        &mut tool_ids,
                        &mut pending_stop_reason,
                    ) {
                        yield Ok(delta);
                    }
                    if anthropic_data::is_message_stop_event(&event_block) {
                        yield Ok(StreamDelta::Done {
                            stop_reason: pending_stop_reason.take(),
                        });
                    }
                }
            }

            // Process remaining buffer
            let remaining = buffer.trim();
            if !remaining.is_empty() {
                if anthropic_data::is_message_stop_event(remaining) {
                    received_message_stop = true;
                }

                if let Some(delta) = anthropic_data::parse_sse_event(
                    remaining,
                    &mut input_tokens,
                    &mut output_tokens,
                    &mut cached_input_tokens,
                    &mut cache_creation_input_tokens,
                    &mut tool_ids,
                    &mut pending_stop_reason,
                ) {
                    yield Ok(delta);
                }
                if anthropic_data::is_message_stop_event(remaining) {
                    yield Ok(StreamDelta::Done {
                        stop_reason: pending_stop_reason.take(),
                    });
                }
            }

            if !received_message_stop {
                log::warn!(
                    "Vertex AI (Claude) SSE stream ended without message_stop"
                );
                yield Ok(StreamDelta::Error {
                    message: "Stream ended unexpectedly without completion".to_string(),
                    kind: StreamErrorKind::ServerError,
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_provider() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "custom-model".to_string(),
        );

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_flash_factory() {
        let provider = VertexProvider::flash(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
        );

        assert_eq!(provider.model(), MODEL_GEMINI_3_FLASH);
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_pro_factory() {
        let provider = VertexProvider::pro(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
        );

        assert_eq!(provider.model(), MODEL_GEMINI_31_PRO);
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_provider_is_cloneable() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "test-model".to_string(),
        );
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }

    #[test]
    fn test_is_claude_model() {
        let claude_provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "us-central1".to_string(),
            "claude-sonnet-4-20250514".to_string(),
        );
        assert!(claude_provider.is_claude_model());

        let gemini_provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "us-central1".to_string(),
            "gemini-3-flash-preview".to_string(),
        );
        assert!(!gemini_provider.is_claude_model());
    }

    #[test]
    fn test_base_url_gemini() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "gemini-3-flash-preview".to_string(),
        );

        let url = provider.base_url("google");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3-flash-preview"
        );
    }

    #[test]
    fn test_base_url_claude() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "claude-sonnet-4-20250514".to_string(),
        );

        let url = provider.base_url("anthropic");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_base_url_with_different_region() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "other-project".to_string(),
            "europe-west4".to_string(),
            "gemini-3.1-pro-preview".to_string(),
        );

        let url = provider.base_url("google");
        assert!(url.starts_with("https://europe-west4-aiplatform.googleapis.com/"));
        assert!(url.contains("/projects/other-project/"));
        assert!(url.contains("/locations/europe-west4/"));
        assert!(url.ends_with("/models/gemini-3.1-pro-preview"));
    }

    #[test]
    fn test_base_url_global_region_has_no_prefix() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "global".to_string(),
            "gemini-3.1-pro-preview".to_string(),
        );

        let url = provider.base_url("google");
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-3.1-pro-preview"
        );
    }

    #[test]
    fn test_vertex_claude_46_rejects_budgeted_thinking() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_SONNET_46.to_string(),
        );

        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("ThinkingConfig::adaptive()"));
    }

    #[test]
    fn test_vertex_claude_opus_47_rejects_budgeted_thinking() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_OPUS_47.to_string(),
        );

        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("ThinkingConfig::adaptive()"));
    }

    #[test]
    fn test_vertex_claude_opus_48_rejects_budgeted_thinking() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_OPUS_48.to_string(),
        );

        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("ThinkingConfig::adaptive()"));
    }

    #[test]
    fn test_vertex_claude_fable_5_rejects_budgeted_thinking() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_FABLE_5.to_string(),
        );

        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("ThinkingConfig::adaptive()"));
    }

    #[test]
    fn test_model_constants() {
        assert_eq!(MODEL_GEMINI_3_FLASH, "gemini-3-flash-preview");
        assert_eq!(MODEL_GEMINI_31_PRO, "gemini-3.1-pro-preview");
        assert_eq!(MODEL_GEMINI_3_PRO, "gemini-3.0-pro");
    }

    fn request_with_max_tokens(max_tokens: u32, explicit: bool) -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![agent_sdk_foundation::llm::Message::user("hi")],
            tools: None,
            max_tokens,
            max_tokens_explicit: explicit,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        }
    }

    #[test]
    fn test_effective_max_tokens_honors_explicit_budget() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_SONNET_46.to_string(),
        );
        let request = request_with_max_tokens(1234, true);
        assert_eq!(provider.effective_max_tokens(&request), 1234);
    }

    #[test]
    fn test_effective_max_tokens_uses_clamped_default_when_implicit() {
        // An implicit budget for a Claude-on-Vertex model must fall back to the
        // clamped default (<= 32k), not the unclamped capability ceiling and
        // not ChatRequest::DEFAULT_MAX_TOKENS.
        let provider = VertexProvider::new(
            "token".to_string(),
            "project".to_string(),
            "global".to_string(),
            MODEL_SONNET_46.to_string(),
        );
        let request = request_with_max_tokens(4096, false);
        let effective = provider.effective_max_tokens(&request);
        assert_eq!(effective, provider.default_max_tokens());
        assert!(effective <= DEFAULT_SAFE_MAX_OUTPUT_TOKENS);
    }
}
