//! `OpenAI` Responses API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Responses API (`/v1/responses`). This provider supports the Codex model family
//! and other agentic `OpenAI` models that expose the Responses surface.

use crate::attachments::validate_request_attachments;
use crate::provider::LlmProvider;
use crate::streaming::{
    SseLineBuffer, StreamBox, StreamDelta, StreamErrorKind, reqwest_body_error_delta,
    reqwest_error_delta,
};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, ResponseFormat, StopReason,
    ThinkingConfig, ToolChoice, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::openai_reasoning::{
    OpenAIAllowedToolsMode, OpenAIPromptCacheMode, OpenAIPromptCacheTtl, OpenAIReasoningConfig,
    OpenAIReasoningContext, OpenAIReasoningEffort, OpenAIReasoningMode, OpenAIReasoningSummary,
    OpenAITextVerbosity, OpenAIToolChoice, is_gpt56_model, legacy_reasoning_effort,
    validate_reasoning_config, validate_tool_choice,
};
use super::openai_schema::normalize_strict_schema;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const ENCRYPTED_REASONING_INCLUDE: &[&str] = &["reasoning.encrypted_content"];
const OPENAI_RESPONSES_STATE_PROVIDER: &str = "openai-responses";
const OPENAI_MESSAGE_ITEM_TYPE: &str = "message";

/// Build an HTTP client with connect/keepalive timeouts matching the sibling
/// providers (`anthropic`, `vertex`). A bare `reqwest::Client::new()` has no
/// connect timeout, so a black-holed connect would wedge `chat`/`chat_stream`
/// forever.
fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

// GPT-5.6 series
pub const MODEL_GPT56: &str = "gpt-5.6";
pub const MODEL_GPT56_SOL: &str = "gpt-5.6-sol";
pub const MODEL_GPT56_TERRA: &str = "gpt-5.6-terra";
pub const MODEL_GPT56_LUNA: &str = "gpt-5.6-luna";

// GPT-5.3-Codex (latest Codex model)
pub const MODEL_GPT53_CODEX: &str = "gpt-5.3-codex";

// GPT-5.2-Codex (legacy Responses-first codex model)
pub const MODEL_GPT52_CODEX: &str = "gpt-5.2-codex";

/// Reasoning effort level for the model.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
    /// Extra-high reasoning for complex problems
    #[serde(rename = "xhigh")]
    XHigh,
    /// GPT-5.6 maximum reasoning effort
    Max,
}

/// `OpenAI` Responses API provider.
///
/// This provider uses the `/v1/responses` endpoint for `OpenAI` models that expose
/// agentic workflows over the Responses API.
#[derive(Clone)]
pub struct OpenAIResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    thinking: Option<ThinkingConfig>,
    reasoning: Option<OpenAIReasoningConfig>,
    /// Extra headers applied to every request (e.g. for gateway / BYOK auth).
    extra_headers: Vec<(String, String)>,
}

impl OpenAIResponsesProvider {
    /// Create a new `OpenAI` Responses API provider.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: build_http_client(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
            thinking: None,
            reasoning: None,
            extra_headers: Vec::new(),
        }
    }

    /// Create a provider with a custom base URL.
    #[must_use]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: build_http_client(),
            api_key,
            model,
            base_url,
            thinking: None,
            reasoning: None,
            extra_headers: Vec::new(),
        }
    }

    /// Add extra HTTP headers applied to every request.
    ///
    /// Used by [`OpenAIProvider`](super::openai::OpenAIProvider)'s transparent
    /// Responses-API reroute to forward its BYOK / gateway auth headers (e.g.
    /// `cf-aig-authorization`) so a rerouted request authenticates correctly.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Reuse an existing pooled `reqwest::Client` instead of building a fresh one.
    ///
    /// `reqwest::Client` is an `Arc` handle (cheap to clone) backed by a
    /// connection pool; reusing it across the reroute preserves keep-alive so a
    /// rerouted agent loop does not pay a new TCP+TLS handshake every turn.
    #[must_use]
    pub(crate) fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Apply auth + extra headers to a request builder. Skips `Authorization`
    /// when `api_key` is empty (BYOK gateway mode — auth is carried by
    /// `extra_headers`).
    fn apply_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let builder = if self.api_key.is_empty() {
            builder
        } else {
            builder.header("Authorization", format!("Bearer {}", self.api_key))
        };
        self.extra_headers
            .iter()
            .fold(builder, |b, (k, v)| b.header(k.as_str(), v.as_str()))
    }

    /// Create a provider using the GPT-5.6 alias, which routes to GPT-5.6 Sol.
    #[must_use]
    pub fn gpt56(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT56.to_owned())
    }

    /// Create a provider using GPT-5.6 Sol (frontier reasoning and coding).
    #[must_use]
    pub fn gpt56_sol(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT56_SOL.to_owned())
    }

    /// Create a provider using GPT-5.6 Terra (balanced intelligence and cost).
    #[must_use]
    pub fn gpt56_terra(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT56_TERRA.to_owned())
    }

    /// Create a provider using GPT-5.6 Luna (cost-sensitive, high-volume work).
    #[must_use]
    pub fn gpt56_luna(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT56_LUNA.to_owned())
    }

    /// Create a provider using GPT-5.3-Codex (latest codex model).
    #[must_use]
    pub fn gpt53_codex(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT53_CODEX.to_owned())
    }

    /// Create a provider using the latest Codex model.
    #[must_use]
    pub fn codex(api_key: String) -> Self {
        Self::gpt53_codex(api_key)
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self.reasoning = None;
        self
    }

    /// Set exact `OpenAI` reasoning and related response controls.
    ///
    /// Calling this after [`with_thinking`](Self::with_thinking) replaces the
    /// legacy provider-owned thinking configuration.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: OpenAIReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self.thinking = None;
        self
    }

    /// Set the reasoning effort level.
    #[must_use]
    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        let effort = match effort {
            ReasoningEffort::Low => OpenAIReasoningEffort::Low,
            ReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
            ReasoningEffort::High => OpenAIReasoningEffort::High,
            ReasoningEffort::XHigh => OpenAIReasoningEffort::XHigh,
            ReasoningEffort::Max => OpenAIReasoningEffort::Max,
        };
        self.with_reasoning(OpenAIReasoningConfig::new().with_effort(effort))
    }

    fn effective_max_tokens(&self, request: &ChatRequest) -> u32 {
        if request.max_tokens_explicit {
            request.max_tokens
        } else {
            self.default_max_tokens()
        }
    }

    fn resolve_openai_reasoning(
        &self,
        request_thinking: Option<&ThinkingConfig>,
    ) -> Result<Option<OpenAIReasoningConfig>> {
        let legacy = if request_thinking.is_some() || self.reasoning.is_none() {
            self.resolve_thinking_config(request_thinking)?
        } else {
            None
        };

        let config = match (self.reasoning.clone(), legacy.as_ref()) {
            (Some(config), Some(thinking)) => {
                Some(config.with_optional_effort(legacy_reasoning_effort(thinking)))
            }
            (Some(config), None) => Some(config),
            (None, Some(thinking)) => Some(
                OpenAIReasoningConfig::new()
                    .with_optional_effort(legacy_reasoning_effort(thinking)),
            ),
            (None, None) => None,
        };

        if let Some(config) = &config {
            validate_reasoning_config(&self.model, config)?;
        }
        Ok(config)
    }

    async fn send_responses_request(
        &self,
        api_request: &ApiResponsesRequest<'_>,
    ) -> Result<(StatusCode, Vec<u8>, Option<std::time::Duration>)> {
        let builder = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Content-Type", "application/json");
        let response = self
            .apply_headers(builder)
            .json(api_request)
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("request failed: {error}"))?;
        let status = response.status();
        let retry_after = (status == StatusCode::TOO_MANY_REQUESTS)
            .then(|| crate::http::retry_after_from_headers(response.headers()))
            .flatten();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read response body: {error}"))?
            .to_vec();
        Ok((status, bytes, retry_after))
    }
}

#[async_trait]
impl LlmProvider for OpenAIResponsesProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let reasoning_config = match self.resolve_openai_reasoning(request.thinking.as_ref()) {
            Ok(reasoning) => reasoning,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if let Err(error) = validate_responses_tool_choice(
            request.tool_choice.as_ref(),
            reasoning_config.as_ref(),
            request.tools.as_deref(),
        ) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if let Err(error) = validate_response_format(request.response_format.as_ref()) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let prompt_cache =
            match resolve_prompt_cache_plan(&self.model, &request, reasoning_config.as_ref()) {
                Ok(plan) => plan,
                Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
            };
        let reasoning = build_api_reasoning(reasoning_config.as_ref());
        let store = reasoning_config
            .as_ref()
            .and_then(OpenAIReasoningConfig::store)
            .unwrap_or(false);
        let include_encrypted_reasoning = !store
            && (reasoning.is_some()
                || self
                    .capabilities()
                    .is_some_and(|capabilities| capabilities.supports_thinking));
        let input = build_api_input(&request, prompt_cache.explicit_breakpoints);
        let text = ApiResponseText::from_options(
            request.response_format.as_ref(),
            reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::verbosity),
        );
        let prompt_cache_options = prompt_cache.options;
        let max_tokens = self.effective_max_tokens(&request);
        let tool_choice =
            resolve_api_tool_choice(request.tool_choice.as_ref(), reasoning_config.as_ref());
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());
        let parallel_tool_calls = reasoning_config
            .as_ref()
            .and_then(OpenAIReasoningConfig::parallel_tool_calls)
            .or_else(|| {
                tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty())
                    .then_some(true)
            });

        let api_request = ApiResponsesRequest {
            model: &self.model,
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens: Some(max_tokens),
            reasoning,
            parallel_tool_calls,
            text,
            tool_choice,
            prompt_cache_options,
            store,
            include: include_encrypted_reasoning.then_some(ENCRYPTED_REASONING_INCLUDE),
            prompt_cache_key: request.session_id.as_deref(),
            safety_identifier: reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::safety_identifier),
        };

        log::debug!(
            "OpenAI Responses API request model={} max_tokens={}",
            self.model,
            max_tokens
        );

        let (status, bytes, retry_after) = self.send_responses_request(&api_request).await?;

        log::debug!(
            "OpenAI Responses API response status={} body_len={}",
            status,
            bytes.len()
        );

        if let Some(outcome) = classify_responses_status(status, &bytes, retry_after) {
            return Ok(outcome);
        }

        let api_response: ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        Ok(build_responses_outcome(api_response))
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        let served_route = self.route().to_owned();
        Box::pin(async_stream::stream! {
            let reasoning_config = match self.resolve_openai_reasoning(request.thinking.as_ref()) {
                Ok(reasoning) => reasoning,
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
            if let Err(error) = validate_responses_tool_choice(
                request.tool_choice.as_ref(),
                reasoning_config.as_ref(),
                request.tools.as_deref(),
            ) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            if let Err(error) = validate_response_format(request.response_format.as_ref()) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            let prompt_cache = match resolve_prompt_cache_plan(
                &self.model,
                &request,
                reasoning_config.as_ref(),
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };
            let reasoning = build_api_reasoning(reasoning_config.as_ref());
            let store = reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::store)
                .unwrap_or(false);
            let include_encrypted_reasoning = !store
                && (reasoning.is_some()
                    || self
                        .capabilities()
                        .is_some_and(|capabilities| capabilities.supports_thinking));
            let input = build_api_input(&request, prompt_cache.explicit_breakpoints);
            let text = ApiResponseText::from_options(
                request.response_format.as_ref(),
                reasoning_config
                    .as_ref()
                    .and_then(OpenAIReasoningConfig::verbosity),
            );
            let prompt_cache_options = prompt_cache.options;
            let max_tokens = self.effective_max_tokens(&request);
            let tool_choice = resolve_api_tool_choice(request.tool_choice.as_ref(), reasoning_config.as_ref());
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| ts.into_iter().map(convert_tool).collect());
            let parallel_tool_calls = reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::parallel_tool_calls)
                .or_else(|| tools.as_ref().is_some_and(|tools| !tools.is_empty()).then_some(true));

            let api_request = ApiResponsesRequestStreaming {
                model: &self.model,
                input: &input,
                tools: tools.as_deref(),
                max_output_tokens: Some(max_tokens),
                reasoning,
                parallel_tool_calls,
                text,
                tool_choice,
                prompt_cache_options,
                store,
                include: include_encrypted_reasoning
                    .then(|| ENCRYPTED_REASONING_INCLUDE.iter().map(|value| (*value).to_owned()).collect()),
                prompt_cache_key: request.session_id.clone(),
                safety_identifier: reasoning_config
                    .as_ref()
                    .and_then(OpenAIReasoningConfig::safety_identifier)
                    .map(ToOwned::to_owned),
                stream: true,
            };

            log::debug!("OpenAI Responses API streaming request model={} max_tokens={}", self.model, max_tokens);

            let stream_builder = self.client
                .post(format!("{}/responses", self.base_url))
                .header("Content-Type", "application/json");
            let response = match self
                .apply_headers(stream_builder)
                .json(&api_request)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    yield Ok(reqwest_error_delta("request failed", &error));
                    return;
                }
            };

            let status = response.status();

            if !status.is_success() {
                // Headers are read before the body: `text()` consumes the response.
                let header_hint = crate::http::retry_after_from_headers(response.headers());
                let body = response.text().await.unwrap_or_default();
                let kind = if status == StatusCode::TOO_MANY_REQUESTS {
                    StreamErrorKind::RateLimited(
                        header_hint.or_else(|| crate::retry_hints::openai_retry_delay(&body)),
                    )
                } else if status.is_server_error() {
                    StreamErrorKind::ServerError
                } else {
                    StreamErrorKind::InvalidRequest
                };
                log::warn!("OpenAI Responses error status={status} body={body}");
                yield Ok(StreamDelta::Error { message: body, kind });
                return;
            }

            let mut sse = SseLineBuffer::new();
            let mut stream = response.bytes_stream();
            let mut tool_calls: std::collections::HashMap<String, ToolCallAccumulator> =
                std::collections::HashMap::new();
            let mut message_state_markers: std::collections::HashMap<usize, serde_json::Value> =
                std::collections::HashMap::new();
            let mut refused = false;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Ok(reqwest_body_error_delta("stream error", &error));
                        return;
                    }
                };
                sse.extend(&chunk);

                while let Some(line) = sse.next_line() {
                    let line = line.trim();
                    if line.is_empty() { continue; }

                    let Some(data) = line.strip_prefix("data: ") else {
                        log::trace!("Responses SSE non-data line bytes={}", line.len());
                        continue;
                    };

                    if data == "[DONE]" {
                        yield Ok(StreamDelta::Error {
                            message: "OpenAI Responses stream ended before a semantic terminal event"
                                .to_owned(),
                            kind: StreamErrorKind::ServerError,
                        });
                        return;
                    }

                    let event = match serde_json::from_str::<ApiStreamEvent>(data) {
                        Ok(event) => event,
                        Err(error) => {
                            log::warn!("Failed to parse Responses SSE event: {error}");
                            yield Ok(StreamDelta::Error {
                                message: format!("invalid OpenAI Responses stream event: {error}"),
                                kind: StreamErrorKind::ServerError,
                            });
                            return;
                        }
                    };
                    log::trace!(
                        "Responses SSE event type={} bytes={}",
                        event.r#type,
                        data.len()
                    );

                    match event.r#type.as_str() {
                            // ── Content deltas ──────────────────────────
                            "response.output_text.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::TextDelta {
                                        delta,
                                        block_index: output_block_index(event.output_index),
                                    });
                                }
                            }
                            "response.refusal.delta" => {
                                refused = true;
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::TextDelta {
                                        delta,
                                        block_index: output_block_index(event.output_index),
                                    });
                                }
                            }
                            "response.output_item.added" => {
                                if let Some(item) = &event.item
                                    && let Some((block_index, marker)) =
                                        message_state_from_output_item(item, event.output_index)
                                    && message_state_markers.get(&block_index) != Some(&marker)
                                {
                                    message_state_markers.insert(block_index, marker.clone());
                                    yield Ok(StreamDelta::OpaqueReasoning {
                                        provider: OPENAI_RESPONSES_STATE_PROVIDER.to_owned(),
                                        data: marker,
                                        block_index,
                                    });
                                }
                                // Register function_call items so we know
                                // the call_id and name before deltas arrive.
                                if let Some(item) = &event.item
                                    && item.get("type").and_then(serde_json::Value::as_str)
                                        == Some("function_call")
                                    && let (Some(item_id), Some(call_id), Some(name)) =
                                        (
                                            json_string(item, "id"),
                                            json_string(item, "call_id"),
                                            json_string(item, "name"),
                                        )
                                {
                                    let order = tool_calls.len();
                                    tool_calls
                                        .entry(item_id.to_owned())
                                        .or_insert_with(|| ToolCallAccumulator {
                                            id: call_id.to_owned(),
                                            name: name.to_owned(),
                                            arguments: String::new(),
                                            order,
                                            block_index: output_block_index(event.output_index),
                                        });
                                }
                            }
                            "response.output_item.done" => {
                                if let Some(item) = event.item {
                                    if let Some((block_index, marker)) =
                                        message_state_from_output_item(&item, event.output_index)
                                        && message_state_markers.get(&block_index) != Some(&marker)
                                    {
                                        message_state_markers.insert(block_index, marker.clone());
                                        yield Ok(StreamDelta::OpaqueReasoning {
                                            provider: OPENAI_RESPONSES_STATE_PROVIDER.to_owned(),
                                            data: marker,
                                            block_index,
                                        });
                                    }
                                    match item.get("type").and_then(serde_json::Value::as_str) {
                                        Some("reasoning") => {
                                            yield Ok(StreamDelta::OpaqueReasoning {
                                                provider: OPENAI_RESPONSES_STATE_PROVIDER.to_owned(),
                                                data: item,
                                                block_index: output_block_index(event.output_index),
                                            });
                                        }
                                        Some("function_call") => {
                                            if let (Some(item_id), Some(call_id), Some(name)) = (
                                                json_string(&item, "id"),
                                                json_string(&item, "call_id"),
                                                json_string(&item, "name"),
                                            ) {
                                                let order = tool_calls.len();
                                                let accumulator = tool_calls
                                                    .entry(item_id.to_owned())
                                                    .or_insert_with(|| ToolCallAccumulator {
                                                        id: call_id.to_owned(),
                                                        name: name.to_owned(),
                                                        arguments: String::new(),
                                                        order,
                                                        block_index: output_block_index(
                                                            event.output_index,
                                                        ),
                                                    });
                                                if let Some(arguments) =
                                                    json_string(&item, "arguments")
                                                {
                                                    arguments.clone_into(&mut accumulator.arguments);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                if let (Some(item_id), Some(delta)) =
                                    (event.resolve_item_id().map(str::to_owned), event.delta)
                                {
                                    let order = tool_calls.len();
                                    let acc =
                                        tool_calls.entry(item_id.clone()).or_insert_with(|| {
                                            ToolCallAccumulator {
                                                id: item_id,
                                                name: event.name.unwrap_or_default(),
                                                arguments: String::new(),
                                                order,
                                                block_index: output_block_index(event.output_index),
                                            }
                                        });
                                    acc.arguments.push_str(&delta);
                                }
                            }
                            // ── Reasoning (thinking) deltas ─────────────
                            "response.reasoning.delta"
                            | "response.reasoning_summary_text.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        delta,
                                        block_index: reasoning_summary_block_index(
                                            event.output_index,
                                        ),
                                    });
                                }
                            }
                            // ── Completion / usage ──────────────────────
                            "response.completed" => {
                                let response = event.response;
                                if let Some(usage) = response.and_then(|response| response.usage) {
                                    yield Ok(StreamDelta::Usage(map_usage(Some(usage))));
                                }
                                let stop_reason = if refused {
                                    StopReason::Refusal
                                } else if !tool_calls.is_empty() {
                                    StopReason::ToolUse
                                } else {
                                    StopReason::EndTurn
                                };
                                if stop_reason == StopReason::ToolUse {
                                    for delta in flush_responses_tool_calls(&tool_calls) {
                                        yield Ok(delta);
                                    }
                                }
                                yield Ok(StreamDelta::Done {
                                    stop_reason: Some(stop_reason),
                                    served_route: Some(served_route.clone()),
                                });
                                return;
                            }
                            "response.incomplete" => {
                                let response = event.response;
                                let stop_reason = response
                                    .as_ref()
                                    .and_then(|response| response.incomplete_details.as_ref())
                                    .and_then(|details| details.reason.as_deref())
                                    .map_or(StopReason::Unknown, incomplete_stop_reason);
                                if let Some(usage) = response.and_then(|response| response.usage) {
                                    yield Ok(StreamDelta::Usage(map_usage(Some(usage))));
                                }
                                yield Ok(StreamDelta::Done {
                                    stop_reason: Some(stop_reason),
                                    served_route: Some(served_route.clone()),
                                });
                                return;
                            }
                            // ── Error ───────────────────────────────────
                            "error" | "response.failed" => {
                                // A failed response still reports the tokens it
                                // burned, so the usage is emitted before the error
                                // that ends the stream — the caller's accumulator
                                // only ever sees deltas that were yielded.
                                let (failed_usage, response_error) = event
                                    .response
                                    .map_or((None, None), |response| {
                                        (response.usage, response.error)
                                    });
                                let failed_usage =
                                    failed_usage.map(|usage| map_usage(Some(usage)));
                                let (error_code, error_message) = response_error
                                    .map_or((None, None), |error| (error.code, error.message));
                                let code = error_code.or(event.code);
                                let message = error_message
                                    .or(event.message)
                                    .unwrap_or_else(|| data.to_owned());
                                let kind = responses_stream_error_kind(
                                    &event.r#type,
                                    code.as_deref(),
                                    &message,
                                    data,
                                );
                                if let Some(usage) = failed_usage {
                                    yield Ok(StreamDelta::Usage(usage));
                                }
                                yield Ok(StreamDelta::Error {
                                    message,
                                    kind,
                                });
                                return;
                            }
                            // ── Lifecycle events (no content) ───────────
                            "response.created"
                            | "response.in_progress"
                            | "response.content_part.added"
                            | "response.content_part.done"
                            | "response.output_text.done"
                            | "response.refusal.done"
                            | "response.function_call_arguments.done"
                            | "response.reasoning.done"
                            | "response.reasoning_summary_text.done" => {}
                            // ── Unknown ─────────────────────────────────
                            other => {
                                log::debug!("Unhandled Responses SSE event type: {other}");
                            }
                        }
                }
            }

            yield Ok(StreamDelta::Error {
                message: "OpenAI Responses stream ended without completed, incomplete, or failed"
                    .to_owned(),
                kind: StreamErrorKind::ServerError,
            });
        })
    }

    async fn probe_connectivity(&self) -> bool {
        crate::provider::probe_http_reachability(&self.client, &self.base_url).await
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "openai-responses"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }
}

// ============================================================================
// Input building
// ============================================================================

fn build_api_input(request: &ChatRequest, explicit_cache_breakpoints: usize) -> Vec<ApiInputItem> {
    let mut items = Vec::new();

    if !request.system.is_empty() {
        items.push(ApiInputItem::Message(ApiMessage {
            role: ApiRole::System,
            content: ApiMessageContent::Parts(vec![ApiInputContent::input_text(
                request.system.clone(),
            )]),
            phase: None,
        }));
    }

    for msg in &request.messages {
        let role = api_role(msg.role);
        match &msg.content {
            Content::Text(text) => {
                items.push(ApiInputItem::Message(ApiMessage {
                    role,
                    content: ApiMessageContent::Parts(vec![ApiInputContent::text(
                        role,
                        text.clone(),
                    )]),
                    phase: api_message_phase(role, false),
                }));
            }
            Content::Blocks(blocks) => append_block_input(&mut items, role, blocks),
        }
    }

    apply_explicit_cache_breakpoints(&mut items, explicit_cache_breakpoints);
    items
}

fn append_block_input(items: &mut Vec<ApiInputItem>, role: ApiRole, blocks: &[ContentBlock]) {
    let mut content_parts = Vec::new();
    let mut phase = api_message_phase(
        role,
        blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. })),
    );

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                content_parts.push(ApiInputContent::text(role, text.clone()));
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::OpaqueReasoning { provider, data }
                if matches!(role, ApiRole::Assistant)
                    && is_message_state_marker(provider, data) =>
            {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                phase = data
                    .get("phase")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned);
            }
            ContentBlock::OpaqueReasoning { provider, data }
                if provider == OPENAI_RESPONSES_STATE_PROVIDER
                    && data.get("type").and_then(serde_json::Value::as_str)
                        == Some("reasoning") =>
            {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                items.push(ApiInputItem::Opaque(data.clone()));
            }
            ContentBlock::OpaqueReasoning { provider, .. } => {
                log::warn!(
                    "Ignoring opaque reasoning owned by provider={provider} in OpenAI Responses input"
                );
            }
            ContentBlock::Image { source } => content_parts.push(ApiInputContent::Image {
                image_url: format!("data:{};base64,{}", source.media_type, source.data),
                prompt_cache_breakpoint: None,
            }),
            ContentBlock::Document { source } => content_parts.push(ApiInputContent::File {
                filename: suggested_filename(&source.media_type),
                file_data: format!("data:{};base64,{}", source.media_type, source.data),
                prompt_cache_breakpoint: None,
            }),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                items.push(ApiInputItem::FunctionCall(ApiFunctionCall::new(
                    id.clone(),
                    name.clone(),
                    input.to_string(),
                )));
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                items.push(ApiInputItem::FunctionCallOutput(
                    ApiFunctionCallOutput::new(tool_use_id.clone(), content.clone()),
                ));
            }
            _ => log::warn!("Skipping unrecognized OpenAI Responses content block"),
        }
    }

    flush_message_parts(items, role, phase, &mut content_parts);
}

const fn api_role(role: agent_sdk_foundation::llm::Role) -> ApiRole {
    match role {
        agent_sdk_foundation::llm::Role::User => ApiRole::User,
        agent_sdk_foundation::llm::Role::Assistant => ApiRole::Assistant,
    }
}

fn message_state_marker(role: &str, phase: Option<&str>) -> serde_json::Value {
    let mut marker = serde_json::Map::new();
    marker.insert(
        "type".to_owned(),
        serde_json::Value::String(OPENAI_MESSAGE_ITEM_TYPE.to_owned()),
    );
    marker.insert(
        "role".to_owned(),
        serde_json::Value::String(role.to_owned()),
    );
    if let Some(phase) = phase {
        marker.insert(
            "phase".to_owned(),
            serde_json::Value::String(phase.to_owned()),
        );
    }
    serde_json::Value::Object(marker)
}

fn is_message_state_marker(provider: &str, data: &serde_json::Value) -> bool {
    provider == OPENAI_RESPONSES_STATE_PROVIDER
        && data.get("type").and_then(serde_json::Value::as_str) == Some(OPENAI_MESSAGE_ITEM_TYPE)
        && data.get("content").is_none()
}

fn api_message_phase(role: ApiRole, has_tool_use: bool) -> Option<String> {
    match (role, has_tool_use) {
        (ApiRole::Assistant, true) => Some("commentary".to_owned()),
        (ApiRole::Assistant, false) => Some("final_answer".to_owned()),
        (ApiRole::System | ApiRole::User, _) => None,
    }
}

fn flush_message_parts(
    items: &mut Vec<ApiInputItem>,
    role: ApiRole,
    phase: Option<String>,
    content_parts: &mut Vec<ApiInputContent>,
) {
    if content_parts.is_empty() {
        return;
    }

    items.push(ApiInputItem::Message(ApiMessage {
        role,
        content: ApiMessageContent::Parts(std::mem::take(content_parts)),
        phase,
    }));
}

fn apply_explicit_cache_breakpoints(items: &mut [ApiInputItem], max_breakpoints: usize) {
    if max_breakpoints == 0 {
        return;
    }

    let mut candidates: Vec<(usize, usize)> = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| match item {
            ApiInputItem::Message(ApiMessage {
                content: ApiMessageContent::Parts(parts),
                ..
            }) => parts
                .iter()
                .rposition(ApiInputContent::supports_explicit_cache_breakpoint)
                .map(|part_index| (index, part_index)),
            _ => None,
        })
        .collect();
    let keep_from = candidates.len().saturating_sub(max_breakpoints.min(4));

    for (item_index, part_index) in candidates.drain(keep_from..) {
        if let ApiInputItem::Message(ApiMessage {
            content: ApiMessageContent::Parts(parts),
            ..
        }) = &mut items[item_index]
            && let Some(part) = parts.get_mut(part_index)
        {
            part.set_explicit_cache_breakpoint();
        }
    }
}

fn convert_tool(tool: agent_sdk_foundation::llm::Tool) -> ApiTool {
    let mut schema = tool.input_schema;

    // Strict mode requires additionalProperties: false on all objects and
    // every property in required. This is incompatible with free-form object
    // schemas (objects with no defined properties). Detect and skip strict
    // for those tools.
    let use_strict = if normalize_strict_schema(&mut schema) {
        Some(true)
    } else {
        log::debug!(
            "Tool '{}' has free-form object schema — disabling strict mode",
            tool.name
        );
        None
    };

    ApiTool {
        r#type: "function".to_owned(),
        name: tool.name,
        description: Some(tool.description),
        parameters: Some(schema),
        strict: use_strict,
    }
}

fn validate_response_format(response_format: Option<&ResponseFormat>) -> Result<()> {
    let Some(response_format) = response_format.filter(|format| format.strict) else {
        return Ok(());
    };

    let mut schema = response_format.schema.clone();
    if !normalize_strict_schema(&mut schema) {
        anyhow::bail!(
            "OpenAI strict structured output `{}` contains a free-form object schema",
            response_format.name
        );
    }
    Ok(())
}

fn suggested_filename(media_type: &str) -> String {
    match media_type {
        "application/pdf" => "attachment.pdf".to_string(),
        "image/png" => "image.png".to_string(),
        "image/jpeg" => "image.jpg".to_string(),
        "image/gif" => "image.gif".to_string(),
        "image/webp" => "image.webp".to_string(),
        _ => "attachment.bin".to_string(),
    }
}

fn build_content_blocks(output: &[ApiOutputItem]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for item in output {
        match item {
            ApiOutputItem::Reasoning { data } => {
                let mut raw = data.clone();
                raw.insert(
                    "type".to_owned(),
                    serde_json::Value::String("reasoning".to_owned()),
                );
                blocks.push(ContentBlock::OpaqueReasoning {
                    provider: "openai-responses".to_owned(),
                    data: serde_json::Value::Object(raw),
                });

                if let Some(summaries) = data.get("summary").and_then(serde_json::Value::as_array) {
                    for summary in summaries {
                        if let Some(thinking) = summary
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .filter(|text| !text.is_empty())
                        {
                            blocks.push(ContentBlock::Thinking {
                                thinking: thinking.to_owned(),
                                signature: None,
                            });
                        }
                    }
                }
            }
            ApiOutputItem::Message {
                role,
                phase,
                content,
            } => {
                blocks.push(ContentBlock::OpaqueReasoning {
                    provider: OPENAI_RESPONSES_STATE_PROVIDER.to_owned(),
                    data: message_state_marker(role, phase.as_deref()),
                });
                for c in content {
                    match c {
                        ApiOutputContent::Text { text }
                        | ApiOutputContent::Refusal { refusal: text }
                            if !text.is_empty() =>
                        {
                            blocks.push(ContentBlock::Text { text: text.clone() });
                        }
                        ApiOutputContent::Text { .. }
                        | ApiOutputContent::Refusal { .. }
                        | ApiOutputContent::Unknown => {}
                    }
                }
            }
            ApiOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input =
                    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}));
                blocks.push(ContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                    thought_signature: None,
                });
            }
            ApiOutputItem::Unknown => {
                // Skip unknown output types
            }
        }
    }

    blocks
}

fn output_contains_refusal(output: &[ApiOutputItem]) -> bool {
    output.iter().any(|item| {
        matches!(
            item,
            ApiOutputItem::Message { content, .. }
                if content
                    .iter()
                    .any(|content| matches!(content, ApiOutputContent::Refusal { .. }))
        )
    })
}

/// Classify an in-band Responses SSE error event.
///
/// A stream that opened with HTTP 200 can still fail mid-flight, and a
/// rate limit reported that way carries no HTTP status: the machine-readable
/// `code` is the only reliable signal. The message is prose and is never used
/// to decide the category — only to recover the retry delay `OpenAI` states in
/// it, since these events carry no `Retry-After` header.
fn responses_stream_error_kind(
    event_type: &str,
    code: Option<&str>,
    message: &str,
    raw: &str,
) -> StreamErrorKind {
    if matches!(code, Some("rate_limit_exceeded" | "rate_limit_error")) {
        log::warn!("Responses API rate limited (recoverable): {message}");
        return StreamErrorKind::RateLimited(crate::retry_hints::openai_retry_delay(message));
    }
    if is_context_window_rejection(code, message) {
        // Fatal request-shape error, not transient: the worker's overflow
        // compaction must fire instead of retrying the same oversized
        // payload (mirrors `openai_codex_responses::is_context_window_rejection`).
        log::warn!("Responses API context window exceeded (fatal): {message}");
        return StreamErrorKind::InvalidRequest;
    }
    if event_type == "response.failed"
        || code == Some("server_error")
        || raw.contains("server_error")
    {
        log::warn!("Responses API server error (recoverable): {message}");
        return StreamErrorKind::ServerError;
    }
    log::error!("Responses API error event: {message}");
    StreamErrorKind::InvalidRequest
}

/// True when a failure code/message rejects the prompt for exceeding the
/// model's context window. The structured `context_length_exceeded` code is
/// the primary signal; the prose shapes cover wrapped or proxied failures
/// that drop the code. Mirrors
/// `openai_codex_responses::is_context_window_rejection`.
fn is_context_window_rejection(code: Option<&str>, message: &str) -> bool {
    if matches!(code, Some("context_length_exceeded")) {
        return true;
    }
    let lower = message.to_lowercase();
    lower.contains("exceeds the context window")
        || lower.contains("maximum context length")
        || lower.contains("context_length_exceeded")
}

/// Classify a non-success HTTP status into an early [`ChatOutcome`].
///
/// Returns `None` when the status is a success and the body should instead be
/// parsed as an [`ApiResponse`].
fn classify_responses_status(
    status: StatusCode,
    bytes: &[u8],
    retry_after: Option<std::time::Duration>,
) -> Option<ChatOutcome> {
    if status == StatusCode::TOO_MANY_REQUESTS {
        let retry_after = retry_after
            .or_else(|| crate::retry_hints::openai_retry_delay(&String::from_utf8_lossy(bytes)));
        return Some(ChatOutcome::RateLimited(retry_after));
    }
    if status.is_server_error() {
        let body = String::from_utf8_lossy(bytes);
        log::error!("OpenAI Responses server error status={status} body={body}");
        return Some(ChatOutcome::ServerError(body.into_owned()));
    }
    if status.is_client_error() {
        let body = String::from_utf8_lossy(bytes);
        log::warn!("OpenAI Responses client error status={status} body={body}");
        return Some(ChatOutcome::InvalidRequest(body.into_owned()));
    }
    None
}

/// Map a parsed Responses API body into a [`ChatOutcome`].
///
/// The Responses API reports generation failures as HTTP 200 with
/// `status=failed` plus an error object. That is surfaced as a server error
/// instead of a successful turn with empty content (mirrors the streaming
/// `response.failed` handling).
fn build_responses_outcome(api_response: ApiResponse) -> ChatOutcome {
    if matches!(api_response.status, Some(ApiStatus::Failed)) {
        let message = api_response
            .error
            .and_then(|error| error.message)
            .unwrap_or_else(|| "OpenAI Responses API reported status=failed".to_owned());
        log::error!("OpenAI Responses generation failed: {message}");
        return ChatOutcome::ServerError(message);
    }

    let refused = output_contains_refusal(&api_response.output);
    let mut content = build_content_blocks(&api_response.output);

    // Determine stop reason based on output content
    let has_tool_calls = content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

    let stop_reason = if matches!(api_response.status, Some(ApiStatus::Incomplete)) {
        Some(
            api_response
                .incomplete_details
                .as_ref()
                .and_then(|details| details.reason.as_deref())
                .map_or(StopReason::Unknown, incomplete_stop_reason),
        )
    } else if refused {
        Some(StopReason::Refusal)
    } else if has_tool_calls {
        Some(StopReason::ToolUse)
    } else {
        api_response.status.map(|s| match s {
            ApiStatus::Completed => StopReason::EndTurn,
            // Unreachable: incomplete and failed are handled above, but map defensively.
            ApiStatus::Incomplete | ApiStatus::Failed => StopReason::Unknown,
        })
    };

    if stop_reason != Some(StopReason::ToolUse) {
        content.retain(|block| !matches!(block, ContentBlock::ToolUse { .. }));
    }

    ChatOutcome::Success(ChatResponse {
        id: api_response.id,
        content,
        model: api_response.model,
        stop_reason,
        usage: map_usage(api_response.usage),
    })
}

/// Convert the Responses API usage object into the SDK [`Usage`] shape.
fn map_usage(usage: Option<ApiUsage>) -> Usage {
    usage.map_or(
        Usage {
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        },
        |u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cached_input_tokens: u
                .input_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            cache_creation_input_tokens: u
                .input_tokens_details
                .as_ref()
                .map_or(0, |details| details.cache_write_tokens),
        },
    )
}

fn build_api_reasoning(config: Option<&OpenAIReasoningConfig>) -> Option<ApiReasoning> {
    let config = config?;
    let reasoning = ApiReasoning {
        effort: config.effort(),
        mode: config.mode(),
        context: config.context(),
        summary: config.summary(),
    };
    reasoning.has_fields().then_some(reasoning)
}

// ============================================================================
// Streaming helpers
// ============================================================================

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
    /// Registration order, used to assign deterministic, distinct block indices
    /// when flushing (`HashMap` iteration order is otherwise nondeterministic).
    order: usize,
    /// Responses output-item position, expanded to leave room for summaries.
    block_index: usize,
}

fn output_block_index(output_index: Option<usize>) -> usize {
    output_index.unwrap_or(0).saturating_mul(2)
}

fn reasoning_summary_block_index(output_index: Option<usize>) -> usize {
    output_block_index(output_index).saturating_add(1)
}

fn message_state_from_output_item(
    item: &serde_json::Value,
    output_index: Option<usize>,
) -> Option<(usize, serde_json::Value)> {
    (item.get("type").and_then(serde_json::Value::as_str) == Some(OPENAI_MESSAGE_ITEM_TYPE)).then(
        || {
            let role = json_string(item, "role").unwrap_or("assistant");
            let phase = json_string(item, "phase");
            (
                output_block_index(output_index),
                message_state_marker(role, phase),
            )
        },
    )
}

fn json_string<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
}

fn incomplete_stop_reason(reason: &str) -> StopReason {
    match reason {
        "max_output_tokens" => StopReason::MaxTokens,
        "content_filter" => StopReason::Refusal,
        "model_context_window_exceeded" => StopReason::ModelContextWindowExceeded,
        _ => StopReason::Unknown,
    }
}

/// Emit accumulated tool calls as stream deltas with distinct, monotonically
/// increasing block indices in registration order.
///
/// The previous implementation assigned every call `block_index: 1` and iterated
/// `HashMap::values()`, so [`StreamAccumulator`](crate::streaming::StreamAccumulator)'s
/// stable sort preserved nondeterministic insertion order — multi-tool turns
/// replayed in different orders run to run. Sorting by registration order with a
/// unique index per call makes the final content-block order deterministic.
fn flush_responses_tool_calls(
    tool_calls: &std::collections::HashMap<String, ToolCallAccumulator>,
) -> Vec<StreamDelta> {
    let mut accs: Vec<&ToolCallAccumulator> = tool_calls.values().collect();
    accs.sort_by_key(|acc| (acc.block_index, acc.order));

    let mut deltas = Vec::with_capacity(accs.len() * 2);
    for acc in accs {
        deltas.push(StreamDelta::ToolUseStart {
            id: acc.id.clone(),
            name: acc.name.clone(),
            block_index: acc.block_index,
            thought_signature: None,
        });
        deltas.push(StreamDelta::ToolInputDelta {
            id: acc.id.clone(),
            delta: acc.arguments.clone(),
            block_index: acc.block_index,
        });
    }
    deltas
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
struct ApiResponsesRequest<'a> {
    model: &'a str,
    input: &'a [ApiInputItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiResponseText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<ApiPromptCacheOptions>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [&'a str]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_identifier: Option<&'a str>,
}

#[derive(Serialize)]
struct ApiResponsesRequestStreaming<'a> {
    model: &'a str,
    input: &'a [ApiInputItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiResponseText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<ApiPromptCacheOptions>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_identifier: Option<String>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<OpenAIReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<OpenAIReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<OpenAIReasoningContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<OpenAIReasoningSummary>,
}

impl ApiReasoning {
    const fn has_fields(&self) -> bool {
        self.effort.is_some()
            || self.mode.is_some()
            || self.context.is_some()
            || self.summary.is_some()
    }
}

/// Responses API structured-output wire field: `{"text": {"format": {...}}}`.
///
/// The Responses API carries JSON-schema structured output under
/// `text.format` (type `json_schema`), unlike Chat Completions' top-level
/// `response_format`.
#[derive(Serialize)]
struct ApiResponseText {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<ApiResponseTextFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<OpenAITextVerbosity>,
}

#[derive(Serialize)]
struct ApiResponseTextFormat {
    #[serde(rename = "type")]
    format_type: &'static str,
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

impl From<&ResponseFormat> for ApiResponseText {
    fn from(rf: &ResponseFormat) -> Self {
        let mut schema = rf.schema.clone();
        if rf.strict {
            let _ = normalize_strict_schema(&mut schema);
        }
        Self {
            format: Some(ApiResponseTextFormat {
                format_type: "json_schema",
                name: rf.name.clone(),
                schema,
                strict: rf.strict,
            }),
            verbosity: None,
        }
    }
}

impl ApiResponseText {
    fn from_options(
        response_format: Option<&ResponseFormat>,
        verbosity: Option<OpenAITextVerbosity>,
    ) -> Option<Self> {
        if response_format.is_none() && verbosity.is_none() {
            return None;
        }

        Some(Self {
            format: response_format.map(|rf| {
                let mut schema = rf.schema.clone();
                if rf.strict {
                    let _ = normalize_strict_schema(&mut schema);
                }
                ApiResponseTextFormat {
                    format_type: "json_schema",
                    name: rf.name.clone(),
                    schema,
                    strict: rf.strict,
                }
            }),
            verbosity,
        })
    }
}

#[derive(Clone, Copy, Serialize)]
struct ApiPromptCacheOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<OpenAIPromptCacheMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<OpenAIPromptCacheTtl>,
}

impl ApiPromptCacheOptions {
    const fn new(
        mode: Option<OpenAIPromptCacheMode>,
        ttl: Option<OpenAIPromptCacheTtl>,
    ) -> Option<Self> {
        if mode.is_none() && ttl.is_none() {
            None
        } else {
            Some(Self { mode, ttl })
        }
    }
}

#[derive(Clone, Copy)]
struct PromptCachePlan {
    options: Option<ApiPromptCacheOptions>,
    explicit_breakpoints: usize,
}

fn resolve_prompt_cache_plan(
    model: &str,
    request: &ChatRequest,
    config: Option<&OpenAIReasoningConfig>,
) -> Result<PromptCachePlan> {
    let exact_mode = config.and_then(OpenAIReasoningConfig::prompt_cache_mode);
    let exact_ttl = config.and_then(OpenAIReasoningConfig::prompt_cache_ttl);

    if !is_gpt56_model(model) && request.cache.is_some() {
        return Ok(PromptCachePlan {
            options: ApiPromptCacheOptions::new(exact_mode, exact_ttl),
            explicit_breakpoints: 0,
        });
    }

    let Some(cache) = request.cache.as_ref() else {
        return Ok(PromptCachePlan {
            options: ApiPromptCacheOptions::new(exact_mode, exact_ttl),
            explicit_breakpoints: 0,
        });
    };

    if let Some(ttl) = cache.ttl {
        anyhow::bail!(
            "OpenAI GPT-5.6 prompt caching supports only ttl=30m; shared cache TTL {} cannot be mapped losslessly. Use OpenAIReasoningConfig::with_prompt_cache_ttl(OpenAIPromptCacheTtl::ThirtyMinutes)",
            ttl.as_wire_str()
        );
    }

    if !cache.enabled {
        return Ok(PromptCachePlan {
            options: ApiPromptCacheOptions::new(Some(OpenAIPromptCacheMode::Explicit), None),
            explicit_breakpoints: 0,
        });
    }

    if let Some(max_breakpoints) = cache.max_breakpoints {
        return Ok(PromptCachePlan {
            options: ApiPromptCacheOptions::new(Some(OpenAIPromptCacheMode::Explicit), exact_ttl),
            explicit_breakpoints: usize::from(max_breakpoints.min(4)),
        });
    }

    Ok(PromptCachePlan {
        options: ApiPromptCacheOptions::new(exact_mode, exact_ttl),
        explicit_breakpoints: 0,
    })
}

/// Responses API `tool_choice` wire format.
///
/// - `"auto"` — model decides.
/// - `{"type": "function", "name": "<name>"}` — force a specific function.
#[derive(Serialize)]
#[serde(untagged)]
enum ApiToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        name: String,
    },
    AllowedTools {
        #[serde(rename = "type")]
        choice_type: &'static str,
        mode: OpenAIAllowedToolsMode,
        tools: Vec<ApiAllowedTool>,
    },
}

#[derive(Serialize)]
struct ApiAllowedTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    name: String,
}

impl From<&ToolChoice> for ApiToolChoice {
    fn from(tc: &ToolChoice) -> Self {
        match tc {
            ToolChoice::Auto => Self::Mode("auto"),
            ToolChoice::Tool(name) => Self::Function {
                choice_type: "function",
                name: name.clone(),
            },
        }
    }
}

impl From<&OpenAIToolChoice> for ApiToolChoice {
    fn from(choice: &OpenAIToolChoice) -> Self {
        match choice {
            OpenAIToolChoice::None => Self::Mode("none"),
            OpenAIToolChoice::Auto => Self::Mode("auto"),
            OpenAIToolChoice::Required => Self::Mode("required"),
            OpenAIToolChoice::Function(name) => Self::Function {
                choice_type: "function",
                name: name.clone(),
            },
            OpenAIToolChoice::AllowedTools { mode, tools } => Self::AllowedTools {
                choice_type: "allowed_tools",
                mode: *mode,
                tools: tools
                    .iter()
                    .map(|name| ApiAllowedTool {
                        tool_type: "function",
                        name: name.clone(),
                    })
                    .collect(),
            },
        }
    }
}

fn resolve_api_tool_choice(
    request_choice: Option<&ToolChoice>,
    config: Option<&OpenAIReasoningConfig>,
) -> Option<ApiToolChoice> {
    request_choice.map(ApiToolChoice::from).or_else(|| {
        config
            .and_then(OpenAIReasoningConfig::tool_choice)
            .map(ApiToolChoice::from)
    })
}

fn validate_responses_tool_choice(
    request_choice: Option<&ToolChoice>,
    config: Option<&OpenAIReasoningConfig>,
    tools: Option<&[agent_sdk_foundation::llm::Tool]>,
) -> Result<()> {
    if let Some(request_choice) = request_choice {
        if let ToolChoice::Tool(name) = request_choice
            && !tools
                .unwrap_or_default()
                .iter()
                .any(|tool| tool.name == *name)
        {
            anyhow::bail!("OpenAI tool_choice names unknown function `{name}`");
        }
        return Ok(());
    }

    validate_tool_choice(config, tools)
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiInputItem {
    Message(ApiMessage),
    FunctionCall(ApiFunctionCall),
    FunctionCallOutput(ApiFunctionCallOutput),
    Opaque(serde_json::Value),
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    System,
    User,
    Assistant,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Parts(Vec<ApiInputContent>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ApiInputContent {
    #[serde(rename = "input_text")]
    InputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<ApiPromptCacheBreakpoint>,
    },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    Image {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<ApiPromptCacheBreakpoint>,
    },
    #[serde(rename = "input_file")]
    File {
        filename: String,
        file_data: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<ApiPromptCacheBreakpoint>,
    },
}

impl ApiInputContent {
    const fn input_text(text: String) -> Self {
        Self::InputText {
            text,
            prompt_cache_breakpoint: None,
        }
    }

    const fn text(role: ApiRole, text: String) -> Self {
        match role {
            ApiRole::Assistant => Self::OutputText { text },
            ApiRole::System | ApiRole::User => Self::input_text(text),
        }
    }

    const fn supports_explicit_cache_breakpoint(&self) -> bool {
        matches!(
            self,
            Self::InputText { .. } | Self::Image { .. } | Self::File { .. }
        )
    }

    const fn set_explicit_cache_breakpoint(&mut self) {
        let breakpoint = Some(ApiPromptCacheBreakpoint {
            mode: OpenAIPromptCacheMode::Explicit,
        });
        match self {
            Self::InputText {
                prompt_cache_breakpoint,
                ..
            }
            | Self::Image {
                prompt_cache_breakpoint,
                ..
            }
            | Self::File {
                prompt_cache_breakpoint,
                ..
            } => *prompt_cache_breakpoint = breakpoint,
            Self::OutputText { .. } => {}
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct ApiPromptCacheBreakpoint {
    mode: OpenAIPromptCacheMode,
}

#[derive(Serialize)]
struct ApiFunctionCall {
    r#type: &'static str,
    call_id: String,
    name: String,
    arguments: String,
}

impl ApiFunctionCall {
    const fn new(call_id: String, name: String, arguments: String) -> Self {
        Self {
            r#type: "function_call",
            call_id,
            name,
            arguments,
        }
    }
}

#[derive(Serialize)]
struct ApiFunctionCallOutput {
    r#type: &'static str,
    call_id: String,
    output: String,
}

impl ApiFunctionCallOutput {
    const fn new(call_id: String, output: String) -> Self {
        Self {
            r#type: "function_call_output",
            call_id,
            output,
        }
    }
}

#[derive(Serialize)]
struct ApiTool {
    r#type: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
struct ApiResponse {
    id: String,
    model: String,
    output: Vec<ApiOutputItem>,
    #[serde(default)]
    status: Option<ApiStatus>,
    #[serde(default)]
    usage: Option<ApiUsage>,
    #[serde(default)]
    error: Option<ApiResponseError>,
    #[serde(default)]
    incomplete_details: Option<ApiIncompleteDetails>,
}

#[derive(Deserialize)]
struct ApiResponseError {
    #[serde(default)]
    message: Option<String>,
    /// Machine-readable error code (e.g. `rate_limit_exceeded`), used to
    /// classify a failure the HTTP status cannot describe because the stream
    /// already opened with 200.
    #[serde(default)]
    code: Option<String>,
}

#[derive(Deserialize)]
struct ApiIncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ApiStatus {
    Completed,
    Incomplete,
    Failed,
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    input_tokens_details: Option<ApiInputTokensDetails>,
}

#[derive(Deserialize)]
struct ApiInputTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
    #[serde(default)]
    cache_write_tokens: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiOutputItem {
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(flatten)]
        data: serde_json::Map<String, serde_json::Value>,
    },
    #[serde(rename = "message")]
    Message {
        role: String,
        #[serde(default)]
        phase: Option<String>,
        content: Vec<ApiOutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiOutputContent {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Unknown,
}

// ============================================================================
// Streaming Types
// ============================================================================

#[derive(Deserialize)]
struct ApiStreamEvent {
    r#type: String,
    #[serde(default)]
    delta: Option<String>,
    /// Present on `output_item.added` / `output_item.done` for `function_call` items.
    #[serde(default)]
    item: Option<serde_json::Value>,
    /// Position of the item in the response output array.
    #[serde(default)]
    output_index: Option<usize>,
    /// Present on `function_call_arguments.delta`.
    #[serde(default)]
    item_id: Option<String>,
    /// Legacy field — some older events use `call_id` instead of `item_id`.
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    response: Option<ApiStreamResponse>,
}

impl ApiStreamEvent {
    /// Resolve the item identifier from whichever field is present.
    fn resolve_item_id(&self) -> Option<&str> {
        self.item_id
            .as_deref()
            .or(self.call_id.as_deref())
            .or_else(|| self.item.as_ref().and_then(|item| json_string(item, "id")))
    }
}

#[derive(Deserialize)]
struct ApiStreamResponse {
    #[serde(default)]
    usage: Option<ApiUsage>,
    #[serde(default)]
    incomplete_details: Option<ApiIncompleteDetails>,
    #[serde(default)]
    error: Option<ApiResponseError>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::llm::{CacheConfig, CacheTtl, Message};
    use anyhow::{Context as _, bail};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

    async fn stream_deltas(body: &str) -> anyhow::Result<Vec<StreamDelta>> {
        let server = MockServer::start().await;
        Mock::given(matchers::method("POST"))
            .and(matchers::path("/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let provider = OpenAIResponsesProvider::with_base_url(
            "test-key".to_owned(),
            MODEL_GPT56.to_owned(),
            server.uri(),
        );
        let mut stream = std::pin::pin!(provider.chat_stream(ChatRequest::new(
            String::new(),
            vec![Message::user("hello")],
        )));
        let mut deltas = Vec::new();
        while let Some(delta) = stream.next().await {
            deltas.push(delta?);
        }
        Ok(deltas)
    }

    #[test]
    fn test_model_constant() {
        assert_eq!(MODEL_GPT56, "gpt-5.6");
        assert_eq!(MODEL_GPT56_SOL, "gpt-5.6-sol");
        assert_eq!(MODEL_GPT56_TERRA, "gpt-5.6-terra");
        assert_eq!(MODEL_GPT56_LUNA, "gpt-5.6-luna");
        assert_eq!(MODEL_GPT53_CODEX, "gpt-5.3-codex");
        assert_eq!(MODEL_GPT52_CODEX, "gpt-5.2-codex");
    }

    #[test]
    fn test_gpt56_factories_create_expected_providers() {
        for (provider, expected_model) in [
            (
                OpenAIResponsesProvider::gpt56("test-key".to_string()),
                MODEL_GPT56,
            ),
            (
                OpenAIResponsesProvider::gpt56_sol("test-key".to_string()),
                MODEL_GPT56_SOL,
            ),
            (
                OpenAIResponsesProvider::gpt56_terra("test-key".to_string()),
                MODEL_GPT56_TERRA,
            ),
            (
                OpenAIResponsesProvider::gpt56_luna("test-key".to_string()),
                MODEL_GPT56_LUNA,
            ),
        ] {
            assert_eq!(provider.model(), expected_model);
            assert_eq!(provider.provider(), "openai-responses");
            assert_eq!(provider.default_max_tokens(), 128_000);
        }
    }

    #[test]
    fn test_codex_factory() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai-responses");
    }

    #[test]
    fn test_gpt53_codex_factory() {
        let provider = OpenAIResponsesProvider::gpt53_codex("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai-responses");
    }

    #[test]
    fn test_reasoning_effort_serialization() -> anyhow::Result<()> {
        let low = serde_json::to_string(&ReasoningEffort::Low)?;
        assert_eq!(low, "\"low\"");

        let xhigh = serde_json::to_string(&ReasoningEffort::XHigh)?;
        assert_eq!(xhigh, "\"xhigh\"");
        Ok(())
    }

    #[test]
    fn test_with_reasoning_effort() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string())
            .with_reasoning_effort(ReasoningEffort::High);
        assert!(provider.thinking.is_none());
        assert_eq!(
            provider
                .reasoning
                .as_ref()
                .and_then(OpenAIReasoningConfig::effort),
            Some(OpenAIReasoningEffort::High)
        );
    }

    #[test]
    fn test_build_api_reasoning_uses_exact_controls() -> anyhow::Result<()> {
        let config = OpenAIReasoningConfig::new()
            .with_effort(OpenAIReasoningEffort::Max)
            .with_mode(OpenAIReasoningMode::Pro)
            .with_context(OpenAIReasoningContext::AllTurns)
            .with_summary(OpenAIReasoningSummary::Detailed);
        let reasoning = build_api_reasoning(Some(&config))
            .ok_or_else(|| anyhow::anyhow!("reasoning config was omitted"))?;
        let json = serde_json::to_value(reasoning)?;
        assert_eq!(json["effort"], "max");
        assert_eq!(json["mode"], "pro");
        assert_eq!(json["context"], "all_turns");
        assert_eq!(json["summary"], "detailed");
        Ok(())
    }

    #[test]
    fn test_build_api_reasoning_omits_adaptive_without_effort() {
        let config = OpenAIReasoningConfig::new()
            .with_optional_effort(legacy_reasoning_effort(&ThinkingConfig::adaptive()));
        assert!(build_api_reasoning(Some(&config)).is_none());
    }

    #[test]
    fn exact_and_legacy_provider_configuration_are_last_call_wins() {
        let exact = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Max);
        let legacy = ThinkingConfig::new(8_192);

        let provider = OpenAIResponsesProvider::gpt56("test-key".to_string())
            .with_thinking(legacy.clone())
            .with_reasoning(exact.clone());
        assert!(provider.thinking.is_none());
        assert_eq!(provider.reasoning, Some(exact));

        let provider = provider.with_thinking(legacy);
        assert!(provider.reasoning.is_none());
        assert!(provider.thinking.is_some());
    }

    #[test]
    fn effective_max_tokens_uses_model_default_unless_explicit() {
        let provider = OpenAIResponsesProvider::gpt56("test-key".to_string());
        let implicit = ChatRequest::new(String::new(), vec![]);
        assert_eq!(provider.effective_max_tokens(&implicit), 128_000);

        let explicit = implicit.with_max_tokens(8_192);
        assert_eq!(provider.effective_max_tokens(&explicit), 8_192);
    }

    #[test]
    fn test_openai_responses_accepts_adaptive_thinking_for_codex() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string());
        assert!(
            provider
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );
    }

    #[test]
    fn test_api_tool_serialization() {
        let tool = ApiTool {
            r#type: "function".to_owned(),
            name: "get_weather".to_owned(),
            description: Some("Get weather".to_owned()),
            parameters: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("\"strict\":true"));
    }

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "id": "resp_123",
            "model": "gpt-5.2-codex",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ]
                }
            ],
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp_123");
        assert_eq!(response.model, "gpt-5.2-codex");
        assert_eq!(response.output.len(), 1);
    }

    #[test]
    fn test_map_usage_preserves_cache_read_and_write_tokens() -> anyhow::Result<()> {
        let api_usage: ApiUsage = serde_json::from_value(serde_json::json!({
            "input_tokens": 42,
            "output_tokens": 7,
            "input_tokens_details": {
                "cached_tokens": 10,
                "cache_write_tokens": 6
            }
        }))?;

        let usage = map_usage(Some(api_usage));
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cached_input_tokens, 10);
        assert_eq!(usage.cache_creation_input_tokens, 6);
        Ok(())
    }

    #[test]
    fn test_api_response_with_function_call() {
        let json = r#"{
            "id": "resp_456",
            "model": "gpt-5.2-codex",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": "{\"path\": \"test.txt\"}"
                }
            ],
            "status": "completed"
        }"#;

        let response: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.output.len(), 1);

        match &response.output[0] {
            ApiOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(name, "read_file");
                assert!(arguments.contains("test.txt"));
            }
            _ => panic!("Expected FunctionCall"),
        }
    }

    #[test]
    fn test_build_content_blocks_text() {
        let output = vec![ApiOutputItem::Message {
            role: "assistant".to_owned(),
            phase: Some("final_answer".to_owned()),
            content: vec![ApiOutputContent::Text {
                text: "Hello!".to_owned(),
            }],
        }];

        let blocks = build_content_blocks(&output);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(
            &blocks[0],
            ContentBlock::OpaqueReasoning { data, .. }
                if data["phase"] == "final_answer"
        ));
        assert!(matches!(&blocks[1], ContentBlock::Text { text } if text == "Hello!"));
    }

    #[test]
    fn test_build_content_blocks_function_call() {
        let output = vec![ApiOutputItem::FunctionCall {
            call_id: "call_123".to_owned(),
            name: "test_tool".to_owned(),
            arguments: r#"{"key": "value"}"#.to_owned(),
        }];

        let blocks = build_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        assert!(
            matches!(&blocks[0], ContentBlock::ToolUse { id, name, .. } if id == "call_123" && name == "test_tool")
        );
    }

    #[test]
    fn test_request_serializes_response_format_as_text_format_and_forced_tool_choice() {
        let req = ApiResponsesRequest {
            model: "gpt-5.3-codex",
            input: &[],
            tools: None,
            max_output_tokens: Some(1024),
            reasoning: None,
            parallel_tool_calls: None,
            text: Some(ApiResponseText::from(&ResponseFormat::new(
                "person",
                serde_json::json!({"type": "object", "properties": {}}),
            ))),
            tool_choice: Some(ApiToolChoice::from(&ToolChoice::Tool("respond".to_owned()))),
            prompt_cache_options: None,
            store: false,
            include: None,
            prompt_cache_key: None,
            safety_identifier: None,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["text"]["format"]["type"], "json_schema");
        assert_eq!(json["text"]["format"]["name"], "person");
        assert_eq!(json["text"]["format"]["strict"], true);
        assert_eq!(json["text"]["format"]["schema"]["type"], "object");
        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["name"], "respond");
    }

    #[test]
    fn test_tool_choice_auto_serializes_as_string() {
        let json = serde_json::to_value(ApiToolChoice::from(&ToolChoice::Auto)).unwrap();
        assert_eq!(json, serde_json::json!("auto"));
    }

    #[test]
    fn test_api_response_failed_status_carries_error_message() {
        let json = r#"{
            "id": "resp_fail",
            "model": "gpt-5.3-codex",
            "output": [],
            "status": "failed",
            "error": {"message": "model produced no output"}
        }"#;

        let resp: ApiResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(resp.status, Some(ApiStatus::Failed)));
        assert_eq!(
            resp.error.and_then(|e| e.message).as_deref(),
            Some("model produced no output")
        );
    }

    #[test]
    fn test_flush_responses_tool_calls_assigns_distinct_ordered_indices() {
        let mut tool_calls = std::collections::HashMap::new();
        tool_calls.insert(
            "b".to_owned(),
            ToolCallAccumulator {
                id: "b".to_owned(),
                name: "second".to_owned(),
                arguments: "{}".to_owned(),
                order: 1,
                block_index: 4,
            },
        );
        tool_calls.insert(
            "a".to_owned(),
            ToolCallAccumulator {
                id: "a".to_owned(),
                name: "first".to_owned(),
                arguments: "{}".to_owned(),
                order: 0,
                block_index: 2,
            },
        );

        let deltas = flush_responses_tool_calls(&tool_calls);
        let starts: Vec<(String, usize)> = deltas
            .iter()
            .filter_map(|d| match d {
                StreamDelta::ToolUseStart {
                    name, block_index, ..
                } => Some((name.clone(), *block_index)),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![("first".to_owned(), 2), ("second".to_owned(), 4)]
        );
    }

    #[test]
    fn input_preserves_reasoning_and_tool_item_order() -> anyhow::Result<()> {
        let reasoning = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque-ciphertext",
            "summary": []
        });
        let request = ChatRequest::new(
            "system",
            vec![Message::assistant_with_content(vec![
                ContentBlock::Text {
                    text: "before".to_owned(),
                },
                ContentBlock::OpaqueReasoning {
                    provider: "openai-responses".to_owned(),
                    data: reasoning.clone(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_owned(),
                    name: "lookup".to_owned(),
                    input: serde_json::json!({"q": "value"}),
                    thought_signature: None,
                },
                ContentBlock::Text {
                    text: "after".to_owned(),
                },
            ])],
        );

        let json = serde_json::to_value(build_api_input(&request, 0))?;
        let items = json
            .as_array()
            .context("input must serialize as an array")?;
        assert_eq!(items.len(), 5);
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[1]["content"][0]["text"], "before");
        assert_eq!(items[1]["phase"], "commentary");
        assert_eq!(items[2], reasoning);
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[4]["content"][0]["text"], "after");
        Ok(())
    }

    #[test]
    fn assistant_message_phases_round_trip_without_duplicate_text() -> anyhow::Result<()> {
        let api_response: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_phases",
            "model": "gpt-5.6",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Working."}]
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Done."}]
                }
            ]
        }))?;
        let ChatOutcome::Success(response) = build_responses_outcome(api_response) else {
            bail!("expected a successful response")
        };
        let request = ChatRequest::new(
            String::new(),
            vec![Message::assistant_with_content(response.content)],
        );
        let value = serde_json::to_value(build_api_input(&request, 0))?;
        let items = value.as_array().context("input must be an array")?;

        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["phase"], "commentary");
        assert_eq!(items[0]["content"][0]["text"], "Working.");
        assert_eq!(items[1]["phase"], "final_answer");
        assert_eq!(items[1]["content"][0]["text"], "Done.");
        assert_eq!(value.to_string().matches("Working.").count(), 1);
        assert_eq!(value.to_string().matches("Done.").count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn streamed_message_phase_marker_precedes_and_round_trips_text() -> anyhow::Result<()> {
        let body = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"commentary\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Working.\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"commentary\",\"content\":[{\"type\":\"output_text\",\"text\":\"Working.\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let deltas = stream_deltas(body).await?;
        let marker_position = deltas
            .iter()
            .position(|delta| matches!(delta, StreamDelta::OpaqueReasoning { data, .. } if data["type"] == "message"))
            .context("stream must contain a message phase marker")?;
        let text_position = deltas
            .iter()
            .position(|delta| matches!(delta, StreamDelta::TextDelta { .. }))
            .context("stream must contain text")?;
        assert!(marker_position < text_position);

        let mut accumulator = crate::streaming::StreamAccumulator::new();
        for delta in &deltas {
            accumulator.apply(delta);
        }
        let request = ChatRequest::new(
            String::new(),
            vec![Message::assistant_with_content(
                accumulator.into_content_blocks(),
            )],
        );
        let value = serde_json::to_value(build_api_input(&request, 0))?;
        let items = value.as_array().context("input must be an array")?;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["phase"], "commentary");
        assert_eq!(items[0]["content"][0]["text"], "Working.");
        assert_eq!(value.to_string().matches("Working.").count(), 1);
        Ok(())
    }

    #[test]
    fn output_preserves_opaque_reasoning_summary_and_refusal() -> anyhow::Result<()> {
        let raw_reasoning = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque-ciphertext",
            "summary": [{"type": "summary_text", "text": "Checked constraints."}],
            "future_field": {"kept": true}
        });
        let response: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1",
            "model": "gpt-5.6",
            "status": "completed",
            "output": [
                raw_reasoning,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}]
                }
            ]
        }))?;

        let ChatOutcome::Success(response) = build_responses_outcome(response) else {
            bail!("expected a successful protocol response carrying a refusal")
        };
        assert_eq!(response.stop_reason, Some(StopReason::Refusal));
        assert!(matches!(
            &response.content[0],
            ContentBlock::OpaqueReasoning { provider, data }
                if provider == "openai-responses"
                    && data["encrypted_content"] == "opaque-ciphertext"
                    && data["future_field"]["kept"] == true
        ));
        assert!(matches!(
            &response.content[1],
            ContentBlock::Thinking { thinking, .. } if thinking == "Checked constraints."
        ));
        assert!(matches!(
            &response.content[2],
            ContentBlock::OpaqueReasoning { data, .. }
                if data["type"] == "message"
        ));
        assert!(matches!(
            &response.content[3],
            ContentBlock::Text { text } if text == "I cannot help with that."
        ));
        Ok(())
    }

    #[test]
    fn incomplete_reason_is_not_misreported_as_successful_tool_use() -> anyhow::Result<()> {
        let response: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_incomplete",
            "model": "gpt-5.6",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{
                "type": "function_call",
                "call_id": "call_partial",
                "name": "lookup",
                "arguments": "{"
            }]
        }))?;

        let ChatOutcome::Success(response) = build_responses_outcome(response) else {
            bail!("expected an incomplete response")
        };
        assert_eq!(response.stop_reason, Some(StopReason::MaxTokens));
        assert!(
            !response
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        );
        Ok(())
    }

    #[test]
    fn refusal_suppresses_partial_tool_calls() -> anyhow::Result<()> {
        let response: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_refusal",
            "model": "gpt-5.6",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_partial",
                    "name": "lookup",
                    "arguments": "{}"
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "refusal", "refusal": "Cannot comply."}]
                }
            ]
        }))?;

        let ChatOutcome::Success(response) = build_responses_outcome(response) else {
            bail!("expected a refusal response")
        };
        assert_eq!(response.stop_reason, Some(StopReason::Refusal));
        assert!(
            !response
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        );
        Ok(())
    }

    #[test]
    fn prompt_cache_plan_maps_exact_controls_and_rejects_anthropic_ttls() -> anyhow::Result<()> {
        let exact = OpenAIReasoningConfig::new()
            .with_prompt_cache_mode(OpenAIPromptCacheMode::Explicit)
            .with_prompt_cache_ttl(OpenAIPromptCacheTtl::ThirtyMinutes);
        let request = ChatRequest::new(String::new(), vec![Message::user("hello")]);
        let plan = resolve_prompt_cache_plan(MODEL_GPT56, &request, Some(&exact))?;
        assert_eq!(plan.explicit_breakpoints, 0);
        let options = plan.options.context("cache options missing")?;
        let json = serde_json::to_value(options)?;
        assert_eq!(json["mode"], "explicit");
        assert_eq!(json["ttl"], "30m");

        let generic_request = request
            .clone()
            .with_cache(CacheConfig::enabled().with_max_breakpoints(4));
        let generic_plan = resolve_prompt_cache_plan(MODEL_GPT56, &generic_request, Some(&exact))?;
        assert_eq!(generic_plan.explicit_breakpoints, 4);

        let incompatible = request.with_cache(CacheConfig::enabled().with_ttl(CacheTtl::OneHour));
        let error = resolve_prompt_cache_plan(MODEL_GPT56, &incompatible, Some(&exact))
            .err()
            .context("Anthropic-only TTL should be rejected")?;
        assert!(error.to_string().contains("cannot be mapped losslessly"));

        let legacy_plan = resolve_prompt_cache_plan(MODEL_GPT53_CODEX, &incompatible, None)?;
        assert!(legacy_plan.options.is_none());
        assert_eq!(legacy_plan.explicit_breakpoints, 0);
        Ok(())
    }

    #[test]
    fn explicit_cache_breakpoints_are_capped_and_applied_to_the_tail() -> anyhow::Result<()> {
        let request = ChatRequest::new(
            "system",
            vec![
                Message::user("one"),
                Message::assistant("two"),
                Message::user("three"),
                Message::assistant("four"),
                Message::user("five"),
            ],
        );
        let json = serde_json::to_string(&build_api_input(&request, 9))?;
        assert_eq!(json.matches("prompt_cache_breakpoint").count(), 4);
        assert!(json.contains("\"text\":\"one\",\"prompt_cache_breakpoint\""));
        assert!(json.contains("\"text\":\"five\",\"prompt_cache_breakpoint\""));
        assert!(!json.contains("\"text\":\"two\",\"prompt_cache_breakpoint\""));
        assert!(!json.contains("\"text\":\"four\",\"prompt_cache_breakpoint\""));
        Ok(())
    }

    #[test]
    fn explicit_cache_breakpoints_only_annotate_input_parts() -> anyhow::Result<()> {
        let request = ChatRequest::new(
            "system",
            vec![
                Message::assistant("assistant output"),
                Message::user("user input"),
                Message::assistant("final output"),
            ],
        );
        let value = serde_json::to_value(build_api_input(&request, 4))?;
        let items = value.as_array().context("input must be an array")?;
        let mut marker_count = 0;

        for part in items
            .iter()
            .filter_map(|item| item.get("content").and_then(serde_json::Value::as_array))
            .flatten()
        {
            if part.get("prompt_cache_breakpoint").is_some() {
                marker_count += 1;
                assert!(matches!(
                    part.get("type").and_then(serde_json::Value::as_str),
                    Some("input_text" | "input_image" | "input_file")
                ));
            }
            if part.get("type").and_then(serde_json::Value::as_str) == Some("output_text") {
                assert!(part.get("prompt_cache_breakpoint").is_none());
            }
        }

        assert_eq!(marker_count, 2);
        Ok(())
    }

    #[test]
    fn strict_response_format_rejects_free_form_objects() {
        let format = ResponseFormat::new("freeform", serde_json::json!({"type": "object"}));
        assert!(validate_response_format(Some(&format)).is_err());
    }

    #[test]
    fn allowed_tools_serializes_complete_responses_policy() -> anyhow::Result<()> {
        let choice = OpenAIToolChoice::AllowedTools {
            mode: OpenAIAllowedToolsMode::Required,
            tools: vec!["lookup".to_owned(), "search".to_owned()],
        };
        let json = serde_json::to_value(ApiToolChoice::from(&choice))?;
        assert_eq!(json["type"], "allowed_tools");
        assert_eq!(json["mode"], "required");
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["name"], "lookup");
        Ok(())
    }

    #[tokio::test]
    async fn in_band_rate_limit_event_is_recoverable_and_keeps_its_hint() -> anyhow::Result<()> {
        // The stream opened 200 and only then hit the quota, so the HTTP-status
        // branch never runs: the delay exists solely in the event's message.
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\n",
            "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"Rate limit reached for gpt-5.6 in organization org-x on tokens per min. Please try again in 20s.\"}\n\n",
        );
        let deltas = stream_deltas(body).await?;
        let kind = deltas
            .iter()
            .find_map(|delta| match delta {
                StreamDelta::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected a stream error delta")?;

        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(20))),
            "an in-band rate limit must be retriable and carry its parsed delay"
        );
        assert!(kind.is_recoverable());
        Ok(())
    }

    #[tokio::test]
    async fn in_band_rate_limit_on_response_failed_is_not_a_server_error() -> anyhow::Result<()> {
        // `response.failed` defaults to ServerError; a rate-limit code inside it
        // must still classify as a rate limit so the hint is not discarded.
        let body = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Please try again in 1.5s\"}}}\n\n";
        let deltas = stream_deltas(body).await?;
        let kind = deltas
            .iter()
            .find_map(|delta| match delta {
                StreamDelta::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected a stream error delta")?;

        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_millis(1500)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn in_band_failed_event_keeps_the_usage_it_reported() -> anyhow::Result<()> {
        // The failed response reports the tokens it burned. They are billed, so
        // they must reach the accumulator — which only sees yielded deltas —
        // before the error that ends the stream.
        let body = "data: {\"type\":\"response.failed\",\"response\":{\"usage\":{\"input_tokens\":140,\"output_tokens\":20},\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Please try again in 12s.\"}}}\n\n";
        let deltas = stream_deltas(body).await?;

        let usage_at = deltas
            .iter()
            .position(|delta| matches!(delta, StreamDelta::Usage(_)))
            .context("the failed event's usage must not be dropped")?;
        let error_at = deltas
            .iter()
            .position(|delta| matches!(delta, StreamDelta::Error { .. }))
            .context("expected a stream error delta")?;
        assert!(
            usage_at < error_at,
            "usage must be emitted before the terminal error, got {deltas:?}"
        );

        let StreamDelta::Usage(usage) = &deltas[usage_at] else {
            bail!("expected a usage delta");
        };
        assert_eq!(usage.input_tokens, 140);
        assert_eq!(usage.output_tokens, 20);
        Ok(())
    }

    #[tokio::test]
    async fn in_band_non_rate_limit_error_keeps_its_previous_classification() -> anyhow::Result<()>
    {
        let body = "data: {\"type\":\"error\",\"code\":\"invalid_prompt\",\"message\":\"bad\"}\n\n";
        let deltas = stream_deltas(body).await?;
        let kind = deltas
            .iter()
            .find_map(|delta| match delta {
                StreamDelta::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected a stream error delta")?;

        assert_eq!(kind, StreamErrorKind::InvalidRequest);
        assert!(!kind.is_recoverable());
        Ok(())
    }

    #[tokio::test]
    async fn in_band_response_failed_context_window_rejection_is_fatal() -> anyhow::Result<()> {
        // `response.failed` defaults to ServerError, but a context-window
        // rejection is a fatal request-shape error: the worker's overflow
        // compaction must fire instead of retrying the oversized payload.
        let body = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"Your input exceeds the context window of this model. Please adjust your input and try again.\"}}}\n\n";
        let deltas = stream_deltas(body).await?;
        let kind = deltas
            .iter()
            .find_map(|delta| match delta {
                StreamDelta::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected a stream error delta")?;

        assert_eq!(kind, StreamErrorKind::InvalidRequest);
        assert!(!kind.is_recoverable());
        Ok(())
    }

    #[tokio::test]
    async fn semantic_terminal_event_preserves_streamed_reasoning_and_usage() -> anyhow::Result<()>
    {
        let body = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Checked.\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"cipher\",\"summary\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Answer\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":3,\"input_tokens_details\":{\"cached_tokens\":4,\"cache_write_tokens\":2}}}}\n\n",
            "data: [DONE]\n\n",
        );
        let deltas = stream_deltas(body).await?;
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::OpaqueReasoning { provider, data, block_index }
                if provider == "openai-responses"
                    && data["encrypted_content"] == "cipher"
                    && *block_index == 0
        )));
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::ThinkingDelta { delta, block_index }
                if delta == "Checked." && *block_index == 1
        )));
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::TextDelta { delta, block_index }
                if delta == "Answer" && *block_index == 2
        )));
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::Usage(usage)
                if usage.cached_input_tokens == 4 && usage.cache_creation_input_tokens == 2
        )));
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn),
                ..
            })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn non_tool_terminals_suppress_streamed_partial_tool_calls() -> anyhow::Result<()> {
        let incomplete = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"fc_1\",\"delta\":\"{\"}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
        );
        let incomplete_deltas = stream_deltas(incomplete).await?;
        assert!(matches!(
            incomplete_deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::MaxTokens),
                ..
            })
        ));
        assert!(!incomplete_deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolUseStart { .. } | StreamDelta::ToolInputDelta { .. }
        )));

        let refusal = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"fc_1\",\"delta\":\"{}\"}\n\n",
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":1,\"delta\":\"Cannot comply.\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let refusal_deltas = stream_deltas(refusal).await?;
        assert!(matches!(
            refusal_deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::Refusal),
                ..
            })
        ));
        assert!(!refusal_deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolUseStart { .. } | StreamDelta::ToolInputDelta { .. }
        )));
        Ok(())
    }

    #[tokio::test]
    async fn done_sentinel_without_semantic_terminal_is_a_recoverable_error() -> anyhow::Result<()>
    {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"partial\"}\n\n",
            "data: [DONE]\n\n",
        );
        let deltas = stream_deltas(body).await?;
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::Error {
                kind: StreamErrorKind::ServerError,
                ..
            })
        ));
        assert!(
            !deltas
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Done { .. }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn streamed_refusal_is_visible_and_has_refusal_stop_reason() -> anyhow::Result<()> {
        let body = concat!(
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"Cannot comply.\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let deltas = stream_deltas(body).await?;
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::TextDelta { delta, .. } if delta == "Cannot comply."
        )));
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::Refusal),
                ..
            })
        ));
        Ok(())
    }
}
