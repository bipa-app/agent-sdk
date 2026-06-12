//! `OpenAI` API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Chat Completions API. It also supports `OpenAI`-compatible APIs (Ollama, vLLM, etc.)
//! via the `with_base_url` constructor.
//!
//! # Transparent Responses-API reroute
//!
//! Some requests cannot be served by Chat Completions and are transparently
//! rerouted to the `OpenAI` Responses API
//! ([`OpenAIResponsesProvider`]). The reroute (`should_use_responses_api`) fires
//! when:
//!
//! - the model only exists on the Responses surface (e.g. `gpt-5.2-codex`), or
//! - the request carries attachments (images / documents), or
//! - the request is *agentic* (has tools or tool-use/tool-result blocks) against
//!   the official `api.openai.com` base URL.
//!
//! The reroute forwards the provider's pooled HTTP client and `extra_headers`
//! (the BYOK / gateway auth mechanism) so a rerouted request keeps connection
//! reuse and authenticates identically to a non-rerouted one.

use crate::attachments::{request_has_attachments, validate_request_attachments};
use crate::provider::LlmProvider;
use crate::streaming::{SseLineBuffer, StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Effort, StopReason,
    ThinkingConfig, ThinkingMode, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::de::Error as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::openai_responses::OpenAIResponsesProvider;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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

/// Check if a model requires the Responses API instead of Chat Completions.
fn requires_responses_api(model: &str) -> bool {
    model == MODEL_GPT52_CODEX
}

fn is_official_openai_base_url(base_url: &str) -> bool {
    base_url == DEFAULT_BASE_URL || base_url.contains("api.openai.com")
}

fn request_is_agentic(request: &ChatRequest) -> bool {
    request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty()) || request.messages.iter().any(|message| {
        matches!(
            &message.content,
            Content::Blocks(blocks)
                if blocks.iter().any(|block| {
                    matches!(block, ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. })
                })
        )
    })
}

fn should_use_responses_api(base_url: &str, model: &str, request: &ChatRequest) -> bool {
    requires_responses_api(model)
        || request_has_attachments(request)
        || (is_official_openai_base_url(base_url) && request_is_agentic(request))
}

// GPT-5.4 series
pub const MODEL_GPT54: &str = "gpt-5.4";

// GPT-5.3 Codex series
pub const MODEL_GPT53_CODEX: &str = "gpt-5.3-codex";

// GPT-5.2 series
pub const MODEL_GPT52_INSTANT: &str = "gpt-5.2-instant";
pub const MODEL_GPT52_THINKING: &str = "gpt-5.2-thinking";
pub const MODEL_GPT52_PRO: &str = "gpt-5.2-pro";
pub const MODEL_GPT52_CODEX: &str = "gpt-5.2-codex";

// GPT-5 series (400k context)
pub const MODEL_GPT5: &str = "gpt-5";
pub const MODEL_GPT5_MINI: &str = "gpt-5-mini";
pub const MODEL_GPT5_NANO: &str = "gpt-5-nano";

// o-series reasoning models
pub const MODEL_O3: &str = "o3";
pub const MODEL_O3_MINI: &str = "o3-mini";
pub const MODEL_O4_MINI: &str = "o4-mini";
pub const MODEL_O1: &str = "o1";
pub const MODEL_O1_MINI: &str = "o1-mini";

// GPT-4.1 series (improved instruction following, 1M context)
pub const MODEL_GPT41: &str = "gpt-4.1";
pub const MODEL_GPT41_MINI: &str = "gpt-4.1-mini";
pub const MODEL_GPT41_NANO: &str = "gpt-4.1-nano";

// GPT-4o series
pub const MODEL_GPT4O: &str = "gpt-4o";
pub const MODEL_GPT4O_MINI: &str = "gpt-4o-mini";

// OpenAI-compatible vendor defaults
pub const BASE_URL_KIMI: &str = "https://api.moonshot.ai/v1";
pub const BASE_URL_ZAI: &str = "https://api.z.ai/api/paas/v4";
pub const BASE_URL_MINIMAX: &str = "https://api.minimax.io/v1";
pub const MODEL_KIMI_K2_5: &str = "kimi-k2.5";
pub const MODEL_KIMI_K2_THINKING: &str = "kimi-k2-thinking";
pub const MODEL_ZAI_GLM5: &str = "glm-5";
pub const MODEL_MINIMAX_M2_5: &str = "MiniMax-M2.5";

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
    thinking: Option<ThinkingConfig>,
    /// Extra headers applied to every request (e.g. for gateway authentication).
    extra_headers: Vec<(String, String)>,
}

impl OpenAIProvider {
    /// The conventional environment variable holding the `OpenAI` API key.
    pub const API_KEY_ENV: &'static str = "OPENAI_API_KEY";

    /// Create a new `OpenAI` provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: build_http_client(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            thinking: None,
            extra_headers: Vec::new(),
        }
    }

    /// Create a provider using GPT-5, reading the API key from the
    /// conventional [`OPENAI_API_KEY`](Self::API_KEY_ENV) environment variable.
    ///
    /// # Panics
    ///
    /// Panics if `OPENAI_API_KEY` is not set. Prefer
    /// [`try_from_env`](Self::try_from_env) outside of examples/tests.
    #[must_use]
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a provider using GPT-5, reading the API key from the
    /// conventional [`OPENAI_API_KEY`](Self::API_KEY_ENV) environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `OPENAI_API_KEY` is unset or not valid UTF-8.
    pub fn try_from_env() -> Result<Self> {
        let api_key = std::env::var(Self::API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!("environment variable `{}` is not set", Self::API_KEY_ENV)
        })?;
        Ok(Self::gpt5(api_key))
    }

    /// Create a new provider with a custom base URL for OpenAI-compatible APIs.
    #[must_use]
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: build_http_client(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            thinking: None,
            extra_headers: Vec::new(),
        }
    }

    /// Create a provider using Moonshot KIMI via OpenAI-compatible Chat Completions.
    #[must_use]
    pub fn kimi(api_key: String, model: String) -> Self {
        Self::with_base_url(api_key, model, BASE_URL_KIMI.to_owned())
    }

    /// Create a provider using KIMI K2.5 (default KIMI model).
    #[must_use]
    pub fn kimi_k2_5(api_key: String) -> Self {
        Self::kimi(api_key, MODEL_KIMI_K2_5.to_owned())
    }

    /// Create a provider using KIMI K2 Thinking.
    #[must_use]
    pub fn kimi_k2_thinking(api_key: String) -> Self {
        Self::kimi(api_key, MODEL_KIMI_K2_THINKING.to_owned())
    }

    /// Create a provider using z.ai via OpenAI-compatible Chat Completions.
    #[must_use]
    pub fn zai(api_key: String, model: String) -> Self {
        Self::with_base_url(api_key, model, BASE_URL_ZAI.to_owned())
    }

    /// Create a provider using z.ai GLM-5 (default z.ai agentic reasoning model).
    #[must_use]
    pub fn zai_glm5(api_key: String) -> Self {
        Self::zai(api_key, MODEL_ZAI_GLM5.to_owned())
    }

    /// Create a provider using `MiniMax` via OpenAI-compatible Chat Completions.
    #[must_use]
    pub fn minimax(api_key: String, model: String) -> Self {
        Self::with_base_url(api_key, model, BASE_URL_MINIMAX.to_owned())
    }

    /// Create a provider using `MiniMax` M2.5 (default `MiniMax` model).
    #[must_use]
    pub fn minimax_m2_5(api_key: String) -> Self {
        Self::minimax(api_key, MODEL_MINIMAX_M2_5.to_owned())
    }

    /// Create a provider using GPT-5.2 Instant (speed-optimized for routine queries).
    #[must_use]
    pub fn gpt52_instant(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_INSTANT.to_owned())
    }

    /// Create a provider using GPT-5.4 (frontier reasoning with 1.05M context).
    #[must_use]
    pub fn gpt54(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT54.to_owned())
    }

    /// Create a provider using GPT-5.3 Codex (latest codex model).
    #[must_use]
    pub fn gpt53_codex(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT53_CODEX.to_owned())
    }

    /// Create a provider using GPT-5.2 Thinking (complex reasoning, coding, analysis).
    #[must_use]
    pub fn gpt52_thinking(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_THINKING.to_owned())
    }

    /// Create a provider using GPT-5.2 Pro (maximum accuracy for difficult problems).
    #[must_use]
    pub fn gpt52_pro(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_PRO.to_owned())
    }

    /// Create a provider using the latest Codex model.
    #[must_use]
    pub fn codex(api_key: String) -> Self {
        Self::gpt53_codex(api_key)
    }

    /// Create a provider using GPT-5 (400k context, coding and reasoning).
    #[must_use]
    pub fn gpt5(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT5.to_owned())
    }

    /// Create a provider using GPT-5-mini (faster, cost-efficient GPT-5).
    #[must_use]
    pub fn gpt5_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT5_MINI.to_owned())
    }

    /// Create a provider using GPT-5-nano (fastest, cheapest GPT-5 variant).
    #[must_use]
    pub fn gpt5_nano(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT5_NANO.to_owned())
    }

    /// Create a provider using o3 (most intelligent reasoning model).
    #[must_use]
    pub fn o3(api_key: String) -> Self {
        Self::new(api_key, MODEL_O3.to_owned())
    }

    /// Create a provider using o3-mini (smaller o3 variant).
    #[must_use]
    pub fn o3_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_O3_MINI.to_owned())
    }

    /// Create a provider using o4-mini (fast, cost-efficient reasoning).
    #[must_use]
    pub fn o4_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_O4_MINI.to_owned())
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

    /// Create a provider using GPT-4.1 (improved instruction following, 1M context).
    #[must_use]
    pub fn gpt41(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT41.to_owned())
    }

    /// Create a provider using GPT-4.1-mini (smaller, faster GPT-4.1).
    #[must_use]
    pub fn gpt41_mini(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT41_MINI.to_owned())
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

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Add extra HTTP headers applied to every request.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Apply auth + extra headers. Skips `Authorization` when `api_key` is
    /// empty (BYOK gateway mode — auth handled via `extra_headers`).
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

    /// Build the `OpenAIResponsesProvider` used for the transparent Responses-API
    /// reroute, forwarding this provider's pooled client, thinking config, and
    /// extra headers so the rerouted request reuses connections and authenticates
    /// identically (critical for BYOK / gateway setups with an empty `api_key`).
    fn responses_reroute(&self) -> OpenAIResponsesProvider {
        let mut provider = OpenAIResponsesProvider::with_base_url(
            self.api_key.clone(),
            self.model.clone(),
            self.base_url.clone(),
        )
        .with_client(self.client.clone())
        .with_extra_headers(self.extra_headers.clone());
        if let Some(thinking) = self.thinking.clone() {
            provider = provider.with_thinking(thinking);
        }
        provider
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        // Route official OpenAI agentic flows to the Responses API, preserving
        // the pooled client and extra_headers (BYOK / gateway auth).
        if should_use_responses_api(&self.base_url, &self.model, &request) {
            return self.responses_reroute().chat(request).await;
        }

        let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
            Ok(thinking) => thinking,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let reasoning = build_api_reasoning(thinking_config.as_ref());
        let messages = build_api_messages(&request);
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());
        let tool_choice = request
            .tool_choice
            .as_ref()
            .map(ApiToolChoice::from_tool_choice);
        let response_format = request
            .response_format
            .as_ref()
            .map(ApiResponseFormat::from_response_format);

        let include_max_tokens_alias = use_max_tokens_alias(&self.base_url);
        let api_request = ApiChatRequest {
            model: &self.model,
            messages: &messages,
            max_completion_tokens: Some(request.max_tokens),
            max_tokens: include_max_tokens_alias.then_some(request.max_tokens),
            tools: tools.as_deref(),
            tool_choice,
            reasoning,
            response_format,
        };

        log::debug!(
            "OpenAI LLM request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let builder = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Content-Type", "application/json");
        let response = self
            .apply_headers(builder)
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
            "OpenAI LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        decode_chat_response(status, &bytes, retry_after)
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        // Route official OpenAI agentic flows to the Responses API, preserving
        // the pooled client and extra_headers (BYOK / gateway auth).
        if should_use_responses_api(&self.base_url, &self.model, &request) {
            let responses_provider = self.responses_reroute();
            return Box::pin(async_stream::stream! {
                let mut stream = std::pin::pin!(responses_provider.chat_stream(request));
                while let Some(item) = futures::StreamExt::next(&mut stream).await {
                    yield item;
                }
            });
        }

        Box::pin(async_stream::stream! {
            let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
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
            let reasoning = build_api_reasoning(thinking_config.as_ref());
            let messages = build_api_messages(&request);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| ts.into_iter().map(convert_tool).collect());
            let tool_choice = request
                .tool_choice
                .as_ref()
                .map(ApiToolChoice::from_tool_choice);
            let response_format = request
                .response_format
                .as_ref()
                .map(ApiResponseFormat::from_response_format);

            let include_max_tokens_alias = use_max_tokens_alias(&self.base_url);
            let include_stream_usage = use_stream_usage_options(&self.base_url);
            let include_openrouter_usage = use_openrouter_usage_options(&self.base_url);
            let api_request = ApiChatRequestStreaming {
                model: &self.model,
                messages: &messages,
                max_completion_tokens: Some(request.max_tokens),
                max_tokens: include_max_tokens_alias.then_some(request.max_tokens),
                tools: tools.as_deref(),
                tool_choice,
                reasoning,
                response_format,
                stream_options: include_stream_usage.then_some(ApiStreamOptions {
                    include_usage: true,
                }),
                usage: include_openrouter_usage
                    .then_some(ApiOpenRouterUsageOptions { include: true }),
                stream: true,
            };

            log::debug!("OpenAI streaming LLM request model={} max_tokens={}", self.model, request.max_tokens);

            let stream_builder = self.client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Content-Type", "application/json");
            let Ok(response) = self
                .apply_headers(stream_builder)
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
                let (kind, level) = if status == StatusCode::TOO_MANY_REQUESTS {
                    (StreamErrorKind::RateLimited, "rate_limit")
                } else if status.is_server_error() {
                    (StreamErrorKind::ServerError, "server_error")
                } else {
                    (StreamErrorKind::InvalidRequest, "client_error")
                };
                log::warn!("OpenAI error status={status} body={body} kind={level}");
                yield Ok(StreamDelta::Error { message: body, kind });
                return;
            }

            // Track tool call state across deltas
            let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
            let mut usage: Option<Usage> = None;
            // The stop reason from `finish_reason`. With stream_options.include_usage
            // (official OpenAI) the usage arrives in a SEPARATE trailing chunk
            // (choices: []) AFTER finish_reason and before [DONE], so we record the
            // stop reason and keep consuming until [DONE] / stream end rather than
            // returning early and dropping that usage chunk.
            let mut stop_reason: Option<StopReason> = None;
            let mut sse = SseLineBuffer::new();
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Err(anyhow::anyhow!("stream error: {error}"));
                        return;
                    }
                };
                sse.extend(&chunk);

                while let Some(line) = sse.next_line() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let Some(data) = line.strip_prefix("data: ") else { continue; };

                    let outcome = step_completion_stream(
                        data,
                        &mut tool_calls,
                        &mut usage,
                        &mut stop_reason,
                    );
                    for delta in outcome.immediate { yield Ok(delta); }
                    if let Some(terminal) = outcome.terminal {
                        for delta in terminal { yield Ok(delta); }
                        return;
                    }
                }
            }

            // Stream ended without [DONE] - emit what we have. Infer the stop
            // reason from accumulated tool calls (same heuristic as the [DONE]
            // arm) so a stream that dies mid-tool-call doesn't report EndTurn.
            let sr = stop_reason.unwrap_or_else(|| fallback_stream_stop_reason(&tool_calls));
            for delta in build_stream_end_deltas(&tool_calls, usage.take(), sr) {
                yield Ok(delta);
            }
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "openai"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }
}

/// Apply a tool call update to the accumulator.
fn apply_tool_call_update(
    tool_calls: &mut std::collections::HashMap<usize, ToolCallAccumulator>,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
) {
    let entry = tool_calls
        .entry(index)
        .or_insert_with(|| ToolCallAccumulator {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
        });
    if let Some(id) = id {
        entry.id = id;
    }
    if let Some(name) = name {
        entry.name = name;
    }
    if let Some(args) = arguments {
        entry.arguments.push_str(&args);
    }
}

/// Immediate + terminal deltas produced by feeding one SSE `data:` line to the
/// Chat Completions streaming state.
struct SseLineOutcome {
    /// Deltas to yield immediately (text / thinking).
    immediate: Vec<StreamDelta>,
    /// When `Some`, the stream finished ([DONE] received): yield these terminal
    /// deltas (tool calls + usage + Done) and stop.
    terminal: Option<Vec<StreamDelta>>,
}

/// Feed one SSE `data:` payload to the streaming state, accumulating tool calls,
/// usage, and the stop reason.
///
/// Text/thinking deltas are returned for immediate emission. A `finish_reason`
/// only records the stop reason (it does NOT finalize) so a trailing usage-only
/// chunk that official `OpenAI` sends after `finish_reason` is still folded in;
/// finalization happens on the `[DONE]` sentinel.
fn step_completion_stream(
    data: &str,
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
    usage: &mut Option<Usage>,
    stop_reason: &mut Option<StopReason>,
) -> SseLineOutcome {
    let mut immediate = Vec::new();
    for result in process_sse_data(data) {
        match result {
            SseProcessResult::TextDelta(c) => {
                immediate.push(StreamDelta::TextDelta {
                    delta: c,
                    block_index: 0,
                });
            }
            SseProcessResult::ThinkingDelta(c) => {
                immediate.push(StreamDelta::ThinkingDelta {
                    delta: c,
                    block_index: 0,
                });
            }
            SseProcessResult::ToolCallUpdate {
                index,
                id,
                name,
                arguments,
            } => apply_tool_call_update(tool_calls, index, id, name, arguments),
            SseProcessResult::Usage(u) => *usage = Some(u),
            SseProcessResult::Done(sr) => *stop_reason = Some(sr),
            SseProcessResult::Sentinel => {
                let sr = stop_reason.unwrap_or_else(|| fallback_stream_stop_reason(tool_calls));
                let terminal = build_stream_end_deltas(tool_calls, usage.take(), sr);
                return SseLineOutcome {
                    immediate,
                    terminal: Some(terminal),
                };
            }
        }
    }
    SseLineOutcome {
        immediate,
        terminal: None,
    }
}

/// Helper to emit tool call deltas and done event.
fn build_stream_end_deltas(
    tool_calls: &std::collections::HashMap<usize, ToolCallAccumulator>,
    usage: Option<Usage>,
    stop_reason: StopReason,
) -> Vec<StreamDelta> {
    let mut deltas = Vec::new();

    // Emit tool calls. `idx` comes from the wire `tool_calls[].index`; use
    // saturating_add so a hostile `usize::MAX` index cannot overflow-panic in
    // debug builds. StreamAccumulator sorts by index so order stays stable.
    for (idx, tool) in tool_calls {
        let block_index = idx.saturating_add(1);
        deltas.push(StreamDelta::ToolUseStart {
            id: tool.id.clone(),
            name: tool.name.clone(),
            block_index,
            thought_signature: None,
        });
        deltas.push(StreamDelta::ToolInputDelta {
            id: tool.id.clone(),
            delta: tool.arguments.clone(),
            block_index,
        });
    }

    // Emit usage
    if let Some(u) = usage {
        deltas.push(StreamDelta::Usage(u));
    }

    // Emit done
    deltas.push(StreamDelta::Done {
        stop_reason: Some(stop_reason),
    });

    deltas
}

/// Result of processing an SSE chunk.
enum SseProcessResult {
    /// Emit a text delta.
    TextDelta(String),
    /// Emit a thinking/reasoning delta (reasoning-model fallback when the model
    /// streams its output via `reasoning_content`/`reasoning` and `content` is
    /// empty, mirroring the non-streaming `build_content_blocks` fallback).
    ThinkingDelta(String),
    /// Update tool call accumulator (index, optional id, optional name, optional args).
    ToolCallUpdate {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    /// Usage information.
    Usage(Usage),
    /// Stream is done with a stop reason.
    Done(StopReason),
    /// Stream sentinel [DONE] was received.
    Sentinel,
}

/// Process an SSE data line and return results to apply.
fn process_sse_data(data: &str) -> Vec<SseProcessResult> {
    if data == "[DONE]" {
        return vec![SseProcessResult::Sentinel];
    }

    let Ok(chunk) = serde_json::from_str::<SseChunk>(data) else {
        return vec![];
    };

    let mut results = Vec::new();

    // Extract usage if present
    if let Some(u) = chunk.usage {
        results.push(SseProcessResult::Usage(Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: u
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            cache_creation_input_tokens: 0,
        }));
    }

    // Process choices
    if let Some(choice) = chunk.choices.into_iter().next() {
        // Handle text content delta. When `content` is empty/absent but the
        // model streamed reasoning tokens (DeepSeek-style answer-in-
        // `reasoning_content`, or `OpenRouter`-normalized `reasoning`), surface
        // the reasoning as a thinking delta so the usable output is not silently
        // dropped under streaming. This mirrors the non-streaming
        // `build_content_blocks` fallback: text content takes precedence and the
        // reasoning fallback only fires when `content` is empty.
        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            results.push(SseProcessResult::TextDelta(content));
        } else if let Some(reasoning) = choice
            .delta
            .reasoning_content
            .as_deref()
            .or(choice.delta.reasoning.as_deref())
            .filter(|r| !r.is_empty())
        {
            results.push(SseProcessResult::ThinkingDelta(reasoning.to_owned()));
        }

        // Handle tool call deltas
        if let Some(tc_deltas) = choice.delta.tool_calls {
            for tc in tc_deltas {
                results.push(SseProcessResult::ToolCallUpdate {
                    index: tc.index,
                    id: tc.id,
                    name: tc.function.as_ref().and_then(|f| f.name.clone()),
                    arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                });
            }
        }

        // Check for finish reason
        if let Some(finish_reason) = choice.finish_reason {
            results.push(SseProcessResult::Done(map_finish_reason(&finish_reason)));
        }
    }

    results
}

fn use_max_tokens_alias(base_url: &str) -> bool {
    base_url.contains("moonshot.ai")
        || base_url.contains("api.z.ai")
        || base_url.contains("minimax.io")
}

/// Every `OpenAI`-compatible endpoint accepts `stream_options.include_usage`;
/// requesting it everywhere ensures `OpenRouter` / `Baseten` / local streams
/// carry a usage frame so `total_usage` and downstream cost ledgers are
/// populated (issue #302), not just first-party `api.openai.com` turns.
const fn use_stream_usage_options(_base_url: &str) -> bool {
    true
}

/// `OpenRouter` requires a separate top-level `usage: { include: true }` flag
/// (distinct from `stream_options.include_usage`) to emit a usage frame.
fn use_openrouter_usage_options(base_url: &str) -> bool {
    base_url.contains("openrouter.ai")
}

/// Infer the stream stop reason when the provider never sent an explicit
/// `finish_reason` (truncated stream / EOF): a turn with accumulated tool
/// calls is a `ToolUse`, otherwise a plain `EndTurn`.
fn fallback_stream_stop_reason(
    tool_calls: &std::collections::HashMap<usize, ToolCallAccumulator>,
) -> StopReason {
    if tool_calls.is_empty() {
        StopReason::EndTurn
    } else {
        StopReason::ToolUse
    }
}

/// Map an HTTP status + body into a [`ChatOutcome`], parsing the success body
/// into a [`ChatResponse`].
fn decode_chat_response(
    status: StatusCode,
    bytes: &[u8],
    retry_after: Option<std::time::Duration>,
) -> Result<ChatOutcome> {
    if status == StatusCode::TOO_MANY_REQUESTS {
        return Ok(ChatOutcome::RateLimited(retry_after));
    }

    if status.is_server_error() {
        let body = String::from_utf8_lossy(bytes);
        log::error!("OpenAI server error status={status} body={body}");
        return Ok(ChatOutcome::ServerError(body.into_owned()));
    }

    if status.is_client_error() {
        let body = String::from_utf8_lossy(bytes);
        log::warn!("OpenAI client error status={status} body={body}");
        return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
    }

    let api_response: ApiChatResponse = serde_json::from_slice(bytes)
        .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

    let choice = api_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no choices in response"))?;

    let content = build_content_blocks(&choice.message);
    let stop_reason = choice.finish_reason.as_deref().map(map_finish_reason);

    Ok(ChatOutcome::Success(ChatResponse {
        id: api_response.id,
        content,
        model: api_response.model,
        stop_reason,
        usage: Usage {
            input_tokens: api_response.usage.prompt_tokens,
            output_tokens: api_response.usage.completion_tokens,
            cached_input_tokens: api_response
                .usage
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            cache_creation_input_tokens: 0,
        },
    }))
}

fn map_finish_reason(finish_reason: &str) -> StopReason {
    match finish_reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" | "network_error" => StopReason::StopSequence,
        "sensitive" => StopReason::Refusal,
        unknown => {
            log::debug!("Unknown finish_reason from OpenAI-compatible API: {unknown}");
            StopReason::StopSequence
        }
    }
}

fn build_api_reasoning(thinking: Option<&ThinkingConfig>) -> Option<ApiReasoning> {
    thinking
        .and_then(resolve_reasoning_effort)
        .map(|effort| ApiReasoning { effort })
}

const fn resolve_reasoning_effort(config: &ThinkingConfig) -> Option<ReasoningEffort> {
    if let Some(effort) = config.effort {
        return Some(map_effort(effort));
    }

    match &config.mode {
        ThinkingMode::Adaptive => None,
        ThinkingMode::Enabled { budget_tokens } => Some(map_budget_to_reasoning(*budget_tokens)),
    }
}

const fn map_effort(effort: Effort) -> ReasoningEffort {
    match effort {
        Effort::Low => ReasoningEffort::Low,
        Effort::Medium => ReasoningEffort::Medium,
        Effort::High => ReasoningEffort::High,
        Effort::Max => ReasoningEffort::XHigh,
    }
}

const fn map_budget_to_reasoning(budget_tokens: u32) -> ReasoningEffort {
    if budget_tokens <= 4_096 {
        ReasoningEffort::Low
    } else if budget_tokens <= 16_384 {
        ReasoningEffort::Medium
    } else if budget_tokens <= 32_768 {
        ReasoningEffort::High
    } else {
        ReasoningEffort::XHigh
    }
}

const fn api_role(role: agent_sdk_foundation::llm::Role) -> ApiRole {
    match role {
        agent_sdk_foundation::llm::Role::User => ApiRole::User,
        agent_sdk_foundation::llm::Role::Assistant => ApiRole::Assistant,
    }
}

/// Convert a `Content::Blocks` message into the `OpenAI` wire messages it maps
/// to, appending them to `messages`.
///
/// Tool results become standalone `tool` messages; text, tool calls and (on
/// assistant tool-call turns) echoed-back reasoning collapse into a single
/// message.
fn append_block_messages(
    messages: &mut Vec<ApiMessage>,
    role: agent_sdk_foundation::llm::Role,
    blocks: &[ContentBlock],
) {
    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::Thinking { thinking, .. } => {
                // DeepSeek-style thinking-mode multi-turn requires the prior
                // assistant reasoning_content to be echoed back on a tool-call
                // turn or the API 400s. Collected here; only carried into
                // reasoning_content below when this turn also has a tool call.
                thinking_parts.push(thinking.clone());
            }
            ContentBlock::RedactedThinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::Document { .. } => {
                // These blocks are not sent to the OpenAI API
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(ApiToolCall {
                    id: id.clone(),
                    r#type: "function".to_owned(),
                    function: ApiFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".to_owned()),
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
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
            // `ContentBlock` is `#[non_exhaustive]`; a block kind this SDK
            // version cannot represent is not sent to OpenAI.
            _ => log::warn!("Skipping unrecognized OpenAI content block"),
        }
    }

    let role = api_role(role);

    // reasoning_content is only echoed back on an assistant turn that ALSO
    // carries a tool call — the one case DeepSeek's thinking-mode protocol
    // requires it. Per that protocol legacy `deepseek-reasoner` 400s if
    // reasoning_content appears in input at all, and DeepSeek V4 thinking-mode
    // only needs it on tool-call turns. So a plain reasoning-only assistant
    // turn (no tool call) does NOT carry reasoning_content, and it is never
    // attached to user messages.
    let reasoning_content =
        if role == ApiRole::Assistant && !thinking_parts.is_empty() && !tool_calls.is_empty() {
            Some(thinking_parts.join("\n"))
        } else {
            None
        };

    // Add the message when it carries text, tool calls, or (for an assistant
    // turn) reasoning to echo back. Only emit if it's an assistant message or
    // has text content.
    let has_payload =
        !text_parts.is_empty() || !tool_calls.is_empty() || reasoning_content.is_some();
    if has_payload && (role == ApiRole::Assistant || !text_parts.is_empty()) {
        messages.push(ApiMessage {
            role,
            content: if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join("\n"))
            },
            reasoning_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        });
    }
}

fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
    let mut messages = Vec::new();

    // Add system message first (OpenAI uses a separate message for system prompt)
    if !request.system.is_empty() {
        messages.push(ApiMessage {
            role: ApiRole::System,
            content: Some(request.system.clone()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert SDK messages to OpenAI format
    for msg in &request.messages {
        match &msg.content {
            Content::Text(text) => {
                messages.push(ApiMessage {
                    role: api_role(msg.role),
                    content: Some(text.clone()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Content::Blocks(blocks) => append_block_messages(&mut messages, msg.role, blocks),
        }
    }

    messages
}

fn convert_tool(t: agent_sdk_foundation::llm::Tool) -> ApiTool {
    ApiTool {
        r#type: "function".to_owned(),
        function: ApiFunction {
            name: t.name,
            description: t.description,
            parameters: t.input_schema,
        },
    }
}

/// Non-empty reasoning text from an `OpenAI`-compatible response message, if any.
///
/// Prefers `DeepSeek`-style `reasoning_content`, falling back to the `reasoning`
/// field used by some `OpenRouter` upstreams.
fn reasoning_text(message: &ApiResponseMessage) -> Option<&str> {
    message
        .reasoning_content
        .as_deref()
        .or(message.reasoning.as_deref())
        .filter(|r| !r.is_empty())
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
    } else if let Some(reasoning) = reasoning_text(message) {
        // Reasoning-model fallback: when `content` is empty/absent but the model
        // produced reasoning tokens (DeepSeek-style answer-in-`reasoning_content`,
        // or any reasoning model truncated under a tight `max_tokens` before it
        // emitted visible content), surface the reasoning as a Thinking block so
        // the usable output is not silently dropped. This is a fallback only —
        // when `content` is present the reasoning is left untouched.
        blocks.push(ContentBlock::Thinking {
            thinking: reasoning.to_owned(),
            signature: None,
        });
    }

    // Add tool calls if present
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
                thought_signature: None,
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
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ApiResponseFormat>,
}

#[derive(Serialize)]
struct ApiChatRequestStreaming<'a> {
    model: &'a str,
    messages: &'a [ApiMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ApiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<ApiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ApiOpenRouterUsageOptions>,
    stream: bool,
}

/// `OpenAI` `tool_choice` wire format.
///
/// - `"auto"` — model decides.
/// - `{"type": "function", "function": {"name": "<name>"}}` — force a specific function.
#[derive(Serialize)]
#[serde(untagged)]
enum ApiToolChoice {
    String(String),
    Named {
        #[serde(rename = "type")]
        choice_type: String,
        function: ApiToolChoiceFunction,
    },
}

#[derive(Serialize)]
struct ApiToolChoiceFunction {
    name: String,
}

impl ApiToolChoice {
    fn from_tool_choice(tc: &agent_sdk_foundation::llm::ToolChoice) -> Self {
        match tc {
            agent_sdk_foundation::llm::ToolChoice::Auto => Self::String("auto".to_owned()),
            agent_sdk_foundation::llm::ToolChoice::Tool(name) => Self::Named {
                choice_type: "function".to_owned(),
                function: ApiToolChoiceFunction { name: name.clone() },
            },
        }
    }
}

/// `OpenAI` `response_format` wire format for structured outputs.
///
/// Emits `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`.
#[derive(Serialize)]
struct ApiResponseFormat {
    #[serde(rename = "type")]
    format_type: &'static str,
    json_schema: ApiJsonSchema,
}

#[derive(Serialize)]
struct ApiJsonSchema {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

impl ApiResponseFormat {
    fn from_response_format(rf: &agent_sdk_foundation::llm::ResponseFormat) -> Self {
        Self {
            format_type: "json_schema",
            json_schema: ApiJsonSchema {
                name: rf.name.clone(),
                schema: rf.schema.clone(),
                strict: rf.strict,
            },
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct ApiStreamOptions {
    include_usage: bool,
}

/// `OpenRouter`'s top-level usage-accounting flag (`usage: { include: true }`),
/// distinct from `stream_options.include_usage`.
#[derive(Clone, Copy, Serialize)]
struct ApiOpenRouterUsageOptions {
    include: bool,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

#[derive(Serialize)]
struct ApiReasoning {
    effort: ReasoningEffort,
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// `DeepSeek`-style thinking-mode multi-turn requires the prior assistant
    /// `reasoning_content` to be echoed back on a tool-call turn or the API
    /// rejects it (HTTP 400). Carried back only for assistant turns that had a
    /// Thinking block AND a tool call; omitted entirely otherwise (including
    /// reasoning-only turns, since legacy `deepseek-reasoner` 400s if
    /// `reasoning_content` appears in input) so the normal path is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
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
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiResponseToolCall>>,
    /// `DeepSeek`-style chain-of-thought, returned at the same level as
    /// `content` (`DeepSeek` V4 / some `OpenRouter` providers).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// `OpenRouter` normalizes reasoning under a `reasoning` field for some
    /// upstreams; treated as an equivalent fallback to `reasoning_content`.
    #[serde(default)]
    reasoning: Option<String>,
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
struct ApiUsage {
    #[serde(deserialize_with = "deserialize_u32_from_number")]
    prompt_tokens: u32,
    #[serde(deserialize_with = "deserialize_u32_from_number")]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<ApiPromptTokensDetails>,
}

#[derive(Deserialize)]
struct ApiPromptTokensDetails {
    #[serde(default, deserialize_with = "deserialize_u32_from_number")]
    cached_tokens: u32,
}

// ============================================================================
// SSE Streaming Types
// ============================================================================

/// Accumulator for tool call state across stream deltas.
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// A single chunk in `OpenAI`'s SSE stream.
#[derive(Deserialize)]
struct SseChunk {
    // A usage-only frame (OpenAI's trailing chunk, OpenRouter, etc.) may omit
    // `choices` entirely; without `default` it fails to deserialize and the
    // usage frame is dropped silently.
    #[serde(default)]
    choices: Vec<SseChoice>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

#[derive(Deserialize)]
struct SseChoice {
    delta: SseDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct SseDelta {
    content: Option<String>,
    tool_calls: Option<Vec<SseToolCallDelta>>,
    /// `DeepSeek`-style streamed chain-of-thought, returned at the same level as
    /// `content` (`DeepSeek` V4 / some `OpenRouter` providers).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// `OpenRouter` normalizes streamed reasoning under a `reasoning` field for
    /// some upstreams; treated as an equivalent fallback to `reasoning_content`.
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Deserialize)]
struct SseToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<SseFunctionDelta>,
}

#[derive(Deserialize)]
struct SseFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct SseUsage {
    #[serde(deserialize_with = "deserialize_u32_from_number")]
    prompt_tokens: u32,
    #[serde(deserialize_with = "deserialize_u32_from_number")]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<ApiPromptTokensDetails>,
}

fn deserialize_u32_from_number<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NumberLike {
        U64(u64),
        F64(f64),
    }

    match NumberLike::deserialize(deserializer)? {
        NumberLike::U64(v) => u32::try_from(v)
            .map_err(|_| D::Error::custom(format!("token count out of range for u32: {v}"))),
        NumberLike::F64(v) => {
            if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= f64::from(u32::MAX) {
                v.to_string().parse::<u32>().map_err(|e| {
                    D::Error::custom(format!(
                        "failed to convert integer-compatible token count {v} to u32: {e}"
                    ))
                })
            } else {
                Err(D::Error::custom(format!(
                    "token count must be a non-negative integer-compatible number, got {v}"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

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
    fn test_gpt52_thinking_factory_creates_provider() {
        let provider = OpenAIProvider::gpt52_thinking("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT52_THINKING);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt54_factory_creates_provider() {
        let provider = OpenAIProvider::gpt54("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT54);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt53_codex_factory_creates_provider() {
        let provider = OpenAIProvider::gpt53_codex("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_codex_factory_points_to_latest_codex_model() {
        let provider = OpenAIProvider::codex("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt5_factory_creates_gpt5_provider() {
        let provider = OpenAIProvider::gpt5("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT5);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt5_mini_factory_creates_provider() {
        let provider = OpenAIProvider::gpt5_mini("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT5_MINI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_o3_factory_creates_o3_provider() {
        let provider = OpenAIProvider::o3("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_O3);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_o4_mini_factory_creates_o4_mini_provider() {
        let provider = OpenAIProvider::o4_mini("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_O4_MINI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_o1_factory_creates_o1_provider() {
        let provider = OpenAIProvider::o1("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_O1);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_gpt41_factory_creates_gpt41_provider() {
        let provider = OpenAIProvider::gpt41("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GPT41);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_kimi_factory_creates_provider_with_kimi_base_url() {
        let provider = OpenAIProvider::kimi("test-api-key".to_string(), "kimi-custom".to_string());

        assert_eq!(provider.model(), "kimi-custom");
        assert_eq!(provider.base_url, BASE_URL_KIMI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_kimi_k2_5_factory_creates_provider() {
        let provider = OpenAIProvider::kimi_k2_5("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_KIMI_K2_5);
        assert_eq!(provider.base_url, BASE_URL_KIMI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_kimi_k2_thinking_factory_creates_provider() {
        let provider = OpenAIProvider::kimi_k2_thinking("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_KIMI_K2_THINKING);
        assert_eq!(provider.base_url, BASE_URL_KIMI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_zai_factory_creates_provider_with_zai_base_url() {
        let provider = OpenAIProvider::zai("test-api-key".to_string(), "glm-custom".to_string());

        assert_eq!(provider.model(), "glm-custom");
        assert_eq!(provider.base_url, BASE_URL_ZAI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_zai_glm5_factory_creates_provider() {
        let provider = OpenAIProvider::zai_glm5("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_ZAI_GLM5);
        assert_eq!(provider.base_url, BASE_URL_ZAI);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_minimax_factory_creates_provider_with_minimax_base_url() {
        let provider =
            OpenAIProvider::minimax("test-api-key".to_string(), "minimax-custom".to_string());

        assert_eq!(provider.model(), "minimax-custom");
        assert_eq!(provider.base_url, BASE_URL_MINIMAX);
        assert_eq!(provider.provider(), "openai");
    }

    #[test]
    fn test_minimax_m2_5_factory_creates_provider() {
        let provider = OpenAIProvider::minimax_m2_5("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_MINIMAX_M2_5);
        assert_eq!(provider.base_url, BASE_URL_MINIMAX);
        assert_eq!(provider.provider(), "openai");
    }

    // ===================
    // Model Constants Tests
    // ===================

    #[test]
    fn test_model_constants_have_expected_values() {
        // GPT-5.4 / GPT-5.3 Codex
        assert_eq!(MODEL_GPT54, "gpt-5.4");
        assert_eq!(MODEL_GPT53_CODEX, "gpt-5.3-codex");
        // GPT-5.2 series
        assert_eq!(MODEL_GPT52_INSTANT, "gpt-5.2-instant");
        assert_eq!(MODEL_GPT52_THINKING, "gpt-5.2-thinking");
        assert_eq!(MODEL_GPT52_PRO, "gpt-5.2-pro");
        assert_eq!(MODEL_GPT52_CODEX, "gpt-5.2-codex");
        // GPT-5 series
        assert_eq!(MODEL_GPT5, "gpt-5");
        assert_eq!(MODEL_GPT5_MINI, "gpt-5-mini");
        assert_eq!(MODEL_GPT5_NANO, "gpt-5-nano");
        // o-series
        assert_eq!(MODEL_O3, "o3");
        assert_eq!(MODEL_O3_MINI, "o3-mini");
        assert_eq!(MODEL_O4_MINI, "o4-mini");
        assert_eq!(MODEL_O1, "o1");
        assert_eq!(MODEL_O1_MINI, "o1-mini");
        // GPT-4.1 series
        assert_eq!(MODEL_GPT41, "gpt-4.1");
        assert_eq!(MODEL_GPT41_MINI, "gpt-4.1-mini");
        assert_eq!(MODEL_GPT41_NANO, "gpt-4.1-nano");
        // GPT-4o series
        assert_eq!(MODEL_GPT4O, "gpt-4o");
        assert_eq!(MODEL_GPT4O_MINI, "gpt-4o-mini");
        // OpenAI-compatible vendor defaults
        assert_eq!(MODEL_KIMI_K2_5, "kimi-k2.5");
        assert_eq!(MODEL_KIMI_K2_THINKING, "kimi-k2-thinking");
        assert_eq!(MODEL_ZAI_GLM5, "glm-5");
        assert_eq!(MODEL_MINIMAX_M2_5, "MiniMax-M2.5");
        assert_eq!(BASE_URL_KIMI, "https://api.moonshot.ai/v1");
        assert_eq!(BASE_URL_ZAI, "https://api.z.ai/api/paas/v4");
        assert_eq!(BASE_URL_MINIMAX, "https://api.minimax.io/v1");
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
            reasoning_content: None,
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
            reasoning_content: None,
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
            reasoning_content: None,
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
    fn test_api_response_with_unknown_finish_reason_deserialization() {
        let json = r#"{
            "id": "chatcmpl-789",
            "choices": [
                {
                    "message": {
                        "content": "ok"
                    },
                    "finish_reason": "vendor_custom_reason"
                }
            ],
            "model": "glm-5",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }"#;

        let response: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("vendor_custom_reason")
        );
        assert_eq!(
            map_finish_reason(response.choices[0].finish_reason.as_deref().unwrap()),
            StopReason::StopSequence
        );
    }

    #[test]
    fn test_map_finish_reason_covers_vendor_specific_values() {
        assert_eq!(map_finish_reason("stop"), StopReason::EndTurn);
        assert_eq!(map_finish_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(map_finish_reason("length"), StopReason::MaxTokens);
        assert_eq!(
            map_finish_reason("content_filter"),
            StopReason::StopSequence
        );
        assert_eq!(map_finish_reason("sensitive"), StopReason::Refusal);
        assert_eq!(map_finish_reason("network_error"), StopReason::StopSequence);
        assert_eq!(
            map_finish_reason("some_new_reason"),
            StopReason::StopSequence
        );
    }

    // ===================
    // Message Conversion Tests
    // ===================

    #[test]
    fn test_build_api_messages_with_system() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![agent_sdk_foundation::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
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
            messages: vec![agent_sdk_foundation::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        };

        let api_messages = build_api_messages(&request);
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, ApiRole::User);
    }

    fn request_with_messages(messages: Vec<agent_sdk_foundation::llm::Message>) -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages,
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        }
    }

    #[test]
    fn test_build_api_messages_echoes_assistant_reasoning_content_on_tool_call()
    -> anyhow::Result<()> {
        // DeepSeek V4 thinking-mode requires the prior assistant turn's
        // reasoning to be echoed back as `reasoning_content` ONLY on a turn
        // that also performed a tool call, or the API 400s.
        let request = request_with_messages(vec![
            agent_sdk_foundation::llm::Message::user("What is the weather?"),
            agent_sdk_foundation::llm::Message::assistant_with_content(vec![
                ContentBlock::Thinking {
                    thinking: "I should call the weather tool.".to_string(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Paris"}),
                    thought_signature: None,
                },
            ]),
        ]);

        let api_messages = build_api_messages(&request);
        let assistant = api_messages
            .iter()
            .find(|m| m.role == ApiRole::Assistant)
            .context("assistant message present")?;
        assert!(assistant.tool_calls.is_some());
        assert_eq!(
            assistant.reasoning_content,
            Some("I should call the weather tool.".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_build_api_messages_reasoning_content_serializes_on_tool_call_turn() -> anyhow::Result<()>
    {
        let request = request_with_messages(vec![
            agent_sdk_foundation::llm::Message::assistant_with_content(vec![
                ContentBlock::Thinking {
                    thinking: "thinking out loud".to_string(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "do_thing".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                },
            ]),
        ]);

        let api_messages = build_api_messages(&request);
        let json = serde_json::to_string(&api_messages).context("serialize api messages")?;
        assert!(json.contains("\"reasoning_content\":\"thinking out loud\""));
        Ok(())
    }

    #[test]
    fn test_build_api_messages_reasoning_only_turn_is_not_echoed() -> anyhow::Result<()> {
        // A reasoning-only assistant turn (no visible text, no tool call) must
        // NOT carry reasoning_content: legacy `deepseek-reasoner` 400s if
        // reasoning_content appears in input, and DeepSeek V4 thinking-mode only
        // needs it on tool-call turns. With no other payload the turn collapses
        // to nothing and is dropped entirely.
        let request = request_with_messages(vec![
            agent_sdk_foundation::llm::Message::assistant_with_content(vec![
                ContentBlock::Thinking {
                    thinking: "pondering".to_string(),
                    signature: None,
                },
            ]),
        ]);

        let api_messages = build_api_messages(&request);
        let json = serde_json::to_string(&api_messages).context("serialize api messages")?;
        assert!(!json.contains("reasoning_content"));
        assert!(api_messages.is_empty());
        Ok(())
    }

    #[test]
    fn test_build_api_messages_reasoning_with_text_no_tool_call_is_not_echoed() -> anyhow::Result<()>
    {
        // An assistant turn carrying reasoning + visible text but NO tool call
        // is emitted for its text, but its reasoning is NOT echoed back.
        let request = request_with_messages(vec![
            agent_sdk_foundation::llm::Message::user("What is 2+2?"),
            agent_sdk_foundation::llm::Message::assistant_with_content(vec![
                ContentBlock::Thinking {
                    thinking: "Let me add 2 and 2.".to_string(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "4".to_string(),
                },
            ]),
            agent_sdk_foundation::llm::Message::user("And 3+3?"),
        ]);

        let api_messages = build_api_messages(&request);
        let json = serde_json::to_string(&api_messages).context("serialize api messages")?;
        assert!(!json.contains("reasoning_content"));
        let assistant = api_messages
            .iter()
            .find(|m| m.role == ApiRole::Assistant)
            .context("assistant message present")?;
        assert_eq!(assistant.content, Some("4".to_string()));
        assert_eq!(assistant.reasoning_content, None);
        Ok(())
    }

    #[test]
    fn test_build_api_messages_normal_path_has_no_reasoning_content() -> anyhow::Result<()> {
        // Normal path unchanged: an assistant turn with no Thinking block must
        // not attach reasoning_content.
        let request = request_with_messages(vec![
            agent_sdk_foundation::llm::Message::user("hi"),
            agent_sdk_foundation::llm::Message::assistant_with_content(vec![ContentBlock::Text {
                text: "hello".to_string(),
            }]),
        ]);

        let api_messages = build_api_messages(&request);
        let json = serde_json::to_string(&api_messages).context("serialize api messages")?;
        assert!(!json.contains("reasoning_content"));
        let assistant = api_messages
            .iter()
            .find(|m| m.role == ApiRole::Assistant)
            .context("assistant message present")?;
        assert_eq!(assistant.reasoning_content, None);
        Ok(())
    }

    #[test]
    fn test_build_api_messages_does_not_attach_reasoning_to_user_blocks() {
        // A user turn carrying a Thinking block (unusual, but possible) must not
        // be turned into a reasoning_content echo.
        let request =
            request_with_messages(vec![agent_sdk_foundation::llm::Message::user_with_content(
                vec![
                    ContentBlock::Thinking {
                        thinking: "user-side thinking".to_string(),
                        signature: None,
                    },
                    ContentBlock::Text {
                        text: "question".to_string(),
                    },
                ],
            )]);

        let api_messages = build_api_messages(&request);
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, ApiRole::User);
        assert_eq!(api_messages[0].reasoning_content, None);
    }

    #[test]
    fn test_convert_tool() {
        let tool = agent_sdk_foundation::llm::Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Test Tool".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
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
            reasoning_content: None,
            reasoning: None,
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
            reasoning_content: None,
            reasoning: None,
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me help."));
        assert!(
            matches!(&blocks[1], ContentBlock::ToolUse { id, name, .. } if id == "call_123" && name == "read_file")
        );
    }

    #[test]
    fn test_build_content_blocks_falls_back_to_reasoning_content_when_content_empty() {
        // DeepSeek-style: answer / usable output arrives in reasoning_content
        // while content is null. Without the fallback this dropped all output.
        let message = ApiResponseMessage {
            content: None,
            tool_calls: None,
            reasoning_content: Some("The answer is 42.".to_string()),
            reasoning: None,
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 1);
        assert!(
            matches!(&blocks[0], ContentBlock::Thinking { thinking, signature } if thinking == "The answer is 42." && signature.is_none())
        );
    }

    #[test]
    fn test_build_content_blocks_falls_back_to_reasoning_field() {
        // Some OpenRouter upstreams normalize reasoning under `reasoning`.
        let message = ApiResponseMessage {
            content: Some(String::new()),
            tool_calls: None,
            reasoning_content: None,
            reasoning: Some("Considering options...".to_string()),
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 1);
        assert!(
            matches!(&blocks[0], ContentBlock::Thinking { thinking, .. } if thinking == "Considering options...")
        );
    }

    #[test]
    fn test_build_content_blocks_prefers_reasoning_content_over_reasoning() {
        let message = ApiResponseMessage {
            content: None,
            tool_calls: None,
            reasoning_content: Some("primary".to_string()),
            reasoning: Some("secondary".to_string()),
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 1);
        assert!(
            matches!(&blocks[0], ContentBlock::Thinking { thinking, .. } if thinking == "primary")
        );
    }

    #[test]
    fn test_build_content_blocks_does_not_add_reasoning_when_content_present() {
        // The normal content-present case must be unchanged: reasoning is NOT
        // surfaced as a Thinking block when there is usable text content.
        let message = ApiResponseMessage {
            content: Some("Final answer.".to_string()),
            tool_calls: None,
            reasoning_content: Some("internal chain of thought".to_string()),
            reasoning: None,
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Final answer."));
    }

    #[test]
    fn test_build_content_blocks_reasoning_fallback_with_tool_calls() {
        // Empty content + reasoning + a tool call: surface the reasoning AND the
        // tool call (reasoning model under tight max_tokens that still tool-called).
        let message = ApiResponseMessage {
            content: None,
            tool_calls: Some(vec![ApiResponseToolCall {
                id: "call_1".to_string(),
                function: ApiResponseFunctionCall {
                    name: "search".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            reasoning_content: Some("I should search.".to_string()),
            reasoning: None,
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 2);
        assert!(
            matches!(&blocks[0], ContentBlock::Thinking { thinking, .. } if thinking == "I should search.")
        );
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "search"));
    }

    #[test]
    fn test_build_content_blocks_empty_message_yields_no_blocks() {
        // Genuine truncation with no reasoning text: still produce nothing
        // (behavior unchanged for the empty case).
        let message = ApiResponseMessage {
            content: None,
            tool_calls: None,
            reasoning_content: None,
            reasoning: None,
        };

        let blocks = build_content_blocks(&message);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_api_response_message_deserializes_reasoning_content() {
        let json = r#"{
            "content": null,
            "reasoning_content": "step by step"
        }"#;

        let message: ApiResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(reasoning_text(&message), Some("step by step"));
        assert!(message.content.is_none());
    }

    // ===================
    // SSE Streaming Type Tests
    // ===================

    #[test]
    fn test_sse_chunk_text_delta_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_sse_chunk_tool_call_delta_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "function": {
                            "name": "read_file",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let tool_calls = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].id, Some("call_abc".to_string()));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name,
            Some("read_file".to_string())
        );
    }

    #[test]
    fn test_sse_chunk_tool_call_arguments_delta_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "{\"path\":"
                        }
                    }]
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let tool_calls = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id, None);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().arguments,
            Some("{\"path\":".to_string())
        );
    }

    #[test]
    fn test_sse_chunk_with_finish_reason_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_sse_chunk_with_usage_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
    }

    #[test]
    fn test_sse_chunk_with_float_usage_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100.0,
                "completion_tokens": 50.0
            }
        }"#;

        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
    }

    #[test]
    fn test_api_usage_deserializes_integer_compatible_numbers() {
        let json = r#"{
            "prompt_tokens": 42.0,
            "completion_tokens": 7
        }"#;

        let usage: ApiUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 7);
    }

    #[test]
    fn test_api_usage_deserializes_cached_tokens() {
        let json = r#"{
            "prompt_tokens": 42,
            "completion_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 10
            }
        }"#;

        let usage: ApiUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 10);
    }

    #[test]
    fn test_process_sse_data_maps_cached_tokens_to_cache_read_usage() {
        let results = process_sse_data(
            r#"{
                "choices": [],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {
                        "cached_tokens": 10
                    }
                }
            }"#,
        );

        assert!(matches!(
            results.as_slice(),
            [SseProcessResult::Usage(Usage {
                input_tokens: 42,
                output_tokens: 7,
                cached_input_tokens: 10,
                cache_creation_input_tokens: 0,
            })]
        ));
    }

    #[test]
    fn test_sse_delta_deserializes_reasoning_fields() -> anyhow::Result<()> {
        // The streaming delta struct must accept DeepSeek `reasoning_content`
        // and OpenRouter-normalized `reasoning` so reasoning tokens are not
        // dropped on deserialization.
        let chunk: SseChunk = serde_json::from_str(
            r#"{
                "choices": [{
                    "delta": {
                        "reasoning_content": "step one"
                    },
                    "finish_reason": null
                }]
            }"#,
        )
        .context("deserialize sse chunk")?;
        assert_eq!(
            chunk.choices[0].delta.reasoning_content,
            Some("step one".to_string())
        );
        assert!(chunk.choices[0].delta.content.is_none());
        Ok(())
    }

    #[test]
    fn test_process_sse_data_emits_thinking_delta_from_reasoning_content() {
        // Reasoning-model fallback under streaming: a delta whose visible
        // `content` is absent but whose `reasoning_content` carries tokens must
        // surface as a ThinkingDelta, mirroring the non-streaming fallback so the
        // output is not silently dropped.
        let results = process_sse_data(
            r#"{
                "choices": [{
                    "delta": { "reasoning_content": "thinking..." },
                    "finish_reason": null
                }]
            }"#,
        );

        assert!(matches!(
            results.as_slice(),
            [SseProcessResult::ThinkingDelta(text)] if text == "thinking..."
        ));
    }

    #[test]
    fn test_process_sse_data_emits_thinking_delta_from_reasoning_field() {
        // OpenRouter-normalized `reasoning` field is an equivalent fallback.
        let results = process_sse_data(
            r#"{
                "choices": [{
                    "delta": { "reasoning": "pondering" },
                    "finish_reason": null
                }]
            }"#,
        );

        assert!(matches!(
            results.as_slice(),
            [SseProcessResult::ThinkingDelta(text)] if text == "pondering"
        ));
    }

    #[test]
    fn test_process_sse_data_prefers_text_content_over_reasoning() {
        // When visible `content` is present, it takes precedence and the
        // reasoning fallback does not fire (mirrors non-streaming behavior).
        let results = process_sse_data(
            r#"{
                "choices": [{
                    "delta": {
                        "content": "answer",
                        "reasoning_content": "ignored"
                    },
                    "finish_reason": null
                }]
            }"#,
        );

        assert!(matches!(
            results.as_slice(),
            [SseProcessResult::TextDelta(text)] if text == "answer"
        ));
    }

    #[test]
    fn test_process_sse_data_empty_content_falls_back_to_reasoning() {
        // An explicitly empty `content` string must still trigger the reasoning
        // fallback rather than emitting an empty TextDelta.
        let results = process_sse_data(
            r#"{
                "choices": [{
                    "delta": {
                        "content": "",
                        "reasoning_content": "fallback"
                    },
                    "finish_reason": null
                }]
            }"#,
        );

        assert!(matches!(
            results.as_slice(),
            [SseProcessResult::ThinkingDelta(text)] if text == "fallback"
        ));
    }

    #[test]
    fn test_api_usage_rejects_fractional_numbers() {
        let json = r#"{
            "prompt_tokens": 42.5,
            "completion_tokens": 7
        }"#;

        let usage: std::result::Result<ApiUsage, _> = serde_json::from_str(json);
        assert!(usage.is_err());
    }

    #[test]
    fn test_use_max_tokens_alias_for_vendor_urls() {
        assert!(!use_max_tokens_alias(DEFAULT_BASE_URL));
        assert!(use_max_tokens_alias(BASE_URL_KIMI));
        assert!(use_max_tokens_alias(BASE_URL_ZAI));
        assert!(use_max_tokens_alias(BASE_URL_MINIMAX));
    }

    #[test]
    fn test_requires_responses_api_only_for_legacy_codex_model() {
        assert!(requires_responses_api(MODEL_GPT52_CODEX));
        assert!(!requires_responses_api(MODEL_GPT53_CODEX));
        assert!(!requires_responses_api(MODEL_GPT54));
    }

    #[test]
    fn test_should_use_responses_api_for_official_agentic_requests() {
        let request = ChatRequest {
            system: String::new(),
            messages: vec![agent_sdk_foundation::llm::Message::user("Hello")],
            tools: Some(vec![agent_sdk_foundation::llm::Tool {
                name: "read_file".to_string(),
                description: "Read a file".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
                display_name: "Read File".to_string(),
                tier: agent_sdk_foundation::ToolTier::Observe,
            }]),
            max_tokens: 1024,
            max_tokens_explicit: true,
            session_id: Some("thread-1".to_string()),
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        };

        assert!(should_use_responses_api(
            DEFAULT_BASE_URL,
            MODEL_GPT54,
            &request
        ));
        assert!(!should_use_responses_api(
            BASE_URL_KIMI,
            MODEL_GPT54,
            &request
        ));
    }

    #[test]
    fn test_build_api_reasoning_maps_enabled_budget_to_effort() {
        let reasoning = build_api_reasoning(Some(&ThinkingConfig::new(40_000))).unwrap();
        assert!(matches!(reasoning.effort, ReasoningEffort::XHigh));
    }

    #[test]
    fn test_build_api_reasoning_uses_explicit_effort() {
        let reasoning =
            build_api_reasoning(Some(&ThinkingConfig::adaptive_with_effort(Effort::High))).unwrap();
        assert!(matches!(reasoning.effort, ReasoningEffort::High));
    }

    #[test]
    fn test_build_api_reasoning_omits_adaptive_without_effort() {
        assert!(build_api_reasoning(Some(&ThinkingConfig::adaptive())).is_none());
    }

    #[test]
    fn test_openai_rejects_adaptive_thinking() {
        let provider = OpenAIProvider::gpt54("test-key".to_string());
        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("adaptive thinking is not supported")
        );
    }

    #[test]
    fn test_openai_non_reasoning_models_reject_thinking() {
        let provider = OpenAIProvider::gpt4o("test-key".to_string());
        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("thinking is not supported"));
    }

    #[test]
    fn test_request_serialization_openai_uses_max_completion_tokens_only() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequest {
            model: "gpt-4o",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(!json.contains("\"max_tokens\""));
    }

    #[test]
    fn test_request_serialization_with_max_tokens_alias() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequest {
            model: "glm-5",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: Some(1024),
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(json.contains("\"max_tokens\":1024"));
    }

    #[test]
    fn test_streaming_request_serialization_openai_default() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequestStreaming {
            model: "gpt-4o",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
            stream_options: Some(ApiStreamOptions {
                include_usage: true,
            }),
            usage: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(json.contains("\"stream_options\":{\"include_usage\":true}"));
        assert!(!json.contains("\"max_tokens\""));
    }

    #[test]
    fn stream_usage_is_requested_for_every_endpoint() {
        // issue #302: usage must be requested on ALL OpenAI-compatible
        // endpoints, not just api.openai.com, so OpenRouter/Baseten/local
        // turns report token usage to cost ledgers and budgets.
        assert!(use_stream_usage_options("https://api.openai.com/v1"));
        assert!(use_stream_usage_options("https://openrouter.ai/api/v1"));
        assert!(use_stream_usage_options("https://host.baseten.co/v1"));
        assert!(use_stream_usage_options("http://localhost:1234/v1"));
    }

    #[test]
    fn openrouter_usage_flag_only_for_openrouter() {
        assert!(use_openrouter_usage_options("https://openrouter.ai/api/v1"));
        assert!(!use_openrouter_usage_options("https://api.openai.com/v1"));
    }

    #[test]
    fn streaming_request_serializes_openrouter_usage_flag() -> anyhow::Result<()> {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("hi".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let request = ApiChatRequestStreaming {
            model: "anthropic/claude-3.5",
            messages: &messages,
            max_completion_tokens: Some(16),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
            stream_options: Some(ApiStreamOptions {
                include_usage: true,
            }),
            usage: Some(ApiOpenRouterUsageOptions { include: true }),
            stream: true,
        };
        let json = serde_json::to_string(&request)?;
        assert!(json.contains("\"usage\":{\"include\":true}"));
        assert!(json.contains("\"stream_options\":{\"include_usage\":true}"));
        Ok(())
    }

    #[test]
    fn usage_only_chunk_without_choices_deserializes() -> anyhow::Result<()> {
        // OpenAI's trailing usage frame (and some OpenRouter frames) omit
        // `choices` entirely; the chunk must still deserialize so the usage is
        // captured instead of being silently dropped (issue #302).
        let no_choices: SseChunk = serde_json::from_str("{}")?;
        assert!(no_choices.choices.is_empty());

        let usage_only: SseChunk =
            serde_json::from_str(r#"{"usage":{"prompt_tokens":10,"completion_tokens":5}}"#)?;
        assert!(usage_only.choices.is_empty());
        assert!(usage_only.usage.is_some());
        Ok(())
    }

    #[test]
    fn test_streaming_request_serialization_with_max_tokens_alias() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequestStreaming {
            model: "kimi-k2-thinking",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: Some(1024),
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
            stream_options: None,
            usage: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(!json.contains("\"stream_options\""));
    }

    #[test]
    fn test_request_serialization_includes_reasoning_when_present() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequest {
            model: MODEL_GPT54,
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: Some(ApiReasoning {
                effort: ReasoningEffort::High,
            }),
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"reasoning\":{\"effort\":\"high\"}"));
    }

    #[test]
    fn test_response_format_serializes_as_json_schema() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response_format = Some(ApiResponseFormat::from_response_format(
            &agent_sdk_foundation::llm::ResponseFormat::new(
                "person",
                serde_json::json!({"type": "object"}),
            ),
        ));

        let request = ApiChatRequest {
            model: "gpt-4o",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format,
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["name"], "person");
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            json["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn test_step_completion_stream_emits_trailing_usage_after_finish_reason() {
        // Official OpenAI with stream_options.include_usage sends the usage in a
        // SEPARATE chunk (choices: []) AFTER the finish_reason chunk, then [DONE].
        // The streaming loop must keep consuming past finish_reason so that usage
        // is captured and emitted (previously it returned early on Done, dropping
        // the usage entirely).
        let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
        let mut usage: Option<Usage> = None;
        let mut stop_reason: Option<StopReason> = None;

        // Chunk 1: text delta + finish_reason — must NOT finalize.
        let o1 = step_completion_stream(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            &mut tool_calls,
            &mut usage,
            &mut stop_reason,
        );
        assert!(o1.terminal.is_none());
        assert!(matches!(stop_reason, Some(StopReason::EndTurn)));

        // Chunk 2: usage-only trailing chunk (choices: []).
        let o2 = step_completion_stream(
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            &mut tool_calls,
            &mut usage,
            &mut stop_reason,
        );
        assert!(o2.terminal.is_none());

        // Chunk 3: [DONE] sentinel finalizes and must carry the trailing usage.
        let o3 = step_completion_stream("[DONE]", &mut tool_calls, &mut usage, &mut stop_reason);
        let terminal = o3.terminal.expect("[DONE] finalizes the stream");
        assert!(terminal.iter().any(|d| matches!(
            d,
            StreamDelta::Usage(Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..
            })
        )));
        assert!(terminal.iter().any(|d| matches!(
            d,
            StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn)
            }
        )));
    }

    #[test]
    fn test_response_format_omitted_when_absent() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequest {
            model: "gpt-4o",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("response_format"));
    }
}
