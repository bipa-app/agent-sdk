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
//! - the model only exists on the Responses surface (e.g. `gpt-5.3-codex`), or
//! - GPT-5.6 is used against the official `OpenAI` API with automatic routing, or
//! - the configured exact reasoning controls require the Responses API, or
//! - the request carries attachments (images / documents), or
//! - the request is *agentic* (has tools or tool-use/tool-result blocks) against
//!   the official `api.openai.com` base URL.
//!
//! The reroute forwards the provider's pooled HTTP client and `extra_headers`
//! (the BYOK / gateway auth mechanism) so a rerouted request keeps connection
//! reuse and authenticates identically to a non-rerouted one.

use crate::attachments::request_has_attachments;
use crate::model_features::{ModelApiSurface, get_model_features};
use crate::provider::LlmProvider;
use crate::streaming::{
    SseLineBuffer, StreamBox, StreamDelta, StreamErrorKind, reqwest_body_error_delta,
    reqwest_error_delta,
};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, StopReason, ThinkingConfig,
    ToolChoice, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::de::Error as _;
use serde::ser::SerializeStruct as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::openai_reasoning::{
    OpenAIAllowedToolsMode, OpenAIApiSurface, OpenAIPromptCacheMode, OpenAIPromptCacheTtl,
    OpenAIReasoningConfig, OpenAIReasoningEffort, OpenAITextVerbosity, OpenAIToolChoice,
    is_gpt56_model, legacy_reasoning_effort, validate_reasoning_config, validate_tool_choice,
};
use super::openai_responses::OpenAIResponsesProvider;
use super::openai_schema::normalize_strict_schema;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_RESPONSES_REASONING_PROVIDER: &str = "openai-responses";

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
        || get_model_features(model).is_some_and(|features| {
            features.api_surfaces.contains(&ModelApiSurface::Responses)
                && !features
                    .api_surfaces
                    .contains(&ModelApiSurface::ChatCompletions)
        })
}

fn is_official_openai_base_url(base_url: &str) -> bool {
    url::Url::parse(base_url).is_ok_and(|url| url.host_str() == Some("api.openai.com"))
}

fn chat_store(base_url: &str, config: Option<&OpenAIReasoningConfig>) -> Option<bool> {
    config
        .and_then(OpenAIReasoningConfig::store)
        .or_else(|| is_official_openai_base_url(base_url).then_some(false))
}

fn chat_prompt_cache_key<'a>(base_url: &str, session_id: Option<&'a str>) -> Option<&'a str> {
    is_official_openai_base_url(base_url)
        .then_some(session_id)
        .flatten()
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

fn request_has_openai_responses_history(request: &ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        matches!(
            &message.content,
            Content::Blocks(blocks)
                if blocks.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::OpaqueReasoning { provider, .. }
                            if provider == OPENAI_RESPONSES_REASONING_PROVIDER
                    )
                })
        )
    })
}

fn should_use_responses_api(
    base_url: &str,
    model: &str,
    request: &ChatRequest,
    reasoning: Option<&OpenAIReasoningConfig>,
) -> bool {
    let surface = reasoning.map_or(OpenAIApiSurface::Auto, OpenAIReasoningConfig::api_surface);
    let explicitly_requests_responses = matches!(surface, OpenAIApiSurface::Responses);
    let explicitly_requests_chat = matches!(surface, OpenAIApiSurface::ChatCompletions);
    let response_only_reasoning = !explicitly_requests_chat
        && reasoning.is_some_and(|config| {
            matches!(config.api_surface(), OpenAIApiSurface::Responses)
                || config.mode().is_some()
                || config.context().is_some()
                || config.summary().is_some()
        });
    let response_history_route =
        !explicitly_requests_chat && request_has_openai_responses_history(request);
    let official_auto_route = is_official_openai_base_url(base_url)
        && !explicitly_requests_chat
        && (is_gpt56_model(model) || request_is_agentic(request));

    let attachment_route = !explicitly_requests_chat
        && request_has_attachments(request)
        && (is_official_openai_base_url(base_url) || explicitly_requests_responses);

    requires_responses_api(model)
        || explicitly_requests_responses
        || response_only_reasoning
        || response_history_route
        || attachment_route
        || official_auto_route
}

// GPT-5.6 series
pub const MODEL_GPT56: &str = "gpt-5.6";
pub const MODEL_GPT56_SOL: &str = "gpt-5.6-sol";
pub const MODEL_GPT56_TERRA: &str = "gpt-5.6-terra";
pub const MODEL_GPT56_LUNA: &str = "gpt-5.6-luna";

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
    reasoning: Option<OpenAIReasoningConfig>,
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
            reasoning: None,
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
            reasoning: None,
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
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self.reasoning = None;
        self
    }

    /// Set exact `OpenAI` reasoning and related response controls.
    ///
    /// This is the lossless path for GPT-5.6 `none` / `xhigh` / `max`, pro
    /// mode, persisted reasoning context, summaries, and API-surface selection.
    /// Calling this after [`with_thinking`](Self::with_thinking) replaces the
    /// legacy provider-owned thinking configuration.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: OpenAIReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self.thinking = None;
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

    fn validate_chat_attachments(&self, request: &ChatRequest) -> Result<()> {
        if request_has_attachments(request) {
            anyhow::bail!(
                "OpenAI Chat Completions request for model={} contains image or document attachments that this provider cannot serialize; use the Responses API",
                self.model
            );
        }
        Ok(())
    }

    fn validate_chat_opaque_reasoning(request: &ChatRequest) -> Result<()> {
        if request_has_openai_responses_history(request) {
            anyhow::bail!(
                "OpenAI Responses reasoning history cannot be serialized through Chat Completions; use the Responses API"
            );
        }
        Ok(())
    }

    fn validate_requested_api_surface(&self) -> Result<()> {
        if self
            .reasoning
            .as_ref()
            .is_some_and(|config| matches!(config.api_surface(), OpenAIApiSurface::ChatCompletions))
            && requires_responses_api(&self.model)
        {
            anyhow::bail!(
                "model={} is only available through the OpenAI Responses API",
                self.model
            );
        }
        Ok(())
    }

    fn validate_chat_reasoning_controls(config: Option<&OpenAIReasoningConfig>) -> Result<()> {
        let Some(config) = config else {
            return Ok(());
        };
        if config.mode().is_some() || config.context().is_some() || config.summary().is_some() {
            anyhow::bail!(
                "OpenAI reasoning mode, context, and summary controls require the Responses API"
            );
        }
        Ok(())
    }

    fn validate_chat_response_format(
        &self,
        response_format: Option<&agent_sdk_foundation::llm::ResponseFormat>,
    ) -> Result<()> {
        if !is_official_openai_base_url(&self.base_url) {
            return Ok(());
        }
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

    fn resolve_chat_prompt_cache_options(
        &self,
        request: &ChatRequest,
        config: Option<&OpenAIReasoningConfig>,
    ) -> Result<ChatPromptCachePlan> {
        let exact_options = ApiPromptCacheOptions::from_config(config);
        if !is_gpt56_model(&self.model) && request.cache.is_some() {
            return Ok(ChatPromptCachePlan {
                options: exact_options,
                explicit_breakpoints: 0,
            });
        }
        let Some(cache) = request.cache.as_ref() else {
            return Ok(ChatPromptCachePlan {
                options: exact_options,
                explicit_breakpoints: 0,
            });
        };

        if let Some(ttl) = cache.ttl {
            anyhow::bail!(
                "OpenAI GPT-5.6 prompt caching supports only a 30m TTL; shared cache TTL {} cannot be represented for model={}",
                ttl.as_wire_str(),
                self.model
            );
        }

        if !cache.enabled {
            return Ok(ChatPromptCachePlan {
                options: Some(ApiPromptCacheOptions::new(OpenAIPromptCacheMode::Explicit)),
                explicit_breakpoints: 0,
            });
        }

        if let Some(max_breakpoints) = cache.max_breakpoints {
            return Ok(ChatPromptCachePlan {
                options: Some(ApiPromptCacheOptions {
                    mode: Some(OpenAIPromptCacheMode::Explicit),
                    ttl: config.and_then(OpenAIReasoningConfig::prompt_cache_ttl),
                }),
                explicit_breakpoints: usize::from(max_breakpoints.min(4)),
            });
        }

        Ok(ChatPromptCachePlan {
            options: exact_options,
            explicit_breakpoints: 0,
        })
    }

    fn resolve_chat_tool_choice(
        request_choice: Option<&ToolChoice>,
        config: Option<&OpenAIReasoningConfig>,
        tools: Option<&[agent_sdk_foundation::llm::Tool]>,
    ) -> Result<Option<ApiToolChoice>> {
        if let Some(choice) = request_choice {
            if let ToolChoice::Tool(name) = choice
                && !tools
                    .unwrap_or_default()
                    .iter()
                    .any(|tool| tool.name == *name)
            {
                anyhow::bail!("OpenAI tool_choice names unknown function `{name}`");
            }
            return Ok(Some(ApiToolChoice::from_tool_choice(choice)));
        }

        validate_tool_choice(config, tools)?;
        Ok(config
            .and_then(OpenAIReasoningConfig::tool_choice)
            .map(ApiToolChoice::from_openai_tool_choice))
    }

    async fn send_chat_request(
        &self,
        api_request: &ApiChatRequest<'_>,
    ) -> Result<(StatusCode, Vec<u8>, Option<std::time::Duration>)> {
        let builder = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
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
        if let Some(reasoning) = self.reasoning.clone() {
            provider = provider.with_reasoning(reasoning);
        }
        provider
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        if let Err(error) = self.validate_requested_api_surface() {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        // Route official OpenAI agentic flows to the Responses API, preserving
        // the pooled client and extra_headers (BYOK / gateway auth).
        if should_use_responses_api(
            &self.base_url,
            &self.model,
            &request,
            self.reasoning.as_ref(),
        ) {
            return self.responses_reroute().chat(request).await;
        }

        let reasoning_config = match self.resolve_openai_reasoning(request.thinking.as_ref()) {
            Ok(reasoning) => reasoning,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = Self::validate_chat_reasoning_controls(reasoning_config.as_ref()) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if let Err(error) = Self::validate_chat_opaque_reasoning(&request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if let Err(error) = self.validate_chat_attachments(&request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if let Err(error) = self.validate_chat_response_format(request.response_format.as_ref()) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let prompt_cache =
            match self.resolve_chat_prompt_cache_options(&request, reasoning_config.as_ref()) {
                Ok(plan) => plan,
                Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
            };
        let tool_choice = match Self::resolve_chat_tool_choice(
            request.tool_choice.as_ref(),
            reasoning_config.as_ref(),
            request.tools.as_deref(),
        ) {
            Ok(choice) => choice,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        let official_openai = is_official_openai_base_url(&self.base_url);
        let reasoning = build_chat_api_reasoning(
            reasoning_config.as_ref(),
            official_openai || self.reasoning.is_some(),
        );
        let verbosity = reasoning_config
            .as_ref()
            .and_then(OpenAIReasoningConfig::verbosity);
        let store = chat_store(&self.base_url, reasoning_config.as_ref());
        let parallel_tool_calls = reasoning_config
            .as_ref()
            .and_then(OpenAIReasoningConfig::parallel_tool_calls);
        let safety_identifier = reasoning_config
            .as_ref()
            .and_then(OpenAIReasoningConfig::safety_identifier);
        let prompt_cache_key = chat_prompt_cache_key(&self.base_url, request.session_id.as_deref());
        let max_tokens = self.effective_max_tokens(&request);
        let messages = build_api_messages_with_cache(&request, prompt_cache.explicit_breakpoints);
        let tools: Option<Vec<ApiTool>> = request.tools.map(|ts| {
            ts.into_iter()
                .map(|tool| convert_tool(tool, official_openai))
                .collect()
        });
        let response_format = request
            .response_format
            .as_ref()
            .map(|format| ApiResponseFormat::from_response_format(format, official_openai));

        let include_max_tokens_alias = use_max_tokens_alias(&self.base_url);
        let api_request = ApiChatRequest {
            model: &self.model,
            messages: &messages,
            max_completion_tokens: Some(max_tokens),
            max_tokens: include_max_tokens_alias.then_some(max_tokens),
            tools: tools.as_deref(),
            tool_choice,
            reasoning,
            response_format,
            verbosity,
            prompt_cache_options: prompt_cache.options,
            store,
            parallel_tool_calls,
            safety_identifier,
            prompt_cache_key,
        };

        log::debug!(
            "OpenAI LLM request model={} max_tokens={}",
            self.model,
            max_tokens
        );

        let (status, bytes, retry_after) = self.send_chat_request(&api_request).await?;

        log::debug!(
            "OpenAI LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        decode_chat_response(status, &bytes, retry_after)
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        if let Err(error) = self.validate_requested_api_surface() {
            return Box::pin(async_stream::stream! {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
            });
        }
        // Route official OpenAI agentic flows to the Responses API, preserving
        // the pooled client and extra_headers (BYOK / gateway auth).
        if should_use_responses_api(
            &self.base_url,
            &self.model,
            &request,
            self.reasoning.as_ref(),
        ) {
            let responses_provider = self.responses_reroute();
            return Box::pin(async_stream::stream! {
                let mut stream = std::pin::pin!(responses_provider.chat_stream(request));
                while let Some(item) = futures::StreamExt::next(&mut stream).await {
                    yield item;
                }
            });
        }

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
            if let Err(error) = Self::validate_chat_reasoning_controls(reasoning_config.as_ref()) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            if let Err(error) = Self::validate_chat_opaque_reasoning(&request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            if let Err(error) = self.validate_chat_attachments(&request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            if let Err(error) = self.validate_chat_response_format(request.response_format.as_ref()) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }
            let prompt_cache = match self
                .resolve_chat_prompt_cache_options(&request, reasoning_config.as_ref())
            {
                Ok(plan) => plan,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };
            let tool_choice = match Self::resolve_chat_tool_choice(
                request.tool_choice.as_ref(),
                reasoning_config.as_ref(),
                request.tools.as_deref(),
            ) {
                Ok(choice) => choice,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };
            let official_openai = is_official_openai_base_url(&self.base_url);
            let reasoning = build_chat_api_reasoning(
                reasoning_config.as_ref(),
                official_openai || self.reasoning.is_some(),
            );
            let verbosity = reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::verbosity);
            let store = chat_store(&self.base_url, reasoning_config.as_ref());
            let parallel_tool_calls = reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::parallel_tool_calls);
            let safety_identifier = reasoning_config
                .as_ref()
                .and_then(OpenAIReasoningConfig::safety_identifier);
            let prompt_cache_key =
                chat_prompt_cache_key(&self.base_url, request.session_id.as_deref());
            let max_tokens = self.effective_max_tokens(&request);
            let messages =
                build_api_messages_with_cache(&request, prompt_cache.explicit_breakpoints);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| {
                    ts.into_iter()
                        .map(|tool| convert_tool(tool, official_openai))
                        .collect()
                });
            let response_format = request
                .response_format
                .as_ref()
                .map(|format| ApiResponseFormat::from_response_format(format, official_openai));

            let include_max_tokens_alias = use_max_tokens_alias(&self.base_url);
            let include_stream_usage = use_stream_usage_options(&self.base_url);
            let include_openrouter_usage = use_openrouter_usage_options(&self.base_url);
            let api_request = ApiChatRequestStreaming {
                model: &self.model,
                messages: &messages,
                max_completion_tokens: Some(max_tokens),
                max_tokens: include_max_tokens_alias.then_some(max_tokens),
                tools: tools.as_deref(),
                tool_choice,
                reasoning,
                response_format,
                verbosity,
                prompt_cache_options: prompt_cache.options,
                store,
                parallel_tool_calls,
                safety_identifier,
                prompt_cache_key,
                stream_options: include_stream_usage.then_some(ApiStreamOptions {
                    include_usage: true,
                }),
                usage: include_openrouter_usage
                    .then_some(ApiOpenRouterUsageOptions { include: true }),
                stream: true,
            };

            log::debug!("OpenAI streaming LLM request model={} max_tokens={}", self.model, max_tokens);

            let stream_builder = self.client
                .post(format!("{}/chat/completions", self.base_url))
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
                let (kind, level) = if status == StatusCode::TOO_MANY_REQUESTS {
                    let retry_after = header_hint
                        .or_else(|| crate::retry_hints::openai_retry_delay(&body));
                    (StreamErrorKind::RateLimited(retry_after), "rate_limit")
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
                        // Usage can precede the transport failure and remains billable.
                        if let Some(usage) = usage.take() {
                            yield Ok(StreamDelta::Usage(usage));
                        }
                        yield Ok(reqwest_body_error_delta("stream error", &error));
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

            // A successful Chat Completions stream terminates with [DONE]. An
            // EOF before that sentinel is transport truncation, even if a
            // finish_reason happened to arrive first; synthesizing Done would
            // let callers commit a partial response as a successful turn.
            //
            // Still surface any usage the truncated stream reported first: the
            // provider billed it, and finalization (which would have yielded it)
            // never ran because [DONE] never arrived.
            if let Some(usage) = usage.take() {
                yield Ok(StreamDelta::Usage(usage));
            }
            yield Err(anyhow::anyhow!("OpenAI stream ended before [DONE] sentinel"));
        })
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
        let builder = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Content-Type", "application/json");
        let builder = self.apply_headers(builder);
        let body = crate::impls::model_listing::fetch_model_list_body(builder, "OpenAI").await?;
        parse_models_list(&body)
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

/// Parse the `OpenAI` `GET /models` response body into [`ModelInfo`] rows.
///
/// The Chat Completions list endpoint returns `{ "data": [{ "id", ... }] }`
/// and reports neither a display name nor token limits, so those fields stay
/// `None`. This shape is shared by the OpenAI-compatible vendor APIs.
fn parse_models_list(body: &str) -> Result<Vec<crate::provider::ModelInfo>> {
    #[derive(Deserialize)]
    struct ListResponse {
        #[serde(default)]
        data: Vec<ModelRow>,
    }
    #[derive(Deserialize)]
    struct ModelRow {
        id: String,
    }
    let parsed: ListResponse = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("failed to parse OpenAI models list: {e}"))?;
    Ok(parsed
        .data
        .into_iter()
        .map(|row| crate::provider::ModelInfo {
            id: row.id,
            display_name: None,
            context_window: None,
            max_output_tokens: None,
        })
        .collect())
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
            SseProcessResult::RefusalDelta(c) => {
                immediate.push(StreamDelta::TextDelta {
                    delta: c,
                    block_index: 0,
                });
                *stop_reason = Some(StopReason::Refusal);
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
            SseProcessResult::Done(sr) => {
                if !matches!(stop_reason, Some(StopReason::Refusal)) {
                    *stop_reason = Some(sr);
                }
            }
            SseProcessResult::Malformed(message) => {
                return SseLineOutcome {
                    immediate,
                    terminal: Some(terminal_error_deltas(
                        usage.take(),
                        message,
                        StreamErrorKind::ServerError,
                    )),
                };
            }
            SseProcessResult::Error { message, kind } => {
                return SseLineOutcome {
                    immediate,
                    terminal: Some(terminal_error_deltas(usage.take(), message, kind)),
                };
            }
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

    if matches!(stop_reason, StopReason::ToolUse) {
        // `idx` comes from the wire `tool_calls[].index`; use saturating_add so
        // a hostile `usize::MAX` index cannot overflow-panic in debug builds.
        // StreamAccumulator sorts by index so order stays stable.
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
    /// Emit structured refusal text and mark the turn as refused.
    RefusalDelta(String),
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
    /// The provider emitted a malformed JSON event.
    Malformed(String),
    /// The provider reported a failure in-band, on an HTTP-200 stream.
    Error {
        message: String,
        kind: StreamErrorKind,
    },
    /// Stream sentinel [DONE] was received.
    Sentinel,
}

/// Terminal deltas for a stream that ends in an error, emitting any usage the
/// stream reported first.
///
/// The provider bills the tokens it generated even when the turn then fails, so
/// the usage must reach the consumer's accumulator — which only ever sees
/// yielded deltas — ahead of the error that stops the stream.
fn terminal_error_deltas(
    usage: Option<Usage>,
    message: String,
    kind: StreamErrorKind,
) -> Vec<StreamDelta> {
    let mut deltas = Vec::new();
    if let Some(usage) = usage {
        deltas.push(StreamDelta::Usage(usage));
    }
    deltas.push(StreamDelta::Error { message, kind });
    deltas
}

/// Classify an in-band Chat Completions error object.
///
/// The stream already answered 200, so the status line says nothing about this
/// failure: the error object's own code is the only classification signal. The
/// message is prose — read only to recover a retry delay, never to decide the
/// category. An unrecognized code is treated as a transient server error, the
/// same conservative default a truncated stream gets.
fn completion_error_kind(code: Option<&serde_json::Value>, message: &str) -> StreamErrorKind {
    let http_code = code.and_then(serde_json::Value::as_u64);
    let symbolic = code.and_then(serde_json::Value::as_str);

    if http_code == Some(429)
        || matches!(symbolic, Some("rate_limit_exceeded" | "rate_limit_error"))
    {
        return StreamErrorKind::RateLimited(crate::retry_hints::openai_retry_delay(message));
    }
    // A non-429 client error is the caller's fault and will fail identically on
    // a retry; everything else (5xx, unknown, absent) stays retriable.
    if http_code.is_some_and(|code| (400..500).contains(&code)) {
        return StreamErrorKind::InvalidRequest;
    }
    StreamErrorKind::ServerError
}

/// Process an SSE data line and return results to apply.
fn process_sse_data(data: &str) -> Vec<SseProcessResult> {
    if data == "[DONE]" {
        return vec![SseProcessResult::Sentinel];
    }

    let chunk = match serde_json::from_str::<SseChunk>(data) {
        Ok(chunk) => chunk,
        Err(error) => {
            return vec![SseProcessResult::Malformed(format!(
                "invalid OpenAI Chat Completions stream event: {error}"
            ))];
        }
    };

    let mut results = Vec::new();

    // Usage is extracted before anything that can end the stream: a terminal
    // chunk may carry the billed usage *and* the failure together, and those
    // tokens are billed whether or not the turn survives.
    if let Some(u) = chunk.usage {
        results.push(SseProcessResult::Usage(Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: u
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            cache_creation_input_tokens: u
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cache_write_tokens),
        }));
    }

    // An in-band error ends the stream: the accompanying choice carries no
    // content (`finish_reason: "error"`), so it is reported instead of being
    // folded in as if the turn had produced output.
    if let Some(error) = chunk.error {
        let message = error
            .message
            .unwrap_or_else(|| "OpenAI-compatible stream reported an error".to_owned());
        let kind = completion_error_kind(error.code.as_ref(), &message);
        log::warn!("OpenAI in-band stream error kind={kind:?} message={message}");
        results.push(SseProcessResult::Error { message, kind });
        return results;
    }

    // Process choices
    if let Some(choice) = chunk.choices.into_iter().next() {
        let content = choice.delta.content.filter(|content| !content.is_empty());
        let refusal = choice.delta.refusal.filter(|refusal| !refusal.is_empty());
        let has_visible_output = content.is_some() || refusal.is_some();

        if let Some(content) = content {
            results.push(SseProcessResult::TextDelta(content));
        }
        if let Some(refusal) = refusal {
            results.push(SseProcessResult::RefusalDelta(refusal));
        }
        if !has_visible_output
            && let Some(reasoning) = choice
                .delta
                .reasoning_content
                .as_deref()
                .or(choice.delta.reasoning.as_deref())
                .filter(|reasoning| !reasoning.is_empty())
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

/// Infer the stop reason at a valid `[DONE]` sentinel when the provider omitted
/// an explicit `finish_reason`.
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
        let retry_after = retry_after
            .or_else(|| crate::retry_hints::openai_retry_delay(&String::from_utf8_lossy(bytes)));
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

    let stop_reason = if refusal_text(&choice.message).is_some() {
        Some(StopReason::Refusal)
    } else {
        choice.finish_reason.as_deref().map(map_finish_reason)
    };
    let mut content = build_content_blocks(&choice.message);
    if !matches!(stop_reason, Some(StopReason::ToolUse)) {
        content.retain(|block| !matches!(block, ContentBlock::ToolUse { .. }));
    }

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
            cache_creation_input_tokens: api_response
                .usage
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cache_write_tokens),
        },
    }))
}

fn map_finish_reason(finish_reason: &str) -> StopReason {
    match finish_reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" | "refusal" | "sensitive" => StopReason::Refusal,
        "network_error" => StopReason::Unknown,
        unknown => {
            log::debug!("Unknown finish_reason from OpenAI-compatible API: {unknown}");
            StopReason::Unknown
        }
    }
}

fn build_chat_api_reasoning(
    config: Option<&OpenAIReasoningConfig>,
    first_party_wire: bool,
) -> Option<ApiChatReasoning> {
    config
        .and_then(OpenAIReasoningConfig::effort)
        .map(|effort| {
            if first_party_wire {
                ApiChatReasoning {
                    reasoning_effort: Some(effort),
                    reasoning: None,
                }
            } else {
                ApiChatReasoning {
                    reasoning_effort: None,
                    reasoning: Some(ApiCompatibleReasoning { effort }),
                }
            }
        })
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
                    prompt_cache_breakpoint: false,
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
            prompt_cache_breakpoint: false,
        });
    }
}

#[cfg(test)]
fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
    build_api_messages_with_cache(request, 0)
}

fn build_api_messages_with_cache(
    request: &ChatRequest,
    explicit_cache_breakpoints: usize,
) -> Vec<ApiMessage> {
    let mut messages = Vec::new();

    // Add system message first (OpenAI uses a separate message for system prompt)
    if !request.system.is_empty() {
        messages.push(ApiMessage {
            role: ApiRole::System,
            content: Some(request.system.clone()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
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
                    prompt_cache_breakpoint: false,
                });
            }
            Content::Blocks(blocks) => append_block_messages(&mut messages, msg.role, blocks),
        }
    }

    apply_chat_cache_breakpoints(&mut messages, explicit_cache_breakpoints);
    messages
}

fn apply_chat_cache_breakpoints(messages: &mut [ApiMessage], max_breakpoints: usize) {
    if max_breakpoints == 0 {
        return;
    }

    let max_breakpoints = max_breakpoints.min(4);
    let message_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| message.content.as_ref().map(|_| index))
        .collect();

    let mut selected = Vec::with_capacity(max_breakpoints);
    // Chat Completions has no marker field on tool definitions. A breakpoint on
    // the system message still closes the prefix after tools + system, so it is
    // the most stable representable boundary before conversation-tail markers.
    if let Some(system_index) = message_indices
        .iter()
        .copied()
        .find(|index| messages[*index].role == ApiRole::System)
    {
        selected.push(system_index);
    }

    let remaining = max_breakpoints.saturating_sub(selected.len());
    let tail_candidates: Vec<usize> = message_indices
        .into_iter()
        .filter(|index| !selected.contains(index))
        .collect();
    let keep_from = tail_candidates.len().saturating_sub(remaining);
    selected.extend_from_slice(&tail_candidates[keep_from..]);

    for index in selected {
        messages[index].prompt_cache_breakpoint = true;
    }
}

fn convert_tool(t: agent_sdk_foundation::llm::Tool, normalize_for_openai_strict: bool) -> ApiTool {
    let mut parameters = t.input_schema;
    let strict =
        (normalize_for_openai_strict && normalize_strict_schema(&mut parameters)).then_some(true);
    ApiTool {
        r#type: "function".to_owned(),
        function: ApiFunction {
            name: t.name,
            description: t.description,
            parameters,
            strict,
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

fn refusal_text(message: &ApiResponseMessage) -> Option<&str> {
    message
        .refusal
        .as_deref()
        .filter(|refusal| !refusal.is_empty())
}

fn build_content_blocks(message: &ApiResponseMessage) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    let content = message
        .content
        .as_ref()
        .filter(|content| !content.is_empty());
    let refusal = refusal_text(message);
    let has_visible_output = content.is_some() || refusal.is_some();
    if let Some(content) = content {
        blocks.push(ContentBlock::Text {
            text: content.clone(),
        });
    }
    if let Some(refusal) = refusal {
        blocks.push(ContentBlock::Text {
            text: refusal.to_owned(),
        });
    }
    if !has_visible_output && let Some(reasoning) = reasoning_text(message) {
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
    #[serde(flatten)]
    reasoning: Option<ApiChatReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ApiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<OpenAITextVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<ApiPromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_identifier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
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
    #[serde(flatten)]
    reasoning: Option<ApiChatReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ApiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<OpenAITextVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<ApiPromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_identifier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
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
    Mode(&'static str),
    Named {
        #[serde(rename = "type")]
        choice_type: &'static str,
        function: ApiToolChoiceFunction,
    },
    AllowedTools {
        #[serde(rename = "type")]
        choice_type: &'static str,
        allowed_tools: ApiChatAllowedTools,
    },
}

#[derive(Serialize)]
struct ApiToolChoiceFunction {
    name: String,
}

#[derive(Serialize)]
struct ApiChatAllowedTools {
    mode: OpenAIAllowedToolsMode,
    tools: Vec<ApiChatAllowedTool>,
}

#[derive(Serialize)]
struct ApiChatAllowedTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: ApiToolChoiceFunction,
}

impl ApiToolChoice {
    fn from_tool_choice(tc: &ToolChoice) -> Self {
        match tc {
            ToolChoice::Auto => Self::Mode("auto"),
            ToolChoice::Tool(name) => Self::Named {
                choice_type: "function",
                function: ApiToolChoiceFunction { name: name.clone() },
            },
        }
    }

    fn from_openai_tool_choice(choice: &OpenAIToolChoice) -> Self {
        match choice {
            OpenAIToolChoice::None => Self::Mode("none"),
            OpenAIToolChoice::Auto => Self::Mode("auto"),
            OpenAIToolChoice::Required => Self::Mode("required"),
            OpenAIToolChoice::Function(name) => Self::Named {
                choice_type: "function",
                function: ApiToolChoiceFunction { name: name.clone() },
            },
            OpenAIToolChoice::AllowedTools { mode, tools } => Self::AllowedTools {
                choice_type: "allowed_tools",
                allowed_tools: ApiChatAllowedTools {
                    mode: *mode,
                    tools: tools
                        .iter()
                        .map(|name| ApiChatAllowedTool {
                            tool_type: "function",
                            function: ApiToolChoiceFunction { name: name.clone() },
                        })
                        .collect(),
                },
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
    fn from_response_format(
        rf: &agent_sdk_foundation::llm::ResponseFormat,
        normalize_for_openai_strict: bool,
    ) -> Self {
        let mut schema = rf.schema.clone();
        if rf.strict && normalize_for_openai_strict {
            let _ = normalize_strict_schema(&mut schema);
        }
        Self {
            format_type: "json_schema",
            json_schema: ApiJsonSchema {
                name: rf.name.clone(),
                schema,
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

#[derive(Serialize)]
struct ApiChatReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<OpenAIReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiCompatibleReasoning>,
}

#[derive(Serialize)]
struct ApiCompatibleReasoning {
    effort: OpenAIReasoningEffort,
}

#[derive(Clone, Copy, Serialize)]
struct ApiPromptCacheOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<OpenAIPromptCacheMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<OpenAIPromptCacheTtl>,
}

#[derive(Clone, Copy)]
struct ChatPromptCachePlan {
    options: Option<ApiPromptCacheOptions>,
    explicit_breakpoints: usize,
}

impl ApiPromptCacheOptions {
    const fn new(mode: OpenAIPromptCacheMode) -> Self {
        Self {
            mode: Some(mode),
            ttl: None,
        }
    }

    fn from_config(config: Option<&OpenAIReasoningConfig>) -> Option<Self> {
        let config = config?;
        let options = Self {
            mode: config.prompt_cache_mode(),
            ttl: config.prompt_cache_ttl(),
        };
        (options.mode.is_some() || options.ttl.is_some()).then_some(options)
    }
}

struct ApiMessage {
    role: ApiRole,
    content: Option<String>,
    /// `DeepSeek`-style thinking-mode multi-turn requires the prior assistant
    /// `reasoning_content` to be echoed back on a tool-call turn or the API
    /// rejects it (HTTP 400). Carried back only for assistant turns that had a
    /// Thinking block AND a tool call; omitted entirely otherwise (including
    /// reasoning-only turns, since legacy `deepseek-reasoner` 400s if
    /// `reasoning_content` appears in input) so the normal path is unchanged.
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ApiToolCall>>,
    tool_call_id: Option<String>,
    prompt_cache_breakpoint: bool,
}

impl Serialize for ApiMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut message = serializer.serialize_struct("ApiMessage", 5)?;
        message.serialize_field("role", &self.role)?;
        if let Some(content) = self.content.as_deref() {
            if self.prompt_cache_breakpoint {
                message.serialize_field(
                    "content",
                    &[ApiChatTextPart {
                        part_type: "text",
                        text: content,
                        prompt_cache_breakpoint: ApiChatPromptCacheBreakpoint {
                            mode: OpenAIPromptCacheMode::Explicit,
                        },
                    }],
                )?;
            } else {
                message.serialize_field("content", content)?;
            }
        }
        if let Some(reasoning_content) = self.reasoning_content.as_deref() {
            message.serialize_field("reasoning_content", reasoning_content)?;
        }
        if let Some(tool_calls) = self.tool_calls.as_ref() {
            message.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(tool_call_id) = self.tool_call_id.as_deref() {
            message.serialize_field("tool_call_id", tool_call_id)?;
        }
        message.end()
    }
}

#[derive(Serialize)]
struct ApiChatTextPart<'a> {
    #[serde(rename = "type")]
    part_type: &'static str,
    text: &'a str,
    prompt_cache_breakpoint: ApiChatPromptCacheBreakpoint,
}

#[derive(Clone, Copy, Serialize)]
struct ApiChatPromptCacheBreakpoint {
    mode: OpenAIPromptCacheMode,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
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
    #[serde(default)]
    refusal: Option<String>,
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
    #[serde(default, deserialize_with = "deserialize_u32_from_number")]
    cache_write_tokens: u32,
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
    /// In-band failure. `OpenAI`-compatible routes (`OpenRouter` among them)
    /// answer 200 and report the failure as a chunk carrying this object, so
    /// the HTTP status never sees it.
    #[serde(default)]
    error: Option<SseError>,
}

#[derive(Deserialize)]
struct SseError {
    #[serde(default)]
    message: Option<String>,
    /// Spelled as an HTTP number by some routes (`429`) and as a symbolic
    /// string by others (`"rate_limit_exceeded"`), so it is kept raw and
    /// interpreted in [`completion_error_kind`].
    #[serde(default)]
    code: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SseChoice {
    delta: SseDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct SseDelta {
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
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

    const OPENAI_MODELS_FIXTURE: &str = r#"{
      "object": "list",
      "data": [
        {"id": "gpt-5.4", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"}
      ]
    }"#;

    fn function_tool(name: &str) -> agent_sdk_foundation::llm::Tool {
        agent_sdk_foundation::llm::Tool {
            name: name.to_owned(),
            description: format!("Call {name}"),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: name.to_owned(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }
    }

    #[test]
    fn parse_models_list_reads_ids() -> anyhow::Result<()> {
        let models = parse_models_list(OPENAI_MODELS_FIXTURE)?;
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-5.4");
        assert_eq!(models[1].id, "gpt-4o");
        // The Chat Completions list endpoint reports no name or limits.
        assert_eq!(models[0].display_name, None);
        assert_eq!(models[0].context_window, None);
        Ok(())
    }

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
    fn test_gpt56_factories_create_expected_providers() {
        for (provider, expected_model) in [
            (
                OpenAIProvider::gpt56("test-api-key".to_string()),
                MODEL_GPT56,
            ),
            (
                OpenAIProvider::gpt56_sol("test-api-key".to_string()),
                MODEL_GPT56_SOL,
            ),
            (
                OpenAIProvider::gpt56_terra("test-api-key".to_string()),
                MODEL_GPT56_TERRA,
            ),
            (
                OpenAIProvider::gpt56_luna("test-api-key".to_string()),
                MODEL_GPT56_LUNA,
            ),
        ] {
            assert_eq!(provider.model(), expected_model);
            assert_eq!(provider.provider(), "openai");
            assert_eq!(provider.default_max_tokens(), 128_000);
        }
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
        // GPT-5.6 series
        assert_eq!(MODEL_GPT56, "gpt-5.6");
        assert_eq!(MODEL_GPT56_SOL, "gpt-5.6-sol");
        assert_eq!(MODEL_GPT56_TERRA, "gpt-5.6-terra");
        assert_eq!(MODEL_GPT56_LUNA, "gpt-5.6-luna");
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
            prompt_cache_breakpoint: false,
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
            prompt_cache_breakpoint: false,
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
            prompt_cache_breakpoint: false,
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
                strict: Some(true),
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
    fn structured_refusal_is_visible_and_sets_refusal_stop_reason() -> anyhow::Result<()> {
        let body = serde_json::json!({
            "id": "chatcmpl-refusal",
            "choices": [{
                "message": {
                    "content": null,
                    "refusal": "I cannot help with that."
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-5.6",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 6
            }
        });
        let bytes = serde_json::to_vec(&body)?;
        let outcome = decode_chat_response(StatusCode::OK, &bytes, None)?;
        let ChatOutcome::Success(response) = outcome else {
            anyhow::bail!("structured refusal did not decode as a successful provider response");
        };

        assert_eq!(response.first_text(), Some("I cannot help with that."));
        assert_eq!(response.stop_reason, Some(StopReason::Refusal));
        Ok(())
    }

    #[test]
    fn non_tool_terminal_suppresses_nonstream_tool_calls() -> anyhow::Result<()> {
        let body = serde_json::json!({
            "id": "chatcmpl-truncated-tool",
            "choices": [{
                "message": {
                    "content": "partial",
                    "tool_calls": [{
                        "id": "call_partial",
                        "type": "function",
                        "function": {
                            "name": "delete_record",
                            "arguments": "{\"id\":"
                        }
                    }]
                },
                "finish_reason": "length"
            }],
            "model": "gpt-5.6",
            "usage": {"prompt_tokens": 12, "completion_tokens": 6}
        });
        let bytes = serde_json::to_vec(&body)?;
        let ChatOutcome::Success(response) = decode_chat_response(StatusCode::OK, &bytes, None)?
        else {
            anyhow::bail!("truncated response did not decode as a successful provider response");
        };

        assert_eq!(response.stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(response.first_text(), Some("partial"));
        assert!(!response.has_tool_use());
        Ok(())
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
            StopReason::Unknown
        );
    }

    #[test]
    fn test_map_finish_reason_covers_vendor_specific_values() {
        assert_eq!(map_finish_reason("stop"), StopReason::EndTurn);
        assert_eq!(map_finish_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(map_finish_reason("length"), StopReason::MaxTokens);
        assert_eq!(map_finish_reason("content_filter"), StopReason::Refusal);
        assert_eq!(map_finish_reason("sensitive"), StopReason::Refusal);
        assert_eq!(map_finish_reason("refusal"), StopReason::Refusal);
        assert_eq!(map_finish_reason("network_error"), StopReason::Unknown);
        assert_eq!(map_finish_reason("some_new_reason"), StopReason::Unknown);
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
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}}
            }),
            display_name: "Test Tool".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        };

        let compatible_tool = convert_tool(tool.clone(), false);
        assert!(compatible_tool.function.strict.is_none());
        assert_eq!(
            compatible_tool.function.parameters,
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}}
            })
        );

        let api_tool = convert_tool(tool, true);
        assert_eq!(api_tool.r#type, "function");
        assert_eq!(api_tool.function.name, "test_tool");
        assert_eq!(api_tool.function.description, "A test tool");
        assert_eq!(api_tool.function.strict, Some(true));
        assert_eq!(api_tool.function.parameters["additionalProperties"], false);
    }

    #[test]
    fn test_build_content_blocks_text_only() {
        let message = ApiResponseMessage {
            content: Some("Hello!".to_string()),
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
            refusal: None,
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
    fn test_process_sse_data_maps_in_band_rate_limit_error_chunk() -> anyhow::Result<()> {
        // OpenRouter answers 200 and reports the failure as a chunk: the HTTP
        // status branch never runs, so the error object must be decoded here.
        let results = process_sse_data(
            r#"{"id":"gen-1","choices":[{"delta":{},"finish_reason":"error","index":0}],"error":{"code":429,"message":"Rate limited by upstream. Please try again in 12s."}}"#,
        );

        let kind = results
            .iter()
            .find_map(|result| match result {
                SseProcessResult::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected an in-band error result")?;

        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(12))),
            "an in-band 429 must be retriable and keep its parsed delay"
        );
        assert!(kind.is_recoverable());
        Ok(())
    }

    #[test]
    fn test_process_sse_data_keeps_usage_carried_by_an_error_chunk() -> anyhow::Result<()> {
        // OpenRouter's terminal chunk reports the billed usage *and* the failure
        // together. The tokens were billed, so they must survive the error —
        // and they must be emitted before it, since the error ends the stream.
        let results = process_sse_data(
            r#"{"choices":[{"delta":{},"finish_reason":"error","index":0}],"usage":{"prompt_tokens":140,"completion_tokens":20},"error":{"code":429,"message":"Rate limited. Please try again in 12s."}}"#,
        );

        let usage_at = results
            .iter()
            .position(|result| matches!(result, SseProcessResult::Usage(_)))
            .context("the error chunk's usage must not be dropped")?;
        let error_at = results
            .iter()
            .position(|result| matches!(result, SseProcessResult::Error { .. }))
            .context("expected an in-band error result")?;
        assert!(
            usage_at < error_at,
            "usage must be emitted before the terminal error"
        );

        let usage = results
            .iter()
            .find_map(|result| match result {
                SseProcessResult::Usage(usage) => Some(usage),
                _ => None,
            })
            .context("expected a usage result")?;
        assert_eq!(usage.input_tokens, 140);
        assert_eq!(usage.output_tokens, 20);

        // And the terminal deltas the stream hands over preserve that order.
        let mut tool_calls = HashMap::new();
        let mut stream_usage = None;
        let mut stop_reason = None;
        let outcome = step_completion_stream(
            r#"{"choices":[{"delta":{},"finish_reason":"error","index":0}],"usage":{"prompt_tokens":140,"completion_tokens":20},"error":{"code":429,"message":"Rate limited. Please try again in 12s."}}"#,
            &mut tool_calls,
            &mut stream_usage,
            &mut stop_reason,
        );
        let terminal = outcome.terminal.context("the error must be terminal")?;
        assert!(
            matches!(terminal.first(), Some(StreamDelta::Usage(usage)) if usage.input_tokens == 140),
            "the usage delta must lead the terminal sequence, got {terminal:?}"
        );
        assert!(matches!(
            terminal.last(),
            Some(StreamDelta::Error {
                kind: StreamErrorKind::RateLimited(Some(_)),
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn test_process_sse_data_maps_in_band_symbolic_and_server_error_chunks() -> anyhow::Result<()> {
        // Symbolic code (OpenAI-style) rather than an HTTP number.
        let symbolic = process_sse_data(
            r#"{"choices":[],"error":{"code":"rate_limit_exceeded","message":"Please try again in 250ms."}}"#,
        );
        let kind = symbolic
            .iter()
            .find_map(|result| match result {
                SseProcessResult::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected an in-band error result")?;
        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_millis(250)))
        );

        // An upstream 5xx stays retriable; a 4xx that is not a rate limit does not.
        let server = process_sse_data(
            r#"{"choices":[],"error":{"code":502,"message":"upstream unavailable"}}"#,
        );
        let server_kind = server
            .iter()
            .find_map(|result| match result {
                SseProcessResult::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected an in-band error result")?;
        assert_eq!(server_kind, StreamErrorKind::ServerError);

        let invalid =
            process_sse_data(r#"{"choices":[],"error":{"code":400,"message":"bad request"}}"#);
        let invalid_kind = invalid
            .iter()
            .find_map(|result| match result {
                SseProcessResult::Error { kind, .. } => Some(*kind),
                _ => None,
            })
            .context("expected an in-band error result")?;
        assert_eq!(invalid_kind, StreamErrorKind::InvalidRequest);
        assert!(!invalid_kind.is_recoverable());
        Ok(())
    }

    #[test]
    fn test_api_usage_deserializes_cache_read_and_write_tokens() -> anyhow::Result<()> {
        let json = r#"{
            "prompt_tokens": 42,
            "completion_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 10,
                "cache_write_tokens": 6
            }
        }"#;

        let usage: ApiUsage = serde_json::from_str(json)?;
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 7);
        let details = usage
            .prompt_tokens_details
            .context("prompt token details missing")?;
        assert_eq!(details.cached_tokens, 10);
        assert_eq!(details.cache_write_tokens, 6);
        Ok(())
    }

    #[test]
    fn test_process_sse_data_maps_cache_read_and_write_usage() {
        let results = process_sse_data(
            r#"{
                "choices": [],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {
                        "cached_tokens": 10,
                        "cache_write_tokens": 6
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
                cache_creation_input_tokens: 6,
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
    fn streamed_refusal_is_visible_and_preserves_refusal_stop_reason() -> anyhow::Result<()> {
        let mut tool_calls = HashMap::new();
        let mut usage = None;
        let mut stop_reason = None;
        let chunk = step_completion_stream(
            r#"{"choices":[{"delta":{"refusal":"Cannot comply."},"finish_reason":"stop"}]}"#,
            &mut tool_calls,
            &mut usage,
            &mut stop_reason,
        );

        assert!(matches!(
            chunk.immediate.as_slice(),
            [StreamDelta::TextDelta { delta, .. }] if delta == "Cannot comply."
        ));
        assert_eq!(stop_reason, Some(StopReason::Refusal));

        let done = step_completion_stream("[DONE]", &mut tool_calls, &mut usage, &mut stop_reason)
            .terminal
            .context("[DONE] did not finalize refusal stream")?;
        assert!(done.iter().any(|delta| matches!(
            delta,
            StreamDelta::Done {
                stop_reason: Some(StopReason::Refusal)
            }
        )));
        Ok(())
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
    fn test_requires_responses_api_for_codex_models() {
        assert!(requires_responses_api(MODEL_GPT52_CODEX));
        assert!(requires_responses_api(MODEL_GPT52_PRO));
        assert!(requires_responses_api(MODEL_GPT53_CODEX));
        assert!(!requires_responses_api(MODEL_GPT54));
        assert!(!requires_responses_api(MODEL_GPT56));
        assert!(!requires_responses_api(MODEL_GPT56_SOL));
        assert!(!requires_responses_api(MODEL_GPT56_TERRA));
        assert!(!requires_responses_api(MODEL_GPT56_LUNA));
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

        for model in [
            MODEL_GPT54,
            MODEL_GPT56,
            MODEL_GPT56_SOL,
            MODEL_GPT56_TERRA,
            MODEL_GPT56_LUNA,
        ] {
            assert!(should_use_responses_api(
                DEFAULT_BASE_URL,
                model,
                &request,
                None
            ));
        }
        assert!(!should_use_responses_api(
            BASE_URL_KIMI,
            MODEL_GPT54,
            &request,
            None,
        ));
    }

    #[test]
    fn official_gpt56_auto_routes_but_custom_base_url_does_not() {
        let request = ChatRequest::new(String::new(), vec![]);
        assert!(should_use_responses_api(
            DEFAULT_BASE_URL,
            MODEL_GPT56,
            &request,
            None,
        ));
        assert!(!should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT56,
            &request,
            None,
        ));
    }

    #[test]
    fn response_only_reasoning_controls_trigger_reroute() {
        let request = ChatRequest::new(String::new(), vec![]);
        let reasoning = OpenAIReasoningConfig::new()
            .with_mode(super::super::openai_reasoning::OpenAIReasoningMode::Pro);
        assert!(should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT56,
            &request,
            Some(&reasoning),
        ));

        let forced_chat = reasoning.with_api_surface(OpenAIApiSurface::ChatCompletions);
        assert!(!should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT56,
            &request,
            Some(&forced_chat),
        ));
        assert!(OpenAIProvider::validate_chat_reasoning_controls(Some(&forced_chat)).is_err());

        let codex = OpenAIProvider::gpt53_codex("test-key".to_owned()).with_reasoning(
            OpenAIReasoningConfig::new().with_api_surface(OpenAIApiSurface::ChatCompletions),
        );
        assert!(codex.validate_requested_api_surface().is_err());
    }

    #[test]
    fn openai_responses_history_reroutes_auto_and_forced_chat_rejects() {
        let request = ChatRequest::new(
            String::new(),
            vec![agent_sdk_foundation::llm::Message::assistant_with_content(
                vec![ContentBlock::OpaqueReasoning {
                    provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
                    data: serde_json::json!({
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "ciphertext"
                    }),
                }],
            )],
        );

        assert!(should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT54,
            &request,
            None,
        ));

        let forced_chat =
            OpenAIReasoningConfig::new().with_api_surface(OpenAIApiSurface::ChatCompletions);
        assert!(!should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT54,
            &request,
            Some(&forced_chat),
        ));
        assert!(OpenAIProvider::validate_chat_opaque_reasoning(&request).is_err());
    }

    #[test]
    fn exact_chat_tool_choice_maps_and_validates() -> anyhow::Result<()> {
        let tools = vec![function_tool("read_file")];

        let required = OpenAIReasoningConfig::new().with_tool_choice(OpenAIToolChoice::Required);
        let choice = OpenAIProvider::resolve_chat_tool_choice(None, Some(&required), Some(&tools))?
            .context("required tool choice was omitted")?;
        assert_eq!(serde_json::to_value(choice)?, serde_json::json!("required"));

        let forced = OpenAIReasoningConfig::new()
            .with_tool_choice(OpenAIToolChoice::Function("read_file".to_owned()));
        let choice = OpenAIProvider::resolve_chat_tool_choice(None, Some(&forced), Some(&tools))?
            .context("forced tool choice was omitted")?;
        assert_eq!(
            serde_json::to_value(choice)?,
            serde_json::json!({"type": "function", "function": {"name": "read_file"}})
        );

        let allowed =
            OpenAIReasoningConfig::new().with_tool_choice(OpenAIToolChoice::AllowedTools {
                mode: OpenAIAllowedToolsMode::Required,
                tools: vec!["read_file".to_owned()],
            });
        let choice = OpenAIProvider::resolve_chat_tool_choice(None, Some(&allowed), Some(&tools))?
            .context("allowed-tools choice was omitted")?;
        assert_eq!(
            serde_json::to_value(choice)?,
            serde_json::json!({
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [
                        {"type": "function", "function": {"name": "read_file"}}
                    ]
                }
            })
        );

        let missing = OpenAIReasoningConfig::new()
            .with_tool_choice(OpenAIToolChoice::Function("missing".to_owned()));
        assert!(
            OpenAIProvider::resolve_chat_tool_choice(None, Some(&missing), Some(&tools)).is_err()
        );
        assert!(OpenAIProvider::resolve_chat_tool_choice(None, Some(&required), None).is_err());
        Ok(())
    }

    #[test]
    fn generic_tool_choice_overrides_provider_exact_choice() -> anyhow::Result<()> {
        let tools = vec![function_tool("read_file")];
        let request_choice = ToolChoice::Tool("read_file".to_owned());
        let provider_choice =
            OpenAIReasoningConfig::new().with_tool_choice(OpenAIToolChoice::AllowedTools {
                mode: super::super::openai_reasoning::OpenAIAllowedToolsMode::Required,
                tools: vec!["missing".to_owned()],
            });
        let request = ChatRequest::new(String::new(), vec![])
            .with_tools(tools.clone())
            .with_tool_choice(request_choice.clone());

        assert!(!should_use_responses_api(
            "https://gateway.example/v1",
            MODEL_GPT54,
            &request,
            Some(&provider_choice),
        ));
        let choice = OpenAIProvider::resolve_chat_tool_choice(
            Some(&request_choice),
            Some(&provider_choice),
            Some(&tools),
        )?
        .context("generic tool choice was omitted")?;
        assert_eq!(
            serde_json::to_value(choice)?,
            serde_json::json!({"type": "function", "function": {"name": "read_file"}})
        );
        Ok(())
    }

    #[test]
    fn generic_cache_control_overrides_exact_defaults_and_rejects_unmappable_ttl()
    -> anyhow::Result<()> {
        use agent_sdk_foundation::llm::{CacheConfig, CacheTtl};

        let provider = OpenAIProvider::gpt56("test-key".to_owned());
        let exact = OpenAIReasoningConfig::new()
            .with_prompt_cache_mode(OpenAIPromptCacheMode::Explicit)
            .with_prompt_cache_ttl(OpenAIPromptCacheTtl::ThirtyMinutes);

        let inherited_plan = provider.resolve_chat_prompt_cache_options(
            &ChatRequest::new(String::new(), vec![]),
            Some(&exact),
        )?;
        assert_eq!(inherited_plan.explicit_breakpoints, 0);
        let inherited = inherited_plan
            .options
            .context("exact prompt-cache options were omitted")?;
        assert_eq!(
            serde_json::to_value(inherited)?,
            serde_json::json!({"mode": "explicit", "ttl": "30m"})
        );

        let enabled = ChatRequest::new(String::new(), vec![]).with_cache(CacheConfig::enabled());
        let enabled_plan = provider.resolve_chat_prompt_cache_options(&enabled, Some(&exact))?;
        assert_eq!(enabled_plan.explicit_breakpoints, 0);
        let enabled = enabled_plan
            .options
            .context("generic enabled cache options were omitted")?;
        assert_eq!(
            serde_json::to_value(enabled)?,
            serde_json::json!({"mode": "explicit", "ttl": "30m"})
        );

        let disabled = ChatRequest::new(String::new(), vec![]).with_cache(CacheConfig::disabled());
        let disabled_plan = provider.resolve_chat_prompt_cache_options(&disabled, Some(&exact))?;
        assert_eq!(disabled_plan.explicit_breakpoints, 0);
        let disabled = disabled_plan
            .options
            .context("generic disabled cache options were omitted")?;
        assert_eq!(
            serde_json::to_value(disabled)?,
            serde_json::json!({"mode": "explicit"})
        );

        let zero_breakpoints = ChatRequest::new(String::new(), vec![])
            .with_cache(CacheConfig::enabled().with_max_breakpoints(0));
        let zero_breakpoints_plan =
            provider.resolve_chat_prompt_cache_options(&zero_breakpoints, Some(&exact))?;
        assert_eq!(zero_breakpoints_plan.explicit_breakpoints, 0);
        let zero_breakpoints = zero_breakpoints_plan
            .options
            .context("zero-breakpoint cache options were omitted")?;
        assert_eq!(
            serde_json::to_value(zero_breakpoints)?,
            serde_json::json!({"mode": "explicit", "ttl": "30m"})
        );

        let two_breakpoints = ChatRequest::new(String::new(), vec![])
            .with_cache(CacheConfig::enabled().with_max_breakpoints(2));
        let two_breakpoints =
            provider.resolve_chat_prompt_cache_options(&two_breakpoints, Some(&exact))?;
        assert_eq!(two_breakpoints.explicit_breakpoints, 2);

        for ttl in [CacheTtl::FiveMinutes, CacheTtl::OneHour] {
            for cache in [
                CacheConfig::enabled().with_ttl(ttl),
                CacheConfig::disabled().with_ttl(ttl),
            ] {
                let request = ChatRequest::new(String::new(), vec![]).with_cache(cache);
                let error = provider
                    .resolve_chat_prompt_cache_options(&request, Some(&exact))
                    .err()
                    .context("shared cache TTL was silently accepted")?;
                assert!(error.to_string().contains("supports only a 30m TTL"));
            }
        }
        Ok(())
    }

    #[test]
    fn chat_cache_breakpoints_prioritize_system_then_conversation_tail() -> anyhow::Result<()> {
        let request = ChatRequest::new(
            "system",
            vec![
                agent_sdk_foundation::llm::Message::user("one"),
                agent_sdk_foundation::llm::Message::assistant("two"),
                agent_sdk_foundation::llm::Message::user("three"),
                agent_sdk_foundation::llm::Message::assistant("four"),
                agent_sdk_foundation::llm::Message::user("five"),
            ],
        );
        let json = serde_json::to_value(build_api_messages_with_cache(&request, 9))?;
        let messages = json.as_array().context("messages must be an array")?;
        let marked = messages
            .iter()
            .filter(|message| message["content"].is_array())
            .count();
        assert_eq!(marked, 4);
        assert_eq!(messages[0]["content"][0]["text"], "system");
        assert_eq!(
            messages[0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
        assert_eq!(messages[1]["content"], "one");
        assert_eq!(messages[2]["content"], "two");
        assert_eq!(messages[5]["content"][0]["text"], "five");
        assert_eq!(
            messages[5]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );

        let one_breakpoint = serde_json::to_value(build_api_messages_with_cache(&request, 1))?;
        let messages = one_breakpoint
            .as_array()
            .context("messages must be an array")?;
        assert!(messages[0]["content"].is_array());
        assert!(messages[5]["content"].is_string());
        Ok(())
    }

    #[test]
    fn chat_strict_response_format_rejects_free_form_objects() -> anyhow::Result<()> {
        let format = agent_sdk_foundation::llm::ResponseFormat::new(
            "freeform",
            serde_json::json!({"type": "object"}),
        );
        assert!(
            OpenAIProvider::gpt54("test-key".to_owned())
                .validate_chat_response_format(Some(&format))
                .is_err()
        );
        assert!(
            OpenAIProvider::with_base_url("test-key", MODEL_GPT54, "https://gateway.example/v1",)
                .validate_chat_response_format(Some(&format))
                .is_ok()
        );

        let structured = agent_sdk_foundation::llm::ResponseFormat::new(
            "structured",
            serde_json::json!({
                "type": "object",
                "properties": {"optional_name": {"type": "string"}}
            }),
        );
        let compatible = ApiResponseFormat::from_response_format(&structured, false);
        let compatible_json = serde_json::to_value(compatible)?;
        assert!(
            compatible_json["json_schema"]["schema"]
                .get("required")
                .is_none()
        );

        let official = ApiResponseFormat::from_response_format(&structured, true);
        let official_json = serde_json::to_value(official)?;
        assert_eq!(
            official_json["json_schema"]["schema"]["required"],
            serde_json::json!(["optional_name"])
        );
        Ok(())
    }

    #[test]
    fn forced_or_custom_chat_rejects_unserialized_attachments() {
        let request = ChatRequest::new(
            String::new(),
            vec![agent_sdk_foundation::llm::Message::user_with_content(vec![
                ContentBlock::Image {
                    source: agent_sdk_foundation::llm::ContentSource::new("image/png", "aGVsbG8="),
                },
            ])],
        );
        let forced_chat =
            OpenAIReasoningConfig::new().with_api_surface(OpenAIApiSurface::ChatCompletions);

        assert!(!should_use_responses_api(
            DEFAULT_BASE_URL,
            MODEL_GPT54,
            &request,
            Some(&forced_chat),
        ));
        assert!(
            OpenAIProvider::gpt54("test-key".to_owned())
                .with_reasoning(forced_chat)
                .validate_chat_attachments(&request)
                .is_err()
        );
        assert!(
            OpenAIProvider::with_base_url("test-key", MODEL_GPT54, "https://gateway.example/v1",)
                .validate_chat_attachments(&request)
                .is_err()
        );
        assert!(should_use_responses_api(
            DEFAULT_BASE_URL,
            MODEL_GPT54,
            &request,
            None,
        ));
    }

    #[test]
    fn legacy_budget_and_effort_mapping_remains_compatible() {
        assert_eq!(
            legacy_reasoning_effort(&ThinkingConfig::new(40_000)),
            Some(OpenAIReasoningEffort::XHigh)
        );
        assert_eq!(
            legacy_reasoning_effort(&ThinkingConfig::adaptive_with_effort(
                agent_sdk_foundation::llm::Effort::High,
            )),
            Some(OpenAIReasoningEffort::High)
        );
        assert_eq!(legacy_reasoning_effort(&ThinkingConfig::adaptive()), None);
    }

    #[test]
    fn effective_max_tokens_uses_model_default_unless_explicit() {
        let provider = OpenAIProvider::gpt56("test-key".to_string());
        let implicit = ChatRequest::new(String::new(), vec![]);
        assert_eq!(provider.effective_max_tokens(&implicit), 128_000);

        let explicit = implicit.with_max_tokens(8_192);
        assert_eq!(provider.effective_max_tokens(&explicit), 8_192);
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
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: Some(false),
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(!json.contains("\"max_tokens\""));
        assert!(json.contains("\"store\":false"));
    }

    #[test]
    fn custom_chat_omits_first_party_storage_and_cache_defaults() {
        let explicit_store = OpenAIReasoningConfig::new().with_store(true);

        assert_eq!(chat_store(DEFAULT_BASE_URL, None), Some(false));
        assert_eq!(
            chat_store("https://gateway.example/v1", Some(&explicit_store)),
            Some(true)
        );
        assert_eq!(chat_store("https://gateway.example/v1", None), None);
        assert_eq!(
            chat_prompt_cache_key(DEFAULT_BASE_URL, Some("thread-42")),
            Some("thread-42")
        );
        assert_eq!(
            chat_prompt_cache_key("https://gateway.example/v1", Some("thread-42")),
            None
        );
    }

    #[test]
    fn official_openai_url_detection_uses_the_exact_parsed_host() {
        assert!(is_official_openai_base_url(DEFAULT_BASE_URL));
        assert!(is_official_openai_base_url(
            "https://api.openai.com/custom-prefix/v1"
        ));
        assert!(!is_official_openai_base_url(
            "https://api.openai.com.attacker.example/v1"
        ));
        assert!(!is_official_openai_base_url(
            "https://gateway.example/v1?upstream=api.openai.com"
        ));
        assert!(!is_official_openai_base_url("not a URL api.openai.com"));
    }

    #[test]
    fn test_request_serializes_openai_application_controls() -> anyhow::Result<()> {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_owned()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
        }];
        let request = ApiChatRequest {
            model: MODEL_GPT56,
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            response_format: None,
            verbosity: None,
            prompt_cache_options: None,
            store: Some(true),
            parallel_tool_calls: Some(false),
            safety_identifier: Some("safety-user-42"),
            prompt_cache_key: Some("thread-42"),
        };

        let json = serde_json::to_value(request)?;
        assert_eq!(json["store"], true);
        assert_eq!(json["parallel_tool_calls"], false);
        assert_eq!(json["safety_identifier"], "safety-user-42");
        assert_eq!(json["prompt_cache_key"], "thread-42");
        Ok(())
    }

    #[test]
    fn test_request_serialization_with_max_tokens_alias() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(!json.contains("\"store\""));
    }

    #[test]
    fn test_streaming_request_serialization_openai_default() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: Some(false),
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
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
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
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
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
            stream_options: None,
            usage: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_completion_tokens\":1024"));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(!json.contains("\"stream_options\""));
        assert!(!json.contains("\"store\""));
    }

    #[test]
    fn test_request_serialization_uses_top_level_reasoning_effort() -> anyhow::Result<()> {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
        }];

        let request = ApiChatRequest {
            model: MODEL_GPT54,
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: build_chat_api_reasoning(
                Some(&OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::High)),
                true,
            ),
            response_format: None,
            verbosity: None,
            prompt_cache_options: None,
            store: Some(false),
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
        };

        let json = serde_json::to_value(&request)?;
        assert_eq!(json["reasoning_effort"], "high");
        assert!(json.get("reasoning").is_none());
        Ok(())
    }

    #[test]
    fn test_compatible_request_serialization_preserves_nested_reasoning() -> anyhow::Result<()> {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
        }];
        let reasoning = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::High);

        let request = ApiChatRequest {
            model: "compatible-model",
            messages: &messages,
            max_completion_tokens: Some(1024),
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: build_chat_api_reasoning(Some(&reasoning), false),
            response_format: None,
            verbosity: None,
            prompt_cache_options: None,
            store: None,
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
        };

        let json = serde_json::to_value(&request)?;
        assert_eq!(json["reasoning"]["effort"], "high");
        assert!(json.get("reasoning_effort").is_none());
        Ok(())
    }

    #[test]
    fn test_response_format_serializes_as_json_schema() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
        }];

        let response_format = Some(ApiResponseFormat::from_response_format(
            &agent_sdk_foundation::llm::ResponseFormat::new(
                "person",
                serde_json::json!({"type": "object"}),
            ),
            true,
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
            verbosity: None,
            prompt_cache_options: None,
            store: Some(false),
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
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
    fn malformed_chat_sse_is_an_immediate_server_error() -> anyhow::Result<()> {
        let mut tool_calls = HashMap::new();
        let mut usage = None;
        let mut stop_reason = None;
        let outcome = step_completion_stream(
            "{not valid json",
            &mut tool_calls,
            &mut usage,
            &mut stop_reason,
        );
        let SseLineOutcome {
            immediate,
            terminal,
        } = outcome;
        let terminal = terminal.context("malformed SSE event did not terminate the stream")?;

        assert!(immediate.is_empty());
        assert!(matches!(
            terminal.as_slice(),
            [StreamDelta::Error {
                kind: StreamErrorKind::ServerError,
                ..
            }]
        ));
        Ok(())
    }

    #[test]
    fn non_tool_stream_terminal_suppresses_accumulated_tool_calls() {
        let tool_calls = HashMap::from([(
            0,
            ToolCallAccumulator {
                id: "call_partial".to_owned(),
                name: "delete_record".to_owned(),
                arguments: "{\"id\":".to_owned(),
            },
        )]);

        let truncated = build_stream_end_deltas(&tool_calls, None, StopReason::MaxTokens);
        assert!(!truncated.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolUseStart { .. } | StreamDelta::ToolInputDelta { .. }
        )));

        let tool_terminal = build_stream_end_deltas(&tool_calls, None, StopReason::ToolUse);
        assert!(
            tool_terminal
                .iter()
                .any(|delta| matches!(delta, StreamDelta::ToolUseStart { .. }))
        );
    }

    #[test]
    fn test_response_format_omitted_when_absent() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            prompt_cache_breakpoint: false,
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
            verbosity: None,
            prompt_cache_options: None,
            store: Some(false),
            parallel_tool_calls: None,
            safety_identifier: None,
            prompt_cache_key: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("response_format"));
    }

    #[tokio::test]
    async fn stream_eof_before_done_is_an_error() -> anyhow::Result<()> {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url("test-key", MODEL_GPT4O, server.uri());
        let request = ChatRequest::new(
            String::new(),
            vec![agent_sdk_foundation::llm::Message::user("hello")],
        );
        let items = provider.chat_stream(request).collect::<Vec<_>>().await;

        assert!(items.iter().any(|item| matches!(
            item,
            Ok(StreamDelta::TextDelta { delta, .. }) if delta == "partial"
        )));
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Ok(StreamDelta::Done { .. })))
        );
        let last = items.last().context("stream emitted no events")?;
        let Err(error) = last else {
            anyhow::bail!("truncated stream did not end with an error");
        };
        assert!(error.to_string().contains("before [DONE] sentinel"));
        Ok(())
    }

    #[tokio::test]
    async fn stream_usage_before_eof_is_surfaced_ahead_of_the_error() -> anyhow::Result<()> {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // A trailing usage-only chunk (choices: []) arrives, then the stream
        // ends before [DONE] — transport truncation. The provider billed those
        // tokens, so the Usage must be surfaced before the transport error.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":140,\"completion_tokens\":20}}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url("test-key", MODEL_GPT4O, server.uri());
        let request = ChatRequest::new(
            String::new(),
            vec![agent_sdk_foundation::llm::Message::user("hello")],
        );
        let items = provider.chat_stream(request).collect::<Vec<_>>().await;

        let usage_at = items
            .iter()
            .position(|item| matches!(item, Ok(StreamDelta::Usage(_))))
            .context("the truncated stream's usage must not be dropped")?;
        let error_at = items
            .iter()
            .position(Result::is_err)
            .context("a truncated stream must end with an error")?;
        assert!(
            usage_at < error_at,
            "usage must be surfaced before the transport error, got {items:?}"
        );
        let Ok(StreamDelta::Usage(usage)) = &items[usage_at] else {
            anyhow::bail!("expected a usage delta");
        };
        assert_eq!(usage.input_tokens, 140);
        assert_eq!(usage.output_tokens, 20);
        Ok(())
    }
}
