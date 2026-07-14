//! Anthropic API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Anthropic
//! Messages API using reqwest for HTTP calls. Supports both streaming and
//! non-streaming responses.

pub(crate) mod data;

use crate::attachments::validate_request_attachments;
use crate::provider::{LlmProvider, thinking_for_forced_tool};
use crate::streaming::{
    StreamBox, StreamDelta, StreamErrorKind, reqwest_body_error_delta, reqwest_error_delta,
};
use agent_sdk_foundation::llm::{
    CacheTtl, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, ThinkingConfig, ThinkingMode,
    Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use data::{
    ApiMessagesRequest, ApiOutputConfig, ApiThinkingConfig, ApiToolChoice, build_api_messages,
    build_api_tools_with_cache, is_message_stop_event, map_content_blocks, map_stop_reason,
    parse_sse_event, take_next_sse_event,
};
use futures::StreamExt;
use reqwest::StatusCode;

const API_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const CLAUDE_CODE_VERSION: &str = "2.1.75";
const DEFAULT_SAFE_MAX_OUTPUT_TOKENS: u32 = 32_000;
/// Max page size the Anthropic `GET /v1/models` endpoint accepts.
const MODELS_PAGE_LIMIT: u32 = 1000;
/// Upper bound on pages followed by `list_models`, guarding against a server
/// that never clears `has_more`. At `MODELS_PAGE_LIMIT` rows/page this covers
/// far more models than any provider ships.
const MODELS_MAX_PAGES: usize = 100;
/// Deadline for receiving HTTP response headers on a **streaming** request.
///
/// With `stream: true` Anthropic returns `200` + `message_start` promptly and
/// pings through any generation gap, so slow headers are never healthy. What
/// slow headers DO indicate is a half-open pooled connection: after a
/// multi-minute streaming response the server/edge can drop connection state
/// without notifying the client, and the next request written to that
/// connection waits for headers forever (the 2026-07-11 incident: streams
/// hanging 3+ minutes with zero SSE events until an external watchdog killed
/// the whole worker). `connect_timeout` never covers reused connections; this
/// deadline does. Elapsing surfaces a retryable transport error ("timed out")
/// so every retry layer above re-sends on a fresh connection.
const STREAM_HEADERS_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);
/// Maximum silence between SSE byte chunks before the stream is treated as a
/// stalled connection.
///
/// Measured at the **byte** level, where `message_start` and periodic `ping`
/// events keep a healthy stream audibly alive even while no content delta is
/// ready (long adaptive-thinking pauses, large-prompt processing). Byte-level
/// silence past this budget therefore means the connection is dead, not that
/// the model is thinking — consumers measuring at the delta level need 300s
/// to say the same safely.
const SSE_BYTE_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);
/// Total request deadline for the **non-streaming** `chat` path, where the
/// server holds headers until generation completes so no faster bound is
/// safe. Mirrors `CHAT_READ_TIMEOUT_SECS` on the Gemini/Vertex providers.
const CHAT_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(5);
/// How long an idle pooled connection may be reused. Bounds the aged-idle
/// half of the stale-connection class (the immediate-reuse half is covered by
/// [`STREAM_HEADERS_TIMEOUT`]); reqwest's default is 90s.
const POOL_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub const MODEL_HAIKU_35: &str = "claude-3-5-haiku-20241022";
pub const MODEL_SONNET_35: &str = "claude-3-5-sonnet-20241022";
pub const MODEL_SONNET_4: &str = "claude-sonnet-4-20250514";
pub const MODEL_OPUS_4: &str = "claude-opus-4-20250514";

pub const MODEL_HAIKU_45: &str = "claude-haiku-4-5-20251001";
pub const MODEL_SONNET_45: &str = "claude-sonnet-4-5-20250929";
pub const MODEL_SONNET_46: &str = "claude-sonnet-4-6";
pub const MODEL_SONNET_5: &str = "claude-sonnet-5";
pub const MODEL_OPUS_46: &str = "claude-opus-4-6";
pub const MODEL_OPUS_47: &str = "claude-opus-4-7";
pub const MODEL_OPUS_48: &str = "claude-opus-4-8";
pub const MODEL_FABLE_5: &str = "claude-fable-5";

/// Claude Code tool name mappings for OAuth mode.
///
/// When using OAuth tokens, tool names must match Claude Code's exact casing.
/// The mapper passes unknown names through unchanged, so extra entries here are
/// harmless — they future-proof against new tools being registered later.
/// Source: <https://cchistory.mariozechner.at/data/prompts-2.1.11.md>
const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
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

/// Detect two distinct user tool names that differ only by ASCII case.
///
/// In OAuth mode tool names are normalized to Claude Code's casing, so two
/// tools that differ only by case (e.g. `task` and `Task`) would both serialize
/// to the same wire name — producing duplicate tool definitions in the request
/// and misrouting every returned `ToolUse` back to whichever original name
/// appears first. Such a configuration is rejected up front.
fn oauth_tool_name_collision(
    tools: Option<&[agent_sdk_foundation::llm::Tool]>,
) -> Option<(String, String)> {
    let tools = tools?;
    for (index, tool) in tools.iter().enumerate() {
        for other in &tools[index + 1..] {
            if tool.name != other.name && tool.name.eq_ignore_ascii_case(&other.name) {
                return Some((tool.name.clone(), other.name.clone()));
            }
        }
    }
    None
}

fn oauth_tool_collision_message(first: &str, second: &str) -> String {
    format!(
        "OAuth tool names collide case-insensitively: '{first}' and '{second}' would map to the same Claude Code tool name; rename one to disambiguate"
    )
}

/// Returns true if the API key is an OAuth token (`sk-ant-oat-*`).
#[must_use]
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.starts_with("sk-ant-oat")
}

/// One page of the Anthropic `GET /v1/models` response: the model rows plus the
/// cursor fields used to follow pagination.
struct AnthropicModelsPage {
    models: Vec<crate::provider::ModelInfo>,
    has_more: bool,
    last_id: Option<String>,
}

/// Parse one page of the Anthropic `GET /v1/models` response body.
///
/// The Messages API list endpoint returns `{ "data": [{ "id", "display_name",
/// ... }], "has_more": bool, "last_id": "..." }`. It paginates with a default
/// `limit` of 20; `has_more` + `last_id` drive the next request. It does not
/// report token limits, so those fields stay `None`.
fn parse_models_page(body: &str) -> Result<AnthropicModelsPage> {
    #[derive(serde::Deserialize)]
    struct ListResponse {
        #[serde(default)]
        data: Vec<ModelRow>,
        #[serde(default)]
        has_more: bool,
        #[serde(default)]
        last_id: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct ModelRow {
        id: String,
        #[serde(default)]
        display_name: Option<String>,
    }
    let parsed: ListResponse = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("failed to parse Anthropic models list: {e}"))?;
    let models = parsed
        .data
        .into_iter()
        .map(|row| crate::provider::ModelInfo {
            id: row.id,
            display_name: row.display_name,
            context_window: None,
            max_output_tokens: None,
        })
        .collect();
    Ok(AnthropicModelsPage {
        models,
        has_more: parsed.has_more,
        last_id: parsed.last_id,
    })
}

/// Cache-control breakpoints resolved for the three cacheable prefixes of an
/// Anthropic request, in decreasing order of prefix stability. A `None` field
/// means "do not mark this prefix with `cache_control`".
struct CacheRegions {
    tools: Option<data::ApiCacheControl>,
    system: Option<data::ApiCacheControl>,
    messages: Option<data::ApiCacheControl>,
}

impl CacheRegions {
    /// No caching anywhere (request opted out via `CacheConfig`).
    const DISABLED: Self = Self {
        tools: None,
        system: None,
        messages: None,
    };
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
    base_url: String,
    auth_mode: AuthMode,
    thinking: Option<ThinkingConfig>,
    /// Extra headers applied to every request (e.g. for gateway authentication).
    extra_headers: Vec<(String, String)>,
    /// Streaming-path deadline for response headers ([`STREAM_HEADERS_TIMEOUT`]).
    stream_headers_timeout: std::time::Duration,
    /// Streaming-path byte-level inactivity budget ([`SSE_BYTE_IDLE_TIMEOUT`]).
    sse_byte_idle_timeout: std::time::Duration,
}

impl AnthropicProvider {
    /// The conventional environment variable holding the Anthropic API key.
    pub const API_KEY_ENV: &'static str = "ANTHROPIC_API_KEY";

    /// Create a new Anthropic provider with the specified API key and model.
    ///
    /// Automatically detects OAuth tokens (`sk-ant-oat-*`) and switches to
    /// Bearer auth with Claude Code identity headers.
    #[must_use]
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let model = model.into();
        let auth_mode = if is_oauth_token(&api_key) {
            AuthMode::OAuth
        } else {
            AuthMode::ApiKey
        };

        // Configure client with appropriate timeouts for streaming
        // - No overall timeout (streaming can take a long time); the
        //   streaming path bounds headers + byte gaps per request instead
        // - 30 second connect timeout (fresh dials only)
        // - TCP keepalive to prevent connection drops
        // - Bounded idle-connection reuse (see POOL_IDLE_TIMEOUT)
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .build()
            .unwrap_or_else(|error| {
                // A default client still works; it just loses the timeout
                // hardening above — say so instead of degrading silently.
                log::error!(
                    "failed to build Anthropic HTTP client with timeouts, \
                     falling back to reqwest defaults: {error}"
                );
                reqwest::Client::default()
            });

        Self {
            client,
            api_key,
            model,
            base_url: API_BASE_URL.to_owned(),
            auth_mode,
            thinking: None,
            extra_headers: Vec::new(),
            stream_headers_timeout: STREAM_HEADERS_TIMEOUT,
            sse_byte_idle_timeout: SSE_BYTE_IDLE_TIMEOUT,
        }
    }

    /// Returns whether this provider is using OAuth authentication.
    #[must_use]
    pub const fn is_oauth(&self) -> bool {
        matches!(self.auth_mode, AuthMode::OAuth)
    }

    /// Applies authentication headers to a request builder.
    ///
    /// When `api_key` is empty the provider-specific credential header is
    /// skipped — useful for BYOK gateways where auth is handled externally
    /// via `extra_headers`.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let builder = if self.api_key.is_empty() {
            builder.header("anthropic-version", API_VERSION)
        } else {
            match self.auth_mode {
                AuthMode::ApiKey => {
                    let builder = builder
                        .header("x-api-key", &self.api_key)
                        .header("anthropic-version", API_VERSION);
                    // Budget-thinking models (e.g. Sonnet 4.5, Haiku 4.5) only
                    // emit interleaved (mid-loop) thinking blocks when this beta
                    // is set. Adaptive-thinking models (4.6+) interleave
                    // server-side and ignore the header, so gate on the same
                    // predicate as the OAuth arm to keep the cached request
                    // prefix minimal and stable. Do NOT add other OAuth-identity
                    // betas here — API-key auth is not Claude Code.
                    if self.requires_adaptive_thinking() {
                        builder
                    } else {
                        builder.header("anthropic-beta", "interleaved-thinking-2025-05-14")
                    }
                }
                AuthMode::OAuth => {
                    // Build beta features list matching Claude Code's behaviour.
                    // Adaptive-thinking models (4.6) have interleaved thinking
                    // built-in; for older reasoning models we need the explicit beta.
                    let mut beta_features = vec![
                        "claude-code-20250219",
                        "oauth-2025-04-20",
                        "fine-grained-tool-streaming-2025-05-14",
                    ];
                    if !self.requires_adaptive_thinking() {
                        beta_features.push("interleaved-thinking-2025-05-14");
                    }
                    builder
                        .header("Authorization", format!("Bearer {}", self.api_key))
                        .header("anthropic-version", API_VERSION)
                        .header("anthropic-beta", beta_features.join(","))
                        .header("user-agent", format!("claude-cli/{CLAUDE_CODE_VERSION}"))
                        .header("x-app", "cli")
                }
            }
        };
        self.extra_headers
            .iter()
            .fold(builder, |b, (k, v)| b.header(k.as_str(), v.as_str()))
    }

    const OAUTH_IDENTITY: &'static str =
        "You are Claude Code, Anthropic's official CLI for Claude.";

    /// Build the system prompt payload, accounting for OAuth mode.
    ///
    /// In OAuth mode the identity string must be a **separate** system block
    /// (matching the layout Claude Code itself sends).  For API-key auth the
    /// user-supplied system prompt is sent as a single block.
    fn build_system_prompt_for_request<'a>(
        &self,
        system: &'a str,
        cache_control: Option<data::ApiCacheControl>,
    ) -> Option<data::ApiSystemPrompt<'a>> {
        match self.auth_mode {
            AuthMode::ApiKey => data::build_api_system_prompt(system, cache_control),
            AuthMode::OAuth => {
                let mut blocks = vec![data::ApiSystemBlock {
                    block_type: "text",
                    text: Self::OAUTH_IDENTITY,
                    cache_control: cache_control.clone(),
                }];
                if !system.is_empty() {
                    blocks.push(data::ApiSystemBlock {
                        block_type: "text",
                        text: system,
                        cache_control,
                    });
                }
                Some(data::ApiSystemPrompt::Blocks(blocks))
            }
        }
    }

    /// Resolve the per-prefix cache breakpoints for a request from its optional
    /// [`CacheConfig`](agent_sdk_foundation::llm::CacheConfig).
    ///
    /// With no config (the default) this reproduces the historical behaviour:
    /// an ephemeral breakpoint on the tools, system, and last-user-message
    /// prefixes. An opted-out config disables all breakpoints; a TTL flows onto
    /// every breakpoint; and `max_breakpoints` caps how many prefixes are
    /// marked, in decreasing order of stability (tools, system, conversation).
    fn cache_regions(request: &ChatRequest) -> CacheRegions {
        let (enabled, ttl, max_breakpoints) =
            request.cache.as_ref().map_or((true, None, None), |cfg| {
                (cfg.enabled, cfg.ttl, cfg.max_breakpoints)
            });
        if !enabled {
            return CacheRegions::DISABLED;
        }
        let control = data::ApiCacheControl::ephemeral_with_ttl(ttl.map(CacheTtl::as_wire_str));
        let limit = max_breakpoints.unwrap_or(u8::MAX);
        CacheRegions {
            tools: (limit >= 1).then(|| control.clone()),
            system: (limit >= 2).then(|| control.clone()),
            messages: (limit >= 3).then_some(control),
        }
    }

    fn build_cached_api_messages(
        request: &ChatRequest,
        cache_control: Option<data::ApiCacheControl>,
    ) -> Vec<data::ApiMessage> {
        let mut messages = build_api_messages(request);
        if let Some(cache_control) = cache_control {
            data::apply_cache_control_to_last_user_message(&mut messages, cache_control);
        }
        messages
    }

    fn effective_max_tokens(&self, request: &ChatRequest) -> u32 {
        if request.max_tokens_explicit {
            request.max_tokens
        } else {
            self.default_max_tokens()
        }
    }

    /// Create a provider using Claude Sonnet, reading the API key from the
    /// conventional [`ANTHROPIC_API_KEY`](Self::API_KEY_ENV) environment
    /// variable.
    ///
    /// This is the zero-ceremony on-ramp for the quickstart. Use
    /// [`try_from_env`](Self::try_from_env) if you want to handle a missing
    /// key without a panic.
    ///
    /// # Panics
    ///
    /// Panics if `ANTHROPIC_API_KEY` is not set. Prefer
    /// [`try_from_env`](Self::try_from_env) outside of examples/tests.
    #[must_use]
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a provider using Claude Sonnet, reading the API key from the
    /// conventional [`ANTHROPIC_API_KEY`](Self::API_KEY_ENV) environment
    /// variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `ANTHROPIC_API_KEY` is unset or not valid UTF-8.
    pub fn try_from_env() -> Result<Self> {
        let api_key = std::env::var(Self::API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!("environment variable `{}` is not set", Self::API_KEY_ENV)
        })?;
        Ok(Self::sonnet(api_key))
    }

    /// Create a provider using Claude Haiku 4.5.
    #[must_use]
    pub fn haiku(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_HAIKU_45)
    }

    /// Create a provider using Claude Sonnet 4.6.
    #[must_use]
    pub fn sonnet(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_SONNET_46)
    }

    /// Create a provider using Claude Sonnet 4.5.
    #[must_use]
    pub fn sonnet_45(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_SONNET_45)
    }

    /// Create a provider using Claude Sonnet 4.6.
    #[must_use]
    pub fn sonnet_46(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_SONNET_46)
    }

    /// Create a provider using Claude Opus 4.6.
    #[must_use]
    pub fn opus(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_OPUS_46)
    }

    /// Create a provider using Claude Opus 4.7.
    ///
    /// Note: Opus 4.7 requires adaptive thinking. Passing a
    /// `ThinkingConfig` with `ThinkingMode::Enabled { budget_tokens }`
    /// will return an `InvalidRequest` — use `ThinkingConfig::adaptive()`
    /// or `ThinkingConfig::adaptive_with_effort(_)` instead.
    #[must_use]
    pub fn opus_47(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_OPUS_47)
    }

    /// Create a provider using Claude Opus 4.8.
    ///
    /// Note: Opus 4.8 requires adaptive thinking. Passing a
    /// `ThinkingConfig` with `ThinkingMode::Enabled { budget_tokens }`
    /// will return an `InvalidRequest` — use `ThinkingConfig::adaptive()`
    /// or `ThinkingConfig::adaptive_with_effort(_)` instead.
    #[must_use]
    pub fn opus_48(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_OPUS_48)
    }

    /// Create a provider using Claude Fable 5.
    ///
    /// Note: Fable 5 is adaptive-only — the API applies adaptive thinking
    /// even when no thinking config is sent, and raw chain of thought is
    /// never returned (thinking blocks arrive with empty content). Passing a
    /// `ThinkingConfig` with `ThinkingMode::Enabled { budget_tokens }`
    /// will return an `InvalidRequest` — use `ThinkingConfig::adaptive()`
    /// or `ThinkingConfig::adaptive_with_effort(_)` instead.
    #[must_use]
    pub fn fable(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_FABLE_5)
    }

    /// Claude Sonnet 5 — adaptive-only like Opus 4.8: manual `budget_tokens`
    /// returns a 400; use `ThinkingConfig::adaptive()` instead.
    #[must_use]
    pub fn sonnet_5(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_SONNET_5)
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Override the base URL (default: `https://api.anthropic.com`).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override the streaming stall guards: the response-headers deadline
    /// (default 1 minute — `STREAM_HEADERS_TIMEOUT`) and the SSE
    /// byte-inactivity budget (default 90s — `SSE_BYTE_IDLE_TIMEOUT`).
    /// The defaults suit `api.anthropic.com`; gateways with slower
    /// first-byte behaviour can widen them.
    #[must_use]
    pub const fn with_stream_stall_timeouts(
        mut self,
        headers_timeout: std::time::Duration,
        byte_idle_timeout: std::time::Duration,
    ) -> Self {
        self.stream_headers_timeout = headers_timeout;
        self.sse_byte_idle_timeout = byte_idle_timeout;
        self
    }

    /// Add extra HTTP headers applied to every request.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    fn requires_adaptive_thinking(&self) -> bool {
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
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
            // Forcing a specific tool is incompatible with extended thinking on
            // Anthropic (the API 400s), so drop thinking at the wire boundary
            // even when it was resurrected from the provider-configured default.
            Ok(thinking) => thinking_for_forced_tool(thinking, request.tool_choice.as_ref()),
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        if self.is_oauth()
            && let Some((first, second)) = oauth_tool_name_collision(request.tools.as_deref())
        {
            return Ok(ChatOutcome::InvalidRequest(oauth_tool_collision_message(
                &first, &second,
            )));
        }
        let CacheRegions {
            tools: tools_cache,
            system: system_cache,
            messages: messages_cache,
        } = Self::cache_regions(&request);
        let messages = Self::build_cached_api_messages(&request, messages_cache);
        let tools = if self.is_oauth() {
            build_api_tools_with_cache(&request, tools_cache).map(|tools| {
                tools
                    .into_iter()
                    .map(|mut t| {
                        t.name = to_claude_code_name(&t.name);
                        t
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            build_api_tools_with_cache(&request, tools_cache)
        };
        let thinking = thinking_config
            .as_ref()
            .map(ApiThinkingConfig::from_thinking_config);
        let output_config = thinking_config
            .as_ref()
            .and_then(|t| t.effort)
            .map(|effort| ApiOutputConfig { effort });

        let system = self.build_system_prompt_for_request(&request.system, system_cache);
        let max_tokens = self.effective_max_tokens(&request);
        let tool_choice = request
            .tool_choice
            .as_ref()
            .map(ApiToolChoice::from_tool_choice);

        let api_request = ApiMessagesRequest {
            model: Some(&self.model),
            max_tokens,
            system,
            messages: &messages,
            tools: tools.as_deref(),
            tool_choice,
            stream: false,
            thinking,
            output_config,
            anthropic_version: None,
        };

        log::debug!(
            "Anthropic LLM request model={} max_tokens={} oauth={}",
            self.model,
            max_tokens,
            self.is_oauth()
        );

        // Log full request payload for debugging
        if log::log_enabled!(log::Level::Debug) {
            match serde_json::to_string_pretty(&api_request) {
                Ok(json) => log::debug!("Anthropic API request payload:\n{json}"),
                Err(e) => log::debug!("Failed to serialize request for logging: {e}"),
            }
        }

        // Non-streaming: the server holds headers until generation completes,
        // so a fast headers deadline is unsafe here — bound the whole request
        // instead (same shape as the Gemini/Vertex providers). Without any
        // bound, a half-open pooled connection hangs this call forever.
        let builder = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .timeout(CHAT_REQUEST_TIMEOUT)
            .header("Content-Type", "application/json");
        let response = self
            .apply_auth(builder)
            .json(&api_request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;

        let status = response.status();
        // Read `Retry-After` off the 429 response before the body is consumed
        // (`bytes()` takes the response by value).
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
            "Anthropic LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited(retry_after));
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
            api_response.usage.total_input_tokens(),
            api_response.usage.output,
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
                input_tokens: api_response.usage.total_input_tokens(),
                output_tokens: api_response.usage.output,
                cached_input_tokens: api_response.usage.cached_input_tokens(),
                cache_creation_input_tokens: api_response.usage.cache_creation_input_tokens(),
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

            if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }

            if is_oauth
                && let Some((first, second)) = oauth_tool_name_collision(request.tools.as_deref())
            {
                yield Ok(StreamDelta::Error {
                    message: oauth_tool_collision_message(&first, &second),
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }

            let CacheRegions {
                tools: tools_cache,
                system: system_cache,
                messages: messages_cache,
            } = Self::cache_regions(&request);
            let messages = Self::build_cached_api_messages(&request, messages_cache);
            let tools = if is_oauth {
                build_api_tools_with_cache(&request, tools_cache).map(|tools| {
                    tools
                        .into_iter()
                        .map(|mut t| {
                            t.name = to_claude_code_name(&t.name);
                            t
                        })
                        .collect::<Vec<_>>()
                })
            } else {
                build_api_tools_with_cache(&request, tools_cache)
            };
            let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
                // Forcing a specific tool is incompatible with extended thinking
                // on Anthropic (the API 400s), so drop thinking at the wire
                // boundary even when it was resurrected from the
                // provider-configured default.
                Ok(thinking) => thinking_for_forced_tool(thinking, request.tool_choice.as_ref()),
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
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

            let system = self.build_system_prompt_for_request(&request.system, system_cache);
            let max_tokens = self.effective_max_tokens(&request);
            let tool_choice = request
                .tool_choice
                .as_ref()
                .map(ApiToolChoice::from_tool_choice);

            let api_request = ApiMessagesRequest {
                model: Some(&self.model),
                max_tokens,
                system,
                messages: &messages,
                tools: tools.as_deref(),
                tool_choice,
                stream: true,
                thinking,
                output_config,
                anthropic_version: None,
            };

            log::debug!("Anthropic streaming LLM request model={} max_tokens={} oauth={}", self.model, max_tokens, is_oauth);

            // Log full request payload for debugging
            if log::log_enabled!(log::Level::Debug) {
                match serde_json::to_string_pretty(&api_request) {
                    Ok(json) => log::debug!("Anthropic streaming API request payload:\n{json}"),
                    Err(e) => log::debug!("Failed to serialize streaming request for logging: {e}"),
                }
            }

            let builder = self
                .client
                .post(format!("{}/v1/messages", self.base_url))
                .header("Content-Type", "application/json");
            // With `stream: true` the server answers promptly (200 +
            // `message_start`, then pings through generation gaps), so slow
            // headers only ever mean a dead connection — typically a pooled
            // keep-alive the server/edge half-closed after a long previous
            // response. `connect_timeout` never covers reused connections;
            // this deadline does. The message says "timed out" so every
            // retry layer classifies it as a retryable transport error and
            // re-sends on a fresh connection.
            let headers_timeout = self.stream_headers_timeout;
            let send = self.apply_auth(builder).json(&api_request).send();
            let response = match tokio::time::timeout(headers_timeout, send).await {
                Ok(Ok(r)) => r,
                Ok(Err(error)) => {
                    yield Ok(reqwest_error_delta("request failed", &error));
                    return;
                }
                Err(_elapsed) => {
                    log::error!(
                        "Anthropic streaming request timed out awaiting response headers after {}s — stalled connection",
                        headers_timeout.as_secs()
                    );
                    yield Ok(StreamDelta::Error {
                        message: format!(
                            "request timed out awaiting response headers after {}s",
                            headers_timeout.as_secs()
                        ),
                        kind: StreamErrorKind::ConnectionLost,
                    });
                    return;
                }
            };

            let status = response.status();

            if status == StatusCode::TOO_MANY_REQUESTS {
                let retry_after = crate::http::retry_after_from_headers(response.headers());
                yield Ok(StreamDelta::Error {
                    message: "Rate limited".to_string(),
                    kind: StreamErrorKind::RateLimited(retry_after),
                });
                return;
            }

            if status.is_server_error() {
                let body = response.text().await.unwrap_or_default();
                log::error!("Anthropic server error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    kind: StreamErrorKind::ServerError,
                });
                return;
            }

            if status.is_client_error() {
                let body = response.text().await.unwrap_or_default();
                log::warn!("Anthropic client error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    kind: StreamErrorKind::InvalidRequest,
                });
                return;
            }

            // Process SSE stream
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;
            let mut cached_input_tokens: u32 = 0;
            let mut cache_creation_input_tokens: u32 = 0;
            // Track tool IDs by block index for correlating input deltas
            let mut tool_ids: std::collections::HashMap<usize, String> =
                std::collections::HashMap::new();

            let mut received_message_stop = false;
            // Set when Anthropic streamed a terminal `error` event (parsed
            // into a StreamDelta::Error below). Suppresses the generic
            // "stream ended without message_stop" fallback so the caller
            // sees the real error, not a second misleading one.
            let mut stream_errored = false;
            let mut pending_stop_reason: Option<agent_sdk_foundation::llm::StopReason> = None;
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
                        // Stream drops are expected when the user cancels a running
                        // agent loop (Esc / Ctrl-C).  Log at debug level so it does
                        // not surface as noise in every cancelled session.
                        log::debug!(
                            "SSE stream dropped before completion at chunk_count={} (task was likely cancelled)",
                            self.chunk_count
                        );
                    }
                }
            }
            let mut drop_guard = StreamDropGuard { completed: false, chunk_count: 0 };

            log::debug!("Starting SSE stream processing");

            // Byte-level inactivity guard: `message_start` and periodic
            // `ping` events keep a healthy stream's bytes flowing even while
            // no content delta is ready, so silence past the budget means
            // the connection is dead — not that the model is thinking.
            let byte_idle_timeout = self.sse_byte_idle_timeout;
            loop {
                let next = match tokio::time::timeout(byte_idle_timeout, stream.next()).await {
                    Ok(next) => next,
                    Err(_elapsed) => {
                        log::error!(
                            "SSE stream timed out: no bytes for {}s chunk_count={chunk_count} total_bytes={total_bytes} — stalled connection",
                            byte_idle_timeout.as_secs()
                        );
                        yield Ok(StreamDelta::Error {
                            message: format!(
                                "SSE stream timed out: no bytes for {}s",
                                byte_idle_timeout.as_secs()
                            ),
                            kind: StreamErrorKind::ConnectionLost,
                        });
                        return;
                    }
                };
                let Some(chunk_result) = next else { break };
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(error) => {
                        log::error!("Stream error while reading chunk error={error} chunk_count={chunk_count} total_bytes={total_bytes}");
                        yield Ok(reqwest_body_error_delta("stream error", &error));
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

                // Process complete SSE events (terminated by a blank line)
                while let Some(event_block) = take_next_sse_event(&mut buffer) {
                    // Track if we received message_stop
                    if is_message_stop_event(&event_block) {
                        log::debug!("Received message_stop event chunk_count={chunk_count} total_bytes={total_bytes}");
                        received_message_stop = true;
                    }

                    // Parse SSE event
                    if let Some(mut delta) = parse_sse_event(
                        &event_block,
                        &mut input_tokens,
                        &mut output_tokens,
                        &mut cached_input_tokens,
                        &mut cache_creation_input_tokens,
                        &mut tool_ids,
                        &mut pending_stop_reason,
                    ) {
                        // Reverse-map tool names from Claude Code casing
                        if is_oauth
                            && let StreamDelta::ToolUseStart { ref mut name, .. } = delta
                        {
                            *name = from_claude_code_name(name, &original_tool_names);
                        }
                        // A terminal error event ends the stream — flag it
                        // so the no-message_stop fallback stays silent.
                        if matches!(delta, StreamDelta::Error { .. }) {
                            stream_errored = true;
                        }
                        yield Ok(delta);
                    }
                    // After message_stop (which emits Usage), emit Done
                    if is_message_stop_event(&event_block) {
                        yield Ok(StreamDelta::Done {
                            stop_reason: pending_stop_reason.take(),
                        });
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
                if is_message_stop_event(remaining) {
                    received_message_stop = true;
                }

                if let Some(mut delta) = parse_sse_event(
                    remaining,
                    &mut input_tokens,
                    &mut output_tokens,
                    &mut cached_input_tokens,
                    &mut cache_creation_input_tokens,
                    &mut tool_ids,
                    &mut pending_stop_reason,
                ) {
                    if is_oauth
                        && let StreamDelta::ToolUseStart { ref mut name, .. } = delta
                    {
                        *name = from_claude_code_name(name, &original_tool_names);
                    }
                    if matches!(delta, StreamDelta::Error { .. }) {
                        stream_errored = true;
                    }
                    yield Ok(delta);
                }
                // After message_stop (which emits Usage), emit Done
                if is_message_stop_event(remaining) {
                    yield Ok(StreamDelta::Done {
                        stop_reason: pending_stop_reason.take(),
                    });
                }
            }

            // Mark stream as properly completed
            drop_guard.completed = true;

            // If stream ended without message_stop AND without a parsed
            // error event, emit a generic server-error (transient) signal.
            // When Anthropic streamed a terminal `error` event, it was
            // already surfaced above with its real message + kind — don't
            // mask it with this generic one.
            if !received_message_stop && !stream_errored {
                log::warn!(
                    "SSE stream ended without message_stop event - stream may have been interrupted chunk_count={chunk_count} total_bytes={total_bytes}"
                );
                yield Ok(StreamDelta::Error {
                    message: "Stream ended unexpectedly without completion".to_string(),
                    kind: StreamErrorKind::ServerError,
                });
            }
        })
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

        if self.requires_adaptive_thinking()
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

    async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
        // The endpoint paginates (default `limit=20`). Request the max page size
        // and follow `has_more` / `last_id` until exhausted, capped to avoid an
        // unbounded loop if the server never clears `has_more`.
        let mut models = Vec::new();
        let mut after_id: Option<String> = None;
        for _ in 0..MODELS_MAX_PAGES {
            let mut query: Vec<(&str, String)> = vec![("limit", MODELS_PAGE_LIMIT.to_string())];
            if let Some(after) = &after_id {
                query.push(("after_id", after.clone()));
            }
            let builder = self
                .client
                .get(format!("{}/v1/models", self.base_url))
                .header("Content-Type", "application/json")
                .query(&query);
            let builder = self.apply_auth(builder);
            let body =
                crate::impls::model_listing::fetch_model_list_body(builder, "Anthropic").await?;
            let page = parse_models_page(&body)?;
            models.extend(page.models);
            if !page.has_more {
                return Ok(models);
            }
            match page.last_id {
                Some(last) => after_id = Some(last),
                // `has_more` with no cursor: stop rather than refetch page 1.
                None => return Ok(models),
            }
        }
        Ok(models)
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

    fn default_max_tokens(&self) -> u32 {
        let model_max = self
            .capabilities()
            .and_then(|caps| caps.max_output_tokens)
            .or_else(|| {
                crate::model_capabilities::default_max_output_tokens(self.provider(), self.model())
            })
            .unwrap_or(4096);
        model_max.clamp(4096, DEFAULT_SAFE_MAX_OUTPUT_TOKENS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANTHROPIC_MODELS_FIXTURE: &str = r#"{
      "data": [
        {"type": "model", "id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
        {"type": "model", "id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"}
      ],
      "has_more": false
    }"#;

    #[test]
    fn parse_models_page_reads_id_and_display_name() -> anyhow::Result<()> {
        let page = parse_models_page(ANTHROPIC_MODELS_FIXTURE)?;
        assert_eq!(page.models.len(), 2);
        assert_eq!(page.models[0].id, "claude-opus-4-8");
        assert_eq!(
            page.models[0].display_name.as_deref(),
            Some("Claude Opus 4.8")
        );
        // Anthropic's listing endpoint reports no token limits.
        assert_eq!(page.models[0].context_window, None);
        assert_eq!(page.models[0].max_output_tokens, None);
        // Single, final page.
        assert!(!page.has_more);
        assert_eq!(page.last_id, None);
        Ok(())
    }

    #[tokio::test]
    async fn list_models_follows_pagination_across_pages() -> anyhow::Result<()> {
        use wiremock::matchers::{method, path, query_param, query_param_is_missing};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        // Page 1: the first request has no `after_id` cursor; it returns
        // `has_more: true` with `last_id` pointing at the next page.
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param_is_missing("after_id"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{
                  "data": [
                    {"type": "model", "id": "claude-opus-4-8", "display_name": "Opus"},
                    {"type": "model", "id": "claude-sonnet-4-5", "display_name": "Sonnet"}
                  ],
                  "has_more": true,
                  "last_id": "claude-sonnet-4-5"
                }"#,
            ))
            .mount(&server)
            .await;

        // Page 2: requested with `after_id=claude-sonnet-4-5`; final page.
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param("after_id", "claude-sonnet-4-5"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{
                  "data": [
                    {"type": "model", "id": "claude-haiku-4-5", "display_name": "Haiku"}
                  ],
                  "has_more": false,
                  "last_id": "claude-haiku-4-5"
                }"#,
            ))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new("test-key-not-a-secret", "claude-test")
            .with_base_url(server.uri());
        let models = provider.list_models().await?;

        // All three models across both pages are returned — none dropped.
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["claude-opus-4-8", "claude-sonnet-4-5", "claude-haiku-4-5"]
        );
        Ok(())
    }

    // ===================
    // Constructor Tests
    // ===================

    #[test]
    fn test_new_creates_provider_with_custom_model() {
        let provider = AnthropicProvider::new("test-api-key", "custom-model");

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
    fn test_anthropic_46_models_reject_budgeted_thinking() {
        let sonnet_46 = AnthropicProvider::sonnet_46("test-api-key".to_string());
        let error = sonnet_46
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("ThinkingConfig::adaptive()"));
    }

    #[test]
    fn test_opus_47_rejects_budgeted_thinking() {
        // Opus 4.7 follows the same adaptive-only policy as 4.6. Without
        // this guard the provider would serialise `thinking.type.enabled`
        // and get a 400 back from the API — we want a clear SDK-level
        // error instead.
        let opus_47 = AnthropicProvider::opus_47("test-api-key".to_string());
        let error = opus_47
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(
            error.to_string().contains("ThinkingConfig::adaptive()"),
            "expected migration hint, got: {error}"
        );
    }

    #[test]
    fn test_opus_47_accepts_adaptive_thinking() {
        let opus_47 = AnthropicProvider::opus_47("test-api-key".to_string());
        assert!(
            opus_47
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );
        assert!(
            opus_47
                .validate_thinking_config(Some(&ThinkingConfig::adaptive_with_effort(
                    agent_sdk_foundation::llm::Effort::High
                )))
                .is_ok()
        );
    }

    #[test]
    fn test_opus_47_factory_creates_opus_47_provider() {
        let provider = AnthropicProvider::opus_47("test-api-key".to_string());
        assert_eq!(provider.model(), MODEL_OPUS_47);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_opus_48_rejects_budgeted_thinking() {
        // Opus 4.8 follows the same adaptive-only policy as 4.6/4.7. Without
        // this guard the provider would serialise `thinking.type.enabled`
        // and get a 400 back from the API — we want a clear SDK-level
        // error instead.
        let opus_48 = AnthropicProvider::opus_48("test-api-key".to_string());
        let error = opus_48
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(
            error.to_string().contains("ThinkingConfig::adaptive()"),
            "expected migration hint, got: {error}"
        );
    }

    #[test]
    fn test_opus_48_accepts_adaptive_thinking() {
        let opus_48 = AnthropicProvider::opus_48("test-api-key".to_string());
        assert!(
            opus_48
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );
        assert!(
            opus_48
                .validate_thinking_config(Some(&ThinkingConfig::adaptive_with_effort(
                    agent_sdk_foundation::llm::Effort::High
                )))
                .is_ok()
        );
    }

    #[test]
    fn test_opus_48_factory_creates_opus_48_provider() {
        let provider = AnthropicProvider::opus_48("test-api-key".to_string());
        assert_eq!(provider.model(), MODEL_OPUS_48);
        assert_eq!(provider.provider(), "anthropic");
    }

    #[test]
    fn test_sonnet_5_rejects_budgeted_thinking() {
        // Sonnet 5 is adaptive-only (like Opus 4.8): manual budget_tokens 400s.
        // Fail fast at the SDK with a migration hint instead of a 400 from the API.
        let sonnet_5 = AnthropicProvider::sonnet_5("test-api-key".to_string());
        let error = sonnet_5
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(
            error.to_string().contains("ThinkingConfig::adaptive()"),
            "expected migration hint, got: {error}"
        );
    }

    #[test]
    fn test_sonnet_5_accepts_adaptive_thinking() {
        let sonnet_5 = AnthropicProvider::sonnet_5("test-api-key".to_string());
        assert!(
            sonnet_5
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );
        assert!(
            sonnet_5
                .validate_thinking_config(Some(&ThinkingConfig::adaptive_with_effort(
                    agent_sdk_foundation::llm::Effort::High
                )))
                .is_ok()
        );
    }

    #[test]
    fn test_fable_5_rejects_budgeted_thinking() {
        // Fable 5 is adaptive-only: the API applies adaptive thinking even
        // when `thinking` is unset and rejects budget-based configs. Fail
        // fast with a migration hint instead of a 400 from the API.
        let fable = AnthropicProvider::fable("test-api-key".to_string());
        let error = fable
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(
            error.to_string().contains("ThinkingConfig::adaptive()"),
            "expected migration hint, got: {error}"
        );
    }

    #[test]
    fn test_fable_5_accepts_adaptive_thinking() {
        let fable = AnthropicProvider::fable("test-api-key".to_string());
        assert!(
            fable
                .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
                .is_ok()
        );
        assert!(
            fable
                .validate_thinking_config(Some(&ThinkingConfig::adaptive_with_effort(
                    agent_sdk_foundation::llm::Effort::High
                )))
                .is_ok()
        );
    }

    #[test]
    fn test_fable_factory_creates_fable_5_provider() {
        let provider = AnthropicProvider::fable("test-api-key".to_string());
        assert_eq!(provider.model(), MODEL_FABLE_5);
        assert_eq!(provider.provider(), "anthropic");
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
        let provider = AnthropicProvider::new("test-api-key", "test-model");
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }

    // ===================
    // OAuth tool-name collision (finding #15)
    // ===================

    fn tool(name: &str) -> agent_sdk_foundation::llm::Tool {
        agent_sdk_foundation::llm::Tool {
            name: name.to_string(),
            description: "desc".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
            display_name: name.to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }
    }

    fn request_with_tools(tools: Vec<agent_sdk_foundation::llm::Tool>) -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![agent_sdk_foundation::llm::Message::user("hi")],
            tools: Some(tools),
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
    fn test_oauth_tool_name_collision_detects_case_variants() {
        let tools = vec![tool("task"), tool("Task")];
        let collision = oauth_tool_name_collision(Some(&tools));
        assert!(collision.is_some());
    }

    #[test]
    fn test_oauth_tool_name_collision_allows_distinct_names() {
        let tools = vec![tool("read"), tool("write"), tool("Read_File")];
        assert!(oauth_tool_name_collision(Some(&tools)).is_none());
        assert!(oauth_tool_name_collision(None).is_none());
    }

    #[tokio::test]
    async fn test_oauth_chat_rejects_case_colliding_tools() -> anyhow::Result<()> {
        // OAuth provider: the collision is caught before any network call.
        let provider = AnthropicProvider::new("sk-ant-oat-test", MODEL_SONNET_45);
        assert!(provider.is_oauth());
        let request = request_with_tools(vec![tool("task"), tool("Task")]);
        let outcome = provider.chat(request).await?;
        match outcome {
            ChatOutcome::InvalidRequest(msg) => {
                assert!(msg.contains("collide case-insensitively"), "got: {msg}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_api_key_chat_does_not_apply_oauth_collision_gate() -> anyhow::Result<()> {
        // For plain API-key auth there is no remapping, so case-differing tool
        // names are not a collision; the request proceeds (and only fails on the
        // network call, which we do not make here — we just confirm the gate is
        // OAuth-only by checking is_oauth()).
        let provider = AnthropicProvider::new("sk-ant-api-test", MODEL_SONNET_45);
        assert!(!provider.is_oauth());
        let tools = vec![tool("task"), tool("Task")];
        // The gate would only trigger in OAuth mode.
        assert!(oauth_tool_name_collision(Some(&tools)).is_some());
        Ok(())
    }

    // ===================
    // Interleaved-thinking beta on API-key auth (Fix 3)
    // ===================

    fn apply_auth_beta_header(provider: &AnthropicProvider) -> anyhow::Result<Option<String>> {
        let builder = reqwest::Client::new().post("http://localhost/v1/messages");
        let request = provider.apply_auth(builder).build()?;
        Ok(request
            .headers()
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned))
    }

    #[test]
    fn api_key_auth_sends_interleaved_beta_for_budget_thinking_models() -> anyhow::Result<()> {
        // Sonnet 4.5 / Haiku 4.5 are budget-thinking models: interleaved
        // mid-loop thinking only appears when this beta header is present.
        for provider in [
            AnthropicProvider::sonnet_45("test-key-not-a-secret"),
            AnthropicProvider::haiku("test-key-not-a-secret"),
        ] {
            assert!(!provider.is_oauth());
            assert_eq!(
                apply_auth_beta_header(&provider)?.as_deref(),
                Some("interleaved-thinking-2025-05-14"),
                "expected interleaved beta for {}",
                provider.model()
            );
        }
        Ok(())
    }

    #[test]
    fn api_key_auth_omits_interleaved_beta_for_adaptive_models() -> anyhow::Result<()> {
        // Adaptive-thinking models (4.6+) interleave server-side and treat the
        // header as deprecated/ignored, so we keep the cached prefix minimal.
        for provider in [
            AnthropicProvider::opus_48("test-key-not-a-secret"),
            AnthropicProvider::sonnet_5("test-key-not-a-secret"),
            AnthropicProvider::fable("test-key-not-a-secret"),
        ] {
            assert!(!provider.is_oauth());
            assert_eq!(
                apply_auth_beta_header(&provider)?,
                None,
                "expected no beta header for adaptive model {}",
                provider.model()
            );
        }
        Ok(())
    }

    // ===================
    // Forced-tool ⇒ thinking dropped on the wire (Fix 8)
    // ===================

    /// Drive a `chat` call against a mock server and return the JSON body the
    /// provider actually put on the wire. The mock replies with a body the
    /// response parser rejects — we only care about the *request*, which
    /// wiremock records regardless of how the outcome is parsed.
    async fn captured_request_body(
        provider: &AnthropicProvider,
        request: ChatRequest,
        server: &wiremock::MockServer,
    ) -> serde_json::Value {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .mount(server)
            .await;

        let _ = provider.chat(request).await;

        let received = server
            .received_requests()
            .await
            .expect("mock server records requests");
        assert_eq!(received.len(), 1, "expected exactly one request");
        serde_json::from_slice(&received[0].body).expect("request body is JSON")
    }

    #[tokio::test]
    async fn forced_tool_drops_configured_thinking_on_the_wire() {
        // Regression for Fix 8: a Claude provider built `.with_thinking(...)`
        // must NOT emit `thinking` on the wire when the request forces a
        // specific tool (the Messages API 400s on that pairing). Clearing
        // `ChatRequest.thinking` alone is insufficient — `resolve_thinking_config`
        // would resurrect the provider-configured default — so the guard lives
        // at the wire boundary.
        let server = wiremock::MockServer::start().await;
        let provider = AnthropicProvider::sonnet_45("sk-ant-api-test")
            .with_thinking(ThinkingConfig::new(10_000))
            .with_base_url(server.uri());

        let mut request = request_with_tools(vec![tool("respond")]);
        request.tool_choice = Some(agent_sdk_foundation::llm::ToolChoice::Tool(
            "respond".to_owned(),
        ));

        let body = captured_request_body(&provider, request, &server).await;

        assert!(
            body.get("thinking").is_none(),
            "thinking must be absent when a tool is forced, got: {body}"
        );
        assert_eq!(
            body["tool_choice"]["type"], "tool",
            "the forced tool_choice must survive, got: {body}"
        );
    }

    #[tokio::test]
    async fn configured_thinking_survives_without_forced_tool() {
        // Guard the other direction: `tool_choice = Auto` (the non-forcing case)
        // keeps the provider-configured thinking on the wire, so the Fix 8 guard
        // is narrowly scoped to tool forcing and does not regress ordinary
        // thinking requests.
        let server = wiremock::MockServer::start().await;
        let provider = AnthropicProvider::sonnet_45("sk-ant-api-test")
            .with_thinking(ThinkingConfig::new(10_000))
            .with_base_url(server.uri());

        let mut request = request_with_tools(vec![tool("read")]);
        request.tool_choice = Some(agent_sdk_foundation::llm::ToolChoice::Auto);

        let body = captured_request_body(&provider, request, &server).await;

        assert_eq!(
            body["thinking"]["type"], "enabled",
            "configured thinking must survive when no tool is forced, got: {body}"
        );
    }

    /// A streaming request whose response headers never arrive (the
    /// half-open pooled-connection shape from the 2026-07-11 incident)
    /// must surface a retryable "timed out" error instead of hanging
    /// until an external watchdog kills the caller.
    #[tokio::test]
    async fn streaming_headers_stall_yields_connection_lost_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(30)))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::sonnet_45("sk-ant-api-test")
            .with_base_url(server.uri())
            .with_stream_stall_timeouts(
                std::time::Duration::from_millis(100),
                std::time::Duration::from_secs(5),
            );

        let items: Vec<_> = provider
            .chat_stream(request_with_tools(vec![]))
            .collect()
            .await;

        assert_eq!(items.len(), 1, "a stalled send yields exactly one item");
        let Some(Ok(StreamDelta::Error { message, kind })) = items.first() else {
            panic!("headers stall must surface as a classified error: {items:?}")
        };
        assert_eq!(*kind, StreamErrorKind::ConnectionLost);
        assert!(
            message.contains("timed out awaiting response headers"),
            "message must name the headers stall: {message}"
        );
    }

    /// A stream that goes byte-silent after headers (server sent a ping,
    /// then the connection went half-open) must surface a retryable
    /// "timed out" error. Hand-rolled server: wiremock cannot hold a
    /// connection open mid-body.
    #[tokio::test]
    async fn sse_byte_stall_yields_connection_lost_error() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            // Consume enough of the request to unblock the client.
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let headers = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
            socket.write_all(headers.as_bytes()).await.expect("headers");
            // One healthy liveness event, then silence with the socket
            // held open — the stall shape the guard exists for.
            let ping = "event: ping\ndata: {\"type\": \"ping\"}\n\n";
            let chunk = format!("{:x}\r\n{ping}\r\n", ping.len());
            socket
                .write_all(chunk.as_bytes())
                .await
                .expect("ping chunk");
            socket.flush().await.expect("flush");
            std::future::pending::<()>().await;
        });

        let provider = AnthropicProvider::sonnet_45("sk-ant-api-test")
            .with_base_url(format!("http://{addr}"))
            .with_stream_stall_timeouts(
                std::time::Duration::from_secs(5),
                std::time::Duration::from_millis(200),
            );

        let items: Vec<_> = provider
            .chat_stream(request_with_tools(vec![]))
            .collect()
            .await;

        let Some(Ok(StreamDelta::Error { message, kind })) = items.last() else {
            panic!("byte stall must surface as a classified error: {items:?}")
        };
        assert_eq!(*kind, StreamErrorKind::ConnectionLost);
        assert!(
            message.contains("no bytes for"),
            "message must name the byte stall: {message}"
        );
        server.abort();
    }
}
