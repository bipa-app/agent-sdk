//! `OpenAI` Codex / `ChatGPT` subscription provider implementation.
//!
//! This mirrors pi's `openai-codex-responses` provider family and talks to the
//! `ChatGPT` Codex backend using OAuth bearer tokens captured from the `ChatGPT`
//! Plus/Pro login flow.

use crate::attachments::validate_request_attachments;
use crate::provider::LlmProvider;
use crate::streaming::{SseLineBuffer, StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Effort, ResponseFormat,
    StopReason, ThinkingConfig, ThinkingMode, ToolChoice, Usage,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine;
use futures::{SinkExt, StreamExt};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WebSocketMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";

/// Connect timeout for the HTTP and WebSocket transports.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound on a single WebSocket frame read / send. A peer that stops sending must
/// not wedge the turn (and, before the lock was released per-turn, the whole
/// session) indefinitely.
const WEBSOCKET_IO_TIMEOUT: Duration = Duration::from_mins(2);
/// Upper bound on the number of cached WebSocket sessions retained at once.
const MAX_WEBSOCKET_SESSIONS: usize = 512;
/// Environment variable that forces the Codex provider onto the HTTP transport,
/// skipping the WebSocket path entirely. Set to a truthy value (`1`, `true`,
/// `yes`, `on`) when the environment cannot complete the `wss` upgrade (a
/// corporate proxy / firewall that black-holes websockets). Honored at provider
/// construction.
const OPENAI_CODEX_DISABLE_WEBSOCKETS_ENV: &str = "OPENAI_CODEX_DISABLE_WEBSOCKETS";

/// Interpret a raw environment-variable value as a boolean. Absent or
/// unrecognized values are treated as `false` so the default (WebSocket-first)
/// behavior is preserved. Split out from the env read so the truthiness logic is
/// unit-testable without mutating the process environment (`std::env::set_var`
/// is `unsafe`, which `#![forbid(unsafe_code)]` rejects even in tests).
fn parse_disable_websockets_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Read [`OPENAI_CODEX_DISABLE_WEBSOCKETS_ENV`] as a boolean.
fn websockets_disabled_from_env() -> bool {
    parse_disable_websockets_value(
        std::env::var(OPENAI_CODEX_DISABLE_WEBSOCKETS_ENV)
            .ok()
            .as_deref(),
    )
}

/// Build an HTTP client with a connect/keepalive timeout, matching the sibling
/// providers (`anthropic`, `vertex`). A bare `reqwest::Client::new()` has no
/// connect timeout, so a black-holed connect would wedge `chat`/`chat_stream`.
fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .tcp_keepalive(CONNECT_TIMEOUT)
        .build()
        .unwrap_or_default()
}
const OPENAI_CODEX_JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const OPENAI_CODEX_ORIGINATOR: &str = "codex_cli_rs";
const OPENAI_CODEX_RESPONSES_BETA_HEADER: &str = "responses=experimental";
const OPENAI_CODEX_RESPONSES_WEBSOCKETS_BETA_HEADER: &str = "responses_websockets=2026-02-06";
const OPENAI_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE: &str =
    "websocket_connection_limit_reached";
const OPENAI_RESPONSES_REASONING_PROVIDER: &str = "openai-responses";
const OPENAI_MESSAGE_ITEM_TYPE: &str = "message";

// GPT-5.4 (frontier reasoning with 1.05M context)
pub const MODEL_GPT54: &str = "gpt-5.4";

// GPT-5.3-Codex (latest Codex model)
pub const MODEL_GPT53_CODEX: &str = "gpt-5.3-codex";

// GPT-5.2-Codex (legacy Responses-first codex model)
pub const MODEL_GPT52_CODEX: &str = "gpt-5.2-codex";

/// Reasoning effort level for the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
    /// Extra-high reasoning for complex problems
    #[serde(rename = "xhigh")]
    XHigh,
}

/// `OpenAI` Codex / `ChatGPT` subscription provider.
///
/// This provider uses the `ChatGPT` Codex backend (`/backend-api/codex/responses`)
/// and requires an OAuth access token obtained from the `ChatGPT` Plus/Pro login flow.
#[derive(Clone)]
pub struct OpenAICodexResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    thinking: Option<ThinkingConfig>,
    account_id: Option<String>,
    websocket_sessions: Arc<Mutex<HashMap<String, Arc<Mutex<WebsocketSessionState>>>>>,
    /// Hard opt-out: when set, the WebSocket transport is never attempted and
    /// every turn goes straight to HTTP. Sourced from
    /// [`with_websockets_disabled`](Self::with_websockets_disabled) or the
    /// [`OPENAI_CODEX_DISABLE_WEBSOCKETS_ENV`] environment variable.
    websockets_disabled: bool,
    /// Cross-session "websockets don't work in this environment" memory. The
    /// per-session [`WebsocketSessionState::websocket_disabled`] flag only
    /// protects the session that hit the failure, so a fresh session would
    /// re-pay the full connect/warmup timeout penalty. A *transport*
    /// (connectivity) failure latches this provider-level flag so every
    /// subsequent session skips the WebSocket attempt and goes straight to
    /// HTTP. The environment self-heals after the first failed session instead
    /// of stalling on every one. Latches once per process and never resets — a
    /// genuinely WS-hostile network does not recover mid-process, and a
    /// once-per-process latch is the simplest correct behavior. Auth/request
    /// failures (401/client errors) deliberately do *not* set this flag.
    websockets_unhealthy: Arc<AtomicBool>,
}

type CodexWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Default)]
struct WebsocketSessionState {
    connection: Option<CodexWebSocket>,
    last_request: Option<ApiStreamingRequest>,
    last_response_id: Option<String>,
    last_response_items: Vec<ApiInputItem>,
    turn_state: Option<String>,
    prewarmed: bool,
    websocket_disabled: bool,
    /// Set while a turn is mid-flight on this session and cleared when it
    /// completes cleanly. If a turn's stream is dropped (cancellation), this
    /// stays set; the next turn detects the abandoned turn on lock acquisition,
    /// discards the half-consumed connection + stale incremental baseline, and
    /// reconnects so the cancelled turn can't poison it.
    in_flight: bool,
    /// Last time this session was touched, for LRU eviction of the bounded map.
    last_used: Option<Instant>,
}

impl OpenAICodexResponsesProvider {
    /// Create a new `OpenAI` Codex provider.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: build_http_client(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
            thinking: None,
            account_id: None,
            websocket_sessions: Arc::new(Mutex::new(HashMap::new())),
            websockets_disabled: websockets_disabled_from_env(),
            websockets_unhealthy: Arc::new(AtomicBool::new(false)),
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
            account_id: None,
            websocket_sessions: Arc::new(Mutex::new(HashMap::new())),
            websockets_disabled: websockets_disabled_from_env(),
            websockets_unhealthy: Arc::new(AtomicBool::new(false)),
        }
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

    /// Create a provider using GPT-5.4 (frontier reasoning with 1.05M context).
    #[must_use]
    pub fn gpt54(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT54.to_owned())
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set a known `ChatGPT` account id, avoiding JWT decoding on each request.
    #[must_use]
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Set the reasoning effort level.
    #[must_use]
    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        self.with_thinking(ThinkingConfig::default().with_effort(map_reasoning_effort(effort)))
    }

    /// Force the HTTP transport, skipping the WebSocket path entirely.
    ///
    /// In a WebSocket-hostile environment (a corporate proxy / firewall that
    /// black-holes the `wss` upgrade) the WebSocket-first transport stalls for
    /// up to the connect + warmup timeout budget on every fresh session before
    /// falling back to HTTP. An operator who knows their network cannot do
    /// `wss` can set this to skip the penalty entirely. The
    /// `OPENAI_CODEX_DISABLE_WEBSOCKETS` environment variable does the
    /// same without a code change.
    #[must_use]
    pub const fn with_websockets_disabled(mut self, disabled: bool) -> Self {
        self.websockets_disabled = disabled;
        self
    }

    /// Whether the WebSocket transport should be skipped for this turn: either
    /// it was hard-disabled (builder / env) or a prior session in this process
    /// already proved the environment cannot complete the `wss` transport.
    fn skip_websocket(&self) -> bool {
        self.websockets_disabled || self.websockets_unhealthy.load(Ordering::Relaxed)
    }

    /// The `ChatGPT`-backend Codex Responses contract does not accept
    /// `max_output_tokens` — the backend manages the output budget and
    /// rejects the parameter with `400 InvalidRequest` ("Unsupported
    /// parameter: `max_output_tokens`", verified live 2026-07-10 against
    /// `gpt-5.4` on a `ChatGPT` account). The official Codex CLI never
    /// sends it, and the reverse-engineered request contract (bip's
    /// `docs/codex-oauth-study-2026-06-13.md`) carries no such field. The
    /// caller's `max_tokens` is therefore intentionally dropped on this
    /// transport regardless of `max_tokens_explicit` — hosts like
    /// `agent-server` always mark a resolved default as explicit, which
    /// would otherwise fail every daemon-path Codex turn.
    const fn max_output_tokens(_request: &ChatRequest) -> Option<u32> {
        None
    }

    fn build_headers(
        &self,
        streaming: bool,
        session_id: Option<&str>,
        turn_state: Option<&str>,
    ) -> Result<reqwest::header::HeaderMap> {
        self.build_headers_with_beta(
            streaming,
            session_id,
            OPENAI_CODEX_RESPONSES_BETA_HEADER,
            turn_state,
        )
    }

    fn build_websocket_headers(
        &self,
        session_id: Option<&str>,
        turn_state: Option<&str>,
    ) -> Result<reqwest::header::HeaderMap> {
        self.build_headers_with_beta(
            false,
            session_id,
            OPENAI_CODEX_RESPONSES_WEBSOCKETS_BETA_HEADER,
            turn_state,
        )
    }

    fn build_headers_with_beta(
        &self,
        streaming: bool,
        session_id: Option<&str>,
        beta_header: &'static str,
        turn_state: Option<&str>,
    ) -> Result<reqwest::header::HeaderMap> {
        use reqwest::header::{
            ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT,
        };

        let account_id = self
            .account_id
            .clone()
            .map_or_else(|| extract_account_id(&self.api_key), Ok)
            .context("failed to extract chatgpt account id from OpenAI Codex OAuth token")?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))?,
        );
        headers.insert("chatgpt-account-id", HeaderValue::from_str(&account_id)?);
        headers.insert("OpenAI-Beta", HeaderValue::from_static(beta_header));
        headers.insert(
            "originator",
            HeaderValue::from_static(OPENAI_CODEX_ORIGINATOR),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&format!(
                "{OPENAI_CODEX_ORIGINATOR}/{} ({} {})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH,
            ))?,
        );
        if streaming {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        if let Some(session_id) = session_id {
            let session_id_header = HeaderValue::from_str(session_id)?;
            headers.insert("session_id", session_id_header.clone());
            headers.insert("x-client-request-id", session_id_header);
        }
        if let Some(turn_state) = turn_state {
            headers.insert(
                OPENAI_CODEX_TURN_STATE_HEADER,
                HeaderValue::from_str(turn_state)?,
            );
        }

        Ok(headers)
    }

    async fn websocket_session(&self, session_id: &str) -> Arc<Mutex<WebsocketSessionState>> {
        let mut sessions = self.websocket_sessions.lock().await;
        if !sessions.contains_key(session_id) && sessions.len() >= MAX_WEBSOCKET_SESSIONS {
            evict_idle_sessions(&mut sessions);
        }
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(WebsocketSessionState::default())))
            .clone()
    }

    async fn connect_websocket(
        &self,
        session_id: Option<&str>,
        turn_state: Option<&str>,
    ) -> Result<(CodexWebSocket, Option<String>)> {
        let headers = self.build_websocket_headers(session_id, turn_state)?;
        let url = codex_websocket_url(&self.base_url)
            .context("failed to build OpenAI Codex websocket URL")?;
        let mut request = url
            .as_str()
            .into_client_request()
            .context("failed to build OpenAI Codex websocket request")?;
        request.headers_mut().extend(headers);

        let (stream, response) = timeout(CONNECT_TIMEOUT, connect_async(request))
            .await
            .context("OpenAI Codex websocket connect timed out")?
            .context("failed to connect OpenAI Codex websocket")?;
        let turn_state = response
            .headers()
            .get(OPENAI_CODEX_TURN_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        Ok((stream, turn_state))
    }

    fn map_response(api_response: ApiResponse) -> ChatResponse {
        let refused = output_contains_refusal(&api_response.output);
        let mut content = build_content_blocks(&api_response.output);
        let has_tool_calls = content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
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
            api_response.status.map(|status| match status {
                ApiStatus::Completed => StopReason::EndTurn,
                // Terminal failures and any non-terminal/unknown status
                // that leaked into a final response map to Unknown, exactly
                // as the old Incomplete|Failed arm did. Spelled out per
                // variant so adding a status forces a decision here.
                ApiStatus::Incomplete
                | ApiStatus::Failed
                | ApiStatus::InProgress
                | ApiStatus::Queued
                | ApiStatus::Cancelled
                | ApiStatus::Other => StopReason::Unknown,
            })
        };

        if stop_reason != Some(StopReason::ToolUse) {
            content.retain(|block| !matches!(block, ContentBlock::ToolUse { .. }));
        }

        ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: api_response.usage.map_or(
                Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
                |usage| usage_from_api_usage(&usage),
            ),
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAICodexResponsesProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let thinking_config = match self.resolve_thinking_config(request.thinking.as_ref()) {
            Ok(thinking) => thinking,
            Err(error) => return Ok(ChatOutcome::InvalidRequest(error.to_string())),
        };
        if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
            return Ok(ChatOutcome::InvalidRequest(error.to_string()));
        }
        let reasoning = build_api_reasoning(thinking_config.as_ref());
        let input = build_api_input(&request);
        let max_output_tokens = Self::max_output_tokens(&request);
        let prompt_cache_key = request.session_id.as_deref();
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .as_ref()
            .map(|ts| ts.iter().cloned().map(convert_tool).collect());
        let parallel_tool_calls = tools.as_ref().is_some_and(|tools| !tools.is_empty());
        let text_format = request
            .response_format
            .as_ref()
            .map(ApiResponseTextFormat::from);
        let tool_choice = codex_tool_choice(request.tool_choice.as_ref());

        let api_request = ApiResponsesRequest {
            model: &self.model,
            instructions: request.system.as_str(),
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens,
            reasoning,
            tool_choice: Some(tool_choice),
            parallel_tool_calls: parallel_tool_calls.then_some(true),
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
                format: text_format,
            }),
            include: Some(&["reasoning.encrypted_content"]),
            prompt_cache_key,
        };

        log::debug!(
            "OpenAI Codex request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let response = self
            .client
            .post(codex_url(&self.base_url))
            .headers(self.build_headers(false, request.session_id.as_deref(), None)?)
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
            "OpenAI Codex response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            let retry_after = retry_after.or_else(|| {
                crate::retry_hints::openai_retry_delay(&String::from_utf8_lossy(&bytes))
            });
            return Ok(ChatOutcome::RateLimited(retry_after));
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("OpenAI Codex server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("OpenAI Codex client error status={status} body={body}");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        // The Responses API reports generation failures as HTTP 200 with
        // status=failed plus an error object. Surface that as a server error
        // instead of a successful turn with empty content (mirrors the streaming
        // `response.failed` handling).
        if matches!(api_response.status, Some(ApiStatus::Failed)) {
            let message = api_response
                .error
                .and_then(|error| error.message)
                .unwrap_or_else(|| "OpenAI Codex reported status=failed".to_owned());
            log::error!("OpenAI Codex generation failed: {message}");
            return Ok(ChatOutcome::ServerError(message));
        }

        Ok(ChatOutcome::Success(Self::map_response(api_response)))
    }

    #[allow(clippy::too_many_lines)]
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
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
            let input = build_api_input(&request);
            let max_output_tokens = Self::max_output_tokens(&request);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .as_ref()
                .map(|ts| ts.iter().cloned().map(convert_tool).collect());
            let parallel_tool_calls = tools.as_ref().is_some_and(|tools| !tools.is_empty());
            let text_format = request.response_format.as_ref().map(ApiResponseTextFormat::from);
            let tool_choice = codex_tool_choice(request.tool_choice.as_ref());
            let api_request = ApiStreamingRequest {
                model: self.model.clone(),
                instructions: request.system.clone(),
                input,
                tools,
                max_output_tokens,
                reasoning,
                tool_choice: Some(tool_choice),
                parallel_tool_calls: parallel_tool_calls.then_some(true),
                store: false,
                text: Some(ApiTextSettings { verbosity: "medium", format: text_format }),
                include: Some(vec!["reasoning.encrypted_content".to_string()]),
                prompt_cache_key: request.session_id.clone(),
                stream: true,
            };

            log::debug!("OpenAI Codex streaming request model={} max_tokens={}", self.model, request.max_tokens);

            let mut sse_turn_state: Option<String> = None;

            // Skip the WebSocket transport entirely when it is hard-disabled
            // (builder / env) or a prior session already proved this environment
            // cannot complete the `wss` transport. The turn falls through to the
            // HTTP request path below without paying any WebSocket connect /
            // warmup timeout penalty.
            if let Some(session_id) = request.session_id.as_deref().filter(|_| !self.skip_websocket()) {
                let session = self.websocket_session(session_id).await;
                let mut websocket_session = session.lock().await;

                // Latched on a TRANSPORT/connectivity failure (connect timeout,
                // upgrade failure, warmup connection-closed/timeout, mid-stream
                // disconnect before output) so every later session in this
                // process skips the WebSocket attempt. Auth/request failures
                // (401 / client-error wrapped events) deliberately do NOT latch
                // this — a transient auth blip must not force HTTP-only forever.
                let mark_websocket_transport_unhealthy = || {
                    self.websockets_unhealthy.store(true, Ordering::Relaxed);
                };

                // If the previous turn for this session was abandoned mid-flight
                // (its stream was dropped — the SDK's cancellation mechanism), its
                // half-consumed connection and incremental baseline are stale.
                // Discard them so we reconnect fresh and never read the cancelled
                // turn's trailing frames or pair a new request with its response id.
                if websocket_session.in_flight {
                    log::warn!(
                        "OpenAI Codex session {session_id} had an abandoned in-flight turn; resetting websocket state"
                    );
                    reset_websocket_connection(&mut websocket_session);
                    websocket_session.last_request = None;
                    websocket_session.last_response_id = None;
                    websocket_session.last_response_items.clear();
                }
                websocket_session.in_flight = true;
                websocket_session.last_used = Some(Instant::now());

                if !websocket_session.websocket_disabled {
                    'websocket_attempts: for attempt in 0..2 {
                        if websocket_session.connection.is_none() {
                            match self
                                .connect_websocket(
                                    Some(session_id),
                                    websocket_session.turn_state.as_deref(),
                                )
                                .await
                            {
                                Ok((connection, turn_state)) => {
                                    websocket_session.connection = Some(connection);
                                    if let Some(turn_state) = turn_state {
                                        websocket_session.turn_state = Some(turn_state);
                                    }
                                    websocket_session.prewarmed = false;
                                }
                                Err(error) => {
                                    log::warn!(
                                        "OpenAI Codex websocket connect failed on attempt {}: {error:#}",
                                        attempt + 1,
                                    );
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                        mark_websocket_transport_unhealthy();
                                    }
                                    continue;
                                }
                            }
                        }

                        if websocket_session.connection.is_some()
                            && websocket_session.last_request.is_none()
                            && !websocket_session.prewarmed
                        {
                            let mut warmup_request = ApiWebsocketRequest::from(&api_request);
                            warmup_request.generate = Some(false);
                            let warmup_payload = match serde_json::to_string(&warmup_request) {
                                Ok(payload) => payload,
                                Err(error) => {
                                    yield Ok(StreamDelta::Error {
                                        message: format!(
                                            "failed to encode websocket warmup request: {error}"
                                        ),
                                        kind: StreamErrorKind::InvalidRequest,
                                    });
                                    return;
                                }
                            };

                            let warmup_send_result = if let Some(connection) =
                                websocket_session.connection.as_mut()
                            {
                                timeout(
                                    WEBSOCKET_IO_TIMEOUT,
                                    connection.send(WebSocketMessage::Text(warmup_payload.into())),
                                )
                                .await
                                .unwrap_or(Err(
                                    tokio_tungstenite::tungstenite::Error::ConnectionClosed,
                                ))
                            } else {
                                Err(tokio_tungstenite::tungstenite::Error::ConnectionClosed)
                            };

                            if let Err(error) = warmup_send_result {
                                log::warn!(
                                    "OpenAI Codex websocket warmup send failed on attempt {}: {error}",
                                    attempt + 1,
                                );
                                reset_websocket_connection(&mut websocket_session);
                                if attempt == 1 {
                                    websocket_session.websocket_disabled = true;
                                    mark_websocket_transport_unhealthy();
                                }
                                continue;
                            }

                            let mut warmup_response_id: Option<String> = None;
                            let mut warmup_response_items = Vec::new();

                            loop {
                                let message_result = if let Some(connection) =
                                    websocket_session.connection.as_mut()
                                {
                                    timeout(WEBSOCKET_IO_TIMEOUT, connection.next()).await.unwrap_or_else(|_| {
                                        log::warn!("OpenAI Codex websocket warmup read timed out");
                                        None
                                    })
                                } else {
                                    None
                                };
                                let Some(message_result) = message_result else {
                                    log::warn!(
                                        "OpenAI Codex websocket warmup closed before completion on attempt {}",
                                        attempt + 1,
                                    );
                                    reset_websocket_connection(&mut websocket_session);
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                        mark_websocket_transport_unhealthy();
                                    }
                                    continue 'websocket_attempts;
                                };

                                let message = match message_result {
                                    Ok(message) => message,
                                    Err(error) => {
                                        log::warn!(
                                            "OpenAI Codex websocket warmup failed on attempt {}: {error}",
                                            attempt + 1,
                                        );
                                        reset_websocket_connection(&mut websocket_session);
                                        if attempt == 1 {
                                            websocket_session.websocket_disabled = true;
                                            mark_websocket_transport_unhealthy();
                                        }
                                        continue 'websocket_attempts;
                                    }
                                };

                                match message {
                                    WebSocketMessage::Text(text) => {
                                        if let Some(error) =
                                            parse_wrapped_websocket_error_event(&text)
                                        {
                                            log::warn!(
                                                "OpenAI Codex websocket warmup wrapped error on attempt {} status={} message={}",
                                                attempt + 1,
                                                error.status,
                                                error.message,
                                            );
                                            // A quota rejection applies to the request, not to
                                            // this socket: reconnecting or falling back to HTTP
                                            // would spend another request inside the window the
                                            // service asked us to sit out. Surface it with its
                                            // delay instead. The connection-limit sentinel shares
                                            // the status but is a transport condition, so it keeps
                                            // its immediate fallback.
                                            if is_websocket_quota_rejection(&error) {
                                                let kind = websocket_error_kind(&error);
                                                end_websocket_turn(&mut websocket_session);
                                                yield Ok(StreamDelta::Error {
                                                    message: error.message,
                                                    kind,
                                                });
                                                return;
                                            }
                                            if error.status == StatusCode::UNAUTHORIZED
                                                || error.status == StatusCode::UPGRADE_REQUIRED
                                                || error.status.is_client_error()
                                            {
                                                websocket_session.websocket_disabled = true;
                                            }
                                            reset_websocket_connection(&mut websocket_session);
                                            continue 'websocket_attempts;
                                        }
                                        let event = match decode_stream_event(&text) {
                                            Ok(event) => event,
                                            Err(error) => {
                                                end_websocket_turn(
                                                    &mut websocket_session,
                                                );
                                                yield Ok(StreamDelta::Error {
                                                    message: error.to_string(),
                                                    kind: StreamErrorKind::ServerError,
                                                });
                                                return;
                                            }
                                        };
                                            match event.r#type.as_str() {
                                                "response.output_item.done" => {
                                                    let item = match decode_output_item(event.item) {
                                                        Ok(item) => item,
                                                        Err(error) => {
                                                            end_websocket_turn(
                                                                &mut websocket_session,
                                                            );
                                                            yield Ok(StreamDelta::Error {
                                                                message: error.to_string(),
                                                                kind: StreamErrorKind::ServerError,
                                                            });
                                                            return;
                                                        }
                                                    };
                                                    if let Some(item) = output_item_to_input_item(item) {
                                                        warmup_response_items.push(item);
                                                    }
                                                }
                                                "response.completed" | "response.done" => {
                                                    if let Some(resp) = event.response
                                                        && let Some(id) = resp.id
                                                    {
                                                        warmup_response_id = Some(id);
                                                    }
                                                    websocket_session.last_request =
                                                        Some(api_request.clone());
                                                    websocket_session.last_response_id =
                                                        warmup_response_id;
                                                    websocket_session.last_response_items =
                                                        warmup_response_items;
                                                    websocket_session.prewarmed = true;
                                                    break;
                                                }
                                                "response.incomplete" | "response.failed" => {
                                                    log::warn!(
                                                        "OpenAI Codex websocket warmup returned {} on attempt {}",
                                                        event.r#type,
                                                        attempt + 1,
                                                    );
                                                    reset_websocket_connection(&mut websocket_session);
                                                    if attempt == 1 {
                                                        websocket_session.websocket_disabled = true;
                                                        mark_websocket_transport_unhealthy();
                                                    }
                                                    continue 'websocket_attempts;
                                                }
                                                _ => {}
                                            }
                                    }
                                    WebSocketMessage::Binary(bytes) => {
                                        let text = match String::from_utf8(bytes.to_vec()) {
                                            Ok(text) => text,
                                            Err(error) => {
                                                end_websocket_turn(
                                                    &mut websocket_session,
                                                );
                                                yield Ok(StreamDelta::Error {
                                                    message: format!(
                                                        "invalid OpenAI Codex websocket UTF-8: {error}"
                                                    ),
                                                    kind: StreamErrorKind::ServerError,
                                                });
                                                return;
                                            }
                                        };
                                            if let Some(error) =
                                                parse_wrapped_websocket_error_event(&text)
                                            {
                                                log::warn!(
                                                    "OpenAI Codex websocket warmup wrapped error on attempt {} status={} message={}",
                                                    attempt + 1,
                                                    error.status,
                                                    error.message,
                                                );
                                                // Same split as the text warmup frame: a quota
                                                // rejection is surfaced with its delay; the
                                                // connection-limit sentinel falls back at once.
                                                if is_websocket_quota_rejection(&error) {
                                                    let kind = websocket_error_kind(&error);
                                                    end_websocket_turn(&mut websocket_session);
                                                    yield Ok(StreamDelta::Error {
                                                        message: error.message,
                                                        kind,
                                                    });
                                                    return;
                                                }
                                                if error.status == StatusCode::UNAUTHORIZED
                                                    || error.status == StatusCode::UPGRADE_REQUIRED
                                                    || error.status.is_client_error()
                                                {
                                                    websocket_session.websocket_disabled = true;
                                                }
                                                reset_websocket_connection(&mut websocket_session);
                                                continue 'websocket_attempts;
                                            }

                                            let event = match decode_stream_event(&text) {
                                                Ok(event) => event,
                                                Err(error) => {
                                                    end_websocket_turn(
                                                        &mut websocket_session,
                                                    );
                                                    yield Ok(StreamDelta::Error {
                                                        message: error.to_string(),
                                                        kind: StreamErrorKind::ServerError,
                                                    });
                                                    return;
                                                }
                                            };
                                                match event.r#type.as_str() {
                                                    "response.output_item.done" => {
                                                        let item = match decode_output_item(event.item) {
                                                            Ok(item) => item,
                                                            Err(error) => {
                                                                end_websocket_turn(
                                                                    &mut websocket_session,
                                                                );
                                                                yield Ok(StreamDelta::Error {
                                                                    message: error.to_string(),
                                                                    kind: StreamErrorKind::ServerError,
                                                                });
                                                                return;
                                                            }
                                                        };
                                                        if let Some(item) = output_item_to_input_item(item) {
                                                            warmup_response_items.push(item);
                                                        }
                                                    }
                                                    "response.completed" | "response.done" => {
                                                        if let Some(resp) = event.response
                                                            && let Some(id) = resp.id
                                                        {
                                                            warmup_response_id = Some(id);
                                                        }
                                                        websocket_session.last_request =
                                                            Some(api_request.clone());
                                                        websocket_session.last_response_id =
                                                            warmup_response_id;
                                                        websocket_session.last_response_items =
                                                            warmup_response_items;
                                                        websocket_session.prewarmed = true;
                                                        break;
                                                    }
                                                    "response.incomplete" | "response.failed" => {
                                                        log::warn!(
                                                            "OpenAI Codex websocket warmup returned {} on attempt {}",
                                                            event.r#type,
                                                            attempt + 1,
                                                        );
                                                        reset_websocket_connection(&mut websocket_session);
                                                        if attempt == 1 {
                                                            websocket_session.websocket_disabled = true;
                                                            mark_websocket_transport_unhealthy();
                                                        }
                                                        continue 'websocket_attempts;
                                                    }
                                                    _ => {}
                                                }
                                    }
                                    WebSocketMessage::Ping(payload) => {
                                        if let Some(connection) =
                                            websocket_session.connection.as_mut()
                                            && let Err(error) = connection
                                                .send(WebSocketMessage::Pong(payload))
                                                .await
                                        {
                                            log::warn!(
                                                "OpenAI Codex websocket warmup pong failed on attempt {}: {error}",
                                                attempt + 1,
                                            );
                                            reset_websocket_connection(&mut websocket_session);
                                            if attempt == 1 {
                                                websocket_session.websocket_disabled = true;
                                                mark_websocket_transport_unhealthy();
                                            }
                                            continue 'websocket_attempts;
                                        }
                                    }
                                    WebSocketMessage::Pong(_) | WebSocketMessage::Frame(_) => {}
                                    WebSocketMessage::Close(_) => {
                                        log::warn!(
                                            "OpenAI Codex websocket warmup closed on attempt {}",
                                            attempt + 1,
                                        );
                                        reset_websocket_connection(&mut websocket_session);
                                        if attempt == 1 {
                                            websocket_session.websocket_disabled = true;
                                            mark_websocket_transport_unhealthy();
                                        }
                                        continue 'websocket_attempts;
                                    }
                                }
                            }
                        }

                        let websocket_request = prepare_websocket_request(
                            &api_request,
                            &websocket_session,
                            websocket_session.prewarmed,
                        );
                        let request_payload = match serde_json::to_string(&websocket_request) {
                            Ok(payload) => payload,
                            Err(error) => {
                                yield Ok(StreamDelta::Error {
                                    message: format!(
                                        "failed to encode websocket request: {error}"
                                    ),
                                    kind: StreamErrorKind::InvalidRequest,
                                });
                                return;
                            }
                        };

                        let send_result = if let Some(connection) = websocket_session.connection.as_mut() {
                            timeout(
                                WEBSOCKET_IO_TIMEOUT,
                                connection.send(WebSocketMessage::Text(request_payload.into())),
                            )
                            .await
                            .unwrap_or(Err(
                                tokio_tungstenite::tungstenite::Error::ConnectionClosed,
                            ))
                        } else {
                            Err(tokio_tungstenite::tungstenite::Error::ConnectionClosed)
                        };

                        if let Err(error) = send_result {
                            log::warn!(
                                "OpenAI Codex websocket send failed on attempt {}: {error}",
                                attempt + 1,
                            );
                            reset_websocket_connection(&mut websocket_session);
                            if attempt == 1 {
                                websocket_session.websocket_disabled = true;
                                mark_websocket_transport_unhealthy();
                            }
                            continue;
                        }

                        let mut usage: Option<Usage> = None;
                        let mut tool_calls: HashMap<String, ToolCallAccumulator> = HashMap::new();
                        let mut response_id: Option<String> = None;
                        let mut response_items = Vec::new();
                        let mut streamed_reasoning_summaries = HashSet::new();
                        let mut emitted_output = false;
                        let mut refused = false;

                        loop {
                            let message_result = if let Some(connection) =
                                websocket_session.connection.as_mut()
                            {
                                timeout(WEBSOCKET_IO_TIMEOUT, connection.next()).await.unwrap_or_else(|_| {
                                    log::warn!("OpenAI Codex websocket read timed out");
                                    None
                                })
                            } else {
                                None
                            };
                            let Some(message_result) = message_result else {
                                if emitted_output {
                                    end_websocket_turn(&mut websocket_session);
                                    yield Ok(StreamDelta::Error {
                                        message: "websocket closed before response.completed"
                                            .to_string(),
                                        kind: StreamErrorKind::ServerError,
                                    });
                                    return;
                                }
                                reset_websocket_connection(&mut websocket_session);
                                if attempt == 1 {
                                    websocket_session.websocket_disabled = true;
                                    mark_websocket_transport_unhealthy();
                                }
                                continue 'websocket_attempts;
                            };

                            let message = match message_result {
                                Ok(message) => message,
                                Err(error) => {
                                    if emitted_output {
                                        end_websocket_turn(&mut websocket_session);
                                        yield Ok(StreamDelta::Error {
                                            message: format!("websocket error: {error}"),
                                            kind: StreamErrorKind::ServerError,
                                        });
                                        return;
                                    }
                                    reset_websocket_connection(&mut websocket_session);
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                        mark_websocket_transport_unhealthy();
                                    }
                                    continue 'websocket_attempts;
                                }
                            };

                            match message {
                                WebSocketMessage::Text(text) => {
                                    if let Some(error) = parse_wrapped_websocket_error_event(&text)
                                    {
                                        let kind =
                                            websocket_error_kind(&error);
                                        // A quota rejection carries the delay the service wants
                                        // observed, so it is surfaced (with its hint) for the
                                        // caller's retry loop rather than re-sent immediately.
                                        // The connection-limit signal shares the status but is a
                                        // transport condition, so it still falls back at once.
                                        if emitted_output || is_websocket_quota_rejection(&error) {
                                            end_websocket_turn(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message: error.message,
                                                kind,
                                            });
                                            return;
                                        }
                                        if error.status == StatusCode::UNAUTHORIZED
                                            || error.status == StatusCode::UPGRADE_REQUIRED
                                            || error.status.is_client_error()
                                        {
                                            websocket_session.websocket_disabled = true;
                                        }
                                        reset_websocket_connection(&mut websocket_session);
                                        continue 'websocket_attempts;
                                    }
                                    let event = match decode_stream_event(&text) {
                                        Ok(event) => event,
                                        Err(error) => {
                                            end_websocket_turn(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message: error.to_string(),
                                                kind: StreamErrorKind::ServerError,
                                            });
                                            return;
                                        }
                                    };
                                    match event.r#type.as_str() {
                                            "response.output_text.delta" => {
                                                if let Some(delta) = event.delta {
                                                    emitted_output = true;
                                                    yield Ok(StreamDelta::TextDelta {
                                                        delta,
                                                        block_index: output_block_index(event.output_index),
                                                    });
                                                }
                                            }
                                            "response.refusal.delta" => {
                                                refused = true;
                                                if let Some(delta) = event.delta {
                                                    emitted_output = true;
                                                    yield Ok(StreamDelta::TextDelta {
                                                        delta,
                                                        block_index: output_block_index(
                                                            event.output_index,
                                                        ),
                                                    });
                                                }
                                            }
                                            "response.reasoning_summary_text.delta" => {
                                                if let Some(delta) = event.delta {
                                                    let output_index = event.output_index.unwrap_or(0);
                                                    streamed_reasoning_summaries.insert(output_index);
                                                    emitted_output = true;
                                                    yield Ok(StreamDelta::ThinkingDelta {
                                                        delta,
                                                        block_index: reasoning_summary_block_index(
                                                            Some(output_index),
                                                        ),
                                                    });
                                                }
                                            }
                                            "response.function_call_arguments.delta" => {
                                                let block_index = event
                                                    .output_index
                                                    .map(|index| index.saturating_mul(2));
                                                if let (Some(call_id), Some(delta)) =
                                                    (event.call_id, event.delta)
                                                {
                                                    emitted_output = true;
                                                    let order = tool_calls.len();
                                                    let acc = tool_calls
                                                        .entry(call_id.clone())
                                                        .or_insert_with(|| ToolCallAccumulator {
                                                            id: call_id,
                                                            name: event.name.unwrap_or_default(),
                                                            arguments: String::new(),
                                                            order,
                                                            block_index,
                                                        });
                                                    acc.arguments.push_str(&delta);
                                                }
                                            }
                                            "response.output_item.done" => {
                                                let item = match decode_output_item(event.item) {
                                                    Ok(item) => item,
                                                    Err(error) => {
                                                        end_websocket_turn(
                                                            &mut websocket_session,
                                                        );
                                                        yield Ok(StreamDelta::Error {
                                                            message: error.to_string(),
                                                            kind: StreamErrorKind::ServerError,
                                                        });
                                                        return;
                                                    }
                                                };
                                                let block_index = event.output_index.unwrap_or(0);
                                                accumulate_completed_tool_call(
                                                    &item,
                                                    block_index,
                                                    &mut tool_calls,
                                                );
                                                let include_summary = !streamed_reasoning_summaries
                                                    .contains(&block_index);
                                                for delta in output_item_stream_deltas(
                                                    &item,
                                                    block_index,
                                                    include_summary,
                                                ) {
                                                    emitted_output = true;
                                                    yield Ok(delta);
                                                }
                                                if let Some(item) = output_item_to_input_item(item) {
                                                    response_items.push(item);
                                                }
                                            }
                                            "response.completed"
                                            | "response.incomplete"
                                            | "response.done" => {
                                                let response_status = event
                                                    .response
                                                    .as_ref()
                                                    .and_then(|response| response.status);
                                                let incomplete_reason = event
                                                    .response
                                                    .as_ref()
                                                    .and_then(|response| {
                                                        response.incomplete_details.as_ref()
                                                    })
                                                    .and_then(|details| details.reason.clone());
                                                if let Some(resp) = event.response {
                                                    if let Some(u) = resp.usage {
                                                        usage = Some(usage_from_api_usage(&u));
                                                    }
                                                    if let Some(id) = resp.id {
                                                        response_id = Some(id);
                                                    }
                                                }
                                                let final_status = match event.r#type.as_str() {
                                                    "response.incomplete" => {
                                                        Some(ApiStatus::Incomplete)
                                                    }
                                                    "response.done" => response_status
                                                        .or(Some(ApiStatus::Completed)),
                                                    _ => Some(ApiStatus::Completed),
                                                };
                                                let stop_reason = stop_reason_from_stream_state(
                                                    &tool_calls,
                                                    final_status,
                                                    refused,
                                                    incomplete_reason.as_deref(),
                                                );
                                                if stop_reason == Some(StopReason::ToolUse) {
                                                    for delta in
                                                        emit_accumulated_tool_calls(&tool_calls)
                                                    {
                                                        yield Ok(delta);
                                                    }
                                                }
                                                if let Some(u) = usage.take() {
                                                    yield Ok(StreamDelta::Usage(u));
                                                }
                                                websocket_session.last_request = Some(api_request.clone());
                                                websocket_session.last_response_id = response_id;
                                                websocket_session.last_response_items = response_items;
                                                websocket_session.prewarmed = false;
                                                // Clean completion: the turn is no
                                                // longer in flight, so the next turn
                                                // may reuse this connection/baseline.
                                                websocket_session.in_flight = false;
                                                yield Ok(StreamDelta::Done {
                                                    stop_reason,
                                                });
                                                return;
                                            }
                                            "response.failed" => {
                                                websocket_session.last_request = None;
                                                websocket_session.last_response_id = None;
                                                websocket_session.last_response_items.clear();
                                                websocket_session.prewarmed = false;
                                                // The turn ends here, so the session must stop
                                                // counting as in-flight or it can never be evicted.
                                                websocket_session.in_flight = false;
                                                let failure =
                                                    codex_response_failed_error(event.response);
                                                if let Some(usage) = failure.usage {
                                                    yield Ok(StreamDelta::Usage(usage));
                                                }
                                                yield Ok(StreamDelta::Error {
                                                    message: failure.message,
                                                    kind: failure.kind,
                                                });
                                                return;
                                            }
                                            _ => {}
                                    }
                                }
                                WebSocketMessage::Binary(bytes) => {
                                    let text = match String::from_utf8(bytes.to_vec()) {
                                        Ok(text) => text,
                                        Err(error) => {
                                            end_websocket_turn(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message: format!(
                                                    "invalid OpenAI Codex websocket UTF-8: {error}"
                                                ),
                                                kind: StreamErrorKind::ServerError,
                                            });
                                            return;
                                        }
                                    };
                                        if let Some(error) =
                                            parse_wrapped_websocket_error_event(&text)
                                        {
                                            let kind =
                                                websocket_error_kind(&error);
                                            // Same split as the text frame: a quota rejection is
                                            // surfaced with its delay; the connection-limit signal
                                            // keeps falling back to HTTP immediately.
                                            if emitted_output
                                                || is_websocket_quota_rejection(&error)
                                            {
                                                end_websocket_turn(&mut websocket_session);
                                                yield Ok(StreamDelta::Error {
                                                    message: error.message,
                                                    kind,
                                                });
                                                return;
                                            }
                                            if error.status == StatusCode::UNAUTHORIZED
                                                || error.status == StatusCode::UPGRADE_REQUIRED
                                                || error.status.is_client_error()
                                            {
                                                websocket_session.websocket_disabled = true;
                                            }
                                            reset_websocket_connection(&mut websocket_session);
                                            continue 'websocket_attempts;
                                        }

                                        let event = match decode_stream_event(&text) {
                                            Ok(event) => event,
                                            Err(error) => {
                                                end_websocket_turn(
                                                    &mut websocket_session,
                                                );
                                                yield Ok(StreamDelta::Error {
                                                    message: error.to_string(),
                                                    kind: StreamErrorKind::ServerError,
                                                });
                                                return;
                                            }
                                        };
                                            match event.r#type.as_str() {
                                                "response.output_text.delta" => {
                                                    if let Some(delta) = event.delta {
                                                        emitted_output = true;
                                                        yield Ok(StreamDelta::TextDelta {
                                                            delta,
                                                            block_index: output_block_index(event.output_index),
                                                        });
                                                    }
                                                }
                                                "response.refusal.delta" => {
                                                    refused = true;
                                                    if let Some(delta) = event.delta {
                                                        emitted_output = true;
                                                        yield Ok(StreamDelta::TextDelta {
                                                            delta,
                                                            block_index: output_block_index(
                                                                event.output_index,
                                                            ),
                                                        });
                                                    }
                                                }
                                                "response.reasoning_summary_text.delta" => {
                                                    if let Some(delta) = event.delta {
                                                        let output_index =
                                                            event.output_index.unwrap_or(0);
                                                        streamed_reasoning_summaries
                                                            .insert(output_index);
                                                        emitted_output = true;
                                                        yield Ok(StreamDelta::ThinkingDelta {
                                                            delta,
                                                            block_index:
                                                                reasoning_summary_block_index(Some(
                                                                    output_index,
                                                                )),
                                                        });
                                                    }
                                                }
                                                "response.function_call_arguments.delta" => {
                                                    let block_index = event
                                                        .output_index
                                                        .map(|index| index.saturating_mul(2));
                                                    if let (Some(call_id), Some(delta)) =
                                                        (event.call_id, event.delta)
                                                    {
                                                        emitted_output = true;
                                                        let order = tool_calls.len();
                                                        let acc = tool_calls
                                                            .entry(call_id.clone())
                                                            .or_insert_with(|| ToolCallAccumulator {
                                                                id: call_id,
                                                                name: event.name.unwrap_or_default(),
                                                                arguments: String::new(),
                                                                order,
                                                                block_index,
                                                            });
                                                        acc.arguments.push_str(&delta);
                                                    }
                                                }
                                                "response.output_item.done" => {
                                                    let item =
                                                        match decode_output_item(event.item) {
                                                            Ok(item) => item,
                                                            Err(error) => {
                                                                end_websocket_turn(
                                                                    &mut websocket_session,
                                                                );
                                                                yield Ok(StreamDelta::Error {
                                                                    message: error.to_string(),
                                                                    kind: StreamErrorKind::ServerError,
                                                                });
                                                                return;
                                                            }
                                                        };
                                                    let block_index =
                                                        event.output_index.unwrap_or(0);
                                                    accumulate_completed_tool_call(
                                                        &item,
                                                        block_index,
                                                        &mut tool_calls,
                                                    );
                                                    let include_summary =
                                                        !streamed_reasoning_summaries
                                                            .contains(&block_index);
                                                    for delta in output_item_stream_deltas(
                                                        &item,
                                                        block_index,
                                                        include_summary,
                                                    ) {
                                                        emitted_output = true;
                                                        yield Ok(delta);
                                                    }
                                                    if let Some(item) =
                                                        output_item_to_input_item(item)
                                                    {
                                                        response_items.push(item);
                                                    }
                                                }
                                                "response.completed"
                                                | "response.incomplete"
                                                | "response.done" => {
                                                    let response_status = event
                                                        .response
                                                        .as_ref()
                                                        .and_then(|response| response.status);
                                                    let incomplete_reason = event
                                                        .response
                                                        .as_ref()
                                                        .and_then(|response| {
                                                            response.incomplete_details.as_ref()
                                                        })
                                                        .and_then(|details| details.reason.clone());
                                                    if let Some(resp) = event.response {
                                                        if let Some(u) = resp.usage {
                                                            usage = Some(usage_from_api_usage(&u));
                                                        }
                                                        if let Some(id) = resp.id {
                                                            response_id = Some(id);
                                                        }
                                                    }
                                                    let final_status =
                                                        match event.r#type.as_str() {
                                                            "response.incomplete" => {
                                                                Some(ApiStatus::Incomplete)
                                                            }
                                                            "response.done" => response_status
                                                                .or(Some(ApiStatus::Completed)),
                                                            _ => Some(ApiStatus::Completed),
                                                        };
                                                    let stop_reason =
                                                        stop_reason_from_stream_state(
                                                            &tool_calls,
                                                            final_status,
                                                            refused,
                                                            incomplete_reason.as_deref(),
                                                        );
                                                    if stop_reason
                                                        == Some(StopReason::ToolUse)
                                                    {
                                                        for delta in
                                                            emit_accumulated_tool_calls(&tool_calls)
                                                        {
                                                            yield Ok(delta);
                                                        }
                                                    }
                                                    if let Some(u) = usage.take() {
                                                        yield Ok(StreamDelta::Usage(u));
                                                    }
                                                    websocket_session.last_request =
                                                        Some(api_request.clone());
                                                    websocket_session.last_response_id = response_id;
                                                    websocket_session.last_response_items =
                                                        response_items;
                                                    websocket_session.prewarmed = false;
                                                    // Clean completion: the turn is no
                                                    // longer in flight, so the next
                                                    // turn may reuse the connection.
                                                    websocket_session.in_flight = false;
                                                    yield Ok(StreamDelta::Done {
                                                        stop_reason,
                                                    });
                                                    return;
                                                }
                                                "response.failed" => {
                                                    websocket_session.last_request = None;
                                                    websocket_session.last_response_id = None;
                                                    websocket_session.last_response_items.clear();
                                                    websocket_session.prewarmed = false;
                                                    // The turn ends here, so the session must stop
                                                    // counting as in-flight or it can never be
                                                    // evicted.
                                                    websocket_session.in_flight = false;
                                                    let failure =
                                                        codex_response_failed_error(event.response);
                                                    if let Some(usage) = failure.usage {
                                                        yield Ok(StreamDelta::Usage(usage));
                                                    }
                                                    yield Ok(StreamDelta::Error {
                                                        message: failure.message,
                                                        kind: failure.kind,
                                                    });
                                                    return;
                                                }
                                                _ => {}
                                    }
                                }
                                WebSocketMessage::Ping(payload) => {
                                    if let Some(connection) = websocket_session.connection.as_mut()
                                        && let Err(error) = connection
                                            .send(WebSocketMessage::Pong(payload))
                                            .await
                                    {
                                        if emitted_output {
                                            end_websocket_turn(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message: format!("websocket pong failed: {error}"),
                                                kind: StreamErrorKind::ServerError,
                                            });
                                            return;
                                        }
                                        reset_websocket_connection(&mut websocket_session);
                                        if attempt == 1 {
                                            websocket_session.websocket_disabled = true;
                                            mark_websocket_transport_unhealthy();
                                        }
                                        continue 'websocket_attempts;
                                    }
                                }
                                WebSocketMessage::Pong(_) | WebSocketMessage::Frame(_) => {}
                                WebSocketMessage::Close(_) => {
                                    if emitted_output {
                                        end_websocket_turn(&mut websocket_session);
                                        yield Ok(StreamDelta::Error {
                                            message: "websocket closed before response.completed"
                                                .to_string(),
                                            kind: StreamErrorKind::ServerError,
                                        });
                                        return;
                                    }
                                    reset_websocket_connection(&mut websocket_session);
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                        mark_websocket_transport_unhealthy();
                                    }
                                    continue 'websocket_attempts;
                                }
                            }
                        }
                    }
                }
                // The websocket turn did not complete (disabled, or attempts
                // exhausted); the turn now falls through to the HTTP SSE path, so
                // no websocket turn is in flight.
                websocket_session.in_flight = false;
                sse_turn_state = websocket_session.turn_state.clone();
                drop(websocket_session);
            }

            let headers = match self.build_headers(
                true,
                request.session_id.as_deref(),
                sse_turn_state.as_deref(),
            ) {
                Ok(headers) => headers,
                Err(error) => {
                    yield Ok(StreamDelta::Error {
                        message: error.to_string(),
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };

            let Ok(response) = self.client
                .post(codex_url(&self.base_url))
                .headers(headers)
                .json(&api_request)
                .send()
                .await
            else {
                yield Err(anyhow::anyhow!("request failed"));
                return;
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
                log::warn!("OpenAI Codex error status={status} body={body}");
                yield Ok(StreamDelta::Error { message: body, kind });
                return;
            }

            if let Some(session_id) = request.session_id.as_deref() {
                let turn_state = response
                    .headers()
                    .get(OPENAI_CODEX_TURN_STATE_HEADER)
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                if let Some(turn_state) = turn_state {
                    let session = self.websocket_session(session_id).await;
                    let mut websocket_session = session.lock().await;
                    websocket_session.turn_state = Some(turn_state);
                }
            }

            let mut sse = SseLineBuffer::new();
            let mut stream = response.bytes_stream();
            let mut usage: Option<Usage> = None;
            let mut tool_calls: HashMap<String, ToolCallAccumulator> = HashMap::new();
            let mut final_status: Option<ApiStatus> = None;
            let mut streamed_reasoning_summaries = HashSet::new();
            let mut refused = false;
            let mut incomplete_reason: Option<String> = None;

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error"));
                    return;
                };
                sse.extend(&chunk);

                while let Some(line) = sse.next_line() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    if data == "[DONE]" {
                        let Some(stop_reason) =
                            stop_reason_from_stream_state(
                                &tool_calls,
                                final_status,
                                refused,
                                incomplete_reason.as_deref(),
                            )
                        else {
                            yield Ok(StreamDelta::Error {
                                message: "OpenAI Codex stream sent [DONE] before a terminal response event"
                                    .to_owned(),
                                kind: StreamErrorKind::ServerError,
                            });
                            return;
                        };
                        if stop_reason == StopReason::ToolUse {
                            for delta in emit_accumulated_tool_calls(&tool_calls) {
                                yield Ok(delta);
                            }
                        }
                        if let Some(u) = usage.take() {
                            yield Ok(StreamDelta::Usage(u));
                        }
                        yield Ok(StreamDelta::Done {
                            stop_reason: Some(stop_reason),
                        });
                        return;
                    }

                    let event = match decode_stream_event(data) {
                        Ok(event) => event,
                        Err(error) => {
                            yield Ok(StreamDelta::Error {
                                message: format!(
                                    "invalid OpenAI Codex Responses stream event: {error}"
                                ),
                                kind: StreamErrorKind::ServerError,
                            });
                            return;
                        }
                    };
                    match event.r#type.as_str() {
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
                            "response.reasoning_summary_text.delta" => {
                                if let Some(delta) = event.delta {
                                    let output_index = event.output_index.unwrap_or(0);
                                    streamed_reasoning_summaries.insert(output_index);
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        delta,
                                        block_index: reasoning_summary_block_index(Some(
                                            output_index,
                                        )),
                                    });
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                let block_index = event
                                    .output_index
                                    .map(|index| index.saturating_mul(2));
                                if let (Some(call_id), Some(delta)) = (event.call_id, event.delta) {
                                    let order = tool_calls.len();
                                    let acc = tool_calls.entry(call_id.clone()).or_insert_with(|| {
                                        ToolCallAccumulator {
                                            id: call_id,
                                            name: event.name.unwrap_or_default(),
                                            arguments: String::new(),
                                            order,
                                            block_index,
                                        }
                                    });
                                    acc.arguments.push_str(&delta);
                                }
                            }
                            "response.output_item.done" => {
                                let item = match decode_output_item(event.item) {
                                    Ok(item) => item,
                                    Err(error) => {
                                        yield Ok(StreamDelta::Error {
                                            message: error.to_string(),
                                            kind: StreamErrorKind::ServerError,
                                        });
                                        return;
                                    }
                                };
                                let block_index = event.output_index.unwrap_or(0);
                                accumulate_completed_tool_call(
                                    &item,
                                    block_index,
                                    &mut tool_calls,
                                );
                                let include_summary =
                                    !streamed_reasoning_summaries.contains(&block_index);
                                for delta in output_item_stream_deltas(
                                    &item,
                                    block_index,
                                    include_summary,
                                ) {
                                    yield Ok(delta);
                                }
                            }
                            "response.completed" | "response.incomplete" | "response.done" => {
                                let response_status = event
                                    .response
                                    .as_ref()
                                    .and_then(|response| response.status);
                                incomplete_reason = event
                                    .response
                                    .as_ref()
                                    .and_then(|response| response.incomplete_details.as_ref())
                                    .and_then(|details| details.reason.clone());
                                if let Some(resp) = event.response
                                    && let Some(u) = resp.usage
                                {
                                    usage = Some(usage_from_api_usage(&u));
                                }
                                final_status = match event.r#type.as_str() {
                                    "response.incomplete" => Some(ApiStatus::Incomplete),
                                    "response.done" => {
                                        response_status.or(Some(ApiStatus::Completed))
                                    }
                                    _ => Some(ApiStatus::Completed),
                                };
                            }
                            "response.failed" => {
                                let failure = codex_response_failed_error(event.response);
                                if let Some(usage) = failure.usage {
                                    yield Ok(StreamDelta::Usage(usage));
                                }
                                yield Ok(StreamDelta::Error {
                                    message: failure.message,
                                    kind: failure.kind,
                                });
                                return;
                            }
                            _ => {}
                    }
                }
            }

            let Some(stop_reason) = stop_reason_from_stream_state(
                &tool_calls,
                final_status,
                refused,
                incomplete_reason.as_deref(),
            ) else {
                yield Ok(StreamDelta::Error {
                    message: "OpenAI Codex stream ended before a terminal response event".to_owned(),
                    kind: StreamErrorKind::ServerError,
                });
                return;
            };
            if stop_reason == StopReason::ToolUse {
                for delta in emit_accumulated_tool_calls(&tool_calls) {
                    yield Ok(delta);
                }
            }
            if let Some(u) = usage {
                yield Ok(StreamDelta::Usage(u));
            }
            yield Ok(StreamDelta::Done {
                stop_reason: Some(stop_reason),
            });
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "openai-codex"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }
}

// ============================================================================
// Input building
// ============================================================================

fn build_api_input(request: &ChatRequest) -> Vec<ApiInputItem> {
    let mut items = Vec::new();

    // Convert user/assistant messages. The system prompt is sent separately as
    // `instructions`, matching pi's Codex transport.
    for msg in &request.messages {
        let role = match msg.role {
            agent_sdk_foundation::llm::Role::User => ApiRole::User,
            agent_sdk_foundation::llm::Role::Assistant => ApiRole::Assistant,
        };
        match &msg.content {
            Content::Text(text) => {
                items.push(ApiInputItem::Message(ApiMessage {
                    role,
                    content: ApiMessageContent::Text(text.clone()),
                    phase: api_message_phase(role, false),
                }));
            }
            Content::Blocks(blocks) => append_block_input(&mut items, role, blocks),
        }
    }

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
                let part = if matches!(role, ApiRole::Assistant) {
                    ApiInputContent::OutputText { text: text.clone() }
                } else {
                    ApiInputContent::InputText { text: text.clone() }
                };
                content_parts.push(part);
            }
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
                if provider == OPENAI_RESPONSES_REASONING_PROVIDER
                    && data.get("type").and_then(serde_json::Value::as_str)
                        == Some("reasoning") =>
            {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                items.push(ApiInputItem::OpaqueReasoning(data.clone()));
            }
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::OpaqueReasoning { .. } => {}
            ContentBlock::Image { source } => content_parts.push(ApiInputContent::Image {
                image_url: format!("data:{};base64,{}", source.media_type, source.data),
            }),
            ContentBlock::Document { source } => content_parts.push(ApiInputContent::File {
                filename: suggested_filename(&source.media_type),
                file_data: format!("data:{};base64,{}", source.media_type, source.data),
            }),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                flush_message_parts(items, role, phase.clone(), &mut content_parts);
                items.push(ApiInputItem::FunctionCall(ApiFunctionCall::new(
                    id.clone(),
                    name.clone(),
                    serde_json::to_string(input).unwrap_or_default(),
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

fn api_message_phase(role: ApiRole, has_tool_use: bool) -> Option<String> {
    match (role, has_tool_use) {
        (ApiRole::Assistant, true) => Some("commentary".to_owned()),
        (ApiRole::Assistant, false) => Some("final_answer".to_owned()),
        (ApiRole::User, _) => None,
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
    provider == OPENAI_RESPONSES_REASONING_PROVIDER
        && data.get("type").and_then(serde_json::Value::as_str) == Some(OPENAI_MESSAGE_ITEM_TYPE)
        && data.get("content").is_none()
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

/// Recursively fix a JSON schema for `OpenAI` strict mode.
///
/// Adds `additionalProperties: false`, marks every property required, and — to
/// keep previously-optional properties from being forced to fabricated values —
/// wraps optional properties in `anyOf: [..., {"type": "null"}]`. This mirrors
/// the sibling `openai_responses` provider so a tool schema behaves identically
/// across both providers.
fn fix_schema_for_strict_mode(schema: &mut serde_json::Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    // Check if this is an object type schema
    let is_object_type = obj
        .get("type")
        .is_some_and(|t| t.as_str() == Some("object"));

    if is_object_type {
        // Add additionalProperties: false
        obj.insert(
            "additionalProperties".to_owned(),
            serde_json::Value::Bool(false),
        );

        // Ensure properties and required exist (strict mode needs them even if empty)
        obj.entry("properties".to_owned())
            .or_insert_with(|| serde_json::json!({}));
        obj.entry("required".to_owned())
            .or_insert_with(|| serde_json::json!([]));

        // Collect the set of originally required keys
        let originally_required: std::collections::HashSet<String> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Wrap previously-optional properties in anyOf with null
        if let Some(serde_json::Value::Object(props)) = obj.get_mut("properties") {
            for (key, prop_schema) in props.iter_mut() {
                if !originally_required.contains(key) {
                    make_nullable(prop_schema);
                }
            }
        }

        // Ensure all properties are marked as required
        if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
            let all_keys: Vec<serde_json::Value> = props
                .keys()
                .map(|k| serde_json::Value::String(k.clone()))
                .collect();
            obj.insert("required".to_owned(), serde_json::Value::Array(all_keys));
        }
    }

    // Recursively process nested schemas
    if let Some(props) = obj.get_mut("properties")
        && let Some(props_obj) = props.as_object_mut()
    {
        for prop_schema in props_obj.values_mut() {
            fix_schema_for_strict_mode(prop_schema);
        }
    }

    // Process array items
    if let Some(items) = obj.get_mut("items") {
        fix_schema_for_strict_mode(items);
    }

    // Process anyOf/oneOf/allOf
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(arr) = obj.get_mut(key)
            && let Some(arr_items) = arr.as_array_mut()
        {
            for item in arr_items {
                fix_schema_for_strict_mode(item);
            }
        }
    }
}

/// Wrap a schema in `anyOf: [{original}, {"type": "null"}]` so the property
/// accepts its original type OR null. Appends the null variant when an `anyOf`
/// already exists.
fn make_nullable(schema: &mut serde_json::Value) {
    if let Some(any_of) = schema
        .as_object_mut()
        .and_then(|o| o.get_mut("anyOf"))
        .and_then(|v| v.as_array_mut())
    {
        let has_null = any_of
            .iter()
            .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
        if !has_null {
            any_of.push(serde_json::json!({"type": "null"}));
        }
        return;
    }

    let original = schema.clone();
    *schema = serde_json::json!({
        "anyOf": [original, {"type": "null"}]
    });
}

/// Check whether a JSON schema contains any object-typed schema without a
/// `properties` map (a free-form object). These are incompatible with
/// `OpenAI` strict mode and must disable it.
fn has_freeform_object(schema: &serde_json::Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };

    let is_object = obj
        .get("type")
        .is_some_and(|t| t.as_str() == Some("object"));

    if is_object && !obj.contains_key("properties") {
        return true;
    }

    if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
        for prop in props.values() {
            if has_freeform_object(prop) {
                return true;
            }
        }
    }

    if let Some(items) = obj.get("items")
        && has_freeform_object(items)
    {
        return true;
    }

    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(arr) = obj.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if has_freeform_object(item) {
                    return true;
                }
            }
        }
    }

    false
}

fn convert_tool(tool: agent_sdk_foundation::llm::Tool) -> ApiTool {
    // Strict mode requires additionalProperties: false on all objects and every
    // property in required, which is incompatible with free-form object schemas
    // (objects with no defined properties). Detect and skip strict for those —
    // matching the sibling openai_responses provider.
    let mut schema = tool.input_schema;
    let use_strict = if has_freeform_object(&schema) {
        log::debug!(
            "Tool '{}' has free-form object schema — disabling strict mode",
            tool.name
        );
        None
    } else {
        fix_schema_for_strict_mode(&mut schema);
        Some(true)
    };

    ApiTool {
        r#type: "function".to_owned(),
        name: tool.name,
        description: Some(tool.description),
        parameters: Some(schema),
        strict: use_strict,
    }
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

fn reasoning_output_item(fields: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    let mut item = fields.clone();
    item.insert(
        "type".to_owned(),
        serde_json::Value::String("reasoning".to_owned()),
    );
    serde_json::Value::Object(item)
}

fn reasoning_summary_texts(fields: &serde_json::Map<String, serde_json::Value>) -> Vec<String> {
    fields
        .get("summary")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|summary| {
            summary.get("type").and_then(serde_json::Value::as_str) == Some("summary_text")
        })
        .filter_map(|summary| {
            summary
                .get("text")
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect()
}

fn build_content_blocks(output: &[ApiOutputItem]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for item in output {
        match item {
            ApiOutputItem::Message {
                role,
                phase,
                content,
            } => {
                blocks.push(ContentBlock::OpaqueReasoning {
                    provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
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
            ApiOutputItem::Reasoning { fields } => {
                blocks.push(ContentBlock::OpaqueReasoning {
                    provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
                    data: reasoning_output_item(fields),
                });
                blocks.extend(reasoning_summary_texts(fields).into_iter().map(|thinking| {
                    ContentBlock::Thinking {
                        thinking,
                        signature: None,
                    }
                }));
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

const fn map_reasoning_effort(effort: ReasoningEffort) -> Effort {
    match effort {
        ReasoningEffort::Low => Effort::Low,
        ReasoningEffort::Medium => Effort::Medium,
        ReasoningEffort::High => Effort::High,
        ReasoningEffort::XHigh => Effort::Max,
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

fn codex_url(base_url: &str) -> String {
    let normalized = base_url.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_string()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

fn codex_websocket_url(base_url: &str) -> Result<url::Url> {
    let mut url = url::Url::parse(&codex_url(base_url))
        .context("failed to parse OpenAI Codex websocket URL")?;

    let scheme = match url.scheme() {
        "http" => Some("ws"),
        "https" => Some("wss"),
        _ => None,
    };

    if let Some(scheme) = scheme {
        let _ = url.set_scheme(scheme);
    }

    Ok(url)
}

fn extract_account_id(token: &str) -> Result<String> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("invalid OpenAI Codex OAuth token"))?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .context("failed to decode OpenAI Codex token payload")?;
    let payload: serde_json::Value =
        serde_json::from_slice(&decoded).context("failed to parse OpenAI Codex token payload")?;
    payload
        .get(OPENAI_CODEX_JWT_CLAIM_PATH)
        .and_then(|value| value.get("chatgpt_account_id"))
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("chatgpt_account_id missing from OpenAI Codex token"))
}

fn is_empty(value: &str) -> bool {
    value.trim().is_empty()
}

// ============================================================================
// Streaming helpers
// ============================================================================

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
    /// Registration order, used to assign deterministic, distinct block indices
    /// when emitting (`HashMap` iteration order is otherwise nondeterministic).
    order: usize,
    /// Responses output-item index, when the stream reports it.
    block_index: Option<usize>,
}

fn usage_from_api_usage(usage: &ApiUsage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .map_or(0, |details| details.cached_tokens),
        cache_creation_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .map_or(0, |details| details.cache_write_tokens),
    }
}

fn output_block_index(output_index: Option<usize>) -> usize {
    output_index.unwrap_or(0).saturating_mul(2)
}

fn reasoning_summary_block_index(output_index: Option<usize>) -> usize {
    output_block_index(output_index).saturating_add(1)
}

fn decode_stream_event(data: &str) -> Result<ApiStreamEvent> {
    serde_json::from_str(data).with_context(|| {
        // Include a bounded snippet of the offending event: SSE payloads
        // carry no credentials, and "invalid stream event" without the
        // event is undiagnosable in the field (2026-07-10, GPT-5.6 rollout).
        let mut snippet: String = data.chars().take(512).collect();
        if data.chars().count() > 512 {
            snippet.push('…');
        }
        format!("invalid OpenAI Codex Responses stream event: {snippet}")
    })
}

fn decode_output_item(item: Option<serde_json::Value>) -> Result<ApiOutputItem> {
    let item = item.context("OpenAI Codex output_item.done omitted item")?;
    serde_json::from_value(item).context("invalid OpenAI Codex output item")
}

fn output_item_stream_deltas(
    item: &ApiOutputItem,
    output_index: usize,
    include_summary: bool,
) -> Vec<StreamDelta> {
    let block_index = output_index.saturating_mul(2);
    match item {
        ApiOutputItem::Message { role, phase, .. } => {
            vec![StreamDelta::OpaqueReasoning {
                provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
                data: message_state_marker(role, phase.as_deref()),
                block_index,
            }]
        }
        ApiOutputItem::Reasoning { fields } => {
            let mut deltas = vec![StreamDelta::OpaqueReasoning {
                provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
                data: reasoning_output_item(fields),
                block_index,
            }];
            if include_summary {
                let summary_block_index = block_index.saturating_add(1);
                deltas.extend(reasoning_summary_texts(fields).into_iter().map(|delta| {
                    StreamDelta::ThinkingDelta {
                        delta,
                        block_index: summary_block_index,
                    }
                }));
            }
            deltas
        }
        ApiOutputItem::FunctionCall { .. } | ApiOutputItem::Unknown => Vec::new(),
    }
}

fn accumulate_completed_tool_call(
    item: &ApiOutputItem,
    output_index: usize,
    tool_calls: &mut HashMap<String, ToolCallAccumulator>,
) {
    let ApiOutputItem::FunctionCall {
        call_id,
        name,
        arguments,
    } = item
    else {
        return;
    };

    let order = tool_calls.len();
    let accumulator = tool_calls
        .entry(call_id.clone())
        .or_insert_with(|| ToolCallAccumulator {
            id: call_id.clone(),
            name: name.clone(),
            arguments: String::new(),
            order,
            block_index: Some(output_index.saturating_mul(2)),
        });
    accumulator.name.clone_from(name);
    accumulator.arguments.clone_from(arguments);
    accumulator.block_index = Some(output_index.saturating_mul(2));
}

fn emit_accumulated_tool_calls(
    tool_calls: &HashMap<String, ToolCallAccumulator>,
) -> Vec<StreamDelta> {
    // Assign distinct, monotonically increasing block indices in registration
    // order. The previous code gave every call the same index (1) and iterated
    // HashMap::values(), so StreamAccumulator's stable sort preserved
    // nondeterministic insertion order for multi-tool turns.
    let mut accs: Vec<&ToolCallAccumulator> = tool_calls.values().collect();
    accs.sort_by_key(|acc| (acc.block_index.unwrap_or(usize::MAX), acc.order));

    let mut deltas = Vec::with_capacity(accs.len() * 2);
    for (idx, acc) in accs.iter().enumerate() {
        let block_index = acc
            .block_index
            .unwrap_or_else(|| idx.saturating_add(1).saturating_mul(2));
        deltas.push(StreamDelta::ToolUseStart {
            id: acc.id.clone(),
            name: acc.name.clone(),
            block_index,
            thought_signature: None,
        });
        deltas.push(StreamDelta::ToolInputDelta {
            id: acc.id.clone(),
            delta: acc.arguments.clone(),
            block_index,
        });
    }
    deltas
}

fn stop_reason_from_stream_state(
    tool_calls: &HashMap<String, ToolCallAccumulator>,
    status: Option<ApiStatus>,
    refused: bool,
    incomplete_reason: Option<&str>,
) -> Option<StopReason> {
    let status = status?;
    Some(match status {
        ApiStatus::Incomplete => {
            incomplete_reason.map_or(StopReason::Unknown, incomplete_stop_reason)
        }
        ApiStatus::Completed if refused => StopReason::Refusal,
        ApiStatus::Completed if !tool_calls.is_empty() => StopReason::ToolUse,
        ApiStatus::Completed => StopReason::EndTurn,
        // Failed, a non-terminal status leaking into the final state, or a
        // status this build does not know: same Unknown as the old Failed
        // arm. Spelled out per variant so adding a status forces a
        // decision here.
        ApiStatus::Failed
        | ApiStatus::InProgress
        | ApiStatus::Queued
        | ApiStatus::Cancelled
        | ApiStatus::Other => StopReason::Unknown,
    })
}

fn incomplete_stop_reason(reason: &str) -> StopReason {
    match reason {
        "max_output_tokens" => StopReason::MaxTokens,
        "content_filter" => StopReason::Refusal,
        "model_context_window_exceeded" => StopReason::ModelContextWindowExceeded,
        _ => StopReason::Unknown,
    }
}

fn reset_websocket_connection(session: &mut WebsocketSessionState) {
    session.connection = None;
    if session.prewarmed {
        session.last_request = None;
        session.last_response_id = None;
        session.last_response_items.clear();
    }
    session.prewarmed = false;
}

/// Reset the connection *and* end the turn, for a stream that is about to yield
/// a terminal error and stop.
///
/// `in_flight` stays set when a stream is merely *dropped* — that is how the
/// next turn for the session learns its state is stale. A stream that finishes
/// by reporting an error has no such successor to warn: the turn is over and
/// the state is already reset here, so the marker must be cleared. Leaving it
/// set makes [`evict_idle_sessions`] skip the entry forever, and the bounded
/// session map then grows without limit — one stranded entry per distinct
/// session id that ever hit a terminal stream error.
fn end_websocket_turn(session: &mut WebsocketSessionState) {
    reset_websocket_connection(session);
    session.in_flight = false;
}

/// Bound the websocket-session map by evicting idle (not in-flight) sessions,
/// oldest-first by last use. The map exists for cross-turn reuse, so completed
/// sessions retain a cached baseline indefinitely; without eviction a host
/// serving many distinct sessions would leak memory and open sockets without
/// bound. Sessions whose lock is currently held (in use) or that are mid-turn
/// are retained.
fn evict_idle_sessions(sessions: &mut HashMap<String, Arc<Mutex<WebsocketSessionState>>>) {
    let mut candidates: Vec<(String, Option<Instant>)> = Vec::new();
    for (key, state) in sessions.iter() {
        if let Ok(guard) = state.try_lock()
            && !guard.in_flight
        {
            candidates.push((key.clone(), guard.last_used));
        }
    }
    // Oldest first; `None` (never used) sorts before `Some`.
    candidates.sort_by_key(|a| a.1);
    let evict_count = candidates.len().min(sessions.len() / 2 + 1);
    for (key, _) in candidates.into_iter().take(evict_count) {
        sessions.remove(&key);
    }
}

/// A decoded wrapped websocket error frame.
struct WrappedWebsocketError {
    status: StatusCode,
    message: String,
    /// The frame is the websocket connection-limit signal, which the service
    /// reports with a 429 even though no model quota was exhausted: it means
    /// *this transport* is full, not that the request must wait. Tracked
    /// separately so the immediate HTTP fallback that resolves it is not
    /// confused with a quota rejection, which must be waited out.
    connection_limit: bool,
    /// Delay the frame stated in a structured field, if any.
    ///
    /// Codex's `usage_limit_reached` frame carries its wait here and leaves the
    /// message as fixed prose ("The usage limit has been reached"), so this is
    /// the only hint that frame has — parsing the message would find nothing.
    reset_after: Option<Duration>,
}

/// Read the wait a wrapped error frame states in a structured field.
///
/// `resets_in_seconds` is preferred: it is relative, so it needs no wall-clock
/// arithmetic and cannot be skewed by clock drift. `resets_at` (Unix seconds) is
/// the fallback, converted defensively — an instant at or before now yields
/// `None` rather than a zero or negative wait. Either way the value passes
/// through the same ceiling as every other hint source.
fn websocket_reset_after(error: Option<&ApiWrappedWebsocketErrorBody>) -> Option<Duration> {
    let error = error?;
    if let Some(seconds) = error.resets_in_seconds {
        return crate::retry_hints::bounded_delay(seconds);
    }

    // Absolute form: convert against the wall clock. Every step that could
    // produce a nonsense wait yields `None` instead — a pre-epoch timestamp
    // (`try_from`), an instant at or before now (`checked_sub`), or one so far
    // out it cannot be a quota reset (`u32::try_from`, i.e. beyond ~136 years).
    let resets_at = u64::try_from(error.resets_at?).ok()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let remaining = resets_at.checked_sub(now)?;
    crate::retry_hints::bounded_delay(f64::from(u32::try_from(remaining).ok()?))
}

fn parse_wrapped_websocket_error_event(payload: &str) -> Option<WrappedWebsocketError> {
    let event: ApiWrappedWebsocketErrorEvent = serde_json::from_str(payload).ok()?;
    if event.kind != "error" {
        return None;
    }

    let reset_after = websocket_reset_after(event.error.as_ref());

    if event.error.as_ref().and_then(|error| error.code.as_deref())
        == Some(OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE)
    {
        let message = event
            .error
            .and_then(|error| error.message)
            .unwrap_or_else(|| "Responses websocket connection limit reached".to_string());
        return Some(WrappedWebsocketError {
            status: StatusCode::TOO_MANY_REQUESTS,
            message,
            connection_limit: true,
            reset_after,
        });
    }

    let status = StatusCode::from_u16(event.status?).ok()?;
    let message = event
        .error
        .and_then(|error| error.message)
        .unwrap_or_else(|| payload.to_string());
    if status.is_success() {
        None
    } else {
        Some(WrappedWebsocketError {
            status,
            message,
            connection_limit: false,
            reset_after,
        })
    }
}

/// `true` when the frame is a model-quota rejection rather than the
/// websocket connection-limit signal the service reports with the same status.
///
/// A quota rejection states how long to wait, so re-sending the request now —
/// over a fresh socket or the HTTP fallback — would spend the attempt inside
/// the window the service just asked us to sit out.
fn is_websocket_quota_rejection(error: &WrappedWebsocketError) -> bool {
    error.status == StatusCode::TOO_MANY_REQUESTS && !error.connection_limit
}

/// Resolve the message and kind of an in-band `response.failed` event.
///
/// The stream already opened with HTTP 200, so a rate limit reported this way
/// has neither a status nor a `Retry-After` header: the machine-readable code
/// is the only classification signal, and the delay — when the service states
/// one — is embedded in the message. Every other failure stays a (retriable)
/// server error, as before.
fn codex_response_failed_error(response: Option<ApiStreamResponse>) -> CodexResponseFailure {
    // A failed response still reports the tokens it burned. They are returned
    // alongside the error so the caller can yield them *before* the terminal
    // error delta: the consumer's accumulator only sees deltas it was handed,
    // and those tokens are billed whether or not the turn survived.
    let usage = response
        .as_ref()
        .and_then(|resp| resp.usage.as_ref())
        .map(usage_from_api_usage);
    let (code, message) = response
        .and_then(|resp| resp.error)
        .map_or((None, None), |error| (error.code, error.message));
    let message = message.unwrap_or_else(|| "Codex response failed".to_string());
    let kind = if matches!(
        code.as_deref(),
        Some("rate_limit_exceeded" | "rate_limit_error")
    ) {
        StreamErrorKind::RateLimited(crate::retry_hints::openai_retry_delay(&message))
    } else {
        StreamErrorKind::ServerError
    };
    CodexResponseFailure {
        message,
        kind,
        usage,
    }
}

/// A `response.failed` event decoded into what the stream must emit for it.
struct CodexResponseFailure {
    message: String,
    kind: StreamErrorKind,
    /// Usage the failed response reported, if any — emitted before the error.
    usage: Option<Usage>,
}

/// Classify a wrapped websocket error event by the status it reports.
///
/// The frame carries no HTTP headers, so a rate limit's delay comes from the
/// frame itself: the structured reset field when the service sent one (Codex's
/// `usage_limit_reached` frames do, and their message is fixed prose that says
/// nothing about timing), otherwise the message ("Please try again in 20s").
fn websocket_error_kind(error: &WrappedWebsocketError) -> StreamErrorKind {
    if error.status == StatusCode::TOO_MANY_REQUESTS {
        let delay = error
            .reset_after
            .or_else(|| crate::retry_hints::openai_retry_delay(&error.message));
        StreamErrorKind::RateLimited(delay)
    } else if error.status.is_server_error() {
        StreamErrorKind::ServerError
    } else {
        StreamErrorKind::InvalidRequest
    }
}

fn output_item_to_input_item(item: ApiOutputItem) -> Option<ApiInputItem> {
    match item {
        ApiOutputItem::Message {
            role,
            phase,
            content,
        } => {
            let role = if role == "user" {
                ApiRole::User
            } else {
                ApiRole::Assistant
            };
            let parts: Vec<ApiInputContent> = content
                .into_iter()
                .filter_map(|content| match content {
                    ApiOutputContent::Text { text }
                    | ApiOutputContent::Refusal { refusal: text }
                        if !text.is_empty() =>
                    {
                        Some(ApiInputContent::OutputText { text })
                    }
                    ApiOutputContent::Unknown
                    | ApiOutputContent::Text { .. }
                    | ApiOutputContent::Refusal { .. } => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(ApiInputItem::Message(ApiMessage {
                    role,
                    content: ApiMessageContent::Parts(parts),
                    phase,
                }))
            }
        }
        ApiOutputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => Some(ApiInputItem::FunctionCall(ApiFunctionCall::new(
            call_id, name, arguments,
        ))),
        ApiOutputItem::Reasoning { fields } => Some(ApiInputItem::OpaqueReasoning(
            reasoning_output_item(&fields),
        )),
        ApiOutputItem::Unknown => None,
    }
}

fn prepare_websocket_request(
    request: &ApiStreamingRequest,
    session: &WebsocketSessionState,
    allow_empty_delta: bool,
) -> ApiWebsocketRequest {
    let mut websocket_request = ApiWebsocketRequest::from(request);

    let Some(last_request) = session.last_request.as_ref() else {
        return websocket_request;
    };
    let Some(last_response_id) = session.last_response_id.as_ref() else {
        return websocket_request;
    };

    let mut previous_without_input = last_request.clone();
    previous_without_input.input.clear();
    let mut current_without_input = request.clone();
    current_without_input.input.clear();
    if previous_without_input != current_without_input {
        return websocket_request;
    }

    let mut baseline = last_request.input.clone();
    baseline.extend(session.last_response_items.clone());
    if request.input.starts_with(&baseline)
        && (allow_empty_delta || baseline.len() < request.input.len())
    {
        websocket_request.previous_response_id = Some(last_response_id.clone());
        websocket_request.input = request.input[baseline.len()..].to_vec();
    }

    websocket_request
}

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
struct ApiResponsesRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "is_empty")]
    instructions: &'a str,
    input: &'a [ApiInputItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiTextSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [&'static str]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiStreamingRequest {
    model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    instructions: String,
    input: Vec<ApiInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiTextSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    stream: bool,
}

#[derive(Clone, Serialize)]
struct ApiWebsocketRequest {
    #[serde(rename = "type")]
    kind: &'static str,
    model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    input: Vec<ApiInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ApiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiTextSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
}

impl From<&ApiStreamingRequest> for ApiWebsocketRequest {
    fn from(request: &ApiStreamingRequest) -> Self {
        Self {
            kind: "response.create",
            model: request.model.clone(),
            instructions: request.instructions.clone(),
            previous_response_id: None,
            input: request.input.clone(),
            tools: request.tools.clone(),
            max_output_tokens: request.max_output_tokens,
            reasoning: request.reasoning.clone(),
            tool_choice: request.tool_choice.clone(),
            parallel_tool_calls: request.parallel_tool_calls,
            store: request.store,
            text: request.text.clone(),
            include: request.include.clone(),
            prompt_cache_key: request.prompt_cache_key.clone(),
            stream: request.stream,
            generate: None,
        }
    }
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiTextSettings {
    verbosity: &'static str,
    /// Structured-output schema (`text.format`), set when the request carries a
    /// `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<ApiResponseTextFormat>,
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiResponseTextFormat {
    #[serde(rename = "type")]
    format_type: &'static str,
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

impl From<&ResponseFormat> for ApiResponseTextFormat {
    fn from(rf: &ResponseFormat) -> Self {
        Self {
            format_type: "json_schema",
            name: rf.name.clone(),
            schema: rf.schema.clone(),
            strict: rf.strict,
        }
    }
}

/// Responses API `tool_choice` wire format.
///
/// - `"auto"` — model decides (the Codex default).
/// - `{"type": "function", "name": "<name>"}` — force a specific function.
#[derive(Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum ApiToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        name: String,
    },
}

/// Map an optional [`ToolChoice`] onto the Codex wire `tool_choice`, defaulting
/// to `"auto"` (the historical Codex behavior) when unset.
fn codex_tool_choice(tool_choice: Option<&ToolChoice>) -> ApiToolChoice {
    match tool_choice {
        Some(ToolChoice::Tool(name)) => ApiToolChoice::Function {
            choice_type: "function",
            name: name.clone(),
        },
        _ => ApiToolChoice::Mode("auto"),
    }
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiReasoning {
    effort: ReasoningEffort,
}

#[derive(Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum ApiInputItem {
    Message(ApiMessage),
    FunctionCall(ApiFunctionCall),
    FunctionCallOutput(ApiFunctionCallOutput),
    OpaqueReasoning(serde_json::Value),
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    User,
    Assistant,
}

#[derive(Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Parts(Vec<ApiInputContent>),
}

#[derive(Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
enum ApiInputContent {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    Image { image_url: String },
    #[serde(rename = "input_file")]
    File { filename: String, file_data: String },
}

#[derive(Clone, PartialEq, Serialize)]
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

#[derive(Clone, PartialEq, Serialize)]
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

#[derive(Clone, PartialEq, Serialize)]
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
    error: Option<ApiErrorBody>,
    #[serde(default)]
    incomplete_details: Option<ApiIncompleteDetails>,
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
    // Non-terminal statuses ride lifecycle events (`response.created`
    // streams `"status":"in_progress"` on the GPT-5.6 ChatGPT backend,
    // observed live 2026-07-10) — they must parse without killing the
    // stream even though no consumer branches on them.
    InProgress,
    Queued,
    Cancelled,
    // Forward-compat: a status this build does not know is not a broken
    // stream. Consumers treat it like the existing non-completed arms.
    #[serde(other)]
    Other,
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
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(flatten)]
        fields: serde_json::Map<String, serde_json::Value>,
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
    output_index: Option<usize>,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    item: Option<serde_json::Value>,
    #[serde(default)]
    response: Option<ApiStreamResponse>,
}

#[derive(Deserialize)]
struct ApiStreamResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    usage: Option<ApiUsage>,
    #[serde(default)]
    error: Option<ApiErrorBody>,
    #[serde(default)]
    status: Option<ApiStatus>,
    #[serde(default)]
    incomplete_details: Option<ApiIncompleteDetails>,
}

#[derive(Deserialize)]
struct ApiErrorBody {
    #[serde(default)]
    message: Option<String>,
    /// Machine-readable error code (e.g. `rate_limit_exceeded`), used to
    /// classify a failure the HTTP status cannot describe because the stream
    /// already opened with 200.
    #[serde(default)]
    code: Option<String>,
}

#[derive(Deserialize)]
struct ApiWrappedWebsocketErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    /// Seconds until the exhausted usage limit resets. Codex states its
    /// standard `usage_limit_reached` delay here rather than in the message
    /// (which is the fixed prose "The usage limit has been reached"), so it is
    /// the only usable hint on that frame.
    #[serde(default)]
    resets_in_seconds: Option<f64>,
    /// Absolute reset instant (Unix seconds), sent by some frames instead of
    /// the relative field.
    #[serde(default)]
    resets_at: Option<i64>,
}

#[derive(Deserialize)]
struct ApiWrappedWebsocketErrorEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(alias = "status_code")]
    status: Option<u16>,
    #[serde(default)]
    error: Option<ApiWrappedWebsocketErrorBody>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_constant() {
        assert_eq!(MODEL_GPT54, "gpt-5.4");
        assert_eq!(MODEL_GPT53_CODEX, "gpt-5.3-codex");
        assert_eq!(MODEL_GPT52_CODEX, "gpt-5.2-codex");
    }

    #[test]
    fn warmup_quota_rejection_is_surfaced_while_connection_limit_still_falls_back()
    -> anyhow::Result<()> {
        // The warmup (`generate=false`) branches route on the same discriminator
        // as the in-flight ones: a quota rejection must be surfaced with its
        // delay rather than immediately re-sent inside the wait window, while
        // the connection-limit sentinel — reported with the same synthetic 429 —
        // must keep falling back to HTTP at once.
        let quota = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"code":"rate_limit_exceeded","message":"Rate limit reached. Please try again in 45s."}}"#,
        )
        .context("expected a wrapped error")?;
        assert!(is_websocket_quota_rejection(&quota));
        let kind = websocket_error_kind(&quota);
        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(45)))
        );
        assert!(kind.is_recoverable());

        // The same warmup path carries a structured reset hint when the frame is
        // Codex's standard usage-limit shape, whose prose states no timing.
        let usage_limit = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached.","resets_in_seconds":900}}"#,
        )
        .context("expected a wrapped error")?;
        assert!(is_websocket_quota_rejection(&usage_limit));
        assert_eq!(
            websocket_error_kind(&usage_limit),
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_mins(15))),
            "the structured reset must reach the retry loop through the warmup path"
        );

        let connection_limit = parse_wrapped_websocket_error_event(&format!(
            r#"{{"type":"error","status":429,"error":{{"code":"{OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE}","message":"limit"}}}}"#,
        ))
        .context("expected a wrapped error")?;
        assert!(
            !is_websocket_quota_rejection(&connection_limit),
            "the connection-limit sentinel must keep its immediate fallback"
        );
        Ok(())
    }

    #[test]
    fn in_band_response_failed_rate_limit_keeps_its_hint() -> anyhow::Result<()> {
        let event: ApiStreamEvent = serde_json::from_str(
            r#"{"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded","message":"Rate limit reached. Please try again in 20s."}}}"#,
        )?;
        let failure = codex_response_failed_error(event.response);

        assert!(failure.message.contains("try again in 20s"));
        assert_eq!(
            failure.kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(20)))
        );
        assert!(failure.kind.is_recoverable());
        Ok(())
    }

    #[test]
    fn in_band_response_failed_without_a_rate_limit_code_is_a_server_error() -> anyhow::Result<()> {
        let event: ApiStreamEvent = serde_json::from_str(
            r#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"upstream blew up"}}}"#,
        )?;
        let failure = codex_response_failed_error(event.response);

        assert_eq!(failure.message, "upstream blew up");
        assert_eq!(failure.kind, StreamErrorKind::ServerError);
        Ok(())
    }

    #[test]
    fn in_band_response_failed_keeps_the_usage_it_reported() -> anyhow::Result<()> {
        // The failed response still burned tokens; they must reach the stream so
        // the caller bills them, not vanish with the error.
        let event: ApiStreamEvent = serde_json::from_str(
            r#"{"type":"response.failed","response":{"usage":{"input_tokens":120,"output_tokens":34},"error":{"code":"rate_limit_exceeded","message":"Please try again in 20s."}}}"#,
        )?;
        let failure = codex_response_failed_error(event.response);

        let usage = failure.usage.context("failed response must report usage")?;
        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.output_tokens, 34);
        assert_eq!(
            failure.kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(20)))
        );
        Ok(())
    }

    #[test]
    fn test_codex_factory() {
        let provider = OpenAICodexResponsesProvider::codex("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai-codex");
    }

    #[test]
    fn test_gpt54_factory() {
        let provider = OpenAICodexResponsesProvider::gpt54("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT54);
        assert_eq!(provider.provider(), "openai-codex");
    }

    #[test]
    fn test_gpt53_codex_factory() {
        let provider = OpenAICodexResponsesProvider::gpt53_codex("test-key".to_string());
        assert_eq!(provider.model(), MODEL_GPT53_CODEX);
        assert_eq!(provider.provider(), "openai-codex");
    }

    #[test]
    fn test_reasoning_effort_serialization() {
        let low = serde_json::to_string(&ReasoningEffort::Low).unwrap();
        assert_eq!(low, "\"low\"");

        let xhigh = serde_json::to_string(&ReasoningEffort::XHigh).unwrap();
        assert_eq!(xhigh, "\"xhigh\"");
    }

    #[test]
    fn test_with_reasoning_effort() {
        let provider = OpenAICodexResponsesProvider::codex("test-key".to_string())
            .with_reasoning_effort(ReasoningEffort::High);
        let thinking = provider.thinking.as_ref().unwrap();
        assert!(matches!(thinking.effort, Some(Effort::High)));
    }

    #[test]
    fn test_build_api_reasoning_uses_explicit_effort() {
        let reasoning =
            build_api_reasoning(Some(&ThinkingConfig::adaptive_with_effort(Effort::Low))).unwrap();
        assert!(matches!(reasoning.effort, ReasoningEffort::Low));
    }

    #[test]
    fn test_build_api_reasoning_omits_adaptive_without_effort() {
        assert!(build_api_reasoning(Some(&ThinkingConfig::adaptive())).is_none());
    }

    #[test]
    fn test_openai_responses_accepts_adaptive_thinking() {
        let provider = OpenAICodexResponsesProvider::codex("test-key".to_string());
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

    fn test_token() -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!(
            r#"{{"{OPENAI_CODEX_JWT_CLAIM_PATH}":{{"chatgpt_account_id":"acct_123"}}}}"#
        ));
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn test_build_headers_match_codex_style_defaults() -> anyhow::Result<()> {
        let provider = OpenAICodexResponsesProvider::codex(test_token());

        let headers = provider.build_headers(true, Some("session-123"), None)?;
        assert_eq!(headers.get("originator").unwrap(), OPENAI_CODEX_ORIGINATOR);
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acct_123");
        assert_eq!(headers.get("session_id").unwrap(), "session-123");
        assert_eq!(headers.get("x-client-request-id").unwrap(), "session-123");
        assert_eq!(
            headers.get("OpenAI-Beta").unwrap(),
            OPENAI_CODEX_RESPONSES_BETA_HEADER
        );

        Ok(())
    }

    #[test]
    fn test_build_websocket_headers_match_codex_style_defaults() -> anyhow::Result<()> {
        let provider = OpenAICodexResponsesProvider::codex(test_token());

        let headers = provider.build_websocket_headers(Some("session-123"), Some("turn-1"))?;
        assert_eq!(headers.get("originator").unwrap(), OPENAI_CODEX_ORIGINATOR);
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acct_123");
        assert_eq!(headers.get("session_id").unwrap(), "session-123");
        assert_eq!(headers.get("x-client-request-id").unwrap(), "session-123");
        assert_eq!(
            headers.get(OPENAI_CODEX_TURN_STATE_HEADER).unwrap(),
            "turn-1"
        );
        assert_eq!(
            headers.get("OpenAI-Beta").unwrap(),
            OPENAI_CODEX_RESPONSES_WEBSOCKETS_BETA_HEADER,
        );

        Ok(())
    }

    #[test]
    fn test_build_headers_uses_configured_account_id_without_jwt_decode() -> anyhow::Result<()> {
        let provider = OpenAICodexResponsesProvider::codex("not-a-jwt".to_string())
            .with_account_id("acct_stored");

        let headers = provider.build_headers(true, Some("session-123"), Some("turn-1"))?;
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acct_stored");
        assert_eq!(
            headers.get(OPENAI_CODEX_TURN_STATE_HEADER).unwrap(),
            "turn-1"
        );

        Ok(())
    }

    #[test]
    fn test_request_serialization_includes_store_false() {
        let request = ApiStreamingRequest {
            model: MODEL_GPT53_CODEX.to_string(),
            instructions: "system".to_string(),
            input: Vec::new(),
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some(ApiToolChoice::Mode("auto")),
            parallel_tool_calls: Some(true),
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
                format: None,
            }),
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            prompt_cache_key: Some("session-123".to_string()),
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"store\":false"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_prepare_websocket_request_uses_previous_response_id_for_incremental_input() {
        let request = ApiStreamingRequest {
            model: MODEL_GPT53_CODEX.to_string(),
            instructions: "system".to_string(),
            input: vec![
                ApiInputItem::Message(ApiMessage {
                    role: ApiRole::User,
                    content: ApiMessageContent::Text("first".to_string()),
                    phase: None,
                }),
                ApiInputItem::Message(ApiMessage {
                    role: ApiRole::Assistant,
                    content: ApiMessageContent::Parts(vec![ApiInputContent::OutputText {
                        text: "answer".to_string(),
                    }]),
                    phase: Some("final_answer".to_owned()),
                }),
                ApiInputItem::Message(ApiMessage {
                    role: ApiRole::User,
                    content: ApiMessageContent::Text("follow up".to_string()),
                    phase: None,
                }),
            ],
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some(ApiToolChoice::Mode("auto")),
            parallel_tool_calls: None,
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
                format: None,
            }),
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            prompt_cache_key: Some("thread-1".to_string()),
            stream: true,
        };
        let previous_request = ApiStreamingRequest {
            input: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Text("first".to_string()),
                phase: None,
            })],
            ..request.clone()
        };
        let session = WebsocketSessionState {
            connection: None,
            last_request: Some(previous_request),
            last_response_id: Some("resp_prev".to_string()),
            last_response_items: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::Assistant,
                content: ApiMessageContent::Parts(vec![ApiInputContent::OutputText {
                    text: "answer".to_string(),
                }]),
                phase: Some("final_answer".to_owned()),
            })],
            turn_state: None,
            prewarmed: false,
            websocket_disabled: false,
            in_flight: false,
            last_used: None,
        };

        let websocket_request = prepare_websocket_request(&request, &session, false);
        assert_eq!(
            websocket_request.previous_response_id.as_deref(),
            Some("resp_prev")
        );
        assert_eq!(websocket_request.input.len(), 1);
        match &websocket_request.input[0] {
            ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Text(text),
                ..
            }) => assert_eq!(text, "follow up"),
            _ => panic!("expected incremental follow-up user message"),
        }
    }

    #[test]
    fn test_parse_wrapped_websocket_error_event_maps_http_status() -> anyhow::Result<()> {
        let payload = r#"{"type":"error","status":401,"error":{"message":"unauthorized"}}"#;
        let parsed =
            parse_wrapped_websocket_error_event(payload).context("expected a wrapped error")?;

        assert_eq!(parsed.status, StatusCode::UNAUTHORIZED);
        assert_eq!(parsed.message, "unauthorized");
        assert!(!parsed.connection_limit);
        Ok(())
    }

    #[test]
    fn test_parse_wrapped_websocket_error_event_maps_connection_limit() -> anyhow::Result<()> {
        let payload = format!(
            r#"{{"type":"error","status":429,"error":{{"code":"{OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE}","message":"limit"}}}}"#,
        );
        let parsed =
            parse_wrapped_websocket_error_event(&payload).context("expected a wrapped error")?;

        assert_eq!(parsed.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(parsed.message, "limit");
        assert!(parsed.connection_limit);
        // The connection limit is a transport condition, not a model quota: the
        // stream must keep falling back to HTTP at once rather than waiting.
        assert!(
            !is_websocket_quota_rejection(&parsed),
            "the connection-limit fallback must not be treated as a quota wait"
        );
        Ok(())
    }

    #[test]
    fn pre_output_websocket_quota_rejection_surfaces_with_its_hint() -> anyhow::Result<()> {
        // A quota 429 normally arrives before any output. It must be surfaced as
        // a recoverable rate limit carrying the advertised delay — not swallowed
        // into an immediate retry that re-sends inside the wait window.
        let payload = r#"{"type":"error","status":429,"error":{"code":"rate_limit_exceeded","message":"Rate limit reached. Please try again in 30s."}}"#;
        let parsed =
            parse_wrapped_websocket_error_event(payload).context("expected a wrapped error")?;

        assert!(!parsed.connection_limit);
        assert!(
            is_websocket_quota_rejection(&parsed),
            "a quota 429 must be surfaced rather than retried immediately"
        );

        let kind = websocket_error_kind(&parsed);
        assert_eq!(
            kind,
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(30)))
        );
        assert!(kind.is_recoverable());
        Ok(())
    }

    /// Current Unix time in seconds, for building `resets_at` fixtures.
    fn unix_now() -> i64 {
        i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or_default(),
        )
        .unwrap_or_default()
    }

    #[test]
    fn usage_limit_frame_carries_its_structured_reset() -> anyhow::Result<()> {
        // Codex's standard usage-limit frame states the wait in a structured
        // field and leaves the message as fixed prose with no timing in it, so
        // the field is the only hint there is.
        let parsed = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached.","resets_in_seconds":1800}}"#,
        )
        .context("expected a wrapped error")?;

        assert_eq!(
            websocket_error_kind(&parsed),
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_mins(30)))
        );
        Ok(())
    }

    #[test]
    fn usage_limit_reset_falls_back_to_the_absolute_instant() -> anyhow::Result<()> {
        let future = unix_now() + 600;
        let parsed = parse_wrapped_websocket_error_event(&format!(
            r#"{{"type":"error","status":429,"error":{{"type":"usage_limit_reached","message":"The usage limit has been reached.","resets_at":{future}}}}}"#,
        ))
        .context("expected a wrapped error")?;

        let StreamErrorKind::RateLimited(Some(delay)) = websocket_error_kind(&parsed) else {
            anyhow::bail!("a usage-limit frame must be a rate limit with a delay");
        };
        // Computed against the wall clock, so assert a window rather than an
        // exact value.
        assert!(
            delay <= std::time::Duration::from_mins(10)
                && delay >= std::time::Duration::from_secs(590),
            "the absolute reset must convert to a ~600s wait, got {delay:?}"
        );
        Ok(())
    }

    #[test]
    fn usage_limit_reset_in_the_past_or_absent_reports_no_hint() -> anyhow::Result<()> {
        // A reset instant that has already elapsed says nothing about how long to
        // wait: it must not become a zero (retry-instantly) or negative delay.
        let past = unix_now() - 60;
        let elapsed = parse_wrapped_websocket_error_event(&format!(
            r#"{{"type":"error","status":429,"error":{{"type":"usage_limit_reached","message":"The usage limit has been reached.","resets_at":{past}}}}}"#,
        ))
        .context("expected a wrapped error")?;
        assert_eq!(
            websocket_error_kind(&elapsed),
            StreamErrorKind::RateLimited(None)
        );

        // No reset fields at all: unchanged behaviour — the message is the only
        // source, and this prose carries no delay.
        let bare = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached."}}"#,
        )
        .context("expected a wrapped error")?;
        assert_eq!(
            websocket_error_kind(&bare),
            StreamErrorKind::RateLimited(None)
        );

        // A zero reset is not a delay either.
        let zero = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached.","resets_in_seconds":0}}"#,
        )
        .context("expected a wrapped error")?;
        assert_eq!(
            websocket_error_kind(&zero),
            StreamErrorKind::RateLimited(None)
        );
        Ok(())
    }

    #[test]
    fn structured_reset_wins_over_the_message_prose() -> anyhow::Result<()> {
        // Both present: the machine-readable field is authoritative.
        let parsed = parse_wrapped_websocket_error_event(
            r#"{"type":"error","status":429,"error":{"code":"rate_limit_exceeded","message":"Please try again in 5s.","resets_in_seconds":120}}"#,
        )
        .context("expected a wrapped error")?;

        assert_eq!(
            websocket_error_kind(&parsed),
            StreamErrorKind::RateLimited(Some(std::time::Duration::from_mins(2)))
        );
        Ok(())
    }

    #[test]
    fn test_prepare_websocket_request_allows_empty_delta_after_prewarm() {
        let request = ApiStreamingRequest {
            model: MODEL_GPT53_CODEX.to_string(),
            instructions: "system".to_string(),
            input: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Text("first".to_string()),
                phase: None,
            })],
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some(ApiToolChoice::Mode("auto")),
            parallel_tool_calls: None,
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
                format: None,
            }),
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            prompt_cache_key: Some("thread-1".to_string()),
            stream: true,
        };
        let session = WebsocketSessionState {
            connection: None,
            last_request: Some(request.clone()),
            last_response_id: Some("resp_prewarm".to_string()),
            last_response_items: Vec::new(),
            turn_state: None,
            prewarmed: true,
            websocket_disabled: false,
            in_flight: false,
            last_used: None,
        };

        let websocket_request = prepare_websocket_request(&request, &session, true);
        assert_eq!(
            websocket_request.previous_response_id.as_deref(),
            Some("resp_prewarm")
        );
        assert!(websocket_request.input.is_empty());
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
    fn test_build_api_input_uses_responses_text_types_by_role() {
        let request = ChatRequest {
            system: "system".to_string(),
            messages: vec![
                agent_sdk_foundation::llm::Message::user_with_content(vec![ContentBlock::Text {
                    text: "question".to_string(),
                }]),
                agent_sdk_foundation::llm::Message {
                    role: agent_sdk_foundation::llm::Role::Assistant,
                    content: Content::Blocks(vec![ContentBlock::Text {
                        text: "answer".to_string(),
                    }]),
                },
            ],
            tools: None,
            max_tokens: 512,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        };

        let input = build_api_input(&request);
        assert_eq!(input.len(), 2);

        match &input[0] {
            ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Parts(parts),
                ..
            }) => assert!(matches!(
                parts.as_slice(),
                [ApiInputContent::InputText { text }] if text == "question"
            )),
            _ => panic!("expected user message with input_text content"),
        }

        match &input[1] {
            ApiInputItem::Message(ApiMessage {
                role: ApiRole::Assistant,
                content: ApiMessageContent::Parts(parts),
                ..
            }) => assert!(matches!(
                parts.as_slice(),
                [ApiInputContent::OutputText { text }] if text == "answer"
            )),
            _ => panic!("expected assistant message with output_text content"),
        }
    }

    #[test]
    fn test_api_input_content_serialization_uses_current_responses_tags() {
        let json = serde_json::to_string(&ApiMessageContent::Parts(vec![
            ApiInputContent::InputText {
                text: "prompt".to_string(),
            },
            ApiInputContent::OutputText {
                text: "reply".to_string(),
            },
            ApiInputContent::Image {
                image_url: "data:image/png;base64,abc".to_string(),
            },
            ApiInputContent::File {
                filename: "notes.txt".to_string(),
                file_data: "data:text/plain;base64,abc".to_string(),
            },
        ]))
        .unwrap();

        assert!(json.contains("\"type\":\"input_text\""));
        assert!(json.contains("\"type\":\"output_text\""));
        assert!(json.contains("\"type\":\"input_image\""));
        assert!(json.contains("\"type\":\"input_file\""));
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
    fn assistant_message_phases_round_trip_without_duplicate_text() -> anyhow::Result<()> {
        let output: Vec<ApiOutputItem> = serde_json::from_value(serde_json::json!([
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
        ]))?;
        let blocks = build_content_blocks(&output);
        let request =
            ChatRequest::new(String::new(), vec![Message::assistant_with_content(blocks)]);
        let value = serde_json::to_value(build_api_input(&request))?;
        let items = value
            .as_array()
            .context("Codex input must serialize as an array")?;

        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["phase"], "commentary");
        assert_eq!(items[0]["content"][0]["text"], "Working.");
        assert_eq!(items[1]["phase"], "final_answer");
        assert_eq!(items[1]["content"][0]["text"], "Done.");
        assert_eq!(value.to_string().matches("Working.").count(), 1);
        assert_eq!(value.to_string().matches("Done.").count(), 1);

        let direct: ApiOutputItem = serde_json::from_value(serde_json::json!({
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Working."}]
        }))?;
        let direct = output_item_to_input_item(direct)
            .context("message output should become a continuation input")?;
        assert_eq!(serde_json::to_value(direct)?["phase"], "commentary");
        Ok(())
    }

    #[test]
    fn streamed_phase_and_reasoning_summary_preserve_block_order() -> anyhow::Result<()> {
        let message: ApiOutputItem = serde_json::from_value(serde_json::json!({
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Working."}]
        }))?;
        let mut accumulator = crate::streaming::StreamAccumulator::new();
        accumulator.apply(&StreamDelta::TextDelta {
            delta: "Working.".to_owned(),
            block_index: output_block_index(Some(0)),
        });
        for delta in output_item_stream_deltas(&message, 0, true) {
            accumulator.apply(&delta);
        }
        let message_blocks = accumulator.into_content_blocks();
        assert!(matches!(
            message_blocks.as_slice(),
            [ContentBlock::OpaqueReasoning { data, .. }, ContentBlock::Text { text }]
                if data["phase"] == "commentary" && text == "Working."
        ));

        let reasoning: ApiOutputItem = serde_json::from_value(serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ciphertext",
            "summary": [{"type": "summary_text", "text": "Checked."}]
        }))?;
        let mut accumulator = crate::streaming::StreamAccumulator::new();
        for delta in output_item_stream_deltas(&reasoning, 1, true) {
            accumulator.apply(&delta);
        }
        let reasoning_blocks = accumulator.into_content_blocks();
        assert!(matches!(
            reasoning_blocks.as_slice(),
            [ContentBlock::OpaqueReasoning { data, .. }, ContentBlock::Thinking { thinking, .. }]
                if data["encrypted_content"] == "ciphertext" && thinking == "Checked."
        ));
        Ok(())
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
    fn incomplete_and_refusal_responses_suppress_partial_tools() -> anyhow::Result<()> {
        let incomplete: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_incomplete",
            "model": "gpt-5.3-codex",
            "status": "incomplete",
            "incomplete_details": {"reason": "model_context_window_exceeded"},
            "output": [{
                "type": "function_call",
                "call_id": "call_partial",
                "name": "lookup",
                "arguments": "{"
            }]
        }))?;
        let incomplete = OpenAICodexResponsesProvider::map_response(incomplete);
        assert_eq!(
            incomplete.stop_reason,
            Some(StopReason::ModelContextWindowExceeded)
        );
        assert!(
            !incomplete
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        );

        let refusal: ApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_refusal",
            "model": "gpt-5.3-codex",
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
        let refusal = OpenAICodexResponsesProvider::map_response(refusal);
        assert_eq!(refusal.stop_reason, Some(StopReason::Refusal));
        assert!(
            !refusal
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        );
        assert!(matches!(
            refusal.content.last(),
            Some(ContentBlock::Text { text }) if text == "Cannot comply."
        ));
        Ok(())
    }

    #[test]
    fn reasoning_output_item_is_preserved_and_summary_is_visible() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "type": "reasoning",
            "id": "rs_123",
            "status": "completed",
            "encrypted_content": "ciphertext",
            "summary": [
                {"type": "summary_text", "text": "Checked the relevant constraints."}
            ]
        });
        let item: ApiOutputItem = serde_json::from_value(raw.clone())?;
        let replay_item = output_item_to_input_item(item);
        let Some(replay_item) = replay_item else {
            anyhow::bail!("reasoning item was not converted to a replay item");
        };
        assert_eq!(serde_json::to_value(replay_item)?, raw);

        let item: ApiOutputItem = serde_json::from_value(raw.clone())?;
        let blocks = build_content_blocks(&[item]);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(
            &blocks[0],
            ContentBlock::OpaqueReasoning { provider, data, .. }
                if provider == OPENAI_RESPONSES_REASONING_PROVIDER && data == &raw
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Thinking { thinking, signature, .. }
                if thinking == "Checked the relevant constraints." && signature.is_none()
        ));
        Ok(())
    }

    #[test]
    fn matching_opaque_reasoning_replays_as_a_top_level_item_in_source_order() -> anyhow::Result<()>
    {
        let raw = serde_json::json!({
            "type": "reasoning",
            "id": "rs_123",
            "encrypted_content": "ciphertext",
            "summary": []
        });
        let request = ChatRequest::new(
            "",
            vec![agent_sdk_foundation::llm::Message {
                role: agent_sdk_foundation::llm::Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Text {
                        text: "before".to_owned(),
                    },
                    ContentBlock::OpaqueReasoning {
                        provider: OPENAI_RESPONSES_REASONING_PROVIDER.to_owned(),
                        data: raw.clone(),
                    },
                    ContentBlock::OpaqueReasoning {
                        provider: "another-provider".to_owned(),
                        data: serde_json::json!({"type": "reasoning", "id": "ignored"}),
                    },
                    ContentBlock::Text {
                        text: "after".to_owned(),
                    },
                ]),
            }],
        );

        assert_eq!(
            serde_json::to_value(build_api_input(&request))?,
            serde_json::json!([
                {
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "before"}]
                },
                raw,
                {
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "after"}]
                }
            ])
        );
        Ok(())
    }

    #[test]
    fn usage_maps_cache_write_tokens() -> anyhow::Result<()> {
        let usage: ApiUsage = serde_json::from_value(serde_json::json!({
            "input_tokens": 2048,
            "output_tokens": 128,
            "input_tokens_details": {
                "cached_tokens": 1024,
                "cache_write_tokens": 512
            }
        }))?;

        let usage = usage_from_api_usage(&usage);
        assert_eq!(usage.cached_input_tokens, 1024);
        assert_eq!(usage.cache_creation_input_tokens, 512);
        Ok(())
    }

    #[test]
    fn stream_stop_reason_requires_a_semantic_terminal_event() {
        let tool_calls = HashMap::new();
        assert!(stop_reason_from_stream_state(&tool_calls, None, false, None).is_none());
        assert!(matches!(
            stop_reason_from_stream_state(&tool_calls, Some(ApiStatus::Completed), false, None,),
            Some(StopReason::EndTurn)
        ));
        assert!(matches!(
            stop_reason_from_stream_state(
                &tool_calls,
                Some(ApiStatus::Incomplete),
                false,
                Some("max_output_tokens"),
            ),
            Some(StopReason::MaxTokens)
        ));
    }

    #[test]
    fn test_request_serializes_response_format_text_and_forced_tool_choice() {
        let request = ApiStreamingRequest {
            model: MODEL_GPT53_CODEX.to_string(),
            instructions: String::new(),
            input: Vec::new(),
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some(codex_tool_choice(Some(&ToolChoice::Tool(
                "respond".to_owned(),
            )))),
            parallel_tool_calls: None,
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
                format: Some(ApiResponseTextFormat::from(&ResponseFormat::new(
                    "person",
                    serde_json::json!({"type": "object"}),
                ))),
            }),
            include: None,
            prompt_cache_key: None,
            stream: true,
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["text"]["format"]["type"], "json_schema");
        assert_eq!(json["text"]["format"]["name"], "person");
        assert_eq!(json["text"]["format"]["strict"], true);
        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["name"], "respond");
    }

    #[test]
    fn test_codex_tool_choice_defaults_to_auto() {
        assert_eq!(
            serde_json::to_value(codex_tool_choice(None)).unwrap(),
            serde_json::json!("auto")
        );
        assert_eq!(
            serde_json::to_value(codex_tool_choice(Some(&ToolChoice::Auto))).unwrap(),
            serde_json::json!("auto")
        );
    }

    #[test]
    fn test_convert_tool_makes_optional_params_nullable() {
        let tool = agent_sdk_foundation::llm::Tool {
            name: "t".to_string(),
            description: "d".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "req": {"type": "string"},
                    "opt": {"type": "string"}
                },
                "required": ["req"]
            }),
            display_name: "T".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        };

        let api_tool = convert_tool(tool);
        assert_eq!(api_tool.strict, Some(true));
        let schema = api_tool.parameters.unwrap();

        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"req"));
        assert!(required.contains(&"opt"));

        // The previously-optional `opt` must be wrapped in anyOf with a null
        // variant so the model is not forced to fabricate a value for it.
        let any_of = schema["properties"]["opt"]["anyOf"].as_array().unwrap();
        assert!(
            any_of
                .iter()
                .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"))
        );
    }

    #[test]
    fn test_convert_tool_disables_strict_for_freeform_object() {
        let tool = agent_sdk_foundation::llm::Tool {
            name: "t".to_string(),
            description: "d".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "T".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        };

        let api_tool = convert_tool(tool);
        assert_eq!(api_tool.strict, None);
    }

    #[test]
    fn test_emit_accumulated_tool_calls_assigns_distinct_ordered_indices() {
        let mut tool_calls = HashMap::new();
        tool_calls.insert(
            "b".to_string(),
            ToolCallAccumulator {
                id: "b".to_string(),
                name: "second".to_string(),
                arguments: "{}".to_string(),
                order: 1,
                block_index: None,
            },
        );
        tool_calls.insert(
            "a".to_string(),
            ToolCallAccumulator {
                id: "a".to_string(),
                name: "first".to_string(),
                arguments: "{}".to_string(),
                order: 0,
                block_index: None,
            },
        );

        let deltas = emit_accumulated_tool_calls(&tool_calls);
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
            vec![("first".to_string(), 2), ("second".to_string(), 4)]
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Transport selection: force-HTTP knob + cross-session WS-unhealthy memory
    // ────────────────────────────────────────────────────────────────────

    use crate::provider::LlmProvider;
    use agent_sdk_foundation::llm::{ChatRequest, Message};
    use std::sync::atomic::AtomicUsize;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A truthy OAuth-shaped token so `build_headers` can extract an account id
    /// without a real network call.
    fn oauth_token() -> String {
        test_token()
    }

    fn streaming_request(session_id: &str) -> ChatRequest {
        ChatRequest::new("You are helpful.", vec![Message::user("hello")])
            .with_max_tokens(1024)
            .with_session_id(session_id)
    }

    /// Read a single HTTP/1.1 request head (up to the blank line) from a stream.
    async fn read_http_head(stream: &mut tokio::net::TcpStream) -> String {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while stream.read_exact(&mut byte).await.is_ok() {
            buf.push(byte[0]);
            if buf.ends_with(b"\r\n\r\n") {
                break;
            }
            if buf.len() > 16 * 1024 {
                break;
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Minimal SSE body that drives the HTTP fallback to a clean `Done`.
    const HTTP_SSE_BODY: &str = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "data: [DONE]\n\n",
    );

    /// Spawn an HTTP-only server that records whether any WebSocket upgrade was
    /// attempted and serves a fixed SSE body to plain POSTs. Returns the base
    /// URL plus the (`ws_attempts`, `http_requests`) counters.
    async fn spawn_http_only_server_with_body(
        sse_body: &'static str,
    ) -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let ws_attempts = Arc::new(AtomicUsize::new(0));
        let http_requests = Arc::new(AtomicUsize::new(0));
        let ws_attempts_task = ws_attempts.clone();
        let http_requests_task = http_requests.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let ws_attempts = ws_attempts_task.clone();
                let http_requests = http_requests_task.clone();
                tokio::spawn(async move {
                    let head = read_http_head(&mut stream).await;
                    if head.to_ascii_lowercase().contains("upgrade: websocket") {
                        ws_attempts.fetch_add(1, Ordering::Relaxed);
                        // Refuse the upgrade; a black-holed proxy would simply
                        // never complete it, but refusing fast keeps the test
                        // deterministic.
                        let _ = stream
                            .write_all(
                                b"HTTP/1.1 426 Upgrade Required\r\ncontent-length: 0\r\n\r\n",
                            )
                            .await;
                        return;
                    }
                    http_requests.fetch_add(1, Ordering::Relaxed);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                        sse_body.len(),
                        sse_body,
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });

        (
            format!("http://{addr}/backend-api"),
            ws_attempts,
            http_requests,
        )
    }

    async fn spawn_http_only_server() -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        spawn_http_only_server_with_body(HTTP_SSE_BODY).await
    }

    /// Drain a stream, returning whether it completed without a transport error.
    async fn drain_ok(provider: &OpenAICodexResponsesProvider, request: ChatRequest) -> bool {
        let mut stream = std::pin::pin!(provider.chat_stream(request));
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamDelta::Error { .. }) | Err(_) => saw_error = true,
                Ok(_) => {}
            }
        }
        !saw_error
    }

    #[tokio::test]
    async fn websockets_disabled_builder_goes_straight_to_http() {
        let (base_url, ws_attempts, http_requests) = spawn_http_only_server().await;
        let provider = OpenAICodexResponsesProvider::with_base_url(
            oauth_token(),
            MODEL_GPT53_CODEX.to_string(),
            base_url,
        )
        .with_websockets_disabled(true);

        assert!(drain_ok(&provider, streaming_request("session-a")).await);

        assert_eq!(
            ws_attempts.load(Ordering::Relaxed),
            0,
            "no websocket upgrade may be attempted when websockets are disabled",
        );
        assert_eq!(http_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn premature_http_stream_termination_is_an_error() {
        for (case, sse_body) in [
            ("done", "data: [DONE]\n\n"),
            (
                "eof",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
            ),
        ] {
            let (base_url, _, _) = spawn_http_only_server_with_body(sse_body).await;
            let provider = OpenAICodexResponsesProvider::with_base_url(
                oauth_token(),
                MODEL_GPT53_CODEX.to_string(),
                base_url,
            )
            .with_websockets_disabled(true);
            let mut stream = std::pin::pin!(
                provider.chat_stream(streaming_request(&format!("premature-{case}")))
            );
            let mut saw_error = false;
            let mut saw_done = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamDelta::Error { .. }) | Err(_) => saw_error = true,
                    Ok(StreamDelta::Done { .. }) => saw_done = true,
                    Ok(_) => {}
                }
            }
            assert!(saw_error, "case={case} must surface a stream error");
            assert!(!saw_done, "case={case} must not synthesize success");
        }
    }

    #[tokio::test]
    async fn malformed_http_events_and_items_fail_closed() -> anyhow::Result<()> {
        for (case, sse_body) in [
            ("event", "data: {not-json}\n\n"),
            (
                "item",
                "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\"}}\n\n",
            ),
        ] {
            let (base_url, _, _) = spawn_http_only_server_with_body(sse_body).await;
            let provider = OpenAICodexResponsesProvider::with_base_url(
                oauth_token(),
                MODEL_GPT53_CODEX.to_owned(),
                base_url,
            )
            .with_websockets_disabled(true);
            let mut stream = std::pin::pin!(
                provider.chat_stream(streaming_request(&format!("malformed-{case}")))
            );
            let first = stream
                .next()
                .await
                .context("malformed stream must emit an error")??;
            assert!(matches!(
                first,
                StreamDelta::Error {
                    kind: StreamErrorKind::ServerError,
                    ..
                }
            ));
            assert!(stream.next().await.is_none());
        }
        Ok(())
    }

    #[tokio::test]
    async fn atomic_http_function_call_is_not_dropped() -> anyhow::Result<()> {
        let sse_body = concat!(
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"ciphertext\",\"summary\":[]}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"src/lib.rs\\\"}\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
            "data: [DONE]\n\n",
        );
        let (base_url, _, _) = spawn_http_only_server_with_body(sse_body).await;
        let provider = OpenAICodexResponsesProvider::with_base_url(
            oauth_token(),
            MODEL_GPT53_CODEX.to_owned(),
            base_url,
        )
        .with_websockets_disabled(true);
        let mut stream =
            std::pin::pin!(provider.chat_stream(streaming_request("atomic-function-call")));
        let mut deltas = Vec::new();
        while let Some(delta) = stream.next().await {
            deltas.push(delta?);
        }

        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolUseStart { id, name, .. }
                if id == "call_1" && name == "read"
        )));
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            StreamDelta::ToolInputDelta { id, delta, .. }
                if id == "call_1" && delta == r#"{"path":"src/lib.rs"}"#
        )));
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::ToolUse)
            })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn non_tool_http_terminals_suppress_partial_tool_calls() -> anyhow::Result<()> {
        for (case, sse_body, expected_stop) in [
            (
                "incomplete",
                concat!(
                    "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"call_id\":\"call_1\",\"name\":\"lookup\",\"delta\":\"{\"}\n\n",
                    "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
                    "data: [DONE]\n\n",
                ),
                StopReason::MaxTokens,
            ),
            (
                "refusal",
                concat!(
                    "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"call_id\":\"call_1\",\"name\":\"lookup\",\"delta\":\"{}\"}\n\n",
                    "data: {\"type\":\"response.refusal.delta\",\"output_index\":1,\"delta\":\"Cannot comply.\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                    "data: [DONE]\n\n",
                ),
                StopReason::Refusal,
            ),
        ] {
            let (base_url, _, _) = spawn_http_only_server_with_body(sse_body).await;
            let provider = OpenAICodexResponsesProvider::with_base_url(
                oauth_token(),
                MODEL_GPT53_CODEX.to_owned(),
                base_url,
            )
            .with_websockets_disabled(true);
            let mut stream = std::pin::pin!(
                provider.chat_stream(streaming_request(&format!("terminal-{case}")))
            );
            let mut deltas = Vec::new();
            while let Some(delta) = stream.next().await {
                deltas.push(delta?);
            }
            assert!(matches!(
                deltas.last(),
                Some(StreamDelta::Done {
                    stop_reason: Some(stop_reason)
                }) if *stop_reason == expected_stop
            ));
            assert!(!deltas.iter().any(|delta| matches!(
                delta,
                StreamDelta::ToolUseStart { .. } | StreamDelta::ToolInputDelta { .. }
            )));
        }
        Ok(())
    }

    #[test]
    fn parse_disable_websockets_value_recognizes_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(
                parse_disable_websockets_value(Some(value)),
                "value={value:?} should disable websockets",
            );
        }
        for value in ["0", "false", "no", "off", "", "maybe"] {
            assert!(
                !parse_disable_websockets_value(Some(value)),
                "value={value:?} should NOT disable websockets",
            );
        }
        assert!(!parse_disable_websockets_value(None));
    }

    #[tokio::test]
    async fn websockets_disabled_via_env_value_goes_straight_to_http() {
        // The env var feeds `with_websockets_disabled` through
        // `parse_disable_websockets_value` at construction. `std::env::set_var`
        // is `unsafe` and rejected by `#![forbid(unsafe_code)]`, so we drive the
        // exact value the env reader would produce for `"1"` end-to-end here.
        let disabled = parse_disable_websockets_value(Some("1"));
        assert!(disabled);

        let (base_url, ws_attempts, http_requests) = spawn_http_only_server().await;
        let provider = OpenAICodexResponsesProvider::with_base_url(
            oauth_token(),
            MODEL_GPT53_CODEX.to_string(),
            base_url,
        )
        .with_websockets_disabled(disabled);

        assert!(drain_ok(&provider, streaming_request("session-env")).await);

        assert_eq!(
            ws_attempts.load(Ordering::Relaxed),
            0,
            "no websocket upgrade may be attempted when the env var forces HTTP-only",
        );
        assert_eq!(http_requests.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn provider_marked_ws_unhealthy_skips_websocket_on_new_session() {
        let (base_url, ws_attempts, http_requests) = spawn_http_only_server().await;
        let provider = OpenAICodexResponsesProvider::with_base_url(
            oauth_token(),
            MODEL_GPT53_CODEX.to_string(),
            base_url,
        );

        // Simulate a prior session having latched the provider-level transport
        // signal after a connectivity failure.
        provider.websockets_unhealthy.store(true, Ordering::Relaxed);

        // A brand-new session must skip the websocket attempt entirely.
        assert!(drain_ok(&provider, streaming_request("fresh-session")).await);

        assert_eq!(
            ws_attempts.load(Ordering::Relaxed),
            0,
            "a websocket-unhealthy provider must not attempt a new upgrade",
        );
        assert_eq!(http_requests.load(Ordering::Relaxed), 1);
    }

    /// Spawn a server that completes the WebSocket handshake then emits a
    /// wrapped 401 error event. Plain POSTs get the HTTP SSE body. Returns the
    /// base URL plus the http-request counter (for the fallback assertion).
    async fn spawn_ws_unauthorized_server() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let http_requests = Arc::new(AtomicUsize::new(0));
        let http_requests_task = http_requests.clone();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let http_requests = http_requests_task.clone();
                tokio::spawn(async move {
                    let mut stream = stream;
                    // Non-destructively peek the head to route WS vs HTTP, then
                    // hand the still-unread stream to the right handler.
                    let mut peek = [0u8; 1024];
                    let Ok(n) = stream.peek(&mut peek).await else {
                        return;
                    };
                    let head = String::from_utf8_lossy(&peek[..n]).to_ascii_lowercase();

                    if head.contains("upgrade: websocket") {
                        // `accept_async` reads and completes the handshake from
                        // the untouched stream, so no manual SHA-1 is needed.
                        let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
                            return;
                        };
                        let payload =
                            r#"{"type":"error","status":401,"error":{"message":"unauthorized"}}"#;
                        let _ = ws
                            .send(WebSocketMessage::Text(payload.to_string().into()))
                            .await;
                        let _ = ws.send(WebSocketMessage::Close(None)).await;
                        return;
                    }

                    // Plain HTTP POST: drain the request head, then reply.
                    let _ = read_http_head(&mut stream).await;
                    http_requests.fetch_add(1, Ordering::Relaxed);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                        HTTP_SSE_BODY.len(),
                        HTTP_SSE_BODY,
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });

        (format!("http://{addr}/backend-api"), http_requests)
    }

    #[tokio::test]
    async fn ws_unauthorized_disables_session_but_not_provider() {
        let (base_url, http_requests) = spawn_ws_unauthorized_server().await;
        let provider = OpenAICodexResponsesProvider::with_base_url(
            oauth_token(),
            MODEL_GPT53_CODEX.to_string(),
            base_url,
        );

        // The websocket warmup hits a wrapped 401, disables the session's
        // websocket, and falls back to HTTP — which completes cleanly.
        assert!(drain_ok(&provider, streaming_request("auth-session")).await);

        // The auth failure is a request problem, NOT a transport problem: the
        // provider-level flag must stay clear so a transient blip does not
        // force HTTP-only forever.
        assert!(
            !provider.websockets_unhealthy.load(Ordering::Relaxed),
            "a 401 must not mark the provider websocket-transport-unhealthy",
        );
        // The per-session flag was set, so this same session now goes straight
        // to HTTP without another websocket attempt.
        let session = provider.websocket_session("auth-session").await;
        assert!(session.lock().await.websocket_disabled);

        assert!(http_requests.load(Ordering::Relaxed) >= 1);
    }
}
