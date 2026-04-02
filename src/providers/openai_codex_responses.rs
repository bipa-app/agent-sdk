//! `OpenAI` Codex / `ChatGPT` subscription provider implementation.
//!
//! This mirrors pi's `openai-codex-responses` provider family and talks to the
//! `ChatGPT` Codex backend using OAuth bearer tokens captured from the `ChatGPT`
//! Plus/Pro login flow.

use crate::llm::attachments::validate_request_attachments;
use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Effort, LlmProvider, StopReason,
    StreamBox, StreamDelta, ThinkingConfig, ThinkingMode, Usage,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine;
use futures::{SinkExt, StreamExt};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WebSocketMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const OPENAI_CODEX_JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const OPENAI_CODEX_ORIGINATOR: &str = "codex_cli_rs";
const OPENAI_CODEX_RESPONSES_BETA_HEADER: &str = "responses=experimental";
const OPENAI_CODEX_RESPONSES_WEBSOCKETS_BETA_HEADER: &str = "responses_websockets=2026-02-06";
const OPENAI_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE: &str =
    "websocket_connection_limit_reached";

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
}

impl OpenAICodexResponsesProvider {
    /// Create a new `OpenAI` Codex provider.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
            thinking: None,
            account_id: None,
            websocket_sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a provider with a custom base URL.
    #[must_use]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
            thinking: None,
            account_id: None,
            websocket_sessions: Arc::new(Mutex::new(HashMap::new())),
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

    const fn max_output_tokens(request: &ChatRequest) -> Option<u32> {
        if request.max_tokens_explicit {
            Some(request.max_tokens)
        } else {
            None
        }
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

        let (stream, response) = connect_async(request)
            .await
            .context("failed to connect OpenAI Codex websocket")?;
        let turn_state = response
            .headers()
            .get(OPENAI_CODEX_TURN_STATE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        Ok((stream, turn_state))
    }

    fn map_response(api_response: ApiResponse) -> ChatResponse {
        let content = build_content_blocks(&api_response.output);
        let has_tool_calls = content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
        let stop_reason = if has_tool_calls {
            Some(StopReason::ToolUse)
        } else {
            api_response.status.map(|status| match status {
                ApiStatus::Completed => StopReason::EndTurn,
                ApiStatus::Incomplete => StopReason::MaxTokens,
                ApiStatus::Failed => StopReason::StopSequence,
            })
        };

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
                },
                |usage| Usage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cached_input_tokens: usage
                        .input_tokens_details
                        .as_ref()
                        .map_or(0, |details| details.cached_tokens),
                },
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

        let api_request = ApiResponsesRequest {
            model: &self.model,
            instructions: request.system.as_str(),
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens,
            reasoning,
            tool_choice: Some("auto"),
            parallel_tool_calls: parallel_tool_calls.then_some(true),
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
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
            return Ok(ChatOutcome::RateLimited);
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
                        recoverable: false,
                    });
                    return;
                }
            };
            if let Err(error) = validate_request_attachments(self.provider(), self.model(), &request) {
                yield Ok(StreamDelta::Error {
                    message: error.to_string(),
                    recoverable: false,
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
            let api_request = ApiStreamingRequest {
                model: self.model.clone(),
                instructions: request.system.clone(),
                input,
                tools,
                max_output_tokens,
                reasoning,
                tool_choice: Some("auto".to_string()),
                parallel_tool_calls: parallel_tool_calls.then_some(true),
                store: false,
                text: Some(ApiTextSettings { verbosity: "medium" }),
                include: Some(vec!["reasoning.encrypted_content".to_string()]),
                prompt_cache_key: request.session_id.clone(),
                stream: true,
            };

            log::debug!("OpenAI Codex streaming request model={} max_tokens={}", self.model, request.max_tokens);

            let mut sse_turn_state: Option<String> = None;

            if let Some(session_id) = request.session_id.as_deref() {
                let session = self.websocket_session(session_id).await;
                let mut websocket_session = session.lock().await;

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
                                        recoverable: false,
                                    });
                                    return;
                                }
                            };

                            let warmup_send_result = if let Some(connection) =
                                websocket_session.connection.as_mut()
                            {
                                connection.send(WebSocketMessage::Text(warmup_payload.into())).await
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
                                }
                                continue;
                            }

                            let mut warmup_response_id: Option<String> = None;
                            let mut warmup_response_items = Vec::new();

                            loop {
                                let message_result = if let Some(connection) =
                                    websocket_session.connection.as_mut()
                                {
                                    connection.next().await
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
                                        }
                                        continue 'websocket_attempts;
                                    }
                                };

                                match message {
                                    WebSocketMessage::Text(text) => {
                                        if let Some((status, message)) =
                                            parse_wrapped_websocket_error_event(&text)
                                        {
                                            log::warn!(
                                                "OpenAI Codex websocket warmup wrapped error on attempt {} status={} message={message}",
                                                attempt + 1,
                                                status,
                                            );
                                            if status == StatusCode::UNAUTHORIZED
                                                || status == StatusCode::UPGRADE_REQUIRED
                                                || status.is_client_error()
                                            {
                                                websocket_session.websocket_disabled = true;
                                            }
                                            reset_websocket_connection(&mut websocket_session);
                                            continue 'websocket_attempts;
                                        }
                                        if let Ok(event) =
                                            serde_json::from_str::<ApiStreamEvent>(&text)
                                        {
                                            match event.r#type.as_str() {
                                                "response.output_item.added" => {
                                                    if let Some(item) = event.item
                                                        && let Ok(item) =
                                                            serde_json::from_value::<ApiOutputItem>(item)
                                                        && let Some(item) =
                                                            output_item_to_input_item(item)
                                                    {
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
                                                    }
                                                    continue 'websocket_attempts;
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    WebSocketMessage::Binary(bytes) => {
                                        if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                            if let Some((status, message)) =
                                                parse_wrapped_websocket_error_event(&text)
                                            {
                                                log::warn!(
                                                    "OpenAI Codex websocket warmup wrapped error on attempt {} status={} message={message}",
                                                    attempt + 1,
                                                    status,
                                                );
                                                if status == StatusCode::UNAUTHORIZED
                                                    || status == StatusCode::UPGRADE_REQUIRED
                                                    || status.is_client_error()
                                                {
                                                    websocket_session.websocket_disabled = true;
                                                }
                                                reset_websocket_connection(&mut websocket_session);
                                                continue 'websocket_attempts;
                                            }

                                            if let Ok(event) =
                                                serde_json::from_str::<ApiStreamEvent>(&text)
                                            {
                                                match event.r#type.as_str() {
                                                    "response.output_item.added" => {
                                                        if let Some(item) = event.item
                                                            && let Ok(item) = serde_json::from_value::<
                                                                ApiOutputItem,
                                                            >(item)
                                                            && let Some(item) =
                                                                output_item_to_input_item(item)
                                                        {
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
                                                        }
                                                        continue 'websocket_attempts;
                                                    }
                                                    _ => {}
                                                }
                                            }
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
                                    recoverable: false,
                                });
                                return;
                            }
                        };

                        let send_result = if let Some(connection) = websocket_session.connection.as_mut() {
                            connection.send(WebSocketMessage::Text(request_payload.into())).await
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
                            }
                            continue;
                        }

                        let mut usage: Option<Usage> = None;
                        let mut tool_calls: HashMap<String, ToolCallAccumulator> = HashMap::new();
                        let mut response_id: Option<String> = None;
                        let mut response_items = Vec::new();
                        let mut emitted_output = false;

                        loop {
                            let message_result = if let Some(connection) =
                                websocket_session.connection.as_mut()
                            {
                                connection.next().await
                            } else {
                                None
                            };
                            let Some(message_result) = message_result else {
                                if emitted_output {
                                    reset_websocket_connection(&mut websocket_session);
                                    yield Ok(StreamDelta::Error {
                                        message: "websocket closed before response.completed"
                                            .to_string(),
                                        recoverable: true,
                                    });
                                    return;
                                }
                                reset_websocket_connection(&mut websocket_session);
                                if attempt == 1 {
                                    websocket_session.websocket_disabled = true;
                                }
                                continue 'websocket_attempts;
                            };

                            let message = match message_result {
                                Ok(message) => message,
                                Err(error) => {
                                    if emitted_output {
                                        reset_websocket_connection(&mut websocket_session);
                                        yield Ok(StreamDelta::Error {
                                            message: format!("websocket error: {error}"),
                                            recoverable: true,
                                        });
                                        return;
                                    }
                                    reset_websocket_connection(&mut websocket_session);
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                    }
                                    continue 'websocket_attempts;
                                }
                            };

                            match message {
                                WebSocketMessage::Text(text) => {
                                    if let Some((status, message)) =
                                        parse_wrapped_websocket_error_event(&text)
                                    {
                                        let recoverable =
                                            status == StatusCode::TOO_MANY_REQUESTS
                                                || status.is_server_error();
                                        if emitted_output {
                                            reset_websocket_connection(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message,
                                                recoverable,
                                            });
                                            return;
                                        }
                                        if status == StatusCode::UNAUTHORIZED
                                            || status == StatusCode::UPGRADE_REQUIRED
                                            || status.is_client_error()
                                        {
                                            websocket_session.websocket_disabled = true;
                                        }
                                        reset_websocket_connection(&mut websocket_session);
                                        continue 'websocket_attempts;
                                    }
                                    if let Ok(event) = serde_json::from_str::<ApiStreamEvent>(&text) {
                                        match event.r#type.as_str() {
                                            "response.output_text.delta" => {
                                                if let Some(delta) = event.delta {
                                                    emitted_output = true;
                                                    yield Ok(StreamDelta::TextDelta {
                                                        delta,
                                                        block_index: 0,
                                                    });
                                                }
                                            }
                                            "response.function_call_arguments.delta" => {
                                                if let (Some(call_id), Some(delta)) =
                                                    (event.call_id, event.delta)
                                                {
                                                    let acc = tool_calls
                                                        .entry(call_id.clone())
                                                        .or_insert_with(|| ToolCallAccumulator {
                                                            id: call_id,
                                                            name: event.name.unwrap_or_default(),
                                                            arguments: String::new(),
                                                        });
                                                    acc.arguments.push_str(&delta);
                                                }
                                            }
                                            "response.output_item.added" => {
                                                if let Some(item) = event.item
                                                    && let Ok(item) =
                                                        serde_json::from_value::<ApiOutputItem>(item)
                                                    && let Some(item) = output_item_to_input_item(item)
                                                {
                                                    response_items.push(item);
                                                }
                                            }
                                            "response.completed"
                                            | "response.incomplete"
                                            | "response.done" => {
                                                if let Some(resp) = event.response {
                                                    if let Some(u) = resp.usage {
                                                        usage = Some(usage_from_api_usage(&u));
                                                    }
                                                    if let Some(id) = resp.id {
                                                        response_id = Some(id);
                                                    }
                                                }
                                                let final_status = Some(match event.r#type.as_str() {
                                                    "response.incomplete" => ApiStatus::Incomplete,
                                                    _ => ApiStatus::Completed,
                                                });
                                                for delta in emit_accumulated_tool_calls(&tool_calls) {
                                                    yield Ok(delta);
                                                }
                                                if let Some(u) = usage.take() {
                                                    yield Ok(StreamDelta::Usage(u));
                                                }
                                                websocket_session.last_request = Some(api_request.clone());
                                                websocket_session.last_response_id = response_id;
                                                websocket_session.last_response_items = response_items;
                                                websocket_session.prewarmed = false;
                                                yield Ok(StreamDelta::Done {
                                                    stop_reason: Some(stop_reason_from_stream_state(
                                                        &tool_calls,
                                                        final_status,
                                                    )),
                                                });
                                                return;
                                            }
                                            "response.failed" => {
                                                websocket_session.last_request = None;
                                                websocket_session.last_response_id = None;
                                                websocket_session.last_response_items.clear();
                                                websocket_session.prewarmed = false;
                                                let message = event
                                                    .response
                                                    .and_then(|resp| resp.error)
                                                    .and_then(|error| error.message)
                                                    .unwrap_or_else(|| {
                                                        "Codex response failed".to_string()
                                                    });
                                                yield Ok(StreamDelta::Error {
                                                    message,
                                                    recoverable: false,
                                                });
                                                return;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                WebSocketMessage::Binary(bytes) => {
                                    if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                        if let Some((status, message)) =
                                            parse_wrapped_websocket_error_event(&text)
                                        {
                                            let recoverable =
                                                status == StatusCode::TOO_MANY_REQUESTS
                                                    || status.is_server_error();
                                            if emitted_output {
                                                reset_websocket_connection(&mut websocket_session);
                                                yield Ok(StreamDelta::Error {
                                                    message,
                                                    recoverable,
                                                });
                                                return;
                                            }
                                            if status == StatusCode::UNAUTHORIZED
                                                || status == StatusCode::UPGRADE_REQUIRED
                                                || status.is_client_error()
                                            {
                                                websocket_session.websocket_disabled = true;
                                            }
                                            reset_websocket_connection(&mut websocket_session);
                                            continue 'websocket_attempts;
                                        }

                                        if let Ok(event) =
                                            serde_json::from_str::<ApiStreamEvent>(&text)
                                        {
                                            match event.r#type.as_str() {
                                                "response.output_text.delta" => {
                                                    if let Some(delta) = event.delta {
                                                        emitted_output = true;
                                                        yield Ok(StreamDelta::TextDelta {
                                                            delta,
                                                            block_index: 0,
                                                        });
                                                    }
                                                }
                                                "response.function_call_arguments.delta" => {
                                                    if let (Some(call_id), Some(delta)) =
                                                        (event.call_id, event.delta)
                                                    {
                                                        let acc = tool_calls
                                                            .entry(call_id.clone())
                                                            .or_insert_with(|| ToolCallAccumulator {
                                                                id: call_id,
                                                                name: event.name.unwrap_or_default(),
                                                                arguments: String::new(),
                                                            });
                                                        acc.arguments.push_str(&delta);
                                                    }
                                                }
                                                "response.output_item.added" => {
                                                    if let Some(item) = event.item
                                                        && let Ok(item) = serde_json::from_value::<
                                                            ApiOutputItem,
                                                        >(item)
                                                        && let Some(item) =
                                                            output_item_to_input_item(item)
                                                    {
                                                        response_items.push(item);
                                                    }
                                                }
                                                "response.completed"
                                                | "response.incomplete"
                                                | "response.done" => {
                                                    if let Some(resp) = event.response {
                                                        if let Some(u) = resp.usage {
                                                            usage = Some(usage_from_api_usage(&u));
                                                        }
                                                        if let Some(id) = resp.id {
                                                            response_id = Some(id);
                                                        }
                                                    }
                                                    let final_status = Some(
                                                        match event.r#type.as_str() {
                                                            "response.incomplete" => ApiStatus::Incomplete,
                                                            _ => ApiStatus::Completed,
                                                        },
                                                    );
                                                    for delta in
                                                        emit_accumulated_tool_calls(&tool_calls)
                                                    {
                                                        yield Ok(delta);
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
                                                    yield Ok(StreamDelta::Done {
                                                        stop_reason: Some(
                                                            stop_reason_from_stream_state(
                                                                &tool_calls,
                                                                final_status,
                                                            ),
                                                        ),
                                                    });
                                                    return;
                                                }
                                                "response.failed" => {
                                                    websocket_session.last_request = None;
                                                    websocket_session.last_response_id = None;
                                                    websocket_session.last_response_items.clear();
                                                    websocket_session.prewarmed = false;
                                                    let message = event
                                                        .response
                                                        .and_then(|resp| resp.error)
                                                        .and_then(|error| error.message)
                                                        .unwrap_or_else(|| {
                                                            "Codex response failed".to_string()
                                                        });
                                                    yield Ok(StreamDelta::Error {
                                                        message,
                                                        recoverable: false,
                                                    });
                                                    return;
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                WebSocketMessage::Ping(payload) => {
                                    if let Some(connection) = websocket_session.connection.as_mut()
                                        && let Err(error) = connection
                                            .send(WebSocketMessage::Pong(payload))
                                            .await
                                    {
                                        if emitted_output {
                                            reset_websocket_connection(&mut websocket_session);
                                            yield Ok(StreamDelta::Error {
                                                message: format!("websocket pong failed: {error}"),
                                                recoverable: true,
                                            });
                                            return;
                                        }
                                        reset_websocket_connection(&mut websocket_session);
                                        if attempt == 1 {
                                            websocket_session.websocket_disabled = true;
                                        }
                                        continue 'websocket_attempts;
                                    }
                                }
                                WebSocketMessage::Pong(_) | WebSocketMessage::Frame(_) => {}
                                WebSocketMessage::Close(_) => {
                                    if emitted_output {
                                        reset_websocket_connection(&mut websocket_session);
                                        yield Ok(StreamDelta::Error {
                                            message: "websocket closed before response.completed"
                                                .to_string(),
                                            recoverable: true,
                                        });
                                        return;
                                    }
                                    reset_websocket_connection(&mut websocket_session);
                                    if attempt == 1 {
                                        websocket_session.websocket_disabled = true;
                                    }
                                    continue 'websocket_attempts;
                                }
                            }
                        }
                    }
                }
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
                        recoverable: false,
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
                let body = response.text().await.unwrap_or_default();
                let recoverable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                log::warn!("OpenAI Codex error status={status} body={body}");
                yield Ok(StreamDelta::Error { message: body, recoverable });
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

            let mut buffer = String::new();
            let mut stream = response.bytes_stream();
            let mut usage: Option<Usage> = None;
            let mut tool_calls: HashMap<String, ToolCallAccumulator> = HashMap::new();
            let mut final_status: Option<ApiStatus> = None;

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error"));
                    return;
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() {
                        continue;
                    }

                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    if data == "[DONE]" {
                        for delta in emit_accumulated_tool_calls(&tool_calls) {
                            yield Ok(delta);
                        }
                        if let Some(u) = usage.take() {
                            yield Ok(StreamDelta::Usage(u));
                        }
                        yield Ok(StreamDelta::Done {
                            stop_reason: Some(stop_reason_from_stream_state(&tool_calls, final_status)),
                        });
                        return;
                    }

                    if let Ok(event) = serde_json::from_str::<ApiStreamEvent>(data) {
                        match event.r#type.as_str() {
                            "response.output_text.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::TextDelta { delta, block_index: 0 });
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                if let (Some(call_id), Some(delta)) = (event.call_id, event.delta) {
                                    let acc = tool_calls.entry(call_id.clone()).or_insert_with(|| {
                                        ToolCallAccumulator {
                                            id: call_id,
                                            name: event.name.unwrap_or_default(),
                                            arguments: String::new(),
                                        }
                                    });
                                    acc.arguments.push_str(&delta);
                                }
                            }
                            "response.completed" | "response.incomplete" | "response.done" => {
                                if let Some(resp) = event.response
                                    && let Some(u) = resp.usage
                                {
                                    usage = Some(usage_from_api_usage(&u));
                                }
                                final_status = Some(match event.r#type.as_str() {
                                    "response.incomplete" => ApiStatus::Incomplete,
                                    _ => ApiStatus::Completed,
                                });
                            }
                            "response.failed" => {
                                let message = event
                                    .response
                                    .and_then(|resp| resp.error)
                                    .and_then(|error| error.message)
                                    .unwrap_or_else(|| "Codex response failed".to_string());
                                yield Ok(StreamDelta::Error {
                                    message,
                                    recoverable: false,
                                });
                                return;
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some(u) = usage {
                yield Ok(StreamDelta::Usage(u));
            }
            yield Ok(StreamDelta::Done {
                stop_reason: Some(stop_reason_from_stream_state(&tool_calls, final_status)),
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
        match &msg.content {
            Content::Text(text) => {
                items.push(ApiInputItem::Message(ApiMessage {
                    role: match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    },
                    content: ApiMessageContent::Text(text.clone()),
                }));
            }
            Content::Blocks(blocks) => {
                let mut content_parts = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            content_parts.push(ApiInputContent::Text { text: text.clone() });
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
                        ContentBlock::Image { source } => {
                            content_parts.push(ApiInputContent::Image {
                                image_url: format!(
                                    "data:{};base64,{}",
                                    source.media_type, source.data
                                ),
                            });
                        }
                        ContentBlock::Document { source } => {
                            content_parts.push(ApiInputContent::File {
                                filename: suggested_filename(&source.media_type),
                                file_data: format!(
                                    "data:{};base64,{}",
                                    source.media_type, source.data
                                ),
                            });
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
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
                            items.push(ApiInputItem::FunctionCallOutput(
                                ApiFunctionCallOutput::new(tool_use_id.clone(), content.clone()),
                            ));
                        }
                    }
                }

                if !content_parts.is_empty() {
                    items.push(ApiInputItem::Message(ApiMessage {
                        role: match msg.role {
                            crate::llm::Role::User => ApiRole::User,
                            crate::llm::Role::Assistant => ApiRole::Assistant,
                        },
                        content: ApiMessageContent::Parts(content_parts),
                    }));
                }
            }
        }
    }

    items
}

/// Recursively fix a JSON schema for `OpenAI` strict mode.
/// Adds `additionalProperties: false` and ensures all properties are required.
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

fn convert_tool(tool: crate::llm::Tool) -> ApiTool {
    // The Responses API with strict: true requires:
    // 1. additionalProperties: false on all object schemas
    // 2. All properties must be in the required array
    // These requirements apply recursively to nested schemas
    let mut schema = tool.input_schema;
    fix_schema_for_strict_mode(&mut schema);

    ApiTool {
        r#type: "function".to_owned(),
        name: tool.name,
        description: Some(tool.description),
        parameters: Some(schema),
        strict: Some(true),
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

fn build_content_blocks(output: &[ApiOutputItem]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for item in output {
        match item {
            ApiOutputItem::Message { content, .. } => {
                for c in content {
                    if let ApiOutputContent::Text { text } = c
                        && !text.is_empty()
                    {
                        blocks.push(ContentBlock::Text { text: text.clone() });
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
}

fn usage_from_api_usage(usage: &ApiUsage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .map_or(0, |details| details.cached_tokens),
    }
}

fn emit_accumulated_tool_calls(
    tool_calls: &HashMap<String, ToolCallAccumulator>,
) -> Vec<StreamDelta> {
    let block_index = usize::from(!tool_calls.is_empty());
    let mut deltas = Vec::new();
    for acc in tool_calls.values() {
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
) -> StopReason {
    if tool_calls.is_empty() {
        match status.unwrap_or(ApiStatus::Completed) {
            ApiStatus::Completed => StopReason::EndTurn,
            ApiStatus::Incomplete => StopReason::MaxTokens,
            ApiStatus::Failed => StopReason::StopSequence,
        }
    } else {
        StopReason::ToolUse
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

fn parse_wrapped_websocket_error_event(payload: &str) -> Option<(StatusCode, String)> {
    let event: ApiWrappedWebsocketErrorEvent = serde_json::from_str(payload).ok()?;
    if event.kind != "error" {
        return None;
    }

    if event.error.as_ref().and_then(|error| error.code.as_deref())
        == Some(OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE)
    {
        let message = event
            .error
            .and_then(|error| error.message)
            .unwrap_or_else(|| "Responses websocket connection limit reached".to_string());
        return Some((StatusCode::TOO_MANY_REQUESTS, message));
    }

    let status = StatusCode::from_u16(event.status?).ok()?;
    let message = event
        .error
        .and_then(|error| error.message)
        .unwrap_or_else(|| payload.to_string());
    if status.is_success() {
        None
    } else {
        Some((status, message))
    }
}

fn output_item_to_input_item(item: ApiOutputItem) -> Option<ApiInputItem> {
    match item {
        ApiOutputItem::Message { content, .. } => {
            let parts: Vec<ApiInputContent> = content
                .into_iter()
                .filter_map(|content| match content {
                    ApiOutputContent::Text { text } if !text.is_empty() => {
                        Some(ApiInputContent::Text { text })
                    }
                    ApiOutputContent::Unknown | ApiOutputContent::Text { .. } => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(ApiInputItem::Message(ApiMessage {
                    role: ApiRole::Assistant,
                    content: ApiMessageContent::Parts(parts),
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
    tool_choice: Option<&'static str>,
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
    tool_choice: Option<String>,
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
    tool_choice: Option<String>,
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
            text: request.text,
            include: request.include.clone(),
            prompt_cache_key: request.prompt_cache_key.clone(),
            stream: request.stream,
            generate: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Serialize)]
struct ApiTextSettings {
    verbosity: &'static str,
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
}

#[derive(Clone, PartialEq, Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
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
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiInputContent {
    Text { text: String },
    Image { image_url: String },
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
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(rename = "role")]
        _role: String,
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
}

#[derive(Deserialize)]
struct ApiErrorBody {
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
struct ApiWrappedWebsocketErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
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
    fn test_openai_responses_rejects_adaptive_thinking() {
        let provider = OpenAICodexResponsesProvider::codex("test-key".to_string());
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
            tool_choice: Some("auto".to_string()),
            parallel_tool_calls: Some(true),
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
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
                }),
                ApiInputItem::Message(ApiMessage {
                    role: ApiRole::Assistant,
                    content: ApiMessageContent::Parts(vec![ApiInputContent::Text {
                        text: "answer".to_string(),
                    }]),
                }),
                ApiInputItem::Message(ApiMessage {
                    role: ApiRole::User,
                    content: ApiMessageContent::Text("follow up".to_string()),
                }),
            ],
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some("auto".to_string()),
            parallel_tool_calls: None,
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
            }),
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            prompt_cache_key: Some("thread-1".to_string()),
            stream: true,
        };
        let previous_request = ApiStreamingRequest {
            input: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Text("first".to_string()),
            })],
            ..request.clone()
        };
        let session = WebsocketSessionState {
            connection: None,
            last_request: Some(previous_request),
            last_response_id: Some("resp_prev".to_string()),
            last_response_items: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::Assistant,
                content: ApiMessageContent::Parts(vec![ApiInputContent::Text {
                    text: "answer".to_string(),
                }]),
            })],
            turn_state: None,
            prewarmed: false,
            websocket_disabled: false,
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
            }) => assert_eq!(text, "follow up"),
            _ => panic!("expected incremental follow-up user message"),
        }
    }

    #[test]
    fn test_parse_wrapped_websocket_error_event_maps_http_status() {
        let payload = r#"{"type":"error","status":401,"error":{"message":"unauthorized"}}"#;
        let parsed = parse_wrapped_websocket_error_event(payload);
        assert_eq!(
            parsed,
            Some((StatusCode::UNAUTHORIZED, "unauthorized".to_string())),
        );
    }

    #[test]
    fn test_parse_wrapped_websocket_error_event_maps_connection_limit() {
        let payload = format!(
            r#"{{"type":"error","status":429,"error":{{"code":"{OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE}","message":"limit"}}}}"#,
        );
        let parsed = parse_wrapped_websocket_error_event(&payload);
        assert_eq!(
            parsed,
            Some((StatusCode::TOO_MANY_REQUESTS, "limit".to_string())),
        );
    }

    #[test]
    fn test_prepare_websocket_request_allows_empty_delta_after_prewarm() {
        let request = ApiStreamingRequest {
            model: MODEL_GPT53_CODEX.to_string(),
            instructions: "system".to_string(),
            input: vec![ApiInputItem::Message(ApiMessage {
                role: ApiRole::User,
                content: ApiMessageContent::Text("first".to_string()),
            })],
            tools: None,
            max_output_tokens: None,
            reasoning: None,
            tool_choice: Some("auto".to_string()),
            parallel_tool_calls: None,
            store: false,
            text: Some(ApiTextSettings {
                verbosity: "medium",
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
    fn test_build_content_blocks_text() {
        let output = vec![ApiOutputItem::Message {
            _role: "assistant".to_owned(),
            content: vec![ApiOutputContent::Text {
                text: "Hello!".to_owned(),
            }],
        }];

        let blocks = build_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello!"));
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
}
