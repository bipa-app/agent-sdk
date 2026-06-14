//! Google Gemini API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Google Gemini
//! API (`generativelanguage.googleapis.com`).

pub(crate) mod data;

use crate::attachments::validate_request_attachments;
use crate::provider::LlmProvider;
use crate::streaming::{StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{ChatOutcome, ChatRequest, ChatResponse, ThinkingConfig};
use anyhow::Result;
use async_trait::async_trait;
use data::{
    ApiContent, ApiFunctionCallingConfig, ApiGenerateContentRequest, ApiGenerateContentResponse,
    ApiGenerationConfig, ApiPart, ApiUsageMetadata, build_api_contents, build_content_blocks,
    convert_tools_to_config, gemini_response_schema, map_finish_reason, map_thinking_config,
};
use reqwest::StatusCode;

const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Connect timeout for the HTTP client (matches Anthropic/Vertex).
const CONNECT_TIMEOUT_SECS: u64 = 30;
/// TCP keepalive interval to keep long streaming connections from dropping.
const TCP_KEEPALIVE_SECS: u64 = 30;
/// Per-request read timeout for the **non-streaming** `chat()` path. Bounds a
/// black-holed endpoint so a single turn cannot hang the agent loop forever.
/// Streaming requests intentionally have no overall timeout.
const CHAT_READ_TIMEOUT_SECS: u64 = 300;

/// Max page size the Gemini `ListModels` endpoint accepts (default is 50).
const MODELS_PAGE_SIZE: u32 = 1000;
/// Upper bound on pages followed by `list_models`, guarding against a server
/// that never clears `nextPageToken`.
const MODELS_MAX_PAGES: usize = 100;

/// Build the shared HTTP client with connect + keepalive timeouts, falling back
/// to a default client (with a logged warning) if the builder fails.
fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .tcp_keepalive(std::time::Duration::from_secs(TCP_KEEPALIVE_SECS))
        .build()
        .unwrap_or_else(|error| {
            log::warn!(
                "failed to build Gemini HTTP client with timeouts ({error}); using default client"
            );
            reqwest::Client::new()
        })
}

// Gemini 3.1 series
pub const MODEL_GEMINI_31_PRO: &str = "gemini-3.1-pro-preview";
pub const MODEL_GEMINI_31_FLASH_LITE: &str = "gemini-3.1-flash-lite-preview";

// Gemini 3 series
pub const MODEL_GEMINI_3_FLASH: &str = "gemini-3-flash-preview";

// Legacy Gemini 3.0 Pro model kept for explicit opt-in.
pub const MODEL_GEMINI_3_PRO: &str = "gemini-3.0-pro";

// Gemini 2.5 series
pub const MODEL_GEMINI_25_FLASH: &str = "gemini-2.5-flash";
pub const MODEL_GEMINI_25_PRO: &str = "gemini-2.5-pro";

// Gemini 2.0 series
pub const MODEL_GEMINI_2_FLASH: &str = "gemini-2.0-flash";
pub const MODEL_GEMINI_2_FLASH_LITE: &str = "gemini-2.0-flash-lite";

/// Google Gemini LLM provider.
#[derive(Clone)]
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    thinking: Option<ThinkingConfig>,
    /// When true, send the API key via `x-goog-api-key` header instead of a
    /// query parameter. Required when routing through proxies.
    use_header_auth: bool,
    /// Extra headers applied to every request (e.g. for gateway authentication).
    extra_headers: Vec<(String, String)>,
}

impl GeminiProvider {
    /// The conventional environment variable holding the Gemini API key.
    pub const API_KEY_ENV: &'static str = "GEMINI_API_KEY";

    /// Create a new Gemini provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: build_http_client(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: API_BASE_URL.to_owned(),
            thinking: None,
            use_header_auth: true,
            extra_headers: Vec::new(),
        }
    }

    /// Effective output-token budget for a request.
    ///
    /// Mirrors the Anthropic provider: when the caller did not explicitly set
    /// `max_tokens`, substitute the provider/model default
    /// ([`default_max_tokens`](LlmProvider::default_max_tokens)) instead of
    /// silently capping at `ChatRequest::DEFAULT_MAX_TOKENS`.
    fn effective_max_tokens(&self, request: &ChatRequest) -> u32 {
        if request.max_tokens_explicit {
            request.max_tokens
        } else {
            self.default_max_tokens()
        }
    }

    /// Create a provider using Gemini Flash, reading the API key from the
    /// conventional [`GEMINI_API_KEY`](Self::API_KEY_ENV) environment variable.
    ///
    /// # Panics
    ///
    /// Panics if `GEMINI_API_KEY` is not set. Prefer
    /// [`try_from_env`](Self::try_from_env) outside of examples/tests.
    #[must_use]
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a provider using Gemini Flash, reading the API key from the
    /// conventional [`GEMINI_API_KEY`](Self::API_KEY_ENV) environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `GEMINI_API_KEY` is unset or not valid UTF-8.
    pub fn try_from_env() -> Result<Self> {
        let api_key = std::env::var(Self::API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!("environment variable `{}` is not set", Self::API_KEY_ENV)
        })?;
        Ok(Self::flash(api_key))
    }

    /// Create a provider using Gemini 3 Flash Preview (fast and capable, current default).
    #[must_use]
    pub fn flash(api_key: impl Into<String>) -> Self {
        Self::new(api_key, MODEL_GEMINI_3_FLASH)
    }

    /// Create a provider using Gemini 3.1 Flash Lite Preview.
    #[must_use]
    pub fn flash_lite_31(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_31_FLASH_LITE.to_owned())
    }

    /// Create a provider using Gemini 2.0 Flash Lite (fastest, most cost-effective).
    #[must_use]
    pub fn flash_lite(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_2_FLASH_LITE.to_owned())
    }

    /// Create a provider using Gemini 3.1 Pro Preview.
    #[must_use]
    pub fn pro_31(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_31_PRO.to_owned())
    }

    /// Create a provider using Gemini 3.1 Pro Preview (current recommended pro model).
    #[must_use]
    pub fn pro(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_31_PRO.to_owned())
    }

    /// Set the provider-owned thinking configuration for this model.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Override the base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Send the API key via `x-goog-api-key` header instead of `?key=` query
    /// parameter. Required when routing through proxies.
    #[must_use]
    pub const fn with_header_auth(mut self) -> Self {
        self.use_header_auth = true;
        self
    }

    /// Add extra HTTP headers applied to every request.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Apply auth + extra headers. Skips provider auth when `api_key` is
    /// empty (BYOK gateway mode).
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let builder = if self.api_key.is_empty() {
            builder
        } else if self.use_header_auth {
            builder.header("x-goog-api-key", &self.api_key)
        } else {
            builder.query(&[("key", &self.api_key)])
        };
        self.extra_headers
            .iter()
            .fold(builder, |b, (k, v)| b.header(k.as_str(), v.as_str()))
    }
}

#[async_trait]
#[allow(clippy::too_many_lines)]
impl LlmProvider for GeminiProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
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
            "Gemini LLM request model={} max_tokens={}",
            self.model,
            max_tokens
        );

        let builder = self
            .client
            .post(format!(
                "{}/models/{}:generateContent",
                self.base_url, self.model
            ))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(CHAT_READ_TIMEOUT_SECS));
        let response = self
            .apply_auth(builder)
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
            "Gemini LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited(retry_after));
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("Gemini server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("Gemini client error status={status} body={body}");
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
                "Gemini parts not converted to content blocks raw_parts={:?}",
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

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
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
            let (response_mime_type, response_schema) = request
                .response_format
                .as_ref()
                .map_or((None, None), |rf| {
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
                "Gemini streaming LLM request model={} max_tokens={}",
                self.model,
                max_tokens
            );

            let stream_builder = self
                .client
                .post(format!(
                    "{}/models/{}:streamGenerateContent",
                    self.base_url, self.model
                ))
                .header("Content-Type", "application/json")
                .query(&[("alt", "sse")]);
            let response = match self
                .apply_auth(stream_builder)
                .json(&api_request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    // Include the cause so 401 detection / diagnostics survive.
                    yield Err(anyhow::anyhow!("request failed: {e}"));
                    return;
                }
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
                log::warn!("Gemini error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    kind,
                });
                return;
            }

            let mut inner = data::stream_gemini_response(response);
            while let Some(item) = futures::StreamExt::next(&mut inner).await {
                yield item;
            }
        })
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
        // The endpoint paginates (default `pageSize=50`). Request the max page
        // size and follow `nextPageToken` until exhausted, collecting *raw*
        // rows. The `generateContent` filter is applied only after every page is
        // in hand, so server-side truncation cannot hide a chat-capable model.
        let mut rows: Vec<GeminiModelRow> = Vec::new();
        let mut page_token: Option<String> = None;
        for _ in 0..MODELS_MAX_PAGES {
            let mut query: Vec<(&str, String)> = vec![("pageSize", MODELS_PAGE_SIZE.to_string())];
            if let Some(token) = &page_token {
                query.push(("pageToken", token.clone()));
            }
            let builder = self
                .client
                .get(format!("{}/models", self.base_url))
                .header("Content-Type", "application/json")
                .query(&query);
            let builder = self.apply_auth(builder);
            let body =
                crate::impls::model_listing::fetch_model_list_body(builder, "Gemini").await?;
            let page = parse_models_page(&body)?;
            rows.extend(page.models);
            match page.next_page_token {
                Some(token) if !token.is_empty() => page_token = Some(token),
                _ => break,
            }
        }
        Ok(finalize_gemini_models(rows))
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "gemini"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }
}

/// A raw Gemini model row, kept un-filtered so the `generateContent` filter can
/// be applied only *after* every page has been collected (so server-side page
/// truncation cannot hide a chat-capable model behind a page boundary).
#[derive(serde::Deserialize)]
struct GeminiModelRow {
    name: String,
    #[serde(rename = "displayName", default)]
    display_name: Option<String>,
    #[serde(rename = "inputTokenLimit", default)]
    input_token_limit: Option<u32>,
    #[serde(rename = "outputTokenLimit", default)]
    output_token_limit: Option<u32>,
    #[serde(rename = "supportedGenerationMethods", default)]
    supported_generation_methods: Vec<String>,
}

/// One page of the Gemini `ListModels` response: raw rows plus the cursor used
/// to follow pagination.
struct GeminiModelsPage {
    models: Vec<GeminiModelRow>,
    next_page_token: Option<String>,
}

/// Parse one page of the Gemini `GET /v1beta/models` response body.
///
/// The endpoint returns `{ "models": [{ "name": "models/<id>", "displayName",
/// "inputTokenLimit", "outputTokenLimit", "supportedGenerationMethods" }],
/// "nextPageToken": "..." }`. It paginates with a default `pageSize` of 50;
/// `nextPageToken` drives the next request. Raw rows are returned un-filtered so
/// the caller can apply the `generateContent` filter once all pages are in hand.
fn parse_models_page(body: &str) -> Result<GeminiModelsPage> {
    #[derive(serde::Deserialize)]
    struct ListResponse {
        #[serde(default)]
        models: Vec<GeminiModelRow>,
        #[serde(rename = "nextPageToken", default)]
        next_page_token: Option<String>,
    }
    let parsed: ListResponse = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("failed to parse Gemini models list: {e}"))?;
    Ok(GeminiModelsPage {
        models: parsed.models,
        next_page_token: parsed.next_page_token,
    })
}

/// Filter accumulated rows to chat-capable models and project them into
/// [`ModelInfo`].
///
/// Entries that do not support `generateContent` (e.g. embedding-only models)
/// are dropped, and the `models/` prefix is stripped from `name` to recover the
/// bare model id the chat endpoint expects. Applied *after* all pages are
/// collected so a chat-capable model never gets hidden by page truncation.
fn finalize_gemini_models(rows: Vec<GeminiModelRow>) -> Vec<crate::provider::ModelInfo> {
    rows.into_iter()
        .filter(|row| {
            row.supported_generation_methods.is_empty()
                || row
                    .supported_generation_methods
                    .iter()
                    .any(|m| m == "generateContent")
        })
        .map(|row| crate::provider::ModelInfo {
            id: match row.name.strip_prefix("models/") {
                Some(stripped) => stripped.to_owned(),
                None => row.name.clone(),
            },
            display_name: row.display_name,
            context_window: row.input_token_limit,
            max_output_tokens: row.output_token_limit,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMINI_MODELS_FIXTURE: &str = r#"{
      "models": [
        {
          "name": "models/gemini-2.5-pro",
          "displayName": "Gemini 2.5 Pro",
          "inputTokenLimit": 1048576,
          "outputTokenLimit": 65536,
          "supportedGenerationMethods": ["generateContent", "countTokens"]
        },
        {
          "name": "models/text-embedding-004",
          "displayName": "Text Embedding 004",
          "inputTokenLimit": 2048,
          "outputTokenLimit": 1,
          "supportedGenerationMethods": ["embedContent"]
        }
      ]
    }"#;

    #[test]
    fn parse_models_page_strips_prefix_and_maps_limits() -> anyhow::Result<()> {
        let page = parse_models_page(GEMINI_MODELS_FIXTURE)?;
        let models = finalize_gemini_models(page.models);
        // The embedding-only model is filtered out (no `generateContent`).
        assert_eq!(models.len(), 1);
        let pro = &models[0];
        assert_eq!(pro.id, "gemini-2.5-pro");
        assert_eq!(pro.display_name.as_deref(), Some("Gemini 2.5 Pro"));
        assert_eq!(pro.context_window, Some(1_048_576));
        assert_eq!(pro.max_output_tokens, Some(65_536));
        assert_eq!(page.next_page_token, None);
        Ok(())
    }

    #[tokio::test]
    async fn list_models_follows_pagination_and_filters_after_all_pages() -> anyhow::Result<()> {
        use wiremock::matchers::{method, path, query_param, query_param_is_missing};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        // Page 1: a chat model plus an embedding-only model, then a page token.
        // The embedding model must NOT be filtered out mid-pagination — the
        // filter runs only after every page is collected.
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param_is_missing("pageToken"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{
                  "models": [
                    {
                      "name": "models/gemini-2.5-pro",
                      "displayName": "Gemini 2.5 Pro",
                      "inputTokenLimit": 1048576,
                      "outputTokenLimit": 65536,
                      "supportedGenerationMethods": ["generateContent"]
                    },
                    {
                      "name": "models/text-embedding-004",
                      "displayName": "Embedding",
                      "supportedGenerationMethods": ["embedContent"]
                    }
                  ],
                  "nextPageToken": "page-2"
                }"#,
            ))
            .mount(&server)
            .await;

        // Page 2: requested with `pageToken=page-2`; final page (no token).
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param("pageToken", "page-2"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{
                  "models": [
                    {
                      "name": "models/gemini-3-flash",
                      "displayName": "Gemini 3 Flash",
                      "inputTokenLimit": 1048576,
                      "outputTokenLimit": 65536,
                      "supportedGenerationMethods": ["generateContent"]
                    }
                  ]
                }"#,
            ))
            .mount(&server)
            .await;

        let provider = GeminiProvider::new("test-key".to_string(), "gemini-test".to_string())
            .with_base_url(server.uri());
        let models = provider.list_models().await?;

        // Both chat models from both pages are returned; the embedding-only
        // model is dropped by the post-pagination filter.
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["gemini-2.5-pro", "gemini-3-flash"]);
        Ok(())
    }

    #[test]
    fn test_new_creates_provider_with_custom_model() {
        let provider = GeminiProvider::new("test-api-key".to_string(), "custom-model".to_string());

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_flash_factory_creates_flash_provider() {
        let provider = GeminiProvider::flash("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_3_FLASH);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_flash_lite_factory_creates_flash_lite_provider() {
        let provider = GeminiProvider::flash_lite("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_2_FLASH_LITE);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_flash_lite_31_factory_creates_flash_lite_provider() {
        let provider = GeminiProvider::flash_lite_31("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_31_FLASH_LITE);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_pro_factory_creates_pro_provider() {
        let provider = GeminiProvider::pro("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_31_PRO);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_pro_31_factory_creates_pro_provider() {
        let provider = GeminiProvider::pro_31("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_31_PRO);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_model_constants_have_expected_values() {
        assert_eq!(MODEL_GEMINI_31_PRO, "gemini-3.1-pro-preview");
        assert_eq!(MODEL_GEMINI_31_FLASH_LITE, "gemini-3.1-flash-lite-preview");
        assert_eq!(MODEL_GEMINI_3_FLASH, "gemini-3-flash-preview");
        assert_eq!(MODEL_GEMINI_3_PRO, "gemini-3.0-pro");
        assert_eq!(MODEL_GEMINI_25_FLASH, "gemini-2.5-flash");
        assert_eq!(MODEL_GEMINI_25_PRO, "gemini-2.5-pro");
        assert_eq!(MODEL_GEMINI_2_FLASH, "gemini-2.0-flash");
        assert_eq!(MODEL_GEMINI_2_FLASH_LITE, "gemini-2.0-flash-lite");
    }

    #[test]
    fn test_gemini_20_models_reject_thinking() {
        let provider = GeminiProvider::flash_lite("test-api-key".to_string());
        let error = provider
            .validate_thinking_config(Some(&ThinkingConfig::new(10_000)))
            .unwrap_err();
        assert!(error.to_string().contains("thinking is not supported"));
    }

    #[test]
    fn test_default_uses_header_auth() {
        let provider = GeminiProvider::new("test-key".to_string(), "model".to_string());
        assert!(
            provider.use_header_auth,
            "Default should use header auth for security"
        );
    }

    #[test]
    fn test_provider_is_cloneable() {
        let provider = GeminiProvider::new("test-api-key".to_string(), "test-model".to_string());
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
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
        let provider = GeminiProvider::pro("test-api-key".to_string());
        let request = request_with_max_tokens(123, true);
        assert_eq!(provider.effective_max_tokens(&request), 123);
    }

    #[test]
    fn test_effective_max_tokens_uses_default_when_implicit() {
        // An implicit budget must fall back to the provider/model default, not
        // be silently capped at ChatRequest::DEFAULT_MAX_TOKENS.
        let provider = GeminiProvider::pro("test-api-key".to_string());
        let request = request_with_max_tokens(4096, false);
        assert_eq!(
            provider.effective_max_tokens(&request),
            provider.default_max_tokens()
        );
    }
}
