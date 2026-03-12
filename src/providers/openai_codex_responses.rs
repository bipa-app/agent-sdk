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
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const OPENAI_CODEX_JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

// GPT-5.4 (frontier reasoning with 1.05M context)
pub const MODEL_GPT54: &str = "gpt-5.4";

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
    ) -> Result<reqwest::header::HeaderMap> {
        use reqwest::header::{
            ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT,
        };

        let account_id = extract_account_id(&self.api_key)
            .context("failed to extract chatgpt account id from OpenAI Codex OAuth token")?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))?,
        );
        headers.insert("chatgpt-account-id", HeaderValue::from_str(&account_id)?);
        headers.insert(
            "OpenAI-Beta",
            HeaderValue::from_static("responses=experimental"),
        );
        headers.insert("originator", HeaderValue::from_static("bip"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&format!(
                "bip ({} {}; {})",
                std::env::consts::OS,
                std::env::consts::ARCH,
                env!("CARGO_PKG_VERSION")
            ))?,
        );
        if streaming {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        if let Some(session_id) = session_id {
            headers.insert("session_id", HeaderValue::from_str(session_id)?);
        }

        Ok(headers)
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

        let api_request = ApiResponsesRequest {
            model: &self.model,
            instructions: request.system.as_str(),
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens,
            reasoning,
            tool_choice: Some("auto"),
            parallel_tool_calls: Some(true),
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
            .headers(self.build_headers(false, request.session_id.as_deref())?)
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

        let content = build_content_blocks(&api_response.output);

        // Determine stop reason based on output content
        let has_tool_calls = content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        let stop_reason = if has_tool_calls {
            Some(StopReason::ToolUse)
        } else {
            api_response.status.map(|s| match s {
                ApiStatus::Completed => StopReason::EndTurn,
                ApiStatus::Incomplete => StopReason::MaxTokens,
                ApiStatus::Failed => StopReason::StopSequence,
            })
        };

        Ok(ChatOutcome::Success(ChatResponse {
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
                |u| Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    cached_input_tokens: u
                        .input_tokens_details
                        .as_ref()
                        .map_or(0, |details| details.cached_tokens),
                },
            ),
        }))
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
            let prompt_cache_key = request.session_id.as_deref();
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .as_ref()
                .map(|ts| ts.iter().cloned().map(convert_tool).collect());

            let api_request = ApiResponsesRequestStreaming {
                model: &self.model,
                instructions: request.system.as_str(),
                input: &input,
                tools: tools.as_deref(),
                max_output_tokens,
                reasoning,
                tool_choice: Some("auto"),
                parallel_tool_calls: Some(true),
                text: Some(ApiTextSettings { verbosity: "medium" }),
                include: Some(&["reasoning.encrypted_content"]),
                prompt_cache_key,
                stream: true,
            };

            log::debug!("OpenAI Codex streaming request model={} max_tokens={}", self.model, request.max_tokens);

            let headers = match self.build_headers(true, request.session_id.as_deref()) {
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

            let mut buffer = String::new();
            let mut stream = response.bytes_stream();
            let mut usage: Option<Usage> = None;
            let mut tool_calls: std::collections::HashMap<String, ToolCallAccumulator> =
                std::collections::HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error"));
                    return;
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() { continue; }

                    let Some(data) = line.strip_prefix("data: ") else { continue; };

                    if data == "[DONE]" {
                        // Emit any accumulated tool calls
                        for acc in tool_calls.values() {
                            yield Ok(StreamDelta::ToolUseStart {
                                id: acc.id.clone(),
                                name: acc.name.clone(),
                                block_index: 1,
                                thought_signature: None,
                            });
                            yield Ok(StreamDelta::ToolInputDelta {
                                id: acc.id.clone(),
                                delta: acc.arguments.clone(),
                                block_index: 1,
                            });
                        }

                        if let Some(u) = usage.take() {
                            yield Ok(StreamDelta::Usage(u));
                        }

                        let stop_reason = if tool_calls.is_empty() {
                            StopReason::EndTurn
                        } else {
                            StopReason::ToolUse
                        };
                        yield Ok(StreamDelta::Done { stop_reason: Some(stop_reason) });
                        return;
                    }

                    // Parse streaming event
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
                            "response.completed" => {
                                if let Some(resp) = event.response
                                    && let Some(u) = resp.usage
                                {
                                    usage = Some(Usage {
                                        input_tokens: u.input_tokens,
                                        output_tokens: u.output_tokens,
                                        cached_input_tokens: u
                                            .input_tokens_details
                                            .as_ref()
                                            .map_or(0, |details| details.cached_tokens),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Stream ended without [DONE]
            if let Some(u) = usage {
                yield Ok(StreamDelta::Usage(u));
            }
            yield Ok(StreamDelta::Done { stop_reason: Some(StopReason::EndTurn) });
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
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiTextSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [&'static str]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
}

#[derive(Serialize)]
struct ApiResponsesRequestStreaming<'a> {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ApiTextSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [&'static str]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    stream: bool,
}

#[derive(Clone, Copy, Serialize)]
struct ApiTextSettings {
    verbosity: &'static str,
}

#[derive(Serialize)]
struct ApiReasoning {
    effort: ReasoningEffort,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiInputItem {
    Message(ApiMessage),
    FunctionCall(ApiFunctionCall),
    FunctionCallOutput(ApiFunctionCallOutput),
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    content: ApiMessageContent,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    User,
    Assistant,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Parts(Vec<ApiInputContent>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiInputContent {
    Text { text: String },
    Image { image_url: String },
    File { filename: String, file_data: String },
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
}

#[derive(Deserialize)]
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
    response: Option<ApiStreamResponse>,
}

#[derive(Deserialize)]
struct ApiStreamResponse {
    #[serde(default)]
    usage: Option<ApiUsage>,
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
