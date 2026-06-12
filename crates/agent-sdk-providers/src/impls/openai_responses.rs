//! `OpenAI` Responses API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Responses API (`/v1/responses`). This provider supports the Codex model family
//! and other agentic `OpenAI` models that expose the Responses surface.

use crate::attachments::validate_request_attachments;
use crate::provider::LlmProvider;
use crate::streaming::{SseLineBuffer, StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, Effort, ResponseFormat,
    StopReason, ThinkingConfig, ThinkingMode, ToolChoice, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

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
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set the reasoning effort level.
    #[must_use]
    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        self.with_thinking(ThinkingConfig::default().with_effort(map_reasoning_effort(effort)))
    }
}

#[async_trait]
impl LlmProvider for OpenAIResponsesProvider {
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
        let text = request.response_format.as_ref().map(ApiResponseText::from);
        let tool_choice = request.tool_choice.as_ref().map(ApiToolChoice::from);
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());
        let parallel_tool_calls = tools.as_ref().is_some_and(|tools| !tools.is_empty());

        let api_request = ApiResponsesRequest {
            model: &self.model,
            input: &input,
            tools: tools.as_deref(),
            max_output_tokens: Some(request.max_tokens),
            reasoning,
            parallel_tool_calls: parallel_tool_calls.then_some(true),
            text,
            tool_choice,
        };

        log::debug!(
            "OpenAI Responses API request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let builder = self
            .client
            .post(format!("{}/responses", self.base_url))
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
            let text = request.response_format.as_ref().map(ApiResponseText::from);
            let tool_choice = request.tool_choice.as_ref().map(ApiToolChoice::from);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| ts.into_iter().map(convert_tool).collect());
            let parallel_tool_calls = tools.as_ref().is_some_and(|tools| !tools.is_empty());

            let api_request = ApiResponsesRequestStreaming {
                model: &self.model,
                input: &input,
                tools: tools.as_deref(),
                max_output_tokens: Some(request.max_tokens),
                reasoning,
                parallel_tool_calls: parallel_tool_calls.then_some(true),
                text,
                tool_choice,
                stream: true,
            };

            log::debug!("OpenAI Responses API streaming request model={} max_tokens={}", self.model, request.max_tokens);

            let stream_builder = self.client
                .post(format!("{}/responses", self.base_url))
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
                let kind = if status == StatusCode::TOO_MANY_REQUESTS {
                    StreamErrorKind::RateLimited
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
            let mut usage: Option<Usage> = None;
            let mut tool_calls: std::collections::HashMap<String, ToolCallAccumulator> =
                std::collections::HashMap::new();

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error"));
                    return;
                };
                sse.extend(&chunk);

                while let Some(line) = sse.next_line() {
                    let line = line.trim();
                    if line.is_empty() { continue; }

                    let Some(data) = line.strip_prefix("data: ") else {
                        log::trace!("Responses SSE non-data line: {line}");
                        continue;
                    };
                    if log::log_enabled!(log::Level::Trace) {
                        let truncated: String = data.chars().take(200).collect();
                        log::trace!("Responses SSE data: {truncated}");
                    }

                    if data == "[DONE]" {
                        // Emit any accumulated tool calls with distinct,
                        // registration-ordered block indices.
                        for delta in flush_responses_tool_calls(&tool_calls) {
                            yield Ok(delta);
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
                    let parse_result = serde_json::from_str::<ApiStreamEvent>(data);
                    if parse_result.is_err() {
                        log::debug!("Failed to parse Responses SSE event: {data}");
                    }
                    if let Ok(event) = parse_result {
                        match event.r#type.as_str() {
                            // ── Content deltas ──────────────────────────
                            "response.output_text.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::TextDelta { delta, block_index: 0 });
                                }
                            }
                            "response.output_item.added" => {
                                // Register function_call items so we know
                                // the call_id and name before deltas arrive.
                                if let Some(item) = &event.item
                                    && item.r#type.as_deref() == Some("function_call")
                                    && let (Some(item_id), Some(call_id), Some(name)) =
                                        (&item.id, &item.call_id, &item.name)
                                {
                                    let order = tool_calls.len();
                                    tool_calls
                                        .entry(item_id.clone())
                                        .or_insert_with(|| ToolCallAccumulator {
                                            id: call_id.clone(),
                                            name: name.clone(),
                                            arguments: String::new(),
                                            order,
                                        });
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
                                            }
                                        });
                                    acc.arguments.push_str(&delta);
                                }
                            }
                            // ── Reasoning (thinking) deltas ─────────────
                            "response.reasoning.delta" => {
                                if let Some(delta) = event.delta {
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        delta,
                                        block_index: 0,
                                    });
                                }
                            }
                            // ── Completion / usage ──────────────────────
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
                                        cache_creation_input_tokens: 0,
                                    });
                                }
                            }
                            // ── Error ───────────────────────────────────
                            "error" | "response.failed" => {
                                let is_server_error = data.contains("server_error");
                                let kind = if is_server_error {
                                    log::warn!("Responses API server error (recoverable): {data}");
                                    StreamErrorKind::ServerError
                                } else {
                                    log::error!("Responses API error event: {data}");
                                    StreamErrorKind::InvalidRequest
                                };
                                yield Ok(StreamDelta::Error {
                                    message: data.to_owned(),
                                    kind,
                                });
                                return;
                            }
                            // ── Lifecycle events (no content) ───────────
                            "response.created"
                            | "response.in_progress"
                            | "response.output_item.done"
                            | "response.content_part.added"
                            | "response.content_part.done"
                            | "response.output_text.done"
                            | "response.function_call_arguments.done"
                            | "response.reasoning.done"
                            | "response.reasoning_summary_text.delta"
                            | "response.reasoning_summary_text.done" => {}
                            // ── Unknown ─────────────────────────────────
                            other => {
                                log::debug!("Unhandled Responses SSE event type: {other}");
                            }
                        }
                    }
                }
            }

            // Stream ended without [DONE] — flush accumulated tool calls with
            // distinct, registration-ordered block indices.
            for delta in flush_responses_tool_calls(&tool_calls) {
                yield Ok(delta);
            }

            if let Some(u) = usage {
                yield Ok(StreamDelta::Usage(u));
            }

            let stop_reason = if tool_calls.is_empty() {
                StopReason::EndTurn
            } else {
                StopReason::ToolUse
            };
            yield Ok(StreamDelta::Done { stop_reason: Some(stop_reason) });
        })
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

fn build_api_input(request: &ChatRequest) -> Vec<ApiInputItem> {
    let mut items = Vec::new();

    // Add system message if present
    if !request.system.is_empty() {
        items.push(ApiInputItem::Message(ApiMessage {
            role: ApiRole::System,
            content: ApiMessageContent::Text(request.system.clone()),
        }));
    }

    // Convert messages
    for msg in &request.messages {
        match &msg.content {
            Content::Text(text) => {
                items.push(ApiInputItem::Message(ApiMessage {
                    role: match msg.role {
                        agent_sdk_foundation::llm::Role::User => ApiRole::User,
                        agent_sdk_foundation::llm::Role::Assistant => ApiRole::Assistant,
                    },
                    content: ApiMessageContent::Text(text.clone()),
                }));
            }
            Content::Blocks(blocks) => {
                let mut content_parts = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            let part = match msg.role {
                                agent_sdk_foundation::llm::Role::Assistant => {
                                    ApiInputContent::OutputText { text: text.clone() }
                                }
                                agent_sdk_foundation::llm::Role::User => {
                                    ApiInputContent::InputText { text: text.clone() }
                                }
                            };
                            content_parts.push(part);
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
                        // `ContentBlock` is `#[non_exhaustive]`; a block kind this
                        // SDK version cannot represent on the wire is skipped.
                        _ => {
                            log::warn!("Skipping unrecognized OpenAI Responses content block");
                        }
                    }
                }

                if !content_parts.is_empty() {
                    items.push(ApiInputItem::Message(ApiMessage {
                        role: match msg.role {
                            agent_sdk_foundation::llm::Role::User => ApiRole::User,
                            agent_sdk_foundation::llm::Role::Assistant => ApiRole::Assistant,
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

fn convert_tool(tool: agent_sdk_foundation::llm::Tool) -> ApiTool {
    let mut schema = tool.input_schema;

    // Strict mode requires additionalProperties: false on all objects and
    // every property in required. This is incompatible with free-form object
    // schemas (objects with no defined properties). Detect and skip strict
    // for those tools.
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

/// Check if a JSON schema contains any object-typed properties without
/// defined `properties` (free-form objects). These are incompatible with
/// `OpenAI` strict mode.
/// Wrap a schema in `anyOf: [{original}, {"type": "null"}]` so that
/// the property accepts its original type OR null.
///
/// If the schema already has an `anyOf`, appends `{"type": "null"}` to it.
fn make_nullable(schema: &mut serde_json::Value) {
    // Already nullable via anyOf — append null variant if missing
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

    // Wrap the original schema in anyOf
    let original = schema.clone();
    *schema = serde_json::json!({
        "anyOf": [original, {"type": "null"}]
    });
}

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

    // Recurse into properties
    if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
        for prop in props.values() {
            if has_freeform_object(prop) {
                return true;
            }
        }
    }

    // Recurse into array items
    if let Some(items) = obj.get("items")
        && has_freeform_object(items)
    {
        return true;
    }

    // Recurse into anyOf/oneOf/allOf
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
            // Unreachable: Failed is handled above, but map defensively.
            ApiStatus::Failed => StopReason::StopSequence,
        })
    };

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
            cache_creation_input_tokens: 0,
        },
    )
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
    accs.sort_by_key(|acc| acc.order);

    let mut deltas = Vec::with_capacity(accs.len() * 2);
    for (idx, acc) in accs.iter().enumerate() {
        let block_index = idx + 1;
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
    stream: bool,
}

#[derive(Serialize)]
struct ApiReasoning {
    effort: ReasoningEffort,
}

/// Responses API structured-output wire field: `{"text": {"format": {...}}}`.
///
/// The Responses API carries JSON-schema structured output under
/// `text.format` (type `json_schema`), unlike Chat Completions' top-level
/// `response_format`.
#[derive(Serialize)]
struct ApiResponseText {
    format: ApiResponseTextFormat,
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
        Self {
            format: ApiResponseTextFormat {
                format_type: "json_schema",
                name: rf.name.clone(),
                schema: rf.schema.clone(),
                strict: rf.strict,
            },
        }
    }
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
    System,
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
}

#[derive(Deserialize)]
struct ApiResponseError {
    #[serde(default)]
    message: Option<String>,
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
    /// Present on `output_item.added` / `output_item.done` for `function_call` items.
    #[serde(default)]
    item: Option<ApiStreamItem>,
    /// Present on `function_call_arguments.delta`.
    #[serde(default)]
    item_id: Option<String>,
    /// Legacy field — some older events use `call_id` instead of `item_id`.
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    response: Option<ApiStreamResponse>,
}

impl ApiStreamEvent {
    /// Resolve the item identifier from whichever field is present.
    fn resolve_item_id(&self) -> Option<&str> {
        self.item_id
            .as_deref()
            .or(self.call_id.as_deref())
            .or_else(|| self.item.as_ref().and_then(|i| i.id.as_deref()))
    }
}

#[derive(Deserialize)]
struct ApiStreamItem {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
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
        assert_eq!(MODEL_GPT53_CODEX, "gpt-5.3-codex");
        assert_eq!(MODEL_GPT52_CODEX, "gpt-5.2-codex");
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
    fn test_reasoning_effort_serialization() {
        let low = serde_json::to_string(&ReasoningEffort::Low).unwrap();
        assert_eq!(low, "\"low\"");

        let xhigh = serde_json::to_string(&ReasoningEffort::XHigh).unwrap();
        assert_eq!(xhigh, "\"xhigh\"");
    }

    #[test]
    fn test_with_reasoning_effort() {
        let provider = OpenAIResponsesProvider::codex("test-key".to_string())
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
        let provider = OpenAIResponsesProvider::codex("test-key".to_string());
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
                serde_json::json!({"type": "object"}),
            ))),
            tool_choice: Some(ApiToolChoice::from(&ToolChoice::Tool("respond".to_owned()))),
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
            },
        );
        tool_calls.insert(
            "a".to_owned(),
            ToolCallAccumulator {
                id: "a".to_owned(),
                name: "first".to_owned(),
                arguments: "{}".to_owned(),
                order: 0,
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
            vec![("first".to_owned(), 1), ("second".to_owned(), 2)]
        );
    }
}
