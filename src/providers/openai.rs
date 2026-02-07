//! `OpenAI` API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the `OpenAI`
//! Chat Completions API. It also supports `OpenAI`-compatible APIs (Ollama, vLLM, etc.)
//! via the `with_base_url` constructor.
//!
//! Models that require the Responses API (like `gpt-5.2-codex`) are automatically
//! routed to the correct endpoint.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, LlmProvider, StopReason,
    StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::openai_responses::OpenAIResponsesProvider;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Check if a model requires the Responses API instead of Chat Completions.
fn requires_responses_api(model: &str) -> bool {
    model.contains("codex")
}

// GPT-5.2 series (latest flagship, Dec 2025)
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
}

impl OpenAIProvider {
    /// Create a new `OpenAI` provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Create a new provider with a custom base URL for OpenAI-compatible APIs.
    #[must_use]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
        }
    }

    /// Create a provider using GPT-5.2 Instant (speed-optimized for routine queries).
    #[must_use]
    pub fn gpt52_instant(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_INSTANT.to_owned())
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

    /// Create a provider using GPT-5.2 Codex (optimized for agentic coding).
    ///
    /// Note: This model uses the Responses API internally.
    #[must_use]
    pub fn codex(api_key: String) -> Self {
        Self::new(api_key, MODEL_GPT52_CODEX.to_owned())
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
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        // Route to Responses API for models that require it (e.g., gpt-5.2-codex)
        if requires_responses_api(&self.model) {
            let responses_provider =
                OpenAIResponsesProvider::new(self.api_key.clone(), self.model.clone());
            return responses_provider.chat(request).await;
        }

        let messages = build_api_messages(&request);
        let tools: Option<Vec<ApiTool>> = request
            .tools
            .map(|ts| ts.into_iter().map(convert_tool).collect());

        let api_request = ApiChatRequest {
            model: &self.model,
            messages: &messages,
            max_completion_tokens: Some(request.max_tokens),
            tools: tools.as_deref(),
        };

        log::debug!(
            "OpenAI LLM request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            "OpenAI LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("OpenAI server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("OpenAI client error status={status} body={body}");
            return Ok(ChatOutcome::InvalidRequest(body.into_owned()));
        }

        let api_response: ApiChatResponse = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse response: {e}"))?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no choices in response"))?;

        let content = build_content_blocks(&choice.message);

        let stop_reason = choice.finish_reason.map(|r| match r {
            ApiFinishReason::Stop => StopReason::EndTurn,
            ApiFinishReason::ToolCalls => StopReason::ToolUse,
            ApiFinishReason::Length => StopReason::MaxTokens,
            ApiFinishReason::ContentFilter => StopReason::StopSequence,
        });

        Ok(ChatOutcome::Success(ChatResponse {
            id: api_response.id,
            content,
            model: api_response.model,
            stop_reason,
            usage: Usage {
                input_tokens: api_response.usage.prompt_tokens,
                output_tokens: api_response.usage.completion_tokens,
            },
        }))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        // Route to Responses API for models that require it (e.g., gpt-5.2-codex)
        if requires_responses_api(&self.model) {
            let api_key = self.api_key.clone();
            let model = self.model.clone();
            return Box::pin(async_stream::stream! {
                let responses_provider = OpenAIResponsesProvider::new(api_key, model);
                let mut stream = std::pin::pin!(responses_provider.chat_stream(request));
                while let Some(item) = futures::StreamExt::next(&mut stream).await {
                    yield item;
                }
            });
        }

        Box::pin(async_stream::stream! {
            let messages = build_api_messages(&request);
            let tools: Option<Vec<ApiTool>> = request
                .tools
                .map(|ts| ts.into_iter().map(convert_tool).collect());

            let api_request = ApiChatRequestStreaming { model: &self.model, messages: &messages, max_completion_tokens: Some(request.max_tokens), tools: tools.as_deref(), stream: true };

            log::debug!("OpenAI streaming LLM request model={} max_tokens={}", self.model, request.max_tokens);

            let Ok(response) = self.client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
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
                let (recoverable, level) = if status == StatusCode::TOO_MANY_REQUESTS {
                    (true, "rate_limit")
                } else if status.is_server_error() {
                    (true, "server_error")
                } else {
                    (false, "client_error")
                };
                log::warn!("OpenAI error status={status} body={body} kind={level}");
                yield Ok(StreamDelta::Error { message: body, recoverable });
                return;
            }

            // Track tool call state across deltas
            let mut tool_calls: std::collections::HashMap<usize, ToolCallAccumulator> =
                std::collections::HashMap::new();
            let mut usage: Option<Usage> = None;
            let mut buffer = String::new();
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let Ok(chunk) = chunk_result else {
                    yield Err(anyhow::anyhow!("stream error: {}", chunk_result.unwrap_err()));
                    return;
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() { continue; }
                    let Some(data) = line.strip_prefix("data: ") else { continue; };

                    for result in process_sse_data(data) {
                        match result {
                            SseProcessResult::TextDelta(c) => yield Ok(StreamDelta::TextDelta { delta: c, block_index: 0 }),
                            SseProcessResult::ToolCallUpdate { index, id, name, arguments } => apply_tool_call_update(&mut tool_calls, index, id, name, arguments),
                            SseProcessResult::Usage(u) => usage = Some(u),
                            SseProcessResult::Done(sr) => {
                                for d in build_stream_end_deltas(&tool_calls, usage.take(), sr) { yield Ok(d); }
                                return;
                            }
                            SseProcessResult::Sentinel => {
                                let sr = if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };
                                for d in build_stream_end_deltas(&tool_calls, usage.take(), sr) { yield Ok(d); }
                                return;
                            }
                        }
                    }
                }
            }

            // Stream ended without [DONE] - emit what we have
            for delta in build_stream_end_deltas(&tool_calls, usage, StopReason::EndTurn) {
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

/// Helper to emit tool call deltas and done event.
fn build_stream_end_deltas(
    tool_calls: &std::collections::HashMap<usize, ToolCallAccumulator>,
    usage: Option<Usage>,
    stop_reason: StopReason,
) -> Vec<StreamDelta> {
    let mut deltas = Vec::new();

    // Emit tool calls
    for (idx, tool) in tool_calls {
        deltas.push(StreamDelta::ToolUseStart {
            id: tool.id.clone(),
            name: tool.name.clone(),
            block_index: *idx + 1,
        });
        deltas.push(StreamDelta::ToolInputDelta {
            id: tool.id.clone(),
            delta: tool.arguments.clone(),
            block_index: *idx + 1,
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
        }));
    }

    // Process choices
    if let Some(choice) = chunk.choices.into_iter().next() {
        // Handle text content delta
        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            results.push(SseProcessResult::TextDelta(content));
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
            let stop_reason = match finish_reason {
                SseFinishReason::Stop => StopReason::EndTurn,
                SseFinishReason::ToolCalls => StopReason::ToolUse,
                SseFinishReason::Length => StopReason::MaxTokens,
                SseFinishReason::ContentFilter => StopReason::StopSequence,
            };
            results.push(SseProcessResult::Done(stop_reason));
        }
    }

    results
}

fn build_api_messages(request: &ChatRequest) -> Vec<ApiMessage> {
    let mut messages = Vec::new();

    // Add system message first (OpenAI uses a separate message for system prompt)
    if !request.system.is_empty() {
        messages.push(ApiMessage {
            role: ApiRole::System,
            content: Some(request.system.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert SDK messages to OpenAI format
    for msg in &request.messages {
        match &msg.content {
            Content::Text(text) => {
                messages.push(ApiMessage {
                    role: match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    },
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Content::Blocks(blocks) => {
                // Handle mixed content blocks
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
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
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_owned()),
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
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id.clone()),
                            });
                        }
                    }
                }

                // Add assistant message with text and/or tool calls
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    let role = match msg.role {
                        crate::llm::Role::User => ApiRole::User,
                        crate::llm::Role::Assistant => ApiRole::Assistant,
                    };

                    // Only add if it's an assistant message or has text content
                    if role == ApiRole::Assistant || !text_parts.is_empty() {
                        messages.push(ApiMessage {
                            role,
                            content: if text_parts.is_empty() {
                                None
                            } else {
                                Some(text_parts.join("\n"))
                            },
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls)
                            },
                            tool_call_id: None,
                        });
                    }
                }
            }
        }
    }

    messages
}

fn convert_tool(t: crate::llm::Tool) -> ApiTool {
    ApiTool {
        r#type: "function".to_owned(),
        function: ApiFunction {
            name: t.name,
            description: t.description,
            parameters: t.input_schema,
        },
    }
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
    tools: Option<&'a [ApiTool]>,
}

#[derive(Serialize)]
struct ApiChatRequestStreaming<'a> {
    model: &'a str,
    messages: &'a [ApiMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ApiTool]>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiMessage {
    role: ApiRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
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
    finish_reason: Option<ApiFinishReason>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiResponseToolCall>>,
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
#[serde(rename_all = "snake_case")]
enum ApiFinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
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
    choices: Vec<SseChoice>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

#[derive(Deserialize)]
struct SseChoice {
    delta: SseDelta,
    finish_reason: Option<SseFinishReason>,
}

#[derive(Deserialize)]
struct SseDelta {
    content: Option<String>,
    tool_calls: Option<Vec<SseToolCallDelta>>,
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
#[serde(rename_all = "snake_case")]
enum SseFinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
}

#[derive(Deserialize)]
struct SseUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ===================
    // Model Constants Tests
    // ===================

    #[test]
    fn test_model_constants_have_expected_values() {
        // GPT-5.2 series
        assert_eq!(MODEL_GPT52_INSTANT, "gpt-5.2-instant");
        assert_eq!(MODEL_GPT52_THINKING, "gpt-5.2-thinking");
        assert_eq!(MODEL_GPT52_PRO, "gpt-5.2-pro");
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
    fn test_api_finish_reason_deserialization() {
        let stop: ApiFinishReason = serde_json::from_str("\"stop\"").unwrap();
        let tool_calls: ApiFinishReason = serde_json::from_str("\"tool_calls\"").unwrap();
        let length: ApiFinishReason = serde_json::from_str("\"length\"").unwrap();
        let content_filter: ApiFinishReason = serde_json::from_str("\"content_filter\"").unwrap();

        assert!(matches!(stop, ApiFinishReason::Stop));
        assert!(matches!(tool_calls, ApiFinishReason::ToolCalls));
        assert!(matches!(length, ApiFinishReason::Length));
        assert!(matches!(content_filter, ApiFinishReason::ContentFilter));
    }

    // ===================
    // Message Conversion Tests
    // ===================

    #[test]
    fn test_build_api_messages_with_system() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![crate::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
            thinking: None,
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
            messages: vec![crate::llm::Message::user("Hello")],
            tools: None,
            max_tokens: 1024,
            thinking: None,
        };

        let api_messages = build_api_messages(&request);
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0].role, ApiRole::User);
    }

    #[test]
    fn test_convert_tool() {
        let tool = crate::llm::Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
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
        };

        let blocks = build_content_blocks(&message);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me help."));
        assert!(
            matches!(&blocks[1], ContentBlock::ToolUse { id, name, .. } if id == "call_123" && name == "read_file")
        );
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
        assert!(matches!(
            chunk.choices[0].finish_reason,
            Some(SseFinishReason::Stop)
        ));
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
    fn test_sse_finish_reason_deserialization() {
        let stop: SseFinishReason = serde_json::from_str("\"stop\"").unwrap();
        let tool_calls: SseFinishReason = serde_json::from_str("\"tool_calls\"").unwrap();
        let length: SseFinishReason = serde_json::from_str("\"length\"").unwrap();
        let content_filter: SseFinishReason = serde_json::from_str("\"content_filter\"").unwrap();

        assert!(matches!(stop, SseFinishReason::Stop));
        assert!(matches!(tool_calls, SseFinishReason::ToolCalls));
        assert!(matches!(length, SseFinishReason::Length));
        assert!(matches!(content_filter, SseFinishReason::ContentFilter));
    }

    #[test]
    fn test_streaming_request_serialization() {
        let messages = vec![ApiMessage {
            role: ApiRole::User,
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = ApiChatRequestStreaming {
            model: "gpt-4o",
            messages: &messages,
            max_completion_tokens: Some(1024),
            tools: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"model\":\"gpt-4o\""));
    }
}
