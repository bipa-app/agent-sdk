//! Google Gemini API provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Google Gemini
//! API (`generativelanguage.googleapis.com`).

pub(crate) mod data;

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamBox, StreamDelta, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use data::{
    ApiContent, ApiGenerateContentRequest, ApiGenerateContentResponse, ApiGenerationConfig,
    ApiPart, ApiUsageMetadata, build_api_contents, build_content_blocks, convert_tools_to_config,
    map_finish_reason, map_thinking_config,
};
use reqwest::StatusCode;

const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// Gemini 3 series (latest, Dec 2025)
pub const MODEL_GEMINI_3_FLASH: &str = "gemini-3.0-flash";
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
}

impl GeminiProvider {
    /// Create a new Gemini provider with the specified API key and model.
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Create a provider using Gemini 3.0 Flash (fast and capable, current default).
    #[must_use]
    pub fn flash(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_3_FLASH.to_owned())
    }

    /// Create a provider using Gemini 2.0 Flash Lite (fastest, most cost-effective).
    #[must_use]
    pub fn flash_lite(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_2_FLASH_LITE.to_owned())
    }

    /// Create a provider using Gemini 3.0 Pro (most capable).
    #[must_use]
    pub fn pro(api_key: String) -> Self {
        Self::new(api_key, MODEL_GEMINI_3_PRO.to_owned())
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let contents = build_api_contents(&request.messages);
        let tools = request.tools.map(convert_tools_to_config);
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

        let thinking_config = request.thinking.as_ref().map(map_thinking_config);

        let api_request = ApiGenerateContentRequest {
            contents: &contents,
            system_instruction: system_instruction.as_ref(),
            tools: tools.as_ref().map(std::slice::from_ref),
            generation_config: Some(ApiGenerationConfig {
                max_output_tokens: Some(request.max_tokens),
                thinking_config,
            }),
        };

        log::debug!(
            "Gemini LLM request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let response = self
            .client
            .post(format!(
                "{API_BASE_URL}/models/{}:generateContent",
                self.model
            ))
            .header("Content-Type", "application/json")
            .query(&[("key", &self.api_key)])
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
            "Gemini LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
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
            .any(|b| matches!(b, crate::llm::ContentBlock::ToolUse { .. }));

        let stop_reason = candidate
            .finish_reason
            .as_ref()
            .map(|r| map_finish_reason(r, has_tool_calls));

        let usage = api_response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
        });

        Ok(ChatOutcome::Success(ChatResponse {
            id: String::new(),
            content,
            model: self.model.clone(),
            stop_reason,
            usage: Usage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count,
            },
        }))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            let contents = build_api_contents(&request.messages);
            let tools = request.tools.map(convert_tools_to_config);
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

            let thinking_config = request.thinking.as_ref().map(map_thinking_config);

            let api_request = ApiGenerateContentRequest {
                contents: &contents,
                system_instruction: system_instruction.as_ref(),
                tools: tools.as_ref().map(std::slice::from_ref),
                generation_config: Some(ApiGenerationConfig {
                    max_output_tokens: Some(request.max_tokens),
                    thinking_config,
                }),
            };

            log::debug!(
                "Gemini streaming LLM request model={} max_tokens={}",
                self.model,
                request.max_tokens
            );

            let Ok(response) = self
                .client
                .post(format!(
                    "{API_BASE_URL}/models/{}:streamGenerateContent",
                    self.model
                ))
                .header("Content-Type", "application/json")
                .query(&[("key", &self.api_key), ("alt", &"sse".to_string())])
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
                let recoverable =
                    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                log::warn!("Gemini error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    recoverable,
                });
                return;
            }

            let mut inner = data::stream_gemini_response(response);
            while let Some(item) = futures::StreamExt::next(&mut inner).await {
                yield item;
            }
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "gemini"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_pro_factory_creates_pro_provider() {
        let provider = GeminiProvider::pro("test-api-key".to_string());

        assert_eq!(provider.model(), MODEL_GEMINI_3_PRO);
        assert_eq!(provider.provider(), "gemini");
    }

    #[test]
    fn test_model_constants_have_expected_values() {
        assert_eq!(MODEL_GEMINI_3_FLASH, "gemini-3.0-flash");
        assert_eq!(MODEL_GEMINI_3_PRO, "gemini-3.0-pro");
        assert_eq!(MODEL_GEMINI_25_FLASH, "gemini-2.5-flash");
        assert_eq!(MODEL_GEMINI_25_PRO, "gemini-2.5-pro");
        assert_eq!(MODEL_GEMINI_2_FLASH, "gemini-2.0-flash");
        assert_eq!(MODEL_GEMINI_2_FLASH_LITE, "gemini-2.0-flash-lite");
    }

    #[test]
    fn test_provider_is_cloneable() {
        let provider = GeminiProvider::new("test-api-key".to_string(), "test-model".to_string());
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }
}
