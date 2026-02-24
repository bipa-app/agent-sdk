//! Google Vertex AI provider implementation.
//!
//! This module provides an implementation of `LlmProvider` for the Google Vertex AI
//! platform, which uses the same Gemini API format but with different URL construction
//! and `OAuth2` Bearer authentication.

use crate::llm::{
    ChatOutcome, ChatRequest, ChatResponse, LlmProvider, StreamBox, StreamDelta, Usage,
};
use crate::providers::gemini::data::{
    ApiContent, ApiGenerateContentRequest, ApiGenerateContentResponse, ApiGenerationConfig,
    ApiPart, ApiUsageMetadata, build_api_contents, build_content_blocks, convert_tools_to_config,
    map_finish_reason, map_thinking_config, stream_gemini_response,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::StatusCode;

pub const MODEL_GEMINI_3_FLASH: &str = "gemini-3.0-flash";
pub const MODEL_GEMINI_3_PRO: &str = "gemini-3.0-pro";

/// Google Vertex AI LLM provider.
///
/// Uses the same Gemini request/response format as `GeminiProvider` but
/// authenticates via `OAuth2` Bearer tokens and routes through the Vertex AI
/// regional endpoint.
#[derive(Clone)]
pub struct VertexProvider {
    client: reqwest::Client,
    access_token: String,
    project_id: String,
    region: String,
    model: String,
}

impl VertexProvider {
    /// Create a new Vertex AI provider with full control over all parameters.
    #[must_use]
    pub fn new(access_token: String, project_id: String, region: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            access_token,
            project_id,
            region,
            model,
        }
    }

    /// Create a provider using Gemini 3.0 Flash on Vertex AI.
    #[must_use]
    pub fn flash(access_token: String, project_id: String, region: String) -> Self {
        Self::new(
            access_token,
            project_id,
            region,
            MODEL_GEMINI_3_FLASH.to_owned(),
        )
    }

    /// Create a provider using Gemini 3.0 Pro on Vertex AI.
    #[must_use]
    pub fn pro(access_token: String, project_id: String, region: String) -> Self {
        Self::new(
            access_token,
            project_id,
            region,
            MODEL_GEMINI_3_PRO.to_owned(),
        )
    }

    /// Build the base URL for this provider's model endpoint.
    ///
    /// For the `global` location the domain is `aiplatform.googleapis.com`
    /// (no region prefix). Regional locations use `{region}-aiplatform.googleapis.com`.
    fn base_url(&self) -> String {
        let domain = if self.region == "global" {
            "aiplatform.googleapis.com".to_owned()
        } else {
            format!("{}-aiplatform.googleapis.com", self.region)
        };
        format!(
            "https://{domain}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}",
            domain = domain,
            region = self.region,
            project = self.project_id,
            model = self.model,
        )
    }
}

#[async_trait]
impl LlmProvider for VertexProvider {
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
            "Vertex AI LLM request model={} max_tokens={}",
            self.model,
            request.max_tokens
        );

        let url = format!("{}:generateContent", self.base_url());

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.access_token)
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
            "Vertex AI LLM response status={} body_len={}",
            status,
            bytes.len()
        );

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Ok(ChatOutcome::RateLimited);
        }

        if status.is_server_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::error!("Vertex AI server error status={status} body={body}");
            return Ok(ChatOutcome::ServerError(body.into_owned()));
        }

        if status.is_client_error() {
            let body = String::from_utf8_lossy(&bytes);
            log::warn!("Vertex AI client error status={status} body={body}");
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
                "Vertex AI parts not converted to content blocks raw_parts={:?}",
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
                "Vertex AI streaming LLM request model={} max_tokens={}",
                self.model,
                request.max_tokens
            );

            let url = format!("{}:streamGenerateContent?alt=sse", self.base_url());

            let Ok(response) = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .bearer_auth(&self.access_token)
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
                log::warn!("Vertex AI error status={status} body={body}");
                yield Ok(StreamDelta::Error {
                    message: body,
                    recoverable,
                });
                return;
            }

            let mut inner = stream_gemini_response(response);
            while let Some(item) = futures::StreamExt::next(&mut inner).await {
                yield item;
            }
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "vertex"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_provider() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "custom-model".to_string(),
        );

        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_flash_factory() {
        let provider = VertexProvider::flash(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
        );

        assert_eq!(provider.model(), MODEL_GEMINI_3_FLASH);
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_pro_factory() {
        let provider = VertexProvider::pro(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
        );

        assert_eq!(provider.model(), MODEL_GEMINI_3_PRO);
        assert_eq!(provider.provider(), "vertex");
    }

    #[test]
    fn test_provider_is_cloneable() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "test-model".to_string(),
        );
        let cloned = provider.clone();

        assert_eq!(provider.model(), cloned.model());
        assert_eq!(provider.provider(), cloned.provider());
    }

    #[test]
    fn test_base_url_construction() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
            "gemini-3.0-flash".to_string(),
        );

        let url = provider.base_url();
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.0-flash"
        );
    }

    #[test]
    fn test_base_url_with_different_region() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "other-project".to_string(),
            "europe-west4".to_string(),
            "gemini-3.0-pro".to_string(),
        );

        let url = provider.base_url();
        assert!(url.starts_with("https://europe-west4-aiplatform.googleapis.com/"));
        assert!(url.contains("/projects/other-project/"));
        assert!(url.contains("/locations/europe-west4/"));
        assert!(url.ends_with("/models/gemini-3.0-pro"));
    }

    #[test]
    fn test_base_url_global_region_has_no_prefix() {
        let provider = VertexProvider::new(
            "token".to_string(),
            "bipa-278720".to_string(),
            "global".to_string(),
            "gemini-3.1-pro-preview".to_string(),
        );

        let url = provider.base_url();
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/bipa-278720/locations/global/publishers/google/models/gemini-3.1-pro-preview"
        );
    }

    #[test]
    fn test_model_constants() {
        assert_eq!(MODEL_GEMINI_3_FLASH, "gemini-3.0-flash");
        assert_eq!(MODEL_GEMINI_3_PRO, "gemini-3.0-pro");
    }
}
