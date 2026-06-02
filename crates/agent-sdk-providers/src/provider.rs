//! LLM provider trait and streaming helpers.
//!
//! This module defines the [`LlmProvider`] trait that all LLM backends implement,
//! as well as the [`collect_stream`] helper for consuming a streaming response.

use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, ThinkingConfig, ThinkingMode, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use crate::model_capabilities::{
    ModelCapabilities, default_max_output_tokens, get_model_capabilities,
};
use crate::streaming::{StreamAccumulator, StreamBox, StreamDelta, StreamErrorKind};

/// How a provider satisfies a [`ResponseFormat`](agent_sdk_foundation::llm::ResponseFormat)
/// structured-output request.
///
/// The structured-output runner consults this to decide how to shape the
/// request and where to read the final structured value from the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputSupport {
    /// The provider applies the schema natively (JSON-mode /
    /// structured-outputs) when it sees `request.response_format`. The final
    /// structured value is the JSON in the assistant's text output.
    Native,
    /// The provider has no native JSON-schema mode. The runner injects a
    /// single forced "respond" tool whose `input_schema` is the output schema,
    /// and reads the structured value from that tool call's input.
    ToolForcing,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Non-streaming chat completion.
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome>;

    /// Streaming chat completion.
    ///
    /// Returns a stream of [`StreamDelta`] events. The default implementation
    /// calls [`chat()`](Self::chat) and converts the result to a single-chunk stream.
    ///
    /// Providers should override this method to provide true streaming support.
    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        Box::pin(async_stream::stream! {
            match self.chat(request).await {
                Ok(outcome) => match outcome {
                    ChatOutcome::Success(response) => {
                        // Emit content as deltas
                        for (idx, block) in response.content.iter().enumerate() {
                            match block {
                                ContentBlock::Text { text } => {
                                    yield Ok(StreamDelta::TextDelta {
                                        delta: text.clone(),
                                        block_index: idx,
                                    });
                                }
                                ContentBlock::Thinking { thinking, .. } => {
                                    yield Ok(StreamDelta::ThinkingDelta {
                                        delta: thinking.clone(),
                                        block_index: idx,
                                    });
                                }
                                ContentBlock::RedactedThinking { .. }
                                | ContentBlock::ToolResult { .. }
                                | ContentBlock::Image { .. }
                                | ContentBlock::Document { .. } => {
                                    // Not streamed in the default implementation
                                }
                                ContentBlock::ToolUse { id, name, input, thought_signature } => {
                                    yield Ok(StreamDelta::ToolUseStart {
                                        id: id.clone(),
                                        name: name.clone(),
                                        block_index: idx,
                                        thought_signature: thought_signature.clone(),
                                    });
                                    yield Ok(StreamDelta::ToolInputDelta {
                                        id: id.clone(),
                                        delta: serde_json::to_string(input).unwrap_or_default(),
                                        block_index: idx,
                                    });
                                }
                                // `ContentBlock` is `#[non_exhaustive]`; a future
                                // block kind we cannot stream is skipped rather than
                                // panicking the default fallback.
                                _ => {
                                    log::warn!(
                                        "chat_stream fallback skipping unrecognized content block at index {idx}"
                                    );
                                }
                            }
                        }
                        yield Ok(StreamDelta::Usage(response.usage));
                        yield Ok(StreamDelta::Done {
                            stop_reason: response.stop_reason,
                        });
                    }
                    ChatOutcome::RateLimited => {
                        yield Ok(StreamDelta::Error {
                            message: "Rate limited".to_string(),
                            kind: StreamErrorKind::RateLimited,
                        });
                    }
                    ChatOutcome::InvalidRequest(msg) => {
                        yield Ok(StreamDelta::Error {
                            message: msg,
                            kind: StreamErrorKind::InvalidRequest,
                        });
                    }
                    ChatOutcome::ServerError(msg) => {
                        yield Ok(StreamDelta::Error {
                            message: msg,
                            kind: StreamErrorKind::ServerError,
                        });
                    }
                    // `ChatOutcome` is `#[non_exhaustive]`; an outcome this SDK
                    // version does not model is surfaced as an unclassified
                    // (non-recoverable) stream error rather than dropped.
                    _ => {
                        yield Ok(StreamDelta::Error {
                            message: "Unrecognized chat outcome".to_string(),
                            kind: StreamErrorKind::Unknown,
                        });
                    }
                },
                Err(e) => yield Err(e),
            }
        })
    }

    fn model(&self) -> &str;
    fn provider(&self) -> &'static str;

    /// Provider-owned thinking configuration, if any.
    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        None
    }

    /// Canonical capability metadata for this provider/model, if known.
    fn capabilities(&self) -> Option<&'static ModelCapabilities> {
        get_model_capabilities(self.provider(), self.model()).or_else(|| match self.provider() {
            "openai-responses" | "openai-codex" => get_model_capabilities("openai", self.model()),
            "vertex" if self.model().starts_with("claude-") => {
                get_model_capabilities("anthropic", self.model())
            }
            "vertex" => get_model_capabilities("gemini", self.model()),
            _ => None,
        })
    }

    /// Validate a thinking configuration against the provider/model capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested thinking mode is not supported by
    /// the active provider/model capability set.
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

        Ok(())
    }

    /// Resolve the effective thinking configuration for a request.
    ///
    /// Request-level thinking overrides provider-owned defaults when present.
    ///
    /// # Errors
    ///
    /// Returns an error when the resolved thinking configuration is not
    /// supported by the active provider/model capability set.
    fn resolve_thinking_config(
        &self,
        request_thinking: Option<&ThinkingConfig>,
    ) -> Result<Option<ThinkingConfig>> {
        let thinking = request_thinking.or_else(|| self.configured_thinking());
        self.validate_thinking_config(thinking)?;
        Ok(thinking.cloned())
    }

    /// Default maximum output tokens for this provider/model when the caller
    /// does not explicitly override `AgentConfig.max_tokens`.
    fn default_max_tokens(&self) -> u32 {
        self.capabilities()
            .and_then(|caps| caps.max_output_tokens)
            .or_else(|| default_max_output_tokens(self.provider(), self.model()))
            .unwrap_or(4096)
    }

    /// How this provider satisfies a structured-output
    /// ([`ResponseFormat`](agent_sdk_foundation::llm::ResponseFormat)) request.
    ///
    /// Providers with a native JSON-schema / JSON-mode wire field
    /// (OpenAI-family, Gemini, Vertex) report
    /// [`StructuredOutputSupport::Native`] and consume
    /// `request.response_format` directly. Providers without one (Anthropic)
    /// report [`StructuredOutputSupport::ToolForcing`] so the runner forces a
    /// single "respond" tool whose schema is the output schema. The default
    /// is the conservative [`StructuredOutputSupport::ToolForcing`], which
    /// works for any tool-capable provider.
    fn structured_output_support(&self) -> StructuredOutputSupport {
        match self.provider() {
            "openai" | "openai-responses" | "openai-codex" | "gemini" => {
                StructuredOutputSupport::Native
            }
            // Vertex multiplexes Anthropic and Gemini models. Only the Gemini
            // side has a native structured-output field; Claude-on-Vertex uses
            // the Messages API shape, which has no `response_format`.
            "vertex" if !self.model().starts_with("claude-") => StructuredOutputSupport::Native,
            _ => StructuredOutputSupport::ToolForcing,
        }
    }
}

/// Helper function to consume a stream and collect it into a `ChatResponse`.
///
/// This is useful for providers that want to test their streaming implementation
/// or for cases where you need the full response after streaming.
///
/// # Errors
///
/// Returns an error if the stream yields an error result.
pub async fn collect_stream(mut stream: StreamBox<'_>, model: String) -> Result<ChatOutcome> {
    let mut accumulator = StreamAccumulator::new();
    let mut last_error: Option<(String, StreamErrorKind)> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(delta) => {
                if let StreamDelta::Error { message, kind } = &delta {
                    last_error = Some((message.clone(), *kind));
                }
                accumulator.apply(&delta);
            }
            Err(e) => return Err(e),
        }
    }

    // If we encountered an error during streaming, map kind directly
    // to the corresponding `ChatOutcome` variant.  No string-matching
    // heuristic is needed because the kind already records the
    // category at the construction site.
    if let Some((message, kind)) = last_error {
        return Ok(match kind {
            StreamErrorKind::RateLimited => ChatOutcome::RateLimited,
            StreamErrorKind::InvalidRequest => ChatOutcome::InvalidRequest(message),
            // `StreamErrorKind::ServerError`, plus the `#[non_exhaustive]`
            // catch-all (`Unknown` / future kinds): an unclassified error is
            // treated as a (non-recoverable) server error so the caller still
            // surfaces the failure rather than silently succeeding.
            _ => ChatOutcome::ServerError(message),
        });
    }

    // Extract usage and stop_reason before consuming the accumulator
    let usage = accumulator.take_usage().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
    });
    let stop_reason = accumulator.take_stop_reason();
    let content = accumulator.into_content_blocks();

    // Log accumulated response for debugging
    log::debug!(
        "Collected stream response: model={} stop_reason={:?} usage={{input_tokens={}, output_tokens={}}} content_blocks={}",
        model,
        stop_reason,
        usage.input_tokens,
        usage.output_tokens,
        content.len()
    );
    for (i, block) in content.iter().enumerate() {
        match block {
            ContentBlock::Text { text } => {
                log::debug!("  content_block[{}]: Text (len={})", i, text.len());
            }
            ContentBlock::Thinking { thinking, .. } => {
                log::debug!("  content_block[{}]: Thinking (len={})", i, thinking.len());
            }
            ContentBlock::RedactedThinking { .. } => {
                log::debug!("  content_block[{i}]: RedactedThinking");
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                log::debug!("  content_block[{i}]: ToolUse id={id} name={name} input={input}");
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content: result_content,
                is_error,
            } => {
                log::debug!(
                    "  content_block[{}]: ToolResult tool_use_id={} is_error={:?} content_len={}",
                    i,
                    tool_use_id,
                    is_error,
                    result_content.len()
                );
            }
            ContentBlock::Image { source } => {
                log::debug!(
                    "  content_block[{i}]: Image media_type={}",
                    source.media_type
                );
            }
            ContentBlock::Document { source } => {
                log::debug!(
                    "  content_block[{i}]: Document media_type={}",
                    source.media_type
                );
            }
            // `ContentBlock` is `#[non_exhaustive]`; log unknown future block
            // kinds generically so the debug dump stays exhaustive.
            _ => {
                log::debug!("  content_block[{i}]: <unrecognized block kind>");
            }
        }
    }

    Ok(ChatOutcome::Success(ChatResponse {
        id: String::new(),
        content,
        model,
        stop_reason,
        usage,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;

    struct Stub {
        provider: &'static str,
        model: &'static str,
    }

    #[async_trait]
    impl LlmProvider for Stub {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::ServerError("unused".to_owned()))
        }

        fn model(&self) -> &str {
            self.model
        }

        fn provider(&self) -> &'static str {
            self.provider
        }
    }

    fn support_for(provider: &'static str, model: &'static str) -> StructuredOutputSupport {
        Stub { provider, model }.structured_output_support()
    }

    #[test]
    fn native_providers_report_native_support() {
        for provider in ["openai", "openai-responses", "openai-codex", "gemini"] {
            assert_eq!(
                support_for(provider, "any-model"),
                StructuredOutputSupport::Native,
                "{provider} should be native"
            );
        }
    }

    #[test]
    fn anthropic_reports_tool_forcing() {
        assert_eq!(
            support_for("anthropic", "claude-sonnet-4-5"),
            StructuredOutputSupport::ToolForcing
        );
    }

    #[test]
    fn vertex_is_native_for_gemini_models_and_tool_forcing_for_claude() {
        assert_eq!(
            support_for("vertex", "gemini-3-flash-preview"),
            StructuredOutputSupport::Native
        );
        assert_eq!(
            support_for("vertex", "claude-sonnet-4-5"),
            StructuredOutputSupport::ToolForcing
        );
    }

    #[test]
    fn unknown_provider_defaults_to_tool_forcing() {
        assert_eq!(
            support_for("some-new-provider", "x"),
            StructuredOutputSupport::ToolForcing
        );
    }
}
