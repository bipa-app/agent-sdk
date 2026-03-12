pub mod attachments;
pub mod router;
pub mod streaming;
pub mod types;

pub use router::{ModelRouter, ModelTier, TaskComplexity};
pub use streaming::{StreamAccumulator, StreamBox, StreamDelta};
pub use types::*;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use crate::model_capabilities::{
    ModelCapabilities, default_max_output_tokens, get_model_capabilities,
};

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
                            recoverable: true,
                        });
                    }
                    ChatOutcome::InvalidRequest(msg) => {
                        yield Ok(StreamDelta::Error {
                            message: msg,
                            recoverable: false,
                        });
                    }
                    ChatOutcome::ServerError(msg) => {
                        yield Ok(StreamDelta::Error {
                            message: msg,
                            recoverable: true,
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
    let mut last_error: Option<(String, bool)> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(delta) => {
                if let StreamDelta::Error {
                    message,
                    recoverable,
                } = &delta
                {
                    last_error = Some((message.clone(), *recoverable));
                }
                accumulator.apply(&delta);
            }
            Err(e) => return Err(e),
        }
    }

    // If we encountered an error during streaming, return it
    if let Some((message, recoverable)) = last_error {
        if !recoverable {
            return Ok(ChatOutcome::InvalidRequest(message));
        }
        // Check if it was a rate limit
        if message.contains("Rate limited") || message.contains("rate limit") {
            return Ok(ChatOutcome::RateLimited);
        }
        return Ok(ChatOutcome::ServerError(message));
    }

    // Extract usage and stop_reason before consuming the accumulator
    let usage = accumulator.take_usage().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
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
