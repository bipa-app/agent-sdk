pub mod router;
pub mod streaming;
pub mod types;

pub use router::{ModelRouter, ModelTier, TaskComplexity};
pub use streaming::{StreamAccumulator, StreamBox, StreamDelta};
pub use types::*;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

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
                                | ContentBlock::ToolResult { .. } => {
                                    // Not streamed in the default implementation
                                }
                                ContentBlock::ToolUse { id, name, input, .. } => {
                                    yield Ok(StreamDelta::ToolUseStart {
                                        id: id.clone(),
                                        name: name.clone(),
                                        block_index: idx,
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
