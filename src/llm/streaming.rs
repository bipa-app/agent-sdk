//! Streaming types for LLM responses.
//!
//! This module provides types for handling streaming responses from LLM providers.
//! The [`StreamDelta`] enum represents individual events in a streaming response,
//! and [`StreamAccumulator`] helps collect these events into a final response.

use crate::llm::{ContentBlock, StopReason, Usage};
use futures::Stream;
use std::pin::Pin;

/// Events yielded during streaming LLM responses.
///
/// Each variant represents a different type of event that can occur
/// during a streaming response from an LLM provider.
#[derive(Debug, Clone)]
pub enum StreamDelta {
    /// A text delta for streaming text content.
    TextDelta {
        /// The text fragment to append
        delta: String,
        /// Index of the content block being streamed
        block_index: usize,
    },

    /// Start of a tool use block (name and id are known).
    ToolUseStart {
        /// Unique identifier for this tool call
        id: String,
        /// Name of the tool being called
        name: String,
        /// Index of the content block
        block_index: usize,
    },

    /// Incremental JSON for tool input (partial/incomplete JSON).
    ToolInputDelta {
        /// Tool call ID this delta belongs to
        id: String,
        /// JSON fragment to append
        delta: String,
        /// Index of the content block
        block_index: usize,
    },

    /// Usage information (typically at stream end).
    Usage(Usage),

    /// Stream completed with stop reason.
    Done {
        /// Why the stream ended
        stop_reason: Option<StopReason>,
    },

    /// Error during streaming.
    Error {
        /// Error message
        message: String,
        /// Whether the error is recoverable (e.g., rate limit)
        recoverable: bool,
    },
}

/// Type alias for a boxed stream of stream deltas.
pub type StreamBox<'a> = Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send + 'a>>;

/// Helper to accumulate streamed content into a final response.
///
/// This struct collects [`StreamDelta`] events and can convert them
/// into the final content blocks once the stream is complete.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    /// Accumulated text for each block index
    text_blocks: Vec<String>,
    /// Accumulated tool use calls
    tool_uses: Vec<ToolUseAccumulator>,
    /// Usage information from the stream
    usage: Option<Usage>,
    /// Stop reason from the stream
    stop_reason: Option<StopReason>,
}

/// Accumulator for a single tool use during streaming.
#[derive(Debug, Default)]
pub struct ToolUseAccumulator {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Accumulated JSON input (may be incomplete during streaming)
    pub input_json: String,
    /// Block index for ordering
    pub block_index: usize,
}

impl StreamAccumulator {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a stream delta to the accumulator.
    pub fn apply(&mut self, delta: &StreamDelta) {
        match delta {
            StreamDelta::TextDelta { delta, block_index } => {
                while self.text_blocks.len() <= *block_index {
                    self.text_blocks.push(String::new());
                }
                self.text_blocks[*block_index].push_str(delta);
            }
            StreamDelta::ToolUseStart {
                id,
                name,
                block_index,
            } => {
                self.tool_uses.push(ToolUseAccumulator {
                    id: id.clone(),
                    name: name.clone(),
                    input_json: String::new(),
                    block_index: *block_index,
                });
            }
            StreamDelta::ToolInputDelta { id, delta, .. } => {
                if let Some(tool) = self.tool_uses.iter_mut().find(|t| t.id == *id) {
                    tool.input_json.push_str(delta);
                }
            }
            StreamDelta::Usage(u) => {
                self.usage = Some(u.clone());
            }
            StreamDelta::Done { stop_reason } => {
                self.stop_reason = *stop_reason;
            }
            StreamDelta::Error { .. } => {}
        }
    }

    /// Get the accumulated usage information.
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Get the stop reason.
    #[must_use]
    pub const fn stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Convert accumulated content to `ContentBlock`s.
    ///
    /// This consumes the accumulator and returns the final content blocks.
    /// Tool use JSON is parsed at this point; invalid JSON results in a null input.
    #[must_use]
    pub fn into_content_blocks(self) -> Vec<ContentBlock> {
        let mut blocks: Vec<(usize, ContentBlock)> = Vec::new();

        // Add text blocks with their indices
        for (idx, text) in self.text_blocks.into_iter().enumerate() {
            if !text.is_empty() {
                blocks.push((idx, ContentBlock::Text { text }));
            }
        }

        // Add tool uses with their indices
        for tool in self.tool_uses {
            let input: serde_json::Value =
                serde_json::from_str(&tool.input_json).unwrap_or(serde_json::Value::Null);
            blocks.push((
                tool.block_index,
                ContentBlock::ToolUse {
                    id: tool.id,
                    name: tool.name,
                    input,
                },
            ));
        }

        // Sort by block index to maintain order
        blocks.sort_by_key(|(idx, _)| *idx);

        blocks.into_iter().map(|(_, block)| block).collect()
    }

    /// Take ownership of accumulated usage.
    pub const fn take_usage(&mut self) -> Option<Usage> {
        self.usage.take()
    }

    /// Take ownership of stop reason.
    pub const fn take_stop_reason(&mut self) -> Option<StopReason> {
        self.stop_reason.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_text_deltas() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "Hello".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: " world".to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello world"));
    }

    #[test]
    fn test_accumulator_multiple_text_blocks() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "First".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: "Second".to_string(),
            block_index: 1,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "First"));
        assert!(matches!(&blocks[1], ContentBlock::Text { text } if text == "Second"));
    }

    #[test]
    fn test_accumulator_tool_use() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_123".to_string(),
            delta: r#"{"path":"#.to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_123".to_string(),
            delta: r#""test.txt"}"#.to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "test.txt");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_mixed_content() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: "Let me read that file.".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_456".to_string(),
            name: "read_file".to_string(),
            block_index: 1,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_456".to_string(),
            delta: r#"{"path":"file.txt"}"#.to_string(),
            block_index: 1,
        });
        acc.apply(&StreamDelta::Usage(Usage {
            input_tokens: 100,
            output_tokens: 50,
        }));
        acc.apply(&StreamDelta::Done {
            stop_reason: Some(StopReason::ToolUse),
        });

        assert!(acc.usage().is_some());
        assert_eq!(acc.usage().map(|u| u.input_tokens), Some(100));
        assert!(matches!(acc.stop_reason(), Some(StopReason::ToolUse)));

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { .. }));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_accumulator_invalid_tool_json() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::ToolUseStart {
            id: "call_789".to_string(),
            name: "test_tool".to_string(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "call_789".to_string(),
            delta: "invalid json {".to_string(),
            block_index: 0,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert!(input.is_null());
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulator_empty() {
        let acc = StreamAccumulator::new();
        let blocks = acc.into_content_blocks();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_accumulator_skips_empty_text() {
        let mut acc = StreamAccumulator::new();

        acc.apply(&StreamDelta::TextDelta {
            delta: String::new(),
            block_index: 0,
        });
        acc.apply(&StreamDelta::TextDelta {
            delta: "Hello".to_string(),
            block_index: 1,
        });

        let blocks = acc.into_content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello"));
    }
}
