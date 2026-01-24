//! Agent events for real-time streaming.
//!
//! The [`AgentEvent`] enum represents all events that can occur during agent
//! execution. These events are streamed via an async channel for real-time
//! UI updates and logging.
//!
//! # Event Flow
//!
//! A typical event sequence looks like:
//! 1. `Start` - Agent begins processing
//! 2. `Text` / `ToolCallStart` / `ToolCallEnd` - Processing events
//! 3. `TurnComplete` - One LLM round-trip finished
//! 4. `Done` - Agent completed successfully, or `Error` if failed

use crate::types::{ThreadId, TokenUsage, ToolResult, ToolTier};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Events emitted by the agent loop during execution.
/// These are streamed to the client for real-time UI updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent loop has started
    Start { thread_id: ThreadId, turn: usize },

    /// Agent is "thinking" - streaming text that may be shown as typing indicator
    Thinking { text: String },

    /// A text delta for streaming responses
    TextDelta { delta: String },

    /// Complete text block from the agent
    Text { text: String },

    /// Agent is about to call a tool
    ToolCallStart {
        id: String,
        name: String,
        display_name: String,
        input: serde_json::Value,
        tier: ToolTier,
    },

    /// Tool execution completed
    ToolCallEnd {
        id: String,
        name: String,
        display_name: String,
        result: ToolResult,
    },

    /// Tool requires confirmation before execution.
    /// The application determines the confirmation type (normal, PIN, biometric).
    ToolRequiresConfirmation {
        id: String,
        name: String,
        input: serde_json::Value,
        description: String,
    },

    /// Agent turn completed (one LLM round-trip)
    TurnComplete { turn: usize, usage: TokenUsage },

    /// Agent loop completed successfully
    Done {
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
    },

    /// An error occurred during execution
    Error { message: String, recoverable: bool },

    /// Context was compacted to reduce size
    ContextCompacted {
        /// Number of messages before compaction
        original_count: usize,
        /// Number of messages after compaction
        new_count: usize,
        /// Estimated tokens before compaction
        original_tokens: usize,
        /// Estimated tokens after compaction
        new_tokens: usize,
    },

    /// Progress update from a running subagent
    SubagentProgress {
        /// ID of the parent tool call that spawned this subagent
        subagent_id: String,
        /// Name of the subagent (e.g., "explore", "plan")
        subagent_name: String,
        /// Tool name that just started or completed
        tool_name: String,
        /// Brief context for the tool (e.g., file path, pattern)
        tool_context: String,
        /// Whether the tool completed (false = started, true = ended)
        completed: bool,
        /// Whether the tool succeeded (only meaningful if completed)
        success: bool,
        /// Current total tool count for this subagent
        tool_count: u32,
        /// Current total tokens used by this subagent
        total_tokens: u64,
    },
}

impl AgentEvent {
    #[must_use]
    pub const fn start(thread_id: ThreadId, turn: usize) -> Self {
        Self::Start { thread_id, turn }
    }

    #[must_use]
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking { text: text.into() }
    }

    #[must_use]
    pub fn text_delta(delta: impl Into<String>) -> Self {
        Self::TextDelta {
            delta: delta.into(),
        }
    }

    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    #[must_use]
    pub fn tool_call_start(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        input: serde_json::Value,
        tier: ToolTier,
    ) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            input,
            tier,
        }
    }

    #[must_use]
    pub fn tool_call_end(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        result: ToolResult,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            result,
        }
    }

    #[must_use]
    pub const fn done(
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
    ) -> Self {
        Self::Done {
            thread_id,
            total_turns,
            total_usage,
            duration,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>, recoverable: bool) -> Self {
        Self::Error {
            message: message.into(),
            recoverable,
        }
    }

    #[must_use]
    pub const fn context_compacted(
        original_count: usize,
        new_count: usize,
        original_tokens: usize,
        new_tokens: usize,
    ) -> Self {
        Self::ContextCompacted {
            original_count,
            new_count,
            original_tokens,
            new_tokens,
        }
    }
}
