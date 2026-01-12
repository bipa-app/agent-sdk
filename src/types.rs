//! Core types for the agent SDK.
//!
//! This module contains the fundamental types used throughout the SDK:
//!
//! - [`ThreadId`]: Unique identifier for conversation threads
//! - [`AgentConfig`]: Configuration for the agent loop
//! - [`TokenUsage`]: Token consumption statistics
//! - [`ToolResult`]: Result returned from tool execution
//! - [`ToolTier`]: Permission tiers for tools
//! - [`PendingAction`]: Actions awaiting user confirmation
//! - [`AgentState`]: Checkpointable agent state

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use time::OffsetDateTime;
use uuid::Uuid;

/// Unique identifier for a conversation thread
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub String);

impl ThreadId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl Default for ThreadId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Configuration for the agent loop
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Maximum number of turns (LLM round-trips) before stopping
    pub max_turns: usize,
    /// Maximum tokens per response
    pub max_tokens: u32,
    /// System prompt for the agent
    pub system_prompt: String,
    /// Model identifier
    pub model: String,
    /// Retry configuration for transient errors
    pub retry: RetryConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tokens: 4096,
            system_prompt: String::new(),
            model: String::from("claude-sonnet-4-20250514"),
            retry: RetryConfig::default(),
        }
    }
}

/// Configuration for retry behavior on transient errors.
#[derive(Clone, Debug)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay_ms: 1000,
            max_delay_ms: 120_000,
        }
    }
}

impl RetryConfig {
    /// Create a retry config with no retries (for testing)
    #[must_use]
    pub const fn no_retry() -> Self {
        Self {
            max_retries: 0,
            base_delay_ms: 0,
            max_delay_ms: 0,
        }
    }

    /// Create a retry config with fast retries (for testing)
    #[must_use]
    pub const fn fast() -> Self {
        Self {
            max_retries: 5,
            base_delay_ms: 10,
            max_delay_ms: 100,
        }
    }
}

/// Token usage statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    pub const fn add(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// Result of a tool execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool execution succeeded
    pub success: bool,
    /// Output content (displayed to user and fed back to LLM)
    pub output: String,
    /// Optional structured data
    pub data: Option<serde_json::Value>,
    /// Duration of the tool execution in milliseconds
    pub duration_ms: Option<u64>,
}

impl ToolResult {
    #[must_use]
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            data: None,
            duration_ms: None,
        }
    }

    #[must_use]
    pub fn success_with_data(output: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            output: output.into(),
            data: Some(data),
            duration_ms: None,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            output: message.into(),
            data: None,
            duration_ms: None,
        }
    }

    #[must_use]
    pub const fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Permission tier for tools
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolTier {
    /// Tier 0: Read-only, always allowed (e.g., `get_balance`)
    Observe,
    /// Tier 1: Requires confirmation before execution
    Confirm,
    /// Tier 3: Requires PIN verification (e.g., `send_pix`, `buy_btc`)
    RequiresPin,
}

/// State of a pending action that requires confirmation or PIN
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingAction {
    pub id: String,
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub tier: ToolTier,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub expires_at: OffsetDateTime,
}

/// Snapshot of agent state for checkpointing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub thread_id: ThreadId,
    pub turn_count: usize,
    pub total_usage: TokenUsage,
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

impl AgentState {
    #[must_use]
    pub fn new(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            turn_count: 0,
            total_usage: TokenUsage::default(),
            metadata: HashMap::new(),
            created_at: OffsetDateTime::now_utc(),
        }
    }
}
