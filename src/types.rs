//! Core types for the agent SDK.
//!
//! This module contains the fundamental types used throughout the SDK:
//!
//! - [`ThreadId`]: Unique identifier for conversation threads
//! - [`AgentConfig`]: Configuration for the agent loop
//! - [`TokenUsage`]: Token consumption statistics
//! - [`ToolResult`]: Result returned from tool execution
//! - [`ToolTier`]: Permission tiers for tools
//! - [`AgentRunState`]: Outcome of running the agent loop (looping mode)
//! - [`TurnOutcome`]: Outcome of running a single turn (single-turn mode)
//! - [`AgentInput`]: Input to start or resume an agent run
//! - [`AgentContinuation`]: Opaque state for resuming after confirmation
//! - [`AgentState`]: Checkpointable agent state

use crate::llm::ThinkingConfig;
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
    /// Optional extended thinking configuration
    pub thinking: Option<ThinkingConfig>,
    /// Enable streaming responses from the LLM.
    ///
    /// When `true`, emits `TextDelta` and `Thinking` events as text arrives
    /// in real-time. When `false` (default), waits for the complete response
    /// before emitting `Text` and `Thinking` events.
    pub streaming: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tokens: 4096,
            system_prompt: String::new(),
            model: String::from("claude-sonnet-4-5-20250929"),
            retry: RetryConfig::default(),
            thinking: None,
            streaming: false,
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
    /// Read-only, always allowed (e.g., `get_balance`)
    Observe,
    /// Requires confirmation before execution.
    /// The application determines the confirmation type (normal, PIN, biometric).
    Confirm,
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

/// Error from the agent loop.
#[derive(Debug, Clone)]
pub struct AgentError {
    /// Error message
    pub message: String,
    /// Whether the error is potentially recoverable
    pub recoverable: bool,
}

impl AgentError {
    #[must_use]
    pub fn new(message: impl Into<String>, recoverable: bool) -> Self {
        Self {
            message: message.into(),
            recoverable,
        }
    }
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AgentError {}

/// Outcome of running the agent loop.
#[derive(Debug)]
pub enum AgentRunState {
    /// Agent completed successfully.
    Done {
        total_turns: u32,
        input_tokens: u64,
        output_tokens: u64,
    },

    /// Agent was refused by the model (safety/policy).
    Refusal {
        total_turns: u32,
        input_tokens: u64,
        output_tokens: u64,
    },

    /// Agent encountered an error.
    Error(AgentError),

    /// Agent is awaiting confirmation for a tool call.
    /// The application should present this to the user and call resume.
    AwaitingConfirmation {
        /// ID of the pending tool call (from LLM)
        tool_call_id: String,
        /// Tool name string (for LLM protocol)
        tool_name: String,
        /// Human-readable display name
        display_name: String,
        /// Tool input parameters
        input: serde_json::Value,
        /// Description of what confirmation is needed
        description: String,
        /// Continuation state for resuming (boxed for enum size efficiency)
        continuation: Box<AgentContinuation>,
    },
}

/// Information about a pending tool call that was extracted from the LLM response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingToolCallInfo {
    /// Unique ID for this tool call (from LLM)
    pub id: String,
    /// Tool name string (for LLM protocol)
    pub name: String,
    /// Human-readable display name
    pub display_name: String,
    /// Tool input parameters
    pub input: serde_json::Value,
    /// Optional context for tools that prepare asynchronously and execute later.
    ///
    /// `deferred_context` is accepted as a legacy alias for backward compatibility.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "deferred_context"
    )]
    pub listen_context: Option<ListenExecutionContext>,
}

/// Context captured for listen/execute tools while awaiting confirmation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListenExecutionContext {
    /// Opaque operation identifier used to execute/cancel.
    ///
    /// `session_id` is accepted as a legacy alias for backward compatibility.
    #[serde(alias = "session_id")]
    pub operation_id: String,
    /// Revision used for optimistic concurrency checks.
    pub revision: u64,
    /// Snapshot shown to the user during confirmation.
    pub snapshot: serde_json::Value,
    /// Optional expiration timestamp (RFC3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

/// Backward-compatible alias for pre-listen naming.
pub type DeferredExecutionContext = ListenExecutionContext;

/// Continuation state that allows resuming the agent loop.
///
/// This contains all the internal state needed to continue execution
/// after receiving a confirmation decision. Pass this back when resuming.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentContinuation {
    /// Thread ID (used for validation on resume)
    pub thread_id: ThreadId,
    /// Current turn number
    pub turn: usize,
    /// Total token usage so far
    pub total_usage: TokenUsage,
    /// Token usage for this specific turn (from the LLM call that generated tool calls)
    pub turn_usage: TokenUsage,
    /// All pending tool calls from this turn
    pub pending_tool_calls: Vec<PendingToolCallInfo>,
    /// Index of the tool call awaiting confirmation
    pub awaiting_index: usize,
    /// Tool results already collected (for tools before the awaiting one)
    pub completed_results: Vec<(String, ToolResult)>,
    /// Agent state snapshot
    pub state: AgentState,
}

/// Input to start or resume an agent run.
#[derive(Debug)]
pub enum AgentInput {
    /// Start a new conversation with user text.
    Text(String),

    /// Resume after a confirmation decision.
    Resume {
        /// The continuation state from `AwaitingConfirmation` (boxed for enum size efficiency).
        continuation: Box<AgentContinuation>,
        /// ID of the tool call being confirmed/rejected.
        tool_call_id: String,
        /// Whether the user confirmed the action.
        confirmed: bool,
        /// Optional reason if rejected.
        rejection_reason: Option<String>,
    },

    /// Continue to the next turn (for single-turn mode).
    ///
    /// Use this after `TurnOutcome::NeedsMoreTurns` to execute the next turn.
    /// The message history already contains tool results from the previous turn.
    Continue,
}

/// Result of tool execution - may indicate async operation in progress.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToolOutcome {
    /// Tool completed synchronously with success
    Success(ToolResult),

    /// Tool completed synchronously with failure
    Failed(ToolResult),

    /// Tool started an async operation - must stream status to completion
    InProgress {
        /// Identifier for the operation (to query status)
        operation_id: String,
        /// Initial message for the user
        message: String,
    },
}

impl ToolOutcome {
    #[must_use]
    pub fn success(output: impl Into<String>) -> Self {
        Self::Success(ToolResult::success(output))
    }

    #[must_use]
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(ToolResult::error(message))
    }

    #[must_use]
    pub fn in_progress(operation_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InProgress {
            operation_id: operation_id.into(),
            message: message.into(),
        }
    }

    /// Returns true if operation is still in progress
    #[must_use]
    pub const fn is_in_progress(&self) -> bool {
        matches!(self, Self::InProgress { .. })
    }
}

// ============================================================================
// Tool Execution Idempotency Types
// ============================================================================

/// Status of a tool execution for idempotency tracking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution started but not yet completed
    InFlight,
    /// Execution completed (success or failure)
    Completed,
}

/// Record of a tool execution for idempotency.
///
/// This struct tracks tool executions to prevent duplicate execution when
/// the agent loop retries after a failure. The write-ahead pattern ensures
/// that execution intent is recorded BEFORE calling the tool, and updated
/// with results AFTER completion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolExecution {
    /// The tool call ID from the LLM (unique per invocation)
    pub tool_call_id: String,
    /// Thread this execution belongs to
    pub thread_id: ThreadId,
    /// Tool name
    pub tool_name: String,
    /// Display name
    pub display_name: String,
    /// Input parameters (for verification)
    pub input: serde_json::Value,
    /// Current status
    pub status: ExecutionStatus,
    /// Result if completed
    pub result: Option<ToolResult>,
    /// For async tools: the operation ID returned by `execute()`
    pub operation_id: Option<String>,
    /// Timestamp when execution started
    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
    /// Timestamp when execution completed
    #[serde(with = "time::serde::rfc3339::option")]
    pub completed_at: Option<OffsetDateTime>,
}

impl ToolExecution {
    /// Create a new in-flight execution record.
    #[must_use]
    pub fn new_in_flight(
        tool_call_id: impl Into<String>,
        thread_id: ThreadId,
        tool_name: impl Into<String>,
        display_name: impl Into<String>,
        input: serde_json::Value,
        started_at: OffsetDateTime,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            thread_id,
            tool_name: tool_name.into(),
            display_name: display_name.into(),
            input,
            status: ExecutionStatus::InFlight,
            result: None,
            operation_id: None,
            started_at,
            completed_at: None,
        }
    }

    /// Mark this execution as completed with a result.
    pub fn complete(&mut self, result: ToolResult) {
        self.status = ExecutionStatus::Completed;
        self.result = Some(result);
        self.completed_at = Some(OffsetDateTime::now_utc());
    }

    /// Set the operation ID for async tool tracking.
    pub fn set_operation_id(&mut self, operation_id: impl Into<String>) {
        self.operation_id = Some(operation_id.into());
    }

    /// Returns true if this execution is still in flight.
    #[must_use]
    pub fn is_in_flight(&self) -> bool {
        self.status == ExecutionStatus::InFlight
    }

    /// Returns true if this execution has completed.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.status == ExecutionStatus::Completed
    }
}

/// Outcome of running a single turn.
///
/// This is returned by `run_turn` to indicate what happened and what to do next.
#[derive(Debug)]
pub enum TurnOutcome {
    /// Turn completed successfully, but more turns are needed.
    ///
    /// Tools were executed and their results are stored in the message history.
    /// Call `run_turn` again with `AgentInput::Continue` to proceed.
    NeedsMoreTurns {
        /// The turn number that just completed
        turn: usize,
        /// Token usage for this turn
        turn_usage: TokenUsage,
        /// Cumulative token usage so far
        total_usage: TokenUsage,
    },

    /// Agent completed successfully (no more tool calls).
    Done {
        /// Total turns executed
        total_turns: u32,
        /// Total input tokens consumed
        input_tokens: u64,
        /// Total output tokens consumed
        output_tokens: u64,
    },

    /// A tool requires user confirmation.
    ///
    /// Present this to the user and call `run_turn` with `AgentInput::Resume`
    /// to continue.
    AwaitingConfirmation {
        /// ID of the pending tool call (from LLM)
        tool_call_id: String,
        /// Tool name string (for LLM protocol)
        tool_name: String,
        /// Human-readable display name
        display_name: String,
        /// Tool input parameters
        input: serde_json::Value,
        /// Description of what confirmation is needed
        description: String,
        /// Continuation state for resuming (boxed for enum size efficiency)
        continuation: Box<AgentContinuation>,
    },

    /// Model refused the request (safety/policy).
    Refusal {
        /// Total turns executed
        total_turns: u32,
        /// Total input tokens consumed
        input_tokens: u64,
        /// Total output tokens consumed
        output_tokens: u64,
    },

    /// An error occurred.
    Error(AgentError),
}
