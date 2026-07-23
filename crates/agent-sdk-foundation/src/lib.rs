//! # Agent SDK Core
//!
//! Shared contract types for the Agent SDK.
//!
//! This crate contains the **data-only** types that both the runtime
//! (`agent-sdk`) and the server need: IDs, events, LLM messages, turn
//! inputs/outcomes, and continuation payloads.
//!
//! It has no async traits, no runtime dependencies, and no provider
//! implementations — just pure data structures with serde support.

#![forbid(unsafe_code)]

pub mod audit;
pub mod events;
pub mod llm;
pub mod privacy;
pub mod types;

// ── Flat re-exports ──────────────────────────────────────────────────
// Downstream crates can `use agent_sdk_foundation::ThreadId` without reaching
// into sub-modules.

pub use audit::{AuditProvenance, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams};
pub use events::{AgentEvent, AgentEventEnvelope, SequenceCounter, TerminalReason};
pub use llm::{
    ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, ContentSource, Effort, Message,
    Role, StopReason, ThinkingConfig, ThinkingMode, Tool, Usage,
};
pub use privacy::{
    REDACTED_MARKER, RedactionLevel, RedactionPolicy, redact_error, redact_for_observability,
    redact_string, redact_value,
};
pub use types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    CONTINUATION_VERSION, ContinuationEnvelope, ExecutionStatus, ExternalToolResult,
    ListenExecutionContext, PendingToolCallInfo, RetryConfig, ThreadId, TokenUsage, ToolExecution,
    ToolInvocation, ToolOutcome, ToolResult, ToolRuntime, ToolTier, TurnOptions, TurnOutcome,
    TurnSummary,
};
