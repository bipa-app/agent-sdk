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
//! - [`TurnSummary`]: Structured server-facing outcome metadata
//! - [`AgentInput`]: Input to start or resume an agent run
//! - [`AgentContinuation`]: Opaque state for resuming after confirmation
//! - [`AgentState`]: Checkpointable agent state

use crate::audit::AuditProvenance;
use crate::llm::{ContentBlock, ContentSource};
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
    pub max_turns: Option<usize>,
    /// Maximum tokens per response.
    ///
    /// If `None`, the SDK uses the provider/model-specific default.
    pub max_tokens: Option<u32>,
    /// System prompt for the agent
    pub system_prompt: String,
    /// Model identifier
    pub model: String,
    /// Retry configuration for transient errors
    pub retry: RetryConfig,
    /// Enable streaming responses from the LLM.
    ///
    /// When `true`, emits `TextDelta` and `ThinkingDelta` events as text arrives
    /// in real-time. When `false` (default), waits for the complete response
    /// before emitting `Text` and `Thinking` events.
    pub streaming: bool,
    /// Optional per-tool execution timeout in milliseconds.
    ///
    /// When set, the agent loop races each tool's `execute()` future
    /// against this budget at the SDK boundary (mirroring
    /// `SubagentConfig::timeout_ms`). A tool that exceeds the budget is
    /// stopped and reported with a synthetic timeout `ToolResult`, keeping
    /// the `tool_use` / `tool_result` history balanced even for
    /// non-cooperative tools. `None` (default) disables the boundary
    /// timeout entirely.
    pub tool_timeout_ms: Option<u64>,
    /// Optional run-level token / cost budgets.
    ///
    /// When set, the agent loop checks the cumulative token usage (and the
    /// estimated USD cost, when the provider/model has pricing metadata)
    /// at every turn-continuation boundary. If a configured limit is
    /// exceeded the run stops with
    /// [`AgentRunState::BudgetExceeded`] / [`TurnOutcome::BudgetExceeded`]
    /// instead of starting another turn. `None` (default) disables
    /// budgeting entirely.
    pub usage_limits: Option<UsageLimits>,
    /// Maximum number of read-only (`ToolTier::Observe`) tool calls the SDK
    /// runs concurrently within a single parallel batch.
    ///
    /// `None` (default) keeps the historical unbounded behavior — every
    /// adjacent observe-tier call in a turn is dispatched at once.
    /// `Some(1)` forces strictly sequential execution; `Some(n)` caps the
    /// in-flight count at `n`. `Some(0)` is not meaningful and is treated
    /// as `Some(1)` (sequential). Result ordering is always preserved.
    pub max_parallel_tools: Option<usize>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: None,
            max_tokens: None,
            system_prompt: String::new(),
            model: String::from("claude-sonnet-4-5-20250929"),
            retry: RetryConfig::default(),
            streaming: false,
            tool_timeout_ms: None,
            usage_limits: None,
            max_parallel_tools: None,
        }
    }
}

/// Run-level token / cost budgets applied by the agent loop.
///
/// Each limit is independent and optional. A `None` field imposes no
/// constraint. When a limit is exceeded the run terminates with a
/// [`BudgetLimitKind`] identifying which limit fired.
///
/// # Evaluation boundaries and bounded overshoot
///
/// Budgets are evaluated at loop boundaries — before a fresh prompt is
/// ingested, before every LLM turn is dispatched, immediately after
/// context-compaction spend is folded in, and before overflow-recovery
/// summarization — never mid-call. Any single boundary may therefore
/// overshoot by the calls already in flight: one turn call, or up to two
/// compaction summarization calls (the second only when the first summary
/// was truncated and retried with a doubled token budget). All such calls
/// are folded into the cumulative usage and re-checked at the next
/// boundary, so the overshoot is bounded and never compounds.
#[derive(Clone, Debug, Default)]
pub struct UsageLimits {
    /// Maximum cumulative tokens (input + output, summed across every
    /// turn) before the run stops.
    pub max_total_tokens: Option<u64>,
    /// Maximum estimated cost in USD before the run stops.
    ///
    /// Only enforced when the run's provider/model has pricing metadata in
    /// [`agent_sdk_providers`](https://docs.rs/agent-sdk-providers); models
    /// without pricing never trip this limit.
    ///
    /// Cost tracking follows the loop's configured provider provenance (the
    /// provider/model the top-level provider reports). Behind
    /// fallback-provider or model-router wrappers the provenance may name a
    /// different backend than the one that actually served a given call, so
    /// the estimated cost — and therefore this limit — may be inaccurate
    /// there.
    pub max_cost_usd: Option<f64>,
}

/// Which run-level budget was exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetLimitKind {
    /// The cumulative token budget ([`UsageLimits::max_total_tokens`]) was hit.
    TotalTokens,
    /// The estimated-cost budget ([`UsageLimits::max_cost_usd`]) was hit.
    CostUsd,
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cached_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

impl TokenUsage {
    pub const fn add(&mut self, other: &Self) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(other.cache_creation_input_tokens);
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
    /// Optional documents (PDFs, images) to pass back to the LLM as native content blocks.
    /// The agent appends these as `ContentBlock::Document` / `ContentBlock::Image` blocks
    /// in the same user message as the tool result, so the model can read them directly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<ContentSource>,
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
            documents: Vec::new(),
            duration_ms: None,
        }
    }

    #[must_use]
    pub fn success_with_data(output: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            output: output.into(),
            data: Some(data),
            documents: Vec::new(),
            duration_ms: None,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            output: message.into(),
            data: None,
            documents: Vec::new(),
            duration_ms: None,
        }
    }

    #[must_use]
    pub const fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Attach documents (PDFs, images) to be sent back to the LLM as native content blocks.
    ///
    /// Use this when a tool produces a binary document that the model should read directly,
    /// e.g. a decrypted PDF that Anthropic can parse natively via its document API.
    ///
    /// # Example
    /// ```rust,ignore
    /// use agent_sdk::{ToolResult, ContentSource};
    ///
    /// Ok(ToolResult::success("PDF decrypted.").with_documents(vec![
    ///     ContentSource::new("application/pdf", base64_data),
    /// ]))
    /// ```
    #[must_use]
    pub fn with_documents(mut self, documents: Vec<ContentSource>) -> Self {
        self.documents = documents;
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
    /// Number of consecutive `on_llm_response` guardrail rejections
    /// (`RetryWithFeedback`) across the thread's turns.
    ///
    /// Persisted so the loop's consecutive-rejection cap also binds
    /// host-driven single-turn orchestration, where each `run_turn` rebuilds
    /// its in-memory context from this state. Reset to zero whenever the
    /// hook accepts a response. Additive and wire-compatible: absent in
    /// older snapshots, defaulting to zero.
    #[serde(default)]
    pub guardrail_retries: usize,
    /// Estimated USD cost accumulated across the thread's LLM calls.
    ///
    /// Each call's usage is priced at the provider/model that served it and
    /// added here, so a thread that rotates models keeps the true sum
    /// instead of repricing its whole history at the newest model's rates.
    /// `None` means no priced usage has been tracked yet: a fresh thread
    /// before its first priced call, a thread whose models have no pricing
    /// metadata, or a snapshot predating this field. Legacy snapshots are
    /// seeded once (best-effort) by repricing the aggregate usage at the
    /// rates current when the thread next runs, then accumulate normally.
    /// Additive and wire-compatible via `#[serde(default)]`.
    #[serde(default)]
    pub accumulated_cost_usd: Option<f64>,
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
            guardrail_retries: 0,
            accumulated_cost_usd: None,
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
#[non_exhaustive]
pub enum AgentRunState {
    /// Agent completed successfully.
    Done {
        total_turns: u32,
        total_usage: TokenUsage,
        /// Estimated cost of the run in USD, when the provider/model has
        /// pricing metadata; `None` otherwise.
        estimated_cost_usd: Option<f64>,
    },

    /// Agent was stopped because a run-level usage budget was exceeded.
    BudgetExceeded {
        total_turns: u32,
        total_usage: TokenUsage,
        /// Estimated cost of the run in USD at the moment the budget was
        /// hit, when pricing metadata is available.
        estimated_cost_usd: Option<f64>,
        /// Which budget limit was exceeded.
        limit: BudgetLimitKind,
    },

    /// Agent was refused by the model (safety/policy).
    Refusal {
        total_turns: u32,
        total_usage: TokenUsage,
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
        /// Versioned continuation envelope for resuming.
        continuation: Box<ContinuationEnvelope>,
    },

    /// Agent run was cancelled via a cancellation token.
    Cancelled {
        total_turns: u32,
        total_usage: TokenUsage,
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
    /// Permission tier of the tool, captured at the moment the LLM
    /// requested the call.
    ///
    /// Persisted on the continuation so that authoritative audit records
    /// on the externalized tool-runtime path can attribute the correct
    /// tier even though the registry is no longer reachable at resume
    /// time. Defaults to [`ToolTier::Confirm`] (the strictest default)
    /// when deserialized from a continuation that predates this field.
    #[serde(default = "default_pending_tier")]
    pub tier: ToolTier,
    /// Tool input parameters as requested by the LLM.
    pub input: serde_json::Value,
    /// Effective input after SDK preparation (e.g. listen-context enrichment).
    ///
    /// For most tools this equals `input`.  The server persists this for
    /// execution while `input` stays as the audit trail.
    #[serde(default)]
    pub effective_input: serde_json::Value,
    /// Optional context for tools that prepare asynchronously and execute later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listen_context: Option<ListenExecutionContext>,
}

/// Default tier used when deserializing a continuation that predates
/// the `tier` field — the strictest default so legacy continuations
/// surface as confirm-tier rather than silently observe-tier.
const fn default_pending_tier() -> ToolTier {
    ToolTier::Confirm
}

// ── Structured policy input ──────────────────────────────────────────

/// Structured input passed to the `pre_tool_use` hook for policy
/// evaluation.
///
/// Bundles every datum that a server-side policy engine needs to make an
/// allow / block / confirm decision, replacing the earlier loose
/// `(tool_name, input, tier)` triple.
///
/// The `AgentHooks` trait itself lives in `agent-sdk-tools` to avoid a
/// dependency cycle; this struct is the stable contract they share.
#[derive(Clone, Debug)]
pub struct ToolInvocation {
    /// Unique ID for this tool call (from LLM).
    pub tool_call_id: String,
    /// Tool name string (for LLM protocol).
    pub tool_name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Permission tier of the tool.
    pub tier: ToolTier,
    /// Input parameters as requested by the LLM (the audit trail).
    pub requested_input: serde_json::Value,
    /// Input after SDK preparation — may differ from `requested_input`
    /// for listen-tools that enrich input during the ready phase.
    pub effective_input: serde_json::Value,
    /// Optional listen-execution context, present when the tool uses
    /// the listen/execute pattern.
    pub listen_context: Option<ListenExecutionContext>,
}

/// Context captured for listen/execute tools while awaiting confirmation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListenExecutionContext {
    /// Opaque operation identifier used to execute/cancel.
    pub operation_id: String,
    /// Revision used for optimistic concurrency checks.
    pub revision: u64,
    /// Snapshot shown to the user during confirmation.
    pub snapshot: serde_json::Value,
    /// Optional expiration timestamp (RFC3339).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "time::serde::rfc3339::option"
    )]
    pub expires_at: Option<OffsetDateTime>,
}

/// Continuation state that allows resuming the agent loop.
///
/// This contains all the internal state needed to continue execution
/// after receiving a confirmation decision. Pass this back when resuming.
///
/// # Turn-summary fields
///
/// `response_id` and `stop_reason` capture the **turn-closing** LLM call
/// that produced [`AgentContinuation::pending_tool_calls`] before the
/// pause. They are carried across the pause boundary so the
/// [`TurnSummary`] emitted on the resume path can report the same LLM
/// metadata as the pre-pause summary for the same turn.
///
/// Both are `Option` and default to `None` for forward compatibility
/// with continuations persisted before these fields existed.
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
    /// Provider response ID from the LLM call that produced this turn's
    /// pending tool calls.
    ///
    /// `None` for continuations persisted before this field was added,
    /// or when the provider did not return an ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Stop reason from the LLM call that produced this turn's pending
    /// tool calls.
    ///
    /// `None` for continuations persisted before this field was added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<crate::llm::StopReason>,
    /// Full content blocks from the LLM response that produced this
    /// turn's pending tool calls (text, thinking, and tool-use blocks).
    ///
    /// When the LLM emits text before tool calls (e.g. "I will run
    /// that." followed by a `tool_use` block), those text blocks must be
    /// preserved so Phase 5 can reconstruct the complete assistant
    /// message in the conversation history.
    ///
    /// Empty for continuations persisted before this field was added.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_content: Vec<crate::llm::ContentBlock>,
}

// ── Versioned continuation envelope ──────────────────────────────────

/// Current envelope version.
pub const CONTINUATION_VERSION: u32 = 1;

/// Versioned wrapper around [`AgentContinuation`].
///
/// This is the **public durable boundary** for server persistence.
/// Servers serialise this envelope (not the raw `AgentContinuation`)
/// so future SDK versions can evolve the inner payload while keeping
/// a stable wire format.
///
/// Unknown versions are rejected at resume time, giving servers a
/// clear upgrade signal instead of silent data corruption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuationEnvelope {
    /// Schema version — currently [`CONTINUATION_VERSION`].
    pub version: u32,
    /// The continuation payload.
    pub payload: AgentContinuation,
}

impl ContinuationEnvelope {
    /// Wrap a continuation in the current version envelope.
    #[must_use]
    pub const fn wrap(payload: AgentContinuation) -> Self {
        Self {
            version: CONTINUATION_VERSION,
            payload,
        }
    }

    /// Validate the envelope version, returning the inner continuation
    /// or an error if the version is unknown.
    ///
    /// # Errors
    ///
    /// Returns an error string if `version` does not match
    /// [`CONTINUATION_VERSION`].
    pub fn unwrap_validated(self) -> Result<AgentContinuation, String> {
        if self.version != CONTINUATION_VERSION {
            return Err(format!(
                "Unsupported continuation version {}: expected {}",
                self.version, CONTINUATION_VERSION,
            ));
        }
        Ok(self.payload)
    }
}

/// A tool result provided by the external runtime for a specific tool call.
///
/// This is the durable handoff payload: a root worker serialises these
/// alongside the [`AgentContinuation`] and provides them on resume via
/// [`AgentInput::SubmitToolResults`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalToolResult {
    /// The tool call ID this result corresponds to (must match a
    /// [`PendingToolCallInfo::id`] from the original
    /// [`TurnOutcome::PendingToolCalls`]).
    pub tool_call_id: String,
    /// The execution result.
    pub result: ToolResult,
}

/// Input to start or resume an agent run.
#[derive(Debug)]
pub enum AgentInput {
    /// Start a new conversation with user text.
    Text(String),

    /// Start a new conversation with rich content (text, images, documents).
    Message(Vec<ContentBlock>),

    /// Resume after a confirmation decision.
    Resume {
        /// The versioned continuation envelope from `AwaitingConfirmation`.
        continuation: Box<ContinuationEnvelope>,
        /// ID of the tool call being confirmed/rejected.
        tool_call_id: String,
        /// Whether the user confirmed the action.
        confirmed: bool,
        /// Optional reason if rejected.
        rejection_reason: Option<String>,
    },

    /// Resume after external tool execution.
    ///
    /// Use this after [`TurnOutcome::PendingToolCalls`] when
    /// [`ToolRuntime::External`] is set.  The caller must provide a result
    /// for **every** pending tool call listed in the continuation.
    ///
    /// The SDK validates the continuation envelope version, appends the
    /// tool results to the message store, and continues to the next LLM turn.
    SubmitToolResults {
        /// The versioned continuation from [`TurnOutcome::PendingToolCalls`].
        continuation: Box<ContinuationEnvelope>,
        /// One result per pending tool call.  The order does not matter,
        /// but every `tool_call_id` from the continuation must be covered.
        results: Vec<ExternalToolResult>,
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
///
/// # Server-facing contract
///
/// Every terminal variant (everything except [`TurnOutcome::Error`]) carries
/// a [`TurnSummary`] with the provider/model/stop-reason/response-id/usage
/// provenance that later server phases need to durably persist. Matching by
/// field name continues to work because the legacy variant fields are
/// preserved alongside the new `summary` field.
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
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },

    /// Agent completed successfully (no more tool calls).
    Done {
        /// Total turns executed
        total_turns: u32,
        /// Cumulative token usage
        total_usage: TokenUsage,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },

    /// A run-level usage budget was exceeded; the turn stops instead of
    /// continuing to the next LLM round-trip.
    BudgetExceeded {
        /// Total turns executed
        total_turns: u32,
        /// Cumulative token usage
        total_usage: TokenUsage,
        /// Estimated cost of the run in USD, when pricing is available.
        estimated_cost_usd: Option<f64>,
        /// Which budget limit was exceeded.
        limit: BudgetLimitKind,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
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
        /// Versioned continuation envelope for resuming.
        continuation: Box<ContinuationEnvelope>,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },

    /// Model refused the request (safety/policy).
    Refusal {
        /// Total turns executed
        total_turns: u32,
        /// Cumulative token usage
        total_usage: TokenUsage,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },

    /// The turn was cancelled via a cancellation token.
    Cancelled {
        /// Total turns executed before cancellation
        total_turns: u32,
        /// Cumulative token usage
        total_usage: TokenUsage,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },

    /// An error occurred.
    ///
    /// No [`TurnSummary`] is attached because the error may have occurred
    /// before the turn produced any durable LLM provenance.
    Error(AgentError),

    /// Tool calls are ready for external execution.
    ///
    /// Only returned when [`ToolRuntime::External`] is set in [`TurnOptions`].
    /// The caller is responsible for executing the tool calls and resuming
    /// with [`AgentInput::SubmitToolResults`], providing one
    /// [`ExternalToolResult`] for each pending tool call.
    ///
    /// The `continuation` must be passed back unmodified — it carries the
    /// turn identity, token usage, and agent state needed to validate and
    /// apply the results.
    PendingToolCalls {
        /// The turn number that produced these tool calls
        turn: usize,
        /// Token usage for this turn's LLM call
        turn_usage: TokenUsage,
        /// Cumulative token usage so far
        total_usage: TokenUsage,
        /// Tool calls to execute externally
        tool_calls: Vec<PendingToolCallInfo>,
        /// Versioned continuation envelope for resuming after external tool execution.
        continuation: Box<ContinuationEnvelope>,
        /// Structured server-facing outcome metadata.
        summary: TurnSummary,
    },
}

impl TurnOutcome {
    /// Returns the attached [`TurnSummary`], if the variant carries one.
    ///
    /// Present on every variant except [`TurnOutcome::Error`].
    #[must_use]
    pub const fn summary(&self) -> Option<&TurnSummary> {
        match self {
            Self::NeedsMoreTurns { summary, .. }
            | Self::Done { summary, .. }
            | Self::BudgetExceeded { summary, .. }
            | Self::AwaitingConfirmation { summary, .. }
            | Self::Refusal { summary, .. }
            | Self::Cancelled { summary, .. }
            | Self::PendingToolCalls { summary, .. } => Some(summary),
            Self::Error(_) => None,
        }
    }
}

// ── Turn summary ─────────────────────────────────────────────────────

/// Structured server-facing outcome metadata for a single turn.
///
/// Captures everything the server needs to durably persist about a
/// turn's LLM-level provenance: thread/turn identity, provider and model
/// identifiers, response ID and stop reason from the turn-closing LLM
/// call, token usage, tool-call count, wall-clock duration, and the
/// [`TurnOptions`] the caller requested.
///
/// # Why this exists
///
/// The original [`TurnOutcome`] only exposed token counts and turn
/// numbers. Later server phases need:
///
/// - **Provider / model** — to correlate rows across provider rotations
///   and to route audit streams by provider.
/// - **Response ID** — to join durable turn rows against the raw
///   provider response stored externally (observability pipelines,
///   replay, support escalations).
/// - **Stop reason** — to branch on `end_turn` vs `tool_use` vs
///   `refusal` without re-parsing message history.
/// - **Tool-call count** — to bill tool execution and detect runaway
///   turns without walking the tool registry.
/// - **Duration** — to feed SLO dashboards and auto-tune retry budgets.
/// - **Tool runtime / strict durability flags** — to record which
///   execution profile was in effect, so later replay can reconstruct
///   the same decisions.
///
/// # Serialization
///
/// `TurnSummary` is fully serializable. Servers are expected to persist
/// it alongside (or inside) their turn rows. Duration is exposed as
/// `duration_ms` (milliseconds) to avoid a serde dance around
/// [`std::time::Duration`].
///
/// # Authoritative vs convenience
///
/// Fields in `TurnSummary` are **authoritative** for server execution:
/// they are produced by the same code path that writes the durable
/// event store and are guaranteed to be consistent with the events the
/// server observed on the wire. Convenience accessors on [`TurnOutcome`]
/// (e.g. the legacy `input_tokens` / `output_tokens` fields on `Done`)
/// are kept only so local callers do not have to break; new code should
/// read from `summary` instead.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSummary {
    /// Thread this turn belongs to.
    ///
    /// Duplicated from the call site so the summary is self-describing
    /// when persisted alone (for durable audit rows).
    pub thread_id: ThreadId,
    /// Turn number that produced this outcome (1-indexed).
    pub turn: usize,
    /// Total number of turns executed in this run so far.
    ///
    /// For mid-run outcomes like `NeedsMoreTurns` / `PendingToolCalls`
    /// this equals `turn`. For terminal outcomes (`Done`, `Refusal`,
    /// `Cancelled`) it reflects the final total.
    pub total_turns: u32,
    /// Token usage for the LLM call(s) that produced this turn.
    pub turn_usage: TokenUsage,
    /// Cumulative token usage across every turn in this run so far.
    pub total_usage: TokenUsage,
    /// Provider / model provenance captured from the turn-closing
    /// LLM call — identical shape to [`AuditProvenance`] so durable
    /// audit rows stay consistent with turn rows.
    pub provenance: AuditProvenance,
    /// Provider response ID from the turn-closing LLM call.
    ///
    /// `None` when the provider did not return an ID or the turn
    /// terminated before the LLM responded (e.g. cancelled before the
    /// first call).
    pub response_id: Option<String>,
    /// Stop reason reported by the turn-closing LLM call.
    ///
    /// `None` when no response was produced for this turn (e.g. the
    /// turn was cancelled before the LLM replied, or the turn was
    /// resumed purely from external tool results without calling the
    /// LLM again).
    pub stop_reason: Option<crate::llm::StopReason>,
    /// Number of tool calls the LLM requested in this turn.
    ///
    /// Zero for pure text turns.
    pub tool_call_count: usize,
    /// Wall-clock duration of this turn, in milliseconds.
    ///
    /// Measured from the start of `run_turn` to the moment the outcome
    /// is returned. Clamped to `u64::MAX` on the unlikely overflow.
    pub duration_ms: u64,
    /// The [`ToolRuntime`] selected for this turn.
    pub tool_runtime: ToolRuntime,
    /// Whether strict durability was requested for this turn.
    pub strict_durability: bool,
}

impl TurnSummary {
    /// Construct an empty summary for a thread / provider / model.
    ///
    /// Used by the runtime as a starting point; it then updates
    /// specific fields as the turn progresses. Tests and downstream
    /// consumers should generally pattern-match on the outcome and
    /// read fields from the populated summary rather than construct
    /// one from scratch.
    #[must_use]
    pub fn new(
        thread_id: ThreadId,
        turn: usize,
        provenance: AuditProvenance,
        options: &TurnOptions,
    ) -> Self {
        Self {
            thread_id,
            turn,
            total_turns: 0,
            turn_usage: TokenUsage::default(),
            total_usage: TokenUsage::default(),
            provenance,
            response_id: None,
            stop_reason: None,
            tool_call_count: 0,
            duration_ms: 0,
            tool_runtime: options.tool_runtime.clone(),
            strict_durability: options.strict_durability,
        }
    }
}

// ── Execution options ────────────────────────────────────────────────

/// How tool calls should be handled during a turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRuntime {
    /// Tools are executed inline by the SDK (the default local-agent behavior).
    #[default]
    Inline,
    /// Tool calls are returned to the caller for external execution.
    ///
    /// When set, `run_turn` yields [`TurnOutcome::PendingToolCalls`] instead
    /// of executing tools itself. The server is responsible for running
    /// tools and calling `run_turn` again.
    External,
}

/// Options that control how a single `run_turn` invocation behaves.
///
/// The default is suitable for local/CLI usage (inline tools, no extra
/// durability). Server mode should set `tool_runtime: External` and
/// `strict_durability: true`.
#[derive(Debug, Clone, Default)]
pub struct TurnOptions {
    /// How tool calls should be handled.
    pub tool_runtime: ToolRuntime,
    /// When true, state is checkpointed at every critical boundary
    /// (before LLM call, after LLM response, after tool execution).
    /// Provides crash-safe server semantics at the cost of extra writes.
    pub strict_durability: bool,
}

// ── RunOptions ───────────────────────────────────────────────────────

/// Per-run trace metadata applied to every span emitted by the agent
/// loop.
///
/// Passed to [`run_with_options`](#method.run_with_options) /
/// [`run_turn_with_options`](#method.run_turn_with_options) /
/// [`run_persistent_with_options`](#method.run_persistent_with_options)
/// so a consumer can configure session / user / Langfuse trace
/// metadata once and have it land on every emitted span — without
/// writing manual span code or pre-installing baggage on the `OTel`
/// context.
///
/// The SDK applies the contents of `RunOptions` at the root
/// `invoke_agent` span:
///
/// * `session_id` / `user_id` — copied to W3C baggage so Langfuse
///   `session.id` / `user.id` filters fire on every child span (the
///   baggage propagation path lives in `agent_sdk::observability::baggage`).
/// * `trace_name` — set as `langfuse.trace.name`.
/// * `trace_tags` — set as `langfuse.trace.tags`.
/// * `trace_metadata` — each entry stamped under `langfuse.trace.metadata.<key>`.
/// * `release` — set as `langfuse.release`.
/// * `environment` — set as `langfuse.environment`.
/// * `trace_text_max_chars` — overrides the default ceiling
///   (`agent_sdk::observability::langfuse::DEFAULT_TRACE_TEXT_MAX_CHARS`)
///   for `langfuse.trace.input` / `langfuse.trace.output`.
///
/// The SDK also computes `langfuse.trace.input` from the supplied
/// [`AgentInput`] (after PII redaction) and
/// streams `langfuse.trace.output` as the agent emits text, tool, and
/// error events.
///
/// `RunOptions` is `Clone + Debug + Default`; it carries only display
/// strings and opaque metadata values (no secrets) so the standard
/// `Debug` derive is safe to expose in error contexts.
///
/// # Example
///
/// ```no_run
/// use agent_sdk_foundation::types::RunOptions;
/// use serde_json::json;
///
/// let opts = RunOptions {
///     session_id: Some("thread-42".to_string()),
///     user_id: Some("user-7".to_string()),
///     trace_name: Some("myapp.assistant.mobile".to_string()),
///     trace_tags: vec!["mobile.android".to_string()],
///     trace_metadata: json!({"version": "1.2.3"})
///         .as_object()
///         .cloned()
///         .unwrap_or_default(),
///     ..Default::default()
/// };
/// # let _ = opts;
/// ```
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Langfuse `session.id` / W3C `session.id` baggage entry.
    pub session_id: Option<String>,
    /// Langfuse `user.id` / W3C `user.id` baggage entry.
    pub user_id: Option<String>,
    /// Display name of the trace in the Langfuse UI.
    pub trace_name: Option<String>,
    /// Free-form labels attached to the trace.
    pub trace_tags: Vec<String>,
    /// Trace-level metadata stamped as `langfuse.trace.metadata.<key>`.
    pub trace_metadata: serde_json::Map<String, serde_json::Value>,
    /// Release identifier for the trace's build.
    pub release: Option<String>,
    /// Langfuse environment slug (`prod`, `staging`, …).
    pub environment: Option<String>,
    /// Override the default character ceiling for trace-level free-text
    /// attributes. `None` falls back to
    /// `agent_sdk::observability::langfuse::DEFAULT_TRACE_TEXT_MAX_CHARS`.
    pub trace_text_max_chars: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::StopReason;

    fn sample_summary() -> TurnSummary {
        TurnSummary {
            thread_id: ThreadId::from_string("t-summary"),
            turn: 2,
            total_turns: 2,
            turn_usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            total_usage: TokenUsage {
                input_tokens: 200,
                output_tokens: 75,
                ..Default::default()
            },
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            response_id: Some("resp_123".into()),
            stop_reason: Some(StopReason::ToolUse),
            tool_call_count: 3,
            duration_ms: 1_234,
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
        }
    }

    #[test]
    fn turn_summary_round_trips_through_json() {
        let original = sample_summary();
        let json = serde_json::to_string(&original).expect("serialize");
        let recovered: TurnSummary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(recovered, original);
    }

    #[test]
    fn turn_summary_json_has_expected_keys() {
        let summary = sample_summary();
        let value = serde_json::to_value(&summary).unwrap();

        // The wire format is the durable server contract — assert
        // every field is present so accidental renames break this
        // test rather than silently corrupting persisted rows.
        for key in [
            "thread_id",
            "turn",
            "total_turns",
            "turn_usage",
            "total_usage",
            "provenance",
            "response_id",
            "stop_reason",
            "tool_call_count",
            "duration_ms",
            "tool_runtime",
            "strict_durability",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }

        // Snake-case tool-runtime variant is stable for server rows.
        assert_eq!(value["tool_runtime"], serde_json::json!("external"));
        // Snake-case stop-reason variant matches the provider wire format.
        assert_eq!(value["stop_reason"], serde_json::json!("tool_use"));
    }

    #[test]
    fn turn_outcome_summary_accessor_works_for_every_variant() {
        let summary = sample_summary();

        let outcomes = vec![
            TurnOutcome::NeedsMoreTurns {
                turn: 1,
                turn_usage: TokenUsage::default(),
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::Done {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::Refusal {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::Cancelled {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
        ];

        for outcome in &outcomes {
            let got = outcome.summary().expect("summary must be present");
            assert_eq!(got, &summary);
        }

        // Error variant has no summary.
        let error_outcome =
            TurnOutcome::Error(AgentError::new("boom", /* recoverable */ false));
        assert!(error_outcome.summary().is_none());
    }

    #[test]
    fn empty_turn_summary_new_captures_options_and_provenance() {
        let opts = TurnOptions {
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
        };
        let provenance = AuditProvenance::new("openai", "gpt-5");
        let summary =
            TurnSummary::new(ThreadId::from_string("t-new"), 7, provenance.clone(), &opts);

        assert_eq!(summary.thread_id, ThreadId::from_string("t-new"));
        assert_eq!(summary.turn, 7);
        assert_eq!(summary.total_turns, 0);
        assert_eq!(summary.provenance, provenance);
        assert_eq!(summary.tool_runtime, ToolRuntime::External);
        assert!(summary.strict_durability);
        assert!(summary.response_id.is_none());
        assert!(summary.stop_reason.is_none());
        assert_eq!(summary.tool_call_count, 0);
        assert_eq!(summary.duration_ms, 0);
    }

    #[test]
    fn stop_reason_as_str_matches_serde_representation() {
        // The durable stop_reason discriminant used in TurnSummary and
        // audit rows must match the serde wire format exactly.
        let cases = [
            (StopReason::EndTurn, "end_turn"),
            (StopReason::ToolUse, "tool_use"),
            (StopReason::MaxTokens, "max_tokens"),
            (StopReason::StopSequence, "stop_sequence"),
            (StopReason::Refusal, "refusal"),
            (
                StopReason::ModelContextWindowExceeded,
                "model_context_window_exceeded",
            ),
        ];
        for (variant, expected) in cases {
            assert_eq!(variant.as_str(), expected);
            let json = serde_json::to_value(variant).unwrap();
            assert_eq!(json, serde_json::json!(expected));
        }
    }

    fn sample_continuation() -> AgentContinuation {
        let thread = ThreadId::from_string("t-continuation");
        AgentContinuation {
            thread_id: thread.clone(),
            turn: 4,
            total_usage: TokenUsage {
                input_tokens: 200,
                output_tokens: 80,
                ..Default::default()
            },
            turn_usage: TokenUsage {
                input_tokens: 50,
                output_tokens: 40,
                ..Default::default()
            },
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_1".into(),
                name: "echo".into(),
                display_name: "Echo".into(),
                tier: ToolTier::Confirm,
                input: serde_json::json!({"message": "hi"}),
                effective_input: serde_json::json!({"message": "hi"}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: Some("resp_7914".into()),
            stop_reason: Some(StopReason::ToolUse),
            response_content: Vec::new(),
        }
    }

    #[test]
    fn agent_continuation_round_trips_llm_metadata() {
        // `response_id` and `stop_reason` travel through
        // durable persistence so the resume-side `TurnSummary` reports
        // the same LLM metadata as the pre-pause summary for the same
        // turn. Guard the wire format so future renames break here
        // rather than silently dropping the fields.
        let original = sample_continuation();
        let json = serde_json::to_string(&original).expect("serialize");

        let value: serde_json::Value = serde_json::from_str(&json).expect("to value");
        assert_eq!(value["response_id"], serde_json::json!("resp_7914"));
        assert_eq!(value["stop_reason"], serde_json::json!("tool_use"));

        let recovered: AgentContinuation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(recovered.response_id.as_deref(), Some("resp_7914"));
        assert_eq!(recovered.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn agent_continuation_deserializes_legacy_payload_without_llm_metadata() {
        // Servers that persisted continuations before this contract
        // landed don't have `response_id` / `stop_reason` fields on
        // disk. Those
        // payloads must still deserialise so running servers do not
        // break on SDK upgrade — the fields default to `None`.
        let thread = ThreadId::from_string("t-legacy");
        let legacy_json = serde_json::json!({
            "thread_id": thread,
            "turn": 1,
            "total_usage": { "input_tokens": 10, "output_tokens": 5 },
            "turn_usage": { "input_tokens": 10, "output_tokens": 5 },
            "pending_tool_calls": [],
            "awaiting_index": 0,
            "completed_results": [],
            "state": AgentState::new(thread.clone()),
        });

        let recovered: AgentContinuation =
            serde_json::from_value(legacy_json).expect("legacy payload deserialises");
        assert_eq!(recovered.thread_id, thread);
        assert_eq!(recovered.turn, 1);
        assert!(
            recovered.response_id.is_none(),
            "legacy payloads default to None",
        );
        assert!(
            recovered.stop_reason.is_none(),
            "legacy payloads default to None",
        );
    }

    #[test]
    fn agent_continuation_omits_llm_metadata_when_none() {
        // `response_id` / `stop_reason` are `skip_serializing_if = None`
        // so that payloads where the provider did not return IDs stay
        // compact and look identical to the legacy wire format. This
        // protects any downstream consumer that matches exact keys.
        let thread = ThreadId::from_string("t-omit");
        let cont = AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        };
        let value = serde_json::to_value(&cont).unwrap();
        assert!(value.get("response_id").is_none());
        assert!(value.get("stop_reason").is_none());
        assert!(value.get("response_content").is_none());
    }
}
