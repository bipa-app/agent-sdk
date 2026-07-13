use crate::authority::EventAuthority;
use crate::context::{CompactionConfig, ContextCompactor};
use crate::hooks::ToolAuditSink;
use crate::llm::StopReason;
use crate::pricing::CostEstimator;
use crate::stores::{EventStore, ToolExecutionStore};
use crate::tools::{ToolContext, ToolRegistry};
#[cfg(feature = "otel")]
use crate::types::RunOptions;
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentState, ListenExecutionContext,
    PendingToolCallInfo, ThreadId, TokenUsage, ToolResult, TurnOptions, UsageLimits,
};
use agent_sdk_foundation::audit::AuditProvenance;
use std::sync::Arc;
use std::time::{Duration, Instant};
use time::OffsetDateTime;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Internal result of executing a single turn.
///
/// This is used internally by both `run_loop` and `run_single_turn`.
pub(super) enum InternalTurnResult {
    /// Turn completed, more turns needed (tools were executed)
    Continue { turn_usage: TokenUsage },
    /// Done - no more tool calls
    Done,
    /// Model refused the request (safety/policy)
    Refusal,
    /// Run was cancelled via the cancellation token while the turn was
    /// in flight — during the LLM call (streaming or non-streaming) or
    /// during context compaction. Carries the partial usage accrued in
    /// the turn before the cancel was honored.
    Cancelled { turn_usage: TokenUsage },
    /// Awaiting confirmation (yields)
    AwaitingConfirmation {
        tool_call_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
        continuation: Box<AgentContinuation>,
    },
    /// A run-level usage budget was crossed by spend that accrued *inside*
    /// the turn (compaction summarization, or the turn's own LLM usage on
    /// the overflow-recovery path), detected before paying for the next
    /// LLM call. The terminal [`crate::events::AgentEvent::BudgetExceeded`]
    /// was already emitted under the still-open turn at the detection site.
    BudgetExceeded {
        limit: crate::types::BudgetLimitKind,
        estimated_cost_usd: Option<f64>,
        /// The stopping turn's own LLM usage: the overflow turn's usage on
        /// the overflow-recovery path, zero for compaction-only stops (the
        /// turn made no main LLM call — compaction spend rides in the
        /// cumulative totals, not the per-turn summary).
        turn_usage: TokenUsage,
    },
    /// Tool calls ready for external execution (server mode)
    PendingToolCalls {
        turn_usage: TokenUsage,
        pending_tool_calls: Vec<PendingToolCallInfo>,
        continuation: Box<AgentContinuation>,
    },
    /// Error
    Error(AgentError),
}

/// Maximum number of compaction retries before giving up.
pub(super) const MAX_COMPACTION_RETRIES: usize = 3;

/// Maximum number of *consecutive* `on_llm_response` rejections
/// ([`crate::hooks::ResponseDecision::RetryWithFeedback`]) before the run is
/// terminated with an error.
///
/// Every guardrail retry pays for a full LLM round-trip, so a
/// deterministically-rejecting hook under the default config (`max_turns:
/// None`, no usage budget) would otherwise loop — and bill — forever. Eight
/// attempts is generous for the intended steering use case (moderation
/// feedback usually lands within one or two retries) while capping the
/// worst-case spend of a mis-implemented hook at single-digit calls.
///
/// The streak lives in [`crate::types::AgentState::guardrail_retries`] and
/// is persisted with every state checkpoint, so the cap binds both
/// in-process looping runs and host-driven single-turn orchestration (where
/// each `run_turn` rehydrates the streak from the state store). It resets
/// whenever the hook accepts a response.
pub(super) const MAX_CONSECUTIVE_GUARDRAIL_RETRIES: usize = 8;

/// Mutable context for turn execution.
///
/// This holds all the state that's modified during execution.
pub(super) struct TurnContext {
    pub(super) thread_id: ThreadId,
    pub(super) turn: usize,
    pub(super) total_usage: TokenUsage,
    pub(super) state: AgentState,
    pub(super) start_time: Instant,
    /// Pricing source for run-level cost budgeting, when the loop was built
    /// with one. Every cost estimate on the turn path reads it from here:
    /// the dynamic catalog is consulted before the static capability table,
    /// so a model the table has never heard of still accrues cost.
    pub(super) cost_estimator: Option<Arc<dyn CostEstimator>>,
    /// Number of consecutive compaction retries for context overflow.
    pub(super) compaction_retries: usize,
    /// Optional system reminder to inject into the next LLM call.
    ///
    /// Set by `begin_turn` when approaching the turn limit, then consumed
    /// by `execute_turn_inner` to append a user message to the conversation.
    pub(super) pending_reminder: Option<String>,
    /// Ingestion-time `pre_llm_request` decision awaiting the run's first
    /// LLM call; consumed (`take`) by `request_turn_response` so the hook
    /// is never invoked twice for that call. See [`PreEvaluatedRequest`].
    pub(super) pending_first_request: Option<PreEvaluatedRequest>,
    // ── Turn summary accumulators ───────────────────────────────────
    //
    // These mirror fields on `agent_sdk_foundation::TurnSummary` and are
    // populated incrementally as the turn progresses, then promoted to
    // a full `TurnSummary` by `build_turn_summary` when the outcome is
    // produced. Keeping the accumulators on `TurnContext` lets the
    // existing flow populate them at the same spots where the legacy
    // usage accounting already happens, without adding another
    // parameter to every function along the path.
    /// Provider response ID from the most recent LLM call in this turn.
    pub(super) response_id: Option<String>,
    /// Stop reason from the most recent LLM call in this turn.
    pub(super) stop_reason: Option<StopReason>,
    /// Number of tool calls the LLM requested in this turn.
    pub(super) tool_call_count: usize,
    /// Static input-kind label captured from the run's [`AgentInput`].
    ///
    /// Threaded onto every per-turn metric so dashboards can group
    /// turn timings by whether the run was kicked off with `text`,
    /// `message`, `continue`, `resume`, or `submit_tool_results`. The
    /// value is constant across turns because every continuation
    /// turn shares the same originating run.
    #[cfg(feature = "otel")]
    pub(super) input_kind: &'static str,
}

/// Data extracted from `AgentInput::Resume` after validation.
pub(super) struct ResumeData {
    pub(super) continuation: Box<AgentContinuation>,
    pub(super) tool_call_id: String,
    pub(super) confirmed: bool,
    pub(super) rejection_reason: Option<String>,
}

/// The `pre_llm_request` decision produced at fresh-input ingestion time,
/// carried into the run's FIRST LLM call so the hook fires exactly once per
/// call.
///
/// For fresh `Text` / `Message` input the hook is evaluated against the
/// request the next turn would send (existing history + the candidate user
/// message) BEFORE the candidate is durably appended — a `Block` must leave
/// nothing ingested, or a same-thread rephrase-and-retry would rebuild its
/// request WITH the blocked content in history and the rejected material
/// would still reach the provider.
pub(super) enum PreEvaluatedRequest {
    /// The hook proceeded: the first turn builds and sends its request
    /// normally (it may differ from the evaluated request by SDK-added
    /// content only — compaction summarizing what the hook already saw, or
    /// a turn-budget reminder) without re-invoking the hook.
    Proceed,
    /// The hook returned `Modify`: this replacement is sent as-is for the
    /// first call. The ORIGINAL user message is what persists, matching
    /// mid-run `Modify` semantics (the hook shapes the outbound request,
    /// never the durable history).
    Modified(Box<crate::llm::ChatRequest>),
}

pub(super) struct InitializedState {
    pub(super) turn: usize,
    pub(super) total_usage: TokenUsage,
    pub(super) state: AgentState,
    pub(super) resume_data: Option<ResumeData>,
    /// Ingestion-time `pre_llm_request` decision for the first LLM call.
    /// `Some` only for fresh `Text` / `Message` input; `None` for
    /// `Resume` / `Continue` / `SubmitToolResults`, whose first call
    /// evaluates the hook at turn time as usual.
    pub(super) first_request_decision: Option<PreEvaluatedRequest>,
}

/// Outcome of executing a single tool call.
pub(super) enum ToolExecutionOutcome {
    /// Tool executed successfully (or failed), result captured
    Completed { tool_id: String, result: ToolResult },
    /// Tool requires user confirmation before execution
    RequiresConfirmation {
        tool_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
        listen_context: Option<ListenExecutionContext>,
    },
    /// Event persistence or other infrastructure failure aborted the tool step.
    Error(AgentError),
}

pub(super) const MAX_LISTEN_UPDATES: usize = 240;
pub(super) const LISTEN_UPDATE_TIMEOUT: Duration = Duration::from_secs(30);
pub(super) const LISTEN_TOTAL_TIMEOUT: Duration = Duration::from_mins(5);

/// Maximum time the SDK waits for the *next* streaming frame before treating
/// the provider connection as stalled.
///
/// `process_stream` otherwise only races `stream.next()` against the cancel
/// token, so a half-open connection (provider HTTP clients set at most a
/// connect timeout) would pin the turn forever. Expiry is surfaced as a
/// recoverable stream error so the retry loop re-establishes the stream.
/// Mirrors the listen-tool per-update bound (`LISTEN_UPDATE_TIMEOUT`); kept
/// generous so legitimate long thinking pauses between chunks are not cut off.
///
/// Reduced under `cfg(test)` so the in-crate unit tests can exercise the
/// stalled-stream path deterministically without `tokio` test-util / a long
/// real wait. The production value applies to integration tests and real
/// builds (where `agent-sdk` is linked as a non-test dependency).
#[cfg(not(test))]
pub(super) const LLM_STREAM_INACTIVITY_TIMEOUT: Duration = Duration::from_mins(2);
#[cfg(test)]
pub(super) const LLM_STREAM_INACTIVITY_TIMEOUT: Duration = Duration::from_millis(20);

/// Overall deadline for a single non-streaming LLM call.
///
/// A stalled non-streaming provider exposes no per-token progress to observe,
/// so without a deadline the call hangs the turn until an explicit cancel.
/// Expiry is reported as a retryable server error. Mirrors
/// `LISTEN_TOTAL_TIMEOUT`.
pub(super) const LLM_CALL_TOTAL_TIMEOUT: Duration = Duration::from_mins(5);

pub(super) struct ListenReady {
    pub(super) operation_id: String,
    pub(super) revision: u64,
    pub(super) snapshot: serde_json::Value,
    pub(super) expires_at: Option<OffsetDateTime>,
}

#[derive(Clone, Copy)]
pub(super) enum ListenProgressStage {
    Update,
    Ready,
    Invalidated,
}

impl ListenProgressStage {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Update => "listen_update",
            Self::Ready => "listen_ready",
            Self::Invalidated => "listen_invalidated",
        }
    }
}

pub(super) enum ListenUpdateHandling {
    Continue,
    Ready(ListenReady),
}

pub(super) struct ListenUpdateContext<'a, H> {
    pub(super) pending: &'a PendingToolCallInfo,
    pub(super) hooks: &'a Arc<H>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
}

pub(super) struct ListenWaitParams<'a, Ctx, H> {
    pub(super) pending: &'a PendingToolCallInfo,
    pub(super) tool: &'a Arc<dyn crate::tools::ErasedListenTool<Ctx>>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) update_ctx: ListenUpdateContext<'a, H>,
}

pub(super) struct ToolCallExecutionContext<'a, Ctx, H> {
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a ToolRegistry<Ctx>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) turn: usize,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
}

/// Snapshot inputs used to build an `AgentContinuation` when a tool outcome
/// pauses the turn for confirmation.
///
/// Groups the turn-scoped values `handle_tool_outcome` needs so the function
/// takes a single context reference instead of ten positional arguments.
pub(super) struct ToolOutcomeContext<'a> {
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    pub(super) response_id: Option<&'a str>,
    pub(super) stop_reason: Option<StopReason>,
}

/// Error type for stream processing.
pub(super) enum StreamError {
    Recoverable {
        message: String,
        /// Server-supplied retry delay carried by a rate-limited stream error
        /// (`Retry-After`, or a hint parsed out of the error body). Overrides
        /// the exponential backoff, exactly as on the non-streaming path.
        retry_after: Option<std::time::Duration>,
    },
    Fatal(String),
    /// The run's cancellation token fired while the stream was still
    /// being consumed. Terminal — not retried.
    Cancelled,
}

/// Three-state outcome of an LLM call (streaming or non-streaming).
///
/// Distinguishes a clean cancellation from an error so the turn loop
/// can return [`InternalTurnResult::Cancelled`] (balanced history, a
/// `Cancelled` terminal event) instead of folding the cancel into a
/// generic [`InternalTurnResult::Error`].
pub(super) enum LlmOutcome {
    Response(crate::llm::ChatResponse),
    Cancelled,
    Error(AgentError),
}

/// Turn-summary metrics captured from the LLM call that preceded the
/// pause, surfaced by [`process_resume`] so the resume handler can
/// build a [`TurnSummary`] from real data instead of fabricating a
/// [`TurnContext`].
///
/// These fields describe the **turn-closing** LLM call for the turn
/// being summarised — the same call whose tool-use blocks produced the
/// pending tool calls that the resume is now finishing. They are
/// threaded through [`AgentContinuation`] so they survive the pause /
/// resume boundary without needing a separate side-channel.
pub(super) struct ResumeSummaryMetrics {
    pub(super) response_id: Option<String>,
    pub(super) stop_reason: Option<StopReason>,
    pub(super) tool_call_count: usize,
}

pub(super) enum ResumeProcessingResult {
    Completed {
        turn_usage: TokenUsage,
        metrics: ResumeSummaryMetrics,
    },
    AwaitingConfirmation {
        tool_call_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
        continuation: Box<AgentContinuation>,
    },
}

pub(super) struct RunLoopParameters<Ctx, P, H, M, S> {
    pub(super) event_store: Arc<dyn EventStore>,
    pub(super) authority: Arc<dyn EventAuthority>,
    pub(super) thread_id: ThreadId,
    pub(super) input: AgentInput,
    pub(super) tool_context: ToolContext<Ctx>,
    pub(super) provider: Arc<P>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) config: AgentConfig,
    /// Pricing source for the run's cost budget; see
    /// [`TurnContext::cost_estimator`].
    pub(super) cost_estimator: Option<Arc<dyn CostEstimator>>,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) compactor: Option<Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: Arc<dyn ToolAuditSink>,
    pub(super) cancel_token: CancellationToken,
    /// Optional channel for receiving new messages in persistent mode.
    pub(super) input_rx: Option<mpsc::Receiver<AgentInput>>,
    /// Per-tool reminder configuration for the run.
    pub(super) reminder_config: Option<crate::reminders::ReminderConfig>,
    /// Trace metadata applied to the root span (and threaded through
    /// every event emission as `langfuse.trace.output`).
    ///
    /// Always present on the otel build path; defaults to
    /// [`RunOptions::default`] when the caller used the historical
    /// `run` / `run_persistent` entry points. Stripped from non-otel
    /// builds because nothing consumes it there.
    #[cfg(feature = "otel")]
    pub(super) run_options: RunOptions,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<Arc<dyn crate::observability::ObservabilityStore>>,
}

pub(super) struct ResumeProcessingParameters<'a, Ctx, H, M> {
    pub(super) resume_data: ResumeData,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
}

pub(super) struct RunLoopTurnsParams<'a, Ctx, P, H, M, S> {
    pub(super) ctx: &'a mut TurnContext,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) state_store: &'a Arc<S>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) config: &'a AgentConfig,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) compactor: Option<&'a Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
    pub(super) cancel_token: &'a CancellationToken,
    /// Optional channel for receiving new messages in persistent mode.
    pub(super) input_rx: Option<&'a mut mpsc::Receiver<AgentInput>>,
    pub(super) turn_options: &'a TurnOptions,
    /// Per-tool reminder configuration applied after tools execute.
    pub(super) reminder_config: Option<&'a crate::reminders::ReminderConfig>,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<&'a Arc<dyn crate::observability::ObservabilityStore>>,
}

pub(super) struct PersistentDoneParams<'a, Ctx, P, H, M, S> {
    pub(super) ctx: &'a TurnContext,
    pub(super) rx: &'a mut mpsc::Receiver<AgentInput>,
    pub(super) message_store: &'a Arc<M>,
    /// Provider / tools / config for the injected-input `pre_llm_request`
    /// guard, which evaluates the hook BEFORE the injected message is
    /// durably appended (the same seam as fresh-input initialization).
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) config: &'a AgentConfig,
    /// State store, needed so a terminal event keyed under a synthetic
    /// (never-executed) turn can persist the advanced `turn_count`.
    pub(super) state_store: &'a Arc<S>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) current_turn: usize,
    pub(super) cancel_token: &'a CancellationToken,
    /// Provider/model provenance, used to estimate the run cost reported
    /// on the terminal `Done` state when the input channel closes.
    pub(super) provenance: &'a AuditProvenance,
    /// Run-level usage budgets, evaluated after every completed turn and
    /// **before** parking for injected input, so an over-budget persistent
    /// run terminates instead of consuming a later prompt it can never
    /// answer.
    pub(super) usage_limits: Option<&'a UsageLimits>,
}

pub(super) struct RunLoopTurnResultParams<'a, Ctx, P, H, M, S> {
    pub(super) result: InternalTurnResult,
    pub(super) ctx: &'a TurnContext,
    pub(super) input_rx: Option<&'a mut mpsc::Receiver<AgentInput>>,
    pub(super) message_store: &'a Arc<M>,
    /// Forwarded into the persistent-mode `Done` handler for the
    /// injected-input `pre_llm_request` guard.
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) config: &'a AgentConfig,
    pub(super) state_store: &'a Arc<S>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) cancel_token: &'a CancellationToken,
    pub(super) current_turn: usize,
    /// Provider/model provenance, used to estimate the run cost reported on
    /// the terminal `Done` state on the persistent input-channel-closed path.
    pub(super) provenance: &'a AuditProvenance,
    /// Run-level usage budgets, forwarded into the persistent-mode `Done`
    /// handler so it can stop before parking for input.
    pub(super) usage_limits: Option<&'a UsageLimits>,
}

pub(super) struct SingleTurnResumeParams<Ctx, H, M, S> {
    pub(super) resume_data: ResumeData,
    pub(super) turn: usize,
    pub(super) total_usage: TokenUsage,
    pub(super) state: AgentState,
    pub(super) thread_id: ThreadId,
    pub(super) tool_context: ToolContext<Ctx>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) event_store: Arc<dyn EventStore>,
    pub(super) authority: Arc<dyn EventAuthority>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: Arc<dyn ToolAuditSink>,
    pub(super) provenance: AuditProvenance,
    /// Execution options selected by the caller, needed so the resume
    /// path can carry [`TurnOptions`] through into the emitted
    /// [`agent_sdk_foundation::TurnSummary`].
    pub(super) turn_options: TurnOptions,
    /// Wall-clock instant when the enclosing `run_turn` invocation
    /// started — used to measure `duration_ms` for the summary.
    pub(super) start_time: Instant,
    /// Run-level usage budgets, checked when the resume completes so the
    /// caller receives [`crate::types::TurnOutcome::BudgetExceeded`] instead
    /// of `NeedsMoreTurns` when the paused turn already crossed a limit.
    pub(super) usage_limits: Option<UsageLimits>,
    /// Pricing source for that budget check; see
    /// [`TurnContext::cost_estimator`].
    pub(super) cost_estimator: Option<Arc<dyn CostEstimator>>,
}

pub(super) struct TurnParameters<Ctx, P, H, M, S> {
    pub(super) event_store: Arc<dyn EventStore>,
    pub(super) authority: Arc<dyn EventAuthority>,
    pub(super) thread_id: ThreadId,
    pub(super) input: AgentInput,
    pub(super) tool_context: ToolContext<Ctx>,
    pub(super) provider: Arc<P>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) config: AgentConfig,
    /// Pricing source for the turn's cost budget; see
    /// [`TurnContext::cost_estimator`].
    pub(super) cost_estimator: Option<Arc<dyn CostEstimator>>,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) compactor: Option<Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: Arc<dyn ToolAuditSink>,
    pub(super) cancel_token: CancellationToken,
    pub(super) turn_options: TurnOptions,
    /// Per-tool reminder configuration for the turn.
    pub(super) reminder_config: Option<crate::reminders::ReminderConfig>,
    /// Trace metadata applied to the root span. See
    /// [`RunLoopParameters::run_options`] for the full contract.
    /// Only consumed on the otel build path — see the gating note
    /// on [`RunLoopParameters::run_options`].
    #[cfg(feature = "otel")]
    pub(super) run_options: RunOptions,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<Arc<dyn crate::observability::ObservabilityStore>>,
}

/// Execute a single turn of the agent loop.
///
/// This is the core turn execution logic shared by both `run_loop` (looping mode)
/// and `run_single_turn` (single-turn mode).
pub(super) struct ExecuteTurnParameters<'a, Ctx, P, H, M, S> {
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) ctx: &'a mut TurnContext,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) state_store: &'a Arc<S>,
    pub(super) config: &'a AgentConfig,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) compactor: Option<&'a Arc<dyn ContextCompactor>>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
    pub(super) turn_options: &'a TurnOptions,
    /// Per-tool reminder configuration applied after tools execute.
    pub(super) reminder_config: Option<&'a crate::reminders::ReminderConfig>,
    /// Run-level cancellation token, threaded into the LLM call and the
    /// compaction path so both phases are raced against cancel.
    pub(super) cancel_token: &'a CancellationToken,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<&'a Arc<dyn crate::observability::ObservabilityStore>>,
}

pub(super) struct TurnMessageLoadParams<'a, P, H, M> {
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) provider: &'a Arc<P>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) compactor: Option<&'a Arc<dyn ContextCompactor>>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    /// Run-level cancellation token; checked before the destructive
    /// `replace_history` write so a cancel during compaction cannot let
    /// a slow summary land after the user asked to stop.
    pub(super) cancel_token: &'a CancellationToken,
}

pub(super) struct LlmCallParams<'a, P, H> {
    pub(super) provider: &'a Arc<P>,
    pub(super) request: crate::llm::ChatRequest,
    pub(super) config: &'a AgentConfig,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    /// Mutable so the streaming retry loop can regenerate the ids per
    /// attempt — isolating each attempt's deltas under a distinct id — and
    /// hand the final attempt's ids back to the caller for the post-stream
    /// Text/Refusal events. The non-streaming path leaves them untouched.
    pub(super) message_id: &'a mut String,
    pub(super) thinking_id: &'a mut String,
    /// Run-level cancellation token, raced against the provider call
    /// (streaming + non-streaming) so a cancel mid-stream or before
    /// the first token stops the LLM phase promptly instead of waiting
    /// for the response to complete.
    pub(super) cancel_token: &'a CancellationToken,
    #[cfg(feature = "otel")]
    pub(super) observability_store: Option<&'a Arc<dyn crate::observability::ObservabilityStore>>,
}

pub(super) struct LlmEventContext<'a, H> {
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    /// Run-level cancellation token observed inside the streaming /
    /// retry loops. See [`LlmCallParams::cancel_token`].
    pub(super) cancel_token: &'a CancellationToken,
}

#[derive(Clone, Copy)]
pub(super) struct LlmStreamIds<'a> {
    pub(super) message_id: &'a str,
    pub(super) thinking_id: &'a str,
}

pub(super) struct ProcessedTurnResponse {
    pub(super) stop_reason: Option<StopReason>,
    pub(super) text_content: Option<String>,
    pub(super) pending_tool_calls: Vec<PendingToolCallInfo>,
}

pub(super) struct TurnResponseProcessingParams<'a, Ctx, H, M> {
    pub(super) response: crate::llm::ChatResponse,
    pub(super) message_id: &'a str,
    pub(super) thinking_id: &'a str,
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
}

pub(super) struct ToolBatchExecutionParams<'a, Ctx, H> {
    pub(super) pending_tool_calls: Vec<PendingToolCallInfo>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    /// Response ID from the LLM call that produced `pending_tool_calls`.
    /// Copied into any [`AgentContinuation`] this phase emits so the
    /// resume-side [`TurnSummary`] can report the same value.
    pub(super) response_id: Option<String>,
    /// Stop reason from the LLM call that produced `pending_tool_calls`.
    /// Copied into any [`AgentContinuation`] this phase emits so the
    /// resume-side [`TurnSummary`] can report the same value.
    pub(super) stop_reason: Option<StopReason>,
    /// Cap on how many adjacent `Observe`-tier tool calls run concurrently
    /// within one parallel batch. `None` keeps the unbounded behavior.
    pub(super) max_parallel_tools: Option<usize>,
    /// Per-tool reminders (keyed by tool name) appended to matching tool
    /// results before they are fed back to the model. `None` disables the
    /// reminder pass.
    pub(super) reminder_config: Option<&'a crate::reminders::ReminderConfig>,
}

pub(super) struct TurnCompletionParams<'a, H, M> {
    pub(super) tool_results: &'a [(String, ToolResult)],
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) message_store: &'a Arc<M>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
}

pub(super) struct TurnToolPhaseParams<'a, Ctx, H, M> {
    pub(super) pending_tool_calls: Vec<PendingToolCallInfo>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) audit_sink: &'a Arc<dyn ToolAuditSink>,
    pub(super) provenance: &'a AuditProvenance,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    pub(super) message_store: &'a Arc<M>,
    /// Response ID from the LLM call that produced `pending_tool_calls`.
    /// Forwarded into any [`AgentContinuation`] this phase emits.
    pub(super) response_id: Option<String>,
    /// Stop reason from the LLM call that produced `pending_tool_calls`.
    /// Forwarded into any [`AgentContinuation`] this phase emits.
    pub(super) stop_reason: Option<StopReason>,
    /// Cap on concurrent `Observe`-tier tool execution; see
    /// [`AgentConfig::max_parallel_tools`].
    pub(super) max_parallel_tools: Option<usize>,
    /// Per-tool reminders to apply to matching results; see
    /// [`crate::reminders::ReminderConfig`].
    pub(super) reminder_config: Option<&'a crate::reminders::ReminderConfig>,
}

pub(super) struct TurnStopReasonParams<'a, P, H, M> {
    pub(super) stop_reason: Option<StopReason>,
    pub(super) text_content: Option<String>,
    pub(super) had_tool_calls: bool,
    pub(super) message_id: String,
    pub(super) turn_usage: TokenUsage,
    pub(super) ctx: &'a mut TurnContext,
    pub(super) provider: &'a Arc<P>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) compactor: Option<&'a Arc<dyn ContextCompactor>>,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    /// Run provenance, used to price the usage of any emergency compaction
    /// triggered by `ModelContextWindowExceeded`.
    pub(super) provenance: &'a AuditProvenance,
    /// Run-level usage budgets, consulted before the overflow-driven
    /// emergency compaction pays for a summarization call.
    pub(super) usage_limits: Option<&'a UsageLimits>,
    /// Run-level cancellation token, forwarded into the overflow-driven
    /// compaction path triggered by `ModelContextWindowExceeded`.
    pub(super) cancel_token: &'a CancellationToken,
}

pub(super) struct ConvertTurnResultParams<'a, H, S> {
    pub(super) result: InternalTurnResult,
    pub(super) ctx: TurnContext,
    pub(super) event_store: &'a Arc<dyn EventStore>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) authority: &'a Arc<dyn EventAuthority>,
    pub(super) thread_id: ThreadId,
    pub(super) current_turn: usize,
    pub(super) state_store: &'a Arc<S>,
    /// Provider / model provenance, captured once at the start of the
    /// turn and promoted into every [`TurnSummary`] this conversion
    /// produces.
    pub(super) provenance: &'a AuditProvenance,
    /// Execution options selected by the caller for this turn.
    pub(super) turn_options: &'a TurnOptions,
    /// Run-level usage budgets, checked at the single-turn continuation
    /// boundary so the turn yields [`TurnOutcome::BudgetExceeded`] instead
    /// of [`TurnOutcome::NeedsMoreTurns`] when a limit is hit.
    pub(super) usage_limits: Option<&'a UsageLimits>,
}

/// Extracted content from an LLM response: (thinking, text, `tool_uses`).
pub(super) type ExtractedContent = (
    Option<String>,
    Option<String>,
    Vec<(String, String, serde_json::Value)>,
);
