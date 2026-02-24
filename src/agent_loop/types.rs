use crate::context::CompactionConfig;
use crate::events::{AgentEventEnvelope, SequenceCounter};
use crate::llm::StopReason;
use crate::stores::ToolExecutionStore;
use crate::tools::{ToolContext, ToolRegistry};
use crate::types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentState, ListenExecutionContext,
    PendingToolCallInfo, ThreadId, TokenUsage, ToolResult,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use time::OffsetDateTime;
use tokio::sync::mpsc;

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
    /// Awaiting confirmation (yields)
    AwaitingConfirmation {
        tool_call_id: String,
        tool_name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
        continuation: Box<AgentContinuation>,
    },
    /// Error
    Error(AgentError),
}

/// Mutable context for turn execution.
///
/// This holds all the state that's modified during execution.
pub(super) struct TurnContext {
    pub(super) thread_id: ThreadId,
    pub(super) turn: usize,
    pub(super) total_usage: TokenUsage,
    pub(super) state: AgentState,
    pub(super) start_time: Instant,
}

/// Data extracted from `AgentInput::Resume` after validation.
pub(super) struct ResumeData {
    pub(super) continuation: Box<AgentContinuation>,
    pub(super) tool_call_id: String,
    pub(super) confirmed: bool,
    pub(super) rejection_reason: Option<String>,
}

/// Result of initializing state from agent input.
pub(super) struct InitializedState {
    pub(super) turn: usize,
    pub(super) total_usage: TokenUsage,
    pub(super) state: AgentState,
    pub(super) resume_data: Option<ResumeData>,
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
}

pub(super) const MAX_LISTEN_UPDATES: usize = 240;
pub(super) const LISTEN_UPDATE_TIMEOUT: Duration = Duration::from_secs(30);
pub(super) const LISTEN_TOTAL_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) struct ListenReady {
    pub(super) operation_id: String,
    pub(super) revision: u64,
    pub(super) snapshot: serde_json::Value,
    pub(super) expires_at: Option<OffsetDateTime>,
}

pub(super) enum ListenUpdateHandling {
    Continue,
    Ready(ListenReady),
}

pub(super) struct ToolCallExecutionContext<'a, Ctx, H> {
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a ToolRegistry<Ctx>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
}

pub(super) struct ConfirmedToolExecutionContext<'a, Ctx, H> {
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a ToolRegistry<Ctx>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
}

/// Error type for stream processing.
pub(super) enum StreamError {
    Recoverable(String),
    Fatal(String),
}

pub(super) enum ResumeProcessingResult {
    Completed {
        turn_usage: TokenUsage,
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
    pub(super) tx: mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: SequenceCounter,
    pub(super) thread_id: ThreadId,
    pub(super) input: AgentInput,
    pub(super) tool_context: ToolContext<Ctx>,
    pub(super) provider: Arc<P>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) config: AgentConfig,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
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
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) message_store: &'a Arc<M>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
}

pub(super) struct RunLoopResumeParams<'a, Ctx, H, M> {
    pub(super) resume_data: ResumeData,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) message_store: &'a Arc<M>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
}

pub(super) struct RunLoopTurnsParams<'a, Ctx, P, H, M, S> {
    pub(super) ctx: &'a mut TurnContext,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) state_store: &'a Arc<S>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) config: &'a AgentConfig,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
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
    pub(super) tx: mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: SequenceCounter,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
}

pub(super) struct TurnParameters<Ctx, P, H, M, S> {
    pub(super) tx: mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: SequenceCounter,
    pub(super) thread_id: ThreadId,
    pub(super) input: AgentInput,
    pub(super) tool_context: ToolContext<Ctx>,
    pub(super) provider: Arc<P>,
    pub(super) tools: Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: Arc<H>,
    pub(super) message_store: Arc<M>,
    pub(super) state_store: Arc<S>,
    pub(super) config: AgentConfig,
    pub(super) compaction_config: Option<CompactionConfig>,
    pub(super) execution_store: Option<Arc<dyn ToolExecutionStore>>,
}

/// Execute a single turn of the agent loop.
///
/// This is the core turn execution logic shared by both `run_loop` (looping mode)
/// and `run_single_turn` (single-turn mode).
pub(super) struct ExecuteTurnParameters<'a, Ctx, P, H, M> {
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) ctx: &'a mut TurnContext,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) provider: &'a Arc<P>,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) config: &'a AgentConfig,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
}

pub(super) struct TurnMessageLoadParams<'a, P, H, M> {
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) provider: &'a Arc<P>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) compaction_config: Option<&'a CompactionConfig>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) seq: &'a SequenceCounter,
}

pub(super) struct LlmCallParams<'a, P, H> {
    pub(super) provider: &'a Arc<P>,
    pub(super) request: crate::llm::ChatRequest,
    pub(super) config: &'a AgentConfig,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) turn: usize,
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
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) message_store: &'a Arc<M>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) seq: &'a SequenceCounter,
}

pub(super) struct ToolBatchExecutionParams<'a, Ctx, H> {
    pub(super) pending_tool_calls: Vec<PendingToolCallInfo>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
}

pub(super) struct TurnCompletionParams<'a, H, M> {
    pub(super) tool_results: &'a [(String, ToolResult)],
    pub(super) thread_id: &'a ThreadId,
    pub(super) turn: usize,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) message_store: &'a Arc<M>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) seq: &'a SequenceCounter,
}

pub(super) struct TurnToolPhaseParams<'a, Ctx, H, M> {
    pub(super) pending_tool_calls: Vec<PendingToolCallInfo>,
    pub(super) tool_context: &'a ToolContext<Ctx>,
    pub(super) thread_id: &'a ThreadId,
    pub(super) tools: &'a Arc<ToolRegistry<Ctx>>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) seq: &'a SequenceCounter,
    pub(super) execution_store: Option<&'a Arc<dyn ToolExecutionStore>>,
    pub(super) turn: usize,
    pub(super) total_usage: &'a TokenUsage,
    pub(super) turn_usage: &'a TokenUsage,
    pub(super) state: &'a AgentState,
    pub(super) message_store: &'a Arc<M>,
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
    pub(super) tx: &'a mpsc::Sender<AgentEventEnvelope>,
    pub(super) hooks: &'a Arc<H>,
    pub(super) seq: &'a SequenceCounter,
}

/// Extracted content from an LLM response: (thinking, text, `tool_uses`).
pub(super) type ExtractedContent = (
    Option<String>,
    Option<String>,
    Vec<(String, String, serde_json::Value)>,
);
