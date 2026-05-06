//! Span attribute helpers and constants.
//!
//! Attribute keys follow OpenTelemetry `GenAI` semantic conventions where
//! applicable, and use the `agent_sdk.*` namespace for SDK-specific values.

use opentelemetry::KeyValue;

// ── GenAI Semconv Attributes ──────────────────────────────────────────

pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
pub const GEN_AI_CONVERSATION_ID: &str = "gen_ai.conversation.id";
pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";
pub const GEN_AI_REQUEST_MAX_OUTPUT_TOKENS: &str = "gen_ai.request.max_output_tokens";

pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
pub const GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: &str =
    "gen_ai.usage.cache_creation.input_tokens";
pub const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
pub const GEN_AI_TOOL_DESCRIPTION: &str = "gen_ai.tool.description";

pub const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";

// ── SDK-Specific Attributes ──────────────────────────────────────────

pub const SDK_PROVIDER_ID: &str = "agent_sdk.provider.id";
pub const SDK_RUN_MODE: &str = "agent_sdk.run.mode";
pub const SDK_INPUT_KIND: &str = "agent_sdk.input.kind";
pub const SDK_CONFIG_STREAMING: &str = "agent_sdk.config.streaming";
pub const SDK_CONFIG_MAX_TURNS: &str = "agent_sdk.config.max_turns";
pub const SDK_TOOLS_COUNT: &str = "agent_sdk.tools.count";
pub const SDK_TOTAL_TURNS: &str = "agent_sdk.total_turns";
pub const SDK_OUTCOME: &str = "agent_sdk.outcome";

// ── Lifecycle Event Payload Attributes ───────────────────────────────

/// Reason why a run was cancelled (`cancel_token` / `error`).
pub const SDK_CANCEL_REASON: &str = "agent_sdk.cancel.reason";

pub const SDK_TURN_NUMBER: &str = "agent_sdk.turn.number";
pub const SDK_TURN_RESUMED: &str = "agent_sdk.turn.resumed";
pub const SDK_TURN_HAD_TOOL_CALLS: &str = "agent_sdk.turn.had_tool_calls";
pub const SDK_TURN_TOOL_CALL_COUNT: &str = "agent_sdk.turn.tool_call_count";
pub const SDK_TURN_STOP_REASON: &str = "agent_sdk.turn.stop_reason";
pub const SDK_TURN_INPUT_TOKENS: &str = "agent_sdk.turn.input_tokens";
pub const SDK_TURN_OUTPUT_TOKENS: &str = "agent_sdk.turn.output_tokens";
pub const SDK_TURN_CACHE_CREATION_INPUT_TOKENS: &str = "agent_sdk.turn.cache_creation_input_tokens";
pub const SDK_TURN_CACHE_READ_INPUT_TOKENS: &str = "agent_sdk.turn.cache_read_input_tokens";

pub const SDK_LLM_STREAMING: &str = "agent_sdk.llm.streaming";
pub const SDK_LLM_HAD_TOOL_CALLS: &str = "agent_sdk.llm.had_tool_calls";
pub const SDK_LLM_TEXT_OUTPUT_PRESENT: &str = "agent_sdk.llm.text_output_present";
pub const SDK_LLM_THINKING_PRESENT: &str = "agent_sdk.llm.thinking_present";

// ── LLM Stream / Retry Event Payload Attributes ──────────────────────

/// Number of streaming deltas observed for an LLM call.
pub const SDK_LLM_STREAM_DELTA_COUNT: &str = "agent_sdk.llm.stream.delta_count";
/// Wall-clock duration (ms) from stream start to event emission.
pub const SDK_LLM_STREAM_DURATION_MS: &str = "agent_sdk.llm.stream.duration_ms";
/// Reason a streaming attempt was abandoned (`recoverable_error` /
/// `fatal_error` / `event_channel_send_failed`).
pub const SDK_LLM_STREAM_DROP_REASON: &str = "agent_sdk.llm.stream.drop_reason";
/// 1-based retry attempt number.
pub const SDK_LLM_RETRY_ATTEMPT: &str = "agent_sdk.llm.retry.attempt";
/// Configured retry budget for the run.
pub const SDK_LLM_RETRY_MAX_ATTEMPTS: &str = "agent_sdk.llm.retry.max_attempts";
/// Backoff delay before the retry (ms).
pub const SDK_LLM_RETRY_DELAY_MS: &str = "agent_sdk.llm.retry.delay_ms";

pub const SDK_TOOL_DISPLAY_NAME: &str = "agent_sdk.tool.display_name";
pub const SDK_TOOL_TIER: &str = "agent_sdk.tool.tier";
pub const SDK_TOOL_KIND: &str = "agent_sdk.tool.kind";
pub const SDK_TOOL_CONFIRMATION_REQUIRED: &str = "agent_sdk.tool.confirmation_required";
pub const SDK_TOOL_OUTCOME: &str = "agent_sdk.tool.outcome";
pub const SDK_TOOL_DURATION_MS: &str = "agent_sdk.tool.duration_ms";

// ── Tool Event Payload Attributes ────────────────────────────────────

/// Stage label reported by an async-tool progress update.
pub const SDK_TOOL_PROGRESS_STAGE: &str = "agent_sdk.tool.progress.stage";
/// Polling sequence number for an async-tool progress update.
pub const SDK_TOOL_POLL_INDEX: &str = "agent_sdk.tool.poll_index";

pub const SDK_COMPACTION_ORIGINAL_COUNT: &str = "agent_sdk.compaction.original_count";
pub const SDK_COMPACTION_NEW_COUNT: &str = "agent_sdk.compaction.new_count";
pub const SDK_COMPACTION_ORIGINAL_TOKENS: &str = "agent_sdk.compaction.original_tokens";
pub const SDK_COMPACTION_NEW_TOKENS: &str = "agent_sdk.compaction.new_tokens";
pub const SDK_COMPACTION_TRIGGER: &str = "agent_sdk.compaction.trigger";

pub const SDK_OTEL_SYSTEM_INSTRUCTIONS_REF: &str =
    "agent_sdk.observability.system_instructions_ref";
pub const SDK_OTEL_INPUT_MESSAGES_REF: &str = "agent_sdk.observability.input_messages_ref";
pub const SDK_OTEL_OUTPUT_MESSAGES_REF: &str = "agent_sdk.observability.output_messages_ref";

// ── Error Attributes ─────────────────────────────────────────────────

pub const ERROR_TYPE: &str = "error.type";

// ── Span Link Attributes ─────────────────────────────────────────────

/// Hex-encoded trace id of the original attempt that triggered a replay.
///
/// Set on a `SpanLink` from a fresh attempt's `chat <model>` (or
/// equivalent) span pointing at the prior attempt that the worker is
/// replaying.  Lets cross-trace queries answer "show me every attempt
/// of this user submission" even when sampling drops one of the spans.
pub const AGENT_REPLAY_ORIGINAL_TRACE_ID: &str = "agent.replay.original_trace_id";
/// Hex-encoded span id of the original attempt that triggered a replay.
pub const AGENT_REPLAY_ORIGINAL_SPAN_ID: &str = "agent.replay.original_span_id";
/// 1-based attempt index for the replay link, mirroring
/// `TurnAttempt::attempt_number`.
pub const AGENT_REPLAY_ATTEMPT_INDEX: &str = "agent.replay.attempt_index";

// ── Helper Functions ─────────────────────────────────────────────────

/// Create a `KeyValue` pair for a string attribute.
#[must_use]
pub fn kv(key: &'static str, value: impl Into<String>) -> KeyValue {
    KeyValue::new(key, value.into())
}

/// Create a `KeyValue` pair for an i64 attribute.
#[must_use]
pub fn kv_i64(key: &'static str, value: i64) -> KeyValue {
    KeyValue::new(key, value)
}

/// Create a `KeyValue` pair for a bool attribute.
#[must_use]
pub fn kv_bool(key: &'static str, value: bool) -> KeyValue {
    KeyValue::new(key, value)
}

/// Map an `AgentInput` variant to a low-cardinality input kind string.
#[must_use]
pub const fn input_kind_str(input: &crate::types::AgentInput) -> &'static str {
    match input {
        crate::types::AgentInput::Text(_) => "text",
        crate::types::AgentInput::Message(_) => "message",
        crate::types::AgentInput::Continue => "continue",
        crate::types::AgentInput::Resume { .. } => "resume",
        crate::types::AgentInput::SubmitToolResults { .. } => "submit_tool_results",
    }
}

/// Map an SDK `StopReason` to a semconv `finish_reason` string.
#[must_use]
pub const fn finish_reason_str(reason: crate::llm::StopReason) -> &'static str {
    match reason {
        crate::llm::StopReason::EndTurn => "stop",
        crate::llm::StopReason::ToolUse => "tool_call",
        crate::llm::StopReason::MaxTokens => "length",
        crate::llm::StopReason::StopSequence => "stop_sequence",
        crate::llm::StopReason::Refusal => "refusal",
        crate::llm::StopReason::ModelContextWindowExceeded => "model_context_window_exceeded",
    }
}

/// Map an SDK `ToolTier` to a string.
#[must_use]
pub const fn tool_tier_str(tier: crate::types::ToolTier) -> &'static str {
    match tier {
        crate::types::ToolTier::Observe => "observe",
        crate::types::ToolTier::Confirm => "confirm",
    }
}
