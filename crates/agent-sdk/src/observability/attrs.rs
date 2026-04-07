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

pub const SDK_TURN_NUMBER: &str = "agent_sdk.turn.number";
pub const SDK_TURN_RESUMED: &str = "agent_sdk.turn.resumed";
pub const SDK_TURN_HAD_TOOL_CALLS: &str = "agent_sdk.turn.had_tool_calls";
pub const SDK_TURN_TOOL_CALL_COUNT: &str = "agent_sdk.turn.tool_call_count";
pub const SDK_TURN_STOP_REASON: &str = "agent_sdk.turn.stop_reason";
pub const SDK_TURN_INPUT_TOKENS: &str = "agent_sdk.turn.input_tokens";
pub const SDK_TURN_OUTPUT_TOKENS: &str = "agent_sdk.turn.output_tokens";

pub const SDK_LLM_STREAMING: &str = "agent_sdk.llm.streaming";
pub const SDK_LLM_HAD_TOOL_CALLS: &str = "agent_sdk.llm.had_tool_calls";
pub const SDK_LLM_TEXT_OUTPUT_PRESENT: &str = "agent_sdk.llm.text_output_present";
pub const SDK_LLM_THINKING_PRESENT: &str = "agent_sdk.llm.thinking_present";

pub const SDK_TOOL_DISPLAY_NAME: &str = "agent_sdk.tool.display_name";
pub const SDK_TOOL_TIER: &str = "agent_sdk.tool.tier";
pub const SDK_TOOL_KIND: &str = "agent_sdk.tool.kind";
pub const SDK_TOOL_CONFIRMATION_REQUIRED: &str = "agent_sdk.tool.confirmation_required";
pub const SDK_TOOL_OUTCOME: &str = "agent_sdk.tool.outcome";
pub const SDK_TOOL_DURATION_MS: &str = "agent_sdk.tool.duration_ms";

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
