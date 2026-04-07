//! Instrumentation helpers called from agent loop code paths.

use opentelemetry::KeyValue;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;

use super::attrs;
use super::provider_name;
use super::spans;
use crate::llm::LlmProvider;
use crate::tools::ToolRegistry;
use crate::types::{AgentConfig, AgentInput, ThreadId};

/// Start the root `invoke_agent` span.
pub fn start_root_span<Ctx, P>(
    provider: &P,
    tools: &ToolRegistry<Ctx>,
    config: &AgentConfig,
    thread_id: &ThreadId,
    input: &AgentInput,
    run_mode: &'static str,
) -> BoxedSpan
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider,
{
    let provider_name_val = provider_name::normalize(provider.provider());
    let mut span_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "invoke_agent"),
        KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, provider_name_val),
        KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, provider.model().to_string()),
        KeyValue::new(attrs::GEN_AI_CONVERSATION_ID, thread_id.to_string()),
        KeyValue::new(attrs::SDK_PROVIDER_ID, provider.provider()),
        KeyValue::new(attrs::SDK_RUN_MODE, run_mode),
        KeyValue::new(attrs::SDK_INPUT_KIND, attrs::input_kind_str(input)),
        attrs::kv_bool(attrs::SDK_CONFIG_STREAMING, config.streaming),
        attrs::kv_i64(
            attrs::SDK_TOOLS_COUNT,
            i64::try_from(tools.len()).unwrap_or(0),
        ),
    ];
    if let Some(max_turns) = config.max_turns {
        span_attrs.push(attrs::kv_i64(
            attrs::SDK_CONFIG_MAX_TURNS,
            i64::try_from(max_turns).unwrap_or(0),
        ));
    }
    spans::start_internal_span("invoke_agent", span_attrs)
}

/// Finalize the root span with outcome attributes and end it.
pub fn end_root_span(
    span: &mut BoxedSpan,
    total_turns: usize,
    input_tokens: u64,
    output_tokens: u64,
    outcome: &'static str,
) {
    span.set_attribute(KeyValue::new(
        attrs::SDK_TOTAL_TURNS,
        i64::try_from(total_turns).unwrap_or(0),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        i64::try_from(input_tokens).unwrap_or(0),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
        i64::try_from(output_tokens).unwrap_or(0),
    ));
    span.set_attribute(KeyValue::new(attrs::SDK_OUTCOME, outcome));
    if outcome == "error" {
        spans::set_span_error(span, "agent_error", "agent invocation failed");
    }
    span.end();
}

/// Map an `AgentRunState` to an outcome string.
#[must_use]
pub const fn run_state_outcome(state: &crate::types::AgentRunState) -> &'static str {
    match state {
        crate::types::AgentRunState::Done { .. } => "done",
        crate::types::AgentRunState::Refusal { .. } => "refusal",
        crate::types::AgentRunState::AwaitingConfirmation { .. } => "awaiting_confirmation",
        crate::types::AgentRunState::Cancelled { .. } => "cancelled",
        crate::types::AgentRunState::Error(_) => "error",
    }
}

/// Map a `TurnOutcome` to an outcome string.
#[must_use]
pub const fn turn_outcome_str(outcome: &crate::types::TurnOutcome) -> &'static str {
    match outcome {
        crate::types::TurnOutcome::Done { .. } => "done",
        crate::types::TurnOutcome::Refusal { .. } => "refusal",
        crate::types::TurnOutcome::NeedsMoreTurns { .. } => "needs_more_turns",
        crate::types::TurnOutcome::AwaitingConfirmation { .. } => "awaiting_confirmation",
        crate::types::TurnOutcome::Cancelled { .. } => "cancelled",
        crate::types::TurnOutcome::Error(_) => "error",
    }
}
