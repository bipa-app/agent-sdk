//! Reusable `chat` / `execute_tool` span + metric helpers.
//!
//! These primitives are shared by every agent-loop implementation so
//! the daemon-hosted worker emits byte-identical telemetry to the
//! in-process loop.
//!
//! ## Why this module exists
//!
//! The SDK ships two agent-loop implementations:
//!
//! * the in-process `agent_loop` (used by `AgentLoop` /
//!   `BipAgent`), and
//! * the daemon-hosted re-implementation in `agent-server`'s worker
//!   (`crates/agent-server/src/worker/root_turn.rs` +
//!   `agent-service-host`'s `registry_tool_executor.rs`).
//!
//! Both drive an LLM call and a tool dispatch, and both must emit the
//! **same** `gen_ai.*` / `agent_sdk.*` telemetry so dashboards built
//! against the in-process loop light up unchanged when a session runs
//! on the daemon. Before this module the worker bypassed the SDK
//! instrumentation entirely and emitted none of it.
//!
//! Rather than copy the span names, attribute keys, and metric label
//! sets into `agent-server` (where they would silently drift), this
//! module exposes the *exact* primitives the in-process loop uses:
//!
//! * [`build_chat_span`] / [`finish_chat_span_success`] /
//!   [`finish_chat_span_error`] mirror `agent_loop::turn`'s
//!   `build_llm_span` / `stamp_llm_success` / `stamp_llm_error`.
//! * [`build_tool_span`] / [`finish_tool_span`] /
//!   [`ToolSpanOutcome`] mirror `agent_loop::tool_execution`'s
//!   `start_tool_span` / `finish_tool_span`.
//!
//! The metric recording underneath delegates to
//! [`crate::observability::metrics::Metrics`] (the same global
//! singleton), so there is a single source of truth for the label
//! sets and no second meter scope.
//!
//! Compiled only with `feature = "otel"`.

use opentelemetry::KeyValue;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span, Status};

use super::metrics::Metrics;
use super::{attrs, baggage, langfuse, provider_name, spans};
use crate::llm::{ChatResponse, StopReason};
use crate::types::ToolTier;

// ── Chat (LLM client) span ───────────────────────────────────────────

/// Inputs needed to open a `chat {model}` CLIENT span.
///
/// Bundled so call sites stay readable and so the field set can grow
/// (e.g. extra `gen_ai.request.*` attributes) without churning every
/// caller. Mirrors the attributes set by `agent_loop::turn::build_llm_span`.
#[derive(Clone, Copy)]
pub struct ChatSpanParams<'a> {
    /// Raw SDK provider id (e.g. `anthropic`, `openai-responses`).
    /// Normalised to the `gen_ai.provider.name` semconv value
    /// internally via [`provider_name::normalize`].
    pub provider_id: &'static str,
    /// Model the SDK is asking for (`gen_ai.request.model`).
    pub model: &'a str,
    /// Whether the call streams (`agent_sdk.llm.streaming`).
    pub streaming: bool,
    /// Configured output-token cap, if any
    /// (`gen_ai.request.max_output_tokens`).
    pub max_tokens: Option<u32>,
}

/// Open a `chat {model}` CLIENT span with the `gen_ai` semconv
/// attributes known before the call.
///
/// Byte-for-byte mirror of `agent_loop::turn::build_llm_span`: same
/// span name, same initial attribute set, same baggage copy, same
/// Langfuse `generation` observation tag. Pair every successful call
/// with [`finish_chat_span_success`] and every failure with
/// [`finish_chat_span_error`].
#[must_use]
pub fn build_chat_span(params: ChatSpanParams<'_>) -> BoxedSpan {
    let span_name = format!("chat {}", params.model);
    let provider = provider_name::normalize(params.provider_id);
    let mut init_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "chat"),
        KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, provider),
        KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, params.model.to_string()),
        attrs::kv_bool(attrs::SDK_LLM_STREAMING, params.streaming),
        KeyValue::new(attrs::SDK_PROVIDER_ID, params.provider_id),
    ];
    if let Some(max_tokens) = params.max_tokens {
        init_attrs.push(attrs::kv_i64(
            attrs::GEN_AI_REQUEST_MAX_OUTPUT_TOKENS,
            i64::from(max_tokens),
        ));
    }
    let mut span = spans::start_client_span(span_name, init_attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Generation);
    span
}

/// Stamp a successful chat response onto `span`, record the
/// `gen_ai.client.token.usage` + `gen_ai.client.operation.duration`
/// metrics, and end the span.
///
/// Mirrors `agent_loop::turn::stamp_llm_success`: same response-model
/// / id / finish-reason / usage / boolean attributes, same metric
/// label sets (via [`Metrics::record_chat_token_usage`] +
/// [`Metrics::record_chat_operation_duration_success`]).
///
/// `provider_id` is the raw SDK id; it is normalised internally so
/// the metric `gen_ai.provider.name` label matches the span.
pub fn finish_chat_span_success(
    span: &mut BoxedSpan,
    response: &ChatResponse,
    elapsed_secs: f64,
    provider_id: &'static str,
    request_model: &str,
) {
    let provider = provider_name::normalize(provider_id);

    if !response.id.is_empty() {
        span.set_attribute(KeyValue::new(
            attrs::GEN_AI_RESPONSE_ID,
            response.id.clone(),
        ));
    }
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_RESPONSE_MODEL,
        response.model.clone(),
    ));
    if let Some(reason) = response.stop_reason {
        span.set_attribute(KeyValue::new(
            attrs::GEN_AI_RESPONSE_FINISH_REASONS,
            finish_reason_str(reason),
        ));
    }
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        i64::from(response.usage.input_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
        i64::from(response.usage.output_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        i64::from(response.usage.cached_input_tokens),
    ));
    span.set_attribute(attrs::kv_i64(
        attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        i64::from(response.usage.cache_creation_input_tokens),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_HAD_TOOL_CALLS,
        response.has_tool_use(),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_TEXT_OUTPUT_PRESENT,
        response.first_text().is_some(),
    ));
    span.set_attribute(attrs::kv_bool(
        attrs::SDK_LLM_THINKING_PRESENT,
        response.first_thinking().is_some(),
    ));

    let metrics = Metrics::global();
    metrics.record_chat_token_usage(&response.usage, provider, request_model, &response.model);
    metrics.record_chat_operation_duration_success(
        elapsed_secs,
        provider,
        request_model,
        &response.model,
    );

    span.end();
}

/// Stamp a chat error onto `span`, record the
/// `gen_ai.client.operation.duration` metric with the `error.type`
/// label, and end the span.
///
/// Mirrors `agent_loop::turn::stamp_llm_error`. `error_type` must be a
/// stable, low-cardinality string (e.g. `rate_limited`,
/// `server_error`, `invalid_request`, `stream_error`); use
/// [`classify_llm_error`] for free-form provider error messages so the
/// vocabulary matches the in-process loop exactly.
pub fn finish_chat_span_error(
    span: &mut BoxedSpan,
    error_type: &'static str,
    message: &str,
    elapsed_secs: f64,
    provider_id: &'static str,
    request_model: &str,
) {
    let provider = provider_name::normalize(provider_id);
    spans::set_span_error(span, error_type, message);
    Metrics::global().record_chat_operation_duration_error(
        elapsed_secs,
        provider,
        request_model,
        error_type,
    );
    span.end();
}

/// Map a free-form LLM error message to the stable `error.type`
/// attribute / metric label.
///
/// Byte-for-byte mirror of `agent_loop::turn::classify_llm_error` so
/// the daemon worker and the in-process loop bucket transient failures
/// identically.
#[must_use]
pub fn classify_llm_error(msg: &str) -> &'static str {
    if msg.contains("Rate limited") {
        "rate_limited"
    } else if msg.contains("Invalid request") {
        "invalid_request"
    } else if msg.contains("Server error") {
        "server_error"
    } else if msg.contains("Stream") {
        "stream_error"
    } else {
        "_OTHER"
    }
}

/// Map an SDK [`StopReason`] to the semconv `finish_reason` string.
///
/// Mirrors `attrs::finish_reason_str` (kept here too so callers that
/// only import this module get a consistent vocabulary).
#[must_use]
pub const fn finish_reason_str(reason: StopReason) -> &'static str {
    attrs::finish_reason_str(reason)
}

// ── Tool execution span ──────────────────────────────────────────────

/// Inputs needed to open an `execute_tool` INTERNAL span.
///
/// Mirrors the attributes set by
/// `agent_loop::tool_execution::start_tool_span`.
#[derive(Clone, Copy)]
pub struct ToolSpanParams<'a> {
    /// `gen_ai.tool.name` — the protocol tool name.
    pub tool_name: &'a str,
    /// `gen_ai.tool.call.id` — the LLM-assigned call id.
    pub tool_call_id: &'a str,
    /// `agent_sdk.tool.display_name`; skipped when empty.
    pub display_name: &'a str,
    /// `agent_sdk.tool.tier`; `None` when the tool is unknown to the
    /// registry (matches the in-process loop's `unknown`-kind path,
    /// which omits the tier attribute).
    pub tier: Option<ToolTier>,
    /// `agent_sdk.tool.kind` — `sync` / `async` / `listen` /
    /// `unknown`.
    pub kind: &'static str,
}

/// Open an `execute_tool` INTERNAL span.
///
/// Byte-for-byte mirror of
/// `agent_loop::tool_execution::start_tool_span`: same span name, same
/// attribute set (operation name, tool name, call id, optional display
/// name, optional tier, kind), same baggage copy, same Langfuse `tool`
/// observation tag. Finish with [`finish_tool_span`].
#[must_use]
pub fn build_tool_span(params: ToolSpanParams<'_>) -> BoxedSpan {
    let mut span_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "execute_tool"),
        KeyValue::new(attrs::GEN_AI_TOOL_NAME, params.tool_name.to_string()),
        KeyValue::new(attrs::GEN_AI_TOOL_CALL_ID, params.tool_call_id.to_string()),
    ];
    if !params.display_name.is_empty() {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_DISPLAY_NAME,
            params.display_name.to_string(),
        ));
    }
    if let Some(tier) = params.tier {
        span_attrs.push(KeyValue::new(
            attrs::SDK_TOOL_TIER,
            attrs::tool_tier_str(tier),
        ));
    }
    span_attrs.push(KeyValue::new(attrs::SDK_TOOL_KIND, params.kind));

    let mut span = spans::start_internal_span("execute_tool", span_attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Tool);
    span
}

/// Terminal outcome of a tool execution, used to stamp the outcome
/// attributes + metric labels on an `execute_tool` span.
///
/// Mirrors the match arms of
/// `agent_loop::tool_execution::finish_tool_span` so the
/// `agent_sdk.tool.outcome` value and `error.type` value match across
/// loops.
pub enum ToolSpanOutcome<'a> {
    /// Tool ran (or was short-circuited) and produced a result. The
    /// outcome string is derived from the result body / success flag
    /// exactly as the in-process loop does.
    Completed {
        /// Tool output body; inspected for the `Unknown tool:` /
        /// `Blocked:` / `Rejected:` sentinels.
        output: &'a str,
        /// Whether the tool reported success.
        success: bool,
        /// Wall-clock duration in milliseconds, if measured.
        duration_ms: Option<u64>,
    },
    /// Tool requires user confirmation before running.
    AwaitingConfirmation,
    /// The execution boundary itself errored (e.g. event-store commit
    /// failure). Recorded with `error.type = event_store`.
    EventStoreError {
        /// Error message for the span status.
        message: &'a str,
    },
}

/// Stamp the terminal outcome on `span`, record the
/// `agent_sdk.tools.execution.{count,duration}` metrics, and end the
/// span.
///
/// Byte-for-byte mirror of
/// `agent_loop::tool_execution::finish_tool_span`'s outcome handling +
/// metric recording (via [`Metrics::record_tool_execution`]).
pub fn finish_tool_span(
    span: &mut BoxedSpan,
    tool_name: &str,
    tool_kind: &'static str,
    outcome: &ToolSpanOutcome<'_>,
) {
    let (outcome_str, duration_ms) = match outcome {
        ToolSpanOutcome::Completed {
            output,
            success,
            duration_ms,
        } => {
            let outcome_str = if output.starts_with("Unknown tool:") {
                span.set_attribute(KeyValue::new(attrs::ERROR_TYPE, "unknown_tool"));
                span.set_status(Status::error((*output).to_string()));
                "error"
            } else if output.starts_with("Blocked:") {
                "blocked"
            } else if output.starts_with("Rejected:") {
                "rejected"
            } else if *success {
                "success"
            } else {
                "error"
            };
            span.set_attribute(KeyValue::new(attrs::SDK_TOOL_OUTCOME, outcome_str));
            if let Some(ms) = duration_ms {
                span.set_attribute(attrs::kv_i64(
                    attrs::SDK_TOOL_DURATION_MS,
                    i64::try_from(*ms).unwrap_or(i64::MAX),
                ));
            }
            (outcome_str, *duration_ms)
        }
        ToolSpanOutcome::AwaitingConfirmation => {
            span.set_attribute(attrs::kv_bool(attrs::SDK_TOOL_CONFIRMATION_REQUIRED, true));
            span.set_attribute(KeyValue::new(
                attrs::SDK_TOOL_OUTCOME,
                "awaiting_confirmation",
            ));
            ("awaiting_confirmation", None)
        }
        ToolSpanOutcome::EventStoreError { message } => {
            span.set_attribute(KeyValue::new(attrs::ERROR_TYPE, "event_store"));
            span.set_status(Status::error((*message).to_string()));
            span.set_attribute(KeyValue::new(attrs::SDK_TOOL_OUTCOME, "error"));
            ("error", None)
        }
    };

    Metrics::global().record_tool_execution(tool_name, tool_kind, outcome_str, duration_ms);
    span.end();
}
