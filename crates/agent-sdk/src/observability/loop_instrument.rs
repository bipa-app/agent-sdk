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

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span, Status, TraceContextExt};
use opentelemetry::{Context, KeyValue, global};

use super::metrics::Metrics;
use super::{attrs, baggage, langfuse, provider_name, spans};
use crate::llm::{ChatResponse, StopReason};
use crate::types::{TokenUsage, ToolTier};

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

// ── Root turn span (daemon cross-task) ───────────────────────────────

/// Inputs needed to open the daemon's root `invoke_agent` span.
///
/// The daemon-hosted worker drives one turn across multiple tokio tasks
/// (execute → suspend at the tool boundary → resume). Unlike the
/// in-process loop's `agent_loop` root span — which keeps a single live
/// span wrapping the whole run future — the worker creates this span
/// live on the FRESH turn, captures its `(trace_id, span_id)` to persist
/// on the turn attempt, and re-parents every later span (resumed `chat`
/// calls, child-task `execute_tool` calls) under those ids via
/// [`remote_parent_context`].
#[derive(Clone, Copy)]
pub struct RootTurnSpanParams<'a> {
    /// Raw SDK provider id (e.g. `anthropic`); normalised to the
    /// `gen_ai.provider.name` semconv value internally.
    pub provider_id: &'static str,
    /// Model driving the turn (`gen_ai.request.model`).
    pub model: &'a str,
    /// Thread / conversation id (`gen_ai.conversation.id`).
    pub conversation_id: &'a str,
}

/// A started root-turn span plus the hex ids the worker persists so the
/// turn's later tasks can re-parent their spans under it.
pub struct StartedRootTurnSpan {
    /// The live `invoke_agent` span.
    ///
    /// Canonical lifecycle: hand this span straight to
    /// [`stash_root_turn_span`] (keyed by task id) and finalize it later —
    /// from whichever task reaches the terminal — via
    /// [`finalize_root_turn_span`], so the exported span carries the
    /// **full** turn duration across the suspend/resume hop. Only callers
    /// that deliberately opt out of the registry hold this span directly
    /// and finish it with [`finish_root_turn_span`] (or let it end on drop,
    /// which truncates the duration to the fresh segment only).
    pub span: BoxedSpan,
    /// Hex-encoded `TraceId` — persist to
    /// `agent_sdk_turn_attempts.otel_trace_id`.
    pub trace_id_hex: String,
    /// Hex-encoded root `SpanId` — persist to
    /// `agent_sdk_turn_attempts.otel_span_id`.
    pub span_id_hex: String,
    /// The root span's real sampled bit, captured from its `SpanContext`
    /// at creation.
    ///
    /// Persist this alongside the ids and pass it to
    /// [`remote_parent_context_with_sampling`] /
    /// [`traceparent_from_ids_with_sampling`] when re-parenting resumed
    /// `chat` calls and child `execute_tool` spans, so those children
    /// honour ratio sampling instead of being force-recorded under a
    /// sampled-out root.
    pub sampled: bool,
}

/// Start the daemon's root `invoke_agent` INTERNAL span and capture the
/// ids the worker persists so the turn's later tasks can nest under it.
///
/// Mirrors the structural / `gen_ai.*` attributes of the in-process
/// loop's root span (operation name, provider, model, conversation id)
/// with the minimal parameter set the daemon worker supplies cheaply,
/// and tags the same Langfuse `agent` observation.
///
/// # Lifecycle
///
/// The `OTel` 0.32 `SpanBuilder` exposes no way to assign a span id, so a
/// single span cannot be reopened on resume to cover the whole turn.
/// The **canonical** pattern is therefore:
///
/// 1. Persist [`StartedRootTurnSpan::trace_id_hex`] /
///    [`StartedRootTurnSpan::span_id_hex`] (and
///    [`StartedRootTurnSpan::sampled`]) on the turn attempt so later tasks
///    re-parent under the root via [`remote_parent_context_with_sampling`].
/// 2. Hand [`StartedRootTurnSpan::span`] to [`stash_root_turn_span`]
///    immediately, then [`finalize_root_turn_span`] it from whichever task
///    reaches the terminal — even across tasks — so the span keeps the
///    full turn duration (it legitimately stays open while tools run).
///
/// End-on-drop is reserved for callers that deliberately opt out of the
/// registry; for them the span's duration covers only the fresh segment.
/// A custom worker that skips [`stash_root_turn_span`] will export
/// truncated root spans.
#[must_use]
pub fn start_root_turn_span(params: RootTurnSpanParams<'_>) -> StartedRootTurnSpan {
    let provider = provider_name::normalize(params.provider_id);
    let span_attrs = vec![
        KeyValue::new(attrs::GEN_AI_OPERATION_NAME, "invoke_agent"),
        KeyValue::new(attrs::GEN_AI_PROVIDER_NAME, provider),
        KeyValue::new(attrs::GEN_AI_REQUEST_MODEL, params.model.to_string()),
        KeyValue::new(
            attrs::GEN_AI_CONVERSATION_ID,
            params.conversation_id.to_string(),
        ),
        KeyValue::new(attrs::SDK_PROVIDER_ID, params.provider_id),
    ];
    let mut span = spans::start_internal_span("invoke_agent", span_attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    langfuse::tag_observation(&mut span, langfuse::ObservationType::Agent);
    let span_context = span.span_context().clone();
    StartedRootTurnSpan {
        trace_id_hex: span_context.trace_id().to_string(),
        span_id_hex: span_context.span_id().to_string(),
        sampled: span_context.is_sampled(),
        span,
    }
}

/// Build a [`Context`] whose active parent is the root-turn span
/// identified by the hex ids persisted on the turn attempt.
///
/// Spans created while this context is attached become children of the
/// root span. This is how the worker re-parents resumed `chat` calls and
/// child-task `execute_tool` calls under the turn root after the original
/// live span has gone (the worker suspended, or a fresh task / process
/// picked the turn up). Returns `None` when the ids are absent or
/// malformed — the caller then leaves the span un-parented (a new trace),
/// exactly as before this wiring.
#[must_use]
pub fn remote_parent_context(trace_id_hex: &str, span_id_hex: &str) -> Option<Context> {
    let span_context = spans::remote_span_context(trace_id_hex, span_id_hex)?;
    Some(Context::current().with_remote_span_context(span_context))
}

/// Re-parent a child onto a remote span, preserving the root's real sampled bit.
///
/// Like [`remote_parent_context`] but propagates the root span's **real**
/// sampled bit (see [`StartedRootTurnSpan::sampled`]) instead of forcing
/// SAMPLED, so a `ParentBased` sampler keeps or drops the re-parented child
/// to match the root's sampling decision. Use this on the resume / child
/// tool paths to stop ratio sampling being silently defeated for daemon
/// workloads. Returns `None` when the ids are absent or malformed.
#[must_use]
pub fn remote_parent_context_with_sampling(
    trace_id_hex: &str,
    span_id_hex: &str,
    sampled: bool,
) -> Option<Context> {
    let span_context =
        spans::remote_span_context_with_sampling(trace_id_hex, span_id_hex, sampled)?;
    Some(Context::current().with_remote_span_context(span_context))
}

/// Finalize the root-turn span with run-outcome attributes + the
/// `agent_sdk.runs.outcome` counter, then end it.
///
/// Mirrors `agent_loop`'s `end_root_span`: same total-turns / usage /
/// outcome attribute set and the same outcome counter. Reached on the
/// text-only terminal path where the worker still holds the live span;
/// tool turns end the span on drop at the suspend boundary (see
/// [`start_root_turn_span`]).
pub fn finish_root_turn_span(
    span: &mut BoxedSpan,
    total_turns: usize,
    total_usage: Option<&TokenUsage>,
    outcome: &'static str,
) {
    Metrics::global()
        .runs_outcome
        .add(1, &[KeyValue::new(attrs::SDK_OUTCOME, outcome)]);
    span.set_attribute(attrs::kv_i64(
        attrs::SDK_TOTAL_TURNS,
        i64::try_from(total_turns).unwrap_or(0),
    ));
    // Usage is optional: the daemon does not aggregate per-turn token
    // usage across the suspend/resume hop, so it passes `None` and the
    // per-call `chat` spans carry the authoritative usage. The in-process
    // loop passes the cumulative `TokenUsage`.
    if let Some(usage) = total_usage {
        span.set_attribute(attrs::kv_i64(
            attrs::GEN_AI_USAGE_INPUT_TOKENS,
            i64::from(usage.input_tokens),
        ));
        span.set_attribute(attrs::kv_i64(
            attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
            i64::from(usage.output_tokens),
        ));
        span.set_attribute(attrs::kv_i64(
            attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            i64::from(usage.cached_input_tokens),
        ));
        span.set_attribute(attrs::kv_i64(
            attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
            i64::from(usage.cache_creation_input_tokens),
        ));
    }
    span.set_attribute(KeyValue::new(attrs::SDK_OUTCOME, outcome));
    if outcome == "error" {
        spans::set_span_error(span, "agent_error", "agent invocation failed");
    }
    span.end();
}

// ── W3C traceparent ⇄ Context propagation ────────────────────────────

/// W3C `traceparent` header key. The only carrier entry the daemon
/// needs to persist a parent span context durably (on `AgentTask`) and
/// rebuild it in a later task / process.
const TRACEPARENT_KEY: &str = "traceparent";

/// Rebuild an `OTel` [`Context`] from a persisted W3C `traceparent`.
///
/// The daemon decouples the gRPC call that submits work from the worker
/// that runs it (a durable task queue, possibly a different process), so
/// the inbound client trace context cannot ride an in-memory
/// [`Context`]. It is persisted as a `traceparent` string on the task
/// and rebuilt here via the globally-installed
/// [`opentelemetry::propagation::TextMapPropagator`]. Spans started while
/// the returned context is attached become children of the encoded span.
///
/// Returns `None` when `traceparent` is empty, malformed, or no
/// propagator is installed (the extracted context then carries no valid
/// span) — the caller leaves its span un-parented, exactly as before
/// this wiring.
#[must_use]
pub fn context_from_traceparent(traceparent: &str) -> Option<Context> {
    if traceparent.is_empty() {
        return None;
    }
    let mut carrier = HashMap::with_capacity(1);
    carrier.insert(TRACEPARENT_KEY.to_string(), traceparent.to_string());
    let cx = global::get_text_map_propagator(|propagator| propagator.extract(&carrier));
    if cx.span().span_context().is_valid() {
        Some(cx)
    } else {
        None
    }
}

/// Encode hex trace + span ids into a W3C `traceparent` string.
///
/// Used to stamp a child tool task's parent span: the root
/// `invoke_agent` span's `(trace_id, span_id)` — persisted on the turn
/// attempt — is encoded here and stored on the child task's
/// `otel_traceparent` so the child's `execute_tool` span nests under the
/// turn root. Returns `None` for malformed / zero ids (validated via
/// [`spans::remote_span_context`]).
///
/// **Legacy entry point — always sets the `-01` (SAMPLED) flag.** That
/// forces the child task's `execute_tool` span to be recorded even when the
/// root turn was sampled out. Prefer [`traceparent_from_ids_with_sampling`]
/// and pass the root span's real sampled bit so ratio sampling is honoured.
#[must_use]
pub fn traceparent_from_ids(trace_id_hex: &str, span_id_hex: &str) -> Option<String> {
    traceparent_from_ids_with_sampling(trace_id_hex, span_id_hex, true)
}

/// Encode hex trace + span ids into a W3C `traceparent`, stamping the
/// root span's **real** sampled bit in the flag byte (`-01` when sampled,
/// `-00` otherwise).
///
/// A downstream [`context_from_traceparent`] parses this flag through the
/// global propagator, so encoding the true bit keeps the child's sampling
/// decision aligned with the root instead of force-recording every child.
/// Returns `None` for malformed / zero ids.
#[must_use]
pub fn traceparent_from_ids_with_sampling(
    trace_id_hex: &str,
    span_id_hex: &str,
    sampled: bool,
) -> Option<String> {
    let span_context = spans::remote_span_context(trace_id_hex, span_id_hex)?;
    let flag = if sampled { "01" } else { "00" };
    Some(format!(
        "00-{}-{}-{flag}",
        span_context.trace_id(),
        span_context.span_id()
    ))
}

// ── Live root-span registry (correct cross-task duration) ────────────

/// Process-global home for the live root-turn span between the task that
/// opens it and the (possibly later) task that finalizes it.
///
/// The daemon drives one turn across several tokio tasks
/// (execute → suspend at the tool boundary → resume), and `OTel` 0.32
/// offers no way to assign a span id, so a root span cannot be re-minted
/// on resume to cover the whole turn. Instead the worker stashes the
/// *live* `invoke_agent` span here at the fresh turn and finalizes it at
/// the terminal — even though that runs in a different task — so the span
/// carries the **full** turn duration (it legitimately stays open while
/// tools run). Children never read this map; they nest via the ids
/// persisted on the turn attempt (see [`remote_parent_context`]), so the
/// tree is correct regardless of whether the live span survives.
///
/// Keyed by `AgentTask` id (stringified). An entry that is never finalized
/// in this process — the daemon crashed/restarted mid-turn, or (in a
/// horizontally-scaled deployment) the resume landed on a different replica
/// so a fresh process owns the terminal — would otherwise leak its span
/// object forever. To bound that leak the registry is swept on every
/// [`stash_root_turn_span`]: entries older than [`MAX_STASH_AGE`] (and the
/// oldest entries once the map reaches [`MAX_LIVE_ROOT_SPANS`]) are ended
/// with an `abandoned` outcome so they still export rather than vanish.
static LIVE_ROOT_SPANS: LazyLock<Mutex<HashMap<String, StashedRootSpan>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A live root-turn span plus the instant it was stashed, so the registry
/// can evict entries that were never finalized (see `LIVE_ROOT_SPANS`).
struct StashedRootSpan {
    span: BoxedSpan,
    stashed_at: Instant,
}

/// Hard cap on the number of live root-turn spans retained at once. Once
/// reached, the oldest entries are evicted (ended as `abandoned`) to make
/// room. Sized generously: real concurrency is bounded by worker leases, so
/// hitting this implies leaked (cross-replica / crashed) entries.
const MAX_LIVE_ROOT_SPANS: usize = 1024;

/// Maximum age a stashed root-turn span is kept before it is force-ended
/// with an `abandoned` outcome. An upper bound on a single turn's
/// wall-clock; anything older is a leaked entry whose terminal will never
/// arrive in this process.
const MAX_STASH_AGE: Duration = Duration::from_hours(1);

/// Remove and return every entry older than [`MAX_STASH_AGE`]. The caller
/// finalizes the returned spans (as `abandoned`) *after* releasing the
/// registry lock so span / metric work never runs under it.
fn take_stale(spans: &mut HashMap<String, StashedRootSpan>, now: Instant) -> Vec<StashedRootSpan> {
    spans
        .extract_if(|_, stashed| now.saturating_duration_since(stashed.stashed_at) >= MAX_STASH_AGE)
        .map(|(_, stashed)| stashed)
        .collect()
}

/// Remove and return the oldest entries until the map has room for one more
/// entry under `cap`. The caller finalizes the returned spans.
fn take_over_capacity(
    spans: &mut HashMap<String, StashedRootSpan>,
    cap: usize,
) -> Vec<StashedRootSpan> {
    let mut evicted = Vec::new();
    while spans.len() >= cap {
        let Some(oldest) = spans
            .iter()
            .min_by_key(|(_, stashed)| stashed.stashed_at)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if let Some(stashed) = spans.remove(&oldest) {
            evicted.push(stashed);
        }
    }
    evicted
}

/// Stash the live root-turn `span` under `task_id` so a later task can
/// finalize it with the correct full duration (see `LIVE_ROOT_SPANS`).
///
/// First write wins: a retry that re-opens the same turn keeps the
/// original span (and its start time) rather than resetting the clock.
///
/// Sweeps the registry first (TTL + capacity) so leaked entries from
/// crashed / cross-replica resumes cannot grow the map without bound; any
/// evicted span is ended with an `abandoned` outcome (see `LIVE_ROOT_SPANS`).
pub fn stash_root_turn_span(task_id: &str, span: BoxedSpan) {
    let evicted = {
        let Ok(mut spans) = LIVE_ROOT_SPANS.lock() else {
            log::warn!("live root-span registry poisoned; dropping root span for task {task_id}");
            return;
        };
        let now = Instant::now();
        let mut evicted = take_stale(&mut spans, now);
        evicted.extend(take_over_capacity(&mut spans, MAX_LIVE_ROOT_SPANS));
        spans.entry(task_id.to_string()).or_insert(StashedRootSpan {
            span,
            stashed_at: now,
        });
        evicted
    };
    // Finalize evicted (leaked) spans outside the registry lock so the
    // metric + export work never runs while holding it. Ending them with an
    // `abandoned` outcome keeps them in the trace rather than vanishing.
    for mut stashed in evicted {
        finish_root_turn_span(&mut stashed.span, 0, None, "abandoned");
    }
}

/// Finalize and end the stashed root-turn span for `task_id`, stamping
/// run-outcome + usage attributes (see [`finish_root_turn_span`]).
///
/// No-op when no span is stashed — the expected path when the daemon
/// restarted mid-turn and a fresh process owns the terminal, or when the
/// entry was already evicted by the registry's TTL / capacity sweep. The
/// outcome counter is still recorded in that case so dashboards see every
/// run.
pub fn finalize_root_turn_span(
    task_id: &str,
    total_turns: usize,
    total_usage: Option<&TokenUsage>,
    outcome: &'static str,
) {
    let stashed = LIVE_ROOT_SPANS
        .lock()
        .ok()
        .and_then(|mut spans| spans.remove(task_id));
    match stashed {
        Some(mut stashed) => {
            finish_root_turn_span(&mut stashed.span, total_turns, total_usage, outcome);
        }
        None => {
            // Live span lost (cross-restart resume). Still record the
            // run-outcome counter so the metric isn't undercounted.
            Metrics::global()
                .runs_outcome
                .add(1, &[KeyValue::new(attrs::SDK_OUTCOME, outcome)]);
        }
    }
}

/// Drop the stashed root-turn span for `task_id` without finalizing.
///
/// For paths that abandon a turn without a meaningful outcome (e.g. an
/// idempotent duplicate-suspension bail) so the registry doesn't leak.
pub fn discard_root_turn_span(task_id: &str) {
    if let Ok(mut spans) = LIVE_ROOT_SPANS.lock() {
        drop(spans.remove(task_id));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Duration, Instant, MAX_STASH_AGE, StashedRootSpan, classify_llm_error,
        discard_root_turn_span, finalize_root_turn_span, spans, stash_root_turn_span,
        take_over_capacity, take_stale, traceparent_from_ids, traceparent_from_ids_with_sampling,
    };
    use anyhow::Context as _;
    use std::collections::HashMap;

    // W3C example ids (RFC trace-context), both non-zero / valid.
    const TRACE_HEX: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
    const SPAN_HEX: &str = "00f067aa0ba902b7";

    #[test]
    fn traceparent_from_valid_ids_is_w3c_sampled() {
        let traceparent = traceparent_from_ids(TRACE_HEX, SPAN_HEX).expect("valid ids");
        assert_eq!(traceparent, format!("00-{TRACE_HEX}-{SPAN_HEX}-01"));
    }

    #[test]
    fn traceparent_encodes_real_sampled_bit() -> anyhow::Result<()> {
        let sampled =
            traceparent_from_ids_with_sampling(TRACE_HEX, SPAN_HEX, true).context("sampled")?;
        assert!(sampled.ends_with("-01"), "sampled traceparent: {sampled}");
        let unsampled =
            traceparent_from_ids_with_sampling(TRACE_HEX, SPAN_HEX, false).context("unsampled")?;
        assert!(
            unsampled.ends_with("-00"),
            "sampled-out root must not force -01: {unsampled}"
        );
        Ok(())
    }

    #[test]
    fn traceparent_from_malformed_or_zero_ids_is_none() {
        assert!(traceparent_from_ids("not-hex", SPAN_HEX).is_none());
        assert!(traceparent_from_ids(TRACE_HEX, "tooshort").is_none());
        // All-zero ids are rejected by `SpanContext::is_valid`.
        assert!(traceparent_from_ids(&"0".repeat(32), &"0".repeat(16)).is_none());
    }

    #[test]
    fn classify_llm_error_vocabulary_is_stable() {
        // Pins the daemon-side `error.type` vocabulary so any drift from the
        // in-process `agent_loop::turn::classify_llm_error` byte-for-byte
        // mirror is caught here (the two copies are not yet deduplicated).
        assert_eq!(
            classify_llm_error("Rate limited: slow down"),
            "rate_limited"
        );
        assert_eq!(
            classify_llm_error("Invalid request: bad"),
            "invalid_request"
        );
        assert_eq!(classify_llm_error("Server error 500"), "server_error");
        assert_eq!(classify_llm_error("Stream closed early"), "stream_error");
        assert_eq!(classify_llm_error("something else entirely"), "_OTHER");
    }

    #[test]
    fn take_stale_removes_entries_past_ttl() {
        let mut map: HashMap<String, StashedRootSpan> = HashMap::new();
        // Anchor everything to a single base instant and advance "now"
        // forward (rather than subtracting from `now`) so the test never
        // underflows the monotonic clock on a freshly-booted machine.
        let base = Instant::now();
        map.insert(
            "old".to_string(),
            StashedRootSpan {
                span: spans::start_internal_span("test", Vec::new()),
                stashed_at: base,
            },
        );
        map.insert(
            "fresh".to_string(),
            StashedRootSpan {
                span: spans::start_internal_span("test", Vec::new()),
                stashed_at: base + MAX_STASH_AGE,
            },
        );
        let eval_now = base + MAX_STASH_AGE + Duration::from_secs(1);
        let evicted = take_stale(&mut map, eval_now);
        assert_eq!(evicted.len(), 1, "exactly the stale entry is evicted");
        assert!(!map.contains_key("old"), "stale entry must be removed");
        assert!(map.contains_key("fresh"), "fresh entry must survive");
    }

    #[test]
    fn take_over_capacity_trims_oldest_to_make_room() {
        let mut map: HashMap<String, StashedRootSpan> = HashMap::new();
        for i in 0u64..4 {
            map.insert(
                format!("t{i}"),
                StashedRootSpan {
                    span: spans::start_internal_span("test", Vec::new()),
                    stashed_at: Instant::now() + Duration::from_millis(i),
                },
            );
        }
        let evicted = take_over_capacity(&mut map, 2);
        assert!(!evicted.is_empty(), "over-capacity entries must be evicted");
        // Leaves room for one more under the cap.
        assert!(
            map.len() < 2,
            "map should be trimmed below cap, got {}",
            map.len()
        );
    }

    #[test]
    fn live_root_span_registry_finalize_is_idempotent() {
        // No provider installed → a no-op span, but the registry's map
        // management (stash → finalize removes the entry) is exercised
        // regardless, and a second finalize must not panic.
        let task_id = "registry-roundtrip-task";
        stash_root_turn_span(task_id, spans::start_internal_span("test", Vec::new()));
        finalize_root_turn_span(task_id, 1, None, "done");
        finalize_root_turn_span(task_id, 1, None, "done");
    }

    #[test]
    fn discard_removes_stashed_span_without_finalize() {
        let task_id = "registry-discard-task";
        stash_root_turn_span(task_id, spans::start_internal_span("test", Vec::new()));
        discard_root_turn_span(task_id);
        // Finalize after discard is a no-op (entry already gone).
        finalize_root_turn_span(task_id, 0, None, "cancelled");
    }
}
