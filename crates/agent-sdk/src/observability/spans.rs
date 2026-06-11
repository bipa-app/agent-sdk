//! Span construction and lifecycle helpers.

use std::borrow::Cow;

use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::trace::{
    Span, SpanContext, SpanId, SpanKind, Status, TraceFlags, TraceId, TraceState, Tracer,
};
use opentelemetry::{InstrumentationScope, KeyValue};

use super::types::CaptureDecision;

const TRACER_NAME: &str = env!("CARGO_PKG_NAME");
const TRACER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get the SDK tracer from the global provider.
///
/// Fetched fresh each time to avoid binding to a no-op if the application
/// installs its provider after the SDK initialises.
fn tracer() -> BoxedTracer {
    let scope = InstrumentationScope::builder(TRACER_NAME)
        .with_version(TRACER_VERSION)
        .build();
    global::tracer_with_scope(scope)
}

/// Start an `INTERNAL` span with the given name and attributes.
#[must_use]
pub fn start_internal_span(name: impl Into<Cow<'static, str>>, attrs: Vec<KeyValue>) -> BoxedSpan {
    let t = tracer();
    t.span_builder(name)
        .with_kind(SpanKind::Internal)
        .with_attributes(attrs)
        .start(&t)
}

/// Start a `CLIENT` span with the given name and attributes.
#[must_use]
pub fn start_client_span(name: impl Into<Cow<'static, str>>, attrs: Vec<KeyValue>) -> BoxedSpan {
    let t = tracer();
    t.span_builder(name)
        .with_kind(SpanKind::Client)
        .with_attributes(attrs)
        .start(&t)
}

/// Set span status to error with a message and `error.type` attribute.
pub fn set_span_error(span: &mut BoxedSpan, error_type: &str, message: &str) {
    span.set_attribute(KeyValue::new(
        super::attrs::ERROR_TYPE,
        error_type.to_string(),
    ));
    span.set_status(Status::error(message.to_string()));
}

/// Add a structured span event with attributes.
///
/// Skips silently when the span is not recording (sampling drop or
/// no-op tracer). Mirrors the guards used by every other helper in
/// this module so callers don't have to check `is_recording` first.
pub fn add_event(span: &mut BoxedSpan, name: impl Into<Cow<'static, str>>, attrs: Vec<KeyValue>) {
    if !span.is_recording() {
        return;
    }
    span.add_event(name, attrs);
}

/// Add an `OTel` span link from `span` to `target`, carrying `attrs`.
///
/// Skips silently when the span is not recording, when `target` is
/// not a valid `SpanContext` (zero trace/span ids), or when no tracer
/// provider is installed.  This mirrors [`add_event`] so call sites
/// don't have to gate every link emission with `is_recording` checks.
pub fn add_link(span: &mut BoxedSpan, target: SpanContext, attrs: Vec<KeyValue>) {
    if !span.is_recording() {
        return;
    }
    if !target.is_valid() {
        return;
    }
    span.add_link(target, attrs);
}

/// Add a `replay-of` link pointing at the original attempt's span.
///
/// `original_trace_id` and `original_span_id` are hex-encoded as
/// stored on `agent_sdk_turn_attempts.{otel_trace_id,otel_span_id}`.
/// Malformed hex values are treated as "no link" so a corrupt journal
/// row never poisons the live span.
///
/// `attempt_index` is 1-based and matches
/// `TurnAttempt::attempt_number` so cross-trace queries can join on it.
pub fn link_to_replay_origin(
    span: &mut BoxedSpan,
    original_trace_id: &str,
    original_span_id: &str,
    attempt_index: u32,
) {
    let Some(target) =
        span_context_from_hex(original_trace_id, original_span_id, TraceFlags::SAMPLED)
    else {
        return;
    };
    add_link(
        span,
        target,
        vec![
            KeyValue::new(
                super::attrs::AGENT_REPLAY_ORIGINAL_TRACE_ID,
                original_trace_id.to_string(),
            ),
            KeyValue::new(
                super::attrs::AGENT_REPLAY_ORIGINAL_SPAN_ID,
                original_span_id.to_string(),
            ),
            super::attrs::kv_i64(
                super::attrs::AGENT_REPLAY_ATTEMPT_INDEX,
                i64::from(attempt_index),
            ),
        ],
    );
}

/// Add a `subagent-of` link pointing at the parent turn's span.
///
/// Even though parent and subagent share an `OTel` context today, the
/// explicit link makes the relationship queryable when one of the
/// spans is dropped by tail sampling.  Malformed ids are silently
/// dropped (see [`link_to_replay_origin`]).
pub fn link_to_parent_turn(span: &mut BoxedSpan, parent_trace_id: &str, parent_span_id: &str) {
    let Some(target) = span_context_from_hex(parent_trace_id, parent_span_id, TraceFlags::SAMPLED)
    else {
        return;
    };
    add_link(span, target, vec![]);
}

/// Build a remote `SpanContext` from hex-encoded trace + span ids, for
/// re-parenting spans under a span that is no longer live in the
/// current task.
///
/// The daemon-hosted worker drives a turn across multiple tasks
/// (execute → suspend at the tool boundary → resume), so its root
/// `invoke_agent` span cannot stay live for the whole turn. The worker
/// persists the root span's `(trace_id, span_id)` and rebuilds a remote
/// parent context from them via this helper so resumed `chat` calls and
/// child-task `execute_tool` calls nest under the turn root. Returns
/// `None` for malformed / zero ids (treated as "no parent").
///
/// **Legacy entry point — assumes the remote parent was sampled.** It
/// always marks the reconstructed context SAMPLED, which forces a
/// `ParentBased` sampler to record every re-parented child even when the
/// root span was sampled out. Prefer [`remote_span_context_with_sampling`]
/// and pass the root span's real sampled bit so ratio sampling configured
/// for the root is honoured by its children.
#[must_use]
pub fn remote_span_context(trace_hex: &str, span_hex: &str) -> Option<SpanContext> {
    span_context_from_hex(trace_hex, span_hex, TraceFlags::SAMPLED)
}

/// Build a remote `SpanContext` that carries the root span's **real**
/// sampled bit.
///
/// A `ParentBased` sampler decides whether to record a child span from its
/// remote parent's sampled flag. Forcing SAMPLED (see
/// [`remote_span_context`]) therefore defeats ratio sampling: a child of a
/// sampled-out root would still be recorded and exported, orphaned from a
/// parent that was never exported. Passing the parent's actual `sampled`
/// state keeps the child's sampling decision aligned with the root's.
///
/// Returns `None` for malformed / zero ids (treated as "no parent").
#[must_use]
pub fn remote_span_context_with_sampling(
    trace_hex: &str,
    span_hex: &str,
    sampled: bool,
) -> Option<SpanContext> {
    span_context_from_hex(trace_hex, span_hex, sampled_flags(sampled))
}

const fn sampled_flags(sampled: bool) -> TraceFlags {
    if sampled {
        TraceFlags::SAMPLED
    } else {
        TraceFlags::NOT_SAMPLED
    }
}

/// Build a `SpanContext` from hex-encoded trace + span ids with explicit
/// trace flags.
///
/// Returns `None` when either id is malformed or zero (`TraceId::INVALID`
/// / `SpanId::INVALID`). The constructed context is marked
/// `is_remote = true`.
fn span_context_from_hex(
    trace_hex: &str,
    span_hex: &str,
    flags: TraceFlags,
) -> Option<SpanContext> {
    let trace_id = TraceId::from_hex(trace_hex).ok()?;
    let span_id = SpanId::from_hex(span_hex).ok()?;
    let ctx = SpanContext::new(trace_id, span_id, flags, true, TraceState::default());
    if !ctx.is_valid() {
        return None;
    }
    Some(ctx)
}

/// Record payload content on an LLM span based on store decisions.
pub fn record_payload_on_span(
    span: &mut BoxedSpan,
    result: &super::types::CaptureResult,
    system_json: Option<&serde_json::Value>,
    input_json: &serde_json::Value,
    output_json: &serde_json::Value,
) {
    use super::attrs;

    if !span.is_recording() {
        return;
    }

    apply_capture_decision(
        span,
        &result.system_instructions,
        system_json,
        attrs::GEN_AI_SYSTEM_INSTRUCTIONS,
        attrs::SDK_OTEL_SYSTEM_INSTRUCTIONS_REF,
    );
    apply_capture_decision(
        span,
        &result.input_messages,
        Some(input_json),
        attrs::GEN_AI_INPUT_MESSAGES,
        attrs::SDK_OTEL_INPUT_MESSAGES_REF,
    );
    apply_capture_decision(
        span,
        &result.output_messages,
        Some(output_json),
        attrs::GEN_AI_OUTPUT_MESSAGES,
        attrs::SDK_OTEL_OUTPUT_MESSAGES_REF,
    );
}

fn apply_capture_decision(
    span: &mut BoxedSpan,
    decision: &CaptureDecision,
    json_value: Option<&serde_json::Value>,
    inline_attr: &'static str,
    ref_attr: &'static str,
) {
    match decision {
        CaptureDecision::Inline => {
            if let Some(val) = json_value {
                span.set_attribute(KeyValue::new(inline_attr, val.to_string()));
            }
        }
        CaptureDecision::Reference(r) => {
            span.set_attribute(KeyValue::new(ref_attr, r.clone()));
        }
        CaptureDecision::Omit => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{remote_span_context, remote_span_context_with_sampling};
    use anyhow::Context as _;

    // Valid W3C example ids (RFC trace-context), both non-zero.
    const TRACE_HEX: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
    const SPAN_HEX: &str = "00f067aa0ba902b7";

    #[test]
    fn remote_span_context_honours_real_sampled_bit() -> anyhow::Result<()> {
        let kept =
            remote_span_context_with_sampling(TRACE_HEX, SPAN_HEX, true).context("sampled ctx")?;
        assert!(kept.is_sampled(), "sampled=true must produce a sampled ctx");
        assert!(kept.is_remote());

        let dropped = remote_span_context_with_sampling(TRACE_HEX, SPAN_HEX, false)
            .context("unsampled ctx")?;
        assert!(
            !dropped.is_sampled(),
            "sampled=false must NOT force the SAMPLED flag (ratio sampling respected)"
        );
        assert!(dropped.is_remote());
        Ok(())
    }

    #[test]
    fn legacy_remote_span_context_stays_sampled() -> anyhow::Result<()> {
        let ctx = remote_span_context(TRACE_HEX, SPAN_HEX).context("ctx")?;
        assert!(ctx.is_sampled());
        Ok(())
    }

    #[test]
    fn remote_span_context_rejects_zero_ids() {
        assert!(
            remote_span_context_with_sampling(&"0".repeat(32), &"0".repeat(16), true).is_none()
        );
        assert!(remote_span_context_with_sampling("not-hex", SPAN_HEX, false).is_none());
    }
}
