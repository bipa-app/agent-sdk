//! Span construction and lifecycle helpers.

use std::borrow::Cow;

use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::trace::{Span, SpanKind, Status, Tracer};
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
