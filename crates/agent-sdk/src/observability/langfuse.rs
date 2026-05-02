//! Native Langfuse attribute helpers.
//!
//! Owns every `langfuse.*` attribute key and the
//! [`tag_observation`] helper that stamps a span with its
//! `langfuse.observation.type`. A vanilla SDK consumer who installs
//! OpenTelemetry and points the OTLP exporter at Langfuse therefore
//! gets a fully-tagged trace tree (Agent / Generation / Tool / Chain /
//! …) without writing any glue.
//!
//! Spec: <https://langfuse.com/integrations/native/opentelemetry#property-mapping>
//!
//! These helpers are pure attribute setters. They do **not** apply
//! redaction — callers are responsible for passing already-redacted
//! values when the attribute may contain user payload (the
//! observability boundary's mandatory redactor lives in C1; the
//! default-deny capture switch lives in C2).
//!
//! All setters guard on [`Span::is_recording`] and skip silently when
//! sampling has dropped the span, mirroring the behaviour of the
//! sibling baggage helpers.

use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;
use opentelemetry::{Array, KeyValue, StringValue, Value};

// ── Trace-level keys ─────────────────────────────────────────────────

/// `langfuse.trace.name` — display name of the trace in the Langfuse UI.
pub const LANGFUSE_TRACE_NAME: &str = "langfuse.trace.name";

/// `langfuse.trace.input` — JSON or text shown as the trace's user input.
pub const LANGFUSE_TRACE_INPUT: &str = "langfuse.trace.input";

/// `langfuse.trace.output` — JSON or text shown as the trace's final output.
pub const LANGFUSE_TRACE_OUTPUT: &str = "langfuse.trace.output";

/// `langfuse.trace.tags` — array of free-form labels attached to the trace.
pub const LANGFUSE_TRACE_TAGS: &str = "langfuse.trace.tags";

/// `langfuse.trace.public` — boolean controlling Langfuse's "share" toggle.
pub const LANGFUSE_TRACE_PUBLIC: &str = "langfuse.trace.public";

/// Prefix for trace-level metadata keys (`langfuse.trace.metadata.<key>`).
pub const LANGFUSE_TRACE_METADATA_PREFIX: &str = "langfuse.trace.metadata.";

/// `langfuse.session.id` — Langfuse session id; mirrors the W3C baggage entry
/// of the same name when present.
pub const LANGFUSE_SESSION_ID: &str = "langfuse.session.id";

/// `langfuse.user.id` — Langfuse end-user id; mirrors the W3C baggage entry
/// of the same name when present.
pub const LANGFUSE_USER_ID: &str = "langfuse.user.id";

/// `langfuse.release` — release identifier for the trace's build.
pub const LANGFUSE_RELEASE: &str = "langfuse.release";

/// `langfuse.version` — application or prompt version associated with the trace.
pub const LANGFUSE_VERSION: &str = "langfuse.version";

/// `langfuse.environment` — Langfuse environment slug (`prod`, `staging`, …).
pub const LANGFUSE_ENVIRONMENT: &str = "langfuse.environment";

// ── Observation-level keys ───────────────────────────────────────────

/// `langfuse.observation.type` — the observation kind tag (see [`ObservationType`]).
pub const LANGFUSE_OBSERVATION_TYPE: &str = "langfuse.observation.type";

/// `langfuse.observation.input` — JSON or text input recorded against an observation.
pub const LANGFUSE_OBSERVATION_INPUT: &str = "langfuse.observation.input";

/// `langfuse.observation.output` — JSON or text output recorded against an observation.
pub const LANGFUSE_OBSERVATION_OUTPUT: &str = "langfuse.observation.output";

/// `langfuse.observation.level` — severity level (`DEBUG` / `DEFAULT` / `WARNING` / `ERROR`).
pub const LANGFUSE_OBSERVATION_LEVEL: &str = "langfuse.observation.level";

/// `langfuse.observation.status_message` — free-form failure context for status panels.
pub const LANGFUSE_OBSERVATION_STATUS_MESSAGE: &str = "langfuse.observation.status_message";

/// `langfuse.observation.usage_details` — JSON token-usage breakdown.
pub const LANGFUSE_OBSERVATION_USAGE_DETAILS: &str = "langfuse.observation.usage_details";

/// `langfuse.observation.cost_details` — JSON cost breakdown.
pub const LANGFUSE_OBSERVATION_COST_DETAILS: &str = "langfuse.observation.cost_details";

/// `langfuse.observation.model.name` — Langfuse pricing-model identifier.
pub const LANGFUSE_OBSERVATION_MODEL_NAME: &str = "langfuse.observation.model.name";

/// `langfuse.observation.prompt.name` — linked Langfuse prompt name.
pub const LANGFUSE_OBSERVATION_PROMPT_NAME: &str = "langfuse.observation.prompt.name";

/// `langfuse.observation.prompt.version` — linked Langfuse prompt version.
pub const LANGFUSE_OBSERVATION_PROMPT_VERSION: &str = "langfuse.observation.prompt.version";

/// Prefix for observation-level metadata keys (`langfuse.observation.metadata.<key>`).
pub const LANGFUSE_OBSERVATION_METADATA_PREFIX: &str = "langfuse.observation.metadata.";

// ── Constants ────────────────────────────────────────────────────────

/// Default character ceiling for trace-level free-text attributes.
///
/// Mirrors Bipa's `truncate_langfuse_trace_text` so that the SDK and
/// Bipa converge on the same number when E2 deletes the Bipa helper.
pub const DEFAULT_TRACE_TEXT_MAX_CHARS: usize = 10_000;

// ── ObservationType ──────────────────────────────────────────────────

/// Langfuse observation kinds — the value written under
/// [`LANGFUSE_OBSERVATION_TYPE`].
///
/// Catalog: <https://langfuse.com/docs/observability/features/observation-types>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationType {
    /// Generic span (default fallback).
    Span,
    /// LLM generation — chat / completion call.
    Generation,
    /// Agent invocation — root of an agent loop.
    Agent,
    /// Tool execution.
    Tool,
    /// Multi-step chain (compaction, MCP control-plane calls, …).
    Chain,
    /// Vector / document retriever step.
    Retriever,
    /// LLM-as-judge or other automated evaluator.
    Evaluator,
    /// Embedding generation step.
    Embedding,
    /// Policy / safety guardrail check.
    Guardrail,
    /// Discrete event marker.
    Event,
}

impl ObservationType {
    /// The wire string Langfuse expects under `langfuse.observation.type`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Span => "span",
            Self::Generation => "generation",
            Self::Agent => "agent",
            Self::Tool => "tool",
            Self::Chain => "chain",
            Self::Retriever => "retriever",
            Self::Evaluator => "evaluator",
            Self::Embedding => "embedding",
            Self::Guardrail => "guardrail",
            Self::Event => "event",
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Stamp a span with `langfuse.observation.type = <ty.as_str()>`.
///
/// Skips when the span is not recording (sampling drop) so callers
/// don't have to guard.
pub fn tag_observation(span: &mut BoxedSpan, ty: ObservationType) {
    if !span.is_recording() {
        return;
    }
    span.set_attribute(KeyValue::new(LANGFUSE_OBSERVATION_TYPE, ty.as_str()));
}

/// Set `langfuse.trace.name` on the span.
pub fn set_trace_name(span: &mut BoxedSpan, value: impl Into<String>) {
    set_string_attribute(span, LANGFUSE_TRACE_NAME, value);
}

/// Set `langfuse.trace.input` on the span.
///
/// The caller is responsible for redaction; this helper is a pure
/// attribute setter.
pub fn set_trace_input(span: &mut BoxedSpan, value: impl Into<String>) {
    set_string_attribute(span, LANGFUSE_TRACE_INPUT, value);
}

/// Set `langfuse.trace.output` on the span.
///
/// The caller is responsible for redaction; this helper is a pure
/// attribute setter.
pub fn set_trace_output(span: &mut BoxedSpan, value: impl Into<String>) {
    set_string_attribute(span, LANGFUSE_TRACE_OUTPUT, value);
}

/// Set `langfuse.release` on the span.
pub fn set_release(span: &mut BoxedSpan, value: impl Into<String>) {
    set_string_attribute(span, LANGFUSE_RELEASE, value);
}

/// Set `langfuse.environment` on the span.
pub fn set_environment(span: &mut BoxedSpan, value: impl Into<String>) {
    set_string_attribute(span, LANGFUSE_ENVIRONMENT, value);
}

/// Set a `langfuse.trace.metadata.<key>` entry on the span.
///
/// `key` must be the unprefixed metadata key. The helper concatenates
/// `LANGFUSE_TRACE_METADATA_PREFIX + key` so callers cannot
/// accidentally collide with reserved trace-level keys.
pub fn set_trace_metadata(span: &mut BoxedSpan, key: &str, value: impl Into<String>) {
    if !span.is_recording() {
        return;
    }
    let attr_key = format!("{LANGFUSE_TRACE_METADATA_PREFIX}{key}");
    span.set_attribute(KeyValue::new(attr_key, value.into()));
}

/// Set `langfuse.trace.tags` as a string array on the span.
///
/// Langfuse parses this attribute as a list when it arrives as an
/// `OTel` string array, so this helper writes
/// `Value::Array(Array::String(..))` rather than a comma-joined string.
pub fn set_trace_tags(span: &mut BoxedSpan, tags: &[String]) {
    if !span.is_recording() {
        return;
    }
    let values: Vec<StringValue> = tags.iter().cloned().map(StringValue::from).collect();
    span.set_attribute(KeyValue::new(
        LANGFUSE_TRACE_TAGS,
        Value::Array(Array::String(values)),
    ));
}

/// Truncate a string to a max char count (UTF-8 safe), appending `…` on
/// overflow.
///
/// * `("", _)` returns an empty string.
/// * `(s, 0)` returns an empty string.
/// * `(s, n)` where `s.chars().count() <= n` returns `s.to_string()`
///   unchanged.
/// * `(s, 1)` for an over-long input returns `"…"`.
/// * Otherwise the result is `s.chars().take(n - 1).collect::<String>() + "…"`.
///
/// This mirrors Bipa's `truncate_langfuse_trace_text` helper so the
/// Bipa-side function becomes a deletion candidate in E2.
#[must_use]
pub fn truncate_trace_text(text: &str, max_chars: usize) -> String {
    if text.is_empty() || max_chars == 0 {
        return String::new();
    }
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars == 1 {
        return "…".to_string();
    }
    let mut out: String = text.chars().take(max_chars - 1).collect();
    out.push('…');
    out
}

fn set_string_attribute(span: &mut BoxedSpan, key: &'static str, value: impl Into<String>) {
    if !span.is_recording() {
        return;
    }
    span.set_attribute(KeyValue::new(key, value.into()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_type_as_str_round_trip() {
        let cases = [
            (ObservationType::Span, "span"),
            (ObservationType::Generation, "generation"),
            (ObservationType::Agent, "agent"),
            (ObservationType::Tool, "tool"),
            (ObservationType::Chain, "chain"),
            (ObservationType::Retriever, "retriever"),
            (ObservationType::Evaluator, "evaluator"),
            (ObservationType::Embedding, "embedding"),
            (ObservationType::Guardrail, "guardrail"),
            (ObservationType::Event, "event"),
        ];
        for (variant, expected) in cases {
            assert_eq!(variant.as_str(), expected);
        }
    }

    #[test]
    fn truncate_trace_text_returns_empty_on_empty_input() {
        assert_eq!(truncate_trace_text("", 100), "");
        assert_eq!(truncate_trace_text("", 0), "");
    }

    #[test]
    fn truncate_trace_text_returns_empty_when_max_is_zero() {
        assert_eq!(truncate_trace_text("anything", 0), "");
    }

    #[test]
    fn truncate_trace_text_no_truncation_when_short() {
        assert_eq!(truncate_trace_text("hello", 10), "hello");
        assert_eq!(truncate_trace_text("hello", 5), "hello");
    }

    #[test]
    fn truncate_trace_text_max_one_returns_ellipsis_for_overlong_input() {
        assert_eq!(truncate_trace_text("hello", 1), "…");
    }

    #[test]
    fn truncate_trace_text_handles_multibyte_chars() {
        // Mix of ASCII, emoji, and CJK so that `chars()` and byte-indexing
        // disagree. A naive byte-slice would panic here.
        let input = "ab😀汉字cd";
        assert_eq!(input.chars().count(), 7);

        let truncated = truncate_trace_text(input, 5);
        // 4 source chars + ellipsis = 5 chars total.
        assert_eq!(truncated.chars().count(), 5);
        assert!(truncated.ends_with('…'));
        assert_eq!(truncated, "ab😀汉…");
    }

    #[test]
    fn truncate_trace_text_default_ceiling_is_ten_thousand() {
        assert_eq!(DEFAULT_TRACE_TEXT_MAX_CHARS, 10_000);
    }
}
