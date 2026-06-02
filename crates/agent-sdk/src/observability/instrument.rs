//! Instrumentation helpers called from agent loop code paths.

use agent_sdk_core::privacy::{RedactionPolicy, redact_string};
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span, TraceContextExt};
use opentelemetry::{Context, KeyValue};

use super::attrs;
use super::baggage;
use super::langfuse;
use super::provider_name;
use super::spans;
use super::trace_io;
use crate::llm::LlmProvider;
use crate::tools::ToolRegistry;
use crate::types::{AgentConfig, AgentInput, RunOptions, ThreadId, TokenUsage};

/// Parameters for [`start_root_span`].
///
/// Bundled into a struct so that the call site stays under the
/// clippy `too_many_arguments` threshold even after A5 added two new
/// fields (`run_options`, `input`).
#[derive(Clone, Copy)]
pub(crate) struct StartRootSpanParams<'a, Ctx, P>
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider,
{
    pub(crate) provider: &'a P,
    pub(crate) tools: &'a ToolRegistry<Ctx>,
    pub(crate) config: &'a AgentConfig,
    pub(crate) thread_id: &'a ThreadId,
    pub(crate) input: &'a AgentInput,
    pub(crate) run_mode: &'static str,
    pub(crate) run_options: &'a RunOptions,
}

/// Result of starting the root `invoke_agent` span.
///
/// The flat `is_recording` flag is captured eagerly because the loop
/// needs to decide whether to build a [`trace_io::RootTraceState`]
/// for streamed events without re-introspecting the span every time.
pub(crate) struct StartedRootSpan {
    pub(crate) sink: RootSpanEventSink,
    pub(crate) span_context: opentelemetry::trace::SpanContext,
    pub(crate) is_recording: bool,
}

/// Start the root `invoke_agent` span.
///
/// The function:
///
/// 1. Builds the structural / `gen_ai.*` / `agent_sdk.*` attributes.
/// 2. Temporarily attaches a baggage-augmented [`Context`] (built
///    from [`RunOptions`]) so the span — and the
///    [`baggage::copy_baggage_to_active_span`] helper that runs
///    inside this function — see the run's session/user/environment
///    entries.
/// 3. Starts an `INTERNAL` span and tags it as a Langfuse `agent`
///    observation.
/// 4. Stamps `langfuse.trace.*` and `langfuse.{release,environment}`
///    on the root span.
/// 5. Computes `langfuse.trace.input` (after PII redaction) and
///    stamps it on the span.
///
/// Baggage propagation to **child** spans is handled by the caller —
/// see [`build_root_context`] which produces the `Context` the
/// caller wraps the agent loop's future with via
/// [`opentelemetry::trace::FutureExt::with_context`].
pub(crate) fn start_root_span<Ctx, P>(params: &StartRootSpanParams<'_, Ctx, P>) -> StartedRootSpan
where
    Ctx: Send + Sync + 'static,
    P: LlmProvider,
{
    let StartRootSpanParams {
        provider,
        tools,
        config,
        thread_id,
        input,
        run_mode,
        run_options,
    } = *params;

    let span_attrs = build_root_attrs(provider, tools, config, thread_id, input, run_mode);

    // Temporarily attach the baggage-augmented context so that
    // `copy_baggage_to_active_span` (and any other helper that
    // reads `Context::current()`) sees the run's baggage entries.
    // The guard detaches at the end of this function — the run's
    // future then re-attaches the same baggage via `with_context`
    // (see `build_root_context`).
    let baggage_cx = run_options_baggage(&Context::current(), run_options);
    let _baggage_guard = baggage_cx.attach();

    let mut span = spans::start_internal_span("invoke_agent", span_attrs);
    super::baggage::copy_baggage_to_active_span(&mut span);
    super::langfuse::tag_observation(&mut span, super::langfuse::ObservationType::Agent);

    apply_run_options_attrs(&mut span, input, run_options);

    let is_recording = span.is_recording();
    let span_context = span.span_context().clone();
    StartedRootSpan {
        sink: RootSpanEventSink::new(span),
        span_context,
        is_recording,
    }
}

fn build_root_attrs<Ctx, P>(
    provider: &P,
    tools: &ToolRegistry<Ctx>,
    config: &AgentConfig,
    thread_id: &ThreadId,
    input: &AgentInput,
    run_mode: &'static str,
) -> Vec<KeyValue>
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
    span_attrs
}

/// Build a fresh [`Context`] for the run's future:
///
/// 1. starts from the caller's ambient context;
/// 2. layers in the baggage entries derived from [`RunOptions`] so
///    every child span emitted under this context inherits them via
///    [`baggage::copy_baggage_to_active_span`];
/// 3. attaches the supplied root span context so child spans become
///    its children.
///
/// The caller passes the result to
/// [`opentelemetry::trace::FutureExt::with_context`] when polling
/// the agent-loop future.
#[must_use]
pub(crate) fn build_root_context(
    span_context: opentelemetry::trace::SpanContext,
    run_options: &RunOptions,
) -> Context {
    let cx = run_options_baggage(&Context::current(), run_options);
    cx.with_remote_span_context(span_context)
}

/// Layer a [`RootSpanEventSink`] onto the supplied context.
///
/// Inner code paths (turn / tool execution / agent loop) call
/// [`record_root_event`] to add events to the root `invoke_agent`
/// span without holding a reference to the boxed span. The sink is
/// retrieved from `Context::current()` and dispatches to the locked
/// span.
#[must_use]
pub(crate) fn attach_root_event_sink(cx: &Context, sink: RootSpanEventSink) -> Context {
    cx.with_value(sink)
}

/// Add an event to the root `invoke_agent` span on the **current**
/// `OTel` context, if a sink has been attached via
/// [`attach_root_event_sink`].
///
/// No-op when no sink is in scope. Used by call sites deep in the
/// agent loop (cancellation paths, max-turn enforcement,
/// context-window-exceeded handling) so they don't have to thread
/// the boxed span through every parameter struct.
pub(crate) fn record_root_event(name: &'static str, attrs: Vec<KeyValue>) {
    if let Some(sink) = Context::current().get::<RootSpanEventSink>() {
        sink.add_event(name, attrs);
    }
}

/// Send-able handle to the root span used to attach events from
/// other tasks under the same context.
///
/// Wraps the boxed span behind an `Arc<Mutex<_>>` so the sink can be
/// stored in an `OTel` context (which requires `Clone + Send + Sync`)
/// without copying the span.
#[derive(Clone)]
pub(crate) struct RootSpanEventSink(
    std::sync::Arc<std::sync::Mutex<opentelemetry::global::BoxedSpan>>,
);

impl RootSpanEventSink {
    /// Build a sink that drains into `span`.
    pub(crate) fn new(span: opentelemetry::global::BoxedSpan) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(span)))
    }

    /// Reclaim ownership of the wrapped span at the end of the run.
    ///
    /// Returns `None` when there are still outstanding clones of the
    /// `Arc`, meaning some child task captured the sink and outlived
    /// the run. Callers fall back to emitting events on the orphaned
    /// span via [`add_event`](Self::add_event); see `end_root_span`.
    pub(crate) fn into_inner(self) -> Option<opentelemetry::global::BoxedSpan> {
        let Self(arc) = self;
        std::sync::Arc::try_unwrap(arc)
            .ok()
            .and_then(|mu| mu.into_inner().ok())
    }

    fn add_event(&self, name: &'static str, attrs: Vec<KeyValue>) {
        let Ok(mut span) = self.0.lock() else {
            log::warn!("root span sink mutex poisoned; dropping event {name}");
            return;
        };
        if !span.is_recording() {
            return;
        }
        span.add_event(name, attrs);
    }

    /// Run `op` against the wrapped span while holding the inner
    /// mutex. Used by [`flush_root_trace_state`] so the
    /// `RootTraceState` can stamp its accumulated narrative on the
    /// span without taking ownership of the sink.
    pub(crate) fn with_span_mut<R>(
        &self,
        op: impl FnOnce(&mut opentelemetry::global::BoxedSpan) -> R,
    ) -> Option<R> {
        let mut span = self.0.lock().ok()?;
        Some(op(&mut span))
    }
}

/// Mirror the session / user / environment IDs from [`RunOptions`]
/// into baggage on `cx`, returning the new context.
///
/// Sets both the canonical W3C key (`session.id`, `user.id`) and the
/// Langfuse-specific alias (`langfuse.session.id`,
/// `langfuse.user.id`) so downstream consumers that filter by either
/// see the value.
fn run_options_baggage(cx: &Context, opts: &RunOptions) -> Context {
    let mut entries: Vec<KeyValue> = Vec::new();
    if let Some(session) = opts.session_id.as_deref() {
        let value = session.to_owned();
        entries.push(KeyValue::new(baggage::BAGGAGE_SESSION_ID, value.clone()));
        entries.push(KeyValue::new(baggage::BAGGAGE_LANGFUSE_SESSION_ID, value));
    }
    if let Some(user) = opts.user_id.as_deref() {
        let value = user.to_owned();
        entries.push(KeyValue::new(baggage::BAGGAGE_USER_ID, value.clone()));
        entries.push(KeyValue::new(baggage::BAGGAGE_LANGFUSE_USER_ID, value));
    }
    if let Some(env) = opts.environment.as_deref() {
        entries.push(KeyValue::new(
            baggage::BAGGAGE_DEPLOYMENT_ENVIRONMENT,
            env.to_owned(),
        ));
    }
    if entries.is_empty() {
        cx.clone()
    } else {
        baggage::with_attributes(cx, entries)
    }
}

/// Stamp the `langfuse.trace.*`, `langfuse.release`,
/// `langfuse.environment`, and `langfuse.trace.input` attributes
/// derived from [`RunOptions`] onto the root span.
fn apply_run_options_attrs(span: &mut BoxedSpan, input: &AgentInput, opts: &RunOptions) {
    if let Some(name) = opts.trace_name.as_deref() {
        langfuse::set_trace_name(span, name);
    }
    if !opts.trace_tags.is_empty() {
        langfuse::set_trace_tags(span, &opts.trace_tags);
    }
    for (key, value) in &opts.trace_metadata {
        let stringified = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        langfuse::set_trace_metadata(span, key, stringified);
    }
    if let Some(release) = opts.release.as_deref() {
        langfuse::set_release(span, release);
    }
    if let Some(env) = opts.environment.as_deref() {
        langfuse::set_environment(span, env);
    }

    let max_chars = opts
        .trace_text_max_chars
        .unwrap_or(langfuse::DEFAULT_TRACE_TEXT_MAX_CHARS);
    if let Some(trace_input) = trace_io::langfuse_trace_input(input, max_chars) {
        let masked = redact_string(&trace_input, &RedactionPolicy::baseline());
        langfuse::set_trace_input(span, masked);
    }
}

/// Build the [`trace_io::RootTraceState`] that the loop pumps from
/// every emitted [`crate::events::AgentEvent`].
///
/// Returns `None` when the supplied span is not recording (sampling
/// drop) — the loop will skip the trace-output update entirely.
///
/// The state owns only the running text buffer; the loop is
/// responsible for calling
/// [`trace_io::RootTraceState::flush`] on the still-live span
/// before [`end_root_span`] is invoked, so the `OTel` SDK records
/// exactly one `langfuse.trace.output` attribute (Rust SDK 0.31's
/// `set_attribute` appends to a Vec without deduplicating by key).
#[must_use]
pub(crate) fn build_root_trace_state(
    is_recording: bool,
    run_options: &RunOptions,
) -> Option<std::sync::Arc<trace_io::RootTraceState>> {
    if !is_recording {
        return None;
    }
    let max_chars = run_options
        .trace_text_max_chars
        .unwrap_or(langfuse::DEFAULT_TRACE_TEXT_MAX_CHARS);
    Some(std::sync::Arc::new(trace_io::RootTraceState::new(
        max_chars,
    )))
}

/// Finalize the root span with outcome attributes and end it.
///
/// The loop is expected to have already called
/// [`trace_io::RootTraceState::flush`] before calling this, so the
/// final `langfuse.trace.output` is a single attribute on the
/// exported span.
pub(crate) fn end_root_span(
    sink: RootSpanEventSink,
    total_turns: usize,
    total_usage: &TokenUsage,
    outcome: &'static str,
) {
    // Record the run-outcome counter unconditionally so dashboards
    // see every run — even sampled-out ones — without depending on
    // the live span.
    let metrics = super::metrics::Metrics::global();
    metrics
        .runs_outcome
        .add(1, &[KeyValue::new(attrs::SDK_OUTCOME, outcome)]);

    let Some(mut span) = sink.into_inner() else {
        log::warn!(
            "root span sink still has outstanding clones at end_root_span; \
             dropping outcome attributes",
        );
        return;
    };
    span.set_attribute(KeyValue::new(
        attrs::SDK_TOTAL_TURNS,
        i64::try_from(total_turns).unwrap_or(0),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_INPUT_TOKENS,
        i64::from(total_usage.input_tokens),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_OUTPUT_TOKENS,
        i64::from(total_usage.output_tokens),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        i64::from(total_usage.cached_input_tokens),
    ));
    span.set_attribute(KeyValue::new(
        attrs::GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        i64::from(total_usage.cache_creation_input_tokens),
    ));
    span.set_attribute(KeyValue::new(attrs::SDK_OUTCOME, outcome));
    if outcome == "error" {
        spans::set_span_error(&mut span, "agent_error", "agent invocation failed");
    }
    span.end();
}

/// Flush any pending [`trace_io::RootTraceState`] onto the live root
/// span before [`end_root_span`] consumes the sink.
pub(crate) fn flush_root_trace_state(sink: &RootSpanEventSink, state: &trace_io::RootTraceState) {
    sink.with_span_mut(|span| state.flush(span));
}

/// Map an `AgentRunState` to an outcome string.
#[must_use]
pub(crate) const fn run_state_outcome(state: &crate::types::AgentRunState) -> &'static str {
    match state {
        crate::types::AgentRunState::Done { .. } => "done",
        crate::types::AgentRunState::Refusal { .. } => "refusal",
        crate::types::AgentRunState::AwaitingConfirmation { .. } => "awaiting_confirmation",
        crate::types::AgentRunState::Cancelled { .. } => "cancelled",
        crate::types::AgentRunState::Error(_) => "error",
        // `AgentRunState` is `#[non_exhaustive]`; an unrecognized future state
        // reports a stable generic outcome.
        _ => "unknown",
    }
}

/// Map a `TurnOutcome` to an outcome string.
#[must_use]
pub(crate) const fn turn_outcome_str(outcome: &crate::types::TurnOutcome) -> &'static str {
    match outcome {
        crate::types::TurnOutcome::Done { .. } => "done",
        crate::types::TurnOutcome::Refusal { .. } => "refusal",
        crate::types::TurnOutcome::NeedsMoreTurns { .. } => "needs_more_turns",
        crate::types::TurnOutcome::AwaitingConfirmation { .. } => "awaiting_confirmation",
        crate::types::TurnOutcome::PendingToolCalls { .. } => "pending_tool_calls",
        crate::types::TurnOutcome::Cancelled { .. } => "cancelled",
        crate::types::TurnOutcome::Error(_) => "error",
    }
}
