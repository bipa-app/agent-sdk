//! Langfuse trace input/output helpers for the agent loop.
//!
//! Lifts Bipa's `langfuse_trace_input` / `langfuse_trace_output`
//! helpers (and the small `summarize_*` / `append_*` helpers they
//! call) into the SDK so that consumers do not have to instrument
//! `langfuse.trace.{input,output}` from outside the loop. After this
//! module ships, Bipa's mirror copy at
//! `bipa/master/src/features/agent.rs:1209-1436` becomes a deletion
//! candidate (E2 cut-over).
//!
//! The module owns:
//!
//! * Pure, side-effect-free helpers that turn an [`AgentInput`] /
//!   [`AgentEvent`] into an opt-in trace string: [`langfuse_trace_input`],
//!   [`langfuse_trace_output`], [`langfuse_trace_event_label`].
//! * [`RootTraceState`], the per-run accumulator that the loop calls
//!   on every emitted event so the root `invoke_agent` span's
//!   `langfuse.trace.output` attribute reflects the running narrative.
//!
//! Every text leaf passes through a structural [`RedactionPolicy`]
//! before it is stamped on the span — the C1 redactor ships with the
//! SDK and `RedactionPolicy::baseline()` is hard-coded as the default
//! so this layer satisfies the A5 hard rule "do not include
//! `langfuse.trace.input` / `langfuse.trace.output` without the
//! redactor".
//!
//! All public items are crate-private; the only public surface area
//! introduced by A5 is `agent_sdk::RunOptions`.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use agent_sdk_core::privacy::{RedactionPolicy, redact_string};
use opentelemetry::Context;
use opentelemetry::KeyValue;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;

use crate::events::AgentEvent;
use crate::llm::ContentBlock;
use crate::types::AgentInput;

use super::langfuse::{LANGFUSE_TRACE_OUTPUT, truncate_trace_text};
// ── Trace input ──────────────────────────────────────────────────────

/// Compute the `langfuse.trace.input` string for an [`AgentInput`].
///
/// Returns `None` for inputs that do not carry a meaningful user
/// summary (e.g. an empty `Text` payload). The result is truncated to
/// `max_chars` UTF-8 codepoints with a trailing ellipsis on overflow,
/// mirroring [`super::langfuse::truncate_trace_text`].
///
/// Lifted from Bipa's `langfuse_trace_input` /
/// `summarize_langfuse_message_input` in
/// `bipa/master/src/features/agent.rs:1209-1304`.
#[must_use]
pub fn langfuse_trace_input(input: &AgentInput, max_chars: usize) -> Option<String> {
    let text = match input {
        AgentInput::Text(text) => {
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            text.to_owned()
        }
        AgentInput::Message(blocks) => summarize_message_input(blocks)?,
        AgentInput::Resume {
            confirmed,
            rejection_reason,
            ..
        } => {
            if *confirmed {
                "User approved tool confirmation".to_owned()
            } else {
                let reason = rejection_reason
                    .as_deref()
                    .map(str::trim)
                    .filter(|reason| !reason.is_empty());
                reason.map_or_else(
                    || "User rejected tool confirmation".to_owned(),
                    |reason| format!("User rejected tool confirmation: {reason}"),
                )
            }
        }
        AgentInput::Continue => "[Turn continuation]".to_owned(),
        AgentInput::SubmitToolResults { .. } => "[External tool results]".to_owned(),
    };

    Some(truncate_trace_text(&text, max_chars))
}

fn summarize_message_input(blocks: &[ContentBlock]) -> Option<String> {
    let first_block_is_text = blocks
        .first()
        .is_some_and(|block| matches!(block, ContentBlock::Text { .. }));

    let summary = match blocks.first() {
        Some(ContentBlock::Text { text }) => {
            let text = text.trim();
            if text.is_empty() {
                None
            } else {
                Some(text.to_owned())
            }
        }
        _ => None,
    };

    let mut text_attachment_count = 0usize;
    let mut image_count = 0usize;
    let mut document_count = 0usize;

    for block in blocks.iter().skip(usize::from(first_block_is_text)) {
        match block {
            ContentBlock::Text { text } => {
                if !text.trim().is_empty() {
                    text_attachment_count += 1;
                }
            }
            ContentBlock::Image { .. } => image_count += 1,
            ContentBlock::Document { .. } => document_count += 1,
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::ToolUse { .. }
            | ContentBlock::ToolResult { .. } => {}
        }
    }

    let mut parts: Vec<String> = Vec::new();
    if let Some(summary) = summary {
        parts.push(summary);
    }

    let mut attachments: Vec<String> = Vec::new();
    if text_attachment_count > 0 {
        attachments.push(format!("{text_attachment_count} text attachment(s)"));
    }
    if image_count > 0 {
        attachments.push(format!("{image_count} image attachment(s)"));
    }
    if document_count > 0 {
        attachments.push(format!("{document_count} document attachment(s)"));
    }
    if !attachments.is_empty() {
        parts.push(format!("[{}]", attachments.join(", ")));
    }

    if parts.is_empty() {
        return None;
    }

    Some(parts.join("\n\n"))
}

// ── Trace output ─────────────────────────────────────────────────────

/// Compute the `langfuse.trace.output` chunk for a single event.
///
/// Returns `None` for events that contribute nothing user-visible
/// (deltas, lifecycle markers, subagent progress, …). Lifted from
/// Bipa's `langfuse_trace_output` in
/// `bipa/master/src/features/agent.rs:1341-1396`.
#[must_use]
pub fn langfuse_trace_output(event: &AgentEvent) -> Option<String> {
    match event {
        AgentEvent::Text { text, .. } => non_empty(text),
        AgentEvent::ToolCallStart {
            name,
            display_name,
            input,
            ..
        } => Some(summarize_tool_call_start(name, display_name, input)),
        AgentEvent::ToolCallEnd { result, .. } => non_empty(&result.output),
        AgentEvent::ToolRequiresConfirmation { description, .. } => non_empty(description),
        AgentEvent::Error { message, .. } => non_empty(message),
        AgentEvent::Refusal { text, .. } => text.as_deref().and_then(non_empty),
        AgentEvent::Start { .. }
        | AgentEvent::Thinking { .. }
        | AgentEvent::ThinkingDelta { .. }
        | AgentEvent::TextDelta { .. }
        | AgentEvent::ToolProgress { .. }
        | AgentEvent::TurnComplete { .. }
        | AgentEvent::Done { .. }
        | AgentEvent::AutoRetryStart { .. }
        | AgentEvent::AutoRetryEnd { .. }
        | AgentEvent::ContextCompacted { .. }
        | AgentEvent::SubagentProgress { .. } => None,
    }
}

/// Label used to group an event chunk inside the accumulator buffer
/// (`[Assistant]`, `[Tool Result]`, …). Lifted from Bipa's
/// `langfuse_trace_event_label`.
#[must_use]
pub const fn langfuse_trace_event_label(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::Text { .. } => "Assistant",
        AgentEvent::ToolCallStart { .. } => "Tool Call",
        AgentEvent::ToolCallEnd { .. } => "Tool Result",
        AgentEvent::ToolRequiresConfirmation { .. } => "Tool Confirmation",
        AgentEvent::Error { .. } => "Error",
        AgentEvent::Refusal { .. } => "Refusal",
        AgentEvent::Start { .. }
        | AgentEvent::Thinking { .. }
        | AgentEvent::ThinkingDelta { .. }
        | AgentEvent::TextDelta { .. }
        | AgentEvent::ToolProgress { .. }
        | AgentEvent::TurnComplete { .. }
        | AgentEvent::Done { .. }
        | AgentEvent::AutoRetryStart { .. }
        | AgentEvent::AutoRetryEnd { .. }
        | AgentEvent::ContextCompacted { .. }
        | AgentEvent::SubagentProgress { .. } => "Event",
    }
}

fn summarize_tool_call_start(name: &str, display_name: &str, input: &serde_json::Value) -> String {
    let tool_name = name.trim();
    let tool_name = if tool_name.is_empty() {
        "Tool requested"
    } else {
        tool_name
    };

    let display_name = display_name.trim();
    let display_name = display_name
        .strip_prefix("Use ")
        .unwrap_or(display_name)
        .trim();

    let mut output = tool_name.to_owned();

    if !display_name.is_empty() && display_name != tool_name {
        let _ = write!(output, "\nDisplay name: {display_name}");
    }

    if let serde_json::Value::Object(args) = input
        && !args.is_empty()
    {
        let mut keys: Vec<&str> = args.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let _ = write!(output, "\nArguments: {}", keys.join(", "));
    }

    output
}

fn non_empty(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

// ── Root trace state ─────────────────────────────────────────────────

/// Per-run accumulator that mirrors emitted events onto
/// `langfuse.trace.output` on the root `invoke_agent` span.
///
/// `RootTraceState` is `Send + Sync` and lives behind an `Arc` for the
/// duration of a run. Concurrent writers (e.g. progress events sent
/// from the listen-tool task) are serialised by the inner buffer
/// mutex.
///
/// The accumulator is propagated to inner code paths via the
/// [`opentelemetry::Context`] using
/// [`Context::with_value`](opentelemetry::Context::with_value) so
/// helpers like `helpers::send_event` can fetch it from the active
/// context without growing every parameter struct in the loop.
///
/// The accumulator deliberately **does not** call
/// [`opentelemetry::trace::Span::set_attribute`] on every event. The
/// 0.31 `OTel` Rust SDK appends every `set_attribute` call to a `Vec`
/// without deduplicating by key (see
/// `opentelemetry_sdk::trace::span::Span::set_attribute` —
/// `data.attributes.push(attribute)`); pushing on every event would
/// leave the exported `SpanData` with one attribute entry per event
/// and most exporters would surface only the first. Instead the
/// loop calls [`flush`](Self::flush) exactly once when the root
/// span is about to end, so the final `langfuse.trace.output`
/// stamps the full accumulated narrative as a single attribute.
pub struct RootTraceState {
    redactor: RedactionPolicy,
    max_chars: usize,
    buffer: Mutex<String>,
}

impl RootTraceState {
    /// Build a state object for a freshly-created root span.
    pub fn new(max_chars: usize) -> Self {
        // Reserve enough headroom that a typical run won't reallocate.
        let buffer = String::with_capacity(max_chars.saturating_add(64));
        Self {
            redactor: RedactionPolicy::baseline(),
            max_chars,
            buffer: Mutex::new(buffer),
        }
    }

    /// Attach this state to an [`opentelemetry::Context`] so helpers
    /// further down the call chain can retrieve it via
    /// [`from_current_context`](Self::from_current_context).
    pub fn attach_to(self: Arc<Self>, cx: &Context) -> Context {
        cx.with_value(TraceStateHandle(self))
    }

    /// Look up the trace state for the **current** `OTel` context.
    ///
    /// Returns `None` when no state was attached for this run (the
    /// non-`run_with_options` paths never attach one) or when the
    /// caller is outside an attached future.
    pub fn from_current_context() -> Option<Arc<Self>> {
        let cx = Context::current();
        cx.get::<TraceStateHandle>().map(|handle| handle.0.clone())
    }

    /// Append one event to the running narrative buffer.
    ///
    /// Skipped silently when:
    /// * the event yields no trace-output text;
    /// * the inner mutex is poisoned (logged once, then dropped —
    ///   trace-output writes must never fail an agent run).
    pub fn observe(&self, event: &AgentEvent) {
        let Some(text) = langfuse_trace_output(event) else {
            return;
        };
        let label = langfuse_trace_event_label(event);
        self.append(label, &text);
    }

    /// Append a free-form `[Error] <message>` chunk.
    ///
    /// Used by terminal error paths so a turn that fails mid-stream
    /// still reports the failure narrative on `langfuse.trace.output`
    /// (Bipa does this today, see
    /// `bipa/master/src/features/agent.rs:1992-2000`).
    pub fn observe_error(&self, message: &str) {
        let trimmed = message.trim();
        if trimmed.is_empty() {
            return;
        }
        self.append("Error", trimmed);
    }

    /// Stamp the accumulated narrative on `span` as a single
    /// `langfuse.trace.output` attribute.
    ///
    /// No-op when the buffer is empty (the run emitted no
    /// trace-output-relevant events) so we don't pollute the span
    /// with a stray empty string.
    ///
    /// Called once by the loop right before [`Span::end`] is
    /// invoked — see `instrument::end_root_span`.
    pub fn flush(&self, span: &mut BoxedSpan) {
        let Ok(buf) = self.buffer.lock() else {
            log::warn!("langfuse trace-output buffer mutex poisoned; dropping flush");
            return;
        };
        if buf.is_empty() {
            return;
        }
        let truncated = truncate_trace_text(&buf, self.max_chars);
        drop(buf);
        span.set_attribute(KeyValue::new(LANGFUSE_TRACE_OUTPUT, truncated));
    }

    fn append(&self, label: &str, text: &str) {
        let masked = redact_string(text, &self.redactor);
        let Ok(mut buf) = self.buffer.lock() else {
            log::warn!("langfuse trace-output buffer mutex poisoned; dropping update");
            return;
        };
        if !buf.is_empty() {
            buf.push_str("\n\n---\n\n");
        }
        let _ = write!(buf, "[{label}]\n{masked}");
    }
}

/// Newtype wrapper used as the [`opentelemetry::Context`] key for
/// [`RootTraceState`].
///
/// `Context::with_value` keys by the `TypeId` of the stored value, so
/// wrapping the `Arc` in a dedicated type guarantees we don't collide
/// with any other code that stores an `Arc` in the context.
#[derive(Clone)]
struct TraceStateHandle(Arc<RootTraceState>);

/// Convenience: observe an event under the **current** context.
///
/// No-op when no [`RootTraceState`] is attached. Used by
/// `helpers::send_event` so every emission point gains
/// `langfuse.trace.output` automatically.
pub fn observe_current(event: &AgentEvent) {
    if let Some(state) = RootTraceState::from_current_context() {
        state.observe(event);
    }
}

/// Convenience: observe a free-form error message under the
/// **current** context.
pub fn observe_current_error(message: &str) {
    if let Some(state) = RootTraceState::from_current_context() {
        state.observe_error(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ContentSource;
    use crate::types::{ContinuationEnvelope, ThreadId};
    use anyhow::Context as _;
    use serde_json::json;

    #[test]
    fn trace_input_returns_none_for_empty_text() {
        let input = AgentInput::Text("   ".to_string());
        assert!(langfuse_trace_input(&input, 100).is_none());
    }

    #[test]
    fn trace_input_truncates_long_text() -> anyhow::Result<()> {
        let long: String = "x".repeat(20);
        let input = AgentInput::Text(long);
        let result = langfuse_trace_input(&input, 5).context("expected Some")?;
        assert_eq!(result.chars().count(), 5);
        assert!(result.ends_with('…'));
        Ok(())
    }

    #[test]
    fn trace_input_message_summarises_attachments() -> anyhow::Result<()> {
        let blocks = vec![
            ContentBlock::Text {
                text: "hello".to_string(),
            },
            ContentBlock::Image {
                source: ContentSource::new("image/png", "aGk="),
            },
            ContentBlock::Document {
                source: ContentSource::new("application/pdf", "cGRm"),
            },
        ];
        let input = AgentInput::Message(blocks);
        let result = langfuse_trace_input(&input, 1000).context("expected Some")?;
        assert!(result.contains("hello"));
        assert!(result.contains("1 image attachment"));
        assert!(result.contains("1 document attachment"));
        Ok(())
    }

    #[test]
    fn trace_input_message_returns_none_when_only_thinking() {
        // Thinking blocks alone do not contribute a user-visible
        // summary.
        let blocks = vec![ContentBlock::Thinking {
            thinking: "internal".to_string(),
            signature: None,
        }];
        let input = AgentInput::Message(blocks);
        assert!(langfuse_trace_input(&input, 100).is_none());
    }

    #[test]
    fn trace_input_resume_confirmed() -> anyhow::Result<()> {
        use crate::types::TokenUsage;
        let env = ContinuationEnvelope::wrap(crate::types::AgentContinuation {
            thread_id: ThreadId::from_string("t"),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: crate::types::AgentState::new(ThreadId::from_string("t")),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        });
        let input = AgentInput::Resume {
            continuation: Box::new(env),
            tool_call_id: "call_1".to_string(),
            confirmed: true,
            rejection_reason: None,
        };
        let out = langfuse_trace_input(&input, 100).context("expected Some")?;
        assert_eq!(out, "User approved tool confirmation");
        Ok(())
    }

    #[test]
    fn trace_input_continue_and_submit() {
        use crate::types::TokenUsage;
        let env = ContinuationEnvelope::wrap(crate::types::AgentContinuation {
            thread_id: ThreadId::from_string("t"),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: crate::types::AgentState::new(ThreadId::from_string("t")),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        });
        let cont = AgentInput::Continue;
        assert_eq!(
            langfuse_trace_input(&cont, 100).as_deref(),
            Some("[Turn continuation]")
        );
        let submit = AgentInput::SubmitToolResults {
            continuation: Box::new(env),
            results: Vec::new(),
        };
        assert_eq!(
            langfuse_trace_input(&submit, 100).as_deref(),
            Some("[External tool results]")
        );
    }

    #[test]
    fn trace_output_text_returns_text() {
        let event = AgentEvent::text("m", "hello");
        assert_eq!(langfuse_trace_output(&event).as_deref(), Some("hello"));
    }

    #[test]
    fn trace_output_skipped_for_internal_events() {
        let start = AgentEvent::Start {
            thread_id: ThreadId::from_string("t"),
            turn: 1,
        };
        assert!(langfuse_trace_output(&start).is_none());
        let delta = AgentEvent::text_delta("m", "x");
        assert!(langfuse_trace_output(&delta).is_none());
    }

    #[test]
    fn trace_output_tool_call_summarises_arguments() -> anyhow::Result<()> {
        let event = AgentEvent::tool_call_start(
            "call_1",
            "ls",
            "List Files",
            json!({"path": "/", "depth": 1}),
            crate::types::ToolTier::Observe,
        );
        let out = langfuse_trace_output(&event).context("expected Some")?;
        assert!(out.contains("ls"));
        // Argument keys are sorted.
        assert!(out.contains("Arguments: depth, path"));
        Ok(())
    }

    #[test]
    fn trace_output_label_for_text() {
        assert_eq!(
            langfuse_trace_event_label(&AgentEvent::text("m", "hi")),
            "Assistant"
        );
        assert_eq!(
            langfuse_trace_event_label(&AgentEvent::tool_call_end(
                "id",
                "ls",
                "List",
                crate::types::ToolResult::success("done"),
            )),
            "Tool Result"
        );
        assert_eq!(
            langfuse_trace_event_label(&AgentEvent::error("boom", false)),
            "Error"
        );
    }
}
