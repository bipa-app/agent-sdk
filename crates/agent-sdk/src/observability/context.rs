//! OpenTelemetry context propagation helpers for async boundaries.
//!
//! The SDK uses `tokio::spawn` in `run()` and `run_turn()`. The spawned
//! futures are wrapped with `FutureExt::with_context()` so the caller's
//! `OTel` context is re-attached on every poll, surviving task migration
//! across tokio worker threads.

use opentelemetry::Context;
use opentelemetry::trace::{SpanContext, TraceContextExt};

/// Capture the current `OTel` context for propagation into a spawned task.
///
/// Call this at the public API boundary (before `tokio::spawn`) and pass
/// the returned `Context` via `FutureExt::with_context()`.
#[must_use]
pub fn capture_context() -> Context {
    Context::current()
}

/// Capture the current `OTel` context and replace the active span.
///
/// This is used after starting a span that should become the parent for work
/// performed inside an async sub-future while preserving any existing baggage
/// from the caller context.
#[must_use]
pub fn current_with_span_context(span_context: SpanContext) -> Context {
    Context::current().with_remote_span_context(span_context)
}
