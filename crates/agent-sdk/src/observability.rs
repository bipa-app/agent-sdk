//! OpenTelemetry observability module.
//!
//! This module is only compiled when the `otel` feature is enabled.
//! It provides span instrumentation, payload capture, and context propagation
//! for the agent SDK's core orchestration boundaries.

pub mod attrs;
pub mod context;
pub mod instrument;
pub mod payload;
pub mod provider_name;
pub mod spans;
pub mod types;

pub use types::{CaptureDecision, CaptureKind, CaptureResult, ObservabilityStore, PayloadBundle};
