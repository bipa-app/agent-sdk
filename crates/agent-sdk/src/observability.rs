//! OpenTelemetry observability module.
//!
//! This module is only compiled when the `otel` feature is enabled.
//! It provides span instrumentation, payload capture, and context propagation
//! for the agent SDK's core orchestration boundaries.

pub mod attrs;
pub mod baggage;
pub mod context;
pub mod instrument;
pub mod langfuse;
pub mod metrics;
pub mod payload;
pub mod payload_capture;
pub mod provider_name;
pub mod spans;
pub(crate) mod trace_io;
pub mod types;

pub use payload_capture::{is_payload_capture_enabled, set_payload_capture_enabled};
pub use types::{CaptureDecision, CaptureKind, CaptureResult, ObservabilityStore, PayloadBundle};
