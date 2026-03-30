//! Core observability types and the `ObservabilityStore` trait.

use crate::types::ThreadId;
use async_trait::async_trait;

/// Identifies the kind of LLM payload capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Normal turn chat request/response.
    TurnChat,
    /// Compaction summarization request/response.
    CompactionChat,
}

impl CaptureKind {
    /// Low-cardinality string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::TurnChat => "turn_chat",
            Self::CompactionChat => "compaction_chat",
        }
    }
}

/// Decision returned by the `ObservabilityStore` for each payload artifact.
#[derive(Debug, Clone)]
pub enum CaptureDecision {
    /// Serialize the payload as a JSON span attribute inline.
    Inline,
    /// Store externally; record only this reference string on the span.
    Reference(String),
    /// Do not record this artifact.
    Omit,
}

/// Structured payload bundle passed to the observability store.
///
/// Contains the `GenAI` semantic-convention-aligned payloads for a single
/// LLM operation, plus metadata needed for external persistence.
#[derive(Debug, Clone)]
pub struct PayloadBundle {
    /// Opaque SDK-generated identifier, unique per capture attempt.
    pub capture_id: String,
    /// Discriminator for the type of LLM operation.
    pub capture_kind: CaptureKind,
    /// Thread this operation belongs to.
    pub thread_id: ThreadId,
    /// Turn number within the current invocation.
    pub turn_number: usize,
    /// Canonical `gen_ai.provider.name` value.
    pub provider_name: String,
    /// Raw SDK provider identifier (e.g. `"openai-responses"`).
    pub provider_id: String,
    /// Whether the current LLM span is recording.
    pub span_is_recording: bool,
    /// Request model string.
    pub request_model: String,
    /// Response model string, if available.
    pub response_model: Option<String>,
    /// System instructions as semconv JSON value, if present.
    pub system_instructions: Option<serde_json::Value>,
    /// Input messages as semconv JSON value.
    pub input_messages: serde_json::Value,
    /// Output messages as semconv JSON value.
    pub output_messages: serde_json::Value,
}

/// Per-artifact decisions returned by the store.
#[derive(Debug, Clone)]
pub struct CaptureResult {
    /// Decision for system instructions.
    pub system_instructions: CaptureDecision,
    /// Decision for input messages.
    pub input_messages: CaptureDecision,
    /// Decision for output messages.
    pub output_messages: CaptureDecision,
}

/// Async trait for `GenAI` payload capture.
///
/// Separate from `MessageStore` / `StateStore`. Called at the LLM
/// instrumentation boundary to decide whether payloads are inlined,
/// externalized, or omitted from spans.
#[async_trait]
pub trait ObservabilityStore: Send + Sync {
    /// Capture or inspect the payload bundle for a single LLM operation.
    ///
    /// Called even when the current span is non-recording (the bundle
    /// includes `span_is_recording` so the store can decide whether to
    /// persist externally).
    ///
    /// # Errors
    ///
    /// Errors are logged and swallowed — they never fail the agent run.
    async fn capture(&self, bundle: &PayloadBundle) -> anyhow::Result<CaptureResult>;
}
