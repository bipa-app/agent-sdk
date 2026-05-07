//! Core observability types and the `ObservabilityStore` trait.

use super::payload::PayloadRedactor;
use crate::types::ThreadId;
use async_trait::async_trait;
use std::sync::LazyLock;

/// Process-wide noop redactor used by the default
/// `ObservabilityStore::redactor` implementation — avoids allocating
/// a fresh `Arc<NoopDetector>` on every call.
static NOOP_REDACTOR: LazyLock<PayloadRedactor> = LazyLock::new(PayloadRedactor::noop);

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

    /// PII redactor applied to every payload converted for this store.
    ///
    /// The agent loop calls this once per LLM round-trip and uses
    /// the returned redactor to mask PII in the system prompt, input
    /// messages, and output messages before building the
    /// [`PayloadBundle`]. The same masked JSON is then recorded on
    /// the `OTel` span, so a single redaction pass covers both
    /// external persistence and local tracing.
    ///
    /// The default returns a shared noop redactor — existing stores
    /// keep their current byte-for-byte output. Stores that need
    /// PII-aware redaction (recommended for financial / regulated
    /// workloads) should override this with a
    /// [`PayloadRedactor`] wrapping a detector such as
    /// [`agent_sdk_core::privacy::BaselineDetector`].
    fn redactor(&self) -> &PayloadRedactor {
        &NOOP_REDACTOR
    }

    /// Affirm that this store has a real PII redactor installed and
    /// is safe to honour [`CaptureDecision::Inline`].
    ///
    /// Returns `false` by default. The SDK gates every `Inline`
    /// decision behind this method **and** the operator-facing
    /// `OtelConfig::capture_payloads` flag — both must be true for
    /// payloads to land on spans inline. Stores that have not
    /// explicitly verified their redactor MUST leave the default in
    /// place; otherwise the SDK silently drops payloads to protect
    /// against PII leakage.
    ///
    /// [`CaptureDecision::Reference`] is **not** affected by this
    /// gate — externalised payloads are always recorded as
    /// references because the underlying content stays out of the
    /// span entirely.
    fn acknowledge_pii_redaction(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use agent_sdk_core::ChatRequest;
    use agent_sdk_core::privacy::BaselineDetector;
    use std::sync::Arc;

    struct NoopStore;

    #[async_trait]
    impl ObservabilityStore for NoopStore {
        async fn capture(&self, _bundle: &PayloadBundle) -> anyhow::Result<CaptureResult> {
            Ok(CaptureResult {
                system_instructions: CaptureDecision::Omit,
                input_messages: CaptureDecision::Omit,
                output_messages: CaptureDecision::Omit,
            })
        }
    }

    struct PrivacyStore {
        redactor: PayloadRedactor,
    }

    #[async_trait]
    impl ObservabilityStore for PrivacyStore {
        async fn capture(&self, _bundle: &PayloadBundle) -> anyhow::Result<CaptureResult> {
            Ok(CaptureResult {
                system_instructions: CaptureDecision::Omit,
                input_messages: CaptureDecision::Omit,
                output_messages: CaptureDecision::Omit,
            })
        }

        fn redactor(&self) -> &PayloadRedactor {
            &self.redactor
        }
    }

    fn sample_request() -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![Message::user("CPF 111.444.777-35 please")],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
        }
    }

    #[test]
    fn default_redactor_is_noop() {
        let store = NoopStore;
        let result = store.redactor().convert_input_messages(&sample_request());
        let text = result[0]["content"][0]["text"].as_str().expect("text");
        // Default impl: no redaction — CPF flows through unchanged.
        assert_eq!(text, "CPF 111.444.777-35 please");
    }

    #[test]
    fn overridden_redactor_masks_pii() {
        let store = PrivacyStore {
            redactor: PayloadRedactor::new(Arc::new(
                BaselineDetector::new().expect("baseline compiles"),
            )),
        };
        let result = store.redactor().convert_input_messages(&sample_request());
        let text = result[0]["content"][0]["text"].as_str().expect("text");
        assert!(
            text.contains("[REDACTED:cpf]"),
            "expected CPF mask via trait, got {text}"
        );
        assert!(!text.contains("111.444.777-35"));
    }
}
