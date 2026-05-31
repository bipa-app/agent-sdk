//! Schema-validated structured output.
//!
//! This module implements a provider-agnostic runner that constrains a model's
//! final answer to a JSON Schema, validates the output, and bounded-re-prompts
//! on mismatch before failing with a typed error.
//!
//! # How it works
//!
//! 1. The caller supplies a [`ChatRequest`] whose
//!    [`response_format`](agent_sdk_core::llm::ChatRequest::response_format) is
//!    set, plus a [`StructuredConfig`] bounding the retries.
//! 2. The runner inspects the provider's
//!    [`structured_output_support`](crate::LlmProvider::structured_output_support):
//!    - [`Native`](crate::StructuredOutputSupport::Native) — the provider
//!      already mapped `response_format` onto its wire request (`OpenAI` JSON
//!      mode, Gemini `responseSchema`). The structured value is parsed from the
//!      assistant's text output.
//!    - [`ToolForcing`](crate::StructuredOutputSupport::ToolForcing) — the
//!      runner injects a single forced "respond" tool whose `input_schema` is
//!      the output schema (the Anthropic fallback) and reads the structured
//!      value from that tool call's input.
//! 3. The candidate value is validated against the schema with `jsonschema`.
//!    On success it is returned. On failure the runner appends the model's
//!    output plus a corrective user message describing the validation errors
//!    and retries, up to [`StructuredConfig::max_retries`] times. Exhausting
//!    the budget yields [`StructuredOutputError::RetriesExhausted`].
//!
//! This mirrors the Claude SDK's `output_format` +
//! `error_max_structured_output_retries` behaviour.

use agent_sdk_core::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Message, ResponseFormat, Tool, ToolChoice,
};
use agent_sdk_core::types::ToolTier;

use crate::provider::{LlmProvider, StructuredOutputSupport};

/// The forced tool name used for the tool-forcing fallback (Anthropic).
const RESPOND_TOOL_NAME: &str = "respond";

/// Bounds for the structured-output re-prompt loop.
#[derive(Debug, Clone, Copy)]
pub struct StructuredConfig {
    /// Maximum number of *re-prompts* after the first attempt. A value of `2`
    /// means up to three model calls total (1 initial + 2 retries) before the
    /// runner gives up with [`StructuredOutputError::RetriesExhausted`].
    ///
    /// Mirrors the Claude SDK `error_max_structured_output_retries`.
    pub max_retries: u32,
}

impl Default for StructuredConfig {
    fn default() -> Self {
        Self { max_retries: 2 }
    }
}

/// A successfully validated structured output and the response that produced it.
#[derive(Debug, Clone)]
pub struct StructuredOutput {
    /// The validated JSON value, guaranteed to satisfy the requested schema.
    pub value: serde_json::Value,
    /// The full provider response that produced [`value`](Self::value), so
    /// callers can still read usage, stop reason, and any leading text.
    pub response: ChatResponse,
    /// Number of re-prompts performed before the value validated (0 when the
    /// first attempt already satisfied the schema).
    pub retries: u32,
}

/// Errors from the structured-output runner.
///
/// These are *typed* terminal outcomes — the runner never panics on a model
/// that fails to produce schema-valid output.
#[derive(Debug, thiserror::Error)]
pub enum StructuredOutputError {
    /// The request did not carry a
    /// [`response_format`](agent_sdk_core::llm::ChatRequest::response_format).
    #[error("structured output requested without a response_format on the request")]
    MissingResponseFormat,

    /// The schema supplied in the response format is not a valid JSON Schema.
    #[error("invalid output JSON schema: {0}")]
    InvalidSchema(String),

    /// The model produced no extractable structured value (no JSON text /
    /// no forced tool call).
    #[error("model produced no structured output to validate")]
    NoStructuredOutput,

    /// The provider returned a non-success outcome (rate limit, server error,
    /// invalid request).
    #[error("provider returned a non-success outcome: {0}")]
    ProviderOutcome(String),

    /// The re-prompt budget was exhausted and the latest output still failed
    /// schema validation. Carries the final validation errors and the last
    /// candidate value for diagnostics.
    #[error(
        "structured output failed schema validation after {attempts} attempt(s); last errors: {errors}"
    )]
    RetriesExhausted {
        /// Total number of model calls made (initial + retries).
        attempts: u32,
        /// Human-readable concatenation of the final validation errors.
        errors: String,
        /// The last candidate value the model produced, if any.
        last_value: Option<serde_json::Value>,
    },

    /// A transport-level error bubbled up from the provider.
    #[error(transparent)]
    Transport(#[from] anyhow::Error),
}

/// Run a bounded, schema-validated structured-output exchange against `provider`.
///
/// The `request`'s
/// [`response_format`](agent_sdk_core::llm::ChatRequest::response_format) must
/// be set; on success the returned [`StructuredOutput::value`] is guaranteed to
/// satisfy that schema.
///
/// # Errors
///
/// Returns a [`StructuredOutputError`] when the request is missing a response
/// format, the schema is invalid, the provider errors, or the model fails to
/// produce schema-valid output within [`StructuredConfig::max_retries`].
pub async fn run_structured(
    provider: &dyn LlmProvider,
    mut request: ChatRequest,
    config: StructuredConfig,
) -> Result<StructuredOutput, StructuredOutputError> {
    let response_format = request
        .response_format
        .clone()
        .ok_or(StructuredOutputError::MissingResponseFormat)?;

    // Compile the validator once; reuse it across every retry.
    let validator = jsonschema::validator_for(&response_format.schema)
        .map_err(|e| StructuredOutputError::InvalidSchema(e.to_string()))?;

    let support = provider.structured_output_support();
    if matches!(support, StructuredOutputSupport::ToolForcing) {
        apply_tool_forcing(&mut request, &response_format);
    }

    let max_attempts = config.max_retries.saturating_add(1);
    let mut last_value: Option<serde_json::Value> = None;
    let mut last_errors = String::new();

    for attempt in 0..max_attempts {
        let outcome = provider.chat(request.clone()).await?;
        let response = match outcome {
            ChatOutcome::Success(response) => response,
            ChatOutcome::RateLimited => {
                return Err(StructuredOutputError::ProviderOutcome(
                    "rate limited".to_owned(),
                ));
            }
            ChatOutcome::InvalidRequest(msg) => {
                return Err(StructuredOutputError::ProviderOutcome(format!(
                    "invalid request: {msg}"
                )));
            }
            ChatOutcome::ServerError(msg) => {
                return Err(StructuredOutputError::ProviderOutcome(format!(
                    "server error: {msg}"
                )));
            }
        };

        let candidate = extract_candidate(&response, support);
        let Some(value) = candidate else {
            // No structured value at all. On the final attempt this is a hard
            // failure; otherwise re-prompt asking for the structured answer.
            if attempt + 1 >= max_attempts {
                return Err(StructuredOutputError::NoStructuredOutput);
            }
            append_correction(
                &mut request,
                &response,
                "Your previous reply did not contain a structured answer. \
                 Respond with a single JSON value that satisfies the requested schema.",
            );
            "missing structured output".clone_into(&mut last_errors);
            continue;
        };

        let errors: Vec<String> = validator
            .iter_errors(&value)
            .map(|error| format!("at `{}`: {error}", error.instance_path()))
            .collect();

        if errors.is_empty() {
            return Ok(StructuredOutput {
                value,
                response,
                retries: attempt,
            });
        }

        last_errors = errors.join("; ");
        last_value = Some(value);

        if attempt + 1 < max_attempts {
            let correction = format!(
                "Your previous JSON output did not satisfy the schema. \
                 Fix these validation errors and resend the full JSON value: {last_errors}"
            );
            append_correction(&mut request, &response, &correction);
        }
    }

    Err(StructuredOutputError::RetriesExhausted {
        attempts: max_attempts,
        errors: last_errors,
        last_value,
    })
}

/// Inject the forced "respond" tool for providers without native JSON mode.
fn apply_tool_forcing(request: &mut ChatRequest, response_format: &ResponseFormat) {
    let respond_tool = Tool {
        name: RESPOND_TOOL_NAME.to_owned(),
        description: format!(
            "Return the final answer as structured data named `{}`. \
             You MUST call this tool exactly once with arguments matching the schema.",
            response_format.name
        ),
        input_schema: response_format.schema.clone(),
        display_name: "Structured response".to_owned(),
        tier: ToolTier::Observe,
    };

    match request.tools {
        Some(ref mut tools) => {
            tools.retain(|t| t.name != RESPOND_TOOL_NAME);
            tools.push(respond_tool);
        }
        None => request.tools = Some(vec![respond_tool]),
    }
    request.tool_choice = Some(ToolChoice::Tool(RESPOND_TOOL_NAME.to_owned()));
}

/// Pull the candidate structured value out of a response according to how the
/// provider satisfied the request.
fn extract_candidate(
    response: &ChatResponse,
    support: StructuredOutputSupport,
) -> Option<serde_json::Value> {
    match support {
        StructuredOutputSupport::ToolForcing => {
            response.content.iter().find_map(|block| match block {
                ContentBlock::ToolUse { name, input, .. } if name == RESPOND_TOOL_NAME => {
                    Some(input.clone())
                }
                _ => None,
            })
        }
        StructuredOutputSupport::Native => {
            let text = response.first_text()?;
            parse_json_text(text)
        }
    }
}

/// Parse a JSON value from model text output.
///
/// Native JSON mode returns a bare JSON document, but models occasionally wrap
/// it in a fenced code block, so this strips a leading/trailing markdown fence
/// before parsing.
fn parse_json_text(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    let unfenced = strip_code_fence(trimmed);
    serde_json::from_str(unfenced).ok()
}

/// Strip a surrounding ```` ```json ... ``` ```` (or plain ```` ``` ````) fence.
fn strip_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    // Drop an optional language tag on the opening fence line.
    let rest = rest.split_once('\n').map_or(rest, |(_, body)| body);
    rest.strip_suffix("```")
        .map_or(text, |inner| inner.trim_end_matches('`').trim())
}

/// Append the assistant's previous output plus a corrective user message so the
/// next attempt sees the validation feedback.
fn append_correction(request: &mut ChatRequest, previous: &ChatResponse, correction: &str) {
    request
        .messages
        .push(Message::assistant_with_content(previous.content.clone()));
    request.messages.push(Message::user(correction));
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agent_sdk_core::llm::{StopReason, Usage};
    use anyhow::Result;
    use async_trait::async_trait;

    use crate::streaming::StreamBox;

    /// A scripted provider: replays a fixed queue of [`ChatOutcome`]s and
    /// reports a configurable [`StructuredOutputSupport`]. It also records every
    /// request it receives so tests can assert on the re-prompt history and on
    /// the tool-forcing injection.
    struct ScriptedProvider {
        provider_name: &'static str,
        model: String,
        support: StructuredOutputSupport,
        outcomes: Mutex<std::collections::VecDeque<ChatOutcome>>,
        seen_requests: Mutex<Vec<ChatRequest>>,
        calls: AtomicUsize,
    }

    impl ScriptedProvider {
        fn new(
            provider_name: &'static str,
            support: StructuredOutputSupport,
            outcomes: Vec<ChatOutcome>,
        ) -> Self {
            Self {
                provider_name,
                model: "scripted-model".to_owned(),
                support,
                outcomes: Mutex::new(outcomes.into()),
                seen_requests: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen_requests
                .lock()
                .expect("seen_requests lock")
                .push(request);
            let outcome = self
                .outcomes
                .lock()
                .expect("outcomes lock")
                .pop_front()
                .expect("ScriptedProvider: ran out of scripted outcomes");
            Ok(outcome)
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            Box::pin(async_stream::stream! {
                yield Err(anyhow::anyhow!("streaming not used in structured tests"));
            })
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn provider(&self) -> &'static str {
            self.provider_name
        }

        fn structured_output_support(&self) -> StructuredOutputSupport {
            self.support
        }
    }

    fn person_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer", "minimum": 0 }
            },
            "required": ["name", "age"],
            "additionalProperties": false
        })
    }

    fn request_with_format() -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![Message::user("Describe a person.")],
            tools: None,
            max_tokens: 256,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: Some(ResponseFormat::new("person", person_schema())),
        }
    }

    fn success(content: Vec<ContentBlock>) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp".to_owned(),
            content,
            model: "scripted-model".to_owned(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    fn text_block(text: &str) -> Vec<ContentBlock> {
        vec![ContentBlock::Text {
            text: text.to_owned(),
        }]
    }

    fn respond_tool_block(input: serde_json::Value) -> Vec<ContentBlock> {
        vec![ContentBlock::ToolUse {
            id: "call_1".to_owned(),
            name: RESPOND_TOOL_NAME.to_owned(),
            input,
            thought_signature: None,
        }]
    }

    // ── Happy path: native (OpenAI / Gemini) ──────────────────────────

    #[tokio::test]
    async fn native_happy_path_validates_json_text() -> Result<()> {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![success(text_block(r#"{"name": "Ada", "age": 36}"#))],
        );

        let out = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        )
        .await?;

        assert_eq!(out.value["name"], "Ada");
        assert_eq!(out.value["age"], 36);
        assert_eq!(out.retries, 0);
        assert_eq!(provider.call_count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn native_happy_path_strips_markdown_fence() -> Result<()> {
        let provider = ScriptedProvider::new(
            "gemini",
            StructuredOutputSupport::Native,
            vec![success(text_block(
                "```json\n{\"name\": \"Grace\", \"age\": 45}\n```",
            ))],
        );

        let out = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        )
        .await?;

        assert_eq!(out.value["name"], "Grace");
        Ok(())
    }

    // ── Happy path: tool-forcing fallback (Anthropic) ─────────────────

    #[tokio::test]
    async fn tool_forcing_happy_path_reads_tool_input() -> Result<()> {
        let provider = ScriptedProvider::new(
            "anthropic",
            StructuredOutputSupport::ToolForcing,
            vec![success(respond_tool_block(
                serde_json::json!({"name": "Linus", "age": 54}),
            ))],
        );

        let out = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        )
        .await?;

        assert_eq!(out.value["name"], "Linus");
        assert_eq!(out.retries, 0);

        // The runner must have injected the forced respond tool.
        let (has_respond_tool, forces_respond) = {
            let seen = provider.seen_requests.lock().expect("seen lock");
            let tools = seen[0].tools.as_ref().expect("tools injected");
            (
                tools.iter().any(|t| t.name == RESPOND_TOOL_NAME),
                matches!(
                    seen[0].tool_choice,
                    Some(ToolChoice::Tool(ref n)) if n == RESPOND_TOOL_NAME
                ),
            )
        };
        assert!(has_respond_tool);
        assert!(forces_respond);
        Ok(())
    }

    // ── Mismatch → retry → success ────────────────────────────────────

    #[tokio::test]
    async fn mismatch_then_retry_succeeds() -> Result<()> {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![
                // First attempt: `age` is a string, violating the schema.
                success(text_block(r#"{"name": "Ada", "age": "old"}"#)),
                // Retry: corrected.
                success(text_block(r#"{"name": "Ada", "age": 36}"#)),
            ],
        );

        let out = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig { max_retries: 2 },
        )
        .await?;

        assert_eq!(out.value["age"], 36);
        assert_eq!(out.retries, 1);
        assert_eq!(provider.call_count(), 2);

        // The corrective re-prompt must have appended the prior answer + a
        // user correction message.
        let grew = {
            let seen = provider.seen_requests.lock().expect("seen lock");
            seen[1].messages.len() > seen[0].messages.len()
        };
        assert!(grew);
        Ok(())
    }

    // ── Retry exhaustion → typed error ────────────────────────────────

    #[tokio::test]
    async fn retry_exhaustion_yields_typed_error() -> Result<()> {
        let provider = ScriptedProvider::new(
            "anthropic",
            StructuredOutputSupport::ToolForcing,
            vec![
                success(respond_tool_block(serde_json::json!({"name": "x"}))),
                success(respond_tool_block(serde_json::json!({"name": "y"}))),
                success(respond_tool_block(serde_json::json!({"name": "z"}))),
            ],
        );

        let err = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig { max_retries: 2 },
        )
        .await
        .expect_err("schema never satisfied");

        match err {
            StructuredOutputError::RetriesExhausted {
                attempts,
                last_value,
                ..
            } => {
                assert_eq!(attempts, 3, "1 initial + 2 retries");
                assert_eq!(
                    last_value.as_ref().and_then(|v| v["name"].as_str()),
                    Some("z")
                );
            }
            other => panic!("expected RetriesExhausted, got {other:?}"),
        }
        // initial + 2 retries == 3 calls.
        assert_eq!(provider.call_count(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn zero_retries_fails_after_single_attempt() -> Result<()> {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![success(text_block(r#"{"name": "Ada"}"#))],
        );

        let err = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig { max_retries: 0 },
        )
        .await
        .expect_err("missing required `age`");

        assert!(matches!(
            err,
            StructuredOutputError::RetriesExhausted { attempts: 1, .. }
        ));
        assert_eq!(provider.call_count(), 1);
        Ok(())
    }

    // ── Error surfaces ────────────────────────────────────────────────

    #[tokio::test]
    async fn missing_response_format_is_typed_error() {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![success(text_block("{}"))],
        );
        let mut req = request_with_format();
        req.response_format = None;

        let err = run_structured(&provider, req, StructuredConfig::default())
            .await
            .expect_err("no response format");
        assert!(matches!(err, StructuredOutputError::MissingResponseFormat));
    }

    #[tokio::test]
    async fn invalid_schema_is_typed_error() {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![success(text_block("{}"))],
        );
        let mut req = request_with_format();
        // `type` must be a string/array, not a number — an invalid schema.
        req.response_format = Some(ResponseFormat::new("bad", serde_json::json!({"type": 123})));

        let err = run_structured(&provider, req, StructuredConfig::default())
            .await
            .expect_err("invalid schema");
        assert!(matches!(err, StructuredOutputError::InvalidSchema(_)));
    }

    #[tokio::test]
    async fn provider_rate_limit_surfaces_as_typed_error() {
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![ChatOutcome::RateLimited],
        );

        let err = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        )
        .await
        .expect_err("rate limited");
        assert!(matches!(err, StructuredOutputError::ProviderOutcome(_)));
    }

    #[tokio::test]
    async fn no_structured_output_on_final_attempt_errors() {
        // Native provider returns non-JSON prose every time.
        let provider = ScriptedProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![
                success(text_block("I cannot do that.")),
                success(text_block("Still prose, sorry.")),
            ],
        );

        let err = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig { max_retries: 1 },
        )
        .await
        .expect_err("never produced JSON");
        assert!(matches!(err, StructuredOutputError::NoStructuredOutput));
        assert_eq!(provider.call_count(), 2);
    }
}
