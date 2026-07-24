//! Schema-validated structured output.
//!
//! This module implements a provider-agnostic runner that constrains a model's
//! final answer to a JSON Schema, validates the output, and bounded-re-prompts
//! on mismatch before failing with a typed error.
//!
//! # How it works
//!
//! 1. The caller supplies a [`ChatRequest`] whose
//!    [`response_format`](agent_sdk_foundation::llm::ChatRequest::response_format) is
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

use std::pin::Pin;

use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Message, ResponseFormat, Tool,
    ToolChoice, Usage,
};
use agent_sdk_foundation::types::ToolTier;
use futures::{Stream, StreamExt};

use crate::provider::{LlmProvider, StructuredOutputSupport};
use crate::streaming::{StreamAccumulator, StreamDelta, StreamErrorKind};

#[cfg(feature = "openai")]
use crate::impls::openai_schema::normalized_strict_schema;

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
    /// [`response_format`](agent_sdk_foundation::llm::ChatRequest::response_format).
    #[error("structured output requested without a response_format on the request")]
    MissingResponseFormat,

    /// The schema supplied in the response format is not a valid JSON Schema.
    #[error("invalid output JSON schema: {0}")]
    InvalidSchema(String),

    /// The schema is valid JSON Schema, but `OpenAI` strict mode cannot preserve
    /// its dynamic-property semantics.
    #[error("output JSON schema is incompatible with OpenAI strict mode: {0}")]
    IncompatibleSchema(String),

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
/// [`response_format`](agent_sdk_foundation::llm::ChatRequest::response_format) must
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
    let validator = validator_for_provider(provider, &response_format)?;

    let support = provider.structured_output_support();
    if matches!(support, StructuredOutputSupport::ToolForcing) {
        apply_tool_forcing(&mut request, &response_format);
    }

    let max_attempts = config.max_retries.saturating_add(1);
    let mut last_value: Option<serde_json::Value> = None;
    let mut last_errors = String::new();

    for attempt in 0..max_attempts {
        // Clone only when a retry may still follow; on the final attempt move the
        // request in (no deep clone of the message history + attachments).
        let attempt_request = if attempt + 1 == max_attempts {
            std::mem::replace(&mut request, ChatRequest::new(String::new(), Vec::new()))
        } else {
            request.clone()
        };
        let outcome = provider.chat(attempt_request).await?;
        let response = match outcome {
            ChatOutcome::Success(response) => response,
            ChatOutcome::RateLimited(_) => {
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
            // `ChatOutcome` is `#[non_exhaustive]`; an unrecognized outcome is
            // surfaced as a provider failure rather than silently retried.
            _ => {
                return Err(StructuredOutputError::ProviderOutcome(
                    "unrecognized provider outcome".to_owned(),
                ));
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
                support,
                "Your previous reply did not contain a structured answer. \
                 Respond with a single JSON value that satisfies the requested schema.",
            );
            "missing structured output".clone_into(&mut last_errors);
            continue;
        };

        let errors = collect_schema_errors(&validator, &value);

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
            append_correction(&mut request, &response, support, &correction);
        }
    }

    Err(StructuredOutputError::RetriesExhausted {
        attempts: max_attempts,
        errors: last_errors,
        last_value,
    })
}

/// An incremental update emitted by [`run_structured_stream`].
#[derive(Debug, Clone)]
pub enum StructuredStreamUpdate {
    /// A best-effort, not-yet-validated object parsed from the partial
    /// response as it streams in. Successive `Partial`s grow toward the final
    /// value; a consumer can render them as a live preview. Because the
    /// underlying JSON is incomplete, a partial may contain truncated string
    /// values and is never schema-validated.
    Partial(serde_json::Value),
    /// The final, schema-validated structured output. Exactly one `Final` is
    /// emitted on success and it is always the last item in the stream.
    Final(StructuredOutput),
}

/// A stream of [`StructuredStreamUpdate`]s produced by [`run_structured_stream`].
pub type StructuredStream<'a> =
    Pin<Box<dyn Stream<Item = Result<StructuredStreamUpdate, StructuredOutputError>> + Send + 'a>>;

/// Streaming counterpart to [`run_structured`].
///
/// Drives the same bounded, schema-validated exchange, but streams the first
/// attempt so callers see the structured object build incrementally
/// ([`StructuredStreamUpdate::Partial`]) before the validated
/// [`StructuredStreamUpdate::Final`] arrives. If the first (streamed) attempt
/// fails schema validation, the bounded re-prompt loop continues
/// non-streaming, reusing the exact retry machinery of [`run_structured`].
///
/// This is additive: existing [`run_structured`] callers are unaffected.
///
/// # Errors
///
/// The stream yields a single [`StructuredOutputError`] (and then ends) when
/// the request lacks a response format, the schema is invalid, the provider
/// errors, or the model never produces schema-valid output within
/// [`StructuredConfig::max_retries`].
pub fn run_structured_stream(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    config: StructuredConfig,
) -> StructuredStream<'_> {
    Box::pin(async_stream::stream! {
        let mut request = request;
        let Some(response_format) = request.response_format.clone() else {
            yield Err(StructuredOutputError::MissingResponseFormat);
            return;
        };
        let validator = match validator_for_provider(provider, &response_format) {
            Ok(validator) => validator,
            Err(error) => {
                yield Err(error);
                return;
            }
        };

        let support = provider.structured_output_support();
        if matches!(support, StructuredOutputSupport::ToolForcing) {
            apply_tool_forcing(&mut request, &response_format);
        }

        let max_attempts = config.max_retries.saturating_add(1);
        let model = provider.model().to_owned();
        let mut last_value: Option<serde_json::Value> = None;
        let mut last_errors = String::new();

        for attempt in 0..max_attempts {
            // The first attempt streams (emitting partials); retries reuse the
            // non-streaming path so the re-prompt machinery is shared verbatim.
            let response = if attempt == 0 {
                let mut attempt_stream =
                    Box::pin(stream_first_attempt(provider, request.clone(), support, model.clone()));
                let mut completed: Option<ChatResponse> = None;
                while let Some(item) = attempt_stream.next().await {
                    match item {
                        StreamAttemptItem::Partial(value) => {
                            yield Ok(StructuredStreamUpdate::Partial(value));
                        }
                        StreamAttemptItem::Complete(response) => completed = Some(response),
                        StreamAttemptItem::Failed(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                }
                // `stream_first_attempt` always terminates with `Complete` or
                // `Failed`; the `None` arm is an unreachable safety net.
                match completed {
                    Some(response) => response,
                    None => return,
                }
            } else {
                match provider.chat(request.clone()).await {
                    Ok(ChatOutcome::Success(response)) => response,
                    Ok(other) => {
                        yield Err(non_success_outcome_error(&other));
                        return;
                    }
                    Err(e) => {
                        yield Err(StructuredOutputError::Transport(e));
                        return;
                    }
                }
            };

            let Some(value) = extract_candidate(&response, support) else {
                if attempt + 1 >= max_attempts {
                    yield Err(StructuredOutputError::NoStructuredOutput);
                    return;
                }
                append_correction(
                    &mut request,
                    &response,
                    support,
                    "Your previous reply did not contain a structured answer. \
                     Respond with a single JSON value that satisfies the requested schema.",
                );
                "missing structured output".clone_into(&mut last_errors);
                continue;
            };

            let errors = collect_schema_errors(&validator, &value);

            if errors.is_empty() {
                yield Ok(StructuredStreamUpdate::Final(StructuredOutput {
                    value,
                    response,
                    retries: attempt,
                }));
                return;
            }

            last_errors = errors.join("; ");
            last_value = Some(value);

            if attempt + 1 < max_attempts {
                let correction = format!(
                    "Your previous JSON output did not satisfy the schema. \
                     Fix these validation errors and resend the full JSON value: {last_errors}"
                );
                append_correction(&mut request, &response, support, &correction);
            }
        }

        yield Err(StructuredOutputError::RetriesExhausted {
            attempts: max_attempts,
            errors: last_errors,
            last_value,
        });
    })
}

/// An item produced by [`stream_first_attempt`]: an incremental partial, the
/// terminal accumulated response, or a typed failure.
enum StreamAttemptItem {
    Partial(serde_json::Value),
    Complete(ChatResponse),
    Failed(StructuredOutputError),
}

/// Stream the first structured-output attempt, emitting de-duplicated partial
/// objects as the response builds and finishing with the accumulated response
/// (or a typed failure). Factored out of [`run_structured_stream`] so the
/// orchestration loop stays small.
fn stream_first_attempt(
    provider: &dyn LlmProvider,
    request: ChatRequest,
    support: StructuredOutputSupport,
    model: String,
) -> impl Stream<Item = StreamAttemptItem> + Send + '_ {
    async_stream::stream! {
        let mut accumulator = StreamAccumulator::new();
        let mut partial_buf = String::new();
        let mut respond_tool_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut last_partial: Option<serde_json::Value> = None;
        let mut stream_error: Option<(String, StreamErrorKind)> = None;

        let mut stream = provider.chat_stream(request);
        while let Some(item) = stream.next().await {
            let delta = match item {
                Ok(delta) => delta,
                Err(e) => {
                    yield StreamAttemptItem::Failed(StructuredOutputError::Transport(e));
                    return;
                }
            };

            accumulate_partial_buffer(&delta, support, &mut partial_buf, &mut respond_tool_ids);
            if let StreamDelta::Error { message, kind } = &delta {
                stream_error = Some((message.clone(), *kind));
            }
            accumulator.apply(&delta);

            if let Some(value) = partial_from_buffer(&partial_buf)
                && last_partial.as_ref() != Some(&value)
            {
                last_partial = Some(value.clone());
                yield StreamAttemptItem::Partial(value);
            }
        }

        if let Some((message, kind)) = stream_error {
            yield StreamAttemptItem::Failed(stream_error_to_outcome(&message, kind));
            return;
        }

        yield StreamAttemptItem::Complete(build_streamed_response(accumulator, model));
    }
}

/// Append the relevant part of a streamed delta to the running partial buffer
/// so [`partial_from_buffer`] can re-parse it. For native providers this is the
/// model's text output; for tool-forcing it is the forced `respond` tool's
/// input JSON.
fn accumulate_partial_buffer(
    delta: &StreamDelta,
    support: StructuredOutputSupport,
    buffer: &mut String,
    respond_tool_ids: &mut std::collections::HashSet<String>,
) {
    match (support, delta) {
        (StructuredOutputSupport::Native, StreamDelta::TextDelta { delta, .. }) => {
            buffer.push_str(delta);
        }
        (StructuredOutputSupport::ToolForcing, StreamDelta::ToolUseStart { id, name, .. })
            if name == RESPOND_TOOL_NAME =>
        {
            respond_tool_ids.insert(id.clone());
        }
        (StructuredOutputSupport::ToolForcing, StreamDelta::ToolInputDelta { id, delta, .. })
            if respond_tool_ids.contains(id) =>
        {
            buffer.push_str(delta);
        }
        _ => {}
    }
}

/// Map a recorded streaming error into the structured-output error surface.
fn stream_error_to_outcome(message: &str, kind: StreamErrorKind) -> StructuredOutputError {
    let label = match kind {
        StreamErrorKind::RateLimited(_) => "rate limited".to_owned(),
        StreamErrorKind::InvalidRequest => format!("invalid request: {message}"),
        _ => format!("server error: {message}"),
    };
    StructuredOutputError::ProviderOutcome(label)
}

/// Map a non-success [`ChatOutcome`] from a retry attempt into a typed error.
fn non_success_outcome_error(outcome: &ChatOutcome) -> StructuredOutputError {
    let label = match outcome {
        ChatOutcome::RateLimited(_) => "rate limited".to_owned(),
        ChatOutcome::InvalidRequest(msg) => format!("invalid request: {msg}"),
        ChatOutcome::ServerError(msg) => format!("server error: {msg}"),
        _ => "unrecognized provider outcome".to_owned(),
    };
    StructuredOutputError::ProviderOutcome(label)
}

/// Materialize a [`ChatResponse`] from a fully-consumed stream accumulator.
fn build_streamed_response(mut accumulator: StreamAccumulator, model: String) -> ChatResponse {
    let usage = accumulator.take_usage().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
    });
    let stop_reason = accumulator.take_stop_reason();
    ChatResponse {
        id: String::new(),
        content: accumulator.into_content_blocks(),
        model,
        stop_reason,
        usage,
    }
}

/// Best-effort parse of a partial JSON object/array from an in-flight buffer.
///
/// Returns the repaired value only when the buffer (after closing any open
/// containers) parses to an object or array; otherwise `None` so the caller
/// simply waits for more data.
fn partial_from_buffer(buffer: &str) -> Option<serde_json::Value> {
    let trimmed = buffer.trim_start();
    // Tolerate a streamed leading ```json fence.
    let body = trimmed
        .strip_prefix("```")
        .and_then(|rest| rest.split_once('\n').map(|(_, body)| body))
        .unwrap_or(trimmed)
        .trim();
    if body.is_empty() {
        return None;
    }
    let repaired = repair_partial_json(body);
    serde_json::from_str::<serde_json::Value>(&repaired)
        .ok()
        .filter(|value| value.is_object() || value.is_array())
}

/// Close any open strings/containers in a partial JSON fragment so it parses.
///
/// Truncated string *values* are kept (closed with `"`); a dangling separator
/// (`,`) is dropped and a dangling key (`"k":`) is completed with `null`. The
/// result is not guaranteed to parse (e.g. a half-typed key), in which case the
/// caller discards it.
fn repair_partial_json(buffer: &str) -> String {
    let mut in_string = false;
    let mut escape = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in buffer.chars() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }

    let mut out = buffer.to_owned();
    if escape {
        out.pop();
    }
    if in_string {
        out.push('"');
    }
    out.truncate(out.trim_end().len());
    if out.ends_with(',') {
        out.pop();
        out.truncate(out.trim_end().len());
    } else if out.ends_with(':') {
        out.push_str(" null");
    }
    for closer in stack.iter().rev() {
        out.push(*closer);
    }
    out
}

/// Collect human-readable schema-validation errors for a candidate value
/// (empty when it satisfies the schema).
fn collect_schema_errors(
    validator: &jsonschema::Validator,
    value: &serde_json::Value,
) -> Vec<String> {
    validator
        .iter_errors(value)
        .map(|error| format!("at `{}`: {error}", error.instance_path()))
        .collect()
}

/// Return the schema against which the structured runner should validate a
/// provider response.
///
/// `OpenAI` strict mode does not use the caller's schema verbatim: it closes
/// objects, requires all properties, and represents caller-optional properties
/// as nullable. The provider sends that normalized schema on the wire, so the
/// local validator must use the same one or it can reject a response `OpenAI`
/// correctly produced (for example, an optional field emitted as `null`).
#[cfg(feature = "openai")]
fn validation_schema_for_provider(
    provider: &dyn LlmProvider,
    response_format: &ResponseFormat,
) -> Result<serde_json::Value, StructuredOutputError> {
    if response_format.strict
        && matches!(
            provider.structured_output_support(),
            StructuredOutputSupport::Native
        )
        && matches!(provider.provider(), "openai" | "openai-responses")
    {
        return normalized_strict_schema(&response_format.schema)
            .map_err(|error| StructuredOutputError::IncompatibleSchema(error.to_string()));
    }

    Ok(response_format.schema.clone())
}

#[cfg(feature = "openai")]
fn validator_for_provider(
    provider: &dyn LlmProvider,
    response_format: &ResponseFormat,
) -> Result<jsonschema::Validator, StructuredOutputError> {
    let validation_schema = validation_schema_for_provider(provider, response_format)?;
    jsonschema::validator_for(&validation_schema)
        .map_err(|error| StructuredOutputError::InvalidSchema(error.to_string()))
}

#[cfg(not(feature = "openai"))]
fn validator_for_provider(
    _provider: &dyn LlmProvider,
    response_format: &ResponseFormat,
) -> Result<jsonschema::Validator, StructuredOutputError> {
    jsonschema::validator_for(&response_format.schema)
        .map_err(|error| StructuredOutputError::InvalidSchema(error.to_string()))
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
    // Forcing a specific tool is incompatible with extended thinking on
    // Anthropic-family models (the API 400s when `thinking` is active alongside
    // a `tool_choice` that names a tool). Clearing `request.thinking` here is
    // *not* sufficient on its own: the Claude providers fall back to their
    // provider-configured thinking default when the request field is `None`
    // (see `resolve_thinking_config`), which would resurrect thinking on the
    // wire. The authoritative guard lives at the wire boundary — the Claude
    // request builders drop thinking whenever `tool_choice` names a tool (see
    // `provider::thinking_for_forced_tool`) — so we do not touch
    // `request.thinking` here and instead rely on that single source of truth.
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
///
/// For the tool-forcing path (Anthropic), the assistant turn carries a forced
/// `respond` `ContentBlock::ToolUse`. The Anthropic Messages API rejects any
/// conversation where a `tool_use` is not immediately followed by a matching
/// `tool_result` in the next user message, so the correction is delivered as a
/// `ToolResult` for that tool-use id (carrying the validation errors) rather than
/// as plain user text — otherwise the very first re-prompt 400s. When no forced
/// tool call is present (or for native providers) the correction is plain text.
fn append_correction(
    request: &mut ChatRequest,
    previous: &ChatResponse,
    support: StructuredOutputSupport,
    correction: &str,
) {
    request
        .messages
        .push(Message::assistant_with_content(previous.content.clone()));

    let respond_tool_use_id = if matches!(support, StructuredOutputSupport::ToolForcing) {
        previous.content.iter().find_map(|block| match block {
            ContentBlock::ToolUse { id, name, .. } if name == RESPOND_TOOL_NAME => Some(id.clone()),
            _ => None,
        })
    } else {
        None
    };

    match respond_tool_use_id {
        Some(tool_use_id) => {
            request
                .messages
                .push(Message::tool_result(tool_use_id, correction, true));
        }
        None => request.messages.push(Message::user(correction)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agent_sdk_foundation::llm::{StopReason, Usage};
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
            cache: None,
        }
    }

    fn request_with_optional_property_format() -> ChatRequest {
        let mut request = request_with_format();
        request.response_format = Some(ResponseFormat::new(
            "optional-person",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nickname": {"type": "string"}
                },
                "required": ["name"]
            }),
        ));
        request
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

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn native_openai_validates_against_the_normalized_strict_schema() -> Result<()> {
        for provider_name in ["openai", "openai-responses"] {
            let provider = ScriptedProvider::new(
                provider_name,
                StructuredOutputSupport::Native,
                vec![success(text_block(r#"{"name": "Ada", "nickname": null}"#))],
            );

            let output = run_structured(
                &provider,
                request_with_optional_property_format(),
                StructuredConfig::default(),
            )
            .await?;

            assert_eq!(output.value["nickname"], serde_json::Value::Null);
            assert_eq!(provider.call_count(), 1);
        }
        Ok(())
    }

    #[tokio::test]
    async fn native_non_openai_uses_the_caller_schema() -> Result<()> {
        let provider = ScriptedProvider::new(
            "gemini",
            StructuredOutputSupport::Native,
            vec![success(text_block(r#"{"name": "Ada"}"#))],
        );

        let output = run_structured(
            &provider,
            request_with_optional_property_format(),
            StructuredConfig::default(),
        )
        .await?;

        assert_eq!(output.value["name"], "Ada");
        assert!(output.value.get("nickname").is_none());
        Ok(())
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn native_openai_rejects_dynamic_property_schemas_before_calling_provider() {
        let provider = ScriptedProvider::new("openai", StructuredOutputSupport::Native, Vec::new());
        let mut request = request_with_format();
        request.response_format = Some(ResponseFormat::new(
            "dynamic-properties",
            serde_json::json!({
                "type": "object",
                "properties": {"fixed": {"type": "string"}},
                "additionalProperties": {"type": "string"}
            }),
        ));

        let result = run_structured(&provider, request, StructuredConfig::default()).await;

        assert!(matches!(
            result,
            Err(StructuredOutputError::IncompatibleSchema(_))
        ));
        assert_eq!(provider.call_count(), 0);
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

    #[tokio::test]
    async fn tool_forcing_retry_appends_tool_result_for_forced_tool_use() -> Result<()> {
        use agent_sdk_foundation::llm::Content;

        let provider = ScriptedProvider::new(
            "anthropic",
            StructuredOutputSupport::ToolForcing,
            vec![
                // First respond: invalid (missing required `age`).
                success(respond_tool_block(serde_json::json!({"name": "x"}))),
                // Retry: valid.
                success(respond_tool_block(
                    serde_json::json!({"name": "x", "age": 1}),
                )),
            ],
        );

        let out = run_structured(
            &provider,
            request_with_format(),
            StructuredConfig { max_retries: 1 },
        )
        .await?;
        assert_eq!(out.retries, 1);

        // The retry request must be a valid Anthropic conversation: the appended
        // assistant `respond` tool_use must be answered by a user tool_result with
        // a matching tool_use_id — not a bare user text message (which 400s).
        let seen = provider.seen_requests.lock().expect("seen lock");
        let retry = &seen[1];

        let assistant_tool_use_id = retry
            .messages
            .iter()
            .find_map(|m| match &m.content {
                Content::Blocks(blocks) => blocks.iter().find_map(|b| match b {
                    ContentBlock::ToolUse { id, name, .. } if name == RESPOND_TOOL_NAME => {
                        Some(id.clone())
                    }
                    _ => None,
                }),
                Content::Text(_) => None,
            })
            .expect("assistant respond tool_use present in retry");

        let has_matching_result = retry.messages.iter().any(|m| match &m.content {
            Content::Blocks(blocks) => blocks.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::ToolResult { tool_use_id, .. }
                        if *tool_use_id == assistant_tool_use_id
                )
            }),
            Content::Text(_) => false,
        });
        drop(seen);
        assert!(
            has_matching_result,
            "retry must carry a tool_result for the forced tool_use id"
        );
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
            vec![ChatOutcome::RateLimited(None)],
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

    // ── Streaming structured output ───────────────────────────────────

    /// A provider that serves a fixed list of streaming deltas from
    /// `chat_stream`. The non-streaming `chat` is only exercised on retries
    /// (none of the streaming tests below need it).
    struct StreamingProvider {
        provider_name: &'static str,
        model: String,
        support: StructuredOutputSupport,
        deltas: Mutex<Vec<StreamDelta>>,
    }

    impl StreamingProvider {
        fn new(
            provider_name: &'static str,
            support: StructuredOutputSupport,
            deltas: Vec<StreamDelta>,
        ) -> Self {
            Self {
                provider_name,
                model: "scripted-model".to_owned(),
                support,
                deltas: Mutex::new(deltas),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for StreamingProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(ChatOutcome::ServerError("chat() not used".to_owned()))
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            let deltas = self.deltas.lock().map(|d| d.clone()).unwrap_or_default();
            Box::pin(async_stream::stream! {
                for delta in deltas {
                    yield Ok(delta);
                }
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

    async fn drive_stream(
        mut stream: StructuredStream<'_>,
    ) -> Result<(Vec<serde_json::Value>, Option<StructuredOutput>)> {
        let mut partials = Vec::new();
        let mut final_out = None;
        while let Some(update) = stream.next().await {
            match update? {
                StructuredStreamUpdate::Partial(value) => partials.push(value),
                StructuredStreamUpdate::Final(out) => final_out = Some(out),
            }
        }
        Ok((partials, final_out))
    }

    #[tokio::test]
    async fn streaming_native_emits_partials_then_validated_final() -> Result<()> {
        let provider = StreamingProvider::new(
            "openai",
            StructuredOutputSupport::Native,
            vec![
                StreamDelta::TextDelta {
                    delta: r#"{"name": "Ada""#.to_owned(),
                    block_index: 0,
                },
                StreamDelta::TextDelta {
                    delta: r#", "age": 36}"#.to_owned(),
                    block_index: 0,
                },
                StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: None,
                },
            ],
        );

        let stream = run_structured_stream(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        );
        let (partials, final_out) = drive_stream(stream).await?;

        assert!(!partials.is_empty(), "expected at least one partial");
        // The first partial sees only the name before the age streamed in.
        assert_eq!(partials[0]["name"], "Ada");
        let final_out = final_out.expect("a validated final value");
        assert_eq!(final_out.value["name"], "Ada");
        assert_eq!(final_out.value["age"], 36);
        assert_eq!(final_out.retries, 0);
        Ok(())
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn streaming_native_openai_uses_the_normalized_strict_schema() -> Result<()> {
        let provider = StreamingProvider::new(
            "openai-responses",
            StructuredOutputSupport::Native,
            vec![
                StreamDelta::TextDelta {
                    delta: r#"{"name": "Ada", "nickname": null}"#.to_owned(),
                    block_index: 0,
                },
                StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                    served_route: None,
                },
            ],
        );

        let stream = run_structured_stream(
            &provider,
            request_with_optional_property_format(),
            StructuredConfig::default(),
        );
        let (_, final_output) = drive_stream(stream).await?;
        let final_output = final_output.ok_or_else(|| anyhow::anyhow!("missing final output"))?;

        assert_eq!(final_output.value["nickname"], serde_json::Value::Null);
        Ok(())
    }

    #[tokio::test]
    async fn streaming_tool_forcing_reads_partial_tool_input() -> Result<()> {
        let provider = StreamingProvider::new(
            "anthropic",
            StructuredOutputSupport::ToolForcing,
            vec![
                StreamDelta::ToolUseStart {
                    id: "call_1".to_owned(),
                    name: RESPOND_TOOL_NAME.to_owned(),
                    block_index: 0,
                    thought_signature: None,
                },
                StreamDelta::ToolInputDelta {
                    id: "call_1".to_owned(),
                    delta: r#"{"name": "Linus""#.to_owned(),
                    block_index: 0,
                },
                StreamDelta::ToolInputDelta {
                    id: "call_1".to_owned(),
                    delta: r#", "age": 54}"#.to_owned(),
                    block_index: 0,
                },
                StreamDelta::Done {
                    stop_reason: Some(StopReason::ToolUse),
                    served_route: None,
                },
            ],
        );

        let stream = run_structured_stream(
            &provider,
            request_with_format(),
            StructuredConfig::default(),
        );
        let (partials, final_out) = drive_stream(stream).await?;

        assert_eq!(partials[0]["name"], "Linus");
        let final_out = final_out.expect("a validated final value");
        assert_eq!(final_out.value["age"], 54);
        Ok(())
    }

    #[tokio::test]
    async fn streaming_missing_response_format_errors() {
        let provider =
            StreamingProvider::new("openai", StructuredOutputSupport::Native, Vec::new());
        let mut req = request_with_format();
        req.response_format = None;

        let mut stream = run_structured_stream(&provider, req, StructuredConfig::default());
        let first = stream.next().await.expect("one item");
        assert!(matches!(
            first,
            Err(StructuredOutputError::MissingResponseFormat)
        ));
    }

    #[test]
    fn partial_from_buffer_repairs_incomplete_json() {
        assert_eq!(
            partial_from_buffer(r#"{"name": "Ada""#).map(|v| v["name"].clone()),
            Some(serde_json::json!("Ada"))
        );
        assert_eq!(
            partial_from_buffer(r#"{"a": 1,"#),
            Some(serde_json::json!({"a": 1}))
        );
        assert_eq!(
            partial_from_buffer(r#"{"a":"#),
            Some(serde_json::json!({"a": null}))
        );
        assert!(partial_from_buffer("").is_none());
        assert!(partial_from_buffer("not json").is_none());
    }

    #[test]
    fn apply_tool_forcing_forces_the_respond_tool() {
        // `apply_tool_forcing` injects the `respond` tool and forces it via
        // `tool_choice`. It intentionally leaves `request.thinking` untouched:
        // clearing it here would be resurrected by `resolve_thinking_config`'s
        // fallback to the provider-configured default, so the actual
        // thinking-incompatible-with-forced-tool guard lives at the Claude wire
        // boundary (see `provider::thinking_for_forced_tool` and the
        // `forced_tool_drops_configured_thinking_on_the_wire` regression in the
        // Anthropic provider). This test locks in the forcing contract.
        let mut request = request_with_format();
        request.thinking = Some(agent_sdk_foundation::llm::ThinkingConfig::new(10_000));
        let response_format = request
            .response_format
            .clone()
            .expect("request_with_format sets a response format");

        apply_tool_forcing(&mut request, &response_format);

        assert!(matches!(
            request.tool_choice,
            Some(ToolChoice::Tool(ref name)) if name == RESPOND_TOOL_NAME
        ));
        assert!(
            request
                .tools
                .as_ref()
                .is_some_and(|tools| tools.iter().any(|t| t.name == RESPOND_TOOL_NAME)),
            "the respond tool must be present"
        );
        // Thinking is deliberately not mutated here; the wire-boundary guard
        // drops it. See the module-level comment in `apply_tool_forcing`.
        assert!(request.thinking.is_some());
    }
}
